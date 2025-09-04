"""
core/cluster.py — InsightFace migration

Builds clustering plan for faces using state-of-the-art embeddings via InsightFace.

Outputs are tailored to the existing Streamlit app expectations:
- eligible_clusters: list[int] — cluster ids that pass size threshold (group_thr)
- cluster_centroids: dict[int, list[float]] — L2-normalized centroid embeddings
- cluster_images: dict[int, list[str]] — file paths per cluster membership (per-face)
- group_only_images: list[str] — images with multiple faces but no eligible cluster hits
- unknown_images: list[str] — images with 0 accepted faces (after filtering)
- stats: {images_total, images_unknown_only, images_group_only}

Config knobs (forwarded from app):
- group_thr (int)
- eps_sim (float, cosine similarity threshold; distance eps = 1 - eps_sim)
- min_samples (int), DBSCAN
- min_face (int, px on shorter bbox side)
- blur_thr (float, variance of Laplacian threshold; higher = sharper)
- det_size (int), detector inference size
- gpu_id (int), InsightFace ctx id (-1 for CPU)
- min_det_score (float) — filter faces below detector confidence
- min_quality (float) — reserved, currently unused beyond blur/min_face checks

Dependencies (install once):
    pip install -U insightface onnxruntime-gpu opencv-python scikit-learn numpy
    # CPU fallback: replace onnxruntime-gpu -> onnxruntime
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np

# Public constant used by app.py
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# ---------------------------
# Lazy engine (InsightFace)
# ---------------------------
_ENGINE_CACHE: Dict[Tuple[int, int], object] = {}


def _lazy_imports():
    global cv2, FaceAnalysis, face_align
    import importlib
    cv2 = importlib.import_module("cv2")
    insightface = importlib.import_module("insightface")
    FaceAnalysis = getattr(insightface.app, "FaceAnalysis")
    face_align = importlib.import_module("insightface.utils.face_align")


def _get_engine(gpu_id: int, det_size: int):
    key = (int(gpu_id), int(det_size))
    eng = _ENGINE_CACHE.get(key)
    if eng is not None:
        return eng
    try:
        _lazy_imports()
    except Exception as e:
        raise RuntimeError(
            "InsightFace is not installed. Please `pip install -U insightface onnxruntime-gpu opencv-python`."
        ) from e

    app = FaceAnalysis(name="buffalo_l")  # robust det + 512-d embeddings
    # ctx_id: -1 CPU, >=0 GPU index
    app.prepare(ctx_id=int(gpu_id), det_size=(int(det_size), int(det_size)))
    _ENGINE_CACHE[key] = app
    return app


# ---------------------------
# Image / quality utilities
# ---------------------------

def _imread_rgb(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _bbox_short_side(bbox: np.ndarray) -> int:
    # bbox: [x1, y1, x2, y2]
    w = float(bbox[2] - bbox[0])
    h = float(bbox[3] - bbox[1])
    return int(min(w, h))


def _l2norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v) + 1e-12)
    return v / n


# ---------------------------
# Core API
# ---------------------------

def build_plan(
    group_dir: Path | str,
    *,
    group_thr: int = 3,
    eps_sim: float = 0.55,
    min_samples: int = 2,
    min_face: int = 110,
    blur_thr: float = 45.0,
    det_size: int = 640,
    gpu_id: int = 0,
    min_det_score: float = 0.50,
    min_quality: float = 0.50,
) -> Dict:
    """Scan images in `group_dir` (non-recursive), detect faces, embed with InsightFace,
    cluster with DBSCAN (cosine), and return a plan dict expected by the app.
    """
    group_dir = Path(group_dir)

    # Collect only top-level image files (do not recurse to avoid re-processing already grouped ones)
    files = [p for p in sorted(group_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]

    stats = {
        "images_total": len(files),
        "images_unknown_only": 0,
        "images_group_only": 0,
    }

    if not files:
        return {
            "eligible_clusters": [],
            "cluster_centroids": {},
            "cluster_images": {},
            "group_only_images": [],
            "unknown_images": [],
            "stats": stats,
        }

    app = _get_engine(gpu_id=gpu_id, det_size=det_size)

    # Storage for accepted faces
    face_embeddings: List[np.ndarray] = []
    face_img_indices: List[int] = []  # index into files

    # Per-image bookkeeping
    per_image_detected: List[int] = [0] * len(files)  # raw detections
    per_image_accepted: List[int] = [0] * len(files)  # passed filters

    # Pre-read and infer
    for i, path in enumerate(files):
        img_rgb = _imread_rgb(path)
        if img_rgb is None:
            continue
        faces = app.get(img_rgb)
        per_image_detected[i] = len(faces)
        if not faces:
            continue

        # Process faces
        for f in faces:
            # Confidence filter
            score = float(getattr(f, "det_score", 0.0))
            if score < float(min_det_score):
                continue
            # Size filter
            bbox = np.asarray(getattr(f, "bbox", None) or [0, 0, 0, 0], dtype=np.float32)
            if _bbox_short_side(bbox) < int(min_face):
                continue
            # Align crop (112x112) for robust embedding & blur estimation
            kps = np.asarray(getattr(f, "kps", None))
            if kps is None or kps.shape != (5, 2):
                # Without landmarks we skip; embeddings would be less stable
                continue
            crop = face_align.norm_crop(img_rgb, landmark=kps, image_size=112)
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            if _laplacian_var(gray) < float(blur_thr):
                continue

            # Embedding (InsightFace already returns L2-normalized embedding)
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                emb = getattr(f, "embedding", None)
            if emb is None:
                # As a fallback, get embedding via internal model if available
                # (FaceAnalysis usually populates normed_embedding)
                continue
            emb = _l2norm(np.asarray(emb, dtype=np.float32))

            face_embeddings.append(emb)
            face_img_indices.append(i)
            per_image_accepted[i] += 1

    # Unknowns: images with 0 accepted faces
    unknown_mask = [cnt == 0 for cnt in per_image_accepted]
    unknown_images = [str(files[i]) for i, m in enumerate(unknown_mask) if m]

    if not face_embeddings:
        stats["images_unknown_only"] = len(unknown_images)
        stats["images_group_only"] = 0
        return {
            "eligible_clusters": [],
            "cluster_centroids": {},
            "cluster_images": {},
            "group_only_images": [],
            "unknown_images": unknown_images,
            "stats": stats,
        }

    X = np.stack(face_embeddings, axis=0).astype(np.float32)

    # --- Clustering ---
    # Convert similarity threshold to cosine distance eps
    from sklearn.cluster import DBSCAN

    eps_dist = max(0.0, 1.0 - float(eps_sim))  # cosine distance in [0,2]
    min_samples = int(min_samples)

    db = DBSCAN(eps=eps_dist, min_samples=min_samples, metric="cosine", n_jobs=-1)
    labels = db.fit_predict(X)

    # Group faces by cluster label (ignore noise -1)
    clusters: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        if lab == -1:
            continue
        clusters.setdefault(int(lab), []).append(idx)

    # Compute centroids and images per cluster
    cluster_centroids: Dict[int, List[float]] = {}
    cluster_images: Dict[int, List[str]] = {}

    for cid, idxs in clusters.items():
        vecs = _l2norm(np.mean(X[idxs], axis=0))
        cluster_centroids[cid] = vecs.astype(np.float32).tolist()
        imgs = [str(files[face_img_indices[j]]) for j in idxs]
        cluster_images[cid] = imgs

    # Eligible clusters (size >= group_thr)
    eligible_clusters = [cid for cid, idxs in clusters.items() if len(idxs) >= int(group_thr)]

    # Group-only images: multi-face photos with no faces in eligible clusters
    eligible_face_img_idxs = set()
    for cid in eligible_clusters:
        for j in clusters[cid]:
            eligible_face_img_idxs.add(face_img_indices[j])

    group_only_images = []
    for i, fpath in enumerate(files):
        if per_image_detected[i] >= 2 and per_image_accepted[i] >= 2:
            # Does this image contribute to any eligible cluster?
            if i not in eligible_face_img_idxs:
                group_only_images.append(str(fpath))

    stats["images_unknown_only"] = len(unknown_images)
    stats["images_group_only"] = len(group_only_images)

    return {
        "eligible_clusters": eligible_clusters,
        "cluster_centroids": cluster_centroids,
        "cluster_images": cluster_images,
        "group_only_images": group_only_images,
        "unknown_images": unknown_images,
        "stats": stats,
    }
