# core/cluster.py
"""
Core clustering planner for a *single group* directory.

Implements rules:
- solo = 1 face; joint = 2..group_thr faces; group = group_thr+1.. faces
- Eligible cluster: appears on at least one non-group image (faces_in_image <= group_thr).
- Plan per image -> clusters; image goes to each eligible cluster that appears on it.
- If image has any ineligible clusters -> stage image to __group_only__ (dedup per image).
- If image has no clustered faces (only noise) -> stage image to __unknown__ (dedup per image).
- NO filesystem writes here. The app layer applies the plan, handles numbering and renames.

The plan returns cluster centroids (normalized) for identity matching across runs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Set, Tuple
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import insightface
from insightface.app import FaceAnalysis

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

@dataclass
class FaceRec:
    img_path: Path
    face_index: int
    faces_in_image: int
    det_score: float
    embedding: np.ndarray

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def load_bgr(path: Path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.array(im)[:, :, ::-1]
    return arr

def laplacian_variance(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

class Embedder:
    def __init__(self, det_size=640, ctx_id=0):
        self.app = FaceAnalysis(allowed_modules=['detection','recognition'])
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    def detect(self, bgr):
        return self.app.get(bgr)

    def embed_face(self, face):
        return np.array(face.embedding, dtype=np.float32)

def collect_faces(group_dir: Path, min_face=110, blur_thr=45.0, det_size=640, gpu_id=0) -> List[FaceRec]:
    emb = Embedder(det_size=det_size, ctx_id=gpu_id)
    records: List[FaceRec] = []
    files = [p for p in group_dir.rglob("*") if p.is_file() and is_image(p)]
    for img_path in tqdm(files, desc=f"Scanning {group_dir.name}"):
        bgr = load_bgr(img_path)
        h, w = bgr.shape[:2]
        if min(h, w) < min_face:
            s = min(h, w)
            y0 = (h - s)//2; x0 = (w - s)//2
            bgr = bgr[y0:y0+s, x0:x0+s]
            if s < min_face:
                bgr = cv2.resize(bgr, (min_face, min_face), interpolation=cv2.INTER_CUBIC)
        if laplacian_variance(bgr) < blur_thr:
            continue
        faces = emb.detect(bgr)
        n_faces = len(faces)
        for idx, f in enumerate(faces):
            records.append(FaceRec(
                img_path=img_path,
                face_index=idx,
                faces_in_image=n_faces,
                det_score=float(f.det_score),
                embedding=emb.embed_face(f)
            ))
    return records

def _norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x) + 1e-12
    return x / n

def cluster(records: List[FaceRec], eps_sim=0.55, min_samples=2):
    if not records:
        return np.array([]), {}
    X = np.stack([r.embedding for r in records]).astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    eps = max(1e-6, 1.0 - float(eps_sim))  # cosine distance
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1)
    labels = db.fit_predict(X)  # -1 = noise

    # Compute centroids per cluster (normalized)
    centroids = {}
    for lab in set(labels):
        if lab == -1:
            continue
        idxs = np.where(labels == lab)[0]
        c = _norm(X[idxs].mean(axis=0).astype(np.float32))
        centroids[int(lab)] = c.tolist()
    return labels, centroids

def build_plan(
    group_dir: Path,
    group_thr: int = 3,
    eps_sim: float = 0.55,
    min_samples: int = 2,
    min_face: int = 110,
    blur_thr: float = 45.0,
    det_size: int = 640,
    gpu_id: int = 0
) -> Dict:
    """Return routing plan for the given group directory."""
    recs = collect_faces(group_dir, min_face=min_face, blur_thr=blur_thr, det_size=det_size, gpu_id=gpu_id)
    if not recs:
        return {
            "eligible_clusters": [],
            "cluster_images": {},
            "cluster_centroids": {},
            "group_only_images": [],
            "unknown_images": [],
            "stats": {"images_total": 0, "images_unknown_only": 0, "images_group_only": 0, "faces_total": 0},
        }
    labels, centroids = cluster(recs, eps_sim=eps_sim, min_samples=min_samples)

    image_idxs: Dict[Path, List[int]] = {}
    for i, r in enumerate(recs):
        image_idxs.setdefault(r.img_path, []).append(i)

    cluster_idxs: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        cluster_idxs.setdefault(int(lab), []).append(i)

    eligible_clusters: Set[int] = set()
    for lab, idxs in cluster_idxs.items():
        if lab == -1:
            continue
        if any(recs[i].faces_in_image <= group_thr for i in idxs):
            eligible_clusters.add(lab)

    cluster_images: Dict[int, Set[Path]] = {lab: set() for lab in eligible_clusters}
    group_only_images: Set[Path] = set()
    unknown_images: Set[Path] = set()

    for img, idxs in image_idxs.items():
        labs_on_image = {int(labels[i]) for i in idxs}
        labs_no_noise = {lab for lab in labs_on_image if lab != -1}
        has_any_cluster = len(labs_no_noise) > 0

        for lab in labs_no_noise:
            if lab in eligible_clusters:
                cluster_images[lab].add(img)

        ineligible_on_image = any((lab not in eligible_clusters) and (lab != -1) for lab in labs_on_image)
        if ineligible_on_image:
            group_only_images.add(img)

        if not has_any_cluster:
            unknown_images.add(img)

    images_total = len(image_idxs)
    stats = {
        "images_total": images_total,
        "images_unknown_only": len(unknown_images),
        "images_group_only": len(group_only_images),
        "faces_total": len(recs),
        "eligible_clusters": len(eligible_clusters),
    }
    return {
        "eligible_clusters": sorted(list(eligible_clusters)),
        "cluster_images": {int(k): sorted([str(p) for p in v]) for k, v in cluster_images.items()},
        "cluster_centroids": {int(k): v for k, v in centroids.items()},
        "group_only_images": sorted([str(p) for p in group_only_images]),
        "unknown_images": sorted([str(p) for p in unknown_images]),
        "stats": stats,
    }
