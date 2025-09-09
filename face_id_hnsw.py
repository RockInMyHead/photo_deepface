"""
Face ID pipeline based on InsightFace (buffalo_l by default) and HNSWLib.

This script builds a gallery of reference embeddings (one directory per identity),
creates a HNSW index for approximate nearest neighbour search on cosine similarity,
and performs single-image or batch identification. It supports trim-mean
aggregation, optional TTA (horizontal flip), basic anti‑spoofing checks, and
configurable thresholds.  Use `--identify_dir` to perform identification on
all images (recursively) in a directory.  Results can be exported in JSONL
and/or CSV format.

Dependencies: insightface, onnxruntime (or onnxruntime-gpu), opencv-python,
numpy, hnswlib.
"""

import os
import glob
import json
import csv
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Configuration dataclass
# ----------------------------------------------------------------------------

@dataclass
class Config:
    """Configuration parameters for face identification."""
    model_name: str = "buffalo_l"    # Recognition model (e.g. buffalo_l, adaface_ir50)
    det_thresh: float = 0.4           # Minimum face detection confidence
    embed_norm: bool = True           # L2-normalise embeddings
    flip_tta: bool = True             # Test-time augmentation via horizontal flip
    gallery_trim: float = 0.1         # Fraction of extreme embeddings trimmed in robust mean
    hnsw_space: str = "cosine"       # Metric space for HNSW (cosine → 1 - cosine similarity)
    hnsw_M: int = 48                  # HNSW parameter: number of neighbours in graph
    hnsw_ef_construction: int = 200   # HNSW parameter: construction complexity
    hnsw_ef_search: int = 64          # HNSW parameter: search complexity (trade-off speed/accuracy)
    top_k: int = 5                    # Number of candidates to return
    default_threshold: float = 0.25   # Default similarity threshold (cosine) for match

# Instantiate default configuration
CFG = Config()

# ----------------------------------------------------------------------------
# InsightFace loader
# ----------------------------------------------------------------------------

def load_insightface(model_name: str):
    """Load InsightFace FaceAnalysis for face detection and recognition."""
    from insightface.app import FaceAnalysis
    # Use both GPU and CPU providers if available. On CPU‑only machines,
    # onnxruntime will gracefully fall back.
    app = FaceAnalysis(name=model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # Prepare with default input size for detection
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------

def read_image(path: str) -> np.ndarray:
    """Read an image from disk (supports unicode paths)."""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img

def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    """Normalise vectors along `axis` with L2 norm."""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, eps, None)

def trim_mean_stack(embeds: np.ndarray, trim: float = 0.1) -> np.ndarray:
    """
    Compute a trimmed mean across the first dimension of `embeds`.
    Trims `trim` fraction from both ends per dimension before averaging.
    """
    if embeds.shape[0] == 1:
        return embeds[0]
    k = max(1, int(embeds.shape[0] * trim))
    sorted_idx = np.argsort(embeds, axis=0)
    cols = []
    for j in range(embeds.shape[1]):
        idx = sorted_idx[:, j][k:embeds.shape[0] - k] if embeds.shape[0] > 2 * k else sorted_idx[:, j]
        cols.append(embeds[idx, j])
    return np.mean(np.stack(cols, axis=1), axis=0)

def basic_anti_spoof_guard(face_img: np.ndarray) -> bool:
    """
    Basic heuristic against presentation attacks: rejects too small or heavily blurred faces.
    Replace with a PAD (presentation attack detection) model for production use.
    """
    h, w = face_img.shape[:2]
    if min(h, w) < 64:
        return False
    lap = cv2.Laplacian(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    return lap > 15.0

# ----------------------------------------------------------------------------
# Face detection and embeddings
# ----------------------------------------------------------------------------

def extract_faces_and_embeddings(app, img_bgr: np.ndarray) -> List[Dict]:
    """
    Detect faces in `img_bgr` and return bounding boxes, detection scores,
    aligned embeddings and cropped face images. Applies optional flip TTA
    and normalisation.
    """
    faces = app.get(img_bgr, max_num=16)
    out = []
    for f in faces:
        if f.det_score < CFG.det_thresh:
            continue
        emb = f.embedding.astype(np.float32)
        if CFG.flip_tta:
            try:
                emb2 = f.normed_embedding if hasattr(f, "normed_embedding") else None
            except Exception:
                emb2 = None
            if emb2 is not None and emb2.shape == emb.shape:
                emb = (emb + emb2) / 2.0
        if CFG.embed_norm:
            emb = emb / (np.linalg.norm(emb) + 1e-12)
        face_img = getattr(f, "crop_img", None)
        if face_img is None:
            x1, y1, x2, y2 = map(int, f.bbox)
            face_img = img_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        out.append({
            "bbox": f.bbox.tolist(),
            "det_score": float(f.det_score),
            "embedding": emb.astype(np.float32),
            "face_img": face_img
        })
    return out

# ----------------------------------------------------------------------------
# Gallery and HNSW index
# ----------------------------------------------------------------------------

def build_gallery(app, gallery_root: str) -> Tuple[np.ndarray, List[str]]:
    """
    Build gallery embeddings. Each subdirectory under `gallery_root` is treated as
    one identity; all images are processed and aggregated.
    Returns an array of shape (num_ids, embedding_dim) and a list of IDs.
    """
    ids, embs = [], []
    for person_dir in sorted([p for p in Path(gallery_root).glob("*") if p.is_dir()]):
        pid = person_dir.name
        arr = []
        for img_path in glob.glob(str(person_dir / "*")):
            try:
                img = read_image(img_path)
                faces = extract_faces_and_embeddings(app, img)
                if not faces:
                    continue
                faces.sort(key=lambda x: x["det_score"], reverse=True)
                face = faces[0]
                if not basic_anti_spoof_guard(face["face_img"]):
                    continue
                arr.append(face["embedding"])
            except Exception:
                continue
        if not arr:
            continue
        arr = np.stack(arr, axis=0)
        centroid = trim_mean_stack(arr, trim=CFG.gallery_trim)
        if CFG.embed_norm:
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        ids.append(pid)
        embs.append(centroid.astype(np.float32))
    if not embs:
        raise RuntimeError("Gallery is empty; no valid embeddings found.")
    return np.stack(embs, axis=0).astype(np.float32), ids

# HNSW components
import hnswlib  # deferred import is not necessary here

def build_index_hnsw(embs: np.ndarray):
    """Build an HNSW index from gallery embeddings."""
    d = embs.shape[1]
    index = hnswlib.Index(space=CFG.hnsw_space, dim=d)
    index.init_index(max_elements=embs.shape[0], ef_construction=CFG.hnsw_ef_construction, M=CFG.hnsw_M)
    index.add_items(embs, np.arange(embs.shape[0]))
    index.set_ef(CFG.hnsw_ef_search)
    return index

def hnsw_search(index, q: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Query HNSW index with `q` (shape [n, d]) and return similarities and labels.
    For cosine space, returned distances = 1 - cosine_similarity.
    """
    labels, distances = index.knn_query(q, k=top_k)
    sims = 1.0 - distances  # convert distance to cosine similarity
    return sims, labels

# ----------------------------------------------------------------------------
# Identification
# ----------------------------------------------------------------------------

def identify(app, index, id_map: List[str], img_path: str, threshold: float = None, top_k: int = None) -> List[Dict]:
    """
    Identify faces in a single image and return a list of matches. Each result
    contains bounding box, best match ID, score, match flag, and candidate list.
    """
    if threshold is None:
        threshold = CFG.default_threshold
    if top_k is None:
        top_k = CFG.top_k
    img = read_image(img_path)
    faces = extract_faces_and_embeddings(app, img)
    if not faces:
        return []
    q = np.stack([f["embedding"] for f in faces], axis=0).astype(np.float32)
    sims, idx = hnsw_search(index, q, top_k)
    results = []
    for i, face in enumerate(faces):
        candidates = []
        for j in range(top_k):
            label = id_map[idx[i, j]]
            score = float(sims[i, j])
            candidates.append({"id": label, "score": score})
        best = candidates[0]
        matched = (best["score"] >= threshold) and basic_anti_spoof_guard(face["face_img"])
        results.append({
            "bbox": face["bbox"],
            "det_score": face["det_score"],
            "best_id": best["id"],
            "best_score": best["score"],
            "matched": matched,
            "candidates": candidates
        })
    return results

# ----------------------------------------------------------------------------
# Batch identification and result writing
# ----------------------------------------------------------------------------

def write_results(results: List[Dict], out_jsonl: str = None, out_csv: str = None) -> None:
    """
    Write batch identification results to JSONL and/or CSV. Each entry must have at least
    keys: file, best_id, best_score, matched, det_score, num_faces, candidates.
    """
    if out_jsonl:
        with open(out_jsonl, "w", encoding="utf-8") as jf:
            for r in results:
                jf.write(json.dumps(r, ensure_ascii=False) + "\n")
    if out_csv:
        fields = ["file", "best_id", "best_score", "matched", "det_score", "num_faces", "candidates"]
        with open(out_csv, "w", newline="", encoding="utf-8") as cf:
            w = csv.DictWriter(cf, fieldnames=fields)
            w.writeheader()
            for r in results:
                w.writerow({
                    "file": r["file"],
                    "best_id": r["best_id"],
                    "best_score": f"{r['best_score']:.6f}" if r.get("best_score") is not None else "",
                    "matched": r["matched"],
                    "det_score": f"{r['det_score']:.6f}" if r.get("det_score") is not None else "",
                    "num_faces": r["num_faces"],
                    "candidates": json.dumps(r["candidates"], ensure_ascii=False)
                })

def identify_dir(app, index, id_map: List[str], root: str, threshold: float, top_k: int,
                 exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> List[Dict]:
    """
    Recursively identify all images in `root` (matching extensions `exts`).
    Returns a list of result dicts summarising each file's best match.
    """
    files = []
    for ext in exts:
        files.extend(Path(root).rglob(f"*{ext}"))
    files = sorted(set(map(str, files)))
    results = []
    for pth in files:
        try:
            res = identify(app, index, id_map, pth, threshold=threshold, top_k=top_k)
            if res:
                res.sort(key=lambda x: x["best_score"], reverse=True)
                top = res[0]
                results.append({
                    "file": pth,
                    "best_id": top["best_id"],
                    "best_score": top["best_score"],
                    "matched": top["matched"],
                    "det_score": top["det_score"],
                    "num_faces": len(res),
                    "candidates": top["candidates"]
                })
            else:
                results.append({
                    "file": pth,
                    "best_id": None,
                    "best_score": None,
                    "matched": False,
                    "det_score": None,
                    "num_faces": 0,
                    "candidates": []
                })
        except Exception as e:
            results.append({
                "file": pth,
                "best_id": None,
                "best_score": None,
                "matched": False,
                "det_score": None,
                "num_faces": -1,
                "candidates": [{"error": str(e)}]
            })
    return results

# ----------------------------------------------------------------------------
# Main CLI entry point
# ----------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser("Face ID (InsightFace + HNSWLib)")
    p.add_argument("--gallery", required=True, help="Directory with enrolment images (subdirs=IDs)")
    p.add_argument("--query", required=False, help="Single image to identify")
    # New flag: directory for batch identification
    p.add_argument("--identify_dir", required=False, help="Directory containing images to identify (recursively)")
    # Backwards compatibility: still accept --queries_dir silently
    p.add_argument("--queries_dir", required=False, help=argparse.SUPPRESS)
    p.add_argument("--model", default=CFG.model_name, help="InsightFace model name (default: buffalo_l)")
    p.add_argument("--threshold", type=float, default=CFG.default_threshold, help="Similarity threshold for match")
    p.add_argument("--top_k", type=int, default=CFG.top_k, help="Number of top candidates to return")
    p.add_argument("--save_index", default="index_hnsw.bin", help="File to save HNSW index")
    p.add_argument("--save_meta", default="meta.json", help="File to save metadata (ID map, threshold)")
    p.add_argument("--out_jsonl", default=None, help="Output JSONL file for batch results")
    p.add_argument("--out_csv", default=None, help="Output CSV file for batch results")
    p.add_argument("--exts", default=".jpg,.jpeg,.png,.bmp,.webp", help="Comma-separated list of extensions to scan in identify_dir")
    args = p.parse_args()

    # Update global config
    global CFG
    CFG.model_name = args.model
    CFG.top_k = args.top_k

    print(f"[Init] loading InsightFace: {CFG.model_name}")
    app = load_insightface(CFG.model_name)

    print("[Build] Building gallery...")
    gallery_embs, ids = build_gallery(app, args.gallery)
    print(f"[Build] Persons: {len(ids)}, embedding dim: {gallery_embs.shape[1]}")

    print("[Index] Building HNSW index...")
    index = build_index_hnsw(gallery_embs)

    # Persist index and metadata for reuse
    index.save_index(args.save_index)
    with open(args.save_meta, "w", encoding="utf-8") as f:
        json.dump({"ids": ids, "threshold": args.threshold, "cfg": asdict(CFG)}, f, ensure_ascii=False, indent=2)

    # Single image identification
    if args.query:
        print(f"[Query] {args.query}")
        res = identify(app, index, ids, args.query, threshold=args.threshold, top_k=args.top_k)
        print(json.dumps(res, ensure_ascii=False, indent=2))

    # Batch identification over a directory
    identify_root = args.identify_dir or args.queries_dir
    if identify_root:
        exts = tuple([e.strip() if e.strip().startswith('.') else '.' + e.strip() for e in args.exts.split(',') if e.strip()])
        print(f"[Batch] identify_dir={identify_root} (exts={exts})")
        batch = identify_dir(app, index, ids, identify_root, threshold=args.threshold, top_k=args.top_k, exts=exts)
        # Default filenames if not provided
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_jsonl = args.out_jsonl or f"identify_{ts}.jsonl"
        out_csv = args.out_csv or f"identify_{ts}.csv"
        write_results(batch, out_jsonl=out_jsonl, out_csv=out_csv)
        print(f"[Batch] Processed {len(batch)} files → {out_jsonl}, {out_csv}")

if __name__ == "__main__":
    main()