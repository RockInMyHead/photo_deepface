import json
import shutil
import socket
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple, Set
from datetime import datetime

import streamlit as st

# Optional: Drag&Drop lists
try:
    from streamlit_sortables import sort_items  # pip install streamlit-sortables
except Exception:
    sort_items = None

# Native folder picker
def pick_folder_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder = filedialog.askdirectory(title="Выберите папку")
        root.destroy()
        return folder
    except Exception:
        return None

# Optional send2trash
try:
    from send2trash import send2trash
except Exception:
    send2trash = None

# Backend (provided by your project)
try:
    from core.cluster import build_plan, IMG_EXTS
except Exception:
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    def build_plan(*args, **kwargs):
        return {
            "eligible_clusters": [],
            "cluster_centroids": {},
            "cluster_images": {},
            "group_only_images": [],
            "unknown_images": [],
            "stats": {
                "images_total": 0,
                "images_unknown_only": 0,
                "images_group_only": 0
            }
        }

# Optional PIL for exact 150x150 thumbnails
try:
    from PIL import Image
except Exception:
    Image = None

# Optional OpenCV and YOLO
try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

try:
    import torch  # noqa: F401  # only to hint that torch might be available
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

st.set_page_config(page_title="Face Sorter — Мини-проводник", layout="wide")

#########################################################################
#                    Configuration and Utility Functions                #
#########################################################################

def _clamp(v, lo, hi, default):
    """
    Safely clamp configuration values to a numeric range. If conversion fails,
    return the provided default.
    """
    try:
        v = type(default)(v)
        return max(lo, min(hi, v))
    except Exception:
        return default

def load_config(base: Path) -> Dict:
    """
    Load configuration from config.json if available. Provides sensible
    defaults for thresholds controlling clustering and matching behaviour.

    Extended options allow enabling YOLO-based face detection. The user can
    specify paths to YOLO configuration and weights files. Without valid
    weights, the application gracefully falls back to the built-in face
    detection.
    """
    p = base / "config.json"
    defaults = {
        # clustering thresholds
        "group_thr": 3,
        "eps_sim": 0.50,         # lowered default for tighter clusters
        "min_samples": 2,
        "min_face": 80,          # accept smaller faces
        "blur_thr": 30.0,        # allow slightly blurrier images
        "det_size": 640,
        "gpu_id": 0,
        # matching thresholds
        "match_thr": 0.45,       # default match threshold
        "top2_margin": 0.08,
        "per_person_min_obs": 5, # fewer images required per person
        "min_det_score": 0.3,
        "min_quality": 0.3,
        # YOLO options
        "use_yolo": False,
        "yolo_cfg": "models/yolov3-face.cfg",
        "yolo_weights": "models/yolov3-wider_16000.weights",
        "yolo_conf_thr": 0.5,
        "yolo_nms_thr": 0.4,
    }
    if p.exists():
        try:
            user_cfg = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(user_cfg, dict):
                defaults.update({k: user_cfg[k] for k in user_cfg})
        except Exception:
            pass

    # clamp numeric ranges
    defaults["eps_sim"] = _clamp(defaults["eps_sim"], 0.0, 1.0, 0.50)
    defaults["match_thr"] = _clamp(defaults["match_thr"], 0.0, 1.0, 0.45)
    defaults["top2_margin"] = _clamp(defaults["top2_margin"], 0.0, 1.0, 0.08)
    defaults["min_face"] = _clamp(defaults["min_face"], 0, 10000, 80)
    defaults["det_size"] = _clamp(defaults["det_size"], 64, 4096, 640)
    defaults["per_person_min_obs"] = max(1, int(defaults.get("per_person_min_obs", 5)))
    return defaults

CFG_BASE = Path(__file__).parent
CFG = load_config(CFG_BASE)

#########################################################################
#                            YOLO Detection                             #
#########################################################################

def load_yolo_model(cfg_path: str, weights_path: str):
    """
    Load YOLO network from configuration and weight files using OpenCV DNN.
    Returns None if OpenCV DNN or the specified files are unavailable.
    """
    if cv2 is None:
        return None
    cfg_file = Path(cfg_path)
    weights_file = Path(weights_path)
    if not cfg_file.exists() or not weights_file.exists():
        # model files not found
        return None
    try:
        net = cv2.dnn.readNetFromDarknet(str(cfg_file), str(weights_file))
        if CFG.get("gpu_id", 0) >= 0 and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            # attempt to use GPU if available
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net
    except Exception:
        return None

def detect_faces_yolo(net, image_path: Path, conf_threshold: float, nms_threshold: float) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in an image using a YOLO network. Returns a list of bounding
    boxes (x1, y1, x2, y2) representing detected faces. If detection fails,
    returns an empty list.
    """
    if net is None or cv2 is None or np is None:
        return []
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (320, 320), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
        outputs = net.forward(output_layers)
        boxes = []
        confidences = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # Many face-specific YOLO models have only one class (face)
                if confidence > conf_threshold:
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
        # Apply Non-Maxima Suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        final_boxes = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (bw, bh) = (boxes[i][2], boxes[i][3])
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w, x + bw)
                y2 = min(h, y + bh)
                final_boxes.append((x1, y1, x2, y2))
        return final_boxes
    except Exception:
        return []

def crop_faces_with_yolo(source_dir: Path, temp_dir: Path, net, conf_thr: float, nms_thr: float):
    """
    Iterate through image files in source_dir and attempt to crop a single face
    from each image. When a YOLO network is provided and properly loaded, it
    performs detection via YOLO to find faces and crops the largest detected
    face. If YOLO is disabled or unavailable, the function falls back to
    OpenCV's built-in Haar cascade to detect frontal faces. If no face is
    detected by either method, the original image is copied unchanged.

    Parameters
    ----------
    source_dir : Path
        Directory containing the original images.
    temp_dir : Path
        Temporary directory where cropped faces or original images are stored.
    net : cv2.dnn.Net or None
        Preloaded YOLO network. If None, Haar cascade is used for detection.
    conf_thr : float
        Confidence threshold for YOLO detections.
    nms_thr : float
        Non‑maximum suppression threshold for YOLO detections.
    """
    ensure_dir(temp_dir)
    # Prepare Haar cascade for fallback detection
    haar_cascade = None
    if cv2 is not None:
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            haar = cv2.CascadeClassifier(cascade_path)
            if not haar.empty():
                haar_cascade = haar
        except Exception:
            pass

    for ext in IMG_EXTS:
        for img_path in source_dir.rglob(f"*{ext}"):
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    # if image cannot be read, copy as-is
                    shutil.copy2(img_path, temp_dir / img_path.name)
                    continue
                h, w = image.shape[:2]
                face_crop_written = False
                # First, attempt YOLO detection if model available
                if net is not None:
                    boxes = detect_faces_yolo(net, img_path, conf_thr, nms_thr)
                    if boxes:
                        # select largest box
                        boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
                        (x1, y1, x2, y2) = boxes[0]
                        face = image[y1:y2, x1:x2]
                        if face.size > 0:
                            cv2.imwrite(str(temp_dir / img_path.name), face)
                            face_crop_written = True
                # If YOLO not used or no detections, try Haar cascade
                if not face_crop_written and haar_cascade is not None:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                    if len(faces) > 0:
                        # choose largest face
                        faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
                        (x, y, bw, bh) = faces[0]
                        x1, y1, x2, y2 = x, y, x + bw, y + bh
                        face = image[y1:y2, x1:x2]
                        if face.size > 0:
                            cv2.imwrite(str(temp_dir / img_path.name), face)
                            face_crop_written = True
                # If still not cropped, copy original image
                if not face_crop_written:
                    shutil.copy2(img_path, temp_dir / img_path.name)
            except Exception:
                # on any failure, copy original
                try:
                    shutil.copy2(img_path, temp_dir / img_path.name)
                except Exception:
                    pass

#########################################################################
#                             Stats Persistence                         #
#########################################################################

def ensure_dir(p: Path):
    """
    Ensure that a directory exists. If not, create it along with parent
    directories.
    """
    p.mkdir(parents=True, exist_ok=True)

def load_index(parent: Path) -> Dict:
    """
    Load global index (counters across all runs) from disk. Creates
    necessary directories if absent.
    """
    ensure_dir(parent)
    p = parent / "global_index.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "group_counts": {},
        "global_stats": {"images_total": 0, "images_unknown_only": 0, "images_group_only": 0},
        "last_run": None
    }

def _atomic_write(path: Path, text: str):
    """
    Atomically write text to a file: write to a temporary file then rename.
    """
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

def save_index(parent: Path, idx: Dict):
    """
    Persist global index with updated last run timestamp.
    """
    ensure_dir(parent)
    idx["last_run"] = datetime.now().isoformat(timespec="seconds")
    _atomic_write(parent / "global_index.json", json.dumps(idx, ensure_ascii=False, indent=2))

#########################################################################
#                    Network URL (for LAN access)                       #
#########################################################################

def get_lan_ip() -> str:
    """
    Attempt to discover the machine's LAN IP. Falls back to host name or
    loopback if necessary.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"

def get_network_url() -> str:
    """
    Build a LAN-accessible URL for the Streamlit server. Considers optional
    base path configuration.
    """
    ip = get_lan_ip()
    port = st.get_option("server.port") or 8501
    base = st.get_option("server.baseUrlPath") or ""
    base = ("/" + base.strip("/")) if base else ""
    return f"http://{ip}:{port}{base}"

#########################################################################
#                               Helpers                                 #
#########################################################################

def _init_state():
    """
    Initialize session state variables. These hold user selections, logging
    information and UI context.
    """
    st.session_state.setdefault("parent_path", None)
    st.session_state.setdefault("current_dir", None)
    st.session_state.setdefault("selected_dirs", set())
    st.session_state.setdefault("rename_target", None)
    st.session_state.setdefault("queue", [])
    st.session_state.setdefault("logs", [])
    st.session_state.setdefault("proc_logs", [])
    st.session_state.setdefault("delete_target", None)

def log(msg: str):
    """
    Append a timestamped message to the session logs.
    """
    st.session_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def list_dir(p: Path) -> List[Path]:
    """
    Return a sorted list of directory entries. Files are listed after
    directories, case-insensitively.
    """
    try:
        items = list(p.iterdir())
    except Exception:
        return []
    items.sort(key=lambda x: (x.is_file(), x.name.lower()))
    return items

def human_size(n: int) -> str:
    """
    Convert a byte count into a human-readable string with units.
    """
    units = ["Б", "КБ", "МБ", "ГБ", "ТБ", "ПБ"]
    size = float(n)
    for i, u in enumerate(units):
        if size < 1024 or i == len(units) - 1:
            return f"{size:.0f} {u}" if u == "Б" else f"{size:.1f} {u}"
        size /= 1024.0

def load_person_index(group_dir: Path) -> Dict:
    """
    Read the person_index.json file for a group directory, performing
    schema migrations if necessary to accommodate older versions.
    """
    p = group_dir / "person_index.json"
    data = {"persons": []}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return data

    persons = data.get("persons", [])
    changed = False
    for person in persons:
        if "protos" not in person:
            if "proto" in person:
                v = person.pop("proto")
                person["protos"] = [v]
                changed = True
            else:
                person["protos"] = []
                changed = True
        if "ema" not in person:
            person["ema"] = person["protos"][0] if person["protos"] else None
            changed = True
        if "count" not in person:
            person["count"] = max(1, len(person.get("protos", [])))
            changed = True
        if "thr" not in person:
            person["thr"] = None
            changed = True

    if changed:
        try:
            _atomic_write(
                group_dir / "person_index.json",
                json.dumps({"persons": persons}, ensure_ascii=False, indent=2),
            )
        except Exception:
            pass

    return {"persons": persons}

def save_person_index(group_dir: Path, data: Dict):
    """
    Persist person-specific index information into person_index.json.
    """
    _atomic_write(group_dir / "person_index.json", json.dumps(data, ensure_ascii=False, indent=2))

def _is_subpath(p: Path, root: Path) -> bool:
    """
    Determine whether path p lies within root. Returns True for relative
    and absolute sub-paths, False otherwise.
    """
    try:
        p.resolve().relative_to(root.resolve()); return True
    except Exception:
        return False

def cleanup_processed_images(group_dir: Path, processed_images: Set[Path]):
    """
    Remove original images after they have been safely copied into person
    directories. Protected roots (unknown/group only and numbered person
    directories) are skipped. Only files inside group_dir are touched.
    """
    protected_roots = {group_dir / "__unknown__", group_dir / "__group_only__"}
    protected_roots |= {d for d in group_dir.iterdir() if d.is_dir() and d.name.isdigit()}

    for img_path in list(processed_images):
        try:
            if not _is_subpath(img_path, group_dir):
                continue
            if any(_is_subpath(img_path, r) for r in protected_roots):
                continue
            if img_path.exists() and img_path.is_file():
                img_path.unlink()
                log(f"Удален оригинал: {img_path.name}")
        except Exception as e:
            log(f"Ошибка удаления {img_path.name}: {e}")

def _normalize_np(v):
    """
    Normalize a vector to unit length using numpy for stability.
    """
    import numpy as _np
    arr = _np.array(v, dtype=_np.float32)
    n = float(_np.linalg.norm(arr) + 1e-12)
    return (arr / n).astype(_np.float32)

def _update_person_proto(person: Dict, new_vec, k_max: int = 5, ema_alpha: float = 0.9):
    """
    Update a person's prototype vectors and exponential moving average (EMA)
    with a new observation. Limits the number of prototypes to k_max using
    farthest point sampling.
    """
    import numpy as _np
    nv = _normalize_np(new_vec).tolist()
    protos = person.get("protos", [])
    protos.append(nv)

    if len(protos) > k_max:
        X = _np.stack([_normalize_np(v) for v in protos], axis=0)
        keep = [int(_np.argmax(_np.linalg.norm(X - X.mean(0), axis=1)))]
        while len(keep) < k_max:
            dists = _np.min((1.0 - X @ X[keep].T), axis=1)
            cand = int(_np.argmax(dists))
            if cand not in keep:
                keep.append(cand)
        protos = [X[i].tolist() for i in keep]

    ema = person.get("ema")
    ema_np = _normalize_np(ema if ema is not None else protos[0])
    ema_np = ema_alpha * ema_np + (1.0 - ema_alpha) * _normalize_np(new_vec)
    ema_np = _normalize_np(ema_np)

    person["protos"] = protos
    person["ema"] = ema_np.tolist()
    person["count"] = int(person.get("count", 0)) + 1

def safe_copy(src: Path, dst_dir: Path) -> Path:
    """
    Copy a file to the destination directory, avoiding name collisions by
    appending a counter if necessary. Returns the destination path.
    """
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    if dst.exists():
        stem, ext = src.stem, src.suffix
        i = 1
        while (dst_dir / f"{stem} ({i}){ext}").exists():
            i += 1
        dst = dst_dir / f"{stem} ({i}){ext}"
    shutil.copy2(src, dst)
    return dst

def safe_move(src: Path, dst_dir: Path) -> Path:
    """
    Move a file to the destination directory, avoiding name collisions by
    appending a counter if necessary. Returns the destination path.
    """
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    if dst.exists():
        stem, ext = src.stem, src.suffix
        i = 1
        while (dst_dir / f"{stem} ({i}){ext}").exists():
            i += 1
        dst = dst_dir / f"{stem} ({i}){ext}"
    shutil.move(str(src), str(dst))
    return dst

def match_and_apply(group_dir: Path, plan: Dict, match_thr: float) -> Tuple[int, Set[Path]]:
    """
    Given a clustering plan, update the person index to incorporate new
    clusters or assign clusters to existing persons. Copy images into
    per-person folders and return the new number of persons and the set of
    processed images (for deletion).
    """
    import numpy as _np

    top2_margin = float(CFG.get("top2_margin", 0.08))

    person_idx = load_person_index(group_dir)
    persons = person_idx.get("persons", [])

    raw_centroids = plan.get("cluster_centroids", {}) or {}
    centroids_norm = {}
    for cid, vec in raw_centroids.items():
        try:
            cid_int = int(cid)
        except Exception:
            cid_int = cid
        centroids_norm[cid_int] = _normalize_np(vec)

    proto_list = []
    proto_owner = []
    per_thr = {}

    for p in persons:
        num = int(p["number"])
        thr_i = p.get("thr")
        if thr_i is not None:
            try:
                per_thr[num] = float(thr_i)
            except Exception:
                pass
        protos = p.get("protos") or []
        if not protos and p.get("ema"):
            protos = [p["ema"]]
        for v in protos:
            proto_list.append(_normalize_np(v))
            proto_owner.append(num)

    if len(proto_list) > 0:
        P = _np.stack(proto_list, axis=0)
    else:
        P = None

    assigned: Dict[int, int] = {}
    new_nums: Dict[int, int] = {}

    existing_nums = sorted([int(d.name) for d in group_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    cur_max = existing_nums[-1] if existing_nums else 0

    eligible = [int(c) if str(c).isdigit() else c for c in plan.get("eligible_clusters", [])]

    for cid in eligible:
        c = centroids_norm.get(cid)
        if c is None:
            continue

        if P is not None and len(P) > 0:
            sims = (P @ c.astype(_np.float32))

            per_person_scores: Dict[int, float] = {}
            for s, owner in zip(sims.tolist(), proto_owner):
                if owner not in per_person_scores or s > per_person_scores[owner]:
                    per_person_scores[owner] = s

            if not per_person_scores:
                best_num = None; s1 = -1.0; s2 = -1.0
            else:
                sorted_pairs = sorted(per_person_scores.items(), key=lambda x: x[1], reverse=True)
                best_num, s1 = sorted_pairs[0]
                s2 = sorted_pairs[1][1] if len(sorted_pairs) > 1 else -1.0

            thr_use = max(float(match_thr), float(per_thr.get(best_num, -1e9)))

            if (best_num is not None) and (s1 >= thr_use) and (s1 - s2 >= top2_margin):
                assigned[cid] = int(best_num)
                for p in persons:
                    if int(p["number"]) == int(best_num):
                        _update_person_proto(p, c)
                        break
            else:
                cur_max += 1
                new_nums[cid] = cur_max
                persons.append({
                    "number": cur_max,
                    "protos": [c.tolist()],
                    "ema": c.tolist(),
                    "count": 1,
                    "thr": None
                })
        else:
            cur_max += 1
            new_nums[cid] = cur_max
            persons.append({
                "number": cur_max,
                "protos": [c.tolist()],
                "ema": c.tolist(),
                "count": 1,
                "thr": None
            })

    for cid, num in {**assigned, **new_nums}.items():
        ensure_dir(group_dir / str(num))

    copied: Set[Tuple[int, Path]] = set()
    cluster_images = {}
    for k, v in (plan.get("cluster_images", {}) or {}).items():
        try:
            cluster_images[int(k)] = v
        except Exception:
            cluster_images[k] = v

    for cid in eligible:
        num = assigned.get(cid, new_nums.get(cid))
        if num is None:
            continue
        for img in cluster_images.get(cid, []):
            p = Path(img)
            key = (num, p)
            if key in copied:
                continue
            dst_dir = group_dir / str(num)
            try:
                safe_copy(p, dst_dir)
            except Exception:
                pass
            copied.add(key)

    go = plan.get("group_only_images", []) or []
    if go:
        dst_dir = group_dir / "__group_only__"
        ensure_dir(dst_dir)
        for img in go:
            p = Path(img)
            try:
                safe_copy(p, dst_dir)
            except Exception:
                pass

    un = plan.get("unknown_images", []) or []
    if un:
        dst_dir = group_dir / "__unknown__"
        ensure_dir(dst_dir)
        for img in un:
            p = Path(img)
            try:
                safe_copy(p, dst_dir)
            except Exception:
                pass

    person_idx["persons"] = persons
    save_person_index(group_dir, person_idx)

    processed_images: Set[Path] = set()
    for cid in eligible:
        for img in cluster_images.get(cid, []):
            processed_images.add(Path(img))
    for img in go:
        processed_images.add(Path(img))
    for img in un:
        processed_images.add(Path(img))

    # Keep only processed images inside group_dir
    processed_images = {p for p in processed_images if _is_subpath(p, group_dir)}
    return len(persons), processed_images

def make_square_thumb(img_path: Path, size: int = 150):
    """
    Create a square thumbnail from the centre of an image using PIL. If PIL
    is unavailable, return None to let Streamlit fallback to raw image.
    """
    if Image is None:
        return None
    try:
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        if w != h:
            min_side = min(w, h)
            left = (w - min_side) // 2
            top = (h - min_side) // 2
            im = im.crop((left, top, left + min_side, top + min_side))
        im = im.resize((size, size))
        return im
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def get_thumb_bytes(path_str: str, size: int, mtime: float):
    """
    Streamlit cache helper to load and cache thumbnails based on file path
    and modification time.
    """
    if Image is None:
        return None
    im = make_square_thumb(Path(path_str), size)
    if im is None:
        return None
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

#########################################################################
#                             Streamlit UI                              #
#########################################################################

_init_state()

# Page title and LAN URL
st.title("Face Sorter — Мини-проводник (YOLO-версия)")

net_url = get_network_url()
st.info(f"Сетевой URL (LAN): {net_url}")
try:
    st.link_button("Открыть по сети", net_url, use_container_width=True)
except Exception:
    st.markdown(f"[Открыть по сети]({net_url})")
st.text_input("Network URL", value=net_url, label_visibility="collapsed")

# Root selection
if st.session_state["parent_path"] is None:
    st.info("Выберите папку для работы.")
    pick_cols = st.columns([0.25, 0.75])
    with pick_cols[0]:
        if st.button("📂 ВЫБРАТЬ ПАПКУ", type="primary", use_container_width=True):
            folder = pick_folder_dialog()
            if folder:
                st.session_state["parent_path"] = folder
                st.session_state["current_dir"] = folder
                st.rerun()
    with pick_cols[1]:
        manual = st.text_input("Или введите путь вручную", value="", placeholder="D:\\Папка\\Проект")
        if st.button("ОК", use_container_width=True):
            if manual and Path(manual).exists():
                st.session_state["parent_path"] = manual
                st.session_state["current_dir"] = manual
                st.rerun()
            else:
                st.error("Путь не существует.")
else:
    curr = Path(st.session_state["current_dir"]).expanduser().resolve()
    parent_root = Path(st.session_state["parent_path"]).expanduser().resolve()

    # Top controls: Up, refresh, breadcrumb navigation
    top_cols = st.columns([0.08, 0.12, 0.80])
    with top_cols[0]:
        up = None if curr == parent_root else curr.parent
        st.button(
            "⬆️ Вверх",
            key="up",
            disabled=(up is None),
            on_click=(lambda: st.session_state.update({"current_dir": str(up)}) if up else None),
            use_container_width=True
        )
    with top_cols[1]:
        if st.button("🔄 Обновить", use_container_width=True):
            st.rerun()
    with top_cols[2]:
        # Build breadcrumb path relative to parent_root
        def _breadcrumbs(path: Path, root: Path):
            chain = []
            p = path
            while True:
                chain.append(p)
                if p == root:
                    break
                p = p.parent
            return list(reversed(chain))
        crumbs = _breadcrumbs(curr, parent_root)
        bc_cols = st.columns(len(crumbs))
        for i, pth in enumerate(crumbs):
            label = pth.name if pth != pth.anchor else str(pth)
            with bc_cols[i]:
                st.button(
                    label or "/",
                    key=f"bc::{i}",
                    use_container_width=True,
                    on_click=lambda p=str(pth): st.session_state.update({"current_dir": p})
                )

    st.markdown("---")

    # Header row
    st.markdown('<div class="row hdr"><div>Превью</div><div>Имя</div><div>Тип</div><div>Изменён</div><div>Размер</div></div>', unsafe_allow_html=True)

    # Scrollable explorer
    with st.container(height=700):
        items = list_dir(curr)
        for item in items:
            is_dir = item.is_dir()
            sel_key = f"sel::{item}"
            name_btn_key = f"open::{item}"
            del_key = f"del::{item}"
            ren_key = f"ren::{item}"
            ren_input_key = f"ren_input::{item}"

            c1, c2, c3, c4, c5 = st.columns([0.14, 0.58, 0.12, 0.14, 0.10])

            # Preview
            with c1:
                if not is_dir and item.suffix.lower() in IMG_EXTS:
                    try:
                        data = get_thumb_bytes(str(item), 150, item.stat().st_mtime)
                    except Exception:
                        data = None
                    if data:
                        st.image(data)
                    else:
                        st.image(str(item), width=150)
                else:
                    st.markdown('<div class="thumbbox">📁</div>' if is_dir else '<div class="thumbbox">🗎</div>', unsafe_allow_html=True)

            # Name + inline icons
            with c2:
                icon = "📁" if is_dir else "🗎"
                name_cols = st.columns([0.72, 0.10, 0.10, 0.08])
                with name_cols[0]:
                    if is_dir:
                        if st.button(f"{icon} {item.name}", key=name_btn_key, use_container_width=True):
                            st.session_state["current_dir"] = str(item)
                            st.rerun()
                    else:
                        st.write(f"{icon} {item.name}")
                with name_cols[1]:
                    if is_dir:
                        checked = st.checkbox(
                            "Выбрать",
                            key=sel_key,
                            value=(str(item) in st.session_state["selected_dirs"]),
                            help="В очередь",
                            label_visibility="collapsed",
                        )
                        if checked:
                            st.session_state["selected_dirs"].add(str(item))
                        else:
                            st.session_state["selected_dirs"].discard(str(item))
                with name_cols[2]:
                    if st.button("✏️", key=ren_key, help="Переименовать", use_container_width=True):
                        st.session_state["rename_target"] = str(item)
                with name_cols[3]:
                    if st.button("🗑️", key=del_key, help="Удалить", use_container_width=True):
                        st.session_state["delete_target"] = str(item)

            with c3:
                st.write("Папка" if is_dir else (item.suffix[1:].upper() if item.suffix else "Файл"))
            with c4:
                try:
                    st.write(datetime.fromtimestamp(item.stat().st_mtime).strftime("%Y-%m-%d %H:%M"))
                except Exception:
                    st.write("—")
            with c5:
                if is_dir:
                    st.write("—")
                else:
                    try:
                        st.write(human_size(item.stat().st_size))
                    except Exception:
                        st.write("—")

            # Inline rename row
            if st.session_state.get("rename_target") == str(item):
                rc1, rc2, rc3 = st.columns([0.70, 0.15, 0.15])
                with rc1:
                    new_name = st.text_input("Новое имя", value=item.name, key=ren_input_key, label_visibility="collapsed")
                with rc2:
                    if st.button("Сохранить", key=f"save::{item}", use_container_width=True):
                        try:
                            new_path = item.parent / new_name
                            if new_path.exists():
                                st.error("Файл/папка с таким именем уже существует.")
                            else:
                                item.rename(new_path)
                                st.session_state["rename_target"] = None
                                st.rerun()
                        except Exception as e:
                            st.error(f"Ошибка: {e}")
                with rc3:
                    if st.button("Отмена", key=f"cancel::{item}", use_container_width=True):
                        st.session_state["rename_target"] = None
                        st.rerun()

            # Inline delete confirm
            if st.session_state.get("delete_target") == str(item):
                dc1, dc2, dc3 = st.columns([0.70, 0.15, 0.15])
                with dc1:
                    st.markdown(f"❗ Подтвердите удаление: **{item.name}**")
                with dc2:
                    if st.button("Удалить", type="primary", key=f"confirm_del::{item}", use_container_width=True):
                        try:
                            # ensure deletion stays within root
                            if not _is_subpath(item, parent_root):
                                st.error("Удаление вне корня запрещено.")
                            else:
                                if send2trash is not None:
                                    send2trash(str(item))
                                else:
                                    if is_dir:
                                        shutil.rmtree(item, ignore_errors=True)
                                    else:
                                        item.unlink(missing_ok=True)
                                st.session_state["delete_target"] = None
                                st.rerun()
                        except Exception as e:
                            st.error(f"Ошибка удаления: {e}")
                            st.session_state["delete_target"] = None
                with dc3:
                    if st.button("Отмена", key=f"cancel_del::{item}", use_container_width=True):
                        st.session_state["delete_target"] = None

    st.markdown("---")

    # Drag & Drop Move panel
    with st.expander("Переместить файлы (Drag & Drop)", expanded=False):
        if sort_items is None:
            st.info("Для DnD установите пакет: pip install streamlit-sortables")
        else:
            files_in_curr = [str(p) for p in curr.iterdir() if p.is_file()]
            subfolders = [p for p in curr.iterdir() if p.is_dir()]

            containers = [{"header": "Файлы (текущая папка)", "items": files_in_curr}]
            for f in subfolders:
                containers.append({"header": f.name, "items": []})

            result = sort_items(containers, multi_containers=True)

            if st.button("Применить переносы", use_container_width=True):
                moves = []
                header_to_dir = {f.name: f for f in subfolders}
                for i, cont in enumerate(result):
                    if i == 0:
                        continue
                    target_name = cont.get("header", "")
                    target_dir = header_to_dir.get(target_name)
                    if not target_dir:
                        continue
                    for src_str in cont.get("items", []):
                        src_path = Path(src_str)
                        if src_path.exists() and src_path.is_file():
                            moves.append((src_path, target_dir))
                ok = 0; errors = 0
                for src, dst_dir in moves:
                    try:
                        safe_move(src, dst_dir)
                        ok += 1
                    except Exception as e:
                        errors += 1
                        st.error(f"Не удалось переместить {src.name}: {e}")
                st.success(f"Перемещено: {ok}, ошибок: {errors}")
                st.rerun()

    # Footer actions
    colA, colB, colC = st.columns([0.35, 0.35, 0.30])
    with colA:
        if st.button("➕ Добавить в очередь", type="secondary", use_container_width=True):
            added = 0
            for d in list(st.session_state["selected_dirs"]):
                if d not in st.session_state["queue"]:
                    st.session_state["queue"].append(d)
                    added += 1
            st.success(f"Добавлено в очередь: {added}")
    with colB:
        if st.button("🧹 Очистить очередь", use_container_width=True):
            st.session_state["queue"] = []
            st.session_state["selected_dirs"] = set()
            st.info("Очередь очищена.")
    with colC:
        st.write(f"В очереди: {len(st.session_state['queue'])}")

    #######################################################################
    #                           Processing Logic                          #
    #######################################################################

    if st.button("▶️ Обработать", type="primary", use_container_width=True):
        parent = parent_root
        idx = load_index(parent)
        st.session_state["proc_logs"] = []

        targets = [Path(p) for p in st.session_state["queue"]]
        if not targets:
            targets = [p for p in curr.iterdir() if p.is_dir()]
        if not targets:
            st.warning("Нет целей для обработки.")
        else:
            tot_total = tot_unknown = tot_group_only = 0
            tot_faces = tot_unique_people = tot_joint = 0

            # Preload YOLO model if enabled
            yolo_net = None
            temp_dirs = {}
            if CFG.get("use_yolo", False):
                yolo_net = load_yolo_model(CFG.get("yolo_cfg"), CFG.get("yolo_weights"))
                if yolo_net is None:
                    st.warning("Модель YOLO не загружена. Обработка будет выполнена без YOLO.")

            status = st.status("Идёт обработка…", expanded=True)
            with status:
                prog = st.progress(0, text=f"0/{len(targets)}")
                for k, gdir in enumerate(targets, start=1):
                    st.write(f"Обработка: **{gdir.name}**")
                    try:
                        # If YOLO is enabled and model loaded, pre-process images
                        working_dir = gdir
                        temp_dir = None
                        if CFG.get("use_yolo", False) and yolo_net is not None:
                            temp_dir = gdir / ".face_yolo_tmp"
                            if temp_dir.exists():
                                shutil.rmtree(temp_dir)
                            temp_dir.mkdir(parents=True, exist_ok=True)
                            crop_faces_with_yolo(
                                gdir,
                                temp_dir,
                                yolo_net,
                                CFG.get("yolo_conf_thr", 0.5),
                                CFG.get("yolo_nms_thr", 0.4),
                            )
                            working_dir = temp_dir

                        plan = build_plan(
                            working_dir,
                            group_thr=CFG["group_thr"],
                            eps_sim=CFG["eps_sim"],
                            min_samples=CFG["min_samples"],
                            min_face=CFG["min_face"],
                            blur_thr=CFG["blur_thr"],
                            det_size=CFG["det_size"],
                            gpu_id=CFG["gpu_id"],
                        )
                        cluster_images = plan.get("cluster_images", {}) or {}
                        faces_detections = sum(len(v) for v in cluster_images.values())
                        unique_people_in_run = len(plan.get("eligible_clusters", []))
                        freq = {}
                        for imgs in cluster_images.values():
                            for pth in imgs:
                                freq[pth] = freq.get(pth, 0) + 1
                        joint_images = sum(1 for v in freq.values() if v >= 2)

                        persons_after, processed_images = match_and_apply(gdir, plan, match_thr=CFG["match_thr"])
                        idx["group_counts"][str(gdir)] = persons_after
                        cleanup_processed_images(gdir, processed_images)

                        tot_total += plan["stats"]["images_total"]
                        tot_unknown += plan["stats"]["images_unknown_only"]
                        tot_group_only += plan["stats"]["images_group_only"]
                        tot_faces += faces_detections
                        tot_unique_people += unique_people_in_run
                        tot_joint += joint_images

                        st.success(
                            f"{gdir.name}: фото={plan['stats']['images_total']}, уник.людей={unique_people_in_run}, "
                            f"детекций лиц={faces_detections}, group_only={plan['stats']['images_group_only']}, совместных={joint_images}"
                        )
                        st.session_state["proc_logs"].append(
                            f"{gdir.name}: людей(детекции)={faces_detections}; уникальные люди={unique_people_in_run}; "
                            f"общие(group_only)={plan['stats']['images_group_only']}; совместные(>1 человек)={joint_images}"
                        )

                        # Remove temporary directory if used
                        if CFG.get("use_yolo", False) and yolo_net is not None and temp_dir is not None:
                            try:
                                shutil.rmtree(temp_dir)
                            except Exception:
                                pass
                    except Exception as e:
                        st.error(f"Ошибка в {gdir.name}: {e}")
                        st.session_state["proc_logs"].append(f"{gdir.name}: ошибка — {e}")

                    prog.progress(k / len(targets), text=f"{k}/{len(targets)}")

                status.update(label="Готово", state="complete")

            idx["global_stats"]["images_total"] += tot_total
            idx["global_stats"]["images_unknown_only"] += tot_unknown
            idx["global_stats"]["images_group_only"] += tot_group_only
            save_index(parent, idx)
            st.session_state["queue"] = []
            st.session_state["selected_dirs"] = set()

            st.success("Обработка завершена.")
            st.markdown("**Сводка за прогон:**")
            st.write(f"- Людей на фото (детекции): **{tot_faces}**")
            st.write(f"- Уникальных людей (кластера): **{tot_unique_people}**")
            st.write(f"- Общих фото (group_only): **{tot_group_only}**")
            st.write(f"- Совместных фото (>1 человек): **{tot_joint}**")

            st.markdown("**Детальные логи по группам:**")
            st.text_area("Логи", value="\n".join(st.session_state.get("proc_logs", [])), height=220)