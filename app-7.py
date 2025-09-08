# app.py — Face Sorter (full single-file project)
# Explorer-like DnD + safe ops + audit + run history + LAN URL picker
# Covers risk fixes 1..11 and replaces compact DnD with Windows-Explorer-like UX.

import json
import shutil
import socket
import os
import hashlib
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from datetime import datetime

import streamlit as st

# ---------- Optional deps ----------
try:
    from streamlit_sortables import sort_items  # pip install streamlit-sortables
except Exception:
    sort_items = None

try:
    from send2trash import send2trash
except Exception:
    send2trash = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from filelock import FileLock
except Exception:
    FileLock = None

# ---- Backend (provided by your project) ----
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
            "stats": {"images_total": 0, "images_unknown_only": 0, "images_group_only": 0}
        }

st.set_page_config(page_title="Face Sorter — Мини-проводник (Explorer)", layout="wide")

# ---------- Utilities ----------

def _clamp(v, lo, hi, default):
    try:
        v = type(default)(v); return max(lo, min(hi, v))
    except Exception:
        return default

CFG_BASE = Path(__file__).parent


def load_config(base: Path) -> Dict:
    p = base / "config.json"
    defaults = {
        "group_thr": 3, "eps_sim": 0.55, "min_samples": 2, "min_face": 110,
        "blur_thr": 45.0, "det_size": 640, "gpu_id": 0,
        "match_thr": 0.52, "top2_margin": 0.08,
        "per_person_min_obs": 10, "min_det_score": 0.50, "min_quality": 0.50
    }
    if p.exists():
        try:
            user_cfg = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(user_cfg, dict):
                defaults.update({k: user_cfg[k] for k in user_cfg})
        except Exception:
            pass
    defaults["eps_sim"] = _clamp(defaults["eps_sim"], 0.0, 1.0, 0.55)
    defaults["match_thr"] = _clamp(defaults["match_thr"], 0.0, 1.0, 0.52)
    defaults["top2_margin"] = _clamp(defaults["top2_margin"], 0.0, 1.0, 0.08)
    defaults["min_face"] = _clamp(defaults["min_face"], 0, 10000, 110)
    defaults["det_size"] = _clamp(defaults["det_size"], 64, 4096, 640)
    return defaults

CFG = load_config(CFG_BASE)

# ---------- Persistence ----------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_index(parent: Path) -> Dict:
    ensure_dir(parent)
    p = parent / "global_index.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"group_counts": {},
            "global_stats": {"images_total": 0, "images_unknown_only": 0, "images_group_only": 0},
            "last_run": None,
            "runs": []}


def _atomic_write(path: Path, text: str):
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def save_index(parent: Path, idx: Dict):
    ensure_dir(parent)
    idx["last_run"] = datetime.now().isoformat(timespec="seconds")
    out = json.dumps(idx, ensure_ascii=False, indent=2)
    target = parent / "global_index.json"
    if FileLock is not None:
        lock = FileLock(str(target) + ".lock")
        with lock:
            _atomic_write(target, out)
    else:
        _atomic_write(target, out)

# ---------- Network URL ----------

PRIVATE_NETS = ("10.", "172.16.", "172.17.", "172.18.", "172.19.", "172.2", "192.168.")


def _ip_is_private(ip: str) -> bool:
    return any(ip.startswith(p) for p in PRIVATE_NETS)


def get_lan_candidates() -> List[str]:
    cand = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            cand.add(s.getsockname()[0])
    except Exception:
        pass
    try:
        cand.add(socket.gethostbyname(socket.gethostname()))
    except Exception:
        pass
    env_ip = os.environ.get("PREFERRED_LAN_IP")
    if env_ip:
        cand.add(env_ip)
    ordered = sorted(cand, key=lambda x: (not _ip_is_private(x), x))
    return ordered or ["127.0.0.1"]


def make_network_url(ip: str) -> str:
    port = st.get_option("server.port") or 8501
    base = st.get_option("server.baseUrlPath") or ""
    base = ("/" + base.strip("/")) if base else ""
    return f"http://{ip}:{port}{base}"

# ---------- State & helpers ----------

def _init_state():
    st.session_state.setdefault("parent_path", None)
    st.session_state.setdefault("current_dir", None)
    st.session_state.setdefault("selected_dirs", set())
    st.session_state.setdefault("rename_target", None)
    st.session_state.setdefault("queue", [])
    st.session_state.setdefault("logs", [])
    st.session_state.setdefault("proc_logs", [])
    st.session_state.setdefault("delete_target", None)
    st.session_state.setdefault("sort_key", "name")
    st.session_state.setdefault("sort_desc", False)
    st.session_state.setdefault("del_after_copy", False)
    st.session_state.setdefault("confirm_hard_delete", False)
    st.session_state.setdefault("lan_ip", None)


def log(msg: str):
    st.session_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def _k(prefix: str, p: Path | str) -> str:
    s = str(p)
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}::{h}"


def _is_subpath(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve()); return True
    except Exception:
        return False


# ---------- Folder picker ----------

def pick_folder_dialog() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
        folder = filedialog.askdirectory(title="Выберите папку")
        root.destroy()
        return folder
    except Exception:
        return None


# ---------- Filesystem helpers ----------

def list_dir(p: Path) -> List[Path]:
    try:
        items = list(p.iterdir())
    except Exception:
        return []
    key = st.session_state.get("sort_key", "name")
    desc = bool(st.session_state.get("sort_desc", False))

    def sort_tuple(x: Path):
        primary = 0 if x.is_dir() else 1  # folders first
        if key == "name":
            sec = x.name.lower()
        elif key == "date":
            try:
                sec = x.stat().st_mtime
            except Exception:
                sec = 0
        elif key == "size":
            try:
                sec = x.stat().st_size if x.is_file() else -1
            except Exception:
                sec = -1
        else:
            sec = x.name.lower()
        return (primary, sec)

    items.sort(key=sort_tuple, reverse=desc)
    return items


def human_size(n: int) -> str:
    units = ["Б", "КБ", "МБ", "ГБ", "ТБ", "ПБ"]
    size = float(n)
    for i, u in enumerate(units):
        if size < 1024 or i == len(units) - 1:
            return f"{size:.0f} {u}" if u == "Б" else f"{size:.1f} {u}"
        size /= 1024.0


# ---------- Person index ----------

def load_person_index(group_dir: Path) -> Dict:
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
                v = person.pop("proto"); person["protos"] = [v]
            else:
                person["protos"] = []
            changed = True
        if "ema" not in person:
            person["ema"] = person["protos"][0] if person["protos"] else None; changed = True
        if "count" not in person:
            person["count"] = max(1, len(person.get("protos", []))); changed = True
        if "thr" not in person:
            person["thr"] = None; changed = True
    if changed:
        try:
            _atomic_write(group_dir / "person_index.json",
                          json.dumps({"persons": persons}, ensure_ascii=False, indent=2))
        except Exception:
            pass
    return {"persons": persons}


def save_person_index(group_dir: Path, data: Dict):
    _atomic_write(group_dir / "person_index.json", json.dumps(data, ensure_ascii=False, indent=2))


# ---------- Embedding helpers ----------

def _normalize_np(v):
    import numpy as np
    arr = np.array(v, dtype=np.float32)
    n = float(np.linalg.norm(arr) + 1e-12)
    return (arr / n).astype(np.float32)


def _update_person_proto(person: Dict, new_vec, k_max: int = 5, ema_alpha: float = 0.9):
    import numpy as np
    nv = _normalize_np(new_vec).tolist()
    protos = person.get("protos", [])
    protos.append(nv)
    if len(protos) > k_max:
        X = np.stack([_normalize_np(v) for v in protos], axis=0)
        d0 = np.linalg.norm(X - X.mean(0), axis=1)
        keep = [int(np.argmax(d0))]
        while len(keep) < k_max and len(keep) < len(X):
            sims = X @ X[keep].T
            dists = 1.0 - np.max(sims, axis=1)
            for idx in keep:
                dists[idx] = -np.inf
            cand = int(np.argmax(dists))
            if dists[cand] == -np.inf:
                break
            keep.append(cand)
        if len(keep) < min(k_max, len(X)):
            for i in range(len(X)):
                if i not in keep:
                    keep.append(i)
                if len(keep) >= min(k_max, len(X)):
                    break
        protos = [X[i].tolist() for i in keep]
    ema = person.get("ema")
    ema_np = _normalize_np(ema if ema is not None else protos[0])
    ema_np = ema_alpha * ema_np + (1.0 - ema_alpha) * _normalize_np(new_vec)
    ema_np = _normalize_np(ema_np)
    person["protos"] = protos
    person["ema"] = ema_np.tolist()
    person["count"] = int(person.get("count", 0)) + 1


# ---------- Safe file ops ----------

def safe_copy(src: Path, dst_dir: Path) -> Path:
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


# ---------- Matching & application ----------

def match_and_apply(group_dir: Path, plan: Dict, match_thr: float) -> Tuple[int, Set[Path], Dict]:
    """Return persons_count, processed_sources, audit_log."""
    import numpy as np
    top2_margin = float(CFG.get("top2_margin", 0.08))
    person_idx = load_person_index(group_dir)
    persons = person_idx.get("persons", [])

    raw_centroids = plan.get("cluster_centroids", {}) or {}
    centroids_norm: Dict[int, any] = {}
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

    P = None
    if len(proto_list) > 0:
        P = np.stack(proto_list, axis=0)

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
            sims = (P @ c.astype("float32"))
            sims = np.nan_to_num(sims, nan=-1.0)
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
            thr_use = max(float(match_thr), float(per_thr.get(best_num, -1e9))) if best_num is not None else float(match_thr)
            if (best_num is not None) and (s1 >= thr_use) and (s1 - s2 >= top2_margin):
                assigned[cid] = int(best_num)
                for p in persons:
                    if int(p["number"]) == int(best_num):
                        _update_person_proto(p, c)
                        break
            else:
                cur_max += 1
                new_nums[cid] = cur_max
                persons.append({"number": cur_max, "protos": [c.tolist()], "ema": c.tolist(), "count": 1, "thr": None})
        else:
            cur_max += 1
            new_nums[cid] = cur_max
            persons.append({"number": cur_max, "protos": [c.tolist()], "ema": c.tolist(), "count": 1, "thr": None})

    for cid, num in {**assigned, **new_nums}.items():
        ensure_dir(group_dir / str(num))

    copied: Set[Tuple[int, Path]] = set()
    audit: Dict[str, Dict] = {"copied": {}, "group_only": 0, "unknown": 0}

    # Normalize keys to int where possible
    cluster_images: Dict[int, List[str]] = {}
    for k, v in (plan.get("cluster_images", {}) or {}).items():
        try:
            cluster_images[int(k)] = v
        except Exception:
            cluster_images[k] = v

    # Copy cluster images
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
                audit["copied"].setdefault(str(num), []).append(str(p))
            except Exception:
                pass
            copied.add(key)

    # Copy group-only & unknown
    go = list(plan.get("group_only_images", []) or [])
    if go:
        dst_dir = group_dir / "__group_only__"; ensure_dir(dst_dir)
        for img in go:
            p = Path(img)
            try:
                safe_copy(p, dst_dir); audit["group_only"] += 1
            except Exception:
                pass

    un = list(plan.get("unknown_images", []) or [])
    if un:
        dst_dir = group_dir / "__unknown__"; ensure_dir(dst_dir)
        for img in un:
            p = Path(img)
            try:
                safe_copy(p, dst_dir); audit["unknown"] += 1
            except Exception:
                pass

    person_idx["persons"] = persons
    save_person_index(group_dir, person_idx)

    # Sources to potentially delete — ONLY those referenced by plan
    processed_sources: Set[Path] = set()
    for imgs in cluster_images.values():
        for s in imgs:
            processed_sources.add(Path(s))
    for s in go:
        processed_sources.add(Path(s))
    for s in un:
        processed_sources.add(Path(s))

    return len(persons), processed_sources, audit


# ---------- Thumbs ----------

def make_square_thumb(img_path: Path, size: int = 150):
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


@st.cache_data(show_spinner=False, ttl=3600)
def get_thumb_bytes(path_str: str, size: int, mtime: float):
    if Image is None:
        return None
    im = make_square_thumb(Path(path_str), size)
    if im is None:
        return None
    buf = BytesIO(); im.save(buf, format="PNG")
    return buf.getvalue()


# ---------- Explorer-like DnD ----------

def render_explorer_dnd(curr: Path, parent_root: Path):
    """Drag-and-drop в стиле Windows Explorer.
    Источники: содержимое текущей папки.
    Приёмники: подпапки текущей + «Вверх».
    Переключатель: копирование/перемещение.
    """
    if sort_items is None:
        st.error("DnD требует пакет: pip install streamlit-sortables")
        return

    # Toolbar
    mode_copy = st.toggle("Копировать (Ctrl‑drag)", value=False, help="Если выключено — перемещение")

    # Sources — current files/folders (filtering images + folders for clarity)
    src_items: List[str] = []
    try:
        for p in curr.iterdir():
            if p.is_dir() or (p.is_file() and p.suffix.lower() in IMG_EXTS):
                src_items.append(str(p))
    except Exception:
        pass

    # Targets: Up + subfolders
    subfolders = [p for p in curr.iterdir() if p.is_dir()]
    up_label = None
    if curr != parent_root:
        up_disp = curr.parent.name or str(curr.parent)
        up_label = f"⬆️ Вверх ({up_disp})"

    containers = [{"header": "📦 Источники (текущая папка)", "items": src_items}]
    if up_label:
        containers.append({"header": up_label, "items": []})
    for f in subfolders:
        containers.append({"header": f"📁 {f.name}", "items": []})

    result = sort_items(containers, multi_containers=True, direction="vertical")

    if st.button("Применить", key=_k("dnd_apply", str(curr)), type="primary", use_container_width=True):
        header_to_dir = {f"📁 {f.name}": f for f in subfolders}
        if up_label:
            header_to_dir[up_label] = curr.parent

        assigned_once: Set[str] = set()
        ok = skp = err = 0
        for cont in (result or []):
            header = cont.get("header", "")
            if header == "📦 Источники (текущая папка)":
                continue
            dst_dir = header_to_dir.get(header)
            if not dst_dir:
                continue
            for src_str in cont.get("items", []):
                if src_str in assigned_once:
                    skp += 1; continue
                assigned_once.add(src_str)
                src = Path(src_str)
                try:
                    if not src.exists():
                        skp += 1; continue
                    # Disallow moving folder into its own subtree
                    if src.is_dir() and _is_subpath(dst_dir, src):
                        skp += 1; continue
                    # No-op: already in that folder
                    if src.parent.resolve() == dst_dir.resolve():
                        skp += 1; continue
                    if mode_copy:
                        safe_copy(src, dst_dir)
                    else:
                        safe_move(src, dst_dir)
                    ok += 1
                except Exception as e:
                    st.error(f"Не удалось обработать {src.name}: {e}"); err += 1
        st.success(f"Операции: {ok}; пропущено: {skp}; ошибок: {err}")
        st.rerun()


# ---------- UI ----------
_init_state()

# Env banner
try:
    import numpy as _np  # noqa: F401
    np_ok = True
except Exception:
    np_ok = False
pil_ok = Image is not None

with st.container(border=True):
    cols = st.columns(4)
    cols[0].markdown(f"**numpy**: {'✅' if np_ok else '❌'}")
    cols[1].markdown(f"**Pillow**: {'✅' if pil_ok else '❌'}")
    cols[2].markdown(f"**send2trash**: {'✅' if send2trash else '⚠️'}")
    cols[3].markdown(f"**DnD**: {'✅' if sort_items else '⚠️'}")
    if not np_ok or not pil_ok:
        st.error("Отсутствуют обязательные зависимости (numpy/Pillow). Обработка будет недоступна.")

# Network URL selector
lan_candidates = get_lan_candidates()
if st.session_state["lan_ip"] is None:
    st.session_state["lan_ip"] = lan_candidates[0]

st.title("Face Sorter — Мини-проводник (Explorer)")

lan_col1, lan_col2 = st.columns([0.7, 0.3])
with lan_col1:
    ip_sel = st.selectbox("LAN интерфейс", options=lan_candidates, index=max(0, lan_candidates.index(st.session_state["lan_ip"]) if st.session_state["lan_ip"] in lan_candidates else 0))
    st.session_state["lan_ip"] = ip_sel
    net_url = make_network_url(ip_sel)
    st.info(f"Сетевой URL (LAN): {net_url}")
    try:
        st.link_button("Открыть по сети", net_url, use_container_width=True)
    except Exception:
        st.markdown(f"[Открыть по сети]({net_url})")
with lan_col2:
    st.text_input("Network URL", value=net_url, label_visibility="visible")

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
                st.error("Путь не существует или недоступен.")
else:
    curr = Path(st.session_state["current_dir"]).expanduser().resolve()
    parent_root = Path(st.session_state["parent_path"]).expanduser().resolve()
    if not _is_subpath(curr, parent_root):
        st.session_state["current_dir"] = str(parent_root); curr = parent_root

    # Toolbar
    top_cols = st.columns([0.08, 0.12, 0.40, 0.40])
    with top_cols[0]:
        up = curr.parent if curr != parent_root else None
        st.button(
            "⬆️ Вверх",
            key="up",
            disabled=(up is None),
            on_click=(lambda p=str(up): st.session_state.update({"current_dir": p})) if up else None,
            use_container_width=True
        )
    with top_cols[1]:
        if st.button("🔄 Обновить", use_container_width=True):
            st.rerun()
    with top_cols[2]:
        sk = st.selectbox("Сортировка", ["name", "date", "size"], index=["name", "date", "size"].index(st.session_state["sort_key"]))
        st.session_state["sort_key"] = sk
    with top_cols[3]:
        st.session_state["sort_desc"] = st.checkbox("По убыванию", value=st.session_state["sort_desc"]) 

    # Breadcrumbs within root
    labels = [parent_root.name if parent_root.name else parent_root.anchor or "/"]
    targets = [parent_root]
    try:
        rel = curr.resolve().relative_to(parent_root.resolve())
        acc = parent_root
        for part in rel.parts:
            acc = acc / part
            labels.append(part); targets.append(acc)
    except Exception:
        pass
    bc_cols = st.columns(len(labels))
    for i, (lbl, tgt) in enumerate(zip(labels, targets)):
        with bc_cols[i]:
            st.button(lbl or "/", key=f"bc::{i}", use_container_width=True,
                      on_click=(lambda p=str(tgt): st.session_state.update({"current_dir": p})))

    st.markdown("---")

    # Header row styles
    st.markdown('<style>.row{display:grid;grid-template-columns:160px 1fr 110px 170px 120px;gap:8px;align-items:center;padding:6px 8px;border-bottom:1px solid #f1f5f9}.hdr{font-weight:600;color:#334155;border-bottom:1px solid #e2e8f0}.thumbbox{width:150px;height:150px;display:flex;align-items:center;justify-content:center;border:1px solid #e5e7eb;border-radius:10px;overflow:hidden;background:#fff}</style>', unsafe_allow_html=True)
    st.markdown('<div class="row hdr"><div>Превью</div><div>Имя</div><div>Тип</div><div>Изменён</div><div>Размер</div></div>', unsafe_allow_html=True)

    with st.container(height=700):
        # Explorer-like DnD block
        render_explorer_dnd(curr, parent_root)
        st.markdown("---")

        # Explorer grid
        items = list_dir(curr)
        for item in items:
            is_dir = item.is_dir()
            sel_key = _k("sel", item)
            name_btn_key = _k("open", item)
            del_key = _k("del", item)
            ren_key = _k("ren", item)
            ren_input_key = _k("ren_input", item)

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

            # Name + actions
            with c2:
                icon = "📁" if is_dir else "🗎"
                name_cols = st.columns([0.72, 0.10, 0.10, 0.08])
                with name_cols[0]:
                    if is_dir:
                        if st.button(f"{icon} {item.name}", key=name_btn_key, use_container_width=True):
                            st.session_state["current_dir"] = str(item); st.rerun()
                    else:
                        st.write(f"{icon} {item.name}")
                with name_cols[1]:
                    if is_dir:
                        checked = st.checkbox("Выбрать", key=sel_key, value=(str(item) in st.session_state["selected_dirs"]), help="В очередь", label_visibility="collapsed")
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

            # Inline rename
            if st.session_state.get("rename_target") == str(item):
                rc1, rc2, rc3 = st.columns([0.70, 0.15, 0.15])
                with rc1:
                    new_name_val = st.text_input("Новое имя", value=item.name, key=ren_input_key, label_visibility="collapsed")
                with rc2:
                    if st.button("Сохранить", key=f"save::{_k('save', item)}", use_container_width=True):
                        try:
                            new_name = (new_name_val or "").strip()
                            bad = {'/', '\\', ':', '*', '?', '"', '<', '>', '|'}
                            if not new_name or any(ch in new_name for ch in bad) or new_name in {'.', '..'}:
                                raise ValueError("Неверное имя файла/папки.")
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
                    if st.button("Отмена", key=f"cancel::{_k('cancel', item)}", use_container_width=True):
                        st.session_state["rename_target"] = None; st.rerun()

            # Inline delete (safer)
            if st.session_state.get("delete_target") == str(item):
                dc1, dc2, dc3 = st.columns([0.60, 0.20, 0.20])
                with dc1:
                    st.markdown(f"❗ Подтвердите удаление: **{item.name}**")
                    if send2trash is None:
                        st.warning("send2trash не установлен — удаление будет **безвозвратным**. Введите имя для подтверждения.")
                        confirm_text = st.text_input("Введите имя для подтверждения", key=f"confirm_name::{_k('confirm', item)}")
                        st.session_state["confirm_hard_delete"] = (confirm_text == item.name)
                with dc2:
                    can_delete = True if send2trash is not None else bool(st.session_state.get("confirm_hard_delete"))
                    if st.button("Удалить", type="primary", key=f"confirm_del::{_k('cd', item)}", use_container_width=True, disabled=not can_delete):
                        try:
                            if send2trash is not None:
                                send2trash(str(item))
                            else:
                                if is_dir:
                                    shutil.rmtree(item, ignore_errors=True)
                                else:
                                    item.unlink(missing_ok=True)
                            st.session_state["delete_target"] = None; st.rerun()
                        except Exception as e:
                            st.error(f"Ошибка удаления: {e}"); st.session_state["delete_target"] = None
                with dc3:
                    if st.button("Отмена", key=f"cancel_del::{_k('canceld', item)}", use_container_width=True):
                        st.session_state["delete_target"] = None

    st.markdown("---")

    # Footer actions
    colA, colB, colC, colD = st.columns([0.30, 0.30, 0.20, 0.20])
    with colA:
        if st.button("➕ Добавить в очередь", type="secondary", use_container_width=True):
            added = 0
            for d in list(st.session_state["selected_dirs"]):
                if d not in st.session_state["queue"]:
                    st.session_state["queue"].append(d); added += 1
            st.success(f"Добавлено в очередь: {added}")
    with colB:
        if st.button("🧹 Очистить очередь", use_container_width=True):
            st.session_state["queue"] = []; st.session_state["selected_dirs"] = set(); st.info("Очередь очищена.")
    with colC:
        st.write(f"В очереди: {len(st.session_state['queue'])}")
    with colD:
        st.session_state["del_after_copy"] = st.toggle("Удалять исходники", value=st.session_state.get("del_after_copy", False), help="Удалять только файлы, которые были источниками текущей обработки")

    # Process button gated by env
    process_disabled = (not np_ok) or (not pil_ok)
    if st.button("▶️ Обработать", type="primary", use_container_width=True, disabled=process_disabled):
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
            run_audit = {"started": datetime.now().isoformat(timespec="seconds"), "groups": []}

            status = st.status("Идёт обработка…", expanded=True)
            with status:
                prog = st.progress(0, text=f"0/{len(targets)}")
                for k, gdir in enumerate(targets, start=1):
                    st.write(f"Обработка: **{gdir.name}**")
                    try:
                        plan = build_plan(
                            gdir,
                            group_thr=CFG["group_thr"], eps_sim=CFG["eps_sim"], min_samples=CFG["min_samples"],
                            min_face=CFG["min_face"], blur_thr=CFG["blur_thr"], det_size=CFG["det_size"], gpu_id=CFG["gpu_id"],
                            per_person_min_obs=CFG.get("per_person_min_obs"), min_det_score=CFG.get("min_det_score"), min_quality=CFG.get("min_quality"),
                        )
                        cluster_images = plan.get("cluster_images", {}) or {}
                        faces_detections = sum(len(v) for v in cluster_images.values())
                        unique_people_in_run = len(plan.get("eligible_clusters", []))
                        freq = {}
                        for imgs in cluster_images.values():
                            for pth in imgs:
                                freq[pth] = freq.get(pth, 0) + 1
                        joint_images = sum(1 for v in freq.values() if v >= 2)

                        persons_after, processed_sources, audit = match_and_apply(gdir, plan, match_thr=CFG["match_thr"])
                        idx["group_counts"][str(gdir)] = persons_after

                        # Conditional deletion of original sources only
                        if st.session_state.get("del_after_copy", False):
                            protected_roots = {gdir / "__unknown__", gdir / "__group_only__"} | {d for d in gdir.iterdir() if d.is_dir() and d.name.isdigit()}
                            for src in list(processed_sources):
                                try:
                                    if not src.exists() or not src.is_file():
                                        continue
                                    if any(_is_subpath(src, r) for r in protected_roots):
                                        continue
                                    src.unlink(); log(f"Удален оригинал: {src.name}")
                                except Exception as e:
                                    log(f"Ошибка удаления {src.name}: {e}")

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
                        run_audit["groups"].append({
                            "group": str(gdir),
                            "faces_detections": faces_detections,
                            "unique_people": unique_people_in_run,
                            "joint_images": joint_images,
                            "copied": audit.get("copied", {}),
                            "group_only": audit.get("group_only", 0),
                            "unknown": audit.get("unknown", 0)
                        })
                    except Exception as e:
                        st.error(f"Ошибка в {gdir.name}: {e}")
                        st.session_state["proc_logs"].append(f"{gdir.name}: ошибка — {e}")

                    prog.progress(k / len(targets), text=f"{k}/{len(targets)}")

                status.update(label="Готово", state="complete")

            # Aggregate + persist with history
            idx["global_stats"]["images_total"] += tot_total
            idx["global_stats"]["images_unknown_only"] += tot_unknown
            idx["global_stats"]["images_group_only"] += tot_group_only
            run_audit.update({
                "finished": datetime.now().isoformat(timespec="seconds"),
                "tot_faces": tot_faces,
                "tot_unique_people": tot_unique_people,
                "tot_group_only": tot_group_only,
                "tot_joint": tot_joint,
            })
            idx.setdefault("runs", []).append(run_audit)
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
