# app.py — patched
# Minimal Explorer (revised++++): LAN URL, cached thumbnails (bounded), safe delete/copy/move,
# atomic JSON writes, UI fixes, root-safe "Up", progress bar inside status,
# scrollable explorer (700px), Drag&Drop move panel, configurable originals cleanup,
# robust person_index compatibility, case-insensitive image scan, breadcrumbs truncation,
# safe rename, basic filters, run report + downloads, config validation warnings.

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

# ---- Native folder picker (Windows) ----
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

st.set_page_config(page_title="Face Sorter — Мини-проводник", layout="wide")

# ---- Minimal CSS ----
st.markdown("""
<style>
  .row { display:grid; grid-template-columns: 160px 1fr 110px 170px 120px; gap:8px; align-items:center; padding:6px 8px; border-bottom:1px solid #f1f5f9;}
  .row:hover { background:#f8fafc; }
  .hdr { font-weight:600; color:#334155; border-bottom:1px solid #e2e8f0; }
  .name { display:flex; align-items:center; gap:8px; }
  .iconbtn { border:1px solid #e5e7eb; background:#fff; border-radius:8px; padding:4px 6px; cursor:pointer; }
  .iconbtn:hover { background:#f1f5f9; }
  .muted { color:#64748b; font-size:12px; }
  .addr { display:flex; gap:8px; align-items:center; }
  .crumb { border:1px solid transparent; padding:4px 8px; border-radius:8px; }
  .crumb:hover { background:#f1f5f9; border-color:#e5e7eb; }
  .thumbbox { width:150px; height:150px; display:flex; align-items:center; justify-content:center; border:1px solid #e5e7eb; border-radius:10px; overflow:hidden; background:#fff; }
</style>
""", unsafe_allow_html=True)

# ---- Config (read-only) ----
ALLOWED_CFG_KEYS = {
    "group_thr", "eps_sim", "min_samples", "min_face", "blur_thr", "det_size",
    "gpu_id", "match_thr", "top2_margin", "per_person_min_obs", "min_det_score", "min_quality",
    "delete_originals"
}

def _clamp(v, lo, hi, default):
    try:
        v = type(default)(v)
        return max(lo, min(hi, v))
    except Exception:
        return default

def load_config(base: Path) -> Dict:
    p = base / "config.json"
    defaults = {
        "group_thr": 3,
        "eps_sim": 0.55,
        "min_samples": 2,
        "min_face": 110,
        "blur_thr": 45.0,
        "det_size": 640,
        "gpu_id": 0,
        "match_thr": 0.44,  # согласовано с клампом
        "top2_margin": 0.08,
        "per_person_min_obs": 10,
        "min_det_score": 0.50,
        "min_quality": 0.50,
        "delete_originals": False,
    }
    unknown_keys = []
    if p.exists():
        try:
            user_cfg = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(user_cfg, dict):
                for k, v in user_cfg.items():
                    if k not in ALLOWED_CFG_KEYS:
                        unknown_keys.append(k)
                    defaults[k] = v
        except Exception:
            pass

    defaults["eps_sim"] = _clamp(defaults["eps_sim"], 0.0, 1.0, 0.55)
    defaults["match_thr"] = _clamp(defaults["match_thr"], 0.0, 1.0, 0.44)
    defaults["top2_margin"] = _clamp(defaults["top2_margin"], 0.0, 1.0, 0.08)
    defaults["min_face"] = _clamp(defaults["min_face"], 0, 10000, 110)
    defaults["det_size"] = _clamp(defaults["det_size"], 64, 4096, 640)

    return defaults, unknown_keys

CFG_BASE = Path(__file__).parent
CFG, UNKNOWN_CFG = load_config(CFG_BASE)

# ---- Persistence for global stats ----
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
    return {
        "group_counts": {},
        "global_stats": {"images_total": 0, "images_unknown_only": 0, "images_group_only": 0},
        "last_run": None
    }

def _atomic_write(path: Path, text: str):
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

def save_index(parent: Path, idx: Dict):
    ensure_dir(parent)
    idx["last_run"] = datetime.now().isoformat(timespec="seconds")
    _atomic_write(parent / "global_index.json", json.dumps(idx, ensure_ascii=False, indent=2))

# ---- Network URL helpers ----
def get_lan_ip() -> str:
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
    ip = get_lan_ip()
    port = st.get_option("server.port") or 8501
    base = st.get_option("server.baseUrlPath") or ""
    base = ("/" + base.strip("/")) if base else ""
    return f"http://{ip}:{port}{base}"

# ---- Helpers ----
def _init_state():
    st.session_state.setdefault("parent_path", None)
    st.session_state.setdefault("current_dir", None)
    st.session_state.setdefault("selected_dirs", set())
    st.session_state.setdefault("rename_target", None)
    st.session_state.setdefault("queue", [])
    st.session_state.setdefault("logs", [])
    st.session_state.setdefault("proc_logs", [])
    st.session_state.setdefault("delete_target", None)
    st.session_state.setdefault("delete_originals", bool(CFG.get("delete_originals", False)))
    st.session_state.setdefault("view_filter", "Все")

def log(msg: str):
    st.session_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def list_dir(p: Path) -> List[Path]:
    try:
        items = list(p.iterdir())
    except Exception:
        return []
    items.sort(key=lambda x: (x.is_file(), x.name.lower()))
    return items

def human_size(n: int) -> str:
    units = ["Б", "КБ", "МБ", "ГБ", "ТБ", "ПБ"]
    size = float(n)
    for i, u in enumerate(units):
        if size < 1024 or i == len(units) - 1:
            return f"{size:.0f} {u}" if u == "Б" else f"{size:.1f} {u}"
        size /= 1024.0

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
        if "number" not in person:
            person["number"] = -1  # маркер для пропуска
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
    _atomic_write(group_dir / "person_index.json", json.dumps(data, ensure_ascii=False, indent=2))

def _is_subpath(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve()); return True
    except Exception:
        return False

def cleanup_processed_images(group_dir: Path, processed_images: Set[Path], *, delete_originals: bool = False):
    """Удаляем только из корня группы и только если включено delete_originals."""
    if not delete_originals:
        return
    for img_path in list(processed_images):
        try:
            if img_path.parent.resolve() != group_dir.resolve():
                continue
            if img_path.exists() and img_path.is_file():
                img_path.unlink()
                log(f"Удален оригинал: {img_path.name}")
        except Exception as e:
            log(f"Ошибка удаления {img_path.name}: {e}")

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
        keep = [int(np.argmax(np.linalg.norm(X - X.mean(0), axis=1)))]
        while len(keep) < k_max:
            dists = np.min((1.0 - X @ X[keep].T), axis=1)
            cand = int(np.argmax(dists))
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

def match_and_apply(group_dir: Path, plan: Dict, match_thr: float) -> Tuple[int, Set[Path]]:
    import numpy as np

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
        try:
            num = int(p["number"])
        except Exception:
            continue
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
        P = np.stack(proto_list, axis=0)
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
            sims = (P @ c.astype(np.float32))

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
                    try:
                        if int(p["number"]) == int(best_num):
                            _update_person_proto(p, c)
                            break
                    except Exception:
                        continue
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

    # Регистр-инвариантный сбор только файлов-изображений в группе
    all_in_group = {f for f in group_dir.rglob("*") if f.is_file() and f.suffix.lower() in IMG_EXTS}
    processed_images = processed_images.intersection(all_in_group)

    return len(persons), processed_images

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

# bounded cache to avoid OOM on huge dirs
@st.cache_data(show_spinner=False, max_entries=5000, ttl=3600)
def get_thumb_bytes(path_str: str, size: int, mtime: float):
    if Image is None:
        return None
    im = make_square_thumb(Path(path_str), size)
    if im is None:
        return None
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

# ---- UI State ----
_init_state()

# ---- Step 1: Pick Folder ----
st.title("Face Sorter — Мини-проводник")

# LAN URL
net_url = get_network_url()
st.info(f"Сетевой URL (LAN): {net_url}")
base_path_hint = st.get_option("server.baseUrlPath") or ""
if base_path_hint:
    st.caption("Запущено за reverse proxy: baseUrlPath активен. Проверьте доступность по внешнему пути.")
try:
    st.link_button("Открыть по сети", net_url, use_container_width=True)
except Exception:
    st.markdown(f"[Открыть по сети]({net_url})")
st.text_input("Network URL", value=net_url, label_visibility="collapsed")

# Warn about unknown config keys
if UNKNOWN_CFG:
    st.warning("Найдены неизвестные ключи в config.json: " + ", ".join(sorted(UNKNOWN_CFG)))

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

    # Top bar with Up/Refresh/Breadcrumbs (truncated)
    top_cols = st.columns([0.08, 0.12, 0.80])
    with top_cols[0]:
        up = None if curr == Path(curr.anchor) else curr.parent
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
        crumbs = list(curr.parts)
        MAX_CRUMBS = 8
        def _accum_path(parts):
            acc = Path(parts[0])
            out = [str(acc)]
            for prt in parts[1:]:
                acc = acc / prt
                out.append(str(acc))
            return out
        accum_all = _accum_path(crumbs)
        shown_parts = crumbs
        shown_paths = accum_all
        if len(crumbs) > MAX_CRUMBS:
            shown_parts = crumbs[:2] + ("…",) + crumbs[-(MAX_CRUMBS-3):]
            # map to paths; ellipsis gets None
            shown_paths = accum_all[:2] + [None] + accum_all[-(MAX_CRUMBS-3):]
        bc_cols = st.columns(len(shown_parts))
        for i, (part, pth) in enumerate(zip(shown_parts, shown_paths)):
            with bc_cols[i]:
                if pth is None:
                    st.button("…", disabled=True, use_container_width=True, key=f"bc_dots::{i}")
                else:
                    st.button(
                        part or "/",
                        key=f"bc::{i}",
                        use_container_width=True,
                        on_click=lambda p=pth: st.session_state.update({"current_dir": p})
                    )

    st.markdown("---")

    # View filters
    fcols = st.columns([0.25, 0.75])
    with fcols[0]:
        st.session_state["view_filter"] = st.selectbox("Показывать", ["Все", "Только папки", "Только изображения"], index=["Все", "Только папки", "Только изображения"].index(st.session_state["view_filter"]))
    with fcols[1]:
        st.session_state["delete_originals"] = st.checkbox("Удалять оригиналы после копирования (корень группы)", value=st.session_state["delete_originals"]) 

    # Header row
    st.markdown('<div class="row hdr"><div>Превью</div><div>Имя</div><div>Тип</div><div>Изменён</div><div>Размер</div></div>', unsafe_allow_html=True)

    # === SCROLLABLE explorer (700 px) ===
    with st.container(height=700):
        items = list_dir(curr)
        # apply filter
        vf = st.session_state["view_filter"]
        if vf == "Только папки":
            items = [i for i in items if i.is_dir()]
        elif vf == "Только изображения":
            items = [i for i in items if i.is_file() and i.suffix.lower() in IMG_EXTS]

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
                        # FIX: непустой лейбл с label_visibility="collapsed"
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
                            candidate = new_name.strip()
                            if not candidate:
                                st.error("Имя не может быть пустым.")
                            else:
                                invalid = set('<>:"/\\|?*')
                                if any(ch in invalid for ch in candidate):
                                    st.error("Имя содержит недопустимые символы.")
                                else:
                                    new_path = item.parent / candidate
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
                            log(f"Ошибка удаления {item}: {e}")
                            # оставляем delete_target для повторной попытки/отмены
                with dc3:
                    if st.button("Отмена", key=f"cancel_del::{item}", use_container_width=True):
                        st.session_state["delete_target"] = None

    st.markdown("---")

    # === Drag & Drop MOVE panel (визуал проводника не меняем) ===
    with st.expander("Переместить файлы (Drag & Drop)", expanded=False):
        if sort_items is None:
            st.info("Для DnD установите пакет: pip install streamlit-sortables")
        else:
            # Источники: файлы текущей папки; Приёмники: подпапки текущей папки
            files_in_curr = [str(p) for p in curr.iterdir() if p.is_file()]
            subfolders = [p for p in curr.iterdir() if p.is_dir()]

            containers = [{"header": "Файлы (текущая папка)", "items": files_in_curr}]
            for f in subfolders:
                containers.append({"header": f.name, "items": []})

            result = sort_items(containers, multi_containers=True)

            # Применяем переносы по кнопке
            if st.button("Применить переносы", use_container_width=True):
                if not result:
                    st.warning("Нет изменений для переноса.")
                else:
                    # result — список словарей с обновлёнными 'items'
                    moves = []
                    # Индекс 0 — исходная корзина файлов
                    header_to_dir = {f.name: f for f in subfolders}
                    for i, cont in enumerate(result):
                        if i == 0:
                            continue  # пропускаем исходный контейнер
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

    # Process
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

            status = st.status("Идёт обработка…", expanded=True)
            with status:
                # прогресс-бар внутри status
                prog = st.progress(0, text=f"0/{len(targets)}")
                for k, gdir in enumerate(targets, start=1):
                    st.write(f"Обработка: **{gdir.name}**")
                    try:
                        plan = build_plan(
                            gdir,
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
                        cleanup_processed_images(gdir, processed_images, delete_originals=st.session_state.get("delete_originals", False))

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

            # Run report + downloads
            st.success("Обработка завершена.")
            st.markdown("**Сводка за прогон:**")
            st.write(f"- Людей на фото (детекции): **{tot_faces}**")
            st.write(f"- Уникальных людей (кластера): **{tot_unique_people}**")
            st.write(f"- Общих фото (group_only): **{tot_group_only}**")
            st.write(f"- Совместных фото (>1 человек): **{tot_joint}**")

            report = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "tot_faces": tot_faces,
                "tot_unique_people": tot_unique_people,
                "tot_group_only": tot_group_only,
                "tot_joint": tot_joint,
                "targets": [str(t) for t in targets],
                "delete_originals": bool(st.session_state.get("delete_originals", False)),
            }
            try:
                _atomic_write(parent / "last_run_report.json", json.dumps(report, ensure_ascii=False, indent=2))
            except Exception:
                pass
            st.download_button("Скачать отчёт JSON", data=json.dumps(report, ensure_ascii=False, indent=2), file_name="run_report.json")

            st.markdown("**Детальные логи по группам:**")
            log_text = "\n".join(st.session_state.get("proc_logs", []))
            st.text_area("Логи", value=log_text, height=220)
            st.download_button("Скачать логи", data=log_text, file_name="proc_logs.txt")
