# app.py
# Windows Explorer-styled UI (Streamlit)
# Focus: 1:1 visual semantics within Streamlit constraints.
import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Set
from datetime import datetime

import streamlit as st

# Optional deps
try:
    import numpy as np  # used by processing tab
except Exception:
    np = None

# ---- Windows folder picker (native dialog when possible) ----
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

# ---- Your processing backend (kept as before) ----
try:
    from core.cluster import build_plan, IMG_EXTS
except Exception:
    # Fallbacks to allow UI to run without backend present
    IMG_EXTS = {".jpg",".jpeg",".png",".webp",".bmp",".tiff"}
    def build_plan(*args, **kwargs):
        return {"eligible_clusters": [], "cluster_centroids": {}, "cluster_images": {}, "group_only_images": [], "unknown_images": [], "stats":{"images_total":0,"images_unknown_only":0,"images_group_only":0}}

# ----------------- Page -----------------
st.set_page_config(page_title="Explorer", page_icon="🗂️", layout="wide")

# ----------------- Styles: Windows 11 look -----------------
st.markdown("""
<style>
:root {
  --win-bg: #ffffff;
  --win-fg: #111827;
  --win-muted: #6b7280;
  --win-border: #e5e7eb;
  --win-hover: #f3f4f6;
  --win-accent: #2563eb;
}
html, body, [data-testid="stAppViewContainer"] {
  background: var(--win-bg);
  color: var(--win-fg);
  font-family: "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, "Noto Sans", "Liberation Sans", sans-serif;
}
/* Title area (fake title bar) */
.win-titlebar{display:flex;align-items:center;gap:8px;padding:8px 10px;border-bottom:1px solid var(--win-border);}
.win-cmdbar{display:flex;align-items:center;gap:8px;padding:6px 10px;border-bottom:1px solid var(--win-border);background:#fafafa;}
.win-btn{border:1px solid var(--win-border);background:#fff;border-radius:8px;padding:6px 10px;cursor:pointer;user-select:none;}
.win-btn:hover{background:var(--win-hover);}
.win-btn[disabled]{opacity:.5;cursor:not-allowed;}
.win-input{border:1px solid var(--win-border);border-radius:8px;padding:6px 10px;background:#fff;}
.win-addr{display:flex;align-items:center;gap:6px;flex:1;}
.win-crumb{border:1px solid transparent;border-radius:8px;padding:4px 8px;}
.win-crumb:hover{background:var(--win-hover);border-color:var(--win-border);}
.win-search{min-width:220px;display:flex;align-items:center;gap:6px;}
.win-view{padding:6px 10px;}
.win-table-head, .win-table-row{display:grid;grid-template-columns: 1.2fr 0.7fr 0.6fr 0.4fr;gap:8px;align-items:center;}
.win-table-head{padding:8px;border-bottom:1px solid var(--win-border);font-weight:600;color:#374151;}
.win-table-row{padding:6px 8px;border-bottom:1px solid #f3f4f6;border-radius:8px;}
.win-table-row:hover{background:var(--win-hover);}
.win-check{transform: scale(1.1);}
.win-name{display:flex;align-items:center;gap:8px;}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid var(--win-border);background:#f8fafc;color:#334155;font-size:12px;}
.statusbar{display:flex;align-items:center;justify-content:space-between;padding:6px 10px;border-top:1px solid var(--win-border);color:#374151;}
.small{color:var(--win-muted);font-size:12px;}
.right{margin-left:auto;}
.icon{width:18px;height:18px;display:inline-block;text-align:center;}
</style>
""", unsafe_allow_html=True)

# ----------------- Config load (read-only) -----------------
def load_config(base: Path) -> Dict:
    cfg_path = base / "config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"group_thr":3,"eps_sim":0.55,"min_samples":2,"min_face":110,"blur_thr":45.0,"det_size":640,"gpu_id":0,"match_thr":0.60}

CFG_BASE = Path(__file__).parent
CFG = load_config(CFG_BASE)

# ----------------- Index persistence -----------------
def load_index(parent: Path) -> Dict:
    p = parent / "global_index.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"group_counts": {}, "global_stats": {"images_total":0,"images_unknown_only":0,"images_group_only":0}, "last_run": None}

def save_index(parent: Path, idx: Dict):
    idx["last_run"] = datetime.now().isoformat(timespec="seconds")
    (parent / "global_index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")

# ----------------- State -----------------
def _init_state():
    st.session_state.setdefault("parent_path", str(Path.home()))
    st.session_state.setdefault("current_dir", st.session_state["parent_path"])
    st.session_state.setdefault("selected", set())
    st.session_state.setdefault("clipboard", {"mode": None, "items": []})  # mode: copy|cut
    st.session_state.setdefault("search", "")
    st.session_state.setdefault("sort", {"col":"name","asc":True})
    st.session_state.setdefault("view", "Details")  # Details | Large icons
    st.session_state.setdefault("history", [st.session_state["parent_path"]])
    st.session_state.setdefault("history_index", 0)
    st.session_state.setdefault("queue", [])  # for processing tab
    st.session_state.setdefault("logs", [])

def log(msg: str):
    st.session_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

_init_state()

# ----------------- Helpers -----------------
def drives() -> List[Path]:
    # Windows drive discovery
    letters = [f"{chr(c)}:\\" for c in range(ord('A'), ord('Z')+1)]
    res = []
    for d in letters:
        try:
            if Path(d).exists():
                res.append(Path(d))
        except Exception:
            pass
    # Fallback to root on non-Windows
    return res or [Path("/")]

def special_folders() -> Dict[str, Path]:
    env = os.environ
    home = Path(env.get("USERPROFILE", str(Path.home())))
    candidates = {
        "Desktop": home / "Desktop",
        "Documents": home / "Documents",
        "Downloads": home / "Downloads",
        "Pictures": home / "Pictures",
        "Music": home / "Music",
        "Videos": home / "Videos",
    }
    return {k:v for k,v in candidates.items() if v.exists()}

def breadcrumb_parts(path: Path) -> List[Path]:
    parts = []
    try:
        p = path.resolve()
    except Exception:
        p = path
    while True:
        parts.insert(0, p)
        if p.parent == p: break
        p = p.parent
    return parts

def list_dir(p: Path) -> List[Path]:
    try:
        items = list(p.iterdir())
    except Exception:
        return []
    # Filter by search
    q = st.session_state["search"].lower().strip()
    if q:
        items = [x for x in items if q in x.name.lower()]
    # Sort
    col = st.session_state["sort"]["col"]
    asc = st.session_state["sort"]["asc"]
    def key_fn(x: Path):
        if col == "name":
            return (x.is_file(), x.name.lower())
        if col == "date":
            try: return x.stat().st_mtime
            except: return 0
        if col == "type":
            return ("" if x.is_dir() else x.suffix.lower())
        if col == "size":
            try: return 0 if x.is_dir() else x.stat().st_size
            except: return 0
        return x.name.lower()
    items.sort(key=key_fn, reverse=not asc)
    # Folders first like Windows (unless sorting by size)
    if col != "size":
        items.sort(key=lambda x: 1 if x.is_file() else 0)
    return items

def human_size(num: int) -> str:
    step = 1024.0
    units = ["Б","КБ","МБ","ГБ","ТБ"]
    for u in units:
        if num < step or u == units[-1]:
            return f"{num:.0f} {u}" if u=="Б" else f"{num/1024:.1f} {u}"
        num /= step

def unique_name(dst_dir: Path, base: str) -> Path:
    target = dst_dir / base
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    i = 1
    while True:
        cand = dst_dir / f"{stem} ({i}){suffix}"
        if not cand.exists():
            return cand
        i += 1

def copy_any(src: Path, dst_dir: Path):
    dst = unique_name(dst_dir, src.name)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return dst

def move_any(src: Path, dst_dir: Path):
    dst = unique_name(dst_dir, src.name)
    shutil.move(str(src), str(dst))
    return dst

def go_to(p: Path, push_history: bool=True):
    p = p.expanduser().resolve()
    if not p.exists() or not p.is_dir():
        return
    st.session_state["current_dir"] = str(p)
    st.session_state["selected"] = set()
    if push_history:
        hist = st.session_state["history"]
        idx = st.session_state["history_index"]
        hist[:] = hist[:idx+1]
        hist.append(str(p))
        st.session_state["history_index"] = len(hist)-1

def nav_back():
    if st.session_state["history_index"]>0:
        st.session_state["history_index"]-=1
        st.session_state["current_dir"]=st.session_state["history"][st.session_state["history_index"]]
        st.session_state["selected"]=set()

def nav_forward():
    if st.session_state["history_index"]<len(st.session_state["history"])-1:
        st.session_state["history_index"]+=1
        st.session_state["current_dir"]=st.session_state["history"][st.session_state["history_index"]]
        st.session_state["selected"]=set()

# ----------------- Title bar (Back/Forward/Up, Address, Search) -----------------
with st.container():
    st.markdown('<div class="win-titlebar">', unsafe_allow_html=True)
    col_nav1, col_nav2, col_nav3, col_addr, col_search = st.columns([0.04,0.04,0.04,0.58,0.30])
    with col_nav1:
        st.button("⬅", key="back", help="Назад", on_click=nav_back, use_container_width=True, disabled=st.session_state["history_index"]<=0)
    with col_nav2:
        st.button("➡", key="forward", help="Вперёд", on_click=nav_forward, use_container_width=True, disabled=st.session_state["history_index"]>=len(st.session_state["history"])-1)
    with col_nav3:
        curr = Path(st.session_state["current_dir"])
        up = curr.parent if curr != curr.anchor else None
        st.button("⬆", key="up", help="Вверх", on_click=(lambda: go_to(up) if up else None), use_container_width=True, disabled=up is None)
    with col_addr:
        # Breadcrumbs
        parts = breadcrumb_parts(Path(st.session_state["current_dir"]))
        bc_cols = st.columns(max(1, len(parts)))
        for i, part in enumerate(parts):
            with bc_cols[i]:
                st.button(part.name if part.name else part.as_posix(), key=f"bc::{i}::{part}", on_click=lambda p=part: go_to(p), use_container_width=True)
    with col_search:
        st.text_input("Поиск", value=st.session_state["search"], key="search", label_visibility="collapsed", placeholder="Поиск в текущей папке", help="Введите часть имени")

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- Command bar -----------------
with st.container():
    st.markdown('<div class="win-cmdbar">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6, space, c7, c8 = st.columns([0.09,0.09,0.09,0.09,0.11,0.10,0.32,0.05,0.06])
    cd = Path(st.session_state["current_dir"])

    with c1:
        if st.button("🗂️ Новая", help="Новая папка", use_container_width=True):
            new = unique_name(cd, "Новая папка")
            new.mkdir(parents=True, exist_ok=False)
    with c2:
        disabled = not st.session_state["selected"]
        if st.button("✂️ Вырезать", disabled=disabled, use_container_width=True):
            st.session_state["clipboard"]={"mode":"cut","items":list(st.session_state["selected"])}
    with c3:
        disabled = not st.session_state["selected"]
        if st.button("📄 Копировать", disabled=disabled, use_container_width=True):
            st.session_state["clipboard"]={"mode":"copy","items":list(st.session_state["selected"])}
    with c4:
        clip = st.session_state["clipboard"]
        if st.button("📋 Вставить", disabled=not clip["items"], use_container_width=True):
            for s in clip["items"]:
                src = Path(s)
                try:
                    if clip["mode"]=="copy":
                        copy_any(src, cd)
                    elif clip["mode"]=="cut":
                        move_any(src, cd)
                except Exception as e:
                    st.warning(f"Ошибка вставки {src.name}: {e}")
            if clip["mode"]=="cut":
                st.session_state["clipboard"]={"mode":None,"items":[]}
    with c5:
        if st.button("✏️ Переименовать", disabled=len(st.session_state["selected"])!=1, use_container_width=True):
            st.session_state["rename_target"]=list(st.session_state["selected"])[0]
    with c6:
        if st.button("🗑️ Удалить", disabled=not st.session_state["selected"], use_container_width=True):
            for s in list(st.session_state["selected"]):
                p = Path(s)
                try:
                    if p.is_dir(): shutil.rmtree(p, ignore_errors=True)
                    else: p.unlink(missing_ok=True)
                except Exception as e:
                    st.error(f"Ошибка удаления {p.name}: {e}")
            st.session_state["selected"]=set()

    # Right side: sort & view
    with c7:
        s1, s2 = st.columns(2, gap="small")
        with s1:
            sort_col = st.selectbox("Сортировка", ["name","date","type","size"], index=["name","date","type","size"].index(st.session_state["sort"]["col"]), label_visibility="collapsed")
        with s2:
            sort_dir = st.selectbox("Порядок", ["asc","desc"], index=0 if st.session_state["sort"]["asc"] else 1, label_visibility="collapsed")
        st.session_state["sort"]={"col":sort_col, "asc": (sort_dir=="asc")}
    with c8:
        st.session_state["view"]=st.selectbox("Вид", ["Details","Large icons"], index=0 if st.session_state["view"]=="Details" else 1, label_visibility="collapsed")

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- Left Navigation -----------------
nav_col, main_col = st.columns([0.23, 0.77], gap="small")

with nav_col:
    st.markdown("**Быстрый доступ**")
    for name, path in special_folders().items():
        if st.button(f"📁 {name}", key=f"q::{name}", use_container_width=True):
            go_to(path)
    st.markdown("---")
    st.markdown("**Этот компьютер**")
    for d in drives():
        label = f"{d}"
        if st.button(f"💽 {label}", key=f"drv::{label}", use_container_width=True):
            go_to(d)

# ----------------- Main View -----------------
with main_col:
    # Address bar quick path input (optional, like Explorer)
    path_str = st.text_input("Адрес", value=st.session_state["current_dir"], label_visibility="collapsed")
    if path_str and path_str != st.session_state["current_dir"]:
        p = Path(path_str)
        if p.exists() and p.is_dir():
            go_to(p)

    items = list_dir(Path(st.session_state["current_dir"]))

    # Select all
    sa_l, sa_r = st.columns([0.2,0.8])
    with sa_l:
        if st.checkbox("Выделить всё", value=(len(st.session_state["selected"])==len(items) and len(items)>0)):
            st.session_state["selected"] = set(str(x) for x in items)

    # Views
    if st.session_state["view"] == "Details":
        head = st.container()
        with head:
            st.markdown('<div class="win-table-head"><div>Имя</div><div>Дата изменения</div><div>Тип</div><div>Размер</div></div>', unsafe_allow_html=True)
        for it in items:
            cols = st.columns([0.05,0.60,0.22,0.08,0.05])
            key = f"sel::{it}"
            with cols[0]:
                checked = st.checkbox("", key=key, value=(str(it) in st.session_state["selected"]))
                if checked:
                    st.session_state["selected"].add(str(it))
                else:
                    st.session_state["selected"].discard(str(it))
            with cols[1]:
                icon = "📁" if it.is_dir() else "🗎"
                if it.is_dir():
                    if st.button(f"{icon} {it.name}", key=f"open::{it}", use_container_width=True):
                        go_to(it)
                else:
                    st.write(f"{icon} {it.name}")
            with cols[2]:
                try:
                    ts = datetime.fromtimestamp(it.stat().st_mtime).strftime("%d.%m.%Y %H:%M")
                except Exception:
                    ts = "—"
                st.write(ts)
            with cols[3]:
                if it.is_dir():
                    st.write("Папка")
                else:
                    st.write((it.suffix[1:].upper() if it.suffix else "Файл"))
            with cols[4]:
                if it.is_dir():
                    st.write("—")
                else:
                    try:
                        st.write(human_size(it.stat().st_size))
                    except Exception:
                        st.write("—")
    else:
        # Large icons grid (4 columns)
        grid = st.container()
        cols = st.columns(6)
        i = 0
        for it in items:
            col = cols[i % len(cols)]
            with col:
                key = f"sel_grid::{it}"
                checked = st.checkbox("", key=key, value=(str(it) in st.session_state["selected"]))
                if checked: st.session_state["selected"].add(str(it))
                else: st.session_state["selected"].discard(str(it))
                if it.is_dir():
                    if st.button(f"📁\n{it.name}", key=f"open_grid::{it}"):
                        go_to(it)
                else:
                    if it.suffix.lower() in IMG_EXTS:
                        try:
                            st.image(str(it), use_container_width=True, caption=it.name)
                        except Exception:
                            st.write(f"🗎 {it.name}")
                    else:
                        st.write(f"🗎 {it.name}")
            i += 1

    # Rename inline (single selection)
    sel = list(st.session_state["selected"])
    if len(sel) == 1:
        target = Path(sel[0])
        st.markdown("---")
        st.write("Переименовать:")
        new_name = st.text_input("Новое имя", value=target.name, key=f"rename::{target}", label_visibility="collapsed")
        c_ren1, c_ren2 = st.columns([0.1,0.9])
        with c_ren1:
            if st.button("OK", key="rename_ok"):
                try:
                    dst = target.parent / new_name
                    target.rename(dst)
                    st.session_state["selected"] = {str(dst)}
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Ошибка: {e}")
        with c_ren2:
            st.caption("Введите новое имя и нажмите OK.")

    # Status bar
    total = len(items)
    selected = len(st.session_state["selected"])
    st.markdown(f'<div class="statusbar"><div>{total} элементов • выбрано: {selected}</div><div class="small right">{st.session_state["current_dir"]}</div></div>', unsafe_allow_html=True)

# ----------------- Processing Tab (unchanged core) -----------------
with st.expander("⚙️ Обработка (Face Sorter) — сверните, если не нужно", expanded=False):
    parent = Path(st.session_state["parent_path"]).expanduser().resolve()
    idx = load_index(parent)

    st.caption("Очередь обработки")
    # Fill queue from child folders of current_dir to keep it intuitive
    children = [p for p in Path(st.session_state["current_dir"]).iterdir() if p.is_dir()]
    names = [c.name for c in children]
    selected_groups = st.multiselect("Добавить в очередь", names, default=[])
    if st.button("➕ В очередь"):
        for n in selected_groups:
            p = Path(st.session_state["current_dir"]) / n
            if str(p) not in st.session_state["queue"]:
                st.session_state["queue"].append(str(p))

    if st.session_state["queue"]:
        st.write("\\n".join(f"{i+1}. {Path(g).name}" for i,g in enumerate(st.session_state["queue"])))
        cqa, cqb = st.columns(2)
        with cqa:
            if st.button("🧹 Очистить"):
                st.session_state["queue"] = []
        with cqb:
            if st.button("↩️ Удалить последний") and st.session_state["queue"]:
                st.session_state["queue"].pop()

    st.markdown("---")
    if st.button("▶️ Старт обработки", type="primary"):
        if st.session_state.get("queue"):
            targets = [Path(g) for g in st.session_state["queue"]]
        else:
            targets = children

        if not targets:
            st.warning("Нет целей для обработки.")
        else:
            tot_total = tot_unknown = tot_group_only = 0
            progress = st.progress(0, text="Инициализация…")
            status = st.status("Запуск обработки", expanded=True)
            with status:
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
                        persons_after, processed_images = match_and_apply(gdir, plan, match_thr=CFG["match_thr"])
                        idx["group_counts"][str(gdir)] = persons_after
                        cleanup_processed_images(gdir, processed_images)
                        tot_total += plan["stats"]["images_total"]
                        tot_unknown += plan["stats"]["images_unknown_only"]
                        tot_group_only += plan["stats"]["images_group_only"]
                        st.success(f"Готово: {gdir.name}")
                    except Exception as e:
                        st.error(f"Ошибка: {e}")
                    progress.progress(k/len(targets), text=f"{k}/{len(targets)}")

            idx["global_stats"]["images_total"] += tot_total
            idx["global_stats"]["images_unknown_only"] += tot_unknown
            idx["global_stats"]["images_group_only"] += tot_group_only
            save_index(parent, idx)
            st.session_state["queue"] = []
            st.toast("Обработка завершена", icon="✅")

# ---- Processing helpers reused ----
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_person_index(group_dir: Path) -> Dict:
    p = group_dir / "person_index.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"persons": []}

def save_person_index(group_dir: Path, data: Dict):
    (group_dir / "person_index.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def cleanup_processed_images(group_dir: Path, processed_images: Set[Path]):
    for img_path in processed_images:
        try:
            if img_path.exists() and img_path.is_file():
                img_path.unlink()
        except Exception:
            pass

def match_and_apply(group_dir: Path, plan: Dict, match_thr: float) -> Tuple[int, Set[Path]]:
    import numpy as np
    person_idx = load_person_index(group_dir)
    centroids = plan.get("cluster_centroids", {})
    assigned = {}
    centroids_norm = {
        int(cid): (np.array(vec, dtype=np.float32) / (np.linalg.norm(vec)+1e-12))
        for cid, vec in centroids.items()
    }
    persons = person_idx.get("persons", [])
    if persons:
        P = np.stack([np.array(p["proto"], dtype=np.float32) for p in persons], axis=0)
        P = P / (np.linalg.norm(P, axis=1, keepdims=True)+1e-12)
        nums = [int(p["number"]) for p in persons]
        for cid, c in centroids_norm.items():
            sims = P @ c
            j = int(np.argmax(sims))
            if float(sims[j]) >= match_thr:
                assigned[int(cid)] = int(nums[j])
    existing_nums = sorted([int(p.name) for p in group_dir.iterdir() if p.is_dir() and p.name.isdigit()])
    cur_max = existing_nums[-1] if existing_nums else 0
    new_nums = {}
    for cid in plan["eligible_clusters"]:
        if cid in assigned: continue
        cur_max += 1
        new_nums[cid] = cur_max
        person_idx["persons"].append({"number": cur_max, "proto": centroids_norm[cid].tolist()})
    for cid, num in {**assigned, **new_nums}.items():
        ensure_dir(group_dir / str(num))
    copied: Set[Tuple[int, Path]] = set()
    cluster_images = {}
    for k, v in plan.get("cluster_images", {}).items():
        try: cluster_images[int(k)] = v
        except: pass
    for cid in plan["eligible_clusters"]:
        num = assigned.get(cid, new_nums.get(cid))
        if num is None: continue
        for img in cluster_images.get(cid, []):
            p = Path(img)
            key = (num, p)
            if key in copied: continue
            dst = group_dir / str(num) / p.name
            if not dst.exists():
                try: shutil.copy2(p, dst)
                except: pass
            copied.add(key)
    go = plan.get("group_only_images", [])
    if go:
        ensure_dir(group_dir / "__group_only__")
        for img in go:
            p = Path(img); dst = group_dir / "__group_only__" / p.name
            if not dst.exists():
                try: shutil.copy2(p, dst)
                except: pass
    un = plan.get("unknown_images", [])
    if un:
        ensure_dir(group_dir / "__unknown__")
        for img in un:
            p = Path(img); dst = group_dir / "__unknown__" / p.name
            if not dst.exists():
                try: shutil.copy2(p, dst)
                except: pass
    save_person_index(group_dir, person_idx)
    processed_images: Set[Path] = set()
    for cid in plan["eligible_clusters"]:
        for img in cluster_images.get(cid, []):
            processed_images.add(Path(img))
    for img in plan.get("group_only_images", []):
        processed_images.add(Path(img))
    for img in plan.get("unknown_images", []):
        processed_images.add(Path(img))
    all_images_in_group = set()
    for ext in IMG_EXTS:
        for img_file in group_dir.rglob(f"*{ext}"):
            if img_file.is_file():
                all_images_in_group.add(img_file)
    processed_images = processed_images.intersection(all_images_in_group)
    return len(person_idx["persons"]), processed_images
