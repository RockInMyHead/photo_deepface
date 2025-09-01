# app.py
import json
import shutil
from pathlib import Path
from typing import List, Dict
from datetime import datetime

import numpy as np
import streamlit as st

# ---- real Windows folder picker ----
def pick_folder_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder = filedialog.askdirectory(title="Выберите родительскую папку")
        root.destroy()
        return folder
    except Exception:
        return None

from core.cluster import build_plan, IMG_EXTS

st.set_page_config(page_title="Face Sorter", layout="wide")

# ----------------- Config -----------------
def load_config(base: Path) -> Dict:
    cfg_path = base / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    return {"group_thr":3,"eps_sim":0.55,"min_samples":2,"min_face":110,"blur_thr":45.0,"det_size":640,"gpu_id":0,"match_thr":0.60}

CFG_BASE = Path(__file__).parent
CFG = load_config(CFG_BASE)

# ----------------- Persistence -----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_index(parent: Path) -> Dict:
    p = parent / "global_index.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"group_order": [], "group_counts": {}, "global_stats": {"images_total":0,"images_unknown_only":0,"images_group_only":0}, "last_run": None}

def save_index(parent: Path, idx: Dict):
    idx["last_run"] = datetime.now().isoformat(timespec="seconds")
    (parent / "global_index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")

# ----- identity persistence per group -----
def load_person_index(group_dir: Path) -> Dict:
    p = group_dir / "person_index.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"persons": []}

def save_person_index(group_dir: Path, data: Dict):
    (group_dir / "person_index.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# ----- cleanup function -----
def cleanup_processed_images(group_dir: Path, processed_images: set):
    """Remove original image files after they have been processed and copied to group folders."""
    for img_path in processed_images:
        try:
            if img_path.exists() and img_path.is_file():
                img_path.unlink()
                st.session_state.setdefault("logs", []).append(f"[{datetime.now().strftime('%H:%M:%S')}] Удален оригинал: {img_path.name}")
        except Exception as e:
            st.session_state.setdefault("logs", []).append(f"[{datetime.now().strftime('%H:%M:%S')}] Ошибка удаления {img_path.name}: {e}")

# ----- core operations -----
def match_and_apply(group_dir: Path, plan: Dict, match_thr: float) -> tuple[int, set]:
    from math import isfinite
    import numpy as np

    person_idx = load_person_index(group_dir)
    centroids = plan.get("cluster_centroids", {})
    assigned = {}
    centroids_norm = {int(cid): (np.array(vec, dtype=np.float32) / (np.linalg.norm(vec)+1e-12)) for cid, vec in centroids.items()}
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
        if cid in assigned:
            continue
        cur_max += 1
        new_nums[cid] = cur_max
        person_idx["persons"].append({"number": cur_max, "proto": centroids_norm[cid].tolist()})

    for cid, num in {**assigned, **new_nums}.items():
        ensure_dir(group_dir / str(num))

    copied = set()
    cluster_images = {}
    for k,v in plan.get("cluster_images", {}).items():
        try:
            cluster_images[int(k)] = v
        except Exception:
            pass

    for cid in plan["eligible_clusters"]:
        num = assigned.get(cid, new_nums.get(cid))
        if num is None:
            continue
        for img in cluster_images.get(cid, []):
            p = Path(img)
            key = (num, p)
            if key in copied:
                continue
            dst = group_dir / str(num) / p.name
            if not dst.exists():
                try:
                    shutil.copy2(p, dst)
                except Exception:
                    pass
            copied.add(key)

    go = plan.get("group_only_images", [])
    if go:
        ensure_dir(group_dir / "__group_only__")
        for img in go:
            p = Path(img)
            dst = group_dir / "__group_only__" / p.name
            if not dst.exists():
                try:
                    shutil.copy2(p, dst)
                except Exception:
                    pass
    un = plan.get("unknown_images", [])
    if un:
        ensure_dir(group_dir / "__unknown__")
        for img in un:
            p = Path(img)
            dst = group_dir / "__unknown__" / p.name
            if not dst.exists():
                try:
                    shutil.copy2(p, dst)
                except Exception:
                    pass

    save_person_index(group_dir, person_idx)

    # Collect all processed images for cleanup
    processed_images = set()
    for cid in plan["eligible_clusters"]:
        for img in cluster_images.get(cid, []):
            processed_images.add(Path(img))

    for img in plan.get("group_only_images", []):
        processed_images.add(Path(img))

    for img in plan.get("unknown_images", []):
        processed_images.add(Path(img))

    # Also collect any other image files that were scanned but not categorized
    from core.cluster import IMG_EXTS
    all_images_in_group = set()
    for ext in IMG_EXTS:
        for img_file in group_dir.rglob(f"*{ext}"):
            if img_file.is_file():
                all_images_in_group.add(img_file)

    # Remove images that are not in the processed set (they might be in subdirectories we don't want to touch)
    processed_images = processed_images.intersection(all_images_in_group)

    return len(person_idx["persons"]), processed_images

# ----- tree rendering -----
def list_children(p: Path):
    try:
        return sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except Exception:
        return []

def render_node(p: Path, depth: int = 0):
    pad = "&nbsp;" * (depth * 4)
    is_dir = p.is_dir()
    if is_dir:
        icon = "📁"
        st.markdown(pad + icon + " " + p.name, unsafe_allow_html=True)
    else:
        # If image, show thumbnail
        if p.suffix.lower() in IMG_EXTS:
            st.image(str(p), width=80, caption=p.name)
        else:
            st.markdown(pad + "🗎 " + p.name, unsafe_allow_html=True)

    # actions row
    cols = st.columns([0.1,0.1,0.8])
    with cols[0]:
        if st.button("🗑", key=f"del::{p}", help="Удалить"):
            try:
                if is_dir:
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
                st.session_state.setdefault("logs", []).append(f"[{datetime.now().strftime('%H:%M:%S')}] Удалено: {p}")
                st.experimental_rerun()
            except Exception as e:
                st.session_state.setdefault("logs", []).append(f"[{datetime.now().strftime('%H:%M:%S')}] Ошибка удаления {p}: {e}")
    with cols[1]:
        if is_dir and st.button("➕", key=f"add::{p}", help="Создать подпапку"):
            newp = p / "new_folder"
            try:
                ensure_dir(newp)
                st.session_state.setdefault("logs", []).append(f"[{datetime.now().strftime('%H:%M:%S')}] Создана папка: {newp}")
                st.experimental_rerun()
            except Exception as e:
                st.session_state.setdefault("logs", []).append(f"[{datetime.now().strftime('%H:%M:%S')}] Ошибка создания папки {newp}: {e}")
    with cols[2]:
        if is_dir:
            uploads = st.file_uploader(f"Загрузить в {p.name}", type=[e[1:] for e in IMG_EXTS], accept_multiple_files=True, key=f"upl::{p}")
            if uploads and st.button("Загрузить", key=f"uplbtn::{p}"):
                for f in uploads:
                    try:
                        (p / Path(f.name).name).write_bytes(f.getbuffer())
                    except Exception:
                        pass
                st.session_state.setdefault("logs", []).append(f"[{datetime.now().strftime('%H:%M:%S')}] Загружено {len(uploads)} файл(ов) в {p}")
                st.experimental_rerun()

    if is_dir and depth < 3:
        for child in list_children(p):
            render_node(child, depth+1)

# ----------------- UI -----------------
st.title("Face Sorter — Streamlit")

with st.sidebar:
    st.header("Родительская папка")
    if "parent_path" not in st.session_state:
        st.session_state["parent_path"] = str(Path.cwd())
    if st.button("📂 Выбрать папку"):
        folder = pick_folder_dialog()
        if folder:
            st.session_state["parent_path"] = folder
    parent = Path(st.session_state["parent_path"]).expanduser().resolve()
    st.caption(str(parent))

    idx = load_index(parent)
    st.header("Очередь")
    groups = [p for p in parent.iterdir() if p.is_dir()]
    group_names = [g.name for g in groups]
    queue = st.session_state.get("queue", [])
    selected_groups = st.multiselect("Добавить группы", group_names, default=[])
    st.session_state["selected_groups_current"] = [str((parent / name)) for name in selected_groups]
    if st.button("➕ В очередь"):
        for name in selected_groups:
            p = parent / name
            if str(p) not in queue:
                queue.append(str(p))
        st.session_state["queue"] = queue

    if queue:
        st.write("Очередь:")
        st.write("\\n".join(f"{i+1}. {Path(g).name}" for i,g in enumerate(queue)))

st.header("Проводник выбранной папки")
render_node(parent, depth=0)

st.divider()
colA, colB = st.columns([0.2, 0.8])
with colA:
    if st.button("▶️ Старт", type="primary"):
        targets = []
        if st.session_state.get("queue"):
            targets = [Path(g) for g in st.session_state["queue"]]
        else:
            # if no queue, fallback to selected groups (all groups under parent)
            groups = [p for p in parent.iterdir() if p.is_dir()]
            if groups:
                targets = groups

        if not targets:
            st.session_state.setdefault("logs", []).append(f"[{datetime.now().strftime('%H:%M:%S')}] Нет выбранных папок для распознавания.")
        else:
            tot_total = tot_unknown = tot_group_only = 0
            for gdir in targets:
                t0 = datetime.now().strftime('%H:%M:%S')
                st.session_state.setdefault("logs", []).append(f"[{t0}] Обработка группы: {gdir.name}")
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

                # Clean up original processed images
                cleanup_processed_images(gdir, processed_images)
                tot_total += plan["stats"]["images_total"]
                tot_unknown += plan["stats"]["images_unknown_only"]
                tot_group_only += plan["stats"]["images_group_only"]
                t1 = datetime.now().strftime('%H:%M:%S')
                st.session_state["logs"].append(f"[{t1}] Готово: {gdir.name} — фото={plan['stats']['images_total']}, люди={persons_after}, group_only={plan['stats']['images_group_only']}, unknown={plan['stats']['images_unknown_only']}")
            idx["global_stats"]["images_total"] += tot_total
            idx["global_stats"]["images_unknown_only"] += tot_unknown
            idx["global_stats"]["images_group_only"] += tot_group_only
            save_index(parent, idx)
            st.session_state["queue"] = []
with colB:
    st.text_area("Логи", value="\\n".join(st.session_state.get("logs", [])), height=220)

st.caption("Параметры алгоритма редактируются в config.json. В UI они скрыты.")
