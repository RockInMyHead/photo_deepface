# app.py
# Minimal Explorer: Pick Folder -> Native dialog -> Minimal file browser -> Queue + Process
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Set
from datetime import datetime

import streamlit as st

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

# ---- Backend (provided by your project) ----
try:
    from core.cluster import build_plan, IMG_EXTS
except Exception:
    IMG_EXTS = {".jpg",".jpeg",".png",".webp",".bmp",".tiff"}
    def build_plan(*args, **kwargs):
        return {"eligible_clusters": [], "cluster_centroids": {}, "cluster_images": {}, "group_only_images": [], "unknown_images": [], "stats":{"images_total":0,"images_unknown_only":0,"images_group_only":0}}

st.set_page_config(page_title="Face Sorter — Explorer", layout="wide")

# ---- Minimal CSS ----
st.markdown("""
<style>
  .row { display:grid; grid-template-columns: 28px 1fr 110px 170px 120px; gap:8px; align-items:center; padding:6px 8px; border-bottom:1px solid #f1f5f9;}
  .row:hover { background:#f8fafc; }
  .hdr { font-weight:600; color:#334155; border-bottom:1px solid #e2e8f0; }
  .name { display:flex; align-items:center; gap:8px; }
  .iconbtn { border:1px solid #e5e7eb; background:#fff; border-radius:8px; padding:4px 6px; cursor:pointer; }
  .iconbtn:hover { background:#f1f5f9; }
  .muted { color:#64748b; font-size:12px; }
  .addr { display:flex; gap:8px; align-items:center; }
  .crumb { border:1px solid transparent; padding:4px 8px; border-radius:8px; }
  .crumb:hover { background:#f1f5f9; border-color:#e5e7eb; }
</style>
""", unsafe_allow_html=True)

# ---- Config (read-only) ----
def load_config(base: Path) -> Dict:
    p = base / "config.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"group_thr":3,"eps_sim":0.55,"min_samples":2,"min_face":110,"blur_thr":45.0,"det_size":640,"gpu_id":0,"match_thr":0.60}

CFG_BASE = Path(__file__).parent
CFG = load_config(CFG_BASE)

# ---- Persistence for global stats ----
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

# ---- Helpers ----
def _init_state():
    st.session_state.setdefault("parent_path", None)
    st.session_state.setdefault("current_dir", None)
    st.session_state.setdefault("selected_dirs", set())  # for queue
    st.session_state.setdefault("rename_target", None)   # str path
    st.session_state.setdefault("queue", [])
    st.session_state.setdefault("logs", [])

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
    step = 1024.0
    units = ["Б","КБ","МБ","ГБ","ТБ"]
    for u in units:
        if n < step or u == units[-1]:
            return f"{n:.0f} {u}" if u=="Б" else f"{n/1024:.1f} {u}"
        n /= step

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
                log(f"Удален оригинал: {img_path.name}")
        except Exception as e:
            log(f"Ошибка удаления {img_path.name}: {e}")

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

    existing_nums = sorted([int(d.name) for d in group_dir.iterdir() if d.is_dir() and d.name.isdigit()])
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

    all_in_group = set()
    for ext in IMG_EXTS:
        for f in group_dir.rglob(f"*{ext}"):
            if f.is_file():
                all_in_group.add(f)
    processed_images = processed_images.intersection(all_in_group)
    return len(person_idx["persons"]), processed_images

# ---- UI State ----
_init_state()

# ---- Step 1: Pick Folder ----
st.title("Face Sorter — Мини‑проводник")

if st.session_state["parent_path"] is None:
    st.info("Выберите папку для работы.")
    if st.button("📂 ВЫБРАТЬ ПАПКУ", type="primary"):
        folder = pick_folder_dialog()
        if folder:
            st.session_state["parent_path"] = folder
            st.session_state["current_dir"] = folder
            st.rerun()
else:
    # Address / Up
    curr = Path(st.session_state["current_dir"]).expanduser().resolve()
    parent_root = Path(st.session_state["parent_path"]).expanduser().resolve()

    addr_col1, addr_col2 = st.columns([0.08, 0.92])
    with addr_col1:
        up = curr.parent if curr != curr.anchor else None
        st.button("⬆️ Вверх", key="up", disabled=(up is None), on_click=(lambda: st.session_state.update({"current_dir": str(up)}) if up else None), use_container_width=True)
    with addr_col2:
        # Breadcrumbs (minimal)
        crumbs = list(curr.parts)
        accum = Path(crumbs[0])
        bc_cols = st.columns(len(crumbs))
        for i, part in enumerate(crumbs):
            if i > 0: accum = accum / part
            with bc_cols[i]:
                st.button(part or "/", key=f"bc::{i}", use_container_width=True, on_click=lambda p=str(accum): st.session_state.update({"current_dir": p}))

    st.markdown("---")

    # Header row
    st.markdown('<div class="row hdr"><div></div><div>Имя</div><div>Тип</div><div>Изменён</div><div>Размер</div></div>', unsafe_allow_html=True)

    # Directory listing
    items = list_dir(curr)
    for item in items:
        is_dir = item.is_dir()
        sel_key = f"sel::{item}"
        name_btn_key = f"open::{item}"
        del_key = f"del::{item}"
        ren_key = f"ren::{item}"
        ren_input_key = f"ren_input::{item}"

        c1, c2, c3, c4, c5 = st.columns([0.06, 0.60, 0.12, 0.14, 0.08])
        with c1:
            if is_dir:
                checked = st.checkbox("", key=sel_key, value=(str(item) in st.session_state["selected_dirs"]))
                if checked:
                    st.session_state["selected_dirs"].add(str(item))
                else:
                    st.session_state["selected_dirs"].discard(str(item))
        with c2:
            icon = "📁" if is_dir else "🗎"
            # Name + inline action icons
            name_cols = st.columns([0.75, 0.08, 0.08, 0.09])
            with name_cols[0]:
                if is_dir:
                    if st.button(f"{icon} {item.name}", key=name_btn_key, use_container_width=True):
                        st.session_state["current_dir"] = str(item)
                        st.rerun()
                else:
                    st.write(f"{icon} {item.name}")
            with name_cols[1]:
                if st.button("✏️", key=ren_key, help="Переименовать", use_container_width=True):
                    st.session_state["rename_target"] = str(item)
            with name_cols[2]:
                if st.button("🗑️", key=del_key, help="Удалить", use_container_width=True):
                    try:
                        if is_dir:
                            shutil.rmtree(item, ignore_errors=True)
                        else:
                            item.unlink(missing_ok=True)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка удаления: {e}")
            with name_cols[3]:
                if not is_dir and item.suffix.lower() in IMG_EXTS:
                    st.button("👁️", key=f"prev::{item}", help="Предпросмотр", use_container_width=True, on_click=lambda p=str(item): st.session_state.update({"preview": p}))
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
                if st.button("Сохранить", key=f"save::{item}"):
                    try:
                        new_path = item.parent / new_name
                        item.rename(new_path)
                        st.session_state["rename_target"] = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка: {e}")
            with rc3:
                if st.button("Отмена", key=f"cancel::{item}"):
                    st.session_state["rename_target"] = None
                    st.rerun()

    # Optional preview (single file image)
    prev = st.session_state.get("preview")
    if prev:
        pp = Path(prev)
        if pp.exists() and pp.is_file() and pp.suffix.lower() in IMG_EXTS:
            st.image(str(pp), caption=pp.name, use_container_width=True)

    st.markdown("---")

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
    if st.button("▶️ Обработать", type="primary"):
        parent = parent_root
        idx = load_index(parent)
        # Targets = queue or all subfolders of current_dir
        targets = [Path(p) for p in st.session_state["queue"]]
        if not targets:
            targets = [p for p in curr.iterdir() if p.is_dir()]
        if not targets:
            st.warning("Нет целей для обработки.")
        else:
            tot_total = tot_unknown = tot_group_only = 0
            progress = st.progress(0, text="Инициализация…")
            status = st.status("Идёт обработка…", expanded=True)
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
                        tot_total  += plan["stats"]["images_total"]
                        tot_unknown += plan["stats"]["images_unknown_only"]
                        tot_group_only += plan["stats"]["images_group_only"]
                        st.success(f"Готово: {gdir.name}")
                    except Exception as e:
                        st.error(f"Ошибка в {gdir.name}: {e}")
                    progress.progress(k/len(targets), text=f"{k}/{len(targets)}")
            idx["global_stats"]["images_total"] += tot_total
            idx["global_stats"]["images_unknown_only"] += tot_unknown
            idx["global_stats"]["images_group_only"] += tot_group_only
            save_index(parent, idx)
            st.session_state["queue"] = []
            st.success("Обработка завершена.")
