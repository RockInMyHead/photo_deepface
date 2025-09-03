# app.py
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Set
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

# ----------------- Page / Theme -----------------
st.set_page_config(
    page_title="Face Sorter",
    page_icon="🧠",
    layout="wide",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": "Face Sorter · Streamlit UI"
    },
)

# ----------------- CSS (lightweight) -----------------
st.markdown(
    """
    <style>
      .app-header {
        padding: 12px 16px;
        border-radius: 14px;
        background: linear-gradient(135deg, #0ea5e9 0%, #22c55e 100%);
        color: white !important;
        margin-bottom: 12px;
      }
      .app-header h1, .app-header p { color: white !important; margin: 0; }
      .stMetric { border-radius: 14px; }
      .small-muted { color: var(--text-color, #6b7280); font-size: 12px; }
      .code-badge {
        display:inline-block; padding:2px 8px; border-radius:999px; background:#eef2f7; font-size:12px;
        border:1px solid #e5e7eb; margin-right:6px;
      }
      .thumb { border-radius: 10px; border: 1px solid #e5e7eb; }
      .queue-table .row {
        display:grid; grid-template-columns: 1fr 110px; gap:8px; padding:8px 10px; border-bottom:1px solid #eef2f7;
      }
      .queue-table .row:hover { background:#fafafa; }
      .action-bar { gap: 6px; display:flex; align-items:center; }
      .log-area textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; }
      .st-emotion-cache-1y4p8pa { overflow: visible !important; } /* allow buttons near images */
    </style>
    """,
    unsafe_allow_html=True,
)

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
    return {
        "group_order": [],
        "group_counts": {},
        "global_stats": {"images_total":0,"images_unknown_only":0,"images_group_only":0},
        "last_run": None
    }

def save_index(parent: Path, idx: Dict):
    idx["last_run"] = datetime.now().isoformat(timespec="seconds")
    (parent / "global_index.json").write_text(
        json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8"
    )

# ----- identity persistence per group -----
def load_person_index(group_dir: Path) -> Dict:
    p = group_dir / "person_index.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"persons": []}

def save_person_index(group_dir: Path, data: Dict):
    (group_dir / "person_index.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

# ----- cleanup function -----
def cleanup_processed_images(group_dir: Path, processed_images: Set[Path]):
    """Remove original image files after they have been processed and copied to group folders."""
    for img_path in processed_images:
        try:
            if img_path.exists() and img_path.is_file():
                img_path.unlink()
                _log(f"Удален оригинал: {img_path.name}")
        except Exception as e:
            _log(f"Ошибка удаления {img_path.name}: {e}")

# ----- core operations -----
def match_and_apply(group_dir: Path, plan: Dict, match_thr: float) -> Tuple[int, Set[Path]]:
    from math import isfinite
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

    existing_nums = sorted([int(p.name) for p in group_dir.iterdir()
                            if p.is_dir() and p.name.isdigit()])
    cur_max = existing_nums[-1] if existing_nums else 0
    new_nums = {}
    for cid in plan["eligible_clusters"]:
        if cid in assigned:
            continue
        cur_max += 1
        new_nums[cid] = cur_max
        person_idx["persons"].append(
            {"number": cur_max, "proto": centroids_norm[cid].tolist()}
        )

    for cid, num in {**assigned, **new_nums}.items():
        ensure_dir(group_dir / str(num))

    copied: Set[Tuple[int, Path]] = set()
    cluster_images = {}
    for k, v in plan.get("cluster_images", {}).items():
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
    processed_images: Set[Path] = set()
    for cid in plan["eligible_clusters"]:
        for img in cluster_images.get(cid, []):
            processed_images.add(Path(img))

    for img in plan.get("group_only_images", []):
        processed_images.add(Path(img))

    for img in plan.get("unknown_images", []):
        processed_images.add(Path(img))

    # Also collect any other image files that were scanned but not categorized
    all_images_in_group = set()
    for ext in IMG_EXTS:
        for img_file in group_dir.rglob(f"*{ext}"):
            if img_file.is_file():
                all_images_in_group.add(img_file)

    # Keep only images that actually reside in current group tree
    processed_images = processed_images.intersection(all_images_in_group)

    return len(person_idx["persons"]), processed_images

# ----------------- Utils: logging -----------------
def _init_state():
    st.session_state.setdefault("logs", [])
    st.session_state.setdefault("queue", [])
    st.session_state.setdefault("parent_path", str(Path.cwd()))
    st.session_state.setdefault("filter_text", "")
    st.session_state.setdefault("max_depth", 3)
    st.session_state.setdefault("show_non_images", False)
    st.session_state.setdefault("thumb_size", 88)

def _log(msg: str):
    ts = datetime.now().strftime('%H:%M:%S')
    st.session_state["logs"].append(f"[{ts}] {msg}")

def safe_rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

# ----------------- Tree rendering -----------------
def list_children(p: Path):
    try:
        return sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except Exception:
        return []

def render_node(p: Path, depth: int = 0):
    """Folder/file renderer with actions; restricted by max_depth and filter."""
    max_depth = st.session_state.get("max_depth", 3)
    filter_text = st.session_state.get("filter_text", "").lower().strip()
    show_non_images = st.session_state.get("show_non_images", False)

    is_dir = p.is_dir()
    name = p.name

    if filter_text and filter_text not in name.lower():
        # If folder doesn't match but it's a directory, still traverse if children may match
        if is_dir and depth < max_depth:
            for child in list_children(p):
                render_node(child, depth+1)
        return

    if is_dir:
        with st.expander(f"📁 {name}", expanded=(depth == 0)):
            cols = st.columns([0.15, 0.15, 0.4, 0.3])
            with cols[0]:
                if st.button("🗑 Удалить", key=f"del::{p}"):
                    try:
                        shutil.rmtree(p, ignore_errors=True)
                        _log(f"Удалено: {p}")
                        safe_rerun()
                    except Exception as e:
                        _log(f"Ошибка удаления {p}: {e}")
            with cols[1]:
                if st.button("➕ Подпапка", key=f"add::{p}"):
                    newp = p / "new_folder"
                    try:
                        ensure_dir(newp)
                        _log(f"Создана папка: {newp}")
                        safe_rerun()
                    except Exception as e:
                        _log(f"Ошибка создания папки {newp}: {e}")
            with cols[2]:
                uploads = st.file_uploader(
                    f"Загрузить в «{name}»",
                    type=[e[1:] for e in IMG_EXTS],
                    accept_multiple_files=True,
                    key=f"upl::{p}",
                )
            with cols[3]:
                if uploads and st.button("⬆️ Загрузить", key=f"uplbtn::{p}", type="secondary"):
                    for f in uploads:
                        try:
                            (p / Path(f.name).name).write_bytes(f.getbuffer())
                        except Exception:
                            pass
                    _log(f"Загружено {len(uploads)} файл(ов) в {p}")
                    safe_rerun()

            if depth < max_depth:
                grid_cols = st.columns(5)
                i = 0
                for child in list_children(p):
                    if child.is_file() and (child.suffix.lower() not in IMG_EXTS) and not show_non_images:
                        continue
                    col = grid_cols[i % len(grid_cols)]
                    with col:
                        if child.is_dir():
                            st.caption(f"📁 {child.name}")
                        else:
                            if child.suffix.lower() in IMG_EXTS:
                                st.image(str(child), width=st.session_state.get("thumb_size", 88), caption=child.name, use_container_width=False)
                            else:
                                st.caption(f"🗎 {child.name}")
                    i += 1

                # Recurse deeper
                for child in list_children(p):
                    render_node(child, depth+1)
    else:
        # Single file (only if not handled via folder grid above)
        if p.suffix.lower() in IMG_EXTS:
            st.image(str(p), width=st.session_state.get("thumb_size", 88), caption=p.name)
        elif st.session_state.get("show_non_images", False):
            st.caption(f"🗎 {p.name}")

# ----------------- App State -----------------
_init_state()

# ----------------- Header -----------------
with st.container():
    st.markdown('<div class="app-header"><h1>Face Sorter — Streamlit</h1><p>Кластеризация лиц, раскладка по персональным папкам, безопасное очищение исходников.</p></div>', unsafe_allow_html=True)

# ----------------- Sidebar -----------------
with st.sidebar:
    st.header("Путь")
    if st.button("📂 Выбрать папку", use_container_width=True):
        folder = pick_folder_dialog()
        if folder:
            st.session_state["parent_path"] = folder
            _log(f"Выбрана папка: {folder}")

    parent = Path(st.session_state["parent_path"]).expanduser().resolve()
    st.caption(str(parent))

    idx = load_index(parent)

    st.divider()
    st.header("Параметры UI")
    st.text_input("Фильтр по имени", key="filter_text", placeholder="введите фрагмент имени…")
    st.slider("Глубина обхода", 1, 6, key="max_depth")
    st.checkbox("Показывать не-изображения", key="show_non_images")
    st.slider("Размер превью, px", 64, 160, key="thumb_size")

    st.divider()
    st.header("Конфиг (read-only)")
    c1, c2 = st.columns(2)
    with c1: st.markdown(f'<span class="code-badge">group_thr: {CFG["group_thr"]}</span>', unsafe_allow_html=True)
    with c2: st.markdown(f'<span class="code-badge">eps_sim: {CFG["eps_sim"]}</span>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3: st.markdown(f'<span class="code-badge">min_samples: {CFG["min_samples"]}</span>', unsafe_allow_html=True)
    with c4: st.markdown(f'<span class="code-badge">min_face: {CFG["min_face"]}</span>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5: st.markdown(f'<span class="code-badge">blur_thr: {CFG["blur_thr"]}</span>', unsafe_allow_html=True)
    with c6: st.markdown(f'<span class="code-badge">det_size: {CFG["det_size"]}</span>', unsafe_allow_html=True)
    c7, c8 = st.columns(2)
    with c7: st.markdown(f'<span class="code-badge">gpu_id: {CFG["gpu_id"]}</span>', unsafe_allow_html=True)
    with c8: st.markdown(f'<span class="code-badge">match_thr: {CFG["match_thr"]}</span>', unsafe_allow_html=True)

    st.caption("Правка значений — через config.json. В UI они скрыты от изменения.")

    st.divider()
    st.header("Очередь")
    groups = [p for p in parent.iterdir() if p.is_dir()]
    group_names = [g.name for g in groups]

    selected_groups = st.multiselect("Добавить группы", group_names, default=[])
    if st.button("➕ В очередь", use_container_width=True):
        for name in selected_groups:
            p = parent / name
            if str(p) not in st.session_state["queue"]:
                st.session_state["queue"].append(str(p))
        _log(f"В очередь: {', '.join(selected_groups) if selected_groups else '—'}")

    if st.session_state["queue"]:
        st.markdown("**Состав очереди**")
        with st.container():
            st.markdown('<div class="queue-table">', unsafe_allow_html=True)
            for i, g in enumerate(st.session_state["queue"], start=1):
                gname = Path(g).name
                count = idx.get("group_counts", {}).get(str(Path(g)), "—")
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.write(f"{i}. {gname}")
                with col2:
                    st.write(f"персоны: {count}")
            st.markdown('</div>', unsafe_allow_html=True)

        cqa, cqb = st.columns(2)
        with cqa:
            if st.button("🧹 Очистить очередь", use_container_width=True):
                st.session_state["queue"] = []
                _log("Очередь очищена")
        with cqb:
            if st.button("↩️ Удалить последний", use_container_width=True):
                if st.session_state["queue"]:
                    removed = Path(st.session_state["queue"].pop()).name
                    _log(f"Из очереди удалено: {removed}")

    st.divider()
    st.header("Сводка")
    gstats = idx.get("global_stats", {})
    m1, m2, m3 = st.columns(3)
    m1.metric("Всего фото", gstats.get("images_total", 0))
    m2.metric("group_only", gstats.get("images_group_only", 0))
    m3.metric("unknown", gstats.get("images_unknown_only", 0))
    st.caption(f"Последний запуск: {idx.get('last_run') or '—'}")

# ----------------- Main: Tabs -----------------
tab_explorer, tab_process, tab_logs, tab_help = st.tabs(["📂 Проводник", "⚙️ Обработка", "🧾 Логи", "❓ Справка"])

with tab_explorer:
    st.subheader("Проводник выбранной папки")
    render_node(parent, depth=0)

with tab_process:
    st.subheader("Пакетная обработка")
    st.write("Сервис пройдётся по очереди; если очередь пуста — по всем подпапкам выбранной родительской папки.")

    c_run, c_export = st.columns([0.3, 0.7])
    start_clicked = c_run.button("▶️ Старт", type="primary", use_container_width=True)

    if start_clicked:
        # Targets
        if st.session_state.get("queue"):
            targets = [Path(g) for g in st.session_state["queue"]]
        else:
            targets = [p for p in parent.iterdir() if p.is_dir()]

        if not targets:
            _log("Нет выбранных папок для распознавания.")
            st.warning("Нет целей для обработки.")
        else:
            tot_total = tot_unknown = tot_group_only = 0
            progress = st.progress(0, text="Инициализация…")
            status = st.status("Запуск обработки", expanded=True)

            with status:
                for k, gdir in enumerate(targets, start=1):
                    t0 = datetime.now().strftime('%H:%M:%S')
                    st.write(f"[{t0}] Обработка группы: **{gdir.name}**")
                    _log(f"Обработка группы: {gdir.name}")

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

                        # Clean up original processed images
                        cleanup_processed_images(gdir, processed_images)

                        # Aggregate stats
                        tot_total  += plan["stats"]["images_total"]
                        tot_unknown += plan["stats"]["images_unknown_only"]
                        tot_group_only += plan["stats"]["images_group_only"]

                        st.success(
                            f"Готово: {gdir.name} — фото={plan['stats']['images_total']}, люди={persons_after}, "
                            f"group_only={plan['stats']['images_group_only']}, unknown={plan['stats']['images_unknown_only']}"
                        )
                        _log(
                            f"Готово: {gdir.name} — фото={plan['stats']['images_total']}, люди={persons_after}, "
                            f"group_only={plan['stats']['images_group_only']}, unknown={plan['stats']['images_unknown_only']}"
                        )
                    except Exception as e:
                        st.error(f"Ошибка в {gdir.name}: {e}")
                        _log(f"Ошибка обработки {gdir.name}: {e}")

                    progress.progress(k / max(len(targets), 1), text=f"{k}/{len(targets)}")

            # Save rollup
            idx["global_stats"]["images_total"] += tot_total
            idx["global_stats"]["images_unknown_only"] += tot_unknown
            idx["global_stats"]["images_group_only"] += tot_group_only
            save_index(parent, idx)

            # Reset queue
            st.session_state["queue"] = []

            # KPIs
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Обработано групп", len(targets))
            m2.metric("Фото (сумма за прогон)", tot_total)
            m3.metric("group_only (сумма)", tot_group_only)
            m4.metric("unknown (сумма)", tot_unknown)
            st.toast("Обработка завершена", icon="✅")

with tab_logs:
    st.subheader("Логи")
    logs = "\n".join(st.session_state.get("logs", []))
    st.text_area("События", logs, height=300, key="logs_view", label_visibility="collapsed", help="Последние события пайплайна", placeholder="", kwargs={"class":"log-area"})
    st.download_button("⬇️ Скачать логи", data=logs or "", file_name=f"face_sorter_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", mime="text/plain")

with tab_help:
    st.subheader("Справка / FAQ")
    st.markdown(
        """
        **Поток:**  
        1) Выберите родительскую папку.  
        2) Сформируйте очередь (или оставьте пустой — тогда возьмутся все подпапки).  
        3) Нажмите «Старт».  
        4) Проверяйте «Проводник» и «Логи».

        **Очистка оригиналов:**  
        После успешной раскладки и копирования изображения-источники удаляются из исходных мест в пределах группы — это контролируется логикой `cleanup_processed_images`.

        **Конфиг:**  
        Пороговые значения/детектор управляются через `config.json`. UI отображает текущие параметры в режиме read-only.
        """
    )
