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
)

# ----------------- Minimal CSS for Windows-like Explorer -----------------
st.markdown(
    """
    <style>
      .address-bar { display:flex; gap:6px; flex-wrap:wrap; align-items:center; }
      .toolbar { display:flex; gap:8px; align-items:center; margin-bottom:8px; }
      .detail-header { font-weight:600; border-bottom:1px solid #e5e7eb; padding:6px 4px; }
      .detail-row { border-bottom:1px solid #f2f2f2; padding:6px 4px; }
      .detail-row:hover { background:#fafafa; }
      .muted { color:#6b7280; font-size:12px; }
      .name-btn div[data-testid="stMarkdownContainer"] p { margin:0; }
      .tree-item { padding:2px 0; }
      .preview { border:1px solid #e5e7eb; border-radius:8px; padding:8px; }
      .badge { display:inline-block; padding:2px 8px; border-radius:999px; background:#eef2f7; font-size:12px; border:1px solid #e5e7eb; }
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

# ----------------- Utils -----------------
def _init_state():
    st.session_state.setdefault("logs", [])
    st.session_state.setdefault("queue", [])
    st.session_state.setdefault("parent_path", str(Path.cwd()))
    st.session_state.setdefault("current_dir", st.session_state["parent_path"])
    st.session_state.setdefault("selected", set())
    st.session_state.setdefault("history", [st.session_state["parent_path"]])
    st.session_state.setdefault("history_index", 0)

def _log(msg: str):
    ts = datetime.now().strftime('%H:%M:%S')
    st.session_state["logs"].append(f"[{ts}] {msg}")

def human_size(n: int) -> str:
    # Windows-like units
    for unit in ["Б", "КБ", "МБ", "ГБ", "ТБ"]:
        if n < 1024 or unit == "ТБ":
            return f"{n:.0f} {unit}" if unit == "Б" else f"{n/1024:.1f} {unit}" if unit != "Б" else f"{n} Б"
        n //= 1024
    return f"{n} Б"

def go_to(path: Path, push_history: bool = True):
    path = path.expanduser().resolve()
    st.session_state["current_dir"] = str(path)
    st.session_state["selected"] = set()
    if push_history:
        hist = st.session_state["history"]
        idx = st.session_state["history_index"]
        # truncate forward history
        hist[:] = hist[:idx+1]
        hist.append(str(path))
        st.session_state["history_index"] = len(hist) - 1

def history_back():
    if st.session_state["history_index"] > 0:
        st.session_state["history_index"] -= 1
        st.session_state["current_dir"] = st.session_state["history"][st.session_state["history_index"]]
        st.session_state["selected"] = set()

def history_forward():
    if st.session_state["history_index"] < len(st.session_state["history"]) - 1:
        st.session_state["history_index"] += 1
        st.session_state["current_dir"] = st.session_state["history"][st.session_state["history_index"]]
        st.session_state["selected"] = set()

def list_dir(p: Path) -> List[Path]:
    try:
        return sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except Exception:
        return []

# ----------------- Tree (left pane) -----------------
def render_tree(root: Path, depth: int = 0, max_depth: int = 4):
    if depth > max_depth or not root.exists():
        return
    # only directories
    try:
        children = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda x: x.name.lower())
    except Exception:
        children = []
    for d in children:
        with st.container():
            c1, c2 = st.columns([0.1, 0.9])
            with c1:
                st.text("📁")
            with c2:
                if st.button(d.name, key=f"tree::{d}", use_container_width=True):
                    go_to(d)
        with st.container():
            with st.expander("", expanded=False):
                render_tree(d, depth+1, max_depth)

# ----------------- Explorer (right pane) -----------------
def render_breadcrumbs(curr: Path, root: Path):
    parts = list(curr.parts)
    root_parts = list(root.parts)
    with st.container():
        bt1, bt2, bt3, _ = st.columns([0.06, 0.06, 0.06, 0.82])
        with bt1:
            st.button("⬅️", key="nav_back", on_click=history_back, help="Назад", use_container_width=True, disabled=st.session_state["history_index"] <= 0)
        with bt2:
            st.button("➡️", key="nav_forward", on_click=history_forward, help="Вперёд", use_container_width=True, disabled=st.session_state["history_index"] >= len(st.session_state["history"])-1)
        with bt3:
            up = curr.parent if curr != root else None
            st.button("⬆️", key="nav_up", on_click=(lambda: go_to(up) if up else None), help="Вверх", use_container_width=True, disabled=up is None)

    crumb_cols = st.columns(len(parts))
    cumulative = Path(parts[0])
    for i, part in enumerate(parts):
        if i > 0:
            cumulative = cumulative / part
        with crumb_cols[i]:
            st.button(part, key=f"bc::{i}::{cumulative}", on_click=lambda p=cumulative: go_to(p), use_container_width=True)

def render_details(curr: Path):
    items = list_dir(curr)
    # Header actions
    with st.container():
        col1, col2, col3, col4, col5 = st.columns([0.12, 0.18, 0.22, 0.28, 0.20])
        with col1:
            if st.button("🔄 Обновить", use_container_width=True):
                st.experimental_rerun()
        with col2:
            if st.button("📁 Новая папка", use_container_width=True):
                newp = curr / "New Folder"
                i = 1
                while newp.exists():
                    newp = curr / f"New Folder ({i})"
                    i += 1
                ensure_dir(newp)
                st.experimental_rerun()
        with col3:
            uploads = st.file_uploader("Загрузить файлы", accept_multiple_files=True, key=f"upl::{curr}")
        with col4:
            if uploads and st.button("⬆️ Загрузить", use_container_width=True):
                for f in uploads:
                    (curr / Path(f.name).name).write_bytes(f.getbuffer())
                st.experimental_rerun()
        with col5:
            if st.button("🗑 Удалить выбранное", use_container_width=True, disabled=not st.session_state["selected"]):
                for s in list(st.session_state["selected"]):
                    p = Path(s)
                    try:
                        if p.is_dir():
                            shutil.rmtree(p, ignore_errors=True)
                        else:
                            p.unlink(missing_ok=True)
                    except Exception as e:
                        pass
                st.session_state["selected"] = set()
                st.experimental_rerun()

    # Table header
    h1, h2, h3, h4, h5 = st.columns([0.06, 0.54, 0.12, 0.14, 0.14])
    with h1: st.markdown('<div class="detail-header">✔</div>', unsafe_allow_html=True)
    with h2: st.markdown('<div class="detail-header">Имя</div>', unsafe_allow_html=True)
    with h3: st.markdown('<div class="detail-header">Тип</div>', unsafe_allow_html=True)
    with h4: st.markdown('<div class="detail-header">Размер</div>', unsafe_allow_html=True)
    with h5: st.markdown('<div class="detail-header">Изменён</div>', unsafe_allow_html=True)

    # Rows
    for child in items:
        c1, c2, c3, c4, c5 = st.columns([0.06, 0.54, 0.12, 0.14, 0.14])
        key = f"sel::{child}"
        selected_now = key in st.session_state and st.session_state[key]
        with c1:
            checked = st.checkbox("", key=key, value=selected_now)
            if checked:
                st.session_state["selected"].add(str(child))
            else:
                st.session_state["selected"].discard(str(child))
        with c2:
            icon = "📁" if child.is_dir() else "🗎"
            if child.is_dir():
                if st.button(f"{icon} {child.name}", key=f"open::{child}", help="Открыть папку", use_container_width=True):
                    go_to(child)
            else:
                st.markdown(f'<div class="name-btn">{icon} {child.name}</div>', unsafe_allow_html=True)
        with c3:
            st.write("Папка" if child.is_dir() else (child.suffix.lower()[1:] if child.suffix else "файл"))
        with c4:
            if child.is_dir():
                st.write("—")
            else:
                try:
                    st.write(f"{child.stat().st_size} Б")
                except Exception:
                    st.write("—")
        with c5:
            try:
                st.write(datetime.fromtimestamp(child.stat().st_mtime).strftime("%Y-%m-%d %H:%M"))
            except Exception:
                st.write("—")

    # Rename action (single selection)
    if len(st.session_state["selected"]) == 1:
        target = Path(list(st.session_state["selected"])[0])
        st.markdown("---")
        st.write("Переименовать выбранный:")
        new_name = st.text_input("Новое имя", value=target.name, key=f"rename::{target}")
        if st.button("✏️ Переименовать"):
            try:
                new_path = target.parent / new_name
                target.rename(new_path)
                st.session_state["selected"] = {str(new_path)}
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Ошибка переименования: {e}")

    # Preview (single image file)
    if len(st.session_state["selected"]) == 1:
        target = Path(list(st.session_state["selected"])[0])
        if target.is_file() and target.suffix.lower() in IMG_EXTS:
            st.markdown("---")
            st.write("Предпросмотр:")
            with st.container():
                st.image(str(target), use_container_width=True)
                st.caption(f"{target.name}")

# ----------------- App State -----------------
_init_state()

# ----------------- Sidebar -----------------
with st.sidebar:
    st.header("Путь")
    if st.button("📂 Выбрать папку", use_container_width=True):
        folder = pick_folder_dialog()
        if folder:
            st.session_state["parent_path"] = folder
            go_to(Path(folder))
    parent = Path(st.session_state["parent_path"]).expanduser().resolve()
    st.caption(str(parent))

    idx = load_index(parent)

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

    if st.session_state["queue"]:
        st.write("Очередь:")
        st.write("\\n".join(f"{i+1}. {Path(g).name}" for i,g in enumerate(st.session_state['queue'])))

    st.divider()
    st.header("Сводка")
    gstats = idx.get("global_stats", {})
    m1, m2, m3 = st.columns(3)
    m1.metric("Всего фото", gstats.get("images_total", 0))
    m2.metric("group_only", gstats.get("images_group_only", 0))
    m3.metric("unknown", gstats.get("images_unknown_only", 0))
    st.caption(f"Последний запуск: {idx.get('last_run') or '—'}")

# ----------------- Main: Tabs -----------------
tab_explorer, tab_process, tab_logs, tab_help = st.tabs(["📁 Проводник (Windows-style)", "⚙️ Обработка", "🧾 Логи", "❓ Справка"])

with tab_explorer:
    # Two-pane Explorer
    left, right = st.columns([0.28, 0.72], gap="small")
    with left:
        st.subheader("Навигация")
        render_tree(Path(st.session_state["parent_path"]).expanduser().resolve())
    with right:
        current = Path(st.session_state["current_dir"]).expanduser().resolve()
        st.subheader("Содержимое")
        render_breadcrumbs(current, Path(st.session_state["parent_path"]).expanduser().resolve())
        render_details(current)

with tab_process:
    st.subheader("Пакетная обработка")
    st.write("Сервис пройдётся по очереди; если очередь пуста — по всем подпапкам выбранной родительской папки.")
    start_clicked = st.button("▶️ Старт", type="primary", use_container_width=False)

    parent = Path(st.session_state["parent_path"]).expanduser().resolve()
    idx = load_index(parent)

    if start_clicked:
        # Targets
        if st.session_state.get("queue"):
            targets = [Path(g) for g in st.session_state["queue"]]
        else:
            targets = [p for p in parent.iterdir() if p.is_dir()]

        if not targets:
            st.warning("Нет целей для обработки.")
        else:
            tot_total = tot_unknown = tot_group_only = 0
            progress = st.progress(0, text="Инициализация…")
            status = st.status("Запуск обработки", expanded=True)

            with status:
                for k, gdir in enumerate(targets, start=1):
                    t0 = datetime.now().strftime('%H:%M:%S')
                    st.write(f"[{t0}] Обработка группы: **{gdir.name}**")

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
                    except Exception as e:
                        st.error(f"Ошибка в {gdir.name}: {e}")

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
    st.text_area("События", logs, height=300, key="logs_view")
    st.download_button("⬇️ Скачать логи", data=logs or "", file_name=f"face_sorter_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", mime="text/plain")

with tab_help:
    st.subheader("Справка / FAQ")
    st.markdown(
        """
        **Проводник (Windows-style):**
        - Левая панель — дерево папок. Правая панель — список с деталями.
        - Навигация: Назад/Вперёд/Вверх, хлебные крошки.
        - Массовые действия: удаление выбранных, загрузка файлов, создание папки.
        - Один файл → блок предпросмотра (для изображений).

        **Поток обработки:**
        1) Выберите родительскую папку (слева).
        2) Сформируйте очередь (или оставьте пустой — возьмутся все подпапки).
        3) Нажмите «Старт» во вкладке «Обработка».

        **Конфиг:**
        Пороговые значения/детектор управляются через `config.json`. UI не содержит настроек — только отображение.
        """
    )
