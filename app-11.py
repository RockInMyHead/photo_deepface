# app.py — Single Explorer (DnD) with folder previews & file thumbnails
#
# Основано на последнем варианте пользователя (функция render_explorer_dnd):
# - Drag source: **все объекты** текущей папки (и файлы, и папки).
# - Drop targets: **подпапки + «Вверх»**.
# Дополнительно:
# - Превью: миниатюры для изображений, обложки для папок (первое изображение внутри).
# - Безопасное перемещение с защитой от коллизий имён (file (n).ext).
# - Один-единственный проводник, без дубликатов UI.

import os
import socket
import shutil
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

# ===== Optional deps =====
try:
    from streamlit_sortables import sort_items  # pip install streamlit-sortables
except Exception:
    sort_items = None

try:
    from PIL import Image
except Exception:
    Image = None

# ===== Settings =====
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

st.set_page_config(page_title="Face Sorter — Один проводник (DnD + превью)", layout="wide")

# ===== Helpers =====

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_move(src: Path, dst_dir: Path) -> Path:
    """Перемещение с защитой от коллизий имён (file (n).ext)."""
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


# ===== Thumbnails (files & folders) =====

def _make_square_thumb(img_path: Path, size: int = 96):
    if Image is None:
        return None
    try:
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        if w != h:
            m = min(w, h); left = (w - m) // 2; top = (h - m) // 2
            im = im.crop((left, top, left + m, top + m))
        im = im.resize((size, size))
        return im
    except Exception:
        return None


def _to_data_url(im: "Image.Image") -> Optional[str]:
    if im is None:
        return None
    try:
        buf = BytesIO(); im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=1800)
def file_thumb_url(path_str: str, size: int, mtime: float) -> Optional[str]:
    """Миниатюра файла как data URL (кешируется по mtime)."""
    if Image is None:
        return None
    im = _make_square_thumb(Path(path_str), size)
    return _to_data_url(im) if im is not None else None


@st.cache_data(show_spinner=False, ttl=1800)
def first_image_in_folder(folder_str: str, folder_mtime: float) -> Optional[str]:
    p = Path(folder_str)
    try:
        for child in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            if child.is_file() and child.suffix.lower() in IMG_EXTS:
                return str(child)
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False, ttl=1800)
def folder_cover_url(folder_str: str, size: int, folder_mtime: float) -> Optional[str]:
    """Превью папки: миниатюра первого изображения в папке (нерекурсивно)."""
    if Image is None:
        return None
    cover = first_image_in_folder(folder_str, folder_mtime)
    if not cover:
        return None
    return file_thumb_url(cover, size, Path(cover).stat().st_mtime)


# ===== LAN URL (инфо) =====

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


def make_network_url() -> str:
    ip = get_lan_ip()
    port = st.get_option("server.port") or 8501
    base = st.get_option("server.baseUrlPath") or ""
    base = ("/" + base.strip("/")) if base else ""
    return f"http://{ip}:{port}{base}"


# ===== UI Styles =====
st.markdown(
    """
<style>
  /* Sortables styles */
  div[data-testid=\"stSortables\"] > div > h3 { margin: 6px 0; font-weight:600; display:flex; align-items:center; gap:10px; }
  div[data-testid=\"stSortables\"] ul { margin:0 !important; padding:0 !important; }
  div[data-testid=\"stSortables\"] li {
      list-style:none; margin:0 0 6px 0 !important; padding:8px 10px !important;
      border:1px dashed #e5e7eb; border-radius:10px; background:#fff;
      display:flex; align-items:center; gap:10px; min-height:64px; font-size: 13px;
  }
  .itm { display:flex; align-items:center; gap:10px; }
  .itm img.thumb { width:64px; height:64px; border-radius:8px; border:1px solid #e5e7eb; object-fit:cover; }
  .itm .name { font-weight:500; color:#1f2937; }
  .hdr img.fold { width:28px; height:28px; border-radius:6px; border:1px solid #e5e7eb; object-fit:cover; }
  .hdr span { line-height:28px; }
</style>
    """,
    unsafe_allow_html=True,
)

# ===== Folder picker =====

def pick_folder_dialog() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
        folder = filedialog.askdirectory(title="Выберите папку")
        root.destroy(); return folder
    except Exception:
        return None


# ===== Single Explorer (DnD only) =====

def render_explorer_dnd(curr: Path, parent_root: Path):
    """Drag-and-drop в стиле Windows Explorer.
    Drag source: **все объекты** в текущей папке (файлы и папки) — каждый с превью/иконкой.
    Drop target: подпапки + "Вверх" (у папок в шапке — обложка по первому изображению).
    """
    if sort_items is None:
        st.error("DnD требует пакет: pip install streamlit-sortables")
        return

    # Формируем источники (все объекты: файлы и папки)
    src_items: List[str] = []
    label_to_path: Dict[str, str] = {}
    try:
        for p in sorted(curr.iterdir(), key=lambda x: x.name.lower()):
            # Иконка/превью
            if p.is_dir():
                try: mtime = p.stat().st_mtime
                except Exception: mtime = 0.0
                cov = folder_cover_url(str(p), 32, mtime)
                icon = f'<img class="fold" src="{cov}" />' if cov else '📁'
                label = f'<div class="itm">{icon}<span class="name">{p.name}</span></div>'
            else:
                try: mtime = p.stat().st_mtime
                except Exception: mtime = 0.0
                url = file_thumb_url(str(p), 96, mtime) if p.suffix.lower() in IMG_EXTS else None
                thumb = f'<img class="thumb" src="{url}" />' if url else '<div class="thumb" style="width:64px;height:64px;border:1px solid #e5e7eb;border-radius:8px;display:flex;align-items:center;justify-content:center;">🗎</div>'
                label = f'<div class="itm">{thumb}<span class="name">{p.name}</span></div>'
            src_items.append(label)
            label_to_path[label] = str(p)
    except Exception:
        pass

    # Приёмники (подпапки + вверх) — папки показываем обложкой
    subfolders = [p for p in curr.iterdir() if p.is_dir()]
    up_label = None
    if curr != parent_root:
        up_disp = curr.parent.name or str(curr.parent)
        up_label = f"⬆️ Вверх ({up_disp})"

    containers: List[Dict] = []
    containers.append({"header": "Текущая папка", "items": src_items})

    header_to_dir: Dict[str, Path] = {}
    if up_label:
        containers.append({"header": up_label, "items": []})
        header_to_dir[up_label] = curr.parent

    for f in sorted(subfolders, key=lambda x: x.name.lower()):
        try:
            mtime = f.stat().st_mtime
        except Exception:
            mtime = 0.0
        cover = folder_cover_url(str(f), 32, mtime)
        icon = f'<img class="fold" src="{cover}" />' if cover else '📁'
        hdr = f'<div class="hdr">{icon}<span>{f.name}</span></div>'
        containers.append({"header": hdr, "items": []})
        header_to_dir[hdr] = f

    # DnD
    result = sort_items(containers, multi_containers=True, direction="vertical")

    if st.button("Применить перемещения", key="apply_dnd", type="primary", use_container_width=True):
        if not result:
            st.info("Нет изменений."); return
        ok = skp = err = 0
        for cont in result:
            header = cont.get("header", "")
            dst_dir = header_to_dir.get(header)
            if not dst_dir:
                continue  # пропускаем контейнер источника
            for item_label in cont.get("items", []):
                src_str = label_to_path.get(item_label)
                if not src_str:
                    continue
                src = Path(src_str)
                try:
                    if not src.exists():
                        skp += 1; continue
                    # Нельзя переместить в ту же директорию
                    if src.parent.resolve() == dst_dir.resolve():
                        skp += 1; continue
                    # Нельзя переместить папку в саму себя / свой подкаталог
                    if src.is_dir():
                        try:
                            dst_dir.resolve().relative_to(src.resolve())
                            skp += 1; continue
                        except Exception:
                            pass
                    safe_move(src, dst_dir)
                    ok += 1
                except Exception as e:
                    st.error(f"Ошибка перемещения {src.name}: {e}")
                    err += 1
        st.success(f"Перемещено: {ok}; пропущено: {skp}; ошибок: {err}")
        st.rerun()


# ===== App =====

st.title("Face Sorter — Один проводник (DnD + превью)")

# LAN URL (информативно)
net_url = make_network_url()
st.info(f"Сетевой URL (LAN): {net_url}")
try:
    st.link_button("Открыть по сети", net_url, use_container_width=True)
except Exception:
    st.markdown(f"[Открыть по сети]({net_url})")

# Root selection
if "parent_path" not in st.session_state:
    st.session_state["parent_path"] = None
    st.session_state["current_dir"] = None

if st.session_state["parent_path"] is None:
    st.info("Выберите корневую папку.")
    c1, c2 = st.columns([0.30, 0.70])
    with c1:
        if st.button("📂 ВЫБРАТЬ ПАПКУ", type="primary", use_container_width=True):
            folder = pick_folder_dialog()
            if folder:
                st.session_state["parent_path"] = folder
                st.session_state["current_dir"] = folder
                st.rerun()
    with c2:
        manual = st.text_input("Или введите путь вручную", value="", placeholder="D:\\Папка\\Проект")
        if st.button("ОК", use_container_width=True):
            p = Path(manual)
            if manual and p.exists() and p.is_dir():
                st.session_state["parent_path"] = str(p)
                st.session_state["current_dir"] = str(p)
                st.rerun()
            else:
                st.error("Путь не существует или это не папка.")
else:
    curr = Path(st.session_state["current_dir"]).expanduser().resolve()
    parent_root = Path(st.session_state["parent_path"]).expanduser().resolve()

    # Навигация (breadcrumbs + Вверх)
    top_cols = st.columns([0.10, 0.90])
    with top_cols[0]:
        up = curr.parent if curr != parent_root else None
        st.button(
            "⬆️ Вверх",
            key="up",
            disabled=(up is None),
            on_click=(lambda p=str(up): st.session_state.update({"current_dir": p})) if up else None,
            use_container_width=True,
        )
    with top_cols[1]:
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
                st.button(
                    lbl or "/",
                    key=f"bc::{i}",
                    use_container_width=True,
                    on_click=(lambda p=str(tgt): st.session_state.update({"current_dir": p})),
                )

    st.markdown("---")

    # Единственный проводник (DnD + превью)
    render_explorer_dnd(curr, parent_root)
