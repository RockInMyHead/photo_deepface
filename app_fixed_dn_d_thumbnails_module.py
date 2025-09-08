# Drop-in for App Fixed: render_explorer_dnd with thumbnail items (files) and folder covers
# - Источник DnD: ВСЕ объекты текущей папки (файлы и папки)
# - Отображение: изображения — миниатюры 72×72; не-изображения — иконка; папки — обложка из первого изображения внутри
# - Цели DnD: подпапки текущей + «Вверх»
# - Перемещение: безопасный физический move (автопереименование при коллизиях), защита от перемещения папки в саму себя/подпапку
# Использование: просто замените существующую функцию render_explorer_dnd на эту версию

from pathlib import Path
from io import BytesIO
import base64
import shutil
import os
import streamlit as st

try:
    from streamlit_sortables import sort_items  # pip install streamlit-sortables
except Exception:
    sort_items = None

try:
    from PIL import Image
except Exception:
    Image = None

# IMG_EXTS — берём из вашего проекта, при отсутствии — дефолт
try:
    from core.cluster import IMG_EXTS  # type: ignore
except Exception:
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def _make_square_thumb(img_path: Path, size: int = 72) -> bytes | None:
    if Image is None:
        return None
    try:
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        if w != h:
            m = min(w, h)
            left = (w - m) // 2
            top = (h - m) // 2
            im = im.crop((left, top, left + m, top + m))
        im = im.resize((size, size))
        buf = BytesIO(); im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=1200)
def _thumb_b64(path_str: str, size: int, mtime: float) -> str | None:
    data = _make_square_thumb(Path(path_str), size)
    if not data:
        return None
    return base64.b64encode(data).decode("ascii")


@st.cache_data(show_spinner=False, ttl=1200)
def _folder_first_image(folder_str: str, folder_mtime: float) -> str | None:
    p = Path(folder_str)
    try:
        for child in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            if child.is_file() and _is_image(child):
                return str(child)
    except Exception:
        pass
    return None


def _folder_cover_b64(folder: Path) -> str | None:
    try:
        m = folder.stat().st_mtime
    except Exception:
        m = 0.0
    first = _folder_first_image(str(folder), m)
    if not first:
        return None
    try:
        return _thumb_b64(first, 72, Path(first).stat().st_mtime)
    except Exception:
        return None


def _unique_dst(dst_dir: Path, name: str, is_file: bool) -> Path:
    stem, ext = (Path(name).stem, Path(name).suffix) if is_file else (name, "")
    i = 0
    while True:
        suff = f" ({i})" if i > 0 else ""
        candidate = dst_dir / f"{stem}{suff}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


def _safe_move(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = _unique_dst(dst_dir, src.name, src.is_file())
    shutil.move(str(src), str(dst))
    return dst


def _extract_path_from_item(item_html: str) -> Path | None:
    """Вытаскиваем путь из скрытого маркера __PATH__…"""
    marker = "__PATH__"
    i = item_html.rfind(marker)
    if i == -1:
        return None
    rest = item_html[i + len(marker):]
    end = rest.find("</span>")
    path_str = rest if end == -1 else rest[:end]
    try:
        return Path(path_str)
    except Exception:
        return None


def _mk_item_label(path: Path) -> str:
    """HTML-элемент с миниатюрой + подписью и скрытым __PATH__ для обратного маппинга."""
    name_html = (
        f"<span style='font-size:12px;color:#334155;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:360px'>{path.name}</span>"
    )
    hidden = f"<span style='display:none'>__PATH__{str(path)}</span>"

    if path.is_dir():
        b64 = _folder_cover_b64(path)
        if b64:
            img_html = f"<img src='data:image/png;base64,{b64}' style='width:72px;height:72px;border-radius:8px;object-fit:cover;border:1px solid #e5e7eb'/>"
        else:
            img_html = "<div style='width:72px;height:72px;display:flex;align-items:center;justify-content:center;border:1px solid #e5e7eb;border-radius:8px;background:#fff'>📁</div>"
        return f"<div style='display:flex;align-items:center;gap:8px'>{img_html}{name_html}{hidden}</div>"

    # file
    if _is_image(path):
        try:
            b64 = _thumb_b64(str(path), 72, path.stat().st_mtime)
        except Exception:
            b64 = None
        if b64:
            img_html = f"<img src='data:image/png;base64,{b64}' style='width:72px;height:72px;border-radius:8px;object-fit:cover;border:1px solid #e5e7eb'/>"
        else:
            img_html = "<div style='width:72px;height:72px;display:flex;align-items:center;justify-content:center;border:1px solid #e5e7eb;border-radius:8px;background:#fff'>🖼️</div>"
    else:
        img_html = "<div style='width:72px;height:72px;display:flex;align-items:center;justify-content:center;border:1px solid #e5e7eb;border-radius:8px;background:#fff'>🗎</div>"

    return f"<div style='display:flex;align-items:center;gap:8px'>{img_html}{name_html}{hidden}</div>"


def render_explorer_dnd(curr: Path, parent_root: Path):
    """Drag-and-drop в стиле Windows Explorer с визуальными элементами.
    Drag source: все объекты в текущей папке (файлы + папки) с превью.
    Drop target: подпапки + "Вверх".
    """
    if sort_items is None:
        st.error("DnD требует пакет: pip install streamlit-sortables")
        return

    # Источники: ВСЁ содержимое текущей папки (файлы + папки). Без рекурсии.
    src_items = []
    try:
        for p in sorted(curr.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            src_items.append(_mk_item_label(p))
    except Exception:
        pass

    # Приёмники: подпапки + «Вверх» (без дублирования источников — это именно цели)
    subfolders = []
    try:
        subfolders = [p for p in curr.iterdir() if p.is_dir()]
    except Exception:
        pass

    up_label = None
    if curr != parent_root:
        up_disp = curr.parent.name or str(curr.parent)
        up_label = f"⬆️ Вверх ({up_disp})"

    containers = [{"header": "Текущая папка", "items": src_items}]
    if up_label:
        containers.append({"header": up_label, "items": []})
    for f in subfolders:
        containers.append({"header": f"📁 {f.name}", "items": []})

    # Рендер DnD
    result = sort_items(containers, multi_containers=True, direction="vertical")

    # Применяем перемещения
    if st.button("Применить перемещения", key="apply_dnd_thumbs", type="primary", use_container_width=True):
        header_to_dir = {f"📁 {f.name}": f for f in subfolders}
        if up_label:
            header_to_dir[up_label] = curr.parent

        ok = skp = err = 0
        for cont in (result or []):
            header = cont.get("header", "")
            dst_dir = header_to_dir.get(header)
            if not dst_dir:
                continue
            for item_html in cont.get("items", []):
                src = _extract_path_from_item(item_html)
                try:
                    if not src or not src.exists():
                        skp += 1; continue
                    if src.parent.resolve() == dst_dir.resolve():
                        skp += 1; continue

                    # Защита: не позволяем перемещать папку в саму себя или свой подкаталог
                    if src.is_dir():
                        try:
                            if dst_dir.resolve().is_relative_to(src.resolve()):
                                skp += 1; continue
                        except Exception:
                            # Python <3.9 не имеет is_relative_to у Path; делаем ручную проверку
                            try:
                                src_res = src.resolve(); dst_res = dst_dir.resolve()
                                if str(dst_res).startswith(str(src_res) + os.sep):
                                    skp += 1; continue
                            except Exception:
                                pass

                    _safe_move(src, dst_dir)
                    ok += 1
                except Exception as e:
                    st.error(f"Ошибка перемещения {src.name if src else '<?>'}: {e}")
                    err += 1
        st.success(f"Перемещено: {ok}; пропущено: {skp}; ошибок: {err}")
        st.rerun()
