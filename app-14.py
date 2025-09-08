"""
Face Sorter — Мини‑проводник (Streamlit).

Этот скрипт реализует файловый проводник с поддержкой безопасного копирования
и перемещения изображений между папками, просмотром превью и перетаскиванием
файлов в подпапки. В отличие от предыдущих версий, в модуле
«Файлы (перетащите на папку ниже)» отображаются не текстовые пути
фотографий, а миниатюры самих изображений. Остальная логика приложения
сохранилась: выбор рабочей директории, навигация по папкам, очередь
директорий для пакетной обработки, переименование и удаление объектов,
экспорт результатов работы `build_plan` в структуру групп, а также
отображение сетевого URL для доступа по LAN.
"""

import base64
import json
import os
import shutil
import socket
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Set, Tuple

import streamlit as st

# Попытка импортировать опциональные зависимости: сортировки, корзину и PIL.
try:
    from streamlit_sortables import sort_items  # компонент для перетаскивания
except Exception:
    sort_items = None

try:
    from send2trash import send2trash  # мягкое удаление в корзину
except Exception:
    send2trash = None

try:
    from core.cluster import build_plan, IMG_EXTS  # логика обработки кластеров
except Exception:
    # Заглушка на случай отсутствия backend
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
                "images_group_only": 0,
            },
        }

try:
    from PIL import Image  # Pillow для миниатюр
except Exception:
    Image = None


###############################################################################
# Конфигурация страницы и глобальные константы
###############################################################################

st.set_page_config(page_title="Face Sorter — Мини‑проводник", layout="wide")

# CSS для таблицы проводника и элементов перетаскивания
st.markdown(
    """
    <style>
      .row { display: grid; grid-template-columns: 160px 1fr 110px 170px 120px; gap: 8px; align-items: center; padding: 6px 8px; border-bottom: 1px solid #f1f5f9; }
      .row:hover { background: #f8fafc; }
      .hdr { font-weight: 600; color: #334155; border-bottom: 1px solid #e2e8f0; }
      .thumbbox { width: 150px; height: 150px; display: flex; align-items: center; justify-content: center; border: 1px solid #e5e7eb; border-radius: 10px; overflow: hidden; background: #fff; }
      .crumb { border: 1px solid transparent; padding: 4px 8px; border-radius: 8px; }
      .crumb:hover { background: #f1f5f9; border-color: #e5e7eb; }
      /* Стили для превью в модуле DnD */
      .dnd-preview-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
      .dnd-preview-item { width: 96px; text-align: center; font-size: 12px; overflow: hidden; }
      .dnd-preview-item img { width: 96px; height: 96px; object-fit: cover; border-radius: 8px; }
      .dnd-preview-item .caption { margin-top: 4px; word-break: break-all; }
    </style>
    """,
    unsafe_allow_html=True,
)


###############################################################################
# Вспомогательные функции
###############################################################################

def _clamp(v, lo, hi, default):
    """Проверяет и ограничивает числовое значение в заданных границах."""
    try:
        v = type(default)(v)
        return max(lo, min(hi, v))
    except Exception:
        return default


def load_config(base: Path) -> Dict:
    """Читает конфигурацию из файла config.json и применяет ограничения."""
    p = base / "config.json"
    defaults = {
        "group_thr": 3,
        "eps_sim": 0.55,
        "min_samples": 2,
        "min_face": 110,
        "blur_thr": 45.0,
        "det_size": 640,
        "gpu_id": 0,
        "match_thr": 0.52,
        "top2_margin": 0.08,
        "per_person_min_obs": 10,
        "min_det_score": 0.50,
        "min_quality": 0.50,
    }
    if p.exists():
        try:
            user_cfg = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(user_cfg, dict):
                defaults.update({k: user_cfg[k] for k in user_cfg})
        except Exception:
            pass
    # Ограничения на значения
    defaults["eps_sim"] = _clamp(defaults["eps_sim"], 0.0, 1.0, 0.55)
    defaults["match_thr"] = _clamp(defaults["match_thr"], 0.0, 1.0, 0.52)
    defaults["top2_margin"] = _clamp(defaults["top2_margin"], 0.0, 1.0, 0.08)
    defaults["min_face"] = _clamp(defaults["min_face"], 0, 10000, 110)
    defaults["det_size"] = _clamp(defaults["det_size"], 64, 4096, 640)
    return defaults


# Глобальные параметры конфигурации
CFG_BASE = Path(__file__).parent
CFG = load_config(CFG_BASE)


def ensure_dir(p: Path):
    """Создаёт директорию, если она не существует."""
    p.mkdir(parents=True, exist_ok=True)


def load_index(parent: Path) -> Dict:
    """Загружает глобальный индекс с информацией о группах и статистике."""
    ensure_dir(parent)
    p = parent / "global_index.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "group_counts": {},
        "global_stats": {
            "images_total": 0,
            "images_unknown_only": 0,
            "images_group_only": 0,
        },
        "last_run": None,
    }


def _atomic_write(path: Path, text: str):
    """Пишет файл атомарно через временный файл."""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def save_index(parent: Path, idx: Dict):
    """Сохраняет глобальный индекс с обновлением времени запуска."""
    ensure_dir(parent)
    idx["last_run"] = datetime.now().isoformat(timespec="seconds")
    _atomic_write(parent / "global_index.json", json.dumps(idx, ensure_ascii=False, indent=2))


def get_lan_ip() -> str:
    """Пытается определить LAN IP-адрес текущей машины."""
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
    """Формирует URL для доступа к приложению по локальной сети."""
    ip = get_lan_ip()
    port = st.get_option("server.port") or 8501
    base = st.get_option("server.baseUrlPath") or ""
    base = ("/" + base.strip("/")) if base else ""
    return f"http://{ip}:{port}{base}"


def _init_state():
    """Инициализирует переменные в session_state."""
    st.session_state.setdefault("parent_path", None)
    st.session_state.setdefault("current_dir", None)
    st.session_state.setdefault("selected_dirs", set())
    st.session_state.setdefault("rename_target", None)
    st.session_state.setdefault("queue", [])
    st.session_state.setdefault("logs", [])
    st.session_state.setdefault("proc_logs", [])
    st.session_state.setdefault("delete_target", None)
    st.session_state.setdefault("file_order", {})  # запоминаем пользовательский порядок файлов


def log(msg: str):
    """Добавляет лог-сообщение с текущим временем."""
    st.session_state["logs"].append(
        f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    )


def list_dir(p: Path) -> List[Path]:
    """Возвращает отсортированный список содержимого директории.

    Папки идут первыми, затем файлы, сортировка по имени (без учета регистра).
    """
    try:
        items = list(p.iterdir())
    except Exception:
        return []
    items.sort(key=lambda x: (x.is_file(), x.name.lower()))
    return items


def human_size(n: int) -> str:
    """Форматирует размер файла в человекочитаемом виде."""
    units = ["Б", "КБ", "МБ", "ГБ", "ТБ", "ПБ"]
    size = float(n)
    for i, u in enumerate(units):
        if size < 1024 or i == len(units) - 1:
            return f"{size:.0f} {u}" if u == "Б" else f"{size:.1f} {u}"
        size /= 1024.0


def load_person_index(group_dir: Path) -> Dict:
    """Загружает person_index.json и приводит структуру к актуальному виду."""
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
        # Обеспечиваем наличие полей
        if "protos" not in person:
            if "proto" in person:
                v = person.pop("proto")
                person["protos"] = [v]
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
    """Сохраняет person_index.json атомарно."""
    _atomic_write(
        group_dir / "person_index.json",
        json.dumps(data, ensure_ascii=False, indent=2),
    )


def _is_subpath(p: Path, root: Path) -> bool:
    """Проверяет, что путь p находится внутри root."""
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def cleanup_processed_images(group_dir: Path, processed_images: Set[Path]):
    """Удаляет оригинальные изображения, которые были обработаны и перемещены.

    Защищены специальные папки __unknown__, __group_only__ и папки с номерами.
    """
    protected_roots = {group_dir / "__unknown__", group_dir / "__group_only__"}
    protected_roots |= {
        d for d in group_dir.iterdir() if d.is_dir() and d.name.isdigit()
    }
    for img_path in list(processed_images):
        try:
            if img_path.is_symlink():
                continue
            if any(_is_subpath(img_path, r) for r in protected_roots):
                continue
            if img_path.exists() and img_path.is_file():
                img_path.unlink()
                log(f"Удален оригинал: {img_path.name}")
        except Exception as e:
            log(f"Ошибка удаления {img_path.name}: {e}")


def _normalize_np(v):
    """Нормализует вектор до единичной длины (numpy)."""
    import numpy as np

    arr = np.array(v, dtype=np.float32)
    n = float(np.linalg.norm(arr) + 1e-12)
    return (arr / n).astype(np.float32)


def _update_person_proto(person: Dict, new_vec, k_max: int = 5, ema_alpha: float = 0.9):
    """Обновляет прототипы и экспоненциальное скользящее среднее для человека."""
    import numpy as np

    nv = _normalize_np(new_vec).tolist()
    protos = person.get("protos", [])
    protos.append(nv)
    # Ограничиваем количество прототипов
    if len(protos) > k_max:
        X = np.stack([_normalize_np(v) for v in protos], axis=0)
        # Удаляем наиболее похожие прототипы, оставляя разнообразие
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
    # Обновляем EMA
    ema = person.get("ema")
    ema_np = _normalize_np(ema if ema is not None else protos[0])
    ema_np = ema_alpha * ema_np + (1.0 - ema_alpha) * _normalize_np(new_vec)
    ema_np = _normalize_np(ema_np)
    person["protos"] = protos
    person["ema"] = ema_np.tolist()
    person["count"] = int(person.get("count", 0)) + 1


def safe_copy(src: Path, dst_dir: Path) -> Path:
    """Копирует файл, избегая коллизий имён."""
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
    """Перемещает файл, избегая коллизий имён."""
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
    """Сопоставляет центроиды кластеров с существующими людьми и копирует изображения.

    Возвращает количество людей после обновления и множество обработанных изображений.
    """
    import numpy as np

    top2_margin = float(CFG.get("top2_margin", 0.08))
    person_idx = load_person_index(group_dir)
    persons = person_idx.get("persons", [])

    # Нормализуем центроиды кластеров
    raw_centroids = plan.get("cluster_centroids", {}) or {}
    centroids_norm = {}
    for cid, vec in raw_centroids.items():
        try:
            cid_int = int(cid)
        except Exception:
            cid_int = cid
        centroids_norm[cid_int] = _normalize_np(vec)

    # Список прототипов и владельцев
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

    existing_nums = sorted(
        [int(d.name) for d in group_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    )
    cur_max = existing_nums[-1] if existing_nums else 0
    eligible = [int(c) if str(c).isdigit() else c for c in plan.get("eligible_clusters", [])]

    for cid in eligible:
        c = centroids_norm.get(cid)
        if c is None:
            continue
        if P is not None and len(P) > 0:
            sims = P @ c.astype("float32")
            sims = np.nan_to_num(sims, nan=-1.0)
            per_person_scores: Dict[int, float] = {}
            for s, owner in zip(sims.tolist(), proto_owner):
                if owner not in per_person_scores or s > per_person_scores[owner]:
                    per_person_scores[owner] = s
            if not per_person_scores:
                best_num = None
                s1 = -1.0
                s2 = -1.0
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
                persons.append(
                    {
                        "number": cur_max,
                        "protos": [c.tolist()],
                        "ema": c.tolist(),
                        "count": 1,
                        "thr": None,
                    }
                )
        else:
            cur_max += 1
            new_nums[cid] = cur_max
            persons.append(
                {
                    "number": cur_max,
                    "protos": [c.tolist()],
                    "ema": c.tolist(),
                    "count": 1,
                    "thr": None,
                }
            )

    # Создаём необходимые папки для новых людей
    for cid, num in {**assigned, **new_nums}.items():
        ensure_dir(group_dir / str(num))

    copied: Set[Tuple[int, Path]] = set()
    cluster_images = {}
    for k, v in (plan.get("cluster_images", {}) or {}).items():
        try:
            cluster_images[int(k)] = v
        except Exception:
            cluster_images[k] = v

    # Копируем изображения кластеров
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

    # Копируем group_only и unknown изображения
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

    # Ограничиваем processed_images только теми, что внутри папки группы
    def _iter_image_files(root: Path, exts: Set[str]):
        exts = {e.lower() for e in exts}
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            for fn in filenames:
                if Path(fn).suffix.lower() in exts:
                    yield Path(dirpath) / fn

    all_in_group = set(_iter_image_files(group_dir, IMG_EXTS))
    processed_images = processed_images.intersection(all_in_group)
    return len(persons), processed_images


def make_square_thumb(img_path: Path, size: int = 150):
    """Создаёт квадратную миниатюру изображения. Возвращает PIL.Image."""
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
    """Возвращает миниатюру для файла в виде байт. Используется кэширование."""
    if Image is None:
        return None
    im = make_square_thumb(Path(path_str), size)
    if im is None:
        return None
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data(show_spinner=False, ttl=3600)
def get_cover_thumb_bytes(folder_path_str: str, size: int, mtime: float):
    """Ищет первое изображение в папке и возвращает миниатюру.

    Не проходит рекурсивно: только текущий уровень.
    """
    p = Path(folder_path_str)
    if not p.is_dir():
        return None
    try:
        first_img = None
        for f in p.iterdir():
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                first_img = f
                break
        if first_img:
            return get_thumb_bytes(str(first_img), size, first_img.stat().st_mtime)
    except Exception:
        pass
    return None


def _sanitize_name(name: str) -> str:
    """Проверяет корректность имени файла/папки и нормализует его."""
    bad = {"/", "\\", ":", "*", "?", '"', "<", ">", "|"}
    name = (name or "").strip()
    if not name or any(ch in name for ch in bad) or name in {".", ".."}:
        raise ValueError("Неверное имя файла/папки.")
    return name


###############################################################################
# Рендер DnD для файлов
###############################################################################

def render_file_dnd(curr: Path, parent_root: Path):
    """Рендерит модуль перетаскивания файлов между папками с превью изображений."""
    if sort_items is None:
        st.error("DnD требует пакет: pip install streamlit-sortables")
        return

    # Собираем список только файлов из текущей директории
    src_files = [p for p in curr.iterdir() if p.is_file()]
    if not src_files:
        return
    # Генерируем превью для файлов
    previews = []
    for f in src_files:
        data = None
        if f.suffix.lower() in IMG_EXTS:
            try:
                data = get_thumb_bytes(str(f), 96, f.stat().st_mtime)
            except Exception:
                data = None
        previews.append((f, data))
    # Отображаем превью сеткой
    st.markdown("**📦 Файлы (перетащите на папку ниже)**")
    with st.container():
        cols = st.columns(min(len(previews), 5))
        for idx, (f, data) in enumerate(previews):
            col = cols[idx % len(cols)]
            with col:
                if data:
                    st.image(data, width=96)
                else:
                    st.markdown('<div class="thumbbox" style="width:96px;height:96px;display:flex;align-items:center;justify-content:center;border:1px solid #e5e7eb;border-radius:8px;background:#fff;font-size:32px;">🗎</div>', unsafe_allow_html=True)
                st.caption(f.name)

    # Формируем контейнеры для sort_items
    containers = []
    # Источник: список файлов в текущей директории
    containers.append({"header": "Файлы", "items": [str(f) for f in src_files]})
    # Приёмник: Вверх
    up_label = None
    if curr != parent_root:
        up_disp = curr.parent.name or str(curr.parent)
        up_label = f"⬆️ Вверх ({up_disp})"
        containers.append({"header": up_label, "items": []})
    # Приёмники: подпапки текущей директории
    for d in curr.iterdir():
        if d.is_dir():
            containers.append({"header": f"📁 {d.name}", "items": []})

    result = sort_items(containers, multi_containers=True, direction="vertical")
    # Кнопка переноса
    if st.button("Перенести", key=f"dnd_files_apply::{str(curr)}", type="primary", use_container_width=True):
        # Сопоставляем заголовок контейнера с реальной директорией
        header_to_dir: Dict[str, Path] = {}
        if up_label:
            header_to_dir[up_label] = curr.parent
        for d in curr.iterdir():
            if d.is_dir():
                header_to_dir[f"📁 {d.name}"] = d
        ok = 0
        skp = 0
        err = 0
        for cont in (result or []):
            header = cont.get("header", "")
            dst_dir = header_to_dir.get(header)
            if not dst_dir:
                continue
            for src_str in cont.get("items", []):
                src = Path(src_str)
                try:
                    if not src.exists():
                        skp += 1
                        continue
                    if src.parent.resolve() == dst_dir.resolve():
                        skp += 1
                        continue
                    # Запрещаем перемещение папок в себя
                    if src.is_dir() and _is_subpath(dst_dir, src):
                        skp += 1
                        continue
                    safe_move(src, dst_dir)
                    ok += 1
                except Exception as e:
                    err += 1
                    st.error(f"Не удалось переместить {src.name}: {e}")
        st.success(f"Перемещено: {ok}; пропущено: {skp}; ошибок: {err}")
        # Перерисовываем после переноса
        st.rerun()


###############################################################################
# Основной UI
###############################################################################

def main():
    """Точка входа приложения."""
    _init_state()

    st.title("Face Sorter — Мини‑проводник")
    # LAN URL
    net_url = get_network_url()
    st.info(f"Сетевой URL (LAN): {net_url}")
    try:
        st.link_button("Открыть по сети", net_url, use_container_width=True)
    except Exception:
        st.markdown(f"[Открыть по сети]({net_url})")
    st.text_input("Network URL", value=net_url, label_visibility="collapsed")

    # Если папка не выбрана
    if st.session_state["parent_path"] is None:
        st.info("Выберите папку для работы.")
        pick_cols = st.columns([0.25, 0.75])
        with pick_cols[0]:
            if st.button("📂 ВЫБРАТЬ ПАПКУ", type="primary", use_container_width=True):
                folder = None
                # Попытка открыть системный диалог (работает в Windows с GUI)
                try:
                    import tkinter as tk
                    from tkinter import filedialog
                    root = tk.Tk()
                    root.withdraw()
                    root.attributes("-topmost", True)
                    folder = filedialog.askdirectory(title="Выберите папку")
                    root.destroy()
                except Exception:
                    folder = None
                if folder:
                    st.session_state["parent_path"] = folder
                    st.session_state["current_dir"] = folder
                    st.rerun()
        with pick_cols[1]:
            manual = st.text_input(
                "Или введите путь вручную",
                value="",
                placeholder="D:\\Папка\\Проект",
            )
            if st.button("ОК", use_container_width=True):
                if manual and Path(manual).exists():
                    st.session_state["parent_path"] = manual
                    st.session_state["current_dir"] = manual
                    st.rerun()
                else:
                    st.error("Путь не существует.")
        return

    # Навигация
    curr = Path(st.session_state["current_dir"]).expanduser().resolve()
    parent_root = Path(st.session_state["parent_path"]).expanduser().resolve()
    # Песочница: не выходим за root
    if not _is_subpath(curr, parent_root):
        st.session_state["current_dir"] = str(parent_root)
        curr = parent_root

    # Верхняя панель: вверх, обновить, хлебные крошки
    top_cols = st.columns([0.08, 0.12, 0.80])
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
        if st.button("🔄 Обновить", use_container_width=True):
            st.rerun()
    with top_cols[2]:
        # Хлебные крошки
        labels = [parent_root.name if parent_root.name else parent_root.anchor or "/"]
        targets = [parent_root]
        try:
            rel = curr.resolve().relative_to(parent_root.resolve())
            acc = parent_root
            for part in rel.parts:
                acc = acc / part
                labels.append(part)
                targets.append(acc)
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

    # Контейнер высотой 700px: DnD + проводник
    with st.container(height=700):
        # Модуль DnD для файлов
        render_file_dnd(curr, parent_root)
        st.markdown("---")

        # Проводник: список всех объектов (папок и файлов)
        items = list_dir(curr)
        for item in items:
            is_dir = item.is_dir()
            sel_key = f"sel::{item}"
            name_btn_key = f"open::{item}"
            del_key = f"del::{item}"
            ren_key = f"ren::{item}"
            ren_input_key = f"ren_input::{item}"

            c1, c2, c3, c4, c5 = st.columns([0.14, 0.58, 0.12, 0.14, 0.10])

            # Превью
            with c1:
                thumb_data = None
                if is_dir:
                    try:
                        thumb_data = get_cover_thumb_bytes(str(item), 150, item.stat().st_mtime)
                    except Exception:
                        thumb_data = None
                    if thumb_data:
                        st.image(thumb_data)
                    else:
                        st.markdown('<div class="thumbbox">📁</div>', unsafe_allow_html=True)
                else:
                    if item.suffix.lower() in IMG_EXTS:
                        try:
                            thumb_data = get_thumb_bytes(str(item), 150, item.stat().st_mtime)
                        except Exception:
                            thumb_data = None
                        if thumb_data:
                            st.image(thumb_data)
                        else:
                            st.image(str(item), width=150)
                    else:
                        st.markdown('<div class="thumbbox">🗎</div>', unsafe_allow_html=True)

            # Имя и inline‑действия
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

            # Тип
            with c3:
                st.write("Папка" if is_dir else (item.suffix[1:].upper() if item.suffix else "Файл"))
            # Дата изменения
            with c4:
                try:
                    st.write(datetime.fromtimestamp(item.stat().st_mtime).strftime("%Y-%m-%d %H:%M"))
                except Exception:
                    st.write("—")
            # Размер
            with c5:
                if is_dir:
                    st.write("—")
                else:
                    try:
                        st.write(human_size(item.stat().st_size))
                    except Exception:
                        st.write("—")

            # Переименование
            if st.session_state.get("rename_target") == str(item):
                rc1, rc2, rc3 = st.columns([0.70, 0.15, 0.15])
                with rc1:
                    new_name_val = st.text_input(
                        "Новое имя",
                        value=item.name,
                        key=ren_input_key,
                        label_visibility="collapsed",
                    )
                with rc2:
                    if st.button("Сохранить", key=f"save::{item}", use_container_width=True):
                        try:
                            new_name = _sanitize_name(new_name_val)
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

            # Удаление
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
                            st.session_state["delete_target"] = None
                with dc3:
                    if st.button("Отмена", key=f"cancel_del::{item}", use_container_width=True):
                        st.session_state["delete_target"] = None

    st.markdown("---")

    # Нижняя панель: очередь и обработка
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

    # Запуск обработки
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
            st.text_area(
                "Логи",
                value="\n".join(st.session_state.get("proc_logs", [])),
                height=220,
            )


if __name__ == "__main__":
    main()