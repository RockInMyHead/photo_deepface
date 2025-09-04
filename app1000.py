# app.py
# Minimal Explorer (revised): Refresh button, 150x150 thumbnails, richer logs after processing
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
        "match_thr": 0.52,          # калиброванный дефолт
        "top2_margin": 0.08,        # отступ между топ-1 и топ-2 по косинусу
        "per_person_min_obs": 10,   # резерв под перс.-пороги
        "min_det_score": 0.50,      # зарезервировано для core.cluster
        "min_quality": 0.50         # зарезервировано для core.cluster
    }
    if p.exists():
        try:
            user_cfg = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(user_cfg, dict):
                defaults.update({k: user_cfg[k] for k in user_cfg})
        except Exception:
            pass
    return defaults

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
    return {
        "group_counts": {},
        "global_stats": {"images_total": 0, "images_unknown_only": 0, "images_group_only": 0},
        "last_run": None
    }

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
    st.session_state.setdefault("proc_logs", [])         # detailed processing logs

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
    units = ["Б", "КБ", "МБ", "ГБ", "ТБ"]
    for u in units:
        if n < step or u == units[-1]:
            return f"{n:.0f} {u}" if u == "Б" else f"{n/1024:.1f} {u}"
        n /= step

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_person_index(group_dir: Path) -> Dict:
    p = group_dir / "person_index.json"
    data = {"persons": []}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return data

    # --- миграция структуры ---
    persons = data.get("persons", [])
    changed = False
    for person in persons:
        # было: {"number": int, "proto": [..]}
        if "protos" not in person:
            if "proto" in person:
                v = person.pop("proto")
                person["protos"] = [v]
                changed = True
            else:
                person["protos"] = []
                changed = True
        if "ema" not in person:
            if person["protos"]:
                person["ema"] = person["protos"][0]
            else:
                person["ema"] = None
            changed = True
        if "count" not in person:
            person["count"] = max(1, len(person.get("protos", [])))
            changed = True
        if "thr" not in person:
            person["thr"] = None
            changed = True

    if changed:
        try:
            (group_dir / "person_index.json").write_text(
                json.dumps({"persons": persons}, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    return {"persons": persons}

def save_person_index(group_dir: Path, data: Dict):
    (group_dir / "person_index.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def cleanup_processed_images(group_dir: Path, processed_images: Set[Path]):
    for img_path in processed_images:
        try:
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
    """
    person: {"number": int, "protos": [[..],..], "ema": [..] or None, "count": int, "thr": float|None}
    new_vec: np.ndarray|list — уже нормированная 1D
    """
    import numpy as np
    nv = _normalize_np(new_vec).tolist()
    protos = person.get("protos", [])
    protos.append(nv)

    # ограничиваем разнообразный набор прототипов до k_max
    if len(protos) > k_max:
        X = np.stack([_normalize_np(v) for v in protos], axis=0)  # [m, d]
        # greedy-покрытие по разбросу
        keep = [int(np.argmax(np.linalg.norm(X - X.mean(0), axis=1)))]
        while len(keep) < k_max:
            dists = np.min((1.0 - X @ X[keep].T), axis=1)  # косинусная "дистанция"
            cand = int(np.argmax(dists))
            if cand not in keep:
                keep.append(cand)
        protos = [X[i].tolist() for i in keep]

    # EMA
    ema = person.get("ema")
    ema_np = _normalize_np(ema if ema is not None else protos[0])
    ema_np = ema_alpha * ema_np + (1.0 - ema_alpha) * _normalize_np(new_vec)
    ema_np = _normalize_np(ema_np)

    person["protos"] = protos
    person["ema"] = ema_np.tolist()
    person["count"] = int(person.get("count", 0)) + 1

def match_and_apply(group_dir: Path, plan: Dict, match_thr: float) -> Tuple[int, Set[Path]]:
    import numpy as np

    CFG_LOCAL = CFG  # доступ к top2_margin и др.
    top2_margin = float(CFG_LOCAL.get("top2_margin", 0.08))

    person_idx = load_person_index(group_dir)
    persons = person_idx.get("persons", [])

    # --- центроиды кластеров ---
    raw_centroids = plan.get("cluster_centroids", {}) or {}
    centroids_norm = {}
    for cid, vec in raw_centroids.items():
        try:
            cid_int = int(cid)
        except Exception:
            cid_int = cid
        centroids_norm[cid_int] = _normalize_np(vec)

    # --- подготовка прототипов всех персон ---
    proto_list = []   # [M, d]
    proto_owner = []  # [M]
    per_thr = {}      # number -> персональный порог (если задан)

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

    if len(proto_list) > 0:
        P = np.stack(proto_list, axis=0)  # [M, d]
    else:
        P = None

    assigned: Dict[int, int] = {}     # cid -> number
    new_nums: Dict[int, int] = {}     # cid -> new number

    existing_nums = sorted([int(d.name) for d in group_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    cur_max = existing_nums[-1] if existing_nums else 0

    eligible = [int(c) if str(c).isdigit() else c for c in plan.get("eligible_clusters", [])]

    for cid in eligible:
        c = centroids_norm.get(cid)
        if c is None:
            continue

        if P is not None and len(P) > 0:
            sims = (P @ c.astype(np.float32))  # [M]

            # агрегируем по персоне: максимум по её прототипам
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

            thr_use = max(float(match_thr), float(per_thr.get(best_num, -1e9)))

            if (best_num is not None) and (s1 >= thr_use) and (s1 - s2 >= top2_margin):
                assigned[cid] = int(best_num)
                # обновим прототипы у найденной персоны
                for p in persons:
                    if int(p["number"]) == int(best_num):
                        _update_person_proto(p, c)
                        break
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

    # создаём папки под назначенные номера
    for cid, num in {**assigned, **new_nums}.items():
        ensure_dir(group_dir / str(num))

    # --- перенос изображений по кластерам ---
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
            dst = group_dir / str(num) / p.name
            if not dst.exists():
                try:
                    shutil.copy2(p, dst)
                except Exception:
                    pass
            copied.add(key)

    # групповые / неизвестные
    go = plan.get("group_only_images", []) or []
    if go:
        ensure_dir(group_dir / "__group_only__")
        for img in go:
            p = Path(img); dst = group_dir / "__group_only__" / p.name
            if not dst.exists():
                try:
                    shutil.copy2(p, dst)
                except Exception:
                    pass

    un = plan.get("unknown_images", []) or []
    if un:
        ensure_dir(group_dir / "__unknown__")
        for img in un:
            p = Path(img); dst = group_dir / "__unknown__" / p.name
            if not dst.exists():
                try:
                    shutil.copy2(p, dst)
                except Exception:
                    pass

    # сохраняем индекс персон
    person_idx["persons"] = persons
    save_person_index(group_dir, person_idx)

    # вычислим множество обработанных картинок (для cleanup)
    processed_images: Set[Path] = set()
    for cid in eligible:
        for img in cluster_images.get(cid, []):
            processed_images.add(Path(img))
    for img in go:
        processed_images.add(Path(img))
    for img in un:
        processed_images.add(Path(img))

    # пересечение с реальными файлами внутри group_dir
    all_in_group = set()
    for ext in IMG_EXTS:
        for f in group_dir.rglob(f"*{ext}"):
            if f.is_file():
                all_in_group.add(f)
    processed_images = processed_images.intersection(all_in_group)

    return len(persons), processed_images

def make_square_thumb(img_path: Path, size: int = 150):
    """Return a PIL.Image square thumbnail (size x size). Fallback: None."""
    if Image is None:
        return None
    try:
        im = Image.open(img_path).convert("RGB")
        # Center-crop to square
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

# ---- UI State ----
_init_state()

# ---- Step 1: Pick Folder ----
st.title("Face Sorter — Мини-проводник")

if st.session_state["parent_path"] is None:
    st.info("Выберите папку для работы.")
    if st.button("📂 ВЫБРАТЬ ПАПКУ", type="primary"):
        folder = pick_folder_dialog()
        if folder:
            st.session_state["parent_path"] = folder
            st.session_state["current_dir"] = folder
            st.rerun()
else:
    # Address / Up / Refresh
    curr = Path(st.session_state["current_dir"]).expanduser().resolve()
    parent_root = Path(st.session_state["parent_path"]).expanduser().resolve()

    top_cols = st.columns([0.08, 0.12, 0.80])
    with top_cols[0]:
        up = curr.parent if curr != curr.anchor else None
        st.button(
            "⬆️ Вверх",
            key="up",
            disabled=(up is None),
            on_click=(lambda: st.session_state.update({"current_dir": str(up)}) if up else None),
            width="stretch"
        )
    with top_cols[1]:
        if st.button("🔄 Обновить", width="stretch"):
            st.rerun()
    with top_cols[2]:
        crumbs = list(curr.parts)
        accum = Path(crumbs[0])
        bc_cols = st.columns(len(crumbs))
        for i, part in enumerate(crumbs):
            if i > 0:
                accum = accum / part
            with bc_cols[i]:
                st.button(
                    part or "/",
                    key=f"bc::{i}",
                    width="stretch",
                    on_click=lambda p=str(accum): st.session_state.update({"current_dir": p})
                )

    st.markdown("---")

    # Header row
    st.markdown('<div class="row hdr"><div>Превью</div><div>Имя</div><div>Тип</div><div>Изменён</div><div>Размер</div></div>', unsafe_allow_html=True)

    # Directory listing
    items = list_dir(curr)
    for item in items:
        is_dir = item.is_dir()
        sel_key = f"sel::{item}"
        name_btn_key = f"open::{item}"
        del_key = f"del::{item}"
        ren_key = f"ren::{item}"
        ren_input_key = f"ren_input::{item}"

        c1, c2, c3, c4, c5 = st.columns([0.14, 0.58, 0.12, 0.14, 0.10])

        # Preview (150x150 for images only)
        with c1:
            if not is_dir and item.suffix.lower() in IMG_EXTS:
                im = make_square_thumb(item, 150)
                if im is not None:
                    st.image(im)  # image already 150x150
                else:
                    st.image(str(item), width=150)  # fallback
            else:
                st.markdown('<div class="thumbbox">📁</div>' if is_dir else '<div class="thumbbox">🗎</div>', unsafe_allow_html=True)

        # Name + inline icons
        with c2:
            icon = "📁" if is_dir else "🗎"
            name_cols = st.columns([0.72, 0.10, 0.10, 0.08])
            with name_cols[0]:
                if is_dir:
                    if st.button(f"{icon} {item.name}", key=name_btn_key, width="stretch"):
                        st.session_state["current_dir"] = str(item)
                        st.rerun()
                else:
                    st.write(f"{icon} {item.name}")
            with name_cols[1]:
                if is_dir:
                    checked = st.checkbox("", key=sel_key, value=(str(item) in st.session_state["selected_dirs"]), help="В очередь")
                    if checked:
                        st.session_state["selected_dirs"].add(str(item))
                    else:
                        st.session_state["selected_dirs"].discard(str(item))
            with name_cols[2]:
                if st.button("✏️", key=ren_key, help="Переименовать", width="stretch"):
                    st.session_state["rename_target"] = str(item)
            with name_cols[3]:
                if st.button("🗑️", key=del_key, help="Удалить", width="stretch"):
                    try:
                        if is_dir:
                            shutil.rmtree(item, ignore_errors=True)
                        else:
                            item.unlink(missing_ok=True)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка удаления: {e}")

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

    st.markdown("---")

    # Footer actions
    colA, colB, colC = st.columns([0.35, 0.35, 0.30])
    with colA:
        if st.button("➕ Добавить в очередь", type="secondary", width="stretch"):
            added = 0
            for d in list(st.session_state["selected_dirs"]):
                if d not in st.session_state["queue"]:
                    st.session_state["queue"].append(d)
                    added += 1
            st.success(f"Добавлено в очередь: {added}")
    with colB:
        if st.button("🧹 Очистить очередь", width="stretch"):
            st.session_state["queue"] = []
            st.session_state["selected_dirs"] = set()
            st.info("Очередь очищена.")
    with colC:
        st.write(f"В очереди: {len(st.session_state['queue'])}")

    # Process
    if st.button("▶️ Обработать", type="primary"):
        parent = parent_root
        idx = load_index(parent)
        st.session_state["proc_logs"] = []  # reset

        # Targets = queue or all subfolders of current_dir
        targets = [Path(p) for p in st.session_state["queue"]]
        if not targets:
            targets = [p for p in curr.iterdir() if p.is_dir()]
        if not targets:
            st.warning("Нет целей для обработки.")
        else:
            tot_total = tot_unknown = tot_group_only = 0
            tot_faces = tot_unique_people = tot_joint = 0

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
                        # Derive per-group metrics
                        cluster_images = plan.get("cluster_images", {}) or {}
                        faces_detections = sum(len(v) for v in cluster_images.values())
                        unique_people_in_run = len(plan.get("eligible_clusters", []))
                        # joint images (same image belongs to >=2 clusters)
                        freq = {}
                        for imgs in cluster_images.values():
                            for pth in imgs:
                                freq[pth] = freq.get(pth, 0) + 1
                        joint_images = sum(1 for v in freq.values() if v >= 2)

                        persons_after, processed_images = match_and_apply(gdir, plan, match_thr=CFG["match_thr"])
                        idx["group_counts"][str(gdir)] = persons_after
                        cleanup_processed_images(gdir, processed_images)

                        # Aggregate stats
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

                    progress.progress(k / len(targets), text=f"{k}/{len(targets)}")

            idx["global_stats"]["images_total"] += tot_total
            idx["global_stats"]["images_unknown_only"] += tot_unknown
            idx["global_stats"]["images_group_only"] += tot_group_only
            save_index(parent, idx)
            st.session_state["queue"] = []

            st.success("Обработка завершена.")
            st.markdown("**Сводка за прогон:**")
            st.write(f"- Людей на фото (детекции): **{tot_faces}**")
            st.write(f"- Уникальных людей (кластера): **{tot_unique_people}**")
            st.write(f"- Общих фото (group_only): **{tot_group_only}**")
            st.write(f"- Совместных фото (>1 человек): **{tot_joint}**")

            st.markdown("**Детальные логи по группам:**")
            st.text_area("Логи", value="\n".join(st.session_state.get("proc_logs", [])), height=220)
