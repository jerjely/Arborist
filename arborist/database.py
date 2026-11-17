"""Experiment database: scans the data folder tree and reports what exists / what is done.

Expected layout (names are flexible, levels are not):

    <root>/<day folder>/<section folder>/<10x whole-section scan>
                                        /<left|right>/<20x Calretinin stack>
                                                     /<63x cell stack(s)>  (+ AnkG channel)
                                                     /<tracing>.swc
                                                     /<cell>_SC_morph.json   (Arborist project)
                                                     /SC_morph_results/<cell>_summary.csv

Magnification comes from the .czi objective metadata; for TIFFs from the file
name (``10x``, ``20x``, ``63x``) or, failing that, from the pixel size.
"""
from __future__ import annotations

import glob
import json

from . import storage
import os
import re

IMG_EXT = (".czi", ".tif", ".tiff")
DB_NAME = "SC_morph_database.json"
RESULTS_DIR = "SC_morph_results"
SIDES = {"left": "left", "right": "right", "bal": "left", "jobb": "right"}
SKIP_DIRS = {RESULTS_DIR.lower()}


def side_of(name: str):
    n = name.lower()
    for k, v in SIDES.items():
        if re.search(rf"(^|[^a-z]){k}([^a-z]|$)", n):
            return v
    return None


def _dirs(p):
    try:
        return sorted(d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))
                      and not d.startswith(".") and d.lower() not in SKIP_DIRS)
    except OSError:
        return []


def _files(p, exts):
    try:
        return sorted(os.path.join(p, f) for f in os.listdir(p)
                      if f.lower().endswith(exts) and os.path.isfile(os.path.join(p, f)))
    except OSError:
        return []


# ------------------------------------------------------------- magnification
def _mag_from_name(path):
    m = re.search(r"(?<!\d)(5|10|20|25|40|60|63|100)\s*[xX](?![a-wyzA-WYZ])", os.path.basename(path))
    return float(m.group(1)) if m else None


def _mag_from_czi(path):
    import czifile
    with czifile.CziFile(path) as czi:
        sc = czi.scenes
        img = sc[list(sc.keys())[0]] if hasattr(sc, "keys") else sc[0]
        obj = getattr(img, "objective", None) or {}
        mag = obj.get("NominalMagnification") if isinstance(obj, dict) else None
        if mag:
            return float(mag), obj.get("Name", "")
        mpp = getattr(img, "mpp", None)
        return _mag_from_pixel(mpp[0] if mpp else None), "pixel size"


def _mag_from_pixel(px_um):
    """Very rough fallback (Zeiss confocal, zoom 1): >0.45 um -> 10x, >0.2 -> 20x, else 63x."""
    if not px_um:
        return None
    return 10.0 if px_um > 0.45 else 20.0 if px_um > 0.2 else 63.0


def _mag_from_tif(path):
    import tifffile
    with tifffile.TiffFile(path) as t:
        try:
            xr = t.pages[0].tags["XResolution"].value
            px = xr[1] / xr[0]
            unit = (t.imagej_metadata or {}).get("unit", "")
            if unit in ("micron", "um", "µm", "\\u00B5m"):
                return _mag_from_pixel(px)
        except Exception:
            pass
    return None


def magnification(path, cache: dict):
    """(mag, source). Cached by path + size + mtime."""
    try:
        st = os.stat(path)
    except OSError:
        return None, "missing"
    key = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    if key in cache:
        return tuple(cache[key])
    mag, src = None, ""
    try:
        if path.lower().endswith(".czi"):
            mag, src = _mag_from_czi(path)
    except Exception as e:
        src = f"unreadable ({e.__class__.__name__})"
    if mag is None:
        mag = _mag_from_name(path)
        src = "file name" if mag else src
    if mag is None and path.lower().endswith((".tif", ".tiff")):
        try:
            mag = _mag_from_tif(path)
            src = "pixel size" if mag else src
        except Exception:
            pass
    cache[key] = [mag, src]
    return mag, src


# ------------------------------------------------------------------ status
def _find_summary(cell_path):
    """A <name>_summary.csv in the cell's results folder made from this 63x image, whatever name the
    analysis was given (it records the image it came from in its source_path column)."""
    import csv
    d = os.path.join(os.path.dirname(cell_path), RESULTS_DIR)
    want = os.path.basename(cell_path).lower()
    seen = set()
    for folder in (d, os.path.dirname(storage.store().local_for(os.path.join(d, "x")))):
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if not f.endswith("_summary.csv") or f.startswith("all_cells") or "_clusters" in f or f in seen:
                continue
            seen.add(f)
            try:
                with open(os.path.join(folder, f), newline="", encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
            except Exception:
                continue
            if rows and os.path.basename(str(rows[-1].get("source_path", ""))).lower() == want:
                return os.path.join(d, f)
    return None


_FILE_CACHE = {}          # (kind, path, extra) -> (mtime_ns, size, value): files are re-read only when they change


def _cached(kind, path, extra, compute):
    try:
        st = os.stat(path)
        sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        return compute()
    key = (kind, path, extra)
    hit = _FILE_CACHE.get(key)
    if hit and hit[:2] == sig:
        return hit[2]
    val = compute()
    _FILE_CACHE[key] = (*sig, val)
    return val


def _project_status(proj_path, cell_name):
    actual = storage.read(proj_path)
    return dict(_cached("project", actual, cell_name, lambda: _project_status_read(actual, cell_name)))


def _project_status_read(actual, cell_name):
    st = {"surface": False, "cr": False, "ais_pts": False}
    try:
        with open(actual) as fh:
            d = json.load(fh)
    except Exception:
        return st
    for s in d.get("stacks", []):
        if s.get("role") == "cell" and os.path.splitext(os.path.basename(s.get("path", "")))[0] == cell_name:
            st["surface"] = len(s.get("surface_points_um") or []) >= 4
            st["ais_pts"] = len(s.get("ais_points_um") or []) >= 2
        if s.get("role") == "ankg" and len(s.get("ais_points_um") or []) >= 2:
            st["ais_pts"] = True
        if s.get("border_result"):
            st["cr"] = True
    if d.get("boundary", {}).get("manual"):
        st["cr"] = True
    return st


def _swc_has_type(path, t=7):
    return _cached("swc_type", path, t, lambda: _swc_has_type_read(path, t))


def _swc_has_type_read(path, t=7):
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2 and parts[1] == str(t):
                    return True
    except OSError:
        pass
    return False


def _best_match(stem, candidates, n_cells=1):
    """File belonging to the cell `stem`: names must start with each other (SNT exports
    '<image name>.swc' or '<image name>-000.swc'); with one cell in the folder, a single
    candidate is accepted whatever its name."""
    if not candidates:
        return None
    a = stem.lower()
    named = [p for p in candidates
             if os.path.splitext(os.path.basename(p))[0].lower().startswith(a)
             or a.startswith(os.path.splitext(os.path.basename(p))[0].lower())]
    if named:
        return min(named, key=lambda p: len(os.path.basename(p)))
    if n_cells == 1 and len(candidates) == 1:
        return candidates[0]
    return None


def project_path_for(cell_stack_path):
    stem = os.path.splitext(os.path.basename(cell_stack_path))[0]
    return os.path.join(os.path.dirname(cell_stack_path), f"{stem}_SC_morph.json")


ANKG_KEYS = ("ankyrin", "ankg", "ank-g", "ank_g", "ais")
RECON_KEYS = ("_recon",)
EXTRA_KEYS = ("axon", "dendrite", "soma", "full", "fill", "colou", "color", "merge", "overlay", "composite")
FILL_PARTS = {"soma": "soma", "dendrites": "dendrite", "axon": "axon"}   # z-projections of the fill


def _has(name, keys):
    n = os.path.basename(name).lower()
    for k in keys:
        if len(k) <= 3:      # short keys (e.g. "ais") must be a separate word
            if re.search(rf"(^|[^a-z]){re.escape(k)}([^a-z]|$)", n):
                return True
        elif k in n:
            return True
    return False


def section_overview(spath, cache):
    """10x whole-section scan: a 'whole section' folder, otherwise the section folder itself."""
    whole = [d for d in _dirs(spath) if "whole" in d.lower()]
    imgs = []
    for d in whole:
        imgs += _files(os.path.join(spath, d), IMG_EXT)
    if whole:
        return imgs, "whole section folder"
    imgs = []
    for f in _files(spath, IMG_EXT):
        name = os.path.basename(f).lower()
        if "10x" in name or "whole" in name:
            imgs.append(f)
        elif "20x" not in name and (magnification(f, cache)[0] or 10) <= 12:
            imgs.append(f)
    return imgs, "section folder"


def section_cr(spath, cache, cr_mag_range=(15.0, 39.0)):
    """20x of the whole SC (both hemispheres): an 'SC' folder, or a 20x scan directly in the
    section folder. Returns (files, source)."""
    imgs = []
    for d in _dirs(spath):
        if d.lower() == "sc" or re.match(r"^sc[ _-]", d.lower()):
            imgs += _files(os.path.join(spath, d), IMG_EXT)
    if imgs:
        return imgs, "SC folder"
    for f in _files(spath, IMG_EXT):
        name = os.path.basename(f).lower()
        if "10x" in name or "whole" in name:
            continue
        m = magnification(f, cache)[0]
        if (m and cr_mag_range[0] <= m <= cr_mag_range[1]) or re.search(r"(?<!\d)20\s*x", name):
            imgs.append(f)
    return imgs, ("section folder" if imgs else "")


def classify_side(path, cache, cell_min_mag=40.0, cr_mag_range=(15.0, 39.0)):
    """Sort the files of a Left/Right folder.

    .czi files are scans; .tif files are derived images (``_recon.tif`` marks a finished
    reconstruction, axon/dendrites/full.tif are extra fills) - unless the folder has no
    .czi at all, then TIFFs are treated as scans too.
    """
    files = _files(path, IMG_EXT)
    for sub in _dirs(path):
        files += _files(os.path.join(path, sub), IMG_EXT)
    czis = [f for f in files if f.lower().endswith(".czi")]
    tifs = [f for f in files if not f.lower().endswith(".czi")]
    recon = [f for f in tifs if _has(f, RECON_KEYS)]
    scans = czis if czis else [f for f in tifs if not _has(f, RECON_KEYS + EXTRA_KEYS)]
    extras = [f for f in tifs if f not in recon and f not in scans]
    fills = {}
    for part, key in FILL_PARTS.items():
        hit = [f for f in tifs if os.path.splitext(os.path.basename(f))[0].lower().startswith(key)]
        if hit:
            fills[part] = min(hit, key=lambda p: len(os.path.basename(p)))
    comp = [f for f in tifs if "composite" in os.path.basename(f).lower()]
    out = dict(ankg=[], stitch=[], original=[], cr=[], unknown=[], recon=recon, extras=extras,
               fills=fills, composite=comp[0] if comp else None)
    for f in scans:
        if _has(f, ANKG_KEYS):
            out["ankg"].append(f)
        elif "stitch" in os.path.basename(f).lower():
            out["stitch"].append(f)
        else:
            m = magnification(f, cache)[0]
            if m and m >= cell_min_mag:
                out["original"].append(f)
            elif m and cr_mag_range[0] <= m <= cr_mag_range[1]:
                out["cr"].append(f)
            else:
                out["unknown"].append(f)
    return out


def find_surface_txt(path):
    """Surface marked in Fiji (point tool table), e.g. SURFACE.txt, in a Left/Right folder."""
    for f in _files(path, (".txt", ".csv", ".tsv")):
        n = os.path.basename(f).lower()
        if n.startswith("surface"):
            return f
    return None


def scan(root: str, cache: dict | None = None):
    """One row per cell (a Stitch file, else an original 63x scan); sides/sections
    without a cell still get a row so that missing scans show up."""
    cache = {} if cache is None else cache
    rows = []
    for day in _dirs(root):
        dpath = os.path.join(root, day)
        for sec in _dirs(dpath):
            spath = os.path.join(dpath, sec)
            overview, ov_src = section_overview(spath, cache)
            sc_cr, sc_src = section_cr(spath, cache)
            sides = [d for d in _dirs(spath) if side_of(d)]
            base = dict(day=day, section=sec, overview=overview, overview_source=ov_src)
            if not sides:
                rows.append({**base, "side": "", "side_path": "", "cr": sc_cr, "cell": None,
                             "key": os.path.relpath(spath, root)})
                continue
            for sd in sides:
                path = os.path.join(spath, sd)
                k = classify_side(path, cache)
                cr = sc_cr + k["cr"]
                cr_src = sc_src if sc_cr else ("side folder" if k["cr"] else "")
                swcs = [p for p in glob.glob(os.path.join(path, "**", "*.swc"), recursive=True)
                        if RESULTS_DIR not in p and not p.lower().endswith("_autotrace.swc")]   # drafts do not count
                surf_txt = find_surface_txt(path)
                cells = k["stitch"] or k["original"]
                common = dict(base, side=side_of(sd), side_folder=sd, side_path=path, cr=cr, cr_source=cr_src,
                              surface_txt=surf_txt,
                              ankg=k["ankg"], recon=None, originals=k["original"], extras=k["extras"],
                              fills=k["fills"], composite=k["composite"],
                              unknown=k["unknown"])
                if not cells:
                    rows.append({**common, "cell": None, "swc": None, "project": None,
                                 "key": os.path.relpath(path, root)})
                    continue
                for c in cells:
                    stem = os.path.splitext(os.path.basename(c))[0]
                    same_dir = [s for s in swcs if os.path.dirname(s) == os.path.dirname(c)]
                    swc = _best_match(stem, same_dir or swcs, len(cells))
                    recon = _best_match(stem, k["recon"], len(cells))
                    ankg = k["ankg"] if len(cells) == 1 else [a for a in k["ankg"] if _best_match(stem, [a], 2)]
                    proj = project_path_for(c)
                    if not storage.exists(proj):
                        js = [j for j in glob.glob(os.path.join(os.path.dirname(c), "*_SC_morph.json"))]
                        by_stem = {j[:-len("_SC_morph.json")] + ".json": j for j in js}
                        m = _best_match(stem, list(by_stem), len(cells))
                        proj = by_stem[m] if m else None
                    pst = _project_status(proj, stem) if proj else {}
                    res = os.path.join(os.path.dirname(c), RESULTS_DIR, f"{stem}_summary.csv")
                    if not storage.exists(res):
                        res = _find_summary(c) or res
                    rows.append({**common, "cell": c, "cell_name": stem, "swc": swc,
                                 "recon": recon or (k["fills"].get("dendrites") or k["fills"].get("axon")
                                                    or k["fills"].get("soma") if len(cells) == 1 else None),
                                 "ankg": ankg, "project": proj,
                                 "surface": pst.get("surface", False) or bool(surf_txt),
                                 "cr_done": pst.get("cr", False),
                                 "ais": bool(pst.get("ais_pts")) or bool(swc and _swc_has_type(swc, 7)),
                                 "analyzed": storage.exists(res), "results": res if storage.exists(res) else None,
                                 "key": os.path.relpath(c, root)})
    return rows


# ------------------------------------------------------------ persistence
def load_db(root):
    p = os.path.join(root, DB_NAME)
    try:
        with open(storage.read(p)) as fh:
            d = json.load(fh)
    except Exception:
        d = {}
    d.setdefault("notes", {}); d.setdefault("cache", {})
    return d


def save_db(root, db):
    p = storage.out(os.path.join(root, DB_NAME))
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(db, fh, indent=1)
    os.replace(tmp, p)


COLUMNS = ["Day", "Section", "Side", "10x section", "20x CR", "63x cell", "AnkG scan", "Tracing (.swc)",
           "Recon tif", "AIS measured", "Surface", "CR border", "Analyzed", "Notes"]


SHORT = ["Day", "Section", "Side", "10x", "20x CR", "63x cell", "AnkG", "SWC", "Recon", "AIS",
         "Surface", "CR", "Done", "Notes"]


def row_cells(r):
    """Display values; True/False/None = done / missing / not applicable."""
    has_side = bool(r.get("side"))
    has_cell = bool(r.get("cell"))
    return [
        r["day"], r["section"], r.get("side") or "no Left/Right folder",
        bool(r.get("overview")),
        bool(r.get("cr")),
        os.path.basename(r["cell"]) if has_cell else (False if has_side else None),
        bool(r.get("ankg")) if has_cell else None,
        bool(r.get("swc")) if has_cell else None,
        bool(r.get("recon")) if has_cell else None,
        r.get("ais") if has_cell else None,
        r.get("surface") if has_cell else None,
        r.get("cr_done") if has_cell else None,
        r.get("analyzed") if has_cell else None,
    ]


def tooltips(r):
    """Per-column hover text (file lists)."""
    j = lambda xs: "\n".join(xs or [])
    return {0: r["day"], 3: f"{r.get('overview_source', '')}\n{j(r.get('overview'))}".strip(),
            4: f"{r.get('cr_source', '')}\n{j(r.get('cr'))}".strip(),
            5: j([r["cell"]] if r.get("cell") else []) + ("\noriginal: " + j(r.get("originals")) if r.get("originals") else ""),
            6: j(r.get("ankg")), 7: r.get("swc") or "",
            8: j(([r["recon"]] if r.get("recon") else []) + [p for p in (r.get("fills") or {}).values() if p != r.get("recon")]),
            10: ("Fiji: " + r["surface_txt"]) if r.get("surface_txt") else "",
            2: ("extra fills: " + ", ".join(os.path.basename(x) for x in r.get("extras", []))) if r.get("extras") else ""}


def to_table(rows, notes):
    import pandas as pd
    out = []
    for r in rows:
        v = row_cells(r)
        v = [("yes" if x is True else "MISSING" if x is False else "" if x is None else x) for x in v]
        out.append(v + [notes.get(r["key"], "")])
    return pd.DataFrame(out, columns=COLUMNS)


# ------------------------------------------------------------- fill preview
def _read_projection(path, max_side=4000):
    """2D image (max over z if it is a stack), strided down to <= max_side, without loading
    the whole file when it can be memory-mapped."""
    import numpy as np
    import tifffile
    try:
        arr = tifffile.memmap(path, mode="r")
    except Exception:
        arr = tifffile.imread(path)
    arr = np.asarray(arr) if arr.ndim <= 2 else arr
    if arr.ndim > 2:
        if arr.shape[-1] in (3, 4) and arr.ndim == 3:          # RGB(A)
            arr = arr[..., :3].max(axis=-1)
        else:
            while arr.ndim > 3:
                arr = arr.max(axis=0) if arr.shape[0] < arr.shape[1] else arr[0]
            s = max(1, int(np.ceil(max(arr.shape[-2:]) / max_side)))
            return np.asarray(arr[:, ::s, ::s]).max(axis=0)
    s = max(1, int(np.ceil(max(arr.shape[-2:]) / max_side)))
    return np.asarray(arr[::s, ::s])


FILL_RGB = {"soma": (0.62, 0.62, 0.62), "dendrites": (1.0, 0.15, 0.15), "axon": (0.15, 1.0, 0.15)}


def fill_composite(row, max_px=2048):
    """RGB uint8 image from the fill projections: soma grey, dendrites red, axon green."""
    import numpy as np
    from PIL import Image
    imgs = {k: _read_projection(p) for k, p in _fill_paths(row).items()}
    shape = tuple(np.max([v.shape for v in imgs.values()], axis=0))
    rgb = np.zeros((*shape, 3), np.float32)
    for k in ("soma", "dendrites", "axon"):           # axon drawn last, on top
        if k not in imgs:
            continue
        v = _norm01(imgs[k])
        if v.shape != shape:
            v = np.asarray(Image.fromarray(v).resize((shape[1], shape[0])))
        col = np.array(FILL_RGB[k], np.float32)
        rgb = np.maximum(rgb, v[..., None] * col)      # brightest wins, colours stay pure
    out = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    s = max(1, int(np.ceil(max(out.shape[:2]) / max_px)))
    return out[::s, ::s]


def make_preview(row, size=360):
    """Colour preview of the fill: soma grey, dendrites red, axon green (cached PNG).

    Falls back to an RGB composite TIFF in the folder. Returns the PNG path or None.
    """
    import numpy as np
    from PIL import Image
    fills = row.get("fills") or {}
    srcs = list(fills.values()) or ([row["composite"]] if row.get("composite") else [])
    if not srcs or not row.get("side_path"):
        return None
    name = row.get("cell_name") or "side"
    out = os.path.join(row["side_path"], RESULTS_DIR, f"{name}_fill_preview.png")
    newest = max(os.path.getmtime(p) for p in srcs)
    cached = storage.read(out)
    if os.path.exists(cached) and os.path.getmtime(cached) >= newest:
        return cached
    def norm(a):
        a = a.astype(np.float32)
        lo, hi = np.percentile(a[::4, ::4], [0.5, 99.8])
        return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    if _fill_paths(row):
        rgb = fill_composite(row, max_px=1024)
    else:
        import tifffile
        a = tifffile.imread(row["composite"])
        if a.ndim == 3 and a.shape[0] in (3, 4):
            a = np.moveaxis(a, 0, -1)
        rgb = (norm(a[..., :3]) * 255).astype(np.uint8) if a.dtype != np.uint8 else a[..., :3]
    im = Image.fromarray(rgb)
    im.thumbnail((size, size), Image.LANCZOS)
    loc = storage.out(out)
    im.save(loc)
    return loc


POOLED_NAME = "all_cells_summary.csv"


def pooled_path(root):
    return os.path.join(root, RESULTS_DIR, POOLED_NAME)


def collect_summaries(root):
    """One table with every analysed cell under `root`: all <cell>_summary.csv files in the
    SC_morph_results folders (including copies still waiting on this computer). A cell analysed more
    than once (same image) keeps its newest analysis. Returns (table, list of files used)."""
    import pandas as pd
    found = {}

    def add(real, actual):
        try:
            found[real] = (os.path.getmtime(actual), actual)
        except OSError:
            pass
    skip = (POOLED_NAME, "_clusters", "_table_with_ephys")
    for base, mapping in ((root, None), (storage.store().local_for(root), root)):
        if not os.path.isdir(base):
            continue
        for d, dirs, files in os.walk(base):
            if os.path.basename(d) != RESULTS_DIR:
                continue
            for f in files:
                if f.endswith("_summary.csv") and not any(k in f for k in skip):
                    actual = os.path.join(d, f)
                    real = actual if mapping is None else os.path.join(mapping, os.path.relpath(actual, base))
                    if mapping is None or real not in found or os.path.getmtime(actual) > found[real][0]:
                        add(real, storage.read(real) if mapping is None else actual)
    rows = []
    for real, (mt, actual) in found.items():
        try:
            t = pd.read_csv(actual)
        except Exception:
            continue
        if not len(t):
            continue
        t = t.iloc[[-1]].copy()
        t["_mtime"] = mt
        t["_summary_file"] = real
        rows.append(t)
    if not rows:
        return pd.DataFrame(), []
    T = pd.concat(rows, ignore_index=True, sort=False)
    key = T["source_path"].fillna("").astype(str) if "source_path" in T.columns else pd.Series([""] * len(T))
    key = key.where(key != "", T["cell"].astype(str))
    T["_key"] = key.str.lower()
    T = T.sort_values("_mtime").drop_duplicates("_key", keep="last").drop(columns=["_key", "_mtime"])
    # where each cell comes from (day / section / side folders): the columns other tables can be joined on
    if "source_path" in T.columns:
        parts = T["source_path"].fillna("").astype(str).str.replace("\\", "/", regex=False).str.split("/")
        for col, k in (("day", -4), ("section", -3), ("side", -2)):
            T.insert(1, col, [p_[k] if len(p_) >= -k else "" for p_ in parts]) if col not in T.columns else None
        T = T[["cell", "day", "section", "side"] + [c for c in T.columns if c not in ("cell", "day", "section", "side")]]
    return T.reset_index(drop=True), sorted(T["_summary_file"])


CHANNEL_RGB = [(1, 0, 1), (0, 1, 0), (1, 1, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0), (0, 0, 1)]   # as the viewer


def _norm01(a):
    import numpy as np
    a = a.astype(np.float32)
    sub = a[:: max(1, a.shape[0] // 512), :: max(1, a.shape[1] // 512)]
    lo, hi = np.percentile(sub, [0.5, 99.8])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def _composite(chans):
    """(C, Y, X) -> RGB uint8 with the viewer's channel colours (one channel: grey)."""
    import numpy as np
    if len(chans) == 1:
        g = _norm01(chans[0])
        return (np.repeat(g[..., None], 3, axis=2) * 255).astype(np.uint8)
    rgb = np.zeros((*chans[0].shape, 3), np.float32)
    for c, a in enumerate(chans):
        rgb += _norm01(a)[..., None] * np.array(CHANNEL_RGB[c % len(CHANNEL_RGB)], np.float32)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _fill_paths(row):
    fills = row.get("fills") or {}
    return {k: v for k, v in fills.items() if k in ("soma", "dendrites", "axon") and v and os.path.exists(v)}


def projection_source(row):
    """('fills', newest fill file) when soma / dendrites / axon projections exist, else ('recon', path) when a
    _recon.tif exists, else ('stitch', path of the 63x stack), else (None, None)."""
    f = _fill_paths(row)
    if f:
        return "fills", max(f.values(), key=os.path.getmtime)
    rec = row.get("recon")
    if rec and "recon" in os.path.basename(rec).lower() and os.path.exists(rec):
        return "recon", rec
    if row.get("cell") and os.path.exists(row["cell"]):
        return "stitch", row["cell"]
    return None, None


def projection_png(row):
    kind, src = projection_source(row)
    if not src:
        return None
    name = row.get("cell_name") or os.path.splitext(os.path.basename(src))[0]
    return os.path.join(os.path.dirname(src), RESULTS_DIR, f"{name}_{kind}_projection.png")


def cached_projection(row):
    """Path of an up-to-date cached projection, or None."""
    kind, src = projection_source(row)
    out = projection_png(row)
    if not out:
        return None
    p = storage.read(out)
    newest = max(os.path.getmtime(x) for x in _fill_paths(row).values()) if kind == "fills" \
        else os.path.getmtime(src)
    if os.path.exists(p) and os.path.getmtime(p) >= newest:
        return p
    return None


def make_projection(row, max_px=2048, progress=None):
    """Max projection of the reconstruction (_recon.tif) or, if there is none yet, of the 63x stitch,
    reduced to <= max_px and coloured like the viewer; cached as a PNG next to the cell.
    Reads one plane at a time, so big stitched .czi files never load whole. Returns (png, kind)."""
    import numpy as np
    from PIL import Image
    kind, src = projection_source(row)
    if not src:
        return None, None
    hit = cached_projection(row)
    if hit:
        return hit, kind
    if kind == "fills":
        rgb = fill_composite(row, max_px)
    elif kind == "recon":
        import tifffile
        a = tifffile.imread(src)
        if a.ndim == 2:
            rgb = _composite([a])
        elif a.ndim == 3 and a.shape[-1] in (3, 4):                 # RGB(A) image
            rgb = a[..., :3] if a.dtype == np.uint8 else (_norm01(a[..., :3]) * 255).astype(np.uint8)
        elif a.ndim == 4 and a.shape[-1] in (3, 4):                 # RGB stack
            m = a[..., :3].max(axis=0)
            rgb = m if m.dtype == np.uint8 else (_norm01(m) * 255).astype(np.uint8)
        elif a.ndim == 3:
            # (C, Y, X) with few channels, or (Z, Y, X)
            rgb = _composite(list(a)) if a.shape[0] <= 4 and a.shape[0] < a.shape[1] // 8 else _composite([a.max(0)])
        else:                                                        # (Z, C, Y, X)
            m = a.reshape(-1, *a.shape[-3:]).max(axis=0)
            rgb = _composite(list(m))
        s = max(1, int(np.ceil(max(rgb.shape[:2]) / max_px)))
        rgb = rgb[::s, ::s]
    else:
        from .io import load_stack
        st = load_stack(src, lazy=True)
        d = st.data
        C, Z, Y, X = d.shape
        s = max(1, int(np.ceil(max(Y, X) / max_px)))
        acc = None
        for z in range(Z):
            for c in range(C):
                pl = d[c, z]
                pl = np.asarray(pl.compute() if hasattr(pl, "compute") else pl)[::s, ::s]
                if acc is None:
                    acc = np.zeros((C, *pl.shape), pl.dtype)
                np.maximum(acc[c], pl, out=acc[c])
            if progress:
                progress(z + 1, Z)
        rgb = _composite(list(acc))
    out = projection_png(row)
    loc = storage.out(out)
    Image.fromarray(rgb).save(loc)
    return loc, kind


def thumbnail(row, size=160, compute_stitch=False):
    """Small picture of the cell for figures: reconstruction projection, else the fill preview, else a
    cached stitch projection (computed only if compute_stitch). RGB uint8 array or None."""
    import numpy as np
    from PIL import Image
    p = None
    kind, _ = projection_source(row)
    if kind in ("fills", "recon") or (kind == "stitch" and cached_projection(row)):
        try:
            p, _ = make_projection(row)
        except Exception:
            p = None
    if p is None:
        try:
            p = make_preview(row)                  # coloured fills: soma / dendrites / axon
        except Exception:
            p = None
    if p is None and kind == "stitch" and compute_stitch:
        try:
            p, _ = make_projection(row)
        except Exception:
            p = None
    if not p:
        return None
    im = Image.open(p).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    return np.asarray(im)
