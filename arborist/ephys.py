"""Slice electrophysiology table (Excel / CSV) -> extra per-cell features for classification.

The table is free-form: one row per cell, some columns that identify the cell (date, section,
side, cell number or a file name) and numeric measurements (Rin, Vrest, sag, AP threshold ...).
Rows are matched to analysed cells through the cell's image path, which carries the day /
Section / Left|Right folders and the file name.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re

import numpy as np
import pandas as pd

PREFIX = "ephys_"

_ID_PATTERNS = [
    ("date", r"date|d[aá]tum|^day$|^nap$|experiment|k[ií]s[eé]rlet"),
    ("section", r"section|slice|metszet|szelet|^sec"),
    ("side", r"side|hemi|oldal|^l/?r$"),
    ("cell", r"cell|sejt|neuron|file|name|^id$|n[eé]v|f[aá]jl|recording"),
]
_STOP = {"cell", "cells", "section", "sec", "slice", "s", "c", "n", "sejt", "metszet", "szelet", "no", "nr",
         "side", "hemisphere", "stitch", "stitched", "czi", "tif", "tiff", "swc", "x", "the", "of", "and"}
_SIDE = {"l": "left", "left": "left", "bal": "left", "lh": "left", "b": "left",
         "r": "right", "right": "right", "jobb": "right", "rh": "right", "j": "right"}
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
          "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15, "xvi": 16, "xvii": 17, "xviii": 18,
          "xix": 19, "xx": 20}
_CELLNUM = re.compile(r"(?:cell|sejt|neuron|\bc|\bn)[\s_\-#]*0*(\d+)", re.I)
_MAG = re.compile(r"\d+\s*x\b", re.I)


# ------------------------------------------------------------------ reading
def sheet_names(path):
    if path.lower().endswith((".xlsx", ".xlsm", ".xls")):
        return list(pd.ExcelFile(path).sheet_names)
    return []


def _to_number(s: pd.Series) -> pd.Series:
    """Numbers stored as text, including Hungarian decimal commas ('0,52')."""
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return s
    t = s.astype(str).str.strip().str.replace("−", "-", regex=False)
    t = t.str.replace(r"(?<=\d)\s+(?=\d)", "", regex=True).str.replace(",", ".", regex=False)
    t = t.replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan, "n/a": np.nan, "NA": np.nan})
    num = pd.to_numeric(t, errors="coerce")
    empty = {"", "-", "–", "—", "n/a", "na", "nan", "none", "?", "x"}
    filled = s.notna() & ~s.astype(str).str.strip().str.lower().isin(empty)
    if filled.sum() and num[filled].notna().mean() >= 0.8:
        return num
    return s


def read_table(path: str, sheet=None) -> pd.DataFrame:
    """Read the sheet, find the header row (skipping title rows) and an optional units row."""
    if path.lower().endswith((".xlsx", ".xlsm", ".xls")):
        raw = pd.read_excel(path, sheet_name=sheet if sheet not in (None, "") else 0, header=None)
    else:
        with open(path, encoding="utf-8-sig", errors="replace") as fh:
            head = fh.read(4096)
        sep = ";" if head.count(";") > head.count(",") else ("\t" if head.count("\t") > head.count(",") else ",")
        raw = pd.read_csv(path, header=None, sep=sep, dtype=str, encoding="utf-8-sig")
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if raw.empty:
        return pd.DataFrame()

    def is_text(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return False
        if isinstance(v, (int, float, np.number, _dt.date, _dt.datetime, pd.Timestamp)):
            return False
        sv = str(v).strip()
        return bool(sv) and pd.isna(pd.to_numeric(sv.replace(",", "."), errors="coerce"))

    hdr = 0
    for i in range(min(15, len(raw) - 1)):
        row = raw.iloc[i]
        filled = row.notna().sum()
        if filled >= max(2, 0.5 * raw.shape[1]) and sum(is_text(v) for v in row) >= 0.6 * filled:
            hdr = i
            break
    names = [str(v).strip() if pd.notna(v) else f"col{j + 1}" for j, v in enumerate(raw.iloc[hdr])]
    body = raw.iloc[hdr + 1:]
    # a units row right under the header: short text in most columns, e.g. "mV", "MΩ", "pA"
    if len(body) > 1:
        u = body.iloc[0]
        units = [str(v).strip() for v in u if pd.notna(v)]
        nxt = body.iloc[1]
        if units and all(is_text(v) for v in u if pd.notna(v)) and max(len(x) for x in units) <= 8 \
                and any(pd.notna(v) and not is_text(v) for v in nxt):
            names = [f"{n} ({str(v).strip()})" if pd.notna(v) and str(v).strip() else n for n, v in zip(names, u)]
            body = body.iloc[1:]
    seen = {}
    uniq = []
    for n in names:
        k = seen.get(n, 0); seen[n] = k + 1
        uniq.append(n if k == 0 else f"{n}.{k}")
    t = pd.DataFrame(body.to_numpy(), columns=uniq).dropna(how="all").reset_index(drop=True)
    for c in t.columns:
        t[c] = _to_number(t[c].infer_objects())
    return t


# ------------------------------------------------------------------ columns
def guess_id_columns(t: pd.DataFrame):
    out = []
    for c in t.columns:
        name = str(c).lower()
        if re.search(r"type|t[ií]pus|class|note|megjegyz|comment", name):
            continue                       # cell-type / notes columns are labels, not identifiers
        if pd.api.types.is_datetime64_any_dtype(t[c]) or any(re.search(p, name) for _, p in _ID_PATTERNS):
            out.append(c)
        elif not pd.api.types.is_numeric_dtype(t[c]) and t[c].nunique() > 0.6 * len(t):
            out.append(c)          # a text column with near-unique values is probably a name
    return out


def column_role(name: str):
    n = str(name).lower()
    for role, p in _ID_PATTERNS:
        if re.search(p, n):
            return role
    return "other"


def feature_columns(t: pd.DataFrame, id_cols):
    return [c for c in t.columns if c not in id_cols and pd.api.types.is_numeric_dtype(t[c])
            and t[c].notna().sum() >= 1 and not _numeric_label(t, c)]


KINDS = {"ephys": "Electrophysiology", "invivo": "In vivo light responses"}
_LABELISH = re.compile(r"cluster|klaszter|type|t[ií]pus|class|label|group|csoport", re.I)


def prefix_of(kind="ephys"):
    return kind + "_"


def feature_name(col: str, prefix: str = PREFIX) -> str:
    s = re.sub(r"[^\w]+", "_", str(col), flags=re.UNICODE).strip("_")
    return prefix + (s or "value")


# ------------------------------------------------------------------ tokens
def _date_variants(v):
    if isinstance(v, (pd.Timestamp, _dt.datetime, _dt.date)):
        d = pd.Timestamp(v)
        return {d.strftime(f) for f in ("%Y%m%d", "%y%m%d", "%d%m%Y", "%d%m%y", "%m%d%Y", "%m%d%y")}
    return set()


def _split(text: str):
    text = _MAG.sub(" ", str(text))
    out = set()
    for part in re.findall(r"[A-Za-zÀ-ſ]+|\d+", text):
        p = part.lower()
        if p.isdigit():
            p = p.lstrip("0") or "0"
        elif p in _ROMAN and part.isupper():          # "III. right" = section 3 (only capital numerals)
            p = str(_ROMAN[p])
        p = _SIDE.get(p, p)
        if p not in _STOP:
            out.add(p)
    return out


def value_tokens(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return set()
    dv = _date_variants(v)
    if dv:
        return {"date:" + x for x in dv}
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        v = int(v)
    s = str(v).strip()
    digits = re.sub(r"\D", "", s)
    if re.fullmatch(r"\d{2,4}[.\-/_]\d{1,2}[.\-/_]\d{1,4}\.?", s) or (len(digits) >= 6 and digits == s):
        try:
            d = pd.to_datetime(s.rstrip("."), dayfirst=not re.match(r"\d{4}", s))
            return {"date:" + x for x in _date_variants(d)} | {"date:" + digits}
        except Exception:
            return {"date:" + digits}
    return _split(s)


_DATE_RE = re.compile(r"(?<!\d)(20\d\d|\d\d)[._\-]?(0?[1-9]|1[0-2])[._\-]?(0?[1-9]|[12]\d|3[01])(?!\d)")


def _dates_in(text):
    """Dates written in folder / file names: 2026.07.15, 2026-07-15, 20260715, 202607.15, 260715 ...
    (other numbers in the name, e.g. GCaMP3 or Bl6, are ignored)."""
    out = set()
    for m in _DATE_RE.finditer(str(text)):
        y, mo, d = m.groups()
        y = int(y) + (2000 if len(y) == 2 else 0)
        try:
            out |= {"date:" + v for v in _date_variants(pd.Timestamp(year=y, month=int(mo), day=int(d)))}
        except ValueError:
            pass
    return out


def cell_signature(name: str, path: str | None):
    """Tokens of the cell: its name, the file stem and the last folders of its path."""
    parts = []
    if path:
        p = os.path.normpath(str(path)).replace("\\", "/").split("/")
        parts = [x for x in p[-6:-1] if x]
        stem = os.path.splitext(p[-1])[0]
    else:
        stem = str(name)
    toks = set()
    for x in parts + [stem, str(name)]:
        toks |= _split(x)
        toks |= _dates_in(x)
    # the section and side of the cell, from the folders (Section 3 / left) or the file name (section3_left)
    sec = side = None
    for x in parts[::-1] + [stem]:
        mm = re.search(r"(?:section|slice|metszet|szelet|sec)[\s_\-]*0*(\d+)", x, re.I)
        if mm and sec is None:
            sec = mm.group(1)
        if side is None:
            for tok in re.findall(r"[A-Za-z]+", x):
                if tok.lower() in ("left", "right", "bal", "jobb"):
                    side = _SIDE[tok.lower()]
    m = _CELLNUM.findall(_MAG.sub(" ", stem + " " + str(name)))
    if m:
        num = m[-1].lstrip("0") or "0"
    else:
        nums = re.findall(r"\d+", _MAG.sub(" ", stem))
        nums = [n for n in nums if len(n) < 6]
        num = (nums[-1].lstrip("0") or "0") if nums else None
    return dict(tokens=toks, cellnum=num, section=sec, side=side, stem=re.sub(r"[^a-z0-9]", "", stem.lower()),
                name=re.sub(r"[^a-z0-9]", "", str(name).lower()))


def row_key(row, id_cols):
    vals = []
    for c in id_cols:
        v = row.get(c)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, (pd.Timestamp, _dt.datetime)):
            v = pd.Timestamp(v).strftime("%Y-%m-%d")
        elif isinstance(v, float) and v.is_integer():
            v = int(v)
        vals.append(str(v).strip())
    return " · ".join(vals)


def row_keys(t, id_cols):
    keys, seen = [], {}
    for _, r in t.iterrows():
        k = row_key(r, id_cols) or "(empty)"
        n = seen.get(k, 0); seen[k] = n + 1
        keys.append(k if n == 0 else f"{k} #{n + 1}")
    return keys


def _row_matches(row, id_cols, sig):
    """True if every identifying value of the row is found in the cell's signature."""
    used = False
    for c in id_cols:
        v = row.get(c)
        if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
            continue
        role = column_role(c)
        toks = value_tokens(v)
        if role == "cell":
            s = re.sub(r"[^a-z0-9]", "", str(v).lower())
            if re.search(r"[a-z]", s) and len(s) >= 6:          # a file / cell name
                if not (s in sig["stem"] or sig["stem"] in s or s in sig["name"] or sig["name"] in s):
                    return False
                used = True
                continue
            nums = re.findall(r"\d+", _MAG.sub(" ", str(v) if not isinstance(v, float) else str(int(v))
                                               if float(v).is_integer() else str(v)))
            if nums:
                if sig["cellnum"] is None or (nums[-1].lstrip("0") or "0") != sig["cellnum"]:
                    return False
                used = True
                continue
        if not toks:
            continue
        if role in ("section", "side") and (sig.get("section") or sig.get("side")):
            # compare with the cell's own section number and side, not with any number in the path
            nums = {x for x in toks if x.isdigit()}
            sides = {x for x in toks if x in ("left", "right")}
            if nums and sig.get("section") is not None and nums != {sig["section"]}:
                return False
            if sides and sig.get("side") is not None and sides != {sig["side"]}:
                return False
            if not (nums or sides):
                return False                  # e.g. "thrown away": no section / side -> no match
            used = True
            continue
        if any(t.startswith("date:") for t in toks):
            if not (toks & sig["tokens"]):
                return False
        elif not toks <= sig["tokens"]:
            return False
        used = True
    return used


def useful_id_columns(t, id_cols, sigs):
    """Identifying columns whose values are found in at least some cell's image path. A column that
    matches nothing (e.g. the ephys recording file number, which is not in the image names) would block
    every match, so it is left out."""
    keep = []
    for c in id_cols:
        hit = 0
        for _, r in t.iterrows():
            v = r.get(c)
            if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
                continue
            if any(_row_matches(r, [c], sg) for sg in sigs.values()):
                hit += 1
                break
        if hit:
            keep.append(c)
    return keep if keep else list(id_cols)


def _label_text(v):
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        v = int(v)                                    # cluster 3, not "3.0"
    return str(v).strip()


def _numeric_label(t, c):
    """A numbered grouping (e.g. a 'cluster' column from the in vivo analysis): a label, not a measurement."""
    if not pd.api.types.is_numeric_dtype(t[c]) or not _LABELISH.search(str(c)):
        return False
    v = t[c].dropna()
    return len(v) > 0 and (v % 1 == 0).all() and v.nunique() <= max(12, len(v) // 2)


def label_columns(t, id_cols):
    """Columns that can label cells in figures: text (e.g. 'cell type based on articles') or a numbered
    grouping such as a cluster id."""
    out = []
    for c in t.columns:
        if c not in id_cols and _numeric_label(t, c):
            out.append(c)
            continue
        if c in id_cols or pd.api.types.is_numeric_dtype(t[c]) or pd.api.types.is_datetime64_any_dtype(t[c]):
            continue
        vals = t[c].dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        if len(vals) >= 2 and vals.nunique() <= max(12, len(vals) // 2):
            out.append(c)
    return out


def auto_match(cells: pd.DataFrame, t: pd.DataFrame, id_cols):
    """{cell name: row index or None}. A row that fits several cells (or a cell several rows)
    is left unmatched so it can be set by hand."""
    if not id_cols or t.empty:
        return {c: None for c in cells["cell"].astype(str)}
    sigs = {str(r["cell"]): cell_signature(r["cell"], r.get("source_path")) for _, r in cells.iterrows()}
    id_cols = useful_id_columns(t, id_cols, sigs)
    hits = {c: [] for c in sigs}
    for i, r in t.iterrows():
        for c, sg in sigs.items():
            if _row_matches(r, id_cols, sg):
                hits[c].append(i)
    used = {}
    for c, h in hits.items():
        if len(h) == 1:
            used.setdefault(h[0], []).append(c)
    out = {}
    for c, h in hits.items():
        out[c] = h[0] if len(h) == 1 and len(used.get(h[0], [])) == 1 else None
    return out


# ------------------------------------------------------------------ link file
def link_path(master_csv, kind="ephys"):
    return os.path.splitext(master_csv)[0] + f"_{kind}_link.json"


def load_link(master_csv, kind="ephys"):
    from . import storage
    p = storage.read(link_path(master_csv, kind))
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def save_link(master_csv, link, kind="ephys"):
    from . import storage
    with open(storage.out(link_path(master_csv, kind)), "w", encoding="utf-8") as fh:
        json.dump(link, fh, indent=1, ensure_ascii=False)


def resolve(cells, t, link):
    """Final {cell: row index or None} from auto matching plus manual choices (stored as row keys)."""
    id_cols = [c for c in link.get("id_cols", []) if c in t.columns]
    m = auto_match(cells, t, id_cols)
    keys = row_keys(t, id_cols)
    by_key = {k: i for i, k in enumerate(keys)}
    for c, k in (link.get("manual") or {}).items():
        if c in m:
            m[c] = by_key.get(k) if k else None
    return m, keys


def merge(cells: pd.DataFrame, master_csv: str, link: dict | None = None, kind: str = "ephys"):
    """cells + <kind>_* columns (ephys_… for the slice sheet, invivo_… for a light-response table).
    Returns (table, info dict)."""
    PREFIX = prefix_of(kind)
    link = link or load_link(master_csv, kind)
    if not link or not os.path.exists(link.get("path", "")):
        return cells, dict(linked=False)
    t = read_table(link["path"], link.get("sheet"))
    id_cols = [c for c in link.get("id_cols", []) if c in t.columns]
    m, keys = resolve(cells, t, link)
    fcols = feature_columns(t, id_cols)
    names = {c: feature_name(c, PREFIX) for c in fcols}
    out = cells.copy()
    for c in fcols:
        out[names[c]] = [t.at[m[str(n)], c] if m.get(str(n)) is not None else np.nan for n in out["cell"]]
    for c in label_columns(t, id_cols):
        out[PREFIX + "label_" + feature_name(c, "")] = [
            _label_text(t.at[m[str(n)], c]) if m.get(str(n)) is not None and pd.notna(t.at[m[str(n)], c]) else ""
            for n in out["cell"]]
    out[PREFIX + "row"] = [keys[m[str(n)]] if m.get(str(n)) is not None else "" for n in out["cell"]]
    matched = sum(v is not None for v in m.values())
    rows_used = {v for v in m.values() if v is not None}
    return out, dict(linked=True, matched=matched, n_cells=len(out), n_rows=len(t),
                     unmatched_rows=[keys[i] for i in range(len(t)) if i not in rows_used],
                     features=list(names.values()), path=link["path"])
