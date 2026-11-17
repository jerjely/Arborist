"""Re-run the analysis of cells from their saved project and .swc, without opening the images.

Used after an update that changes the measurements: every cell that has a surface and a tracing is
analysed again with the settings saved in its project. AIS values measured on the ankyrin scan are
carried over from the previous results (they need the image); everything else is recomputed.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import analysis, storage
from . import database as dbm
from .surface import BrainSurface
from .swc import Neuron


def _types(text, default):
    try:
        v = tuple(int(x) for x in str(text).replace(";", ",").split(",") if x.strip())
        return v or default
    except ValueError:
        return default


def reanalyse_cell(row, cover=True) -> tuple[str, str]:
    """('ok' | 'skipped' | 'failed', message)."""
    name = row.get("cell_name") or "?"
    try:
        proj = row.get("project")
        if not proj or not storage.exists(proj):
            return "skipped", "no saved project (surface not marked yet)"
        with open(storage.read(proj)) as fh:
            d = json.load(fh)
        cells = [s for s in d.get("stacks", []) if s.get("role") == "cell"]
        sd = next((s for s in cells if os.path.splitext(os.path.basename(s.get("path", "")))[0] == name),
                  cells[0] if cells else None)
        if sd is None or len(sd.get("surface_points_um") or []) < 2:
            return "skipped", "no surface line"
        if row.get("swc") and os.path.exists(row["swc"]):
            neuron = Neuron.read(row["swc"])
        elif sd.get("neuron"):
            neuron = Neuron(pd.DataFrame(sd["neuron"], columns=Neuron.COLS))
        else:
            return "skipped", "no tracing (.swc)"
        if not len(neuron):
            return "skipped", "the tracing is empty"
        side = dbm_radii(row["cell"])
        if side is not None:
            from .morphometry import apply_radii
            apply_radii(neuron, side)
        pts = np.asarray(sd["surface_points_um"], float)
        params = dict(sd.get("surface") or d.get("surface_params") or {})
        params.pop("method", None)
        xyz = neuron.xyz
        z_range = (min(pts[:, 2].min(), xyz[:, 2].min()) - 1.0, max(pts[:, 2].max(), xyz[:, 2].max()) + 1.0)
        surf = BrainSurface(pts, method="slices", cover_xyz=xyz if cover else None, z_range=z_range, **params)
        soma = neuron.df[neuron.df["type"] == 1]
        surf.orient_towards((soma if len(soma) else neuron.df)[["x", "y", "z"]].to_numpy(float))
        b = d.get("boundary", {})
        if b.get("manual"):
            boundary = float(b.get("manual_um") or 0) or None
        else:
            vals = [s["border_result"]["mean"] for s in d.get("stacks", [])
                    if s.get("border_result") and s.get("include", True)]
            boundary = float(np.mean(vals)) if vals else None
        an = d.get("analysis", {})
        out = os.path.join(os.path.dirname(row["cell"]), dbm.RESULTS_DIR)
        # AIS measured on the ankyrin scan cannot be recomputed without that image: keep it
        ais = None
        prev = os.path.join(out, f"{name}_summary.csv")
        if not storage.exists(prev) and row.get("results"):
            prev = row["results"]
        if storage.exists(prev):
            try:
                old = pd.read_csv(storage.read(prev)).iloc[-1]
                if str(old.get("ais_source", "")) == "AnkG stack":
                    ais = {k: (v.item() if hasattr(v, "item") else v) for k, v in old.items()
                           if k.startswith(("ais_", "acd_")) and k != "ais_normalized_depth" and pd.notna(v)}
            except Exception:
                ais = None
        analysis.run(neuron, surf, storage.out_dir(out), name, boundary_um=boundary,
                     dend_types=_types(an.get("dend"), (3, 4)), axon_types=_types(an.get("axon"), (2, 7)),
                     sholl_step_um=float(an.get("sholl_step", 5.0)), master_csv=None,
                     manual_type=str(an.get("manual_type", "") or "").strip(),
                     ais_type=int(an.get("ais_type", 7)), ais_points_xyz=None, ais_override=ais,
                     source_path=row["cell"])
        neuron.write(storage.out(os.path.join(out, f"{name}.swc")), f"source image: {row['cell']}")
        note = "" if boundary else " (no CR depth saved: laminar % not computed)"
        return "ok", f"analysed{note}"
    except Exception as ex:                        # one bad cell must not stop the others
        return "failed", f"{type(ex).__name__}: {ex}"


def dbm_radii(cell_path):
    """The diameters measured from the image for this cell, if any."""
    from .morphometry import radii_sidecar
    p = radii_sidecar(cell_path)
    if storage.exists(p):
        try:
            return pd.read_csv(storage.read(p))
        except Exception:
            return None
    return None


def reanalyse_all(rows, step=None):
    """[(cell name, status, message)] for every row with a 63x cell."""
    out = []
    todo = [r for r in rows if r.get("cell")]
    for k, r in enumerate(todo):
        if step:
            step(f"Re-analysing {k + 1} of {len(todo)}: {r.get('cell_name', '')}")
        st, msg = reanalyse_cell(r)
        out.append((r.get("cell_name", ""), st, msg))
    return out
