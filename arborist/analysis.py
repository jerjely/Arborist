"""Depth / laminar analysis of a traced neuron relative to the brain surface
and the Calretinin (sensory SC) border."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .swc import Neuron, TYPE_COLORS
from . import style as S
S.apply_rc()
from .surface import BrainSurface


def compartment_of(t: int, dend_types, axon_types) -> str:
    if t == 1:
        return "soma"
    if t in axon_types:
        return "axon"
    if t in dend_types:
        return "dendrite"
    return "other"


def soma_center(neuron: Neuron) -> np.ndarray:
    df = neuron.df
    s = df[df["type"] == 1]
    if len(s):
        return s[["x", "y", "z"]].to_numpy(float).mean(0)
    root = df[df["parent"] == -1]
    r = root if len(root) else df
    return r[["x", "y", "z"]].to_numpy(float)[0]


def node_table(neuron: Neuron, surf: BrainSurface, boundary_um=None,
               dend_types=(3, 4), axon_types=(2,)) -> pd.DataFrame:
    df = neuron.df.copy()
    xyz = df[["x", "y", "z"]].to_numpy(float)
    uvn = surf.plane_coords(xyz)
    df["compartment"] = [compartment_of(t, dend_types, axon_types) for t in df["type"]]
    df["depth_mesh_um"] = surf.signed_depth(xyz)          # Method 1
    df["depth_plane_um"] = uvn[:, 2]                      # Method 2
    df["z_tilt_corrected_um"] = uvn[:, 1]                 # Method 3
    df["along_surface_um"] = surf.along_surface(xyz)
    df["within_surface_footprint"] = surf.footprint_fraction(xyz)
    if boundary_um:
        df["depth_below_CR_border_um"] = df["depth_mesh_um"] - boundary_um
        df["normalized_depth"] = df["depth_mesh_um"] / boundary_um   # 0 = surface, 1 = CR border
        df["zone"] = np.where(df["depth_mesh_um"] < 0, "above surface",
                              np.where(df["depth_mesh_um"] <= boundary_um, "sensory (CR+)", "below CR border"))
    return df


def segment_table(neuron: Neuron, surf: BrainSurface, boundary_um=None,
                  dend_types=(3, 4), axon_types=(2,)) -> pd.DataFrame:
    """One row per parent-child edge with its length and midpoint depth.

    Length-weighted statistics avoid the bias of uneven SWC node spacing.
    """
    c, p, t = neuron.segments()
    if not len(c):
        return pd.DataFrame(columns=["type", "compartment", "length_um", "depth_mesh_um"])
    mid = (c + p) / 2
    seg = pd.DataFrame({
        "type": t,
        "compartment": [compartment_of(int(x), dend_types, axon_types) for x in t],
        "length_um": np.linalg.norm(c - p, axis=1),
        "mid_x": mid[:, 0], "mid_y": mid[:, 1], "mid_z": mid[:, 2],
        "depth_mesh_um": surf.signed_depth(mid),
        "depth_plane_um": surf.plane_depth(mid),
    })
    if boundary_um:
        seg["normalized_depth"] = seg["depth_mesh_um"] / boundary_um
    return seg


def _wquantile(x, w, q):
    if not len(x):
        return np.nan
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(np.interp(q * cw[-1], cw, x[o]))


def summary(neuron: Neuron, surf: BrainSurface, seg: pd.DataFrame, boundary_um=None,
            cell_name: str = "cell") -> pd.DataFrame:
    sc = soma_center(neuron)
    row = {"cell": cell_name,
           "soma_depth_mesh_um": float(surf.signed_depth(sc)[0]),
           "soma_depth_plane_um": float(surf.plane_depth(sc)[0]),
           "CR_border_depth_um": boundary_um if boundary_um else np.nan}
    if boundary_um:
        row["soma_normalized_depth"] = row["soma_depth_mesh_um"] / boundary_um
    for comp in ("dendrite", "axon"):
        s = seg[seg["compartment"] == comp]
        L = s["length_um"].to_numpy(float)
        d = s["depth_mesh_um"].to_numpy(float)
        row[f"{comp}_length_um"] = float(L.sum())
        row[f"{comp}_depth_min_um"] = float(d.min()) if len(d) else np.nan
        row[f"{comp}_depth_max_um"] = float(d.max()) if len(d) else np.nan
        row[f"{comp}_depth_median_um"] = _wquantile(d, L, 0.5)
        row[f"{comp}_depth_p10_um"] = _wquantile(d, L, 0.1)
        row[f"{comp}_depth_p90_um"] = _wquantile(d, L, 0.9)
        if boundary_um:
            ins = (d >= 0) & (d <= boundary_um)
            row[f"{comp}_length_in_sensory_um"] = float(L[ins].sum())
            row[f"{comp}_length_below_CR_um"] = float(L[d > boundary_um].sum())
            row[f"{comp}_fraction_in_sensory"] = float(L[ins].sum() / L.sum()) if L.sum() else np.nan
    return pd.DataFrame([row])


def sholl(neuron: Neuron, types, center, step_um: float = 5.0, max_r: float | None = None):
    c, p, _ = neuron.segments(types)
    if not len(c):
        return np.zeros(0), np.zeros(0, int)
    d0 = np.linalg.norm(c - center, axis=1)
    d1 = np.linalg.norm(p - center, axis=1)
    lo, hi = np.minimum(d0, d1), np.maximum(d0, d1)
    if max_r is None:
        max_r = hi.max()
    radii = np.arange(0, max_r + step_um, step_um)
    counts = np.array([np.sum((lo < r) & (hi >= r)) for r in radii])
    return radii, counts


def sholl_table(neuron, dend_types, axon_types, step_um=5.0) -> pd.DataFrame:
    cen = soma_center(neuron)
    c, p, _ = neuron.segments()
    max_r = float(np.max(np.linalg.norm(np.vstack([c, p]) - cen, axis=1))) if len(c) else 0
    r, dn = sholl(neuron, dend_types, cen, step_um, max_r)
    _, ax = sholl(neuron, axon_types, cen, step_um, max_r)
    if not len(r):
        r = np.arange(0, max_r + step_um, step_um)
    dn = dn if len(dn) else np.zeros(len(r), int)
    ax = ax if len(ax) else np.zeros(len(r), int)
    return pd.DataFrame({"radius_um": r, "dendrite_intersections": dn, "axon_intersections": ax})


# ---------------------------------------------------------------- figures
def make_figures(neuron: Neuron, surf: BrainSurface, nodes: pd.DataFrame, seg: pd.DataFrame,
                 sholl_df: pd.DataFrame, boundary_um=None, cell_name="cell"):
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.collections import LineCollection

    figs = {}
    comp_col = {"dendrite": TYPE_COLORS[3], "axon": TYPE_COLORS[2]}

    # 1) laminar side view: in-plane axis vs depth
    fig = Figure(figsize=(7, 6), layout="constrained")
    ax = fig.add_subplot(111)
    c, p, t = neuron.segments()
    if len(c):
        cu = surf.along_surface(c); pu = surf.along_surface(p)
        cd = surf.signed_depth(c); pd_ = surf.signed_depth(p)
        lines = np.stack([np.column_stack([cu, cd]), np.column_stack([pu, pd_])], axis=1)
        cols = [TYPE_COLORS.get(int(k), "#999") for k in t]
        ax.add_collection(LineCollection(lines, colors=cols, linewidths=1.0))
    sc = soma_center(neuron)
    ax.plot(surf.along_surface(sc)[0], surf.signed_depth(sc)[0], "o", color=S.SOMA, ms=9, mec="white", mew=1.5,
            label="soma", zorder=5)
    ax.axhline(0, color=S.DOME_LINE, lw=1.5, label="brain surface")
    if boundary_um:
        ax.axhline(boundary_um, color=S.LAYER_LINE, lw=1.5, ls=(0, (6, 5)), label=f"CR border ({boundary_um:.0f} µm)")
        ax.axhspan(0, boundary_um, color=S.DOME_FILL, zorder=0, lw=0)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()
    ax.set_xlabel("Position along surface (µm)")
    ax.set_ylabel("Depth from surface (µm)")
    ax.set_title(f"{cell_name}: laminar position (side view)")
    ax.legend(loc="lower right", fontsize=8)
    figs["laminar_profile"] = fig

    # 2) length-weighted depth histogram, Method 1 vs Method 2
    fig = Figure(figsize=(10, 4.5), layout="constrained")
    axs = fig.subplots(1, 2, sharey=True)
    allv = seg[["depth_mesh_um", "depth_plane_um"]].to_numpy().ravel() if len(seg) else np.array([0, 1])
    bins = np.linspace(np.nanmin(allv), np.nanmax(allv), 31) if len(allv) else 30
    for a, col, title in [(axs[0], "depth_mesh_um", "Method 1: distance to surface mesh"),
                          (axs[1], "depth_plane_um", "Method 2: best-fit plane normal")]:
        for comp in ("dendrite", "axon"):
            s = seg[seg["compartment"] == comp]
            if len(s):
                a.hist(s[col], bins=bins, weights=s["length_um"], alpha=0.6, color=comp_col[comp], label=comp,
                       orientation="horizontal")
        a.axhline(0, color=S.DOME_LINE, lw=1.2)
        if boundary_um:
            a.axhline(boundary_um, color=S.LAYER_LINE, ls=(0, (6, 5)), lw=1.2, label="CR border")
        a.set_title(title, fontsize=10)
        a.set_xlabel("Neurite length (µm)")
    axs[0].set_ylabel("Depth from surface (µm)")
    axs[0].invert_yaxis()
    axs[0].legend(fontsize=8)
    figs["depth_histograms"] = fig

    # 2b) laminar distribution in % of the CR depth (0 = surface, 100 = CR border)
    if boundary_um:
        figs["laminar_percent"] = laminar_figure(neuron, surf, boundary_um, cell_name)

    # 3) Sholl: profile (intersections / branch points / length), 3D shells, crossings by depth
    figs["sholl_3d"] = sholl_profile_figure(neuron, float(sholl_df.radius_um.diff().dropna().median())
                                            if len(sholl_df) > 1 else 10.0, cell_name=cell_name)
    figs["sholl_shells"] = sholl_3d_figure(neuron, float(sholl_df.radius_um.diff().dropna().median())
                                           if len(sholl_df) > 1 else 10.0)
    if boundary_um:
        figs["sholl_depth"] = sholl_depth_figure(neuron, surf, boundary_um,
                                                 float(sholl_df.radius_um.diff().dropna().median())
                                                 if len(sholl_df) > 1 else 10.0)
    return figs


LAMINAR_COLORS = {"dendrite": S.DEND, "axon": S.AXON}
INK, INK2, GRID = S.INK, S.MUTED, S.GRID


def laminar_figure(neuron: Neuron, surf: BrainSurface, boundary_um, cell_name="cell",
                   dend_types=(3, 4), axon_types=(2, 7), step=5, mode="percent", show_axon=True,
                   show_dendrites=True, smooth=False):
    """Laminar profile: % of neurite length per depth band (0 % = brain surface, 100 % = CR border),
    one panel per compartment on a shared depth axis."""
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyBboxPatch
    from .morphometry import laminar_profile, LAMINAR_BINS

    lp = laminar_profile(neuron, surf, boundary_um, dend_types, axon_types, step_pct=step)
    comps = ([("dendrite", "Dendrites")] if show_dendrites else []) + \
            ([("axon", "Axon")] if (show_axon and lp.axon_length_um.sum() > 0) else [])
    if not comps:
        comps = [("dendrite", "Dendrites")]
    unit = "µm" if mode == "um" else "%"
    vcol = "{}_length_um" if mode == "um" else "{}_percent_of_length"
    used = lp[(lp.dendrite_length_um + lp.axon_length_um) > 0]
    lo = min(-5.0, float(used.bin_start_pct.min()) - step if len(used) else 0.0)
    hi = max(105.0, float(used.bin_end_pct.max()) + step if len(used) else 100.0)
    soma_pct = 100 * surf.signed_depth(soma_center(neuron))[0] / boundary_um

    fig = Figure(figsize=(3.4 * len(comps) + 1.6, 6.2), layout="constrained")
    fig.set_facecolor("white")
    axs = fig.subplots(1, len(comps), sharey=True, squeeze=False)[0]

    def nice_max(v, floor):
        return max(floor, np.ceil(v * 1.25 / 10) * 10)
    shared = nice_max(max(float(lp[vcol.format(c)].max()) for c, _ in comps), 10.0)
    for ax, (comp, title) in zip(axs, comps):
        # % mode: one scale so the panels compare; µm mode: each panel on its own scale
        xmax = shared if mode != "um" else nice_max(float(lp[vcol.format(comp)].max()), 5.0)
        ax.set_facecolor("white")
        col = LAMINAR_COLORS[comp]
        vals = lp[vcol.format(comp)].to_numpy()
        L = lp[f"{comp}_length_um"].to_numpy()
        # quiet bands: sensory SC (0-100 %) in quarters
        ax.axhspan(0, 100, color=S.DOME_FILL, zorder=0, lw=0, alpha=0.5)       # sensory SC (poster dome tone)
        for q0 in (0, 50):
            ax.axhspan(q0, q0 + 25, color=S.DOME_FILL, zorder=0, lw=0)
        for b0, v in zip(lp.bin_start_pct, vals):
            if v <= 0:
                continue
            h = step * 0.72
            ax.add_patch(FancyBboxPatch((0, b0 + (step - h) / 2), v, h, mutation_scale=1,
                                        boxstyle="round,pad=0,rounding_size=0.6",
                                        fc=col, ec="none", zorder=2))
        # quarter totals, written at the right edge of each band
        c, p, _ = neuron.segments(dend_types if comp == "dendrite" else axon_types)
        if len(c):
            Lseg = np.linalg.norm(c - p, axis=1)
            pct = 100 * surf.signed_depth((c + p) / 2) / boundary_um
            for b_lo, b_hi, name in LAMINAR_BINS:
                f = Lseg[(pct >= b_lo) & (pct < b_hi)].sum() / Lseg.sum()
                if f < 0.005:
                    continue
                y = (max(b_lo, lo) + min(b_hi, hi)) / 2
                ax.text(xmax * 0.98, y, f"{100 * f:.0f}%", ha="right", va="center", fontsize=8.5,
                        color=INK, fontweight="bold", zorder=4)
        # reference lines, labelled directly
        ax.axhline(0, color=S.DOME_LINE, lw=1.5, zorder=3)
        ax.axhline(100, color=S.LAYER_LINE, lw=1.3, ls=(0, (6, 5)), zorder=3)
        ax.axhline(soma_pct, color=S.SOMA, lw=1.2, ls=(0, (2, 2)), zorder=3)
        ax.text(xmax * 0.02, 0 - 1, "brain surface", va="bottom", ha="left", fontsize=8, color=INK, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.text(xmax * 0.02, 100 - 1, "CR border", va="bottom", ha="left", fontsize=8, color=S.LAYER_TEXT, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.text(xmax * 0.02, soma_pct - 1, f"soma {soma_pct:.0f}%", va="bottom", ha="left", fontsize=8, color=INK, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.set_xlim(0, xmax)
        ax.set_ylim(hi, lo)
        ax.annotate("■", xy=(0, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points",
                    fontsize=12, color=col, va="bottom", ha="left")
        ax.set_title(f"     {title}", fontsize=11, color=INK, loc="left", fontweight="bold")
        ax.set_title(f"{L.sum():.0f} µm", fontsize=8.5, color=INK2, loc="right", fontweight="normal")
        if smooth and L.sum() > 0:
            # length-weighted kernel density along depth, scaled to the bar units
            cc, pp, _ = neuron.segments(dend_types if comp == "dendrite" else axon_types)
            Ls = np.linalg.norm(cc - pp, axis=1)
            dp = 100 * surf.signed_depth((cc + pp) / 2) / boundary_um
            yy = np.linspace(lo, hi, 400)
            bw = max(step * 0.75, 2.0)
            dens = (Ls[None, :] * np.exp(-0.5 * ((yy[:, None] - dp[None, :]) / bw) ** 2)).sum(1)
            dens = dens / (bw * np.sqrt(2 * np.pi)) * step              # length per bin of width `step`
            if mode != "um":
                dens = 100 * dens / Ls.sum()
            ax.plot(dens, yy, color=INK, lw=1.4, zorder=4)
        ax.set_xlabel(f"{'length per band (µm)' if mode == 'um' else '% of length'}", fontsize=9, color=INK2)
        ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    axs[0].set_yticks([0, 25, 50, 75, 100] + ([t for t in (125, 150) if t < hi]))
    axs[0].set_yticklabels([f"{t}%" for t in axs[0].get_yticks()])
    axs[0].set_ylabel("Depth  (0 % = brain surface, 100 % = CR border)", fontsize=9, color=INK2)
    fig.suptitle(f"Laminar distribution  ·  {cell_name}", fontsize=12, fontweight="bold", color=INK, x=0.02, ha="left")
    return fig


def sholl_profile_figure(neuron: Neuron, step_um=10.0, dend_types=(3, 4), axon_types=(2, 7), cell_name="cell"):
    """3D Sholl from the soma: intersections, branch points (+ tips) and length per shell."""
    from matplotlib.figure import Figure
    from .morphometry import sholl_detailed
    cen = soma_center(neuron)
    fig = Figure(figsize=(7, 7.5), layout="constrained")
    axs = fig.subplots(3, 1, sharex=True)
    for comp, types, col in (("dendrite", dend_types, S.DEND), ("axon", axon_types, S.AXON)):
        t, _ = sholl_detailed(neuron, types, cen, step_um)
        if not len(t):
            continue
        ls = "-" if comp == "dendrite" else (0, (5, 3))
        axs[0].plot(t.radius_um, t.intersections, color=col, ls=ls, label=comp)
        axs[1].plot(t.radius_um, t.branch_points, color=col, ls=ls, label=f"{comp} branch points")
        axs[1].plot(t.radius_um, t.tips, color=col, ls=(0, (1, 2)), lw=1.4, label=f"{comp} tips")
        axs[2].plot(t.radius_um, t.length_um, color=col, ls=ls, label=comp)
    axs[0].set_ylabel("Intersections"); axs[1].set_ylabel("Count per shell"); axs[2].set_ylabel("Length per shell (µm)")
    axs[2].set_xlabel(f"Distance from soma (µm) · {step_um:g} µm shells")
    for ax_ in axs:
        ax_.yaxis.grid(True); ax_.set_axisbelow(True)
        ax_.legend(fontsize=8, loc="upper right")
    axs[0].set_title(f"3D Sholl analysis · {cell_name}")
    return fig


def sholl_3d_figure(neuron: Neuron, step_um=10.0, dend_types=(3, 4), n_shells=4):
    """The neuron with Sholl spheres around the soma; crossing points coloured by radius."""
    from matplotlib.figure import Figure
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from .morphometry import sholl_detailed
    cen = soma_center(neuron)
    t, cr = sholl_detailed(neuron, dend_types, cen, step_um)
    fig = Figure(figsize=(7, 6.5), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    c, p, tt = neuron.segments()
    if len(c):
        ax.add_collection3d(Line3DCollection(np.stack([c, p], 1), colors=[TYPE_COLORS.get(int(k), "#999") for k in tt],
                                             linewidths=0.9, alpha=0.75))
    ramp = LinearSegmentedColormap.from_list("morph_seq", ["#f4c7c0", S.DEND, "#6e1f17"])
    if len(t):
        rmax = t.radius_um.max()
        shells = np.linspace(rmax / n_shells, rmax, n_shells)
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        for r in shells:
            ax.plot_wireframe(cen[0] + r * np.cos(u) * np.sin(v), cen[1] + r * np.sin(u) * np.sin(v),
                              cen[2] + r * np.cos(v), color=S.LAYER_LINE, lw=0.25, alpha=0.22)
        if len(cr):
            sc = ax.scatter(cr.x, cr.y, cr.z, c=cr.radius_um, cmap=ramp, s=14, depthshade=False,
                            edgecolor="white", linewidth=0.3, vmin=0, vmax=rmax)
            cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.08)
            cb.set_label("Sholl radius (µm)")
        lo, hi = cen - rmax * 1.05, cen + rmax * 1.05
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect((1, 1, 1))
    ax.scatter(*cen, color=S.SOMA, s=70, depthshade=False, edgecolor="white", linewidth=1.2)
    ax.invert_yaxis()
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)"); ax.set_zlabel("z (µm)")
    ax.set_title("Sholl spheres · dendrite crossings  (drag to rotate)")
    return fig


def sholl_depth_figure(neuron: Neuron, surf: BrainSurface, boundary_um, step_um=10.0, depth_bin=10.0,
                       dend_types=(3, 4)):
    """Where in depth the dendrites cross each Sholl shell: radius x laminar depth (0 % surface, 100 % CR)."""
    from matplotlib.figure import Figure
    from matplotlib.colors import LinearSegmentedColormap
    from .morphometry import sholl_detailed
    cen = soma_center(neuron)
    t, cr = sholl_detailed(neuron, dend_types, cen, step_um, surf, boundary_um)
    fig = Figure(figsize=(7, 5.2), layout="constrained")
    ax = fig.add_subplot(111)
    if not len(cr) or "depth_pct" not in cr:
        ax.text(0.5, 0.5, "needs a CR border depth", ha="center", transform=ax.transAxes, color=S.MUTED)
        return fig
    rb = np.arange(step_um / 2, cr.radius_um.max() + step_um, step_um)       # bins centred on each shell
    lo = min(-10.0, np.floor(cr.depth_pct.min() / depth_bin) * depth_bin)
    hi = max(110.0, np.ceil(cr.depth_pct.max() / depth_bin) * depth_bin)
    db = np.arange(lo, hi + depth_bin, depth_bin)
    H, _, _ = np.histogram2d(cr.radius_um, cr.depth_pct, bins=[rb, db])
    ramp = LinearSegmentedColormap.from_list("morph_seq0", ["#ffffff", "#f4c7c0", S.DEND, "#6e1f17"])
    m = ax.pcolormesh(rb, db, H.T, cmap=ramp, shading="flat")
    ax.axhline(0, color=S.DOME_LINE, lw=1.4); ax.axhline(100, color=S.LAYER_LINE, lw=1.3, ls=(0, (6, 5)))
    sp = 100 * surf.signed_depth(cen)[0] / boundary_um
    ax.axhline(sp, color=S.SOMA, lw=1.1, ls=(0, (2, 2)))
    ax.text(rb[0], 0, " brain surface", va="bottom", fontsize=8, color=S.INK)
    ax.text(rb[0], 100, " CR border", va="bottom", fontsize=8, color=S.LAYER_TEXT)
    ax.text(rb[0], sp, f" soma {sp:.0f}%", va="bottom", fontsize=8, color=S.INK)
    ax.set_ylim(hi, lo)
    ax.set_xlabel("Sholl radius (µm)"); ax.set_ylabel("Depth (% of surface → CR border)")
    cb = fig.colorbar(m, ax=ax, shrink=0.8); cb.set_label("dendrite crossings")
    ax.set_title("Sholl crossings by depth")
    return fig


def make_3d_figure(neuron: Neuron, surf: BrainSurface, boundary_um=None, margin_um: float = 60.0):
    """Rotatable 3D view (matplotlib): brain surface, CR border surface and the SWC paths."""
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    fig = Figure(figsize=(7, 6), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    xyz = neuron.xyz
    # show the stretch of brain surface and CR border above / below the cell too: include the points
    # on both surfaces closest to the cell (the CR border can be far below the imaged cell)
    import trimesh
    sample = xyz[:: max(1, len(xyz) // 200)]
    pts = [xyz]
    meshes = [surf.mesh] + ([surf.offset_mesh(boundary_um)] if boundary_um else [])
    for m in meshes:
        try:
            if len(m.faces):
                pts.append(trimesh.proximity.closest_point(m, sample)[0])
        except Exception:
            pass
    allp = np.vstack(pts)
    lo, hi = allp.min(0) - margin_um, allp.max(0) + margin_um
    lo[2], hi[2] = xyz[:, 2].min() - margin_um, xyz[:, 2].max() + margin_um

    def crop(m):
        tri = m.vertices[m.faces]
        keep = np.all((tri.mean(1) >= lo) & (tri.mean(1) <= hi), axis=1)
        return tri[keep]

    tri = crop(surf.mesh)
    if len(tri):
        ax.add_collection3d(Poly3DCollection(tri, facecolor=S.DOME_LINE, edgecolor="none", alpha=0.22))
    if boundary_um:
        tri_o = crop(surf.offset_mesh(boundary_um))
        if len(tri_o):
            ax.add_collection3d(Poly3DCollection(tri_o, facecolor="#e0a823", edgecolor="none", alpha=0.2))
    c, p, t = neuron.segments()
    if len(c):
        ax.add_collection3d(Line3DCollection(np.stack([c, p], axis=1),
                                             colors=[TYPE_COLORS.get(int(k), "#999") for k in t], linewidths=1.2))
    sc = soma_center(neuron)
    ax.scatter(*sc, color=S.SOMA, s=70, depthshade=False, edgecolor="white", linewidth=1.2)
    span = hi - lo
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(span / span.max())
    ax.invert_yaxis()                         # image orientation (y down)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)"); ax.set_zlabel("z (µm)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color=S.DOME_LINE, alpha=0.4, label="brain surface")]
    if boundary_um:
        handles.append(Patch(color="#e0a823", alpha=0.45, label="CR border"))
    handles += [Line2D([], [], color=TYPE_COLORS[3], label="dendrite"), Line2D([], [], color=TYPE_COLORS[2], label="axon")]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    ax.set_title("Drag to rotate · right-drag to zoom")
    return fig


def interactive_html(path, neuron: Neuron, surf: BrainSurface, boundary_um=None):
    import plotly.graph_objects as go

    fig = go.Figure()
    m = surf.mesh
    fig.add_trace(go.Mesh3d(x=m.vertices[:, 0], y=m.vertices[:, 1], z=m.vertices[:, 2],
                            i=m.faces[:, 0], j=m.faces[:, 1], k=m.faces[:, 2],
                            color="lightgray", opacity=0.35, name="brain surface", showlegend=True))
    if boundary_um:
        o = surf.offset_mesh(boundary_um)
        fig.add_trace(go.Mesh3d(x=o.vertices[:, 0], y=o.vertices[:, 1], z=o.vertices[:, 2],
                                i=o.faces[:, 0], j=o.faces[:, 1], k=o.faces[:, 2],
                                color="goldenrod", opacity=0.25, name="CR border", showlegend=True))
    names = {1: "soma", 2: "axon", 3: "dendrite", 4: "apical dendrite"}
    for tp in sorted(set(neuron.df["type"])):
        c, p, _ = neuron.segments([tp])
        if not len(c):
            continue
        xs = np.column_stack([c[:, 0], p[:, 0], np.full(len(c), np.nan)]).ravel()
        ys = np.column_stack([c[:, 1], p[:, 1], np.full(len(c), np.nan)]).ravel()
        zs = np.column_stack([c[:, 2], p[:, 2], np.full(len(c), np.nan)]).ravel()
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=names.get(tp, f"type {tp}"),
                                   line=dict(color=TYPE_COLORS.get(tp, "#999"), width=4 if tp == 1 else 2)))
    fig.update_layout(scene=dict(xaxis_title="X (µm)", yaxis_title="Y (µm)", zaxis_title="Z (µm)",
                                 aspectmode="data"), title="Neuron with brain surface (drag to rotate)")
    fig.write_html(path)


def export_point_cloud(path, neuron: Neuron):
    import trimesh
    cmap = {1: [220, 20, 60, 255], 2: [30, 90, 220, 255], 3: [40, 160, 70, 255], 4: [40, 160, 70, 255]}
    cols = np.array([cmap.get(int(t), [128, 128, 128, 255]) for t in neuron.df["type"]], dtype=np.uint8)
    trimesh.points.PointCloud(neuron.xyz, colors=cols).export(path)


# ---------------------------------------------------------------- run all
def run(neuron: Neuron, surf: BrainSurface, out_dir: str, cell_name: str = "cell",
        boundary_um=None, dend_types=(3, 4), axon_types=(2,), sholl_step_um=5.0,
        master_csv: str | None = None, save=True, manual_type: str = "",
        ais_type: int = 7, ais_points_xyz=None, ais_override: dict | None = None, source_path: str = ""):
    from .morphometry import ais_features, cell_features
    nodes = node_table(neuron, surf, boundary_um, dend_types, axon_types)
    seg = segment_table(neuron, surf, boundary_um, dend_types, axon_types)
    summ = summary(neuron, surf, seg, boundary_um, cell_name)
    sh = sholl_table(neuron, dend_types, axon_types, sholl_step_um)
    feats = cell_features(neuron, surf, boundary_um, dend_types, axon_types, sh)
    ais = ais_override if ais_override else ais_features(neuron, surf, ais_type, ais_points_xyz,
                                                          dend_types, tuple(axon_types))
    if ais and boundary_um and "ais_depth_um" in ais:
        ais["ais_normalized_depth"] = ais["ais_depth_um"] / boundary_um
    feats.update(ais)
    for k, v in feats.items():
        if k not in summ.columns:
            summ[k] = [v]
    summ["manual_type"] = [manual_type]
    summ["source_path"] = [source_path]
    figs = make_figures(neuron, surf, nodes, seg, sh, boundary_um, cell_name)
    res = dict(nodes=nodes, segments=seg, summary=summ, sholl=sh, figures=figs, features=feats)
    if boundary_um:
        from .morphometry import laminar_profile
        res["laminar"] = laminar_profile(neuron, surf, boundary_um, dend_types, axon_types)
    if not save:
        return res
    os.makedirs(out_dir, exist_ok=True)
    pre = os.path.join(out_dir, cell_name)
    nodes.to_csv(pre + "_nodes.csv", index=False)
    seg.to_csv(pre + "_segments.csv", index=False)
    summ.to_csv(pre + "_summary.csv", index=False)
    sh.to_csv(pre + "_sholl.csv", index=False)
    if "laminar" in res:
        res["laminar"].to_csv(pre + "_laminar_distribution.csv", index=False)
    for k, f in figs.items():
        f.savefig(f"{pre}_{k}.png", dpi=150)
        f.savefig(f"{pre}_{k}.svg")
    surf.mesh.export(pre + "_brain_surface.stl")
    if boundary_um:
        surf.offset_mesh(boundary_um).export(pre + "_CR_border_surface.stl")
    export_point_cloud(pre + "_neuron_colored.ply", neuron)
    try:
        interactive_html(pre + "_3d.html", neuron, surf, boundary_um)
    except Exception as e:  # plotly missing etc.
        print("interactive html skipped:", e)
    if master_csv:
        if os.path.exists(master_csv):
            old = pd.read_csv(master_csv)
            old = old[old["cell"] != cell_name]
            summ_all = pd.concat([old, summ], ignore_index=True)
        else:
            summ_all = summ
        summ_all.to_csv(master_csv, index=False)
    return res
