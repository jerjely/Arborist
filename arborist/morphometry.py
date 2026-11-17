"""Morphological features in the brain-surface (laminar) frame + cell classification.

Frame, relative to the soma:
  dd  = depth from surface - soma depth   (positive = ventral / deeper)
  du, dv = in-plane coordinates parallel to the surface (best-fit plane axes)
In acute slices one in-plane axis is cut by the slice faces, so the
dendritic field width is reported along the major in-plane axis (and minor).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from .analysis import soma_center
from .surface import BrainSurface
from .swc import Neuron

# --------------------------------------------------------------------------
# Type-hint thresholds. STARTING GUESSES from the qualitative descriptions of
# mouse sSC types (wide-field, narrow-field, horizontal, stellate, marginal);
# tune them on your own labelled cells.
HINT_RULES = {
    "marginal_max_soma_norm_depth": 0.12,   # soma in the stratum zonale
    "wide_field_min_width_um": 250.0,       # in-plane dendritic field width
    "narrow_field_max_width_um": 150.0,
    "dorsal_min_fraction": 0.65,            # fraction of dendrite above soma
    "vertical_min_aspect": 1.2,             # height / width
    "horizontal_max_aspect": 0.5,
}

TYPE_LABELS = ["", "wide-field", "narrow-field", "horizontal", "stellate", "marginal", "other"]

DEFAULT_CLUSTER_FEATURES = [
    "soma_normalized_depth", "dend_width_major_um", "dend_height_um", "dend_vertical_aspect",
    "dend_dorsal_fraction", "dend_dorsal_reach_um", "dend_ventral_reach_um", "dend_planar_elongation",
    "dend_n_primary", "dend_n_branch_points", "dend_n_branches", "dend_max_branch_order",
    "dendrite_length_um", "dend_tortuosity", "sholl_peak_radius_um",
    "dend_frac_0_25pct", "dend_frac_25_50pct", "dend_frac_50_75pct", "dend_frac_75_100pct",
]
LAMINAR_BINS = [(-np.inf, 0, "above_surface"), (0, 25, "0_25pct"), (25, 50, "25_50pct"),
                (50, 75, "50_75pct"), (75, 100, "75_100pct"), (100, np.inf, "below_CR")]


def _hull(pts):
    try:
        if len(pts) >= pts.shape[1] + 1:
            return float(ConvexHull(pts).volume)
    except Exception:
        pass
    return np.nan


def cell_features(neuron: Neuron, surf: BrainSurface, boundary_um=None,
                  dend_types=(3, 4), axon_types=(2,), sholl_df: pd.DataFrame | None = None) -> dict:
    df = neuron.df
    sc = soma_center(neuron)
    d0 = float(surf.signed_depth(sc)[0])
    s_uvn = surf.plane_coords(sc)[0]
    f = {"soma_depth_um": d0}
    if boundary_um:
        f["soma_normalized_depth"] = d0 / boundary_um

    def frame(xyz):
        uvn = surf.plane_coords(xyz)
        return uvn[:, 0] - s_uvn[0], uvn[:, 1] - s_uvn[1], surf.signed_depth(xyz) - d0

    # ---------------------------------------------------------- dendrites
    dn = df[df["type"].isin(dend_types)]
    c, p, _ = neuron.segments(dend_types)
    if len(dn) and len(c):
        du, dv, dd = frame(dn[["x", "y", "z"]].to_numpy(float))
        L = np.linalg.norm(c - p, axis=1)
        mu, mv, md = frame((c + p) / 2)
        # in-plane principal axes (length-weighted)
        P = np.column_stack([mu, mv])
        w = L / L.sum()
        cov = np.cov(P.T, aweights=w) if len(P) > 2 else np.eye(2)
        evals, evecs = np.linalg.eigh(cov)
        major, minor = evecs[:, 1], evecs[:, 0]
        Q = np.column_stack([du, dv])
        width_major = float(np.ptp(Q @ major))
        width_minor = float(np.ptp(Q @ minor))
        height = float(np.ptp(dd))
        f.update({
            "dend_width_u_um": float(np.ptp(du)), "dend_width_v_um": float(np.ptp(dv)),
            "dend_width_major_um": width_major, "dend_width_minor_um": width_minor,
            "dend_height_um": height,
            "dend_vertical_aspect": height / width_major if width_major else np.nan,
            "dend_planar_elongation": width_minor / width_major if width_major else np.nan,
            "dend_dorsal_reach_um": float(max(0.0, -dd.min())),
            "dend_ventral_reach_um": float(max(0.0, dd.max())),
            "dend_dorsal_fraction": float(L[md < 0].sum() / L.sum()),
            "dend_centroid_dorsal_um": float(-(md * L).sum() / L.sum()),
            "dend_centroid_lateral_um": float(np.hypot((mu * L).sum(), (mv * L).sum()) / L.sum()),
            "dend_max_reach_um": float(np.linalg.norm(dn[["x", "y", "z"]].to_numpy(float) - sc, axis=1).max()),
            "dend_hull_area_side_um2": _hull(np.column_stack([Q @ major, dd])),
            "dend_hull_volume_um3": _hull(np.column_stack([du, dv, dd])),
        })
        if boundary_um:
            dep = dd + d0
            f["dend_top_normalized"] = float(dep.min() / boundary_um)
            f["dend_bottom_normalized"] = float(dep.max() / boundary_um)
        # topology
        ids = set(dn["id"])
        soma_ids = set(df.loc[df["type"] == 1, "id"])
        children = df.groupby("parent")["id"].count()
        dchildren = dn.groupby("parent")["id"].count()
        prim = dn[dn["parent"].isin(soma_ids) | (dn["parent"] == -1) |
                  ~dn["parent"].isin(ids | soma_ids)]
        f["dend_n_primary"] = int(len(prim))
        nch = dn["id"].map(dchildren).fillna(0).to_numpy()
        f["dend_n_branch_points"] = int((nch >= 2).sum())
        f["dend_n_tips"] = int((dn["id"].map(children).fillna(0).to_numpy() == 0).sum())
        f["dendrite_length_um"] = float(L.sum())
    # --------------------------------------------------------------- axon
    ca, pa, _ = neuron.segments(axon_types)
    f["has_axon"] = bool(len(ca))
    if len(ca):
        La = np.linalg.norm(ca - pa, axis=1)
        au, av, ad = frame((ca + pa) / 2)
        f["axon_length_um"] = float(La.sum())
        f["axon_centroid_dorsal_um"] = float(-(ad * La).sum() / La.sum())
        f["axon_centroid_lateral_um"] = float(np.hypot((au * La).sum(), (av * La).sum()) / La.sum())
    # -------------------------------------------------------------- sholl
    if sholl_df is not None and len(sholl_df) and sholl_df["dendrite_intersections"].max() > 0:
        i = int(np.argmax(sholl_df["dendrite_intersections"].to_numpy()))
        f["sholl_peak_intersections"] = int(sholl_df["dendrite_intersections"].iloc[i])
        f["sholl_peak_radius_um"] = float(sholl_df["radius_um"].iloc[i])
    f.update(soma_size(neuron))
    f.update(thickness_features(neuron, dend_types, "dend"))
    f["dend_radii_source"] = "image (width at half brightness)" if getattr(neuron, "radii_source", None) == "image" \
        else ("SWC" if f.get("dend_radii_fitted") else "")
    f.update(topology_features(neuron, dend_types, "dend"))
    if f.get("has_axon"):
        f.update(topology_features(neuron, axon_types, "axon"))
    if boundary_um:
        f.update(laminar_fractions(neuron, surf, boundary_um, dend_types, "dend"))
        if f.get("has_axon"):
            f.update(laminar_fractions(neuron, surf, boundary_um, axon_types, "axon"))
    hint, why = type_hint(f)
    f["type_hint"] = hint
    f["type_hint_reason"] = why
    return f


def _max_extent(P):
    """Largest distance between two points, without comparing every pair: the farthest pair always lies
    on the convex hull, which has few points even for tens of thousands of inputs."""
    from scipy.spatial.distance import pdist
    P = np.asarray(P, float)
    if len(P) > 200:
        try:
            from scipy.spatial import ConvexHull
            P = P[ConvexHull(P).vertices]
        except Exception:                          # flat / degenerate: fall back to a 2D hull or a sample
            try:
                from scipy.spatial import ConvexHull
                k = np.argsort(np.ptp(P, axis=0))[1:]
                P = P[ConvexHull(P[:, k]).vertices]
            except Exception:
                P = P[np.linspace(0, len(P) - 1, 2000).astype(int)]
    return float(pdist(P).max()) if len(P) > 1 else 0.0


def soma_size(neuron: Neuron) -> dict:
    """Soma size from the SWC: contour (several soma nodes) -> max extent; one node -> 2 x radius.
    SNT writes a default radius if the soma was not sized, so check soma_size_source."""
    s = neuron.df[neuron.df["type"] == 1]
    if not len(s):
        return {}
    P = s[["x", "y", "z"]].to_numpy(float)
    r = s["radius"].to_numpy(float)
    if len(s) >= 3:
        d = _max_extent(P)
        out = {"soma_diameter_um": float(max(d, 2 * r.max())), "soma_size_source": "soma contour",
               "soma_n_nodes": int(len(s))}
        if len(s) > 500 or d > 60:
            out["soma_warning"] = (f"{len(s)} SWC points are tagged as soma (extent {d:.0f} µm) - check the "
                                   "path types in SNT; dendrites tagged as soma are left out of the dendrite features.")
        return out
    if r.max() > 0:
        return {"soma_diameter_um": float(2 * r.max()), "soma_size_source": "SWC radius"}
    return {}


def topology_features(neuron: Neuron, types, prefix: str) -> dict:
    """Branches (sections between branch points / tips), max branch order, mean branch length,
    mean terminal branch length, tortuosity (path length / straight distance soma -> tip) and the
    mean remote bifurcation angle."""
    df = neuron.df
    xyz = dict(zip(df["id"], df[["x", "y", "z"]].to_numpy(float)))
    sub = df[df["type"].isin(types)]
    if not len(sub):
        return {}
    ids = set(sub["id"])
    parent = dict(zip(df["id"], df["parent"]))
    children = {}
    for nid, par in zip(sub["id"], sub["parent"]):
        children.setdefault(par, []).append(nid)
    # section starts: nodes whose parent is outside the compartment (soma, other type, root)
    starts = [n for n in sub["id"] if parent.get(n) not in ids]
    sections, order_of, stack = [], {}, [(n, 1) for n in starts]
    sc = soma_center(neuron)
    angles, tort = [], []
    while stack:
        n, order = stack.pop()
        par = parent.get(n, -1)
        path = [xyz[par]] if par in xyz else []
        cur = n
        path.append(xyz[cur])
        while len(children.get(cur, [])) == 1:
            cur = children[cur][0]
            path.append(xyz[cur])
        P = np.array(path)
        L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0
        kids = children.get(cur, [])
        sections.append((L, order, len(kids) == 0))
        if len(kids) >= 2:
            ends = []
            for k_ in kids[:2]:
                e_ = k_
                while len(children.get(e_, [])) == 1:
                    e_ = children[e_][0]
                ends.append(xyz[e_] - xyz[cur])
            a, b = ends
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0 and nb > 0:
                angles.append(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1))))
        for k_ in kids:
            stack.append((k_, order + 1))
    # tortuosity per tip: path length back to the compartment start / straight distance to the soma
    for tip in [n for n in sub["id"] if not children.get(n)]:
        L, cur = 0.0, tip
        while parent.get(cur, -1) in xyz and cur in ids:
            L += np.linalg.norm(xyz[cur] - xyz[parent[cur]]); cur = parent[cur]
        straight = np.linalg.norm(xyz[tip] - sc)
        if straight > 1:
            tort.append(L / straight)
    lens = np.array([x[0] for x in sections]); orders = np.array([x[1] for x in sections])
    term = np.array([x[0] for x in sections if x[2]])
    out = {f"{prefix}_n_branches": int(len(sections)),
           f"{prefix}_max_branch_order": int(orders.max()) if len(orders) else 0,
           f"{prefix}_mean_branch_length_um": float(lens.mean()) if len(lens) else np.nan,
           f"{prefix}_mean_terminal_branch_um": float(term.mean()) if len(term) else np.nan,
           f"{prefix}_tortuosity": float(np.mean(tort)) if tort else np.nan,
           f"{prefix}_mean_bifurcation_angle_deg": float(np.mean(angles)) if angles else np.nan}
    if prefix == "axon":
        out["axon_n_branch_points"] = int(sum(1 for v in children.values() if len(v) >= 2))
    return out


def laminar_fractions(neuron: Neuron, surf: BrainSurface, boundary_um, types, prefix: str) -> dict:
    """Fraction of neurite length per laminar band (0% = brain surface, 100% = CR border)."""
    c, p, _ = neuron.segments(types)
    if not len(c):
        return {}
    L = np.linalg.norm(c - p, axis=1)
    pct = 100 * surf.signed_depth((c + p) / 2) / boundary_um
    out = {}
    for lo, hi, name in LAMINAR_BINS:
        out[f"{prefix}_frac_{name}"] = float(L[(pct >= lo) & (pct < hi)].sum() / L.sum())
    return out


def laminar_profile(neuron: Neuron, surf: BrainSurface, boundary_um, dend_types, axon_types, step_pct=10):
    """Length per 10% depth bin (for the plot and CSV)."""
    edges = np.arange(-20, 160 + step_pct, step_pct, dtype=float)
    rows = {"bin_start_pct": edges[:-1], "bin_end_pct": edges[1:]}
    for name, types in (("dendrite", dend_types), ("axon", axon_types)):
        c, p, _ = neuron.segments(types)
        if len(c):
            L = np.linalg.norm(c - p, axis=1)
            pct = np.clip(100 * surf.signed_depth((c + p) / 2) / boundary_um, edges[0], edges[-1] - 1e-9)
            h, _ = np.histogram(pct, bins=edges, weights=L)
            rows[f"{name}_length_um"] = h
            rows[f"{name}_percent_of_length"] = 100 * h / L.sum()
        else:
            rows[f"{name}_length_um"] = np.zeros(len(edges) - 1)
            rows[f"{name}_percent_of_length"] = np.zeros(len(edges) - 1)
    return pd.DataFrame(rows)


def type_hint(f: dict, rules: dict = HINT_RULES):
    """Rule-of-thumb suggestion, not a classification. Returns (label, reason)."""
    nd = f.get("soma_normalized_depth", np.nan)
    w = f.get("dend_width_major_um", np.nan)
    asp = f.get("dend_vertical_aspect", np.nan)
    dors = f.get("dend_dorsal_fraction", np.nan)
    if np.isnan(w):
        return "", "no dendrites"
    if not np.isnan(nd) and nd <= rules["marginal_max_soma_norm_depth"]:
        return "marginal", f"soma at {nd:.2f} of CR depth (stratum zonale)"
    if w >= rules["wide_field_min_width_um"] and dors >= rules["dorsal_min_fraction"]:
        return "wide-field", f"width {w:.0f} µm, {100 * dors:.0f}% of dendrite dorsal"
    if asp <= rules["horizontal_max_aspect"]:
        return "horizontal", f"field {w:.0f} µm wide, aspect {asp:.2f}"
    if w <= rules["narrow_field_max_width_um"] and dors >= rules["dorsal_min_fraction"] \
            and asp >= rules["vertical_min_aspect"]:
        return "narrow-field", f"width {w:.0f} µm, aspect {asp:.2f}, {100 * dors:.0f}% dorsal"
    if 0.25 <= dors <= 0.75:
        return "stellate", f"radial field, {100 * dors:.0f}% dorsal, aspect {asp:.2f}"
    return "unclear", f"width {w:.0f} µm, aspect {asp:.2f}, {100 * dors:.0f}% dorsal"


# ------------------------------------------------------------- clustering
def cluster_cells(table: pd.DataFrame, features, n_clusters: int = 4, method: str = "ward",
                  balance_prefix: str | None = None, max_missing: float = 1 / 3):
    """Hierarchical clustering on z-scored features. Returns dict with labels, linkage, PCA.

    balance_prefix: features starting with it (e.g. "ephys_") form one block and the rest another;
    each block is scaled by 1/sqrt(n features) so both blocks weigh the same however many
    columns each has."""
    from scipy.cluster.hierarchy import fcluster, linkage

    feats = [c for c in features if c in table.columns]
    X = table[feats].apply(pd.to_numeric, errors="coerce")
    # leave out features most cells do not have (e.g. diameters without fitted radii, AIS not measured,
    # ephys rows not matched) and features that are the same in every cell: filling them in would only
    # pull cells together
    miss_frac = X.isna().mean()
    too_sparse = [c for c in X.columns if miss_frac[c] > max_missing or X[c].notna().sum() < 2]
    constant = [c for c in X.columns if c not in too_sparse and X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=too_sparse + constant)
    if X.shape[1] == 0:
        raise ValueError("None of the ticked features is measured in enough cells "
                         f"(left out: {', '.join(too_sparse + constant)}).")
    feats = list(X.columns)
    missing = X.isna().to_numpy()                 # remembered, so the heatmap can show them as missing
    X = X.fillna(X.median())
    sd = X.std(ddof=0).replace(0, 1)
    Z = ((X - X.mean()) / sd).to_numpy()
    if len(Z) < 3:
        raise ValueError("Need at least 3 analysed cells to cluster.")
    Zw = Z.copy()
    if balance_prefix:
        # one block per prefix (e.g. "ephys_", "invivo_") and one for everything else (morphology)
        prefixes = [balance_prefix] if isinstance(balance_prefix, str) else list(balance_prefix)
        block = np.array([next((k + 1 for k, pf in enumerate(prefixes) if f.startswith(pf)), 0) for f in feats])
        if len(set(block)) > 1:
            for b_ in set(block):
                sel = block == b_
                Zw[:, sel] /= np.sqrt(sel.sum())
    Z_show, Z = Z, Zw
    link = linkage(Z, method=method)
    sil = {}
    if n_clusters in (None, 0, "auto"):
        # number of clusters chosen from the data: the cut with the best average silhouette
        for kk in range(2, min(8, len(Z) - 1) + 1):
            lab = fcluster(link, kk, criterion="maxclust")
            if len(set(lab)) > 1:
                sil[kk] = silhouette(Z, lab)
        k = max(sil, key=sil.get) if sil else min(2, len(Z))
    else:
        k = max(1, min(int(n_clusters), len(Z)))
        for kk in range(2, min(8, len(Z) - 1) + 1):
            lab = fcluster(link, kk, criterion="maxclust")
            if len(set(lab)) > 1:
                sil[kk] = silhouette(Z, lab)
    labels = fcluster(link, k, criterion="maxclust")
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    pcs = U[:, :2] * S[:2]
    expl = (S ** 2) / max((S ** 2).sum(), 1e-12)
    load = Vt[:2]
    if pcs.shape[1] < 2:                      # one feature (or two cells): no second axis
        pcs = np.column_stack([pcs, np.zeros(len(pcs))])
        expl = np.r_[expl, 0.0]
        load = np.vstack([load, np.zeros(load.shape[1])])
    return dict(features=feats, Z=Z_show, linkage=link, labels=labels, pcs=pcs,
                explained=expl[:2], loadings=load, k=int(len(set(labels))), silhouettes=sil,
                auto_k=n_clusters in (None, 0, "auto"), missing=missing, Zw=Z, method=method,
                dropped_sparse=too_sparse, dropped_constant=constant)


def short_names(names, max_len=30):
    """Readable short labels: the words every name shares are dropped (date / project / '63x_Stitch'),
    what is left is cut in the middle if still long. Falls back to the full name when not unique."""
    import re
    names = [str(x) for x in names]
    toks = [[t for t in re.split(r"[_\s]+", n) if t] for n in names]
    if len(names) > 1:
        common = set(toks[0]).intersection(*map(set, toks[1:]))
        short = ["_".join(t for t in tk if t not in common) for tk in toks]
        if all(short) and len(set(short)) == len(short):
            names = short
    return [n if len(n) <= max_len else n[:max_len // 2 - 1] + "…" + n[-(max_len // 2):] for n in names]


def cluster_stability(res):
    """Leave-one-out stability: the clustering is repeated with each cell left out in turn. Returns the
    co-clustering matrix (how often two cells end up together), per-cell and overall agreement with the
    full clustering (1 = every pair keeps its relation), or None with fewer than 5 cells."""
    from scipy.cluster.hierarchy import fcluster, linkage
    Z, labels = res.get("Zw"), np.asarray(res["labels"])
    n = len(labels)
    k = int(res.get("k", len(set(labels))))
    if Z is None or n < 5 or k < 2:
        return None
    together = np.zeros((n, n)); seen = np.zeros((n, n))
    for i in range(n):
        keep = np.array([j for j in range(n) if j != i])
        lab = fcluster(linkage(Z[keep], method=res.get("method", "ward")), min(k, len(keep)), criterion="maxclust")
        same = (lab[:, None] == lab[None, :]).astype(float)
        together[np.ix_(keep, keep)] += same
        seen[np.ix_(keep, keep)] += 1
    co = np.divide(together, seen, out=np.full((n, n), np.nan), where=seen > 0)
    full = labels[:, None] == labels[None, :]
    agree = np.where(full, co, 1 - co)
    off = ~np.eye(n, dtype=bool)
    per_cell = np.array([np.nanmean(agree[i, off[i]]) for i in range(n)])
    return dict(co=co, per_cell=per_cell, overall=float(np.nanmean(agree[off])))


def adjusted_rand(a, b):
    """Adjusted Rand index of two groupings (1 = identical, about 0 = no better than chance)."""
    a, b = pd.Series(list(a)), pd.Series(list(b))
    ct = pd.crosstab(a, b).to_numpy(float)
    comb = lambda x: x * (x - 1) / 2.0
    n = ct.sum()
    if n < 2:
        return np.nan
    sum_ij = comb(ct).sum(); sa = comb(ct.sum(1)).sum(); sb = comb(ct.sum(0)).sum()
    expected = sa * sb / comb(n)
    mx = 0.5 * (sa + sb)
    return float((sum_ij - expected) / (mx - expected)) if mx != expected else 1.0


def silhouette(Z, labels):
    """Mean silhouette: for each cell, (b - a) / max(a, b) with a = mean distance to its own cluster and
    b = mean distance to the nearest other cluster (1 = clear clusters, 0 = no structure)."""
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(Z))
    labels = np.asarray(labels)
    out = []
    for i in range(len(Z)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            out.append(0.0)                       # a cluster of one cell counts as 0
            continue
        a = D[i, same].mean()
        b = min(D[i, labels == l].mean() for l in set(labels) if l != labels[i])
        out.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(out))


def cluster_figures(table, res, label_col="manual_type", thumbs=None, label_name=None):
    from matplotlib.figure import Figure
    from scipy.cluster.hierarchy import dendrogram

    full_names = table["cell"].astype(str).tolist()
    names = short_names(full_names)
    manual = table[label_col].fillna("").astype(str).tolist() if label_col in table else [""] * len(names)
    leaf = [f"{n} [{m}]" if m else n for n, m in zip(names, manual)]
    from . import style as S
    S.apply_rc()
    pal = S.TYPE_PALETTE
    cmap = lambda i: pal[int(i) % len(pal)]
    figs = {}

    have_pics = thumbs is not None and any(t is not None for t in thumbs)
    if not have_pics:
        fig = Figure(figsize=(7, max(4, 0.28 * len(names) + 1.5)), layout="constrained")
        ax = fig.add_subplot(111)
        d = dendrogram(res["linkage"], labels=leaf, orientation="right", ax=ax, color_threshold=0,
                       above_threshold_color="#555")
        lab_by_leaf = dict(zip(leaf, res["labels"]))
        for t in ax.get_yticklabels():
            t.set_color(cmap(lab_by_leaf.get(t.get_text(), 1) - 1))
    else:
        # a picture of each cell at its leaf, in a strip that shares the dendrogram's y axis, so pictures
        # stay on their branch and scale with the panel; framed in the cluster colour, name on the left
        n = len(names)
        fig = Figure(figsize=(9, max(4, 1.15 * n + 1.0)))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 4], wspace=0.02, left=0.22, right=0.97,
                              top=1 - 0.55 / fig.get_figheight(), bottom=max(0.12, 0.65 / fig.get_figheight()))
        ax = fig.add_subplot(gs[0, 1])
        d = dendrogram(res["linkage"], labels=leaf, orientation="right", ax=ax, color_threshold=0,
                       above_threshold_color="#555")
        ax.set_yticklabels([]); ax.tick_params(axis="y", length=0)
        axp = fig.add_subplot(gs[0, 0], sharey=ax)
        axp.set_xlim(0, 9.4)
        axp.set_aspect("equal", adjustable="box", anchor="E")   # square cells, hugging the dendrogram
        axp.axis("off")
        for k, i in enumerate(d["leaves"]):
            y = 5 + 10 * k
            col = cmap(res["labels"][i] - 1)
            short = names[i] if len(names[i]) <= 34 else names[i][:16] + "…" + names[i][-16:]
            label = short + (f"\n[{manual[i]}]" if manual[i] else "")
            th = thumbs[i]
            x0, x1, y0, y1 = 0.5, 9.0, y - 4.25, y + 4.25
            if th is not None:
                h, w_ = th.shape[:2]
                sq = max(h, w_)                                     # letterbox to a square, keep the aspect
                pad = np.zeros((sq, sq, 3), th.dtype)
                pad[(sq - h) // 2:(sq - h) // 2 + h, (sq - w_) // 2:(sq - w_) // 2 + w_] = th[..., :3]
                axp.imshow(pad, extent=[x0, x1, y0, y1], origin="upper", interpolation="lanczos", zorder=1)
            from matplotlib.patches import Rectangle
            axp.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=col, lw=2.0, zorder=2,
                                    clip_on=False))
            axp.text(x0 - 0.4, y, label, ha="right", va="center", fontsize=7, color=col, clip_on=False)
        ax.set_ylim(0, 10 * n)
    ax.set_title("Hierarchical clustering (Ward, z-scored features)")
    ax.set_xlabel("Linkage distance")
    figs["dendrogram"] = (fig, d["ivl"])

    fig = Figure(figsize=(7, 6), layout="constrained")
    ax = fig.add_subplot(111)
    pcs = res["pcs"]
    for k in sorted(set(res["labels"])):
        m = res["labels"] == k
        ax.scatter(pcs[m, 0], pcs[m, 1], s=70, color=cmap(k - 1), label=f"cluster {k}", edgecolor="white", lw=1.2, zorder=3)
    for i, (n, mt) in enumerate(zip(names, manual)):
        ax.annotate(f"{n}" + (f"\n{mt}" if mt else ""), pcs[i], fontsize=7, xytext=(5, 5),
                    textcoords="offset points", color=S.INK, zorder=4)
    # biplot: the features that pull cells along the two axes (longest arrows = strongest)
    load = np.asarray(res.get("loadings"))
    if load.ndim == 2 and load.shape[0] >= 2 and load.shape[1] >= 2:
        strength = np.hypot(load[0], load[1])
        top = np.argsort(strength)[::-1][:min(6, len(strength))]
        reach = 0.85 * np.abs(pcs[:, :2]).max() / max(strength[top].max(), 1e-9)
        for j in top:
            x, y = load[0, j] * reach, load[1, j] * reach
            ax.annotate("", (x, y), (0, 0), arrowprops=dict(arrowstyle="-|>", color=S.MUTED, lw=1.0, alpha=0.8),
                        zorder=2)
            ax.annotate(res["features"][j], (x, y), fontsize=6.5, color=S.MUTED, zorder=2,
                        xytext=(3 if x >= 0 else -3, 3 if y >= 0 else -3), textcoords="offset points",
                        ha="left" if x >= 0 else "right", va="bottom" if y >= 0 else "top")
        ax.axhline(0, color=S.LAYER_LINE, lw=0.6, zorder=1); ax.axvline(0, color=S.LAYER_LINE, lw=0.6, zorder=1)
    ax.margins(0.18)
    ax.set_xlabel(f"PC1 ({100 * res['explained'][0]:.0f}%)")
    ax.set_ylabel(f"PC2 ({100 * res['explained'][1]:.0f}%)" if len(res["explained"]) > 1 else "PC2")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Cells in feature space (PCA) · arrows: the features that separate them most", loc="left",
                 fontsize=10)
    figs["pca"] = fig

    order = [leaf.index(x) for x in d["ivl"]][::-1]
    fig = Figure(figsize=(max(6, 0.45 * len(res["features"]) + 3), max(4, 0.28 * len(names) + 2)),
                 layout="constrained")
    ax = fig.add_subplot(111)
    from matplotlib.colors import LinearSegmentedColormap
    div = LinearSegmentedColormap.from_list("poster_div", [S.INVIVO, "#f2f3f5", S.MORPH])
    div = div.with_extremes(bad="#b9bdc6")
    Zs = np.clip(res["Z"][order], -2.5, 2.5).astype(float)
    miss = res.get("missing")
    if miss is not None and miss.shape == res["Z"].shape:
        Zs = np.ma.masked_array(Zs, mask=miss[order])       # missing: grey, not "average" white
    im = ax.imshow(Zs, cmap=div, aspect="auto", vmin=-2.5, vmax=2.5, interpolation="nearest")
    ax.set_yticks(range(len(order)), [leaf[i] for i in order], fontsize=7)
    ax.set_xticks(range(len(res["features"])), res["features"], rotation=60, ha="right", fontsize=7)
    fig.colorbar(im, ax=ax, label="z-score", shrink=0.6)
    n_miss = int(miss.sum()) if miss is not None else 0
    ax.set_title("Features (rows in dendrogram order)", loc="left", pad=16 if n_miss else 6)
    if n_miss:
        ax.annotate(f"grey = not measured ({n_miss} values; filled with the median for clustering)",
                    (0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                    fontsize=8, color=S.MUTED, va="bottom", ha="left")
    figs["heatmap"] = fig
    sil = res.get("silhouettes") or {}
    if sil:
        fig = Figure(figsize=(6, 3.6), layout="constrained")
        ax = fig.add_subplot(111)
        ks = sorted(sil)
        ax.plot(ks, [sil[k] for k in ks], "-o", color=S.MUTED, lw=1.5, ms=6, zorder=2)
        kb = res.get("k")
        if kb in sil:
            ax.scatter([kb], [sil[kb]], s=110, color=S.MORPH, zorder=3,
                       label=f"chosen: {kb} clusters" if res.get("auto_k") else f"{kb} clusters (set by hand)")
            ax.legend(frameon=False, loc="best")
        ax.set_xticks(ks)
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Mean silhouette (higher = clearer)")
        ax.set_ylim(min(0, min(sil.values()) - 0.05), max(1e-3, max(sil.values())) * 1.15)
        ax.axhline(0, color=S.LAYER_LINE, lw=0.8)
        ax.set_title(f"How many clusters? ({len(res['labels'])} cells)", loc="left")
        figs["silhouette"] = fig

    # ---- stability (leave one cell out)
    stab = cluster_stability(res)
    res["stability"] = stab
    if stab is not None:
        from matplotlib.colors import LinearSegmentedColormap
        seq = LinearSegmentedColormap.from_list("poster_seq", ["#f2f3f5", S.INVIVO]).with_extremes(bad="#ffffff")
        lv = d["leaves"][::-1]
        M = stab["co"][np.ix_(lv, lv)].copy()
        np.fill_diagonal(M, np.nan)
        nn = len(lv)
        fig = Figure(figsize=(max(6, 0.42 * nn + 3.6), max(4.5, 0.42 * nn + 2)), layout="constrained")
        ax = fig.add_subplot(111)
        im = ax.imshow(M, cmap=seq, vmin=0, vmax=1, interpolation="nearest")
        lab_o = np.asarray(res["labels"])[lv]
        for b_ in np.flatnonzero(np.diff(lab_o)) + 0.5:               # borders between the clusters
            ax.axhline(b_, color=S.INK, lw=1.2); ax.axvline(b_, color=S.INK, lw=1.2)
        ax.set_xticks(range(nn), [names[i] for i in lv], rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(nn), [f"{names[i]}  ({100 * stab['per_cell'][i]:.0f}%)" for i in lv], fontsize=7)
        for t, i in zip(ax.get_yticklabels(), lv):
            t.set_color(cmap(res["labels"][i] - 1))
        fig.colorbar(im, ax=ax, label="how often the two cells end up together", shrink=0.6)
        ax.set_title(f"Stability: {100 * stab['overall']:.0f}% of cell pairs keep their relation", loc="left", pad=16)
        ax.annotate("clustering repeated with each cell left out in turn · % after a name = that cell's agreement",
                    (0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points", fontsize=8,
                    color=S.MUTED, va="bottom")
        figs["stability"] = fig

    # ---- clusters against the chosen label (your own types, ephys types, in vivo clusters ...)
    lab = pd.Series(manual).replace("", np.nan)
    if lab.notna().sum() >= 3 and lab.nunique() >= 2:
        ok = lab.notna().to_numpy()
        ct = pd.crosstab(pd.Series(np.asarray(res["labels"])[ok], name="cluster"), lab[ok].reset_index(drop=True))
        ari = adjusted_rand(np.asarray(res["labels"])[ok], lab[ok])
        res["agreement"] = dict(ari=ari, n=int(ok.sum()), table=ct)
        from matplotlib.colors import LinearSegmentedColormap
        seq = LinearSegmentedColormap.from_list("poster_seq2", ["#f2f3f5", S.MERGE])
        fig = Figure(figsize=(max(5, 0.9 * ct.shape[1] + 3), max(3.5, 0.6 * ct.shape[0] + 2)), layout="constrained")
        ax = fig.add_subplot(111)
        ax.imshow(ct.to_numpy(), cmap=seq, vmin=0, aspect="auto", interpolation="nearest")
        for (r_, c_), v_ in np.ndenumerate(ct.to_numpy()):
            if v_:
                ax.text(c_, r_, int(v_), ha="center", va="center", fontsize=10,
                        color="white" if v_ > 0.6 * ct.to_numpy().max() else S.INK)
        ax.set_xticks(range(ct.shape[1]), [str(c_) for c_ in ct.columns], rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(ct.shape[0]), [f"cluster {c_}" for c_ in ct.index], fontsize=8)
        ax.set_xlabel(label_name or label_col)
        ax.set_title(f"Clusters vs labels · agreement (adjusted Rand) = {ari:.2f}", loc="left", pad=16)
        ax.annotate(f"{int(ok.sum())} labelled cells · 1 = same grouping, 0 = no better than chance",
                    (0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points", fontsize=8,
                    color=S.MUTED, va="bottom")
        figs["agreement"] = fig
    figs["dendrogram"] = figs["dendrogram"][0]
    return figs


# ------------------------------------------------------ axon initial segment
AIS_SWC_TYPE = 7   # SNT path type "Custom"


def _tree(neuron: Neuron):
    df = neuron.df
    xyz = dict(zip(df["id"], df[["x", "y", "z"]].to_numpy(float)))
    parent = dict(zip(df["id"], df["parent"]))
    ntype = dict(zip(df["id"], df["type"]))
    return xyz, parent, ntype


def _path_to_root(nid, parent):
    out, seen = [nid], {nid}
    while parent.get(out[-1], -1) in parent and parent[out[-1]] not in seen:
        out.append(parent[out[-1]]); seen.add(out[-1])
    return out


def _tree_distance(a, b, xyz, parent):
    """Distance along the tree between nodes a and b (NaN if not connected)."""
    pa, pb = _path_to_root(a, parent), _path_to_root(b, parent)
    sb = set(pb)
    lca = next((n for n in pa if n in sb), None)
    if lca is None:
        return np.nan

    def length(path, stop):
        L = 0.0
        for i in range(len(path) - 1):
            if path[i] == stop:
                break
            L += np.linalg.norm(xyz[path[i]] - xyz[path[i + 1]])
        return L
    return length(pa, lca) + length(pb, lca)


def _split_nearest_edge(neuron: Neuron, q, types):
    """Insert a node on the nearest edge (child type in ``types``) at the point closest to q."""
    df = neuron.df
    idx = {i: k for k, i in enumerate(df["id"])}
    X = df[["x", "y", "z"]].to_numpy(float)
    child = df[df["type"].isin(types) & df["parent"].isin(idx)]
    if not len(child):
        child = df[df["parent"].isin(idx)]
    if not len(child):
        d = np.linalg.norm(X - q, axis=1); i = int(np.argmin(d))
        return int(df["id"].iloc[i]), float(d[i])
    c = child[["x", "y", "z"]].to_numpy(float)
    p = X[[idx[i] for i in child["parent"]]]
    ab = c - p
    t = np.clip(np.einsum("ij,ij->i", q - p, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
    proj = p + t[:, None] * ab
    d = np.linalg.norm(proj - q, axis=1)
    k = int(np.argmin(d))
    row = child.iloc[k]
    if t[k] >= 1 - 1e-6:
        return int(row["id"]), float(d[k])
    if t[k] <= 1e-6:
        return int(row["parent"]), float(d[k])
    new_id = neuron.next_id()
    newrow = {"id": new_id, "type": int(row["type"]), "x": proj[k, 0], "y": proj[k, 1], "z": proj[k, 2],
              "radius": float(row["radius"]), "parent": int(row["parent"])}
    neuron.df.loc[neuron.df["id"] == row["id"], "parent"] = new_id
    neuron._append([[newrow[c_] for c_ in Neuron.COLS]])
    return new_id, float(d[k])


def locate_ais(neuron: Neuron, ais_type=AIS_SWC_TYPE, points_xyz=None, axon_types=(2,), snap_um=5.0):
    """Return (start_id, end_id, note). Start = end closer to the soma (along the tree)."""
    df = neuron.df
    xyz, parent, _ = _tree(neuron)
    if points_xyz is not None and len(points_xyz) >= 2:
        # snap each point onto the nearest axon edge and split that edge with a new node
        ids, note = [], ""
        for q in np.asarray(points_xyz, float)[:2]:
            nid, d = _split_nearest_edge(neuron, q, tuple(axon_types) + (ais_type,))
            ids.append(nid)
            if d > snap_um:
                note = f"a marked AIS point is {d:.1f} µm from the traced axon"
        a, b = ids
        df = neuron.df
        xyz, parent, _ = _tree(neuron)
    else:
        ais = df[df["type"] == ais_type]
        if not len(ais):
            return None, None, "no AIS"
        ids = set(ais["id"])
        # ends of the AIS stretch: nodes whose parent is not AIS / that have no AIS child
        a_cand = [n for n in ais["id"] if parent.get(n) not in ids]
        kids = set(ais["parent"])
        b_cand = [n for n in ais["id"] if n not in kids]
        a, b = a_cand[0], b_cand[0]
        if parent.get(a, -1) in xyz:
            a_par = parent[a]
            # include the edge entering the AIS so its length is not short by one segment
            a = a_par if a_par not in ids else a
        note = "" if len(a_cand) == 1 and len(b_cand) == 1 else "AIS has several pieces; using the first"
    # orient: start is the end nearer the root/soma
    if len(_path_to_root(b, parent)) < len(_path_to_root(a, parent)):
        a, b = b, a
    return a, b, note


def ais_features(neuron: Neuron, surf: BrainSurface, ais_type=AIS_SWC_TYPE, points_xyz=None,
                 dend_types=(3, 4), axon_types=(2,)) -> dict:
    if points_xyz is not None and len(points_xyz) >= 2:
        neuron = Neuron(neuron.df.copy())       # edges get split; keep the caller's tree intact
    a, b, note = locate_ais(neuron, ais_type, points_xyz, axon_types)
    if a is None:
        return {}
    xyz, parent, ntype = _tree(neuron)
    df = neuron.df
    sc = soma_center(neuron)
    soma_ids = set(df.loc[df["type"] == 1, "id"])
    # soma node to measure from along the tree: first soma node on the way to the root
    up = _path_to_root(a, parent)
    soma_node = next((n for n in up if n in soma_ids), up[-1])
    f = {"ais_length_um": _tree_distance(a, b, xyz, parent),
         "ais_length_straight_um": float(np.linalg.norm(xyz[a] - xyz[b])),
         "ais_start_dist_soma_path_um": _tree_distance(a, soma_node, xyz, parent),
         "ais_end_dist_soma_path_um": _tree_distance(b, soma_node, xyz, parent),
         "ais_start_dist_soma_straight_um": float(np.linalg.norm(xyz[a] - sc)),
         "ais_end_dist_soma_straight_um": float(np.linalg.norm(xyz[b] - sc))}
    # origin: walk from the AIS towards the soma; a dendrite node on the way = axon-carrying dendrite
    origin, stem_node = "soma", None
    for n in up[1:]:
        if n in soma_ids:
            break
        if ntype.get(n) in dend_types:
            origin, stem_node = "dendrite", n
            break
    if not soma_ids.intersection(up):
        origin = origin if stem_node else "unconnected"
        for k in ("ais_start_dist_soma_path_um", "ais_end_dist_soma_path_um"):
            f[k] = np.nan
    f["ais_origin"] = origin
    f["ais_from_dendrite"] = {"dendrite": 1.0, "soma": 0.0}.get(origin, np.nan)
    if stem_node is not None:
        f["acd_branch_dist_soma_path_um"] = _tree_distance(stem_node, soma_node, xyz, parent)
        f["acd_branch_to_ais_start_um"] = _tree_distance(stem_node, a, xyz, parent)
    mid = (xyz[a] + xyz[b]) / 2
    f["ais_depth_um"] = float(surf.signed_depth(mid)[0])
    vec = xyz[b] - xyz[a]
    if np.linalg.norm(vec) > 0:
        # 0 deg = pointing straight to the surface (dorsal), 180 = straight down (ventral)
        cosang = np.dot(vec / np.linalg.norm(vec), -surf.n)
        f["ais_angle_from_dorsal_deg"] = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    if note:
        f["ais_note"] = note
    return f


def follow_signal(vol_zyx, pts_vox, voxel, margin_um: float = 4.0):
    """Path through the bright (ankyrin) signal between consecutive clicked points.

    vol_zyx: one channel (numpy or dask, Z,Y,X). pts_vox: (N, 3) voxel coords (z, y, x).
    Each pair is joined by the cheapest path on a small crop, cost = 1 / brightness.
    Returns (M, 3) voxel coordinates.
    """
    from skimage.graph import MCP_Geometric
    vz = np.asarray(voxel, float)
    pts = np.round(np.asarray(pts_vox, float)).astype(int)
    shape = np.array(vol_zyx.shape)
    pts = np.clip(pts, 0, shape - 1)
    out = [pts[:1].astype(float)]
    m = np.ceil(margin_um / vz).astype(int)
    for s_, e_ in zip(pts[:-1], pts[1:]):
        lo = np.maximum(np.minimum(s_, e_) - m, 0)
        hi = np.minimum(np.maximum(s_, e_) + m + 1, shape)
        sub = vol_zyx[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        sub = np.asarray(sub.compute() if hasattr(sub, "compute") else sub, np.float32)
        p1, p99 = np.percentile(sub, [1, 99.8])
        norm = np.clip((sub - p1) / max(p99 - p1, 1e-6), 0, 1)
        mcp = MCP_Geometric(1.0 / (norm + 0.02), sampling=tuple(vz), fully_connected=True)
        mcp.find_costs(starts=[tuple(s_ - lo)], ends=[tuple(e_ - lo)])
        seg = np.asarray(mcp.traceback(tuple(e_ - lo)), float) + lo
        out.append(seg[1:])
    path = np.vstack(out)
    # light smoothing removes the voxel staircase, which otherwise inflates the length
    if len(path) > 7:
        from scipy.ndimage import uniform_filter1d
        sm = uniform_filter1d(path, size=5, axis=0, mode="nearest")
        sm[0], sm[-1] = path[0], path[-1]
        path = sm
    return path


def order_route(start, middle, end):
    """Order optional middle points (bends, branch point) from start to end by nearest neighbour."""
    rest = [np.asarray(p, float) for p in middle]
    out, cur = [np.asarray(start, float)], np.asarray(start, float)
    while rest:
        i = int(np.argmin([np.linalg.norm(p - cur) for p in rest]))
        cur = rest.pop(i)
        out.append(cur)
    out.append(np.asarray(end, float))
    return np.array(out)


def route_along_signal(vol_zyx, nodes_vox, voxel, margin_um: float = 4.0):
    """Follow the (fill) signal through the route nodes. Returns (path_vox, length_to_each_node_um)."""
    vz = np.asarray(voxel, float)
    paths, cum = [np.asarray(nodes_vox[:1], float)], [0.0]
    for a, b in zip(nodes_vox[:-1], nodes_vox[1:]):
        seg = follow_signal(vol_zyx, np.array([a, b]), voxel, margin_um)
        cum.append(cum[-1] + _polyline_length(seg * vz))
        paths.append(seg[1:])
    return np.vstack(paths), np.array(cum)


def _polyline_length(P):
    P = np.atleast_2d(P)
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0


def ais_features_from_points(ais_xyz, soma_exit_xyz=None, branch_xyz=None, origin: str = "soma",
                             path_xyz=None, route_to_ais_um=None, route_to_branch_um=None) -> dict:
    """AIS measured on a separate ankyrin-G scan (not registered to the SWC).

    ais_xyz: clicked points along the AIS (start -> end). path_xyz: the AIS path following the
    ankyrin staining between those clicks (if computed) - its length is the AIS length.
    soma_exit_xyz: where the axon-bearing process (axon or axon-carrying dendrite) leaves the soma.
    branch_xyz: where the axon leaves the dendrite (dendrite origin).
    """
    P = np.atleast_2d(np.asarray(ais_xyz, float))
    if len(P) < 2:
        return {}
    pick = lambda q: None if q is None or not len(np.atleast_2d(q)) else np.atleast_2d(q)[0]
    exit_, br = pick(soma_exit_xyz), pick(branch_xyz)
    path = None if path_xyz is None or len(np.atleast_2d(path_xyz)) < 2 else np.atleast_2d(np.asarray(path_xyz, float))
    ref = br if (origin == "dendrite" and br is not None) else exit_
    if ref is not None and np.linalg.norm(P[-1] - ref) < np.linalg.norm(P[0] - ref):
        P = P[::-1]                                    # start = soma side
        if path is not None:
            path = path[::-1]
    f = {"ais_source": "AnkG stack",
         "ais_length_um": _polyline_length(path) if path is not None else _polyline_length(P),
         "ais_length_clicks_um": _polyline_length(P),
         "ais_length_straight_um": float(np.linalg.norm(P[-1] - P[0])),
         "ais_length_method": "follows ankyrin signal" if path is not None else "clicked polyline",
         "ais_origin": origin,
         "ais_from_dendrite": {"dendrite": 1.0, "soma": 0.0}.get(origin, np.nan)}
    # distances from the soma exit point: along the fill signal if the route was followed,
    # otherwise straight lines (always reported as *_straight_um too)
    if exit_ is not None:
        f["ais_start_dist_soma_exit_straight_um"] = float(np.linalg.norm(P[0] - exit_))
        f["ais_end_dist_soma_exit_straight_um"] = float(np.linalg.norm(P[-1] - exit_))
        dend = origin == "dendrite" and br is not None
        straight_route = (np.linalg.norm(br - exit_) + np.linalg.norm(P[0] - br)) if dend \
            else f["ais_start_dist_soma_exit_straight_um"]
        followed = route_to_ais_um is not None
        f["ais_start_dist_soma_exit_um"] = float(route_to_ais_um) if followed else float(straight_route)
        f["ais_end_dist_soma_exit_um"] = f["ais_start_dist_soma_exit_um"] + f["ais_length_um"]
        f["route_method"] = "follows fill signal" if followed else "straight lines"
        if dend:
            f["acd_branch_dist_soma_exit_straight_um"] = float(np.linalg.norm(br - exit_))
            f["acd_branch_dist_soma_exit_um"] = float(route_to_branch_um) if (followed and route_to_branch_um is not None) \
                else f["acd_branch_dist_soma_exit_straight_um"]
            f["acd_branch_to_ais_start_um"] = f["ais_start_dist_soma_exit_um"] - f["acd_branch_dist_soma_exit_um"]
    elif origin == "dendrite" and br is not None:
        f["acd_branch_to_ais_start_um"] = float(np.linalg.norm(P[0] - br))
    return f


# ------------------------------------------------------------- Sholl, detailed
def sholl_detailed(neuron: Neuron, types, center, step_um: float = 10.0, surf: BrainSurface | None = None,
                   boundary_um=None):
    """Per Sholl shell: intersections, branch points, tips and neurite length; plus every crossing point.

    Returns (table, crossings) where crossings has columns radius_um, x, y, z (and depth_pct with a CR depth).
    """
    df = neuron.df
    c, p, _ = neuron.segments(types)
    if not len(c):
        return pd.DataFrame(), pd.DataFrame()
    center = np.asarray(center, float)
    d0 = np.linalg.norm(c - center, axis=1); d1 = np.linalg.norm(p - center, axis=1)
    rmax = max(d0.max(), d1.max())
    radii = np.arange(step_um, rmax + step_um, step_um)
    rows, pts = [], []
    sub = df[df["type"].isin(types)]
    nchild = sub.groupby("parent")["id"].count()
    xyz = sub[["x", "y", "z"]].to_numpy(float)
    dn = np.linalg.norm(xyz - center, axis=1)
    kids = sub["id"].map(nchild).fillna(0).to_numpy()
    mid_d = np.linalg.norm((c + p) / 2 - center, axis=1)
    L = np.linalg.norm(c - p, axis=1)
    for r in radii:
        cross = (np.minimum(d0, d1) < r) & (np.maximum(d0, d1) >= r)
        idx = np.where(cross)[0]
        if len(idx):
            t = (r - d1[idx]) / np.where(np.abs(d0[idx] - d1[idx]) > 1e-12, d0[idx] - d1[idx], 1)
            P = p[idx] + np.clip(t, 0, 1)[:, None] * (c[idx] - p[idx])
            pts.append(np.column_stack([np.full(len(P), r), P]))
        shell = (dn >= r - step_um) & (dn < r)
        rows.append({"radius_um": r, "intersections": int(len(idx)),
                     "branch_points": int(((kids >= 2) & shell).sum()), "tips": int(((kids == 0) & shell).sum()),
                     "length_um": float(L[(mid_d >= r - step_um) & (mid_d < r)].sum())})
    table = pd.DataFrame(rows)
    cr = pd.DataFrame(np.vstack(pts), columns=["radius_um", "x", "y", "z"]) if pts else pd.DataFrame(
        columns=["radius_um", "x", "y", "z"])
    if surf is not None and boundary_um and len(cr):
        cr["depth_pct"] = 100 * surf.signed_depth(cr[["x", "y", "z"]].to_numpy()) / boundary_um
    return table, cr


# ------------------------------------------------------------- thickness
def thickness_features(neuron: Neuron, types, prefix: str) -> dict:
    """Diameters from the SWC radius column - only meaningful if SNT fitted the radii
    ("Fit paths"); otherwise every node has the same default radius and this is flagged."""
    df = neuron.df
    sub = df[df["type"].isin(types)]
    if not len(sub):
        return {}
    r = sub["radius"].to_numpy(float)
    if np.nanstd(r) < 1e-6 or np.nanmax(r) <= 0:
        return {f"{prefix}_radii_fitted": False}
    c, p, _ = neuron.segments(types)
    idx = {i: k for k, i in enumerate(df["id"])}
    child = df[df["type"].isin(types) & df["parent"].isin(idx)]
    L = np.linalg.norm(c - p, axis=1)
    rc = child["radius"].to_numpy(float)
    out = {f"{prefix}_radii_fitted": True,
           f"{prefix}_mean_diameter_um": float(2 * np.average(rc, weights=np.maximum(L, 1e-9))) if len(rc) else np.nan}
    parent = dict(zip(df["id"], df["parent"])); ids = set(sub["id"])
    children = sub.groupby("parent")["id"].apply(list).to_dict()
    starts = [n for n in sub["id"] if parent.get(n) not in ids]
    rad = dict(zip(df["id"], df["radius"]))
    first, tips = [], []
    for s0 in starts:                          # first ~3 nodes of every primary branch
        cur, vals = s0, []
        while cur is not None and len(vals) < 3:
            vals.append(rad[cur]); ch = children.get(cur, [])
            cur = ch[0] if len(ch) == 1 else None
        first.append(np.mean(vals))
    for n in sub["id"]:
        if n not in children:
            tips.append(rad[n])
    if first:
        out[f"{prefix}_primary_diameter_um"] = float(2 * np.mean(first))
    if tips:
        out[f"{prefix}_tip_diameter_um"] = float(2 * np.mean(tips))
    if first and tips and np.mean(first) > 0:
        out[f"{prefix}_taper_ratio"] = float(np.mean(tips) / np.mean(first))
    return out


# ------------------------------------------------- neurite diameter from the image (fill channel)
def radii_sidecar(cell_path: str) -> str:
    """File next to the 63x image that keeps the diameters measured from the image."""
    import os
    return os.path.splitext(cell_path)[0] + "_radii_from_image.csv"


def pick_fill_channel(neuron: Neuron, data, voxel, n=200):
    """The channel in which the traced neurites are brightest relative to the background."""
    xyz = neuron.xyz
    if not len(xyz):
        return 0
    C, Z, Y, X = data.shape
    sel = xyz[np.linspace(0, len(xyz) - 1, min(n, len(xyz))).astype(int)]
    zi = np.clip(np.round(sel[:, 2] / voxel[0]).astype(int), 0, Z - 1)
    yi = np.clip(np.round(sel[:, 1] / voxel[1]).astype(int), 0, Y - 1)
    xi = np.clip(np.round(sel[:, 0] / voxel[2]).astype(int), 0, X - 1)
    best, score = 0, -np.inf
    for c in range(C):
        vals, bg = [], []
        for z in np.unique(zi)[:12]:                      # a few planes are enough
            pl = data[c, int(z)]
            pl = np.asarray(pl.compute() if hasattr(pl, "compute") else pl)
            m = zi == z
            vals.extend(pl[yi[m], xi[m]].astype(float))
            bg.append(float(np.median(pl[::8, ::8])))
        s = (np.median(vals) - np.median(bg)) / (np.median(bg) + 1e-6) if vals else -np.inf
        if s > score:
            best, score = c, s
    return best


def estimate_radii(neuron: Neuron, get_plane, voxel, n_planes, types=(2, 3, 4, 7), max_r_um=4.0,
                   progress=None) -> pd.DataFrame:
    """Apparent neurite diameter at every traced node, from the fill channel.

    At each node the brightness is read along a short line across the neurite (square to its course, in
    the optical section, max of the section and its two neighbours) and the width at half height (FWHM)
    is taken. Nodes where the neurite runs along z, is too dim, or has no clear edges get no value.
    Returns id, radius_um, measured. The width includes the optical blur (about 0.25-0.35 um at 63x), so
    the thinnest dendrites are over-estimated; comparisons between cells imaged alike are unaffected.
    """
    from scipy.ndimage import map_coordinates
    df = neuron.df
    ids = df["id"].to_numpy()
    pos = {i: k for k, i in enumerate(ids)}
    xyz = neuron.xyz
    par = df["parent"].to_numpy()
    child = {}
    for i, p_ in zip(ids, par):
        child.setdefault(p_, i)
    vz, vy, vx = voxel
    step = min(vx, vy) / 2.0
    offs = np.arange(-max_r_um, max_r_um + step / 2, step)
    centre = np.abs(offs) <= 1.0
    q = max(3, len(offs) // 6)
    r = np.full(len(df), np.nan)
    want = np.flatnonzero(df["type"].isin(types).to_numpy())
    zi_all = np.clip(np.round(xyz[:, 2] / vz).astype(int), 0, n_planes - 1)
    planes = {}

    def plane(z):
        if z not in planes:
            if len(planes) > 6:
                planes.pop(next(iter(planes)))
            planes[z] = np.asarray(get_plane(z), dtype=np.float32)
        return planes[z]

    zs = np.unique(zi_all[want])
    for n_done, z in enumerate(zs):
        if progress:
            progress(n_done, len(zs))
        img = plane(z)
        for dz in (-1, 1):
            if 0 <= z + dz < n_planes:
                img = np.maximum(img, plane(z + dz))
        for k in want[zi_all[want] == z]:
            a = xyz[pos[par[k]]] if par[k] in pos else xyz[k]
            b = xyz[pos[child[ids[k]]]] if ids[k] in child else xyz[k]
            t = (b - a)[:2]
            L = np.linalg.norm(t)
            if L < 1e-6 or L < 0.5 * abs((b - a)[2]):          # runs along z: no cross-section in the plane
                continue
            t /= L
            nrm = np.array([-t[1], t[0]])
            P = xyz[k, :2][None] + offs[:, None] * nrm[None]
            prof = map_coordinates(img, [P[:, 1] / vy, P[:, 0] / vx], order=1, mode="nearest")
            c_idx = np.flatnonzero(centre)
            pk = c_idx[int(np.argmax(prof[c_idx]))]
            left, right = prof[:q], prof[-q:]
            bg = min(np.median(left), np.median(right))
            noise = max(np.std(left if np.median(left) <= np.median(right) else right), 1e-6)
            amp = prof[pk] - bg
            if amp < 4 * noise or amp < 0.15 * max(bg, 1e-6):
                continue
            half = bg + 0.5 * amp
            xl = xr = None
            for j in range(pk, 0, -1):
                if prof[j - 1] < half <= prof[j]:
                    xl = offs[j] - (prof[j] - half) / max(prof[j] - prof[j - 1], 1e-9) * step
                    break
            for j in range(pk, len(prof) - 1):
                if prof[j + 1] < half <= prof[j]:
                    xr = offs[j] + (prof[j] - half) / max(prof[j] - prof[j + 1], 1e-9) * step
                    break
            if xl is None or xr is None:
                continue
            r[k] = 0.5 * (xr - xl)
    if progress:
        progress(len(zs), len(zs))
    # steady the values along each neurite: median of the node, its parent and its child
    sm = r.copy()
    for k in want:
        vals = [r[k]]
        if par[k] in pos:
            vals.append(r[pos[par[k]]])
        if ids[k] in child:
            vals.append(r[pos[child[ids[k]]]])
        vals = [v for v in vals if np.isfinite(v)]
        if np.isfinite(r[k]) and vals:
            sm[k] = float(np.median(vals))
    return pd.DataFrame({"id": ids, "radius_um": sm, "measured": np.isfinite(sm)})


def apply_radii(neuron: Neuron, radii: pd.DataFrame) -> int:
    """Put measured radii into the neuron; nodes without a value take the nearest measured value up the
    tree (else the median). Returns how many nodes were measured, 0 if the table does not fit the tracing."""
    df = neuron.df
    rad = radii[radii["measured"].astype(bool)] if "measured" in radii else radii.dropna(subset=["radius_um"])
    m = dict(zip(rad["id"].astype(int), rad["radius_um"].astype(float)))
    hit = df["id"].isin(m.keys())
    if not len(m) or hit.sum() < 0.9 * len(m):
        return 0
    med = float(np.median(list(m.values())))
    parent = dict(zip(df["id"], df["parent"]))
    out = df["radius"].to_numpy(float).copy()
    typ = df["type"].to_numpy()
    for k, i in enumerate(df["id"].to_numpy()):
        if typ[k] == 1:
            continue                                   # the soma keeps its own radius
        cur, n = i, 0
        while cur not in m and cur in parent and n < 200:
            cur = parent[cur]; n += 1
        out[k] = m.get(cur, med)
    neuron.df["radius"] = out
    neuron.radii_source = "image"
    return int(hit.sum())
