"""Soma-seeded automatic first draft of a tracing, to be corrected in SNT.

From a click on the soma:
  1. the fill channel is read in a box around the soma (reduced in xy so it fits in memory),
  2. the slowly varying background is removed and everything brighter than `sensitivity` x noise that is
     connected to the soma is kept,
  3. the cheapest path (through bright voxels) from the soma to every kept voxel is computed in one pass,
  4. branches are taken longest first: the farthest voxel not yet explained is followed back to the tree;
     voxels around an accepted branch count as explained, branches shorter than `min_branch_um` are dropped.
The result is a tree of centre lines with a radius at every node (SWC). All branches are tagged dendrite:
telling axon from dendrite, cutting false joins with neighbouring cells and adding missed dim branches is
left to proof-editing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .swc import Neuron


def _plane(data, c, z):
    pl = data[c, int(z)]
    return np.asarray(pl.compute() if hasattr(pl, "compute") else pl)


def pick_channel_at(data, voxel, seed_xyz):
    """The channel that is brightest at the soma relative to the rest of that optical section."""
    C, Z, Y, X = data.shape
    z = int(np.clip(round(seed_xyz[2] / voxel[0]), 0, Z - 1))
    y = int(np.clip(round(seed_xyz[1] / voxel[1]), 0, Y - 1))
    x = int(np.clip(round(seed_xyz[0] / voxel[2]), 0, X - 1))
    best, score = 0, -np.inf
    for c in range(C):
        pl = _plane(data, c, z)
        patch = pl[max(0, y - 6):y + 7, max(0, x - 6):x + 7].astype(float)
        bg = float(np.median(pl[::8, ::8]))
        sd = float(np.std(pl[::8, ::8])) + 1e-6
        s = (float(np.median(patch)) - bg) / sd
        if s > score:
            best, score = c, s
    return best


def load_box(data, voxel, channel, seed_xyz, reach_um=300.0, max_voxels=2.5e7, progress=None):
    """Fill channel in a box of +-reach around the soma (all sections), block-averaged in xy.
    Returns (volume float32 (Z, y, x), spacing (z, y, x) um, origin (x0, y0) um)."""
    C, Z, Y, X = data.shape
    vz, vy, vx = voxel
    x0 = int(max(0, np.floor((seed_xyz[0] - reach_um) / vx))); x1 = int(min(X, np.ceil((seed_xyz[0] + reach_um) / vx)))
    y0 = int(max(0, np.floor((seed_xyz[1] - reach_um) / vy))); y1 = int(min(Y, np.ceil((seed_xyz[1] + reach_um) / vy)))
    f = max(1, int(round(0.35 / min(vx, vy))))
    while Z * ((y1 - y0) // f) * ((x1 - x0) // f) > max_voxels:
        f += 1
    ny, nx = (y1 - y0) // f, (x1 - x0) // f
    vol = np.empty((Z, ny, nx), np.float32)
    for z in range(Z):
        if progress:
            progress(f"Reading the image: section {z + 1} of {Z}")
        pl = _plane(data, channel, z)[y0:y0 + ny * f, x0:x0 + nx * f].astype(np.float32)
        vol[z] = pl.reshape(ny, f, nx, f).mean(axis=(1, 3)) if f > 1 else pl
    return vol, (vz, vy * f, vx * f), (x0 * vx + (f - 1) * vx / 2, y0 * vy + (f - 1) * vy / 2)


def prepare(vol, spacing):
    """Background removed, lightly smoothed, in units of the noise (so a threshold of 5 means 5 x noise)."""
    from scipy.ndimage import gaussian_filter
    out = np.empty_like(vol)
    sig = (12.0 / spacing[1], 12.0 / spacing[2])               # background varies over > ~12 um
    for z in range(len(vol)):
        out[z] = vol[z] - gaussian_filter(vol[z], sig, mode="nearest")
    out = gaussian_filter(out, (0.5, 0.7, 0.7))
    sub = out[::2, ::3, ::3]
    med = float(np.median(sub))
    noise = 1.4826 * float(np.median(np.abs(sub - med))) + 1e-6
    return (out - med) / noise


def trace(snr, spacing, origin, seed_xyz, sensitivity=5.0, min_branch_um=8.0, max_branches=600, progress=None):
    """Tree from the soma through the bright voxels connected to it. Returns (Neuron, info dict)."""
    from scipy import ndimage as ndi
    from skimage.graph import MCP_Geometric
    sz, sy, sx = spacing
    Z, Y, X = snr.shape
    seed = np.array([np.clip(round(seed_xyz[2] / sz), 0, Z - 1),
                     np.clip(round((seed_xyz[1] - origin[1]) / sy), 0, Y - 1),
                     np.clip(round((seed_xyz[0] - origin[0]) / sx), 0, X - 1)], int)
    if progress:
        progress("Finding what is connected to the soma…")
    mask = snr > float(sensitivity)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3, 3), bool))           # bridge one-voxel gaps
    lab, n = ndi.label(mask, structure=np.ones((3, 3, 3), bool))
    if not lab[tuple(seed)]:
        # the click was a little off (or the soma centre is dimmer than its rim): nearest bright voxel
        zz, yy, xx = np.nonzero(lab[max(0, seed[0] - 3):seed[0] + 4])
        if not len(zz):
            raise ValueError("Nothing bright enough near the click - lower the sensitivity value or pick "
                             "the fill channel by hand.")
        zz = zz + max(0, seed[0] - 3)
        d = (zz - seed[0]) ** 2 * sz ** 2 + (yy - seed[1]) ** 2 * sy ** 2 + (xx - seed[2]) ** 2 * sx ** 2
        k = int(np.argmin(d))
        if d[k] > 15.0 ** 2:
            raise ValueError("Nothing bright enough within 15 µm of the click - click on the soma, lower the "
                             "sensitivity value or pick the fill channel by hand.")
        seed = np.array([zz[k], yy[k], xx[k]])
    comp = lab == lab[tuple(seed)]
    del lab, mask
    n_vox = int(comp.sum())
    # local radius of the bright structure (um), for node radii and for 'explained' neighbourhoods
    edt = ndi.distance_transform_edt(comp, sampling=spacing).astype(np.float32)
    r_soma = float(np.clip(edt[tuple(seed)], 2.0, 15.0))
    if progress:
        progress(f"Following the processes ({n_vox:,} bright voxels)…")
    cost = np.full(snr.shape, np.inf, np.float32)
    # cheap along the bright centre of a process, expensive at its dim edge
    cost[comp] = 1.0 / np.clip(snr[comp] * np.clip(edt[comp] / max(sx, sy), 0.5, 3.0), 1.0, None)
    mcp = MCP_Geometric(cost, sampling=spacing, fully_connected=True)
    mcp.find_costs([tuple(seed)])
    # candidates for branch ends are taken farthest first, by path LENGTH inside the structure from the
    # soma (not by brightness-weighted cost: dim voxels on the rim of a thick junction are costly to
    # reach and would be mistaken for far ends)
    cost[comp] = 1.0
    far, _ = MCP_Geometric(cost, sampling=spacing, fully_connected=True).find_costs([tuple(seed)])
    del cost
    idx = np.flatnonzero(comp.ravel())
    order = idx[np.argsort(far.ravel()[idx])[::-1]]
    del far
    covered = np.zeros(snr.shape, bool)
    node_of = {}                                                    # flat voxel index -> node number
    nodes = [[*seed, r_soma, -1, 1]]                                # z, y, x (voxels), radius um, parent, type
    node_of[int(np.ravel_multi_index(tuple(seed), snr.shape))] = 0
    step_um = np.array(spacing)

    def stamp(path, extra=0.0):
        for (z, y, x) in path[:: 2]:
            r = float(edt[z, y, x]) * 1.6 + 1.5 * max(sx, sy) + extra
            rz, ry, rx = int(np.ceil(r / sz)), int(np.ceil(r / sy)), int(np.ceil(r / sx))
            covered[max(0, z - rz):z + rz + 1, max(0, y - ry):y + ry + 1, max(0, x - rx):x + rx + 1] = True

    # everything inside the soma is explained from the start
    rz, ry, rx = (int(np.ceil(1.3 * r_soma / s)) for s in spacing)
    covered[max(0, seed[0] - rz):seed[0] + rz + 1, max(0, seed[1] - ry):seed[1] + ry + 1,
            max(0, seed[2] - rx):seed[2] + rx + 1] = True
    branches, total, log = 0, 0.0, []
    cov_flat = covered.ravel()
    for flat in order:
        if cov_flat[flat]:
            continue
        tip = np.unravel_index(flat, snr.shape)
        path = np.asarray(mcp.traceback(tip), int)[::-1]            # tip ... seed
        flats = np.ravel_multi_index(path.T, snr.shape)
        join = next((k for k, f_ in enumerate(flats) if int(f_) in node_of), len(path) - 1)
        new = path[:join]                                           # tip -> just before the tree
        if not len(new):
            cov_flat[flat] = True
            continue
        seg = np.vstack([new, path[join][None]]).astype(float) * step_um
        length = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())
        # only the part that leaves what is already explained counts: inside a thick process many
        # cheapest paths run side by side, and those must not become branches of their own
        steps = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        free = float(steps[~covered[new[:, 0], new[:, 1], new[:, 2]]].sum())
        if free < min_branch_um:
            stamp(new)
            continue
        # smooth the centre line a little (ends stay), then add the nodes from the tree outwards
        P = new[::-1].astype(float)
        if len(P) >= 5:
            from scipy.ndimage import uniform_filter1d
            Q = uniform_filter1d(P, size=5, axis=0, mode="nearest")
            Q[0], Q[-1] = P[0], P[-1]
            P = Q
        parent = node_of[int(flats[join])]
        for k, (vox, p) in enumerate(zip(new[::-1], P)):
            nodes.append([p[0], p[1], p[2], max(float(edt[tuple(vox)]), 0.5 * min(sx, sy)), parent, 3])
            parent = len(nodes) - 1
            node_of[int(np.ravel_multi_index(tuple(vox), snr.shape))] = parent
        stamp(new)
        log.append((length, free, tuple(new[0]), tuple(path[join])))
        branches += 1
        total += length
        if progress and branches % 10 == 0:
            progress(f"Following the processes: {branches} branches, {total:.0f} µm so far…")
        if branches >= max_branches:
            break
    neuron = _to_neuron(nodes, spacing, origin, keep_every_um=1.0, r_soma=r_soma)
    info = dict(branches=branches, length_um=total, soma_radius_um=r_soma, bright_voxels=n_vox,
                spacing=spacing, capped=branches >= max_branches, log=log)
    return neuron, info


def _to_neuron(nodes, spacing, origin, keep_every_um=1.0, r_soma=5.0) -> Neuron:
    """Voxel tree -> SWC in um: keep the soma, branch points, tips and a node about every `keep_every_um`."""
    A = np.array(nodes, float)
    n = len(A)
    parent = A[:, 4].astype(int)
    xyz = np.column_stack([A[:, 2] * spacing[2] + origin[0], A[:, 1] * spacing[1] + origin[1], A[:, 0] * spacing[0]])
    nchild = np.bincount(parent[parent >= 0], minlength=n)
    keep = (nchild != 1) | (parent < 0)
    # nodes inside the soma are dropped: branches start at its surface
    inside = np.linalg.norm(xyz - xyz[0], axis=1) < r_soma
    inside[0] = False
    run = np.zeros(n)
    for i in range(1, n):                                           # parents always come before children
        p = parent[i]
        d = np.linalg.norm(xyz[i] - xyz[p])
        run[i] = 0.0 if keep[p] else run[p]
        run[i] += d
        if run[i] >= keep_every_um:
            keep[i] = True
            run[i] = 0.0
    keep &= ~inside
    keep[0] = True
    new_id = -np.ones(n, int)
    rows = []
    for i in range(n):
        if not keep[i]:
            continue
        p = parent[i]
        while p >= 0 and not keep[p]:
            p = parent[p]
        new_id[i] = len(rows) + 1
        rows.append([new_id[i], int(A[i, 5]), xyz[i, 0], xyz[i, 1], xyz[i, 2], float(A[i, 3]),
                     int(new_id[p]) if p >= 0 else -1])
    # tips that ended up with no length (everything inside the soma) are removed
    df = pd.DataFrame(rows, columns=Neuron.COLS)
    return Neuron(df)


def autotrace(data, voxel, seed_xyz, channel=None, reach_um=300.0, sensitivity=5.0, min_branch_um=8.0,
              progress=None, cache=None):
    """Everything in one call. `cache` (a dict) keeps the prepared image, so trying another sensitivity
    or branch length does not read the image again."""
    c = pick_channel_at(data, voxel, seed_xyz) if channel is None else int(channel)
    key = (id(data), c, tuple(np.round(seed_xyz, 1)), float(reach_um))
    if cache is not None and cache.get("key") == key:
        snr, spacing, origin = cache["snr"], cache["spacing"], cache["origin"]
    else:
        vol, spacing, origin = load_box(data, voxel, c, seed_xyz, reach_um, progress=progress)
        if progress:
            progress("Removing the background…")
        snr = prepare(vol, spacing)
        del vol
        if cache is not None:
            cache.clear()
            cache.update(key=key, snr=snr, spacing=spacing, origin=origin)
    neuron, info = trace(snr, spacing, origin, seed_xyz, sensitivity, min_branch_um, progress=progress)
    info["channel"] = c
    return neuron, info
