"""Brain-surface mesh from clicked points, signed depth, parallel offset surfaces.

All coordinates are (x, y, z) in micrometres, same convention as SNT .swc files.
Positive depth always means "into the brain".
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import Delaunay, cKDTree


def fit_plane(pts: np.ndarray):
    """Best-fit plane via SVD -> centroid, u, v (in-plane), n (normal)."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, vt[0], vt[1], vt[2]


def _grid_faces(keep: np.ndarray) -> np.ndarray:
    """Triangles of a regular grid; keep is a (ny, nx) bool mask of valid nodes."""
    ny, nx = keep.shape
    idx = np.arange(ny * nx).reshape(ny, nx)
    a = idx[:-1, :-1].ravel(); b = idx[:-1, 1:].ravel()
    c = idx[1:, :-1].ravel(); d = idx[1:, 1:].ravel()
    k = keep.ravel()
    t1 = np.column_stack([a, b, d]); t2 = np.column_stack([a, d, c])
    t1 = t1[k[t1].all(axis=1)]; t2 = t2[k[t2].all(axis=1)]
    return np.vstack([t1, t2])


def _order_chain(P2):
    """Order 2D points into a chain: start at one end of the point cloud, then nearest neighbour."""
    if len(P2) <= 2:
        return np.arange(len(P2))
    c = P2.mean(0)
    start = int(np.argmax(np.linalg.norm(P2 - c, axis=1)))
    left = list(range(len(P2)))
    order = [left.pop(start)]
    tree_pts = P2
    while left:
        d = np.linalg.norm(tree_pts[left] - tree_pts[order[-1]], axis=1)
        order.append(left.pop(int(np.argmin(d))))
    return np.array(order)


def _backbone(P2, cell):
    """Ordered line through a cloud of marked points (any number of overlapping strokes, any
    direction): thin to one point per `cell` um, connect them by a minimum spanning tree and take
    its longest path. Side spurs from jitter and doubled strokes drop out."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
    if len(P2) <= 2:
        return P2
    key = np.floor(P2 / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    R = np.column_stack([np.bincount(inv, P2[:, k]) for k in range(2)]) / np.bincount(inv)[:, None]
    if n <= 2:
        return R
    kq = min(12, n - 1)
    d, j = cKDTree(R).query(R, k=kq + 1)
    rows = np.repeat(np.arange(n), kq); cols = j[:, 1:].ravel(); w = d[:, 1:].ravel()
    G = coo_matrix((w + 1e-9, (rows, cols)), shape=(n, n)).tocsr()
    G = G.maximum(G.T)
    T = minimum_spanning_tree(G)
    T = T + T.T
    d0 = shortest_path(T, directed=False, indices=0)
    d0[~np.isfinite(d0)] = -1
    a = int(np.argmax(d0))
    da, pred = shortest_path(T, directed=False, indices=a, return_predecessors=True)
    da[~np.isfinite(da)] = -1
    b = int(np.argmax(da))
    path = [b]
    while path[-1] != a and pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    return R[path[::-1]]


def _resample(P, step):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] <= 0:
        return P[:1]
    n = max(2, int(np.ceil(s[-1] / step)) + 1)
    t = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(t, s, P[:, k]) for k in range(P.shape[1])])


def _smooth_curve(P, window):
    """Moving average along the curve, ends kept in place."""
    from scipy.ndimage import uniform_filter1d
    w = int(window)
    if w < 3 or len(P) < w + 2:
        return P
    Q = uniform_filter1d(P, size=w, axis=0, mode="nearest")
    Q[0], Q[-1] = P[0], P[-1]
    return Q


def _extend_curve(P, dist):
    if dist <= 0 or len(P) < 3:
        return P
    k = min(5, len(P) - 1)
    t0 = P[0] - P[k]; t1 = P[-1] - P[-1 - k]
    t0 /= max(np.linalg.norm(t0), 1e-9); t1 /= max(np.linalg.norm(t1), 1e-9)
    return np.vstack([P[0] + dist * t0, P, P[-1] + dist * t1])


def _tile(A, B, ia, ib):
    """Triangulate the band between curves A and B (shortest-diagonal rule).
    ia, ib: vertex indices of A and B. Triangles are wound A->B consistently."""
    faces = []
    i = j = 0
    n, m = len(A), len(B)
    while i < n - 1 or j < m - 1:
        if i == n - 1:
            adv_a = False
        elif j == m - 1:
            adv_a = True
        else:
            adv_a = np.linalg.norm(A[i + 1] - B[j]) <= np.linalg.norm(A[i] - B[j + 1])
        if adv_a:
            faces.append([ia[i], ia[i + 1], ib[j]]); i += 1
        else:
            faces.append([ia[i], ib[j + 1], ib[j]]); j += 1
    return faces


def _spine(groups2d, spacing, smooth_um, extend_um):
    """Smooth guide curve along the marked lines.

    Traced along the longest line drawn on a single slice (one slice gives a clean line; tracing
    through all slices at once can double back where the surface sits at different heights on
    different slices). Only lightly smoothed so sharp bends are kept."""
    best, best_len = None, -1.0
    for P2 in groups2d:
        S = _backbone(P2, cell=max(6.0, 3 * spacing))
        if len(S) < 2:
            continue
        L = float(np.linalg.norm(np.diff(S, axis=0), axis=1).sum())
        if L > best_len:
            best, best_len = S, L
    if best is None:
        c = np.vstack(groups2d).mean(0)
        best = np.vstack([c - [50, 0], c + [50, 0]])
    S = _resample(best, 5.0)
    w = max(3, int(round(smooth_um / 5.0)) | 1)
    if len(S) > w + 2:
        from scipy.ndimage import uniform_filter1d
        for _ in range(2):
            Q = uniform_filter1d(S, size=w, axis=0, mode="nearest")
            Q[0], Q[-1] = S[0], S[-1]
            S = Q
    k = min(8, len(S) - 1)
    t0 = S[0] - S[k]; t1 = S[-1] - S[-1 - k]
    t0 /= max(np.linalg.norm(t0), 1e-9); t1 /= max(np.linalg.norm(t1), 1e-9)
    n_ext = int(np.ceil(extend_um / 5.0)) + 1
    pre = S[0] + t0 * 5.0 * np.arange(n_ext, 0, -1)[:, None]
    post = S[-1] + t1 * 5.0 * np.arange(1, n_ext + 1)[:, None]
    return _resample(np.vstack([pre, S, post]), 2.0)


def _spine_coords(P2, spine):
    """(s along the guide, signed offset n from it) for 2D points."""
    A, B = spine[:-1], spine[1:]
    d, cl, k = _nearest_on_segments(P2, A, B)
    seg = np.linalg.norm(B - A, axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    s = cum[k] + np.linalg.norm(cl - A[k], axis=1)
    t = (B - A)[k] / np.maximum(seg[k], 1e-9)[:, None]
    nrm = np.column_stack([-t[:, 1], t[:, 0]])
    n = np.einsum("ij,ij->i", P2 - cl, nrm)
    return s, n


def _row_profile(s, n, grid, spacing, smoothing):
    """Offset of one slice's line on the s grid: median of everything drawn at each position
    (doubled strokes, back-and-forth scribbles), spurs removed, small gaps bridged; NaN where
    nothing was drawn."""
    from scipy.ndimage import median_filter
    nb = len(grid)
    b = np.clip(np.round((s - grid[0]) / spacing).astype(int), 0, nb - 1)
    prof = np.full(nb, np.nan)
    order = np.argsort(b)
    bs, ns = b[order], n[order]
    cuts = np.flatnonzero(np.diff(bs)) + 1
    for grp_b, grp_n in zip(np.split(bs, cuts), np.split(ns, cuts)):
        if len(grp_b):
            prof[grp_b[0]] = np.median(grp_n)
    known = np.isfinite(prof)
    if known.sum() < 2:
        return prof
    idx = np.flatnonzero(known)
    # bridge gaps up to ~60 um inside the drawn part
    gap = max(60.0, 10 * spacing) / spacing
    full = np.interp(np.arange(nb), idx, prof[idx])
    inside = (np.arange(nb) >= idx[0]) & (np.arange(nb) <= idx[-1])
    pos = np.searchsorted(idx, np.arange(nb))
    left = idx[np.clip(pos - 1, 0, len(idx) - 1)]; right = idx[np.clip(pos, 0, len(idx) - 1)]
    nearest = np.minimum(np.abs(np.arange(nb) - left), np.abs(right - np.arange(nb)))
    gap_len = right - left
    fill = inside & ~known & (gap_len <= gap)
    prof[fill] = full[fill]
    known = np.isfinite(prof)
    # spurs: points far from the local median of the line
    w = max(5, int(round(max(120.0, 6 * smoothing) / spacing)) | 1)
    run = np.where(known, prof, full)
    med = median_filter(run, size=w, mode="nearest")
    r = prof - med
    mad = np.nanmedian(np.abs(r[known])) * 1.4826 if known.any() else 0
    bad = known & (np.abs(r) > max(8.0, 4 * mad))
    if bad.any() and bad.sum() < 0.3 * known.sum():
        prof[bad] = np.nan
        known = np.isfinite(prof)
        idx = np.flatnonzero(known)
        full = np.interp(np.arange(nb), idx, prof[idx])
        prof[bad] = full[bad]
    return prof


def loft_slices(pts, spacing=2.0, smoothing=5.0, extend_um=20.0, z_range=None, z_tol=None, fade_um=200.0):
    """Surface through the lines marked on individual optical sections.

    A smooth guide curve runs along all marked lines; each slice's line is stored as its distance
    from the guide at every position s (median of everything drawn there, so doubled strokes and
    spurs do not matter). Where a slice's line is missing it follows the shape of the nearest
    drawn slice, blending into it, so there are no steps where lines of different length meet.
    Slices in between are interpolated. The mesh is a regular grid (s x marked slices): it cannot
    fold, and it works for any curvature along the guide (e.g. the SC dome).
    """
    from scipy.ndimage import gaussian_filter1d
    pts = np.asarray(pts, float)
    z = pts[:, 2]
    if z_tol is None:
        dz = np.diff(np.unique(np.round(z, 4)))
        z_tol = 0.5 * dz.min() if len(dz) else 0.5
    groups = []
    used = np.zeros(len(pts), bool)
    for zz in np.sort(np.unique(np.round(z, 4))):
        sel = (~used) & (np.abs(z - zz) <= z_tol)
        if sel.sum() >= 2:
            groups.append(pts[sel]); used |= sel
        elif sel.sum() == 1:
            used |= sel                       # a lone point on a slice cannot form a line
    if not groups:
        raise ValueError("Mark the surface as a line (at least 2 points) on at least one slice.")
    spine = _spine([g[:, :2] for g in groups], spacing, max(40.0, 4 * smoothing), extend_um + 150.0)
    zs = [float(np.median(g[:, 2])) for g in groups]
    if len(groups) >= 3:
        # second pass: guide = the typical line across slices (median offset at each position), so a
        # spur or odd stroke on the slice the first guide came from does not bend it
        rs, rn = zip(*[_spine_coords(g[:, :2], spine) for g in groups])
        seg0 = np.linalg.norm(np.diff(spine, axis=0), axis=1)
        cum0 = np.r_[0, np.cumsum(seg0)]
        grid0 = np.arange(0.0, cum0[-1], 5.0)
        F0 = np.vstack([_row_profile(s, n, grid0, 5.0, smoothing) for s, n in zip(rs, rn)])
        cnt = np.isfinite(F0).sum(0)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # positions no slice reaches
            med = np.where(cnt > 0, np.nanmedian(F0, axis=0), np.nan)
        ok = np.isfinite(med)
        if ok.sum() > 3:
            med = np.interp(np.arange(len(grid0)), np.flatnonzero(ok), med[ok])
            gx = np.interp(grid0, cum0, spine[:, 0]); gy = np.interp(grid0, cum0, spine[:, 1])
            tx = np.gradient(gx); ty = np.gradient(gy); tl = np.maximum(np.hypot(tx, ty), 1e-9)
            core = np.column_stack([gx - ty / tl * med, gy + tx / tl * med])[ok.argmax(): len(ok) - ok[::-1].argmax()]
            spine = _spine([core], 5.0, max(40.0, 4 * smoothing), extend_um + 150.0)
    rows_s, rows_n = [], []
    for g in groups:
        s, n = _spine_coords(g[:, :2], spine)
        rows_s.append(s); rows_n.append(n)
    s_lo = min(r.min() for r in rows_s) - extend_um
    s_hi = max(r.max() for r in rows_s) + extend_um
    seg = np.linalg.norm(np.diff(spine, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    s_lo, s_hi = max(s_lo, 0.0), min(s_hi, cum[-1])
    grid = np.arange(s_lo, s_hi + spacing * 0.5, spacing)
    K = len(groups)
    F = np.vstack([_row_profile(s, n, grid, spacing, smoothing) for s, n in zip(rows_s, rows_n)])
    known = np.isfinite(F)
    zs = np.array(zs)
    # fill each slice where it was not drawn: follow the nearest drawn slice, keep the offset where
    # the line ended and fade it out over fade_um
    G = F.copy()
    fade = max(fade_um, 5 * spacing) / spacing
    for k in range(K):
        miss = ~known[k]
        if not miss.any():
            continue
        kidx = np.flatnonzero(known[k])
        if not len(kidx):
            continue
        order = [o for o in np.argsort(np.abs(zs - zs[k])) if o != k]
        J = np.flatnonzero(miss)
        pos = np.searchsorted(kidx, J)
        lft = kidx[np.clip(pos - 1, 0, len(kidx) - 1)]; rgt = kidx[np.clip(pos, 0, len(kidx) - 1)]
        nb_ = np.where(np.abs(J - lft) <= np.abs(rgt - J), lft, rgt)        # where this slice's line ends
        w = np.clip(1 - np.abs(J - nb_) / fade, 0, 1)
        out = F[k, nb_].copy()                                             # no other slice: keep level
        done = np.zeros(len(J), bool)
        for o in order:
            ok = ~done & known[o, J]
            if not ok.any():
                continue
            ref = np.where(known[o, nb_[ok]], F[o, nb_[ok]], F[o, J[ok]])
            out[ok] = F[o, J[ok]] + (F[k, nb_[ok]] - ref) * w[ok]
            done |= ok
            if done.all():
                break
        G[k, J] = out
    # smooth along the line
    sig = max(smoothing, spacing) / spacing / 2.0
    if sig > 0.3:
        G = gaussian_filter1d(G, sig, axis=1, mode="nearest")
    # rows -> vertices
    sx = np.interp(grid, cum, spine[:, 0]); sy = np.interp(grid, cum, spine[:, 1])
    tx = np.gradient(sx); ty = np.gradient(sy)
    tl = np.maximum(np.hypot(tx, ty), 1e-9); tx /= tl; ty /= tl
    nx, ny = -ty, tx
    rows = [np.column_stack([sx + G[k] * nx, sy + G[k] * ny, np.full(len(grid), zs[k])]) for k in range(K)]
    # caps above / below the marked slices
    zlo, zhi = zs[0], zs[-1]
    lo = min(z_range[0], zlo) if z_range is not None else zlo - extend_um
    hi = max(z_range[1], zhi) if z_range is not None else zhi + extend_um
    if K == 1:
        lo = min(lo, zlo - max(extend_um, 1.0)); hi = max(hi, zhi + max(extend_um, 1.0))
    if lo < zlo - 1e-6:
        c = rows[0].copy(); c[:, 2] = lo; rows.insert(0, c)
    if hi > zhi + 1e-6:
        c = rows[-1].copy(); c[:, 2] = hi; rows.append(c)
    m = len(grid)
    verts = np.vstack(rows)
    idx = [np.arange(r * m, (r + 1) * m) for r in range(len(rows))]
    faces = []
    i = np.arange(m - 1)
    for r in range(len(rows) - 1):
        a0, a1, b0, b1 = idx[r][i], idx[r][i + 1], idx[r + 1][i], idx[r + 1][i + 1]
        faces.append(np.column_stack([a0, a1, b0])); faces.append(np.column_stack([a1, b1, b0]))
    loft_slices.last_normals = np.column_stack([nx, ny])     # in-plane direction square to the guide
    return verts, np.vstack(faces).astype(int), idx


class BrainSurface:
    """Surface mesh built from points marked on the brain surface.

    method:
        "slices"   - (default) one line per optical section, joined between slices; follows any
                     curvature of the surface (SC dome)
        "smooth"   - thin-plate-spline height field over the best-fit plane (only for gently
                     curved surfaces)
        "delaunay" - triangulate the raw points over the best-fit plane (original notebook)
    Positive depth = into the brain; orientation is set with orient_towards / orient_by_image / flip.
    """

    def __init__(self, pts_xyz, method: str = "slices", smoothing: float = 10.0,
                 grid_um: float = 2.0, extend_um: float = 20.0, cover_xyz=None, z_range=None):
        pts = np.asarray(pts_xyz, float)
        if len(pts) < 2:
            raise ValueError("Need more surface points.")
        self.points = pts
        self.method = method
        self.params = dict(smoothing=smoothing, grid_um=grid_um, extend_um=extend_um)
        self.centroid, self.u, self.v, self.n = fit_plane(pts)
        # "slices" surfaces are measured in the section plane: depth = distance to the surface line on
        # the same optical section, CR offset sideways in the plane (never along z)
        self.inplane = method == "slices"

        if method == "slices":
            zr = None if z_range is None else list(z_range)
            if cover_xyz is not None and len(cover_xyz):
                cz = np.asarray(cover_xyz, float)[:, 2]
                zr = [min(cz.min(), zr[0]) if zr else cz.min(), max(cz.max(), zr[1]) if zr else cz.max()]
            ext = float(extend_um)
            if cover_xyz is not None and len(cover_xyz):
                # make the lines long enough to pass every neurite point
                d = cKDTree(pts[:, :2]).query(np.asarray(cover_xyz, float)[:, :2], workers=-1)[0]
                ext = max(ext, float(d.max()) + 10.0)
            verts, faces, self._curve_idx = loft_slices(pts, spacing=max(float(grid_um), 0.1),
                                                        smoothing=float(smoothing), extend_um=ext, z_range=zr)
            self._grid_normals = loft_slices.last_normals
            self._spacing = max(float(grid_um), 0.1)
            self.mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            self._rbf = None
            return

        if len(pts) < 4:
            raise ValueError("Need at least 4 surface points.")
        if np.ptp((pts - self.centroid) @ self.v) < 1e-6:
            raise ValueError("Surface points are collinear - mark the surface on at least "
                             "two slices that are some distance apart.")
        P = pts - self.centroid
        ab = np.column_stack([P @ self.u, P @ self.v])
        h = P @ self.n
        if method == "delaunay":
            tri = Delaunay(ab)
            verts, faces = pts, tri.simplices
            self._rbf = None
        else:
            from scipy.interpolate import RBFInterpolator
            ab_u, inv = np.unique(np.round(ab, 6), axis=0, return_inverse=True)
            h_u = np.bincount(inv.ravel(), weights=h) / np.bincount(inv.ravel())
            self._rbf = RBFInterpolator(ab_u, h_u, kernel="thin_plate_spline", smoothing=float(smoothing))
            g = max(float(grid_um), 1.0)
            ab_all = ab
            if cover_xyz is not None and len(cover_xyz):
                C = np.asarray(cover_xyz, float) - self.centroid
                ab_all = np.vstack([ab, np.column_stack([C @ self.u, C @ self.v])])
            lo = ab_all.min(0) - extend_um - g
            hi = ab_all.max(0) + extend_um + g
            A, B = np.meshgrid(np.arange(lo[0], hi[0] + g, g), np.arange(lo[1], hi[1] + g, g))
            q = np.column_stack([A.ravel(), B.ravel()])
            inside = Delaunay(ab).find_simplex(q) >= 0
            dist, _ = cKDTree(ab).query(q)
            keep = inside | (dist <= max(extend_um, 1.5 * g))
            if cover_xyz is not None and len(cover_xyz):
                keep |= np.all((q >= ab_all.min(0) - extend_um - g) & (q <= ab_all.max(0) + extend_um + g), axis=1)
            H = self._rbf(q)
            verts = self.centroid + q[:, :1] * self.u + q[:, 1:] * self.v + H[:, None] * self.n
            faces = _grid_faces(keep.reshape(A.shape))
        m = trimesh.Trimesh(vertices=np.asarray(verts, float), faces=np.asarray(faces), process=False)
        m.remove_unreferenced_vertices()
        flip = (m.face_normals @ self.n) < 0          # height field: wind every face along +n
        if flip.any():
            f = m.faces.copy(); f[flip] = f[flip][:, ::-1]
            m = trimesh.Trimesh(vertices=m.vertices, faces=f, process=False)
        self.mesh = m

    # ---------------------------------------------------------- orientation
    def _invert_mesh(self):
        self.mesh = trimesh.Trimesh(vertices=self.mesh.vertices, faces=self.mesh.faces[:, ::-1], process=False)
        self.clear_caches()

    def clear_caches(self):
        """CR lines / CR surfaces depend on which side is inside: drop them when that changes."""
        for k in ("_line_cache", "_offset_cache", "_seg_cache", "_row_z", "_depth_cache"):
            self.__dict__.pop(k, None)

    def slice_segments(self, z):
        """Cut of the surface with the optical section at height z: (segments (M,2,3), face index)."""
        cache = self.__dict__.setdefault("_seg_cache", {})
        key = round(float(z), 4)
        if key not in cache:
            # the slices you drew on are rows of mesh corners; a cut exactly through a row of corners is
            # unreliable (missing / doubled pieces, small loops), so cut a hair beside it
            zc = float(z)
            rows = self.__dict__.get("_row_z")
            if rows is None:
                rows = self._row_z = np.unique(np.round(np.asarray(self.mesh.vertices)[:, 2], 6))
            if len(rows) and np.min(np.abs(rows - zc)) < 1e-5:
                zc = zc + 1e-4 if zc < rows[-1] - 1e-4 else zc - 1e-4
            try:
                segs, faces = trimesh.intersections.mesh_plane(self.mesh, [0, 0, 1], [0, 0, zc],
                                                               return_faces=True)
            except Exception:
                segs, faces = np.zeros((0, 2, 3)), np.zeros(0, int)
            if len(cache) > 2000:
                cache.clear()
            cache[key] = (segs, faces)
        return cache[key]

    def cr_line_cached(self, z, distance_um):
        """The cached CR line for this section, or False if it still has to be computed."""
        key = (round(float(z), 3), round(float(distance_um), 3))
        return self.__dict__.get("_line_cache", {}).get(key, False)

    def flip(self):
        """Swap which side counts as "inside the brain"."""
        self._invert_mesh()
        self.n = -self.n
        self.v = -self.v          # keep (u, v, n) right-handed

    def _align_plane_normal(self):
        vn = self.mesh.vertex_normals.mean(0)
        if np.dot(vn, self.n) < 0:
            self.n = -self.n; self.v = -self.v

    def orient_towards(self, inside_xyz) -> None:
        """Orient so that the given points (soma, CR border...) have positive depth."""
        inside_xyz = np.atleast_2d(np.asarray(inside_xyz, float))
        if not len(inside_xyz):
            return
        if np.median(self.signed_depth(inside_xyz)) < 0:
            self._invert_mesh()
        if np.mean((inside_xyz - self.centroid) @ self.n) < 0:
            self.n = -self.n; self.v = -self.v

    def orient_by_image(self, vol_zyx: np.ndarray, voxel_zyx, probe_um: float = 15.0) -> None:
        """Brain tissue is brighter than the space outside the slice: inside = the bright side."""
        from scipy.ndimage import map_coordinates
        vz = np.asarray(voxel_zyx, float)
        V, N = self.mesh.vertices, self.mesh.vertex_normals
        if len(V) > 3000:
            k = np.random.default_rng(0).choice(len(V), 3000, replace=False)
            V, N = V[k], N[k]

        def sample(P):
            zyx = (P[:, ::-1] / vz).T
            return map_coordinates(vol_zyx.astype(np.float32), zyx, order=1, mode="nearest")

        if self.inplane:
            N = N.copy(); N[:, 2] = 0
            N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
        if sample(V - probe_um * N).mean() > sample(V + probe_um * N).mean():
            self._invert_mesh()
        self._align_plane_normal()

    # ---------------------------------------------------------------- depth
    def signed_depth(self, xyz) -> np.ndarray:
        """Distance to the surface, positive inside the brain (Method 1).

        "slices" surfaces: distance to the surface line on the point's own optical section
        (in-plane); other surfaces: nearest point on the 3D mesh."""
        xyz = np.atleast_2d(np.asarray(xyz, float))
        if not len(xyz):
            return np.zeros(0)
        # an analysis asks for the depth of the same points many times (tables, features, figures):
        # remember the last few answers
        cache = self.__dict__.setdefault("_depth_cache", {})
        key = (xyz.shape, hash(xyz.tobytes()))
        hit = cache.get(key)
        if hit is not None:
            return hit.copy()
        out = self._inplane_depth(xyz) if getattr(self, "inplane", False) else self._mesh_depth(xyz)
        if len(cache) >= 24:
            cache.pop(next(iter(cache)))
        cache[key] = out.copy()
        return out

    def _inplane_depth(self, xyz, zq: float = 0.25):
        out = np.full(len(xyz), np.nan)
        zmin, zmax = self.mesh.vertices[:, 2].min(), self.mesh.vertices[:, 2].max()
        hz = np.clip(np.round(xyz[:, 2] / zq) * zq, zmin + 1e-6, zmax - 1e-6)
        fn = self.mesh.face_normals
        for h in np.unique(hz):
            sel = np.where(hz == h)[0]
            segs, faces = self.slice_segments(h)
            if not len(segs):
                continue
            P = xyz[sel, :2]
            d, cl, k = _nearest_on_segments(P, segs[:, 0, :2], segs[:, 1, :2])
            nxy = fn[faces[k], :2]
            sgn = np.sign(np.einsum("ij,ij->i", P - cl, nxy)); sgn[sgn == 0] = 1
            out[sel] = sgn * d
        miss = np.isnan(out)
        if miss.any():
            # above / below the surface's z range: measure on the nearest section that has a line
            # (consistent with every other depth; the 3D search is very slow for many far points)
            zr = np.asarray(self.mesh.vertices)[:, 2]
            lo, hi = zr.min() + 1e-3, zr.max() - 1e-3
            if hi > lo:
                q = xyz[miss].copy(); q[:, 2] = np.clip(q[:, 2], lo, hi)
                sub = self._inplane_depth_no_fallback(q, zq)
                bad = np.isnan(sub)
                if bad.any():
                    sub[bad] = self._mesh_depth(xyz[miss][bad])
                out[miss] = sub
            else:
                out[miss] = self._mesh_depth(xyz[miss])
        return out

    def _inplane_depth_no_fallback(self, xyz, zq):
        out = np.full(len(xyz), np.nan)
        fn = self.mesh.face_normals
        hz = np.round(xyz[:, 2] / zq) * zq
        for h in np.unique(hz):
            sel = np.where(hz == h)[0]
            segs, faces = self.slice_segments(h)
            if not len(segs):
                continue
            P = xyz[sel, :2]
            d, cl, k = _nearest_on_segments(P, segs[:, 0, :2], segs[:, 1, :2])
            nxy = fn[faces[k], :2]
            sgn = np.sign(np.einsum("ij,ij->i", P - cl, nxy)); sgn[sgn == 0] = 1
            out[sel] = sgn * d
        return out

    def along_surface(self, xyz) -> np.ndarray:
        """Position along the surface line (um, arc length on the nearest marked slice line), so
        side views of curved surfaces are 'unrolled'. Falls back to the plane axis u."""
        xyz = np.atleast_2d(np.asarray(xyz, float))
        if not (getattr(self, "inplane", False) and getattr(self, "_curve_idx", None)):
            return self.plane_coords(xyz)[:, 0]
        V = self.mesh.vertices
        curves = [V[idx] for idx in self._curve_idx if len(idx) > 1]
        zs = np.array([c[0, 2] for c in curves])
        out = np.zeros(len(xyz))
        ref = None
        for k, c in enumerate(curves):
            arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1))]
            if ref is None:
                ref = c[len(c) // 2, :2]
            # align the arc-length origin of every slice line to the point nearest the reference
            arc = arc - arc[int(np.argmin(np.linalg.norm(c[:, :2] - ref, axis=1)))]
            curves[k] = (c, arc)
        nearest = np.argmin(np.abs(xyz[:, 2][:, None] - zs[None]), axis=1)
        for k in np.unique(nearest):
            sel = nearest == k
            c, arc = curves[k]
            out[sel] = arc[cKDTree(c[:, :2]).query(xyz[sel, :2], workers=-1)[1]]
        return out

    def _offset_parallel(self, distance_um):
        """CR surface: the brain surface moved `distance_um` into the brain, square to the surface, along
        its whole length (so it covers the same stretch however deep the CR border is, also beyond the
        image). The direction is smoothed more for deeper borders so curved surfaces do not fold."""
        from scipy.ndimage import gaussian_filter1d
        cache = self.__dict__.setdefault("_offset_cache", {})
        key = ("par", round(float(distance_um), 3))
        if key in cache:
            return cache[key]
        V = np.array(self.mesh.vertices)
        m = len(self._grid_normals)
        rows = len(V) // m
        # the CR border follows the large-scale shape of the surface, not the wiggles of a hand-drawn line
        # or a hook at its end: smooth each row along its length over ~half the depth, then move it
        # square to that smooth line. (Depths are still measured from the drawn line itself.)
        R = V.reshape(rows, m, 3).copy()

        def moved(sig):
            pad = int(min(3 * sig, m - 1))
            base = np.empty_like(R[:, :, :2])
            for r in range(rows):
                P = R[r, :, :2]
                if pad > 0:                                   # straight-line extension at both ends
                    Pp = np.vstack([2 * P[0] - P[pad:0:-1], P, 2 * P[-1] - P[-2:-pad - 2:-1]])
                    base[r] = gaussian_filter1d(Pp, sig, axis=0, mode="nearest")[pad:pad + m]
                else:
                    base[r] = P
            T = np.gradient(base, axis=1)
            T /= np.maximum(np.linalg.norm(T, axis=2, keepdims=True), 1e-9)
            Ns = np.stack([-T[..., 1], T[..., 0]], axis=2)   # square to the smooth line
            # which side is "into the brain": try both and keep the one whose points come out at about
            # the CR depth (testing one point near a strongly smoothed line can land on the wrong side)
            rr = np.linspace(0, rows - 1, min(rows, 4)).astype(int)
            cc = np.linspace(0, m - 1, min(m, 25)).astype(int)
            best = None
            for sgn in (1.0, -1.0):
                Q = base[np.ix_(rr, cc)] + Ns[np.ix_(rr, cc)] * (sgn * float(distance_um))
                q = np.column_stack([Q.reshape(-1, 2), np.repeat(R[rr, 0, 2], len(cc))])
                err = np.nanmedian(np.abs(self.signed_depth(q) - float(distance_um)))
                if best is None or err < best[0]:
                    best = (err, sgn)
            W = R.copy()
            W[:, :, :2] = base + Ns * (best[1] * float(distance_um))
            return W.reshape(-1, 3)

        # start smoothing over half the depth; where the moved line still loops (a bend tighter than
        # that, e.g. a hook at the end of the line), smooth more
        sig = max(abs(distance_um) / 2.0, 4.0) / max(self._spacing, 1e-6)
        sig_max = max(abs(distance_um), 8.0) / max(self._spacing, 1e-6)    # stay close to the real shape
        while True:
            W = moved(sig)
            if sig >= sig_max or not any(_self_loops(W[r * m:(r + 1) * m, :2]).any() for r in range(rows)):
                break
            sig = min(sig * 1.6, sig_max)
        faces = np.asarray(self.mesh.faces)
        # inside a bend the moved line runs past itself (a loop, or two arms crossing at a sharp corner):
        # cut each row where it crosses itself and drop the part in between, as for any parallel line
        bad = np.zeros(len(V), bool)
        for r in range(rows):
            bad[r * m:(r + 1) * m] = _self_loops(W[r * m:(r + 1) * m, :2])
        if bad.any():
            faces = faces[~bad[faces].any(axis=1)]
        mesh = trimesh.Trimesh(vertices=W, faces=faces, process=False)
        cache[key] = mesh
        return mesh

    def _offset_inplane(self, distance_um):
        """CR surface for in-plane surfaces: on every slice line level, the line at in-plane distance
        `distance_um` inside the brain (contour of the signed distance field - a true parallel line,
        independent of the order of the marked points); the levels are then joined like the surface."""
        cache = self.__dict__.setdefault("_offset_cache", {})
        key = round(float(distance_um), 3)
        if key in cache:
            return cache[key]
        V = self.mesh.vertices
        zlo, zhi = V[:, 2].min(), V[:, 2].max()
        levels = sorted({round(float(V[idx[0], 2]), 4) for idx in self._curve_idx if len(idx)})
        curves, done = [], {}
        for z in levels:
            zq = min(max(z, zlo + 1e-3), zhi - 1e-3)
            P = self._parallel_line(zq, distance_um)
            if P is None or len(P) < 2:
                continue
            P = _resample(P, self._spacing)
            curves.append(np.column_stack([P, np.full(len(P), z)]))
        if len(curves) < 2:
            if len(curves) == 1:
                c2 = curves[0].copy(); c2[:, 2] = zhi if curves[0][0, 2] < zhi else zlo
                curves.append(c2); curves.sort(key=lambda c: c[0, 2])
            else:
                return self._offset_along_normals(distance_um)
        for k in range(1, len(curves)):
            A, B = curves[k - 1], curves[k]
            if (np.linalg.norm(A[0] - B[0]) + np.linalg.norm(A[-1] - B[-1]) >
                    np.linalg.norm(A[0] - B[-1]) + np.linalg.norm(A[-1] - B[0])):
                curves[k] = B[::-1]
        verts, faces, off, idx = [], [], 0, []
        for c in curves:
            verts.append(c); idx.append(np.arange(off, off + len(c))); off += len(c)
        for k in range(len(curves) - 1):
            faces += _tile(curves[k], curves[k + 1], idx[k], idx[k + 1])
        m = trimesh.Trimesh(vertices=np.vstack(verts), faces=np.array(faces, int), process=False)
        cache[key] = m
        return m

    def cr_line_on_slice(self, z, distance_um):
        """CR line on one optical section (cached) - for display while scrolling."""
        cache = self.__dict__.setdefault("_line_cache", {})
        key = (round(float(z), 3), round(float(distance_um), 3))
        if key not in cache:
            V = self.mesh.vertices
            zq = min(max(float(z), V[:, 2].min() + 1e-3), V[:, 2].max() - 1e-3)
            if getattr(self, "_grid_normals", None) is not None:
                cache[key] = self._parallel_row(zq, distance_um)
            else:
                cache[key] = self._parallel_line(zq, distance_um)
        return cache[key]

    def _parallel_row(self, z, distance_um):
        """CR line on section z, taken from the parallel CR surface (same line as in 3D)."""
        m = len(self._grid_normals)
        M = self.offset_mesh(distance_um)
        W = np.asarray(M.vertices).reshape(-1, m, 3)
        zr = W[:, 0, 2]
        r = int(np.clip(np.searchsorted(zr, z) - 1, 0, len(zr) - 2))
        t = 0.0 if zr[r + 1] == zr[r] else float(np.clip((z - zr[r]) / (zr[r + 1] - zr[r]), 0, 1))
        P = (1 - t) * W[r, :, :2] + t * W[r + 1, :, :2]
        F = np.asarray(M.faces)
        band = F[((F // m) == r).any(axis=1) & ((F // m) == r + 1).any(axis=1)]   # faces between rows r, r+1
        used = np.zeros(m, bool)
        used[np.unique(band % m)] = True
        P = P[used]
        return P if len(P) >= 2 else None

    def _parallel_line(self, z, distance_um):
        """2D line at signed in-plane distance `distance_um` from the surface line on section z."""
        from scipy.spatial import cKDTree
        from skimage.measure import find_contours
        segs, faces = self.slice_segments(z)
        if not len(segs):
            return None
        res = max(1.0, abs(distance_um) / 60.0)
        A, B = segs[:, 0, :2], segs[:, 1, :2]
        nrm = self.mesh.face_normals[faces][:, :2]
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        # dense samples along the line, each with its inward normal
        L = np.linalg.norm(B - A, axis=1)
        k = np.maximum(1, np.ceil(L / (res / 2)).astype(int))
        seg = np.repeat(np.arange(len(A)), k + 1)                 # dense samples, vectorised
        start = np.repeat(np.cumsum(k + 1) - (k + 1), k + 1)
        frac = (np.arange(len(seg)) - start) / k[seg]
        S = A[seg] + frac[:, None] * (B[seg] - A[seg])
        Ns = nrm[seg]
        lo = S.min(0) - abs(distance_um) - 3 * res
        hi = S.max(0) + abs(distance_um) + 3 * res
        gx = np.arange(lo[0], hi[0], res); gy = np.arange(lo[1], hi[1], res)
        if len(gx) * len(gy) > 4e6:                       # keep the grid manageable
            res *= np.sqrt(len(gx) * len(gy) / 4e6)
            gx = np.arange(lo[0], hi[0], res); gy = np.arange(lo[1], hi[1], res)
        GX, GY = np.meshgrid(gx, gy)
        G = np.column_stack([GX.ravel(), GY.ravel()])
        # nearest line sample for every grid point: a distance transform finds the nearest occupied
        # pixel (fast, linear time); the distance is then measured exactly to the sample in it
        from scipy.ndimage import distance_transform_edt
        ij = np.clip(np.round((S - [gx[0], gy[0]]) / res).astype(int), 0, [len(gx) - 1, len(gy) - 1])
        owner = np.full(GX.shape, -1, int)
        owner[ij[:, 1], ij[:, 0]] = np.arange(len(S))
        _, (ri, ci) = distance_transform_edt(owner < 0, return_indices=True)
        i = owner[ri, ci].ravel()
        d = np.linalg.norm(G - S[i], axis=1)
        band = abs(distance_um) + 4 * res
        far = d > band
        vec = G - S[i]
        dot = np.einsum("ij,ij->i", vec, Ns[i])
        signed = np.sign(dot) * d
        # no round caps past the two ends of the line: drop grid points whose nearest surface point
        # is (next to) an end of the line
        ends = np.vstack([A, B])
        key = np.round(ends, 5)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        endpts = ends[cnt[inv.ravel()] == 1]
        cap = np.zeros(len(G), bool)
        if len(endpts):
            # grid points near either end: find their exact nearest sample (few points) and drop
            # those whose nearest line point is (next to) an end, as before
            near_end = cKDTree(endpts).query(S)[0] <= 2 * res
            cand = np.where(cKDTree(endpts).query(S[i])[0] <= 8 * res + band)[0]
            cand = cand[~far[cand]]
            if len(cand):
                _, ii = cKDTree(S).query(G[cand], workers=-1)
                cap[cand] = near_end[ii]
        F = signed.reshape(GX.shape)
        mask = (~cap & ~far).reshape(GX.shape)
        try:
            cs = find_contours(F, level=float(distance_um), mask=mask)
        except Exception:
            return None
        if not cs:
            return None
        c = max(cs, key=len)
        return np.column_stack([gx[0] + c[:, 1] * res, gy[0] + c[:, 0] * res])

    def _offset_along_normals(self, distance_um):
        V = np.array(self.mesh.vertices)
        vn = np.array(self.mesh.vertex_normals); vn[:, 2] = 0
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)
        return trimesh.Trimesh(vertices=V + distance_um * vn, faces=self.mesh.faces, process=False)

    def _mesh_depth(self, xyz):
        closest, dist, tri = trimesh.proximity.closest_point(self.mesh, xyz)
        s = np.sign(np.einsum("ij,ij->i", xyz - closest, self.mesh.face_normals[tri]))
        s[s == 0] = 1
        return s * dist

    def plane_depth(self, xyz) -> np.ndarray:
        """Projection onto the best-fit plane normal (Method 2)."""
        return (np.atleast_2d(xyz) - self.centroid) @ self.n

    def plane_coords(self, xyz) -> np.ndarray:
        """(u, v, n) coordinates in the best-fit plane frame (Method 3 uses v)."""
        P = np.atleast_2d(xyz) - self.centroid
        return np.column_stack([P @ self.u, P @ self.v, P @ self.n])

    def footprint_fraction(self, xyz) -> np.ndarray:
        """True where the nearest surface point is a perpendicular foot (not the mesh edge)."""
        xyz = np.atleast_2d(np.asarray(xyz, float))
        if not len(xyz):
            return np.zeros(0, bool)
        if getattr(self, "inplane", False):
            # in-plane surfaces: on each section, is the nearest point on the line a square foot
            # (not one of the line's ends)?
            zr = np.asarray(self.mesh.vertices)[:, 2]
            q = xyz.copy(); q[:, 2] = np.clip(q[:, 2], zr.min() + 1e-3, zr.max() - 1e-3)
            out = np.zeros(len(q), bool)
            hz = np.round(q[:, 2] / 0.25) * 0.25
            for h in np.unique(hz):
                sel = np.where(hz == h)[0]
                segs, _ = self.slice_segments(h)
                if not len(segs):
                    continue
                A, B = segs[:, 0, :2], segs[:, 1, :2]
                d, cl, k = _nearest_on_segments(q[sel, :2], A, B)
                t = (B - A)[k]
                t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
                v = q[sel, :2] - cl
                nv = np.linalg.norm(v, axis=1)
                cos = np.abs(np.einsum("ij,ij->i", v, t)) / np.maximum(nv, 1e-9)
                out[sel] = (cos < 0.2) | (nv < 1e-6)
            return out
        closest, dist, tri = trimesh.proximity.closest_point(self.mesh, xyz)
        d = xyz - closest
        nd = np.linalg.norm(d, axis=1)
        cos = np.abs(np.einsum("ij,ij->i", d, self.mesh.face_normals[tri])) / np.maximum(nd, 1e-9)
        return (cos > 0.98) | (nd < 1e-6)

    def offset_mesh(self, distance_um: float) -> trimesh.Trimesh:
        """Parallel surface ``distance_um`` deeper into the brain.

        Directions come from a more strongly smoothed copy, so clicking jitter on the surface is not
        amplified at depth; the surface itself is not changed."""
        if getattr(self, "_grid_normals", None) is not None and getattr(self, "_curve_idx", None):
            return self._offset_parallel(distance_um)
        if getattr(self, "inplane", False) and getattr(self, "_curve_idx", None):
            return self._offset_inplane(distance_um)
        sm = self.mesh.copy()
        try:
            trimesh.smoothing.filter_taubin(sm, iterations=30)
            vn = np.array(sm.vertex_normals)
        except Exception:
            vn = np.array(self.mesh.vertex_normals)
        V = self.mesh.vertices + distance_um * vn
        return trimesh.Trimesh(vertices=V, faces=self.mesh.faces, process=False)


def _self_loops(P):
    """For a polyline P (m, 2): True for the points inside the loops it makes where it crosses itself.
    Candidate segment pairs come from a KD-tree on segment midpoints."""
    from scipy.spatial import cKDTree
    m = len(P)
    out = np.zeros(m, bool)
    if m < 4:
        return out
    A, B = P[:-1], P[1:]
    L = np.linalg.norm(B - A, axis=1)
    mid = 0.5 * (A + B)
    pairs = cKDTree(mid).query_pairs(max(float(L.max()), 1e-9), output_type="ndarray")
    if not len(pairs):
        return out
    i, j = np.minimum(pairs[:, 0], pairs[:, 1]), np.maximum(pairs[:, 0], pairs[:, 1])
    keep = j > i + 1
    i, j = i[keep], j[keep]

    def cr(u, v):
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    d1 = cr(B[i] - A[i], A[j] - A[i]); d2 = cr(B[i] - A[i], B[j] - A[i])
    d3 = cr(B[j] - A[j], A[i] - A[j]); d4 = cr(B[j] - A[j], B[i] - A[j])
    hit = (d1 * d2 < 0) & (d3 * d4 < 0)
    for a, b in zip(i[hit], j[hit]):
        out[a + 1:b + 1] = True                 # the loop between the two crossing segments
    return out


def _nearest_on_segments(P, A, B, k=64):
    """Distance, closest point and index of the nearest 2D segment (A[i], B[i]) for every point in P.
    Candidates come from a KD-tree on the segment midpoints, so this scales to long lines."""
    from scipy.spatial import cKDTree
    P = np.atleast_2d(P)
    if len(P) > 4000:                              # keep the candidate arrays small
        parts = [_nearest_on_segments(P[i:i + 4000], A, B, k) for i in range(0, len(P), 4000)]
        return tuple(np.concatenate(x) for x in zip(*parts))
    if not len(A):
        return np.full(len(P), np.inf), np.zeros_like(P), np.zeros(len(P), int)
    k = int(min(k, len(A)))
    mid = 0.5 * (A + B)
    half = 0.5 * np.linalg.norm(B - A, axis=1).max()
    tree = cKDTree(mid)
    _, cand = tree.query(P, k=k, workers=-1 if len(P) > 2000 else 1)
    cand = np.asarray(cand).reshape(len(P), k)
    AB = B[cand] - A[cand]
    t = np.einsum("pmk,pmk->pm", P[:, None, :] - A[cand], AB) / np.maximum(np.einsum("pmk,pmk->pm", AB, AB), 1e-12)
    C = A[cand] + np.clip(t, 0, 1)[..., None] * AB
    d = np.linalg.norm(P[:, None, :] - C, axis=2)
    j = np.argmin(d, axis=1)
    r = np.arange(len(P))
    best, cl, idx = d[r, j], C[r, j], cand[r, j]
    # a segment whose midpoint was not among the k candidates can only be closer if it lies within
    # best + half its length; check those few points exactly
    if k < len(A):
        far_mid = np.linalg.norm(P - mid[cand[:, -1]], axis=1)
        redo = np.where(far_mid < best + half)[0]
        if len(redo):
            balls = tree.query_ball_point(P[redo], best[redo] + half + 1e-9, workers=-1)
            for q, c in zip(redo, balls):
                c = np.asarray(c, int)
                ab = B[c] - A[c]
                tq = np.clip(np.einsum("mk,mk->m", P[q] - A[c], ab) / np.maximum(np.einsum("mk,mk->m", ab, ab), 1e-12), 0, 1)
                cq = A[c] + tq[:, None] * ab
                dq = np.linalg.norm(cq - P[q], axis=1)
                m = int(np.argmin(dq))
                if dq[m] < best[q]:
                    best[q], cl[q], idx[q] = dq[m], cq[m], c[m]
    return best, cl, idx


# ----------------------------------------------------- low-mag CR-border distance
def _order_polyline(pts2d: np.ndarray) -> np.ndarray:
    c = pts2d.mean(0)
    _, _, vt = np.linalg.svd(pts2d - c, full_matrices=False)
    return pts2d[np.argsort((pts2d - c) @ vt[0])]


def _dist_to_polyline(p: np.ndarray, line: np.ndarray) -> float:
    a, b = line[:-1], line[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
    proj = a + t[:, None] * ab
    return float(np.min(np.linalg.norm(proj - p, axis=1)))


def measure_border_distance(surface_xyz, border_xyz, z_tol_um: float | None = None,
                            smoothing: float = 5.0) -> dict:
    """Perpendicular distance from the brain surface to the Calretinin border.

    If the surface points come from a single optical section (or a few
    near-identical ones) the surface is a curve: each border point is measured
    to the surface polyline drawn on its own slice (2D mode).  Otherwise a
    smooth 3D surface is fitted and each border point's signed depth is used.
    """
    S = np.asarray(surface_xyz, float)
    B = np.asarray(border_xyz, float)
    if len(S) < 2 or len(B) < 1:
        raise ValueError("Mark at least 2 surface points and 1 border point.")
    zs = np.unique(np.round(S[:, 2], 3))
    mode = "3D"
    dists = None
    if len(zs) >= 2 and len(S) >= 6:
        try:
            surf = BrainSurface(S, method="slices", smoothing=smoothing,
                                grid_um=max(1.0, np.ptp(S[:, :2]) / 150), extend_um=50.0)
            surf.orient_towards(B)
            dists = surf.signed_depth(B)
        except ValueError:
            dists = None
    if dists is None:
        mode = "2D (per slice)"
        dists = []
        tol = z_tol_um if z_tol_um is not None else 1e-3
        for p in B:
            same = S[np.abs(S[:, 2] - p[2]) <= tol]
            if len(same) < 2:
                # nearest slice that has surface points
                zn = zs[np.argmin(np.abs(zs - p[2]))]
                same = S[np.abs(S[:, 2] - zn) <= 1e-3]
            if len(same) < 2:
                same = S
            line = _order_polyline(same[:, :2])
            dists.append(_dist_to_polyline(p[:2], line))
        dists = np.asarray(dists)
    return dict(mode=mode, n=int(len(dists)), mean=float(np.mean(dists)),
                median=float(np.median(dists)), sd=float(np.std(dists, ddof=1)) if len(dists) > 1 else 0.0,
                values=dists.tolist())
