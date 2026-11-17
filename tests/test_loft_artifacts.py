"""Hand-drawn surface lines that differ between slices: no combs, steps or spikes in between."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from arborist.surface import BrainSurface

rng = np.random.default_rng(5)
R, C = 700.0, np.array([800.0, 900.0])

def arc(a0, a1, r, z, n=400, wob=1.5):
    a = np.linspace(a0, a1, n)
    rr = r + rng.normal(0, wob, n)
    return np.column_stack([C[0] + rr * np.sin(a), C[1] - rr * np.cos(a), np.full(n, z)])

lines = [arc(-1.1, 1.1, R, 0.0),                                   # full line
         arc(-1.1, 0.25, R - 15, 10.0),                            # shorter, 15 um lower (inside)
         np.vstack([arc(-1.0, 1.0, R, 20.0), arc(-0.6, 0.9, R + 6, 20.0, wob=3)]),   # doubled wobbly stroke
         arc(-1.1, 1.1, R, 30.0)]
# a 30 um spur on the last slice
a = np.linspace(-0.03, 0.03, 30)
spur = np.column_stack([C[0] + (R + 30 * np.sin(np.linspace(0, np.pi, 30))) * np.sin(a),
                        C[1] - (R + 30 * np.sin(np.linspace(0, np.pi, 30))) * np.cos(a), np.full(30, 30.0)])
pts = np.vstack(lines + [spur])
s = BrainSurface(pts, z_range=(0, 40))

def cut_quality(z):
    import trimesh
    segs = trimesh.intersections.mesh_plane(s.mesh, [0, 0, 1], [0, 0, float(z)])
    P = segs.reshape(-1, 3)[:, :2]
    ang = np.arctan2(P[:, 0] - C[0], -(P[:, 1] - C[1]))
    r = np.hypot(*(P - C).T)
    o = np.argsort(ang); ang, r = ang[o], r[o]
    keep = (ang > -1.0) & (ang < 1.0)
    ang, r = ang[keep], r[keep]
    # bin by angle (1 um bins along the arc) and look at the radius profile
    b = np.round(ang * R).astype(int)
    ub, inv = np.unique(b, return_inverse=True)
    rmax = np.bincount(inv, r) / np.bincount(inv)
    spread = np.array([np.ptp(r[inv == i]) for i in range(len(ub))])       # >0 where the cut doubles back
    step = np.abs(np.diff(rmax)).max()                                      # jump between neighbouring um
    mids = segs[:, :, :2].mean(1)
    am = np.arctan2(mids[:, 0] - C[0], -(mids[:, 1] - C[1]))
    inr = (am > -1.0) & (am < 1.0)
    length = np.linalg.norm(segs[inr, 1, :2] - segs[inr, 0, :2], axis=1).sum()
    return spread.max(), step, length / (2.0 * R), rmax

print("slice   doubled-back  biggest step  length/arc   radius range")
for z in (0, 5, 10, 15, 20, 25, 30, 35):
    sp, st, lr, rr = cut_quality(z)
    print(f"z={z:4.1f}   {sp:8.2f} um   {st:8.2f} um   {lr:7.3f}    {rr.min():.1f}..{rr.max():.1f}")
    assert sp < 3.0, "the cut doubles back (comb / fold)"
    assert st < 2.0, "step in the surface line"
    assert lr < 1.05, "cut much longer than the arc (zigzag)"
    assert rr.max() < R + 8, "spur kept"
print("LOFT ARTIFACTS OK")

# sharp bend (like the SC edge) with the line at different heights on different slices: the surface
# line must never cross itself or break into pieces, on drawn slices and in between
def bent(rng, z, shift):
    t = np.linspace(rng.uniform(0, 80), rng.uniform(820, 900), 450)
    x = t.copy(); y = 330 - 0.35 * t
    left = t < 200
    y[left] += (200 - t[left]) ** 2 / 60.0
    y += shift + rng.normal(0, 1.5, len(t))
    return np.column_stack([x, y, np.full(len(t), z)])

def pieces_and_crossings(segs):
    import trimesh
    P = np.round(segs.reshape(-1, 3)[:, :2], 5)
    u, inv = np.unique(P, axis=0, return_inverse=True); inv = inv.ravel()
    deg = np.bincount(inv, minlength=len(u))
    ends = int((deg == 1).sum())
    return ends // 2, int((deg > 2).sum())

for seed in range(12):
    rng = np.random.default_rng(seed)
    pts = np.vstack([bent(rng, z, rng.uniform(-25, 25)) for z in np.arange(0, 40, 5.0)])
    sb = BrainSurface(pts, z_range=(0, 45))
    for z in sorted(set(np.arange(0, 41, 2.5))):
        npieces, branch = pieces_and_crossings(sb.slice_segments(z)[0])
        assert npieces == 1 and branch == 0, (seed, z, npieces, branch)
    # the steep part of the line is part of the surface
    segs = sb.slice_segments(20.0)[0].reshape(-1, 3)
    assert segs[:, 1].max() > 600, "steep part missing"
print("SHARP BEND OK")

# CR border inside a sharp corner: the moved arms must stop where they meet, not run through each other
from arborist.surface import _self_loops
pts = []
for z in (0, 20, 40, 60, 80):
    a = np.column_stack([np.arange(0, 300, 2.0), np.full(150, 300.0)])
    b = np.column_stack([np.full(120, 300.0), 300 - np.arange(2, 242, 2.0)])
    P = np.vstack([a, b]) + rng.normal(0, 1, (270, 2))
    pts.append(np.column_stack([P, np.full(len(P), z)]))
sc = BrainSurface(np.vstack(pts), z_range=(-40, 120)); sc.orient_towards([[150, 150, 40]])
for d in (60.0, 150.0):
    m = sc.offset_mesh(d)
    V = np.asarray(m.vertices); used = np.unique(np.asarray(m.faces))
    dep = sc.signed_depth(V[used][::5])
    # straight stretches at the right depth; the corner itself is rounded (deeper), by design
    assert np.percentile(dep, 1) > d - 3 and abs(np.median(dep) - d) < 0.15 * d, (d, dep.min(), np.median(dep))
    L = sc.cr_line_on_slice(40.0, d)
    assert not _self_loops(L).any(), "CR line crosses itself"
print("CORNER OK")

# CR border under a wiggly hand-drawn line with a hook at its end: one smooth piece, no loops
rng2 = np.random.default_rng(4)
def wline(z, shift):
    tt = np.arange(40, 430, 2.0); x = tt; y = 120 + 0.08 * tt + 6 * np.sin(tt / 23) + 3 * np.sin(tt / 9)
    hk = tt > 400; y[hk] += (tt[hk] - 400) * 1.6
    P = np.column_stack([x, y]) + rng2.normal(0, 1, (len(tt), 2)) + shift
    return np.column_stack([P, np.full(len(tt), z)])
sw = BrainSurface(np.vstack([wline(z, rng2.uniform(-6, 6)) for z in (0, 20, 40, 60)]), z_range=(-20, 80))
sw.orient_towards([[240, 330, 30]])
import trimesh as _tm
for d in (60.0, 150.0, 250.0):
    m = sw.offset_mesh(d)
    assert len(_tm.graph.connected_components(m.edges, min_len=3)) == 1, d
    L = sw.cr_line_on_slice(30.0, d)
    ang = np.degrees(np.abs(np.diff(np.unwrap(np.arctan2(*np.diff(L, axis=0).T[::-1])))))
    assert not _self_loops(L).any() and ang.max() < 30, (d, ang.max())
    dep = sw.signed_depth(np.asarray(m.vertices)[np.unique(m.faces)][::5])
    assert abs(np.median(dep) - d) < 0.05 * d + 3, (d, np.median(dep))
print("WIGGLY HOOK OK")
