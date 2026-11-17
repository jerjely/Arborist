import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from arborist.io import load_stack
from arborist.surface import BrainSurface, measure_border_distance
from arborist.swc import Neuron
from arborist import analysis
from make_synthetic import f_surface, main as make

D = os.path.join(os.path.dirname(__file__), "synthetic")
make(D)
st = load_stack(os.path.join(D, "synthetic_cell.tif"))
print("stack", st.data.shape, st.voxel, st.channel_names, st.calibrated)
assert st.data.shape == (2, 60, 300, 300) and np.allclose(st.voxel, (1, .5, .5))

# --- surface from clicked points (every 5th slice, 15 points per slice, ~1 um click noise)
rng = np.random.default_rng(0)
pts = []
for z in range(0, 60, 5):
    for x in np.linspace(5, 145, 15):
        pts.append([x, f_surface(x, z) + rng.normal(0, 0.7), z])
pts = np.array(pts)
for method in ("smooth", "delaunay"):
    s = BrainSurface(pts, method=method)
    s.orient_by_image(st.data[1], st.voxel)
    # test points 50 um below the surface along the true local normal
    q = []
    for x, z in [(60, 20), (90, 35), (75, 30)]:
        g = np.array([-0.15, 1.0, -0.008 * (z - 30)])
        g /= np.linalg.norm(g)
        q.append(np.array([x, f_surface(x, z), z]) + 50 * g)
    d = s.signed_depth(np.array(q))
    print(method, "depth of points 50 um deep:", np.round(d, 2), "faces", len(s.mesh.faces))
    assert np.all(np.abs(d - 50) < 2.0), d

# --- neuron as it would come from SNT (soma + 3 dendrites + axon)
xyz = np.array([[75, 95, 30], [60, 85, 26], [40, 70, 20]], float)
n = Neuron()
sid = n.add_soma([75, 95, 30])
n.add_path(xyz, 3, parent_id=sid)
n.add_path(np.array([[75, 95, 30], [110, 75, 40]], float), 3, parent_id=sid)
n.add_path(np.array([[75, 95, 30], [80, 130, 32]], float), 2, parent_id=sid)
n.write(os.path.join(D, "test.swc"))
n2 = Neuron.read(os.path.join(D, "test.swc"))
assert len(n2) == len(n)

# --- CR border on low-mag stack
lm = load_stack(os.path.join(D, "synthetic_lowmag.tif"))
ang = np.deg2rad(20); nv = np.array([-np.sin(ang), np.cos(ang)]); tv = np.array([np.cos(ang), np.sin(ang)])
base = np.array([250, 150.])
surf_pts = [np.r_[base + t * tv, 8.0] for t in np.linspace(-150, 150, 12)]
bord_pts = [np.r_[base + t * tv + 120 * nv + rng.normal(0, 2, 2), 8.0] for t in np.linspace(-120, 120, 8)]
r = measure_border_distance(np.array(surf_pts), np.array(bord_pts))
print("CR border (2D):", r["mode"], round(r["mean"], 2), "+/-", round(r["sd"], 2))
assert abs(r["mean"] - 120) < 3
surf3 = [np.r_[base + t * tv, z] for t in np.linspace(-150, 150, 12) for z in (0, 8, 16, 20)]
r3 = measure_border_distance(np.array(surf3), np.array(bord_pts))
print("CR border (3D):", r3["mode"], round(r3["mean"], 2))
assert abs(r3["mean"] - 120) < 3

# --- analysis
s = BrainSurface(pts, method="smooth", cover_xyz=n.xyz, extend_um=10)
s.orient_towards(n.xyz[:1])
res = analysis.run(n, s, os.path.join(D, "out"), "synthetic_cell", boundary_um=60,
                   master_csv=os.path.join(D, "out", "all_cells_summary.csv"))
print(res["summary"].T)
print("outputs:", sorted(os.listdir(os.path.join(D, "out"))))
print("ALL OK")
