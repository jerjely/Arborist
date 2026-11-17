"""Soma-seeded draft tracing on a synthetic filled cell with noise, uneven background and a neighbour."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from arborist import autotrace as AT

rng = np.random.default_rng(0)
vox = (0.5, 0.2, 0.2)                                   # z, y, x um
Z, Y, X = 40, 900, 1100                                 # 20 x 180 x 220 um
soma = np.array([110.0, 90.0, 10.0])                    # x, y, z um
# centre lines: 4 primary dendrites, two of them branching, with a little wiggle and a change in depth
def line(p0, direction, length, wig=0.0, dz=0.0, step=0.2):
    t = np.arange(0, length, step)
    d = np.array(direction, float); d /= np.linalg.norm(d)
    nrm = np.array([-d[1], d[0]])
    P = p0[:2] + t[:, None] * d + (wig * np.sin(t / 9.0))[:, None] * nrm
    return np.column_stack([P, p0[2] + dz * t / max(length, 1)])
truth = []      # (points, diameter)
a = line(soma, (1, 0.15), 85, wig=2.0, dz=4);   truth.append((a, 1.6))
truth.append((line(a[len(a) // 2], (0.6, 1), 45, dz=-3), 1.0))
b = line(soma, (-1, 0.3), 80, wig=1.5, dz=-4);  truth.append((b, 1.4))
truth.append((line(b[len(b) * 2 // 3], (-0.3, -1), 35), 0.8))
truth.append((line(soma, (0.1, 1), 60, wig=1.0, dz=3), 1.2))
truth.append((line(soma, (0.2, -1), 55, dz=-2), 0.9))
neighbour = line(np.array([40.0, 150.0, 10.0]), (1, 0.1), 60)              # another cell, not connected
vol = np.zeros((Z, Y, X), np.float32)
zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]
def paint(P, dia, val=1.0):
    for p in P[:: 3]:
        r = dia / 2
        z0, z1 = int((p[2] - r) / vox[0]), int((p[2] + r) / vox[0]) + 2
        y0, y1 = int((p[1] - r) / vox[1]), int((p[1] + r) / vox[1]) + 2
        x0, x1 = int((p[0] - r) / vox[2]), int((p[0] + r) / vox[2]) + 2
        s = (slice(max(z0, 0), z1), slice(max(y0, 0), y1), slice(max(x0, 0), x1))
        d2 = ((zz[s] * vox[0] - p[2]) ** 2 + (yy[s] * vox[1] - p[1]) ** 2 + (xx[s] * vox[2] - p[0]) ** 2)
        vol[s][d2 <= r * r] = val
for P, dia in truth:
    paint(P, dia)
paint(soma[None], 12.0)
paint(neighbour, 1.5); paint(np.array([[40.0, 150.0, 10.0]]), 10.0)
true_len = sum(np.linalg.norm(np.diff(P, axis=0), axis=1).sum() for P, _ in truth)
img = gaussian_filter(vol, (0.6, 1.0, 1.0)) * 600
img += 120 + 60 * (xx / X) + rng.normal(0, 12, vol.shape)                  # uneven background + noise
data = np.stack([rng.normal(120, 12, vol.shape).astype(np.float32), img.astype(np.float32)])
del zz, yy, xx

assert AT.pick_channel_at(data, vox, soma) == 1
cache = {}
t = time.perf_counter()
n, info = AT.autotrace(data, vox, soma + [1.5, -1.0, 0.5], reach_um=150, cache=cache)     # a slightly off click
dt = time.perf_counter() - t
c, p, _ = n.segments((3,))
L = np.linalg.norm(c - p, axis=1).sum()
allpts = np.vstack([P for P, _ in truth])
d_nodes = cKDTree(allpts).query(n.xyz[1:])[0]                              # traced nodes -> true centre lines
far_soma = np.linalg.norm(allpts - soma, axis=1) > 8
d_truth = cKDTree(n.xyz).query(allpts[far_soma])[0]                        # true centre lines -> tracing
tips = int((~n.df["id"].isin(n.df["parent"])).sum())
d_nb = cKDTree(n.xyz).query(neighbour)[0].min()
print(f"{dt:.1f}s for {data[1].size / 1e6:.0f} M voxels | branches {info['branches']} tips {tips} | "
      f"length {L:.0f} um (true {true_len:.0f}, minus the parts inside the soma)")
print(f"traced nodes off the true centre line: median {np.median(d_nodes):.2f} um, 95% {np.percentile(d_nodes, 95):.2f} um")
print(f"true processes found: {np.mean(d_truth < 2.0):.0%} | nearest traced node to the unconnected neighbour: {d_nb:.0f} um")
assert tips == 6, tips
assert 0.85 * (true_len - 6 * 4) < L < 1.08 * true_len, (L, true_len)
assert np.median(d_nodes) < 0.5 and np.percentile(d_nodes, 95) < 1.5
assert np.mean(d_truth < 2.0) > 0.97
assert d_nb > 10, "the unconnected neighbour was traced"
assert (n.df["parent"] == -1).sum() == 1 and n.df["type"].iloc[0] == 1
r = n.df["radius"].to_numpy()[1:]
assert 0.2 < np.median(r) < 1.2, np.median(r)
# another sensitivity reuses the prepared image; a very high threshold finds less
t = time.perf_counter(); n2, info2 = AT.autotrace(data, vox, soma, reach_um=150, sensitivity=40, cache={}); 
assert info2["length_um"] <= info["length_um"] + 1
try:
    AT.autotrace(data, vox, np.array([200.0, 20.0, 10.0]), reach_um=60)
    raise SystemExit("a click on empty background should be refused")
except ValueError as ex:
    print("click on background:", str(ex)[:60])
print("AUTOTRACE OK")

# ---------------------------------------------------------------- through the GUI
import tempfile, tifffile
T = tempfile.mkdtemp(); os.environ["ARBORIST_STORAGE_ROOT"] = os.path.join(T, "local")
d = os.path.join(T, "data", "2026.11.01_test", "Section 1", "Left"); os.makedirs(d)
img_path = os.path.join(d, "cellA_63x_Stitch.tif")
tifffile.imwrite(img_path, np.clip(data, 0, 65535).astype(np.uint16).transpose(1, 0, 2, 3), imagej=True,
                 resolution=(1 / vox[2], 1 / vox[1]), metadata={"axes": "ZCYX", "spacing": vox[0], "unit": "um"})
import napari
from qtpy.QtWidgets import QApplication, QMessageBox
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
from arborist import storage, database as dbm
storage.store().set_mode("never")
from arborist.gui import SCMorphWidget, dock_widget
v = napari.Viewer(show=False); w = SCMorphWidget(v); dock_widget(v, w)
e = w.add_stack(img_path, "cell")
assert np.allclose(e.voxel, vox, atol=1e-3), e.voxel
w.sp_at_reach.setValue(150)
w._at_seed = soma.copy()
w._autotrace_run(); QApplication.processEvents()
print("GUI:", w.lbl_at.text()[:90])
draft = os.path.join(d, "cellA_63x_Stitch_autotrace.swc")
assert os.path.exists(draft) and e.neuron.source == "autotrace" and "6 branches" in w.lbl_at.text()
assert "AUTOMATIC DRAFT" in open(draft).read(300)
w._autosave()
proj = json.load(open(dbm.project_path_for(img_path))) if (json := __import__("json")) else None
assert "neuron" not in proj["stacks"][0], "the unchecked draft was stored in the project"
w.db.ed_root.setText(os.path.join(T, "data")); w.db.rescan()
row = next(r for r in w.db.rows if r.get("cell"))
assert not row.get("swc"), "a draft must not count as a finished tracing"
w._closing_ok = True; v.close()
print("AUTOTRACE GUI OK")
