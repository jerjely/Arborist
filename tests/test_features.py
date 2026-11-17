"""Diameters from the image, batch re-analysis, the in vivo table link and the Excel copies."""
import os, sys, tempfile, json, shutil, subprocess, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
T = tempfile.mkdtemp()
os.environ["ARBORIST_STORAGE_ROOT"] = os.path.join(T, "local")
import numpy as np, pandas as pd, tifffile
from scipy.ndimage import gaussian_filter
from arborist.swc import Neuron
from arborist import storage, morphometry as M, ephys
storage.store().set_mode("never")
rng = np.random.default_rng(0)

# ------------------------------------------------ 1. diameters from the image: tubes of known width
vox = (0.3, 0.1, 0.1)                                   # z, y, x in um
Z, Y, X = 30, 500, 700
vol = np.zeros((Z, Y, X), np.float32)
yy, xx = np.mgrid[0:Y, 0:X]
truth, rows = {}, [[1, 1, 35.0, 25.0, 4.5, 4.0, -1]]
for b, (dia, ang) in enumerate([(0.8, 0.0), (1.5, 0.5), (3.0, -0.45)]):
    prev = 1
    p0 = np.array([35.0, 25.0])
    for j in range(1, 60):
        p = p0 + j * 0.5 * np.array([np.cos(ang), np.sin(ang)])
        rows.append([len(rows) + 1, 3, p[0], p[1], 4.5, 0.1, prev]); prev = len(rows); truth[prev] = dia
    # paint the tube: all voxels within dia/2 of the centre line (in 3D)
    d_line = np.abs(-(xx * vox[2] - p0[0]) * np.sin(ang) + (yy * vox[1] - p0[1]) * np.cos(ang))
    along = (xx * vox[2] - p0[0]) * np.cos(ang) + (yy * vox[1] - p0[1]) * np.sin(ang)
    for z in range(Z):
        dz = (z * vox[0] - 4.5)
        vol[z][(np.hypot(d_line, dz) <= dia / 2) & (along >= 0) & (along <= 30)] = 1000
vol = gaussian_filter(vol, (0.5, 1.2, 1.2)) + 100 + rng.normal(0, 8, vol.shape)     # optical blur + noise
other = rng.normal(100, 8, vol.shape).astype(np.float32)                              # a channel without the cell
n = Neuron(pd.DataFrame(rows, columns=Neuron.COLS))
data = np.stack([other, vol])
assert M.pick_fill_channel(n, data, vox) == 1
tab = M.estimate_radii(n, lambda z: data[1, z], vox, Z)
got = pd.Series(2 * tab["radius_um"].to_numpy(), index=tab["id"])
print("true -> measured diameter (um):")
for dia in (0.8, 1.5, 3.0):
    ids = [i for i, d in truth.items() if d == dia][22:-5]      # away from the soma, where the tubes run apart
    m = got.loc[ids]
    print(f"   {dia:.1f} -> {m.median():.2f}  (measured at {m.notna().mean():.0%} of points)")
    assert m.notna().mean() > 0.9 and abs(m.median() - dia) < max(0.25, 0.12 * dia), (dia, m.median())
assert M.apply_radii(n, tab) > 150 and n.radii_source == "image"
f = M.thickness_features(n, (3, 4), "dend")
assert f["dend_radii_fitted"] and 1.0 < f["dend_mean_diameter_um"] < 2.6, f
assert M.apply_radii(Neuron(pd.DataFrame(rows[:5], columns=Neuron.COLS)), tab.assign(id=tab["id"] + 9999)) == 0
print("diameters from the image OK")

# ------------------------------------------------ 2. GUI: batch re-analysis gives the same numbers as Analyze
HERE = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(HERE, "synthetic", "synthetic_cell.tif")):
    subprocess.run([sys.executable, os.path.join(HERE, "make_synthetic.py")], cwd=HERE, capture_output=True)
G = os.path.join(T, "data")
import napari
from qtpy.QtWidgets import QApplication, QMessageBox
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
from arborist.gui import SCMorphWidget, dock_widget
from arborist import database as dbm, batch
from make_synthetic import f_surface
cells = []
for k, (day, sec, side) in enumerate([("2026.10.29_Bl6", "section 1", "left"), ("2026.10.29_Bl6", "section 2", "right"),
                                      ("2026.06.24_PV", "section 1", "left"), ("2026.06.24_PV", "section 3", "right"),
                                      ("2026.07.15_Thy1", "section 2", "left")]):
    d = os.path.join(G, day, sec, side); os.makedirs(d)
    stem = f"cell{k}_63x_Stitch"
    shutil.copy(os.path.join(HERE, "synthetic", "synthetic_cell.tif"), os.path.join(d, stem + ".tif"))
    cells.append((d, stem))
v = napari.Viewer(show=False); w = SCMorphWidget(v); dock_widget(v, w)
w.db.ed_root.setText(G); w.db.rescan()
st0 = w.add_stack(os.path.join(cells[0][0], cells[0][1] + ".tif"), "cell")
vz, vy, vx = st0.voxel; Zs, Ys, Xs = st0.stack.data.shape[1:]
w._closing_ok = True
def make_swc(path, seed, spread):
    r = np.random.default_rng(seed)
    cx, cy, cz = Xs * vx / 2, Ys * vy / 2, Zs * vz / 2
    rows = [[1, 1, cx, cy, cz, 4, -1]]
    for b in range(5):
        prev, p = 1, np.array([cx, cy, cz]); dvec = r.normal(0, 1, 3) * [spread, 1, .2]; dvec /= np.linalg.norm(dvec)
        for j in range(25):
            dvec = dvec + r.normal(0, .2, 3) * [1, 1, .2]; dvec /= np.linalg.norm(dvec); p = p + 1.5 * dvec
            p = np.clip(p, [2, 2, 1], [Xs * vx - 2, Ys * vy - 2, Zs * vz - 1])
            rows.append([len(rows) + 1, 3 if b < 4 else 2, *p, .5, prev]); prev = len(rows)
    Neuron(pd.DataFrame(rows, columns=Neuron.COLS)).write(path)
for k, (d, stem) in enumerate(cells):
    make_swc(os.path.join(d, stem + ".swc"), k, 1 + 2 * (k % 2))
w.db.rescan()
interactive = {}
for k, (d, stem) in enumerate(cells):
    row = next(r for r in w.db.rows if r.get("cell_name") == stem)
    w._open_from_db(row); QApplication.processEvents()
    e = w.cell_entry()
    xs = np.linspace(2, Xs * vx - 2, 60)
    pts = np.vstack([np.column_stack([np.full(60, z / vz), np.full(60, 6.0 + k) / vy, xs / vx]) for z in (2 * vz, (Zs - 3) * vz)])
    e.layers["surface_pts"].data = pts
    w._marks_timer.stop(); w._apply_mark_changes(); w._build_and_save()
    w.rb_manual.setChecked(True); w.sp_manual.setValue(40.0 + 5 * k)
    w.run_analysis(); QApplication.processEvents()
    interactive[stem] = pd.read_csv(os.path.join(d, "SC_morph_results", stem + "_summary.csv")).iloc[0]
w._autosave(); w.db.rescan()
assert all(r["analyzed"] for r in w.db.rows if r.get("cell"))
# damage the results, then re-analyse everything from the saved projects
for d, stem in cells:
    pd.DataFrame([{"cell": stem, "source_path": os.path.join(d, stem + ".tif"), "dendrite_length_um": -1}]).to_csv(
        os.path.join(d, "SC_morph_results", stem + "_summary.csv"), index=False)
res = w.reanalyse_all()
assert res and all(r[1] == "ok" for r in res), res
worst = 0
for d, stem in cells:
    new = pd.read_csv(os.path.join(d, "SC_morph_results", stem + "_summary.csv")).iloc[0]
    old = interactive[stem]
    for c in old.index:
        if isinstance(old[c], (int, float, np.floating)) and pd.notna(old[c]) and c in new.index:
            diff = abs(float(new[c]) - float(old[c])) / max(1.0, abs(float(old[c])))
            worst = max(worst, diff)
            assert diff < 1e-6, (stem, c, old[c], new[c])
    assert abs(new["CR_border_depth_um"] - old["CR_border_depth_um"]) < 1e-9
print(f"batch re-analysis = Analyze (largest relative difference {worst:.1e}) OK")

# ------------------------------------------------ 3. pooled table: Excel copy + id columns; in vivo link
pooled = dbm.pooled_path(G)
assert os.path.exists(pooled) and os.path.exists(os.path.splitext(pooled)[0] + ".xlsx")
tp = pd.read_excel(os.path.splitext(pooled)[0] + ".xlsx")
assert list(tp.columns[:4]) == ["cell", "day", "section", "side"] and set(tp["side"]) == {"left", "right"}, tp.columns[:6]
iv = pd.DataFrame({"date": ["2026-10-29", "2026-10-29", "2026-06-24", "2026-06-24", "2026-07-15"],
                   "section": ["I.", "II.", "I.", "III.", "II."], "side": ["left", "right", "left", "right", "left"],
                   "functional cluster": [1, 2, 1, 2, 2], "DSI": [0.1, 0.6, 0.15, 0.7, 0.55], "RF size deg": [8, 20, 9, 22, 19]})
ivp = os.path.join(T, "invivo.xlsx"); iv.to_excel(ivp, index=False)
t_iv = ephys.read_table(ivp)
ids = ephys.guess_id_columns(t_iv)
assert "functional cluster" not in ids and "functional cluster" in ephys.label_columns(t_iv, ids)
assert set(ephys.feature_columns(t_iv, ids)) == {"DSI", "RF size deg"}
ephys.save_link(pooled, dict(path=ivp, sheet=None, id_cols=ids, manual={}), "invivo")
t = w.load_master()
assert "invivo_DSI" in t.columns and t["invivo_DSI"].notna().sum() == 5, t.filter(like="invivo").head()
assert "5 of 5" in w.lbl_invivo.text(), w.lbl_invivo.text()
i = w.cmb_label.findData("invivo_label_functional_cluster"); assert i >= 0
w.cmb_label.setCurrentIndex(i)
w._select_features("invivo"); w.cb_auto_k.setChecked(False); w.sp_k.setValue(2)
w.run_clustering()
txt = w.lbl_cluster.text()
print(txt.replace("<br>", "\n   ")[:400])
assert "adjusted Rand 1.00" in txt and "Stability" in txt
tabs = [w.results.tabs.tabText(i) for i in range(w.results.tabs.count())]
assert "Stability" in tabs and "vs labels" in tabs, tabs
assert os.path.exists(os.path.splitext(pooled)[0] + "_clusters.xlsx")
print("in vivo link, stability, agreement, Excel copies OK")

# ------------------------------------------------ 4. diameters through the GUI, kept for the next opening
row = next(r for r in w.db.rows if r.get("cell_name") == cells[0][1])
w._open_from_db(row); QApplication.processEvents()
w.measure_radii(); QApplication.processEvents()
print("GUI:", w.lbl_radii.text()[:110])
v.close()
print("FEATURES OK")
