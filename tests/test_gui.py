import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import napari
from qtpy.QtWidgets import QMessageBox
from arborist import storage
from arborist.gui import SCMorphWidget, xyz2vox, dock_widget
from make_synthetic import f_surface

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
D = os.path.join(os.path.dirname(__file__), "synthetic")
SHOT = os.path.join(os.path.dirname(__file__), "shots")
os.makedirs(SHOT, exist_ok=True)

v = napari.Viewer(title="Arborist test", show=True)
v.window._qt_window.resize(1600, 1000)
w = SCMorphWidget(v)
dock_widget(v, w)

e = w.add_stack(os.path.join(D, "synthetic_cell.tif"), "cell")
pts = np.array([[x, f_surface(x, z), z] for z in range(0, 60, 6) for x in np.linspace(5, 145, 12)])
e.layers["surface_pts"].data = xyz2vox(pts, e.voxel)
w.build_surface()
print(w.lbl_surface.text())

lm = w.add_stack(os.path.join(D, "synthetic_lowmag.tif"), "lowmag")
ang = np.deg2rad(20); nv = np.array([-np.sin(ang), np.cos(ang)]); tv = np.array([np.cos(ang), np.sin(ang)])
base = np.array([250, 150.])
S = np.array([np.r_[base + t * tv, 8.0] for t in np.linspace(-150, 150, 12)])
B = np.array([np.r_[base + t * tv + 120 * nv, 8.0] for t in np.linspace(-120, 120, 8)])
lm.layers["surface_pts"].data = xyz2vox(S, lm.voxel)
lm.layers["border_pts"].data = xyz2vox(B, lm.voxel)
v.dims.set_point(0, 8.0)
w.measure_border()
print(w.status.text(), "| boundary:", w.boundary_um())
w.tabs.setCurrentIndex(2)
v.screenshot(os.path.join(SHOT, "lowmag.png"), canvas_only=False)

# load the SWC written by test_core (as if exported from SNT)
w.set_active(w.cell_entry())
w._load_swc(w.cell_entry(), os.path.join(D, "test.swc"))
print(w.lbl_trace.text())
v.dims.set_point(0, 25.0)
e.images[1].visible = False
w.tabs.setCurrentIndex(3)
v.screenshot(os.path.join(SHOT, "neuron.png"), canvas_only=False)

# AIS on the axon (soma (75,95,30) -> (80,130,32)), marked by two clicks
e.layers["ais_pts"].data = xyz2vox(np.array([[76, 100, 30.3], [78.5, 118, 31.3]]), e.voxel)
w._ais_preview(); print("AIS:", w.lbl_ais.text())
w.tabs.setCurrentIndex(4)
v.screenshot(os.path.join(SHOT, "neuron_ais.png"), canvas_only=False)
w.cmb_manual.setCurrentText("stellate")
w.ed_out.setText(os.path.join(D, "gui_out"))
# a 'Flip' made earlier (against an image-based guess) must not turn the soma-based side around
ce = w.cell_entry()
ce.flipped = True
w._build_and_save() if hasattr(w, "_build_and_save") else w.build_surface(ce)
soma = ce.neuron.df[ce.neuron.df["type"] == 1][["x", "y", "z"]].to_numpy(float)
assert ce.surface.signed_depth(soma).min() > 0 and not ce.flipped, "soma ended up outside the brain"
print("flip vs soma OK")
w.run_analysis()
print(w.status.text())
w.tabs.setCurrentIndex(5)
v.screenshot(os.path.join(SHOT, "analysis.png"), canvas_only=False)
w.results.tabs.setCurrentIndex(0)
w.results.resize(700, 900)
w.results.grab().save(os.path.join(SHOT, "results_3d.png"))
w.results.tabs.setCurrentIndex(1); w.results.grab().save(os.path.join(SHOT, "results_laminar.png"))
w.results.tabs.setCurrentIndex(w.results.tabs.count() - 1); w.results.grab().save(os.path.join(SHOT, "results_summary.png"))
print("result tabs:", [w.results.tabs.tabText(i) for i in range(w.results.tabs.count())])
from qtpy.QtWidgets import QApplication
r = w.results
assert r.opt_box.isVisibleTo(r) and r.ctx is not None
names = [r.tabs.tabText(i) for i in range(r.tabs.count())]
for need in ("Sholl profile", "Sholl 3D", "Sholl × depth", "Laminar %"):
    assert need in names, (need, names)
r.sp_lam_bin.setValue(10); r.cmb_lam_mode.setCurrentIndex(1); r.cb_lam_smooth.setChecked(True)
r.sp_sholl_step.setValue(20); r.sp_shells.setValue(3)
r.redraw(); QApplication.processEvents()
for nm in ("Laminar %", "Sholl profile", "Sholl 3D", "Sholl × depth"):
    i = [r.tabs.tabText(k) for k in range(r.tabs.count())].index(nm)
    r.tabs.setCurrentIndex(i); QApplication.processEvents()
    r.grab().save(os.path.join(SHOT, "opt_" + nm.split()[-1].replace("%", "pct").replace("×","x") + "_" + str(i) + ".png"))
import unittest.mock as um
with um.patch("arborist.gui.QMessageBox.information"):
    r.save_figs()
saved = [f for f in os.listdir(r.ctx["out"]) if "bin10_shell20" in f]
assert len(saved) >= 8, saved
print("plot options OK, saved", len(saved))

# classify tab on the 15 synthetic cells from test_classify
w.ed_master.setText(os.path.join(D, "classify", "all_cells_summary.csv"))
w.load_master(); w.sp_k.setValue(5); w.run_clustering()
print(w.lbl_cluster.text())
w.tabs.setCurrentIndex(6)
v.screenshot(os.path.join(SHOT, "classify.png"), canvas_only=False)
v.dims.ndisplay = 3
v.camera.angles = (10, -30, 60)
v.screenshot(os.path.join(SHOT, "3d.png"), canvas_only=True)
v.dims.ndisplay = 2

w._save_project(os.path.join(D, "proj.json"))
# an older project saved with the 'smooth' (height field) method is switched to 'Follow slices'
pj = json.load(open(storage.read(os.path.join(D, "proj.json"))))
for sd in pj["stacks"]:
    if sd.get("surface"):
        sd["surface"]["method"] = "smooth"
json.dump(pj, open(storage.out(os.path.join(D, "proj.json")), "w"))
w._load_project(os.path.join(D, "proj.json"))
c_ = w.cell_entry()
assert c_.surface is not None and c_.surface.method == "slices" and w.cmb_method.currentIndex() == 0
pj = json.load(open(storage.read(os.path.join(D, "proj.json"))))
assert all(sd["surface"]["method"] == "slices" for sd in pj["stacks"] if sd.get("surface"))
print("old project switched to Follow slices and saved")
print("after reload:", len(w.cell_entry().neuron), "nodes; boundary", w.boundary_um(), "|", w.status.text())
v.close()
print("GUI OK")
