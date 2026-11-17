"""Cells analysed one by one (each in its own results folder) are pooled for classification."""
import os, sys, tempfile, time
os.environ["ARBORIST_STORAGE_ROOT"] = tempfile.mkdtemp()
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from arborist import storage, database as dbm
storage.store().set_mode("never")
root = tempfile.mkdtemp()
rng = np.random.default_rng(0)
types = ["wide", "narrow", "horizontal"]
for k in range(7):
    cell = os.path.join(root, f"2026.10.{k + 1:02d}", "Section 1", "Left" if k % 2 else "Right", f"Cell{k}_Stitch.czi")
    res = os.path.join(os.path.dirname(cell), "SC_morph_results")
    os.makedirs(res)
    m = {"wide": (300, .8), "narrow": (90, .3), "horizontal": (250, .1)}[types[k % 3]]
    row = dict(cell=f"Cell{k}", source_path=cell, field_width_um=m[0] + rng.normal(0, 10),
               dend_frac_q2=m[1] + rng.normal(0, .03), dend_n_branch_points=rng.integers(5, 40), manual_type=types[k % 3])
    pd.DataFrame([row]).to_csv(os.path.join(res, f"Cell{k}_summary.csv"), index=False)
    pd.DataFrame([row]).to_csv(os.path.join(res, "all_cells_summary.csv"), index=False)   # per-cell 'master'
# cell 0 analysed again later, under another name: newest kept, no duplicate
time.sleep(0.05)
r0 = os.path.join(root, "2026.10.01", "Section 1", "Right", "SC_morph_results")
d = pd.read_csv(os.path.join(r0, "Cell0_summary.csv")); d["cell"] = "Cell0_v2"; d["field_width_um"] = 333
d.to_csv(os.path.join(r0, "Cell0_redo_summary.csv"), index=False)

t, files = dbm.collect_summaries(root)
print("pooled:", len(t), "cells")
assert len(t) == 7 and "Cell0_v2" in set(t["cell"]) and "Cell0" not in set(t["cell"])

from qtpy.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
import napari
from arborist.gui import SCMorphWidget, dock_widget
v = napari.Viewer(show=False)
w = SCMorphWidget(v); dock_widget(v, w)
w.db.root = root
# like the screenshot: a per-cell table with one cell is selected
w.ed_master.setText(os.path.join(r0, "all_cells_summary.csv"))
w.sp_k.setValue(3)
w.run_clustering()
print(w.lbl_cluster.text().splitlines()[0])
assert w.ed_master.text() == dbm.pooled_path(root) and os.path.exists(dbm.pooled_path(root))
assert "7" in w.lbl_cluster.text() or len(pd.read_csv(dbm.pooled_path(root))) == 7
# one more cell analysed -> the pooled table picks it up on the next load
cell = os.path.join(root, "2026.10.09", "Section 2", "Left", "Cell9.czi"); res = os.path.join(os.path.dirname(cell), "SC_morph_results")
os.makedirs(res); pd.DataFrame([dict(cell="Cell9", source_path=cell, field_width_um=120, dend_frac_q2=.3,
                                     dend_n_branch_points=9, manual_type="narrow")]).to_csv(os.path.join(res, "Cell9_summary.csv"), index=False)
t2 = w.load_master()
assert len(t2) == 8, len(t2)
v.close()
print("POOL OK")
