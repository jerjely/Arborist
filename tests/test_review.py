"""Regression tests for the problems found in the November code review."""
import os, sys, tempfile, time, shutil
ROOT = os.environ.get("ARBORIST_UNDER_TEST") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
T = tempfile.mkdtemp()
os.environ["ARBORIST_STORAGE_ROOT"] = os.path.join(T, "local")
import numpy as np
from arborist.swc import Neuron
from arborist import storage

# ---------------------------------------------------------------- SWC reader
def swc(txt):
    p = os.path.join(T, "t.swc"); open(p, "w", newline="").write(txt); return Neuron.read(p)
n = swc("# c\n1 1 0 0 0 5 -1 0\n2 3 1 0 0 1 1 0\n3 3 2 0 0 1 2 0\n")            # an extra column
assert n.df["id"].tolist() == [1, 2, 3] and n.df["type"].tolist() == [1, 3, 3], n.df
assert len(n.segments()[0]) == 2
n = swc("1 1 0 0 0 5 -1\n2 3 1 0 0 1 1\n2 3 2 0 0 1 1\n3 3 3 0 0 1 2\n")          # a duplicate id
assert n.df["id"].tolist() == [1, 2, 3] and any("more than once" in w for w in n.warnings)
n = swc("1,1,0,0,0,5,-1\r\n2,3,1,0,0,NaN,1\r\nnot a line\r\n")                  # commas, NaN radius, junk
assert len(n) == 2 and n.df["radius"].tolist() == [5.0, 0.0] and n.warnings
assert len(swc("# nothing\n")) == 0
print("swc reader OK")

# ---------------------------------------------------------------- storage
st = storage.store(); st.set_mode("always")
usb = os.path.join(T, "E", "SC_cells", "2026.10.29_Bl6", "section 3", "left"); os.makedirs(usb)
real = os.path.join(usb, "cell_SC_morph.json")

# 1. a stale local copy must never overwrite a newer file on the drive
open(storage.out(real), "w").write("old, longer text")           # local only (never copied: 'drive unplugged')
os.utime(st.local_for(real), (time.time() - 3600,) * 2)
open(real, "w").write("NEW")                                     # the drive copy was changed elsewhere, later
assert storage.read(real) == real, "the newer file on the drive must be the one that is read"
st.sync()
assert open(real).read() == "NEW", "a stale local copy overwrote the newer file"
print("stale copy does not overwrite OK")

# 2. timestamp rounding on FAT / exFAT sticks must not leave files 'waiting' for ever
f2 = os.path.join(usb, "surface.csv")
loc = storage.out(f2); open(loc, "w").write("1,2,3")
orig_copy2 = shutil.copy2
def fat_copy2(a, b, **k):                                        # the stick rounds times down to 2 s
    r = orig_copy2(a, b, **k); m = os.path.getmtime(b); os.utime(b, (m, m - (m % 2) - 0.0)); return r
shutil.copy2 = fat_copy2
os.utime(loc, (time.time(), 2 * (time.time() // 2) + 1.7))       # an odd time: rounding changes it
try:
    assert st.sync()[0] == 1
finally:
    shutil.copy2 = orig_copy2
assert st.summary()[0] == 0 and st.pending_count()[0] == 0, "file still waiting after it was copied"
storage._store = None; st = storage.store()                      # ... also after a restart
assert st.summary()[0] == 0
print("FAT rounding OK")

# 3. files queued for an old drive letter / renamed folder find their folder again
old = os.path.join(T, "D", "SC cells", "2026.10.29_Bl6", "section 3", "left")      # does not exist any more
for name in ("cellX_SC_morph.json", os.path.join("SC_morph_results", "cellX_summary.csv")):
    p = storage.out(os.path.join(old, name)); open(p, "w").write("queued " + name)
shutil.rmtree(os.path.join(T, "D"), ignore_errors=True)          # out() made the folders; the 'D:' stick is gone
st.alt_roots = lambda: [os.path.join(T, "E"), os.path.join(T, "F")]
copied, failed = st.sync()
assert not failed, failed
assert open(os.path.join(usb, "cellX_SC_morph.json")).read().startswith("queued")
assert os.path.exists(os.path.join(usb, "SC_morph_results", "cellX_summary.csv"))
assert not os.path.exists(os.path.join(T, "D")), "a folder tree was built at the old place"
# ... and when the folder really is nowhere, nothing is created and the files stay safe locally
p = storage.out(os.path.join(T, "D", "gone", "day", "sec", "x.json")); open(p, "w").write("x")
shutil.rmtree(os.path.join(T, "D"), ignore_errors=True)
copied, failed = st.sync()
assert len(failed) == 1 and os.path.exists(p) and not os.path.exists(os.path.join(T, "D"))
print("moved drive OK")

# ---------------------------------------------------------------- GUI: project of one cell is never
# overwritten by another cell; a damaged project does not lock the cell; autosave is light
import json, subprocess
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
HERE = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(HERE, "synthetic", "synthetic_cell.tif")):
    subprocess.run([sys.executable, os.path.join(HERE, "make_synthetic.py")], cwd=HERE, capture_output=True)
storage._store = None; storage.store().set_mode("never")
G = os.path.join(T, "gui"); A = os.path.join(G, "day", "Section 1", "Left"); B = os.path.join(G, "day", "Section 2", "Left")
for d, nm in ((A, "cellA_63x_Stitch.tif"), (B, "cellB_63x_Stitch.tif")):
    os.makedirs(d); shutil.copy(os.path.join(HERE, "synthetic", "synthetic_cell.tif"), os.path.join(d, nm))
import napari
from qtpy.QtWidgets import QApplication, QMessageBox
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
from arborist.gui import SCMorphWidget, dock_widget
from arborist import database as dbm
v = napari.Viewer(show=False); w = SCMorphWidget(v); dock_widget(v, w)
w.db.ed_root.setText(G); w.db.rescan()
rowA = next(r for r in w.db.rows if r.get("cell") and "cellA" in r["cell"])
rowB = next(r for r in w.db.rows if r.get("cell") and "cellB" in r["cell"])
w._open_from_db(rowA)
e = w.cell_entry()
e.layers["surface_pts"].data = np.column_stack([np.full(30, 3.0), np.full(30, 20.0), np.linspace(5, 100, 30)])
w._marks_timer.stop(); w._apply_mark_changes()
projA = dbm.project_path_for(rowA["cell"])
ptsA = json.load(open(projA))["stacks"][0]["surface_points_um"]
assert len(ptsA) == 30
# open another cell BY HAND (tab 1) while A's project is the current one
w.add_stack(rowB["cell"], "cell")
w._autosave()
dA = json.load(open(projA))
assert os.path.basename(dA["stacks"][0]["path"]).startswith("cellA") and len(dA["stacks"][0]["surface_points_um"]) == 30, \
    "cell A's project was overwritten by cell B"
assert os.path.exists(dbm.project_path_for(rowB["cell"]))
print("projects kept apart OK")
# no leftover temporary file, and the project is valid JSON
assert not [f for f in os.listdir(A) if f.endswith(".tmp")]
# a damaged project: the cell still opens, the damaged file is kept aside
open(projA, "w").write('{"version": "x", "stacks": [{"path": ')
w.db.rescan(); rowA = next(r for r in w.db.rows if r.get("cell") and "cellA" in r["cell"])
w._open_from_db(rowA)
assert w.cell_entry() is not None and "cellA" in w.cell_entry().stack.path
assert os.path.exists(projA + ".unreadable")
print("damaged project OK")
# autosave after an unrelated change does not rewrite the surface files or rescan the database
e = w.cell_entry()
e.layers["surface_pts"].data = np.column_stack([np.full(30, 3.0), np.full(30, 20.0), np.linspace(5, 100, 30)])
w._marks_timer.stop(); w._apply_mark_changes(); w._build_and_save()
stl = os.path.join(A, e.name + "_brain_surface.stl"); m0 = os.stat(stl).st_mtime_ns
scans = []; orig_scan = dbm.scan; dbm.scan = lambda *a, **k: (scans.append(1), orig_scan(*a, **k))[1]
time.sleep(0.05); w._autosave(); QApplication.processEvents()
assert os.stat(stl).st_mtime_ns == m0, "surface file rewritten although nothing changed"
assert not scans, "autosave rescanned the whole database"
dbm.scan = orig_scan
w._closing_ok = True; v.close()
print("light autosave OK")

# ---------------------------------------------------------------- rename SC_morph -> Arborist
import importlib
B = os.path.join(T, "appdata")
old = os.path.join(B, "SC_morph"); new = os.path.join(B, "Arborist")
os.makedirs(os.path.join(old, "work", "E", "data", "cell1"))
open(os.path.join(old, "work", "E", "data", "cell1", "p.json"), "w").write("waiting")
json.dump({"mode": "always"}, open(os.path.join(old, "settings.json"), "w"))
json.dump({os.path.join(old, "work", "E", "data", "cell1"): os.path.join(T, "E", "data", "cell1")},
          open(os.path.join(old, "pending.json"), "w"))
storage._migrate_from_sc_morph(new)
assert not os.path.exists(old) and os.path.exists(os.path.join(new, "work", "E", "data", "cell1", "p.json"))
st2 = storage.Store(new)
assert st2.mode == "always" and list(st2.pending) == [os.path.join(new, "work", "E", "data", "cell1")], st2.pending
assert st2.pending_count()[0] == 1
storage._migrate_from_sc_morph(new)                 # running again does nothing
print("rename migration OK")
print("REVIEW OK")
