"""Work locally, copy to the (slow) drive later: writes, reads, database status, failures, retry."""
import os, sys, tempfile, time, stat, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
T = tempfile.mkdtemp()
os.environ["ARBORIST_STORAGE_ROOT"] = os.path.join(T, "local")
from arborist import storage
st = storage.store()
usb = os.path.join(T, "usb", "2026_11_09", "Section 1", "Left")
os.makedirs(usb)
open(os.path.join(usb, "existing_summary.csv"), "w").write("cell\nA\n")

st.set_mode("never")
assert storage.out(os.path.join(usb, "x.txt")) == os.path.join(usb, "x.txt")
st.set_mode("always")
p = storage.out(os.path.join(usb, "proj_SC_morph.json"))
assert p.startswith(st.work) and not os.path.exists(os.path.join(usb, "proj_SC_morph.json"))
json.dump({"a": 1}, open(p, "w"))
assert storage.read(os.path.join(usb, "proj_SC_morph.json")) == p
assert storage.exists(os.path.join(usb, "proj_SC_morph.json"))
# read-modify-write: an existing file is copied first
q = storage.out(os.path.join(usb, "existing_summary.csv"))
assert open(q).read() == "cell\nA\n"
time.sleep(0.05); open(q, "a").write("B\n")
assert storage.read(os.path.join(usb, "existing_summary.csv")) == q
# a whole results folder
d = storage.out_dir(os.path.join(usb, "SC_morph_results"))
for k in range(5):
    open(os.path.join(d, f"r{k}.png"), "w").write("x" * 100)
n, drives = st.summary()
print("waiting:", n, drives)
assert n == 7
# the drive is "unplugged": the results folder cannot be created
blocker = os.path.join(usb, "SC_morph_results")
open(blocker, "w").write("in the way")
copied, failed = st.sync()
print("first sync: copied", copied, "failed", len(failed))
assert copied == 2 and len(failed) == 5
assert open(os.path.join(usb, "existing_summary.csv")).read() == "cell\nA\nB\n"
assert json.load(open(os.path.join(usb, "proj_SC_morph.json"))) == {"a": 1}
assert st.summary()[0] == 5
# a fresh start still knows what is waiting
storage._store = None
st = storage.store()
assert st.summary()[0] == 5
os.remove(blocker)
copied, failed = st.sync()
print("after plugging back in: copied", copied, "failed", len(failed))
assert copied == 5 and not failed and st.summary()[0] == 0
assert sorted(os.listdir(os.path.join(usb, "SC_morph_results"))) == [f"r{k}.png" for k in range(5)]
assert not os.path.exists(os.path.dirname(storage.store().local_for(os.path.join(usb, "SC_morph_results", "r0.png"))))
# after copying, reads go to the drive again
assert storage.read(os.path.join(usb, "proj_SC_morph.json")) == os.path.join(usb, "proj_SC_morph.json")
print("STORAGE OK")

# ---- closing the app: copy, or stay open if the drive is missing and the user cancels
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from qtpy.QtWidgets import QApplication, QMessageBox
import unittest.mock as um
app = QApplication.instance() or QApplication([])
import napari
from arborist.gui import SCMorphWidget, _install_close_hook
v = napari.Viewer(show=False)
w = SCMorphWidget(v)
_install_close_hook(v, w)
assert w.cmb_storage.currentIndex() == storage.MODES.index("always")
f = storage.out(os.path.join(usb, "late.txt")); open(f, "w").write("late")
w._storage_status(); assert "1 file" in w.lbl_storage.text(), w.lbl_storage.text()
open(os.path.join(usb, "blocked_dir"), "w").write("x")
g = storage.out(os.path.join(usb, "blocked_dir", "a.txt")); open(g, "w").write("a")
with um.patch.object(QMessageBox, "warning", return_value=QMessageBox.Cancel):
    assert w._on_app_close() is False                 # stay open
assert open(os.path.join(usb, "late.txt")).read() == "late"
with um.patch.object(QMessageBox, "warning", return_value=QMessageBox.Close):
    assert w._on_app_close() is True                  # close; the file waits for next time
assert storage.store().summary()[0] == 1
os.remove(os.path.join(usb, "blocked_dir"))
assert w._on_app_close() is True and storage.store().summary()[0] == 0
w._closing_ok = True
v.close()
print("CLOSE HOOK OK")
