"""test_database.py with every folder treated as a USB stick: work locally, then copy."""
import os, sys, tempfile, glob
os.environ["ARBORIST_STORAGE_ROOT"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from arborist import storage
storage.store().set_mode("always")
ns = {"__file__": os.path.join(os.path.dirname(__file__), "test_database.py"), "__name__": "__main__"}
src = open(ns["__file__"]).read()
exec(compile(src, ns["__file__"], "exec"), ns)
st = storage.store()
n, _ = st.summary()
print("files waiting on this computer:", n)
assert n > 5
# nothing of ours was written into the data folders yet
root = ns.get("ROOT") or ns.get("root")
written = [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
           if p.endswith(("_SC_morph.json", "_summary.csv", "_brain_surface.stl"))]
print("in the data folders before copying:", len(written))
assert not written, written
copied, failed = st.sync()
assert not failed, failed
written = [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
           if p.endswith(("_SC_morph.json", "_summary.csv", "_brain_surface.stl"))]
print("after copying:", sorted(os.path.basename(p) for p in written))
assert any(p.endswith("_SC_morph.json") for p in written) and any(p.endswith("_summary.csv") for p in written)
assert st.summary()[0] == 0
print("DATABASE ON USB OK")
