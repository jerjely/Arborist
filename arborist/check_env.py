"""Check that Arborist's packages import; with --repair, reinstall the ones that are broken.

Run by the launchers before the app starts:  python -m arborist.check_env --repair
A package can be present but broken (e.g. a half-finished upgrade leaves pandas without dateutil);
this finds the module that actually fails and reinstalls its pip package.
"""
import importlib
import os
import subprocess
import sys

# runnable as a plain script from anywhere (python path/to/arborist/check_env.py), not only as a module
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# module to import -> pip package that provides it
MODULES = {
    "numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "dateutil": "python-dateutil",
    "matplotlib": "matplotlib", "skimage": "scikit-image",
    "trimesh": "trimesh", "rtree": "rtree", "tifffile": "tifffile", "czifile": "czifile",
    "imagecodecs": "imagecodecs", "openpyxl": "openpyxl", "plotly": "plotly", "dask": "dask",
    "qtpy": "qtpy", "PySide6": "PySide6", "superqt": "superqt", "napari": "napari[pyside6]",
}
PIP_NAME = {"dateutil": "python-dateutil", "skimage": "scikit-image", "yaml": "pyyaml", "PIL": "pillow",
            "cv2": "opencv-python-headless", "sklearn": "scikit-learn", "attr": "attrs"}


def _missing_module(exc):
    """Name of the module that is really broken or missing, following 'during handling' / 'from' chains.

    Uses, in order: a ModuleNotFoundError's name, pandas' "required dependency X" message, and the
    package (folder under site-packages) of the file where the innermost error was raised."""
    import os
    import re
    chain, seen = [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc)); chain.append(exc)
        exc = exc.__cause__ or exc.__context__
    for e in chain:
        if isinstance(e, ModuleNotFoundError) and e.name:
            return e.name.split(".")[0]
    for e in chain:
        m = re.search(r"required dependency[:\s]+([A-Za-z_][\w]*)", str(e))
        if m:
            return m.group(1)
    tb = chain[-1].__traceback__
    last = None
    while tb is not None:
        last = tb.tb_frame.f_code.co_filename; tb = tb.tb_next
    if last:
        parts = os.path.normpath(last).split(os.sep)
        for i, part in enumerate(parts[:-1]):
            if part in ("site-packages", "dist-packages"):
                top = parts[i + 1]
                return os.path.splitext(top)[0]
    return None


def check():
    """List of (module, pip package to reinstall, error text)."""
    bad = []
    for mod, pkg in MODULES.items():
        try:
            importlib.import_module(mod)
        except Exception as ex:                    # noqa: BLE001 - any failure means broken
            root = _missing_module(ex)
            fix = PIP_NAME.get(root, MODULES.get(root, root)) if root else pkg
            bad.append((mod, fix, f"{type(ex).__name__}: {ex}"))
    return bad


def main():
    repair = "--repair" in sys.argv
    bad = check()
    if not bad:
        if repair:
            print("All packages import fine.")
        return 0
    print("Arborist: some packages do not import:")
    for mod, fix, err in bad:
        print(f"  {mod}: {err}")
    if not repair:
        return 1
    # reinstall only the root causes (pandas is fine once dateutil is fixed)
    pkgs = sorted({fix for _, fix, _ in bad})
    print("Repairing:", " ".join(pkgs))
    cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps"] + pkgs
    subprocess.call(cmd)
    # dependencies of the reinstalled packages, without touching what already works
    subprocess.call([sys.executable, "-m", "pip", "install"] + pkgs)
    importlib.invalidate_caches()
    for m in list(sys.modules):
        if m.split(".")[0] in {mod for mod, _, _ in bad} | {f.split("[")[0] for f in pkgs}:
            sys.modules.pop(m, None)
    still = subprocess.call([sys.executable, os.path.abspath(__file__)])
    if still:
        print("\nStill broken. Run install.bat again (it rebuilds the environment).")
        return 1
    print("Repaired.")
    return 0 if "--install" in sys.argv else 10      # launchers start the app again on 10


if __name__ == "__main__":
    sys.exit(main())
