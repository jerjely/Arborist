import os
import sys


def _conda_paths():
    """When started from the installer's shortcut the environment is not
    'activated', so on Windows put its DLL folders on PATH (Qt, HDF5, etc.)."""
    if os.name != "nt":
        return
    p = sys.prefix
    dirs = [p, os.path.join(p, "Library", "mingw-w64", "bin"), os.path.join(p, "Library", "usr", "bin"),
            os.path.join(p, "Library", "bin"), os.path.join(p, "Scripts")]
    cur = os.environ.get("PATH", "")
    have = {d.lower() for d in cur.split(os.pathsep)}
    add = [d for d in dirs if os.path.isdir(d) and d.lower() not in have]
    if add:
        os.environ["PATH"] = os.pathsep.join(add + [cur])
    for d in add:
        try:
            os.add_dll_directory(d)
        except (OSError, AttributeError):
            pass
    os.environ.setdefault("CONDA_PREFIX", p)


def _log_when_no_console():
    """Started from a shortcut there is no console: keep messages and errors in a log file."""
    try:
        if sys.stderr is not None and sys.stderr.isatty():
            return
    except Exception:
        pass
    if sys.platform == "darwin":
        d = os.path.expanduser("~/Library/Logs")
    elif os.name == "nt":
        d = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "Arborist")
    else:
        d = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "Arborist")
    try:
        os.makedirs(d, exist_ok=True)
        f = open(os.path.join(d, "Arborist.log"), "a", buffering=1, encoding="utf-8", errors="replace")
        import time
        f.write(f"\n===== Arborist started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        sys.stdout = sys.stderr = f
    except Exception:
        pass


_conda_paths()
_log_when_no_console()

from .gui import main  # noqa: E402

if __name__ == "__main__":
    main()
