"""Work on the computer's own disk; copy to a USB stick / external drive when the app closes.

Every file Arborist writes (project, surface points and meshes, analysis results, previews, the
database) goes through `out(path)`. If the destination is on a slow drive (USB stick, external
or network drive - or always, if chosen), the file is written to a mirror folder on the computer
instead, and `sync()` copies it to its real place later: when the app closes, when "Copy now" is
pressed, or at the next start if the app was closed abruptly. Reads go through `read(path)`, which
returns the newer local copy while it has not been copied yet.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time

MODES = ("auto", "always", "never")
_lock = threading.RLock()


def base_dir():
    if os.environ.get("ARBORIST_STORAGE_ROOT"):          # tests
        return os.environ["ARBORIST_STORAGE_ROOT"]
    if sys.platform.startswith("win"):
        root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return os.path.join(root, "Arborist")
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Caches/Arborist")
    return os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "Arborist")


def _migrate_from_sc_morph(new_root):
    """The program used to be called SC_morph: move its local folder (files still waiting to be copied to a
    USB stick, the saving setting) to the new name and fix the paths stored in the list of waiting files."""
    import shutil
    old_root = os.path.join(os.path.dirname(new_root), "SC_morph")
    if os.path.basename(new_root) != "Arborist" or not os.path.isdir(old_root) or os.path.exists(new_root):
        return
    try:
        shutil.move(old_root, new_root)
    except OSError:
        return
    p = os.path.join(new_root, "pending.json")
    try:
        with open(p, encoding="utf-8") as fh:
            pend = json.load(fh)
        fixed = {(new_root + k[len(old_root):] if k.startswith(old_root) else k): v for k, v in pend.items()}
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(fixed, fh, indent=1)
    except (OSError, ValueError):
        pass


class Store:
    def __init__(self, root=None):
        self.root = root or base_dir()
        if root is None:
            _migrate_from_sc_morph(self.root)
        self.work = os.path.join(self.root, "work")
        self._settings_path = os.path.join(self.root, "settings.json")
        self._manifest_path = os.path.join(self.root, "pending.json")
        self.settings = self._load(self._settings_path, {"mode": "auto"})
        self.pending = self._load(self._manifest_path, {})      # local dir -> real dir
        self._slow_cache = {}
        self._synced = {}            # local file -> (mtime_ns, size) when it was last identical to the real one
        self.alt_roots = None        # tests: callable returning the drive roots to search for moved folders

    # -------------------------------------------------------------- settings
    @staticmethod
    def _load(p, default):
        try:
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return dict(default)

    def _save(self, p, d):
        os.makedirs(self.root, exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=1)
        os.replace(tmp, p)

    @property
    def mode(self):
        return self.settings.get("mode", "auto")

    def set_mode(self, mode):
        assert mode in MODES
        self.settings["mode"] = mode
        self._save(self._settings_path, self.settings)
        self._slow_cache.clear()

    # -------------------------------------------------------------- which drive
    @staticmethod
    def _drive_root(path):
        path = os.path.abspath(path)
        if sys.platform.startswith("win"):
            d = os.path.splitdrive(path)[0]
            return (d + "\\") if d else path
        parts = path.split(os.sep)
        if sys.platform == "darwin" and len(parts) > 2 and parts[1] == "Volumes":
            return os.sep.join(parts[:3])
        if len(parts) > 3 and parts[1] in ("media", "mnt") or (len(parts) > 4 and parts[1:3] == ["run", "media"]):
            return os.sep.join(parts[:4] if parts[1] == "run" else parts[:3])
        return os.sep

    def is_slow(self, path):
        """USB stick, external or network drive?"""
        if self.mode == "never":
            return False
        if self.mode == "always":
            return True
        root = self._drive_root(path)
        if root in self._slow_cache:
            return self._slow_cache[root]
        slow = False
        try:
            if sys.platform.startswith("win"):
                import ctypes
                t = ctypes.windll.kernel32.GetDriveTypeW(ctypes.c_wchar_p(root))
                # 2 = removable (USB sticks, card readers), 4 = network; external USB disks report 3
                # (fixed), so also treat any drive other than the system drive as external
                sysdrive = (os.environ.get("SystemDrive", "C:") + "\\").upper()
                slow = t in (2, 4) or (t == 3 and root.upper() != sysdrive)
            elif sys.platform == "darwin":
                slow = root.startswith("/Volumes/")
            else:
                slow = root != os.sep
        except Exception:
            slow = False
        self._slow_cache[root] = slow
        return slow

    # -------------------------------------------------------------- paths
    def local_for(self, real):
        real = os.path.abspath(real)
        drive, rest = os.path.splitdrive(real)
        tag = drive.replace(":", "").replace("\\", "_").replace("/", "_") or "root"
        return os.path.join(self.work, tag, rest.lstrip("\\/"))

    # ---- comparing a local copy with the real file. Timestamps are compared with a tolerance because
    # USB sticks (FAT / exFAT) round them (up to 2 s), and never by "sizes differ" alone: a local copy
    # that is older than the real file must never be written over it.
    TOL = 2.5

    @staticmethod
    def _sig(f):
        st = os.stat(f)
        return (st.st_mtime_ns, st.st_size)

    def _real_newer(self, loc, real):
        try:
            return os.path.isfile(real) and os.path.getmtime(real) > os.path.getmtime(loc) + self.TOL
        except OSError:
            return False

    def out(self, real):
        """Where to write `real` now. Creates the folder; a file that already exists at the real place
        is copied first, so read-modify-write (e.g. appending to a table) works."""
        if not self.is_slow(real):
            d = os.path.dirname(os.path.abspath(real))
            os.makedirs(d, exist_ok=True)
            return real
        loc = self.local_for(real)
        with _lock:
            os.makedirs(os.path.dirname(loc), exist_ok=True)
            if (not os.path.exists(loc) or self._real_newer(loc, real)) and os.path.isfile(real):
                try:
                    shutil.copy2(real, loc)
                    self._synced[loc] = self._sig(loc)          # identical to the real file for now
                except OSError:
                    pass
            ld, rd = os.path.dirname(loc), os.path.dirname(os.path.abspath(real))
            if self.pending.get(ld) != rd:
                self.pending[ld] = rd
                self._save(self._manifest_path, self.pending)
        return loc

    def out_dir(self, real_dir):
        """A folder to write many files into (e.g. analysis results)."""
        loc = os.path.dirname(self.out(os.path.join(real_dir, ".arborist_dir")))
        try:
            os.remove(os.path.join(loc, ".arborist_dir"))
        except OSError:
            pass
        return loc

    def read(self, real):
        """The newest copy of `real`: the local one while it has not been copied yet."""
        loc = self.local_for(real)
        if os.path.isfile(loc) and os.path.dirname(loc) in self.pending and not self._real_newer(loc, real):
            return loc
        return real

    def exists(self, real):
        return os.path.exists(self.read(real))

    # -------------------------------------------------------------- copying back
    def _is_waiting(self, lf, rf):
        try:
            if not os.path.isfile(rf):
                return True
            sig = self._sig(lf)
            if self._synced.get(lf) == sig:
                return False
            ml, mr = os.path.getmtime(lf), os.path.getmtime(rf)
            if mr > ml + self.TOL:
                return False                       # the real file is newer (changed elsewhere): keep it
            if abs(ml - mr) <= self.TOL and sig[1] == os.path.getsize(rf):
                return False
            return True
        except OSError:
            return True

    def _local_files(self):
        for ld, rd in list(self.pending.items()):
            if not os.path.isdir(ld):
                continue
            for name in os.listdir(ld):
                lf = os.path.join(ld, name)
                if os.path.isfile(lf) and not name.endswith(".tmp"):
                    yield lf, os.path.join(rd, name)

    def waiting(self):
        """[(local file, real file)] not copied yet."""
        return [(lf, rf) for lf, rf in self._local_files() if self._is_waiting(lf, rf)]

    def pending_count(self):
        """Quick count for the status line: looks only at this computer, never at the (slow) drive."""
        n, drives = 0, set()
        for lf, rf in self._local_files():
            try:
                if self._synced.get(lf) != self._sig(lf):
                    n += 1
                    drives.add(self._drive_root(rf))
            except OSError:
                pass
        return n, sorted(drives)

    # ---- folders that moved (the stick got another drive letter, or the data folder was renamed)
    def _roots(self):
        if self.alt_roots is not None:
            return list(self.alt_roots())
        if sys.platform.startswith("win"):
            return [f"{c}:\\" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.exists(f"{c}:\\")]
        out = []
        for base in ("/Volumes", "/media", "/run/media", "/mnt"):
            if os.path.isdir(base):
                for d in os.listdir(base):
                    q = os.path.join(base, d)
                    out.append(q)
                    if base != "/Volumes" and os.path.isdir(q):
                        out += [os.path.join(q, x) for x in os.listdir(q)]
        return out

    def _find_moved(self, folder):
        """`folder` under another drive letter / a renamed top folder, if exactly one place fits."""
        import glob
        parts = [x for x in os.path.splitdrive(os.path.abspath(folder))[1].replace("\\", "/").split("/") if x]
        for k in range(len(parts), 1, -1):                      # longest matching end of the path first
            tail = parts[-k:]
            found = set()
            for root in self._roots():
                cands = [os.path.join(root, *tail)]
                cands += glob.glob(os.path.join(glob.escape(root), "*", *[glob.escape(t) for t in tail]))
                for cand in cands:
                    if os.path.isdir(cand) and os.path.abspath(cand) != os.path.abspath(folder):
                        found.add(os.path.abspath(cand))
            if found:
                return found.pop() if len(found) == 1 else None
        return None

    def _relocate(self, rd):
        new = self._find_moved(rd)
        if new:
            return new
        parent = self._find_moved(os.path.dirname(rd))          # e.g. a results folder not made yet
        return os.path.join(parent, os.path.basename(rd)) if parent else None

    def _fix_moved(self):
        changed = False
        for ld, rd in list(self.pending.items()):
            if os.path.isdir(rd) or os.path.isdir(os.path.dirname(rd)) or not os.path.isdir(ld) \
                    or not any(os.path.isfile(os.path.join(ld, x)) for x in os.listdir(ld)):
                continue
            new = self._relocate(rd)
            if new:
                self.pending[ld] = new
                changed = True
        if changed:
            self._save(self._manifest_path, self.pending)

    def sync(self, progress=None):
        """Copy everything waiting to its real place. Returns (copied, failed [(file, error)])."""
        with _lock:
            self._fix_moved()
            todo = self.waiting()
            copied, failed = 0, []
            for i, (lf, rf) in enumerate(todo):
                if progress:
                    progress(i, len(todo), rf)
                try:
                    rd = os.path.dirname(rf)
                    if not os.path.isdir(rd) and not os.path.isdir(os.path.dirname(rd)):
                        # the data folder is not there (other drive letter?): never build a tree elsewhere
                        raise OSError(f"folder not found: {os.path.dirname(rd)}")
                    os.makedirs(rd, exist_ok=True)
                    tmp = rf + ".arborist_tmp"
                    shutil.copy2(lf, tmp)
                    os.replace(tmp, rf)
                    self._synced[lf] = self._sig(lf)
                    copied += 1
                except OSError as ex:
                    failed.append((rf, str(ex)))
            # forget folders that are fully copied; their local copies are removed to save space
            bad_dirs = {os.path.dirname(f) for f, _ in failed}
            still = {os.path.dirname(l) for l, _ in self.waiting()}
            for ld, rd in list(self.pending.items()):
                if rd in bad_dirs or ld in still:
                    continue
                # remove only this folder's own copied files - never subfolders, which are separate
                # entries (e.g. the results folder) and may still be waiting
                try:
                    for name in os.listdir(ld):
                        f = os.path.join(ld, name)
                        if os.path.isfile(f):
                            os.remove(f)
                            self._synced.pop(f, None)
                    os.rmdir(ld)
                except OSError:
                    pass
                del self.pending[ld]
            self._save(self._manifest_path, self.pending)
            if progress:
                progress(len(todo), len(todo), "")
            return copied, failed

    def summary(self):
        w = self.waiting()
        drives = sorted({self._drive_root(r) for _, r in w})
        return len(w), drives


_store = None


def store():
    global _store
    if _store is None:
        _store = Store()
    return _store


def out(path):
    return store().out(path)


def out_dir(path):
    return store().out_dir(path)


def read(path):
    return store().read(path)


def exists(path):
    return store().exists(path)
