"""Loading confocal z-stacks (.czi, .tif/.tiff) with their voxel size.

Every stack is returned as a ``Stack`` with data ordered (C, Z, Y, X) and the
voxel size in micrometres as (z, y, x).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Stack:
    path: str
    data: np.ndarray                      # (C, Z, Y, X)
    voxel: tuple                          # (z, y, x) in um
    channel_names: list = field(default_factory=list)
    calibrated: bool = True               # False if voxel size was guessed

    @property
    def name(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def shape_zyx(self):
        return self.data.shape[1:]


# --------------------------------------------------------------------- helpers
def _to_czyx(arr: np.ndarray, axes: str) -> np.ndarray:
    """Reorder an array with tifffile/czifile axis letters to (C, Z, Y, X).

    Unknown axes (T, scenes, ...) are reduced by taking index 0.
    RGB samples ('S') are treated as channels.
    """
    axes = axes.upper()
    arr = np.asarray(arr)
    # merge S into C
    if "S" in axes and "C" not in axes:
        axes = axes.replace("S", "C")
    elif "S" in axes and "C" in axes:
        i = axes.index("S")
        arr = np.take(arr, 0, axis=i)
        axes = axes[:i] + axes[i + 1:]
    # drop everything that is not CZYX
    for ax in list(axes):
        if ax not in "CZYX":
            i = axes.index(ax)
            arr = np.take(arr, 0, axis=i)
            axes = axes[:i] + axes[i + 1:]
    for ax in "CZ":
        if ax not in axes:
            arr = arr[np.newaxis]
            axes = ax + axes
    order = [axes.index(a) for a in "CZYX"]
    return np.transpose(arr, order)


def _unit_to_um(unit: str | None) -> float | None:
    if unit is None:
        return None
    u = str(unit).strip().lower().replace("\\u00b5", "u")
    table = {"um": 1, "µm": 1, "micron": 1, "microns": 1, "micrometer": 1,
             "micrometre": 1, "mm": 1e3, "nm": 1e-3, "m": 1e6, "meter": 1e6,
             "metre": 1e6, "cm": 1e4}
    return table.get(u)


# ------------------------------------------------------------------------ TIFF
def _load_tiff(path: str) -> Stack:
    import tifffile

    tif_lazy = None
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes
        arr = None
        if series.nbytes > LAZY_BYTES and "S" not in axes.upper():
            tif_lazy = _lazy_tiff_planes(path, series)
        if tif_lazy is None:
            arr = series.asarray()
        vz = vy = vx = None
        calibrated = False
        names: list = []

        # OME-TIFF
        if tif.is_ome and tif.ome_metadata:
            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(tif.ome_metadata)
                pix = next(el for el in root.iter() if el.tag.endswith("Pixels"))
                f = {}
                for ax in "XYZ":
                    val = pix.get(f"PhysicalSize{ax}")
                    unit = pix.get(f"PhysicalSize{ax}Unit", "µm")
                    mult = _unit_to_um(unit) or 1.0
                    f[ax] = float(val) * mult if val else None
                vx, vy, vz = f["X"], f["Y"], f["Z"]
                calibrated = vx is not None
                names = [c.get("Name", f"C{i}") for i, c in
                         enumerate(el for el in pix if el.tag.endswith("Channel"))]
            except Exception:
                pass

        # ImageJ hyperstack
        if not calibrated:
            ij = tif.imagej_metadata or {}
            page = tif.pages[0]
            mult = _unit_to_um(ij.get("unit"))
            try:
                xr = page.tags["XResolution"].value
                yr = page.tags["YResolution"].value
                px_x = xr[1] / xr[0]
                px_y = yr[1] / yr[0]
            except Exception:
                px_x = px_y = None
            if mult is not None and px_x:
                vx, vy = px_x * mult, px_y * mult
                vz = ij.get("spacing", None)
                vz = vz * mult if vz else None
                calibrated = True
            elif ij.get("spacing"):
                vz = ij.get("spacing")
            labels = ij.get("Labels") or ij.get("labels")
            if not names and isinstance(labels, (list, tuple)):
                names = list(labels)

    data = tif_lazy if tif_lazy is not None else _to_czyx(arr, axes)
    vx = vx or 1.0
    vy = vy or vx
    vz = vz or 1.0
    if len(names) != data.shape[0]:
        names = [f"C{i + 1}" for i in range(data.shape[0])]
    return Stack(path, data, (float(vz), float(vy), float(vx)), names, calibrated)


def _lazy_tiff_planes(path, series):
    """Big TIFF read one plane (page) at a time: dask array (C, Z, Y, X), or None if the page layout is
    not simple enough (then the file is read whole)."""
    import tifffile
    axes = series.axes.upper()
    shape = series.shape
    if axes[-2:] != "YX":
        return None
    lead = axes[:-2]
    lead_shape = shape[:-2]
    try:
        n_pages = len(series.pages)
    except Exception:
        return None
    if int(np.prod(lead_shape)) != n_pages:
        return None
    C = lead_shape[lead.index("C")] if "C" in lead else 1
    Z = lead_shape[lead.index("Z")] if "Z" in lead else 1
    Y, X = shape[-2:]
    fh = {"tif": None}
    lock = __import__("threading").Lock()

    def read(c, z):
        idx = []
        for ax, n in zip(lead, lead_shape):
            idx.append(c if ax == "C" else z if ax == "Z" else 0)
        flat = int(np.ravel_multi_index(idx, lead_shape)) if lead else 0
        with lock:
            if fh["tif"] is None:
                fh["tif"] = tifffile.TiffFile(path)
            return fh["tif"].series[0].pages[flat].asarray()
    return lazy_planes(read, C, Z, Y, X, series.dtype)


# ------------------------------------------------------------------------- CZI
LAZY_BYTES = 3e8            # stacks bigger than this are read plane by plane (only what is shown)
_OPEN = {}                  # keep lazily-read CZI files open


def _lazy_czi(path, key, img):
    """Dask array (C, Z, Y, X) that reads one plane at a time from the CZI file."""
    import threading
    import czifile
    import dask
    import dask.array as da

    dims = list(img.dims)
    sizes = img.sizes
    start = dict(zip(dims, img.start))
    C, Z, Y, X = sizes.get("C", 1), sizes.get("Z", 1), sizes["Y"], sizes["X"]
    fixed = {d: start[d] for d in dims if d not in ("C", "Z", "Y", "X", "S")}
    if path not in _OPEN:
        _OPEN[path] = (czifile.CziFile(path), threading.Lock())
    fh, lock = _OPEN[path]

    def read(c, z):
        sel = dict(fixed)
        if "C" in dims:
            sel["C"] = start["C"] + c
        if "Z" in dims:
            sel["Z"] = start["Z"] + z
        with lock:
            a = np.asarray(fh.scenes[key](**sel).asarray())
        if a.ndim == 3 and "S" in dims:
            a = a[..., 0]
        while a.ndim > 2:
            a = a[0]
        return a

    return lazy_planes(read, C, Z, Y, X, img.dtype)


def _cache_budget_bytes():
    """Memory for cached planes: a quarter of the RAM, between 0.5 and 4 GB."""
    try:
        import psutil
        ram = psutil.virtual_memory().total
    except Exception:
        try:
            ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except Exception:
            ram = 8e9
    return int(min(4e9, max(5e8, ram / 4)))


class PlaneCache:
    """Keeps recently viewed planes in memory and reads the next plane in the scroll direction in the
    background, so stepping back and forth while marking does not re-read the file."""

    def __init__(self, read, nz, plane_bytes, budget=None):
        import collections
        import threading
        self.read, self.nz = read, nz
        self.max_planes = max(4, int((budget or _cache_budget_bytes()) // max(plane_bytes, 1)))
        self.data = collections.OrderedDict()
        self.lock = threading.Lock()
        self.busy = {}
        self.last_z = {}
        self.pool = None
        self.hits = self.misses = 0

    def _store(self, key, a):
        with self.lock:
            self.data[key] = a
            self.data.move_to_end(key)
            while len(self.data) > self.max_planes:
                self.data.popitem(last=False)

    def _load(self, key):
        import threading
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
                return self.data[key]
            ev = self.busy.get(key)
            if ev is None:
                ev = self.busy[key] = threading.Event()
                mine = True
            else:
                mine = False
        if not mine:                     # being read in the background right now: wait for it
            ev.wait()
            with self.lock:
                if key in self.data:
                    return self.data[key]
            return self._load(key)
        try:
            a = self.read(*key)
            self._store(key, a)
            return a
        finally:
            with self.lock:
                self.busy.pop(key, None)
            ev.set()

    def get(self, c, z):
        with self.lock:
            hit = (c, z) in self.data
        self.hits += hit; self.misses += not hit
        a = self._load((c, z))
        step = z - self.last_z.get(c, z)
        self.last_z[c] = z
        nxt = z + (1 if step >= 0 else -1)
        if 0 <= nxt < self.nz:
            with self.lock:
                need = (c, nxt) not in self.data and (c, nxt) not in self.busy
            if need:
                if self.pool is None:
                    from concurrent.futures import ThreadPoolExecutor
                    self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arborist_prefetch")
                self.pool.submit(self._load, (c, nxt))
        return a


def lazy_planes(read, C, Z, Y, X, dtype, budget=None):
    """Dask array (C, Z, Y, X) whose planes come from read(c, z), through a PlaneCache."""
    import dask
    import dask.array as da
    cache = PlaneCache(read, Z, Y * X * np.dtype(dtype).itemsize, budget)
    planes = [[da.from_delayed(dask.delayed(cache.get, pure=True)(c, z), (Y, X), dtype=dtype)
               for z in range(Z)] for c in range(C)]
    arr = da.stack([da.stack(p) for p in planes])
    arr._sc_cache = cache
    return arr


def _load_czi(path: str, scene: int = 0, lazy: bool | None = None) -> Stack:
    import czifile

    with czifile.CziFile(path) as czi:
        scenes = czi.scenes
        keys = list(scenes.keys()) if hasattr(scenes, "keys") else list(range(len(scenes)))
        key = keys[min(scene, len(keys) - 1)] if len(keys) else None
        img = scenes[key] if len(keys) else czi
        dims = "".join(img.dims)
        if lazy is None:
            lazy = img.nbytes > LAZY_BYTES
        arr = None if lazy else img.asarray()
        scales = dict(getattr(img, "coord_scales", {}) or {})
        units = dict(getattr(img, "coord_units", {}) or {})
        try:
            names = list(img.channels.keys())
        except Exception:
            names = []

    def um(ax):
        s = scales.get(ax)
        if s is None:
            return None
        mult = _unit_to_um(units.get(ax, "meter")) or 1e6
        return float(s) * mult

    vx, vy, vz = um("X"), um("Y"), um("Z")
    calibrated = vx is not None
    data = _lazy_czi(path, key, img) if lazy else _to_czyx(arr, dims)
    if len(names) != data.shape[0]:
        names = [f"C{i + 1}" for i in range(data.shape[0])]
    return Stack(path, data, (vz or 1.0, vy or vx or 1.0, vx or 1.0), names, calibrated)


def load_stack(path: str, scene: int = 0, lazy: bool | None = None) -> Stack:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".czi":
        return _load_czi(path, scene, lazy)
    if ext in (".tif", ".tiff"):
        return _load_tiff(path)
    raise ValueError(f"Unsupported file type: {ext} (use .czi or .tif)")


# ------------------------------------------------------- legacy Fiji point list
def read_fiji_points(path: str, voxel_z: float, slice_one_based: bool = True) -> np.ndarray:
    """Read a Fiji point-tool table (index, X, Y, Slice) -> (N, 3) xyz in um.

    X and Y are expected in calibrated units (as Fiji exports them for a
    calibrated image).  ``Slice`` is 1-based in Fiji; it is converted to the
    same z origin SNT uses for .swc files (first slice = 0 um).
    """
    import pandas as pd

    df = pd.read_csv(path, sep=None, engine="python")
    cols = {c.lower().strip(): c for c in df.columns}
    x = df[cols["x"]].to_numpy(float)
    y = df[cols["y"]].to_numpy(float)
    zc = cols.get("slice") or cols.get("z")
    s = df[zc].to_numpy(float)
    if slice_one_based and "slice" in cols:
        s = s - 1
    return np.column_stack([x, y, s * voxel_z])


def is_lazy(arr) -> bool:
    return not isinstance(arr, np.ndarray)


def preview_volume(data_czyx, voxel, max_xy=512, max_z=48, channel=None):
    """Small in-memory copy (max over channels, or one channel) for whole-volume tasks."""
    C, Z, Y, X = data_czyx.shape
    s = max(1, int(np.ceil(max(Y, X) / max_xy)))
    sz = max(1, int(np.ceil(Z / max_z)))
    sub = data_czyx[:, ::sz, ::s, ::s] if channel is None else data_czyx[channel:channel + 1, ::sz, ::s, ::s]
    sub = np.asarray(sub.compute() if hasattr(sub, "compute") else sub)
    vol = sub.max(axis=0)
    return vol, (voxel[0] * sz, voxel[1] * s, voxel[2] * s)


def contrast_limits(data_czyx):
    """Per-channel display range from the middle plane (avoids reading whole lazy stacks)."""
    C, Z = data_czyx.shape[:2]
    out = []
    for c in range(C):
        pl = data_czyx[c, Z // 2]
        pl = np.asarray(pl.compute() if hasattr(pl, "compute") else pl)
        lo, hi = np.percentile(pl[::4, ::4], [0.5, 99.9])
        out.append([float(lo), float(max(hi, lo + 1))])
    return out
