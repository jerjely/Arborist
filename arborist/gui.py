"""Arborist - napari GUI for brain-surface marking, CR-border depth, SNT-style tracing
and neurite depth analysis."""
from __future__ import annotations

import json
import os
import sys
import traceback

import numpy as np
import napari
import trimesh
from qtpy.QtGui import QColor
from qtpy.QtCore import Qt, QObject, QEvent
from qtpy.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton, QRadioButton,
    QListWidget, QListWidgetItem, QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from . import analysis
from . import storage
from .io import contrast_limits, is_lazy, load_stack, preview_volume, read_fiji_points
from .surface import BrainSurface, measure_border_distance
from .swc import Neuron, TYPE_COLORS
from . import database as dbm
from .morphometry import DEFAULT_CLUSTER_FEATURES, TYPE_LABELS, cluster_cells, cluster_figures

__version__ = "0.3.0"
ROLE_TAG = {"cell": ("CELL", "[63x cell]"), "lowmag": ("LOW", "[20x CR]"), "ankg": ("ANKG", "[AnkG]"),
            "overview": ("10X", "[10x section]")}
VIEWS = [("overview", "Whole section\n10x"), ("lowmag", "Low mag\n20x CR"),
         ("cell", "High mag\n63x cell"), ("ankg", "Ankyrin G\nAIS")]
CMAPS = ["magenta", "green", "gray", "cyan", "yellow", "red", "blue"]


# --------------------------------------------------------------- coordinates
def vox2xyz(data, voxel):
    """napari layer data (z, y, x voxels) -> (x, y, z) um."""
    d = np.atleast_2d(np.asarray(data, float))
    if not d.size:
        return np.zeros((0, 3))
    return (d * np.asarray(voxel, float))[:, ::-1]


def xyz2vox(xyz, voxel):
    d = np.atleast_2d(np.asarray(xyz, float))
    if not d.size:
        return np.zeros((0, 3))
    return d[:, ::-1] / np.asarray(voxel, float)


CHECKED_STYLE = "QPushButton:checked { background-color: #d97706; color: white; font-weight: bold; }"


def _btn(text, slot=None, checkable=False, tip=None):
    b = QPushButton(text)
    b.setCheckable(checkable)
    if checkable:
        b.setStyleSheet(CHECKED_STYLE)
    if slot:
        (b.toggled if checkable else b.clicked).connect(slot)
    if tip:
        b.setToolTip(tip)
    return b


def _lbl(text):
    l = QLabel(text)
    l.setWordWrap(True)
    l.setTextFormat(Qt.RichText)
    return l


def _dspin(val, lo, hi, step=1.0, dec=1, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi); s.setDecimals(dec); s.setSingleStep(step); s.setValue(val)
    if suffix:
        s.setSuffix(suffix)
    return s


def wheel_handler(viewer, event):
    """Wheel = step through z; Ctrl+wheel = zoom about the cursor (2D view only)."""
    if viewer.dims.ndisplay != 2 or viewer.dims.ndim < 3:
        return
    dy = float(event.delta[1]) if len(event.delta) > 1 else 0.0
    if dy == 0:
        return
    mods = event.modifiers
    # zoom with Ctrl (Windows/Linux) or, on a Mac, Cmd (Qt reports it as "Control") or the
    # physical Control key (reported as "Meta")
    if "Control" in mods or "Meta" in mods:
        f = 1.15 if dy > 0 else 1 / 1.15
        cam = viewer.scene.camera if hasattr(viewer, "scene") else viewer.camera
        disp = list(viewer.dims.displayed)
        pos = np.asarray(event.position, float)[disp]
        c = np.asarray(cam.center, float)
        c2 = c[-2:]
        new = pos + (c2 - pos) / f
        cam.zoom = cam.zoom * f
        cam.center = tuple(c[:-2]) + tuple(new)
        event.handled = True
    elif not mods:
        try:
            inverted = event.native.inverted()
        except Exception:
            inverted = False
        step = (1 if inverted else -1) * int(np.sign(dy))
        axis = viewer.dims.not_displayed[0] if viewer.dims.not_displayed else 0
        cur = list(viewer.dims.current_step)
        n = viewer.dims.nsteps[axis]
        cur[axis] = int(min(max(cur[axis] + step, 0), n - 1))
        viewer.dims.current_step = tuple(cur)
        event.handled = True


# --------------------------------------------------------------- data holder
class StackEntry:
    def __init__(self, stack, role):
        self.stack = stack
        self.role = role                   # "cell", "lowmag" (20x CR) or "ankg" (ankyrin-G scan)
        self.layers: dict = {}
        self.images: list = []
        self.surface: BrainSurface | None = None
        self.flipped = False
        self.border_result: dict | None = None
        self.include = True                # include border measurement in mean
        self.neuron = Neuron() if role == "cell" else None

    @property
    def name(self):
        return self.stack.name

    @property
    def voxel(self):
        return self.stack.voxel

    def all_layers(self):
        return list(self.images) + [l for l in self.layers.values() if l is not None]


# ============================================================== main widget
class _WheelGuard(QObject):
    """Scrolling a panel with the mouse wheel must not change a number box or drop-down the pointer
    happens to pass over; they react to the wheel only after being clicked into."""

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Wheel and not obj.hasFocus():
            ev.ignore()
            return True
        return False


def _guard_wheels(root, guard):
    from qtpy.QtWidgets import QAbstractSpinBox, QComboBox
    from qtpy.QtCore import Qt as _Qt
    for wdg in root.findChildren(QAbstractSpinBox) + root.findChildren(QComboBox):
        if not wdg.property("_wheel_guarded"):
            wdg.setFocusPolicy(_Qt.StrongFocus)
            wdg.installEventFilter(guard)
            wdg.setProperty("_wheel_guarded", True)


class SCMorphWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.entries: list[StackEntry] = []
        self.active: StackEntry | None = None
        self.project_path = None

        self.setMinimumWidth(440)
        root = QVBoxLayout(self)
        vg = QGroupBox("View"); vl = QVBoxLayout(vg)
        grid = QHBoxLayout()
        self.view_btns = {}
        self.view_group = QButtonGroup(self); self.view_group.setExclusive(True)
        for role, label in VIEWS:
            b_ = QPushButton(label); b_.setCheckable(True); b_.setMinimumHeight(44); b_.setEnabled(False)
            b_.clicked.connect(lambda _=False, r=role: self.show_role(r))
            self.view_group.addButton(b_)
            self.view_btns[role] = b_
            grid.addWidget(b_)
        vl.addLayout(grid)
        h = QHBoxLayout()
        self.lbl_stack = QLabel("Stack:")
        self.cmb_active = QComboBox()
        self.cmb_active.currentIndexChanged.connect(self._on_active_changed)
        h.addWidget(self.lbl_stack); h.addWidget(self.cmb_active, 1)
        vl.addLayout(h)
        root.addWidget(vg)
        self.pending_overview: list = []
        prj = QHBoxLayout()
        prj.addWidget(_btn("Open project…", self.load_project))
        prj.addWidget(_btn("Save project…", self.save_project))
        root.addLayout(prj)

        root.addWidget(self._display_box())
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        self._wheel_guard = _WheelGuard(self)
        self.tabs.addTab(self._scroll(self._tab_stacks()), "1 Stacks")
        self.tabs.addTab(self._scroll(self._tab_surface()), "2 Surface")
        self.tabs.addTab(self._scroll(self._tab_border()), "3 CR")
        self.tabs.addTab(self._scroll(self._tab_neuron()), "4 Neuron")
        self.tabs.addTab(self._scroll(self._tab_ais()), "5 AIS")
        self.tabs.addTab(self._scroll(self._tab_analyze()), "6 Analyze")
        self.tabs.addTab(self._scroll(self._tab_classify()), "7 Classify")
        _guard_wheels(self, self._wheel_guard)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.status = QLabel("Open a cell stack to begin.")
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(48)
        self.status.setStyleSheet("color:#9cf; padding:4px")
        root.addWidget(self.status)

        self.results = ResultsWidget()
        self.results_dock = viewer.window.add_dock_widget(self.results, name="Results", area="right")
        self.results_dock.setVisible(False)

        viewer.dims.events.current_step.connect(self._update_slice_lines)
        # wheel = move through the z-stack, Ctrl+wheel = zoom (napari's default is the reverse)
        for cb in list(viewer.mouse_wheel_callbacks):
            if getattr(cb, "__name__", "") == "dims_scroll":
                viewer.mouse_wheel_callbacks.remove(cb)
        viewer.mouse_wheel_callbacks.append(wheel_handler)
        viewer.dims.events.ndisplay.connect(self._on_ndisplay)
        # napari turns mouse panning back on whenever the selected layer changes (e.g. when the surface
        # mesh layer is added after a stroke); keep it off while drawing with the pen
        from qtpy.QtCore import QTimer as _QT
        viewer.layers.selection.events.active.connect(lambda *_: _QT.singleShot(0, self._pen_update_pan))
        self._install_open_hook()
        viewer.mouse_drag_callbacks.insert(0, self._on_soma_click)
        viewer.mouse_drag_callbacks.append(self._on_pen)
        viewer.bind_key("Space", self._space_pan, overwrite=True)

    # ------------------------------------------------------------ display
    def _display_box(self):
        from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider
        g = QGroupBox("Display (brightness / contrast)"); f = QFormLayout(g)
        h = QHBoxLayout()
        self.cmb_disp = QComboBox(); self.cmb_disp.currentIndexChanged.connect(self._disp_channel_changed)
        self.cb_disp_vis = QCheckBox("show"); self.cb_disp_vis.toggled.connect(self._disp_set_visible)
        h.addWidget(self.cmb_disp, 1); h.addWidget(self.cb_disp_vis)
        f.addRow("Channel", h)
        self.sl_contrast = QLabeledDoubleRangeSlider(Qt.Horizontal)
        self.sl_contrast.valueChanged.connect(self._disp_set_contrast)
        f.addRow("Min / max", self.sl_contrast)
        self.sl_gamma = QLabeledDoubleSlider(Qt.Horizontal)
        self.sl_gamma.setRange(0.2, 2.0); self.sl_gamma.setValue(1.0)
        self.sl_gamma.valueChanged.connect(self._disp_set_gamma)
        f.addRow("Gamma", self.sl_gamma)
        h = QHBoxLayout()
        h.addWidget(_btn("Auto (this slice)", self._disp_auto, tip="Stretch to the 0.5–99.8 % range of the current slice"))
        h.addWidget(_btn("Auto all channels", lambda: self._disp_auto(all_channels=True)))
        f.addRow(h)
        zoom_key = "⌘ or Ctrl" if sys.platform == "darwin" else "Ctrl"
        f.addRow(_lbl(f"<i>Mouse wheel / two-finger scroll: move through the stack · {zoom_key}+wheel: zoom · "
                      "drag: pan (while drawing with the pen: hold Space and drag)</i>"))
        self._disp_layer = None
        self._disp_lock = False
        return g

    def _refresh_display_channels(self):
        if not hasattr(self, "cmb_disp"):
            return
        self.cmb_disp.blockSignals(True)
        self.cmb_disp.clear()
        if self.active:
            for l in self.active.images:
                self.cmb_disp.addItem(l.name.split(" | ")[-1])
        self.cmb_disp.blockSignals(False)
        self._disp_channel_changed(0)

    def _disp_channel_changed(self, idx):
        if self._disp_layer is not None:
            try:
                self._disp_layer.events.contrast_limits.disconnect(self._disp_from_layer)
            except Exception:
                pass
        self._disp_layer = None
        if not self.active or idx < 0 or idx >= len(self.active.images):
            return
        lay = self.active.images[idx]
        self._disp_layer = lay
        lay.events.contrast_limits.connect(self._disp_from_layer)
        self._disp_from_layer()

    def _disp_from_layer(self, *_):
        lay = self._disp_layer
        if lay is None or self._disp_lock:
            return
        self._disp_lock = True
        try:
            lo, hi = lay.contrast_limits_range
            clo, chi = lay.contrast_limits
            lo, hi = min(lo, clo), max(hi, chi)
            self.sl_contrast.setRange(float(lo), float(hi))
            self.sl_contrast.setValue((float(clo), float(chi)))
            self.sl_gamma.setValue(float(lay.gamma))
            self.cb_disp_vis.setChecked(bool(lay.visible))
        finally:
            self._disp_lock = False

    def _disp_set_contrast(self, val):
        if self._disp_layer is not None and not self._disp_lock and val[1] > val[0]:
            self._disp_lock = True
            try:
                self._disp_layer.contrast_limits = (float(val[0]), float(val[1]))
            finally:
                self._disp_lock = False

    def _disp_set_gamma(self, v):
        if self._disp_layer is not None and not self._disp_lock:
            self._disp_layer.gamma = float(v)

    def _disp_set_visible(self, on):
        if self._disp_layer is not None and not self._disp_lock:
            self._disp_layer.visible = on

    @staticmethod
    def _pyramid(arr, tile=2048, start=4096):
        """Large planes as an image pyramid (full, 1/2, 1/4 ...): napari then draws only what is on
        screen at the resolution needed, instead of shrinking a huge plane on every slice change
        (planes wider than the graphics card's texture limit, e.g. stitched 63x scans)."""
        Y, X = arr.shape[-2:]
        if max(Y, X) <= start:
            return arr
        levels, s = [arr], 2
        while max(Y, X) / s > tile / 2:
            levels.append(arr[..., ::s, ::s])
            s *= 2
        return levels

    def _current_plane(self, lay):
        data = lay.data
        if getattr(lay, "multiscale", False):
            # a level with about a million pixels per plane is plenty for contrast percentiles
            sizes = [int(np.prod(l.shape[-2:])) for l in data]
            data = data[max([i for i, n in enumerate(sizes) if n >= 1e6] or [len(sizes) - 1])]
        zi = int(round(float(self.viewer.dims.point[0]) / float(lay.scale[0])))
        zi = max(0, min(zi, data.shape[0] - 1))
        pl = data[zi]
        return np.asarray(pl.compute() if hasattr(pl, "compute") else pl)

    def _disp_auto(self, all_channels=False):
        if not self.active:
            return
        lays = self.active.images if all_channels else ([self._disp_layer] if self._disp_layer else [])
        for lay in lays:
            pl = self._current_plane(lay)
            step = max(1, int(np.sqrt(pl.size / 1e6)))
            lo, hi = np.percentile(pl[::step, ::step], [0.5, 99.8])
            if hi <= lo:
                hi = lo + 1
            rlo, rhi = lay.contrast_limits_range
            lay.contrast_limits_range = (min(rlo, lo), max(rhi, hi))
            lay.contrast_limits = (float(lo), float(hi))
        self._disp_from_layer()

    @staticmethod
    def _scroll(w):
        s = QScrollArea(); s.setWidgetResizable(True); s.setWidget(w)
        s.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return s

    def msg(self, text, error=False):
        self.status.setText(text)
        self.status.setStyleSheet(f"color:{'#f88' if error else '#9cf'}; padding:4px")

    def _guard(self, fn, *a):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            return fn(*a)
        except Exception as e:  # show errors instead of crashing the GUI
            traceback.print_exc()
            self.msg(f"Error: {e}", error=True)
        finally:
            QApplication.restoreOverrideCursor()

    # ================================================================ tab 1
    def _tab_stacks(self):
        w = QWidget(); L = QVBoxLayout(w)
        L.addWidget(_lbl("<b>Cell stack</b> = high-mag stack with the filled neuron.<br>"
                           "<b>Low-mag stacks</b> = overview stacks used only to measure the depth "
                           "of the Calretinin border."))
        L.addWidget(_btn("Open cell stack (high-mag)…", lambda: self.open_stacks("cell")))
        L.addWidget(_btn("Open low-mag stack(s)…", lambda: self.open_stacks("lowmag")))
        L.addWidget(_btn("Open ankyrin-G (AIS) stack…", lambda: self.open_stacks("ankg")))
        L.addWidget(_btn("Open whole-section (10x) image…", lambda: self.open_stacks("overview")))
        self.tbl_stacks = QTableWidget(0, 4)
        self.tbl_stacks.setHorizontalHeaderLabels(["Stack", "Role", "Voxel z,y,x (µm)", "Channels"])
        self.tbl_stacks.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_stacks.setEditTriggers(QTableWidget.NoEditTriggers)
        L.addWidget(self.tbl_stacks)
        g = QGroupBox("Voxel size of active stack"); f = QFormLayout(g)
        self.sp_vz = _dspin(1, 0.001, 1000, 0.1, 4, " µm")
        self.sp_vy = _dspin(1, 0.001, 1000, 0.1, 4, " µm")
        self.sp_vx = _dspin(1, 0.001, 1000, 0.1, 4, " µm")
        f.addRow("z", self.sp_vz); f.addRow("y", self.sp_vy); f.addRow("x", self.sp_vx)
        f.addRow(_btn("Apply voxel size", self.apply_voxel))
        L.addWidget(g)
        L.addWidget(_btn("Remove active stack", self.remove_active))
        L.addWidget(self._storage_box())
        L.addStretch()
        return w

    # ------------------------------------------------------------ saving to a USB stick
    def _storage_box(self):
        g = QGroupBox("Saving (USB stick / external drive)"); v = QVBoxLayout(g)
        self.cmb_storage = QComboBox()
        self.cmb_storage.addItems(["Work on this computer, copy to USB / external drives when closing",
                                   "Always work on this computer, copy when closing",
                                   "Save straight into the folders"])
        self.cmb_storage.setCurrentIndex(storage.MODES.index(storage.store().mode))
        self.cmb_storage.currentIndexChanged.connect(self._storage_mode_changed)
        self.cmb_storage.setToolTip("Writing to a USB stick is slow. With the first option the project, surface "
                                    "and results are saved on this computer while you work and copied to the "
                                    "stick when Arborist closes (or when you press Copy now).")
        v.addWidget(self.cmb_storage)
        h = QHBoxLayout()
        self.lbl_storage = _lbl("")
        h.addWidget(self.lbl_storage, 1)
        h.addWidget(_btn("Copy now", lambda: self._storage_sync(ask=False)))
        h.addWidget(_btn("Show…", self._storage_show, tip="Open the folder on this computer that holds the "
                                                           "files not copied to the drive yet."))
        v.addLayout(h)
        from qtpy.QtCore import QTimer
        self._storage_timer = QTimer(self); self._storage_timer.timeout.connect(self._storage_status)
        self._storage_timer.start(5000)
        QTimer.singleShot(1500, self._storage_startup)
        return g

    def _storage_mode_changed(self, i):
        # copy what is waiting first, so that files written straight to the folders afterwards never
        # start from an older copy
        if storage.store().pending_count()[0]:
            self._storage_sync(ask=False, quiet_ok=True)
        storage.store().set_mode(storage.MODES[i])
        self._storage_status()

    def _storage_show(self):
        from qtpy.QtCore import QUrl
        from qtpy.QtGui import QDesktopServices
        d = storage.store().work
        os.makedirs(d, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(d))

    def _storage_status(self):
        try:
            n, drives = storage.store().pending_count()     # looks only at this computer: cheap
        except Exception:
            return
        if not hasattr(self, "lbl_storage"):
            return
        if n:
            self.lbl_storage.setText(f"<span style='color:#e67e22'><b>{n} file{'s' if n != 1 else ''}</b> waiting "
                                     f"to be copied to {', '.join(drives)}</span>")
        else:
            self.lbl_storage.setText("<i>Everything is in its folder.</i>")

    def _storage_startup(self):
        """Files left on this computer last time (app closed abruptly or the stick was unplugged)."""
        n, drives = storage.store().summary()
        if n:
            self.msg(f"{n} files from last time were not copied to {', '.join(drives)} yet - copying now.")
            self._storage_sync(ask=False, quiet_ok=True)
        self._storage_status()

    def _storage_sync(self, ask=True, quiet_ok=False):
        """Copy everything waiting. Returns True if nothing is left waiting (or the user accepts that)."""
        from qtpy.QtWidgets import QProgressDialog
        st = storage.store()
        n, drives = st.summary()
        if not n:
            self._storage_status()
            return True
        dlg = QProgressDialog(f"Copying {n} files to {', '.join(drives)}…", None, 0, n, self.window())
        dlg.setWindowTitle("Arborist"); dlg.setMinimumDuration(400); dlg.setValue(0)

        def prog(i, total, name):
            dlg.setMaximum(max(total, 1)); dlg.setValue(i)
            if name:
                dlg.setLabelText(f"Copying {i + 1} of {total}\n{os.path.basename(name)}")
            QApplication.processEvents()
        copied, failed = st.sync(prog)
        dlg.close()
        self._storage_status()
        if not failed:
            if not quiet_ok:
                self.msg(f"Copied {copied} files to {', '.join(drives)}.")
            return True
        detail = "\n".join(f"{os.path.basename(f)}: {e}" for f, e in failed[:6])
        if not ask:
            self.msg(f"{len(failed)} files could not be copied (is the drive plugged in?). They stay on this "
                     f"computer and are copied later.", error=True)
            return False
        r = QMessageBox.warning(
            self.window(), "Not copied yet",
            f"{len(failed)} files could not be copied to {', '.join(drives)} - is the drive plugged in?\n\n"
            f"{detail}\n\nThey are kept safe on this computer and copied the next time Arborist starts.",
            QMessageBox.Retry | QMessageBox.Close | QMessageBox.Cancel, QMessageBox.Retry)
        if r == QMessageBox.Retry:
            return self._storage_sync(ask=True)
        return r == QMessageBox.Close

    def _on_app_close(self):
        """Save the project, then copy everything to the USB stick. False = stay open."""
        try:
            self._autosave()
        except Exception:
            pass
        return self._storage_sync(ask=True)

    # napari's own File > Open and drag-and-drop: route our file types to Arborist
    def _install_open_hook(self):
        try:
            qv = self.viewer.window._qt_viewer
            orig = qv._qt_open
        except Exception:
            return

        def _qt_open(filenames, stack=False, choose_plugin=False, plugin=None, layer_type=None, **kw):
            rest = []
            for f in filenames:
                f = str(f)
                ext = os.path.splitext(f)[1].lower()
                if ext in (".czi", ".tif", ".tiff"):
                    role = self._ask_role(f)
                    if role:
                        self._guard(self.add_stack, f, role)
                elif ext == ".swc":
                    e = self.cell_entry()
                    if e:
                        self._guard(self._load_swc, e, f)
                    else:
                        self.msg("Open the cell stack before its SWC file.", error=True)
                elif ext == ".json":
                    self._guard(self._load_project, f)
                else:
                    rest.append(f)
            if rest:
                orig(rest, stack=stack, choose_plugin=choose_plugin, plugin=plugin, layer_type=layer_type, **kw)

        qv._qt_open = _qt_open

    def _ask_role(self, path):
        box = QMessageBox(self)
        box.setWindowTitle("Open z-stack")
        box.setText(f"Open {os.path.basename(path)} as:")
        b_cell = box.addButton("Cell stack (high-mag)", QMessageBox.AcceptRole)
        b_low = box.addButton("Low-mag stack (CR border)", QMessageBox.AcceptRole)
        b_ank = box.addButton("Ankyrin-G stack (AIS)", QMessageBox.AcceptRole)
        b_ov = box.addButton("Whole section (10x)", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        n = os.path.basename(path).lower()
        box.setDefaultButton(b_ank if "ankyrin" in n or "ankg" in n else b_low if self.cell_entry() else b_cell)
        box.exec()
        c = box.clickedButton()
        return {b_cell: "cell", b_low: "lowmag", b_ank: "ankg", b_ov: "overview"}.get(c)

    def open_stacks(self, role):
        paths, _ = QFileDialog.getOpenFileNames(self, "Open z-stack", "", "Z-stacks (*.czi *.tif *.tiff)")
        for p in paths:
            self._guard(self.add_stack, p, role)

    def add_stack(self, path, role, voxel=None):
        self.end_preview()
        if role == "cell" and any(e.role == "cell" for e in self.entries):
            old = next(e for e in self.entries if e.role == "cell")
            if QMessageBox.question(self, "Replace cell stack?",
                                    f"Replace cell stack '{old.name}'? (one cell per project)") != QMessageBox.Yes:
                return None
            self._remove(old)
        if role == "cell" and not getattr(self, "_loading", False):
            # another cell: its project is its own file - never keep saving into the previous cell's project
            keep = self.project_path and os.path.normcase(os.path.abspath(self.project_path)) == \
                os.path.normcase(os.path.abspath(dbm.project_path_for(path)))
            if not keep:
                self.project_path = None
        self.msg(f"Loading {os.path.basename(path)}…")
        QApplication.processEvents()
        st = load_stack(path)
        if voxel is not None:
            st.voxel = tuple(voxel)
        e = StackEntry(st, role)
        pre = f"{ROLE_TAG[role][0]} {st.name}"
        kw = {}
        data = st.data
        defer = is_lazy(st.data) and role != "cell" and self.cell_entry() is not None
        if is_lazy(st.data):     # big stack read plane by plane: give napari a display range
            if defer:
                # not shown yet (20x / ankyrin while the cell is open): read nothing now - brightness
                # and the first plane come when its view is first chosen
                mx = float(np.iinfo(st.data.dtype).max) if np.issubdtype(st.data.dtype, np.integer) else 1.0
                cl = [[0.0, mx]] * st.n_channels
                kw["visible"] = False
            else:
                cl = contrast_limits(st.data)
            kw["contrast_limits"] = cl if st.n_channels > 1 else cl[0]
            data = self._pyramid(st.data)
            if defer:
                # napari slices (and makes thumbnails of) every layer, even hidden ones: give it an
                # all-zero stand-in of the same size until the view is first shown
                import dask.array as da

                def blank(a):
                    return da.zeros(a.shape, dtype=a.dtype, chunks=(1, 1) + tuple(a.shape[-2:]))
                e._real_data = data
                data = [blank(a) for a in data] if isinstance(data, list) else blank(data)
            if isinstance(data, list):
                kw["multiscale"] = True
        imgs = self.viewer.add_image(
            data, channel_axis=0, scale=st.voxel, blending="additive",
            name=[f"{pre} | {c}" for c in st.channel_names],
            colormap=CMAPS[:st.n_channels] if st.n_channels > 1 else "gray", **kw)
        e.images = imgs if isinstance(imgs, list) else [imgs]
        e.deferred = defer
        if role == "overview":
            return self._register(e)
        if role in ("cell", "lowmag"):
            e.layers["surface_pts"] = self.viewer.add_points(
                np.zeros((0, 3)), ndim=3, scale=st.voxel, name=f"{pre} | surface points",
                face_color="#00e5ff", border_width=0, size=self._pt_size(st, 3.0))
            e.layers["surface_line"] = self._add_lines(e, "surface (slice)", "#00e5ff")
            # the line being drawn: a small layer of its own, so a stroke does not reprocess every
            # point already marked (that is what made drawing slow once many slices were marked)
            e.layers["pen_preview"] = self.viewer.add_points(
                np.zeros((0, 3)), ndim=3, scale=st.voxel, name=f"{pre} | line being drawn",
                face_color="#ffffff", border_width=0, size=self._pt_size(st, 3.0))
            e.layers["border_line"] = self._add_lines(e, "CR border (slice)", "#ffb000")
        if role in ("lowmag", "cell"):
            e.layers["border_pts"] = self.viewer.add_points(
                np.zeros((0, 3)), ndim=3, scale=st.voxel, name=f"{pre} | CR border points",
                face_color="#ffb000", border_width=0, size=self._pt_size(st, 3.0))
        if role == "ankg":
            e.layers["ais_pts"] = self.viewer.add_points(
                np.zeros((0, 3)), ndim=3, scale=st.voxel, name=f"{pre} | AIS points",
                face_color="#ff00ff", border_width=0, size=self._pt_size(st, 2.0), symbol="diamond")
            e.layers["ais_path"] = self.viewer.add_points(
                np.zeros((0, 3)), ndim=3, scale=st.voxel, name=f"{pre} | AIS path (follows staining)",
                face_color="#ff7ff0", border_width=0, size=self._pt_size(st, 0.8))
            e.layers["route_path"] = self.viewer.add_points(
                np.zeros((0, 3)), ndim=3, scale=st.voxel, name=f"{pre} | soma→AIS route (follows fill)",
                face_color="#7fffd4", border_width=0, size=self._pt_size(st, 0.6))
            e.layers["route_pts"] = self.viewer.add_points(
                np.zeros((0, 3)), ndim=3, scale=st.voxel, name=f"{pre} | route bend points",
                face_color="#00ffff", border_width=0, size=self._pt_size(st, 2.0), symbol="cross")
            e.layers["soma_pt"] = self.viewer.add_points(
                np.zeros((0, 3)), ndim=3, scale=st.voxel, name=f"{pre} | soma exit point",
                face_color="#dc143c", border_width=0, size=self._pt_size(st, 4.0))
            e.layers["branch_pt"] = self.viewer.add_points(
                np.zeros((0, 3)), ndim=3, scale=st.voxel, name=f"{pre} | axon branch point",
                face_color="#ffff00", border_width=0, size=self._pt_size(st, 3.0), symbol="square")
        elif role == "cell":
            e.layers["ais_pts"] = self.viewer.add_points(
                np.zeros((0, 3)), ndim=3, scale=st.voxel, name=f"{pre} | AIS points",
                face_color="#ff00ff", border_width=0, size=self._pt_size(st, 3.0), symbol="diamond")
            e.layers["trace"] = self.viewer.add_points(
                np.zeros((0, 3)), ndim=3, scale=st.voxel, name=f"{pre} | neuron (SWC)", border_width=0,
                size=self._pt_size(st, 1.5), face_color="#28a046", blending="translucent_no_depth")
        for l in e.layers.values():
            l.editable = l.name.endswith("points") or l.name.endswith("point")
        if role in ("cell", "lowmag"):   # rebuild the surface + save whenever marks change
            for k in ("surface_pts", "border_pts"):
                e.layers[k].events.data.connect(lambda *_a, e=e, k=k: self._marks_changed(e, k))
        if role == "ankg":          # live result + autosave whenever a mark changes
            for k in ("ais_pts", "soma_pt", "branch_pt", "route_pts"):
                e.layers[k].events.data.connect(self._ais_marks_changed)
        return self._register(e)

    def _register(self, e):
        st = e.stack
        self.entries.append(e)
        if getattr(e, "deferred", False) and self.active is not None:
            # a 20x / ankyrin stack opened with the cell: keep the cell on screen and read nothing
            # from this one until its view is chosen
            for l in e.all_layers():
                if l in self.viewer.layers:
                    self.viewer.layers.remove(l)
            self._refresh_view_controls()
        else:
            self.set_active(e)
        self._refresh_stack_table()
        warn = "" if st.calibrated else "  ⚠ no calibration found - set the voxel size in tab 1!"
        self.msg(f"Loaded {st.name}: {st.data.shape[1]} slices, voxel {np.round(st.voxel, 4)} µm.{warn}",
                 error=not st.calibrated)
        return e

    # ------------------------------------------------------------ views
    # ------------------------------------------------------------ cell preview (Database click)
    def preview_cell(self, row):
        """Z-projection of the reconstruction (or, if not done yet, of the stitch) on the canvas."""
        import threading
        self._pv_token = getattr(self, "_pv_token", 0) + 1
        token = self._pv_token
        name = row.get("cell_name") or os.path.basename(row["cell"])
        hit = dbm.cached_projection(row)
        if hit:
            self._show_preview(hit, name, dbm.projection_source(row)[0]); return
        kind = dbm.projection_source(row)[0]
        if kind is None:
            self.msg("No image found for this cell.", error=True); return
        box = {"p": None, "z": 0, "Z": 0, "done": False}

        def work():
            try:
                box["p"], _ = dbm.make_projection(row, progress=lambda z, Z: box.update(z=z, Z=Z))
            except Exception as ex:
                box["err"] = str(ex)
            box["done"] = True
        threading.Thread(target=work, daemon=True).start()
        from qtpy.QtCore import QTimer
        t = QTimer(self)

        def poll():
            if token != self._pv_token:          # another cell was clicked meanwhile
                t.stop(); return
            if not box["done"]:
                what = {"fills": "fills", "recon": "reconstruction"}.get(kind, "stitch")
                prog = f" {box['z']}/{box['Z']} slices" if box["Z"] else ""
                self.msg(f"Making a z-projection of the {what} of {name}…{prog} (first time only)")
                return
            t.stop()
            if box.get("err") or not box["p"]:
                self.msg("Could not make the projection: " + box.get("err", "no image"), error=True)
            else:
                self._show_preview(box["p"], name, kind)
        t.timeout.connect(poll); t.start(250)

    def _show_preview(self, png, name, kind):
        from PIL import Image
        rgb = np.asarray(Image.open(png).convert("RGB"))
        v = self.viewer
        if getattr(self, "_pv_layer", None) is None:
            # set aside whatever is on the canvas; it comes back when a cell is opened / a view chosen
            self._pv_parked = [l for l in v.layers]
            for l in list(v.layers):
                v.layers.remove(l)
        else:
            try:
                v.layers.remove(self._pv_layer)
            except ValueError:
                pass
        what = {"fills": "fills (soma grey, dendrites red, axon green)", "recon": "reconstruction"}.get(
            kind, "stitch (no reconstruction yet)")
        self._pv_layer = v.add_image(rgb, rgb=True, name=f"PREVIEW {name} | z-projection of the {what}")
        v.reset_view()
        self.msg(f"{name}: z-projection of the {what}. Double-click the row to open the cell.")

    def end_preview(self, restore=True):
        """Leave the preview; restore=False when another cell is being opened (the set-aside layers
        belong to the cell that is being closed and must not come back)."""
        lay = getattr(self, "_pv_layer", None)
        if lay is None:
            return
        if not restore:
            self._pv_parked = []
        v = self.viewer
        try:
            v.layers.remove(lay)
        except ValueError:
            pass
        for l in getattr(self, "_pv_parked", []):
            if l not in v.layers:
                v.layers.append(l)
        self._pv_layer, self._pv_parked = None, []
        self._pv_token = getattr(self, "_pv_token", 0) + 1

    def set_active(self, e):
        """Show one stack: only its layers are in the layer list (left), the others are parked."""
        self.end_preview()
        if e is not None and getattr(e, "deferred", False):
            e.deferred = False
            real = getattr(e, "_real_data", None)
            if real is not None:
                for c, im in enumerate(e.images):     # swap the stand-in for the real planes
                    im.data = [lvl[c] for lvl in real] if isinstance(real, list) else real[c]
                e._real_data = None
            try:
                cl = contrast_limits(e.stack.data)
                for im, c in zip(e.images, cl):
                    im.contrast_limits = c
            except Exception as ex:
                print("contrast:", ex)
            for im in e.images:
                im.visible = True
        self.active = e
        if e is not None:
            # add the new view's layers first, then park the others (the viewer is never empty,
            # which otherwise leaves stale slider widgets behind)
            for l in e.all_layers():
                if l not in self.viewer.layers:
                    self.viewer.layers.append(l)
            for x in self.entries:
                if x is e:
                    continue
                for l in x.all_layers():
                    if l in self.viewer.layers:
                        self.viewer.layers.remove(l)
            if e.images:
                self.viewer.layers.selection.active = e.images[0]
        self._refresh_view_controls()
        self._refresh_ais_tab()
        if e is None:
            return
        self._on_ndisplay()
        self.sp_vz.setValue(e.voxel[0]); self.sp_vy.setValue(e.voxel[1]); self.sp_vx.setValue(e.voxel[2])
        self._zoom_to(e)
        z = e.stack.data.shape[1]
        try:
            self.viewer.dims.set_point(0, (z // 2) * e.voxel[0])
        except Exception:
            pass
        self._load_surface_params()
        self._refresh_display_channels()
        self._update_slice_lines()

    def _park(self, e, layer):
        """A layer just created for a stack that is not on screen: take it out of the layer list."""
        if e is not self.active and layer in self.viewer.layers:
            self.viewer.layers.remove(layer)

    def _refresh_view_controls(self):
        if hasattr(self, "cmb_radii_ch"):
            c_ = self.cell_entry()
            names = ["fill channel: automatic"] + ([str(x) for x in c_.stack.channel_names] if c_ else [])
            if [self.cmb_radii_ch.itemText(i) for i in range(self.cmb_radii_ch.count())] != names:
                self.cmb_radii_ch.clear(); self.cmb_radii_ch.addItems(names)
        if not hasattr(self, "view_btns"):
            return
        for role, b_ in self.view_btns.items():
            has = any(x.role == role for x in self.entries) or (role == "overview" and bool(self.pending_overview))
            b_.setEnabled(has)
            b_.setChecked(bool(self.active and self.active.role == role))
        same = [x for x in self.entries if self.active and x.role == self.active.role]
        self.cmb_active.blockSignals(True)
        self.cmb_active.clear()
        for x in same:
            self.cmb_active.addItem(x.name, id(x))
        if self.active in same:
            self.cmb_active.setCurrentIndex(same.index(self.active))
        self.cmb_active.blockSignals(False)
        multi = len(same) > 1
        self.cmb_active.setVisible(multi); self.lbl_stack.setVisible(multi)

    def show_role(self, role):
        if role == "overview" and not any(x.role == "overview" for x in self.entries) and self.pending_overview:
            paths, self.pending_overview = self.pending_overview, []
            for p in paths[:1]:
                self._guard(self.add_stack, p, "overview")
            return
        cands = [x for x in self.entries if x.role == role]
        if cands:
            self.set_active(cands[0] if self.active not in cands else self.active)
        else:
            self._refresh_view_controls()

    def _pt_size(self, st, um):
        return max(1.0, um / st.voxel[2])

    def _add_lines(self, e, label, color):
        pre = f"{ROLE_TAG[e.role][0]} {e.name}"
        lay = self.viewer.add_vectors(np.zeros((0, 2, 3)), ndim=3, scale=e.voxel, name=f"{pre} | {label}",
                                      edge_color=color, edge_width=max(1.0, 0.6 / e.voxel[2]),
                                      vector_style="line", opacity=0.9)
        return lay

    def _refresh_stack_table(self):
        self.tbl_stacks.setRowCount(len(self.entries))
        for i, e in enumerate(self.entries):
            for j, txt in enumerate([e.name, e.role, ", ".join(f"{v:.4g}" for v in e.voxel),
                                     ", ".join(e.stack.channel_names)]):
                self.tbl_stacks.setItem(i, j, QTableWidgetItem(txt))
        self._refresh_border_table()

    def _on_active_changed(self, idx):
        key = self.cmb_active.itemData(idx)
        e = next((x for x in self.entries if id(x) == key), None)
        if e is not None and e is not self.active:
            self.set_active(e)

    def _zoom_to(self, e):
        """Fit the view to one stack (plain reset_view would fit all stacks, even hidden ones)."""
        try:
            self.viewer.reset_view(layers=e.images)
        except TypeError:          # older napari
            self.viewer.reset_view()

    def apply_voxel(self):
        e = self.active
        if not e:
            return
        old = np.array(e.voxel)
        new = np.array([self.sp_vz.value(), self.sp_vy.value(), self.sp_vx.value()])
        # marked points stay on their voxels; the SWC is already in um and stays put
        e.stack.voxel = tuple(new)
        e.stack.calibrated = True
        for l in e.all_layers():
            l.scale = new
        if e.surface is not None:
            self.build_surface()
        self._refresh_stack_table()
        self.msg(f"Voxel size set to {new} µm.")

    def remove_active(self):
        if self.active:
            self._remove(self.active)

    def _remove(self, e):
        parked = getattr(self, "_pv_parked", [])
        for l in e.all_layers():
            if l in self.viewer.layers:
                self.viewer.layers.remove(l)
            if l in parked:
                parked.remove(l)
        self.entries.remove(e)
        if self.active is e:
            self.active = None
            nxt = self.cell_entry() or (self.entries[0] if self.entries else None)
            self.set_active(nxt) if nxt else self._refresh_view_controls()
        else:
            self._refresh_view_controls()
        self._refresh_stack_table()

    def cell_entry(self):
        return next((e for e in self.entries if e.role == "cell"), None)

    # ================================================================ tab 2
    def _tab_surface(self):
        w = QWidget(); L = QVBoxLayout(w)
        L.addWidget(_lbl("Mark the brain (pial) surface on every 5th–10th slice: click points, or with "
                           "<b>pen</b> on drag along the surface. Select points + Delete to remove them."))
        self.btn_mark_surf = _btn("✎ Mark surface", self._toggle_mark_surface, checkable=True)
        L.addWidget(self.btn_mark_surf)
        h = QHBoxLayout()
        self.cb_pen = QCheckBox("✍ Pen: drag to draw a line")
        self.cb_pen.setChecked(True)
        self.cb_pen.toggled.connect(self._pen_toggled)
        self.sp_pen = _dspin(5.0, 0.5, 100, 1, 1, " µm")
        self.sp_pen.setToolTip("Spacing of the points placed along a drawn line")
        h.addWidget(self.cb_pen); h.addWidget(QLabel("every")); h.addWidget(self.sp_pen)
        L.addLayout(h)
        L.addWidget(_btn("↶ Undo last line / point", self.undo_mark))
        self.lbl_saved = _lbl("")
        h = QHBoxLayout()
        h.addWidget(_btn("Import Fiji point table…", self.import_fiji_points,
                         tip="Old SURFACE.txt from the Fiji point tool (index, X, Y, Slice)"))
        h.addWidget(_btn("Export points…", self.export_surface_points))
        h.addWidget(_btn("Clear this slice", lambda: self.clear_slice("surface_pts"),
                         tip="Delete the surface marks on the slice shown, to redraw it"))
        h.addWidget(_btn("🗑 Clear surface", self.clear_surface_points,
                         tip="Delete all surface marks of this image and the built surface"))
        L.addLayout(h)
        g = QGroupBox("Mesh"); f = QFormLayout(g)
        self.cmb_method = QComboBox()
        self.cmb_method.addItems(["Follow slices (any curvature)", "Height field (gently curved only)",
                                  "Delaunay (raw points, old notebook)"])
        self.sp_smooth = _dspin(10, 0, 1000, 1, 1, " µm")
        self.sp_smooth.setToolTip("Smooths the drawn line along the surface (larger = smoother)")
        self.sp_smooth.setToolTip("Higher = smoother surface, less sensitive to click noise")
        self.sp_grid = _dspin(2, 0.1, 100, 0.5, 1, " µm")
        self.sp_grid.setToolTip("Spacing of the mesh points along each slice line")
        self.sp_extend = _dspin(20, 0, 1000, 5, 0, " µm")
        self.sp_extend.setToolTip("Extrapolate the smooth surface this far beyond the clicked patch")
        f.addRow("Method", self.cmb_method); f.addRow("Smoothing", self.sp_smooth)
        f.addRow("Point spacing", self.sp_grid); f.addRow("Extend beyond ends", self.sp_extend)
        L.addWidget(g)
        self.btn_build = _btn("Build surface", lambda: self._guard(self._build_and_save),
                              tip="Draw the surface line on as many slices as you like, then build once.")
        L.addWidget(self.btn_build)
        self.cb_autobuild = QCheckBox("Rebuild after every line (slow on big files)")
        self.cb_autobuild.setChecked(False)
        L.addWidget(self.cb_autobuild)
        L.addWidget(_btn("Flip brain side (if depths come out negative)", lambda: self._guard(self.flip_surface)))
        L.addWidget(_btn("Save surface mesh (STL)…", self.save_surface_stl))
        self.lbl_surface = QLabel("No surface yet."); self.lbl_surface.setWordWrap(True)
        L.addWidget(self.lbl_surface)
        L.addWidget(self.lbl_saved)
        L.addWidget(_lbl("<i>The surface is rebuilt and saved next to the cell automatically while you mark "
                         "(project, &lt;cell&gt;_surface_points.csv, &lt;cell&gt;_brain_surface.stl).</i>"))
        L.addWidget(_lbl("<i>Cyan line = surface where it cuts the current slice; check it follows "
                           "the edge of the tissue. In 3D view (toggle at the bottom left) the mesh is shown.</i>"))
        L.addStretch()
        return w

    def _select_points_layer(self, key, on):
        e = self.active
        if not e or key not in e.layers:
            return
        lay = e.layers[key]
        if on:
            if self.viewer.dims.ndisplay != 2:
                self.viewer.dims.ndisplay = 2
            self.viewer.layers.selection.active = lay
            lay.mode = "pan_zoom" if self._pen_on_for(key) else "add"
        else:
            lay.mode = "pan_zoom"
        self._pen_update_pan()

    # ------------------------------------------------------------ pen drawing
    PEN_KEYS = ("surface_pts", "border_pts")

    def _marking_key(self):
        if getattr(self, "btn_mark_surf", None) is not None and self.btn_mark_surf.isChecked():
            return "surface_pts"
        if getattr(self, "btn_mark_border", None) is not None and self.btn_mark_border.isChecked():
            return "border_pts"
        return None

    def _pen_on_for(self, key):
        return key in self.PEN_KEYS and getattr(self, "cb_pen", None) is not None and self.cb_pen.isChecked()

    def _pen_toggled(self, on):
        if on and not self._marking_key() and self.tabs.currentIndex() == 1 and self.cell_entry() is not None:
            self.btn_mark_surf.setChecked(True)          # pen on in the Surface tab: start drawing the surface
        key = self._marking_key()
        if key and self.active and key in self.active.layers:
            self.active.layers[key].mode = "pan_zoom" if on else "add"
        self._pen_update_pan()

    def _camera(self):
        return self.viewer.scene.camera if hasattr(self.viewer, "scene") else self.viewer.camera

    def _pen_update_pan(self):
        """No panning while drawing with the pen (wheel zoom/scroll still work)."""
        key = self._marking_key()
        try:
            self._camera().mouse_pan = not (key and self._pen_on_for(key))
        except Exception:
            pass

    def _space_pan(self, viewer):
        """Hold Space to pan while the pen is on; release to keep drawing."""
        self._space_down = True
        try:
            self._camera().mouse_pan = True
        except Exception:
            pass
        yield
        self._space_down = False
        self._pen_update_pan()

    def _on_pen(self, viewer, event):
        if getattr(self, "_click_taken", False):
            self._click_taken = False
            return
        if getattr(self, "_space_down", False):
            return
        key = self._marking_key()
        if not key and getattr(self, "cb_pen", None) is not None and self.cb_pen.isChecked() \
                and event.type == "mouse_press" and self.tabs.currentIndex() in (1, 2) \
                and not getattr(self, "_pen_hint_shown", False):
            self._pen_hint_shown = True       # say it once; after that a drag simply pans
            self.msg("To draw, first press ✎ Surface points (or ✎ CR border points) - it turns orange.", error=True)
            return
        if not key or not self._pen_on_for(key) or viewer.dims.ndisplay != 2:
            return
        if event.type != "mouse_press" or event.button != 1:
            return
        e = self.active
        if e is None or key not in e.layers:
            return
        try:
            self._camera().mouse_pan = False            # whatever napari set, a pen stroke never pans
        except Exception:
            pass
        import time as _time
        lay = e.layers[key]
        scale = np.asarray(lay.scale, float)
        start_n = len(lay.data)
        pending = []
        stroke = []
        flushed = [_time.perf_counter()]
        prev = e.layers.get("pen_preview")
        if prev is not None:
            prev.face_color = lay.face_color[0] if len(lay.face_color) else "#ffffff"
            prev.current_face_color = prev.face_color[0] if len(prev.face_color) else "#ffffff"

        def flush(force=False):
            """Show the stroke in batches (~25 per second) on the small preview layer; the marks layer
            itself gets the whole line once, on release."""
            if not pending or (not force and _time.perf_counter() - flushed[0] < 0.04):
                return
            stroke.extend(pending); pending.clear()
            if prev is not None:
                prev.data = np.array(stroke)
            flushed[0] = _time.perf_counter()

        def commit():
            flush(force=True)
            if stroke:
                V = np.array(stroke)
                lay.data = np.vstack([lay.data, V]) if len(lay.data) else V
            if prev is not None:
                prev.data = np.zeros((0, 3))
        zi = float(round(float(viewer.dims.point[0]) / scale[0]))
        last = None
        step = self.sp_pen.value()

        def put(p):
            v = p / scale
            v[0] = zi
            pending.append(v)
            flush(force=True)

        def add(pos_world):
            """Place points every `step` um along the stroke (interpolated between mouse events)."""
            nonlocal last
            p = np.asarray(pos_world, float)[-3:].copy()
            if last is None:
                last = p; put(p); return
            d = p - last; d[0] = 0
            L = float(np.linalg.norm(d))
            new = []
            while L >= step:
                last = last + d / L * step
                new.append(last.copy())
                d = p - last; d[0] = 0
                L = float(np.linalg.norm(d))
            if new:
                V = np.array(new) / scale
                V[:, 0] = zi
                pending.extend(V)
                flush()

        add(event.position)
        yield
        while event.type == "mouse_move":
            add(event.position)
            yield
        add(event.position) if last is not None and np.linalg.norm(
            (np.asarray(event.position, float)[-3:] - last)[1:]) > 0.3 * step else None
        commit()
        hist = self.__dict__.setdefault("_pen_hist", {})
        hist.setdefault(id(lay), []).append(start_n)
        self.msg(f"Line drawn: {len(lay.data) - start_n} points on slice {int(zi) + 1}.")

    def _marks_changed(self, e, key):
        from qtpy.QtCore import QTimer
        if getattr(self, "_loading", False):
            return
        self.__dict__.setdefault("_dirty", set()).add((id(e), key))
        if not hasattr(self, "_marks_timer"):
            self._marks_timer = QTimer(self); self._marks_timer.setSingleShot(True)
            self._marks_timer.timeout.connect(self._apply_mark_changes)
        self._marks_timer.start(900)

    def _apply_mark_changes(self):
        dirty, self._dirty = self.__dict__.get("_dirty", set()), set()
        for e in list(self.entries):
            if (id(e), "surface_pts") in dirty:
                pts = self.surface_xyz(e)
                try:
                    if len(pts) >= 2 and not self.cb_autobuild.isChecked():
                        self._set_surface_stale(e, True)      # points saved; surface built on request
                    elif len(pts) >= 2:
                        self.build_surface(e)
                    else:
                        self._remove_surface(e)
                        self._delete_saved_surface_files(e)
                        self._update_border_surface()
                except Exception as ex:
                    self._remove_surface(e)
                    self.lbl_surface.setText(f"Surface not built yet: {ex}")
        self._autosave()

    @staticmethod
    def _fold_ratio(surf, pts):
        """Length of the surface cut on the marked slices / length of the drawn lines there (≈1 if fine)."""
        cut = drawn = 0.0
        for z in np.unique(np.round(pts[:, 2], 3))[:12]:
            P = pts[np.abs(pts[:, 2] - z) < 1e-3, :2]
            if len(P) < 3:
                continue
            from scipy.sparse.csgraph import minimum_spanning_tree
            from scipy.spatial.distance import pdist, squareform
            P = P[:: max(1, len(P) // 600)]
            drawn += minimum_spanning_tree(squareform(pdist(P))).sum()
            segs = surf.slice_segments(float(z))[0]
            cut += np.linalg.norm(segs[:, 1, :2] - segs[:, 0, :2], axis=1).sum() if len(segs) else 0
        return cut / drawn if drawn > 0 else 1.0

    def _build_and_save(self):
        e = self.active
        if e is not None and e.role not in ("cell", "lowmag"):
            e = self.cell_entry()
        if e is None or len(self.surface_xyz(e)) < 2:
            self.msg("Draw the surface line on at least one slice first.", error=True); return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.build_surface(e)
        finally:
            QApplication.restoreOverrideCursor()
        self._autosave()

    def _set_surface_stale(self, e, stale):
        e.surface_stale = stale
        if not hasattr(self, "btn_build"):
            return
        if stale:
            n_sl = len(np.unique(np.round(e.layers["surface_pts"].data[:, 0]))) if "surface_pts" in e.layers else 0
            self.btn_build.setStyleSheet("background:#d35400; color:white; font-weight:bold;")
            self.btn_build.setText(f"Build surface  (lines on {n_sl} slice{'s' if n_sl != 1 else ''})")
            self.lbl_surface.setText("<b>Surface not updated yet</b> - keep drawing on other slices, then press "
                                     "<b>Build surface</b>." + (" The lines shown are from the last build."
                                                                if e.surface is not None else ""))
        else:
            self.btn_build.setStyleSheet("")
            self.btn_build.setText("Build surface")

    def _autosave(self):
        """Project + surface files next to the 63x stack."""
        c = self.cell_entry()
        if c is None:
            return
        try:
            self._save_project(self.project_path or dbm.project_path_for(c.stack.path), quiet=True)
            folder, stem = os.path.dirname(c.stack.path), c.name
            # the surface files are rewritten only when the points / the surface changed (not after every
            # AIS or CR mark)
            saved = self.__dict__.setdefault("_surf_saved", {})
            pts = self.surface_xyz(c)
            key_p = (c.stack.path, len(pts), float(pts.sum()) if len(pts) else 0.0)
            if len(pts) and saved.get("pts") != key_p:
                np.savetxt(storage.out(os.path.join(folder, f"{stem}_surface_points.csv")), pts,
                           delimiter=",", header="x_um,y_um,z_um", comments="", fmt="%.3f")
                saved["pts"] = key_p
            if c.surface is not None and not getattr(c, "surface_stale", False):
                V = c.surface.mesh.vertices
                key_m = (c.stack.path, len(V), float(np.asarray(V).sum()), len(c.surface.mesh.faces),
                         int(np.asarray(c.surface.mesh.faces[0]).tolist()[1]) if len(c.surface.mesh.faces) else 0)
                if saved.get("mesh") != key_m:
                    c.surface.mesh.export(storage.out(os.path.join(folder, f"{stem}_brain_surface.stl")))
                    saved["mesh"] = key_m
                # (the CR border surface STL is written by Analyze)
            import time
            local = storage.store().is_slow(c.stack.path)
            self.lbl_saved.setText(f"<i>Saved {time.strftime('%H:%M:%S')} "
                                   + ("on this computer (copied to the drive when closing)" if local else "next to the cell")
                                   + f": project, surface points{', mesh' if c.surface is not None else ''}</i>")
            self._storage_status()
        except Exception as ex:
            self.lbl_saved.setText(f"<span style='color:#f88'>Autosave failed: {ex}</span>")

    def undo_mark(self):
        key = self._marking_key() or "surface_pts"
        e = self.active
        if e is None or key not in e.layers:
            return
        lay = e.layers[key]
        hist = self.__dict__.setdefault("_pen_hist", {}).get(id(lay), [])
        while hist and hist[-1] >= len(lay.data):
            hist.pop()
        n = hist.pop() if hist else max(0, len(lay.data) - 1)
        lay.data = lay.data[:n] if n else np.zeros((0, 3))

    def ankg_entry(self):
        return next((e for e in self.entries if e.role == "ankg"), None)

    def _mark_ais_layer(self, key, on):
        btns = {"ais_pts": self.btn_mark_ais, "soma_pt": self.btn_mark_soma, "branch_pt": self.btn_mark_branch,
                "route_pts": self.btn_mark_route}
        if on:
            for k, b_ in btns.items():
                if k != key:
                    b_.setChecked(False)
            self.btn_mark_surf.setChecked(False)
            # AIS goes on the AnkG stack if there is one (or the active cell stack); soma/branch only on AnkG
            target = self.active if (self.active and key in self.active.layers) else (
                self.ankg_entry() or (self.cell_entry() if key == "ais_pts" else None))
            if target is None or key not in target.layers:
                self.msg("Open the ankyrin-G stack first (the soma exit and branch points are marked there).", error=True)
                btns[key].setChecked(False); return
            if self.active is not target:
                self.set_active(target)
            if key in ("soma_pt", "branch_pt"):
                target.layers[key].data = np.zeros((0, 3))     # one point each
            self.msg({"ais_pts": "Click along the AIS from its soma-side start to its end.",
                      "soma_pt": "Click where the axon-bearing process leaves the soma.",
                      "branch_pt": "Click where the axon leaves the dendrite.",
                      "route_pts": "Click on the process where the soma→AIS route bends."}[key])
        self._select_points_layer(key, on)
        lay_ = self.active.layers.get(key) if self.active else None
        if on and key in ("soma_pt", "branch_pt"):
            lay = self.active.layers[key]

            def one_point(event=None, lay=lay, key=key):
                if len(lay.data) >= 1:
                    btns[key].setChecked(False)
            lay.events.data.connect(one_point)
        if not on:
            self._ais_preview()

    def clear_ais(self):
        e = self.active if self.active and "ais_pts" in self.active.layers else (self.ankg_entry() or self.cell_entry())
        if e:
            for k in ("ais_pts", "soma_pt", "branch_pt", "ais_path", "route_pts", "route_path"):
                if k in e.layers:
                    e.layers[k].data = np.zeros((0, 3))
            self._ais_cache = None
            self._ais_preview()

    def _ais_points(self, e):
        """AIS start/end marked on the cell stack (snapped onto the SWC)."""
        d = e.layers["ais_pts"].data
        return vox2xyz(np.array([d[0], d[-1]]), e.voxel) if len(d) >= 2 else None

    def current_ais(self):
        """(features dict, source text) from the best available AIS source."""
        from .morphometry import ais_features, ais_features_from_points
        a_e = self.ankg_entry()
        if a_e is not None and len(a_e.layers["ais_pts"].data) >= 2:
            pts = vox2xyz(a_e.layers["ais_pts"].data, a_e.voxel)
            soma = vox2xyz(a_e.layers["soma_pt"].data, a_e.voxel)
            br = vox2xyz(a_e.layers["branch_pt"].data, a_e.voxel)
            follow = self.cb_ais_follow.isChecked()
            path = self._ais_path(a_e) if follow else None
            a_e.layers["ais_path"].data = xyz2vox(path, a_e.voxel) if path is not None else np.zeros((0, 3))
            origin = self.cmb_ais_origin.currentText()
            r_ais = r_br = None
            a_e.layers["route_path"].data = np.zeros((0, 3))
            if follow and len(soma):
                rt = self._route(a_e, origin)
                if rt is not None:
                    rpath, r_ais, r_br = rt
                    a_e.layers["route_path"].data = rpath
            f = ais_features_from_points(pts, soma if len(soma) else None, br if len(br) else None,
                                         origin, path_xyz=path, route_to_ais_um=r_ais, route_to_branch_um=r_br)
            if not len(soma):
                f["ais_note"] = "mark where the axon-bearing process leaves the soma (step 2) to get distances"
            return f, "AnkG stack"
        e = self.cell_entry()
        if e is None or not len(e.neuron) or e.surface is None:
            return {}, ""
        f = ais_features(e.neuron, e.surface, self.sp_ais_type.value(), self._ais_points(e),
                         self._types(self.ed_dend.text()), self._types(self.ed_axon.text()))
        return f, ("marked on cell stack" if self._ais_points(e) is not None else "SWC type")

    def _route(self, a_e, origin):
        """Soma exit -> (bends, branch point) -> AIS start along the fill channel.
        Returns (path in voxels, length to AIS start um, length to branch point um or None)."""
        from .morphometry import order_route, route_along_signal
        ex = np.asarray(a_e.layers["soma_pt"].data, float)[0]
        ais = np.asarray(a_e.layers["ais_pts"].data, float)
        br = np.asarray(a_e.layers["branch_pt"].data, float)
        bends = [p for p in np.asarray(a_e.layers["route_pts"].data, float)]
        vz = np.asarray(a_e.voxel)
        ref = br[0] if (origin == "dendrite" and len(br)) else ex
        start = ais[0] if np.linalg.norm((ais[0] - ref) * vz) <= np.linalg.norm((ais[-1] - ref) * vz) else ais[-1]
        middle = bends + ([br[0]] if (origin == "dendrite" and len(br)) else [])
        nodes = order_route(ex, middle, start)
        ch = max(0, self.cmb_route_ch.currentIndex())
        key = (id(a_e), ch, nodes.round(3).tobytes())
        if getattr(self, "_route_cache", None) and self._route_cache[0] == key:
            return self._route_cache[1]
        try:
            path, cum = route_along_signal(a_e.stack.data[ch], nodes, a_e.voxel)
            r_br = None
            if origin == "dendrite" and len(br):
                i = int(np.argmin(np.linalg.norm(nodes - br[0], axis=1)))
                r_br = float(cum[i])
            res = (path, float(cum[-1]), r_br)
        except Exception as ex_:
            print("route failed:", ex_)
            res = None
        self._route_cache = (key, res)
        return res

    def _ais_marks_changed(self, *_):
        """Debounced: refresh the result and save the project."""
        from qtpy.QtCore import QTimer
        if getattr(self, "_loading", False):
            return
        if not hasattr(self, "_ais_timer"):
            self._ais_timer = QTimer(self); self._ais_timer.setSingleShot(True)
            self._ais_timer.timeout.connect(self._ais_update_and_save)
        self._ais_timer.start(700)

    def _ais_update_and_save(self):
        self._ais_preview()
        a_e, c = self.ankg_entry(), self.cell_entry()
        target = c or a_e
        if target is None:
            return
        path = self.project_path or dbm.project_path_for(target.stack.path)
        try:
            self._save_project(path, quiet=True)
            import time
            self.lbl_ais_saved.setText(f"<i>AIS marks saved {time.strftime('%H:%M:%S')} → {os.path.basename(path)}</i>")
        except Exception as ex:
            self.lbl_ais_saved.setText(f"<span style='color:#f88'>Could not save: {ex}</span>")

    def _ais_path(self, a_e):
        """AIS path following the ankyrin signal (cached for the same clicks/channel)."""
        from .morphometry import follow_signal
        d = np.asarray(a_e.layers["ais_pts"].data, float)
        ch = max(0, self.cmb_ais_ch.currentIndex())
        key = (id(a_e), ch, d.round(3).tobytes())
        if getattr(self, "_ais_cache", None) and self._ais_cache[0] == key:
            return self._ais_cache[1]
        try:
            pv = follow_signal(a_e.stack.data[ch], d, a_e.voxel)
            path = vox2xyz(pv, a_e.voxel)
        except Exception as ex:
            print("AIS path failed:", ex)
            path = None
        self._ais_cache = (key, path)
        return path

    def _ais_preview(self):
        if not hasattr(self, "lbl_ais"):
            return
        a, src = self.current_ais()
        if not a:
            e = self.cell_entry()
            hint = ("Build the surface on the cell stack to measure an AIS from the SWC."
                    if e is not None and len(e.neuron) and e.surface is None else "No AIS marked yet.")
            self.lbl_ais.setText(hint); return
        txt = f"[{src}] AIS length <b>{a['ais_length_um']:.1f} µm</b>"
        if "ais_length_method" in a:
            txt += f" ({a['ais_length_method']}; straight start-end {a['ais_length_straight_um']:.1f} µm)"
        if "ais_start_dist_soma_path_um" in a and np.isfinite(a["ais_start_dist_soma_path_um"]):
            txt += f"<br>starts {a['ais_start_dist_soma_path_um']:.1f} µm from the soma along the neurite"
        elif "ais_start_dist_soma_exit_um" in a:
            txt += f"<br>starts {a['ais_start_dist_soma_exit_um']:.1f} µm from where the process leaves the soma"
        txt += f"<br>origin: <b>{a['ais_origin']}</b>"
        if "acd_branch_dist_soma_path_um" in a:
            txt += f" - axon leaves the dendrite {a['acd_branch_dist_soma_path_um']:.1f} µm from the soma"
        if "acd_branch_dist_soma_exit_um" in a:
            txt += (f" - axon leaves the dendrite {a['acd_branch_dist_soma_exit_um']:.1f} µm from the soma; "
                    f"AIS starts {a.get('acd_branch_to_ais_start_um', np.nan):.1f} µm after the branch point")
        if a.get("ais_note"):
            txt += f"<br><span style='color:#f88'>{a['ais_note']}</span>"
        self.lbl_ais.setText(txt)

    def _toggle_mark_surface(self, on):
        if on and hasattr(self, "btn_mark_border"):
            self.btn_mark_border.setChecked(False)
        self._select_points_layer("surface_pts", on)
        if on:
            self.msg("Drag along the surface to draw a line (pen) - hold Space and drag to move the image. "
                     "Turn the button off when done.")

    def surface_xyz(self, e):
        if "surface_pts" not in e.layers:
            return np.zeros((0, 3))
        return vox2xyz(e.layers["surface_pts"].data, e.voxel)

    def import_fiji_points(self):
        e = self.active
        if not e:
            return
        p, _ = QFileDialog.getOpenFileName(self, "Fiji point table", "", "Text (*.txt *.csv *.tsv);;All (*)")
        if not p:
            return
        try:
            xyz = read_fiji_points(p, e.voxel[0])
        except Exception as ex:
            self.msg(f"Could not read {p}: {ex}", error=True); return
        lay = e.layers["surface_pts"]
        lay.data = np.vstack([lay.data, xyz2vox(xyz, e.voxel)])
        self.msg(f"Imported {len(xyz)} surface points (Fiji Slice 1 = z 0 µm, the SNT convention).")

    def _import_surface_txt(self, e, path):
        """Fiji SURFACE.txt -> surface points of the cell stack, then build the surface."""
        try:
            xyz = read_fiji_points(path, e.voxel[0])
        except Exception as ex:
            self.msg(f"Could not read {os.path.basename(path)}: {ex}", error=True); return
        e.layers["surface_pts"].data = xyz2vox(xyz, e.voxel)
        try:
            self.build_surface(e)
        except Exception as ex:
            self.msg(f"Surface from {os.path.basename(path)} could not be built: {ex}", error=True); return
        self.msg(f"Surface read from Fiji {os.path.basename(path)} ({len(xyz)} points) - check that the cyan "
                 "line follows the tissue edge (it must have been marked on this same stack).")

    def export_surface_points(self):
        e = self.active
        if not e:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save surface points", f"{e.name}_surface_points.csv", "CSV (*.csv)")
        if p:
            np.savetxt(storage.out(p), self.surface_xyz(e), delimiter=",", header="x_um,y_um,z_um", comments="")

    def _img_label(self, e):
        return {"cell": "63x", "lowmag": "20x"}.get(e.role, "") + " " + e.name

    def _remove_surface(self, e):
        """Forget the built surface of a stack (mesh, slice lines, CR mesh on the 63x)."""
        e.surface = None
        for k in ("mesh", "border_mesh"):
            lay = e.layers.get(k)
            if lay is not None:
                if lay in self.viewer.layers:
                    self.viewer.layers.remove(lay)
                e.layers[k] = None
        for k in ("surface_line", "border_line"):
            if e.layers.get(k) is not None:
                e.layers[k].data = np.zeros((0, 2, 3))
        if e is self.active:
            self.lbl_surface.setText("No surface yet.")

    def _delete_saved_surface_files(self, e):
        if e.role != "cell":
            return
        folder, stem = os.path.dirname(e.stack.path), e.name
        for suf in ("_brain_surface.stl", "_surface_points.csv", "_CR_border_surface.stl"):
            p = os.path.join(folder, stem + suf)
            for f in (p, storage.store().local_for(p)):
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except OSError as ex:
                    print("could not delete", f, ex)

    def clear_surface_points(self):
        e = self.active
        if e is None or "surface_pts" not in e.layers:
            return
        extra = ("\n\nThe saved surface files next to the cell are deleted too." if e.role == "cell" else
                 "\n\nIts CR measurement is removed too.")
        if QMessageBox.question(self, "Clear surface",
                                f"Delete all surface marks and the built surface of {self._img_label(e)}?{extra}") \
                != QMessageBox.Yes:
            return
        prev, self._loading = getattr(self, "_loading", False), True      # no rebuild while clearing
        try:
            e.layers["surface_pts"].data = np.zeros((0, 3))
        finally:
            self._loading = prev
        self.__dict__.setdefault("_pen_hist", {}).pop(id(e.layers["surface_pts"]), None)
        self._remove_surface(e)
        if e.role == "lowmag":
            e.border_result = None
            self._refresh_border_table()
        self._delete_saved_surface_files(e)
        self._update_border_surface()
        self._autosave()
        self.msg(f"Surface of {self._img_label(e)} cleared.")

    def clear_border_points(self):
        e = self.active
        if e is None or "border_pts" not in e.layers:
            return
        if QMessageBox.question(self, "Clear CR points",
                                f"Delete all CR border points of {self._img_label(e)} and its measurement?") \
                != QMessageBox.Yes:
            return
        e.layers["border_pts"].data = np.zeros((0, 3))
        self.__dict__.setdefault("_pen_hist", {}).pop(id(e.layers["border_pts"]), None)
        e.border_result = None
        self._refresh_border_table()
        self._update_border_surface()
        self._autosave()
        self.msg(f"CR border points of {self._img_label(e)} cleared.")

    def clear_slice(self, key):
        """Delete the marks (surface or CR) on the slice currently shown."""
        e = self.active
        if e is None or key not in e.layers:
            return
        lay = e.layers[key]
        if not len(lay.data):
            return
        zi = round(float(self.viewer.dims.point[0]) / float(lay.scale[0]))
        keep = np.round(lay.data[:, 0]).astype(int) != zi
        n = int((~keep).sum())
        if n == 0:
            self.msg(f"No {'surface' if key == 'surface_pts' else 'CR'} marks on slice {zi + 1}."); return
        lay.data = lay.data[keep] if keep.any() else np.zeros((0, 3))
        self.__dict__.setdefault("_pen_hist", {}).pop(id(lay), None)
        if key == "border_pts" and e.border_result:
            e.border_result = None          # measured with the old points: measure again
            self._refresh_border_table(); self._update_border_surface()
        self.msg(f"Deleted {n} {'surface' if key == 'surface_pts' else 'CR'} marks on slice {zi + 1}.")

    METHODS = ["slices", "smooth", "delaunay"]

    def _surface_kwargs(self):
        return dict(method=self.METHODS[max(0, self.cmb_method.currentIndex())],
                    smoothing=self.sp_smooth.value(), grid_um=self.sp_grid.value(),
                    extend_um=self.sp_extend.value())

    def _set_method(self, m):
        self.cmb_method.setCurrentIndex(self.METHODS.index(m) if m in self.METHODS else 0)

    @staticmethod
    def _z_range(e):
        return (0.0, (e.stack.data.shape[1] - 1) * e.voxel[0])

    def _load_surface_params(self):
        e = self.active
        if e is None or e.surface is None:
            return
        self._set_method(e.surface.method)
        self.sp_smooth.setValue(e.surface.params["smoothing"])
        self.sp_grid.setValue(e.surface.params["grid_um"])
        self.sp_extend.setValue(e.surface.params["extend_um"])

    def _orient(self, e, surf):
        """Auto-orient: soma / CR border points / image brightness, then apply user flip."""
        ref = None
        sure = False                          # the soma / CR points are in the brain by definition
        if e.neuron is not None and len(e.neuron):
            s = e.neuron.df[e.neuron.df["type"] == 1]
            ref = (s if len(s) else e.neuron.df)[["x", "y", "z"]].to_numpy(float)
            sure = True
        elif "border_pts" in e.layers and len(e.layers["border_pts"].data):
            ref = vox2xyz(e.layers["border_pts"].data, e.voxel)
            sure = True
        if ref is None and getattr(e, "inside_ref", None) is not None:
            ref = e.inside_ref                  # worked out from the image once, reused
        if ref is not None:
            surf.orient_towards(ref)
        elif is_lazy(e.stack.data):
            # big file read plane by plane: judge brightness on the marked slices only (already in
            # memory from viewing them) instead of reading the whole stack
            ref = self._inside_from_planes(e, surf)
            if ref is not None:
                e.inside_ref = ref
                surf.orient_towards(ref)
        else:
            vol, vox = preview_volume(e.stack.data, e.voxel)
            surf.orient_by_image(vol, vox)
        if e.flipped and sure:
            # a flip made earlier against an image-based guess must not turn a soma-based answer around
            e.flipped = False
            self.msg("The neuron now decides which side is the brain (an earlier 'Flip' was undone).")
        elif e.flipped:
            surf.flip()

    def _inside_from_planes(self, e, surf, probe_um=15.0, max_planes=3):
        """A point on the brighter (tissue) side of the surface line, from a few marked slices."""
        pts = self.surface_xyz(e)
        if not len(pts):
            return None
        zi_all = np.unique(np.round(pts[:, 2] / e.voxel[0]).astype(int))
        cur = int(round(float(self.viewer.dims.point[0]) / e.voxel[0]))
        zi_all = sorted(zi_all, key=lambda z: abs(z - cur))[:max_planes]
        Y, X = e.stack.data.shape[-2:]
        st = max(1, int(np.ceil(max(Y, X) / 2048)))
        votes, refs = 0.0, []
        for zi in zi_all:
            pl = None
            for c in range(e.stack.data.shape[0]):
                a = e.stack.data[c, int(zi)]
                a = np.asarray(a.compute() if hasattr(a, "compute") else a)[::st, ::st].astype(np.float32)
                pl = a if pl is None else np.maximum(pl, a)
            segs, faces = surf.slice_segments(zi * e.voxel[0])
            if not len(segs):
                continue
            mid = segs[:, :, :2].mean(1)
            nrm = surf.mesh.face_normals[faces][:, :2]
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)

            def sample(P):
                r = np.clip((P[:, 1] / e.voxel[1] / st).astype(int), 0, pl.shape[0] - 1)
                c_ = np.clip((P[:, 0] / e.voxel[2] / st).astype(int), 0, pl.shape[1] - 1)
                inside = (P[:, 1] >= 0) & (P[:, 0] >= 0) & (P[:, 1] / e.voxel[1] < Y) & (P[:, 0] / e.voxel[2] < X)
                return pl[r, c_], inside
            a_, ia = sample(mid + probe_um * nrm)
            b_, ib = sample(mid - probe_um * nrm)
            ok = ia & ib
            if ok.sum() < 5:
                continue
            side = 1.0 if np.median(a_[ok]) >= np.median(b_[ok]) else -1.0
            votes += side
            k = len(mid) // 2
            refs.append([*(mid[k] + side * probe_um * nrm[k]), zi * e.voxel[0]])
        if not refs:
            return None
        return np.array([refs[0]])

    def build_surface(self, e=None, cover=None):
        e = e or self.active
        if e is None:
            return
        pts = self.surface_xyz(e)
        surf = BrainSurface(pts, cover_xyz=cover, z_range=self._z_range(e), **self._surface_kwargs())
        self._orient(e, surf)
        e.surface = surf
        self._set_surface_stale(e, False)
        self._show_mesh(e)
        res = surf.signed_depth(pts)
        fold = self._fold_ratio(surf, pts)
        if fold > 1.4:
            self.msg(f"The surface folds over itself (its cut is {fold:.1f}× longer than your lines). "
                     + ("Use the method 'Follow slices' for a curved surface." if surf.method != "slices"
                        else "Check for stray lines with Clear slice."), error=True)
        self.lbl_surface.setText(
            f"{len(pts)} points on {len(np.unique(np.round(pts[:, 2], 3)))} slices → "
            f"{len(surf.mesh.faces)} faces. Fit residual (RMS) {np.sqrt(np.mean(res ** 2)):.2f} µm.\n"
            f"Normal (into brain) x,y,z = {np.round(surf.n, 2)}")
        self.msg("Surface built.")
        if e.role == "cell":
            self._update_border_surface()
        return surf

    def flip_surface(self):
        e = self.active
        if not e or e.surface is None:
            return
        e.flipped = not e.flipped
        e.surface.flip()
        self._show_mesh(e)
        self._update_border_surface()
        self.msg("Brain side flipped.")

    def _show_mesh(self, e):
        m = e.surface.mesh
        data = (xyz2vox(m.vertices, e.voxel), np.asarray(m.faces))
        pre = f"{ROLE_TAG[e.role][0]} {e.name}"
        if e.layers.get("mesh") is not None:
            e.layers["mesh"].data = data
        else:
            e.layers["mesh"] = self.viewer.add_surface(data, scale=e.voxel, name=f"{pre} | surface mesh",
                                                       colormap="cyan", opacity=0.35, blending="translucent")
            self._park(e, e.layers["mesh"])
        # a mesh drawn in the 2D view shows up as broken triangles: only ever show it in 3D
        e.layers["mesh"].visible = self.viewer.dims.ndisplay == 3
        self._on_ndisplay()
        self._update_slice_lines()

    def save_surface_stl(self):
        e = self.active
        if not e or e.surface is None:
            self.msg("Build the surface first.", error=True); return
        p, _ = QFileDialog.getSaveFileName(self, "Save STL", f"{e.name}_brain_surface.stl", "STL (*.stl)")
        if p:
            e.surface.mesh.export(storage.out(p))

    # slice lines -------------------------------------------------------
    def _slice_vectors(self, mesh, e, surf=None):
        z_um = float(self.viewer.dims.point[0])
        zi = round(z_um / e.voxel[0])
        if surf is not None:
            segs = surf.slice_segments(zi * e.voxel[0])[0]
        else:
            try:
                segs = trimesh.intersections.mesh_plane(mesh, plane_normal=[0, 0, 1],
                                                        plane_origin=[0, 0, zi * e.voxel[0] + 1e-4])
            except Exception:
                segs = np.zeros((0, 2, 3))
        if len(segs):
            # a folded mesh (e.g. Delaunay on a dome) gives pieces spanning far across the slice; a real
            # cut is made of short pieces, about the mesh spacing
            L = np.linalg.norm(segs[:, 1, :2] - segs[:, 0, :2], axis=1)
            segs = segs[L <= max(25.0, 10 * self.sp_grid.value())]
        if not len(segs):
            return np.zeros((0, 2, 3))
        a = xyz2vox(segs[:, 0], e.voxel); b = xyz2vox(segs[:, 1], e.voxel)
        a[:, 0] = zi; b[:, 0] = zi
        return np.stack([a, b - a], axis=1)

    def _update_slice_lines(self, *_):
        e = self.active
        if not e or self.viewer.dims.ndisplay != 2 or "surface_line" not in e.layers:
            return
        if e.surface is not None:
            e.layers["surface_line"].data = self._slice_vectors(e.surface.mesh, e, e.surface)
        else:
            e.layers["surface_line"].data = np.zeros((0, 2, 3))
        b = self.boundary_um()
        if e.role == "cell" and e.surface is not None and b:
            zi = round(float(self.viewer.dims.point[0]) / e.voxel[0])
            if e.surface.cr_line_cached(zi * e.voxel[0], b) is False and not getattr(self, "_cr_now", False):
                # not computed yet for this slice: wait until scrolling pauses, then draw it
                from qtpy.QtCore import QTimer
                if not hasattr(self, "_cr_timer"):
                    self._cr_timer = QTimer(self); self._cr_timer.setSingleShot(True)
                    self._cr_timer.timeout.connect(self._draw_cr_now)
                self._cr_timer.start(150)
                return
            P = e.surface.cr_line_on_slice(zi * e.voxel[0], b)
            if (P is None or len(P) < 2) and e.surface.method != "slices" and not getattr(self, "_cr_hint", False):
                self._cr_hint = True
                self.msg("The CR line can't be drawn on the slices with this surface method. "
                         "Use 'Follow slices' (Surface tab → Method) for curved surfaces.", error=True)
            if P is not None and len(P) > 1:
                a3 = np.column_stack([np.full(len(P) - 1, zi), P[:-1, 1] / e.voxel[1], P[:-1, 0] / e.voxel[2]])
                b3 = np.column_stack([np.full(len(P) - 1, zi), P[1:, 1] / e.voxel[1], P[1:, 0] / e.voxel[2]])
                e.layers["border_line"].data = np.stack([a3, b3 - a3], axis=1)
            else:
                e.layers["border_line"].data = np.zeros((0, 2, 3))
        else:
            e.layers["border_line"].data = np.zeros((0, 2, 3))

    def _draw_cr_now(self):
        self._cr_now = True
        try:
            self._update_slice_lines()
        finally:
            self._cr_now = False

    def _on_ndisplay(self, *_):
        e = self.active
        if not e:
            return
        three = self.viewer.dims.ndisplay == 3
        if three and e.role == "cell" and e.surface is not None and self.boundary_um() \
                and e.layers.get("border_mesh") is None:
            self._update_border_surface()
        for x in self.entries:
            for k in ("mesh", "border_mesh"):
                if x.layers.get(k) is not None:
                    x.layers[k].visible = three
        for k in ("surface_line", "border_line"):
            if e.layers.get(k) is not None:
                e.layers[k].visible = not three
        self._update_slice_lines()

    # ================================================================ tab 3
    def _tab_border(self):
        w = QWidget(); L = QVBoxLayout(w)
        L.addWidget(_lbl(
            "Mark the brain surface and the lower edge of the Calretinin-positive band <b>on the same "
            "image</b> - normally the 20x low mag; on the 63x if the CR edge is visible there (its surface "
            "from tab 2 is used). The perpendicular surface→CR distance is measured inside that image only "
            "and then drawn on the 63x as a surface parallel to its brain surface. Points can be on one "
            "slice or several."))
        h = QHBoxLayout()
        h.addWidget(QLabel("Mark on:"))
        self.rb_cr_low = QRadioButton("Low mag (20x)"); self.rb_cr_high = QRadioButton("High mag (63x)")
        self.rb_cr_low.setChecked(True)
        bgc = QButtonGroup(w); bgc.addButton(self.rb_cr_low); bgc.addButton(self.rb_cr_high)
        self.rb_cr_low.toggled.connect(lambda on: self._cr_switch())
        h.addWidget(self.rb_cr_low); h.addWidget(self.rb_cr_high); h.addStretch()
        L.addLayout(h)
        h = QHBoxLayout()
        self.btn_mark_surf2 = _btn("✎ Surface points", lambda on: self.btn_mark_surf.setChecked(on), checkable=True)
        self.btn_mark_border = _btn("✎ CR border points", self._toggle_mark_border, checkable=True)
        self.btn_mark_surf.toggled.connect(self.btn_mark_surf2.setChecked)
        h.addWidget(self.btn_mark_surf2); h.addWidget(self.btn_mark_border)
        L.addLayout(h)
        h = QHBoxLayout()
        self.cb_pen2 = QCheckBox("✍ Pen (drag to draw)")
        self.cb_pen2.setChecked(True)
        self.cb_pen2.toggled.connect(self.cb_pen.setChecked)
        self.cb_pen.toggled.connect(self.cb_pen2.setChecked)
        h.addWidget(self.cb_pen2); h.addWidget(_btn("↶ Undo last line / point", self.undo_mark))
        L.addLayout(h)
        h = QHBoxLayout()
        h.addWidget(_btn("Clear this slice", lambda: self.clear_slice(self._marking_key() or "border_pts"),
                         tip="Delete the marks on the slice shown (the line you are marking: surface or CR)"))
        h.addWidget(_btn("🗑 Clear surface", self.clear_surface_points))
        h.addWidget(_btn("🗑 Clear CR points", self.clear_border_points))
        L.addLayout(h)
        L.addWidget(_btn("Measure surface → CR border distance (on the image shown)",
                         lambda: self._guard(self.measure_border)))
        self.tbl_border = QTableWidget(0, 6)
        self.tbl_border.setHorizontalHeaderLabels(["Use", "Image", "Mode", "n", "Mean µm", "SD µm"])
        hb = self.tbl_border.horizontalHeader()
        hb.setSectionResizeMode(QHeaderView.ResizeToContents)
        hb.setSectionResizeMode(1, QHeaderView.Stretch)          # long image names are elided, no sideways scroll
        self.tbl_border.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tbl_border.setTextElideMode(Qt.ElideLeft)
        self.tbl_border.verticalHeader().setVisible(False)
        self.tbl_border.itemChanged.connect(self._on_border_check)
        L.addWidget(self.tbl_border)
        g = QGroupBox("CR border depth used for the cell"); v = QVBoxLayout(g)
        self.rb_mean = QRadioButton("Mean of checked measurements"); self.rb_mean.setChecked(True)
        self.rb_manual = QRadioButton("Manual:")
        bg = QButtonGroup(g); bg.addButton(self.rb_mean); bg.addButton(self.rb_manual)
        self.sp_manual = _dspin(150, 0, 5000, 5, 1, " µm")
        hh = QHBoxLayout(); hh.addWidget(self.rb_manual); hh.addWidget(self.sp_manual)
        v.addWidget(self.rb_mean); v.addLayout(hh)
        self.lbl_boundary = _lbl("—"); v.addWidget(self.lbl_boundary)
        for s in (self.rb_mean.toggled, self.sp_manual.valueChanged):
            s.connect(self._update_border_surface)
        L.addWidget(g)
        L.addStretch()
        return w

    def _toggle_mark_border(self, on):
        if on:
            self.btn_mark_surf.setChecked(False)
            if not self.active or "border_pts" not in self.active.layers:
                self.msg("Show the 20x (or 63x) image to mark the CR border.", error=True)
                self.btn_mark_border.setChecked(False)
                return
        self._select_points_layer("border_pts", on)

    def measure_border(self):
        e = self.active
        if not e or "border_pts" not in e.layers:
            self.msg("Show the 20x or 63x image the CR border was marked on.", error=True); return
        S = self.surface_xyz(e)
        if len(S) < 2:
            self.msg(f"Mark the brain surface on this image ({'63x' if e.role == 'cell' else '20x'}) first.",
                     error=True); return
        B = vox2xyz(e.layers["border_pts"].data, e.voxel)
        r = measure_border_distance(S, B, z_tol_um=e.voxel[0] / 2)
        e.border_result = r
        e.include = True                      # a fresh measurement is used (ticked) by default
        # show the measured border on this low-mag stack too (the 63x surface is kept as it is)
        if e.role == "lowmag" and r["mode"] == "3D" and len(S) >= 6:
            try:
                self.build_surface(e)
            except Exception:
                pass
        self._refresh_border_table()
        self._update_border_surface()
        self._autosave()
        where = "63x" if e.role == "cell" else "20x"
        self.msg(f"{where} {e.name}: surface → CR border = {r['mean']:.1f} ± {r['sd']:.1f} µm "
                 f"(n={r['n']}, {r['mode']}), measured within that image.")

    def _refresh_border_table(self):
        if not hasattr(self, "tbl_border"):
            return
        rows = [e for e in self.entries if e.border_result]
        self.tbl_border.blockSignals(True)
        self.tbl_border.setRowCount(len(rows))
        for i, e in enumerate(rows):
            r = e.border_result
            it = QTableWidgetItem(); it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            it.setCheckState(Qt.Checked if e.include else Qt.Unchecked)
            it.setData(Qt.UserRole, e.name)
            self.tbl_border.setItem(i, 0, it)
            img = ("63x " if e.role == "cell" else "20x ") + e.name
            for j, t in enumerate([img, r["mode"], str(r["n"]), f"{r['mean']:.1f}", f"{r['sd']:.1f}"], 1):
                it_ = QTableWidgetItem(t)
                if j == 1:
                    it_.setToolTip(img)
                self.tbl_border.setItem(i, j, it_)
        self.tbl_border.blockSignals(False)
        self._update_boundary_label()

    def _on_border_check(self, item):
        if item.column() != 0:
            return
        name = item.data(Qt.UserRole)
        for e in self.entries:
            if e.name == name:
                e.include = item.checkState() == Qt.Checked
        self._update_border_surface()

    def boundary_um(self):
        if not hasattr(self, "rb_manual"):
            return None
        if self.rb_manual.isChecked():
            return self.sp_manual.value() or None
        vals = [e.border_result["mean"] for e in self.entries if e.border_result and e.include]
        return float(np.mean(vals)) if vals else None

    def _update_boundary_label(self):
        b = self.boundary_um()
        vals = [e.border_result["mean"] for e in self.entries if e.border_result and e.include]
        n_meas = sum(1 for e in self.entries if e.border_result)
        if b is None and n_meas and not self.rb_manual.isChecked():
            self.lbl_boundary.setText(f"<span style='color:#f88'>{n_meas} measurement(s) exist but none is ticked in "
                                      "the <b>Use</b> column - tick one, or the analysis runs without a CR depth.</span>")
        elif b is None:
            self.lbl_boundary.setText("No CR border depth yet (analysis runs without it).")
        elif self.rb_manual.isChecked():
            self.lbl_boundary.setText(f"Using manual depth: <b>{b:.1f} µm</b>")
        else:
            sd = f" ± {np.std(vals, ddof=1):.1f} (SD across stacks)" if len(vals) > 1 else ""
            self.lbl_boundary.setText(f"Using mean of {len(vals)} stack(s): <b>{b:.1f} µm</b>{sd}")

    def _update_border_surface(self, *_):
        self._update_boundary_label()
        e = self.cell_entry()
        if e is None:
            return
        b = self.boundary_um()
        if (e.surface is None or not b) and e.layers.get("border_mesh") is not None:
            lay = e.layers["border_mesh"]
            if lay in self.viewer.layers:
                self.viewer.layers.remove(lay)
            e.layers["border_mesh"] = None
        if e.surface is not None and b and self.viewer.dims.ndisplay == 3:
            o = e.surface.offset_mesh(b)
            data = (xyz2vox(o.vertices, e.voxel), np.asarray(o.faces))
            if e.layers.get("border_mesh") is not None:
                e.layers["border_mesh"].data = data
            else:
                e.layers["border_mesh"] = self.viewer.add_surface(
                    data, scale=e.voxel, name=f"CELL {e.name} | CR border mesh", colormap="orange", opacity=0.3)
                e.layers["border_mesh"].visible = self.viewer.dims.ndisplay == 3
                self._park(e, e.layers["border_mesh"])
        self._update_slice_lines()

    # ================================================================ tab 4
    def _tab_neuron(self):
        w = QWidget(); L = QVBoxLayout(w)
        L.addWidget(_lbl("Trace the cell in Fiji (SNT) on the same stack and export it as <b>.swc</b>. "
                         "Load it here (or drag the .swc onto the window); it is drawn over the cell stack "
                         "so you can check that it lines up."))
        L.addWidget(_btn("Load SWC (from SNT)…", self.load_swc))
        self.lbl_trace = _lbl("No neuron loaded.")
        L.addWidget(self.lbl_trace)
        L.addWidget(_lbl("<i>If the tracing does not overlay the cell, the voxel size in tab 1 differs "
                         "from the one SNT used - fix it there (the .swc itself is already in µm).</i>"))
        g = QGroupBox("Automatic first draft (to correct in SNT)"); gl = QVBoxLayout(g)
        gl.addWidget(_lbl("Click the soma and Arborist follows the filled processes connected to it. The result "
                          "is a <b>draft</b>: every branch is tagged dendrite, neighbouring cells that touch may "
                          "be joined on, dim branches may be missing. It is saved as "
                          "<b>…_autotrace.swc</b>; open that in SNT, correct it, tag the axon and save it under "
                          "the cell's normal .swc name."))
        f = QFormLayout()
        self.sp_at_reach = _dspin(250, 30, 2000, 10, 0, " µm"); f.addRow("Reach around the soma", self.sp_at_reach)
        self.sp_at_sens = _dspin(5, 2, 60, 0.5, 1, " × noise")
        self.sp_at_sens.setToolTip("How bright a process must be. Lower: finds dimmer branches, but joins more "
                                   "background and neighbours. Higher: only the clear processes.")
        f.addRow("Sensitivity", self.sp_at_sens)
        self.sp_at_min = _dspin(8, 2, 100, 1, 0, " µm"); f.addRow("Shortest branch kept", self.sp_at_min)
        gl.addLayout(f)
        h = QHBoxLayout()
        self.btn_at_pick = _btn("① Click the soma…", self._autotrace_pick, checkable=True)
        h.addWidget(self.btn_at_pick, 1)
        h.addWidget(_btn("Trace again with these settings", lambda: self._guard(self._autotrace_run)), 1)
        gl.addLayout(h)
        self.lbl_at = _lbl("")
        gl.addWidget(self.lbl_at)
        L.addWidget(g)
        g = QGroupBox("Neurite thickness from the image"); gl = QVBoxLayout(g)
        gl.addWidget(_lbl("For tracings without fitted radii: measures the width of the filled neurite at every "
                          "traced point (width at half brightness across the neurite). Saved next to the cell "
                          "and used by Analyze for the diameter features."))
        h = QHBoxLayout()
        self.cmb_radii_ch = QComboBox(); self.cmb_radii_ch.addItem("fill channel: automatic")
        h.addWidget(self.cmb_radii_ch, 1)
        h.addWidget(_btn("Measure diameters", lambda: self._guard(self.measure_radii)), 1)
        gl.addLayout(h)
        self.lbl_radii = _lbl("")
        gl.addWidget(self.lbl_radii)
        L.addWidget(g)
        L.addStretch()
        return w

    # ---- automatic first draft
    def _autotrace_pick(self, on=True):
        if on and self.cell_entry() is None:
            self.btn_at_pick.setChecked(False)
            self.msg("Open the cell stack first.", error=True); return
        if on:
            self.set_active(self.cell_entry())
            self.msg("Scroll to the slice where the soma is brightest and click its centre.")

    def _on_soma_click(self, viewer, event):
        """One-shot: the next click on the cell image sets the soma for the automatic draft."""
        if not (hasattr(self, "btn_at_pick") and self.btn_at_pick.isChecked()) or event.type != "mouse_press" \
                or event.button != 1 or getattr(self, "_space_down", False):
            return
        e = self.cell_entry()
        if e is None or self.active is not e:
            return
        pos = np.asarray(event.position, float)[-3:]              # world: z, y, x in µm
        self._at_seed = np.array([pos[2], pos[1], pos[0]])
        self._click_taken = True                                  # not also a pen / marking click
        self.btn_at_pick.setChecked(False)
        try:
            event.handled = True
        except Exception:
            pass
        from qtpy.QtCore import QTimer
        QTimer.singleShot(0, lambda: self._guard(self._autotrace_run))

    def _autotrace_run(self):
        from . import autotrace as AT
        e = self.cell_entry()
        seed = getattr(self, "_at_seed", None)
        if e is None or seed is None:
            self.msg("Press 'Click the soma…' and click the soma first.", error=True); return
        if e.neuron is not None and len(e.neuron) and getattr(e.neuron, "source", "") != "autotrace":
            if QMessageBox.question(self, "Automatic draft",
                                    "A tracing is already loaded for this cell. Show the automatic draft "
                                    "instead? (Your .swc file is not changed.)") != QMessageBox.Yes:
                return
        data, voxel = e.stack.data, e.voxel
        ch = self.cmb_radii_ch.currentIndex() - 1
        kw = dict(channel=ch if 0 <= ch < data.shape[0] else None, reach_um=self.sp_at_reach.value(),
                  sensitivity=self.sp_at_sens.value(), min_branch_um=self.sp_at_min.value())
        cache = self.__dict__.setdefault("_at_cache", {})
        neuron, info = self._in_background(lambda step: AT.autotrace(data, voxel, seed, progress=step, cache=cache, **kw),
                                           "Automatic draft…")
        if info["branches"] == 0:
            self.lbl_at.setText("<span style='color:#f88'>No process found from that point - lower the sensitivity "
                                "value, or choose the fill channel by hand (box below).</span>")
            return
        neuron.source = "autotrace"
        out = os.path.splitext(e.stack.path)[0] + "_autotrace.swc"
        neuron.write(storage.out(out), f"AUTOMATIC DRAFT by Arborist - not proof-read. source image: {e.stack.path}")
        e.neuron = neuron
        self._show_trace(e)
        if e.surface is not None:
            self._orient(e, e.surface); self._show_mesh(e)
        cname = e.stack.channel_names[info["channel"]] if info["channel"] < len(e.stack.channel_names) else info["channel"]
        self.lbl_at.setText(
            f"<b>Draft:</b> {info['branches']} branches, {info['length_um']:.0f} µm, on channel <b>{cname}</b>"
            + (" (stopped at the branch limit - raise the sensitivity value)" if info.get("capped") else "")
            + f". Saved as <b>{os.path.basename(out)}</b>. Too much joined on? Raise the sensitivity value; "
              "branches missing? Lower it - then press 'Trace again'. Correct it in SNT before you analyse.")
        self.msg(f"Automatic draft: {info['branches']} branches, {info['length_um']:.0f} µm. Not proof-read yet.")

    def measure_radii(self):
        from .morphometry import estimate_radii, apply_radii, pick_fill_channel, radii_sidecar
        e = self.cell_entry()
        if e is None or e.neuron is None or not len(e.neuron):
            self.msg("Open the cell and load its SWC first.", error=True); return
        data, voxel, neuron = e.stack.data, e.voxel, e.neuron
        C, Z = data.shape[0], data.shape[1]
        ch = self.cmb_radii_ch.currentIndex() - 1
        types = tuple(set(self._types(self.ed_dend.text())) | set(self._types(self.ed_axon.text())))

        def work(step):
            c = ch if 0 <= ch < C else pick_fill_channel(neuron, data, voxel)

            def get_plane(z):
                pl = data[c, int(z)]
                return np.asarray(pl.compute() if hasattr(pl, "compute") else pl)
            tab = estimate_radii(neuron, get_plane, voxel, Z, types=types,
                                 progress=lambda i, n: step(f"Measuring neurite widths: slice {i} of {n}"))
            return c, tab
        c, tab = self._in_background(work, "Measuring neurite widths…")
        n_meas = int(tab["measured"].sum())
        n_want = int(neuron.df["type"].isin(types).sum())
        if n_meas < max(10, 0.2 * n_want):
            self.lbl_radii.setText(f"<span style='color:#f88'>Only {n_meas} of {n_want} points could be measured on "
                                   f"channel {e.stack.channel_names[c] if c < len(e.stack.channel_names) else c} - "
                                   "is this the fill channel? Nothing was changed.</span>")
            return
        tab.to_csv(storage.out(radii_sidecar(e.stack.path)), index=False)
        apply_radii(neuron, tab)
        d = 2 * tab.loc[tab["measured"] & neuron.df["type"].isin(self._types(self.ed_dend.text())).to_numpy(),
                        "radius_um"]
        self.lbl_radii.setText(
            f"Measured {n_meas} of {n_want} points on channel "
            f"<b>{e.stack.channel_names[c] if c < len(e.stack.channel_names) else c}</b>; median dendrite "
            f"diameter {d.median():.2f} µm (thinnest 10%: {d.quantile(0.1):.2f}, thickest 10%: {d.quantile(0.9):.2f}). "
            "Includes the optical blur, so the thinnest branches read too thick. Run Analyze to update the features.")
        self._autosave()

    def _apply_saved_radii(self, e):
        """Diameters measured earlier from the image (kept next to the cell) go back onto the tracing."""
        from . import batch
        from .morphometry import apply_radii
        if e is None or e.neuron is None or e.role != "cell":
            return 0
        tab = batch.dbm_radii(e.stack.path)
        n = apply_radii(e.neuron, tab) if tab is not None else 0
        if hasattr(self, "lbl_radii"):
            self.lbl_radii.setText(f"Diameters measured from the image are in use ({n} points)." if n else "")
        return n

    # ================================================================ tab 5
    def _tab_ais(self):
        w = QWidget(); L = QVBoxLayout(w)
        L.addWidget(_lbl("Measure the <b>axon initial segment</b> on the ankyrin-G scan of this cell."))
        self.lbl_ais_stack = _lbl("")
        L.addWidget(self.lbl_ais_stack)
        h = QHBoxLayout()
        h.addWidget(_btn("Show ankyrin image", lambda: self.show_role("ankg")))
        h.addWidget(_btn("Open ankyrin stack…", lambda: self.open_stacks("ankg")))
        L.addLayout(h)
        g = QGroupBox("Mark"); v = QVBoxLayout(g)
        f = QFormLayout()
        self.cmb_ais_ch = QComboBox()
        self.cmb_ais_ch.currentIndexChanged.connect(lambda *_: self._ais_preview())
        f.addRow("Ankyrin channel", self.cmb_ais_ch)
        self.cmb_route_ch = QComboBox()
        self.cmb_route_ch.currentIndexChanged.connect(lambda *_: self._ais_preview())
        f.addRow("Cell fill channel", self.cmb_route_ch)
        self.cb_ais_follow = QCheckBox("Follow the staining (AIS: ankyrin channel; soma→AIS route: fill channel)")
        self.cb_ais_follow.setChecked(True)
        self.cb_ais_follow.toggled.connect(lambda *_: self._ais_preview())
        f.addRow(self.cb_ais_follow)
        v.addLayout(f)
        self.btn_mark_ais = _btn("1  ✎ AIS: click its start (soma side) and its end",
                                 lambda on: self._mark_ais_layer("ais_pts", on), checkable=True,
                                 tip="The path follows the staining between clicks; add a click on sharp bends. "
                                     "Turn the button off when done.")
        self.btn_mark_soma = _btn("2  ● Where the axon-bearing process leaves the soma",
                                  lambda on: self._mark_ais_layer("soma_pt", on), checkable=True,
                                  tip="The point on the soma outline where the axon (or the axon-carrying "
                                      "dendrite) starts. Distances are measured from here.")
        for b_ in (self.btn_mark_ais, self.btn_mark_soma):
            b_.setMinimumHeight(34); v.addWidget(b_)
        h = QHBoxLayout()
        h.addWidget(QLabel("3  Axon comes from the"))
        self.cmb_ais_origin = QComboBox()
        self.cmb_ais_origin.addItems(["soma", "dendrite"])
        self.cmb_ais_origin.currentIndexChanged.connect(self._ais_origin_changed)
        h.addWidget(self.cmb_ais_origin, 1)
        v.addLayout(h)
        self.btn_mark_branch = _btn("4  ◆ Branch point (where the axon leaves the dendrite)",
                                    lambda on: self._mark_ais_layer("branch_pt", on), checkable=True)
        self.btn_mark_branch.setMinimumHeight(34); self.btn_mark_branch.setEnabled(False)
        v.addWidget(self.btn_mark_branch)
        self.btn_mark_route = _btn("(optional)  ◇ Bend points on the soma→AIS route",
                                   lambda on: self._mark_ais_layer("route_pts", on), checkable=True,
                                   tip="Only needed if the route jumps to the wrong process: click on the "
                                       "process where it bends")
        v.addWidget(self.btn_mark_route)
        self.lbl_ais_saved = _lbl("")
        v.addWidget(self.lbl_ais_saved)
        v.addWidget(_btn("Clear marks", self.clear_ais))
        L.addWidget(g)
        g = QGroupBox("Result"); v = QVBoxLayout(g)
        self.lbl_ais = _lbl("No AIS marked yet.")
        v.addWidget(self.lbl_ais)
        L.addWidget(g)
        g = QGroupBox("Other ways to give the AIS"); v = QVBoxLayout(g)
        v.addWidget(_lbl("<i>Traced in SNT as path type Custom (SWC type below), or marked with button 1 on the "
                         "63x stack itself - then it snaps onto the traced axon and the origin is found "
                         "automatically. The ankyrin-stack marks win if both exist.</i>"))
        f = QFormLayout()
        self.sp_ais_type = QSpinBox(); self.sp_ais_type.setRange(0, 99); self.sp_ais_type.setValue(7)
        f.addRow("AIS SWC type", self.sp_ais_type)
        v.addLayout(f)
        L.addWidget(g)
        L.addStretch()
        return w

    def _ais_origin_changed(self, *_):
        dend = self.cmb_ais_origin.currentText() == "dendrite"
        self.btn_mark_branch.setEnabled(dend)
        if not dend:
            self.btn_mark_branch.setChecked(False)
        self._ais_preview()
        if self.ankg_entry() is not None:
            self._ais_marks_changed()

    def _refresh_ais_tab(self):
        if not hasattr(self, "lbl_ais_stack"):
            return
        a = self.ankg_entry()
        names = a.stack.channel_names if a else []
        if [self.cmb_ais_ch.itemText(i) for i in range(self.cmb_ais_ch.count())] != list(names):
            guess = next((i for i, n in enumerate(names) if any(k in n.lower() for k in ("ank", "cy5", "647", "far"))), 0)
            fill = next((i for i, n in enumerate(names) if i != guess and any(
                k in n.lower() for k in ("gfp", "fitc", "488", "alexa 488", "nb", "neurobiotin", "streptavidin"))),
                1 if len(names) > 1 and guess == 0 else 0)
            for cmb, idx in ((self.cmb_ais_ch, guess), (self.cmb_route_ch, fill)):
                cmb.blockSignals(True)
                cmb.clear(); cmb.addItems(names)
                cmb.setCurrentIndex(idx if names else -1)
                cmb.blockSignals(False)
        self.lbl_ais_stack.setText(f"Ankyrin stack: <b>{a.name}</b>" if a else
                                   "<span style='color:#f88'>No ankyrin stack loaded</span> - open it "
                                   "(or open the cell from the Database, which loads it).")

    def _cr_role(self):
        if hasattr(self, "rb_cr_high") and self.rb_cr_high.isChecked():
            return "cell"
        return "lowmag" if any(x.role == "lowmag" for x in self.entries) else "cell"

    def _cr_switch(self):
        for b_ in (self.btn_mark_surf, self.btn_mark_border):
            b_.setChecked(False)
        role = self._cr_role()
        if any(x.role == role for x in self.entries):
            self.show_role(role)
        else:
            self.msg("No 20x stack loaded - the CR border can be marked on the 63x instead.", error=True)

    def _on_tab_changed(self, i):
        """Each step shows the image it works on."""
        role = {1: "cell", 2: self._cr_role(), 3: "cell", 4: "ankg"}.get(i)
        if role and any(x.role == role for x in self.entries) and (not self.active or self.active.role != role):
            self.show_role(role)
        if i == 4:
            self._refresh_ais_tab(); self._ais_preview()

    def _show_trace(self, e):
        df = e.neuron.df
        lay = e.layers["trace"]
        if not len(df):
            lay.data = np.zeros((0, 3)); return
        # draw dense points along every edge so the trace looks continuous on each slice
        c, p, t = e.neuron.segments()
        pts = [df[["x", "y", "z"]].to_numpy(float)]
        cols = [[TYPE_COLORS.get(int(k), "#999") for k in df["type"]]]
        step = min(e.voxel[1:]) * 0.8
        for ci, pi, ti in zip(c, p, t):
            n = int(np.linalg.norm(ci - pi) / step)
            if n > 1:
                s = np.linspace(0, 1, n + 1)[1:-1, None]
                pts.append(pi + s * (ci - pi)); cols.append([TYPE_COLORS.get(int(ti), "#999")] * len(s))
        P = np.vstack(pts)
        C = sum(cols, [])
        lay.data = xyz2vox(P, e.voxel)
        lay.face_color = C
        lay.size = self._pt_size(e.stack, 1.5)
        soma = df[df["type"] == 1]
        Ld = sum(np.linalg.norm(c - p, axis=1)[np.isin(t, [3, 4])]) if len(c) else 0
        La = sum(np.linalg.norm(c - p, axis=1)[t == 2]) if len(c) else 0
        if hasattr(self, "lbl_trace"):
            self.lbl_trace.setText(f"{len(df)} nodes, {len(soma)} soma node(s), dendrite {Ld:.0f} µm, axon {La:.0f} µm.")

    def load_swc(self):
        e = self.cell_entry()
        if e is None:
            self.msg("Open the cell stack first (SWC coordinates are in its µm space).", error=True); return
        p, _ = QFileDialog.getOpenFileName(self, "Load SWC", "", "SWC (*.swc)")
        if p:
            self._guard(self._load_swc, e, p)

    def _load_swc(self, e, p):
        e.neuron = Neuron.read(p)
        e.neuron.paths = []
        self._apply_saved_radii(e)
        self._show_trace(e)
        if e.surface is not None:
            self._orient(e, e.surface); self._show_mesh(e)
        self._ais_preview()
        problems = self._swc_checks(e)
        if problems:
            self.msg(f"Loaded {os.path.basename(p)} ({len(e.neuron)} nodes) - please check: "
                     + " · ".join(problems), error=True)
        else:
            self.msg(f"Loaded {os.path.basename(p)} ({len(e.neuron)} nodes). Check it overlays the cell; "
                     "if not, the voxel size differs from the one used in SNT.")

    def _swc_checks(self, e):
        """Things that silently spoil an analysis: unreadable lines, untagged paths, no soma, a tracing
        that does not sit inside the image (pixel units or another voxel size)."""
        n = e.neuron
        out = list(getattr(n, "warnings", []))
        if not len(n):
            return out + ["the file has no nodes"]
        types = n.df["type"]
        dend = set(self._types(self.ed_dend.text())) if hasattr(self, "ed_dend") else {3, 4}
        axon = set(self._types(self.ed_axon.text())) if hasattr(self, "ed_axon") else {2, 7}
        if not (types == 1).any():
            out.append("no node is tagged as soma (type 1): depths are measured from the first root instead")
        other = ~types.isin(dend | axon | {1})
        if not types.isin(dend).any():
            out.append("no node is tagged as dendrite - tag the paths in SNT (Path Manager > Tag > Type)")
        elif other.mean() > 0.05:
            kinds = ", ".join(str(int(t)) for t in sorted(types[other].unique())[:5])
            out.append(f"{int(other.sum())} nodes have a type that is neither soma, dendrite nor axon "
                       f"(type {kinds}) and are left out of the measurements")
        try:
            C, Z, Y, X = e.stack.data.shape
            hi = np.array([X * e.voxel[2], Y * e.voxel[1], Z * e.voxel[0]])
            xyz = n.xyz
            inside = ((xyz >= -0.02 * hi) & (xyz <= 1.02 * hi)).all(axis=1).mean()
            if inside < 0.8:
                span = np.ptp(xyz, axis=0)
                hint = ""
                if (xyz.max(0)[:2] <= np.array([X, Y]) * 1.02).all() and (span[:2] > hi[:2]).any():
                    hint = " - the coordinates look like pixels, not µm (set the image calibration in Fiji and export again)"
                out.append(f"only {100 * inside:.0f}% of the tracing lies inside the image{hint}")
        except Exception:
            pass
        return out

    # ================================================================ tab 5
    def _tab_analyze(self):
        w = QWidget(); L = QVBoxLayout(w)
        g = QGroupBox("Settings"); f = QFormLayout(g)
        self.ed_cell = QLineEdit(); self.ed_cell.setPlaceholderText("default: cell stack name")
        f.addRow("Cell name", self.ed_cell)
        self.ed_dend = QLineEdit("3,4"); f.addRow("Dendrite SWC types", self.ed_dend)
        self.ed_axon = QLineEdit("2,7"); f.addRow("Axon SWC types", self.ed_axon)
        self.sp_sholl = _dspin(5, 0.5, 100, 1, 1, " µm"); f.addRow("Sholl step", self.sp_sholl)
        self.cb_cover = QCheckBox("Extend the surface past every neurite (recommended)")
        self.cb_cover.setChecked(True)
        f.addRow(self.cb_cover)
        h = QHBoxLayout(); self.ed_out = QLineEdit(); h.addWidget(self.ed_out)
        h.addWidget(_btn("…", self._pick_out)); f.addRow("Output folder", h)
        self.cb_master = QCheckBox("Add/update row in all_cells_summary.csv in that folder")
        self.cb_master.setChecked(True); f.addRow(self.cb_master)
        self.cmb_manual = QComboBox(); self.cmb_manual.setEditable(True)
        self.cmb_manual.addItems(TYPE_LABELS)
        self.cmb_manual.setToolTip("Your own call for this cell (type any label). Stored in the summary "
                                   "and shown on the clustering plots.")
        f.addRow("Manual type label", self.cmb_manual)
        L.addWidget(g)
        L.addWidget(_btn("▶ Run analysis", lambda: self._guard(self.run_analysis)))
        L.addWidget(_lbl(
            "<b>Depth</b> = signed nearest distance to the surface mesh (positive = into the brain; "
            "Method 1 of the old notebook). Method 2 (best-fit plane normal) and Method 3 (tilt-"
            "corrected z) are exported too. Histograms are <b>length-weighted</b>, so uneven node "
            "spacing does not bias them. <b>Normalized depth</b> = depth / CR-border depth "
            "(0 = surface, 1 = end of the sensory SC)."))
        L.addStretch()
        return w

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.ed_out.text())
        if d:
            self.ed_out.setText(d)

    @staticmethod
    def _types(txt):
        return tuple(int(t) for t in txt.replace(";", ",").split(",") if t.strip())

    def _in_background(self, fn, title):
        """Run fn(step) in a worker thread while the window stays responsive (progress window).
        fn must not touch the window; step(text) updates the progress text."""
        import threading
        import traceback
        from qtpy.QtCore import QEventLoop
        from qtpy.QtWidgets import QProgressDialog
        box = {"text": title}
        th = threading.Thread(target=lambda: box.update(
            _run_catch(fn, lambda t: box.__setitem__("text", t))), daemon=True)
        dlg = QProgressDialog(title, None, 0, 0, self.window())
        dlg.setWindowTitle("Arborist"); dlg.setWindowModality(Qt.WindowModal); dlg.setMinimumDuration(300)
        th.start()
        while th.is_alive():
            dlg.setLabelText(box["text"])
            QApplication.processEvents(QEventLoop.AllEvents, 50)
            th.join(0.04)
        dlg.close()
        if "error" in box:
            print(box["trace"])
            raise box["error"]
        return box["value"]

    def run_analysis(self):
        e = self.cell_entry()
        if e is None or e.neuron is None or not len(e.neuron):
            self.msg("Load the SWC of the cell first (tab 4).", error=True); return
        if getattr(e.neuron, "source", "") == "autotrace" and QMessageBox.question(
                self, "Automatic draft", "The loaded tracing is an automatic draft that has not been corrected "
                "in SNT (no axon, possible false joins). Analyse it anyway?") != QMessageBox.Yes:
            return
        if e.surface is None or getattr(e, "surface_stale", False):
            if len(e.layers["surface_pts"].data) >= 2:
                self.build_surface(e)
            else:
                self.msg("Mark and build the surface on the cell stack first.", error=True); return
        surf0 = e.surface
        cover = self.cb_cover.isChecked() and surf0.method in ("smooth", "slices")
        out = self.ed_out.text().strip() or os.path.join(os.path.dirname(e.stack.path), "SC_morph_results")
        self.ed_out.setText(out)
        name = self.ed_cell.text().strip() or e.name
        b = self.boundary_um()
        if b is None and any(x.border_result for x in self.entries):
            self.msg("Note: a CR measurement exists but is not ticked in tab 3 - analysing without CR depth.", error=True)
        # everything the calculation needs, read from the window now
        kw = dict(boundary_um=b, dend_types=self._types(self.ed_dend.text()),
                  axon_types=self._types(self.ed_axon.text()), sholl_step_um=self.sp_sholl.value(),
                  master_csv=storage.out(os.path.join(out, "all_cells_summary.csv")) if self.cb_master.isChecked() else None,
                  manual_type=self.cmb_manual.currentText().strip(),
                  ais_type=self.sp_ais_type.value(), ais_points_xyz=self._ais_points(e),
                  ais_override=self.current_ais()[0] if self.ankg_entry() is not None else None,
                  source_path=e.stack.path)
        z_range = self._z_range(e)
        out_w = storage.out_dir(out)          # on a USB stick: written locally, copied when closing
        neuron = e.neuron
        soma = neuron.df[neuron.df["type"] == 1]
        ref = (soma if len(soma) else neuron.df)[["x", "y", "z"]].to_numpy(float)

        def work(step):
            surf = surf0
            if cover:
                step("Extending the surface over the whole neuron…")
                surf = BrainSurface(surf0.points, method=surf0.method, cover_xyz=neuron.xyz, z_range=z_range,
                                    **surf0.params)
                surf.orient_towards(ref)              # the soma is in the brain: no flip on top
            step("Measuring depths, Sholl, laminar profile and features…")
            res = analysis.run(neuron, surf, out_w, name, **kw)
            neuron.write(storage.out(os.path.join(out, f"{name}.swc")), f"source image: {e.stack.path}")
            step("Drawing the 3D view…")
            try:
                res["view3d"] = analysis.make_3d_figure(neuron, surf, b)
            except Exception as ex:
                print("3D view failed:", ex)
            return res, surf

        res, surf = self._in_background(work, "Analysing the cell…")
        warn = res["summary"].iloc[0].get("soma_warning") if "soma_warning" in res["summary"].columns else None
        if isinstance(warn, str) and warn:
            QMessageBox.warning(self, "Check the soma in the SWC", warn)
        if self._db_root():
            self.ed_master.setText(dbm.pooled_path(self._db_root()))      # all cells of the database
        elif not self.ed_master.text().strip() and self.cb_master.isChecked():
            self.ed_master.setText(os.path.join(out, "all_cells_summary.csv"))
        self.results.sp_sholl_step.blockSignals(True)
        self.results.sp_sholl_step.setValue(self.sp_sholl.value())
        self.results.sp_sholl_step.blockSignals(False)
        self.results.show_results(res, ctx=dict(
            neuron=e.neuron, surf=surf, b=b, name=name, out=out,
            dend=self._types(self.ed_dend.text()), axon=self._types(self.ed_axon.text())))
        self.results_dock.setVisible(True)
        self.results_dock.raise_()
        s = res["summary"].iloc[0]
        txt = f"Soma depth {s.soma_depth_mesh_um:.1f} µm"
        if b:
            txt += (f" ({s.soma_normalized_depth:.2f} of the CR border). "
                    f"Dendrite in sensory SC: {100 * s.dendrite_fraction_in_sensory:.0f}%")
        ft = res["features"]
        if ft.get("type_hint"):
            txt += f". Type hint: {ft['type_hint']} ({ft['type_hint_reason']})"
        try:   # keep the project next to the cell stack so the database sees the progress
            self._save_project(self.project_path or dbm.project_path_for(e.stack.path))
        except Exception as ex:
            print("project autosave failed:", ex)
        if "ais_length_um" in ft:
            txt += f". AIS {ft['ais_length_um']:.1f} µm, from {ft['ais_origin']}"
        self._ais_preview()
        self.msg(f"{txt}. Results saved to {out}")

    # ================================================================ tab 6
    def _tab_classify(self):
        w = QWidget(); L = QVBoxLayout(w)
        L.addWidget(_lbl("Pools every cell analysed so far (one row per cell in "
                         "<b>all_cells_summary.csv</b>) and groups them by morphology. Features are "
                         "measured in the laminar frame: depth from the surface, width parallel to it."))
        h = QHBoxLayout(); self.ed_master = QLineEdit(); h.addWidget(self.ed_master)
        h.addWidget(_btn("…", self._pick_master)); L.addLayout(h)
        h = QHBoxLayout()
        h.addWidget(_btn("Collect all analysed cells", lambda: self._guard(self.collect_cells),
                         tip="Gathers every analysed cell in the database folder into one table "
                             "(SC_morph_results/all_cells_summary.csv in the root folder)."), 2)
        h.addWidget(_btn("Load this table", lambda: self._guard(self.load_master)), 1)
        L.addLayout(h)
        box = QGroupBox("Slice electrophysiology (optional)"); bl = QVBoxLayout(box)
        bl.addWidget(_lbl("An Excel or CSV sheet with one row per cell. Rows are matched to cells by "
                          "date / section / side / cell number (or file name), using the folders of each "
                          "cell's image. Numeric columns become <b>ephys_…</b> features."))
        h = QHBoxLayout()
        h.addWidget(_btn("Link electrophysiology table…", lambda: self._guard(self.link_ephys)))
        h.addWidget(_btn("Unlink", lambda: self._guard(self.unlink_ephys)))
        bl.addLayout(h)
        self.lbl_ephys = _lbl("<i>No table linked.</i>")
        bl.addWidget(self.lbl_ephys)
        L.addWidget(box)
        box = QGroupBox("In vivo light responses (optional)"); bl = QVBoxLayout(box)
        bl.addWidget(_lbl("A table (Excel / CSV, e.g. exported from the Neuropixels analysis) with one row per "
                          "cell that was recorded in vivo and then reconstructed (juxtacellular). Matched "
                          "like the sheet above. Numeric columns become <b>invivo_…</b> features; a cluster "
                          "or type column becomes a label to compare the morphology clusters with."))
        h = QHBoxLayout()
        h.addWidget(_btn("Link in vivo table…", lambda: self._guard(self.link_ephys, "invivo")))
        h.addWidget(_btn("Unlink", lambda: self._guard(self.unlink_ephys, "invivo")))
        bl.addLayout(h)
        self.lbl_invivo = _lbl("<i>No table linked.</i>")
        bl.addWidget(self.lbl_invivo)
        L.addWidget(box)
        L.addWidget(_lbl("<b>Features used for clustering</b> (tick/untick):"))
        h = QHBoxLayout()
        for txt, which in (("Morphology", "morph"), ("Ephys", "ephys"), ("In vivo", "invivo"), ("All", "both"),
                           ("None", "none")):
            h.addWidget(_btn(txt, lambda _=False, w_=which: self._select_features(w_)))
        L.addLayout(h)
        self.lst_feat = QListWidget(); self.lst_feat.setMinimumHeight(260)
        L.addWidget(self.lst_feat)
        f = QFormLayout()
        self.cb_auto_k = QCheckBox("Choose the number of clusters automatically (best silhouette)")
        self.cb_auto_k.setChecked(True)
        self.cb_auto_k.setToolTip("Tries 2-8 clusters and keeps the cut where cells are most clearly closer to "
                                  "their own cluster than to the next one. The 'How many?' tab shows the scores.")
        f.addRow(self.cb_auto_k)
        self.sp_k = QSpinBox(); self.sp_k.setRange(2, 20); self.sp_k.setValue(4)
        self.sp_k.setEnabled(False)
        self.cb_auto_k.toggled.connect(lambda on: self.sp_k.setEnabled(not on))
        f.addRow("Number of clusters", self.sp_k)
        self.cmb_label = QComboBox()
        self.cmb_label.addItem("manual type (tab 6)", "manual_type")
        self.cmb_label.setToolTip("Shown next to each cell in the dendrogram and plots - e.g. the cell type "
                                  "from your electrophysiology sheet, to compare with the morphology clusters.")
        f.addRow("Label cells by", self.cmb_label)
        self.cb_balance = QCheckBox("Weigh morphology, ephys and in vivo equally")
        self.cb_balance.setChecked(True)
        self.cb_balance.setToolTip("Otherwise the group with more ticked features dominates the clustering.")
        self.cb_ephys_only = QCheckBox("Leave out cells without ephys / in vivo data")
        self.cb_ephys_only.setChecked(True)
        self.cb_ephys_only.setToolTip("When ephys or in vivo features are ticked. Unticked: missing values are filled "
                                      "with the median, which pulls those cells towards the middle.")
        f.addRow(self.cb_balance); f.addRow(self.cb_ephys_only)
        L.addLayout(f)
        L.addWidget(_btn("▶ Cluster cells", lambda: self._guard(self.run_clustering)))
        L.addWidget(_btn("Re-analyse all cells…", lambda: self._guard(self.reanalyse_all),
                         tip="Runs Analyze again for every cell that has a surface and a tracing, from its saved "
                             "project and .swc (the images are not opened). Use after an update that changes "
                             "the measurements."))
        self.lbl_cluster = _lbl("")
        L.addWidget(self.lbl_cluster)
        L.addWidget(_lbl("<i>The <b>type hint</b> column in the table is a rule of thumb from soma depth, "
                         "field width, aspect and dorsal fraction (thresholds in morphometry.py); the "
                         "<b>manual_type</b> column is your own label. Edit labels in the CSV if needed.</i>"))
        L.addStretch()
        return w

    def reanalyse_all(self):
        from . import batch
        if not self._db_root():
            self.msg("Open your data folder in the Database tab first.", error=True); return
        self._autosave()
        self.db.rescan(quiet=True)
        rows = [r for r in self.db.rows if r.get("cell")]
        ready = [r for r in rows if r.get("project") and r.get("surface") and r.get("swc")]
        if not ready:
            self.msg("No cell has both a surface and a tracing yet.", error=True); return
        if QMessageBox.question(
                self, "Re-analyse all cells",
                f"{len(ready)} of {len(rows)} cells have a surface and a tracing and will be analysed again with "
                "their saved settings (CR depth, SWC types, Sholl step). Their result files are replaced.\n\n"
                "AIS values measured on the ankyrin scan are kept as they are. Continue?",
                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        results = self._in_background(lambda step: batch.reanalyse_all(ready, step), "Re-analysing…")
        ok = [r for r in results if r[1] == "ok"]
        bad = [r for r in results if r[1] != "ok"]
        self.db.rescan(quiet=True)
        self.collect_cells(quiet=True)
        txt = f"Re-analysed {len(ok)} of {len(results)} cells."
        if bad:
            txt += "<br>Not done: " + "; ".join(f"{n}: {m}" for n, _, m in bad[:8])
        no_cr = [n for n, _, m in ok if "no CR depth" in m]
        if no_cr:
            txt += f"<br>{len(no_cr)} cell(s) have no CR depth saved, so no laminar %: " + ", ".join(no_cr[:6])
        self.lbl_cluster.setText(txt)
        self.msg(txt.replace("<br>", " "), error=bool(bad))
        return results

    def _db_row_for(self, cell_name, source_path):
        rows = [r for r in (getattr(getattr(self, "db", None), "rows", None) or []) if r.get("cell")]
        sp = os.path.normcase(os.path.abspath(source_path)) if source_path else ""
        for r in rows:
            if sp and os.path.normcase(os.path.abspath(r["cell"])) == sp:
                return r
        for r in rows:
            if str(r.get("cell_name")) == str(cell_name):
                return r
        return None

    def _cell_thumbnails(self, t):
        """Picture of every cell in the table (reconstruction, else stitch projection); missing
        projections are made once in the background and cached next to the cell."""
        rows = [self._db_row_for(r["cell"], r.get("source_path") if isinstance(r.get("source_path"), str) else "")
                for _, r in t.iterrows()]
        if not any(rows):
            return None

        def work(step):
            out = []
            for k, r in enumerate(rows):
                step(f"Pictures of the cells for the dendrogram: {k + 1} of {len(rows)} (made once, then cached)")
                try:
                    out.append(dbm.thumbnail(r, compute_stitch=True) if r else None)
                except Exception as ex:
                    print("thumbnail failed:", ex)
                    out.append(None)
            return out
        return self._in_background(work, "Pictures of the cells…")

    def _db_root(self):
        r = getattr(getattr(self, "db", None), "root", "") or ""
        return r if r and os.path.isdir(r) else ""

    def _write_xlsx(self, table, csv_path):
        """An Excel copy next to a CSV (a CSV opens as one column in Excel with decimal commas)."""
        x = os.path.splitext(csv_path)[0] + ".xlsx"
        try:
            table.to_excel(storage.out(x), index=False)
            return x
        except PermissionError:
            self.msg(f"{os.path.basename(x)} is open in Excel - close it to get the updated copy.", error=True)
        except Exception as ex:
            print("xlsx not written:", ex)
        return None

    def collect_cells(self, quiet=False):
        """Pool every analysed cell under the database root into one table and load it."""
        root = self._db_root()
        if not root:
            self.msg("Open your data folder in the Database tab first (it knows where all cells are).", error=True)
            return None
        t, files = dbm.collect_summaries(root)
        if not len(t):
            self.msg("No analysed cells found yet under " + root, error=True)
            return None
        p = dbm.pooled_path(root)
        t.to_csv(storage.out(p), index=False)
        self._write_xlsx(t, p)
        self.ed_master.setText(p)
        prev, self._collecting = getattr(self, "_collecting", False), True
        try:
            out = self.load_master()
        finally:
            self._collecting = prev
        if not quiet:
            self.msg(f"Collected {len(t)} analysed cells into {p}")
        return out

    def _pick_master(self):
        p, _ = QFileDialog.getOpenFileName(self, "Cell table", self.ed_master.text(), "CSV (*.csv)")
        if p:
            self.ed_master.setText(p)
            self._guard(self.load_master)

    def _fill_source_paths(self, t):
        """Older summaries have no image path; take it from the database where possible."""
        if "source_path" not in t.columns:
            t["source_path"] = ""
        t["source_path"] = t["source_path"].fillna("").astype(str)
        by_name = {}
        for r in (getattr(getattr(self, "db", None), "rows", None) or []):
            if r.get("cell"):
                by_name.setdefault(r.get("cell_name") or os.path.splitext(os.path.basename(r["cell"]))[0], r["cell"])
        for i in t.index[t["source_path"] == ""]:
            t.at[i, "source_path"] = by_name.get(str(t.at[i, "cell"]), "")
        return t

    def load_master(self):
        import pandas as pd
        from . import ephys
        p = self.ed_master.text().strip()
        root = self._db_root()
        if root and (not p or os.path.normcase(os.path.abspath(p)) == os.path.normcase(dbm.pooled_path(root))) \
                and not getattr(self, "_collecting", False):
            # the pooled table of the whole database: rebuild it so newly analysed cells are in
            self._collecting = True
            try:
                return self.collect_cells(quiet=True)
            finally:
                self._collecting = False
        if not p or not storage.exists(p):
            self.msg("Press 'Collect all analysed cells', or choose an all_cells_summary.csv.", error=True)
            return None
        t = self._fill_source_paths(pd.read_csv(storage.read(p)))
        info = dict(linked=False)
        for kind in ephys.KINDS:
            try:
                t, inf = ephys.merge(t, p, kind=kind)
            except Exception as ex:
                self._link_label(kind).setText(
                    f"<span style='color:#c0392b'>Could not read the {kind} table: {ex}</span>")
            else:
                self._ephys_label(inf, kind)
                if kind == "ephys":
                    info = inf
        self._ephys_info = info
        prev = {self.lst_feat.item(i).text(): self.lst_feat.item(i).checkState()
                for i in range(self.lst_feat.count()) if self.lst_feat.item(i).flags() & Qt.ItemIsUserCheckable}
        self.lst_feat.clear()
        skip = {"cell", "type_hint", "type_hint_reason", "manual_type", "has_axon", "ais_origin", "ais_note",
                "source_path", "day", "section", "side"} | {ephys.prefix_of(k) + "row" for k in ephys.KINDS}
        num = [c for c in t.columns if c not in skip and pd.api.types.is_numeric_dtype(t[c])]
        pfx = tuple(ephys.prefix_of(k) for k in ephys.KINDS)
        groups = [("Morphology", [c for c in num if not c.startswith(pfx)])]
        groups += [(title, [c for c in num if c.startswith(ephys.prefix_of(k))]) for k, title in ephys.KINDS.items()]
        for title, cols in groups:
            if not cols:
                continue
            hd = QListWidgetItem(f"— {title} ({len(cols)}) —")
            fnt = hd.font(); fnt.setBold(True); hd.setFont(fnt)
            hd.setFlags(Qt.NoItemFlags)
            self.lst_feat.addItem(hd)
            for c in cols:
                it = QListWidgetItem(c)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                default = Qt.Checked if (c in DEFAULT_CLUSTER_FEATURES or c.startswith(pfx)) else Qt.Unchecked
                it.setCheckState(prev.get(c, default))
                self.lst_feat.addItem(it)
        cur = self.cmb_label.currentData()
        self.cmb_label.blockSignals(True)
        self.cmb_label.clear()
        self.cmb_label.addItem("manual type (tab 6)", "manual_type")
        for c in t.columns:
            for k in ephys.KINDS:
                lp = ephys.prefix_of(k) + "label_"
                if c.startswith(lp):
                    self.cmb_label.addItem(("ephys sheet: " if k == "ephys" else "in vivo: ")
                                           + c[len(lp):].replace("_", " "), c)
        i = self.cmb_label.findData(cur)
        self.cmb_label.setCurrentIndex(max(i, 0))
        self.cmb_label.blockSignals(False)
        self.lbl_cluster.setText(f"{len(t)} cells in table.")
        return t

    def _link_label(self, kind="ephys"):
        return self.lbl_invivo if kind == "invivo" else self.lbl_ephys

    def _ephys_label(self, info, kind="ephys"):
        lbl = self._link_label(kind)
        if not info.get("linked"):
            lbl.setText("<i>No table linked.</i>"); return
        txt = (f"<b>{os.path.basename(info['path'])}</b>: {info['matched']} of {info['n_cells']} cells matched, "
               f"{len(info['features'])} {kind.replace('invivo', 'in vivo')} features.")
        if info["unmatched_rows"]:
            txt += f" {len(info['unmatched_rows'])} table rows not used (no analysed cell)."
        lbl.setText(txt)

    def _select_features(self, which):
        from . import ephys
        for i in range(self.lst_feat.count()):
            it = self.lst_feat.item(i)
            if not it.flags() & Qt.ItemIsUserCheckable:
                continue
            is_e = it.text().startswith(ephys.PREFIX)
            is_v = it.text().startswith(ephys.prefix_of("invivo"))
            on = {"morph": not (is_e or is_v) and it.text() in DEFAULT_CLUSTER_FEATURES,
                  "ephys": is_e, "invivo": is_v,
                  "both": is_e or is_v or it.text() in DEFAULT_CLUSTER_FEATURES, "none": False}[which]
            it.setCheckState(Qt.Checked if on else Qt.Unchecked)

    def link_ephys(self, kind="ephys"):
        import pandas as pd
        from . import ephys
        master = self.ed_master.text().strip()
        if not master or not storage.exists(master):
            self.msg("Load the cell table (all_cells_summary.csv) first.", error=True); return
        link = ephys.load_link(master, kind) or {}
        start = link.get("path") or os.path.dirname(master)
        title = ephys.KINDS[kind] + " table"
        p, _ = QFileDialog.getOpenFileName(self, title, start, "Tables (*.xlsx *.xlsm *.xls *.csv *.txt)")
        if not p:
            return
        cells = self._fill_source_paths(pd.read_csv(storage.read(master)))
        dlg = EphysLinkDialog(self, p, cells, link if link.get("path") == p else {})
        dlg.setWindowTitle("Link " + title.lower())
        if dlg.exec():
            ephys.save_link(master, dlg.result_link(), kind)
            self.load_master()
            self.msg(f"{title} linked. " + self._link_label(kind).text().replace("<b>", "").replace("</b>", ""))

    def unlink_ephys(self, kind="ephys"):
        from . import ephys
        master = self.ed_master.text().strip()
        lp = ephys.link_path(master, kind) if master else ""
        for f in (lp, storage.store().local_for(lp)) if master else ():
            if os.path.exists(f):
                os.remove(f)
        self.load_master()

    def run_clustering(self):
        t = self.load_master()
        if t is None and not self._db_root():
            return
        from . import ephys
        if (t is None or len(t) < 3) and self._db_root():
            t2 = self.collect_cells(quiet=True)
            if t2 is not None and len(t2) > (0 if t is None else len(t)):
                t = t2
                self.msg(f"The chosen table had too few cells - using all {len(t)} analysed cells in the database.")
        if t is None or len(t) < 3:
            n_ = 0 if t is None else len(t)
            raise ValueError(f"Clustering needs at least 3 analysed cells; found {n_}. "
                             "Analyse more cells (tab 6) or press 'Collect all analysed cells'.")
        feats = [self.lst_feat.item(i).text() for i in range(self.lst_feat.count())
                 if self.lst_feat.item(i).flags() & Qt.ItemIsUserCheckable
                 and self.lst_feat.item(i).checkState() == Qt.Checked]
        pfx = tuple(ephys.prefix_of(k) for k in ephys.KINDS)
        efeats = [f for f in feats if f.startswith(pfx)]
        dropped = 0
        if efeats and self.cb_ephys_only.isChecked():
            keep = np.ones(len(t), bool)
            for pf in pfx:                       # a cell needs data from every linked table that is ticked
                cols_ = [f for f in efeats if f.startswith(pf)]
                if cols_:
                    keep &= t[cols_].notna().any(axis=1).to_numpy()
            dropped = int((~keep).sum())
            t = t[keep].reset_index(drop=True)
            if len(t) < 3:
                raise ValueError(f"Only {len(t)} cells have data in the ticked linked tables - untick those "
                                 "features, or untick 'Leave out cells without ephys / in vivo data'.")
        res = cluster_cells(t, feats, "auto" if self.cb_auto_k.isChecked() else self.sp_k.value(),
                            balance_prefix=pfx if self.cb_balance.isChecked() else None)
        t = t.copy(); t["cluster"] = res["labels"]
        thumbs = self._cell_thumbnails(t)
        figs = cluster_figures(t, res, label_col=self.cmb_label.currentData() or "manual_type", thumbs=thumbs,
                               label_name=self.cmb_label.currentText())
        out = os.path.splitext(self.ed_master.text())[0] + "_clusters"
        cols = [c for c in ["cell", "day", "section", "side", "cluster", "manual_type", "type_hint"]
                + [pf + "row" for pf in pfx] if c in t.columns]
        cols += [c for c in t.columns if c.startswith(tuple(pf + "label_" for pf in pfx))]
        t[cols].to_csv(storage.out(out + ".csv"), index=False)
        self._write_xlsx(t[cols], out + ".csv")
        if efeats:
            t.to_csv(storage.out(out + "_table_with_ephys.csv"), index=False)
            self._write_xlsx(t, out + "_table_with_ephys.csv")
        for k, fig in figs.items():
            fig.savefig(storage.out(f"{out}_{k}.png"), dpi=150); fig.savefig(storage.out(f"{out}_{k}.svg"))
        lines = []
        lc = self.cmb_label.currentData() or "manual_type"
        lname = "manual" if lc == "manual_type" else \
            self.cmb_label.currentText().replace("ephys sheet: ", "").replace("in vivo: ", "")
        for k in sorted(set(res["labels"])):
            sub = t[t["cluster"] == k]
            lab = ", ".join(f"{a}" for a in sub.get(lc, sub["cell"]).fillna("").astype(str) if a)
            lines.append(f"Cluster {k}: {len(sub)} cells" + (f" ({lname}: {lab})" if lab else ""))
        sil = res.get("silhouettes") or {}
        head = (f"{res['k']} clusters" + (" (chosen automatically" if res.get("auto_k") else " (set by hand")
                + (f", silhouette {sil[res['k']]:.2f})" if res["k"] in sil else ")"))
        if sil and len(t) < 10:
            head += f" - with only {len(t)} cells the grouping is still tentative"
        if res.get("stability"):
            st_ = res["stability"]
            weak = [str(n_) for n_, v_ in zip(t["cell"], st_["per_cell"]) if v_ < 0.7]
            head += (f"\nStability (each cell left out in turn): {100 * st_['overall']:.0f}% of cell pairs keep "
                     "their relation" + (f"; least stable: {', '.join(weak[:4])}" if weak else ""))
        if res.get("agreement"):
            head += (f"\nAgreement with '{self.cmb_label.currentText()}': adjusted Rand "
                     f"{res['agreement']['ari']:.2f} over {res['agreement']['n']} labelled cells "
                     "(1 = same grouping, 0 = chance)")
        txt = head + "\n" + "\n".join(lines)
        if res.get("dropped_sparse"):
            txt += (f"\nLeft out (missing in more than a third of the cells): "
                    + ", ".join(res["dropped_sparse"]))
        if res.get("dropped_constant"):
            txt += "\nLeft out (same value in every cell): " + ", ".join(res["dropped_constant"])
        if dropped:
            txt += f"\n({dropped} cells without ephys data left out)"
        self.lbl_cluster.setText(txt.replace("\n", "<br>"))
        self.results.show_figures(figs, txt + "\n\nFeatures: " + ", ".join(res["features"]),
                                  {"dendrogram": "Dendrogram", "pca": "PCA", "heatmap": "Features",
                                   "silhouette": "How many?", "stability": "Stability",
                                   "agreement": "vs labels"})
        self.results_dock.setVisible(True); self.results_dock.raise_()
        self.msg(f"Clustered {len(t)} cells on {len(res['features'])} features. Saved {out}.csv + figures.")

    # ============================================================ database
    def open_from_db(self, row):
        prev, self._loading = getattr(self, "_loading", False), True
        try:
            self._open_from_db(row)
        finally:
            self._loading = prev

    def _open_from_db(self, row):
        if not row.get("cell"):
            self.msg("No 63x cell stack in this folder yet.", error=True); return
        # the click preview set the open cell's layers aside: bring them back so they can be saved /
        # removed properly, then drop the preview
        self.end_preview(restore=True)
        if self.entries:
            # everything is saved automatically after each change, so there is nothing to ask: make sure
            # the last changes are written and move on
            if hasattr(self, "_marks_timer") and self._marks_timer.isActive():
                self._marks_timer.stop()
                self._apply_mark_changes()
            else:
                self._autosave()
        loaded = False
        if row.get("project") and storage.exists(row["project"]):
            try:
                self._load_project(row["project"])
                loaded = True
            except (ValueError, KeyError, TypeError, IndexError) as ex:
                # a damaged project file must not lock the cell: keep it aside and start from the images
                bad = storage.read(row["project"])
                try:
                    os.replace(bad, bad + ".unreadable")
                except OSError:
                    pass
                self.msg(f"The saved project of this cell could not be read ({ex}); it was kept as "
                         f"{os.path.basename(bad)}.unreadable and the cell is opened fresh.", error=True)
        if loaded:
            # results always next to the cell, named after its 63x image (that is where the Database
            # looks); a project saved earlier may carry an old drive letter or another name
            self.ed_cell.setText(row.get("cell_name", ""))
            self.ed_out.setText(os.path.join(os.path.dirname(row["cell"]), dbm.RESULTS_DIR))
            c = self.cell_entry()
            if c is not None and row.get("swc") and os.path.exists(row["swc"]):
                self._load_swc(c, row["swc"])        # the .swc file wins over the copy in the project
            if c is not None and row.get("surface_txt") and len(c.layers["surface_pts"].data) < 4:
                self._import_surface_txt(c, row["surface_txt"])
        else:
            for e in list(self.entries):
                self._remove(e)
            self.project_path = dbm.project_path_for(row["cell"])
            self.add_stack(row["cell"], "cell")
            for f in row.get("cr", []):
                self.add_stack(f, "lowmag")
            for f in row.get("ankg", [])[:1]:
                self.add_stack(f, "ankg")
            self.pending_overview = list(row.get("overview") or [])
            if row.get("swc"):
                self._load_swc(self.cell_entry(), row["swc"])
            if row.get("surface_txt"):
                self._import_surface_txt(self.cell_entry(), row["surface_txt"])
            self.ed_cell.setText(row.get("cell_name", ""))
            self.ed_out.setText(os.path.join(os.path.dirname(row["cell"]), dbm.RESULTS_DIR))
            c = self.cell_entry()
            if c:
                self.set_active(c)
        if not any(x.role == "overview" for x in self.entries):
            self.pending_overview = list(row.get("overview") or [])
        self._refresh_view_controls()
        self.msg(f"Opened {row['day']} / {row['section']} / {row.get('side', '')}: {row.get('cell_name', '')}"
                 + (" - the 10x whole section loads when you click its view button." if self.pending_overview else ""))

    # ============================================================ project io
    def save_project(self):
        c = self.cell_entry()
        default = self.project_path or (dbm.project_path_for(c.stack.path) if c else "Arborist_project.json")
        p, _ = QFileDialog.getSaveFileName(self, "Save project", default,
                                           "Arborist project (*.json)")
        if p:
            self._guard(self._save_project, p)

    def _save_project(self, p, quiet=False):
        d = {"version": __version__, "stacks": [], "boundary": {
            "manual": self.rb_manual.isChecked(), "manual_um": self.sp_manual.value()},
            "surface_params": self._surface_kwargs(),
            "analysis": {"out": self.ed_out.text(), "cell": self.ed_cell.text(),
                         "manual_type": self.cmb_manual.currentText(), "ais_type": self.sp_ais_type.value(),
                         "ais_origin": self.cmb_ais_origin.currentText(),
                         "ais_channel": self.cmb_ais_ch.currentText(), "ais_follow": self.cb_ais_follow.isChecked(),
                         "route_channel": self.cmb_route_ch.currentText(),
                         "dend": self.ed_dend.text(), "axon": self.ed_axon.text(),
                         "sholl_step": self.sp_sholl.value()}}
        for e in self.entries:
            sd = {"path": e.stack.path, "role": e.role, "voxel": list(e.voxel),
                  "surface_points_um": self.surface_xyz(e).tolist(), "flipped": e.flipped,
                  "include": e.include, "border_result": e.border_result,
                  "surface": None if e.surface is None else {"method": e.surface.method, **e.surface.params}}
            if "border_pts" in e.layers:
                sd["border_points_um"] = vox2xyz(e.layers["border_pts"].data, e.voxel).tolist()
            if e.role == "ankg":
                for k in ("ais_pts", "soma_pt", "branch_pt", "route_pts"):
                    sd[k.replace("_pts", "_points").replace("_pt", "_point") + "_um"] = \
                        vox2xyz(e.layers[k].data, e.voxel).tolist()
            if e.neuron is not None and getattr(e.neuron, "source", "") != "autotrace":   # drafts live in their own file
                sd["neuron"] = e.neuron.df.values.tolist()
                sd["ais_points_um"] = vox2xyz(e.layers["ais_pts"].data, e.voxel).tolist()
            d["stacks"].append(sd)
        # written to a temporary file first and then swapped in, so a crash or a full disk in the middle
        # of saving can never leave a half-written project
        target = storage.out(p)
        tmp = target + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(d, fh, separators=(",", ":"))
        os.replace(tmp, target)
        self.project_path = p
        if not quiet:
            self.msg(f"Project saved: {p}")
        if getattr(self, "db", None):
            if quiet:
                self.db.refresh_soon()           # autosave: update the table later, not after every mark
            else:
                self.db.rescan(quiet=True)

    def load_project(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open project", "", "Arborist project (*.json)")
        if p:
            self._guard(self._load_project, p)

    def _load_project(self, p):
        prev, self._loading = getattr(self, "_loading", False), True
        self._migrated = False
        try:
            self._load_project_impl(p)
        finally:
            self._loading = prev
        if self._migrated and not prev:
            self.project_path = self.project_path or p
            self._autosave()                       # keep the switch to 'Follow slices'

    def _load_project_impl(self, p):
        with open(storage.read(p)) as fh:
            d = json.load(fh)
        for e in list(self.entries):
            self._remove(e)
        base = os.path.dirname(p)
        for sd in d["stacks"]:
            path = sd["path"]
            if not os.path.exists(path):          # project moved together with the data / new drive letter
                alt = _relocate(path, base)
                if alt:
                    path = alt
                else:
                    self.msg(f"Missing stack: {path}", error=True); continue
            e = self.add_stack(path, sd["role"], voxel=sd["voxel"])
            if e is None:
                continue
            if "surface_pts" in e.layers:
                e.layers["surface_pts"].data = xyz2vox(np.array(sd.get("surface_points_um") or np.zeros((0, 3))), e.voxel)
            if "border_pts" in e.layers and sd.get("border_points_um"):
                e.layers["border_pts"].data = xyz2vox(np.array(sd["border_points_um"]), e.voxel)
            e.flipped = sd.get("flipped", False)
            e.include = sd.get("include", True)
            e.border_result = sd.get("border_result")
            for k, key in (("ais_pts", "ais_points_um"), ("soma_pt", "soma_point_um"), ("branch_pt", "branch_point_um"),
                           ("route_pts", "route_points_um")):
                if k in e.layers and sd.get(key):
                    e.layers[k].data = xyz2vox(np.array(sd[key]), e.voxel)
            if e.neuron is not None and sd.get("neuron"):
                import pandas as pd
                e.neuron = Neuron(pd.DataFrame(sd["neuron"], columns=Neuron.COLS))
                self._show_trace(e)
            if sd.get("surface") and "surface_pts" in e.layers and len(e.layers["surface_pts"].data) >= 2:
                s = sd["surface"]
                if s.get("method", "slices") != "slices":
                    # projects from before 'Follow slices' existed: the older methods cannot follow a
                    # curved surface (folds, ragged edges, no CR line on the slices)
                    old = {"smooth": "Height field", "delaunay": "Delaunay"}.get(s["method"], s["method"])
                    self.msg(f"This project used the older '{old}' surface method; switched to 'Follow slices' "
                             f"and rebuilt the surface from your points.", error=True)
                    s = dict(s, method="slices")
                    self._migrated = True
                self._set_method(s["method"])
                self.sp_smooth.setValue(s["smoothing"]); self.sp_grid.setValue(s["grid_um"])
                self.sp_extend.setValue(s["extend_um"])
                self.build_surface(e)
        b = d.get("boundary", {})
        self.rb_manual.setChecked(b.get("manual", False)); self.rb_mean.setChecked(not b.get("manual", False))
        self.sp_manual.setValue(b.get("manual_um", 150))
        a = d.get("analysis", {})
        self.cmb_manual.setCurrentText(a.get("manual_type", ""))
        self.ed_out.setText(a.get("out", "")); self.ed_cell.setText(a.get("cell", ""))
        self.ed_dend.setText(a.get("dend", "3,4")); self.ed_axon.setText(a.get("axon", "2,7"))
        self.sp_ais_type.setValue(a.get("ais_type", 7))
        self.cmb_ais_origin.setCurrentText(a.get("ais_origin", "soma"))
        self._refresh_ais_tab()
        if a.get("ais_channel"):
            self.cmb_ais_ch.setCurrentText(a["ais_channel"])
        if a.get("route_channel"):
            self.cmb_route_ch.setCurrentText(a["route_channel"])
        self.cb_ais_follow.setChecked(a.get("ais_follow", True))
        if a.get("sholl_step"):
            self.sp_sholl.setValue(float(a["sholl_step"]))
        self._ais_preview()
        self._refresh_border_table()
        self._update_border_surface()
        c = self.cell_entry()
        if c:
            self.set_active(c)
        self.project_path = p
        self.msg(f"Project loaded: {os.path.basename(p)}")


# ================================================================ results dock
class EphysLinkDialog(QDialog):
    """Choose the sheet and identifying columns, then check / correct which row goes with which cell."""

    def __init__(self, parent, path, cells, link):
        super().__init__(parent)
        from . import ephys
        self.E, self.path, self.cells = ephys, path, cells
        self.manual = dict(link.get("manual") or {})
        self.setWindowTitle("Link electrophysiology table")
        self.resize(900, 700)
        L = QVBoxLayout(self)
        L.addWidget(_lbl(f"<b>{os.path.basename(path)}</b>"))
        f = QFormLayout()
        self.cmb_sheet = QComboBox()
        sheets = ephys.sheet_names(path)
        self.cmb_sheet.addItems(sheets or ["(single table)"])
        if link.get("sheet") in sheets:
            self.cmb_sheet.setCurrentText(link["sheet"])
        self.cmb_sheet.setEnabled(bool(sheets))
        f.addRow("Sheet", self.cmb_sheet)
        L.addLayout(f)
        L.addWidget(_lbl("<b>Columns that identify the cell</b> (date, section, side, cell no. or file name). "
                         "Every other numeric column becomes a feature."))
        self.lst_id = QListWidget(); self.lst_id.setMaximumHeight(150)
        L.addWidget(self.lst_id)
        self.lbl_feat = _lbl("")
        L.addWidget(self.lbl_feat)
        L.addWidget(_lbl("<b>Matches</b> — automatic where the identifiers fit exactly one cell; "
                         "change any row with the drop-down."))
        self.tbl = QTableWidget(0, 3)
        self.tbl.setHorizontalHeaderLabels(["Cell (analysed)", "Row in the table", ""])
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tbl.verticalHeader().setVisible(False)
        L.addWidget(self.tbl, 1)
        self.lbl_sum = _lbl("")
        L.addWidget(self.lbl_sum)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        L.addWidget(bb)
        self._init_ids = link.get("id_cols")
        self.cmb_sheet.currentIndexChanged.connect(lambda *_: self._load_sheet())
        self.lst_id.itemChanged.connect(lambda *_: self._rematch())
        self._load_sheet()

    def _sheet(self):
        return self.cmb_sheet.currentText() if self.cmb_sheet.isEnabled() else None

    def _load_sheet(self):
        import pandas as pd
        self.t = self.E.read_table(self.path, self._sheet())
        guess = [c for c in (self._init_ids or []) if c in self.t.columns] or self.E.guess_id_columns(self.t)
        self._init_ids = None
        self.lst_id.blockSignals(True)
        self.lst_id.clear()
        for c in self.t.columns:
            def short(v):
                if isinstance(v, pd.Timestamp):
                    return v.strftime("%Y-%m-%d")
                if isinstance(v, float):
                    return f"{v:.4g}"
                return str(v)[:18]
            ex = ", ".join(short(v) for v in self.t[c].dropna().head(3))
            it = QListWidgetItem(f"{c}    e.g. {ex}")
            it.setData(Qt.UserRole, c)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if c in guess else Qt.Unchecked)
            self.lst_id.addItem(it)
        self.lst_id.blockSignals(False)
        self._rematch()

    def id_cols(self):
        return [self.lst_id.item(i).data(Qt.UserRole) for i in range(self.lst_id.count())
                if self.lst_id.item(i).checkState() == Qt.Checked]

    def _rematch(self):
        ids = self.id_cols()
        feats = self.E.feature_columns(self.t, ids)
        self.lbl_feat.setText(f"{len(feats)} feature columns: " + ", ".join(map(str, feats[:12]))
                              + (" …" if len(feats) > 12 else ""))
        self.auto = self.E.auto_match(self.cells, self.t, ids)
        self.keys = self.E.row_keys(self.t, ids)
        names = self.cells["cell"].astype(str).tolist()
        self.tbl.setRowCount(len(names))
        self.combos = {}
        for r, n in enumerate(names):
            self.tbl.setItem(r, 0, QTableWidgetItem(n))
            cb = QComboBox(); cb.addItem("— none —"); cb.addItems(self.keys)
            a = self.auto.get(n)
            if n in self.manual:
                k = self.manual[n]
                cb.setCurrentIndex(self.keys.index(k) + 1 if k in self.keys else 0)
            elif a is not None:
                cb.setCurrentIndex(a + 1)
            cb.currentIndexChanged.connect(lambda _i, n_=n: self._changed(n_))
            self.combos[n] = cb
            self.tbl.setCellWidget(r, 1, cb)
            self._status(r, n)
        self._summary()

    def _status(self, r, n):
        i = self.combos[n].currentIndex() - 1
        a = self.auto.get(n)
        txt = "manual" if n in self.manual and i != (a if a is not None else -1) else ("auto" if i >= 0 else "no match")
        it = QTableWidgetItem(txt)
        it.setForeground(QColor({"auto": "#3a9a3a", "manual": "#2c6fb7", "no match": "#c0392b"}[txt]))
        self.tbl.setItem(r, 2, it)

    def _changed(self, n):
        i = self.combos[n].currentIndex() - 1
        self.manual[n] = self.keys[i] if i >= 0 else ""
        self._status(list(self.combos).index(n), n)
        self._summary()

    def _summary(self):
        chosen = [cb.currentIndex() - 1 for cb in self.combos.values()]
        used = [c for c in chosen if c >= 0]
        dup = len(used) - len(set(used))
        txt = f"{len(used)} of {len(chosen)} cells matched; {len(self.keys) - len(set(used))} table rows unused."
        if dup:
            txt += f" <span style='color:#c0392b'>{dup} row(s) assigned to more than one cell.</span>"
        self.lbl_sum.setText(txt)

    def result_link(self):
        manual = {}
        for n, cb in self.combos.items():
            i = cb.currentIndex() - 1
            a = self.auto.get(n)
            if i != (a if a is not None else -1):
                manual[n] = self.keys[i] if i >= 0 else ""
        return dict(path=self.path, sheet=self._sheet(), id_cols=self.id_cols(), manual=manual)


class ResultsWidget(QWidget):
    def __init__(self):
        super().__init__()
        L = QVBoxLayout(self)
        box = QGroupBox("Plot options"); f = QFormLayout(box)
        h = QHBoxLayout()
        self.sp_lam_bin = QSpinBox(); self.sp_lam_bin.setRange(1, 50); self.sp_lam_bin.setValue(5)
        self.sp_lam_bin.setSuffix(" %")
        self.cmb_lam_mode = QComboBox(); self.cmb_lam_mode.addItems(["% of length", "µm per band"])
        self.cb_lam_dend = QCheckBox("dendrites"); self.cb_lam_dend.setChecked(True)
        self.cb_lam_axon = QCheckBox("axon"); self.cb_lam_axon.setChecked(True)
        self.cb_lam_smooth = QCheckBox("smooth line")
        for wdg in (self.sp_lam_bin, self.cmb_lam_mode):
            h.addWidget(wdg)
        f.addRow("Laminar bins", h)
        h = QHBoxLayout()
        for wdg in (self.cb_lam_dend, self.cb_lam_axon, self.cb_lam_smooth):
            h.addWidget(wdg)
        h.addStretch(1)
        f.addRow("Show", h)
        h = QHBoxLayout()
        self.sp_sholl_step = _dspin(10, 1, 200, 1, 0, " µm")
        self.sp_depth_bin = QSpinBox(); self.sp_depth_bin.setRange(2, 50); self.sp_depth_bin.setValue(10)
        self.sp_depth_bin.setSuffix(" % depth bins")
        self.sp_shells = QSpinBox(); self.sp_shells.setRange(1, 12); self.sp_shells.setValue(4)
        self.sp_shells.setSuffix(" spheres")
        for wdg in (self.sp_sholl_step, self.sp_shells):
            h.addWidget(wdg)
        f.addRow("Sholl shell", h)
        f.addRow("Sholl × depth", self.sp_depth_bin)
        h = QHBoxLayout()
        h.addWidget(_btn("Redraw", self.redraw))
        h.addWidget(_btn("Save these figures…", self.save_figs, tip="PNG + SVG into the results folder"))
        f.addRow(h)
        for sig in (self.sp_lam_bin.valueChanged, self.cmb_lam_mode.currentIndexChanged, self.cb_lam_dend.toggled,
                    self.cb_lam_axon.toggled, self.cb_lam_smooth.toggled, self.sp_sholl_step.valueChanged,
                    self.sp_depth_bin.valueChanged, self.sp_shells.valueChanged):
            sig.connect(lambda *_: self._redraw_soon())
        box.setVisible(False)
        self.opt_box = box
        self._wheel_guard = _WheelGuard(self)
        _guard_wheels(box, self._wheel_guard)
        L.addWidget(box)
        self.tabs = QTabWidget(); L.addWidget(self.tabs, 1)
        self.setMinimumWidth(480)
        self.ctx = None
        self._figs = {}

    def _redraw_soon(self):
        from qtpy.QtCore import QTimer
        if not hasattr(self, "_rt"):
            self._rt = QTimer(self); self._rt.setSingleShot(True); self._rt.timeout.connect(self.redraw)
        self._rt.start(350)

    def _plot_figures(self):
        """Figures that depend on the plot options."""
        c = self.ctx
        figs = {}
        if c["b"]:
            figs["laminar_percent"] = analysis.laminar_figure(
                c["neuron"], c["surf"], c["b"], c["name"], c["dend"], c["axon"], step=self.sp_lam_bin.value(),
                mode="um" if self.cmb_lam_mode.currentIndex() == 1 else "percent",
                show_dendrites=self.cb_lam_dend.isChecked(), show_axon=self.cb_lam_axon.isChecked(),
                smooth=self.cb_lam_smooth.isChecked())
        st = self.sp_sholl_step.value()
        figs["sholl_3d"] = analysis.sholl_profile_figure(c["neuron"], st, c["dend"], c["axon"], c["name"])
        figs["sholl_shells"] = analysis.sholl_3d_figure(c["neuron"], st, c["dend"], n_shells=self.sp_shells.value())
        if c["b"]:
            figs["sholl_depth"] = analysis.sholl_depth_figure(c["neuron"], c["surf"], c["b"], st,
                                                              float(self.sp_depth_bin.value()), c["dend"])
        return figs

    def redraw(self):
        if not self.ctx:
            return
        cur = self.tabs.currentIndex()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._figs.update(self._plot_figures())
            self._show(self._figs, self._summary)
        finally:
            QApplication.restoreOverrideCursor()
        self.tabs.setCurrentIndex(min(cur, self.tabs.count() - 1))

    def save_figs(self):
        if not self.ctx:
            return
        out = self.ctx["out"]
        os.makedirs(out, exist_ok=True)
        tag = f"bin{self.sp_lam_bin.value()}_shell{self.sp_sholl_step.value():g}"
        for k, fig in self._figs.items():
            base = os.path.join(out, f"{self.ctx['name']}_{k}_{tag}")
            fig.savefig(storage.out(base + ".png"), dpi=200); fig.savefig(storage.out(base + ".svg"))
        QMessageBox.information(self, "Saved", f"{len(self._figs)} figures saved to\n{out}")

    def show_figures(self, figs, text, titles=None):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        titles = titles or {}
        self.tabs.clear()
        self._canvases = []
        for k, fig in figs.items():
            page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(0, 0, 0, 0)
            cv = FigureCanvasQTAgg(fig)
            lay.addWidget(NavigationToolbar2QT(cv, page)); lay.addWidget(cv, 1)
            self._canvases.append(cv)
            self.tabs.addTab(page, titles.get(k, k))
        t = QTextEdit(); t.setReadOnly(True)
        t.setHtml(text) if text.lstrip().startswith("<") else t.setPlainText(text)
        self.tabs.addTab(t, "Summary")

    @staticmethod
    def _summary_html(s: dict) -> str:
        def row(label, key, fmt="{:.1f}", unit=""):
            v = s.get(key)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return ""
            try:
                txt = fmt.format(v)
            except (ValueError, TypeError):
                txt = str(v)
            return f"<tr><td>{label}</td><td align='right'><b>{txt}</b> {unit}</td></tr>"
        pct = lambda k: (s.get(k) * 100 if isinstance(s.get(k), (int, float)) and np.isfinite(s.get(k)) else None)
        lam = "".join(f"<tr><td>{lab}</td><td align='right'><b>{pct(f'dend_frac_{k}'):.0f}%</b>"
                      + (f" &nbsp;/&nbsp; {pct(f'axon_frac_{k}'):.0f}%" if pct(f'axon_frac_{k}') is not None else "")
                      + "</td></tr>"
                      for k, lab in (("above_surface", "above surface"), ("0_25pct", "0–25 %"), ("25_50pct", "25–50 %"),
                                     ("50_75pct", "50–75 %"), ("75_100pct", "75–100 %"), ("below_CR", "below CR border"))
                      if pct(f"dend_frac_{k}") is not None)
        h = [f"<h3>{s.get('cell', '')}</h3>",
             "<p><i>Depth axis is perpendicular to the brain surface; 0 % = surface, 100 % = CR border.</i></p>",
             "<h4>Position</h4><table width='100%'>",
             row("Soma depth", "soma_depth_um", unit="µm"),
             row("Soma depth (% of CR)", "soma_normalized_depth", "{:.0%}"),
             row("CR border depth", "CR_border_depth_um", unit="µm"),
             row("Type hint", "type_hint", "{}"), row("Manual label", "manual_type", "{}"), "</table>"]
        if lam:
            h += ["<h4>Laminar distribution (dendrite / axon length)</h4><table width='100%'>", lam, "</table>"]
        else:
            h += ["<p style='color:#c60'>No CR border depth - measure it in tab 3 (or type it) to get the % distribution.</p>"]
        h += ["<h4>Dendrites</h4><table width='100%'>",
              row("Total length", "dendrite_length_um", unit="µm"), row("Primary dendrites", "dend_n_primary", "{:d}"),
              row("Branches", "dend_n_branches", "{:d}"), row("Branch points", "dend_n_branch_points", "{:d}"),
              row("Tips", "dend_n_tips", "{:d}"), row("Max branch order", "dend_max_branch_order", "{:d}"),
              row("Mean branch length", "dend_mean_branch_length_um", unit="µm"),
              row("Tortuosity", "dend_tortuosity", "{:.2f}"), row("Mean bifurcation angle", "dend_mean_bifurcation_angle_deg", unit="°"),
              row("Field width (parallel to surface)", "dend_width_major_um", unit="µm"),
              row("Field height (perpendicular)", "dend_height_um", unit="µm"),
              row("Vertical aspect (height/width)", "dend_vertical_aspect", "{:.2f}"),
              row("Dendrite above soma", "dend_dorsal_fraction", "{:.0%}"),
              row("Sholl peak", "sholl_peak_intersections", "{:d}", "intersections"),
              row("Sholl peak radius", "sholl_peak_radius_um", unit="µm"), "</table>",
              "<h4>Soma</h4><table width='100%'>", row("Diameter", "soma_diameter_um", unit="µm"),
              row("Source", "soma_size_source", "{}"), "</table>",
              "<h4>Axon / AIS</h4><table width='100%'>", row("Axon length", "axon_length_um", unit="µm"),
              row("Axon branches", "axon_n_branches", "{:d}"),
              row("AIS length", "ais_length_um", unit="µm"), row("Axon origin", "ais_origin", "{}"),
              row("AIS start from soma exit", "ais_start_dist_soma_exit_um", unit="µm"),
              row("Axon leaves dendrite at", "acd_branch_dist_soma_exit_um", unit="µm from soma"), "</table>"]
        return "".join(h)

    TITLES = {"view3d": "3D view", "laminar_percent": "Laminar %", "laminar_profile": "Side view",
              "sholl_3d": "Sholl profile", "sholl_shells": "Sholl 3D", "sholl_depth": "Sholl × depth",
              "depth_histograms": "Depth (µm)"}
    ORDER = ["view3d", "laminar_percent", "laminar_profile", "sholl_3d", "sholl_shells", "sholl_depth",
             "depth_histograms"]

    def _show(self, figs, summary):
        ordered = {k: figs[k] for k in self.ORDER if k in figs}
        self.show_figures(ordered, summary, self.TITLES)

    def show_results(self, res, ctx=None):
        figs = dict(res["figures"])
        if res.get("view3d") is not None:
            figs["view3d"] = res["view3d"]
        self._summary = self._summary_html(res["summary"].iloc[0].to_dict())
        self.ctx = ctx
        self._figs = figs
        self.opt_box.setVisible(ctx is not None)
        if ctx is not None:
            self._figs.update(self._plot_figures())
        self._show(self._figs, self._summary)


def dock_widget(viewer, w):
    dock = viewer.window.add_dock_widget(w, name="Arborist", area="right")
    w.main_dock = dock
    w.db = DatabaseWidget(w)
    w.db_dock = viewer.window.add_dock_widget(w.db, name="Database", area="right")
    try:  # results and database as tabs next to the controls
        qw = viewer.window._qt_window
        qw.tabifyDockWidget(dock, w.results_dock)
        qw.tabifyDockWidget(dock, w.db_dock)
        w.db_dock.raise_()
    except Exception:
        pass
    w.db.restore_root()
    return dock


# ================================================================ database dock
class DatabaseWidget(QWidget):
    COLORS = {True: "#1f5f2a", False: "#7a2020", None: None}

    def __init__(self, main: "SCMorphWidget"):
        super().__init__()
        self.main = main
        self.root = ""
        self.rows: list = []
        self.dbdata = {"notes": {}, "cache": {}}
        L = QVBoxLayout(self)
        h = QHBoxLayout()
        self.ed_root = QLineEdit(); self.ed_root.setPlaceholderText("data root, e.g. D:\\SC cells")
        h.addWidget(self.ed_root, 1)
        h.addWidget(_btn("…", self._pick_root))
        L.addLayout(h)
        h = QHBoxLayout()
        h.addWidget(_btn("⟳ Rescan", lambda: self.main._guard(self.rescan)))
        h.addWidget(_btn("Open selected cell", self.open_selected))
        h.addWidget(_btn("Export table…", self.export))
        self.btn_float = _btn("⇱ Big window", self._toggle_float, checkable=True,
                              tip="Show the database in its own resizable window")
        h.addWidget(self.btn_float)
        L.addLayout(h)
        self.cb_incomplete = QCheckBox("Show only rows with something missing")
        self.cb_incomplete.toggled.connect(self._fill)
        L.addWidget(self.cb_incomplete)
        self.lbl = _lbl("Choose the folder that contains the experiment-day folders.")
        L.addWidget(self.lbl)
        self.tbl = QTableWidget(0, len(dbm.COLUMNS))
        self.tbl.setHorizontalHeaderLabels(dbm.SHORT)
        for j, full in enumerate(dbm.COLUMNS):
            self.tbl.horizontalHeaderItem(j).setToolTip(full)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl.setSortingEnabled(False)
        self.tbl.cellDoubleClicked.connect(self._on_double)
        # single click: show a z-projection of the cell (reconstruction, else the stitch) on the canvas
        from qtpy.QtCore import QTimer
        self._click_timer = QTimer(self); self._click_timer.setSingleShot(True)
        self._click_timer.timeout.connect(self._preview_selected)
        self.tbl.cellClicked.connect(self._on_click)
        self.tbl.itemChanged.connect(self._note_changed)
        self.tbl.viewport().installEventFilter(self)     # hover preview of the fill
        L.addWidget(self.tbl, 1)
        L.addWidget(_lbl("<i>Double-click a row to open that cell (63x + its 20x CR stacks + SWC, or its "
                         "saved project). The Notes column is editable and saved in "
                         f"{dbm.DB_NAME} in the root folder.</i>"))
        self.setMinimumWidth(440)

    # ------------------------------------------------------------------
    def _settings(self):
        from qtpy.QtCore import QSettings
        st = QSettings("Arborist", "Arborist")
        if not st.allKeys():                       # first start after the rename: take over SC_morph's settings
            old = QSettings("SC_morph", "SC_morph")
            for k in old.allKeys():
                st.setValue(k, old.value(k))
        return st

    def restore_root(self):
        r = self._settings().value("db_root", "")
        if r and os.path.isdir(r):
            self.ed_root.setText(r)
            self.main._guard(self.rescan, True)

    def _pick_root(self):
        d = QFileDialog.getExistingDirectory(self, "Data root folder", self.ed_root.text())
        if d:
            self.ed_root.setText(d)
            self.main._guard(self.rescan)

    def refresh_soon(self, delay_ms=4000):
        """Rescan a little later, and only while the table is on screen; otherwise when it is next shown.
        (A rescan reads every cell folder - too slow to do after each autosave on a USB stick.)"""
        from qtpy.QtCore import QTimer
        self._stale = True
        if not hasattr(self, "_refresh_timer"):
            self._refresh_timer = QTimer(self); self._refresh_timer.setSingleShot(True)
            self._refresh_timer.timeout.connect(self._refresh_if_visible)
        self._refresh_timer.start(delay_ms)

    def _refresh_if_visible(self):
        if getattr(self, "_stale", False) and self.isVisible() and not self.visibleRegion().isEmpty():
            self.rescan(quiet=True)

    def showEvent(self, ev):
        super().showEvent(ev)
        if getattr(self, "_stale", False):
            from qtpy.QtCore import QTimer
            QTimer.singleShot(0, self._refresh_if_visible)

    def rescan(self, quiet=False):
        self._stale = False
        root = self.ed_root.text().strip()
        if not root or not os.path.isdir(root):
            if not quiet:
                self.main.msg("Choose an existing data root folder.", error=True)
            return
        if root != self.root:
            self.dbdata = dbm.load_db(root)
        self.root = root
        self._settings().setValue("db_root", root)
        self.rows = dbm.scan(root, self.dbdata["cache"])
        try:
            dbm.save_db(root, self.dbdata)
        except OSError as e:
            print("database not saved:", e)
        self._fill()

    def _incomplete(self, r):
        v = dbm.row_cells(r)[3:]
        return any(x is False for x in v) or not r.get("side") or not r.get("cell")

    def _fill(self, *_):
        self.tbl.blockSignals(True)
        shown = [r for r in self.rows if not self.cb_incomplete.isChecked() or self._incomplete(r)]
        self.tbl.setRowCount(len(shown))
        for i, r in enumerate(shown):
            vals = dbm.row_cells(r)
            tips = dbm.tooltips(r)
            for j, v in enumerate(vals):
                if v is True:
                    txt = "✓"
                elif v is False:
                    txt = "✗"
                elif v is None:
                    txt = ""
                else:
                    txt = str(v)
                it = QTableWidgetItem(txt)
                it.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                it.setData(Qt.UserRole, r["key"])
                col = self.COLORS.get(v) if isinstance(v, bool) or v is None else (self.COLORS[True] if j == 5 else None)
                if col:
                    from qtpy.QtGui import QColor
                    it.setBackground(QColor(col))
                tip = tips.get(j)
                if tip:
                    it.setToolTip(tip)
                self.tbl.setItem(i, j, it)
            note = QTableWidgetItem(self.dbdata["notes"].get(r["key"], ""))
            note.setData(Qt.UserRole, r["key"])
            self.tbl.setItem(i, len(dbm.COLUMNS) - 1, note)
        self.tbl.blockSignals(False)
        hh = self.tbl.horizontalHeader()
        for j, wmax in ((0, 130), (2, 70), (5, 150)):      # long day / file names: elide, full text on hover
            hh.setSectionResizeMode(j, QHeaderView.Interactive)
            self.tbl.setColumnWidth(j, min(wmax, max(60, self.tbl.sizeHintForColumn(j))))
        for i in range(self.tbl.rowCount()):
            it = self.tbl.item(i, 0)
            if it:
                it.setToolTip(it.text())
        cells = [r for r in self.rows if r.get("cell")]
        n = lambda k: sum(1 for r in cells if not r.get(k))
        sides_wo_cell = sum(1 for r in self.rows if r.get("side") and not r.get("cell"))
        no_cr = sum(1 for r in self.rows if r.get("side") and not r.get("cr"))
        no_10x = len({(r["day"], r["section"]) for r in self.rows if not r.get("overview")})
        self.lbl.setText(
            f"<b>{len(cells)}</b> cells in {len({(r['day'], r['section']) for r in self.rows})} sections. "
            f"To scan: 10x {no_10x} section(s), 20x CR {no_cr} side(s), 63x {sides_wo_cell} side(s). "
            f"AnkG {n('ankg')} cell(s). To do: tracing {n('swc')}, AIS {n('ais')}, surface {n('surface')}, "
            f"CR border {n('cr_done')}, analysis {n('analyzed')}.")

    def eventFilter(self, obj, ev):
        from qtpy.QtCore import QEvent, QUrl
        from qtpy.QtWidgets import QToolTip
        if obj is self.tbl.viewport() and ev.type() == QEvent.ToolTip:
            idx = self.tbl.indexAt(ev.pos())
            if idx.isValid() and idx.column() in (5, 7, 8):
                it = self.tbl.item(idx.row(), idx.column())
                r = next((x for x in self.rows if it and x["key"] == it.data(Qt.UserRole)), None)
                if r and (r.get("fills") or r.get("composite")):
                    QApplication.setOverrideCursor(Qt.WaitCursor)
                    try:
                        png = dbm.make_preview(r)
                    except Exception as ex:
                        png = None
                        print("preview failed:", ex)
                    finally:
                        QApplication.restoreOverrideCursor()
                    if png:
                        parts = ", ".join(f"{k}: {os.path.basename(p)}" for k, p in (r.get("fills") or {}).items())
                        html = (f"<b>{r.get('cell_name', '')}</b><br><img src='{QUrl.fromLocalFile(png).toString()}'>"
                                f"<br><span style='color:#bbb'>grey = soma · red = dendrites · green = axon</span>"
                                f"<br><small>{parts or os.path.basename(r.get('composite') or '')}</small>")
                        QToolTip.showText(ev.globalPos(), html, self.tbl)
                        return True
        return super().eventFilter(obj, ev)

    def _toggle_float(self, on):
        dock = getattr(self.main, "db_dock", None)
        if dock is None:
            return
        dock.setFloating(on)
        if on:
            dock.resize(1400, 650)
            dock.raise_()

    def _note_changed(self, item):
        if item.column() != len(dbm.COLUMNS) - 1 or not self.root:
            return
        self.dbdata["notes"][item.data(Qt.UserRole)] = item.text()
        try:
            dbm.save_db(self.root, self.dbdata)
        except OSError as e:
            self.main.msg(f"Could not save notes: {e}", error=True)

    def selected_row(self):
        items = self.tbl.selectedItems()
        if not items:
            return None
        key = items[0].data(Qt.UserRole)
        return next((r for r in self.rows if r["key"] == key), None)

    def _on_click(self, r, c):
        if c == len(dbm.COLUMNS) - 1:            # Notes column: editing, no preview
            return
        self._click_timer.start(QApplication.doubleClickInterval() + 30)

    def _on_double(self, r, c):
        self._click_timer.stop()
        if c != len(dbm.COLUMNS) - 1:
            self.open_selected()

    def _preview_selected(self):
        r = self.selected_row()
        if r is not None and r.get("cell"):
            self.main.preview_cell(r)

    def open_selected(self):
        r = self.selected_row()
        if r is None:
            self.main.msg("Select a row first.", error=True); return
        self.main._guard(self.main.open_from_db, r)
        if getattr(self.main, "main_dock", None):
            self.main.main_dock.raise_()

    def export(self):
        if not self.rows:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Export", os.path.join(self.root, "Arborist_status.csv"),
                                           "CSV (*.csv);;Excel (*.xlsx)")
        if not p:
            return
        t = dbm.to_table(self.rows, self.dbdata["notes"])
        if p.lower().endswith(".xlsx"):
            t.to_excel(storage.out(p), index=False)
        else:
            t.to_csv(storage.out(p), index=False)
        self.main.msg(f"Exported {p}")


ASSETS = os.path.join(os.path.dirname(__file__), "assets")


def _set_icon(viewer):
    from qtpy.QtGui import QIcon
    ico = os.path.join(ASSETS, "Arborist.ico" if os.name == "nt" else "Arborist.png")
    if os.path.exists(ico):
        icon = QIcon(ico)
        QApplication.instance().setWindowIcon(icon)
        viewer.window._qt_window.setWindowIcon(icon)


def _relocate(path, base, up_levels=5, tail_parts=6):
    """A file saved with another drive letter / root: find it by the end of its path, starting from the
    project folder and its parents (e.g. D:/SC cells/day/Section 1/x.czi -> E:/SC_cells/day/Section 1/x.czi)."""
    parts = [p for p in os.path.normpath(str(path)).replace("\\", "/").split("/") if p and not p.endswith(":")]
    anc = os.path.abspath(base)
    for _ in range(up_levels + 1):
        for k in range(1, min(tail_parts, len(parts)) + 1):
            cand = os.path.join(anc, *parts[-k:])
            if os.path.exists(cand):
                return cand
        nxt = os.path.dirname(anc)
        if nxt == anc:
            break
        anc = nxt
    return None


def _run_catch(fn, step):
    import traceback
    try:
        return {"value": fn(step)}
    except Exception as ex:                       # handed back to the main thread
        return {"error": ex, "trace": traceback.format_exc()}


def _install_close_hook(viewer, w):
    """Closing the window: save, copy to the USB stick, and only then close."""
    from qtpy.QtCore import QObject, QEvent

    class _CloseFilter(QObject):
        def eventFilter(self, obj, ev):
            if ev.type() == QEvent.Close and not getattr(w, "_closing_ok", False):
                if not w._on_app_close():
                    ev.ignore()
                    return True
                w._closing_ok = True
            return False
    qwin = viewer.window._qt_window
    w._close_filter = _CloseFilter(qwin)
    qwin.installEventFilter(w._close_filter)


def main():
    if os.name == "nt":   # own taskbar entry + icon instead of python.exe's
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("PTE.Arborist")
        except Exception:
            pass
    viewer = napari.Viewer(title=f"Arborist {__version__}")
    try:
        _set_icon(viewer)
    except Exception:
        pass
    w = SCMorphWidget(viewer)
    dock_widget(viewer, w)
    _install_close_hook(viewer, w)
    napari.run()
    return viewer, w
