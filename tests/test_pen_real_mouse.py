import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import napari
from qtpy.QtCore import Qt, QPoint, QPointF, QEvent
from qtpy.QtGui import QMouseEvent, QKeyEvent
from qtpy.QtWidgets import QApplication, QMessageBox
from arborist.gui import SCMorphWidget, dock_widget
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)
D = os.path.join(os.path.dirname(__file__), 'synthetic')
v = napari.Viewer(); v.window._qt_window.resize(1600, 1000); v.window._qt_window.show()
w = SCMorphWidget(v); dock_widget(v, w); w.main_dock.raise_()
e = w.add_stack(os.path.join(D, 'synthetic_cell.tif'), 'cell')
w.tabs.setCurrentIndex(1)
w.btn_mark_surf.setChecked(True); w.cb_pen.setChecked(True)
canvas = v.window._qt_viewer.canvas.native
canvas.setFocus(); QApplication.processEvents()
W, H = canvas.width(), canvas.height()
def mouse(kind, x, y, buttons):
    p = QPointF(x, y); g = QPointF(canvas.mapToGlobal(QPoint(int(x), int(y))))
    t = {"press": QEvent.MouseButtonPress, "move": QEvent.MouseMove, "release": QEvent.MouseButtonRelease}[kind]
    btn = Qt.LeftButton if kind != "move" else Qt.NoButton
    QApplication.sendEvent(canvas, QMouseEvent(t, p, g, btn, buttons, Qt.NoModifier)); QApplication.processEvents()
def drag(x0, x1, y):
    mouse("press", x0, y, Qt.LeftButton)
    for x in np.linspace(x0, x1, 40):
        mouse("move", x, y + 0.1 * (x - x0), Qt.LeftButton)
    mouse("release", x1, y + 0.1 * (x1 - x0), Qt.NoButton)
lay = e.layers["surface_pts"]
center0 = tuple(w._camera().center)
import arborist.gui as G
orig = w._on_pen
seen = []
def spy(viewer, event):
    gen = orig(viewer, event)
    if gen is None: return
    seen.append(("press", tuple(np.round(event.position, 1)), event.type))
    try:
        next(gen)
        while True:
            yield
            seen.append((event.type, tuple(np.round(event.position, 1))))
            gen.send(None)
    except StopIteration:
        pass
v.mouse_drag_callbacks[v.mouse_drag_callbacks.index(w._on_pen)] = spy
drag(W * 0.3, W * 0.7, H * 0.3)
print(len(seen), seen[:3], seen[-3:]); print("zoom", w._camera().zoom, "W", W)
print("pen drag -> points:", len(lay.data), "| camera moved:", tuple(w._camera().center) != center0)
assert len(lay.data) >= 12 and tuple(w._camera().center) == center0
# after the surface is rebuilt (mesh layer added -> napari re-enables panning) the next stroke still draws
# drawing does not rebuild: the points are saved and the Build button turns orange
had = e.surface
w._marks_timer.stop(); w._apply_mark_changes(); QApplication.processEvents()
assert e.surface is had and e.surface_stale and "slice" in w.btn_build.text(), w.btn_build.text()
print("after a line: surface not rebuilt, button says:", w.btn_build.text())
w._build_and_save(); QApplication.processEvents()
assert e.surface is not None and e.layers.get("mesh") is not None and not e.surface_stale
assert w.btn_build.text() == "Build surface"
# CR line on the slice: one clean line for every surface method (Delaunay used to tangle)
w.boundary_um = lambda: 60.0
for m in (0, 2):
    w.cmb_method.setCurrentIndex(m); w._build_and_save(); w._draw_cr_now(); QApplication.processEvents()
    V = e.layers["border_line"].data
    seg = np.linalg.norm(V[:, 1, 1:] * e.voxel[1:], axis=1) if len(V) else np.zeros(1)
    print(f"method {w.METHODS[m]}: CR line {len(V)} segments, longest {seg.max():.1f} um")
    assert seg.max() < 10 and (len(V) > 5 or m == 2)      # never a tangle; Delaunay may show none
w.cmb_method.setCurrentIndex(0); w._build_and_save()
del w.boundary_um
n0 = len(lay.data)
drag(W * 0.3, W * 0.7, H * 0.45)
print("second stroke after rebuild -> points:", len(lay.data) - n0, "| camera moved:", tuple(w._camera().center) != center0)
assert len(lay.data) - n0 >= 12 and tuple(w._camera().center) == center0
assert len(e.layers["pen_preview"].data) == 0
# hold Space: drag pans instead of drawing
n1 = len(lay.data)
QApplication.sendEvent(canvas, QKeyEvent(QEvent.KeyPress, Qt.Key_Space, Qt.NoModifier, " ")); QApplication.processEvents()
print("space down:", w._space_down, "mouse_pan:", w._camera().mouse_pan)
drag(W * 0.5, W * 0.6, H * 0.6)
QApplication.sendEvent(canvas, QKeyEvent(QEvent.KeyRelease, Qt.Key_Space, Qt.NoModifier, " ")); QApplication.processEvents()
print("space drag -> new points:", len(lay.data) - n1, "| camera moved:", tuple(w._camera().center) != center0,
      "| after release mouse_pan:", w._camera().mouse_pan)
assert len(lay.data) == n1 and tuple(w._camera().center) != center0 and w._camera().mouse_pan is False
v.close(); print("REAL MOUSE OK")
