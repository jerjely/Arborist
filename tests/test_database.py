"""Fake experiment tree in the lab's real layout -> database scan -> open cells -> AIS on AnkG stack."""
import os
import shutil
import sys

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from arborist import storage
from arborist import database as dbm

HERE = os.path.dirname(__file__)
D = os.path.join(HERE, "synthetic")
ROOT = os.path.join(D, "SC cells")
sys.path.insert(0, HERE)
import czi_sample
CZI = czi_sample.get()                                     # real Zeiss file, 20x objective in metadata
if CZI is None:
    if os.environ.get("CI"):
        raise SystemExit("the CZI test file could not be downloaded")
    print("SKIPPED: no CZI test file (offline?) - see tests/czi_sample.py")
    sys.exit(0)
shutil.rmtree(ROOT, ignore_errors=True)


def tif(path, px=0.1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tifffile.imwrite(path, np.zeros((3, 1, 20, 20), np.uint16), imagej=True, resolution=(1 / px, 1 / px),
                     metadata={"spacing": 1.0, "unit": "um", "axes": "ZCYX"})


def cp(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)


day = os.path.join(ROOT, "2026.06.02_PV_GF_Patch_Cells")
s1 = os.path.join(day, "Section 1")
tif(os.path.join(s1, "whole section", "Section1_whole_10x.tif"), 0.9)
cp(CZI, os.path.join(s1, "SC", "Section1_SC_20x.czi"))
L = os.path.join(s1, "Left")
stem = "202606.02_PV_GF_Patch_Cells_Slice1_Dapi_NBCy5_Left_Cell_63x_Stitch"
cp(CZI, os.path.join(L, stem + ".czi"))
cp(CZI, os.path.join(L, "2026.06.02_PV_GFP_Patch_cells_section3_left_cell_ankyrin.czi"))
cp(os.path.join(D, "test.swc"), os.path.join(L, stem + ".swc"))
cp(os.path.join(D, "test.swc"), os.path.join(L, stem + "2.swc"))
for extra in (stem + "_recon.tif", "axon.tif", "dendrites.tif", "full.tif"):
    tif(os.path.join(L, extra), 0.1)
open(os.path.join(L, "SURFACE.txt"), "w").write("x")
os.makedirs(os.path.join(s1, "Right"), exist_ok=True)              # not scanned yet
# Section 2: 10x in the section root, no SC folder, TIFF-only side with its own 20x
s2 = os.path.join(day, "Section 2")
tif(os.path.join(s2, "Section2_10x.tif"), 0.9)
cp(os.path.join(D, "synthetic_cell.tif"), os.path.join(s2, "Right", "cellX_63x.tif"))
cp(os.path.join(D, "synthetic_lowmag.tif"), os.path.join(s2, "Right", "cellX_CR_20x.tif"))
cp(os.path.join(D, "test.swc"), os.path.join(s2, "Right", "cellX_63x.swc"))
os.makedirs(os.path.join(ROOT, "2026.07.10_PV_GF_Patch_Cells", "Section 1"), exist_ok=True)   # empty day
# section 4 layout: 20x SC (both hemispheres) and 10x whole brain directly in the section folder
s4 = os.path.join(ROOT, "2026.10.29_Bl6_Patch_cells", "section 4")
cp(CZI, os.path.join(s4, "2026.10.29_Bl6_Patch_cells_section4_SC_20x.czi"))          # 20x by metadata
open(os.path.join(s4, "2026.10.29_Bl6_Patch_cells_section4_SC_20x.queue"), "w").write("x")
tif(os.path.join(s4, "2026.10.29_Bl6_Patch_cells_section4_whole_brain_10x.tif"), 0.9)
cp(CZI, os.path.join(s4, "left", "2026.10.29_Bl6_Patch_cells_section4_left_63x_Stitch.czi"))
os.makedirs(os.path.join(s4, "right"), exist_ok=True)
# Bl6 example: 20x inside the side folder, cell.swc, fill projections soma/dendrites/axon.tif
bl = os.path.join(ROOT, "2026.10.29_Bl6_Patch_cells", "Section 1", "Right")
for n in ("2026.10.29_Bl6_Patch_cells_section1_right_63x_ankyrin.czi",
          "2026.10.29_Bl6_Patch_cells_section1_right_SC_20x.czi",
          "202610.29_Bl6_Patch_cells_section1_right_63x_Stitch.czi"):
    cp(CZI, os.path.join(bl, n))
cp(os.path.join(D, "test.swc"), os.path.join(bl, "cell.swc"))
yy, xx = np.mgrid[0:400, 0:400]
soma_img = (((yy - 200) ** 2 + (xx - 200) ** 2) < 400).astype(np.uint16) * 3000
dend_img = np.zeros((400, 400), np.uint16); dend_img[100:200, 198:202] = 3000; dend_img[150:154, 100:200] = 3000
axon_img = np.zeros((400, 400), np.uint16); axon_img[200:390, 205:208] = 3000
for n, im in (("soma.tif", soma_img), ("dendrites.tif", dend_img), ("axon.tif", axon_img)):
    tifffile.imwrite(os.path.join(bl, n), im)
tifffile.imwrite(os.path.join(bl, "STD_Composite (RGB).tif"), np.zeros((50, 50, 3), np.uint8), photometric="rgb")
from make_synthetic import f_surface
with open(os.path.join(s2, "Right", "SURFACE.txt"), "w") as fh:
    fh.write(" \tX\tY\tSlice\n")
    k = 1
    for z in range(0, 60, 6):
        for x in np.linspace(5, 145, 12):
            fh.write(f"{k}\t{x:.3f}\t{f_surface(x, z):.3f}\t{z + 1}\n"); k += 1

rows = dbm.scan(ROOT, {})
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(dbm.to_table(rows, {}).drop(columns=["Notes", "Day"]).to_string())
r = {(x["day"][:10], x["section"], x.get("side")): x for x in rows}
a = r[("2026.06.02", "Section 1", "left")]
assert a["cell"].endswith(stem + ".czi") and a["swc"].endswith(stem + ".swc") and a["recon"]
assert a["ankg"] and a["overview"] and a["cr"] and a["cr_source"] == "SC folder"
assert {os.path.basename(x) for x in a["extras"]} == {"axon.tif", "dendrites.tif", "full.tif"}
b = r[("2026.06.02", "Section 1", "right")]
assert b["cell"] is None and b["cr"]                                 # SC folder covers both sides
c = r[("2026.06.02", "Section 2", "right")]
assert c["cell"].endswith("cellX_63x.tif") and c["cr_source"] == "side folder" and c["overview"]
assert not c["ankg"] and not c["recon"]
assert c["surface_txt"] and c["surface"]                         # Fiji SURFACE.txt counts as marked
assert r[("2026.07.10", "Section 1", "")]["cell"] is None
for side in ("left", "right"):
    s4row = r[("2026.10.29", "section 4", side)]
    print("section 4", side, "| 20x:", [os.path.basename(x) for x in s4row["cr"]], s4row["cr_source"],
          "| 10x:", [os.path.basename(x) for x in s4row["overview"]])
    assert len(s4row["cr"]) == 1 and s4row["cr"][0].endswith("SC_20x.czi") and s4row["cr_source"] == "section folder"
    assert len(s4row["overview"]) == 1 and "whole_brain_10x" in s4row["overview"][0]
bl_row = r[("2026.10.29", "Section 1", "right")]
assert bl_row["cell"].endswith("63x_Stitch.czi") and bl_row["swc"].endswith("cell.swc") and bl_row["ankg"]
assert bl_row["cr"] and bl_row["cr_source"] == "side folder" and bl_row["recon"]
assert set(bl_row["fills"]) == {"soma", "dendrites", "axon"}
png = dbm.make_preview(bl_row)
from PIL import Image
im = np.asarray(Image.open(png).convert("RGB")).astype(int)
sc = 360 / 400
px = lambda y, x: im[int(y * sc), int(x * sc)]
print("preview", png, "soma", px(200, 200), "dend", px(120, 200), "axon", px(350, 206))
assert px(200, 200).min() > 150                                 # soma grey/white
assert px(120, 200)[0] > 150 and px(120, 200)[1] < 100          # dendrite red
assert px(350, 206)[1] > 150 and px(350, 206)[0] < 100          # axon green
shutil.copy(png, os.path.join(HERE, "fill_preview_test.png"))
print("scan OK")

# ---------------------------------------------------------------- GUI
import napari
from qtpy.QtWidgets import QMessageBox
from qtpy.QtCore import Qt
from arborist.gui import SCMorphWidget, dock_widget, xyz2vox
from make_synthetic import f_surface

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)
os.makedirs(os.path.join(HERE, "shots"), exist_ok=True)
v = napari.Viewer(); v.window._qt_window.resize(1800, 1000)
w = SCMorphWidget(v); dock_widget(v, w)
w.db.ed_root.setText(ROOT); w.db.rescan()
print(w.db.lbl.text())
# hover over the Bl6 cell -> preview tooltip
from qtpy.QtCore import QEvent, QPoint
from qtpy.QtGui import QHelpEvent
w.db_dock.raise_(); QApplication = __import__("qtpy.QtWidgets", fromlist=["QApplication"]).QApplication
QApplication.processEvents()
row_i = next(i for i in range(w.db.tbl.rowCount()) if w.db.tbl.item(i, 0).text().startswith("2026.10.29"))
rect = w.db.tbl.visualItemRect(w.db.tbl.item(row_i, 5))
ev = QHelpEvent(QEvent.ToolTip, rect.center(), w.db.tbl.viewport().mapToGlobal(rect.center()))
assert w.db.eventFilter(w.db.tbl.viewport(), ev), "no preview tooltip"
print("hover preview OK")
w.db_dock.raise_()
v.screenshot(os.path.join(HERE, "shots", "database.png"), canvas_only=False)

# single click: z-projection preview (recon when there is one, else the stitch) on the canvas
import time as _t
def click_preview(row):
    w.preview_cell(row)
    for _ in range(400):
        QApplication.processEvents(); _t.sleep(0.02)
        if getattr(w, "_pv_layer", None) is not None and row.get("cell_name", "") in w._pv_layer.name:
            break
    return w._pv_layer
lay = click_preview(bl_row)
assert lay is not None and lay.rgb and "fills" in lay.name, lay        # soma/axon/dendrites.tif exist -> fills
px = lay.data.reshape(-1, 3).astype(int)
lit = px[px.max(1) > 40]
red = ((lit[:, 0] > 2 * lit[:, 1]) & (lit[:, 0] > 2 * lit[:, 2])).mean()
green = ((lit[:, 1] > 2 * lit[:, 0]) & (lit[:, 1] > 2 * lit[:, 2])).mean()
print(f"fills preview: {red:.0%} red (dendrites), {green:.0%} green (axon)")
assert red > 0.05 and green > 0.05
print("preview:", lay.name)
no_recon = next(x for x in w.db.rows if x.get("cell") and dbm.projection_source(x)[0] == "stitch")
lay = click_preview(no_recon)
assert "stitch" in lay.name and [l.name for l in v.layers] == [lay.name]
assert dbm.cached_projection(no_recon)                      # cached: next click is instant
print("preview without recon:", lay.name)

# Section 1 / Left: Stitch + SC 20x + AnkG scan; AIS measured on the AnkG stack
w.open_from_db(a)
assert getattr(w, "_pv_layer", None) is None and not any(l.name.startswith("PREVIEW") for l in v.layers)
print([(e.name, e.role) for e in w.entries])
names = lambda: [l.name.split(" ")[0] for l in v.layers]
print("layer list (cell view):", sorted(set(names())))
assert set(names()) == {"CELL"}
w.show_role("ankg"); assert set(names()) == {"ANKG"}, names()
w.show_role("lowmag"); assert set(names()) == {"LOW"}
assert w.view_btns["overview"].isEnabled() and not any(x.role == "overview" for x in w.entries)
w.show_role("overview"); assert set(names()) == {"10X"}, names()       # loaded on demand
w.show_role("cell")
# wheel: plain = next slice, ctrl = zoom
from arborist.gui import wheel_handler
class Ev:
    def __init__(s, mods, dy):
        s.modifiers = mods; s.delta = (0, dy); s.handled = False
        s.position = tuple(v.dims.point); s.native = type("N", (), {"inverted": lambda self: False})()
z0 = v.dims.current_step[0]; zoom0 = v.scene.camera.zoom
ev = Ev((), -1); wheel_handler(v, ev); assert v.dims.current_step[0] == z0 + 1 and ev.handled
ev = Ev(("Control",), 1); wheel_handler(v, ev); assert v.scene.camera.zoom > zoom0 * 1.1 and ev.handled
zoom1 = v.scene.camera.zoom
ev = Ev(("Meta",), 1); wheel_handler(v, ev); assert v.scene.camera.zoom > zoom1 * 1.1   # Mac Control key
print("wheel OK")
# brightness / contrast
lay = w._disp_layer; before = tuple(lay.contrast_limits)
w._disp_auto(); after = tuple(lay.contrast_limits)
w.sl_contrast.setValue((after[0], after[0] + 0.5 * (after[1] - after[0])))
assert abs(lay.contrast_limits[1] - (after[0] + 0.5 * (after[1] - after[0]))) < 1e-3
w.sl_gamma.setValue(0.7); assert abs(lay.gamma - 0.7) < 1e-6
print("contrast OK", before, "->", after)
ank = w.ankg_entry()
assert ank is not None and w.cell_entry().name == stem
ank.layers["soma_pt"].data = xyz2vox(np.array([[100, 100, 2]]), ank.voxel)
ank.layers["ais_pts"].data = xyz2vox(np.array([[100, 130, 2], [104, 140, 2.5], [106, 150, 3]]), ank.voxel)
w.cmb_ais_origin.setCurrentText("soma")
w.tabs.setCurrentIndex(4)                                         # AIS tab -> ankyrin view
assert w.active is ank and set(names()) == {"ANKG"}
w._ais_preview(); print("AIS:", w.lbl_ais.text())
print("NDIM", v.dims.ndim, [(l.name, l.ndim) for l in v.layers if l.ndim != 3]); v.screenshot(os.path.join(HERE, "shots", "ais_tab.png"), canvas_only=False)
w.cmb_ais_origin.setCurrentText("dendrite"); assert w.btn_mark_branch.isEnabled()
w.cmb_ais_origin.setCurrentText("soma"); assert not w.btn_mark_branch.isEnabled()
f, src = w.current_ais()
assert src == "AnkG stack" and abs(f["ais_start_dist_soma_exit_straight_um"] - 30) < 1e-6
# autosave of AIS marks (debounced timer -> call directly) incl. a route bend point
ank.layers["route_pts"].data = xyz2vox(np.array([[100, 115, 2]]), ank.voxel)
w._ais_update_and_save()
import json
pj = json.load(open(storage.read(dbm.project_path_for(a["cell"]))))
sd = next(s for s in pj["stacks"] if s["role"] == "ankg")
assert len(sd["ais_points_um"]) == 3 and len(sd["route_points_um"]) == 1 and len(sd["soma_point_um"]) == 1
print("autosave:", w.lbl_ais_saved.text())
f, _ = w.current_ais(); print("route:", f.get("route_method"), round(f["ais_start_dist_soma_exit_um"], 1))
w.db.rescan()
a2 = next(x for x in w.db.rows if x.get("cell") == a["cell"])
assert a2["ais"], "AIS status not picked up from project"

# pen: drag along a slice on the 63x stack -> evenly spaced surface points, undo removes the line
w.show_role("cell"); w.tabs.setCurrentIndex(1)
lay = w.cell_entry().layers["surface_pts"]; n0 = len(lay.data)
w.btn_mark_surf.setChecked(True); w.cb_pen.setChecked(True)
assert lay.mode == "pan_zoom" and w._camera().mouse_pan is False
class PE:
    def __init__(s): s.type = "mouse_press"; s.button = 1; s.position = (v.dims.point[0], 10.0, 10.0)
pe = PE(); gen = w._on_pen(v, pe); next(gen)
for x in np.linspace(10, 110, 60):                      # 100 um stroke
    pe.type = "mouse_move"; pe.position = (v.dims.point[0], 10.0 + 0.1 * x, x); next(gen)
pe.type = "mouse_release"
try:
    next(gen)
except StopIteration:
    pass
added = len(lay.data) - n0
print("pen added", added, "points for a ~100 um line at 5 um spacing")
assert 18 <= added <= 23 and np.allclose(lay.data[n0:, 0], lay.data[n0, 0])     # all on one slice
w.undo_mark(); assert len(lay.data) == n0
w.btn_mark_surf.setChecked(False); assert w._camera().mouse_pan is True
print("pen OK")

# re-traced in SNT after the project was saved -> reopening must use the new .swc
a3 = next(x for x in w.db.rows if x.get("cell") == a["cell"])
assert a3["project"]
with open(a3["swc"], "a") as fh:
    fh.write("9999 3 1 1 1 0.5 1\n")
w.open_from_db(a3)
assert (w.cell_entry().neuron.df["id"] == 9999).any(), "old tracing from the project was used"
print("swc reload OK")

# Section 2 / Right: full analysis -> row turns complete
w.open_from_db(c)
e = w.cell_entry()
assert e.surface is not None and len(e.layers["surface_pts"].data) == 120, "SURFACE.txt not imported"
print(w.lbl_surface.text())
assert e.surface.method == "slices"
w._autosave()
cell_dir = os.path.dirname(c["cell"])
for fn in ("cellX_63x_surface_points.csv", "cellX_63x_brain_surface.stl", "cellX_63x_SC_morph.json"):
    assert storage.exists(os.path.join(cell_dir, fn)), fn
print("autosaved next to cell:", w.lbl_saved.text())
# CR on the 20x (its own surface + CR points) and on the 63x (CR points vs the 63x surface)
w.tabs.setCurrentIndex(2); assert w.active.role == "lowmag"            # CR tab opens the low mag
lm = w.active
ang = np.deg2rad(20); nv = np.array([-np.sin(ang), np.cos(ang)]); tv = np.array([np.cos(ang), np.sin(ang)])
base = np.array([250, 150.])
lm.layers["surface_pts"].data = xyz2vox(np.array([np.r_[base + s * tv, 8.0] for s in np.linspace(-150, 150, 12)]), lm.voxel)
lm.layers["border_pts"].data = xyz2vox(np.array([np.r_[base + s * tv + 120 * nv, 8.0] for s in np.linspace(-120, 120, 8)]), lm.voxel)
w.measure_border()
w.rb_cr_high.setChecked(True); assert w.active.role == "cell"           # switch to the 63x
g_ = lambda x, z: np.array([-0.15, 1.0, -0.008 * (z - 30)]) / np.linalg.norm([-0.15, 1.0, -0.008 * (z - 30)])
crp = np.array([np.array([x, f_surface(x, z), z]) + 40 * g_(x, z) for z in (12, 24, 36) for x in (40, 75, 110)])
e.layers["border_pts"].data = xyz2vox(crp, e.voxel)
w.measure_border()
res = {x.role: round(x.border_result["mean"], 1) for x in w.entries if x.border_result}
print("CR measurements per image:", res, "-> used:", round(w.boundary_um(), 1))
assert abs(res["lowmag"] - 120) < 3 and abs(res["cell"] - 40) < 2 and abs(w.boundary_um() - 80) < 3
w.tbl_border.item(0, 0).setCheckState(Qt.Unchecked)                    # use only one of them
print("after unticking row 1:", round(w.boundary_um(), 1))
# untick everything, re-measure on the 20x -> the new measurement is used again
for i in range(w.tbl_border.rowCount()):
    w.tbl_border.item(i, 0).setCheckState(Qt.Unchecked)
assert w.boundary_um() is None and "none is ticked" in w.lbl_boundary.text()
w.show_role("lowmag"); w.measure_border()
assert abs(w.boundary_um() - 120) < 3, w.boundary_um()
print("re-measure is ticked again:", round(w.boundary_um(), 1))
w.rb_cr_low.setChecked(True)
# --- clearing
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
w.show_role("cell")
sp = e.layers["surface_pts"]; n_all = len(sp.data)
v.dims.set_point(0, 12.0); w.clear_slice("surface_pts")
print("clear slice:", n_all, "->", len(sp.data)); assert len(sp.data) == n_all - 12
w._autosave(); stl = os.path.join(os.path.dirname(c["cell"]), "cellX_63x_brain_surface.stl")
w.clear_surface_points()
assert len(sp.data) == 0 and e.surface is None and e.layers.get("mesh") is None and not storage.exists(stl)
assert not len(e.layers["surface_line"].data)
w.show_role("lowmag"); lm_ = w.active
w.clear_border_points()
assert not len(lm_.layers["border_pts"].data) and lm_.border_result is None
assert all(x.role != "lowmag" for x in w.entries if x.border_result)
print("clear surface / CR OK; table rows:", w.tbl_border.rowCount())
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)
# restore the surface for the analysis below
w.show_role("cell"); w._import_surface_txt(e, c["surface_txt"])
w.rb_manual.setChecked(True); w.sp_manual.setValue(120)
w.run_analysis()
w.db.rescan()
c2 = next(x for x in w.db.rows if x.get("cell") == c["cell"])
print("after analysis:", {k: c2[k] for k in ("surface", "cr_done", "analyzed")})
assert c2["surface"] and c2["cr_done"] and c2["analyzed"]
# analysed under another name (e.g. a name kept from an older project): still counted as done
import glob as _g, shutil as _sh
res_dir = os.path.dirname(storage.read(c2["results"]))
for f in _g.glob(os.path.join(res_dir, "*_summary.csv")):
    if not os.path.basename(f).startswith("all_cells"):
        _sh.move(f, os.path.join(res_dir, "some_other_name_summary.csv"))
w.db.rescan()
c3 = next(x for x in w.db.rows if x.get("cell") == c["cell"])
assert c3["analyzed"], "done not recognised under another name"
print("done under another name OK")
w.main_dock.raise_()
v.screenshot(os.path.join(HERE, "shots", "views.png"), canvas_only=False)
w.db_dock.raise_()
v.screenshot(os.path.join(HERE, "shots", "database_after.png"), canvas_only=False)
# pictures for the dendrogram come from the database rows
import pandas as _pd
th = w._cell_thumbnails(_pd.DataFrame([dict(cell=a["cell_name"], source_path=a["cell"]),
                                       dict(cell="unknown", source_path="")]))
assert th is not None and th[0] is not None and th[0].ndim == 3 and th[1] is None
print("dendrogram pictures OK")
w.db.btn_float.setChecked(True)
from qtpy.QtWidgets import QApplication; QApplication.processEvents()
w.db.grab().save(os.path.join(HERE, "shots", "database_window.png"))
v.close()
print("DATABASE OK")
