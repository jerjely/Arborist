"""Electrophysiology table: messy Excel -> matched rows -> merged features -> clustering + link dialog."""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from arborist import ephys
from arborist.morphometry import cluster_cells

rng = np.random.default_rng(3)
D = tempfile.mkdtemp()
root = "/data/SC_patch"

# ---- analysed cells (what all_cells_summary.csv holds)
cells = []
types = ["wide-field", "narrow-field", "horizontal"]
specs = []   # (date, section, side, cellno, type)
for k in range(12):
    date = pd.Timestamp(2026, 3, 10 + k // 4)
    sec, side, no = 1 + (k % 2), ["Left", "Right"][(k // 2) % 2], 1 + k % 3
    typ = types[k % 3]
    specs.append((date, sec, side, no, typ))
    day = date.strftime("%Y.%m.%d") if k < 6 else date.strftime("%Y%m%d")
    stem = f"Cell{no}_Stitch" if k != 5 else f"S{sec}_{side[0]}_c{no}_63x"
    m = {"wide-field": (300, 0.8), "narrow-field": (90, 0.3), "horizontal": (250, 0.1)}[typ]
    cells.append(dict(cell=f"{date:%m%d}_S{sec}{side[0]}_{stem}",
                      source_path=f"{root}/{day}/Section {sec}/{side}/{stem}.czi",
                      field_width_um=m[0] + rng.normal(0, 15), dend_frac_q2=m[1] + rng.normal(0, .05),
                      soma_depth_um=rng.normal(120, 10), manual_type=typ))
cells.append(dict(cell="0399_lonely", source_path=f"{root}/20260399/Section 9/Left/Cell9.czi",
                  field_width_um=100, dend_frac_q2=.5, soma_depth_um=100, manual_type=""))
cells = pd.DataFrame(cells)
master = os.path.join(D, "all_cells_summary.csv")
cells.to_csv(master, index=False)

# ---- the Excel, as people actually write them
rows = []
for i, (date, sec, side, no, typ) in enumerate(specs):
    e = {"wide-field": (-62, 180, 0.25), "narrow-field": (-70, 420, 0.05), "horizontal": (-58, 250, 0.4)}[typ]
    rows.append([date if i % 3 else date.strftime("%Y.%m.%d."), f"S{sec}" if i % 2 else sec,
                 {"Left": "B", "Right": "J"}[side] if i < 6 else side[0], f"cell {no}" if i % 2 else no,
                 str(round(e[0] + rng.normal(0, 1.5), 1)).replace(".", ","), e[1] + rng.normal(0, 20),
                 round(e[2] + rng.normal(0, .03), 3), "ok"])
rows.append([pd.Timestamp(2026, 4, 1), 3, "B", 4, "-65,0", 300, 0.2, "no image yet"])
hdr = ["Dátum", "Metszet", "Oldal", "Sejt", "Vrest", "Rin", "Sag ratio", "megjegyzés"]
units = ["", "", "", "", "mV", "MΩ", "", ""]
sheet = [["sSC patch clamp 2026 – Szarka"] + [None] * 7, [None] * 8, hdr, units] + rows
xl = os.path.join(D, "ephys.xlsx")
with pd.ExcelWriter(xl) as w:
    pd.DataFrame([["other sheet"]]).to_excel(w, sheet_name="notes", header=False, index=False)
    pd.DataFrame(sheet).to_excel(w, sheet_name="cells", header=False, index=False)

assert ephys.sheet_names(xl) == ["notes", "cells"]
t = ephys.read_table(xl, "cells")
print("columns:", list(t.columns))
assert "Vrest (mV)" in t.columns and "Rin (MΩ)" in t.columns and len(t) == 13
assert pd.api.types.is_numeric_dtype(t["Vrest (mV)"]), t["Vrest (mV)"].head()
ids = ephys.guess_id_columns(t)
print("guessed id columns:", ids)
assert set(ids) >= {"Dátum", "Metszet", "Oldal", "Sejt"} and "Rin (MΩ)" not in ids
feats = ephys.feature_columns(t, ids)
assert set(feats) == {"Vrest (mV)", "Rin (MΩ)", "Sag ratio"}, feats

m = ephys.auto_match(cells, t, ids)
ok = sum(m[c["cell"]] == i for i, c in cells.iloc[:12].iterrows())
print("auto matched correctly:", ok, "/ 12 | lonely cell ->", m["0399_lonely"])
assert ok == 12 and m["0399_lonely"] is None

link = dict(path=xl, sheet="cells", id_cols=ids, manual={})
ephys.save_link(master, link)
merged, info = ephys.merge(cells, master)
print("merge:", {k: v for k, v in info.items() if k != "path"})
assert info["matched"] == 12 and len(info["unmatched_rows"]) == 1
assert np.isnan(merged.loc[12, "ephys_Rin_MΩ"]) and merged["ephys_Vrest_mV"].iloc[:12].notna().all()

# manual override is kept (by row key) and wins over auto
keys = ephys.row_keys(t, ids)
link["manual"] = {"0399_lonely": keys[12]}
ephys.save_link(master, link)
merged, info = ephys.merge(cells, master)
assert info["matched"] == 13 and merged.loc[12, "ephys_Rin_MΩ"] == 300

# clustering: ephys alone separates the three types; balancing keeps 3 morph vs 3 ephys features even
sub = merged.iloc[:12].reset_index(drop=True)
ef = [c for c in sub.columns if c.startswith("ephys_") and c != "ephys_row"]
for name, fs, bal in (("ephys only", ef, None), ("morph+ephys balanced", ef + ["field_width_um", "dend_frac_q2"], "ephys_")):
    r = cluster_cells(sub, fs, 3, balance_prefix=bal)
    pur = sum(sub.groupby(r["labels"])["manual_type"].agg(lambda s: s.value_counts().iloc[0]))
    print(f"{name}: purity {pur}/12")
    assert pur >= 11
r1 = cluster_cells(sub, ef + ["field_width_um"], 3, balance_prefix="ephys_")
assert np.allclose(r1["Z"].std(0), 1)          # heatmap still shows plain z-scores

# ---- the dialog and the Classify tab
from qtpy.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from arborist.gui import EphysLinkDialog
dlg = EphysLinkDialog(None, xl, cells, ephys.load_link(master))
assert dlg.cmb_sheet.currentText() == "cells"
n_auto = sum(1 for cb in dlg.combos.values() if cb.currentIndex() > 0)
dlg.combos["0399_lonely"].setCurrentIndex(0)
L = dlg.result_link()
assert L["manual"] == {} and n_auto == 13, (L, n_auto)   # lonely back to none = same as auto
dlg.combos[cells.cell[0]].setCurrentIndex(13)
assert dlg.result_link()["manual"] == {cells.cell[0]: keys[12]}
dlg.resize(900, 650); dlg.show(); app.processEvents()
shots = os.path.join(os.path.dirname(__file__), "shots"); os.makedirs(shots, exist_ok=True)
dlg.grab().save(os.path.join(shots, "ephys_dialog.png"))
dlg.close()

import napari
from arborist.gui import SCMorphWidget
v = napari.Viewer(show=False)
w = SCMorphWidget(v)
ephys.save_link(master, dict(path=xl, sheet="cells", id_cols=ids, manual={}))
w.ed_master.setText(master)
tbl = w.load_master()
items = [w.lst_feat.item(i).text() for i in range(w.lst_feat.count())]
print("feature list:", items)
assert any("Electrophysiology" in x for x in items) and "ephys_Sag_ratio" in items
assert "12 of 13" in w.lbl_ephys.text()
w._select_features("ephys"); w.sp_k.setValue(3)
w.run_clustering()
print(w.lbl_cluster.text())
assert "without ephys" in w.lbl_cluster.text()
assert os.path.exists(os.path.join(D, "all_cells_summary_clusters_table_with_ephys.csv"))
w.lst_feat.setMinimumHeight(330)
w.tabs.setCurrentIndex(w.tabs.count() - 1)
w.resize(520, 1100); w.show(); app.processEvents()
w.grab().save(os.path.join(shots, "classify_ephys.png"))
v.close()
print("EPHYS OK")
