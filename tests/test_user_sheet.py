"""The layout of the lab's electrophysiology sheet: roman section numbers, dates in folder names that
also contain other numbers (GCaMP3Thy1, Bl6), 'thrown away' rows, a recording-number column, '-' for
missing values and two cells on one side of a section."""
import os, sys, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from arborist import ephys
import tempfile
rows = [['Date', 'File name', 'Section and side', 'Cell type based on me', 'cell type based articles', None, 'Tau', 'sag', 'first 2 AP'], ['2026-06-24', '26624016', 'II. left', 'early fireing shifting', 'horizontal', None, 17.5, 6.1, '-'], ['2026-06-24', '26624061', 'IV. right', None, 'horizontal', None, 11.0, 1.2, '33.1'], ['2026-06-24', '26624084', 'thrown away', 'few_AP', 'horizontal', None, 9.0, 2.0, '-'], ['2026-06-24', '26624026', 'II. right', 'early fireing shifting', 'stellate', None, 20.3, 7.3, '-'], ['2026-06-24', '26624050', 'IV. left', 'FS', 'stellate', None, 20.1, 6.1, '3.3'], ['2026-07-15', '26715022', 'II. left', 'few_AP', 'horizontal', None, 31.4, 8.2, '6.6'], ['2026-07-15', '26715001', 'I. left', 'few_AP', 'horizontal', None, 14.9, 10.1, '-'], ['2026-07-15', '26715046', 'III. left', 'FS', 'stellate', None, 16.7, 6.1, '-'], ['2026-10-29', '26o29066-074', 'IV. right', None, None, None, 12.0, 3.0, '-'], ['2026-10-29', '26o29075-080', 'IV. right right', None, None, None, 13.0, 3.5, '-'], ['2026-10-29', '26o29036-048', 'II. left', None, None, None, 15.0, 4.0, '5.0']]
rows = [rows[0]] + [[pd.Timestamp(r[0])] + r[1:] for r in rows[1:]]
p = os.path.join(tempfile.mkdtemp(), "sheet.xlsx")
pd.DataFrame(rows).to_excel(p, header=False, index=False, sheet_name="SC cells")
t = ephys.read_table(p, "SC cells")
ids = ephys.guess_id_columns(t)
assert ids == ["Date", "File name", "Section and side"], ids
assert "first 2 AP" in ephys.feature_columns(t, ids)                 # '-' read as empty, not text
assert {"Cell type based on me", "cell type based articles"} <= set(ephys.label_columns(t, ids))
days = {"2026-06-24": "2026.06.24_PV_Cre_Patch_cells", "2026-07-15": "2026.07.15_GCaMP3Thy1_Patch",
        "2026-10-29": "2026.10.29_Bl6_Patch_cells"}
spec = [("2026-06-24", 2, "left", 0), ("2026-06-24", 2, "right", 3), ("2026-06-24", 4, "left", 4),
        ("2026-06-24", 1, "left", None), ("2026-07-15", 2, "left", 5), ("2026-07-15", 3, "left", 7),
        ("2026-10-29", 4, "right", None), ("2026-10-29", 2, "left", 10)]
cells = []
for d, sec, side, _ in spec:
    day = days[d]; short = day[:4] + day[5:7] + "." + day[8:10]
    stem = f"{short}_{day.split('_', 1)[1]}_section{sec}_{side}_63x_Stitch"
    cells.append(dict(cell=stem, source_path=f"E:/SC_cells/{day}/section {sec}/{side}/{stem}.czi"))
cells = pd.DataFrame(cells)
m = ephys.auto_match(cells, t, ids)
for (d, sec, side, want), c in zip(spec, cells["cell"]):
    assert m[c] == want, (d, sec, side, m[c], want)
print("USER SHEET OK")
