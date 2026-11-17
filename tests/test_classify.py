"""Synthetic sSC-like cells of 5 shapes -> features, type hints, clustering."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from arborist import analysis
from arborist.morphometry import cluster_cells, cluster_figures
from arborist.surface import BrainSurface
from arborist.swc import Neuron

rng = np.random.default_rng(3)
OUT = os.path.join(os.path.dirname(__file__), "synthetic", "classify")
os.makedirs(OUT, exist_ok=True)
CR = 300.0   # CR border depth (um); surface is y = 0, brain at y > 0

# surface points
surf_pts = np.array([[x, 0.0 + rng.normal(0, .3), z] for x in np.linspace(-400, 400, 25) for z in (0, 40, 80)])
surf = BrainSurface(surf_pts, method="smooth", extend_um=50)
surf.orient_towards([[0, 100, 40]])


def make(kind, j):
    s = 1 + rng.normal(0, 0.1)
    if kind == "wide-field":     soma, dirs = 220, [(-170, -150), (170, -150), (-90, -170), (90, -170)]
    if kind == "narrow-field":   soma, dirs = 160, [(-25, -120), (20, -110), (0, -90), (5, 30)]
    if kind == "horizontal":     soma, dirs = 90, [(-200, -10), (200, 10), (-150, 15), (150, -15)]
    if kind == "stellate":       soma, dirs = 150, [(-60, -50), (60, -50), (-60, 50), (60, 50), (0, 70)]
    if kind == "marginal":       soma, dirs = 15, [(-30, 70), (30, 80), (0, 90)]
    n = Neuron()
    sid = n.add_soma([0, soma * s, 40])
    for dx, dy in dirs:
        end = np.array([dx * s, soma * s + dy * s, 40 + rng.normal(0, 10)])
        mid = (np.array([0, soma * s, 40]) + end) / 2 + rng.normal(0, 5, 3)
        ids = n.add_path(np.array([[0, soma * s, 40], mid, end]), 3, parent_id=sid)
        br = end + np.array([rng.normal(0, 20), rng.normal(0, 15), 0])
        n.add_path(np.array([n.node_xyz(ids[0]), br]), 3, parent_id=ids[0])
    n.add_path(np.array([[0, soma * s, 40], [10, soma * s + 120, 40]]), 2, parent_id=sid)
    return n


kinds = ["wide-field", "narrow-field", "horizontal", "stellate", "marginal"]
master = os.path.join(OUT, "all_cells_summary.csv")
if os.path.exists(master):
    os.remove(master)
hits = 0
for k in kinds:
    for j in range(3):
        name = f"{k}_{j}"
        r = analysis.run(make(k, j), surf, OUT, name, boundary_um=CR, master_csv=master, manual_type=k)
        h = r["features"]["type_hint"]
        hits += h == k
        print(f"{name:16s} hint={h:13s} ({r['features']['type_hint_reason']})")
print(f"type hints correct: {hits}/15")
t = pd.read_csv(master)
res = cluster_cells(t, [c for c in __import__('arborist.morphometry', fromlist=['x']).DEFAULT_CLUSTER_FEATURES], 5)
t["cluster"] = res["labels"]
print(pd.crosstab(t["manual_type"], t["cluster"]))
for k, f in cluster_figures(t, res).items():
    f.savefig(os.path.join(OUT, f"cluster_{k}.png"), dpi=110)
print("CLASSIFY OK")
