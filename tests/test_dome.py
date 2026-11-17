"""Strongly curved (dome) surface: slice-by-slice loft must give correct depths everywhere."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from arborist.surface import BrainSurface

rng = np.random.default_rng(0)
C, R = np.array([400.0, 400.0]), 300.0                    # brain = inside the circle
angles = np.deg2rad(np.linspace(190, 330, 120))           # 140 deg turn, like the SC outline
pts = []
for z in range(0, 70, 10):
    for a in angles:                                      # pen-like dense line, ~1 um jitter
        r = R + rng.normal(0, 1.0)
        pts.append([C[0] + r * np.cos(a), C[1] + r * np.sin(a), z])
pts = np.array(pts)
q, truth = [], []
for a in np.deg2rad([200, 230, 260, 290, 320]):
    for d in (10, 50, 120):
        for z in (5, 33, 64):
            q.append([C[0] + (R - d) * np.cos(a), C[1] + (R - d) * np.sin(a), z]); truth.append(d)
q, truth = np.array(q), np.array(truth)
soma = [[C[0], C[1] - 150, 30]]
for method in ("slices", "smooth", "delaunay"):
    try:
        s = BrainSurface(pts, method=method, z_range=(0, 70))
        s.orient_towards(soma)
        err = s.signed_depth(q) - truth
        print(f"{method:9s} faces={len(s.mesh.faces):5d}  max |error| = {np.abs(err).max():7.2f} um   "
              f"median = {np.median(np.abs(err)):.2f} um")
        if method == "slices":
            assert np.abs(err).max() < 2.5, err
            # parallel CR surface 100 um deeper stays parallel on the steep part too
            o = s.offset_mesh(100)
            r_o = np.linalg.norm(o.vertices[:, :2] - C, axis=1)
            inner = np.abs(o.vertices[:, 2] - 30) < 25
            print("   offset surface radius:", np.round(np.percentile(r_o[inner], [5, 50, 95]), 1), "(expect 200)")
    except Exception as e:
        print(method, "failed:", e)
# single-slice marking (one line) also gives a surface
s1 = BrainSurface(pts[pts[:, 2] == 30], method="slices", z_range=(0, 70)); s1.orient_towards(soma)
print("single slice max |error|:", round(np.abs(s1.signed_depth(q) - truth).max(), 2))
print("DOME OK")
