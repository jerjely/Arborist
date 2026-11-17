"""Thin 63x stack, deep CR border: the CR surface must stay in the section plane (full line on every slice)."""
import os, sys
import numpy as np
import trimesh
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from arborist.surface import BrainSurface

rng = np.random.default_rng(2)
Cc, R = np.array([500.0, 1500.0]), 1400.0         # gentle arc (radius 1.4 mm), brain below (larger y)
pts = []
for z in np.arange(0, 31, 5.0):
    shift = rng.normal(0, 3, 2)                    # hand differences between slices
    for a in np.deg2rad(np.linspace(250, 290, 160)):
        r = R + rng.normal(0, 2)
        pts.append([Cc[0] + r * np.cos(a) + shift[0], Cc[1] + r * np.sin(a) + shift[1], z])
pts = np.array(pts)
B = 150.0
q = []
for a in np.deg2rad([255, 265, 275, 285]):
    for z in (2.5, 12.3, 27.9):
        q.append([Cc[0] + (R - B) * np.cos(a), Cc[1] + (R - B) * np.sin(a), z])
q = np.array(q)
for inplane in (False, True):
    s = BrainSurface(pts, method="slices", z_range=(0, 30))
    s.inplane = inplane
    s.orient_towards([[500, 400, 15]])
    d = s.signed_depth(q)
    o = s.offset_mesh(B)
    segs = trimesh.intersections.mesh_plane(o, [0, 0, 1], [0, 0, 12.3])
    L = np.linalg.norm(segs[:, 0] - segs[:, 1], axis=1).sum() if len(segs) else 0
    print(f"{'in-plane' if inplane else '3D (old)'}: depth of 150 um points = {d.min():.1f}..{d.max():.1f} | "
          f"CR line on slice 12.3: {L:.0f} um long | CR surface z-range {o.vertices[:, 2].min():.0f}..{o.vertices[:, 2].max():.0f}")
assert abs(d - B).max() < 5 and L > 700 and o.vertices[:, 2].min() >= -1e-6 and o.vertices[:, 2].max() <= 30 + 1e-6
print("THIN STACK OK")
