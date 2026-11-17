"""Realistic pen marking: several overlapping strokes per slice, drawn in both directions, with
jitter and little back-and-forth wiggles. Surface and CR line must stay clean."""
import os, sys
import numpy as np
import trimesh
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from arborist.surface import BrainSurface

rng = np.random.default_rng(5)
C, R = np.array([400.0, 900.0]), 800.0           # SC-like curvature, brain inside
def arcpts(a0, a1, n, z):
    a = np.deg2rad(np.linspace(a0, a1, n))
    r = R + rng.normal(0, 1.5, n)
    return np.column_stack([C[0] + r * np.cos(a), C[1] + r * np.sin(a), np.full(n, z)])
pts = []
for z in np.arange(0, 41, 5.0):
    pts.append(arcpts(235, 275, 120, z))           # stroke 1 left -> right
    pts.append(arcpts(300, 262, 110, z))           # stroke 2 right -> left, overlapping
    w = arcpts(266, 270, 30, z); w[::2] = w[::-1][::2]   # a scribble that goes back and forth
    pts.append(w)
pts = np.vstack(pts)
B = 180.0
s = BrainSurface(pts, method="slices", z_range=(0, 40))
s.orient_towards([[C[0], C[1] - 300, 20]])
# 1) surface: no long thin spike triangles
e = s.mesh.edges_unique_length
print("surface: longest mesh edge %.1f um (lines spaced 5 um in z)" % e.max())
# 2) depths of points 180 um inside
q = np.array([[C[0] + (R - B) * np.cos(a), C[1] + (R - B) * np.sin(a), z]
              for a in np.deg2rad([245, 260, 275, 290]) for z in (3, 17, 38)])
d = s.signed_depth(q); print("depth of 180 um points: %.1f .. %.1f" % (d.min(), d.max()))
# 3) CR line on a slice: every point is ~180 um deep, no jumps
o = s.offset_mesh(B)
segs = trimesh.intersections.mesh_plane(o, [0, 0, 1], [0, 0, 17.0])
P = segs.reshape(-1, 3)
r = np.linalg.norm(P[:, :2] - C, axis=1)
seglen = np.linalg.norm(segs[:, 0] - segs[:, 1], axis=1)
print("CR line radius 1-99%%: %s (expect %.0f) | longest piece %.1f um | CR mesh z %.0f..%.0f"
      % (np.percentile(r, [1, 99]).round(1), R - B, seglen.max(), o.vertices[:, 2].min(), o.vertices[:, 2].max()))
assert e.max() < 40 and np.abs(d - B).max() < 6
assert np.abs(r - (R - B)).max() < 10 and seglen.max() < 20 and o.vertices[:, 2].min() >= 0 and o.vertices[:, 2].max() <= 40
print("MESSY STROKES OK")
