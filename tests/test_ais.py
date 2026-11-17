import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from arborist.swc import Neuron
from arborist.surface import BrainSurface
from arborist.morphometry import ais_features

surf = BrainSurface(np.array([[x, 0.0, z] for x in np.linspace(-200, 200, 15) for z in (0, 40, 80)]), extend_um=50)
surf.orient_towards([[0, 100, 40]])

# A) axon from the soma: soma -> 5 um axon -> 20 um AIS (type 7) -> axon
n = Neuron(); s = n.add_soma([0, 100, 40])
n.add_path(np.array([[0, 100, 40], [0, 140, 40]]), 3, parent_id=s)                 # dendrite
ax = n.add_path(np.array([[0, 100, 40], [0, 95, 40]]), 2, parent_id=s)              # 5 um axon hillock
ais = n.add_path(np.array([[0, 95, 40], [0, 85, 40], [0, 75, 40]]), 7, parent_id=ax[-1])  # 20 um AIS, going dorsal
n.add_path(np.array([[0, 75, 40], [0, 40, 40]]), 2, parent_id=ais[-1])
a = ais_features(n, surf)
print("A", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in a.items()})
assert a["ais_origin"] == "soma" and abs(a["ais_length_um"] - 20) < 1e-6 and abs(a["ais_start_dist_soma_path_um"] - 5) < 1e-6
assert a["ais_angle_from_dorsal_deg"] < 1

# B) axon-carrying dendrite: dendrite 30 um ventral, axon branches off at 15 um, AIS starts 3 um later
n = Neuron(); s = n.add_soma([0, 100, 40])
d = n.add_path(np.array([[0, 100, 40], [0, 115, 40], [0, 130, 40]]), 3, parent_id=s)
ax = n.add_path(np.array([[0, 115, 40], [3, 115, 40]]), 2, parent_id=d[0])
ais = n.add_path(np.array([[3, 115, 40], [28, 115, 40]]), 7, parent_id=ax[-1])
b = ais_features(n, surf)
print("B", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in b.items()})
assert b["ais_origin"] == "dendrite" and abs(b["acd_branch_dist_soma_path_um"] - 15) < 1e-6
assert abs(b["ais_start_dist_soma_path_um"] - 18) < 1e-6 and abs(b["ais_length_um"] - 25) < 1e-6

# C) same cell traced all as axon (type 2), AIS given by two clicked points (slightly off the trace)
n.df.loc[n.df["type"] == 7, "type"] = 2
c = ais_features(n, surf, points_xyz=np.array([[27.5, 116, 40], [3.2, 114.5, 40.5]]))
print("C", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in c.items()})
assert c["ais_origin"] == "dendrite" and abs(c["ais_length_um"] - 24.3) < 1e-6 and abs(c["ais_start_dist_soma_path_um"] - 18.2) < 1e-6

# D) AIS on a separate ankyrin scan: curved AIS, length must follow the staining
from arborist.morphometry import follow_signal, ais_features_from_points
vox = (1.0, 0.2, 0.2)
vol = np.random.default_rng(0).normal(100, 10, (9, 120, 120)).astype(np.float32)
th = np.linspace(0, np.pi / 2, 400)
cy, cx, R = 20, 20, 60                                   # quarter circle, radius 60 px = 12 um
for t in th:
    y, x = int(round(cy + R * np.sin(t))), int(round(cx + R * np.cos(t)))
    vol[4, y - 1:y + 2, x - 1:x + 2] = 2000
clicks = np.array([[4, cy, cx + R], [4, cy + R, cx]], float)          # just start and end
path = follow_signal(vol, clicks, vox)
to_xyz = lambda zyx: (np.asarray(zyx) * np.array(vox))[:, ::-1]
f = ais_features_from_points(to_xyz(clicks), soma_exit_xyz=to_xyz([[4, cy, cx + R + 25]]), path_xyz=to_xyz(path))
true_len = np.pi / 2 * R * 0.2
print("D", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in f.items()}, "true arc", round(true_len, 2))
assert abs(f["ais_length_um"] - true_len) / true_len < 0.03, "path does not follow the curve"
assert f["ais_length_straight_um"] < 0.92 * true_len
assert abs(f["ais_start_dist_soma_exit_um"] - 5.0) < 1e-6        # exit point is 25 px = 5 um from the start
print("AIS OK")

# E) soma exit -> branch -> AIS start route along a curved fill signal
from arborist.morphometry import order_route, route_along_signal
fill = np.random.default_rng(1).normal(100, 10, (9, 120, 120)).astype(np.float32)
for t in th:                                               # same quarter circle, now as the dendrite
    y, x = int(round(cy + R * np.sin(t))), int(round(cx + R * np.cos(t)))
    fill[4, y - 1:y + 2, x - 1:x + 2] = 2000
mid_t = np.pi / 4
branch = np.array([4, cy + R * np.sin(mid_t), cx + R * np.cos(mid_t)]).round()
nodes = order_route([4, cy, cx + R], [branch], [4, cy + R, cx])
path, cum = route_along_signal(fill, nodes, vox)
print("E route to branch %.2f (true %.2f), to end %.2f (true %.2f)" % (cum[1], true_len / 2, cum[-1], true_len))
assert abs(cum[1] - true_len / 2) / (true_len / 2) < 0.05 and abs(cum[-1] - true_len) / true_len < 0.03
f = ais_features_from_points(to_xyz([[4, cy + R, cx], [4, cy + R + 30, cx]]), soma_exit_xyz=to_xyz([[4, cy, cx + R]]),
                             branch_xyz=to_xyz([branch]), origin="dendrite", route_to_ais_um=cum[-1], route_to_branch_um=cum[1])
print("E", {k: round(v, 2) for k, v in f.items() if k.startswith(("acd", "ais_start"))})
assert f["route_method"] == "follows fill signal" and f["acd_branch_dist_soma_exit_um"] > f["acd_branch_dist_soma_exit_straight_um"]
print("ROUTE OK")
