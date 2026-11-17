"""Minimal SWC neuron model: read, write, edit, segments."""
from __future__ import annotations

import numpy as np
import pandas as pd

SOMA, AXON, DENDRITE, APICAL = 1, 2, 3, 4
TYPE_NAMES = {1: "soma", 2: "axon", 3: "dendrite", 4: "apical dendrite", 7: "AIS (custom)", 0: "undefined"}
from .style import DEND as _D, AXON as _A, SOMA as _S, AIS as _I
TYPE_COLORS = {1: _S, 2: _A, 3: _D, 4: _D, 7: _I, 0: "#999999"}


class Neuron:
    """Tree of nodes, stored as arrays. Coordinates are in micrometres (x, y, z)."""

    COLS = ["id", "type", "x", "y", "z", "radius", "parent"]

    def __init__(self, df: pd.DataFrame | None = None):
        if df is None:
            df = pd.DataFrame(columns=self.COLS)
        self.df = df.reset_index(drop=True).astype(
            {"id": int, "type": int, "x": float, "y": float, "z": float,
             "radius": float, "parent": int})
        self.paths: list[list[int]] = []   # node ids added per traced path (undo stack)
        self.warnings: list[str] = []

    # ------------------------------------------------------------------ io
    @classmethod
    def read(cls, path: str) -> "Neuron":
        """Read an SWC file. Tolerates extra columns, commas as separators, blank / comment lines,
        non-numeric lines and duplicate node ids (the first one is kept); problems are listed in
        ``neuron.warnings``."""
        rows, bad = [], 0
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = line.replace(",", " ").replace(";", " ").split()
                if len(parts) < 7:
                    bad += 1
                    continue
                try:
                    v = [float(x) for x in parts[:7]]
                except ValueError:
                    bad += 1
                    continue
                if not all(np.isfinite(v[i]) for i in (0, 1, 2, 3, 4, 6)):
                    bad += 1
                    continue
                if not np.isfinite(v[5]):
                    v[5] = 0.0
                rows.append(v)
        df = pd.DataFrame(rows, columns=cls.COLS)
        warnings = []
        if bad:
            warnings.append(f"{bad} line(s) could not be read and were skipped")
        dup = df["id"].duplicated()
        if dup.any():
            warnings.append(f"{int(dup.sum())} node id(s) occur more than once (first one kept)")
            df = df[~dup]
        if len(df):
            orphan = ~df["parent"].isin(df["id"]) & (df["parent"] > 0)
            if orphan.any():
                warnings.append(f"{int(orphan.sum())} node(s) point to a parent that is not in the file "
                                "(treated as separate trees)")
        out = cls(df)
        out.warnings = warnings
        return out

    def write(self, path: str, comment: str = "") -> None:
        with open(path, "w", newline="\n") as fh:
            fh.write("# SWC written by Arborist\n")
            if comment:
                for line in comment.splitlines():
                    fh.write(f"# {line}\n")
            fh.write("# id type x y z radius parent\n")
            for r in self.df.itertuples(index=False):
                fh.write(f"{r.id} {r.type} {r.x:.4f} {r.y:.4f} {r.z:.4f} {r.radius:.4f} {r.parent}\n")

    # ------------------------------------------------------------- queries
    def __len__(self):
        return len(self.df)

    @property
    def xyz(self) -> np.ndarray:
        return self.df[["x", "y", "z"]].to_numpy(float)

    def next_id(self) -> int:
        return int(self.df["id"].max()) + 1 if len(self.df) else 1

    def nearest_node(self, xyz, max_dist: float = np.inf):
        if not len(self.df):
            return None, np.inf
        d = np.linalg.norm(self.xyz - np.asarray(xyz, float), axis=1)
        i = int(np.argmin(d))
        if d[i] > max_dist:
            return None, d[i]
        return int(self.df["id"].iloc[i]), float(d[i])

    def node_xyz(self, node_id: int) -> np.ndarray:
        r = self.df.loc[self.df["id"] == node_id]
        return r[["x", "y", "z"]].to_numpy(float)[0]

    def segments(self, types=None):
        """Return (child_xyz, parent_xyz, child_type) arrays for every parent-child edge.

        The edge takes the child's type, which is how SNT colours paths.
        """
        df = self.df
        if not len(df):
            z = np.zeros((0, 3))
            return z, z, np.zeros(0, int)
        idx = pd.Series(np.arange(len(df)), index=df["id"].to_numpy())
        has_parent = df["parent"].isin(idx.index).to_numpy()
        child = df[has_parent]
        if types is not None:
            child = child[child["type"].isin(types)]
        pr = idx.loc[child["parent"].to_numpy()].to_numpy()
        c = child[["x", "y", "z"]].to_numpy(float)
        p = df[["x", "y", "z"]].to_numpy(float)[pr]
        return c, p, child["type"].to_numpy(int)

    # -------------------------------------------------------------- editing
    def add_soma(self, xyz, radius: float = 5.0) -> int:
        nid = self.next_id()
        self._append([[nid, SOMA, *xyz, radius, -1]])
        self.paths.append([nid])
        return nid

    def add_path(self, pts_xyz: np.ndarray, node_type: int, parent_id: int | None = None,
                 radius: float = 0.5) -> list[int]:
        """Append a polyline. If ``parent_id`` is given the first point is attached to
        that node (the first point is dropped if it coincides with the parent)."""
        pts = np.asarray(pts_xyz, float)
        if parent_id is not None and len(pts):
            if np.linalg.norm(pts[0] - self.node_xyz(parent_id)) < 1e-6:
                pts = pts[1:]
        rows, ids = [], []
        nid = self.next_id()
        parent = parent_id if parent_id is not None else -1
        for p in pts:
            rows.append([nid, node_type, p[0], p[1], p[2], radius, parent])
            ids.append(nid)
            parent = nid
            nid += 1
        self._append(rows)
        self.paths.append(ids)
        return ids

    def undo_last_path(self) -> bool:
        if not self.paths:
            return False
        ids = set(self.paths.pop())
        self.df = self.df[~self.df["id"].isin(ids)].reset_index(drop=True)
        # any later children of removed nodes become roots
        self.df.loc[self.df["parent"].isin(ids), "parent"] = -1
        return True

    def set_type_all(self, old: int, new: int):
        self.df.loc[self.df["type"] == old, "type"] = new

    def _append(self, rows):
        if not rows:
            return
        new = pd.DataFrame(rows, columns=self.COLS)
        new = new.astype(self.df.dtypes.to_dict()) if len(self.df) else new
        self.df = pd.concat([self.df, new], ignore_index=True) if len(self.df) else new.reset_index(drop=True)
        self.df = self.df.astype({"id": int, "type": int, "parent": int})
