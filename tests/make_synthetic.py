"""Synthetic test data: a tilted, gently curved brain surface and a bright neuron.

Brain tissue lies at y > f(x, z).  Writes ImageJ hyperstack TIFFs:
  synthetic_cell.tif   (0.5 x 0.5 x 1.0 um voxels, 2 channels)
  synthetic_lowmag.tif (2 um voxels, CR band 120 um thick)
"""
import os
import sys

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter


def f_surface(x, z):
    return 40 + 0.15 * (x - 75) + 0.004 * (z - 30) ** 2


def draw_tube(vol, voxel, p0, p1, r_um=0.8, val=3000):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = int(np.linalg.norm(p1 - p0) / 0.25) + 2
    for t in np.linspace(0, 1, n):
        x, y, z = p0 + t * (p1 - p0)
        zi, yi, xi = int(round(z / voxel[0])), int(round(y / voxel[1])), int(round(x / voxel[2]))
        r = int(np.ceil(r_um / voxel[2]))
        vol[max(zi - 1, 0):zi + 2, max(yi - r, 0):yi + r + 1, max(xi - r, 0):xi + r + 1] = val


def main(out):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(1)
    vox = (1.0, 0.5, 0.5)
    Z, Y, X = 60, 300, 300                         # 60 x 150 x 150 um
    z, y, x = np.meshgrid(np.arange(Z) * vox[0], np.arange(Y) * vox[1], np.arange(X) * vox[2], indexing="ij")
    tissue = y > f_surface(x, z)
    vol = np.where(tissue, 300.0, 40.0)
    soma = np.array([75.0, 95.0, 30.0])            # x, y, z  (um)
    dends = [(soma, [40, 70, 20]), (soma, [110, 75, 40]), ([40, 70, 20], [25, 62, 15]),
             (soma, [80, 130, 32])]
    for a, b in dends:
        draw_tube(vol, vox, a, b)
    zz, yy, xx = np.round(soma[::-1] / vox).astype(int)
    vol[zz - 3:zz + 4, yy - 10:yy + 11, xx - 10:xx + 11] = 3500
    vol = gaussian_filter(vol, (0.7, 1.2, 1.2)) + rng.normal(0, 25, vol.shape)
    ch2 = np.where(tissue, 500.0, 30.0) + rng.normal(0, 20, vol.shape)
    data = np.stack([vol, ch2], axis=1).clip(0, 65535).astype(np.uint16)   # Z C Y X
    tifffile.imwrite(os.path.join(out, "synthetic_cell.tif"), data, imagej=True,
                     resolution=(1 / vox[2], 1 / vox[1]), metadata={"spacing": vox[0], "unit": "um", "axes": "ZCYX"})

    # low-mag: flat surface at y = 50 um, CR+ band 120 um deep (tilted 20 deg)
    vl = (4.0, 2.0, 2.0)
    Z, Y, X = 6, 250, 250
    z, y, x = np.meshgrid(np.arange(Z) * vl[0], np.arange(Y) * vl[1], np.arange(X) * vl[2], indexing="ij")
    ang = np.deg2rad(20)
    nvec = np.array([-np.sin(ang), np.cos(ang)])   # into-brain normal in x/y
    depth = (x - 250) * nvec[0] + (y - 150) * nvec[1]
    cr = np.where((depth > 0) & (depth < 120), 1500.0, np.where(depth > 0, 300.0, 20.0))
    cr = gaussian_filter(cr, (0, 1.5, 1.5)) + rng.normal(0, 30, cr.shape)
    tifffile.imwrite(os.path.join(out, "synthetic_lowmag.tif"), cr.clip(0, 65535).astype(np.uint16)[:, None],
                     imagej=True, resolution=(1 / vl[2], 1 / vl[1]),
                     metadata={"spacing": vl[0], "unit": "um", "axes": "ZCYX"})
    print("written to", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synthetic")
