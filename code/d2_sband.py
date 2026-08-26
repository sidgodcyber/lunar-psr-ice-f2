"""
S-band compact-pol CPR/DOP over F2 (multi-frequency depth stratification).

Identical m-chi compact-pol pipeline as the L-band pass 6 (LH/LV -> circular
child Stokes -> CPR/DOP), applied to the S-band (ncxs, 2.5 GHz) channels of the
same 20200808 acquisition. S-band (lambda 0.12 m) senses shallower than L-band
(lambda 0.24 m), enabling depth stratification of the F2 ice signal.

Run: python code/d2_sband.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform as warp_transform, reproject, Resampling
from rasterio.crs import CRS
from scipy.interpolate import RegularGridInterpolator, griddata

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
import d2_config as cfg
from utils.sar_io import multilook_real, multilook_complex, lee_filter, load_tie_grid
from utils.polarimetry import cpr_dop_circular

P6D = cfg.SAR_DIR / "pass 6/data/calibrated/20200808"
P6G = cfg.SAR_DIR / "pass 6/geometry/calibrated/20200808"
PS = "ch2_sar_ncxs_20200808t201154198"
LH = P6D / f"{PS}_d_sli_xx_cp_lh_d18.tif"
LV = P6D / f"{PS}_d_sli_xx_cp_lv_d18.tif"
G_SLI = P6G / f"{PS}_g_sli_xx_cp_xx_d18.csv"
SLI_AZ, SLI_RNG = 355768, 759
ML_AZ, ML_RNG = 83, 4

CPR_TIF = cfg.GEOTIFF_DIR / "cpr_sband.tif"
DOP_TIF = cfg.GEOTIFF_DIR / "dop_sband.tif"
DUAL_TIF = cfg.GEOTIFF_DIR / "dual_criterion_sband.tif"


def _tie_shape():
    d = np.genfromtxt(str(G_SLI), delimiter=",", skip_header=1)
    sr = d[:, 2]; resets = np.where(np.diff(sr) < -100)[0]
    tie_rng = int(np.bincount(np.diff(resets)).argmax()) if len(resets) > 1 else 25
    tie_az = len(d) // tie_rng
    return tie_az, tie_rng


def read_ml_moments():
    n_az = (SLI_AZ // ML_AZ) * ML_AZ
    block = ML_AZ * 40
    Mhh, Mvv, Mhv = [], [], []
    t = time.time()
    with rasterio.open(str(LH)) as slh, rasterio.open(str(LV)) as slv:
        for r0 in range(0, n_az, block):
            r1 = min(r0 + block, n_az); win = ((r0, r1), (0, SLI_RNG))
            lh = slh.read(indexes=[1, 2], window=win); lv = slv.read(indexes=[1, 2], window=win)
            e_h = lh[0].astype(np.float32) + 1j * lh[1].astype(np.float32)
            e_v = lv[0].astype(np.float32) + 1j * lv[1].astype(np.float32)
            Mhh.append(multilook_real((e_h.real**2 + e_h.imag**2).astype(np.float32), ML_AZ, ML_RNG))
            Mvv.append(multilook_real((e_v.real**2 + e_v.imag**2).astype(np.float32), ML_AZ, ML_RNG))
            Mhv.append(multilook_complex((e_h * np.conj(e_v)).astype(np.complex64), ML_AZ, ML_RNG))
    M_hh = np.concatenate(Mhh); M_vv = np.concatenate(Mvv); M_hv = np.concatenate(Mhv)
    enl = ML_AZ * ML_RNG
    M_hh = lee_filter(M_hh, cfg.REFINED_LEE_WIN, enl); M_vv = lee_filter(M_vv, cfg.REFINED_LEE_WIN, enl)
    print(f"      S-band multilooked to {M_hh.shape} in {time.time()-t:.0f}s")
    return M_hh, M_vv, M_hv


def geocode(arrs, names, tfm, shape, crs, tie_az, tie_rng):
    naz, nrng = arrs[0].shape
    az_px, rng_px, lat_g, lon_g = load_tie_grid(G_SLI, tie_az, tie_rng, SLI_AZ, SLI_RNG)
    flat = RegularGridInterpolator((az_px, rng_px), lat_g, bounds_error=False, fill_value=None)
    flon = RegularGridInterpolator((az_px, rng_px), lon_g, bounds_error=False, fill_value=None)
    ii = np.arange(naz) * ML_AZ + ML_AZ / 2.0; jj = np.arange(nrng) * ML_RNG + ML_RNG / 2.0
    AZ, RNG = np.meshgrid(ii, jj, indexing="ij")
    pts = np.column_stack([AZ.ravel(), RNG.ravel()])
    lat = flat(pts); lon = flon(pts)
    geo = CRS.from_proj4(cfg.LUNAR_GEO_PROJ4)
    xs, ys = warp_transform(geo, crs, lon.tolist(), lat.tolist()); xs = np.array(xs); ys = np.array(ys)
    x0, y1 = tfm.c, tfm.f; n = shape[0]; px = tfm.a
    gx = x0 + (np.arange(n) + 0.5) * px; gy = y1 + (np.arange(n) + 0.5) * tfm.e
    GX, GY = np.meshgrid(gx, gy)
    half = n * px
    m = (xs > x0 - 500) & (xs < x0 + half + 500) & (ys < y1 + 500) & (ys > y1 + n * tfm.e - 500)
    src = np.column_stack([xs[m], ys[m]])
    print(f"      geocode: {m.sum()} ML pts -> {n}x{n}")
    return {nm: griddata(src, a.ravel()[m], (GX, GY), method="linear").astype(np.float32)
            for a, nm in zip(arrs, names)}


def main():
    t0 = time.time()
    print("=" * 72); print("[S-band] compact-pol CPR/DOP over F2 (2.5 GHz)")
    tie_az, tie_rng = _tie_shape()
    print(f"      tie grid {tie_az} x {tie_rng}")
    with rasterio.open(cfg.CPR_TIF) as s:
        tfm = s.transform; crs = s.crs; shape = (s.height, s.width)
    M_hh, M_vv, M_hv = read_ml_moments()
    cpr, dop, s0 = cpr_dop_circular(M_hh, M_vv, M_hv)
    print(f"      slant CPR p50/95 {np.nanpercentile(cpr,50):.2f}/{np.nanpercentile(cpr,95):.2f}")
    g = geocode([cpr, dop, s0], ["cpr", "dop", "s0"], tfm, shape, crs, tie_az, tie_rng)
    cg, dg, s0g = g["cpr"], g["dop"], g["s0"]
    fin = s0g[np.isfinite(s0g) & (s0g > 0)]; noise = np.percentile(fin, 5) if fin.size else 0
    valid = np.isfinite(cg) & np.isfinite(dg) & (s0g > 2 * noise)
    dual = (cg > cfg.CPR_THRESHOLD) & (dg < cfg.DOP_THRESHOLD_RELAXED) & valid

    def save(p, a, dt, nd):
        pr = {"driver": "GTiff", "height": a.shape[0], "width": a.shape[1], "count": 1,
              "dtype": dt, "crs": crs, "transform": tfm, "nodata": nd, "compress": "lzw"}
        with rasterio.open(str(p), "w", **pr) as d: d.write(a.astype(dt), 1)
    save(CPR_TIF, np.nan_to_num(cg, nan=-1), "float32", -1)
    save(DOP_TIF, np.nan_to_num(dg, nan=-1), "float32", -1)
    save(DUAL_TIF, dual.astype(np.uint8), "uint8", 255)

    def og(p, rs=Resampling.nearest):
        with rasterio.open(str(p)) as s:
            d = np.zeros(shape, np.float32)
            reproject(rasterio.band(s, 1), d, src_transform=s.transform, src_crs=s.crs,
                      dst_transform=tfm, dst_crs=crs, resampling=rs)
        return d
    floor = (og(cfg.F2_AOI_TIF) > 0.5) & (og(cfg.DSC_MASK_TIF) > 0.5) & valid
    ni = int(floor.sum())
    if ni:
        c = cg[floor]
        print(f"      F2 floor (n={ni}): CPR mean {c.mean():.2f} max {c.max():.2f} >1={100*(c>1).mean():.0f}% | dual={100*dual[floor].mean():.0f}%")
    print(f"      [ok] S-band saved | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
