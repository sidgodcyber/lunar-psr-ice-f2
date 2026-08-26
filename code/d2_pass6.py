"""
Pass 6 (20200808) COMPACT-POL L-band CPR/DOP over F2, for two-pass merge with pass 4.

Compact-pol transmits circular (Left) and receives H,V linearly -> the two complex
channels ARE the received fields E_h=LH, E_v=LV (no synthesis needed). The child-wave
Stokes are computed directly (m-chi basis):
    S0=<|LH|^2>+<|LV|^2>, S1=<|LH|^2>-<|LV|^2>, S2=2Re<LH conj LV>, S3=-2Im<LH conj LV>
    CPR=(S0-S3)/(S0+S3),  DOP=sqrt(S1^2+S2^2+S3^2)/S0
This is the SAME circular-Stokes formulation used for pass 4, but applied to the
*measured* compact-pol channels rather than channels synthesized from full-pol.

SLI is 355768x759 complex (2.2 GB/channel) -> read + multilook in azimuth blocks.
"""

from __future__ import annotations

import sys
import time
import glob
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine, rowcol
from rasterio.warp import transform as warp_transform
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

P6_DIR = cfg.SAR_DIR / "pass 6/data/calibrated/20200808"
P6_GEOM = cfg.SAR_DIR / "pass 6/geometry/calibrated/20200808"
P6 = "ch2_sar_ncxl_20200808t201154198"
LH = P6_DIR / f"{P6}_d_sli_xx_cp_lh_d18.tif"
LV = P6_DIR / f"{P6}_d_sli_xx_cp_lv_d18.tif"
G_SLI = P6_GEOM / f"{P6}_g_sli_xx_cp_xx_d18.csv"
SLI_AZ, SLI_RNG = 355768, 759
TIE_AZ, TIE_RNG = 11119, 25
ML_AZ, ML_RNG = 83, 4

CPR_TIF = cfg.GEOTIFF_DIR / "cpr_pass6.tif"
DOP_TIF = cfg.GEOTIFF_DIR / "dop_pass6.tif"
DUAL_TIF = cfg.GEOTIFF_DIR / "dual_criterion_pass6.tif"
TARGET_HALF_KM = 9.0


def read_ml_moments_blocked():
    """Block-wise (azimuth) multilooked compact-pol moments M_hh,M_vv,M_hv."""
    n_az = (SLI_AZ // ML_AZ) * ML_AZ
    block = ML_AZ * 40                     # 40 output az rows per block
    Mhh, Mvv, Mhv = [], [], []
    t = time.time()
    with rasterio.open(str(LH)) as slh, rasterio.open(str(LV)) as slv:
        for r0 in range(0, n_az, block):
            r1 = min(r0 + block, n_az)
            win = ((r0, r1), (0, SLI_RNG))
            lh = slh.read(indexes=[1, 2], window=win)
            lv = slv.read(indexes=[1, 2], window=win)
            e_h = lh[0].astype(np.float32) + 1j * lh[1].astype(np.float32)
            e_v = lv[0].astype(np.float32) + 1j * lv[1].astype(np.float32)
            mhh = (e_h.real ** 2 + e_h.imag ** 2).astype(np.float32)
            mvv = (e_v.real ** 2 + e_v.imag ** 2).astype(np.float32)
            mhv = (e_h * np.conj(e_v)).astype(np.complex64)
            Mhh.append(multilook_real(mhh, ML_AZ, ML_RNG))
            Mvv.append(multilook_real(mvv, ML_AZ, ML_RNG))
            Mhv.append(multilook_complex(mhv, ML_AZ, ML_RNG))
    M_hh = np.concatenate(Mhh); M_vv = np.concatenate(Mvv); M_hv = np.concatenate(Mhv)
    enl = ML_AZ * ML_RNG
    M_hh = lee_filter(M_hh, cfg.REFINED_LEE_WIN, enl)
    M_vv = lee_filter(M_vv, cfg.REFINED_LEE_WIN, enl)
    print(f"      multilooked to {M_hh.shape} ({ML_AZ}x{ML_RNG}) in {time.time()-t:.0f}s")
    return M_hh, M_vv, M_hv


def geocode(arrays, names, target_tfm, target_shape, crs):
    naz_ml, nrng_ml = arrays[0].shape
    az_px, rng_px, lat_g, lon_g = load_tie_grid(G_SLI, TIE_AZ, TIE_RNG, SLI_AZ, SLI_RNG)
    flat = RegularGridInterpolator((az_px, rng_px), lat_g, bounds_error=False, fill_value=None)
    flon = RegularGridInterpolator((az_px, rng_px), lon_g, bounds_error=False, fill_value=None)
    ii = np.arange(naz_ml) * ML_AZ + ML_AZ / 2.0
    jj = np.arange(nrng_ml) * ML_RNG + ML_RNG / 2.0
    AZ, RNG = np.meshgrid(ii, jj, indexing="ij")
    pts = np.column_stack([AZ.ravel(), RNG.ravel()])
    lat = flat(pts); lon = flon(pts)
    geo = CRS.from_proj4(cfg.LUNAR_GEO_PROJ4)
    xs, ys = warp_transform(geo, crs, lon.tolist(), lat.tolist())
    xs = np.array(xs); ys = np.array(ys)
    t = target_tfm
    x0 = t.c; y1 = t.f
    n = target_shape[0]
    px = t.a
    gx = x0 + (np.arange(n) + 0.5) * px
    gy = y1 + (np.arange(n) + 0.5) * t.e
    GX, GY = np.meshgrid(gx, gy)
    half = n * px
    m = (xs > x0 - 500) & (xs < x0 + half + 500) & (ys < y1 + 500) & (ys > y1 + n * t.e - 500)
    src = np.column_stack([xs[m], ys[m]])
    print(f"      geocode: {m.sum()} ML points -> {n}x{n} grid")
    out = {}
    for arr, nm in zip(arrays, names):
        out[nm] = griddata(src, arr.ravel()[m], (GX, GY), method="linear").astype(np.float32)
    return out


def main():
    t0 = time.time()
    print("=" * 72); print("[pass6] Compact-pol L-band CPR/DOP over F2")
    # target grid from pass4 CPR
    with rasterio.open(cfg.CPR_TIF) as s:
        tfm = s.transform; crs = s.crs; shape = (s.height, s.width)
    M_hh, M_vv, M_hv = read_ml_moments_blocked()
    cpr, dop, s0 = cpr_dop_circular(M_hh, M_vv, M_hv)
    print(f"      slant CPR p50/95 {np.nanpercentile(cpr,50):.2f}/{np.nanpercentile(cpr,95):.2f} "
          f"| DOP p50 {np.nanpercentile(dop,50):.2f}")
    geo = geocode([cpr, dop, s0], ["cpr", "dop", "s0"], tfm, shape, crs)
    cg, dg, s0g = geo["cpr"], geo["dop"], geo["s0"]
    fin = s0g[np.isfinite(s0g) & (s0g > 0)]
    noise = np.percentile(fin, 5) if fin.size else 0.0
    valid = np.isfinite(cg) & np.isfinite(dg) & (s0g > 2 * noise)
    dual = (cg > cfg.CPR_THRESHOLD) & (dg < cfg.DOP_THRESHOLD_RELAXED) & valid

    def save(path, arr, dt, nd):
        p = {"driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1], "count": 1,
             "dtype": dt, "crs": crs, "transform": tfm, "nodata": nd, "compress": "lzw"}
        with rasterio.open(str(path), "w", **p) as d:
            d.write(arr.astype(dt), 1)
    save(CPR_TIF, np.nan_to_num(cg, nan=-1), "float32", -1)
    save(DOP_TIF, np.nan_to_num(dg, nan=-1), "float32", -1)
    save(DUAL_TIF, dual.astype(np.uint8), "uint8", 255)

    # F2-floor stats for sanity
    def og(path, rs=rasterio.enums.Resampling.nearest):
        from rasterio.warp import reproject
        with rasterio.open(str(path)) as s:
            d = np.zeros(shape, np.float32)
            reproject(rasterio.band(s, 1), d, src_transform=s.transform, src_crs=s.crs,
                      dst_transform=tfm, dst_crs=crs, resampling=rs)
        return d
    floor = (og(cfg.F2_AOI_TIF) > 0.5) & (og(cfg.DSC_MASK_TIF) > 0.5) & valid
    ni = int(floor.sum())
    if ni:
        c = cg[floor]
        print(f"      F2 floor (n={ni}): CPR mean {c.mean():.2f} max {c.max():.2f} "
              f">1={100*(c>1).mean():.0f}% | dual={100*dual[floor].mean():.0f}%")
    print(f"      ✓ pass6 CPR/DOP/dual saved | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
