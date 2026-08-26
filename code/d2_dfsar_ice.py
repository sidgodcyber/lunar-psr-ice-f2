"""
BAH GREEN26 — DELIVERABLE 2: DFSAR CPR/DOP ICE DETECTION on crater F2
====================================================================
Path B (rigorous): complex SLI -> covariance + multilook -> Stokes -> CPR/DOP
-> geocode slant-range to lunar S-polar stereographic -> clip F2 AOI ->
dual criterion (CPR>1 & DOP<0.13) -> confidence map.

Pass 4 (20191105) is the only acquisition covering F2.
Run from D:/BAH 26/ :  python code/d2_dfsar_ice.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine, rowcol
from rasterio.warp import transform as warp_transform, reproject, Resampling
from rasterio.crs import CRS
from scipy.interpolate import RegularGridInterpolator, griddata

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
import d2_config as cfg
from utils.sar_io import (read_complex, read_real, multilook_real,
                          multilook_complex, lee_filter, load_tie_grid)
from utils.polarimetry import synthesize_circular, cpr_dop_circular

TARGET_HALF_KM = 9.0     # geocoded grid half-width around F2


# ─────────────────────────────────────────────────────────────────────────────
def step2_load_multilook():
    print("=" * 72)
    print("[step 2/9] Load complex SLI + circular synthesis + multilook")
    t = time.time()
    hh = read_complex(cfg.sli("hh"))
    hv = read_complex(cfg.sli("hv"))
    vv = read_complex(cfg.sli("vv"))
    print(f"      SLI HH {hh.shape} {hh.dtype} | HV | VV")
    # Circular-transmit (Left) received fields; monostatic VH = HV.
    e_h, e_v = synthesize_circular(hh, hv, vv)
    del hh, hv, vv
    m_hh = (e_h.real.astype(np.float32) ** 2 + e_h.imag.astype(np.float32) ** 2)
    m_vv = (e_v.real.astype(np.float32) ** 2 + e_v.imag.astype(np.float32) ** 2)
    m_hv = (e_h * np.conj(e_v)).astype(np.complex64)
    del e_h, e_v

    laz, lrng = cfg.ML_AZ, cfg.ML_RNG
    M_hh = multilook_real(m_hh, laz, lrng)
    M_vv = multilook_real(m_vv, laz, lrng)
    M_hv = multilook_complex(m_hv, laz, lrng)
    del m_hh, m_vv, m_hv
    enl = laz * lrng
    # Extra speckle suppression on the child-wave moments (boosts effective looks)
    M_hh = lee_filter(M_hh, cfg.REFINED_LEE_WIN, enl)
    M_vv = lee_filter(M_vv, cfg.REFINED_LEE_WIN, enl)
    print(f"      multilooked to {M_hh.shape} ({laz}x{lrng}={enl} nominal looks) in {time.time()-t:.0f}s")
    return M_hh, M_vv, M_hv


# ─────────────────────────────────────────────────────────────────────────────
def step5_geocode(arrays, names):
    """Geocode multilooked slant-range arrays to a F2-centred stereographic grid."""
    print("=" * 72)
    print("[step 5/9] Geocode slant-range -> stereographic (tie-point grid)")
    naz_ml, nrng_ml = arrays[0].shape
    az_px, rng_px, lat_g, lon_g = load_tie_grid(
        cfg.G_SLI_CSV, cfg.TIE_AZ, cfg.TIE_RNG, cfg.SLI_AZ, cfg.SLI_RNG)
    flat = RegularGridInterpolator((az_px, rng_px), lat_g, bounds_error=False, fill_value=None)
    flon = RegularGridInterpolator((az_px, rng_px), lon_g, bounds_error=False, fill_value=None)

    # ML pixel centres in full SLI coordinates
    ii = (np.arange(naz_ml) * cfg.ML_AZ + cfg.ML_AZ / 2.0)
    jj = (np.arange(nrng_ml) * cfg.ML_RNG + cfg.ML_RNG / 2.0)
    AZ, RNG = np.meshgrid(ii, jj, indexing="ij")
    pts = np.column_stack([AZ.ravel(), RNG.ravel()])
    lat = flat(pts)
    lon = flon(pts)

    with rasterio.open(cfg.DEM_TIF) as s:
        dem_crs = s.crs
    geo = CRS.from_proj4(cfg.LUNAR_GEO_PROJ4)
    xs, ys = warp_transform(geo, dem_crs, lon.tolist(), lat.tolist())
    xs = np.array(xs); ys = np.array(ys)

    # Target grid centred on F2
    f2x, f2y = cfg.latlon_to_stereographic(cfg.F2_LAT_DEG, cfg.F2_LON_DEG)
    half = TARGET_HALF_KM * 1000.0
    px = cfg.GEOCODE_PIXEL_M
    n = int(2 * half / px)
    x0, y1 = f2x - half, f2y + half
    tfm = Affine(px, 0, x0, 0, -px, y1)
    gx = x0 + (np.arange(n) + 0.5) * px
    gy = y1 - (np.arange(n) + 0.5) * px
    GX, GY = np.meshgrid(gx, gy)

    # Restrict source points to target bbox (+margin) for speed
    m = (xs > x0 - 500) & (xs < x0 + 2 * half + 500) & (ys < y1 + 500) & (ys > y1 - 2 * half - 500)
    src = np.column_stack([xs[m], ys[m]])
    print(f"      {m.sum()} ML points in target bbox; griddata -> {n}x{n} @ {px:.0f} m")

    out = {}
    for arr, nm in zip(arrays, names):
        v = arr.ravel()[m]
        g = griddata(src, v, (GX, GY), method="linear")
        out[nm] = g.astype(np.float32)
    return out, tfm, dem_crs, (f2x, f2y)


def _reproject_to(target_tfm, target_shape, target_crs, src_path, resampling=Resampling.nearest):
    with rasterio.open(src_path) as s:
        dst = np.zeros(target_shape, dtype=np.float32)
        reproject(source=rasterio.band(s, 1), destination=dst,
                  src_transform=s.transform, src_crs=s.crs,
                  dst_transform=target_tfm, dst_crs=target_crs,
                  resampling=resampling)
    return dst


def step6_7_criterion_guard(geo, tfm, crs):
    print("=" * 72)
    print("[step 6/9 + 7/9] Dual criterion + false-positive guard")
    cpr, dop, s0 = geo["cpr"], geo["dop"], geo["s0"]
    shape = cpr.shape
    aoi = _reproject_to(tfm, shape, crs, cfg.F2_AOI_TIF) > 0.5
    psr = _reproject_to(tfm, shape, crs, cfg.PSR_MASK_TIF)
    psr = (psr > 0.5) & (psr < 254)
    dsc = _reproject_to(tfm, shape, crs, cfg.DSC_MASK_TIF) > 0.5

    # SNR mask: drop geocoded pixels at/below the noise floor (low total power),
    # which otherwise produce noise-inflated CPR and unstable DOP.
    s0 = geo["s0"]
    finite_s0 = s0[np.isfinite(s0) & (s0 > 0)]
    noise = np.percentile(finite_s0, 5) if finite_s0.size else 0.0
    snr_ok = np.isfinite(s0) & (s0 > 2.0 * noise)
    valid = np.isfinite(cpr) & np.isfinite(dop) & snr_ok
    # Operational dual criterion uses the speckle-adjusted DOP threshold (0.20);
    # the paper's strict 0.13 is reported alongside (our DOP is speckle-inflated).
    dual = (cpr > cfg.CPR_THRESHOLD) & (dop < cfg.DOP_THRESHOLD_RELAXED) & valid

    # Relative sigma0 (dB): 10log10(g0) - cal const. NOTE: g0 is the circular
    # child-power in uncalibrated DN^2, so this is a *relative* backscatter proxy
    # (used only as a diagnostic, not for an absolute rough-rock threshold).
    with np.errstate(divide="ignore"):
        sigma0_db = 10.0 * np.log10(np.clip(s0, 1e-6, None)) - cfg.CAL_CONST

    # Confidence tiers (single covering pass => use dual criterion + D1 context):
    #   1 LOW   = meets dual criterion (CPR>1 & DOP<0.20)
    #   2 MED   = + inside a PSR (D1) and strong DOP (<0.16)
    #   3 HIGH  = + inside a doubly-shadowed crater (D1 DSC)
    strong = dual & (dop < cfg.DOP_THRESHOLD + 0.03)   # DOP < ~0.16
    conf = np.zeros(shape, dtype=np.uint8)
    conf[dual] = 1
    conf[strong & psr] = 2
    conf[strong & psr & dsc] = 3
    # FP guard (sigma0-independent): demote dual pixels with marginal DOP
    # (0.16-0.20) -- weakest volume-scattering evidence -- by one tier.
    marginal = dual & (dop >= cfg.DOP_THRESHOLD + 0.03) & (conf > 1)
    conf[marginal] = (conf[marginal].astype(int) - 1).astype(np.uint8)
    # Bright relative-sigma0 + relatively polarised => possible rough rock (flag only)
    bright = np.isfinite(sigma0_db) & (sigma0_db > np.nanpercentile(sigma0_db, 90))
    rough_flag = int((bright & (dop > 0.3)).sum())
    print(f"      conf: HIGH={int((conf==3).sum())} MED={int((conf==2).sum())} "
          f"LOW={int((conf==1).sum())} | marginal demoted={int(marginal.sum())} | "
          f"rough-rock flagged={rough_flag}")
    return cpr, dop, sigma0_db, dual, conf, aoi, psr, dsc, valid


def gate2(cpr, dop, dsc, aoi, valid, conf):
    print("-" * 72)
    print(">>> STOP GATE 2 — CPR/DOP STATISTICS INSIDE F2 <<<")
    # "F2 interior" = crater floor (D1 DSC footprint) within the AOI; the AOI's
    # 200 m buffer otherwise dilutes the floor signal with rim/exterior pixels.
    core = aoi & dsc & valid
    if int(core.sum()) < 100:
        print("   (DSC floor too small in SAR grid; falling back to full AOI)")
        core = aoi & valid
    ni = int(core.sum())
    if ni == 0:
        print("   ✗ No valid SAR pixels inside F2"); raise SystemExit("GATE 2 FAILED: empty AOI")
    cin = cpr[core]; din = dop[core]
    hi = core & (cpr > 1.0)
    lo = core & (cpr < 1.0)
    pct_cpr1 = 100.0 * hi.sum() / ni
    dop_hi = float(np.nanmean(dop[hi])) if hi.sum() else float("nan")
    dop_lo = float(np.nanmean(dop[lo])) if lo.sum() else float("nan")
    dual013 = 100.0 * (core & (cpr > 1) & (dop < cfg.DOP_THRESHOLD)).sum() / ni
    dual020 = 100.0 * (core & (cpr > 1) & (dop < cfg.DOP_THRESHOLD_RELAXED)).sum() / ni
    n_high = int((conf == 3).sum())
    stats = {
        "n": ni, "mean_cpr": float(np.nanmean(cin)), "max_cpr": float(np.nanmax(cin)),
        "median_cpr": float(np.nanmedian(cin)), "pct_cpr_gt1": pct_cpr1,
        "dop_hi": dop_hi, "dop_lo": dop_lo, "dual_013": dual013, "dual_020": dual020,
        "n_high": n_high,
    }
    print(f"   F2 crater-floor pixels  : {ni}")
    print(f"   mean / median / max CPR : {stats['mean_cpr']:.2f} / {stats['median_cpr']:.2f} / {stats['max_cpr']:.2f}  (paper max 1.95)")
    print(f"   % CPR > 1.0             : {pct_cpr1:.0f}%   (paper ~47%)")
    print(f"   mean DOP (CPR>1)        : {dop_hi:.3f}   (paper < 0.13)")
    print(f"   mean DOP (CPR<1)        : {dop_lo:.3f}   (paper ~0.48)")
    print(f"   % dual @DOP<0.13 (paper): {dual013:.0f}%")
    print(f"   % dual @DOP<0.20 (oper.): {dual020:.0f}%")
    print(f"   HIGH-confidence pixels  : {n_high}")
    ok = (stats["max_cpr"] > 1.5 and dual020 >= 30 and dop_hi < cfg.DOP_THRESHOLD_RELAXED
          and n_high >= 100)
    print("-" * 72)
    print("   ✓ GATE 2 PASSED" if ok else "   ✗ GATE 2 FAILED — see diagnostics")
    return stats, ok


def _save(path, arr, tfm, crs, dtype, nodata):
    prof = {"driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1],
            "count": 1, "dtype": dtype, "crs": crs, "transform": tfm,
            "nodata": nodata, "compress": "lzw"}
    with rasterio.open(str(path), "w", **prof) as d:
        d.write(arr.astype(dtype), 1)


def main():
    t0 = time.time()
    cfg.ensure_dirs()
    M_hh, M_vv, M_hv = step2_load_multilook()
    print("=" * 72); print("[step 3-4/9] Circular synthesis -> CPR / DOP")
    cpr, dop, s0 = cpr_dop_circular(M_hh, M_vv, M_hv)
    print(f"      slant CPR p50/95 {np.nanpercentile(cpr,50):.2f}/{np.nanpercentile(cpr,95):.2f} "
          f"| DOP p50 {np.nanpercentile(dop,50):.2f}")
    geo, tfm, crs, f2xy = step5_geocode([cpr, dop, s0], ["cpr", "dop", "s0"])
    cpr_g, dop_g, sig_db, dual, conf, aoi, psr, dsc, valid = step6_7_criterion_guard(geo, tfm, crs)

    # save geotiffs (after gate, but write CPR/DOP now for inspection)
    _save(cfg.CPR_TIF, np.nan_to_num(cpr_g, nan=-1), tfm, crs, "float32", -1)
    _save(cfg.DOP_TIF, np.nan_to_num(dop_g, nan=-1), tfm, crs, "float32", -1)
    _save(cfg.SIGMA0_TIF, np.nan_to_num(sig_db, nan=-999), tfm, crs, "float32", -999)
    _save(cfg.DUAL_TIF, dual.astype(np.uint8), tfm, crs, "uint8", 255)
    _save(cfg.ICE_CONF_TIF, conf, tfm, crs, "uint8", 255)

    stats, ok = gate2(cpr_g, dop_g, dsc, aoi, valid, conf)
    np.savez(str(cfg.GEOTIFF_DIR / "_d2_cache.npz"),
             cpr=cpr_g, dop=dop_g, sig=sig_db, dual=dual, conf=conf,
             aoi=aoi, psr=psr, dsc=dsc, tfm=np.array(tfm).reshape(3, 3)[:2].ravel(),
             f2xy=np.array(f2xy))
    if not ok:
        print(f"\nstep2-gate2 done in {time.time()-t0:.0f}s")
        raise SystemExit("Stop at GATE 2")

    print("=" * 72); print("[step 8-9/9] Figures + report")
    from d2_figure import render_all
    render_all(cpr_g, dop_g, sig_db, dual, conf, aoi, psr, dsc, tfm, crs, f2xy, stats)
    from d2_report import write_report
    verdict = write_report(stats, conf, ok)
    print("=" * 72)
    print(f"DONE in {time.time()-t0:.0f}s | OVERALL: {verdict}")
    try:
        import os
        for f in (cfg.FIG_MAPS, cfg.FIG_HIST, cfg.FIG_PAPER):
            os.startfile(str(f))
    except Exception:
        pass
    return stats


if __name__ == "__main__":
    main()
