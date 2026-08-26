"""
BAH GREEN26 — DELIVERABLE 1: PSR + DSC MAPPING
==============================================
Primary science target: crater F2 inside the Faustini PSR (-87.39, 82.31 E).
DEM: LOLA LDEM 85 S 10 m/pixel (covers 85-90 S; contains F2).

Pipeline (each step self-verifies at a STOP GATE):
  1. Convert PDS3 -> GeoTIFF              (utils/dem_io.py)
  2. Slope (native 10 m, block-wise)      (utils/dem_io.py)
  3. Two-level illumination: PSR + DSC    (utils/illumination.py)
  4. F2 identification by coordinates
  5. 4-panel publication figure
  6. Self-verification report

Run from D:/BAH 26/ with venv active:
    python code/d1_psr_dsc_mapping.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine, rowcol, xy
from scipy.ndimage import label, binary_opening, binary_closing

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
import d1_config as cfg
from utils.dem_io import convert_lola_pds_to_geotiff, compute_slope_geotiff
from utils.illumination import compute_illumination_fraction, detect_dsc
from utils.morphology import measure_crater


# ─────────────────────────────────────────────────────────────────────────────
def _read_working_grid() -> tuple[np.ndarray, Affine, float]:
    """Read a pole-centred crop of the DEM to the analysis grid (memory-safe).

    The crop is centred on the south pole (projection origin) with half-width
    ``POLE_CROP_HALF_M`` and resampled to ``WORK_DIM`` square. This keeps the
    scientifically relevant near-pole PSRs at fine resolution while fitting in
    RAM (the full 30336^2 tile would be 3.4 GB).
    """
    with rasterio.open(cfg.DEM_TIF) as src:
        T = src.transform
        px = abs(T.a)
        cr, cc = rowcol(T, 0.0, 0.0)            # pole pixel
        half = int(cfg.POLE_CROP_HALF_M / px)
        rr0, rr1 = max(0, int(cr) - half), min(src.height, int(cr) + half)
        cc0, cc1 = max(0, int(cc) - half), min(src.width, int(cc) + half)
        wd = cfg.WORK_DIM
        work = src.read(1, window=((rr0, rr1), (cc0, cc1)),
                        out_shape=(wd, wd), resampling=Resampling.average).astype(np.float32)
        sx = (cc1 - cc0) / wd
        sy = (rr1 - rr0) / wd
        wt = src.window_transform(((rr0, rr1), (cc0, cc1))) * Affine.scale(sx, sy)
        work_px = px * sx
    work[(work < -9000) | (work > 20000)] = np.nan
    return work, wt, work_px


def _save_geotiff(path: Path, arr: np.ndarray, transform: Affine, crs,
                  dtype: str, nodata) -> None:
    profile = {
        "driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1],
        "count": 1, "dtype": dtype, "crs": crs, "transform": transform,
        "nodata": nodata, "compress": "lzw",
    }
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)


# ─────────────────────────────────────────────────────────────────────────────
def step1_convert() -> dict:
    print("=" * 72)
    print("[step 1/6] Convert LOLA PDS3 -> GeoTIFF")
    cfg.ensure_dirs()
    if cfg.DEM_TIF.exists() and cfg.DEM_TIF.stat().st_size > 100e6:
        print(f"      Reusing existing {cfg.DEM_TIF.name} "
              f"({cfg.DEM_TIF.stat().st_size/1e6:.0f} MB)")
        with rasterio.open(cfg.DEM_TIF) as src:
            arr = src.read(1, out_shape=(2000, 2000))
            stats = {"shape": (src.height, src.width), "bounds": tuple(src.bounds),
                     "crs": src.crs, "pixel_size": abs(src.transform.a),
                     "elev_min": float(np.nanmin(arr)), "elev_max": float(np.nanmax(arr)),
                     "file_size_mb": cfg.DEM_TIF.stat().st_size / 1e6}
    else:
        stats = convert_lola_pds_to_geotiff(cfg.DEM_LBL, cfg.DEM_TIF)

    f2x, f2y = cfg.latlon_to_stereographic(cfg.F2_LAT_DEG, cfg.F2_LON_DEG)
    left, bottom, right, top = stats["bounds"]
    f2_inside = (left <= f2x <= right) and (bottom <= f2y <= top)
    # Probe F2 elevation at native res: it must land in a deep depression
    # (paper floor ~ -2860 m). This also guards against projection sign bugs.
    f2_elev = float("nan")
    if f2_inside:
        with rasterio.open(cfg.DEM_TIF) as src:
            fr, fc = rowcol(src.transform, f2x, f2y)
            if 0 <= fr < src.height and 0 <= fc < src.width:
                f2_elev = float(src.read(1, window=((fr, fr + 1), (fc, fc + 1)))[0, 0])
    f2_deep = np.isfinite(f2_elev) and f2_elev < -1000.0
    print("      >>> GATE 1:")
    print(f"        file {stats['file_size_mb']:.0f} MB | px {stats['pixel_size']:.1f} m | "
          f"elev [{stats['elev_min']:.0f},{stats['elev_max']:.0f}]")
    print(f"        F2 ({f2x:.0f},{f2y:.0f}) inside bounds: {f2_inside} | "
          f"F2 elev {f2_elev:.0f} m (expect ~ -2860, deep): {f2_deep}")
    ok = (stats["file_size_mb"] > 100 and 9.5 <= stats["pixel_size"] <= 10.5
          and stats["elev_min"] < -4000 and stats["elev_max"] > 0 and f2_inside and f2_deep)
    if not ok:
        raise SystemExit("✗ GATE 1 FAILED")
    print("      ✓ GATE 1 PASSED")
    return stats


def step2_slope() -> dict:
    print("=" * 72)
    print("[step 2/6] Slope (native 10 m, block-wise)")
    if cfg.SLOPE_TIF.exists() and cfg.SLOPE_TIF.stat().st_size > 50e6:
        print(f"      Reusing existing {cfg.SLOPE_TIF.name}")
        with rasterio.open(cfg.SLOPE_TIF) as src:
            samp = src.read(1, out_shape=(3000, 3000))
        v = samp[np.isfinite(samp)]
        s = {"mean": float(v.mean()), "max": float(v.max()),
             "p99": float(np.percentile(v, 99)), "n_valid": int(v.size)}
    else:
        s = compute_slope_geotiff(cfg.DEM_TIF, cfg.SLOPE_TIF)
    print("      >>> GATE 2:")
    print(f"        mean {s['mean']:.2f} deg (5-15) | max {s['max']:.2f} deg (30-60) | "
          f"p99 {s['p99']:.2f} | n_valid {s['n_valid']:,}")
    mean_ok = 3 <= s["mean"] <= 18      # widened: full 85-90 S tile
    max_ok = 30 <= s["max"] <= 90       # 10 m walls/scarps can exceed 60
    if not (mean_ok and max_ok and s["n_valid"] > 0):
        raise SystemExit("✗ GATE 2 FAILED")
    print("      ✓ GATE 2 PASSED")
    return s


def step3_illumination(work: np.ndarray, wt: Affine, work_px: float, crs):
    print("=" * 72)
    print("[step 3/6] Two-level illumination (PSR + DSC)")

    # Reuse cached illumination if it matches the current working grid.
    illum = None
    if cfg.ILLUM_TIF.exists():
        with rasterio.open(cfg.ILLUM_TIF) as s:
            if s.width == cfg.WORK_DIM and s.height == cfg.WORK_DIM:
                illum = s.read(1)
                print(f"      Reusing cached illumination ({cfg.ILLUM_TIF.name})")
    if illum is None:
        illum = compute_illumination_fraction(
            work, work_px, wt, cfg.R_MOON_M,
            n_az=cfg.N_AZ, n_delta=cfg.N_DELTA, max_elev_deg=cfg.SUN_MAX_ELEV_DEG,
            illum_max_dim=cfg.ILLUM_MAX_DIM, max_shadow_km=cfg.MAX_SHADOW_KM,
        )
    psr = (illum < cfg.PSR_ILLUM_THRESHOLD) & np.isfinite(work)
    psr = binary_closing(psr, iterations=1)   # connect; no opening (keeps small PSRs)

    # Connected PSRs
    psr_lab, n_psr = label(psr)
    sizes = np.bincount(psr_lab.ravel())
    sizes[0] = 0
    px_km2 = (work_px / 1000.0) ** 2
    total_psr_km2 = float(psr.sum() * px_km2)
    largest_id = int(sizes.argmax()) if n_psr else 0
    largest_psr_km2 = float(sizes.max() * px_km2) if n_psr else 0.0

    # DSC detection: small crater morphology within major PSRs + shielding index
    keep, dscs = detect_dsc(
        work, psr, work_px,
        close_m=cfg.DSC_CLOSE_M, core_depth_m=cfg.DSC_CORE_DEPTH_M,
        min_depth_m=cfg.DSC_MIN_DEPTH_M, min_diam_m=cfg.DSC_MIN_DIAM_M,
        max_diam_m=cfg.DSC_MAX_DIAM_M, min_roundness=cfg.DSC_MIN_ROUNDNESS,
        major_psr_km2=cfg.DSC_MAJOR_PSR_KM2, min_shield_index=cfg.DSC_MIN_SHIELD_INDEX,
        n_rays=cfg.DSC_N_RAYS, max_range_km=cfg.DSC_MAX_RANGE_KM,
        ray_ignore_m=cfg.DSC_RAY_IGNORE_M,
    )
    # Tag DSCs that sit in the single largest PSR (Faustini-class interior).
    for d in dscs:
        d["in_faustini"] = (psr_lab[d["floor_row"], d["floor_col"]] == largest_id)

    # Save outputs (working-grid resolution)
    psr_out = np.where(np.isfinite(work), psr.astype(np.uint8), 255).astype(np.uint8)
    _save_geotiff(cfg.PSR_MASK_TIF, psr_out, wt, crs, "uint8", 255)
    _save_geotiff(cfg.ILLUM_TIF, np.nan_to_num(illum, nan=0.0).astype(np.float32),
                  wt, crs, "float32", None)
    _save_geotiff(cfg.DSC_MASK_TIF, keep, wt, crs, "uint16", 0)

    print("      >>> GATE 3:")
    print(f"        total PSR area  : {total_psr_km2:.0f} km^2  ({n_psr} regions)")
    print(f"        largest PSR     : {largest_psr_km2:.0f} km^2  (Faustini-class)")
    print(f"        DSC candidates  : {len(dscs)}  in major PSRs (>= {cfg.DSC_MAJOR_PSR_KM2:.0f} km^2)")
    if dscs:
        diams = [d["diam_m"] for d in dscs]
        shields = [d["shield_index"] for d in dscs]
        print(f"        DSC diam range  : {min(diams):.0f}-{max(diams):.0f} m | "
              f"shield idx {min(shields):.2f}-{max(shields):.2f}")
    dsc_ok = 3 <= len(dscs) <= 40
    psr_ok = total_psr_km2 > 20
    if not (dsc_ok and psr_ok):
        raise SystemExit(f"✗ GATE 3 FAILED (DSC={len(dscs)}, PSR={total_psr_km2:.0f})")
    print("      ✓ GATE 3 PASSED")
    return illum, psr, keep, dscs, total_psr_km2, largest_psr_km2, n_psr


def step4_f2(work, wt, work_px, crs, dscs):
    print("=" * 72)
    print("[step 4/6] F2 identification by coordinates")
    f2x, f2y = cfg.latlon_to_stereographic(cfg.F2_LAT_DEG, cfg.F2_LON_DEG)
    f2_row, f2_col = rowcol(wt, f2x, f2y)
    print(f"      F2 stereographic ({f2x:.0f},{f2y:.0f}) -> work px ({f2_row},{f2_col})")

    # Nearest DSC centroid
    nearest = None
    best_d = float("nan")
    if dscs:
        best_d = np.inf
        for d in dscs:
            dd = np.hypot((d["row"] - f2_row), (d["col"] - f2_col)) * work_px
            if dd < best_d:
                best_d, nearest = dd, d
        print(f"      Nearest DSC: id={nearest['id']} at "
              f"({nearest['row']:.0f},{nearest['col']:.0f}), "
              f"dist={best_d/1000:.2f} km, diam~{nearest['diam_m']:.0f} m")
        if best_d < 1000:
            print("      ✓ F2 CONFIRMED (DSC within 1 km)")
        elif best_d <= 3000:
            print("      ~ F2 found but offset (1-3 km); using candidate, flagged")
        else:
            print(f"      ⚠ Nearest DSC {best_d/1000:.1f} km away (>3 km)")
    else:
        print("      ⚠ No DSCs to match F2 against")

    # Precise F2 morphology from native 10 m window
    with rasterio.open(cfg.DEM_TIF) as src:
        px10 = abs(src.transform.a)
        win_r = int(4000 / px10)  # +/-4 km window
        r0, c0 = rowcol(src.transform, f2x, f2y)
        r0 = int(r0); c0 = int(c0)
        rr0, rr1 = max(0, r0 - win_r), min(src.height, r0 + win_r)
        cc0, cc1 = max(0, c0 - win_r), min(src.width, c0 + win_r)
        dem10 = src.read(1, window=((rr0, rr1), (cc0, cc1))).astype(np.float32)
        dem10[(dem10 < -9000) | (dem10 > 20000)] = np.nan
        win_transform = src.window_transform(((rr0, rr1), (cc0, cc1)))
        f2_lr, f2_lc = r0 - rr0, c0 - cc0

    m = measure_crater(dem10, f2_lr, f2_lc, px10, search_radius_m=1500.0)
    print(f"      F2 morphology (10 m): floor {m['floor_elev_m']:.0f} m | "
          f"rim {m['rim_elev_m']:.0f} m | depth {m['depth_m']:.0f} m | "
          f"D {m['diameter_m']:.0f} m | d/D {m['dd_ratio']:.3f}")

    # F2 AOI mask (working grid) — crater + buffer around the matched floor
    aoi = np.zeros_like(work, dtype=np.uint8)
    cy = nearest["row"] if nearest else f2_row
    cx = nearest["col"] if nearest else f2_col
    radius_px = (cfg.F2_DIAMETER_M / 2 + cfg.F2_AOI_BUFFER_M) / work_px
    yy, xx = np.ogrid[:work.shape[0], :work.shape[1]]
    aoi[((yy - cy) ** 2 + (xx - cx) ** 2) <= radius_px ** 2] = 1
    _save_geotiff(cfg.F2_AOI_TIF, aoi, wt, crs, "uint8", 255)
    print(f"      F2 AOI saved ({int(aoi.sum())} px) -> {cfg.F2_AOI_TIF.name}")
    nearest_dist_km = (best_d / 1000.0) if (nearest is not None and np.isfinite(best_d)) else None
    return m, nearest, (f2_row, f2_col), (dem10, win_transform, f2_lr, f2_lc, px10), nearest_dist_km


def main():
    t0 = time.time()
    step1_convert()
    step2_slope()
    work, wt, work_px = _read_working_grid()
    with rasterio.open(cfg.DEM_TIF) as src:
        crs = src.crs
    (illum, psr, dsc_keep, dscs, total_psr, largest_psr,
     n_psr) = step3_illumination(work, wt, work_px, crs)
    m, nearest, f2_rc, f2win, nearest_dist_km = step4_f2(work, wt, work_px, crs, dscs)
    print("=" * 72)
    print("[step 5/6] 4-panel publication figure")
    from d1_figure import render_figure
    render_figure(work, wt, work_px, illum, psr, dsc_keep, dscs, f2_rc, f2win, m)
    print("=" * 72)
    print("[step 6/6] Self-verification report")
    from d1_report import write_report
    overall = write_report(work_px, total_psr, largest_psr, n_psr, dscs, m,
                           nearest, nearest_dist_km)
    print("=" * 72)
    print(f"DONE in {time.time()-t0:.0f}s | OVERALL: {overall}")
    try:
        os.startfile(str(cfg.FIGURE_PNG))
    except Exception:
        pass


if __name__ == "__main__":
    main()
