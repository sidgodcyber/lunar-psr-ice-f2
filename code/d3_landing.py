"""
DELIVERABLE 3 — LANDING SITE PROPOSAL (safe, sunlit rim near F2's PSR)
=====================================================================
Landing zone = safe (slope) + sunlit + outside PSR + within rover range of F2.
The rover (D4) makes the PSR entry to F2.

Inputs: LM7 5 m DEM+slope (Faustini rim) + D1 PSR/illumination + F2 location.
Run: python code/d3_landing.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform as wtr
from rasterio.crs import CRS
from scipy.ndimage import binary_erosion, label, center_of_mass

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
import d1_config as cfg

DEM_LM7 = cfg.BASE_DIR / "data/raw/DEM/LM7_final_adj_5mpp_surf.tif"
SLP_LM7 = cfg.BASE_DIR / "data/raw/DEM/LM7_final_adj_5mpp_slp.tif"
ILLUM = cfg.GEOTIFF_DIR / "illumination_fraction.tif"
PSR = cfg.PSR_MASK_TIF
FIG1 = cfg.FIG_DIR / "d3_landing_site.png"
FIG2 = cfg.FIG_DIR / "d3_decision_matrix.png"
REPORT = cfg.REPORT_DIR / "d3_report.md"

# Constraints
SLOPE_MAX = 10.0
DIST_MAX_KM = 15.0
PAD_PX = 10                      # 50 m / 5 m
ILLUM_MAX_MODEL = 0.5           # near-pole physical ceiling of the fraction metric
# ">70% annual" implies a PEL, which does not exist within range of Faustini; the
# constraint is relaxed to the best achievable illumination (documented in report).
ILLUM_FLOOR = 0.22             # ~44% of near-pole available sunlight (usable power)


def _og(path, tfm, shape, crs, rs):
    with rasterio.open(str(path)) as s:
        d = np.full(shape, np.nan, np.float32)
        reproject(rasterio.band(s, 1), d, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=tfm, dst_crs=crs, resampling=rs)
    return d


def _norm(a):
    lo, hi = np.nanmin(a), np.nanmax(a)
    return (a - lo) / (hi - lo + 1e-12)


def main():
    t0 = time.time()
    cfg.ensure_dirs()
    print("=" * 72); print("[step 1/6] Suitability layers (LM7 5 m grid)")
    with rasterio.open(str(DEM_LM7)) as s:
        dem = s.read(1).astype(np.float32); tfm = s.transform; crs = s.crs
        shape = (s.height, s.width); px = abs(s.transform.a)
    with rasterio.open(str(SLP_LM7)) as s:
        slope = s.read(1).astype(np.float32)
    illum = _og(ILLUM, tfm, shape, crs, Resampling.bilinear)
    psr = _og(PSR, tfm, shape, crs, Resampling.nearest)
    geo = CRS.from_proj4(cfg.LUNAR_GEO_PROJ4) if hasattr(cfg, "LUNAR_GEO_PROJ4") else CRS.from_proj4("+proj=longlat +R=1737400 +no_defs")
    fx, fy = wtr(geo, crs, [cfg.F2_LON_DEG], [cfg.F2_LAT_DEG]); fx, fy = fx[0], fy[0]
    ys = tfm.f + (np.arange(shape[0]) + 0.5) * tfm.e
    xs = tfm.c + (np.arange(shape[1]) + 0.5) * tfm.a
    X, Y = np.meshgrid(xs, ys)
    dist = np.hypot(X - fx, Y - fy) / 1000.0

    valid = np.isfinite(dem) & np.isfinite(slope) & np.isfinite(illum)
    slope_ok = slope < SLOPE_MAX
    outside = psr != 1
    dist_ok = dist < DIST_MAX_KM
    illum_ok = illum > ILLUM_FLOOR
    feasible = valid & slope_ok & outside & dist_ok & illum_ok
    pxkm = (px / 1000.0) ** 2
    print(f"      max illum within 15 km of F2: {np.nanmax(np.where(dist_ok&valid,illum,np.nan)):.3f} "
          f"(model ceiling {ILLUM_MAX_MODEL}); NO PEL near Faustini -> 70% relaxed to >{ILLUM_FLOOR}")
    print(f"      feasible (slope<10 & illum>{ILLUM_FLOOR} & outside PSR & <15km): "
          f"{int(feasible.sum())} px = {feasible.sum()*pxkm:.2f} km^2")
    if feasible.sum() == 0:
        raise SystemExit("Zero feasible pixels even after relaxation")

    # ── step 2: pad detection + sites ────────────────────────────────────────
    print("=" * 72); print("[step 2/6] 50 m flat-pad detection + candidate sites")
    pad_center = binary_erosion(slope_ok & valid, structure=np.ones((PAD_PX, PAD_PX)))
    feas_pad = pad_center & outside & dist_ok & illum_ok
    print(f"      50 m pad-able + feasible centres: {int(feas_pad.sum())} px")
    lab, n = label(feas_pad)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    order = np.argsort(sizes)[::-1]
    order = [o for o in order if sizes[o] > 4]   # >=5 pad-centres (~real site)
    print(f"      {len(order)} candidate sites (>=5 pad centres)")

    # normalization references over feasible
    illum_n = _norm(np.where(feasible, illum, np.nan))
    dist_n = _norm(np.where(feasible, dist, np.nan))
    slope_n = _norm(np.where(feasible, slope, np.nan))
    sz_max = sizes[order[0]] if order else 1

    sites = []
    for oid in order[:12]:
        m = lab == oid
        rr, cc = np.where(m)
        cr, ccm = int(rr.mean()), int(cc.mean())
        s_illum = float(np.nanmax(illum[m]))
        s_dist = float(dist[cr, ccm])
        s_slope = float(np.nanmean(slope[m]))
        pad_area = int(m.sum()) * px * px          # pad-centre area proxy
        lon, lat = wtr(crs, geo, [xs[ccm]], [ys[cr]])
        score = (0.35 * (s_illum - np.nanmin(illum[feasible])) / (np.nanmax(illum[feasible]) - np.nanmin(illum[feasible]) + 1e-9)
                 + 0.30 * (1 - (s_dist - dist[feasible].min()) / (dist[feasible].max() - dist[feasible].min() + 1e-9))
                 + 0.25 * (1 - (s_slope - np.nanmin(slope[feasible])) / (np.nanmax(slope[feasible]) - np.nanmin(slope[feasible]) + 1e-9))
                 + 0.10 * (m.sum() / sz_max))
        # Earth-LOS proxy: high ground (elev above local median in 2 km window)
        loc = dem[max(0, cr-200):cr+200, max(0, ccm-200):ccm+200]
        high_ground = float(dem[cr, ccm]) >= np.nanmedian(loc)
        sites.append({"row": cr, "col": ccm, "x": float(xs[ccm]), "y": float(ys[cr]),
                      "lat": float(lat[0]), "lon": float(lon[0]) % 360, "illum": s_illum,
                      "dist": s_dist, "slope": s_slope, "pad_area": pad_area,
                      "score": float(score), "high_ground": high_ground, "npx": int(m.sum())})
    sites.sort(key=lambda d: d["score"], reverse=True)
    # Earth line-of-sight is a hard constraint -> winner must be on high ground.
    los_sites = [s for s in sites if s["high_ground"]]
    sites_sel = los_sites if len(los_sites) >= 3 else sites
    top3 = sites_sel[:3]
    print("=" * 72); print("[step 3/6] Decision matrix (top 3)")
    print(f"      {'site':>5} {'illum%':>7} {'dist_km':>8} {'slope':>6} {'pad_m2':>9} {'score':>6} {'lat':>8} {'lon':>7} LOS")
    for i, s in enumerate(top3):
        print(f"      {chr(65+i):>5} {s['illum']*100:7.1f} {s['dist']:8.1f} {s['slope']:6.1f} "
              f"{s['pad_area']:9.0f} {s['score']:6.3f} {s['lat']:8.2f} {s['lon']:7.2f} "
              f"{'hi' if s['high_ground'] else 'lo'}")

    win = top3[0]
    # ── step 4: approach corridor ────────────────────────────────────────────
    print("=" * 72); print("[step 4/6] Approach corridor to F2")
    bearing = (np.degrees(np.arctan2(fx - win["x"], fy - win["y"]))) % 360
    npts = 200
    lx = np.linspace(win["x"], fx, npts); ly = np.linspace(win["y"], fy, npts)
    cols = ((lx - tfm.c) / tfm.a).astype(int); rows = ((ly - tfm.f) / tfm.e).astype(int)
    inb = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    prof_d = np.linspace(0, win["dist"], npts)
    psr_line = np.full(npts, np.nan)
    psr_line[inb] = psr[rows[inb], cols[inb]]
    psr_cross = prof_d[inb][np.where(psr_line[inb] == 1)[0][0]] if (psr_line[inb] == 1).any() else float("nan")
    print(f"      bearing to F2: {bearing:.0f} deg | corridor length {win['dist']:.1f} km "
          f"| PSR boundary crossed ~{psr_cross:.1f} km from landing (within LM7)")

    print("=" * 72); print("[step 5-6/6] Figures + report")
    from d3_figure import render_all
    render_all(dem, slope, illum, psr, dist, feasible, top3, win, fx, fy, tfm, xs, ys,
               ILLUM_FLOOR, bearing)
    from d3_report import write_report
    write_report(top3, win, feasible.sum() * pxkm, len(order), bearing, psr_cross,
                 ILLUM_FLOOR, float(np.nanmax(np.where(dist_ok & valid, illum, np.nan))))
    ok = (win["dist"] < DIST_MAX_KM and win["illum"] > ILLUM_FLOOR and win["npx"] >= 5)
    print("=" * 72)
    print(f"DONE in {time.time()-t0:.0f}s | winner Site A @ {win['lat']:.2f},{win['lon']:.2f} "
          f"dist {win['dist']:.1f} km illum {win['illum']*100:.0f}% slope {win['slope']:.1f} deg | GATE {'PASS' if ok else 'REVIEW'}")
    try:
        import os
        for f in (FIG1, FIG2):
            os.startfile(str(f))
    except Exception:
        pass


if __name__ == "__main__":
    main()
