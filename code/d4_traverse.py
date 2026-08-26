"""
DELIVERABLE 4 — ROVER TRAVERSE PATH DESIGN (D3 landing site -> F2 ice)
=====================================================================
A* on a slope-weighted, PSR-penalised cost grid:
    cost(move) = distance * (1 + 3*tan(slope_dest)) * (5 if PSR_dest else 1)
Impassable: slope > 25 deg.

PRIMARY ENGINEERING CHALLENGE (per brief): the ~9 km PSR crossing (no solar power,
battery only). Battery-hours for the PSR dash are computed explicitly; if infeasible,
staging/opportunity-charging is analysed, and if none exists that is reported as a
valid finding (argues for a hopper / lander-hop architecture).

Run: python code/d4_traverse.py
"""

from __future__ import annotations

import sys
import time
import heapq
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine, rowcol, xy
from rasterio.warp import reproject, Resampling, transform as wtr
from rasterio.crs import CRS

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
import d1_config as cfg
import d2_config as cfg2

# Mission assumptions
SPEED_SUN = 0.10        # m/s in sunlight
SPEED_PSR = 0.05        # m/s in PSR (cautious)
BATTERY_H = 5.0         # single-charge endurance (mid of 4-6 h), no solar in PSR
SLOPE_MAX = 25.0
SLOPE_MAX_RELAX = 30.0
GRAZE_ILLUM = 0.02      # any illumination >= this counts as opportunity charging

SITE_A_LAT, SITE_A_LON = -87.87, 82.66

FIG1 = cfg.FIG_DIR / "d4_traverse.png"
FIG2 = cfg.FIG_DIR / "d4_waypoints.png"
PATH_TIF = cfg2.GEOTIFF_DIR / "traverse_path.tif"
WPT_CSV = cfg.REPORT_DIR / "d4_waypoints.csv"
REPORT = cfg.REPORT_DIR / "d4_report.md"


def astar(slope, psr, impassable, start, goal, px):
    rows, cols = slope.shape
    N = rows * cols
    g = np.full(N, np.inf, np.float64)
    parent = np.full(N, -1, np.int64)
    tan_s = np.tan(np.radians(np.clip(slope, 0, 89)))
    mult = np.where(psr, 5.0, 1.0)
    si, gi = start[0] * cols + start[1], goal[0] * cols + goal[1]
    gr, gc = goal
    g[si] = 0.0
    openh = [(0.0, si)]
    steps = [(-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
             (0, -1, 1.0), (0, 1, 1.0),
             (1, -1, math.sqrt(2)), (1, 0, 1.0), (1, 1, math.sqrt(2))]
    hyp = math.hypot
    while openh:
        f, cur = heapq.heappop(openh)
        if cur == gi:
            break
        r, c = divmod(cur, cols)
        gcur = g[cur]
        if f - hyp(r - gr, c - gc) * px > gcur + 1e-6:
            continue
        for dr, dc, dd in steps:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not impassable[nr, nc]:
                ni = nr * cols + nc
                move = dd * px * (1.0 + 3.0 * tan_s[nr, nc]) * mult[nr, nc]
                ng = gcur + move
                if ng < g[ni]:
                    g[ni] = ng
                    parent[ni] = cur
                    heapq.heappush(openh, (ng + hyp(nr - gr, nc - gc) * px, ni))
    if not np.isfinite(g[gi]):
        return None, None
    path = []
    cur = gi
    while cur != -1:
        path.append(divmod(cur, cols))
        cur = parent[cur]
    return path[::-1], float(g[gi])


def _window(src, x0, x1, y0, y1):
    T = src.transform
    r_top, c_left = rowcol(T, x0, y1)
    r_bot, c_right = rowcol(T, x1, y0)
    r_top, c_left = max(0, int(r_top)), max(0, int(c_left))
    r_bot, c_right = min(src.height, int(r_bot)), min(src.width, int(c_right))
    win = ((r_top, r_bot), (c_left, c_right))
    arr = src.read(1, window=win)
    wt = src.window_transform(win)
    return arr, wt


def _reproj(path, wt, shape, crs, rs=Resampling.nearest):
    with rasterio.open(str(path)) as s:
        d = np.full(shape, np.nan, np.float32)
        reproject(rasterio.band(s, 1), d, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=wt, dst_crs=crs, resampling=rs)
    return d


def naive_line(start, goal):
    n = int(np.hypot(goal[0] - start[0], goal[1] - start[1])) + 1
    rr = np.linspace(start[0], goal[0], n).round().astype(int)
    cc = np.linspace(start[1], goal[1], n).round().astype(int)
    return list(zip(rr, cc))


def path_metrics(path, slope, psr, px):
    seg = [np.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1]) * px for i in range(len(path)-1)]
    length = float(np.sum(seg))
    inpsr = np.array([psr[r, c] for r, c in path], bool)
    psr_len = float(sum(seg[i] for i in range(len(seg)) if inpsr[i+1]))
    slopes = np.array([slope[r, c] for r, c in path])
    return length, psr_len, float(slopes.max()), float(slopes.mean()), inpsr, seg


def main():
    t0 = time.time()
    cfg.ensure_dirs()
    print("=" * 72); print("[step 1/8] Build cost grid (10 m, bbox of start+goal)")
    sx, sy = cfg.latlon_to_stereographic(SITE_A_LAT, SITE_A_LON)
    with rasterio.open(str(cfg2.ICE_CONF_TIF)) as s:
        conf = s.read(1); ct = s.transform
    hi = np.argwhere(conf == 3)
    gx = ct.c + hi[:, 1].mean() * ct.a; gy = ct.f + hi[:, 0].mean() * ct.e

    mg = max(0.2 * abs(gx - sx), 0.2 * abs(gy - sy), 2000)
    x0, x1 = min(sx, gx) - mg, max(sx, gx) + mg
    y0, y1 = min(sy, gy) - mg, max(sy, gy) + mg
    with rasterio.open(str(cfg.SLOPE_TIF)) as s:
        slope, wt = _window(s, x0, x1, y0, y1); crs = s.crs
    with rasterio.open(str(cfg.DEM_TIF)) as s:
        dem, _ = _window(s, x0, x1, y0, y1)
    shape = slope.shape; px = abs(wt.a)
    psr = _reproj(cfg2.PSR_MASK_TIF, wt, shape, crs) == 1
    illum = _reproj(cfg.ILLUM_TIF, wt, shape, crs, Resampling.bilinear)
    conf_g = _reproj(cfg2.ICE_CONF_TIF, wt, shape, crs)

    slope = np.nan_to_num(slope, nan=90.0)
    impassable = slope > SLOPE_MAX
    start = tuple(int(v) for v in rowcol(wt, sx, sy))
    # goal = nearest HIGH-conf passable cell to the ice centroid
    gr0, gc0 = rowcol(wt, gx, gy); gr0, gc0 = int(gr0), int(gc0)
    hic = np.argwhere((conf_g == 3) & (~impassable))
    if len(hic) == 0:
        hic = np.argwhere((conf_g >= 2) & (~impassable))
    goal = tuple(hic[np.argmin(np.hypot(hic[:, 0]-gr0, hic[:, 1]-gc0))])
    sl = np.hypot(goal[0]-start[0], goal[1]-start[1]) * px / 1000
    print(f"      grid {shape} @ {px:.0f} m | start {start} slope {slope[start]:.1f} | "
          f"goal {goal} | straight-line {sl:.1f} km | impassable {100*impassable.mean():.0f}%")

    print("=" * 72); print("[step 2/8] A* pathfinding")
    t = time.time()
    path, cost = astar(slope, psr, impassable, start, goal, px)
    relaxed = False
    if path is None:
        print("      no path at 25 deg -> relaxing to 30 deg")
        impassable = slope > SLOPE_MAX_RELAX; relaxed = True
        path, cost = astar(slope, psr, impassable, start, goal, px)
    print(f"      A* done in {time.time()-t:.0f}s | path cells {len(path) if path else 0} | cost {cost:.0f}")
    if path is None:
        raise SystemExit("No path found even at 30 deg")

    print("=" * 72); print("[step 3/8] Path metrics")
    length, psr_len, smax, smean, inpsr, seg = path_metrics(path, slope, psr, px)
    # time model
    tvec = [seg[i] / (SPEED_PSR if inpsr[i+1] else SPEED_SUN) for i in range(len(seg))]
    total_h = sum(tvec) / 3600
    psr_h = sum(tvec[i] for i in range(len(seg)) if inpsr[i+1]) / 3600
    print(f"      length {length/1000:.1f} km | PSR {psr_len/1000:.1f} km ({100*psr_len/length:.0f}%) | "
          f"max slope {smax:.1f} | mean {smean:.1f} | time {total_h:.0f} h (PSR {psr_h:.0f} h)")

    print("=" * 72); print("[step 4/8] Battery / PSR-dash feasibility (PRIMARY challenge)")
    dash_h = psr_len / SPEED_PSR / 3600
    feasible = dash_h <= BATTERY_H
    # PSR entry point = first inpsr cell along path
    entry_idx = int(np.argmax(inpsr)) if inpsr.any() else len(path)-1
    er, ec = path[entry_idx]; ex, ey = xy(wt, er, ec)
    elon, elat = wtr(crs, CRS.from_proj4(cfg2.LUNAR_GEO_PROJ4), [ex], [ey])
    # opportunity charging: any PSR-portion cell with illum >= GRAZE_ILLUM
    graze = [(i) for i in range(len(path)) if inpsr[i] and illum[path[i]] >= GRAZE_ILLUM]
    entry_km = sum(seg[:entry_idx]) / 1000
    print(f"      PSR dash {psr_len/1000:.1f} km @ {SPEED_PSR} m/s = {dash_h:.0f} h in shadow "
          f"vs battery {BATTERY_H} h -> {'FEASIBLE' if feasible else 'INFEASIBLE on single charge'}")
    print(f"      PSR entry ~{elat[0]:.2f}S {elon[0]%360:.2f}E ({entry_km:.1f} km from start)")
    print(f"      opportunity-charging cells inside PSR (illum>={GRAZE_ILLUM}): {len(graze)}")
    staging = "none (true PSR: no in-shadow charging possible)" if not graze else f"{len(graze)} grazing-light cells"
    verdict = ("single-charge traverse INFEASIBLE; no in-PSR charging -> recommend hopper/lander-hop"
               if (not feasible and not graze) else
               ("feasible on battery" if feasible else "needs staging (grazing light available)"))

    print("=" * 72); print("[step 5/8] Naive straight-line comparison")
    npath = [p for p in naive_line(start, goal) if not impassable[p]]
    nlen, npsr, nsmax, nsmean, ninp, nseg = path_metrics(naive_line(start, goal), slope, psr, px)
    # naive crosses impassable?
    naive_blocked = any(impassable[p] for p in naive_line(start, goal))
    print(f"      naive: length {nlen/1000:.1f} km | max slope {nsmax:.1f} | crosses impassable(>{SLOPE_MAX if not relaxed else SLOPE_MAX_RELAX}deg)={naive_blocked}")

    # ── outputs ──────────────────────────────────────────────────────────────
    print("=" * 72); print("[step 6-8] waypoints, figures, report")
    _save_path_tif(path, wt, crs, shape)
    wpts = _waypoints(path, seg, inpsr, slope, illum, wt, crs)
    _write_csv(wpts)
    metrics = dict(length=length, psr_len=psr_len, smax=smax, smean=smean, total_h=total_h,
                   psr_h=psr_h, cost=cost, dash_h=dash_h, feasible=feasible, battery=BATTERY_H,
                   entry_lat=float(elat[0]), entry_lon=float(elon[0]) % 360,
                   entry_km=sum(seg[:entry_idx])/1000, n_graze=len(graze), staging=staging,
                   verdict=verdict, relaxed=relaxed,
                   nlen=nlen, nsmax=nsmax, npsr=npsr, naive_blocked=naive_blocked,
                   start=start, goal=goal, sl=sl)
    from d4_figure import render_all
    render_all(dem, slope, psr, illum, path, naive_line(start, goal), start, goal,
               wt, crs, inpsr, seg, metrics)
    from d4_report import write_report
    write_report(metrics, wpts)
    print("=" * 72)
    print(f"DONE in {time.time()-t0:.0f}s | {verdict}")
    try:
        import os
        for f in (FIG1, FIG2):
            os.startfile(str(f))
    except Exception:
        pass


def _save_path_tif(path, wt, crs, shape):
    arr = np.zeros(shape, np.uint8)
    for r, c in path:
        arr[r, c] = 1
    prof = {"driver": "GTiff", "height": shape[0], "width": shape[1], "count": 1,
            "dtype": "uint8", "crs": crs, "transform": wt, "nodata": 255, "compress": "lzw"}
    with rasterio.open(str(PATH_TIF), "w", **prof) as d:
        d.write(arr, 1)


def _waypoints(path, seg, inpsr, slope, illum, wt, crs):
    geo = CRS.from_proj4(cfg2.LUNAR_GEO_PROJ4)
    cum = 0.0; cumt = 0.0; out = []
    every = max(1, int(500 / abs(wt.a)))   # ~every 500 m
    for i, (r, c) in enumerate(path):
        if i > 0:
            cum += seg[i-1]
            cumt += seg[i-1] / (SPEED_PSR if inpsr[i] else SPEED_SUN)
        if i % every == 0 or i == len(path)-1:
            x, y = xy(wt, r, c); lon, lat = wtr(crs, geo, [x], [y])
            out.append({"lat": float(lat[0]), "lon": float(lon[0]) % 360, "x": x, "y": y,
                        "cum_m": cum, "slope": float(slope[r, c]),
                        "in_psr": int(bool(inpsr[i])), "cum_h": cumt/3600,
                        "illum": float(illum[r, c])})
    return out


def _write_csv(wpts):
    import csv
    with open(str(WPT_CSV), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lat", "lon", "x", "y", "cum_m", "slope", "in_psr", "cum_h", "illum"])
        w.writeheader()
        for wp in wpts:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in wp.items()})
    print(f"      [ok] {WPT_CSV.name} ({len(wpts)} waypoints)")


if __name__ == "__main__":
    main()
