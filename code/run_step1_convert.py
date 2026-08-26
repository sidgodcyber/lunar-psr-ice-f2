"""STEP 1 runner: convert LOLA PDS3 DEM to GeoTIFF and verify STOP GATE 1."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
import d1_config as cfg
from utils.dem_io import convert_lola_pds_to_geotiff


def main() -> int:
    print("=" * 72)
    print("STEP 1/6 — Convert LOLA PDS3 -> GeoTIFF")
    print("=" * 72)
    cfg.ensure_dirs()

    if cfg.DEM_TIF.exists() and cfg.DEM_TIF.stat().st_size > 100e6:
        print(f"      GeoTIFF already exists ({cfg.DEM_TIF.stat().st_size/1e6:.0f} MB) — "
              "reading metadata instead of re-converting.")
        with rasterio.open(cfg.DEM_TIF) as src:
            t = src.transform
            left, bottom, right, top = src.bounds
            # decimated scan for elev range (low memory)
            arr = src.read(1, out_shape=(2000, 2000))
            stats = {
                "shape": (src.height, src.width),
                "bounds": (left, bottom, right, top),
                "crs": src.crs,
                "pixel_size": abs(t.a),
                "elev_min": float(np.nanmin(arr)),
                "elev_max": float(np.nanmax(arr)),
                "file_size_mb": cfg.DEM_TIF.stat().st_size / 1e6,
            }
    else:
        stats = convert_lola_pds_to_geotiff(cfg.DEM_LBL, cfg.DEM_TIF)

    shape = stats["shape"]
    left, bottom, right, top = stats["bounds"]
    crs = stats["crs"]
    px = stats["pixel_size"]
    emin = stats["elev_min"]
    emax = stats["elev_max"]
    size_mb = stats["file_size_mb"]

    print("\n" + "-" * 72)
    print(">>> STOP GATE 1 — VERIFY CONVERSION <<<")
    print("-" * 72)

    checks: list[tuple[str, bool, str]] = []

    # 1. File exists and > 100 MB
    exists = cfg.DEM_TIF.exists()
    checks.append((
        f"File exists & >100MB  ({size_mb:.0f} MB)",
        exists and size_mb > 100,
        str(cfg.DEM_TIF),
    ))

    # 2. CRS south polar stereographic, R=1737400
    crs_wkt = crs.to_wkt() if crs else ""
    crs_ok = ("Stereographic" in crs_wkt or "stere" in str(crs).lower()) and (
        "1737400" in crs_wkt or "1737400" in crs.to_proj4()
    )
    checks.append((f"CRS = S-polar stereographic, R=1737400", crs_ok, crs.to_proj4()))

    # 3. Pixel size ~ 10 m
    px_ok = 9.5 <= px <= 10.5
    checks.append((f"Pixel size ~10 m  ({px:.2f} m)", px_ok, ""))

    # 4. Elevation range plausible
    elev_ok = (emin < -4000.0) and (emax > 0.0)
    checks.append((
        f"Elev range plausible  (min {emin:.0f} < -4000, max {emax:.0f} > 0)",
        elev_ok,
        "",
    ))

    # 5. Bounds cover the 85S tile (half-width ~151680 m)
    bounds_ok = (left <= -150000) and (right >= 150000) and (bottom <= -150000) and (top >= 150000)
    checks.append((
        f"Bounds cover tile  X[{left:.0f},{right:.0f}] Y[{bottom:.0f},{top:.0f}]",
        bounds_ok,
        "",
    ))

    # 6. F2 inside bounds
    f2x, f2y = cfg.latlon_to_stereographic(cfg.F2_LAT_DEG, cfg.F2_LON_DEG)
    with rasterio.open(cfg.DEM_TIF) as src:
        f2_row, f2_col = rasterio.transform.rowcol(src.transform, f2x, f2y)
        in_array = (0 <= f2_row < src.height) and (0 <= f2_col < src.width)
        # also fetch the elevation at F2 centre as a sanity probe
        if in_array:
            win = src.read(1, window=((max(f2_row - 1, 0), f2_row + 2),
                                      (max(f2_col - 1, 0), f2_col + 2)))
            f2_elev = float(np.nanmean(win))
        else:
            f2_elev = float("nan")
    f2_inside = (left <= f2x <= right) and (bottom <= f2y <= top) and in_array
    checks.append((
        f"F2 inside bounds  (x={f2x:.0f}, y={f2y:.0f} -> px row={f2_row}, col={f2_col})",
        f2_inside,
        f"F2 centre elevation probe ~ {f2_elev:.0f} m",
    ))

    # Report
    print()
    all_pass = True
    for desc, ok, note in checks:
        mark = "[PASS]" if ok else "[FAIL]"
        all_pass &= ok
        print(f"  {mark} {desc}")
        if note:
            print(f"         {note}")

    print("\n" + "-" * 72)
    if all_pass:
        print("✓ DEM CONVERSION VERIFIED — proceeding to illumination model")
        return 0
    print("✗ STOP GATE 1 FAILED — halting. See failed checks above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
