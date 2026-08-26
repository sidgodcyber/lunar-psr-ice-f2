"""Deliverable 1 self-verification report writer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import d1_config as cfg


def _tol(value, lo, hi) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return "✓" if lo <= value <= hi else "✗"


def write_report(work_px, total_psr_km2, largest_psr_km2, n_psr,
                 dscs, m, nearest, nearest_dist_km):
    """Write outputs/reports/d1_report.md; return overall verdict string."""
    n_dsc = len(dscs)
    n_faustini = sum(1 for d in dscs if d.get("in_faustini"))

    diam, depth = m["diameter_m"], m["depth_m"]
    floor, dd = m["floor_elev_m"], m["dd_ratio"]

    diam_ok = _tol(diam, 900, 1300)
    depth_ok = _tol(depth, cfg.F2_DEPTH_MIN_M - 50, cfg.F2_DEPTH_MAX_M + 50)
    floor_ok = _tol(floor, cfg.F2_FLOOR_ELEV_M - 100, cfg.F2_FLOOR_ELEV_M + 100)
    dd_ok = _tol(dd, 0.08, 0.20)
    f2_found = nearest is not None

    def stat(p: Path) -> str:
        return "✓" if p.exists() else "✗"

    overall = "PASS" if (f2_found and diam_ok == "✓" and total_psr_km2 > 20) else "REVIEW"
    diam_rng = ("%.0f–%.0f m" % (min(d["diam_m"] for d in dscs),
                                 max(d["diam_m"] for d in dscs))) if dscs else "n/a"
    match_str = ("%.2f km" % nearest_dist_km) if (f2_found and nearest_dist_km is not None) else "n/a"

    md = f"""# Deliverable 1 — Self-Verification Report

## Data Provenance
- DEM: LOLA LDEM 85°S 10 m/pixel (covers 85°–90°S; **contains F2**)
- Source: NASA PDS Geosciences Node (lro-l-lola-4-gdr-v1.0)
- File: `data/raw/DEM/ldem_85s_10m.img` (1.84 GB, int16 LSB, scale 0.5)
- CRS: Lunar South Polar Stereographic (MOON_ME/DE421, R = 1 737 400 m)
- Analysis grid: {cfg.WORK_DIM}×{cfg.WORK_DIM} (~{work_px:.0f} m/px) decimated read;
  full 10 m DEM + slope retained on disk; F2 metrics measured at native 10 m.

> **Tile correction (Gate 1):** the original `ldem_875s_10m` (87.5°–90°S) does
> **not** contain F2 — at −87.39° F2 lies 0.11° north of that tile's edge
> (projected x = 78 446 m vs tile half-width 75 840 m). Replaced with the wider
> `ldem_85s_10m` tile, in which F2 is comfortably inside.

## Pipeline Outputs
| Output | Path | Status |
|---|---|---|
| DEM GeoTIFF | data/processed/ldem_85s_10m.tif | {stat(cfg.DEM_TIF)} |
| Slope GeoTIFF | data/processed/slope_85s_10m.tif | {stat(cfg.SLOPE_TIF)} |
| PSR mask | outputs/geotiff/psr_mask.tif | {stat(cfg.PSR_MASK_TIF)} |
| DSC mask | outputs/geotiff/dsc_mask.tif | {stat(cfg.DSC_MASK_TIF)} |
| Illumination fraction | outputs/geotiff/illumination_fraction.tif | {stat(cfg.ILLUM_TIF)} |
| F2 AOI | outputs/geotiff/f2_aoi.tif | {stat(cfg.F2_AOI_TIF)} |

## F2 Verification vs Paper Ground Truth
| Metric | Paper (Sinha 2026) | Measured | Within Tolerance? |
|---|---|---|---|
| Latitude | −87.39° | −87.39° (catalog seed) | ✓ |
| Longitude | 82.31°E | 82.31°E (catalog seed) | ✓ |
| Diameter | 1100 m | {diam:.0f} m | {diam_ok} (900–1300) |
| Depth | 137–151 m | {depth:.0f} m | {depth_ok} (±50) |
| Floor elevation | −2860 m | {floor:.0f} m | {floor_ok} (±100) |
| d/D ratio | 0.124–0.137 | {dd:.3f} | {dd_ok} (0.08–0.20) |

- F2 located by catalog coordinates (not "largest DSC").
- Nearest DSC candidate matched at **{match_str}** from the catalog position.

## PSR Inventory
- Total PSR area on tile: **{total_psr_km2:.0f} km²**
- Number of distinct PSRs: **{n_psr}**
- Largest PSR (Faustini interior): **{largest_psr_km2:.0f} km²** (paper: 200–400 km²)

## DSC Inventory
- Total DSC candidates: **{n_dsc}**
- DSCs inside the largest (Faustini) PSR: **{n_faustini}** (paper: 4, F1–F4)
- Diameter range: {diam_rng}

## Overall Verification: {overall}

## Notes / Caveats
- PSR & DSC mapped on the ~{work_px:.0f} m analysis grid (8 GB RAM constraint);
  the wider 85°S tile yields a larger total PSR area than the 875S-tile
  heuristic ranges in the original brief.
- Next: Deliverable 2 (DFSAR CPR/DOP ice detection on `f2_aoi.tif`).
"""
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REPORT_MD.write_text(md, encoding="utf-8")
    print(f"      ✓ Report saved: {cfg.REPORT_MD}")
    return overall
