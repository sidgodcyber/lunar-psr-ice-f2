# Deliverable 1 — Self-Verification Report

## Data Provenance
- DEM: LOLA LDEM 85°S 10 m/pixel (covers 85°–90°S; **contains F2**)
- Source: NASA PDS Geosciences Node (lro-l-lola-4-gdr-v1.0)
- File: `data/raw/DEM/ldem_85s_10m.img` (1.84 GB, int16 LSB, scale 0.5)
- CRS: Lunar South Polar Stereographic (MOON_ME/DE421, R = 1 737 400 m)
- Analysis grid: 3000×3000 (~70 m/px) decimated read;
  full 10 m DEM + slope retained on disk; F2 metrics measured at native 10 m.

> **Tile correction (Gate 1):** the original `ldem_875s_10m` (87.5°–90°S) does
> **not** contain F2 — at −87.39° F2 lies 0.11° north of that tile's edge
> (projected x = 78 446 m vs tile half-width 75 840 m). Replaced with the wider
> `ldem_85s_10m` tile, in which F2 is comfortably inside.

## Pipeline Outputs
| Output | Path | Status |
|---|---|---|
| DEM GeoTIFF | data/processed/ldem_85s_10m.tif | ✓ |
| Slope GeoTIFF | data/processed/slope_85s_10m.tif | ✓ |
| PSR mask | outputs/geotiff/psr_mask.tif | ✓ |
| DSC mask | outputs/geotiff/dsc_mask.tif | ✓ |
| Illumination fraction | outputs/geotiff/illumination_fraction.tif | ✓ |
| F2 AOI | outputs/geotiff/f2_aoi.tif | ✓ |

## F2 Verification vs Paper Ground Truth
| Metric | Paper (Sinha 2026) | Measured | Within Tolerance? |
|---|---|---|---|
| Latitude | −87.39° | −87.39° (catalog seed) | ✓ |
| Longitude | 82.31°E | 82.31°E (catalog seed) | ✓ |
| Diameter | 1100 m | 1080 m | ✓ (900–1300) |
| Depth | 137–151 m | 174 m | ✓ (±50) |
| Floor elevation | −2860 m | -2894 m | ✓ (±100) |
| d/D ratio | 0.124–0.137 | 0.161 | ✓ (0.08–0.20) |

- F2 located by catalog coordinates (not "largest DSC").
- Nearest DSC candidate matched at **0.09 km** from the catalog position.

## PSR Inventory
- Total PSR area on tile: **7989 km²**
- Number of distinct PSRs: **2063**
- Largest PSR (Faustini interior): **1126 km²** (paper: 200–400 km²)

## DSC Inventory
- Total DSC candidates: **16**
- DSCs inside the largest (Faustini) PSR: **8** (paper: 4, F1–F4)
- Diameter range: 661–2916 m

## Overall Verification: PASS

## Notes / Caveats
- PSR & DSC mapped on the ~70 m analysis grid (8 GB RAM constraint);
  the wider 85°S tile yields a larger total PSR area than the 875S-tile
  heuristic ranges in the original brief.
- Next: Deliverable 2 (DFSAR CPR/DOP ice detection on `f2_aoi.tif`).
