"""Shared configuration and constants for Deliverable 1 (PSR + DSC mapping)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(r"D:/BAH 26")

DEM_LBL: Path = BASE_DIR / "data/raw/DEM/ldem_85s_10m.lbl"
DEM_TIF: Path = BASE_DIR / "data/processed/ldem_85s_10m.tif"
SLOPE_TIF: Path = BASE_DIR / "data/processed/slope_85s_10m.tif"

GEOTIFF_DIR: Path = BASE_DIR / "outputs/geotiff"
FIG_DIR: Path = BASE_DIR / "outputs/figures"
REPORT_DIR: Path = BASE_DIR / "outputs/reports"

PSR_MASK_TIF: Path = GEOTIFF_DIR / "psr_mask.tif"
DSC_MASK_TIF: Path = GEOTIFF_DIR / "dsc_mask.tif"
ILLUM_TIF: Path = GEOTIFF_DIR / "illumination_fraction.tif"
F2_AOI_TIF: Path = GEOTIFF_DIR / "f2_aoi.tif"

FIGURE_PNG: Path = FIG_DIR / "d1_psr_dsc_map.png"
REPORT_MD: Path = REPORT_DIR / "d1_report.md"

# ── Physical constants ───────────────────────────────────────────────────────
R_MOON_M: float = 1737400.0          # lunar reference sphere radius (m)
SUN_MAX_ELEV_DEG: float = 1.54       # max solar elevation at the pole (obliquity)

# ── F2 crater ground truth (Sinha & Bharti 2026, Table 1) ────────────────────
F2_LAT_DEG: float = -87.39
F2_LON_DEG: float = 82.31            # positive East
F2_DIAMETER_M: float = 1100.0
F2_DEPTH_MIN_M: float = 137.0
F2_DEPTH_MAX_M: float = 151.0
F2_FLOOR_ELEV_M: float = -2860.0
F2_DD_MIN: float = 0.124
F2_DD_MAX: float = 0.137
F2_WALL_SLOPE_MIN: float = 20.0
F2_WALL_SLOPE_MAX: float = 27.0

# ── Analysis region (pole-centred crop; full 30336^2 cannot fit in 8 GB RAM) ─
# All named PSRs (Faustini, Shoemaker, Haworth) and F2 (rho~79 km) lie within
# ~90 km of the pole, where the near-pole solar model is valid.
POLE_CROP_HALF_M: float = 105000.0   # half-width of pole-centred analysis crop
WORK_DIM: int = 3000                 # analysis grid side (~70 m/px over the crop)

# ── Illumination ─────────────────────────────────────────────────────────────
PSR_ILLUM_THRESHOLD: float = 0.005   # illumination fraction below this => PSR
N_AZ: int = 40                       # solar azimuth samples per lunar day
N_DELTA: int = 7                     # seasonal (sub-solar lat) samples per year
ILLUM_MAX_DIM: int = 1100            # shadow-scan grid (~190 m/px)
MAX_SHADOW_KM: float = 35.0          # long range so big crater rims shadow floors

# ── DSC detection (morphology of small craters within major PSRs) ────────────
DSC_CLOSE_M: float = 2500.0          # black-top-hat scale (fills pits up to this)
DSC_CORE_DEPTH_M: float = 40.0       # shallow core threshold (for crater extent)
DSC_MIN_DEPTH_M: float = 120.0       # min crater peak depth (deep/fresh craters)
DSC_MIN_DIAM_M: float = 500.0        # min DSC diameter
DSC_MAX_DIAM_M: float = 3000.0       # max DSC diameter
DSC_MIN_ROUNDNESS: float = 0.45      # circularity filter
DSC_MAJOR_PSR_KM2: float = 800.0     # host PSR >= this => Faustini/Shoemaker/Haworth class
DSC_MIN_SHIELD_INDEX: float = 0.9    # fraction of horizon shielded from sunlit terrain
DSC_N_RAYS: int = 36                 # rays for double-shielding index
DSC_MAX_RANGE_KM: float = 5.0        # ray reach for double-shielding index
DSC_RAY_IGNORE_M: float = 500.0      # ignore terrain within this (own rim)

F2_AOI_BUFFER_M: float = 200.0       # buffer around F2 for the AOI mask


def latlon_to_stereographic(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """Convert planetocentric lat/lon to lunar south polar stereographic (m).

    True-at-pole spherical stereographic, matching the LOLA GDR products.

    Args:
        lat_deg: Latitude in degrees (negative for south).
        lon_deg: Longitude in degrees, positive East.

    Returns:
        (x, y) in metres in the projection.
    """
    colat = np.radians(90.0 + lat_deg)          # colatitude from south pole
    rho = 2.0 * R_MOON_M * np.tan(colat / 2.0)
    # South-polar stereographic (proj convention, verified vs rasterio.warp):
    #   x = rho*sin(lon),  y = rho*cos(lon)   (NOTE +cos, not -cos)
    x = rho * np.sin(np.radians(lon_deg))
    y = rho * np.cos(np.radians(lon_deg))
    return float(x), float(y)


def ensure_dirs() -> None:
    """Create all output directories."""
    for d in (DEM_TIF.parent, GEOTIFF_DIR, FIG_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
