"""Configuration for Deliverable 2 (DFSAR CPR/DOP ice detection on F2)."""

from __future__ import annotations

from pathlib import Path
import glob

import numpy as np

import d1_config as d1  # reuse F2 ground truth, paths, projection helper

BASE_DIR: Path = d1.BASE_DIR
SAR_DIR: Path = BASE_DIR / "data/raw/SAR"

# Pass 4 (20191105) is the acquisition that actually covers F2.
PASS4_DIR: Path = SAR_DIR / "pass4/data/calibrated/20191105"
PASS4_GEOM: Path = SAR_DIR / "pass4/geometry/calibrated/20191105"
PASS4_PREFIX: str = "ch2_sar_ncxl_20191105t180525404"


def sli(pol: str) -> Path:
    return PASS4_DIR / f"{PASS4_PREFIX}_d_sli_xx_fp_{pol}_m65.tif"


def sri(pol: str) -> Path:
    return PASS4_DIR / f"{PASS4_PREFIX}_d_sri_xx_fp_{pol}_m65.tif"


SRI_MASK: Path = PASS4_DIR / f"{PASS4_PREFIX}_d_sri_ma_fp_xx_m65.tif"
SRI_INC: Path = PASS4_DIR / f"{PASS4_PREFIX}_d_sri_in_fp_xx_m65.tif"
G_SLI_CSV: Path = PASS4_GEOM / f"{PASS4_PREFIX}_g_sli_xx_fp_xx_m65.csv"
SLI_XML: Path = PASS4_DIR / f"{PASS4_PREFIX}_d_sli_xx_fp_xx_m65.xml"
SRI_XML: Path = PASS4_DIR / f"{PASS4_PREFIX}_d_sri_xx_fp_xx_m65.xml"

# D1 outputs reused
DEM_TIF: Path = d1.DEM_TIF
F2_AOI_TIF: Path = d1.F2_AOI_TIF
PSR_MASK_TIF: Path = d1.PSR_MASK_TIF
DSC_MASK_TIF: Path = d1.DSC_MASK_TIF

# Outputs
GEOTIFF_DIR: Path = d1.GEOTIFF_DIR
FIG_DIR: Path = d1.FIG_DIR
REPORT_DIR: Path = d1.REPORT_DIR

CPR_TIF: Path = GEOTIFF_DIR / "cpr_pass4.tif"
DOP_TIF: Path = GEOTIFF_DIR / "dop_pass4.tif"
SIGMA0_TIF: Path = GEOTIFF_DIR / "sigma0_pass4.tif"
DUAL_TIF: Path = GEOTIFF_DIR / "dual_criterion_pass4.tif"
ICE_CONF_TIF: Path = GEOTIFF_DIR / "ice_confidence_f2.tif"
FIG_MAPS: Path = FIG_DIR / "d2_cpr_dop_maps.png"
FIG_HIST: Path = FIG_DIR / "d2_histograms.png"
FIG_PAPER: Path = FIG_DIR / "d2_paper_comparison.png"
REPORT_MD: Path = REPORT_DIR / "d2_report.md"

# ── SAR geometry / format (from XML) ─────────────────────────────────────────
SLI_AZ: int = 57880          # azimuth lines (slant range)
SLI_RNG: int = 512           # range samples
RANGE_PIXEL_M: float = 9.593359     # slant-range pixel spacing
CAL_CONST: float = 70.308868        # calibration constant (dB) from XML
INCIDENCE_DEG: float = 26.007991
TIE_AZ: int = 1810           # g_sli azimuth tie lines
TIE_RNG: int = 18            # g_sli range tie points

# ── Processing parameters ────────────────────────────────────────────────────
ML_AZ: int = 70              # azimuth multilook (heavy: az is oversampled ~3x)
ML_RNG: int = 4             # range multilook (slant 9.59 m -> ~38 m)
REFINED_LEE_WIN: int = 7
GEOCODE_PIXEL_M: float = 20.0       # target geocoded grid spacing

# ── Dual criterion (Sinha & Bharti 2026) ─────────────────────────────────────
CPR_THRESHOLD: float = 1.0
DOP_THRESHOLD: float = 0.13
DOP_THRESHOLD_RELAXED: float = 0.20  # gate tolerance (speckle inflates DOP)

R_MOON_M: float = d1.R_MOON_M
F2_LAT_DEG: float = d1.F2_LAT_DEG
F2_LON_DEG: float = d1.F2_LON_DEG
latlon_to_stereographic = d1.latlon_to_stereographic  # reuse D1 projection helper

LUNAR_GEO_PROJ4: str = "+proj=longlat +R=1737400 +no_defs"


def ensure_dirs() -> None:
    for d in (GEOTIFF_DIR, FIG_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
