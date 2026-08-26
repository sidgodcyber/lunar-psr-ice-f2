"""Configuration for Deliverable 5 (ice volume estimation)."""

from __future__ import annotations

from pathlib import Path
import glob

import d2_config as d2

# Reuse D2 output grid (20 m, F2-centred stereographic) and D1 masks
CPR_TIF = d2.CPR_TIF
DOP_TIF = d2.DOP_TIF
SIGMA0_TIF = d2.SIGMA0_TIF
ICE_CONF_TIF = d2.ICE_CONF_TIF
F2_AOI_TIF = d2.F2_AOI_TIF
DSC_MASK_TIF = d2.DSC_MASK_TIF
GEOCODE_PIXEL_M = d2.GEOCODE_PIXEL_M     # 20 m ice-map grid

GEOTIFF_DIR = d2.GEOTIFF_DIR
FIG_DIR = d2.FIG_DIR
REPORT_DIR = d2.REPORT_DIR
PASS4_GEOM = d2.PASS4_GEOM
G_SLI_CSV = d2.G_SLI_CSV

ICE_FRACTION_TIF = GEOTIFF_DIR / "ice_fraction.tif"
FIG_VOLUME = FIG_DIR / "d5_ice_volume.png"
FIG_FORWARD = FIG_DIR / "d5_iem_forward_curve.png"
FIG_COMPARE = FIG_DIR / "d5_comparison.png"
REPORT_MD = REPORT_DIR / "d5_report.md"

# Physical constants
RHO_ICE = 917.0                          # kg/m^3
WAVELENGTH_L = 0.24                       # m
DEPTH_SCENARIOS_M = [1.0, 3.0, 5.0, 10.0]
CENTRAL_DEPTH_M = 5.0
DIELECTRIC_UNCERTAINTY = 0.5             # +/-50%

# F2 geometry (from D1)
F2_DIAMETER_M = d2.d1.F2_DIAMETER_M
F2_DEPTH_M = 174.0                        # measured in D1 (10 m DEM)


def g_oat_csv() -> Path:
    hits = glob.glob(str(PASS4_GEOM / "*_g_oat_xx_fp_xx_m65.csv"))
    return Path(hits[0]) if hits else Path("")
