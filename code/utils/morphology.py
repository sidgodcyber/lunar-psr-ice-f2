"""Terrain morphology helpers: slope, local minima, crater measurement."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import minimum_filter


def compute_slope_degrees(dem: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """Slope magnitude in degrees from a DEM patch.

    Uses numpy gradient (central differences). NaN in the DEM propagates to
    NaN slope at and adjacent to nodata, which correctly suppresses spurious
    edge gradients (the previous nan_to_num approach inflated slopes at
    data/nodata borders).

    Args:
        dem: 2-D elevation array (metres); may contain NaN.
        pixel_size_m: Ground sample distance (metres).

    Returns:
        float32 slope array in degrees, same shape as ``dem``.
    """
    gy, gx = np.gradient(dem.astype(np.float64), pixel_size_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    return slope.astype(np.float32)


def local_minima_mask(dem: np.ndarray, window_px: int) -> np.ndarray:
    """Boolean mask of pixels that are the minimum within a square window.

    Args:
        dem: 2-D elevation array (NaN allowed).
        window_px: Side length of the neighbourhood in pixels.

    Returns:
        Boolean array, True at local-minimum pixels (NaN excluded).
    """
    filled = np.where(np.isnan(dem), np.inf, dem)
    mins = minimum_filter(filled, size=max(int(window_px), 3), mode="nearest")
    return (filled == mins) & np.isfinite(dem)


def measure_crater(
    dem: np.ndarray,
    row: int,
    col: int,
    pixel_size_m: float,
    search_radius_m: float = 1500.0,
) -> dict:
    """Measure crater diameter / depth around a seed floor pixel.

    Estimates the rim by the radial elevation maximum and the floor by the
    local minimum, using radial profiles in 16 directions.

    Args:
        dem: 2-D elevation array (metres).
        row, col: Seed pixel (near the crater floor).
        pixel_size_m: Ground sample distance (metres).
        search_radius_m: Max radius to search for the rim (metres).

    Returns:
        dict with floor_elev_m, rim_elev_m, depth_m, diameter_m, dd_ratio,
        floor_row, floor_col.
    """
    h, w = dem.shape
    max_r = int(search_radius_m / pixel_size_m)

    # Refine floor: minimum within a small window around the seed.
    r0, r1 = max(0, row - max_r // 3), min(h, row + max_r // 3 + 1)
    c0, c1 = max(0, col - max_r // 3), min(w, col + max_r // 3 + 1)
    sub = dem[r0:r1, c0:c1]
    if np.all(np.isnan(sub)):
        floor_row, floor_col, floor_elev = row, col, float("nan")
    else:
        fr, fc = np.unravel_index(np.nanargmin(sub), sub.shape)
        floor_row, floor_col = r0 + fr, c0 + fc
        floor_elev = float(dem[floor_row, floor_col])

    # Radial rim search: for each of 16 azimuths, find the elevation peak.
    rim_elevs = []
    rim_radii = []
    for az in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        prev = floor_elev
        peak = floor_elev
        peak_r = 0.0
        for rr in range(1, max_r):
            pr = int(round(floor_row + rr * np.sin(az)))
            pc = int(round(floor_col + rr * np.cos(az)))
            if not (0 <= pr < h and 0 <= pc < w):
                break
            v = dem[pr, pc]
            if np.isnan(v):
                break
            if v > peak:
                peak = v
                peak_r = rr * pixel_size_m
            # stop after we've clearly come down from a rim
            if v < peak - 50 and peak_r > 0:
                break
            prev = v
        if peak_r > 0:
            rim_elevs.append(peak)
            rim_radii.append(peak_r)

    if rim_elevs:
        rim_elev = float(np.median(rim_elevs))
        radius_m = float(np.median(rim_radii))
    else:
        rim_elev = float("nan")
        radius_m = float("nan")

    depth = rim_elev - floor_elev if np.isfinite(rim_elev) else float("nan")
    diameter = 2.0 * radius_m if np.isfinite(radius_m) else float("nan")
    dd = depth / diameter if (np.isfinite(depth) and diameter and diameter > 0) else float("nan")

    return {
        "floor_elev_m": floor_elev,
        "rim_elev_m": rim_elev,
        "depth_m": depth,
        "diameter_m": diameter,
        "dd_ratio": dd,
        "floor_row": int(floor_row),
        "floor_col": int(floor_col),
    }
