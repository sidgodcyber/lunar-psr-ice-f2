"""DFSAR SAR I/O and speckle utilities (complex SLI, multilook, Lee filter)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter


def read_complex(path: Path) -> np.ndarray:
    """Read a ComplexLSB8 (CFloat32) SLI GeoTIFF as a complex64 2-D array."""
    with rasterio.open(str(path)) as s:
        a = s.read(1)
    if not np.iscomplexobj(a):
        # Fallback: some drivers expose two bands (real, imag)
        with rasterio.open(str(path)) as s:
            if s.count >= 2:
                re = s.read(1).astype(np.float32)
                im = s.read(2).astype(np.float32)
                a = re + 1j * im
            else:
                a = a.astype(np.complex64)
    return a.astype(np.complex64)


def read_real(path: Path) -> np.ndarray:
    """Read a single-band real raster as float32 (uint16/float kept as values)."""
    with rasterio.open(str(path)) as s:
        return s.read(1).astype(np.float32)


def multilook_real(a: np.ndarray, laz: int, lrng: int) -> np.ndarray:
    """Boxcar multilook (block mean) a real array by (laz, lrng)."""
    h = (a.shape[0] // laz) * laz
    w = (a.shape[1] // lrng) * lrng
    a = a[:h, :w]
    return a.reshape(h // laz, laz, w // lrng, lrng).mean(axis=(1, 3))


def multilook_complex(a: np.ndarray, laz: int, lrng: int) -> np.ndarray:
    """Boxcar multilook (block mean) a complex array by (laz, lrng)."""
    h = (a.shape[0] // laz) * laz
    w = (a.shape[1] // lrng) * lrng
    a = a[:h, :w]
    return a.reshape(h // laz, laz, w // lrng, lrng).mean(axis=(1, 3))


def lee_filter(img: np.ndarray, size: int = 7, enl: float = 70.0) -> np.ndarray:
    """Lee speckle filter for an intensity image (multiplicative noise model).

    Args:
        img: intensity (linear) array.
        size: window side.
        enl: equivalent number of looks (sets noise variance Cu^2 = 1/ENL).

    Returns:
        Filtered intensity, same shape.
    """
    img = np.nan_to_num(img.astype(np.float32))
    mean = uniform_filter(img, size)
    mean_sq = uniform_filter(img * img, size)
    var = np.clip(mean_sq - mean * mean, 0, None)
    cu2 = 1.0 / enl
    ci2 = np.divide(var, mean * mean + 1e-12)
    w = 1.0 - cu2 / np.clip(ci2, 1e-6, None)
    w = np.clip(w, 0.0, 1.0)
    return mean + w * (img - mean)


def load_tie_grid(csv_path: Path, tie_az: int, tie_rng: int,
                  sli_az: int, sli_rng: int) -> Tuple[np.ndarray, np.ndarray,
                                                      np.ndarray, np.ndarray]:
    """Load the g_sli tie-point grid and its SLI pixel coordinates.

    The CSV is a regular ``tie_az x tie_rng`` subsample (row-major, range fastest)
    of the SLI grid. Returns the tie-point azimuth/range pixel positions and the
    lat/lon grids, all shaped (tie_az, tie_rng).

    Returns:
        (az_px[tie_az], rng_px[tie_rng], lat[tie_az,tie_rng], lon[tie_az,tie_rng])
    """
    d = np.genfromtxt(str(csv_path), delimiter=",", skip_header=1)
    n = tie_az * tie_rng
    d = d[:n]
    lat = d[:, 0].reshape(tie_az, tie_rng)
    lon = d[:, 1].reshape(tie_az, tie_rng)
    az_px = np.linspace(0, sli_az - 1, tie_az)
    rng_px = np.linspace(0, sli_rng - 1, tie_rng)
    return az_px, rng_px, lat, lon
