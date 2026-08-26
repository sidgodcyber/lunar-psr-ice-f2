"""
LOLA PDS3 -> GeoTIFF conversion utilities.

Converts a LOLA LDEM PDS3 product (.lbl + .img) to a properly georeferenced,
compressed GeoTIFF in the lunar south polar stereographic projection.

The LDEM polar products are:
  - 16-bit signed LSB (little-endian) integers
  - HEIGHT_in_metres = DN * SCALING_FACTOR   (NOTE: do NOT add OFFSET; OFFSET
    is the reference-sphere radius and only applies to PLANETARY_RADIUS, not
    height. GDAL's PDS driver tends to add OFFSET, which silently produces
    planetary-radius values ~1.7e6 m instead of elevations -- this module
    parses the label manually to avoid that pitfall.)
  - true-at-pole spherical polar stereographic, R = 1737400 m
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

# Lunar reference sphere radius (A_AXIS_RADIUS in the label), metres.
R_MOON_M: float = 1737400.0

# proj4 string for lunar south polar stereographic, true at the pole.
LUNAR_SPS_PROJ4: str = (
    "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 "
    f"+a={R_MOON_M:.0f} +b={R_MOON_M:.0f} +units=m +no_defs"
)


def compute_slope_geotiff(
    dem_tif: Path,
    out_tif: Path,
    block_rows: int = 2048,
) -> Dict[str, float]:
    """Compute slope (degrees) from a DEM GeoTIFF, block-wise, at native res.

    Processes the DEM in horizontal strips with a 1-row halo so slope is
    computed at full resolution without loading the whole array. NaN propagates
    through the gradient, suppressing spurious nodata-edge slopes.

    Args:
        dem_tif: Input DEM GeoTIFF.
        out_tif: Output slope GeoTIFF (degrees, float32).
        block_rows: Strip height in rows.

    Returns:
        Stats dict: mean, max, p99 slope (degrees) over valid pixels, plus n_valid.
    """
    with rasterio.open(str(dem_tif)) as src:
        profile = src.profile.copy()
        h, w = src.height, src.width
        px = abs(src.transform.a)
        profile.update(dtype="float32", nodata=float("nan"), compress="lzw",
                       predictor=3, BIGTIFF="YES")

        total = 0.0
        count = 0
        smax = -np.inf
        # reservoir-ish: collect a decimated sample for percentile
        sample_vals = []

        out_tif.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(str(out_tif), "w", **profile) as dst:
            for r0 in range(0, h, block_rows):
                r1 = min(r0 + block_rows, h)
                rr0 = max(0, r0 - 1)
                rr1 = min(h, r1 + 1)
                dem = src.read(1, window=((rr0, rr1), (0, w))).astype(np.float64)
                gy, gx = np.gradient(dem, px)
                slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
                # trim halo
                top = r0 - rr0
                slope_core = slope[top:top + (r1 - r0), :]
                dst.write(slope_core, 1, window=((r0, r1), (0, w)))

                valid = slope_core[np.isfinite(slope_core)]
                if valid.size:
                    total += float(valid.sum())
                    count += int(valid.size)
                    smax = max(smax, float(valid.max()))
                    sample_vals.append(valid[::997])  # sparse sample

    sample = np.concatenate(sample_vals) if sample_vals else np.array([0.0])
    return {
        "mean": total / count if count else float("nan"),
        "max": smax,
        "p99": float(np.percentile(sample, 99)),
        "n_valid": count,
    }


def parse_pds_label(lbl_path: Path) -> Dict[str, object]:
    """Parse the subset of PDS3 label keywords needed for conversion.

    Args:
        lbl_path: Path to the ``.lbl`` file.

    Returns:
        Mapping of keyword -> value (ints/floats parsed where numeric).
    """
    numeric_keys = {
        "LINES",
        "LINE_SAMPLES",
        "SAMPLE_BITS",
        "SCALING_FACTOR",
        "OFFSET",
        "MAP_SCALE",
        "MAP_RESOLUTION",
        "LINE_PROJECTION_OFFSET",
        "SAMPLE_PROJECTION_OFFSET",
        "MAXIMUM_LATITUDE",
        "MINIMUM_LATITUDE",
        "A_AXIS_RADIUS",
        "CENTER_LATITUDE",
        "CENTER_LONGITUDE",
    }
    meta: Dict[str, object] = {}
    with open(lbl_path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if "=" not in line or line.startswith("/*"):
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().rstrip(",")
            # Strip PDS unit suffixes like "10 <m/pix>" and quotes.
            value = value.split("<")[0].strip().strip('"').strip("'")
            if key in numeric_keys:
                try:
                    meta[key] = float(value) if ("." in value or "e" in value.lower()) else int(value)
                except ValueError:
                    pass
            else:
                meta[key] = value
    return meta


def _dtype_from_label(meta: Dict[str, object]) -> np.dtype:
    """Map PDS SAMPLE_TYPE/SAMPLE_BITS to a numpy dtype."""
    sample_type = str(meta.get("SAMPLE_TYPE", "LSB_INTEGER")).upper()
    bits = int(meta.get("SAMPLE_BITS", 16))
    little = "LSB" in sample_type or "PC" in sample_type
    endian = "<" if little else ">"
    if bits == 16:
        return np.dtype(f"{endian}i2")
    if bits == 32:
        return np.dtype(f"{endian}i4")
    if bits == 8:
        return np.dtype("u1")
    raise ValueError(f"Unsupported SAMPLE_BITS={bits}")


def _build_transform(meta: Dict[str, object]) -> Affine:
    """Build the rasterio affine transform from PDS projection offsets.

    LOLA GDR DSMAP convention (pixel-registered, 1-based pixels):
        X(sample) = (sample - 1 - SAMPLE_PROJECTION_OFFSET) * MAP_SCALE   [pixel centre]
        Y(line)   = (LINE_PROJECTION_OFFSET - (line - 1)) * MAP_SCALE     [pixel centre]
    The affine maps the *upper-left corner* of pixel (0,0), hence the half-pixel shift.
    """
    scale = float(meta["MAP_SCALE"])
    spo = float(meta["SAMPLE_PROJECTION_OFFSET"])
    lpo = float(meta["LINE_PROJECTION_OFFSET"])
    # Centre of pixel (row=0, col=0) i.e. PDS line=1, sample=1:
    x_centre_00 = (1 - 1 - spo) * scale
    y_centre_00 = (lpo - 0) * scale
    # Upper-left corner = centre shifted by half a pixel.
    x_ul = x_centre_00 - scale / 2.0
    y_ul = y_centre_00 + scale / 2.0
    return Affine(scale, 0.0, x_ul, 0.0, -scale, y_ul)


def convert_lola_pds_to_geotiff(lbl_path: Path, out_tif: Path) -> Dict[str, object]:
    """Convert a LOLA PDS3 (.lbl + .img) product to a compressed GeoTIFF.

    Args:
        lbl_path: Path to the ``.lbl`` label file (``.img`` assumed alongside).
        out_tif:  Output GeoTIFF path (parents created as needed).

    Returns:
        Stats dict: ``shape``, ``bounds`` (left, bottom, right, top), ``crs``,
        ``pixel_size``, ``elev_min``, ``elev_max``, ``file_size_mb``.
    """
    lbl_path = Path(lbl_path)
    out_tif = Path(out_tif)
    print(f"      Parsing label: {lbl_path.name}")
    meta = parse_pds_label(lbl_path)

    lines = int(meta["LINES"])
    samples = int(meta["LINE_SAMPLES"])
    scaling = float(meta.get("SCALING_FACTOR", 1.0))
    dtype = _dtype_from_label(meta)
    img_name = str(meta.get("FILE_NAME", lbl_path.with_suffix(".img").name))
    img_path = lbl_path.parent / img_name
    if not img_path.exists():  # case-insensitive fallback
        img_path = lbl_path.with_suffix(".img")

    print(f"      Reading IMG:   {img_path.name} ({lines}x{samples}, {dtype})")
    expected = lines * samples * dtype.itemsize
    actual = img_path.stat().st_size
    if actual < expected:
        raise ValueError(
            f"IMG truncated: {actual} bytes on disk, expected {expected}."
        )

    # Memory-map the raw int16 image so we never hold the whole 3.4 GB float
    # array in RAM; convert and write in row-blocks instead.
    raw = np.memmap(img_path, dtype=dtype, mode="r", shape=(lines, samples))

    crs = CRS.from_proj4(LUNAR_SPS_PROJ4)
    transform = _build_transform(meta)

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": samples,
        "height": lines,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": np.float32(np.nan),
        "compress": "lzw",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "YES",
    }

    scaling32 = np.float32(scaling)
    block_rows = 1024
    elev_min = np.inf
    elev_max = -np.inf
    print(f"      Writing GeoTIFF in row-blocks of {block_rows}: {out_tif}")
    with rasterio.open(str(out_tif), "w", **profile) as dst:
        for r0 in range(0, lines, block_rows):
            r1 = min(r0 + block_rows, lines)
            block = np.asarray(raw[r0:r1, :], dtype=np.float32) * scaling32
            # Defensive nodata masking outside plausible lunar topo range.
            block[(block < -9000.0) | (block > 20000.0)] = np.nan
            finite = block[np.isfinite(block)]
            if finite.size:
                elev_min = min(elev_min, float(finite.min()))
                elev_max = max(elev_max, float(finite.max()))
            dst.write(block, 1, window=((r0, r1), (0, samples)))
    del raw

    left = transform.c
    top = transform.f
    right = left + samples * transform.a
    bottom = top + lines * transform.e  # transform.e is negative
    stats: Dict[str, object] = {
        "shape": (lines, samples),
        "bounds": (left, bottom, right, top),
        "crs": crs,
        "pixel_size": abs(transform.a),
        "elev_min": elev_min,
        "elev_max": elev_max,
        "file_size_mb": out_tif.stat().st_size / 1e6,
    }
    return stats
