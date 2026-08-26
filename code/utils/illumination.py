"""Two-level illumination model for PSR and doubly-shadowed-crater mapping.

Level 1 (PSR): annual illumination fraction from a horizon shadow scan.
Level 2 (DSC): doubly-shielded test -- a PSR floor pixel that cannot see any
               directly illuminated terrain along low-elevation rays.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

from .morphology import local_minima_mask


def _shift2d(a: np.ndarray, dr: int, dc: int, fill: float) -> np.ndarray:
    """Return b where b[i,j] = a[i+dr, j+dc], out-of-range filled with ``fill``."""
    out = np.full_like(a, fill)
    h, w = a.shape
    sr0, sr1 = max(0, dr), min(h, h + dr)
    sc0, sc1 = max(0, dc), min(w, w + dc)
    dr0, dr1 = max(0, -dr), min(h, h - dr)
    dc0, dc1 = max(0, -dc), min(w, w - dc)
    out[dr0:dr1, dc0:dc1] = a[sr0:sr1, sc0:sc1]
    return out


def compute_illumination_fraction(
    dem: np.ndarray,
    pixel_size_m: float,
    transform,
    r_moon_m: float,
    n_az: int = 48,
    n_delta: int = 9,
    max_elev_deg: float = 1.54,
    illum_max_dim: int = 1200,
    max_shadow_km: float = 30.0,
) -> np.ndarray:
    """Annual illumination fraction in [0, 1] per pixel (near-pole geometry).

    Physically correct south-polar solar model: the Sun's elevation above a
    pixel's local horizontal is

        el(pixel) = -delta + colat * cos(A - lon)              [small-angle, rad]

    where ``delta`` is the sub-solar latitude (sweeps +/-1.54 deg over the lunar
    year), ``A`` is the solar azimuth (sweeps 360 deg per lunar day), and
    ``colat``/``lon`` are the pixel's colatitude/longitude. This captures that
    max solar elevation grows with colatitude (~1.54 deg at the pole, larger
    away from it) -- a constant 1.54 deg grossly over-predicts PSRs off-pole.

    The DEM is downsampled to <= ``illum_max_dim`` for the horizon scan; the
    result is upsampled back to the input shape.

    Args:
        dem: 2-D elevation array (NaN allowed).
        pixel_size_m: Ground sample distance of ``dem`` (m).
        transform: Affine transform of ``dem`` (maps col/row -> x/y metres).
        r_moon_m: Lunar reference radius (m).
        n_az: Solar azimuth samples per lunar day.
        n_delta: Sub-solar-latitude (seasonal) samples over a lunar year.
        max_elev_deg: Peak |sub-solar latitude| (obliquity, deg).
        illum_max_dim: Max dimension for the shadow-scan grid.
        max_shadow_km: Max ray length to search for a horizon blocker.

    Returns:
        float32 illumination fraction array, same shape as ``dem``.
    """
    h, w = dem.shape
    scale = min(1.0, illum_max_dim / max(h, w))
    if scale < 1.0:
        dem_ds = zoom(np.nan_to_num(dem, nan=np.nanmin(dem)), scale, order=1)
        ps = pixel_size_m / scale
    else:
        dem_ds = np.nan_to_num(dem, nan=np.nanmin(dem)).astype(np.float32)
        ps = pixel_size_m
    rows, cols = dem_ds.shape

    # Per-pixel colatitude (rad) and longitude (rad) on the downsampled grid.
    # Pixel centres in projection metres (UL corner = transform.c/.f).
    xs = transform.c + (np.arange(cols) + 0.5) * ps
    ys = transform.f - (np.arange(rows) + 0.5) * ps
    X, Y = np.meshgrid(xs, ys)
    rho = np.hypot(X, Y)
    colat = 2.0 * np.arctan(rho / (2.0 * r_moon_m))   # exact stereographic
    lon = np.arctan2(X, Y)                             # x=rho sin lon, y=rho cos lon

    illum = np.zeros((rows, cols), dtype=np.float32)
    fill = float(dem_ds.min()) - 1e4
    max_steps = max(1, int(max_shadow_km * 1000.0 / ps))

    # Seasonal (delta) sampled uniformly in time: delta = 1.54 sin(phase).
    phases = np.linspace(0.0, 2.0 * np.pi, n_delta, endpoint=False)
    deltas = np.radians(max_elev_deg) * np.sin(phases)
    azimuths = np.linspace(0.0, 2.0 * np.pi, n_az, endpoint=False)
    total = n_delta * n_az

    print(f"      Illumination grid {rows}x{cols} ({ps:.0f} m/px), "
          f"{n_delta} seasons x {n_az} az = {total} sun positions, "
          f"<= {max_steps} ray steps")

    steps_cache = {}
    for delta in tqdm(deltas, desc="      sun-scan", ncols=70):
        for A in azimuths:
            el = -delta + colat * np.cos(A - lon)   # radians, per pixel
            up = el > 0
            if not up.any():
                continue
            tan_el = np.tan(np.clip(el, 1e-6, None))
            ddc = np.sin(A)
            ddr = -np.cos(A)
            shadowed = np.zeros((rows, cols), dtype=bool)
            for step in range(1, max_steps + 1):
                row_off = int(round(step * ddr))
                col_off = int(round(step * ddc))
                if row_off == 0 and col_off == 0:
                    continue
                dist = step * ps
                blocker = _shift2d(dem_ds, row_off, col_off, fill)
                shadowed |= blocker > (dem_ds + dist * tan_el)
            illum += (up & ~shadowed).astype(np.float32)

    illum /= float(total)

    if dem_ds.shape != dem.shape:
        illum_full = zoom(illum, (h / rows, w / cols), order=1)
        illum_full = np.clip(illum_full, 0.0, 1.0).astype(np.float32)
    else:
        illum_full = illum
    illum_full[np.isnan(dem)] = 0.0
    return illum_full


def _open_ray_count(base_dem, psr, pr, pc, work_px, max_range_km, r_ignore_m, n_rays):
    """Number of azimuth rays along which pixel (pr,pc) can see illuminated
    (non-PSR) terrain above its local horizon -- the double-shielding index
    (0 = fully shielded)."""
    h, w = base_dem.shape
    ms = max(1, int(max_range_km * 1000.0 / work_px))
    steps = np.arange(1, ms + 1)
    dist = steps * work_px
    b = base_dem[pr, pc]
    eps = np.tan(np.radians(0.05))
    oc = 0
    for a in np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False):
        rr = pr + np.round(steps * (-np.cos(a))).astype(np.int32)
        cc = pc + np.round(steps * (np.sin(a))).astype(np.int32)
        inb = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        if not inb.all():
            fo = int(np.argmin(inb))
            if fo == 0:
                continue
            rr, cc, d = rr[:fo], cc[:fo], dist[:fo]
        else:
            d = dist
        elev = base_dem[rr, cc]
        valid = elev > -1e8
        ang = np.where(valid, (elev - b) / d, -np.inf)
        prev = np.maximum.accumulate(np.concatenate([[-np.inf], ang[:-1]]))
        if ((~psr[rr, cc]) & valid & (ang > prev + eps) & (d > r_ignore_m)).any():
            oc += 1
    return oc


def detect_dsc(
    dem: np.ndarray,
    psr_mask: np.ndarray,
    pixel_size_m: float,
    close_m: float = 2500.0,
    core_depth_m: float = 40.0,
    min_depth_m: float = 40.0,
    min_diam_m: float = 500.0,
    max_diam_m: float = 3000.0,
    min_roundness: float = 0.45,
    major_psr_km2: float = 50.0,
    min_shield_index: float = 0.0,
    n_rays: int = 36,
    max_range_km: float = 5.0,
    ray_ignore_m: float = 500.0,
) -> Tuple[np.ndarray, list]:
    """Detect doubly-shadowed crater (DSC) candidates.

    A DSC candidate is a small (``min_diam_m``..``max_diam_m``), roundish,
    closed depression (black-top-hat depth >= ``min_depth_m``) that lies inside
    a *major* PSR (host connected PSR area >= ``major_psr_km2``). For each, a
    double-shielding index (number of azimuths from which the floor can still
    see directly-illuminated terrain; 0 = fully shielded) is recorded.

    Restricting to major PSRs isolates the Faustini/Shoemaker/Haworth-class
    hosts catalogued in the literature, rather than the hundreds of small
    craters scattered across every minor PSR.

    Returns:
        (dsc_label_uint16, dsc_list). Each dsc dict has: id, row, col, diam_m,
        depth_m, floor_elev_m, open_count, shield_index (0-1), host_psr_km2.
    """
    from scipy.ndimage import label, grey_closing

    h, w = dem.shape
    psr = psr_mask.astype(bool)
    valid = np.isfinite(dem)
    base_dem = np.where(valid, dem, -1e9).astype(np.float64)
    px_km2 = (pixel_size_m / 1000.0) ** 2

    # Host PSR component areas
    plab, _ = label(psr)
    psizes = np.bincount(plab.ravel()) * px_km2
    if psizes.size:
        psizes[0] = 0.0

    # Black top-hat depression depth
    filled = np.where(valid, dem, np.nanmax(dem)).astype(np.float32)
    closed = grey_closing(filled, size=max(3, int(close_m / pixel_size_m)))
    depr = closed - filled

    # Form depression cores at a shallow threshold so the full crater extent is
    # captured for diameter; the deeper ``min_depth_m`` filter is applied per
    # crater (on its peak depth) below.
    core = (depr >= core_depth_m) & psr & valid
    clab, n = label(core)

    dsc_label = np.zeros((h, w), dtype=np.uint16)
    dscs: list = []
    new_id = 0
    for lid in range(1, n + 1):
        m = clab == lid
        npx = int(m.sum())
        diam = 2.0 * np.sqrt(npx / np.pi) * pixel_size_m
        if not (min_diam_m <= diam <= max_diam_m):
            continue
        rr, cc = np.where(m)
        cr, cc2 = rr.mean(), cc.mean()
        d = np.hypot(rr - cr, cc - cc2)
        req = np.sqrt(npx / np.pi)
        roundness = 1.0 - d.std() / (req + 1e-6)
        if roundness < min_roundness:
            continue
        depth = float(depr[m].max())
        if depth < min_depth_m:
            continue
        sub = base_dem[rr, cc]
        fi = int(sub.argmin())
        fr, fc = int(rr[fi]), int(cc[fi])
        host_km2 = float(psizes[plab[fr, fc]]) if plab[fr, fc] else 0.0
        if host_km2 < major_psr_km2:
            continue
        # Expensive ray-cast only for survivors of the cheap filters.
        oc = _open_ray_count(base_dem, psr, fr, fc, pixel_size_m,
                             max_range_km, ray_ignore_m, n_rays)
        shield = (n_rays - oc) / n_rays
        if shield < min_shield_index:
            continue
        new_id += 1
        dsc_label[m] = new_id
        dscs.append({
            "id": new_id, "row": cr, "col": cc2, "floor_row": fr, "floor_col": fc,
            "diam_m": float(diam), "depth_m": depth,
            "floor_elev_m": float(dem[fr, fc]), "open_count": int(oc),
            "shield_index": float(shield), "host_psr_km2": host_km2,
        })
    return dsc_label, dscs


def test_double_shielding(
    dem: np.ndarray,
    psr_mask: np.ndarray,
    pixel_size_m: float,
    n_rays: int = 36,
    max_range_km: float = 5.0,
    local_min_window_m: float = 500.0,
    min_relief_m: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Doubly-shielded test for PSR floor pixels.

    A candidate PSR pixel is *doubly shielded* if, along every one of
    ``n_rays`` outward azimuths, it cannot see any directly illuminated
    (non-PSR) terrain -- every ray either stays inside the PSR for its full
    length or the nearest illuminated terrain is hidden behind closer terrain
    (line-of-sight blocked).

    Only PSR pixels that are local minima (crater-floor candidates) are tested,
    for tractability.

    Args:
        dem: 2-D elevation array (NaN allowed).
        psr_mask: Boolean/0-1 array, True inside PSR.
        pixel_size_m: Ground sample distance (m).
        n_rays: Number of azimuth rays per pixel.
        max_range_km: Max ray length (scattered-light reach).
        local_min_window_m: Window for selecting floor-candidate minima.

    Returns:
        (dsc_mask, candidate_mask) boolean arrays, same shape as ``dem``.
    """
    h, w = dem.shape
    psr = psr_mask.astype(bool)
    base_dem = np.where(np.isnan(dem), -1e9, dem).astype(np.float64)

    from scipy.ndimage import uniform_filter
    win_px = max(3, int(local_min_window_m / pixel_size_m))
    # Relief = how far a pixel sits below its wider neighbourhood mean; keeps
    # genuine depression floors and drops flat-PSR speckle.
    big = max(win_px * 3, 9)
    filled = np.where(np.isnan(dem), np.nanmax(dem), dem).astype(np.float32)
    relief = uniform_filter(filled, size=big) - filled
    candidates = local_minima_mask(dem, win_px) & psr & (relief >= min_relief_m)
    cand_rc = np.argwhere(candidates)

    max_steps = max(1, int(max_range_km * 1000.0 / pixel_size_m))
    steps = np.arange(1, max_steps + 1)
    dist = steps * pixel_size_m

    # Precompute integer offsets per ray.
    azimuths = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
    ray_dr = [np.round(steps * (-np.cos(a))).astype(np.int32) for a in azimuths]
    ray_dc = [np.round(steps * (np.sin(a))).astype(np.int32) for a in azimuths]

    dsc = np.zeros((h, w), dtype=bool)
    eps = np.tan(np.radians(0.05))

    print(f"      Double-shielding: {len(cand_rc)} PSR floor candidates, "
          f"{n_rays} rays x {max_steps} steps")

    for (pr, pc) in tqdm(cand_rc, desc="      dsc-rays", ncols=70):
        base = base_dem[pr, pc]
        if base < -1e8:
            continue
        sees_illuminated = False
        for a in range(n_rays):
            rr = pr + ray_dr[a]
            cc = pc + ray_dc[a]
            inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            if not inside.any():
                continue
            # Truncate ray at first out-of-bounds step.
            if not inside.all():
                first_out = int(np.argmin(inside))
                if first_out == 0:
                    continue
                rr = rr[:first_out]
                cc = cc[:first_out]
                d = dist[:first_out]
            else:
                d = dist
            elev = base_dem[rr, cc]
            valid = elev > -1e8
            ang = np.where(valid, np.arctan((elev - base) / d), -np.inf)
            # Max angle of terrain strictly closer than each step.
            prev_max = np.maximum.accumulate(
                np.concatenate([[-np.inf], ang[:-1]])
            )
            illuminated = ~psr[rr, cc] & valid
            visible = illuminated & (ang > prev_max + eps)
            if visible.any():
                sees_illuminated = True
                break
        if not sees_illuminated:
            dsc[pr, pc] = True

    return dsc, candidates
