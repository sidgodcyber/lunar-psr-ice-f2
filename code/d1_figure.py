"""Deliverable 1 publication figure (4-panel, dark theme, 300 DPI)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol

sys.path.insert(0, str(Path(__file__).parent))
import d1_config as cfg


def _hillshade(dem: np.ndarray, px: float, az_deg: float = 315.0,
               alt_deg: float = 30.0) -> np.ndarray:
    """Standard hillshade in [0, 1]."""
    z = np.nan_to_num(dem, nan=np.nanmin(dem))
    gy, gx = np.gradient(z, px)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = np.radians(360.0 - az_deg + 90.0)
    alt = np.radians(alt_deg)
    hs = (np.sin(alt) * np.cos(slope)
          + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(hs, 0, 1)


def _extent_km(shape, wt, f2x, f2y):
    h, w = shape
    left = (wt.c - f2x) / 1000.0
    right = (wt.c + w * wt.a - f2x) / 1000.0
    top = (wt.f - f2y) / 1000.0
    bottom = (wt.f + h * wt.e - f2y) / 1000.0
    return [left, right, bottom, top]


def _rc_to_km(row, col, wt, f2x, f2y):
    x_km = (wt.c + col * wt.a - f2x) / 1000.0
    y_km = (wt.f + row * wt.e - f2y) / 1000.0
    return x_km, y_km


def _add_scalebar(ax, length_km, label=None, color="white"):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    sx = x0 + 0.06 * (x1 - x0)
    sy = y0 + 0.08 * (y1 - y0)
    ax.plot([sx, sx + length_km], [sy, sy], color=color, lw=3, solid_capstyle="butt")
    ax.text(sx + length_km / 2, sy + 0.02 * (y1 - y0),
            label or f"{length_km:g} km", color=color, ha="center",
            va="bottom", fontsize=8, fontweight="bold")


def _add_north(ax, f2x, f2y, color="white"):
    """Arrow pointing toward the south pole (stereographic origin)."""
    x1, x0 = ax.get_xlim()[1], ax.get_xlim()[0]
    y0, y1 = ax.get_ylim()
    px = x1 - 0.10 * (x1 - x0)
    py = y1 - 0.16 * (y1 - y0)
    # direction toward pole from F2, in km-offset coords
    dirx, diry = (0 - f2x), (0 - f2y)
    n = np.hypot(dirx, diry)
    dirx, diry = dirx / n, diry / n
    L = 0.10 * (y1 - y0)
    ax.annotate("", xy=(px + dirx * L, py + diry * L), xytext=(px, py),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))
    ax.text(px + dirx * L, py + diry * L, " pole", color=color, fontsize=7)


def _style(ax):
    ax.set_facecolor("#0a0a0a")
    ax.tick_params(colors="white", labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor("#444")
    ax.set_xlabel("E offset from F2 (km)", color="white", fontsize=8)
    ax.set_ylabel("N offset from F2 (km)", color="white", fontsize=8)


def render_figure(work, wt, work_px, illum, psr, dsc_keep, dscs, f2_rc, f2win, m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    f2x, f2y = cfg.latlon_to_stereographic(cfg.F2_LAT_DEG, cfg.F2_LON_DEG)
    ext = _extent_km(work.shape, wt, f2x, f2y)
    dem10, win_t, f2_lr, f2_lc, px10 = f2win

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.patch.set_facecolor("#0a0a0a")

    dsc_xy = [_rc_to_km(d["row"], d["col"], wt, f2x, f2y) for d in dscs]

    # ── Panel A: hillshade + PSR + DSC + F2 ──────────────────────────────────
    ax = axes[0, 0]
    hs = _hillshade(work, work_px)
    ax.imshow(hs, cmap="gray", extent=ext, origin="upper", aspect="equal")
    ax.imshow(np.where(np.isfinite(work), work, np.nan), cmap="terrain", alpha=0.45,
              extent=ext, origin="upper", aspect="equal")
    ax.contour(psr.astype(float), levels=[0.5], colors=["red"], linewidths=1.0,
               extent=ext, origin="upper")
    for (xk, yk) in dsc_xy:
        ax.plot(xk, yk, "+", color="cyan", ms=9, mew=1.6)
    ax.add_patch(Circle((0, 0), 3.0, fill=False, ec="yellow", lw=2.0))
    ax.text(0, 3.4, "F2", color="yellow", ha="center", fontsize=10, fontweight="bold")
    ax.set_title("A. LOLA DEM hillshade — PSR (red), DSC (cyan +), F2 (yellow)",
                 color="white", fontsize=11)
    _style(ax); _add_scalebar(ax, 50); _add_north(ax, f2x, f2y)

    # ── Panel B: illumination fraction ───────────────────────────────────────
    ax = axes[0, 1]
    im = ax.imshow(illum, cmap="inferno", vmin=0, vmax=illum.max() if illum.max() > 0 else 1,
                   extent=ext, origin="upper", aspect="equal")
    ax.contour(psr.astype(float), levels=[0.5], colors=["red"], linewidths=1.0,
               extent=ext, origin="upper")
    for (xk, yk) in dsc_xy:
        ax.plot(xk, yk, "+", color="cyan", ms=9, mew=1.6)
    ax.add_patch(Circle((0, 0), 3.0, fill=False, ec="yellow", lw=2.0))
    ax.set_title("B. Annual illumination fraction (0 = PSR)", color="white", fontsize=11)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(colors="white", labelsize=7)
    _style(ax); _add_scalebar(ax, 50); _add_north(ax, f2x, f2y)

    # ── Panel C: slope within 20 km of F2 ────────────────────────────────────
    ax = axes[1, 0]
    half = int(10000 / px10)
    with rasterio.open(cfg.SLOPE_TIF) as src:
        r0, c0 = rowcol(src.transform, f2x, f2y)
        r0, c0 = int(r0), int(c0)
        rr0, rr1 = max(0, r0 - half), min(src.height, r0 + half)
        cc0, cc1 = max(0, c0 - half), min(src.width, c0 + half)
        slope_z = src.read(1, window=((rr0, rr1), (cc0, cc1)))
        st = src.window_transform(((rr0, rr1), (cc0, cc1)))
    sext = _extent_km(slope_z.shape, st, f2x, f2y)
    im = ax.imshow(np.clip(slope_z, 0, 40), cmap="hot", extent=sext, origin="upper",
                   aspect="equal")
    safe = np.where(slope_z < 15, 1.0, np.nan)
    ax.imshow(safe, cmap="summer", alpha=0.30, extent=sext, origin="upper", aspect="equal")
    ax.add_patch(Circle((0, 0), cfg.F2_DIAMETER_M / 2000.0, fill=False, ec="cyan", lw=2.0))
    ax.set_title("C. Slope within 20 km of F2 (green = <15 deg, landable)",
                 color="white", fontsize=11)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="slope (deg)")
    cb.ax.tick_params(colors="white", labelsize=7); cb.set_label("slope (deg)", color="white")
    _style(ax); _add_scalebar(ax, 5)

    # ── Panel D: F2 close-up (native 10 m) ───────────────────────────────────
    ax = axes[1, 1]
    dext = _extent_km(dem10.shape, win_t, f2x, f2y)
    hs2 = _hillshade(dem10, px10)
    ax.imshow(hs2, cmap="gray", extent=dext, origin="upper", aspect="equal")
    ax.imshow(np.where(np.isfinite(dem10), dem10, np.nan), cmap="terrain", alpha=0.5,
              extent=dext, origin="upper", aspect="equal")
    # F2 floor marker from morphology
    fr_km, fc_km = _rc_to_km(m["floor_row"], m["floor_col"], win_t, f2x, f2y)
    ax.plot(fc_km, fr_km, "+", color="cyan", ms=14, mew=2)
    ax.add_patch(Circle((fc_km, fr_km), m["diameter_m"] / 2000.0 if np.isfinite(m["diameter_m"]) else 0.55,
                        fill=False, ec="yellow", lw=2))
    label = (f"F2\nD={m['diameter_m']:.0f} m\ndepth={m['depth_m']:.0f} m\n"
             f"floor={m['floor_elev_m']:.0f} m\n{cfg.F2_LAT_DEG}, {cfg.F2_LON_DEG} E")
    ax.text(fc_km + 1.0, fr_km + 0.8, label, color="yellow", fontsize=8,
            fontweight="bold", va="bottom")
    ax.set_xlim(dext[0], dext[1]); ax.set_ylim(dext[2], dext[3])
    ax.set_title("D. F2 close-up (native 10 m DEM)", color="white", fontsize=11)
    _style(ax); _add_scalebar(ax, 2)

    fig.suptitle(
        "Deliverable 1: PSR + Doubly-Shadowed-Crater Mapping\n"
        f"LOLA LDEM 85 S 10 m | Target F2 inside Faustini PSR | "
        f"{len(dscs)} DSC candidates",
        color="white", fontsize=13, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(cfg.FIGURE_PNG, dpi=300, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    print(f"      ✓ Figure saved: {cfg.FIGURE_PNG}")
