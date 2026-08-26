"""Deliverable 2 figures: CPR/DOP maps, histograms, paper comparison (dark, 300 DPI)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

sys.path.insert(0, str(Path(__file__).parent))
import d2_config as cfg


def _extent_km(shape, tfm, f2x, f2y):
    h, w = shape
    left = (tfm.c - f2x) / 1000.0
    right = (tfm.c + w * tfm.a - f2x) / 1000.0
    top = (tfm.f - f2y) / 1000.0
    bottom = (tfm.f + h * tfm.e - f2y) / 1000.0
    return [left, right, bottom, top]


def _hillshade(z, px=20.0, az=315.0, alt=35.0):
    z = np.nan_to_num(z, nan=float(np.nanmin(z)))
    gy, gx = np.gradient(z, px)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a = np.radians(360.0 - az + 90.0)
    al = np.radians(alt)
    hs = np.sin(al) * np.cos(slope) + np.cos(al) * np.sin(slope) * np.cos(a - aspect)
    return np.clip(hs, 0, 1)


def _dem_on_grid(tfm, shape, crs):
    with rasterio.open(cfg.DEM_TIF) as s:
        dst = np.full(shape, np.nan, np.float32)
        reproject(rasterio.band(s, 1), dst, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=tfm, dst_crs=crs, resampling=Resampling.bilinear)
    return dst


def _style(ax, ext, title):
    ax.set_facecolor("#0a0a0a")
    ax.set_title(title, color="white", fontsize=10)
    ax.set_xlabel("E offset from F2 (km)", color="white", fontsize=8)
    ax.set_ylabel("N offset from F2 (km)", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for s in ax.spines.values():
        s.set_edgecolor("#444")


def _outline(ax, mask, ext, color, lw=1.2):
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=lw,
               extent=ext, origin="upper")


def render_all(cpr, dop, sig, dual, conf, aoi, psr, dsc, tfm, crs, f2xy, stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    f2x, f2y = f2xy
    ext = _extent_km(cpr.shape, tfm, f2x, f2y)
    floor = aoi & dsc
    cprm = np.ma.masked_invalid(np.where(cpr >= 0, cpr, np.nan))
    dopm = np.ma.masked_invalid(np.where(dop >= 0, dop, np.nan))
    hs = _hillshade(_dem_on_grid(tfm, cpr.shape, crs))

    # ── Figure 1: 6-panel maps ───────────────────────────────────────────────
    fig, ax = plt.subplots(2, 3, figsize=(19, 12)); fig.patch.set_facecolor("#0a0a0a")
    im = ax[0, 0].imshow(cprm, cmap="inferno", vmin=0, vmax=2, extent=ext, origin="upper")
    _outline(ax[0, 0], floor, ext, "cyan"); _outline(ax[0, 0], aoi, ext, "yellow", 0.7)
    plt.colorbar(im, ax=ax[0, 0], fraction=0.046, label="CPR"); _style(ax[0, 0], ext, "A. CPR (cyan=F2 floor, yellow=AOI)")
    im = ax[0, 1].imshow(dopm, cmap="viridis", vmin=0, vmax=0.6, extent=ext, origin="upper")
    _outline(ax[0, 1], floor, ext, "cyan"); plt.colorbar(im, ax=ax[0, 1], fraction=0.046, label="DOP")
    _style(ax[0, 1], ext, "B. DOP (low = volume scatter / ice)")
    im = ax[0, 2].imshow(np.ma.masked_invalid(sig), cmap="gray", extent=ext, origin="upper")
    _outline(ax[0, 2], floor, ext, "cyan"); plt.colorbar(im, ax=ax[0, 2], fraction=0.046, label="rel. σ0 (dB)")
    _style(ax[0, 2], ext, "C. Relative σ0 (uncalibrated)")
    ax[1, 0].imshow(hs, cmap="gray", extent=ext, origin="upper")
    dm = np.ma.masked_where(~dual, np.ones_like(cpr))
    ax[1, 0].imshow(dm, cmap=ListedColormap(["#00e5ff"]), extent=ext, origin="upper", alpha=0.8)
    _outline(ax[1, 0], floor, ext, "yellow"); _style(ax[1, 0], ext, "D. Dual criterion (CPR>1 & DOP<0.20)")
    cmap = ListedColormap(["#202020", "#1f6feb", "#f2c744", "#ff3b30"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    ax[1, 1].imshow(hs, cmap="gray", extent=ext, origin="upper")
    im = ax[1, 1].imshow(np.ma.masked_where(conf == 0, conf), cmap=cmap, norm=norm,
                         extent=ext, origin="upper", alpha=0.85)
    cb = plt.colorbar(im, ax=ax[1, 1], fraction=0.046, ticks=[1, 2, 3]); cb.set_label("confidence")
    _outline(ax[1, 1], floor, ext, "cyan"); _style(ax[1, 1], ext, "E. Ice confidence (1 low–3 high) over hillshade")
    ice = (cpr > 1) & (dop < 0.20)
    ax[1, 2].imshow(cprm, cmap="inferno", vmin=0, vmax=2, extent=ext, origin="upper")
    ax[1, 2].imshow(np.ma.masked_where(~ice, np.ones_like(cpr)),
                    cmap=ListedColormap(["#39ff14"]), extent=ext, origin="upper", alpha=0.6)
    _outline(ax[1, 2], floor, ext, "cyan"); _style(ax[1, 2], ext, "F. Ice candidates (green) over CPR")
    for a in ax.ravel():
        a.set_xlim(-4, 4); a.set_ylim(-4, 4)
    fig.suptitle("Deliverable 2: DFSAR CPR/DOP ice detection at F2 (pass 4, 20191105)",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(cfg.FIG_MAPS, dpi=300, facecolor=fig.get_facecolor()); plt.close(fig)
    print(f"      ✓ {cfg.FIG_MAPS.name}")

    # ── Figure 2: histograms + scatter ───────────────────────────────────────
    inside = floor & np.isfinite(cpr) & np.isfinite(dop)
    outside = (~aoi) & np.isfinite(cpr) & np.isfinite(dop) & (cpr >= 0)
    ci, di = cpr[inside], dop[inside]; co, do = cpr[outside], dop[outside]
    fig, ax = plt.subplots(2, 2, figsize=(14, 11)); fig.patch.set_facecolor("#0a0a0a")
    for a in ax.ravel():
        a.set_facecolor("#0a0a0a"); a.tick_params(colors="white"); [s.set_edgecolor("#444") for s in a.spines.values()]
    ax[0, 0].hist(co, bins=60, range=(0, 2.5), color="#888", alpha=0.6, density=True, label="outside F2")
    ax[0, 0].hist(ci, bins=60, range=(0, 2.5), color="#ff3b30", alpha=0.7, density=True, label="F2 floor")
    ax[0, 0].axvline(1.0, color="cyan", ls="--"); ax[0, 0].legend(facecolor="#222", labelcolor="white")
    ax[0, 0].set_title("A. CPR: F2 floor vs outside", color="white"); ax[0, 0].set_xlabel("CPR", color="white")
    ax[0, 1].hist(do, bins=60, range=(0, 1), color="#888", alpha=0.6, density=True, label="outside F2")
    ax[0, 1].hist(di, bins=60, range=(0, 1), color="#1f6feb", alpha=0.7, density=True, label="F2 floor")
    ax[0, 1].axvline(0.13, color="yellow", ls="--", label="0.13 (paper)"); ax[0, 1].axvline(0.20, color="orange", ls=":")
    ax[0, 1].legend(facecolor="#222", labelcolor="white"); ax[0, 1].set_title("B. DOP: F2 floor vs outside", color="white")
    ax[0, 1].set_xlabel("DOP", color="white")
    sc = ax[1, 0].scatter(ci, di, s=4, c=ci, cmap="inferno", vmin=0, vmax=2, alpha=0.5)
    ax[1, 0].axvline(1.0, color="cyan", ls="--"); ax[1, 0].axhline(0.13, color="yellow", ls="--")
    ax[1, 0].axhline(0.20, color="orange", ls=":")
    ax[1, 0].text(1.5, 0.05, "ICE\n(CPR>1, DOP low)", color="#39ff14", fontsize=9, ha="center")
    ax[1, 0].set_xlim(0, 2.5); ax[1, 0].set_ylim(0, 1); ax[1, 0].set_title("C. F2 floor: CPR vs DOP", color="white")
    ax[1, 0].set_xlabel("CPR", color="white"); ax[1, 0].set_ylabel("DOP", color="white")
    # D: cumulative CPR
    ax[1, 1].hist(ci, bins=80, range=(0, 2.5), color="#ff3b30", cumulative=True, density=True, histtype="step")
    ax[1, 1].axvline(1.0, color="cyan", ls="--")
    ax[1, 1].set_title(f"D. F2 floor CPR CDF ({stats['pct_cpr_gt1']:.0f}% > 1, paper ~47%)", color="white")
    ax[1, 1].set_xlabel("CPR", color="white"); ax[1, 1].set_ylabel("cumulative fraction", color="white")
    fig.suptitle("Deliverable 2 — F2 polarimetric distributions", color="white", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig(cfg.FIG_HIST, dpi=300, facecolor=fig.get_facecolor()); plt.close(fig)
    print(f"      ✓ {cfg.FIG_HIST.name}")

    # ── Figure 3: paper comparison ───────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(9, 8)); fig.patch.set_facecolor("#0a0a0a")
    im = ax.imshow(cprm, cmap="jet", vmin=0, vmax=2, extent=ext, origin="upper")
    _outline(ax, floor, ext, "white", 1.5)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    cb = plt.colorbar(im, ax=ax, fraction=0.046); cb.set_label("CPR", color="white"); cb.ax.tick_params(colors="white")
    txt = (f"Measured (this work) vs Sinha & Bharti 2026\n"
           f"max CPR: {stats['max_cpr']:.2f}  (paper 1.95)\n"
           f"% CPR>1: {stats['pct_cpr_gt1']:.0f}%  (paper ~47%)\n"
           f"mean DOP(CPR>1): {stats['dop_hi']:.2f}  (paper <0.13)")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", color="white", fontsize=9,
            bbox=dict(facecolor="#111", edgecolor="#555"))
    _style(ax, ext, "F2 CPR map — paper-style (jet, 0–2)")
    plt.tight_layout(); plt.savefig(cfg.FIG_PAPER, dpi=300, facecolor=fig.get_facecolor()); plt.close(fig)
    print(f"      ✓ {cfg.FIG_PAPER.name}")
