"""Deliverable 5 figures: IEM forward curve, ice-volume panel, context comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

sys.path.insert(0, str(Path(__file__).parent))
import d5_config as cfg
import d2_config as d2


def _extent_km(shape, tfm, f2x, f2y):
    h, w = shape
    return [(tfm.c - f2x) / 1000.0, (tfm.c + w * tfm.a - f2x) / 1000.0,
            (tfm.f + h * tfm.e - f2y) / 1000.0, (tfm.f - f2y) / 1000.0]


def _on_grid(src_path, tfm, shape, crs, resamp=Resampling.bilinear):
    with rasterio.open(str(src_path)) as s:
        dst = np.full(shape, np.nan, np.float32)
        reproject(rasterio.band(s, 1), dst, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=tfm, dst_crs=crs, resampling=resamp)
    return dst


def _hillshade(z, px=20.0, az=315.0, alt=35.0):
    z = np.nan_to_num(z, nan=float(np.nanmin(z)))
    gy, gx = np.gradient(z, px)
    slope = np.arctan(np.hypot(gx, gy)); aspect = np.arctan2(-gx, gy)
    a = np.radians(360 - az + 90); al = np.radians(alt)
    return np.clip(np.sin(al) * np.cos(slope) + np.cos(al) * np.sin(slope) * np.cos(a - aspect), 0, 1)


def render_all(cpr, dop, f_ice, conf, tfm, crs, fwd, theta, results, s0_mean):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    f, eps, s0_db, dyn = fwd
    f2x, f2y = d2.latlon_to_stereographic(cfg.F2_DIAMETER_M * 0 + d2.F2_LAT_DEG, d2.F2_LON_DEG)
    ext = _extent_km(cpr.shape, tfm, f2x, f2y)
    hs = _hillshade(_on_grid(d2.DEM_TIF, tfm, cpr.shape, crs))
    dsc = _on_grid(cfg.DSC_MASK_TIF, tfm, cpr.shape, crs, Resampling.nearest) > 0.5
    depths = cfg.DEPTH_SCENARIOS_M
    vols = np.array([results[d]["vol_m3"] for d in depths])
    vh = np.array([results[d]["vol_high"] for d in depths])
    vm = np.array([results[d]["vol_med"] for d in depths])

    # ── d5_iem_forward_curve.png ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6)); fig.patch.set_facecolor("#0a0a0a"); ax.set_facecolor("#0a0a0a")
    ax.plot(f * 100, s0_db, color="#39ff14", lw=2.2)
    ax.set_ylim(s0_db.mean() - 6, s0_db.mean() + 6)   # show flatness honestly (~10 dB obs swing)
    ax.set_xlabel("ice fraction (%)", color="white"); ax.set_ylabel("IEM σ0 (dB)", color="white")
    ax.set_title(f"IEM forward model — σ0 vs ice fraction (θ={theta:.0f}°, L-band)\n"
                 f"σ0 changes only {dyn:.2f} dB over 0–50% ice  ⇒  σ0 cannot constrain ice fraction",
                 color="white", fontsize=11)
    ax.tick_params(colors="white"); [s.set_edgecolor("#444") for s in ax.spines.values()]; ax.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(cfg.FIG_FORWARD, dpi=300, facecolor=fig.get_facecolor()); plt.close(fig)
    print(f"      ✓ {cfg.FIG_FORWARD.name}")

    # ── d5_ice_volume.png (4 panels) ─────────────────────────────────────────
    fig, ax = plt.subplots(2, 2, figsize=(15, 12)); fig.patch.set_facecolor("#0a0a0a")
    # A: forward curve
    a = ax[0, 0]; a.set_facecolor("#0a0a0a"); a.plot(f * 100, s0_db, color="#39ff14", lw=2)
    a.set_ylim(s0_db.mean() - 6, s0_db.mean() + 6)   # honest scale: flat vs ~10 dB obs swing
    a.set_title("A. IEM σ0 vs ice fraction (θ=%.0f°)\n(flat ⇒ σ0-inversion degenerate; CPR used instead)" % theta,
                color="white", fontsize=10)
    a.set_xlabel("ice fraction (%)", color="white"); a.set_ylabel("σ0 (dB)", color="white")
    a.tick_params(colors="white"); [s.set_edgecolor("#444") for s in a.spines.values()]; a.grid(alpha=0.2)
    # B: ice fraction map
    a = ax[0, 1]; a.set_facecolor("#0a0a0a"); a.imshow(hs, cmap="gray", extent=ext, origin="upper")
    im = a.imshow(np.ma.masked_invalid(f_ice) * 100, cmap="YlGnBu_r", vmin=0, vmax=30,
                  extent=ext, origin="upper", alpha=0.9)
    a.contour(dsc.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.2, extent=ext, origin="upper")
    a.contour((conf == 3).astype(float), levels=[0.5], colors=["red"], linewidths=0.8, extent=ext, origin="upper")
    plt.colorbar(im, ax=a, fraction=0.046, label="ice fraction (%)")
    a.set_xlim(-3, 3); a.set_ylim(-3, 3)
    a.set_title("B. CPR-derived ice fraction (cyan=F2 floor, red=HIGH conf)", color="white", fontsize=10)
    a.set_xlabel("E offset from F2 (km)", color="white"); a.set_ylabel("N offset (km)", color="white")
    a.tick_params(colors="white"); [s.set_edgecolor("#444") for s in a.spines.values()]
    # C: volume sensitivity (stacked bars HIGH/MED) + ±50%
    a = ax[1, 0]; a.set_facecolor("#0a0a0a"); x = np.arange(len(depths))
    a.bar(x, vh, color="#ff3b30", label="HIGH conf")
    a.bar(x, vm, bottom=vh, color="#f2c744", label="MED conf")
    a.errorbar(x, vols, yerr=vols * cfg.DIELECTRIC_UNCERTAINTY, fmt="none", ecolor="white", capsize=4)
    a.set_xticks(x); a.set_xticklabels([f"{d:.0f} m" for d in depths])
    a.set_title("C. Ice volume vs assumed depth (±50%)", color="white", fontsize=10)
    a.set_ylabel("ice volume (m³)", color="white"); a.set_xlabel("layer depth", color="white")
    a.legend(facecolor="#222", labelcolor="white"); a.tick_params(colors="white")
    [s.set_edgecolor("#444") for s in a.spines.values()]
    # D: cumulative mass vs depth
    a = ax[1, 1]; a.set_facecolor("#0a0a0a")
    dd = np.linspace(0, 10, 50)
    per_m = (vols[0] / depths[0])  # m^3 per metre depth (linear)
    mass = dd * per_m * cfg.RHO_ICE
    a.plot(dd, mass / 1e6, color="#1f6feb", lw=2)
    a.fill_between(dd, mass / 1e6 * 0.5, mass / 1e6 * 1.5, color="#1f6feb", alpha=0.25)
    cm = cfg.CENTRAL_DEPTH_M * per_m * cfg.RHO_ICE / 1e6
    a.axvline(cfg.CENTRAL_DEPTH_M, color="white", ls="--")
    a.annotate(f"5 m central\n{cm:.1f} Mkg", (cfg.CENTRAL_DEPTH_M, cm), color="white", fontsize=9,
               xytext=(6, cm * 0.7))
    a.set_title("D. Cumulative ice mass vs depth (±50% band)", color="white", fontsize=10)
    a.set_xlabel("layer depth (m)", color="white"); a.set_ylabel("ice mass (Mkg)", color="white")
    a.tick_params(colors="white"); [s.set_edgecolor("#444") for s in a.spines.values()]; a.grid(alpha=0.2)
    fig.suptitle("Deliverable 5: F2 ice volume — CPR-derived abundance + IEM/Maxwell-Garnett framework",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig(cfg.FIG_VOLUME, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig); print(f"      ✓ {cfg.FIG_VOLUME.name}")

    # ── d5_comparison.png ────────────────────────────────────────────────────
    c5 = results[cfg.CENTRAL_DEPTH_M]
    v_all = c5["vol_m3"]; v_high = c5["vol_high"]
    labels = ["F2 HIGH-conf\n(this work, 5 m)", "F2 all-candidate\n(this work, 5 m)",
              "Cabeus LCROSS\n(~excavated ref.)", "Typical PSR\ncold-trap (lit.)"]
    # reference order-of-magnitude context values (m^3)
    vals = [v_high, v_all, 1.0e3, 1.0e5]
    colors = ["#ff3b30", "#f2c744", "#888", "#1f6feb"]
    fig, ax = plt.subplots(figsize=(10, 6)); fig.patch.set_facecolor("#0a0a0a"); ax.set_facecolor("#0a0a0a")
    ax.bar(labels, vals, color=colors)
    ax.set_yscale("log"); ax.set_ylabel("ice volume (m³, log)", color="white")
    ax.set_title("Deliverable 5: F2 ice volume in context (order-of-magnitude)", color="white", fontsize=12)
    ax.tick_params(colors="white"); [s.set_edgecolor("#444") for s in ax.spines.values()]
    for i, v in enumerate(vals):
        ax.text(i, v * 1.2, f"{v:.1e}", ha="center", color="white", fontsize=8)
    plt.tight_layout(); plt.savefig(cfg.FIG_COMPARE, dpi=300, facecolor=fig.get_facecolor()); plt.close(fig)
    print(f"      ✓ {cfg.FIG_COMPARE.name}")
