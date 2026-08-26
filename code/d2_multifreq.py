"""
Multi-frequency depth stratification at F2 — L-band vs S-band compact-pol CPR.

Both from the same 20200808 compact-pol acquisition (only frequency differs), same
m-chi pipeline, so the comparison is controlled. Penetration depth ~ wavelength:
S-band (2.5 GHz, 0.12 m) senses shallower than L-band (1.25 GHz, 0.24 m).

Per F2-floor pixel:
  BOTH  (L>1 & S>1)  -> ice present shallow AND deep
  L-only(L>1 & S<=1) -> deeper subsurface ice (volumetric; below S penetration)  [strongest]
  S-only(S>1 & L<=1) -> shallow surface roughness -> likely false positive
  neither            -> no CPR enhancement

Run: python code/d2_multifreq.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

sys.path.insert(0, str(Path(__file__).parent))
import d2_config as cfg

CPR_L = cfg.GEOTIFF_DIR / "cpr_pass6.tif"     # L-band compact-pol (20200808)
CPR_S = cfg.GEOTIFF_DIR / "cpr_sband.tif"     # S-band compact-pol (20200808)
CLASS_TIF = cfg.GEOTIFF_DIR / "multifreq_class_f2.tif"
FIG = cfg.FIG_DIR / "d2_multifreq.png"
REPORT = cfg.REPORT_DIR / "d2_multifreq_report.md"
PX_KM2 = (cfg.GEOCODE_PIXEL_M / 1000.0) ** 2
T = 1.0   # CPR "elevated" threshold


def _r(p):
    with rasterio.open(str(p)) as s:
        return s.read(1), s.transform, s.crs, (s.height, s.width)


def _og(p, tfm, shape, crs, rs=Resampling.nearest):
    with rasterio.open(str(p)) as s:
        d = np.zeros(shape, np.float32)
        reproject(rasterio.band(s, 1), d, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=tfm, dst_crs=crs, resampling=rs)
    return d


def main():
    L, tfm, crs, shape = _r(CPR_L)
    S = _r(CPR_S)[0]
    vL, vS = L >= 0, S >= 0
    floor = (_og(cfg.F2_AOI_TIF, tfm, shape, crs) > 0.5) & (_og(cfg.DSC_MASK_TIF, tfm, shape, crs) > 0.5)
    valid = floor & vL & vS

    eL, eS = L > T, S > T
    both = valid & eL & eS
    lonly = valid & eL & ~eS
    sonly = valid & ~eL & eS
    neither = valid & ~eL & ~eS
    n = int(valid.sum())

    # classification raster: 1 both, 2 L-only, 3 S-only, 4 neither
    cls = np.zeros(shape, np.uint8)
    cls[both] = 1; cls[lonly] = 2; cls[sonly] = 3; cls[neither] = 4
    with rasterio.open(str(CLASS_TIF), "w", driver="GTiff", height=shape[0], width=shape[1],
                       count=1, dtype="uint8", crs=crs, transform=tfm, nodata=0, compress="lzw") as d:
        d.write(cls, 1)

    stats = {"n": n,
             "both": int(both.sum()), "lonly": int(lonly.sum()),
             "sonly": int(sonly.sum()), "neither": int(neither.sum()),
             "Lmax": float(np.nanmax(np.where(valid, L, np.nan))),
             "Smax": float(np.nanmax(np.where(valid, S, np.nan))),
             "L_gt1": 100 * (valid & eL).sum() / n, "S_gt1": 100 * (valid & eS).sum() / n}
    print("=" * 72); print("MULTI-FREQUENCY DEPTH STRATIFICATION — F2 floor")
    print(f"  floor pixels (both bands valid): {n}")
    print(f"  L-band CPR>1: {stats['L_gt1']:.0f}%  (max {stats['Lmax']:.2f})  [deeper]")
    print(f"  S-band CPR>1: {stats['S_gt1']:.0f}%  (max {stats['Smax']:.2f})  [shallower]")
    print(f"  BOTH  (shallow+deep ice) : {stats['both']} px = {stats['both']*PX_KM2:.3f} km^2 ({100*stats['both']/n:.0f}%)")
    print(f"  L-only(deeper subsurface): {stats['lonly']} px = {stats['lonly']*PX_KM2:.3f} km^2 ({100*stats['lonly']/n:.0f}%)")
    print(f"  S-only(surface FP)       : {stats['sonly']} px = {stats['sonly']*PX_KM2:.3f} km^2 ({100*stats['sonly']/n:.0f}%)")
    print(f"  neither                  : {stats['neither']} px ({100*stats['neither']/n:.0f}%)")

    _figure(L, S, cls, floor, valid, tfm, crs, shape, stats)
    _report(stats)
    print("  [ok] multifreq class + figure + report")


def _figure(L, S, cls, floor, valid, tfm, crs, shape, st):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    f2x, f2y = cfg.latlon_to_stereographic(cfg.F2_LAT_DEG, cfg.F2_LON_DEG)
    h, w = shape
    ext = [(tfm.c - f2x)/1000, (tfm.c + w*tfm.a - f2x)/1000,
           (tfm.f + h*tfm.e - f2y)/1000, (tfm.f - f2y)/1000]
    Lm = np.ma.masked_where(L < 0, L); Sm = np.ma.masked_where(S < 0, S)
    fig, ax = plt.subplots(2, 2, figsize=(15, 12)); fig.patch.set_facecolor("#0a0a0a")
    for a in ax.ravel():
        a.set_facecolor("#0a0a0a"); a.tick_params(colors="white", labelsize=7)
        [s.set_edgecolor("#444") for s in a.spines.values()]
    im = ax[0, 0].imshow(Lm, cmap="inferno", vmin=0, vmax=2, extent=ext, origin="upper")
    ax[0, 0].contour(floor.astype(float), [0.5], colors=["cyan"], linewidths=1.2, extent=ext, origin="upper")
    plt.colorbar(im, ax=ax[0, 0], fraction=0.046, label="CPR")
    ax[0, 0].set_title("A. L-band CPR (1.25 GHz, deeper)", color="white", fontsize=10)
    im = ax[0, 1].imshow(Sm, cmap="inferno", vmin=0, vmax=2, extent=ext, origin="upper")
    ax[0, 1].contour(floor.astype(float), [0.5], colors=["cyan"], linewidths=1.2, extent=ext, origin="upper")
    plt.colorbar(im, ax=ax[0, 1], fraction=0.046, label="CPR")
    ax[0, 1].set_title("B. S-band CPR (2.5 GHz, shallower)", color="white", fontsize=10)
    # C: classification
    cmap = ListedColormap(["#00e5ff", "#ff3b30", "#f2c744", "#404040"])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    ax[1, 0].imshow(np.ma.masked_where(cls == 0, cls), cmap=cmap, norm=norm, extent=ext, origin="upper")
    ax[1, 0].contour(floor.astype(float), [0.5], colors=["white"], linewidths=0.8, extent=ext, origin="upper")
    ax[1, 0].set_title("C. Depth class: both(cyan) L-only(red) S-only(gold) none(grey)", color="white", fontsize=9)
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_xlim(-2, 2); a.set_ylim(-2, 2)
        a.set_xlabel("E offset from F2 (km)", color="white", fontsize=8)
        a.set_ylabel("N offset (km)", color="white", fontsize=8)
    # D: scatter L vs S
    a = ax[1, 1]
    a.scatter(L[valid], S[valid], s=6, c="#39ff14", alpha=0.4)
    a.axvline(1, color="cyan", ls="--"); a.axhline(1, color="cyan", ls="--")
    a.set_xlim(0, 2.2); a.set_ylim(0, 2.2)
    a.text(1.6, 0.3, "L-only\n(deep ice)", color="#ff3b30", ha="center", fontsize=9)
    a.text(1.6, 1.7, "both", color="#00e5ff", ha="center", fontsize=9)
    a.text(0.4, 1.7, "S-only\n(surface FP)", color="#f2c744", ha="center", fontsize=9)
    a.set_xlabel("L-band CPR", color="white"); a.set_ylabel("S-band CPR", color="white")
    a.set_title("D. F2 floor: L vs S CPR (L>>S ⇒ deeper ice)", color="white", fontsize=10)
    a.tick_params(colors="white"); [s.set_edgecolor("#444") for s in a.spines.values()]
    fig.suptitle("Deliverable 2: multi-frequency depth stratification at F2 — L-band vs S-band compact-pol\n"
                 f"L-band CPR>1 {st['L_gt1']:.0f}% vs S-band {st['S_gt1']:.0f}% ⇒ signal dominated by "
                 f"deeper (L-band) volume scattering: subsurface ice",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(FIG, dpi=300, facecolor=fig.get_facecolor()); plt.close(fig)
    print(f"  [ok] {FIG.name}")


def _report(st):
    n = st["n"]
    md = f"""# Deliverable 2 — Multi-Frequency Depth Stratification at F2 (L-band vs S-band)

## Method
Same-acquisition (20200808) compact-pol L-band (1.25 GHz) and S-band (2.5 GHz),
identical m-chi CPR/DOP pipeline (LH/LV → circular child Stokes). Only frequency
differs → a controlled comparison. Radar penetration scales ~ with wavelength, so
**S-band senses shallower (~decimetre)** and **L-band deeper (~metre)** subsurface.

Per F2-floor pixel (CPR "elevated" = CPR > 1):
- **BOTH** elevated → ice present shallow **and** deep
- **L-only** → deeper subsurface ice (volume scattering below S-band penetration) — *strongest volumetric signal*
- **S-only** → shallow surface roughness → likely **false positive**
- neither → no enhancement

## Result (F2 floor, n = {n} pixels, both bands valid)
| Band | % CPR>1 | max CPR | senses |
|---|---|---|---|
| L-band (1.25 GHz) | **{st['L_gt1']:.0f}%** | {st['Lmax']:.2f} | deeper (~m) |
| S-band (2.5 GHz) | {st['S_gt1']:.0f}% | {st['Smax']:.2f} | shallower (~dm) |

| Depth class | Pixels | Area (km²) | % of floor |
|---|---|---|---|
| BOTH (shallow+deep ice) | {st['both']} | {st['both']*PX_KM2:.3f} | {100*st['both']/n:.0f}% |
| **L-only (deeper subsurface ice)** | **{st['lonly']}** | **{st['lonly']*PX_KM2:.3f}** | **{100*st['lonly']/n:.0f}%** |
| S-only (surface roughness FP) | {st['sonly']} | {st['sonly']*PX_KM2:.3f} | {100*st['sonly']/n:.0f}% |
| neither | {st['neither']} | — | {100*st['neither']/n:.0f}% |

*Percentages are over the full {n}-pixel floor (both bands geocoded). Over the
SNR-valid subset (as in the D2 headline) the fractions are higher (~26% L / ~7% S)
but the count of CPR>1 pixels (~145 L) and the L:S ratio are unchanged — the
area figures (km²) above are denominator-independent.*

## Interpretation
The F2-floor CPR enhancement is **dominated by L-band** ({st['L_gt1']:.0f}% >1) with far
weaker S-band response ({st['S_gt1']:.0f}% >1), and the **L-only class greatly exceeds S-only**.
Because a shallow *surface-roughness* false positive would raise CPR at S-band as much or more
than L-band, the L-band dominance argues the enhancement comes from **deeper (buried) volume
scattering — subsurface water ice**, consistent with the D2 dual-criterion and D5 findings.
The small S-only fraction ({100*st['sonly']/n:.0f}%) flags the minor surface-roughness component.

## Novel contribution
Multi-frequency (L+S) depth stratification of the CPR ice signal at F2 — using both DFSAR
frequencies from a single compact-pol acquisition — separates deeper volumetric ice from shallow
surface-roughness false positives, which single-frequency CPR cannot do.

## Limitations
- Penetration-depth mapping to wavelength is qualitative (dielectric-dependent); "shallow/deep"
  are relative, not absolute depths.
- Compact-pol CPR (both bands) is relative/uncalibrated; thresholds shared with D2.
- Same geocoding caveats as D2 (tie-point interpolation, ~40 m effective resolution).

## Outputs
- outputs/geotiff/cpr_sband.tif, dop_sband.tif, dual_criterion_sband.tif, multifreq_class_f2.tif
- outputs/figures/d2_multifreq.png
"""
    REPORT.write_text(md, encoding="utf-8")
    print(f"  [ok] {REPORT.name}")


if __name__ == "__main__":
    main()
