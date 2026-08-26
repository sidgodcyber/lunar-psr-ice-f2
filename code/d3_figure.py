"""Deliverable 3 figures: landing-site map (4-panel) + decision matrix."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import d3_landing as d3  # for constants (FIG paths)


def _hillshade(z, px=5.0, az=315.0, alt=35.0):
    z = np.nan_to_num(z, nan=float(np.nanmin(z)))
    gy, gx = np.gradient(z, px)
    slope = np.arctan(np.hypot(gx, gy)); aspect = np.arctan2(-gx, gy)
    a = np.radians(360 - az + 90); al = np.radians(alt)
    return np.clip(np.sin(al) * np.cos(slope) + np.cos(al) * np.sin(slope) * np.cos(a - aspect), 0, 1)


def render_all(dem, slope, illum, psr, dist, feasible, top3, win, fx, fy, tfm, xs, ys,
               illum_floor, bearing):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.colors import ListedColormap

    # extent in km offset from F2
    ext = [(xs[0] - fx) / 1000, (xs[-1] - fx) / 1000, (ys[-1] - fy) / 1000, (ys[0] - fy) / 1000]

    def xk(x): return (x - fx) / 1000
    def yk(y): return (y - fy) / 1000

    hs = _hillshade(dem)
    fig, ax = plt.subplots(2, 2, figsize=(16, 14)); fig.patch.set_facecolor("#0a0a0a")
    for a in ax.ravel():
        a.set_facecolor("#0a0a0a"); a.tick_params(colors="white", labelsize=7)
        [s.set_edgecolor("#444") for s in a.spines.values()]
        a.set_xlabel("E offset from F2 (km)", color="white", fontsize=8)
        a.set_ylabel("N offset from F2 (km)", color="white", fontsize=8)

    def mark_sites(a):
        for i, s in enumerate(top3):
            a.plot(xk(s["x"]), yk(s["y"]), "o", mfc="none", mec="white", ms=14, mew=2)
            a.text(xk(s["x"]) + 0.4, yk(s["y"]) + 0.4, chr(65 + i), color="white",
                   fontsize=12, fontweight="bold")
        a.plot(0, 0, "*", color="yellow", ms=18)   # F2 (may be at edge)
        a.annotate("F2", (0, 0), color="yellow", fontsize=10, fontweight="bold", xytext=(-2, -1.2))

    # A: feasibility
    a = ax[0, 0]
    a.imshow(hs, cmap="gray", extent=ext, origin="upper")
    fe = np.ma.masked_where(~feasible, np.ones_like(dem))
    a.imshow(fe, cmap=ListedColormap(["#39ff14"]), extent=ext, origin="upper", alpha=0.45)
    a.contour((psr == 1).astype(float), [0.5], colors=["#1f6feb"], linewidths=1.0, extent=ext, origin="upper")
    mark_sites(a)
    a.set_title("A. Feasibility (green) — hillshade, PSR (blue), sites, F2", color="white", fontsize=11)

    # B: illumination
    a = ax[0, 1]
    im = a.imshow(illum * 100, cmap="inferno", vmin=0, vmax=50, extent=ext, origin="upper")
    a.contour(illum, [illum_floor], colors=["cyan"], linewidths=1.0, extent=ext, origin="upper")
    plt.colorbar(im, ax=a, fraction=0.046, label="annual sunlit (%)")
    mark_sites(a)
    a.set_title(f"B. Illumination (cyan = {illum_floor*100:.0f}% floor; no PEL >50% here)", color="white", fontsize=11)

    # C: slope
    a = ax[1, 0]
    im = a.imshow(np.clip(slope, 0, 30), cmap="hot", extent=ext, origin="upper")
    a.contour(slope, [10.0], colors=["cyan"], linewidths=0.8, extent=ext, origin="upper")
    plt.colorbar(im, ax=a, fraction=0.046, label="slope (deg)")
    mark_sites(a)
    a.set_title("C. Slope (cyan = 10° safe contour)", color="white", fontsize=11)

    # D: chosen site close-up + pad + bearing to F2
    a = ax[1, 1]
    wr, wc = win["row"], win["col"]
    hw = 300  # +/-1.5 km at 5 m
    r0, r1 = max(0, wr - hw), min(dem.shape[0], wr + hw)
    c0, c1 = max(0, wc - hw), min(dem.shape[1], wc + hw)
    sub_ext = [xk(xs[c0]), xk(xs[c1 - 1]), yk(ys[r1 - 1]), yk(ys[r0])]
    a.imshow(_hillshade(dem[r0:r1, c0:c1]), cmap="gray", extent=sub_ext, origin="upper")
    im = a.imshow(np.clip(slope[r0:r1, c0:c1], 0, 20), cmap="hot", alpha=0.4, extent=sub_ext, origin="upper")
    # 50 m pad
    a.add_patch(Rectangle((xk(win["x"]) - 0.025, yk(win["y"]) - 0.025), 0.05, 0.05,
                          fill=False, ec="#39ff14", lw=2))
    # bearing arrow toward F2
    dx, dy = (fx - win["x"]), (fy - win["y"]); n = np.hypot(dx, dy)
    a.annotate("", xy=(xk(win["x"]) + dx / n * 1.0, yk(win["y"]) + dy / n * 1.0),
               xytext=(xk(win["x"]), yk(win["y"])),
               arrowprops=dict(arrowstyle="-|>", color="yellow", lw=2))
    a.text(xk(win["x"]), yk(win["y"]) - 0.4, f"Site A → F2  {bearing:.0f}°\n{win['dist']:.1f} km",
           color="yellow", fontsize=9, ha="center")
    a.set_title("D. Winner Site A — 50 m pad (green) + bearing to F2", color="white", fontsize=11)

    fig.suptitle("Deliverable 3: Landing site proposal near Faustini rim (F2 access)\n"
                 f"Winner Site A: {win['lat']:.2f}°S, {win['lon']:.2f}°E | {win['dist']:.1f} km to F2 | "
                 f"illum {win['illum']*100:.0f}% | slope {win['slope']:.1f}°",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(d3.FIG1, dpi=300, facecolor=fig.get_facecolor()); plt.close(fig)
    print(f"      [ok] {d3.FIG1.name}")

    # ── decision matrix (grouped bars, normalized) ───────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(15, 6)); fig.patch.set_facecolor("#0a0a0a")
    labels = [f"Site {chr(65+i)}" for i in range(len(top3))]
    crit = {"illum %": [s["illum"] * 100 for s in top3],
            "dist km": [s["dist"] for s in top3],
            "slope °": [s["slope"] for s in top3],
            "score": [s["score"] * 100 for s in top3]}
    a = ax[0]; a.set_facecolor("#0a0a0a"); w = 0.2
    xpos = np.arange(len(labels))
    for k, (nm, v) in enumerate(crit.items()):
        a.bar(xpos + k * w, v, w, label=nm)
    a.set_xticks(xpos + 1.5 * w); a.set_xticklabels(labels, color="white")
    a.legend(facecolor="#222", labelcolor="white"); a.tick_params(colors="white")
    [s.set_edgecolor("#444") for s in a.spines.values()]
    a.set_title("Decision criteria by site", color="white", fontsize=11)
    # weighted score highlight
    a = ax[1]; a.set_facecolor("#0a0a0a")
    scores = [s["score"] for s in top3]
    colors = ["#ff3b30"] + ["#888"] * (len(top3) - 1)
    a.barh(labels[::-1], scores[::-1], color=colors[::-1])
    a.set_title("Weighted score (Site A = winner)", color="white", fontsize=11)
    a.tick_params(colors="white"); [s.set_edgecolor("#444") for s in a.spines.values()]
    for i, sc in enumerate(scores[::-1]):
        a.text(sc, i, f" {sc:.3f}", color="white", va="center")
    fig.suptitle("Deliverable 3: Landing-site decision matrix", color="white", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(d3.FIG2, dpi=300, facecolor=fig.get_facecolor()); plt.close(fig)
    print(f"      [ok] {d3.FIG2.name}")
