"""Deliverable 4 figures: traverse (4-panel) + waypoints/solar-timeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import d4_traverse as d4
import d1_config as cfg


def _hillshade(z, px=10.0, az=315.0, alt=35.0):
    z = np.nan_to_num(z, nan=float(np.nanmin(z[np.isfinite(z)])) if np.isfinite(z).any() else 0)
    gy, gx = np.gradient(z, px)
    slope = np.arctan(np.hypot(gx, gy)); aspect = np.arctan2(-gx, gy)
    a = np.radians(360 - az + 90); al = np.radians(alt)
    return np.clip(np.sin(al) * np.cos(slope) + np.cos(al) * np.sin(slope) * np.cos(a - aspect), 0, 1)


def render_all(dem, slope, psr, illum, path, naive, start, goal, wt, crs, inpsr, seg, m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    f2x, f2y = cfg.latlon_to_stereographic(cfg.F2_LAT_DEG, cfg.F2_LON_DEG)
    h, w = slope.shape
    ext = [(wt.c - f2x)/1000, (wt.c + w*wt.a - f2x)/1000,
           (wt.f + h*wt.e - f2y)/1000, (wt.f - f2y)/1000]

    def rc_km(rc):
        r, c = rc
        return (wt.c + (c+0.5)*wt.a - f2x)/1000, (wt.f + (r+0.5)*wt.e - f2y)/1000
    pax = np.array([rc_km(p) for p in path])
    nax = np.array([rc_km(p) for p in naive])
    sk = rc_km(start); gk = rc_km(goal)
    hs = _hillshade(dem)

    fig, ax = plt.subplots(2, 2, figsize=(17, 12)); fig.patch.set_facecolor("#0a0a0a")
    for a in ax.ravel():
        a.set_facecolor("#0a0a0a"); a.tick_params(colors="white", labelsize=7)
        [s.set_edgecolor("#444") for s in a.spines.values()]

    # A: traverse on hillshade
    a = ax[0, 0]
    a.imshow(hs, cmap="gray", extent=ext, origin="upper", aspect="equal")
    pr = np.ma.masked_where(~psr, np.ones_like(slope))
    a.imshow(pr, cmap=ListedColormap(["#1f6feb"]), extent=ext, origin="upper", alpha=0.30, aspect="equal")
    a.plot(nax[:, 0], nax[:, 1], "--", color="#ff3b30", lw=1.2, label="naive line")
    a.plot(pax[:, 0], pax[:, 1], "-", color="cyan", lw=2, label="A* path")
    a.plot(*sk, "*", color="#39ff14", ms=18, label="landing (Site A)")
    a.plot(*gk, "o", color="yellow", ms=10, label="F2 ice goal")
    a.legend(facecolor="#222", labelcolor="white", fontsize=7, loc="lower right")
    a.set_title("A. Traverse — A* (cyan) vs naive (red), PSR (blue)", color="white", fontsize=10)

    # B: cost grid
    a = ax[0, 1]
    tan_s = np.tan(np.radians(np.clip(slope, 0, 89)))
    cost = (1 + 3*tan_s) * np.where(psr, 5.0, 1.0)
    im = a.imshow(np.log10(cost), cmap="magma", extent=ext, origin="upper", aspect="equal")
    a.plot(pax[:, 0], pax[:, 1], "-", color="cyan", lw=1.5)
    plt.colorbar(im, ax=a, fraction=0.046, label="log10 cost multiplier")
    a.set_title("B. Cost grid (slope×PSR) with A* path", color="white", fontsize=10)

    # C: profile along path
    a = ax[1, 0]
    d = np.concatenate([[0], np.cumsum(seg)]) / 1000
    elev = np.array([dem[r, c] for r, c in path])
    slp = np.array([slope[r, c] for r, c in path])
    a.plot(d, elev, color="#f2c744", lw=1.5, label="elevation")
    a.set_xlabel("distance (km)", color="white"); a.set_ylabel("elevation (m)", color="#f2c744")
    a.tick_params(axis="y", colors="#f2c744")
    a2 = a.twinx(); a2.plot(d, slp, color="#ff6b6b", lw=1, alpha=0.8, label="slope")
    a2.axhline(25, color="red", ls=":", lw=1); a2.set_ylabel("slope (deg)", color="#ff6b6b")
    a2.tick_params(axis="y", colors="#ff6b6b"); a2.set_ylim(0, 35)
    # PSR entry
    ent = np.argmax(inpsr) if inpsr.any() else len(path)-1
    a.axvline(d[ent], color="cyan", ls="--"); a.text(d[ent], np.nanmax(elev), " PSR entry", color="cyan", fontsize=8)
    a.set_title("C. Elevation + slope along path (25° limit dotted)", color="white", fontsize=10)
    a.tick_params(colors="white"); [s.set_edgecolor("#444") for s in a.spines.values()]

    # D: PSR entry zoom + staging
    a = ax[1, 1]
    er, ec = path[ent]
    hw = 120
    r0, r1 = max(0, er-hw), min(h, er+hw); c0, c1 = max(0, ec-hw), min(w, ec+hw)
    sub_ext = [(wt.c+(c0)*wt.a - f2x)/1000, (wt.c+(c1)*wt.a - f2x)/1000,
               (wt.f+(r1)*wt.e - f2y)/1000, (wt.f+(r0)*wt.e - f2y)/1000]
    a.imshow(hs[r0:r1, c0:c1], cmap="gray", extent=sub_ext, origin="upper", aspect="equal")
    a.imshow(np.ma.masked_where(~psr[r0:r1, c0:c1], np.ones((r1-r0, c1-c0))),
             cmap=ListedColormap(["#1f6feb"]), extent=sub_ext, origin="upper", alpha=0.3, aspect="equal")
    a.plot(pax[:, 0], pax[:, 1], "-", color="cyan", lw=2)
    a.plot(*rc_km(path[ent]), "o", color="orange", ms=10)
    # staging = last sunlit waypoint before entry
    stg = ent-1
    while stg > 0 and inpsr[stg]:
        stg -= 1
    a.plot(*rc_km(path[stg]), "s", color="#39ff14", ms=10)
    a.annotate("staging\n(last sun)", rc_km(path[stg]), color="#39ff14", fontsize=8)
    a.annotate("PSR entry", rc_km(path[ent]), color="orange", fontsize=8)
    a.set_xlim(sub_ext[0], sub_ext[1]); a.set_ylim(sub_ext[2], sub_ext[3])
    a.set_title("D. PSR-entry dash — staging point (green) → shadow", color="white", fontsize=10)
    for a in ax.ravel():
        a.set_xlabel(a.get_xlabel() or "E offset from F2 (km)", color="white", fontsize=8)
    fig.suptitle("Deliverable 4: Rover traverse Site A → F2 ice — "
                 f"{m['length']/1000:.1f} km ({m['psr_len']/1000:.1f} km in PSR), max slope {m['smax']:.0f}°\n"
                 f"PSR dash {m['dash_h']:.0f} h @ {d4.SPEED_PSR} m/s vs {m['battery']:.0f} h battery → "
                 f"{'FEASIBLE' if m['feasible'] else 'INFEASIBLE on single charge'}",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(d4.FIG1, dpi=300, facecolor=fig.get_facecolor()); plt.close(fig)
    print(f"      [ok] {d4.FIG1.name}")

    # ── Figure 2: solar timeline + waypoint summary ──────────────────────────
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[1, 2]); fig.patch.set_facecolor("#0a0a0a")
    a = ax[0]; a.set_facecolor("#0a0a0a")
    d = np.concatenate([[0], np.cumsum(seg)]) / 1000
    inp = np.array([psr[r, c] for r, c in path])
    for i in range(len(path)-1):
        a.axvspan(d[i], d[i+1], color=("#101010" if inp[i+1] else "#f2c744"))
    a.set_yticks([]); a.set_xlim(0, d[-1]); a.set_xlabel("distance along traverse (km)", color="white")
    a.set_title("Solar exposure along traverse (gold = sunlit, black = PSR / battery-only)",
                color="white", fontsize=11)
    a.tick_params(colors="white")
    # cumulative time
    a = ax[1]; a.set_facecolor("#0a0a0a")
    cumt = np.concatenate([[0], np.cumsum([seg[i]/(d4.SPEED_PSR if inp[i+1] else d4.SPEED_SUN) for i in range(len(seg))])])/3600
    a.plot(d, cumt, color="cyan", lw=2)
    a.axhline(m['battery'], color="red", ls="--", label=f"battery endurance {m['battery']:.0f} h")
    ent = np.argmax(inp) if inp.any() else 0
    a.axvline(d[ent], color="#1f6feb", ls=":", label="PSR entry")
    a.fill_between(d, 0, cumt, where=inp[:len(d)], color="#1f6feb", alpha=0.2)
    a.set_xlabel("distance (km)", color="white"); a.set_ylabel("cumulative time (h)", color="white")
    a.legend(facecolor="#222", labelcolor="white"); a.tick_params(colors="white")
    [s.set_edgecolor("#444") for s in a.spines.values()]
    a.set_title(f"Cumulative traverse time vs battery — total {m['total_h']:.0f} h, "
                f"{m['psr_h']:.0f} h in PSR (>> {m['battery']:.0f} h battery)", color="white", fontsize=11)
    fig.suptitle("Deliverable 4: traverse solar/time profile", color="white", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(d4.FIG2, dpi=300, facecolor=fig.get_facecolor()); plt.close(fig)
    print(f"      [ok] {d4.FIG2.name}")
