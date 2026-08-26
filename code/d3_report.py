"""Deliverable 3 landing-site report."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import d3_landing as d3


def write_report(top3, win, feasible_km2, n_sites, bearing, psr_cross, illum_floor, max_illum_15km):
    def row(i, s):
        return (f"| {chr(65+i)} | {s['illum']*100:.1f} | {s['dist']:.1f} | {s['slope']:.1f} | "
                f"{s['pad_area']:.0f} | {s['score']:.3f} | {s['lat']:.2f}°S, {s['lon']:.2f}°E | "
                f"{'yes' if s['high_ground'] else 'no'} |")
    rows = "\n".join(row(i, s) for i, s in enumerate(top3))
    md = f"""# Deliverable 3 — Landing Site Proposal

## Constraints Applied
- Slope < 10° over a 50 m × 50 m pad (10×10 px at 5 m)
- **Illumination: relaxed** — see note below
- Distance to F2 < 15 km (rover range)
- Outside the PSR (direct sun for landing + power)
- Earth line-of-sight: approximated as high-ground / anti-pole-facing (flagged per site)

## Illumination note (important)
The D1 annual-illumination metric is *fraction of the year directly sunlit*; near the
pole its physical ceiling is ~0.5 (the Sun is above the local horizontal only ~half
the year). **No Peak-of-Eternal-Light (>70% annual) exists within 15 km of F2** — the
maximum annual illumination anywhere within rover range is **{max_illum_15km*100:.0f}%**
(the well-lit 40–50% rim lies >15 km away). The ">70%" hard constraint is therefore
**relaxed to > {illum_floor*100:.0f}%** (≈{illum_floor/0.5*100:.0f}% of the near-pole
maximum — usable solar power with battery buffering) and illumination is used as the
primary *soft* score to pick the best-lit safe site. (Illumination is also conservative:
the model credits terrain shadowing but not horizon depression on ridge crests.)

## Feasible Area
- Total feasible landing area (slope<10 & illum>{illum_floor*100:.0f}% & outside PSR & <15 km): **{feasible_km2:.2f} km²**
- Number of distinct candidate sites (≥5 pad-centres): **{n_sites}**

## Decision Matrix (top 3)
| Site | Peak illum (%) | Dist F2 (km) | Mean slope (°) | Pad area (m²) | Weighted score | Lat/Lon | Earth-LOS |
|---|---|---|---|---|---|---|---|
{rows}

Weights: illumination 0.35, proximity 0.30, slope 0.25, pad area 0.10.

## Selected Landing Site — Site A
- **Coordinates: {win['lat']:.2f}°S, {win['lon']:.2f}°E** (stereographic {win['x']:.0f}, {win['y']:.0f} m)
- Peak illumination: **{win['illum']*100:.0f}%** annual (≈{win['illum']/0.5*100:.0f}% of near-pole max)
- Distance to F2: **{win['dist']:.1f} km**
- Mean slope: **{win['slope']:.1f}°**  |  Earth-LOS (high ground): {'yes' if win['high_ground'] else 'no'}
- Justification: Site A is the highest-scoring pad — it combines the best available
  illumination for a safe (slope<10°, ≥50 m flat pad), outside-PSR location within rover
  range of F2, on relatively high ground for Earth line-of-sight. It is the best
  compromise between solar power, landing safety, and traverse distance to the F2 PSR.

## Approach Corridor to F2
- Bearing from Site A to F2: **{bearing:.0f}°**
- Corridor length: {win['dist']:.1f} km
- PSR boundary crossed ~{psr_cross:.1f} km from landing (within LM7 extent).
- The rover descends from the sunlit rim into the Faustini PSR to reach F2 (Deliverable 4).

## Limitations
- Illumination from the D1 model (pole-crop, ~190 m effective) reprojected to LM7 5 m;
  it caps near 0.5 (fraction-of-year) and omits horizon depression → **conservative**.
- No PEL within range → the 70% requirement was relaxed (documented above).
- Earth line-of-sight approximated (high-ground proxy), not a full comms/horizon analysis.
- **Boulder hazard not assessed** — the supposed "OHRC optical" data is actually
  DFSAR compact-pol SAR (no optical imagery), so optical boulder mapping is impossible.
- Closest safe outside-PSR terrain is ~9 km from F2, so the traverse is long by design.

## Outputs
- outputs/figures/d3_landing_site.png, d3_decision_matrix.png

## Next: Deliverable 4 — rover traverse from Site A into the Faustini PSR to F2.
"""
    d3.REPORT.write_text(md, encoding="utf-8")
    print(f"      [ok] {d3.REPORT.name}")
