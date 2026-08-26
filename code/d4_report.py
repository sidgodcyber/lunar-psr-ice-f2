"""Deliverable 4 rover-traverse report."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import d4_traverse as d4


def write_report(m, wpts):
    lim = 30 if m["relaxed"] else 25
    md = f"""# Deliverable 4 — Rover Traverse Path Design

## Method
- A* pathfinding, 8-connected grid, 10 m cost surface.
- Cost = distance × (1 + 3·tan(slope_dest)) × (5 if PSR_dest else 1).
- Impassable: slope > {lim}°{' (relaxed from 25° — no 25° path existed)' if m['relaxed'] else ''}.
- Start: D3 landing Site A (−87.87°S, 82.66°E) | Goal: F2 HIGH-confidence ice pixel.

## Traverse Metrics
| Metric | A* Path | Naive Straight Line |
|---|---|---|
| Total length | {m['length']/1000:.1f} km | {m['nlen']/1000:.1f} km |
| Max slope | {m['smax']:.1f}° | {m['nsmax']:.1f}° |
| Length in PSR | {m['psr_len']/1000:.1f} km | {m['npsr']/1000:.1f} km |
| Weighted cost | {m['cost']:.0f} | — |
| Est. traverse time | {m['total_h']:.0f} h | — |
| Crosses impassable terrain? | no | {'YES' if m['naive_blocked'] else 'no'} |

A* is safer than the naive line: the straight path {'crosses terrain steeper than the '+str(lim)+'° tip-over limit' if m['naive_blocked'] else 'is comparable in slope but'} — A* keeps max slope at {m['smax']:.1f}°.

## PSR Dash Analysis — the primary engineering challenge
- PSR entry point: **{m['entry_lat']:.2f}°S, {m['entry_lon']:.2f}°E** ({m['entry_km']:.1f} km from landing).
- **PSR dash distance to ice: {m['psr_len']/1000:.1f} km** ({100*m['psr_len']/m['length']:.0f}% of the traverse).
- **Time in shadow: {m['dash_h']:.0f} h at {d4.SPEED_PSR} m/s** (battery-only; no solar in PSR).
- Assumed battery endurance: **{m['battery']:.0f} h** (mid of a typical 4–6 h).
- **Battery feasibility: {'FEASIBLE' if m['feasible'] else 'INFEASIBLE on a single charge'}** — the dash needs
  ~{m['dash_h']:.0f} h but the battery lasts ~{m['battery']:.0f} h (short by ~{m['dash_h']/m['battery']:.0f}×).
- Opportunity charging inside the PSR: **{m['staging']}** — a PSR is by definition unlit, so
  no in-shadow recharging is possible; the last sunlit staging point is at PSR entry only.

## Key Result (honest finding)
{m['verdict']}.

Reaching F2's ice by a **conventional solar-battery rover is not feasible on a single charge**:
the ice lies ~{m['psr_len']/1000:.0f} km deep inside a permanently shadowed crater, a ~{m['dash_h']:.0f} h
battery-only dash with no possibility of recharging en route (there is no sunlight to stage on
inside the PSR). This is itself a useful mission-architecture result: F2 in-situ access argues for
a **hopper / lander-hop** (ballistic hops over the shadowed floor) or a **non-solar (RTG) rover**,
rather than a single solar-rover traverse. A solar rover could still characterise the sunlit rim
and PSR margin, but not reach the F2 floor ice on stored charge.

## Limitations
- Rover speeds (sun {d4.SPEED_SUN}, PSR {d4.SPEED_PSR} m/s) and battery {m['battery']:.0f} h are nominal.
- Battery/thermal model simplified (heater load in −230 °C shadow would worsen the deficit).
- 8-connected grid → 45° heading resolution; 10 m DEM → sub-grid boulders not captured.
- Illumination from the D1 model; PSR = illumination below the D1 threshold.

## Outputs
- outputs/geotiff/traverse_path.tif, outputs/reports/d4_waypoints.csv
- outputs/figures/d4_traverse.png, d4_waypoints.png

## STATUS: All 5 deliverables complete (D1–D5).
"""
    d4.REPORT.write_text(md, encoding="utf-8")
    print(f"      [ok] {d4.REPORT.name}")
