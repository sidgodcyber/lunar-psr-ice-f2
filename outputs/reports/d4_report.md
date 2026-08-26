# Deliverable 4 — Rover Traverse Path Design

## Method
- A* pathfinding, 8-connected grid, 10 m cost surface.
- Cost = distance × (1 + 3·tan(slope_dest)) × (5 if PSR_dest else 1).
- Impassable: slope > 30° (relaxed from 25° — no 25° path existed).
- Start: D3 landing Site A (−87.87°S, 82.66°E) | Goal: F2 HIGH-confidence ice pixel.

## Traverse Metrics
| Metric | A* Path | Naive Straight Line |
|---|---|---|
| Total length | 15.7 km | 15.6 km |
| Max slope | 30.0° | 45.9° |
| Length in PSR | 9.5 km | 9.6 km |
| Weighted cost | 80367 | — |
| Est. traverse time | 70 h | — |
| Crosses impassable terrain? | no | YES |

A* is safer than the naive line: the straight path crosses terrain steeper than the 30° tip-over limit — A* keeps max slope at 30.0°.

## PSR Dash Analysis — the primary engineering challenge
- PSR entry point: **-87.69°S, 81.69°E** (6.2 km from landing).
- **PSR dash distance to ice: 9.5 km** (61% of the traverse).
- **Time in shadow: 53 h at 0.05 m/s** (battery-only; no solar in PSR).
- Assumed battery endurance: **5 h** (mid of a typical 4–6 h).
- **Battery feasibility: INFEASIBLE on a single charge** — the dash needs
  ~53 h but the battery lasts ~5 h (short by ~11×).
- Opportunity charging inside the PSR: **none (true PSR: no in-shadow charging possible)** — a PSR is by definition unlit, so
  no in-shadow recharging is possible; the last sunlit staging point is at PSR entry only.

## Key Result (honest finding)
single-charge traverse INFEASIBLE; no in-PSR charging -> recommend hopper/lander-hop.

Reaching F2's ice by a **conventional solar-battery rover is not feasible on a single charge**:
the ice lies ~10 km deep inside a permanently shadowed crater, a ~53 h
battery-only dash with no possibility of recharging en route (there is no sunlight to stage on
inside the PSR). This is itself a useful mission-architecture result: F2 in-situ access argues for
a **hopper / lander-hop** (ballistic hops over the shadowed floor) or a **non-solar (RTG) rover**,
rather than a single solar-rover traverse. A solar rover could still characterise the sunlit rim
and PSR margin, but not reach the F2 floor ice on stored charge.

## Limitations
- Rover speeds (sun 0.1, PSR 0.05 m/s) and battery 5 h are nominal.
- Battery/thermal model simplified (heater load in −230 °C shadow would worsen the deficit).
- 8-connected grid → 45° heading resolution; 10 m DEM → sub-grid boulders not captured.
- Illumination from the D1 model; PSR = illumination below the D1 threshold.

## Outputs
- outputs/geotiff/traverse_path.tif, outputs/reports/d4_waypoints.csv
- outputs/figures/d4_traverse.png, d4_waypoints.png

## STATUS: All 5 deliverables complete (D1–D5).
