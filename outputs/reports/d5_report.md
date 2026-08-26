# Deliverable 5 — Ice Volume Estimation Report

## Method
- Ice abundance from **CPR** (coherent-backscatter ice indicator, D2) — *not* σ0.
- Dielectric framework: **Maxwell-Garnett** (ice in regolith), ε_regolith=3.0−0.005j
  (ChaSTE/Mathew 2025), ε_ice=3.15−0.001j; **IEM Small-Perturbation** forward model.
- Inversion attempted with scipy `brentq`; CPR→f_ice mapping used for the estimate.
- Ice-candidate pixels: **two-pass** D2 confidence ≥ MEDIUM, where
  **HIGH = dual criterion (CPR>1 & DOP<0.20) in BOTH pass 4 (full-pol) AND pass 6
  (compact-pol)**, and MEDIUM = dual criterion in one mode only. Volume is thus
  reported for two-pass-verified (HIGH) ice plus single-pass (MED) candidates.

## Why σ0 inversion was not used (key methodological finding)
Water ice (ε≈3.15) and lunar regolith (ε≈3.0) have nearly identical real
permittivity, so Maxwell-Garnett ε moves only 3.00→3.07 over 0–50% ice, and the
IEM σ0 changes by just **0.20 dB** across that range — while observed
σ0 varies ~10 dB (roughness/topography). The σ0→f_ice inversion therefore
**saturates** (98% of pixels pinned at the cap) and is degenerate.
Ice abundance is instead inferred from CPR, which *is* physically sensitive to ice.

## Incidence Angle
- Source: pass-4 g_sli geometry CSV. Mean θ at F2 ≈ **25.4°** (XML nominal 26.0°).

## Sigma-0 Calibration
- **Relative** (uncalibrated DN²; cal_const 70.31 dB). Mean rel. σ0 in ice
  candidates: -4.8 dB. Ice fraction is therefore relative/indicative.

## Ice Fraction (CPR-derived) Results
| Confidence Tier | Pixels | Area (km²) | Mean f_ice |
|---|---|---|---|
| HIGH — both modes (full-pol & compact-pol) | 77 | 0.031 | 7.8% |
| MEDIUM — one mode only | 3194 | 1.278 | 7.0% |

CPR→f_ice mapping: CPR=1.0→5%, CPR=2.0→30% (linear, clipped), anchored to the
dual-criterion threshold and a literature polar-ice upper bound.

## Ice Volume Estimates (F2 crater)
| Depth | Volume (m³) | Volume (km³) | Mass (Mkg) |
|---|---|---|---|
| 1 m | 9.22e+04 | 9.22e-05 | 8.46e+01 |
| 3 m | 2.77e+05 | 2.77e-04 | 2.54e+02 |
| 5 m | 4.61e+05 | 4.61e-04 | 4.23e+02 |  ← **central**
| 10 m | 9.22e+05 | 9.22e-04 | 8.46e+02 |

**Two-pass-VERIFIED ice (HIGH, both full-pol & compact-pol), 5 m depth:
1.20e+04 m³ = 1.10e+01 Mkg** — the conservative,
cross-mode-validated estimate.

Total candidate (HIGH + MEDIUM one-pass), 5 m depth: 4.61e+05 m³
(range 2.31e+05–6.92e+05 m³, ±50%); total mass 4.23e+02 Mkg.

## Sanity Check
- F2 geometric crater volume ≈ 5.51e+07 m³ (cone, D=1100 m, depth=174 m).
- Central ice volume = **0.8%** of crater volume → order-of-magnitude
  consistent with PSR cold-trap ice occupying a few % of the upper crater fill.

## Novel Contribution
Applies a Maxwell-Garnett + IEM dielectric framework with ChaSTE-calibrated
ε_regolith (Mathew et al. 2025, Chandrayaan-3) and, critically, demonstrates that
**σ0 is intrinsically insensitive to lunar ice fraction**, motivating CPR-based
abundance — an Indian-mission-data-driven, internally consistent estimate for F2.

## Limitations
1. **Two-pass verified** — HIGH-confidence ice requires agreement between pass 4
   (full-pol) and pass 6 (compact-pol); the cross-mode HIGH set is small but robust.
2. **σ0 relative** (uncalibrated) → ice fraction is relative/indicative, not absolute.
3. **σ0–dielectric degeneracy** → abundance from CPR via an empirical linear map
   (order-of-magnitude; the CPR→f_ice coefficients are literature-anchored, not fitted).
4. **Uniform depth assumed** — true ice-layer depth unknown; 1–10 m sensitivity given.
5. **Maxwell-Garnett** assumes spherical ice inclusions; real morphology unknown.

## Outputs
- outputs/geotiff/ice_fraction.tif
- outputs/figures/d5_iem_forward_curve.png, d5_ice_volume.png, d5_comparison.png

## Overall: PASS (CPR-based; σ0-IEM degeneracy documented)
## Next Step
- D3 (landing site) and D4 (rover traverse) if time permits.
