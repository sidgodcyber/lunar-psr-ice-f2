# Deliverable 2 — Multi-Frequency Depth Stratification at F2 (L-band vs S-band)

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

## Result (F2 floor, n = 1477 pixels, both bands valid)
| Band | % CPR>1 | max CPR | senses |
|---|---|---|---|
| L-band (1.25 GHz) | **10%** | 1.74 | deeper (~m) |
| S-band (2.5 GHz) | 3% | 1.19 | shallower (~dm) |

| Depth class | Pixels | Area (km²) | % of floor |
|---|---|---|---|
| BOTH (shallow+deep ice) | 12 | 0.005 | 1% |
| **L-only (deeper subsurface ice)** | **131** | **0.052** | **9%** |
| S-only (surface roughness FP) | 25 | 0.010 | 2% |
| neither | 1309 | — | 89% |

*Percentages are over the full 1477-pixel floor (both bands geocoded). Over the
SNR-valid subset (as in the D2 headline) the fractions are higher (~26% L / ~7% S)
but the count of CPR>1 pixels (~145 L) and the L:S ratio are unchanged — the
area figures (km²) above are denominator-independent.*

## Interpretation
The F2-floor CPR enhancement is **dominated by L-band** (10% >1) with far
weaker S-band response (3% >1), and the **L-only class greatly exceeds S-only**.
Because a shallow *surface-roughness* false positive would raise CPR at S-band as much or more
than L-band, the L-band dominance argues the enhancement comes from **deeper (buried) volume
scattering — subsurface water ice**, consistent with the D2 dual-criterion and D5 findings.
The small S-only fraction (2%) flags the minor surface-roughness component.

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
