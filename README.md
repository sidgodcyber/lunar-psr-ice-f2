# lunar-psr-ice-f2

Water-ice detection and mission planning for **crater F2** (−87.39°S, 82.31°E) in the
lunar south polar region, built from **LOLA LDEM** topography and **Chandrayaan-2 DFSAR**
polarimetric SAR.

The pipeline runs as five gated deliverables — each one self-verifies against published
ground truth before the next stage consumes its output.

## Deliverables

| # | Stage | Key outputs |
|---|---|---|
| D1 | PSR + DSC mapping from LOLA LDEM 85°S 10 m/px; annual illumination model; F2 located by catalog coordinates | `psr_mask.tif`, `dsc_mask.tif`, `illumination_fraction.tif`, `f2_aoi.tif` |
| D2 | DFSAR CPR/DOP ice detection on the F2 floor — L-band full-pol (pass 4), compact-pol (pass 6), S-band, and a two-pass merge | `cpr_*.tif`, `dop_*.tif`, `dual_criterion_*.tif`, `ice_confidence_f2.tif` |
| D3 | Landing-site selection under slope, illumination, PSR-exclusion and rover-range constraints | `d3_landing_site.png`, decision matrix |
| D4 | Rover traverse from the landing site to the F2 ice deposit | `traverse_path.tif`, `d4_waypoints.csv` |
| D5 | Ice-volume estimate via Maxwell-Garnett mixing + IEM small-perturbation forward model | `ice_fraction.tif`, `d5_report.md` |

Written reports for every stage live in [`outputs/reports/`](outputs/reports/), figures in
[`outputs/figures/`](outputs/figures/), and rasters in [`outputs/geotiff/`](outputs/geotiff/).

## Selected results

- **F2 verified against Sinha 2026**: diameter 1080 m (paper 1100 m), floor −2894 m
  (paper −2860 m), nearest DSC candidate 0.09 km from the catalog position.
- **Landing Site A**: −87.87°S, 82.66°E — 29.3% annual illumination, 5.1° mean slope,
  14.6 km from F2, outside the PSR.
- **No Peak-of-Eternal-Light exists within rover range of F2**; the >70% illumination
  constraint is relaxed to >22% and documented in [`d3_report.md`](outputs/reports/d3_report.md).
- **σ0 inversion is degenerate for lunar ice** (ice ε≈3.15 vs regolith ε≈3.0 ⇒ 0.20 dB
  swing across 0–50% ice), so ice abundance is inferred from **CPR**, not σ0 — see
  [`d5_report.md`](outputs/reports/d5_report.md).

## Layout

```
code/            deliverable scripts (d1_*.py … d5_*.py) + utils/
  utils/         dem_io, sar_io, illumination, polarimetry, iem, morphology
outputs/
  geotiff/       GeoTIFF rasters
  figures/       publication figures
  reports/       per-deliverable self-verification reports
data/            NOT in git — raw + processed mission data (~50 GB)
```

## Running

Paths are rooted at `BASE_DIR` in [`code/d1_config.py`](code/d1_config.py); set it to your
checkout before running.

```bash
python code/d1_psr_dsc_mapping.py   # D1
python code/d2_dfsar_ice.py         # D2 (pass 4, L-band full-pol)
python code/d2_pass6.py             # D2 (pass 6, compact-pol)
python code/d2_twopass.py           # D2 two-pass merge
python code/d3_landing.py           # D3
python code/d4_traverse.py          # D4
python code/d5_ice_volume.py        # D5
```

Requires `rasterio`, `numpy`, `scipy`, `matplotlib`, `gdal`, `tqdm`.

## Data sources

- **LOLA LDEM 85°S 10 m/px** — NASA PDS Geosciences Node (`lro-l-lola-4-gdr-v1.0`)
- **Chandrayaan-2 DFSAR** L- and S-band SLI/SRI products — ISRO PRADAN

Mission data is not redistributed here; download it from the sources above into `data/raw/`.
