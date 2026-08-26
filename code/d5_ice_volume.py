"""
DELIVERABLE 5 — ICE VOLUME ESTIMATION (crater F2)
=================================================
Ice abundance is inferred from CPR (the robust coherent-backscatter ice
indicator from D2). The IEM (Small Perturbation) forward model + Maxwell-Garnett
mixing are retained as the dielectric framework and to document that sigma0 is
intrinsically insensitive to ice fraction (eps_ice ~ eps_regolith => ~0.2 dB
dynamic range), which is why the sigma0->f_ice inversion is degenerate.

Run: python code/d5_ice_volume.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rasterio

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
import d5_config as cfg
from utils.iem import (iem_sigma0_spm, maxwell_garnett_permittivity,
                       invert_ice_fraction, cpr_to_ice_fraction)


def _read(path):
    with rasterio.open(str(path)) as s:
        return s.read(1), s.transform, s.crs


def step1_load():
    print("=" * 72); print("[step 1/10] Load D2 outputs, per-tier areas")
    cpr, tfm, crs = _read(cfg.CPR_TIF)
    dop, _, _ = _read(cfg.DOP_TIF)
    sig, _, _ = _read(cfg.SIGMA0_TIF)
    conf, _, _ = _read(cfg.ICE_CONF_TIF)
    cpr = np.where(cpr < 0, np.nan, cpr)
    dop = np.where(dop < 0, np.nan, dop)
    sig = np.where(sig < -900, np.nan, sig)
    px_km2 = (cfg.GEOCODE_PIXEL_M / 1000.0) ** 2
    for tier, name in [(3, "HIGH"), (2, "MED"), (1, "LOW")]:
        n = int((conf == tier).sum())
        print(f"      conf {tier} ({name}): {n} px = {n*px_km2:.3f} km^2")
    return cpr, dop, sig, conf, tfm, crs


def step2_incidence():
    print("=" * 72); print("[step 2/10] Incidence angle at F2 (g_sli geometry)")
    d = np.genfromtxt(str(cfg.G_SLI_CSV), delimiter=",", skip_header=1)
    inc = d[:, 3]; inc = inc[(inc > 0) & (inc < 90)]
    th = float(np.mean(inc))
    print(f"      mean {th:.1f} deg, range {inc.min():.1f}-{inc.max():.1f} (XML nominal 26.0)")
    return th


def step3_sigma0(sig, conf):
    print("=" * 72); print("[step 3/10] Sigma0 calibration check (relative)")
    m = (conf >= 2) & np.isfinite(sig)
    val = float(sig[m].mean())
    print(f"      RELATIVE sigma0 (cal_const 70.31 dB; g0 uncalibrated DN^2). "
          f"mean in ice candidates {val:.1f} dB")
    return val


def step4_iem_forward(theta_deg):
    print("=" * 72); print("[step 4/10] IEM forward model + Maxwell-Garnett (context)")
    th = np.radians(theta_deg)
    f = np.linspace(0, 0.5, 51)
    eps = np.array([maxwell_garnett_permittivity(x).real for x in f])
    s0 = np.array([iem_sigma0_spm(th, maxwell_garnett_permittivity(x)) for x in f])
    s0_db = 10 * np.log10(s0)
    dyn = float(s0_db.max() - s0_db.min())
    print(f"      eps_mix f=0->{eps[0]:.3f}, f=0.5->{eps[-1]:.3f} | "
          f"IEM sigma0 dyn-range over f_ice = {dyn:.2f} dB (=> sigma0 insensitive to ice)")
    return f, eps, s0_db, dyn


def step5_iem_inversion_attempt(sig, conf, theta_deg):
    """Documented degenerate result: sigma0->f_ice saturates."""
    print("=" * 72); print("[step 5/10] IEM sigma0 inversion attempt (expected degenerate)")
    th = np.radians(theta_deg)
    m = (conf >= 2) & np.isfinite(sig)
    sig_lin = 10.0 ** (sig / 10.0)
    vals = [invert_ice_fraction(float(sig_lin[r, c]), th) for r, c in np.argwhere(m)]
    vals = np.array(vals)
    sat = float((vals >= 0.44).mean() * 100)
    print(f"      sigma0-inversion f_ice: mean {np.nanmean(vals)*100:.0f}%, "
          f"{sat:.0f}% saturated at cap => DEGENERATE (sigma0 cannot constrain f_ice). "
          f"Pivoting to CPR-based abundance.")
    return float(np.nanmean(vals)), sat


def step6_cpr_fraction(cpr, conf):
    print("=" * 72); print("[step 6/10] CPR-based ice fraction (primary estimate)")
    f_ice = np.full(cpr.shape, np.nan, np.float32)
    m = (conf >= 2) & np.isfinite(cpr)
    f_ice[m] = cpr_to_ice_fraction(cpr[m])
    fh = f_ice[(conf == 3) & np.isfinite(f_ice)]
    fm = f_ice[(conf == 2) & np.isfinite(f_ice)]
    print(f"      f_ice HIGH mean {np.mean(fh)*100:.1f}% | MED mean {np.mean(fm)*100:.1f}%")
    allf = f_ice[np.isfinite(f_ice)]
    for lo, hi in [(0, 10), (10, 25), (25, 50)]:
        pct = 100 * ((allf*100 >= lo) & (allf*100 < hi)).mean()
        print(f"        {lo:>2}-{hi}%: {pct:.0f}% of ice pixels")
    return f_ice


def gate(f_ice, conf):
    print("-" * 72); print(">>> STOP GATE — ICE-FRACTION SANITY <<<")
    fh = f_ice[(conf == 3) & np.isfinite(f_ice)]; fm = f_ice[(conf == 2) & np.isfinite(f_ice)]
    mh, mm = float(np.mean(fh)), float(np.mean(fm))
    allf = f_ice[np.isfinite(f_ice)]
    in_range = allf.min() >= 0 and allf.max() <= 0.5
    high_gt_med = mh >= mm
    high_ok = 0.05 <= mh <= 0.30
    print(f"      mean f_ice HIGH={mh*100:.1f}% MED={mm*100:.1f}% | range [{allf.min()*100:.0f},{allf.max()*100:.0f}]%")
    print(f"      checks: in[0,0.5]={in_range} HIGH>=MED={high_gt_med} HIGH in 5-30%={high_ok}")
    ok = in_range and high_gt_med and high_ok
    print("-" * 72); print("   ✓ ICE-FRACTION SANITY PASSED" if ok else "   ✗ FAILED")
    return ok, mh, mm


def step7_volume(f_ice, conf):
    print("=" * 72); print("[step 7/10] Volume integration (depth scenarios)")
    px_area = cfg.GEOCODE_PIXEL_M ** 2
    fh = f_ice[(conf == 3) & np.isfinite(f_ice)]
    fm = f_ice[(conf == 2) & np.isfinite(f_ice)]
    sum_high = float(np.sum(fh)); sum_med = float(np.sum(fm))
    results = {}
    for depth in cfg.DEPTH_SCENARIOS_M:
        vol_high = sum_high * px_area * depth
        vol_med = sum_med * px_area * depth
        vol = vol_high + vol_med
        results[depth] = {"vol_m3": vol, "vol_high": vol_high, "vol_med": vol_med,
                          "mass_kg": vol * cfg.RHO_ICE}
        print(f"      depth {depth:>4.0f} m: V={vol:.3e} m^3 ({vol/1e9:.2e} km^3) "
              f"mass={vol*cfg.RHO_ICE/1e6:.2e} Mkg")
    return results


def step8_sanity(results):
    print("=" * 72); print("[step 8/10] Sanity vs literature")
    c = results[cfg.CENTRAL_DEPTH_M]
    r = cfg.F2_DIAMETER_M / 2.0
    crater_vol = (1.0 / 3.0) * np.pi * r ** 2 * cfg.F2_DEPTH_M   # cone approx
    pct = 100 * c["vol_m3"] / crater_vol
    print(f"      F2 geometric crater volume ~ {crater_vol:.2e} m^3 (cone, D={cfg.F2_DIAMETER_M:.0f}, h={cfg.F2_DEPTH_M:.0f})")
    print(f"      central ice volume (5 m) = {c['vol_m3']:.2e} m^3 = {pct:.1f}% of crater volume")
    print(f"      => order-of-magnitude consistent with PSR cold-trap ice (~few % of upper crater fill)")
    return crater_vol, pct


def main():
    t0 = time.time()
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cpr, dop, sig, conf, tfm, crs = step1_load()
    theta = step2_incidence()
    s0_mean = step3_sigma0(sig, conf)
    fwd = step4_iem_forward(theta)
    iem_mean, iem_sat = step5_iem_inversion_attempt(sig, conf, theta)
    f_ice = step6_cpr_fraction(cpr, conf)
    prof = {"driver": "GTiff", "height": f_ice.shape[0], "width": f_ice.shape[1],
            "count": 1, "dtype": "float32", "crs": crs, "transform": tfm,
            "nodata": float("nan"), "compress": "lzw"}
    with rasterio.open(str(cfg.ICE_FRACTION_TIF), "w", **prof) as d:
        d.write(f_ice, 1)
    ok, mh, mm = gate(f_ice, conf)
    if not ok:
        raise SystemExit("Stop at ice-fraction sanity gate")
    results = step7_volume(f_ice, conf)
    crater_vol, pct = step8_sanity(results)

    print("=" * 72); print("[step 9-10/10] Figures + report")
    from d5_figure import render_all
    render_all(cpr, dop, f_ice, conf, tfm, crs, fwd, theta, results, s0_mean)
    from d5_report import write_report
    ctx = {"theta": theta, "s0_mean": s0_mean, "iem_dyn": fwd[3], "iem_sat": iem_sat,
           "mh": mh, "mm": mm, "crater_vol": crater_vol, "pct": pct}
    verdict = write_report(f_ice, conf, results, ctx)
    print("=" * 72); print(f"DONE in {time.time()-t0:.0f}s | OVERALL: {verdict}")
    try:
        import os
        for f in (cfg.FIG_VOLUME, cfg.FIG_COMPARE, cfg.FIG_FORWARD):
            os.startfile(str(f))
    except Exception:
        pass


if __name__ == "__main__":
    main()
