"""IEM (Small Perturbation) backscatter forward model + Maxwell-Garnett mixing
and per-pixel ice-fraction inversion (Deliverable 5).

NOTE on conditioning: lunar regolith (eps~3.0) and water ice (eps~3.15) have
nearly identical real permittivity, so sigma0 is only weakly sensitive to ice
fraction through the dielectric. The sigma0->f_ice inversion is therefore
intrinsically ill-conditioned; CPR (Deliverable 2) is the robust ice indicator.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

# Lunar dielectric end-members
EPS_REGOLITH: complex = complex(3.0, -0.005)   # ChaSTE-calibrated (Mathew 2025)
EPS_ICE: complex = complex(3.15, -0.001)        # pure water ice (~25 K)
WAVELENGTH_L: float = 0.24                       # m (L-band, 1.25 GHz)


def maxwell_garnett_permittivity(f_ice: float) -> complex:
    """Effective permittivity of ice inclusions in a regolith host (Maxwell-Garnett).

    Args:
        f_ice: volumetric ice fraction (0..0.5).

    Returns:
        Effective complex permittivity.
    """
    eps_r = EPS_REGOLITH
    eps_i = EPS_ICE
    beta = (eps_i - eps_r) / (eps_i + 2.0 * eps_r)
    return eps_r * (1.0 + 3.0 * f_ice * beta / (1.0 - f_ice * beta))


def iem_sigma0_spm(theta_rad: float, eps_r: complex,
                   sigma_s: float = 0.05, l: float = 0.20,
                   wavelength: float = WAVELENGTH_L) -> float:
    """First-order Small-Perturbation backscatter (co-pol), linear sigma0.

    Valid for k*sigma_s < 0.3; at L-band sigma_s=0.05 m is slightly outside
    strict validity but gives physically reasonable relative behaviour.
    """
    k = 2.0 * np.pi / wavelength
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    eps = complex(eps_r)
    sqrt_term = np.sqrt(eps - sin_t ** 2)
    r_v = (eps * cos_t - sqrt_term) / (eps * cos_t + sqrt_term)
    f_vv = 2.0 * r_v / cos_t
    k_rho = 2.0 * k * sin_t
    w = (l ** 2 / (4.0 * np.pi)) * np.exp(-(k_rho * l) ** 2 / 4.0)
    sigma0 = k ** 4 * sigma_s ** 2 * np.abs(f_vv) ** 2 * w * cos_t ** 2
    return float(np.real(sigma0))


def cpr_to_ice_fraction(cpr: np.ndarray, cpr0: float = 1.0, cpr1: float = 2.0,
                        f0: float = 0.05, f1: float = 0.30) -> np.ndarray:
    """Map CPR to volumetric ice fraction (physically-motivated linear relation).

    Because sigma0 cannot constrain ice fraction (eps_ice~eps_regolith => IEM
    response is ~flat), ice abundance is inferred from CPR, the coherent-backscatter
    ice indicator. The mapping is anchored to the dual-criterion threshold
    (CPR=cpr0=1.0 => f0, the minimum detectable ice) and a high-CPR icy regime
    (CPR=cpr1=2.0 => f1, a literature upper bound for polar ice abundance),
    clipped to [0, f1]. This is an order-of-magnitude estimate, not absolute.

    Args:
        cpr: CPR array.
        cpr0, cpr1: CPR anchors. f0, f1: corresponding ice fractions.

    Returns:
        ice fraction array, clipped to [0, f1].
    """
    frac = f0 + (f1 - f0) * (cpr - cpr0) / (cpr1 - cpr0)
    return np.clip(frac, 0.0, f1).astype(np.float32)


def invert_ice_fraction(sigma0_obs: float, theta_rad: float,
                        sigma_s: float = 0.05, l: float = 0.20,
                        wavelength: float = WAVELENGTH_L,
                        f_max: float = 0.45) -> float:
    """Solve iem_sigma0_spm(eps_mix(f_ice)) = sigma0_obs for f_ice via Brent.

    Returns f_ice in [0, f_max], the nearest boundary if no sign change, or NaN.
    """
    def residual(f):
        eps = maxwell_garnett_permittivity(f)
        return iem_sigma0_spm(theta_rad, eps, sigma_s, l, wavelength) - sigma0_obs

    try:
        f_lo, f_hi = residual(0.0), residual(f_max)
        if f_lo * f_hi > 0:
            return 0.0 if abs(f_lo) < abs(f_hi) else f_max
        return float(brentq(residual, 0.0, f_max, xtol=1e-4, maxiter=100))
    except Exception:
        return float("nan")
