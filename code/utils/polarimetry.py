"""CP decomposition: circular-transmit synthesis -> CPR, DOP (Sinha & Bharti 2026).

For lunar ice the relevant observables are derived by synthesising the response
to a circularly-polarised transmit (as Mini-RF measures directly) from the
quad-pol complex scattering matrix, then taking the *child* wave's Stokes vector
(properly normalised, DOP in [0, 1]).

Transmit Left-circular t = (H + iV)/sqrt(2); received (monostatic, S_vh = S_hv):
    E_h = (S_hh + i S_hv)/sqrt(2)
    E_v = (S_hv + i S_vv)/sqrt(2)
Child Stokes from multilooked second moments M_hh=<|E_h|^2>, M_vv=<|E_v|^2>,
M_hv=<E_h conj(E_v)>:
    g0 = M_hh + M_vv;  g1 = M_hh - M_vv;  g2 = 2 Re M_hv;  g3 = -2 Im M_hv
    CPR = (g0 - g3) / (g0 + g3);  DOP = sqrt(g1^2 + g2^2 + g3^2) / g0
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def synthesize_circular(hh: np.ndarray, hv: np.ndarray, vv: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Left-circular transmit received H/V complex fields (monostatic)."""
    inv = np.float32(1.0 / np.sqrt(2.0))
    e_h = (hh + 1j * hv) * inv
    e_v = (hv + 1j * vv) * inv
    return e_h.astype(np.complex64), e_v.astype(np.complex64)


def cpr_dop_circular(
    m_hh: np.ndarray, m_vv: np.ndarray, m_hv: np.ndarray,
    cpr_clip: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CPR and DOP from multilooked circular child-wave moments.

    Args:
        m_hh: <|E_h|^2> (multilooked, real)
        m_vv: <|E_v|^2> (multilooked, real)
        m_hv: <E_h conj(E_v)> (multilooked, complex)

    Returns:
        (cpr, dop, g0): cpr in [0, cpr_clip], dop in [0, 1], g0 = total power.
    """
    g0 = m_hh + m_vv
    g1 = m_hh - m_vv
    g2 = 2.0 * np.real(m_hv)
    g3 = -2.0 * np.imag(m_hv)
    eps = 1e-10
    cpr = (g0 - g3) / (g0 + g3 + eps)
    cpr = np.clip(cpr, 0.0, cpr_clip)
    dop = np.sqrt(g1 * g1 + g2 * g2 + g3 * g3) / (g0 + eps)
    dop = np.clip(dop, 0.0, 1.0)
    return cpr.astype(np.float32), dop.astype(np.float32), g0.astype(np.float32)
