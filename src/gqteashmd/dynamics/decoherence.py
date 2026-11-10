"""Energy-based decoherence correction (EDC; method reference 7), in two conventions.

With K the kinetic energy, C = ``param`` and, for j != active,
tau_j = (1 / |E_j - E_active|) (1 + C / K) (hbar = 1):

- ``"granucci_persico"``: c_j <- c_j exp(-dt / tau_j). Granucci & Persico, J. Chem. Phys. 126,
  134114 (2007), as reproduced in Runeson, arXiv:2507.18381 (2025), Sec. III.3, eqs. (44a),
  (44b), (45).
- ``"sharc"``: c_j <- c_j exp[-(1/2) dt |E_j - E_active| (1 + C / K)^-1] = c_j exp(-dt / (2 tau_j)),
  the form of the public SHARC 4.1 manual, Sec. 8.7.1 "Energy-based decoherence", eq. (8.25).
  The SHARC 4.1 driver, run as a black box (Task 9), damps exactly at this rate.

In both, the active amplitude is then rescaled keeping its phase so the norm is 1 (manual
eq. (8.26); Runeson eq. (45)).
"""

from __future__ import annotations

import numpy as np

SCHEMES = ("edc", "none")
EDC_CONVENTIONS = ("sharc", "granucci_persico")
_EXPONENT_FACTOR = {"sharc": 0.5, "granucci_persico": 1.0}


def apply_decoherence(
    c: np.ndarray,
    active: int,
    energies: np.ndarray,
    kinetic_energy: float,
    dt: float,
    *,
    scheme: str,
    param: float,
    convention: str,
) -> np.ndarray:
    """Return a new amplitude vector after EDC; the input is not modified.

    For j != active, c_j *= exp(-f dt / tau_j) with f = 1/2 (``"sharc"``) or 1
    (``"granucci_persico"``); c_j is unchanged if the gap or K is zero. The active amplitude is
    then rescaled, keeping its phase, so the norm is exactly 1. If |c_active| = 0, ``c`` is
    returned unchanged (as a copy).
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    if convention not in EDC_CONVENTIONS:
        raise ValueError(f"convention must be one of {EDC_CONVENTIONS}, got {convention!r}")
    out = np.array(c, dtype=complex, copy=True)
    if scheme == "none":
        return out
    if abs(out[active]) == 0.0 or kinetic_energy <= 0.0:
        return out
    factor = 1.0 + param / kinetic_energy
    f = _EXPONENT_FACTOR[convention]
    for j in range(out.size):
        if j == active:
            continue
        gap = abs(float(energies[j]) - float(energies[active]))
        if gap == 0.0:
            continue
        tau = factor / gap
        out[j] *= np.exp(-f * dt / tau)
    others = float(np.sum(np.abs(out) ** 2) - abs(out[active]) ** 2)
    target = max(0.0, 1.0 - others)
    out[active] *= np.sqrt(target) / abs(out[active])
    return out
