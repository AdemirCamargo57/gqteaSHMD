"""Energy-based decoherence correction (Granucci & Persico 2007; method reference 7)."""

from __future__ import annotations

import numpy as np

SCHEMES = ("edc", "none")


def apply_decoherence(
    c: np.ndarray,
    active: int,
    energies: np.ndarray,
    kinetic_energy: float,
    dt: float,
    *,
    scheme: str,
    param: float,
) -> np.ndarray:
    """Return a new amplitude vector after EDC; the input is not modified.

    For j != active, tau_j = (1 / |E_j - E_active|) (1 + param / K) and c_j *= exp(-dt / tau_j)
    (unchanged if the gap or K is zero). The active amplitude is then rescaled, keeping its phase,
    so the norm is exactly 1. If |c_active| = 0, ``c`` is returned unchanged (as a copy).
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    out = np.array(c, dtype=complex, copy=True)
    if scheme == "none":
        return out
    if abs(out[active]) == 0.0 or kinetic_energy <= 0.0:
        return out
    factor = 1.0 + param / kinetic_energy
    for j in range(out.size):
        if j == active:
            continue
        gap = abs(float(energies[j]) - float(energies[active]))
        if gap == 0.0:
            continue
        tau = factor / gap
        out[j] *= np.exp(-dt / tau)
    others = float(np.sum(np.abs(out) ** 2) - abs(out[active]) ** 2)
    target = max(0.0, 1.0 - others)
    out[active] *= np.sqrt(target) / abs(out[active])
    return out
