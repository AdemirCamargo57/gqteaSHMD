"""Velocity adjustment after a hop and frustrated-hop handling (method reference 6).

Atomic units; ``masses`` are per degree of freedom (length f). Energy conservation across a hop
at fixed ``q`` is exact: K' = K - dE with dE = E_new - E_old.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

METHODS = ("parallel_vel", "parallel_nac")
FRUSTRATED = ("keep", "reflect")
_A_MIN = 1e-30


@dataclass(frozen=True)
class Adjustment:
    accepted: bool
    velocity: np.ndarray  # adjusted v, or (frustrated) the kept or reflected v


def _frustrated(v: np.ndarray, masses: np.ndarray, u: np.ndarray | None, mode: str) -> Adjustment:
    """Return the velocity after a frustrated hop. ``u`` None means u = M v (v' = -v)."""
    if mode == "keep":
        return Adjustment(False, v.copy())
    if u is None:
        return Adjustment(False, -v)
    minv_u = u / masses
    denom = float(u @ minv_u)
    if denom <= _A_MIN * 2:
        return Adjustment(False, v.copy())
    return Adjustment(False, v - 2.0 * (float(v @ u) / denom) * minv_u)


def adjust_velocity(
    v: np.ndarray,
    masses: np.ndarray,
    delta_e: float,
    *,
    method: str,
    nac: np.ndarray | None,
    frustrated: str,
) -> Adjustment:
    """Adjust ``v`` for a hop with ``delta_e = E_new - E_old``; see method reference (6).

    ``parallel_vel`` with zero kinetic energy cannot put energy into a system at rest: for
    ``delta_e > 0`` the hop is frustrated, for ``delta_e < 0`` the direction is undefined and a
    ``ValueError`` is raised (use ``parallel_nac``); ``delta_e == 0`` leaves ``v`` unchanged.
    ``parallel_nac`` with a vanishing NAC falls back to that rule; at rest and downhill it raises
    a ``ValueError`` naming the zero NAC.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if frustrated not in FRUSTRATED:
        raise ValueError(f"frustrated must be one of {FRUSTRATED}, got {frustrated!r}")
    v = np.asarray(v, dtype=float)
    masses = np.asarray(masses, dtype=float)
    u: np.ndarray | None = None
    if method == "parallel_nac":
        if nac is None:
            raise ValueError("parallel_nac needs the NAC vector (nac is None)")
        u = np.asarray(nac, dtype=float)
        a = 0.5 * float(np.sum(u * u / masses))
        if a > _A_MIN:
            b = float(v @ u)
            disc = b * b - 4.0 * a * delta_e
            if disc < 0.0:
                return _frustrated(v, masses, u, frustrated)
            s = float(np.sqrt(disc))
            # roots q/a and delta_e/q with q = -(b + sign(b) s)/2; the second is the smaller
            q = -0.5 * (b + (s if b >= 0.0 else -s))
            gamma = delta_e / q if q != 0.0 else 0.0
            return Adjustment(True, v + gamma * (u / masses))
    # parallel_vel rule (also the fallback of parallel_nac for a vanishing NAC)
    kin = 0.5 * float(np.sum(masses * v * v))
    if delta_e == 0.0:
        return Adjustment(True, v.copy())
    if kin <= 0.0:
        if delta_e < 0.0:
            if method == "parallel_nac":
                raise ValueError(
                    "cannot release energy into a system at rest: no adjustment direction "
                    "(zero NAC)"
                )
            raise ValueError(
                "parallel_vel cannot put energy into a system at rest; use parallel_nac"
            )
        return _frustrated(v, masses, None, frustrated)
    ratio = 1.0 - delta_e / kin
    if ratio < 0.0:
        return _frustrated(v, masses, None, frustrated)
    return Adjustment(True, float(np.sqrt(ratio)) * v)
