"""Hopping probabilities (Method reference 3-5) and the hop decision."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from gqteashmd.dynamics.propagators import substep_couplings

DENOMINATOR_FLOOR = 1e-14


def _cap(h: np.ndarray) -> np.ndarray:
    total = float(h.sum())
    return h / total if total > 1.0 else h


def fssh_probabilities(
    active: int,
    substep_amplitudes: np.ndarray | Sequence[np.ndarray],
    nac0: np.ndarray,
    nac1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    dt: float,
    *,
    interpolation: str = "midpoint",
) -> np.ndarray:
    """Tully flux from ``active``, T taken at the same substep points as the propagator.

    ``substep_amplitudes`` are the amplitudes at the start of each substep, (N_s, n) (from
    ``propagate``; a sequence of N_s vectors is also accepted). ``interpolation`` must be the
    one ``propagate`` used (``"midpoint"`` or ``"endpoint"``). Substeps with
    |c_active|^2 <= ``DENOMINATOR_FLOOR`` contribute nothing. Returns (n,) with 0 at
    ``active``; capped so the sum is <= 1 (scaled proportionally).
    """
    amps = np.asarray(substep_amplitudes, dtype=complex)
    ns = amps.shape[0]
    delta = dt / ns
    t_active = substep_couplings(nac0, nac1, v0, v1, ns, interpolation)[:, active, :]  # (N_s, n)
    c_active = amps[:, active]
    pop = np.abs(c_active) ** 2
    keep = pop > DENOMINATOR_FLOOR
    terms = (
        delta
        * 2.0
        * np.real(np.conj(c_active[keep])[:, None] * t_active[keep] * amps[keep])
        / pop[keep][:, None]
    )
    flux = terms.sum(axis=0)  # (n,); zeros when no substep contributes
    h = np.maximum(flux, 0.0)
    h[active] = 0.0
    return _cap(h)


def sharc_probabilities(
    active: int, c_old: np.ndarray, c_new: np.ndarray, propagator: np.ndarray
) -> np.ndarray:
    """Mai, Marquetand and Gonzalez (2015) probabilities from ``active``."""
    n = c_old.shape[0]
    h = np.zeros(n)
    pop_old = abs(c_old[active]) ** 2
    pop_new = abs(c_new[active]) ** 2
    if pop_old <= DENOMINATOR_FLOOR or pop_new >= pop_old:
        return h
    denom = pop_old - np.real(
        c_new[active] * np.conj(propagator[active, active]) * np.conj(c_old[active])
    )
    if denom <= DENOMINATOR_FLOOR:
        return h
    num = np.real(c_new * np.conj(propagator[:, active]) * np.conj(c_old[active]))
    h = (1.0 - pop_new / pop_old) * num / denom
    h = np.maximum(h, 0.0)
    h[active] = 0.0
    return _cap(h)


def choose_state(active: int, probabilities: np.ndarray, xi: float) -> int | None:
    """First j != active (index order) with xi < cumulative probability, else None."""
    cumulative = 0.0
    for j, p in enumerate(probabilities):
        if j == active:
            continue
        cumulative += float(p)
        if xi < cumulative:
            return j
    return None


def hopping_probabilities(
    scheme: str,
    active: int,
    c_old: np.ndarray,
    c_new: np.ndarray,
    propagator: np.ndarray,
    substep_amplitudes: np.ndarray | Sequence[np.ndarray],
    nac0: np.ndarray,
    nac1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    dt: float,
    *,
    interpolation: str = "midpoint",
) -> np.ndarray:
    if scheme == "sharc":
        return sharc_probabilities(active, c_old, c_new, propagator)
    if scheme == "fssh":
        return fssh_probabilities(
            active, substep_amplitudes, nac0, nac1, v0, v1, dt, interpolation=interpolation
        )
    if scheme == "none":
        return np.zeros(c_old.shape[0])
    raise ValueError(f"hopping must be 'sharc', 'fssh' or 'none', got {scheme!r}")
