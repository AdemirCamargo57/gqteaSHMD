"""Hopping probabilities (Method reference 3-5) and the hop decision."""

from __future__ import annotations

import numpy as np

from gqteashmd.dynamics.propagators import substep_hamiltonian

DENOMINATOR_FLOOR = 1e-14


def _cap(h: np.ndarray) -> np.ndarray:
    total = float(h.sum())
    return h / total if total > 1.0 else h


def fssh_probabilities(
    active: int,
    substep_amplitudes: list[np.ndarray],
    nac0: np.ndarray,
    nac1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Tully flux from ``active``, T taken at substep midpoints like the propagator.

    ``substep_amplitudes`` are the amplitudes at the start of each substep (from ``propagate``).
    Returns (n,) with 0 at ``active``; capped so the sum is <= 1 (scaled proportionally).
    """
    ns = len(substep_amplitudes)
    n = substep_amplitudes[0].shape[0]
    delta = dt / ns
    flux = np.zeros(n)
    zeros = np.zeros(n)
    for m, c in enumerate(substep_amplitudes, start=1):
        pop = abs(c[active]) ** 2
        if pop <= DENOMINATOR_FLOOR:
            continue
        _, t = substep_hamiltonian(zeros, zeros, nac0, nac1, v0, v1, (m - 0.5) / ns)
        flux += delta * 2.0 * np.real(np.conj(c[active]) * t[active, :] * c) / pop
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
    substep_amplitudes: list[np.ndarray],
    nac0: np.ndarray,
    nac1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    dt: float,
) -> np.ndarray:
    if scheme == "sharc":
        return sharc_probabilities(active, c_old, c_new, propagator)
    if scheme == "fssh":
        return fssh_probabilities(active, substep_amplitudes, nac0, nac1, v0, v1, dt)
    if scheme == "none":
        return np.zeros(c_old.shape[0])
    raise ValueError(f"hopping must be 'sharc', 'fssh' or 'none', got {scheme!r}")
