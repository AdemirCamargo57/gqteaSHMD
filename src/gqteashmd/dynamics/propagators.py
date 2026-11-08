"""Exponential electronic propagator (Method reference 2)."""

from __future__ import annotations

import numpy as np


def coupling_matrix(nacs: np.ndarray, v: np.ndarray) -> np.ndarray:
    """T_jk = v . d_jk, shape (n, n)."""
    t: np.ndarray = np.einsum(
        "jkf,f->jk", np.asarray(nacs, dtype=float), np.asarray(v, dtype=float)
    )
    return t


def interpolated(x0: np.ndarray, x1: np.ndarray, s: float) -> np.ndarray:
    """Linear interpolation between the endpoint values at fraction s."""
    out: np.ndarray = (1.0 - s) * np.asarray(x0) + s * np.asarray(x1)
    return out


def substep_hamiltonian(
    e0: np.ndarray,
    e1: np.ndarray,
    nac0: np.ndarray,
    nac1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (E(s), T(s)) with E, d and v interpolated linearly at fraction s."""
    e = interpolated(e0, e1, s)
    t = coupling_matrix(interpolated(nac0, nac1, s), interpolated(v0, v1, s))
    return e, t


def propagate(
    c: np.ndarray,
    e0: np.ndarray,
    e1: np.ndarray,
    nac0: np.ndarray,
    nac1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    dt: float,
    nsubsteps: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Propagate c over dt. Returns (c(t+dt), P, amplitudes at the start of each substep)."""
    if nsubsteps < 1:
        raise ValueError("nsubsteps must be >= 1")
    c = np.asarray(c, dtype=complex)
    n = c.shape[0]
    delta = dt / nsubsteps
    p = np.eye(n, dtype=complex)
    current = c.copy()
    starts: list[np.ndarray] = []
    for m in range(1, nsubsteps + 1):
        s = (m - 0.5) / nsubsteps
        e, t = substep_hamiltonian(e0, e1, nac0, nac1, v0, v1, s)
        h_eff = np.diag(e).astype(complex) - 1j * t
        w, u = np.linalg.eigh(h_eff)
        p_m = (u * np.exp(-1j * w * delta)) @ u.conj().T
        starts.append(current.copy())
        current = p_m @ current
        p = p_m @ p
    return current, p, starts
