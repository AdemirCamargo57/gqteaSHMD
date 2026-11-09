"""Exponential electronic propagator (Method reference 2).

Two substep interpolations (``interpolation``):

- ``"midpoint"`` (default): substep m = 1..N_s at s_m = (m - 1/2)/N_s, with E, d and v each
  interpolated linearly, T(s) = v(s) . d(s).
- ``"endpoint"``: s_m = m/N_s, with E and the coupling T = v . d each interpolated linearly
  between T(t) = v(t) . d(t) and T(t + dt) = v(t + dt) . d(t + dt). This is the scheme of the
  public SHARC 4.1 manual, Sec. 8.33.1, eqs. (8.191)-(8.194). It is first order in the substep
  length; the midpoint rule is second order. With v varying over the step the continuous limits
  differ slightly (T linear in s against v(s) . d(s) quadratic in s).

All substeps of one nuclear step are built at once: E and H_eff are stacked as (N_s, n, n)
arrays, diagonalised by one batched ``eigh`` and exponentiated together. Only the ordered
product P = P_{N_s} ... P_1 and the start-of-substep amplitudes are formed in a short
sequential loop.
"""

from __future__ import annotations

import numpy as np

INTERPOLATIONS = ("midpoint", "endpoint")


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


def midpoint_fractions(nsubsteps: int) -> np.ndarray:
    """s_m = (m - 1/2) / N_s for m = 1..N_s, shape (N_s,)."""
    s: np.ndarray = (np.arange(1, nsubsteps + 1, dtype=float) - 0.5) / nsubsteps
    return s


def midpoint_couplings(
    nac0: np.ndarray, nac1: np.ndarray, v0: np.ndarray, v1: np.ndarray, nsubsteps: int
) -> np.ndarray:
    """T(s_m) = v(s_m) . d(s_m) at every substep midpoint, shape (N_s, n, n)."""
    s = midpoint_fractions(nsubsteps)
    r = 1.0 - s
    nac0 = np.asarray(nac0, dtype=float)
    nac1 = np.asarray(nac1, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    nacs = r[:, None, None, None] * nac0 + s[:, None, None, None] * nac1  # (N_s, n, n, f)
    vel = r[:, None] * v0 + s[:, None] * v1  # (N_s, f)
    t: np.ndarray = np.einsum("mjkf,mf->mjk", nacs, vel)
    return t


def _check_interpolation(interpolation: str) -> None:
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"interpolation must be one of {INTERPOLATIONS}, got {interpolation!r}")


def substep_fractions(nsubsteps: int, interpolation: str = "midpoint") -> np.ndarray:
    """The fractions s_m of the step at which substep m = 1..N_s is evaluated, shape (N_s,)."""
    _check_interpolation(interpolation)
    if interpolation == "midpoint":
        return midpoint_fractions(nsubsteps)
    s: np.ndarray = np.arange(1, nsubsteps + 1, dtype=float) / nsubsteps
    return s


def substep_couplings(
    nac0: np.ndarray,
    nac1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    nsubsteps: int,
    interpolation: str = "midpoint",
) -> np.ndarray:
    """T at every substep for ``interpolation`` (module docstring), shape (N_s, n, n)."""
    _check_interpolation(interpolation)
    if interpolation == "midpoint":
        return midpoint_couplings(nac0, nac1, v0, v1, nsubsteps)
    s = substep_fractions(nsubsteps, interpolation)
    t0, t1 = coupling_matrix(nac0, v0), coupling_matrix(nac1, v1)
    t: np.ndarray = (1.0 - s)[:, None, None] * t0 + s[:, None, None] * t1
    return t


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
    *,
    interpolation: str = "midpoint",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate c over dt. Returns (c(t+dt), P, amplitudes at the start of each substep).

    ``interpolation`` is ``"midpoint"`` or ``"endpoint"`` (module docstring).

    The start-of-substep amplitudes are an (N_s, n) complex array; row m - 1 is the amplitude
    vector before substep m (row 0 is ``c``). H_eff is Hermitized explicitly before ``eigh``,
    which reads only one triangle. NumPy only; BLAS threading is irrelevant for these tiny
    (n x n) problems.
    """
    if nsubsteps < 1:
        raise ValueError("nsubsteps must be >= 1")
    _check_interpolation(interpolation)
    c = np.asarray(c, dtype=complex)
    n = c.shape[0]
    delta = dt / nsubsteps
    s = substep_fractions(nsubsteps, interpolation)
    energies = (1.0 - s)[:, None] * np.asarray(e0, dtype=float) + s[:, None] * np.asarray(
        e1, dtype=float
    )  # (N_s, n)
    h = -1j * substep_couplings(nac0, nac1, v0, v1, nsubsteps, interpolation)  # (N_s, n, n)
    h.reshape(nsubsteps, n * n)[:, :: n + 1] += energies  # the diagonals, as a view
    h = 0.5 * (h + np.conj(h).swapaxes(-1, -2))
    w, u = np.linalg.eigh(h)
    factors = (u * np.exp(-1j * w * delta)[:, None, :]) @ np.conj(u).swapaxes(-1, -2)
    # Ordered products P_m ... P_1; the amplitude before substep m + 1 is (P_m ... P_1) c.
    cumulative = np.empty_like(factors)
    p = factors[0]
    cumulative[0] = p
    for m in range(1, nsubsteps):
        p = factors[m] @ p
        cumulative[m] = p
    starts = np.empty((nsubsteps, n), dtype=complex)
    starts[0] = c
    starts[1:] = cumulative[:-1] @ c
    return p @ c, p.copy(), starts
