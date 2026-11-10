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

`propagate_diagonal` (P7a-2, Method reference 7) propagates in the diagonal basis with the
full complex MCH Hamiltonian and either NAC couplings (``"nacdr"``) or local diabatization from
the snapshot overlaps (``"overlap"``, Granucci, Persico & Toniolo, J. Chem. Phys. 114, 10608
(2001)). Without a diagonal basis and with ``"nacdr"`` it calls `propagate` unchanged.
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


def substep_exponentials(h: np.ndarray, delta: float) -> np.ndarray:
    """exp(-i H_m delta) for a stack of substep matrices H_m, shape (N_s, n, n).

    Each H_m is Hermitised explicitly first (``eigh`` reads only one triangle), then all of
    them are diagonalised by one batched ``eigh``. ``h`` is not modified.
    """
    h = 0.5 * (h + np.conj(h).swapaxes(-1, -2))
    w, u = np.linalg.eigh(h)
    factors: np.ndarray = (u * np.exp(-1j * w * delta)[:, None, :]) @ np.conj(u).swapaxes(-1, -2)
    return factors


def ordered_product(factors: np.ndarray) -> np.ndarray:
    """P_{N_s} ... P_1 (later substeps on the left) of a stack of factors (N_s, n, n)."""
    p: np.ndarray = factors[0].copy()
    for m in range(1, factors.shape[0]):
        p = factors[m] @ p
    return p


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
    diag = np.arange(n)
    h[:, diag, diag] += energies  # the diagonals (fancy indexing: no reliance on a view)
    factors = substep_exponentials(h, delta)
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


COUPLINGS = ("nacdr", "overlap")


def _check_square(name: str, x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x)
    if x.shape != (n, n):
        raise ValueError(f"{name} must have shape ({n}, {n}); got {x.shape}")
    return x


def propagate_diagonal(
    c_diag: np.ndarray,
    u0: np.ndarray | None,
    u1: np.ndarray | None,
    h0: np.ndarray,
    h1: np.ndarray,
    nac0: np.ndarray,
    nac1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    dt: float,
    nsubsteps: int,
    *,
    coupling: str,
    overlaps: np.ndarray | None,
    interpolation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(c^diag(t+dt), R^diag, substep start amplitudes or None), Method reference 7.

    ``h0``/``h1`` are the complex MCH Hamiltonians at t and t + dt, ``u0``/``u1`` the diagonal
    bases (H^MCH = U diag(E^diag) U^dagger; ``None`` is the identity in MCH order) and
    ``nac0``/``nac1`` the (masked) MCH NACs.

    - Fast path (u0 = u1 = None, coupling ``"nacdr"``): the P7a-1 `propagate` on Re diag(h),
      returning its outputs unchanged (bit-identical, including the substep start amplitudes).
    - ``"nacdr"``: R^MCH = prod_m exp(-i (H(s_m) - i T(s_m)) delta), H(s) = (1 - s) h0 + s h1
      with the full complex Hamiltonian and T the P7a-1 coupling for ``interpolation``.
    - ``"overlap"`` (local diabatization; NACs ignored): with S = ``overlaps``,
      S[b, a] = <psi_b(t)|psi_a(t+dt)>, H_tra = S h1 S^dagger is h1 written in the MCH basis
      at t, H~(s) = (1 - s) h0 + s H_tra, and R^MCH = S^dagger prod_m exp(-i H~(s_m) delta).
      Why S^dagger: a state |Psi> = sum_b c_b |psi_b(t)> has the t + dt components
      <psi_a(t+dt)|Psi> = sum_b conj(S[b, a]) c_b = (S^dagger c)_a; likewise
      <psi_b(t)|H|psi_b'(t)> = sum_aa' S[b, a] h1[a, a'] conj(S[b', a']) = (S h1 S^dagger).
      ``coupling = "overlap"`` without overlaps raises ``ValueError``.

    R^diag = U1^dagger R^MCH U0 and c' = R^diag c^diag. Products have later substeps on the
    left; the substep matrices are Hermitised and diagonalised by one batched ``eigh``. The
    substep start amplitudes are returned only on the fast path (only ``"fssh"`` needs them).
    """
    if coupling not in COUPLINGS:
        raise ValueError(f"coupling must be one of {COUPLINGS}, got {coupling!r}")
    if coupling == "overlap" and overlaps is None:
        raise ValueError("coupling = 'overlap' needs the snapshot overlaps at t + dt")
    if u0 is None and u1 is None and coupling == "nacdr":
        e0 = np.real(np.diagonal(np.asarray(h0)))
        e1 = np.real(np.diagonal(np.asarray(h1)))
        c_new, p, starts = propagate(
            c_diag, e0, e1, nac0, nac1, v0, v1, dt, nsubsteps, interpolation=interpolation
        )
        return c_new, p, starts
    if nsubsteps < 1:
        raise ValueError("nsubsteps must be >= 1")
    _check_interpolation(interpolation)
    c = np.asarray(c_diag, dtype=complex)
    n = c.shape[0]
    h0 = _check_square("h0", np.asarray(h0, dtype=complex), n)
    h1 = _check_square("h1", np.asarray(h1, dtype=complex), n)
    delta = dt / nsubsteps
    s = substep_fractions(nsubsteps, interpolation)
    r = 1.0 - s
    s_ovl: np.ndarray | None = None
    h_end = h1
    if coupling == "overlap":
        s_ovl = _check_square("overlaps", np.asarray(overlaps, dtype=complex), n)
        h_end = s_ovl @ h1 @ np.conj(s_ovl).T  # H_tra: h1 in the MCH basis at t
    h = r[:, None, None] * h0 + s[:, None, None] * h_end  # (N_s, n, n)
    if s_ovl is None:  # "nacdr": the P7a-1 coupling T(s_m) of this interpolation
        h = h - 1j * substep_couplings(nac0, nac1, v0, v1, nsubsteps, interpolation)
    r_mch = ordered_product(substep_exponentials(h, delta))
    if s_ovl is not None:  # back from the t basis to the t + dt basis
        r_mch = np.conj(s_ovl).T @ r_mch
    if u0 is not None:
        r_mch = r_mch @ _check_square("u0", np.asarray(u0, dtype=complex), n)
    if u1 is not None:
        r_mch = np.conj(_check_square("u1", np.asarray(u1, dtype=complex), n)).T @ r_mch
    return r_mch @ c, r_mch, None
