"""Overlaps between many-electron states written as Slater-determinant expansions.

Backend-independent. A backend supplies, for each geometry, its MO coefficients and a
DeterminantExpansion; with s_mo = C_bra^T S_AO(R_bra, R_ket) C_ket (MO overlap across the two
geometries) Löwdin's rule gives <A|B> = det(s_mo[a, b]) for each spin, a and b being the ordered
occupied-orbital lists of the two determinants, and

    S_IJ = sum_kl c^bra_Ik c^ket_Jl det(s_mo[alpha_k, alpha_l]) det(s_mo[beta_k, beta_l]).

Determinants are evaluated once per unique alpha string and once per unique beta string.
The contraction never forms a determinant-pair matrix (see state_overlaps).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

_MAX_BLOCK_ELEMENTS = 20_000_000  # float64 elements per batched-determinant block (~160 MB)
MAX_OVERLAP_BYTES = 2 * 1024**3  # state_overlaps refuses to allocate more than this (estimate)


@dataclass(frozen=True, eq=False)
class DeterminantExpansion:
    """States as linear combinations of a shared list of Slater determinants.

    alpha[k] / beta[k]: MO indices occupied by the alpha / beta electrons of determinant k, in
    the order that defines its sign. coeffs[s, k]: coefficient of determinant k in state s.
    """

    alpha: np.ndarray
    beta: np.ndarray
    coeffs: np.ndarray

    def __init__(self, alpha: ArrayLike, beta: ArrayLike, coeffs: ArrayLike) -> None:
        a = np.asarray(alpha, dtype=np.int64)
        b = np.asarray(beta, dtype=np.int64)
        c = np.asarray(coeffs, dtype=float)
        if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
            raise ValueError("alpha, beta and coeffs must be 2-D arrays")
        if not a.shape[0] == b.shape[0] == c.shape[1]:
            raise ValueError(
                f"inconsistent determinant counts: alpha {a.shape[0]}, beta {b.shape[0]}, "
                f"coeffs {c.shape[1]}"
            )
        object.__setattr__(self, "alpha", a)
        object.__setattr__(self, "beta", b)
        object.__setattr__(self, "coeffs", c)

    @property
    def n_states(self) -> int:
        return int(self.coeffs.shape[0])

    @property
    def n_dets(self) -> int:
        return int(self.coeffs.shape[1])

    def with_signs(self, signs: ArrayLike) -> DeterminantExpansion:
        """The same states, state s multiplied by signs[s] (a phase change)."""
        return DeterminantExpansion(
            self.alpha, self.beta, self.coeffs * np.asarray(signs, dtype=float)[:, None]
        )


def truncate(expansion: DeterminantExpansion, norm_kept: float) -> DeterminantExpansion:
    """Restrict every state to the union of the determinants that are important for some state.

    For each state, the important determinants are the largest-|c| ones (stable order for ties)
    whose squared coefficients reach `norm_kept` of the state's squared norm; for norm_kept = 1.0
    that is every determinant with a nonzero coefficient, with no margin. For norm_kept < 1.0 a
    relative margin of 1e-12 guards only against floating-point cumulative-sum shortfall. The
    union U of these sets over all states is kept, and every state keeps ALL its coefficients on
    U (nothing inside U is zeroed). Determinants outside U are dropped and each state is
    renormalised (this assumes distinct determinants over orthonormal MOs).

    Cross-state coefficients must be kept: a determinant that is negligible for state I but
    important for state J still carries the off-diagonal overlap S_IJ, and those tiny
    first-order terms become the derivative couplings.
    """
    if not 0.0 < norm_kept <= 1.0:
        raise ValueError(f"norm_kept must be in (0, 1], got {norm_kept}")
    if not np.all(np.isfinite(expansion.coeffs)):
        raise ValueError("coefficients must be finite")
    coeffs = expansion.coeffs.copy()
    used = np.zeros(coeffs.shape[1], dtype=bool)
    for s in range(coeffs.shape[0]):
        weights = coeffs[s] ** 2
        total = float(weights.sum())
        if total == 0.0:
            raise ValueError(f"state {s} has only zero coefficients")
        if norm_kept == 1.0:
            used |= coeffs[s] != 0.0
            continue
        order = np.argsort(-weights, kind="stable")
        cumulative = np.cumsum(weights[order])
        n_keep = int(np.searchsorted(cumulative, norm_kept * total * (1.0 - 1e-12))) + 1
        used[order[:n_keep]] = True
    coeffs = coeffs[:, used]
    coeffs /= np.linalg.norm(coeffs, axis=1)[:, None]
    return DeterminantExpansion(expansion.alpha[used], expansion.beta[used], coeffs)


def _unique_rows(occ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if occ.shape[1] == 0:
        return occ[:1], np.zeros(occ.shape[0], dtype=np.int64)
    unique, inverse = np.unique(occ, axis=0, return_inverse=True)
    return unique, inverse.reshape(-1)


def _determinants(s_mo: np.ndarray, bra_rows: np.ndarray, ket_rows: np.ndarray) -> np.ndarray:
    """D[u, v] = det(s_mo[bra_rows[u]][:, ket_rows[v]]), batched in memory-bounded blocks."""
    n_bra, n = bra_rows.shape
    n_ket = ket_rows.shape[0]
    if n == 0:
        return np.ones((n_bra, n_ket))
    out = np.empty((n_bra, n_ket))
    step = max(1, _MAX_BLOCK_ELEMENTS // (n_ket * n * n))
    for start in range(0, n_bra, step):
        rows = bra_rows[start : start + step]
        block = s_mo[rows[:, None, :, None], ket_rows[None, :, None, :]]
        out[start : start + step] = np.linalg.det(block)
    return out


def _scatter(
    expansion: DeterminantExpansion, ia: np.ndarray, ib: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """C[s, u, v] = coefficient of state s on the determinant with alpha string u, beta string v."""
    c = np.zeros((expansion.n_states, *shape))
    for s in range(expansion.n_states):
        np.add.at(c[s], (ia, ib), expansion.coeffs[s])
    return c


def _estimate_bytes(
    n_bra: int, n_ket: int, ua: int, va: int, ub: int, vb: int, n_a: int, n_b: int
) -> int:
    """Bytes held at the peak of state_overlaps: the two determinant tables, the scattered
    coefficient matrices, two work matrices and the largest batched-determinant block."""
    tables = ua * va + ub * vb
    scattered = n_bra * ua * ub + n_ket * va * vb
    work = ua * vb + ua * ub
    block = min(_MAX_BLOCK_ELEMENTS, max(ua * va * n_a * n_a, ub * vb * n_b * n_b))
    return 8 * (tables + scattered + work + block)


def state_overlaps(
    bra: DeterminantExpansion,
    ket: DeterminantExpansion,
    s_mo: ArrayLike,
    s_mo_beta: ArrayLike | None = None,
) -> np.ndarray:
    """S[I, J] = <bra_I | ket_J> given the MO overlap s_mo[p, q] = <phi^bra_p | phi^ket_q>.

    s_mo_beta is the beta-spin MO overlap (default: s_mo, a spin-restricted reference).

    The n_det x n_det matrix of determinant-pair overlaps is never formed. With D_alpha[u, v] the
    determinant for unique alpha strings u (bra) and v (ket), D_beta likewise, and C_I[u, v] the
    coefficient of state I on the determinant (alpha string u, beta string v),
    S_IJ = sum_uv C^bra_I[u, v] (D_alpha C^ket_J D_beta^T)[u, v]. Memory is
    O(N_alpha_unique * N_beta_unique) per state.
    """
    if bra.alpha.shape[1] != ket.alpha.shape[1] or bra.beta.shape[1] != ket.beta.shape[1]:
        raise ValueError(
            f"different electron counts: bra ({bra.alpha.shape[1]} alpha, {bra.beta.shape[1]} "
            f"beta), ket ({ket.alpha.shape[1]} alpha, {ket.beta.shape[1]} beta)"
        )
    s = np.asarray(s_mo, dtype=float)
    s_beta = s if s_mo_beta is None else np.asarray(s_mo_beta, dtype=float)
    ua, ia = _unique_rows(bra.alpha)
    va, ja = _unique_rows(ket.alpha)
    ub, ib = _unique_rows(bra.beta)
    vb, jb = _unique_rows(ket.beta)
    estimate = _estimate_bytes(
        bra.n_states,
        ket.n_states,
        ua.shape[0],
        va.shape[0],
        ub.shape[0],
        vb.shape[0],
        ua.shape[1],
        ub.shape[1],
    )
    if estimate > MAX_OVERLAP_BYTES:
        raise MemoryError(
            f"The wavefunction overlap would exceed the memory limit of "
            f"{MAX_OVERLAP_BYTES / 2**30:.3g} GiB: estimated {estimate / 2**30:.3g} GiB; "
            "lower overlap_norm_kept or the active space"
        )
    d_alpha = _determinants(s, ua, va)
    d_beta = _determinants(s_beta, ub, vb)
    c_bra = _scatter(bra, ia, ib, (ua.shape[0], ub.shape[0])).reshape(bra.n_states, -1)
    c_ket = _scatter(ket, ja, jb, (va.shape[0], vb.shape[0]))
    out = np.empty((bra.n_states, ket.n_states))
    for j in range(ket.n_states):
        out[:, j] = c_bra @ (d_alpha @ c_ket[j] @ d_beta.T).reshape(-1)
    return out
