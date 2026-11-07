"""Overlaps between many-electron states written as Slater-determinant expansions.

Backend-independent. A backend supplies, for each geometry, its MO coefficients and a
DeterminantExpansion; with s_mo = C_bra^T S_AO(R_bra, R_ket) C_ket (MO overlap across the two
geometries) Löwdin's rule gives <A|B> = det(s_mo[a, b]) for each spin, a and b being the ordered
occupied-orbital lists of the two determinants, and

    S_IJ = sum_kl c^bra_Ik c^ket_Jl det(s_mo[alpha_k, alpha_l]) det(s_mo[beta_k, beta_l]).

Determinants are evaluated once per unique alpha string and once per unique beta string.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

_MAX_BLOCK_ELEMENTS = 20_000_000  # float64 elements per batched-determinant block (~160 MB)


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
    """Keep, per state, the largest coefficients until their squared sum reaches `norm_kept` of
    the state's squared norm; drop determinants no state keeps; renormalise every state."""
    if not 0.0 < norm_kept <= 1.0:
        raise ValueError(f"norm_kept must be in (0, 1], got {norm_kept}")
    coeffs = expansion.coeffs.copy()
    for s in range(coeffs.shape[0]):
        weights = coeffs[s] ** 2
        total = float(weights.sum())
        if total == 0.0:
            raise ValueError(f"state {s} has only zero coefficients")
        order = np.argsort(-weights, kind="stable")
        cumulative = np.cumsum(weights[order])
        n_keep = int(np.searchsorted(cumulative, norm_kept * total * (1.0 - 1e-12))) + 1
        coeffs[s, order[n_keep:]] = 0.0
    used = np.any(coeffs != 0.0, axis=0)
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


def state_overlaps(
    bra: DeterminantExpansion, ket: DeterminantExpansion, s_mo: ArrayLike
) -> np.ndarray:
    """S[I, J] = <bra_I | ket_J> given the MO overlap s_mo[p, q] = <phi^bra_p | phi^ket_q>."""
    if bra.alpha.shape[1] != ket.alpha.shape[1] or bra.beta.shape[1] != ket.beta.shape[1]:
        raise ValueError(
            f"different electron counts: bra ({bra.alpha.shape[1]} alpha, {bra.beta.shape[1]} "
            f"beta), ket ({ket.alpha.shape[1]} alpha, {ket.beta.shape[1]} beta)"
        )
    s = np.asarray(s_mo, dtype=float)
    ua, ia = _unique_rows(bra.alpha)
    va, ja = _unique_rows(ket.alpha)
    ub, ib = _unique_rows(bra.beta)
    vb, jb = _unique_rows(ket.beta)
    d = _determinants(s, ua, va)[np.ix_(ia, ja)] * _determinants(s, ub, vb)[np.ix_(ib, jb)]
    return np.asarray(bra.coeffs @ d @ ket.coeffs.T)
