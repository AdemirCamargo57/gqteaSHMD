"""Test-only brute-force reference for spin-orbit matrix elements (spec 3.4). Not collected.

States are explicit vectors over spin-orbital determinants of all MOs; matrix elements of a
one-body operator are evaluated by the Slater-Condon rules with explicit fermionic signs. It
shares no code with `gqteashmd.electronic.soc`.

Conventions
-----------
- Spin orbitals: alpha MO p is spin orbital p, beta MO p is spin orbital n_mo + p ("all alpha,
  then all beta").
- A determinant with occupied spin orbitals s_1 < s_2 < ... < s_N is
  ``a+_{s_1} a+_{s_2} ... a+_{s_N} |vac>``; it is stored as (alpha MO tuple, beta MO tuple),
  both ascending.
- The operator is ``sum_k sum_pq h_k[p, q] sum_{sigma sigma'} a+_{p sigma} (s_k)_{sigma sigma'}
  a_{q sigma'}`` with ``s = sigma_Pauli / 2`` (sigma = alpha, beta = up, down).

PySCF CI vectors (the meaning of their coefficients, read from PySCF 2.14:
``fci.cistring.cre_sign`` gives ``a+_p`` the sign (-1)^(occupied orbitals above p), and
``fci.addons.cre_b``/``des_b`` add (-1)^(n_alpha)): coefficient C[I_a, I_b] multiplies
``A(I_a) B(I_b) |vac>`` where A and B are the alpha and beta active creators in *descending*
orbital order, alpha block to the left. The doubly occupied core is embedded as a block of
pairs ``prod_c a+_{c alpha} a+_{c beta}`` to the left (an even product, which commutes with
everything). Bringing ``core-pairs A_desc B_desc`` to the canonical order above gives the factor
``(-1)^(ncore (ncore - 1) / 2 + na (na - 1) / 2 + nb (nb - 1) / 2 + na ncore)`` (pairs regrouped
into alpha core then beta core; the alpha and the beta active strings reversed; the alpha active
string moved past the beta core), with na, nb the active alpha and beta electrons.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gqteashmd.core.states import StateLabel

Determinant = tuple[tuple[int, ...], tuple[int, ...]]

_S = (
    0.5 * np.array([[0, 1], [1, 0]], dtype=complex),
    0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex),
    0.5 * np.array([[1, 0], [0, -1]], dtype=complex),
)


@dataclass
class ReferenceState:
    """A many-electron state: coefficient of every (alpha occ, beta occ) determinant."""

    n_mo: int
    coeffs: dict[Determinant, complex] = field(default_factory=dict)

    def norm(self) -> float:
        return float(np.sqrt(sum(abs(c) ** 2 for c in self.coeffs.values())))


def _to_spin_orbitals(det: Determinant, n_mo: int) -> tuple[int, ...]:
    alpha, beta = det
    return tuple(alpha) + tuple(n_mo + p for p in beta)


def _to_determinant(spin_orbitals: tuple[int, ...], n_mo: int) -> Determinant:
    return (
        tuple(s for s in spin_orbitals if s < n_mo),
        tuple(s - n_mo for s in spin_orbitals if s >= n_mo),
    )


def _annihilate(v: int, occ: tuple[int, ...]) -> tuple[int, tuple[int, ...]] | None:
    if v not in occ:
        return None
    k = occ.index(v)  # creators to the left of a+_v
    return (-1) ** k, occ[:k] + occ[k + 1 :]


def _create(u: int, occ: tuple[int, ...]) -> tuple[int, tuple[int, ...]] | None:
    if u in occ:
        return None
    k = sum(1 for s in occ if s < u)
    return (-1) ** k, occ[:k] + (u,) + occ[k:]


def _excite(u: int, v: int, occ: tuple[int, ...]) -> tuple[int, tuple[int, ...]] | None:
    """a+_u a_v |occ> as (sign, occupation) or None."""
    first = _annihilate(v, occ)
    if first is None:
        return None
    second = _create(u, first[1])
    if second is None:
        return None
    return first[0] * second[0], second[1]


def _apply_one_body(state: ReferenceState, matrix: np.ndarray) -> ReferenceState:
    """sum_uv matrix[u, v] a+_u a_v |state> over spin orbitals."""
    n = state.n_mo
    out = ReferenceState(n)
    for det, c in state.coeffs.items():
        occ = _to_spin_orbitals(det, n)
        for v in occ:
            for u in range(2 * n):
                if matrix[u, v] == 0:
                    continue
                hit = _excite(u, v, occ)
                if hit is None:
                    continue
                sign, new = hit
                key = _to_determinant(new, n)
                out.coeffs[key] = out.coeffs.get(key, 0.0) + sign * matrix[u, v] * c
    return out


def _spin_orbital_matrix(h_mo: np.ndarray) -> np.ndarray:
    """The (2n, 2n) spin-orbital matrix of sum_k h_k s_k (alpha block first)."""
    n = h_mo.shape[1]
    m = np.zeros((2 * n, 2 * n), dtype=complex)
    for k in range(3):
        for a in range(2):
            for b in range(2):
                m[a * n : (a + 1) * n, b * n : (b + 1) * n] += _S[k][a, b] * h_mo[k]
    return m


def lowering(state: ReferenceState) -> ReferenceState:
    """S_minus |state> with S_minus = sum_p a+_{p beta} a_{p alpha} over all MOs."""
    n = state.n_mo
    m = np.zeros((2 * n, 2 * n))
    for p in range(n):
        m[n + p, p] = 1.0
    return _apply_one_body(state, m)


def _scaled(state: ReferenceState, factor: float) -> ReferenceState:
    return ReferenceState(state.n_mo, {k: factor * c for k, c in state.coeffs.items()})


def _from_ci(
    ci: Any, n_mo: int, ncore: int, ncas: int, n_alpha: int, n_beta: int
) -> ReferenceState:
    from pyscf.fci import cistring

    def strings(n_el: int) -> list[tuple[int, ...]]:
        return [
            tuple(ncore + k for k in range(ncas) if int(s) >> k & 1)
            for s in cistring.make_strings(range(ncas), n_el)
        ]

    alpha, beta = strings(n_alpha), strings(n_beta)
    c = np.asarray(ci, dtype=float).reshape(len(alpha), len(beta))
    exponent = (
        ncore * (ncore - 1) // 2
        + n_alpha * (n_alpha - 1) // 2
        + n_beta * (n_beta - 1) // 2
        + n_alpha * ncore
    )
    sign = (-1) ** exponent
    core = tuple(range(ncore))
    state = ReferenceState(n_mo)
    for i, a in enumerate(alpha):
        for j, b in enumerate(beta):
            if c[i, j] != 0.0:
                state.coeffs[(core + a, core + b)] = sign * complex(c[i, j])
    return state


def casscf_reference_states(
    mc: Any, states: Sequence[StateLabel]
) -> dict[tuple[StateLabel, int], ReferenceState]:
    """The SA-CASSCF states as explicit determinant vectors, keyed (label, Ms).

    Singlets (Ms = 0) from their CI vectors; triplets: the Ms = +1 CI vector (one more alpha and
    one fewer beta active electron), then |T, M - 1> = S_minus |T, M> / sqrt(2) twice.
    """
    n_mo = int(mc.mo_coeff.shape[1])
    ncore, ncas = int(mc.ncore), int(mc.ncas)
    n_alpha, n_beta = (int(n) for n in mc.nelecas)
    out: dict[tuple[StateLabel, int], ReferenceState] = {}
    for label, ci in zip(states, mc.ci, strict=True):
        if label.multiplicity == 1:
            out[(label, 0)] = _from_ci(ci, n_mo, ncore, ncas, n_alpha, n_beta)
            continue
        plus = _from_ci(ci, n_mo, ncore, ncas, n_alpha + 1, n_beta - 1)
        zero = _scaled(lowering(plus), 1 / np.sqrt(2))
        minus = _scaled(lowering(zero), 1 / np.sqrt(2))
        out[(label, 1)], out[(label, 0)], out[(label, -1)] = plus, zero, minus
    return out


def one_body_element(bra: ReferenceState, ket: ReferenceState, h_mo: np.ndarray) -> complex:
    """<bra| sum_k sum_pq h_k[p, q] S^k_pq |ket> by the Slater-Condon rules (h_mo (3, n, n))."""
    image = _apply_one_body(ket, _spin_orbital_matrix(np.asarray(h_mo)))
    return complex(sum(np.conj(c) * image.coeffs.get(det, 0.0) for det, c in bra.coeffs.items()))


def overlap(bra: ReferenceState, ket: ReferenceState) -> complex:
    return complex(sum(np.conj(c) * ket.coeffs.get(det, 0.0) for det, c in bra.coeffs.items()))
