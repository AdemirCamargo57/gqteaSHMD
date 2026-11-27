"""Effective one-electron spin-orbit operators (spec 3.1): the spin-orbit mean field (SOMF) and the
one-electron Breit-Pauli operator with effective nuclear charges. PySCF is imported inside the
functions only. Atomic units (Eh).

Convention (binding for the SOC sub-project): every function returns ``h`` with shape
(3, nao, nao), complex, ``h[k] = 1j * A[k]`` with ``A[k]`` real antisymmetric (k = x, y, z), so
that the operator is ``H_SO = sum_pq sum_k h[k]_pq S^k_pq`` with
``S^k_pq = sum_{sigma sigma'} a+_{p sigma} (s_k)_{sigma sigma'} a_{q sigma'}`` and
``s = sigma_Pauli / 2``; ``h`` is Hermitian and purely imaginary.

Derivation
----------
The Breit-Pauli spin-orbit Hamiltonian (Hess, Marian, Wahlgren, Gropen, Chem. Phys. Lett. 251,
365 (1996); Neese, J. Chem. Phys. 122, 034107 (2005)), with ``alpha = 1 / c``::

    H_SO = (alpha^2 / 2) [ sum_i (grad_i V_N x p_i) . s_i
                           + sum_{i != j} (grad_i (1 / r_ij) x p_i) . (s_i + 2 s_j) ]

with ``V_N = -sum_A Z_A / |r - R_A|`` the nuclear potential energy (``Z r_A / r_A^3 =
grad(-Z / r_A)`` and ``-r_ij / r_ij^3 = grad_i(1 / r_ij)``), so the nuclei and the other
electrons enter the same way, each through the gradient of its own potential energy.

Integrals (checked numerically, PySCF 2.14, H2CO/STO-3G):

- ``int1e_pnucxp``: ``P[k]_pq = eps_kab <d_a p | V_N | d_b q>`` (equal to the antisymmetrised
  ``int1e_ipnucip`` to 1e-14; ``int1e_nuc`` is negative). Integration by parts (the term with
  ``d_a d_b q`` is symmetric in a, b and drops out) gives ``P[k]_pq = -<p| (grad V_N x grad)_k
  |q>``, and with ``p = -i grad``: ``<p| (grad V_N x p)_k |q> = 1j * P[k]_pq``.
- ``int1e_prinvxp`` with the origin at ``R``: the same with ``V = 1 / |r - R|``. So
  ``P = -sum_A Z_A prinvxp(R_A)`` (equal to 5e-15).
- ``int2e_p1vxp1``: ``G[k]_{pq,rs} = eps_kab (d_a p(1) d_b q(1) | 1/r12 | r(2) s(2))``, the
  operator on electron 1 of the bra pair (p, q); antisymmetric in (p, q), symmetric in (r, s);
  with a very tight normalised s function for (r, s) at ``R`` it equals ``prinvxp(R)`` (1e-8).
  So ``<p(1) r(2)| grad_1(1 / r12) x p_1 |q(1) s(2)>_k = 1j * G[k]_{pq,rs}``.

Mean field. With ``f(1, 2) = g(1, 2) . (s_1 + 2 s_2)`` and the two-body operator written as
``sum <ab|f|cd> a+_a a+_b a_d a_c`` over spin orbitals, the one-body part of its normal ordering
with respect to a closed-shell determinant (``<a+_{p sigma} a_{q sigma'}> = delta_{sigma sigma'}
D_pq / 2``, D the total AO density) is ``sum <ab|f|cd> (g_ac a+_b a_d + g_bd a+_a a_c - g_ad a+_b
a_c - g_bc a+_a a_d)``. Summing the spins:

- ``g_bd a+_a a_c`` (spin-own-orbit Coulomb): spin factor 2, giving ``sum_rs D_rs G_{pq,rs}``;
- ``g_ac a+_b a_d`` (spin-other-orbit Coulomb): ``sum_pq D_pq G_{pq,rs} = 0`` (G antisymmetric
  in p, q, D symmetric);
- each exchange term: spin factor ``1 + 2 = 3`` times ``1 / 2`` from ``D / 2``, i.e. ``-3/2``.

Result (the SOMF operator; Neese 2005)::

    h[k]_mn = 1j (alpha^2 / 2) [ P[k]_mn + J[k]_mn - 3/2 (K1[k]_mn + K2[k]_mn) ]
    J[k]_mn  = sum_rs D_rs G[k]_{mn,rs}      ("ijkl,kl->ij")
    K1[k]_mn = sum_ps D_ps G[k]_{pn,ms}      ("ijkl,il->kj")
    K2[k]_mn = sum_rq D_rq G[k]_{mq,rn}      ("ijkl,jk->il")

``K1 = -K2^T`` (the Hermiticity of the two-electron operator), so the bracket is antisymmetric.
The Coulomb part has the opposite sign of the nuclear part (screening). The overall sign of
``1j`` is a phase convention (complex conjugation of H) that no root sum of squares can see; it is
fixed as above, from the operator itself. ORCA's printed sign of ``<T1|H_SO|S0>`` cannot fix it,
because that sign also follows the arbitrary sign of T1's CI vector (Task 2 report).

Effective nuclear charges: the one-electron term with each ``Z_A`` replaced by ``Z_eff,A``::

    h[k] = 1j (alpha^2 / 2) (-sum_A Z_eff,A prinvxp(R_A)[k])

which equals the one-electron Breit-Pauli term for ``Z_eff = Z``.

Matrix elements (spec 3.2)
--------------------------
With ``h_plus = h_x + i h_y``, ``h_minus = h_x - i h_y`` the operator is
``H_SO = sum_pq [h_z,pq S^z_pq + h_plus,pq S^-_pq / 2 + h_minus,pq S^+_pq / 2]`` with
``S^-_pq = a+_{p beta} a_{q alpha}``, ``S^+_pq = a+_{p alpha} a_{q beta}``,
``S^z_pq = (a+_{p alpha} a_{q alpha} - a+_{p beta} a_{q beta}) / 2``; ``S_- = sum_p S^-_pp``,
``S_+ = S_-^dagger`` and ``|T, M - 1> = S_- |T, M> / sqrt(2)`` (Condon-Shortley). From
``[a+_i a_j, a+_k a_l] = delta_jk a+_i a_l - delta_il a+_k a_j``:
``[S^+_pq, S_-] = 2 S^z_pq``, ``[S^z_pq, S_-] = -S^-_pq``, ``[S^-_pq, S_-] = 0``,
``[S^-_pq, S_+] = -2 S^z_pq``, ``[S^z_pq, S_+] = S^+_pq``. Every element reduces to the Ms = +1
CI vectors (``<S| S_- = 0``, ``S_+ |T, +1> = 0``):

- Singlet-triplet, with the spin-flip density ``D_pq = <S| a+_{p beta} a_{q alpha} |T, +1>``::

      <S|H|T, +1> = (1/2) sum h_plus D,  <S|H|T, 0> = -(1/sqrt 2) sum h_z D,
      <S|H|T, -1> = -(1/2) sum h_minus D           (sum = sum_pq of the elementwise product)

- Triplet-triplet, with ``Z_pq = <T, +1| S^z_pq |T', +1>``: within the two triplets every
  spin-tensor density is Z times the spin-1 matrix (``<M|S^z_pq|M'> = Z_pq (J_z)_MM'``,
  ``<M|S^-_pq|M'> = Z_pq (J_-)_MM'``, ``<M|S^+_pq|M'> = Z_pq (J_+)_MM'``), so with
  ``z = sum h_z Z``, ``p = sum h_plus Z``, ``m = sum h_minus Z`` the 3 x 3 block (rows M, columns
  M', order -1, 0, +1) is::

      [[-z,          p / sqrt 2,  0         ],
       [ m / sqrt 2,  0,          p / sqrt 2],
       [ 0,           m / sqrt 2, z         ]]

The doubly occupied core contributes nothing (a spin flip or S^z on a closed shell vanishes); it
enters only through the SOMF density. Only active-orbital indices appear.

TDA/TDDFT (CIS-like states, X only)
-----------------------------------
Occupied MOs i, j, virtual MOs a, b (``mf.mo_coeff``, all MOs, h transformed to the MO basis),
``|i->a, sigma> = a+_{a sigma} a_{i sigma} |Phi0>``, ``X[i, a]`` of PySCF's ``td.xy[k][0]``
(nocc, nvir). PySCF's restricted X has ``sum X^2 = 1/2`` for TDA and ``sum X^2 - sum Y^2 = 1/2``
for TDDFT; Y is dropped and every state's X is rescaled to ``x = X / sqrt(2 sum X^2)``, so
``sum x^2 = 1/2`` (the overlaps' approximation). The states (the layout of
``pyscf_response.response_expansion`` and ``response_triplet_expansion``), all normalised::

    S0 = |Phi0>,  S_m = sum x_ia (|i->a, alpha> + |i->a, beta>),
    |T_n, 0> = sum x_ia (|i->a, alpha> - |i->a, beta>)

From ``S_- a+_{a alpha} a_{i beta} |Phi0> = |i->a, beta> - |i->a, alpha>`` and Condon-Shortley,
``|T_n, +1> = -sqrt 2 sum x_ia a+_{a alpha} a_{i beta} |Phi0>``. Evaluating the densities by
the anticommutation rules (``a+_{p beta} a_{q alpha}`` on ``a+_{a alpha} a_{i beta} |Phi0>``
lands on a single excitation only for q = a, p virtual, giving ``|i->p, beta>``, and for q = j
occupied, p = i, giving ``-|j->a, alpha>``; on the same ket ``a+_{p alpha} a_{q alpha}`` gives
``+a+_{b alpha} a_{i beta}|Phi0>`` for (p, q) = (b, a) and ``a+_{p beta} a_{q beta}`` gives
``-a+_{a alpha} a_{j beta}|Phi0>`` for (p, q) = (i, j), i != j. On the occupied diagonal
(p = q = k) the alpha part gives ``S I`` (``S = sum x^T x^T'``) and the beta part gives
``S I - (x^T' x^T^T)[k, k]`` (no beta electron in the hole i = k); only the ``S I`` parts cancel
in ``S^z_pq``, and the surviving ``(x^T' x^T^T)[k, k]`` is the i = j case of the Z[i, j] formula
below), with ``x^S`` the singlet's and ``x^T``, ``x^T'`` the triplets' amplitudes (bra first)::

    S0-T:   D[i, a]  = -sqrt 2 x^T[i, a]                       (occupied row, virtual column)
    S_m-T:  D[b, a]  = -sqrt 2 (x^S^T x^T)[b, a],   D[i, j] = +sqrt 2 (x^T x^S^T)[i, j]
    T-T':   Z[b, a]  =  (x^T^T x^T')[b, a],          Z[i, j] = (x^T' x^T^T)[i, j]

and the components follow from D and Z exactly as for SA-CASSCF above. So
``<S0|H|T_n, 0> = sum_ia h_z,ia x_ia`` and ``<S_m|H|T_n, 0> = sum_ab h_z,ba (x^S^T x^T)_ba -
sum_ij h_z,ij (x^T x^S^T)_ij`` (the virtual-virtual and the occupied-occupied contractions).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from gqteashmd.core.states import StateLabel
from gqteashmd.core.units import SPEED_OF_LIGHT_AU
from gqteashmd.dynamics.spin import expanded_states
from gqteashmd.electronic.pyscf_errors import wrap_failure
from gqteashmd.errors import ConfigError

# Effective nuclear charges of the one-electron spin-orbit operator: the values ORCA 6.1.1 uses
# (measured by atomic ratios of its SOCType 1 to its one-electron Breit-Pauli elements, SOC
# sub-project 2, Task 1: H 1.000, C 3.600, O 5.600, S 13.600, each to about 1e-4). ORCA attributes
# its table to S. Koseki, M. W. Schmidt, M. S. Gordon, J. Phys. Chem. 96, 10768 (1992) and later
# papers; these four values were not checked against the papers' tables.
# Only elements whose value is established are listed; any other element is refused.
KOSEKI_ZEFF: dict[str, float] = {"H": 1.0, "C": 3.6, "O": 5.6, "S": 13.6}

SOC_OPERATORS = ("somf", "effective_charge")
_PREFACTOR = 0.5 / SPEED_OF_LIGHT_AU**2  # alpha^2 / 2


def _as_operator(spatial: np.ndarray) -> np.ndarray:
    """``1j * (alpha^2 / 2) * A`` as a complex (3, nao, nao) array."""
    return 1j * _PREFACTOR * np.asarray(spatial, dtype=float)


def _nuclear_term(mol: Any) -> np.ndarray:
    try:
        return np.asarray(mol.intor("int1e_pnucxp", comp=3))
    except Exception as exc:
        raise wrap_failure("Spin-orbit one-electron integrals", exc) from exc


def one_electron_operator(mol: Any) -> np.ndarray:
    """The one-electron Breit-Pauli spin-orbit operator with the true nuclear charges."""
    return _as_operator(_nuclear_term(mol))


def effective_charge_operator(mol: Any, zeff: Mapping[str, float] | None = None) -> np.ndarray:
    """The one-electron operator with effective nuclear charges (default `KOSEKI_ZEFF`)."""
    table = KOSEKI_ZEFF if zeff is None else zeff
    if any(mol.atom_charge(i) == 0 for i in range(mol.natm)):
        raise ConfigError("ghost atoms are not supported by the effective-charge operator")
    symbols = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
    missing = list(dict.fromkeys(s for s in symbols if s not in table))
    if missing:
        raise ConfigError(
            f"no established effective nuclear charge for {', '.join(missing)} "
            f"(available: {', '.join(table)}); use soc_operator = 'somf'"
        )
    spatial = np.zeros((3, mol.nao, mol.nao))
    try:
        for i, symbol in enumerate(symbols):
            with mol.with_rinv_origin(mol.atom_coord(i)):
                spatial -= float(table[symbol]) * np.asarray(mol.intor("int1e_prinvxp", comp=3))
    except Exception as exc:
        raise wrap_failure("Spin-orbit effective-charge integrals", exc) from exc
    return _as_operator(spatial)


def somf_operator(mol: Any, dm: np.ndarray, *, two_electron: bool = True) -> np.ndarray:
    """The SOMF operator of the total AO density ``dm`` (the module docstring has the formula)."""
    spatial = _nuclear_term(mol)
    if two_electron:
        dm = np.asarray(dm, dtype=float)
        try:
            from pyscf.scf import jk

            coulomb, exchange_1, exchange_2 = jk.get_jk(
                mol,
                [dm, dm, dm],
                ["ijkl,kl->ij", "ijkl,il->kj", "ijkl,jk->il"],
                intor="int2e_p1vxp1",
                comp=3,
            )
        except Exception as exc:
            raise wrap_failure("Spin-orbit mean field", exc) from exc
        spatial = spatial + coulomb - 1.5 * (exchange_1 + exchange_2)
    return _as_operator(spatial)


def soc_operator(mol: Any, dm: np.ndarray, kind: str) -> np.ndarray:
    """The operator named by ``[electronic] soc_operator``: ``"somf"`` or ``"effective_charge"``."""
    if kind == "somf":
        return somf_operator(mol, dm)
    if kind == "effective_charge":
        return effective_charge_operator(mol)
    raise ValueError(f"unknown spin-orbit operator {kind!r} (expected one of {SOC_OPERATORS})")


@dataclass(frozen=True)
class SocBlocks:
    """Spin-orbit blocks (Eh) of one geometry, the M index 0, 1, 2 = Ms -1, 0, +1.

    st: (n_singlets, n_triplets, 3) = <S_I|H_SO|T_J, M>; tt: (n_triplets, n_triplets, 3, 3) =
    <T_J, M|H_SO|T_J', M'>. Singlet-singlet elements are zero and not stored.
    """

    st: np.ndarray
    tt: np.ndarray


def _st_components(
    h_z: np.ndarray, h_plus: np.ndarray, h_minus: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """<S|H|T, M> for M = -1, 0, +1 from the spin-flip density D = <S|a+_{p beta} a_{q alpha}
    |T, +1> (module docstring)."""
    return np.array(
        [
            -0.5 * np.sum(h_minus * d),
            -np.sum(h_z * d) / np.sqrt(2.0),
            0.5 * np.sum(h_plus * d),
        ]
    )


def _tt_block(
    h_z: np.ndarray, h_plus: np.ndarray, h_minus: np.ndarray, z_density: np.ndarray
) -> np.ndarray:
    """The 3 x 3 block <T, M|H|T', M'> from Z = <T, +1|S^z_pq|T', +1> (module docstring)."""
    z = np.sum(h_z * z_density)
    plus = np.sum(h_plus * z_density) / np.sqrt(2.0)
    minus = np.sum(h_minus * z_density) / np.sqrt(2.0)
    return np.array([[-z, plus, 0.0], [minus, 0.0, plus], [0.0, minus, z]])


def _ci_matrix(ci: Any, ncas: int, nelec: tuple[int, int]) -> np.ndarray:
    from pyscf.fci import cistring

    shape = (cistring.num_strings(ncas, nelec[0]), cistring.num_strings(ncas, nelec[1]))
    return np.asarray(ci, dtype=float).reshape(shape)


def _excitations(
    ket: np.ndarray, ncas: int, nelec: tuple[int, int], destroy: str, create: str
) -> np.ndarray:
    """(ncas, ncas, ...) images ``a+_{p create} a_{q destroy} |ket>`` for every (p, q)."""
    from pyscf.fci import addons

    mid = (nelec[0] - (destroy == "a"), nelec[1] - (destroy == "b"))
    destroy_op = addons.des_a if destroy == "a" else addons.des_b
    create_op = addons.cre_a if create == "a" else addons.cre_b
    images = []
    for p in range(ncas):
        row = []
        for q in range(ncas):
            removed = destroy_op(ket, ncas, nelec, q)
            row.append(create_op(removed, ncas, mid, p))
        images.append(row)
    return np.array(images)


def casscf_soc_blocks(mc: Any, states: Sequence[StateLabel], h_ao: np.ndarray) -> SocBlocks:
    """The SA-CASSCF spin-orbit blocks of `states` (mc.ci in their order: singlets with
    mc.nelecas, triplets at Ms = +1 with one more alpha and one fewer beta active electron) for
    the operator `h_ao` (3, nao, nao) of `soc_operator`; the formulas are in the module docstring.
    """
    singlets = [ci for s, ci in zip(states, mc.ci, strict=True) if s.multiplicity == 1]
    triplets = [ci for s, ci in zip(states, mc.ci, strict=True) if s.multiplicity == 3]
    n_s, n_t = len(singlets), len(triplets)
    st = np.zeros((n_s, n_t, 3), dtype=complex)
    tt = np.zeros((n_t, n_t, 3, 3), dtype=complex)
    if not n_t:
        return SocBlocks(st, tt)
    ncore, ncas = int(mc.ncore), int(mc.ncas)
    n_alpha, n_beta = (int(n) for n in mc.nelecas)
    s_nelec = (n_alpha, n_beta)
    t_nelec = (n_alpha + 1, n_beta - 1)
    mo = np.asarray(mc.mo_coeff)[:, ncore : ncore + ncas]
    h = np.einsum("pi,kpq,qj->kij", mo, np.asarray(h_ao, dtype=complex), mo)
    h_z, h_plus, h_minus = h[2], h[0] + 1j * h[1], h[0] - 1j * h[1]
    try:
        s_ci = [_ci_matrix(ci, ncas, s_nelec) for ci in singlets]
        t_ci = [_ci_matrix(ci, ncas, t_nelec) for ci in triplets]
        for j, ket in enumerate(t_ci):
            flips = _excitations(ket, ncas, t_nelec, "a", "b")
            for i, bra in enumerate(s_ci):
                d = np.einsum("ab,pqab->pq", bra, flips)
                st[i, j] = _st_components(h_z, h_plus, h_minus, d)
            alpha = _excitations(ket, ncas, t_nelec, "a", "a")
            beta = _excitations(ket, ncas, t_nelec, "b", "b") if t_nelec[1] else None
            for jb, bra in enumerate(t_ci):
                z_density = 0.5 * np.einsum("ab,pqab->pq", bra, alpha)
                if beta is not None:
                    z_density -= 0.5 * np.einsum("ab,pqab->pq", bra, beta)
                tt[jb, j] = _tt_block(h_z, h_plus, h_minus, z_density)
    except Exception as exc:
        raise wrap_failure("The SA-CASSCF spin-orbit matrix elements", exc) from exc
    return SocBlocks(st, tt)


def _normalised_amplitudes(td: Any, index: int, shape: tuple[int, int]) -> np.ndarray:
    """X of root `index` (0-based), which must have `shape` (nocc, nvir), scaled so that
    sum X^2 = 1/2 (TDDFT's Y is dropped; for TDA PySCF's X already has this norm). A wrong shape
    (which would broadcast silently into the density blocks) or a zero norm is a ValueError."""
    x = np.asarray(td.xy[index][0], dtype=float)
    if x.shape != shape:
        raise ValueError(f"response amplitudes of root {index} have shape {x.shape}, not {shape}")
    norm_squared = float(np.sum(x * x))
    if not norm_squared > 0.0:
        raise ValueError(f"response amplitudes of root {index} have zero norm")
    scale: float = 1.0 / np.sqrt(2.0 * norm_squared)
    return scale * x


def response_soc_blocks(
    mf: Any, td: Any, td_triplet: Any, states: Sequence[StateLabel], h_ao: np.ndarray
) -> SocBlocks:
    """The TDA/TDDFT spin-orbit blocks of `states` (S0 the SCF determinant, S_m root m - 1 of
    `td`, T_n root n - 1 of `td_triplet`) for the operator `h_ao` (3, nao, nao) of
    `soc_operator`; X only, renormalised per state. The formulas are in the module docstring.
    """
    singlets = [s for s in states if s.multiplicity == 1]
    triplets = [s for s in states if s.multiplicity == 3]
    n_s, n_t = len(singlets), len(triplets)
    st = np.zeros((n_s, n_t, 3), dtype=complex)
    tt = np.zeros((n_t, n_t, 3, 3), dtype=complex)
    if not n_t:
        return SocBlocks(st, tt)
    excited = [str(s) for s in singlets if s.index > 0]
    if excited and td is None:
        raise ValueError(
            f"the spin-orbit elements of {', '.join(excited)} need the singlet response object (td)"
        )
    try:
        mo = np.asarray(mf.mo_coeff)
        nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0))
        n_mo = mo.shape[1]
        shape = (nocc, n_mo - nocc)
        h = np.einsum("pi,kpq,qj->kij", mo, np.asarray(h_ao, dtype=complex), mo)
        h_z, h_plus, h_minus = h[2], h[0] + 1j * h[1], h[0] - 1j * h[1]
        occ, vir = slice(0, nocc), slice(nocc, n_mo)
        xs = [
            None if s.index == 0 else _normalised_amplitudes(td, s.index - 1, shape)
            for s in singlets
        ]
        xt = [_normalised_amplitudes(td_triplet, t.index - 1, shape) for t in triplets]
        for j, x_t in enumerate(xt):
            for i, x_s in enumerate(xs):
                d = np.zeros((n_mo, n_mo))
                if x_s is None:
                    d[occ, vir] = -np.sqrt(2.0) * x_t
                else:
                    d[vir, vir] = -np.sqrt(2.0) * (x_s.T @ x_t)
                    d[occ, occ] = np.sqrt(2.0) * (x_t @ x_s.T)
                st[i, j] = _st_components(h_z, h_plus, h_minus, d)
            for jb, x_b in enumerate(xt):
                z_density = np.zeros((n_mo, n_mo))
                z_density[vir, vir] = x_b.T @ x_t
                z_density[occ, occ] = x_t @ x_b.T
                tt[jb, j] = _tt_block(h_z, h_plus, h_minus, z_density)
    except Exception as exc:
        raise wrap_failure("The TDA/TDDFT spin-orbit matrix elements", exc) from exc
    return SocBlocks(st, tt)


# A spin-orbit Hamiltonian further than this from Hermitian (Eh) is an internal error (spec 3.3).
HERMITIAN_TOLERANCE = 1e-12


def hamiltonian_labels(n_singlets: int, n_triplets: int) -> tuple[str, ...]:
    """The labels of the spin-expanded states in the canonical order of `dynamics.spin`."""
    return tuple(s.label() for s in expanded_states((n_singlets, 0, n_triplets)))


def assemble_hamiltonian(
    energies_s: np.ndarray,
    energies_t: np.ndarray,
    blocks: SocBlocks,
    phases_s: np.ndarray,
    phases_t: np.ndarray,
) -> np.ndarray:
    """The complex spin-expanded Hamiltonian (Eh) in the canonical order S0 ... S(n-1), then
    T1 ... Tm with Ms = -1, then 0, then +1 (`dynamics.spin`).

    The diagonal holds the spin-free energies (each triplet energy at its three components); the
    S-T blocks hold ``p_I p_J st``, the T-S blocks their conjugates, the T-T blocks
    ``p_J p_J' tt`` (the three components of a triplet share its phase); S-S elements are zero.
    Hermiticity is checked, not imposed: a T-T block further than `HERMITIAN_TOLERANCE` from
    Hermitian, or a spin-orbit element on the diagonal (zero for real spin-free states), raises
    AssertionError (an internal error).
    """
    e_s = np.asarray(energies_s, dtype=float)
    e_t = np.asarray(energies_t, dtype=float)
    n_s, n_t = e_s.size, e_t.size
    st, tt = np.asarray(blocks.st), np.asarray(blocks.tt)
    if st.shape != (n_s, n_t, 3) or tt.shape != (n_t, n_t, 3, 3):
        raise ValueError(
            f"spin-orbit blocks of shape {st.shape} and {tt.shape} do not fit {n_s} singlets "
            f"and {n_t} triplets"
        )
    p_s = np.asarray(phases_s, dtype=float)
    p_t = np.asarray(phases_t, dtype=float)
    if p_s.shape != (n_s,) or p_t.shape != (n_t,):
        raise ValueError(
            f"phases of shape {p_s.shape} and {p_t.shape} do not fit {n_s} singlets and "
            f"{n_t} triplets"
        )
    # (J, M) -> n_s + M n_t + J: M outermost, the state index innermost
    st_flat = (
        (p_s[:, None, None] * p_t[None, :, None] * st).transpose(0, 2, 1).reshape(n_s, 3 * n_t)
    )
    tt_phased = p_t[:, None, None, None] * p_t[None, :, None, None] * tt
    tt_flat = tt_phased.transpose(2, 0, 3, 1).reshape(3 * n_t, 3 * n_t)
    diagonal_soc = float(np.max(np.abs(np.diag(tt_flat)), initial=0.0))
    if diagonal_soc > HERMITIAN_TOLERANCE:
        raise AssertionError(
            f"the spin-orbit block has an element {diagonal_soc:.3e} Eh on its diagonal "
            f"(tolerance {HERMITIAN_TOLERANCE:g})"
        )
    deviation = float(np.max(np.abs(tt_flat - tt_flat.conj().T), initial=0.0))
    if deviation > HERMITIAN_TOLERANCE:
        raise AssertionError(
            f"the triplet-triplet spin-orbit block is not Hermitian: deviation {deviation:.3e} Eh "
            f"(tolerance {HERMITIAN_TOLERANCE:g})"
        )
    n = n_s + 3 * n_t
    h = np.zeros((n, n), dtype=complex)
    h[:n_s, n_s:] = st_flat
    h[n_s:, :n_s] = st_flat.conj().T
    h[n_s:, n_s:] = tt_flat
    # the diagonal is the spin-free energies; the T-T diagonal was checked to be below 1e-12 Eh
    h[np.diag_indices(n)] = np.concatenate([e_s, np.tile(e_t, 3)])
    return h
