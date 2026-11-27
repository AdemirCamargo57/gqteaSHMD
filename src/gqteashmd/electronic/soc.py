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
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from gqteashmd.core.units import SPEED_OF_LIGHT_AU
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
