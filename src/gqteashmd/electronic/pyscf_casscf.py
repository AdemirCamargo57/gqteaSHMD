"""SA-CASSCF with PySCF: retry ladder, gradients, NACs, transition dipoles, determinants.

PySCF is imported inside the functions only. All quantities are in atomic units.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from gqteashmd.config.schema import ElectronicSection
from gqteashmd.core.states import StateLabel
from gqteashmd.electronic.pyscf_errors import wrap_failure
from gqteashmd.errors import ElectronicStructureError
from gqteashmd.overlaps.determinants import DeterminantExpansion

SPIN_TOLERANCE = 1e-3  # |<S^2>| above this means a root is not a singlet

# (name, start from the previous geometry's orbitals, spin-penalty shift / Eh, macro-cycle factor)
_LADDER: tuple[tuple[str, bool, float, int], ...] = (
    ("previous orbitals", True, 0.0, 1),
    ("SCF orbitals", False, 0.0, 1),
    ("SCF orbitals + spin penalty", False, 0.5, 3),
)


@dataclass(eq=False)
class CasscfOutcome:
    mc: Any
    rung: str
    energies: dict[StateLabel, float]
    spin_squares: dict[StateLabel, float]


def _converged(mc: Any) -> bool:
    """Convergence test of a CASSCF run (a seam for tests)."""
    return bool(mc.converged)


def check_active_space(mol: Any, settings: ElectronicSection) -> None:
    cas = settings.casscf
    assert cas is not None  # guaranteed by ElectronicSection validation
    core_electrons = mol.nelectron - cas.nelecas
    if core_electrons < 0 or core_electrons % 2:
        raise ElectronicStructureError(
            f"The CASSCF active space ({cas.nelecas} electrons) does not fit the molecule's "
            f"{mol.nelectron} electrons."
        )
    if core_electrons // 2 + cas.ncas > mol.nao:
        raise ElectronicStructureError(
            f"The CASSCF active space needs {core_electrons // 2 + cas.ncas} orbitals, but the "
            f"basis '{settings.basis}' has only {mol.nao}."
        )


def run_casscf(
    mol: Any,
    mf: Any,
    settings: ElectronicSection,
    states: tuple[StateLabel, ...],
    previous: tuple[np.ndarray, Any] | None,
) -> CasscfOutcome:
    """Equal-weight SA-CASSCF over `states` (singlets), trying the rungs of _LADDER in order.

    `previous` = (MO coefficients, Mole) of the previous geometry, projected onto this geometry
    for the first rung.
    """
    from pyscf import fci, mcscf

    cas = settings.casscf
    assert cas is not None
    check_active_space(mol, settings)
    n_states = len(states)
    problems: list[str] = []
    for name, use_previous, spin_shift, macro_factor in _LADDER:
        if use_previous and previous is None:
            continue
        mc = mcscf.CASSCF(mf, cas.ncas, cas.nelecas)
        mc.fcisolver = fci.solver(mol, singlet=True)  # singlet-only CI (excludes triplets)
        mc = mc.state_average_([1.0 / n_states] * n_states)
        if spin_shift:
            mc.fix_spin_(ss=0, shift=spin_shift)  # also penalise quintets and higher spins
        mc.conv_tol = cas.conv_tol
        mc.conv_tol_grad = cas.conv_tol_grad
        mc.max_cycle_macro = mc.max_cycle_macro * macro_factor
        mo0 = mf.mo_coeff
        if use_previous:
            assert previous is not None
            previous_mo, previous_mol = previous
            try:
                # orbitals of another geometry are not orthonormal here: project them first
                mo0 = mcscf.project_init_guess(mc, previous_mo, previous_mol)
            except Exception as exc:
                raise wrap_failure("Projecting the previous orbitals", exc) from exc
        try:
            mc.kernel(mo_coeff=mo0)
        except Exception as exc:
            # an exception (unlike non-convergence) signals a broken setup: abort the ladder
            raise wrap_failure(f"CASSCF ({name})", exc) from exc
        if not _converged(mc):
            problems.append(f"{name}: not converged")
            continue
        spins = {
            label: float(fci.spin_op.spin_square0(ci, mc.ncas, mc.nelecas)[0])
            for label, ci in zip(states, mc.ci, strict=True)
        }
        wrong = [
            f"{label} <S^2> = {s:.4f}" for label, s in spins.items() if abs(s) > SPIN_TOLERANCE
        ]
        if wrong:
            problems.append(f"{name}: not all roots are singlets ({', '.join(wrong)})")
            continue
        energies = {label: float(e) for label, e in zip(states, mc.e_states, strict=True)}
        return CasscfOutcome(mc, name, energies, spins)
    raise ElectronicStructureError(
        f"SA-CASSCF failed on every rung (conv_tol = {cas.conv_tol}, conv_tol_grad = "
        f"{cas.conv_tol_grad}): {'; '.join(problems)}."
    )


def casscf_gradients(mc: Any, labels: Iterable[StateLabel]) -> dict[StateLabel, np.ndarray]:
    gradients: dict[StateLabel, np.ndarray] = {}
    wanted = sorted(labels)
    if not wanted:
        return gradients
    solver = mc.nuc_grad_method()
    for label in wanted:
        try:
            g = solver.kernel(state=label.index)
        except Exception as exc:
            raise wrap_failure(f"The SA-CASSCF gradient of {label}", exc) from exc
        if not getattr(solver, "converged", True):
            raise ElectronicStructureError(f"The SA-CASSCF gradient of {label} did not converge.")
        gradients[label] = np.asarray(g, dtype=float)
    return gradients


def casscf_nacs(
    mc: Any, pairs: Iterable[tuple[StateLabel, StateLabel]]
) -> dict[tuple[StateLabel, StateLabel], np.ndarray]:
    """d_IJ = <Psi_I|dPsi_J/dR> (1/bohr), including the CSF term (PySCF's default convention)."""
    nacs: dict[tuple[StateLabel, StateLabel], np.ndarray] = {}
    wanted = sorted(pairs)
    if not wanted:
        return nacs
    solver = mc.nac_method()
    for a, b in wanted:
        try:
            d = solver.kernel(state=(a.index, b.index))
        except Exception as exc:
            raise wrap_failure(f"The SA-CASSCF coupling {a}-{b}", exc) from exc
        if not getattr(solver, "converged", True):
            raise ElectronicStructureError(f"The SA-CASSCF coupling {a}-{b} did not converge.")
        nacs[(a, b)] = np.asarray(d, dtype=float)
    return nacs


def casscf_dipoles(
    mol: Any, mc: Any, states: tuple[StateLabel, ...]
) -> dict[StateLabel, np.ndarray]:
    """S0 -> state transition dipoles mu = -sum_pq r_pq gamma^{0I}_pq over active orbitals."""
    dipoles: dict[StateLabel, np.ndarray] = {}
    try:
        r_ints = mol.intor("int1e_r", comp=3)
        mo_cas = mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas]
        for label in states[1:]:
            t_mo = mc.fcisolver.trans_rdm1(mc.ci[0], mc.ci[label.index], mc.ncas, mc.nelecas)
            t_ao = mo_cas @ t_mo @ mo_cas.T
            dipoles[label] = -np.einsum("xij,ji->x", r_ints, t_ao)
    except Exception as exc:
        raise wrap_failure("The transition dipole calculation", exc) from exc
    return dipoles


def casscf_expansion(mc: Any) -> DeterminantExpansion:
    """SA-CASSCF states as determinants over mc.mo_coeff: core orbitals, then the active orbitals
    of each alpha/beta FCI string in ascending order (alpha string = slow CI index)."""
    from pyscf.fci import cistring

    ncore, ncas = int(mc.ncore), int(mc.ncas)
    n_alpha, n_beta = (int(n) for n in mc.nelecas)
    core = list(range(ncore))

    def occupations(n_el: int) -> list[list[int]]:
        return [
            core + [ncore + k for k in range(ncas) if int(s) >> k & 1]
            for s in cistring.make_strings(range(ncas), n_el)
        ]

    occ_alpha, occ_beta = occupations(n_alpha), occupations(n_beta)
    alpha = [a for a in occ_alpha for _ in occ_beta]
    beta = [b for _ in occ_alpha for b in occ_beta]
    coeffs = np.array([np.asarray(ci, dtype=float).reshape(-1) for ci in mc.ci])
    return DeterminantExpansion(alpha, beta, coeffs)
