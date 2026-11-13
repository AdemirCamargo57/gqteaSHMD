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


@dataclass(frozen=True)
class Rung:
    """One attempt of the SA-CASSCF ladder.

    start: "previous" (the previous geometry's orbitals, skipped without them), "scf" (the SCF
    orbitals) or "previous_or_scf". conv_tol/conv_tol_grad: None = the [electronic.casscf]
    value; a number loosens it (max(user, rung)), never tightens it.
    """

    name: str
    start: str
    spin_shift: float
    macro_factor: int
    conv_tol: float | None = None
    conv_tol_grad: float | None = None


LADDER: tuple[Rung, ...] = (
    Rung("previous orbitals", "previous", 0.0, 1),
    Rung("SCF orbitals", "scf", 0.0, 1),
    Rung("SCF orbitals + spin penalty", "scf", 0.5, 3),
)
# P9a plan 2: off unless a caller enables them; no stage does (ruling P9a2-7: the probe's
# accuracy conditions were not met)
EXTRA_RUNGS: tuple[Rung, ...] = (
    Rung("previous orbitals, more iterations", "previous", 0.0, 4),
    Rung("loose tolerances", "previous_or_scf", 0.0, 3, 1e-8, 1e-4),
)
EXTRA_RUNG_NAMES: tuple[str, ...] = tuple(r.name for r in EXTRA_RUNGS)


def check_extra_rungs(names: Iterable[str]) -> tuple[str, ...]:
    chosen = tuple(names)
    unknown = sorted(set(chosen) - set(EXTRA_RUNG_NAMES))
    if unknown:
        raise ValueError(
            f"unknown CASSCF extra rung(s) {unknown}; known: {', '.join(EXTRA_RUNG_NAMES)}"
        )
    return chosen


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
    *,
    extra_rungs: tuple[str, ...] = (),
) -> CasscfOutcome:
    """Equal-weight SA-CASSCF over `states` (singlets): the rungs of LADDER, then `extra_rungs`.

    `previous` = (MO coefficients, Mole) of the previous geometry, projected onto this geometry
    for the first rung.
    """
    from pyscf import fci, mcscf

    cas = settings.casscf
    assert cas is not None
    check_active_space(mol, settings)
    n_states = len(states)
    problems: list[str] = []
    enabled = set(check_extra_rungs(extra_rungs))
    rungs = LADDER + tuple(r for r in EXTRA_RUNGS if r.name in enabled)
    for rung in rungs:
        name = rung.name
        use_previous = rung.start == "previous" or (
            rung.start == "previous_or_scf" and previous is not None
        )
        if rung.start == "previous" and previous is None:
            continue
        mc = mcscf.CASSCF(mf, cas.ncas, cas.nelecas)
        mc.fcisolver = fci.solver(mol, singlet=True)  # singlet-only CI (excludes triplets)
        mc = mc.state_average_([1.0 / n_states] * n_states)
        if rung.spin_shift:
            mc.fix_spin_(ss=0, shift=rung.spin_shift)  # also penalise quintets and higher spins
        mc.conv_tol = cas.conv_tol if rung.conv_tol is None else max(cas.conv_tol, rung.conv_tol)
        mc.conv_tol_grad = (
            cas.conv_tol_grad
            if rung.conv_tol_grad is None
            else max(cas.conv_tol_grad, rung.conv_tol_grad)
        )
        mc.max_cycle_macro = mc.max_cycle_macro * rung.macro_factor
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
