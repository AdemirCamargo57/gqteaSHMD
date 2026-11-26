"""SA-CASSCF with PySCF: retry ladder, gradients, NACs, transition dipoles, determinants.

PySCF is imported inside the functions only. All quantities are in atomic units.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from gqteashmd.config.schema import ElectronicSection
from gqteashmd.core.states import StateLabel
from gqteashmd.electronic.pyscf_errors import wrap_failure
from gqteashmd.errors import ElectronicStructureError
from gqteashmd.overlaps.determinants import DeterminantExpansion

SPIN_TOLERANCE = 1e-3  # |<S^2> - target| above this means a root has the wrong spin

REFERENCE_RUNG = "reference orbitals"
REFERENCE_TIE_EH = 1e-8  # SA energies within this of the lowest are ties: the first guess wins
REFERENCE_OVERLAP_WARNING = 0.9  # warn below this active-orbital singular value (no effect)

Orbitals = tuple[np.ndarray, Any]  # (MO coefficients, the Mole they belong to)


def guess_active_sets(n_occupied: int, n_mo: int, ncas: int, nelecas: int) -> list[tuple[int, ...]]:
    """0-based active-orbital choices around the Fermi level, PySCF's default first.

    The occupied window is the nelecas/2 + 1 highest doubly occupied orbitals, the virtual
    window the ncas - nelecas/2 + 1 lowest virtual ones, both clipped to the orbitals that
    exist. The other choices follow in lexicographic order of their sorted indices.
    """
    n_occ_act = nelecas // 2
    n_virt_act = ncas - n_occ_act
    occ_window = range(max(0, n_occupied - n_occ_act - 1), n_occupied)
    virt_window = range(n_occupied, min(n_mo, n_occupied + n_virt_act + 1))
    default = tuple(range(n_occupied - n_occ_act, n_occupied + n_virt_act))
    choices = {
        tuple(sorted((*o, *v)))
        for o in itertools.combinations(occ_window, n_occ_act)
        for v in itertools.combinations(virt_window, n_virt_act)
    }
    choices.discard(default)
    return [default, *sorted(choices)]


def pick_lowest(sa_energies: Sequence[float]) -> int:
    """Index of the first SA energy within REFERENCE_TIE_EH of the lowest."""
    if not sa_energies:
        raise ValueError("no candidate SA energies")
    lowest = min(sa_energies)
    return next(i for i, e in enumerate(sa_energies) if e <= lowest + REFERENCE_TIE_EH)


@dataclass(frozen=True)
class Rung:
    """One attempt of the SA-CASSCF ladder.

    start: "previous" (the previous geometry's orbitals), "reference" (the backend's reference
    orbitals), both skipped when absent, or "scf" (the SCF orbitals).
    """

    name: str
    start: str
    spin_shift: float
    macro_factor: int


LADDER: tuple[Rung, ...] = (
    Rung("previous orbitals", "previous", 0.0, 1),
    Rung(REFERENCE_RUNG, "reference", 0.0, 1),
    Rung("SCF orbitals", "scf", 0.0, 1),
    Rung("SCF orbitals + spin penalty", "scf", 0.5, 3),
)


@dataclass(frozen=True, eq=False)
class Reference:
    """The reference solution at the reference geometry (see find_reference)."""

    mo_coeff: np.ndarray
    mol: Any
    guess: tuple[int, ...]
    sa_energy: float
    n_guesses: int  # size of the guess set
    n_solved: int  # guesses actually solved (1 when the default counts)
    n_counted: int  # solved guesses that converged with every root of the expected spin

    def meta(self) -> dict[str, Any]:
        return {
            "guess": list(self.guess),
            "sa_energy_Eh": self.sa_energy,
            "n_guesses": self.n_guesses,
            "n_solved": self.n_solved,
            "n_counted": self.n_counted,
        }


@dataclass(eq=False)
class CasscfOutcome:
    mc: Any
    rung: str
    energies: dict[StateLabel, float]
    spin_squares: dict[StateLabel, float]
    reference_active_min_sv: float | None = None


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


def state_counts(states: Sequence[StateLabel]) -> tuple[int, int]:
    """(singlets, triplets) among `states`."""
    return (
        sum(1 for s in states if s.multiplicity == 1),
        sum(1 for s in states if s.multiplicity == 3),
    )


def mixed_solver(
    mol: Any, mf: Any, settings: ElectronicSection, n_singlets: int, n_triplets: int, rung: Rung
) -> Any:
    """Equal-weight SA-CASSCF over singlets and triplets with one orbital set (spec 3.2).

    The roots are the singlets, then the triplets (Ms = +1 in the active space); mc.ci and
    mc.e_states follow that order.
    """
    from pyscf import fci, mcscf

    cas = settings.casscf
    assert cas is not None
    singlet = fci.solver(mol, singlet=True)
    singlet.nroots = n_singlets
    singlet.spin = 0
    triplet = fci.solver(mol, singlet=False)
    triplet.nroots = n_triplets
    triplet.spin = 2  # Ms = +1 in the active space
    if rung.spin_shift:
        singlet = fci.addons.fix_spin_(singlet, shift=rung.spin_shift, ss=0)
        triplet = fci.addons.fix_spin_(triplet, shift=rung.spin_shift, ss=2)
    n = n_singlets + n_triplets
    mc = mcscf.CASSCF(mf, cas.ncas, cas.nelecas)
    mc = mcscf.state_average_mix_(mc, [singlet, triplet], [1.0 / n] * n)
    mc.conv_tol = cas.conv_tol
    mc.conv_tol_grad = cas.conv_tol_grad
    mc.max_cycle_macro = mc.max_cycle_macro * rung.macro_factor
    return mc


def _solver(
    mol: Any, mf: Any, settings: ElectronicSection, states: Sequence[StateLabel], rung: Rung
) -> Any:
    n_singlets, n_triplets = state_counts(states)
    if n_triplets:
        return mixed_solver(mol, mf, settings, n_singlets, n_triplets, rung)
    from pyscf import fci, mcscf

    cas = settings.casscf
    assert cas is not None
    n_states = n_singlets
    mc = mcscf.CASSCF(mf, cas.ncas, cas.nelecas)
    mc.fcisolver = fci.solver(mol, singlet=True)  # singlet-only CI (excludes triplets)
    mc = mc.state_average_([1.0 / n_states] * n_states)
    if rung.spin_shift:
        mc.fix_spin_(ss=0, shift=rung.spin_shift)  # also penalise quintets and higher spins
    mc.conv_tol = cas.conv_tol
    mc.conv_tol_grad = cas.conv_tol_grad
    mc.max_cycle_macro = mc.max_cycle_macro * rung.macro_factor
    return mc


def _root_nelec(mc: Any, label: StateLabel) -> tuple[int, int]:
    """Active (alpha, beta) electrons of a root: Ms = 0 for singlets, Ms = +1 for triplets."""
    n_alpha, n_beta = (int(n) for n in mc.nelecas)
    if label.multiplicity == 3:
        return n_alpha + 1, n_beta - 1
    return n_alpha, n_beta


def _spins(mc: Any, states: Sequence[StateLabel]) -> tuple[dict[StateLabel, float], list[str]]:
    """<S^2> of each root, and the roots whose <S^2> is off their multiplicity's value."""
    from pyscf import fci

    spins = {
        label: float(fci.spin_op.spin_square0(ci, mc.ncas, _root_nelec(mc, label))[0])
        for label, ci in zip(states, mc.ci, strict=True)
    }
    wrong = [
        f"{label} <S^2> = {s:.4f}"
        for label, s in spins.items()
        if abs(s - (0.0 if label.multiplicity == 1 else 2.0)) > SPIN_TOLERANCE
    ]
    return spins, wrong


def _spin_problem(states: Sequence[StateLabel], wrong: list[str]) -> str:
    what = "have the expected spin" if state_counts(states)[1] else "are singlets"
    return f"not all roots {what} ({', '.join(wrong)})"


def active_min_singular_value(
    mol: Any, mo_start: np.ndarray, mo_final: np.ndarray, ncore: int, ncas: int
) -> float:
    """Smallest singular value of C_start^T S C_final over the active orbitals."""
    s_ao = mol.intor("int1e_ovlp")
    a = np.asarray(mo_start)[:, ncore : ncore + ncas]
    b = np.asarray(mo_final)[:, ncore : ncore + ncas]
    return float(np.linalg.svd(a.T @ s_ao @ b, compute_uv=False).min())


def find_reference(
    mol: Any, mf: Any, settings: ElectronicSection, states: tuple[StateLabel, ...]
) -> tuple[Reference | None, list[str]]:
    """The reference: PySCF's default active space if it counts, else the lowest-SA other guess.

    The guesses of guess_active_sets are solved fresh, the default first. A guess counts if it
    converges and every root has the spin of its multiplicity. When the default counts it is the
    reference and no other guess is solved (owner decision 2026-11-14: a lower-SA non-default
    solution can have a different character, e.g. ethylene SA2-CAS(2,2)/STO-3G guess [6, 8], S0
    1.27 eV higher, whose projected starts split between solutions at displaced geometries).
    Otherwise the other guesses are solved and the lowest SA energy among those that count wins
    (pick_lowest).
    Returns (None, problems) when none counts. An exception (not a non-convergence) propagates,
    wrapped.
    """
    from pyscf import mcscf

    cas = settings.casscf
    assert cas is not None
    check_active_space(mol, settings)
    guesses = guess_active_sets(
        mol.nelectron // 2, int(np.asarray(mf.mo_coeff).shape[1]), cas.ncas, cas.nelecas
    )
    plain = Rung("reference guess", "scf", 0.0, 1)
    problems: list[str] = []
    counted: list[tuple[tuple[int, ...], float, np.ndarray]] = []
    n_solved = 0
    for guess in guesses:
        if n_solved == 1 and counted:  # the default (first) guess counts: it is the reference
            break
        n_solved += 1
        mc = _solver(mol, mf, settings, states, plain)
        try:
            mo0 = mcscf.sort_mo(mc, mf.mo_coeff, [i + 1 for i in guess])  # 1-based
            mc.kernel(mo_coeff=mo0)
        except Exception as exc:
            raise wrap_failure(f"CASSCF (reference guess {list(guess)})", exc) from exc
        if not _converged(mc):
            problems.append(f"guess {list(guess)}: not converged")
            continue
        _, wrong = _spins(mc, states)
        if wrong:
            problems.append(f"guess {list(guess)}: {_spin_problem(states, wrong)}")
            continue
        counted.append((guess, float(np.mean(mc.e_states)), np.asarray(mc.mo_coeff)))
    if not counted:
        return None, problems
    best = pick_lowest([sa for _, sa, _ in counted])
    guess, sa, mo = counted[best]
    return Reference(mo, mol, guess, sa, len(guesses), n_solved, len(counted)), problems


def run_casscf(
    mol: Any,
    mf: Any,
    settings: ElectronicSection,
    states: tuple[StateLabel, ...],
    previous: Orbitals | None,
    *,
    reference: Callable[[], Orbitals | None] | None = None,
) -> CasscfOutcome:
    """Equal-weight SA-CASSCF over `states` (singlets, then triplets): the rungs of LADDER.

    `previous` = (MO coefficients, Mole) of the previous geometry; `reference` is called (at
    most once) when the reference rung is reached and returns the reference orbitals or None.
    Both are projected onto this geometry.
    """
    from pyscf import mcscf

    cas = settings.casscf
    assert cas is not None
    check_active_space(mol, settings)
    problems: list[str] = []
    for rung in LADDER:
        name = rung.name
        start: Orbitals | None = None
        if rung.start == "previous":
            if previous is None:
                continue
            start = previous
        elif rung.start == "reference":
            start = None if reference is None else reference()
            if start is None:
                continue
        mc = _solver(mol, mf, settings, states, rung)
        mo0 = mf.mo_coeff
        if start is not None:
            start_mo, start_mol = start
            what = "previous" if rung.start == "previous" else "reference"
            try:
                # orbitals of another geometry are not orthonormal here: project them first
                mo0 = mcscf.project_init_guess(mc, start_mo, start_mol)
            except Exception as exc:
                raise wrap_failure(f"Projecting the {what} orbitals", exc) from exc
        try:
            mc.kernel(mo_coeff=mo0)
        except Exception as exc:
            # an exception (unlike non-convergence) signals a broken setup: abort the ladder
            raise wrap_failure(f"CASSCF ({name})", exc) from exc
        if not _converged(mc):
            problems.append(f"{name}: not converged")
            continue
        spins, wrong = _spins(mc, states)
        if wrong:
            problems.append(f"{name}: {_spin_problem(states, wrong)}")
            continue
        energies = {label: float(e) for label, e in zip(states, mc.e_states, strict=True)}
        min_sv = None
        if rung.start == "reference":
            min_sv = active_min_singular_value(mol, mo0, mc.mo_coeff, mc.ncore, mc.ncas)
        return CasscfOutcome(mc, name, energies, spins, min_sv)
    raise ElectronicStructureError(
        f"SA-CASSCF failed on every rung (conv_tol = {cas.conv_tol}, conv_tol_grad = "
        f"{cas.conv_tol_grad}): {'; '.join(problems)}."
    )


def casscf_gradients(
    mc: Any, labels: Iterable[StateLabel], states: Sequence[StateLabel]
) -> dict[StateLabel, np.ndarray]:
    """SA-CASSCF gradients (Eh/bohr) of `labels`, each root addressed by its position in states."""
    gradients: dict[StateLabel, np.ndarray] = {}
    wanted = sorted(labels)
    if not wanted:
        return gradients
    solver = mc.nuc_grad_method()
    for label in wanted:
        try:
            g = solver.kernel(state=states.index(label))
        except Exception as exc:
            raise wrap_failure(f"The SA-CASSCF gradient of {label}", exc) from exc
        if not getattr(solver, "converged", True):
            raise ElectronicStructureError(f"The SA-CASSCF gradient of {label} did not converge.")
        gradients[label] = np.asarray(g, dtype=float)
    return gradients


def casscf_nacs(
    mc: Any,
    pairs: Iterable[tuple[StateLabel, StateLabel]],
    states: Sequence[StateLabel],
    natom: int,
) -> dict[tuple[StateLabel, StateLabel], np.ndarray]:
    """d_IJ = <Psi_I|dPsi_J/dR> (1/bohr), including the CSF term (PySCF's default convention).

    A singlet-triplet pair is zero by spin symmetry, without a PySCF call; the PySCF solver is
    built only when a pair of one multiplicity is requested.
    """
    nacs: dict[tuple[StateLabel, StateLabel], np.ndarray] = {}
    solver = None
    for a, b in sorted(pairs):
        if a.multiplicity != b.multiplicity:
            nacs[(a, b)] = np.zeros((natom, 3))
            continue
        if solver is None:
            solver = mc.nac_method()
        try:
            d = solver.kernel(state=(states.index(a), states.index(b)))
        except Exception as exc:
            raise wrap_failure(f"The SA-CASSCF coupling {a}-{b}", exc) from exc
        if not getattr(solver, "converged", True):
            raise ElectronicStructureError(f"The SA-CASSCF coupling {a}-{b} did not converge.")
        nacs[(a, b)] = np.asarray(d, dtype=float)
    return nacs


def casscf_dipoles(mol: Any, mc: Any, states: Sequence[StateLabel]) -> dict[StateLabel, np.ndarray]:
    """S0 -> state transition dipoles mu = -sum_pq r_pq gamma^{0I}_pq over active orbitals.

    S0 -> triplet dipoles are zero (spin-forbidden). With triplets mc.fcisolver is the mixed
    solver, so the singlet transition density comes from direct_spin1.trans_rdm1.
    """
    dipoles: dict[StateLabel, np.ndarray] = {}
    try:
        trans_rdm1 = mc.fcisolver.trans_rdm1
        if state_counts(states)[1]:
            from pyscf.fci import direct_spin1

            trans_rdm1 = direct_spin1.trans_rdm1
        r_ints = mol.intor("int1e_r", comp=3)
        mo_cas = mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas]
        ci0 = mc.ci[0]  # S0 is the first state
        for label in states[1:]:
            if label.multiplicity != 1:
                dipoles[label] = np.zeros(3)
                continue
            t_mo = trans_rdm1(ci0, mc.ci[states.index(label)], mc.ncas, mc.nelecas)
            t_ao = mo_cas @ t_mo @ mo_cas.T
            dipoles[label] = -np.einsum("xij,ji->x", r_ints, t_ao)
    except Exception as exc:
        raise wrap_failure("The transition dipole calculation", exc) from exc
    return dipoles


def _expansion(
    ncore: int, ncas: int, n_alpha: int, n_beta: int, cis: Sequence[Any]
) -> DeterminantExpansion:
    """CI vectors with n_alpha/n_beta active electrons as determinants (see casscf_expansion)."""
    from pyscf.fci import cistring

    core = list(range(ncore))

    def occupations(n_el: int) -> list[list[int]]:
        return [
            core + [ncore + k for k in range(ncas) if int(s) >> k & 1]
            for s in cistring.make_strings(range(ncas), n_el)
        ]

    occ_alpha, occ_beta = occupations(n_alpha), occupations(n_beta)
    alpha = [a for a in occ_alpha for _ in occ_beta]
    beta = [b for _ in occ_alpha for b in occ_beta]
    coeffs = np.array([np.asarray(ci, dtype=float).reshape(-1) for ci in cis])
    return DeterminantExpansion(alpha, beta, coeffs)


def casscf_expansion(mc: Any) -> DeterminantExpansion:
    """SA-CASSCF states as determinants over mc.mo_coeff: core orbitals, then the active orbitals
    of each alpha/beta FCI string in ascending order (alpha string = slow CI index)."""
    n_alpha, n_beta = (int(n) for n in mc.nelecas)
    return _expansion(int(mc.ncore), int(mc.ncas), n_alpha, n_beta, list(mc.ci))


def casscf_expansions(
    mc: Any, states: Sequence[StateLabel]
) -> tuple[DeterminantExpansion, DeterminantExpansion | None]:
    """The singlet block (Ms = 0, mc.nelecas) and the triplet block (Ms = +1, one more alpha and
    one fewer beta electron) of the SA-CASSCF states, each over mc.mo_coeff with the layout of
    casscf_expansion; the triplet block is None without triplets."""
    n_singlets, n_triplets = state_counts(states)
    ncore, ncas = int(mc.ncore), int(mc.ncas)
    n_alpha, n_beta = (int(n) for n in mc.nelecas)
    cis = list(mc.ci)
    singlet = _expansion(ncore, ncas, n_alpha, n_beta, cis[:n_singlets])
    if not n_triplets:
        return singlet, None
    triplet = _expansion(ncore, ncas, n_alpha + 1, n_beta - 1, cis[n_singlets:])
    return singlet, triplet
