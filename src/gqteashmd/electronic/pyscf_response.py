"""TDA / TDDFT (CIS / TDHF with functional "hf") with PySCF: retry ladder, gradients, dipoles,
determinant expansion. Triplets (separate response solve) have zero S0->T dipoles and are
expanded as their Ms = 0 component (plan ruling PR1).
PySCF is imported inside the functions only. Atomic units."""

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

# (name, extra roots in the Davidson solver, max_cycle factor). Extra roots are asked on every
# rung: with exactly n roots the initial guess at a symmetric geometry can lack a vector of one
# irrep, and the solver then returns a higher root as S_n without any error. Only the lowest n
# roots are kept and used (energies, gradients, dipoles, determinant expansion).
_LADDER: tuple[tuple[str, int, int], ...] = (("default", 3, 1), ("extra roots", 6, 3))


@dataclass(eq=False)
class ResponseOutcome:
    td: Any
    rung: str | None
    energies: dict[StateLabel, float]
    td_triplet: Any = None
    triplet_rung: str | None = None


def _converged_roots(td: Any, n: int) -> bool:
    """True when the lowest n roots exist and converged (a seam for tests)."""
    converged = np.atleast_1d(np.asarray(td.converged, dtype=bool))
    return len(td.e) >= n and bool(converged[:n].all())


def _solve(
    mf: Any, settings: ElectronicSection, n_roots: int, n_single: int, *, singlet: bool
) -> tuple[Any, str]:
    from pyscf import tddft

    method = settings.method.upper()
    label = method if singlet else f"{method} triplets"
    for name, extra, cycle_factor in _LADDER:
        td = tddft.TDA(mf) if settings.method == "tda" else tddft.TDDFT(mf)
        if not singlet:
            td.singlet = False
        td.nstates = min(n_roots + extra, n_single)
        td.conv_tol = settings.td_conv_tol
        td.max_cycle = td.max_cycle * cycle_factor
        try:
            td.kernel()
        except Exception as exc:
            raise wrap_failure(f"{label} ({name})", exc) from exc
        if _converged_roots(td, n_roots):
            return td, name
    tried = ", ".join(name for name, *_ in _LADDER)
    kind = "" if singlet else "triplet "
    raise ElectronicStructureError(
        f"{method} did not converge for all {n_roots} excited {kind}states (tried: {tried}; "
        f"td_conv_tol = {settings.td_conv_tol})."
    )


def solve_triplets(
    mf: Any, settings: ElectronicSection, n_roots: int, n_single: int
) -> tuple[Any, str]:
    """The triplet response solve (a seam for tests)."""
    return _solve(mf, settings, n_roots, n_single, singlet=False)


def run_response(
    mf: Any, settings: ElectronicSection, states: tuple[StateLabel, ...]
) -> ResponseOutcome:
    method = settings.method.upper()
    excited = [s for s in states[1:] if s.multiplicity == 1]
    triplets = [s for s in states if s.multiplicity == 3]
    energies: dict[StateLabel, float] = {states[0]: float(mf.e_tot)}
    if not excited and not triplets:
        return ResponseOutcome(None, None, energies)
    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    n_single = nocc * (int(mf.mo_coeff.shape[1]) - nocc)
    for group, kind in ((excited, ""), (triplets, "triplet ")):
        if len(group) > n_single:
            raise ElectronicStructureError(
                f"{method} cannot give {len(group)} excited {kind}states: the molecule has only "
                f"{n_single} single excitation{'s' if n_single != 1 else ''} in this basis."
            )
    td = rung = None
    if excited:
        td, rung = _solve(mf, settings, len(excited), n_single, singlet=True)
        for label, omega in zip(excited, td.e[: len(excited)], strict=True):
            energies[label] = float(mf.e_tot + omega)
    td_t = rung_t = None
    if triplets:
        td_t, rung_t = solve_triplets(mf, settings, len(triplets), n_single)
        for label, omega in zip(triplets, td_t.e[: len(triplets)], strict=True):
            energies[label] = float(mf.e_tot + omega)
    return ResponseOutcome(td, rung, energies, td_t, rung_t)


def response_gradients(
    mf: Any, td: Any, labels: Iterable[StateLabel], method: str, td_triplet: Any = None
) -> dict[StateLabel, np.ndarray]:
    gradients: dict[StateLabel, np.ndarray] = {}
    for label in sorted(labels):
        try:
            if label.multiplicity == 3:
                g = td_triplet.nuc_grad_method().kernel(state=label.index)
            elif label.index == 0:
                g = mf.nuc_grad_method().kernel()
            else:
                g = td.nuc_grad_method().kernel(state=label.index)
        except Exception as exc:
            if label.multiplicity == 3:
                raise wrap_failure(f"{method.upper()} triplet gradient of {label}", exc) from exc
            what = "SCF" if label.index == 0 else method.upper()
            raise wrap_failure(f"{what} gradient of {label}", exc) from exc
        gradients[label] = np.asarray(g, dtype=float)
    return gradients


def response_dipoles(
    td: Any, states: tuple[StateLabel, ...], method: str
) -> dict[StateLabel, np.ndarray]:
    excited = [s for s in states[1:] if s.multiplicity == 1]
    out: dict[StateLabel, np.ndarray] = {}
    if excited:
        try:
            mus = td.transition_dipole()[: len(excited)]
        except Exception as exc:
            raise wrap_failure(f"{method.upper()} transition dipole", exc) from exc
        out = {label: np.asarray(mu, dtype=float) for label, mu in zip(excited, mus, strict=True)}
    for label in states:
        if label.multiplicity == 3:
            out[label] = np.zeros(3)
    return out


def response_expansion(mf: Any, td: Any, n_excited: int) -> DeterminantExpansion:
    """Ground determinant plus spin-adapted single excitations over mf.mo_coeff.

    State k+1 = sum_ia X_k[i, a] (|i->a, alpha> + |i->a, beta>), the excited orbital taking i's
    place in the occupied list (the sign convention of a_a^+ a_i). For TDDFT only X is used (the
    RPA de-excitation part Y is dropped); truncate() renormalises.
    """
    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    nvir = int(mf.mo_coeff.shape[1]) - nocc
    occ = np.arange(nocc)
    n_single = nocc * nvir
    excited_occ = np.tile(occ, (n_single, 1))
    i_index, a_index = np.divmod(np.arange(n_single), nvir)
    excited_occ[np.arange(n_single), i_index] = nocc + a_index
    ground = np.tile(occ, (n_single, 1))
    alpha = np.vstack([occ[None, :], excited_occ, ground])
    beta = np.vstack([occ[None, :], ground, excited_occ])
    coeffs = np.zeros((1 + n_excited, 1 + 2 * n_single))
    coeffs[0, 0] = 1.0
    for k in range(n_excited):
        x = np.asarray(td.xy[k][0], dtype=float).reshape(-1)
        coeffs[k + 1, 1 : 1 + n_single] = x
        coeffs[k + 1, 1 + n_single :] = x
    return DeterminantExpansion(alpha, beta, coeffs)


def response_triplet_expansion(mf: Any, td: Any, n_triplets: int) -> DeterminantExpansion:
    """Triplets as their Ms = 0 component over mf.mo_coeff (plan ruling PR1):
    T_k = sum_ia X_k[i, a] (|i->a, alpha> - |i->a, beta>), the singlet layout of
    response_expansion without the ground determinant and with the beta part negated. X only
    (TDDFT's Y is dropped); truncate() renormalises."""
    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    nvir = int(mf.mo_coeff.shape[1]) - nocc
    occ = np.arange(nocc)
    n_single = nocc * nvir
    excited_occ = np.tile(occ, (n_single, 1))
    i_index, a_index = np.divmod(np.arange(n_single), nvir)
    excited_occ[np.arange(n_single), i_index] = nocc + a_index
    ground = np.tile(occ, (n_single, 1))
    alpha = np.vstack([excited_occ, ground])
    beta = np.vstack([ground, excited_occ])
    coeffs = np.zeros((n_triplets, 2 * n_single))
    for k in range(n_triplets):
        x = np.asarray(td.xy[k][0], dtype=float).reshape(-1)
        coeffs[k, :n_single] = x
        coeffs[k, n_single:] = -x
    return DeterminantExpansion(alpha, beta, coeffs)
