"""TDA / TDDFT (CIS / TDHF with functional "hf") with PySCF: retry ladder, gradients, dipoles,
determinant expansion. PySCF is imported inside the functions only. Atomic units."""

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


def _converged_roots(td: Any, n: int) -> bool:
    """True when the lowest n roots exist and converged (a seam for tests)."""
    converged = np.atleast_1d(np.asarray(td.converged, dtype=bool))
    return len(td.e) >= n and bool(converged[:n].all())


def run_response(
    mf: Any, settings: ElectronicSection, states: tuple[StateLabel, ...]
) -> ResponseOutcome:
    from pyscf import tddft

    method = settings.method.upper()
    excited = states[1:]
    energies: dict[StateLabel, float] = {states[0]: float(mf.e_tot)}
    if not excited:
        return ResponseOutcome(None, None, energies)
    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    n_single = nocc * (int(mf.mo_coeff.shape[1]) - nocc)
    if len(excited) > n_single:
        raise ElectronicStructureError(
            f"{method} cannot give {len(excited)} excited states: the molecule has only "
            f"{n_single} single excitation{'s' if n_single != 1 else ''} in this basis."
        )
    for name, extra, cycle_factor in _LADDER:
        td = tddft.TDA(mf) if settings.method == "tda" else tddft.TDDFT(mf)
        td.nstates = min(len(excited) + extra, n_single)
        td.conv_tol = settings.td_conv_tol
        td.max_cycle = td.max_cycle * cycle_factor
        try:
            td.kernel()
        except Exception as exc:
            raise wrap_failure(f"{method} ({name})", exc) from exc
        if _converged_roots(td, len(excited)):
            for label, omega in zip(excited, td.e[: len(excited)], strict=True):
                energies[label] = float(mf.e_tot + omega)
            return ResponseOutcome(td, name, energies)
    tried = ", ".join(name for name, *_ in _LADDER)
    raise ElectronicStructureError(
        f"{method} did not converge for all {len(excited)} excited states (tried: {tried}; "
        f"td_conv_tol = {settings.td_conv_tol})."
    )


def response_gradients(
    mf: Any, td: Any, labels: Iterable[StateLabel], method: str
) -> dict[StateLabel, np.ndarray]:
    gradients: dict[StateLabel, np.ndarray] = {}
    for label in sorted(labels):
        try:
            if label.index == 0:
                g = mf.nuc_grad_method().kernel()
            else:
                g = td.nuc_grad_method().kernel(state=label.index)
        except Exception as exc:
            what = "SCF" if label.index == 0 else method.upper()
            raise wrap_failure(f"{what} gradient of {label}", exc) from exc
        gradients[label] = np.asarray(g, dtype=float)
    return gradients


def response_dipoles(
    td: Any, states: tuple[StateLabel, ...], method: str
) -> dict[StateLabel, np.ndarray]:
    excited = states[1:]
    if not excited:
        return {}
    try:
        mus = td.transition_dipole()[: len(excited)]
    except Exception as exc:
        raise wrap_failure(f"{method.upper()} transition dipole", exc) from exc
    return {label: np.asarray(mu, dtype=float) for label, mu in zip(excited, mus, strict=True)}


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
