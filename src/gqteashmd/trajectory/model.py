"""BackendModel: the dynamics core's ElectronicModel over an ElectronicBackend (spec 3.1).

Singlets only (U stays None: the MCH Hamiltonian is diagonal). Each evaluation requests the
energies, the gradient of the active state only (the caller sets ``active`` before every call),
the overlaps with the previous geometry (always) and, for ``coupling = "nacdr"``, every NAC pair.
The other gradient rows are NaN, a tripwire: the core reads only the active row (the gradient of
the old active state at q' for the velocities, and the new one at the start of the next step,
which ``refresh_active_gradient`` supplies after a hop). The overlap matrix is Lowdin-
orthonormalised, because the propagator assumes a unitary S; the raw smallest singular value
stays in ``last_meta`` for the step-rejection test.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from gqteashmd.core.states import StateLabel
from gqteashmd.dynamics.records import ElectronicSnapshot, TrajectoryState
from gqteashmd.electronic.base import ElectronicBackend, ElectronicResult, Request
from gqteashmd.errors import ElectronicStructureError
from gqteashmd.overlaps.tracking import lowdin_orthonormalize

FIRST_CASSCF_RUNG = "previous orbitals"
COUPLINGS = ("overlap", "nacdr")


class BackendModel:
    def __init__(self, backend: ElectronicBackend, masses: np.ndarray, *, coupling: str) -> None:
        labels = tuple(backend.states)
        if any(label.multiplicity != 1 for label in labels):
            raise ValueError("the dynamics stage supports singlet states only (O7)")
        if len(labels) < 2:
            raise ValueError("dynamics needs at least two electronic states")
        if coupling not in COUPLINGS:
            raise ValueError(f"coupling must be one of {', '.join(COUPLINGS)}; got {coupling!r}")
        if coupling == "nacdr" and not backend.capabilities.nacs:
            raise ValueError(
                "coupling = 'nacdr' needs analytic NACs (SA-CASSCF); use coupling = 'overlap'"
            )
        flat = np.asarray(masses, dtype=float).reshape(-1)
        if flat.size == 0 or flat.size % 3 or not np.all(np.isfinite(flat) & (flat > 0)):
            raise ValueError("masses must be positive, finite and 3 per atom")
        self.backend = backend
        self.labels: tuple[StateLabel, ...] = labels
        self.n_states = len(labels)
        self.n_dof = int(flat.size)
        self.masses = flat
        self.coupling = coupling
        self.active = 0
        self.last_meta: dict[str, Any] = {}
        self._had_previous = False
        self._pairs = (
            frozenset((a, b) for i, a in enumerate(labels) for b in labels[i + 1 :])
            if coupling == "nacdr"
            else frozenset()
        )

    @property
    def state_counts(self) -> tuple[int, ...]:
        return (self.n_states,)

    @property
    def min_singular_value(self) -> float | None:
        value = self.last_meta.get("overlap_min_singular_value")
        return None if value is None else float(value)

    @property
    def rung_changed(self) -> bool:
        """True when CASSCF left its first ladder rung at a geometry with a previous one (R4)."""
        rung = self.last_meta.get("casscf", FIRST_CASSCF_RUNG)
        return self._had_previous and rung != FIRST_CASSCF_RUNG

    def evaluate(self, q: np.ndarray, previous: ElectronicSnapshot | None) -> ElectronicSnapshot:
        flat = np.asarray(q, dtype=float).reshape(-1)
        if flat.shape != (self.n_dof,):
            raise ValueError(f"q must have {self.n_dof} components; got {flat.size}")
        if not 0 <= self.active < self.n_states:
            raise ValueError(f"active must be in 0..{self.n_states - 1}; got {self.active}")
        self.last_meta = {}
        self._had_previous = previous is not None
        result = self.backend.compute(
            Request(
                flat.reshape(-1, 3),
                gradients=frozenset({self.labels[self.active]}),
                nacs=self._pairs,
                overlaps=True,
            )
        )
        self.last_meta = dict(result.meta)
        if previous is not None and result.overlaps is None:
            raise ValueError("the backend returned no overlaps with the previous geometry")
        self._check_finite(result, self.labels[self.active])
        return self._snapshot(result)

    def _check_finite(self, result: ElectronicResult, label: StateLabel) -> None:
        """Non-finite or missing electronic data is a (non-retryable) electronic failure."""
        energies = np.array([float(result.energies[s]) for s in self.labels])
        if not np.all(np.isfinite(energies)):
            raise ElectronicStructureError(
                f"the backend returned non-finite energies {energies.tolist()}", retryable=False
            )
        self._check_gradient(result, label)
        if self.coupling == "nacdr":
            bad = sorted(
                f"{a}-{b}"
                for (a, b), d in result.nacs.items()
                if not np.all(np.isfinite(np.asarray(d, dtype=float)))
            )
            if bad:
                raise ElectronicStructureError(
                    f"the backend returned a non-finite NAC for {', '.join(bad)}",
                    retryable=False,
                )
        if result.overlaps is not None and not np.all(np.isfinite(result.overlaps)):
            raise ElectronicStructureError(
                "the backend returned non-finite overlaps with the previous geometry",
                retryable=False,
            )

    @staticmethod
    def _check_gradient(result: ElectronicResult, label: StateLabel) -> None:
        gradient = result.gradients.get(label)
        if gradient is None:
            raise ElectronicStructureError(
                f"the backend returned no gradient for the active state {label}", retryable=False
            )
        if not np.all(np.isfinite(gradient)):
            raise ElectronicStructureError(
                f"the backend returned a non-finite gradient for the active state {label}",
                retryable=False,
            )

    def _snapshot(self, result: ElectronicResult) -> ElectronicSnapshot:
        n, f = self.n_states, self.n_dof
        index = {label: i for i, label in enumerate(self.labels)}
        energies = np.array([float(result.energies[label]) for label in self.labels])
        gradients = np.full((n, f), np.nan)
        for label, g in result.gradients.items():
            gradients[index[label]] = np.asarray(g, dtype=float).reshape(-1)
        nacs = np.zeros((n, n, f))
        for (a, b), d in result.nacs.items():
            i, j = index[a], index[b]
            nacs[i, j] = np.asarray(d, dtype=float).reshape(-1)
            nacs[j, i] = -nacs[i, j]
        overlaps = None if result.overlaps is None else lowdin_orthonormalize(result.overlaps)
        return ElectronicSnapshot(
            energies=energies,
            gradients=gradients,
            nacs=nacs,
            hamiltonian=np.diag(energies).astype(complex),
            overlaps=overlaps,
        )

    def refresh_active_gradient(self, state: TrajectoryState) -> TrajectoryState:
        """The state with the gradient of its (new) active state computed at its geometry."""
        label = self.labels[state.active]
        result = self.backend.compute_more(gradients=frozenset({label}))
        self._check_gradient(result, label)
        gradients = np.array(state.snapshot.gradients, copy=True)
        gradients[state.active] = np.asarray(result.gradients[label], dtype=float).reshape(-1)
        snapshot = dataclasses.replace(state.snapshot, gradients=gradients)
        return dataclasses.replace(state, snapshot=snapshot)
