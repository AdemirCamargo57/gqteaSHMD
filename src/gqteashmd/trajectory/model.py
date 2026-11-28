"""BackendModel: the dynamics core's ElectronicModel over an ElectronicBackend (spec 3.1).

Without triplets (U stays None: the MCH Hamiltonian is diagonal), each evaluation requests the
energies, the gradient of the active state only (the caller sets ``active`` before every call),
the overlaps with the previous geometry (always) and, for ``coupling = "nacdr"``, every NAC pair.
The other gradient rows are NaN, a tripwire: the core reads only the active row (the gradient of
the old active state at q' for the velocities, and the new one at the start of the next step,
which ``refresh_active_gradient`` supplies after a hop). The overlap matrix is Lowdin-
orthonormalised, because the propagator assumes a unitary S; the raw smallest singular value
stays in ``last_meta`` for the step-rejection test.

With triplets (backend states S0..S(n-1), T1..Tm; dynamics with spin-orbit coupling, spec 3.2-3.3)
the core works in the expanded MCH basis of N = n + 3m states (``spin_expansion``): each
evaluation requests the energies, the spin-orbit Hamiltonian (``Request(soc=True)``), the overlaps
and, without ``eselect_Eh`` or at the first geometry (plan ruling PR3), every spin-free gradient;
no NACs (``coupling = "overlap"`` only, there are no triplet-triplet NACs). The snapshot holds the
expanded energies and gradients, zero NACs, the Lowdin-orthonormalised spin-free overlap blocks
expanded over Ms, and the backend's Hamiltonian, so the core takes its diagonal-basis path (U
tracked). With ``eselect_Eh`` the adapter applies the core's own mask: the diagonal basis
``diagonal_basis(H', u)`` against the tracked U (the runner sets ``u`` and ``active`` before every
step), ``selection_mask`` around the active diagonal energy, and ``compute_more`` for the kept
spin-free states; the other rows are NaN (the tripwire; the core zeroes them through the mask).
After a hop ``refresh_active_gradient`` fetches the missing gradients of the new mask.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from gqteashmd.core.states import StateLabel, states_from_counts
from gqteashmd.dynamics.records import ElectronicSnapshot, TrajectoryState
from gqteashmd.dynamics.representation import (
    diagonal_basis,
    diagonal_energies,
    mch_state,
    selection_mask,
)
from gqteashmd.dynamics.spin import expanded_states
from gqteashmd.electronic.base import ElectronicBackend, ElectronicResult, Request
from gqteashmd.errors import ElectronicStructureError
from gqteashmd.overlaps.tracking import lowdin_orthonormalize
from gqteashmd.trajectory.spin_expansion import (
    expand_energies,
    expand_gradients,
    expand_overlaps,
    expanded_size,
    spin_free_index,
    spin_free_rows,
)

FIRST_CASSCF_RUNG = "previous orbitals"
HERMITICITY_TOLERANCE_EH = 1e-12  # max |H - H^dagger| accepted from the backend (triplets)
COUPLINGS = ("overlap", "nacdr")


class BackendModel:
    def __init__(
        self,
        backend: ElectronicBackend,
        masses: np.ndarray,
        *,
        coupling: str,
        eselect_Eh: float | None = None,
    ) -> None:
        labels = tuple(backend.states)
        n = sum(1 for label in labels if label.multiplicity == 1)
        m = len(labels) - n
        if n < 1 or labels != states_from_counts(n, m):
            listed = ", ".join(str(label) for label in labels)
            raise ValueError(
                f"the backend states must be S0..S(n-1) followed by T1..Tm; got {listed}"
            )
        if len(labels) < 2:
            raise ValueError("dynamics needs at least two electronic states")
        if coupling not in COUPLINGS:
            raise ValueError(f"coupling must be one of {', '.join(COUPLINGS)}; got {coupling!r}")
        if m and coupling != "overlap":
            raise ValueError(
                f"coupling = {coupling!r} is not available with triplets (no triplet-triplet "
                "NACs); use coupling = 'overlap'"
            )
        if m and not backend.capabilities.soc:
            raise ValueError("triplets need a backend with spin-orbit couplings (Capabilities.soc)")
        if eselect_Eh is not None and not (math.isfinite(eselect_Eh) and eselect_Eh > 0):
            raise ValueError(f"eselect_Eh must be finite and > 0 or None; got {eselect_Eh!r}")
        if coupling == "nacdr" and not backend.capabilities.nacs:
            raise ValueError(
                "coupling = 'nacdr' needs analytic NACs (SA-CASSCF); use coupling = 'overlap'"
            )
        flat = np.asarray(masses, dtype=float).reshape(-1)
        if flat.size == 0 or flat.size % 3 or not np.all(np.isfinite(flat) & (flat > 0)):
            raise ValueError("masses must be positive, finite and 3 per atom")
        self.backend = backend
        self.labels: tuple[StateLabel, ...] = labels  # the spin-free states (PR2)
        self.n_singlets = n
        self.n_triplets = m
        self.n_spin_free = len(labels)
        self.n_states = expanded_size(n, m)  # N, the core's state count (= n without triplets)
        self.n_dof = int(flat.size)
        self.masses = flat
        self.coupling = coupling
        self.eselect_Eh = None if eselect_Eh is None else float(eselect_Eh)
        self.active = 0  # the core's active (diagonal) state, set by the caller before each step
        self.u: np.ndarray | None = None  # the core's tracked U, likewise (used by eselect only)
        self.last_meta: dict[str, Any] = {}
        self._had_previous = False
        self._pairs = (
            frozenset((a, b) for i, a in enumerate(labels) for b in labels[i + 1 :])
            if coupling == "nacdr"
            else frozenset()
        )

    @property
    def state_counts(self) -> tuple[int, ...]:
        if self.n_triplets:
            return (self.n_singlets, 0, self.n_triplets)
        return (self.n_states,)

    def active_spin_free(self, u: np.ndarray | None, active: int) -> int:
        """The spin-free state (index into ``labels``) of the dominant MCH state of ``active``."""
        k = mch_state(u, active)
        if not self.n_triplets:
            return k
        return spin_free_index(self.n_singlets, self.n_triplets, k)

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
        if self.n_triplets:
            return self._evaluate_with_triplets(flat, previous)
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

    # ---------------------------------------------------------------- triplets (spec 3.2-3.3)
    def _evaluate_with_triplets(
        self, flat: np.ndarray, previous: ElectronicSnapshot | None
    ) -> ElectronicSnapshot:
        n, m = self.n_singlets, self.n_triplets
        window = self.eselect_Eh is not None and previous is not None  # PR3: all at the start
        requested = frozenset() if window else frozenset(self.labels)
        result = self.backend.compute(
            Request(flat.reshape(-1, 3), gradients=requested, overlaps=True, soc=True)
        )
        self.last_meta = dict(result.meta)
        if previous is not None and result.overlaps is None:
            raise ValueError("the backend returned no overlaps with the previous geometry")
        energies_sf = np.array([float(result.energies[s]) for s in self.labels])
        if not np.all(np.isfinite(energies_sf)):
            raise ElectronicStructureError(
                f"the backend returned non-finite energies {energies_sf.tolist()}",
                retryable=False,
            )
        hamiltonian = self._hamiltonian(result)
        if result.overlaps is not None and not np.all(np.isfinite(result.overlaps)):
            raise ElectronicStructureError(
                "the backend returned non-finite overlaps with the previous geometry",
                retryable=False,
            )
        energies = expand_energies(energies_sf, n, m)
        gradients_sf = np.full((self.n_spin_free, self.n_dof), np.nan)
        self._fill_gradients(gradients_sf, result, requested)
        if window:
            # the core's steps 4-5: E^diag' against the tracked U, the mask around the active
            # (old lambda) diagonal energy; the same functions, so the same mask
            e_diag, _ = diagonal_basis(hamiltonian, self.u)
            kept = self._window(energies, float(e_diag[self.active]))
            if kept:
                self._fill_gradients(gradients_sf, self.backend.compute_more(gradients=kept), kept)
        return ElectronicSnapshot(
            energies=energies,
            gradients=expand_gradients(gradients_sf, n, m),
            nacs=np.zeros((self.n_states, self.n_states, self.n_dof)),
            hamiltonian=hamiltonian,
            overlaps=None if result.overlaps is None else self._overlaps(result.overlaps),
        )

    def _hamiltonian(self, result: ElectronicResult) -> np.ndarray:
        if result.hamiltonian is None:
            raise ElectronicStructureError(
                "the backend returned no spin-orbit Hamiltonian (Request.soc)", retryable=False
            )
        h = np.array(result.hamiltonian, dtype=complex)
        if h.shape != (self.n_states, self.n_states):
            raise ValueError(
                f"the spin-orbit Hamiltonian must have shape {(self.n_states, self.n_states)}; "
                f"got {h.shape}"
            )
        if not np.all(np.isfinite(h)):
            raise ElectronicStructureError(
                "the backend returned a non-finite spin-orbit Hamiltonian", retryable=False
            )
        if result.hamiltonian_labels is not None:
            expected = tuple(s.label() for s in expanded_states(self.state_counts))
            given = tuple(result.hamiltonian_labels)
            if given != expected:
                raise ElectronicStructureError(
                    f"the backend's spin-orbit Hamiltonian is in the order {', '.join(given)}; "
                    f"the dynamics needs {', '.join(expected)}",
                    retryable=False,
                )
        deviation = float(np.max(np.abs(h - h.conj().T)))
        if deviation > HERMITICITY_TOLERANCE_EH:
            raise ElectronicStructureError(
                f"the backend's spin-orbit Hamiltonian is not Hermitian: max |H - H^dagger| = "
                f"{deviation:.3e} Eh exceeds {HERMITICITY_TOLERANCE_EH:.0e} Eh",
                retryable=False,
            )
        return h

    def _overlaps(self, raw: np.ndarray) -> np.ndarray:
        """Lowdin per spin-free block (singlets, triplets), then expanded over Ms."""
        n, m = self.n_singlets, self.n_triplets
        s = np.asarray(raw, dtype=float)
        if s.shape != (self.n_spin_free, self.n_spin_free):
            raise ValueError(
                f"the overlaps must have shape {(self.n_spin_free, self.n_spin_free)}; "
                f"got {s.shape}"
            )
        if np.any(s[:n, n:] != 0) or np.any(s[n:, :n] != 0):
            raise ValueError("the backend returned nonzero singlet-triplet overlaps")
        orthonormal = np.zeros_like(s)
        orthonormal[:n, :n] = lowdin_orthonormalize(s[:n, :n])
        orthonormal[n:, n:] = lowdin_orthonormalize(s[n:, n:])
        return expand_overlaps(orthonormal, n, m)

    def _window(self, energies: np.ndarray, e_reference: float) -> frozenset[StateLabel]:
        """The spin-free states of ``selection_mask(energies, e_reference, eselect_Eh)``."""
        mask = selection_mask(energies, e_reference, self.eselect_Eh)
        assert mask is not None
        n, m = self.n_singlets, self.n_triplets
        return frozenset(self.labels[spin_free_index(n, m, int(k))] for k in np.flatnonzero(mask))

    def _fill_gradients(
        self, gradients_sf: np.ndarray, result: ElectronicResult, labels: frozenset[StateLabel]
    ) -> None:
        """Spin-free rows of ``labels`` from ``result``; a missing or non-finite one is a
        non-retryable electronic failure."""
        for label in sorted(labels):
            gradient = result.gradients.get(label)
            if gradient is None:
                raise ElectronicStructureError(
                    f"the backend returned no gradient for the state {label}", retryable=False
                )
            if not np.all(np.isfinite(gradient)):
                raise ElectronicStructureError(
                    f"the backend returned a non-finite gradient for the state {label}",
                    retryable=False,
                )
            gradients_sf[self.labels.index(label)] = np.asarray(gradient, dtype=float).reshape(-1)

    def _refresh_with_triplets(self, state: TrajectoryState) -> TrajectoryState:
        if self.eselect_Eh is None:
            return state  # every spin-free gradient came with compute
        n, m = self.n_singlets, self.n_triplets
        snap = state.snapshot
        # the next step's (1): the mask around E^diag_lambda of the state with its own U
        e_reference = float(diagonal_energies(snap.hamiltonian, state.u)[state.active])
        kept = self._window(snap.energies, e_reference)
        rows = spin_free_rows(n, m)
        gradients_sf = np.array(snap.gradients, dtype=float)[rows]
        missing = frozenset(
            label
            for label in kept
            if not np.all(np.isfinite(gradients_sf[self.labels.index(label)]))
        )
        if not missing:
            return state
        self._fill_gradients(gradients_sf, self.backend.compute_more(gradients=missing), missing)
        snapshot = dataclasses.replace(snap, gradients=expand_gradients(gradients_sf, n, m))
        return dataclasses.replace(state, snapshot=snapshot)

    def refresh_active_gradient(self, state: TrajectoryState) -> TrajectoryState:
        """The state with the gradient of its (new) active state computed at its geometry.

        With triplets: the missing gradients of the states the new active state's mask keeps
        (none without ``eselect_Eh``, where every gradient is already there)."""
        if self.n_triplets:
            return self._refresh_with_triplets(state)
        label = self.labels[state.active]
        result = self.backend.compute_more(gradients=frozenset({label}))
        self._check_gradient(result, label)
        gradients = np.array(state.snapshot.gradients, copy=True)
        gradients[state.active] = np.asarray(result.gradients[label], dtype=float).reshape(-1)
        snapshot = dataclasses.replace(state.snapshot, gradients=gradients)
        return dataclasses.replace(state, snapshot=snapshot)
