"""An ElectronicBackend over an analytic ElectronicModel, with scripted faults (tests only).

The model's flat coordinates are the backend's Cartesian coordinates (natom, 3) flattened. The
previous ElectronicSnapshot plays the role of PySCF's reference wavefunction: the model's own
``evaluate(q, previous)`` gives the phase-aligned states and the overlaps. Faults are keyed by
the 1-based number of the ``compute`` call (``calls`` sets the count already made, so a backend
built for a resumed run can continue the numbering of a crashed one).

Fault kinds: ``error`` / ``retryable`` (compute raises), ``low_overlap`` (min singular value 0.1),
``rung`` (CASSCF leaves its first rung), ``no_overlaps``, ``crash`` (SimulatedCrash), ``kill``
(os._exit, needs ``marker_dir``), ``nan_energy`` (every returned energy is NaN), ``nan_gradient``
(every returned gradient is NaN), ``nan_nac`` (every returned NAC is NaN) and ``more_error``
(the next ``compute_more`` after this compute call raises a non-retryable
ElectronicStructureError). Only the returned result is faulty; the backend's own state stays
clean.
"""

from __future__ import annotations

import dataclasses
import io
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.core.states import StateLabel, states_from_counts
from gqteashmd.dynamics.models import ElectronicModel
from gqteashmd.dynamics.records import ElectronicSnapshot
from gqteashmd.electronic.base import Capabilities, ElectronicBackend, ElectronicResult, Request
from gqteashmd.errors import ElectronicStructureError
from gqteashmd.util.fileio import write_bytes_atomic

FAULTS = (
    "error",
    "retryable",
    "low_overlap",
    "rung",
    "no_overlaps",
    "crash",
    "kill",
    "nan_energy",
    "nan_gradient",
    "nan_nac",
    "more_error",
)
_ARRAYS = ("energies", "gradients", "nacs", "hamiltonian", "vectors", "overlaps")


class SimulatedCrash(BaseException):
    """Stands in for a kill inside a test process (not an Exception, so nothing catches it)."""


class ModelBackend(ElectronicBackend):
    def __init__(
        self,
        model: ElectronicModel,
        *,
        faults: Mapping[int, str] | None = None,
        calls: int = 0,
        nacs: bool = True,
        marker_dir: Path | None = None,
    ) -> None:
        bad = sorted({kind for kind in (faults or {}).values() if kind not in FAULTS})
        if bad:
            raise ValueError(f"unknown fault kind(s) {bad}; known: {', '.join(FAULTS)}")
        self._model = model
        self._states = states_from_counts(int(model.n_states))
        self._faults = dict(faults or {})
        self.calls = int(calls)
        self._nacs = bool(nacs)
        self._marker_dir = None if marker_dir is None else Path(marker_dir)
        self._previous: ElectronicSnapshot | None = None
        self._live: ElectronicSnapshot | None = None
        self._more_error = False

    @property
    def states(self) -> tuple[StateLabel, ...]:
        return self._states

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            gradients=True, nacs=self._nacs, overlaps=True, transition_dipoles=False
        )

    def compute(self, request: Request) -> ElectronicResult:
        self.calls += 1
        self._live = None
        self._more_error = False
        fault = self._faults.get(self.calls)
        if fault == "crash":
            raise SimulatedCrash(f"simulated crash at compute call {self.calls}")
        if fault == "kill" and self._marker_dir is not None:
            marker = self._marker_dir / "killed_once"
            if not marker.exists():
                marker.write_text("killed")
                os._exit(137)
        if fault in ("error", "retryable"):
            raise ElectronicStructureError(
                f"scripted failure at compute call {self.calls}", retryable=fault == "retryable"
            )
        q = np.asarray(request.coords_bohr, dtype=float).reshape(-1)
        previous = self._previous
        snap = self._model.evaluate(q, previous)
        meta: dict[str, Any] = {"backend": "model", "casscf": "previous orbitals"}
        overlaps = None
        if previous is None:
            meta["casscf"] = "SCF orbitals"
        else:
            assert snap.overlaps is not None
            meta["overlap_min_singular_value"] = float(
                np.linalg.svd(np.asarray(snap.overlaps, dtype=float), compute_uv=False).min()
            )
            if request.overlaps and fault != "no_overlaps":
                overlaps = np.array(snap.overlaps, dtype=float)
        if fault == "low_overlap":
            meta["overlap_min_singular_value"] = 0.1
        if fault == "rung":
            meta["casscf"] = "SCF orbitals"
        self._previous = snap
        self._live = snap
        self._more_error = fault == "more_error"
        result = self._result(snap, request.gradients, request.nacs, overlaps, meta)
        if fault == "nan_energy":
            result = dataclasses.replace(
                result, energies={label: float("nan") for label in result.energies}
            )
        if fault == "nan_gradient":
            result = dataclasses.replace(
                result,
                gradients={label: np.full_like(g, np.nan) for label, g in result.gradients.items()},
            )
        if fault == "nan_nac":
            result = dataclasses.replace(
                result, nacs={pair: np.full_like(d, np.nan) for pair, d in result.nacs.items()}
            )
        return result

    def compute_more(
        self,
        gradients: frozenset[StateLabel] = frozenset(),
        nacs: frozenset[tuple[StateLabel, StateLabel]] = frozenset(),
    ) -> ElectronicResult:
        if self._live is None:
            raise ValueError("compute_more() needs a successful preceding compute()")
        if self._more_error:
            self._more_error = False
            raise ElectronicStructureError(
                f"scripted compute_more failure after compute call {self.calls}", retryable=False
            )
        return self._result(self._live, gradients, nacs, None, {"backend": "model"})

    def _result(
        self,
        snap: ElectronicSnapshot,
        gradients: frozenset[StateLabel],
        nacs: frozenset[tuple[StateLabel, StateLabel]],
        overlaps: np.ndarray | None,
        meta: dict[str, Any],
    ) -> ElectronicResult:
        index = {label: i for i, label in enumerate(self._states)}
        natom = snap.gradients.shape[1] // 3
        return ElectronicResult(
            states=self._states,
            energies={label: float(snap.energies[i]) for label, i in index.items()},
            gradients={
                label: np.array(snap.gradients[index[label]]).reshape(natom, 3)
                for label in gradients
            },
            nacs={
                (a, b): np.array(snap.nacs[index[a], index[b]]).reshape(natom, 3) for a, b in nacs
            },
            overlaps=overlaps,
            phases=np.ones(len(self._states)),
            meta=meta,
        )

    def reset(self) -> None:
        self._previous = None
        self._live = None

    def save_state(self, path: Path) -> None:
        if self._previous is None:
            raise ValueError("there is no state to save before the first compute()")
        arrays: dict[str, Any] = {
            name: np.asarray(getattr(self._previous, name))
            for name in _ARRAYS
            if getattr(self._previous, name) is not None
        }
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        write_bytes_atomic(Path(path), buffer.getvalue())

    def load_state(self, path: Path) -> None:
        with np.load(Path(path), allow_pickle=False) as data:
            values = {name: np.array(data[name]) for name in data.files}
        self._previous = ElectronicSnapshot(
            energies=values["energies"],
            gradients=values["gradients"],
            nacs=values["nacs"],
            hamiltonian=values["hamiltonian"],
            vectors=values.get("vectors"),
            overlaps=values.get("overlaps"),
        )
        self._live = None
