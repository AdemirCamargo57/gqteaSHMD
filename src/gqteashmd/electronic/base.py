"""Electronic-structure backend contract. All quantities in atomic units."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from gqteashmd.core.states import StateLabel
from gqteashmd.core.units import HARTREE_IN_EV
from gqteashmd.electronic.properties import oscillator_strength


@dataclass(frozen=True)
class MoleculeSpec:
    symbols: tuple[str, ...]
    charge: int = 0
    multiplicity: int = 1


@dataclass(frozen=True, eq=False)
class Request:
    """One electronic-structure evaluation at a geometry (coordinates in bohr)."""

    coords_bohr: np.ndarray
    gradients: frozenset[StateLabel] = frozenset()
    transition_dipoles: bool = False

    def __init__(
        self,
        coords_bohr: ArrayLike,
        gradients: frozenset[StateLabel] = frozenset(),
        transition_dipoles: bool = False,
    ) -> None:
        coords = np.asarray(coords_bohr, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords_bohr must have shape (natom, 3), got {coords.shape}")
        object.__setattr__(self, "coords_bohr", coords)
        object.__setattr__(self, "gradients", frozenset(gradients))
        object.__setattr__(self, "transition_dipoles", transition_dipoles)


@dataclass(frozen=True, eq=False)
class ElectronicResult:
    """Energies (Eh), gradients (Eh/bohr), S0→state transition dipoles (e·a0), ⟨S²⟩."""

    states: tuple[StateLabel, ...]
    energies: Mapping[StateLabel, float]
    gradients: Mapping[StateLabel, np.ndarray] = field(default_factory=dict)
    transition_dipoles: Mapping[StateLabel, np.ndarray] = field(default_factory=dict)
    spin_squares: Mapping[StateLabel, float] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)

    def excitation_energy(self, label: StateLabel) -> float:
        return self.energies[label] - self.energies[self.states[0]]

    def to_json(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for label in self.states:
            delta = self.excitation_energy(label)
            tdip = self.transition_dipoles.get(label)
            rows.append(
                {
                    "label": str(label),
                    "energy_hartree": float(self.energies[label]),
                    "excitation_ev": delta * HARTREE_IN_EV,
                    "transition_dipole_au": None if tdip is None else [float(x) for x in tdip],
                    "oscillator_strength": None
                    if tdip is None
                    else oscillator_strength(delta, tdip),
                    "spin_square": self.spin_squares.get(label),
                }
            )
        return {
            "schema_version": 1,
            "states": rows,
            "gradients_hartree_per_bohr": {
                str(label): np.asarray(g, dtype=float).tolist()
                for label, g in sorted(self.gradients.items())
            },
            "meta": dict(self.meta),
        }


class ElectronicBackend(ABC):
    @property
    @abstractmethod
    def states(self) -> tuple[StateLabel, ...]: ...

    @abstractmethod
    def compute(self, request: Request) -> ElectronicResult: ...
