"""Electronic-structure backend contract. All quantities in atomic units."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
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


@dataclass(frozen=True)
class Capabilities:
    """What a backend can compute (strategy §2.3)."""

    gradients: bool
    nacs: bool  # analytic nonadiabatic coupling vectors d_IJ = <I|dJ/dR>
    overlaps: bool  # <Psi_I(previous geometry)|Psi_J(current geometry)>
    transition_dipoles: bool


@dataclass(frozen=True, eq=False)
class Request:
    """One electronic-structure evaluation at a geometry (coordinates in bohr).

    nacs: pairs (I, J) with I < J; the result holds d_IJ = <Psi_I|dPsi_J/dR> (1/bohr).
    overlaps: also compute the overlaps with the previous geometry and align the phases.
    """

    coords_bohr: np.ndarray
    gradients: frozenset[StateLabel] = frozenset()
    transition_dipoles: bool = False
    nacs: frozenset[tuple[StateLabel, StateLabel]] = frozenset()
    overlaps: bool = False

    def __init__(
        self,
        coords_bohr: ArrayLike,
        gradients: frozenset[StateLabel] = frozenset(),
        transition_dipoles: bool = False,
        nacs: frozenset[tuple[StateLabel, StateLabel]] = frozenset(),
        overlaps: bool = False,
    ) -> None:
        coords = np.asarray(coords_bohr, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords_bohr must have shape (natom, 3), got {coords.shape}")
        pairs = frozenset(nacs)
        bad = sorted(f"({a}, {b})" for a, b in pairs if not a < b)
        if bad:
            raise ValueError(f"NAC pairs must be ordered (I, J) with I < J, got {', '.join(bad)}")
        object.__setattr__(self, "coords_bohr", coords)
        object.__setattr__(self, "gradients", frozenset(gradients))
        object.__setattr__(self, "transition_dipoles", transition_dipoles)
        object.__setattr__(self, "nacs", pairs)
        object.__setattr__(self, "overlaps", overlaps)


@dataclass(frozen=True, eq=False)
class ElectronicResult:
    """Energies (Eh), gradients (Eh/bohr), NACs d_IJ (1/bohr), S0→state transition dipoles (e·a0),
    ⟨S²⟩, and the overlaps with the previous geometry and the phases that align the states."""

    states: tuple[StateLabel, ...]
    energies: Mapping[StateLabel, float]
    gradients: Mapping[StateLabel, np.ndarray] = field(default_factory=dict)
    transition_dipoles: Mapping[StateLabel, np.ndarray] = field(default_factory=dict)
    spin_squares: Mapping[StateLabel, float] = field(default_factory=dict)
    nacs: Mapping[tuple[StateLabel, StateLabel], np.ndarray] = field(default_factory=dict)
    overlaps: np.ndarray | None = None
    phases: np.ndarray | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def excitation_energy(self, label: StateLabel) -> float:
        return self.energies[label] - self.energies[self.states[0]]

    def to_json(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for label in self.states:
            delta = self.excitation_energy(label)
            tdip = self.transition_dipoles.get(label)
            spin = self.spin_squares.get(label)
            rows.append(
                {
                    "label": str(label),
                    "energy_hartree": float(self.energies[label]),
                    "excitation_ev": float(delta * HARTREE_IN_EV),
                    "transition_dipole_au": None if tdip is None else [float(x) for x in tdip],
                    "oscillator_strength": None
                    if tdip is None
                    else float(oscillator_strength(delta, tdip)),
                    "spin_square": None if spin is None else float(spin),
                }
            )
        return {
            "schema_version": 1,
            "states": rows,
            "gradients_hartree_per_bohr": {
                str(label): np.asarray(g, dtype=float).tolist()
                for label, g in sorted(self.gradients.items())
            },
            "nacs_per_bohr": {
                f"{a}-{b}": np.asarray(v, dtype=float).tolist()
                for (a, b), v in sorted(self.nacs.items())
            },
            "overlaps": None if self.overlaps is None else np.asarray(self.overlaps).tolist(),
            "phases": None if self.phases is None else [float(p) for p in self.phases],
            "meta": dict(self.meta),
        }


class ElectronicBackend(ABC):
    @property
    @abstractmethod
    def states(self) -> tuple[StateLabel, ...]: ...

    @property
    @abstractmethod
    def capabilities(self) -> Capabilities: ...

    @abstractmethod
    def compute(self, request: Request) -> ElectronicResult: ...

    @abstractmethod
    def compute_more(
        self,
        gradients: frozenset[StateLabel] = frozenset(),
        nacs: frozenset[tuple[StateLabel, StateLabel]] = frozenset(),
    ) -> ElectronicResult:
        """Extra gradients/NACs at the geometry of the last compute() (e.g. after a hop)."""

    @abstractmethod
    def save_state(self, path: Path) -> None:
        """Write the reference wavefunction state the next compute() depends on."""

    @abstractmethod
    def load_state(self, path: Path) -> None:
        """Restore a state written by save_state() (same settings and molecule)."""
