"""Geometry optimization with geomeTRIC (imported lazily) driving any energy+gradient function."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from gqteashmd.core.units import ANGSTROM_TO_BOHR, BOHR_IN_ANGSTROM

EnergyGradient = Callable[[np.ndarray], tuple[float, np.ndarray]]


@dataclass(frozen=True, eq=False)
class OptimizationResult:
    coords_bohr: np.ndarray
    energy_hartree: float
    converged: bool
    n_gradients: int
    path_bohr: list[np.ndarray] = field(default_factory=list)
    energies_hartree: list[float] = field(default_factory=list)


def optimize_geometry(
    symbols: Sequence[str],
    coords_bohr: ArrayLike,
    energy_and_gradient: EnergyGradient,
    *,
    max_steps: int,
    convergence: str,
    work_dir: Path,
) -> OptimizationResult:
    import geometric.engine
    import geometric.molecule
    import geometric.optimize
    from geometric.errors import GeomOptNotConvergedError

    path: list[np.ndarray] = []
    energies: list[float] = []

    class _Engine(geometric.engine.Engine):
        def calc_new(self, coords: Any, dirname: Any) -> dict[str, Any]:
            c = np.asarray(coords, dtype=float).reshape(-1, 3)
            energy, gradient = energy_and_gradient(c)
            path.append(c.copy())
            energies.append(float(energy))
            return {"energy": float(energy), "gradient": np.asarray(gradient, dtype=float).ravel()}

    molecule = geometric.molecule.Molecule()
    molecule.elem = list(symbols)
    molecule.xyzs = [np.asarray(coords_bohr, dtype=float) * BOHR_IN_ANGSTROM]
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    # geomeTRIC reconfigures the root logger (replaces its handlers, sets the level);
    # keep gqteashmd's logging intact and close geomeTRIC's own handlers afterwards.
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # geomeTRIC internals
            result = geometric.optimize.run_optimizer(
                customengine=_Engine(molecule),
                check=1,
                input=str(work_dir / "opt"),
                maxiter=max_steps,
                convergence_set=convergence,
            )
    except GeomOptNotConvergedError:
        return OptimizationResult(path[-1], energies[-1], False, len(path), path, energies)
    finally:
        for handler in list(root.handlers):
            if handler not in saved_handlers:
                root.removeHandler(handler)
                handler.close()
        for handler in saved_handlers:
            if handler not in root.handlers:
                root.addHandler(handler)
        root.setLevel(saved_level)
    final = np.asarray(result.xyzs[-1], dtype=float) * ANGSTROM_TO_BOHR
    return OptimizationResult(final, float(result.qm_energies[-1]), True, len(path), path, energies)
