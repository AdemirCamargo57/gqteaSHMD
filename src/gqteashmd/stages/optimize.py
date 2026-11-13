"""Stage 1 (optional): optimize the S0 geometry on the [electronic] surface with geomeTRIC."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.chem.xyz import Geometry, format_xyz
from gqteashmd.config.schema import ShmdConfig
from gqteashmd.core.units import BOHR_IN_ANGSTROM
from gqteashmd.electronic.base import Request
from gqteashmd.electronic.factory import ground_state_fingerprint
from gqteashmd.errors import TaskError
from gqteashmd.geomopt import OptimizationFailedError, optimize_geometry
from gqteashmd.stages.base import Stage, StageContext, TaskSpec
from gqteashmd.stages.geometry import prepared_molecule, stage_backend
from gqteashmd.util.fileio import write_json_atomic, write_text_atomic

ARTIFACTS = ("optimized.xyz", "optimization_path.xyz", "optimize.json")


def _xyz(symbols: tuple[str, ...], coords_bohr: np.ndarray, comment: str) -> str:
    angstrom = np.asarray(coords_bohr) * BOHR_IN_ANGSTROM
    coords = tuple((float(x), float(y), float(z)) for x, y, z in angstrom)
    return format_xyz(Geometry(symbols, coords, comment=comment))


class OptimizeStage(Stage):
    name = "optimize"
    version = 1
    depends_on = ("prepare",)
    config_sections = ("electronic", "optimize")
    required_modules = ("pyscf>=2.6", "geometric>=1.0")

    def enabled(self, config: ShmdConfig) -> bool:
        return config.optimize.enabled

    def fingerprint_inputs(self, ctx: StageContext) -> dict[str, Any]:
        """Only what determines S0 (not the excited-state response settings) and [optimize]."""
        electronic = ctx.config.electronic
        return {
            "electronic": None if electronic is None else ground_state_fingerprint(electronic),
            "optimize": ctx.config.optimize.model_dump(mode="json"),
        }

    def plan(self, ctx: StageContext) -> list[TaskSpec]:
        return [TaskSpec("optimize")]

    def run_task(self, ctx: StageContext, task: TaskSpec, task_dir: Path) -> dict[str, Any]:
        options = ctx.config.optimize
        molecule, start = prepared_molecule(ctx.upstream_dir("prepare"))
        backend = stage_backend(ctx, self.name, molecule, task_dir, reference_coords_bohr=start)
        s0 = backend.states[0]

        def energy_and_gradient(coords: np.ndarray) -> tuple[float, np.ndarray]:
            result = backend.compute(Request(coords, gradients=frozenset({s0})))
            return result.energies[s0], result.gradients[s0]

        if len(molecule.symbols) == 1:
            return self._single_atom(
                molecule.symbols, start, energy_and_gradient, task_dir, options
            )
        try:
            opt = optimize_geometry(
                molecule.symbols,
                start,
                energy_and_gradient,
                max_steps=options.max_steps,
                convergence=options.convergence,
                work_dir=task_dir / "geometric",
            )
        except OptimizationFailedError as exc:
            raise TaskError(
                f"The geometry optimizer (geomeTRIC) failed: {exc}. Check the starting geometry "
                "(overlapping or very distant atoms, nearly linear angles), or disable "
                "[optimize] and provide an optimized geometry.",
                retryable=False,
            ) from exc
        path_text = "".join(
            _xyz(molecule.symbols, c, f"step {i} energy {e:.10f} Eh")
            for i, (c, e) in enumerate(zip(opt.path_bohr, opt.energies_hartree, strict=True))
        )
        write_text_atomic(task_dir / "optimization_path.xyz", path_text)
        if not opt.converged:
            write_text_atomic(
                task_dir / "best.xyz",
                _xyz(
                    molecule.symbols,
                    opt.coords_bohr,
                    f"lowest energy {opt.energy_hartree:.10f} Eh (not converged)",
                ),
            )
            raise TaskError(
                f"The geometry optimization did not converge in {options.max_steps} steps "
                f"(convergence = '{options.convergence}'). Increase [optimize] max_steps, start "
                "from a better geometry, or loosen [optimize] convergence; the lowest-energy "
                f"geometry found is in {task_dir / 'best.xyz'}",
                retryable=False,
            )
        _, final_gradient = energy_and_gradient(opt.coords_bohr)
        max_gradient = float(np.abs(final_gradient).max())
        write_text_atomic(
            task_dir / "optimized.xyz",
            _xyz(molecule.symbols, opt.coords_bohr, f"energy {opt.energy_hartree:.10f} Eh"),
        )
        write_json_atomic(
            task_dir / "optimize.json",
            {
                "schema_version": 1,
                "energy_hartree": opt.energy_hartree,
                "converged": True,
                "n_gradients": opt.n_gradients,
                "max_gradient_hartree_per_bohr": max_gradient,
                "convergence": options.convergence,
                "max_steps": options.max_steps,
                "symbols": list(molecule.symbols),
                "coords_bohr": opt.coords_bohr.tolist(),
                "coords_angstrom": (opt.coords_bohr * BOHR_IN_ANGSTROM).tolist(),
            },
        )
        return {
            "energy_hartree": opt.energy_hartree,
            "converged": True,
            "n_gradients": opt.n_gradients,
            "max_gradient_hartree_per_bohr": max_gradient,
        }

    def _single_atom(
        self,
        symbols: tuple[str, ...],
        start: np.ndarray,
        energy_and_gradient: Any,
        task_dir: Path,
        options: Any,
    ) -> dict[str, Any]:
        """One atom has no geometry to optimize: write the outputs without calling geomeTRIC."""
        energy, gradient = energy_and_gradient(start)
        max_gradient = float(np.abs(gradient).max())
        write_text_atomic(
            task_dir / "optimization_path.xyz",
            _xyz(symbols, start, f"step 0 energy {energy:.10f} Eh"),
        )
        write_text_atomic(
            task_dir / "optimized.xyz", _xyz(symbols, start, f"energy {energy:.10f} Eh")
        )
        write_json_atomic(
            task_dir / "optimize.json",
            {
                "schema_version": 1,
                "energy_hartree": energy,
                "converged": True,
                "n_gradients": 1,
                "max_gradient_hartree_per_bohr": max_gradient,
                "convergence": options.convergence,
                "max_steps": options.max_steps,
                "symbols": list(symbols),
                "coords_bohr": start.tolist(),
                "coords_angstrom": (start * BOHR_IN_ANGSTROM).tolist(),
            },
        )
        return {
            "energy_hartree": energy,
            "converged": True,
            "n_gradients": 1,
            "max_gradient_hartree_per_bohr": max_gradient,
        }

    def finalize(self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]) -> None:
        task_dir = ctx.layout.task_dir(ctx.stage_dir, "optimize")
        for name in ARTIFACTS:
            write_text_atomic(ctx.stage_dir / name, (task_dir / name).read_text(encoding="utf-8"))
