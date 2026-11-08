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
from gqteashmd.errors import TaskError
from gqteashmd.geomopt import optimize_geometry
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

    def plan(self, ctx: StageContext) -> list[TaskSpec]:
        return [TaskSpec("optimize")]

    def run_task(self, ctx: StageContext, task: TaskSpec, task_dir: Path) -> dict[str, Any]:
        options = ctx.config.optimize
        molecule, start = prepared_molecule(ctx.upstream_dir("prepare"))
        backend = stage_backend(ctx, self.name, molecule, task_dir)
        s0 = backend.states[0]

        def energy_and_gradient(coords: np.ndarray) -> tuple[float, np.ndarray]:
            result = backend.compute(Request(coords, gradients=frozenset({s0})))
            return result.energies[s0], result.gradients[s0]

        opt = optimize_geometry(
            molecule.symbols,
            start,
            energy_and_gradient,
            max_steps=options.max_steps,
            convergence=options.convergence,
            work_dir=task_dir / "geometric",
        )
        path_text = "".join(
            _xyz(molecule.symbols, c, f"step {i} energy {e:.10f} Eh")
            for i, (c, e) in enumerate(zip(opt.path_bohr, opt.energies_hartree, strict=True))
        )
        write_text_atomic(task_dir / "optimization_path.xyz", path_text)
        if not opt.converged:
            write_text_atomic(
                task_dir / "last.xyz", _xyz(molecule.symbols, opt.coords_bohr, "last")
            )
            raise TaskError(
                f"The geometry optimization did not converge in {options.max_steps} steps "
                f"(convergence = '{options.convergence}'). Increase [optimize] max_steps, start "
                "from a better geometry, or loosen [optimize] convergence; the last geometry is "
                f"in {task_dir / 'last.xyz'}",
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

    def finalize(self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]) -> None:
        task_dir = ctx.layout.task_dir(ctx.stage_dir, "optimize")
        for name in ARTIFACTS:
            write_text_atomic(ctx.stage_dir / name, (task_dir / name).read_text(encoding="utf-8"))
