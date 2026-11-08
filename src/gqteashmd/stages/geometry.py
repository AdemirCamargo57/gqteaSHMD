"""Geometry and backend helpers shared by the quantum-chemistry stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gqteashmd.config.schema import ElectronicSection
from gqteashmd.core.units import ANGSTROM_TO_BOHR
from gqteashmd.electronic.base import ElectronicBackend, MoleculeSpec
from gqteashmd.electronic.factory import ground_state_settings, make_backend
from gqteashmd.stages.base import StageContext
from gqteashmd.util.fileio import read_json


def prepared_molecule(prepare_dir: Path) -> tuple[MoleculeSpec, np.ndarray]:
    """The molecule and its input geometry (bohr) from the `prepare` stage output."""
    data = read_json(Path(prepare_dir) / "molecule.json")
    spec = MoleculeSpec(tuple(data["symbols"]), int(data["charge"]), int(data["multiplicity"]))
    return spec, np.asarray(data["coords_angstrom"], dtype=float) * ANGSTROM_TO_BOHR


def working_geometry(ctx: StageContext) -> tuple[MoleculeSpec, np.ndarray, str]:
    """The optimized geometry if `optimize` ran, else the prepared input geometry (bohr)."""
    spec, coords = prepared_molecule(ctx.upstream_dir("prepare"))
    if "optimize" in ctx.upstream:
        data = read_json(ctx.upstream["optimize"] / "optimize.json")
        return spec, np.asarray(data["coords_bohr"], dtype=float), "optimize"
    return spec, coords, "prepare"


def stage_backend(
    ctx: StageContext,
    stage_name: str,
    molecule: MoleculeSpec,
    task_dir: Path,
    *,
    settings: ElectronicSection | None = None,
) -> ElectronicBackend:
    """One backend for a stage task, using the stage's effective resources."""
    electronic = ctx.config.electronic
    assert electronic is not None  # guaranteed by check_stage_inputs
    res = ctx.config.resources.for_stage(stage_name)
    return make_backend(
        ctx.config,
        molecule,
        settings=settings or ground_state_settings(electronic),
        log_file=task_dir / "electronic.log",
        threads=res.threads_per_task,
        memory_mb=res.memory_mb_per_task,
        scratch_dir=task_dir / "scratch",
    )
