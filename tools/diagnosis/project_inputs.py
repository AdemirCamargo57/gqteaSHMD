"""The fixed initial conditions and settings of a gqteashmd project (P9a spec 3.3).

Reads the config snapshot, 00_prepare/molecule.json, <nn>_sampling/samples.h5 and
<nn>_selection/trajectories.json, exactly the inputs the dynamics stage reads.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gqteashmd.config.loader import load_snapshot
from gqteashmd.config.schema import ShmdConfig
from gqteashmd.excite.artifacts import read_trajectories_json
from gqteashmd.stages.geometry import prepared_molecule
from gqteashmd.vib.artifacts import read_samples_h5

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forensics import find_stage_dir  # noqa: E402


@dataclass(frozen=True)
class TrajectoryStart:
    task_id: str
    index: int
    sample: int
    state: str
    q: np.ndarray
    v: np.ndarray


@dataclass(frozen=True)
class ProjectInputs:
    project_dir: Path
    config: ShmdConfig
    symbols: tuple[str, ...]
    charge: int
    multiplicity: int
    masses_amu: np.ndarray
    starts: tuple[TrajectoryStart, ...]


def load_project_inputs(project_dir: Path) -> ProjectInputs:
    project_dir = Path(project_dir).expanduser()
    config = load_snapshot(project_dir / ".gqteashmd" / "config.json").config
    molecule, _ = prepared_molecule(find_stage_dir(project_dir, "prepare"))
    samples = read_samples_h5(find_stage_dir(project_dir, "sampling") / "samples.h5")
    selected = read_trajectories_json(
        find_stage_dir(project_dir, "selection") / "trajectories.json"
    )
    if tuple(samples["symbols"]) != tuple(molecule.symbols):
        raise ValueError(f"{project_dir}: the atoms of samples.h5 differ from molecule.json")
    starts = []
    for t in selected:
        if dict(t.rng) != {"stage": "dynamics", "task_index": t.index}:
            raise ValueError(f"{t.task_id} has the random stream {dict(t.rng)!r}")
        if not 1 <= t.sample <= samples["n_samples"]:
            raise ValueError(f"{t.task_id}: sample {t.sample} is not in samples.h5")
        starts.append(
            TrajectoryStart(
                t.task_id,
                t.index,
                t.sample,
                t.state,
                np.asarray(samples["positions_bohr"][t.sample - 1], dtype=float).reshape(-1),
                np.asarray(samples["velocities_au"][t.sample - 1], dtype=float).reshape(-1),
            )
        )
    return ProjectInputs(
        project_dir,
        config,
        tuple(molecule.symbols),
        int(molecule.charge),
        int(molecule.multiplicity),
        np.asarray(samples["masses_amu"], dtype=float),
        tuple(starts),
    )
