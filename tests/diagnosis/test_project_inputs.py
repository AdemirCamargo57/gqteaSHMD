from pathlib import Path

import numpy as np
import pytest
from tests.diagnosis.conftest import load_tool

from gqteashmd.config.loader import load_config, write_snapshot
from gqteashmd.dynamics.lvc import LVCModel
from gqteashmd.testing.canned_lvc import (
    write_canned_molecule,
    write_canned_samples,
    write_canned_trajectories,
)
from gqteashmd.vib.artifacts import read_samples_h5

LVC = Path(__file__).resolve().parents[1] / "data" / "lvc" / "lvc_singlets.template"

INPUT = """
[project]
name = "canned"
directory = "run"
seed = {seed}

[molecule]
geometry = "mol.xyz"
charge = 0
multiplicity = 1

[resources]
total_cores = 2
total_memory_mb = 2000
max_parallel_tasks = 1
threads_per_task = 1
memory_mb_per_task = 500

[electronic]
method = "casscf"
basis = "sto-3g"
singlets = {singlets}

[electronic.casscf]
ncas = 4
nelecas = 4

[dynamics]
tmax_fs = 2.0
dt_fs = 0.5
"""


def make_project(tmp_path: Path, seed: int = 7, n_traj: int = 2) -> Path:
    """A finished-upstream project from the canned LVC model (no PySCF)."""
    model = LVCModel.from_template(LVC, spin_orbit=False)
    singlets = model.state_counts[0]
    (tmp_path / "mol.xyz").write_text("1\n\nH 0 0 0\n")
    (tmp_path / "input.toml").write_text(INPUT.format(seed=seed, singlets=singlets))
    project = tmp_path / "run"
    for d in ("00_prepare", "03_sampling", "05_selection", ".gqteashmd"):
        (project / d).mkdir(parents=True)
    write_snapshot(load_config(tmp_path / "input.toml"), project / ".gqteashmd" / "config.json")
    write_canned_molecule(project / "00_prepare", model)
    write_canned_samples(project / "03_sampling", model, n_samples=5)
    write_canned_trajectories(
        project / "05_selection", [(k + 1, "S1") for k in range(n_traj)], seed
    )
    return project


def test_load_project_inputs(tmp_path):
    pi = load_tool("diagnosis", "project_inputs")
    project = make_project(tmp_path)
    inputs = pi.load_project_inputs(project)
    samples = read_samples_h5(project / "03_sampling" / "samples.h5")
    assert inputs.config.project.seed == 7
    assert len(inputs.starts) == 2
    first = inputs.starts[0]
    assert (first.task_id, first.index, first.sample, first.state) == ("traj_00000", 0, 1, "S1")
    np.testing.assert_array_equal(first.q, samples["positions_bohr"][0].reshape(-1))
    np.testing.assert_array_equal(first.v, samples["velocities_au"][0].reshape(-1))
    np.testing.assert_array_equal(inputs.masses_amu, samples["masses_amu"])
    assert inputs.symbols == tuple(samples["symbols"])


def test_load_project_inputs_refuses_a_foreign_rng_stream(tmp_path):
    pi = load_tool("diagnosis", "project_inputs")
    project = make_project(tmp_path)
    path = project / "05_selection" / "trajectories.json"
    path.write_text(path.read_text().replace('"dynamics"', '"other"', 1))
    with pytest.raises(ValueError, match="random stream"):
        pi.load_project_inputs(project)
