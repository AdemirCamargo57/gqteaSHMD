import json

import numpy as np
import pytest
from tests.diagnosis.conftest import TOOLS, load_tool
from tests.diagnosis.test_project_inputs import LVC, make_project

from gqteashmd.config.schema import DynamicsStageSection
from gqteashmd.core.states import StateLabel
from gqteashmd.core.units import DALTON_IN_ME
from gqteashmd.dynamics.lvc import LVCModel
from gqteashmd.dynamics.randomness import GeneratorSource
from gqteashmd.stages.dynamics import run_settings
from gqteashmd.testing.fake_backend import ModelBackend
from gqteashmd.trajectory.model import BackendModel
from gqteashmd.trajectory.runner import InitialConditions, TrajectoryRunner
from gqteashmd.trajectory.store import StepLog
from gqteashmd.util.seeds import task_rng

rv = load_tool("diagnosis", "run_variants")
pi = load_tool("diagnosis", "project_inputs")


def test_deep_merge_and_variant_sections(tmp_path):
    assert rv.deep_merge({"a": 1, "c": {"x": 1, "y": 2}}, {"c": {"y": 3}, "d": 4}) == {
        "a": 1,
        "c": {"x": 1, "y": 3},
        "d": 4,
    }
    inputs = pi.load_project_inputs(make_project(tmp_path))
    variant = rv.Variant("dt_half", {"casscf": {"conv_tol_grad": 1e-5}}, {"dt_fs": 0.25})
    electronic, dynamics = rv.variant_sections(inputs.config, variant)
    assert electronic.casscf.conv_tol_grad == 1e-5
    assert electronic.casscf.ncas == inputs.config.electronic.casscf.ncas
    assert dynamics.dt_fs == 0.25 and dynamics.tmax_fs == inputs.config.dynamics.tmax_fs


def test_read_variants(tmp_path):
    path = tmp_path / "v.toml"
    path.write_text(
        '[[variant]]\nname = "base"\n\n[[variant]]\nname = "sa3"\n'
        "[variant.electronic]\nsinglets = 3\n"
        "[variant.dynamics]\ndt_fs = 0.25\n"
    )
    variants = rv.read_variants(path)
    assert [v.name for v in variants] == ["base", "sa3"]
    assert dict(variants[1].electronic) == {"singlets": 3}
    assert dict(variants[1].dynamics) == {"dt_fs": 0.25}
    assert dict(variants[0].electronic) == {} and dict(variants[0].dynamics) == {}


def test_read_variants_refuses_bad_names(tmp_path):
    path = tmp_path / "v.toml"
    path.write_text('[[variant]]\nname = "Bad Name"\n')
    with pytest.raises(ValueError, match="name"):
        rv.read_variants(path)


def test_variants_p9a_file():
    variants = rv.read_variants(TOOLS / "diagnosis" / "variants_p9a.toml")
    assert [v.name for v in variants] == [
        "base",
        "nacdr",
        "dt_quarter",
        "sa3_cas22",
        "sa3_cas23",
        "sa2_cas43",
    ]
    by = {v.name: v for v in variants}
    assert dict(by["base"].electronic) == {} and dict(by["base"].dynamics) == {}
    assert dict(by["nacdr"].dynamics) == {"coupling": "nacdr"}
    assert dict(by["dt_quarter"].dynamics) == {"dt_fs": 0.25}
    assert dict(by["sa3_cas22"].electronic) == {"singlets": 3}
    assert dict(by["sa3_cas23"].electronic) == {"singlets": 3, "casscf": {"ncas": 3, "nelecas": 2}}
    assert dict(by["sa2_cas43"].electronic) == {"casscf": {"ncas": 3, "nelecas": 4}}


def test_the_kept_rule():
    assert rv.is_kept("finished", 0.1)
    assert rv.is_kept("stopped_in_s0", 0.0)
    assert not rv.is_kept("finished", 0.1000001)
    assert not rv.is_kept("energy_drift", 0.0)


def test_run_variant_reproduces_the_stage_stream(tmp_path):
    """The variant runner gives, bit for bit, the steps.log of a direct runner with the
    stage's random stream task_rng(seed, "dynamics", index)."""
    project = make_project(tmp_path)
    inputs = pi.load_project_inputs(project)
    out = tmp_path / "out"
    summary = rv.run_variant(
        inputs,
        rv.Variant("base", {}, {}),
        out,
        workers=1,
        threads=1,
        memory_mb=500,
        backend=f"lvc:{LVC}",
    )
    assert summary["n_trajectories"] == 2
    assert sum(summary["outcomes"].values()) == 2
    assert 0 <= summary["n_kept"] <= 2
    assert (out / "base" / "summary.json").is_file()
    json.loads((out / "base" / "summary.json").read_text())
    row = summary["trajectories"][0]
    assert row["task"] == "traj_00000" and row["onset"] in rv.ONSETS

    # direct runner with the same stream
    model = LVCModel.from_template(LVC, spin_orbit=False)
    start = inputs.starts[0]
    masses = np.repeat(inputs.masses_amu * DALTON_IN_ME, 3)
    section = DynamicsStageSection.model_validate(inputs.config.dynamics.model_dump())
    bm = BackendModel(ModelBackend(model), masses, coupling=section.coupling)
    direct = tmp_path / "direct"
    TrajectoryRunner(bm, run_settings(section), direct).run(
        InitialConditions(start.q, start.v, bm.labels.index(StateLabel.parse("S1"))),
        GeneratorSource(task_rng(inputs.config.project.seed, "dynamics", start.index)),
    )
    n, f = bm.n_states, bm.n_dof
    a = StepLog(out / "base" / "traj_00000" / "steps.log", n, f).read()
    b = StepLog(direct / "steps.log", n, f).read()
    assert a.tobytes() == b.tobytes()


def test_run_variant_only_runs_the_named_tasks(tmp_path):
    inputs = pi.load_project_inputs(make_project(tmp_path))
    summary = rv.run_variant(
        inputs,
        rv.Variant("base", {}, {}),
        tmp_path / "out",
        workers=1,
        threads=1,
        memory_mb=500,
        backend=f"lvc:{LVC}",
        only=["traj_00001"],
    )
    assert [r["task"] for r in summary["trajectories"]] == ["traj_00001"]
