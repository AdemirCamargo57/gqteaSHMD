import json
from pathlib import Path

import pytest
from tests.diagnosis.conftest import TOOLS, load_tool
from tests.diagnosis.test_project_inputs import LVC, make_project

from gqteashmd.project.layout import ProjectLayout
from gqteashmd.stages.base import StageContext
from gqteashmd.stages.dynamics import DynamicsStage
from gqteashmd.testing import canned_lvc
from gqteashmd.trajectory.store import StepLog

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


def test_run_variant_reproduces_the_stage_stream(tmp_path, monkeypatch):
    """The variant runner gives, bit for bit, the steps.log of DynamicsStage.run_task on the
    canned LVC pipeline (P9a plan 1 final review, minor 3: not a second copy of the recipe)."""
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

    # the stage itself on the canned LVC pipeline, same project, same stream
    monkeypatch.setenv(canned_lvc.LVC_TEMPLATE_ENV, str(LVC))
    ctx = StageContext(
        inputs.config,
        ProjectLayout(project),
        tmp_path / "stage",
        {
            "prepare": project / "00_prepare",
            "sampling": project / "03_sampling",
            "selection": project / "05_selection",
        },
    )
    stage = DynamicsStage(
        backend_factory=canned_lvc.lvc_backend_factory, required_modules=("h5py>=3.9",)
    )
    task = stage.plan(ctx)[0]
    task_dir = ctx.stage_dir / "tasks" / task.task_id
    task_dir.mkdir(parents=True)
    stage.run_task(ctx, task, task_dir)
    model = canned_lvc.canned_model()
    a = StepLog(out / "base" / "traj_00000" / "steps.log", model.n_states, model.n_dof).read()
    b = StepLog(task_dir / "steps.log", model.n_states, model.n_dof).read()
    assert a.size > 1
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


def test_rerun_into_the_same_out_reuses_finished_trajectories(tmp_path):
    inputs = pi.load_project_inputs(make_project(tmp_path))
    kwargs = dict(workers=1, threads=1, memory_mb=500, backend=f"lvc:{LVC}")
    variant = rv.Variant("base", {}, {})
    first = rv.run_variant(inputs, variant, tmp_path / "out", **kwargs)
    assert (tmp_path / "out" / "base" / "variant.json").is_file()
    second = rv.run_variant(inputs, variant, tmp_path / "out", **kwargs)
    assert second["outcomes"] == first["outcomes"]
    assert [r["outcome"] for r in second["trajectories"]] == [
        r["outcome"] for r in first["trajectories"]
    ]


def test_rerun_with_other_settings_is_refused(tmp_path):
    inputs = pi.load_project_inputs(make_project(tmp_path))
    kwargs = dict(workers=1, threads=1, memory_mb=500, backend=f"lvc:{LVC}")
    rv.run_variant(inputs, rv.Variant("base", {}, {}), tmp_path / "out", **kwargs)
    with pytest.raises(ValueError, match="variant.json"):
        rv.run_variant(inputs, rv.Variant("base", {}, {"dt_fs": 0.25}), tmp_path / "out", **kwargs)


def test_variants_p9a2_file(tmp_path):
    path = Path(__file__).resolve().parents[2] / "tools" / "diagnosis" / "variants_p9a2.toml"
    variants = rv.read_variants(path)
    assert [v.name for v in variants] == ["base_no_f1", "sa3_cas22_final"]
    inputs = pi.load_project_inputs(make_project(tmp_path))
    for variant in variants:
        rv.variant_sections(inputs.config, variant)
