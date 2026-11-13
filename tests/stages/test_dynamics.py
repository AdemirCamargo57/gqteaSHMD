"""DynamicsStage on the fake LVC backend, called directly (no orchestrator)."""

import json
from pathlib import Path

import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import StageFailedError, TaskError
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.stages.base import StageContext
from gqteashmd.stages.dynamics import DynamicsStage, run_settings
from gqteashmd.testing import canned_lvc
from gqteashmd.trajectory import runner as runner_module
from gqteashmd.trajectory.runner import RestartError

LVC = Path(__file__).resolve().parents[1] / "data" / "lvc" / "lvc_singlets.template"
ELECTRONIC = '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 3\n'


@pytest.fixture(autouse=True)
def _template(monkeypatch):
    monkeypatch.setenv(canned_lvc.LVC_TEMPLATE_ENV, str(LVC))


def _context(tmp_path, write_input, dynamics="tmax_fs = 2.0\ndt_fs = 0.25\n", seed=7):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = write_input(tmp_path, extra=ELECTRONIC + "[dynamics]\n" + dynamics)
    loaded = load_config(path).config
    cfg = loaded.model_copy(update={"project": loaded.project.model_copy(update={"seed": seed})})
    layout = ProjectLayout(tmp_path / "run")
    model = canned_lvc.canned_model()
    dirs = {name: tmp_path / "run" / name for name in ("00_prepare", "01_sampling", "02_selection")}
    for d in dirs.values():
        d.mkdir(parents=True)
    canned_lvc.write_canned_molecule(dirs["00_prepare"], model)
    canned_lvc.write_canned_samples(dirs["01_sampling"], model, 4)
    canned_lvc.write_canned_trajectories(dirs["02_selection"], [(1, "S1"), (2, "S2")], seed)
    upstream = {
        "prepare": dirs["00_prepare"],
        "sampling": dirs["01_sampling"],
        "selection": dirs["02_selection"],
    }
    stage_dir = tmp_path / "run" / "03_dynamics"
    stage_dir.mkdir()
    return StageContext(cfg, layout, stage_dir, upstream)


def _stage():
    return DynamicsStage(
        backend_factory=canned_lvc.lvc_backend_factory, required_modules=("h5py>=3.9",)
    )


def _run_all(ctx, stage):
    results = {}
    for task in stage.plan(ctx):
        task_dir = ctx.stage_dir / "tasks" / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        results[task.task_id] = stage.run_task(ctx, task, task_dir)
    return results


def test_plan_run_and_finalize(tmp_path, write_input):
    ctx = _context(tmp_path, write_input)
    stage = _stage()
    tasks = stage.plan(ctx)
    assert [t.task_id for t in tasks] == ["traj_00000", "traj_00001"]
    assert tasks[1].payload["sample"] == 2 and tasks[1].payload["state"] == "S2"
    results = _run_all(ctx, stage)
    assert {r["outcome"] for r in results.values()} == {"finished"}
    assert results["traj_00000"]["t_final_fs"] == pytest.approx(2.0)
    stage.finalize(ctx, results)
    summary = json.loads((ctx.stage_dir / "dynamics_summary.json").read_text())
    assert summary["outcomes"]["finished"] == 2 and summary["missing_output"] == []
    assert (ctx.stage_dir / "dynamics_summary.md").exists()


def test_the_seed_reaches_the_trajectories(tmp_path, write_input):
    _run_all(_context(tmp_path / "a", write_input, seed=7), _stage())
    _run_all(_context(tmp_path / "b", write_input, seed=7), _stage())
    _run_all(_context(tmp_path / "c", write_input, seed=8), _stage())

    def read(root):
        return (root / "run/03_dynamics/tasks/traj_00001/steps.log").read_bytes()

    assert read(tmp_path / "a") == read(tmp_path / "b") != read(tmp_path / "c")


def test_fingerprint_and_reopen(tmp_path, write_input):
    ctx = _context(tmp_path, write_input)
    stage = _stage()
    fp = stage.fingerprint_inputs(ctx)
    assert "tmax_fs" not in fp["dynamics"] and "max_failed_fraction" not in fp["dynamics"]
    assert fp["dynamics"]["output_every"] == 1 and fp["project_seed"] == 7
    assert fp["electronic"]["overlap_norm_kept"] == 0.9999
    results = _run_all(ctx, stage)
    assert stage.reopen_tasks(ctx, results) == []
    longer = ctx.config.dynamics.model_copy(update={"tmax_fs": 3.0})
    ctx2 = StageContext(
        ctx.config.model_copy(update={"dynamics": longer}), ctx.layout, ctx.stage_dir, ctx.upstream
    )
    assert stage.reopen_tasks(ctx2, results) == ["traj_00000", "traj_00001"]


def test_enabled_and_check_config(tmp_path, write_input):
    ctx = _context(tmp_path, write_input)
    stage = _stage()
    assert stage.enabled(ctx.config)
    assert not stage.enabled(ctx.config.model_copy(update={"dynamics": None}))
    one = ctx.config.electronic.model_copy(update={"singlets": 1})
    assert (
        "singlets >= 2" in stage.check_config(ctx.config.model_copy(update={"electronic": one}))[0]
    )


def test_bad_upstream_data_is_a_non_retryable_task_error(tmp_path, write_input):
    ctx = _context(tmp_path, write_input)
    stage = _stage()
    task = stage.plan(ctx)[0]
    bad = type(task)(task.task_id, {**task.payload, "sample": 99})
    with pytest.raises(TaskError, match="--rerun-from sampling") as info:
        stage.run_task(ctx, bad, ctx.stage_dir / "x")
    assert info.value.retryable is False
    bad = type(task)(task.task_id, {**task.payload, "rng": {"stage": "dynamics", "task_index": 5}})
    with pytest.raises(TaskError, match="--rerun-from selection"):
        stage.run_task(ctx, bad, ctx.stage_dir / "y")


@pytest.mark.parametrize("damage", ["log", "current"])
def test_an_unusable_restart_advises_rerun_from_dynamics(tmp_path, write_input, damage):
    ctx = _context(tmp_path, write_input)
    stage = _stage()
    task = stage.plan(ctx)[0]
    task_dir = ctx.stage_dir / "tasks" / task.task_id
    task_dir.mkdir(parents=True)
    stage.run_task(ctx, task, task_dir)
    if damage == "log":
        log_file = task_dir / "steps.log"
        log_file.write_bytes(log_file.read_bytes()[:100])  # below the restart's record count
    else:
        (task_dir / "restart" / "current").write_text("garbage\n")
    with pytest.raises(TaskError, match="--rerun-from dynamics") as info:
        stage.run_task(ctx, task, task_dir)
    assert info.value.retryable is False
    assert isinstance(info.value.__cause__, RestartError)


def test_another_value_error_is_reported_without_rerun_advice(tmp_path, write_input, monkeypatch):
    def failing_step(*args, **kwargs):
        raise ValueError("injected failure inside a step")

    monkeypatch.setattr(runner_module, "step", failing_step)
    ctx = _context(tmp_path, write_input)
    stage = _stage()
    task = stage.plan(ctx)[0]
    task_dir = ctx.stage_dir / "tasks" / task.task_id
    task_dir.mkdir(parents=True)
    with pytest.raises(TaskError, match="trajectory traj_00000: injected failure") as info:
        stage.run_task(ctx, task, task_dir)
    assert info.value.retryable is False
    assert "--rerun-from" not in str(info.value)


def test_no_results_fails_the_stage(tmp_path, write_input):
    ctx = _context(tmp_path, write_input)
    with pytest.raises(StageFailedError):
        _stage().finalize(ctx, {})


def test_run_settings_converts_units(tmp_path, write_input):
    ctx = _context(
        tmp_path, write_input, dynamics="tmax_fs = 2.0\ndt_fs = 0.25\nstop_after_s0_fs = 1.0\n"
    )
    settings = run_settings(ctx.config.dynamics)
    assert settings.n_grid_steps == 8 and settings.options.coupling == "overlap"
    assert settings.stop_after_s0_au == pytest.approx(1.0 / 0.024188843265857)


def test_finalize_reports_rejection_signals_and_resumes(tmp_path, write_input, caplog):
    ctx = _context(tmp_path, write_input)
    stage = _stage()
    results = _run_all(ctx, stage)
    results["traj_00000"] = {
        **results["traj_00000"],
        "rejections_by_signal": {"overlap": 2, "rung": 1},
        "resumes": ["gen_00000003"],
    }
    results["traj_00001"] = {**results["traj_00001"], "rejections_by_signal": {"overlap": 1}}
    with caplog.at_level("INFO", logger="gqteashmd.stages.dynamics"):
        stage.finalize(ctx, results)
    summary = json.loads((ctx.stage_dir / "dynamics_summary.json").read_text())
    rows = {r["task"]: r for r in summary["trajectories"]}
    assert rows["traj_00000"]["rejections_by_signal"] == {"overlap": 2, "rung": 1}
    assert rows["traj_00000"]["n_resumes"] == 1 and rows["traj_00001"]["n_resumes"] == 0
    assert summary["totals"]["rejections_by_signal"] == {"overlap": 3, "rung": 1}
    assert "traj_00000 resumed from restart generation(s) gen_00000003" in caplog.text
    assert "rejections by signal" in (ctx.stage_dir / "dynamics_summary.md").read_text()
