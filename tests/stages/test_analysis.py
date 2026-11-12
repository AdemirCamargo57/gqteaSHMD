"""AnalysisStage on synthetic output.h5 files, called directly (no orchestrator)."""

import json
import logging

import h5py
import numpy as np
import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.config.schema import ENERGY_JUMP_DEFAULT_EV
from gqteashmd.core.units import AU_TIME_IN_FS, HARTREE_IN_EV
from gqteashmd.errors import StageFailedError, TaskError
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.stages.analysis import AnalysisStage
from gqteashmd.stages.base import StageContext
from gqteashmd.testing import canned_lvc
from gqteashmd.trajectory.store import log_dtype, log_row, write_output_h5
from gqteashmd.util.fileio import write_json_atomic

ELECTRONIC = '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 2\n'
DYNAMICS = "[dynamics]\ntmax_fs = 2.0\ndt_fs = 0.5\n"
LABELS = ("S0", "S1")


def _write_traj(
    dyn_dir,
    task,
    outcome,
    active_by_step,
    *,
    e_total=None,
    populations=None,
    labels=LABELS,
    dt_fs=0.5,
    output_every=1,
):
    n = len(labels)
    dtype = log_dtype(n, 3)
    rows = []
    for k, active in enumerate(active_by_step):
        pops = np.zeros(n)
        pops[active] = 1.0
        if populations is not None:
            pops = np.asarray(populations[k], dtype=float)
        previous = active_by_step[k - 1] if k else active
        hopped = active != previous
        rows.append(
            log_row(
                dtype,
                step=k,
                grid_step=k,
                ticks=0,
                depth=0,
                time_au=k * dt_fs / AU_TIME_IN_FS,
                active=active,
                hop_from=previous if hopped else -1,
                hop_to=active if hopped else -1,
                e_total=-1.0 if e_total is None else e_total[k],
                c_real=np.sqrt(pops),
            )
        )
    task_dir = dyn_dir / "tasks" / task
    task_dir.mkdir(parents=True, exist_ok=True)
    write_output_h5(
        task_dir / "output.h5",
        np.array(rows, dtype=dtype),
        labels=labels,
        n_atoms=1,
        output_every=output_every,
        attrs={
            "dt_fs": dt_fs,
            "outcome": outcome,
            "task": task,
            "initial_state": labels[active_by_step[0] if active_by_step else 1],
            "sample": 1,
            "symbols": ["H"],
        },
    )


def _context(tmp_path, write_input, trajs, *, analysis="", seed=None, file_only=()):
    """`trajs`: list of (outcome, active_by_step, kwargs) or None (no file, in the summary).

    `file_only`: indices whose output.h5 exists but which are absent from the summary.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = write_input(
        tmp_path, extra=ELECTRONIC + DYNAMICS + "[analysis]\nn_bootstrap = 100\n" + analysis
    )
    cfg = load_config(path).config
    if seed is not None:
        cfg = cfg.model_copy(update={"project": cfg.project.model_copy(update={"seed": seed})})
    layout = ProjectLayout(cfg.project.directory)
    sel_dir = layout.root / "02_selection"
    dyn_dir = layout.root / "03_dynamics"
    stage_dir = layout.root / "04_analysis"
    for d in (sel_dir, dyn_dir, stage_dir):
        d.mkdir(parents=True)
    canned_lvc.write_canned_trajectories(
        sel_dir, [(k + 1, "S1") for k in range(len(trajs))], cfg.project.seed
    )
    rows = []
    for k, spec in enumerate(trajs):
        task = f"traj_{k:05d}"
        if spec is None:
            rows.append({"task": task, "outcome": "finished", "t_final_fs": 2.0})
            continue
        outcome, active, kwargs = spec
        _write_traj(dyn_dir, task, outcome, active, **kwargs)
        if k not in file_only:
            t_final = 0.5 * max(len(active) - 1, 0)
            rows.append({"task": task, "outcome": outcome, "t_final_fs": t_final})
    write_json_atomic(
        dyn_dir / "dynamics_summary.json", {"schema_version": 1, "trajectories": rows}
    )
    return StageContext(cfg, layout, stage_dir, {"selection": sel_dir, "dynamics": dyn_dir})


def run_stage(ctx, stage=None):
    stage = stage or AnalysisStage()
    results = {}
    for task in stage.plan(ctx):
        task_dir = ctx.layout.task_dir(ctx.stage_dir, task.task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        results[task.task_id] = stage.run_task(ctx, task, task_dir)
    stage.finalize(ctx, results)
    return ctx.stage_dir


def _run_task(ctx):
    """run_task only (no finalize); returns the task result."""
    stage = AnalysisStage()
    (task,) = stage.plan(ctx)
    task_dir = ctx.layout.task_dir(ctx.stage_dir, task.task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    return stage.run_task(ctx, task, task_dir)


def _analysis(ctx):
    return json.loads((ctx.stage_dir / "analysis.json").read_text())


FIN = "finished"


def test_populations_and_files(tmp_path, write_input):
    ctx = _context(
        tmp_path,
        write_input,
        [
            (FIN, [1, 1, 1, 1, 1], {}),
            (FIN, [1, 1, 0, 0, 0], {}),
            ("energy_drift", [1, 1], {}),
        ],
    )
    stage_dir = run_stage(ctx)
    data = json.loads((stage_dir / "analysis.json").read_text())
    assert data["n_selected"] == 3 and data["n_included"] == 2
    assert data["excluded"] == [
        {"task": "traj_00002", "reason": "energy_drift", "t_final_fs": 0.5, "energy_jump_eV": None}
    ]
    assert data["included"] == ["traj_00000", "traj_00001"]
    assert data["outcomes"]["finished"] == 2 and data["outcomes"]["energy_drift"] == 1
    assert data["outcomes"]["task_failed"] == 0
    assert data["horizon_fs"] == pytest.approx(2.0) and data["dt_out_fs"] == pytest.approx(0.5)
    assert data["hops"]["pairs"] == {"S1->S0": 1}
    assert data["stage_failed"] is False and data["error"] is None
    assert data["fit"]["states"] == ["S1"] and data["fit"]["population"] == "classical"
    assert [row["time_fs"] for row in data["table"]] == pytest.approx([0, 0.5, 1.0, 1.5, 2.0])
    assert data["settings"]["project_seed"] == ctx.config.project.seed
    with h5py.File(stage_dir / "populations.h5") as f:
        np.testing.assert_allclose(f["classical"][()][:, 0], [0, 0, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(f["quantum"][()][:, 1], [1, 1, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(f["fit_target"][()], [1, 1, 0.5, 0.5, 0.5])
        assert f.attrs["n_included"] == 2 and f.attrs["n_selected"] == 3
        assert list(f.attrs["fit_states"]) == ["S1"]
    for name in ("populations.csv", "hops.csv", "report.md"):
        assert (stage_dir / name).exists()
    hops_csv = (stage_dir / "hops.csv").read_text().splitlines()
    assert hops_csv[1] == "traj_00001,1.000000,S1,S0,0,1,0"


def test_no_included_trajectory_stops_after_writing_the_report(tmp_path, write_input):
    ctx = _context(
        tmp_path,
        write_input,
        [("energy_drift", [1, 1], {}), ("energy_drift", [1, 1, 1], {})],
    )
    stage_dir = ctx.stage_dir
    with pytest.raises(StageFailedError, match="No trajectory can be analysed"):
        run_stage(ctx)
    data = json.loads((stage_dir / "analysis.json").read_text())
    assert data["stage_failed"] is True and (stage_dir / "report.md").exists()
    assert data["small_ensemble"] is True  # present on the stop paths too
    assert "energy_drift 2" in data["error"]
    assert data["n_included"] == 0 and data["fit"] is None
    assert not (stage_dir / "populations.h5").exists()


def test_no_included_after_jump_exclusion_advises_exclude_energy_jumps(tmp_path, write_input):
    jump = [-1.0, -1.0, -1.0 + 0.2 / HARTREE_IN_EV, -1.0 + 0.2 / HARTREE_IN_EV, -1.0]
    ctx = _context(
        tmp_path,
        write_input,
        [(FIN, [1, 1, 1, 1, 1], {"e_total": jump})],
        analysis="exclude_energy_jumps = true\n",
    )
    with pytest.raises(StageFailedError, match="exclude_energy_jumps = false"):
        run_stage(ctx)


def test_a_broken_norm_stops_as_a_program_error(tmp_path, write_input):
    pops = [[0.0, 1.0]] * 5
    pops[3] = [0.1, 1.0]
    ctx = _context(
        tmp_path,
        write_input,
        [(FIN, [1, 1, 1, 1, 1], {}), (FIN, [1, 1, 1, 1, 1], {"populations": pops})],
    )
    with pytest.raises(StageFailedError, match="program error"):
        run_stage(ctx)
    data = _analysis(ctx)
    assert data["stage_failed"] is True and "traj_00001" in data["error"]


def test_an_unreadable_output_h5_is_a_non_retryable_task_error(tmp_path, write_input):
    ctx = _context(tmp_path, write_input, [(FIN, [1, 1, 1, 1, 1], {})] * 2)
    path = ctx.upstream["dynamics"] / "tasks" / "traj_00001" / "output.h5"
    path.write_bytes(b"not an hdf5 file at all")
    with pytest.raises(TaskError, match="--rerun-from dynamics") as info:
        run_stage(ctx)
    assert info.value.retryable is False
    assert "traj_00001" in str(info.value)


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"labels": ("S0", "S2")}, "state_labels"),
        ({"dt_fs": 0.25}, "dt_fs"),
        ({"output_every": 2}, "output_every"),
    ],
)
def test_inconsistent_labels_are_a_task_error(tmp_path, write_input, kwargs, field):
    ctx = _context(
        tmp_path, write_input, [(FIN, [1, 1, 1, 1, 1], {}), (FIN, [1, 1, 1, 1, 1], kwargs)]
    )
    with pytest.raises(TaskError, match="--rerun-from dynamics") as info:
        run_stage(ctx)
    assert field in str(info.value) and info.value.retryable is False
    assert "traj_00001" in str(info.value)


def test_non_finite_energy_in_an_included_trajectory_is_a_task_error(tmp_path, write_input):
    energies = [-1.0, -1.0, float("nan"), -1.0, -1.0]
    ctx = _context(
        tmp_path,
        write_input,
        [(FIN, [1, 1, 1, 1, 1], {}), (FIN, [1, 1, 1, 1, 1], {"e_total": energies})],
    )
    with pytest.raises(TaskError, match="non-finite e_total") as info:
        run_stage(ctx)
    assert info.value.retryable is False and "--rerun-from dynamics" in str(info.value)


def test_missing_output_counts_as_task_failed(tmp_path, write_input):
    ctx = _context(
        tmp_path,
        write_input,
        [(FIN, [1, 1, 1, 1, 1], {}), None, (FIN, [1, 1, 1, 1, 1], {})],
        file_only=(2,),
    )
    run_stage(ctx)
    data = _analysis(ctx)
    assert data["n_included"] == 1 and data["outcomes"]["task_failed"] == 2
    assert data["excluded"] == [
        {"task": "traj_00001", "reason": "task_failed", "t_final_fs": None, "energy_jump_eV": None},
        {"task": "traj_00002", "reason": "task_failed", "t_final_fs": None, "energy_jump_eV": None},
    ]


def test_missing_trajectories_json_names_rerun_from_selection(tmp_path, write_input):
    ctx = _context(tmp_path, write_input, [(FIN, [1, 1, 1, 1, 1], {})])
    (ctx.upstream["selection"] / "trajectories.json").unlink()
    with pytest.raises(TaskError, match="--rerun-from selection") as info:
        run_stage(ctx)
    assert info.value.retryable is False


def test_malformed_dynamics_summary_names_rerun_from_dynamics(tmp_path, write_input):
    ctx = _context(tmp_path, write_input, [(FIN, [1, 1, 1, 1, 1], {})])
    (ctx.upstream["dynamics"] / "dynamics_summary.json").write_text('{"schema_version": 1}')
    with pytest.raises(TaskError, match="dynamics_summary.json.*--rerun-from dynamics") as info:
        run_stage(ctx)
    assert info.value.retryable is False


def test_excluded_fraction_warning(tmp_path, write_input, caplog):
    trajs = [(FIN, [1, 1, 1, 1, 1], {})] * 3 + [("energy_drift", [1, 1], {})]
    ctx = _context(tmp_path / "a", write_input, trajs, analysis="warn_excluded_fraction = 0.2\n")
    with caplog.at_level(logging.WARNING, logger="gqteashmd.stages.analysis"):
        run_stage(ctx)
    assert any("selection bias" in w for w in _analysis(ctx)["warnings"])
    assert "selection bias" in caplog.text
    caplog.clear()
    ctx = _context(tmp_path / "b", write_input, trajs, analysis="warn_excluded_fraction = 0.3\n")
    with caplog.at_level(logging.WARNING, logger="gqteashmd.stages.analysis"):
        run_stage(ctx)
    assert not any("selection bias" in w for w in _analysis(ctx)["warnings"])
    assert "selection bias" not in caplog.text


def test_energy_jump_flag_and_exclusion(tmp_path, write_input):
    jump = [-1.0, -1.0, -1.0 + 0.2 / HARTREE_IN_EV, -1.0 + 0.2 / HARTREE_IN_EV, -1.0]
    trajs = [(FIN, [1, 1, 1, 1, 1], {"e_total": jump}), (FIN, [1, 1, 1, 1, 1], {})]
    ctx = _context(tmp_path / "a", write_input, trajs, analysis="energy_jump_eV = 0.1\n")
    run_stage(ctx)
    data = _analysis(ctx)
    assert data["n_included"] == 2 and "traj_00000" in data["included"]
    assert [f["task"] for f in data["energy"]["flagged"]] == ["traj_00000"]
    assert data["energy"]["flagged"][0]["energy_jump_eV"] == pytest.approx(0.2)
    assert data["energy"]["jump_threshold_eV"] == pytest.approx(0.1)
    assert JUMP_WARNING.format(k=1, x=0.1) in data["warnings"]

    ctx = _context(
        tmp_path / "b",
        write_input,
        trajs,
        analysis="energy_jump_eV = 0.1\nexclude_energy_jumps = true\n",
    )
    run_stage(ctx)
    data = _analysis(ctx)
    assert data["n_included"] == 1 and data["included"] == ["traj_00001"]
    (excluded,) = data["excluded"]
    assert excluded["task"] == "traj_00000" and excluded["reason"] == "energy_jump"
    assert excluded["energy_jump_eV"] == pytest.approx(0.2)
    assert excluded["t_final_fs"] == pytest.approx(2.0)
    assert not any("included trajectories have an energy jump" in w for w in data["warnings"])


JUMP_WARNING = (
    "{k} included trajectories have an energy jump above energy_jump_eV = {x} eV "
    "(see energy.flagged); consider exclude_energy_jumps = true"
)
SMALL_WARNING = (
    "only {n} trajectories are included (fewer than 20): populations, intervals and the fit "
    "are not statistically reliable"
)


def test_one_trajectory_is_a_small_ensemble_without_fit_uncertainties(tmp_path, write_input):
    ctx = _context(tmp_path, write_input, [(FIN, [1, 1, 0, 0, 0], {})])
    run_stage(ctx)
    data = _analysis(ctx)
    assert data["n_included"] == 1 and data["small_ensemble"] is True
    assert SMALL_WARNING.format(n=1) in data["warnings"]
    fit = data["fit"]
    assert fit["tau_fs"] is not None
    for key in ("t0_interval_fs", "tau_interval_fs", "t0_std_fs", "tau_std_fs"):
        assert fit[key] is None, key
    assert "- small_ensemble: True\n" in (ctx.stage_dir / "report.md").read_text()


def test_twenty_trajectories_are_not_a_small_ensemble(tmp_path, write_input):
    ctx = _context(tmp_path, write_input, _varied_trajectories(5) * 4)
    run_stage(ctx)
    data = _analysis(ctx)
    assert data["n_included"] == 20 and data["small_ensemble"] is False
    assert not any("fewer than 20" in w for w in data["warnings"])
    assert "- small_ensemble: False\n" in (ctx.stage_dir / "report.md").read_text()


def test_a_stop_removes_the_stage_files_of_an_earlier_success(tmp_path, write_input):
    ctx = _context(tmp_path, write_input, [(FIN, [1, 1, 1, 1, 1], {})])
    run_stage(ctx)
    for name in ("populations.h5", "populations.csv", "hops.csv"):
        assert (ctx.stage_dir / name).exists()
    dyn_dir = ctx.upstream["dynamics"]
    _write_traj(dyn_dir, "traj_00000", "energy_drift", [1, 1])
    write_json_atomic(
        dyn_dir / "dynamics_summary.json",
        {
            "schema_version": 1,
            "trajectories": [{"task": "traj_00000", "outcome": "energy_drift", "t_final_fs": 0.5}],
        },
    )
    with pytest.raises(StageFailedError, match="No trajectory can be analysed"):
        run_stage(ctx)
    for name in ("populations.h5", "populations.csv", "hops.csv"):
        assert not (ctx.stage_dir / name).exists(), name
    assert _analysis(ctx)["stage_failed"] is True


def _varied_trajectories(n=5):
    trajs = []
    for k in range(n):
        p1 = [1.0 - m * (0.05 + 0.03 * k) for m in range(5)]
        trajs.append((FIN, [1, 1, 1, 1, 1], {"populations": [[1 - p, p] for p in p1]}))
    return trajs


def test_bootstrap_is_reproducible_from_the_seed(tmp_path, write_input):
    texts = {}
    for name, seed in (("a", 7), ("b", 7), ("c", 8)):
        ctx = _context(tmp_path / name, write_input, _varied_trajectories(), seed=seed)
        run_stage(ctx)
        texts[name] = (ctx.stage_dir / "populations.csv").read_text()
    assert texts["a"] == texts["b"]
    assert texts["a"] != texts["c"]

    def columns(text):
        lines = [line for line in text.splitlines() if not line.startswith("#")]
        header = lines[0].split(",")
        rows = [line.split(",") for line in lines[1:]]
        return {h: [r[i] for r in rows] for i, h in enumerate(header)}

    a, c = columns(texts["a"]), columns(texts["c"])
    assert a["P_S1"] == c["P_S1"] and a["rho_S1"] == c["rho_S1"]
    assert a["rho_S1_lower"] != c["rho_S1_lower"] or a["rho_S1_upper"] != c["rho_S1_upper"]


def test_quantum_fit_population(tmp_path, write_input):
    ctx = _context(
        tmp_path, write_input, _varied_trajectories(), analysis='fit_population = "quantum"\n'
    )
    run_stage(ctx)
    data = _analysis(ctx)
    assert data["fit"]["population"] == "quantum"
    with h5py.File(ctx.stage_dir / "populations.h5") as f:
        np.testing.assert_allclose(f["fit_target"][()], f["quantum"][()][:, 1], atol=1e-15)


def test_fingerprint_inputs(tmp_path, write_input):
    # without enabled and warn_excluded_fraction; with project_seed
    ctx = _context(tmp_path, write_input, [(FIN, [1, 1, 1, 1, 1], {})])
    assert AnalysisStage().fingerprint_inputs(ctx) == {
        "analysis": {
            "fit_states": None,
            "fit_population": "classical",
            "n_bootstrap": 100,
            "energy_jump_eV": ENERGY_JUMP_DEFAULT_EV,
            "exclude_energy_jumps": False,
        },
        "project_seed": ctx.config.project.seed,
    }


def test_enabled_needs_a_dynamics_table(tmp_path, write_input):
    stage = AnalysisStage()
    without = load_config(write_input(tmp_path, extra=ELECTRONIC + "[analysis]\n")).config
    assert stage.enabled(without) is False
    with_dynamics = load_config(write_input(tmp_path, extra=ELECTRONIC + DYNAMICS)).config
    assert stage.enabled(with_dynamics) is True
    disabled = load_config(
        write_input(tmp_path, extra=ELECTRONIC + DYNAMICS + "[analysis]\nenabled = false\n")
    ).config
    assert stage.enabled(disabled) is False


def test_stage_attributes():
    stage = AnalysisStage()
    assert stage.name == "analysis" and stage.version == 1
    assert stage.depends_on == ("selection", "dynamics")
    assert stage.config_sections == ("analysis",)
    assert stage.required_modules == ("h5py>=3.9", "scipy>=1.11")
    assert stage.max_failed_fraction == 0.0


def test_a_step_0_electronic_failure_is_excluded(tmp_path, write_input):
    # P7b ruling R6: a failure at step 0 leaves an output.h5 with zero records
    ctx = _context(
        tmp_path,
        write_input,
        [(FIN, [1, 1, 1, 1, 1], {}), ("electronic_failure", [], {})],
    )
    run_stage(ctx)
    data = _analysis(ctx)
    assert data["n_included"] == 1 and data["included"] == ["traj_00000"]
    assert data["outcomes"]["electronic_failure"] == 1 and data["outcomes"]["task_failed"] == 0
    assert data["excluded"] == [
        {
            "task": "traj_00001",
            "reason": "electronic_failure",
            "t_final_fs": 0.0,
            "energy_jump_eV": None,
        }
    ]
    assert data["energy"]["flagged"] == []
    assert (ctx.stage_dir / "hops.csv").read_text().splitlines() == [
        "task,time_fs,from,to,frustrated,included,after_horizon"
    ]


def test_a_zero_record_finished_file_is_a_task_error(tmp_path, write_input):
    ctx = _context(tmp_path, write_input, [(FIN, [1, 1, 1, 1, 1], {}), (FIN, [], {})])
    with pytest.raises(TaskError, match="t = 0") as info:
        run_stage(ctx)
    assert info.value.retryable is False and "--rerun-from dynamics" in str(info.value)
    assert "traj_00001" in str(info.value)


def test_n_total_is_in_every_result(tmp_path, write_input):
    ok = _run_task(_context(tmp_path / "ok", write_input, [(FIN, [1, 1, 1, 1, 1], {})]))
    assert ok["stage_failed"] is False and ok["n_total"] == 4
    none = _run_task(_context(tmp_path / "none", write_input, [("energy_drift", [1, 1], {})]))
    assert none["stage_failed"] is True and none["n_total"] == 4
    pops = [[0.0, 1.0]] * 5
    pops[3] = [0.1, 1.0]
    norm = _run_task(
        _context(tmp_path / "norm", write_input, [(FIN, [1, 1, 1, 1, 1], {"populations": pops})])
    )
    assert norm["stage_failed"] is True and norm["n_total"] == 4


def test_reopen_tasks_when_tmax_changes(tmp_path, write_input):
    ctx = _context(tmp_path, write_input, [(FIN, [1, 1, 1, 1, 1], {})])  # tmax 2.0, dt 0.5
    stage = AnalysisStage()
    assert stage.reopen_tasks(ctx, {"analysis": {"n_total": 4}}) == []
    assert stage.reopen_tasks(ctx, {"analysis": {"n_total": 6}}) == ["analysis"]
    assert stage.reopen_tasks(ctx, {"analysis": {"stage_failed": False}}) == ["analysis"]
    assert stage.reopen_tasks(ctx, {}) == []
