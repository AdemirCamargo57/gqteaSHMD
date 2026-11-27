"""AnalysisStage on synthetic output.h5 files, called directly (no orchestrator)."""

import json
import logging
import types

import h5py
import numpy as np
import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.config.schema import ENERGY_JUMP_DEFAULT_EV
from gqteashmd.core.units import AU_TIME_IN_FS, HARTREE_IN_EV
from gqteashmd.errors import StageFailedError, TaskError
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.stages import analysis as analysis_module
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
    triplets=0,
    hops=None,
):
    if triplets:
        _write_soc_traj(dyn_dir, task, outcome, active_by_step, labels, triplets, hops, dt_fs)
        return
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


def _write_soc_traj(dyn_dir, task, outcome, active_by_step, labels, triplets, hops, dt_fs):
    """A schema-2 output.h5 (triplets, spec 4.2): `active_by_step` holds the active spin-free
    state of every record; `hops` the records with an accepted diagonal hop (default: those
    where the spin-free state changes). The diagonal index is taken equal to the spin-free one
    (the analysis reads only the spin-free datasets)."""
    n_sf = len(labels)
    n_singlets = n_sf - triplets
    n_exp = n_singlets + 3 * triplets
    dtype = log_dtype(n_exp, 3, n_spin_free=n_sf)
    if hops is None:
        hops = [
            k for k in range(1, len(active_by_step)) if active_by_step[k - 1] != active_by_step[k]
        ]
    rows = []
    for k, active in enumerate(active_by_step):
        pops = np.zeros(n_sf)
        pops[active] = 1.0
        c = np.zeros(n_exp)
        c[active] = 1.0
        hopped = k in hops
        rows.append(
            log_row(
                dtype,
                step=k,
                grid_step=k,
                ticks=0,
                depth=0,
                time_au=k * dt_fs / AU_TIME_IN_FS,
                active=active,
                hop_from=active_by_step[k - 1] if hopped else -1,
                hop_to=active if hopped else -1,
                e_total=-1.0,
                c_real=c,
                active_mch=active,
                populations_mch=pops,
            )
        )
    task_dir = dyn_dir / "tasks" / task
    task_dir.mkdir(parents=True, exist_ok=True)
    write_output_h5(
        task_dir / "output.h5",
        np.array(rows, dtype=dtype),
        labels=labels,
        n_atoms=1,
        output_every=1,
        attrs={
            "dt_fs": dt_fs,
            "outcome": outcome,
            "task": task,
            "initial_state": labels[active_by_step[0]],
            "sample": 1,
            "symbols": ["H"],
            "soc_operator": "somf",
        },
        n_singlets=n_singlets,
        n_triplets=triplets,
    )


def _context(
    tmp_path,
    write_input,
    trajs,
    *,
    analysis="",
    seed=None,
    file_only=(),
    estimator="complete",
    electronic=ELECTRONIC,
    dynamics=DYNAMICS,
):
    """`trajs`: list of (outcome, active_by_step, kwargs) or None (no file, in the summary).

    `file_only`: indices whose output.h5 exists but which are absent from the summary.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = write_input(
        tmp_path,
        extra=electronic
        + dynamics
        + f'[analysis]\nn_bootstrap = 100\nestimator = "{estimator}"\n'
        + analysis,
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
    "only {n} trajectories are included (fewer than "
    + str(analysis_module.SMALL_ENSEMBLE)
    + "): populations, intervals and the fit are not statistically reliable (measured 95 % "
    "tau interval " + analysis_module.SMALL_ENSEMBLE_COVERAGE + ")"
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


def test_n_min_trajectories_are_not_a_small_ensemble(tmp_path, write_input):
    n = analysis_module.SMALL_ENSEMBLE
    ctx = _context(tmp_path, write_input, _varied_trajectories(5) * (n // 5))
    run_stage(ctx)
    data = _analysis(ctx)
    assert data["n_included"] == n and data["small_ensemble"] is False
    assert not any("trajectories are included (fewer than" in w for w in data["warnings"])
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
            "estimator": "complete",
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
    assert stage.name == "analysis" and stage.version == 4
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


def test_an_electronic_discontinuity_is_excluded_and_counted(tmp_path, write_input):
    ctx = _context(
        tmp_path,
        write_input,
        [(FIN, [1, 1, 1, 1, 1], {}), ("electronic_discontinuity", [1, 1], {})],
    )
    stage_dir = run_stage(ctx)
    summary = json.loads((stage_dir / "analysis.json").read_text())
    assert summary["outcomes"]["electronic_discontinuity"] == 1
    assert any(e["reason"] == "electronic_discontinuity" for e in summary["excluded"])


def test_the_small_ensemble_threshold_is_the_measured_n_min():
    # P9b Task 6 measurement, ~/p9b/tau_coverage.json: 0.895 / 0.9275 / 0.935 at N = 20 / 50 / 100
    assert analysis_module.SMALL_ENSEMBLE == 50
    assert analysis_module.SMALL_ENSEMBLE_COVERAGE == (
        "coverage 0.895 at N = 20, 0.9275 at N = 50, 0.935 at N = 100"
    )


def test_censored_estimator_uses_a_lost_trajectory_up_to_its_end(tmp_path, write_input):
    trajs = [
        (FIN, [1, 1, 1, 0, 0], {}),
        (FIN, [1, 1, 1, 1, 1], {}),
        ("overlap_failure", [1, 0, 0], {}),
    ]
    ctx = _context(tmp_path, write_input, trajs, estimator="censored")
    run_stage(ctx)
    data = _analysis(ctx)
    assert data["schema_version"] == 3 and data["estimator"] == "censored"
    assert data["n_complete"] == 2 and data["n_censored"] == 1 and data["n_included"] == 3
    assert data["censored"] == [
        {
            "task": "traj_00002",
            "outcome": "overlap_failure",
            "reason": "outcome",
            "censored_at_fs": 1.0,
        }
    ]
    assert set(data["estimators"]) == {"complete", "censored", "hold_in_s0"}
    assert data["estimators"]["complete"]["n"] == 2
    assert data["estimators"]["censored"]["n"] == 3
    # at t = 1.0 fs: traj 0 in S1, traj 1 in S1, traj 2 in S0 -> P(S0) = 1/3; after that the
    # censored trajectory's S0 probability is carried by the estimator, not dropped
    row = next(r for r in data["table"] if r["time_fs"] == 1.0)
    assert row["P"]["S0"] == pytest.approx(1 / 3)
    complete_row = next(r for r in data["estimators"]["complete"]["table"] if r["time_fs"] == 1.0)
    assert complete_row["P"]["S0"] == 0.0  # the complete rule dropped the decayed trajectory


def test_without_losses_the_two_estimators_agree(tmp_path, write_input):
    ctx = _context(tmp_path, write_input, _varied_trajectories(), estimator="censored")
    run_stage(ctx)
    data = _analysis(ctx)
    for a, b in zip(data["table"], data["estimators"]["complete"]["table"], strict=True):
        for s, value in a["P"].items():
            assert value == pytest.approx(b["P"][s], abs=1e-12)
    assert data["n_censored"] == 0 and data["estimators"]["censored"]["empty_rows"] == 0


def test_the_censored_fraction_warning(tmp_path, write_input, caplog):
    trajs = [(FIN, [1, 1, 1, 1, 1], {})] + [("overlap_failure", [1, 0, 0], {})] * 3
    ctx = _context(tmp_path, write_input, trajs, estimator="censored")
    run_stage(ctx)
    text = " ".join(_analysis(ctx)["warnings"])
    assert "75% of the selected trajectories are not complete" in text
    assert "3 censored, 0 without data" in text


def test_the_files_carry_the_other_estimator(tmp_path, write_input):
    h5py = pytest.importorskip("h5py")
    trajs = [(FIN, [1, 1, 1, 0, 0], {}), ("overlap_failure", [1, 0, 0], {})]
    ctx = _context(tmp_path, write_input, trajs, estimator="censored")
    run_stage(ctx)
    with h5py.File(ctx.stage_dir / "populations.h5", "r") as f:
        assert "complete_classical" in f and "complete_quantum_upper" in f
    header = next(
        line
        for line in (ctx.stage_dir / "populations.csv").read_text().splitlines()
        if not line.startswith("#")
    )
    assert "complete_P_S0" in header.split(",")


def test_energy_jump_censoring_keeps_the_part_before_the_jump(tmp_path, write_input):
    jump = [-1.0, -1.0, -1.0 + 0.2 / HARTREE_IN_EV, -1.0 + 0.2 / HARTREE_IN_EV, -1.0]
    trajs = [(FIN, [1, 1, 1, 1, 1], {"e_total": jump}), (FIN, [1, 1, 1, 1, 1], {})]
    ctx = _context(
        tmp_path,
        write_input,
        trajs,
        estimator="censored",
        analysis="energy_jump_eV = 0.1\nexclude_energy_jumps = true\n",
    )
    run_stage(ctx)
    data = _analysis(ctx)
    # the jump is at record 2 (1.0 fs): censored at the last grid time before it, 0.5 fs
    assert data["censored"] == [
        {
            "task": "traj_00000",
            "outcome": "finished",
            "reason": "energy_jump",
            "censored_at_fs": 0.5,
        }
    ]
    assert data["excluded"] == [] and data["n_included"] == 2
    assert data["estimators"]["complete"]["n"] == 1  # the complete rule excludes it


def test_a_trajectory_lost_after_the_horizon_counts_as_complete(tmp_path, write_input):
    # the shortest finished trajectory sets the horizon at 1.0 fs (grid step 2); the lost one
    # has records up to grid step 4, so it is complete up to the horizon (ruling PR5)
    trajs = [(FIN, [1, 1, 0], {}), ("energy_drift", [1, 1, 1, 0, 0], {})]
    ctx = _context(tmp_path, write_input, trajs, estimator="censored")
    run_stage(ctx)
    data = _analysis(ctx)
    assert data["horizon_fs"] == pytest.approx(1.0)
    assert data["n_censored"] == 0 and data["censored"] == [] and data["n_included"] == 2
    assert data["estimators"]["complete"]["n"] == 1


def test_without_a_complete_trajectory_the_horizon_is_capped(tmp_path, write_input):
    # no finished trajectory: the horizon is the last observed grid step, 1.5 fs (traj_00001,
    # energy_drift, records up to grid step 3), not tmax; traj_00001 is then observed through
    # the horizon and counts as complete there (ruling PR5); the complete block stays null
    trajs = [("overlap_failure", [1, 0, 0], {}), ("energy_drift", [1, 1, 1, 1], {})]
    ctx = _context(tmp_path, write_input, trajs, estimator="censored")
    run_stage(ctx)
    data = _analysis(ctx)
    assert data["horizon_fs"] == pytest.approx(1.5) and data["horizon_capped"] is True
    assert data["n_complete"] == 0 and data["n_censored"] == 1 and data["n_included"] == 2
    assert data["estimators"]["complete"] is None and data["consistency"] is None
    assert data["hops"]["pairs"] == {}
    with h5py.File(ctx.stage_dir / "populations.h5", "r") as f:
        assert not any(name.startswith("complete_") for name in f)
        assert f.attrs["estimator"] == "censored" and f.attrs["n_complete"] == 0
        assert f["time_fs"][()][-1] == pytest.approx(1.5)
    assert "complete_P_S0" not in (ctx.stage_dir / "populations.csv").read_text()
    assert CAPPED_WARNING.format(t="1.5") in data["warnings"]
    assert not any("is held constant" in w for w in data["warnings"])
    # traj_00000 is in S0 at its censoring step (1.0 fs) and nobody is at risk in S0 after it
    n_empty = data["estimators"]["censored"]["empty_rows"]
    assert n_empty == 1 and EMPTY_ROWS_WARNING.format(n=n_empty) in data["warnings"]


def test_a_cap_at_zero_stops_the_stage(tmp_path, write_input):
    trajs = [("overlap_failure", [1], {}), ("energy_drift", [1], {})]
    ctx = _context(tmp_path, write_input, trajs, estimator="censored")
    with pytest.raises(StageFailedError, match="none is observed after t = 0"):
        run_stage(ctx)
    data = _analysis(ctx)
    assert data["stage_failed"] is True and data["horizon_capped"] is True
    assert data["error"] == analysis_module.NO_DATA_AFTER_START


def test_with_a_complete_trajectory_the_horizon_is_not_capped(tmp_path, write_input):
    trajs = [(FIN, [1, 1, 1, 1, 1], {}), ("overlap_failure", [1, 0, 0], {})]
    ctx = _context(tmp_path, write_input, trajs, estimator="censored")
    run_stage(ctx)
    data = _analysis(ctx)
    assert data["horizon_capped"] is False and data["horizon_fs"] == pytest.approx(2.0)


def test_the_capped_horizon_never_exceeds_tmax(tmp_path, write_input):
    # final review I1: tmax_fs = 2.0 (n_total 4), e.g. after lowering it; both lost
    # trajectories have records beyond it (grid steps 6 and 5), so the cap is clamped to
    # n_total and both are observed through the horizon, i.e. complete there (ruling PR5)
    trajs = [("overlap_failure", [1, 1, 1, 1, 1, 1, 0], {}), ("energy_drift", [1] * 6, {})]
    ctx = _context(tmp_path, write_input, trajs, estimator="censored")
    run_stage(ctx)
    data = _analysis(ctx)
    assert data["horizon_fs"] == pytest.approx(2.0) and data["horizon_capped"] is True
    assert data["n_censored"] == 0 and data["censored"] == [] and data["n_included"] == 2
    assert CAPPED_WARNING.format(t="2") in data["warnings"]


def _estimate(tau_fs):
    return types.SimpleNamespace(
        fit=types.SimpleNamespace(fit=types.SimpleNamespace(tau_fs=tau_fs))
    )


def test_tau_range_takes_the_extremes_of_the_fitted_estimators():
    blocks = {
        "complete": _estimate(80.0),
        "censored": _estimate(62.5),
        "hold_in_s0": _estimate(70.0),
        "other": _estimate(None),  # not fitted: ignored
    }
    assert analysis_module._tau_range(blocks) == {
        "min": 62.5,
        "max": 80.0,
        "min_estimator": "censored",
        "max_estimator": "complete",
    }


@pytest.mark.parametrize(
    "blocks",
    [
        {},
        {"censored": _estimate(62.5)},
        {"complete": None, "censored": _estimate(62.5), "hold_in_s0": _estimate(None)},
    ],
)
def test_tau_range_needs_two_fitted_estimators(blocks):
    assert analysis_module._tau_range(blocks) is None


CAPPED_WARNING = (
    "no trajectory is complete: the analysis horizon is capped at {t} fs, the last time any "
    "trajectory is observed"
)
EMPTY_ROWS_WARNING = (
    "the censored estimator held {n} state probabilities over intervals with no trajectory at "
    "risk (empty_rows = {n}); the censored curve and its fit are not supported by data there"
)


def test_held_probabilities_are_warned_about_with_complete_trajectories(tmp_path, write_input):
    # the complete trajectory stays in S1; the censored one is in S0 at its censoring step
    # (1.0 fs), so S0 has probability and nobody at risk over the last two intervals
    trajs = [(FIN, [1, 1, 1, 1, 1], {}), ("overlap_failure", [1, 0, 0], {})]
    for estimator in ("censored", "complete"):
        ctx = _context(tmp_path / estimator, write_input, trajs, estimator=estimator)
        run_stage(ctx)
        data = _analysis(ctx)
        assert data["estimators"]["censored"]["empty_rows"] == 2
        assert EMPTY_ROWS_WARNING.format(n=2) in data["warnings"], estimator
        assert not any(w.startswith("no trajectory is complete") for w in data["warnings"])
        assert "- empty_rows: 2\n" in (ctx.stage_dir / "report.md").read_text()


def test_no_held_probability_no_warning(tmp_path, write_input):
    ctx = _context(tmp_path, write_input, _varied_trajectories(), estimator="censored")
    run_stage(ctx)
    data = _analysis(ctx)
    assert data["estimators"]["censored"]["empty_rows"] == 0
    assert not any("held" in w for w in data["warnings"])
    assert "- empty_rows: 0\n" in (ctx.stage_dir / "report.md").read_text()


def test_a_damaged_censored_trajectory_stops_a_complete_analysis(tmp_path, write_input):
    # final review T5a (ruling T5-1): both estimators are computed, so a censored trajectory is
    # checked also under estimator = "complete", where main would have excluded it unread
    damaged = [-1.0, float("nan"), -1.0]
    trajs = [(FIN, [1, 1, 1, 1, 1], {}), ("overlap_failure", [1, 0, 0], {"e_total": damaged})]
    ctx = _context(tmp_path, write_input, trajs, estimator="complete")
    with pytest.raises(TaskError, match="non-finite e_total in an included trajectory") as info:
        run_stage(ctx)
    assert info.value.retryable is False
    assert "traj_00001" in str(info.value) and "--rerun-from dynamics" in str(info.value)
    assert not (ctx.stage_dir / "analysis.json").exists()  # a task error, not a recorded stop


def _baseline_trajectories():
    """Six finished trajectories with hops and graded quantum populations, plus two losses."""
    actives = [
        [1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
    ]
    trajs = []
    for k, active in enumerate(actives):
        p1 = [max(0.0, 1.0 - m * (0.1 + 0.05 * k)) for m in range(5)]
        trajs.append((FIN, active, {"populations": [[1.0 - p, p] for p in p1]}))
    trajs.append(("overlap_failure", [1, 0, 0], {}))
    trajs.append(("energy_drift", [1, 1], {}))
    return trajs


# Final review T5b: produced on main f738b35 (before the censored analysis) by running this
# scenario through main's AnalysisStage (tests/stages/test_analysis.py `_context`/`run_stage`,
# seed 2026, n_bootstrap 100) and printing repr floats; the branch must reproduce them under
# estimator = "complete". The counts and the classical populations (fractions of trajectories)
# are compared exactly. The Wilson bounds and quantum averages are compared to a relative
# 1e-9, the fit values (estimates, intervals, std, rms) to a relative 1e-6: environment.yml does
# not pin NumPy or SciPy, and OpenBLAS picks its kernels by CPU, so the last bits differ between
# machines; least_squares then stops at a slightly different point within its own tolerance
# (t0 moved by 2e-9 relative with OPENBLAS_CORETYPE=SandyBridge, measured 2026-11-20). A NumPy
# release that changes the stream of Generator.integers changes the bootstrap resamples
# themselves; then regenerate these numbers on the old code rather than widen the tolerance.
REL = 1e-9
FIT_REL = 1e-6
ABS = 1e-12
BASELINE_P_S1 = [1.0, 0.8333333333333334, 0.6666666666666666, 0.5, 0.3333333333333333]
BASELINE_P_S1_LOWER = [
    0.6096657120978346,
    0.43649717781352987,
    0.299993315138392,
    0.1876163064826506,
    0.09677141110578041,
]
BASELINE_P_S1_UPPER = [
    1.0,
    0.9699466302516935,
    0.9032285888942195,
    0.8123836935173494,
    0.700006684861608,
]
BASELINE_RHO_S1 = [1.0, 0.775, 0.55, 0.3333333333333333, 0.19999999999999998]
BASELINE_RHO_S1_LOWER = [
    1.0,
    0.7039583333333335,
    0.4079166666666667,
    0.13729166666666665,
    0.03333333333333332,
]
BASELINE_RHO_S1_UPPER = [
    1.0,
    0.8460416666666667,
    0.6920833333333333,
    0.5381249999999997,
    0.39999999999999997,
]
BASELINE_FIT = {
    "classical": {
        "A": 1.0,
        "t0_fs": 0.2119968412487169,
        "tau_fs": 1.7687999587775307,
        "t0_interval_fs": [1.0524126595077609e-18, 1.5959787372074765],
        "tau_interval_fs": [0.3479278723827433, 3.57966856953346],
        "t0_std_fs": 0.4749146353703487,
        "tau_std_fs": 0.9034104838892965,
        "rms": 0.020903470034306416,
        "flags": [],
        "bootstrap_failures": 0,
    },
    "quantum": {
        "A": 1.0,
        "t0_fs": 0.22031920968379914,
        "tau_fs": 1.179935785581576,
        "t0_interval_fs": [0.162871532680283, 0.28117708361593835],
        "tau_interval_fs": [0.6963675824051245, 2.0707755783289676],
        "t0_std_fs": 0.03444332309211499,
        "tau_std_fs": 0.3732270255036135,
        "rms": 0.01895484498317354,
        "flags": [],
        "bootstrap_failures": 0,
    },
}


@pytest.mark.parametrize("population", ["classical", "quantum"])
def test_the_complete_estimator_reproduces_main(tmp_path, write_input, population):
    ctx = _context(
        tmp_path,
        write_input,
        _baseline_trajectories(),
        seed=2026,
        analysis=f'fit_population = "{population}"\n',
        estimator="complete",
    )
    run_stage(ctx)
    data = _analysis(ctx)
    assert data["n_included"] == 6 and data["horizon_fs"] == 2.0
    assert data["included"] == [f"traj_{k:05d}" for k in range(6)]
    for key, expected in BASELINE_FIT[population].items():
        got = data["fit"][key]
        if key in ("flags", "bootstrap_failures"):
            assert got == expected, key
        else:
            assert got == pytest.approx(expected, rel=FIT_REL, abs=ABS), key

    def close(values, expected):
        return values == pytest.approx(expected, rel=REL, abs=ABS)

    with h5py.File(ctx.stage_dir / "populations.h5") as f:
        assert f["classical"][()][:, 1].tolist() == BASELINE_P_S1
        assert close(f["classical_lower"][()][:, 1].tolist(), BASELINE_P_S1_LOWER)
        assert close(f["classical_upper"][()][:, 1].tolist(), BASELINE_P_S1_UPPER)
        assert close(f["quantum"][()][:, 1].tolist(), BASELINE_RHO_S1)
        assert close(f["quantum_lower"][()][:, 1].tolist(), BASELINE_RHO_S1_LOWER)
        assert close(f["quantum_upper"][()][:, 1].tolist(), BASELINE_RHO_S1_UPPER)
        if population == "classical":
            assert f["fit_target"][()].tolist() == BASELINE_P_S1
        else:
            assert close(f["fit_target"][()].tolist(), BASELINE_RHO_S1)


def test_the_hold_in_s0_block_and_the_lifetime_range(tmp_path, write_input):
    trajs = [
        (FIN, [1, 1, 1, 1, 1], {}),
        (FIN, [1, 1, 1, 0, 0], {}),
        ("overlap_failure", [1, 1, 1], {}),  # lost in S1 at 1.0 fs: held in S0 after it
    ]
    ctx = _context(tmp_path, write_input, trajs, estimator="censored")
    run_stage(ctx)
    data = _analysis(ctx)
    block = data["estimators"]["hold_in_s0"]
    assert block["n"] == 3 and block["rule"] == analysis_module.HOLD_RULE
    row = next(r for r in block["table"] if r["time_fs"] == 1.5)
    assert row["P"]["S0"] == pytest.approx(2 / 3)  # traj_00001 decayed, traj_00002 held
    names = [n for n in ("complete", "censored", "hold_in_s0") if data["estimators"][n]]
    taus = {n: data["estimators"][n]["fit"]["tau_fs"] for n in names}
    taus = {n: t for n, t in taus.items() if t is not None}
    r = data["tau_range_fs"]
    if len(taus) >= 2:
        assert r["min"] == min(taus.values()) and r["max"] == max(taus.values())
        assert taus[r["min_estimator"]] == r["min"] and taus[r["max_estimator"]] == r["max"]
    else:
        assert r is None
    header = next(
        line
        for line in (ctx.stage_dir / "populations.csv").read_text().splitlines()
        if not line.startswith("#")
    ).split(",")
    assert "hold_in_s0_P_S0" in header and "hold_in_s0_rho_S1_upper" in header
    with h5py.File(ctx.stage_dir / "populations.h5", "r") as f:
        assert "hold_in_s0_classical" in f and "hold_in_s0_quantum_upper" in f
    assert "## Lifetime range" in (ctx.stage_dir / "report.md").read_text()


def test_without_a_censored_trajectory_there_is_no_hold_in_s0_block(tmp_path, write_input):
    ctx = _context(tmp_path, write_input, _varied_trajectories(), estimator="censored")
    run_stage(ctx)
    data = _analysis(ctx)
    assert data["estimators"]["hold_in_s0"] is None
    assert "hold_in_s0_P_S0" not in (ctx.stage_dir / "populations.csv").read_text()


# ------------------------------------------------- triplets (sub-project 3, spec 4.3)
ELECTRONIC_SOC = ELECTRONIC + "triplets = 2\n"
DYNAMICS_SOC = DYNAMICS + 'gradient_transformation = "none"\n'
SOC_LABELS = ("S0", "S1", "T1", "T2")


def _soc_context(tmp_path, write_input, trajs, **kwargs):
    specs = [
        None if t is None else (t[0], t[1], {"labels": SOC_LABELS, "triplets": 2, **t[2]})
        for t in trajs
    ]
    return _context(
        tmp_path, write_input, specs, electronic=ELECTRONIC_SOC, dynamics=DYNAMICS_SOC, **kwargs
    )


SOC_TRAJS = [
    # S1 -> T1 at a hop, T1 -> T2 without a hop, T2 -> S0 at a hop
    (FIN, [1, 1, 2, 3, 0], {"hops": [2, 4]}),
    # S1 -> T1 at a hop, then a hop within T1 (between its Ms components): no transition
    (FIN, [1, 2, 2, 2, 2], {"hops": [1, 2]}),
    (FIN, [1, 1, 1, 1, 1], {}),
]


def test_a_triplet_project_reports_its_spin_free_transitions(tmp_path, write_input):
    ctx = _soc_context(tmp_path, write_input, SOC_TRAJS)
    run_stage(ctx)
    data = _analysis(ctx)
    t = data["transitions"]
    assert t["kinds"] == {
        "S-S": {"at_hop": 0, "without_hop": 0},
        "S-T": {"at_hop": 2, "without_hop": 0},
        "T-S": {"at_hop": 1, "without_hop": 0},
        "T-T": {"at_hop": 0, "without_hop": 1},
    }
    assert t["pairs"] == {"S1->T1": 2, "T1->T2": 1, "T2->S0": 1}
    assert t["first_singlet_to_triplet_fs"] == {
        "per_trajectory": {"traj_00000": 1.0, "traj_00001": 0.5, "traj_00002": None},
        "n_crossed": 2,
        "n_censored": 1,
        "mean": 0.75,
        "median": 0.75,
    }
    # the hop table keeps the hop within T1 (ledger ruling)
    assert data["hops"]["pairs"] == {"S1->T1": 2, "T1->T1": 1, "T2->S0": 1}
    report_md = (ctx.stage_dir / "report.md").read_text()
    assert "\n## Intersystem crossing\n" in report_md
    assert "- S-T: at hop 2, without hop 0\n" in report_md
    assert "- first S-to-T time (fs): n_crossed 2, n_censored 1, mean 0.75, median 0.75" in (
        report_md
    )


def test_a_triplet_project_fits_the_excited_singlets_by_default(tmp_path, write_input):
    ctx = _soc_context(tmp_path, write_input, SOC_TRAJS)
    run_stage(ctx)
    data = _analysis(ctx)
    assert data["fit"]["states"] == ["S1"]  # owner decision 6: not S1, T1, T2
    assert list(data["table"][0]["P"]) == list(SOC_LABELS)
    with h5py.File(ctx.stage_dir / "populations.h5") as f:
        np.testing.assert_array_equal(f["fit_target"][()], f["classical"][()][:, 1])
        assert list(f.attrs["fit_states"]) == ["S1"]
        np.testing.assert_allclose(f["classical"][()][:, 2], [0, 1 / 3, 2 / 3, 1 / 3, 1 / 3])


def test_an_explicit_fit_states_is_kept_with_triplets(tmp_path, write_input):
    ctx = _soc_context(tmp_path, write_input, SOC_TRAJS, analysis='fit_states = ["S1"]\n')
    run_stage(ctx)
    assert _analysis(ctx)["fit"]["states"] == ["S1"]


@pytest.mark.parametrize(
    ("labels", "n_singlets", "n_triplets", "expected"),
    [
        (["S0", "S1", "T1", "T2"], 2, 2, ["S1"]),
        (["S0", "S1", "S2", "T1", "T2", "T3"], 3, 3, ["S1", "S2"]),
        (["S0", "S1", "S2"], 3, 0, ["S1", "S2"]),  # without triplets: labels[1:]
        (["S0", "S1"], 2, 0, ["S1"]),
    ],
)
def test_the_default_fit_states(labels, n_singlets, n_triplets, expected):
    assert analysis_module._default_fit_states(labels, n_singlets, n_triplets) == expected


def test_a_singlet_project_has_no_transitions(tmp_path, write_input):
    ctx = _context(tmp_path, write_input, [(FIN, [1, 1, 0, 0, 0], {})])
    run_stage(ctx)
    assert "transitions" not in _analysis(ctx)
    assert "Intersystem crossing" not in (ctx.stage_dir / "report.md").read_text()


def test_a_stopped_triplet_analysis_has_a_null_transitions_key(tmp_path, write_input):
    ctx = _soc_context(tmp_path, write_input, [("energy_drift", [1, 1], {})])
    with pytest.raises(StageFailedError):
        run_stage(ctx)
    data = _analysis(ctx)
    assert "transitions" in data and data["transitions"] is None
    assert "## Intersystem crossing\nnot computed\n" in (ctx.stage_dir / "report.md").read_text()


@pytest.mark.parametrize(("attr", "value"), [("n_singlets", 3), ("n_triplets", 1)])
def test_spin_counts_that_do_not_fit_the_labels_are_a_task_error(
    tmp_path, write_input, attr, value
):
    ctx = _soc_context(tmp_path, write_input, SOC_TRAJS)
    path = ctx.upstream["dynamics"] / "tasks" / "traj_00001" / "output.h5"
    with h5py.File(path, "a") as f:
        f.attrs[attr] = value
    with pytest.raises(TaskError, match="--rerun-from dynamics") as info:
        run_stage(ctx)
    assert attr in str(info.value) and "traj_00001" in str(info.value)
    assert info.value.retryable is False
