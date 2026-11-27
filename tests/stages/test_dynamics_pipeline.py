"""The dynamics stage under the orchestrator, with the canned LVC pipeline."""

import json
from pathlib import Path

import pytest

from gqteashmd.cli import main
from gqteashmd.project.state import StageStatus, StateStore, TaskStatus
from gqteashmd.testing import canned_lvc

LVC = Path(__file__).resolve().parents[1] / "data" / "lvc" / "lvc_singlets.template"
CANNED = "gqteashmd.testing.pipelines:canned_dynamics"
FAST = "poll_interval_s = 0.05\nstop_grace_period_s = 2.0"
ELECTRONIC = '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 3\n'


@pytest.fixture(autouse=True)
def _template(monkeypatch):
    monkeypatch.setenv(canned_lvc.LVC_TEMPLATE_ENV, str(LVC))


def _input(tmp_path, write_input, tmax="2.0", extra=""):
    return write_input(
        tmp_path,
        execution=FAST,
        extra=ELECTRONIC + f"[dynamics]\ntmax_fs = {tmax}\ndt_fs = 0.25\n" + extra,
    )


def _main(path, *args):
    return main(["run", str(path), "--pipeline", CANNED, *args])


def _stage_dir(tmp_path):
    return tmp_path / "run" / "03_dynamics"


def _summary(tmp_path):
    return json.loads((_stage_dir(tmp_path) / "dynamics_summary.json").read_text())


def test_canned_run_and_second_run_is_a_no_op(tmp_path, write_input):
    path = _input(tmp_path, write_input)
    assert _main(path) == 0
    summary = _summary(tmp_path)
    assert summary["outcomes"]["finished"] == 3 and summary["missing_output"] == []
    log = _stage_dir(tmp_path) / "tasks" / "traj_00000" / "steps.log"
    before = log.stat().st_mtime_ns
    assert _main(path) == 0
    assert log.stat().st_mtime_ns == before


def test_a_killed_worker_resumes_to_the_same_trajectory(tmp_path, write_input, monkeypatch):
    ref_root, run_root = tmp_path / "ref", tmp_path / "kill"
    ref_root.mkdir()
    run_root.mkdir()
    assert _main(_input(ref_root, write_input)) == 0
    monkeypatch.setenv(canned_lvc.FAULTS_ENV, json.dumps({"traj_00001": {"5": "kill"}}))
    assert _main(_input(run_root, write_input)) == 0
    task = Path("tasks") / "traj_00001" / "steps.log"
    assert (_stage_dir(run_root) / "tasks" / "traj_00001" / "killed_once").exists()
    assert (_stage_dir(run_root) / task).read_bytes() == (_stage_dir(ref_root) / task).read_bytes()


def test_raising_tmax_extends_and_changing_dt_is_stale(tmp_path, write_input):
    path = _input(tmp_path, write_input)
    assert _main(path) == 0
    path = _input(tmp_path, write_input, tmax="3.0")
    assert _main(path) == 0
    finals = [t["t_final_fs"] for t in _summary(tmp_path)["trajectories"]]
    assert len(finals) == 3 and finals == [pytest.approx(3.0)] * 3
    stale = write_input(
        tmp_path,
        execution=FAST,
        extra=ELECTRONIC + "[dynamics]\ntmax_fs = 3.0\ndt_fs = 0.5\n",
    )
    assert _main(stale) == 3


def test_raising_tmax_after_a_stop_extends_every_trajectory_in_one_run(tmp_path, write_input):
    path = _input(tmp_path, write_input)
    assert _main(path) == 0
    # The state after a user stop: the stage PENDING, one trajectory PENDING, two DONE.
    with StateStore(tmp_path / "run" / ".gqteashmd" / "state.db") as state:
        state.set_stage("dynamics", StageStatus.PENDING, message="stopped by user")
        state.set_task("dynamics", "traj_00002", TaskStatus.PENDING)
    path = _input(tmp_path, write_input, tmax="3.0")
    assert _main(path) == 0
    summary = _summary(tmp_path)
    assert summary["tmax_fs"] == 3.0
    finals = [t["t_final_fs"] for t in summary["trajectories"]]
    assert len(finals) == 3 and finals == [pytest.approx(3.0)] * 3


def test_without_a_dynamics_table_the_stage_is_skipped(tmp_path, write_input):
    path = write_input(tmp_path, execution=FAST, extra=ELECTRONIC)
    assert _main(path) == 0
    assert not (_stage_dir(tmp_path) / "dynamics_summary.json").exists()


def test_validate_rejects_nacdr_without_casscf(tmp_path, write_input):
    path = _input(tmp_path, write_input, extra='coupling = "nacdr"\n')
    assert main(["validate", str(path)]) == 2


# ---------------------------------------------------------------- triplets (sub-project 3)
LVC_SOC = Path(__file__).resolve().parents[1] / "data" / "lvc" / "lvc_soc.template"
ELECTRONIC_SOC = (
    '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 2\n'
    "triplets = 2\n"
)
SOC_LABELS = ["S0", "S1", "T1", "T2"]
SOC_TASKS = ["traj_00000", "traj_00001", "traj_00002", "traj_00003"]


@pytest.fixture
def soc(monkeypatch):
    """The canned spin-orbit pipeline (plan ruling PR5): lvc_soc.template, 2 singlets + 2
    triplets, four trajectories starting in S1."""
    monkeypatch.setenv(canned_lvc.LVC_TEMPLATE_ENV, str(LVC_SOC))
    monkeypatch.setenv(canned_lvc.LVC_SOC_ENV, "1")


def _soc_input(root, write_input, tmax="20.0"):
    return write_input(
        root,
        execution=FAST,
        extra=ELECTRONIC_SOC
        + f'[dynamics]\ntmax_fs = {tmax}\ndt_fs = 0.25\ngradient_transformation = "none"\n',
    )


def _output(root, task):
    return _stage_dir(root) / "tasks" / task / "output.h5"


def _read_h5(path):
    import h5py

    with h5py.File(path, "r") as f:
        return dict(f.attrs), {name: f[name][()] for name in f}


def _direct(root, path, task_index, n_steps):
    """The canned spin-orbit LVC model run by the core alone (no adapter, runner or stage),
    from the stage's initial conditions with the trajectory's task_rng stream: the spin-free
    populations and positions of every record, t = 0 included."""
    import numpy as np

    from gqteashmd.config.loader import load_config
    from gqteashmd.core.units import AU_TIME_IN_FS
    from gqteashmd.dynamics.driver import initial_state, run_trajectory
    from gqteashmd.dynamics.randomness import GeneratorSource
    from gqteashmd.dynamics.representation import mch_coefficients
    from gqteashmd.stages.dynamics import run_settings
    from gqteashmd.trajectory.spin_expansion import spin_free_populations
    from gqteashmd.util.seeds import task_rng
    from gqteashmd.vib.artifacts import read_samples_h5

    config = load_config(path).config
    options = run_settings(config.dynamics).options
    assert options.dt_au == pytest.approx(0.25 / AU_TIME_IN_FS)
    model = canned_lvc.canned_model()
    samples = read_samples_h5(root / "run" / "01_sampling" / "samples.h5")
    entry = canned_lvc.canned_selection()[task_index]
    q0 = np.asarray(samples["positions_bohr"][entry[0] - 1], dtype=float).reshape(-1)
    v0 = np.asarray(samples["velocities_au"][entry[0] - 1], dtype=float).reshape(-1)
    state = initial_state(model, q0, v0, 1, options=options)  # S1: expanded MCH index 1
    records: list = []
    source = GeneratorSource(task_rng(config.project.seed, "dynamics", task_index))
    run_trajectory(model, state, options, source, max_steps=n_steps, on_step=records.append)
    populations = [spin_free_populations(np.abs(mch_coefficients(state.u, state.c)) ** 2, 2, 2)]
    populations += [spin_free_populations(np.abs(r.c_mch) ** 2, 2, 2) for r in records]
    positions = [q0] + [r.q for r in records]
    return np.array(populations), np.array(positions)


def _assert_matches_the_direct_run(root, path, n_steps):
    import numpy as np

    summary = _summary(root)
    for k, task in enumerate(SOC_TASKS):
        row = next(r for r in summary["trajectories"] if r["task"] == task)
        assert row["outcome"] == "finished" and row["n_rejections"] == 0  # one step per record
        _, data = _read_h5(_output(root, task))
        populations, positions = _direct(root, path, k, n_steps)
        assert data["populations"].shape == populations.shape == (n_steps + 1, 4)
        np.testing.assert_allclose(data["populations"], populations, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            data["positions"].reshape(n_steps + 1, -1), positions, rtol=0, atol=1e-10
        )


def test_the_canned_spin_orbit_pipeline_runs_and_follows_the_model(tmp_path, write_input, soc):
    path = _soc_input(tmp_path, write_input)
    assert _main(path) == 0
    summary = _summary(tmp_path)
    assert summary["outcomes"]["finished"] == 4 and summary["missing_output"] == []
    rows = {r["task"]: r for r in summary["trajectories"]}
    assert sorted(rows) == SOC_TASKS
    for task in SOC_TASKS:
        assert rows[task]["initial_state"] == "S1"
        assert rows[task]["final_state"] in SOC_LABELS  # PR6: the active spin-free state
        attrs, data = _read_h5(_output(tmp_path, task))
        assert attrs["schema_version"] == 2
        assert [str(x) for x in attrs["state_labels"]] == SOC_LABELS
        assert (attrs["n_singlets"], attrs["n_triplets"]) == (2, 2)
        assert attrs["soc_operator"] == "somf"
        assert rows[task]["final_state"] == SOC_LABELS[int(data["active"][-1])]
        assert data["populations"][:, 2:].sum(axis=1).max() > 1e-3  # the coupling acts
    markdown = (_stage_dir(tmp_path) / "dynamics_summary.md").read_text()
    assert "| final state |" in markdown
    _assert_matches_the_direct_run(tmp_path, path, 80)


@pytest.mark.slow
def test_a_killed_spin_orbit_worker_resumes_to_the_same_output(
    tmp_path, write_input, soc, monkeypatch
):
    ref_root, run_root = tmp_path / "ref", tmp_path / "kill"
    ref_root.mkdir()
    run_root.mkdir()
    assert _main(_soc_input(ref_root, write_input)) == 0
    monkeypatch.setenv(canned_lvc.FAULTS_ENV, json.dumps({"traj_00001": {"30": "kill"}}))
    assert _main(_soc_input(run_root, write_input)) == 0
    assert (_stage_dir(run_root) / "tasks" / "traj_00001" / "killed_once").exists()
    assert _summary(run_root)["trajectories"][1]["n_resumes"] >= 1
    for task in SOC_TASKS:
        assert _output(run_root, task).read_bytes() == _output(ref_root, task).read_bytes()


def test_raising_tmax_extends_the_spin_orbit_trajectories(tmp_path, write_input, soc):
    assert _main(_soc_input(tmp_path, write_input, tmax="10.0")) == 0
    path = _soc_input(tmp_path, write_input)
    assert _main(path) == 0
    finals = [t["t_final_fs"] for t in _summary(tmp_path)["trajectories"]]
    assert finals == [pytest.approx(20.0)] * 4
    _assert_matches_the_direct_run(tmp_path, path, 80)


PAUSE = '[workflow]\npause_after = ["selection"]\n'


def test_pause_after_selection_does_not_pause_without_dynamics(tmp_path, write_input):
    path = write_input(tmp_path, execution=FAST, extra=PAUSE + ELECTRONIC)
    assert _main(path) == 0


def test_pause_after_selection_pauses_when_dynamics_is_enabled(tmp_path, write_input):
    path = write_input(
        tmp_path,
        execution=FAST,
        extra=PAUSE + ELECTRONIC + "[dynamics]\ntmax_fs = 2.0\ndt_fs = 0.25\n",
    )
    assert _main(path) == 5
    assert not (_stage_dir(tmp_path) / "dynamics_summary.json").exists()
    assert _main(path) == 0
    assert _summary(tmp_path)["outcomes"]["finished"] == 3
