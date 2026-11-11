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
