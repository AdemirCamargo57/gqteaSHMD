import json
import subprocess
import sys
import time

import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import ConfigError, ProjectError, StageFailedError, StaleStageError
from gqteashmd.orchestrator import Orchestrator
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.project.lock import ProjectLock
from gqteashmd.project.state import StageStatus, StateStore, TaskStatus
from gqteashmd.testing.fake_stages import BEHAVIOUR_ENV
from gqteashmd.util.fileio import read_json

FAKE = "gqteashmd.testing.fake_stages:make_pipeline"
FAST = "poll_interval_s = 0.05\nstop_grace_period_s = 2.0"
ALL = ["prepare", "optimize", "frequencies", "sampling", "dynamics"]


@pytest.fixture
def behaviour(monkeypatch):
    def _set(**kwargs):
        monkeypatch.setenv(BEHAVIOUR_ENV, json.dumps(kwargs))

    _set()
    return _set


@pytest.fixture
def project(tmp_path, write_input):
    return write_input(tmp_path, execution=FAST)


def _run(path, **kwargs):
    return Orchestrator(load_config(path), pipeline_ref=FAKE, **kwargs).run()


def _state(path):
    return StateStore(path.parent / "run" / ".gqteashmd" / "state.db")


def _run_dir(path):
    return path.parent / "run"


def test_full_run_completes_all_stages(project, behaviour):
    summary = _run(project)
    assert summary.completed == ALL
    assert summary.stopped is False
    assert read_json(_run_dir(project) / "04_dynamics" / "summary.json") == {"values": [0, 1, 2, 3]}
    assert (_run_dir(project) / "00_prepare" / "geometry.xyz").exists()
    with _state(project) as state:
        assert all(state.get_stage(name).status == StageStatus.DONE for name in ALL)


def test_second_run_is_a_no_op(project, behaviour):
    _run(project)
    summary = _run(project)
    assert summary.completed == []
    assert summary.up_to_date == ALL


def test_crashed_task_is_retried(project, behaviour):
    behaviour(crash_once=["sampling:t001"])
    _run(project)
    with _state(project) as state:
        task = state.get_task("sampling", "t001")
    assert (task.status, task.attempts) == (TaskStatus.DONE, 2)


def test_retryable_error_is_retried(project, behaviour):
    behaviour(retryable_once=["dynamics:t000"])
    _run(project)
    with _state(project) as state:
        task = state.get_task("dynamics", "t000")
    assert (task.status, task.attempts) == (TaskStatus.DONE, 2)


def test_tolerated_failures_do_not_stop_the_workflow(project, behaviour):
    behaviour(fail=["dynamics:t002"])
    summary = _run(project)
    assert summary.completed == ALL
    assert read_json(_run_dir(project) / "04_dynamics" / "summary.json") == {"values": [0, 1, 3]}


def test_intolerable_failure_stops_and_rerun_retries(project, behaviour):
    behaviour(fail=["sampling:t000"])
    with pytest.raises(StageFailedError, match=r"(?s)sampling.*t000.*worker\.log"):
        _run(project)
    with _state(project) as state:
        assert state.get_stage("sampling").status == StageStatus.FAILED
        assert state.get_stage("dynamics") is None
    behaviour()
    summary = _run(project)
    assert summary.completed == ["sampling", "dynamics"]


def test_changed_input_is_detected_and_rerun_from_recomputes(tmp_path, write_input, behaviour):
    path = write_input(tmp_path, execution=FAST)
    _run(path)
    path = write_input(tmp_path, execution=FAST, molecule="charge = 2")
    with pytest.raises(StaleStageError, match="--rerun-from prepare"):
        _run(path)
    summary = _run(path, rerun_from="prepare")
    assert summary.completed == ALL
    assert any((tmp_path / "run" / ".gqteashmd" / "superseded").iterdir())
    assert read_json(tmp_path / "run" / "00_prepare" / "molecule.json")["charge"] == 2


def test_disabled_stage_is_skipped(project, behaviour):
    behaviour(skip_optimize=True)
    summary = _run(project)
    assert summary.skipped == ["optimize"]
    assert "optimize" not in summary.completed
    with _state(project) as state:
        assert state.get_stage("optimize").status == StageStatus.SKIPPED


def test_stop_request_pauses_and_resume_completes(project, behaviour):
    behaviour(sleep=0.3)
    marker = _run_dir(project) / "03_sampling" / "tasks" / "t000" / "result.json"
    summary = Orchestrator(load_config(project), pipeline_ref=FAKE).run(should_stop=marker.exists)
    assert summary.stopped is True
    with _state(project) as state:
        assert state.get_stage("sampling").status == StageStatus.PENDING
        assert state.get_stage("dynamics") is None
    behaviour()
    summary = _run(project)
    assert summary.stopped is False
    assert "dynamics" in summary.completed


def test_second_controller_is_refused(project, behaviour):
    layout = ProjectLayout(load_config(project).config.project.directory)
    layout.ensure()
    with (
        ProjectLock(layout.lock_file),
        pytest.raises(ProjectError, match="Another gqteashmd controller"),
    ):
        _run(project)


def test_unknown_rerun_stage_is_config_error(project, behaviour):
    with pytest.raises(ConfigError, match="frobnicate"):
        Orchestrator(load_config(project), pipeline_ref=FAKE, rerun_from="frobnicate")


def test_rerun_from_terminates_orphaned_workers(project, behaviour):
    _run(project)
    task_file = _run_dir(project) / "03_sampling" / "tasks" / "t000" / "task.json"
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", str(task_file)],
        start_new_session=True,
    )
    try:
        with _state(project) as state:
            state.set_stage("sampling", StageStatus.RUNNING)
            state.set_task("sampling", "t000", TaskStatus.RUNNING, pid=proc.pid)
        start = time.monotonic()
        summary = Orchestrator(load_config(project), pipeline_ref=FAKE, rerun_from="sampling").run()
        assert proc.wait(timeout=10) is not None
        assert "dynamics" in summary.completed
        assert time.monotonic() - start < 30
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
