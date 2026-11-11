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


def test_plan_error_marks_stage_failed_and_rerun_is_not_stale(project, behaviour):
    behaviour(plan_raises=["sampling"])
    with pytest.raises(RuntimeError, match="plan exploded"):
        _run(project)
    with _state(project) as state:
        assert state.get_stage("sampling").status == StageStatus.FAILED
        assert "RuntimeError" in state.get_stage("sampling").message
    behaviour()
    summary = _run(project)
    assert summary.completed == ["sampling", "dynamics"]


def test_stopped_stage_with_finished_tasks_is_protected_from_changed_inputs(
    tmp_path, write_input, behaviour
):
    path = write_input(tmp_path, execution=FAST)
    behaviour(sleep=0.3)
    marker = _run_dir(path) / "03_sampling" / "tasks" / "t000" / "result.json"
    summary = Orchestrator(load_config(path), pipeline_ref=FAKE).run(should_stop=marker.exists)
    assert summary.stopped is True
    with _state(path) as state:
        assert state.get_stage("sampling").status == StageStatus.PENDING
        assert state.task_counts("sampling")[TaskStatus.DONE] >= 1
    behaviour()
    with _state(path) as state:  # simulate changed inputs that reach only this stage
        state.set_stage("sampling", StageStatus.PENDING, fingerprint="from-older-inputs")
    with pytest.raises(StaleStageError, match="--rerun-from sampling"):
        _run(path)
    assert _run(path, rerun_from="sampling").completed == ["sampling", "dynamics"]


def test_orphaned_worker_is_reconciled_on_a_normal_run(project, behaviour):
    _run(project)
    task_file = _run_dir(project) / "03_sampling" / "tasks" / "t000" / "task.json"
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", str(task_file)],
        start_new_session=True,
    )
    try:
        with _state(project) as state:
            before = state.get_task("sampling", "t000").attempts
            state.set_stage("sampling", StageStatus.RUNNING)
            state.set_task("sampling", "t000", TaskStatus.RUNNING, pid=proc.pid)
        summary = _run(project)
        assert proc.wait(timeout=10) is not None
        assert "sampling" in summary.completed
        with _state(project) as state:
            task = state.get_task("sampling", "t000")
        assert task.status == TaskStatus.DONE
        assert task.attempts <= before + 1
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_attempts_exhaustion_fails_the_stage(tmp_path, write_input, behaviour):
    path = write_input(tmp_path, execution=FAST + "\nmax_attempts = 1")
    behaviour(crash_once=["sampling:t001"])
    with pytest.raises(StageFailedError):
        _run(path)
    with _state(path) as state:
        task = state.get_task("sampling", "t001")
    assert (task.status, task.attempts) == (TaskStatus.FAILED, 1)


def test_stage_resource_override_reaches_workers(tmp_path, write_input, behaviour):
    path = write_input(
        tmp_path, execution=FAST, resources="[resources.stages.sampling]\nthreads_per_task = 2"
    )
    _run(path)
    run = tmp_path / "run"
    assert (
        read_json(run / "03_sampling" / "tasks" / "t000" / "result.json")["result"]["threads"]
        == "2"
    )
    assert (
        read_json(run / "04_dynamics" / "tasks" / "t000" / "result.json")["result"]["threads"]
        == "1"
    )


def test_unknown_stage_in_resource_overrides_is_config_error(tmp_path, write_input, behaviour):
    path = write_input(
        tmp_path, execution=FAST, resources="[resources.stages.nosuch]\nthreads_per_task = 1"
    )
    with pytest.raises(ConfigError, match="nosuch"):
        Orchestrator(load_config(path), pipeline_ref=FAKE)


PAUSE = '[workflow]\npause_after = ["sampling"]'


def test_pause_after_stops_after_the_stage_and_resumes(tmp_path, write_input, behaviour):
    path = write_input(tmp_path, execution=FAST, extra=PAUSE)
    first = _run(path)
    assert first.paused_after == "sampling"
    assert first.completed[-1] == "sampling"
    with _state(path) as state:
        assert state.get_stage("dynamics") is None
    second = _run(path)
    assert second.paused_after is None
    assert second.completed == ["dynamics"]


def test_pause_after_the_last_stage_does_not_pause(tmp_path, write_input, behaviour):
    path = write_input(tmp_path, execution=FAST, extra='[workflow]\npause_after = ["dynamics"]')
    assert _run(path).paused_after is None


def test_unknown_pause_stage_is_config_error(tmp_path, write_input, behaviour):
    path = write_input(tmp_path, execution=FAST, extra='[workflow]\npause_after = ["nosuch"]')
    with pytest.raises(ConfigError, match="pause_after.*nosuch"):
        Orchestrator(load_config(path), pipeline_ref=FAKE)


def test_stale_hint_quotes_paths_with_spaces(tmp_path, write_input, behaviour):
    base = tmp_path / "with space"
    base.mkdir()
    path = write_input(base, execution=FAST)
    _run(path)
    path = write_input(base, execution=FAST, molecule="charge = 2")
    with pytest.raises(StaleStageError, match="'.*with space/input.toml' --rerun-from prepare"):
        _run(path)


def test_non_json_fingerprint_names_the_stage(project, behaviour):
    behaviour(bad_fingerprint=["sampling"])
    with pytest.raises(ProjectError, match="sampling.*not JSON-serializable"):
        _run(project)


def test_rerun_from_is_applied_once_per_orchestrator(project, behaviour):
    _run(project)
    orchestrator = Orchestrator(load_config(project), pipeline_ref=FAKE, rerun_from="sampling")
    assert "sampling" in orchestrator.run().completed
    assert "sampling" in orchestrator.run().up_to_date


def test_session_takes_the_lock_before_creating_anything(tmp_path, write_input):
    orch = Orchestrator(load_config(write_input(tmp_path)), pipeline_ref=FAKE)
    # another controller holds the project
    with (
        ProjectLock(orch.layout.lock_file),
        pytest.raises(ProjectError, match="Another gqteashmd controller"),
        orch.session(),
    ):
        pass
    assert not orch.layout.logs_dir.exists()


def test_session_removes_stale_temp_files_and_nests(tmp_path, write_input, behaviour):
    orch = Orchestrator(load_config(write_input(tmp_path, execution=FAST)), pipeline_ref=FAKE)
    orch.layout.meta_dir.mkdir(parents=True)
    stale = orch.layout.meta_dir / ".config.json.12345.tmp"
    stale.write_text("partial")
    with orch.session():
        assert not stale.exists()
        with orch.session():  # nested use does not try to lock twice
            pass
        summary = orch.run()
    assert summary.completed


def test_a_done_stage_whose_directory_was_deleted_is_recomputed(project, behaviour, caplog):
    import shutil

    first = _run(project)
    assert first.completed == ALL
    orch = Orchestrator(load_config(project), pipeline_ref=FAKE)
    victim = orch.stages[0]
    shutil.rmtree(orch.layout.stage_dir(0, victim.name))
    with caplog.at_level("WARNING", logger="gqteashmd.orchestrator"):
        second = orch.run()
    assert victim.name in second.completed
    assert all(name in second.up_to_date for name in first.completed if name != victim.name)
    assert orch.layout.stage_dir(0, victim.name).is_dir()
    assert "is missing; recomputing" in caplog.text


def _records(path):
    with _state(path) as state:
        return {n: (state.get_stage(n).status, state.get_stage(n).fingerprint) for n in ALL}


def test_a_refused_run_changes_nothing(tmp_path, write_input, behaviour):
    """A stale run is refused before any file or stage record is written."""
    path = write_input(tmp_path, execution=FAST)
    assert _run(path).completed == ALL
    config_json = _run_dir(path) / ".gqteashmd" / "config.json"
    snapshot = config_json.read_bytes()
    before = _records(path)
    path = write_input(tmp_path, execution=FAST, molecule="charge = 2")
    with pytest.raises(StaleStageError):
        _run(path)
    assert config_json.read_bytes() == snapshot
    assert _records(path) == before


def test_disabling_then_reenabling_a_finished_stage_keeps_its_results(project, behaviour):
    # In the fake pipeline nothing depends on "optimize", so disabling it is not stale.
    assert _run(project).completed == ALL
    before = _records(project)
    behaviour(skip_optimize=True)
    summary = _run(project)
    assert summary.skipped == ["optimize"]
    assert _records(project) == before
    assert _records(project)["optimize"][0] == StageStatus.DONE
    behaviour()
    summary = _run(project)
    assert "optimize" in summary.up_to_date
    assert summary.completed == []


def test_a_pending_stage_with_done_tasks_and_a_deleted_directory_is_recomputed(
    project, behaviour, caplog
):
    import shutil

    behaviour(sleep=0.3)
    marker = _run_dir(project) / "03_sampling" / "tasks" / "t000" / "result.json"
    orch = Orchestrator(load_config(project), pipeline_ref=FAKE)
    assert orch.run(should_stop=marker.exists).stopped is True
    with _state(project) as state:
        assert state.get_stage("sampling").status == StageStatus.PENDING
        assert state.task_counts("sampling")[TaskStatus.DONE] >= 1
    shutil.rmtree(_run_dir(project) / "03_sampling")
    behaviour()
    with caplog.at_level("WARNING", logger="gqteashmd.orchestrator"):
        summary = _run(project)
    assert "sampling" in summary.completed
    assert "is missing; recomputing" in caplog.text


def test_two_runs_in_one_session_append_one_provenance_record(project, behaviour):
    orch = Orchestrator(load_config(project), pipeline_ref=FAKE)
    with orch.session():
        orch.run()
        orch.run()
    runs = read_json(orch.layout.provenance_file)["runs"]
    assert len(runs) == 1


CANNED = "gqteashmd.testing.pipelines:canned_selection"


def test_o20_stop_then_allow_fewer_reruns_selection_finalize_only(tmp_path, write_input):
    sel = '[selection]\nmode = "fixed"\nstates = ["S2"]\nn_trajectories = 50\n'
    path = write_input(tmp_path, execution=FAST, extra=sel)
    with pytest.raises(StageFailedError, match="allow_fewer = true and run again"):
        Orchestrator(load_config(path), pipeline_ref=CANNED).run()
    with _state(path) as state:
        assert state.get_stage("selection").status == StageStatus.FAILED
        assert state.get_task("selection", "select").status == TaskStatus.DONE
    # The documented remedy: a finalize-only key, so no StaleStageError (exit 3).
    path = write_input(tmp_path, execution=FAST, extra=sel + "allow_fewer = true\n")
    summary = Orchestrator(load_config(path), pipeline_ref=CANNED).run()
    assert summary.completed == ["selection"]
    with _state(path) as state:
        assert state.get_task("selection", "select").attempts == 1  # the task did not rerun
    (stage_dir,) = _run_dir(path).glob("*_selection")
    assert read_json(stage_dir / "trajectories.json")["n_trajectories"] == 20
    # A key that changes the selected pairs needs --rerun-from selection, as the advice says.
    path = write_input(
        tmp_path, execution=FAST, extra=sel.replace("= 50", "= 10") + "allow_fewer = true\n"
    )
    with pytest.raises(StaleStageError, match="--rerun-from selection"):
        Orchestrator(load_config(path), pipeline_ref=CANNED).run()
    summary = Orchestrator(load_config(path), pipeline_ref=CANNED, rerun_from="selection").run()
    assert summary.completed == ["selection"]
    assert read_json(stage_dir / "trajectories.json")["n_trajectories"] == 10


def test_reopened_task_runs_again_and_finalize_repeats(project, behaviour):
    _run(project)
    stage_dir = _run_dir(project) / "04_dynamics"
    summary = stage_dir / "summary.json"
    before = summary.stat().st_mtime_ns
    behaviour(reopen=["dynamics:t001"])
    result = _run(project)
    assert result.completed == ["dynamics"]
    runs = {
        t: read_json(stage_dir / "tasks" / t / "result.json")["result"]["runs"]
        for t in ("t000", "t001", "t002", "t003")
    }
    assert runs == {"t000": 1, "t001": 2, "t002": 1, "t003": 1}
    assert summary.stat().st_mtime_ns > before
    third = _run(project)  # runs == 2 now: nothing to reopen
    assert "dynamics" in third.up_to_date


def test_reopening_an_unknown_task_is_a_project_error(project, behaviour):
    _run(project)
    behaviour(reopen_bad=["dynamics"])
    with pytest.raises(ProjectError, match="t999"):
        _run(project)


def test_reopen_interrupted_midway_is_resumed(project, behaviour, monkeypatch):
    _run(project)
    behaviour(reopen=["dynamics:t001"])
    original = StateStore.set_task

    def dying(self, stage, task_id, status, **kwargs):
        if kwargs.get("message") == "reopened":
            raise RuntimeError("controller died")
        return original(self, stage, task_id, status, **kwargs)

    monkeypatch.setattr(StateStore, "set_task", dying)
    with pytest.raises(RuntimeError, match="controller died"):
        _run(project)
    monkeypatch.setattr(StateStore, "set_task", original)
    _run(project)
    stage_dir = _run_dir(project) / "04_dynamics"
    assert read_json(stage_dir / "tasks" / "t001" / "result.json")["result"]["runs"] == 2
    with _state(project) as state:
        assert state.get_stage("dynamics").status == StageStatus.DONE


def _runs(project):
    stage_dir = _run_dir(project) / "04_dynamics"
    return {
        t: read_json(stage_dir / "tasks" / t / "result.json")["result"]["runs"]
        for t in ("t000", "t001", "t002", "t003")
    }


def test_reopen_hook_applies_to_a_stopped_stage(project, behaviour):
    _run(project)
    # The state after a user stop: the stage PENDING, one task PENDING, the others DONE.
    with _state(project) as state:
        state.set_stage("dynamics", StageStatus.PENDING, message="stopped by user")
        state.set_task("dynamics", "t003", TaskStatus.PENDING)
    behaviour(reopen=["dynamics:t001"])
    result = _run(project)
    assert result.completed == ["dynamics"]
    assert _runs(project) == {"t000": 1, "t001": 2, "t002": 1, "t003": 2}
    with _state(project) as state:
        assert state.get_stage("dynamics").status == StageStatus.DONE


def test_reopen_hook_applies_to_a_failed_stage(project, behaviour):
    behaviour(fail=["dynamics:t002", "dynamics:t003"])
    with pytest.raises(StageFailedError):
        _run(project)
    with _state(project) as state:
        assert state.get_stage("dynamics").status == StageStatus.FAILED
    behaviour(reopen=["dynamics:t001"])
    result = _run(project)
    assert result.completed == ["dynamics"]
    assert _runs(project) == {"t000": 1, "t001": 2, "t002": 1, "t003": 1}
    with _state(project) as state:
        assert state.get_stage("dynamics").status == StageStatus.DONE


def test_stage_without_the_hook_ignores_damaged_results(project, behaviour):
    _run(project)
    for result in (_run_dir(project) / "00_prepare" / "tasks").glob("*/result.json"):
        result.unlink()
    assert "prepare" in _run(project).up_to_date
