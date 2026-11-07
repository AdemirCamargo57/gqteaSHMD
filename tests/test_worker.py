import json
import os

import pytest

from gqteashmd.config.loader import load_config, write_snapshot
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.stages.base import StageContext, TaskSpec
from gqteashmd.testing.fake_stages import BEHAVIOUR_ENV
from gqteashmd.worker import TASK_FILE, read_result, run_task_file, write_task_file

FAKE = "gqteashmd.testing.fake_stages:make_pipeline"


@pytest.fixture
def make_task(tmp_path, write_input, monkeypatch):
    def _make(stage_name="sampling", task_id="t001", behaviour=None):
        monkeypatch.setenv(BEHAVIOUR_ENV, json.dumps(behaviour or {}))
        loaded = load_config(write_input(tmp_path))
        layout = ProjectLayout(loaded.config.project.directory)
        layout.ensure()
        write_snapshot(loaded, layout.config_snapshot)
        stage_dir = layout.stage_dir(3, stage_name)
        task_dir = layout.task_dir(stage_dir, task_id)
        task_dir.mkdir(parents=True)
        ctx = StageContext(loaded.config, layout, stage_dir, {})
        write_task_file(
            task_dir / TASK_FILE,
            pipeline_ref=FAKE,
            config_snapshot=layout.config_snapshot,
            ctx=ctx,
            stage_name=stage_name,
            task=TaskSpec(task_id, {"index": 1}),
            task_dir=task_dir,
        )
        return task_dir

    return _make


def test_worker_writes_ok_result(make_task):
    task_dir = make_task()
    assert run_task_file(task_dir / TASK_FILE) == 0
    assert read_result(task_dir) == {
        "status": "ok",
        "result": {"index": 1, "upstream": [], "threads": os.environ.get("OMP_NUM_THREADS")},
    }
    assert (task_dir / "out.txt").read_text() == "sampling:t001"


def test_worker_reports_task_error(make_task):
    task_dir = make_task(behaviour={"fail": ["sampling:t001"]})
    assert run_task_file(task_dir / TASK_FILE) == 1
    result = read_result(task_dir)
    assert result["status"] == "error"
    assert result["error_type"] == "TaskError"
    assert result["retryable"] is False
    assert "fake failure" in result["message"]


def test_worker_reports_unexpected_exception_with_traceback(make_task):
    task_dir = make_task(behaviour={"unexpected": ["sampling:t001"]})
    assert run_task_file(task_dir / TASK_FILE) == 1
    result = read_result(task_dir)
    assert result["error_type"] == "RuntimeError"
    assert result["retryable"] is False
    assert "Traceback" in result["traceback"]


def test_worker_unknown_stage_is_reported(make_task):
    task_dir = make_task(stage_name="nope")
    assert run_task_file(task_dir / TASK_FILE) == 1
    result = read_result(task_dir)
    assert result["error_type"] == "ProjectError"
    assert "nope" in result["message"]


def test_read_result_missing_or_corrupt_returns_none(tmp_path):
    assert read_result(tmp_path) is None
    (tmp_path / "result.json").write_text("{half")
    assert read_result(tmp_path) is None


def test_worker_task_error_goes_to_stderr(make_task, capsys):
    task_dir = make_task(behaviour={"fail": ["sampling:t001"]})
    assert run_task_file(task_dir / TASK_FILE) == 1
    err = capsys.readouterr().err
    assert "TaskError: fake failure in sampling:t001" in err
    assert "Traceback" not in err


def test_worker_unexpected_exception_traceback_goes_to_stderr(make_task, capsys):
    task_dir = make_task(behaviour={"unexpected": ["sampling:t001"]})
    assert run_task_file(task_dir / TASK_FILE) == 1
    err = capsys.readouterr().err
    assert "RuntimeError: unexpected fake error" in err
    assert "Traceback" in err


def test_unreadable_task_file_gives_an_error_result(tmp_path):
    task_dir = tmp_path / "tasks" / "t0"
    task_dir.mkdir(parents=True)
    task_file = task_dir / "task.json"
    task_file.write_text("{ not json")
    assert run_task_file(task_file) == 1
    result = json.loads((task_dir / "result.json").read_text())
    assert result["status"] == "error" and result["error_type"] == "ProjectError"
    assert result["retryable"] is False and "Cannot read task file" in result["message"]


def test_task_file_missing_keys_gives_an_error_result(tmp_path):
    task_file = tmp_path / "task.json"
    task_file.write_text("{}")
    assert run_task_file(task_file) == 1
    assert json.loads((tmp_path / "result.json").read_text())["error_type"] == "ProjectError"


@pytest.mark.parametrize("kind", ["circular", "set"])
def test_unserialisable_result_gives_an_error_result(make_task, monkeypatch, kind):
    from gqteashmd.testing.fake_stages import FakeStage

    def bad_result(self, ctx, task, task_dir):
        if kind == "set":
            return {"x": {1, 2}}
        d = {}
        d["self"] = d
        return d

    monkeypatch.setattr(FakeStage, "run_task", bad_result)
    task_dir = make_task()
    assert run_task_file(task_dir / TASK_FILE) == 1
    result = read_result(task_dir)
    assert result["status"] == "error" and result["retryable"] is False
    assert "task result is not JSON-serializable" in result["message"]
