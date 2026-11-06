import json
import logging
import os
import subprocess
import sys
import time

import pytest

from gqteashmd.cli import EXIT_STOPPED, main
from gqteashmd.config.loader import load_config
from gqteashmd.logging_setup import configure_logging
from gqteashmd.project.state import StageStatus, StateStore, TaskStatus
from gqteashmd.testing.fake_stages import BEHAVIOUR_ENV

FAKE = "gqteashmd.testing.fake_stages:make_pipeline"
FAST = "poll_interval_s = 0.05\nstop_grace_period_s = 2.0"


@pytest.fixture
def project(tmp_path, write_input, monkeypatch):
    monkeypatch.setenv(BEHAVIOUR_ENV, "{}")
    return write_input(tmp_path, execution=FAST)


def test_init_writes_a_valid_template(tmp_path, water_xyz):
    target = tmp_path / "input.toml"
    assert main(["init", str(target)]) == 0
    (tmp_path / "molecule.xyz").write_text(water_xyz)
    assert load_config(target).config.project.name == "my_molecule"


def test_init_refuses_to_overwrite(tmp_path, capsys):
    target = tmp_path / "input.toml"
    target.write_text("keep me")
    assert main(["init", str(target)]) == 2
    assert "already exists" in capsys.readouterr().err
    assert target.read_text() == "keep me"


def test_validate_reports_success_and_errors(tmp_path, write_input, capsys):
    good = write_input(tmp_path)
    assert main(["validate", str(good)]) == 0
    assert "is valid" in capsys.readouterr().out
    (tmp_path / "water.xyz").unlink()
    assert main(["validate", str(good)]) == 2
    assert "Geometry file not found" in capsys.readouterr().err


def test_run_then_status(project, capsys):
    assert main(["run", str(project), "--pipeline", FAKE]) == 0
    assert "Workflow complete" in capsys.readouterr().out
    assert main(["status", str(project), "--pipeline", FAKE]) == 0
    out = capsys.readouterr().out
    assert "Controller: not running" in out
    assert "dynamics" in out and "done" in out


def test_status_before_start(project, capsys):
    assert main(["status", str(project), "--pipeline", FAKE]) == 0
    assert "has not been started" in capsys.readouterr().out


def test_stop_without_controller(project, capsys):
    assert main(["stop", str(project)]) == 0
    assert "No running controller" in capsys.readouterr().out


def test_run_failure_returns_stage_failed_code(project, monkeypatch, capsys):
    monkeypatch.setenv(BEHAVIOUR_ENV, json.dumps({"fail": ["sampling:t000"]}))
    assert main(["run", str(project), "--pipeline", FAKE]) == 4
    assert "Stage 'sampling' failed" in capsys.readouterr().err


def test_configure_logging_writes_file(tmp_path):
    log_file = tmp_path / "logs" / "gqteashmd.log"
    configure_logging(log_file)
    logging.getLogger("gqteashmd.test").debug("hello file")
    for handler in logging.getLogger("gqteashmd").handlers:
        handler.flush()
    assert "hello file" in log_file.read_text()


def test_exit_stopped_constant():
    assert EXIT_STOPPED == 5


@pytest.mark.slow
def test_killed_controller_resumes(project, monkeypatch):
    monkeypatch.setenv(BEHAVIOUR_ENV, json.dumps({"sleep": 1.0}))
    proc = subprocess.Popen(
        [sys.executable, "-m", "gqteashmd", "run", str(project), "--pipeline", FAKE],
        env=dict(os.environ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    db = project.parent / "run" / ".gqteashmd" / "state.db"
    deadline = time.monotonic() + 60
    try:
        while time.monotonic() < deadline:
            if db.exists():
                with StateStore(db) as state:
                    if any(t.status == TaskStatus.DONE for t in state.get_tasks("sampling")):
                        break
            time.sleep(0.1)
        else:
            pytest.fail("controller made no progress within 60 s")
    finally:
        proc.kill()
        proc.wait()
    assert main(["run", str(project), "--pipeline", FAKE]) == 0
    with StateStore(db) as state:
        assert state.get_stage("dynamics").status == StageStatus.DONE
