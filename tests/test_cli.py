import json
import logging
import os
import signal
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


def test_validate_rejects_a_directory_as_geometry(tmp_path, write_input, capsys):
    path = write_input(tmp_path)
    (tmp_path / "water.xyz").unlink()
    (tmp_path / "water.xyz").mkdir()
    assert main(["validate", str(path)]) == 2
    assert "not a regular file" in capsys.readouterr().err


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


def test_init_into_missing_directory_is_a_config_error(tmp_path, capsys):
    assert main(["init", str(tmp_path / "nodir" / "x.toml")]) == 2
    assert "Cannot write" in capsys.readouterr().err


def test_status_with_corrupt_state_exits_3(project, capsys):
    db = project.parent / "run" / ".gqteashmd" / "state.db"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"garbage" * 100)
    assert main(["status", str(project), "--pipeline", FAKE]) == 3
    assert "unreadable or corrupt" in capsys.readouterr().err


def test_unexpected_exception_becomes_internal_error(project, monkeypatch, capsys):
    def boom(_args):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("gqteashmd.cli._cmd_validate", boom)
    assert main(["validate", str(project)]) == 1
    err = capsys.readouterr().err
    assert "internal error: RuntimeError: kaboom" in err
    assert "please report this" in err


def test_stop_os_error_is_a_project_error(project, monkeypatch, capsys):
    monkeypatch.setattr("gqteashmd.cli.locked_by", lambda _path: 12345)

    def deny(_pid, _sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr("gqteashmd.cli.os.kill", deny)
    assert main(["stop", str(project)]) == 3
    assert "Cannot signal controller pid 12345" in capsys.readouterr().err


@pytest.mark.slow
def test_sigterm_stops_gracefully_and_resume_completes(project, monkeypatch):
    monkeypatch.setenv(BEHAVIOUR_ENV, json.dumps({"sleep": 1.0}))
    proc = subprocess.Popen(
        [sys.executable, "-m", "gqteashmd", "run", str(project), "--pipeline", FAKE],
        env=dict(os.environ),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
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
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert proc.returncode == EXIT_STOPPED
    assert "Stopped." in out
    monkeypatch.setenv(BEHAVIOUR_ENV, "{}")
    assert main(["run", str(project), "--pipeline", FAKE]) == 0
    with StateStore(db) as state:
        assert state.get_stage("dynamics").status == StageStatus.DONE


def test_status_does_not_modify_the_state_database(project, capsys):
    assert main(["run", str(project), "--pipeline", FAKE]) == 0
    db = project.parent / "run" / ".gqteashmd" / "state.db"
    before = db.read_bytes()
    capsys.readouterr()
    assert main(["status", str(project), "--pipeline", FAKE]) == 0
    assert db.read_bytes() == before


def test_pause_returns_exit_5_with_message(tmp_path, write_input, monkeypatch, capsys):
    monkeypatch.setenv(BEHAVIOUR_ENV, "{}")
    path = write_input(tmp_path, execution=FAST, extra='[workflow]\npause_after = ["sampling"]')
    assert main(["run", str(path), "--pipeline", FAKE]) == EXIT_STOPPED
    assert "Paused after stage 'sampling'" in capsys.readouterr().out


def test_validate_rejects_unknown_override_stage(tmp_path, write_input, capsys):
    path = write_input(tmp_path, resources="[resources.stages.nosuch]\nthreads_per_task = 1")
    assert main(["validate", str(path), "--pipeline", FAKE]) == 2
    assert "[resources.stages] names unknown stage(s): nosuch" in capsys.readouterr().err


def test_validate_rejects_unknown_pause_stage(tmp_path, write_input, capsys):
    path = write_input(tmp_path, extra='[workflow]\npause_after = ["nosuch"]')
    assert main(["validate", str(path), "--pipeline", FAKE]) == 2
    assert "workflow.pause_after names unknown stage(s): nosuch" in capsys.readouterr().err


def test_run_with_project_directory_that_is_a_file_exits_2(tmp_path, write_input, capsys):
    path = write_input(tmp_path)
    (tmp_path / "run").write_text("in the way")
    assert main(["run", str(path), "--pipeline", FAKE]) == 2
    assert "not a directory" in capsys.readouterr().err


def test_run_backstop_turns_os_error_into_config_error(tmp_path, write_input, monkeypatch, capsys):
    from gqteashmd.project.layout import ProjectLayout

    def fail(self):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(ProjectLayout, "ensure", fail)
    path = write_input(tmp_path)
    assert main(["run", str(path), "--pipeline", FAKE]) == 2
    assert "Cannot create project directory" in capsys.readouterr().err


def test_a_run_refused_by_the_lock_writes_no_log(tmp_path, write_input, capsys):
    from gqteashmd.config.loader import load_config
    from gqteashmd.project.layout import ProjectLayout
    from gqteashmd.project.lock import ProjectLock

    path = write_input(tmp_path)
    layout = ProjectLayout(load_config(path).config.project.directory)
    with ProjectLock(layout.lock_file):
        code = main(["run", str(path), "--pipeline", FAKE])
    assert code == 3
    assert not (layout.logs_dir / "gqteashmd.log").exists()


def test_ctrl_c_anywhere_exits_cleanly(monkeypatch, capsys):
    import argparse

    from gqteashmd import cli

    def interrupted(args):
        raise KeyboardInterrupt

    class Parser:
        def parse_args(self, argv=None):
            return argparse.Namespace(func=interrupted)

    monkeypatch.setattr(cli, "build_parser", lambda: Parser())
    assert cli.main(["anything"]) == cli.EXIT_STOPPED
    assert capsys.readouterr().err.count("gqteashmd: interrupted") == 1


def test_internal_error_traceback_goes_to_the_log_file_only(project, monkeypatch, capsys):
    from gqteashmd.orchestrator import Orchestrator
    from gqteashmd.project.layout import ProjectLayout

    def boom(self, state, should_stop):
        raise RuntimeError("controller kaboom")

    monkeypatch.setattr(Orchestrator, "_run_pipeline", boom)
    assert main(["run", str(project), "--pipeline", FAKE]) == 1
    err = capsys.readouterr().err
    assert "internal error: RuntimeError: controller kaboom" in err
    assert "Traceback" not in err
    layout = ProjectLayout(load_config(project).config.project.directory)
    log_text = (layout.logs_dir / "gqteashmd.log").read_text()
    assert "Traceback (most recent call last)" in log_text
    assert "controller kaboom" in log_text


def test_status_while_a_controller_holds_the_lock(project, capsys):
    from gqteashmd.project.layout import ProjectLayout
    from gqteashmd.project.lock import ProjectLock

    assert main(["run", str(project), "--pipeline", FAKE]) == 0
    layout = ProjectLayout(load_config(project).config.project.directory)
    capsys.readouterr()
    with ProjectLock(layout.lock_file):
        assert main(["status", str(project), "--pipeline", FAKE]) == 0
    assert f"running (pid {os.getpid()})" in capsys.readouterr().out


def test_preflight_warnings_are_printed_once(tmp_path, write_input, monkeypatch, capsys):
    monkeypatch.setenv(BEHAVIOUR_ENV, "{}")
    path = write_input(tmp_path, execution=FAST, total_cores=100000, total_memory_mb=10**9)
    main(["run", str(path), "--pipeline", FAKE])
    err = capsys.readouterr().err
    lines = {line for line in err.splitlines() if "Preflight:" in line}
    assert lines
    for line in lines:
        assert err.count(line) == 1
