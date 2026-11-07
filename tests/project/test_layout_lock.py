import errno
import fcntl
import os
import threading
import time

import pytest

from gqteashmd.errors import ProjectError
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.project.lock import ProjectLock, locked_by


def test_stage_and_task_directories(tmp_path):
    layout = ProjectLayout(tmp_path)
    stage_dir = layout.stage_dir(3, "sampling")
    assert stage_dir == tmp_path / "03_sampling"
    assert layout.task_dir(stage_dir, "t001") == tmp_path / "03_sampling" / "tasks" / "t001"
    assert layout.state_db == tmp_path / ".gqteashmd" / "state.db"
    assert layout.config_snapshot == tmp_path / ".gqteashmd" / "config.json"
    assert layout.lock_file == tmp_path / ".gqteashmd" / "controller.lock"


def test_ensure_creates_meta_and_logs(tmp_path):
    layout = ProjectLayout(tmp_path / "proj")
    layout.ensure()
    assert layout.meta_dir.is_dir()
    assert layout.logs_dir.is_dir()


def test_supersede_moves_directory_instead_of_deleting(tmp_path):
    layout = ProjectLayout(tmp_path)
    layout.ensure()
    stage_dir = layout.stage_dir(1, "optimize")
    stage_dir.mkdir()
    (stage_dir / "result.txt").write_text("old")
    target = layout.supersede(stage_dir)
    assert not stage_dir.exists()
    assert target is not None
    assert target.parent == layout.superseded_dir
    assert target.name.endswith("_01_optimize")
    assert (target / "result.txt").read_text() == "old"


def test_supersede_missing_directory_returns_none(tmp_path):
    layout = ProjectLayout(tmp_path)
    assert layout.supersede(layout.stage_dir(0, "prepare")) is None


def test_lock_is_exclusive_and_released(tmp_path):
    lock_path = tmp_path / "controller.lock"
    assert locked_by(lock_path) is None
    with ProjectLock(lock_path):
        assert locked_by(lock_path) == os.getpid()
        with (
            pytest.raises(ProjectError, match="Another gqteashmd controller"),
            ProjectLock(lock_path),
        ):
            pass
    assert locked_by(lock_path) is None
    with ProjectLock(lock_path):
        pass


def test_lock_waits_briefly_for_a_short_holder(tmp_path):
    lock_path = tmp_path / "controller.lock"
    held = threading.Event()

    def hold_briefly():
        with ProjectLock(lock_path):
            held.set()
            time.sleep(0.05)

    worker = threading.Thread(target=hold_briefly)
    worker.start()
    assert held.wait(5)
    with ProjectLock(lock_path):
        pass
    worker.join()


def _open_fds() -> int:
    return len(os.listdir("/proc/self/fd"))


def test_unsupported_locking_is_a_project_error_and_leaks_nothing(tmp_path, monkeypatch):
    def no_locks(fh, flags):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", no_locks)
    before = _open_fds()
    with pytest.raises(ProjectError, match="locking"), ProjectLock(tmp_path / "controller.lock"):
        pass
    assert _open_fds() == before


def test_failure_after_locking_releases_the_lock(tmp_path, monkeypatch):
    lock_path = tmp_path / "controller.lock"
    real_open = open
    calls = []

    class FailingWrite:
        def __init__(self, fh):
            self._fh = fh

        def __getattr__(self, name):
            return getattr(self._fh, name)

        def write(self, text):
            raise OSError(errno.ENOSPC, "No space left on device")

    def fake_open(path, *args, **kwargs):
        fh = real_open(path, *args, **kwargs)
        if str(path) == str(lock_path) and not calls:
            calls.append(1)
            return FailingWrite(fh)
        return fh

    monkeypatch.setattr("builtins.open", fake_open)
    before = _open_fds()
    with pytest.raises(OSError, match="No space"), ProjectLock(lock_path):
        pass
    monkeypatch.undo()
    assert _open_fds() == before
    with ProjectLock(lock_path):  # the lock was released
        pass
