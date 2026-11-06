import os

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
