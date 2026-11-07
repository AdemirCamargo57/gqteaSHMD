import sqlite3

import pytest

from gqteashmd.errors import ProjectError
from gqteashmd.project.state import StageRecord, StageStatus, StateStore, TaskStatus


@pytest.fixture
def store(tmp_path):
    with StateStore(tmp_path / "state.db") as s:
        yield s


def test_stage_roundtrip_keeps_fingerprint_when_not_given(store):
    assert store.get_stage("sampling") is None
    store.set_stage("sampling", StageStatus.RUNNING, fingerprint="abc")
    store.set_stage("sampling", StageStatus.DONE, message="ok")
    assert store.get_stage("sampling") == StageRecord("sampling", StageStatus.DONE, "abc", "ok")


def test_upsert_tasks_is_idempotent_and_sorted(store):
    store.upsert_tasks("s", ["t2", "t1"])
    store.upsert_tasks("s", ["t1", "t2"])
    tasks = store.get_tasks("s")
    assert [t.task_id for t in tasks] == ["t1", "t2"]
    assert all(t.status == TaskStatus.PENDING and t.attempts == 0 for t in tasks)


def test_set_task_updates_attempts_pid_and_message(store):
    store.upsert_tasks("s", ["t1"])
    store.set_task("s", "t1", TaskStatus.RUNNING, pid=42, attempts_delta=1)
    t = store.get_task("s", "t1")
    assert (t.status, t.pid, t.attempts) == (TaskStatus.RUNNING, 42, 1)
    store.set_task("s", "t1", TaskStatus.PENDING, message="interrupted", attempts_delta=-1)
    t = store.get_task("s", "t1")
    assert (t.status, t.pid, t.attempts, t.message) == (TaskStatus.PENDING, None, 0, "interrupted")


def test_attempts_never_go_negative(store):
    store.upsert_tasks("s", ["t1"])
    store.set_task("s", "t1", TaskStatus.PENDING, attempts_delta=-1)
    assert store.get_task("s", "t1").attempts == 0


def test_unknown_task_raises(store):
    with pytest.raises(ProjectError, match="s/nope"):
        store.set_task("s", "nope", TaskStatus.DONE)
    with pytest.raises(ProjectError, match="s/nope"):
        store.get_task("s", "nope")


def test_requeue_failed_resets_attempts(store):
    store.upsert_tasks("s", ["t1", "t2"])
    store.set_task("s", "t1", TaskStatus.FAILED, attempts_delta=2, message="boom")
    store.set_task("s", "t2", TaskStatus.DONE, attempts_delta=1)
    assert store.requeue_failed("s") == 1
    t1 = store.get_task("s", "t1")
    assert (t1.status, t1.attempts) == (TaskStatus.PENDING, 0)
    assert store.get_task("s", "t2").status == TaskStatus.DONE


def test_reset_stage_removes_stage_and_tasks(store):
    store.set_stage("s", StageStatus.DONE, fingerprint="x")
    store.upsert_tasks("s", ["t1"])
    store.reset_stage("s")
    assert store.get_stage("s") is None
    assert store.get_tasks("s") == []


def test_task_counts_include_all_statuses(store):
    store.upsert_tasks("s", ["a", "b", "c"])
    store.set_task("s", "a", TaskStatus.DONE)
    counts = store.task_counts("s")
    assert counts[TaskStatus.DONE] == 1
    assert counts[TaskStatus.PENDING] == 2
    assert counts[TaskStatus.FAILED] == 0
    assert counts[TaskStatus.RUNNING] == 0


def test_state_persists_across_connections(tmp_path):
    db = tmp_path / "state.db"
    with StateStore(db) as s:
        s.set_stage("prepare", StageStatus.DONE, fingerprint="f")
    with StateStore(db) as s:
        assert s.get_stage("prepare").status == StageStatus.DONE


def test_corrupt_database_is_a_project_error(tmp_path):
    db = tmp_path / "state.db"
    db.write_bytes(b"this is definitely not a sqlite database" * 50)
    with pytest.raises(ProjectError, match="unreadable or corrupt"):
        StateStore(db)


def test_directory_as_db_path_is_project_error(tmp_path):
    db = tmp_path / "state.db"
    db.mkdir()
    with pytest.raises(ProjectError, match="unreadable or corrupt"):
        StateStore(db)


def test_readonly_open_reads_and_refuses_writes(tmp_path):
    db = tmp_path / "state.db"
    with StateStore(db) as s:
        s.set_stage("prepare", StageStatus.DONE, fingerprint="f")
    with StateStore(db, readonly=True) as ro:
        assert ro.get_stage("prepare").status == StageStatus.DONE
        with pytest.raises(ProjectError, match="readonly database"):
            ro.set_stage("prepare", StageStatus.FAILED)


def test_readonly_missing_db_is_project_error(tmp_path):
    with pytest.raises(ProjectError, match="unreadable or corrupt"):
        StateStore(tmp_path / "missing.db", readonly=True)


def test_unknown_status_is_project_error(tmp_path):
    db = tmp_path / "state.db"
    with StateStore(db) as s:
        s.set_stage("prepare", StageStatus.DONE)
        s.upsert_tasks("prepare", ["t1"])
    conn = sqlite3.connect(db)
    conn.execute("UPDATE stages SET status = 'weird'")
    conn.execute("UPDATE tasks SET status = 'weird'")
    conn.commit()
    conn.close()
    with StateStore(db) as s:
        with pytest.raises(ProjectError, match="unknown stage status 'weird'"):
            s.get_stage("prepare")
        with pytest.raises(ProjectError, match="unknown task status 'weird'"):
            s.get_tasks("prepare")


def test_unknown_stored_status_is_a_project_error(tmp_path):
    db = tmp_path / "state.db"
    with StateStore(db) as s:
        s.set_stage("prepare", StageStatus.DONE, fingerprint="x")
    conn = sqlite3.connect(db)
    conn.execute("UPDATE stages SET status = 'from-the-future'")
    conn.commit()
    conn.close()
    with StateStore(db) as s, pytest.raises(ProjectError, match="unknown stage status"):
        s.get_stage("prepare")


def test_use_after_close_is_a_project_error(tmp_path):
    s = StateStore(tmp_path / "state.db")
    s.close()
    with pytest.raises(ProjectError, match="state database"):
        s.get_stage("prepare")


def test_query_time_sqlite_errors_become_project_errors(tmp_path, monkeypatch):
    s = StateStore(tmp_path / "state.db")

    class Broken:
        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError("disk I/O error")

        def executemany(self, *args, **kwargs):
            raise sqlite3.OperationalError("disk I/O error")

        def close(self):
            pass

    monkeypatch.setattr(s, "_conn", Broken())
    for call in (
        lambda: s.get_stage("prepare"),
        lambda: s.set_stage("prepare", StageStatus.RUNNING),
        lambda: s.upsert_tasks("prepare", ["t0"]),
        lambda: s.get_tasks("prepare"),
        lambda: s.task_counts("prepare"),
    ):
        with pytest.raises(ProjectError, match="disk I/O error"):
            call()
