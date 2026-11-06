"""Persistent stage/task state. The controller is the only writer."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from gqteashmd.errors import ProjectError

SCHEMA_VERSION = "1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS stages (
    name TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    fingerprint TEXT,
    message TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    stage TEXT NOT NULL,
    task_id TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    pid INTEGER,
    message TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (stage, task_id)
);
"""


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class StageRecord:
    name: str
    status: StageStatus
    fingerprint: str | None
    message: str | None


@dataclass(frozen=True)
class TaskRecord:
    stage: str
    task_id: str
    status: TaskStatus
    attempts: int
    pid: int | None
    message: str | None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class StateStore:
    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(db_path, isolation_level=None, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        (version,) = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if version != SCHEMA_VERSION:
            self._conn.close()
            raise ProjectError(
                f"State database {db_path} has schema version {version}; "
                f"this gqteashmd version expects {SCHEMA_VERSION}."
            )

    def __enter__(self) -> StateStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    # ---- stages -----------------------------------------------------------------
    def get_stage(self, name: str) -> StageRecord | None:
        row = self._conn.execute(
            "SELECT name, status, fingerprint, message FROM stages WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        return StageRecord(row[0], StageStatus(row[1]), row[2], row[3])

    def set_stage(
        self,
        name: str,
        status: StageStatus,
        *,
        fingerprint: str | None = None,
        message: str | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO stages(name, status, fingerprint, message, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                status = excluded.status,
                fingerprint = COALESCE(excluded.fingerprint, stages.fingerprint),
                message = excluded.message,
                updated_at = excluded.updated_at
            """,
            (name, status.value, fingerprint, message, _now()),
        )

    def reset_stage(self, name: str) -> None:
        with self._transaction():
            self._conn.execute("DELETE FROM tasks WHERE stage = ?", (name,))
            self._conn.execute("DELETE FROM stages WHERE name = ?", (name,))

    # ---- tasks ------------------------------------------------------------------
    def upsert_tasks(self, stage: str, task_ids: Iterable[str]) -> None:
        now = _now()
        with self._transaction():
            self._conn.executemany(
                "INSERT OR IGNORE INTO tasks(stage, task_id, status, attempts, updated_at) "
                "VALUES (?, ?, ?, 0, ?)",
                [(stage, task_id, TaskStatus.PENDING.value, now) for task_id in task_ids],
            )

    def get_tasks(self, stage: str) -> list[TaskRecord]:
        rows = self._conn.execute(
            "SELECT stage, task_id, status, attempts, pid, message FROM tasks "
            "WHERE stage = ? ORDER BY task_id",
            (stage,),
        ).fetchall()
        return [TaskRecord(r[0], r[1], TaskStatus(r[2]), r[3], r[4], r[5]) for r in rows]

    def get_task(self, stage: str, task_id: str) -> TaskRecord:
        row = self._conn.execute(
            "SELECT stage, task_id, status, attempts, pid, message FROM tasks "
            "WHERE stage = ? AND task_id = ?",
            (stage, task_id),
        ).fetchone()
        if row is None:
            raise ProjectError(f"Unknown task {stage}/{task_id} in state database")
        return TaskRecord(row[0], row[1], TaskStatus(row[2]), row[3], row[4], row[5])

    def set_task(
        self,
        stage: str,
        task_id: str,
        status: TaskStatus,
        *,
        pid: int | None = None,
        message: str | None = None,
        attempts_delta: int = 0,
    ) -> None:
        cursor = self._conn.execute(
            "UPDATE tasks SET status = ?, pid = ?, message = ?, "
            "attempts = MAX(0, attempts + ?), updated_at = ? "
            "WHERE stage = ? AND task_id = ?",
            (status.value, pid, message, attempts_delta, _now(), stage, task_id),
        )
        if cursor.rowcount == 0:
            raise ProjectError(f"Unknown task {stage}/{task_id} in state database")

    def requeue_failed(self, stage: str) -> int:
        cursor = self._conn.execute(
            "UPDATE tasks SET status = ?, attempts = 0, updated_at = ? "
            "WHERE stage = ? AND status = ?",
            (TaskStatus.PENDING.value, _now(), stage, TaskStatus.FAILED.value),
        )
        return cursor.rowcount

    def task_counts(self, stage: str) -> dict[TaskStatus, int]:
        counts = {status: 0 for status in TaskStatus}
        for status, count in self._conn.execute(
            "SELECT status, COUNT(*) FROM tasks WHERE stage = ? GROUP BY status", (stage,)
        ):
            counts[TaskStatus(status)] = count
        return counts
