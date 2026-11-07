# P3c: Orchestrator Hardening Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the robustness gaps left open by P1, P2, P3a and P3b in the controller, state store, lock, executor, worker and CLI, so that P4 can add the first real quantum-chemistry stages on a solid foundation.

**Architecture:** This plan adds no new subsystem. Each task hardens one module and adds tests:

- **Lock:** the project lock is taken *before* anything is created or logged in the project.
- **State store:** query-time sqlite failures become `ProjectError`.
- **Processes:** a worker's whole process group is killed, not only its leader.
- **Worker:** the worker reports an unreadable task file as a result instead of crashing.
- **Errors:** `TaskError` survives pickling.
- **Stage directories:** a DONE stage whose directory was deleted is recomputed.
- **Interrupts:** Ctrl-C at any point gives a clean exit.
- **Test tiers:** a `pyscf` test marker gives a fast development tier again.

**Tech Stack:** Python 3.12, sqlite3, fcntl, subprocess and signals (Linux), pytest, ruff 0.17.4, mypy.

**Spec:**
- [`2026-11-06-gqteashmd-master-plan.md`](2026-11-06-gqteashmd-master-plan.md): §3 orchestration and §6 error handling. The §9 P3c row reads: "The deferred items of the P1–P3b ledgers".
- [`2026-11-06-p2-input-preflight.md`](2026-11-06-p2-input-preflight.md): its "Scope rulings" assign these items to the P3 hardening sweep.
- The local, git-ignored P2 ledger `.superpowers/sdd/2026-11-06-p2-input-preflight/final-fix-findings.md` §D, plus the P1, P3a and P3b ledgers.

## Global Constraints

- **Platform.** The code runs on Linux only (target Oracle Linux 9). Run every command inside WSL Oracle Linux 9.5:
  `wsl -d OracleLinux_9_5 -- bash -lc 'cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && conda activate gqteashmd-dev && <command>'`.
  Edit files with the file tools at the Windows paths, using LF line endings. Put scratch work in `mktemp -d /tmp/gq-XXXX`, never in `$HOME` or the repository. Delete scratch directories by their exact path. Run one pytest process at a time.
- **Dependencies.** No new runtime dependencies. The runtime set stays `pydantic>=2.6`, `numpy>=1.26`, `scipy>=1.11`, `pyscf>=2.6`.
- **Error classes and exit codes.** These are unchanged. Every deliberate error reaches the user once, as `gqteashmd: error: <message>`.

  | Error | Exit code |
  |---|---|
  | `ShmdError` | 1 |
  | `ConfigError` | 2 |
  | `ProjectError` / `StaleStageError` | 3 |
  | `StageFailedError` | 4 |
  | Stopped or paused (`cli.EXIT_STOPPED`) | 5 |
- **Behaviour must not regress.** Keep:
  - fingerprints and `StaleStageError`;
  - `--rerun-from` (supersede, never delete);
  - retries;
  - orphan reconciliation;
  - pause;
  - read-only `status`;
  - preflight before the project directory is touched.

  Every existing test must keep passing.
- **Style:** ruff 0.17.4 with line length 100 and rule sets E, F, W, I, B, UP, SIM. mypy must stay clean on `src/`.
- **Git:** work on branch `feat/p3c-hardening`, in WSL Git only, with the pre-commit hooks. Never use `--no-verify`. Use conventional commit subjects and end each message with the `Co-Authored-By:` trailer of the model that wrote it. Do not push.
- **Documentation:** the last task updates CLAUDE.md, USER_MANUAL.md and the master plan.

## Scope rulings (made when writing this plan)

- **In scope**, from the ledgers:
  - the lock fd leak and non-blocking `OSError`;
  - the lock-test assertion;
  - git calls under the lock;
  - `ensure()`/logging set up before the lock;
  - `.tmp` cleanup;
  - query-time sqlite errors;
  - the unknown-status and closed-connection tests;
  - SIGKILL to the group after the leader exits;
  - an unreadable `task.json`;
  - `ValueError` during result serialisation;
  - `TaskError` pickling (P3a review);
  - a DONE stage directory deleted by hand;
  - Ctrl-C before the handlers are installed;
  - the tests for: traceback in the log file, concurrent `status`, warnings printed once, and the preflight `ImportError` branch;
  - the CPU model in provenance;
  - fast-tier test speed.
- **Moved to P4.** Each item here fits better with the work that needs it:

  | Item | Reason |
  |---|---|
  | Refused runs still writing state (snapshot before refusal, SKIPPED over DONE, `_reset_from` before the stale check) | Belongs with optional stages, as ruled in P2 |
  | The schema validator reporting only the first violation | Belongs with the new `[optimize]` and `[frequencies]` sections |
  | PySCF/geomeTRIC version and writable-scratch preflight | Belongs with the first stage that declares `required_modules=("pyscf",)` and a scratch directory |
- **Moved to P7b:** the worker orphan window between `Popen` and `on_start`.
- **Deferred to release packaging:** the provenance build stamp (hatch-vcs). There is no non-editable install yet.

## File Structure

```text
src/gqteashmd/project/lock.py           # Task 1: close the handle on every failure; OSError -> ProjectError
src/gqteashmd/util/fileio.py            # Task 2: remove the temp file when an atomic write fails
src/gqteashmd/project/provenance.py     # Task 2: cpu_model
src/gqteashmd/orchestrator.py           # Tasks 2, 6: session() (lock first), meta .tmp cleanup, deleted DONE stage dir
src/gqteashmd/cli.py                    # Tasks 2, 7: lock before ensure/logging; KeyboardInterrupt
src/gqteashmd/project/state.py          # Task 3: sqlite errors at query time -> ProjectError
src/gqteashmd/execution/local.py        # Task 4: kill the whole group after the leader exits
src/gqteashmd/execution/processes.py    # Task 4: same for orphan termination
src/gqteashmd/worker.py                 # Task 5: unreadable task.json; ValueError in serialisation
src/gqteashmd/errors.py                 # Task 5: TaskError pickling
pyproject.toml + tests (pytestmark)     # Task 8: `pyscf` marker
CLAUDE.md, USER_MANUAL.md, master plan  # Task 9
```

---

### Task 1: Lock hardening

**Files:**
- Modify: `src/gqteashmd/project/lock.py`
- Test: `tests/project/test_layout_lock.py`

**Interfaces:**
- `ProjectLock.__enter__` never leaks its file handle. A non-blocking `OSError` from `flock` raises `ProjectError` saying file locking is unavailable at this path; `ENOLCK` and `EOPNOTSUPP` on network filesystems are examples. Any failure after the lock is taken releases the lock and closes the handle. This covers the truncate and write steps.
- The behaviour on success and on `BlockingIOError` is unchanged.

- [ ] **Step 1: Write the failing tests.** Append to `tests/project/test_layout_lock.py` (merge imports with the existing ones):

```python
import errno
import fcntl
import os

import pytest

from gqteashmd.errors import ProjectError
from gqteashmd.project.lock import ProjectLock


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
```

In `test_lock_waits_briefly_for_a_short_holder`, replace `held.wait(5)` with `assert held.wait(5)`.

- [ ] **Step 2: Run them to verify they fail.**
  Run `pytest tests/project/test_layout_lock.py -v -W error`.
  Expected: the ENOLCK test fails because a raw `OSError` escapes, and the fd count assertion fails in the second test.

- [ ] **Step 3: Implement.** Replace `ProjectLock.__enter__` with:

```python
    def __enter__(self) -> ProjectLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+", encoding="utf-8")  # noqa: SIM115 - kept open while locked
        try:
            for attempt in range(2):
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    if attempt == 0:
                        time.sleep(0.2)  # `locked_by` may be probing the lock for an instant
                        continue
                    holder = locked_by(self.path)
                    raise ProjectError(
                        "Another gqteashmd controller is running for this project "
                        f"(pid {holder if holder is not None else 'unknown'}, "
                        f"lock file {self.path}). "
                        "Use `gqteashmd status` or `gqteashmd stop`."
                    ) from exc
                except OSError as exc:
                    raise ProjectError(
                        f"File locking is not available for {self.path} "
                        f"({exc.strerror or exc}); put the project directory on a local Linux "
                        "filesystem."
                    ) from exc
            try:
                fh.seek(0)
                fh.truncate()
                fh.write(f"{os.getpid()}\n")
                fh.flush()
            except BaseException:
                fcntl.flock(fh, fcntl.LOCK_UN)
                raise
        except BaseException:
            fh.close()
            raise
        self._fh = fh
        return self
```

Note: in the `BlockingIOError` branch `locked_by` opens a second handle; it is closed by its own `with`, and `fh` is closed by the outer `except`.

- [ ] **Step 4: Run the tests to verify they pass.**
  Run `pytest tests/project -v -W error && mypy`.
  Expected: all pass.

- [ ] **Step 5: Lint, full suite, commit**

```bash
ruff format src tests tools && ruff check src tests tools && pytest -m "not slow" -W error
git add src/gqteashmd/project/lock.py tests/project/test_layout_lock.py
git commit -m "fix: never leak the lock file handle and report unsupported locking clearly"
```

---

### Task 2: Lock before anything is written; temp-file hygiene; provenance outside the lock

**Files:**
- Modify: `src/gqteashmd/orchestrator.py`, `src/gqteashmd/cli.py`, `src/gqteashmd/util/fileio.py`, `src/gqteashmd/project/provenance.py`
- Test: `tests/test_orchestrator.py`, `tests/test_cli.py`, `tests/test_errors_fileio.py`, `tests/project/test_provenance.py`

**Interfaces:**
- `Orchestrator.session() -> ContextManager[None]`: acquires the project lock, then runs `layout.ensure()`, then removes stale `.*.tmp` files in `layout.meta_dir`. These are safe to remove because the controller holds the lock and is the only writer there. Re-entering is allowed: nested use only increments a counter.
- `Orchestrator.run(should_stop)` uses `with self.session():`. When the CLI already holds a session the call nests, so the lock is not taken twice.
- `cli._cmd_run`: `preflight()` comes first (unchanged). Then `with orchestrator.session():` wraps `configure_logging(...)`, the signal handlers and `orchestrator.run(...)`. A run refused by the lock therefore writes nothing to the project log.
- `collect_provenance()` runs **before** the lock is taken, in `Orchestrator.session()` before `ProjectLock`. `append_provenance` stays under the lock. git can be slow, and the lock must not wait on it.
- `write_bytes_atomic`: if writing or `os.replace` raises, the temp file is removed and the exception propagates.
- `collect_provenance()` gains a `"cpu_model"` key: the first `model name` from `/proc/cpuinfo`, or `None`.

- [ ] **Step 1: Write the failing tests.**

Append to `tests/test_errors_fileio.py`:

```python
def test_failed_atomic_write_leaves_no_temp_file(tmp_path, monkeypatch):
    import os

    from gqteashmd.util.fileio import write_bytes_atomic

    def broken_replace(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", broken_replace)
    target = tmp_path / "out.json"
    with pytest.raises(OSError, match="No space"):
        write_bytes_atomic(target, b"data")
    assert list(tmp_path.iterdir()) == []
```

Append to `tests/project/test_provenance.py`:

```python
def test_provenance_records_the_cpu_model():
    from gqteashmd.project.provenance import collect_provenance

    entry = collect_provenance()
    assert "cpu_model" in entry
    assert entry["cpu_model"] is None or isinstance(entry["cpu_model"], str)
```

Append to `tests/test_orchestrator.py`. Reuse the module's existing helpers for building an orchestrator over the fake pipeline; read the top of the file first.

```python
def test_session_takes_the_lock_before_creating_anything(tmp_path, write_input):
    from gqteashmd.config.loader import load_config
    from gqteashmd.errors import ProjectError
    from gqteashmd.orchestrator import Orchestrator
    from gqteashmd.project.lock import ProjectLock

    orch = Orchestrator(load_config(write_input(tmp_path)), pipeline_ref=FAKE_PIPELINE)
    lock = orch.layout.lock_file
    with ProjectLock(lock):  # another controller holds the project
        with pytest.raises(ProjectError, match="Another gqteashmd controller"):
            with orch.session():
                pass
    assert not orch.layout.logs_dir.exists()


def test_session_removes_stale_temp_files_and_nests(tmp_path, write_input):
    from gqteashmd.config.loader import load_config
    from gqteashmd.orchestrator import Orchestrator

    orch = Orchestrator(load_config(write_input(tmp_path)), pipeline_ref=FAKE_PIPELINE)
    orch.layout.meta_dir.mkdir(parents=True)
    stale = orch.layout.meta_dir / ".config.json.12345.tmp"
    stale.write_text("partial")
    with orch.session():
        assert not stale.exists()
        with orch.session():  # nested use does not try to lock twice
            pass
        summary = orch.run()
    assert summary.completed
```

(`FAKE_PIPELINE` is the pipeline reference string the module's other tests already pass, for example `"gqteashmd.testing.fake_stages:make_pipeline"`. Use the existing constant or fixture.)

Append to `tests/test_cli.py`. Use the existing helpers that run `main([...])` against the fake pipeline:

```python
def test_a_run_refused_by_the_lock_writes_no_log(tmp_path, write_input, capsys):
    from gqteashmd.cli import main
    from gqteashmd.config.loader import load_config
    from gqteashmd.project.layout import ProjectLayout
    from gqteashmd.project.lock import ProjectLock

    path = write_input(tmp_path)
    layout = ProjectLayout(load_config(path).config.project.directory)
    with ProjectLock(layout.lock_file):
        code = main(["run", str(path), "--pipeline", FAKE_PIPELINE])
    assert code == 3
    assert not (layout.logs_dir / "gqteashmd.log").exists()
```

- [ ] **Step 2: Run them to verify they fail.**
  Run `pytest tests/test_errors_fileio.py tests/project/test_provenance.py tests/test_orchestrator.py tests/test_cli.py -v -W error -k "temp or cpu_model or session or refused"`.
  Expected: FAIL. `session` is missing, the temp file remains, `cpu_model` is missing, and the log file exists.

- [ ] **Step 3: Implement.**

`src/gqteashmd/util/fileio.py`, in `write_bytes_atomic`:

```python
def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write `data` to `path` so that readers never see a partially written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
```

`src/gqteashmd/project/provenance.py`: add the function below, and add `"cpu_model": _cpu_model(),` after `"cpu_count"` in `collect_provenance`:

```python
def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip() or None
    except OSError:
        return None
    return None
```

`src/gqteashmd/orchestrator.py`:

- Add the imports `from collections.abc import Iterator` and `from contextlib import contextmanager`.
- In `__init__`, add `self._session_depth = 0` and `self._provenance: dict[str, Any] | None = None`.
- Add the method:

```python
    @contextmanager
    def session(self) -> Iterator[None]:
        """Hold the project lock; create the project skeleton only once the lock is ours."""
        if self._session_depth:
            self._session_depth += 1
            try:
                yield
            finally:
                self._session_depth -= 1
            return
        provenance = collect_provenance()  # before the lock: git may be slow
        with ProjectLock(self.layout.lock_file):
            self.layout.ensure()
            for stale in self.layout.meta_dir.glob(".*.tmp"):
                stale.unlink(missing_ok=True)  # left by a crashed controller; we hold the lock
            self._provenance = provenance
            self._session_depth = 1
            try:
                yield
            finally:
                self._session_depth = 0
                self._provenance = None
```

Replace the body of `run` from `self.layout.ensure()` onward with:

```python
        with self.session(), StateStore(self.layout.state_db) as state:
            write_snapshot(self.loaded, self.layout.config_snapshot)
            assert self._provenance is not None
            append_provenance(self.layout.provenance_file, self._provenance)
            if self.rerun_from is not None:
                self._reset_from(state, self.rerun_from)
                self.rerun_from = None
            return self._run_pipeline(state, should_stop)
```

Remove the now-unused separate `self.layout.ensure()` line from `run`.

`src/gqteashmd/cli.py`, `_cmd_run`: replace everything from the `try: orchestrator.layout.ensure()` block to the end of the function with:

```python
    stop_requested = threading.Event()
    with orchestrator.session():
        try:
            configure_logging(orchestrator.layout.logs_dir / "gqteashmd.log")
        except OSError as exc:
            raise ConfigError(
                f"Cannot create project directory {orchestrator.layout.root}: "
                f"{exc.strerror or exc}"
            ) from exc
        logger = logging.getLogger("gqteashmd.cli")
        for message in warnings:
            logger.warning("Preflight: %s", message)

        def _on_signal(signum: int, _frame: object) -> None:
            if not stop_requested.is_set():
                logger.warning("Stop requested (signal %d): stopping running tasks...", signum)
            stop_requested.set()

        previous: dict[int, Any] = {}
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, _on_signal)
            summary = orchestrator.run(should_stop=stop_requested.is_set)
        finally:
            for prev_sig, handler in previous.items():
                signal.signal(prev_sig, handler)
```

Keep the remaining `if summary.stopped: ...` and `print` code after the `with` block, unchanged.

**Risk:** `ProjectLock` creates `meta_dir` (its parent) before `ensure()` runs. That is intended: `.gqteashmd/` is the only thing a refused run may create. The P2 backstop also has to move. An `OSError` from `ProjectLock`'s `mkdir` becomes `ConfigError`; wrap the `with orchestrator.session():` line accordingly if the existing P2 test `test_unwritable_project_location...` (or similar) expects `ConfigError`. Report any test you had to adapt.

- [ ] **Step 4: Run the tests to verify they pass.**
  Run `pytest tests -v -W error -m "not slow" -x && mypy`.
  Expected: all pass, including the P1/P2 lock, preflight and provenance tests.

- [ ] **Step 5: Lint, commit**

```bash
ruff format src tests tools && ruff check src tests tools
git add src/gqteashmd/orchestrator.py src/gqteashmd/cli.py src/gqteashmd/util/fileio.py src/gqteashmd/project/provenance.py tests
git commit -m "fix: take the project lock before creating or logging anything"
```

---

### Task 3: State store errors at query time

**Files:**
- Modify: `src/gqteashmd/project/state.py`
- Test: `tests/project/test_state.py`

**Interfaces:**
- **Query-time errors.** Every public `StateStore` method converts `sqlite3.Error` raised while querying into `ProjectError`. Examples are a disk I/O error, a database locked past the 30 s timeout, or use after `close()`. The message names the database and the operation and advises a remedy: check free disk space, and check that no other program has the database open.
- **Existing behaviour stays.** The open-time handling, the unknown-status `ProjectError`s and the read-only mode are unchanged.

- [ ] **Step 1: Write the failing tests** in `tests/project/test_state.py`. Merge the imports with the existing ones.

```python
import sqlite3

import pytest

from gqteashmd.errors import ProjectError
from gqteashmd.project.state import StageStatus, StateStore


def test_unknown_stored_status_is_a_project_error(tmp_path):
    db = tmp_path / "state.db"
    with StateStore(db) as store:
        store.set_stage("prepare", StageStatus.DONE, fingerprint="x")
    conn = sqlite3.connect(db)
    conn.execute("UPDATE stages SET status = 'from-the-future'")
    conn.commit()
    conn.close()
    with StateStore(db) as store, pytest.raises(ProjectError, match="unknown stage status"):
        store.get_stage("prepare")


def test_use_after_close_is_a_project_error(tmp_path):
    store = StateStore(tmp_path / "state.db")
    store.close()
    with pytest.raises(ProjectError, match="state database"):
        store.get_stage("prepare")


def test_query_time_sqlite_errors_become_project_errors(tmp_path, monkeypatch):
    store = StateStore(tmp_path / "state.db")

    class Broken:
        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError("disk I/O error")

        def executemany(self, *args, **kwargs):
            raise sqlite3.OperationalError("disk I/O error")

        def close(self):
            pass

    monkeypatch.setattr(store, "_conn", Broken())
    for call in (
        lambda: store.get_stage("prepare"),
        lambda: store.set_stage("prepare", StageStatus.RUNNING),
        lambda: store.upsert_tasks("prepare", ["t0"]),
        lambda: store.get_tasks("prepare"),
        lambda: store.task_counts("prepare"),
    ):
        with pytest.raises(ProjectError, match="disk I/O error"):
            call()
```

- [ ] **Step 2: Run them to verify they fail.**
  Run: `pytest tests/project/test_state.py -v -W error`
  Expected: the last two tests FAIL, with raw `sqlite3.ProgrammingError` and `OperationalError` escaping.

- [ ] **Step 3: Implement** in `state.py`.

Add a decorator, and apply it to every public method after `__init__`:
`get_stage`, `set_stage`, `reset_stage`, `upsert_tasks`, `get_tasks`, `get_task`, `set_task`, `requeue_failed`, `task_counts`.

```python
import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _guarded(method: Callable[_P, _R]) -> Callable[_P, _R]:
    """Turn sqlite failures at query time into ProjectError (exit 3) with advice."""

    @functools.wraps(method)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return method(*args, **kwargs)
        except sqlite3.Error as exc:
            store = args[0]
            raise ProjectError(
                f"The state database {getattr(store, '_path', '?')} failed during "
                f"{method.__name__}: {exc}. Check free disk space and that no other program "
                "has the database open, then run the same command again."
            ) from exc

    return wrapper
```

In `__init__`, store `self._path = Path(db_path)` (before connecting). Leave `close()` unguarded.

- [ ] **Step 4: Run the tests to verify they pass.**
  Run: `pytest tests/project tests/test_orchestrator.py -v -W error && mypy`
  Expected: all pass.

- [ ] **Step 5: Lint, full suite, commit**

```bash
ruff format src tests tools && ruff check src tests tools && pytest -m "not slow" -W error
git add src/gqteashmd/project/state.py tests/project/test_state.py
git commit -m "fix: report state database failures at query time as project errors"
```

---

### Task 4: Kill a worker's whole process group

**Files:**
- Modify: `src/gqteashmd/execution/local.py`, `src/gqteashmd/execution/processes.py`
- Test: `tests/execution/test_local.py`, `tests/execution/test_processes.py`

**Interfaces:**
- **After a normal exit.** When a job's leader process exits, the executor sends SIGKILL to the job's process group. `ProcessLookupError` is ignored, because the group may already be empty. This removes stray children a worker left behind, such as a later ORCA subprocess.
- **In `_reap_all`.** After the leader is reaped, whether by exiting within the grace period or by SIGKILL, the executor sends SIGKILL to the group as well.
- **In `terminate_process_group(pid, grace)`.** When the leader is gone, whether before or after the grace period, the function still sends SIGKILL to the group. It ignores `ProcessLookupError`.
- **Why reusing the id is safe.** Linux never hands out a PID equal to an existing process-group id, so signalling the group id after the leader exits reaches only that group's remaining members.

- [ ] **Step 1: Write the failing tests.**

Append to `tests/execution/test_local.py`. Reuse the module's existing helpers for building `Job`s; read the file first.

```python
import os
import sys
import textwrap
import time


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = open(f"/proc/{pid}/stat").read()
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def test_stray_children_are_killed_when_the_worker_exits(tmp_path):
    from gqteashmd.execution.local import Job, LocalExecutor

    script = tmp_path / "leaves_a_child.py"
    pid_file = tmp_path / "child.pid"
    script.write_text(textwrap.dedent(f"""
        import subprocess, sys
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open({str(pid_file)!r}, "w").write(str(child.pid))
    """))
    task = tmp_path / "task.json"
    task.write_text("{}")
    executor = LocalExecutor(max_parallel=1, threads_per_task=1, memory_mb_per_task=100,
                             poll_interval=0.05, command=[sys.executable, str(script)])
    finished = executor.run([Job("t", task, tmp_path / "w.log")], on_start=lambda j, p: None,
                            on_finish=lambda o: None, should_stop=lambda: False)
    assert finished
    child = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while _alive(child) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(child)
```

Append to `tests/execution/test_processes.py`:

```python
def test_terminate_process_group_kills_members_after_the_leader_exits(tmp_path):
    import subprocess
    import sys
    import textwrap
    import time

    from gqteashmd.execution.processes import _alive, terminate_process_group

    pid_file = tmp_path / "child.pid"
    leader = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(f"""
            import signal, subprocess, sys, time
            child = subprocess.Popen([sys.executable, "-c",
                "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
            open({str(pid_file)!r}, "w").write(str(child.pid))
            time.sleep(60)
        """)],
        start_new_session=True,
    )
    deadline = time.monotonic() + 10
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    child = int(pid_file.read_text())
    terminate_process_group(leader.pid, grace=2.0)  # the leader dies on SIGTERM; the child ignores it
    leader.wait(timeout=5)
    deadline = time.monotonic() + 5
    while _alive(child) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(child)
```

(If `_alive` cannot be imported, because it is private, define a local helper as in the first test.)

- [ ] **Step 2: Run them to verify they fail.**
  Run: `pytest tests/execution -v -W error -k "stray or after_the_leader"`
  Expected: FAIL, because the children survive.

- [ ] **Step 3: Implement.**

`local.py`: in `run`, where a finished job is detected (`code is not None`), call `_signal_group(item.process.pid, signal.SIGKILL)` before `item.log.close()`. In `_reap_all`, after `item.process.wait(...)` (in both branches), call `_signal_group(item.process.pid, signal.SIGKILL)`.

`processes.py`: replace `terminate_process_group` with:

```python
def terminate_process_group(pid: int, grace: float) -> None:
    """SIGTERM the process group led by `pid`; SIGKILL whatever remains of it after the leader
    exits or after `grace` seconds."""
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and _alive(pid):
        time.sleep(0.05)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)
```

- [ ] **Step 4: Run the tests to verify they pass.**
  Run: `pytest tests/execution tests/test_orchestrator.py -v -W error && pytest -m slow -W error tests/test_orchestrator.py`
  Expected: all pass, including the slow kill/resume tests.

- [ ] **Step 5: Lint, full suite, commit**

```bash
ruff format src tests tools && ruff check src tests tools && pytest -m "not slow" -W error
git add src/gqteashmd/execution tests/execution
git commit -m "fix: kill a worker's whole process group, not only its leader"
```

---

### Task 5: Worker robustness and picklable task errors

**Files:**
- Modify: `src/gqteashmd/worker.py`, `src/gqteashmd/errors.py`
- Test: `tests/test_worker.py`, `tests/test_errors_fileio.py`

**Interfaces:**
- **Unreadable `task.json`.** `run_task_file(task_file)` handles a `task.json` that is missing, not valid JSON, or missing a key (`OSError`, `ValueError`, `KeyError`). It writes `{"status": "error", "error_type": "ProjectError", "message": "Cannot read task file …", "retryable": false}` to `task_file.parent / "result.json"` and returns 1. The task directory is the task file's parent by construction (`Orchestrator._prepare_job`).
- **Unserialisable result.** A result that cannot be serialised as JSON fails with `TypeError` or `ValueError` (for example a circular reference). Either gives the existing "task result is not JSON-serializable" error result.
- **Pickling.** `TaskError` and its subclasses (`ElectronicStructureError`) round-trip through `pickle` with their message and `retryable` flag preserved.

- [ ] **Step 1: Write the failing tests.**

Append to `tests/test_worker.py`:

```python
import json

from gqteashmd.worker import RESULT_FILE, run_task_file


def test_unreadable_task_file_gives_an_error_result(tmp_path):
    task_dir = tmp_path / "tasks" / "t0"
    task_dir.mkdir(parents=True)
    task_file = task_dir / "task.json"
    task_file.write_text("{ not json")
    assert run_task_file(task_file) == 1
    result = json.loads((task_dir / RESULT_FILE).read_text())
    assert result["status"] == "error" and result["error_type"] == "ProjectError"
    assert result["retryable"] is False and "Cannot read task file" in result["message"]


def test_task_file_missing_keys_gives_an_error_result(tmp_path):
    task_file = tmp_path / "task.json"
    task_file.write_text("{}")
    assert run_task_file(task_file) == 1
    assert json.loads((tmp_path / RESULT_FILE).read_text())["error_type"] == "ProjectError"
```

Also add a circular-reference result case. Use the existing fake-stage mechanism, or monkeypatch a stage's `run_task` to return `d = {}; d["self"] = d`. Follow the pattern of the existing "not JSON-serializable" test in this file, and assert the same error result.

Append to `tests/test_errors_fileio.py`:

```python
def test_task_errors_survive_pickling():
    import pickle

    from gqteashmd.errors import ElectronicStructureError, TaskError

    for err in (TaskError("boom", retryable=True), ElectronicStructureError("scf", retryable=True),
                ElectronicStructureError("spin")):
        copy = pickle.loads(pickle.dumps(err))
        assert type(copy) is type(err) and str(copy) == str(err)
        assert copy.retryable == err.retryable
```

- [ ] **Step 2: Run them to verify they fail.**
  Run: `pytest tests/test_worker.py tests/test_errors_fileio.py -v -W error`
  Expected: FAIL. The JSON error escapes, the KeyError escapes, and unpickling raises `TypeError` (missing `retryable`).

- [ ] **Step 3: Implement.**

`errors.py`: add a rebuild helper, plus a `__reduce__` method on `TaskError` that every subclass inherits:

```python
def _rebuild_task_error(cls: type[TaskError], message: str, retryable: bool) -> TaskError:
    return cls(message, retryable=retryable)
```

```python
    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (_rebuild_task_error, (type(self), str(self), self.retryable))
```

(Add `from typing import Any`. `_rebuild_task_error` must be defined at module level, after the class, so that pickle can find it.)

`worker.py`: at the start of `run_task_file`, replace `spec = read_json(task_file)` and `task_dir = Path(spec["task_dir"])` with:

```python
    try:
        spec = read_json(task_file)
        task_dir = Path(spec["task_dir"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        message = f"Cannot read task file {task_file}: {type(exc).__name__}: {exc}"
        print(message, file=sys.stderr)
        write_json_atomic(
            Path(task_file).parent / RESULT_FILE,
            _error(ProjectError(message), retryable=False, with_traceback=False),
        )
        return 1
```

Change the serialisation guard from `except TypeError as exc:` to `except (TypeError, ValueError) as exc:`, and keep its message.

- [ ] **Step 4: Run the tests to verify they pass.**
  Run: `pytest tests/test_worker.py tests/test_errors_fileio.py tests/electronic/test_base.py -v -W error && mypy`
  Expected: all pass.

- [ ] **Step 5: Lint, full suite, commit**

```bash
ruff format src tests tools && ruff check src tests tools && pytest -m "not slow" -W error
git add src/gqteashmd/worker.py src/gqteashmd/errors.py tests/test_worker.py tests/test_errors_fileio.py
git commit -m "fix: report unreadable task files as results and make task errors picklable"
```

---

### Task 6: A DONE stage whose directory was deleted is recomputed

**Files:**
- Modify: `src/gqteashmd/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- **Missing directory.** In `_run_pipeline`, when a stage's record is DONE with a matching fingerprint but its `stage_dir` does not exist, the controller logs a warning ("Stage X: results directory … is missing; recomputing"). It then calls `state.reset_stage(stage.name)` and runs the stage again. There is nothing to supersede.
- **Downstream stages.** The recomputed stage produces the same fingerprint, so downstream DONE stages stay up to date. A downstream stage whose own directory is missing is recomputed by the same rule.

- [ ] **Step 1: Write the failing test.** Append it to `tests/test_orchestrator.py`, using the module's helpers to run the fake pipeline twice:

```python
def test_a_done_stage_whose_directory_was_deleted_is_recomputed(tmp_path, write_input, caplog):
    import shutil

    from gqteashmd.config.loader import load_config
    from gqteashmd.orchestrator import Orchestrator

    path = write_input(tmp_path)
    first = Orchestrator(load_config(path), pipeline_ref=FAKE_PIPELINE).run()
    assert first.completed
    orch = Orchestrator(load_config(path), pipeline_ref=FAKE_PIPELINE)
    victim = orch.stages[0]
    shutil.rmtree(orch.layout.stage_dir(0, victim.name))
    with caplog.at_level("WARNING", logger="gqteashmd.orchestrator"):
        second = orch.run()
    assert victim.name in second.completed
    assert all(name in second.up_to_date for name in first.completed if name != victim.name)
    assert orch.layout.stage_dir(0, victim.name).is_dir()
    assert "is missing; recomputing" in caplog.text
```

The logger has `propagate=False` only after `configure_logging`, and these tests do not call it, so `caplog` sees the record. If the module's existing tests use another way to capture logs, follow that.

- [ ] **Step 2: Run it to verify it fails.**
  Run: `pytest tests/test_orchestrator.py -v -W error -k deleted`
  Expected: FAIL, because the stage is reported as up to date.

- [ ] **Step 3: Implement** in `_run_pipeline`, just before the `if record is not None and record.status == StageStatus.DONE:` block:

```python
            if (
                record is not None
                and record.status == StageStatus.DONE
                and not stage_dir.exists()
            ):
                log.warning(
                    "Stage %s: results directory %s is missing; recomputing", stage.name, stage_dir
                )
                state.reset_stage(stage.name)
                record = None
```

- [ ] **Step 4: Run the tests to verify they pass.**
  Run: `pytest tests/test_orchestrator.py -v -W error`
  Expected: all pass.

- [ ] **Step 5: Lint, full suite, commit**

```bash
ruff format src tests tools && ruff check src tests tools && pytest -m "not slow" -W error
git add src/gqteashmd/orchestrator.py tests/test_orchestrator.py
git commit -m "fix: recompute a completed stage whose results directory was deleted"
```

---

### Task 7: CLI robustness and the missing CLI tests

**Files:**
- Modify: `src/gqteashmd/cli.py`
- Test: `tests/test_cli.py`, `tests/test_cli_output.py`, `tests/test_preflight.py`

**Interfaces:**
- **Ctrl-C handling.** `main()` catches `KeyboardInterrupt` anywhere, including before `run` installs its signal handlers. It prints `gqteashmd: interrupted` to stderr once and returns `EXIT_STOPPED` (5). The `finally: close_logging()` block stays.
- **New tests, with no new behaviour:**
  - an unexpected exception in the controller writes its traceback to `logs/gqteashmd.log`, but not to the console;
  - `status` works while a controller holds the lock, and reports it as running;
  - preflight warnings are printed exactly once;
  - `_module_available` returns False when `find_spec` raises `ImportError`.

- [ ] **Step 1: Write the failing tests.** Use the existing CLI test helpers and the fake pipeline reference. Read `tests/test_cli.py` and `tests/test_cli_output.py` first and follow their patterns.

```python
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


def test_internal_error_traceback_goes_to_the_log_file_only(tmp_path, write_input, monkeypatch,
                                                            capsys):
    import json

    from gqteashmd.cli import main
    from gqteashmd.config.loader import load_config
    from gqteashmd.testing.fake_stages import BEHAVIOUR_ENV, make_pipeline

    path = write_input(tmp_path)
    first = make_pipeline(load_config(path).config)[0].name
    monkeypatch.setenv(BEHAVIOUR_ENV, json.dumps({"plan_raises": [first]}))
    assert main(["run", str(path), "--pipeline", FAKE_PIPELINE]) == 1
    err = capsys.readouterr().err
    assert "internal error" in err and "Traceback" not in err
    log_text = (tmp_path / "run" / "logs" / "gqteashmd.log").read_text()
    assert "Traceback (most recent call last)" in log_text


def test_status_while_a_controller_holds_the_lock(tmp_path, write_input, capsys):
    from gqteashmd.cli import main
    from gqteashmd.config.loader import load_config
    from gqteashmd.project.layout import ProjectLayout
    from gqteashmd.project.lock import ProjectLock

    path = write_input(tmp_path)
    assert main(["run", str(path), "--pipeline", FAKE_PIPELINE]) == 0
    layout = ProjectLayout(load_config(path).config.project.directory)
    capsys.readouterr()
    with ProjectLock(layout.lock_file):
        assert main(["status", str(path)]) == 0
    out = capsys.readouterr().out
    assert f"running (pid {os.getpid()})" in out


def test_preflight_warnings_are_printed_once(tmp_path, write_input, capsys):
    from gqteashmd.cli import main

    path = write_input(tmp_path, total_cores=100000, total_memory_mb=10**9)
    main(["run", str(path), "--pipeline", FAKE_PIPELINE])
    err = capsys.readouterr().err
    assert err.count("Preflight:") >= 1
    for line in {line for line in err.splitlines() if "Preflight:" in line}:
        assert err.count(line) == 1
```

In the internal-error test: if the console handler writes INFO records to stderr, assert only that no "Traceback" line reaches stderr. If `write_input`'s project directory is not `tmp_path / "run"`, take the log path from `ProjectLayout(load_config(path).config.project.directory).logs_dir`.

In `tests/test_preflight.py`:

```python
def test_module_check_treats_import_errors_as_missing(monkeypatch):
    import importlib.util

    from gqteashmd import preflight

    def broken(name):
        raise ImportError("broken parent package")

    monkeypatch.setattr(importlib.util, "find_spec", broken)
    assert preflight._module_available("anything") is False
```

- [ ] **Step 2: Run them to verify they fail.**
  Run: `pytest tests/test_cli.py tests/test_cli_output.py tests/test_preflight.py -v -W error -k "ctrl_c or traceback or status_while or printed_once or import_errors"`
  Expected: the Ctrl-C test FAILS, because `KeyboardInterrupt` escapes. The others may already pass. They are regression tests for deferred gaps, so record which ones passed first time.

- [ ] **Step 3: Implement.** In `main()`, add this before `except ShmdError`:

```python
    except KeyboardInterrupt:
        print("gqteashmd: interrupted", file=sys.stderr)
        return EXIT_STOPPED
```

- [ ] **Step 4: Run the tests to verify they pass.**
  Run: `pytest tests/test_cli.py tests/test_cli_output.py tests/test_preflight.py -v -W error && mypy`
  Expected: all pass.

- [ ] **Step 5: Lint, full suite, commit**

```bash
ruff format src tests tools && ruff check src tests tools && pytest -m "not slow" -W error
git add src/gqteashmd/cli.py tests/test_cli.py tests/test_cli_output.py tests/test_preflight.py
git commit -m "fix: exit cleanly on Ctrl-C anywhere and test the remaining CLI paths"
```

---

### Task 8: A fast test tier: the `pyscf` marker

**Files:**
- Modify: `pyproject.toml`
- Modify every test module that runs real PySCF calculations: `tests/electronic/test_pyscf_*.py`, `tests/electronic/test_couplings.py`, `tests/electronic/test_state.py`, `tests/electronic/test_gradients_fd.py`, `tests/electronic/test_couplings_fd.py`, `tests/test_cli_single_point.py`, and any other module whose tests construct `PySCFBackend` or call PySCF directly. Find them with `grep -rl "PySCFBackend\|from pyscf" tests`.

**Interfaces:**
- **New marker.** Register `"pyscf: runs PySCF calculations (deselect for a fast loop with -m 'not slow and not pyscf')"` in `pyproject.toml`.
- **Where it applies.** Add a module-level `pytestmark = pytest.mark.pyscf` to every such module. If a module already has `pytestmark`, use a list.
- **Fast tier.** `pytest -m "not slow and not pyscf" -W error` must collect every orchestrator, CLI, config, overlaps and tracking test and finish in about 1–2 minutes. Measure the time and report it.
- **Default tiers unchanged.** `pytest -m "not slow"` still runs the PySCF tests. CI is unchanged.
- **Lazy-import test.** `test_importing_the_cli_and_worker_does_not_import_pyscf` is cheap, so it stays in the fast tier. Move it to `tests/test_package.py` if its module gets the marker.

- [ ] **Step 1: Mark the modules.** Add the marker to every module that needs it, then check that the fast tier excludes them all:
  `pytest -m "not slow and not pyscf" --collect-only -q | tail -3`
  `grep -rL "pytest.mark.pyscf" $(grep -rl "PySCFBackend\|from pyscf" tests)` must print nothing.
- [ ] **Step 2: Time the three tiers.** Run each one and report its wall time:
  `pytest -m "not slow and not pyscf" -W error`, `pytest -m "not slow" -W error`, `pytest -W error`.
- [ ] **Step 3: Commit**

```bash
ruff format src tests tools && ruff check src tests tools
git add pyproject.toml tests
git commit -m "test: add a pyscf marker for a fast development test tier"
```

---

### Task 9: Documentation current with P3c

**Files:**
- Modify: `CLAUDE.md`, `USER_MANUAL.md`, `docs/superpowers/plans/2026-11-06-gqteashmd-master-plan.md`

- [ ] **Step 1: USER_MANUAL.md.**
  - In the CLI and run sections, record three behaviours: the lock is taken before anything is created, so a refused run writes nothing but `.gqteashmd/`; Ctrl-C at any time exits with code 5; a completed stage whose directory was deleted is recomputed.
  - In §9, add the new error messages: file locking unavailable, and state-database failure during an operation.
  - Update the troubleshooting text where it mentioned these gaps.
- [ ] **Step 2: CLAUDE.md.**
  - Status: P3c done on `feat/p3c-hardening`, not merged.
  - Architecture: `Orchestrator.session()`; the lock comes before `ensure()` and logging; the state store's guarded queries; whole-group kills.
  - Test tiers, in this order: the fast tier, `-m "not slow and not pyscf"`, with its measured time; then `-m "not slow"`; then everything.
  - Remove the fixed items from Known limitations, and list the items moved to P4 and P7b per this plan's scope rulings.
  - Remaining work: P4 is next.
- [ ] **Step 3: Master plan.** Mark P3c as delivered in §9, and note the items moved to P4 and P7b.
- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md USER_MANUAL.md docs/superpowers/plans/2026-11-06-gqteashmd-master-plan.md
git commit -m "docs: document the P3c hardening sweep"
```

---

## Self-review notes

- **Coverage of the ledger list:**
  - lock items → Task 1;
  - lock before ensure/logging, git under the lock, `.tmp` cleanup, CPU model → Task 2;
  - query-time sqlite errors, unknown-status and closed-connection tests → Task 3;
  - group SIGKILL → Task 4;
  - unreadable `task.json`, `ValueError` in serialisation, `TaskError` pickling → Task 5;
  - deleted DONE stage directory → Task 6;
  - Ctrl-C, traceback-in-log, concurrent-status, warning-once and `ImportError`-branch tests → Task 7;
  - fast tier → Task 8;
  - documentation → Task 9.

  The items moved to P4 and P7b are recorded under Scope rulings.
- **Consistency:**
  - `Orchestrator.session()` is used by Task 2's CLI change and by `run()`.
  - `write_bytes_atomic` keeps its signature.
  - `terminate_process_group(pid, grace)` keeps its signature.
  - `run_task_file(task_file) -> int` keeps its signature.
  - No test changes an exit code.
