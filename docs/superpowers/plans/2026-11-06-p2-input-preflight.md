# P2: Input Schema, Preflight & Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make gqteashmd's input and start-up production-grade:
- fix the two P1 residual defects;
- require user-given workstation hardware (D17) and check every task setting against it;
- allow per-stage resources;
- run preflight checks before any work;
- record run provenance;
- support review pauses;
- clear the P2 items of the P1 deferred list;
- type-check with mypy;
- keep CLAUDE.md and USER_MANUAL.md current.

**Architecture:** The existing layers stay as they are (CLI → Orchestrator → LocalExecutor → worker → stage). Additions:
- `ResourcesSection` gains hardware keys and per-stage overrides, validated in the schema.
- A new `gqteashmd.preflight` module runs before the project directory is touched.
- A new `gqteashmd.project.provenance` module appends one record per run.
- The orchestrator builds one executor per stage and can pause after configured stages.
- Console vs file logging is separated with a record attribute.

**Tech Stack:** Python 3.12, pydantic v2, stdlib (`sqlite3`, `logging`, `shutil`, `importlib`, `platform`, `subprocess`), pytest, ruff 0.17.4, mypy.

**Spec:** [`docs/superpowers/plans/2026-11-06-gqteashmd-master-plan.md`](2026-11-06-gqteashmd-master-plan.md) §3.5, §4.1–4.3, §6, D17 and D20; [`CLAUDE.md`](../../../CLAUDE.md) "Known limitations".

## Global Constraints

- **Clean room:** no code from SHARC 4.1 is copied, translated or adapted.
- **Licence:** MIT.
- **Runtime:** Linux only (target Oracle Linux 9). All commands run inside WSL Oracle Linux 9.5:
  `wsl -d OracleLinux_9_5 -- bash -lc 'cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && conda activate gqteashmd-dev && <command>'`.
  Create and edit files with the file tools at Windows paths under `C:\Users\ajc\OneDrive\gqteaSHMD\`, with LF line endings.
- **Python 3.12.** The only runtime dependency is `pydantic>=2.6`; everything else is stdlib.
- **Style:** ruff 0.17.4 (line length 100, rules E, F, W, I, B, UP, SIM). Type hints on all public functions. Before every commit run `ruff format src tests && ruff check --fix src tests`.
- **Git:** branch `feat/p2-input-preflight`, WSL Git only. Pre-commit hooks are installed; never use `--no-verify`. Use conventional-commit subjects. End every commit message with the `Co-Authored-By:` trailer for the model that wrote it. Do not push.
- **Atomic file writes** (`write_text_atomic` / `write_json_atomic`). The controller is the only writer of `state.db`. `status` opens the database read-only (Task 2).
- **Errors** reach the user as `gqteashmd: error: <message>` with exit codes 0, 1 (internal), 2 (`ConfigError`), 3 (`ProjectError`), 4 (`StageFailedError`) and 5 (stopped or paused).
- **Units in key names:** `_mb`, `_s`, ….
- **Documentation:** the last task updates CLAUDE.md and USER_MANUAL.md to match the implementation. USER_MANUAL.md must distinguish [IMPLEMENTED] from [PLANNED].

## Scope rulings (recorded in the master plan by Task 11)

- **Later input sections stay out of P2.** `[electronic]`, `[optimize]`, `[frequencies]`, `[sampling]`, `[excitations]`, `[selection]`, `[dynamics]` and `[analysis]` are added by the sub-project that consumes them (P3–P8), because their keys depend on open items O3, O4, O8 and O19. State labels (`S0`/`S1`/`T1`) arrive with `[electronic]` in P3.
- **Deferred P1 findings that stay deferred:**
  - refused runs still writing state (snapshot before refusal; SKIPPED-over-DONE): P4, together with optional stages;
  - the worker orphan window between Popen and on_start: P7b;
  - SIGKILL to the group after the leader exits: P3;
  - an unreadable `task.json`: P3;
  - `.tmp` cleanup, a DONE stage directory deleted by hand, and `ensure()`/logging set up before the lock: P3 (hardening sweep).

## File Structure

```text
src/gqteashmd/__init__.py            # Task 1: NullHandler
src/gqteashmd/logging_setup.py       # Task 1: console filter, close_logging()
src/gqteashmd/cli.py                 # Tasks 1, 2, 3, 4, 6, 8: file-only error logging, read-only status, template, preflight, pause
src/gqteashmd/project/state.py       # Task 2: robust open, read-only mode, COMMIT in try, unknown status
src/gqteashmd/chem/xyz.py            # Task 3: finite coordinates
src/gqteashmd/config/loader.py       # Task 3: ~ expansion
src/gqteashmd/stages/base.py         # Tasks 3, 6: is_safe_name(), required_modules
src/gqteashmd/stages/pipeline.py     # Task 3: use is_safe_name()
src/gqteashmd/config/schema.py       # Tasks 4, 6, 8: hardware, StageResources, EffectiveResources, min_free_disk_mb, WorkflowSection
src/gqteashmd/orchestrator.py        # Tasks 5, 6, 7, 8, 9: per-stage executor, preflight, provenance, pause, robustness
src/gqteashmd/testing/fake_stages.py # Tasks 5, 6, 9: threads in result, requires, bad_fingerprint
src/gqteashmd/preflight.py           # Task 6 (new)
src/gqteashmd/project/provenance.py  # Task 7 (new)
src/gqteashmd/project/layout.py      # Task 7: provenance_file
src/gqteashmd/project/lock.py        # Task 9: retry once, "pid unknown"
tests/conftest.py                    # Task 4: hardware keys in write_input
tests/test_cli_output.py             # Task 1 (new)
tests/project/test_state.py          # Task 2
tests/chem/test_xyz.py, tests/config/test_loader.py, tests/stages/test_base_pipeline.py, tests/stages/test_prepare.py, tests/test_cli.py   # Task 3
tests/config/test_resources.py       # Task 4 (new)
tests/test_orchestrator.py, tests/test_worker.py   # Tasks 5, 8, 9
tests/test_preflight.py              # Task 6 (new)
tests/project/test_provenance.py     # Task 7 (new)
tests/project/test_layout_lock.py    # Task 9
pyproject.toml, environment.yml, .github/workflows/ci.yml   # Task 10: mypy
CLAUDE.md, USER_MANUAL.md, docs/superpowers/plans/2026-11-06-gqteashmd-master-plan.md   # Task 11
```

---

### Task 1: Single, clean error output on the console

**Files:**
- Modify: `src/gqteashmd/__init__.py`, `src/gqteashmd/logging_setup.py`, `src/gqteashmd/cli.py`
- Test: `tests/test_cli_output.py` (new)

**Interfaces:**
- Consumes: `cli.main`, `_cmd_*` (existing); `configure_logging(log_file, *, verbose=False)` (existing); `gqteashmd.testing.fake_stages.BEHAVIOUR_ENV`.
- Produces:
  - `logging_setup.FILE_ONLY: dict[str, bool] = {"console": False}`, which is passed as `extra=` on log calls whose records must not reach the console.
  - `logging_setup.close_logging() -> None`.
  - `main()` always calls `close_logging()` in `finally`.

- [ ] **Step 1: Write the failing tests** in `tests/test_cli_output.py`

```python
"""Console output of real `python -m gqteashmd` processes (capsys cannot see these problems)."""

import json
import os
import subprocess
import sys

from gqteashmd.testing.fake_stages import BEHAVIOUR_ENV

FAKE = "gqteashmd.testing.fake_stages:make_pipeline"
FAST = "poll_interval_s = 0.05\nstop_grace_period_s = 2.0"


def _cli(args, cwd, env=None):
    return subprocess.run(
        [sys.executable, "-m", "gqteashmd", *args],
        cwd=cwd,
        env=env or dict(os.environ),
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_config_error_is_printed_exactly_once(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("project = [")
    proc = _cli(["validate", str(bad)], tmp_path)
    assert proc.returncode == 2
    assert proc.stderr.count("is not valid TOML") == 1
    assert proc.stderr.startswith("gqteashmd: error:")


def test_internal_error_has_no_traceback_on_console(tmp_path):
    script = (
        "import sys, gqteashmd.cli as cli\n"
        "def boom(args):\n"
        "    raise RuntimeError('kaboom')\n"
        "cli._cmd_validate = boom\n"
        "sys.exit(cli.main(['validate', 'x.toml']))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 1
    assert proc.stderr.count("internal error: RuntimeError: kaboom") == 1
    assert "Traceback" not in proc.stderr


def test_stage_failure_is_printed_once_and_logged_to_file(tmp_path, write_input):
    path = write_input(tmp_path, execution=FAST)
    env = dict(os.environ)
    env[BEHAVIOUR_ENV] = json.dumps({"fail": ["sampling:t000"]})
    proc = _cli(["run", str(path), "--pipeline", FAKE], tmp_path, env=env)
    assert proc.returncode == 4
    assert proc.stderr.count("Stage 'sampling' failed") == 1
    log_text = (tmp_path / "run" / "logs" / "gqteashmd.log").read_text()
    assert "Stage 'sampling' failed" in log_text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_cli_output.py -v`
Expected: 3 FAILED. The message is counted twice, and "Traceback" is present.

- [ ] **Step 3: Implement**

`src/gqteashmd/__init__.py`:

```python
"""gqteashmd: automated surface-hopping molecular dynamics workflow."""

import logging

__version__ = "0.1.0.dev0"

# Without a handler, Python's last-resort handler would echo our log records to stderr.
logging.getLogger("gqteashmd").addHandler(logging.NullHandler())
```

`src/gqteashmd/logging_setup.py` (full new content):

```python
"""Logging: INFO to the console, DEBUG to the project log file."""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

# Pass as `extra=FILE_ONLY` for records that must reach the log file but not the console
# (e.g. errors the CLI already prints in its own format).
FILE_ONLY: dict[str, bool] = {"console": False}


class _ConsoleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, "console", True))


def _remove_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def configure_logging(log_file: Path | None, *, verbose: bool = False) -> None:
    logger = logging.getLogger("gqteashmd")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    _remove_handlers(logger)
    formatter = logging.Formatter(_FORMAT, "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    console.addFilter(_ConsoleFilter())
    logger.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def close_logging() -> None:
    """Close all gqteashmd handlers and return to the silent import-time state."""
    logger = logging.getLogger("gqteashmd")
    _remove_handlers(logger)
    logger.addHandler(logging.NullHandler())
    logger.propagate = True
```

In `src/gqteashmd/cli.py`:
- Change the import to `from gqteashmd.logging_setup import FILE_ONLY, close_logging, configure_logging`.
- Install the signal handlers inside the `try` of `_cmd_run`.
- Replace `main()`.

```python
    stop_requested = threading.Event()

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
        for sig, handler in previous.items():
            signal.signal(sig, handler)
```

(Add `from typing import Any` to the imports.)

```python
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = logging.getLogger("gqteashmd.cli")
    try:
        return args.func(args)
    except ShmdError as exc:
        log.error("%s", exc, extra=FILE_ONLY)
        print(f"gqteashmd: error: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:
        log.error("Unexpected error", exc_info=True, extra=FILE_ONLY)
        print(
            f"gqteashmd: error: internal error: {type(exc).__name__}: {exc} "
            "(please report this; details in the project log if one was configured)",
            file=sys.stderr,
        )
        return 1
    finally:
        close_logging()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_cli_output.py tests/test_cli.py -v`
Expected: all PASS, including the existing CLI tests.

- [ ] **Step 5: Full suite, lint, commit**

```bash
pytest -m "not slow" -W error && ruff format src tests && ruff check --fix src tests
git add src/gqteashmd/__init__.py src/gqteashmd/logging_setup.py src/gqteashmd/cli.py tests/test_cli_output.py
git commit -m "fix: print each error once and keep tracebacks out of the console"
```

---

### Task 2: Robust and read-only state database

**Files:**
- Modify: `src/gqteashmd/project/state.py`, `src/gqteashmd/cli.py` (`_cmd_status`)
- Test: `tests/project/test_state.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `ProjectError`.
- Produces:
  - `StateStore(db_path: Path, *, readonly: bool = False)`. Any failure to open or read the database raises `ProjectError("State database … is unreadable or corrupt …")`, and an unknown status value raises `ProjectError`.
  - `_cmd_status` uses `readonly=True`.

- [ ] **Step 1: Write the failing tests** (append to `tests/project/test_state.py`)

```python
import sqlite3


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
        with pytest.raises(sqlite3.OperationalError):
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
```

Append to `tests/test_cli.py`:

```python
def test_status_does_not_modify_the_state_database(project, capsys):
    assert main(["run", str(project), "--pipeline", FAKE]) == 0
    db = project.parent / "run" / ".gqteashmd" / "state.db"
    before = db.read_bytes()
    capsys.readouterr()
    assert main(["status", str(project), "--pipeline", FAKE]) == 0
    assert db.read_bytes() == before
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/project/test_state.py tests/test_cli.py::test_status_does_not_modify_the_state_database -v`
Expected: the new tests FAIL. `readonly` is an unexpected keyword, the directory path gives a raw `OperationalError`, and an unknown status gives `ValueError`.

- [ ] **Step 3: Implement** in `src/gqteashmd/project/state.py`

Add `import contextlib` at the top. Add the helpers below after `_now()`. Then replace `__init__` and `_transaction`, and use the helpers wherever `StageStatus(...)` or `TaskStatus(...)` is built from a row (`get_stage`, `get_tasks`, `get_task`, `task_counts`):

```python
def _stage_status(value: str) -> StageStatus:
    try:
        return StageStatus(value)
    except ValueError as exc:
        raise ProjectError(
            f"State database contains unknown stage status {value!r}; it was written by a "
            "newer gqteashmd version or is corrupt."
        ) from exc


def _task_status(value: str) -> TaskStatus:
    try:
        return TaskStatus(value)
    except ValueError as exc:
        raise ProjectError(
            f"State database contains unknown task status {value!r}; it was written by a "
            "newer gqteashmd version or is corrupt."
        ) from exc
```

```python
class StateStore:
    def __init__(self, db_path: Path, *, readonly: bool = False) -> None:
        conn: sqlite3.Connection | None = None
        try:
            if readonly:
                uri = Path(db_path).resolve().as_uri() + "?mode=ro"
                conn = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=30.0)
            else:
                conn = sqlite3.connect(db_path, isolation_level=None, timeout=30.0)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(_SCHEMA)
                conn.execute(
                    "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                    (SCHEMA_VERSION,),
                )
            row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if row is None:
                raise sqlite3.DatabaseError("schema_version entry is missing")
        except sqlite3.DatabaseError as exc:
            if conn is not None:
                conn.close()
            raise ProjectError(
                f"State database {db_path} is unreadable or corrupt: {exc}. "
                "Move it aside to start the project over."
            ) from exc
        (version,) = row
        if version != SCHEMA_VERSION:
            conn.close()
            raise ProjectError(
                f"State database {db_path} has schema version {version}; "
                f"this gqteashmd version expects {SCHEMA_VERSION}."
            )
        self._conn = conn
```

```python
    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._conn.execute("COMMIT")
        except BaseException:
            with contextlib.suppress(sqlite3.Error):
                self._conn.execute("ROLLBACK")
            raise
```

In `task_counts`, replace `counts[TaskStatus(status)] = count` with `counts[_task_status(status)] = count`.

In `src/gqteashmd/cli.py` `_cmd_status`, change `with StateStore(layout.state_db) as state:` to `with StateStore(layout.state_db, readonly=True) as state:`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/project/test_state.py tests/test_cli.py -v`
Expected: all PASS.

- [ ] **Step 5: Full suite, lint, commit**

```bash
pytest -m "not slow" -W error && ruff format src tests && ruff check --fix src tests
git add src/gqteashmd/project/state.py src/gqteashmd/cli.py tests/project/test_state.py tests/test_cli.py
git commit -m "fix: open the state database robustly and read-only for status"
```

---

### Task 3: Input hardening (coordinates, paths, names, validate, prepare tests)

**Files:**
- Modify: `src/gqteashmd/chem/xyz.py`, `src/gqteashmd/config/loader.py`, `src/gqteashmd/stages/base.py`, `src/gqteashmd/stages/pipeline.py`, `src/gqteashmd/cli.py` (`_cmd_validate`)
- Test: `tests/chem/test_xyz.py`, `tests/config/test_loader.py`, `tests/stages/test_base_pipeline.py`, `tests/stages/test_prepare.py`, `tests/test_cli.py`

**Interfaces:**
- Produces:
  - `gqteashmd.stages.base.is_safe_name(name: str) -> bool`. It is true for names matching `[A-Za-z0-9_.-]+` other than `.` and `..`.
  - `parse_xyz` rejects non-finite coordinates.
  - Paths in the input file expand `~`.
  - `validate` requires the geometry path to be a regular file.

- [ ] **Step 1: Write the failing tests**

Append to `tests/chem/test_xyz.py`:

```python
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1e999"])
def test_non_finite_coordinates_are_rejected(value):
    with pytest.raises(ValueError, match="line 3: coordinates must be finite"):
        parse_xyz(f"1\n\nH 0.0 {value} 0.0\n")
```

Append to `tests/config/test_loader.py`:

```python
def test_home_directory_is_expanded(tmp_path, write_input, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    path = write_input(tmp_path, extra="")
    text = path.read_text().replace('directory = "run"', 'directory = "~/runs/test"')
    path.write_text(text)
    cfg = load_config(path).config
    assert cfg.project.directory == (tmp_path / "home" / "runs" / "test").resolve()
```

Append to `tests/stages/test_base_pipeline.py`:

```python
from gqteashmd.stages.base import is_safe_name


@pytest.mark.parametrize("name", [".", ".."])
def test_dot_names_are_not_safe(name):
    assert is_safe_name(name) is False
    with pytest.raises(ValueError):
        TaskSpec(name)
    with pytest.raises(ProjectError, match="Invalid stage name"):
        validate_pipeline([_stage(name)])


def test_ordinary_names_are_safe():
    assert is_safe_name("t001") and is_safe_name("03_sampling.v2")
```

Append to `tests/stages/test_prepare.py`:

```python
O2_XYZ = "2\noxygen\nO 0.0 0.0 0.0\nO 0.0 0.0 1.21\n"


@pytest.mark.parametrize(
    "molecule, ok",
    [
        ("charge = 1\nmultiplicity = 2", True),    # H2O+ : 9 electrons, doublet
        ("charge = 1\nmultiplicity = 1", False),   # 9 electrons cannot be a singlet
        ("multiplicity = 12", False),              # 11 unpaired > 10 electrons
        ("charge = 10", False),                    # 0 electrons
    ],
)
def test_electron_count_rules(tmp_path, write_input, molecule, ok):
    ctx = _ctx(write_input(tmp_path, molecule=molecule))
    if ok:
        assert PrepareStage().run_task(ctx, TaskSpec("molecule"), tmp_path / "t")["n_electrons"] == 9
    else:
        with pytest.raises(TaskError, match="inconsistent"):
            PrepareStage().run_task(ctx, TaskSpec("molecule"), tmp_path / "t")


def test_triplet_oxygen_is_accepted(tmp_path, write_input):
    path = write_input(tmp_path, molecule="multiplicity = 3")
    (tmp_path / "water.xyz").write_text(O2_XYZ)
    result = PrepareStage().run_task(_ctx(path), TaskSpec("molecule"), tmp_path / "t")
    assert (result["n_electrons"], result["multiplicity"]) == (16, 3)


def test_fingerprint_does_not_depend_on_location(tmp_path, write_input):
    first = PrepareStage().fingerprint_inputs(_ctx(write_input(tmp_path)))
    other = tmp_path / "elsewhere"
    other.mkdir()
    second = PrepareStage().fingerprint_inputs(_ctx(write_input(other)))
    assert first == second
```

Append to `tests/test_cli.py`:

```python
def test_validate_rejects_a_directory_as_geometry(tmp_path, write_input, capsys):
    path = write_input(tmp_path)
    (tmp_path / "water.xyz").unlink()
    (tmp_path / "water.xyz").mkdir()
    assert main(["validate", str(path)]) == 2
    assert "not a regular file" in capsys.readouterr().err
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/chem tests/config tests/stages tests/test_cli.py -v`
Expected: the new tests FAIL. The electron-count cases that should be rejected already pass, which is fine; they are regression tests.

- [ ] **Step 3: Implement**

`src/gqteashmd/chem/xyz.py`:
- Add `import math`.
- After the `try/except` block in the atom loop, before the next iteration, add:

```python
        if not all(math.isfinite(v) for v in coords[-1]):
            raise ValueError(f"line {offset}: coordinates must be finite numbers, got {line!r}")
```

`src/gqteashmd/config/loader.py`, in `_resolve_paths`:

```python
    def absolute(p: Path) -> Path:
        p = p.expanduser()
        return p if p.is_absolute() else (base / p).resolve()
```

`src/gqteashmd/stages/base.py`:
- Add after `SAFE_NAME`:

```python
def is_safe_name(name: str) -> bool:
    """True for filesystem-safe task ids and stage names ('.' and '..' are not)."""
    return bool(SAFE_NAME.fullmatch(name)) and name not in {".", ".."}
```

- In `TaskSpec.__post_init__`, replace the check with `if not is_safe_name(self.task_id):`.

`src/gqteashmd/stages/pipeline.py`:
- Change the import to `from gqteashmd.stages.base import Stage, is_safe_name`.
- Replace `if not SAFE_NAME.fullmatch(stage.name):` with `if not is_safe_name(stage.name):`.

`src/gqteashmd/cli.py` `_cmd_validate`:

```python
    if not cfg.molecule.geometry.is_file():
        raise ConfigError(
            f"Geometry file not found or not a regular file: {cfg.molecule.geometry}"
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/chem tests/config tests/stages tests/test_cli.py -v`
Expected: all PASS. The existing test `test_validate_reports_success_and_errors` still finds "Geometry file not found".

- [ ] **Step 5: Full suite, lint, commit**

```bash
pytest -m "not slow" -W error && ruff format src tests && ruff check --fix src tests
git add src/gqteashmd/chem/xyz.py src/gqteashmd/config/loader.py src/gqteashmd/stages/base.py src/gqteashmd/stages/pipeline.py src/gqteashmd/cli.py tests/chem/test_xyz.py tests/config/test_loader.py tests/stages/test_base_pipeline.py tests/stages/test_prepare.py tests/test_cli.py
git commit -m "fix: reject non-finite coordinates and dot names, expand ~ in paths"
```

---

### Task 4: Workstation hardware and per-stage resources in the schema (D17)

**Files:**
- Modify: `src/gqteashmd/config/schema.py`, `tests/conftest.py`, `src/gqteashmd/cli.py` (`TEMPLATE`)
- Test: `tests/config/test_resources.py` (new)

**Interfaces:**
- Consumes: `_Section` (schema).
- Produces:
  - `StageResources(max_parallel_tasks: int | None, threads_per_task: int | None, memory_mb_per_task: int | None)`.
  - `EffectiveResources(max_parallel_tasks: int, threads_per_task: int, memory_mb_per_task: int)` (frozen dataclass).
  - `ResourcesSection` has the required keys `total_cores: int ≥ 1` and `total_memory_mb: int ≥ 100`, plus `stages: dict[str, StageResources]` and `for_stage(name: str) -> EffectiveResources`.
  - Its validator raises when `max_parallel_tasks × threads_per_task > total_cores` or `max_parallel_tasks × memory_mb_per_task > total_memory_mb`, for the global settings and for every override.
  - `ShmdConfig.resources` is required.
  - The `write_input` fixture gains `total_cores=4, total_memory_mb=4000` keyword arguments.

- [ ] **Step 1: Update the fixture.** In `tests/conftest.py`, change the `_write` signature and the resources line:

```python
    def _write(
        directory: Path,
        *,
        molecule: str = "",
        resources: str = "",
        execution: str = "",
        extra: str = "",
        total_cores: int = 4,
        total_memory_mb: int = 4000,
    ) -> Path:
        (directory / "water.xyz").write_text(WATER_XYZ)
        text = (
            '[project]\nname = "test"\ndirectory = "run"\n\n'
            f'[molecule]\ngeometry = "water.xyz"\n{molecule}\n\n'
            f"[resources]\ntotal_cores = {total_cores}\ntotal_memory_mb = {total_memory_mb}\n"
            f"{resources}\n\n"
            f"[execution]\n{execution}\n\n"
            f"{extra}\n"
        )
```

- [ ] **Step 2: Write the failing tests** in `tests/config/test_resources.py`

```python
import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.config.schema import EffectiveResources
from gqteashmd.errors import ConfigError


def test_hardware_keys_are_required(tmp_path, write_input):
    path = write_input(tmp_path)
    path.write_text(path.read_text().replace("total_cores = 4\n", ""))
    with pytest.raises(ConfigError, match=r"resources\.total_cores: Field required"):
        load_config(path)


def test_missing_resources_section_is_an_error(tmp_path):
    (tmp_path / "w.xyz").write_text("1\n\nH 0 0 0\n")
    path = tmp_path / "input.toml"
    path.write_text('[project]\nname = "x"\ndirectory = "run"\n[molecule]\ngeometry = "w.xyz"\n')
    with pytest.raises(ConfigError, match="resources: Field required"):
        load_config(path)


def test_cores_must_fit(tmp_path, write_input):
    path = write_input(tmp_path, resources="max_parallel_tasks = 3\nthreads_per_task = 2")
    with pytest.raises(ConfigError, match="= 6 exceeds total_cores = 4"):
        load_config(path)


def test_memory_must_fit(tmp_path, write_input):
    path = write_input(tmp_path, resources="max_parallel_tasks = 3\nmemory_mb_per_task = 2000")
    with pytest.raises(ConfigError, match="= 6000 MB exceeds total_memory_mb = 4000"):
        load_config(path)


def test_stage_override_must_fit_and_is_named(tmp_path, write_input):
    path = write_input(tmp_path, resources="[resources.stages.frequencies]\nthreads_per_task = 8")
    with pytest.raises(ConfigError, match=r"\[resources\.stages\.frequencies\]"):
        load_config(path)


def test_for_stage_merges_overrides(tmp_path, write_input):
    path = write_input(
        tmp_path,
        resources="max_parallel_tasks = 2\n[resources.stages.frequencies]\n"
        "max_parallel_tasks = 1\nthreads_per_task = 4",
    )
    res = load_config(path).config.resources
    assert res.for_stage("frequencies") == EffectiveResources(1, 4, 2000)
    assert res.for_stage("sampling") == EffectiveResources(2, 1, 2000)


def test_unknown_override_key_is_rejected(tmp_path, write_input):
    path = write_input(tmp_path, resources="[resources.stages.sampling]\nthreds = 2")
    with pytest.raises(ConfigError, match="threds"):
        load_config(path)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/config -v`
Expected: the new tests FAIL (`EffectiveResources` cannot be imported).

- [ ] **Step 4: Implement** in `src/gqteashmd/config/schema.py`

Change the pydantic import to `from pydantic import BaseModel, ConfigDict, Field, model_validator`, add `from dataclasses import dataclass`, and replace `ResourcesSection`:

```python
class StageResources(_Section):
    """Per-stage override; unset keys fall back to the global [resources] values."""

    max_parallel_tasks: int | None = Field(default=None, ge=1)
    threads_per_task: int | None = Field(default=None, ge=1)
    memory_mb_per_task: int | None = Field(default=None, ge=100)


@dataclass(frozen=True)
class EffectiveResources:
    max_parallel_tasks: int
    threads_per_task: int
    memory_mb_per_task: int


class ResourcesSection(_Section):
    total_cores: int = Field(ge=1)
    total_memory_mb: int = Field(ge=100)
    max_parallel_tasks: int = Field(default=1, ge=1)
    threads_per_task: int = Field(default=1, ge=1)
    memory_mb_per_task: int = Field(default=2000, ge=100)
    stages: dict[str, StageResources] = Field(default_factory=dict)

    def for_stage(self, name: str) -> EffectiveResources:
        o = self.stages.get(name, StageResources())
        return EffectiveResources(
            max_parallel_tasks=(
                o.max_parallel_tasks if o.max_parallel_tasks is not None else self.max_parallel_tasks
            ),
            threads_per_task=(
                o.threads_per_task if o.threads_per_task is not None else self.threads_per_task
            ),
            memory_mb_per_task=(
                o.memory_mb_per_task
                if o.memory_mb_per_task is not None
                else self.memory_mb_per_task
            ),
        )

    @model_validator(mode="after")
    def _fits_the_workstation(self) -> ResourcesSection:
        checks = [("[resources]", self.for_stage(""))]
        checks += [(f"[resources.stages.{name}]", self.for_stage(name)) for name in self.stages]
        for label, eff in checks:
            cores = eff.max_parallel_tasks * eff.threads_per_task
            if cores > self.total_cores:
                raise ValueError(
                    f"{label}: max_parallel_tasks × threads_per_task = {cores} exceeds "
                    f"total_cores = {self.total_cores}"
                )
            memory = eff.max_parallel_tasks * eff.memory_mb_per_task
            if memory > self.total_memory_mb:
                raise ValueError(
                    f"{label}: max_parallel_tasks × memory_mb_per_task = {memory} MB exceeds "
                    f"total_memory_mb = {self.total_memory_mb}"
                )
        return self
```

In `ShmdConfig`, replace the `resources` line with `resources: ResourcesSection`.

In `src/gqteashmd/cli.py` `TEMPLATE`, replace the `[resources]` block:

```toml
[resources]
total_cores = 8               # REQUIRED: cores of this workstation you allow gqteashmd to use
total_memory_mb = 16000       # REQUIRED: RAM (MB) you allow gqteashmd to use
max_parallel_tasks = 1        # tasks running at the same time
threads_per_task = 1          # OpenMP/BLAS threads per task
memory_mb_per_task = 2000     # passed to PySCF as max_memory (MB)
# Per-stage overrides (unset keys use the values above), e.g.:
# [resources.stages.prepare]
# threads_per_task = 2
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/config tests/test_cli.py -v`, then `pytest -m "not slow" -W error`
Expected: all PASS. Every existing test uses `write_input`, which now includes the hardware keys. `test_init_writes_a_valid_template` loads the new template.

- [ ] **Step 6: Lint and commit**

```bash
ruff format src tests && ruff check --fix src tests
git add src/gqteashmd/config/schema.py src/gqteashmd/cli.py tests/conftest.py tests/config/test_resources.py
git commit -m "feat: require workstation hardware and validate task resources against it"
```

---

### Task 5: One executor per stage, from the effective resources

**Files:**
- Modify: `src/gqteashmd/orchestrator.py`, `src/gqteashmd/testing/fake_stages.py`
- Test: `tests/test_orchestrator.py`, `tests/test_worker.py`

**Interfaces:**
- Consumes: `ResourcesSection.for_stage(name) -> EffectiveResources` (Task 4); `LocalExecutor(*, max_parallel, threads_per_task, memory_mb_per_task, poll_interval, stop_grace_period)`.
- Produces:
  - `Orchestrator._executor_for(stage: Stage) -> LocalExecutor`, which replaces the `self.executor` attribute.
  - `Orchestrator.__init__` raises `ConfigError` when `[resources.stages]` names a stage that is not in the pipeline.
  - The `FakeStage.run_task` result gains `"threads": os.environ.get("OMP_NUM_THREADS")`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_orchestrator.py`)

```python
def test_stage_resource_override_reaches_workers(tmp_path, write_input, behaviour):
    path = write_input(
        tmp_path, execution=FAST, resources="[resources.stages.sampling]\nthreads_per_task = 2"
    )
    _run(path)
    run = tmp_path / "run"
    assert read_json(run / "03_sampling" / "tasks" / "t000" / "result.json")["result"]["threads"] == "2"
    assert read_json(run / "04_dynamics" / "tasks" / "t000" / "result.json")["result"]["threads"] == "1"


def test_unknown_stage_in_resource_overrides_is_config_error(tmp_path, write_input, behaviour):
    path = write_input(tmp_path, execution=FAST, resources="[resources.stages.nosuch]\nthreads_per_task = 1")
    with pytest.raises(ConfigError, match="nosuch"):
        Orchestrator(load_config(path), pipeline_ref=FAKE)
```

In `tests/test_worker.py`, change the expected result in `test_worker_writes_ok_result` to:

```python
    assert read_result(task_dir) == {
        "status": "ok",
        "result": {"index": 1, "upstream": [], "threads": os.environ.get("OMP_NUM_THREADS")},
    }
```

(add `import os`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_orchestrator.py -k "override" tests/test_worker.py -v`
Expected: FAIL (KeyError `threads`, and no ConfigError is raised).

- [ ] **Step 3: Implement**

`src/gqteashmd/testing/fake_stages.py`, in `FakeStage.run_task`:
`return {"index": task.payload["index"], "upstream": upstream, "threads": os.environ.get("OMP_NUM_THREADS")}`.

`src/gqteashmd/orchestrator.py`:
- In `__init__`, replace the `resources, execution = …` / `self.executor = LocalExecutor(...)` block with:

```python
        unknown = sorted(set(self.config.resources.stages) - set(names))
        if unknown:
            raise ConfigError(
                f"[resources.stages] names unknown stage(s): {', '.join(unknown)}; "
                f"stages are: {', '.join(names)}"
            )
```

- Add the method:

```python
    def _executor_for(self, stage: Stage) -> LocalExecutor:
        res = self.config.resources.for_stage(stage.name)
        execution = self.config.execution
        return LocalExecutor(
            max_parallel=res.max_parallel_tasks,
            threads_per_task=res.threads_per_task,
            memory_mb_per_task=res.memory_mb_per_task,
            poll_interval=execution.poll_interval_s,
            stop_grace_period=execution.stop_grace_period_s,
        )
```

- In `_run_stage`, add `executor = self._executor_for(stage)` before the `while True:` loop, and replace `self.executor.run(` with `executor.run(`.
- Search the tests for `.executor` (`grep -rn "\.executor" tests src`) and update any remaining use.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_orchestrator.py tests/test_worker.py -v`
Expected: all PASS.

- [ ] **Step 5: Full suite, lint, commit**

```bash
pytest -m "not slow" -W error && ruff format src tests && ruff check --fix src tests
git add src/gqteashmd/orchestrator.py src/gqteashmd/testing/fake_stages.py tests/test_orchestrator.py tests/test_worker.py
git commit -m "feat: run each stage with its own effective resources"
```

---

### Task 6: Preflight checks before any work

**Files:**
- Create: `src/gqteashmd/preflight.py`
- Modify: `src/gqteashmd/config/schema.py` (`ExecutionSection.min_free_disk_mb`), `src/gqteashmd/stages/base.py` (`required_modules`), `src/gqteashmd/testing/fake_stages.py` (`requires` behaviour), `src/gqteashmd/orchestrator.py` (`preflight()`, call in `run()`), `src/gqteashmd/cli.py` (call before `ensure()`, template line)
- Test: `tests/test_preflight.py` (new)

**Interfaces:**
- Consumes: `ShmdConfig`, `Stage`, `ConfigError`.
- Produces:
  - `MachineInfo(cores: int | None, memory_mb: int | None)` and `detect_machine() -> MachineInfo`.
  - `run_preflight(config, stages, *, machine: MachineInfo | None = None) -> list[str]`. It raises `ConfigError` for blocking problems and returns warnings.
  - `Stage.required_modules: tuple[str, ...] = ()`.
  - `ExecutionSection.min_free_disk_mb: int = 1000` (≥ 0).
  - `Orchestrator.preflight() -> list[str]`, cached. `run()` calls it before creating the project directory.

- [ ] **Step 1: Write the failing tests** in `tests/test_preflight.py`

```python
import json
from pathlib import Path

import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import ConfigError
from gqteashmd.orchestrator import Orchestrator
from gqteashmd.preflight import MachineInfo, detect_machine, run_preflight
from gqteashmd.stages.prepare import PrepareStage
from gqteashmd.testing.fake_stages import BEHAVIOUR_ENV

FAKE = "gqteashmd.testing.fake_stages:make_pipeline"


def _with_directory(cfg, directory):
    return cfg.model_copy(update={"project": cfg.project.model_copy(update={"directory": directory})})


def test_detect_machine_reports_positive_values():
    info = detect_machine()
    assert info.cores is not None and info.cores >= 1
    assert info.memory_mb is not None and info.memory_mb >= 1


def test_no_warnings_when_hardware_fits(tmp_path, write_input):
    cfg = load_config(write_input(tmp_path)).config
    assert run_preflight(cfg, [PrepareStage()], machine=MachineInfo(cores=64, memory_mb=10**6)) == []


def test_warns_when_user_hardware_exceeds_machine(tmp_path, write_input):
    cfg = load_config(write_input(tmp_path)).config  # total_cores 4, total_memory_mb 4000
    warnings = run_preflight(cfg, [PrepareStage()], machine=MachineInfo(cores=2, memory_mb=1000))
    assert len(warnings) == 2
    assert "total_cores = 4 exceeds the 2 cores" in warnings[0]
    assert "total_memory_mb = 4000 exceeds the 1000 MB" in warnings[1]


@pytest.mark.parametrize("directory", ["/mnt/c/Users/x/run", "/home/x/OneDrive/run"])
def test_windows_or_onedrive_project_directory_is_refused(tmp_path, write_input, directory):
    cfg = _with_directory(load_config(write_input(tmp_path)).config, Path(directory))
    with pytest.raises(ConfigError, match="local Linux disk"):
        run_preflight(cfg, [PrepareStage()], machine=MachineInfo(None, None))


def test_too_little_free_disk_is_refused(tmp_path, write_input):
    cfg = load_config(write_input(tmp_path, execution="min_free_disk_mb = 1000000000000")).config
    with pytest.raises(ConfigError, match="min_free_disk_mb"):
        run_preflight(cfg, [PrepareStage()], machine=MachineInfo(None, None))


def test_missing_module_is_refused_before_the_project_is_created(tmp_path, write_input, monkeypatch):
    monkeypatch.setenv(BEHAVIOUR_ENV, json.dumps({"requires": {"sampling": ["no_such_module_xyz"]}}))
    orchestrator = Orchestrator(load_config(write_input(tmp_path)), pipeline_ref=FAKE)
    with pytest.raises(ConfigError, match="no_such_module_xyz"):
        orchestrator.run()
    assert not (tmp_path / "run").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_preflight.py -v`
Expected: FAIL (`ModuleNotFoundError: gqteashmd.preflight`).

- [ ] **Step 3: Implement**

`src/gqteashmd/config/schema.py`, in `ExecutionSection`, add:
`min_free_disk_mb: int = Field(default=1000, ge=0)`.

`src/gqteashmd/stages/base.py`, in `class Stage`, add the attribute
`required_modules: tuple[str, ...] = ()` after `max_failed_fraction`, and document it in the module docstring as "Python modules the stage's tasks import; checked by preflight".

`src/gqteashmd/testing/fake_stages.py`:
- Add the parameter `required_modules: Sequence[str] = ()` to `FakeStage.__init__`, setting `self.required_modules = tuple(required_modules)`.
- In `make_pipeline`, pass `required_modules=requires.get("<name>", ())` for each fake stage, with `requires = behaviour.get("requires", {})`.
- Document the key in the module docstring: `requires: {"stage": ["module", ...]}`.

`src/gqteashmd/preflight.py`:

```python
"""Checks that run before any stage starts and before the project directory is touched."""

from __future__ import annotations

import importlib.util
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from gqteashmd.config.schema import ShmdConfig
from gqteashmd.errors import ConfigError
from gqteashmd.stages.base import Stage


@dataclass(frozen=True)
class MachineInfo:
    cores: int | None
    memory_mb: int | None


def detect_machine() -> MachineInfo:
    try:
        cores: int | None = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = os.cpu_count()
    memory_mb: int | None = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                memory_mb = int(line.split()[1]) // 1024
                break
    except (OSError, ValueError, IndexError):
        memory_mb = None
    return MachineInfo(cores=cores, memory_mb=memory_mb)


def _nearest_existing(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate
    return Path("/")


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def run_preflight(
    config: ShmdConfig, stages: Sequence[Stage], *, machine: MachineInfo | None = None
) -> list[str]:
    """Raise ConfigError for problems that must stop the run; return warnings."""
    machine = machine or detect_machine()
    warnings: list[str] = []
    project_dir = config.project.directory

    text = str(project_dir)
    if text.startswith("/mnt/") or "onedrive" in text.lower():
        raise ConfigError(
            f"Project directory {project_dir} is on a Windows or OneDrive folder, where file "
            "locks and many small files are unreliable. Set project.directory to a local Linux "
            f"disk, e.g. ~/gqteashmd-runs/{config.project.name}."
        )

    free_mb = shutil.disk_usage(_nearest_existing(project_dir)).free // (1024 * 1024)
    minimum = config.execution.min_free_disk_mb
    if free_mb < minimum:
        raise ConfigError(
            f"Only {free_mb} MB are free on the disk holding {project_dir}; at least {minimum} MB "
            "are required (execution.min_free_disk_mb)."
        )

    res = config.resources
    if machine.cores is not None and res.total_cores > machine.cores:
        warnings.append(
            f"resources.total_cores = {res.total_cores} exceeds the {machine.cores} cores "
            "detected on this machine; tasks may run slower than expected."
        )
    if machine.memory_mb is not None and res.total_memory_mb > machine.memory_mb:
        warnings.append(
            f"resources.total_memory_mb = {res.total_memory_mb} exceeds the {machine.memory_mb} MB "
            "of RAM detected on this machine; tasks may be killed for lack of memory."
        )

    for stage in stages:
        if not stage.enabled(config):
            continue
        for module in stage.required_modules:
            if not _module_available(module):
                raise ConfigError(
                    f"Stage '{stage.name}' needs the Python module '{module}', which is not "
                    "installed in this environment."
                )
    return warnings
```

`src/gqteashmd/orchestrator.py`:
- Add `from gqteashmd.preflight import run_preflight`.
- Initialise `self._preflight_warnings: list[str] | None = None` in `__init__`.
- Add the method:

```python
    def preflight(self) -> list[str]:
        """Run the preflight checks once; raises ConfigError, returns warnings."""
        if self._preflight_warnings is None:
            self._preflight_warnings = run_preflight(self.config, self.stages)
        return self._preflight_warnings
```

- At the start of `run()`, before `self.layout.ensure()`:

```python
        already_checked = self._preflight_warnings is not None
        warnings = self.preflight()
        if not already_checked:
            for message in warnings:
                log.warning("Preflight: %s", message)
```

`src/gqteashmd/cli.py` `_cmd_run`:
- Call `warnings = orchestrator.preflight()` before `orchestrator.layout.ensure()`.
- After `configure_logging(...)`, add `for message in warnings: logger.warning("Preflight: %s", message)`. (Move the `logger = …` line above that loop.)
- In `TEMPLATE`, add to `[execution]`:
  `min_free_disk_mb = 1000      # refuse to start with less free disk space (MB)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_preflight.py tests/test_cli.py tests/test_orchestrator.py -v`
Expected: all PASS.

- [ ] **Step 5: Full suite, lint, commit**

```bash
pytest -m "not slow" -W error && ruff format src tests && ruff check --fix src tests
git add src/gqteashmd/preflight.py src/gqteashmd/config/schema.py src/gqteashmd/stages/base.py src/gqteashmd/testing/fake_stages.py src/gqteashmd/orchestrator.py src/gqteashmd/cli.py tests/test_preflight.py
git commit -m "feat: add preflight checks for location, disk, hardware and modules"
```

---

### Task 7: Run provenance

**Files:**
- Create: `src/gqteashmd/project/provenance.py`
- Modify: `src/gqteashmd/project/layout.py` (`provenance_file`), `src/gqteashmd/orchestrator.py` (append on each run)
- Test: `tests/project/test_provenance.py` (new)

**Interfaces:**
- Consumes: `write_json_atomic`, `read_json`, `gqteashmd.__version__`.
- Produces:
  - `collect_provenance() -> dict[str, Any]`, with keys `started_at, gqteashmd, git_commit, git_dirty, python, platform, hostname, cpu_count, packages`.
  - `append_provenance(path: Path, entry: dict[str, Any]) -> None`, which writes `{"schema_version": 1, "runs": [...]}`.
  - `ProjectLayout.provenance_file`, at `.gqteashmd/provenance.json`.

- [ ] **Step 1: Write the failing tests** in `tests/project/test_provenance.py`

```python
import json

import gqteashmd
from gqteashmd.config.loader import load_config
from gqteashmd.orchestrator import Orchestrator
from gqteashmd.project.provenance import append_provenance, collect_provenance
from gqteashmd.testing.fake_stages import BEHAVIOUR_ENV
from gqteashmd.util.fileio import read_json

FAKE = "gqteashmd.testing.fake_stages:make_pipeline"
FAST = "poll_interval_s = 0.05\nstop_grace_period_s = 2.0"


def test_collect_provenance_has_the_documented_keys():
    entry = collect_provenance()
    assert set(entry) == {
        "started_at", "gqteashmd", "git_commit", "git_dirty", "python",
        "platform", "hostname", "cpu_count", "packages",
    }
    assert entry["gqteashmd"] == gqteashmd.__version__
    assert entry["packages"]["pydantic"]


def test_append_keeps_history_and_survives_corruption(tmp_path):
    path = tmp_path / "provenance.json"
    append_provenance(path, {"n": 1})
    append_provenance(path, {"n": 2})
    assert [r["n"] for r in read_json(path)["runs"]] == [1, 2]
    path.write_text("{broken")
    append_provenance(path, {"n": 3})
    assert read_json(path) == {"schema_version": 1, "runs": [{"n": 3}]}


def test_each_run_appends_one_record(tmp_path, write_input, monkeypatch):
    monkeypatch.setenv(BEHAVIOUR_ENV, json.dumps({}))
    path = write_input(tmp_path, execution=FAST)
    Orchestrator(load_config(path), pipeline_ref=FAKE).run()
    Orchestrator(load_config(path), pipeline_ref=FAKE).run()
    runs = read_json(tmp_path / "run" / ".gqteashmd" / "provenance.json")["runs"]
    assert len(runs) == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/project/test_provenance.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

`src/gqteashmd/project/layout.py`, add the property:

```python
    @property
    def provenance_file(self) -> Path:
        return self.meta_dir / "provenance.json"
```

`src/gqteashmd/project/provenance.py`:

```python
"""Record which software produced a project's results (one entry per `gqteashmd run`)."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from gqteashmd import __version__
from gqteashmd.util.fileio import read_json, write_json_atomic

PACKAGES = ("pydantic", "numpy", "scipy", "h5py", "pyscf", "geometric")


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git(*args: str) -> str | None:
    repo = Path(__file__).resolve().parents[3]
    if not (repo / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def collect_provenance() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain") if commit else None
    return {
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "gqteashmd": __version__,
        "git_commit": commit or None,
        "git_dirty": (bool(status) if status is not None else None),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "packages": {name: _package_version(name) for name in PACKAGES},
    }


def append_provenance(path: Path, entry: dict[str, Any]) -> None:
    runs: list[Any] = []
    if path.exists():
        try:
            data = read_json(path)
            runs = list(data["runs"]) if isinstance(data, dict) else []
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            runs = []
    runs.append(entry)
    write_json_atomic(path, {"schema_version": 1, "runs": runs})
```

`src/gqteashmd/orchestrator.py`:
- Add `from gqteashmd.project.provenance import append_provenance, collect_provenance`.
- In `run()`, inside the `with ProjectLock(...), StateStore(...) as state:` block, after `write_snapshot(...)`:
  `append_provenance(self.layout.provenance_file, collect_provenance())`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/project/test_provenance.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Full suite, lint, commit**

```bash
pytest -m "not slow" -W error && ruff format src tests && ruff check --fix src tests
git add src/gqteashmd/project/provenance.py src/gqteashmd/project/layout.py src/gqteashmd/orchestrator.py tests/project/test_provenance.py
git commit -m "feat: record run provenance in .gqteashmd/provenance.json"
```

---

### Task 8: Review pauses (`[workflow] pause_after`)

**Files:**
- Modify: `src/gqteashmd/config/schema.py` (`WorkflowSection`), `src/gqteashmd/orchestrator.py` (`RunSummary.paused_after`, pause logic, name validation), `src/gqteashmd/cli.py` (message, exit 5, template)
- Test: `tests/test_orchestrator.py`, `tests/test_cli.py`

**Interfaces:**
- Produces:
  - `WorkflowSection(pause_after: list[str] = [])`, available as `ShmdConfig.workflow`.
  - `RunSummary.paused_after: str | None`.
  - The orchestrator pauses after a stage named in `pause_after` that **completed in this invocation** and is not the last stage.
  - The CLI prints `Paused after stage '<name>' as requested (workflow.pause_after). Review its results, then run the same command again to continue.` and returns `EXIT_STOPPED` (5).
  - Unknown names raise `ConfigError`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_orchestrator.py`:

```python
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
```

Append to `tests/test_cli.py`:

```python
def test_pause_returns_exit_5_with_message(tmp_path, write_input, monkeypatch, capsys):
    monkeypatch.setenv(BEHAVIOUR_ENV, "{}")
    path = write_input(tmp_path, execution=FAST, extra='[workflow]\npause_after = ["sampling"]')
    assert main(["run", str(path), "--pipeline", FAKE]) == EXIT_STOPPED
    assert "Paused after stage 'sampling'" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_orchestrator.py -k pause tests/test_cli.py -k pause -v`
Expected: FAIL (`workflow` is an unknown section).

- [ ] **Step 3: Implement**

`src/gqteashmd/config/schema.py`:

```python
class WorkflowSection(_Section):
    pause_after: list[str] = Field(default_factory=list)
```

Add `workflow: WorkflowSection = Field(default_factory=WorkflowSection)` to `ShmdConfig`.

`src/gqteashmd/orchestrator.py`:
- In `RunSummary`, add `paused_after: str | None = None`.
- In `__init__`, after the resources check, add:

```python
        unknown_pauses = sorted(set(self.config.workflow.pause_after) - set(names))
        if unknown_pauses:
            raise ConfigError(
                f"workflow.pause_after names unknown stage(s): {', '.join(unknown_pauses)}; "
                f"stages are: {', '.join(names)}"
            )
```

- In `_run_pipeline`, after `summary.completed.append(stage.name)`:

```python
            if stage.name in self.config.workflow.pause_after and index < len(self.stages) - 1:
                summary.paused_after = stage.name
                log.info(
                    "Paused after stage %s (workflow.pause_after); run again to continue",
                    stage.name,
                )
                break
```

`src/gqteashmd/cli.py`:
- In `_cmd_run`, after the `if summary.stopped:` block:

```python
    if summary.paused_after is not None:
        print(
            f"Paused after stage '{summary.paused_after}' as requested (workflow.pause_after). "
            "Review its results, then run the same command again to continue."
        )
        return EXIT_STOPPED
```

- Append to `TEMPLATE`:

```toml

[workflow]
pause_after = []              # stage names to pause after for review, e.g. ["selection"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_orchestrator.py tests/test_cli.py -v`
Expected: all PASS.

- [ ] **Step 5: Full suite, lint, commit**

```bash
pytest -m "not slow" -W error && ruff format src tests && ruff check --fix src tests
git add src/gqteashmd/config/schema.py src/gqteashmd/orchestrator.py src/gqteashmd/cli.py tests/test_orchestrator.py tests/test_cli.py
git commit -m "feat: pause after configured stages for review"
```

---

### Task 9: Robustness fixes from the P1 deferred list

**Files:**
- Modify: `src/gqteashmd/project/lock.py`, `src/gqteashmd/orchestrator.py`, `src/gqteashmd/testing/fake_stages.py`
- Test: `tests/project/test_layout_lock.py`, `tests/test_orchestrator.py`

**Interfaces:**
- Produces:
  - `ProjectLock` retries once after 0.2 s before refusing. This covers the window in which `locked_by` briefly probes the lock.
  - Its error says `pid unknown` when no PID is readable.
  - The StaleStageError hint shell-quotes the input path.
  - A non-JSON-serialisable `fingerprint_inputs` raises `ProjectError` naming the stage.
  - `--rerun-from` is applied once per `Orchestrator` instance.
  - New fake behaviour `bad_fingerprint: [stage, ...]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/project/test_layout_lock.py`:

```python
import threading
import time


def test_lock_waits_briefly_for_a_short_holder(tmp_path):
    lock_path = tmp_path / "controller.lock"
    held = threading.Event()

    def hold_briefly():
        with ProjectLock(lock_path):
            held.set()
            time.sleep(0.05)

    worker = threading.Thread(target=hold_briefly)
    worker.start()
    held.wait(5)
    with ProjectLock(lock_path):
        pass
    worker.join()
```

Append to `tests/test_orchestrator.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/project/test_layout_lock.py tests/test_orchestrator.py -v`
Expected: the new tests FAIL. The lock test may pass by chance under heavy load; that is acceptable, because the retry makes it deterministic once implemented.

- [ ] **Step 3: Implement**

`src/gqteashmd/project/lock.py`:
- Add `import time`.
- In `ProjectLock.__enter__`, replace the single `flock` attempt with two attempts:

```python
        fh = open(self.path, "a+", encoding="utf-8")  # noqa: SIM115 - held for the lock lifetime
        for attempt in range(2):
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if attempt == 0:
                    time.sleep(0.2)  # `locked_by` may be probing the lock for an instant
                    continue
                fh.close()
                holder = locked_by(self.path)
                raise ProjectError(
                    "Another gqteashmd controller is running for this project "
                    f"(pid {holder if holder is not None else 'unknown'}, lock file {self.path}). "
                    "Use `gqteashmd status` or `gqteashmd stop`."
                ) from exc
```

(Keep the remainder of `__enter__`, which writes the PID, unchanged. If the existing `open(...)` line has no `noqa`, keep it as it is.)

`src/gqteashmd/testing/fake_stages.py`, add to `FakeStage`:

```python
    def fingerprint_inputs(self, ctx: StageContext) -> dict[str, Any]:
        if self.name in _behaviour().get("bad_fingerprint", []):
            return {"not_json": {1, 2}}
        return super().fingerprint_inputs(ctx)
```

Document the key in the module docstring.

`src/gqteashmd/orchestrator.py`:
- Add `import shlex`.
- In the StaleStageError message, replace `{self.loaded.source}` with `{shlex.quote(str(self.loaded.source))}`.
- Wrap the fingerprint computation:

```python
            try:
                fp = _fingerprint(
                    {
                        "stage": stage.name,
                        "version": stage.version,
                        "inputs": stage.fingerprint_inputs(ctx),
                        "upstream": {d: fingerprints[d] for d in stage.depends_on},
                    }
                )
            except TypeError as exc:
                raise ProjectError(
                    f"Stage {stage.name!r}: fingerprint inputs are not JSON-serializable: {exc}"
                ) from exc
```

- In `run()`, after `self._reset_from(state, self.rerun_from)`, add `self.rerun_from = None`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/project/test_layout_lock.py tests/test_orchestrator.py -v`
Expected: all PASS.

- [ ] **Step 5: Full suite, lint, commit**

```bash
pytest -W error && ruff format src tests && ruff check --fix src tests
git add src/gqteashmd/project/lock.py src/gqteashmd/orchestrator.py src/gqteashmd/testing/fake_stages.py tests/project/test_layout_lock.py tests/test_orchestrator.py
git commit -m "fix: lock retry, quoted rerun hint, fingerprint errors, one-shot rerun-from"
```

---

### Task 10: Static type checking with mypy

**Files:**
- Modify: `environment.yml`, `pyproject.toml`, `.github/workflows/ci.yml`, and any `src/gqteashmd/**/*.py` files mypy reports
- Test: `mypy` itself

**Interfaces:**
- Produces: `mypy` is clean on `src/`. CI runs it after lint.

- [ ] **Step 1: Add mypy**
  - In `environment.yml`, under `ruff`, add `- mypy=1.*`.
  - In `pyproject.toml` `[project.optional-dependencies] dev`, add `"mypy>=1.10"`.
  - Install it into the existing environment with `conda install -y -n gqteashmd-dev -c conda-forge "mypy=1.*"`.
  - Append to `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.12"
files = ["src"]
plugins = ["pydantic.mypy"]
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_return_any = true
strict_equality = true
```

- [ ] **Step 2: Run mypy to see the current errors**

Run: `mypy`
Expected: a list of errors, or `Success: no issues found`. Record the count in the report.

- [ ] **Step 3: Fix the reported errors.**
  - Prefer precise annotations, such as `dict[str, Any]` for JSON data, and narrowing with `isinstance` or `assert x is not None` where the invariant is real.
  - Use `# type: ignore[code]` only with a one-line reason, and only for third-party typing gaps.
  - Do not change behaviour.
  - Re-run `mypy` until it reports `Success`.

- [ ] **Step 4: Add mypy to CI.** In `.github/workflows/ci.yml`, after the `Lint` step:

```yaml
      - name: Type check
        run: mypy
```

- [ ] **Step 5: Full suite, lint, commit**

```bash
mypy && pytest -m "not slow" -W error && ruff format src tests && ruff check --fix src tests
git add environment.yml pyproject.toml .github/workflows/ci.yml src
git commit -m "build: type-check the package with mypy"
```

---

### Task 11: Documentation current with P2

**Files:**
- Modify: `CLAUDE.md`, `USER_MANUAL.md`, `docs/superpowers/plans/2026-11-06-gqteashmd-master-plan.md`

**Interfaces:**
- Consumes: the behaviour from Tasks 1–10. Read the current code (`cli.py` `TEMPLATE`, `schema.py`, `preflight.py`, `orchestrator.py`) rather than this plan, because fix rounds may have changed details.

- [ ] **Step 1: Update `USER_MANUAL.md`** to match the implementation, and mark each item [IMPLEMENTED]:
  - **§4.2:** the full key table, adding `resources.total_cores` and `resources.total_memory_mb` (required), the `[resources.stages.<name>]` overrides with their merge rule, `execution.min_free_disk_mb`, and `[workflow] pause_after`. Paste the current `gqteashmd init` template.
  - **§4.4:** remove the items that are now implemented.
  - **§5.1:** exit code 5 also means "paused for review".
  - **§6:** add `.gqteashmd/provenance.json` with its fields, and say how to read it.
  - **§7:** add "7.8 Review pauses" and "7.9 Checks before a run (preflight)". The preflight section lists every blocking check and warning with its message.
  - **§8.4:** the resource inequalities, now enforced in the input file:

    $n_p n_t \le N_{\text{cores}}$ and $n_p m \le M_{\text{total}}$, where $N_{\text{cores}}$ = `total_cores` and $M_{\text{total}}$ = `total_memory_mb`, both for the global settings and for each stage override. The hardware detection is a warning only. Define every symbol.
  - **§9:** add the new messages (hardware exceeded, unknown stage in overrides or pauses, preflight refusals, corrupt or unreadable state).
  - **§10:** remove the two fixed limitations; keep the rest.
- [ ] **Step 2: Update `CLAUDE.md`**:
  - **Status:** P2 done on branch `feat/p2-input-preflight`, with its test count.
  - **Architecture:** per-stage executors, preflight, provenance, pause, the `FILE_ONLY` logging rule, read-only status.
  - **Known limitations:** remove the fixed items; point to the P1 ledger for the remaining deferred items listed in this plan's "Scope rulings".
  - **Remaining work:** P3 next.
- [ ] **Step 3: Update the master plan**:
  - In §9, change the P2 row to describe what was delivered, and add the scope ruling: later input sections are added by the sub-projects that consume them.
  - In §11, mark O12 (unattended running) as resolved as "documented `nohup`/`tmux`; no detach" if nothing else was decided.
- [ ] **Step 4: Verify the documents against reality.** In a WSL scratch directory (`mktemp -d /tmp/gq-doc-XXXX`), run the manual's quick start verbatim and confirm the output matches. Then remove the directory.
- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md USER_MANUAL.md docs/superpowers/plans/2026-11-06-gqteashmd-master-plan.md
git commit -m "docs: update CLAUDE.md, USER_MANUAL.md and master plan for P2"
```

---

## Self-review notes

- **Spec coverage:**
  - D17, the hardware keys and the warning: Tasks 4 and 6.
  - §3.5 per-stage resources: Tasks 4 and 5.
  - §4.2 environment preflight: Task 6.
  - §3.6 provenance: Task 7.
  - §1.3 / §4.3 `pause_after`: Task 8.
  - §6 error handling (residuals I1 and I2): Tasks 1 and 2.
  - The P1 deferred P2 items: Tasks 3 and 9.
  - mypy (master plan §8): Task 10.
  - Documentation rule (CLAUDE.md): Task 11.
  - Deliberately excluded: the later input sections and the other deferred items. See "Scope rulings".
- **Type consistency:**
  - `EffectiveResources(max_parallel_tasks, threads_per_task, memory_mb_per_task)` is used identically in Tasks 4 and 5.
  - `MachineInfo(cores, memory_mb)` appears only in Task 6.
  - `RunSummary.paused_after` is defined in Task 8 and used by the CLI in Task 8.
  - `StateStore(db_path, *, readonly=False)` is defined in Task 2 and used by the CLI in Task 2.
  - `FILE_ONLY` and `close_logging` are defined in Task 1 and used only in `cli.main`.
