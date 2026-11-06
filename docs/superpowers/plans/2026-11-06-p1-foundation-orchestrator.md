# P1: Foundation & Workflow Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `gqteashmd` repository, development environment and the resumable workflow controller. At the end, `gqteashmd run input.toml` executes an ordered pipeline of stages as parallel subprocess tasks. It survives crashes and restarts, retries failures, detects changed inputs, and stops gracefully. The first real stage (`prepare`) is included.

**Architecture:**
- A single-writer controller (`Orchestrator`) keeps stage and task state in SQLite.
- It launches one worker subprocess per task (`python -m gqteashmd.worker task.json`) through a `LocalExecutor` with a concurrency limit.
- Workers report back through atomically written `result.json` files.
- Stages implement a three-method contract (`plan`, `run_task`, `finalize`). Fake stages in `gqteashmd.testing` exercise the orchestrator; the real pipeline currently contains only `prepare`.

**Tech Stack:** Python 3.12, pydantic v2, stdlib (`tomllib`, `sqlite3`, `subprocess`, `fcntl`, `argparse`, `logging`), pytest, ruff, micromamba/conda-forge, GitHub Actions (Oracle Linux 9 container).

**Spec:** [`docs/superpowers/plans/2026-11-06-gqteashmd-master-plan.md`](2026-11-06-gqteashmd-master-plan.md) (§2–§8 and decisions D1–D16) and [`SHMD_implementation_strategy.md`](../../../SHMD_implementation_strategy.md).

## Global Constraints

- **Clean room:** no code from SHARC 4.1 (`\\wsl.localhost\OracleLinux_9_5\home\ajc\sharc4-4.1`) is copied, translated or adapted.
- **Licence:** MIT.
- **Runtime:** Linux only (target Oracle Linux 9). Development and all test runs happen in WSL2 (Oracle Linux 9.5). Code may use `fcntl`, `/proc` and POSIX process groups.
- **Python:** 3.12. Runtime dependency in P1: `pydantic>=2.6` only. Everything else is stdlib.
- **Repository root:** `C:\Users\ajc\OneDrive\gqteaSHMD` (WSL path `/mnt/c/Users/ajc/OneDrive/gqteaSHMD`). Never create environments, project runs or large test outputs inside this folder.
- **Git:** only Git inside WSL. `core.autocrlf=false`, `core.filemode=false`, LF line endings.
- **Style:** ruff (line length 100, rules E, F, W, I, B, UP, SIM). Type hints on all public functions.
- **Files written by the program** are written atomically (temporary file + `os.replace`).
- **State database:** only the controller writes it.
- **Test commands** are run from the repository root inside the activated `gqteashmd-dev` environment in WSL.
- **Branch:** all P1 work is committed on `feat/p1-foundation` (one commit per task) and pushed at the end. The owner merges the PR (D20).

## Execution environment (for agents running on the Windows host)

Agents run on Windows, but every command in this plan (`git`, `conda`, `pytest`, `ruff`, `gqteashmd`) must run in WSL. Wrap each command like this:

```bash
wsl -d OracleLinux_9_5 -- bash -lc 'cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && conda activate gqteashmd-dev && <command>'
```

- Before Task 1 Step 7 creates the environment, leave out `conda activate gqteashmd-dev &&`.
- Create and edit files with the normal file tools, using Windows paths under `C:\Users\ajc\OneDrive\gqteaSHMD\`. Write them with LF line endings.
- Commit with **WSL Git** (pre-commit hooks installed in Task 12 only work there). The repository already has a repo-local Git identity (`Ademir Camargo <ajc@ueg.br>`), `core.autocrlf=false` and `core.filemode=false`.
- WSL is Oracle Linux 9.5 with Miniforge conda (no micromamba), 16 cores and about 7 GB RAM. Keep pytest temporary files on the Linux disk (the pytest default `/tmp` is fine).

## File Structure (created by this plan)

```text
.gitattributes  .gitignore  .pre-commit-config.yaml  LICENSE  README.md
environment.yml  pyproject.toml
.github/workflows/ci.yml
docs/development.md
src/gqteashmd/__init__.py            # version
src/gqteashmd/__main__.py            # `python -m gqteashmd` → CLI
src/gqteashmd/cli.py                 # init | validate | run | status | stop
src/gqteashmd/errors.py              # error taxonomy + exit codes
src/gqteashmd/logging_setup.py       # console + file logging
src/gqteashmd/orchestrator.py        # the controller
src/gqteashmd/worker.py              # task subprocess entry point; task/result file I/O
src/gqteashmd/util/__init__.py
src/gqteashmd/util/fileio.py         # atomic writes, JSON helpers
src/gqteashmd/config/__init__.py
src/gqteashmd/config/schema.py       # pydantic models (core sections)
src/gqteashmd/config/loader.py       # TOML loading, path resolution, snapshot, fingerprint sections
src/gqteashmd/project/__init__.py
src/gqteashmd/project/layout.py      # project directory layout
src/gqteashmd/project/lock.py        # single-controller lock
src/gqteashmd/project/state.py       # SQLite state store
src/gqteashmd/execution/__init__.py
src/gqteashmd/execution/local.py     # LocalExecutor
src/gqteashmd/execution/processes.py # orphan detection/termination
src/gqteashmd/stages/__init__.py
src/gqteashmd/stages/base.py         # Stage, TaskSpec, StageContext
src/gqteashmd/stages/pipeline.py     # pipeline import + validation
src/gqteashmd/stages/prepare.py      # PrepareStage
src/gqteashmd/stages/registry.py     # default_pipeline()
src/gqteashmd/chem/__init__.py
src/gqteashmd/chem/elements.py       # element symbols → Z
src/gqteashmd/chem/xyz.py            # XYZ parse/format
src/gqteashmd/testing/__init__.py
src/gqteashmd/testing/fake_stages.py # deterministic fake stages for tests
tests/conftest.py
tests/test_package.py
tests/test_errors_fileio.py
tests/config/test_loader.py
tests/project/test_layout_lock.py
tests/project/test_state.py
tests/stages/test_base_pipeline.py
tests/chem/test_xyz.py
tests/stages/test_prepare.py
tests/test_worker.py
tests/execution/test_local.py
tests/execution/test_processes.py
tests/test_orchestrator.py
tests/test_cli.py
```

There are no `__init__.py` files in `tests/` subdirectories. pytest's default `rootdir`-based discovery with `importmode=importlib` (set in `pyproject.toml`) handles same-named test files.

---

### Task 1: Repository scaffolding and development environment

**Files:**
- Create: `pyproject.toml`, `environment.yml`, `LICENSE`, `README.md`, `.gitignore`, `.gitattributes`, `src/gqteashmd/__init__.py`
- Test: `tests/test_package.py`

**Interfaces:**
- Consumes: nothing.
- Produces: installable package `gqteashmd` with `gqteashmd.__version__ == "0.1.0.dev0"`. Console script `gqteashmd = "gqteashmd.cli:main"` is declared; `cli.py` arrives in Task 11, and until then the script entry fails on use, which is expected. The `gqteashmd-dev` conda environment.

- [ ] **Step 1: Check the repository state (it already exists and is on GitHub)**

```bash
cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD
git branch --show-current
git config --local --get core.autocrlf
git config --local --get core.filemode
```

Expected: `feat/p1-foundation`, `false`, `false`. (The repository, branch, repo-local config and Git identity were set up before execution started.)

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling>=1.24"]
build-backend = "hatchling.build"

[project]
name = "gqteashmd"
version = "0.1.0.dev0"
description = "Automated surface-hopping molecular dynamics workflow (PySCF backend)"
readme = "README.md"
license = { file = "LICENSE" }
requires-python = ">=3.12"
dependencies = ["pydantic>=2.6"]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.6", "pre-commit>=3"]

[project.scripts]
gqteashmd = "gqteashmd.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/gqteashmd"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --import-mode=importlib -p no:cacheprovider"
markers = [
    "slow: long-running tests (deselect with -m 'not slow')",
]

[tool.ruff]
line-length = 100
target-version = "py312"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "SIM"]
```

(`-p no:cacheprovider` keeps pytest from writing `.pytest_cache` into the OneDrive folder.)

- [ ] **Step 3: Create `environment.yml`**

```yaml
name: gqteashmd-dev
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - pydantic>=2.6
  - pytest>=8
  - ruff>=0.6
  - pre-commit>=3
  # needed from P3 on; installed now so the environment does not change shape later
  - pyscf
  - numpy
  - scipy
  - h5py
  - geometric
```

- [ ] **Step 4: Create `LICENSE` (MIT)**

```text
MIT License

Copyright (c) 2026 The gqteashmd developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 5: Create `README.md`, `.gitignore`, `.gitattributes`**

`README.md`:

````markdown
# gqteashmd

Automated surface-hopping molecular dynamics workflow:
geometry optimization → frequencies → Wigner sampling → initial conditions → trajectories,
driven from a single input file. Electronic structure: PySCF.

Status: early development (P1: foundation & orchestrator). Linux only.

```bash
gqteashmd init input.toml      # write a commented template
gqteashmd validate input.toml
gqteashmd run input.toml       # run all stages; re-run to resume
gqteashmd status input.toml
gqteashmd stop input.toml
```

Developer setup: see [docs/development.md](docs/development.md).
Licence: MIT.
````

`.gitignore`:

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.venv/
# gqteashmd project runs must never live in the repository
.gqteashmd/
run/
```

`.gitattributes`:

```gitattributes
* text=auto eol=lf
*.png binary
*.h5 binary
```

- [ ] **Step 6: Write the failing test** `tests/test_package.py`

```python
import gqteashmd


def test_version_is_defined():
    assert gqteashmd.__version__ == "0.1.0.dev0"
```

- [ ] **Step 7: Create the environment and run the test to verify it fails**

```bash
conda env create -f environment.yml          # WSL has Miniforge conda (micromamba also works)
conda activate gqteashmd-dev
pytest tests/test_package.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'gqteashmd'`.

- [ ] **Step 8: Create `src/gqteashmd/__init__.py` and install editable**

```python
"""gqteashmd: automated surface-hopping molecular dynamics workflow."""

__version__ = "0.1.0.dev0"
```

```bash
pip install -e . --no-deps
```

- [ ] **Step 9: Run the test to verify it passes**

Run: `pytest tests/test_package.py -v`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml environment.yml LICENSE README.md .gitignore .gitattributes src/gqteashmd/__init__.py tests/test_package.py
git commit -m "chore: scaffold gqteashmd package, environment and licence"
```

---

### Task 2: Error taxonomy and atomic file I/O

**Files:**
- Create: `src/gqteashmd/errors.py`, `src/gqteashmd/util/__init__.py` (empty), `src/gqteashmd/util/fileio.py`
- Test: `tests/test_errors_fileio.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ShmdError(Exception)` with class attribute `exit_code: int = 1`;
  - `ConfigError` (exit 2), `ProjectError` (exit 3), `StaleStageError(ProjectError)`, `StageFailedError` (exit 4);
  - `TaskError(message: str, *, retryable: bool)` with attribute `retryable`;
  - `write_text_atomic(path: Path, text: str) -> None`, `write_json_atomic(path: Path, data: Any) -> None`, `read_json(path: Path) -> Any`.

- [ ] **Step 1: Write the failing tests** `tests/test_errors_fileio.py`

```python
import json

import pytest

from gqteashmd.errors import (
    ConfigError,
    ProjectError,
    ShmdError,
    StageFailedError,
    StaleStageError,
    TaskError,
)
from gqteashmd.util.fileio import read_json, write_json_atomic, write_text_atomic


def test_exit_codes_distinguish_error_classes():
    assert ShmdError("x").exit_code == 1
    assert ConfigError("x").exit_code == 2
    assert ProjectError("x").exit_code == 3
    assert StaleStageError("x").exit_code == 3
    assert StageFailedError("x").exit_code == 4


def test_stale_stage_is_a_project_error():
    assert issubclass(StaleStageError, ProjectError)


def test_task_error_carries_retryable_flag():
    err = TaskError("scf failed", retryable=True)
    assert err.retryable is True
    assert str(err) == "scf failed"
    assert isinstance(err, ShmdError)


def test_write_text_atomic_creates_parents_and_leaves_no_temp(tmp_path):
    target = tmp_path / "a" / "b" / "out.txt"
    write_text_atomic(target, "hello\n")
    assert target.read_text() == "hello\n"
    assert [p.name for p in target.parent.iterdir()] == ["out.txt"]


def test_json_roundtrip_is_sorted_and_newline_terminated(tmp_path):
    target = tmp_path / "data.json"
    write_json_atomic(target, {"b": 1, "a": [1, 2]})
    text = target.read_text()
    assert text.endswith("\n")
    assert text.index('"a"') < text.index('"b"')
    assert read_json(target) == {"a": [1, 2], "b": 1}


def test_read_json_propagates_decode_errors(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        read_json(bad)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_errors_fileio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gqteashmd.errors'`

- [ ] **Step 3: Implement `src/gqteashmd/errors.py`**

```python
"""Error taxonomy. Every deliberate error raised by gqteashmd derives from ShmdError.

Exit codes: 0 ok, 1 generic, 2 configuration, 3 project/state, 4 stage failed,
5 stopped by user (defined in gqteashmd.cli).
"""

from __future__ import annotations


class ShmdError(Exception):
    """Base class for all errors raised deliberately by gqteashmd."""

    exit_code: int = 1


class ConfigError(ShmdError):
    """The input file or command-line arguments are invalid. Nothing has been run."""

    exit_code = 2


class ProjectError(ShmdError):
    """The project directory or its state cannot be used (lock held, corrupt state, ...)."""

    exit_code = 3


class StaleStageError(ProjectError):
    """A completed stage has results that were produced from different inputs."""


class StageFailedError(ShmdError):
    """A stage could not complete (too many failed tasks or failed physical checks)."""

    exit_code = 4


class TaskError(ShmdError):
    """A single task failed. `retryable` tells the controller whether to try again."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable
```

- [ ] **Step 4: Implement `src/gqteashmd/util/fileio.py`** (and an empty `src/gqteashmd/util/__init__.py`)

```python
"""Atomic file writes and JSON helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_text_atomic(path: Path, text: str) -> None:
    """Write `text` to `path` so that readers never see a partially written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_json_atomic(path: Path, data: Any) -> None:
    write_text_atomic(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_errors_fileio.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/gqteashmd/errors.py src/gqteashmd/util tests/test_errors_fileio.py
git commit -m "feat: add error taxonomy and atomic file helpers"
```

---

### Task 3: Input file schema and loader

**Files:**
- Create: `src/gqteashmd/config/__init__.py` (empty), `src/gqteashmd/config/schema.py`, `src/gqteashmd/config/loader.py`, `tests/conftest.py`
- Test: `tests/config/test_loader.py`

**Interfaces:**
- Consumes: `ConfigError` (Task 2), `write_text_atomic` (Task 2).
- Produces:
  - `ShmdConfig` with sections `project` (`name: str`, `directory: Path`, `seed: int`), `molecule` (`geometry: Path`, `charge: int`, `multiplicity: int`), `resources` (`max_parallel_tasks`, `threads_per_task`, `memory_mb_per_task`), and `execution` (`max_attempts`, `poll_interval_s`, `stop_grace_period_s`).
  - `LoadedConfig(config: ShmdConfig, source: Path)`.
  - `load_config(path: Path) -> LoadedConfig`: paths become absolute, resolved against the input file's directory.
  - `write_snapshot(loaded: LoadedConfig, path: Path) -> None`.
  - `load_snapshot(path: Path) -> LoadedConfig`.
  - `section_fingerprint(config: ShmdConfig, sections: Iterable[str]) -> dict[str, Any]`.
  - Test fixtures `write_input` and `water_xyz` in `tests/conftest.py`.

- [ ] **Step 1: Create the shared fixtures** `tests/conftest.py`

```python
from pathlib import Path

import pytest

WATER_XYZ = """3
water
O    0.000000    0.000000    0.117300
H    0.000000    0.757200   -0.469200
H    0.000000   -0.757200   -0.469200
"""


@pytest.fixture
def water_xyz() -> str:
    return WATER_XYZ


@pytest.fixture
def write_input():
    """Write water.xyz and input.toml into `directory`; return the input path.

    Keyword arguments insert raw TOML lines into the given section;
    `extra` is appended at the end of the file (top level).
    """

    def _write(
        directory: Path,
        *,
        molecule: str = "",
        resources: str = "",
        execution: str = "",
        extra: str = "",
    ) -> Path:
        (directory / "water.xyz").write_text(WATER_XYZ)
        text = (
            '[project]\nname = "test"\ndirectory = "run"\n\n'
            f'[molecule]\ngeometry = "water.xyz"\n{molecule}\n\n'
            f"[resources]\n{resources}\n\n"
            f"[execution]\n{execution}\n\n"
            f"{extra}\n"
        )
        path = directory / "input.toml"
        path.write_text(text)
        return path

    return _write
```

- [ ] **Step 2: Write the failing tests** `tests/config/test_loader.py`

```python
import pytest

from gqteashmd.config.loader import load_config, load_snapshot, section_fingerprint, write_snapshot
from gqteashmd.errors import ConfigError


def test_minimal_input_gets_defaults_and_absolute_paths(tmp_path, write_input):
    loaded = load_config(write_input(tmp_path))
    cfg = loaded.config
    assert loaded.source == (tmp_path / "input.toml").resolve()
    assert cfg.project.directory == (tmp_path / "run").resolve()
    assert cfg.molecule.geometry == (tmp_path / "water.xyz").resolve()
    assert cfg.molecule.charge == 0
    assert cfg.molecule.multiplicity == 1
    assert cfg.project.seed == 12345
    assert cfg.resources.max_parallel_tasks == 1
    assert cfg.execution.max_attempts == 2


def test_unknown_section_is_rejected(tmp_path, write_input):
    path = write_input(tmp_path, extra="[molecule_typo]\ncharge = 0")
    with pytest.raises(ConfigError, match="molecule_typo"):
        load_config(path)


def test_unknown_key_is_rejected_with_location(tmp_path, write_input):
    path = write_input(tmp_path, molecule="chrage = 1")
    with pytest.raises(ConfigError, match=r"molecule\.chrage"):
        load_config(path)


def test_invalid_value_reports_field(tmp_path, write_input):
    path = write_input(tmp_path, resources="max_parallel_tasks = 0")
    with pytest.raises(ConfigError, match=r"resources\.max_parallel_tasks"):
        load_config(path)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_bad_toml(tmp_path):
    path = tmp_path / "input.toml"
    path.write_text("project = [")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(path)


def test_snapshot_roundtrip(tmp_path, write_input):
    loaded = load_config(write_input(tmp_path, molecule="charge = 2"))
    snap = tmp_path / "snap.json"
    write_snapshot(loaded, snap)
    again = load_snapshot(snap)
    assert again.config == loaded.config
    assert again.source == snap.resolve()


def test_snapshot_missing_is_config_error(tmp_path):
    with pytest.raises(ConfigError, match="snapshot"):
        load_snapshot(tmp_path / "missing.json")


def test_section_fingerprint_selects_sections(tmp_path, write_input):
    cfg = load_config(write_input(tmp_path)).config
    fp = section_fingerprint(cfg, ["molecule", "resources"])
    assert set(fp) == {"molecule", "resources"}
    assert fp["molecule"]["charge"] == 0
    assert isinstance(fp["molecule"]["geometry"], str)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/config/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gqteashmd.config'`

- [ ] **Step 4: Implement `src/gqteashmd/config/schema.py`** (and an empty `src/gqteashmd/config/__init__.py`)

```python
"""Input file schema (core sections; P2 adds electronic, sampling, dynamics, ...)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectSection(_Section):
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    directory: Path
    seed: int = Field(default=12345, ge=0)


class MoleculeSection(_Section):
    geometry: Path
    charge: int = 0
    multiplicity: int = Field(default=1, ge=1)


class ResourcesSection(_Section):
    max_parallel_tasks: int = Field(default=1, ge=1)
    threads_per_task: int = Field(default=1, ge=1)
    memory_mb_per_task: int = Field(default=2000, ge=100)


class ExecutionSection(_Section):
    max_attempts: int = Field(default=2, ge=1)
    poll_interval_s: float = Field(default=0.5, gt=0)
    stop_grace_period_s: float = Field(default=30.0, ge=0)


class ShmdConfig(_Section):
    schema_version: Literal[1] = 1
    project: ProjectSection
    molecule: MoleculeSection
    resources: ResourcesSection = Field(default_factory=ResourcesSection)
    execution: ExecutionSection = Field(default_factory=ExecutionSection)
```

- [ ] **Step 5: Implement `src/gqteashmd/config/loader.py`**

```python
"""Load and validate the TOML input file; snapshot the resolved configuration."""

from __future__ import annotations

import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from gqteashmd.config.schema import ShmdConfig
from gqteashmd.errors import ConfigError
from gqteashmd.util.fileio import write_text_atomic


@dataclass(frozen=True)
class LoadedConfig:
    config: ShmdConfig
    source: Path  # absolute path of the input file (or of the snapshot it was read from)


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    lines = [f"Invalid input file {path}:"]
    for err in exc.errors():
        location = ".".join(str(part) for part in err["loc"]) or "<top level>"
        lines.append(f"  - {location}: {err['msg']}")
    return "\n".join(lines)


def _resolve_paths(cfg: ShmdConfig, base: Path) -> ShmdConfig:
    def absolute(p: Path) -> Path:
        return p if p.is_absolute() else (base / p).resolve()

    return cfg.model_copy(
        update={
            "project": cfg.project.model_copy(
                update={"directory": absolute(cfg.project.directory)}
            ),
            "molecule": cfg.molecule.model_copy(
                update={"geometry": absolute(cfg.molecule.geometry)}
            ),
        }
    )


def load_config(path: Path) -> LoadedConfig:
    path = Path(path).resolve()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Input file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Input file {path} is not valid TOML: {exc}") from exc
    try:
        cfg = ShmdConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(path, exc)) from exc
    return LoadedConfig(config=_resolve_paths(cfg, path.parent), source=path)


def write_snapshot(loaded: LoadedConfig, path: Path) -> None:
    write_text_atomic(path, loaded.config.model_dump_json(indent=2) + "\n")


def load_snapshot(path: Path) -> LoadedConfig:
    path = Path(path)
    try:
        cfg = ShmdConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise ConfigError(f"Cannot read config snapshot {path}: {exc}") from exc
    return LoadedConfig(config=cfg, source=path.resolve())


def section_fingerprint(config: ShmdConfig, sections: Iterable[str]) -> dict[str, Any]:
    """JSON-compatible dump of the named top-level sections (used for stage fingerprints)."""
    dumped = config.model_dump(mode="json")
    return {name: dumped[name] for name in sections}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/config/test_loader.py -v`
Expected: 9 passed

- [ ] **Step 7: Commit**

```bash
git add src/gqteashmd/config tests/conftest.py tests/config
git commit -m "feat: add strict TOML input schema, loader and config snapshot"
```

---

### Task 4: Project directory layout and single-controller lock

**Files:**
- Create: `src/gqteashmd/project/__init__.py` (empty), `src/gqteashmd/project/layout.py`, `src/gqteashmd/project/lock.py`
- Test: `tests/project/test_layout_lock.py`

**Interfaces:**
- Consumes: `ProjectError` (Task 2).
- Produces:
  - `ProjectLayout(root: Path)` with properties `meta_dir`, `state_db`, `lock_file`, `config_snapshot`, `superseded_dir`, `logs_dir`, and methods `stage_dir(index: int, name: str) -> Path`, `task_dir(stage_dir: Path, task_id: str) -> Path`, `ensure() -> None`, `supersede(stage_dir: Path) -> Path | None`.
  - `ProjectLock(path: Path)`, a context manager that raises `ProjectError("Another gqteashmd controller ...")` if the lock is held.
  - `locked_by(path: Path) -> int | None`.

- [ ] **Step 1: Write the failing tests** `tests/project/test_layout_lock.py`

```python
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
        with pytest.raises(ProjectError, match="Another gqteashmd controller"), ProjectLock(lock_path):
            pass
    assert locked_by(lock_path) is None
    with ProjectLock(lock_path):
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/project/test_layout_lock.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gqteashmd.project'`

- [ ] **Step 3: Implement `src/gqteashmd/project/layout.py`** (and an empty `src/gqteashmd/project/__init__.py`)

```python
"""Directory layout of a gqteashmd project."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ProjectLayout:
    root: Path

    @property
    def meta_dir(self) -> Path:
        return self.root / ".gqteashmd"

    @property
    def state_db(self) -> Path:
        return self.meta_dir / "state.db"

    @property
    def lock_file(self) -> Path:
        return self.meta_dir / "controller.lock"

    @property
    def config_snapshot(self) -> Path:
        return self.meta_dir / "config.json"

    @property
    def superseded_dir(self) -> Path:
        return self.meta_dir / "superseded"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def stage_dir(self, index: int, name: str) -> Path:
        return self.root / f"{index:02d}_{name}"

    def task_dir(self, stage_dir: Path, task_id: str) -> Path:
        return stage_dir / "tasks" / task_id

    def ensure(self) -> None:
        for directory in (self.meta_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def supersede(self, stage_dir: Path) -> Path | None:
        """Move an old stage directory into .gqteashmd/superseded/ (never delete results)."""
        if not stage_dir.exists():
            return None
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        target = self.superseded_dir / f"{stamp}_{stage_dir.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stage_dir), str(target))
        return target
```

- [ ] **Step 4: Implement `src/gqteashmd/project/lock.py`**

```python
"""Ensure only one controller runs per project (Linux flock)."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import IO

from gqteashmd.errors import ProjectError


class ProjectLock:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fh: IO[str] | None = None

    def __enter__(self) -> ProjectLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+", encoding="utf-8")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            fh.close()
            holder = locked_by(self.path)
            raise ProjectError(
                f"Another gqteashmd controller is running for this project "
                f"(pid {holder}, lock file {self.path}). "
                "Use `gqteashmd status` or `gqteashmd stop`."
            ) from exc
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()}\n")
        fh.flush()
        self._fh = fh
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._fh is not None:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None


def locked_by(path: Path) -> int | None:
    """PID of the controller holding the lock, or None if nobody holds it."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "a+", encoding="utf-8") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.seek(0)
            text = fh.read().strip()
            return int(text) if text.isdigit() else None
        fcntl.flock(fh, fcntl.LOCK_UN)
        return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/project/test_layout_lock.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/gqteashmd/project tests/project/test_layout_lock.py
git commit -m "feat: add project layout and single-controller lock"
```

---

### Task 5: Persistent state store (SQLite)

**Files:**
- Create: `src/gqteashmd/project/state.py`
- Test: `tests/project/test_state.py`

**Interfaces:**
- Consumes: `ProjectError` (Task 2).
- Produces:
  - Enums `StageStatus` (`PENDING`, `RUNNING`, `DONE`, `FAILED`, `SKIPPED`) and `TaskStatus` (`PENDING`, `RUNNING`, `DONE`, `FAILED`), both `StrEnum`s whose values are lowercase.
  - `StageRecord(name, status, fingerprint, message)` and `TaskRecord(stage, task_id, status, attempts, pid, message)`, both frozen dataclasses.
  - `StateStore(db_path)`, usable as a context manager. Methods:
    - `get_stage(name) -> StageRecord | None`
    - `set_stage(name, status, *, fingerprint=None, message=None)`: a `fingerprint` of `None` keeps the stored one.
    - `reset_stage(name)`
    - `upsert_tasks(stage, task_ids)`
    - `get_tasks(stage) -> list[TaskRecord]`, sorted by `task_id`
    - `get_task(stage, task_id) -> TaskRecord`
    - `set_task(stage, task_id, status, *, pid=None, message=None, attempts_delta=0)`
    - `requeue_failed(stage) -> int`
    - `task_counts(stage) -> dict[TaskStatus, int]`
    - `close()`

- [ ] **Step 1: Write the failing tests** `tests/project/test_state.py`

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/project/test_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gqteashmd.project.state'`

- [ ] **Step 3: Implement `src/gqteashmd/project/state.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/project/test_state.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/gqteashmd/project/state.py tests/project/test_state.py
git commit -m "feat: add SQLite stage/task state store"
```

---

### Task 6: Stage contract and pipeline loading

**Files:**
- Create: `src/gqteashmd/stages/__init__.py` (empty), `src/gqteashmd/stages/base.py`, `src/gqteashmd/stages/pipeline.py`
- Test: `tests/stages/test_base_pipeline.py`

**Interfaces:**
- Consumes: `ShmdConfig`, `section_fingerprint` (Task 3); `ProjectLayout` (Task 4); `ProjectError` (Task 2).
- Produces:
  - `TaskSpec(task_id: str, payload: Mapping[str, Any] = {})`, which validates `task_id` against `[A-Za-z0-9_.-]+`.
  - `StageContext(config: ShmdConfig, layout: ProjectLayout, stage_dir: Path, upstream: Mapping[str, Path])` with method `upstream_dir(name) -> Path`.
  - Abstract `Stage` with attributes `name: str`, `version: int = 1`, `depends_on: tuple[str, ...] = ()`, `config_sections: tuple[str, ...] = ()`, `max_failed_fraction: float = 0.0`. Methods:
    - `enabled(config) -> bool` (default `True`)
    - `fingerprint_inputs(ctx) -> dict`
    - abstract `plan(ctx) -> list[TaskSpec]`
    - abstract `run_task(ctx, task, task_dir) -> dict`
    - abstract `finalize(ctx, results: Mapping[str, Mapping[str, Any]]) -> None`
  - `PipelineFactory = Callable[[ShmdConfig], list[Stage]]`, `DEFAULT_PIPELINE = "gqteashmd.stages.registry:default_pipeline"`, `import_pipeline(ref: str) -> PipelineFactory`, `validate_pipeline(stages: Sequence[Stage]) -> None`.

- [ ] **Step 1: Write the failing tests** `tests/stages/test_base_pipeline.py`

```python
import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import ProjectError
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.stages.base import Stage, StageContext, TaskSpec
from gqteashmd.stages.pipeline import import_pipeline, validate_pipeline


class _Dummy(Stage):
    name = "dummy"
    config_sections = ("molecule", "resources")

    def plan(self, ctx):
        return [TaskSpec("only")]

    def run_task(self, ctx, task, task_dir):
        return {}

    def finalize(self, ctx, results):
        return None


def _stage(name: str, deps: tuple[str, ...] = ()) -> Stage:
    stage = _Dummy()
    stage.name = name
    stage.depends_on = deps
    return stage


def _ctx(tmp_path, write_input) -> StageContext:
    cfg = load_config(write_input(tmp_path)).config
    layout = ProjectLayout(cfg.project.directory)
    return StageContext(cfg, layout, layout.stage_dir(0, "dummy"), {})


def test_task_id_must_be_filesystem_safe():
    TaskSpec("ok_1.x-2")
    with pytest.raises(ValueError, match="a/b"):
        TaskSpec("a/b")
    with pytest.raises(ValueError):
        TaskSpec("")


def test_default_fingerprint_uses_declared_sections(tmp_path, write_input):
    fp = _Dummy().fingerprint_inputs(_ctx(tmp_path, write_input))
    assert set(fp) == {"molecule", "resources"}
    assert fp["molecule"]["charge"] == 0


def test_upstream_dir_lookup(tmp_path, write_input):
    ctx = _ctx(tmp_path, write_input)
    ctx_with = StageContext(ctx.config, ctx.layout, ctx.stage_dir, {"prepare": tmp_path / "p"})
    assert ctx_with.upstream_dir("prepare") == tmp_path / "p"
    with pytest.raises(ProjectError, match="optimize"):
        ctx_with.upstream_dir("optimize")


def test_import_pipeline_resolves_module_attribute():
    factory = import_pipeline("os.path:join")
    assert callable(factory)


@pytest.mark.parametrize(
    "ref, message",
    [
        ("nocolon", "expected 'module:function'"),
        ("no_such_module_xyz:f", "Cannot import"),
        ("os:no_such_attr", "has no attribute"),
    ],
)
def test_import_pipeline_reports_bad_references(ref, message):
    with pytest.raises(ProjectError, match=message):
        import_pipeline(ref)


def test_validate_pipeline_accepts_ordered_dependencies():
    validate_pipeline([_stage("a"), _stage("b", ("a",)), _stage("c", ("a", "b"))])


def test_validate_pipeline_rejects_duplicates():
    with pytest.raises(ProjectError, match="Duplicate stage name 'a'"):
        validate_pipeline([_stage("a"), _stage("a")])


def test_validate_pipeline_rejects_forward_dependencies():
    with pytest.raises(ProjectError, match="'a' depends on 'b'"):
        validate_pipeline([_stage("a", ("b",)), _stage("b")])


def test_validate_pipeline_rejects_unsafe_names():
    with pytest.raises(ProjectError, match="Invalid stage name"):
        validate_pipeline([_stage("bad name")])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/stages/test_base_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gqteashmd.stages'`

- [ ] **Step 3: Implement `src/gqteashmd/stages/base.py`** (and an empty `src/gqteashmd/stages/__init__.py`)

```python
"""The stage contract: plan (controller) → run_task (worker) → finalize (controller).

Rules for implementers:
- plan() must be deterministic: same inputs → same task ids, same order.
- run_task() runs in a separate process, may be re-run after a crash, and must be
  idempotent. It may only write inside `task_dir`. It returns a small JSON-serializable
  dict; large outputs go to files in `task_dir`.
- finalize() combines task results into the stage's artifacts inside `ctx.stage_dir`
  and raises StageFailedError if physical/sanity checks fail.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gqteashmd.config.loader import section_fingerprint
from gqteashmd.config.schema import ShmdConfig
from gqteashmd.errors import ProjectError
from gqteashmd.project.layout import ProjectLayout

SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not SAFE_NAME.fullmatch(self.task_id):
            raise ValueError(f"Invalid task id {self.task_id!r}: use letters, digits, _ . -")


@dataclass(frozen=True)
class StageContext:
    config: ShmdConfig
    layout: ProjectLayout
    stage_dir: Path
    upstream: Mapping[str, Path]

    def upstream_dir(self, name: str) -> Path:
        try:
            return self.upstream[name]
        except KeyError as exc:
            raise ProjectError(
                f"Output of stage '{name}' is not available (stage skipped or not run)"
            ) from exc


class Stage(ABC):
    name: str
    version: int = 1
    depends_on: tuple[str, ...] = ()
    config_sections: tuple[str, ...] = ()
    max_failed_fraction: float = 0.0

    def enabled(self, config: ShmdConfig) -> bool:
        return True

    def fingerprint_inputs(self, ctx: StageContext) -> dict[str, Any]:
        return section_fingerprint(ctx.config, self.config_sections)

    @abstractmethod
    def plan(self, ctx: StageContext) -> list[TaskSpec]: ...

    @abstractmethod
    def run_task(self, ctx: StageContext, task: TaskSpec, task_dir: Path) -> dict[str, Any]: ...

    @abstractmethod
    def finalize(self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]) -> None: ...
```

- [ ] **Step 4: Implement `src/gqteashmd/stages/pipeline.py`**

```python
"""Locate and validate the ordered list of stages."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence

from gqteashmd.config.schema import ShmdConfig
from gqteashmd.errors import ProjectError
from gqteashmd.stages.base import SAFE_NAME, Stage

PipelineFactory = Callable[[ShmdConfig], list[Stage]]
DEFAULT_PIPELINE = "gqteashmd.stages.registry:default_pipeline"


def import_pipeline(ref: str) -> PipelineFactory:
    module_name, sep, attribute = ref.partition(":")
    if not sep or not module_name or not attribute:
        raise ProjectError(f"Invalid pipeline reference {ref!r}; expected 'module:function'")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ProjectError(f"Cannot import pipeline module {module_name!r}: {exc}") from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ProjectError(f"Module {module_name!r} has no attribute {attribute!r}") from exc


def validate_pipeline(stages: Sequence[Stage]) -> None:
    seen: set[str] = set()
    for stage in stages:
        if not SAFE_NAME.fullmatch(stage.name):
            raise ProjectError(f"Invalid stage name {stage.name!r}")
        if stage.name in seen:
            raise ProjectError(f"Duplicate stage name {stage.name!r} in pipeline")
        for dependency in stage.depends_on:
            if dependency not in seen:
                raise ProjectError(
                    f"Stage {stage.name!r} depends on {dependency!r}, "
                    "which does not come earlier in the pipeline"
                )
        seen.add(stage.name)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/stages/test_base_pipeline.py -v`
Expected: 11 passed

- [ ] **Step 6: Commit**

```bash
git add src/gqteashmd/stages tests/stages/test_base_pipeline.py
git commit -m "feat: add stage contract and pipeline validation"
```

---

### Task 7: Geometry parsing and the `prepare` stage

**Files:**
- Create: `src/gqteashmd/chem/__init__.py` (empty), `src/gqteashmd/chem/elements.py`, `src/gqteashmd/chem/xyz.py`, `src/gqteashmd/stages/prepare.py`, `src/gqteashmd/stages/registry.py`
- Test: `tests/chem/test_xyz.py`, `tests/stages/test_prepare.py`

**Interfaces:**
- Consumes: `Stage`, `StageContext`, `TaskSpec` (Task 6); `TaskError`, `ConfigError` (Task 2); `write_json_atomic`, `write_text_atomic` (Task 2).
- Produces:
  - `normalize_symbol(symbol: str) -> str` and `atomic_number(symbol: str) -> int` (both raise `ValueError` for unknown elements).
  - `Geometry(symbols: tuple[str, ...], coords_angstrom: tuple[tuple[float, float, float], ...], comment: str = "")`, `parse_xyz(text: str) -> Geometry` (raises `ValueError` with the line number), `format_xyz(geometry: Geometry) -> str`.
  - `PrepareStage`, with name `"prepare"` and a single task `"molecule"`. It writes `<stage_dir>/geometry.xyz` and `<stage_dir>/molecule.json` (`schema_version`, `symbols`, `atomic_numbers`, `coords_angstrom`, `charge`, `multiplicity`, `n_electrons`).
  - `default_pipeline(config) -> list[Stage]`, which returns `[PrepareStage()]`.

- [ ] **Step 1: Write the failing tests** `tests/chem/test_xyz.py`

```python
import pytest

from gqteashmd.chem.elements import atomic_number, normalize_symbol
from gqteashmd.chem.xyz import Geometry, format_xyz, parse_xyz


def test_atomic_numbers_and_normalization():
    assert atomic_number("H") == 1
    assert atomic_number("cl") == 17
    assert atomic_number("RN") == 86
    assert normalize_symbol("br") == "Br"
    with pytest.raises(ValueError, match="Xx"):
        atomic_number("Xx")


def test_parse_water(water_xyz):
    geom = parse_xyz(water_xyz)
    assert geom.symbols == ("O", "H", "H")
    assert geom.coords_angstrom[1] == (0.0, 0.7572, -0.4692)
    assert geom.comment == "water"


def test_parse_normalizes_symbols_and_ignores_extra_columns_and_trailing_blank_lines():
    geom = parse_xyz("1\n\ncl 1.0 2.0 3.0 0.5\n\n\n")
    assert geom.symbols == ("Cl",)
    assert geom.coords_angstrom == ((1.0, 2.0, 3.0),)


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "line 1"),
        ("two\n\nH 0 0 0\n", "line 1"),
        ("0\n\n", "at least one atom"),
        ("2\n\nH 0 0 0\n", "expected 2 atoms"),
        ("1\n\nH 0 0\n", "line 3"),
        ("1\n\nH 0 x 0\n", "line 3"),
        ("1\n\nQq 0 0 0\n", "line 3"),
        ("1\n\nH 0 0 0\nH 1 1 1\n", "more lines"),
    ],
)
def test_parse_errors_report_location(text, message):
    with pytest.raises(ValueError, match=message):
        parse_xyz(text)


def test_format_roundtrip(water_xyz):
    geom = parse_xyz(water_xyz)
    again = parse_xyz(format_xyz(geom))
    assert again == geom
    assert format_xyz(Geometry(("H",), ((0.0, 0.0, 0.0),))).splitlines()[0] == "1"
```

- [ ] **Step 2: Write the failing tests** `tests/stages/test_prepare.py`

```python
import json

import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import ConfigError, TaskError
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.stages.base import StageContext, TaskSpec
from gqteashmd.stages.prepare import PrepareStage
from gqteashmd.stages.registry import default_pipeline


def _ctx(path) -> StageContext:
    cfg = load_config(path).config
    layout = ProjectLayout(cfg.project.directory)
    return StageContext(cfg, layout, layout.stage_dir(0, "prepare"), {})


def test_plan_has_single_task(tmp_path, write_input):
    ctx = _ctx(write_input(tmp_path))
    assert PrepareStage().plan(ctx) == [TaskSpec("molecule")]


def test_run_task_counts_electrons(tmp_path, write_input):
    ctx = _ctx(write_input(tmp_path))
    result = PrepareStage().run_task(ctx, TaskSpec("molecule"), tmp_path / "task")
    assert result["n_electrons"] == 10
    assert result["atomic_numbers"] == [8, 1, 1]
    assert result["symbols"] == ["O", "H", "H"]


def test_inconsistent_multiplicity_is_non_retryable(tmp_path, write_input):
    ctx = _ctx(write_input(tmp_path, molecule="multiplicity = 2"))
    with pytest.raises(TaskError, match="inconsistent") as info:
        PrepareStage().run_task(ctx, TaskSpec("molecule"), tmp_path / "task")
    assert info.value.retryable is False


def test_bad_geometry_is_non_retryable(tmp_path, write_input):
    path = write_input(tmp_path)
    (tmp_path / "water.xyz").write_text("3\n\nO 0 0 0\n")
    with pytest.raises(TaskError, match="Invalid geometry") as info:
        PrepareStage().run_task(_ctx(path), TaskSpec("molecule"), tmp_path / "task")
    assert info.value.retryable is False


def test_finalize_writes_artifacts(tmp_path, write_input):
    ctx = _ctx(write_input(tmp_path))
    stage = PrepareStage()
    result = stage.run_task(ctx, TaskSpec("molecule"), tmp_path / "task")
    stage.finalize(ctx, {"molecule": result})
    data = json.loads((ctx.stage_dir / "molecule.json").read_text())
    assert data["schema_version"] == 1
    assert data["n_electrons"] == 10
    assert (ctx.stage_dir / "geometry.xyz").read_text().splitlines()[0] == "3"


def test_fingerprint_tracks_geometry_content_not_path(tmp_path, write_input):
    path = write_input(tmp_path)
    stage = PrepareStage()
    first = stage.fingerprint_inputs(_ctx(path))
    assert set(first) == {"charge", "multiplicity", "geometry_sha256"}
    (tmp_path / "water.xyz").write_text((tmp_path / "water.xyz").read_text() + "\n")
    assert stage.fingerprint_inputs(_ctx(path)) != first


def test_fingerprint_with_missing_geometry_is_config_error(tmp_path, write_input):
    path = write_input(tmp_path)
    (tmp_path / "water.xyz").unlink()
    with pytest.raises(ConfigError, match="Geometry file not found"):
        PrepareStage().fingerprint_inputs(_ctx(path))


def test_default_pipeline_contains_prepare(tmp_path, write_input):
    cfg = load_config(write_input(tmp_path)).config
    assert [s.name for s in default_pipeline(cfg)] == ["prepare"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/chem/test_xyz.py tests/stages/test_prepare.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gqteashmd.chem'`

- [ ] **Step 4: Implement `src/gqteashmd/chem/elements.py`** (and an empty `src/gqteashmd/chem/__init__.py`)

```python
"""Element symbols H..Rn and atomic numbers."""

from __future__ import annotations

_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn"
).split()

_Z_BY_SYMBOL = {symbol: z for z, symbol in enumerate(_SYMBOLS, start=1)}


def normalize_symbol(symbol: str) -> str:
    candidate = symbol.strip().capitalize()
    if candidate not in _Z_BY_SYMBOL:
        raise ValueError(f"Unknown element symbol {candidate!r}")
    return candidate


def atomic_number(symbol: str) -> int:
    return _Z_BY_SYMBOL[normalize_symbol(symbol)]
```

- [ ] **Step 5: Implement `src/gqteashmd/chem/xyz.py`**

```python
"""XYZ geometry files (coordinates in Angstrom)."""

from __future__ import annotations

from dataclasses import dataclass

from gqteashmd.chem.elements import normalize_symbol

Coord = tuple[float, float, float]


@dataclass(frozen=True)
class Geometry:
    symbols: tuple[str, ...]
    coords_angstrom: tuple[Coord, ...]
    comment: str = ""


def parse_xyz(text: str) -> Geometry:
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        raise ValueError("line 1: expected the number of atoms, file is empty")
    try:
        n_atoms = int(lines[0].split()[0])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"line 1: expected the number of atoms, got {lines[0]!r}") from exc
    if n_atoms < 1:
        raise ValueError("line 1: the geometry must contain at least one atom")
    comment = lines[1].strip() if len(lines) > 1 else ""
    atom_lines = lines[2:]
    if len(atom_lines) < n_atoms:
        raise ValueError(f"expected {n_atoms} atoms but found {len(atom_lines)} atom lines")
    if len(atom_lines) > n_atoms:
        raise ValueError(f"file contains more lines than the {n_atoms} atoms declared on line 1")
    symbols: list[str] = []
    coords: list[Coord] = []
    for offset, line in enumerate(atom_lines, start=3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"line {offset}: expected 'symbol x y z', got {line!r}")
        try:
            symbols.append(normalize_symbol(fields[0]))
            coords.append((float(fields[1]), float(fields[2]), float(fields[3])))
        except ValueError as exc:
            raise ValueError(f"line {offset}: {exc}") from exc
    return Geometry(tuple(symbols), tuple(coords), comment)


def format_xyz(geometry: Geometry) -> str:
    rows = [str(len(geometry.symbols)), geometry.comment]
    for symbol, (x, y, z) in zip(geometry.symbols, geometry.coords_angstrom, strict=True):
        rows.append(f"{symbol:<3}{x:16.10f}{y:16.10f}{z:16.10f}")
    return "\n".join(rows) + "\n"
```

- [ ] **Step 6: Implement `src/gqteashmd/stages/prepare.py`**

```python
"""Stage 0: validate the molecule and store a normalized copy in the project."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gqteashmd.chem.elements import atomic_number
from gqteashmd.chem.xyz import Geometry, format_xyz, parse_xyz
from gqteashmd.errors import ConfigError, TaskError
from gqteashmd.stages.base import Stage, StageContext, TaskSpec
from gqteashmd.util.fileio import write_json_atomic, write_text_atomic


class PrepareStage(Stage):
    name = "prepare"
    version = 1
    config_sections = ("molecule",)

    def fingerprint_inputs(self, ctx: StageContext) -> dict[str, Any]:
        molecule = ctx.config.molecule
        try:
            digest = hashlib.sha256(molecule.geometry.read_bytes()).hexdigest()
        except FileNotFoundError as exc:
            raise ConfigError(f"Geometry file not found: {molecule.geometry}") from exc
        return {
            "charge": molecule.charge,
            "multiplicity": molecule.multiplicity,
            "geometry_sha256": digest,
        }

    def plan(self, ctx: StageContext) -> list[TaskSpec]:
        return [TaskSpec("molecule")]

    def run_task(self, ctx: StageContext, task: TaskSpec, task_dir: Path) -> dict[str, Any]:
        molecule = ctx.config.molecule
        try:
            geometry = parse_xyz(molecule.geometry.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise TaskError(f"Geometry file not found: {molecule.geometry}", retryable=False) from exc
        except ValueError as exc:
            raise TaskError(
                f"Invalid geometry file {molecule.geometry}: {exc}", retryable=False
            ) from exc
        numbers = [atomic_number(symbol) for symbol in geometry.symbols]
        n_electrons = sum(numbers) - molecule.charge
        unpaired = molecule.multiplicity - 1
        if n_electrons <= 0 or unpaired > n_electrons or n_electrons % 2 != unpaired % 2:
            raise TaskError(
                f"Charge {molecule.charge} and multiplicity {molecule.multiplicity} are "
                f"inconsistent with {n_electrons} electrons. Check [molecule] in the input file.",
                retryable=False,
            )
        return {
            "symbols": list(geometry.symbols),
            "atomic_numbers": numbers,
            "coords_angstrom": [list(c) for c in geometry.coords_angstrom],
            "charge": molecule.charge,
            "multiplicity": molecule.multiplicity,
            "n_electrons": n_electrons,
        }

    def finalize(self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]) -> None:
        data = dict(results["molecule"])
        geometry = Geometry(
            tuple(data["symbols"]),
            tuple((c[0], c[1], c[2]) for c in data["coords_angstrom"]),
            comment=ctx.config.project.name,
        )
        write_text_atomic(ctx.stage_dir / "geometry.xyz", format_xyz(geometry))
        write_json_atomic(ctx.stage_dir / "molecule.json", {"schema_version": 1, **data})
```

- [ ] **Step 7: Implement `src/gqteashmd/stages/registry.py`**

```python
"""The production pipeline. Later sub-plans append optimize, frequencies, sampling, ..."""

from __future__ import annotations

from gqteashmd.config.schema import ShmdConfig
from gqteashmd.stages.base import Stage
from gqteashmd.stages.prepare import PrepareStage


def default_pipeline(config: ShmdConfig) -> list[Stage]:
    return [PrepareStage()]
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/chem/test_xyz.py tests/stages/test_prepare.py -v`
Expected: 20 passed (12 + 8)

- [ ] **Step 9: Commit**

```bash
git add src/gqteashmd/chem src/gqteashmd/stages/prepare.py src/gqteashmd/stages/registry.py tests/chem tests/stages/test_prepare.py
git commit -m "feat: add xyz parsing and the prepare stage"
```

---

### Task 8: Fake stages and the task worker

**Files:**
- Create: `src/gqteashmd/testing/__init__.py` (empty), `src/gqteashmd/testing/fake_stages.py`, `src/gqteashmd/worker.py`
- Test: `tests/test_worker.py`

**Interfaces:**
- Consumes: `load_snapshot`, `write_snapshot`, `load_config` (Task 3); `ProjectLayout` (Task 4); `Stage`, `StageContext`, `TaskSpec` (Task 6); `import_pipeline` (Task 6); `PrepareStage` (Task 7); errors and fileio (Task 2).
- Produces:
  - Constants `TASK_FILE = "task.json"` and `RESULT_FILE = "result.json"`.
  - `write_task_file(path, *, pipeline_ref: str, config_snapshot: Path, ctx: StageContext, stage_name: str, task: TaskSpec, task_dir: Path) -> None`.
  - `read_result(task_dir: Path) -> dict | None`: returns `None` if the file is missing or corrupt.
  - `run_task_file(task_file: Path) -> int`: 0 on success, 1 on a handled error.
  - `main(argv: list[str] | None = None) -> int`.
  - Result file format: `{"status": "ok", "result": {...}}` or `{"status": "error", "error_type": str, "message": str, "retryable": bool, "traceback"?: str}`.
  - `gqteashmd.testing.fake_stages`: `BEHAVIOUR_ENV = "GQTEASHMD_FAKE_BEHAVIOUR"`, `FakeStage`, and `make_pipeline(config)`, which returns stages `prepare, optimize, frequencies, sampling(3 tasks), dynamics(4 tasks, max_failed_fraction 0.25)`. The `GQTEASHMD_FAKE_BEHAVIOUR` environment variable takes JSON with these keys:
    - `sleep` (seconds);
    - `fail`, `unexpected`, `crash_once`, `retryable_once` (lists of `"stage:task_id"`);
    - `skip_optimize` (bool).

- [ ] **Step 1: Write the failing tests** `tests/test_worker.py`

```python
import json

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
    assert read_result(task_dir) == {"status": "ok", "result": {"index": 1, "upstream": []}}
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_worker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gqteashmd.testing'`

- [ ] **Step 3: Implement `src/gqteashmd/testing/fake_stages.py`** (and an empty `src/gqteashmd/testing/__init__.py`)

```python
"""Deterministic fake stages for orchestrator tests. Not used by production pipelines.

Behaviour is controlled by the JSON environment variable GQTEASHMD_FAKE_BEHAVIOUR, which
worker subprocesses inherit from the controller:
  sleep: float seconds per task
  fail / unexpected / crash_once / retryable_once: lists of "stage:task_id"
  skip_optimize: bool
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gqteashmd.config.schema import ShmdConfig
from gqteashmd.errors import TaskError
from gqteashmd.stages.base import Stage, StageContext, TaskSpec
from gqteashmd.stages.prepare import PrepareStage
from gqteashmd.util.fileio import read_json, write_json_atomic

BEHAVIOUR_ENV = "GQTEASHMD_FAKE_BEHAVIOUR"


def _behaviour() -> dict[str, Any]:
    return json.loads(os.environ.get(BEHAVIOUR_ENV, "{}"))


class FakeStage(Stage):
    def __init__(
        self,
        name: str,
        n_tasks: int,
        depends_on: Sequence[str] = (),
        *,
        max_failed_fraction: float = 0.0,
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.n_tasks = n_tasks
        self.depends_on = tuple(depends_on)
        self.max_failed_fraction = max_failed_fraction
        self._enabled = enabled

    def enabled(self, config: ShmdConfig) -> bool:
        return self._enabled

    def plan(self, ctx: StageContext) -> list[TaskSpec]:
        return [TaskSpec(f"t{i:03d}", {"index": i}) for i in range(self.n_tasks)]

    def run_task(self, ctx: StageContext, task: TaskSpec, task_dir: Path) -> dict[str, Any]:
        key = f"{self.name}:{task.task_id}"
        behaviour = _behaviour()
        time.sleep(float(behaviour.get("sleep", 0.0)))
        task_dir.mkdir(parents=True, exist_ok=True)
        if key in behaviour.get("crash_once", []):
            marker = task_dir / "crashed_once"
            if not marker.exists():
                marker.write_text("crashed")
                os._exit(137)
        if key in behaviour.get("retryable_once", []):
            marker = task_dir / "failed_once"
            if not marker.exists():
                marker.write_text("failed")
                raise TaskError(f"transient fake failure in {key}", retryable=True)
        if key in behaviour.get("fail", []):
            raise TaskError(f"fake failure in {key}", retryable=False)
        if key in behaviour.get("unexpected", []):
            raise RuntimeError(f"unexpected fake error in {key}")
        upstream = sorted(
            name for name, path in ctx.upstream.items() if (path / "summary.json").exists()
        )
        (task_dir / "out.txt").write_text(key)
        return {"index": task.payload["index"], "upstream": upstream}

    def finalize(self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]) -> None:
        values = sorted(int(r["index"]) for r in results.values())
        write_json_atomic(ctx.stage_dir / "summary.json", {"values": values})


def make_pipeline(config: ShmdConfig) -> list[Stage]:
    behaviour = _behaviour()
    return [
        PrepareStage(),
        FakeStage("optimize", 1, ("prepare",), enabled=not behaviour.get("skip_optimize", False)),
        FakeStage("frequencies", 1, ("prepare",)),
        FakeStage("sampling", 3, ("frequencies",)),
        FakeStage("dynamics", 4, ("sampling",), max_failed_fraction=0.25),
    ]


def read_summary(stage_dir: Path) -> dict[str, Any]:
    return read_json(stage_dir / "summary.json")
```

- [ ] **Step 4: Implement `src/gqteashmd/worker.py`**

```python
"""Task worker: `python -m gqteashmd.worker <task.json>`. Runs exactly one task of one stage."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

from gqteashmd.config.loader import load_snapshot
from gqteashmd.errors import ProjectError, ShmdError, TaskError
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.stages.base import StageContext, TaskSpec
from gqteashmd.stages.pipeline import import_pipeline
from gqteashmd.util.fileio import read_json, write_json_atomic

TASK_FILE = "task.json"
RESULT_FILE = "result.json"


def write_task_file(
    path: Path,
    *,
    pipeline_ref: str,
    config_snapshot: Path,
    ctx: StageContext,
    stage_name: str,
    task: TaskSpec,
    task_dir: Path,
) -> None:
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "pipeline_ref": pipeline_ref,
            "config_snapshot": str(config_snapshot),
            "project_root": str(ctx.layout.root),
            "stage": stage_name,
            "stage_dir": str(ctx.stage_dir),
            "upstream": {name: str(p) for name, p in ctx.upstream.items()},
            "task_id": task.task_id,
            "payload": dict(task.payload),
            "task_dir": str(task_dir),
        },
    )


def read_result(task_dir: Path) -> dict[str, Any] | None:
    try:
        return read_json(Path(task_dir) / RESULT_FILE)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _error(exc: BaseException, *, retryable: bool, with_traceback: bool) -> dict[str, Any]:
    data: dict[str, Any] = {
        "status": "error",
        "error_type": type(exc).__name__,
        "message": str(exc),
        "retryable": retryable,
    }
    if with_traceback:
        data["traceback"] = traceback.format_exc()
    return data


def run_task_file(task_file: Path) -> int:
    spec = read_json(task_file)
    task_dir = Path(spec["task_dir"])
    try:
        loaded = load_snapshot(Path(spec["config_snapshot"]))
        stages = {s.name: s for s in import_pipeline(spec["pipeline_ref"])(loaded.config)}
        stage = stages.get(spec["stage"])
        if stage is None:
            raise ProjectError(f"Stage {spec['stage']!r} is not part of the pipeline")
        ctx = StageContext(
            config=loaded.config,
            layout=ProjectLayout(Path(spec["project_root"])),
            stage_dir=Path(spec["stage_dir"]),
            upstream={name: Path(p) for name, p in spec["upstream"].items()},
        )
        result = stage.run_task(ctx, TaskSpec(spec["task_id"], spec["payload"]), task_dir)
    except TaskError as exc:
        write_json_atomic(task_dir / RESULT_FILE, _error(exc, retryable=exc.retryable, with_traceback=False))
        return 1
    except ShmdError as exc:
        write_json_atomic(task_dir / RESULT_FILE, _error(exc, retryable=False, with_traceback=False))
        return 1
    except Exception as exc:  # a bug: keep the traceback for the developer
        write_json_atomic(task_dir / RESULT_FILE, _error(exc, retryable=False, with_traceback=True))
        return 1
    try:
        write_json_atomic(task_dir / RESULT_FILE, {"status": "ok", "result": result})
    except TypeError as exc:
        write_json_atomic(
            task_dir / RESULT_FILE,
            _error(TypeError(f"task result is not JSON-serializable: {exc}"),
                   retryable=False, with_traceback=False),
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m gqteashmd.worker TASK_FILE", file=sys.stderr)
        return 2
    return run_task_file(Path(args[0]))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_worker.py -v`
Expected: 5 passed

- [ ] **Step 6: Format, then lint**

Run: `ruff format src tests && ruff check --fix src tests`
Expected: the formatter reflows a few long lines (e.g. in `worker.py`), then `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add src/gqteashmd/testing src/gqteashmd/worker.py tests/test_worker.py
git commit -m "feat: add task worker and deterministic fake stages"
```

---

### Task 9: Local executor and process utilities

**Files:**
- Create: `src/gqteashmd/execution/__init__.py` (empty), `src/gqteashmd/execution/local.py`, `src/gqteashmd/execution/processes.py`
- Test: `tests/execution/test_local.py`, `tests/execution/test_processes.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (stdlib only).
- Produces:
  - `Job(key: str, task_file: Path, log_file: Path)` and `JobOutcome(job: Job, returncode: int | None)`, where `None` means terminated by a stop request.
  - `LocalExecutor(*, max_parallel, threads_per_task, memory_mb_per_task, poll_interval=0.5, stop_grace_period=30.0, command=None)`. Methods:
    - `environment() -> dict[str, str]`
    - `run(jobs, *, on_start: Callable[[Job, int], None], on_finish: Callable[[JobOutcome], None], should_stop: Callable[[], bool]) -> bool`: returns `False` if stopped early.
  - `THREAD_ENV_VARS`.
  - `is_worker_process(pid: int, task_file: Path) -> bool` and `terminate_process_group(pid: int, grace: float) -> None`.

- [ ] **Step 1: Write the failing tests** `tests/execution/test_local.py`

```python
import json
import sys
import time

from gqteashmd.execution.local import Job, LocalExecutor

SCRIPT = """
import json, os, pathlib, sys, time
spec = json.loads(pathlib.Path(sys.argv[1]).read_text())
print("hello from", spec["name"], flush=True)
time.sleep(spec.get("sleep", 0))
pathlib.Path(spec["out"]).write_text(
    os.environ.get("OMP_NUM_THREADS", "") + " " + os.environ.get("PYSCF_MAX_MEMORY", "")
)
sys.exit(spec.get("exit", 0))
"""


def _jobs(tmp_path, specs):
    script = tmp_path / "script.py"
    script.write_text(SCRIPT)
    jobs = []
    for i, spec in enumerate(specs):
        task_dir = tmp_path / f"job{i}"
        task_dir.mkdir()
        task_file = task_dir / "task.json"
        task_file.write_text(json.dumps({"name": f"job{i}", "out": str(task_dir / "out"), **spec}))
        jobs.append(Job(f"job{i}", task_file, task_dir / "worker.log"))
    return script, jobs


def _executor(script, **kw):
    defaults = dict(max_parallel=2, threads_per_task=3, memory_mb_per_task=1500,
                    poll_interval=0.02, stop_grace_period=5.0)
    defaults.update(kw)
    return LocalExecutor(command=[sys.executable, str(script)], **defaults)


def test_runs_all_jobs_and_reports_exit_codes(tmp_path):
    script, jobs = _jobs(tmp_path, [{}, {"exit": 3}, {}])
    outcomes = {}
    finished = _executor(script).run(
        jobs, on_start=lambda job, pid: None,
        on_finish=lambda o: outcomes.__setitem__(o.job.key, o.returncode),
        should_stop=lambda: False,
    )
    assert finished is True
    assert outcomes == {"job0": 0, "job1": 3, "job2": 0}


def test_never_exceeds_max_parallel(tmp_path):
    script, jobs = _jobs(tmp_path, [{"sleep": 0.3}] * 5)
    active, peak = set(), []

    def start(job, pid):
        active.add(job.key)
        peak.append(len(active))

    _executor(script, max_parallel=2).run(
        jobs, on_start=start, on_finish=lambda o: active.discard(o.job.key),
        should_stop=lambda: False,
    )
    assert max(peak) == 2


def test_sets_thread_and_memory_environment_and_captures_log(tmp_path):
    script, jobs = _jobs(tmp_path, [{}])
    _executor(script).run(jobs, on_start=lambda j, p: None, on_finish=lambda o: None,
                          should_stop=lambda: False)
    assert (tmp_path / "job0" / "out").read_text() == "3 1500"
    assert "hello from job0" in (tmp_path / "job0" / "worker.log").read_text()


def test_stop_terminates_running_jobs(tmp_path):
    script, jobs = _jobs(tmp_path, [{"sleep": 60}] * 3)
    started, outcomes = [], []
    t0 = time.monotonic()
    finished = _executor(script).run(
        jobs, on_start=lambda job, pid: started.append(job.key),
        on_finish=outcomes.append, should_stop=lambda: len(started) >= 2,
    )
    assert finished is False
    assert sorted(o.job.key for o in outcomes) == ["job0", "job1"]
    assert all(o.returncode is None for o in outcomes)
    assert time.monotonic() - t0 < 20
```

- [ ] **Step 2: Write the failing tests** `tests/execution/test_processes.py`

```python
import subprocess
import sys

from gqteashmd.execution.processes import is_worker_process, terminate_process_group


def test_detects_and_terminates_worker_process(tmp_path):
    task_file = tmp_path / "task.json"
    task_file.write_text("{}")
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", str(task_file)],
        start_new_session=True,
    )
    try:
        assert is_worker_process(proc.pid, task_file) is True
        assert is_worker_process(proc.pid, tmp_path / "other.json") is False
        terminate_process_group(proc.pid, grace=5.0)
        assert proc.wait(timeout=5) != 0
    finally:
        if proc.poll() is None:
            proc.kill()


def test_unknown_pid_is_not_a_worker(tmp_path):
    assert is_worker_process(2**22 + 12345, tmp_path / "task.json") is False
    terminate_process_group(2**22 + 12345, grace=0.1)  # must not raise
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/execution -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gqteashmd.execution'`

- [ ] **Step 4: Implement `src/gqteashmd/execution/processes.py`** (and an empty `src/gqteashmd/execution/__init__.py`)

```python
"""Linux process helpers used to clean up orphaned workers after a controller crash."""

from __future__ import annotations

import contextlib
import os
import signal
import time
from pathlib import Path


def is_worker_process(pid: int, task_file: Path) -> bool:
    """True if `pid` is alive and its command line contains `task_file`."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return str(task_file).encode() in cmdline


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    state = stat.rsplit(")", 1)[1].split()[0]
    return state != "Z"  # a zombie has already exited


def terminate_process_group(pid: int, grace: float) -> None:
    """SIGTERM the process group led by `pid`; SIGKILL it after `grace` seconds."""
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not _alive(pid):
            return
        time.sleep(0.05)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)
```

- [ ] **Step 5: Implement `src/gqteashmd/execution/local.py`**

```python
"""Run task subprocesses on the local machine with a concurrency limit."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@dataclass(frozen=True)
class Job:
    key: str
    task_file: Path
    log_file: Path


@dataclass(frozen=True)
class JobOutcome:
    job: Job
    returncode: int | None  # None: terminated because a stop was requested


@dataclass
class _Running:
    job: Job
    process: subprocess.Popen[bytes]
    log: IO[bytes]


def _signal_group(pid: int, sig: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, sig)


class LocalExecutor:
    def __init__(
        self,
        *,
        max_parallel: int,
        threads_per_task: int,
        memory_mb_per_task: int,
        poll_interval: float = 0.5,
        stop_grace_period: float = 30.0,
        command: Sequence[str] | None = None,
    ) -> None:
        if max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        self.max_parallel = max_parallel
        self.threads_per_task = threads_per_task
        self.memory_mb_per_task = memory_mb_per_task
        self.poll_interval = poll_interval
        self.stop_grace_period = stop_grace_period
        self.command = list(command) if command is not None else [sys.executable, "-m", "gqteashmd.worker"]

    def environment(self) -> dict[str, str]:
        env = dict(os.environ)
        for name in THREAD_ENV_VARS:
            env[name] = str(self.threads_per_task)
        env["PYSCF_MAX_MEMORY"] = str(self.memory_mb_per_task)
        return env

    def run(
        self,
        jobs: Sequence[Job],
        *,
        on_start: Callable[[Job, int], None],
        on_finish: Callable[[JobOutcome], None],
        should_stop: Callable[[], bool],
    ) -> bool:
        """Run all jobs. Returns False if a stop was requested before all jobs finished."""
        queue = deque(jobs)
        running: dict[str, _Running] = {}
        env = self.environment()
        while queue or running:
            if should_stop():
                self._terminate_all(running, on_finish)
                return False
            while queue and len(running) < self.max_parallel:
                job = queue.popleft()
                item = self._launch(job, env)
                running[job.key] = item
                on_start(job, item.process.pid)
            for key in list(running):
                item = running[key]
                code = item.process.poll()
                if code is not None:
                    item.log.close()
                    del running[key]
                    on_finish(JobOutcome(item.job, code))
            if queue or running:
                time.sleep(self.poll_interval)
        return True

    def _launch(self, job: Job, env: dict[str, str]) -> _Running:
        job.log_file.parent.mkdir(parents=True, exist_ok=True)
        log = open(job.log_file, "ab")
        try:
            process = subprocess.Popen(
                [*self.command, str(job.task_file)],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                cwd=job.task_file.parent,
                start_new_session=True,  # Ctrl-C on the controller does not hit workers
            )
        except OSError:
            log.close()
            raise
        return _Running(job, process, log)

    def _terminate_all(
        self, running: dict[str, _Running], on_finish: Callable[[JobOutcome], None]
    ) -> None:
        for item in running.values():
            _signal_group(item.process.pid, signal.SIGTERM)
        deadline = time.monotonic() + self.stop_grace_period
        for item in running.values():
            try:
                item.process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                _signal_group(item.process.pid, signal.SIGKILL)
                item.process.wait()
            item.log.close()
            on_finish(JobOutcome(item.job, None))
        running.clear()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/execution -v`
Expected: 6 passed

- [ ] **Step 7: Commit**

```bash
git add src/gqteashmd/execution tests/execution
git commit -m "feat: add local subprocess executor with graceful stop"
```

---

### Task 10: The orchestrator

**Files:**
- Create: `src/gqteashmd/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes everything from Tasks 2–9, by exact name:
  - `LoadedConfig`, `write_snapshot`
  - `ProjectLayout`, `ProjectLock`
  - `StateStore`, `StageStatus`, `TaskStatus`
  - `Stage`, `StageContext`, `TaskSpec`
  - `import_pipeline`, `validate_pipeline`, `DEFAULT_PIPELINE`
  - `Job`, `JobOutcome`, `LocalExecutor`
  - `is_worker_process`, `terminate_process_group`
  - `TASK_FILE`, `RESULT_FILE`, `read_result`, `write_task_file`
- Produces:
  - `RunSummary(completed: list[str], up_to_date: list[str], skipped: list[str], stopped: bool)`.
  - `Orchestrator(loaded: LoadedConfig, *, pipeline_ref: str = DEFAULT_PIPELINE, rerun_from: str | None = None)` with attributes `layout: ProjectLayout`, `stages: list[Stage]`, and method `run(should_stop: Callable[[], bool] = lambda: False) -> RunSummary`.
  - Raises `ConfigError` (unknown `rerun_from`), `ProjectError` (lock held), `StaleStageError` and `StageFailedError`.

- [ ] **Step 1: Write the failing tests** `tests/test_orchestrator.py`

```python
import json

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
    with ProjectLock(layout.lock_file), pytest.raises(ProjectError, match="Another gqteashmd controller"):
        _run(project)


def test_unknown_rerun_stage_is_config_error(project, behaviour):
    with pytest.raises(ConfigError, match="frobnicate"):
        Orchestrator(load_config(project), pipeline_ref=FAKE, rerun_from="frobnicate")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_orchestrator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gqteashmd.orchestrator'`

- [ ] **Step 3: Implement `src/gqteashmd/orchestrator.py`**

```python
"""The workflow controller: runs stages in order, tasks in parallel, and survives restarts."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gqteashmd.config.loader import LoadedConfig, write_snapshot
from gqteashmd.errors import ConfigError, ProjectError, ShmdError, StageFailedError, StaleStageError
from gqteashmd.execution.local import Job, JobOutcome, LocalExecutor
from gqteashmd.execution.processes import is_worker_process, terminate_process_group
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.project.lock import ProjectLock
from gqteashmd.project.state import StageStatus, StateStore, TaskRecord, TaskStatus
from gqteashmd.stages.base import Stage, StageContext, TaskSpec
from gqteashmd.stages.pipeline import DEFAULT_PIPELINE, import_pipeline, validate_pipeline
from gqteashmd.worker import RESULT_FILE, TASK_FILE, read_result, write_task_file

log = logging.getLogger("gqteashmd.orchestrator")

SKIPPED_FINGERPRINT = "skipped"


@dataclass
class RunSummary:
    completed: list[str] = field(default_factory=list)
    up_to_date: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    stopped: bool = False


def _fingerprint(data: Mapping[str, Any]) -> str:
    text = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Orchestrator:
    def __init__(
        self,
        loaded: LoadedConfig,
        *,
        pipeline_ref: str = DEFAULT_PIPELINE,
        rerun_from: str | None = None,
    ) -> None:
        self.loaded = loaded
        self.config = loaded.config
        self.pipeline_ref = pipeline_ref
        self.stages: list[Stage] = import_pipeline(pipeline_ref)(self.config)
        validate_pipeline(self.stages)
        names = [s.name for s in self.stages]
        if rerun_from is not None and rerun_from not in names:
            raise ConfigError(
                f"Unknown stage {rerun_from!r} for --rerun-from; stages are: {', '.join(names)}"
            )
        self.rerun_from = rerun_from
        self.layout = ProjectLayout(self.config.project.directory)
        resources, execution = self.config.resources, self.config.execution
        self.executor = LocalExecutor(
            max_parallel=resources.max_parallel_tasks,
            threads_per_task=resources.threads_per_task,
            memory_mb_per_task=resources.memory_mb_per_task,
            poll_interval=execution.poll_interval_s,
            stop_grace_period=execution.stop_grace_period_s,
        )

    # ------------------------------------------------------------------ public API
    def run(self, should_stop: Callable[[], bool] = lambda: False) -> RunSummary:
        self.layout.ensure()
        with ProjectLock(self.layout.lock_file), StateStore(self.layout.state_db) as state:
            write_snapshot(self.loaded, self.layout.config_snapshot)
            if self.rerun_from is not None:
                self._reset_from(state, self.rerun_from)
            return self._run_pipeline(state, should_stop)

    # ------------------------------------------------------------------ pipeline
    def _run_pipeline(self, state: StateStore, should_stop: Callable[[], bool]) -> RunSummary:
        summary = RunSummary()
        fingerprints: dict[str, str] = {}
        upstream_dirs: dict[str, Path] = {}
        for index, stage in enumerate(self.stages):
            if should_stop():
                summary.stopped = True
                break
            stage_dir = self.layout.stage_dir(index, stage.name)
            ctx = StageContext(
                self.config,
                self.layout,
                stage_dir,
                {d: upstream_dirs[d] for d in stage.depends_on if d in upstream_dirs},
            )
            if not stage.enabled(self.config):
                fingerprints[stage.name] = SKIPPED_FINGERPRINT
                state.set_stage(stage.name, StageStatus.SKIPPED, fingerprint=SKIPPED_FINGERPRINT)
                summary.skipped.append(stage.name)
                log.info("Stage %s: skipped (disabled in the input file)", stage.name)
                continue
            fp = _fingerprint(
                {
                    "stage": stage.name,
                    "version": stage.version,
                    "inputs": stage.fingerprint_inputs(ctx),
                    "upstream": {d: fingerprints[d] for d in stage.depends_on},
                }
            )
            fingerprints[stage.name] = fp
            record = state.get_stage(stage.name)
            if record is not None and record.fingerprint != fp:
                if record.status in (StageStatus.DONE, StageStatus.RUNNING):
                    raise StaleStageError(
                        f"Stage '{stage.name}' has results produced from different inputs "
                        "(the input file or an earlier stage changed). Run "
                        f"`gqteashmd run {self.loaded.source} --rerun-from {stage.name}` "
                        "to move the old results aside and recompute from this stage onward."
                    )
                log.info("Stage %s: inputs changed since last attempt; starting over", stage.name)
                self._reset_stage(state, index, stage)
                record = None
            if record is not None and record.status == StageStatus.DONE:
                summary.up_to_date.append(stage.name)
                upstream_dirs[stage.name] = stage_dir
                log.info("Stage %s: up to date", stage.name)
                continue
            if not self._run_stage(state, stage, ctx, fp, should_stop):
                summary.stopped = True
                break
            upstream_dirs[stage.name] = stage_dir
            summary.completed.append(stage.name)
        return summary

    def _reset_from(self, state: StateStore, name: str) -> None:
        start = [s.name for s in self.stages].index(name)
        for index in range(start, len(self.stages)):
            self._reset_stage(state, index, self.stages[index])

    def _reset_stage(self, state: StateStore, index: int, stage: Stage) -> None:
        moved = self.layout.supersede(self.layout.stage_dir(index, stage.name))
        state.reset_stage(stage.name)
        if moved is not None:
            log.info("Stage %s: previous results moved to %s", stage.name, moved)

    # ------------------------------------------------------------------ one stage
    def _run_stage(
        self,
        state: StateStore,
        stage: Stage,
        ctx: StageContext,
        fp: str,
        should_stop: Callable[[], bool],
    ) -> bool:
        log.info("Stage %s: starting", stage.name)
        state.set_stage(stage.name, StageStatus.RUNNING, fingerprint=fp)
        ctx.stage_dir.mkdir(parents=True, exist_ok=True)
        tasks = stage.plan(ctx)
        ids = [t.task_id for t in tasks]
        if len(set(ids)) != len(ids):
            raise ProjectError(f"Stage {stage.name!r} planned duplicate task ids")
        known = {r.task_id for r in state.get_tasks(stage.name)}
        if known and known != set(ids):
            raise ProjectError(
                f"Stage {stage.name!r} planned different tasks than recorded; "
                f"run with --rerun-from {stage.name}"
            )
        state.upsert_tasks(stage.name, ids)
        state.requeue_failed(stage.name)
        self._reconcile_running(state, stage, ctx)
        specs = {t.task_id: t for t in tasks}
        max_attempts = self.config.execution.max_attempts

        while True:
            runnable = [
                r
                for r in state.get_tasks(stage.name)
                if r.status == TaskStatus.PENDING and r.attempts < max_attempts
            ]
            if not runnable:
                break
            jobs = [self._prepare_job(stage, ctx, specs[r.task_id]) for r in runnable]
            finished = self.executor.run(
                jobs,
                on_start=lambda job, pid: state.set_task(
                    stage.name, job.key, TaskStatus.RUNNING, pid=pid, attempts_delta=1
                ),
                on_finish=lambda outcome: self._record_outcome(
                    state, stage, ctx, outcome, max_attempts
                ),
                should_stop=should_stop,
            )
            if not finished:
                state.set_stage(stage.name, StageStatus.PENDING, message="stopped by user")
                log.warning("Stage %s: stopped; run the same command again to resume", stage.name)
                return False

        records = state.get_tasks(stage.name)
        failed = [r for r in records if r.status != TaskStatus.DONE]
        if records and len(failed) / len(records) > stage.max_failed_fraction:
            message = self._failure_message(stage, ctx, failed, len(records))
            state.set_stage(stage.name, StageStatus.FAILED, message=message)
            raise StageFailedError(message)
        if failed:
            log.warning(
                "Stage %s: %d of %d tasks failed (tolerated)", stage.name, len(failed), len(records)
            )
        results = self._collect_results(stage, ctx, records)
        try:
            stage.finalize(ctx, results)
        except ShmdError as exc:
            message = f"Stage '{stage.name}' could not finalize its results: {exc}"
            state.set_stage(stage.name, StageStatus.FAILED, message=message)
            raise StageFailedError(message) from exc
        state.set_stage(
            stage.name,
            StageStatus.DONE,
            message=f"{len(results)}/{len(records)} tasks succeeded",
        )
        log.info("Stage %s: done (%d/%d tasks)", stage.name, len(results), len(records))
        return True

    def _prepare_job(self, stage: Stage, ctx: StageContext, task: TaskSpec) -> Job:
        task_dir = self.layout.task_dir(ctx.stage_dir, task.task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / RESULT_FILE).unlink(missing_ok=True)
        task_file = task_dir / TASK_FILE
        write_task_file(
            task_file,
            pipeline_ref=self.pipeline_ref,
            config_snapshot=self.layout.config_snapshot,
            ctx=ctx,
            stage_name=stage.name,
            task=task,
            task_dir=task_dir,
        )
        return Job(key=task.task_id, task_file=task_file, log_file=task_dir / "worker.log")

    def _record_outcome(
        self,
        state: StateStore,
        stage: Stage,
        ctx: StageContext,
        outcome: JobOutcome,
        max_attempts: int,
    ) -> None:
        task_id = outcome.job.key
        result = read_result(self.layout.task_dir(ctx.stage_dir, task_id))
        if result is not None and result.get("status") == "ok":
            state.set_task(stage.name, task_id, TaskStatus.DONE)
            return
        if outcome.returncode is None:
            state.set_task(
                stage.name, task_id, TaskStatus.PENDING,
                message="interrupted by stop request", attempts_delta=-1,
            )
            return
        if result is None:
            retryable = True
            message = (
                f"worker exited with code {outcome.returncode} without writing a result "
                f"(crash, out-of-memory or kill); see {outcome.job.log_file}"
            )
        else:
            retryable = bool(result.get("retryable", False))
            message = f"{result.get('error_type')}: {result.get('message')}"
        attempts = state.get_task(stage.name, task_id).attempts
        if retryable and attempts < max_attempts:
            log.warning("Task %s/%s failed, will retry: %s", stage.name, task_id, message)
            state.set_task(stage.name, task_id, TaskStatus.PENDING, message=f"retrying after: {message}")
        else:
            log.error("Task %s/%s failed: %s", stage.name, task_id, message)
            state.set_task(stage.name, task_id, TaskStatus.FAILED, message=message)

    def _reconcile_running(self, state: StateStore, stage: Stage, ctx: StageContext) -> None:
        """Tasks left RUNNING by a controller that died: finish or requeue them."""
        for record in state.get_tasks(stage.name):
            if record.status != TaskStatus.RUNNING:
                continue
            task_dir = self.layout.task_dir(ctx.stage_dir, record.task_id)
            if record.pid is not None and is_worker_process(record.pid, task_dir / TASK_FILE):
                log.warning(
                    "Terminating orphaned worker %d of %s/%s", record.pid, stage.name, record.task_id
                )
                terminate_process_group(record.pid, grace=self.config.execution.stop_grace_period_s)
            result = read_result(task_dir)
            if result is not None and result.get("status") == "ok":
                state.set_task(stage.name, record.task_id, TaskStatus.DONE)
            else:
                state.set_task(
                    stage.name, record.task_id, TaskStatus.PENDING,
                    message="previous controller stopped while this task was running",
                    attempts_delta=-1,
                )

    def _collect_results(
        self, stage: Stage, ctx: StageContext, records: list[TaskRecord]
    ) -> dict[str, Mapping[str, Any]]:
        results: dict[str, Mapping[str, Any]] = {}
        for record in records:
            if record.status != TaskStatus.DONE:
                continue
            result = read_result(self.layout.task_dir(ctx.stage_dir, record.task_id))
            if result is None or result.get("status") != "ok":
                raise ProjectError(
                    f"Result of completed task {stage.name}/{record.task_id} is missing or "
                    f"corrupt; run with --rerun-from {stage.name}"
                )
            results[record.task_id] = result["result"]
        return results

    def _failure_message(
        self, stage: Stage, ctx: StageContext, failed: list[TaskRecord], total: int
    ) -> str:
        lines = [
            f"Stage '{stage.name}' failed: {len(failed)} of {total} tasks did not succeed "
            f"(tolerated fraction: {stage.max_failed_fraction:.0%})."
        ]
        for record in failed[:5]:
            log_path = self.layout.task_dir(ctx.stage_dir, record.task_id) / "worker.log"
            lines.append(f"  - {record.task_id}: {record.message} (log: {log_path})")
        if len(failed) > 5:
            lines.append(f"  ... and {len(failed) - 5} more; see `gqteashmd status`.")
        lines.append("Fix the cause and run the same command again; failed tasks will be retried.")
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_orchestrator.py -v`
Expected: 11 passed (each spawns worker subprocesses, so the run takes roughly 30–60 s)

- [ ] **Step 5: Run the whole suite and ruff**

Run: `pytest -m "not slow" && ruff format src tests && ruff check --fix src tests`
Expected: all tests pass; `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/gqteashmd/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add resumable workflow orchestrator"
```

---

### Task 11: Command-line interface and logging

**Files:**
- Create: `src/gqteashmd/cli.py`, `src/gqteashmd/__main__.py`, `src/gqteashmd/logging_setup.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `load_config` (Task 3); `Orchestrator` (Task 10); `ProjectLayout` (Task 4); `locked_by` (Task 4); `StateStore`, `TaskStatus` (Task 5); `DEFAULT_PIPELINE`, `import_pipeline` (Task 6); `ShmdError`, `ConfigError` (Task 2).
- Produces:
  - `main(argv: list[str] | None = None) -> int`.
  - `EXIT_STOPPED = 5`.
  - `TEMPLATE` (the commented input file).
  - `configure_logging(log_file: Path | None, *, verbose: bool = False) -> None`.
  - Commands: `gqteashmd init PATH [--force]`, `gqteashmd validate INPUT`, `gqteashmd run INPUT [--rerun-from STAGE]`, `gqteashmd status INPUT`, `gqteashmd stop INPUT`. `run` and `status` also take a hidden `--pipeline REF` used in tests.

- [ ] **Step 1: Write the failing tests** `tests/test_cli.py`

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gqteashmd.cli'`

- [ ] **Step 3: Implement `src/gqteashmd/logging_setup.py`**

```python
"""Logging: INFO to the console, DEBUG to the project log file."""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(log_file: Path | None, *, verbose: bool = False) -> None:
    logger = logging.getLogger("gqteashmd")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(_FORMAT, "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
```

- [ ] **Step 4: Implement `src/gqteashmd/cli.py`**

```python
"""Command-line interface: gqteashmd init | validate | run | status | stop."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from pathlib import Path

from gqteashmd import __version__
from gqteashmd.config.loader import load_config
from gqteashmd.errors import ConfigError, ShmdError
from gqteashmd.logging_setup import configure_logging
from gqteashmd.orchestrator import Orchestrator
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.project.lock import locked_by
from gqteashmd.project.state import StateStore, TaskStatus
from gqteashmd.stages.pipeline import DEFAULT_PIPELINE, import_pipeline

EXIT_STOPPED = 5

TEMPLATE = """\
# gqteashmd input file. Paths are relative to this file.
schema_version = 1

[project]
name = "my_molecule"          # letters, digits, _ . -
directory = "run"             # results go here; use a local Linux disk (not OneDrive, not /mnt/c)
seed = 12345                  # master random seed for the whole workflow

[molecule]
geometry = "molecule.xyz"     # XYZ file, Angstrom
charge = 0
multiplicity = 1

[resources]
max_parallel_tasks = 1        # tasks running at the same time
threads_per_task = 1          # OpenMP/BLAS threads per task
memory_mb_per_task = 2000     # passed to PySCF as max_memory (MB)

[execution]
max_attempts = 2              # attempts per task for retryable failures
stop_grace_period_s = 30.0    # time running tasks get to stop after `gqteashmd stop`
"""


def _cmd_init(args: argparse.Namespace) -> int:
    path: Path = args.path
    if path.exists() and not args.force:
        raise ConfigError(f"{path} already exists; use --force to overwrite it.")
    path.write_text(TEMPLATE, encoding="utf-8")
    print(f"Wrote template input file {path}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    loaded = load_config(args.input)
    cfg = loaded.config
    if not cfg.molecule.geometry.exists():
        raise ConfigError(f"Geometry file not found: {cfg.molecule.geometry}")
    print(f"Input file {loaded.source} is valid.")
    print(f"  project directory: {cfg.project.directory}")
    print(f"  geometry:          {cfg.molecule.geometry}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    loaded = load_config(args.input)
    orchestrator = Orchestrator(loaded, pipeline_ref=args.pipeline, rerun_from=args.rerun_from)
    orchestrator.layout.ensure()
    configure_logging(orchestrator.layout.logs_dir / "gqteashmd.log")
    logger = logging.getLogger("gqteashmd.cli")
    stop_requested = threading.Event()

    def _on_signal(signum: int, _frame: object) -> None:
        if not stop_requested.is_set():
            logger.warning("Stop requested (signal %d): stopping running tasks...", signum)
        stop_requested.set()

    previous = {sig: signal.signal(sig, _on_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        summary = orchestrator.run(should_stop=stop_requested.is_set)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if summary.stopped:
        print("Stopped. Run the same command again to resume.")
        return EXIT_STOPPED
    print(
        f"Workflow complete: {len(summary.completed)} stage(s) run, "
        f"{len(summary.up_to_date)} already up to date, {len(summary.skipped)} skipped."
    )
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    cfg = load_config(args.input).config
    layout = ProjectLayout(cfg.project.directory)
    if not layout.state_db.exists():
        print(f"Project '{cfg.project.name}' has not been started (no state in {layout.root}).")
        return 0
    pid = locked_by(layout.lock_file)
    print(f"Project: {cfg.project.name}  ({layout.root})")
    print(f"Controller: {f'running (pid {pid})' if pid else 'not running'}")
    print(
        f"{'stage':<16}{'status':<13}{'done':>6}{'failed':>8}"
        f"{'pending':>9}{'running':>9}  message"
    )
    with StateStore(layout.state_db) as state:
        for stage in import_pipeline(args.pipeline)(cfg):
            record = state.get_stage(stage.name)
            counts = state.task_counts(stage.name)
            status = record.status.value if record else "not started"
            message = (record.message or "").splitlines()[0] if record and record.message else ""
            print(
                f"{stage.name:<16}{status:<13}{counts[TaskStatus.DONE]:>6}"
                f"{counts[TaskStatus.FAILED]:>8}{counts[TaskStatus.PENDING]:>9}"
                f"{counts[TaskStatus.RUNNING]:>9}  {message}"
            )
    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    cfg = load_config(args.input).config
    pid = locked_by(ProjectLayout(cfg.project.directory).lock_file)
    if pid is None:
        print("No running controller for this project.")
        return 0
    os.kill(pid, signal.SIGTERM)
    print(f"Stop requested for controller pid {pid}; running tasks are being stopped.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gqteashmd", description="Surface-hopping MD workflow")
    parser.add_argument("--version", action="version", version=f"gqteashmd {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="write a commented template input file")
    p_init.add_argument("path", type=Path)
    p_init.add_argument("--force", action="store_true", help="overwrite an existing file")
    p_init.set_defaults(func=_cmd_init)

    p_validate = sub.add_parser("validate", help="check an input file without running anything")
    p_validate.add_argument("input", type=Path)
    p_validate.set_defaults(func=_cmd_validate)

    p_run = sub.add_parser("run", help="run (or resume) the workflow")
    p_run.add_argument("input", type=Path)
    p_run.add_argument("--rerun-from", metavar="STAGE", help="discard results from STAGE onward")
    p_run.add_argument("--pipeline", default=DEFAULT_PIPELINE, help=argparse.SUPPRESS)
    p_run.set_defaults(func=_cmd_run)

    p_status = sub.add_parser("status", help="show progress of a project")
    p_status.add_argument("input", type=Path)
    p_status.add_argument("--pipeline", default=DEFAULT_PIPELINE, help=argparse.SUPPRESS)
    p_status.set_defaults(func=_cmd_status)

    p_stop = sub.add_parser("stop", help="ask the running controller to stop gracefully")
    p_stop.add_argument("input", type=Path)
    p_stop.set_defaults(func=_cmd_stop)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ShmdError as exc:
        print(f"gqteashmd: error: {exc}", file=sys.stderr)
        return exc.exit_code
```

- [ ] **Step 5: Implement `src/gqteashmd/__main__.py`**

```python
from gqteashmd.cli import main

raise SystemExit(main())
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: 10 passed (including the `slow` kill-and-resume test)

- [ ] **Step 7: Try the real pipeline by hand**

```bash
mkdir -p ~/gqteashmd-try && cd ~/gqteashmd-try
gqteashmd init input.toml
printf '3\nwater\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n' > molecule.xyz
gqteashmd validate input.toml && gqteashmd run input.toml && gqteashmd status input.toml
```

Expected: `Workflow complete: 1 stage(s) run, ...`, and `status` shows `prepare  done`.

- [ ] **Step 8: Commit**

```bash
cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD
git add src/gqteashmd/cli.py src/gqteashmd/__main__.py src/gqteashmd/logging_setup.py tests/test_cli.py
git commit -m "feat: add gqteashmd command-line interface"
```

---

### Task 12: CI, pre-commit, developer documentation, GitHub

**Files:**
- Create: `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, `docs/development.md`

**Interfaces:**
- Consumes: the whole test suite.
- Produces: CI on every push/PR (Oracle Linux 9 container), local pre-commit hooks, and the developer guide.

- [ ] **Step 1: Create `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

Run: `pre-commit install && pre-commit run --all-files`
Expected: `ruff....Passed` and `ruff-format....Passed`

- [ ] **Step 2: Create `.github/workflows/ci.yml`**

```yaml
name: CI
on:
  push:
  pull_request:
  schedule:
    - cron: "0 3 * * *"   # nightly: includes slow tests

jobs:
  test:
    name: tests (Oracle Linux 9)
    runs-on: ubuntu-latest
    container: oraclelinux:9
    defaults:
      run:
        shell: bash -el {0}
    steps:
      - name: Install base tools
        run: dnf install -y git tar bzip2 findutils procps-ng
      - uses: actions/checkout@v4
      - uses: mamba-org/setup-micromamba@v2
        with:
          environment-file: environment.yml
          cache-environment: true
      - name: Install gqteashmd
        run: pip install -e . --no-deps
      - name: Lint
        run: ruff check . && ruff format --check .
      - name: Fast tests
        run: pytest -m "not slow"
      - name: Slow tests
        run: pytest -m slow
```

(Check the latest major versions of `actions/checkout` and `mamba-org/setup-micromamba` when this workflow first runs. Bump them if GitHub reports deprecations.)

- [ ] **Step 3: Create `docs/development.md`**

````markdown
# Developer guide

## Where things live
- Source: `C:\Users\ajc\OneDrive\gqteaSHMD` = `/mnt/c/Users/ajc/OneDrive/gqteaSHMD` in WSL.
- Run everything (tests, gqteashmd) **inside WSL (Oracle Linux 9.5)** or on the Oracle Linux workstation.
  gqteashmd is Linux-only (PySCF, fcntl, /proc).
- Never put conda environments, gqteashmd project runs or large outputs in the OneDrive folder.
  Use `~/gqteashmd-runs/` (WSL) or a local data disk on the workstation.

## One-time setup (WSL)
```bash
# micromamba: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html
cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD
micromamba create -y -f environment.yml
micromamba activate gqteashmd-dev
pip install -e . --no-deps
pre-commit install
git config core.autocrlf false
git config core.filemode false
```
Use **only Git inside WSL** (VS Code: "WSL: Open Folder in WSL"). Mixing Windows Git and WSL Git
on the same OneDrive folder causes line-ending and permission noise.

## Daily workflow
```bash
git switch -c feat/<topic>
pytest -m "not slow"          # fast suite
pytest -m slow                # kill/resume tests
ruff check . && ruff format .
git push -u origin feat/<topic> && gh pr create
```
Commit messages: conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
`main` is protected: PR + green CI + one review.

## OneDrive caveats
- OneDrive also syncs `.git/`. Push often; GitHub is the source of truth.
- Set the folder to "Always keep on this device".
- If OneDrive reports a sync conflict inside `.git/`, do not merge the copies. Re-clone from GitHub.

## Running gqteashmd unattended (workstation)
```bash
cd ~/gqteashmd-runs/my_molecule
nohup gqteashmd run input.toml > gqteashmd.out 2>&1 &     # or inside tmux
gqteashmd status input.toml
gqteashmd stop input.toml                              # graceful; `gqteashmd run` resumes
```

## Clean-room rule
Do not copy, translate or adapt code from SHARC 4.1. Implement from the literature.
SHARC may only be executed as a black box by scripts in `tools/sharc_compare/` (comparison
tests, marker `sharc`), never imported by `gqteashmd`.
````

- [ ] **Step 4: Run the full suite one last time**

Run: `pytest && ruff check . && ruff format --check .`
Expected: all tests pass (fast + slow); `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml .pre-commit-config.yaml docs/development.md
git commit -m "ci: add Oracle Linux 9 CI, pre-commit hooks and developer guide"
```

- [ ] **Step 6: Push the feature branch (the repository `AdemirCamargo57/gqteaSHMD` already exists)**

```bash
git push -u origin feat/p1-foundation
```

If WSL Git has no GitHub credentials, push from Windows instead (Git Credential Manager): `git -C C:\Users\ajc\OneDrive\gqteaSHMD push -u origin feat/p1-foundation`. Pushing runs no hooks, so using Windows Git for this one command is safe.

Expected: the branch appears on GitHub, and the CI workflow runs on the push and is green.

- [ ] **Step 7: Owner actions on github.com (not done by the agent)**

  1. Open a PR from `feat/p1-foundation` to `main`, review it and merge it.
  2. Settings → Branches → protect `main`: require a PR, require the `tests (Oracle Linux 9)` check, and require one approval.
  3. Decide whether the repository stays public (open item O13).

---

## Self-review notes (completed while writing this plan)

- **Spec coverage (master plan §2–§8 within P1 scope):**
  - Error taxonomy and exit codes: Tasks 2 and 11.
  - Strict TOML input with snapshot: Task 3.
  - Project layout, lock and superseded results: Task 4.
  - SQLite state: Task 5.
  - Stage contract: Task 6.
  - Physics preflight (`prepare`): Task 7.
  - Subprocess workers: Task 8.
  - Concurrency, thread variables and graceful stop: Task 9.
  - Fingerprints, stale detection, `--rerun-from`, retries, tolerances, orphan reconciliation and resume: Task 10.
  - CLI `init/validate/run/status/stop`: Task 11.
  - CI in Oracle Linux 9, Git/OneDrive setup and the clean-room rule: Tasks 1 and 12.
  - Deferred to P2 by design: per-stage resources, `pause_after`, environment preflight, provenance, mypy.
- **Type consistency:**
  - `StateStore.set_task(..., attempts_delta=...)` is used identically in Tasks 5 and 10.
  - `Job.key` is the task ID throughout.
  - `read_result` / `RESULT_FILE` / `TASK_FILE` come from `gqteashmd.worker` in Tasks 8 and 10.
  - `StageContext(config, layout, stage_dir, upstream)` is constructed in the same field order in Tasks 6, 7, 8 and 10.
