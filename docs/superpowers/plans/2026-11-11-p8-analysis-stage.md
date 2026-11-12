# P8 `analysis` Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `analysis` stage (`07_analysis/`): classical and quantum populations with 95 % intervals, a delayed mono-exponential fit with bootstrap uncertainties, hop and energy statistics, from the per-trajectory `output.h5` files; the `[analysis]` input; the orchestrator rule that recomputes the dependants of a reopened stage; the M4 study.

**Architecture:** A pure `gqteashmd/analysis/` package (NumPy; SciPy imported inside the fit function) does the arithmetic on plain arrays; `trajectory/store.py:read_output_h5` and `excite/artifacts.py:read_trajectories_json` read the upstream artifacts; `analysis/artifacts.py` writes the outputs (h5py inside functions); `stages/analysis.py` is a thin one-task stage. The orchestrator resets the later dependants of a stage before reopening its tasks.

**Tech Stack:** Python 3.12, NumPy, SciPy (`optimize.least_squares`), h5py, pydantic 2, pytest.

**Spec:** `docs/superpowers/specs/2026-11-11-p8-analysis-stage-design.md` (owner-approved 2026-11-11). The spec is binding; the "Plan rulings" below settle what it leaves open or where the code forces a detail.

## Global Constraints

- Run everything in WSL: `wsl -d OracleLinux_9_5 -- bash -lc 'cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && conda activate gqteashmd-dev && <command>'`. Git from WSL only (pre-commit hooks, ruff 0.17.4); never `--no-verify`. A zero-byte `.git/index.lock` with no git process in WSL may be removed (Windows-side editor polling).
- One commit per task, conventional subject (`feat:`, `fix:`, `test:`, `docs:`), ending with the `Co-Authored-By:` trailer of the model that wrote it.
- Lint: `ruff format . && ruff check .` (line length 100; rules E, F, W, I, B, UP, SIM). Types: `mypy` clean on `src/`.
- Tests run with `-W error`. Fast tier: `pytest -m "not slow and not pyscf" -W error`.
- `import gqteashmd.cli` must not import pyscf, geometric, h5py **or scipy** (the registry imports `stages/analysis.py`; SciPy and h5py are imported inside functions).
- Dependency rule for `gqteashmd/analysis/`: it may import NumPy, `gqteashmd.core` and itself (tests may also import `gqteashmd.dynamics.ensemble.wilson_interval` as a reference); never `stages/`, the orchestrator, `config/`, `electronic/`, `trajectory/` or PySCF. Only `analysis/artifacts.py` imports h5py, inside functions; only `analysis/kinetics.py` imports SciPy, inside functions.
- `src/gqteashmd/dynamics/` and `src/gqteashmd/trajectory/runner.py` are not modified; the golden test `tests/dynamics/test_golden.py` stays green. `stages/dynamics.py` changes only by reading `trajectories.json` through the shared reader (no behaviour change).
- Seeds (D12): the bootstrap draws from `task_rng(project.seed, "analysis", 0)` only.
- Artifacts: every HDF5 dataset has a string attribute `units`; file attributes `schema_version` = 1 and `gqteashmd_version`; JSON never contains NaN or infinity (`null`); every file is written atomically (`util.fileio.write_json_atomic`, `write_text_atomic`, `write_bytes_atomic`; HDF5 via a temporary file and `os.replace`).
- Clean room: SHARC is not read, run or imported in P8.
- CLAUDE.md and USER_MANUAL.md are changed only in Task 12.
- Keep project runs and large outputs out of `/mnt/c` (pytest `tmp_path` does this; the M4 study runs under `~/p8_m4`).

## Plan rulings

- **PR1** Output writers live in `gqteashmd/analysis/artifacts.py` with h5py imported inside the functions (the `vib/artifacts.py` pattern); the computational modules import no h5py (the spec's "no h5py" applies to them).
- **PR2** A frustrated hop's target state is not recorded in `output.h5` (`hop_from`/`hop_to` are -1 on a frustrated record; the core records only accepted hops). Frustrated hops are therefore counted **per originating state** (the active state of the record), and `hops.csv` leaves `to` empty for them. The spec's "frustrated hops per pair" becomes "per originating state".
- **PR3** The energy jump of a trajectory is taken over its records with `grid_step <= n_total` (the current `tmax_fs`), before the horizon is known, because `exclude_energy_jumps` acts on the inclusion that defines the horizon. The drift statistics use the records up to the horizon.
- **PR4** `trajectories.json` is read by `excite/artifacts.py:read_trajectories_json`, used by `dynamics.plan` (same `ProjectError` message as now) and by `analysis`.
- **PR5** `n_total = round(tmax_fs / dt_fs)` from the current config; an `output.h5` is consistent when its `state_labels` equal the first file's, `output_every` equals `[dynamics] output_every`, and `|dt_fs - [dynamics] dt_fs| <= 1e-9 * dt_fs` (the runner stores `dt_au * AU_TIME_IN_FS`, which is not bit-equal to the input).
- **PR6** `run_task` writes `analysis.json`, `populations.h5`, `populations.csv` and `hops.csv` in its task directory; `finalize` adds the finalize-only warning (`warn_excluded_fraction`), writes `07_analysis/analysis.json` and `report.md`, copies the other files, then raises the stops recorded by `run_task` (`stage_failed`). `run_task` never raises for a physics or statistics stop, only for unreadable or inconsistent inputs (`TaskError`, non-retryable).
- **PR7** A new test pipeline `gqteashmd.testing.pipelines:canned_analysis` (= `canned_dynamics` + the real `AnalysisStage`); `canned_dynamics` is unchanged, so the P7b pipeline tests keep their stage numbers.
- **PR8** The fake test pipeline gains an optional fake `analysis` stage (behaviour keys `analysis` and `analysis_disabled`); without them it is unchanged (`ALL` in `tests/test_orchestrator.py` stays valid).
- **PR9** The default `energy_jump_eV` is decided by Task 1 under the spec's rule (0.1 eV unless the measured largest per-step `|dE_tot|` exceeds 0.01 eV, then ten times that value rounded up to two significant digits). The controller records the value in the ledger and passes it to Task 2.

## File structure

- Create `tools/validation/energy_jump_probe.py` (Task 1): scans `output.h5` files, prints and writes per-step jump numbers.
- Modify `src/gqteashmd/config/schema.py`, `src/gqteashmd/cli.py` (Task 2): `AnalysisSection`, `ShmdConfig.analysis`, the `init` template block.
- Modify `src/gqteashmd/trajectory/store.py`, `src/gqteashmd/excite/artifacts.py`, `src/gqteashmd/stages/dynamics.py` (Task 3): `TrajectoryData`, `read_output_h5`, `SelectedTrajectory`, `read_trajectories_json`.
- Create `src/gqteashmd/analysis/__init__.py`, `grid.py`, `populations.py` (Task 4); `kinetics.py` (Task 5); `hops.py`, `energy.py` (Task 6); `artifacts.py`, `report.py` (Task 7).
- Modify `src/gqteashmd/orchestrator.py`, `src/gqteashmd/testing/fake_stages.py` (Task 8).
- Create `src/gqteashmd/stages/analysis.py`; modify `src/gqteashmd/stages/registry.py`, `src/gqteashmd/testing/pipelines.py` (Task 9).
- Modify `tests/test_e2e_dynamics.py` (Task 10).
- Tests: `tests/config/test_analysis_section.py`, `tests/trajectory/test_read_output.py`, `tests/excite/test_trajectories_json.py`, `tests/analysis/test_grid.py`, `test_populations.py`, `test_kinetics.py`, `test_hops.py`, `test_energy.py`, `test_artifacts.py`, `test_report.py`, `tests/stages/test_analysis.py`, `tests/stages/test_analysis_pipeline.py`, additions to `tests/test_orchestrator.py`, `tests/stages/test_base_pipeline.py`.

---

### Task 1: Measure the per-step energy changes and fix the `energy_jump_eV` default

**Files:**
- Create: `tools/validation/energy_jump_probe.py`
- Report: `.superpowers/sdd/2026-11-11-p8-analysis-stage/task-1-report.md`

**Interfaces:**
- Produces: the measured number and the decided default (PR9), in the report; nothing in `src/`.

- [ ] **Step 1: Write the probe script**

```python
"""Largest |E_tot(r+1) - E_tot(r)| over consecutive records of output.h5 files (P8 Task 1).

Run: python tools/validation/energy_jump_probe.py ROOT [--json OUT]
Scans ROOT recursively for output.h5, prints one line per file and the overall maximum.
Not collected by pytest; nothing is asserted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

HARTREE_IN_EV = 27.211386245988


def scan(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("output.h5")):
        with h5py.File(path, "r") as f:
            e = np.asarray(f["e_total"][()], dtype=float)
            hops = int(np.count_nonzero(np.asarray(f["hop_to"][()]) >= 0))
            outcome = str(f.attrs.get("outcome", ""))
            output_every = int(f.attrs.get("output_every", 1))
        jumps = np.abs(np.diff(e)) * HARTREE_IN_EV
        rows.append(
            {
                "file": str(path),
                "records": int(e.size),
                "output_every": output_every,
                "outcome": outcome,
                "hops": hops,
                "max_jump_eV": float(jumps.max()) if jumps.size else 0.0,
                "p99_jump_eV": float(np.percentile(jumps, 99)) if jumps.size else 0.0,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    rows = scan(args.root)
    for r in rows:
        print(
            f"{r['max_jump_eV']:.3e} eV max, {r['p99_jump_eV']:.3e} eV p99, "
            f"{r['records']} records, {r['hops']} hops, {r['outcome']}: {r['file']}"
        )
    overall = max((r["max_jump_eV"] for r in rows), default=0.0)
    print(f"overall max jump: {overall:.6e} eV over {len(rows)} files")
    if args.json:
        args.json.write_text(json.dumps({"files": rows, "overall_max_jump_eV": overall}, indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Produce the P7b runs and scan them**

Run (all markers of the P7b PySCF validation file, about 10-15 min; the trajectories write `output.h5` into pytest's `tmp_path`, kept under the given base temp):

```bash
rm -rf ~/p8_probe && mkdir -p ~/p8_probe
pytest tests/validation/test_pyscf_dynamics.py -q -W error -p no:cacheprovider --basetemp ~/p8_probe/tmp
python tools/validation/energy_jump_probe.py ~/p8_probe/tmp --json ~/p8_probe/jumps.json
```

Expected: the pytest run ends as in P7b (passed tests plus the two strict xfails); the probe prints one line per `output.h5` and the overall maximum. The ethylene step-24 solution change is not in these runs (it was in the SHARC comparison), so no file is excluded; if a file has a jump above 0.1 eV, report it with its file and do not exclude it silently.

- [ ] **Step 3: Apply the rule and write the report**

Rule (fixed in the spec, PR9): `v` = the overall maximum in eV. If `v <= 0.01`, the default is `0.1`. Otherwise the default is `10 * v` rounded **up** to two significant digits. Write `task-1-report.md` with the per-file table, `v`, the decided default, and the run times.

- [ ] **Step 4: Commit**

```bash
ruff format tools/validation/energy_jump_probe.py && ruff check tools/validation/energy_jump_probe.py
git add tools/validation/energy_jump_probe.py
git commit -m "test: probe the per-step energy changes of the P7b runs (P8 energy-jump default)"
```

---

### Task 2: The `[analysis]` input section

**Files:**
- Modify: `src/gqteashmd/config/schema.py` (after `DynamicsStageSection`, and `ShmdConfig`)
- Modify: `src/gqteashmd/cli.py` (`TEMPLATE`, after the commented `[dynamics]` block)
- Test: `tests/config/test_analysis_section.py`

**Interfaces:**
- Produces: `AnalysisSection` (fields below), `ShmdConfig.analysis: AnalysisSection` (always present, default factory). `ENERGY_JUMP_DEFAULT_EV` constant in `schema.py` with the Task 1 value (the controller passes it; 0.1 unless Task 1 decided otherwise).

- [ ] **Step 1: Write the failing tests**

```python
"""[analysis] input section (P8 spec section 10)."""

import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.config.schema import ENERGY_JUMP_DEFAULT_EV
from gqteashmd.errors import ConfigError

ELECTRONIC = '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 3\n'


def _load(tmp_path, write_input, extra):
    return load_config(write_input(tmp_path, extra=ELECTRONIC + extra)).config


def test_defaults_without_a_table(tmp_path, write_input):
    a = _load(tmp_path, write_input, "").analysis
    assert a.enabled is True and a.fit_states is None and a.fit_population == "classical"
    assert a.n_bootstrap == 1000 and a.energy_jump_eV == ENERGY_JUMP_DEFAULT_EV
    assert a.exclude_energy_jumps is False and a.warn_excluded_fraction == 0.2


def test_fit_states_are_upper_cased_excited_singlets(tmp_path, write_input):
    a = _load(tmp_path, write_input, '[analysis]\nfit_states = ["s1", "S2"]\n').analysis
    assert a.fit_states == ["S1", "S2"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('fit_states = ["S0"]', "S0"),
        ('fit_states = ["T1"]', "triplet"),
        ('fit_states = ["S1", "S1"]', "duplicate"),
        ("fit_states = []", "empty"),
        ('fit_states = ["S3"]', "does not exist"),
        ("n_bootstrap = 99", "n_bootstrap"),
        ("energy_jump_eV = 0.0", "energy_jump_eV"),
        ("energy_jump_eV = inf", "energy_jump_eV"),
        ("warn_excluded_fraction = 1.5", "warn_excluded_fraction"),
        ('fit_population = "both"', "fit_population"),
        ("unknown_key = 1", "unknown_key"),
    ],
)
def test_invalid_values_are_config_errors(tmp_path, write_input, text, message):
    with pytest.raises(ConfigError, match=message):
        _load(tmp_path, write_input, f"[analysis]\n{text}\n")


def test_all_violations_of_the_cross_field_rule_are_reported(tmp_path, write_input):
    with pytest.raises(ConfigError) as info:
        _load(tmp_path, write_input, '[analysis]\nfit_states = ["S3", "S4"]\n')
    assert "S3" in str(info.value) and "S4" in str(info.value)


def test_init_template_has_a_commented_analysis_block(tmp_path):
    from gqteashmd.cli import TEMPLATE

    assert "# [analysis]" in TEMPLATE and "# fit_states" in TEMPLATE
```

- [ ] **Step 2: Run them to see them fail**

Run: `pytest tests/config/test_analysis_section.py -q -W error`
Expected: FAIL (`ImportError: cannot import name 'ENERGY_JUMP_DEFAULT_EV'`).

- [ ] **Step 3: Implement**

In `schema.py`, after `DynamicsStageSection`:

```python
ENERGY_JUMP_DEFAULT_EV = 0.1  # P8 Task 1 measurement (plan ruling PR9)


class AnalysisSection(_Section):
    """[analysis] of the workflow (P8). Meanings: USER_MANUAL.md section 4."""

    enabled: bool = True
    fit_states: list[str] | None = None
    fit_population: Literal["classical", "quantum"] = "classical"
    n_bootstrap: int = Field(default=1000, ge=100)
    energy_jump_eV: float = Field(default=ENERGY_JUMP_DEFAULT_EV, gt=0, allow_inf_nan=False)
    exclude_energy_jumps: bool = False
    warn_excluded_fraction: float = Field(default=0.2, ge=0, le=1, allow_inf_nan=False)

    @field_validator("fit_states")
    @classmethod
    def _fit_states_are_excited_singlets(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        from gqteashmd.core.states import StateLabel

        labels: list[StateLabel] = []
        for text in value:
            try:
                label = StateLabel.parse(text)
            except ValueError as exc:
                raise ValueError(f"invalid state label {text!r}: {exc}") from exc
            if label.multiplicity != 1:
                raise ValueError(f"{label} is a triplet; analysis.fit_states takes singlets")
            if label.index == 0:
                raise ValueError("S0 cannot be in analysis.fit_states (fit excited states)")
            if label in labels:
                raise ValueError(f"duplicate state {label} in analysis.fit_states")
            labels.append(label)
        if not labels:
            raise ValueError("analysis.fit_states must not be empty (omit it to use all)")
        return [str(label) for label in labels]
```

(`StateLabel.parse` accepts lower case; check by running the test, and upper-case the text before parsing if it does not.)

In `ShmdConfig` add `analysis: AnalysisSection = Field(default_factory=AnalysisSection)` after `dynamics`, and in `_consistent`, before `if problems:`:

```python
        if self.electronic is not None and self.analysis.fit_states is not None:
            from gqteashmd.core.states import StateLabel

            singlets = self.electronic.singlets
            for text in self.analysis.fit_states:
                if StateLabel.parse(text).index >= singlets:
                    problems.append(
                        f"analysis.fit_states: {text} does not exist with [electronic] "
                        f"singlets = {singlets} (S0..S{singlets - 1})"
                    )
```

In `cli.py`, change the end of `TEMPLATE` from `# energy_drift_limit_eV = 0.5"""` to:

```text
# energy_drift_limit_eV = 0.5

# [analysis]                         # optional; runs after dynamics by default
# fit_states = ["S1", "S2"]          # fitted population (default: every excited state)
# fit_population = "classical"       # or "quantum"
# n_bootstrap = 1000
# energy_jump_eV = 0.1               # flag trajectories with a larger E_tot jump
# exclude_energy_jumps = false"""
```

(Write the `energy_jump_eV` comment with the Task 1 value.)

- [ ] **Step 4: Run the tests and the config suite**

Run: `pytest tests/config tests/test_cli.py -q -W error`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gqteashmd/config/schema.py src/gqteashmd/cli.py tests/config/test_analysis_section.py
git commit -m "feat: add the [analysis] input section"
```

---

### Task 3: Readers for `output.h5` and `trajectories.json`

**Files:**
- Modify: `src/gqteashmd/trajectory/store.py` (append after `write_output_h5`)
- Modify: `src/gqteashmd/excite/artifacts.py` (append)
- Modify: `src/gqteashmd/stages/dynamics.py:144-163` (`plan` uses the reader)
- Test: `tests/trajectory/test_read_output.py`, `tests/excite/test_trajectories_json.py`

**Interfaces:**
- Consumes: `write_output_h5(path, rows, *, labels, n_atoms, output_every, attrs)`, `log_dtype`, `log_row` (store.py).
- Produces:
  - `OUTPUT_DATASETS: tuple[str, ...]` (the 22 dataset names written by `write_output_h5`, in that order).
  - `@dataclass(frozen=True) class TrajectoryData: path: Path; attrs: Mapping[str, Any]; data: Mapping[str, np.ndarray]` with `__getitem__(name) -> np.ndarray` and properties `labels -> tuple[str, ...]`, `output_every -> int`, `dt_fs -> float`, `outcome -> str`, `task -> str`, `initial_state -> str`, `n_records -> int`.
  - `read_output_h5(path: Path) -> TrajectoryData`; raises `ValueError` naming the path (missing file, unreadable HDF5, missing dataset or attribute, datasets of different first length).
  - `@dataclass(frozen=True) class SelectedTrajectory: index: int; sample: int; state: str; rng: Mapping[str, Any]` with property `task_id -> str` (`f"traj_{index:05d}"`).
  - `read_trajectories_json(path: Path) -> list[SelectedTrajectory]`; raises `ValueError` naming the path for a missing or malformed file.

- [ ] **Step 1: Write the failing tests**

`tests/trajectory/test_read_output.py`:

```python
import h5py
import numpy as np
import pytest

from gqteashmd.trajectory.store import (
    OUTPUT_DATASETS,
    log_dtype,
    log_row,
    read_output_h5,
    write_output_h5,
)

ATTRS = {"dt_fs": 0.5, "outcome": "finished", "task": "traj_00000", "initial_state": "S1",
         "sample": 3, "symbols": ["H", "H"]}


def _rows(n=5):
    dtype = log_dtype(3, 6)
    rows = np.stack([log_row(dtype, step=k, grid_step=k, time_au=20.0 * k, active=1,
                             hop_from=-1, hop_to=-1, e_total=-1.0 + 1e-6 * k,
                             c_real=[0.0, 1.0, 0.0]) for k in range(n)])
    return rows


def test_round_trip(tmp_path):
    path = tmp_path / "output.h5"
    write_output_h5(path, _rows(), labels=["S0", "S1", "S2"], n_atoms=2, output_every=1,
                    attrs=ATTRS)
    data = read_output_h5(path)
    assert set(OUTPUT_DATASETS) == set(data.data)
    assert data.labels == ("S0", "S1", "S2") and data.output_every == 1
    assert data.dt_fs == 0.5 and data.outcome == "finished" and data.task == "traj_00000"
    assert data.initial_state == "S1" and data.n_records == 5
    np.testing.assert_array_equal(data["grid_step"], np.arange(5))
    np.testing.assert_array_equal(data["populations"][:, 1], np.ones(5))
    assert data["positions"].shape == (5, 2, 3)


def test_missing_file_names_the_path(tmp_path):
    with pytest.raises(ValueError, match="output.h5"):
        read_output_h5(tmp_path / "output.h5")


def test_garbage_file_names_the_path(tmp_path):
    path = tmp_path / "output.h5"
    path.write_bytes(b"not hdf5")
    with pytest.raises(ValueError, match=str(path.name)):
        read_output_h5(path)


def test_missing_dataset_is_an_error(tmp_path):
    path = tmp_path / "output.h5"
    write_output_h5(path, _rows(), labels=["S0", "S1", "S2"], n_atoms=2, output_every=1,
                    attrs=ATTRS)
    with h5py.File(path, "a") as f:
        del f["e_total"]
    with pytest.raises(ValueError, match="e_total"):
        read_output_h5(path)


def test_inconsistent_lengths_are_an_error(tmp_path):
    path = tmp_path / "output.h5"
    write_output_h5(path, _rows(), labels=["S0", "S1", "S2"], n_atoms=2, output_every=1,
                    attrs=ATTRS)
    with h5py.File(path, "a") as f:
        del f["active"]
        f.create_dataset("active", data=np.zeros(4, dtype=int)).attrs["units"] = "-"
    with pytest.raises(ValueError, match="active"):
        read_output_h5(path)
```

`tests/excite/test_trajectories_json.py` (create `tests/excite/__init__.py` only if the other test packages have one):

```python
import json

import pytest

from gqteashmd.excite.artifacts import SelectedTrajectory, read_trajectories_json


def _write(path, trajectories):
    path.write_text(json.dumps({"schema_version": 1, "n_trajectories": len(trajectories),
                                "trajectories": trajectories}))


def test_reads_the_entries(tmp_path):
    path = tmp_path / "trajectories.json"
    _write(path, [{"index": 0, "sample": 4, "state": "S2",
                   "rng": {"stage": "dynamics", "task_index": 0}}])
    (entry,) = read_trajectories_json(path)
    assert entry == SelectedTrajectory(0, 4, "S2", {"stage": "dynamics", "task_index": 0})
    assert entry.task_id == "traj_00000"


@pytest.mark.parametrize("content", ["not json", '{"trajectories": [{"index": 0}]}', "{}"])
def test_malformed_file_is_a_value_error_naming_it(tmp_path, content):
    path = tmp_path / "trajectories.json"
    path.write_text(content)
    with pytest.raises(ValueError, match="trajectories.json"):
        read_trajectories_json(path)


def test_missing_file_is_a_value_error(tmp_path):
    with pytest.raises(ValueError, match="trajectories.json"):
        read_trajectories_json(tmp_path / "trajectories.json")
```

- [ ] **Step 2: Run them to see them fail**

Run: `pytest tests/trajectory/test_read_output.py tests/excite/test_trajectories_json.py -q -W error`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement**

Append to `trajectory/store.py` (add `from dataclasses import dataclass` at the top):

```python
OUTPUT_DATASETS = (
    "time_fs", "step", "grid_step", "ticks", "depth", "active", "hop_from", "hop_to",
    "frustrated", "rung_changed", "rejections", "random_number", "min_singular_value",
    "energies", "e_total", "e_kinetic", "populations", "coefficients_real",
    "coefficients_imag", "probabilities", "positions", "velocities",
)
_REQUIRED_ATTRS = ("state_labels", "output_every", "dt_fs", "outcome", "task", "initial_state")


def _plain(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class TrajectoryData:
    """The contents of one output.h5 (USER_MANUAL section 6)."""

    path: Path
    attrs: Mapping[str, Any]
    data: Mapping[str, np.ndarray]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.data[name]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(str(x) for x in self.attrs["state_labels"])

    @property
    def output_every(self) -> int:
        return int(self.attrs["output_every"])

    @property
    def dt_fs(self) -> float:
        return float(self.attrs["dt_fs"])

    @property
    def outcome(self) -> str:
        return str(self.attrs["outcome"])

    @property
    def task(self) -> str:
        return str(self.attrs["task"])

    @property
    def initial_state(self) -> str:
        return str(self.attrs["initial_state"])

    @property
    def n_records(self) -> int:
        return int(self.data["time_fs"].shape[0])


def read_output_h5(path: Path) -> TrajectoryData:
    """Read one trajectory's output.h5; ValueError naming the file for any problem."""
    import h5py

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path} does not exist")
    try:
        with h5py.File(path, "r") as f:
            missing = [n for n in OUTPUT_DATASETS if n not in f]
            missing += [a for a in _REQUIRED_ATTRS if a not in f.attrs]
            if missing:
                raise ValueError(f"{path} lacks {', '.join(missing)}")
            data = {n: np.asarray(f[n][()]) for n in OUTPUT_DATASETS}
            attrs = {k: _plain(v) for k, v in f.attrs.items()}
    except OSError as exc:
        raise ValueError(f"{path} is not a readable HDF5 file: {exc}") from exc
    lengths = {n: a.shape[0] if a.ndim else -1 for n, a in data.items()}
    m = lengths["time_fs"]
    wrong = sorted(n for n, k in lengths.items() if k != m)
    if wrong:
        raise ValueError(f"{path}: datasets {', '.join(wrong)} do not have {m} records")
    return TrajectoryData(path, attrs, data)
```

Append to `excite/artifacts.py` (add imports `json`, `dataclass`, `Mapping`, `Any`, `Path` as needed):

```python
@dataclass(frozen=True)
class SelectedTrajectory:
    """One entry of trajectories.json (USER_MANUAL section 6)."""

    index: int
    sample: int
    state: str
    rng: Mapping[str, Any]

    @property
    def task_id(self) -> str:
        return f"traj_{self.index:05d}"


def read_trajectories_json(path: Path) -> list[SelectedTrajectory]:
    """The selected trajectories; ValueError naming the file if it is missing or malformed."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [
            SelectedTrajectory(int(e["index"]), int(e["sample"]), str(e["state"]), e["rng"])
            for e in data["trajectories"]
        ]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{path} is missing or malformed ({exc!r})") from exc
```

In `stages/dynamics.py:plan`, replace the `try` block (lines 146-163) by:

```python
        try:
            selected = read_trajectories_json(path)
        except ValueError as exc:
            raise ProjectError(f"trajectories.json is missing or malformed ({exc}){advice}") from exc
        return [
            TaskSpec(
                t.task_id,
                {"index": t.index, "sample": t.sample, "state": t.state, "rng": dict(t.rng)},
            )
            for t in selected
        ]
```

and drop the now unused `read_json` import if ruff reports it.

- [ ] **Step 4: Run the new tests and the dynamics stage tests**

Run: `pytest tests/trajectory tests/excite tests/stages/test_dynamics.py tests/stages/test_dynamics_pipeline.py tests/dynamics/test_golden.py -q -W error`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gqteashmd/trajectory/store.py src/gqteashmd/excite/artifacts.py src/gqteashmd/stages/dynamics.py tests/trajectory/test_read_output.py tests/excite/test_trajectories_json.py
git commit -m "feat: add readers for output.h5 and trajectories.json"
```

---

### Task 4: Inclusion rule, common grid, populations and intervals

**Files:**
- Create: `src/gqteashmd/analysis/__init__.py` (docstring only), `src/gqteashmd/analysis/grid.py`, `src/gqteashmd/analysis/populations.py`
- Test: `tests/analysis/__init__.py` (if the other test packages have one), `tests/analysis/test_grid.py`, `tests/analysis/test_populations.py`

**Interfaces:**
- Produces (`analysis/grid.py`):
  - `FINISHED = "finished"`, `STOPPED_IN_S0 = "stopped_in_s0"`, `INCLUDED_OUTCOMES = (FINISHED, STOPPED_IN_S0)`.
  - `class InconsistentData(ValueError)`.
  - `@dataclass(frozen=True) class Trajectory: task: str; outcome: str; initial_state: int; grid_step: np.ndarray; ticks: np.ndarray; time_fs: np.ndarray; active: np.ndarray; hop_from: np.ndarray; hop_to: np.ndarray; frustrated: np.ndarray; e_total: np.ndarray; populations: np.ndarray` with properties `t_final_fs -> float` and `last_full_step -> int`.
  - `@dataclass(frozen=True) class Excluded: task: str; reason: str; t_final_fs: float | None; energy_jump_eV: float | None = None`.
  - `select(tasks: Sequence[str], trajectories: Mapping[str, Trajectory | None], jumps_eV: Mapping[str, float], *, energy_jump_eV: float, exclude_energy_jumps: bool) -> tuple[list[Trajectory], list[Excluded]]`.
  - `horizon_steps(included: Sequence[Trajectory], n_total: int) -> int`.
  - `@dataclass(frozen=True) class GridData: grid_steps: np.ndarray; time_fs: np.ndarray; active: np.ndarray; populations: np.ndarray; tasks: tuple[str, ...]; horizon_steps: int; horizon_fs: float; cut: tuple[str, ...]`.
  - `on_grid(included: Sequence[Trajectory], *, output_every: int, dt_fs: float, n_total: int, n_states: int) -> GridData`.
- Produces (`analysis/populations.py`):
  - `Z_95 = 1.959963984540054`.
  - `classical_populations(active: np.ndarray, n_states: int) -> tuple[np.ndarray, np.ndarray]` returning `(P (G, n), counts (G, n))`.
  - `wilson_intervals(counts: np.ndarray, n: int, z: float = Z_95) -> tuple[np.ndarray, np.ndarray]`.
  - `quantum_populations(populations: np.ndarray) -> np.ndarray` (mean over axis 0).
  - `bootstrap_indices(rng: np.random.Generator, n: int, b: int) -> np.ndarray` (`rng.integers(0, n, size=(b, n))`).
  - `bootstrap_intervals(per_trajectory: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]` (2.5 and 97.5 percentiles of the resample means; `per_trajectory` has the trajectory axis first).
  - `consistency(classical: np.ndarray, quantum: np.ndarray) -> np.ndarray` (`max over t of |P - rho|` per state).

- [ ] **Step 1: Write the failing tests**

`tests/analysis/test_grid.py`:

```python
import numpy as np
import pytest

from gqteashmd.analysis.grid import (
    Excluded,
    InconsistentData,
    Trajectory,
    horizon_steps,
    on_grid,
    select,
)


def traj(task, outcome, active, *, every=1, ticks=None, extra_grid=None, initial=1, n=3):
    """A trajectory with one record per grid step 0..len(active)-1 (ticks 0)."""
    k = len(active)
    grid = np.arange(k) if extra_grid is None else np.asarray(extra_grid)
    ticks = np.zeros(len(grid), dtype=int) if ticks is None else np.asarray(ticks)
    act = np.asarray(active)
    pops = np.zeros((len(grid), n))
    pops[np.arange(len(grid)), act] = 1.0
    return Trajectory(task, outcome, initial, grid, ticks, grid * 0.5, act,
                      np.full(len(grid), -1), np.full(len(grid), -1),
                      np.zeros(len(grid), bool), np.full(len(grid), -1.0), pops)


def test_inclusion_rule():
    t = {"a": traj("a", "finished", [1, 1]), "b": traj("b", "stopped_in_s0", [1, 0]),
         "c": traj("c", "energy_drift", [1]), "d": None,
         "e": traj("e", "finished", [1, 1])}
    included, excluded = select(["a", "b", "c", "d", "e"], t, {"e": 0.5},
                                energy_jump_eV=0.1, exclude_energy_jumps=False)
    assert [x.task for x in included] == ["a", "b", "e"]
    assert excluded == [Excluded("c", "energy_drift", 0.0), Excluded("d", "task_failed", None)]
    included, excluded = select(["e"], t, {"e": 0.5}, energy_jump_eV=0.1,
                                exclude_energy_jumps=True)
    assert included == [] and excluded == [Excluded("e", "energy_jump", 0.5, 0.5)]


def test_horizon_is_the_shortest_finished_trajectory_and_at_most_n_total():
    a = traj("a", "finished", [1] * 9)
    b = traj("b", "finished", [1] * 6)
    s = traj("s", "stopped_in_s0", [1, 0])
    assert horizon_steps([a, b, s], n_total=8) == 5
    assert horizon_steps([a], n_total=4) == 4
    assert horizon_steps([s], n_total=4) == 4


def test_grid_holds_stopped_trajectories_in_s0_and_lists_cut_ones():
    a = traj("a", "finished", [1, 1, 2, 2, 2])
    s = traj("s", "stopped_in_s0", [1, 0])
    g = on_grid([a, s], output_every=2, dt_fs=0.5, n_total=3, n_states=3)
    np.testing.assert_array_equal(g.grid_steps, [0, 2])
    np.testing.assert_allclose(g.time_fs, [0.0, 1.0])
    np.testing.assert_array_equal(g.active, [[1, 2], [1, 0]])
    np.testing.assert_array_equal(g.populations[1, 1], [1.0, 0.0, 0.0])
    assert g.horizon_steps == 3 and g.horizon_fs == 1.5 and g.cut == ("a",)
    assert g.tasks == ("a", "s")


def test_substep_records_are_not_grid_points():
    a = traj("a", "finished", [1, 1, 2], extra_grid=[0, 0, 1], ticks=[0, 4, 0])
    g = on_grid([a], output_every=1, dt_fs=0.5, n_total=1, n_states=3)
    np.testing.assert_array_equal(g.active, [[1, 2]])


def test_a_missing_grid_record_is_inconsistent():
    a = traj("a", "finished", [1, 1], extra_grid=[0, 2])
    with pytest.raises(InconsistentData, match="a"):
        on_grid([a], output_every=1, dt_fs=0.5, n_total=2, n_states=3)


def test_non_finite_populations_are_inconsistent():
    a = traj("a", "finished", [1, 1])
    a.populations[1, 0] = np.nan
    with pytest.raises(InconsistentData, match="non-finite"):
        on_grid([a], output_every=1, dt_fs=0.5, n_total=1, n_states=3)
```

`tests/analysis/test_populations.py` (the coverage bounds are fixed here, before any run: R = 400 repetitions, N = 100, B = 500, accepted band [0.90, 0.99] for the observed coverage of the 95 % bootstrap interval):

```python
import numpy as np

from gqteashmd.analysis.populations import (
    bootstrap_indices,
    bootstrap_intervals,
    classical_populations,
    consistency,
    quantum_populations,
    wilson_intervals,
)
from gqteashmd.dynamics.ensemble import wilson_interval


def test_classical_populations_count_active_states():
    active = np.array([[1, 1, 0], [1, 2, 0]])
    p, counts = classical_populations(active, 3)
    np.testing.assert_array_equal(counts, [[0, 2, 0], [0, 1, 1], [2, 0, 0]])
    np.testing.assert_allclose(p, counts / 2)


def test_wilson_matches_the_scalar_formula():
    counts = np.array([[0, 3, 10], [7, 1, 2]])
    lo, hi = wilson_intervals(counts, 10)
    for idx in np.ndindex(counts.shape):
        assert (lo[idx], hi[idx]) == wilson_interval(int(counts[idx]), 10)


def test_bootstrap_is_reproducible_and_brackets_the_mean():
    rng = np.random.default_rng(1)
    x = rng.random((50, 4, 2))
    i1 = bootstrap_indices(np.random.default_rng(5), 50, 300)
    i2 = bootstrap_indices(np.random.default_rng(5), 50, 300)
    np.testing.assert_array_equal(i1, i2)
    lo, hi = bootstrap_intervals(x, i1)
    mean = quantum_populations(x)
    assert lo.shape == mean.shape and np.all(lo <= mean) and np.all(mean <= hi)


def test_bootstrap_interval_coverage_on_synthetic_ensembles():
    rng = np.random.default_rng(20261111)
    true_mean = 2.0 / 7.0  # Beta(2, 5)
    covered = 0
    repetitions = 400
    for _ in range(repetitions):
        x = rng.beta(2.0, 5.0, size=(100, 1, 1))
        lo, hi = bootstrap_intervals(x, bootstrap_indices(rng, 100, 500))
        covered += bool(lo[0, 0] <= true_mean <= hi[0, 0])
    assert 0.90 <= covered / repetitions <= 0.99, covered


def test_consistency_is_the_largest_difference_per_state():
    p = np.array([[1.0, 0.0], [0.5, 0.5]])
    rho = np.array([[0.9, 0.1], [0.5, 0.5]])
    np.testing.assert_allclose(consistency(p, rho), [0.1, 0.1])
```

- [ ] **Step 2: Run them to see them fail**

Run: `pytest tests/analysis -q -W error`
Expected: FAIL (`ModuleNotFoundError: gqteashmd.analysis`).

- [ ] **Step 3: Implement `grid.py`**

```python
"""Inclusion rule and common time grid (P8 spec sections 3.1 and 3.2)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

FINISHED = "finished"
STOPPED_IN_S0 = "stopped_in_s0"
INCLUDED_OUTCOMES = (FINISHED, STOPPED_IN_S0)


class InconsistentData(ValueError):
    """A trajectory's records do not fit the common grid (a file or program error)."""


@dataclass(frozen=True)
class Trajectory:
    task: str
    outcome: str
    initial_state: int
    grid_step: np.ndarray
    ticks: np.ndarray
    time_fs: np.ndarray
    active: np.ndarray
    hop_from: np.ndarray
    hop_to: np.ndarray
    frustrated: np.ndarray
    e_total: np.ndarray
    populations: np.ndarray

    @property
    def t_final_fs(self) -> float:
        return float(self.time_fs[-1])

    @property
    def last_full_step(self) -> int:
        full = self.grid_step[self.ticks == 0]
        return int(full.max()) if full.size else 0


@dataclass(frozen=True)
class Excluded:
    task: str
    reason: str
    t_final_fs: float | None
    energy_jump_eV: float | None = None


def select(
    tasks: Sequence[str],
    trajectories: Mapping[str, Trajectory | None],
    jumps_eV: Mapping[str, float],
    *,
    energy_jump_eV: float,
    exclude_energy_jumps: bool,
) -> tuple[list[Trajectory], list[Excluded]]:
    included: list[Trajectory] = []
    excluded: list[Excluded] = []
    for task in tasks:
        t = trajectories.get(task)
        if t is None:
            excluded.append(Excluded(task, "task_failed", None))
        elif t.outcome not in INCLUDED_OUTCOMES:
            excluded.append(Excluded(task, t.outcome, t.t_final_fs))
        elif exclude_energy_jumps and jumps_eV.get(task, 0.0) > energy_jump_eV:
            jump = float(jumps_eV[task])
            excluded.append(Excluded(task, "energy_jump", t.t_final_fs, jump))
        else:
            included.append(t)
    return included, excluded


def horizon_steps(included: Sequence[Trajectory], n_total: int) -> int:
    ends = [t.last_full_step for t in included if t.outcome == FINISHED]
    return int(min([n_total, *ends]))


@dataclass(frozen=True)
class GridData:
    grid_steps: np.ndarray
    time_fs: np.ndarray
    active: np.ndarray
    populations: np.ndarray
    tasks: tuple[str, ...]
    horizon_steps: int
    horizon_fs: float
    cut: tuple[str, ...]


def on_grid(
    included: Sequence[Trajectory],
    *,
    output_every: int,
    dt_fs: float,
    n_total: int,
    n_states: int,
) -> GridData:
    h = horizon_steps(included, n_total)
    steps = np.arange(0, h + 1, output_every, dtype=np.int64)
    active = np.zeros((len(included), steps.size), dtype=np.int64)
    populations = np.zeros((len(included), steps.size, n_states))
    ground = np.zeros(n_states)
    ground[0] = 1.0
    cut: list[str] = []
    for i, t in enumerate(included):
        full = np.flatnonzero(t.ticks == 0)
        where = {int(t.grid_step[j]): int(j) for j in full}
        end = t.last_full_step
        for k, s in enumerate(steps):
            j = where.get(int(s))
            if j is not None:
                active[i, k] = int(t.active[j])
                populations[i, k] = t.populations[j]
            elif t.outcome == STOPPED_IN_S0 and s > end:
                populations[i, k] = ground
            else:
                raise InconsistentData(f"{t.task}: no record at grid step {int(s)}")
        if np.any((t.grid_step > h) | ((t.grid_step == h) & (t.ticks > 0))):
            cut.append(t.task)
        if not np.all(np.isfinite(populations[i])):
            raise InconsistentData(f"{t.task}: non-finite populations")
    return GridData(
        steps,
        steps * float(dt_fs),
        active,
        populations,
        tuple(t.task for t in included),
        h,
        h * float(dt_fs),
        tuple(cut),
    )
```

- [ ] **Step 4: Implement `populations.py`**

```python
"""Classical and quantum populations with 95 % intervals (P8 spec section 4)."""

from __future__ import annotations

import numpy as np

Z_95 = 1.959963984540054


def classical_populations(active: np.ndarray, n_states: int) -> tuple[np.ndarray, np.ndarray]:
    n = active.shape[0]
    counts = np.stack([(active == j).sum(axis=0) for j in range(n_states)], axis=-1)
    return counts / n, counts


def wilson_intervals(
    counts: np.ndarray, n: int, z: float = Z_95
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(counts, dtype=float) / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return np.maximum(0.0, centre - half), np.minimum(1.0, centre + half)


def quantum_populations(populations: np.ndarray) -> np.ndarray:
    result: np.ndarray = populations.mean(axis=0)
    return result


def bootstrap_indices(rng: np.random.Generator, n: int, b: int) -> np.ndarray:
    return rng.integers(0, n, size=(b, n))


def bootstrap_intervals(
    per_trajectory: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    means = np.stack([per_trajectory[idx].mean(axis=0) for idx in indices])
    lo, hi = np.percentile(means, [2.5, 97.5], axis=0)
    return lo, hi


def consistency(classical: np.ndarray, quantum: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.abs(classical - quantum).max(axis=0)
    return result
```

If `test_wilson_matches_the_scalar_formula` fails only in the last bit, change it to `pytest.approx(..., abs=1e-15)` and record the size of the difference in the report.

- [ ] **Step 5: Run the tests**

Run: `pytest tests/analysis -q -W error`
Expected: PASS. Record the observed coverage (print it once with `-s` or from the assertion message of a temporary failing run) in the report.

- [ ] **Step 6: Commit**

```bash
git add src/gqteashmd/analysis tests/analysis
git commit -m "feat: add the analysis grid, inclusion rule and population intervals"
```

---

### Task 5: The delayed mono-exponential fit

**Files:**
- Create: `src/gqteashmd/analysis/kinetics.py`
- Test: `tests/analysis/test_kinetics.py`

**Interfaces:**
- Produces:
  - `model_curve(t: np.ndarray, a: float, t0: float, tau: float) -> np.ndarray` (`a * exp(-max(t - t0, 0) / tau)`).
  - `@dataclass(frozen=True) class FitResult: a: float; t0_fs: float | None; tau_fs: float | None; rms: float | None; success: bool; flags: tuple[str, ...]`.
  - `fit_delayed_exponential(t: np.ndarray, y: np.ndarray, *, dt_out: float) -> FitResult`.
  - `@dataclass(frozen=True) class FitWithUncertainty: fit: FitResult; t0_interval_fs: tuple[float, float] | None; tau_interval_fs: tuple[float, float] | None; t0_std_fs: float | None; tau_std_fs: float | None; bootstrap_failures: int`.
  - `bootstrap_fit(t: np.ndarray, per_trajectory: np.ndarray, indices: np.ndarray, *, dt_out: float) -> FitWithUncertainty` (`per_trajectory` is (N, G); the target is its mean over axis 0).
  - Flag names (exact strings): `no_population`, `no_decay`, `fit_failed`, `tau_beyond_horizon`, `little_decay`, `t0_at_upper_bound`.

Rules (spec section 5): `a = y[0]`; `a <= 0` gives `no_population` (t0, tau `None`, success true); `min(y) >= 0.99 a` gives `no_decay` (t0, tau `None`, success true); otherwise least squares over `x = (t0, log tau)` with bounds `t0 in [0, T]` (`T = t[-1]`), starts `t0 in (0, T/4, T/2)` and `tau = max(t_e - t0_start, dt_out)` with `t_e` the first `t` where `y < a/e` (else `T`); the smallest cost wins; every start failing gives `fit_failed` (success false). After a fit: `tau > T` adds `tau_beyond_horizon`, `y[-1] > 0.9 a` adds `little_decay`, `t0 > T - dt_out` adds `t0_at_upper_bound`. `rms = sqrt(mean(residual^2))`. Bootstrap: the main fit on the full mean; if it has no `tau`, no resample is fitted (`bootstrap_failures = 0`, intervals `None`); otherwise each resample is fitted and a resample whose fit has no `tau` counts as a failure and is left out; with fewer than 2 successful resamples the intervals and standard deviations are `None`; otherwise percentiles 2.5/97.5 and the standard deviation with `ddof=1`.

Test bounds (fixed here, before any run): noise-free model data are recovered to relative 1e-6; for the synthetic ensemble (N = 2000 first-passage times `t_i = 10 + Exp(40)` fs, seed 20261111, grid 0..150 fs step 0.5 fs, B = 200), `|tau - 40| <= 4 tau_std`, `tau_std < 3`, `|t0 - 10| <= 4 t0_std + 0.5`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from gqteashmd.analysis.kinetics import (
    bootstrap_fit,
    fit_delayed_exponential,
    model_curve,
)
from gqteashmd.analysis.populations import bootstrap_indices

T = np.arange(0.0, 150.0 + 1e-9, 0.5)


def test_noise_free_model_is_recovered():
    y = model_curve(T, 1.0, 12.0, 35.0)
    r = fit_delayed_exponential(T, y, dt_out=0.5)
    assert r.success and r.flags == ()
    assert r.t0_fs == pytest.approx(12.0, rel=1e-6) and r.tau_fs == pytest.approx(35.0, rel=1e-6)
    assert r.rms < 1e-8


def test_zero_delay_is_not_flagged():
    r = fit_delayed_exponential(T, model_curve(T, 1.0, 0.0, 30.0), dt_out=0.5)
    assert r.t0_fs == pytest.approx(0.0, abs=1e-6) and "t0_at_upper_bound" not in r.flags


def test_flags():
    assert fit_delayed_exponential(T, np.ones_like(T), dt_out=0.5).flags == ("no_decay",)
    assert fit_delayed_exponential(T, np.zeros_like(T), dt_out=0.5).flags == ("no_population",)
    slow = fit_delayed_exponential(T, model_curve(T, 1.0, 0.0, 5000.0), dt_out=0.5)  # y(T) 0.97
    assert "tau_beyond_horizon" in slow.flags and "little_decay" in slow.flags
    late = fit_delayed_exponential(T, model_curve(T, 1.0, 149.8, 0.2), dt_out=0.5)
    assert "t0_at_upper_bound" in late.flags  # only the last point decays; if the optimizer
    # cannot place t0 there from the fixed starts, report it (do not loosen the rule)


def test_synthetic_ensemble_recovers_tau_and_t0_within_the_bootstrap_error():
    rng = np.random.default_rng(20261111)
    times = 10.0 + rng.exponential(40.0, size=2000)
    per_trajectory = (T[None, :] < times[:, None]).astype(float)
    result = bootstrap_fit(T, per_trajectory, bootstrap_indices(rng, 2000, 200), dt_out=0.5)
    fit = result.fit
    assert fit.success and result.tau_std_fs is not None and result.t0_std_fs is not None
    assert result.tau_std_fs < 3.0
    assert abs(fit.tau_fs - 40.0) <= 4 * result.tau_std_fs
    assert abs(fit.t0_fs - 10.0) <= 4 * result.t0_std_fs + 0.5
    lo, hi = result.tau_interval_fs
    assert lo < fit.tau_fs < hi and result.bootstrap_failures == 0


def test_no_fit_means_no_bootstrap():
    per = np.ones((10, T.size))
    result = bootstrap_fit(T, per, bootstrap_indices(np.random.default_rng(1), 10, 100),
                           dt_out=0.5)
    assert result.fit.flags == ("no_decay",) and result.tau_interval_fs is None
    assert result.bootstrap_failures == 0
```

- [ ] **Step 2: Run them to see them fail**

Run: `pytest tests/analysis/test_kinetics.py -q -W error`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
"""Delayed mono-exponential fit with bootstrap uncertainties (P8 spec section 5)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def model_curve(t: np.ndarray, a: float, t0: float, tau: float) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore"):
        curve: np.ndarray = a * np.exp(-np.maximum(np.asarray(t) - t0, 0.0) / tau)
    return curve


@dataclass(frozen=True)
class FitResult:
    a: float
    t0_fs: float | None
    tau_fs: float | None
    rms: float | None
    success: bool
    flags: tuple[str, ...]


def fit_delayed_exponential(t: np.ndarray, y: np.ndarray, *, dt_out: float) -> FitResult:
    from scipy.optimize import least_squares

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    a = float(y[0])
    if a <= 0.0:
        return FitResult(a, None, None, None, True, ("no_population",))
    if float(y.min()) >= 0.99 * a:
        return FitResult(a, None, None, None, True, ("no_decay",))
    horizon = float(t[-1])
    below = np.flatnonzero(y < a / math.e)
    t_e = float(t[below[0]]) if below.size else horizon

    def residual(x: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore"):
            tau = float(np.exp(min(x[1], 700.0)))
        return model_curve(t, a, float(x[0]), tau) - y

    best = None
    for t0_start in (0.0, horizon / 4, horizon / 2):
        x0 = np.array([t0_start, math.log(max(t_e - t0_start, dt_out))])
        try:
            r = least_squares(residual, x0, bounds=([0.0, -np.inf], [horizon, np.inf]))
        except (ValueError, FloatingPointError):
            continue
        if not r.success or not np.all(np.isfinite(r.x)):
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return FitResult(a, None, None, None, False, ("fit_failed",))
    t0 = float(best.x[0])
    tau = float(np.exp(min(best.x[1], 700.0)))
    rms = float(np.sqrt(np.mean(residual(best.x) ** 2)))
    flags: list[str] = []
    if tau > horizon:
        flags.append("tau_beyond_horizon")
    if float(y[-1]) > 0.9 * a:
        flags.append("little_decay")
    if t0 > horizon - dt_out:
        flags.append("t0_at_upper_bound")
    return FitResult(a, t0, tau, rms, True, tuple(flags))


@dataclass(frozen=True)
class FitWithUncertainty:
    fit: FitResult
    t0_interval_fs: tuple[float, float] | None
    tau_interval_fs: tuple[float, float] | None
    t0_std_fs: float | None
    tau_std_fs: float | None
    bootstrap_failures: int


def bootstrap_fit(
    t: np.ndarray, per_trajectory: np.ndarray, indices: np.ndarray, *, dt_out: float
) -> FitWithUncertainty:
    fit = fit_delayed_exponential(t, per_trajectory.mean(axis=0), dt_out=dt_out)
    if fit.tau_fs is None:
        return FitWithUncertainty(fit, None, None, None, None, 0)
    taus: list[float] = []
    t0s: list[float] = []
    failures = 0
    for idx in indices:
        r = fit_delayed_exponential(t, per_trajectory[idx].mean(axis=0), dt_out=dt_out)
        if r.tau_fs is None or r.t0_fs is None:
            failures += 1
            continue
        taus.append(r.tau_fs)
        t0s.append(r.t0_fs)
    if len(taus) < 2:
        return FitWithUncertainty(fit, None, None, None, None, failures)
    t0_lo, t0_hi = (float(v) for v in np.percentile(t0s, [2.5, 97.5]))
    tau_lo, tau_hi = (float(v) for v in np.percentile(taus, [2.5, 97.5]))
    return FitWithUncertainty(
        fit,
        (t0_lo, t0_hi),
        (tau_lo, tau_hi),
        float(np.std(t0s, ddof=1)),
        float(np.std(taus, ddof=1)),
        failures,
    )
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/analysis/test_kinetics.py -q -W error --durations=5`
Expected: PASS; the ensemble test takes a few seconds. Record the fitted values and standard deviations in the report. If a bound fails, report the numbers; do not change a bound (the controller rules).

- [ ] **Step 5: Commit**

```bash
git add src/gqteashmd/analysis/kinetics.py tests/analysis/test_kinetics.py
git commit -m "feat: add the delayed mono-exponential fit with bootstrap uncertainties"
```

---

### Task 6: Hop statistics and energy diagnostics

**Files:**
- Create: `src/gqteashmd/analysis/hops.py`, `src/gqteashmd/analysis/energy.py`
- Test: `tests/analysis/test_hops.py`, `tests/analysis/test_energy.py`

**Interfaces:**
- Consumes: `Trajectory` (Task 4).
- Produces (`hops.py`):
  - `@dataclass(frozen=True) class HopEvent: task: str; time_fs: float; from_state: int; to_state: int | None; frustrated: bool; included: bool; after_horizon: bool`.
  - `hop_events(t: Trajectory, *, included: bool, horizon_fs: float) -> list[HopEvent]` (accepted hops: `hop_to >= 0`, `from = hop_from`; frustrated: `frustrated`, `from = active` of the record, `to = None` (PR2); `after_horizon = time_fs > horizon_fs + 1e-9`).
  - `hop_statistics(events: Sequence[HopEvent], included: Sequence[Trajectory], labels: Sequence[str], horizon_fs: float) -> dict[str, Any]` returning exactly the keys `pairs` (`{"S2->S1": n}`, included, not after the horizon, accepted only), `frustrated_from` (`{"S1": n}`, same filter), `per_trajectory` (`{"min", "median", "max", "zero"}` over included trajectories, counts of accepted hops before the horizon), `first_hop_fs` (`{"n", "mean", "median"}`), `first_s0_arrival_fs` (`{"n_arrived", "n_censored", "mean", "median"}`, the first record with active 0 at or before the horizon; a `stopped_in_s0` trajectory always arrives), `totals_all` (`{"hops", "frustrated"}` over every event). Empty statistics are `None`, never NaN.
- Produces (`energy.py`): `max_jump_eV(e_total_hartree: np.ndarray) -> float` (0.0 for fewer than 2 records), `max_drift_eV(e_total_hartree: np.ndarray) -> float`, `summary_eV(values: Sequence[float]) -> dict[str, float | None]` (`median`, `p95`, `max`; `None` for an empty list). The conversion uses `gqteashmd.core.units.HARTREE_IN_EV`.

- [ ] **Step 1: Write the failing tests**

`tests/analysis/test_hops.py`:

```python
import numpy as np

from gqteashmd.analysis.grid import Trajectory
from gqteashmd.analysis.hops import HopEvent, hop_events, hop_statistics

LABELS = ("S0", "S1", "S2")


def traj(task, outcome, active, hop_from, hop_to, frustrated, dt=0.5):
    n = len(active)
    return Trajectory(task, outcome, int(active[0]), np.arange(n), np.zeros(n, int),
                      np.arange(n) * dt, np.asarray(active), np.asarray(hop_from),
                      np.asarray(hop_to), np.asarray(frustrated, bool),
                      np.zeros(n), np.zeros((n, 3)))


A = traj("a", "finished", [2, 2, 1, 1, 0], [-1, -1, 2, -1, 1], [-1, -1, 1, -1, 0],
         [0, 1, 0, 0, 0])
B = traj("b", "finished", [1, 1, 1, 1, 1], [-1] * 5, [-1] * 5, [0] * 5)


def test_events_and_after_horizon():
    events = hop_events(A, included=True, horizon_fs=1.5)
    assert events == [
        HopEvent("a", 0.5, 2, None, True, True, False),
        HopEvent("a", 1.0, 2, 1, False, True, False),
        HopEvent("a", 2.0, 1, 0, False, True, True),
    ]


def test_statistics():
    events = hop_events(A, included=True, horizon_fs=1.5) + hop_events(
        B, included=True, horizon_fs=1.5)
    s = hop_statistics(events, [A, B], LABELS, 1.5)
    assert s["pairs"] == {"S2->S1": 1}
    assert s["frustrated_from"] == {"S2": 1}
    assert s["per_trajectory"] == {"min": 0, "median": 0.5, "max": 1, "zero": 1}
    assert s["first_hop_fs"] == {"n": 1, "mean": 1.0, "median": 1.0}
    assert s["first_s0_arrival_fs"] == {"n_arrived": 0, "n_censored": 2, "mean": None,
                                        "median": None}
    assert s["totals_all"] == {"hops": 2, "frustrated": 1}
```

`tests/analysis/test_energy.py`:

```python
import numpy as np
import pytest

from gqteashmd.analysis.energy import max_drift_eV, max_jump_eV, summary_eV
from gqteashmd.core.units import HARTREE_IN_EV


def test_jump_and_drift():
    e = np.array([-1.0, -1.0 + 1e-4, -1.0 + 3e-4, -1.0 - 1e-3])
    assert max_jump_eV(e) == pytest.approx(1.3e-3 * HARTREE_IN_EV)
    assert max_drift_eV(e) == pytest.approx(1e-3 * HARTREE_IN_EV)
    assert max_jump_eV(e[:1]) == 0.0


def test_summary():
    assert summary_eV([]) == {"median": None, "p95": None, "max": None}
    s = summary_eV([1.0, 2.0, 3.0])
    assert s["median"] == 2.0 and s["max"] == 3.0 and 2.0 < s["p95"] <= 3.0
```

- [ ] **Step 2: Run them to see them fail**

Run: `pytest tests/analysis/test_hops.py tests/analysis/test_energy.py -q -W error`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `energy.py`**

```python
"""Energy drift and jump diagnostics (P8 spec section 7)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from gqteashmd.core.units import HARTREE_IN_EV


def max_jump_eV(e_total_hartree: np.ndarray) -> float:
    e = np.asarray(e_total_hartree, dtype=float)
    return float(np.abs(np.diff(e)).max() * HARTREE_IN_EV) if e.size >= 2 else 0.0


def max_drift_eV(e_total_hartree: np.ndarray) -> float:
    e = np.asarray(e_total_hartree, dtype=float)
    return float(np.abs(e - e[0]).max() * HARTREE_IN_EV) if e.size else 0.0


def summary_eV(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p95": None, "max": None}
    v = np.asarray(values, dtype=float)
    return {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max())}
```

- [ ] **Step 4: Implement `hops.py`**

```python
"""Hop statistics (P8 spec section 6; plan ruling PR2 for frustrated hops)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from gqteashmd.analysis.grid import STOPPED_IN_S0, Trajectory

_TOL = 1e-9


@dataclass(frozen=True)
class HopEvent:
    task: str
    time_fs: float
    from_state: int
    to_state: int | None
    frustrated: bool
    included: bool
    after_horizon: bool


def hop_events(t: Trajectory, *, included: bool, horizon_fs: float) -> list[HopEvent]:
    events: list[HopEvent] = []
    for j in range(t.time_fs.size):
        time = float(t.time_fs[j])
        after = time > horizon_fs + _TOL
        if int(t.hop_to[j]) >= 0:
            events.append(HopEvent(t.task, time, int(t.hop_from[j]), int(t.hop_to[j]), False,
                                   included, after))
        elif bool(t.frustrated[j]):
            events.append(HopEvent(t.task, time, int(t.active[j]), None, True, included, after))
    return events


def _stats(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.median(values))


def hop_statistics(
    events: Sequence[HopEvent],
    included: Sequence[Trajectory],
    labels: Sequence[str],
    horizon_fs: float,
) -> dict[str, Any]:
    counted = [e for e in events if e.included and not e.after_horizon]
    pairs = Counter(f"{labels[e.from_state]}->{labels[e.to_state]}" for e in counted
                    if e.to_state is not None)
    frustrated = Counter(labels[e.from_state] for e in counted if e.frustrated)
    per = [sum(1 for e in counted if e.task == t.task and not e.frustrated) for t in included]
    first_hop = [min(e.time_fs for e in counted if e.task == t.task and not e.frustrated)
                 for t in included
                 if any(e.task == t.task and not e.frustrated for e in counted)]
    arrivals: list[float] = []
    for t in included:
        hits = np.flatnonzero((t.active == 0) & (t.time_fs <= horizon_fs + _TOL))
        if hits.size:
            arrivals.append(float(t.time_fs[hits[0]]))
        elif t.outcome == STOPPED_IN_S0:
            arrivals.append(float(t.time_fs[-1]))
    hop_mean, hop_median = _stats(first_hop)
    s0_mean, s0_median = _stats(arrivals)
    return {
        "pairs": dict(sorted(pairs.items())),
        "frustrated_from": dict(sorted(frustrated.items())),
        "per_trajectory": (
            {"min": int(min(per)), "median": float(np.median(per)), "max": int(max(per)),
             "zero": int(sum(1 for p in per if p == 0))}
            if per else {"min": None, "median": None, "max": None, "zero": 0}
        ),
        "first_hop_fs": {"n": len(first_hop), "mean": hop_mean, "median": hop_median},
        "first_s0_arrival_fs": {"n_arrived": len(arrivals),
                                "n_censored": len(included) - len(arrivals),
                                "mean": s0_mean, "median": s0_median},
        "totals_all": {"hops": sum(1 for e in events if not e.frustrated),
                       "frustrated": sum(1 for e in events if e.frustrated)},
    }
```

(`per_trajectory.median` of `[1, 0]` is 0.5 and of an integer list may be a float; the test expects `0.5`.)

- [ ] **Step 5: Run the tests**

Run: `pytest tests/analysis -q -W error`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/gqteashmd/analysis/hops.py src/gqteashmd/analysis/energy.py tests/analysis/test_hops.py tests/analysis/test_energy.py
git commit -m "feat: add hop statistics and energy diagnostics for the analysis"
```

---

### Task 7: Output writers and the report

**Files:**
- Create: `src/gqteashmd/analysis/artifacts.py`, `src/gqteashmd/analysis/report.py`
- Test: `tests/analysis/test_artifacts.py`, `tests/analysis/test_report.py`

**Interfaces:**
- Consumes: `HopEvent` (Task 6).
- Produces (`artifacts.py`, PR1):
  - `write_populations_h5(path: Path, *, time_fs, labels, classical, classical_lower, classical_upper, quantum, quantum_lower, quantum_upper, fit_target, fit_curve, attrs: Mapping[str, Any]) -> None` (atomic; units `fs` for `time_fs`, `-` for the others; `state_labels` as a string dataset with units `-`; file attributes `schema_version` = 1, `gqteashmd_version`, plus `attrs`).
  - `format_populations_csv(*, time_fs, labels, classical, classical_lower, classical_upper, quantum, quantum_lower, quantum_upper, fit_target, fit_curve, comments: Sequence[str]) -> str` (`#` lines, a header, `%.6e` values; an empty field where `fit_curve` is NaN).
  - `format_hops_csv(events: Sequence[HopEvent], labels: Sequence[str]) -> str` (header `task,time_fs,from,to,frustrated,included,after_horizon`; `to` empty for a frustrated hop; `time_fs` in `%.6f`).
- Produces (`report.py`): `analysis_markdown(summary: Mapping[str, Any]) -> str` for the `analysis.json` dictionary of spec section 11 (exact keys listed in Task 9), and `json_safe(value: Any) -> Any` (recursively replaces non-finite floats by `None` and NumPy scalars/arrays by Python values).

- [ ] **Step 1: Write the failing tests**

`tests/analysis/test_artifacts.py`:

```python
import h5py
import numpy as np

from gqteashmd.analysis.artifacts import (
    format_hops_csv,
    format_populations_csv,
    write_populations_h5,
)
from gqteashmd.analysis.hops import HopEvent

LABELS = ("S0", "S1")
T = np.array([0.0, 0.5])
P = np.array([[0.0, 1.0], [0.5, 0.5]])
ARRAYS = dict(time_fs=T, labels=LABELS, classical=P, classical_lower=P - 0.1,
              classical_upper=P + 0.1, quantum=P, quantum_lower=P, quantum_upper=P,
              fit_target=np.array([1.0, 0.5]), fit_curve=np.array([1.0, np.nan]))


def test_populations_h5_has_units_and_attributes(tmp_path):
    path = tmp_path / "populations.h5"
    write_populations_h5(path, **ARRAYS, attrs={"n_included": 2, "horizon_fs": 0.5})
    with h5py.File(path, "r") as f:
        assert f.attrs["schema_version"] == 1 and f.attrs["n_included"] == 2
        assert "gqteashmd_version" in f.attrs
        for name in f:
            assert "units" in f[name].attrs, name
        np.testing.assert_array_equal(f["classical"][()], P)
        assert f["time_fs"].attrs["units"] == "fs"
    assert not list(tmp_path.glob(".*tmp"))


def test_populations_csv():
    text = format_populations_csv(**ARRAYS, comments=["populations", "n_included = 2"])
    lines = text.splitlines()
    assert lines[0] == "# populations" and lines[1] == "# n_included = 2"
    header = lines[2].split(",")
    assert header[:4] == ["time_fs", "P_S0", "P_S0_lower", "P_S0_upper"]
    assert header[-2:] == ["fit_target", "fit_curve"] and "rho_S1_upper" in header
    assert lines[3].split(",")[0] == "0.000000e+00"
    assert lines[4].endswith(",")  # fit_curve NaN -> empty field


def test_hops_csv():
    text = format_hops_csv([HopEvent("a", 1.0, 1, 0, False, True, False),
                            HopEvent("a", 0.5, 1, None, True, True, False)], LABELS)
    assert text.splitlines() == [
        "task,time_fs,from,to,frustrated,included,after_horizon",
        "a,1.000000,S1,S0,0,1,0",
        "a,0.500000,S1,,1,1,0",
    ]
```

`tests/analysis/test_report.py`:

```python
import json
import math

import numpy as np

from gqteashmd.analysis.report import analysis_markdown, json_safe


def test_json_safe_replaces_non_finite_values():
    value = {"a": math.nan, "b": [1.0, math.inf], "c": np.float64(2.5), "d": np.arange(2)}
    safe = json_safe(value)
    assert safe == {"a": None, "b": [1.0, None], "c": 2.5, "d": [0, 1]}
    json.dumps(safe, allow_nan=False)


def test_markdown_mentions_the_main_numbers(summary_example):
    text = analysis_markdown(summary_example)
    assert text.startswith("# Analysis")
    for needle in ("S0", "tau", "excluded", "horizon", "traj_00001"):
        assert needle in text, needle
```

Add a `summary_example` fixture in `tests/analysis/conftest.py` with a complete dictionary in the Task 9 schema (copy the key list of Task 9 Step 3 and fill small values: two included tasks, one excluded `traj_00001` with reason `energy_drift`, a fit with `tau_fs = 40.0`, a five-point `table` list of `{time_fs, P, rho}` rows).

- [ ] **Step 2: Run them to see them fail**

Run: `pytest tests/analysis/test_artifacts.py tests/analysis/test_report.py -q -W error`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

`artifacts.py`: follow `vib/artifacts.py:write_frequencies_h5` for the temporary file, `os.replace` and the `gqteashmd_version` attribute (import it from the same place `vib/artifacts.py` does); string dataset for `state_labels` with `h5py.string_dtype()`. `format_populations_csv`: columns `time_fs`; per state `P_<L>`, `P_<L>_lower`, `P_<L>_upper`; per state `rho_<L>`, `rho_<L>_lower`, `rho_<L>_upper`; `fit_target`, `fit_curve`; every value `format(x, ".6e")`, a NaN value an empty field; lines joined by `\n` with a final newline. `format_hops_csv` as in the test.

`report.py`: `json_safe` recursive over dict, list, tuple, `np.ndarray` (`tolist()` then recurse), `np.generic` (`item()`), float (non-finite to `None`). `analysis_markdown(summary)` writes, in order: `# Analysis`; a settings list (`fit_states`, `fit_population`, `n_bootstrap`, `energy_jump_eV`, `exclude_energy_jumps`, `project_seed`); `n_selected`, `n_included`, `horizon_fs`, `dt_out_fs`; the outcome counts; a table of the excluded trajectories (task, reason, t_final_fs, energy_jump_eV); the cut tasks; the population table from `summary["table"]` (columns `time_fs`, `P_<L>` and `rho_<L>` per state); the fit (A, t0, tau, intervals, standard deviations, rms, flags, bootstrap failures); hop tables (pairs, frustrated per originating state, first hop, first S0 arrival); energy (drift summary, flagged tasks with their jumps); the consistency line; the warnings; if `stage_failed`, a bold line with `error`; the paths `populations.csv`, `populations.h5`, `hops.csv`, `analysis.json`. Numbers in `.4g`; `None` as `-`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/analysis -q -W error`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gqteashmd/analysis/artifacts.py src/gqteashmd/analysis/report.py tests/analysis
git commit -m "feat: add the analysis output writers and report"
```

---

### Task 8: Orchestrator — reset the dependants of a reopened stage

**Files:**
- Modify: `src/gqteashmd/orchestrator.py` (`_reopen`, a new `_reset_dependants`)
- Modify: `src/gqteashmd/testing/fake_stages.py` (`make_pipeline`, docstring)
- Test: `tests/test_orchestrator.py` (append)

**Interfaces:**
- Consumes: `Orchestrator._reset_stage(state, index, stage)`, `Orchestrator._has_results(state, stage, record)`, `_reopen` (orchestrator.py:484-507).
- Produces: `Orchestrator._reset_dependants(state: StateStore, stage: Stage) -> list[str]` (names reset); behaviour keys `analysis` (bool: append `FakeStage("analysis", 1, ("dynamics",))`) and `analysis_disabled` (bool: that stage disabled) in the fake pipeline.

Rule (spec section 9): in `_reopen`, when `ids` is non-empty, **before** `state.set_stage(stage.name, StageStatus.RUNNING, ...)`, call `_reset_dependants(state, stage)`: walk the later stages in pipeline order; a later stage is affected if its `depends_on` contains the reopened stage or an affected stage; an affected stage whose record exists and `_has_results` is true is reset with `_reset_stage` (directory to `.gqteashmd/superseded/`, record reset), whether it is enabled or not, with `log.warning("Stage %s is recomputed because %s was extended", later.name, stage.name)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_orchestrator.py`:

```python
def _analysis_runs(project):
    path = _run_dir(project) / "05_analysis" / "tasks" / "t000" / "result.json"
    return read_json(path)["result"]["runs"]


def test_a_reopen_recomputes_the_dependants(project, behaviour, caplog):
    behaviour(analysis=True)
    _run(project)
    assert _analysis_runs(project) == 1
    behaviour(analysis=True, reopen=["dynamics:t001"])
    result = _run(project)
    assert result.completed == ["dynamics", "analysis"]
    assert _analysis_runs(project) == 1  # a fresh directory: the old one was superseded
    superseded = _run_dir(project) / ".gqteashmd" / "superseded"
    assert any(p.name.endswith("_05_analysis") for p in superseded.iterdir())
    assert "analysis is recomputed because dynamics was extended" in caplog.text
    with _state(project) as state:
        assert state.get_stage("analysis").status == StageStatus.DONE


def test_without_a_reopen_the_dependants_stay_up_to_date(project, behaviour):
    behaviour(analysis=True)
    _run(project)
    assert "analysis" in _run(project).up_to_date


def test_a_disabled_dependant_with_kept_results_is_reset(project, behaviour):
    behaviour(analysis=True)
    _run(project)
    behaviour(analysis=True, analysis_disabled=True, reopen=["dynamics:t001"])
    result = _run(project)
    assert result.completed == ["dynamics"] and "analysis" in result.skipped
    with _state(project) as state:
        assert state.get_stage("analysis").status == StageStatus.SKIPPED
    behaviour(analysis=True)
    assert "analysis" in _run(project).completed  # re-enabled: recomputed, not up to date


def test_a_kill_between_the_reset_and_the_running_mark_is_safe(project, behaviour, monkeypatch):
    behaviour(analysis=True)
    _run(project)
    behaviour(analysis=True, reopen=["dynamics:t001"])
    original = StateStore.set_stage

    def dying(self, name, status, **kwargs):
        if name == "dynamics" and status == StageStatus.RUNNING:
            raise RuntimeError("controller died")
        return original(self, name, status, **kwargs)

    monkeypatch.setattr(StateStore, "set_stage", dying)
    with pytest.raises(RuntimeError, match="controller died"):
        _run(project)
    with _state(project) as state:
        assert state.get_stage("dynamics").status == StageStatus.DONE
        assert state.get_stage("analysis") is None or state.get_stage(
            "analysis").status != StageStatus.DONE
    monkeypatch.setattr(StateStore, "set_stage", original)
    result = _run(project)
    assert result.completed == ["dynamics", "analysis"]
    assert _runs(project)["t001"] == 2


def test_a_kill_after_the_running_mark_is_safe(project, behaviour, monkeypatch):
    behaviour(analysis=True)
    _run(project)
    behaviour(analysis=True, reopen=["dynamics:t001"])
    original = StateStore.set_task

    def dying(self, stage, task_id, status, **kwargs):
        if kwargs.get("message") == "reopened":
            raise RuntimeError("controller died")
        return original(self, stage, task_id, status, **kwargs)

    monkeypatch.setattr(StateStore, "set_task", dying)
    with pytest.raises(RuntimeError, match="controller died"):
        _run(project)
    monkeypatch.setattr(StateStore, "set_task", original)
    result = _run(project)
    assert "analysis" in result.completed
    with _state(project) as state:
        assert state.get_stage("analysis").status == StageStatus.DONE
```

(Check the `set_stage` signature in `project/state.py` and adapt the `dying` wrapper to it; check that `StateStore.reset_stage` leaves `get_stage` returning `None` or a PENDING record and keep the assertion that it is not DONE.)

- [ ] **Step 2: Run them to see them fail**

Run: `pytest tests/test_orchestrator.py -q -W error -k "dependant or kill_between or kill_after or up_to_date"`
Expected: FAIL (no `05_analysis`; the fake pipeline has no analysis stage).

- [ ] **Step 3: Implement**

In `fake_stages.py:make_pipeline`, build the list as now and append, when `behaviour.get("analysis")`:

```python
    if behaviour.get("analysis", False):
        stages.append(
            FakeStage(
                "analysis",
                1,
                ("dynamics",),
                enabled=not behaviour.get("analysis_disabled", False),
            )
        )
    return stages
```

and document the two keys in the module docstring.

In `orchestrator.py`, in `_reopen` after `if not ids: return ids`:

```python
        # Dependants first: a kill before the RUNNING mark leaves this stage as it was and its
        # dependants without results, so the next run reopens again and recomputes them (P8).
        self._reset_dependants(state, stage)
        # RUNNING first: a kill part-way leaves a stage the normal path resumes.
        state.set_stage(stage.name, StageStatus.RUNNING, fingerprint=fp)
```

and the new method next to `_reset_stage`:

```python
    def _reset_dependants(self, state: StateStore, stage: Stage) -> list[str]:
        """Reset every later stage that depends on `stage` and has results (P8 spec 9)."""
        start = [s.name for s in self.stages].index(stage.name)
        affected = {stage.name}
        reset: list[str] = []
        for index in range(start + 1, len(self.stages)):
            later = self.stages[index]
            if affected.isdisjoint(later.depends_on):
                continue
            affected.add(later.name)
            record = state.get_stage(later.name)
            if record is None or not self._has_results(state, later, record):
                continue
            log.warning("Stage %s is recomputed because %s was extended", later.name, stage.name)
            self._reset_stage(state, index, later)
            reset.append(later.name)
        return reset
```

- [ ] **Step 4: Run the orchestrator tests**

Run: `pytest tests/test_orchestrator.py tests/stages/test_dynamics_pipeline.py -q -W error`
Expected: PASS (the existing reopen tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/gqteashmd/orchestrator.py src/gqteashmd/testing/fake_stages.py tests/test_orchestrator.py
git commit -m "feat: recompute the dependants of a reopened stage"
```

---

### Task 9: The `analysis` stage and its pipeline tests

**Files:**
- Create: `src/gqteashmd/stages/analysis.py`
- Modify: `src/gqteashmd/stages/registry.py`, `src/gqteashmd/testing/pipelines.py`, `tests/stages/test_base_pipeline.py:109-121`
- Test: `tests/stages/test_analysis.py`, `tests/stages/test_analysis_pipeline.py`

**Interfaces:**
- Consumes: `AnalysisSection` (Task 2); `read_output_h5`, `TrajectoryData`, `OUTPUT_FILE` (`trajectory/runner.py`); `read_trajectories_json` (Task 3); `analysis.grid`, `populations`, `kinetics`, `hops`, `energy`, `artifacts`, `report` (Tasks 4-7); `task_rng`; `read_json`, `write_json_atomic`, `write_text_atomic`, `write_bytes_atomic`; `StageFailedError`, `TaskError`.
- Produces: `AnalysisStage` (`name = "analysis"`, `version = 1`, `depends_on = ("selection", "dynamics")`, `config_sections = ("analysis",)`, `required_modules = ("h5py>=3.9", "scipy>=1.11")`, `max_failed_fraction = 0.0`); `NORM_TOLERANCE = 1e-6`; `default_pipeline` ends with `AnalysisStage()`; `gqteashmd.testing.pipelines:canned_analysis` (PR7).

- [ ] **Step 1: Write the failing stage tests**

`tests/stages/test_analysis.py` builds a fake project directory in `tmp_path` without the orchestrator: a `StageContext` whose upstream `selection` dir holds a `trajectories.json` (written with `canned_lvc.write_canned_trajectories`) and whose `dynamics` dir holds `dynamics_summary.json` and `tasks/traj_<k>/output.h5` files written with `write_output_h5` from synthetic rows (helper `_write_traj(dir, task, outcome, active_by_step, *, e_total=None, populations=None)`; rows at every grid step with ticks 0, `time_au = grid_step * dt_fs / AU_TIME_IN_FS`, `c_real = sqrt(populations)`). Use `load_config(write_input(tmp_path, extra=ELECTRONIC + "[dynamics]\ntmax_fs = 2.0\ndt_fs = 0.5\n" + analysis_extra))` for the config and `ProjectLayout(config.project.directory)`. Tests (each calls `run_task` then `finalize` directly):

```python
def test_populations_and_files(tmp_path, write_input):
    # three trajectories over 4 grid steps: finished S1->S1, finished S1->S0 at step 2,
    # energy_drift (excluded)
    ...
    stage_dir = run_stage(ctx)  # helper: plan, run_task in the task dir, finalize
    data = json.loads((stage_dir / "analysis.json").read_text())
    assert data["n_selected"] == 3 and data["n_included"] == 2
    assert data["excluded"] == [{"task": "traj_00002", "reason": "energy_drift",
                                 "t_final_fs": 0.5, "energy_jump_eV": None}]
    with h5py.File(stage_dir / "populations.h5") as f:
        np.testing.assert_allclose(f["classical"][()][:, 0], [0, 0, 0.5, 0.5, 0.5])
    for name in ("populations.csv", "hops.csv", "report.md"):
        assert (stage_dir / name).exists()


def test_no_included_trajectory_stops_after_writing_the_report(tmp_path, write_input):
    ...  # every trajectory energy_drift
    with pytest.raises(StageFailedError, match="No trajectory can be analysed"):
        run_stage(ctx)
    data = json.loads((stage_dir / "analysis.json").read_text())
    assert data["stage_failed"] is True and (stage_dir / "report.md").exists()


def test_a_broken_norm_stops_as_a_program_error(tmp_path, write_input):
    ...  # populations summing to 1.1 in one record
    with pytest.raises(StageFailedError, match="program error"):
        run_stage(ctx)


def test_an_unreadable_output_h5_is_a_non_retryable_task_error(tmp_path, write_input):
    ...  # garbage bytes in one output.h5
    with pytest.raises(TaskError, match="--rerun-from dynamics") as info:
        run_stage(ctx)
    assert info.value.retryable is False


def test_inconsistent_labels_are_a_task_error(tmp_path, write_input): ...
def test_missing_output_counts_as_task_failed(tmp_path, write_input): ...
def test_excluded_fraction_warning(tmp_path, write_input, caplog): ...
def test_energy_jump_flag_and_exclusion(tmp_path, write_input): ...
def test_bootstrap_is_reproducible_from_the_seed(tmp_path, write_input): ...
def test_fingerprint_inputs(tmp_path, write_input):
    # without enabled and warn_excluded_fraction; with project_seed
    ...
def test_enabled_needs_a_dynamics_table(tmp_path, write_input): ...
```

Write each test fully (the `...` above mark where the test builds its inputs with `_write_traj`; the asserted values are the ones in the comments). `test_excluded_fraction_warning`: 1 of 4 excluded with `warn_excluded_fraction = 0.2` → a warning containing `selection bias` in the log and in `analysis.json["warnings"]`; with 0.3 → none. `test_energy_jump_flag_and_exclusion`: one trajectory with a 0.2 eV jump (`e_total` step of `0.2 / HARTREE_IN_EV`) and `energy_jump_eV = 0.1`: listed in `energy.flagged` and included; with `exclude_energy_jumps = true`: excluded with reason `energy_jump`. `test_bootstrap_is_reproducible_from_the_seed`: two runs give byte-identical `populations.csv`; a different `project.seed` changes the quantum interval columns (use non-trivial populations, e.g. 0.3/0.7 splits differing by trajectory).

- [ ] **Step 2: Run them to see them fail**

Run: `pytest tests/stages/test_analysis.py -q -W error`
Expected: FAIL (`ModuleNotFoundError: gqteashmd.stages.analysis`).

- [ ] **Step 3: Implement `stages/analysis.py`**

```python
"""Stage 7: populations, a kinetic fit, hop and energy statistics (P8)."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.analysis import artifacts, energy, grid, hops, kinetics, populations, report
from gqteashmd.config.schema import ShmdConfig
from gqteashmd.errors import StageFailedError, TaskError
from gqteashmd.excite.artifacts import read_trajectories_json
from gqteashmd.stages.base import Stage, StageContext, TaskSpec
from gqteashmd.trajectory.outcomes import TrajectoryOutcome
from gqteashmd.trajectory.runner import OUTPUT_FILE
from gqteashmd.trajectory.store import TrajectoryData, read_output_h5
from gqteashmd.util.fileio import read_json, write_bytes_atomic, write_json_atomic, write_text_atomic
from gqteashmd.util.seeds import task_rng

log = logging.getLogger("gqteashmd.stages.analysis")
NORM_TOLERANCE = 1e-6
FILES = ("populations.h5", "populations.csv", "hops.csv")
DYNAMICS_ADVICE = "rerun with --rerun-from dynamics"


class AnalysisStage(Stage):
    name = "analysis"
    version = 1
    depends_on = ("selection", "dynamics")
    config_sections = ("analysis",)
    required_modules: tuple[str, ...] = ("h5py>=3.9", "scipy>=1.11")

    def enabled(self, config: ShmdConfig) -> bool:
        return config.dynamics is not None and config.analysis.enabled

    def fingerprint_inputs(self, ctx: StageContext) -> dict[str, Any]:
        section = ctx.config.analysis.model_dump(mode="json")
        section.pop("enabled")
        section.pop("warn_excluded_fraction")  # finalize-only
        return {"analysis": section, "project_seed": ctx.config.project.seed}

    def plan(self, ctx: StageContext) -> list[TaskSpec]:
        return [TaskSpec("analysis")]

    def run_task(self, ctx: StageContext, task: TaskSpec, task_dir: Path) -> dict[str, Any]:
        ...  # Step 3 below

    def finalize(self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]) -> None:
        ...  # Step 4 below
```

`run_task`, in this order (each numbered item is code the implementer writes in the method or a private helper of the module):

1. `section = ctx.config.analysis`, `dyn = ctx.config.dynamics` (assert not `None`), `n_total = round(dyn.tmax_fs / dyn.dt_fs)`, `dt_out = dyn.output_every * dyn.dt_fs`.
2. `selected = read_trajectories_json(ctx.upstream_dir("selection") / "trajectories.json")`; a `ValueError` becomes `TaskError(f"{exc}; rerun with --rerun-from selection", retryable=False)`.
3. `summary = read_json(dynamics_dir / "dynamics_summary.json")`, `rows = {r["task"]: r for r in summary["trajectories"]}`; `OSError`, `ValueError`, `KeyError`, `TypeError` become `TaskError(f"dynamics_summary.json is missing or malformed ({exc!r}); {DYNAMICS_ADVICE}", retryable=False)`.
4. For each selected entry: `path = ctx.layout.task_dir(dynamics_dir, entry.task_id) / OUTPUT_FILE`; if the task is not in `rows` or `path` does not exist, `trajectories[task] = None`; else `data = read_output_h5(path)` (`ValueError` → `TaskError(f"{exc}; {DYNAMICS_ADVICE}", retryable=False)`), check PR5 consistency against the first file and the config (`TaskError` naming the file, the field and `--rerun-from dynamics`), check that `e_total`, `populations` and `active` are finite (`TaskError`), convert with `_trajectory(data)` into `grid.Trajectory` (`initial_state = labels.index(data.initial_state)`; populations from the `populations` dataset).
5. `labels` = the first file's labels (if no file exists: `[str(l) for l in states_from_counts(ctx.config.electronic.singlets)]`); `fit_states = section.fit_states or labels[1:]`; `fit_index = [labels.index(s) for s in fit_states]`.
6. `jumps = {task: energy.max_jump_eV(t.e_total[t.grid_step <= n_total]) for task, t in trajectories.items() if t is not None}` (PR3); `included, excluded = grid.select([e.task_id for e in selected], trajectories, jumps, energy_jump_eV=section.energy_jump_eV, exclude_energy_jumps=section.exclude_energy_jumps)`.
7. Base summary dictionary (exact keys; the same keys are present in every case, values `None` where not computed): `schema_version` (1), `n_selected`, `n_included`, `outcomes` (counts of every `TrajectoryOutcome` value over the selected trajectories with a file, plus `task_failed`), `included` (tasks), `excluded` (list of `{"task", "reason", "t_final_fs", "energy_jump_eV"}`), `cut`, `horizon_fs`, `dt_out_fs`, `fit` (`{"states", "population", "A", "t0_fs", "tau_fs", "t0_interval_fs", "tau_interval_fs", "t0_std_fs", "tau_std_fs", "rms", "flags", "bootstrap_failures"}`), `hops`, `energy` (`{"drift_eV": summary_eV, "jump_threshold_eV", "flagged": [{"task", "energy_jump_eV"}]}`), `consistency` (`{label: max|P - rho|}`), `table` (at most about 25 rows `{"time_fs", "P": {label: p}, "rho": {label: r}}`, every `ceil(G / 25)`-th grid point and the last), `settings` (the `[analysis]` values and `project_seed`), `warnings` (list), `stage_failed` (bool), `error` (`None` or the message).
8. No included trajectory: `stage_failed = True`, `error = "No trajectory can be analysed: outcomes " + counts + "; only 'finished' and 'stopped_in_s0' trajectories enter the averages (USER_MANUAL 8.13)"`, plus, if `exclude_energy_jumps` excluded any, `"; set [analysis] exclude_energy_jumps = false and run with --rerun-from analysis"`. Write `analysis.json` (via `report.json_safe`) in `task_dir` and return `{"stage_failed": True, "n_included": 0}`.
9. Norm check (spec 8): `deviation = max over included of max |t.populations.sum(axis=1) - 1|`; above `NORM_TOLERANCE`: `stage_failed`, `error = f"populations of {task} sum to 1 within {deviation:.3e} only (tolerance 1e-6): this indicates a program error"`; write `analysis.json`; return.
10. `g = grid.on_grid(included, output_every=dyn.output_every, dt_fs=dyn.dt_fs, n_total=n_total, n_states=len(labels))`; `InconsistentData` → `TaskError(f"{exc}; {DYNAMICS_ADVICE}", retryable=False)`.
11. `P, counts = populations.classical_populations(g.active, n)`; `lo, hi = populations.wilson_intervals(counts, N)`; `rho = populations.quantum_populations(g.populations)`; `idx = populations.bootstrap_indices(task_rng(ctx.config.project.seed, "analysis", 0), N, section.n_bootstrap)`; `qlo, qhi = populations.bootstrap_intervals(g.populations, idx)`.
12. Fit: `per = np.isin(g.active, fit_index).astype(float)` (`classical`) or `g.populations[:, :, fit_index].sum(axis=-1)` (`quantum`); `fitu = kinetics.bootstrap_fit(g.time_fs, per, idx, dt_out=dt_out)`; `curve = kinetics.model_curve(g.time_fs, fit.a, fit.t0_fs, fit.tau_fs)` if `tau_fs` is not `None` else NaN; each flag adds a warning `f"fit: {flag}"` with its meaning (USER_MANUAL wording in Task 12).
13. Hops: events of every trajectory with a file (`included` true for the included ones), `hops.hop_statistics(events, included, labels, g.horizon_fs)`.
14. Energy: drifts of the included trajectories over records with `time_fs <= horizon_fs + 1e-9`; `flagged` = trajectories with a file whose jump exceeds the threshold.
15. Write in `task_dir`: `artifacts.write_populations_h5(...)` with attrs `n_included`, `n_selected`, `horizon_fs`, `dt_out_fs`, `fit_population`, `fit_states`; `populations.csv` (comments: title, units `fs` and population fraction, `n_included`, `horizon_fs`, `fit_population`, `fit_states`); `hops.csv`; `analysis.json` (`json_safe`). Return `{"stage_failed": False, "n_included": N, "horizon_fs": g.horizon_fs, "tau_fs": fit.tau_fs}`.

- [ ] **Step 4: Implement `finalize`**

```python
    def finalize(self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]) -> None:
        task_dir = ctx.layout.task_dir(ctx.stage_dir, "analysis")
        summary = read_json(task_dir / "analysis.json")
        n_selected = int(summary["n_selected"])
        fraction = (n_selected - int(summary["n_included"])) / n_selected if n_selected else 0.0
        limit = ctx.config.analysis.warn_excluded_fraction
        if fraction > limit:
            summary["warnings"].append(
                f"{fraction:.0%} of the selected trajectories are excluded (more than "
                f"warn_excluded_fraction = {limit}): possible selection bias, see 'excluded'"
            )
        for name in FILES:
            source = task_dir / name
            if source.exists():
                write_bytes_atomic(ctx.stage_dir / name, source.read_bytes())
        write_json_atomic(ctx.stage_dir / "analysis.json", summary)
        write_text_atomic(ctx.stage_dir / "report.md", report.analysis_markdown(summary))
        for warning in summary["warnings"]:
            log.warning("Stage analysis: %s", warning)
        if summary["stage_failed"]:
            raise StageFailedError(str(summary["error"]))
```

(Check that `write_json_atomic` refuses NaN; `summary` comes from JSON, so it has none.)

- [ ] **Step 5: Registry and canned pipeline**

`registry.py`: import `AnalysisStage`, append `AnalysisStage()` after the `DynamicsStage(...)` entry, update the module docstring (`... dynamics (optional: a [dynamics] table), analysis (after dynamics; [analysis] enabled)`). `tests/stages/test_base_pipeline.py:test_default_pipeline_order`: append `"analysis"`. `pipelines.py`:

```python
def canned_analysis(config: ShmdConfig) -> list[Stage]:
    """`canned_dynamics` followed by the real `analysis` stage (plan ruling PR7)."""
    from gqteashmd.stages.analysis import AnalysisStage

    return [*canned_dynamics(config), AnalysisStage()]
```

- [ ] **Step 6: Write the pipeline tests**

`tests/stages/test_analysis_pipeline.py` (same fixtures as `test_dynamics_pipeline.py`: `LVC`, the autouse template fixture, `FAST`, `ELECTRONIC`; `CANNED = "gqteashmd.testing.pipelines:canned_analysis"`; the analysis stage dir is `run/04_analysis`, the dynamics dir `run/03_dynamics`):

```python
def test_stage_populations_equal_a_direct_recomputation(tmp_path, write_input):
    path = _input(tmp_path, write_input)  # tmax 2.0, dt 0.25
    assert _main(path) == 0
    trajs = [read_output_h5(p) for p in sorted(_dyn(tmp_path).glob("tasks/traj_*/output.h5"))]
    full = [t["ticks"] == 0 for t in trajs]
    active = np.stack([t["active"][f] for t, f in zip(trajs, full)])
    pops = np.stack([t["populations"][f] for t, f in zip(trajs, full)])
    with h5py.File(_ana(tmp_path) / "populations.h5") as f:
        for j in range(pops.shape[2]):
            np.testing.assert_array_equal(f["classical"][()][:, j], (active == j).mean(axis=0))
        np.testing.assert_allclose(f["quantum"][()], pops.mean(axis=0), rtol=0, atol=1e-15)


def test_raising_tmax_recomputes_the_analysis_in_one_run(tmp_path, write_input):
    assert _main(_input(tmp_path, write_input)) == 0
    assert _analysis(tmp_path)["horizon_fs"] == pytest.approx(2.0)
    assert _main(_input(tmp_path, write_input, tmax="3.0")) == 0
    assert _analysis(tmp_path)["horizon_fs"] == pytest.approx(3.0)
    superseded = tmp_path / "run" / ".gqteashmd" / "superseded"
    assert any(p.name.endswith("_04_analysis") for p in superseded.iterdir())


def test_pause_after_dynamics_pauses_before_the_analysis(tmp_path, write_input):
    path = _input(tmp_path, write_input, extra_top='[workflow]\npause_after = ["dynamics"]\n')
    assert _main(path) == 5 and not (_ana(tmp_path) / "analysis.json").exists()
    assert _main(path) == 0 and (_ana(tmp_path) / "analysis.json").exists()


def test_disabled_analysis_is_skipped(tmp_path, write_input):
    assert _main(_input(tmp_path, write_input, extra="[analysis]\nenabled = false\n")) == 0
    assert not (_ana(tmp_path) / "analysis.json").exists()


def test_analysis_without_dynamics_is_skipped(tmp_path, write_input):
    path = write_input(tmp_path, execution=FAST, extra=ELECTRONIC + "[analysis]\n")
    assert _main(path) == 0
    assert not (_ana(tmp_path) / "analysis.json").exists()


def test_changing_an_analysis_key_is_stale_and_rerun_from_analysis_recomputes(
    tmp_path, write_input
):
    assert _main(_input(tmp_path, write_input)) == 0
    changed = _input(tmp_path, write_input, extra="[analysis]\nn_bootstrap = 200\n")
    assert _main(changed) == 3
    assert _main(changed, "--rerun-from", "analysis") == 0
    assert _analysis(tmp_path)["settings"]["n_bootstrap"] == 200


def test_warn_excluded_fraction_is_finalize_only(tmp_path, write_input):
    assert _main(_input(tmp_path, write_input)) == 0
    changed = _input(tmp_path, write_input, extra="[analysis]\nwarn_excluded_fraction = 0.5\n")
    assert _main(changed) == 0  # not stale
```

Write the helpers `_input(tmp_path, write_input, tmax="2.0", extra="", extra_top="")` (puts `extra_top` before `ELECTRONIC` and `extra` after the `[dynamics]` table), `_main`, `_dyn`, `_ana`, `_analysis` (reads `run/04_analysis/analysis.json`).

- [ ] **Step 7: Run the tests, the fast tier, lint and types**

Run: `pytest tests/stages tests/analysis tests/test_orchestrator.py tests/test_cli.py -q -W error`, then `pytest -m "not slow and not pyscf" -W error -q` once (record count and time), then `ruff format . && ruff check . && mypy`.
Expected: PASS everywhere. Also run `python -c "import sys, gqteashmd.cli; print([m for m in ('pyscf','geometric','h5py','scipy') if m in sys.modules])"` and expect `[]`.

- [ ] **Step 8: Commit**

```bash
git add src/gqteashmd/stages/analysis.py src/gqteashmd/stages/registry.py src/gqteashmd/testing/pipelines.py tests/stages/test_analysis.py tests/stages/test_analysis_pipeline.py tests/stages/test_base_pipeline.py
git commit -m "feat: add the analysis stage"
```

---

### Task 10: End-to-end through `analysis` (PySCF water, canned kill and resume)

**Files:**
- Modify: `tests/test_e2e_dynamics.py`
- Modify: `tests/stages/test_analysis_pipeline.py` (one slow test)

**Interfaces:**
- Consumes: the stage of Task 9; the e2e helpers of `tests/test_e2e_dynamics.py` (`_layout`, `_summary`, the stage index 7 for `analysis`).

- [ ] **Step 1: Extend the water e2e test**

In `test_workflow_runs_dynamics_resumes_and_extends`, after the first `_check_rows(layout)`:

```python
    analysis = json.loads((layout.stage_dir(7, "analysis") / "analysis.json").read_text())
    included = {r["task"] for r in _summary(layout)["trajectories"]
                if r["outcome"] in ("finished", "stopped_in_s0")}
    assert set(analysis["included"]) == included and analysis["stage_failed"] is False
    assert analysis["horizon_fs"] == pytest.approx(1.0)
    assert (layout.stage_dir(7, "analysis") / "populations.csv").exists()
```

and after the extension run:

```python
    analysis = json.loads((layout.stage_dir(7, "analysis") / "analysis.json").read_text())
    assert analysis["horizon_fs"] == pytest.approx(1.5)  # recomputed after the reopen (P8)
```

In `test_a_killed_controller_resumes_the_dynamics`, after the second run assert that `07_analysis/analysis.json` exists and `stage_failed` is false.

- [ ] **Step 2: Add the slow canned kill-and-resume test**

In `tests/stages/test_analysis_pipeline.py`:

```python
@pytest.mark.slow
def test_a_killed_controller_resumes_through_the_analysis(tmp_path, write_input):
    ref, run = tmp_path / "ref", tmp_path / "kill"
    ref.mkdir()
    run.mkdir()
    assert _main(_input(ref, write_input, tmax="25.0")) == 0
    path = _input(run, write_input, tmax="25.0")
    marker = run / "run" / "03_dynamics" / "tasks" / "traj_00000" / "restart" / "current"
    env = dict(os.environ)
    proc = subprocess.Popen(
        [sys.executable, "-m", "gqteashmd", "run", str(path), "--pipeline", CANNED],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 120
        while not marker.exists():
            if proc.poll() is not None:
                pytest.fail(f"the run ended (exit {proc.returncode}) before {marker} existed")
            if time.monotonic() > deadline:
                pytest.fail(f"{marker} did not appear within 120 s")
            time.sleep(0.01)
        assert proc.poll() is None, "the controller finished before it could be killed"
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    assert _main(path) == 0
    a, b = (r / "run" / "04_analysis" / "populations.csv" for r in (ref, run))
    assert a.read_bytes() == b.read_bytes()
```

(`tmax_fs = 25.0` gives 100 steps per trajectory so the kill falls inside the dynamics; record in the report how long the canned run takes. If the controller regularly finishes before the marker appears, raise `tmax_fs` and record the value.)

- [ ] **Step 3: Run**

Run: `pytest tests/test_e2e_dynamics.py tests/stages/test_analysis_pipeline.py -q -W error -m "pyscf or slow or not slow"` (all markers; the water tests need PySCF, a few minutes).
Expected: PASS. Record times.

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e_dynamics.py tests/stages/test_analysis_pipeline.py
git commit -m "test: run the workflow end to end through the analysis, with a kill and resume"
```

---

### Task 11: The M4 study (report-only)

**Files:**
- Report: `.superpowers/sdd/2026-11-11-p8-analysis-stage/task-11-report.md`
- No change to `src/` or `tests/` (a defect found here is reported, and the controller rules on a fix task).

**Interfaces:**
- Consumes: the whole pipeline at the branch head.

- [ ] **Step 1: Write the input (WSL native filesystem)**

```bash
mkdir -p ~/p8_m4 && cp /mnt/c/Users/ajc/OneDrive/gqteaSHMD/tests/data/ethylene.xyz ~/p8_m4/
cat > ~/p8_m4/input.toml <<'EOF'
[project]
name = "ethylene_m4"
directory = "run"
seed = 2026

[molecule]
geometry = "ethylene.xyz"
charge = 0
multiplicity = 1

[resources]
total_cores = 16
total_memory_mb = 7000
max_parallel_tasks = 8
threads_per_task = 2
memory_mb_per_task = 800

[electronic]
method = "casscf"
basis = "6-31g*"
singlets = 2

[electronic.casscf]
ncas = 2
nelecas = 2

[sampling]
n_samples = 100

[selection]
n_trajectories = 20

[dynamics]
tmax_fs = 100.0
dt_fs = 0.5
EOF
```

(Check every key against the schema with `gqteashmd validate ~/p8_m4/input.toml` first; exit 0 expected.)

- [ ] **Step 2: Run, kill once during the dynamics, resume**

```bash
cd ~/p8_m4 && date -Is > started.txt
nohup gqteashmd run input.toml > run1.log 2>&1 &
echo $! > controller.pid
```

Wait (bounded checks, every 5-10 min) until `run/06_dynamics/tasks/` has at least one `restart/current` and at least 10 min of dynamics have passed; then `kill -KILL -$(ps -o pgid= $(cat controller.pid) | tr -d ' ')` (the controller's process group; workers have their own sessions and are reconciled on the restart). Then `nohup gqteashmd run input.toml > run2.log 2>&1 &` and wait for its exit code (`echo $?` from a wrapper, or the last log line). If `selection` stops with too few trajectories (O20), stop and report the selection numbers; the controller rules (raise `n_samples`).

- [ ] **Step 3: Collect the numbers**

From `run/logs/gqteashmd.log` the start and end time of every stage; from `run/06_dynamics/dynamics_summary.json` the outcome counts, rejections, max drift; from `run/07_analysis/analysis.json` `n_included`, the excluded list, the horizon, the fit (A, t0, tau with intervals and flags), the hop statistics, the energy summary and flagged jumps, the consistency; the populations at 0, 25, 50, 75, 100 fs from `populations.csv`; which trajectories were running at the kill and that they resumed (their `restart/` generations and `progress.json`); the total wall time. Write everything into `task-11-report.md` with the exact input, and copy `run/07_analysis/report.md` into the report as an appendix.

- [ ] **Step 4: No commit** (the report is in the git-ignored ledger directory; Task 12 puts the numbers into USER_MANUAL).

---

### Task 12: Documents

**Files:**
- Modify: `USER_MANUAL.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: the spec; the reports of Tasks 1, 4, 5, 9, 10, 11 (numbers quoted from them, never from memory); the ledger's rulings.

- [ ] **Step 1: USER_MANUAL.md**

- §1 and the contents: the workflow ends with `analysis`.
- §4.2: the `[analysis]` table (spec section 10, with the Task 1 default) and the cross-field rule; §4.4 "Planned sections": remove `[analysis]`.
- §6: `07_analysis/` in the directory tree; tables for `analysis.json`, `populations.h5`, `populations.csv`, `hops.csv`, `report.md` (spec section 11; frustrated hops per originating state, PR2).
- §7.5: the reopen rule (spec section 9): raising `tmax_fs` recomputes `analysis` automatically; a changed `[analysis]` key needs `--rerun-from analysis`; `warn_excluded_fraction` is finalize-only. §7.8: `pause_after = ["dynamics"]` pauses before the analysis.
- §8.13 becomes [IMPLEMENTED]: the inclusion rule and why (selection bias, fixed before data), the horizon and grid, Wilson and bootstrap formulas, the fit model, starts, flags and bootstrap, hop statistics, energy diagnostics (jump between kept records, PR3), reproducibility from `project.seed`.
- A new §5.9: the M4 study (input, times, outcomes, excluded fraction, populations table, tau/t0 with intervals and flags, the kill and resume), report-only, with an honest summary.
- §9 troubleshooting rows: `No trajectory can be analysed`, `sum to 1 within ... program error`, `output.h5 ... --rerun-from dynamics`, the excluded-fraction warning, the fit flags.
- §10 known limitations: no plots; one kinetic model; the jump measured between kept records (several steps apart with `output_every > 1`); excluding early-ended trajectories can bias populations when failures correlate with the dynamics (report the excluded fraction); `first_s0_arrival` is reported, not fitted; frustrated-hop targets are not recorded.
- §11 roadmap: P8 done; next P9.

- [ ] **Step 2: CLAUDE.md**

- Architecture: the `analysis/` package and `stages/analysis.py` in the diagram; a "P8 modules and rules" block (modules, the dependency rule, PR1-PR9, the stage contract, the reopen rule, the fingerprint, the measured numbers of Tasks 1, 4, 5, 10, 11).
- Orchestrator rules: the dependants reset of spec section 9.
- Config: `[analysis]`.
- Preflight: `analysis` needs `h5py>=3.9`, `scipy>=1.11`.
- Key decisions: O10 resolved (owner decisions 2026-11-11, the table of spec section 0); the P7b energy-jump owner decision resolved as a diagnostic.
- Current status: P8 on `feat/p8-analysis-stage`, final review and merge pending; test counts with commits.
- Known limitations and the deferred list: remove the P8 items it resolved (the staleness rule, the common-time rule, the energy-jump decision) and add the P8 limitations; keep everything else.
- Remaining work table: P8 done on the branch; P9 next.

- [ ] **Step 3: Check and commit**

Run: `pytest tests/test_cli.py -q -W error` (the docs are not tested; this catches a template change by mistake), `git diff --stat`.

```bash
git add USER_MANUAL.md CLAUDE.md
git commit -m "docs: document the analysis stage and the M4 study"
```
