# P7b: the `dynamics` stage with PySCF — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one surface-hopping trajectory per entry of `trajectories.json` with PySCF electronic structure, as the resumable workflow stage `dynamics`, reusing the validated core `gqteashmd.dynamics` unchanged.

**Architecture:** A new package `gqteashmd.trajectory` holds `BackendModel` (the core's `ElectronicModel` over any `ElectronicBackend`), the trajectory runner (steps, step rejection with halving, outcomes, per-step restart, resume, extension) and its files. `stages/dynamics.py` is a thin stage over it. A fake backend over the analytic LVC models makes the runner and the stage testable in the fast tier without PySCF; PySCF and SHARC comparisons come last.

**Tech Stack:** Python 3.12, NumPy, pydantic 2, h5py (inside functions), PySCF 2.14 (only through `electronic/`), pytest.

**Spec:** [docs/superpowers/specs/2026-11-11-p7b-dynamics-stage-design.md](../specs/2026-11-11-p7b-dynamics-stage-design.md) (approved 2026-11-11). Executors read both. Where this plan's Rulings differ from the spec, the Rulings win (each says why).

## Global Constraints

- Every command runs in WSL: `wsl -d OracleLinux_9_5 -- bash -lc 'cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && conda activate gqteashmd-dev && <command>'`. Below, only `<command>` is written.
- Git from WSL only (pre-commit hooks run there); never `--no-verify`. Branch `feat/p7b-dynamics-stage`; one commit per task; conventional subjects (`feat:`, `fix:`, `test:`, `docs:`).
- End every commit message with the `Co-Authored-By:` trailer of the model that wrote the commit.
- Files use LF line endings. Lint: `ruff format . && ruff check .` (line length 100, rules E, F, W, I, B, UP, SIM). `mypy` must stay clean on `src/`.
- **Clean room:** SHARC 4.1 is executed only, from `tools/sharc_compare/`; never read, copy or adapt SHARC source; formats come from the public SHARC 4.1 manual and from our own existing harness code.
- **Dependency rules:** `gqteashmd/dynamics/` imports nothing from `electronic/`, `stages/`, the orchestrator or `config/` and is **not modified** by this plan. `gqteashmd/trajectory/` may import `dynamics/`, `electronic/base.py`, `overlaps/`, `core/`, `errors`, `util/`; never `stages/`, the orchestrator, `config/` or PySCF. h5py is imported inside functions. `import gqteashmd.cli` must not import pyscf, geometric or h5py.
- The golden test `tests/dynamics/test_golden.py` stays green in every task.
- **Seeds (D12):** trajectory k draws from `task_rng(project.seed, "dynamics", k)` (the stream `trajectories.json` records as `rng: {"stage": "dynamics", "task_index": k}`); exactly one uniform number per call of `driver.step`.
- Atomic units inside; fs and eV only in the input and in reports. Every HDF5 dataset carries a `units` attribute.
- Errors: input problems are `ConfigError` (exit 2); a task's non-retryable problem is `TaskError(..., retryable=False)`; `ElectronicStructureError` with `retryable=True` (MemoryError/OSError inside PySCF) propagates so the orchestrator retries the task. Every advice that requires a hashed key change names the `--rerun-from` stage.
- **House rules for validation:** tolerances and bounds are fixed in this plan before any number is seen; a failing assertion is reported with its numbers, never loosened; an assertion changed after numbers were seen is labelled post hoc in the test and in USER_MANUAL.md.
- Fast tier: `pytest -m "not slow and not pyscf" -W error` (about 10 min on `/mnt/c`). Run the targeted test files in each task; run the fast tier at the end of Tasks 5, 8 and 11.
- Keep `CLAUDE.md` and `USER_MANUAL.md` current in the same branch (Task 11 does the full update; earlier tasks change no documents except where stated).

## Rulings (decisions this plan makes where the spec left a choice open or the code forced one)

- **R1 (deviates from spec §7):** the stage is **enabled iff the input has a `[dynamics]` table** (`DynamicsStage.enabled` returns `config.dynamics is not None`). Why: every existing input, test and example ends at `selection`; making `[dynamics]` mandatory would break them all. Cost if wrong: a user who forgets the table gets a workflow that ends at `selection` with the log line "Stage dynamics: skipped (disabled in the input file)"; USER_MANUAL says so.
- **R2:** `steps.log` holds **every accepted (sub)step**; `output.h5` is thinned by `output_every`: it keeps the full-step records whose step number is a multiple of `output_every`, every record with a hop, a frustrated hop, a flag or a preceding rejection, and the last record. Why: hops and rejections must never be lost from the output. `output_every` stays hashed (spec §7).
- **R3:** time is kept on an **integer grid**: `grid_step` (completed full steps) and `ticks` (position inside the current interval, in units of dt / 2^`max_step_halvings`). `tmax_fs` must be an integer multiple of `dt_fs` (relative tolerance 1e-9; otherwise a `ConfigError`). The core's own `state.time_au` is not used for decisions.
- **R4:** the rung signal is CASSCF only: `meta["casscf"] != "previous orbitals"` at a geometry with a previous one. TD and SCF rung changes are not signals.
- **R5 (deviates from spec §7 "finalize builds any missing output.h5"):** `run_task` always writes `output.h5` before it returns (every outcome); `finalize` only lists missing ones in `dynamics_summary.json` and logs a warning. Why: building it needs the model's labels and sizes, which `finalize` does not have; `run_task` always has them.
- **R6:** a non-retryable failure of the first calculation (step 0) ends the trajectory as `electronic_failure` at t = 0 and writes **no restart**; rerunning that task recomputes step 0.
- **R7:** a failure of the gradient refresh after a hop (`compute_more`) is an electronic rejection signal for that step.
- **R8:** `n_rejections` counts every rejected attempt, including the last one at the smallest step that ends a trajectory.
- **R9:** after an extension (O16) E_tot(0) stays the value at t = 0 (it is in the core restart).
- **R10:** a fresh start calls `backend.reset()` before step 0; a resume loads the backend snapshot of the current restart generation.

## File structure

| File | Task | Responsibility |
|---|---|---|
| `src/gqteashmd/config/schema.py` (modify) | 1 | `DynamicsStageSection`, `ShmdConfig.dynamics`, cross-field rules |
| `src/gqteashmd/testing/fake_backend.py` (create) | 2 | `ModelBackend` (an `ElectronicBackend` over an analytic model), scripted faults, `SimulatedCrash` |
| `src/gqteashmd/trajectory/__init__.py`, `model.py`, `outcomes.py` (create) | 3 | `BackendModel`, `TrajectoryOutcome` |
| `src/gqteashmd/trajectory/store.py` (create) | 4 | `StepLog`, `log_dtype`, `log_row`, `RestartStore`, `write_output_h5` |
| `src/gqteashmd/trajectory/runner.py` (create) | 5 | `RunSettings`, `InitialConditions`, `Progress`, `TrajectoryRunner` |
| `src/gqteashmd/stages/base.py`, `orchestrator.py`, `testing/fake_stages.py` (modify) | 6 | `Stage.reopen_tasks` hook (O16) |
| `src/gqteashmd/stages/dynamics.py`, `src/gqteashmd/testing/canned_lvc.py` (create) | 7 | `DynamicsStage`, `run_settings`; canned upstream writers |
| `src/gqteashmd/stages/registry.py`, `testing/pipelines.py`, `cli.py` (modify) | 8 | pipeline, `canned_dynamics`, `init` template |
| `tests/validation/test_pyscf_dynamics.py`, `tests/test_e2e_dynamics.py` (create) | 9 | PySCF validation and end-to-end |
| `tools/sharc_compare/run_sharc_pyscf_dynamics.py`, `tests/validation/test_sharc_pyscf_dynamics.py` (create) | 10 | SHARC black box |
| `CLAUDE.md`, `USER_MANUAL.md` | 11 | documents and final verification |

Tests: `tests/config/test_dynamics_section.py` (1), `tests/trajectory/test_fake_backend.py` (2), `tests/trajectory/test_model.py` (3), `tests/trajectory/test_store.py` (4), `tests/trajectory/test_runner.py` (5), `tests/test_orchestrator.py` (6), `tests/stages/test_dynamics.py` (7), `tests/stages/test_dynamics_pipeline.py` (8). Create `tests/trajectory/__init__.py` only if the other test directories have one (follow the existing layout).

LVC fixtures used throughout: `tests/data/lvc/lvc_singlets.template` (3 singlets, 3 atoms C O O, 9 Cartesian coordinates; `V0.txt` next to it), loaded as `LVCModel.from_template(path, spin_orbit=False)`.

---

### Task 1: The `[dynamics]` input section

**Files:**
- Modify: `src/gqteashmd/config/schema.py` (add the section class after `SelectionSection`; add the field to `ShmdConfig`; extend `_consistent`)
- Test: `tests/config/test_dynamics_section.py`

**Interfaces:**
- Produces: `gqteashmd.config.schema.DynamicsStageSection` (fields below) and `ShmdConfig.dynamics: DynamicsStageSection | None = None`. The name is `DynamicsStageSection` because `config/model_dynamics.py` already has a `DynamicsSection`.

- [ ] **Step 1: Write the failing tests**

```python
"""The [dynamics] section of the workflow input (P7b)."""

import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import ConfigError

TDA = '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 3\n'
CAS = (
    '[electronic]\nmethod = "casscf"\nbasis = "sto-3g"\nsinglets = 2\n'
    "[electronic.casscf]\nncas = 2\nnelecas = 2\n"
)


def _load(tmp_path, write_input, extra):
    return load_config(write_input(tmp_path, extra=extra)).config


def test_absent_table_means_no_dynamics(tmp_path, write_input):
    assert _load(tmp_path, write_input, TDA).dynamics is None


def test_defaults(tmp_path, write_input):
    d = _load(tmp_path, write_input, TDA + "[dynamics]\ntmax_fs = 10.0\n").dynamics
    assert d is not None
    assert (d.tmax_fs, d.dt_fs, d.nsubsteps, d.coupling, d.hopping) == (
        10.0, 0.5, 25, "overlap", "sharc"
    )
    assert (d.ekin_correction, d.frustrated, d.decoherence, d.decoherence_param_Eh) == (
        "parallel_vel", "keep", "edc", 0.1
    )
    assert (d.edc_convention, d.substep_interpolation, d.energy_drift_limit_eV) == (
        "sharc", "midpoint", 0.5
    )
    assert (d.stop_after_s0_fs, d.overlap_threshold, d.max_step_halvings) == (None, 0.5, 3)
    assert (d.output_every, d.max_failed_fraction) == (1, 0.1)


def test_tmax_is_required(tmp_path, write_input):
    with pytest.raises(ConfigError, match="tmax_fs"):
        _load(tmp_path, write_input, TDA + "[dynamics]\ndt_fs = 0.5\n")


@pytest.mark.parametrize(
    ("lines", "needle"),
    [
        ("tmax_fs = 0.5\ndt_fs = 0.5\n", "dt_fs"),
        ("tmax_fs = 10.2\ndt_fs = 0.5\n", "integer multiple"),
        ('tmax_fs = 10.0\nhopping = "fssh"\n', "fssh"),
        ('tmax_fs = 10.0\nekin_correction = "parallel_nac"\n', "parallel_nac"),
        ("tmax_fs = 10.0\nmax_step_halvings = 7\n", "max_step_halvings"),
        ("tmax_fs = 10.0\noverlap_threshold = 1.0\n", "overlap_threshold"),
        ("tmax_fs = 10.0\nstop_after_s0_fs = 0.0\n", "stop_after_s0_fs"),
        ("tmax_fs = 10.0\neselect_Eh = 0.1\n", "eselect_Eh"),
        ('tmax_fs = 10.0\ncoupling = "nacdr"\n', "nacdr"),
    ],
)
def test_invalid_values_name_the_key(tmp_path, write_input, lines, needle):
    with pytest.raises(ConfigError, match=needle):
        _load(tmp_path, write_input, TDA + "[dynamics]\n" + lines)


def test_nacdr_and_parallel_nac_are_allowed_with_casscf(tmp_path, write_input):
    extra = (
        CAS + '[dynamics]\ntmax_fs = 10.0\ncoupling = "nacdr"\n'
        'ekin_correction = "parallel_nac"\nhopping = "fssh"\n'
    )
    d = _load(tmp_path, write_input, extra).dynamics
    assert d is not None and d.coupling == "nacdr"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/config/test_dynamics_section.py -q -W error`
Expected: FAIL (`dynamics` is an unknown top-level key / attribute missing).

- [ ] **Step 3: Implement**

In `schema.py`, after `SelectionSection`:

```python
class DynamicsStageSection(_Section):
    """[dynamics] of the workflow (P7b). Option meanings: USER_MANUAL.md section 4."""

    tmax_fs: float = Field(gt=0, allow_inf_nan=False)
    dt_fs: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    nsubsteps: int = Field(default=25, ge=1)
    coupling: Literal["overlap", "nacdr"] = "overlap"
    hopping: Literal["sharc", "fssh", "none"] = "sharc"
    ekin_correction: Literal["parallel_vel", "parallel_nac"] = "parallel_vel"
    frustrated: Literal["keep", "reflect"] = "keep"
    decoherence: Literal["edc", "none"] = "edc"
    decoherence_param_Eh: float = Field(default=0.1, gt=0, allow_inf_nan=False)
    edc_convention: Literal["sharc", "granucci_persico"] = "sharc"
    substep_interpolation: Literal["midpoint", "endpoint"] = "midpoint"
    energy_drift_limit_eV: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    stop_after_s0_fs: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    overlap_threshold: float = Field(default=0.5, gt=0, lt=1, allow_inf_nan=False)
    max_step_halvings: int = Field(default=3, ge=0, le=6)
    output_every: int = Field(default=1, ge=1)
    max_failed_fraction: float = Field(default=0.1, ge=0, lt=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def _consistent_options(self) -> DynamicsStageSection:
        problems: list[str] = []
        if not self.dt_fs < self.tmax_fs:
            problems.append(f"dynamics.dt_fs = {self.dt_fs} must be smaller than tmax_fs")
        else:
            ratio = self.tmax_fs / self.dt_fs
            if abs(ratio - round(ratio)) > 1e-9 * ratio:
                problems.append(
                    f"dynamics.tmax_fs = {self.tmax_fs} must be an integer multiple of "
                    f"dt_fs = {self.dt_fs}"
                )
        if self.hopping == "fssh" and self.coupling != "nacdr":
            problems.append("dynamics.hopping = 'fssh' needs coupling = 'nacdr'")
        if self.ekin_correction == "parallel_nac" and self.coupling != "nacdr":
            problems.append("dynamics.ekin_correction = 'parallel_nac' needs coupling = 'nacdr'")
        if problems:
            raise ValueError("; ".join(problems))
        return self
```

Check the class sits after `math` is imported (it is, `SelectionSection` uses it). Add to `ShmdConfig` after `selection`:

```python
    dynamics: DynamicsStageSection | None = None
```

In `ShmdConfig._consistent`, before `if problems:`:

```python
        if (
            self.dynamics is not None
            and self.dynamics.coupling == "nacdr"
            and self.electronic is not None
            and self.electronic.method != "casscf"
        ):
            problems.append(
                "dynamics.coupling = 'nacdr' needs analytic NACs, which exist for "
                "[electronic] method = 'casscf' only; use coupling = 'overlap'"
            )
```

`eselect_Eh` is rejected by `extra="forbid"` (the base `_Section`), which names the key; keep it so (SOC is not in scope).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/config/test_dynamics_section.py tests/config -q -W error`
Expected: PASS (and the existing config tests unchanged).

- [ ] **Step 5: Commit**

```bash
ruff format . && ruff check . && mypy
git add src/gqteashmd/config/schema.py tests/config/test_dynamics_section.py
git commit -m "feat: add the [dynamics] input section of the workflow"
```

---

### Task 2: `ModelBackend`, a fake electronic backend over an analytic model

**Files:**
- Create: `src/gqteashmd/testing/fake_backend.py`
- Test: `tests/trajectory/test_fake_backend.py`

**Interfaces:**
- Consumes: `ElectronicBackend`, `Request`, `ElectronicResult`, `Capabilities` (`electronic/base.py`); `ElectronicModel` (`dynamics/models.py`; here `LVCModel`); `states_from_counts`.
- Produces:
  - `class SimulatedCrash(BaseException)` — raised by the `"crash"` fault (a `BaseException` so no `except Exception` swallows it; it imitates a kill inside a test process).
  - `ModelBackend(model, *, faults: Mapping[int, str] | None = None, calls: int = 0, nacs: bool = True, marker_dir: Path | None = None)`; attribute `calls` (number of `compute` calls so far, 1-based numbering of the next call is `calls + 1`). Fault kinds, keyed by the 1-based compute call number: `"error"` (non-retryable `ElectronicStructureError`), `"retryable"` (retryable one), `"low_overlap"` (meta `overlap_min_singular_value = 0.1`, data unchanged), `"rung"` (meta `casscf = "SCF orbitals"`), `"no_overlaps"` (overlaps `None` although a previous geometry exists), `"crash"` (raise `SimulatedCrash`), `"kill"` (`os._exit(137)` once: only when `marker_dir/"killed_once"` does not exist, which it then creates first).
  - Meta of every successful compute: `{"backend": "model", "casscf": "previous orbitals" | "SCF orbitals"}` (`"SCF orbitals"` at a first geometry), plus `overlap_min_singular_value` (smallest singular value of the model's overlap matrix) when a previous geometry exists.

- [ ] **Step 1: Write the failing tests**

```python
"""The fake backend used by the fast tests of the dynamics stage."""

from pathlib import Path

import numpy as np
import pytest

from gqteashmd.core.states import StateLabel
from gqteashmd.dynamics.lvc import LVCModel
from gqteashmd.electronic.base import Request
from gqteashmd.errors import ElectronicStructureError
from gqteashmd.testing.fake_backend import ModelBackend, SimulatedCrash

LVC = Path(__file__).resolve().parents[1] / "data" / "lvc" / "lvc_singlets.template"
S0, S1, S2 = StateLabel(1, 0), StateLabel(1, 1), StateLabel(1, 2)


def _model():
    return LVCModel.from_template(LVC, spin_orbit=False)


def _coords(model, shift=0.0):
    q = np.array(model.reference_q, dtype=float)
    q[0] += shift
    return q.reshape(-1, 3)


def test_energies_gradients_and_nacs_are_the_models():
    model = _model()
    backend = ModelBackend(model)
    result = backend.compute(Request(_coords(model), gradients=frozenset({S1}), nacs=frozenset({(S0, S2)})))
    snap = model.evaluate(_coords(model).reshape(-1), None)
    assert result.states == (S0, S1, S2)
    assert [result.energies[s] for s in (S0, S1, S2)] == list(snap.energies)
    np.testing.assert_array_equal(result.gradients[S1].reshape(-1), snap.gradients[1])
    np.testing.assert_array_equal(result.nacs[(S0, S2)].reshape(-1), snap.nacs[0, 2])
    assert set(result.gradients) == {S1}
    assert result.overlaps is None and result.meta["casscf"] == "SCF orbitals"


def test_overlaps_follow_the_previous_geometry_and_state_round_trips(tmp_path):
    model = _model()
    backend = ModelBackend(model)
    backend.compute(Request(_coords(model)))
    backend.save_state(tmp_path / "e.npz")
    second = backend.compute(Request(_coords(model, 0.01), overlaps=True))
    other = ModelBackend(model)
    other.load_state(tmp_path / "e.npz")
    again = other.compute(Request(_coords(model, 0.01), overlaps=True))
    np.testing.assert_array_equal(second.overlaps, again.overlaps)
    assert second.meta["casscf"] == "previous orbitals"
    assert 0.9 < second.meta["overlap_min_singular_value"] <= 1.0 + 1e-12


def test_compute_more_returns_extra_gradients_at_the_last_geometry():
    model = _model()
    backend = ModelBackend(model)
    with pytest.raises(ValueError):
        backend.compute_more(gradients=frozenset({S2}))
    backend.compute(Request(_coords(model), gradients=frozenset({S1})))
    more = backend.compute_more(gradients=frozenset({S2}))
    snap = model.evaluate(_coords(model).reshape(-1), None)
    np.testing.assert_array_equal(more.gradients[S2].reshape(-1), snap.gradients[2])


@pytest.mark.parametrize("kind", ["error", "retryable"])
def test_error_faults(kind):
    model = _model()
    backend = ModelBackend(model, faults={2: kind})
    backend.compute(Request(_coords(model)))
    with pytest.raises(ElectronicStructureError) as info:
        backend.compute(Request(_coords(model, 0.01)))
    assert info.value.retryable is (kind == "retryable")
    with pytest.raises(ValueError):
        backend.compute_more(gradients=frozenset({S1}))  # a failed compute discards the last


def test_meta_faults_and_calls_offset():
    model = _model()
    backend = ModelBackend(model, faults={3: "low_overlap", 4: "rung", 5: "no_overlaps"}, calls=1)
    backend.compute(Request(_coords(model)))  # call 2
    r3 = backend.compute(Request(_coords(model, 0.01), overlaps=True))
    r4 = backend.compute(Request(_coords(model, 0.02), overlaps=True))
    r5 = backend.compute(Request(_coords(model, 0.03), overlaps=True))
    assert backend.calls == 5
    assert r3.meta["overlap_min_singular_value"] == 0.1 and r3.overlaps is not None
    assert r4.meta["casscf"] == "SCF orbitals"
    assert r5.overlaps is None


def test_crash_fault_and_capabilities():
    model = _model()
    backend = ModelBackend(model, faults={1: "crash"}, nacs=False)
    assert backend.capabilities.nacs is False and backend.capabilities.overlaps is True
    with pytest.raises(SimulatedCrash):
        backend.compute(Request(_coords(model)))


def test_reset_forgets_the_previous_geometry():
    model = _model()
    backend = ModelBackend(model)
    backend.compute(Request(_coords(model)))
    backend.reset()
    assert backend.compute(Request(_coords(model), overlaps=True)).overlaps is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/trajectory/test_fake_backend.py -q -W error`
Expected: FAIL (`ModuleNotFoundError: gqteashmd.testing.fake_backend`).

- [ ] **Step 3: Implement `src/gqteashmd/testing/fake_backend.py`**

```python
"""An ElectronicBackend over an analytic ElectronicModel, with scripted faults (tests only).

The model's flat coordinates are the backend's Cartesian coordinates (natom, 3) flattened. The
previous ElectronicSnapshot plays the role of PySCF's reference wavefunction: the model's own
``evaluate(q, previous)`` gives the phase-aligned states and the overlaps. Faults are keyed by
the 1-based number of the ``compute`` call (``calls`` sets the count already made, so a backend
built for a resumed run can continue the numbering of a crashed one).
"""

from __future__ import annotations

import io
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.core.states import StateLabel, states_from_counts
from gqteashmd.dynamics.models import ElectronicModel
from gqteashmd.dynamics.records import ElectronicSnapshot
from gqteashmd.electronic.base import Capabilities, ElectronicBackend, ElectronicResult, Request
from gqteashmd.errors import ElectronicStructureError
from gqteashmd.util.fileio import write_bytes_atomic

FAULTS = ("error", "retryable", "low_overlap", "rung", "no_overlaps", "crash", "kill")
_ARRAYS = ("energies", "gradients", "nacs", "hamiltonian", "vectors", "overlaps")


class SimulatedCrash(BaseException):
    """Stands in for a kill inside a test process (not an Exception, so nothing catches it)."""


class ModelBackend(ElectronicBackend):
    def __init__(
        self,
        model: ElectronicModel,
        *,
        faults: Mapping[int, str] | None = None,
        calls: int = 0,
        nacs: bool = True,
        marker_dir: Path | None = None,
    ) -> None:
        bad = sorted({kind for kind in (faults or {}).values() if kind not in FAULTS})
        if bad:
            raise ValueError(f"unknown fault kind(s) {bad}; known: {', '.join(FAULTS)}")
        self._model = model
        self._states = states_from_counts(int(model.n_states))
        self._faults = dict(faults or {})
        self.calls = int(calls)
        self._nacs = bool(nacs)
        self._marker_dir = None if marker_dir is None else Path(marker_dir)
        self._previous: ElectronicSnapshot | None = None
        self._live: ElectronicSnapshot | None = None

    @property
    def states(self) -> tuple[StateLabel, ...]:
        return self._states

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(gradients=True, nacs=self._nacs, overlaps=True, transition_dipoles=False)

    def compute(self, request: Request) -> ElectronicResult:
        self.calls += 1
        self._live = None
        fault = self._faults.get(self.calls)
        if fault == "crash":
            raise SimulatedCrash(f"simulated crash at compute call {self.calls}")
        if fault == "kill" and self._marker_dir is not None:
            marker = self._marker_dir / "killed_once"
            if not marker.exists():
                marker.write_text("killed")
                os._exit(137)
        if fault in ("error", "retryable"):
            raise ElectronicStructureError(
                f"scripted failure at compute call {self.calls}", retryable=fault == "retryable"
            )
        q = np.asarray(request.coords_bohr, dtype=float).reshape(-1)
        previous = self._previous
        snap = self._model.evaluate(q, previous)
        meta: dict[str, Any] = {"backend": "model", "casscf": "previous orbitals"}
        overlaps = None
        if previous is None:
            meta["casscf"] = "SCF orbitals"
        else:
            assert snap.overlaps is not None
            meta["overlap_min_singular_value"] = float(
                np.linalg.svd(np.asarray(snap.overlaps, dtype=float), compute_uv=False).min()
            )
            if request.overlaps and fault != "no_overlaps":
                overlaps = np.array(snap.overlaps, dtype=float)
        if fault == "low_overlap":
            meta["overlap_min_singular_value"] = 0.1
        if fault == "rung":
            meta["casscf"] = "SCF orbitals"
        self._previous = snap
        self._live = snap
        return self._result(snap, request.gradients, request.nacs, overlaps, meta)

    def compute_more(
        self,
        gradients: frozenset[StateLabel] = frozenset(),
        nacs: frozenset[tuple[StateLabel, StateLabel]] = frozenset(),
    ) -> ElectronicResult:
        if self._live is None:
            raise ValueError("compute_more() needs a successful preceding compute()")
        return self._result(self._live, gradients, nacs, None, {"backend": "model"})

    def _result(
        self,
        snap: ElectronicSnapshot,
        gradients: frozenset[StateLabel],
        nacs: frozenset[tuple[StateLabel, StateLabel]],
        overlaps: np.ndarray | None,
        meta: dict[str, Any],
    ) -> ElectronicResult:
        index = {label: i for i, label in enumerate(self._states)}
        natom = snap.gradients.shape[1] // 3
        return ElectronicResult(
            states=self._states,
            energies={label: float(snap.energies[i]) for label, i in index.items()},
            gradients={
                label: np.array(snap.gradients[index[label]]).reshape(natom, 3)
                for label in gradients
            },
            nacs={
                (a, b): np.array(snap.nacs[index[a], index[b]]).reshape(natom, 3)
                for a, b in nacs
            },
            overlaps=overlaps,
            phases=np.ones(len(self._states)),
            meta=meta,
        )

    def reset(self) -> None:
        self._previous = None
        self._live = None

    def save_state(self, path: Path) -> None:
        if self._previous is None:
            raise ValueError("there is no state to save before the first compute()")
        arrays = {
            name: np.asarray(getattr(self._previous, name))
            for name in _ARRAYS
            if getattr(self._previous, name) is not None
        }
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        write_bytes_atomic(Path(path), buffer.getvalue())

    def load_state(self, path: Path) -> None:
        with np.load(Path(path), allow_pickle=False) as data:
            values = {name: np.array(data[name]) for name in data.files}
        self._previous = ElectronicSnapshot(
            energies=values["energies"],
            gradients=values["gradients"],
            nacs=values["nacs"],
            hamiltonian=values["hamiltonian"],
            vectors=values.get("vectors"),
            overlaps=values.get("overlaps"),
        )
        self._live = None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/trajectory/test_fake_backend.py -q -W error`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
ruff format . && ruff check . && mypy
git add src/gqteashmd/testing/fake_backend.py tests/trajectory/
git commit -m "test: add a fake electronic backend over analytic models with scripted faults"
```

---

### Task 3: `BackendModel`, the core's ElectronicModel over a backend

**Files:**
- Create: `src/gqteashmd/trajectory/__init__.py` (docstring only), `src/gqteashmd/trajectory/outcomes.py`, `src/gqteashmd/trajectory/model.py`
- Test: `tests/trajectory/test_model.py`

**Interfaces:**
- Consumes: `ModelBackend` (Task 2); `driver.initial_state`, `driver.step`, `ReplaySource`, `ElectronicSnapshot`, `TrajectoryState` (core); `lowdin_orthonormalize`.
- Produces:
  - `gqteashmd.trajectory.outcomes.TrajectoryOutcome` (`StrEnum`): `FINISHED = "finished"`, `ENERGY_DRIFT = "energy_drift"`, `STOPPED_IN_S0 = "stopped_in_s0"`, `OVERLAP_FAILURE = "overlap_failure"`, `ELECTRONIC_FAILURE = "electronic_failure"`.
  - `gqteashmd.trajectory.model.BackendModel(backend, masses, *, coupling)`: attributes `backend`, `labels: tuple[StateLabel, ...]`, `n_states`, `n_dof`, `masses` (flat, m_e), `coupling`, `active: int` (set by the caller before every `evaluate`), `last_meta: dict`; properties `state_counts`, `min_singular_value: float | None`, `rung_changed: bool`; methods `evaluate(q, previous) -> ElectronicSnapshot`, `refresh_active_gradient(state) -> TrajectoryState`. Constant `FIRST_CASSCF_RUNG = "previous orbitals"`.

- [ ] **Step 1: Write the failing tests**

```python
"""BackendModel: the dynamics core over an electronic backend (spec section 3.1)."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from gqteashmd.dynamics.driver import initial_state, step
from gqteashmd.dynamics.lvc import LVCModel, lvc_wigner_samples
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.randomness import ReplaySource
from gqteashmd.overlaps.tracking import lowdin_orthonormalize
from gqteashmd.testing.fake_backend import ModelBackend
from gqteashmd.trajectory.model import BackendModel

LVC = Path(__file__).resolve().parents[1] / "data" / "lvc" / "lvc_singlets.template"
N_STEPS = 120


def _model():
    return LVCModel.from_template(LVC, spin_orbit=False)


class _Lowdin:
    """The direct model as BackendModel presents it with coupling = "overlap": the same Lowdin
    step on the overlaps, and zero NACs (BackendModel requests none for "overlap")."""

    def __init__(self, model):
        self._m = model
        self.n_states, self.n_dof, self.masses = model.n_states, model.n_dof, model.masses

    @property
    def state_counts(self):
        return self._m.state_counts

    def evaluate(self, q, previous):
        snap = self._m.evaluate(q, previous)
        snap = dataclasses.replace(snap, nacs=np.zeros_like(snap.nacs))
        if snap.overlaps is None:
            return snap
        return dataclasses.replace(snap, overlaps=lowdin_orthonormalize(snap.overlaps))


def _start(model):
    q, v = lvc_wigner_samples(model, 1, np.random.default_rng(11))
    return q[0], v[0]


def _direct(model, options, active):
    q0, v0 = _start(model)
    state = initial_state(model, q0, v0, active, options=options)
    source = ReplaySource([0.0] * N_STEPS)  # xi = 0: hop whenever a probability is > 0
    records = []
    for _ in range(N_STEPS):
        state, record = step(model, state, options, source)
        records.append(record)
    return records


def _adapted(model, options, active):
    q0, v0 = _start(model)
    adapter = BackendModel(ModelBackend(model), model.masses, coupling=options.coupling)
    adapter.active = active
    state = initial_state(adapter, q0, v0, active, options=options)
    source = ReplaySource([0.0] * N_STEPS)
    records = []
    for _ in range(N_STEPS):
        adapter.active = state.active
        state, record = step(adapter, state, options, source)
        if record.hop_to is not None:
            state = adapter.refresh_active_gradient(state)
        records.append(record)
    return records


@pytest.mark.parametrize("coupling", ["nacdr", "overlap"])
def test_backend_model_reproduces_the_direct_model_bit_for_bit(coupling):
    options = DynamicsOptions(dt_au=10.0, coupling=coupling)
    model = _model()
    direct = _direct(model if coupling == "nacdr" else _Lowdin(model), options, 2)
    adapted = _adapted(model, options, 2)
    assert any(r.hop_to is not None for r in direct)
    for a, b in zip(direct, adapted, strict=True):
        for name in ("q", "v", "c", "energies", "probabilities"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        assert (a.active, a.e_total, a.hop_from, a.hop_to, a.frustrated) == (
            b.active, b.e_total, b.hop_from, b.hop_to, b.frustrated
        )


def test_only_the_active_gradient_is_computed_and_no_nan_reaches_a_record():
    records = _adapted(_model(), DynamicsOptions(dt_au=10.0, coupling="overlap"), 2)
    assert any(r.hop_to is not None for r in records)
    for r in records:
        for name in ("q", "v", "c", "energies", "probabilities"):
            assert np.all(np.isfinite(getattr(r, name)))
        assert np.isfinite(r.e_total)


def test_snapshot_layout():
    model = _model()
    adapter = BackendModel(ModelBackend(model), model.masses, coupling="nacdr")
    adapter.active = 1
    snap = adapter.evaluate(np.array(model.reference_q), None)
    assert np.all(np.isfinite(snap.gradients[1]))
    assert np.all(np.isnan(snap.gradients[[0, 2]]))
    np.testing.assert_array_equal(snap.nacs, -snap.nacs.transpose(1, 0, 2))
    np.testing.assert_array_equal(snap.hamiltonian, np.diag(snap.energies).astype(complex))
    assert snap.overlaps is None and adapter.min_singular_value is None
    assert adapter.rung_changed is False  # a first geometry is never a rung change


def test_meta_signals():
    model = _model()
    backend = ModelBackend(model, faults={2: "low_overlap", 3: "rung"})
    adapter = BackendModel(backend, model.masses, coupling="overlap")
    q = np.array(model.reference_q)
    first = adapter.evaluate(q, None)
    adapter.evaluate(q + 0.001, first)
    assert adapter.min_singular_value == 0.1 and adapter.rung_changed is False
    adapter.evaluate(q + 0.002, first)
    assert adapter.rung_changed is True


def test_missing_overlaps_with_a_previous_geometry_is_an_error():
    model = _model()
    adapter = BackendModel(ModelBackend(model, faults={2: "no_overlaps"}), model.masses, coupling="overlap")
    q = np.array(model.reference_q)
    first = adapter.evaluate(q, None)
    with pytest.raises(ValueError, match="no overlaps"):
        adapter.evaluate(q + 0.001, first)


def test_construction_errors():
    model = _model()
    with pytest.raises(ValueError, match="nacdr"):
        BackendModel(ModelBackend(model, nacs=False), model.masses, coupling="nacdr")
    with pytest.raises(ValueError, match="coupling"):
        BackendModel(ModelBackend(model), model.masses, coupling="ddr")
    with pytest.raises(ValueError, match="masses"):
        BackendModel(ModelBackend(model), model.masses[:-1], coupling="overlap")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/trajectory/test_model.py -q -W error`
Expected: FAIL (`ModuleNotFoundError: gqteashmd.trajectory`).

- [ ] **Step 3: Implement**

`src/gqteashmd/trajectory/__init__.py`:

```python
"""Trajectories of the dynamics stage: the backend adapter, the runner and its files (P7b)."""
```

`src/gqteashmd/trajectory/outcomes.py`:

```python
"""How a trajectory of the dynamics stage ended (spec section 5); every outcome is a DONE task."""

from __future__ import annotations

from enum import StrEnum


class TrajectoryOutcome(StrEnum):
    FINISHED = "finished"
    ENERGY_DRIFT = "energy_drift"
    STOPPED_IN_S0 = "stopped_in_s0"
    OVERLAP_FAILURE = "overlap_failure"
    ELECTRONIC_FAILURE = "electronic_failure"
```

`src/gqteashmd/trajectory/model.py`:

```python
"""BackendModel: the dynamics core's ElectronicModel over an ElectronicBackend (spec 3.1).

Singlets only (U stays None: the MCH Hamiltonian is diagonal). Each evaluation requests the
energies, the gradient of the active state only (the caller sets ``active`` before every call),
the overlaps with the previous geometry (always) and, for ``coupling = "nacdr"``, every NAC pair.
The other gradient rows are NaN, a tripwire: the core reads only the active row (the gradient of
the old active state at q' for the velocities, and the new one at the start of the next step,
which ``refresh_active_gradient`` supplies after a hop). The overlap matrix is Lowdin-
orthonormalised, because the propagator assumes a unitary S; the raw smallest singular value
stays in ``last_meta`` for the step-rejection test.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from gqteashmd.core.states import StateLabel
from gqteashmd.dynamics.records import ElectronicSnapshot, TrajectoryState
from gqteashmd.electronic.base import ElectronicBackend, ElectronicResult, Request
from gqteashmd.overlaps.tracking import lowdin_orthonormalize

FIRST_CASSCF_RUNG = "previous orbitals"
COUPLINGS = ("overlap", "nacdr")


class BackendModel:
    def __init__(self, backend: ElectronicBackend, masses: np.ndarray, *, coupling: str) -> None:
        labels = tuple(backend.states)
        if any(label.multiplicity != 1 for label in labels):
            raise ValueError("the dynamics stage supports singlet states only (O7)")
        if len(labels) < 2:
            raise ValueError("dynamics needs at least two electronic states")
        if coupling not in COUPLINGS:
            raise ValueError(f"coupling must be one of {', '.join(COUPLINGS)}; got {coupling!r}")
        if coupling == "nacdr" and not backend.capabilities.nacs:
            raise ValueError(
                "coupling = 'nacdr' needs analytic NACs (SA-CASSCF); use coupling = 'overlap'"
            )
        flat = np.asarray(masses, dtype=float).reshape(-1)
        if flat.size == 0 or flat.size % 3 or not np.all(np.isfinite(flat) & (flat > 0)):
            raise ValueError("masses must be positive, finite and 3 per atom")
        self.backend = backend
        self.labels: tuple[StateLabel, ...] = labels
        self.n_states = len(labels)
        self.n_dof = int(flat.size)
        self.masses = flat
        self.coupling = coupling
        self.active = 0
        self.last_meta: dict[str, Any] = {}
        self._had_previous = False
        self._pairs = (
            frozenset((a, b) for i, a in enumerate(labels) for b in labels[i + 1 :])
            if coupling == "nacdr"
            else frozenset()
        )

    @property
    def state_counts(self) -> tuple[int, ...]:
        return (self.n_states,)

    @property
    def min_singular_value(self) -> float | None:
        value = self.last_meta.get("overlap_min_singular_value")
        return None if value is None else float(value)

    @property
    def rung_changed(self) -> bool:
        """True when CASSCF left its first ladder rung at a geometry with a previous one (R4)."""
        rung = self.last_meta.get("casscf", FIRST_CASSCF_RUNG)
        return self._had_previous and rung != FIRST_CASSCF_RUNG

    def evaluate(self, q: np.ndarray, previous: ElectronicSnapshot | None) -> ElectronicSnapshot:
        flat = np.asarray(q, dtype=float).reshape(-1)
        if flat.shape != (self.n_dof,):
            raise ValueError(f"q must have {self.n_dof} components; got {flat.size}")
        if not 0 <= self.active < self.n_states:
            raise ValueError(f"active must be in 0..{self.n_states - 1}; got {self.active}")
        self.last_meta = {}
        self._had_previous = previous is not None
        result = self.backend.compute(
            Request(
                flat.reshape(-1, 3),
                gradients=frozenset({self.labels[self.active]}),
                nacs=self._pairs,
                overlaps=True,
            )
        )
        self.last_meta = dict(result.meta)
        if previous is not None and result.overlaps is None:
            raise ValueError("the backend returned no overlaps with the previous geometry")
        return self._snapshot(result)

    def _snapshot(self, result: ElectronicResult) -> ElectronicSnapshot:
        n, f = self.n_states, self.n_dof
        index = {label: i for i, label in enumerate(self.labels)}
        energies = np.array([float(result.energies[label]) for label in self.labels])
        gradients = np.full((n, f), np.nan)
        for label, g in result.gradients.items():
            gradients[index[label]] = np.asarray(g, dtype=float).reshape(-1)
        nacs = np.zeros((n, n, f))
        for (a, b), d in result.nacs.items():
            i, j = index[a], index[b]
            nacs[i, j] = np.asarray(d, dtype=float).reshape(-1)
            nacs[j, i] = -nacs[i, j]
        overlaps = None if result.overlaps is None else lowdin_orthonormalize(result.overlaps)
        return ElectronicSnapshot(
            energies=energies,
            gradients=gradients,
            nacs=nacs,
            hamiltonian=np.diag(energies).astype(complex),
            overlaps=overlaps,
        )

    def refresh_active_gradient(self, state: TrajectoryState) -> TrajectoryState:
        """The state with the gradient of its (new) active state computed at its geometry."""
        label = self.labels[state.active]
        result = self.backend.compute_more(gradients=frozenset({label}))
        gradients = np.array(state.snapshot.gradients, copy=True)
        gradients[state.active] = np.asarray(result.gradients[label], dtype=float).reshape(-1)
        snapshot = dataclasses.replace(state.snapshot, gradients=gradients)
        return dataclasses.replace(state, snapshot=snapshot)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/trajectory/test_model.py tests/dynamics/test_golden.py -q -W error`
Expected: PASS. If the bit-for-bit test fails for `overlap`, check that the `_Lowdin` wrapper and `BackendModel` apply `lowdin_orthonormalize` to the same matrix; do not loosen the test.

- [ ] **Step 5: Commit**

```bash
ruff format . && ruff check . && mypy
git add src/gqteashmd/trajectory/ tests/trajectory/test_model.py
git commit -m "feat: add BackendModel, the dynamics core's model over an electronic backend"
```

---

### Task 4: Trajectory files: step log, restart generations, `output.h5`

**Files:**
- Create: `src/gqteashmd/trajectory/store.py`
- Test: `tests/trajectory/test_store.py`

**Interfaces:**
- Produces:
  - `log_dtype(n_states: int, n_dof: int) -> np.dtype` with fields, in order: `step` i8, `grid_step` i8, `ticks` i8, `depth` i8, `time_au` f8, `active` i8, `hop_from` i8 (-1 = none), `hop_to` i8 (-1 = none), `frustrated` i1, `rung_changed` i1, `rejections` i8, `random_number` f8 (NaN at step 0), `min_singular_value` f8 (NaN when unknown), `e_total` f8, `e_kinetic` f8, `energies` f8 (n,), `c_real` f8 (n,), `c_imag` f8 (n,), `probabilities` f8 (n,), `q` f8 (f,), `v` f8 (f,). All little-endian.
  - `log_row(dtype, **values) -> np.ndarray` (a 0-d structured array; unspecified fields are zero).
  - `StepLog(path, n_states, n_dof)`: `.dtype`, `len()`, `append(row)` (write + flush + fsync), `truncate(n_records)` (`ValueError` if the file holds fewer complete records), `read() -> np.ndarray` (complete records only).
  - `RestartStore(directory)`: `current() -> Path | None`, `write(fill: Callable[[Path], None]) -> Path`, `clean_temporary()`. Generations `gen_00000000`, `gen_00000001`, ...; marker file `current`.
  - `write_output_h5(path, rows, *, labels: Sequence[str], n_atoms: int, output_every: int, attrs: Mapping[str, Any]) -> None` (atomic; selection rule R2).

- [ ] **Step 1: Write the failing tests**

```python
"""Files of one trajectory (spec section 6)."""

import numpy as np
import pytest

from gqteashmd.trajectory.store import RestartStore, StepLog, log_dtype, log_row, write_output_h5


def _row(dtype, step, *, grid_step=None, ticks=0, hop_to=-1, rejections=0):
    return log_row(
        dtype,
        step=step,
        grid_step=step if grid_step is None else grid_step,
        ticks=ticks,
        time_au=10.0 * step,
        hop_from=-1,
        hop_to=hop_to,
        rejections=rejections,
        energies=[0.0, 0.1],
        c_real=[1.0, 0.0],
        q=np.arange(6.0) + step,
        v=np.ones(6),
    )


def test_log_append_read_and_truncate(tmp_path):
    log = StepLog(tmp_path / "steps.log", 2, 6)
    assert len(log) == 0
    for k in range(4):
        log.append(_row(log.dtype, k))
    assert len(log) == 4
    rows = log.read()
    assert list(rows["step"]) == [0, 1, 2, 3]
    np.testing.assert_array_equal(rows["q"][2], np.arange(6.0) + 2)
    with open(log.path, "ab") as fh:  # a record cut by a kill
        fh.write(b"\x01" * 5)
    assert len(log) == 4
    log.truncate(3)
    assert len(log) == 3 and log.path.stat().st_size == 3 * log.dtype.itemsize
    with pytest.raises(ValueError, match="records"):
        log.truncate(4)


def test_dtype_fields():
    dtype = log_dtype(3, 9)
    assert dtype["energies"].shape == (3,) and dtype["q"].shape == (9,)
    assert dtype.names[:4] == ("step", "grid_step", "ticks", "depth")


def test_restart_generations(tmp_path):
    store = RestartStore(tmp_path / "restart")
    assert store.current() is None
    first = store.write(lambda d: (d / "a.txt").write_text("1"))
    second = store.write(lambda d: (d / "a.txt").write_text("2"))
    assert first.name == "gen_00000000" and second.name == "gen_00000001"
    assert store.current() == second and not first.exists()
    assert (second / "a.txt").read_text() == "2"


def test_a_kill_inside_a_generation_keeps_the_previous_one(tmp_path):
    store = RestartStore(tmp_path / "restart")
    good = store.write(lambda d: (d / "a.txt").write_text("1"))

    def broken(directory):
        (directory / "a.txt").write_text("half")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        store.write(broken)
    assert store.current() == good and (good / "a.txt").read_text() == "1"
    assert list(store.directory.glob(".gen_*.tmp"))
    store.clean_temporary()
    assert not list(store.directory.glob(".gen_*.tmp"))
    (store.directory / "gen_00000001").mkdir()  # renamed, but `current` was never updated
    again = store.write(lambda d: (d / "a.txt").write_text("3"))
    assert again.name == "gen_00000001" and (again / "a.txt").read_text() == "3"


def test_a_bad_current_marker_is_an_error(tmp_path):
    store = RestartStore(tmp_path / "restart")
    store.directory.mkdir()
    (store.directory / "current").write_text("gen_00000009\n")
    with pytest.raises(ValueError, match="restart generation"):
        store.current()


def test_output_h5_selection_and_units(tmp_path):
    h5py = pytest.importorskip("h5py")
    dtype = log_dtype(2, 6)
    rows = np.stack(
        [
            _row(dtype, 0),
            _row(dtype, 1),
            _row(dtype, 2, hop_to=1),
            _row(dtype, 3, grid_step=2, ticks=4, rejections=1),
            _row(dtype, 4, grid_step=3),
            _row(dtype, 5, grid_step=4),
            _row(dtype, 6, grid_step=5),
        ]
    )
    path = tmp_path / "output.h5"
    write_output_h5(path, rows, labels=["S0", "S1"], n_atoms=2, output_every=2,
                    attrs={"outcome": "finished"})
    with h5py.File(path, "r") as f:
        # grid points 0, 2, 4 kept; the hop (step 2) and the rejected interval (step 3) kept;
        # the last record (step 6) kept
        assert list(f["step"][()]) == [0, 2, 3, 5, 6]
        assert f["positions"].shape == (5, 2, 3)
        for name, dataset in f.items():
            assert "units" in dataset.attrs, name
        assert f["time_fs"].attrs["units"] == "fs"
        assert f.attrs["outcome"] == "finished"
        assert list(f.attrs["state_labels"]) == ["S0", "S1"]
        np.testing.assert_allclose(f["populations"][0], [1.0, 0.0])


def test_output_h5_of_an_empty_log(tmp_path):
    h5py = pytest.importorskip("h5py")
    write_output_h5(tmp_path / "o.h5", np.zeros(0, log_dtype(2, 6)), labels=["S0", "S1"],
                    n_atoms=2, output_every=1, attrs={})
    with h5py.File(tmp_path / "o.h5", "r") as f:
        assert f["step"].shape == (0,) and f["positions"].shape == (0, 2, 3)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/trajectory/test_store.py -q -W error`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `src/gqteashmd/trajectory/store.py`**

```python
"""Files of one trajectory (spec section 6, rulings R2 and R3).

steps.log: append-only fixed-size binary records, one per accepted (sub)step and one for t = 0;
the restart's meta.json stores how many records belong to it, so a resume truncates the records
written after the last restart (and any record cut by a kill). restart/: generations gen_<n>,
each written into a temporary directory and renamed; the small file `current` (written
atomically) names the newest complete one; older ones are deleted only after `current` points
to the new one, so a kill at any instant leaves one complete generation. output.h5 is built from
the log when a trajectory ends (h5py imported inside the function).
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.core.units import AU_TIME_IN_FS
from gqteashmd.util.fileio import write_text_atomic

_GENERATION = re.compile(r"gen_(\d{8})")


def log_dtype(n_states: int, n_dof: int) -> np.dtype:
    n, f = int(n_states), int(n_dof)
    return np.dtype(
        [
            ("step", "<i8"),
            ("grid_step", "<i8"),
            ("ticks", "<i8"),
            ("depth", "<i8"),
            ("time_au", "<f8"),
            ("active", "<i8"),
            ("hop_from", "<i8"),
            ("hop_to", "<i8"),
            ("frustrated", "i1"),
            ("rung_changed", "i1"),
            ("rejections", "<i8"),
            ("random_number", "<f8"),
            ("min_singular_value", "<f8"),
            ("e_total", "<f8"),
            ("e_kinetic", "<f8"),
            ("energies", "<f8", (n,)),
            ("c_real", "<f8", (n,)),
            ("c_imag", "<f8", (n,)),
            ("probabilities", "<f8", (n,)),
            ("q", "<f8", (f,)),
            ("v", "<f8", (f,)),
        ]
    )


def log_row(dtype: np.dtype, **values: Any) -> np.ndarray:
    row = np.zeros((), dtype=dtype)
    for name, value in values.items():
        row[name] = value
    return row


class StepLog:
    def __init__(self, path: Path, n_states: int, n_dof: int) -> None:
        self.path = Path(path)
        self.dtype = log_dtype(n_states, n_dof)

    def __len__(self) -> int:
        if not self.path.exists():
            return 0
        return self.path.stat().st_size // self.dtype.itemsize

    def append(self, row: np.ndarray) -> None:
        data = np.asarray(row, dtype=self.dtype).reshape(1).tobytes()
        with open(self.path, "ab") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())

    def truncate(self, n_records: int) -> None:
        have = len(self)
        if n_records > have:
            raise ValueError(
                f"{self.path} holds {have} complete records, but the restart expects "
                f"{n_records}"
            )
        with open(self.path, "ab"):
            pass
        os.truncate(self.path, n_records * self.dtype.itemsize)

    def read(self) -> np.ndarray:
        n = len(self)
        if n == 0:
            return np.zeros(0, dtype=self.dtype)
        return np.fromfile(self.path, dtype=self.dtype, count=n)


class RestartStore:
    CURRENT = "current"

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def current(self) -> Path | None:
        marker = self.directory / self.CURRENT
        if not marker.exists():
            return None
        name = marker.read_text(encoding="utf-8").strip()
        path = self.directory / name
        if not _GENERATION.fullmatch(name) or not path.is_dir():
            raise ValueError(f"{marker} names {name!r}, which is not a complete restart generation")
        return path

    def write(self, fill: Callable[[Path], None]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        current = self.current()
        number = 0 if current is None else int(current.name[4:]) + 1
        name = f"gen_{number:08d}"
        tmp = self.directory / f".{name}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        fill(tmp)
        final = self.directory / name
        shutil.rmtree(final, ignore_errors=True)  # renamed by a run killed before `current`
        os.replace(tmp, final)
        write_text_atomic(self.directory / self.CURRENT, name + "\n")
        for old in self.directory.glob("gen_*"):
            if old.name != name:
                shutil.rmtree(old, ignore_errors=True)
        return final

    def clean_temporary(self) -> None:
        for tmp in self.directory.glob(".gen_*.tmp"):
            shutil.rmtree(tmp, ignore_errors=True)


def _selected(rows: np.ndarray, output_every: int) -> np.ndarray:
    keep = (rows["ticks"] == 0) & (rows["grid_step"] % output_every == 0)
    keep |= rows["hop_to"] >= 0
    keep |= rows["frustrated"] != 0
    keep |= rows["rung_changed"] != 0
    keep |= rows["rejections"] > 0
    if rows.size:
        keep[-1] = True
    return rows[keep]


def write_output_h5(
    path: Path,
    rows: np.ndarray,
    *,
    labels: Sequence[str],
    n_atoms: int,
    output_every: int,
    attrs: Mapping[str, Any],
) -> None:
    import h5py

    path = Path(path)
    kept = _selected(rows, output_every)
    m = kept.shape[0]
    populations = kept["c_real"] ** 2 + kept["c_imag"] ** 2
    datasets: list[tuple[str, np.ndarray, str]] = [
        ("time_fs", kept["time_au"] * AU_TIME_IN_FS, "fs"),
        ("step", kept["step"], "-"),
        ("grid_step", kept["grid_step"], "-"),
        ("ticks", kept["ticks"], "-"),
        ("depth", kept["depth"], "-"),
        ("active", kept["active"], "state index (0 = S0)"),
        ("hop_from", kept["hop_from"], "state index (-1 = no hop)"),
        ("hop_to", kept["hop_to"], "state index (-1 = no hop)"),
        ("frustrated", kept["frustrated"], "bool"),
        ("rung_changed", kept["rung_changed"], "bool"),
        ("rejections", kept["rejections"], "-"),
        ("random_number", kept["random_number"], "-"),
        ("min_singular_value", kept["min_singular_value"], "-"),
        ("energies", kept["energies"], "Eh"),
        ("e_total", kept["e_total"], "Eh"),
        ("e_kinetic", kept["e_kinetic"], "Eh"),
        ("populations", populations, "-"),
        ("coefficients_real", kept["c_real"], "-"),
        ("coefficients_imag", kept["c_imag"], "-"),
        ("probabilities", kept["probabilities"], "-"),
        ("positions", kept["q"].reshape(m, n_atoms, 3), "bohr"),
        ("velocities", kept["v"].reshape(m, n_atoms, 3), "bohr per atomic unit of time"),
    ]
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["schema_version"] = 1
            f.attrs["state_labels"] = [str(label) for label in labels]
            f.attrs["output_every"] = int(output_every)
            for key, value in attrs.items():
                f.attrs[key] = value
            for name, data, units in datasets:
                f.create_dataset(name, data=np.asarray(data)).attrs["units"] = units
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
```

Note: `attrs` values must be HDF5-storable (str, int, float, list of str); the runner passes only such values and never `None` (it omits keys whose value is `None`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/trajectory/test_store.py -q -W error`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
ruff format . && ruff check . && mypy
git add src/gqteashmd/trajectory/store.py tests/trajectory/test_store.py
git commit -m "feat: add the step log, restart generations and output.h5 of a trajectory"
```

---

### Task 5: The trajectory runner (steps, rejection with halving, outcomes, restart, resume, extension)

**Files:**
- Create: `src/gqteashmd/trajectory/runner.py`
- Test: `tests/trajectory/test_runner.py`

**Interfaces:**
- Consumes: `BackendModel`, `TrajectoryOutcome` (Task 3); `StepLog`, `RestartStore`, `log_row`, `write_output_h5` (Task 4); `ModelBackend`, `SimulatedCrash` (Task 2, tests); core `initial_state`, `step`, `kinetic_energy`, `save_state`, `load_state`, `source_from_state`, `GeneratorSource`, `ReplaySource`, `DynamicsOptions`.
- Produces:
  - `RunSettings(options: DynamicsOptions, tmax_au: float, overlap_threshold: float = 0.5, max_step_halvings: int = 3, output_every: int = 1, stop_after_s0_au: float | None = None)`; properties `n_grid_steps: int`, `ticks_per_step: int` (= 2**max_step_halvings). `ValueError` when `tmax_au / options.dt_au` is not an integer >= 1 to 1e-9 (relative), or any other field is out of range.
  - `InitialConditions(q: np.ndarray, v: np.ndarray, active: int)` (flat bohr; bohr per atomic time unit; MCH index).
  - `Progress` (dataclass stored as the restart's `meta.json`): `grid_step, ticks, depth, n_steps, n_hops, n_frustrated, n_rejections, n_flagged, pending_rejections, max_drift_au, s0_time_au, log_records, outcome, message`; `to_json()`, `from_json(data)`.
  - `TrajectoryRunner(model: BackendModel, settings: RunSettings, directory: Path, *, attrs: Mapping[str, Any] | None = None)` with `run(initial: InitialConditions, source: RandomSource) -> dict[str, Any]`. The result dict: `outcome` (str), `t_final_fs`, `n_grid_steps`, `n_steps`, `n_hops`, `n_frustrated`, `n_rejections`, `n_flagged`, `max_drift_eV` (`None` when non-finite), `message` (str or `None`), `wall_time_s`.
  - File names: `STATE_FILE = "state.npz"`, `ELECTRONIC_FILE = "electronic.npz"`, `META_FILE = "meta.json"`, `LOG_FILE = "steps.log"`, `OUTPUT_FILE = "output.h5"`, `PROGRESS_FILE = "progress.json"`.

**Algorithm (binding; spec sections 3.2, 4, 5, 6 and rulings R2-R10):**

- `run`: create the directory; `clean_temporary`; resume if `restart/current` exists, else start. On resume: a stored `finished` with `grid_step < n_grid_steps` is reopened (O16: outcome cleared); any other stored outcome returns at once. Loop while no outcome: at `ticks == 0 and grid_step >= n_grid_steps` set `finished`, save, stop; else `_advance`. Finally write `output.h5` and `progress.json` and return the result.
- **Start:** `log.truncate(0)`, `backend.reset()`, `model.active = initial.active`, `initial_state(model, q, v, active, options=options)`. A non-retryable `ElectronicStructureError` gives `electronic_failure` at t = 0, no restart (R6); a retryable one propagates. Append the t = 0 record (`random_number` NaN, `min_singular_value` NaN, `hop_from = hop_to = -1`, `e_total = e_total_initial`, `e_kinetic = kinetic_energy(v, masses)`, `probabilities` zero), save.
- **`_advance`:** `dt = options.dt_au / 2**depth` (`dataclasses.replace(options, dt_au=dt)` when `depth > 0`); `saved = source.state()`; `model.active = state.active`; call `step`; on a hop call `refresh_active_gradient` (R7: its failure is an electronic signal). Signals: non-retryable `ElectronicStructureError` → `"electronic"`; `min_singular_value` not `>= overlap_threshold` → `"overlap"`; `rung_changed` → `"rung"`. A `"rung"` signal at `depth == max_step_halvings` is accepted and flagged. Any other signal: roll back (`backend.load_state(current/electronic.npz)`, `source = source_from_state(saved)`), `n_rejections += 1`, `pending_rejections += 1`; if `depth < max_step_halvings`, `depth += 1` and return the old state (no save); else set `overlap_failure` or `electronic_failure` with a message naming the time, save the old state, return.
- **Accept:** `depth_used = depth`; `n_steps += 1`; `ticks += ticks_per_step >> depth`; at `ticks == ticks_per_step` set `grid_step += 1, ticks = 0, depth = 0`; count hops, frustrated hops, flags; drift `|e_total - e_total_initial|` (non-finite counts as infinite); `s0_time_au += dt` while the active state is S0 (index 0), else 0; append the record (time `= (grid_step + ticks / ticks_per_step) * dt_full`, `depth = depth_used`, `rejections = pending_rejections`, then `pending_rejections = 0`); then outcome checks: drift not `<= energy_drift_limit_au` → `energy_drift`; else `stop_after_s0_au` set, `ticks == 0` and `s0_time_au >= stop_after_s0_au * (1 - 1e-12)` → `stopped_in_s0`; save.
- **Save:** write a restart generation with `save_state(gen/state.npz, state, source)`, `backend.save_state(gen/electronic.npz)` and `meta.json` (`Progress.to_json()`); then `progress.json` (atomic): `{"grid_step", "time_fs", "active", "outcome"}`. The log record is appended **before** the generation is written.
- **Resume:** `load_state(gen/state.npz, model)`, `backend.load_state(gen/electronic.npz)`, `Progress.from_json`, `log.truncate(progress.log_records)`.

- [ ] **Step 1: Write the failing tests**

```python
"""The trajectory runner on the fake backend (fast tier; spec sections 3.2-6)."""

import json
from pathlib import Path

import numpy as np
import pytest

from gqteashmd.core.units import AU_TIME_IN_FS
from gqteashmd.dynamics.driver import initial_state, step
from gqteashmd.dynamics.lvc import LVCModel, lvc_wigner_samples
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.randomness import GeneratorSource, ReplaySource
from gqteashmd.dynamics.restart import load_state
from gqteashmd.errors import ElectronicStructureError
from gqteashmd.testing.fake_backend import ModelBackend, SimulatedCrash
from gqteashmd.trajectory import runner as runner_module
from gqteashmd.trajectory.model import BackendModel
from gqteashmd.trajectory.runner import InitialConditions, RunSettings, TrajectoryRunner
from gqteashmd.trajectory.store import RestartStore, StepLog

LVC = Path(__file__).resolve().parents[1] / "data" / "lvc" / "lvc_singlets.template"
DT = 10.0
N = 30


def _model():
    return LVCModel.from_template(LVC, spin_orbit=False)


def _initial(model, active=2):
    q, v = lvc_wigner_samples(model, 1, np.random.default_rng(11))
    return InitialConditions(q=q[0], v=v[0], active=active)


def _settings(n=N, **kwargs):
    options = DynamicsOptions(dt_au=DT, coupling="overlap", **kwargs.pop("options", {}))
    return RunSettings(options, tmax_au=n * DT, **kwargs)


def _run(tmp_path, *, name="a", settings=None, faults=None, calls=0, source=None, active=2):
    model = _model()
    adapter = BackendModel(ModelBackend(model, faults=faults, calls=calls), model.masses,
                           coupling="overlap")
    runner = TrajectoryRunner(adapter, settings or _settings(), tmp_path / name)
    if source is None:
        source = GeneratorSource(np.random.default_rng(5))
    return runner.run(_initial(model, active), source), tmp_path / name


def _log(directory):
    return StepLog(directory / "steps.log", 3, 9).read()


def test_a_run_finishes_and_writes_its_files(tmp_path):
    h5py = pytest.importorskip("h5py")
    result, d = _run(tmp_path)
    assert result["outcome"] == "finished" and result["n_grid_steps"] == N
    assert result["t_final_fs"] == pytest.approx(N * DT * AU_TIME_IN_FS)
    rows = _log(d)
    assert len(rows) == N + 1 and list(rows["grid_step"]) == list(range(N + 1))
    with h5py.File(d / "output.h5", "r") as f:
        assert f["step"].shape == (N + 1,)
    assert json.loads((d / "progress.json").read_text())["outcome"] == "finished"


def test_the_runner_equals_the_manual_core_loop(tmp_path):
    model = _model()
    adapter = BackendModel(ModelBackend(model), model.masses, coupling="overlap")
    init = _initial(model)
    options = _settings().options
    adapter.active = init.active
    state = initial_state(adapter, init.q, init.v, init.active, options=options)
    source = GeneratorSource(np.random.default_rng(5))
    expected = []
    for _ in range(N):
        adapter.active = state.active
        state, record = step(adapter, state, options, source)
        if record.hop_to is not None:
            state = adapter.refresh_active_gradient(state)
        expected.append(record)
    _, d = _run(tmp_path)
    rows = _log(d)[1:]
    for row, record in zip(rows, expected, strict=True):
        np.testing.assert_array_equal(row["q"], record.q)
        np.testing.assert_array_equal(row["c_real"] + 1j * row["c_imag"], record.c)
        assert (row["active"], row["e_total"], row["random_number"]) == (
            record.active, record.e_total, record.random_number
        )


@pytest.mark.parametrize("crash_call", [2, 11, 25])
def test_a_crash_and_resume_is_bit_identical(tmp_path, crash_call):
    _, reference = _run(tmp_path, name="ref")
    with pytest.raises(SimulatedCrash):
        _run(tmp_path, name="b", faults={crash_call: "crash"})
    result, d = _run(tmp_path, name="b", calls=crash_call - 1)
    assert result["outcome"] == "finished"
    assert _log(d).tobytes() == _log(reference).tobytes()


def test_a_kill_between_the_log_and_the_restart_is_bit_identical(tmp_path, monkeypatch):
    _, reference = _run(tmp_path, name="ref")
    original = RestartStore.write
    count = {"n": 0}

    def flaky(self, fill):
        count["n"] += 1
        if count["n"] == 10:
            raise SimulatedCrash("killed after the log record, before the restart")
        return original(self, fill)

    monkeypatch.setattr(RestartStore, "write", flaky)
    with pytest.raises(SimulatedCrash):
        _run(tmp_path, name="b")
    monkeypatch.setattr(RestartStore, "write", original)
    calls = 10  # compute calls made before the kill: step 0 and steps 1..9
    _run(tmp_path, name="b", calls=calls)
    assert _log(tmp_path / "b").tobytes() == _log(reference).tobytes()


def test_extension_continues_a_finished_trajectory(tmp_path):
    _, reference = _run(tmp_path, name="ref", settings=_settings(2 * N))
    first, d = _run(tmp_path, name="b", settings=_settings(N))
    assert first["outcome"] == "finished"
    again, _ = _run(tmp_path, name="b", settings=_settings(N))
    assert again["n_steps"] == first["n_steps"]  # nothing recomputed
    longer, _ = _run(tmp_path, name="b", settings=_settings(2 * N))
    assert longer["outcome"] == "finished" and longer["n_grid_steps"] == 2 * N
    assert _log(d).tobytes() == _log(reference).tobytes()


def test_energy_drift_outcome(tmp_path):
    result, d = _run(tmp_path, settings=_settings(options={"energy_drift_limit_au": 1e-14}))
    assert result["outcome"] == "energy_drift" and result["n_grid_steps"] == 1
    again, _ = _run(tmp_path, settings=_settings(options={"energy_drift_limit_au": 1e-14}))
    assert again == {**result, "wall_time_s": again["wall_time_s"]}  # a final outcome stays


def test_stopped_in_s0_outcome(tmp_path):
    settings = _settings(stop_after_s0_au=3 * DT, options={"hopping": "none"})  # stays in S0
    result, _ = _run(tmp_path, settings=settings, active=0)
    assert result["outcome"] == "stopped_in_s0" and result["n_grid_steps"] == 3


def test_a_failure_at_step_0(tmp_path):
    h5py = pytest.importorskip("h5py")
    result, d = _run(tmp_path, faults={1: "error"})
    assert result["outcome"] == "electronic_failure" and result["t_final_fs"] == 0.0
    assert "step 0" in result["message"]
    assert RestartStore(d / "restart").current() is None
    with h5py.File(d / "output.h5", "r") as f:
        assert f["step"].shape == (0,)


def test_a_retryable_error_propagates_and_resumes(tmp_path):
    _, reference = _run(tmp_path, name="ref")
    with pytest.raises(ElectronicStructureError) as info:
        _run(tmp_path, name="b", faults={4: "retryable"})
    assert info.value.retryable
    _run(tmp_path, name="b", calls=3)
    assert _log(tmp_path / "b").tobytes() == _log(reference).tobytes()


def test_one_low_overlap_halves_one_interval(tmp_path):
    result, d = _run(tmp_path, faults={6: "low_overlap"})
    assert result["outcome"] == "finished" and result["n_rejections"] == 1
    rows = _log(d)
    assert len(rows) == N + 2  # one interval in two halves
    half, whole = rows[5], rows[6]
    assert (half["grid_step"], half["ticks"], half["depth"], half["rejections"]) == (4, 4, 1, 1)
    assert (whole["grid_step"], whole["ticks"], whole["depth"]) == (5, 0, 1)
    assert half["time_au"] == 4.5 * DT and whole["time_au"] == 5 * DT
    assert rows[7]["depth"] == 0


def test_a_rejected_step_returns_its_random_number(tmp_path):
    values = list(np.linspace(0.01, 0.99, 200))
    result, d = _run(tmp_path, faults={6: "low_overlap"}, source=ReplaySource(values))
    rows = _log(d)
    np.testing.assert_array_equal(rows["random_number"][1:], values[: result["n_steps"]])


@pytest.mark.parametrize(("kind", "outcome"), [("low_overlap", "overlap_failure"),
                                               ("error", "electronic_failure")])
def test_a_persistent_signal_ends_the_trajectory(tmp_path, kind, outcome):
    result, d = _run(tmp_path, faults={c: kind for c in (6, 7, 8, 9)})
    assert result["outcome"] == outcome and result["n_rejections"] == 4
    assert result["n_grid_steps"] == 4 and "fs" in result["message"]
    state, _ = load_state(RestartStore(d / "restart").current() / "state.npz",
                          BackendModel(ModelBackend(_model()), _model().masses,
                                       coupling="overlap"))
    assert state.step == 4  # the restart holds the last accepted state


def test_a_persistent_rung_change_is_accepted_at_the_smallest_step(tmp_path):
    result, d = _run(tmp_path, faults={c: "rung" for c in (6, 7, 8, 9)})
    assert result["outcome"] == "finished"
    assert (result["n_rejections"], result["n_flagged"]) == (3, 1)
    rows = _log(d)
    flagged = rows[rows["rung_changed"] != 0]
    assert len(flagged) == 1 and flagged[0]["depth"] == 3 and flagged[0]["ticks"] == 1


def test_a_resume_inside_a_halved_interval_is_bit_identical(tmp_path):
    _, reference = _run(tmp_path, name="ref", faults={6: "low_overlap"})
    with pytest.raises(SimulatedCrash):
        _run(tmp_path, name="b", faults={6: "low_overlap", 8: "crash"})
    _run(tmp_path, name="b", faults={6: "low_overlap"}, calls=7)
    assert _log(tmp_path / "b").tobytes() == _log(reference).tobytes()


def test_hops_leave_a_finite_active_gradient_in_the_restart(tmp_path):
    result, d = _run(tmp_path, source=ReplaySource([0.0] * 200))
    assert result["n_hops"] > 0
    rows = _log(d)
    assert np.all(np.isfinite(rows["e_total"])) and np.all(np.isfinite(rows["q"]))
    model = BackendModel(ModelBackend(_model()), _model().masses, coupling="overlap")
    state, _ = load_state(RestartStore(d / "restart").current() / "state.npz", model)
    assert np.all(np.isfinite(state.snapshot.gradients[state.active]))


def test_settings_validation():
    options = DynamicsOptions(dt_au=DT)
    with pytest.raises(ValueError, match="multiple"):
        RunSettings(options, tmax_au=10.5 * DT)
    with pytest.raises(ValueError, match="max_step_halvings"):
        RunSettings(options, tmax_au=10 * DT, max_step_halvings=7)
    assert RunSettings(options, tmax_au=10 * DT).ticks_per_step == 8
    assert runner_module.LOG_FILE == "steps.log"
```

The call numbers in these tests: call 1 is step 0; call k + 1 is the attempt from grid step k - 1... more precisely, without faults call `c` computes the step that ends at grid step `c - 1`. With `{6: "low_overlap"}`, the step from grid step 4 is rejected at call 6, calls 7 and 8 are its two halves, call 9 is the step from grid step 5. The implementer checks these against the algorithm above; if a number in a test is inconsistent with the algorithm, the test is wrong (fix the test, state it in the report), the algorithm is binding.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/trajectory/test_runner.py -q -W error`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `src/gqteashmd/trajectory/runner.py`**

```python
"""One trajectory of the dynamics stage (spec sections 3.2-6, plan rulings R2-R10).

The core's ``driver.step`` advances one (sub)step; this module owns everything around it: the
integer time grid, step rejection with halving, outcomes, the step log, per-step restart
generations, resume and extension. It reimplements no part of the step order.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.core.units import AU_TIME_IN_FS, HARTREE_IN_EV
from gqteashmd.dynamics.driver import initial_state, kinetic_energy, step
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.randomness import RandomSource, source_from_state
from gqteashmd.dynamics.records import StepRecord, TrajectoryState
from gqteashmd.dynamics.restart import load_state, save_state
from gqteashmd.errors import ElectronicStructureError
from gqteashmd.trajectory.model import BackendModel
from gqteashmd.trajectory.outcomes import TrajectoryOutcome
from gqteashmd.trajectory.store import RestartStore, StepLog, log_row, write_output_h5
from gqteashmd.util.fileio import read_json, write_json_atomic

STATE_FILE = "state.npz"
ELECTRONIC_FILE = "electronic.npz"
META_FILE = "meta.json"
LOG_FILE = "steps.log"
OUTPUT_FILE = "output.h5"
PROGRESS_FILE = "progress.json"
GRID_TOLERANCE = 1e-9
MAX_HALVINGS = 6


@dataclass(frozen=True)
class RunSettings:
    options: DynamicsOptions
    tmax_au: float
    overlap_threshold: float = 0.5
    max_step_halvings: int = 3
    output_every: int = 1
    stop_after_s0_au: float | None = None

    def __post_init__(self) -> None:
        ratio = float(self.tmax_au) / self.options.dt_au
        if not math.isfinite(ratio) or round(ratio) < 1 or abs(ratio - round(ratio)) > (
            GRID_TOLERANCE * ratio
        ):
            raise ValueError(
                f"tmax_au = {self.tmax_au} must be an integer multiple (>= 1) of "
                f"dt_au = {self.options.dt_au}"
            )
        if not 0 < self.overlap_threshold < 1:
            raise ValueError(f"overlap_threshold must be in (0, 1); got {self.overlap_threshold}")
        if not 0 <= self.max_step_halvings <= MAX_HALVINGS:
            raise ValueError(
                f"max_step_halvings must be in 0..{MAX_HALVINGS}; got {self.max_step_halvings}"
            )
        if self.output_every < 1:
            raise ValueError(f"output_every must be >= 1; got {self.output_every}")
        if self.stop_after_s0_au is not None and not self.stop_after_s0_au > 0:
            raise ValueError(f"stop_after_s0_au must be > 0; got {self.stop_after_s0_au}")

    @property
    def n_grid_steps(self) -> int:
        return int(round(float(self.tmax_au) / self.options.dt_au))

    @property
    def ticks_per_step(self) -> int:
        return 1 << self.max_step_halvings


@dataclass(frozen=True)
class InitialConditions:
    q: np.ndarray
    v: np.ndarray
    active: int


@dataclass
class Progress:
    grid_step: int = 0
    ticks: int = 0
    depth: int = 0
    n_steps: int = 0
    n_hops: int = 0
    n_frustrated: int = 0
    n_rejections: int = 0
    n_flagged: int = 0
    pending_rejections: int = 0
    max_drift_au: float = 0.0
    s0_time_au: float = 0.0
    log_records: int = 0
    outcome: str | None = None
    message: str | None = None

    def to_json(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        if not math.isfinite(self.max_drift_au):
            data["max_drift_au"] = None  # strict JSON: None means non-finite
        return data

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Progress:
        values = dict(data)
        if values.get("max_drift_au") is None:
            values["max_drift_au"] = math.inf
        return cls(**values)


class TrajectoryRunner:
    def __init__(
        self,
        model: BackendModel,
        settings: RunSettings,
        directory: Path,
        *,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.settings = settings
        self.directory = Path(directory)
        self.attrs = dict(attrs or {})
        self.restarts = RestartStore(self.directory / "restart")
        self.log = StepLog(self.directory / LOG_FILE, model.n_states, model.n_dof)

    # ---------------------------------------------------------------- public
    def run(self, initial: InitialConditions, source: RandomSource) -> dict[str, Any]:
        started = time.perf_counter()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.restarts.clean_temporary()
        resumed = self._resume()
        state: TrajectoryState | None
        if resumed is None:
            state, source, progress = self._start(initial, source)
        else:
            state, source, progress = resumed
            if (
                progress.outcome == TrajectoryOutcome.FINISHED
                and progress.grid_step < self.settings.n_grid_steps
            ):
                progress.outcome = None  # O16: tmax was raised after the trajectory finished
                progress.message = None
        while progress.outcome is None and state is not None:
            if progress.ticks == 0 and progress.grid_step >= self.settings.n_grid_steps:
                progress.outcome = TrajectoryOutcome.FINISHED
                self._save(state, source, progress)
                break
            state, source = self._advance(state, source, progress)
        return self._finish(progress, time.perf_counter() - started)

    # ---------------------------------------------------------------- start and resume
    def _start(
        self, initial: InitialConditions, source: RandomSource
    ) -> tuple[TrajectoryState | None, RandomSource, Progress]:
        self.log.truncate(0)
        progress = Progress()
        self.model.backend.reset()
        self.model.active = int(initial.active)
        try:
            state = initial_state(
                self.model,
                initial.q,
                initial.v,
                int(initial.active),
                options=self.settings.options,
            )
        except ElectronicStructureError as exc:
            if exc.retryable:
                raise
            progress.outcome = TrajectoryOutcome.ELECTRONIC_FAILURE
            progress.message = f"step 0: {exc}"
            return None, source, progress
        self._append(state, None, progress, depth=0, flagged=False)
        self._save(state, source, progress)
        return state, source, progress

    def _resume(self) -> tuple[TrajectoryState, RandomSource, Progress] | None:
        generation = self.restarts.current()
        if generation is None:
            return None
        state, source = load_state(generation / STATE_FILE, self.model)
        self.model.backend.load_state(generation / ELECTRONIC_FILE)
        progress = Progress.from_json(read_json(generation / META_FILE))
        self.log.truncate(progress.log_records)
        return state, source, progress

    # ---------------------------------------------------------------- one (sub)step
    def _advance(
        self, state: TrajectoryState, source: RandomSource, progress: Progress
    ) -> tuple[TrajectoryState, RandomSource]:
        settings = self.settings
        dt = settings.options.dt_au / (1 << progress.depth)
        options = (
            settings.options
            if progress.depth == 0
            else dataclasses.replace(settings.options, dt_au=dt)
        )
        saved = source.state()
        self.model.active = state.active
        signal: str | None = None
        message = ""
        try:
            new_state, record = step(self.model, state, options, source)
            if record.hop_to is not None:
                new_state = self.model.refresh_active_gradient(new_state)
        except ElectronicStructureError as exc:
            if exc.retryable:
                raise
            signal, message = "electronic", str(exc)
        else:
            sv = self.model.min_singular_value
            if sv is not None and not sv >= settings.overlap_threshold:
                signal = "overlap"
                message = (
                    f"smallest singular value of the state overlap matrix {sv:.3g} is below "
                    f"overlap_threshold = {settings.overlap_threshold}"
                )
            elif self.model.rung_changed:
                signal = "rung"
                message = f"CASSCF ladder rung {self.model.last_meta.get('casscf')!r}"
        flagged = False
        if signal == "rung" and progress.depth >= settings.max_step_halvings:
            signal, flagged = None, True
        if signal is not None:
            source = self._rollback(saved)
            progress.n_rejections += 1
            progress.pending_rejections += 1
            if progress.depth < settings.max_step_halvings:
                progress.depth += 1
                return state, source
            progress.outcome = (
                TrajectoryOutcome.OVERLAP_FAILURE
                if signal == "overlap"
                else TrajectoryOutcome.ELECTRONIC_FAILURE
            )
            when = self._time_au(progress) * AU_TIME_IN_FS
            progress.message = f"t = {when:.4f} fs, step dt/{1 << progress.depth}: {message}"
            self._save(state, source, progress)
            return state, source
        return self._accept(new_state, record, source, progress, dt, flagged), source

    def _rollback(self, saved: dict[str, Any]) -> RandomSource:
        generation = self.restarts.current()
        assert generation is not None  # a step is only attempted after a restart was written
        self.model.backend.load_state(generation / ELECTRONIC_FILE)
        return source_from_state(saved)

    def _accept(
        self,
        state: TrajectoryState,
        record: StepRecord,
        source: RandomSource,
        progress: Progress,
        dt: float,
        flagged: bool,
    ) -> TrajectoryState:
        settings = self.settings
        depth_used = progress.depth
        progress.n_steps += 1
        progress.ticks += settings.ticks_per_step >> progress.depth
        if progress.ticks >= settings.ticks_per_step:
            progress.grid_step += 1
            progress.ticks = 0
            progress.depth = 0
        progress.n_hops += record.hop_to is not None
        progress.n_frustrated += bool(record.frustrated)
        progress.n_flagged += flagged
        drift = abs(record.e_total - state.e_total_initial)
        if not math.isfinite(drift):
            drift = math.inf
        progress.max_drift_au = max(progress.max_drift_au, drift)
        progress.s0_time_au = progress.s0_time_au + dt if state.active == 0 else 0.0
        self._append(state, record, progress, depth=depth_used, flagged=flagged)
        if not drift <= settings.options.energy_drift_limit_au:
            progress.outcome = TrajectoryOutcome.ENERGY_DRIFT
            progress.message = (
                f"|E_tot - E_tot(0)| = {drift * HARTREE_IN_EV:.4g} eV exceeds "
                f"energy_drift_limit_eV = {settings.options.energy_drift_limit_au * HARTREE_IN_EV:.4g}"
            )
        elif (
            settings.stop_after_s0_au is not None
            and progress.ticks == 0
            and progress.s0_time_au >= settings.stop_after_s0_au * (1 - 1e-12)
        ):
            progress.outcome = TrajectoryOutcome.STOPPED_IN_S0
        self._save(state, source, progress)
        return state

    # ---------------------------------------------------------------- files
    def _time_au(self, progress: Progress) -> float:
        s = self.settings
        return (progress.grid_step + progress.ticks / s.ticks_per_step) * s.options.dt_au

    def _append(
        self,
        state: TrajectoryState,
        record: StepRecord | None,
        progress: Progress,
        *,
        depth: int,
        flagged: bool,
    ) -> None:
        n = self.model.n_states
        sv = self.model.min_singular_value if record is not None else None
        if record is None:
            values: dict[str, Any] = {
                "step": 0,
                "active": state.active,
                "hop_from": -1,
                "hop_to": -1,
                "random_number": math.nan,
                "e_total": state.e_total_initial,
                "e_kinetic": kinetic_energy(state.v, self.model.masses),
                "energies": state.snapshot.energies,
                "c_real": state.c.real,
                "c_imag": state.c.imag,
                "probabilities": np.zeros(n),
                "q": state.q,
                "v": state.v,
            }
        else:
            values = {
                "step": progress.n_steps,
                "active": record.active,
                "hop_from": -1 if record.hop_from is None else record.hop_from,
                "hop_to": -1 if record.hop_to is None else record.hop_to,
                "frustrated": int(record.frustrated),
                "random_number": record.random_number,
                "e_total": record.e_total,
                "e_kinetic": record.e_kinetic,
                "energies": record.energies,
                "c_real": record.c.real,
                "c_imag": record.c.imag,
                "probabilities": record.probabilities,
                "q": record.q,
                "v": record.v,
            }
        values.update(
            grid_step=progress.grid_step,
            ticks=progress.ticks,
            depth=depth,
            time_au=self._time_au(progress),
            rung_changed=int(flagged),
            rejections=progress.pending_rejections,
            min_singular_value=math.nan if sv is None else sv,
        )
        self.log.append(log_row(self.log.dtype, **values))
        progress.log_records += 1
        progress.pending_rejections = 0

    def _save(self, state: TrajectoryState, source: RandomSource, progress: Progress) -> None:
        def fill(directory: Path) -> None:
            save_state(directory / STATE_FILE, state, source)
            self.model.backend.save_state(directory / ELECTRONIC_FILE)
            write_json_atomic(directory / META_FILE, progress.to_json())

        self.restarts.write(fill)
        self._write_progress(progress, state.active)

    def _write_progress(self, progress: Progress, active: int | None) -> None:
        write_json_atomic(
            self.directory / PROGRESS_FILE,
            {
                "grid_step": progress.grid_step,
                "time_fs": self._time_au(progress) * AU_TIME_IN_FS,
                "active": None if active is None else str(self.model.labels[active]),
                "outcome": None if progress.outcome is None else str(progress.outcome),
            },
        )

    def _finish(self, progress: Progress, elapsed: float) -> dict[str, Any]:
        outcome = str(progress.outcome)
        attrs = {**self.attrs, "outcome": outcome, "dt_fs": self.settings.options.dt_au * AU_TIME_IN_FS}
        if progress.message is not None:
            attrs["message"] = progress.message
        write_output_h5(
            self.directory / OUTPUT_FILE,
            self.log.read(),
            labels=[str(label) for label in self.model.labels],
            n_atoms=self.model.n_dof // 3,
            output_every=self.settings.output_every,
            attrs=attrs,
        )
        rows = self.log.read()
        self._write_progress(progress, int(rows["active"][-1]) if rows.size else None)
        drift = progress.max_drift_au
        return {
            "outcome": outcome,
            "t_final_fs": self._time_au(progress) * AU_TIME_IN_FS,
            "n_grid_steps": progress.grid_step,
            "n_steps": progress.n_steps,
            "n_hops": progress.n_hops,
            "n_frustrated": progress.n_frustrated,
            "n_rejections": progress.n_rejections,
            "n_flagged": progress.n_flagged,
            "max_drift_eV": drift * HARTREE_IN_EV if math.isfinite(drift) else None,
            "message": progress.message,
            "wall_time_s": elapsed,
        }
```

Notes for the implementer: (1) `self.attrs` values must not be `None` (h5py cannot store it). (2) In `_append` for step 0 `min_singular_value` is NaN by construction. (3) Keep the order "append the log record, then write the generation" — the kill test depends on it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/trajectory -q -W error` then `pytest tests/dynamics/test_golden.py -q -W error`
Expected: PASS. Then the fast tier: `pytest -m "not slow and not pyscf" -W error -q` → all pass (record the count in the report).

- [ ] **Step 5: Commit**

```bash
ruff format . && ruff check . && mypy
git add src/gqteashmd/trajectory/runner.py tests/trajectory/test_runner.py
git commit -m "feat: add the trajectory runner with step rejection, outcomes, restart and resume"
```

---

### Task 6: Orchestrator hook to reopen tasks of a DONE stage (O16)

**Files:**
- Modify: `src/gqteashmd/stages/base.py` (add `reopen_tasks`), `src/gqteashmd/orchestrator.py` (`_run_pipeline` DONE branch, new `_reopen`), `src/gqteashmd/testing/fake_stages.py` (run counter, `reopen` behaviour)
- Test: `tests/test_orchestrator.py` (append)

**Interfaces:**
- Produces: `Stage.reopen_tasks(self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]) -> list[str]` (default `[]`). The orchestrator calls it for a DONE stage with an unchanged fingerprint, with the results of its DONE tasks; returned ids go back to PENDING with 0 attempts and the stage runs again (only those tasks run; `finalize` runs again with all results). Fake behaviour key `reopen`: list of `"stage:task_id"`; `FakeStage.reopen_tasks` returns those whose result has `runs < 2`; `reopen_bad`: list of stage names whose `reopen_tasks` returns `["t999"]`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_orchestrator.py`)

```python
def test_reopened_task_runs_again_and_finalize_repeats(project, behaviour):
    _run(project)
    stage_dir = _run_dir(project) / "04_dynamics"
    summary = stage_dir / "summary.json"
    before = summary.stat().st_mtime_ns
    behaviour(reopen=["dynamics:t001"])
    result = _run(project)
    assert result.completed == ["dynamics"]
    runs = {
        t: read_json(stage_dir / "tasks" / t / "result.json")["result"]["runs"]
        for t in ("t000", "t001", "t002", "t003")
    }
    assert runs == {"t000": 1, "t001": 2, "t002": 1, "t003": 1}
    assert summary.stat().st_mtime_ns > before
    third = _run(project)  # runs == 2 now: nothing to reopen
    assert "dynamics" in third.up_to_date


def test_reopening_an_unknown_task_is_a_project_error(project, behaviour):
    _run(project)
    behaviour(reopen_bad=["dynamics"])
    with pytest.raises(ProjectError, match="t999"):
        _run(project)
```

Check the stage directory index of the fake `dynamics` stage in `make_pipeline` (prepare 0, optimize 1, frequencies 2, sampling 3, dynamics 4) and adjust the name if the layout differs.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_orchestrator.py -q -W error -k reopen`
Expected: FAIL (`runs` key missing / no reopen).

- [ ] **Step 3: Implement**

`stages/base.py`, inside `Stage` after `fingerprint_inputs`:

```python
    def reopen_tasks(
        self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]
    ) -> list[str]:
        """Ids of DONE tasks to run again although the stage is DONE and up to date (O16).

        Called by the controller for a DONE stage whose fingerprint is unchanged, with the
        results of its DONE tasks; e.g. `dynamics` reopens finished trajectories when
        `tmax_fs` (not hashed) was raised. Must be deterministic and read no upstream artifacts.
        """
        return []
```

`orchestrator.py`, replace the DONE branch in `_run_pipeline`:

```python
            if record is not None and record.status == StageStatus.DONE:
                reopened = self._reopen(state, stage, ctx)
                if not reopened:
                    summary.up_to_date.append(stage.name)
                    upstream_dirs[stage.name] = stage_dir
                    log.info("Stage %s: up to date", stage.name)
                    continue
                log.info(
                    "Stage %s: reopening %d task(s): %s",
                    stage.name,
                    len(reopened),
                    ", ".join(reopened),
                )
```

and add the method next to `_collect_results`:

```python
    def _reopen(self, state: StateStore, stage: Stage, ctx: StageContext) -> list[str]:
        records = state.get_tasks(stage.name)
        results = self._collect_results(stage, ctx, records)
        ids = list(stage.reopen_tasks(ctx, results))
        unknown = sorted(set(ids) - set(results))
        if unknown:
            raise ProjectError(
                f"Stage {stage.name!r} asked to reopen tasks that are not done: "
                f"{', '.join(unknown)}"
            )
        attempts = {r.task_id: r.attempts for r in records}
        for task_id in ids:
            state.set_task(
                stage.name,
                task_id,
                TaskStatus.PENDING,
                message="reopened",
                attempts_delta=-attempts[task_id],
            )
        return ids
```

`testing/fake_stages.py`: document the two keys in the module docstring; in `run_task`, before the `return`, count runs:

```python
        counter = task_dir / "runs.txt"
        runs = int(counter.read_text()) + 1 if counter.exists() else 1
        counter.write_text(str(runs))
```

and add `"runs": runs` to the returned dict; add the method:

```python
    def reopen_tasks(
        self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]
    ) -> list[str]:
        behaviour = _behaviour()
        if self.name in behaviour.get("reopen_bad", []):
            return ["t999"]
        wanted = {
            key.split(":", 1)[1]
            for key in behaviour.get("reopen", [])
            if key.split(":", 1)[0] == self.name
        }
        return sorted(t for t in wanted if t in results and int(results[t].get("runs", 1)) < 2)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_orchestrator.py tests/stages/test_base_pipeline.py -q -W error`
Expected: PASS (all existing orchestrator tests too).

- [ ] **Step 5: Commit**

```bash
ruff format . && ruff check . && mypy
git add src/gqteashmd/stages/base.py src/gqteashmd/orchestrator.py src/gqteashmd/testing/fake_stages.py tests/test_orchestrator.py
git commit -m "feat: let a DONE stage reopen tasks (extension of finished work, O16)"
```

---

### Task 7: `DynamicsStage`

**Files:**
- Create: `src/gqteashmd/stages/dynamics.py`, `src/gqteashmd/testing/canned_lvc.py`
- Test: `tests/stages/test_dynamics.py`

**Interfaces:**
- Consumes: Tasks 1-6; `prepared_molecule`, `stage_backend` (`stages/geometry.py`); `read_samples_h5`; `task_rng`; `GeneratorSource`.
- Produces:
  - `stages/dynamics.py`: `BackendFactory = Callable[[StageContext, MoleculeSpec, Path], ElectronicBackend]`; `run_settings(section: DynamicsStageSection) -> RunSettings`; `class DynamicsStage(Stage)` with `name = "dynamics"`, `version = 1`, `depends_on = ("prepare", "sampling", "selection")`, `config_sections = ("electronic", "dynamics")`, `required_modules = ("pyscf>=2.6", "h5py>=3.9")`, `__init__(self, *, max_failed_fraction: float = 0.1, backend_factory: BackendFactory | None = None, required_modules: tuple[str, ...] | None = None)`, `enabled`, `check_config`, `fingerprint_inputs`, `plan`, `run_task`, `finalize`, `reopen_tasks`. Task ids `traj_<index:05d>`.
  - `testing/canned_lvc.py`: `LVC_TEMPLATE_ENV = "GQTEASHMD_CANNED_LVC_TEMPLATE"`, `FAULTS_ENV = "GQTEASHMD_CANNED_FAULTS"` (JSON `{"traj_00001": {"5": "kill"}}`), `canned_model() -> LVCModel` (template from the environment variable, `spin_orbit=False`), `write_canned_molecule(stage_dir, model)`, `write_canned_samples(stage_dir, model, n_samples, seed=0)`, `write_canned_trajectories(stage_dir, entries: list[tuple[int, str]], project_seed: int)`, `lvc_backend_factory(ctx, molecule, task_dir) -> ModelBackend` (faults from `FAULTS_ENV` for `task_dir.name`, `marker_dir=task_dir`).

`run_task` steps (binding):
1. Read `samples.h5` (`read_samples_h5`; `ValueError` → non-retryable `TaskError`). Check `1 <= sample <= n_samples` (else non-retryable `TaskError` naming `--rerun-from sampling`) and that `samples["symbols"] == molecule.symbols` (else `TaskError` naming `--rerun-from sampling`).
2. Check `task.payload["rng"] == {"stage": "dynamics", "task_index": index}` (else non-retryable `TaskError` naming `--rerun-from selection`).
3. Build the backend with the factory (default: `stage_backend(ctx, "dynamics", molecule, task_dir, settings=ctx.config.electronic)`), `masses = np.repeat(samples["masses_amu"] * DALTON_IN_ME, 3)`, `BackendModel(backend, masses, coupling=section.coupling)` (`ValueError` → non-retryable `TaskError`).
4. `active = model.labels.index(StateLabel.parse(state))` (`ValueError` → non-retryable `TaskError`).
5. `TrajectoryRunner(model, run_settings(section), task_dir, attrs={"symbols": [...], "sample": sample, "initial_state": state, "task": task.task_id})`; `run(InitialConditions(positions[sample - 1].reshape(-1), velocities[sample - 1].reshape(-1), active), GeneratorSource(task_rng(seed, "dynamics", index)))`. A `ValueError` from the runner (corrupt or mismatched restart, log shorter than the restart) → non-retryable `TaskError` naming `--rerun-from dynamics`.
6. Return the runner's dict plus `sample`, `initial_state`.

`finalize` writes `dynamics_summary.json` (strict JSON) with `schema_version` 1, `n_results`, `outcomes` (count per outcome, all five keys present), `tmax_fs`, `dt_fs`, `totals` (`n_hops`, `n_frustrated`, `n_rejections`, `n_flagged`), `max_drift_eV` (largest finite value or `None`), `missing_output` (task ids without `output.h5`, R5), `trajectories` (one row per task, sorted by id: task, sample, initial_state, outcome, t_final_fs, n_hops, n_rejections, n_flagged, max_drift_eV, message), and `dynamics_summary.md` (a table of the same rows and the counts). No results at all → `StageFailedError`. A missing `output.h5` → WARNING naming the tasks.

- [ ] **Step 1: Write the failing tests** (`tests/stages/test_dynamics.py`)

```python
"""DynamicsStage on the fake LVC backend, called directly (no orchestrator)."""

import json
from pathlib import Path

import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import StageFailedError, TaskError
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.stages.base import StageContext
from gqteashmd.stages.dynamics import DynamicsStage, run_settings
from gqteashmd.testing import canned_lvc

LVC = Path(__file__).resolve().parents[1] / "data" / "lvc" / "lvc_singlets.template"
ELECTRONIC = '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 3\n'


@pytest.fixture(autouse=True)
def _template(monkeypatch):
    monkeypatch.setenv(canned_lvc.LVC_TEMPLATE_ENV, str(LVC))


def _context(tmp_path, write_input, dynamics="tmax_fs = 2.0\ndt_fs = 0.25\n", seed=7):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = write_input(tmp_path, extra=ELECTRONIC + "[dynamics]\n" + dynamics)
    loaded = load_config(path).config
    cfg = loaded.model_copy(update={"project": loaded.project.model_copy(update={"seed": seed})})
    layout = ProjectLayout(tmp_path / "run")
    model = canned_lvc.canned_model()
    dirs = {name: tmp_path / "run" / name for name in ("00_prepare", "01_sampling", "02_selection")}
    for d in dirs.values():
        d.mkdir(parents=True)
    canned_lvc.write_canned_molecule(dirs["00_prepare"], model)
    canned_lvc.write_canned_samples(dirs["01_sampling"], model, 4)
    canned_lvc.write_canned_trajectories(dirs["02_selection"], [(1, "S1"), (2, "S2")], seed)
    upstream = {"prepare": dirs["00_prepare"], "sampling": dirs["01_sampling"],
                "selection": dirs["02_selection"]}
    stage_dir = tmp_path / "run" / "03_dynamics"
    stage_dir.mkdir()
    return StageContext(cfg, layout, stage_dir, upstream)


def _stage():
    return DynamicsStage(backend_factory=canned_lvc.lvc_backend_factory,
                         required_modules=("h5py>=3.9",))


def _run_all(ctx, stage):
    results = {}
    for task in stage.plan(ctx):
        task_dir = ctx.stage_dir / "tasks" / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        results[task.task_id] = stage.run_task(ctx, task, task_dir)
    return results


def test_plan_run_and_finalize(tmp_path, write_input):
    ctx = _context(tmp_path, write_input)
    stage = _stage()
    tasks = stage.plan(ctx)
    assert [t.task_id for t in tasks] == ["traj_00000", "traj_00001"]
    assert tasks[1].payload["sample"] == 2 and tasks[1].payload["state"] == "S2"
    results = _run_all(ctx, stage)
    assert {r["outcome"] for r in results.values()} == {"finished"}
    assert results["traj_00000"]["t_final_fs"] == pytest.approx(2.0)
    stage.finalize(ctx, results)
    summary = json.loads((ctx.stage_dir / "dynamics_summary.json").read_text())
    assert summary["outcomes"]["finished"] == 2 and summary["missing_output"] == []
    assert (ctx.stage_dir / "dynamics_summary.md").exists()


def test_the_seed_reaches_the_trajectories(tmp_path, write_input):
    a = _run_all(_context(tmp_path / "a", write_input, seed=7), _stage())
    b = _run_all(_context(tmp_path / "b", write_input, seed=7), _stage())
    c = _run_all(_context(tmp_path / "c", write_input, seed=8), _stage())
    read = lambda root: (root / "run/03_dynamics/tasks/traj_00001/steps.log").read_bytes()  # noqa: E731
    assert read(tmp_path / "a") == read(tmp_path / "b") != read(tmp_path / "c")


def test_fingerprint_and_reopen(tmp_path, write_input):
    ctx = _context(tmp_path, write_input)
    stage = _stage()
    fp = stage.fingerprint_inputs(ctx)
    assert "tmax_fs" not in fp["dynamics"] and "max_failed_fraction" not in fp["dynamics"]
    assert fp["dynamics"]["output_every"] == 1 and fp["project_seed"] == 7
    assert fp["electronic"]["overlap_norm_kept"] == 0.9999
    results = _run_all(ctx, stage)
    assert stage.reopen_tasks(ctx, results) == []
    longer = ctx.config.dynamics.model_copy(update={"tmax_fs": 3.0})
    ctx2 = StageContext(ctx.config.model_copy(update={"dynamics": longer}), ctx.layout,
                        ctx.stage_dir, ctx.upstream)
    assert stage.reopen_tasks(ctx2, results) == ["traj_00000", "traj_00001"]


def test_enabled_and_check_config(tmp_path, write_input):
    ctx = _context(tmp_path, write_input)
    stage = _stage()
    assert stage.enabled(ctx.config)
    assert not stage.enabled(ctx.config.model_copy(update={"dynamics": None}))
    one = ctx.config.electronic.model_copy(update={"singlets": 1})
    assert "singlets >= 2" in stage.check_config(ctx.config.model_copy(update={"electronic": one}))[0]


def test_bad_upstream_data_is_a_non_retryable_task_error(tmp_path, write_input):
    ctx = _context(tmp_path, write_input)
    stage = _stage()
    task = stage.plan(ctx)[0]
    bad = type(task)(task.task_id, {**task.payload, "sample": 99})
    with pytest.raises(TaskError, match="--rerun-from sampling") as info:
        stage.run_task(ctx, bad, ctx.stage_dir / "x")
    assert info.value.retryable is False
    bad = type(task)(task.task_id, {**task.payload, "rng": {"stage": "dynamics", "task_index": 5}})
    with pytest.raises(TaskError, match="--rerun-from selection"):
        stage.run_task(ctx, bad, ctx.stage_dir / "y")


def test_no_results_fails_the_stage(tmp_path, write_input):
    ctx = _context(tmp_path, write_input)
    with pytest.raises(StageFailedError):
        _stage().finalize(ctx, {})


def test_run_settings_converts_units(tmp_path, write_input):
    ctx = _context(tmp_path, write_input, dynamics="tmax_fs = 2.0\ndt_fs = 0.25\nstop_after_s0_fs = 1.0\n")
    settings = run_settings(ctx.config.dynamics)
    assert settings.n_grid_steps == 8 and settings.options.coupling == "overlap"
    assert settings.stop_after_s0_au == pytest.approx(1.0 / 0.024188843265857)
```

`write_input` writes `[project]` without a seed, so the seed is set with `model_copy`.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/stages/test_dynamics.py -q -W error`
Expected: FAIL (modules missing).

- [ ] **Step 3: Implement**

`src/gqteashmd/testing/canned_lvc.py`: the helpers listed in Interfaces. `write_canned_molecule` writes `molecule.json` as `PrepareStage` does (`schema_version`, `symbols` from the model's `V0.txt`, `charge` 0, `multiplicity` 1, `coords_angstrom = reference_q.reshape(-1, 3) * BOHR_IN_ANGSTROM`). `write_canned_samples` writes `samples.h5` with h5py containing exactly the datasets and attributes `vib/artifacts.py:read_samples_h5` reads (look at that function: `symbols`, `masses`, `equilibrium_coordinates`, `positions`, `velocities`, attributes `n_samples`, `temperature_K`, plus whatever its shape check requires), each with a `units` attribute, positions/velocities from `lvc_wigner_samples(model, n, np.random.default_rng(seed))` reshaped to `(n, natom, 3)`, masses from the model's `V0.txt` masses (amu). `write_canned_trajectories` writes `trajectories.json` in the `selection` format (`schema_version` 1, `n_trajectories`, `mode` "fixed", `project_seed`, `trajectories` entries with `id` `t{k+1:05d}`, `index` k, `sample`, `state`, `excitation_energy_eV` 0.0, `oscillator_strength` 0.0, `rng` `{"stage": "dynamics", "task_index": k}`). `lvc_backend_factory` returns `ModelBackend(canned_model(), faults={int(c): kind for c, kind in faults.get(task_dir.name, {}).items()}, marker_dir=task_dir)`.

`src/gqteashmd/stages/dynamics.py`:

```python
"""Stage 6: one surface-hopping trajectory per entry of trajectories.json (P7b)."""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.config.schema import DynamicsStageSection, ShmdConfig
from gqteashmd.core.states import StateLabel
from gqteashmd.core.units import AU_TIME_IN_FS, DALTON_IN_ME, HARTREE_IN_EV
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.randomness import GeneratorSource
from gqteashmd.electronic.base import ElectronicBackend, MoleculeSpec
from gqteashmd.errors import ProjectError, StageFailedError, TaskError
from gqteashmd.stages.base import Stage, StageContext, TaskSpec
from gqteashmd.stages.geometry import prepared_molecule, stage_backend
from gqteashmd.trajectory.model import BackendModel
from gqteashmd.trajectory.outcomes import TrajectoryOutcome
from gqteashmd.trajectory.runner import (
    OUTPUT_FILE,
    InitialConditions,
    RunSettings,
    TrajectoryRunner,
)
from gqteashmd.util.fileio import read_json, write_json_atomic, write_text_atomic
from gqteashmd.util.seeds import task_rng
from gqteashmd.vib.artifacts import read_samples_h5

log = logging.getLogger("gqteashmd.stages.dynamics")
BackendFactory = Callable[[StageContext, MoleculeSpec, Path], ElectronicBackend]


def _pyscf_backend(ctx: StageContext, molecule: MoleculeSpec, task_dir: Path) -> ElectronicBackend:
    return stage_backend(ctx, "dynamics", molecule, task_dir, settings=ctx.config.electronic)


def run_settings(section: DynamicsStageSection) -> RunSettings:
    options = DynamicsOptions(
        dt_au=section.dt_fs / AU_TIME_IN_FS,
        nsubsteps=section.nsubsteps,
        hopping=section.hopping,
        ekin_correction=section.ekin_correction,
        frustrated=section.frustrated,
        decoherence=section.decoherence,
        decoherence_param_Eh=section.decoherence_param_Eh,
        energy_drift_limit_au=section.energy_drift_limit_eV / HARTREE_IN_EV,
        edc_convention=section.edc_convention,
        substep_interpolation=section.substep_interpolation,
        coupling=section.coupling,
    )
    return RunSettings(
        options,
        tmax_au=section.tmax_fs / AU_TIME_IN_FS,
        overlap_threshold=section.overlap_threshold,
        max_step_halvings=section.max_step_halvings,
        output_every=section.output_every,
        stop_after_s0_au=(
            None if section.stop_after_s0_fs is None else section.stop_after_s0_fs / AU_TIME_IN_FS
        ),
    )
```

then the class, following the Interfaces and the binding `run_task` steps above. Key parts:

```python
class DynamicsStage(Stage):
    name = "dynamics"
    version = 1
    depends_on = ("prepare", "sampling", "selection")
    config_sections = ("electronic", "dynamics")
    required_modules = ("pyscf>=2.6", "h5py>=3.9")

    def __init__(
        self,
        *,
        max_failed_fraction: float = 0.1,
        backend_factory: BackendFactory | None = None,
        required_modules: tuple[str, ...] | None = None,
    ) -> None:
        self.max_failed_fraction = max_failed_fraction
        self._backend_factory = backend_factory or _pyscf_backend
        if required_modules is not None:
            self.required_modules = required_modules

    def enabled(self, config: ShmdConfig) -> bool:
        return config.dynamics is not None  # plan ruling R1

    def check_config(self, config: ShmdConfig) -> list[str]:
        if config.electronic is not None and config.electronic.singlets < 2:
            return [
                "Stage 'dynamics' needs [electronic] singlets >= 2 (S0 and at least one "
                "excited state)"
            ]
        return []

    def fingerprint_inputs(self, ctx: StageContext) -> dict[str, Any]:
        assert ctx.config.dynamics is not None and ctx.config.electronic is not None
        dynamics = ctx.config.dynamics.model_dump(mode="json")
        dynamics.pop("tmax_fs")  # O16: raising it extends finished trajectories
        dynamics.pop("max_failed_fraction")  # finalize-only
        return {
            "electronic": ctx.config.electronic.model_dump(mode="json"),
            "dynamics": dynamics,
            "project_seed": ctx.config.project.seed,
        }

    def reopen_tasks(
        self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]
    ) -> list[str]:
        assert ctx.config.dynamics is not None
        n_total = run_settings(ctx.config.dynamics).n_grid_steps
        return sorted(
            task_id
            for task_id, result in results.items()
            if result.get("outcome") == TrajectoryOutcome.FINISHED
            and int(result.get("n_grid_steps", n_total)) < n_total
        )
```

`plan` reads `trajectories.json` (a missing or malformed file → `ProjectError` naming `--rerun-from selection`), logs the WARNING "DFT gradients in PySCF omit the grid response; use [electronic] grid_level >= 5 for dynamics" when `method in ("tda", "tddft")`, `functional` is not `"hf"` and `grid_level < 5`, and returns `TaskSpec(f"traj_{index:05d}", {"index": index, "sample": sample, "state": state, "rng": rng})` in file order. `run_task` and `finalize` as specified above.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/stages/test_dynamics.py tests/stages -q -W error`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
ruff format . && ruff check . && mypy
git add src/gqteashmd/stages/dynamics.py src/gqteashmd/testing/canned_lvc.py tests/stages/test_dynamics.py
git commit -m "feat: add the dynamics stage"
```

---

### Task 8: The stage in the pipeline (registry, canned pipeline, `init`, orchestrator-level tests)

**Files:**
- Modify: `src/gqteashmd/stages/registry.py`, `src/gqteashmd/testing/pipelines.py`, `src/gqteashmd/cli.py` (`TEMPLATE` gets a commented `[dynamics]` block after `[selection]`)
- Test: `tests/stages/test_dynamics_pipeline.py`; update `tests/test_cli.py` if it asserts the template text or the stage list

**Interfaces:**
- Produces: `default_pipeline` ends with `DynamicsStage(max_failed_fraction=config.dynamics.max_failed_fraction if config.dynamics is not None else 0.1)`; `testing/pipelines.py:canned_dynamics(config)` = `[CannedPrepare, CannedSampling, CannedSelection, DynamicsStage(backend_factory=canned_lvc.lvc_backend_factory, required_modules=("h5py>=3.9",), max_failed_fraction=...)]`, where the three canned stages (names `prepare`, `sampling`, `selection`; one task each; `finalize` calls the Task 7 writers; `CannedSelection` writes entries `[(1, "S1"), (2, "S2"), (3, "S1")]` with `config.project.seed`).

- [ ] **Step 1: Write the failing tests** (`tests/stages/test_dynamics_pipeline.py`; subprocess-heavy, like `tests/test_orchestrator.py`)

```python
"""The dynamics stage under the orchestrator, with the canned LVC pipeline."""

import json
from pathlib import Path

import pytest

from gqteashmd.cli import main
from gqteashmd.testing import canned_lvc

LVC = Path(__file__).resolve().parents[1] / "data" / "lvc" / "lvc_singlets.template"
CANNED = "gqteashmd.testing.pipelines:canned_dynamics"
FAST = "poll_interval_s = 0.05\nstop_grace_period_s = 2.0"
ELECTRONIC = '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 3\n'


@pytest.fixture(autouse=True)
def _template(monkeypatch):
    monkeypatch.setenv(canned_lvc.LVC_TEMPLATE_ENV, str(LVC))


def _input(tmp_path, write_input, tmax="2.0", extra=""):
    return write_input(
        tmp_path, execution=FAST,
        extra=ELECTRONIC + f"[dynamics]\ntmax_fs = {tmax}\ndt_fs = 0.25\n" + extra,
    )


def _main(path, *args):
    return main(["run", str(path), "--pipeline", CANNED, *args])


def _stage_dir(tmp_path):
    return tmp_path / "run" / "03_dynamics"


def _summary(tmp_path):
    return json.loads((_stage_dir(tmp_path) / "dynamics_summary.json").read_text())


def test_canned_run_and_second_run_is_a_no_op(tmp_path, write_input):
    path = _input(tmp_path, write_input)
    assert _main(path) == 0
    summary = _summary(tmp_path)
    assert summary["outcomes"]["finished"] == 3 and summary["missing_output"] == []
    log = _stage_dir(tmp_path) / "tasks" / "traj_00000" / "steps.log"
    before = log.stat().st_mtime_ns
    assert _main(path) == 0
    assert log.stat().st_mtime_ns == before


def test_a_killed_worker_resumes_to_the_same_trajectory(tmp_path, write_input, monkeypatch):
    ref_root, run_root = tmp_path / "ref", tmp_path / "kill"
    ref_root.mkdir()
    run_root.mkdir()
    assert _main(_input(ref_root, write_input)) == 0
    monkeypatch.setenv(canned_lvc.FAULTS_ENV, json.dumps({"traj_00001": {"5": "kill"}}))
    assert _main(_input(run_root, write_input)) == 0
    task = Path("tasks") / "traj_00001" / "steps.log"
    assert (_stage_dir(run_root) / "tasks" / "traj_00001" / "killed_once").exists()
    assert (_stage_dir(run_root) / task).read_bytes() == (_stage_dir(ref_root) / task).read_bytes()


def test_raising_tmax_extends_and_changing_dt_is_stale(tmp_path, write_input):
    path = _input(tmp_path, write_input)
    assert _main(path) == 0
    path = _input(tmp_path, write_input, tmax="3.0")
    assert _main(path) == 0
    assert {t["t_final_fs"] for t in _summary(tmp_path)["trajectories"]} == {pytest.approx(3.0)}
    stale = write_input(tmp_path, execution=FAST,
                        extra=ELECTRONIC + "[dynamics]\ntmax_fs = 3.0\ndt_fs = 0.5\n")
    assert _main(stale) == 3


def test_without_a_dynamics_table_the_stage_is_skipped(tmp_path, write_input):
    path = write_input(tmp_path, execution=FAST, extra=ELECTRONIC)
    assert _main(path) == 0
    assert not (_stage_dir(tmp_path) / "dynamics_summary.json").exists()


def test_validate_rejects_nacdr_without_casscf(tmp_path, write_input):
    path = _input(tmp_path, write_input, extra='coupling = "nacdr"\n')
    assert main(["validate", str(path)]) == 2
```

Check how `main` accepts the hidden `--pipeline` option (existing CLI tests use it) and adapt the call shape if needed.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/stages/test_dynamics_pipeline.py -q -W error`
Expected: FAIL (`canned_dynamics` missing).

- [ ] **Step 3: Implement** the registry change, the canned pipeline, and this block in the CLI `TEMPLATE` after the `[selection]` table (keep the template valid TOML with the block commented out):

```toml
# [dynamics]                         # remove the leading '# ' to run trajectories
# tmax_fs = 100.0                    # must be a multiple of dt_fs
# dt_fs = 0.5
# coupling = "overlap"               # "nacdr" needs method = "casscf"
# hopping = "sharc"
# decoherence = "edc"
# energy_drift_limit_eV = 0.5
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/stages/test_dynamics_pipeline.py tests/test_cli.py tests/test_orchestrator.py -q -W error`, then the fast tier `pytest -m "not slow and not pyscf" -W error -q` (record the count; CLAUDE.md's last count was 922 passed, 20 skipped at `19b43b6`).
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
ruff format . && ruff check . && mypy
git add src/gqteashmd/stages/registry.py src/gqteashmd/testing/pipelines.py src/gqteashmd/cli.py tests/
git commit -m "feat: run the dynamics stage at the end of the workflow"
```

---

### Task 9: PySCF validation and the end-to-end run

**Files:**
- Create: `tests/validation/test_pyscf_dynamics.py` (markers `pyscf`; `slow` where stated), `tests/test_e2e_dynamics.py` (marker `pyscf`; the kill test also `slow`)

**Interfaces:**
- Consumes: `BackendModel`, `TrajectoryRunner`, `RunSettings`, `InitialConditions` (Tasks 3-5); `make_backend` (`electronic/factory.py`); `tests/data/ethylene.xyz`; `chem/masses.isotope_mass`.

**Bounds, fixed now (before any run):**

| Check | System | Bound |
|---|---|---|
| (a) conservation | ethylene SA2-CASSCF(2,2)/STO-3G, start S1, `hopping = "none"`, `decoherence = "none"`, `coupling = "overlap"`, dt 0.5 fs, 20 steps | max \|E_tot - E_tot(0)\| <= 1e-4 Eh; every step's smallest singular value >= 0.9; no rejection |
| (b) dt^2 scaling (`slow`) | as (a), 5 fs at dt 0.5 and 0.25 fs | ratio drift(0.5) / drift(0.25) in [2.5, 6.0], asserted only if drift(0.5) >= 1e-6 Eh; otherwise the test reports the drifts and is skipped with that reason (convergence noise dominates) |
| (c) resume | as (a): one run of 10 steps, against a run to tmax = 5 steps followed by a new `PySCFBackend`, `BackendModel` and `TrajectoryRunner` on the same directory with tmax = 10 steps (resume from the restart, O16 extension) | positions <= 1e-6 bohr, energies <= 1e-7 Eh, populations <= 1e-6 |
| (d) TDA | water CIS (`tda`, `functional = "hf"`, STO-3G, singlets 3), start S1, `hopping = "none"`, dt 0.25 fs, 20 steps | max \|E_tot - E_tot(0)\| <= 1e-4 Eh, no rejection |
| (e) end to end | water HF/STO-3G workflow of `tests/test_e2e_excitations.py` (`EXTRA`) + `[dynamics] tmax_fs = 1.0, dt_fs = 0.25` | exit 0; `dynamics_summary.json` holds one row per trajectory, each with an outcome; second run exit 0 with no change; `tmax_fs = 1.5` reopens and every `finished` trajectory ends at 1.5 fs |
| (f) kill (`slow`) | as (e) with `tmax_fs = 2.0`: start `gqteashmd run` as a subprocess, SIGKILL its process group when the first `restart/current` of `traj_00000` exists, run again | second run exit 0, all trajectories have an outcome |

Initial velocities for (a)-(d): `v = rng.standard_normal((natom, 3)) * sqrt(k_B T / m)` with `T = 300 K`, `rng = np.random.default_rng(2026)`, masses from `isotope_mass`, no momentum removal; positions from the XYZ file (bohr). Electronic settings for ethylene: `method = "casscf"`, `basis = "sto-3g"`, `singlets = 2`, `ncas = 2`, `nelecas = 2`, default tolerances. Use `make_backend(config, molecule, settings=..., scratch_dir=tmp_path / "scratch", log_file=tmp_path / "e.log", threads=2, memory_mb=1000)` (build the `ShmdConfig` with `write_input` + `load_config`).

- [ ] **Step 1: Write the tests** as in the table. Each test prints its measured numbers (`print(...)` is captured; also put them in the assertion messages) so the report can quote them.
- [ ] **Step 2: Run them** — `pytest tests/validation/test_pyscf_dynamics.py tests/test_e2e_dynamics.py -q -W error -m "pyscf"` (with `slow`: add `-m "pyscf or slow"`; or run the files without `-m`). Record wall times.
- [ ] **Step 3: If a bound fails:** do not change the bound. Report the numbers, the diagnosis and a proposal; the controller decides (house rule). Fix only defects in our code (with a test that fails before the fix).
- [ ] **Step 4: Commit**

```bash
ruff format . && ruff check . && mypy
git add tests/validation/test_pyscf_dynamics.py tests/test_e2e_dynamics.py
git commit -m "test: validate PySCF dynamics (energy conservation, resume, end to end)"
```

---

### Task 10: SHARC 4.1 black-box comparison (one hopping-off trajectory)

**Files:**
- Create: `tools/sharc_compare/run_sharc_pyscf_dynamics.py`, `tests/validation/test_sharc_pyscf_dynamics.py` (markers `sharc`, `pyscf`, `slow`; skipped without `SHARC_HOME`/`SHARC_PYTHON`)

**Interfaces:**
- Consumes: our harness code only: `tools/sharc_compare/run_sharc_model.py` (`run_in_session`, `sharc_environment`), `tools/sharc_compare/run_sharc_lvc.py` (the `sharc.x` input writer and the `output.dat` parser), `tools/sharc_compare/run_sharc_pyscf.py` (how `SHARC_PYSCF.py` is set up for single points, P3a/P6). The `sharc.x` input keywords and the `SHARC_PYSCF.py` template keywords come from the public SHARC 4.1 manual (as in P7a-1/P7a-2). Never read SHARC source.

**Setup (binding):** ethylene SA2-CASSCF(2,2)/STO-3G (the P3a validation system, same settings as Task 9 (a)); initial geometry and velocities as Task 9 (a); SHARC: `surf sharc`, hopping off, decoherence none, `coupling overlap`, `ekincorrect parallel_vel`, dt 0.5 fs, `nsubsteps 25`, singlet-only states 2, initial state S1. Ours: `hopping = "none"`, `decoherence = "none"`, `coupling = "overlap"`, `substep_interpolation = "endpoint"` (SHARC's scheme, measured in P7a-1), same dt/nsubsteps. **Start ours from SHARC's interface geometry:** sharc.x rounds the `QM.in` geometry to 7 decimals in angstrom (measured in P7a-2), so our positions are compared with SHARC's printed positions and our electronic structure at step 0 uses the same rounded geometry (`round(q * a0, 7) / a0`).

**Length:** first measure one SHARC step's wall time (`--steps 2`); choose the number of steps n as the largest of 10, 20, 40 that keeps the SHARC run under 15 min; record the choice and the time in the report before comparing.

**Tolerances, fixed now:** positions <= 1e-6 bohr, velocities <= 1e-7 bohr per atomic time unit, MCH energies <= 1e-6 Eh, populations <= 1e-4, at every compared step.

**Report only (not asserted):** the swap test — `sharc.x` with our `PySCFBackend` behind the `QM.out` provider pattern of `run_sharc_lvc.py` — same trajectory, the same numbers, written to `~/p7b_reports/sharc_pyscf_dynamics.json`.

- [ ] **Step 1:** Write `run_sharc_pyscf_dynamics.py` (functions `run_sharc_trajectory(geometry_bohr, velocities_au, *, n_steps, workdir) -> dict` returning per-step positions, velocities, energies, populations parsed from `output.dat`; a `--steps` CLI for the timing probe; harness timeouts with a whole-group kill via `run_in_session`). Add parser unit tests to `tests/sharc_compare/` that run without SHARC (on a small captured `output.dat` text written by the test itself, following `tests/sharc_compare/test_run_sharc_lvc.py`).
- [ ] **Step 2:** Run the timing probe; record it.
- [ ] **Step 3:** Write the asserted test with the tolerances above; run it with `SHARC_HOME=/home/ajc/sharc4-4.1 SHARC_PYTHON=/home/ajc/miniforge3/envs/sharc4.1/bin/python pytest tests/validation/test_sharc_pyscf_dynamics.py -q -W error`.
- [ ] **Step 4:** If it fails, report the per-step differences (first step where a tolerance is exceeded, which quantity) and a diagnosis; do not loosen. Run the swap report once.
- [ ] **Step 5: Commit**

```bash
ruff format . && ruff check . && mypy
git add tools/sharc_compare/run_sharc_pyscf_dynamics.py tests/validation/test_sharc_pyscf_dynamics.py tests/sharc_compare/
git commit -m "test: compare a PySCF trajectory with SHARC 4.1 (black box, hopping off)"
```

---

### Task 11: Documents and final verification

**Files:**
- Modify: `CLAUDE.md`, `USER_MANUAL.md`, `docs/superpowers/specs/2026-11-11-p7b-dynamics-stage-design.md` (erratum block at the top: rulings R1 and R5)

- [ ] **Step 1: USER_MANUAL.md.** Mark `dynamics` [IMPLEMENTED]: the `[dynamics]` key table (Task 1 defaults and rules, R1, R3), the stage's inputs and outputs (`06_dynamics/tasks/traj_<k>/`: `steps.log`, `output.h5` datasets with units, `progress.json`, `restart/`; `dynamics_summary.json/.md`), outcomes (spec §5), step rejection (spec §4, R4, R7, R8), restart and resume (and that PySCF resume is not bit-identical; the Task 9 (c) numbers), extension by raising `tmax_fs` (O16) and that changing any other `[dynamics]` key needs `--rerun-from dynamics`, `pause_after = ["selection"]` now pausing before the dynamics, a real example run (water HF/STO-3G from Task 9 (e): input, wall time, summary excerpt), every measured number of Tasks 9 and 10 marked asserted or report, and limitations (singlets only; no live `output.h5`; DFT grid response; no SOC/triplets from PySCF; no thermostat).
- [ ] **Step 2: CLAUDE.md.** Architecture block for `trajectory/` and `stages/dynamics.py` (modules, rules, rulings R1-R10, the `reopen_tasks` hook in the orchestrator rules), Config (`[dynamics]`), Environment (test files, markers, measured times, the new fast-tier count), Current status (P7b on its branch), O8/O9/O16 resolved in the key decisions, Known limitations 1 (the workflow now ends at `dynamics`), Remaining work (P8 next). Remove the P7b deferred items this plan resolved (Löwdin in the adapter; the P7b-tagged deferrals that are now done) and keep the ones still open, each with its state.
- [ ] **Step 3: Verification.** Run and record (commit, counts, time): the fast tier; `pytest -m sharc` with SHARC; then the full suite `pytest -W error` with `SHARC_HOME`, `SHARC_PYTHON` and `ORCA_EXE` set (about 2.5-3 h; run it with `nohup` into `~/p7b_reports/full_suite.log` and wait for it; a run that dies is not a result). Put the numbers in both documents.
- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md USER_MANUAL.md docs/superpowers/specs/2026-11-11-p7b-dynamics-stage-design.md
git commit -m "docs: document the dynamics stage (P7b)"
```

---

## Self-review (done while writing)

- **Spec coverage:** §3.1 → Task 3; §3.2 → Task 5; §4 → Task 5 (rejection) and R4/R7/R8; §5 → Tasks 3 and 5; §6 → Tasks 4 and 5; §7 → Tasks 1, 7, 8 (R1, R5 deviations recorded); §8 → Tasks 6 and 7 (`reopen_tasks`) and 8 (orchestrator-level test); §9 → Tasks 2-5, 7, 8 (fast tier), 9 (PySCF), 10 (SHARC); §10 → Task 11.
- **Names used across tasks:** `ModelBackend`, `SimulatedCrash`, `BackendModel.refresh_active_gradient`, `min_singular_value`, `rung_changed`, `TrajectoryOutcome`, `StepLog`, `RestartStore`, `log_row`, `write_output_h5`, `RunSettings.n_grid_steps`/`ticks_per_step`, `InitialConditions`, `TrajectoryRunner.run`, `run_settings`, `DynamicsStage(backend_factory=, required_modules=)`, `Stage.reopen_tasks`, `canned_lvc.*`, `canned_dynamics`, `DynamicsStageSection` — each defined in the task that first uses it.
