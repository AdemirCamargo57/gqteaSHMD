# Censored Analysis and Hop-Step Energy Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The analysis uses trajectories that end early as censored data (discrete-time Aalen-Johansen estimator, the new default `[analysis] estimator = "censored"`, with today's rule kept as `"complete"`). Under `hop_energy = "previous"`, the energy-jump signal sees the energy before the velocity adjustment at an accepted hop.

**Architecture:**
- **New module `gqteashmd/analysis/censoring.py`:** which trajectories are censored, and the estimators. It is NumPy only.
- **New module `gqteashmd/analysis/estimators.py`:** one function per estimator, turning a `GridData` into populations, intervals and the fit. The stage and the report tool both call these functions, so the logic exists once.
- **`grid.on_grid`:** gains an at-risk mask (`observed`).
- **`stages/analysis.py`:** computes both estimators. The dynamics side is one new `StepRecord` field and one line in the runner's energy check.

**Tech Stack:** Python 3.12, NumPy, SciPy (fit only, as today), h5py (writers only), pydantic 2, pytest.

**Spec:** [docs/superpowers/specs/2026-11-19-censored-analysis-design.md](../specs/2026-11-19-censored-analysis-design.md) (owner-approved 2026-11-19). The plan rulings below are binding where they differ from the spec.

## Global Constraints

- Run everything in WSL: `wsl -d OracleLinux_9_5 -- bash -lc 'cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && conda activate gqteashmd-dev && <command>'`. Git from WSL only (pre-commit hooks run ruff 0.17.4); never `--no-verify`. LF line endings.
- One commit per task, conventional subjects (`feat:`, `fix:`, `test:`, `docs:`). End each commit message with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` (subagents) or `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (controller).
- Lint: `ruff format . && ruff check .` (line length 100; E, F, W, I, B, UP, SIM); `mypy src` clean.
- Dependency rule: `analysis/` imports only `core/`, `gqteashmd.__version__` and itself; SciPy and h5py only inside functions; `import gqteashmd.cli` must not import pyscf, geometric, h5py or scipy. `dynamics/` imports nothing from `electronic/`, `stages/`, the orchestrator or `config/`.
- Bit-identity: with `hop_energy = "current"` every dynamics result is unchanged (`tests/dynamics/test_golden.py` unchanged and green; nothing regenerated). With `estimator = "complete"` the analysis numbers (populations, intervals, fit, bootstrap) are unchanged against `main` (same random stream `task_rng(seed, "analysis", 0)`).
- Seeds (D12): the complete estimator draws from `task_rng(seed, "analysis", 0)` as today, the censored one from `task_rng(seed, "analysis", 1)`. Both use `populations.bootstrap_indices(rng, n, n_bootstrap)`.
- Clean room: SHARC is not involved; nothing under `/home/ajc/sharc4-4.1` is read.
- Runs and outputs stay out of OneDrive and `/mnt/c` (`~/censoring/` in WSL for Task 7).
- Every test run uses `-W error`.
- Keep CLAUDE.md and USER_MANUAL.md current in this branch (Task 8).

## Plan rulings (binding where they differ from the spec)

- **PR1, the synthetic Markov rule (spec 4.1 item 2).** The spec's rule, 3 SE at every one of 101 times and 3 states, makes 303 comparisons. A correct estimator would exceed 3 SE in about 0.8 of them on average, so the rule would fail by chance more often than not. **Rule used:** the comparisons are at steps 10, 20, ..., 100 and every state (30 comparisons), with the bound **4 SE** (family-wise false-alarm probability about 2e-3). The "must be able to fail" check uses the same 30 points and 4 SE. Decided before any run.
- **PR2, `empty_rows` (spec 2.2).** Counted only for an interval and a state with no trajectory at risk **and** a positive estimated population at the start of the interval. These are the identity rows that carry probability. A state nobody occupies would otherwise be counted at every step.
- **PR3, random streams.** The complete estimator keeps `task_rng(seed, "analysis", 0)`, so its numbers are unchanged; the censored one uses `task_rng(seed, "analysis", 1)`.
- **PR4, `analysis.json` (spec 2.5).**
  - **Top level, for the estimator in use:** `n_included`, `included`, `excluded`, `small_ensemble`, `fit` and `table` mirror the estimator in use.
  - **`estimators`:** holds both blocks.
  - **Always computed:** `n_complete`, `n_censored`, `censored` and `estimator`.
  - **Over the complete trajectories, as today:** `hops`, `energy.drift_eV` and `consistency`. `consistency` is `null` when there is no complete trajectory.
  - **The included-jump warning:** counts the flagged trajectories among those used by the estimator in use.
  - `schema_version` 2.
- **PR5, censored at or after the horizon.** A trajectory whose censoring grid step is at or after the horizon counts as complete in the censored estimator. It is not in `censored` and not in `n_censored`.
- **PR6, the second estimator in the files.** `populations.csv` and `populations.h5` hold the estimator in use under today's names. The other estimator's classical and quantum curves and intervals are added under the prefix `complete_` or `censored_`: CSV columns `complete_P_S1`, `complete_P_S1_lower`, ..., and h5 datasets `complete_classical`, `complete_classical_lower`, .... They are left out when the other estimator has no trajectory.
- **PR7, the horizon.** It is computed from the complete trajectories (`grid.horizon_steps(complete, n_total)`) and passed to both grids. A trajectory censored before a jump keeps its outcome `finished` but does not set the horizon.
- **PR8, the report's bias unit (spec 4.2 item 5).** "The mean bootstrap SE" is the bootstrap standard deviation of the **censored** estimate (B = 200), averaged over the R repetitions, at each time and state. All four estimators are divided by that same unit. Points where it is 0 are skipped.
- **PR9, the tool reads the project's settings** from `<project>/.gqteashmd/config.json` (`dynamics.tmax_fs`, `dynamics.dt_fs`, `dynamics.output_every`, `analysis.n_bootstrap`, `project.seed`).
- **PR10, `e_total_before_hop` only at an accepted hop (spec 3).** The spec says "accepted or frustrated". At a frustrated attempt the velocity is kept or reflected, so `K` is unchanged and the record's `e_total` already equals `E_diag'_lambda + K(v')`, the value the field would hold. The check therefore sees the same energy either way. Setting the field only at an accepted hop keeps `test_previous_hop_energy_frustrated_hop` (records identical under both rules) valid.

---

### Task 1: The hop-step energy check (`dynamics`, `trajectory`, `DynamicsStage.version` 3)

**Files:**
- Modify: `src/gqteashmd/dynamics/records.py` (`StepRecord`)
- Modify: `src/gqteashmd/dynamics/driver.py:310-394` (step 8 and the record)
- Modify: `src/gqteashmd/trajectory/runner.py:292-307` (`_energy_jump`)
- Modify: `src/gqteashmd/stages/dynamics.py:95` (`version = 2` -> `3`)
- Test: `tests/dynamics/test_driver.py`, `tests/trajectory/test_runner.py`, `tests/stages/test_dynamics.py`

**Interfaces:**
- Produces: `StepRecord.e_total_before_hop: float | None = None` (last field). Set only under `hop_energy = "previous"` at an accepted hop, to `float(e_diag1[old lambda]) + kinetic_energy(v1 before the adjustment, masses)`.

- [ ] **Step 1: Write the failing driver test** (append to `tests/dynamics/test_driver.py`)

```python
@pytest.mark.parametrize("rule", ["previous", "current"])
def test_e_total_before_hop_is_the_unadjusted_energy(monkeypatch, rule):
    """Under "previous" an accepted hop records E_lambda(q') + K(v') before the adjustment
    (recomputed here from the velocity passed to adjust_velocity); otherwise the field is None."""
    from gqteashmd.dynamics import driver as driver_module

    model = TullySimpleAvoidedCrossing()
    options = DynamicsOptions(dt_au=1.0, hopping="sharc", hop_energy=rule)
    state = _start(model, x0=-4.0, k=20.0)
    seen = []
    original = driver_module.adjust_velocity

    def spy(velocity, masses, delta_e, **kwargs):
        seen.append(velocity.copy())
        return original(velocity, masses, delta_e, **kwargs)

    monkeypatch.setattr(driver_module, "adjust_velocity", spy)
    _, _, records = _run(model, state, options, ReplaySource([0.0] * 800), max_steps=800)
    attempts = [i for i, r in enumerate(records) if r.hop_to is not None or r.frustrated]
    assert len(attempts) == len(seen) >= 2
    accepted = 0
    for i, v_before in zip(attempts, seen, strict=True):
        r = records[i]
        if r.hop_to is None or rule == "current":
            assert r.e_total_before_hop is None
            continue
        kinetic = 0.5 * float(np.sum(model.masses * v_before * v_before))
        expected = float(r.energies[r.hop_from]) + kinetic
        assert r.e_total_before_hop == pytest.approx(expected, abs=1e-12)
        accepted += 1
    assert accepted >= (2 if rule == "previous" else 0)
    assert all(r.e_total_before_hop is None for r in records if r.hop_to is None)
```

- [ ] **Step 2: Run it to see it fail**

Run: `pytest tests/dynamics/test_driver.py -k e_total_before_hop -q -W error`
Expected: FAIL with `AttributeError: 'StepRecord' object has no attribute 'e_total_before_hop'`.

- [ ] **Step 3: Implement the field and the driver change**

In `src/gqteashmd/dynamics/records.py`, add as the last field of `StepRecord`:

```python
    u: np.ndarray | None = None  # copy of U' (None = identity)
    # hop_energy = "previous", accepted hop: E_diag'_lambda + K(v') before the velocity
    # adjustment (the runner's energy-jump signal; not stored in steps.log or output.h5)
    e_total_before_hop: float | None = None
```

In `src/gqteashmd/dynamics/driver.py`, step 8 becomes (keep everything else as it is):

```python
    # 8: hop with velocity adjustment at fixed q'
    hop_from: int | None = None
    hop_to: int | None = None
    frustrated = False
    e_total_before_hop: float | None = None
    target = choose_state(lam, probabilities, xi)
    if target is not None:
        unadjusted: float | None = None
        if options.hop_energy == "previous":
            # E_tot of the step's starting state = the previous record's E_tot (item 11)
            e_total_previous = float(
                diagonal_energies(snap0.hamiltonian, u0)[state.active]
            ) + kinetic_energy(v0, masses)
            kinetic_before = kinetic_energy(v1, masses)
            unadjusted = float(e_diag1[lam]) + kinetic_before
            delta_e = hop_delta_e(
                "previous",
                e_diag1[target],
                e_diag1[lam],
                kinetic_before,
                e_total_previous,
            )
        else:
            delta_e = hop_delta_e("current", e_diag1[target], e_diag1[lam], 0.0, 0.0)
        direction = None
        if options.ekin_correction == "parallel_nac":
            direction = _hop_direction(masked1, g_diag1, e_diag1, lam, target)
        adjustment = adjust_velocity(
            v1,
            masses,
            delta_e,
            method=options.ekin_correction,
            nac=direction,
            frustrated=options.frustrated,
        )
        v1 = adjustment.velocity
        if adjustment.accepted:
            hop_from, hop_to, lam = lam, target, target
            e_total_before_hop = unadjusted
        else:
            frustrated = True
```

and the record gets `e_total_before_hop=e_total_before_hop,` after `u=...`. `kinetic_before` is the same float the old code computed inline (`kinetic_energy(v1, masses)`), so `"previous"` stays bit-identical too.

- [ ] **Step 4: Run the driver tests**

Run: `pytest tests/dynamics -q -W error -m "not slow"`
Expected: PASS. That covers 392 tests plus the two new ones, and `test_golden.py` and `test_previous_hop_energy_frustrated_hop` stay green.

- [ ] **Step 5: Write the failing runner test** (append to `tests/trajectory/test_runner.py`)

```python
@pytest.mark.parametrize("rule", ["previous", "current"])
def test_an_energy_jump_at_a_hop_step_is_seen_under_both_rules(tmp_path, rule):
    """An energy jump (every energy + 0.01 Eh) in the call of the first hop step is rejected
    under both rules: under "previous" the hop record's e_total is pinned to the previous
    record's, so only e_total_before_hop shows it (spec section 3)."""
    settings = _settings(options={"hop_energy": rule}, energy_jump_limit_au=F1)
    _, reference = _run(tmp_path, name="ref", settings=settings, source=ReplaySource([0.0] * 400))
    rows = _log(reference)
    hop_row = int(np.flatnonzero(rows["hop_to"] >= 0)[0])
    assert hop_row > 0
    # without rejections log row r is grid step r, computed by call r + 1 (call 1 is step 0)
    result, _ = _run(
        tmp_path,
        name="jump",
        settings=settings,
        faults={hop_row + 1: "energy_jump"},
        source=ReplaySource([0.0] * 400),
    )
    assert result["rejections_by_signal"] == {"energy": 1}, result["rejections_by_signal"]


def test_previous_without_faults_rejects_nothing_new(tmp_path):
    settings = _settings(options={"hop_energy": "previous"}, energy_jump_limit_au=F1)
    result, _ = _run(tmp_path, settings=settings, source=ReplaySource([0.0] * 400))
    assert result["n_hops"] > 0 and result["n_rejections"] == 0
```

Before the fix, the first test fails for `"previous"`: it gives `{}`, because the pinned record hides the jump. The `"current"` case passes both before and after, which is the discriminating pair.

- [ ] **Step 6: Run it to see the `"previous"` case fail**

Run: `pytest tests/trajectory/test_runner.py -k "hop_step or rejects_nothing_new" -q -W error`
Expected: `test_an_energy_jump_at_a_hop_step_is_seen_under_both_rules[previous]` FAILS (`{} != {'energy': 1}`); the other two pass.

- [ ] **Step 7: Implement the runner change** (`trajectory/runner.py:_energy_jump`)

Replace the jump line:

```python
        # Under hop_energy = "previous" an accepted hop's e_total equals the previous record's
        # by construction; the energy before the adjustment is what can jump (spec section 3).
        energy = record.e_total if record.e_total_before_hop is None else record.e_total_before_hop
        # A non-finite E_tot gives a NaN jump, rejected here as an energy jump (practically
        # unreachable: BackendModel already rejects non-finite energies as electronic failures).
        jump = abs(float(energy) - previous)
```

Update the docstring to: `"""|E - E_tot(previous record)| if it exceeds energy_jump_limit_au, else None; E is the record's E_tot, or its e_total_before_hop under hop_energy = "previous" at a hop."""`

- [ ] **Step 8: Raise the dynamics stage version**

In `src/gqteashmd/stages/dynamics.py`, change `version = 2` to `version = 3`, with this comment above it:

```python
    # 3: under hop_energy = "previous" the energy-jump signal sees the energy before the hop's
    # velocity adjustment (censored analysis spec, section 3); 2: P9a plan 2, F1
```

In `tests/stages/test_dynamics.py`, rename `test_the_stage_version_is_two` to `test_the_stage_version_is_three` and assert `DynamicsStage.version == 3`. Then run `grep -rn "DynamicsStage.version\|version == 2" tests tools` and update every other assertion of the dynamics version 2 that it finds (an analysis version assertion stays at 2 until Task 5).

- [ ] **Step 9: Run the affected tests**

Run: `pytest tests/dynamics tests/trajectory tests/stages/test_dynamics.py tests/stages/test_dynamics_pipeline.py -q -W error -m "not slow and not pyscf"`
Expected: PASS.

- [ ] **Step 10: Lint and commit**

```bash
ruff format . && ruff check . && mypy src
git add src/gqteashmd/dynamics/records.py src/gqteashmd/dynamics/driver.py src/gqteashmd/trajectory/runner.py src/gqteashmd/stages/dynamics.py tests/dynamics/test_driver.py tests/trajectory/test_runner.py tests/stages/test_dynamics.py
git commit -m "feat: the energy-jump signal sees the energy before a previous-rule hop"
```

---

### Task 2: The estimators (`analysis/censoring.py`, `energy.first_jump_index`, `kinetics.bootstrap_fit_curves`)

**Files:**
- Create: `src/gqteashmd/analysis/censoring.py`
- Modify: `src/gqteashmd/analysis/energy.py` (add `first_jump_index`)
- Modify: `src/gqteashmd/analysis/kinetics.py` (add `bootstrap_fit_curves`; `bootstrap_fit` calls it)
- Test: `tests/analysis/test_censoring.py` (new), `tests/analysis/test_energy.py`, `tests/analysis/test_kinetics.py`

**Interfaces:**
- Produces (exact signatures used by Tasks 3, 5 and 6):
  - `censoring.AalenJohansen(populations: np.ndarray, empty_rows: int)` (frozen dataclass), with `populations` of shape (G, S).
  - `censoring.transition_indicators(active: np.ndarray, observed: np.ndarray, n_states: int) -> np.ndarray`, shape (n, G-1, S, S).
  - `censoring.aalen_johansen(active: np.ndarray, observed: np.ndarray, n_states: int) -> AalenJohansen`.
  - `censoring.resample_weights(indices: np.ndarray, n: int) -> np.ndarray`, shape (B, n).
  - `censoring.bootstrap_aalen_johansen(active, observed, n_states, indices) -> np.ndarray`, shape (B, G, S).
  - `censoring.at_risk_average(values: np.ndarray, observed: np.ndarray) -> np.ndarray`. `values` has shape (n, G, ...) and the result (G, ...); it is NaN where nobody is at risk.
  - `censoring.bootstrap_at_risk_average(values, observed, indices) -> np.ndarray`, shape (B, G, ...).
  - `energy.first_jump_index(e_total_hartree: np.ndarray, threshold_eV: float) -> int | None`.
  - `kinetics.bootstrap_fit_curves(t: np.ndarray, curve: np.ndarray, resampled: Iterable[np.ndarray], *, n_trajectories: int, dt_out: float) -> FitWithUncertainty`.

- [ ] **Step 1: Write the failing unit tests** (`tests/analysis/test_censoring.py`)

```python
"""Censored estimators (censored analysis spec, sections 2.2 and 4.1 items 1-2)."""

import numpy as np
import pytest

from gqteashmd.analysis import censoring, populations

T = np.array([[0.98, 0.02, 0.0], [0.04, 0.95, 0.01], [0.0, 0.05, 0.95]])  # rows: from S0, S1, S2
STEPS = 100
N = 100
R = 400
CENSOR = 0.02  # per step, in S0 only
POINTS = range(10, STEPS + 1, 10)  # plan ruling PR1
BOUND_SE = 4.0


def _truth():
    p = np.zeros((STEPS + 1, 3))
    p[0, 2] = 1.0
    for k in range(STEPS):
        p[k + 1] = p[k] @ T
    return p


def _simulate(rng, n=N, censor=CENSOR):
    cum = np.cumsum(T, axis=1)
    active = np.empty((n, STEPS + 1), dtype=np.int64)
    observed = np.ones((n, STEPS + 1), dtype=bool)
    active[:, 0] = 2
    alive = np.ones(n, dtype=bool)
    for k in range(STEPS):
        u = rng.random(n)
        active[:, k + 1] = (u[:, None] > cum[active[:, k]]).sum(axis=1)
        # a trajectory in S0 at step k is lost before step k + 1 with probability `censor`
        lost = alive & (active[:, k] == 0) & (rng.random(n) < censor)
        alive &= ~lost
        observed[:, k + 1] = alive
    return active, observed


def test_without_censoring_it_equals_the_classical_fractions():
    rng = np.random.default_rng(1)
    active, observed = _simulate(rng, censor=0.0)
    assert observed.all()
    aj = censoring.aalen_johansen(active, observed, 3)
    fractions, _ = populations.classical_populations(active, 3)
    assert np.abs(aj.populations - fractions).max() <= 1e-12
    assert aj.empty_rows == 0


def test_a_lost_trajectory_leaves_the_risk_set_of_its_interval():
    # two trajectories: one stays in S1, one goes S1 -> S0 and is lost after step 1
    active = np.array([[1, 1, 1], [1, 0, 0]])
    observed = np.array([[True, True, True], [True, True, False]])
    aj = censoring.aalen_johansen(active, observed, 2)
    np.testing.assert_allclose(aj.populations[1], [0.5, 0.5])
    # interval 1: only trajectory 0 at risk (S1 -> S1); S0 has nobody at risk but p = 0.5:
    # its identity row keeps it (one empty row, ruling PR2)
    np.testing.assert_allclose(aj.populations[2], [0.5, 0.5])
    assert aj.empty_rows == 1


def test_the_estimator_recovers_a_markov_truth_and_the_other_rules_do_not():
    rng = np.random.default_rng(20261119)
    truth = _truth()
    est = {"censored": [], "still_running": [], "complete_only": []}
    for _ in range(R):
        active, observed = _simulate(rng)
        est["censored"].append(censoring.aalen_johansen(active, observed, 3).populations)
        onehot = np.eye(3)[active]
        est["still_running"].append(censoring.at_risk_average(onehot, observed))
        never = observed[:, -1]
        est["complete_only"].append(populations.classical_populations(active[never], 3)[0])
    worst = {}
    for name, values in est.items():
        v = np.asarray(values)  # (R, G, S)
        mean = v.mean(axis=0)
        se = v.std(axis=0, ddof=1) / np.sqrt(R)
        z = [
            abs(mean[k, s] - truth[k, s]) / se[k, s] if se[k, s] > 0 else 0.0
            for k in POINTS
            for s in range(3)
        ]
        worst[name] = max(z)
    assert worst["censored"] <= BOUND_SE, worst
    assert worst["still_running"] > BOUND_SE, worst  # the test can fail
    assert worst["complete_only"] > BOUND_SE, worst


@pytest.mark.slow
def test_the_bootstrap_interval_covers_the_truth():
    rng = np.random.default_rng(20261120)
    truth = _truth()[50, 0]
    covered = 0
    for _ in range(R):
        active, observed = _simulate(rng)
        idx = populations.bootstrap_indices(rng, N, 500)
        boot = censoring.bootstrap_aalen_johansen(active, observed, 3, idx)
        lo, hi = np.percentile(boot[:, 50, 0], [2.5, 97.5])
        covered += int(lo <= truth <= hi)
    assert 0.90 <= covered / R <= 0.99, covered / R


def test_the_bootstrap_of_the_identity_resample_is_the_estimate():
    rng = np.random.default_rng(3)
    active, observed = _simulate(rng, n=20)
    idx = np.arange(20)[None, :]
    boot = censoring.bootstrap_aalen_johansen(active, observed, 3, idx)
    aj = censoring.aalen_johansen(active, observed, 3)
    assert np.abs(boot[0] - aj.populations).max() <= 1e-12


def test_at_risk_average_and_its_bootstrap():
    values = np.array([[[1.0], [1.0]], [[0.0], [5.0]]])  # (n=2, G=2, 1)
    observed = np.array([[True, True], [True, False]])
    np.testing.assert_allclose(censoring.at_risk_average(values, observed)[:, 0], [0.5, 1.0])
    idx = np.array([[0, 1], [1, 1]])
    boot = censoring.bootstrap_at_risk_average(values, observed, idx)
    np.testing.assert_allclose(boot[0, :, 0], [0.5, 1.0])
    assert boot[1, 0, 0] == 0.0 and np.isnan(boot[1, 1, 0])  # nobody at risk: NaN


def test_resample_weights_count_multiplicities():
    w = censoring.resample_weights(np.array([[0, 0, 2], [1, 2, 2]]), 3)
    np.testing.assert_array_equal(w, [[2, 0, 1], [0, 1, 2]])
```

The Markov test runs `aalen_johansen` 400 times. If it takes longer than 60 s, mark it `@pytest.mark.slow` as well (spec 4.1 item 2) and say so in the report.

- [ ] **Step 2: Run it to see it fail**

Run: `pytest tests/analysis/test_censoring.py -q -W error -m "not slow"`
Expected: FAIL with `ImportError: cannot import name 'censoring'`.

- [ ] **Step 3: Implement `src/gqteashmd/analysis/censoring.py`**

```python
"""Estimators for trajectories that end early (censored analysis spec, section 2.2).

A trajectory is *observed* at a grid time while it has data there; once its data ends it is
censored and leaves the risk set. The discrete-time Aalen-Johansen estimator multiplies the
interval transition matrices estimated from the trajectories at risk at both ends of each
interval. With no censoring it equals the plain fractions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AalenJohansen:
    populations: np.ndarray  # (G, S)
    empty_rows: int  # (interval, state) pairs with nobody at risk but probability > 0 (PR2)


def transition_indicators(active: np.ndarray, observed: np.ndarray, n_states: int) -> np.ndarray:
    """(n, G-1, S, S): 1 where trajectory i is at risk over [t_k, t_k+1] and goes a -> b."""
    active = np.asarray(active, dtype=np.int64)
    observed = np.asarray(observed, dtype=bool)
    n, g = active.shape
    x = np.zeros((n, max(g - 1, 0), n_states, n_states))
    if g < 2:
        return x
    risk = observed[:, :-1] & observed[:, 1:]
    i, k = np.nonzero(risk)
    x[i, k, active[i, k], active[i, k + 1]] = 1.0
    return x


def _product(initial: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """initial (B, S), counts (B, G-1, S, S) -> populations (B, G, S), empty rows (B,)."""
    b, steps, s, _ = counts.shape
    p = np.empty((b, steps + 1, s))
    p[:, 0] = initial
    empty = np.zeros(b, dtype=np.int64)
    eye = np.eye(s)
    for k in range(steps):
        n_from = counts[:, k].sum(axis=-1)  # (B, S): at risk in each state at t_k
        missing = n_from == 0
        with np.errstate(invalid="ignore", divide="ignore"):
            rows = counts[:, k] / n_from[..., None]
        rows = np.where(missing[..., None], eye, rows)
        empty += np.sum(missing & (p[:, k] > 0.0), axis=-1)
        p[:, k + 1] = np.einsum("ba,bac->bc", p[:, k], rows)
    return p, empty


def aalen_johansen(active: np.ndarray, observed: np.ndarray, n_states: int) -> AalenJohansen:
    active = np.asarray(active, dtype=np.int64)
    n = active.shape[0]
    if n == 0:
        raise ValueError("aalen_johansen needs at least one trajectory")
    initial = np.bincount(active[:, 0], minlength=n_states)[None, :] / n
    counts = transition_indicators(active, observed, n_states).sum(axis=0)[None]
    p, empty = _product(initial, counts)
    return AalenJohansen(p[0], int(empty[0]))


def resample_weights(indices: np.ndarray, n: int) -> np.ndarray:
    """(B, n): how often each trajectory appears in each bootstrap resample."""
    return np.stack([np.bincount(np.asarray(idx), minlength=n) for idx in indices]).astype(float)


def bootstrap_aalen_johansen(
    active: np.ndarray, observed: np.ndarray, n_states: int, indices: np.ndarray
) -> np.ndarray:
    """(B, G, S): the estimator on every resample (rows of `indices`), vectorised."""
    active = np.asarray(active, dtype=np.int64)
    n = active.shape[0]
    w = resample_weights(indices, n)
    x = transition_indicators(active, observed, n_states)
    counts = np.tensordot(w, x, axes=(1, 0))  # (B, G-1, S, S)
    initial = w @ np.eye(n_states)[active[:, 0]] / n
    p, _ = _product(initial, counts)
    return p


def at_risk_average(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """(G, ...): the mean of `values` (n, G, ...) over the trajectories observed at each time."""
    w = np.asarray(observed, dtype=float)
    v = np.asarray(values, dtype=float)
    num = np.einsum("ng,ng...->g...", w, v)
    den = w.sum(axis=0).reshape((-1,) + (1,) * (v.ndim - 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        result: np.ndarray = num / den
    return result


def bootstrap_at_risk_average(
    values: np.ndarray, observed: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """(B, G, ...): `at_risk_average` on every resample."""
    v = np.asarray(values, dtype=float)
    w = resample_weights(indices, v.shape[0])[:, :, None] * np.asarray(observed, dtype=float)[None]
    num = np.einsum("bng,ng...->bg...", w, v)
    den = w.sum(axis=1).reshape(w.shape[0], -1, *((1,) * (v.ndim - 2)))
    with np.errstate(invalid="ignore", divide="ignore"):
        result: np.ndarray = num / den
    return result
```

- [ ] **Step 4: Add `first_jump_index` and `bootstrap_fit_curves`, with tests**

Append to `tests/analysis/test_energy.py`:

```python
from gqteashmd.analysis.energy import first_jump_index
from gqteashmd.core.units import HARTREE_IN_EV


def test_first_jump_index():
    e = np.array([0.0, 0.001, 0.001 + 0.2 / HARTREE_IN_EV, 0.0])
    assert first_jump_index(e, 0.1) == 2
    assert first_jump_index(e[:2], 0.1) is None
    assert first_jump_index(np.array([1.0]), 0.1) is None
```

(Add `import numpy as np` if the file lacks it; put the imports at the top of the file.)

Append to `src/gqteashmd/analysis/energy.py`:

```python
def first_jump_index(e_total_hartree: np.ndarray, threshold_eV: float) -> int | None:
    """Index of the first record whose |E_tot - E_tot(previous record)| exceeds threshold_eV."""
    e = np.asarray(e_total_hartree, dtype=float)
    if e.size < 2:
        return None
    above = np.flatnonzero(np.abs(np.diff(e)) * HARTREE_IN_EV > threshold_eV)
    return int(above[0]) + 1 if above.size else None
```

Append to `tests/analysis/test_kinetics.py`:

```python
def test_bootstrap_fit_is_bootstrap_fit_curves_on_the_means():
    from gqteashmd.analysis.kinetics import bootstrap_fit, bootstrap_fit_curves

    rng = np.random.default_rng(4)
    t = np.arange(0.0, 100.5, 0.5)
    per = (rng.exponential(40.0, size=(60, 1)) + 10.0 > t[None, :]).astype(float)
    idx = rng.integers(0, 60, size=(50, 60))
    a = bootstrap_fit(t, per, idx, dt_out=0.5)
    b = bootstrap_fit_curves(
        t, per.mean(axis=0), (per[i].mean(axis=0) for i in idx), n_trajectories=60, dt_out=0.5
    )
    assert a == b
```

In `src/gqteashmd/analysis/kinetics.py`, replace `bootstrap_fit` with the two functions below (the arithmetic is the old function's, in the same order; add `from collections.abc import Iterable`):

```python
def bootstrap_fit(
    t: np.ndarray, per_trajectory: np.ndarray, indices: np.ndarray, *, dt_out: float
) -> FitWithUncertainty:
    return bootstrap_fit_curves(
        t,
        per_trajectory.mean(axis=0),
        (per_trajectory[idx].mean(axis=0) for idx in indices),
        n_trajectories=int(per_trajectory.shape[0]),
        dt_out=dt_out,
    )


def bootstrap_fit_curves(
    t: np.ndarray,
    curve: np.ndarray,
    resampled: Iterable[np.ndarray],
    *,
    n_trajectories: int,
    dt_out: float,
) -> FitWithUncertainty:
    """The fit of `curve` with percentile intervals from the fits of the `resampled` curves."""
    fit = fit_delayed_exponential(t, curve, dt_out=dt_out)
    if fit.tau_fs is None or n_trajectories < 2:
        # a resample of one trajectory is that trajectory: no information on the spread
        return FitWithUncertainty(fit, None, None, None, None, 0)
    taus: list[float] = []
    t0s: list[float] = []
    failures = 0
    for y in resampled:
        r = fit_delayed_exponential(t, y, dt_out=dt_out)
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

- [ ] **Step 5: Run the analysis tests**

Run: `pytest tests/analysis -q -W error -m "not slow"`, then `pytest tests/analysis/test_censoring.py -q -W error -m slow --durations=5`.
Expected: all PASS, with the coverage test inside [0.90, 0.99]. Record in the report the Markov test's worst z for each estimator, the coverage and the durations. If an asserted rule fails, do not change a bound, a seed or a point: report it as BLOCKED (spec 4.2, last paragraph).

- [ ] **Step 6: Lint and commit**

```bash
ruff format . && ruff check . && mypy src
git add src/gqteashmd/analysis/censoring.py src/gqteashmd/analysis/energy.py src/gqteashmd/analysis/kinetics.py tests/analysis/test_censoring.py tests/analysis/test_energy.py tests/analysis/test_kinetics.py
git commit -m "feat: the Aalen-Johansen estimator for censored trajectories"
```

---

### Task 3: Which trajectories are censored, and the at-risk grid (`censoring.select_censored`, `grid.on_grid`)

**Files:**
- Modify: `src/gqteashmd/analysis/censoring.py` (add `Censored`, `censoring_step`, `select_censored`)
- Modify: `src/gqteashmd/analysis/grid.py` (`GridData.observed`; `on_grid(..., censored_at=, horizon=)`)
- Test: `tests/analysis/test_censoring.py`, `tests/analysis/test_grid.py`

**Interfaces:**
- Consumes: `grid.Trajectory`, `grid.Excluded`, `grid.INCLUDED_OUTCOMES`, `energy.first_jump_index` (Task 2).
- Produces:
  - `censoring.Censored(task: str, outcome: str, reason: str, grid_step: int, time_fs: float)` (frozen). `reason` is `"outcome"` or `"energy_jump"`.
  - `censoring.censoring_step(t: Trajectory, *, output_every: int, before_record: int | None = None) -> int`.
  - `censoring.select_censored(tasks: Sequence[str], trajectories: Mapping[str, Trajectory | None], *, output_every: int, n_total: int, dt_fs: float, energy_jump_eV: float | None) -> tuple[list[Trajectory], dict[str, int], list[Censored], list[Excluded]]`. It returns, in task order:
    - the used trajectories;
    - task -> censoring grid step;
    - the censored entries;
    - the excluded ones.
  - `grid.GridData` gains the last field `observed: np.ndarray` (n, G) bool. `grid.on_grid(included, *, output_every, dt_fs, n_total, n_states, censored_at: Mapping[str, int] | None = None, horizon: int | None = None) -> GridData`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/analysis/test_censoring.py`:

```python
from gqteashmd.analysis.grid import Trajectory, on_grid
from gqteashmd.core.units import HARTREE_IN_EV


def _traj(task, outcome, active, *, e_total=None, every=1):
    n = len(active)
    return Trajectory(
        task=task,
        outcome=outcome,
        initial_state=int(active[0]) if n else 1,
        grid_step=np.arange(n, dtype=np.int64),
        ticks=np.zeros(n, dtype=np.int64),
        time_fs=0.5 * np.arange(n),
        active=np.asarray(active, dtype=np.int64),
        hop_from=np.full(n, -1),
        hop_to=np.full(n, -1),
        frustrated=np.zeros(n, dtype=bool),
        e_total=np.zeros(n) if e_total is None else np.asarray(e_total, dtype=float),
        populations=np.eye(2)[np.asarray(active, dtype=np.int64)],
    )


def test_select_censored_sorts_every_ending():
    trajs = {
        "traj_00000": _traj("traj_00000", "finished", [1, 1, 0, 0, 0]),
        "traj_00001": _traj("traj_00001", "overlap_failure", [1, 0, 0]),
        "traj_00002": _traj("traj_00002", "electronic_failure", []),
        "traj_00003": None,
        "traj_00004": _traj(
            "traj_00004", "finished", [1, 1, 1, 1, 1], e_total=[0, 0, 0, 0.2 / HARTREE_IN_EV, 0]
        ),
    }
    tasks = list(trajs)
    used, at, cens, excl = censoring.select_censored(
        tasks, trajs, output_every=1, n_total=4, dt_fs=0.5, energy_jump_eV=0.1
    )
    assert [t.task for t in used] == ["traj_00000", "traj_00001", "traj_00004"]
    assert at == {"traj_00001": 2, "traj_00004": 2}  # the jump is at record 3: censored at 2
    assert [(c.task, c.reason, c.grid_step, c.time_fs) for c in cens] == [
        ("traj_00001", "outcome", 2, 1.0),
        ("traj_00004", "energy_jump", 2, 1.0),
    ]
    assert [(e.task, e.reason) for e in excl] == [
        ("traj_00002", "electronic_failure"),
        ("traj_00003", "task_failed"),
    ]
    _, at_off, _, _ = censoring.select_censored(
        tasks, trajs, output_every=1, n_total=4, dt_fs=0.5, energy_jump_eV=None
    )
    assert at_off == {"traj_00001": 2}  # exclude_energy_jumps = false: no jump censoring


def test_on_grid_marks_the_censored_part_as_unobserved():
    a = _traj("a", "finished", [1, 1, 0, 0, 0])
    b = _traj("b", "overlap_failure", [1, 0, 0])
    g = on_grid([a, b], output_every=1, dt_fs=0.5, n_total=4, n_states=2, censored_at={"b": 2})
    np.testing.assert_array_equal(g.observed, [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    assert g.cut == () and g.horizon_steps == 4
    g2 = on_grid([a], output_every=1, dt_fs=0.5, n_total=4, n_states=2, horizon=2)
    assert g2.horizon_steps == 2 and g2.observed.shape == (1, 3) and g2.observed.all()
```

Add to `tests/analysis/test_grid.py` one assertion in an existing `on_grid` test, or as a new test using that file's builders, that a plain `on_grid` call returns `observed` all `True` with shape `active.shape`.

- [ ] **Step 2: Run them to see them fail**

Run: `pytest tests/analysis/test_censoring.py tests/analysis/test_grid.py -q -W error -m "not slow"`
Expected: FAIL (`select_censored` and `observed` do not exist).

- [ ] **Step 3: Implement**

Append to `src/gqteashmd/analysis/censoring.py`, keeping the imports at the top of the file:

```python
from collections.abc import Mapping, Sequence

from gqteashmd.analysis.energy import first_jump_index
from gqteashmd.analysis.grid import INCLUDED_OUTCOMES, Excluded, Trajectory


@dataclass(frozen=True)
class Censored:
    task: str
    outcome: str
    reason: str  # "outcome" (ended early) or "energy_jump" (exclude_energy_jumps = true)
    grid_step: int  # the last full-step grid point used
    time_fs: float


def censoring_step(t: Trajectory, *, output_every: int, before_record: int | None = None) -> int:
    """The last grid step (a multiple of output_every) with a full-step record, among the
    records before index `before_record` (all records if None); 0 if there is none."""
    stop = t.grid_step.size if before_record is None else int(before_record)
    steps = t.grid_step[:stop]
    full = steps[(t.ticks[:stop] == 0) & (steps % output_every == 0)]
    return int(full.max()) if full.size else 0


def select_censored(
    tasks: Sequence[str],
    trajectories: Mapping[str, Trajectory | None],
    *,
    output_every: int,
    n_total: int,
    dt_fs: float,
    energy_jump_eV: float | None,
) -> tuple[list[Trajectory], dict[str, int], list[Censored], list[Excluded]]:
    """Spec section 2.1: complete and censored trajectories are used, the others excluded.

    `energy_jump_eV` is the threshold when exclude_energy_jumps = true, else None: a
    trajectory is then censored before its first jump over grid steps <= n_total (P8 PR3).
    """
    used: list[Trajectory] = []
    at: dict[str, int] = {}
    censored: list[Censored] = []
    excluded: list[Excluded] = []
    for task in tasks:
        t = trajectories.get(task)
        if t is None:
            excluded.append(Excluded(task, "task_failed", None))
            continue
        if t.time_fs.size == 0:
            excluded.append(Excluded(task, t.outcome, t.t_final_fs))
            continue
        jump_at = None
        if energy_jump_eV is not None:
            window = int(np.count_nonzero(t.grid_step <= n_total))  # records are time-ordered
            jump_at = first_jump_index(t.e_total[:window], energy_jump_eV)
        if jump_at is not None:
            reason = "energy_jump"
            step = censoring_step(t, output_every=output_every, before_record=jump_at)
        elif t.outcome in INCLUDED_OUTCOMES:
            used.append(t)
            continue
        else:
            reason = "outcome"
            step = censoring_step(t, output_every=output_every)
        used.append(t)
        at[task] = step
        censored.append(Censored(task, t.outcome, reason, step, step * float(dt_fs)))
    return used, at, censored, excluded
```

In `src/gqteashmd/analysis/grid.py`:
- add `observed: np.ndarray` as the **last** field of `GridData`;
- extend `on_grid` with the keyword arguments `censored_at: Mapping[str, int] | None = None` and `horizon: int | None = None`.

The body becomes:

```python
    h = horizon_steps(included, n_total) if horizon is None else int(horizon)
    censored_at = censored_at or {}
    steps = np.arange(0, h + 1, output_every, dtype=np.int64)
    active = np.zeros((len(included), steps.size), dtype=np.int64)
    populations = np.zeros((len(included), steps.size, n_states))
    observed = np.ones((len(included), steps.size), dtype=bool)
    ground = np.zeros(n_states)
    ground[0] = 1.0
    cut: list[str] = []
    for i, t in enumerate(included):
        full = np.flatnonzero(t.ticks == 0)
        where = {int(t.grid_step[j]): int(j) for j in full}
        end = t.last_full_step
        stop = censored_at.get(t.task)
        for k, s in enumerate(steps):
            if stop is not None and s > stop:
                observed[i, k] = False  # censored: no data used after its censoring step
                continue
            j = where.get(int(s))
            if j is not None:
                active[i, k] = int(t.active[j])
                populations[i, k] = t.populations[j]
            elif t.outcome == STOPPED_IN_S0 and s > end:
                populations[i, k] = ground
            else:
                raise InconsistentData(f"{t.task}: no record at grid step {int(s)}")
        if stop is None and np.any((t.grid_step > h) | ((t.grid_step == h) & (t.ticks > 0))):
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
        observed,
    )
```

Without `censored_at` and `horizon` the result is the old one plus `observed` all `True`.

- [ ] **Step 4: Run the analysis tests**

Run: `pytest tests/analysis -q -W error -m "not slow"`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
ruff format . && ruff check . && mypy src
git add src/gqteashmd/analysis/censoring.py src/gqteashmd/analysis/grid.py tests/analysis/test_censoring.py tests/analysis/test_grid.py
git commit -m "feat: censored trajectories and the at-risk grid of the analysis"
```

---

### Task 4: The `estimator` key and the per-estimator results (`config`, `analysis/estimators.py`)

**Files:**
- Modify: `src/gqteashmd/config/schema.py:321-...` (`AnalysisSection`)
- Modify: `src/gqteashmd/cli.py:140-146` (the `init` template)
- Create: `src/gqteashmd/analysis/estimators.py`
- Test: `tests/config/test_analysis_section.py`, `tests/analysis/test_estimators.py` (new)

**Interfaces:**
- Consumes: Tasks 2 and 3 (`censoring.*`, `GridData.observed`, `kinetics.bootstrap_fit_curves`).
- Produces:
  - `AnalysisSection.estimator: Literal["censored", "complete"] = "censored"`.
  - `estimators.Estimate` (frozen dataclass) with the fields:
    - `name: str` and `n: int`;
    - `classical`, `classical_lower`, `classical_upper`, `quantum`, `quantum_lower`, `quantum_upper`, `target`, all `np.ndarray`;
    - `fit: kinetics.FitWithUncertainty`;
    - `empty_rows: int | None`.
  - `estimators.complete_estimate(g: GridData, fit_index: Sequence[int], *, fit_population: str, rng: np.random.Generator, n_bootstrap: int, dt_out: float) -> Estimate`.
  - `estimators.censored_estimate(g: GridData, fit_index: Sequence[int], *, rng: np.random.Generator, n_bootstrap: int, dt_out: float) -> Estimate`.

- [ ] **Step 1: Write the failing config tests** (append to `tests/config/test_analysis_section.py`; `_load` is that file's helper)

```python
def test_estimator_defaults_to_censored(tmp_path, write_input):
    assert _load(tmp_path, write_input, "").analysis.estimator == "censored"


def test_estimator_complete_is_accepted(tmp_path, write_input):
    a = _load(tmp_path, write_input, '[analysis]
estimator = "complete"
').analysis
    assert a.estimator == "complete"


def test_an_unknown_estimator_is_rejected(tmp_path, write_input):
    with pytest.raises(ConfigError, match="estimator"):
        _load(tmp_path, write_input, '[analysis]
estimator = "kaplan"
')


def test_a_quantum_fit_needs_the_complete_estimator(tmp_path, write_input):
    with pytest.raises(ConfigError, match='estimator = "complete"'):
        _load(tmp_path / "a", write_input, '[analysis]
fit_population = "quantum"
')
    text = '[analysis]
fit_population = "quantum"
estimator = "complete"
'
    assert _load(tmp_path / "b", write_input, text).analysis.fit_population == "quantum"
```

Add `assert a.estimator == "censored"` to `test_defaults_without_a_table`. Any existing test in the file that sets `fit_population = "quantum"` and expects it to load must also set `estimator = "complete"`. If the file has tests that compare `AnalysisSection().model_dump()` with a literal dict, add `"estimator": "censored"` to that dict.

- [ ] **Step 2: Implement the key**

In `AnalysisSection` add `estimator: Literal["censored", "complete"] = "censored"` after `fit_population`, and the validator:

```python
    @model_validator(mode="after")
    def _quantum_fit_needs_complete(self) -> AnalysisSection:
        if self.fit_population == "quantum" and self.estimator == "censored":
            raise ValueError(
                'fit_population = "quantum" needs estimator = "complete" (the censored '
                "estimator applies to the classical populations only; USER_MANUAL 8.13)"
            )
        return self
```

In the `init` template of `cli.py`, insert after the `fit_population` line:

```
# estimator = "censored"             # or "complete": only finished trajectories
```

- [ ] **Step 3: Write the failing estimator tests** (`tests/analysis/test_estimators.py`)

```python
"""Per-estimator results (censored analysis spec, sections 2.2-2.4)."""

import numpy as np

from gqteashmd.analysis import estimators, kinetics, populations
from gqteashmd.analysis.grid import GridData


def _grid(active, observed):
    active = np.asarray(active, dtype=np.int64)
    n, g = active.shape
    return GridData(
        np.arange(g),
        0.5 * np.arange(g),
        active,
        np.eye(3)[active],
        tuple(f"t{i}" for i in range(n)),
        g - 1,
        0.5 * (g - 1),
        (),
        np.asarray(observed, dtype=bool),
    )


def _decaying(rng, n=40, g=81):
    times = 10 + rng.exponential(10.0, size=n)
    active = np.where(0.5 * np.arange(g)[None, :] < times[:, None], 1, 0)
    return active


def test_complete_estimate_equals_the_old_stage_arithmetic():
    rng = np.random.default_rng(7)
    g = _grid(_decaying(rng), np.ones((40, 81), dtype=bool))
    est = estimators.complete_estimate(
        g, [1, 2], fit_population="classical", rng=np.random.default_rng(9), n_bootstrap=100,
        dt_out=0.5,
    )
    p, counts = populations.classical_populations(g.active, 3)
    np.testing.assert_array_equal(est.classical, p)
    idx = populations.bootstrap_indices(np.random.default_rng(9), 40, 100)
    per = np.isin(g.active, [1, 2]).astype(float)
    assert est.fit == kinetics.bootstrap_fit(g.time_fs, per, idx, dt_out=0.5)
    assert est.n == 40 and est.empty_rows is None


def test_censored_estimate_without_censoring_has_the_same_curve():
    rng = np.random.default_rng(7)
    g = _grid(_decaying(rng), np.ones((40, 81), dtype=bool))
    est = estimators.censored_estimate(
        g, [1, 2], rng=np.random.default_rng(10), n_bootstrap=100, dt_out=0.5
    )
    p, _ = populations.classical_populations(g.active, 3)
    assert np.abs(est.classical - p).max() <= 1e-12
    assert est.empty_rows == 0
    np.testing.assert_allclose(est.target, 1.0 - p[:, 0], atol=1e-12)
    assert est.fit.fit.tau_fs is not None and est.fit.tau_interval_fs is not None


def test_censored_quantum_is_the_at_risk_average():
    active = np.array([[1, 1, 0], [1, 0, 0]])
    observed = np.array([[True, True, True], [True, True, False]])
    est = estimators.censored_estimate(
        _grid(active, observed), [1, 2], rng=np.random.default_rng(1), n_bootstrap=100, dt_out=0.5
    )
    np.testing.assert_allclose(est.quantum[2], [1.0, 0.0, 0.0])  # only trajectory 0 at t_2
```

- [ ] **Step 4: Implement `src/gqteashmd/analysis/estimators.py`**

```python
"""The populations, intervals and fit of each estimator (censored analysis spec, section 2).

`complete_estimate` is the P8 arithmetic unchanged (the stage's numbers before this
sub-project); `censored_estimate` is the Aalen-Johansen curve with bootstrap intervals and the
fit to its excited-state sum. The caller passes the random generator (seed rule D12).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from gqteashmd.analysis import censoring, kinetics, populations
from gqteashmd.analysis.grid import GridData


@dataclass(frozen=True)
class Estimate:
    name: str
    n: int
    classical: np.ndarray
    classical_lower: np.ndarray
    classical_upper: np.ndarray
    quantum: np.ndarray
    quantum_lower: np.ndarray
    quantum_upper: np.ndarray
    target: np.ndarray
    fit: kinetics.FitWithUncertainty
    empty_rows: int | None


def _percentiles(boot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2.5/97.5 % over the resamples; NaN where a resample is not finite (nobody at risk)."""
    lo = np.full(boot.shape[1:], np.nan)
    hi = np.full(boot.shape[1:], np.nan)
    ok = np.all(np.isfinite(boot), axis=0)
    if np.any(ok):
        lo[ok], hi[ok] = np.percentile(boot[:, ok], [2.5, 97.5], axis=0)
    return lo, hi


def complete_estimate(
    g: GridData,
    fit_index: Sequence[int],
    *,
    fit_population: str,
    rng: np.random.Generator,
    n_bootstrap: int,
    dt_out: float,
) -> Estimate:
    n = int(g.active.shape[0])
    n_states = int(g.populations.shape[-1])
    p_classical, n_counts = populations.classical_populations(g.active, n_states)
    p_lo, p_hi = populations.wilson_intervals(n_counts, n)
    rho = populations.quantum_populations(g.populations)
    idx = populations.bootstrap_indices(rng, n, n_bootstrap)
    rho_lo, rho_hi = populations.bootstrap_intervals(g.populations, idx)
    if fit_population == "classical":
        per = np.isin(g.active, list(fit_index)).astype(float)
    else:
        per = g.populations[:, :, list(fit_index)].sum(axis=-1)
    fit = kinetics.bootstrap_fit(g.time_fs, per, idx, dt_out=dt_out)
    return Estimate(
        "complete", n, p_classical, p_lo, p_hi, rho, rho_lo, rho_hi, per.mean(axis=0), fit, None
    )


def censored_estimate(
    g: GridData,
    fit_index: Sequence[int],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    dt_out: float,
) -> Estimate:
    n = int(g.active.shape[0])
    n_states = int(g.populations.shape[-1])
    aj = censoring.aalen_johansen(g.active, g.observed, n_states)
    idx = populations.bootstrap_indices(rng, n, n_bootstrap)
    boot = censoring.bootstrap_aalen_johansen(g.active, g.observed, n_states, idx)
    p_lo, p_hi = _percentiles(boot)
    rho = censoring.at_risk_average(g.populations, g.observed)
    rho_lo, rho_hi = _percentiles(censoring.bootstrap_at_risk_average(g.populations, g.observed, idx))
    columns = list(fit_index)
    target = aj.populations[:, columns].sum(axis=1)
    fit = kinetics.bootstrap_fit_curves(
        g.time_fs, target, boot[:, :, columns].sum(axis=-1), n_trajectories=n, dt_out=dt_out
    )
    return Estimate(
        "censored", n, aj.populations, p_lo, p_hi, rho, rho_lo, rho_hi, target, fit, aj.empty_rows
    )
```

Note that `bootstrap_fit_curves` iterates over the rows of the (B, G) array.

- [ ] **Step 5: Run the tests**

Run: `pytest tests/config tests/analysis tests/test_cli.py -q -W error -m "not slow and not pyscf"`
Expected: PASS, except possibly the stage tests that set `fit_population = "quantum"` without `estimator`. Those are in `tests/stages/`, which Task 5 fixes, so they are not run here.

- [ ] **Step 6: Lint and commit**

```bash
ruff format . && ruff check . && mypy src
git add src/gqteashmd/config/schema.py src/gqteashmd/cli.py src/gqteashmd/analysis/estimators.py tests/config/test_analysis_section.py tests/analysis/test_estimators.py
git commit -m "feat: the [analysis] estimator key and the per-estimator results"
```

---

### Task 5: The analysis stage with both estimators (`AnalysisStage.version` 3)

**Files:**
- Modify: `src/gqteashmd/stages/analysis.py` (run_task, finalize, summary, version)
- Modify: `src/gqteashmd/analysis/artifacts.py` (the extra blocks, ruling PR6)
- Modify: `src/gqteashmd/analysis/report.py` (estimator lines and the censored table)
- Test: `tests/stages/test_analysis.py`, `tests/stages/test_analysis_pipeline.py`, `tests/analysis/test_artifacts.py`, `tests/analysis/test_report.py`, `tests/test_e2e_dynamics.py`

**Interfaces:**
- Consumes: `censoring.select_censored`, `grid.on_grid(censored_at=, horizon=)`, `estimators.complete_estimate`, `estimators.censored_estimate`, `AnalysisSection.estimator`.
- Produces: `analysis.json` `schema_version` 2 with the keys of ruling PR4. The task result gains `"estimator"`.

- [ ] **Step 1: Keep the existing stage tests on today's rule**

In `tests/stages/test_analysis.py`, give `_context` a keyword `estimator="complete"` and build the input with:

```python
        tmp_path,
        extra=ELECTRONIC
        + DYNAMICS
        + f'[analysis]\nn_bootstrap = 100\nestimator = "{estimator}"\n'
        + analysis,
```

Every existing test then exercises the unchanged `complete` path. Also update:
- `test_stage_attributes`: version 3;
- `test_fingerprint_inputs`: the hashed dict gains `"estimator"`.

- [ ] **Step 2: Write the failing stage tests** (append to `tests/stages/test_analysis.py`; `_write_traj` and `_context` are that file's builders, and `(outcome, active_by_step, kwargs)` is its trajectory format)

```python
def test_censored_estimator_uses_a_lost_trajectory_up_to_its_end(tmp_path, write_input):
    trajs = [
        (FIN, [1, 1, 1, 0, 0], {}),
        (FIN, [1, 1, 1, 1, 1], {}),
        ("overlap_failure", [1, 0, 0], {}),
    ]
    ctx = _context(tmp_path, write_input, trajs, estimator="censored")
    run_stage(ctx)
    data = _analysis(ctx)
    assert data["schema_version"] == 2 and data["estimator"] == "censored"
    assert data["n_complete"] == 2 and data["n_censored"] == 1 and data["n_included"] == 3
    assert data["censored"] == [
        {"task": "traj_00002", "outcome": "overlap_failure", "reason": "outcome",
         "censored_at_fs": 1.0}
    ]
    assert set(data["estimators"]) == {"complete", "censored"}
    assert data["estimators"]["complete"]["n"] == 2
    assert data["estimators"]["censored"]["n"] == 3
    # at t = 1.0 fs: traj 0 in S1, traj 1 in S1, traj 2 in S0 -> P(S0) = 1/3; after that the
    # censored trajectory's S0 probability is carried by the estimator, not dropped
    row = next(r for r in data["table"] if r["time_fs"] == 1.0)
    assert row["P"]["S0"] == pytest.approx(1 / 3)
    complete_row = next(
        r for r in data["estimators"]["complete"]["table"] if r["time_fs"] == 1.0
    )
    assert complete_row["P"]["S0"] == 0.0  # the complete rule dropped the decayed trajectory


def test_without_losses_the_two_estimators_agree(tmp_path, write_input):
    ctx = _context(tmp_path, write_input, _varied_trajectories(), estimator="censored")
    run_stage(ctx)
    data = _analysis(ctx)
    for a, b in zip(data["table"], data["estimators"]["complete"]["table"], strict=True):
        for s, value in a["P"].items():
            assert value == pytest.approx(b["P"][s], abs=1e-12)
    assert data["n_censored"] == 0 and data["estimators"]["censored"]["empty_rows"] == 0


def test_the_censored_fraction_warning(tmp_path, write_input, caplog):
    trajs = [(FIN, [1, 1, 1, 1, 1], {})] + [("overlap_failure", [1, 0, 0], {})] * 3
    ctx = _context(tmp_path, write_input, trajs, estimator="censored")
    run_stage(ctx)
    text = " ".join(_analysis(ctx)["warnings"])
    assert "75% of the selected trajectories are not complete" in text
    assert "3 censored, 0 without data" in text


def test_the_files_carry_the_other_estimator(tmp_path, write_input):
    h5py = pytest.importorskip("h5py")
    trajs = [(FIN, [1, 1, 1, 0, 0], {}), ("overlap_failure", [1, 0, 0], {})]
    ctx = _context(tmp_path, write_input, trajs, estimator="censored")
    run_stage(ctx)
    with h5py.File(ctx.stage_dir / "populations.h5", "r") as f:
        assert "complete_classical" in f and "complete_quantum_upper" in f
    header = next(
        line for line in (ctx.stage_dir / "populations.csv").read_text().splitlines()
        if not line.startswith("#")
    )
    assert "complete_P_S0" in header.split(",")


def test_energy_jump_censoring_keeps_the_part_before_the_jump(tmp_path, write_input):
    jump = [-1.0, -1.0, -1.0 + 0.2 / HARTREE_IN_EV, -1.0 + 0.2 / HARTREE_IN_EV, -1.0]
    trajs = [(FIN, [1, 1, 1, 1, 1], {"e_total": jump}), (FIN, [1, 1, 1, 1, 1], {})]
    ctx = _context(
        tmp_path,
        write_input,
        trajs,
        estimator="censored",
        analysis="energy_jump_eV = 0.1\nexclude_energy_jumps = true\n",
    )
    run_stage(ctx)
    data = _analysis(ctx)
    # the jump is at record 2 (1.0 fs): censored at the last grid time before it, 0.5 fs
    assert data["censored"] == [
        {"task": "traj_00000", "outcome": "finished", "reason": "energy_jump",
         "censored_at_fs": 0.5}
    ]
    assert data["excluded"] == [] and data["n_included"] == 2
    assert data["estimators"]["complete"]["n"] == 1  # the complete rule excludes it


def test_a_trajectory_lost_after_the_horizon_counts_as_complete(tmp_path, write_input):
    # the shortest finished trajectory sets the horizon at 1.0 fs (grid step 2); the lost one
    # has records up to grid step 4, so it is complete up to the horizon (ruling PR5)
    trajs = [(FIN, [1, 1, 0], {}), ("energy_drift", [1, 1, 1, 0, 0], {})]
    ctx = _context(tmp_path, write_input, trajs, estimator="censored")
    run_stage(ctx)
    data = _analysis(ctx)
    assert data["horizon_fs"] == pytest.approx(1.0)
    assert data["n_censored"] == 0 and data["censored"] == [] and data["n_included"] == 2
    assert data["estimators"]["complete"]["n"] == 1
```

`HARTREE_IN_EV` is already imported by the test file (used by `test_energy_jump_flag_and_exclusion`).

- [ ] **Step 3: Run them to see them fail**

Run: `pytest tests/stages/test_analysis.py -q -W error -m "not slow"`
Expected: the new tests FAIL (`KeyError: 'estimator'` or similar). The existing ones still PASS, except the version and fingerprint tests that Step 1 changed.

- [ ] **Step 4: Implement the stage**

In `src/gqteashmd/stages/analysis.py`:

1. `version = 3`, with the comment: `# 3: the censored estimator (default) beside the complete rule (censored analysis spec)`.
2. Import `censoring` and `estimators` with the other analysis modules.
3. In `run_task`, after `jumps` is computed, replace everything from `included, excluded = grid.select(...)` up to the end of the fit block with the sequence below. Keep the hop, energy, summary and writer parts, changed as shown in points 4-6.

```python
        tasks = [e.task_id for e in selected]
        complete, excluded_complete = grid.select(
            tasks,
            trajectories,
            jumps,
            energy_jump_eV=section.energy_jump_eV,
            exclude_energy_jumps=section.exclude_energy_jumps,
        )
        used, censor_steps, censored, excluded_censored = censoring.select_censored(
            tasks,
            trajectories,
            output_every=dyn.output_every,
            n_total=n_total,
            dt_fs=dyn.dt_fs,
            energy_jump_eV=section.energy_jump_eV if section.exclude_energy_jumps else None,
        )
        for t in used:  # `used` contains every complete trajectory
            _check_included(t, files[t.task])
        h = grid.horizon_steps(complete, n_total)  # ruling PR7
        censor_steps = {k: v for k, v in censor_steps.items() if v < h}  # ruling PR5
        censored = [c for c in censored if c.task in censor_steps]
        in_use = used if section.estimator == "censored" else complete
        excluded = excluded_censored if section.estimator == "censored" else excluded_complete
```

`_base_summary` now takes `in_use`, `excluded`, `complete`, `censored` and the estimator. The `no trajectory` stop tests `if not in_use:`, and under `censored` its message reads `No trajectory can be analysed: outcomes ...; no selected trajectory has a record (USER_MANUAL 8.13)`; the `complete` message is unchanged. The norm check loops over `used`.

The two grids and estimates:

```python
        def _grid(trajs, at):
            try:
                return grid.on_grid(
                    trajs,
                    output_every=dyn.output_every,
                    dt_fs=dyn.dt_fs,
                    n_total=n_total,
                    n_states=len(labels),
                    censored_at=at,
                    horizon=h,
                )
            except grid.InconsistentData as exc:
                raise _task_error(f"{exc}; {DYNAMICS_ADVICE}") from exc

        seed = ctx.config.project.seed
        g_complete = _grid(complete, None) if complete else None
        g_censored = _grid(used, censor_steps)
        est_complete = (
            estimators.complete_estimate(
                g_complete,
                fit_index,
                fit_population=section.fit_population,
                rng=task_rng(seed, "analysis", 0),
                n_bootstrap=section.n_bootstrap,
                dt_out=dt_out,
            )
            if g_complete is not None
            else None
        )
        est_censored = estimators.censored_estimate(
            g_censored,
            fit_index,
            rng=task_rng(seed, "analysis", 1),
            n_bootstrap=section.n_bootstrap,
            dt_out=dt_out,
        )
        est = est_censored if section.estimator == "censored" else est_complete
        assert est is not None  # `in_use` is not empty
        g = g_censored if section.estimator == "censored" else g_complete
        assert g is not None
        n = est.n
```

The small-ensemble warning, the fit flags and the bootstrap-failure warning use `est` and `n`. Under `complete` the small-ensemble text is unchanged. Under `censored` it reads:

`f"only {n} trajectories are used ({n - len(censored)} complete, {len(censored)} censored; fewer than {SMALL_ENSEMBLE}): populations, intervals and the fit are not statistically reliable (measured 95 % tau interval {SMALL_ENSEMBLE_COVERAGE}, for complete ensembles)"`

`fit`, `target`, `curve` and the `arrays` dict take their values from `est`: `est.classical`, `est.classical_lower`, ..., `est.target`, and `model_curve` from `est.fit.fit`.

4. Hops and drifts stay over `complete` (ruling PR4): `hop_events(..., included=traj.task in {t.task for t in complete}, horizon_fs=g.horizon_fs)`, `hop_statistics(events, complete, labels, g.horizon_fs)`, and the drifts over `complete`. The included-jump warning counts the flagged tasks in `{t.task for t in in_use}`. `consistency` comes from `est_complete` (`populations.consistency(est_complete.classical, est_complete.quantum)`), or is `None` without one.

5. The summary (ruling PR4) adds:

```python
                "estimator": section.estimator,
                "estimators": {
                    "complete": None if est_complete is None else _estimate_summary(
                        est_complete, g_complete, labels, fit_states, section.fit_population
                    ),
                    "censored": _estimate_summary(
                        est_censored, g_censored, labels, fit_states, "classical"
                    ),
                },
```

with the new helper:

```python
def _fit_summary(
    est: estimators.Estimate, fit_states: Sequence[str], population: str
) -> dict[str, Any]:
    fitu, fit = est.fit, est.fit.fit
    return {
        "states": list(fit_states),
        "population": population,
        "A": fit.a,
        "t0_fs": fit.t0_fs,
        "tau_fs": fit.tau_fs,
        "t0_interval_fs": fitu.t0_interval_fs,
        "tau_interval_fs": fitu.tau_interval_fs,
        "t0_std_fs": fitu.t0_std_fs,
        "tau_std_fs": fitu.tau_std_fs,
        "rms": fit.rms,
        "flags": list(fit.flags),
        "bootstrap_failures": fitu.bootstrap_failures,
    }


def _estimate_summary(
    est: estimators.Estimate,
    g: grid.GridData,
    labels: Sequence[str],
    fit_states: Sequence[str],
    population: str,
) -> dict[str, Any]:
    out = {
        "n": est.n,
        "small_ensemble": est.n < SMALL_ENSEMBLE,
        "fit": _fit_summary(est, fit_states, population),
        "table": _table(g.time_fs, labels, est.classical, est.quantum),
    }
    if est.name == "censored":
        out["empty_rows"] = est.empty_rows
        out["quantum_rule"] = "at_risk_average"
    return out
```

The top-level `"fit"` is `_fit_summary(est, fit_states, section.fit_population if section.estimator == "complete" else "classical")`, and `"table"` is `_table(g.time_fs, labels, est.classical, est.quantum)`.

`_base_summary` returns `schema_version` 2 and the additional keys:
- `"estimator"`;
- `"n_complete": len(complete)`;
- `"n_censored": len(censored)`;
- `"censored": [{"task": c.task, "outcome": c.outcome, "reason": c.reason, "censored_at_fs": c.time_fs} for c in censored]`;
- `"estimators": None` (filled later);
- `"n_included": len(in_use)`, `"included": [t.task for t in in_use]` and `"small_ensemble": len(in_use) < SMALL_ENSEMBLE`.

6. The writers (ruling PR6). `artifacts.write_populations_h5` gains the keyword argument `extra: Mapping[str, ArrayLike] | None = None`, written as further datasets with `units` `"-"`. `artifacts.format_populations_csv` gains `extra_blocks: Sequence[tuple[str, ArrayLike, ArrayLike, ArrayLike]] = ()`, one `(prefix, mean, lower, upper)` per block. Its columns `f"{prefix}_{label}"`, `_lower` and `_upper` are appended after `fit_curve`, in block order. The stage passes the other estimator (`other = est_complete if section.estimator == "censored" else est_censored`), when it is not `None`, as:

```python
        prefix = other.name  # "complete" or "censored"
        extra = {
            f"{prefix}_classical": other.classical,
            f"{prefix}_classical_lower": other.classical_lower,
            f"{prefix}_classical_upper": other.classical_upper,
            f"{prefix}_quantum": other.quantum,
            f"{prefix}_quantum_lower": other.quantum_lower,
            f"{prefix}_quantum_upper": other.quantum_upper,
        }
        extra_blocks = [
            (f"{prefix}_P", other.classical, other.classical_lower, other.classical_upper),
            (f"{prefix}_rho", other.quantum, other.quantum_lower, other.quantum_upper),
        ]
```

The h5 attribute `n_included` stays, and new attributes `estimator`, `n_complete` and `n_censored` are added. The CSV comments gain `estimator = <name>`. Add one test to `tests/analysis/test_artifacts.py` that writes an extra block to both files and reads it back.

7. `finalize`. When `summary["estimator"] == "censored"`:

```python
        n_not_complete = n_selected - (int(summary["n_included"]) - int(summary["n_censored"]))
        fraction = n_not_complete / n_selected if n_selected else 0.0
        if fraction > limit:
            summary["warnings"].append(
                f"{fraction:.0%} of the selected trajectories are not complete (more than "
                f"warn_excluded_fraction = {limit}): {summary['n_censored']} censored, "
                f"{len(summary['excluded'])} without data; the censored estimator assumes "
                "that a censored trajectory would have gone on like those still running in "
                "its state (USER_MANUAL 8.13)"
            )
```

Under `complete` the existing computation and text are unchanged.

8. The result dict gains `"estimator": section.estimator`.

9. `report.analysis_markdown`:
- add `estimator` to the settings keys and the lines `- n_complete:` and `- n_censored:` after `n_included`;
- add a section `## Censored trajectories` (a table `| task | outcome | reason | censored_at_fs |`, or `none censored`) after the excluded table;
- after `## Fit`, add `## Other estimator`, which prints that estimator's `tau_fs`, `tau_interval_fs`, `t0_fs` and `t0_interval_fs` from `estimators`, or `-`.

Add tests to `tests/analysis/test_report.py` for the three new sections, with and without data.

- [ ] **Step 5: Update the pipeline and end-to-end tests**

- `tests/stages/test_analysis_pipeline.py`: the canned trajectories all finish. Add a test that the censored default gives `P` within 1e-12 of `estimators.complete.table` at every row (spec 4.1 item 1, on the canned pipeline). Fix any assertion that fails on `schema_version` or on the new keys.
- `tests/test_e2e_dynamics.py:69` (a `pyscf` test): replace `assert set(analysis["included"]) == included` with:

```python
    censored = {c["task"] for c in analysis["censored"]}
    assert set(analysis["included"]) == included | censored
    assert analysis["n_complete"] == len(included)
```

- [ ] **Step 6: Run the tests**

Run: `pytest tests/analysis tests/stages/test_analysis.py tests/stages/test_analysis_pipeline.py tests/config tests/test_orchestrator.py -q -W error -m "not pyscf"`
Then: `pytest tests/test_e2e_dynamics.py -q -W error -m "not slow"`.
Expected: PASS. Record both counts and durations in the report.

- [ ] **Step 7: Lint and commit**

```bash
ruff format . && ruff check . && mypy src
git add src/gqteashmd/stages/analysis.py src/gqteashmd/analysis/artifacts.py src/gqteashmd/analysis/report.py tests/stages/test_analysis.py tests/stages/test_analysis_pipeline.py tests/analysis/test_artifacts.py tests/analysis/test_report.py tests/test_e2e_dynamics.py
git commit -m "feat: the analysis stage reports the censored and the complete estimator"
```

---

### Task 6: The report script (`tools/validation/censoring_report.py`)

**Files:**
- Create: `tools/validation/censoring_report.py`
- Test: `tests/validation/test_censoring_report.py` (helpers on synthetic trajectories; fast tier)

**Interfaces:**
- Consumes:
  - `censoring.select_censored`, `grid.on_grid`, `estimators.complete_estimate`, `estimators.censored_estimate`, `censoring.at_risk_average`, `censoring.bootstrap_aalen_johansen`, `populations.bootstrap_indices`, `kinetics.fit_delayed_exponential`, `util.seeds.task_rng`;
  - `tools/sharc_compare/sharc_trajectories.read_our_trajectory`, `tools/diagnosis/forensics.find_stage_dir`;
  - `read_output_h5` (for `e_kinetic`).
- Produces: a CLI `censoring_report.py --artificial RUN [--project RUN ...] --out JSON [--repetitions 200] [--bootstrap 200] [--seed 20261119]`. It writes one JSON with the keys `artificial`, `sensitivity`, `back_hop_rates` and `reanalysis`.

- [ ] **Step 1: Write the helper tests** (`tests/validation/test_censoring_report.py`; the import pattern of `tests/validation/test_hop_energy_report.py`)

```python
"""Helpers of tools/validation/censoring_report.py on synthetic trajectories."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "validation"))
import censoring_report as cr  # noqa: E402

from gqteashmd.analysis.grid import Trajectory  # noqa: E402


def _traj(task, outcome, active):
    n = len(active)
    return Trajectory(
        task, outcome, int(active[0]), np.arange(n), np.zeros(n, dtype=np.int64),
        0.5 * np.arange(n), np.asarray(active), np.full(n, -1), np.full(n, -1),
        np.zeros(n, dtype=bool), np.zeros(n), np.eye(2)[np.asarray(active)],
    )


def test_loss_delays_are_measured_from_the_first_s0_record():
    lost = _traj("a", "overlap_failure", [1, 1, 0, 0, 0, 0])  # S0 from 1.0 fs, ends 2.5 fs
    kept = _traj("b", "finished", [1, 0, 0])
    assert cr.loss_delays_fs([lost, kept]) == [1.5]
    assert cr.reaching_s0_fraction([lost, kept]) == (1, 2)


def test_impose_censoring_cuts_after_the_first_s0_arrival():
    t = _traj("a", "finished", [1, 1, 0, 0, 0, 0, 0])
    assert cr.censor_step_for(t, delay_fs=1.0, dt_fs=0.5, output_every=1) == 4  # 1.0 + 1.0 fs
    assert cr.censor_step_for(t, delay_fs=99.0, dt_fs=0.5, output_every=1) is None
    assert cr.censor_step_for(_traj("b", "finished", [1, 1, 1]), 1.0, 0.5, 1) is None


def test_hold_in_s0_fills_the_censored_part_with_s0():
    active = np.array([[1, 0, 0, 1]])
    observed = np.array([[True, True, False, False]])
    np.testing.assert_array_equal(cr.hold_in_s0(active, observed), [[1, 0, 0, 0]])


def test_bias_ratio_skips_zero_units():
    mean = np.array([[0.5, 0.5], [0.6, 0.4]])
    truth = np.array([[0.5, 0.5], [0.5, 0.5]])
    unit = np.array([[0.0, 0.0], [0.05, 0.05]])
    assert cr.max_bias_ratio(mean, truth, unit) == pytest.approx(2.0)


def test_poisson_interval_brackets_the_rate():
    lo, hi = cr.poisson_rate_interval(4, 2.0)
    assert lo < 2.0 < hi
    assert cr.poisson_rate_interval(0, 2.0)[0] == 0.0
```

- [ ] **Step 2: Run them to see them fail**

Run: `pytest tests/validation/test_censoring_report.py -q -W error`
Expected: FAIL (`ModuleNotFoundError: censoring_report`).

- [ ] **Step 3: Implement the script**

```python
"""Censored analysis report (spec 2026-11-19-censored-analysis-design.md, section 4.2).

Report only: every number goes to the JSON file; the one verdict ("good", item 5) was fixed in
the spec before any run. Never imported by gqteashmd. It calls gqteashmd.analysis on existing
output.h5 files and does not run the stage (DynamicsStage.version 3 makes those projects stale).

Usage: censoring_report.py --artificial RUN [--project RUN ...] --out JSON
       [--repetitions 200] [--bootstrap 200] [--seed 20261119]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sharc_compare"))
sys.path.insert(0, str(HERE.parent / "diagnosis"))
from forensics import find_stage_dir, jsonable  # noqa: E402
from sharc_trajectories import read_our_trajectory  # noqa: E402

from gqteashmd.analysis import censoring, estimators, grid, populations  # noqa: E402
from gqteashmd.analysis.grid import INCLUDED_OUTCOMES, Trajectory  # noqa: E402
from gqteashmd.analysis.kinetics import fit_delayed_exponential  # noqa: E402
from gqteashmd.util.seeds import task_rng  # noqa: E402

REPORT_TIMES_FS = (25.0, 50.0, 75.0, 100.0)


# ------------------------------------------------------------------------- helpers
def first_s0_time(t: Trajectory) -> float | None:
    hits = np.flatnonzero(t.active == 0)
    return float(t.time_fs[hits[0]]) if hits.size else None


def loss_delays_fs(trajs: Sequence[Trajectory]) -> list[float]:
    """t_final - first S0 arrival of every lost trajectory that reached S0 (spec 1.1)."""
    out = []
    for t in trajs:
        if t.outcome in INCLUDED_OUTCOMES or t.time_fs.size == 0:
            continue
        s0 = first_s0_time(t)
        if s0 is not None:
            out.append(float(t.time_fs[-1]) - s0)
    return out


def reaching_s0_fraction(trajs: Sequence[Trajectory]) -> tuple[int, int]:
    """(lost among those reaching S0, those reaching S0)."""
    reach = [t for t in trajs if t.time_fs.size and first_s0_time(t) is not None]
    lost = [t for t in reach if t.outcome not in INCLUDED_OUTCOMES]
    return len(lost), len(reach)


def censor_step_for(
    t: Trajectory, delay_fs: float, dt_fs: float, output_every: int
) -> int | None:
    """The grid step at first S0 arrival + delay (rounded down to the output grid), or None if
    the trajectory never reaches S0 or that time is beyond its last record."""
    s0 = first_s0_time(t)
    if s0 is None or s0 + delay_fs > float(t.time_fs[-1]):
        return None
    step = int(np.floor((s0 + delay_fs) / dt_fs + 1e-9))
    return (step // output_every) * output_every


def hold_in_s0(active: np.ndarray, observed: np.ndarray) -> np.ndarray:
    held = np.array(active, copy=True)
    held[~np.asarray(observed, dtype=bool)] = 0
    return held


def max_bias_ratio(mean: np.ndarray, truth: np.ndarray, unit: np.ndarray) -> float:
    ok = unit > 0
    return float(np.max(np.abs(mean - truth)[ok] / unit[ok])) if np.any(ok) else 0.0


def poisson_rate_interval(k: int, exposure: float) -> tuple[float, float]:
    from scipy.stats import chi2

    lo = 0.0 if k == 0 else float(chi2.ppf(0.025, 2 * k) / 2 / exposure)
    hi = float(chi2.ppf(0.975, 2 * k + 2) / 2 / exposure)
    return lo, hi


# --------------------------------------------------------------------------- input
def load_project(run: Path) -> dict[str, Any]:
    config = json.loads((run / ".gqteashmd" / "config.json").read_text())  # ruling PR9
    dyn = config["dynamics"]
    dyn_dir = find_stage_dir(run, "dynamics")
    trajs: dict[str, Trajectory | None] = {}
    kinetic: dict[str, np.ndarray] = {}
    from gqteashmd.trajectory.store import read_output_h5

    for task_dir in sorted((dyn_dir / "tasks").glob("traj_*")):
        path = task_dir / "output.h5"
        trajs[task_dir.name] = read_our_trajectory(path, task_dir.name)
        if path.is_file():
            kinetic[task_dir.name] = np.asarray(read_output_h5(path)["e_kinetic"], dtype=float)
    first = next(t for t in trajs.values() if t is not None)
    return {
        "trajectories": trajs,
        "e_kinetic": kinetic,
        "dt_fs": float(dyn["dt_fs"]),
        "output_every": int(dyn.get("output_every", 1)),
        "n_total": round(float(dyn["tmax_fs"]) / float(dyn["dt_fs"])),
        "n_states": int(first.populations.shape[1]),
        "seed": int(config["project"]["seed"]),
        "n_bootstrap": int(config.get("analysis", {}).get("n_bootstrap", 1000)),
    }


# -------------------------------------------------------------- the four estimators
def four_estimates(
    complete: Sequence[Trajectory], steps: dict[str, int], p: dict[str, Any]
) -> tuple[dict[str, np.ndarray], grid.GridData]:
    h = grid.horizon_steps(complete, p["n_total"])
    g = grid.on_grid(
        complete,
        output_every=p["output_every"],
        dt_fs=p["dt_fs"],
        n_total=p["n_total"],
        n_states=p["n_states"],
        censored_at={k: v for k, v in steps.items() if v < h},
        horizon=h,
    )
    s = p["n_states"]
    never = g.observed[:, -1]
    out = {
        "censored": censoring.aalen_johansen(g.active, g.observed, s).populations,
        "still_running": censoring.at_risk_average(np.eye(s)[g.active], g.observed),
        "complete_only": populations.classical_populations(g.active[never], s)[0]
        if never.any()
        else np.full((g.active.shape[1], s), np.nan),
        "hold_in_s0": populations.classical_populations(hold_in_s0(g.active, g.observed), s)[0],
    }
    return out, g


def fit_excited(t: np.ndarray, p: np.ndarray, dt_out: float) -> tuple[float | None, float | None]:
    r = fit_delayed_exponential(t, 1.0 - p[:, 0], dt_out=dt_out)
    return r.tau_fs, r.t0_fs


def artificial(p: dict[str, Any], *, repetitions: int, bootstrap: int, seed: int,
               hot: bool) -> dict[str, Any]:
    trajs = [t for t in p["trajectories"].values() if t is not None]
    complete = [t for t in trajs if t.outcome == "finished"]
    delays = loss_delays_fs(trajs)
    n_lost, n_reach = reaching_s0_fraction(trajs)
    fraction = n_lost / n_reach
    dt_out = p["output_every"] * p["dt_fs"]
    truth_est, g0 = four_estimates(complete, {}, p)
    truth = truth_est["censored"]  # no censoring: the classical fractions of the complete set
    tau_true, t0_true = fit_excited(g0.time_fs, truth, dt_out)
    rng = np.random.default_rng(seed + (1 if hot else 0))
    reach = [t for t in complete if first_s0_time(t) is not None]
    if hot:  # spec 4.2 item 6: the fraction with the highest mean kinetic energy in S0
        def mean_k_s0(t: Trajectory) -> float:
            k = p["e_kinetic"][t.task]
            return float(k[t.active == 0].mean())
        ranked = sorted(reach, key=mean_k_s0, reverse=True)
        hot_set = {t.task for t in ranked[: round(fraction * len(reach))]}
    curves: dict[str, list[np.ndarray]] = {k: [] for k in truth_est}
    fits: dict[str, list[tuple[float | None, float | None]]] = {k: [] for k in truth_est}
    units = []
    for _ in range(repetitions):
        steps: dict[str, int] = {}
        for t in reach:
            chosen = (t.task in hot_set) if hot else (rng.random() < fraction)
            if not chosen:
                continue
            step = censor_step_for(t, float(rng.choice(delays)), p["dt_fs"], p["output_every"])
            if step is not None:
                steps[t.task] = step
        est, g = four_estimates(complete, steps, p)
        for k, v in est.items():
            curves[k].append(v)
            fits[k].append(fit_excited(g.time_fs, v, dt_out))
        idx = populations.bootstrap_indices(rng, len(complete), bootstrap)
        boot = censoring.bootstrap_aalen_johansen(g.active, g.observed, p["n_states"], idx)
        units.append(boot.std(axis=0, ddof=1))
    unit = np.mean(units, axis=0)
    result: dict[str, Any] = {
        "n_complete": len(complete),
        "loss_fraction": [n_lost, n_reach],
        "delays_fs": delays,
        "repetitions": repetitions,
        "truth": {"tau_fs": tau_true, "t0_fs": t0_true},
        "estimators": {},
    }
    for k in truth_est:
        mean = np.nanmean(np.asarray(curves[k]), axis=0)
        taus = [f[0] for f in fits[k] if f[0] is not None]
        t0s = [f[1] for f in fits[k] if f[1] is not None]
        result["estimators"][k] = {
            "max_bias_over_se": max_bias_ratio(mean, truth, unit),
            "max_abs_bias": float(np.nanmax(np.abs(mean - truth))),
            "tau_bias_fs": (float(np.mean(taus)) - tau_true) if taus and tau_true else None,
            "t0_bias_fs": (float(np.mean(t0s)) - t0_true) if t0s and t0_true is not None
            else None,
            "fits_failed": sum(1 for f in fits[k] if f[0] is None),
        }
    if not hot:  # the verdict fixed in spec 4.2 item 5
        r = {k: v["max_bias_over_se"] for k, v in result["estimators"].items()}
        result["verdict_good"] = bool(
            r["censored"] <= 1.0
            and r["censored"] < r["complete_only"]
            and r["censored"] < r["still_running"]
        )
    return result


def back_hop_rates(p: dict[str, Any]) -> dict[str, Any]:
    """Hops S0 -> excited per fs of S0 time, lost (before the loss) and complete (spec 4.2 6)."""
    out = {}
    for name, keep in (("lost", lambda t: t.outcome not in INCLUDED_OUTCOMES),
                       ("complete", lambda t: t.outcome == "finished")):
        hops = 0
        exposure = 0.0
        for t in p["trajectories"].values():
            if t is None or t.time_fs.size < 2 or not keep(t) or first_s0_time(t) is None:
                continue
            dt = np.diff(t.time_fs)
            in_s0 = t.active[:-1] == 0
            exposure += float(dt[in_s0].sum())
            hops += int(np.sum((t.hop_from[1:] == 0) & (t.hop_to[1:] > 0)))
        lo, hi = poisson_rate_interval(hops, exposure) if exposure > 0 else (None, None)
        out[name] = {"hops": hops, "s0_time_fs": exposure,
                     "rate_per_fs": hops / exposure if exposure > 0 else None,
                     "interval_95": [lo, hi]}
    return out


def reanalysis(p: dict[str, Any]) -> dict[str, Any]:
    trajs = p["trajectories"]
    tasks = list(trajs)
    complete, _ = grid.select(tasks, trajs, {}, energy_jump_eV=0.1, exclude_energy_jumps=False)
    used, steps, cens, excl = censoring.select_censored(
        tasks, trajs, output_every=p["output_every"], n_total=p["n_total"], dt_fs=p["dt_fs"],
        energy_jump_eV=None,
    )
    h = grid.horizon_steps(complete, p["n_total"])
    steps = {k: v for k, v in steps.items() if v < h}
    kw = {"output_every": p["output_every"], "dt_fs": p["dt_fs"], "n_total": p["n_total"],
          "n_states": p["n_states"], "horizon": h}
    g_c = grid.on_grid(complete, **kw)
    g_u = grid.on_grid(used, censored_at=steps, **kw)
    dt_out = p["output_every"] * p["dt_fs"]
    fit_index = list(range(1, p["n_states"]))
    ests = {
        "complete": estimators.complete_estimate(
            g_c, fit_index, fit_population="classical", rng=task_rng(p["seed"], "analysis", 0),
            n_bootstrap=p["n_bootstrap"], dt_out=dt_out),
        "censored": estimators.censored_estimate(
            g_u, fit_index, rng=task_rng(p["seed"], "analysis", 1),
            n_bootstrap=p["n_bootstrap"], dt_out=dt_out),
    }
    out: dict[str, Any] = {"n_complete": len(complete), "n_censored": len(steps),
                           "n_excluded": len(excl), "horizon_fs": h * p["dt_fs"]}
    for name, e in ests.items():
        g = g_u if name == "censored" else g_c
        rows = {}
        for when in REPORT_TIMES_FS:
            k = int(np.argmin(np.abs(g.time_fs - when)))
            rows[f"{g.time_fs[k]:.2f}"] = {
                "P": e.classical[k].tolist(),
                "lower": e.classical_lower[k].tolist(),
                "upper": e.classical_upper[k].tolist(),
            }
        f = e.fit
        out[name] = {
            "populations": rows,
            "tau_fs": f.fit.tau_fs, "tau_interval_fs": f.tau_interval_fs,
            "t0_fs": f.fit.t0_fs, "t0_interval_fs": f.t0_interval_fs,
            "empty_rows": e.empty_rows,
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artificial", type=Path, required=True)
    ap.add_argument("--project", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repetitions", type=int, default=200)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261119)
    args = ap.parse_args(argv)
    base = load_project(args.artificial)
    report = {
        "artificial": artificial(base, repetitions=args.repetitions, bootstrap=args.bootstrap,
                                 seed=args.seed, hot=False),
        "sensitivity": artificial(base, repetitions=args.repetitions, bootstrap=args.bootstrap,
                                  seed=args.seed, hot=True),
        "back_hop_rates": back_hop_rates(base),
        "reanalysis": {str(r): reanalysis(load_project(r))
                       for r in [args.artificial, *args.project]},
    }
    args.out.write_text(json.dumps(jsonable(report), indent=2) + "\n")
    print(json.dumps({"verdict_good": report["artificial"]["verdict_good"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run `ruff format` on the file: the long expressions above are written compactly, and ruff decides the layout. Two checks:
- If `jsonable` is not exported by `forensics`, use the one `hop_energy_report.py` imports.
- `four_estimates` treats only `finished` trajectories as complete. In P9b none is `stopped_in_s0` (`stop_after_s0_fs` is off), which is as the spec asks; say so in a comment.

- [ ] **Step 4: Run the helper tests**

Run: `pytest tests/validation/test_censoring_report.py -q -W error`
Expected: PASS.

- [ ] **Step 5: A smoke run on the canned pipeline is not possible** (the canned project has no losses after S0). Instead, run `python tools/validation/censoring_report.py --help` and confirm exit 0. The real run is Task 7.

- [ ] **Step 6: Lint and commit**

```bash
ruff format . && ruff check .
git add tools/validation/censoring_report.py tests/validation/test_censoring_report.py
git commit -m "test: the censored analysis report script"
```

---

### Task 7: The measurements (controller; no code change)

- [ ] **Step 1:** In WSL, `mkdir -p ~/censoring`, then run, detached (`setsid nohup`, an EXIT file, the controller's waiter):

```bash
python tools/validation/censoring_report.py --artificial ~/p9b/ours/run \
  --project ~/hop/ours_previous/run --project ~/p9a/acceptance/run \
  --out ~/censoring/report.json > ~/censoring/run.log 2>&1
```

- [ ] **Step 2:** Record the following in `.superpowers/sdd/2026-11-19-censored-analysis/task-7-report.md`:
  - the printed verdict;
  - every number of the four blocks;
  - the run time.

  Nothing is re-tuned. If `verdict_good` is false, that is the result.

---

### Task 8: Documents

**Files:** `USER_MANUAL.md`, `CLAUDE.md`, `docs/superpowers/reports/2026-11-19-censored-analysis.md` (new), the spec's erratum.

- [ ] **Step 1: `USER_MANUAL.md`**
  - **Section 4, the `[analysis]` table:** the `estimator` key (default `"censored"`), and the quantum-fit refusal.
  - **Section 6:** the new `analysis.json` keys (`schema_version` 2, PR4) and the extra file columns (PR6).
  - **8.13, the analysis method:**
    - the censoring rule (spec 2.1);
    - the estimator and its assumption (spec 1.3, with the four references);
    - the quantum rule;
    - which statistics stay over the complete trajectories (PR4).
  - **8.12 and 8.18:** under `"previous"` the energy-jump signal sees `e_total_before_hop` (PR10); the drift check is unchanged.
  - **The rerun advice:** `AnalysisStage.version` 3 (`--rerun-from analysis`), and `DynamicsStage.version` 3 (**every** project with `dynamics` results: `--rerun-from dynamics`).
  - **A new 8.19:** every number of Tasks 2, 5 and 7, each marked asserted or report.
  - **Section 10, the limitations:** the Markov assumption; hops and drift over the complete trajectories only; the quantum curve's bias; the small-ensemble coverage measured for complete ensembles only.
  - **Section 11.**
- [ ] **Step 2: `CLAUDE.md`**
  - **"Read first":** a bullet for this spec, plan, ledger and report.
  - **A module block:** `analysis/censoring.py`, `analysis/estimators.py`, the stage changes, `StepRecord.e_total_before_hop`, both version bumps, the seed streams 0 and 1 (the Seeds rule).
  - **Status:** add an entry for this sub-project.
  - **Known limitations 1a, 1b and 1c:** the excluded-fraction bias is now addressed by the default estimator, and the I1 gap is closed for the energy-jump signal.
  - **Remaining work.**
  - **The test counts.**
- [ ] **Step 3: The report** `docs/superpowers/reports/2026-11-19-censored-analysis.md`: the measured numbers of Tasks 2, 5 and 7, each marked asserted or report; the plan rulings PR1-PR10; and an honest summary.
- [ ] **Step 4:** Add the spec's erratum at the top: rulings PR1, PR2, PR4-PR6 and PR10 change details of the spec, and say what was decided after numbers were seen (if anything).
- [ ] **Step 5:** Commit: `docs: the censored analysis report and documents`.
