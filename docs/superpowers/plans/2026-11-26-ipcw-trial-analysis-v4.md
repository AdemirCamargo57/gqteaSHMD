# The IPCW trial and analysis stage version 4: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, with rules fixed before the run, whether an IPCW estimator with the S0 kinetic energy as covariate corrects the analysis where the censored estimator fails; and, whatever the result, make the `analysis` stage cap its horizon without complete trajectories and report the complete, censored and hold-in-S0 lifetimes as a range (`AnalysisStage.version` 4).

**Architecture:** The trial is tools only: a new `tools/validation/ipcw.py` (discrete-time logistic censoring hazard with a nonparametric baseline, IPCW weights and populations) used by `tools/validation/censoring_report.py`, which gains a fifth estimator, a control test, a real-data gate and the go decision. Version 4 changes `stages/analysis.py`, `analysis/estimators.py` and `analysis/report.py`; the complete and censored numbers do not change.

**Tech Stack:** Python 3.12, NumPy, SciPy (`scipy.optimize.brentq`, `minimize_scalar`, `scipy.special.expit`), h5py, pytest; everything runs in WSL Oracle Linux 9.5.

**Spec:** `docs/superpowers/specs/2026-11-26-ipcw-trial-analysis-v4-design.md` (owner-approved 2026-11-26). Read it before any task; the plan rulings below are binding where they differ.

## Global Constraints

- Run every command in WSL: `wsl -d OracleLinux_9_5 -- bash -lc 'cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && conda activate gqteashmd-dev && <command>'`. Below, `W '<command>'` stands for that wrapper.
- Git from WSL only (pre-commit hooks run there); never `--no-verify`. One commit per task, conventional subjects; end each message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Files with LF line endings. Lint: `ruff format . && ruff check .` (line length 100); `mypy` clean on `src/`.
- `tools/` is never imported by `gqteashmd`; `import gqteashmd.cli` must not import scipy or h5py (SciPy only inside functions in `src/`).
- Tests: `pytest ... -W error`. A new test must not compare BLAS/LAPACK results bit for bit or rely on an eigenvector's sign; check new tests once with `OPENBLAS_CORETYPE=SandyBridge`.
- Seeds (D12): the stage's streams are `task_rng(seed, "analysis", 0)` (complete, unchanged), `1` (censored, unchanged), `2` (hold-in-S0, new). The report script's streams: `seed` (artificial), `seed + 1` (sensitivity), `seed + 2` (control), `seed + 3` (gate), `seed + 4` (IPCW reanalysis bootstrap); default `seed = 20261119`.
- Fixed numbers (spec 2.5): control `beta_true = 1.0` per SD, rule `max_bias_over_se <= 1`; hot-S0 rule strictly below censored and hold-in-S0; gate B = 1000, at least 950 successes, 2.5 % bound of β > 0; β bounds [-10, 10] per SD; R = 200, B = 200 (script defaults).
- Not in scope (spec 6): IPCW in the stage, other covariates, other molecules, rerunning dynamics, weight truncation or normalisation, changing the default `estimator`.
- The ledger is `.superpowers/sdd/2026-11-26-ipcw-trial-analysis-v4/progress.md` (git-ignored); every task writes `task-N-report.md` there with the commands run and their counts.

## Plan rulings (binding where they differ from the spec)

- **PR1, version numbers.** `AnalysisStage.version` 4 and `analysis.json` `schema_version` 3 are set in Task 4 (the first behaviour change), not Task 5. Both tasks are on one branch with no release between.
- **PR2, the IPCW reanalysis stream.** The spec gives B (the project's `n_bootstrap`) but no stream for the IPCW bootstrap of section 2.6: it is `np.random.default_rng(seed + 4)`.
- **PR3, an uninformative fit.** A censoring fit with no S0 interval that has both a censored and an uncensored trajectory at risk (every S0 event cell has `d = r`) cannot identify β: it is `fit_failed` with reason `no informative S0 interval` (spec 2.3 lists two failure reasons; this is a third).
- **PR4, the control's denominator.** "S0-reaching" in the control's lost fraction uses `first_s0_time(t) is not None` on all records, the definition under which 33/90 was measured (`reaching_s0_fraction`). A trajectory reaching S0 only at its last grid record has no S0 interval and cannot be censored; it still counts in the denominator.
- **PR5, test 4.1 item 8.** "A small canned project" is an in-memory synthetic project (`tests/validation/censoring_synthetic.py`). The script's numbers before the change are written by the unmodified script to `tests/validation/data/censoring_report_before.json` (Task 2 Step 1) and compared to a relative 1e-9 (bias ratios, curves) and 1e-6 (tau and t0), not bit for bit (BLAS rule).
- **PR6, the hold-in-S0 helper.** `estimators.complete_estimate` gains `name: str = "complete"`; `estimators.hold_in_s0_grid(g)` builds the held grid; `estimators.hold_in_s0_estimate(...)` calls both.
- **PR7, the bracket of the intercept solve.** For an S0 cell with `0 < d < r` and β ≠ 0 the intercept solves `Σ expit(α + β z_i) = d` with `brentq` on `[logit(d/r) - s, logit(d/r) + s]`, `s = max|β z_i| + 1`, which brackets the root by monotonicity.

## File structure

| File | Task | Responsibility |
|---|---|---|
| `tools/validation/ipcw.py` (new) | 1 | covariate on the grid, running S0 mean, censoring hazard fit, weights, populations |
| `tests/validation/test_ipcw.py` (new) | 1 | spec 4.1 items 1-7 |
| `tests/validation/censoring_synthetic.py` (new, not collected) | 2 | in-memory synthetic project for the report script tests |
| `tests/validation/data/censoring_report_before.json` (new) | 2 | the unmodified script's numbers (PR5) |
| `tools/validation/censoring_report.py` | 2 | fifth estimator, `control`, `gate`, IPCW reanalysis, `ipcw_go` |
| `tests/validation/test_censoring_report.py` | 2 | spec 4.1 item 8 and the new functions |
| `src/gqteashmd/stages/analysis.py` | 4, 5 | horizon cap, version, hold-in-S0 block, `tau_range_fs`, files |
| `src/gqteashmd/analysis/estimators.py` | 5 | `name` argument, `hold_in_s0_grid`, `hold_in_s0_estimate` |
| `src/gqteashmd/analysis/report.py` | 5 | `## Lifetime range` section |
| `tests/stages/test_analysis.py`, `tests/analysis/test_estimators.py`, `tests/analysis/test_report.py` | 4, 5 | spec 4.2 |
| documents | 6 | report, USER_MANUAL, CLAUDE.md, CHANGELOG |

---

### Task 1: `tools/validation/ipcw.py`

**Files:**
- Create: `tools/validation/ipcw.py`
- Test: `tests/validation/test_ipcw.py`

**Interfaces:**
- Consumes: `gqteashmd.analysis.grid.Trajectory` (fields `task`, `grid_step`, `ticks`, `active`); `gqteashmd.analysis.censoring.at_risk_average`; `gqteashmd.core.units.HARTREE_IN_EV`.
- Produces (used by Task 2):
  - `kinetic_on_grid(trajectories: Sequence[Trajectory], kinetic: Mapping[str, np.ndarray], grid_steps: np.ndarray) -> np.ndarray` (n, G), NaN where a trajectory has no full-step record at that grid step.
  - `running_s0_mean(active: np.ndarray, kinetic: np.ndarray, observed: np.ndarray) -> np.ndarray` (n, G), NaN before the first observed S0 record.
  - `risk_and_events(observed) -> tuple[np.ndarray, np.ndarray]` (n, G-1) bool each.
  - `CensoringFit` (frozen dataclass: `status: str`, `reason: str | None`, `beta: float | None`, `sd_eV: float | None`, `hazard: np.ndarray | None` (n, G-1), `n_events_s0: int`, `n_informative: int`; properties `ok: bool`, `beta_per_eV: float | None`; method `summary() -> dict`).
  - `fit_censoring_hazard(active, observed, kinetic, n_states, *, covariate: bool = True) -> CensoringFit`.
  - `censoring_weights(hazard: np.ndarray, observed: np.ndarray) -> np.ndarray` (n, G), 0 where not observed.
  - `ipcw_populations(active, weights, n_states) -> np.ndarray` (G, S), `(1/n) Σ_i w_i 1{active_i = s}`.
  - `ipcw_estimate(active, observed, kinetic, n_states, *, covariate: bool = True) -> tuple[np.ndarray | None, CensoringFit]` (populations `None` when the fit failed).
  - Constants `BETA_BOUND = 10.0`, `BOUND_TOLERANCE = 1e-3`.

- [ ] **Step 1: Write the failing tests** (`tests/validation/test_ipcw.py`)

```python
"""tools/validation/ipcw.py (spec 2026-11-26-ipcw-trial-analysis-v4-design.md, 4.1 items 1-7)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "validation"))
import ipcw  # noqa: E402

from gqteashmd.analysis import censoring  # noqa: E402
from gqteashmd.analysis.grid import Trajectory  # noqa: E402
from gqteashmd.core.units import HARTREE_IN_EV  # noqa: E402

T_MARKOV = np.array([[0.97, 0.02, 0.01], [0.05, 0.90, 0.05], [0.0, 0.08, 0.92]])


def markov_sample(rng, n=300, g=60, hazard=(0.06, 0.01, 0.01)):
    """Spec 1.1: start S2, censoring hazard by the current state."""
    act = np.zeros((n, g), dtype=np.int64)
    act[:, 0] = 2
    for k in range(1, g):
        cum = np.cumsum(T_MARKOV[act[:, k - 1]], axis=1)
        act[:, k] = (rng.random(n)[:, None] > cum).sum(axis=1)
    obs = np.ones((n, g), dtype=bool)
    hz = np.asarray(hazard)
    for k in range(g - 1):
        cut = obs[:, k] & (rng.random(n) < hz[act[:, k]])
        obs[cut, k + 1 :] = False
    return act, obs


def per_cell_hazard(act, obs, n_states, *, by_state=True):
    risk, event = ipcw.risk_and_events(obs)
    h = np.zeros(risk.shape)
    for k in range(risk.shape[1]):
        groups = range(n_states) if by_state else [None]
        for s in groups:
            m = risk[:, k] & ((act[:, k] == s) if s is not None else True)
            if m.any():
                h[m, k] = event[m, k].sum() / m.sum()
    return h


def test_marginal_weights_give_the_still_running_average():  # item 1
    act, obs = markov_sample(np.random.default_rng(1))
    w = ipcw.censoring_weights(per_cell_hazard(act, obs, 3, by_state=False), obs)
    np.testing.assert_allclose(
        ipcw.ipcw_populations(act, w, 3),
        censoring.at_risk_average(np.eye(3)[act], obs),
        rtol=0,
        atol=1e-12,
    )


def test_without_the_covariate_the_hazard_is_d_over_r_per_cell():  # item 2
    act, obs = markov_sample(np.random.default_rng(2), n=200, g=40)
    fit = ipcw.fit_censoring_hazard(act, obs, np.zeros(act.shape), 3, covariate=False)
    assert fit.ok and fit.beta == 0.0
    expected = per_cell_hazard(act, obs, 3)
    np.testing.assert_allclose(fit.hazard, expected, rtol=0, atol=1e-12)
    w = np.ones(act.shape)
    with np.errstate(divide="ignore", invalid="ignore"):  # h = 1 where every one is censored
        for k in range(act.shape[1] - 1):
            w[:, k + 1] = w[:, k] / (1.0 - expected[:, k])
    w[~obs] = 0.0
    hand = np.stack([((act == s) * w).sum(axis=0) for s in range(3)], axis=-1) / act.shape[0]
    curve, _ = ipcw.ipcw_estimate(act, obs, np.zeros(act.shape), 3, covariate=False)
    np.testing.assert_allclose(curve, hand, rtol=0, atol=1e-12)


def test_ipcw_is_unbiased_under_state_dependent_censoring():  # item 3, the PR1 form
    rng = np.random.default_rng(20261126)
    g = 61  # grid points 0..60, so the spec's points 10, 20, ..., 60 exist
    truth = np.zeros((g, 3))
    truth[0, 2] = 1.0
    for k in range(1, g):
        truth[k] = truth[k - 1] @ T_MARKOV
    curves = []
    for _ in range(100):
        act, obs = markov_sample(rng, g=g)
        curve, fit = ipcw.ipcw_estimate(act, obs, np.zeros(act.shape), 3, covariate=False)
        assert fit.ok
        curves.append(curve)
    c = np.asarray(curves)
    se = c.std(axis=0, ddof=1) / np.sqrt(len(curves))
    for k in (10, 20, 30, 40, 50, 60):
        for s in range(3):
            if se[k, s] > 0:
                assert abs(c[:, k, s].mean() - truth[k, s]) <= 4 * se[k, s], (k, s)


def hot_censoring_sample(rng, n=400, g=40, a=-3.0, beta=1.0):
    """Everyone in S0; a constant kinetic level per trajectory; logit h = a + beta z."""
    act = np.zeros((n, g), dtype=np.int64)
    level = rng.normal(0.01, 0.002, n)
    kin = np.repeat(level[:, None], g, axis=1)
    z = (level - level.mean()) / level.std()
    h = 1.0 / (1.0 + np.exp(-(a + beta * z)))
    obs = np.ones((n, g), dtype=bool)
    for k in range(g - 1):
        cut = obs[:, k] & (rng.random(n) < h)
        obs[cut, k + 1 :] = False
    return act, obs, kin


def test_the_fit_recovers_a_known_beta():  # item 4
    # compared per eV: the fit standardises over the S0 cells at risk (hot trajectories leave
    # early), the simulation over the trajectories, so the per-SD values differ by design
    rng = np.random.default_rng(7)
    act, obs, kin = hot_censoring_sample(rng)
    true_per_eV = 1.0 / (np.std(kin[:, 0]) * HARTREE_IN_EV)
    fit = ipcw.fit_censoring_hazard(act, obs, kin, 1)
    assert fit.ok and fit.sd_eV > 0
    assert fit.beta_per_eV == pytest.approx(fit.beta / fit.sd_eV)
    boots = []
    for _ in range(40):
        r = rng.integers(0, act.shape[0], act.shape[0])
        f = ipcw.fit_censoring_hazard(act[r], obs[r], kin[r], 1)
        assert f.ok
        boots.append(f.beta_per_eV)
    assert abs(fit.beta_per_eV - true_per_eV) <= 4 * np.std(boots, ddof=1)


def test_without_censoring_the_populations_are_the_plain_fractions():  # item 5
    act, _ = markov_sample(np.random.default_rng(3), n=50, g=20)
    obs = np.ones(act.shape, dtype=bool)
    curve, fit = ipcw.ipcw_estimate(act, obs, np.zeros(act.shape), 3, covariate=False)
    assert fit.ok and not fit.hazard.any()
    plain = np.stack([(act == s).mean(axis=0) for s in range(3)], axis=-1)
    np.testing.assert_array_equal(curve, plain)


def test_running_s0_mean():  # item 6
    active = np.array([[1, 0, 0, 1, 0]])
    kinetic = np.array([[5.0, 1.0, 3.0, 7.0, 2.0]])
    observed = np.ones((1, 5), dtype=bool)
    out = ipcw.running_s0_mean(active, kinetic, observed)
    assert np.isnan(out[0, 0])
    np.testing.assert_allclose(out[0, 1:], [1.0, 2.0, 2.0, 2.0])


def test_fit_failures():  # item 7
    act = np.zeros((4, 5), dtype=np.int64)
    obs = np.ones((4, 5), dtype=bool)
    fit = ipcw.fit_censoring_hazard(act, obs, np.ones((4, 5)), 1)
    assert not fit.ok and fit.reason == "no censoring event in S0"
    # complete separation: the two hot trajectories are lost at step 1, the cold ones never
    kin = np.repeat(np.array([[2.0], [2.0], [1.0], [1.0]]), 5, axis=1)
    obs2 = obs.copy()
    obs2[:2, 2:] = False
    fit = ipcw.fit_censoring_hazard(act, obs2, kin, 1)
    assert not fit.ok and fit.reason == "beta on a bound"
    # every S0 event cell has d = r: no informative interval (plan ruling PR3)
    obs3 = obs.copy()
    obs3[:, 2:] = False
    fit = ipcw.fit_censoring_hazard(act, obs3, kin, 1)
    assert not fit.ok and fit.reason == "no informative S0 interval"


def test_kinetic_on_grid_reads_full_step_records():
    t = Trajectory(
        "a",
        "finished",
        1,
        np.array([0, 1, 1, 2]),
        np.array([0, 0, 1, 0]),
        np.array([0.0, 0.5, 0.75, 1.0]),
        np.array([1, 0, 0, 0]),
        np.full(4, -1),
        np.full(4, -1),
        np.zeros(4, dtype=bool),
        np.zeros(4),
        np.eye(2)[[1, 0, 0, 0]],
    )
    out = ipcw.kinetic_on_grid([t], {"a": np.array([1.0, 2.0, 9.0, 3.0])}, np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(out[0, :3], [1.0, 2.0, 3.0])
    assert np.isnan(out[0, 3])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `W 'pytest tests/validation/test_ipcw.py -q -W error'`
Expected: collection error, `ModuleNotFoundError: No module named 'ipcw'`.

- [ ] **Step 3: Write `tools/validation/ipcw.py`**

```python
"""IPCW estimator of state occupation (spec 2026-11-26-ipcw-trial-analysis-v4-design.md, 2).

Report tool, never imported by gqteashmd. The censoring hazard is a discrete-time logistic
model, logit h_i(k) = alpha_{a,k} + beta z_i(k) 1{a = 0}, with one intercept per grid step and
state (nonparametric baseline) and z the standardised running mean of the S0 kinetic energy.
The populations are (1/n) sum_i w_i(k) 1{active_i(k) = s}, w_i the inverse probability of
remaining uncensored (not normalised).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from gqteashmd.analysis.grid import Trajectory
from gqteashmd.core.units import HARTREE_IN_EV

BETA_BOUND = 10.0
BOUND_TOLERANCE = 1e-3


@dataclass(frozen=True)
class CensoringFit:
    status: str  # "ok" or "fit_failed"
    reason: str | None
    beta: float | None  # per SD of the covariate; 0.0 without it
    sd_eV: float | None  # SD of the running S0 kinetic energy over the S0 cells at risk
    hazard: np.ndarray | None  # (n, G-1) fitted per-interval censoring probability
    n_events_s0: int
    n_informative: int  # S0 cells with 0 < d < r

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def beta_per_eV(self) -> float | None:
        if self.beta is None or not self.sd_eV:
            return None
        return self.beta / self.sd_eV

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "beta_per_sd": self.beta,
            "sd_eV": self.sd_eV,
            "beta_per_eV": self.beta_per_eV,
            "n_events_s0": self.n_events_s0,
            "n_informative": self.n_informative,
        }


def _failed(reason: str, n_events_s0: int, n_informative: int = 0) -> CensoringFit:
    return CensoringFit("fit_failed", reason, None, None, None, n_events_s0, n_informative)


def kinetic_on_grid(
    trajectories: Sequence[Trajectory],
    kinetic: Mapping[str, np.ndarray],
    grid_steps: np.ndarray,
) -> np.ndarray:
    """(n, G): e_kinetic at each trajectory's full-step record of every grid step (NaN if none)."""
    out = np.full((len(trajectories), len(grid_steps)), np.nan)
    for i, t in enumerate(trajectories):
        values = np.asarray(kinetic[t.task], dtype=float)
        where = {int(t.grid_step[j]): int(j) for j in np.flatnonzero(t.ticks == 0)}
        for m, s in enumerate(grid_steps):
            j = where.get(int(s))
            if j is not None:
                out[i, m] = values[j]
    return out


def running_s0_mean(active: np.ndarray, kinetic: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """(n, G): the mean kinetic energy over the observed S0 records up to each grid step."""
    in_s0 = (np.asarray(active) == 0) & np.asarray(observed, dtype=bool)
    total = np.cumsum(np.where(in_s0, np.asarray(kinetic, dtype=float), 0.0), axis=1)
    count = np.cumsum(in_s0, axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def risk_and_events(observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, G-1) each: at risk over [k, k+1] (observed at k); censored there (not at k+1)."""
    obs = np.asarray(observed, dtype=bool)
    risk = obs[:, :-1]
    return risk, risk & ~obs[:, 1:]


def _event_cells(
    active: np.ndarray, risk: np.ndarray, event: np.ndarray, n_states: int
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """(k, state, members, censored) of every cell with at least one event."""
    a = np.asarray(active)[:, :-1]
    cells = []
    for k in range(risk.shape[1]):
        for s in range(n_states):
            members = np.flatnonzero(risk[:, k] & (a[:, k] == s))
            if members.size and event[members, k].any():
                cells.append((k, s, members, event[members, k]))
    return cells


def _hazard(
    beta: float,
    z: np.ndarray | None,
    cells: Sequence[tuple[int, int, np.ndarray, np.ndarray]],
    shape: tuple[int, int],
) -> tuple[np.ndarray, float]:
    """The profile hazard for one beta and its log-likelihood (cells without events: h = 0)."""
    from scipy.optimize import brentq
    from scipy.special import expit

    h = np.zeros(shape)
    loglik = 0.0
    for k, s, members, censored in cells:
        r, d = members.size, int(censored.sum())
        if d == r:
            h[members, k] = 1.0
            continue
        if s != 0 or beta == 0.0 or z is None:
            hc = np.full(r, d / r)
        else:
            x = beta * z[members, k]
            base = math.log(d / (r - d))
            spread = float(np.max(np.abs(x))) + 1.0  # plan ruling PR7
            alpha = brentq(
                lambda al: float(expit(al + x).sum()) - d,
                base - spread,
                base + spread,
                xtol=1e-14,
            )
            hc = expit(alpha + x)
        h[members, k] = hc
        with np.errstate(divide="ignore"):  # expit can round to 1.0 near the beta bounds
            loglik += float(np.log(hc[censored]).sum() + np.log1p(-hc[~censored]).sum())
    return h, loglik


def fit_censoring_hazard(
    active: np.ndarray,
    observed: np.ndarray,
    kinetic: np.ndarray,
    n_states: int,
    *,
    covariate: bool = True,
) -> CensoringFit:
    from scipy.optimize import minimize_scalar

    active = np.asarray(active, dtype=np.int64)
    risk, event = risk_and_events(observed)
    in_s0 = risk & (active[:, :-1] == 0)
    n_events_s0 = int((event & in_s0).sum())
    cells = _event_cells(active, risk, event, n_states)
    informative = sum(1 for _, s, m, c in cells if s == 0 and 0 < int(c.sum()) < m.size)
    if not covariate:
        h, _ = _hazard(0.0, None, cells, risk.shape)
        return CensoringFit("ok", None, 0.0, None, h, n_events_s0, informative)
    if n_events_s0 == 0:
        return _failed("no censoring event in S0", n_events_s0)
    if informative == 0:
        return _failed("no informative S0 interval", n_events_s0)  # plan ruling PR3
    kbar = running_s0_mean(active, kinetic, observed)[:, :-1]
    values = kbar[in_s0]
    sd = float(values.std()) if values.size else 0.0
    if not np.all(np.isfinite(values)) or sd <= 0.0:
        return _failed("the covariate has no spread", n_events_s0, informative)
    z = np.where(in_s0, (kbar - float(values.mean())) / sd, 0.0)
    result = minimize_scalar(
        lambda b: -_hazard(float(b), z, cells, risk.shape)[1],
        bounds=(-BETA_BOUND, BETA_BOUND),
        method="bounded",
        options={"xatol": 1e-8},
    )
    beta = float(result.x)
    if not result.success or abs(beta) > BETA_BOUND - BOUND_TOLERANCE:
        return _failed("beta on a bound", n_events_s0, informative)
    h, _ = _hazard(beta, z, cells, risk.shape)
    return CensoringFit("ok", None, beta, sd * HARTREE_IN_EV, h, n_events_s0, informative)


def censoring_weights(hazard: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """(n, G): w(0) = 1, w(k+1) = w(k) / (1 - h(k)); 0 where a trajectory is not observed."""
    obs = np.asarray(observed, dtype=bool)
    w = np.ones(obs.shape)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        w[:, 1:] = np.cumprod(1.0 / (1.0 - np.asarray(hazard, dtype=float)), axis=1)
    return np.where(obs, w, 0.0)


def ipcw_populations(active: np.ndarray, weights: np.ndarray, n_states: int) -> np.ndarray:
    """(G, S): (1/n) sum_i w_i(k) 1{active_i(k) = s}."""
    a = np.asarray(active, dtype=np.int64)
    onehot = np.eye(n_states)[a]
    return np.einsum("ng,ngs->gs", np.asarray(weights, dtype=float), onehot) / a.shape[0]


def ipcw_estimate(
    active: np.ndarray,
    observed: np.ndarray,
    kinetic: np.ndarray,
    n_states: int,
    *,
    covariate: bool = True,
) -> tuple[np.ndarray | None, CensoringFit]:
    fit = fit_censoring_hazard(active, observed, kinetic, n_states, covariate=covariate)
    if not fit.ok or fit.hazard is None:
        return None, fit
    weights = censoring_weights(fit.hazard, observed)
    return ipcw_populations(active, weights, n_states), fit
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `W 'pytest tests/validation/test_ipcw.py -q -W error'`
Expected: 8 passed. If `test_the_fit_recovers_a_known_beta` fails, check the profile likelihood (the sign of β, the bracket) before touching the test: the rule (4 bootstrap SE) is fixed by the spec.

Then once with another BLAS kernel: `W 'OPENBLAS_CORETYPE=SandyBridge pytest tests/validation/test_ipcw.py -q -W error'` (expected: 8 passed).

- [ ] **Step 5: Lint and commit**

```bash
W 'ruff format tools/validation/ipcw.py tests/validation/test_ipcw.py && ruff check tools/validation/ipcw.py tests/validation/test_ipcw.py'
W 'git add tools/validation/ipcw.py tests/validation/test_ipcw.py && git commit -F - <<EOF
feat: add the IPCW estimator of the censoring report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF'
```

---

### Task 2: the report script: fifth estimator, control, gate, go

**Files:**
- Create: `tests/validation/censoring_synthetic.py`, `tests/validation/data/censoring_report_before.json`
- Modify: `tools/validation/censoring_report.py` (functions `four_estimates`, `fit_excited`, `artificial`, `reanalysis`, `main`; new `estimates`, `_summarise`, `_censored_grid`, `control`, `gate`, `_ipcw_block`, `ipcw_go`)
- Test: `tests/validation/test_censoring_report.py`

**Interfaces:**
- Consumes (Task 1): `ipcw.kinetic_on_grid`, `ipcw.running_s0_mean`, `ipcw.ipcw_estimate`, `ipcw.fit_censoring_hazard`, `ipcw.censoring_weights`, `CensoringFit.ok/.beta/.summary()`.
- Produces: `ESTIMATORS = ("censored", "still_running", "complete_only", "hold_in_s0", "ipcw")`; `estimates(complete, steps, p) -> tuple[dict[str, np.ndarray], GridData, CensoringFit]`; `artificial(...)` result gains `estimators["ipcw"]`, `ipcw_fit = {"failures", "beta_mean"}` and, for `hot=True`, `ipcw_wins: bool`; `control(p, *, repetitions, bootstrap, seed) -> dict` with `passed: bool`; `gate(p, *, seed, resamples=1000) -> dict` with `passed: bool`; `reanalysis(p, *, seed) -> dict` gains `ipcw`; `ipcw_go(report: dict) -> bool`. The JSON written by `main` gains `control`, `gate`, `ipcw_go`.

- [ ] **Step 1: Write the synthetic project and record the unmodified script's numbers (PR5)**

`tests/validation/censoring_synthetic.py`:

```python
"""An in-memory project for tools/validation/censoring_report.py (plan ruling PR5).

Three states, start S2, decay S2 -> S1 -> S0 with back hops; the kinetic energy in S0 is a
per-trajectory level; every third trajectory that reaches S0 early is lost after its arrival.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gqteashmd.analysis.grid import Trajectory

TRANSITIONS = np.array([[0.93, 0.07, 0.0], [0.10, 0.90, 0.0], [0.0, 0.10, 0.90]])


def _trajectory(task: str, outcome: str, active: np.ndarray) -> Trajectory:
    n = active.size
    hop_from = np.full(n, -1)
    hop_to = np.full(n, -1)
    changed = np.flatnonzero(np.diff(active) != 0) + 1
    hop_from[changed] = active[changed - 1]
    hop_to[changed] = active[changed]
    return Trajectory(
        task,
        outcome,
        int(active[0]),
        np.arange(n, dtype=np.int64),
        np.zeros(n, dtype=np.int64),
        0.5 * np.arange(n),
        active.astype(np.int64),
        hop_from,
        hop_to,
        np.zeros(n, dtype=bool),
        np.zeros(n),
        np.eye(3)[active],
    )


def synthetic_project(seed: int = 11, n: int = 30, n_steps: int = 41) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    trajectories: dict[str, Trajectory | None] = {}
    kinetic: dict[str, np.ndarray] = {}
    for i in range(n):
        task = f"traj_{i:05d}"
        active = [2]
        for _ in range(n_steps - 1):
            active.append(int(rng.choice(3, p=TRANSITIONS[active[-1]])))
        a = np.asarray(active)
        heat = rng.uniform(0.005, 0.02)
        kin = np.where(a == 0, heat, 0.004) + rng.normal(0.0, 1e-4, n_steps)
        outcome, end = "finished", n_steps
        s0 = np.flatnonzero(a == 0)
        if i % 3 == 0 and s0.size and s0[0] < n_steps - 3:
            end = int(rng.integers(s0[0] + 1, n_steps - 1))
            outcome = "overlap_failure"
        trajectories[task] = _trajectory(task, outcome, a[:end])
        kinetic[task] = kin[:end]
    return {
        "trajectories": trajectories,
        "e_kinetic": kinetic,
        "dt_fs": 0.5,
        "output_every": 1,
        "n_total": n_steps - 1,
        "n_states": 3,
        "seed": 2028,
        "n_bootstrap": 50,
    }
```

Then, **with `censoring_report.py` still unmodified**, record its numbers:

```bash
W 'mkdir -p tests/validation/data && python - <<EOF
import json, sys
sys.path[:0] = ["tests/validation", "tools/validation"]
import censoring_report as cr
from censoring_synthetic import synthetic_project
from forensics import jsonable
p = synthetic_project()
out = {
    "artificial": cr.artificial(p, repetitions=4, bootstrap=20, seed=5, hot=False),
    "sensitivity": cr.artificial(p, repetitions=4, bootstrap=20, seed=5, hot=True),
    "reanalysis": cr.reanalysis(p),
}
open("tests/validation/data/censoring_report_before.json", "w").write(json.dumps(jsonable(out), indent=2) + "\n")
EOF'
```

Expected: the file exists; `git diff --stat tools/` is empty. Check that `artificial.loss_fraction` has a non-zero first entry (the synthetic project has losses). Also check, with Task 1's module, that the censoring fit succeeds on this project (`ipcw.ipcw_estimate` on the `reanalysis` grid gives a curve, and on two or three `artificial` repetitions). If it fails (for example by chance separation, `beta on a bound`), change the `synthetic_project` default seed and record the file again **now**, before Step 2; the fixture is a test input, not a pre-registered rule.

- [ ] **Step 2: Write the failing tests** (append to `tests/validation/test_censoring_report.py`)

```python
import json  # noqa: E402  (at the top of the file, with the other imports)

sys.path.insert(0, str(ROOT / "tests" / "validation"))
from censoring_synthetic import synthetic_project  # noqa: E402

BEFORE = json.loads((ROOT / "tests" / "validation" / "data" / "censoring_report_before.json").read_text())
OLD = ("censored", "still_running", "complete_only", "hold_in_s0")


def _close(a, b, rel):
    if a is None or b is None:
        assert a is None and b is None
    else:
        assert a == pytest.approx(b, rel=rel, abs=1e-12)


@pytest.mark.parametrize("hot,key", [(False, "artificial"), (True, "sensitivity")])
def test_the_four_estimators_are_unchanged_by_the_ipcw_addition(hot, key):  # spec 4.1 item 8
    new = cr.artificial(synthetic_project(), repetitions=4, bootstrap=20, seed=5, hot=hot)
    old = BEFORE[key]
    for name in OLD:
        a, b = new["estimators"][name], old["estimators"][name]
        _close(a["max_bias_over_se"], b["max_bias_over_se"], 1e-9)
        _close(a["max_abs_bias"], b["max_abs_bias"], 1e-9)
        _close(a["tau_bias_fs"], b["tau_bias_fs"], 1e-6)
        _close(a["t0_bias_fs"], b["t0_bias_fs"], 1e-6)
        assert a["fits_failed"] == b["fits_failed"]
    assert "ipcw" in new["estimators"] and new["ipcw_fit"]["failures"] == 0
    assert ("ipcw_wins" in new) is hot and ("verdict_good" in new) is (not hot)


def test_the_reanalysis_is_unchanged_and_gains_ipcw():
    new = cr.reanalysis(synthetic_project(), seed=5)
    for name in ("complete", "censored"):
        _close(new[name]["tau_fs"], BEFORE["reanalysis"][name]["tau_fs"], 1e-6)
        _close(new[name]["t0_fs"], BEFORE["reanalysis"][name]["t0_fs"], 1e-6)
    block = new["ipcw"]
    assert block["fit"]["status"] == "ok"
    assert set(block) >= {"populations", "tau_fs", "tau_interval_fs", "max_weight"}
    assert block["max_weight"] >= 1.0


def test_control_and_gate_report_their_rules():
    p = synthetic_project()
    c = cr.control(p, repetitions=4, bootstrap=20, seed=5)
    assert set(c["estimators"]) == set(cr.ESTIMATORS)
    assert c["target_fraction"] == pytest.approx(c["loss_fraction"][0] / c["loss_fraction"][1])
    assert c["expected_lost_fraction"] == pytest.approx(c["target_fraction"], abs=1e-9)
    assert isinstance(c["passed"], bool) and c["beta_true"] == 1.0
    g = cr.gate(p, seed=5, resamples=40)
    assert g["fit"]["status"] == "ok" and g["successes"] <= 40
    assert isinstance(g["passed"], bool)


def test_ipcw_go_needs_all_three_parts():
    ok = {"control": {"passed": True}, "sensitivity": {"ipcw_wins": True}, "gate": {"passed": True}}
    assert cr.ipcw_go(ok)
    for part, key in (("control", "passed"), ("sensitivity", "ipcw_wins"), ("gate", "passed")):
        bad = json.loads(json.dumps(ok))
        bad[part][key] = False
        assert not cr.ipcw_go(bad)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `W 'pytest tests/validation/test_censoring_report.py -q -W error'`
Expected: the 5 old tests pass; the new ones fail (`KeyError: 'ipcw'`, `TypeError: reanalysis() got an unexpected keyword argument 'seed'`, `AttributeError: ... 'control'`).

- [ ] **Step 4: Modify `tools/validation/censoring_report.py`**

Imports (after `sys.path.insert` lines; `HERE` is the script directory, so `ipcw` is importable as a script and from the tests):

```python
import warnings

import ipcw  # noqa: E402  (tools/validation/ipcw.py)

from gqteashmd.analysis import censoring, estimators, grid, kinetics, populations  # noqa: E402
```

Update the module docstring's usage line and first paragraph: the script now also reports the IPCW trial (spec 2026-11-26) with three pre-registered rules and `ipcw_go`.

Constants after `REPORT_TIMES_FS`:

```python
ESTIMATORS = ("censored", "still_running", "complete_only", "hold_in_s0", "ipcw")
CONTROL_BETA = 1.0  # spec 2.5 item 1, per SD of the running S0 kinetic energy
CONTROL_RULE = 1.0
GATE_RESAMPLES = 1000
GATE_MIN_SUCCESS_FRACTION = 0.95  # 950 of 1000
```

Replace `fit_excited` (the default path is unchanged: a non-finite curve gave `(None, None)` before too, through the fit's own failure):

```python
def fit_excited(
    t: np.ndarray, p: np.ndarray, dt_out: float, *, excited_sum: bool = False
) -> tuple[float | None, float | None]:
    y = p[:, 1:].sum(axis=1) if excited_sum else 1.0 - p[:, 0]
    if not np.all(np.isfinite(y)):
        return None, None
    r = fit_delayed_exponential(t, y, dt_out=dt_out)
    return r.tau_fs, r.t0_fs
```

Add after `four_estimates` (which stays unchanged):

```python
def estimates(
    complete: Sequence[Trajectory], steps: dict[str, int], p: dict[str, Any]
) -> tuple[dict[str, np.ndarray], grid.GridData, ipcw.CensoringFit]:
    """The four estimators plus IPCW (spec 2.4); IPCW draws no random numbers."""
    out, g = four_estimates(complete, steps, p)
    kin = ipcw.kinetic_on_grid(complete, p["e_kinetic"], g.grid_steps)
    curve, fit = ipcw.ipcw_estimate(g.active, g.observed, kin, p["n_states"])
    out["ipcw"] = curve if curve is not None else np.full_like(out["censored"], np.nan)
    return out, g, fit


def _summarise(
    curves: dict[str, list[np.ndarray]],
    fits: dict[str, list[tuple[float | None, float | None]]],
    truth: np.ndarray,
    unit: np.ndarray,
    tau_true: float | None,
    t0_true: float | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with warnings.catch_warnings():  # an estimator that failed everywhere is all NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in curves:
            mean = np.nanmean(np.asarray(curves[k]), axis=0)
            taus = [f[0] for f in fits[k] if f[0] is not None]
            t0s = [f[1] for f in fits[k] if f[1] is not None]
            out[k] = {
                "max_bias_over_se": max_bias_ratio(mean, truth, unit),
                "max_abs_bias": float(np.nanmax(np.abs(mean - truth))),
                "tau_bias_fs": (float(np.mean(taus)) - tau_true) if taus and tau_true else None,
                "t0_bias_fs": (float(np.mean(t0s)) - t0_true)
                if t0s and t0_true is not None
                else None,
                "fits_failed": sum(1 for f in fits[k] if f[0] is None),
            }
    return out


def _ipcw_fit_summary(failures: int, betas: list[float]) -> dict[str, Any]:
    return {"failures": failures, "beta_mean": float(np.mean(betas)) if betas else None}
```

Replace the body of `artificial` from `curves: ...` to the end (everything before it is unchanged, so the random draws are unchanged):

```python
    curves: dict[str, list[np.ndarray]] = {k: [] for k in ESTIMATORS}
    fits: dict[str, list[tuple[float | None, float | None]]] = {k: [] for k in ESTIMATORS}
    units = []
    failures, betas = 0, []
    for _ in range(repetitions):
        steps: dict[str, int] = {}
        for t in reach:
            chosen = (t.task in hot_set) if hot else (rng.random() < fraction)
            if not chosen:
                continue
            step = censor_step_for(t, float(rng.choice(delays)), p["dt_fs"], p["output_every"])
            if step is not None:
                steps[t.task] = step
        est, g, cfit = estimates(complete, steps, p)
        for k, v in est.items():
            curves[k].append(v)
            fits[k].append(fit_excited(g.time_fs, v, dt_out, excited_sum=k == "ipcw"))
        if cfit.ok and cfit.beta is not None:
            betas.append(cfit.beta)
        else:
            failures += 1
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
        "estimators": _summarise(curves, fits, truth, unit, tau_true, t0_true),
        "ipcw_fit": _ipcw_fit_summary(failures, betas),
    }
    r = {k: v["max_bias_over_se"] for k, v in result["estimators"].items()}
    if not hot:  # the verdict fixed in spec 4.2 item 5 (censored analysis)
        result["verdict_good"] = bool(
            r["censored"] <= 1.0
            and r["censored"] < r["complete_only"]
            and r["censored"] < r["still_running"]
        )
    else:  # spec 2026-11-26, 2.5 item 2
        result["ipcw_wins"] = bool(
            failures == 0 and r["ipcw"] < r["censored"] and r["ipcw"] < r["hold_in_s0"]
        )
    return result
```

Add `control`:

```python
def control(p: dict[str, Any], *, repetitions: int, bootstrap: int, seed: int) -> dict[str, Any]:
    """Spec 2.5 item 1: censoring in S0 with logit h = a + 1.0 z(running S0 mean)."""
    from scipy.optimize import brentq
    from scipy.special import expit

    trajs = [t for t in p["trajectories"].values() if t is not None]
    complete = [t for t in trajs if t.outcome == "finished"]
    n_lost, n_reach = reaching_s0_fraction(trajs)
    target = n_lost / n_reach
    dt_out = p["output_every"] * p["dt_fs"]
    truth_est, g0 = four_estimates(complete, {}, p)
    truth = truth_est["censored"]
    tau_true, t0_true = fit_excited(g0.time_fs, truth, dt_out)
    kin = ipcw.kinetic_on_grid(complete, p["e_kinetic"], g0.grid_steps)
    kbar = ipcw.running_s0_mean(g0.active, kin, g0.observed)[:, :-1]
    in_s0 = g0.active[:, :-1] == 0
    values = kbar[in_s0]
    z = np.where(in_s0, (kbar - values.mean()) / values.std(), 0.0)
    reach = np.array([first_s0_time(t) is not None for t in complete])  # plan ruling PR4

    def hazard(a: float) -> np.ndarray:
        return np.where(in_s0, expit(a + CONTROL_BETA * z), 0.0)

    def lost(a: float) -> float:
        return float(1.0 - np.prod(1.0 - hazard(a)[reach], axis=1).mean())

    a = float(brentq(lambda x: lost(x) - target, -30.0, 10.0, xtol=1e-12))
    h_true = hazard(a)
    rng = np.random.default_rng(seed + 2)
    curves: dict[str, list[np.ndarray]] = {k: [] for k in ESTIMATORS}
    fits: dict[str, list[tuple[float | None, float | None]]] = {k: [] for k in ESTIMATORS}
    units, realised = [], []
    failures, betas = 0, []
    for _ in range(repetitions):
        hit = rng.random(h_true.shape) < h_true
        steps = {
            t.task: int(g0.grid_steps[np.flatnonzero(hit[i])[0]])
            for i, t in enumerate(complete)
            if hit[i].any()
        }
        realised.append(len(steps) / int(reach.sum()))
        est, g, cfit = estimates(complete, steps, p)
        for k, v in est.items():
            curves[k].append(v)
            fits[k].append(fit_excited(g.time_fs, v, dt_out, excited_sum=k == "ipcw"))
        if cfit.ok and cfit.beta is not None:
            betas.append(cfit.beta)
        else:
            failures += 1
        idx = populations.bootstrap_indices(rng, len(complete), bootstrap)
        boot = censoring.bootstrap_aalen_johansen(g.active, g.observed, p["n_states"], idx)
        units.append(boot.std(axis=0, ddof=1))
    unit = np.mean(units, axis=0)
    summary = _summarise(curves, fits, truth, unit, tau_true, t0_true)
    return {
        "n_complete": len(complete),
        "loss_fraction": [n_lost, n_reach],
        "target_fraction": target,
        "expected_lost_fraction": lost(a),
        "mean_realised_lost_fraction": float(np.mean(realised)),
        "intercept": a,
        "beta_true": CONTROL_BETA,
        "repetitions": repetitions,
        "truth": {"tau_fs": tau_true, "t0_fs": t0_true},
        "estimators": summary,
        "ipcw_fit": _ipcw_fit_summary(failures, betas),
        "passed": bool(failures == 0 and summary["ipcw"]["max_bias_over_se"] <= CONTROL_RULE),
    }
```

Extract the grid building of `reanalysis` into `_censored_grid` (the same code, so the numbers are unchanged) and add the gate and the IPCW block:

```python
def _censored_grid(p: dict[str, Any]) -> dict[str, Any]:
    trajs = p["trajectories"]
    tasks = list(trajs)
    complete, _ = grid.select(tasks, trajs, {}, energy_jump_eV=0.1, exclude_energy_jumps=False)
    used, steps, _cens, excl = censoring.select_censored(
        tasks,
        trajs,
        output_every=p["output_every"],
        n_total=p["n_total"],
        dt_fs=p["dt_fs"],
        energy_jump_eV=None,
    )
    h = grid.horizon_steps(complete, p["n_total"])
    steps = {k: v for k, v in steps.items() if v < h}
    kw = {
        "output_every": p["output_every"],
        "dt_fs": p["dt_fs"],
        "n_total": p["n_total"],
        "n_states": p["n_states"],
        "horizon": h,
    }
    return {
        "complete": complete,
        "used": used,
        "steps": steps,
        "excluded": excl,
        "h": h,
        "g_c": grid.on_grid(complete, **kw),
        "g_u": grid.on_grid(used, censored_at=steps, **kw),
    }


def gate(p: dict[str, Any], *, seed: int, resamples: int = GATE_RESAMPLES) -> dict[str, Any]:
    """Spec 2.5 item 3: does the real loss hazard in S0 grow with the S0 kinetic energy?"""
    c = _censored_grid(p)
    used, g, s = c["used"], c["g_u"], p["n_states"]
    kin = ipcw.kinetic_on_grid(used, p["e_kinetic"], g.grid_steps)
    fit = ipcw.fit_censoring_hazard(g.active, g.observed, kin, s)
    out: dict[str, Any] = {
        "fit": fit.summary(),
        "resamples": resamples,
        "successes": 0,
        "beta_interval_per_sd": None,
        "beta_interval_per_eV": None,
        "passed": False,
    }
    if not fit.ok:
        return out
    idx = populations.bootstrap_indices(np.random.default_rng(seed + 3), len(used), resamples)
    betas = []
    for r in idx:
        f = ipcw.fit_censoring_hazard(g.active[r], g.observed[r], kin[r], s)
        if f.ok and f.beta is not None:
            betas.append(f.beta)
    out["successes"] = len(betas)
    if len(betas) >= 2:
        lo, hi = (float(v) for v in np.percentile(betas, [2.5, 97.5]))
        out["beta_interval_per_sd"] = [lo, hi]
        if fit.sd_eV:
            out["beta_interval_per_eV"] = [lo / fit.sd_eV, hi / fit.sd_eV]
        out["passed"] = bool(len(betas) >= GATE_MIN_SUCCESS_FRACTION * resamples and lo > 0.0)
    return out


def _ipcw_block(
    p: dict[str, Any], used: Sequence[Trajectory], g: grid.GridData, *, seed: int
) -> dict[str, Any]:
    """Spec 2.6: IPCW on a real project, the censoring model refitted in every resample."""
    s = p["n_states"]
    dt_out = p["output_every"] * p["dt_fs"]
    kin = ipcw.kinetic_on_grid(used, p["e_kinetic"], g.grid_steps)
    curve, fit = ipcw.ipcw_estimate(g.active, g.observed, kin, s)
    out: dict[str, Any] = {"fit": fit.summary()}
    if curve is None or fit.hazard is None:
        return out
    weights = ipcw.censoring_weights(fit.hazard, g.observed)
    rng = np.random.default_rng(seed + 4)  # plan ruling PR2
    resampled, refit_failures = [], 0
    for r in populations.bootstrap_indices(rng, len(used), p["n_bootstrap"]):
        c, _ = ipcw.ipcw_estimate(g.active[r], g.observed[r], kin[r], s)
        if c is None:
            refit_failures += 1
        else:
            resampled.append(c[:, 1:].sum(axis=1))
    fu = kinetics.bootstrap_fit_curves(
        g.time_fs, curve[:, 1:].sum(axis=1), resampled, n_trajectories=len(used), dt_out=dt_out
    )
    rows = {}
    for when in REPORT_TIMES_FS:
        k = int(np.argmin(np.abs(g.time_fs - when)))
        rows[f"{g.time_fs[k]:.2f}"] = {"P": curve[k].tolist()}
    out.update(
        {
            "populations": rows,
            "tau_fs": fu.fit.tau_fs,
            "tau_interval_fs": fu.tau_interval_fs,
            "t0_fs": fu.fit.t0_fs,
            "t0_interval_fs": fu.t0_interval_fs,
            "max_weight": float(weights.max()),
            "max_row_sum_deviation": float(np.abs(curve.sum(axis=1) - 1.0).max()),
            "refit_failures": refit_failures,
            "fit_bootstrap_failures": fu.bootstrap_failures,
        }
    )
    return out


def ipcw_go(report: dict[str, Any]) -> bool:
    """Spec 2.5: control and verdict and gate."""
    return bool(
        report["control"]["passed"]
        and report["sensitivity"]["ipcw_wins"]
        and report["gate"]["passed"]
    )
```

In `reanalysis`, change the signature to `def reanalysis(p: dict[str, Any], *, seed: int) -> dict[str, Any]:`, replace its first block (from `trajs = p["trajectories"]` to `g_u = grid.on_grid(used, censored_at=steps, **kw)`) with:

```python
    c = _censored_grid(p)
    complete, used, steps, excl, h = c["complete"], c["used"], c["steps"], c["excluded"], c["h"]
    g_c, g_u = c["g_c"], c["g_u"]
```

keep the rest, and before `return out` add `out["ipcw"] = _ipcw_block(p, used, g_u, seed=seed)`.

In `main`:

```python
    report = {
        "artificial": artificial(
            base, repetitions=args.repetitions, bootstrap=args.bootstrap, seed=args.seed, hot=False
        ),
        "sensitivity": artificial(
            base, repetitions=args.repetitions, bootstrap=args.bootstrap, seed=args.seed, hot=True
        ),
        "control": control(
            base, repetitions=args.repetitions, bootstrap=args.bootstrap, seed=args.seed
        ),
        "gate": gate(base, seed=args.seed),
        "back_hop_rates": back_hop_rates(base),
        "reanalysis": {
            str(r): reanalysis(load_project(r), seed=args.seed)
            for r in [args.artificial, *args.project]
        },
    }
    report["ipcw_go"] = ipcw_go(report)
    args.out.write_text(json.dumps(jsonable(report), indent=2) + "\n")
    print(
        json.dumps(
            {"verdict_good": report["artificial"]["verdict_good"], "ipcw_go": report["ipcw_go"]}
        )
    )
    return 0
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `W 'pytest tests/validation/test_censoring_report.py tests/validation/test_ipcw.py -q -W error'`
Expected: 8 + 5 old + 5 new = 18 passed (the parametrised test counts twice). A failure of `test_the_four_estimators_are_unchanged_by_the_ipcw_addition` means the refactor moved a random draw or changed arithmetic: fix the code, not the stored JSON.

Then `W 'OPENBLAS_CORETYPE=SandyBridge pytest tests/validation/test_censoring_report.py -q -W error'` (expected: all pass).

- [ ] **Step 6: Lint and commit**

```bash
W 'ruff format tools/validation tests/validation && ruff check tools/validation tests/validation'
W 'git add tools/validation/censoring_report.py tests/validation/test_censoring_report.py tests/validation/censoring_synthetic.py tests/validation/data/censoring_report_before.json && git commit -F - <<EOF
feat: add the IPCW trial to the censoring report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF'
```

---

### Task 3: the measurement (no diff)

**Files:** none in the repository; output `~/ipcw/report.json` in WSL; ledger `task-3-report.md`.

**Interfaces:** Consumes the script of Task 2 at its commit; produces the numbers of the report (Task 6).

- [ ] **Step 1: Check the inputs exist**

Run: `W 'ls ~/p9b/ours/run/.gqteashmd/config.json ~/hop/ours_previous/run/.gqteashmd/config.json ~/p9a/acceptance/run/.gqteashmd/config.json && git log --oneline -1'`
Expected: three paths and the Task 2 commit. If a project is missing, stop and report to the controller (do not substitute another project).

- [ ] **Step 2: Run the script once, detached, with the fixed rules**

```bash
W 'mkdir -p ~/ipcw && cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && nohup /usr/bin/time -v python tools/validation/censoring_report.py --artificial ~/p9b/ours/run --project ~/hop/ours_previous/run --project ~/p9a/acceptance/run --out ~/ipcw/report.json > ~/ipcw/run.log 2>&1 &'
```

Defaults (R 200, B 200, seed 20261119) are not overridden. Wait for it to end (check `~/ipcw/run.log` for the final JSON line and the `time -v` block); record the elapsed time and the exit status.

- [ ] **Step 3: Check that the four existing estimators reproduce the 2026-11-19 run**

```bash
W 'python - <<EOF
import json, os
new = json.load(open(os.path.expanduser("~/ipcw/report.json")))
old = json.load(open(os.path.expanduser("~/censoring/report.json")))
for part in ("artificial", "sensitivity"):
    for k in ("censored", "still_running", "complete_only", "hold_in_s0"):
        a, b = new[part]["estimators"][k], old[part]["estimators"][k]
        print(part, k, a["max_bias_over_se"], b["max_bias_over_se"], a["tau_bias_fs"], b["tau_bias_fs"])
for path, r in new["reanalysis"].items():
    print(path, r["complete"]["tau_fs"], old["reanalysis"][path]["complete"]["tau_fs"], r["censored"]["tau_fs"], old["reanalysis"][path]["censored"]["tau_fs"])
EOF'
```

Expected: equal to at least 9 significant digits (censored 0.561 / 1.244, hold-in-S0 1.152 / 0.630, tau 80.136 / 62.494 fs, ...). If not, stop: the trial is not comparable with the earlier report, and the controller decides.

- [ ] **Step 4: Record the numbers** in `task-3-report.md`: the control (`intercept`, expected and realised lost fraction, every estimator's ratio, IPCW failures, `passed`), the sensitivity (every ratio, `ipcw_wins`), the gate (β per SD and per eV, its interval, successes, `n_events_s0`, `n_informative`, `passed`), `ipcw_go`, the IPCW reanalysis of the three projects (tau, t0 and intervals, populations at 25/50/75/100 fs, largest weight, row-sum deviation, failures), the run time. Copy `~/ipcw/report.json` to the ledger as `task-3-report.json`. **No rule, bound, seed or setting is changed after the numbers are seen**; a wrong rule goes to the controller as a question for the owner.

---

### Task 4: analysis stage version 4, the horizon cap

**Files:**
- Modify: `src/gqteashmd/stages/analysis.py` (`AnalysisStage.version`, `run_task` around lines 192-208 and 291-296, `_base_summary`)
- Test: `tests/stages/test_analysis.py`

**Interfaces:**
- Produces: `analysis.json` top-level `horizon_capped: bool` (always present; `False` on a stop before the horizon is known is not possible: it is computed before every stop), `schema_version: 3`; `AnalysisStage.version == 4`; the stop text `NO_DATA_AFTER_START = "no trajectory is complete and none is observed after t = 0"` (module constant); the warning `no trajectory is complete: the analysis horizon is capped at {t:g} fs, the last time any trajectory is observed`.

- [ ] **Step 1: Write the failing tests** (in `tests/stages/test_analysis.py`)

Replace `test_without_a_complete_trajectory_the_censored_estimator_still_runs` and the constant `NO_COMPLETE_WARNING` with:

```python
def test_without_a_complete_trajectory_the_horizon_is_capped(tmp_path, write_input):
    # no finished trajectory: the horizon is the last observed grid step, 1.5 fs (traj_00001,
    # energy_drift, records up to grid step 3), not tmax; traj_00001 is then observed through
    # the horizon and counts as complete there (ruling PR5); the complete block stays null
    trajs = [("overlap_failure", [1, 0, 0], {}), ("energy_drift", [1, 1, 1, 1], {})]
    ctx = _context(tmp_path, write_input, trajs, estimator="censored")
    run_stage(ctx)
    data = _analysis(ctx)
    assert data["horizon_fs"] == pytest.approx(1.5) and data["horizon_capped"] is True
    assert data["n_complete"] == 0 and data["n_censored"] == 1 and data["n_included"] == 2
    assert data["estimators"]["complete"] is None and data["consistency"] is None
    assert data["hops"]["pairs"] == {}
    with h5py.File(ctx.stage_dir / "populations.h5", "r") as f:
        assert not any(name.startswith("complete_") for name in f)
        assert f.attrs["estimator"] == "censored" and f.attrs["n_complete"] == 0
        assert f["time_fs"][()][-1] == pytest.approx(1.5)
    assert "complete_P_S0" not in (ctx.stage_dir / "populations.csv").read_text()
    assert CAPPED_WARNING.format(t="1.5") in data["warnings"]
    assert not any("is held constant" in w for w in data["warnings"])
    # traj_00000 is in S0 at its censoring step (1.0 fs) and nobody is at risk in S0 after it
    n_empty = data["estimators"]["censored"]["empty_rows"]
    assert n_empty == 1 and EMPTY_ROWS_WARNING.format(n=n_empty) in data["warnings"]


def test_a_cap_at_zero_stops_the_stage(tmp_path, write_input):
    trajs = [("overlap_failure", [1], {}), ("energy_drift", [1], {})]
    ctx = _context(tmp_path, write_input, trajs, estimator="censored")
    with pytest.raises(StageFailedError, match="none is observed after t = 0"):
        run_stage(ctx)
    data = _analysis(ctx)
    assert data["stage_failed"] is True and data["horizon_capped"] is True
    assert data["error"] == analysis_module.NO_DATA_AFTER_START


def test_with_a_complete_trajectory_the_horizon_is_not_capped(tmp_path, write_input):
    trajs = [(FIN, [1, 1, 1, 1, 1], {}), ("overlap_failure", [1, 0, 0], {})]
    ctx = _context(tmp_path, write_input, trajs, estimator="censored")
    run_stage(ctx)
    data = _analysis(ctx)
    assert data["horizon_capped"] is False and data["horizon_fs"] == pytest.approx(2.0)


CAPPED_WARNING = (
    "no trajectory is complete: the analysis horizon is capped at {t} fs, the last time any "
    "trajectory is observed"
)
```

Change the two version assertions: in `test_stage_attributes` `stage.version == 4`; in `test_censored_estimator_uses_a_lost_trajectory_up_to_its_end` `data["schema_version"] == 3`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `W 'pytest tests/stages/test_analysis.py -q -W error'`
Expected: the three new tests and the two version assertions fail (`KeyError: 'horizon_capped'`, `assert 3 == 4`, `assert 2 == 3`, the horizon 2.0 instead of 1.5).

- [ ] **Step 3: Implement**

In `stages/analysis.py`:

```python
NO_DATA_AFTER_START = "no trajectory is complete and none is observed after t = 0"
```

next to the other constants; `version = 4` with the comment `# 4: the horizon cap without complete trajectories and the hold-in-S0 estimate (spec 2026-11-26)`.

Replace

```python
        h = grid.horizon_steps(complete, n_total)  # ruling PR7
```

with

```python
        h = grid.horizon_steps(complete, n_total)  # ruling PR7
        # spec 2026-11-26, 3.1: with no complete trajectory, the last observed grid step
        horizon_capped = not complete and bool(censor_steps)
        if horizon_capped:
            h = max(censor_steps.values())
```

Pass `horizon_capped` to `_base_summary` (new keyword parameter `horizon_capped: bool`, stored as `"horizon_capped": horizon_capped` after `"horizon_fs"`), and change `"schema_version": 2` to `3`.

After the `if not in_use:` stop, add:

```python
        if horizon_capped and h == 0:
            return _stop(task_dir, summary, NO_DATA_AFTER_START, len(in_use), n_total)
```

Replace the `if not complete:` warning block with:

```python
        if horizon_capped:
            warnings.append(
                f"no trajectory is complete: the analysis horizon is capped at "
                f"{g_censored.horizon_fs:g} fs, the last time any trajectory is observed"
            )
```

In the module docstring, after the sentence about the two inclusion rules, add: `With no complete trajectory the horizon is capped at the last grid step at which any trajectory is observed (spec 2026-11-26, 3.1).`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `W 'pytest tests/stages/test_analysis.py tests/stages/test_analysis_pipeline.py tests/analysis -q -W error -m "not slow"'`
Expected: all pass (the pin test T5b `test_the_complete_estimator_reproduces_main` too: the complete path is unchanged).

- [ ] **Step 5: Lint, type-check and commit**

```bash
W 'ruff format src tests && ruff check src tests && mypy'
W 'git add src/gqteashmd/stages/analysis.py tests/stages/test_analysis.py && git commit -F - <<EOF
feat: cap the analysis horizon when no trajectory is complete

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF'
```

---

### Task 5: the hold-in-S0 estimate and the lifetime range

**Files:**
- Modify: `src/gqteashmd/analysis/estimators.py`, `src/gqteashmd/stages/analysis.py` (`run_task` estimates, `summary.update`, the file writers, `_estimate_summary`, `_other_estimator` -> `_extra_curves`, `_base_summary`), `src/gqteashmd/analysis/report.py`
- Test: `tests/analysis/test_estimators.py`, `tests/stages/test_analysis.py`, `tests/analysis/test_report.py`

**Interfaces:**
- Consumes: `estimators.complete_estimate`, `grid.GridData` (fields `active`, `populations`, `observed`).
- Produces:
  - `estimators.complete_estimate(..., name: str = "complete")`;
  - `estimators.hold_in_s0_grid(g: GridData) -> GridData` (unobserved steps: active 0, populations `(1, 0, ...)`, observed all `True`);
  - `estimators.hold_in_s0_estimate(g, fit_index, *, fit_population, rng, n_bootstrap, dt_out) -> Estimate` (`name == "hold_in_s0"`);
  - `analysis.json`: `estimators.hold_in_s0` (`n`, `small_ensemble`, `fit`, `table`, `rule`) or `null`; top-level `tau_range_fs` (`min`, `max`, `min_estimator`, `max_estimator`) or `null`;
  - files: `hold_in_s0_P_<state>`, `hold_in_s0_rho_<state>` (+ `_lower`, `_upper`) CSV columns and `hold_in_s0_classical`, ..., `hold_in_s0_quantum_upper` HDF5 datasets;
  - `report.md` section `## Lifetime range`;
  - stage constant `HOLD_RULE = "lost trajectories held in S0 from their censoring step"`.

- [ ] **Step 1: Write the failing tests**

`tests/analysis/test_estimators.py` (append; reuse the file's existing imports, add `from gqteashmd.analysis import estimators, grid` if missing):

```python
def _held_example():
    active = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]])
    observed = np.array([[True, True, True, False], [True, True, False, False], [True] * 4])
    pops = np.eye(2)[active].astype(float)
    g = grid.GridData(
        np.arange(4),
        0.5 * np.arange(4),
        active,
        pops,
        ("a", "b", "c"),
        3,
        1.5,
        (),
        observed,
    )
    return g


def test_hold_in_s0_grid_puts_the_unobserved_steps_in_s0():
    held = estimators.hold_in_s0_grid(_held_example())
    np.testing.assert_array_equal(held.active, [[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]])
    assert held.observed.all()
    np.testing.assert_array_equal(held.populations[1, 3], [1.0, 0.0])


def test_hold_in_s0_estimate_is_the_complete_arithmetic_on_the_held_grid():
    g = _held_example()
    kw = {"fit_population": "classical", "n_bootstrap": 100, "dt_out": 0.5}
    hold = estimators.hold_in_s0_estimate(g, [1], rng=np.random.default_rng(3), **kw)
    direct = estimators.complete_estimate(
        estimators.hold_in_s0_grid(g), [1], rng=np.random.default_rng(3), **kw
    )
    assert hold.name == "hold_in_s0" and direct.name == "complete"
    np.testing.assert_array_equal(hold.classical, direct.classical)
    np.testing.assert_array_equal(hold.classical_lower, direct.classical_lower)
    assert hold.fit.fit.tau_fs == direct.fit.fit.tau_fs
    np.testing.assert_allclose(hold.classical[:, 0], [0.0, 1 / 3, 2 / 3, 2 / 3])
```

`tests/stages/test_analysis.py` (append):

```python
def test_the_hold_in_s0_block_and_the_lifetime_range(tmp_path, write_input):
    trajs = [
        (FIN, [1, 1, 1, 1, 1], {}),
        (FIN, [1, 1, 1, 0, 0], {}),
        ("overlap_failure", [1, 1, 1], {}),  # lost in S1 at 1.0 fs: held in S0 after it
    ]
    ctx = _context(tmp_path, write_input, trajs, estimator="censored")
    run_stage(ctx)
    data = _analysis(ctx)
    block = data["estimators"]["hold_in_s0"]
    assert block["n"] == 3 and block["rule"] == analysis_module.HOLD_RULE
    row = next(r for r in block["table"] if r["time_fs"] == 1.5)
    assert row["P"]["S0"] == pytest.approx(2 / 3)  # traj_00001 decayed, traj_00002 held
    names = [n for n in ("complete", "censored", "hold_in_s0") if data["estimators"][n]]
    taus = {n: data["estimators"][n]["fit"]["tau_fs"] for n in names}
    taus = {n: t for n, t in taus.items() if t is not None}
    r = data["tau_range_fs"]
    if len(taus) >= 2:
        assert r["min"] == min(taus.values()) and r["max"] == max(taus.values())
        assert taus[r["min_estimator"]] == r["min"] and taus[r["max_estimator"]] == r["max"]
    else:
        assert r is None
    header = next(
        line
        for line in (ctx.stage_dir / "populations.csv").read_text().splitlines()
        if not line.startswith("#")
    ).split(",")
    assert "hold_in_s0_P_S0" in header and "hold_in_s0_rho_S1_upper" in header
    with h5py.File(ctx.stage_dir / "populations.h5", "r") as f:
        assert "hold_in_s0_classical" in f and "hold_in_s0_quantum_upper" in f
    assert "## Lifetime range" in (ctx.stage_dir / "report.md").read_text()


def test_without_a_censored_trajectory_there_is_no_hold_in_s0_block(tmp_path, write_input):
    ctx = _context(tmp_path, write_input, _varied_trajectories(), estimator="censored")
    run_stage(ctx)
    data = _analysis(ctx)
    assert data["estimators"]["hold_in_s0"] is None
    assert "hold_in_s0_P_S0" not in (ctx.stage_dir / "populations.csv").read_text()
```

That streams 0 and 1 are unchanged is checked by the existing tests staying green: the pin test T5b (`test_the_complete_estimator_reproduces_main`, the complete estimator against `main` `f738b35`) and `test_without_losses_the_two_estimators_agree`; no new test is needed for it.

`tests/analysis/test_report.py` (append):

```python
def test_markdown_shows_the_lifetime_range(summary_example):
    s = dict(summary_example)
    fit = {"tau_fs": 50.0, "tau_interval_fs": [40.0, 60.0], "t0_fs": 5.0, "t0_interval_fs": [4.0, 6.0]}
    s["estimators"] = {
        "complete": {"fit": dict(fit, tau_fs=80.0)},
        "censored": {"fit": dict(fit, tau_fs=62.5)},
        "hold_in_s0": {"fit": fit},
    }
    s["tau_range_fs"] = {
        "min": 50.0,
        "max": 80.0,
        "min_estimator": "hold_in_s0",
        "max_estimator": "complete",
    }
    text = analysis_markdown(s)
    section = text.split("## Lifetime range\n", 1)[1].split("\n## ", 1)[0]
    assert "- complete: tau 80 fs [40, 60], t0 5 fs [4, 6]" in section
    assert "- range: 50-80 fs (hold_in_s0 to complete)" in section
    assert "Which estimate is closest to the truth is not known (USER_MANUAL 8.19)." in section
    assert text.index("## Other estimator") < text.index("## Lifetime range")
    empty = analysis_markdown(summary_example).split("## Lifetime range\n", 1)[1]
    assert "- range: -" in empty
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `W 'pytest tests/analysis/test_estimators.py tests/analysis/test_report.py tests/stages/test_analysis.py -q -W error'`
Expected: the new tests fail (`AttributeError: ... 'hold_in_s0_grid'`, `KeyError: 'hold_in_s0'`, the missing section).

- [ ] **Step 3: Implement**

`analysis/estimators.py`: add `import dataclasses` at the top; give `complete_estimate` the keyword `name: str = "complete"` and use `name` instead of the literal `"complete"` in its `Estimate(...)`; add:

```python
def hold_in_s0_grid(g: GridData) -> GridData:
    """The held grid of spec 2026-11-26, 3.2: from its censoring step on, a censored trajectory
    is in S0 (classical) with populations (1, 0, ...) (quantum)."""
    observed = np.asarray(g.observed, dtype=bool)
    ground = np.zeros(g.populations.shape[-1])
    ground[0] = 1.0
    return dataclasses.replace(
        g,
        active=np.where(observed, g.active, 0),
        populations=np.where(observed[..., None], g.populations, ground),
        observed=np.ones_like(observed),
    )


def hold_in_s0_estimate(
    g: GridData,
    fit_index: Sequence[int],
    *,
    fit_population: str,
    rng: np.random.Generator,
    n_bootstrap: int,
    dt_out: float,
) -> Estimate:
    return complete_estimate(
        hold_in_s0_grid(g),
        fit_index,
        fit_population=fit_population,
        rng=rng,
        n_bootstrap=n_bootstrap,
        dt_out=dt_out,
        name="hold_in_s0",
    )
```

Update the module docstring: three estimators, `hold_in_s0_estimate` reported only.

`stages/analysis.py`:

- Constant `HOLD_RULE = "lost trajectories held in S0 from their censoring step"`.
- After `est_censored = ...`:

```python
        # spec 2026-11-26, 3.2: reported only; stream 2, so streams 0 and 1 are unchanged
        est_hold = (
            estimators.hold_in_s0_estimate(
                g_censored,
                fit_index,
                fit_population=section.fit_population,
                rng=task_rng(seed, "analysis", 2),
                n_bootstrap=section.n_bootstrap,
                dt_out=dt_out,
            )
            if censor_steps
            else None
        )
```

- In `summary.update`, add to `"estimators"`:

```python
                    "hold_in_s0": None
                    if est_hold is None
                    else _estimate_summary(
                        est_hold, g_censored, labels, fit_states, section.fit_population
                    ),
```

  and the top-level `"tau_range_fs": _tau_range({"complete": est_complete, "censored": est_censored, "hold_in_s0": est_hold}),`.
- `_base_summary`: add `"tau_range_fs": None,` after `"estimators": None,`.
- `_estimate_summary`: after the `censored` branch add `if est.name == "hold_in_s0": out["rule"] = HOLD_RULE`.
- Replace `_other_estimator` by `_extra_curves` (same keys per estimate as today) and change its call to `extra, extra_blocks = _extra_curves(est_complete if use_censored else est_censored, est_hold)`:

```python
def _extra_curves(
    *others: estimators.Estimate | None,
) -> tuple[dict[str, np.ndarray], list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]]:
    """The curves of the estimators not in use, under their names as prefix (ruling PR6; spec
    2026-11-26, 3.3); nothing for an estimator without trajectories."""
    extra: dict[str, np.ndarray] = {}
    extra_blocks: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for other in others:
        if other is None:
            continue
        prefix = other.name  # "complete", "censored" or "hold_in_s0"
        extra |= {
            f"{prefix}_classical": other.classical,
            f"{prefix}_classical_lower": other.classical_lower,
            f"{prefix}_classical_upper": other.classical_upper,
            f"{prefix}_quantum": other.quantum,
            f"{prefix}_quantum_lower": other.quantum_lower,
            f"{prefix}_quantum_upper": other.quantum_upper,
        }
        extra_blocks += [
            (f"{prefix}_P", other.classical, other.classical_lower, other.classical_upper),
            (f"{prefix}_rho", other.quantum, other.quantum_lower, other.quantum_upper),
        ]
    return extra, extra_blocks
```
- Add:

```python
def _tau_range(blocks: Mapping[str, estimators.Estimate | None]) -> dict[str, Any] | None:
    """Spec 2026-11-26, 3.3: over the estimators that exist and have a fitted tau."""
    taus = [
        (name, float(e.fit.fit.tau_fs))
        for name, e in blocks.items()
        if e is not None and e.fit.fit.tau_fs is not None
    ]
    if len(taus) < 2:
        return None
    low = min(taus, key=lambda x: x[1])
    high = max(taus, key=lambda x: x[1])
    return {"min": low[1], "max": high[1], "min_estimator": low[0], "max_estimator": high[0]}
```

`analysis/report.py`, after the `## Other estimator` block:

```python
    blocks = s.get("estimators") or {}
    out += ["", "## Lifetime range"]
    for name in ("complete", "censored", "hold_in_s0"):
        block = blocks.get(name)
        if not block:
            out.append(f"- {name}: -")
            continue
        f = block.get("fit") or {}
        out.append(
            f"- {name}: tau {_n(f.get('tau_fs'))} fs {_n(f.get('tau_interval_fs'))}, "
            f"t0 {_n(f.get('t0_fs'))} fs {_n(f.get('t0_interval_fs'))}"
        )
    r = s.get("tau_range_fs")
    if r:
        out.append(
            f"- range: {_n(r.get('min'))}-{_n(r.get('max'))} fs "
            f"({r.get('min_estimator')} to {r.get('max_estimator')})"
        )
    else:
        out.append("- range: -")
    out.append("Which estimate is closest to the truth is not known (USER_MANUAL 8.19).")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `W 'pytest tests/analysis tests/stages/test_analysis.py tests/stages/test_analysis_pipeline.py tests/config tests/test_orchestrator.py -q -W error -m "not pyscf"'`
Expected: all pass, among them T5b and `test_without_losses_the_two_estimators_agree`.
Then `W 'OPENBLAS_CORETYPE=SandyBridge pytest tests/analysis/test_estimators.py tests/analysis/test_report.py tests/stages/test_analysis.py -q -W error'` (all pass).

- [ ] **Step 5: Lint, type-check and commit**

```bash
W 'ruff format src tests && ruff check src tests && mypy'
W 'git add src/gqteashmd/analysis/estimators.py src/gqteashmd/analysis/report.py src/gqteashmd/stages/analysis.py tests/analysis/test_estimators.py tests/analysis/test_report.py tests/stages/test_analysis.py && git commit -F - <<EOF
feat: report the hold-in-S0 lifetime and the lifetime range

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF'
```

---

### Task 6: the report and the documents

**Files:**
- Create: `docs/superpowers/reports/2026-11-<dd>-ipcw-trial.md` (the date of the task)
- Modify: `USER_MANUAL.md` (the "Read this first" summary, section 6 `analysis.json` keys, 8.13 (the horizon cap, the hold-in-S0 block, the range), 8.19 (the trial and its numbers), the limitations list, section 11 "Next"), `CLAUDE.md` (the seeds rule: stream 2 and the report script's streams; the censored-analysis block's horizon text; a new block for this sub-project; Current status; Known limitations 1b; Remaining work: the parked items replaced), `CHANGELOG.md` (`Unreleased`: changed behaviour, the version 4 rerun advice), the spec (an erratum block at its top listing the plan rulings PR1-PR7 and anything else that differed)

**Interfaces:** Consumes the Task 3 numbers (`task-3-report.md`) and the Task 4-5 behaviour.

- [ ] **Step 1: Write the report.** Sections: status (one paragraph: `ipcw_go` and why), what was built, tests (each command and count), the control, the hot-S0 verdict, the gate, the IPCW reanalysis (tau, t0, intervals beside complete and censored; the version 4 hold-in-S0 and range numbers of the real projects are not part of the trial and are reported only if the controller reran the analysis on copies of the projects), plan rulings, "what was decided after numbers were seen" (state "nothing" only if true), honest summary. Every number marked *asserted*, *pre-registered verdict (report script)*, *report* or *inference*, as in `docs/superpowers/reports/2026-11-19-censored-analysis.md`.
- [ ] **Step 2: Update USER_MANUAL.md, CLAUDE.md, CHANGELOG.md and the spec erratum** as listed above. USER_MANUAL marks IPCW as **not implemented** in the stage in every case (with `ipcw_go` true it is the next spec, planned). Say that **every project with `analysis` results needs `--rerun-from analysis`** (version 4).
- [ ] **Step 3: Check the site test** (a renamed USER_MANUAL heading breaks a site link): `W 'pytest tests/site -q -W error'`. Expected: all pass.
- [ ] **Step 4: Run the fast tier** (the controller may run it on a WSL-native clone instead; record where): `W 'pytest -m "not slow and not pyscf" -W error -q'`. Record the counts and time in the ledger and in CLAUDE.md's Environment list.
- [ ] **Step 5: Commit**

```bash
W 'git add docs USER_MANUAL.md CLAUDE.md CHANGELOG.md && git commit -F - <<EOF
docs: record the IPCW trial and analysis version 4

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF'
```

After Task 6: the final whole-branch review, its fix wave, and superpowers:finishing-a-development-branch (the owner decides the merge).
