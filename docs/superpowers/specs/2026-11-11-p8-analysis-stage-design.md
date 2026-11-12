# P8: the `analysis` stage — design

Status: design approved by the owner in chat on 2026-11-11 (sections 1-4); this written spec awaits review.
Branch: `feat/p8-analysis-stage` (from `main` at `cc9d203`).

> **Erratum (Task 12, 2026-11-11): where the implementation differs from this spec.** The plan's rulings PR1-PR9 and the controller rulings P8-1 to P8-4 (ledger `.superpowers/sdd/2026-11-11-p8-analysis-stage/progress.md`) are binding; USER_MANUAL.md §4.2, §6, §7.5 and §8.13 document the code.
> - **Frustrated hops (PR2):** `output.h5` does not record the target of a frustrated hop, so section 6's "frustrated hops per pair" is **per originating state** (`hops.frustrated_from`), and `hops.csv` leaves `to` empty for them.
> - **Energy jump (PR3):** the jump of section 7 is taken over the records with `grid_step <= n_total` (the current `tmax_fs`), not "up to `T`", because `exclude_energy_jumps` decides the inclusion that defines `T`; the drift is taken up to the horizon.
> - **Reopening the analysis itself (P8-3):** `AnalysisStage` overrides `Stage.reopen_tasks` and reopens its task `analysis` when the recorded result's `n_total` differs from `round(tmax_fs / dt_fs)`. Lowering `tmax_fs` after a complete run therefore re-analyses at the new horizon on a plain run (the trajectories that ran longer are `cut`); raising it reopens the trajectories and section 9 resets the analysis first. `tmax_fs` is still in no fingerprint. Every task result carries `n_total`.
> - **Step-0 failures (P7b R6):** an `output.h5` without records (`electronic_failure` at t = 0) is excluded with `t_final_fs = 0`; the section 8 checks "no record at t = 0" and "non-finite `e_total`, `populations`, `active`" apply to **included** trajectories only. The other consistency checks (labels, `output_every`, `dt_fs` within 1e-9 relative (PR5), plus the `task` and `initial_state` attributes and the population width, which section 8 does not list) apply to every file.
> - **Shared reader (PR4, P8-1):** `trajectories.json` is read by `excite/artifacts.py:read_trajectories_json` in both stages; the `dynamics` error message is unchanged. `read_output_h5` is in `trajectory/store.py` as specified; the writers are in `analysis/artifacts.py` (PR1, h5py inside the functions).
> - **Additions not in this spec:** a fit flag `no_population` (`y(0) <= 0`, no fit); `t0` is bounded by the last grid time; the standard deviations use ddof = 1 and the intervals are `null` with fewer than two successful resamples; `analysis.json` has a `table` key (the rows of the report's population table, every `ceil(M/25)`-th point plus the last); `populations.h5`, `populations.csv` and `hops.csv` are not written when the stage stops (stale copies removed); an unknown `fit_states` label is also a defensive non-retryable `TaskError` in `run_task`.
> - **Final fix wave (after the final review; the three defects listed at Task 12 are fixed):** `report.md` prints integers exactly (`project_seed` 12345 as `12345`, no longer `1.234e+04`), floats with `.4g` and non-finite values as `-`; a trajectory without an active-0 record at or before the horizon is censored in `first_s0_arrival_fs`, also a `stopped_in_s0` one (as section 6 says; the implementation had counted it as arrived at its end).
> - **Implementation additions of the final fix wave:** **controller ruling P8-5:** a module constant `SMALL_ENSEMBLE = 20` in `stages/analysis.py` (not an input key); `analysis.json` gains the top-level boolean `small_ensemble` (`n_included < 20`, present on every path including the stops), `report.md` a `small_ensemble` line, and an analysis computed with `N < 20` the warning `only N trajectories are included (fewer than 20): populations, intervals and the fit are not statistically reliable`; with `N = 1` `bootstrap_fit` returns `null` for `t0_interval_fs`, `tau_interval_fs`, `t0_std_fs`, `tau_std_fs` (the quantum intervals stay as computed). Companions: a warning when `k > 0` included trajectories are in `energy.flagged` (`k included trajectories have an energy jump above energy_jump_eV = x eV (see energy.flagged); consider exclude_energy_jumps = true`); `log tau` is clipped to `[-700, 700]` in the fit (`kinetics.tau_from_log`), so `tau` cannot underflow to 0; on a stop `finalize` also removes stage-level `populations.h5/.csv` and `hops.csv` left by an earlier successful run.

## 0. Owner decisions (2026-11-11, multiple choice)

| Question | Decision |
|---|---|
| Observables (resolves **O10**) | Classical and quantum populations with 95 % confidence intervals, hop statistics, energy diagnostics, a delayed mono-exponential fit with bootstrap uncertainties |
| Which trajectories enter the averages | `finished` and `stopped_in_s0` only, up to a common end time; the early-ended ones are excluded and reported; warning when the excluded fraction is large. The rule is fixed here, before any data is seen |
| Analysis after a reopened `dynamics` stage | Recomputed automatically in the same run (no stale error) |
| Plots | None: CSV, HDF5, JSON and Markdown only (as P6 for the spectrum) |
| Fit target | The summed population of `fit_states` (default every excited state, i.e. `1 - P(S0)`), model with a delay `t0` and a lifetime `tau` |
| Energy jumps (the open P7b decision) | A diagnostic in `analysis`: flag trajectories with a large per-record `E_tot` jump; kept in the averages unless `exclude_energy_jumps = true`. The `dynamics` stage is not changed |
| Validation | Synthetic ensembles with known answers and model (LVC, fake backend) pipelines, asserted; no SHARC comparison in P8 (P9) |
| Milestone M4 | A documented real study: ethylene SA2-CASSCF(2,2)/6-31G\*, about 20 trajectories of 100 fs, with a kill and resume; the automated end-to-end test stays the small water CIS run |

## 1. Scope

In: the `analysis` stage (`07_analysis/`), the `[analysis]` input section, the `gqteashmd.analysis` package, a reader for `output.h5`, the orchestrator rule for the dependants of a reopened stage, tests, the M4 study, and the documents (CLAUDE.md, USER_MANUAL.md).

Out: plots; kinetic models other than the delayed mono-exponential; populations per initial state; geometry analysis (hop geometries, internal coordinates); comparison with SHARC's `populations.py`/`make_fit.py` (P9); any change to the `dynamics` stage, its outputs or its fingerprint; a standalone `analyze` command.

## 2. Architecture

```text
06_dynamics/tasks/traj_*/output.h5 ──read_output_h5──► TrajectoryData
05_selection/trajectories.json ──────────────────────► the selected set (failed dynamics tasks included)
06_dynamics/dynamics_summary.json ───────────────────► outcome per task
            │
analysis task (worker): inclusion rule → common grid → populations + intervals → fit (+ bootstrap)
                        → hop statistics → energy diagnostics → files in the task directory
            │
finalize (controller): copy to 07_analysis/, report.md, warnings, stop checks
```

Dependencies point down only:

- `gqteashmd/analysis/` (new package): plain functions on NumPy arrays, SciPy for the fit. It imports nothing from `stages/`, the orchestrator, `config/`, `electronic/` or PySCF, and no h5py. Modules:
  - `grid.py`: the inclusion rule and the common time grid (section 3).
  - `populations.py`: classical and quantum populations, Wilson and bootstrap intervals (section 4).
  - `kinetics.py`: the delayed mono-exponential fit and its bootstrap (section 5).
  - `hops.py`: hop statistics (section 6).
  - `energy.py`: drift and jump diagnostics (section 7).
  - `report.py`: `analysis.json` content and the `report.md` text.
- `trajectory/store.py` gains `read_output_h5(path) -> TrajectoryData` next to `write_output_h5` (one reader and writer per artifact). h5py is imported inside the function. `TrajectoryData` is a frozen dataclass with the datasets and attributes listed in USER_MANUAL §6 (`output.h5`), arrays as NumPy.
- The parse of `trajectories.json` now in `stages/dynamics.py` moves to one shared helper used by both stages (no behaviour change for `dynamics`).
- `stages/analysis.py`: `AnalysisStage`, thin: reads the inputs through the readers, calls the package, writes the files.

## 3. Stage contract

- `name = "analysis"`, `version = 1`, `depends_on = ("selection", "dynamics")` (the selection fingerprint is already chained through `dynamics`; listing it gives the stage `ctx.upstream_dir("selection")`), `config_sections = ("analysis",)` with a default section, `required_modules = ("h5py>=3.9", "scipy>=1.11")`.
- `enabled(config)`: true iff `config.dynamics is not None` and `config.analysis.enabled`. A missing `[analysis]` table gives the defaults, so the stage runs by default whenever `dynamics` runs (D6). An `[analysis]` table without `[dynamics]` is not an error: the stage is disabled, and the manual says so (as `pause_after` without a later stage).
- `default_pipeline` appends `AnalysisStage` after `dynamics`; it is the last stage.
- One task, `analysis`. `run_task` writes every output in its task directory and returns a small summary; `finalize` copies the outputs to `07_analysis/`, writes `report.md`, logs the warnings and raises the stops (section 8).
- Fingerprint: `[analysis]` without `enabled` and `warn_excluded_fraction` (finalize-only), plus `project.seed` (the bootstrap), plus the chained `selection` and `dynamics` fingerprints. No upstream artifact is read by `fingerprint_inputs`.
- Random numbers: `task_rng(project.seed, "analysis", 0)`; the resamples are drawn once (`integers(0, N, size=(B, N))`) and reused for the quantum intervals and the fit.

### 3.1 Inclusion rule (fixed before any data is seen)

For each trajectory selected in `trajectories.json`:

| Situation | Status | Reason recorded |
|---|---|---|
| outcome `finished` | included | — |
| outcome `stopped_in_s0` | included; after its end it is held in S0 (classical: active S0; quantum: `|c_S0|^2 = 1`, others 0) | — |
| outcome `energy_drift`, `overlap_failure`, `electronic_failure` | excluded | the outcome and `t_final_fs` |
| no result in `dynamics_summary.json` (a failed `dynamics` task tolerated by `max_failed_fraction`) or no `output.h5` | excluded | `task_failed` |
| `exclude_energy_jumps = true` and flagged by section 7 | excluded | `energy_jump` and the jump in eV |

`N` is the number of included trajectories. The excluded fraction is `(selected - N) / selected`.

### 3.2 Common time grid

- `dt_out = output_every * dt_fs`. Grid points `t_m = m * dt_out`, `m = 0, 1, ...`, while `t_m <= T` (comparison on the integer `grid_step`, never on floating time, as P7b ruling R3).
- The horizon `T` is `min(tmax_fs, min over the included finished trajectories of t_final_fs)`; with no finished trajectory, `T = tmax_fs`. The grid ends at the last multiple of `dt_out` not after `T`.
- A record at `t_m` is the record with `ticks = 0` and `grid_step = m * output_every`; P7b keeps every such record, and the last one.
- Records after `T` are ignored; the trajectories that had records after `T` are listed as `cut` (this happens when `tmax_fs` was lowered after they finished).
- The active state at `t_m` is exact (every hop is kept): the active state of the record at `t_m`. A `stopped_in_s0` trajectory past its end is in S0.

## 4. Populations

With `N_J(t_m)` the number of included trajectories whose active state is `J` at `t_m`:

- Classical: `P_J(t_m) = N_J(t_m) / N`, 95 % Wilson score interval with `z = 1.959963984540054`:
  `centre = (P + z^2/(2N)) / (1 + z^2/N)`, `half = z / (1 + z^2/N) * sqrt(P(1-P)/N + z^2/(4N^2))`.
- Quantum: `rho_J(t_m) = (1/N) * sum_n |c_J^(n)(t_m)|^2` (the `populations` dataset of `output.h5`). 95 % bootstrap percentile interval over `B = n_bootstrap` resamples of the N trajectories with replacement (percentiles 2.5 and 97.5, NumPy `percentile` with its default linear method).
- Consistency, reported only: `max over t_m of |P_J - rho_J|` per state.

## 5. Kinetic fit

- Target: `y(t_m) = sum over J in fit_states of P_J(t_m)` (`fit_population = "classical"`, default) or of `rho_J(t_m)` (`"quantum"`). Default `fit_states`: every excited state.
- Model: `f(t) = A` for `t < t0`, `f(t) = A * exp(-(t - t0) / tau)` for `t >= t0`, with `A = y(0)` fixed (1 whenever `fit_states` holds every initial state).
- Least squares, unweighted, over every grid point, with `scipy.optimize.least_squares`; parameters `t0` in `[0, T]` and `log tau` free (so `tau > 0`); multi-start from `t0` in `{0, T/4, T/2}` and `tau` from the time at which `y` first falls below `A/e` (or `T` if it never does); the start with the smallest cost wins.
- Uncertainty: the same fit on each bootstrap resample's target; 95 % percentile intervals and standard deviations of `tau` and `t0`.
- Reliability flags (never a stop; each one a warning and a flag in `analysis.json`):
  - `no_decay`: `y` never drops below `0.99 A`; then `tau` and `t0` are `null`;
  - `tau_beyond_horizon`: `tau > T`;
  - `little_decay`: `y(T) > 0.9 A`;
  - `t0_at_upper_bound`: `t0 > T - dt_out` (the decay starts at the end of the simulated time; a delay of 0 is a legitimate result and is not flagged);
  - `fit_failed`: `least_squares` reports failure for every start;
  - `bootstrap_failures`: the number of resamples whose fit failed (their values are left out of the intervals).

## 6. Hop statistics

Over the included trajectories up to `T` (and, beside them, totals over every trajectory with an `output.h5`): hops per pair (from, to) and frustrated hops per pair; the distribution of hops per trajectory (min, median, max, and the count with zero hops); the time of the first hop out of the initial state; the first arrival time in S0 (the first record with active S0): mean and median over the trajectories that arrive, and the number that have not arrived by `T` (censored). The first-arrival numbers are reported, not fitted.

## 7. Energy diagnostics

- Drift per trajectory: `max |E_tot - E_tot(0)|` over its records up to `T` (or its end); median, 95th percentile and maximum over the included ones.
- Jump per trajectory: `max |E_tot(r+1) - E_tot(r)|` over consecutive kept records up to `T`. With `output_every > 1` two consecutive full-step records are several steps apart; the manual says so.
- A trajectory is flagged when its jump exceeds `energy_jump_eV`. Flagged trajectories stay in the averages unless `exclude_energy_jumps = true` (section 3.1).
- **The default of `energy_jump_eV` is 0.1 eV**, with a rule fixed here: a plan task measures the largest per-step `|dE_tot|` over the accepted steps of the existing P7b validation runs (excluding the ethylene step-24 solution change of 0.976 eV); if that value exceeds 0.01 eV, the default becomes ten times it. This is decided before the M4 run and recorded with the number.

## 8. Checks, errors and stops

| Situation | Result |
|---|---|
| `trajectories.json` or `dynamics_summary.json` missing or malformed | non-retryable `TaskError` naming the file and `--rerun-from selection` or `--rerun-from dynamics` |
| an `output.h5` unreadable, or inconsistent with the others (different `state_labels`, `dt_fs` or `output_every`; no record at t = 0; a grid record missing before its end; non-finite `e_total`, `populations` or `active` in an included trajectory) | non-retryable `TaskError` naming the file and `--rerun-from dynamics` |
| `|sum_J |c_J|^2 - 1| > 1e-6` in an included record | `StageFailedError` ("indicates a program error"), after `analysis.json` is written |
| no trajectory included | `StageFailedError` with advice (the outcome counts, `exclude_energy_jumps`), after `analysis.json` and `report.md` are written with `stage_failed: true` |
| excluded fraction > `warn_excluded_fraction` | warning (log, report, `analysis.json`): possible selection bias |
| fit failed or unreliable | flags and warnings (section 5) |

Input errors (`ConfigError`, exit 2, every violation reported at once by the cross-field validator): `fit_states` with an unknown label, `S0`, a triplet, a duplicate or an empty list (checked against `[electronic] singlets`); `n_bootstrap < 100`; `energy_jump_eV <= 0` or not finite; `warn_excluded_fraction` outside `[0, 1]`; `fit_population` not `classical`/`quantum`; unknown keys (`extra="forbid"`).

## 9. Orchestrator: dependants of a reopened stage

Generic for every stage that overrides `Stage.reopen_tasks` (today `dynamics`):

1. When the hook returns a non-empty list for stage X (any path: DONE, RUNNING, PENDING, FAILED), the orchestrator **first** resets every later stage that depends on X directly or through other stages and has results (DONE, RUNNING, or any DONE task), whether enabled or disabled with a kept record: `_reset_stage` (its directory moves to `.gqteashmd/superseded/`, its record is reset), with the warning `Stage <name> is recomputed because <X> was extended`.
2. **Then** X is marked RUNNING (same fingerprint) and the tasks are requeued, as in P7b.

A kill at any point is safe: after step 1 and before step 2, X is still in its old state, the deterministic hook returns the same list, and the dependants have no results, so step 1 does nothing more; after step 2 the existing P7b path applies. No fingerprint changes, the stale pre-pass and `--rerun-from` are unchanged, and raising `tmax_fs` stays a plain `run`. A dependant without results, or skipped without a kept record, is left alone. This resolves the P8 deferral "a staleness rule for the dependants of a reopened `dynamics` stage".

Pause: `pause_after = ["dynamics"]` now pauses before `analysis` (a later stage is enabled); a reopened `dynamics` pauses again (P7b rule).

## 10. Input: `[analysis]` (optional; defaults for every key)

| Key | Type | Default | Rule | Hashed |
|---|---|---|---|---|
| `enabled` | bool | `true` | — | no |
| `fit_states` | list of labels | every excited state | excited singlet labels, upper-cased | yes |
| `fit_population` | string | `"classical"` | `classical` \| `quantum` | yes |
| `n_bootstrap` | int | 1000 | >= 100 | yes |
| `energy_jump_eV` | float | 0.1 (section 7 rule) | finite, > 0 | yes |
| `exclude_energy_jumps` | bool | `false` | — | yes |
| `warn_excluded_fraction` | float | 0.2 | 0 <= x <= 1 | no (finalize only) |

The `init` template gets a commented `[analysis]` block.

## 11. Outputs (`07_analysis/`)

Every HDF5 dataset carries a `units` attribute; file attributes `schema_version` = 1 and `gqteashmd_version`; JSON never holds NaN (`null` instead).

- `populations.h5`: `time_fs` (M), `state_labels` (n), `classical`, `classical_lower`, `classical_upper`, `quantum`, `quantum_lower`, `quantum_upper` (M x n), `fit_target`, `fit_curve` (M; `fit_curve` NaN-filled if there is no fit); attributes `n_included`, `n_selected`, `horizon_fs`, `dt_out_fs`, `fit_population`, `fit_states`.
- `populations.csv`: `#` comment lines (title, units, `n_included`, `horizon_fs`, settings), a header, one line per grid point in `%.6e`: `time_fs`, then per state `P_<L>`, `P_<L>_lower`, `P_<L>_upper`, then per state `rho_<L>`, `rho_<L>_lower`, `rho_<L>_upper`, then `fit_target`, `fit_curve` (empty when there is no fit).
- `hops.csv`: one row per hop or frustrated hop of every trajectory with an `output.h5`: `task`, `time_fs`, `from`, `to`, `frustrated` (0/1), `included` (0/1), `after_horizon` (0/1).
- `analysis.json`: `schema_version`; `n_selected`, `n_included`; `outcomes` (counts); `included` (tasks); `excluded` (`[{task, reason, t_final_fs, energy_jump_eV}]`); `cut` (tasks); `horizon_fs`, `dt_out_fs`; `fit` (`states`, `population`, `A`, `t0_fs`, `tau_fs`, `t0_interval_fs`, `tau_interval_fs`, `t0_std_fs`, `tau_std_fs`, `rms`, `flags`, `bootstrap_failures`); `hops` (section 6); `energy` (section 7, with `flagged`); `consistency` (`max |P - rho|` per state); `settings` (the `[analysis]` values and `project_seed`); `warnings`; `stage_failed`; `error`.
- `report.md`: the same, to read: settings, inclusion table, a population table (every k-th grid point so that it has at most about 25 rows), the fit with its flags, hop and energy tables, warnings, and the paths of the other files.

## 12. Testing

Asserted; every bound below is fixed in the plan before its first run.

- Fast tier (`analysis/` unit tests, no h5py or PySCF): Wilson values against a hand-computed case; interval coverage on synthetic Bernoulli ensembles (the observed coverage of the 95 % intervals over R repetitions must lie inside a binomial acceptance band fixed in the plan); the fit recovering known `tau` and `t0` from synthetic ensembles of delayed-exponential first-passage times (within a bound fixed in the plan, stated as a multiple of the bootstrap standard deviation); the flags; the inclusion, horizon, cut and censoring rules; `stopped_in_s0` holding; the jump flag; bootstrap determinism from the seed.
- Fast tier: `read_output_h5` round trip against `write_output_h5`; the shared `trajectories.json` parse (existing `dynamics` tests stay green).
- Fast tier (orchestrator, fake stages): a reopen resets the dependants with results (also a disabled one with a kept record) and leaves the others; kill points between step 1 and step 2 and after step 2; nothing is deleted (superseded).
- Fast tier (pipeline: canned LVC upstream + fake backend + real `dynamics` + real `analysis`): the stage's populations equal a direct recomputation from the `output.h5` files (1e-15); raising `tmax_fs` recomputes the analysis at the new horizon in one run; `pause_after = ["dynamics"]` pauses before analysis. Slow: a kill and resume through `analysis`.
- `pyscf`: the existing water CIS end-to-end test runs through `analysis`.
- Config: every `[analysis]` rule, all violations at once.

## 13. Milestone M4 study (report-only, documented)

Ethylene SA2-CASSCF(2,2)/6-31G\* through the whole pipeline from `tests/data/ethylene.xyz` (optimize, finite-difference frequencies, sampling, excitations, selection of S1, dynamics, analysis), about 20 trajectories of 100 fs at `dt_fs = 0.5`, with one deliberate kill and resume during `dynamics`; run on the WSL native filesystem. The plan fixes the exact input before the run. Recorded in a new USER_MANUAL §5.9: the input, the wall time per stage, the outcome counts, the excluded fraction, the populations and `tau`/`t0` with intervals and flags. No bound is asserted on `tau`. Known risk: P7b saw a CASSCF solution change on one ethylene trajectory near 12 fs; trajectories may end with `energy_drift` and be excluded; the report states the excluded fraction as measured, and a large fraction becomes a P9 finding, not a tuning target.

## 14. Documents

CLAUDE.md and USER_MANUAL.md in the same branch: §8.13 becomes [IMPLEMENTED] with the formulas of sections 4-5; the `[analysis]` keys; the `07_analysis/` files; the reopen rule (§7.5 and the orchestrator rules); O10 resolved; the energy-jump owner decision recorded as resolved by a diagnostic; the P8 deferrals (staleness, common time) marked resolved; the M4 study; known limitations (no plots, one kinetic model, the jump measured between kept records, the inclusion rule's possible bias).
