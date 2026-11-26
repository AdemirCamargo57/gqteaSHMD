# The IPCW trial and analysis stage version 4: design

Status: written 2026-11-26; owner-approved 2026-11-26. Branch `feat/ipcw-trial-analysis-v4` (from `main` `43a25c7`). The report is `docs/superpowers/reports/2026-11-26-ipcw-trial.md`.

> **Erratum (2026-11-26, after the implementation).** Where the work differs from this spec:
>
> - **Plan rulings** (`docs/superpowers/plans/2026-11-26-ipcw-trial-analysis-v4.md`, binding where they differ):
>   - **PR1:** `AnalysisStage.version` 4 and `schema_version` 3 are set in Task 4 (the horizon cap), not Task 5.
>   - **PR2:** the IPCW bootstrap of section 2.6 draws from `default_rng(seed + 4)` (the spec gave no stream).
>   - **PR3:** a third failure reason of section 2.3, `no informative S0 interval` (every S0 event cell has `d = r`, so β is not identified).
>   - **PR4:** "S0-reaching" in the control's lost fraction is `first_s0_time(t) is not None` on all records.
>   - **PR5:** "a small canned project" of test 4.1 item 8 is an in-memory synthetic project. It is compared with the unmodified script's numbers to a relative 1e-9 (bias ratios, curves) and 1e-6 (tau, t0), not bit for bit.
>   - **PR6:** `complete_estimate(name=)`, `hold_in_s0_grid` and `hold_in_s0_estimate`.
>   - **PR7:** the intercept solve uses `brentq` on `[logit(d/r) - s, logit(d/r) + s]`, `s = max|β z_i| + 1`, instead of a vectorised bisection.
> - **Task 2 review additions** (fix commit `aed305a`, before the measurement ran):
>   - the IPCW re-analysis reports the β interval of section 2.6, which the plan had left out;
>   - the gate's per-eV interval comes from the resamples' own per-eV values, not from the full-data SD.
>
>   Neither touches a rule.
> - **The report script keeps `four_estimates`** beside the new `estimates`; section 2.1 said it would be renamed.
> - **Nothing was decided after numbers were seen that changed a rule**, bound, window, seed or setting. The measurement ran once (its first launch failed before Python started and was relaunched with the same command). The result: control passed, verdict failed, gate passed, so `ipcw_go` is false.

This is the "next analysis change" parked on 2026-11-20 (CLAUDE.md, "Remaining work"): (1) with no complete trajectory, cap the analysis horizon at the last observed grid time; (2) a trial of an inverse-probability-of-censoring-weighted (IPCW) estimator with one covariate (the S0 kinetic energy) on the hot-S0 sensitivity test, implemented in the stage only if it beats the censored and hold-in-S0 estimators there; until then the complete, censored and hold-in-S0 lifetimes are reported as a range.

## 0. Owner decisions (2026-11-26, multiple choice and approved sections)

1. **The verdict test:** the existing hot-S0 sensitivity test of `tools/validation/censoring_report.py`, unchanged, decides. A second test whose censoring hazard depends only on the observed past is added as a **correctness control**: an IPCW failure there is a bug, not a "no".
2. **The real data is part of the gate:** IPCW is built into the stage only if, in addition, the loss hazard of the real P9b ensemble depends on the S0 kinetic energy (the 95 % interval of the coefficient excludes 0, on the positive side).
3. **Scope:** this spec covers the trial (tools only) and `AnalysisStage.version` 4 (the horizon cap plus a hold-in-S0 estimate, so the stage reports the three lifetimes as a range). If the trial says go, IPCW in the stage gets its own spec, designed with the trial's numbers.
4. **The censoring model:** a discrete-time logistic hazard, approach A of the three offered (A logistic hazard; B Kaplan-Meier within covariate bins; C a Cox model), with the running mean of the S0 kinetic energy as the covariate and unnormalised weights (the `1/n` form).
5. **Revision of 4 after a check during the design (section 1.1):** the baseline of the hazard is nonparametric, one intercept per grid step and state, not one constant intercept per state.
6. **Sections 1-3 of the design** (the trial, version 4, the order of work) were approved as presented, with the revision of item 5.

## 1. Why

The censored analysis report (`docs/superpowers/reports/2026-11-19-censored-analysis.md`) left three facts:

- On the real ethylene data the lost trajectories hop back up from S0 at 0.0036/fs (95 % [0.00074, 0.0104]) before their loss, the complete ones at 0.0216/fs ([0.0160, 0.0286]): the censoring depends on the trajectory beyond its current state, against the Aalen-Johansen estimator's assumption.
- When the hottest S0 trajectories are the ones lost (the sensitivity test), the censored estimator's max bias is 1.244 SE and hold-in-S0 is the least biased (0.630); still-running 1.482, complete-only 1.536.
- On the three real projects the censored tau (62.5 / 62.6 / 59.7 fs) is shorter than the complete one (80.1 / 77.2 / 75.2 fs); which is closer to the truth is not known. Stopping in S0 does not settle it either (USER_MANUAL 5.10: the lifetime of the P9b ensemble lies between about 47 and 80 fs depending only on how the S0 phase after the hop is treated).

For censoring that depends on an observed covariate, the literature gives the IPCW estimator (Datta and Satten 2002 [26]; in the simulations of Gunnes et al. 2007 [25] with dependent censoring it was approximately unbiased where Aalen-Johansen was not). Whether it helps here depends on two things this trial measures: whether it corrects censoring that depends on the S0 kinetic energy, and whether the real losses depend on it.

### 1.1 Measured during the design (throwaway script, seen before the rules were fixed)

A 3-state Markov chain (transition matrix `[[0.97, 0.02, 0.01], [0.05, 0.90, 0.05], [0.0, 0.08, 0.92]]`, start S2, 60 steps, n = 300), censoring with a hazard that depends on the current state (S0 0.06 per step, otherwise 0.01), 100 repetitions, seed 7; truth from the matrix powers:

| Estimator | max \|mean - truth\| | max \|z\| | mean SD |
|---|---|---|---|
| Aalen-Johansen | 0.0084 | 2.68 | 0.0323 |
| IPCW, hazard `d/r` per grid step and state | 0.0094 | 2.58 | 0.0369 |
| the same, weights normalised per time | 0.0097 | 2.49 | 0.0358 |
| still-running average | 0.1593 | 42.26 | 0.0336 |

And, in one sample (seed 1): IPCW with **marginal** Kaplan-Meier censoring weights equals the still-running average to **4.8e-15**, and differs from Aalen-Johansen by 0.186.

Consequences for this design:

- On the analysis grid a trajectory leaves the risk set only by censoring, so marginal censoring weights reduce exactly to the still-running average. The identity "IPCW with no covariate = Aalen-Johansen" does **not** hold in that form and is not used as a test.
- IPCW with a censoring hazard per grid step and state is unbiased under state-dependent censoring but not identical to Aalen-Johansen.
- A baseline intercept constant in time would be misspecified on the hot-S0 test, whose losses happen at "first S0 arrival + a delay drawn from the measured 4-82.6 fs"; hence the nonparametric baseline (owner decision 5).

## 2. The trial (tools only; nothing under `src/` changes in this part)

### 2.1 Where the code lives

- `tools/validation/ipcw.py` (new; NumPy and SciPy; never imported by `gqteashmd`): `running_s0_mean`, `fit_censoring_hazard`, `censoring_weights`, `ipcw_populations`, `CensoringFit`.
- `tools/validation/censoring_report.py` imports it and adds `ipcw` as a fifth estimator to `four_estimates` (renamed `estimates`), the control test, the real-data gate and the go decision.
- The IPCW estimator draws **no** random numbers inside the existing repetition loops, so the four existing estimators reproduce the 2026-11-19 numbers bit for bit (artificial: censored 0.561, still-running 0.914, complete-only 1.084, hold-in-S0 1.152; sensitivity: 1.244, 1.482, 1.536, 0.630). That equality is the check that nothing else moved.
- If the trial says go, the module moves to `src/gqteashmd/analysis/` under the later spec.

### 2.2 The data on the grid

Everything is on the estimator's grid `g` (`grid.on_grid`, full-step records at `grid_step = m * output_every`, up to the horizon h), as in the report script today.

- **Risk set and events.** Trajectory i is at risk over the interval [k, k+1] (k < h) when it is observed at k. It is **censored in that interval** when it is observed at k and not at k+1. A trajectory observed at h has no event.
- **The covariate.** `K̄_i(k)` = the mean of `e_kinetic` over the grid records of trajectory i at grid steps <= k with `active == 0` (its S0 records so far, since its first S0 arrival). It is defined when `active_i(k) == 0`; it is used only then. `e_kinetic` is read from `output.h5` at the records the grid uses.
- **Standardisation.** `z = (K̄ - m) / sd`, m and sd over the at-risk cells (i, k) with `active_i(k) == 0` of the data being fitted. β is reported per SD and, with sd, per eV.

### 2.3 The censoring model

`logit h_i(k) = α_{a,k} + β · z_i(k) · 1{a = 0}`, with `a = active_i(k)`.

- **Baseline:** one intercept per (grid step k, state a), nonparametric. A cell with no censoring event gets ĥ = 0 exactly (α = -∞); a cell in which every trajectory at risk is censored gets ĥ = 1.
- **Fit:** maximum likelihood over the at-risk cells. For a given β each intercept of a cell with at least one event solves `Σ_{i in cell} expit(α + β z_i) = d_cell` (monotone in α; bisection, vectorised over cells); for a ≠ 0 this is `logit(d/r)`. β maximises the profile log-likelihood (`scipy.optimize.minimize_scalar`, bounded to [-10, 10] per SD).
- **Failure:** no censoring event in S0, or the optimum on a bound of [-10, 10], is `fit_failed` (never a crash).
- **With β = 0** every intercept is `logit(d/r)`: the per-step, per-state hazard of section 1.1.

### 2.4 The estimator

- Weights: `w_i(0) = 1`, `w_i(k+1) = w_i(k) / (1 - ĥ_i(k))` while i is observed at k+1; `w_i(k) = 0` where i is not observed.
- Populations: `P_s(k) = (1/n) Σ_i 1{observed_i(k), active_i(k) = s} · w_i(k)` (not normalised; rows need not sum to 1; the largest row-sum deviation and the largest weight are reported).
- The fit target is the `fit_states` sum `Σ_{s >= 1} P_s`, fitted with `fit_delayed_exponential` (the other estimators of the report script keep `1 - P_S0`, unchanged).

### 2.5 The three results (rules fixed here, before any run)

All three use `~/p9b/ours/run` (the 67 complete trajectories for 1 and 2, all 100 for 3), the report script's defaults R = 200, B = 200, seed 20261119, and its bias unit (PR8 of the censored analysis: the censored estimate's bootstrap SE averaged over the repetitions). `max_bias_over_se` is the existing `max_bias_ratio` over every grid time and state.

1. **Control test (correctness).** On the 67 complete trajectories, censoring is drawn at every grid step k at which a trajectory is in S0, with the true hazard `logit h = a + 1.0 · z(K̄)`; z uses the standardisation of section 2.2 over all S0 cells of the 67 (fixed, computed once). `a` is set by bisection so that the expected lost fraction of the S0-reaching trajectories, `1 - mean_i Π_k (1 - h_i(k))`, is 33/90 (computed from the data alone, before any estimator and without random numbers). Random stream `default_rng(seed + 2)` (the artificial test uses `seed`, the sensitivity test `seed + 1`). **Rule: the IPCW `max_bias_over_se` is <= 1.** The other four are reported. A fit failure in any repetition fails the control (the number of failures is reported).
2. **Hot-S0 verdict.** The existing sensitivity test (`artificial(..., hot=True)`), unchanged. **Rule: the IPCW `max_bias_over_se` is strictly below both the censored and the hold-in-S0 values of the same run.** A fit failure in any repetition fails the verdict.
3. **Real-data gate.** On the reanalysis grid of `~/p9b/ours/run` (all used trajectories, the real censoring steps), fit the censoring model. Its β has a 95 % percentile interval from B = 1000 resamples of whole trajectories (`populations.bootstrap_indices` with `default_rng(seed + 3)`, the model refitted in each). **Rule: the fit on the data succeeds, at least 950 resamples succeed, and the 2.5 % bound of β is > 0** (hotter trajectories are lost more often, the hypothesis behind the hot-S0 test).

**Go:** `ipcw_go = control and verdict and gate`, written to the JSON with each part's numbers whichever way it falls, and printed on stdout with the existing `verdict_good`.

### 2.6 Reported, with no rule

- IPCW on the three real projects (`~/p9b/ours/run`, `~/hop/ours_previous/run`, `~/p9a/acceptance/run`): tau and t0 with bootstrap intervals (B = the project's `n_bootstrap`; the censoring model refitted in every resample; `kinetics.bootstrap_fit_curves`), populations at 25/50/75/100 fs, β per SD and per eV, the largest weight, beside the complete and censored values of the existing `reanalysis`.
- In the control test, the four other estimators' ratios (whether the test discriminates).
- β, its interval and the fit status on all three projects (the gate uses the first only).

## 3. Analysis stage version 4 (built whatever the trial finds)

### 3.1 The horizon cap

- When **no trajectory is complete** (the P8 rule's set is empty), the horizon is the largest censoring step among the used trajectories, the last grid step at which any is observed, instead of `n_total`. The censoring steps are filtered by PR5 against this horizon, so a trajectory observed through it counts as complete there (as today for a censoring step at or after the horizon).
- If that step is 0 (no trajectory observed after t = 0) the stage stops like the existing "no selected trajectory has a record" case: recorded in `analysis.json` (`stage_failed`, `error` = `no trajectory is complete and none is observed after t = 0`), then `finalize` raises `StageFailedError`.
- The warning `no trajectory is complete: the censored curve beyond {t:g} fs, the last time any trajectory is observed, is held constant` is replaced by `no trajectory is complete: the analysis horizon is capped at {t:g} fs, the last time any trajectory is observed`.
- `analysis.json` gains the top-level `horizon_capped` (bool).
- With at least one complete trajectory nothing changes (the horizon from the complete trajectories, PR7).

### 3.2 The hold-in-S0 estimate

- A third block `estimators.hold_in_s0`, **reported only**: `[analysis] estimator` keeps its two values and no input key changes.
- **Rule:** the trajectories the censored estimator uses, on its grid; from the step after its censoring step on, a censored trajectory counts as in S0, also one lost in an excited state (the report script's definition). Its quantum populations there are `(1, 0, ...)`.
- **Arithmetic:** the complete estimator's, on that held grid (`estimators.complete_estimate` with a name argument, so `Estimate.name` is `"hold_in_s0"`): Wilson intervals, the quantum bootstrap, `bootstrap_fit`, `fit_population` honoured.
- **Random stream:** `task_rng(seed, "analysis", 2)`. Streams 0 (complete) and 1 (censored) are unchanged, so the complete and censored numbers equal version 3's.
- **Block contents:** `n`, `small_ensemble` (`n < SMALL_ENSEMBLE`), `fit`, `table`, `rule: "lost trajectories held in S0 from their censoring step"`. It is `null` when no trajectory is censored (it would equal the complete block) and when the censored estimator has no trajectory.

### 3.3 The lifetime range

- Top-level `tau_range_fs` = `{min, max, min_estimator, max_estimator}` over the blocks that exist and have a fitted tau (`complete`, `censored`, `hold_in_s0`); `null` with fewer than two.
- `report.md` gains a section `## Lifetime range`: one line per estimator (tau, t0 with their intervals), the range, and `Which estimate is closest to the truth is not known (USER_MANUAL 8.19).`
- `populations.csv` gains `hold_in_s0_P_<state>`, `_lower`, `_upper` and `hold_in_s0_rho_...` columns, `populations.h5` the datasets `hold_in_s0_classical`, `..._lower`, `..._upper`, `hold_in_s0_quantum`, ... (as PR6 adds the other estimator's curves); left out when the block is `null`.

### 3.4 Unchanged, and the cost

- Unchanged: the inclusion rules; the complete and censored numbers; hops, drift and `consistency` (over the complete trajectories); the fingerprint keys; the input.
- `analysis.json` `schema_version` 2 -> 3 (fields added).
- **`AnalysisStage.version` 3 -> 4: every project with `analysis` results is stale** (exit 3, `--rerun-from analysis`, seconds; the trajectories stay).

## 4. Tests

### 4.1 The trial (`tests/validation/test_ipcw.py`, fast tier, asserted)

1. **Weight arithmetic:** with a marginal per-step hazard (`d/r` over all states) the IPCW populations equal `censoring.at_risk_average` to 1e-12 (the identity of section 1.1).
2. **β = 0:** with the covariate removed the fitted intercepts equal `logit(d/r)` per cell and the populations equal a hand computation with `d/r` weights to 1e-12.
3. **Unbiasedness:** on the Markov model of section 1.1 (its own seed, R = 100), the IPCW estimate with β = 0 is within 4 SE of the truth at steps 10, 20, ..., 60 and every state (the PR1 form).
4. **Recovery:** on synthetic data with a known β (S0 censoring `logit h = a + β z`), the fit recovers β within 4 of its bootstrap SEs.
5. **No censoring:** the IPCW populations equal the plain fractions exactly.
6. `running_s0_mean` on hand-made records (before the first S0 arrival undefined; a back hop excluded from the mean while in S1).
7. `fit_failed` without an S0 event and with β on a bound.
8. The report script: on a small canned project the four existing estimators' numbers are unchanged by the addition (compared with the values of the script before the change), and the JSON has `control`, `gate`, `ipcw_go`.

### 4.2 Version 4 (`tests/stages/test_analysis.py`, `tests/analysis/`)

- No complete trajectory: the horizon is the largest censoring step, the new warning, `horizon_capped: true`; a cap of 0 stops the stage with the recorded error.
- The hold-in-S0 block equals a direct recomputation (held grid, Wilson intervals, fit); `null` without a censored trajectory.
- Streams 0 and 1 unchanged: the existing pin test T5b stays green and the censored block equals version 3's on the canned pipeline.
- `tau_range_fs`, the `report.md` section, the CSV/HDF5 columns.
- `AnalysisStage.version == 4`, `schema_version == 3`.
- The rule of CLAUDE.md: no bit-for-bit BLAS comparison in a new test; new tests checked with `OPENBLAS_CORETYPE=SandyBridge`.

## 5. Order of work (one commit per task)

1. `tools/validation/ipcw.py` and tests 4.1 items 1-7 (item 1 first).
2. `censoring_report.py`: the fifth estimator, the control test, the gate, `ipcw_go`; test 4.1 item 8.
3. **Measurement** (no diff): one run on WSL, `censoring_report.py --artificial ~/p9b/ours/run --project ~/hop/ours_previous/run --project ~/p9a/acceptance/run --out ~/ipcw/report.json`, with the fixed rules, run once. A rule found wrong after a number is seen is changed only by an owner ruling, recorded in the report's "decided after numbers were seen".
4. Version 4, part 1: the horizon cap.
5. Version 4, part 2: hold-in-S0, `tau_range_fs`, the files, `report.md`, the version and schema numbers.
6. Documents: the report `docs/superpowers/reports/<date>-ipcw-trial.md` (every number marked asserted, pre-registered verdict or report); USER_MANUAL 6, 8.13, 8.19, the limitations and section 11; CLAUDE.md (the seeds rule gains stream 2; status; remaining work: the parked items closed or replaced by the IPCW-in-the-stage spec); CHANGELOG `Unreleased`.

Tasks 4 and 5 do not depend on the trial's outcome.

**After the trial:** with `ipcw_go` true, a separate spec designs IPCW in the stage (an `estimator` value, its bootstrap, its warnings). With `ipcw_go` false, the report says which part failed, the parked IPCW item is closed in CLAUDE.md, and the three-estimator range of version 4 stays the answer.

## 6. Not in scope

IPCW in the stage; covariates other than the S0 kinetic energy; other molecules, methods or seeds; rerunning any dynamics; weight truncation or normalisation; changing the default `estimator`.

## 7. References

Numbered as in USER_MANUAL section 12 (verified in the censored analysis, its report's references).

- [25] N. Gunnes, Ø. Borgan, O. O. Aalen, "Estimating stage occupation probabilities in non-Markov models", Lifetime Data Analysis (2007), doi:10.1007/s10985-007-9034-4.
- [26] S. Datta, G. A. Satten, "Estimation of integrated transition hazards and stage occupation probabilities for non-Markov systems under dependent censoring", Biometrics (2002), doi:10.1111/j.0006-341x.2002.00792.x.
