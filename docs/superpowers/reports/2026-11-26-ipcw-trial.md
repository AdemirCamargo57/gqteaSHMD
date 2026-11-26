# The IPCW trial and analysis stage version 4: report

Status: **by the pre-registered rule, IPCW is not built into the stage: `ipcw_go` is false.** Of its three parts, the control test passed (IPCW max bias / SE 0.265, rule <= 1) and the real-data gate passed (the loss hazard of the P9b ensemble rises with the S0 kinetic energy: beta 0.631 per SD, 95 % [0.199, 1.202]). The hot-S0 verdict failed: IPCW (1.189) is below the censored estimator (1.244) but not below hold-in-S0 (0.630). All three are pre-registered verdicts (report script). Whatever the trial found, the `analysis` stage is now at version 4: with no complete trajectory it caps its horizon at the last observed grid time, and it reports a third, hold-in-S0 lifetime beside the complete and censored ones, with the range of the three. Which of these lifetimes is closest to the truth is still not known. No bound, window, seed or setting was changed after a number was seen.

Date: 2026-11-26. Branch `feat/ipcw-trial-analysis-v4` (from `main` `43a25c7`). The spec is `docs/superpowers/specs/2026-11-26-ipcw-trial-analysis-v4-design.md` (owner-approved 2026-11-26; spec commit `a44f4e0`; erratum at its top). The plan is `docs/superpowers/plans/2026-11-26-ipcw-trial-analysis-v4.md` (plan commit `39213af`; its "Plan rulings" PR1-PR7 are binding where they differ from the spec). The code tasks are these commits:

- `94cd39b`: Task 1, `tools/validation/ipcw.py`;
- `cf00381` and its fix `aed305a`: Task 2, the trial in the report script;
- `62f74af`: Task 4, the horizon cap;
- `b10f6f7`: Task 5, hold-in-S0 and the range.

Task 3 is the measurement, with no diff.

**Sources of every number:**

- the task reports in the git-ignored ledger `.superpowers/sdd/2026-11-26-ipcw-trial-analysis-v4/` (`task-1-report.md` to `task-5-report.md`, `progress.md`);
- the JSON file `~/ipcw/report.json` on the WSL native filesystem, copied as `task-3-report.json`;
- the projects `~/p9b/ours/run`, `~/hop/ours_previous/run` and `~/p9a/acceptance/run`.

**Labels:**

- *Asserted*: a pytest assertion.
- *Pre-registered verdict (report script)*: a rule fixed in the spec (section 2.5) before the run, computed by `tools/validation/censoring_report.py`, not by pytest.
- *Report*: measured, with no rule.
- *Inference*: not established by a measurement.

## 1. What was built

### 1.1 The trial (tools only; nothing under `src/` changed for it)

- **`tools/validation/ipcw.py`** (new; NumPy and SciPy; never imported by `gqteashmd`). It holds `kinetic_on_grid`, `running_s0_mean`, `risk_and_events`, `fit_censoring_hazard`, `censoring_weights`, `ipcw_populations`, `ipcw_estimate` and `CensoringFit`.
  - **The covariate:** `K̄_i(k)`, the mean S0 kinetic energy of trajectory i over its S0 grid records up to step k. It is standardised over the at-risk S0 cells of the data being fitted.
  - **The censoring model** (spec 2.3): `logit h_i(k) = α_{a,k} + β z_i(k) 1{a = 0}`. The baseline has one intercept per grid step and state (nonparametric, owner decision 5).
    - For a given β, each intercept solves `Σ expit(α + β z_i) = d` (`brentq` on a bracket around `logit(d/r)`, PR7).
    - β maximises the profile likelihood, bounded to [-10, 10] per SD.
    - Fit failures: no S0 event; the optimum on a bound; or no informative S0 interval (PR3). Each is reported as `fit_failed`, never a crash.
  - **The estimator** (spec 2.4): weights `w_i(k+1) = w_i(k) / (1 - ĥ_i(k))` while i is observed. Populations `P_s(k) = (1/n) Σ_i 1{observed, active = s} w_i(k)`. They are not normalised; the largest row-sum deviation and the largest weight are reported.
- **`tools/validation/censoring_report.py`** gains:
  - `ipcw` as a fifth estimator (`estimates`; the other four unchanged, `four_estimates` kept);
  - the control test (`control`), the real-data gate (`gate`) and the go decision (`ipcw_go`);
  - IPCW in the re-analysis of the real projects (`_ipcw_block`): tau, t0, their intervals, populations, β with its interval, the largest weight;
  - a shared repetition helper (`_repetition`) for the artificial and control tests.
- **Random streams:** `seed` for the artificial test, `seed + 1` for the sensitivity test, `seed + 2` for the control, `seed + 3` for the gate and `seed + 4` for the IPCW re-analysis bootstrap (PR2). The default seed is 20261119. IPCW draws no random numbers inside the existing repetition loops, so the four existing estimators keep their 2026-11-19 numbers (section 3).

### 1.2 Analysis stage version 4 (built whatever the trial found)

- **The horizon cap** (spec 3.1; Task 4, `stages/analysis.py`). The cap applies when no trajectory is complete under the P8 rule and at least one is censored.
  - The horizon is then the largest censoring step among the used trajectories, the last grid step at which any is observed, instead of $n_{\text{tot}}$.
  - The PR5 filter of the censored analysis runs against this horizon, so the trajectories observed through it count as complete there.
  - A cap at step 0 stops the stage: `stage_failed`, with the error `no trajectory is complete and none is observed after t = 0` (`NO_DATA_AFTER_START`). `finalize` then raises.
  - The old warning (`… the censored curve beyond t fs … is held constant`) is replaced by `no trajectory is complete: the analysis horizon is capped at {t:g} fs, the last time any trajectory is observed`.
  - `analysis.json` gains the top-level `horizon_capped` (bool).
  - With at least one complete trajectory nothing changes.
- **The hold-in-S0 estimate** (spec 3.2, PR6; Task 5, `analysis/estimators.py`, `stages/analysis.py`). It is **reported only**: `[analysis] estimator` keeps its two values, and no input key changed.
  - `estimators.hold_in_s0_grid(g)` takes the censored estimator's grid. From its censoring step on, a censored trajectory is in S0 (classical), with quantum populations `(1, 0, ...)`. A trajectory lost in an excited state is held in S0 too.
  - `estimators.hold_in_s0_estimate` runs the complete arithmetic on that grid (`complete_estimate(..., name="hold_in_s0")`): Wilson intervals, the quantum bootstrap and `bootstrap_fit`, with `fit_population` honoured.
  - It draws from the stream `task_rng(seed, "analysis", 2)`. Streams 0 (complete) and 1 (censored) are unchanged.
  - The block `estimators.hold_in_s0` holds `n`, `small_ensemble`, `fit`, `table` and `rule: "lost trajectories held in S0 from their censoring step"`. It is `null` when no trajectory is censored.
- **The lifetime range** (spec 3.3; Task 5).
  - `analysis.json` gains the top-level `tau_range_fs`: `{min, max, min_estimator, max_estimator}` over the blocks that exist and have a fitted tau. It is `null` with fewer than two.
  - `report.md` gains a section `## Lifetime range`: one line per estimator, the range, and `Which estimate is closest to the truth is not known (USER_MANUAL 8.19).`
  - `populations.csv` gains the columns `hold_in_s0_P_<L>`, `_lower`, `_upper` and `hold_in_s0_rho_…`. `populations.h5` gains the datasets `hold_in_s0_classical`, …, `hold_in_s0_quantum`, …. Both are left out when the block is `null`.
- **Versions** (PR1, set in Task 4): `AnalysisStage.version` 4 and `analysis.json` `schema_version` 3. **Every project with `analysis` results is stale** (exit 3, `--rerun-from analysis`, seconds; the trajectories stay). The complete and censored numbers do not change.

## 2. Tests (asserted)

| Task | Command | Result |
|---|---|---|
| 1 | `pytest tests/validation/test_ipcw.py -q -W error` | **8 passed** (3.16 s); with `OPENBLAS_CORETYPE=SandyBridge` 8 passed |
| 2 (and its fix) | `pytest tests/validation/test_censoring_report.py tests/validation/test_ipcw.py -q -W error` | **18 passed**; `test_censoring_report.py` with `OPENBLAS_CORETYPE=SandyBridge` 10 passed |
| 4 | `pytest tests/stages/test_analysis.py tests/stages/test_analysis_pipeline.py tests/analysis -q -W error -m "not slow"` | 111 passed, 1 failed (the pipeline's `schema_version` assertion, 2 -> 3, fixed; re-run passed); `tests/stages/test_analysis.py` **42 passed**; the three new tests with `OPENBLAS_CORETYPE=SandyBridge` 3 passed; `tests/validation/test_hop_energy_report.py` + `test_censoring_report.py` 15 passed |
| 5 | `pytest tests/analysis tests/stages/test_analysis.py tests/stages/test_analysis_pipeline.py -m "not pyscf" -W error` | **119 passed**; `test_estimators.py`, `test_report.py`, `test_analysis.py` with `OPENBLAS_CORETYPE=SandyBridge` 61 passed; the controller's re-run of `tests/config tests/test_orchestrator.py -m "not pyscf"` at `b10f6f7` **158 passed** (383.83 s) |
| 6 | `pytest tests/site -q -W error` (after the document edits); the fast tier `pytest -m "not slow and not pyscf" -W error -q` (on `/mnt/c`, code at `b10f6f7`; started before the document edits, which change only files the site tests read) | site **21 passed** (3.15 s); fast tier **1418 passed, 22 skipped, 224 deselected in 1033.22 s** (17 min 13 s) |

What the tests assert, among others:

- **The IPCW module** (spec 4.1 items 1-7, `tests/validation/test_ipcw.py`):
  - with a marginal per-step hazard, the IPCW populations equal the still-running average to 1e-12 (the identity of spec 1.1);
  - without the covariate, the hazard is `d/r` per cell and the populations equal a hand computation;
  - on a 3-state Markov model with state-dependent censoring (R = 100), the β = 0 estimate is within 4 SE of the truth at steps 10, 20, …, 60 and every state;
  - the fit recovers a known β within 4 bootstrap SEs;
  - without censoring, the populations are the plain fractions;
  - `running_s0_mean` on hand-made records;
  - the three `fit_failed` cases.
- **The report script** (spec 4.1 item 8, PR5; `tests/validation/test_censoring_report.py`):
  - on the in-memory synthetic project `tests/validation/censoring_synthetic.py`, the four existing estimators' numbers equal those the unmodified script wrote to `tests/validation/data/censoring_report_before.json` (relative 1e-9 for bias ratios and curves, 1e-6 for tau and t0);
  - the re-analysis gains `ipcw` (with its β interval) and is otherwise unchanged;
  - `control` and `gate` report their rules;
  - `ipcw_go` needs all three parts.
- **Version 4** (`tests/stages/test_analysis.py`, `tests/analysis/test_estimators.py`, `tests/analysis/test_report.py`):
  - without a complete trajectory: the horizon is capped (`horizon_capped: true`, the new warning);
  - a cap at 0 stops the stage with the recorded error;
  - with a complete trajectory: no cap;
  - the hold-in-S0 grid and estimate equal the complete arithmetic on the held grid;
  - the hold-in-S0 block and `tau_range_fs` in a stage run; no hold-in-S0 block without a censored trajectory;
  - the `## Lifetime range` section of `report.md`;
  - `AnalysisStage.version == 4` and `schema_version == 3`.
- **Streams 0 and 1 unchanged:** the pin of the complete estimator to `main` `f738b35` (T5b of the censored analysis) and the agreement of the two estimators without losses stay green.

## 3. The measurement (Task 3)

Run once, on WSL, with the code at `aed305a` and the script's defaults (R = 200, B = 200, seed 20261119):

`python tools/validation/censoring_report.py --artificial ~/p9b/ours/run --project ~/hop/ours_previous/run --project ~/p9a/acceptance/run --out ~/ipcw/report.json`

- It started at 15:19:04 and wrote `report.json` at 15:22:23, so it took about 3 min 20 s (from the file times).
- stdout: `{"verdict_good": true, "ipcw_go": false}`.

**Reproduction check (report):** the four existing estimators reproduce `~/censoring/report.json` of 2026-11-19 exactly (difference 0.0). This holds for `max_bias_over_se` and `tau_bias_fs` in the artificial and sensitivity tests, and for the complete and censored tau of the three re-analyses. So the addition moved nothing else.

**The data:** the 67 complete trajectories of the P9b ensemble (ethylene SA3-CASSCF(2,2)/6-31G\*, seed 2028, 100 fs) for the control and the verdict, and all 100 for the gate. The truth of the control and the verdict is the uncensored 67: tau 80.136 fs, t0 12.835 fs. The bias unit is the censored estimate's bootstrap SE, averaged over the repetitions (PR8 of the censored analysis). `max_bias_over_se` is the largest |mean - truth| / unit over every grid time and state.

### 3.1 The artificial test of 2026-11-19, with IPCW added (report for IPCW)

The 2026-11-19 verdict (censored <= 1 and below complete-only and still-running) is again "good". IPCW is reported beside it, with no rule.

| Estimator | max bias / SE | max abs bias | tau bias (fs) | t0 bias (fs) |
|---|---|---|---|---|
| censored | 0.5612 | 0.0373 | +7.42 | -1.04 |
| still-running | 0.9140 | 0.0602 | +13.74 | -1.14 |
| complete-only | 1.0839 | 0.0681 | +13.42 | +1.07 |
| hold-in-S0 | 1.1516 | 0.0799 | -10.39 | +2.28 |
| IPCW | 0.2009 | 0.0124 | +0.76 | -0.01 |

No fit failed. These artificial losses are random by construction, and the fitted β (0.035 per SD on average) agrees.

## 4. The control test (pre-registered verdict, report script): PASSED

**Set-up** (spec 2.5 item 1; PR4 for the denominator). At every grid step at which one of the 67 trajectories is in S0, it is censored with the true hazard `logit h = a + 1.0 · z(K̄)`. Here z is standardised over all S0 cells of the 67. The intercept `a = -5.2229` was solved with `brentq` on [-30, 10] (`xtol` 1e-12), from the data alone and before any estimator, so that the expected lost fraction of the S0-reaching trajectories is 33/90 = 0.36667. The realised mean lost fraction was 0.3624. The stream is `seed + 2`.

**Rule:** the IPCW `max_bias_over_se` is <= 1, with no fit failure.

| Estimator | max bias / SE | max abs bias | tau bias (fs) | t0 bias (fs) |
|---|---|---|---|---|
| **IPCW** | **0.2648** | 0.0165 | +0.34 | -0.03 |
| censored | 1.1695 | 0.0857 | +16.55 | -2.62 |
| still-running | 1.8667 | 0.1362 | +33.38 | -3.23 |
| complete-only | 2.4777 | 0.1625 | +43.69 | +0.76 |
| hold-in-S0 | 0.6038 | 0.0447 | -4.56 | +0.92 |

- IPCW passes (0.265; 0 fit failures; fitted β mean 1.127 per SD against the true 1.0).
- The test discriminates (report): the censored, still-running and complete-only estimators exceed 1 SE under this censoring.
- **What it shows:** when censoring depends on the kinetic energy already observed at the time of loss and the fitted censoring model contains the true hazard (correctly specified), the IPCW code removed the bias (one run). A failure here would have been a bug, not a "no" (owner decision 1).

## 5. The hot-S0 verdict (pre-registered verdict, report script): FAILED

**Set-up** (the 2026-11-19 sensitivity test, unchanged). In every repetition the censored set is the same 33/90 fraction of the S0-reaching trajectories: those with the highest mean kinetic energy over **all** their S0 records, including the records after the loss. Each is lost at a delay after its first S0 arrival, drawn from the 33 measured delays. The stream is `seed + 1`.

**Rule:** the IPCW `max_bias_over_se` is strictly below both the censored and the hold-in-S0 values of the same run, with no fit failure.

| Estimator | max bias / SE | max abs bias | tau bias (fs) | t0 bias (fs) |
|---|---|---|---|---|
| censored | 1.2440 | 0.0872 | +13.80 | -2.24 |
| still-running | 1.4821 | 0.1032 | +20.29 | -2.35 |
| complete-only | 1.5356 | 0.1070 | +23.25 | -0.89 |
| **hold-in-S0** | **0.6301** | 0.0435 | -8.27 | +1.52 |
| IPCW | 1.1894 | 0.0829 | +5.51 | -0.99 |

- IPCW is below the censored estimator (1.189 < 1.244) but **not below hold-in-S0** (0.630). So `ipcw_wins` is false.
- There were 0 fit failures. The fitted β averaged 1.306 per SD: the model sees that the hot trajectories are lost, but the correction is small.
- **Report, no rule:** IPCW's tau bias (+5.51 fs) is the smallest in absolute value here (hold-in-S0 -8.27 fs, censored +13.80 fs). The verdict is on the population bias ratio, as fixed in the spec, not on tau.
- **Why IPCW does little here is an inference:** the test selects on the S0 kinetic energy over the whole S0 time, after the loss too. That information is not in the observed past at the time of loss, which is all a censoring model with a running mean can use. This explanation was not tested.

## 6. The real-data gate (pre-registered verdict, report script): PASSED

**Set-up** (spec 2.5 item 3). The censoring model is fitted on the re-analysis grid of `~/p9b/ours/run`: all 100 used trajectories, with their real censoring steps. The interval comes from B = 1000 resamples of whole trajectories (`populations.bootstrap_indices`, stream `seed + 3`), with the model refitted in each.

**Rule:** the fit on the data succeeds, at least 950 resamples succeed, and the 2.5 % bound of β is > 0.

- The fit succeeded: β = **0.6306 per SD** (0.5796 per eV; the SD of the covariate is 1.088 eV), from 33 S0 censoring events in 29 informative cells.
- 95 % percentile interval: **[0.1989, 1.2023] per SD** ([0.1860, 1.1188] per eV, from the resamples' own per-eV values).
- 1000 of 1000 resamples succeeded.
- **What it shows:** in the P9b ensemble, trajectories with a hotter S0 history are lost more often. The interval excludes 0 on the positive side. The censored estimator does not model this dependence.

**Go:** `ipcw_go = control and verdict and gate = true and false and true = false`.

## 7. IPCW on the three real projects (report)

The censoring model is refitted in every resample, with B = the project's `n_bootstrap` (1000) and stream `seed + 4` (PR2). The complete and censored values are those of the 2026-11-19 report, reproduced exactly.

| Project | tau complete (fs) [95 %] | tau censored (fs) [95 %] | tau IPCW (fs) [95 %] | t0 complete / censored / IPCW (fs) |
|---|---|---|---|---|
| P9b ours (`~/p9b/ours/run`) | 80.14 [57.62, 110.06] | 62.49 [47.24, 80.29] | **55.97** [42.47, 70.15] | 12.84 / 11.69 / 13.29 |
| ours with `"previous"` (`~/hop/ours_previous/run`) | 77.16 [54.91, 106.37] | 62.59 [47.03, 80.28] | **55.41** [41.97, 69.66] | 14.92 / 11.71 / 13.59 |
| P9a acceptance (`~/p9a/acceptance/run`) | 75.20 [25.34, 135.30] | 59.74 [34.75, 97.62] | **58.84** [35.49, 111.62] | 11.22 / 11.58 / 11.75 |

| Project | β per SD [95 %] | β per eV | censoring fits failed / resamples | largest weight | largest row-sum deviation |
|---|---|---|---|---|---|
| P9b ours | 0.631 [0.197, 1.183] | 0.580 | 0 / 1000 | 6.24 | 0.034 |
| `"previous"` | 0.551 [0.105, 1.069] | 0.511 | 0 / 1000 | 6.31 | 0.042 |
| acceptance | 0.955 [0.078, 3.539] | 0.783 | 17 / 1000 | 6.76 | 0.148 |

- The IPCW populations [S0, S1, S2] at 100 fs are not normalised, so the rows need not sum to 1:
  - P9b ours: 0.751, 0.232, 0.050 (sum 1.034);
  - `"previous"`: 0.785, 0.206, 0.050;
  - acceptance: 0.873, 0.275, 0.000.
- For comparison, P9b ours at 100 fs: P(S0) complete 0.612, censored 0.657 (2026-11-19 report).
- The fitted β is positive on all three projects. The β interval of P9b ours here ([0.197, 1.183], stream `seed + 4`) differs slightly from the gate's ([0.199, 1.202], stream `seed + 3`): same data, other resamples.
- The acceptance project has 6 S0 events in 5 informative cells. Its censoring fit failed in 17 of 1000 resamples, and its rows deviate by up to 0.148 from 1, so its IPCW numbers are poorly determined.
- On the two 100-trajectory projects, IPCW gives the shortest tau of the three estimators and the highest P(S0) at 100 fs.
- **No rule applies here.** These numbers do not decide anything; the rule of section 5 decided.

**The version 4 range and the hold-in-S0 block on the real projects were not computed.** They would need `--rerun-from analysis` on copies of the three projects, which was not done. The report script's own hold-in-S0 estimator exists only in the artificial tests (sections 3-5).

## 8. Plan rulings PR1-PR7

- **PR1:** the version numbers (`AnalysisStage.version` 4, `schema_version` 3) are set in Task 4, not Task 5.
- **PR2:** the IPCW re-analysis bootstrap uses the stream `default_rng(seed + 4)`.
- **PR3:** a censoring fit with no informative S0 interval is `fit_failed`, reason `no informative S0 interval`.
- **PR4:** the control's "S0-reaching" uses `first_s0_time(t) is not None` on all records.
- **PR5:** test 4.1 item 8 runs on an in-memory synthetic project, compared with the unmodified script's numbers to a relative 1e-9 / 1e-6.
- **PR6:** `complete_estimate(name=)`, `hold_in_s0_grid` and `hold_in_s0_estimate`.
- **PR7:** the `brentq` bracket of the intercept solve.

Rulings made in the tasks (ledger):

- **The commit trailer:** Task 1's commit carries the plan's Opus trailer though Sonnet wrote it. It was left as is, with no history rewrite.
- **Task 2's review added two things, both fixed in `aed305a` before the measurement ran:**
  - β intervals in the IPCW re-analysis (spec 2.6 required them; the plan had left them out);
  - the gate's per-eV interval, now taken from the resamples' own per-eV values instead of from the full-data SD.

  It also replaced a duplicated repetition loop by one helper that keeps the order of the random draws; the stored-JSON test guards that. None of this touches a rule.
- **Task 3 was run by the controller, not a subagent**, in parallel with Task 4. The running script had loaded its modules at start, so the `src/` edits of Task 4 could not affect it.

## 9. What was decided after numbers were seen

Nothing that changed a bound, window, seed or setting. The rules of spec 2.5 and the plan rulings were fixed before the run, and Task 3 was run once with the defaults. Process-only facts:

- The first launch of the measurement failed before Python started (`/usr/bin/time` is not installed in WSL). It was relaunched with the same command.
- The duration (about 3 min 20 s) comes from the file times, because the timer echo of the relaunch printed empty variables (a quoting error).
- The two Task 2 review additions (section 8) were made before the measurement ran.

By the pre-registered rule the answer is **no**: IPCW is not built into the stage. The control and the gate both pass, and the verdict fails only because hold-in-S0 is better on the hot-S0 test. Revisiting the rule after seeing these numbers would be a new owner decision. This report recommends neither way.

## 10. Honest summary

- **Shown (asserted):**
  - the IPCW module reduces to the still-running average with marginal weights and to the plain fractions without censoring;
  - it is unbiased on a Markov model with state-dependent censoring;
  - its fit recovers a known β;
  - the four earlier estimators of the report script are unchanged;
  - version 4 caps the horizon without a complete trajectory and reports a hold-in-S0 lifetime and the range, with the complete and censored numbers unchanged.
- **Shown (pre-registered verdicts, report script):**
  - *the gate:* on the P9b ensemble the real losses depend on the observed S0 kinetic energy; hotter trajectories are lost more often (β 0.63 per SD, 95 % [0.20, 1.20]);
  - *the control:* where the fitted censoring model contains the true hazard (correctly specified), IPCW removed the bias (0.265, where the censored estimator has 1.17 and hold-in-S0 0.60; one run);
  - *the verdict:* on the hot-S0 test, IPCW improves on the censored estimator only slightly (1.189 against 1.244), and hold-in-S0 stays best (0.630). That test's censoring selects on the whole S0 time, also after the loss.
- **Report:** IPCW's tau on the three real projects is 55.97 / 55.41 / 58.84 fs. The censored tau is 62.49 / 62.59 / 59.74 fs and the complete tau 80.14 / 77.16 / 75.20 fs. Hold-in-S0 on the real projects was not computed.
- **Inference, not measured:** whether the real losses look more like the control (the loss depends on the observed past) or like the hot-S0 test (it depends on what happens after the loss). Hence it is still not known which of the lifetimes is closest to the truth.
- **Decision by rule:** `ipcw_go` is false, so IPCW is not built into the stage. The stage reports the complete, censored and hold-in-S0 lifetimes as a range (`tau_range_fs`), with the statement that the closest one is not known.
- **Cost:** every project with `analysis` results must be rerun from `analysis` (version 4; seconds; the trajectories stay).
- **Not measured:** other covariates, molecules, methods or seeds; weight truncation or normalisation (out of scope, spec 6); the coverage of IPCW intervals.

## References

Numbered as in USER_MANUAL section 12.

- [25] N. Gunnes, Ø. Borgan, O. O. Aalen, "Estimating stage occupation probabilities in non-Markov models", Lifetime Data Analysis (2007), doi:10.1007/s10985-007-9034-4.
- [26] S. Datta, G. A. Satten, "Estimation of integrated transition hazards and stage occupation probabilities for non-Markov systems under dependent censoring", Biometrics (2002), doi:10.1111/j.0006-341x.2002.00792.x.
