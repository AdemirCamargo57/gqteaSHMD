# The censored analysis and the hop-step energy check: report

Status: **the `analysis` stage now treats a trajectory that ends early as censored data (the discrete-time Aalen-Johansen estimator, the new default `[analysis] estimator = "censored"`) and reports the P8 rule (`estimator = "complete"`) beside it. Under `hop_energy = "previous"` the energy-jump signal of the `dynamics` stage now sees the energy before the hop's velocity adjustment. On a synthetic Markov ensemble the estimator is unbiased within 4 standard errors and its interval covers the truth in 0.9625 of 400 repetitions (asserted). On the real ethylene data the measured back-hop rates speak against its assumption: the lost trajectories hop back up from S0 about six times less often than the complete ones (report). So the shorter lifetime it gives on the three real projects (59.7-62.6 fs against 75.2-80.1 fs) is not shown to be unbiased.** No bound, window, seed or setting was changed after a number was seen.

Date: 2026-11-19. Branch `feat/censored-analysis` (from `main` `f738b35`). The spec is `docs/superpowers/specs/2026-11-19-censored-analysis-design.md` (owner-approved 2026-11-19; erratum at its top), the plan is `docs/superpowers/plans/2026-11-19-censored-analysis.md` (plan commit `75676ad`; its "Plan rulings" PR1-PR10 are binding where they differ from the spec). The code tasks are the commits `bb0b1b0` (Task 1, the hop check), `1abe816` (Task 2, the estimator), `03d570f` (Task 3, censoring and the at-risk grid), `b387cf8` (Task 4, the input key and `analysis/estimators.py`), `6cb0348` (Task 5, the stage) and `75cdffb` (Task 6, the report script). Task 7 is a measurement with no diff.

**Sources of every number:** the task reports in the git-ignored ledger `.superpowers/sdd/2026-11-19-censored-analysis/` (`task-1-report.md` to `task-7-report.md`, `task-7-report.json`, `progress.md`); the JSON file `~/censoring/report.json` on the WSL native filesystem (copied as `task-7-report.json`); the projects `~/p9b/ours/run`, `~/hop/ours_previous/run` and `~/p9a/acceptance/run`.

**Labels.** *Asserted*: a pytest assertion. *Pre-registered verdict (report script)*: a rule fixed in the spec before the run, computed by the report script, not by pytest. *Report*: measured, with no rule. *Inference*: not established by a measurement.

## 1. What was built

- **The censoring rule** (spec 2.1; `analysis/censoring.py:select_censored`). Under `estimator = "censored"`:
  - `finished` and `stopped_in_s0` trajectories are complete, as before;
  - a trajectory with any other outcome and at least one record is **censored** at its last full-step record on the output grid;
  - with `exclude_energy_jumps = true`, a trajectory whose jump exceeds `energy_jump_eV` is censored at its last full-step record **before** the first such jump (also a `finished` one);
  - a trajectory with no record (a failed task, `electronic_failure` at t = 0) is excluded, as before;
  - a trajectory censored at or after the horizon counts as complete there (PR5);
  - the horizon is computed from the complete trajectories of the P8 rule and passed to both grids (PR7).
- **The estimator** (spec 2.2; `censoring.aalen_johansen`). On the full-step grid up to the horizon, a trajectory is at risk over an interval when it has records at both ends. Each interval's transition matrix is `P_k[a, b] = n_k(a -> b) / n_k(a)`, and `p(t_{k+1}) = p(t_k) P_k` from the observed initial distribution. A state with nobody at risk gets the identity row; `empty_rows` counts those rows that carry probability (PR2). With no censored trajectory the curve equals the plain fractions.
- **Intervals and fit.** The bootstrap resamples all trajectories that the estimator uses (censored ones included) and recomputes the curve (`bootstrap_aalen_johansen`); the 95 % intervals are its 2.5th and 97.5th percentiles (the classical curve and its interval are always finite, because a state with nobody at risk keeps its probability through the identity row; only the at-risk quantum interval can be NaN, at a time where a resample has nobody at risk). The delayed exponential of P8 is fitted to the censored curve's `fit_states` sum, and each resample's curve is refitted (`kinetics.bootstrap_fit_curves`). Streams (PR3): the complete estimator draws from `task_rng(seed, "analysis", 0)` as before, the censored one from `task_rng(seed, "analysis", 1)`.
- **The quantum curve** under `censored` is the average of `|c|^2` over the trajectories at risk at each time (`at_risk_average`), marked `quantum_rule: "at_risk_average"` in `analysis.json` only (not in the CSV or HDF5 files). It is biased towards the excited states when censoring happens in S0. `fit_population = "quantum"` with `estimator = "censored"` is an input error (exit 2).
- **Over the complete trajectories, as before (PR4):** hop statistics, energy drift and `consistency` (`null` when there is no complete trajectory).
- **The stage** (`stages/analysis.py`, `AnalysisStage.version` 3): both estimators are always computed; the one in use fills the top-level `n_included`, `included`, `excluded`, `small_ensemble`, `fit`, `table` and the files; `estimators` holds both blocks (PR4, ruling T5-4); `analysis.json` `schema_version` 2 adds `estimator`, `n_complete`, `n_censored`, `censored`, `estimators`. `populations.csv`/`.h5` add the other estimator's curves under the prefix `complete_` or `censored_` (PR6). `report.md` gains the estimator, `n_complete`, `n_censored`, a censored-trajectories table and an "Other estimator" section with its tau and t0; since the final fix wave also `empty_rows`, and the hop and energy sections are headed `Hops (complete trajectories)` and `Energy (complete trajectories)`.
- **Rulings made in the tasks** (ledger): **T5-1**, the data checks (a t = 0 record, finite data, the norm) run over every trajectory either estimator uses, also under `estimator = "complete"`, so a lost trajectory with a damaged `output.h5` now stops the stage where `main` excluded it unchecked (ruling F2 of the final fix wave keeps it as coded and documents it in full: the checks cover all records of such a trajectory, not only those it contributes, and under `"complete"` a censored one with a missing grid record before its censoring step also stops the analysis, `InconsistentData`); **T5-2**, the included-jump warning leaves out trajectories censored for an energy jump; **T5-3**, `n_complete` is the complete rule's count, which can differ from `n_included - n_censored` (a trajectory lost at or after the horizon counts as complete in the censored estimator, PR5); **T5-4**, the per-estimator blocks follow PR4 (`n`, `small_ensemble`, `fit`, `table`, plus `empty_rows` and `quantum_rule` for `censored`), not spec 2.5. C1 and C2 are layout and ordering rulings.
- **The hop check** (spec 3, PR10; `dynamics/records.py`, `dynamics/driver.py`, `trajectory/runner.py`). Under `hop_energy = "previous"`, an **accepted** hop's record carries `e_total_before_hop = E_diag'_lambda + K(v')` with the old active state and the Verlet-completed velocity before the adjustment. The runner's energy-jump signal compares that value, not the pinned `e_total`, with the previous record's `e_total`. A jump above `energy_jump_limit_eV` is rejected and halved, and persisting at the smallest step ends as `electronic_discontinuity`. The field is not stored (`steps.log`, `output.h5`, restart formats unchanged). Under `"current"` it stays `None` (bit-identical; the golden test is unchanged). **The drift check is unchanged**: it still uses `e_total`, so under `"previous"` it still sees nothing at an accepted hop step. `DynamicsStage.version` 3 makes **every** project with `dynamics` results stale (exit 3, `--rerun-from dynamics`), also under `"current"` (owner decision 5).
- **The report script** `tools/validation/censoring_report.py` (JSON out; one pre-registered verdict; never imported by `gqteashmd`). It calls `gqteashmd.analysis` on existing `output.h5` files and reads each project's settings from `.gqteashmd/config.json` (PR9).

## 2. Tests (asserted)

| Task | Command | Result |
|---|---|---|
| 1 | `pytest tests/dynamics tests/trajectory tests/stages/test_dynamics.py tests/stages/test_dynamics_pipeline.py -m "not slow and not pyscf" -W error` | **497 passed** in 160.73 s (the golden test green, unchanged) |
| 2 | `pytest tests/analysis -q -W error -m "not slow"`; `-m slow` on `test_censoring.py` | **42 passed**, 1 deselected (6.06 s); the slow coverage test **1 passed** (9.33 s) |
| 3 | `pytest tests/analysis tests/stages/test_analysis.py -q -W error -m "not slow"` | **72 passed**, 1 deselected (9.93 s) |
| 4 | `pytest tests/config tests/analysis tests/test_cli.py -q -W error -m "not slow and not pyscf"` | **190 passed**, 3 deselected (56.37 s) |
| 5 | `pytest tests/analysis tests/stages/test_analysis.py tests/stages/test_analysis_pipeline.py tests/config tests/test_orchestrator.py -q -W error -m "not pyscf"`; `pytest tests/test_e2e_dynamics.py -m "not slow"` | **254 passed** (550.39 s), then **87 passed** for `tests/stages/test_analysis.py tests/analysis` after the extra test; e2e **1 passed**, 1 deselected (33.91 s) |
| 6 | `pytest tests/validation/test_censoring_report.py -q -W error` | **5 passed** (1.76 s) |

**Full suite `pytest -W error` with SHARC and ORCA at `33242d3` (2026-11-19, the documents commit, before the final fix wave): 1561 passed, 2 skipped, 4 xfailed in 10533.17 s (2 h 55 min 33 s)** (the same two skips by design and four xfails as the hop energy option). The final fix wave (`5dd0e60` to `59cd773`) came after it and was run alone: `tests/analysis tests/stages/test_analysis.py tests/stages/test_analysis_pipeline.py tests/trajectory tests/validation/test_hop_energy_report.py tests/validation/test_censoring_report.py tests/config` (all markers) 315 passed; its re-review's non-slow run 135 passed.

What the tests assert, among others:

- **Reduction (spec 4.1 item 1):** with no censored trajectory, the censored curve equals the classical fractions (unit tests); on the canned pipeline the censored default equals `estimators.complete.table` within 1e-12, and the existing exact-equality test of the classical fractions also passes under the censored default.
- **The complete path is unchanged:** every existing stage test passes with `estimator = "complete"` pinned. Task 5 also compared the complete-rule arrays and fit against a copy of `main` on an 8-trajectory ensemble: bit-identical (report; an off-tree check, not a committed test). Since the final fix wave a committed test (T5b) pins the complete estimator's fit, intervals and populations of a stage run with losses, exactly, to the values produced on `main` `f738b35`.
- **The stage (spec 4.1 item 3):** the `estimator` validation, the quantum-fit refusal, both estimators in `analysis.json`, the censoring time per trajectory (also before the first jump with `exclude_energy_jumps`), the version 3, the warning texts, the fingerprint (the key is hashed), and a censored trajectory with records past the horizon counted as complete.
- **The hop check (spec 4.1 item 4):** a fake-backend runner run with a scripted energy jump (every energy + 0.01 Eh) in the call of the first hop step gives `rejections_by_signal == {"energy": 1}` under both rules (under `"current"` the record's `e_total` shows the jump, as on `main`; under `"previous"` only `e_total_before_hop` does); a `"previous"` run without faults has hops and no rejection; on Tully's simple avoided crossing with forced hops `e_total_before_hop` equals `E_lambda(q') + K` of the velocity passed to `adjust_velocity` to 1e-12 Eh at every accepted hop under `"previous"` and is `None` otherwise; the golden test is unchanged; `DynamicsStage.version` 3. **Since the final fix wave also at a hop step** (T1): energy jumps scripted at the hop step's call and at every halved retry end the trajectory as `electronic_discontinuity` with `rejections_by_signal == {"energy": 4}` (`max_step_halvings` 3), and a single one is rejected once and the run finishes.

## 3. The synthetic Markov ensemble (asserted; Task 2)

- **The model** (spec 4.1 item 2): 3 states, 100 steps on a 1 fs grid, a fixed transition matrix with decay S2 -> S1 -> S0 and a hop back up S0 -> S1 (probability 0.02 per step), all trajectories starting in S2; censoring only in S0, probability 0.02 per step; the truth is the matrix powers; N = 100, R = 400 repetitions, the test's own seed.
- **The rule** (PR1, fixed before any run): at steps 10, 20, ..., 100 and every state (30 comparisons), `|mean of the estimate - truth| <= 4 SE`. The spec's rule (3 SE at all 303 points) would fail by chance in most runs.
- **Measured:** the worst |z| of the censored estimator is **2.84** (passes). The two rules that must be able to fail do fail: still-running average **31.9**, complete-only **50.7**.
- **Coverage** (slow test): the bootstrap 95 % interval of P(S0) at step 50 (B = 500) covers the truth in **0.9625** of the 400 repetitions (band [0.90, 0.99]).
- Durations: the Markov test 0.97 s (fast tier), the coverage test 8.89 s (marked `slow`).
- **What this shows:** the estimator is unbiased, and its intervals cover, when its assumption holds (a Markov process, censoring that depends only on the current state).

## 4. Artificial censoring of the P9b ensemble (pre-registered verdict, report script; Task 7)

Run: `censoring_report.py --artificial ~/p9b/ours/run --project ~/hop/ours_previous/run --project ~/p9a/acceptance/run --out ~/censoring/report.json`, code at `75cdffb`, 72 s, exit 0. Defaults: R = 200, B = 200, seed 20261119.

- **Set-up** (spec 4.2 item 5): the 67 complete P9b trajectories (ethylene SA3-CASSCF(2,2)/6-31G\*, seed 2028, 100 fs). Each one that reaches S0 is censored with probability 33/90, at a delay after its first S0 arrival drawn from the 33 measured delays (4.0-82.625 fs). The truth is the uncensored 67: tau **80.136 fs**, t0 **12.835 fs**. The unit (PR8) is the censored estimate's bootstrap SE (B 200), averaged over R.

| Estimator | max bias / SE | max abs bias | tau bias (fs) | t0 bias (fs) |
|---|---|---|---|---|
| censored | **0.561** | 0.0373 | +7.417 | -1.035 |
| still-running | 0.914 | 0.0602 | +13.745 | -1.142 |
| complete-only | 1.084 | 0.0681 | +13.420 | +1.067 |
| hold-in-S0 | 1.152 | 0.0799 | -10.386 | +2.283 |

No fit failed in any repetition.

- **The pre-registered verdict is "good"** (the censored ratio 0.561 is <= 1 and below the complete-only 1.084 and the still-running 0.914).
- **What it does not test:** these losses are random once a trajectory is in S0, so they satisfy the estimator's assumption by construction.
- **Even here tau is biased** by +7.4 fs (9 % of 80.1 fs), about half the complete-only and still-running bias (report; tau has no verdict).

## 5. Sensitivity: the hottest S0 trajectories are lost (report; Task 7)

In each repetition the censored trajectories are the 33/90 fraction of the S0-reaching ones with the highest mean kinetic energy in S0 (spec 4.2 item 6).

| Estimator | max bias / SE | max abs bias | tau bias (fs) | t0 bias (fs) |
|---|---|---|---|---|
| censored | 1.244 | 0.0872 | +13.795 | -2.239 |
| still-running | 1.482 | 0.1032 | +20.286 | -2.350 |
| complete-only | 1.536 | 0.1070 | +23.245 | -0.885 |
| hold-in-S0 | **0.630** | 0.0435 | -8.272 | +1.515 |

When the losses depend on how hot the molecule is, the censored ratio is **above 1** (1.244), still below complete-only and still-running; hold-in-S0 is the least biased here.

## 6. Hops back up from S0 (report; Task 7)

| Set | hops | S0 time (fs) | rate (1/fs) | Poisson 95 % |
|---|---|---|---|---|
| the 33 lost, before their loss | 3 | 839.19 | 0.0036 | [0.00074, 0.0104] |
| the 57 complete that reach S0 | 49 | 2267.25 | 0.0216 | [0.0160, 0.0286] |

The intervals do not overlap: before their loss, the lost trajectories hop back up from S0 about six times less often than the complete ones. **So before their loss the lost trajectories did not behave like the complete ones in S0, and these rates speak against the estimator's assumption on this data** (a proxy: they describe the lost trajectories' past, not their unobserved future). The argument for the assumption from censoring in S0 only applies to these post-hop losses; censoring on an excited state (`energy_drift`, an early `electronic_failure`) relies on the same assumption without that argument. That this biases the real re-analysis below is an inference; the size of that bias is not measured.

## 7. Re-analysis of the three real projects (report; Task 7)

The horizon is 100 fs and `n_excluded` 0 in all three; `empty_rows` 0 in all three.

| Project | n_complete / n_censored | tau complete (fs) [95 %] | tau censored (fs) [95 %] | t0 complete / censored (fs) |
|---|---|---|---|---|
| P9b ours (`~/p9b/ours/run`) | 67 / 33 | 80.136 [57.62, 110.06] | **62.494** [47.24, 80.29] | 12.835 / 11.691 |
| ours with `"previous"` (`~/hop/ours_previous/run`) | 67 / 33 | 77.157 [54.91, 106.37] | **62.588** [47.03, 80.28] | 14.918 / 11.715 |
| P9a acceptance, seed 2027 (`~/p9a/acceptance/run`) | 13 / 7 | 75.200 [25.34, 135.30] | **59.743** [34.75, 97.62] | 11.224 / 11.582 |

Populations [S0, S1, S2]:

| Project | t (fs) | complete | censored |
|---|---|---|---|
| P9b ours | 50 | 0.3433, 0.4925, 0.1642 | 0.4671, 0.4220, 0.1109 |
| | 100 | 0.6119, 0.3134, 0.0746 | 0.6569, 0.2907, 0.0524 |
| `"previous"` | 50 | 0.3134, 0.5224, 0.1642 | 0.4527, 0.4362, 0.1111 |
| | 100 | 0.6418, 0.2836, 0.0746 | 0.6823, 0.2649, 0.0529 |
| acceptance | 50 | 0.3846, 0.6154, 0.0000 | 0.5000, 0.5000, 0.0000 |
| | 100 | 0.6923, 0.3077, 0.0000 | 0.7244, 0.2756, 0.0000 |

- The complete numbers equal the earlier reports (P9b, the hop energy report, the acceptance report).
- At 100 fs, P9b ours, the intervals are S0 [0.492, 0.720], S1 [0.215, 0.432], S2 [0.032, 0.163] (complete) and S0 [0.544, 0.764], S1 [0.187, 0.405], S2 [0.011, 0.104] (censored).
- In the acceptance project the censored S2 interval is [0, 0] at 50 and 100 fs (nobody in S2: a degenerate bootstrap).
- On all three, the censored estimator gives a higher P(S0) from 50 fs on, a shorter tau and a narrower tau interval. The direction matches the hypothesis that the complete rule loses decaying trajectories; the rates of section 6 speak against the censored estimator's own assumption too.
- **The `"previous"` project's trajectories were computed before the hop check** (dynamics version 2), so they are what the old energy-jump signal accepted. A rerun at version 3 could change outcomes; that was not measured.

## 8. Plan rulings PR1-PR10

- **PR1** the Markov rule: 30 points, 4 SE (section 3). **PR2** `empty_rows` counts only rows that carry probability. **PR3** streams 0 (complete) and 1 (censored). **PR4** the `analysis.json` layout; hops, drift and `consistency` over the complete trajectories. **PR5** censored at or after the horizon counts as complete. **PR6** the other estimator's curves in the files under a prefix. **PR7** the horizon from the complete trajectories. **PR8** the report's bias unit is the censored estimate's bootstrap SE. **PR9** the tool reads the project's settings. **PR10** `e_total_before_hop` only at an accepted hop (at a frustrated one the record's `e_total` already holds that value).

## 9. What was decided after numbers were seen

Nothing that changed a bound, window, seed or setting. PR1 was fixed before any run. The rulings C1, C2 and T5-1 to T5-4 were made while writing the code, not after a measurement. Task 7 was run once with the defaults; the first detached launch never started and was relaunched with the same command (process only).

## 10. Honest summary

- **Shown (asserted):** the estimator reduces to the P8 fractions with no censoring; on a synthetic Markov ensemble it is unbiased within 4 SE at 30 pre-registered points (worst 2.84), where the still-running and complete-only rules fail (31.9 and 50.7), and its 95 % interval covers in 0.9625 of 400 repetitions. The complete rule's numbers are unchanged. Under `"previous"` the energy-jump signal sees the energy before the adjustment at an accepted hop.
- **Shown (pre-registered verdict, report script):** with losses imposed at random once a trajectory is in S0, the censored estimator is the least biased of the three named estimators (0.561 SE against 1.084 and 0.914). Those losses satisfy its assumption by construction, and its tau is still biased by +7.4 fs.
- **Not shown, and partly contradicted (report):** that the censored lifetime is unbiased on the real data. The lost trajectories hop back up from S0 at 0.0036/fs against 0.0216/fs for the complete ones (Poisson intervals disjoint), which speaks against censoring that is independent of the future dynamics (the rates describe the lost trajectories' past). When the hottest S0 trajectories are the ones lost, the censored ratio is 1.244 (above 1), and hold-in-S0 does best (0.630). On all three real projects the censored tau is shorter (62.5 / 62.6 / 59.7 fs against 80.1 / 77.2 / 75.2 fs); which of the two numbers is closer to the truth is not known.
- **Not implemented (literature):** the back-hop rates indicate censoring that depends on the trajectory, beyond its current state. For that case the literature gives an inverse-probability-of-censoring-weighted (IPCW) correction (Datta and Satten 2002 [26], evaluated by Gunnes et al. 2007 [25], whose simulations with dependent censoring found the Aalen-Johansen estimator under- or overestimating the occupation probabilities and the IPCW estimator approximately unbiased). It is not implemented, and any remedy is the owner's decision.
- **Still blind under `"previous"`:** the drift check at an accepted hop step (unchanged by design).
- **Cost:** every project with `dynamics` results must be rerun from `dynamics` (version 3), also with the default `hop_energy = "current"`, whose numbers do not change.
- **Not measured:** the coverage of the small-ensemble rule (50) for censored ensembles; any other molecule, method or seed.

## 11. Final fix wave (2026-11-19)

After the final whole-branch review (ledger `final-review.md` and `final-fix-brief.md`; every command and count in `final-fix-report.md`). Branch `fix/censored-final`, from `33242d3`.

- **I1, bootstrap memory (fix).** `bootstrap_at_risk_average` contracts the (B, n) resample weights over the trajectories before expanding, so no (B, n, G) array is formed. `bootstrap_aalen_johansen` builds its counts one state pair at a time instead of from the dense (n, G-1, S, S) indicators. Peak memory (`tracemalloc`, n = 100, G = 401, B = 1000, S = 3; report): at-risk average **343.3 MB -> 24.6 MB**; Aalen-Johansen bootstrap 42.4 MB -> 39.6 MB (its (B, G-1, S, S) counts remain). The results equal the old ones: max difference 4.4e-16 for the at-risk average (same NaN positions), 0.0 for the Aalen-Johansen bootstrap. Asserted: equality to the dense forms within 1e-12 on smaller inputs, and two memory bounds.
- **I2, held probabilities (fix, ruling F1).** The stage warns when the censored estimate has `empty_rows > 0` (under either `estimator`). With no complete trajectory a second warning names the last time any used trajectory is observed, beyond which the censored curve is held constant. `report.md` prints `empty_rows`. The horizon is not capped and the stage does not stop; whether to do either is left to the owner.
- **m1.** `tools/validation/hop_energy_report.py` reads the complete rule from `estimators["complete"]` under `analysis.json` schema 2.
- **m2, m3 (documents only, ruling F2).** Section 1 now says that the checks cover all records of a censored trajectory, and that a missing grid record before its censoring step also stops a `"complete"` analysis.
- **m4.** The `report.md` hop and energy sections are headed `(complete trajectories)`.
- **m6.** Section 6 says that the S0-only argument for the assumption applies to the post-hop losses only.
- **Tests:**
  - T1: the `electronic_discontinuity` ending at a hop step (section 2);
  - T2: random bootstrap resamples;
  - T3: `censoring_step` with `output_every > 1`, substep records and the `n_total` clip;
  - T5a: a damaged censored trajectory stops a `"complete"` analysis;
  - T5b: the complete estimator pinned exactly to `main` `f738b35`.
- **Task 8 review:**
  - M1: the assumption wording now follows the strength of the evidence;
  - M2: section 1 says which intervals can be NaN;
  - M3: the quantum-fit refusal points at USER_MANUAL 8.13.3;
  - M4: a dated note in the hop energy report.
- **References:** verified by the controller (Consensus) and listed below. Volumes and pages were not checked.

## References

Numbered as in USER_MANUAL section 12.

- [20] A. Nießl, A. Allignol et al., "Statistical inference for state occupation and transition probabilities in non-Markov multi-state models subject to both random left-truncation and right-censoring", Econometrics and Statistics (2021), doi:10.1016/j.ecosta.2021.09.008. A rigorous proof that the Aalen-Johansen estimator of the state occupation probabilities is consistent without the Markov property when censoring is unrelated to the multi-state data.
- [21] T. Yan, R. B. Geskus, H. Putter, "Alternative representations of the Aalen-Johansen estimator of the state occupation probabilities in multi-state models", Statistica Neerlandica (2026), doi:10.1111/stan.70035. The equivalence to the IPCW empirical average.
- [22] R. Hoff, H. Putter, I. S. Mehlum, J. M. Gran, "Landmark estimation of transition probabilities in non-Markov multi-state models with covariates", Lifetime Data Analysis (2019), doi:10.1007/s10985-019-09474-0.
- [23] M. Overgaard, "State occupation probabilities in non-Markov models", Mathematical Methods of Statistics (2019), doi:10.3103/S1066530719040033. The consistency by another route, under a weak independent-censoring assumption.
- [24] S. Datta, G. A. Satten, "Validity of the Aalen-Johansen estimators of stage occupation probabilities and Nelson-Aalen estimators of integrated transition hazards for non-Markov models", Statistics & Probability Letters (2001), doi:10.1016/S0167-7152(01)00155-9. The original validity result; [20] and [23] report concerns about its martingale argument.
- [25] N. Gunnes, Ø. Borgan, O. O. Aalen, "Estimating stage occupation probabilities in non-Markov models", Lifetime Data Analysis (2007), doi:10.1007/s10985-007-9034-4. In their simulations with dependent censoring, the Aalen-Johansen estimator under- or overestimates the occupation probabilities, while the IPCW estimator of Datta and Satten is approximately unbiased.
- [26] S. Datta, G. A. Satten, "Estimation of integrated transition hazards and stage occupation probabilities for non-Markov systems under dependent censoring", Biometrics (2002), doi:10.1111/j.0006-341x.2002.00792.x. The IPCW estimator, with censoring weights from a covariate model.
