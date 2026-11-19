# The censored analysis and the hop-step energy check: design

Status: written 2026-11-19; owner-approved 2026-11-19. Branch `feat/censored-analysis` (from `main` `f738b35`).

> **Erratum (2026-11-19, after the implementation).** The plan's rulings (`docs/superpowers/plans/2026-11-19-censored-analysis.md`, "Plan rulings") and the task rulings of the ledger change these details. The measured numbers are in `docs/superpowers/reports/2026-11-19-censored-analysis.md`.
>
> - **4.1 item 2 (PR1):** the Markov rule compares at steps 10, 20, ..., 100 and every state (30 comparisons) with the bound **4 SE**, not 3 SE at every time and state (303 comparisons would fail by chance in most runs). The "must be able to fail" check uses the same points and bound. Fixed before any run. Measured: worst |z| censored 2.84, still-running 31.9, complete-only 50.7; coverage 0.9625.
> - **2.2 (PR2):** `empty_rows` counts only an interval and state with nobody at risk **and** a positive estimated population at its start.
> - **2.2 (PR3):** the censored estimator's bootstrap draws from `task_rng(seed, "analysis", 1)`, not from stream 0; the complete estimator keeps stream 0, so its numbers are unchanged.
> - **2.5 (PR4, ruling T5-4):** the top-level `n_included`, `included`, `excluded`, `small_ensemble`, `fit` and `table` belong to the estimator in use; `estimators` holds a block per estimator with `n`, `small_ensemble`, `fit` and `table` (plus `empty_rows` and `quantum_rule` for `censored`). The blocks carry **no** per-time populations with intervals and no `n_excluded`; the full curves with intervals of both estimators are in `populations.csv`/`.h5` (PR6), and `excluded` is top-level only. `n_complete`, `n_censored`, `censored` and `estimator` are always computed. Hops, `energy.drift_eV` and `consistency` are over the complete trajectories; `consistency` is `null` without a complete trajectory.
> - **2.1 (PR5):** a trajectory whose censoring grid step is at or after the horizon counts as complete in the censored estimator and is not in `censored` or `n_censored`. So `n_complete` (the complete rule's count, ruling T5-3) can differ from `n_included - n_censored`.
> - **2.5 (PR6):** the other estimator's columns are named `complete_P_S1`, `complete_P_S1_lower`, `complete_P_S1_upper`, ..., `complete_rho_...` (CSV) and `complete_classical`, `complete_classical_lower`, ..., `complete_quantum_...` (HDF5), or with the prefix `censored_`; they are left out when that estimator has no trajectory. Its fit target and curve are not in the files.
> - **2.1 (PR7):** the horizon is computed from the complete trajectories of the P8 rule and passed to both grids.
> - **3 (PR10):** `e_total_before_hop` is set only at an **accepted** hop, not at a frustrated one: there the record's `e_total` already equals $E_{\text{diag}}'_\lambda + K(\mathbf v')$.
> - **2.1, the checks (ruling T5-1):** the data checks (a t = 0 record, finite data, the norm) run over every trajectory either estimator uses, also under `estimator = "complete"`. A lost trajectory with a damaged `output.h5` therefore stops the stage under `"complete"`, where `main` excluded it unchecked.
> - **2.5, the included-jump warning (ruling T5-2):** it counts the flagged trajectories among those the estimator in use uses, minus those censored for an energy jump.
> - **2.4:** the files do not mark the censored quantum curve as the at-risk average; only `analysis.json` (`quantum_rule: "at_risk_average"`) and the documents say so.
> - **4.1 item 4, the hop check as tested:** the runner test scripts one energy jump in the call of the first hop step and asserts one `energy` rejection under both rules; it does not assert the `electronic_discontinuity` ending at the smallest step at a hop step, nor that the run finishes after the halved retry (a deferred minor of Task 1). The ending itself is the F1 path of P9a plan 2.
> - **What was decided after numbers were seen:** nothing that changed a bound, window, seed or setting.

## 0. Owner decisions (2026-11-19, multiple choice)

These follow the owner decision of 2026-11-18 that the losses after a hop are the next sub-project (hop energy option spec, section 0, item 4).

1. **Goal:** an analysis that the losses do not bias. A lost trajectory is treated as censored data. The dynamics are not changed to keep trajectories alive.
2. **Estimator:** the discrete-time Aalen-Johansen estimator of the classical state populations. The other two rules are reported alongside: hold in S0 for the sensitivity study, and the still-running average.
3. **Default:** `[analysis] estimator = "censored"` is the default. The current rule stays as `estimator = "complete"`, and `analysis.json` reports both.
4. **The hop-step energy check** (the open item I1 of the hop energy option) is part of this sub-project: under `hop_energy = "previous"` the energy-jump signal sees the energy before the velocity adjustment.
5. **Sections 1-3 of the design** were approved as presented, including `DynamicsStage.version` 3, which makes every project with `dynamics` results stale.

## 1. Why

### 1.1 What was measured during the design

These numbers come from throwaway scripts on `~/p9b/ours/run`: the P9b ensemble, ethylene SA3-CASSCF(2,2)/6-31G\*, seed 2028, 100 trajectories of 100 fs. They are facts the rules below rely on, **seen before the rules were fixed**.

- **The losses:** 33 of 100 end early (23 `overlap_failure`, 10 `electronic_discontinuity`). **All 33 end on S0, after their first arrival in S0.** The delay from the first S0 arrival to the loss is 4.00-82.62 fs, median 22.00 fs (24 of 33 later than 10 fs).
- **S0 is not absorbing:** 90 of 100 trajectories reach S0, and 33 of those 90 later leave it again (a hop back up). Of the 57 `finished` trajectories that reached S0, 15 end on S1.
- **The mechanism** (P9a, the acceptance report section 3.3): a second SA-CASSCF solution in the hot ground state, which the run ends on before it disappears. SHARC_PYSCF makes the same kind of switch (P9a plan 1, P9b direct). It is a property of the method, not something our code gets wrong.

### 1.2 Consequence

The current inclusion rule (P8, O10 item 2) averages only over the `finished` and `stopped_in_s0` trajectories. It therefore drops exactly the trajectories that relaxed, so the populations and the lifetime are biased towards trajectories that stay excited longer. The only report of this today is the excluded-fraction warning (33 % in P9b).

### 1.3 The estimator and its assumption

The Aalen-Johansen estimator generalises Kaplan-Meier to state occupation probabilities in a multi-state process with right censoring (Nießl et al., Econometrics and Statistics 2021; Yan et al., Statistica Neerlandica 2026, which also shows its equivalence to an inverse-probability-of-censoring-weighted average).

- **Without the Markov property**, it is consistent for occupation probabilities when censoring is independent of the states occupied and of the transition times (Datta and Satten 2001, as stated by Hoff et al., Lifetime Data Analysis 2019, and Overgaard, Mathematical Methods of Statistics 2019).
- **Our case:** our censoring happens only in S0, so we rely on the weaker condition. Among the trajectories in S0 at time t, the future transitions of the lost ones are like those of the ones that go on. This holds for a Markov process when censoring depends only on the current state.
- **It may fail:** surface hopping is not Markov in the active state, and a hotter S0 may be both more likely to fail and more likely to hop back up.
- **What is done about it:** the assumption is stated in the documents, and its consequences are measured (section 4.2), not assumed away.

## 2. The censored analysis

### 2.1 Which trajectories, and how

| Ending | `complete` (today) | `censored` (new default) |
|---|---|---|
| `finished` | included | included, complete |
| `stopped_in_s0` | included, held in S0 after the stop | the same |
| any other outcome with at least one record (`overlap_failure`, `electronic_discontinuity`, `energy_drift`, `electronic_failure` after t = 0) | excluded | **censored** at its last full-step record on the grid |
| no record (a failed task, `electronic_failure` at t = 0) | excluded | excluded |
| `exclude_energy_jumps = true` and a jump above `energy_jump_eV` | excluded | **censored** at its last full-step record **before** the first such jump |

- The horizon rule is unchanged: the shortest included `finished` trajectory sets it (`grid.horizon_steps`).
- A censored trajectory whose last full-step record lies after the horizon counts as complete up to the horizon.
- The checks on included trajectories (a record at t = 0, finite data, the norm) apply to censored ones too, over the records they contribute.

### 2.2 The estimator (classical populations)

- **The grid:** times `t_0 = 0 < t_1 < ... < t_H` (the full-step grid of P8, up to the horizon).
- **At risk:** a trajectory is at risk over `[t_k, t_{k+1}]` if it has full-step records at both ends. A censored trajectory leaves the risk set from the interval in which its data ends; a complete one stays to the horizon.
- **The transition matrix of an interval:** `P_k[a, b] = n_k(a -> b) / n_k(a)`, where `n_k(a)` is the number of trajectories at risk in state a at `t_k`, and `n_k(a -> b)` those of them in state b at `t_{k+1}`.
  - If `n_k(a) = 0`, the row is the identity row (no information, no transition). Every such interval and state is counted, and the count is reported (`empty_rows`).
- **The populations:** `p(t_0)` is the observed initial distribution, `p(t_{k+1}) = p(t_k) P_k`.
- **Reduction:** with no censored trajectory, `p(t_k)` equals today's classical fractions exactly (up to round-off from the products; the test bound is 1e-12).
- **Intervals:** per-time 95 % intervals by the existing bootstrap. Resample all trajectories with records, censored ones included, with the indices of `populations.bootstrap_indices` from the existing stream `task_rng(seed, "analysis", 0)`, and recompute the curve. The percentile intervals are the existing `bootstrap_intervals`.
  - The Wilson intervals stay a `complete`-only quantity, because they assume binomial counts.

### 2.3 The fit

- The delayed exponential of P8 is fitted to `sum over fit_states of p_k(t)` from the censored curve (default: every excited state, `1 - p_S0`).
- The bootstrap fit refits each resample's censored curve. `bootstrap_fit` takes the per-resample curves instead of per-trajectory arrays (a new entry point; the `complete` path is unchanged).

### 2.4 Quantum populations

- The estimator does not apply to amplitudes. Under `censored`, the quantum curve is the average of `|c|^2` over the trajectories at risk at each time, marked in `analysis.json` (`quantum_rule: "at_risk_average"`) and in the documents as biased towards the excited states when censoring happens in S0.
- `fit_population = "quantum"` with `estimator = "censored"` is a `ConfigError` (exit 2) that names `estimator = "complete"`.
- The classical-against-quantum `consistency` number stays a `complete`-only quantity.

### 2.5 Input, outputs, warnings

- **Input:** `[analysis] estimator = "censored" | "complete"`, default `"censored"`, hashed (it changes the results). `AnalysisStage.version` 2 -> 3: a project with `analysis` results needs `--rerun-from analysis`. The `init` template's commented `[analysis]` block gets the key.
- **`analysis.json`:** a top-level `estimator` (the one in use) and, for **both** estimators:
  - `populations` (the classical and the quantum curves with their intervals);
  - the `fit` with its bootstrap;
  - `n_complete`, `n_censored`, `n_excluded`;
  - `small_ensemble`.

  The estimator in use comes first and feeds `report.md` and the warnings. The `censored` block also carries `censored` (per trajectory: task, outcome, censoring time) and `empty_rows`. `schema_version` 2.
- **`populations.csv` and `populations.h5`:** the columns or datasets of the estimator in use, plus the other estimator's under a prefix (`complete_` or `censored_`).
- **Small ensemble:** under `censored`, N = complete + censored, with the existing `SMALL_ENSEMBLE` = 50 and its coverage text (P9b). The coverage was measured for complete ensembles only, which the warning says.
- **Excluded or censored fraction:** the existing `warn_excluded_fraction` warning keeps its key and threshold. Under `censored` it is about trajectories that are not complete (censored plus excluded), and its text names the estimator's assumption (section 1.3) instead of "possible selection bias". Under `complete` it is unchanged.
- **Unchanged:** the hop statistics, the energy diagnostics, the excluded list.

## 3. The hop-step energy check (`hop_energy = "previous"`)

- **The gap** (hop energy option report, final review I1): under `"previous"`, an accepted hop's record has `E_tot` equal to the previous record's. The runner's energy-jump signal (F1) therefore sees 0 at every hop step, whatever the energies did.
- **Driver (`dynamics/driver.py`, `dynamics/records.py`):** a new `StepRecord.e_total_before_hop: float | None = None`.
  - **When it is set:** only under `hop_energy = "previous"` at a hop attempt, accepted or frustrated.
  - **Its value:** `E_diag'_lambda + K(v')` with the old active state `lambda` and the Verlet-completed velocity `v'` before the adjustment. That is the value `"current"` would record for an accepted hop.
  - **Under `"current"`:** it stays `None`, so the default is bit-identical (the golden test is unchanged and nothing is regenerated).
- **Runner (`trajectory/runner.py:_energy_jump`):** uses `record.e_total_before_hop` when it is not `None`, otherwise `record.e_total`, against `Progress.last_e_total_au` as today.
  - A jump above `energy_jump_limit_au` is rejected and halved by the existing path.
  - At the smallest step it ends as `electronic_discontinuity`.
  - `last_e_total_au` is still set from the accepted record's `e_total`.
- **Unchanged:**
  - the record's stored `e_total`;
  - `steps.log` (`log_dtype`) and `output.h5`, since the field is not stored;
  - the restart formats;
  - the drift check (still on the recorded `E_tot`; under `"previous"` it leaves out each hop step's error, as documented);
  - the step order and the arithmetic.
- **Version:** `DynamicsStage.version` 2 -> 3, because the results of a `"previous"` project can change. **Every project with `dynamics` results becomes stale** (exit 3, `--rerun-from dynamics`). That is the approved cost (owner decision 5), chosen over a hash change for `"previous"` projects only.

## 4. Validation (every rule fixed here, before any run)

### 4.1 Asserted (fast tier unless marked)

1. **Reduction:** with no censored trajectory, the censored curve equals the classical fractions within 1e-12 at every time, on the P8 synthetic ensembles and on the canned pipeline (`canned_analysis`).
2. **A synthetic Markov ensemble with a known truth:**
   - **The model:** 3 states, discrete steps on a 1 fs grid of 100 steps, a fixed transition matrix with decay S2 -> S1 -> S0 and a hop back up S0 -> S1 (probability 0.02 per step), all trajectories starting in S2.
   - **The censoring:** only in S0, with probability 0.02 per step.
   - **The truth:** the matrix powers, exactly.
   - **The run:** N = 100, R = 400 repetitions, from the test's own fixed seed.
   - **Pass:** at every time and state, `|mean over R of the censored estimate - truth| <= 3 SE` (SE = std over R / sqrt(R)).
   - **Must be able to fail:** the still-running average and the complete-only rule (here: never censored) must each exceed 3 SE at some time and state.
   - **Coverage:** the bootstrap 95 % interval of P(S0) at step 50 (B = 500) covers the truth in a fraction in [0.90, 0.99] of the R repetitions, the P8 band.
   - It may be marked `slow` if it takes more than 60 s.
3. **The stage:** `estimator` validation; the quantum-fit refusal (exit 2); both estimators in `analysis.json`; the censoring time per trajectory (the last full-step record, before the first jump with `exclude_energy_jumps`); `AnalysisStage.version` 3; the warning texts; the fingerprint (the key is hashed); a censored trajectory with records past the horizon counts as complete there.
4. **The hop check:**
   - **Under `"previous"`:** a fake-backend runner run with a scripted `energy_jump` fault at a hop step is rejected and halved, and at the smallest step persisting it ends `electronic_discontinuity`.
   - **Under `"current"`:** the same script behaves as on `main`.
   - **The value:** a unit test computes `e_total_before_hop` independently at a forced hop.
   - **The default:** `"current"` leaves the field `None`, and the golden test is unchanged.
   - **The version:** `DynamicsStage.version` 3.

### 4.2 Report (`tools/validation/censoring_report.py`; JSON out; never imported by `gqteashmd`)

The report calls `gqteashmd.analysis` on existing `output.h5` files directly. It does not run the stage: with `DynamicsStage.version` 3 the existing projects are stale, and re-running their dynamics is not needed for this.

5. **Artificial censoring of the 67 complete P9b trajectories** (`~/p9b/ours/run`):
   - **Which trajectories:** each one that reaches S0 is censored with probability 33/90.
   - **When:** at a delay after its first S0 arrival, drawn uniformly from the 33 measured delays of section 1.1; a time beyond 100 fs means no censoring.
   - **Repetitions:** R = 200, from a fixed seed.
   - **The truth:** the uncensored 67 (classical fractions and their fit).
   - **Measured, per estimator** (censored, complete-only, still-running, hold-in-S0): the largest absolute mean bias of each state's population over time, divided by the mean bootstrap SE at that time (B = 200); the mean bias of tau and t0.
   - **Verdict "good", fixed here:** the censored estimator's largest ratio is <= 1 and smaller than the complete-only and still-running ratios.
   - **What it does not test:** the imposed losses are random once a trajectory is in S0, so they satisfy the estimator's assumption by construction.
6. **Sensitivity** (no verdict): the same, but in each repetition the censored trajectories are the 33/90 fraction of S0-reaching trajectories with the highest mean kinetic energy in S0 (losses that depend on how hot the molecule is). Also, the observed rate of hops back up from S0 per fs of S0 time, in the 33 lost trajectories before their loss and in the 57 complete ones that reach S0, each with a Poisson 95 % interval.
7. **Re-analysis** (no verdict): P9b ours (`~/p9b/ours/run`), ours with `"previous"` (`~/hop/ours_previous/run`) and the P9a seed-2027 acceptance project, under both estimators. Reported: the populations at 25, 50, 75 and 100 fs, tau and t0 with intervals, `n_complete`, `n_censored` and `empty_rows`.

Nothing is re-tuned on these numbers. If an asserted rule fails, it is reported, not adjusted (the P9b ruling P5 precedent).

### 4.3 The full suite

The full suite (`pytest -W error`, with SHARC and ORCA) runs before the final review, as usual.

## 5. Components

- `src/gqteashmd/analysis/censoring.py` (new; NumPy only):
  - `censoring_time` per trajectory;
  - `aalen_johansen(active_on_grid, at_risk, n_states) -> (populations, empty_rows)`;
  - `at_risk_average`;
  - the per-resample curves for the bootstrap.
- `analysis/grid.py`: `select` returns complete, censored and excluded trajectories; `on_grid` produces an at-risk mask. The `complete` path stays bit-identical.
- `analysis/kinetics.py`: a bootstrap entry point that takes per-resample curves.
- `stages/analysis.py`, `analysis/report.py`, `analysis/artifacts.py`: both estimators, the outputs and warnings of section 2.5, version 3.
- `config/schema.py`: `AnalysisSection.estimator` and the cross-field refusal. `cli.py`: the `init` template.
- `dynamics/records.py`, `dynamics/driver.py`: `e_total_before_hop`. `trajectory/runner.py`: `_energy_jump`. `stages/dynamics.py`: version 3.
- Tests: `tests/analysis/` (new `test_censoring.py`, the Markov ensemble), `tests/stages/test_analysis.py`, `tests/stages/test_analysis_pipeline.py`, `tests/config/test_analysis_section.py`, `tests/dynamics/` and `tests/trajectory/test_runner.py` (the hop check), `tests/stages/test_dynamics.py` (version).
- `tools/validation/censoring_report.py` and a test of its helpers.
- **Dependency rules unchanged:** `analysis/` imports only `core/`, `gqteashmd.__version__` and itself; SciPy and h5py only inside functions.

## 6. Documents

- **`USER_MANUAL.md`:**
  - the `estimator` key (section 4);
  - the analysis method section (8.13): the estimator, its assumption, the quantum rule;
  - the outputs (6);
  - the version bumps and their reruns;
  - the hop check (8.12, 8.18);
  - a new subsection in 8 with every number marked asserted or report;
  - the limitations (10).
- **`CLAUDE.md`:** "Read first", a module block, status, the known limitations (1a, 1b, 1c), remaining work.
- **A short report** `docs/superpowers/reports/<date>-censored-analysis.md` with the numbers of 4.2.
