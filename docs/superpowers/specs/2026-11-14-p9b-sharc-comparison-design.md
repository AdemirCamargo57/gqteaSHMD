# P9b: the SHARC comparison campaign (design)

> **Erratum (2026-11-18, written with the report `docs/superpowers/reports/2026-11-18-p9b-sharc-comparison.md`).** Where the work differs from this design:
>
> **Plan rulings** (`docs/superpowers/plans/2026-11-14-p9b-sharc-comparison.md`, binding where they differ from this spec):
>
> - **PR1:** the SHARC reader produces `analysis.grid.Trajectory`, not `TrajectoryData` (7.2).
> - **PR2:** ours-nacdr is a copy of the seed-2028 project rerun from `dynamics` with `coupling = "nacdr"`, not a `run_variants.py` run (3.1). `run_variants.py` gives no reference geometry, so the start rule would not apply there. The copy path was taken: `gqteashmd status` worked on the copy, and the two `trajectories.json` files are identical.
> - **PR3:** the ensembles keep `substep_interpolation = "midpoint"`; only the replay uses `"endpoint"`.
> - **PR4:** SHARC's frustrated hops are reported as `null`. Task 1 found them only in `output.lis` (`# Jump frustrated.`), never in `output.dat`.
> - **PR5:** a fit without an estimate or std, or a KS sample below 2, is `inconclusive` with the reason. It did not occur.
> - **PR6:** the comparison grid is every 1 fs, with `dt_out` 1.0 fs.
> - **PR7:** "exact" classical equality means a difference within half the last printed decimal.
> - **PR8:** the swap provider already supplies overlaps, so 7.1's "added if missing" needed no code; the swap jobs run in spawn processes.
> - **PR9:** the replay (5) comes after the runs.
>
> **Controller rulings** (ledger `.superpowers/sdd/2026-11-14-p9b-sharc-comparison/progress.md`):
>
> - **P1:** Task 5's live test body was written by the implementer from the Task 1 facts.
> - **P2:** a partly tautological Task 3 test was kept as written; no plan step changed.
> - **P3:** the long runs were started detached in WSL (`~/p9b/run_all.sh`).
> - **P4:** after the stop of the ours-nacdr run, it was resumed by a plain `gqteashmd run`, not by the chain's `--rerun-from dynamics`.
> - **P5:** the replay test failed its bounds and is `xfail(strict=True)` with its measured numbers. The bounds, window and start are unchanged.
> - **P6:** a report-only diagnosis of the replay's hop step was added, decided after the numbers were seen: the energies offline, and one pair of runs with `decoherence = "none"`.
> - **P7:** the Task 3 tool defect was fixed (`637a9f8`). `ensemble_compare.py` read `06_dynamics/<task>/output.h5` instead of `06_dynamics/tasks/<task>/output.h5` and silently paired 0. Task 9 first used symlinked shadow projects; the reruns on the real projects are identical.
>
> **Other changes:**
>
> - **The fit of section 4 is undelayed:** `make_fit.py` cannot express a delay (Task 1, F4), so the shared model is $y = A\,e^{-t/\tau}$ with $A = y(0)$. $s_{\text{SHARC}}$ is `make_fit.py`'s bootstrap std, which is not seedable.
> - **The quantum print bound is $10^{-9}$** (`%17.9f`, F5).
> - **SHARC's `data_extractor.x` is run by the tool**, not by `populations.py`, whose in-script call fails in this setup (F6).
> - **The SHARC print level is 2.**
> - **The `n_samples` 500 fallback did not fire** (100 of 100 selected at 300).
> - **A SHARC trajectory whose QM call failed** ("stopped": 1 swap, 5 direct) is not kept (3.2).
> - **Costs (3.1):** ours took 1 h 56 min; ours-nacdr about 7 h 34 min over three windows, after a deliberate stop and a WSL VM stop (two resumes, 16 trajectories resumed); swap 5 h 37 min; direct 15 h 35 min wall clock, but 13 h 44 min of process time (`wall_time_s` 49423.3 s; the gap is most likely a host sleep, not established). The coverage study took 979 s, not about 30 min.
> - **Section 6:** the warning wording was adapted to contain the word "coverage". The coverage text is a constant, and the "none reaches 0.90 -> 100" fallback is not coded, because N_min = 50 was reached.
> - **The replay (5) fails its bounds after the hop at step 23.** At every SHARC hop, SHARC's $E_{\text{tot}}$ equals the previous record's (measured over 221 swap hops); ours keeps the current step's Verlet-completed $E_{\text{tot}}$. That this is SHARC's rule is inferred. No `src/` change was made (8); this is decision 1 of the report.
> - **The direct ensemble** has 13 paired, so it is `inconclusive`. 75 of 100 start on another solution, and all of those jump on SHARC's side.
> - **The verdicts:** the swap rules pass on 48 paired, and the section-4 checks pass. N_min = 50 (coverage 0.895 / 0.9275 / 0.935), `SMALL_ENSEMBLE` = 50 and `AnalysisStage.version` 2.

Status: design approved section by section by the owner on 2026-11-14; this file awaits the owner's review. Sub-project P9b of the master plan's P9 (SHARC validation campaign), split from P9a by owner decision (P9a spec, section 0, decision 1). P9a and the CASSCF start rule are closed and merged into `main` (`07fc528`). Branch `feat/p9b-sharc-comparison`.

## 0. Owner decisions (2026-11-14, multiple choice; every recommended option)

| # | Question | Decision |
|---|---|---|
| 1 | Scope | The SHARC pieces and the report: (1) a molecular ensemble comparison with SHARC, (2) our analysis against SHARC's `populations.py`/`make_fit.py`, (3) the tau coverage study and a minimum-N rule, (4) a single-trajectory SHARC population check that can fail, (6) the comparison report. (5) TDA/TDDFT root-skip detection moves to a later robustness sub-project. |
| 2 | The molecular ensemble | N = 100. A **swap** ensemble (`sharc.x` driving our PySCF backend) compared with our driver under pass rules fixed before the run (asserted); a **direct** ensemble (`sharc.x` with SHARC's `SHARC_PYSCF.py`) as a report, its differences examined against the known start-solution difference. |
| 3 | Lost trajectories | Asserted comparison on the **paired intersection**: the initial conditions kept by both sides, by the same rule; each side's own kept set and the loss counts are reported. |
| 4 | The minimum-N rule | Warn below the measured N_min: the analysis keeps fitting at every N; the `small_ensemble` threshold moves from 20 to N_min and the warning states the measured coverage. |
| 5 | Approach | A: one spec, one plan; tools and tests first, then the statistics and the analysis change, then the single-trajectory check, then the long runs (measurements), then the report and documents. Every pass rule is fixed in this spec, before any run. |
| 6 | Design sections 1-3 | Approved as presented ("Yes." to each). |

## 1. Background

What is already compared with SHARC 4.1 (executed as a black box): single points (CASSCF energies, gradients, NACs, dipoles; P3a, P3b, P6); Tully and LVC single trajectories, asserted (P7a-1, P7a-2); Tully and LVC ensembles, report-only (the LVC one a swap test); one ethylene SA2-CASSCF(2,2)/6-31G\* trajectory over 23 steps directly and over 41 steps as a swap test (P7b, asserted: swap dq 2.134e-7 bohr, dv 2.115e-9, dE 4.550e-8 Eh, dP 8.629e-7). The direct single-trajectory population bound cannot discriminate (P(S0) <= 2.4e-5 in its window). P9a ran SHARC on 20 M4 initial conditions as a diagnostic (`tools/sharc_compare/run_sharc_pyscf_ensemble.py`): SA3 took 10549 s with 8 workers; **SHARC's interface starts on a different SA-CASSCF solution than ours in 13 of 20 starts** (P9a acceptance report). Our acceptance run (seed 2027) kept 13 of 20; the 7 losses all came after a hop. `sharc.x` does not end a trajectory at a solution switch; ours does (`overlap_failure`, `electronic_discontinuity`).

## 2. Scope and goal

**Goal.** Measure whether our dynamics driver and analysis reproduce SHARC 4.1 at the ensemble level on a molecule, with pass rules fixed before any run; settle the statistics question of the analysis (tau coverage, minimum N); and add a single-trajectory SHARC check whose population part can fail.

**In scope.** Sections 3-6. One behaviour change under `src/`: the analysis' small-ensemble threshold and warning (section 6). Everything else lives in `tools/` and `tests/`; `gqteashmd` never imports it.

**Out of scope.** TDA/TDDFT root-skip detection (a later robustness entry); O17 (a SHARC `input` importer); new dynamics features; fixes of any behaviour a comparison may expose (section 8).

## 3. The ensemble comparison

### 3.1 System and runs

- Ethylene (`tests/data/ethylene.xyz`), SA3-CASSCF(2,2)/6-31G\*, `[selection] states = ["S1"]`: the P9a acceptance setup, with the CASSCF start rule; `tmax_fs` 100, `dt_fs` 0.5, every other `[dynamics]` key at its default (`energy_jump_limit_eV` 0.1).
- **Seed 2028** (new, so nothing is tuned on trajectories already seen), `n_trajectories` 100. Sampling rule, fixed now: `n_samples` 300; if fewer than 100 trajectories are selected, `n_samples` is raised to 500 before any dynamics run (the dynamics settings are not touched).
- Four runs, one after another on the WSL native filesystem in `~/p9b/`, 8 workers x 2 threads, all from the same 100 initial conditions (`trajectories.json` and `samples.h5` of the seed-2028 project):

| Run | Driver | Electronic structure | Coupling | Estimated cost |
|---|---|---|---|---|
| ours | `gqteashmd run` | our backend | `overlap` (default) | about 2.5 h |
| ours-nacdr | `tools/diagnosis/run_variants.py` | our backend | `nacdr` | about 2.5 h |
| swap | `sharc.x` | our backend (`SwapModel`) | `overlap` | about 3-5 h |
| direct | `sharc.x` | `SHARC_PYSCF.py` | `nacdr` (its only option) | about 15 h |

The SHARC settings are translated from our `[dynamics]` by `run_sharc_pyscf_ensemble.sharc_settings` (which refuses untranslated values). The SHARC print level must record the per-step random numbers and what `populations.py`/`make_fit.py` need.

### 3.2 Kept and paired

- **Kept (ours):** the analysis rule: outcome `finished` or `stopped_in_s0`, and not in `energy.flagged` (no record-to-record total-energy jump above 0.1 eV).
- **Kept (SHARC):** the same rule applied to SHARC's output: the trajectory reaches `tmax_fs`, and no record-to-record total-energy jump above 0.1 eV.
- **Paired set** of a comparison: the initial conditions kept by both runs compared.
- If the swap comparison's paired set has **fewer than 30** trajectories, every swap rule's verdict is `inconclusive` (neither pass nor fail).

### 3.3 Swap pass rules (asserted)

On the paired set of ours vs swap, with both sides analysed by our `analysis/` functions (identical code), the bootstrap resampling **initial-condition indices** jointly (both members of a pair stay together), B = 1000, the stream `task_rng(2028, "p9b-compare", 0)`:

1. **Classical populations** of S0, S1, S2 at every 1 fs output time: `|P_ours(t) - P_swap(t)| <= 4 SE(t) + 0.02`, SE from the paired bootstrap of the difference.
2. **Lifetime and delay** of the delayed mono-exponential fit of `1 - P(S0)` (the analysis default target): `|tau_ours - tau_swap| <= 4 sqrt(s_ours^2 + s_swap^2)` with the bootstrap standard deviations of each side; the same for `t0`.
3. **First-hop time distributions** (trajectories with at least one accepted hop on the paired set): two-sample Kolmogorov-Smirnov `p >= 1e-3`.

Each rule gets the verdict `pass`, `fail` or `inconclusive`. Report-only: quantum populations, hop counts per state pair, frustrated hops, the total-energy drift distributions, and each side's own kept set with its loss counts by outcome.

### 3.4 Direct ensemble (report)

The same tables, for direct vs ours and direct vs ours-nacdr (the like-for-like coupling), on their paired sets; no verdict. Differences are examined against the start-solution difference (the step-0 energies of SHARC's interface against our step-0 energies, per initial condition, as in P9a) and the coupling difference.

## 4. Our analysis against SHARC's tools

- SHARC's `populations.py` and `make_fit.py` are executed as black boxes on **all 100 swap-ensemble SHARC trajectories** (SHARC's tools exclude nothing), driven non-interactively (as `run_excite.py` drives `excite.py`), with inputs from the public manual and the prompts observed at execution.
- Our `analysis/` functions run on the same 100 trajectories (read by `sharc_trajectories.py`, no exclusion).
- **Classical populations:** exact equality at every shared time.
- **Quantum populations:** `|difference|` at most one unit in the last digit SHARC prints; the plan's first task observes the print format and records the bound before any comparison.
- **Fit:** parameters within the combined uncertainties, on a model both programs express. The plan's first task records from the public manual what `make_fit.py` supports; the model is ruled and recorded before any fit is compared.

## 5. The replayed single trajectory (asserted)

- `sharc.x` on our backend (swap provider) with hopping on; its per-step random numbers are read from its output and replayed through our driver with `ReplaySource`. Both run on our backend, settings as in 3.1.
- **Start** (rule fixed now): the first seed-2028 swap trajectory, in index order, with an accepted hop within its first 20 fs; its positions, velocities and initial state are stored in `tests/data/ethylene_hop_start.json`, so the test does not need the seed-2028 project.
- **Bounds**, step by step up to the end of comparability: positions <= 1e-5 bohr, velocities <= 1e-6 a.u., populations and hopping probabilities <= 1e-5, and an identical active state at every step (so every hop at the same step).
- **End of comparability:** the first step at which the random number lies within 1e-5 of a cumulative hopping probability on either side (round-off could flip the decision), otherwise 80 steps (40 fs).
- **The check must be able to fail:** at least one hop and P(S0) >= 1e-3 inside the compared window. If the chosen start gives neither, the next start in index order is taken by the same rule; the choice is recorded.
- Markers `sharc`, `pyscf`, `slow`; a few minutes.

## 6. Statistics and the minimum-N rule

- `tools/validation/tau_coverage.py` (report script): synthetic ensembles drawn from the delayed mono-exponential model of P8 Task 5 (first-passage times `t0 + Exp(tau)` with t0 10 fs, tau 40 fs, on the 0.5 fs grid to 100 fs) at **N = 20, 50, 100**; **400 repetitions** each; **B = 500** bootstrap resamples through `analysis.kinetics.bootstrap_fit` (the code the stage runs). Coverage = the fraction of repetitions whose 95 % tau interval contains 40 fs (standard error about 0.015 at 0.9). JSON out, nothing asserted, fixed seed.
- **N_min** = the smallest of 20, 50, 100 with coverage >= 0.90. If none reaches it, N_min = 100 and the warning says the coverage was not reached at any tested N.
- `src/gqteashmd/stages/analysis.py`: `SMALL_ENSEMBLE` = N_min; the warning names N_min and the measured coverage at the tested N; `small_ensemble` = `n_included < N_min`. `AnalysisStage.version` 1 -> 2 (existing projects need `--rerun-from analysis`, seconds). A test asserts the threshold and the text.

## 7. Components

1. `tools/sharc_compare/run_sharc_pyscf_ensemble.py` (extended): `--provider sharc_pyscf|swap`; `swap` runs `sharc.x` with `SwapModel` (our backend and start rule) as the `QM.out` provider; the provider supplies overlaps for `coupling = "overlap"` (added if missing); resume and timeout behaviour kept; the print level of 3.1.
2. `tools/sharc_compare/sharc_trajectories.py` (new): a SHARC trajectory directory (`output.dat` and the random numbers) into `trajectory/store.py:TrajectoryData`; the kept rule of 3.2.
3. `tools/sharc_compare/run_sharc_analysis.py` (new): drives `populations.py` and `make_fit.py`, parses their outputs.
4. `tools/sharc_compare/ensemble_compare.py` (new): reads our `output.h5` files and the SHARC ensemble directories, builds the paired sets, computes the rules of 3.3 and the tables of 3.3-3.4 and section 4, writes one JSON with every number and a verdict per rule. The rules are pure functions.
5. The replay test (section 5) in `tests/validation/test_sharc_pyscf_dynamics.py`.
6. `tools/validation/tau_coverage.py` and the analysis change (section 6).
7. The report `docs/superpowers/reports/YYYY-MM-DD-p9b-sharc-comparison.md`, dated the day it is written.

Dependency rules unchanged: nothing in `src/` imports `tools/`; SHARC never imported.

## 8. Failures and honesty

- A failed swap rule or analysis comparison is reported with whatever diagnosis the existing replay tools give. It is **not tuned away**: no new seed, no re-run with other settings, no changed bound.
- P9b contains **no behaviour fix** beyond section 6. A failure traced to our code goes to the owner as a decision in the report.
- An `inconclusive` swap comparison is a result, reported with its loss counts. The direct ensemble has no verdict.
- The report and the manual state everything decided after numbers were seen.

## 9. Testing

- **Fast tier** (no SHARC): the comparison rules on synthetic ensembles with known outcomes (identical ensembles pass; a shifted population fails; fewer than 30 paired is `inconclusive`; the KS rule; the paired bootstrap keeps pairs together); the SHARC trajectory reader and kept rule on small fixtures produced by executing SHARC (outputs only) in `tests/data/sharc/`; the analysis threshold change.
- **Marker `sharc`:** the `populations.py`/`make_fit.py` drivers on a two-trajectory fixture.
- **Markers `sharc`, `pyscf`, `slow`:** the replay test (section 5).
- The runs (about 25 h) and the coverage study (about 30 min) are runs by hand, not tests.
- The full suite `pytest -W error` with SHARC and ORCA runs before the final review.

## 10. Constraints

- Clean room: SHARC 4.1 is executed only, from `tools/sharc_compare/`; inputs from the public manual and observed prompts; nothing in the installation is read, copied or adapted.
- Everything runs in WSL; git from WSL; never `--no-verify`; runs and large outputs in `~/p9b/`, not in OneDrive or `/mnt/c`.
- Seeds: every new random stream derives from a fixed seed through `task_rng`.

## 11. Documents

- CLAUDE.md: status, a P9b block, the Environment test counts; the deferred items resolved (a SHARC population check that can fail; the analysis against `populations.py`/`make_fit.py`; tau coverage and the minimum-N rule; molecular ensembles against SHARC); the TDA/TDDFT root-skip detection moved to a "later (robustness)" entry.
- USER_MANUAL.md: a new comparison subsection with every number (asserted or report), the analysis minimum-N change and version 2, sections 10 and 11.
