# P9b report: the SHARC 4.1 ensemble comparison, the analysis tools, the replay and the minimum N

Status: **the swap comparison passes all four of its rules on 48 paired trajectories; the comparison of our analysis with SHARC's `populations.py` and `make_fit.py` passes all three checks (the lifetime check uses a tool-local fit and a bound that does not discriminate, section 4); the direct ensemble is inconclusive (13 paired, fewer than 30); the replayed hopping-on trajectory agrees within its bounds before the hop, fails them after it, and the full-window test is a strict xfail; the tau coverage study gives N_min = 50.** No bound was changed, no seed replaced and nothing re-run with other settings (spec 8). The only behaviour change is the analysis threshold of section 6.

Date: 2026-11-18. Branch `feat/p9b-sharc-comparison`. The spec is `docs/superpowers/specs/2026-11-14-p9b-sharc-comparison-design.md` (owner-approved 2026-11-14; erratum at its top) and the plan is `docs/superpowers/plans/2026-11-14-p9b-sharc-comparison.md` (plan commit `563a9f4`; its "Plan rulings" PR1-PR9 are binding where they differ from the spec). The code tasks are the commits `9ffe8fe` to `e1ef1f8`, plus the tool fix `637a9f8` (ruling P7). Tasks 7 and 9 are measurements with no diff.

**Sources of every number:**

- the task reports in the git-ignored ledger `.superpowers/sdd/2026-11-14-p9b-sharc-comparison/` (`task-1-facts.md`, `task-1-report.md` to `task-9-report.md`, `fix-p7-report.md`) and its `progress.md`, which holds the controller rulings P1-P7;
- the JSON files on the WSL native filesystem of the development machine:
  - `~/p9b/compare/swap_fixed.json`, `direct_vs_ours_fixed.json` and `direct_vs_ours_nacdr_fixed.json`: the comparisons, rerun with the fixed tool (identical to the first runs except the path fields, ruling P7);
  - `~/p9b/compare/tools.json`: section 4. It was produced from the symlink shadow project (`project: ~/p9b/task9/shadow_ours`) and was not rerun on the real project, because `run_sharc_analysis.py` uses our project only for its configuration and was not affected by the P7 defect;
  - `~/p9b/tau_coverage.json`: section 6;
  - `~/p9b/swap/project/sharc_summary.json`, `~/p9b/direct/project/sharc_summary.json`;
  - `~/p9b/ours/run/` and `~/p9b/ours_nacdr/run/`: our projects (`06_dynamics/dynamics_summary.json`, `07_analysis/analysis.json`, `.gqteashmd/provenance.json`).
- Four counts were computed **while this report was written** (Task 10), by reading those files and running no calculation; each is marked *(Task 10)*:
  - the location of the largest population difference (section 2);
  - the start solutions of the paired and kept direct trajectories (section 3);
  - the aborted SHARC trajectories against the jump counts (sections 2 and 3);
  - the hops before our losses (section 7).

**Labels.**

- **Asserted**: a pass rule fixed in the spec before any run. Its verdict (`pass`, `fail` or `inconclusive`) is computed by the tool (`ensemble_compare.py --asserted`, `run_sharc_analysis.py`) or by a pytest assertion where this is said. The ensemble verdicts are **not** pytest tests: they come from measurements made by hand.
- **Report**: measured, with no rule.
- **Inference**: a reading that no measurement here establishes.

**Definitions** (spec 3.2):

- **Kept (ours)**: outcome `finished` or `stopped_in_s0`, and no record-to-record $\lvert\Delta E_{\text{tot}}\rvert$ above 0.1 eV.
- **Kept (SHARC)**: the trajectory reaches `tmax_fs` (100 fs), and no record-to-record jump above 0.1 eV. A SHARC trajectory whose QM call failed ("stopped") is not kept.
- **Paired set**: the initial conditions kept by both runs compared.
- **Fewer than 30 paired**: every rule is `inconclusive`.

## 0. Summary

- **Swap** (`sharc.x` driving our PySCF backend against our driver; asserted):
  - kept: ours 67, SHARC 58; **48 paired**;
  - populations **pass**: no point fails; the largest |dP| is 0.1875, and the worst ratio is S0 at 34 fs, dP 0.125 against a bound of 0.212;
  - tau **pass** (difference 39.65 fs against a bound of 93.78 fs);
  - t0 **pass** (difference 21.77 fs against 38.01 fs);
  - first-hop KS **pass** (p = 0.2395).
  - The two fit passes rest on wide intervals at N = 48: the point estimates differ by 0.42 and 0.57 of the bound.
- **Direct** (`sharc.x` with `SHARC_PYSCF.py`; report): SHARC kept 19, so **13 paired** against each of our runs, and every rule is `inconclusive`.
  - 75 of 100 trajectories start on another SA-CASSCF solution than ours.
  - *(Task 10)* All 75 of them jump above 0.1 eV on SHARC's side, and none is kept. All 19 SHARC-kept trajectories, and so all 13 pairs, share our start.
- **Our analysis against SHARC's tools** (all 100 swap SHARC trajectories; asserted): all three checks **pass**.
  - Classical populations: max |diff| 0.0.
  - Quantum populations: 4.998e-10 against 1e-9.
  - Fit of the undelayed model: tau 78.491 against 78.629 fs, a difference of 0.138 fs against a bound of 36.93 fs. **Our side of this fit is the tool-local `fit_undelayed`, not the stage's fit code** (`analysis.kinetics`). So only our population functions are validated against SHARC's tools, not the stage's lifetime fit. The bound does not discriminate (section 4).
- **Replay** (asserted, pytest): it **fails its pre-registered bounds** after the hop at step 23. Since the final review the test is split:
  - one passing test checks the bounds on the records before the hop;
  - the full-window test stays a strict xfail (ruling P5).
  - Measured at that hop: SHARC's total energy after the hop equals the previous step's total energy to 4e-13 Eh, while ours keeps the current step's Verlet-completed total energy.
  - Over all 221 SHARC swap hops: |dE_tot| <= 9.71e-12 Eh at the hop, against a non-hop median of 8.31e-5 Eh.
  - That SHARC targets the previous record's energy as a rule is an inference.
- **Coverage** (report): the coverage of the 95 % tau interval is 0.895 at N = 20, 0.9275 at N = 50 and 0.935 at N = 100.
  - **N_min = 50**, so `SMALL_ENSEMBLE` = 50 and `AnalysisStage.version` = 2. Existing projects need `--rerun-from analysis`.
  - N = 20 misses 0.90 by 0.005, within one standard error (0.0153). This is the rule's result, not a sharp boundary.
  - At every tested N the coverage is **below the nominal 0.95**: 3.6, 1.7 and 1.2 standard errors below.
- **Found and fixed during the work:** a Task 3 tool defect, fixed in commit `637a9f8` (ruling P7). As first written, `ensemble_compare.py` paired 0 trajectories silently. The reruns are identical to the shadow-project results.

## 1. What was run

**Machine and software.**

- **Machine:** the development workstation, a 12th Gen Intel Core i7-1260P. WSL Oracle Linux 9.5 shows 16 logical CPUs and 7.9 GB of memory (`MemTotal` 7940272 kB).
- **Software** (from the provenance records of our runs, all at commit `72d736a`, `git_dirty` false or unreadable): Python 3.12.16, PySCF 2.14.0, NumPy 2.5.4 and SciPy 1.18.1.
- **SHARC 4.1** at `/home/ajc/sharc4-4.1`, run with its conda Python `/home/ajc/miniforge3/envs/sharc4.1/bin/python`. Its `output.dat` header says `SHARC_version 4.0`.
- **Location:** every run used the WSL native filesystem under `~/p9b/`. The repository stayed at `72d736a` until the run chain ended; the Task 8 and 9 commits came after it.

**System** (spec 3.1, `tools/sharc_compare/p9b/ethylene_seed2028.toml`):

- ethylene `tests/data/ethylene.xyz`, SA3-CASSCF(2,2)/6-31G\* with the CASSCF start rule;
- `[selection] states = ["S1"]`, seed 2028, `n_samples` 300, `n_trajectories` 100;
- `tmax_fs` 100, `dt_fs` 0.5, every other `[dynamics]` key at its default (`coupling = "overlap"`, `energy_jump_limit_eV` 0.1, `substep_interpolation = "midpoint"`, PR3);
- 8 tasks x 2 threads, 800 MB per task.

The selection gave 100 of 100, so the `n_samples` 500 fallback did not fire.

**Code tasks** (one commit each; review clean, Task 4 after one fix round, Task 8 after one fix round):

| Task | Commit | What |
|---|---|---|
| 1 | `9ffe8fe` | SHARC formats and tools observed (facts F1-F6); fixtures `tests/data/sharc/p9b/` |
| 2 | `8525830` | `tools/sharc_compare/sharc_trajectories.py`: SHARC and our trajectories as `grid.Trajectory` (PR1); the kept rule |
| 3 | `fcec793` | `tools/sharc_compare/ensemble_compare.py`: paired set, the rules, the tables |
| 4 | `ee721aa`, fix `345c2d9` | `run_sharc_pyscf_ensemble.py --provider swap` (spawn processes, PR8); `ProjectInputs.equilibrium_bohr` |
| 5 | `ebcb39e` | `tools/sharc_compare/run_sharc_analysis.py`: drives `populations.py` and `make_fit.py` |
| 6 | `a52e7c0` | `tools/validation/tau_coverage.py`; `SMALL_ENSEMBLE` = 50, `AnalysisStage.version` 2 |
| 7 | `72d736a` | the inputs `tools/sharc_compare/p9b/` |
| 8 | `ad8f36d`, fix `e1ef1f8` | the replay test and `tests/data/ethylene_hop_start.json`; strict xfail (P5) |
| fix P7 | `637a9f8` | `ensemble_compare.py` reads `06_dynamics/tasks/<task>/output.h5` and fails loudly when it finds none |

**The four runs** (Task 7; a detached chain `~/p9b/run_all.sh`, ruling P3; started 2026-11-14 10:03:23, `ALL_DONE` 2026-11-18 11:50:30):

| Run | Command (abridged) | Wall time | Ended |
|---|---|---|---|
| ours (`overlap`) | `gqteashmd run input.toml` in `~/p9b/ours` | 1 h 56 min (dynamics 1 h 52 min 42 s) | finished 67, `overlap_failure` 23, `electronic_discontinuity` 10 |
| ours-nacdr | copy of `ours`, `coupling = "nacdr"`, `gqteashmd run input.toml --rerun-from dynamics` (PR2), then two plain resumes (P4) | about 7 h 34 min, the sum of three windows | finished 69, `overlap_failure` 25, `electronic_discontinuity` 6 |
| swap | `run_sharc_pyscf_ensemble.py ~/p9b/ours/run tools/sharc_compare/p9b/variants_p9b.toml --variant project --provider swap --out ~/p9b/swap --sharc-home ... --python ... --workers 8 --threads 2 --memory-mb 800` | 5 h 37 min (20240.5 s) | finished 99, stopped 1, no errors |
| direct | the same with `--provider sharc_pyscf --out ~/p9b/direct` | 15 h 35 min wall clock; 13 h 44 min of process time (`wall_time_s` 49423.3 s) | finished 95, stopped 5, no errors |

**The two direct times.** The clock span of the direct run, from the chain log, is 2026-11-17 20:15:30 to 2026-11-18 11:50:30, which is 15 h 35 min (56100 s). The tool's `wall_time_s` is 49423.3 s, which is 13 h 44 min.

- The timer is `time.perf_counter()` (Linux `CLOCK_MONOTONIC`) around the whole ensemble (`run_sharc_pyscf_ensemble.py`, read by the controller), so set-up time is not the cause of the 1 h 51 min difference.
- *Inference, not established:* `CLOCK_MONOTONIC` does not advance while the WSL VM is suspended, and the guest's wall clock jumps forward when it resyncs after a host sleep. So the Windows host most likely slept for about 1 h 51 min overnight. The file modification times show only the batch rhythm, with no distinguishing gap.
- **The swap run is consistent:** `wall_time_s` 20240.5 s against a span of 20242 s (14:38:08 to 20:15:30).
- **Ours and ours-nacdr have only clock times** (spans from their run logs; no second timer over the whole run), so the data cannot rule out a host sleep inside their windows either.

Totals of our runs:

| Quantity | ours | ours-nacdr |
|---|---|---|
| hops | 233 | 216 |
| frustrated hops | 2 | 1 |
| rejections | 212 (energy 83, overlap 128, rung 1) | 193 (electronic 1, energy 69, overlap 123) |
| flagged steps | 1 | 0 |
| max drift | 0.2735 eV | 0.2735 eV |
| analysis N | 67, excluded fraction 33 % | 69, excluded fraction 31 % |
| fit (classical, `1 - P(S0)`), t0 | 12.84 fs [6.66, 22.13] (std 3.77) | 13.75 fs [7.66, 25.27] (std 4.47) |
| fit, tau | 80.14 fs [57.62, 110.06] (std 13.09) | 87.87 fs [62.44, 118.71] (std 14.45) |
| fit rms | 0.0400 | 0.0228 |

These are the stages' own analyses (report). Both projects share the selection and the dynamics random streams. The `nacdr` run is about 3.5 times slower per trajectory, and 16 of its trajectories were resumed (`n_resumes` > 0). Its wall time is therefore not like-for-like with ours.

**Interruptions of ours-nacdr** (not a program fault):

1. On 2026-11-14 at 12:22:36: a deliberate `gqteashmd stop` before the owner shut the computer down (exit 5).
2. A resume at 17:38:58 by a plain `gqteashmd run` (ruling P4, so the partial dynamics was not superseded).
3. That controller died with the WSL VM after 20:35:51, at 41 trajectories done and 8 running. The VM had rebooted by 2026-11-17 10:10. The likely cause is WSL's idle shutdown, which is not established.
4. A second resume on 2026-11-17 at 10:28:12. The 8 RUNNING tasks were reconciled and requeued, and none failed on its restart files.

The owner was offered `vmIdleTimeout=-1`; it was not set.

**SHARC's aborted trajectories.** A SHARC "stopped" trajectory is one whose QM call failed ("QM call was not successful, aborting the run.").

- **Swap:** `traj_00085` at 76.5 fs (max jump 0.291 eV): our backend failed inside the swap provider. In our own run the same initial condition ended `overlap_failure` at 62.25 fs *(Task 10)*.
- **Direct:** `SHARC_PYSCF.py` failed for these trajectories:

| Trajectory | t (fs) | max jump (eV) |
|---|---|---|
| `traj_00009` | 30.0 | 0.039 |
| `traj_00036` | 79.0 | 0.269 |
| `traj_00059` | 41.0 | 0.330 |
| `traj_00090` | 50.0 | 0.881 |
| `traj_00092` | 89.5 | 0.264 |

**The comparisons** (Task 9, with the fixed tool `637a9f8` on the real projects):

| Step | Command (abridged) | Wall time (shadow run) |
|---|---|---|
| swap | `ensemble_compare.py ~/p9b/ours/run ~/p9b/swap/project --label swap --asserted --out ...` | 19.0 s |
| direct vs ours | `ensemble_compare.py ~/p9b/ours/run ~/p9b/direct/project --label direct_vs_ours` | 6.4 s |
| direct vs ours-nacdr | the same against `~/p9b/ours_nacdr/run` | 6.7 s |
| tools | `run_sharc_analysis.py ~/p9b/swap/project <ours> --sharc-home ... --python ... --work ~/p9b/tools --out tools.json --quantum-bound 1e-9` | 39.7 s |
| hop energies | throwaway `~/p9b/task9/hopE.py` | 6.5 s |

Two more runs:

- the replay test (Task 8), 1 xfailed in 372.05 s;
- the coverage study (Task 6), 979.2 s with 8 workers.

## 2. The swap comparison (asserted)

`sharc.x` drives our `PySCFBackend` (with the CASSCF start rule) through the swap provider, and is compared with our driver (`ours`). Both sides are analysed by our `analysis/` functions. The bootstrap resamples initial-condition indices jointly, with B = 1000 and the stream `task_rng(2028, "p9b-compare", 0)`. The comparison grid is every 1 fs (PR6).

**Kept and paired:**

- Ours: 67 kept of 100 (finished 67, `overlap_failure` 23, `electronic_discontinuity` 10).
- SHARC: 58 kept. 99 trajectories reached 100 fs and 42 have a jump above 0.1 eV.
  - *(Task 10, from `sharc_summary.json`)* The aborted `traj_00085` is among those 42 (max jump 0.291 eV). Of the 42, 41 finished and 1 aborted, so 99 - 41 = 58.
- **Paired: 48** (at least 30, so the rules are conclusive).
- No trajectory is missing on either side.
- *(report)* Every paired trajectory starts on the same solution on both sides: 0 of 100 step-0 differences exceed 1e-3 eV (max 1.28e-6 eV).

| Rule (fixed in spec 3.3) | Bound | Measured | Verdict |
|---|---|---|---|
| 1. Classical S0, S1, S2 at every 1 fs: $\lvert dP\rvert \le 4\,\mathrm{SE} + 0.02$ | per point | 0 points failing; max SE 0.0982 | **pass** |
| 2a. $\lvert\tau_{\text{ours}} - \tau_{\text{swap}}\rvert \le 4\sqrt{s_a^2 + s_b^2}$ | 93.78 fs | ours 93.55 fs (std 18.64), swap 53.89 fs (std 14.22); difference 39.65 fs | **pass** |
| 2b. the same for $t_0$ | 38.01 fs | ours 12.63 fs (std 5.17), swap 34.40 fs (std 7.98); difference -21.77 fs | **pass** |
| 3. First-hop times, two-sample KS $p \ge 10^{-3}$ | $10^{-3}$ | D = 0.2128, p = 0.2395; n = 47 and 47 (one pair has no hop on either side) | **pass** |

Rule 1 in detail: the largest |dP| and the point with the largest ratio to its bound are different points. *(Task 10: the per-point values were recomputed from the paired trajectories with the tool's own functions. The recomputation reproduces the JSON's `max_abs_difference` 0.1875 and `worst` (time index 34, S0, 0.125, bound 0.2120).)*

| Point | dP | Bound | Ratio |
|---|---|---|---|
| **largest \|dP\|**, S1 at 55 fs (first of three ties at 9/48) | -0.1875 | 0.4105 | 0.457 |
| the same \|dP\|, S1 at 87 fs | +0.1875 | 0.3558 | 0.527 |
| the same \|dP\|, S0 at 87 fs | -0.1875 | 0.4010 | 0.468 |
| **largest ratio**, S0 at 34 fs | +0.1250 | 0.2120 | 0.590 |

**Report items on the paired set:**

- Quantum populations: max |difference| 0.1697.
- Hops per state pair:

| Pair | ours | SHARC |
|---|---|---|
| S0->S1 | 35 | 22 |
| S0->S2 | 1 | 1 |
| S1->S0 | 64 | 56 |
| S1->S2 | 21 | 19 |
| S2->S0 | 1 | 2 |
| S2->S1 | 16 | 18 |

- Frustrated hops: ours 1; SHARC not available, because `output.dat` does not record them (PR4).
- Total-energy drift (eV): ours median 0.0477, p95 0.1995, max 0.2735; SHARC median 0.0406, p95 0.1079, max 0.1539.

**Reading:**

- All four rules pass. The tau and t0 passes, however, rest on wide uncertainties at N = 48. The point estimates differ by 40 fs (tau) and 22 fs (t0), which are 0.42 and 0.57 of the bounds.
- The two fits trade a delay against a lifetime: SHARC's side has a later delay and a shorter lifetime. The rule cannot separate that trade-off from agreement *(inference)*.
- Our drift tail is larger (p95 0.20 against 0.11 eV). No rule compares it. Section 5 suggests a cause *(inference)*.

## 3. The direct ensemble (report)

`sharc.x` with SHARC's own interface `SHARC_PYSCF.py` uses `nacdr`, its only option, against ours (`overlap`) and ours-nacdr (the like-for-like coupling). No verdict: by spec 3.4 the direct ensemble is a report. The tool also returns `inconclusive` for every rule, with the reason "only 13 paired trajectories (fewer than 30)", and computes no fit, population or KS number for it.

**Kept and paired:**

- SHARC kept 19 of 100. 95 trajectories reached 100 fs, 5 aborted, and 80 have a jump above 0.1 eV.
  - *(Task 10, from `sharc_summary.json`)* 4 of the 5 aborted trajectories are among the 80 with a jump; `traj_00009` (aborted at 30.0 fs, max jump 0.039 eV) is not. Of the 80, 76 finished, so 95 - 76 = 19.
- Ours kept 67 (`overlap`) and 69 (`nacdr`).
- **Paired: 13** against each.

| Item | direct vs ours (`overlap`) | direct vs ours-nacdr |
|---|---|---|
| paired | 13 | 13 |
| kept ours / SHARC | 67 / 19 | 69 / 19 |
| quantum max \|difference\| | 0.4010 | 0.4558 |
| hop pairs ours (paired) | S0->S1 8, S1->S0 16, S1->S2 7, S2->S1 6 | S0->S1 7, S1->S0 16, S1->S2 6, S2->S1 5 |
| hop pairs SHARC (paired) | S0->S1 3, S1->S0 11, S1->S2 4, S2->S1 4 | the same |
| frustrated ours | 1 | 0 |
| drift ours, eV (median / p95 / max) | 0.0521 / 0.2222 / 0.2295 | 0.0468 / 0.2198 / 0.2295 |
| drift SHARC, eV | 0.0346 / 0.0441 / 0.0511 | the same |

**Start solutions** (spec 3.4). The step-0 difference is $\lvert E_{\text{tot}}(0)_{\text{ours}} - E_{\text{tot}}(0)_{\text{SHARC}}\rvert$ per initial condition. The threshold is the tool's `START_DIFFERENCE_EV` = 1e-3 eV.

- The direct values fall in two clusters: 25 values from 3.0e-8 to 2.5e-6 eV, and 75 values from 0.139 to 0.459 eV. Every threshold inside the empty gap gives the same count.
- **75 of 100 direct starts are on another SA-CASSCF solution than ours**, for both our couplings (the start does not depend on the coupling). P9a measured 13 of 20 on seed 2026.
- Swap, as a sanity check: 0 of 100 (max 1.28e-6 eV, median about 3e-7 eV).

*(Task 10, from `direct_vs_ours_fixed.json` and `sharc_summary.json`)*

| Start | Trajectories | SHARC kept | SHARC jumped > 0.1 eV | SHARC aborted |
|---|---|---|---|---|
| same as ours | 25 | 19 | 5 | 1 |
| different from ours | 75 | **0** | **75** | 4 (all among the jumped) |

So all 19 SHARC-kept direct trajectories start on our solution, and all 13 pairs do too. **Every** trajectory that starts on another solution jumps above 0.1 eV at some record on SHARC's side.

*Inference, not measured:* those jumps are switches back to, or between, SA-CASSCF solutions, which `sharc.x` accepts and our stage rejects (F1 and the overlap signal). The same is suggested by the measured accepted SHARC hops in section 5, by the 42 jumps in the swap ensemble, and by P9a's finding of switches at the same step.

**Coupling.** Like-for-like coupling did not reduce the quantum difference: 0.4010 against ours (`overlap`), 0.4558 against ours-nacdr, and 0.1697 for the swap pairs. With 13 pairs this is not shown to be a coupling or a start effect *(report; no cause established)*.

**Consequence:** a direct comparison on these settings cannot isolate driver differences. The start differs in 75 of 100 trajectories, and those 75 all jump on SHARC's side.

## 4. Our analysis against SHARC's `populations.py` and `make_fit.py` (asserted)

SHARC's tools ran as black boxes on **all 100 swap SHARC trajectories**. Our `analysis/` functions ran on the same 100 trajectories, read by `sharc_trajectories.py` with no exclusion.

- **The tools:** `populations.py` mode 2 (classical, active MCH state) with bootstrap data, and mode 8 (quantum, $\lvert c\rvert^2$ in the MCH basis). `data_extractor.x` ran by hand in each trajectory directory (F6), so `populations.py` was answered `False` to its extractor prompt. `make_fit.py` then ran in normal mode for tau and in bootstrap mode for its std.
- **The bounds** were fixed in Task 1, before any comparison:
  - classical, PR7: $\le 0.5\times10^{-9}$, since the values are printed with 9 decimals;
  - quantum, F5: $10^{-9}$, one unit of `%17.9f`;
  - fit, F4: the shared model is $y = A\,e^{-t/\tau}$ with $A = y(0)$, because `make_fit.py` cannot express a delay. The target is the classical `1 - P(S0)` on SHARC's printed 0.5 fs grid, and $s_{\text{SHARC}}$ is the bootstrap arithmetic std of `make_fit.py`.
- `skipped` is empty, so all 100 trajectories were used.

| Check | Bound | Measured | Verdict |
|---|---|---|---|
| classical populations | exact ($0.5\times10^{-9}$) | max \|diff\| 0.0 | **pass** |
| quantum populations | $10^{-9}$ | $4.998\times10^{-10}$ | **pass** |
| fit, $\lvert\tau_{\text{ours}} - \tau_{\text{SHARC}}\rvert \le 4\sqrt{s_{\text{ours}}^2 + s_{\text{SHARC}}^2}$ | 36.93 fs | ours 78.491 fs (std 6.485), SHARC 78.629 fs (std 6.570); difference -0.138 fs | **pass** |

**Caveats (report):**

- `make_fit.py` has no seed prompt, so its bootstrap std is not reproducible (`sharc_bootstrap_reproducible` false). It was run once and recorded.
- Our side used B = 1000 with seed 2028.
- This undelayed tau is not the delayed tau of section 2: the comparison is between the two programs' fits of the same model. Our delayed fit (t0, tau) has no SHARC counterpart.
- **What the fit check validates (final review).** Our side is `fit_undelayed` in `tools/sharc_compare/run_sharc_analysis.py`, a fit written for this comparison. It is not the stage's `gqteashmd.analysis.kinetics` (`fit_delayed_exponential`, `bootstrap_fit`). So the stage's fit code is **not** validated against `make_fit.py`; the populations (`classical_populations`, `quantum_populations`) are.
- **The fit bound does not discriminate.** The bound, 4 × the combined bootstrap std = 36.93 fs, is a sampling bound applied to two fits of the *same* data. It would pass a difference of up to 47 % of tau (36.93 of 78.63 fs).
- **The measured 0.138 fs (0.18 %) difference is unexplained.** `make_fit.py`'s fitting method, the columns it fits and its weighting are possible causes; none is established.
- **The live tests of this comparison run on a fixture with no decay** (`tests/sharc_compare/test_run_sharc_analysis.py`; the two Task 1 fixture trajectories stay in S1, SHARC's tau 104172 fs). So no test exercises the fit path on decaying data; only the Task 9 measurement does.

## 5. The replayed hopping-on trajectory (asserted; strict xfail)

**Setup (spec 5, PR9).**

- `sharc.x` on our backend (swap provider) with hopping on. Its per-step random numbers (`output.dat` block 9) are replayed through our `TrajectoryRunner` with `ReplaySource`.
- Both sides: SA3-CASSCF(2,2)/6-31G\*, `overlap` coupling, EDC, `parallel_vel`, `endpoint` interpolation (PR3), and `frustrated` keep (SHARC's `reflect_frustrated none`).
- **Start, by the fixed rule:** the first swap trajectory in index order with an accepted hop within 20 fs, which is start 9 (`traj_00009`, `rngseed` 10, a hop S1 -> S0 at 11.5 fs). It met the "must be able to fail" conditions: a hop at step 23 inside the window, and max P(S0) 0.99999975 in the window. So no other start was tried. The start is in `tests/data/ethylene_hop_start.json`.
- Test: `tests/validation/test_sharc_pyscf_dynamics.py` (markers `sharc`, `pyscf`, `slow`). The Task 8 test was `test_replayed_hopping_on_trajectory`. Since the final review it is a module fixture `replay`, which runs both sides once, plus two tests:
  - `test_replayed_trajectory_agrees_before_the_hop` (passing): the preconditions, the same active state over the window, and the four bounds over the records before the first hop;
  - `test_replayed_trajectory_over_the_pre_registered_window` (strict xfail): the four bounds over the whole window, each assertion naming its key.
- The window also ends at our first rejected step or substep record, a plan-level addition to spec 5 (a halving would shift the replayed stream). It did not fire.

**Result.**

- The window is 81 records (steps 0-80), ended by the 80-step limit, with no near tie, no rejection and no halving.
- There is one hop, at step 23, with the same active state in every record.
- Steps 0-22 agree to max dq 1.3e-7 bohr, dv 1.2e-9, d populations 1.0e-7 and d probabilities 1.0e-7. The hop probability at step 23 agrees to 3.1e-7.
- The first exceedance is at the hop step: dv 3.3e-6 and d populations 1.1e-4 at step 23, then positions at step 24 (6.7e-5 bohr). After that the difference grows: dq 1.3e-4 at step 25, 3.9e-3 at 50, 0.1 at 68 and 0.45 at 80.

| Window maximum | Bound (spec 5) | Measured |
|---|---|---|
| positions | $10^{-5}$ bohr | **0.4547** bohr |
| velocities | $10^{-6}$ a.u. | **$2.79\times10^{-3}$** |
| populations | $10^{-5}$ | **$2.87\times10^{-4}$** |
| hopping probabilities | $10^{-5}$ | **$5.19\times10^{-4}$** |
| active state | identical | identical |

The first run failed in 379.45 s. Ruling P5 then made the test `xfail(strict=True, raises=AssertionError)`, with these numbers in its reason. The rerun gave 1 xfailed in 372.05 s with identical numbers. The bounds, the window and the start were not touched. The split of the final review is recorded in section 8, with its run.

**Diagnosis (ruling P6, decided after the numbers were seen; report only, nothing changed).**

At the hop step (23) the energies, in Eh, are:

| Step | Side | Active | $E_{\text{kin}}$ | $E_{\text{pot}}$ | $E_{\text{tot}}$ |
|---|---|---|---|---|---|
| 22 | SHARC | S1 | 0.1118460514 | -77.7363383060 | -77.6244922546 |
| 22 | ours | S1 | 0.1118460438 | -77.7363383153 | -77.6244922715 |
| 23 | SHARC | S0 | 0.1323744822 | -77.7568667367 | **-77.6244922546** |
| 23 | ours | S0 | 0.1325454102 | -77.7568667165 | -77.6243213063 |

- **Measured, SHARC side:** after the hop, $E_{\text{pot,new}} + K_{\text{after}} - E_{\text{tot}}(22) = -4.0\times10^{-13}\,E_h$. SHARC's printed total energy after the hop equals the previous step's total energy.
- **Measured, our side:** $K_{\text{after}}$ equals the Verlet-completed $K$ minus $\Delta E$, 1.71e-4 Eh above $E_{\text{tot}}(22)$. This is our documented step order (USER_MANUAL 8.12: the hop adjusts the velocity at the completed step, so $E_{\text{tot}}$ is continuous with that step).
- **Same energy before the adjustment:** the Verlet-completed $K$ before the hop is the same on both sides to 8e-9 Eh (0.1014612737 against 0.1014612814).
- **Velocity:** SHARC's v(23) is parallel to the reconstructed pre-hop velocity to 7.6e-16, so only the scale differs: observed 1.142225800 against our 1.142963056.
  - The scale ratio is 0.999355, i.e. $\sqrt{0.1323744822/0.1325454102}$.
  - That ratio is the whole step-23 dv of 3.25e-6.
- **Populations:** the step-23 population difference of 1.13e-4 is the EDC evaluated with the two different post-hop kinetic energies.
  - Our EDC applied with the other side's $K$ predicts each side's printed values to 2.6e-7.
  - The recovered pre-EDC populations agree to 3.7e-7.
- **The pair of runs with `decoherence = "none"`** (same start and stream; 80 steps; two hops, at steps 23 and 26):
  - the step-23 population difference falls to 4.3e-7;
  - dv at step 23 stays 3.25e-6;
  - the window maxima are dq 1.65e-2, dv 7.54e-5, d populations 1.05e-2 and d probabilities 1.55e-2.

**Every swap hop (Task 9, controller addition; report).** dE is $E_{\text{tot}}(\text{record}) - E_{\text{tot}}(\text{previous record})$, with $E_{\text{tot}}$ = active MCH energy + kinetic energy. A SHARC hop is a change of active state between printed records. For ours, a hop step is a full grid step following a full grid step; hop records inside halved intervals are listed apart.

| Set | n | median \|dE\| (Eh) | p95 | max |
|---|---|---|---|---|
| SHARC swap, hop steps | 221 | 2.59e-12 | 7.57e-12 | **9.71e-12** |
| SHARC swap, non-hop steps | 19732 | 8.31e-05 | 6.81e-04 | 3.99e-02 |
| SHARC direct, hop steps | 167 | 0 | 4.3e-14 | 1.14e-13 |
| SHARC direct, non-hop steps | 19412 | 8.00e-05 | 5.50e-04 | 3.24e-02 |
| ours (`overlap`), hop steps | 224 | 1.36e-04 | 1.22e-03 | 3.63e-03 |
| ours (`overlap`), non-hop steps | 17330 | 7.42e-05 | 5.35e-04 | 3.65e-03 |
| ours (`overlap`), hop records not a full consecutive step | 9 | 8.37e-04 | 2.63e-03 | 2.68e-03 |
| ours (`nacdr`), hop steps | 211 | 1.45e-04 | 1.14e-03 | 3.60e-03 |
| ours (`nacdr`), non-hop steps | 17307 | 7.46e-05 | 5.32e-04 | 3.65e-03 |
| ours (`nacdr`), hop records not a full step | 5 | 2.11e-03 | 2.63e-03 | 2.68e-03 |

**What is measured and what is inferred:**

- **Measured on the SHARC side:** at every one of the 221 swap hops (and the 167 direct hops), SHARC's printed $E_{\text{tot}}$ equals the previous record's $E_{\text{tot}}$ to the print precision ($\le 9.71\times10^{-12}\,E_h$, against a non-hop median of $8.31\times10^{-5}$).
- **Measured on our side:** our hop steps have a median |dE| of 1.36e-4 / 1.45e-4 Eh, about 1.8 times the non-hop median. Their signed median is -1.3e-5 (`overlap`) and -2.1e-5 (`nacdr`), against -8.6e-7 and -1.0e-6 for non-hop steps. The maxima are of the non-hop size, so the hop steps are not outliers.
- **Inferred, not measured:**
  - that SHARC's velocity adjustment *targets* the previous record's total energy, i.e. that it "conserves" that energy as a rule (SHARC's source was not read);
  - that our hop steps carry a systematic bias from our convention;
  - that this explains our larger drift tail in section 2;
  - that the growth after step 23 in the replay (0.455 bohr at step 80) is the dynamics amplifying the 3e-6 velocity difference, which was not isolated.

## 6. The tau coverage study and N_min (report; the threshold in the code is asserted by a test)

`tools/validation/tau_coverage.py` (spec 6, fixed before the run):

- first-passage times $t_0 + \mathrm{Exp}(\tau)$ with $t_0$ = 10 fs and $\tau$ = 40 fs, on a 0.5 fs grid to 100 fs;
- N = 20, 50, 100, with 400 repetitions each;
- B = 500, through `analysis.kinetics.bootstrap_fit`;
- seed 2028, stream `p9b-coverage`.

Coverage is the fraction of repetitions whose 95 % tau interval contains 40 fs. The study took 979.2 s with 8 workers and `OMP_NUM_THREADS=1`.

| N | covered / 400 | coverage | standard error | no interval | median tau (fs) |
|---|---|---|---|---|---|
| 20 | 358 | 0.895 | 0.0153 | 0 | 38.13 |
| 50 | 371 | 0.9275 | 0.0130 | 0 | 38.80 |
| 100 | 374 | 0.935 | 0.0123 | 0 | 39.80 |

**N_min = 50**, by the rule "the smallest N with coverage >= 0.90". N = 20 at 0.895 misses 0.90 by 0.005, within one standard error. N_min = 50 is the rule's result, not a sharp boundary: the coverage at N = 20 is statistically indistinguishable from 0.90.

**The nominal level is not reached.** The interval is a nominal 95 % interval, and its measured coverage is below 0.95 at every tested N:

| N | coverage | standard errors below 0.95 |
|---|---|---|
| 20 | 0.895 | 3.6 |
| 50 | 0.9275 | 1.7 |
| 100 | 0.935 | 1.2 |

So N = 50 reaches the **pre-registered 0.90 threshold**, not the nominal level. The median fitted tau is also biased low at small N: 38.13 fs at N = 20, 38.80 at N = 50 and 39.80 at N = 100, against the true 40 fs.

**What changed in the code** (`src/gqteashmd/stages/analysis.py`, the only change under `src/`; commit `a52e7c0`, the text corrected by the final review):

- `SMALL_ENSEMBLE = 50` (was 20), and `small_ensemble` is `n_included < 50`.
- `SMALL_ENSEMBLE_COVERAGE = "coverage 0.895 at N = 20, 0.9275 at N = 50, 0.935 at N = 100"`, so the warning reads: `only N trajectories are included (fewer than 50): populations, intervals and the fit are not statistically reliable (measured 95 % tau interval coverage 0.895 at N = 20, 0.9275 at N = 50, 0.935 at N = 100)`.
  - Task 6 printed 0.927 for 0.9275. The final review changed it to 0.9275, and the test now asserts the whole text.
  - The wording was adapted to read well with the word "coverage" that the test requires.
- `AnalysisStage.version` 1 -> 2: an existing project with `analysis` results stops with exit 3 and needs `--rerun-from analysis` (seconds; the trajectories stay).

The tests assert the threshold, the version and the warning text: `tests/stages/test_analysis.py`, which checks the text through the constants, and `tests/validation/test_tau_coverage_tool.py` (3 passed). The analysis keeps fitting at every N (owner decision 4 of the spec: warn, do not refuse).

## 7. Decisions for the owner

Each item is a failed or inconclusive result, or a finding, with its diagnosis and the options. Nothing in P9b changed behaviour beyond section 6 (spec 8).

**1. The hop energy convention (the replay failure).**

Diagnosis (section 5):

- **Measured:** at every SHARC hop, the total energy after the hop equals the previous record's. Ours keeps the current step's Verlet-completed total energy. The difference at the replay hop is 1.71e-4 Eh in $E_{\text{tot}}$ and 6.5e-4 in the velocity scale.
- **Inferred:** that this is SHARC's rule.

Options:

- (a) **Keep ours** and document the difference as a defined convention difference. Cost: the replay test stays a strict xfail, and the difference stays in the manual. None for the physics as far as measured: the swap rules pass.
- (b) **Adopt SHARC's convention** in `dynamics/driver.py` (the energy target of the hop's velocity adjustment). Cost: a `src/` change in the core, and new golden data (`tests/dynamics/data/tully_golden.npz` holds trajectories with hops, so bit-identity with P7a-1 is lost). It also needs a new replay measurement (the test takes about 6 min). *Estimate, not established:* the asserted Tully and LVC comparisons with hops would have to be re-run too (the P7a-1 exact comparison alone takes about 94 min), and the documents change.
- (c) **Make it an option** (for example a `[dynamics]` key), default unchanged. Cost: the work of (b) plus a hashed key, its validation and tests; the default case keeps the xfail.

**2. The direct ensemble is inconclusive (13 paired).**

Diagnosis:

- **Measured:** 75 of 100 start on another SA-CASSCF solution. *(Task 10)* All 75 jump above 0.1 eV on SHARC's side, and none is kept; the 19 SHARC-kept and the 13 paired all share our start.
- **Inferred:** the jumps are solution switches that `SHARC_PYSCF.py`/`sharc.x` accept.

Options:

- (a) **Accept it as a report and close.** The drivers are compared by the swap test.
- (b) **Pursue a direct comparison**, for example with more initial conditions so that the same-start subset reaches 30 pairs. At the measured rate of 13 pairs per 100 that is about 231 trajectories. *Estimate:* about **32-36 h** of `SHARC_PYSCF.py` on this machine. The lower end uses the process time (13.7 h per 100); the upper end uses the wall clock (15.6 h per 100). About 4.5 h of ours comes on top (1 h 56 min per 100). *Inference:* a pass would then compare only the trajectories that do not switch.
- (c) A kept rule that accepts SHARC's jumps would need a new spec, because the kept rule is pre-registered, and it would compare trajectories on different solutions.
- (d) Making SHARC's interface start on our solution would need inputs to `SHARC_PYSCF.py` not known from the public manual. It was not attempted.

**3. The tau and t0 passes rest on wide intervals.**

- **Measured:** at N = 48 the point estimates differ by 39.65 fs (tau, 0.42 of the bound) and 21.77 fs (t0, 0.57). The two sides trade delay against lifetime (ours t0 12.63 / tau 93.55 fs, SHARC 34.40 / 53.89 fs).
- Options:
  - (a) accept the pass as measured;
  - (b) a larger paired set (cost per 100 more trajectories: about 2 h for ours plus 5 h 37 min for swap; the paired yield was 48 of 100);
  - (c) a different, pre-registered fit rule (a new spec).

**4. The N_min = 50 rule and the borderline N = 20.**

- **Measured:** coverage 0.895 at N = 20 (SE 0.0153), within one SE of 0.90. N_min = 50 follows from the fixed rule.
- Options:
  - (a) keep 50;
  - (b) measure N = 30 and 40, to locate the boundary (about 16 min per three sizes on this machine);
  - (c) refuse the fit below N_min (spec owner decision 4 chose a warning).
- Deferred minors that go with it:
  - the warning printed 0.927 for 0.9275 (fixed by the final review);
  - the spec's "else 100" fallback and its warning text are not coded (the constant was set by hand);
  - the coverage text is hard-coded (its test now asserts the whole text, final review).
- Also: the interval does not reach its nominal 0.95 at any tested N, and the median tau is biased low at small N (section 6). Options: accept, since the threshold rule is 0.90; or study another interval method in a later sub-project.

**5. The excluded fraction of 31-33 % in our runs (selection bias).**

- **Measured:** our analysis excluded 33 % (ours) and 31 % (ours-nacdr), with the excluded-fraction warning in both.
- *(Task 10, from `dynamics_summary.json`)* All 33 losses of ours (23 `overlap_failure`, 10 `electronic_discontinuity`) have at least one hop before their end. For ours-nacdr, 29 of 31 do; one `overlap_failure` and one `electronic_discontinuity` have no hop.
- This is the P9a pattern (losses after hops). *Inferred, not replayed in P9b:* the post-hop solution switch of P9a.
- Options:
  - (a) keep the P9a closure (not a work item);
  - (b) reopen the post-hop losses as a sub-project.
- The swap comparison is on the paired set only, so it says nothing about the bias.

**6. Smaller items for the owner.**

- *SHARC's frustrated hops are not recorded* in `output.dat`, so the frustrated-hop column is null (PR4). Accept, or parse `output.lis` (`# Jump frustrated.`; F1) in a later tool change.
- *`make_fit.py`'s bootstrap std is not reproducible* (no seed prompt). Accept.
- *The WSL VM stopped during a long run* (likely the idle timeout, not established). Option: `vmIdleTimeout=-1` in `.wslconfig` for runs of several hours (a machine setting; not set).
- *Our backend failed inside the swap provider once* (`traj_00085` at 76.5 fs). In our own run the same initial condition ended `overlap_failure` at 62.25 fs. *(report)*
- *The replay test's `raises=AssertionError` scope* (it once covered the hop, P(S0) and active-state assertions too) was **resolved by the final review**. Those checks moved to the passing pre-hop test, and each bound assertion of the xfail names its key.
- *The analysis' own lifetime fit is not compared with `make_fit.py`* (section 4: our side of the fit check is a tool-local fit). Option: drive `make_fit.py` against the stage's fit with the delay fixed at 0, or accept that the fit is validated on synthetic data only.

### Owner decisions (2026-11-18)

The owner took the recommended option of each item:

1. **Hop energy convention: (c), an option.** The current rule stays the default, and SHARC's convention becomes an option. The replay is rerun with the option on and asserted up to 20 steps after the hop. Our ensemble is rerun with the option as a report. Spec: `docs/superpowers/specs/2026-11-18-hop-energy-option-design.md`.
2. **Direct ensemble: (a)**, accepted as a report and closed.
3. **tau/t0 intervals: (a)**, the pass is accepted as measured.
4. **N_min: (a)**, it stays 50. The interval's coverage below its nominal 0.95 is accepted.
5. **Excluded fraction: (b)**, the losses after a hop are reopened as a sub-project, after the hop energy option.
6. **Smaller items:** all accepted as they are. The stage's lifetime fit stays validated on synthetic data only. `vmIdleTimeout=-1` was set in `.wslconfig` on 2026-11-18.

## 8. The honest summary

**What P9b shows:**

- Our driver and SHARC 4.1's driver, on the same electronic structure (swap), give ensembles that agree under every rule fixed before the runs: classical populations, delayed-fit tau and t0, and first-hop times, on 48 paired trajectories.
- Our analysis' population functions reproduce SHARC's `populations.py` exactly (classical) and to its print precision (quantum).
- A tool-local undelayed fit (`fit_undelayed`, not the stage's `analysis.kinetics`) gives `make_fit.py`'s lifetime to 0.14 fs (0.18 %, unexplained). The 36.93 fs bound of that check does not discriminate. So **the stage's lifetime fit is not validated against `make_fit.py`**.
- The tau interval of our analysis reaches the pre-registered 0.90 coverage threshold from N = 50. It stays below its nominal 0.95 at every tested N (3.6, 1.7 and 1.2 SE below at N = 20, 50, 100), and the median tau is biased low at small N (38.13 fs at N = 20 against 40).

**What it does not show:**

- The fit agreement is weak: the uncertainties at N = 48 are wide.
- The direct comparison with SHARC's own electronic structure is inconclusive: the start solutions differ in 75 of 100, and those 75 all jump.
- The replayed trajectory does not agree after its hop: a convention difference at the hop, measured on SHARC's side at every hop, inferred as a rule.

**Decided after numbers were seen** (none changed a bound, a seed or a run setting of the comparison):

1. **Ruling P4 (Task 7):** after the stop, ours-nacdr was resumed by a plain `gqteashmd run`, not by the chain's `--rerun-from dynamics`. That continued the partial dynamics instead of discarding 23 min of it. 16 trajectories carry `n_resumes` > 0; a PySCF resume is round-off identical (1e-13).
2. **Ruling P5 (Task 8):** the failed replay test became a strict xfail with its measured numbers. Its bounds, window and start were kept.
3. **Ruling P6 (Task 8):** a bounded diagnosis of the hop step was added: the energies offline, and one pair of runs with `decoherence = "none"`.
4. **Task 9 hop-energy check** over every swap hop, added by the controller after Task 8. It has its own definitions: a SHARC hop is a change of active state; for ours, "full consecutive step" excludes hop records in halved intervals. Nothing is asserted.
5. **Task 9 shadow projects:** the first comparison run paired 0 trajectories because of a tool defect. Shadow projects with symlinks gave the numbers. The invalid first run was discarded and is not reported as a result.
6. **Ruling P7:** the tool defect was fixed (`637a9f8`), and the comparisons were rerun on the real projects. The JSON is identical to the shadow runs except the path fields.
7. **Task 9 start-solution threshold:** 1e-3 eV, the tool's constant. It is justified only after the run, from the empty gap between 2.5e-6 and 0.139 eV; any threshold in the gap gives the same count.
8. **Task 6:** the warning wording was adapted after the numbers, to read with "coverage". N_min itself follows the fixed rule.
9. **Task 5:** `make_fit.py` runs twice (normal mode for tau, bootstrap mode for the std), and our side holds a trajectory's last values past its end, as `populations.py` does. Both were decided while the tool was written, before the Task 9 comparison but after the Task 1 observations.
10. **Task 10 (this report):** the location of the largest |dP|, the start split of the direct kept set, the aborted-against-jumped counts and the hops before our losses were read from the existing files. No calculation was run.
11. **The final review (after all numbers were seen):**
    - **The replay test was split.** A module fixture runs both sides once; a new passing test asserts the spec-5 preconditions, the same active state over the window and the four bounds over the records before the first hop; the full-window test keeps its bounds, window, start and settings as a strict xfail, with each bound assertion naming its key. The split tightens and adds; it loosens nothing. Run: 1 passed, 1 xfailed (the task-10 ledger report has the output).
    - **A tool change:** a missing bootstrap std in `run_sharc_analysis.py` is now `inconclusive` (PR5), where it was taken as 0. It did not occur, so `tools.json` is unaffected.
    - **The analysis warning text** prints 0.9275 instead of 0.927.
    - **The claims about `make_fit.py` and the coverage** were corrected (sections 0, 4, 6 and this one).

**Before any comparison, from the Task 1 observations** (not decisions after comparison numbers): the undelayed shared fit model (F4, because `make_fit.py` cannot express a delay), the quantum bound 1e-9 (F5) and the extractor and environment workarounds (F6).
