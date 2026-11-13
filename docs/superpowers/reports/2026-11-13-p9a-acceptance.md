# P9a acceptance report: the fixes of plan 2 and the seed-2027 run

Status: **the acceptance run missed the success rule: 13 of 20 trajectories kept, 16 needed** (spec section 2, rule 3). The shortfall is documented here; the threshold was not lowered and nothing was re-tuned on seed 2027. The merge of `feat/p9a-trajectory-losses` waits for the owner.

Date: 2026-11-13. Branch `feat/p9a-trajectory-losses`. The spec is `docs/superpowers/specs/2026-11-12-p9a-trajectory-losses-design.md`, the approved diagnosis is `docs/superpowers/reports/2026-11-12-p9a-diagnosis.md` (with its "Owner decisions (2026-11-13)", binding for plan 2), and plan 2 is `docs/superpowers/plans/2026-11-13-p9a2-fixes.md` (plan commit `f050839`). The code tasks are the commits `0631880` to `8c1e6f8`; Tasks 7, 10 and 11 are measurements with no diff, and Task 8 made no code change (ruling P9a2-7).

The numbers come from these sources:

- the task reports in the git-ignored ledger `.superpowers/sdd/2026-11-13-p9a2-fixes/` (`task-1-report.md` to `task-11-report.md`) and its `progress.md`, which holds the controller rulings P9a2-1 to P9a2-9;
- the JSON files they quote, on the WSL native filesystem of the development machine:
  - `~/p9a/plan2/convergence_probe.json` (the F4 probe);
  - `~/p9a/plan2/start_solutions.json` (the start-solution survey);
  - `~/p9a/plan2/forensics_sa3.json` and `~/p9a/plan2/replay_sa3/traj_*.json` (the `sa3_cas22` losses);
  - `~/p9a/plan2/replay_hop_to_end/traj_*.json` (the hop-to-end replays);
  - `~/p9a/plan2/variants/base_no_f1/traj_00014/` (00014);
  - `~/p9a/plan2/variants/sa3_cas22_final/summary.json` (the seed-2026 re-run);
  - `~/p9a/acceptance/run/06_dynamics/dynamics_summary.json`, `~/p9a/acceptance/run/07_analysis/analysis.json` and `~/p9a/plan2/forensics_acceptance.json` (the acceptance run).

Each statement names its source. A reading that is an inference says so.

**Definitions** are those of the diagnosis report: **kept** = included by the analysis (`finished` or `stopped_in_s0`) **and** no per-record $E_{\text{tot}}$ jump above 0.1 eV (`energy.flagged` of `analysis.json`; `run_variants.is_kept` in the variant runs); the labels **ours**, **method** and **settings** as in spec 2, rule 1, with the "several solutions" reading that owner decision 9 accepted.

## 0. Summary

- **Code that entered.** Three changes, each with a test that failed first:
  - the runner records **why** it rejects a step (`rejections_by_signal`) and **that** it resumed (`resumes`, `n_resumes`, an INFO line); output only;
  - **F1**, the energy-jump rejection signal: `[dynamics] energy_jump_limit_eV` (default 0.1 eV), the outcome `electronic_discontinuity`, and `DynamicsStage.version` 2;
  - **F4**, two extra CASSCF rungs behind a backend option that is off by default. **The probe did not meet the conditions fixed beforehand, so no stage enables them** (ruling P9a2-7).
- **No defect of ours was found** in the three candidate cases 00014, 00007 and 00016 (ruling P9a2-8), so no "ours" fix was inserted.
- **Seed 2026 with the final code** (`sa3_cas22_final`): **12 of 20 kept** (plan 1: 11). All 8 losses are `overlap_failure` after a hop.
- **Acceptance, seed 2027** (SA3-CASSCF(2,2)/6-31G\*, `states = ["S1"]`): **13 of 20 kept: a miss.** All 7 losses come after a hop (5 `overlap_failure`, 2 `electronic_discontinuity`), 6 of them on S0 after an S1 → S0 hop.
- **Cause of the shortfall.** Measured: every loss ends at $\Delta t/8$ after the halving ladder, 9.9-62.9 fs after the last hop. By inference (not replayed on seed 2027): the post-hop SA3-CASSCF solution switch that the seed-2026 replays reproduced in 4 of 5 `sa3_cas22` overlap failures. No fix for this class exists in plan 2.

## 1. What was run

Machine: the development workstation, WSL Oracle Linux 9.5, 16 cores (USER_MANUAL 5.9). Every measurement ran on the WSL native filesystem. SHARC 4.1 was not run in plan 2; its plan-1 outputs (`~/p9a/sharc/base`, `~/p9a/sharc/sa3_cas22`) were only read.

**Code tasks** (one commit each, review clean; Task 4 and Task 6 after one fix round):

| Task | Commit | What |
|---|---|---|
| 1 | `0631880` | `rejections_by_signal`, `resumes`; summary keys and the resume INFO line |
| 2 | `c444f53` | F1: `energy_jump_limit_eV`, `electronic_discontinuity`, `dynamics` version 2 |
| 3 | `e985d97` | F4: `EXTRA_RUNGS`, `PySCFBackend(casscf_extra_rungs=)`, off by default |
| 4 | `c3ee1f1`, fix `95fb881` | `tools/diagnosis/convergence_probe.py` |
| 5 | `28a6da3` | `tools/diagnosis/start_solutions.py` |
| 6 | `051c84f`, fix `ed2f3ef` | replays of variant runs and from the hop to the end; `variants_p9a2.toml` |
| 8 | none | F4 does not enter (ruling P9a2-7) |
| 9 | `8c1e6f8` | the deferred spec-5 test items and the plan 1 review minors |

**Measurements** (no diff):

| Measurement | Command (abridged) | Code | Wall time | Output |
|---|---|---|---|---|
| Task 7, Step 1: F4 probe | `tools/diagnosis/convergence_probe.py ~/p8_m4/run --sharc ~/p9a/sharc/base --out ~/p9a/plan2/convergence_probe.json` | `ed2f3ef` | 3 min 16 s | `convergence_probe.json` |
| Task 7, Step 2: start-solution survey | `tools/diagnosis/start_solutions.py ~/p8_m4/run --variant-dir ~/p9a/variants/sa3_cas22 --sharc ~/p9a/sharc/sa3_cas22 ... --workers 8` | `ed2f3ef` | 10 s | `start_solutions.json` |
| Task 7, Step 3: the `sa3_cas22` losses | `trajectory_forensics.py ... --trajectories ~/p9a/variants/sa3_cas22`; `replay_points.py ... --base-variant sa3_cas22 --all-lost traj_00007 traj_00008 --window 6 --workers 4` | `ed2f3ef` | a few seconds; 3 min 18 s | `forensics_sa3.json`, `replay_sa3/` |
| Task 7, Step 4: hop to end | `replay_points.py ~/p8_m4/run traj_00001 traj_00002 traj_00005 traj_00013 traj_00015 --from-first-hop --workers 5` | `ed2f3ef` | 3 min 47 s | `replay_hop_to_end/` |
| Task 7, Step 5: 00014 | `run_variants.py ~/p8_m4/run tools/diagnosis/variants_p9a2.toml --variant base_no_f1 --task traj_00014` | `ed2f3ef` | 2 min 7 s | `variants/base_no_f1/traj_00014/` |
| Task 10: seed-2026 re-run | `run_variants.py ~/p8_m4/run tools/diagnosis/variants_p9a2.toml --variant sa3_cas22_final --workers 8 --threads 2 --memory-mb 800` | `8c1e6f8` | 1602.6 s | `variants/sa3_cas22_final/` |
| Task 11: acceptance | `gqteashmd run ~/p9a/acceptance/input.toml` | `8c1e6f8` | 27 min 43 s | `~/p9a/acceptance/run` |

Steps 1, 3 and 5 of Task 7 ran together (at most 6 processes x 2 threads); Steps 2 and 4 ran alone. The variant `base_no_f1` is M4's setup with `energy_jump_limit_eV = 1.0e6` (F1 effectively off); `sa3_cas22_final` is `singlets = 3` with every `[dynamics]` key at its plan-2 default, F1 at 0.1 eV included (`tools/diagnosis/variants_p9a2.toml`).

## 2. The fixes that entered, and their tests

### 2.1 Recording rejection signals and resumes (Task 1, output only)

- `TrajectoryRunner` counts every rejected attempt by its signal (`electronic`, `overlap`, `energy`, `rung`) in `rejections_by_signal`, and appends the generation name to `resumes` at each resume (the runner also logs `resumed from restart generation <gen> at t = ... fs` on `gqteashmd.trajectory.runner`; a worker process configures no log handler, so that INFO record is not shown in a stage run).
- `dynamics_summary.json` gains, per row, `rejections_by_signal` and `n_resumes`, and in `totals` the summed `rejections_by_signal`; `dynamics_summary.md` a line "rejections by signal"; the stage logs one INFO line per resumed trajectory, `Stage dynamics: <task> resumed from restart generation(s) <gen>, ...`. `schema_version` stays 1.
- Tests: RED 4 failed (`KeyError: 'resumes'`, `KeyError: 'rejections_by_signal'`); GREEN in the targeted set (`task-1-report.md`). An existing test changed: a second invocation of a finished trajectory now resumes from its final generation and records it (`test_energy_drift_outcome`).
- Deferred minor (ledger): a resume followed by a crash before the next generation write is not persisted; the next resume records its own generation.

### 2.2 F1, the energy-jump signal (Task 2)

- **Rule.** After each (sub)step, $\lvert E_{\text{tot}} - E_{\text{tot}}(\text{previous accepted record})\rvert$ above `energy_jump_limit_au` is a rejection signal (`energy`), checked after the electronic and overlap signals and **before** the rung signal. It is rejected and halved like the others. At the smallest step it ends the trajectory with the new outcome `electronic_discontinuity`; a jump together with a rung change at the smallest step is therefore **not** accepted and flagged.
- **Input.** `[dynamics] energy_jump_limit_eV`, default 0.1 eV (fixed in the spec), > 0 and finite; part of the fingerprint. `RunSettings.energy_jump_limit_au` defaults to `None` (off), so library users outside the stage keep their behaviour.
- **Version.** `DynamicsStage.version` 1 → 2: a project with `dynamics` results needs `--rerun-from dynamics`.
- **Analysis.** `electronic_discontinuity` is excluded (it is not `finished` or `stopped_in_s0`) and counted in `outcomes` (test `test_an_electronic_discontinuity_is_excluded_and_counted`).
- **Tests.** RED: 5 failed (`unexpected keyword argument 'energy_jump_limit_au'`). New scripted faults `energy_jump` (every energy + 0.01 Eh) and `energy_jump_rung` (with a rung change, for the signal order). Fast tier after the change: **1182 passed, 20 skipped, 190 deselected in 716.63 s**; the live SHARC module `test_sharc_pyscf_dynamics.py` **18 passed in 596.97 s** (`task-2-report.md`).
- **The P7b step-24 run with F1** (spec 5, re-check): both couplings end `electronic_discontinuity` at `t_final_fs` 11.8125 after **23** completed grid steps, `rejections_by_signal` `{"energy": 4}`, message `|E_tot - E_tot(previous record)| = 0.9431 eV exceeds energy_jump_limit_eV = 0.1` at $\Delta t/8$. The plan asserted 24 grid steps; **ruling P9a2-4** changed the assertion after the number was seen (section 7).

### 2.3 F4, the extra CASSCF rungs (Tasks 3 and 8)

- **What exists.** `pyscf_casscf.EXTRA_RUNGS`, fixed by owner decision 5 before any probe number:
  - rung 4, `"previous orbitals, more iterations"`: the previous orbitals, 4 x the macro iterations, the nominal tolerances;
  - rung 5, `"loose tolerances"`: the previous orbitals or else the SCF orbitals, 3 x the macro iterations, `conv_tol` 1e-8, `conv_tol_grad` 1e-4 (each `max(user, rung)`, never tighter than the input).

  They are tried after the three standard rungs only when a caller passes `PySCFBackend(casscf_extra_rungs=...)` (also through `make_backend` and `stage_backend`). Tests: RED 4 failed; GREEN `tests/electronic` (not slow) 79 passed; fast tier 1182 passed, 20 skipped, 195 deselected in 801.19 s (`task-3-report.md`).
- **What does not exist.** The probe (section 3.1) did not meet the conditions of Task 4, so **Task 8 made no change: no stage passes the option**, the dynamics runs of Tasks 10 and 11 used the three standard rungs, and the code comment in `pyscf_casscf.py` that names "the dynamics stage, Task 8" as the caller describes a path that was not taken.

### 2.4 "Ours" fixes

None. Ruling P9a2-8: none of 00014, 00007 and 00016 is "ours" (section 3.5).

### 2.5 The deferred test items and the plan 1 review minors (Task 9; no `src/` change)

- **CASSCF energy consistency in the non-slow `pyscf` tier:** `test_ethylene_casscf_energy_stays_consistent`, guard 1e-3 Eh, set after the measured 2.682e-4 Eh of P7b test (a).
- **The finer dt pair** (slow): drift 9.916e-5 / 2.439e-5 Eh at 0.25 / 0.125 fs over 5 fs, **ratio 4.065** in [3, 5] (bounds fixed in the plan; P7b had seen 4.07 as a probe).
- **The kill test keeps its output** (`first_run.log`) and prints its recovery path; on this machine: *reconciled from result.json (the orphaned workers finished)* (killed after 25.0 s, second run 7.5 s). No assertion on the path, by design (both paths are correct).
- **Plan 1 review minors:** `run_sharc_pyscf_ensemble.sharc_settings` refuses the `[dynamics]` values it does not translate (and `fssh`); a timed-out SHARC run's summary row is built from its partial `output.dat` as a rerun would; the stub tests now check the arguments passed (`start.q`, `v`, state, masses, seed), and the stream test compares `run_job` with `DynamicsStage.run_task` on the canned LVC pipeline (identical `steps.log` bytes).
- Runs (`task-9-report.md`): `tests/diagnosis` + `tests/sharc_compare` (not `pyscf`) 129 passed, 1 skipped, 2 deselected; `test_pyscf_dynamics.py` (not slow) 5 passed, 2 xfailed; the two slow tests 2 passed in 117 s.

## 3. The Task 7 measurements (seed 2026)

### 3.1 The F4 probe

Per failure geometry and configuration (`convergence_probe.json`, `geometries`; a rung counts only if the energies **and** the active-state gradient succeed, ruling P9a2-6):

| Geometry | `default` | `more_iterations` | `loose` | `both` |
|---|---|---|---|---|
| `traj_00000_failed_step` | fails every rung | fails (+ rung 4) | `loose tolerances` | `loose tolerances` |
| `traj_00007_full_step` | fails every rung | fails | `loose tolerances` | `loose tolerances` |
| `traj_00016_sharc_17fs` | fails every rung | fails | `loose tolerances` | `loose tolerances` |

The loose-rung energies lie on SHARC's solution: 00007 at 0.5 fs +5.8e-5 / −5.7e-5 eV (S0 / S1) from SHARC's record; 00016 at 17.0 fs −1.437e-3 / +1.440e-3 eV, equal and opposite, so the SA average agrees to about 3e-6 eV (a throwaway script against `~/p9a/sharc/base/traj_<n>/output.dat`, `task-7-report.md` 1.1).

**Accuracy of the loose tolerances** (fresh solves at the nominal tolerances against `conv_tol` 1e-8, `conv_tol_grad` 1e-4; `accuracy`):

| Point | max \|dE\| (Eh) | max \|dg\| (Eh/bohr) |
|---|---|---|
| `traj_00000[-1]` | 7.131e-5 | 2.138e-3 |
| `traj_00007[-1]` | 1.324e-6 | 2.730e-5 |
| `traj_00016[-1]` | unusable: the nominal fresh solve fails every rung | — |
| `traj_00001[0]` | 1.092e-5 | 9.737e-5 |
| `traj_00002[0]` | 4.445e-6 | 1.980e-5 |

**`f4_decision`** (verbatim, `decision`): `enabled: []`; reasons "more_iterations converged with its rung at 0 of 3 (need 1)", "loose converged with its rung at 3 of 3 (need 2)", "4 of 5 accuracy points usable", "loose accuracy: max |dE| 7.13e-05 Eh (<= 1e-06), max |dg| 0.00214 Eh/bohr (<= 0.0001): False". The loose rung fails the energy bound at every usable point (by a factor of 71 at the largest) and the gradient bound at one (a factor of 21), independent of the unusable point. *Inference, not measured:* SA-CASSCF state energies are not variational, so their error is first order in the orbital gradient, and a 1e-6 Eh bound is about as tight as `conv_tol_grad` 1e-6 itself.

**A plan defect (ruling P9a2-9).** The accuracy point `traj_00016[-1]` (r37) is the point plan 1 had shown to fail on every fresh solve, so "5 of 5 usable" could not be met with this point list. It does not change the decision.

### 3.2 The start-solution survey (facts only; the rule is owner decision 3)

`start_solutions.json`; SA3-CAS(2,2)/6-31G\*, the 4 guesses of `guess_pairs` (0-based active pairs (6,8), (6,9), (7,8), (7,9)), each solved fresh at the 20 starts and at the equilibrium geometry:

- all 84 solves converged; every start has **4 distinct solutions**, one per guess;
- "sol0", the (7,8) HOMO/LUMO solution, is the lowest in SA energy **and** the most like the equilibrium solution in **20 of 20** starts (min active-orbital singular value 0.950-0.997); the other three lie 1.75-9.05 eV higher in SA energy, with min active-orbital singular values 0.0016-0.154 (another active-orbital character);
- **our `sa3_cas22` run started on sol0 in 20 of 20** (to at most 5.8e-12 eV);
- **SHARC started on sol0 in 7 of 20** (to at most 1.35e-6 eV); in the other **13** its solution is **none of the four** (0.176-0.372 eV from sol0 in some state), with an SA energy above ours in 8 and below in 5 (−0.0062 to −0.1136 eV);
- all 11 trajectories kept by plan 1's `sa3_cas22` started on sol0; 6 of them (00000, 00004, 00012, 00014, 00018, 00019) are starts where SHARC used a solution the survey did not find.

So the fixed guess set cannot separate "lowest SA energy" from "most like equilibrium" (they coincide everywhere), and it does not reach SHARC's second solution in 13 of 20. Wall time per solve 0.14-1.05 s.

### 3.3 The nine `sa3_cas22` losses, replayed (owner decision 7)

`forensics_sa3.json`, `replay_sa3/traj_*.json`; the replay's `base` solves use the run's own SA3 settings (`--base-variant sa3_cas22`). Determinism pairs at most 3.6e-12 eV in all 9.

| Trajectory | Outcome (plan 1) | Replay finding | Class (spec 3.2 patterns) |
|---|---|---|---|
| 00003 | `overlap_failure` 90.5625 fs | continued = fresh at the failed step, state sv 0.2872 (run 0.287) | solution switch, failure reproduced |
| 00005 | `overlap_failure` 60.6875 fs | sv 1.211e-4 (run 1.21e-4) | switch, reproduced |
| 00011 | `overlap_failure` 96.0 fs | sv 0.01708 (run 0.0171) | switch, reproduced |
| 00013 | `overlap_failure` 68.125 fs | sv 0.3929 (run 0.393); its first jump (r71) has all modes agreeing | switch, reproduced; r71 a feature of the SA3 surface |
| 00016 | `overlap_failure` 91.375 fs | continued sv 0.9922 at the failed step (run 0.0574) | two solutions coexist from r181; failure **not reproduced** |
| 00006 | `energy_drift` 40.25 fs | continued ≠ fresh at the jump (0.65 eV), but continued does not land on fresh | switch pattern |
| 00009 | `energy_drift` 59.0 fs | equal to r116; at r117 0.0995 eV per state, −0.0001 eV in SA energy | switch pattern |
| 00007 | `finished`, jump 0.1178 eV at the hop record | all modes agree to 3.1e-6 eV | hop-record jump |
| 00008 | `finished`, jump 0.1448 eV at the hop record | all modes agree to 2.3e-6 eV | hop-record jump |

- In the reproduced overlap failures the run sits on the **higher** solution in SA energy where fresh differs (00005 +0.83 to +1.35 eV, 00011 +0.50, 00016 +0.34 to +0.59 eV), and that solution disappears at the failed step: the plan-1 SA2 overlap-failure pattern, now in SA3.
- The hop-record jumps of 00007 and 00008 are persistent $E_{\text{tot}}$ steps (+0.0845 and −0.1534 eV against $E_{\text{tot}}(0)$ at the hop record, then +0.078 to +0.083 and −0.149 to −0.170 eV), after S1-S0 gaps of 0.071 and 0.097 eV, with $E_{\text{tot}} = E_{\text{active}} + E_{\text{kin}}$ to 1.9e-13 eV (from `~/p9a/variants/sa3_cas22/traj_0000{7,8}/steps.log`). *Inference, not measured:* a Verlet error on S1 across a near-degeneracy with S0; its dt dependence was not measured.
- No case met "all variants agree" in the spec's full sense. **The class labels were assigned after the numbers were seen**, by the plan-1 patterns.

### 3.4 The five unreproduced base overlap failures, from the hop to the end

`replay_hop_to_end/traj_*.json`; determinism at most 2.7e-12 eV.

| Trajectory | Records | Continued vs stored | Failed step: continued sv (run) | Label |
|---|---|---|---|---|
| 00001 | 59 | ≤ 1.03e-5 eV at every record | 0.4399 (0.44) | method |
| 00002 | 14 | ≤ 6.1e-6 eV | 0.07676 (0.0768) | method |
| 00013 | 15 | ≤ 1.47e-5 eV | 0.1919 (0.192) | method |
| 00015 | 18 | ≤ 1.15e-5 eV | 0.1706 (0.171) | method |
| 00005 | 24 | 0.32-3.59 eV (the first point, from SCF orbitals, lands on another solution) | 0.9993 (0.343) | not established |

In the four reproduced cases the run does not switch: the continued solve follows it (active-orbital sv ≥ 0.93, state sv ≥ 0.92). A second solution appears from SCF orbitals a few records after the hop, first below or just above the run's in SA energy; the two cross, so the run ends on the higher one (+1.1 to +2.1 eV in SA energy at the maximum) until it disappears at the failed step. This is the plan-1 mechanism of 00008, 00017 and 00018, now reproduced from the hop. **The labels were assigned after the numbers were seen.**

### 3.5 The three candidate "ours" cases (owner decision 6)

- **00014: method/settings.** `base_no_f1` reproduces M4 (`energy_drift` at 11.0 fs, `n_rejections` 4, `max_drift_eV` 0.77331754824226 against 0.77331754823723; over 30 `steps.log` rows max |dq| 8.5e-12 bohr, |dE| 4.2e-11 eV). `rejections_by_signal` `{"overlap": 2, "rung": 2}`. The 1.0-1.5 fs full step converged on the first rung with the energy check off, so by elimination the **overlap** signal rejected it (its singular value is not logged; "below 0.5" is inferred). Its state energies (`electronic.log` lines 634-635) equal SHARC's 1.5 fs record to −2.2e-6 / +1.4e-6 eV: our solver lands on the same second SA2 solution as SHARC, and the rejection refused the switch as designed; our accepted halved record at 1.5 fs is −0.543 / +1.003 eV from SHARC. The programs part because ours rejects and SHARC does not. *This reading was made after the numbers were seen.*
- **00007: not established as ours.** At the 0.5 fs full-step geometry the default ladder and rung 4 fail; rung 5 converges on SHARC's solution (section 3.1). No cause in our code was identified, so there is nothing to test against.
- **00016: method/settings for the failure; the 17.0 fs disagreement not established as ours.** The failure is shared (plan 1); at SHARC's 17.0 fs geometry only rung 5 converges, on SHARC's SA energy. The run's own convergence at r37 stays unreproduced.

Ruling P9a2-8: none is "ours"; SHARC's interface plausibly accepts a looser effective criterion (black box, not shown), and 00007 stays documented as "not established".

## 4. Seed 2026 with the final code (Task 10)

`~/p9a/plan2/variants/sa3_cas22_final/summary.json`; the same 20 seed-2026 initial conditions and rng streams as plan 1; F1 at 0.1 eV, no F4 rung.

- **Kept: 12 of 20** (plan 1's `sa3_cas22`: 11). Outcomes: `finished` 12, `overlap_failure` 8, `energy_drift` 0, `electronic_discontinuity` 0, `electronic_failure` 0. Onsets: `failure_after_hop` 8, none 12.
- `rejections_by_signal` totals: overlap 32, energy 10 (00006 1, 00007 1, 00008 1, 00009 3, 00013 4); electronic 0, rung 0. No resume.
- **Path comparison** (a throwaway script on the two `steps.log` files): 14 trajectories follow plan 1's path to round-off (max |dq| 1.4e-12 to 9.4e-10 bohr) with identical outcomes; 00017 crosses 1e-6 bohr at 84.0 fs with no rejection in either run (round-off growth, inferred; outcome unchanged); **5 diverge exactly at their first `energy` rejection**.
- **The five changed trajectories** (which attempt carried which signal is inferred from the totals and the record depths; only totals are recorded):

  | Trajectory | Plan 1 | Plan 2 | Reading |
  |---|---|---|---|
  | 00006 | `energy_drift` 40.25 fs (jump 1.101 eV) | `overlap_failure` 40.1875 fs (sv 0.0118) | F1 refuses the jump; the smaller steps meet the solution change as an overlap collapse |
  | 00007 | `finished`, jump 0.1178 eV at the hop, not kept | `finished`, max jump 0.0632 eV, **kept** | F1 rejects the hop step; hop at 17.75 fs with a 0.0335 eV jump; the +1 of the count |
  | 00008 | `finished`, jump 0.1448 eV, not kept | `overlap_failure` 98.8125 fs (sv 0.19) | path changed after the F1 rejection; lost either way |
  | 00009 | `energy_drift` 59.0 fs | `overlap_failure` 61.875 fs (sv 0.287) | F1 refuses the jumps, the overlap signal ends it |
  | 00013 | `overlap_failure` 68.125 fs | `overlap_failure` 71.0625 fs (sv 0.133) | path changed after the F1 rejection |

- **No trajectory ended `electronic_discontinuity`:** where F1 fired, the halved retries either passed the jump test or met the overlap signal first (the overlap test runs before the energy test). Wall time 1602.6 s (plan 1: 1452.1 s).
- The diagnosis report expected F1 to change the kept count "probably" not at all (a reasoning, section 5 there). Measured: +1, through a hop-record jump (00007), a case the report had not linked to F1.
- The Task 10 rule ("a kept count far below 11 is reported before Task 11") did not apply; the controller recorded that 12 is not far below 11 and the acceptance run proceeded.

## 5. The acceptance run on seed 2027 (Task 11): a miss

### 5.1 Setup

`~/p9a/acceptance/input.toml` is the M4 input with exactly four lines changed (`diff` in `task-11-report.md`): `name = "ethylene_p9a_acceptance"`, `seed = 2027`, `[electronic] singlets = 3`, `[selection] states = ["S1"]`. Everything else is M4's: ethylene `ethylene.xyz` (byte-identical to M4's copy), SA-CASSCF(2,2)/6-31G\* at the default tolerances, `n_samples` 100, `n_trajectories` 20, `tmax_fs` 100, `dt_fs` 0.5, `coupling = "overlap"`, `hopping = "sharc"`, EDC, `energy_drift_limit_eV` 0.5, `overlap_threshold` 0.5, F1 at its default 0.1 eV, 16 cores, 7000 MB, 8 tasks x 2 threads, 800 MB per task, no `[analysis]` table. Code `8c1e6f8`. One `gqteashmd run`, started 10:40:56 and ended 11:08:39 (2026-11-13, −03:00): **27 min 43 s**, no restart and no error.

| Stage | Wall | Tasks |
|---|---|---|
| prepare | 3 s | 1 |
| optimize | 10 s | 1 (converged, GAU) |
| frequencies | 25 s | 37 (12 real modes, 887.4-3425.6 cm^-1, no imaginary mode, no warning) |
| sampling | 2 s | 1 |
| excitations | 39 s | 101 (100 of 100 samples OK; S1 9.979 ± 0.318 eV, f 0.516; S2 15.064 ± 0.759 eV, f 1.3e-4) |
| selection | 2 s | 1 (100 candidates, 92 accepted, 20 selected, all S1; no weight outlier) |
| dynamics | 26 min 11 s | 20 |
| analysis | 7 s | 1 |

### 5.2 Result

From `dynamics_summary.json` and `analysis.json`:

- **Outcomes:** `finished` 13, `overlap_failure` 5, `electronic_discontinuity` 2, `energy_drift` 0, `electronic_failure` 0, `stopped_in_s0` 0. Totals: 45 hops, 0 frustrated, 35 rejections (overlap 28, energy 7), 0 flagged steps; largest drift 0.272 eV (`traj_00007`). No trajectory resumed.
- **Analysis:** 13 included (traj_00000, 00002, 00003, 00004, 00006, 00009, 00011, 00012, 00014, 00015, 00016, 00017, 00019), **`energy.flagged` empty**, `small_ensemble: true`; the two warnings *only 13 trajectories are included (fewer than 20) ...* and *35% of the selected trajectories are excluded (more than warn_excluded_fraction = 0.2) ...*. Drift of the included: max 0.218, median 0.035, p95 0.115 eV. Fit (classical, S1 + S2): $t_0$ = 11.22 fs [3.38, 57.89], $\tau$ = 75.20 fs [25.34, 135.30], no flag, 0 bootstrap failures. Hop pairs (`hops.pairs`, the included trajectories): S1→S0 19, S0→S1 10, S1→S2 3, S2→S1 3.
- **Kept = 13 − 0 flagged = 13 of 20. MISS** (16 needed).

### 5.3 The seven losses and the cause of the shortfall

| Task | Outcome | $t_{\text{final}}$ (fs) | Message (end) | Hops | Active at end |
|---|---|---|---|---|---|
| 00001 | `overlap_failure` | 62.1875 | sv 0.478 at $\Delta t/8$ | S1→S2 10.5, S2→S1 30.5, S1→S2 45.0, S2→S1 46.5 | S1 |
| 00005 | `electronic_discontinuity` | 88.25 | 0.221 eV jump at $\Delta t/8$ | S1→S0 49.0 | S0 |
| 00007 | `overlap_failure` | 72.6875 | sv 0.0806 | S1→S0 17.0 | S0 |
| 00008 | `electronic_discontinuity` | 88.375 | 0.113 eV jump | S1→S0 25.5 | S0 |
| 00010 | `overlap_failure` | 89.125 | sv 0.371 | S1→S0 76.0 | S0 |
| 00013 | `overlap_failure` | 70.9375 | sv 0.147 | S1→S0 25.5 | S0 |
| 00018 | `overlap_failure` | 52.4375 | sv 0.195 | S1→S0 42.5 | S0 |

**Measured** (`forensics_acceptance.json`, `dynamics_summary.json`):

- every loss comes **after a hop**; 6 of 7 are on S0 after an S1 → S0 hop, the seventh (00001) on S1 after S2 → S1;
- each ends at the smallest step $\Delta t/8$, after the halving ladder, **9.9 to 62.9 fs after the last hop** (00018 9.9, 00010 13.1, 00001 15.7, 00005 39.3, 00013 45.4, 00007 55.7, 00008 62.9);
- the last accepted records have smallest singular values of 0.87-1.00: the drop below 0.5, or the jump above 0.1 eV, is in the rejected step;
- at the last accepted record S1 lies 5.9-7.4 eV above S0 in 5 of the 6 S0 losses (00013: 3.5 eV), so these trajectories are deep in a hot ground state, not near a crossing;
- no loss before a hop, no `energy_drift`, no `electronic_failure`: the SA2 classes of M4 are absent, as on seed 2026.

**Inference, not measured:** the losses are the post-hop SA3-CASSCF solution switch in the hot ground state that the seed-2026 replays reproduced in 4 of 5 `sa3_cas22` overlap failures (section 3.3) and, for SA2, in 7 of the 8 base overlap failures (plan 1 and section 3.4). The overlap signal sees it in 5 cases and F1 in 2. The seed-2027 losses were **not replayed**, and no intruder state above S2 was checked. The forensics onset is "none" for the two discontinuities only because the onset classes predate the new outcome (ledger minor).

**Why plan 2 could not close the gap.** On seed 2026 every loss of the final code was already of this class (8 of 8, section 4). Plan 2 had no fix for it: F1 is detection (it converts drifts into explicit endings and, once, rescued a hop step), F2 (re-solving from continuous orbitals) was dropped by owner decision 4 and its re-solve part had been ruled out in plan 1 (the continued solve itself lands on the other solution), and F4 did not enter.

## 6. For the owner: the decisions left open

1. **The shortfall.** The success rule (16 of 20) is missed with 13. Spec 2, rule 3 says the shortfall is documented, not tuned away. Open: whether P9a closes with this documented miss (and the branch is merged), or whether the post-hop loss class becomes its own work item (P9b or a new sub-project), with a new acceptance run on a new seed.
2. **The start-solution rule (owner decision 3).** The survey facts (section 3.2): with the fixed 4-guess set, every start has 4 solutions; the lowest-SA and the most-equilibrium-like solution coincide in 20 of 20 and our runs always start on it; SHARC starts on another solution, found by no guess, in 13 of 20 (SA energy above ours in 8, below in 5); all 11 plan-1 kept trajectories started on our solution. A rule needs a criterion that these data cannot give (the guess set never separates the candidate criteria). Options: no rule; a rule with a criterion of the owner's choice, measured on seed 2026 first; a wider guess set first.
3. **The dormant F4 rungs.** They are in the backend, tested, and enabled by no stage. Keep them as a library option (documented as such) or remove them.
4. **F1's default.** 0.1 eV was fixed in the spec. On seed 2026 F1 fired on 5 trajectories (10 rejections) and ended none; on seed 2027 it fired on 4 (00005 1, 00007 1, 00008 4, 00019 1: 7 rejections) and ended 2; no measurement of the false-positive rate on correct runs beyond the P8 Task 1 bound (7.7e-3 eV largest per-step jump) exists for SA3.
5. **Items raised by Tasks 7-11, not decided:**
   - the hop-record jumps (00007, 00008 of plan 1): their dt dependence was not measured; F1 now rejects them, which rescued 00007 on seed 2026;
   - `sa3_cas22` 00016's overlap failure is not reproduced by the replay (sv 0.9922 against the run's 0.0574);
   - base 00005 cannot be replayed from its hop; a replay that starts before the hop is needed to label it;
   - 00007 and 00016: SHARC's interface converges at nominal tolerances where our solver needs `conv_tol_grad` 1e-4 (not established as ours, ruling P9a2-8);
   - the acceptance trajectory traj_00019 is kept with a drift of 0.218 eV and one energy rejection: the kept rule looks at per-record jumps, not at the drift;
   - the excitations of seed 2027 have S2 at 15.06 eV with f 1.3e-4, so `states = ["S1"]` excluded nothing in practice; the default selection with `singlets = 3` was not measured.
6. **The merge** of `feat/p9a-trajectory-losses` (after the full suite and the final review, which the controller runs after this report).

## 7. Honest summary: what was decided after the numbers were seen

- **Ruling P9a2-1** (before any number, pre-flight scan): Task 6's CLI test must assert the new options in the `--help` text, not only that `--help` exits. A stronger test than the plan's.
- **Ruling P9a2-4** (after the live number): the plan asserted `n_grid_steps == 24` for the step-24 F1 run; measured 23. The assertion became `n_grid_steps == CHANGE_STEP - 1` plus $(\text{CHANGE\_STEP} - 1)\,\Delta t < t_{\text{final}} < \text{CHANGE\_STEP}\,\Delta t$ (11.5 < 11.8125 < 12.0 fs), and its docstring says it was changed after the measurement. Reason: with F1 the jumping 24th interval is rejected, so 23 grid steps complete; the old `energy_drift` test counted the accepted jump step.
- **Ruling P9a2-5:** the plan's tests used a `load_tool(name)` fixture that does not exist; the tests use the repository's `load_tool(subdir, name)` with the plan's assertions unchanged.
- **Ruling P9a2-6** (after review, before the probe ran): a probe configuration counts as converged only if the active-state gradient also succeeds. This made the F4 condition stricter than the plan's text.
- **Ruling P9a2-7** (after the probe): F4 does not enter. The conditions were fixed beforehand; the ruling applies them, and states that the energy and gradient bounds fail independent of the unusable fifth point.
- **Ruling P9a2-8** (after the measurements): none of 00014, 00007, 00016 is "ours", so no fix was inserted. The 00014 reading (the overlap signal working as designed) and the 00007/00016 reading (SHARC plausibly accepts a looser effective criterion; black box) were made after the numbers.
- **Ruling P9a2-9:** the probe's accuracy set included a point plan 1 had shown to fail fresh (a plan defect); left as is.
- **The class labels** of sections 3.3 and 3.4 (switch, surface feature, hop-record jump; method / not established) were assigned after the numbers were seen, by the plan-1 patterns, with the "several solutions" reading of owner decision 9.
- **The Task 9 guard** of 1e-3 Eh was set after the measured 2.682e-4 Eh (stated in its docstring); the finer-pair bounds [3, 5] were fixed in the plan, after P7b had seen 4.07 as a probe.
- **The go-on after Task 10** (12 is "not far below 11") was a controller judgment on a rule that named no number.
- **Not tuned:** the acceptance setup is owner decision 1 and 8 as written before Task 10; nothing was changed after the seed-2026 re-run, and nothing was re-run on seed 2027.
