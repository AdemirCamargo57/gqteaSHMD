# P9a diagnosis report: why the M4 study lost 18 of 20 trajectories

Status: **for the owner's approval** (spec section 3.5). Plan 2 (fixes and the acceptance run) is written only after this report is approved.
Date: 2026-11-12. Branch `feat/p9a-trajectory-losses`, from `main` at `4f076a5`. The spec is `docs/superpowers/specs/2026-11-12-p9a-trajectory-losses-design.md` (`e09cd79`) and plan 1 is `docs/superpowers/plans/2026-11-12-p9a1-diagnosis.md` (`722b393`). Tasks 1-5 built the tools (`ab4cb04` to `58811f6`). Tasks 6 and 7 ran the measurements; they have no diff. The final fix wave, after the whole-branch review, corrected this report and added the measurements listed in section 7. Nothing under `src/` changed in plan 1.

The numbers come from these sources:

- the task reports in the git-ignored ledger `.superpowers/sdd/2026-11-12-p9a1-diagnosis/` (`task-6-report.md`, `task-7-report.md`);
- the JSON files they quote, which live on the WSL native filesystem of the development machine:
  - `~/p9a/forensics_m4.json`;
  - `~/p9a/replay/traj_*.json`;
  - `~/p9a/variants/<variant>/summary.json`;
  - `~/p9a/sharc/base/sharc_summary.json` and `~/p9a/sharc/sa3_cas22/sharc_summary.json`;
  - the final fix wave's measurements, made after the final review with throwaway scripts that call the tools' functions. The scripts and their JSON are in `~/p9a/final_fix/`, not in the repository:
    - `i1_traj00007.json` and `i1b_trials.json`: the first full step of 00007;
    - `i2b_traj00016.json`: our backend at SHARC's 00016 geometries;
    - `i23_compare.json` and `i23b_paths.json`: ours against SHARC at every common full step of the 20 trajectories, and the replay against the stored energies.

Each statement names its source. A number that was estimated, or a reading that is an inference, says so.

**Definitions used throughout.**

- **Kept** (spec 4.3) means: outcome `finished` or `stopped_in_s0`, **and** a largest per-record jump `|ΔE_tot|` of at most 0.1 eV. In the variant runs this is `run_variants.is_kept`, computed over the `output.h5` records with `grid_step <= n_total`. That is the same set of records the analysis uses for `energy.flagged` when `output_every = 1`.
- **Labels** (spec 2, rule 1):
  - **ours**: our result differs from an independent reference, either SHARC 4.1 on the same initial conditions or a fresh CASSCF solve at the same geometry;
  - **method**: the electronic-structure method at these settings behaves this way in both programs, or in every solve;
  - **settings**: a change of a setting removes the loss.

  When the two references disagree, the report says so, and the case goes to owner decision 6.
- **Several solutions: how the fresh-solve test is read.** This reading was adopted **after the numbers were seen** (section 7; owner decision 9).
  - Read literally, the fresh-solve test is met in most lost trajectories: a fresh solve often lands on another SA-CASSCF solution than the run (sections 2.1 and 2.2).
  - The report does not count such a case as "ours" when the run's solution is itself reproduced, by a replay continued from the previous orbitals or by SHARC at the same geometry. Spec 3.2 supports this reading: "(i) and (ii) disagree = a solution switch along the path".
  - A case where SHARC converges and our solver does not, at the same geometry, is not covered by this reading (00007, 00016; section 2.3).

## 0. Summary

- **No measured setup reached 16 of 20 on the seed-2026 initial conditions.** The best is SA3-CASSCF(2,2), with 11 of 20.
  - The success rule itself (at least 16 of 20 kept) is judged on the seed-2027 acceptance run, with the fixed code of plan 2. Plan 1 cannot test it.
  - The variant table below is the evidence that the rule is at risk.

  | Setup | Kept of 20 |
  |---|---|
  | base (M4 as run) | 0 |
  | `coupling = "nacdr"` | 0 |
  | `dt_fs` 0.25 | 1 |
  | SA3-CAS(2,2) | **11** |
  | SA3-CAS(2,3) | 0 |
  | SA2-CAS(4,3) | 0 |
  | SHARC 4.1, base | 1 |
  | SHARC 4.1, SA3-CAS(2,2) | 2 |

  Source: `~/p9a/variants/*/summary.json` `n_kept`, and Task 7 for SHARC.
- **One mechanism dominates.** The SA2-CASSCF(2,2)/6-31G\* problem has more than one solution along these paths, and the run moves from one to another. The two rows below give the reproduction; section 2 gives the label of each class.
  - The `energy_drift` class is this switch in 6 of 7 trajectories: Task 6 reproduced it by replay, and SHARC_PYSCF makes the same jump at the same step (Task 7). In the seventh, 00014, the terminal jump was not replayed and SHARC differs.
  - The `overlap_failure` class is the same switch after the hop in 3 of 8 (reproduced). The other 5 have indirect evidence only.
- **No intruder state was seen.** In SA3-CAS(2,2), S2 lies at least 0.589 eV above S1 in every overlap-failure window and at least 1.973 eV above it in the drift windows (Task 6 sections 4.1 and 4.2). The working hypothesis of spec section 1, "a state outside the computed set", is therefore not supported.
- **The overlap failures do not come "right after" the hop.** They come 5.8-42.9 fs after it (Task 6 table 2.4).
- **The `electronic_failure` class.** In all three, CASSCF stops at the 150-macro-iteration limit with a stationary energy.
  - **00000:** a fresh solve at the failed geometry converges when `conv_tol_grad` is looser.
  - **00007:** at the geometry of our rejected first full step (0.5 fs), `SHARC_PYSCF.py` returns a converged result. At the same geometry and the same nominal tolerances, our solver fails on every rung, both continued and fresh. The two references of the "ours" definition disagree.
  - **00016:** both programs converge at 17.0 fs and fail on the next QM call, so this pattern is shared. But at SHARC's own 17.0 fs geometry our solver also fails where SHARC converged, which is the 00007 disagreement again.
- **No defect of our program has been established.** These cases come closest:
  - **00014:** the two programs part at 1.0-1.5 fs, where our run rejected and halved a full step that SHARC took in one; SHARC jumps by -0.994 eV there;
  - **00007 and 00016:** SHARC's interface converges at a geometry where our three rungs fail (section 2.3);
  - **a detection gap in our design:** none of our rejection signals sees a solution switch of 0.64-0.93 eV;
  - **the literal fresh-solve test** is met in most lost trajectories. The report reads it as several valid solutions, not as a defect; that reading was made after the numbers (Definitions; decision 9).

## 1. What was run

Machine: the development workstation, WSL Oracle Linux 9.5, 16 cores, 7754 MB RAM (USER_MANUAL 5.9). Every run was on the WSL native filesystem. The code is at `58811f6` (the scripts of Tasks 1-5); the program code is that of `main`. The input is the M4 project `~/p8_m4/run` (run A of P8 Task 11, code `2631eff`), which was only read. SHARC 4.1 (`/home/ajc/sharc4-4.1`) was executed only: `sharc.x` with `SHARC_PYSCF.py`, under the clean-room rule.

| Measurement | Command (abridged) | Wall time | Output |
|---|---|---|---|
| Forensics (Task 6, Step 1) | `tools/diagnosis/trajectory_forensics.py ~/p8_m4/run --out ~/p9a/forensics_m4.json` | seconds (file dated 13:55:19) | `forensics_m4.json` |
| Replay (Task 6, Step 2) | `tools/diagnosis/replay_points.py ~/p8_m4/run --all-lost --out ~/p9a/replay --window 6 --workers 4 --threads 2 --memory-mb 1500` | 5 min 06 s for 18 trajectories (from the file mtimes; the `END` line of `replay.log` is wrong) | `replay/traj_*.json`: 4 variants x 3 modes x 134 points = 1608 solves, plus determinism pairs and convergence trials |
| Our runner, 6 variants (Task 7) | `tools/diagnosis/run_variants.py ~/p8_m4/run tools/diagnosis/variants_p9a.toml --out ~/p9a/variants --variant <v> --workers 8 --threads 2 --memory-mb 800` | 336-1452 s per variant (table in section 4) | `variants/<v>/summary.json` |
| SHARC, base (Task 7) | `tools/sharc_compare/run_sharc_pyscf_ensemble.py ~/p8_m4/run tools/diagnosis/variants_p9a.toml --variant base --out ~/p9a/sharc --workers 8` | 4565 s | `sharc/base/sharc_summary.json` |
| SHARC, sa3_cas22 (Task 7) | the same command with `--variant sa3_cas22` | 10549 s (estimated beforehand at 2.1 h) | `sharc/sa3_cas22/sharc_summary.json` |
| Final fix wave (after the final review) | `~/p9a/final_fix/i1_traj00007.py`, `i1b_trials.py`, `i2b_traj00016.py` (calls of `replay_points.solve_path`, `_active_gradient`, `convergence_trials`; 23 single points, 2 of them with a gradient, 2 threads); `i23_compare.py`, `i23b_paths.py` (existing files only) | under a minute each, not timed | `~/p9a/final_fix/*.json` |

Task 7 took 6 h 05 min in total (16:17:41-22:22:14). The probes ran first, with `tmax_fs = 4` in a scratchpad copy of the variants file; their output was deleted.

**What the variant runs fix and what they vary.**

- **Fixed in every variant** (by design, spec 3.3): the 20 initial conditions (the M4 `samples.h5` and `trajectories.json`), the initial state S1, and the rng streams `task_rng(2026, "dynamics", k)`. The runs call `TrajectoryRunner` directly, not the orchestrator.
- **Varied** (spec 3.3, fixed before the runs in `tools/diagnosis/variants_p9a.toml`):

  | Variant | What changes from base |
  |---|---|
  | `base` | nothing (M4 as run) |
  | `nacdr` | `coupling = "nacdr"` |
  | `dt_quarter` | `dt_fs = 0.25` (despite its name, this is half the M4 step) |
  | `sa3_cas22` | `singlets = 3` |
  | `sa3_cas23` | `singlets = 3`, `ncas = 3`, `nelecas = 2` |
  | `sa2_cas43` | `ncas = 3`, `nelecas = 4` |

  No `[dynamics]`-threshold variant was added.

**The script check (gate).** The `base` variant reproduces M4 in all 20 trajectories: the same outcome and the same `t_final_fs` to 4 decimals, the same singular values in the termination messages, and the same drifts (Task 7, Step 1).

## 2. The loss classes

### 2.1 `energy_drift` (7: 00004, 00006, 00009, 00011, 00012, 00014, 00019)

**Evidence.**

- *Forensics* (`forensics_m4.json`, Task 6 table 2.3):
  - every terminal jump is +0.64 to +0.93 eV in E_tot, made by the active state S1, which rises by +0.66 to +1.20 eV;
  - at the jump the S1-S0 gap is 3.4-5.1 eV and the smallest state-overlap singular value is 0.72-0.82, above `overlap_threshold` 0.5;
  - the CASSCF rung did not change.

  So **no rejection signal fired**; only the cumulative drift limit stopped the run.
- *Replay* (`replay/traj_*.json`, Task 6 section 4.1):
  - The continued base solve reproduces the stored energies to at most 9.9e-6 eV across the jump, and the run's state singular value to four digits (00004: 0.7723447 against 0.7723450).
  - 1-5 records before the jump, a fresh or backward solve finds a **second SA2-CAS(2,2) solution**, 0.08-1.01 eV away per state.
  - At the jump record, continued equals fresh to at most 1.03e-5 eV, so the run has landed on that second solution.
  - In SA-average energy the run's solution was the lower one at the first point where the other solution appears (all 7). At the last point before the jump it was the higher one in 5 of 7 (+0.0118 to +0.0576 eV).
- *SHARC on the same initial conditions* (`sharc/base/sharc_summary.json`, Task 7 section 4). In **6 of 7** trajectories, SHARC makes the identical jump at the identical time:

  | traj | ours: t (fs), ΔE_tot (eV) | SHARC: t (fs), ΔE_tot (eV) |
  |---|---|---|
  | 00004 | 11.0, +0.8257 | 11, +0.826 |
  | 00006 | 8.5, +0.6861 | 8.5, +0.686 |
  | 00009 | 10.0, +0.9319 | 10, +0.932 |
  | 00011 | 11.0, +0.6408 | 11, +0.641 |
  | 00012 | 7.5, +0.6809 | 7.5, +0.681 |
  | 00019 | 6.5, +0.7691 | 6.5, +0.769 |

  In these 6, ours and SHARC agree to at most 3.2e-5 eV in the state energies, and to at most 5.6e-5 eV in E_tot, at every common full step, the jump included (`final_fix/i23_compare.json`). At t = 0 both programs agree to 7.0e-6 eV in every state energy, over the 20 trajectories.

  Before the first hop, the two programs follow the same path in most trajectories, but not in all (section 6, item 11). The exceptions are 00003, 00014 and 00016, where the paths part at a step our run halved, and 00010 and 00018, where they drift apart slowly.

  SHARC has no drift limit and runs on. It then hops S1 -> S0 3.5-6 fs after the jump, with S1-S0 gaps of 0.10-1.97 eV at its later jumps.
- *Variants* (Task 7):
  - `dt_quarter` drifts in the same 6 trajectories at nearly the same times (6.5-11 fs, 0.67-0.91 eV), so a smaller step does not help;
  - `sa3_cas22` and `sa2_cas43` remove all 7 early drifts;
  - `sa3_cas23` still drifts in 00006 (8.0 fs).

  In the replay windows, SA3-CAS(2,2) has a single solution in all 7: continued equals fresh to at most 1.3e-5 eV, the state singular value is at least 0.965 and the active-orbital singular value is at least 0.9963 (Task 6 section 4.1).
- *00014 is the exception.*
  - SHARC jumps by -0.994 eV at 1.5 fs and not at 11 fs. Ours has a rung change with +0.143 eV at 1.75 fs, then -0.291 eV at 3.0 fs, and the terminal +0.905 eV at 11.0 fs.
  - The replay window covered only the first jump (the window rule takes the first jump), so the terminal jump was not replayed.
  - *Where the paths part* (`final_fix/i23b_paths.json`):
    - up to 1.0 fs the geometries agree to 2.0e-7 bohr;
    - our run then rejected the full step from 1.0 to 1.5 fs and halved it: the record at 1.25 fs has depth 1 after one rejection; the run reached 1.5 fs at dt/2, then halved again, to dt/4 and dt/8, up to 2.0 fs;
    - SHARC took that step in one and jumped there;
    - from 1.5 fs on, the geometries differ (1.0e-3 bohr at 1.5 fs).

    Why our full step was rejected is not recorded, because rejected attempts appear in no log record.

**Label: method**, with a detection gap in our program.

- *Why method:*
  - SHARC_PYSCF, with an independent driver and an independent CASSCF call, lands on the same second SA2 solution at the same step in 6 of 7. In those 6 its state energies equal ours to at most 3.2e-5 eV at every common full step, the jump included.
  - The continued replay reproduces the run to at most 9.9e-6 eV.
  - So the run's solution is reproduced both by SHARC and by a continued solve.
- *The fresh-solve test, read literally, is met here* (`final_fix/i23_compare.json`, `replay_vs_stored`):
  - At the jump record, fresh equals the run to at most 1.1e-5 eV.
  - At 1-4 records before the jump, fresh differs from the run by 0.088-0.965 eV per state. In those records, fresh is on the other solution and the run is not.

  The report reads this as two valid solutions, not as a defect (Definitions). That reading was made after the numbers were seen (section 7; decision 9).
- *The detection gap:* this is a shortcoming of the dynamics stage's design, known since P7b (USER_MANUAL 8.16.3); it is not a wrong result. A jump of this size is accepted, and only the cumulative drift limit ends the run, after the trajectory has already continued on another solution.
- *00014:* it formally meets the "ours" definition, because the two programs differ from 1.5 fs, on the same initial conditions.
  - The difference begins where our run halved a step that SHARC took in one, so from then on the two programs no longer evaluate the same geometries.
  - Whether this is a defect of ours, or the step rejection working as designed and keeping the run on the old solution, is not known.
  - **The evidence is mixed** for this one trajectory (decision 6).

**Candidate fixes (spec 4.2).**

- **F1** (per-step energy check as a rejection signal): **supported as detection.** It would end these trajectories at the jump, as `electronic_discontinuity`, instead of at the drift limit, and it would catch jumps below the remaining drift margin.

  *Not measured, a reasoned expectation:* F1 will not raise the kept count. At the jump the old solution is no longer found from the previous orbitals: continued equals fresh, and `dt_quarter` meets the same jumps. So halving the step should not avoid the switch, and a trajectory with a jump above 0.1 eV is not kept anyway.
- **F2** (active-orbital continuity): **supported as detection only; its re-solve part is ruled out here.** The continued replay already starts from the previous step's orbitals and still lands on the other solution.

  As a detection signal, the active-orbital singular values separate the cases in these windows, but no threshold has been validated:
  - at the drift jumps, the window minimum of continued base is 0.8835-0.9472;
  - in base windows without a switch it is 0.9906-1.0000 (00002, 00007, 00013, 00015, 00016).
- **F3** (state-count warning): **its mechanism is ruled out** (there is no intruder). Its practical effect, computing a third state, is what removes the early drifts; but that works through the state average, not through tracking an extra state.
- **F4** (convergence ladder): not relevant to this class.

### 2.2 `overlap_failure` (8: 00001, 00002, 00005, 00008, 00013, 00015, 00017, 00018)

**Evidence.**

- *Forensics* (Task 6 table 2.4): every one has exactly one hop, S1 -> S0, and **ends 5.8-42.9 fs after it**:

  | traj | end - hop (fs) |
  |---|---|
  | 00001 | 28.375 |
  | 00002 | 5.8125 |
  | 00005 | 11.0625 |
  | 00008 | 42.875 |
  | 00013 | 6.1875 |
  | 00015 | 8.25 |
  | 00017 | 11.875 |
  | 00018 | 6.875 |

  Each ends on the active state S0 at step dt/8. In the last five records before each end:
  - the state singular value is 0.92-0.9995;
  - the rung never changes;
  - the S1-S0 gap is 3.4-7.5 eV.

  So the statements in M4 and spec section 1 that the failures come "right after the first hop" are wrong.
- *Reproduced in the replay* (00008, 00017, 00018; Task 6 section 4.2):
  - continued base reproduces the run up to the last record (1.2e-5, 1.4e-6 and 6.1e-7 eV);
  - at the record before the failed step, fresh finds another solution (0.96, 1.17 and 0.51 eV away per state);
  - at the failed step, continued falls onto the fresh solution. The state singular value is 0.3718, 0.3783 and 0.3595 (the run had 0.372, 0.378 and 0.36), and one active orbital rotates out (active-orbital singular values 0.4268, 0.4274 and 0.4896).

  This is the energy-drift mechanism, but this time the overlap signal sees it.
- *Not reproduced* (00001, 00002, 00005, 00013, 00015).
  - The 6-record replay window ending at the last record starts from SCF orbitals. There, the continued and backward replays differ from the stored energies by 0.19-2.28 eV at every record (`final_fix/i23_compare.json`, `replay_vs_stored`).
  - The fresh solves differ by the same amounts, with one exception: in 00001, at r77 and r78, fresh reproduces the run to 5.9e-7 and 9.6e-7 eV. There, the run's solution can be reached from SCF orbitals.
  - Up to the first hop, ours and SHARC agree to at most 5.0e-5 eV in the state energies at every common full step: 00001 2.2e-5, 00002 3.9e-5, 00005 5.0e-5, 00013 3.3e-5 and 00015 3.1e-5 eV (`final_fix/i23_compare.json`). So the run was on SHARC's solution until the hop. After the hop the paths differ, because the hop times differ.
  - The run's solution lies **above** the replayed one in SA-average energy (up to +1.20 eV). In 00001 the sign changes within the window.
  - The failed-step geometry was built on the replayed solution, so the failure itself was not reproduced (Task 6 "does not fit" item 2).
  - Task 7 compared these 5 with SHARC, on different post-hop paths (SHARC has its own random stream):
    - **00013:** SHARC hops at 12.5 fs, and its S0 (active) drops by 1.823 eV at 18.0 fs (ΔE_tot -1.944 eV). We end with `overlap_failure` at 17.6875 fs.
    - **00001:** SHARC drops by -2.482 eV in E_tot at 23.5 fs (S0 -2.466 eV). We fail at 39.875 fs.
    - **00005:** SHARC has discontinuities of +2.683 eV in S1, +1.513 eV in S0 and -2.188 eV in S1 at 27.5-28.5 fs, 2.4-3.4 fs after our failure times (25.0625 fs in base, 24.9375 fs in `nacdr`).
    - **00002:** depends on the path. SHARC does not hop; it jumps by -0.287 eV at 27.5 fs and stops at 32.5 fs. Our `nacdr` variant does not hop either and fails at 27.4375 fs; our base hops at 16.5 fs and fails at 22.3125 fs.
    - **00015:** depends on the path. SHARC hops 1 -> 0 -> 1 at 9.5/10.0 fs and finishes. Our `nacdr` variant hops at 9.5 fs and finishes, with a largest jump of 0.1476 eV. Our base stays on S0 and fails at 18.25 fs.
- *Intruder:* not seen. In SA3-CAS(2,2), S2 lies at least 0.589 eV above S1 in every overlap-failure window; in SA3-CAS(2,3) the S2-S1 gap is 0.384-0.562 eV, with no crossing (Task 6 section 4.2).
- *Variants* (Task 7):
  - `sa3_cas22` keeps 5 of these 8 (00001, 00002, 00015, 00017, 00018). It finishes 00008 but does not keep it, because of a jump of 0.1448 eV at a hop record. It delays 00005 to 60.6875 fs and 00013 to 68.125 fs;
  - `dt_quarter` does not remove them;
  - `nacdr` removes only the path-dependent 00015;
  - `sa2_cas43` keeps none of them: they end at 13.88-84.25 fs, 3 of them (00002, 00008, 00018) as electronic failures.

**Label: method**, established for 3 of 8; the most likely reading for the other 5, but not established.

- For 00008, 00017 and 00018, the run's solution disappears at the failed step: continued equals fresh there. Halving to dt/8 cannot avoid it, and our overlap signal stops the run as designed.
- For the other 5 the evidence is indirect:
  - the run was on a higher SA2 solution after the hop;
  - in 00013 and 00001, SHARC shows an S0 drop of 1.9-2.5 eV. In 00013 it comes 0.3 fs after our failure time, which we see only as an overlap failure. In 00001 it comes 16 fs before our failure time, on a path that is 0.5 fs apart at the hop;
  - in 00002 and 00015 the outcome depends on the hop path.

  After the hop there is no common geometry with SHARC, so the SHARC test cannot be applied. The literal fresh-solve test is met in all 5, because fresh differs from the run in the tail window (00001 r77-r78 excepted). As in 2.1, this is read as several solutions, not as a defect (Definitions; decision 9). So the label is **method, not established**, and "ours" is not excluded for these 5.

  **The step where each of these 5 runs moved onto the higher solution, if there was one, was not found**: no jump above 0.1 eV lies between the hop and the end in 00001, 00002, 00005 and 00013 (Task 6 section 4.2).
- *Proposed, not run:* a continued replay from the hop record to the end for these 5.
  - It would start where the run is still on SHARC's solution.
  - Its first point is solved from SCF orbitals, and whether that lands on the run's solution is not known. If it does not, the replay must start earlier.
  - Size: 130 records in total (00001 59, 00002 14, 00005 24, 00013 15, 00015 18), counted from `steps.log`.
  - Cost: a few minutes at about 0.5 s per solve (section 4), estimated.
  - It is in section 5 and in decision 7.

**Candidate fixes.**

- **F1:** neutral for these 8. The overlap signal already ends them.
- **F2:** supported as detection for the 3 reproduced cases (active-orbital singular value 0.43-0.49); its re-solve part is ruled out (continued falls onto the fresh solution).
- **F3:** its mechanism (an intruder) is ruled out.
- **F4:** not relevant.

The measured remedy is a setting: SA3-CAS(2,2) keeps 5 of the 8.

### 2.3 `electronic_failure` (3: 00000, 00007, 00016)

**Evidence.**

- *Solver log* (`forensics_m4.json`, Task 6 section 2.5). All three stop at the 150-macro-iteration limit on every rung while the energy is stationary:
  - 00000: |dE| of 2.4e-12 to 1.4e-10 per macro iteration;
  - 00007 and 00016: about 1e-14.

  The kept log lines contain no orbital-gradient norm. That `conv_tol_grad` = 1e-6 is the criterion that fails is therefore an **inference**, supported by the trials below.
- *00000* (48.75 fs, after a hop at 41.0 fs):
  - *Replay:* reproduced; every base mode fails at the failed step with the run's error text. The window holds two base solutions (r96-r98). The rung change and the 0.234 eV jump at r99 are reproduced.
  - *Convergence trials* (fresh starts): not converged at `conv_tol_grad` 1e-6 or 1e-5; **converged at 1e-4** (`conv_tol` 1e-8).
  - *SHARC:* takes another path (hops at 38.5 fs) and finishes.
- *00007* (0.25 fs):
  - *Forensics:* the first full step from t = 0 was rejected (`n_rejections` 4), and the trajectory went on at dt/8 (records at 0.0625-0.25 fs, depth 3). The CASSCF rung changed at 0.25 fs (flagged), and the next dt/8 step failed on every rung. The run does not record why the attempts were rejected (rejected attempts appear in no log record).
  - *Replay of the failed step (0.3125 fs):* reproduced; a single solution in the window (singular values 1.0000).
  - *Convergence trials at the failed step:* converged at `conv_tol_grad` 1e-5 and 1e-4 (the two agree to 1e-4 eV).
  - *The first full step, replayed in the final fix wave* (`final_fix/i1_traj00007.json`, `i1b_trials.json`; M4 `[electronic]` settings, 2 threads).
    - *Method:* the S1 gradient was computed at t = 0 from a fresh solve, as the run starts. The Verlet positions of the full step and of its retries at dt/2, dt/4 and dt/8 were built from it. The dt/8 position equals the run's first record to 4.4e-16 bohr.
    - *What each attempt gives, continued from the t = 0 solve:*

      | Attempt | Continued from t = 0 | Fresh |
      |---|---|---|
      | dt (0.5 fs) | fails on every rung | fails on every rung |
      | dt/2 (0.25 fs) | converges on the SCF-orbitals rung only | not solved |
      | dt/4 (0.125 fs) | converges on the SCF-orbitals rung only | not solved |
      | dt/8 (0.0625 fs) | converges on the previous-orbitals rung | not solved |

      At dt/8 the replay reproduces the run's first record to 5.4e-12 eV.
    - *So, by inference from the replay:* the full step was rejected for a convergence failure, and the dt/2 and dt/4 retries for a CASSCF rung change, which is a rejection signal.
    - *Convergence trials at 0.5 fs* (fresh):
      - `conv_tol_grad` 1e-6 and 1e-5 fail;
      - 1e-4 converges, at +0.15484 / +0.11036 eV against t = 0.
  - *SHARC at the same geometry* (`sharc/base/traj_00007/output.dat`):
    - SHARC's step 1 (0.5 fs) is a converged record: its state energies are +0.15477 / +0.11042 eV against its t = 0.
    - Its geometry equals our full-step position to 1.0e-7 bohr. The t = 0 geometries agree to 9.4e-8 bohr and the t = 0 energies to 6.2e-7 eV.
    - Our solver fails at SHARC's own step-1 geometry too, both continued and fresh.
    - Our 1e-4 trial lies within 6.2e-5 eV of SHARC's energies, in each state.
    - SHARC then runs on to a drift-type jump of +0.988 eV at 11.5 fs. Our `dt_quarter` variant also passes 0.25 fs, and then drifts at 11.5 fs (0.9491 eV).
  - *Not known:* whether `SHARC_PYSCF.py`'s result meets `conv_tol_grad` 1e-6 by our criterion, or whether SHARC accepts a result that we would refuse (black box).
  - This confirms the final review's probe.
- *00016* (17.0 fs, no hop):
  - *SHARC* (`sharc/base/traj_00016/output.dat` and `output.log`, the output of our own SHARC run; `final_fix/i23_compare.json`):
    - SHARC's last record is step 34, at 17.0 fs, with energies.
    - Its log then enters timestep 35 (17.5 fs) and prints "QM call was not successful" (error code 256).
    - So **SHARC converged at 17.0 fs and failed on the 17.5 fs call.**
    - That its failure has a convergence cause is an inference: SHARC's error text is not in the stderr tail the harness keeps.
  - *Our run* (`final_fix/i23b_paths.json`):
    - It rejected the full step from 16.5 to 17.0 fs and went on in halved steps: 16.75 fs (depth 1), 16.875 fs (depth 2) and 16.9375 fs (depth 3), each after one rejection.
    - It converged at 17.0 fs (r37, depth 3) and then failed on the next step, which it halved down to dt/8 (17.0625 fs).
    - **Both programs converge at 17.0 fs and fail on the next QM call.** This pattern is shared.
  - *The two paths at 17.0 fs.* Up to 16.5 fs ours and SHARC agree to 2.8e-6 bohr and 2.2e-5 eV. At 17.0 fs our halved path and SHARC's full step differ:
    - in geometry by 1.4e-3 bohr;
    - in energy by -4.7 meV (S0) and -10.2 meV (S1), ours minus SHARC, and by -9.3 meV in E_tot.

    This 10 meV difference is a difference between paths (different geometries), not between two solves at one geometry.
  - *Our solver at SHARC's geometries* (`final_fix/i2b_traj00016.json`):
    - Continued along SHARC's 16.0, 16.5 and 17.0 fs geometries: at 16.0 and 16.5 fs it converges, matching SHARC's energies to 8.0e-6 and 1.4e-5 eV; at 17.0 fs it **fails on every rung**.
    - A fresh solve also fails at 17.0 fs.
    - Both fail at the Verlet position of SHARC's failed 17.5 fs call, which was built from SHARC's 17.0 fs record.
    - So **at SHARC's 17.0 fs geometry SHARC converges and our solver does not**: the 00007 disagreement again.
  - *Our run against the replay at r37.*
    - Every replay mode fails at r37, and so do the gradient backend and both fresh determinism solves; no convergence trial could run there.
    - The continued mode started from SCF orbitals at r32 (16.0 fs). `replay_points.solve_path` has no previous orbitals at a segment's first point; the rung of r32 is "SCF orbitals" in `replay/traj_00016.json`.
    - It reproduced r32-r36 to at most 8.9e-6 eV, along the run's own halved path, and then failed at r37.
    - So **the run's convergence at r37 is not reproduced by a fresh solve or by a continued solve from 16.0 fs**. This is unexplained (Task 6 "does not fit" item 3).
    - Convergence that depends on the path is not excluded. It would be the same kind of effect as known limitation 3a (CLAUDE.md; the intermittent water CAS(4,4) test of USER_MANUAL §10).
    - SHARC also converged at its own 17.0 fs geometry, where our solver fails both continued and fresh. So, at least in our solver, convergence at this point depends on more than the geometry. How SHARC's interface starts its solve is not known (black box).
- *Variants:*
  - `sa3_cas22` has **no electronic failure** in any of the 20;
  - `dt_quarter` has 4;
  - `sa2_cas43` has 7, two of them at step 0.

**Label: settings for 00000. 00007: settings by the fresh-solve test, a candidate "ours" by the SHARC test; the two references disagree. 00016: method or settings for the failure itself, with the same disagreement at SHARC's 17.0 fs geometry.**

- *00000:* SHARC takes another path, so the only test at the failed geometry is a fresh solve.
  - At the run's settings (`conv_tol_grad` 1e-6) the fresh solve fails, as the run did.
  - With `conv_tol_grad` 1e-4 it converges.
  - The solver sits at a stationary energy at the iteration limit.

  So it is not ours by that test.
- *00007:* the two references disagree.
  - *Fresh-solve test: settings.* At the run's settings our fresh solve fails at the run's geometries, both at the rejected full step (0.5 fs) and at the failed dt/8 step (0.3125 fs), as the run did. A looser `conv_tol_grad` converges: 1e-4 at 0.5 fs, and 1e-5 or 1e-4 at 0.3125 fs.
  - *SHARC test: a candidate "ours".* At the 0.5 fs geometry (the same to 1.0e-7 bohr), `SHARC_PYSCF.py` returns a converged result with the same nominal tolerances (`conv_tol` 1e-10, `conv_tol_grad` 1e-6), and our three rungs fail.
  - *What is not known:* whether SHARC's result meets our convergence criterion (black box). Our 1e-4 solve lands within 6.2e-5 eV of it, so it is the same solution.

  The case goes to decision 6.
- *00016:* the failure is shared, the 17.0 fs solve is not.
  - Both programs converge at 17.0 fs and fail on the next call. That supports "method or settings", by inference, because SHARC's error text is not recorded. `sa3_cas22` removes the failure.
  - At SHARC's 17.0 fs geometry, SHARC converges and our solver does not, continued or fresh: the 00007 disagreement (decision 6).
  - Why our run converged at its own 17.0 fs geometry (r37), where the fresh and continued replays fail, is not identified; convergence that depends on the path is not excluded.

**Candidate fixes.**

- **F4: supported for 2 of 3.** Fresh solves converge with a looser `conv_tol_grad` (00000, 00007).
  - SHARC's interface gets through two geometries where our three rungs fail at the same nominal tolerances: 00007 at 0.5 fs and 00016 at 17.0 fs. At the first, our solve converges only at `conv_tol_grad` 1e-4, and then on SHARC's solution (within 6.2e-5 eV).
  - Why SHARC converges there is not known (black box). It may be a different start, a different convergence check, or a looser effective criterion.
  - What was not measured:
    - whether a looser rung degrades the gradients or the energy conservation;
    - whether a larger `max_cycle` or another initial guess would also converge (not tried; `replay_points.py` tries the two tolerances only, although spec 3.2 lists `max_cycle` and the initial guess too);
    - the "previous orbitals" rung with a looser tolerance (the trials used fresh starts only).
- **F1, F2, F3:** not relevant.

The measured remedy for the seed-2026 set is SA3-CAS(2,2), which has no electronic failure.

## 3. The determinism check

- **Fresh starts in separate processes.** Two fresh base solves were run in separate spawned processes at the failure record of each trajectory (Task 6 section 5):
  - the largest |ΔE| over 17 trajectories is **4.55e-12 eV** (traj_00017);
  - 00006 and 00013 give exactly 0;
  - in 00016 both solves fail at r37.
- **Whole runs.** The `base` variant of Task 7 reproduces M4 run A in all 20 trajectories, in outcome and `t_final_fs` to 4 decimals. The run is from another process and another day, with the same program code. M4 run B had already reproduced run A in outcome, hops and rejections (USER_MANUAL 5.9).
- **What this does and does not show.** The fresh-start pairs show no dependence on the process (at most 4.55e-12 eV). They do **not** exclude convergence that depends on the history of the path.
  - In 00016, our run converged at r37, where every replay mode and both fresh solves fail (section 2.3). This is unexplained.
  - SHARC also converged at 17.0 fs, on its own geometry, where our solver fails both continued and fresh (section 2.3). In 00007 likewise, SHARC converged at 0.5 fs where our solver fails. Neither case is a determinism finding: each is a difference between two programs at one geometry.
  - It is the same kind of effect as known limitation 3a (water CAS(4,4)/STO-3G, a solution that depends on the run). That limitation itself was not reproduced here.
- **Coverage:**
  - the fresh-start check covers one geometry per trajectory, the base variant and fresh starts only;
  - SA3 determinism was not checked.

  The SA3 start-solution difference between ours and SHARC (section 6, item 5) is a difference between two programs, not a determinism finding.

## 4. Time per step

**Full runs.** From `~/p9a/variants/*/summary.json` and Task 7. The time per step is the sum of the per-trajectory wall times divided by the accepted (sub)steps; 8 workers x 2 threads run concurrently. The (sub)steps include halved ones (ledger minor: `s_per_step` is per accepted (sub)step).

| Variant | Kept | Steps | s per step | Run wall (20 x up to 100 fs) | Probe, 1 worker alone (s per step) |
|---|---|---|---|---|---|
| base SA2-CAS(2,2), dt 0.5 | 0 | 1353 | 3.07 | 678 s | 1.54 |
| nacdr | 0 | 1528 | 5.93 | 1425 s | 2.96 |
| dt_quarter (dt 0.25) | 1 | 2447 | 3.12 | 1197 s | 1.36 |
| sa3_cas22 | 11 | 3618 | 2.84 | 1452 s | 1.34 |
| sa3_cas23 | 0 | 528 | 4.09 | 336 s (every trajectory ends by 24.44 fs) | 1.49 |
| sa2_cas43 | 0 | 1934 | 3.57 | 997 s | 1.86 |
| SHARC base (nacdr, 1 thread) | 1 | | about 10-11 | 4565 s | about 7.3 (start-up included) |
| SHARC sa3_cas22 | 2 | | | 10549 s | 48.2 s for 4 steps (1.66 x the base probe) |

**Single points.** Task 6 section 7: successful solves without gradients, 2 threads, 4 concurrent workers, mean (s):

| Variant | Continued | Fresh |
|---|---|---|
| base | 0.49 | 0.57 |
| SA3-CAS(2,2) | 0.26 | 0.42 |
| SA3-CAS(2,3) | 0.36 | 0.51 |
| SA2-CAS(4,3) | 0.49 | 0.78 |

A failed base solve, which runs every rung to its limit, costs 5.7-7.4 s.

**Cost.** SA3-CAS(2,2) costs no more per step than base. Its run wall is the longest because 13 of its 20 trajectories reach 100 fs.

## 5. The proposed acceptance setup

**Proposal.** The SA3-CAS(2,2) variant, the only measured setup that keeps a clear part of the trajectories:

```toml
[electronic]
method = "casscf"
basis = "6-31g*"
singlets = 3            # S0, S1, S2 in the state average (M4: 2)

[electronic.casscf]
ncas = 2
nelecas = 2             # conv_tol 1e-10 and conv_tol_grad 1e-6 at their defaults, as in M4

[selection]
states = ["S1"]         # proposal: keeps the S1 starts that were measured (see below)

[dynamics]
tmax_fs = 100.0
dt_fs = 0.5             # every other key at its default, as in M4: coupling "overlap",
                        # hopping "sharc", EDC, energy_drift_limit_eV 0.5, overlap_threshold 0.5
```

The rest follows spec 4.3: ethylene `tests/data/ethylene.xyz`, seed 2027, `n_samples` 100, `n_trajectories` 20, the M4 resources, and `[analysis]` at its defaults.

**Reason** (section 2, Task 7):

- on the seed-2026 initial conditions it removes every early `energy_drift` (7 to 0 before 40 fs) and every `electronic_failure` (3 to 0);
- it keeps 5 of the 8 base overlap failures;
- it costs the same per step as base (2.84 against 3.07 s).

The other variants are ruled out by their kept counts (0, 0, 1, 0, 0), in particular:

- `dt_quarter` does not touch the drift class;
- `sa3_cas23` and `sa2_cas43` have multiple solutions and failures of their own (Task 6 "does not fit" item 8; Task 7).

**Evidence on seed 2026: 11 of 20 kept.** The kept trajectories are 00000, 00001, 00002, 00004, 00010, 00012, 00014, 00015, 00017, 00018 and 00019, with largest jumps of 0.0182-0.0978 eV (`variants/sa3_cas22/summary.json`). **This is below the 16 of the success rule.** The 9 that are not kept:

| Loss | Trajectories |
|---|---|
| `overlap_failure` after a hop | 00003 at 90.5625 fs (sv 0.287), 00005 at 60.6875 fs (sv 0.000121), 00011 at 96.0 fs (sv 0.0171), 00013 at 68.125 fs (sv 0.393, jump 0.2518 eV), 00016 at 91.375 fs (sv 0.0574) |
| `energy_drift` after a hop | 00006 at 40.25 fs (jump 1.101 eV), 00009 at 59.0 fs (jump 0.3768 eV) |
| `finished` with a jump at a hop record | 00007 (0.1178 eV), 00008 (0.1448 eV) |

None of these 9 was replayed, so **their causes are not diagnosed**.

SA3-CAS(2,2) is not free of switches, so an SA3 solution switch is a candidate cause of at least some of them. This is not measured:

- the two late drifts have jumps of 1.101 and 0.3768 eV after a hop;
- in the replay, SA3-CAS(2,2) has two solutions, 1.05 eV apart, at the failed step of base 00018 (Task 6 section 4.2);
- its "one solution in all 7 drift windows" (section 2.1) holds inside those windows only.

**Caveats that plan 2 must carry.**

1. **The SA3 yield depends on the starting solution.** SA3-CAS(2,2) also has two solutions at the initial geometries: our run and SHARC's start on different SA3 solutions in 13 of 20 (0.176-0.372 eV in some state). In those 13, SHARC switches within 0.5-3.5 fs, by 0.17-0.40 eV. **6 of our 11 kept trajectories start on the solution that SHARC did not use** (Task 7). The 11 is therefore partly a property of our initial guess, and neither solution has been shown to be the right one (the SA-average differences have both signs).
2. **The initial conditions will differ.** In the acceptance run they come from SA3 optimize, frequencies and sampling at seed 2027. The seed-2026 evidence was measured on SA2-based initial conditions, as spec risk 5 states.
3. **Selection of the initial state.** With `singlets = 3`, the default selection may also start trajectories in S2, which no measurement covers; `states = ["S1"]` keeps the measured case. This is a proposal, not a measurement.
4. **Cost** (estimated, not measured). The dynamics should take about 25 min, by the 1452 s of the variant run. The SA3 finite-difference frequencies (37 gradient tasks) and the excitations add a few minutes, by the M4 stage times of USER_MANUAL 5.9.

**Proposals beyond the measured variants. None of these has been measured; plan 2 would have to measure each on seed 2026 before the acceptance run (spec 4.1 rule 4).**

| Proposal | Basis | Expected effect on the kept count | What plan 2 must measure |
|---|---|---|---|
| **F1**, the energy check as a rejection signal, at 0.1 eV | the drift detection gap (2.1) | probably none; it makes losses explicit and catches jumps below the drift margin (reasoning, 2.1) | `sa3_cas22` with F1 on the 20 seed-2026 conditions: kept count, and which losses become `electronic_discontinuity` |
| **An initial-solution rule for SA3**, e.g. solve t = 0 from several guesses and choose by a stated criterion | the two solutions in 13 of 20 (caveat 1) | unknown; it could go either way, because 6 of our kept start on the solution SHARC did not use | at t = 0 of all 20: the solutions found from several guesses and their SA energies; then a `sa3_cas22` run with the rule |
| **F4**, a longer convergence ladder | 00000 and 00007 converge at a looser `conv_tol_grad`; SHARC's interface converges at two geometries where our rungs fail (2.3) | none on seed 2026 (`sa3_cas22` has no electronic failure); it is protection | gradient accuracy and drift with the looser rung; the previous-orbitals rung, `max_cycle` and the initial guess, at the two geometries where SHARC converges |
| **Replay from the hop to the end** for the 5 unreproduced overlap failures | the run was on SHARC's solution up to the hop (2.2) | none by itself; it would settle their label | a continued replay of the 130 records from each hop record to the end (00001, 00002, 00005, 00013, 00015); a few minutes, estimated |
| **F2** as detection | the active-orbital singular values (2.1, 2.2) | none expected (detection) | a threshold against the measured distributions, if F2 is wanted besides F1 |
| **Replay of the `sa3_cas22` losses** | 9 losses, undiagnosed | none by itself | forensics and replay of the 9, in particular the jumps at hop records (section 6, item 2) |

## 6. Findings that fit no class

1. **"Right after the hop" was wrong.** The overlap failures come 5.8-42.9 fs after the hop (section 2.2). Likewise, the M4 reading "after the hop the two computed states are not enough", with its pointer to the state-count guidance (USER_MANUAL 5.9), is not supported: no intruder was seen.
2. **Jumps of 0.12-0.15 eV at accepted hop records**, which are not solution switches:

   | Run | Trajectory | Jump |
   |---|---|---|
   | base | 00015 | 0.1217 eV, gap 0.349 eV, all base replay modes agree to 4.2e-6 eV |
   | `nacdr` | 00015 | 0.1476 eV |
   | `sa3_cas22` | 00007, 00008 | 0.1178 and 0.1448 eV |
   | `sa2_cas43` | 00010, 00012 | 0.1217 and 0.1238 eV |

   Their cause is not identified. They decide the kept status of 2 `sa3_cas22` trajectories. Whether they shrink with dt (an integration error near the crossing) or not was not measured.
3. **traj_00005's step at 1.5 fs:** ΔE_S0 is +0.573 eV in one 0.5 fs step, identical in all base modes (agreement 4e-6 eV). The variants change by at most 0.216-0.369 eV over the same segment, so it is a step specific to the SA2-CAS(2,2) surface; its cause is not identified (Task 6 "does not fit" item 5).
4. **traj_00016's last converged point is not reproduced.** The run converged at r37, but every replay mode, the gradient backend and the determinism pair fail there (section 2.3). SHARC shows the same pattern: it converges at 17.0 fs and fails on its next call. Our solver also fails at SHARC's 17.0 fs geometry.
5. **SA3-CAS(2,2) has two solutions at the start** in 13 of 20 (ours against SHARC; section 5, caveat 1). Task 6's "SA3 has one solution" holds only inside the replay windows.
6. **SA3-CAS(2,3) and SA2-CAS(4,3) are not cleaner than SA2-CAS(2,2):**
   - SA3-CAS(2,3) switches on its own in 00019 (active-orbital singular value 0.0300);
   - SA2-CAS(4,3) has two solutions in 5 windows and fails at step 0 in 00000 and 00009.

   Sources: Task 6 "does not fit" item 8; Task 7.
7. **SHARC also stops runs.** It stopped 4 base runs and 2 SA3 runs with `QM call was not successful` (error code 256); the cause is not recorded by the harness. It applies no drift or overlap check: 18 of its 20 base runs carry a jump above 0.1 eV, including the S0 drops of 1.9-2.5 eV (00013, 00001).
8. **traj_00008's first jump (r45) is not a switch:** continued equals the stored energies while fresh differs by 0.642 eV. The larger jump at r46 lies outside the window.
9. **The jump is measured on different records.** Task 6 measured it on `steps.log` (every (sub)step); the variant runs and the kept rule use `output.h5`. Base 00001, 00002, 00013, 00016, 00017 and 00018 are below 0.1 eV on `output.h5`; they are lost by outcome anyway.
10. **A minor tool finding.** The SHARC `sa3_cas22` run took 2 h 56 min against an estimate of 2.1 h (the probe ran with 1 thread, alone on the machine).
11. **Ours against SHARC before the first hop** (base, all 20; the state energies at every common full step, up to the first hop of either program or our end; `final_fix/i23_compare.json`, `i23b_paths.json`):

    | Trajectories | Largest state-energy difference | Where the paths part |
    |---|---|---|
    | 00000, 00001, 00002, 00004, 00005, 00006, 00008, 00009, 00011, 00012, 00013, 00015, 00017, 00019 | 9.5e-6 to 1.1e-4 eV | they do not part |
    | 00007 | 6.2e-7 eV (t = 0 only; our run never reached 0.5 fs) | |
    | 00003 | 2.2 meV from 9.0 fs; 25 meV by 49.0 fs | at a step our run halved (8.5 to 9.0 fs; 6.4e-4 bohr at 9.0 fs) |
    | 00014 | 1.00 eV at 1.5 fs (SHARC's jump) | at a step our run halved (1.0 to 1.5 fs; 1.0e-3 bohr) |
    | 00016 | 10.2 meV at 17.0 fs | at a step our run halved (16.5 to 17.0 fs; 1.4e-3 bohr) |
    | 00010 | 0.69 meV by 28.5 fs | slowly, with no halved step: the geometries differ by more than 1e-5 bohr from 6.5 fs, and by more than 1e-4 bohr from 26.0 fs |
    | 00018 | 5.9 meV by 78.5 fs (1.2 meV at 62.5 fs) | slowly, with no halved step: more than 1e-5 bohr from 24.0 fs, more than 1e-4 bohr from 47.5 fs |

    - Where our run halves a step, SHARC (which has no step rejection) takes it in one, and from then on the two programs integrate different paths.
    - The slow divergence of 00010 and 00018 is not explained. Growth of a small difference along the path is the likely reading, but it was not measured.

## 7. Honest summary: what was decided after the numbers were seen

- **The proposed setup** (`sa3_cas22`, and `[selection] states = ["S1"]`) was chosen after the seed-2026 numbers were seen. Spec 4.1 rule 4 allows that. But **no measured setup reaches 16 of 20 on seed 2026**: the best is 11. The success rule itself is judged on seed 2027 with the fixed code.
- **The "several solutions" reading of rule 1** (Definitions) was made after the numbers were seen.
  - Read literally, the fresh-solve test of "ours" is met in most lost trajectories: before every drift jump (all 7), in the tail windows of the 5 unreproduced overlap failures, before the failed step of the 3 reproduced ones, and in 00000 (two base solutions, r96-r98).
  - The report does not count these as "ours", because the run's solution is reproduced by a continued replay or by SHARC.
  - The reading came from the final review (I3). Before it, the report applied the fresh-solve test to 00016 only, and so applied it selectively.
  - It is owner decision 9.
- **The final fix wave added measurements after the final review:**
  - the first full step of 00007, continued and fresh, with convergence trials;
  - our solver at SHARC's 00016 geometries;
  - ours against SHARC at every common full step of the 20 trajectories.

  The scripts are in `~/p9a/final_fix/`, not in the repository, and they call the tools' functions. The 00007 replay confirmed the final review's one-off probe. The 00016 probes added the finding that our solver fails at SHARC's converged 17.0 fs geometry. The path comparison showed that the 17.0 fs energy difference of 00016 is a difference between paths, not between two solves at one geometry.
- **The 00007 label changed twice.** It was "settings" after the Task 8 review. It is now "the references disagree" (final review I1). SHARC's converged 0.5 fs record is at the geometry of our rejected first full step, which the earlier text ("a comparison of paths only") had missed.
- **The SHARC SA3 run** was made because `sa3_cas22` kept 11. The spec allowed running "the variant chosen in section 3.3" in SHARC; the choice itself came after the numbers.
- **The replay window** (6 records) was fixed before the run. It turned out too short for 5 of the 8 overlap failures, which were therefore not reproduced. Their label rests on SHARC runs that took different paths after the hop, an interpretation made after the numbers were seen. No longer replay was run.
- **The "method" label of the drift class** rests on two things: the SHARC comparison (the same jump at the same step in 6 of 7, with energies equal to 3.2e-5 eV) and the continued replay. The fresh solves disagree with the run before the jump, and the label depends on the "several solutions" reading above. Reading the absence of any rejection as a detection gap in our design, which supports F1, is an interpretation.
- **The variant list** was fixed in the spec before any run; no `[dynamics]`-threshold variant was added. The F1 default (0.1 eV) was fixed in the spec before the runs.
- **One test bound** in Task 3 was relaxed after measurement (ruling P9a1-6: the state-overlap bound of the replay test went from 0.99 to 0.95; measured 0.9863). It is a test of the tool, not a physical threshold.
- **Not tuned:** no setting was changed in response to the seed-2026 kept counts beyond choosing among the measured variants.

## Decisions for the owner

1. **The acceptance setup.** Take `sa3_cas22` (11 of 20 on seed 2026) into the acceptance run, or not.
2. **The 16/20 target.** Whether it stays, given that the best measured setup reaches 11 on seed 2026 (the rule itself is judged on seed 2027). One option is to run the acceptance as planned and document the expected shortfall (spec 2, rule 3). Another is to first measure the unmeasured proposals of section 5 on seed 2026 in plan 2.
3. **An initial-solution rule for SA3.** Whether to add one, given that SA3 has two solutions at the start geometries in 13 of 20, and if so with which criterion. It would be measured on seed 2026 first.
4. **F1, and F2 as detection.** Whether they enter plan 2 under spec 4.1 rule 1, although their class is labelled "method". The case for them is a detection gap in our design, not a wrong result.
5. **F4.** Whether it enters. It converts 2 of 3 SA2 failures, but `sa3_cas22` had none. SHARC's interface converges at two geometries where our three rungs fail (decision 6), and F4 (a looser rung, `max_cycle`, another initial guess) is the measured way to look at that.
6. **The candidate "ours" cases.** Spec 2, rule 2, would require a fix with a failing test for each case counted as "ours"; no cause is identified. In each of these, the two references of the definition disagree, or the SHARC test meets "ours":
   - **00014:** the two programs part at 1.0-1.5 fs, before any hop. Our run rejected and halved a full step that SHARC took in one, and SHARC jumps by -0.994 eV there (section 2.1). Why our step was rejected is not recorded.
   - **00007:** at the geometry of our rejected first full step (0.5 fs), `SHARC_PYSCF.py` returns a converged result. At the same nominal tolerances our three rungs fail, continued and fresh. The fresh-solve test says "settings", the SHARC test "ours" (section 2.3).
   - **00016:** both programs converge at 17.0 fs and fail on the next call, so the failure is shared and is evidence against "ours". But at SHARC's 17.0 fs geometry our solver fails where SHARC converged, which is the 00007 disagreement. Our run's convergence at r37, which no replay reproduces, is not identified (section 2.3).

   Whether SHARC's results at those two geometries meet our convergence criterion is not known (black box).
7. **Further replays before the acceptance run.** Whether plan 2 replays:
   - the 9 `sa3_cas22` losses, including the jumps at hop records;
   - the 5 unreproduced base overlap failures, from the hop to the end (section 2.2; 130 records).
8. **`[selection] states = ["S1"]`.** Whether the acceptance run uses it, as measured, or the default selection with `singlets = 3`.
9. **The reading of the "ours" rule with several solutions** (Definitions; section 7). Whether a fresh solve that lands on another SA-CASSCF solution, while the run's solution is reproduced by a continued replay or by SHARC, counts as "not ours". The reading was made after the numbers were seen. Without it, most lost trajectories meet the literal fresh-solve test of "ours".
