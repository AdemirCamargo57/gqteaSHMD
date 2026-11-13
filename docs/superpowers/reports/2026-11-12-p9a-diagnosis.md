# P9a diagnosis report: why the M4 study lost 18 of 20 trajectories

Status: **for the owner's approval** (spec section 3.5). Plan 2 (fixes and the acceptance run) is written only after this report is approved.
Date: 2026-11-12. Branch `feat/p9a-trajectory-losses`, from `main` at `4f076a5`. The spec is `docs/superpowers/specs/2026-11-12-p9a-trajectory-losses-design.md` (`e09cd79`) and plan 1 is `docs/superpowers/plans/2026-11-12-p9a1-diagnosis.md` (`722b393`). Tasks 1-5 built the tools (`ab4cb04` to `58811f6`). Tasks 6 and 7 ran the measurements; they have no diff. Nothing under `src/` changed in plan 1.

The numbers come from these sources:

- the task reports in the git-ignored ledger `.superpowers/sdd/2026-11-12-p9a1-diagnosis/` (`task-6-report.md`, `task-7-report.md`);
- the JSON files they quote, which live on the WSL native filesystem of the development machine:
  - `~/p9a/forensics_m4.json`;
  - `~/p9a/replay/traj_*.json`;
  - `~/p9a/variants/<variant>/summary.json`;
  - `~/p9a/sharc/base/sharc_summary.json` and `~/p9a/sharc/sa3_cas22/sharc_summary.json`.

Each statement names its source. A number that was estimated, or a reading that is an inference, says so.

**Definitions used throughout.**

- **Kept** (spec 4.3) means: outcome `finished` or `stopped_in_s0`, **and** a largest per-record jump `|ΔE_tot|` of at most 0.1 eV. In the variant runs this is `run_variants.is_kept`, computed over the `output.h5` records with `grid_step <= n_total`. That is the same set of records the analysis uses for `energy.flagged` when `output_every = 1`.
- **Labels** (spec 2, rule 1):
  - **ours**: our result differs from an independent reference, either SHARC 4.1 on the same initial conditions or a fresh CASSCF solve at the same geometry;
  - **method**: the electronic-structure method at these settings behaves this way in both programs, or in every solve;
  - **settings**: a change of a setting removes the loss.

## 0. Summary

- **The success rule cannot be met by any measured setup.** It needs at least 16 of 20 kept on the seed-2027 acceptance run. On the 20 seed-2026 M4 initial conditions, **no variant reached 16**. The best is SA3-CASSCF(2,2) with 11 of 20.

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
  - The `energy_drift` class is this switch in all 7 trajectories. Task 6 reproduced it by replay. SHARC_PYSCF makes the same jump at the same step in 6 of 7 (Task 7).
  - The `overlap_failure` class is the same switch after the hop in 3 of 8 (reproduced). The other 5 have indirect evidence only.
- **No intruder state was seen.** In SA3-CAS(2,2), S2 lies at least 0.589 eV above S1 in every overlap-failure window and at least 1.973 eV above it in the drift windows (Task 6 sections 4.1 and 4.2). The working hypothesis of spec section 1, "a state outside the computed set", is therefore not supported.
- **The overlap failures do not come "right after" the hop.** They come 5.8-42.9 fs after it (Task 6 table 2.4).
- **The `electronic_failure` class:**
  - In 2 of 3 trajectories, CASSCF stops at the 150-macro-iteration limit with a stationary energy. A fresh solve converges there when `conv_tol_grad` is looser.
  - In the third (00016), SHARC_PYSCF also fails at the same place.
- **No defect of our program has been shown in the sense of "ours".** Three findings come closest:
  - two trajectories formally meet the definition: 00007 (SHARC converges where we fail) and 00014 (the two programs take different solution branches at 1.5-1.75 fs);
  - **a detection gap in our design**: none of our rejection signals sees a solution switch of 0.64-0.93 eV.

## 1. What was run

Machine: the development workstation, WSL Oracle Linux 9.5, 16 cores, 7754 MB RAM (USER_MANUAL 5.9). Every run was on the WSL native filesystem. The code is at `58811f6` (the scripts of Tasks 1-5); the program code is that of `main`. The input is the M4 project `~/p8_m4/run` (run A of P8 Task 11, code `2631eff`), which was only read. SHARC 4.1 (`/home/ajc/sharc4-4.1`) was executed only: `sharc.x` with `SHARC_PYSCF.py`, under the clean-room rule.

| Measurement | Command (abridged) | Wall time | Output |
|---|---|---|---|
| Forensics (Task 6, Step 1) | `tools/diagnosis/trajectory_forensics.py ~/p8_m4/run --out ~/p9a/forensics_m4.json` | seconds (file dated 13:55:19) | `forensics_m4.json` |
| Replay (Task 6, Step 2) | `tools/diagnosis/replay_points.py ~/p8_m4/run --all-lost --out ~/p9a/replay --window 6 --workers 4 --threads 2 --memory-mb 1500` | 5 min 06 s for 18 trajectories (from the file mtimes; the `END` line of `replay.log` is wrong) | `replay/traj_*.json`: 4 variants x 3 modes x 134 points = 1608 solves, plus determinism pairs and convergence trials |
| Our runner, 6 variants (Task 7) | `tools/diagnosis/run_variants.py ~/p8_m4/run tools/diagnosis/variants_p9a.toml --out ~/p9a/variants --variant <v> --workers 8 --threads 2 --memory-mb 800` | 336-1452 s per variant (table in section 4) | `variants/<v>/summary.json` |
| SHARC, base (Task 7) | `tools/sharc_compare/run_sharc_pyscf_ensemble.py ~/p8_m4/run tools/diagnosis/variants_p9a.toml --variant base --out ~/p9a/sharc --workers 8` | 4565 s | `sharc/base/sharc_summary.json` |
| SHARC, sa3_cas22 (Task 7) | the same command with `--variant sa3_cas22` | 10549 s (estimated beforehand at 2.1 h) | `sharc/sa3_cas22/sharc_summary.json` |

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

  The path before any hop is deterministic in both programs. At t = 0 both programs agree to 7.0e-6 eV in every state energy, over the 20 trajectories.

  SHARC has no drift limit and runs on. It then hops S1 -> S0 3.5-7 fs after the jump, with S1-S0 gaps of 0.10-1.97 eV at its later jumps.
- *Variants* (Task 7):
  - `dt_quarter` drifts in the same 6 trajectories at nearly the same times (6.5-11 fs, 0.67-0.91 eV), so a smaller step does not help;
  - `sa3_cas22` and `sa2_cas43` remove all 7 early drifts;
  - `sa3_cas23` still drifts in 00006 (8.0 fs).

  In the replay windows, SA3-CAS(2,2) has a single solution in all 7: continued equals fresh to at most 1.3e-5 eV, the state singular value is at least 0.965 and the active-orbital singular value is at least 0.9963 (Task 6 section 4.1).
- *00014 is the exception.* SHARC jumps by -0.994 eV at 1.5 fs and not at 11 fs. Ours has a rung change with +0.143 eV at 1.75 fs, then -0.291 eV at 3.0 fs, and the terminal +0.905 eV at 11.0 fs. The replay window covered only the first jump (the window rule takes the first jump), so the terminal jump was not replayed.

**Label: method**, with a detection gap in our program.

- *Why method:* SHARC_PYSCF, with an independent driver and an independent CASSCF call, lands on the same second SA2 solution at the same step in 6 of 7. In our own replay, the fresh solve agrees with the continued one at the jump. So neither the reference nor a fresh solve differs from our result.
- *The detection gap:* this is a shortcoming of the dynamics stage's design, known since P7b (USER_MANUAL 8.16.3); it is not a wrong result. A jump of this size is accepted, and only the cumulative drift limit ends the run, after the trajectory has already continued on another solution.
- *00014:* it formally meets the "ours" definition, because the two programs differ at 1.5-1.75 fs. The underlying cause is the same multiplicity of solutions: the two programs took different branches. **The evidence is mixed** for this one trajectory.

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
  - The 6-record replay window starts from SCF orbitals and lands on a solution other than the run's: it differs from the stored energies by 0.19-2.28 eV at every record of the window.
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

  None of this shows our program giving a result that differs from a reference at the same geometry, so the label is **not ours**. **The step where each of these 5 runs moved onto the higher solution, if there was one, was not found**: no jump above 0.1 eV lies between the hop and the end in 00001, 00002, 00005 and 00013 (Task 6 section 4.2).

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
  - *Forensics:* the first full step from t = 0 was rejected (`n_rejections` 4), and the trajectory went on at dt/8 (records at 0.0625-0.25 fs, depth 3). The CASSCF rung changed at 0.25 fs (flagged), and the next dt/8 step failed on every rung. The reason for the first rejection is not recorded (rejections that end a trajectory appear in no log record).
  - *Replay:* reproduced; a single solution in the window (singular values 1.0000).
  - *Convergence trials:* converged at `conv_tol_grad` 1e-5 and 1e-4 (the two agree to 1e-4 eV).
  - *SHARC:* converges at 0.5 fs and runs on to a drift-type jump of +0.988 eV at 11.5 fs. Our `dt_quarter` variant also passes 0.25 fs, and then drifts at 11.5 fs (0.9491 eV).
- *00016* (17.0 fs, no hop):
  - *SHARC:* `SHARC_PYSCF.py` also fails at **17.0 fs** on the same deterministic path (no hop in either program). That the failure has the same convergence cause is an inference from the place; SHARC's error text is not in the stderr tail the harness keeps.
  - *Replay:* fails at r37, which the run had converged. The gradient backend and the determinism pair fail there too, so no convergence trial ran. This is unexplained (Task 6 "does not fit" item 3).
- *Variants:*
  - `sa3_cas22` has **no electronic failure** in any of the 20;
  - `dt_quarter` has 4;
  - `sa2_cas43` has 7, two of them at step 0.

**Label: settings for 00000 and 00007; method or settings for 00016.**

- *00000 and 00007:* a fresh solve at the failed-step geometry converges with a looser `conv_tol_grad` (1e-4 and 1e-5). The solver sits at a stationary energy at the iteration limit.
- *00007 formally meets the "ours" definition:* SHARC converges at the 0.5 fs geometry, which is reached identically in both programs before any hop. Why our ladder fails there was not identified. Possibly SHARC's interface uses a different CI solver: per USER_MANUAL 8.16.8 C, SHARC runs with `fix-spin-shift`, while ours is singlet-only. That has not been tested, and the interface source is not read under the clean-room rule.
- *00016:* SHARC fails at the same place, so it is not ours.

**Candidate fixes.**

- **F4: supported for 2 of 3.** Fresh solves converge with a looser `conv_tol_grad`. What was not measured:
  - whether a looser rung degrades the gradients or the energy conservation;
  - whether a larger `max_cycle` would also converge (not tried);
  - the "previous orbitals" rung (the trials used fresh starts only).
- **F1, F2, F3:** not relevant.

The measured remedy for the seed-2026 set is SA3-CAS(2,2), which has no electronic failure.

## 3. The determinism check

- **Fresh starts in separate processes.** Two fresh base solves were run in separate spawned processes at the failure record of each trajectory (Task 6 section 5):
  - the largest |ΔE| over 17 trajectories is **4.55e-12 eV** (traj_00017);
  - 00006 and 00013 give exactly 0;
  - in 00016 both solves fail at r37.
- **Whole runs.** The `base` variant of Task 7 reproduces M4 run A in all 20 trajectories, in outcome and `t_final_fs` to 4 decimals. The run is from another process and another day, with the same program code. M4 run B had already reproduced run A in outcome, hops and rejections (USER_MANUAL 5.9).
- **What was not seen:** the process-dependent solution of known limitation 3a (water CAS(4,4)/STO-3G).
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
| **F4**, a longer convergence ladder | 00000 and 00007 converge at a looser `conv_tol_grad` | none on seed 2026 (`sa3_cas22` has no electronic failure); it is protection | gradient accuracy and drift with the looser rung; the previous-orbitals rung and `max_cycle` |
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
4. **traj_00016 is not reproduced:** the run converged r37, but every replay mode, the gradient backend and the determinism pair fail there (section 2.3).
5. **SA3-CAS(2,2) has two solutions at the start** in 13 of 20 (ours against SHARC; section 5, caveat 1). Task 6's "SA3 has one solution" holds only inside the replay windows.
6. **SA3-CAS(2,3) and SA2-CAS(4,3) are not cleaner than SA2-CAS(2,2):**
   - SA3-CAS(2,3) switches on its own in 00019 (active-orbital singular value 0.0300);
   - SA2-CAS(4,3) has two solutions in 5 windows and fails at step 0 in 00000 and 00009.

   Sources: Task 6 "does not fit" item 8; Task 7.
7. **SHARC also stops runs.** It stopped 4 base runs and 2 SA3 runs with `QM call was not successful` (error code 256); the cause is not recorded by the harness. It applies no drift or overlap check: 18 of its 20 base runs carry a jump above 0.1 eV, including the S0 drops of 1.9-2.5 eV (00013, 00001).
8. **traj_00008's first jump (r45) is not a switch:** continued equals the stored energies while fresh differs by 0.642 eV. The larger jump at r46 lies outside the window.
9. **The jump is measured on different records.** Task 6 measured it on `steps.log` (every (sub)step); the variant runs and the kept rule use `output.h5`. Base 00001, 00002, 00013, 00016, 00017 and 00018 are below 0.1 eV on `output.h5`; they are lost by outcome anyway.
10. **Minor tool findings.** `replay.log` has an `END` timestamp equal to its `START`. The SHARC `sa3_cas22` run took 2 h 56 min against an estimate of 2.1 h (the probe ran with 1 thread, alone on the machine).

## 7. Honest summary: what was decided after the numbers were seen

- **The proposed setup** (`sa3_cas22`, and `[selection] states = ["S1"]`) was chosen after the seed-2026 numbers were seen. Spec 4.1 rule 4 allows that, but **no measured setup reaches the 16 of 20 that the success rule requires**: the best is 11.
- **The SHARC SA3 run** was made because `sa3_cas22` kept 11. The spec allowed running "the variant chosen in section 3.3" in SHARC; the choice itself came after the numbers.
- **The replay window** (6 records) was fixed before the run. It turned out too short for 5 of the 8 overlap failures, which were therefore not reproduced. Their label rests on SHARC runs that took different paths after the hop, an interpretation made after the numbers were seen. No longer replay was run.
- **The "method" label of the drift class** rests on the SHARC comparison and the fresh solves, as the spec's definition requires. Reading the absence of any rejection as a detection gap in our design, which supports F1, is an interpretation.
- **The variant list** was fixed in the spec before any run; no `[dynamics]`-threshold variant was added. The F1 default (0.1 eV) was fixed in the spec before the runs.
- **One test bound** in Task 3 was relaxed after measurement (ruling P9a1-6: the state-overlap bound of the replay test went from 0.99 to 0.95; measured 0.9863). It is a test of the tool, not a physical threshold.
- **Not tuned:** no setting was changed in response to the seed-2026 kept counts beyond choosing among the measured variants.

## Decisions for the owner

1. **The acceptance setup.** Take `sa3_cas22` (11 of 20 on seed 2026) into the acceptance run, or not.
2. **The 16/20 target.** Whether it stays, given that the best measured setup reaches 11. One option is to run the acceptance as planned and document the expected shortfall (spec 2, rule 3). Another is to first measure the unmeasured proposals of section 5 on seed 2026 in plan 2.
3. **An initial-solution rule for SA3.** Whether to add one, given that SA3 has two solutions at the start geometries in 13 of 20, and if so with which criterion. It would be measured on seed 2026 first.
4. **F1, and F2 as detection.** Whether they enter plan 2 under spec 4.1 rule 1, although their class is labelled "method". The case for them is a detection gap in our design, not a wrong result.
5. **F4.** Whether it enters: it converts 2 of 3 SA2 failures, but `sa3_cas22` had none.
6. **The formal "ours" cases.** Whether 00007 (SHARC converges where we fail) and 00014 (different solution branches) count as "ours" under spec 2, rule 2, which would require a fix with a failing test. Their causes are not identified.
7. **The `sa3_cas22` losses.** Whether plan 2 replays the 9, including the jumps at hop records, before the acceptance run.
8. **`[selection] states = ["S1"]`.** Whether the acceptance run uses it, as measured, or the default selection with `singlets = 3`.
