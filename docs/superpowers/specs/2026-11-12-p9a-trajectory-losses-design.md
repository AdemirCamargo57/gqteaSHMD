# P9a: the trajectory losses of the M4 study (design)

Status: owner-approved design, 2026-11-12. Sub-project P9a of the master plan's P9 (SHARC validation campaign), split by owner decision: **P9a** (this: diagnose and reduce the trajectory losses) first, **P9b** (the ensemble comparison campaign with SHARC, the analysis against SHARC's tools, the statistics questions, the comparison report) afterwards, with its own spec.

## 0. Owner decisions (2026-11-12, multiple choice; every recommended option)

| # | Question | Decision |
|---|---|---|
| 1 | Scope of P9 | Split: P9a = the M4 losses (with SHARC on the same initial conditions as a diagnostic, plus the deferred PySCF test items it touches); P9b = the comparison campaign, analysis vs `populations.py`/`make_fit.py`, minimum-N rule, tau coverage, report. |
| 2 | What counts as done | A measured root cause for every loss class (ours / method / settings), our bugs fixed, and an acceptance re-run that keeps >= 80 % of 20 trajectories; the threshold is fixed now; a miss is documented, not tuned away. |
| 3 | Freedom of the setup | Ethylene and 6-31G\* stay; the number of states, the active space, `dt_fs` and the `[dynamics]` thresholds may change, chosen from the diagnosis; the chosen setup becomes the documented ethylene guidance. |
| 4 | Behaviour changes | Allowed in the `dynamics` stage and the PySCF backend where the measured diagnosis calls for them, each with its own test and a version bump. This reopens O10 item 6 (P8) for the dynamics side; the `analysis` energy-jump diagnostic stays. |
| 5 | Approach | A: diagnose first, then plan the fixes. One spec; plan 1 = diagnosis; plan 2 = fixes and acceptance, written only after the owner approves the diagnosis report. |
| 6 | Design sections 1-4 | Approved as presented, including the stricter definition of a kept trajectory (section 4.3). |

## 1. Background: the M4 losses

M4 (P8 Task 11, USER_MANUAL 5.9; projects `~/p8_m4` and `~/p8_m4b` in WSL, 23 MB each, kept): ethylene SA2-CASSCF(2,2)/6-31G\*, seed 2026, 100 Wigner samples, 20 trajectories started in S1, 100 fs at `dt_fs` 0.5, every other setting at its default. Mean S1 excitation energy 9.362 ± 0.628 eV. Outcomes: finished 2, `overlap_failure` 8, `energy_drift` 7, `electronic_failure` 3.

| Class | n | Observation | Working hypothesis (not established) |
|---|---|---|---|
| `energy_drift` | 7 | 6.5-11 fs, on S1, no hop, 0-4 rejections, jumps of 0.6-0.93 eV | the SA-CASSCF solution changes and no rejection signal sees it (the P7b limitation, Task 10 step 24) |
| `overlap_failure` | 8 | right after the first S1->S0 hop; smallest singular value 0.08-0.44 at dt/8 | the 2-state space at t + dt is not contained in the one at t: active-orbital rotation or a state outside the computed set |
| `electronic_failure` | 3 | SA-CASSCF fails on every rung; one at 0.25 fs | convergence settings or a hard region of the surface |

Both included trajectories carry energy jumps of about 0.32 eV (the analysis flagged 13 of 20).

## 2. Goal, scope and the success rule

**Goal.** Explain and reduce the losses, so that a realistic SA-CASSCF study of ethylene keeps most of its trajectories.

**In scope.** The three loss classes; code fixes in the `dynamics` stage and the PySCF backend where the diagnosis calls for them; the documented ethylene guidance (states, active space, `dt_fs`, thresholds); the deferred P9 items listed for P9a in section 5.

**Out of scope (P9b or later).** Ensemble comparisons with SHARC; the analysis against `populations.py`/`make_fit.py`; the minimum-N rule and the tau coverage study; O17; the deferred items listed for P9b in section 5.

**Success rule (fixed before any new data).**
1. Every loss class has a measured root cause labelled **ours**, **method** or **settings**. "Ours" means our result differs from an independent reference (SHARC on the same initial conditions, or a fresh CASSCF solve at the same geometry). Each label names its evidence.
2. Every class labelled "ours" is fixed, with a test that fails before the fix.
3. **Acceptance:** a new 20-trajectory study with the chosen setup and the fixed code keeps **at least 16 of 20** trajectories (section 4.3 defines "kept"). It uses **seed 2027** (the diagnosis uses the seed-2026 initial conditions), so the setup is not tuned to the trajectories it is judged on. If the threshold is missed, the shortfall and its cause are documented; the threshold is not lowered and the setup is not re-tuned on seed 2027.
4. An honest summary: everything decided after numbers were seen is stated as such.

## 3. Plan 1: the diagnosis

Report scripts only; **no change to the program's behaviour.** New code lives in `tools/diagnosis/` and one SHARC harness in `tools/sharc_compare/`. The scripts write every number to a JSON file and assert nothing; their parsing and conversion helpers have unit tests in the fast tier (no PySCF). They may import `gqteashmd`, PySCF and h5py; `gqteashmd` never imports them. Measurements run on the WSL native filesystem (`~/p9a/...`).

### 3.1 Forensics (`tools/diagnosis/trajectory_forensics.py`)
Input: a project directory. Reads every `output.h5` (`read_output_h5`), `steps.log` (`StepLog`), `dynamics_summary.json` and `electronic.log`. Per trajectory:
- every record with `|ΔE_tot| > 0.1 eV` against the previous record, with the state whose energy jumped most (`ΔE_I` per state), the S1-S0 gap, `min_singular_value`, `rung_changed`, `depth`;
- for `overlap_failure`: the last hop before the end (time, from/to, gap at the hop), the singular values of the last records;
- for `electronic_failure`: the last CASSCF/SCF messages of `electronic.log` (convergence, rung);
- a one-line onset classification per trajectory (jump without hop, failure after hop, convergence failure, none).
Output: one JSON per project. Run on `~/p8_m4/run` first.

### 3.2 Replayed single points (`tools/diagnosis/replay_points.py`)
Input: a trajectory directory and a window of records around its failure (default the 6 records before the first flagged jump or the end, and the failing step's geometry when known). For each geometry in the window, with `PySCFBackend` or PySCF directly:
- (i) **continued:** one backend along the stored geometries from the window start (should reproduce the stored energies; the difference is reported);
- (ii) **fresh:** a new backend at each geometry (SCF guess);
- (iii) **several starting guesses:** the previous point's orbitals projected, the SCF orbitals, and the continued solution of the next point used backwards;
- (iv) **method variants:** SA3-CAS(2,2), SA3-CAS(2,3), SA2-CAS(4,3) (6-31G\*), each continued and fresh.
Recorded per point and variant: state energies, ⟨S²⟩, natural-orbital occupations of the active space, the overlap of the active orbitals with the previous point's active orbitals (singular values of the active-space MO overlap in the AO metric), the state overlap matrix with the previous point, convergence and iterations, wall time.
**Determinism:** each probe point is solved twice in separate processes (fresh start, same settings); any difference is reported (known limitation 3a suggests CASSCF may land on different solutions).
For the three `electronic_failure` geometries it also tries the CASSCF convergence settings (`max_cycle`, `conv_tol`, `conv_tol_grad`, the initial guess).
Reading of the patterns (for the report): (i) and (ii) disagree = a solution switch along the path; all variants agree = a real feature of the surface; a third state coming down in SA3 = an intruder outside the computed set.

### 3.3 Our runner on the same initial conditions (`tools/diagnosis/run_variants.py`)
Runs `trajectory.runner.TrajectoryRunner` with `BackendModel` directly (no orchestrator) on the 20 seed-2026 initial conditions (positions and velocities of `03_sampling/samples.h5` for the samples named in `05_selection/trajectories.json`, initial state S1, the recorded rng streams via `task_rng(2026, "dynamics", k)`). Bypassing the orchestrator is deliberate: a changed `[electronic]` section would change the S0 surface, the optimisation, the frequencies and the samples, so the initial conditions would differ.
Variants (a list in a TOML or JSON file given to the script): `coupling` overlap / nacdr; `dt_fs` 0.5 / 0.25; SA2-CAS(2,2), SA3-CAS(2,2), SA3-CAS(2,3), SA2-CAS(4,3); further `[dynamics]` thresholds if section 3.2 suggests them. Parallel workers (`spawn`, `__main__` guard), threads and memory per worker as in M4. Per variant: the outcome table, the forensics summary of section 3.1, the time per step, and the kept count by the section 4.3 rule. The first variant reproduces M4 (same settings): its outcomes are compared with `~/p8_m4` as a check of the script.

### 3.4 SHARC on the same initial conditions (`tools/sharc_compare/run_sharc_pyscf_ensemble.py`)
Writes the 20 seed-2026 initial conditions as SHARC inputs and runs `sharc.x` with `SHARC_PYSCF.py` as a black box: `nacdr` (the SHARC PySCF interface has no overlaps), the same `dt`, states, active space, decoherence and hopping scheme, 100 fs, built on `run_sharc_pyscf_dynamics.py` (P7b) and its `output.dat` parser. Per trajectory: energy jumps by the section 3.1 rule, the drift, hops, how and when it ended. Runs in parallel within the machine limits. The written SHARC inputs are read back by our own parser in a unit test. If the SA2 runs are affordable, the variant chosen in section 3.3 is run in SHARC too.

### 3.5 The diagnosis report
`docs/superpowers/reports/2026-11-12-p9a-diagnosis.md`, committed. Per loss class: the evidence (numbers and the JSON files they come from), the label (ours / method / settings), the candidate fixes of section 4.2 it supports or rules out, and the setup proposed for the acceptance run (section 4.3). It also reports the determinism check, the time per step of every variant, and every finding that does not fit a class. **Plan 2 is written only after the owner approves this report.**

## 4. Plan 2: the fixes and the acceptance run (rules fixed now)

### 4.1 Rules for fixes
1. A fix enters only if the diagnosis report labels its class "ours" or "settings" and names the evidence.
2. Each fix has a test that fails before it (a replayed geometry, a scripted `ModelBackend` fault, or a PySCF test).
3. A behaviour change raises the `version` of the stage it touches (existing projects then need `--rerun-from <stage>`); CLAUDE.md, USER_MANUAL.md and the analysis follow it (a new outcome string is known to the analysis as an excluded reason and is counted in `outcomes`).
4. Fixes and settings are chosen on the seed-2026 initial conditions only.

### 4.2 Candidate fixes (hypotheses; the report keeps or drops each)
- **F1, a per-step energy check as a rejection signal.** A new `[dynamics]` key (name fixed in plan 2): a (sub)step whose `|ΔE_tot|` exceeds it is rejected and halved like the other signals; a jump that persists at the smallest step ends the trajectory with a new outcome `electronic_discontinuity`, excluded by the analysis. **Default fixed now: 0.1 eV**, by the rule of `energy_jump_eV` (P8 Task 1: the largest per-step jump in correct runs was 7.7e-3 eV <= 0.01 eV).
- **F2, CASSCF orbital continuity.** The overlap of the active orbitals between consecutive steps is measured; below a threshold the step is rejected and re-solved from the previous step's orbitals.
- **F3, guidance and a warning for the state count.** `check_config` of the `dynamics` stage warns when an initial state can be the highest computed state (the existing advice "compute one or two states above the highest initial state", USER_MANUAL 8.16).
- **F4, a longer CASSCF convergence ladder,** if the three non-converged geometries converge with other settings.

### 4.3 The acceptance run
- A full `gqteashmd run` on the WSL native filesystem: ethylene (`tests/data/ethylene.xyz`), 6-31G\*, the chosen `[electronic]` and `[dynamics]` settings, **seed 2027**, `n_samples` 100, `n_trajectories` 20, `tmax_fs` 100, the M4 resources (16 cores, 8 tasks x 2 threads, 800 MB per task), `[analysis]` at its defaults.
- **Kept** = included by the analysis (`finished` or `stopped_in_s0`) **and** not in `energy.flagged` (no energy jump above `energy_jump_eV` = 0.1 eV). Read from `analysis.json`.
- **Pass:** kept >= 16 of 20. Otherwise the shortfall and its cause are documented; no re-tuning on seed 2027.

## 5. The deferred items marked P9 in CLAUDE.md

- **Plan 1** (it extends the SHARC harness): the P7b harness minors: the swap provider gets the finite checks of `BackendModel._snapshot`; `run_sharc_trajectory` takes explicit coupling and interface arguments instead of the `overlap` + `sharc_pyscf` defaults, and a timeout scaled to the run; `step_differences` checks unique, sorted steps; the position numbers are written to the report file, not only printed; the `runQM.sh` comment is corrected.
- **Plan 2** (it touches the dynamics tests anyway): the CASSCF energy-consistency assertion in the non-slow `pyscf` tier; the finer dt pair (0.25 / 0.125 fs) for the thin dt^2 margin of P7b (b); which recovery path the kill test (f) takes, with its output kept instead of discarded; the step-24 SHARC assertion re-checked after any fix; an INFO log line "resumed from restart generation N" in the runner (M4 finding).
- **P9b:** a SHARC population check that can fail (P(S0) >= 1e-3 in the window, or a replayed hopping-on comparison); detecting a TDA/TDDFT root skip that survives the extra roots; the minimum-N rule; the tau coverage study at N = 20, 50, 100.

## 6. Testing, documents, constraints

- **Tests.** Plan 1: unit tests of the helpers (forensics parsing on small written `output.h5`/`steps.log` files; the SHARC input writer read back by our parser) in the fast tier; the probe and ensemble scripts are not collected by pytest. Plan 2: every fix with a test that fails before it; the full suite (`pytest -W error` with SHARC and ORCA) at the end of each plan.
- **Clean room.** SHARC is executed only (`sharc.x`, `SHARC_PYSCF.py`), from `tools/sharc_compare/`; input and output formats come from the existing harness and the public SHARC 4.1 manual; nothing in the installation is read.
- **Dependency rules.** Unchanged: `dynamics/` imports nothing from `electronic/`, `stages/`, the orchestrator or `config/`; `trajectory/` as in P7b; `analysis/` as in P8; `import gqteashmd.cli` imports no pyscf, geometric, h5py or scipy. `tools/` is never imported by `gqteashmd`.
- **Seeds.** The diagnosis uses the seed-2026 streams exactly as the M4 stage did (`task_rng(2026, "dynamics", k)`); the acceptance run uses `project.seed = 2027`.
- **Documents.** The diagnosis report is committed (section 3.5). CLAUDE.md and USER_MANUAL.md are updated in the same branch: a new USER_MANUAL section with the ethylene findings and guidance, section 10 with the root causes, and every behaviour change of plan 2.
- **Branch and files.** Branch `feat/p9a-trajectory-losses`; this spec; plans `docs/superpowers/plans/2026-11-12-p9a1-diagnosis.md` and, after the report, `…-p9a2-fixes.md`; ledgers under `.superpowers/sdd/<plan>/`.

## 7. Risks

1. SHARC may fail like ours. The class is then labelled method or settings, and the fix is guidance, not code.
2. PySCF CASSCF may not be deterministic across processes (limitation 3a). The replay solves each probe point twice in separate processes, so a solution switch is not mistaken for a bug.
3. SA3 or larger active spaces cost more per step. The time per step is measured; a variant too slow for 20 x 100 fs on this machine (16 cores, about 7.7 GB) is reported as such.
4. The 80 % target may be out of reach for SA-CASSCF at this level. The shortfall is documented (section 2, rule 3).
5. Changing the electronic method changes the S0 surface and so the initial conditions; the diagnosis therefore runs on fixed initial conditions (section 3.3), and the acceptance run regenerates them for its own setup with seed 2027.
