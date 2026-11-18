# The hop energy option: report

Status: **`hop_energy = "previous"` is implemented as an option; the default `"current"` is unchanged and bit-identical to `main`. With the option, the replayed trajectory agrees with SHARC 4.1 over the pre-registered window (steps 0 to h + 20), which it did not do before (asserted). The ensemble rerun (report) removes the per-hop energy step, moves our drift tail towards SHARC's, and leaves the losses unchanged: 33 of 100 under both rules, all after a hop.** No bound was changed and nothing was re-tuned.

Date: 2026-11-18. Branch `feat/hop-energy-option` (from `main` `4e61daa`). The spec is `docs/superpowers/specs/2026-11-18-hop-energy-option-design.md` (owner-approved 2026-11-18; erratum at its top) and the plan is `docs/superpowers/plans/2026-11-18-hop-energy-option.md` (plan commits `b191ebf`, `84e59f2`). The code tasks are the commits `c1ec7de`, `03af04a`, `b1b073f` and `972f478`. Task 5 is a measurement with no diff.

**Sources of every number:** the task reports in the git-ignored ledger `.superpowers/sdd/2026-11-18-hop-energy-option/` (`task-1-report.md` to `task-5-report.md`, `task-3-run.log`, `progress.md`); the JSON files on the WSL native filesystem: `~/hop/report.json` (the P9b run against the new run), `~/hop/swap_previous.json` (the new run against the P9b swap run), `~/hop/ours_previous/run` (the new project) and, for P9b, `~/p9b/ours/run`, `~/p9b/compare/swap_fixed.json`.

**Labels.** *Asserted*: a pytest assertion or a rule fixed in the spec before any run. *Report*: measured, with no rule. *Inference*: not established by a measurement.

## 1. What was built

- **The rule.** `DynamicsOptions.hop_energy`, one of `HOP_ENERGIES = ("current", "previous")`, default `"current"`; any other value is a `ValueError` naming the option. At a hop attempt from the active state $\lambda$ to $j$ (step 8 of the step order):
  - `"current"`: $\Delta E = E_j(\mathbf q') - E_\lambda(\mathbf q')$, as before;
  - `"previous"`: $\Delta E = E_j(\mathbf q') + K(\mathbf v') - E_{\text{tot,prev}}$, where $\mathbf v'$ is the Verlet-completed velocity before the adjustment and $E_{\text{tot,prev}}$ the total energy of the step's starting state. `adjust_velocity` is unchanged. With `parallel_vel` an accepted hop gives $K_{\text{new}} = E_{\text{tot,prev}} - E_j$, so the record's $E_{\text{tot}}$ equals the previous record's. A hop that would need $K_{\text{new}} < 0$ is frustrated.
  - With `parallel_nac` the same $\Delta E$ is used. That is our own definition: P9b measured SHARC with `parallel_vel` only.
  - The step order, the one random number per step, the restart format and the record fields are unchanged.
- **The key.** `[dynamics] hop_energy` of a project (`DynamicsStageSection`) and of `model-dynamics` (`DynamicsSection`, Tully and LVC), the same two values, the default taken from `DynamicsOptions`; a line in the `init` template.
- **The fingerprint.** The `dynamics` stage fingerprint contains `hop_energy` only when it is not `"current"`. Existing projects stay up to date; changing the key is stale (exit 3) and needs `--rerun-from dynamics`. `DynamicsStage.version` stays 2.
- **Tests.**
  - Task 1, `pytest tests/dynamics -q -W error -m "not slow"`: **391 passed** (asserted). The golden test is unchanged and green.
  - Task 2, `pytest tests/config tests/stages/test_dynamics.py tests/test_cli_model_dynamics.py tests/test_cli.py -q -W error -m "not slow and not pyscf"`: **173 passed, 2 deselected** (asserted).
  - Task 3, the live replay file `-k replayed`: **2 passed, 18 deselected, 1 xfailed in 468.85 s** (asserted).
  - Task 4, `tests/validation/test_hop_energy_report.py`: **4 passed** (asserted).
  - The full suite has not run yet.
- **The report script.** `tools/validation/hop_energy_report.py` compares two projects (JSON out, nothing asserted, never imported by `gqteashmd`).

## 2. The replay with `hop_energy = "previous"` (asserted)

The P9b setup, unchanged: start 9 (`rngseed` 10, `tests/data/ethylene_hop_start.json`), `sharc.x` on our backend, ethylene SA3-CASSCF(2,2)/6-31G\*, `overlap`, EDC, `endpoint`, `parallel_vel`, `keep`, 80 steps of 0.5 fs, replayed random numbers. Only our side changes. The test is `test_replayed_trajectory_with_the_previous_hop_energy` (markers `sharc`, `pyscf`, `slow`); the two P9b tests are unchanged (the pre-hop one passes, the full-window one on the default rule is still a strict xfail with the same numbers).

**The window** was fixed in the spec before any run with the option: steps 0 to h + 20. The hop is at step h = 23 (11.5 fs), so the window has 44 records (natural end 81). It must be able to fail: h <= 59, P(S0) >= 1e-3 in the window (max P(S0) 1.000), the same active state at every step.

| Window maximum (steps 0-43) | Bound | Default rule (P9b, steps 0-80) | `"previous"` (asserted) |
|---|---|---|---|
| positions | 1e-5 bohr | 0.455 | **2.93e-07** |
| velocities | 1e-6 | 2.79e-3 | **1.99e-09** |
| populations | 1e-5 | 2.87e-4 | **1.41e-07** |
| hopping probabilities | 1e-5 | 5.19e-4 | **3.10e-07** |
| \|E_tot(h) - E_tot(h-1)\| | 1e-9 Eh | (1.71e-4 Eh at the hop) | **0.0** |
| active state | identical | identical | identical |

**Over all 81 records (report; the growth after the window is not asserted):** dq 4.68e-05 bohr, dv 3.38e-07, dpopulations 1.41e-07, dprobabilities 3.10e-07.

**Per step, from h - 1 to h + 20** (`[hop_energy previous]` lines of the run; report):

| step | dq (bohr) | dv | dpop | dprob | E_tot ours (Eh) |
|---|---|---|---|---|---|
| 22 (h-1) | 1.26e-07 | 1.18e-09 | 1.02e-07 | 1.04e-07 | -77.6244922715 |
| 23 (h) | 1.17e-07 | 1.55e-09 | 1.41e-07 | 3.10e-07 | -77.6244922715 |
| 24 | 1.16e-07 | 1.54e-09 | 4.52e-09 | 0.00e+00 | -77.6247593328 |
| 25 | 1.15e-07 | 1.71e-09 | 2.02e-08 | 2.42e-08 | -77.6248545267 |
| 26 | 1.15e-07 | 1.99e-09 | 2.97e-08 | 3.59e-09 | -77.6248380147 |
| 27 | 1.25e-07 | 1.91e-09 | 4.86e-09 | 1.98e-08 | -77.6247226854 |
| 28 | 1.45e-07 | 1.12e-09 | 1.51e-08 | 0.00e+00 | -77.6245411548 |
| 29 | 1.62e-07 | 9.54e-10 | 9.90e-09 | 0.00e+00 | -77.6244127653 |
| 30 | 1.78e-07 | 1.56e-09 | 9.60e-10 | 6.47e-09 | -77.6245108101 |
| 31 | 1.91e-07 | 1.83e-09 | 4.06e-10 | 4.17e-10 | -77.6248078607 |
| 32 | 2.03e-07 | 1.78e-09 | 3.59e-09 | 1.50e-08 | -77.6252252237 |
| 33 | 2.14e-07 | 1.49e-09 | 2.13e-10 | 0.00e+00 | -77.6255373587 |
| 34 | 2.24e-07 | 1.45e-09 | 1.71e-09 | 7.16e-09 | -77.6250969500 |
| 35 | 2.30e-07 | 1.68e-09 | 1.65e-09 | 7.50e-09 | -77.6247941672 |
| 36 | 2.35e-07 | 1.90e-09 | 1.24e-10 | 1.15e-09 | -77.6245999382 |
| 37 | 2.38e-07 | 1.64e-09 | 1.93e-09 | 5.55e-09 | -77.6245406743 |
| 38 | 2.41e-07 | 1.30e-09 | 1.32e-10 | 2.29e-09 | -77.6246165610 |
| 39 | 2.42e-07 | 1.31e-09 | 4.05e-10 | 0.00e+00 | -77.6247552294 |
| 40 | 2.55e-07 | 1.50e-09 | 1.29e-10 | 1.09e-10 | -77.6248771910 |
| 41 | 2.69e-07 | 1.56e-09 | 5.44e-10 | 1.81e-09 | -77.6249444665 |
| 42 | 2.82e-07 | 1.62e-09 | 5.76e-10 | 1.28e-09 | -77.6249625817 |
| 43 (h+20) | 2.93e-07 | 1.68e-09 | 1.45e-09 | 5.89e-09 | -77.6249618015 |

**With the default rule (P9b lines of the same run, report)** the same quantities after the hop were: step 23 dv 3.25e-06, dpop 1.13e-04, E_tot sharc -77.6244922546, ours -77.6243213063 (1.71e-4 Eh higher); step 30 dq 4.95e-04, dv 5.00e-06. With the option the hop-step velocity difference is 1.55e-09 and the energies agree.

## 3. The ensemble (report; nothing asserted)

The P9b run `~/p9b/ours/run` (ethylene SA3-CASSCF(2,2)/6-31G\*, seed 2028, 100 trajectories, 100 fs, `overlap`) was copied to `~/hop/ours_previous`; its `input.toml` differs by exactly the line `hop_energy = "previous"` in `[dynamics]`. `gqteashmd run input.toml --rerun-from dynamics` ran detached from 2026-11-18 18:44:56, exit 0 after 6621 s (dynamics 1 h 50 min 09 s; P9b: 1 h 56 min; analysis 7 s). Same initial conditions and random streams.

| quantity (all 100 trajectories unless stated) | `"current"` (P9b) | `"previous"` |
|---|---|---|
| finished | 67 | 67 |
| overlap_failure | 23 | 26 |
| electronic_discontinuity | 10 | 7 |
| kept (`is_kept`) | 67 | 67 |
| lost / lost after a hop | 33 / 33 | 33 / 33 |
| hops / frustrated | 233 / 2 | 237 / 2 |
| rejections (energy, overlap, rung) | 212 (83, 128, 1) | 202 (63, 139, 0) |
| flagged steps | 1 | 0 |
| max_drift_eV median / p95 / max | 0.05458 / 0.18384 / 0.27348 | 0.05338 / 0.15808 / 0.29977 |
| record \|dE\| at hops, eV, median / p95 / max | 0.003859 / 0.03748 / 0.09889 (n 233) | **0.0 / 0.0 / 0.0** (n 237) |
| record \|dE\| elsewhere, eV, median / p95 / max | 0.002034 / 0.01517 / 0.09936 (n 17503) | 0.002047 / 0.01513 / 0.09978 (n 17399) |
| analysis N included / selected | 67 / 100 (33 % excluded, warning) | 67 / 100 (33 % excluded, warning) |
| t0, fs [95 %] | 12.84 [6.66, 22.13] | 14.92 [8.12, 24.52] |
| tau, fs [95 %] | 80.14 [57.62, 110.06] | 77.16 [54.91, 106.37] |

**Outcome changes: 8 of 100** (traj_00012 and 00029 `electronic_discontinuity` to `finished`; 00015 and 00091 `finished` to `overlap_failure` and `electronic_discontinuity`; 00061, 00067 and 00080 `electronic_discontinuity` to `overlap_failure`; 00074 `overlap_failure` to `electronic_discontinuity`). Two losses became kept and two kept became losses: the net is 0. Every loss under both rules comes after a hop.

**The swap rules** (`ensemble_compare.py` of the new run against the P9b swap run, no `--asserted`, so a report; P9b's asserted run is `~/p9b/compare/swap_fixed.json`):

| | P9b (`"current"`) | `"previous"` |
|---|---|---|
| kept ours / SHARC / paired | 67 / 58 / 48 | 67 / 58 / 48 (the pair set differs by one: P9b has traj_00091, the new set traj_00012) |
| populations | pass; max abs 0.1875, worst S0 at 34 fs 0.125 vs bound 0.212 | pass; max abs 0.2083, worst S0 at 34 fs 0.125 vs bound 0.210 |
| tau | pass; 93.55 (18.64) vs 53.89 (14.22), d 39.65, bound 93.78 | pass; 97.01 (19.81) vs 51.99 (13.90), d 45.01, bound 96.79 |
| t0 | pass; 12.63 (5.17) vs 34.40 (7.98), d -21.77, bound 38.01 | pass; 12.77 (5.69) vs 34.33 (7.85), d -21.56, bound 38.79 |
| first-hop KS | pass; D 0.213, p 0.2395 | pass; D 0.234, p 0.1529 |
| quantum max abs diff | 0.1697 | 0.1904 |
| ours drift over the kept, eV, median / p95 / max | 0.0477 / 0.1995 / 0.2735 | 0.0464 / **0.1352** / 0.2883 |
| SHARC drift over the kept, eV, median / p95 / max | 0.0406 / 0.1079 / 0.1539 | 0.0406 / 0.1079 / 0.1539 |

The SHARC-side numbers that differ (tau 51.99, hop pairs) are not a changed SHARC run; they follow from the one-trajectory difference in the pair set.

## 4. What was decided after numbers were seen

- **Thresholds, bounds, window, seed, settings: nothing.** Every bound and the window were fixed in the spec before any run with the option; no number changed them.
- **Process rulings** (ledger): H1, the controller started the Task 5 run itself, detached, as in the P9b ruling P3, because a 2 h run must not depend on a subagent's lifetime (Task 3 showed how a tool timeout breaks a live run: repeated launches with the same `--basetemp` wiped the working directories of earlier runs, and the controller relaunched one clean run); H2, the controller made Task 5's recording (two report commands and the report, numbers copied from the JSON). S1, Task 1 left the one `model-dynamics` defaults test red until Task 2 (the plan's own order). None of these changed a measurement or a rule.

## 5. The honest summary

- **Done and asserted:** the rule is in, the default is bit-identical, and the replay with `"previous"` agrees with SHARC over 20 steps after its first hop within the pre-registered bounds (dq 2.93e-07 against 1e-5 and the others within 1e-6 to 1e-5), where the default rule failed at the hop. The E_tot step at the hop is 0.0.
- **What this shows:** the post-hop difference of P9b was the hop energy convention, to the precision of this one trajectory. The P9b inference (SHARC keeps the previous record's total energy) is therefore supported: our driver with that rule reproduces SHARC's post-hop dynamics here. It is one trajectory, one hop, one system, `parallel_vel`; the window is 20 steps.
- **Not shown:** the growth after the window is not asserted (dq 4.68e-05 over all 81 records, above the bound but not tested); no other system, coupling, `parallel_nac` or SOC case was compared with SHARC.
- **The ensemble (report):** the option removes the per-hop energy step (0.0 at all 237 hops) and lowers our drift p95 over the kept from 0.1995 to 0.1352 eV (SHARC 0.1079). This is consistent with the P9b inference that the hop convention causes our larger drift tail; it does not prove it. It does **not** change the losses: 33 under both rules, all after a hop, so the kept count, N and the excluded fraction are unchanged. tau and t0 moved within their intervals. The swap verdicts are unchanged and the fit comparison stays weak (the tau difference grew from 39.65 to 45.01 fs against an equally wide bound). One ensemble, one seed, no repetition: the eight outcome changes are within what a different but equivalent trajectory could do, and no significance is claimed.
- **Next:** the losses after a hop are the next sub-project (owner decision 2026-11-18, spec section 0 item 4). They are not caused by the hop energy convention.
- **Not run:** the full suite (the controller runs it after this task).
