# P7b design: the `dynamics` stage with PySCF (singlets only)

> **Erratum (from the implementation plan, 2026-11-11):** (1) Section 7: the stage is enabled
> iff the input has a `[dynamics]` table; without it the workflow ends at `selection` as before
> (plan ruling R1), instead of failing `check_stage_inputs`. (2) Section 7: `finalize` does not
> build missing `output.h5` files; `run_task` always writes it and `finalize` lists missing ones
> (plan ruling R5). (3) Sections 3.2 and 6: `steps.log` holds every accepted (sub)step and
> `output.h5` is thinned by `output_every`, keeping every hop, flag and rejection (plan ruling
> R2). The plan's Rulings R1-R10 are binding where they differ from this text.

Status: approved by the owner on 2026-11-11 (design in chat; this file records it). The
implementation plan follows in `docs/superpowers/plans/`.

Owner decisions of 2026-11-11 (multiple choice, recorded here):

1. **Coupling:** `coupling = "overlap"` (local diabatization) is the default for every method;
   `"nacdr"` is allowed for `casscf` only.
2. **Termination (O8):** end on the energy-drift limit (0.5 eV default) and on a persistent
   electronic-structure failure; an optional `stop_after_s0_fs` (off by default).
3. **Step rejection:** retry an unreliable step with smaller steps (halving), then end with an
   outcome.
4. **Output and disk (O9):** output every step (thinnable with `output_every`), restart every
   step, only the last electronic snapshot kept.
5. **Three additions to the master plan, approved with the design:** (a) a step log instead of a
   live `output.h5`; (b) gradients of the active state only; (c) one orchestrator hook for
   extending finished trajectories (O16).

## 1. Goal and scope

The stage `dynamics` runs one surface-hopping trajectory per entry of `trajectories.json`
(written by `selection`) with PySCF electronic structure, resumes after any crash or stop, and
records every trajectory's ending as an outcome. It reuses the validated core of P7a-1/P7a-2
(`gqteashmd.dynamics`) unchanged.

**In scope:** singlets (`[electronic] triplets = 0`), `casscf`, `tda`, `tddft`; the
`[dynamics]` input section; step rejection; per-step restart; outcomes; `dynamics_summary.json`;
extension of finished trajectories (O16); validation (section 9).

**Out of scope:** triplets and SOC from PySCF (O7), TDDFT NACs, an S1/S0 gap rule for linear
response, thermostats, `eselect` with PySCF (refused as an input error), the `analysis` stage
(P8), ensemble comparison with SHARC (P9).

## 2. Approach

An adapter and a runner live **outside** `gqteashmd.dynamics`. The core keeps its dependency rule
(`dynamics/` imports nothing from `electronic/`, `stages/`, the orchestrator or `config/`) and
stays bit-identical (the golden test `tests/dynamics/test_golden.py` stays green in every task).

Rejected alternatives: extending `driver.run_trajectory` (puts backend state, rejection and files
into the core and endangers the golden tests); a stage that calls the backend directly (duplicates
the validated step logic).

## 3. Components

New package `gqteashmd.trajectory` (it may import `dynamics/` and `electronic/base.py`, never
`stages/` or the orchestrator; PySCF only through the backend; h5py imported inside functions):

| Module | Responsibility |
|---|---|
| `trajectory/model.py` | `BackendModel`: an `ElectronicModel` over any `ElectronicBackend` |
| `trajectory/runner.py` | The trajectory loop: steps, rejection, termination, output, restart, resume, extension |
| `trajectory/store.py` | Files of one trajectory: restart generations, step log, `output.h5`, `progress.json` |
| `trajectory/outcomes.py` | `TrajectoryOutcome` (`StrEnum`) and the per-trajectory result dict |
| `stages/dynamics.py` | `DynamicsStage` (thin: plan, run_task, finalize, fingerprint, check_config) |
| `config/schema.py` | `DynamicsSection` (`[dynamics]`) and its cross-field rules |
| `testing/fake_backend.py` | `ModelBackend`: an `ElectronicBackend` over an analytic model, for fast tests |

### 3.1 `BackendModel`

- `evaluate(q, previous)` builds a `Request` at `q` (flat bohr reshaped to `(natom, 3)`) with
  `overlaps = True` always, `gradients = {active}` and, for `coupling = "nacdr"`, all NAC pairs
  I < J. It returns an `ElectronicSnapshot`:
  - `energies` in state order (S0, S1, ...), `hamiltonian = diag(energies)` (complex), so U stays
    `None` and the driver takes its MCH fast path (singlets only);
  - `gradients` (n, f): the computed row of the active state; **every other row is NaN**, a
    tripwire: if the driver ever read another row, the energy would become NaN and the run would
    stop. A fast test asserts that no NaN reaches a record over trajectories with hops;
  - `nacs` (n, n, f): the backend's antisymmetrised NACs for `nacdr`, zeros for `overlap`;
  - `overlaps`: the backend's phased overlap matrix, **Löwdin-orthonormalised**
    (`overlaps.tracking.lowdin_orthonormalize`), because `propagate_diagonal` assumes a unitary S
    and does not check it. The raw smallest singular value is kept for the runner.
- The active state is set by the runner before each step (`model.active = state.active`); the
  model does not guess it.
- `last_meta` exposes the backend metadata of the last `compute` (ladder rungs, smallest singular
  value, root flips, wall time) for rejection and the step log.
- `refresh_active_gradient(state)`: after a hop, `compute_more(gradients={new active})` at the
  same geometry and a new snapshot with that row filled. This is why gradients of the active
  state only are enough: the core reads the gradient of the *old* active state at q' (velocity
  completion, step 5 of the step order) and the new active state's gradient only at the start of
  the next step (the acceleration, step 1).
- `masses`: from `samples.h5` (`masses`, amu) converted to m_e and repeated per Cartesian
  component.

### 3.2 Runner and step order

One task = one trajectory. The core's `driver.step` is called once per (sub)step; the runner does
not reimplement any part of the step order (USER_MANUAL.md section 8.12).

1. **Start** (no restart on disk): read the sample's positions and velocities (bohr; bohr per
   atomic unit of time) and the initial state (`"S1"` -> index 1, `c0 = e_1`); evaluate at q0
   without a previous geometry; `driver.initial_state(..., options=options)`; write restart
   generation 0 and the first log record (step 0).
2. **Each step:** set `model.active`; save the RNG state; `driver.step`; check the rejection
   signals (section 4); on a hop call `refresh_active_gradient`; termination checks (section 5);
   append the step record (every `output_every` steps and always the last); write the restart
   (section 6).
3. **End:** build `output.h5` from the log, write `progress.json` and return the result dict.

Random numbers: the core draws exactly one per call of `driver.step`. A rejected step's draw is
undone by restoring the saved RNG state, so the stream depends only on the accepted (sub)steps.

## 4. Step rejection

**Signals** (checked after each `driver.step` at the new geometry):

- the electronic calculation raised a non-retryable `ElectronicStructureError`;
- the raw smallest singular value of the state-overlap matrix is below `overlap_threshold`
  (default 0.5);
- CASSCF left its first ladder rung (`meta["casscf"] != "previous orbitals"` with a previous
  geometry), i.e. the solution may have jumped. TD and SCF rung changes are recorded, not
  signals.

A retryable error (`MemoryError`, `OSError` inside PySCF) is not a rejection: it propagates as a
retryable `TaskError` and the orchestrator retries the task, which resumes from the restart.

**Procedure:** the restart is written after every **accepted (sub)step**, so the current
restart generation always holds time t. Discard the step; restore the backend snapshot of time t
(`backend.load_state` from the current generation) and the RNG state; redo the interval [t, t + dt] as two steps of
dt/2 (each of which may be halved again), down to dt / 2^`max_step_halvings` (default 3, so
dt/8). After the interval the trajectory returns to the full dt, so it stays on the time grid of
`dt_fs` and the outputs of different trajectories line up. Substeps use
`dataclasses.replace(options, dt_au=dt / 2^k)`; `nsubsteps` stays per (sub)step.

**At the smallest step:** a CASSCF rung change with a good overlap is accepted and flagged in the
log; an overlap still below the threshold ends the trajectory with `overlap_failure`; a failed
calculation ends it with `electronic_failure`. Every rejection and every flag is in the step log
and counted in the result.

The first geometry (step 0) has no previous geometry and no overlaps; a failed calculation there
ends the trajectory with `electronic_failure` (nothing to roll back to).

## 5. Outcomes and termination

`TrajectoryOutcome` (`StrEnum`), recorded in the task result; the task is DONE for every outcome:

| Outcome | When |
|---|---|
| `finished` | time reached `tmax_fs` |
| `energy_drift` | \|E_tot - E_tot(0)\| > `energy_drift_limit_eV` (or non-finite) after an accepted step |
| `stopped_in_s0` | `stop_after_s0_fs` set and the active state was S0 for that long without interruption |
| `overlap_failure` | rejection at the smallest step because of the overlap (section 4) |
| `electronic_failure` | rejection at the smallest step because the calculation failed, or a failure at step 0 |

Only infrastructure failures (crash, invalid input files, a retryable error that exhausts
`max_attempts`) are task failures and count against `max_failed_fraction` (D10).

## 6. Files of one trajectory and restart

Task directory `06_dynamics/tasks/traj_<k:05d>/`:

- `restart/gen_<g>/`: `state.npz` (core restart format 2, `dynamics.restart.save_state`, which
  stores q, v, c, the snapshot, U, time, step, active, E_tot(0) and the RNG state),
  `electronic.npz` (`backend.save_state`), `meta.json` (step, time, accepted-step counters, S0
  timer, rejection counters, log length in records, and the position inside a halved interval:
  its end time and the current halving depth, so a resume in the middle of a halved interval
  finishes it with the same substep size before returning to the full dt). A generation is written into a temporary
  directory and renamed; `restart/current` (a small file written atomically) names the newest
  complete generation; the older generation is deleted only after `current` points to the new
  one. A kill at any instant leaves one complete generation.
- `steps.log`: append-only fixed-size binary records (one per logged step): step, time, active
  state, energies, E_tot, E_kin, |c|^2, hop from/to, frustrated, random number, smallest
  singular value, rung flags, rejection depth, q and v. On resume it is truncated to the length
  stored in the restart's `meta.json`, so a step written after the last restart is not
  duplicated.
- `output.h5`: built atomically from `steps.log` when the task ends (every outcome) or when the
  stage finalizes; `units` attribute on every dataset (house rule). Not readable while the
  trajectory runs (owner decision 5a): an HDF5 file killed during a write can become unreadable,
  which is not acceptable for runs of days.
- `progress.json`: step, time in fs, active state, outcome so far; written atomically each step.
- `electronic.log`, `scratch/`: the backend's log and scratch (as in the other stages).

**Resume** (restart present): read `current`, load the core state with
`dynamics.restart.load_state`, load the backend snapshot, truncate the log, continue. The core
restart stores no model identity or options (P7a deferral); the stage fingerprint covers them
(section 7), so a resumed trajectory always continues with the same input.

**Reproducibility:** with the analytic fake backend, resume is bit-identical (asserted). With
PySCF it is reproducible to the convergence of SCF/CASSCF, not to the bit (iterative solvers
restarted from a saved snapshot); measured in the plan and asserted with a tolerance fixed before
the measurement.

**Stop and kill:** no signal handler in the worker. The executor's SIGTERM ends the worker; every
file is written atomically, so at most the step in progress is lost, and the rerun resumes it.

## 7. Stage, input and fingerprint

`DynamicsStage`: `name = "dynamics"`, `depends_on = ("prepare", "sampling", "selection")`,
`config_sections = ("electronic", "dynamics")`, `required_modules = ("pyscf>=2.6", "h5py>=3.9")`.

- `plan`: one task per entry of `trajectories.json`, ids `traj_00000` ... (index k as in the
  file); reads only `trajectories.json`.
- `run_task`: checks that the sample index exists in `samples.h5` (else a non-retryable
  `TaskError` naming `--rerun-from sampling`), builds one backend per trajectory with the full
  `[electronic]` settings (`stage_backend`), the RNG `task_rng(seed, "dynamics", k)` (the stream
  recorded in `trajectories.json`), and runs the runner. Returns `{outcome, t_final_fs, n_steps,
  n_hops, max_drift_eV, n_rejections, n_flagged, wall_time_s}`.
- `finalize`: builds any missing `output.h5`, writes `dynamics_summary.json` (counts per outcome,
  final times, hops, drift and rejection statistics) and `dynamics_summary.md`; raises
  `StageFailedError` if no trajectory reached any outcome.

`[dynamics]` (`DynamicsSection`, strict pydantic, `extra="forbid"`; the table must be present
because `tmax_fs` has no default; a pipeline without it fails `check_stage_inputs`, exit 2):

| Key | Default | Rule |
|---|---|---|
| `tmax_fs` | required | > 0 |
| `dt_fs` | 0.5 | > 0, < `tmax_fs` |
| `nsubsteps` | 25 | >= 1 |
| `coupling` | `"overlap"` | `"overlap"` \| `"nacdr"`; `nacdr` only with `casscf` |
| `hopping` | `"sharc"` | `"sharc"` \| `"fssh"` \| `"none"`; `fssh` refused with `overlap` (core rule) |
| `ekin_correction` | `"parallel_vel"` | `parallel_nac` only with `nacdr` |
| `frustrated` | `"keep"` | `"keep"` \| `"reflect"` |
| `decoherence` | `"edc"` | `"edc"` \| `"none"` |
| `decoherence_param_Eh` | 0.1 | > 0 |
| `edc_convention` | `"sharc"` | as the core |
| `substep_interpolation` | `"midpoint"` | as the core |
| `energy_drift_limit_eV` | 0.5 | > 0 |
| `stop_after_s0_fs` | none | > 0 when given |
| `overlap_threshold` | 0.5 | 0 < x < 1 |
| `max_step_halvings` | 3 | 0..6 |
| `output_every` | 1 | >= 1 |
| `max_failed_fraction` | 0.1 | 0 <= x < 1 |

`[electronic] triplets` must stay 0 (unchanged rule). A `check_config` hook adds: `singlets >= 2`
(no dynamics on one state) and a **warning** (not an error) for `tda`/`tddft` with
`grid_level < 5` (PySCF DFT gradients omit the grid response).

**Fingerprint:** the whole `[electronic]` section (including `overlap_norm_kept`, which changes
the overlaps), `[dynamics]` **without** `tmax_fs` (O16, section 8) and `max_failed_fraction`
(finalize-only), plus `project.seed`; the orchestrator chains the upstream fingerprints.
`output_every` is hashed although it does not change the trajectory, because changing it on a
resumed trajectory would mix record spacings in one log.

## 8. Extending finished trajectories (O16)

`tmax_fs` is not hashed. Raising it after the stage is DONE must continue the `finished`
trajectories (other outcomes are final). The orchestrator skips a DONE stage with an unchanged
fingerprint, so one hook is added to the stage contract:

- `Stage.reopen_tasks(ctx) -> list[str]` (default `[]`), called by the orchestrator for a DONE
  stage with an unchanged fingerprint. A non-empty list sets those tasks back to PENDING and the
  stage to RUNNING, then the stage runs as usual (only those tasks), and `finalize` runs again.
- `DynamicsStage.reopen_tasks` returns the tasks whose result is `finished` with
  `t_final_fs < tmax_fs` (read from the state database results, not from upstream artifacts).
- The runner resumes from the restart and continues to the new `tmax_fs`; the outcome is
  recomputed.

Lowering `tmax_fs` reopens nothing (results are kept; the summary reports the longer times). The
hook is the only orchestrator change; it is tested with the fake stages and with the fake
backend.

## 9. Validation and tests

Fast tier (no PySCF, `testing/fake_backend.py`):

- `BackendModel` over the fake backend reproduces the direct model runs of the core (energies,
  populations, hops) on Tully and LVC singlet models, `nacdr` bit for bit; `overlap` within
  1e-12 (the Löwdin step).
- The NaN tripwire: no NaN in any record over trajectories with hops.
- Rejection, rollback and halving forced by a scripted backend (failure, low singular value, rung
  change at chosen steps); the RNG is restored (accepted-step streams equal those of a run without
  rejection where the interval is the same); each outcome reached.
- Restart: kill at chosen points (between log write and restart write, inside a generation
  write) and resume bit-identically; log truncation; `output.h5` built after every outcome.
- Stage: plan from `trajectories.json`, fingerprint rules, O16 reopen, `check_config`, config
  errors (exit 2), orchestrator kill and resume with the fake backend.

PySCF tier (marker `pyscf`, and `slow` where long):

- Energy conservation on a short ethylene SA2-CASSCF(2,2)/STO-3G trajectory and on a water TDA
  trajectory, and the dt^2 scaling of the drift (bounds fixed before the run).
- Resume with PySCF: continuous vs stopped-and-resumed, difference within a tolerance fixed before
  the measurement.
- End-to-end: the whole pipeline through `dynamics` on water HF/STO-3G-sized settings (few samples,
  2 trajectories, a few fs), including a kill and resume.

SHARC 4.1 black box (marker `sharc`, never in CI; formats from the public manual only):

- **Asserted:** one hopping-off trajectory of `sharc.x` + `SHARC_PYSCF.py` against ours on ethylene
  SA2-CASSCF(2,2), positions, velocities, energies and populations step by step; tolerances fixed
  from the print precision before comparing; the time of one SHARC step is measured first and the
  length is chosen from it.
- **Report:** the swap test with our backend as `QM.out` provider behind `sharc.x` (reuses the
  P7a-2 provider pattern).

The house rules apply: failures are reported, not loosened; any assertion changed after numbers
were seen is labelled post hoc in the test and in USER_MANUAL.md.

## 10. Documents

`CLAUDE.md` and `USER_MANUAL.md` are updated in the same branch: the `[dynamics]` section, the
stage, outputs, outcomes, rejection, restart, O8/O9/O16 resolved, limitations (no live
`output.h5`; PySCF resume not bit-identical; singlets only).
