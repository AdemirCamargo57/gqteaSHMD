# gqteashmd: Master Implementation Plan

**Date:** 2026-11-06
**Spec:** [`SHMD_implementation_strategy.md`](../../../SHMD_implementation_strategy.md) (the strategy report, PySCF-first revision)
**First detailed sub-plan:** [`2026-11-06-p1-foundation-orchestrator.md`](2026-11-06-p1-foundation-orchestrator.md)

This document turns the strategy into a build plan for **`gqteashmd`**, our own surface-hopping MD program. The repository is `github.com/AdemirCamargo57/gqteaSHMD`; the Python package and command are `gqteashmd` (D18).

It is the *umbrella* plan. It fixes the architecture, the stage contracts and the cross-cutting rules (inputs, errors, testing, development setup), and it splits the work into sub-projects **P1–P9**. Each sub-project gets its own detailed, task-by-task plan when work on it starts. Only P1 is detailed now, because the later plans depend on answers to the open items in §11.

> **Ground rule: clean-room implementation.** SHARC 4.1 serves only as (a) a description of the overall workflow and (b) a black-box reference whose *outputs* we compare against. No SHARC module, function or code fragment is copied, translated or adapted. Methods are implemented from the published literature. SHARC programs are only ever *executed*, as separate processes, in comparison tests that are not part of the shipped package.

---

## 1. What we are automating

### 1.1 The SHARC 4.1 workflow (reference) and its `gqteashmd` equivalent

In SHARC, each step is a separate tool the user runs by hand, often interactively. `gqteashmd` runs them all from one input file.

| # | SHARC 4.1 (manual) | What the SHARC user does | `gqteashmd` stage | Automated how |
|---|---|---|---|---|
| 1 | Optimization in an external QM program | Runs an optimization, checks convergence | `optimize` (optional, on by default) | PySCF + geomeTRIC, 1 task |
| 2 | Frequency job + `*_freq.py` → Molden file | Runs frequencies, converts output | `frequencies` | PySCF analytic Hessian (1 task) or finite differences of gradients (1 + 6N parallel tasks) |
| 3 | `wigner.py` | Samples geometries and velocities | `sampling` | Own harmonic Wigner sampler, 1 task |
| 4 | `setup_init.py` + running `ICOND_*` jobs | Creates directories, submits single points | `excitations` | One parallel task per sample, plus the reference geometry |
| 5 | `excite.py` (interactive) | Picks initial states | `selection` | Non-interactive, rules from the input file, 1 task |
| 6 | `setup_traj.py` + submitting `TRAJ_*` | Creates trajectory directories, submits | `dynamics` | One parallel task per trajectory; resumable from per-trajectory restart files |
| 7 | `data_collector.py`, `populations.py`, … | Collects and analyzes | `analysis` (optional, on by default) | Populations, summary report, 1 task |

A `prepare` stage runs first. It validates the molecule and creates the project. The user's four stages map as follows: *frequency calculations* → `optimize` + `frequencies`; *Wigner sampling* → `sampling`; *initial-condition setup* → `excitations` + `selection`; *trajectory calculations* → `dynamics`.

### 1.2 Pipeline at a glance

```
input.toml
   │
   ▼
prepare ─► optimize? ─► frequencies ─► sampling ─► excitations ─► selection ─► dynamics ─► analysis?
1 task      1 task      1 or 1+6N      1 task      1 + N_samples    1 task       N_traj      1 task
                        tasks                      tasks (parallel)              tasks (parallel)
```

- The order is fixed and stages run **one after another**. Inside a stage, tasks run **in parallel** up to the configured limit.
- A stage starts only when everything it depends on is DONE (or SKIPPED, for optional stages).
- The user writes one input file and runs one command:

```bash
gqteashmd run input.toml          # runs every stage that is not yet done, in order
gqteashmd status input.toml       # progress: per-stage status and task counts
gqteashmd stop input.toml         # graceful stop; `gqteashmd run` again resumes
```

### 1.3 What still needs a human

- Writing the input file (`gqteashmd init` produces a commented template).
- Fixing problems the program cannot fix itself: for example, imaginary frequencies, or too few samples with the requested excitation. The program stops with a message saying exactly what to change. `gqteashmd run` then resumes.
- Optional review pauses (`workflow.pause_after = ["selection"]`). For example, look at the absorption spectrum and the selected initial states before starting the expensive dynamics.

---

## 2. Architecture

### 2.1 Layers

```
┌──────────────────────────────────────────────────────────────────────┐
│ CLI (gqteashmd init | validate | run | status | stop)                     │
├──────────────────────────────────────────────────────────────────────┤
│ Orchestrator: stage order, fingerprints, resume, retries, policies   │
│   ├─ StateStore (SQLite)   ├─ ProjectLayout/Lock   ├─ LocalExecutor  │
├──────────────────────────────────────────────────────────────────────┤
│ Stages (plan → run_task → finalize):                                 │
│   prepare, optimize, frequencies, sampling, excitations, selection,  │
│   dynamics, analysis                                                 │
├──────────────────────────────────────────────────────────────────────┤
│ Domain libraries (no knowledge of the workflow):                     │
│   chem/ (elements, xyz, masses)  vib/ (harmonic analysis, Wigner)    │
│   electronic/ (backend contract + PySCF backend)                     │
│   dynamics/ (integrators, propagators, hopping, decoherence, …)      │
│   artifacts/ (versioned JSON/HDF5 readers & writers)                 │
└──────────────────────────────────────────────────────────────────────┘
```

**Dependency rule:** arrows point down only. Domain libraries never import stages or the orchestrator, so the dynamics engine can be tested on model systems without the workflow (strategy §6, phases 1–2). Stages are thin adapters: they turn config and upstream artifacts into domain calls, and domain results into artifacts.

### 2.2 Package layout (target, filled in by P1–P9)

```
src/gqteashmd/
├── __init__.py, __main__.py, cli.py, logging_setup.py
├── errors.py                 # error taxonomy (§7)
├── orchestrator.py           # P1
├── worker.py                 # P1: `python -m gqteashmd.worker task.json`
├── config/                   # P1 core, P2 complete schema
│   ├── schema.py  loader.py
├── core/                     # P3a: units.py, states.py (state labels "S1", "T2")
├── project/                  # P1
│   ├── layout.py  lock.py  state.py  provenance.py (P2)
├── execution/                # P1
│   ├── local.py  processes.py
├── stages/
│   ├── base.py  pipeline.py  registry.py           # P1
│   ├── prepare.py                                  # P1
│   ├── optimize.py  frequencies.py                 # P4
│   ├── sampling.py                                 # P5
│   ├── excitations.py  selection.py                # P6
│   ├── dynamics.py                                 # P7b
│   └── analysis.py                                 # P8
├── chem/        elements.py  xyz.py  electrons.py (P3a)  masses.py (P4)
├── vib/         harmonic.py  molden.py  artifacts.py (P4)  wigner.py (P5)
├── geomopt.py                # P4: geomeTRIC driver around any energy+gradient function
├── electronic/  base.py  properties.py  pyscf_backend.py  factory.py (P3a)
│                pyscf_errors.py  pyscf_casscf.py  pyscf_response.py (P3b: failure wrapping;
│                CASSCF ladder, NACs, expansion; TDA/TDDFT ladder, expansion)
├── overlaps/    determinants.py  tracking.py (P3b; backend-independent, never imports PySCF:
│                DeterminantExpansion, truncate, state_overlaps; track, lowdin_orthonormalize)
├── dynamics/    (P7a/P7b; see strategy §2.2 for the inner layout)
├── artifacts/   (one module per artifact type, from P4 on)
├── testing/     fake_stages.py (P1; test-only helpers shipped for subprocess tests)
└── util/        fileio.py
tests/            mirrors src/gqteashmd; plus tests/e2e/ and tests/sharc_compare/
tools/sharc_compare/  scripts that RUN SHARC as a black box (never imported by gqteashmd)
```

---

## 3. Workflow orchestration

### 3.1 Project directory

All results for one input file live in `project.directory`. That should be a local Linux disk, **not** OneDrive and not `/mnt/c`.

```
<project>/
├── .gqteashmd/
│   ├── state.db            # SQLite: stage + task status (controller is the only writer)
│   ├── config.json         # resolved config snapshot used by workers
│   ├── controller.lock     # flock + PID of the running controller
│   ├── provenance.json     # (P2) gqteashmd/pyscf/numpy versions, git commit, host
│   └── superseded/         # old stage results moved here by --rerun-from (never deleted)
├── logs/gqteashmd.log           # controller log
├── 00_prepare/   geometry.xyz  molecule.json  tasks/molecule/{task.json,result.json,worker.log}
├── 01_optimize/  …
├── 02_frequencies/ …
├── 03_sampling/ …
├── 04_excitations/ tasks/ref, tasks/s00001 … 
├── 05_selection/ …
├── 06_dynamics/  tasks/traj_00001 … (each with output.h5, restart.h5)
└── 07_analysis/ …
```

### 3.2 Stage contract (implemented in P1)

Every stage implements three methods:

| Method | Runs in | Purpose |
|---|---|---|
| `plan(ctx) -> list[TaskSpec]` | controller | List the tasks, *deterministically* (same inputs → same task IDs) |
| `run_task(ctx, task, task_dir) -> dict` | worker subprocess | Do one unit of work. Large outputs go to files in `task_dir`; a small JSON-serializable summary is returned. **Must be idempotent and restartable.** |
| `finalize(ctx, results)` | controller | Combine task results into the stage's artifacts, run physical sanity checks, and raise `StageFailedError` if they fail |

Each stage also declares:
- its name and `version`;
- `depends_on`;
- the `config_sections` that affect it;
- `max_failed_fraction`, the share of tasks allowed to fail;
- `enabled(config)`, used for optional stages.

### 3.3 State machines

```
Stage:  (none) ─► RUNNING ─► DONE
                    │  ▲
                    ▼  │ (re-run)
                  FAILED / PENDING(stopped)          SKIPPED (optional stage disabled)

Task:   PENDING ─► RUNNING ─► DONE
           ▲          │
           └──retry───┤ (crash or retryable error, attempts < max_attempts)
                      └─► FAILED (non-retryable, or attempts exhausted)
```

### 3.4 Resume, change detection, re-runs

- **Fingerprints.** Each stage gets a SHA-256 fingerprint of its version, its config sections, its input-file contents (e.g. the geometry), and the fingerprints of the stages it depends on. The fingerprint is stored when the stage starts.
- **Resume.** On `gqteashmd run`:
  - stages that are DONE with a matching fingerprint are skipped ("up to date");
  - a RUNNING or PENDING stage continues where it stopped;
  - tasks that were RUNNING when the controller died are reconciled. A finished `result.json` makes the task DONE. Otherwise any orphaned worker is terminated and the task goes back to PENDING.
- **Changed inputs are never silently mixed with old results.** If a DONE stage's fingerprint no longer matches, the run stops with `StaleStageError` and tells the user to run `gqteashmd run input.toml --rerun-from <stage>`. That command moves the old results of that stage and all later ones to `.gqteashmd/superseded/`; nothing is deleted. A stage with no finished tasks — FAILED, or stopped before any task finished — whose inputs changed is reset automatically; a stage with finished tasks is protected like a DONE stage.
- **Retrying failed work.** Running `gqteashmd run` again after a FAILED stage gives its failed tasks a fresh set of attempts.

### 3.5 Execution on the workstation

- **One controller per project**, enforced with `flock`. It is the only writer of `state.db`.
- **One subprocess per task** (`python -m gqteashmd.worker task.json`), each in its own process session. A crash, segfault or out-of-memory kill in PySCF therefore cannot take the controller down. Each task's stdout/stderr goes to `worker.log`.
- **Concurrency.** `resources.max_parallel_tasks` tasks run at a time, each with `threads_per_task` threads. The controller sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `PYSCF_MAX_MEMORY` for each worker. **Per-stage overrides** (P2), e.g. `[resources.stages.frequencies] threads_per_task = 32`, let single-task stages use the whole machine.
- **Stopping.** SIGINT/SIGTERM or `gqteashmd stop` triggers a graceful stop:
  1. no new tasks are started;
  2. running workers get SIGTERM, then SIGKILL after `stop_grace_period_s`. The grace period applies to the worker (the group leader); once the leader has exited, the rest of its process group is killed at once with SIGKILL;
  3. interrupted tasks go back to PENDING, and the interruption does not count as an attempt.

  Dynamics tasks write a restart file every step, so at most one step is lost.
- **Running unattended.** P1 documents `nohup gqteashmd run input.toml > gqteashmd.out 2>&1 &` and `tmux`. A built-in `--detach` or systemd user service was considered (O12) and not built: `nohup`/`tmux` stay the supported way (decided in P2).
- **Executor abstraction.** The `LocalExecutor` interface (`run(jobs, on_start, on_finish, should_stop)`) is small enough that a SLURM executor could be added later without changing stages. That is **not** planned now (YAGNI).

### 3.6 Reproducibility

- **Seeds.** One master seed, `project.seed`. Each stage derives its own seed stream with `numpy.random.SeedSequence([project.seed, crc32(stage_name)])`, and each task spawns a child stream by its index. Re-running with the same input reproduces every random choice: Wigner samples, state selection and hopping random numbers.
- **Provenance (P2).** `.gqteashmd/provenance.json` records the versions of gqteashmd (with git commit), Python, PySCF, NumPy, SciPy and h5py, plus the host and CPU. Every artifact carries `schema_version` and the gqteashmd version.

---

## 4. Input handling

### 4.1 Rules (decided)

- **One TOML file** per project (stdlib `tomllib`), validated with **pydantic v2**. Validation is strict: unknown keys are errors, which catches typos.
- **Units are in the key names** (`dt_fs`, `tmax_fs`, `temperature_K`, `energy_window_eV`, `fd_step_bohr`). Everything is converted to atomic units at load time.
- **Relative paths** resolve against the input file's directory. The resolved config is snapshotted to `.gqteashmd/config.json`, so editing the input file mid-run cannot affect running workers.
- **States are labelled** `"S0"`, `"S1"`, `"T1"` in the input, which avoids SHARC's 1-based numbering. Internally all indices are 0-based.
- `schema_version = 1` sits at the top. Later breaking schema changes get a migration message instead of a cryptic error.

### 4.2 Validation layers

| Layer | When | Examples | Implemented in |
|---|---|---|---|
| Syntax and types | `load_config` | Unknown key, wrong type, value out of range | P1 (core sections), P2 (all sections) |
| Cross-field rules | `load_config` (pydantic model validators) | Selected states exist in `electronic.states`; `n_trajectories ≥ 1`; `dt_fs < tmax_fs`; CASSCF needs `ncas`/`nelecas` | P2 (resource fit rules); rules for the later sections arrive with their sections (P3 onward) |
| Environment preflight | Start of `gqteashmd run`, before any stage | PySCF importable with a supported version; geomeTRIC present if `optimize` is enabled; **user-given hardware** (`resources.total_cores`, `resources.total_memory_mb`) bounds the task settings (`max_parallel × threads ≤ total_cores`, `max_parallel × memory ≤ total_memory_mb`; violation = `ConfigError`); the user-given values are compared with the detected cores/RAM and a **warning** (not an error) is printed if they exceed the machine (D17); free disk space; scratch directory writable; project directory not on `/mnt/c` | P2 (hardware fit and warning, free disk, project location, writable project location, required modules); P3a (PySCF version, checked when the backend is built); P4: a stage's `required_modules` accept `name>=X.Y` and are checked against the installed version (`optimize`: pyscf>=2.6, geometric>=1.0; `frequencies`: pyscf>=2.6, h5py>=3.9). **Ruling:** PySCF scratch lives in each task directory inside the project (`task_dir/scratch`), whose writability the project-directory check already covers, so there is no separate scratch check |
| Physics preflight | `prepare` stage | xyz syntax, known elements, charge/multiplicity consistent with the electron count (`prepare`; `single-point` does the same check when it builds the backend, `validate` does not) | P1 |
| Smoke calculation | First task of `optimize`/`frequencies` (P3/P4) | SCF converges at the start geometry with the requested method/basis | P3 |

### 4.3 Proposed complete input file

The sections and keys are decided. Values marked `# OPEN` depend on §11.

```toml
schema_version = 1

[project]
name = "thiophene_s1"
directory = "/data/ajc/gqteashmd-runs/thiophene_s1"   # local Linux disk
seed = 20261106

[molecule]
geometry = "thiophene.xyz"          # Angstrom
charge = 0
multiplicity = 1

[electronic]                         # P3a (density_fitting is not implemented yet)
method = "tda"                       # "tda" | "tddft" | "casscf"   (mc-pdft later)
basis = "def2-svp"
functional = "b3lyp"                 # rejected for casscf (ConfigError)
density_fitting = true
grid_level = 3
singlets = 4                         # number of singlet states INCLUDING S0 (O19, resolved)
triplets = 0                         # 0 until ISC is in scope (O7)
# [electronic.casscf]  ncas = 6  nelecas = 6

[optimize]                           # P4, implemented
enabled = true
max_steps = 100
convergence = "GAU"                  # geomeTRIC set: GAU (default), GAU_LOOSE, GAU_TIGHT, GAU_VERYTIGHT, NWCHEM_LOOSE, TURBOMOLE, INTERFRAG_TIGHT

[frequencies]                        # P4, implemented
method = "auto"                      # "auto" | "analytic" | "finite_difference"
fd_step_bohr = 0.005
imaginary_cutoff_cm1 = 20.0          # imaginary modes beyond this (nu < -cutoff) stop the workflow; smaller ones warn

[sampling]                           # P5, implemented
n_samples = 200                      # >= 1
temperature_K = 0.0                  # >= 0; 0 = zero-point distribution; T > 0: exact harmonic thermal Wigner function (O4)
low_frequency_cutoff_cm1 = 10.0      # > 0; modes below it (also small imaginary ones) are not sampled

[excitations]                        # P6, implemented
include_reference = true             # also compute the equilibrium geometry
samples_per_task = 1                 # >= 1
max_failed_fraction = 0.05           # 0 <= x < 1; failed samples above this fraction stop the stage
broadening_fwhm_eV = 0.1             # > 0; Gaussian FWHM of the nuclear-ensemble spectrum
spectrum_step_eV = 0.01              # > 0; energy grid step of spectrum.csv

[selection]                          # P6, implemented
mode = "bright"                      # "bright" | "fixed" (O3 resolved: see §11)
weighting = "f_over_dE2"             # "f_over_dE2" (default, SHARC excite.py-compatible) | "f"; bright mode only
states = ["S1", "S2", "S3"]          # excited singlets; default: all; "fixed" needs exactly one
energy_window_eV = [4.5, 6.0]        # inclusive; default: none
n_trajectories = 100                 # >= 1
allow_fewer = false                  # O20 resolved: false = stop with advice if fewer can be selected; true = warn

[dynamics]                           # P7 — option names follow strategy §4
tmax_fs = 500.0
dt_fs = 0.5
nsubsteps = 25
coupling = "overlap"                 # "overlap" | "nacdr"
hopping = "sharc"                    # "fssh" | "sharc"
ekin_correction = "parallel_vel"
decoherence = "edc"
decoherence_param_Eh = 0.1
energy_drift_limit_eV = 0.5          # OPEN (O8): termination criteria
max_failed_fraction = 0.1

[analysis]                           # P8
enabled = true

[resources]                          # P1 (global), P2 (per-stage, hardware)
total_cores = 32                     # REQUIRED (P2): the user describes the workstation (D17)
total_memory_mb = 128000             # REQUIRED (P2)
max_parallel_tasks = 8
threads_per_task = 4
memory_mb_per_task = 4000
scratch = "/scratch/ajc"             # planned (not in the schema yet; the P3b backend already takes a scratch_dir)
[resources.stages.frequencies]
max_parallel_tasks = 1
threads_per_task = 32

[execution]                          # P1
max_attempts = 2
poll_interval_s = 0.5
stop_grace_period_s = 30.0

[workflow]                           # P2
pause_after = ["selection"]          # review spectrum before dynamics
```

---

## 5. Stage interfaces

**Artifact conventions (decided):**
- Small metadata goes in JSON. Arrays go in HDF5 (`h5py`), with attributes `schema_version`, `gqteashmd_version` and `units` on each dataset. Files meant for people are XYZ (Å), Molden (frequencies) and CSV/PNG.
- Internal files store atomic units. Each artifact type has exactly one reader/writer module in `gqteashmd/artifacts/`, and stages never open another stage's files directly.

| Stage | Depends on | Config sections | Tasks | Output artifacts | Checks → failure policy |
|---|---|---|---|---|---|
| `prepare` | — | `molecule` (+ geometry file hash) | 1 | `geometry.xyz`, `molecule.json` (symbols, Z, coordinates, charge, multiplicity, electron count) | Syntax, elements, charge/multiplicity parity → **stop** (non-retryable) |
| `optimize` (optional) | prepare | `electronic`, `optimize` | 1 | `optimized.xyz`, `optimize.json` (energy, converged, steps), `optimization_path.xyz` | Not converged in `max_steps` → **stop** (non-retryable task failure; `last.xyz` is written), suggesting more steps, a better start or a looser `convergence` |
| `frequencies` | prepare, optimize | `electronic`, `frequencies` | 1 (analytic Hessian) or 1 + 6N (± displacement per Cartesian coordinate) | `frequencies.h5` (masses, coordinates, Hessian, frequencies in cm⁻¹, normal modes, reduced masses), `frequencies.molden`, `frequencies.json` | Imaginary mode with ν < −cutoff → **stop** (`StageFailedError`, exit 4) with the mode list, after the artifacts are written; −cutoff ≤ ν < 0 → **warning** (`small_imaginary_modes`). **Ruling (P4):** the near-zero-mode stop became a **warning** when the translation/rotation residual exceeds 100 cm⁻¹ (measured 22–41 cm⁻¹ at optimized geometries); a **warning** also when max \|gradient\| > 1e-3 Eh/bohr (not a stationary point) |
| `sampling` | frequencies | `sampling` (+ `project.seed`) | 1 | `samples.h5` (positions, velocities, per-mode q/p, harmonic energy), `samples.xyz`, `samples.json` | Per-mode energy check, exact for every N: S_k = Σ_samples E_k/⟨E_k⟩ ~ Gamma(N, 1) with ⟨E_k⟩ = ½ω_k·coth(ω_k/2k_BT) (= ½ω_k at T = 0); **stop** if the smallest two-sided tail probability is below 1e-9 / n_sampled (false-alarm probability ≤ 1e-9); zero total linear and angular momentum (|P| < 1e-8, |L| < 1e-6 a.u.) → **stop** if violated; no sampled mode at all → **stop** |
| `excitations` | sampling (+ geometry from optimize/prepare) | `electronic`, `excitations` | 1 + ⌈N_samples / samples_per_task⌉ | `excitations.h5` (energies, excitation energies in eV, transition dipoles, oscillator strengths, per-sample `ok` flag, `reference` group; every dataset with a `units` attribute), `excitations.json` (per-state means and standard deviations, reference values, failed samples), `spectrum.csv` (CSV only; no PNG, owner decision) | One backend per task with the full `[electronic]` settings and `reset()` before every sample; a sample whose electronic structure fails is recorded and the task continues; failed samples > `max_failed_fraction` → **stop**; a failed reference → **stop**; `[electronic] singlets >= 2` is checked by `validate` (stage `check_config` hook). **Implemented in P6** |
| `selection` | excitations (and so, transitively, sampling: `depends_on = ("excitations",)`) | `selection` without `allow_fewer` (finalize-only) and without `weighting` in fixed mode (+ seed) | 1 | `trajectories.json` (trajectory ID, sample index, initial state label, excitation energy, oscillator strength, `rng: {stage: "dynamics", task_index: k}` per D12), `selection_report.json/.md` | Fewer than `n_trajectories` selectable → **stop** with advice, unless `allow_fewer = true` (warning); zero selectable is always an error (O20); state labels must exist (`validate`). **Implemented in P6** |
| `dynamics` | selection, sampling | `electronic`, `dynamics` | N_traj | Per trajectory: `output.h5`, `restart.h5`, result `{outcome, t_final_fs, n_hops, max_drift_eV}`; stage: `dynamics_summary.json` | Infrastructure failures ≤ `max_failed_fraction`. Physics endings (energy drift, persistent electronic-structure failure) are **outcomes**, not task failures |
| `analysis` (optional) | dynamics | `analysis` | 1 | `populations.h5/.csv`, `report.md`, plots | No completed trajectories → **stop** |

`frequencies` depends on `optimize`. If optimize is SKIPPED, it uses the geometry from `prepare`, and its fingerprint records that.

---

## 6. Error handling

### 6.1 Taxonomy (`gqteashmd/errors.py`, P1)

| Class / situation | Example | Detected by | Result | Exit code |
|---|---|---|---|---|
| `ConfigError` | Typo, invalid value, missing geometry file, unknown `--rerun-from` stage | Loader, preflight | Nothing runs; message names the key and the file | 2 |
| `ProjectError` | Lock held by another controller, corrupt state, plan changed | Controller | Nothing runs | 3 |
| `StaleStageError` (a `ProjectError`) | Input changed after a stage finished | Controller, before the stage | Stops; message gives the exact `--rerun-from` command | 3 |
| `TaskError(retryable=True)` | Transient I/O error | Worker | Task retried until `max_attempts` | — |
| `TaskError(retryable=False)` | Inconsistent multiplicity; SCF still failing after the retry ladder | Worker | Task FAILED; stage checks its tolerance | — |
| Worker crash (no `result.json`) | Segfault, out-of-memory kill, SIGKILL | Controller | Treated as retryable | — |
| `StageFailedError` | Too many failed tasks, imaginary frequencies, too few bright samples | Controller (`finalize`) | Stage FAILED; workflow stops | 4 |
| Trajectory outcome | Energy drift above limit, electronic-structure failure mid-trajectory | Dynamics worker | Task DONE with `outcome` recorded; analysis decides whether to include it | — |
| Unexpected exception (bug) | `KeyError` in our code | Worker | Non-retryable; full traceback in `result.json` and `worker.log` | — |
| Stopped by user | SIGINT/SIGTERM, `gqteashmd stop` (also during start-up, once the lock is held) | CLI | State saved; resumable | 5 |
| Interrupted (Ctrl-C outside the run loop) | KeyboardInterrupt while starting | CLI | Lock released; resumable | 5 |
| Unreadable `task.json` | `OSError` reading it / malformed or missing keys | Worker | `OSError`: retryable (the controller rewrites the file before each attempt); others: not retryable | — |
| File locking unavailable | `flock` fails (e.g. ENOLCK) on the project filesystem | Controller, `status`, `stop` | `ProjectError` ("File locking is not available for …") | 3 |
| State database failure at query time | `sqlite3.Error` in a query | Controller | `ProjectError` (check disk space and other programs holding the database) | 3 |

### 6.2 Rules

- **Every message says what happened, where (file or log path) and what to do next.** Example: `Stage 'sampling' failed: 1 of 3 tasks did not succeed … - t000: TaskError: … (log: …/worker.log). Fix the cause and run the same command again; failed tasks will be retried.`
- **Electronic-structure retry ladder (P3), inside the worker before giving up.** SCF:
  1. guess from the previous step or the reference;
  2. level shift + damping;
  3. fresh atomic guess + second-order solver.

  CASSCF and TDA/TDDFT have their own ladders (**implemented in P3b**), and the converged rung is recorded in the result meta (`meta.scf`, `meta.casscf`, `meta.td`):
  - CASSCF: (1) the previous geometry's orbitals, projected (skipped without a snapshot); (2) SCF orbitals; (3) SCF orbitals plus a spin penalty (`fix_spin_`, shift 0.5) and 3× macro iterations. A rung fails on non-convergence or a root with |<S²>| > 1e-3; a PySCF exception aborts the ladder.
  - TDA/TDDFT: (1) defaults; (2) `nstates + 3` roots and 3× `max_cycle`, lowest kept. Roots are capped at the number of single excitations.
- **Fail early.** All config and environment checks run before the first expensive calculation.
- **Never delete results automatically.** Superseded results are moved; cleaning up is an explicit user action.
- **Logs.** The controller writes `logs/gqteashmd.log` (INFO to console, DEBUG to file). Each task has `worker.log` (stdout/stderr) plus a backend log (e.g. `pyscf.log`) in its task directory.

---

## 7. Testing

### 7.1 Test layers and markers

| Layer | Content | Marker | Where it runs |
|---|---|---|---|
| Unit | Pure functions: parsers, harmonic analysis, Wigner formulas, selection rules, propagators, state store | (none) | Every push (CI) |
| Orchestrator | Fake stages in subprocesses: resume, retry, crash, stop, stale detection, lock | (none), kill/resume = `slow` | CI |
| PySCF component | H₂O HF/STO-3G, CH₂O TDA/def2-SVP, C₂H₄ SA-CASSCF(2,2)/6-31G: energies vs reference values; gradients vs finite differences; analytic vs FD Hessian | `pyscf` | CI (small), nightly (larger) |
| Stage | Each stage on a tiny system, checking artifact schema and physics checks | `pyscf` | CI |
| End-to-end | Full workflow on a tiny system (≈ 5 samples, 2 trajectories, 5 fs), including a kill and resume | `e2e`, `slow` | CI nightly + before each release |
| Determinism | Same input + seed → byte-identical artifacts (excluding timestamps) | `slow` | CI |
| SHARC comparison | Runs SHARC 4.1 tools as black boxes (see 7.2) | `sharc` (needs `SHARC_HOME`) | Manually / nightly on the workstation only |

### 7.2 Stage-by-stage comparison with SHARC (black box)

| Stage | Comparison |
|---|---|
| `frequencies` | Frequencies and modes vs an independent program (e.g. Gaussian or ORCA at the same level). Our Molden file must be readable by SHARC's `wigner.py`, which checks the format. |
| `sampling` | Run SHARC `wigner.py` on *our* Molden file. Compare distributions at T = 0 and T = 1000 K: means and standard deviations of internal coordinates (both within 4 standard errors), kinetic energy before and after momentum removal, two-sample Kolmogorov–Smirnov tests (p > 1e-3). Not per-sample (different random number generators). At 3000 K SHARC's `-t` samples are measurably narrower than the exact thermal Wigner function (normal-coordinate variances 1–8 % low, kurtosis 3.04–3.12); that case is a strict `xfail` documenting the difference (USER_MANUAL §8.8). |
| `excitations` | Same geometries through `SHARC_PYSCF.py` (via `QM.in`/`QM.out`) vs our backend (water CAS(4,4)/STO-3G, reference + 2 Wigner geometries): ΔE ≤ 8.7e-8 Eh, abs(μ) ≤ 1.2e-7, f ≤ 7.4e-9 (**done in P6**); our SHARC-format writers are read back by `excite.py` to 1.4e-7 eV (ΔE) and 1.4e-8 (f). |
| `selection` | Given identical excitation data, compare selected-state statistics with `excite.py` output (`tools/sharc_compare/sharc_formats.py`, `run_excite.py`; **done in P6**, black box). 3000 synthetic samples, window [4, 8] eV: SHARC selected [370, 699] (S1, S2); the default rule f/ΔE² (window maximum) expects [349.8, 654.6] (z = +1.19, +2.36), the all-pairs maximum is rejected (z = +22.4, +32.8), so the window maximum normalises. The weighting exponents are measured directly (final fix wave) by maximum likelihood on SHARC's per-pair selections, w = f^a ΔE^k: k = −1.989 ± 0.050 (constant f, ΔE ~ U(2, 10) eV), a = 0.993 ± 0.028 (constant ΔE, f ~ U(0.01, 0.4)), so SHARC uses (f/ΔE²)/max; ours gives [353, 666]; KS p = 0.96 for the selected S1 energies. |
| `dynamics` | As in strategy §5.4–5.5 (models, LVC, swap tests, ensembles). |

### 7.3 Continuous integration

GitHub Actions runs on `ubuntu-latest` inside an **`oraclelinux:9` container**, so it matches the target OS. Lint (ruff) and tests run on every push and PR. `slow`/`e2e` run nightly. `sharc` tests never run in CI. CI first ran on 2026-11-07 (`main` at `e73609f`) and passed.

---

## 8. Development setup

| Topic | Decision |
|---|---|
| Source location | `C:\Users\ajc\OneDrive\gqteaSHMD` = `/mnt/c/Users/ajc/OneDrive/gqteaSHMD` in WSL. Edit with VS Code (Remote-WSL recommended). |
| Where code runs | **Only in WSL2 (Oracle Linux 9.5) or on the Oracle Linux workstation.** PySCF has no native Windows build, and the runtime uses Linux-only features (`fcntl`, `/proc`, process groups). |
| Environment | `micromamba`/conda with `environment.yml` (conda-forge, Python 3.12), created in the WSL home directory, **never inside OneDrive**. `pip install -e . --no-deps`. A lock file (`conda-lock`) is added once the dependency set settles (P3). |
| Git | **One Git client: Git inside WSL** (VS Code Remote-WSL uses it). `core.autocrlf=false`, `core.filemode=false` (DrvFS reports every file as executable), `.gitattributes` with `* text=auto eol=lf`. |
| OneDrive caveats | OneDrive syncs `.git/` too. Push to GitHub often (GitHub is the source of truth), mark the folder "Always keep on this device", and never put environments, project runs or large test outputs in the folder. pytest writes temporary files to `/tmp` (Linux disk). |
| GitHub | Repository `github.com/AdemirCamargo57/gqteaSHMD` (currently **public**). Work happens on feature branches (`feat/...`) that are pushed; the owner reviews and merges PRs on github.com (the `gh` CLI is not installed). `main` is protected: PRs only, CI must pass, one reviewer. Conventional-commit messages. SemVer `0.x` until the first production study. MIT `LICENSE` from the first commit. |
| Style and tools | `src/` layout, `pyproject.toml` (hatchling), ruff (lint + format, line length 100), pytest, pre-commit hooks. Type hints everywhere; mypy added in P2. |
| Deployment on the workstation | `git clone` → `micromamba create -f environment.yml` → `pip install .`. Users run `gqteashmd` from that environment. Project directories go on a local data disk. |

---

## 9. Sub-project roadmap

Each sub-project produces working, tested software and gets its own detailed plan in `docs/superpowers/plans/` when it starts. Effort assumes one full-time developer. P7a can run in parallel with P3–P6 if a second person is available.

| ID | Sub-project | Delivers | Depends on | Strategy phase | Est. |
|---|---|---|---|---|---|
| **P1** | **Foundation & orchestrator** (detailed plan exists) | Repo, environment, CI, errors, config core, project layout, state store, stage contract, worker, local executor, orchestrator (resume/retry/stop/stale), CLI, `prepare` stage | — | 0 | 3–4 wk |
| P2 | Input schema & preflight — **DONE** (merged into `main`) | Delivered: required `total_cores` / `total_memory_mb` with the fit rules (`n_p·n_t ≤ total_cores`, `n_p·m ≤ total_memory_mb`, globally and per stage) and a detection **warning** (D17); `[resources.stages.<name>]` overrides with one executor per stage; `execution.min_free_disk_mb`; preflight before the project directory is touched (Windows drive / OneDrive refusal, free disk, required modules); `.gqteashmd/provenance.json`; `[workflow] pause_after` (exit 5); once-only error output with `FILE_ONLY` logging; state-db failures as `ProjectError` and read-only `status`; non-finite coordinates, `~` expansion, `.`/`..` name checks; lock retry; mypy in CI; the `gqteashmd init` template for the implemented sections. **Scope ruling:** the later input sections (`[electronic]`, `[optimize]`, `[frequencies]`, `[sampling]`, `[excitations]`, `[selection]`, `[dynamics]`, `[analysis]`) and state labels are added by the sub-project that consumes them (P3–P8), because their keys depend on open items O3, O4, O8 and O19. The remaining deferred P1 findings stay in the ledgers with an owner (see the P2 plan, "Scope rulings") | P1 | 0 | 2 wk |
| **P3a** | PySCF electronic-structure layer — **DONE** (merged into local `main` on 2026-11-07) | Delivered: the `[electronic]` / `[electronic.casscf]` input; `core/` (units, state labels); the backend contract and `PySCFBackend` (RHF/RKS, SA-CASSCF with a singlet-only CI solver and an ⟨S²⟩ guard, TDA, TDDFT, CIS, TDHF); energies, analytic gradients and S0→state transition dipoles; the SCF retry ladder; orbital reuse by projection; threads/memory; `gqteashmd single-point`; ethylene validation against SHARC 4.1 (`tools/sharc_compare/`, `pytest -m sharc`) and against finite differences. **Not delivered, moved:** NACs, wavefunction overlaps, phase tracking, scratch/chkfile policy and TDDFT/CASSCF retry ladders to P3b; the Hessian to P4 | P1 (P2 for config) | 3 | 5–6 wk |
| **P3b** | Couplings for dynamics — **DONE** (merged into local `main` on 2026-11-07) | **Delivered:** SA-CASSCF NACs (agree with SHARC 4.1 to 9e-8 up to sign); `gqteashmd.overlaps` (Löwdin determinant overlaps, union truncation `electronic.overlap_norm_kept`, Hungarian state matching, phase tracking); `capabilities`, `compute_more`, `save_state`/`load_state` (restart reproduces to 2e-11); scratch policy (no chkfile, TMPDIR); CASSCF and TD retry ladders. TDDFT overlaps use X only; TDA/TDDFT have no analytic NACs in PySCF 2.14. Library only: no CLI command or stage uses these yet. **Moved to P3c by ruling:** the hardening sweep. Original scope: Nonadiabatic couplings (CASSCF) and wavefunction overlaps with phase tracking for TDA/TDDFT; the rest of the strategy §2.3 backend contract: `capabilities`, `compute_more` (an extra gradient at the same geometry after a hop) and explicit, serialisable reference-wavefunction state (`save_state`/`load_state`, or a state object in/out of `compute`) so that restarts are reproducible and `run_task` stays idempotent; scratch directory and chkfile policy; CASSCF/TDDFT retry ladders. Needed by P7b, not by P4–P6 | P3a | 3 | 3–4 wk |
| P3c | Hardening sweep — **DONE** (merged into local `main` on 2026-11-07) | **Delivered:** lock before `ensure()`/logging via `Orchestrator.session()` (no leaked handle; "File locking is not available"); stale `.tmp` cleanup and failed-write cleanup; `cpu_model` in provenance; sqlite errors at query time become `ProjectError`; SIGKILL to a worker's whole group; unreadable `task.json` reported as a result; picklable `TaskError`; a DONE stage with a deleted directory is recomputed; Ctrl-C exits 5; `pyscf` test marker (fast tier 293 tests, about 7 min on `/mnt/c`). **Moved by ruling:** refused runs still writing state, optional stages, the validator reporting all violations, and the PySCF/geomeTRIC version and writable-scratch preflight checks → P4; the worker orphan window → P7b; the build stamp → release | P3b | 3 | 1–2 wk (estimate) |
| P4 | `optimize` + `frequencies` stages — **DONE** (merged into local `main` on 2026-11-08) | **Delivered:** `[optimize]` and `[frequencies]` input; the `optimize` stage (geomeTRIC custom engine on the `[electronic]` S0 surface, one backend for the whole path, convergence sets GAU...INTERFRAG_TIGHT, `optimized.xyz`, `optimization_path.xyz`, `optimize.json`); the `frequencies` stage (analytic RHF/RKS Hessian, or 1 + 6N central-difference gradient tasks, each with a fresh backend, for SA-CASSCF; S0 settings from `ground_state_settings`); own harmonic analysis (`vib/harmonic.py`: mass weighting, Eckart projection, Gaussian-convention reduced masses); `frequencies.h5`, `frequencies.molden`, `frequencies.json`; `chem/masses.py` (O5); imaginary-mode stop and the residual and stationarity warnings; preflight version checks (`name>=X.Y`); `check_stage_inputs` (a stage without its input section is a `ConfigError`); the orchestrator refuses a stale run before changing anything, keeps the results of a disabled optional stage, and recomputes any stage whose directory was deleted; the validator reports all violations. Validated against PySCF `thermo.harmonic_analysis` (1e-6 cm⁻¹), analytic vs finite-difference Hessians, ORCA 6.1.1 (water RHF/6-31G*, 0.05 cm⁻¹) and SHARC 4.1 `wigner.py` reading our Molden file. Original scope: geomeTRIC driver; **P4 and P6 must call `reset()` or use a fresh backend for independent geometries (a PySCF backend keeps the previous geometry)**; PySCF Hessian (moved here from P3); own harmonic analysis; imaginary-mode policy; Molden writer; `frequencies.h5` | P3 | 5 | 3 wk |
| P5 | `sampling` stage — **DONE** (merged into `main`, 2026-11-08) | **Delivered:** `[sampling]` input (`n_samples`, `temperature_K`, `low_frequency_cutoff_cm1`); own harmonic Wigner sampler `vib/wigner.py` (mass-weighted normal coordinates, exact thermal Wigner widths with coth, T = 0 limit, Cartesian transform, removal of linear and angular momentum without rescaling, removed kinetic energy recorded); `util/seeds.py` (seed hierarchy D12); `stages/sampling.py` (one task, `samples.h5`, `samples.xyz`, `samples.json`, per-mode exact Gamma energy check and momentum stop checks); **O4 resolved**; `tools/sharc_compare/initconds.py` and the live comparison with SHARC 4.1 `wigner.py` (water RHF/STO-3G, 20000 vs 2000 samples, T = 0 and 1000 K: means within 1.6 and standard deviations within 1.4 standard errors, KS p between 0.13 and 0.97; SHARC's Eharm header = ZPE to better than 1e-8 at both temperatures; same masses; at 3000 K SHARC's thermal samples are 1–8 % narrower per mode than the exact thermal Wigner function, documented as a strict xfail). Original scope: Own Wigner sampler (T = 0, thermal option), velocities, momentum removal, `samples.h5`; SHARC `wigner.py` distribution comparison. **Note for P6:** `excitations` runs on the `positions` of `samples.h5`; P7 uses its `velocities` | P4 | 5 | 2–3 wk |
| P6 | `excitations` + `selection` stages — **DONE** (merged into `main`, 2026-11-08) | **Delivered:** `[excitations]` and `[selection]` input; the `excite/` package (`spectrum.py`: Crespo-Otero & Barbatti nuclear-ensemble spectrum, prefactor 1.09761 eV·Å², ε = σ × 26153.8; `selection.py`: candidates, bright selection with weighting `f_over_dE2` (default) or `f`, fixed selection; `artifacts.py`: `excitations.h5`, `spectrum.csv`); the `excitations` stage (reference + one task per `samples_per_task` samples, one backend per task with the full `[electronic]` settings, `reset()` before every sample, per-sample failure records, `max_failed_fraction`) and the `selection` stage (`trajectories.json`, `selection_report.json/.md`); the `Stage.check_config` hook; `read_samples_h5`; the pipeline now ends at `selection`; `tools/sharc_compare/sharc_formats.py` and `run_excite.py` and the black-box comparisons with `SHARC_PYSCF.py` and `excite.py`; **O3 and O20 resolved**. Original scope: Parallel single points, spectrum, selection modes, `trajectories.json`, review report for `pause_after`; SHARC excite comparison | P3, P5 | 5 | 3–4 wk |
| P7a-1 | Dynamics core on analytic models — **DONE on `feat/p7a1-dynamics-core`, not merged yet** (2026-11-08) | **Delivered:** `gqteashmd.dynamics` (NumPy only; imports nothing from `electronic/`, `stages/`, the orchestrator or `config/`): `DynamicsOptions`, records, replayable random sources (one uniform number per step), Tully SAC/DAC/ECR and a 2D test model, `diagonal_basis` with the SOC guard, vectorised exponential propagator (`midpoint`/`endpoint` substep interpolation), velocity Verlet, Tully FSSH and SHARC-type hopping, `parallel_vel`/`parallel_nac`, frustrated keep/reflect, EDC (`sharc`/`granucci_persico`), driver and outcomes, bit-identical restarts, ensemble runner with `workers`, `gqteashmd model-dynamics`; validation against our exact split-operator wavepacket solver and against SHARC 4.1 as a black box (published tabulated values: none open-access, skipped). Details: USER_MANUAL §8.10–§8.12, §8.14. **O6 resolved, O7 designed** | P1 | 1–2 | 5–7 wk |
| P7a-2 | Dynamics core, second part — **next** | **Owner split of P7a (2026-11-08):** LVC backend, overlap-based local diabatization, triplet spin expansion, SOC with the diagonal representation and gradient corrections (the data model — complex MCH Hamiltonian, MCH→diagonal matrix U, guard — is already in P7a-1; U must also be carried in the state and the restart), `eselect`, SHARC LVC comparisons (reuse `tools/sharc_compare/run_sharc_model.py`) | P7a-1 | 1–2 | 5–7 wk |
| P7b | `dynamics` stage with PySCF | Overlaps and phase tracking with PySCF wavefunctions (treat `overlap_min_singular_value < 0.5` and a CASSCF rung change as step-rejection signals; consider rank-2 updates for the TDA single excitations if the overlap cost matters), per-step restart, SIGTERM handling, outcomes, `dynamics_summary.json` | P3b, P6, P7a | 3–4 | 6–8 wk |
| P8 | `analysis` stage + end-to-end workflow | Populations with bootstrap confidence intervals, kinetics fit, `report.md`; nightly end-to-end test | P7b | 5 | 3 wk |
| P9 | SHARC validation campaign | Strategy §5 on LVC + PySCF systems; documented comparison report | P5–P8 | 5 | 4–6 wk (overlaps) |

**Critical path (one developer):** P1 → P2 → P3a → P4 → P5 → P6 → (P7a) → P3b → P7b → P8 ≈ 9–12 months. With a second developer on P7a from month 2, about 7–9 months. ISC (strategy phase 6) and further backends (strategy phase 7) follow afterwards.

**Milestones:**
- **M1** (end of P2): `gqteashmd run` validates the core input file (project, molecule, resources, execution, workflow) and runs `prepare`; CI green on Oracle Linux 9.
- **M2** (end of P5) — **reached 2026-11-08** (merged into `main`): from one input, the program produces optimized geometry → frequencies → Wigner samples, matching SHARC `wigner.py` statistics.
- **M3** (end of P6) — **reached on the feature branch (2026-11-08)**: the full front half of the workflow runs unattended (water HF/STO-3G through `selection` in about one minute), with a review pause after `excitations` (the last stage, `selection`, never pauses until `dynamics` exists).
- **M4** (end of P8): first complete unattended study (input → populations) on a small molecule, kill-and-resume tested.

---

## 10. Decisions made now

| ID | Decision | Rationale |
|---|---|---|
| D1 | Python 3.12, conda-forge environment (micromamba), `src/` layout, hatchling | Matches the PySCF ecosystem; the stdlib has `tomllib`/`StrEnum` |
| D2 | Runtime is Linux only (Oracle Linux 9 target); develop in WSL2 on the OneDrive folder; no Windows runtime | PySCF has no Windows build; we rely on `fcntl`, `/proc` and process groups |
| D3 | MIT licence; clean-room implementation; SHARC only executed as a black box in comparison tools | Licence choice; GPL-3 of SHARC |
| D4 | PySCF is the only electronic-structure backend in the first release; backend contract per strategy §2.3 | Strategy decision of 2026-11-06 |
| D5 | Single workstation, no scheduler; resumable controller; one subprocess per task | User decision; isolates crashes; simple to operate |
| D6 | Fixed stage list and order (§1.2); `optimize` and `analysis` optional, on by default | User decision (optional optimization) |
| D7 | One TOML input, strict pydantic validation, units in key names, `schema_version`, config snapshot | §4.1 |
| D8 | State in SQLite (WAL); controller is the single writer; workers talk via `task.json` / `result.json` files written atomically | Robust to crashes; no concurrent-writer problems |
| D9 | Fingerprint-based change detection; `StaleStageError` + `--rerun-from`; old results moved, never deleted | No silent mixing of results from different inputs |
| D10 | Error taxonomy and exit codes in §6; per-stage failure tolerances; physics endings of trajectories are outcomes, not failures | Predictable behaviour; no selection bias hidden in "failures" |
| D11 | Artifacts: JSON metadata + HDF5 arrays + XYZ/Molden/CSV for people; schema-versioned; one module per artifact type | Self-describing, partially readable while running |
| D12 | Seed hierarchy from `project.seed` via `SeedSequence` | Full reproducibility |
| D13 | State labels `S0/S1/T1` in input; 0-based internally | Avoids SHARC's 1-based ambiguity |
| D14 | CLI: `init`, `validate`, `run [--rerun-from]`, `status`, `stop` | Minimal surface for "inputs once, run to completion" |
| D15 | Git in WSL only (commits); LF line endings; GitHub repo `AdemirCamargo57/gqteaSHMD`; protected `main`; CI in an `oraclelinux:9` container | §8 |
| D16 | Test layers and markers per §7; SHARC comparisons outside CI | §7 |
| D17 | Workstation hardware is **given by the user** in `[resources]` (`total_cores`, `total_memory_mb`, required from P2). gqteashmd compares it with the detected hardware and only **warns** if the user values exceed the machine | User decision 2026-11-06 |
| D18 | Python package and command are named **`gqteashmd`** (repository `gqteaSHMD`); internal class names keep the `Shmd` prefix (`ShmdError`, `ShmdConfig`) | User decision 2026-11-06 |
| D19 | P3 validation target: **ethylene** (`Eteno.xyz`), **SA2-CASSCF(2,2)/6-31G\*** first, then TDA | User decision 2026-11-06 |
| D20 | Each sub-project is built on a feature branch (`feat/<sub-project>`) with one commit per task, pushed to GitHub; the owner merges the PR. Plans are executed subagent-driven | User decision 2026-11-06 |

## 11. Open items (still to be specified)

| ID | Item | Why it matters | Proposed default | Needed by |
|---|---|---|---|---|
| O1 | ~~Workstation hardware~~ **Resolved → D17** (user input + warning check) | | | |
| O2 | ~~First target molecule and method~~ **Resolved → D19** (ethylene, SA2-CASSCF(2,2)/6-31G\*). Still open: number of states and basis for the TDA follow-up | TDA test fixture | TDA-B3LYP/def2-SVP, 3 singlets | P3 |
| O3 | ~~Excitation selection rule~~ **Resolved in P6 (owner decisions 2026-11-08):** bright mode: candidate (sample i, state n) if sample i succeeded, n is in `states` and ΔE is in the window (inclusive, eV); a candidate with weight w is accepted if r < w/w_max, r ~ U[0,1) from the selection stream (one draw per candidate, sample-major then S1, S2, …), w_max the maximum over the candidates; accepted pairs are taken in order up to `n_trajectories` (one sample may start several trajectories). `weighting = "f_over_dE2"` (default, w = f/ΔE², SHARC 4.1 `excite.py`-compatible, measured by black-box maximum likelihood: ΔE exponent −1.99 ± 0.05, f exponent 0.99 ± 0.03; no paper is cited for it; it favours low-energy pairs, so an energy window is recommended and the stage warns when the w_max pair exceeds 10× the 99th percentile of the weights) or `"f"` (w = f, the oscillator-strength convention of Newton-X-style double criteria). `fixed` mode: the first `n_trajectories` eligible samples in the one given state (weighting ignored) | Initial-state distribution | | P6 |
| O4 | ~~Wigner sampling for T > 0 and treatment of very low-frequency modes~~ **Resolved in P5 (owner decision 2026-11-08):** T > 0 uses the exact harmonic thermal Wigner function, σ_Q² = coth(ω/2k_BT)/(2ω), σ_P² = ω·coth(ω/2k_BT)/2 (ħ = 1, a.u.), which reduces to the ground-state distribution at T = 0; modes with ν̃ < `low_frequency_cutoff_cm1` (default 10 cm⁻¹, must be > 0), including the small imaginary modes kept by `frequencies`, are not sampled (equilibrium coordinate, zero velocity) and are listed in a warning | Initial-condition quality for floppy molecules | | P5 |
| O5 | ~~Mass convention (most-abundant isotope vs average atomic weight)~~ **Resolved in P4:** most-abundant-isotope masses in amu (`gqteashmd.chem.masses`, H–Rn); reduced masses use the Gaussian/PySCF convention μ = 1/Σ(l²/m). Sampling (P5) uses the same masses | Frequencies and sampling, and SHARC comparability | | P4 |
| O6 | ~~Dynamics options in the first release~~ **Resolved in P7a-1 (owner decisions 2026-11-08):** hopping `sharc` (default, propagator-based; Mai, Marquetand & González, IJQC 115, 1215 (2015)) / `fssh` (Tully) / `none`; `ekin_correction` `parallel_vel` (default) / `parallel_nac`; `frustrated` `keep` (default) / `reflect` (momentum component along the adjustment direction reversed in the mass-weighted metric); `decoherence` `edc` (default, C = 0.1 Eh) / `none`; **`edc_convention`** `sharc` (default; SHARC 4.1 manual §8.7.1 eq. 8.25, half the Granucci–Persico damping rate) / `granucci_persico` (as reproduced in Runeson, arXiv:2507.18381, Sec. III.3, eqs. 44–45, citing Granucci & Persico, JCP 126, 134114 (2007)); **`substep_interpolation`** `midpoint` (default, second order) / `endpoint` (SHARC manual scheme, first order); velocity Verlet with a fixed step; `nacdr` only (`overlap` coupling and local diabatization in P7a-2). Deferred: A-FSSH, NPI, GFSH, `parallel_diff`, thermostats. EDC with C = 0.1 biases the Tully benchmarks (USER_MANUAL §8.14.1) | Scope of P7a/P7b | | P7a-1 |
| O7 | Triplets/ISC in the first release? **SOC designed, built in P7a-2 (owner decision 2026-11-08):** the P7a-1 data model carries the complex MCH Hamiltonian and the MCH→diagonal matrix U, and a guard rejects a non-diagonal U (`NotImplementedError`); the SOC propagation, diagonal-representation gradients and triplet spin expansion are built in P7a-2. Triplets from PySCF (P3/P7b) remain open | Strategy risk R1 | Singlets only first | P3/P7a-2 |
| O8 | Trajectory termination criteria (energy drift limit, stop after N fs in S0, S1/S0 gap rule for TDDFT) | Outcomes, cost | Drift > 0.5 eV → end; optional stop after 50 fs in S0 | P7b |
| O9 | Data retention: chkfiles, per-step output frequency, disk budget per project | Disk use on the workstation | Keep only last-step chkfile; output every step | P7b |
| O10 | Required analysis outputs (which observables, plots, fits) | Scope of P8 | Populations (classical + quantum), hop statistics, mono-exponential fit | P8 |
| O11 | ~~Notifications on completion or failure (email, chat)?~~ **Resolved (P2): none; log + exit code** | Unattended operation | None; log + exit code | P2 |
| O12 | ~~Unattended running: `nohup`/`tmux` vs `gqteashmd run --detach` vs systemd user service~~ **Resolved (P2): `nohup`/`tmux`, no detach** | Convenience, robustness to logout | Documented `nohup`/`tmux`; no detach (USER_MANUAL §5.2). Revisit only if a need appears | P2 |
| O13 | ~~Program name, GitHub repository~~ **Resolved → D18/D15.** Still open: keep the repository public, or make it private until the first release? | Visibility of unreleased code | Your choice; the MIT licence is added in P1 Task 1 | P1 Task 12 |
| O14 | ~~Copyright holder in the MIT licence~~ **Resolved:** "The gqteashmd developers" (user decision 2026-11-06) | | | |
| O15 | ~~Several molecules or parameter scans per input file?~~ **Resolved (P2): no, one project per input file** | Input design | No: one project per input | P2 |
| O16 | Extending finished dynamics (e.g. increasing `tmax_fs`) | Resume semantics | Allowed: `tmax_fs` excluded from the fingerprint and handled as an "extend" | P7b |
| O17 | SHARC `input`-file importer (strategy §2.4) needed? | Validation convenience | Only inside `tools/sharc_compare/` | P9 |
| O18 | ~~PySCF version pin~~ **Resolved (P3a): `pyscf=2.14.*` pinned in `environment.yml` (2.14.0 at P3a); the runtime requirement is `pyscf>=2.6`, `numpy>=1.26`, and `require_pyscf` rejects older versions** | Reproducibility | Latest conda-forge release at P3 start | P3a |
| O19 | ~~State-count convention (`singlets` includes S0?)~~ **Resolved (P3a, owner decision 2026-11-07): yes, `singlets` includes S0; labels are `S0`, `S1`, … (0-based)** | Input clarity | Yes, count includes S0 | P3a |
| O20 | ~~Behaviour when fewer eligible samples than `n_trajectories`~~ **Resolved in P6:** stop with advice (numbers, `n_samples`, window, states, `allow_fewer`); `allow_fewer = true` accepts fewer with a warning (finalize-only, not fingerprinted, so a plain rerun after the stop applies it; the other remedies name `--rerun-from selection` or `--rerun-from sampling`); zero selected is always an error | Unattended runs | | P6 |

## 12. Workflow-specific risks

| Risk | Mitigation |
|---|---|
| Controller killed (logout, reboot) during days-long dynamics | Resume by design (§3.4); per-step restart files; `tmux`/`nohup` documented |
| Out-of-memory kills when many PySCF tasks run in parallel | Preflight `max_parallel × memory ≤ RAM`; `PYSCF_MAX_MEMORY` per worker; crash → retry |
| Thread oversubscription (BLAS × OpenMP × tasks) | Thread env variables set per worker; preflight `max_parallel × threads ≤ cores` |
| Disk fills during a long run | Preflight free-space check; per-stage check before starting; O9 retention policy |
| Silent mixing of results from changed inputs | Fingerprints + `StaleStageError` (D9) |
| Orphaned workers after a controller crash | Reconciliation on resume: PID + `/proc/<pid>/cmdline` check, then terminate |
| OneDrive/WSL file-system quirks (locks, permissions, slow I/O) | Runs never on `/mnt/c`; Git from WSL only; GitHub as source of truth |
| Scope creep toward SHARC parity | Sub-plans with explicit deliverables; strategy §6 feature freeze per phase |
