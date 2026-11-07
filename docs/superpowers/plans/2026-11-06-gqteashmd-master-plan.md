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
│   ├── schema.py  loader.py  labels.py (state labels "S1", "T2")
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
├── chem/        elements.py  xyz.py  masses.py (P4)  molden.py (P4)
├── vib/         harmonic.py (P4)  wigner.py (P5)
├── electronic/  base.py  pyscf_backend/ (P3)
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
  2. running workers get SIGTERM, then SIGKILL after `stop_grace_period_s`;
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
| Environment preflight | Start of `gqteashmd run`, before any stage | PySCF importable with a supported version; geomeTRIC present if `optimize` is enabled; **user-given hardware** (`resources.total_cores`, `resources.total_memory_mb`) bounds the task settings (`max_parallel × threads ≤ total_cores`, `max_parallel × memory ≤ total_memory_mb`; violation = `ConfigError`); the user-given values are compared with the detected cores/RAM and a **warning** (not an error) is printed if they exceed the machine (D17); free disk space; scratch directory writable; project directory not on `/mnt/c` | P2 (hardware fit and warning, free disk, project location, writable project location, required modules); P3 (PySCF version, geomeTRIC, writable scratch) |
| Physics preflight | `prepare` stage | xyz syntax, known elements, charge/multiplicity consistent with the electron count | P1 |
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
functional = "b3lyp"                 # ignored for casscf
density_fitting = true
grid_level = 3
singlets = 4                         # number of singlet states INCLUDING S0 (O19, resolved)
triplets = 0                         # 0 until ISC is in scope (O7)
# [electronic.casscf]  ncas = 6  nelecas = 6

[optimize]                           # P4
enabled = true
max_steps = 100
convergence = "default"              # geomeTRIC preset

[frequencies]                        # P4
method = "auto"                      # "auto" | "analytic" | "finite_difference"
fd_step_bohr = 0.005
imaginary_cutoff_cm1 = 20.0          # |imag| above this stops the workflow

[sampling]                           # P5
n_samples = 200
temperature_K = 0.0                  # OPEN (O4): scheme for T > 0

[excitations]                        # P6
include_reference = true             # also compute the equilibrium geometry
samples_per_task = 1
max_failed_fraction = 0.05

[selection]                          # P6
mode = "bright"                      # "fixed" | "bright"   # OPEN (O3): formula
states = ["S1", "S2", "S3"]
energy_window_eV = [4.5, 6.0]
n_trajectories = 100

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
scratch = "/scratch/ajc"             # P3 (not in the P2 schema: added with the PySCF layer)
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
| `optimize` (optional) | prepare | `electronic`, `optimize` | 1 | `optimized.xyz`, `optimize.json` (energy, converged, steps), `optimization_path.xyz` | Not converged in `max_steps` → **stop**, suggesting more steps or a better start |
| `frequencies` | prepare, optimize | `electronic`, `frequencies` | 1 (analytic Hessian) or 1 + 6N (± displacement per Cartesian coordinate) | `frequencies.h5` (masses, coordinates, Hessian, frequencies in cm⁻¹, normal modes, reduced masses), `frequencies.molden`, `frequencies.json` | Imaginary mode with \|ν\| > cutoff → **stop** with the mode list; the wrong number of near-zero modes after projection → **stop** |
| `sampling` | frequencies | `sampling` (+ `project.seed`) | 1 | `samples.h5` (positions, velocities, per-mode q/p, harmonic energy), `samples.xyz` | Mean harmonic energy ≈ ZPE at T = 0 (statistical tolerance); zero total linear and angular momentum → **stop** if violated |
| `excitations` | sampling (+ geometry from optimize/prepare) | `electronic`, `excitations` | 1 + ⌈N_samples / samples_per_task⌉ | `excitations.h5` (energies [N, nstates], oscillator strengths, transition dipoles, per-sample status), `spectrum.csv`, `spectrum.png` | Failed samples ≤ `max_failed_fraction`, else **stop**; roots missing or unconverged → that sample fails |
| `selection` | excitations, sampling | `selection` (+ seed) | 1 | `trajectories.json` (trajectory ID, sample index, initial state label, seed), `selection_report.json/.md` | Fewer eligible samples than `n_trajectories` → **stop** (O20); state labels must exist |
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
| Stopped by user | SIGINT/SIGTERM, `gqteashmd stop` | CLI | State saved; resumable | 5 |

### 6.2 Rules

- **Every message says what happened, where (file or log path) and what to do next.** Example: `Stage 'sampling' failed: 1 of 3 tasks did not succeed … - t000: TaskError: … (log: …/worker.log). Fix the cause and run the same command again; failed tasks will be retried.`
- **Electronic-structure retry ladder (P3), inside the worker before giving up.** SCF:
  1. guess from the previous step or the reference;
  2. level shift + damping;
  3. fresh atomic guess + second-order solver.

  TDDFT/CASSCF have their own ladders (more roots, a different initial guess). Each rung is logged.
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
| `sampling` | Run SHARC `wigner.py` on *our* Molden file. Compare distributions: means/variances of internal coordinates, kinetic-energy histogram, two-sample Kolmogorov–Smirnov tests. Not per-sample (different random number generators). |
| `excitations` | Same geometries through `SHARC_PYSCF.py` (via `QM.in`/`QM.out`) vs our backend: energies and oscillator strengths within strategy §5.3 tolerances. |
| `selection` | Given identical excitation data, compare selected-state statistics with `excite.py` output (requires a small `tools/` writer for SHARC's `initconds` format). |
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
| P2 | Input schema & preflight — **DONE** (branch `feat/p2-input-preflight`, not yet merged) | Delivered: required `total_cores` / `total_memory_mb` with the fit rules (`n_p·n_t ≤ total_cores`, `n_p·m ≤ total_memory_mb`, globally and per stage) and a detection **warning** (D17); `[resources.stages.<name>]` overrides with one executor per stage; `execution.min_free_disk_mb`; preflight before the project directory is touched (Windows drive / OneDrive refusal, free disk, required modules); `.gqteashmd/provenance.json`; `[workflow] pause_after` (exit 5); once-only error output with `FILE_ONLY` logging; state-db failures as `ProjectError` and read-only `status`; non-finite coordinates, `~` expansion, `.`/`..` name checks; lock retry; mypy in CI; the `gqteashmd init` template for the implemented sections. **Scope ruling:** the later input sections (`[electronic]`, `[optimize]`, `[frequencies]`, `[sampling]`, `[excitations]`, `[selection]`, `[dynamics]`, `[analysis]`) and state labels are added by the sub-project that consumes them (P3–P8), because their keys depend on open items O3, O4, O8 and O19. The remaining deferred P1 findings stay in the ledgers with an owner (see the P2 plan, "Scope rulings") | P1 | 0 | 2 wk |
| **P3a** | PySCF electronic-structure layer — **DONE** (branch `feat/p3a-pyscf-backend`, not yet merged) | Delivered: the `[electronic]` / `[electronic.casscf]` input; `core/` (units, state labels); the backend contract and `PySCFBackend` (RHF/RKS, SA-CASSCF with a singlet-only CI solver and an ⟨S²⟩ guard, TDA, TDDFT, CIS, TDHF); energies, analytic gradients and S0→state transition dipoles; the SCF retry ladder; orbital reuse by projection; threads/memory; `gqteashmd single-point`; ethylene validation against SHARC 4.1 (`tools/sharc_compare/`, `pytest -m sharc`) and against finite differences. **Not delivered, moved:** NACs, wavefunction overlaps, phase tracking, scratch/chkfile policy and TDDFT/CASSCF retry ladders to P3b; the Hessian to P4 | P1 (P2 for config) | 3 | 5–6 wk |
| P3b | Couplings for dynamics | Nonadiabatic couplings (CASSCF) and wavefunction overlaps with phase tracking for TDA/TDDFT; scratch directory and chkfile policy; CASSCF/TDDFT retry ladders. Needed by P7b, not by P4–P6 | P3a | 3 | 3–4 wk |
| P4 | `optimize` + `frequencies` stages | geomeTRIC driver; **PySCF Hessian (analytic + finite-difference gradient tasks; moved here from P3)**; own harmonic analysis (mass weighting, Eckart projection, frequencies, normal modes); imaginary-mode policy; Molden writer; `frequencies.h5` | P3 | 5 | 3 wk |
| P5 | `sampling` stage | Own Wigner sampler (T = 0, thermal option), velocities, momentum removal, `samples.h5`; SHARC `wigner.py` distribution comparison | P4 | 5 | 2–3 wk |
| P6 | `excitations` + `selection` stages | Parallel single points, spectrum, selection modes, `trajectories.json`, review report for `pause_after`; SHARC excite comparison | P3, P5 | 5 | 3–4 wk |
| P7a | Dynamics core on model systems | Strategy phases 1–2: integrator, propagators, hopping, rescaling, EDC, MCH/diagonal, analytic + LVC backends, standalone runner | P1 | 1–2 | 10–14 wk |
| P7b | `dynamics` stage with PySCF | Overlaps and phase tracking with PySCF wavefunctions, per-step restart, SIGTERM handling, outcomes, `dynamics_summary.json` | P3b, P6, P7a | 3–4 | 6–8 wk |
| P8 | `analysis` stage + end-to-end workflow | Populations with bootstrap confidence intervals, kinetics fit, `report.md`; nightly end-to-end test | P7b | 5 | 3 wk |
| P9 | SHARC validation campaign | Strategy §5 on LVC + PySCF systems; documented comparison report | P5–P8 | 5 | 4–6 wk (overlaps) |

**Critical path (one developer):** P1 → P2 → P3a → P4 → P5 → P6 → (P7a) → P3b → P7b → P8 ≈ 9–12 months. With a second developer on P7a from month 2, about 7–9 months. ISC (strategy phase 6) and further backends (strategy phase 7) follow afterwards.

**Milestones:**
- **M1** (end of P2): `gqteashmd run` validates the core input file (project, molecule, resources, execution, workflow) and runs `prepare`; CI green on Oracle Linux 9.
- **M2** (end of P5): from one input, the program produces optimized geometry → frequencies → Wigner samples, matching SHARC `wigner.py` statistics.
- **M3** (end of P6): the full front half of the workflow runs unattended, with a review pause before dynamics.
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
| O3 | Excitation selection rule (formula, bright-state threshold, energy-window semantics) | Initial-state distribution | Stochastic, weighted by oscillator strength, within window; exact formula from literature, cross-checked with SHARC `excite.py` | P6 |
| O4 | Wigner sampling for T > 0 and treatment of very low-frequency modes | Initial-condition quality for floppy molecules | T = 0 only in first release; thermal scheme decided in P5 | P5 |
| O5 | Mass convention (most-abundant isotope vs average atomic weight) | Frequencies and sampling, and SHARC comparability | Most-abundant isotope masses (verify SHARC's choice in comparison) | P4 |
| O6 | Dynamics options in the first release | Scope of P7a/P7b | Strategy §4: ddr/overlap, LD propagator, SHARC + FSSH hopping, parallel_vel/nac, EDC | P7a |
| O7 | Triplets/ISC in the first release? | Strategy risk R1 | No; singlets only first | P3/P7 |
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
| O20 | Behaviour when fewer eligible samples than `n_trajectories` | Unattended runs | Stop with a message (no silent reduction); option `allow_fewer = true` | P6 |

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
