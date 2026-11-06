# gqteashmd — User Manual

**Version:** 0.1.0.dev0 (development snapshot, 2026-11-06)
**Program:** `gqteashmd`, automated surface-hopping molecular dynamics with PySCF
**Licence:** MIT

> **Read this first: what works today.** This is an early development version. The **workflow engine is complete**: input file (with hardware and per-stage resource settings), checks before a run (preflight), project management, parallel tasks, crash-safe resume, graceful stop, review pauses, change detection, and a record of the software used (provenance). Its first stage is **`prepare`**, which validates and stores the molecule. **No quantum-chemistry stage exists yet.** Optimization, frequencies, Wigner sampling, initial conditions, dynamics and analysis are *planned*. They are described in §8.6–§8.13 so that you can see in advance how results will be computed. Every section is marked:
>
> - **[IMPLEMENTED]**: available in this version.
> - **[PLANNED]**: designed but not yet available; details may still change.

---

## Contents
1. [What the program does](#1-what-the-program-does)
2. [Installation](#2-installation)
3. [Quick start](#3-quick-start)
4. [The input file](#4-the-input-file)
5. [Commands](#5-commands)
6. [Project directory and output files](#6-project-directory-and-output-files)
7. [How a run works, and typical workflows](#7-how-a-run-works-and-typical-workflows)
8. [Computational methods and formulas](#8-computational-methods-and-formulas)
9. [Error messages and troubleshooting](#9-error-messages-and-troubleshooting)
10. [Known limitations](#10-known-limitations)
11. [Roadmap](#11-roadmap)
12. [References](#12-references)

---

## 1. What the program does

`gqteashmd` will run a complete photodynamics study from **one input file**. The planned stages run in this order:

| Order | Stage | What it does | Status |
|---|---|---|---|
| 0 | `prepare` | Reads and checks the molecule (XYZ syntax, elements, charge and multiplicity); stores a normalized copy | **[IMPLEMENTED]** |
| 1 | `optimize` (optional) | Ground-state geometry optimization | [PLANNED, P4] |
| 2 | `frequencies` | Hessian, harmonic frequencies and normal modes | [PLANNED, P4] |
| 3 | `sampling` | Wigner sampling of initial geometries and velocities | [PLANNED, P5] |
| 4 | `excitations` | Vertical excitation energies and oscillator strengths at each sampled geometry | [PLANNED, P6] |
| 5 | `selection` | Choice of the initial electronic state for each trajectory | [PLANNED, P6] |
| 6 | `dynamics` | Surface-hopping trajectories, run in parallel | [PLANNED, P7] |
| 7 | `analysis` (optional) | State populations, lifetimes, summary report | [PLANNED, P8] |

In SHARC 4.1 these steps are separate manual tools: `wigner.py`, `setup_init.py`, `excite.py`, `setup_traj.py` and `populations.py`. `gqteashmd` runs them automatically, one after another. You only intervene when the program asks you to fix something.

The electronic-structure program is **PySCF** [PLANNED, P3]. ORCA, Gaussian and others may be added later.

---

## 2. Installation

**[IMPLEMENTED]**

### 2.1 Requirements
- **Operating system:** Linux, with Oracle Linux 9 as the target. On Windows, use WSL2 with Oracle Linux 9. Native Windows is not supported.
- **Conda:** Miniforge or Miniconda (conda-forge channel). micromamba works the same way.
- **Git**, to obtain the source.

### 2.2 Install
Run these commands in a Linux shell:

```bash
git clone https://github.com/AdemirCamargo57/gqteaSHMD.git
cd gqteaSHMD
conda env create -f environment.yml        # creates the environment "gqteashmd-dev"
conda activate gqteashmd-dev
pip install -e . --no-deps
gqteashmd --version                        # → gqteashmd 0.1.0.dev0
```

The environment also installs PySCF, NumPy, SciPy, h5py and geomeTRIC. The current version doesn't use them yet; later versions need them.

### 2.3 Where to run projects
Keep project directories on a **local Linux disk**, for example `~/gqteashmd-runs/` or a data disk on the workstation. **Do not use OneDrive folders or Windows drives (`/mnt/c/...`).** The program writes many small files and uses file locks, and synchronized or Windows-mounted folders are slow and unreliable for both. `gqteashmd run` refuses such a project directory (§7.9). Ordinary Linux data disks mounted under `/mnt`, such as `/mnt/data`, are accepted.

---

## 3. Quick start

**[IMPLEMENTED]**

```bash
mkdir -p ~/gqteashmd-runs/water && cd ~/gqteashmd-runs/water
gqteashmd init input.toml                  # writes a commented template
sed -i 's/^total_cores = .*/total_cores = 4/; s/^total_memory_mb = .*/total_memory_mb = 6000/' input.toml
cat > molecule.xyz <<'EOF'
3
water
O    0.000000    0.000000    0.117300
H    0.000000    0.757200   -0.469200
H    0.000000   -0.757200   -0.469200
EOF
gqteashmd validate input.toml              # checks the input; runs nothing
gqteashmd run input.toml                   # runs every stage that is not yet done
gqteashmd status input.toml                # shows progress
```

The `sed` line sets `total_cores` and `total_memory_mb`, which are **required** and describe *your* workstation (the cores and the RAM in MB that you allow `gqteashmd` to use). The template's values (8 cores, 16000 MB) are only examples. If your values exceed what the program detects on the machine, `run` prints a warning but continues (§7.9). Set the two values to something that fits your machine, as above.

Expected output of `validate` (the paths are those of your machine):

```text
Input file /home/you/gqteashmd-runs/water/input.toml is valid.
  project directory: /home/you/gqteashmd-runs/water/run
  geometry:          /home/you/gqteashmd-runs/water/molecule.xyz
```

Expected output of `run` in this version (the dates differ):

```text
2026-11-06 20:40:01 INFO    gqteashmd.orchestrator: Stage prepare: starting
2026-11-06 20:40:02 INFO    gqteashmd.orchestrator: Stage prepare: done (1/1 tasks)
Workflow complete: 1 stage(s) run, 0 already up to date, 0 skipped.
```

Expected output of `status`:

```text
Project: my_molecule  (/home/you/gqteashmd-runs/water/run)
Controller: not running
stage           status         done  failed  pending  running  message
prepare         done              1       0        0        0  1/1 tasks succeeded
```

The results are in `~/gqteashmd-runs/water/run/` (see §6).

---

## 4. The input file

### 4.1 General rules
**[IMPLEMENTED]**
- **Format:** [TOML](https://toml.io). Comments start with `#`.
- **Encoding:** UTF-8; a byte-order mark is accepted. Files saved in another encoding, for example Latin-1 from an old Windows editor, are rejected with a message asking you to re-save them as UTF-8.
- **Strict validation:** every key is checked.
  - An **unknown key or section is an error**. This catches typos such as `chrage`.
  - Wrong types or out-of-range values are errors.
  - The message names the exact key, for example `molecule.chrage: Extra inputs are not permitted`.
- **Paths:** relative paths are resolved against the **directory of the input file**, not the directory you run the command from. A leading `~` means your home directory (`~/runs` and `~ajc/runs` both work). An unknown user name, as in `~nobody_here/runs`, is an input error.
- **Snapshot:** when a run starts, the resolved input is copied to `<project>/.gqteashmd/config.json`. Editing the input file while a run is in progress does not affect running tasks. The next `gqteashmd run` detects the change (see §7.5).
- **Units:** keys that carry a unit have it in their name, for example `stop_grace_period_s` (seconds). Future keys follow the same rule: `dt_fs`, `temperature_K`, `energy_window_eV`.

### 4.2 Sections and keys
**[IMPLEMENTED]** `gqteashmd init input.toml` writes this template. Everything after `#` is a comment:

```toml
# gqteashmd input file. Paths are relative to this file.
schema_version = 1

[project]
name = "my_molecule"          # letters, digits, _ . -
directory = "run"             # results go here; use a local Linux disk (not OneDrive, not /mnt/c)
seed = 12345                  # master random seed for the whole workflow

[molecule]
geometry = "molecule.xyz"     # XYZ file, Angstrom
charge = 0
multiplicity = 1

[resources]
total_cores = 8               # REQUIRED: cores of this workstation you allow gqteashmd to use
total_memory_mb = 16000       # REQUIRED: RAM (MB) you allow gqteashmd to use
max_parallel_tasks = 1        # tasks running at the same time
threads_per_task = 1          # OpenMP/BLAS threads per task
memory_mb_per_task = 2000     # passed to PySCF as max_memory (MB)
# Per-stage overrides (unset keys use the values above), e.g.:
# [resources.stages.prepare]
# threads_per_task = 2

[execution]
max_attempts = 2              # attempts per task for retryable failures
stop_grace_period_s = 30.0    # time running tasks get to stop after `gqteashmd stop`
min_free_disk_mb = 1000      # refuse to start with less free disk space (MB)

[workflow]
pause_after = []              # stage names to pause after for review, e.g. ["selection"]
```

The template omits one key, `execution.poll_interval_s` (default 0.5). The table lists every key:

| Key | Type | Default | Allowed | Meaning |
|---|---|---|---|---|
| `schema_version` | int | 1 | 1 | Input-format version |
| `project.name` | str | — (required) | `[A-Za-z0-9_.-]+` | Name used in logs and file comments |
| `project.directory` | path | — (required) | any local Linux disk | Where all results go (created if missing). Checked by preflight (§7.9) |
| `project.seed` | int | 12345 | ≥ 0 | Master seed; every random choice will derive from it [used from P5] |
| `molecule.geometry` | path | — (required) | existing XYZ file | Starting geometry |
| `molecule.charge` | int | 0 | any integer | Total charge $q$ |
| `molecule.multiplicity` | int | 1 | ≥ 1 | $M = 2S+1$; checked against the electron count (§8.2) |
| `resources.total_cores` | int | — (**required**) | ≥ 1 | Cores of your workstation that you allow the program to use ($N_{\text{cores}}$, §8.4) |
| `resources.total_memory_mb` | int | — (**required**) | ≥ 100 | RAM in MB that you allow the program to use ($M_{\text{total}}$, §8.4) |
| `resources.max_parallel_tasks` | int | 1 | ≥ 1 | Tasks running at the same time ($n_p$) |
| `resources.threads_per_task` | int | 1 | ≥ 1 | Sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` for each task ($n_t$) |
| `resources.memory_mb_per_task` | int | 2000 | ≥ 100 | Sets `PYSCF_MAX_MEMORY` for each task ($m$) |
| `resources.stages.<name>.max_parallel_tasks` | int | the global value | ≥ 1 | Override for one stage |
| `resources.stages.<name>.threads_per_task` | int | the global value | ≥ 1 | Override for one stage |
| `resources.stages.<name>.memory_mb_per_task` | int | the global value | ≥ 100 | Override for one stage |
| `execution.max_attempts` | int | 2 | ≥ 1 | Attempts before a task counts as failed |
| `execution.poll_interval_s` | float | 0.5 | > 0 | Controller polling interval (s) |
| `execution.stop_grace_period_s` | float | 30.0 | ≥ 0 | Time between SIGTERM and SIGKILL on stop (s) |
| `execution.min_free_disk_mb` | int | 1000 | ≥ 0 | `run` refuses to start if the disk holding the project has less free space (MB). `0` switches the check off |
| `workflow.pause_after` | list of str | `[]` | stage names | Pause for review after these stages (§7.8) |

**Hardware keys.** `total_cores` and `total_memory_mb` are required, so that the input file records which machine the settings were written for. The input is rejected if the settings do not fit inside them (see the rules below and §8.4). The program also detects the real machine; if your values exceed it, `run` only **warns** (§7.9).

**Per-stage overrides.** A section `[resources.stages.<name>]` changes the resources for one stage, for example to give a single-task stage the whole machine:

```toml
[resources]
total_cores = 32
total_memory_mb = 128000
max_parallel_tasks = 8
threads_per_task = 4
memory_mb_per_task = 4000

[resources.stages.frequencies]     # a stage with one big task
max_parallel_tasks = 1
threads_per_task = 32
```

- **Merge rule:** a key you leave out of an override takes the global value from `[resources]`. In the example, `frequencies` uses `memory_mb_per_task = 4000`.
- **Fit rule:** for the global settings **and for every override** (after merging), the input must satisfy $n_p n_t \le N_{\text{cores}}$ and $n_p m \le M_{\text{total}}$ (§8.4). Otherwise the input is rejected, naming `[resources]` or `[resources.stages.<name>]` and the numbers. Note that an override that sets only `max_parallel_tasks` is also checked against the global `threads_per_task`.
- **Stage names:** an override for a stage that does not exist is an input error (`[resources.stages] names unknown stage(s): …`). The only stage in this version is `prepare`; the `frequencies` example above becomes valid when that stage exists (P4).

### 4.3 The XYZ geometry file
**[IMPLEMENTED]**

```text
<number of atoms N>
<comment line, may be empty>
<symbol> <x> <y> <z>        # N lines, coordinates in Å; extra columns are ignored
```

- Element symbols H to Rn (Z = 1–86) are accepted. Case is ignored (`cl`, `CL` → `Cl`).
- The file must contain exactly N atom lines. Trailing blank lines are allowed.
- Coordinates must be finite numbers. `nan` and `inf` are rejected by the `prepare` stage (not by `validate`, which does not read the XYZ contents) with `line N: coordinates must be finite numbers, got '…'`.
- Every error message gives the line number, for example `line 3: expected 'symbol x y z'`.

### 4.4 Planned sections
**[PLANNED]** Their key names and meanings are in the master plan §4.3. Each sub-project adds the sections it needs, and this manual is updated each time. The sections for the later stages are not part of the input yet, because their keys depend on decisions that are still open:
- `[electronic]`: method (`tda`, `tddft`, `casscf`), basis, functional, number of singlet/triplet states, density fitting, CASSCF active space (P3).
- `[optimize]`, `[frequencies]` (P4); `[sampling]` (P5); `[excitations]`, `[selection]` (P6); `[dynamics]` (P7); `[analysis]` (P8).
- Electronic states will be named `S0`, `S1`, `T1`, … (arrives with `[electronic]` in P3).
- A `scratch` directory key in `[resources]` (P3, when PySCF writes temporary files).

Until then, an input file that contains one of these sections is rejected as an unknown section (§4.1).

---

## 5. Commands

**[IMPLEMENTED]**

| Command | What it does |
|---|---|
| `gqteashmd init PATH [--force]` | Writes a commented template input file. Refuses to overwrite an existing file unless `--force` is given |
| `gqteashmd validate INPUT` | Checks the input file and that the geometry is an existing regular file. Runs nothing, and does not run the preflight checks (§7.9) |
| `gqteashmd run INPUT` | Runs the preflight checks (§7.9), then runs or resumes every stage that is not yet done |
| `gqteashmd run INPUT --rerun-from STAGE` | Moves the results of `STAGE` and all later stages to `.gqteashmd/superseded/`, then recomputes them |
| `gqteashmd status INPUT` | Shows whether a controller is running, and per stage the status, the task counts (done/failed/pending/running) and the last message. It opens the progress database read-only and changes nothing, so it is safe at any time |
| `gqteashmd stop INPUT` | Asks the running controller to stop gracefully |
| `gqteashmd --version` | Prints the version |

### 5.1 Exit codes
Scripts can rely on these:

| Code | Meaning |
|---|---|
| 0 | Success (workflow complete, or command done) |
| 1 | Internal error, i.e. a bug. Please report it with the log |
| 2 | Input or configuration error: input file, geometry, arguments. Nothing was run |
| 3 | Project or state error: another controller is running, inputs changed (`--rerun-from` needed), or the state database is unreadable or corrupt |
| 4 | A stage failed: too many failed tasks, or a failed check |
| 5 | Stopped by the user, **or paused for review** (§7.8). In both cases run the same command again to continue |

Errors are printed once, as `gqteashmd: error: <what happened, where, what to do>`, with no Python traceback. The error and, for internal errors, the traceback are also written to `logs/gqteashmd.log` (§6) when the project directory exists.

### 5.2 Running unattended
A run can take days. Start it so that it survives closing the terminal:

```bash
nohup gqteashmd run input.toml > gqteashmd.out 2>&1 &      # or run it inside tmux / screen
gqteashmd status input.toml                                 # check at any time
gqteashmd stop input.toml                                   # graceful stop
```

There is no built-in `--detach` option; `nohup`, `tmux` or `screen` is the supported way.

---

## 6. Project directory and output files

**[IMPLEMENTED]** Everything for one input file lives in `project.directory`:

```text
<project>/
├── .gqteashmd/
│   ├── state.db            # progress database (SQLite). Do not edit (while a run is active,
│   │                       #   state.db-wal and state.db-shm may appear next to it)
│   ├── config.json         # resolved input snapshot used by the tasks
│   ├── provenance.json     # which software produced the results; one record per run
│   ├── controller.lock     # lock + PID of the running controller
│   └── superseded/         # old results moved aside by --rerun-from (never deleted automatically)
├── logs/
│   └── gqteashmd.log       # controller log (DEBUG level; the console shows INFO)
└── 00_prepare/
    ├── geometry.xyz        # normalized geometry (Å, 10 decimals)
    ├── molecule.json       # see below
    └── tasks/molecule/
        ├── task.json       # what the task was asked to do
        ├── result.json     # task result or error (with traceback for internal errors)
        └── worker.log      # the task's own output and error messages
```

Later stages will add `01_optimize/`, `02_frequencies/` and so on, each with the same `tasks/` structure.

**`provenance.json`** [IMPLEMENTED]. Every `gqteashmd run` that passes the preflight checks appends one record, so the file shows the software behind every run of the project. Open it with any text editor, or with `python -m json.tool .gqteashmd/provenance.json`.

```json
{"schema_version": 1, "runs": [ { "started_at": "…", "gqteashmd": "…", "…": "…" } ]}
```

| Field | Meaning |
|---|---|
| `started_at` | Start of the run, UTC, ISO 8601 |
| `gqteashmd` | Program version |
| `git_commit` | Commit of the source tree the program runs from; `null` if it is not a git checkout or git is not available |
| `git_dirty` | `true` if that source tree has uncommitted changes; `null` if unknown |
| `python` | Python version |
| `platform` | Operating system and kernel |
| `hostname` | Name of the machine |
| `cpu_count` | CPU count reported by the operating system |
| `packages` | Versions of `pydantic`, `numpy`, `scipy`, `h5py`, `pyscf`, `geometric`; `null` for one that is not installed |

**`molecule.json`** [IMPLEMENTED]:

| Field | Meaning | Unit |
|---|---|---|
| `schema_version` | File-format version (1) | — |
| `symbols` | Element symbols, in input order | — |
| `atomic_numbers` | $Z_A$ for each atom | — |
| `coords_angstrom` | Cartesian coordinates | Å |
| `charge` | $q$ | e |
| `multiplicity` | $M$ | — |
| `n_electrons` | $N_e$ (§8.2) | — |

---

## 7. How a run works, and typical workflows

**[IMPLEMENTED]**

### 7.1 Stages and tasks
- Each stage is split into **tasks**. `prepare` has one task. Planned stages have many; for example, `dynamics` will have one task per trajectory.
- Stages run in order. A stage starts only when the stages it depends on are done.
- Within a stage, up to `max_parallel_tasks` tasks run at the same time. Each task runs as a separate process, so a crashing calculation cannot bring down the controller.

### 7.2 First run
Run `gqteashmd run input.toml`. It finishes with exit code 0 and prints `Workflow complete: …`.

### 7.3 Resuming after a crash, reboot or kill
Simply **run the same command again**:
- Stages that are already done are skipped ("up to date").
- An interrupted stage continues where it stopped.
- Tasks that were running when the controller died are checked. Finished ones are kept. Left-over processes are terminated and the tasks are run again.

### 7.4 Stopping and resuming
`gqteashmd stop input.toml`, or Ctrl-C in the terminal running the controller, does the following:
1. No new tasks start.
2. Running tasks get SIGTERM, then SIGKILL after `stop_grace_period_s`.
3. The run ends with exit code 5 and the message `Stopped. Run the same command again to resume.`

Interrupted tasks are not counted as failed attempts.

### 7.5 Changing the input after results exist
The program records which inputs produced each stage's results (§8.3). Suppose you change something that affects a stage which already has results: a stage that is done, running, or has at least one finished task. The run then **stops with exit code 3 instead of mixing old and new results**:

```text
gqteashmd: error: Stage 'prepare' has results produced from different inputs (the input file or an
earlier stage changed). Run `gqteashmd run /home/ajc/gqteashmd-runs/water/input.toml --rerun-from prepare`
to move the old results aside and recompute from this stage onward.
```

Run the suggested command. The old results are moved to `.gqteashmd/superseded/<timestamp>_<stage>/`, not deleted, and the stage and all later stages are recomputed.

A stage that has **no** finished tasks (it failed, or was stopped before any task finished) is simply started over when its inputs change.

### 7.6 When a stage fails
The run stops with exit code 4 and lists up to five failed tasks with the path to each `worker.log`:

```text
gqteashmd: error: Stage 'prepare' failed: 1 of 1 tasks did not succeed (tolerated fraction: 0%).
  - molecule: TaskError: Charge 0 and multiplicity 2 are inconsistent with 10 electrons. Check [molecule] in the input file. (log: …/worker.log)
Fix the cause and run the same command again; failed tasks will be retried.
```

Fix the cause, here `multiplicity`, and run again. Because the input changed and the stage has no finished task, it simply starts over.

### 7.7 Two controllers
Only one controller can run per project. A second `gqteashmd run` on the same project exits with code 3. The error names the process id of the first controller, or says `pid unknown` when it cannot be read. The program tries the lock twice, 0.2 s apart, so a `status` command that is looking at the lock at that moment does not cause a false refusal. `status` can be used at any time.

### 7.8 Review pauses
**[IMPLEMENTED]** `[workflow] pause_after` lists stages after which the run stops so that you can inspect their results:

```toml
[workflow]
pause_after = ["selection"]        # example; "selection" is a planned stage
```

- When a listed stage **finishes in this run**, the run stops, prints the message below, and exits with code **5**:

  ```text
  Paused after stage 'selection' as requested (workflow.pause_after). Review its results, then run the same command again to continue.
  ```

- Review the results in the stage directory, then run **the same command again**. The finished stage is "up to date", so the run continues with the next stage and does not pause again.
- A stage that was **skipped** (disabled in the input) never pauses. A stage that was already up to date from an earlier run does not pause either.
- The **last** stage never pauses, because nothing follows it. In this version the only stage is `prepare`, so `pause_after` has no visible effect yet; it becomes useful when the later stages exist.
- A stage name that does not exist is an input error (`workflow.pause_after names unknown stage(s): …`), found before anything runs.

### 7.9 Checks before a run (preflight)
**[IMPLEMENTED]** `gqteashmd run` checks the machine and the input **before it creates or changes the project directory**. A refusal exits with code 2 and leaves the project directory untouched. `validate` does not run these checks.

**Checks that stop the run** (exit code 2):

| Check | Message (start) | What to do |
|---|---|---|
| Project directory on a Windows drive in WSL (`/mnt/c/…`, any single letter) | `Project directory … is on a Windows drive mounted in WSL, where file locks and many small files are unreliable. Set project.directory to a local Linux disk, e.g. ~/gqteashmd-runs/<name>.` | Choose a directory on the Linux disk. `/mnt/data` and similar Linux disks are allowed |
| Project directory inside a OneDrive folder (a path part named `OneDrive` or `OneDrive - …`, any case) | `Project directory … is on a OneDrive folder, where …` | Same |
| Free disk space below `execution.min_free_disk_mb` | `Only N MB are free on the disk holding … ; at least M MB are required (execution.min_free_disk_mb).` | Free space, choose another disk, or lower the limit |
| The free space cannot be determined | `Cannot determine free disk space for …` | Check that the path is on a readable disk |
| An enabled stage needs a Python module that is not installed | `Stage '…' needs the Python module '…', which is not installed in this environment.` | Install the module in the active conda environment |

The disk is the one that holds the project directory, or its nearest existing parent folder if the directory does not exist yet. The location check looks at the path **as written** (after `~` and relative paths are resolved); it does not follow symbolic links (§10).

**Warnings** (the run continues; they appear on the console and in `logs/gqteashmd.log`):

```text
WARNING gqteashmd.cli: Preflight: resources.total_cores = 64 exceeds the 16 cores detected on this machine; tasks may run slower than expected.
WARNING gqteashmd.cli: Preflight: resources.total_memory_mb = 16000 exceeds the 7754 MB of RAM detected on this machine; tasks may be killed for lack of memory.
```

These two warnings appear when the values you entered exceed the cores or the RAM the program detects. The program does not stop, because it trusts the values you give, for example on a machine whose detection is wrong or that is shared.

---

## 8. Computational methods and formulas

This chapter defines every quantity the program computes or will compute.
- **§8.1–§8.5** describe **what the current version does**.
- **§8.6–§8.13** describe the **planned** methods. Those formulas document the intended implementation; they are not available yet, and details may change after validation against SHARC 4.1 and the literature.

### 8.1 Units and constants
**[IMPLEMENTED]** for input and output:
- Coordinates in XYZ files are in **Ångström (Å)**.
- Charges are in units of the elementary charge $e$.
- Times in `[execution]` are in seconds.

**[PLANNED]** for the calculations:
- All internal quantities will be in **Hartree atomic units**, where $\hbar = m_e = e = 4\pi\varepsilon_0 = 1$.
- Conversions happen only when files are read or written, using CODATA values:

| Quantity | Atomic unit | Conversion |
|---|---|---|
| Length | bohr $a_0$ | 1 $a_0$ = 0.529177210903 Å |
| Energy | hartree $E_h$ | 1 $E_h$ = 27.211386245988 eV = 219474.6313632 cm⁻¹ |
| Time | $\hbar/E_h$ | 1 a.u. = 0.0241888432658 fs |
| Mass | electron mass $m_e$ | 1 u (Da) = 1822.888486 $m_e$ |
| Velocity | $a_0 E_h/\hbar$ | 1 a.u. = 2.18769126 × 10⁶ m s⁻¹ |

### 8.2 Electron count and spin-multiplicity check
**[IMPLEMENTED]**, in the `prepare` stage.

For a molecule with atoms $A = 1,\dots,N_{\text{at}}$:

$$
N_e \;=\; \sum_{A=1}^{N_{\text{at}}} Z_A \;-\; q,
\qquad
n_u \;=\; M - 1 .
$$

| Symbol | Meaning | Unit |
|---|---|---|
| $Z_A$ | Atomic number of atom $A$ | — |
| $q$ | Total molecular charge (`molecule.charge`) | e |
| $N_e$ | Total number of electrons | — |
| $M$ | Spin multiplicity (`molecule.multiplicity`), $M = 2S+1$ | — |
| $S$ | Total spin quantum number | ħ |
| $n_u$ | Number of unpaired electrons implied by $M$ (high-spin count, $n_u = 2S$) | — |

The input is accepted only if **all three** conditions hold:

$$
N_e > 0, \qquad n_u \le N_e, \qquad N_e \equiv n_u \pmod 2 .
$$

- **Why parity:** $N_e = 2n_p + n_u$, where $n_p$ is the number of electron pairs. So $N_e$ and $n_u$ must both be even or both be odd.
- **Assumptions:**
  - All electrons are counted, with no effective core potentials.
  - An ECP replaces an even number of core electrons, so it would not change the parity result.
  - The check is purely arithmetic. It does not decide whether the multiplicity is physically the ground state.
- **Examples:**

| Molecule | $\sum Z_A$ | $q$ | $M$ | $N_e$ | $n_u$ | Result |
|---|---|---|---|---|---|---|
| H₂O | 10 | 0 | 1 | 10 | 0 | accepted |
| H₂O | 10 | 0 | 2 | 10 | 1 | **rejected** (parity) |
| H₂O⁺ | 10 | +1 | 2 | 9 | 1 | accepted |
| O₂ (triplet) | 16 | 0 | 3 | 16 | 2 | accepted |
| C₂H₄ (ethylene) | 16 | 0 | 1 | 16 | 0 | accepted |

- **Interpreting the result:** `n_electrons` in `molecule.json` is $N_e$. A rejection is reported as a non-retryable task error that names the charge, the multiplicity and $N_e$ (§7.6).

### 8.3 Change detection (stage fingerprints)
**[IMPLEMENTED]**

Each stage $s$ receives a fingerprint

$$
F_s \;=\; \mathrm{SHA256}\!\left(\mathrm{JSON}\big(\,\text{name}_s,\ \text{version}_s,\ \text{inputs}_s,\ \{F_d : d \in \text{deps}(s)\}\,\big)\right),
$$

- **Inputs:**
  - $\text{inputs}_s$ are the input-file values the stage uses.
  - For `prepare`: `charge`, `multiplicity` and the SHA-256 of the geometry file's **contents**. The path is not included, so moving the file does not invalidate results, but editing it does.
- **Canonical form:** JSON is written with sorted keys and fixed separators, so the same inputs always give the same $F_s$.
- **Chaining:** a fingerprint includes the fingerprints of the stages it depends on. A change therefore invalidates every later stage that depends on it.
- **Interpreting it:** you never see $F_s$ directly. It produces the "results produced from different inputs" message in §7.5.

### 8.4 Parallel resources, retries and failure tolerance
**[IMPLEMENTED]**

**Resource use.** Let

| Symbol | Meaning | Input key | Unit |
|---|---|---|---|
| $n_p$ | Tasks running at the same time | `max_parallel_tasks` | — |
| $n_t$ | Threads per task | `threads_per_task` | — |
| $m$ | Memory per task | `memory_mb_per_task` | MB |
| $N_{\text{cores}}$ | Cores you allow the program to use | `resources.total_cores` | — |
| $M_{\text{total}}$ | RAM you allow the program to use | `resources.total_memory_mb` | MB |

A fully loaded stage uses at most $n_p n_t$ cores and $n_p m$ megabytes. The input file is accepted only if

$$
n_p\, n_t \;\le\; N_{\text{cores}},
\qquad
n_p\, m \;\le\; M_{\text{total}} .
$$

- **Where it applies:** to the global `[resources]` values **and** to each `[resources.stages.<name>]` override. For an override, $n_p$, $n_t$ and $m$ are the *merged* values: a key that the override leaves unset takes the global value (§4.2).
- **What happens on a violation:** the input is rejected (exit code 2) with a message that names the section and the numbers, for example `[resources]: max_parallel_tasks × threads_per_task = 12 exceeds total_cores = 8`.
- **Hardware detection:** the program also detects the cores and RAM of the machine. If $N_{\text{cores}}$ or $M_{\text{total}}$ is larger than the detected values, it only **warns** (§7.9). The inequalities above are checked against the values you wrote, not against the detected hardware.
- **Choosing the values:** oversubscribing cores slows every task, and exceeding RAM gets tasks killed by the operating system. Set $N_{\text{cores}}$ and $M_{\text{total}}$ to what you really want to give the program, leaving room for the operating system.

**Retries.** Let $a_i$ be the number of attempts used by task $i$, and $a_{\max}$ = `max_attempts`.
- A task is retried after a crash or a *retryable* error while $a_i < a_{\max}$.
- A non-retryable error, such as invalid input, fails the task immediately.
- An attempt interrupted by `gqteashmd stop` is not counted.
- Running `gqteashmd run` again after a failed stage gives the failed tasks a fresh set of $a_{\max}$ attempts.

**Failure tolerance.** With $n_{\text{fail}}$ failed tasks out of $n_{\text{tot}}$, a stage **fails** if

$$
\frac{n_{\text{fail}}}{n_{\text{tot}}} \;>\; f_{\max},
$$

where $f_{\max}$ is the stage's tolerated fraction. For `prepare`, $f_{\max} = 0$: any failure stops the workflow. Planned stages with many tasks will tolerate a small fraction, for example a few trajectories whose electronic structure fails. The tolerated tasks are reported, not hidden.

### 8.5 Graceful stop timing
**[IMPLEMENTED]** On stop, each running task's process group receives SIGTERM at time $t_0$ and SIGKILL at $t_0 + t_g$, with $t_g$ = `stop_grace_period_s`. A task that finishes within $t_g$ keeps its result. Planned dynamics tasks will save a restart file after every step, so at most one step is lost.

---

> **Everything below is [PLANNED].** The stages, keys and outputs in §8.6–§8.13 do not exist in the current version.

### 8.6 Electronic structure (PySCF)
**[PLANNED, P3]**
- **Methods:** TDA/TDDFT (singlets first) and state-averaged CASSCF (SA-CASSCF), with optional MC-PDFT.
- **Quantities:** for each geometry $\mathbf R$ the program will compute:
  - adiabatic energies $E_I(\mathbf R)$ of states $I = S_0, S_1, \dots$;
  - gradients $\nabla E_I$;
  - transition dipoles $\boldsymbol\mu_{0I}$;
  - nonadiabatic couplings $\mathbf d_{IJ} = \langle \Psi_I | \nabla_{\mathbf R} \Psi_J \rangle$ for CASSCF, or wavefunction overlaps $S_{IJ}(t) = \langle \Psi_I(t-\Delta t) | \Psi_J(t) \rangle$.
- **Validation:** ethylene at SA2-CASSCF(2,2)/6-31G\*, compared with SHARC's PySCF interface.
- **Assumption (TDDFT):** linear-response TDDFT describes the S₁/S₀ crossing seam incorrectly. Use CASSCF when S₁→S₀ decay matters.

### 8.7 Harmonic analysis (frequencies)
**[PLANNED, P4]** From the Cartesian Hessian $H_{ij} = \partial^2 E_0 / \partial x_i \partial x_j$ at the optimized geometry:

$$
H^{\text{mw}}_{ij} = \frac{H_{ij}}{\sqrt{m_i m_j}}, \qquad
H^{\text{mw}} \mathbf L_k = \lambda_k \mathbf L_k, \qquad
\omega_k = \sqrt{\lambda_k}, \qquad
\tilde\nu_k\,[\text{cm}^{-1}] = 219474.63\;\omega_k\,[\text{a.u.}] .
$$

| Symbol | Meaning | Unit |
|---|---|---|
| $x_i$ | Cartesian coordinate $i$ ($3N_{\text{at}}$ of them) | $a_0$ |
| $m_i$ | Mass of the atom that owns coordinate $i$ (most abundant isotope; open item O5) | $m_e$ |
| $H_{ij}$ | Hessian | $E_h\,a_0^{-2}$ |
| $\lambda_k$, $\mathbf L_k$ | Eigenvalues and orthonormal eigenvectors (normal modes) of the mass-weighted Hessian | a.u. |
| $\omega_k$ | Angular frequency | $E_h/\hbar$ |
| $\tilde\nu_k$ | Wavenumber | cm⁻¹ |

- **Removed modes:** translations and rotations are projected out, leaving $3N_{\text{at}}-6$ vibrational modes, or $3N_{\text{at}}-5$ for linear molecules.
- **Imaginary modes:** $\lambda_k < 0$ means an imaginary frequency, which is reported as a negative wavenumber. If $|\tilde\nu_k|$ exceeds `imaginary_cutoff_cm1` (planned default 20 cm⁻¹), the geometry is not a minimum and the workflow stops.
- **Assumptions:** harmonic approximation, and the Born–Oppenheimer ground state.
- **Output:** `frequencies.h5` and a Molden file for visualization.

### 8.8 Wigner sampling of initial conditions
**[PLANNED, P5]**
- **Coordinates:** each vibrational mode $k$ is treated as an independent harmonic oscillator in mass-weighted normal coordinate $Q_k$ and momentum $P_k$.
- **Distribution at $T = 0$:** the Wigner distribution of the vibrational ground state is

$$
W_k(Q_k, P_k) = \frac{1}{\pi\hbar}\exp\!\left(-\frac{\omega_k Q_k^2}{\hbar} - \frac{P_k^2}{\hbar\,\omega_k}\right),
$$

i.e. $Q_k$ and $P_k$ are independent Gaussians with

$$
\sigma_{Q,k}^2 = \frac{\hbar}{2\omega_k}, \qquad \sigma_{P,k}^2 = \frac{\hbar\,\omega_k}{2}.
$$

- **Cartesian geometry and velocities** (with $\mathbf M$ the diagonal $3N_{\text{at}}\times 3N_{\text{at}}$ matrix of masses):

$$
\mathbf x = \mathbf x_{\text{eq}} + \mathbf M^{-1/2}\sum_k \mathbf L_k Q_k,
\qquad
\mathbf v = \mathbf M^{-1/2}\sum_k \mathbf L_k P_k .
$$

- **Check performed automatically:** the mean harmonic energy per mode must approach the zero-point energy,
  $\langle \tfrac12 P_k^2 + \tfrac12\omega_k^2 Q_k^2\rangle = \tfrac12\hbar\omega_k$.
  Net linear and angular momentum are removed from each sample.
- **Temperature $T > 0$** (open item O4): one option replaces $\hbar/(2\omega_k)$ by $\frac{\hbar}{2\omega_k}\coth\!\big(\frac{\hbar\omega_k}{2k_BT}\big)$, and likewise for $\sigma_P^2$. The exact scheme is decided in P5.
- **Assumptions:** harmonic, uncoupled modes. Very low-frequency (floppy) modes are poorly described.
- **Interpreting the samples:** each sample is one classical "snapshot" of the quantum vibrational distribution. Distributions of bond lengths and kinetic energies will be compared with SHARC's `wigner.py`.

### 8.9 Initial excitation
**[PLANNED, P6]** At each sampled geometry, vertical excitation energies $\Delta E_{0J} = E_J - E_0$ and oscillator strengths are computed:

$$
f_{0J} = \frac{2}{3}\,\Delta E_{0J}\,\big|\boldsymbol\mu_{0J}\big|^2 \qquad \text{(atomic units)},
$$

where $\boldsymbol\mu_{0J} = \langle \Psi_0 | \hat{\boldsymbol\mu} | \Psi_J \rangle$ is the transition dipole in $e\,a_0$, and $f_{0J}$ is dimensionless.
- **Spectrum:** the absorption spectrum is the sum of $f_{0J}$ over all samples, broadened with Gaussians.
- **Initial states:** they are chosen within a user-defined energy window, with a probability that increases with $f_{0J}$. The exact rule is open item O3 and will be documented here when fixed.

### 8.10 Nuclear motion (velocity Verlet)
**[PLANNED, P7]** Nuclei move classically on the active surface $\lambda$:

$$
\mathbf a(t) = -\mathbf M^{-1}\nabla E_\lambda(\mathbf R(t)),\qquad
\mathbf R(t+\Delta t) = \mathbf R + \mathbf v\,\Delta t + \tfrac12 \mathbf a\,\Delta t^2,\qquad
\mathbf v(t+\Delta t) = \mathbf v + \tfrac12\big[\mathbf a(t) + \mathbf a(t+\Delta t)\big]\Delta t .
$$

- **Inputs:** $\Delta t$ = `dt_fs`; a typical value is 0.5 fs.
- **Quality check:** the total energy $E_{\text{tot}} = \tfrac12\mathbf v^{\mathsf T}\mathbf M\mathbf v + E_\lambda$ should be conserved. A trajectory whose drift $|E_{\text{tot}}(t) - E_{\text{tot}}(0)|$ exceeds a limit (planned 0.5 eV, open item O8) is ended and flagged.

### 8.11 Electronic motion
**[PLANNED, P7]** The electronic wavefunction $\Psi(t) = \sum_J c_J(t)\,\Psi_J$ obeys

$$
i\hbar\,\dot c_I = \sum_J \big(H_{IJ} - i\hbar\,T_{IJ}\big)\, c_J,
\qquad
T_{IJ} = \Big\langle \Psi_I \Big| \frac{\partial \Psi_J}{\partial t}\Big\rangle = \mathbf v\cdot\mathbf d_{IJ}.
$$

| Symbol | Meaning |
|---|---|
| $c_J$ | Complex amplitude of state $J$ (dimensionless); $\lvert c_J\rvert^2$ is the quantum population |
| $H_{IJ}$ | Electronic Hamiltonian in the state basis (diagonal $E_I$ without spin–orbit coupling), in $E_h$ |
| $T_{IJ}$ | Time-derivative coupling, in inverse time (atomic unit: $E_h/\hbar$) |

- **Integration:** with small sub-steps between nuclear steps.
- **Overlaps:** with overlap matrices, the robust *local diabatization* scheme of Granucci et al. (2001) is used.
- **Norm:** $\sum_J |c_J|^2 = 1$ is conserved and monitored.

### 8.12 Surface hopping, energy adjustment, decoherence
**[PLANNED, P7]**

**Hopping probability** (Tully fewest switches) from the active state $\lambda$ to $J$ in one step:

$$
P_{\lambda\to J} = \max\!\left(0,\; -\frac{2\,\Delta t}{|c_\lambda|^2}\,\mathrm{Re}\!\left(c_J^{*}\,c_\lambda\,T_{J\lambda}\right)\right).
$$

A uniform random number $\zeta\in[0,1)$ selects the hop when $\sum_{K<J} P_{\lambda\to K} < \zeta \le \sum_{K\le J} P_{\lambda\to K}$. The random numbers come from `project.seed`, so runs are reproducible. SHARC's generalization for arbitrary representations is also planned.

**Energy conservation at a hop** (rescaling along the velocity). With kinetic energy $K$:

$$
\mathbf v' = \alpha\,\mathbf v,\qquad \alpha = \sqrt{1 - \frac{E_J - E_\lambda}{K}} .
$$

If $K < E_J - E_\lambda$, the hop is **frustrated**: it is rejected and logged. Rescaling along the nonadiabatic coupling vector or the gradient difference is planned as an option.

**Decoherence** (energy-based correction; Granucci & Persico 2007). After each step, every inactive amplitude decays:

$$
c_J \leftarrow c_J\, e^{-\Delta t/\tau_J},\qquad
\tau_J = \frac{\hbar}{|E_J - E_\lambda|}\left(1 + \frac{C}{K}\right),\qquad J\neq\lambda,
$$

and $c_\lambda$ is rescaled to restore $\sum|c_J|^2 = 1$. The planned default is $C = 0.1\,E_h$.

**Assumptions:**
- Classical nuclei.
- Independent trajectories.
- The surface-hopping ansatz, which approximates the quantum branching of the nuclear wavepacket.

### 8.13 Analysis: populations and lifetimes
**[PLANNED, P8]** With $N$ trajectories and $N_J(t)$ of them in state $J$ at time $t$:

$$
P_J(t) = \frac{N_J(t)}{N},\qquad
\sigma_{P_J}(t) \approx \sqrt{\frac{P_J(t)\,[1-P_J(t)]}{N}} .
$$

- **Two population measures:**
  - The quantum population $\bar\rho_J(t) = \frac1N\sum_n |c_J^{(n)}(t)|^2$ is also reported.
  - Agreement between $P_J$ and $\bar\rho_J$ indicates internally consistent dynamics.
- **Lifetimes:** these come from kinetic fits, for example $P_{S_1}(t) = e^{-t/\tau}$, with fit uncertainties.
- **Precision and ensemble size:** with $N = 100$, $\sigma_P \approx 0.05$ at $P = 0.5$. Differences smaller than about $2\sigma$ are not significant.

---

## 9. Error messages and troubleshooting

**[IMPLEMENTED]**

| Message (start) | Cause | What to do |
|---|---|---|
| `Invalid input file …: - molecule.chrage: Extra inputs are not permitted` | Typo or unknown key | Correct the key name (§4.2) |
| `Input file … is not valid UTF-8 text` | The file was saved in another encoding | Re-save it as UTF-8 |
| `Input file … is not valid TOML` | Syntax error (quotes, brackets) | Fix the line shown |
| `Geometry file not found` / `Cannot read geometry file` | Wrong path, or the path is a directory | Paths are relative to the input file |
| `Invalid geometry file …: line N: …` | XYZ format problem | See §4.3 |
| `Charge … and multiplicity … are inconsistent with … electrons` | Parity rule in §8.2 | Correct `charge` or `multiplicity` |
| `Another gqteashmd controller is running for this project (pid …` | A run is already active for this project (`pid unknown` if it cannot be read) | Use `status`, or `stop` first |
| `Invalid input file …: - resources.total_cores: Field required` | A required hardware key is missing | Add `total_cores` and `total_memory_mb` to `[resources]` (§4.2) |
| `[resources]: max_parallel_tasks × threads_per_task = … exceeds total_cores = …` (also `… × memory_mb_per_task = … MB exceeds total_memory_mb = …`, and the same for `[resources.stages.<name>]`) | The resource settings do not fit in the machine you described | Lower `max_parallel_tasks`, `threads_per_task` or `memory_mb_per_task`, or raise the totals (§8.4) |
| `[resources.stages] names unknown stage(s): …; stages are: …` | A resource override for a stage that does not exist | Correct the stage name |
| `workflow.pause_after names unknown stage(s): …; stages are: …` | A pause for a stage that does not exist | Correct the stage name |
| `Cannot expand '~' in path …` | `~user` names a user that does not exist | Use a real user name or a full path |
| `Invalid geometry file …: line N: coordinates must be finite numbers, got …` | `nan` or `inf` in the XYZ file (reported when `prepare` runs) | Correct the coordinates |
| `Project directory … is on a Windows drive mounted in WSL …` / `… on a OneDrive folder …` | Preflight refusal (§7.9) | Use a local Linux disk |
| `Only N MB are free on the disk holding … ` | Not enough free disk space (§7.9) | Free space or lower `execution.min_free_disk_mb` |
| `Stage '…' needs the Python module '…', which is not installed` | Preflight: a missing module | Install it in the conda environment |
| `Preflight: resources.total_cores = … exceeds the … cores detected` (also for memory) | **Warning**, not an error (§7.9) | Check that the values describe your machine |
| `Paused after stage '…' as requested (workflow.pause_after)` | Not an error: exit code 5 (§7.8) | Review, then run the same command again |
| `State database … is unreadable or corrupt: …` | `state.db` is damaged, is not a database, or is a directory. A message `… has schema version …` or `… unknown stage/task status …` means another program version wrote it | Move `.gqteashmd/state.db` aside to start the project over (finished results stay on disk), or use the matching version |
| `Stage '…' has results produced from different inputs` | The input changed after results existed | Run the suggested `--rerun-from` command (§7.5) |
| `Stage '…' failed: k of n tasks did not succeed` | Task failures | Read the listed `worker.log` files, fix, run again |
| `internal error: …` | A program bug | Report it with `logs/gqteashmd.log` and the task's `result.json` |

---

## 10. Known limitations

- **Feature scope:** only the `prepare` stage exists. No quantum chemistry is performed yet, so the hardware and per-stage resource settings, and `pause_after`, have little visible effect until the later stages arrive.
- **Platform:** Linux only, with no batch-scheduler support. Runs are designed for a single workstation.
- **Project-folder check:** the check for Windows drives and OneDrive (§7.9) looks at the text of the path. An absolute path that is a symbolic link into a Windows drive or a OneDrive folder is not detected.
- **Provenance file:** `provenance.json` is written by `gqteashmd` itself. If it was damaged and contains bytes that are not valid UTF-8, the next run stops with an error. Move the file aside in that case. This is unlikely in normal use.
- **No automated build checks yet:** the continuous-integration workflow has never run, because the repository has not been pushed to GitHub. The tests have only been run on the development machine.

## 11. Roadmap

P1 foundation (done) → P2 input schema and pre-run checks (done) → **P3 PySCF backend (next)** → P4 optimization and frequencies → P5 Wigner sampling → P6 excitations and initial states → P7 dynamics → P8 analysis → P9 systematic validation against SHARC 4.1. This manual is updated with each step. Sections marked [PLANNED] become [IMPLEMENTED] when their sub-project is finished and validated.

## 12. References
1. J. C. Tully, *J. Chem. Phys.* **93**, 1061 (1990): fewest-switches surface hopping.
2. G. Granucci, M. Persico, A. Toniolo, *J. Chem. Phys.* **114**, 10608 (2001): local diabatization.
3. G. Granucci, M. Persico, *J. Chem. Phys.* **126**, 134114 (2007): energy-based decoherence correction.
4. S. Mai, P. Marquetand, L. González, *WIREs Comput. Mol. Sci.* **8**, e1370 (2018): the SHARC approach.
5. E. Wigner, *Phys. Rev.* **40**, 749 (1932); M. Barbatti, K. Sen, *Int. J. Quantum Chem.* **116**, 762 (2016): Wigner sampling for initial conditions.
6. Q. Sun et al., *J. Chem. Phys.* **153**, 024109 (2020): PySCF.
7. CODATA 2018 recommended values of the fundamental physical constants.
