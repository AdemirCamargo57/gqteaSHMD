# gqteashmd — User Manual

**Version:** 0.1.0.dev0 (development snapshot, 2026-11-07)
**Program:** `gqteashmd`, automated surface-hopping molecular dynamics with PySCF
**Licence:** MIT

> **Read this first: what works today.** This is an early development version. The **workflow engine is complete**: input file (with hardware and per-stage resource settings), checks before a run (preflight), project management, parallel tasks, crash-safe resume, graceful stop, review pauses, change detection, and a record of the software used (provenance). Its first stage is **`prepare`**, which validates and stores the molecule. The **electronic-structure layer** (PySCF: SCF, SA-CASSCF, TDA, TDDFT, CIS, TDHF; energies, analytic gradients, transition dipoles) is also **implemented** and validated against SHARC 4.1; you can use it today with **`gqteashmd single-point`** (§5.3). **No quantum-chemistry *stage* exists yet.** Optimization, frequencies, Wigner sampling, initial conditions, dynamics and analysis are *planned*; so are the use of nonadiabatic couplings and wavefunction overlaps in dynamics, and the Hessian (SA-CASSCF couplings and the overlap engine exist as a library, §8.6.8, but no command reaches them). They are described in §8.7–§8.13 so that you can see in advance how results will be computed. Every section is marked:
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

The electronic-structure program is **PySCF** **[IMPLEMENTED, P3a]** as a library layer: the `[electronic]` input section (§4.2) and the command `gqteashmd single-point` (§5.3) use it today. The workflow **stages** that will call it (`optimize`, `frequencies`, `excitations`, `dynamics`) are still planned, as is their use in dynamics; SA-CASSCF nonadiabatic couplings, wavefunction overlaps and phase tracking (P3b) exist as a library only (§8.6.8). ORCA, Gaussian and others may be added later.

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

The environment installs PySCF (pinned to the 2.14 series), NumPy, SciPy, h5py and geomeTRIC. The program itself requires `pydantic>=2.6`, `numpy>=1.26` and `pyscf>=2.6`; SciPy, h5py and geomeTRIC are for later stages. A PySCF older than 2.6 is refused when an electronic-structure calculation starts (§9).

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

# Electronic structure (needed by the quantum-chemistry stages and `gqteashmd single-point`):
# [electronic]
# method = "casscf"             # "casscf" | "tda" | "tddft"
# basis = "6-31g*"
# singlets = 2                  # number of singlet states INCLUDING S0
# [electronic.casscf]
# ncas = 2                      # active orbitals
# nelecas = 2                   # active electrons
```

The `[electronic]` block is commented out in the template: it is optional for `run` (today's only stage, `prepare`, does not use it) and required by `single-point`. The template omits one key, `execution.poll_interval_s` (default 0.5). The table lists every key:

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
| `execution.min_free_disk_mb` | int | 1000 | ≥ 0 | `run` refuses to start if the disk holding the project has less free space (MB). `0` switches the check off (the disk is then not queried at all) |
| `workflow.pause_after` | list of str | `[]` | stage names | Pause for review after these stages (§7.8) |
| `electronic.method` | str | — (required if `[electronic]` is present) | `"casscf"`, `"tda"`, `"tddft"` | Electronic-structure method (§8.6). `"tda"` and `"tddft"` are linear-response methods on a restricted SCF reference |
| `electronic.basis` | str | — (required) | any PySCF basis name, e.g. `"6-31g*"`, `"def2-svp"` | Gaussian basis set. An unknown name fails at the first calculation (§9) |
| `electronic.functional` | str | none | required for `tda`/`tddft`; **not allowed** for `casscf` | Exchange–correlation functional, e.g. `"b3lyp"`. `"hf"` selects Hartree–Fock: `tda` + `"hf"` is CIS, `tddft` + `"hf"` is TDHF (§8.6.3) |
| `electronic.singlets` | int | — (required) | ≥ 1; for `casscf` at most the number of singlet states of the active space (§4.2 rules) | Number of singlet states **including S0**: `singlets = 2` gives S0 and S1 |
| `electronic.triplets` | int | 0 | must be 0 | Triplet states are **not supported yet** (planned together with intersystem crossing); any value above 0 is an input error |
| `electronic.grid_level` | int | 3 | 0–9 | PySCF integration-grid level for DFT (`tda`/`tddft` with a functional other than `"hf"`). Use **≥ 5 for dynamics** (§8.6.4) |
| `electronic.scf_conv_tol` | float | 1e-10 | > 0 | SCF energy convergence threshold (Eh) |
| `electronic.scf_max_cycle` | int | 100 | ≥ 1 | Maximum SCF iterations per rung of the convergence ladder (§8.6.1) |
| `electronic.td_conv_tol` | float | 1e-8 | > 0 | Convergence threshold of the linear-response (Davidson) solver |
| `electronic.casscf.ncas` | int | — (required for `casscf`) | ≥ 1 | Number of active orbitals $n_{\text{cas}}$ |
| `electronic.casscf.nelecas` | int | — (required for `casscf`) | even, ≥ 1, ≤ 2 `ncas` | Number of active electrons $n_{e,\text{cas}}$ (singlet states need an even number) |
| `electronic.casscf.conv_tol` | float | 1e-10 | > 0 | SA-CASSCF energy convergence threshold (Eh) |
| `electronic.casscf.conv_tol_grad` | float | 1e-6 | > 0 | SA-CASSCF orbital-gradient convergence threshold. Tightening it much further (for example to 1e-7 with `conv_tol` 1e-12) can fail to converge at displaced geometries |

**Hardware keys.** `total_cores` and `total_memory_mb` are required, so that the input file records which machine the settings were written for. The input is rejected if the settings do not fit inside them (see the rules below and §8.4). The program also detects the real machine; if your values exceed it, `run` only **warns** (§7.9).

**Per-stage overrides.** A section `[resources.stages.<name>]` changes the resources for one stage. A runnable example for this version, which has only the `prepare` stage:

```toml
[resources]
total_cores = 8
total_memory_mb = 16000

[resources.stages.prepare]         # prepare gets 2 threads; other keys use the [resources] values
threads_per_task = 2
```

**From P4 on** (when the `frequencies` stage exists; today `validate` rejects this example), the same mechanism can give a single-task stage the whole machine:

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

- **Merge rule:** a key you leave out of an override takes the global value from `[resources]`. In the second example, `frequencies` uses `memory_mb_per_task = 4000`.
- **Fit rule:** for the global settings **and for every override** (after merging), the input must satisfy $n_p n_t \le N_{\text{cores}}$ and $n_p m \le M_{\text{total}}$ (§8.4). Otherwise the input is rejected, naming `[resources]` or `[resources.stages.<name>]` and the numbers. Note that an override that sets only `max_parallel_tasks` is also checked against the global `threads_per_task`.
- **Stage names:** an override for a stage that does not exist is an input error (`[resources.stages] names unknown stage(s): …`). The only stage in this version is `prepare`; the `frequencies` example above becomes valid when that stage exists (P4).

#### The `[electronic]` section
**[IMPLEMENTED]** (P3a). A complete example for ethylene, SA2-CASSCF(2,2)/6-31G\*:

```toml
[electronic]
method = "casscf"
basis = "6-31g*"
singlets = 2                      # S0 and S1

[electronic.casscf]
ncas = 2
nelecas = 2
```

and a TDA example (TDA-B3LYP, three excited states plus S0):

```toml
[electronic]
method = "tda"
functional = "b3lyp"
basis = "def2-svp"
singlets = 4                      # S0, S1, S2, S3
grid_level = 5
```

The rules, all checked when the input is loaded (`validate`, `run`, `single-point`), with exit code 2:

- `method = "casscf"` needs an `[electronic.casscf]` table and does **not** accept `functional`. `method = "tda"` or `"tddft"` needs `functional` and does **not** accept `[electronic.casscf]`.
- `nelecas` must be even and at most `2 × ncas`.
- For `casscf`, `singlets` cannot exceed the number of singlet states of the active space, $N_0 = \binom{n+1}{N/2}\binom{n+1}{N/2+1}/(n+1)$ with $n$ = `ncas` and $N$ = `nelecas` (CAS(2,2): 3, CAS(4,4): 20, CAS(6,6): 175). Example: `singlets = 4` with CAS(2,2) is rejected: *electronic.singlets = 4 exceeds the 3 singlet states of CAS(2,2) (nelecas = 2, ncas = 2)*.
- `triplets > 0` is rejected: *triplets are not supported yet*.
- When `[electronic]` is present, `molecule.multiplicity` must be 1: *the PySCF backend supports closed-shell singlet molecules only*. A charge that is inconsistent with the electron count (§8.2) is also an input error (exit code 2), but it is **not** found by `validate`, which does not read the geometry: `single-point` finds it when it builds the backend, and `run` finds it in the `prepare` stage.
- **State labels** are `S0` (ground state), `S1`, `S2`, … (0-based, singlets); triplets will be `T1`, `T2`, … when they arrive. `singlets` counts S0, so `singlets = N` gives S0 … S(N−1).
- `overlap_norm_kept` (float, default `0.9999`, allowed range $0.9 < x \le 1$): the fraction of each state's squared wavefunction norm that the determinant expansion must keep when wavefunction overlaps between geometries are computed (§8.6.8). Smaller values are faster and less accurate; `1.0` keeps every determinant. Values $\le 0.9$ or $> 1$ are an input error. It only matters for the overlap/phase machinery, which the CLI does not reach yet (§8.6.8).
- Not implemented yet: density fitting, spin-flip or unrestricted references, ECP-specific options, a `scratch` directory key in the input file (the backend itself already supports a scratch directory, §8.6.8), MC-PDFT.

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
- `[optimize]`, `[frequencies]` (P4); `[sampling]` (P5); `[excitations]`, `[selection]` (P6); `[dynamics]` (P7); `[analysis]` (P8).
- Additional `[electronic]` keys: density fitting, MC-PDFT, triplet states (with intersystem crossing).
- A `scratch` directory key in the input file (the backend takes a scratch directory and has no checkpoint file since P3b, §8.6.8; the input key arrives with the stage that uses it).

Until then, an input file that contains one of these sections or keys is rejected as unknown (§4.1).

---

## 5. Commands

**[IMPLEMENTED]**

| Command | What it does |
|---|---|
| `gqteashmd init PATH [--force]` | Writes a commented template input file. Refuses to overwrite an existing file unless `--force` is given |
| `gqteashmd validate INPUT` | Checks the input file, that the geometry is an existing regular file, and that `[resources.stages]` and `workflow.pause_after` name stages that exist. It does not read the XYZ contents (coordinates, elements, charge/multiplicity); `prepare` does. Runs nothing, and does not run the preflight checks (§7.9) |
| `gqteashmd run INPUT` | Runs the preflight checks (§7.9), then runs or resumes every stage that is not yet done |
| `gqteashmd run INPUT --rerun-from STAGE` | Moves the results of `STAGE` and all later stages to `.gqteashmd/superseded/`, then recomputes them |
| `gqteashmd status INPUT` | Shows whether a controller is running, and per stage the status, the task counts (done/failed/pending/running) and the last message. It opens the progress database read-only and changes nothing, so it is safe at any time |
| `gqteashmd stop INPUT` | Asks the running controller to stop gracefully |
| `gqteashmd single-point INPUT [--json FILE] [--log FILE]` | Computes the electronic structure at the input geometry (§5.3) |
| `gqteashmd --version` | Prints the version |

### 5.1 Exit codes
Scripts can rely on these:

| Code | Meaning |
|---|---|
| 0 | Success (workflow complete, or command done) |
| 1 | Internal error, i.e. a bug. Please report it with the log |
| 2 | Input or configuration error: input file, geometry, arguments. Nothing was run |
| 3 | Project or state error: another controller is running, inputs changed (`--rerun-from` needed), or the state database is unreadable or corrupt |
| 4 | A stage failed (too many failed tasks, or a failed check), **or an electronic-structure calculation failed** (`single-point`: no convergence, unknown basis, wrong spin, …) |
| 5 | Stopped by the user, **or paused for review** (§7.8). In both cases run the same command again to continue |

Errors are printed once, as `gqteashmd: error: <what happened, where, what to do>`, with no Python traceback. Only errors raised during `gqteashmd run`, after logging is set up, also reach `logs/gqteashmd.log` (§6), with the traceback for internal errors. Load errors, preflight errors, and errors from `validate`, `status`, `init` and `stop` go to the console only.

### 5.2 Running unattended
A run can take days. Start it so that it survives closing the terminal:

```bash
nohup gqteashmd run input.toml > gqteashmd.out 2>&1 &      # or run it inside tmux / screen
gqteashmd status input.toml                                 # check at any time
gqteashmd stop input.toml                                   # graceful stop
```

There is no built-in `--detach` option; `nohup`, `tmux` or `screen` is the supported way.

### 5.3 `gqteashmd single-point`
**[IMPLEMENTED]** (P3a). It computes the energies of S0 … S(N−1), the S0→SI transition dipoles and the oscillator strengths at the geometry of the input file. Use it to test an electronic-structure setup (method, basis, active space) before a long run.

```text
gqteashmd single-point INPUT [--json FILE] [--log FILE]
```

| Option | Meaning |
|---|---|
| `--json FILE` | Also write the results to a JSON file (fields below) |
| `--log FILE` | Write PySCF's own output (SCF iterations, CASSCF macro-iterations, …) to this file. Without it PySCF prints nothing |

- **Needs:** a valid input file with an `[electronic]` section (§4.2) and an existing geometry. `[molecule]` charge and multiplicity must be consistent with the elements (§8.2), and the multiplicity must be 1.
- **Resources:** it uses `resources.threads_per_task` threads and `resources.memory_mb_per_task` MB (as PySCF `max_memory`). It does not run the preflight checks (§7.9), does not create a project directory, and does not use `[execution]` or `[workflow]`.
- **Not computed:** gradients (the backend can compute them; the command does not ask for them), nonadiabatic couplings, overlaps.

**Example.** Ethylene, SA2-CASSCF(2,2)/6-31G, in an empty directory with `molecule.xyz` (the 6-atom ethylene geometry, in Å) next to this `input.toml`:

```toml
[project]
name = "ethylene"
directory = "run"

[molecule]
geometry = "molecule.xyz"

[resources]
total_cores = 4
total_memory_mb = 4000
threads_per_task = 2

[electronic]
method = "casscf"
basis = "6-31g"
singlets = 2

[electronic.casscf]
ncas = 2
nelecas = 2
```

```bash
gqteashmd single-point input.toml --json sp.json --log pyscf.log
```

The console output (real output from a development machine; the PySCF version and, in the JSON, `wall_time_s` differ on yours):

```text
Single point: casscf / 6-31g (PySCF 2.14.0)
state        energy (Eh)   dE (eV)        f
S0         -78.025499771    0.0000        -
S1         -77.638305147   10.5361   0.6206
Results written to sp.json
```

The table columns are the absolute energy $E_I$ in hartree, the excitation energy $\Delta E_{0I} = E_I - E_0$ in eV, and the oscillator strength $f_{0I}$ (§8.6.6; `-` for S0). The command exits with code 0.

**JSON file** (`schema_version` 1; all numbers in the units shown):

| Field | Meaning | Unit |
|---|---|---|
| `schema_version` | File-format version (1) | — |
| `states[]` | One entry per state, S0 first | — |
| `states[].label` | State label, e.g. `"S1"` | — |
| `states[].energy_hartree` | Total energy $E_I$ | $E_h$ |
| `states[].excitation_ev` | $E_I - E_0$ | eV |
| `states[].transition_dipole_au` | $\boldsymbol\mu_{0I}$ as `[x, y, z]`; `null` for S0. **The overall sign is arbitrary** (§8.6.5) | $e\,a_0$ |
| `states[].oscillator_strength` | $f_{0I}$; `null` for S0 | — |
| `states[].spin_square` | $\langle S^2\rangle$ of the state (CASSCF only; `null` for linear response) | — |
| `gradients_hartree_per_bohr` | Gradient per state label; empty (`{}`) for `single-point` | $E_h/a_0$ |
| `meta.backend`, `meta.pyscf_version` | `"pyscf"` and its version | — |
| `meta.method`, `meta.basis`, `meta.functional` | Copied from the input (`functional` is `null` for CASSCF) | — |
| `meta.scf` | The rung of the SCF ladder that converged: `"default"`, `"level shift + damping"` or `"second-order"` (§8.6.1) | — |
| `meta.scf_energy_hartree` | $E_{\text{SCF}}$ of the reference (§8.6.1) | $E_h$ |
| `meta.wall_time_s` | Wall time of the calculation | s |

For the example above the JSON contains (shortened):

```json
{"schema_version": 1,
 "states": [
  {"label": "S0", "energy_hartree": -78.025499770717, "excitation_ev": 0.0,
   "transition_dipole_au": null, "oscillator_strength": null, "spin_square": 0.0},
  {"label": "S1", "energy_hartree": -77.63830514694367, "excitation_ev": 10.536102459866353,
   "transition_dipole_au": [1.5496152512384933, 0.054692141511617484, 2.4863201618893523e-07],
   "oscillator_strength": 0.6206210093458465, "spin_square": 0.0}],
 "gradients_hartree_per_bohr": {},
 "meta": {"backend": "pyscf", "basis": "6-31g", "functional": null, "method": "casscf",
          "pyscf_version": "2.14.0", "scf": "default",
          "scf_energy_hartree": -78.00336573553878, "wall_time_s": 0.21}}
```

The sign of `transition_dipole_au` in this example may differ from other runs, geometries or programs: the overall sign of a transition dipole is arbitrary (§8.6.5).

**Exit codes of `single-point`:**

| Code | When |
|---|---|
| 0 | The calculation finished |
| 2 | Input error: invalid input file or geometry, a missing `[electronic]` section, a multiplicity other than 1, charge/multiplicity inconsistent with the electron count, PySCF missing or older than 2.6 |
| 4 | The calculation failed: unknown basis, SCF or CASSCF/TD non-convergence, a root that is not a singlet, an active space that does not fit, a failed gradient or dipole |
| 1 | Internal error (a bug) |

Errors are printed once as `gqteashmd: error: …` (§5.1). An electronic-structure failure is not retried, because the same input would fail again, with one exception: a `MemoryError` or `OSError` raised inside PySCF (transient memory or file-system trouble) is a *retryable* task error, which `run` retries up to `execution.max_attempts` (§7.6). Every other failure is non-retryable (exit 4). §9 lists the messages.

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

**`provenance.json`** [IMPLEMENTED]. Every `gqteashmd run` appends one record once the controller holds the project lock and has opened `state.db` (one run is one record), so the file shows the software behind every run of the project. A run that is later refused because a stage is stale (`StaleStageError`, §7.5) has a record; a run refused by the preflight checks, by the lock, or by a corrupt `state.db` has none. Open it with any text editor, or with `python -m json.tool .gqteashmd/provenance.json`.

```json
{"schema_version": 1, "runs": [ { "started_at": "…", "gqteashmd": "…", "…": "…" } ]}
```

| Field | Meaning |
|---|---|
| `started_at` | Start of the run, UTC, ISO 8601 |
| `gqteashmd` | Program version |
| `git_commit` | Commit of the source tree the program runs from; `null` unless `gqteashmd` runs from a git checkout (an editable install, `pip install -e .`) and git is available |
| `git_dirty` | `true` if that source tree has uncommitted changes; `null` if unknown |
| `python` | Python version |
| `platform` | Operating system and kernel |
| `hostname` | Name of the machine |
| `cpu_count` | CPU count reported by the operating system |
| `cpu_model` | CPU model name from `/proc/cpuinfo`; `null` if it cannot be read |
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
- A stage that was done but whose results directory you deleted by hand is **computed again**, with a warning (`results directory … is missing; recomputing`).
- Temporary files (`.…tmp`) left in `.gqteashmd/` by a crashed controller are removed when the next run starts.
- Tasks that were running when the controller died are checked. Finished ones are kept. Left-over processes are terminated and the tasks are run again.

### 7.4 Stopping and resuming
`gqteashmd stop input.toml`, or Ctrl-C in the terminal running the controller, does the following:
1. No new tasks start.
2. Running tasks get SIGTERM, then SIGKILL after `stop_grace_period_s`.
3. The run ends with exit code 5 and the message `Stopped. Run the same command again to resume.`

Ctrl-C at any other moment (for example while the command is starting) prints `gqteashmd: interrupted` and also exits with code 5.

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

The lock is taken before anything else is created. A run that the lock refuses therefore creates nothing except the `.gqteashmd/` directory (for the lock file), and writes no log and no provenance record. A worker's whole process group is killed when it is stopped or when the worker ends, so no stray child process survives.

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
| `project.directory` exists but is not a directory | `project.directory … exists but is not a directory.` | Choose another path or remove the file |
| The project directory cannot be created or written (the nearest existing parent folder is not a directory or is not writable) | `Cannot create or write the project directory … (no write permission in <parent>).` | Choose a directory you may write to |
| Free disk space below `execution.min_free_disk_mb` | `Only N MB are free on the disk holding …; at least M MB are required (execution.min_free_disk_mb).` | Free space, choose another disk, or lower the limit |
| The free space cannot be determined | `Cannot determine free disk space for …` | Check that the path is on a readable disk |
| An enabled stage needs a Python module that is not installed | `Stage '…' needs the Python module '…', which is not installed in this environment.` | Install the module in the active conda environment |

The disk is the one that holds the project directory, or its nearest existing parent folder if the directory does not exist yet. The location check looks at the path text. A relative path is resolved against the input file's directory with symbolic links followed, so it is judged by its target; an absolute path is taken as written, and a symbolic link in it is not followed (§10).

**Warnings** (the run continues; they appear on the console and in `logs/gqteashmd.log`):

```text
2026-11-06 21:03:44 WARNING gqteashmd.cli: Preflight: resources.total_cores = 64 exceeds the 16 cores detected on this machine; tasks may run slower than expected.
2026-11-06 21:03:44 WARNING gqteashmd.cli: Preflight: resources.total_memory_mb = 16000 exceeds the 7754 MB of RAM detected on this machine; tasks may be killed for lack of memory.
```

These two warnings appear when the values you entered exceed the cores or the RAM the program detects. The program does not stop, because it trusts the values you give, for example on a machine whose detection is wrong or that is shared.

---

## 8. Computational methods and formulas

This chapter defines every quantity the program computes or will compute.
- **§8.1–§8.6** describe **what the current version does**.
- **§8.7–§8.13** describe the **planned** methods. Those formulas document the intended implementation; they are not available yet, and details may change after validation against SHARC 4.1 and the literature.

### 8.1 Units and constants
**[IMPLEMENTED]** for input and output:
- Coordinates in XYZ files are in **Ångström (Å)**.
- Charges are in units of the elementary charge $e$.
- Times in `[execution]` are in seconds.

**[IMPLEMENTED]** in the electronic-structure layer (§8.6): coordinates are converted from Å to bohr once, when the XYZ file is read, using the CODATA 2018 values below. Energies are in $E_h$, gradients in $E_h/a_0$ and transition dipoles in $e\,a_0$; the only conversion on output is $E_h \to$ eV for the excitation energies.

**[PLANNED]** for the other calculations:
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

A fully loaded stage runs $n_p$ tasks at once, with $n_t$ threads each, so it uses at most $n_p n_t$ cores. $n_p m$ is the memory *budget* that the rule checks: PySCF's `max_memory` is a soft target and real use can exceed it, so leave headroom. The input file is accepted only if

$$
n_p\, n_t \;\le\; N_{\text{cores}},
\qquad
n_p\, m \;\le\; M_{\text{total}} .
$$

- **Where it applies:** to the global `[resources]` values **and** to each `[resources.stages.<name>]` override. For an override, $n_p$, $n_t$ and $m$ are the *merged* values: a key that the override leaves unset takes the global value (§4.2).
- **What happens on a violation:** the input is rejected (exit code 2) with a message that names the section and the numbers, for example (the console shows the first line, then the second):

  ```text
  gqteashmd: error: Invalid input file /home/you/input.toml:
    - resources: Value error, [resources]: max_parallel_tasks × threads_per_task = 3 × 2 = 6 exceeds total_cores = 4
  ```

  The memory rule reads `… max_parallel_tasks × memory_mb_per_task = 3 × 2000 = 6000 MB exceeds total_memory_mb = 4000`.
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

> **§8.6 is [IMPLEMENTED].** **Everything below it (§8.7–§8.13) is [PLANNED].** The stages, keys and outputs there do not exist in the current version.

### 8.6 Electronic structure (PySCF)
**[IMPLEMENTED, P3a]** Energies, analytic gradients and transition dipoles. **[IMPLEMENTED, P3b, library only]** SA-CASSCF nonadiabatic couplings, wavefunction overlaps with phase tracking, `compute_more`, restarts (§8.6.8). **[PLANNED]** use of couplings and overlaps in dynamics (P7), TDDFT couplings, triplets and spin–orbit coupling, the Hessian (P4).

**What is computed.** For a geometry $\mathbf R$ (nuclear coordinates, bohr) and the states $I = S_0, S_1, \dots$ requested by `electronic.singlets`:

| Quantity | Symbol | Unit | Status |
|---|---|---|---|
| Adiabatic energy | $E_I(\mathbf R)$ | $E_h$ | **[IMPLEMENTED]** |
| Excitation energy | $\Delta E_{0I} = E_I - E_0$ | $E_h$ (eV on output) | **[IMPLEMENTED]** |
| Gradient | $\nabla_{\mathbf R} E_I$, a $3N_{\text{at}}$-vector | $E_h/a_0$ | **[IMPLEMENTED]** (backend; `single-point` does not request it) |
| Transition dipole | $\boldsymbol\mu_{0I}$ | $e\,a_0$ | **[IMPLEMENTED]** |
| Oscillator strength | $f_{0I}$ | — | **[IMPLEMENTED]** |
| Nonadiabatic couplings $\mathbf d_{IJ} = \langle \Psi_I \vert \nabla_{\mathbf R} \Psi_J \rangle$ (CASSCF); wavefunction overlaps $S_{IJ}(t) = \langle \Psi_I(t-\Delta t) \vert \Psi_J(t) \rangle$ (TDA/TDDFT) | $\mathbf d_{IJ}$, $S_{IJ}$ | $a_0^{-1}$, — | **[IMPLEMENTED, P3b]** $\mathbf d_{IJ}$ for CASSCF only; $S_{IJ}$ for all methods; library only, not reachable from the CLI (§8.6.8) |
| Hessian $\partial^2 E_0/\partial x_i\partial x_j$ | $H_{ij}$ | $E_h/a_0^2$ | [PLANNED, P4] |

**Assumptions that apply to everything in §8.6:**
- **Closed-shell singlet molecules only.** The reference is a restricted determinant (RHF or RKS), so `molecule.multiplicity` must be 1. Open-shell references and radicals are refused (§9).
- **Singlet excited states only.** `electronic.triplets` must be 0. S→T transitions and intersystem crossing are not available yet.
- **Born–Oppenheimer, nonrelativistic, all-electron**, in the Gaussian basis `electronic.basis`; no solvent, no external field.
- **State ordering** is by energy within the computed roots. The labels $S_I$ refer to that order at each geometry; following a state through a crossing uses the overlaps and phase tracking of §8.6.8 (library only for now).

#### 8.6.1 The SCF reference and its convergence ladder
Every method starts from a self-consistent-field (SCF) calculation on a single closed-shell determinant $\Phi_0$ of doubly occupied orbitals:
- Hartree–Fock (RHF) for `casscf` and for `functional = "hf"`, with energy $E_{\text{SCF}} = \langle\Phi_0\vert\hat H\vert\Phi_0\rangle$;
- Kohn–Sham DFT (RKS) for any other functional, with $E_{\text{SCF}} = E[\rho; \text{functional}]$ evaluated on a numerical grid of level `grid_level`.

$E_{\text{SCF}}$ is reported as `meta.scf_energy_hartree` (in $E_h$). The SCF is converged when the energy change between iterations is below `scf_conv_tol` (and PySCF's orbital-gradient criterion, derived from it, is met) within `scf_max_cycle` iterations.

**Convergence ladder.** If the SCF does not converge, the program tries the next rung; it never silently relaxes a threshold:

| Rung | Settings | Reported as `meta.scf` |
|---|---|---|
| 1 | PySCF defaults (DIIS); if the same backend object has already computed another geometry, the starting density is that geometry's converged density | `default` |
| 2 | Level shift $\sigma = 0.2\,E_h$ (virtual-orbital energies raised by $\sigma$) and damping factor 0.5 (the new Fock matrix is mixed 1:1 with the previous one). PySCF damps only the cycles before DIIS starts, so DIIS is delayed to cycle 8: damping acts in the first cycles, DIIS finishes the convergence | `level shift + damping` |
| 3 | Second-order (Newton) orbital optimization | `second-order` |

Only rung 1 uses the previous geometry's density; rungs 2 and 3 start from PySCF's default guess, because a stale guess is the most likely reason rung 1 failed. If all three rungs fail, the calculation stops with *SCF did not converge (tried: …)* (exit code 4, not retried). An exception inside PySCF (as opposed to non-convergence) stops immediately, because another rung would not fix a broken setup.

#### 8.6.2 State-averaged CASSCF (`method = "casscf"`)
In a complete-active-space calculation the $n_{\text{core}} = (N_e - n_{e,\text{cas}})/2$ lowest orbitals are doubly occupied and frozen in the configuration space, and the $n_{e,\text{cas}}$ active electrons are distributed in all possible ways over $n_{\text{cas}}$ active orbitals (full CI in the active space; "CAS($n_{e,\text{cas}}$, $n_{\text{cas}}$)"). Here $N_e$ is the total electron count (§8.2), and $n_{e,\text{cas}}$, $n_{\text{cas}}$ are `nelecas`, `ncas`.

For $N$ = `singlets` states, the orbitals $\{\phi_p\}$ and the CI vectors $\{\Psi_I\}$ are optimized together to minimize the **state-averaged energy**

$$
E_{\mathrm{SA}} \;=\; \sum_{I=0}^{N-1} w_I\, E_I,
\qquad
E_I = \langle \Psi_I \vert \hat H \vert \Psi_I \rangle,
\qquad
w_I = \frac{1}{N} .
$$

| Symbol | Meaning | Unit |
|---|---|---|
| $\Psi_I$ | CI wavefunction of state $I$ (orthonormal for different $I$), built from the shared orbitals | — |
| $E_I$ | Energy of state $I$, **reported** as the state's energy | $E_h$ |
| $w_I$ | Averaging weight; equal for all states | — |
| $N$ | Number of states in the average (`electronic.singlets`, including S0) | — |
| $E_{\mathrm{SA}}$ | The minimized quantity; **not** reported as a state energy | $E_h$ |

The common orbital set gives smooth, balanced surfaces for S0 and S1, which is why SA-CASSCF is the recommended method for S1→S0 dynamics (§8.6.4). The reported $E_I$ are the individual state energies, not $E_{\mathrm{SA}}$. Convergence is controlled by `casscf.conv_tol` (change of $E_{\mathrm{SA}}$) and `casscf.conv_tol_grad` (orbital gradient).

- **Singlet-only CI.** PySCF's default CI solver also returns triplet and quintet roots of the active space. The program uses the singlet-restricted solver, which removes the triplets but **not higher spins**: in larger active spaces a quintet root ($S = 2$) can still appear among the lowest roots. That is why the spin check below exists.
- **Spin check.** After convergence, the spin expectation value of every root is computed. A root is accepted only if $\lvert\langle S^2\rangle_I\rvert \le 10^{-3}$; otherwise the run stops (*CASSCF root S1 has <S^2> = …; it is not a singlet*). A singlet has $\langle S^2\rangle = S(S+1) = 0$. The value is stored as `spin_square`.
- **Orbital reuse by projection.** A backend object that computes several geometries one after another (as the dynamics stage will) starts each CASSCF from the previous geometry's converged orbitals. Those orbitals belong to the old nuclear positions, so they are not orthonormal in the new basis; they are first projected onto the new geometry's atomic-orbital basis (PySCF `project_init_guess`) and re-orthonormalized. The first calculation of a backend object, and every calculation by a new object, starts from the SCF orbitals. The reuse changes the starting point, not the converged result.
- **Active space.** The program stops with a message if the active electrons do not fit in the molecule, or if the basis has fewer orbitals than the core plus active orbitals require (§9).

#### 8.6.3 Linear response: TDA, TDDFT, CIS, TDHF
For `method = "tda"` or `"tddft"` the excited states come from the response of the SCF reference to a time-dependent perturbation. The singlet excitation energies $\omega_k$ ($E_h$) solve the Casida equation

$$
\begin{pmatrix} \mathbf A & \mathbf B \\ -\mathbf B & -\mathbf A \end{pmatrix}
\begin{pmatrix} \mathbf X_k \\ \mathbf Y_k \end{pmatrix}
= \omega_k
\begin{pmatrix} \mathbf X_k \\ \mathbf Y_k \end{pmatrix},
\qquad
E_k = E_{\mathrm{SCF}} + \omega_k ,
$$

with, for an occupied orbital pair $i, j$ and virtual pair $a, b$ of the reference,

$$
A_{ia,jb} = \delta_{ij}\delta_{ab}(\varepsilon_a - \varepsilon_i) + 2(ia\vert jb) - c_x\,(ij\vert ab) + (ia\vert f_{xc}\vert jb) .
$$

| Symbol | Meaning | Unit |
|---|---|---|
| $\varepsilon_p$ | Orbital energy of orbital $p$ | $E_h$ |
| $(pq\vert rs)$ | Two-electron repulsion integral | $E_h$ |
| $c_x$ | Fraction of exact exchange of the functional (1 for Hartree–Fock) | — |
| $f_{xc}$ | Exchange–correlation kernel (zero for Hartree–Fock) | — |
| $\mathbf X_k, \mathbf Y_k$ | Excitation and de-excitation amplitudes of root $k$ (spin-adapted, singlet) | — |
| $\omega_k$ | Excitation energy of root $k$ | $E_h$ |
| $E_k$ | Total energy of the state; the labels $S_1, S_2, \dots$ are assigned to $k = 1, 2, \dots$ in order of increasing $\omega_k$ | $E_h$ |

$\mathbf B$ is the coupling between excitations and de-excitations; $\mathbf A$ has the form above and $\mathbf B$ is analogous (with the corresponding $(ib\vert ja)$ terms).
- **TDA** (Tamm–Dancoff approximation) sets $\mathbf B = 0$, so $\mathbf A\mathbf X_k = \omega_k\mathbf X_k$ and $\mathbf Y_k = 0$.
- **TDDFT** (random-phase approximation) keeps $\mathbf B$.
- **`functional = "hf"`:** the reference is Hartree–Fock and $f_{xc} = 0$, $c_x = 1$. Then `tda` is **CIS** (configuration interaction singles) and `tddft` is **TDHF**.
- The ground state S0 is the SCF state itself, $E_0 = E_{\mathrm{SCF}}$ ($\omega_0 = 0$). The iterative solver stops when its residual is below `td_conv_tol`; failure to converge all requested roots stops the run (§9).

#### 8.6.4 Analytic gradients
$\nabla_{\mathbf R} E_I$ is computed analytically (not by finite differences): PySCF's SCF gradient for S0 of the linear-response methods, the excited-state response (Z-vector) gradient for $S_{k>0}$ of TDA/TDDFT, and the state-specific SA-CASSCF gradient (including the response of the state-averaged orbitals). The gradient is a vector of $3N_{\text{at}}$ components in $E_h/a_0$; the force on the nuclei is $-\nabla_{\mathbf R} E_I$.

**Assumptions and limits:**
- **DFT gradients omit the integration-grid response.** PySCF's analytic DFT and TDDFT gradients treat the numerical grid as fixed, so they are not the exact derivative of the energy computed on the grid. The error shrinks with `grid_level`. In the validation of §8.6.7 (TDA-B3LYP/STO-3G) the maximum deviation from a finite-difference derivative was $7.6\times10^{-6}\,E_h/a_0$ at the default `grid_level = 3` and $5.7\times10^{-8}\,E_h/a_0$ at `grid_level = 5`. For dynamics, where the gradient drives the nuclei and energy conservation matters, **use `grid_level` ≥ 5**. The effect does not exist for `casscf` or `functional = "hf"`.
- **TDDFT topology.** Linear-response TDDFT (and TDA) with a ground-state reference does not describe the S₁/S₀ crossing seam correctly: the S₀–S₁ intersection has the wrong dimensionality, because the reference-state S₀ is not coupled to the excited states. Use **SA-CASSCF** (or another multi-reference method) when S₁→S₀ decay matters.

#### 8.6.5 Transition dipoles
The (electronic) transition dipole from S0 to state $I$ is

$$
\boldsymbol\mu_{0I} = \langle \Psi_0 \vert\, \hat{\boldsymbol\mu}\, \vert \Psi_I \rangle ,
\qquad
\hat{\boldsymbol\mu} = -\sum_{i=1}^{N_e} \mathbf r_i ,
$$

in $e\,a_0$. The nuclear part of the dipole operator does not contribute, because $\langle\Psi_0\vert\Psi_I\rangle = 0$, so $\boldsymbol\mu_{0I}$ does not depend on the origin of the coordinate system.

**CASSCF.** Using the one-electron transition density matrix $\gamma^{0I}_{pq} = \langle\Psi_0\vert \hat E_{pq}\vert\Psi_I\rangle$ with $\hat E_{pq} = \sum_\sigma a^\dagger_{p\sigma} a_{q\sigma}$ (active orbitals $p, q$; the doubly occupied core orbitals cancel),

$$
\boldsymbol\mu_{0I} = -\sum_{pq} \mathbf r_{pq}\,\gamma^{0I}_{pq},
\qquad
\mathbf r_{pq} = \langle\phi_p\vert\,\mathbf r\,\vert\phi_q\rangle .
$$

| Symbol | Meaning | Unit |
|---|---|---|
| $\phi_p$ | Active molecular orbital | — |
| $\mathbf r_{pq}$ | Position-operator (length) integral between orbitals $p$ and $q$ | $a_0$ |
| $\gamma^{0I}_{pq}$ | Transition density matrix between the CI vectors of S0 and state $I$ | — |

**Linear response (TDA/TDDFT/CIS/TDHF).** With the amplitudes of §8.6.3 (spatial orbitals $i$ occupied, $a$ virtual)

$$
\boldsymbol\mu_{0k} = \sqrt{2}\,\sum_{ia} \mathbf r_{ia}\,(X_{k,ia} + Y_{k,ia}) ,
$$

as returned by PySCF `td.transition_dipole()` (length gauge) and used without any change of sign. The factor $\sqrt2$ comes from the sum over the two spin functions in the closed-shell singlet, and $Y = 0$ for TDA, CIS. The amplitudes are normalized as $\sum_{ia}(X_{ia}^2 - Y_{ia}^2) = 1$.

**The sign is arbitrary.** The sign of the linear-response dipole, like that of the CASSCF one, is whatever PySCF returns and carries no meaning. The CI vectors (and the response vectors) are defined only up to an overall sign, $\Psi_I \to -\Psi_I$, which flips $\boldsymbol\mu_{0I}$. Two programs, or two geometries, can therefore give opposite signs for the same physical state. Only $\lvert\boldsymbol\mu_{0I}\rvert$, $f_{0I}$ and quantities that are even in the sign are meaningful. The validation (§8.6.7) compares dipoles up to a global sign.

#### 8.6.6 Oscillator strength
The dimensionless oscillator strength of the transition $S_0\to S_I$ (length gauge) is

$$
f_{0I} = \frac{2}{3}\,\Delta E_{0I}\,\big\lvert\boldsymbol\mu_{0I}\big\rvert^2
\qquad \text{(atomic units)},
$$

where $\Delta E_{0I} = E_I - E_0$ is in $E_h$ and $\boldsymbol\mu_{0I}$ in $e\,a_0$. It is printed by `single-point` and stored as `oscillator_strength`. It does not depend on the sign of $\boldsymbol\mu_{0I}$. Example: for ethylene SA2-CASSCF(2,2)/6-31G, $\Delta E_{01} = 10.5361$ eV and $f_{01} = 0.6206$ (§5.3). The use of $f$ to build absorption spectra and to choose initial states is planned (§8.9).

#### 8.6.7 Validation
**Against SHARC 4.1.** Ethylene (`tests/data/ethylene.xyz`), SA2-CASSCF(2,2)/6-31G\*, singlet-only solver. SHARC 4.1's PySCF interface (`SHARC_PYSCF.py`) was *executed* as a black box with `fix-spin-shift 0.2`, `conv-tol 1e-10` and `conv-tol-grad 1e-6`; nothing of SHARC is part of `gqteashmd`. SHARC's output is stored in `tests/data/reference/`, and a fresh SHARC run reproduces it exactly (`pytest -m sharc`, needs a SHARC installation).

| Quantity | `gqteashmd` | SHARC 4.1 | Largest difference |
|---|---|---|---|
| $E_0$ (S0) | −78.04902442 $E_h$ | −78.04902443 $E_h$ | $1.4\times10^{-8}\,E_h$ (largest over both states) |
| $E_1$ (S1) | −77.67250837 $E_h$ | −77.67250836 $E_h$ | |
| Gradients of S0 and S1 | — | — | $1.4\times10^{-8}\,E_h/a_0$ |
| Transition dipole $\boldsymbol\mu_{01}$ (≈ (+1.530, +0.054, 0) $e\,a_0$ here, (−1.530, −0.054, 0) in SHARC) | — | — | $4.5\times10^{-7}\,e\,a_0$, after one global sign flip |
| $\langle S^2\rangle$ of both roots | < $10^{-6}$ | — | — |

The numbers in the `gqteashmd` column were obtained with the program itself at these settings (`method = "casscf"`, `basis = "6-31g*"`, `singlets = 2`, CAS(2,2), `tests/data/ethylene.xyz`). The sign of $\boldsymbol\mu_{01}$ in `gqteashmd` is opposite to SHARC's, which is allowed (§8.6.5). The automated tests use looser tolerances than the measured differences: $10^{-6}\,E_h$ for energies, $10^{-5}\,E_h/a_0$ for gradients and $10^{-4}\,e\,a_0$ for the dipole (up to sign).

**Against finite differences** (slow tests; central differences with step $h = 10^{-4}$ bohr on each Cartesian component, a fresh backend for every displaced geometry so that no orbital reuse can bias the reference):

| Method | Check | Largest difference between analytic and finite-difference gradient | Tolerance |
|---|---|---|---|
| SA-CASSCF(2,2)/6-31G, ethylene | S0 and S1, three components | $1.8\times10^{-7}\,E_h/a_0$ | $10^{-6}$ |
| TDA-B3LYP/STO-3G, `grid_level = 5`, `td_conv_tol = 1e-9` | S1, two components | $5.7\times10^{-8}\,E_h/a_0$ | $5\times10^{-6}$ |
| TDA-B3LYP/STO-3G, `grid_level = 3` (default) | S1, two components | $7.6\times10^{-6}\,E_h/a_0$ | — (grid response, §8.6.4) |

**Couplings and overlaps (P3b).** Ethylene, SA2-CASSCF(2,2). The finite-difference rows were measured with step `h` = $10^{-4}$ bohr, `conv_tol` = $10^{-11}$, `conv_tol_grad` = $10^{-7}$ (6-31G, one backend snapshot).

| Check | Result | Tolerance / comment |
|---|---|---|
| $\mathbf d_{01}$ against SHARC 4.1 (6-31G\*, executed as a black box), up to one global sign | largest difference $9.1\times10^{-8}\,a_0^{-1}$ | SHARC also uses the full convention: `use_etfs=True` (no CSF term) differs by 0.27 |
| $\mathbf d_{01}$ against central finite differences of the program's own overlaps: atom C1, x | $0.2566853$ (analytic) vs $0.2567711$ (FD), difference $8.6\times10^{-5}$ | constant offset from the AO-derivative convention (§8.6.8) |
| same: atom H3, y | $-6.27706\times10^{-3}$ vs $-6.31245\times10^{-3}$, difference $3.5\times10^{-5}$ | |
| same: atom H5, x | $-9.53438\times10^{-3}$ vs $-9.58974\times10^{-3}$, difference $5.5\times10^{-5}$ | |
| Overlaps, random 0.01 bohr step on all atoms, CASSCF | diagonal $0.9952$, $0.9950$; smallest singular value $0.9950$; antisymmetry residual of the off-diagonal $6.4\times10^{-5}$ | |
| Overlaps, same step, TDA-B3LYP/STO-3G, 3 excited states | diagonal $0.9953$, $0.9927$, $0.9926$; smallest singular value $0.9952$; antisymmetry residual $4.8\times10^{-4}$ | |
| Overlaps at identical geometries | identity to $10^{-6}$ | |
| Orientation: step of $10^{-3}$ bohr along $\mathbf d_{01}$ | $S_{01} \approx \mathbf d_{01}\cdot\Delta\mathbf R$ within 10 %, $S_{10}$ of opposite sign | confirms the index order and the sign convention |
| Restart from `save_state`/`load_state` (same and new process) | phased energies, gradients, NACs, dipoles and overlaps agree with the continued run to $\le 2\times10^{-11}$ | |

The finite-difference NAC test is a slow test (15 s); the NAC-against-SHARC test is part of the live SHARC test (`pytest -m sharc`, about 8 s).

**Linear response against direct PySCF calls:** energies, excitation energies and gradients agree to $10^{-13}$ or better. Bright-state transition dipoles and oscillator strengths were checked on small test molecules (TDA-B3LYP/STO-3G S3: $\lvert\boldsymbol\mu\rvert = 1.54$, $f = 0.68$; CIS S1: $f = 0.72$; TDHF S1: $f = 0.50$).

#### 8.6.8 Nonadiabatic couplings, wavefunction overlaps, restarts (P3b)
**[IMPLEMENTED, P3b]** as a Python library (`gqteashmd.electronic.pyscf_backend.PySCFBackend`, built by `gqteashmd.electronic.factory.make_backend`; `gqteashmd.overlaps.determinants` and `gqteashmd.overlaps.tracking`; the packages re-export nothing). **Not reachable from the command line yet:** `single-point` does not request couplings or overlaps, and no workflow stage uses them. They exist for the future dynamics stage (P7). **[PLANNED]** their use in dynamics (local diabatization, norm-preserving interpolation, hopping, P7), triplet states and spin–orbit coupling, and TDA/TDDFT couplings.

**Capabilities.** The backend reports what it can do: gradients, overlaps and transition dipoles for all methods; analytic NACs for `casscf` only. A request for NACs with `tda` or `tddft` is refused with *Nonadiabatic coupling vectors are not available for method = '…'; request overlaps instead* (PySCF 2.14 has no analytic TDA/TDDFT couplings).

**Nonadiabatic coupling vectors (SA-CASSCF).**

$$
\mathbf d_{IJ} = \langle \Psi_I \vert \nabla_{\mathbf R} \Psi_J \rangle ,
\qquad \mathbf d_{JI} = -\mathbf d_{IJ} .
$$

| Symbol | Meaning | Unit |
|---|---|---|
| $\mathbf d_{IJ}$ | Coupling vector between states $I$ and $J$, a $3N_{\text{at}}$-vector | $a_0^{-1}$ |
| $\Psi_I$ | State-averaged CASSCF wavefunction of state $I$ | — |
| $\mathbf R$ | Nuclear coordinates | $a_0$ |

The derivative acts on both the CI vector and the orbitals, so $\mathbf d_{IJ}$ includes the configuration-state-function (CSF) term (PySCF's default `nac_method()`; the variant without it, `use_etfs=True`, differs by 0.27 $a_0^{-1}$ on ethylene and is not used). Pairs are requested with $I < J$; the result for $I > J$ is the negative. The overall sign of each state is arbitrary (§8.6.5); the program fixes it by phase tracking (below).

*AO-derivative convention.* The analytic CSF term uses the antisymmetrized basis-function derivative $\tfrac12(\langle\chi\vert\partial\chi\rangle - \langle\partial\chi\vert\chi\rangle)$, whereas a finite difference of overlaps differentiates only the ket basis functions. The two differ by $\tfrac12\sum_{\mu\nu}\gamma^{IJ}_{\mu\nu}\,\partial S_{\mu\nu}/\partial\mathbf R$ ($\gamma^{IJ}$: transition density matrix in the AO basis, $S_{\mu\nu}$: AO overlap). This is a roughly constant offset of $4$–$9\times10^{-5}\,a_0^{-1}$ in the validation of §8.6.7, not a numerical error.

**Wavefunction overlaps.** Between the previous geometry $\mathbf R_{\text{prev}}$ and the current one $\mathbf R_{\text{cur}}$,

$$
S_{IJ} = \langle \Psi_I(\mathbf R_{\text{prev}}) \vert \Psi_J(\mathbf R_{\text{cur}}) \rangle
= \sum_{k,l} c^{\text{prev}}_{Ik}\, c^{\text{cur}}_{Jl}\,
\det\!\big(\mathbf s^{\alpha}_{kl}\big)\,\det\!\big(\mathbf s^{\beta}_{kl}\big),
\qquad
\mathbf s = \mathbf C_{\text{prev}}^{T}\,\mathbf S_{\text{AO}}(\mathbf R_{\text{prev}},\mathbf R_{\text{cur}})\,\mathbf C_{\text{cur}} .
$$

| Symbol | Meaning | Unit |
|---|---|---|
| $k, l$ | Slater determinants of the previous and current expansions | — |
| $c^{\text{prev}}_{Ik}$, $c^{\text{cur}}_{Jl}$ | Coefficient of determinant $k$ ($l$) in state $I$ ($J$) | — |
| $\mathbf s^{\alpha}_{kl}$, $\mathbf s^{\beta}_{kl}$ | Sub-matrix of $\mathbf s$ with the rows of the occupied $\alpha$ ($\beta$) orbitals of $k$ and the columns of those of $l$ (Löwdin's rule) | — |
| $\mathbf C_{\text{prev}}$, $\mathbf C_{\text{cur}}$ | Molecular-orbital coefficients at the two geometries (AO × MO) | — |
| $\mathbf S_{\text{AO}}(\mathbf R_{\text{prev}},\mathbf R_{\text{cur}})$ | AO overlap between the basis at the previous geometry (rows) and the current one (columns) | — |

The first index of $S_{IJ}$ is the previous geometry. Determinants are evaluated once per distinct $\alpha$ string and per distinct $\beta$ string (in memory-bounded batches). The matrix of determinant-pair overlaps is never formed: each state's coefficients are placed on a (distinct $\alpha$ strings) $\times$ (distinct $\beta$ strings) matrix, which the two determinant tables multiply from both sides. The memory is therefore proportional to $N_\alpha N_\beta$ per state ($N_\alpha$, $N_\beta$: numbers of distinct strings), not to the square of the number of determinants. Before it allocates, the overlap routine estimates its memory; above 2 GiB it stops with *The wavefunction overlap would exceed the memory limit … lower overlap_norm_kept or the active space*.

*Truncation.* For each state, the largest-coefficient determinants are collected until they hold at least the fraction `overlap_norm_kept` of that state's squared norm. The **union** of these sets over all states is kept, and every state keeps all its own coefficients on that union (small coefficients shared between states carry the off-diagonal overlaps, i.e. the couplings; cutting each state separately was found to destroy them). Each state is then renormalized. `overlap_norm_kept = 1.0` keeps every determinant that is nonzero in any state. The cost of the overlap grows with the number of determinants kept.

*Expansions.*
- **CASSCF:** the doubly occupied core orbitals, followed by the active orbitals of each full-CI $\alpha$ string and $\beta$ string (the CI vector is the coefficient matrix over the string pairs). Checked against PySCF's CI-vector transformation (an orbital rotation) to $10^{-10}$ in the tests.
- **TDA (and CIS):** the SCF determinant $\Phi_0$ for S0, and for an excited state $k$
  $\Psi_k = \sum_{ia} X_{k,ia}\,(\lvert i\to a\,\alpha\rangle + \lvert i\to a\,\beta\rangle)$ with $\sum_{ia}X_{k,ia}^2 = \tfrac12$, so that $\langle\Psi_k\vert\Psi_k\rangle = 1$ (checked against an explicit orbital-rotation formula to $10^{-12}$ in the tests).
- **TDDFT, TDHF:** the same expression with the excitation amplitudes $\mathbf X$ only; the de-excitation amplitudes $\mathbf Y$ are dropped and the state renormalized. **This is an approximation.** For ethylene/STO-3G TDHF S1, $\sum Y^2 \approx 1.1\times10^{-2}$, so the $X$-only norm is $\approx 1 + 2\sum Y^2 \approx 1.023$ before renormalization, and different TDDFT states are no longer exactly orthogonal. TDA and CIS do not have this problem.

**Phase alignment and root-flip matching.** The wavefunction signs are arbitrary, so the overlap matrix is processed: states of the current geometry are matched to those of the previous one by the assignment $a$ that maximizes $\sum_j \lvert S_{a(j)j}\rvert$ (Hungarian algorithm), and the phase of current state $j$ is $p_j = \operatorname{sign} S_{a(j)j}$ ($+1$ if the overlap is 0). A matched pair with $a(j)\neq j$ is reported as a **root flip** (two states exchanged energy order); the pairs are listed in `meta.root_flips`. The reported overlap matrix has its columns multiplied by $p_j$. The **smallest singular value** of $\mathbf S$ is stored as `meta.overlap_min_singular_value`; for a small step it is close to 1, and below 0.5 a warning is logged (an intruder state, or too few computed states). Löwdin orthonormalization $\mathbf U\mathbf V^T$ (from $\mathbf S=\mathbf U\boldsymbol\Sigma\mathbf V^T$) is available as a library function for the dynamics (P7); the backend does not apply it.

*Effect of the phases on the outputs.* The phases are applied to the stored wavefunction (so the sign frame is carried from step to step) and to the outputs:

$$
\mathbf d_{IJ} \to p_I\,p_J\,\mathbf d_{IJ},
\qquad
\boldsymbol\mu_{0I} \to p_0\,p_I\,\boldsymbol\mu_{0I}.
$$

Whenever a previous geometry exists, the phases are **always** tracked and applied, whether or not the caller asks for overlaps; the request flag only decides whether `overlaps` and `phases` are returned (`meta.overlaps` is `computed`, `tracked, not returned` or `no previous geometry`). The first geometry of a run fixes an arbitrary sign frame: PySCF's raw CI signs are **not reproducible** between processes, but phased results are reproducible relative to a saved snapshot. Energies and gradients do not depend on the phases.

**Noise of overlap-based couplings.** A time-derivative coupling from overlaps, $\sigma_{IJ}\approx (S_{IJ}-S_{JI})/(2\Delta t)$, carries the noise of $S$ divided by $\Delta t$. At the default CASSCF convergence (`conv_tol_grad = 1e-6`) the overlap noise is of order $10^{-7}$, so the noise in $\sigma$ is of order $10^{-7}/\Delta t$. In the finite-difference test the error of the overlap-based derivative grew when the step $h$ shrank (atom 0: $1.2\times10^{-3}$ at $h=10^{-4}$ bohr). Tighter `casscf.conv_tol_grad` reduces it.

**Retry ladders.** Each rung is tried in order; the rung that converged is stored in the result `meta`.

| Method | Rung | Settings | `meta` |
|---|---|---|---|
| `casscf` | 1 | Orbitals of the previous geometry, projected (skipped if there is none) | `casscf = "previous orbitals"` |
| | 2 | SCF orbitals | `"SCF orbitals"` |
| | 3 | SCF orbitals plus a spin penalty (`fix_spin_`, $\langle S^2\rangle=0$, shift 0.5) and 3× the macro iterations | `"SCF orbitals + spin penalty"` |
| `tda`, `tddft` | 1 | PySCF defaults | `td = "default"` |
| | 2 | $N_{\text{states}}+3$ roots solved (the lowest kept) and 3× `max_cycle` | `"extra roots"` |

A CASSCF rung fails if it does not converge or if a root has $\lvert\langle S^2\rangle\rvert > 10^{-3}$; a PySCF exception stops the ladder. On ethylene the spin-penalty rung reproduces the plain rung (energies within $10^{-6}\,E_h$, S1 gradient within $10^{-5}\,E_h/a_0$). The number of linear-response roots is limited to the number of single excitations; asking for more is an error. If all rungs fail the run stops (exit code 4, §9).

**`compute_more(gradients, nacs)`.** After a `compute()` at a geometry, further gradients and NACs of that same geometry can be requested without repeating the SCF, CASSCF or TD calculation: the PySCF objects of the last successful `compute()` are reused and the same phases applied. It raises an error before any `compute()`, after `load_state()` (the live objects are not stored) and after a `compute()` that failed (a failed `compute()` discards the previous live objects).

**Restarts.** `save_state(path)` writes the minimal cross-geometry state (geometry, orbitals, density, truncated and phase-aligned determinant expansion, and a fingerprint of the method, basis, functional, atoms, charge, multiplicity, states, CASSCF active space (`ncas`, `nelecas`) and `overlap_norm_kept`) to an `.npz` file, atomically; `load_state(path)` reads it into a new backend object, also in a new process. A file written with different settings is refused (the error names the file). A restarted run reproduces the energies, gradients, NACs, dipoles and overlaps of the uninterrupted run to $2\times10^{-11}$ or better. Restarting from an older snapshot is safe for the signs: the snapshot carries the aligned sign frame, so the recomputed steps are aligned to it like the original ones. Continuity is lost only when a state that the next step needs was never saved (a new backend object then draws a new arbitrary sign frame), so save the state at every step you may restart from. A state file that is missing, damaged, of another format version, or has inconsistent array shapes is refused with *… is not a readable gqteashmd electronic state*. The fingerprint holds only the CASSCF active space (`ncas`, `nelecas`), not the CASSCF tolerances, so changing `conv_tol` does not invalidate a saved state.

**One backend per trajectory.** A backend object remembers the previous geometry (orbital guess, projection, state tracking, phases). Call `reset()` (or use a new backend) before computing an unrelated geometry; the next `compute()` is then a first geometry (`meta.overlaps = no previous geometry`).

**Failures of the overlap step.** An error inside the overlap, truncation or tracking step is reported as *The wavefunction overlap failed: …* (not retried); an out-of-memory condition is reported with the memory estimate and is retryable.

**Scratch policy.** No SCF checkpoint file is written. PySCF's temporary directory is pointed at the backend's `scratch_dir` (default: the system temporary directory) during `compute` and `compute_more` and restored afterwards, also after a failure. Put the scratch directory on a local disk, not on OneDrive or `/mnt/<letter>` in WSL. An input-file key for it is [PLANNED].

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
**[PLANNED, P6]** At each sampled geometry, the vertical excitation energies $\Delta E_{0J} = E_J - E_0$ and oscillator strengths $f_{0J}$ will be computed with the formulas of §8.6.5 and §8.6.6 (**[IMPLEMENTED]** there for one geometry; this stage will run them at every sampled geometry).
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
| `Invalid input file …:` followed by `  - resources: Value error, [resources]: max_parallel_tasks × threads_per_task = 3 × 2 = 6 exceeds total_cores = 4` (also `… × memory_mb_per_task = 3 × 2000 = 6000 MB exceeds total_memory_mb = 4000`, and the same for `[resources.stages.<name>]`) | The resource settings do not fit in the machine you described | Lower `max_parallel_tasks`, `threads_per_task` or `memory_mb_per_task`, or raise the totals (§8.4) |
| `[resources.stages] names unknown stage(s): …; stages are: …` | A resource override for a stage that does not exist (also reported by `validate`) | Correct the stage name |
| `workflow.pause_after names unknown stage(s): …; stages are: …` | A pause for a stage that does not exist (also reported by `validate`) | Correct the stage name |
| `Cannot expand '~' in path …` | `~user` names a user that does not exist | Use a real user name or a full path |
| `Invalid geometry file …: line N: coordinates must be finite numbers, got …` | `nan` or `inf` in the XYZ file (reported when `prepare` runs) | Correct the coordinates |
| `Project directory … is on a Windows drive mounted in WSL …` / `… on a OneDrive folder …` | Preflight refusal (§7.9) | Use a local Linux disk |
| `Only N MB are free on the disk holding … ` | Not enough free disk space (§7.9) | Free space or lower `execution.min_free_disk_mb` |
| `Stage '…' needs the Python module '…', which is not installed` | Preflight: a missing module | Install it in the conda environment |
| `Preflight: resources.total_cores = … exceeds the … cores detected` (also for memory) | **Warning**, not an error (§7.9) | Check that the values describe your machine |
| `Paused after stage '…' as requested (workflow.pause_after)` | Not an error: exit code 5 (§7.8) | Review, then run the same command again |
| `File locking is not available for … (…); put the project directory on a local Linux filesystem.` | The filesystem of the project directory does not support file locks (exit 3) | Choose a local Linux disk (§2.3) |
| `The state database … failed during <operation>: …` | An SQLite operation failed while the program ran (disk full, database opened by another program) (exit 3) | Check free disk space and that no other program has the database open, then run the same command again |
| `gqteashmd: interrupted` | Ctrl-C while the command was starting or finishing (exit 5) | Run the same command again |
| `Cannot read task file …` | A task description file was damaged or incomplete; the task fails without retry and the message is in its `result.json` and `worker.log` (exit 4 after the stage fails) | Run the same command again; if it repeats, report it |
| `State database … is unreadable or corrupt: …` | `state.db` is damaged, is not a database, or is a directory. A message `… has schema version …` or `… unknown stage/task status …` means another program version wrote it | Move the whole `.gqteashmd/` directory aside (it holds `state.db` with its `-wal`/`-shm` files) to start the project over; stage results will be recomputed. Or use the matching version |
| `Stage '…' has results produced from different inputs` | The input changed after results existed | Run the suggested `--rerun-from` command (§7.5) |
| `Stage '…' failed: k of n tasks did not succeed` | Task failures | Read the listed `worker.log` files, fix, run again |
| `This calculation needs an [electronic] section in the input file (method, basis, singlets, ...).` | `single-point` without `[electronic]` (exit 2) | Add the section (§4.2) |
| `Invalid input file …:` with `electronic.casscf` / `electronic.functional` / `electronic.triplets` text (for example `method = 'casscf' needs an [electronic.casscf] table with ncas and nelecas`, `electronic.functional is required for method = 'tda' (use 'hf' for CIS/TDHF)`, `electronic.casscf.nelecas must be even for singlet states`, `electronic.triplets > 0 is not supported yet`) | Inconsistent `[electronic]` keys (§4.2) | Correct the keys |
| `electronic.singlets = … exceeds the … singlet states of CAS(…) (nelecas = …, ncas = …)` | More singlet states requested than the active space has (exit 2) | Lower `singlets` or enlarge the active space |
| `the PySCF backend supports closed-shell singlet molecules only (molecule.multiplicity = 1)` | An `[electronic]` section together with a multiplicity other than 1 (exit 2) | Use a closed-shell singlet molecule |
| `Charge … and multiplicity … are inconsistent with … electrons. Check [molecule] in the input file.` | Same parity rule as §8.2, found by the electronic-structure layer (exit 2) | Correct `charge` or `multiplicity` |
| `PySCF is not installed in this environment …` / `PySCF … is too old; gqteashmd needs PySCF >= 2.6.0.` | Missing or old PySCF (exit 2) | Install or update PySCF in the conda environment |
| `PySCF could not build the molecule with basis '…'` | Unknown basis set name or a basis that has no entry for an element (exit 4) | Check `electronic.basis` |
| `SCF did not converge (tried: default, level shift + damping, second-order; …)` / `PySCF SCF (…) failed: …` | The SCF ladder (§8.6.1) is exhausted, or PySCF raised an error (exit 4; retried only for `MemoryError`/`OSError`) | Check the geometry, charge and basis; raise `scf_max_cycle`; read the `--log` file |
| `SA-CASSCF failed on every rung (conv_tol = …, conv_tol_grad = …): previous orbitals: not converged; SCF orbitals: …` / `CASSCF (…) failed: …` | All rungs of the CASSCF ladder (§8.6.8) failed, by non-convergence or a non-singlet root (exit 4) | Choose another active space or geometry; loosen `casscf.conv_tol_grad` slightly |
| `CASSCF root S1 has <S^2> = …; it is not a singlet.` | A root of the active space is not a singlet (exit 4) | Check the active space and the number of states |
| `The CASSCF active space (… electrons) does not fit the molecule's … electrons.` / `The CASSCF active space needs … orbitals, but the basis '…' has only …` | `ncas`/`nelecas` do not fit the molecule or the basis (exit 4) | Reduce the active space or use a larger basis |
| `TDA did not converge for all … excited states (tried: default, extra roots; …)` (also `TDDFT`) | Both rungs of the linear-response ladder failed (exit 4) | Ask for fewer states or loosen `td_conv_tol` |
| `TDA cannot give … excited states: the molecule has only … single excitations in this basis.` | More excited states than single excitations (exit 4) | Lower `singlets` or use a larger basis |
| `Nonadiabatic coupling vectors are not available for method = '…'; request overlaps instead.` | NACs requested from a `tda`/`tddft` backend (raised as `ValueError` by a library call; not reachable from the CLI) | Use SA-CASSCF, or use overlaps |
| `The electronic state in … was saved with different settings or a different molecule; it cannot be reused.` | `load_state` with a file from other settings (raised as `ValueError` by a library call; not reachable from the CLI) | Use the state file of the same run |
| `… is not a readable gqteashmd electronic state: …` | `load_state` with a missing, damaged or inconsistent state file (raised as `ValueError` by a library call; not reachable from the CLI) | Use a file written by `save_state` of the same version |
| `The wavefunction overlap failed: MemoryError: The wavefunction overlap would exceed the memory limit of … GiB: estimated … GiB; lower overlap_norm_kept or the active space` | The overlap between two geometries would need more than 2 GiB (raised as a retryable `ElectronicStructureError` by a library call; not reachable from the CLI) | Lower `electronic.overlap_norm_kept` or the active space |
| `The wavefunction overlap failed: …` | The truncation, overlap or tracking step failed (`ElectronicStructureError`, not retried; library call, not reachable from the CLI) | Read the message; report it if the geometry is ordinary |
| `The SA-CASSCF gradient of … failed` / `… gradient of … failed` / `The transition dipole calculation failed` | PySCF failed in a derivative or dipole step (exit 4) | Read the `--log` file; report if the SCF itself converged |
| `internal error: …` | A program bug | Report it with `logs/gqteashmd.log` and the task's `result.json` |

---

## 10. Known limitations

- **Feature scope:** only the `prepare` stage exists. The electronic-structure layer works through `gqteashmd single-point` only; no workflow stage uses it yet, so the hardware and per-stage resource settings and `pause_after` have little visible effect until the later stages arrive.
- **Electronic structure:** closed-shell singlet molecules with a restricted reference only; no triplet states; nonadiabatic couplings (SA-CASSCF only) and wavefunction overlaps exist only as a library and are not used by any command or stage yet (P7), and the Hessian is missing (P4); TDA/TDDFT have no analytic couplings, and their overlaps use the excitation amplitudes X only (§8.6.8); the cost of the overlap engine grows with the number of determinants kept (`overlap_norm_kept`); overlap-based couplings carry noise of order $10^{-7}/\Delta t$ at default CASSCF convergence; no density fitting; no scratch-directory input key yet (the backend supports one, and writes no checkpoint file); TDDFT/TDA describe the S₁/S₀ crossing seam incorrectly (§8.6.4); DFT gradients omit the grid response, so use `grid_level` ≥ 5 for dynamics (§8.6.4). The SCF, CASSCF and TD calculations each have a retry ladder (§8.6.1, §8.6.8); a calculation that fails every rung stops.
- **Workflow checks still planned (P4):** the preflight does not yet check the PySCF and geomeTRIC versions or that a scratch directory is writable; `validate` reports only the first violation it finds in the input file; a run refused because a stage is stale still opens and writes `state.db`.
- **Platform:** Linux only, with no batch-scheduler support. Runs are designed for a single workstation.
- **Project-folder check:** the check for Windows drives and OneDrive (§7.9) looks at the text of the path. An absolute path that is a symbolic link into a Windows drive or a OneDrive folder is not detected.
- **SHARC comparison is not automated:** the tests against SHARC 4.1 need a SHARC installation and run only on the development machine (`pytest -m sharc`); the continuous-integration workflow (first run 2026-11-07, passing) runs everything else.

## 11. Roadmap

P1 foundation (done) → P2 input schema and pre-run checks (done) → P3a PySCF backend (done: energies, gradients, transition dipoles, `single-point`) → P3b couplings, overlaps, phase tracking and restarts (done, library only) → P3c hardening sweep (done) → **P4 optimization, frequencies and Hessian (next)** → P5 Wigner sampling → P6 excitations and initial states → P7 dynamics → P8 analysis → P9 systematic validation against SHARC 4.1. This manual is updated with each step. Sections marked [PLANNED] become [IMPLEMENTED] when their sub-project is finished and validated.

## 12. References
1. J. C. Tully, *J. Chem. Phys.* **93**, 1061 (1990): fewest-switches surface hopping.
2. G. Granucci, M. Persico, A. Toniolo, *J. Chem. Phys.* **114**, 10608 (2001): local diabatization.
3. G. Granucci, M. Persico, *J. Chem. Phys.* **126**, 134114 (2007): energy-based decoherence correction.
4. S. Mai, P. Marquetand, L. González, *WIREs Comput. Mol. Sci.* **8**, e1370 (2018): the SHARC approach.
5. E. Wigner, *Phys. Rev.* **40**, 749 (1932); M. Barbatti, K. Sen, *Int. J. Quantum Chem.* **116**, 762 (2016): Wigner sampling for initial conditions.
6. Q. Sun et al., *J. Chem. Phys.* **153**, 024109 (2020): PySCF.
7. CODATA 2018 recommended values of the fundamental physical constants.
