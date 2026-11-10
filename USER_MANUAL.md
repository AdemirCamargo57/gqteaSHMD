# gqteashmd — User Manual

**Version:** 0.1.0.dev0 (development snapshot, 2026-11-08)
**Program:** `gqteashmd`, automated surface-hopping molecular dynamics with PySCF
**Licence:** MIT

> **Read this first: what works today.** This is an early development version. The **workflow engine is complete**: input file (with hardware and per-stage resource settings), checks before a run (preflight), project management, parallel tasks, crash-safe resume, graceful stop, review pauses, change detection, and a record of the software used (provenance). Six stages exist: **`prepare`** (validates and stores the molecule), **`optimize`** (optional S0 geometry optimization with geomeTRIC), **`frequencies`** (S0 Hessian, harmonic frequencies, normal modes, Molden file), **`sampling`** (harmonic Wigner sampling of initial geometries and velocities, from the frequencies), **`excitations`** (vertical excitation energies and oscillator strengths at every sample, and the absorption spectrum) and **`selection`** (the initial excited state of every trajectory). `optimize`, `frequencies` and `excitations` work at the level of the `[electronic]` input section. The **electronic-structure layer** (PySCF: SCF, SA-CASSCF, TDA, TDDFT, CIS, TDHF; energies, analytic gradients, transition dipoles, the S0 Hessian) is **implemented** and validated against SHARC 4.1 and other programs; you can also use it directly with **`gqteashmd single-point`** (§5.3). The **surface-hopping dynamics core** (velocity Verlet, electronic propagation, Tully and SHARC-type hopping, velocity adjustment, decoherence, restarts) is **implemented and validated on Tully's analytic models** (P7a-1): run it with **`gqteashmd model-dynamics`** (§5.7, §8.10–§8.12, §8.14). **The `dynamics` stage for molecules, and analysis, are *planned*; so is the use of nonadiabatic couplings and wavefunction overlaps in dynamics** (SA-CASSCF couplings and the overlap engine exist as a library, §8.6.8, but no project command reaches them), and so are the LVC model, local diabatization from overlaps and spin–orbit coupling (P7a-2, P7b). The analysis is described in §8.13 so that you can see in advance how results will be computed. Every section is marked:
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

`gqteashmd` will run a complete photodynamics study from **one input file**; the first six stages below exist today. The stages run in this order:

| Order | Stage | What it does | Status |
|---|---|---|---|
| 0 | `prepare` | Reads and checks the molecule (XYZ syntax, elements, charge and multiplicity); stores a normalized copy | **[IMPLEMENTED]** |
| 1 | `optimize` (optional) | Ground-state (S0) geometry optimization with geomeTRIC, on the surface of the `[electronic]` section | **[IMPLEMENTED, P4]** |
| 2 | `frequencies` | S0 Hessian, harmonic frequencies and normal modes (analytic Hessian or finite differences of gradients); Molden, HDF5 and JSON output | **[IMPLEMENTED, P4]** |
| 3 | `sampling` | Harmonic Wigner sampling of initial geometries and velocities, at zero or finite temperature | **[IMPLEMENTED, P5]** |
| 4 | `excitations` | Vertical excitation energies and oscillator strengths at the reference and at each sampled geometry; nuclear-ensemble absorption spectrum | **[IMPLEMENTED, P6]** |
| 5 | `selection` | Choice of the initial electronic state for each trajectory (bright-state rule or one fixed state) | **[IMPLEMENTED, P6]** |
| 6 | `dynamics` | Surface-hopping trajectories, run in parallel | [PLANNED, P7b] (the dynamics core exists: §5.7, §8.10–§8.14) |
| 7 | `analysis` (optional) | State populations, lifetimes, summary report | [PLANNED, P8] |

In SHARC 4.1 these steps are separate manual tools: `wigner.py`, `setup_init.py`, `excite.py`, `setup_traj.py` and `populations.py`. `gqteashmd` runs them automatically, one after another. You only intervene when the program asks you to fix something.

The electronic-structure program is **PySCF** **[IMPLEMENTED]**: the `[electronic]` input section (§4.2), the command `gqteashmd single-point` (§5.3) and the stages `optimize`, `frequencies` and `excitations` use it today. The stage that will call it later (`dynamics`) is still planned, as is the use of couplings in dynamics; SA-CASSCF nonadiabatic couplings, wavefunction overlaps and phase tracking (P3b) exist as a library only (§8.6.8). ORCA, Gaussian and others may be added later.

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

The environment installs PySCF (pinned to the 2.14 series), NumPy, SciPy, h5py and geomeTRIC. The program requires `pydantic>=2.6`, `numpy>=1.26`, `scipy>=1.11`, `pyscf>=2.6`, `h5py>=3.9` and `geometric>=1.0` (the environment pins the geomeTRIC 1.1 series). `gqteashmd run` checks the versions that the enabled stages need before it starts (§7.9); a PySCF older than 2.6 is also refused when an electronic-structure calculation starts (§9).

### 2.3 Where to run projects
Keep project directories on a **local Linux disk**, for example `~/gqteashmd-runs/` or a data disk on the workstation. **Do not use OneDrive folders or Windows drives (`/mnt/c/...`).** The program writes many small files and uses file locks, and synchronized or Windows-mounted folders are slow and unreliable for both. `gqteashmd run` refuses such a project directory (§7.9). Ordinary Linux data disks mounted under `/mnt`, such as `/mnt/data`, are accepted.

---

## 3. Quick start

**[IMPLEMENTED]**

```bash
mkdir -p ~/gqteashmd-runs/water && cd ~/gqteashmd-runs/water
gqteashmd init input.toml                  # writes a commented template
sed -i 's/^total_cores = .*/total_cores = 4/; s/^total_memory_mb = .*/total_memory_mb = 6000/; s/^basis = .*/basis = "sto-3g"/; s/^functional = .*/functional = "hf"/' input.toml
sed -i 's/^n_samples = .*/n_samples = 20/; s/^n_trajectories = .*/n_trajectories = 5/; s/^allow_fewer = .*/allow_fewer = true/' input.toml
cat > molecule.xyz <<'EOF'
3
water
O    0.000000    0.000000    0.117300
H    0.000000    0.757200   -0.469200
H    0.000000   -0.757200   -0.469200
EOF
gqteashmd validate input.toml              # checks the input; runs nothing
gqteashmd run input.toml                   # runs the six stages (about 1 min here)
gqteashmd status input.toml                # shows progress
```

The `sed` line sets `total_cores` and `total_memory_mb`, which are **required** and describe *your* workstation (the cores and the RAM in MB that you allow `gqteashmd` to use). The template's values (8 cores, 16000 MB) are only examples. If your values exceed what the program detects on the machine, `run` prints a warning but continues (§7.9). Set the two values to something that fits your machine, as above. The first `sed` line also replaces the template's electronic-structure setting (TDA-B3LYP/def2-SVP) by Hartree–Fock/STO-3G (`method = "tda"` with `functional = "hf"` is CIS), so that this first example is fast. The second `sed` line shrinks the run to **20 samples** and **5 trajectories** and sets `allow_fewer = true` (the template asks for 200 samples and 100 trajectories, which would take several minutes here, and stops if fewer than 100 trajectories can be selected, §8.9.3). `optimize` and `frequencies` need the `[electronic]` section (§4.2).

Expected output of `validate` (the paths are those of your machine):

```text
Input file /home/you/gqteashmd-runs/water/input.toml is valid.
  project directory: /home/you/gqteashmd-runs/water/run
  geometry:          /home/you/gqteashmd-runs/water/molecule.xyz
```

Expected output of `run` (real output from a development machine; the dates and times differ):

```text
2026-11-08 12:25:54 INFO    gqteashmd.orchestrator: Stage prepare: starting
2026-11-08 12:25:58 INFO    gqteashmd.orchestrator: Stage prepare: done (1/1 tasks)
2026-11-08 12:25:58 INFO    gqteashmd.orchestrator: Stage optimize: starting
2026-11-08 12:26:02 INFO    gqteashmd.orchestrator: Stage optimize: done (1/1 tasks)
2026-11-08 12:26:02 INFO    gqteashmd.orchestrator: Stage frequencies: starting
2026-11-08 12:26:06 INFO    gqteashmd.orchestrator: Stage frequencies: done (1/1 tasks)
2026-11-08 12:26:06 INFO    gqteashmd.orchestrator: Stage sampling: starting
2026-11-08 12:26:07 INFO    gqteashmd.orchestrator: Stage sampling: done (1/1 tasks)
2026-11-08 12:26:07 INFO    gqteashmd.orchestrator: Stage excitations: starting
2026-11-08 12:26:52 INFO    gqteashmd.stages.excitations: Stage excitations: S1  mean dE = 12.157 +- 1.683 eV, mean f = 0.0024 +- 0.0006 (20 ok)
2026-11-08 12:26:52 INFO    gqteashmd.stages.excitations: Stage excitations: S2  mean dE = 14.901 +- 1.965 eV, mean f = 0.0260 +- 0.0391 (20 ok)
2026-11-08 12:26:52 INFO    gqteashmd.orchestrator: Stage excitations: done (21/21 tasks)
2026-11-08 12:26:52 INFO    gqteashmd.orchestrator: Stage selection: starting
2026-11-08 12:26:54 INFO    gqteashmd.orchestrator: Stage selection: done (1/1 tasks)
Workflow complete: 6 stage(s) run, 0 already up to date, 0 skipped.
```

Running the same command again recomputes nothing:

```text
2026-11-08 12:26:56 INFO    gqteashmd.orchestrator: Stage prepare: up to date
2026-11-08 12:26:56 INFO    gqteashmd.orchestrator: Stage optimize: up to date
2026-11-08 12:26:56 INFO    gqteashmd.orchestrator: Stage frequencies: up to date
2026-11-08 12:26:56 INFO    gqteashmd.orchestrator: Stage sampling: up to date
2026-11-08 12:26:56 INFO    gqteashmd.orchestrator: Stage excitations: up to date
2026-11-08 12:26:56 INFO    gqteashmd.orchestrator: Stage selection: up to date
Workflow complete: 0 stage(s) run, 6 already up to date, 0 skipped.
```

Expected output of `status`:

```text
Project: my_molecule  (/home/you/gqteashmd-runs/water/run)
Controller: not running
stage           status         done  failed  pending  running  message
prepare         done              1       0        0        0  1/1 tasks succeeded
optimize        done              1       0        0        0  1/1 tasks succeeded
frequencies     done              1       0        0        0  1/1 tasks succeeded
sampling        done              1       0        0        0  1/1 tasks succeeded
excitations     done             21       0        0        0  21/21 tasks succeeded
selection       done              1       0        0        0  1/1 tasks succeeded
```

The results are in `~/gqteashmd-runs/water/run/` (see §6). §5.4, §5.5 and §5.6 show what the files of this run contain. With the default input the `sampling` stage draws 200 samples and the `excitations` stage computes all of them; the captured output above and in §5.5 and §5.6 used `n_samples = 20` (the log lines do not show the number). The `excitations` stage dominates the run time: 21 single-point tasks, one after the other because `max_parallel_tasks = 1`.

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

# Electronic structure (used by optimize, frequencies and the later quantum-chemistry stages):
[electronic]
method = "tda"                # "casscf" | "tda" | "tddft"
basis = "def2-svp"
functional = "b3lyp"          # not used by method = "casscf"; use "hf" for CIS/TDHF
singlets = 3                  # number of singlet states INCLUDING S0
# CASSCF alternative (replace the block above):
# [electronic]
# method = "casscf"
# basis = "6-31g*"
# singlets = 2
# [electronic.casscf]
# ncas = 2                      # active orbitals
# nelecas = 2                   # active electrons

[optimize]
enabled = true                # false: use the input geometry as it is
max_steps = 100
convergence = "GAU"           # geomeTRIC set: GAU, GAU_LOOSE, GAU_TIGHT, GAU_VERYTIGHT, ...

[frequencies]
method = "auto"               # "auto" | "analytic" | "finite_difference" (no analytic for casscf)
fd_step_bohr = 0.005          # displacement for finite differences
imaginary_cutoff_cm1 = 20.0   # imaginary modes above this wavenumber are errors

[sampling]
n_samples = 200               # number of Wigner-sampled initial geometries and velocities
temperature_K = 0.0           # 0 = zero-point (vibrational ground state) distribution
low_frequency_cutoff_cm1 = 10.0   # modes below this wavenumber are not sampled

[excitations]
include_reference = true      # also compute the vertical excitations at the reference geometry
samples_per_task = 1          # samples per task (raise for cheap methods: start-up dominates)
max_failed_fraction = 0.05    # stop if more samples than this fraction fail
broadening_fwhm_eV = 0.1      # Gaussian width (FWHM) of the nuclear-ensemble spectrum
spectrum_step_eV = 0.01       # energy grid step of the spectrum

[selection]
# n_samples x (number of states) x acceptance (typically 0.1-0.3) must reach n_trajectories,
# or the stage stops (set allow_fewer = true to accept fewer). f_over_dE2 favours low-energy
# pairs: set energy_window_eV around the band of interest.
mode = "bright"               # "bright": accept by oscillator strength; "fixed": one state
weighting = "f_over_dE2"      # "f_over_dE2" (SHARC excite.py-compatible) or "f"
# states = ["S1", "S2"]       # candidate excited states (default: all excited singlets)
# energy_window_eV = [4.5, 6.0]   # accepted excitation energy range (default: none)
n_trajectories = 100          # number of trajectories to start
allow_fewer = false           # accept fewer trajectories if too few pass the selection
```

The template's `[electronic]` block is active (TDA-B3LYP/def2-SVP with S0 and two excited states), because `optimize` and `frequencies` need it, and so does `single-point`; the CASSCF alternative is given as comments. `[optimize]`, `[frequencies]`, `[sampling]`, `[excitations]` and `[selection]` show their defaults (the commented `states` and `energy_window_eV` have no default value: the default is all excited singlets and no window). The template omits one key, `execution.poll_interval_s` (default 0.5). The table lists every key:

| Key | Type | Default | Allowed | Meaning |
|---|---|---|---|---|
| `schema_version` | int | 1 | 1 | Input-format version |
| `project.name` | str | — (required) | `[A-Za-z0-9_.-]+` | Name used in logs and file comments |
| `project.directory` | path | — (required) | any local Linux disk | Where all results go (created if missing). Checked by preflight (§7.9) |
| `project.seed` | int | 12345 | ≥ 0 | Master seed; every random choice derives from it. Since P5 it controls the Wigner samples (§8.8): the same input gives the same samples, another seed gives other samples. Changing it after `sampling` has run is a stale-input stop (§7.5) |
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
| `optimize.enabled` | bool | `true` | `true`, `false` | `false` skips the stage; `frequencies` then uses the geometry of `prepare` as it is (§7.5) |
| `optimize.max_steps` | int | 100 | ≥ 1 | Maximum number of geomeTRIC iterations (energy and gradient evaluations) before the stage fails |
| `optimize.convergence` | str | `"GAU"` | `"GAU"`, `"GAU_LOOSE"`, `"GAU_TIGHT"`, `"GAU_VERYTIGHT"`, `"NWCHEM_LOOSE"`, `"TURBOMOLE"`, `"INTERFRAG_TIGHT"` | Name of the geomeTRIC convergence set. `"GAU"` uses Gaussian-style thresholds on forces and displacements |
| `frequencies.method` | str | `"auto"` | `"auto"`, `"analytic"`, `"finite_difference"` | How the S0 Hessian is obtained (§8.7). `"auto"` is `"analytic"` for `tda`/`tddft` and `"finite_difference"` for `casscf`. `"analytic"` with `casscf` is an input error |
| `frequencies.fd_step_bohr` | float | 0.005 | > 0 and ≤ 0.1 | Cartesian displacement $h$ of the central differences (bohr); used only by `"finite_difference"` |
| `frequencies.imaginary_cutoff_cm1` | float | 20.0 | ≥ 0 | An imaginary mode with $\tilde\nu < -$ this value stops the workflow; smaller ones only warn (§8.7). It is a check on the finished results only: changing it does not recompute anything (§8.3) |
| `sampling.n_samples` | int | 200 | ≥ 1 | Number $N_s$ of Wigner samples (initial geometries with velocities) to draw (§8.8) |
| `sampling.temperature_K` | float | 0.0 | ≥ 0 | Temperature $T$ of the thermal Wigner distribution, in kelvin. `0` is the vibrational ground state (zero-point distribution) (§8.8) |
| `sampling.low_frequency_cutoff_cm1` | float | 10.0 | > 0 | Modes with wavenumber $\tilde\nu$ below this value, including the small imaginary modes that `frequencies` accepts, are **not sampled**: they stay at the equilibrium coordinate with zero velocity, and a warning lists them (§8.8). `0` is not allowed |
| `excitations.include_reference` | bool | `true` | `true`, `false` | Also compute the vertical excitations at the reference (equilibrium) geometry, as task `ref`. The values go into the `reference` group of `excitations.h5` and into `excitations.json`; they are not used for the spectrum or the selection (§8.9). Part of the fingerprint: changing it after results exist needs `--rerun-from excitations`, which recomputes every sample |
| `excitations.samples_per_task` | int | 1 | ≥ 1 | Wigner samples computed by one task (§8.9.1). Larger values make fewer, longer tasks; a retry of a task repeats all its samples. **Raise it for cheap methods**: with one sample per task, Python and PySCF start-up (about 2 s per task, §5.6) dominates |
| `excitations.max_failed_fraction` | float | 0.05 | $0 \le x < 1$ | The stage fails if more than this fraction of the samples (or of the tasks) failed (§8.9.1). Equality is allowed. It is a check on the finished results: changing it does not recompute anything (§8.3) |
| `excitations.broadening_fwhm_eV` | float | 0.1 | > 0 | Full width at half maximum $\delta$ of the Gaussian that broadens every line of the spectrum (§8.9.2). Changing it does not recompute the excitations |
| `excitations.spectrum_step_eV` | float | 0.01 | > 0 | Energy grid step of `spectrum.csv` (§8.9.2). Changing it does not recompute the excitations |
| `selection.mode` | str | `"bright"` | `"bright"`, `"fixed"` | `"bright"`: stochastic choice weighted by the oscillator strength. `"fixed"`: all trajectories start in the one state of `selection.states` (§8.9.3) |
| `selection.weighting` | str | `"f_over_dE2"` | `"f_over_dE2"`, `"f"` | Acceptance weight $w$ of the bright mode (§8.9.3): $w=f/\Delta E^2$ (SHARC 4.1 `excite.py`-compatible, measured in §8.9.4) or $w=f$ (the oscillator-strength convention). `f_over_dE2` favours low-energy pairs, so **use an `energy_window_eV`** with it (§8.9.3). Ignored by `"fixed"` (and then not part of the fingerprint) |
| `selection.states` | list of str | all excited singlets | labels `"S1"`, `"S2"`, ...; each must exist (`S0` and triplets are rejected, no duplicates, not empty) | Candidate states. `"fixed"` needs exactly one |
| `selection.energy_window_eV` | `[lo, hi]` | none (no window) | finite, $0 \le lo < hi$ | Only pairs with $lo \le \Delta E \le hi$ (inclusive, in eV) are candidates. **Recommended with `f_over_dE2`**: put it around the band to excite (read it from `spectrum.csv`); it keeps a low-gap outlier from setting $w_{\max}$ (§8.9.3) |
| `selection.n_trajectories` | int | 100 | ≥ 1 | Number of trajectories (initial conditions) to select. Budget: $N_s \times$ (number of states) $\times$ acceptance (typically 0.1-0.3; 6/40 in §5.6) must reach it, otherwise the stage stops (O20, §8.9.3) |
| `selection.allow_fewer` | bool | `false` | `true`, `false` | `false`: stop if fewer than `n_trajectories` can be selected. `true`: accept the smaller number with a warning (§8.9.3). Zero selected is always an error. Acts only in the finalization and is not part of the fingerprint: after an O20 stop, set it and run again (no `--rerun-from`) |

**Hardware keys.** `total_cores` and `total_memory_mb` are required, so that the input file records which machine the settings were written for. The input is rejected if the settings do not fit inside them (see the rules below and §8.4). The program also detects the real machine; if your values exceed it, `run` only **warns** (§7.9).

**Per-stage overrides.** A section `[resources.stages.<name>]` changes the resources for one stage. A runnable example:

```toml
[resources]
total_cores = 8
total_memory_mb = 16000

[resources.stages.prepare]         # prepare gets 2 threads; other keys use the [resources] values
threads_per_task = 2
```

A stage with one big task can be given the whole machine. `optimize` always has one task, and `frequencies` has one task with the analytic Hessian. With the finite-difference Hessian, `frequencies` has $1+6N_{\text{at}}$ independent tasks (§8.7), which run in parallel, so leave that stage on the global settings:

```toml
[resources]
total_cores = 32
total_memory_mb = 128000
max_parallel_tasks = 8
threads_per_task = 4
memory_mb_per_task = 4000

[resources.stages.optimize]        # one task: give it the whole machine
max_parallel_tasks = 1
threads_per_task = 32
```

- **Merge rule:** a key you leave out of an override takes the global value from `[resources]`. In the second example, `optimize` uses `memory_mb_per_task = 4000`.
- **Fit rule:** for the global settings **and for every override** (after merging), the input must satisfy $n_p n_t \le N_{\text{cores}}$ and $n_p m \le M_{\text{total}}$ (§8.4). Otherwise the input is rejected, naming `[resources]` or `[resources.stages.<name>]` and the numbers. Note that an override that sets only `max_parallel_tasks` is also checked against the global `threads_per_task`.
- **Stage names:** an override for a stage that does not exist is an input error (`[resources.stages] names unknown stage(s): …`). The stages in this version are `prepare`, `optimize`, `frequencies`, `sampling`, `excitations` and `selection`.

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

#### The `[optimize]` and `[frequencies]` sections
**[IMPLEMENTED]** (P4). Both stages work on the **S0 surface of the `[electronic]` section**, that is, on the ground-state surface that the dynamics will start from (§8.7). The keys are in the table above.

- **`[electronic]` is required.** `optimize` (when enabled) and `frequencies` need it. If it is missing, `validate` and `run` stop with exit code 2: *Stage 'optimize' needs an [electronic] section in the input file; Stage 'frequencies' needs an [electronic] section in the input file. See USER_MANUAL.md §4.* (every stage that lacks it is listed, separated by `; `).
- **`[optimize]` and `[frequencies]` are optional.** If a section is absent, its defaults apply.
- **`frequencies.method = "analytic"` with `electronic.method = "casscf"`** is rejected when the input is loaded, because PySCF has no analytic SA-CASSCF Hessian. When several cross-field rules are violated, the message lists all of them, separated by `; `.
- **`singlets`** does not change the S0 calculation of `tda` and `tddft` (only S0 is computed for the gradients and the Hessian). For `casscf` it does, because the state average defines the S0 surface (§8.7).
- **Fingerprints (§8.3):** `optimize` depends on `[electronic]` and `[optimize]`; `frequencies` on `[electronic]` and `[frequencies]`. Changing a key of one of these sections after results exist is a stale-input stop (§7.5).

#### The `[sampling]` section
**[IMPLEMENTED]** (P5). The section is optional; if it is absent, the defaults of the table apply (200 samples, $T = 0$ K, cutoff 10 cm⁻¹).

- **Requires `frequencies`.** `sampling` reads `frequencies.h5` of the stage before it; it needs no `[electronic]` key of its own.
- **`project.seed` controls the samples.** The random numbers come from a generator that depends only on `project.seed` and on the stage (§8.8, seed hierarchy), so a project run twice with the same input gives identical samples, also after a crash and a restart.
- **Fingerprint (§8.3):** `sampling` depends on `[sampling]`, `project.seed` and the fingerprint of `frequencies`. Changing any of them after results exist is a stale-input stop (§7.5): use `--rerun-from sampling` to draw new samples and keep the earlier stages.
- **Input errors** (exit code 2, found by `validate`): `n_samples < 1`, `temperature_K < 0` or `low_frequency_cutoff_cm1 <= 0`.

#### The `[excitations]` and `[selection]` sections
**[IMPLEMENTED]** (P6). Both sections are optional; if one is absent, the defaults of the table apply. The keys are in the table above.

- **`[electronic]` is required, with `singlets >= 2`.** `excitations` runs at the level of the `[electronic]` section, **with all its settings** (for `casscf` the whole state average, for `tda`/`tddft` the functional and the number of roots), because it needs the excited states. If `[electronic]` is missing, `validate` and `run` stop with exit code 2 (the rule of `optimize` and `frequencies`, above). `singlets = 1` (only S0) is rejected when the input is checked: *the excitations stage needs at least one excited state: [electronic] singlets must be >= 2*. The spectrum and the selection use the states S1, ..., S($n-1$) with $n$ = `singlets`.
- **The stages and what they read.** `excitations` reads `samples.h5` of `sampling` (the reference geometry and the sample positions) and the molecule of `prepare`. `selection` reads `excitations.h5`. Neither needs an extra input file.
- **State labels (`selection.states`) are checked when the input is loaded** (`validate`, `run`): a state that does not exist with the chosen `singlets` is an input error (exit code 2), for example *selection.states: S5 does not exist with [electronic] singlets = 3 (S0..S2)*; `"fixed"` without exactly one state is also an input error. When several rules are violated the message lists all of them, separated by `; `.
- **The random numbers.** `selection` draws from the stream of `project.seed` and the stage name `selection` (§8.9.3). The same input, with the same excitations, selects the same trajectories.
- **Fingerprints (§8.3):** `excitations` depends on `[electronic]` (without `overlap_norm_kept`), on `include_reference` and `samples_per_task`, on `sampling.n_samples` and on the fingerprint of `sampling`; **not** on `max_failed_fraction`, `broadening_fwhm_eV` or `spectrum_step_eV`, which act only on the finished results (if the stage stopped on `max_failed_fraction`, raise the limit and run again: only the finalization is repeated from the stored task results). `selection` depends on the `[selection]` section **without `allow_fewer`** (and without `weighting` when `mode = "fixed"`, where it has no effect), `project.seed` and the fingerprint of `excitations`; so after a stop on too few trajectories, setting `allow_fewer = true` and running again repeats only the finalization. Changing any other key after results exist is a stale-input stop (§7.5): use `--rerun-from excitations` (this recomputes every sample, e.g. after changing `include_reference`) or `--rerun-from selection`.
- **Input errors** (exit code 2, found by `validate`): `samples_per_task < 1`, `max_failed_fraction` outside $[0,1)$, a non-positive `broadening_fwhm_eV` or `spectrum_step_eV`, an unknown `mode` or `weighting`, a bad `energy_window_eV` (not two finite numbers with $0 \le lo < hi$), `n_trajectories < 1`, and the state-label problems above.

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
**[PLANNED]** Their key names and meanings are in the master plan §4.3 (`[optimize]`, `[frequencies]`, `[sampling]`, `[excitations]` and `[selection]` are implemented, see above). Each sub-project adds the sections it needs, and this manual is updated each time. The sections for the later stages are not part of the input yet, because their keys depend on decisions that are still open:
- `[dynamics]` of the project input (P7b; the small input file of `gqteashmd model-dynamics`, §5.7, is separate and already has a `[dynamics]` table with the options of §8.12); `[analysis]` (P8).
- Additional `[electronic]` keys: density fitting, MC-PDFT, triplet states (with intersystem crossing).
- A `scratch` directory key in the input file (the backend takes a scratch directory and has no checkpoint file since P3b, §8.6.8; the input key arrives with the stage that uses it).

Until then, an input file that contains one of these sections or keys is rejected as unknown (§4.1).

---

## 5. Commands

**[IMPLEMENTED]**

| Command | What it does |
|---|---|
| `gqteashmd init PATH [--force]` | Writes a commented template input file. Refuses to overwrite an existing file unless `--force` is given |
| `gqteashmd validate INPUT` | Checks the input file, that the geometry is an existing regular file, and that `[resources.stages]` and `workflow.pause_after` name stages that exist. It checks that every enabled stage has the input section it needs (`[electronic]`, §4.2). It does not read the XYZ contents (coordinates, elements, charge/multiplicity); `prepare` does. Runs nothing, and does not run the preflight checks (§7.9) |
| `gqteashmd run INPUT` | Runs the preflight checks (§7.9), then runs or resumes every stage that is not yet done |
| `gqteashmd run INPUT --rerun-from STAGE` | Moves the results of `STAGE` and all later stages to `.gqteashmd/superseded/`, then recomputes them |
| `gqteashmd status INPUT` | Shows whether a controller is running, and per stage the status, the task counts (done/failed/pending/running) and the last message. It opens the progress database read-only and changes nothing, so it is safe at any time |
| `gqteashmd stop INPUT` | Asks the running controller to stop gracefully |
| `gqteashmd single-point INPUT [--json FILE] [--log FILE]` | Computes the electronic structure at the input geometry (§5.3) |
| `gqteashmd model-dynamics INPUT [--json FILE]` | Runs a surface-hopping ensemble on one of Tully's 1D models and prints the branching probabilities (§5.7) |
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

### 5.4 Example: `optimize` and `frequencies` on water
**[IMPLEMENTED]** (P4). The run of §3 (water, HF/STO-3G, analytic Hessian) took 14 s in total and wrote, next to `00_prepare/`, the files below. The field tables are in §6; this section shows real contents.

| File | Contents | Units |
|---|---|---|
| `01_optimize/optimized.xyz` | Optimized geometry; the comment line holds the S0 energy | Å (energy in $E_h$) |
| `01_optimize/optimization_path.xyz` | One frame per gradient evaluation of the optimizer; frame 0 is the starting geometry | Å (energies in $E_h$) |
| `01_optimize/optimize.json` | Energy, convergence, number of gradients, final maximum gradient, optimized coordinates | $E_h$, $E_h/a_0$, bohr, Å |
| `02_frequencies/frequencies.h5` | Masses, coordinates, Hessian, frequencies, normal modes, reduced masses | each dataset has a `units` attribute |
| `02_frequencies/frequencies.molden` | Frequencies and normal modes for a viewer (Molden, Jmol, Avogadro, ...); SHARC 4.1's `wigner.py` also reads it | bohr, cm⁻¹ |
| `02_frequencies/frequencies.json` | Frequencies, reduced masses, warnings, method, checks | cm⁻¹, amu, $E_h/a_0$ |

`optimized.xyz`. The input geometry had an O–H distance of 0.958 Å and an H–O–H angle of 104.5°; the optimizer ends at 0.989 Å and 100.0°:

```text
3
energy -74.9659011864 Eh
O     -0.0000000000   -0.0000000000    0.1502207532
H     -0.0000000000    0.7580696780   -0.4856603766
H      0.0000000000   -0.7580696780   -0.4856603766
```

The comment lines of `optimization_path.xyz` show the energy falling over the 5 gradient evaluations:

```text
step 0 energy -74.9630231385 Eh
step 1 energy -74.9657219406 Eh
step 2 energy -74.9658625222 Eh
step 3 energy -74.9659003417 Eh
step 4 energy -74.9659011864 Eh
```

`optimize.json` (the coordinates are left out here):

```json
{"schema_version": 1, "converged": true, "convergence": "GAU", "max_steps": 100,
 "energy_hartree": -74.96590118638045, "n_gradients": 5,
 "max_gradient_hartree_per_bohr": 8.566197959058997e-05, "symbols": ["O", "H", "H"]}
```

`frequencies.json`:

```json
{"schema_version": 1, "method": "analytic", "geometry_source": "optimize",
 "energy_hartree": -74.96590118638039, "max_gradient_hartree_per_bohr": 8.565669726223746e-05,
 "linear": false, "n_external": 6, "external_residual_cm1": 32.43849554737321,
 "frequencies_cm1": [2170.19451623371, 4139.57161119818, 4390.548011804773],
 "reduced_masses_amu": [1.078489404457653, 1.0490981633895675, 1.0773535803600447],
 "imaginary_modes": [], "small_imaginary_modes": [], "warnings": []}
```

The start of `frequencies.molden` (coordinates in bohr, wavenumbers in cm⁻¹; the file continues with `[FR-COORD]` and `[FR-NORM-COORD]`):

```text
[Molden Format]
[Atoms] AU
O      1    8      -0.0000000000      -0.0000000000       0.2838760818
H      2    1      -0.0000000000       1.4325440749      -0.9177651014
H      3    1       0.0000000000      -1.4325440749      -0.9177651014
[FREQ]
     2170.194516
     4139.571611
     4390.548012
```

The three frequencies are the bend (2170 cm⁻¹) and the two stretches of water at this level (HF/STO-3G overestimates all three). The optimization stops at the `GAU` criterion with a maximum gradient of $8.6\times10^{-5}\,E_h/a_0$, so these frequencies differ by up to 0.5 cm⁻¹ from those at a tightly converged geometry (§8.7, validation). Each task directory also holds working files: `tasks/optimize/geometric/`, `tasks/optimize/electronic.log` (PySCF's output) and `tasks/hessian/hessian.npy` (§6).

**What a stale run looks like.** If you now set `[optimize] enabled = false` and run again, the `frequencies` results were computed from the optimized geometry, so the program stops before changing anything (exit code 3):

```text
gqteashmd: error: Stage 'frequencies' has results produced from different inputs (the input file or an earlier stage changed). Run `gqteashmd run /home/you/gqteashmd-runs/water/input.toml --rerun-from frequencies` to move the old results aside and recompute from this stage onward.
```

### 5.5 Example: `sampling` on water
**[IMPLEMENTED]** (P5). Real output from a development machine: the run of §3 (water, HF/STO-3G, analytic Hessian) with `[sampling] n_samples = 20`, otherwise the defaults ($T = 0$ K, `low_frequency_cutoff_cm1 = 10`, `project.seed = 12345`). All four stages took 8 s of computation in total; `sampling` itself took about 2 s, most of it Python start-up. The directory `03_sampling/` holds:

| File | Contents | Units |
|---|---|---|
| `samples.h5` | Positions and velocities of all samples, the normal-mode coordinates and momenta behind them, energies; every dataset has a `units` attribute (§6) | bohr, bohr per atomic unit of time, $E_h$, ... |
| `samples.xyz` | One XYZ frame per sample, for a viewer; the comment line holds the harmonic energy | Å ($E_h$ in the comment) |
| `samples.json` | Summary and the numbers used by the checks (§6, §8.8) | $E_h$, a.u. |
| `tasks/samples/` | `task.json`, `result.json`, `worker.log`, and the original `samples.h5` and `samples.xyz` of the task | |

The first frame of `samples.xyz` (compare with the equilibrium geometry in §5.4: the O–H distances are 1.057 Å and 1.099 Å here instead of 0.989 Å, because each sample is a snapshot of the zero-point motion):

```text
3
sample 1  E_harm = 0.0215368999 Eh
O     -0.0000000000    0.0018898404    0.1563234572
H     -0.0000000000    0.8130669794   -0.5215081833
H      0.0000000000   -0.8430601202   -0.5466669159
```

`samples.json`:

```json
{"schema_version": 1, "n_samples": 20, "n_vibrations": 3, "n_sampled_modes": 3,
 "unsampled_frequencies_cm1": [], "temperature_K": 0.0, "low_frequency_cutoff_cm1": 10.0,
 "project_seed": 12345,
 "mean_energy_hartree": 0.030444773642583044,
 "expected_mean_energy_hartree": 0.024377109264918033,
 "expected_energy_std_hartree": 0.014609203391788141,
 "mean_kinetic_energy_hartree": 0.013506766656044491,
 "mean_removed_kinetic_energy_hartree": 1.4501743702676359e-05,
 "max_linear_momentum_au": 1.9860273225978185e-15,
 "max_angular_momentum_au": 1.5543122344752192e-15,
 "mode_energy_ratio_sums": [13.748202178317413, 33.76726441337912, 22.242311550678913],
 "mode_energy_min_p_value": 0.008363599777671078,
 "mode_energy_p_value_threshold": 3.3333333333333337e-10}
```

(The file itself has its keys in alphabetical order.) How to read it:

- The expected mean harmonic energy is the zero-point energy $\tfrac12\sum_k\hbar\omega_k = 0.024377\,E_h$ (the three frequencies of §5.4). The 20 samples average $0.030445\,E_h$. The stage does not compare this total with a fixed tolerance; it checks the energy of each mode against its exact distribution (§8.8). For scale: $\sigma_E/\sqrt{N_s} = 0.014609/\sqrt{20} = 0.0033\,E_h$, so the deviation here, $0.0061\,E_h$, is $1.9$ standard errors of the mean, which is ordinary statistical noise for 20 samples. With 200 samples the standard error is about 0.001 $E_h$.
- The mean kinetic energy, $0.0135\,E_h$, scatters around half the zero-point energy, $\tfrac14\sum_k\hbar\omega_k = 0.0122\,E_h$, minus the energy removed with the translation and rotation of each sample ($1.5\times10^{-5}\,E_h$ on average, §8.8).
- The total linear and angular momentum of every sample is zero to $2\times10^{-15}$ atomic units.
- `mode_energy_ratio_sums` are the $S_k=\sum_s E_{k,s}/\langle E_k\rangle$ of the three modes (expected value $N_s = 20$ each). The smallest two-sided Gamma tail probability is $p = 0.0084$, far above the limit $10^{-9}/3 = 3.3\times10^{-10}$, so the check passes (§8.8).

### 5.6 Example: `excitations` and `selection` on water
**[IMPLEMENTED]** (P6). Real output from a development machine: the run of §3 (water, HF/STO-3G, that is CIS, `singlets = 3`: S0, S1, S2), with the **small settings** `[sampling] n_samples = 20`, `[selection] n_trajectories = 5` and `allow_fewer = true`, otherwise the defaults (`project.seed = 12345`, broadening 0.1 eV, grid step 0.01 eV, bright mode with `weighting = "f_over_dE2"`, no window). The excitation energies are far too high (12-17 eV) because CIS/STO-3G is a poor description of water; the example shows the workflow and the files, not chemistry. All six stages took 64 s with `max_parallel_tasks = 1`, 45 s of them in `excitations` (21 tasks, one per sample plus the reference); `selection` took about 2 s, mostly Python start-up. The console output is in §3.

`04_excitations/` holds:

| File | Contents | Units |
|---|---|---|
| `excitations.h5` | Energies, excitation energies, transition dipoles and oscillator strengths of all samples and of the reference; every dataset has a `units` attribute (§6) | $E_h$, eV, $e\,a_0$, 1 |
| `excitations.json` | Per-state means and standard deviations, reference values, failed samples (§6) | eV, 1 |
| `spectrum.csv` | The absorption spectrum (columns in §6) | eV, Å² per molecule, M⁻¹ cm⁻¹ |
| `tasks/ref/`, `tasks/s00001/`, ... | `task.json`, `result.json`, `worker.log`, `electronic.log` (PySCF output) and `scratch/` of each task | |

`05_selection/` holds `trajectories.json`, `selection_report.json`, `selection_report.md` and `tasks/select/`.

`excitations.json` (the file has its keys in alphabetical order; `failed` is empty):

```json
{"schema_version": 1, "n_samples": 20, "n_ok": 20, "failed": [],
 "error": null, "reference_error": null,
 "state_labels": ["S0", "S1", "S2"], "include_reference": true,
 "broadening_fwhm_eV": 0.1, "spectrum_step_eV": 0.01,
 "states": {
   "S1": {"mean_excitation_eV": 12.157048, "std_excitation_eV": 1.682796,
          "mean_oscillator_strength": 0.0024137, "std_oscillator_strength": 0.00056057,
          "reference_excitation_eV": 12.485060, "reference_oscillator_strength": 0.0030410},
   "S2": {"mean_excitation_eV": 14.901126, "std_excitation_eV": 1.965218,
          "mean_oscillator_strength": 0.025989, "std_oscillator_strength": 0.039082,
          "reference_excitation_eV": 16.456694, "reference_oscillator_strength": 0.081386}}}
```

(The numbers are shortened here.) How to read it: the 20 samples spread the S1 excitation energy over $\pm1.7$ eV around a mean 0.33 eV below the value at the equilibrium geometry (12.485 eV); the S2 oscillator strength changes by a factor of the order of its own mean from geometry to geometry (mean 0.026, standard deviation 0.039, reference 0.081). The spectrum, `spectrum.csv`, has 880 grid points from 9.17 eV in steps of 0.01 eV; its maximum, $\sigma = 0.0473$ Å² ($\varepsilon = 1237\ \mathrm{M^{-1}cm^{-1}}$), is at 16.81 eV and belongs to S2:

```text
# gqteashmd nuclear-ensemble absorption spectrum
# units: energy_eV = eV, cross sections = angstrom^2 per molecule, epsilon = M^-1 cm^-1
# n_ok = 20
# fwhm_eV = 0.1
# step_eV = 0.01
energy_eV,cross_section_A2,epsilon_M-1cm-1,S1_A2,S2_A2
9.170000e+00,8.687311e-08,2.272064e-03,8.687311e-08,0.000000e+00
9.180000e+00,2.285757e-07,5.978128e-03,2.285757e-07,0.000000e+00
```

When you pause with `pause_after = ["excitations"]`, `selection` has not run yet: read `spectrum.csv` and `excitations.json` (above) to choose `[selection]` (for example an `energy_window_eV` around the band), then run again. After the run, read `selection_report.md` (the 5 trajectories are all in S2, in the order of the samples):

```text
# Initial-state selection

- mode: bright
- weighting: f_over_dE2 (w_max = 0.0003954)
- w_max pair: sample 3, S2, dE = 14.2923 eV, f = 0.0808, w = 0.0003954 (99th percentile of the weights: 0.0003831)
- states: S1, S2
- energy window (eV): none
- trajectories: 5 selected of 5 requested
- candidates: 40, accepted: 6, f_max: 0.09174
- project seed: 12345

## Per state

| state | candidates | accepted | selected | mean dE (eV) | mean f |
|---|---|---|---|---|---|
| S1 | 20 | 0 | 0 | - | - |
| S2 | 20 | 6 | 5 | 16.133 | 0.0845 |

## First 5 trajectories

| id | sample | state | dE (eV) | f |
|---|---|---|---|---|
| t00001 | 2 | S2 | 16.809 | 0.0917 |
| t00002 | 3 | S2 | 14.292 | 0.0808 |
| t00003 | 12 | S2 | 15.996 | 0.0840 |
| t00004 | 14 | S2 | 15.780 | 0.0906 |
| t00005 | 16 | S2 | 17.785 | 0.0756 |
```

(The report ends with the paths of `spectrum.csv` and `excitations.json` and the line *If the run paused here (`pause_after = ["selection"]`, effective once a later stage follows `selection`), review this report, then run `gqteashmd run` again to continue.*; while `selection` is the last stage, it never pauses.) How to read it: there are $20 \times 2 = 40$ candidate pairs (no window). The $w_{\max}$ pair is only slightly above the 99th percentile of the weights, so no single pair dominates (no outlier warning, §8.9.3). The weight is $w=f/\Delta E^2$ and the largest weight, $w_{\max} = 3.954\times10^{-4}\ \mathrm{eV^{-2}}$, belongs to S2 of sample 3 ($f=0.0808$, $\Delta E = 14.292$ eV: $0.0808/14.292^2 = 3.95\times10^{-4}$); it is not the pair with the largest $f$ (S2 of sample 2, $f_{\max}=0.0917$ at 16.81 eV), because a lower energy raises the weight. A pair is accepted with probability $w/w_{\max}$. An S1 pair has $f/\Delta E^2\approx0.0024/12.2^2 = 1.6\times10^{-5}$, about 4 % of $w_{\max}$, so none of the 20 S1 pairs was accepted; 6 S2 pairs were, and the first 5 were kept because `n_trajectories = 5`. `trajectories.json` lists them:

```json
{"schema_version": 1, "n_trajectories": 5, "mode": "bright", "project_seed": 12345,
 "trajectories": [
   {"id": "t00001", "index": 0, "sample": 2, "state": "S2",
    "excitation_energy_eV": 16.809196, "oscillator_strength": 0.091742,
    "rng": {"stage": "dynamics", "task_index": 0}},
   {"id": "t00002", "index": 1, "sample": 3, "state": "S2", "...": "..."}]}
```

`sample` is the 1-based number of the geometry and velocity in `samples.h5` (§6); `rng` names the random stream that `dynamics` will use for this trajectory (§8.9.3).

### 5.7 `gqteashmd model-dynamics`
**[IMPLEMENTED]** (P7a-1). It runs an ensemble of surface-hopping trajectories on one of Tully's one-dimensional two-state models (§8.14) and prints the branching probabilities. It is the way to try the dynamics core (§8.10–§8.12) and to validate your settings against a known answer. It needs no project directory, no molecule and no electronic-structure program (NumPy only), does not run the preflight checks (§7.9) and writes no files unless you ask for JSON. The input file is a **separate small TOML file** (not the project input of §4); its `[dynamics]` table is the dynamics options of §8.12, not the future `[dynamics]` section of a project.

```text
gqteashmd model-dynamics INPUT [--json FILE]
```

| Option | Meaning |
|---|---|
| `--json FILE` | Also write the results to a JSON file (fields below). Refuses an existing directory or an unwritable path before the run |

Exit codes: **0** on success; **2** for an input problem (unknown key, invalid value, unknown model parameter; the message names the key, e.g. `dynamics.edc_convention`); **1** when a trajectory fails although the input is valid (`model-dynamics run failed inside a trajectory: …`, e.g. two states degenerate at the start because `parameters` removed the coupling), and also for an internal error (`internal error: …`, e.g. the `NotImplementedError` of the spin–orbit guard, §8.11); **5** when interrupted with Ctrl-C. Every table is strict: unknown keys are rejected. Everything is in atomic units (§8.10).

| Table | Key | Default | Meaning |
|---|---|---|---|
| `[model]` | `name` | required | `tully_sac`, `tully_dac` or `tully_ecr` |
| | `mass` | 2000 | Particle mass in $m_e$ (> 0) |
| | `parameters` | none | Overrides of the model constants, e.g. `parameters = { A = 0.02 }`; an unknown name is an error that lists the allowed ones (§8.14) |
| `[initial]` | `x0` | −10 | Centre of the packet, in bohr |
| | `k` | required | Mean momentum, in atomic units (non-zero) |
| | `sigma` | `20/|k|` | Packet width $\sigma$ (> 0) |
| | `state` | 0 | Initial adiabatic state (0 = lower) |
| | `sampling` | `"wigner"` | `"wigner"` or `"fixed"` (§8.14) |
| `[dynamics]` | `dt_au` | 1.0 | Time step, in atomic units of time (> 0) |
| | `nsubsteps` | 25 | Electronic substeps per step (≥ 1) |
| | `hopping` | `"sharc"` | `"sharc"`, `"fssh"` or `"none"` (§8.12) |
| | `ekin_correction` | `"parallel_vel"` | `"parallel_vel"` or `"parallel_nac"` |
| | `frustrated` | `"keep"` | `"keep"` or `"reflect"` |
| | `decoherence` | `"edc"` | `"edc"` or `"none"` |
| | `decoherence_param_Eh` | 0.1 | EDC constant $C$, in $E_h$ (≥ 0) |
| | `edc_convention` | `"sharc"` | `"sharc"` or `"granucci_persico"` |
| | `substep_interpolation` | `"midpoint"` | `"midpoint"` or `"endpoint"` (§8.11) |
| | `energy_drift_limit_au` | 0.5 eV = 0.0183746 | Energy-drift limit, in $E_h$ (> 0). Every `[dynamics]` default is the library default of `DynamicsOptions` (§8.12), except `dt_au`, which the library requires |
| `[ensemble]` | `n_trajectories` | 1000 | Number of trajectories |
| | `seed` | 12345 | Seed (≥ 0); the results are reproducible (§8.14) |
| | `x_exit` | 10 | A trajectory stops when $|x|$ > `x_exit` while moving outward. A sample that **starts** beyond `x_exit` and already moves outward stops before its first step and is counted on its starting side, in its initial state; keep the packet inside `x_exit` with $k$ pointing inward (the defaults do) |
| | `max_steps` | 200000 | Step limit per trajectory (the trajectory then counts as `unfinished`) |
| | `workers` | 1 | Worker processes; the counts do **not** depend on it. Use at most the physical cores |

The console output lists, per branch, the count, the probability over the finished trajectories and its 95 % Wilson interval, then the number of unfinished and energy-drift trajectories. The JSON file has `schema_version` (1), `model`, `initial` (with the resolved `sigma`), `dynamics`, `ensemble` (all as run), `counts`, `probabilities`, `intervals_95` (as `[low, high]`), and `n_finished`. The branches are `transmitted_lower`, `transmitted_upper`, `reflected_lower` and `reflected_upper` (§8.14); `unfinished` and `energy_drift` are counted separately.

**Example.** Tully SAC at $k = 20$ (a mean energy of $k^2/2m = 0.1\,E_h$), 200 trajectories on 8 workers, fewest-switches surface hopping without decoherence (`decoherence = "none"`) with the default SHARC-type hopping probability: Tully's plain-FSSH setting, for which the benchmark is asserted (§8.14.1). The input (`sac_k20.toml`):

```toml
[model]
name = "tully_sac"

[initial]
x0 = -10.0
k = 20.0
state = 0

[dynamics]
dt_au = 1.0
decoherence = "none"

[ensemble]
n_trajectories = 200
seed = 20261108
x_exit = 10.0
workers = 8
```

and the real output (WSL, this workstation; 18.0 s):

```text
$ gqteashmd model-dynamics sac_k20.toml --json out.json
Model dynamics: tully_sac, 200 trajectories, 200 finished
branch                 count        P          95% interval
transmitted_lower        104   0.5200      [0.4510, 0.5882]
transmitted_upper         96   0.4800      [0.4118, 0.5490]
reflected_lower            0   0.0000      [0.0000, 0.0188]
reflected_upper            0   0.0000      [0.0000, 0.0188]
unfinished: 0, energy_drift: 0
Results written to out.json
```

The exact quantum result at $k = 20$ is 0.5077 lower and 0.4923 upper (§8.14.1), inside the intervals. The interval of 200 trajectories is wide ($\pm$ 0.07); 2000 trajectories narrow it to $\pm$ 0.02.

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

Stage directories are numbered by the position of the stage in the pipeline and each has the same `tasks/<task id>/` structure. With `optimize` and `frequencies` the project also holds:

```text
├── 01_optimize/                # [IMPLEMENTED, P4]
│   ├── optimized.xyz           # final geometry (Å)
│   ├── optimization_path.xyz   # one frame per gradient evaluation (Å)
│   ├── optimize.json           # see below
│   └── tasks/optimize/         # task.json, result.json, worker.log, electronic.log (PySCF output),
│                               #   scratch/ (PySCF temporary files), geometric/ (geomeTRIC's log and
│                               #   trajectory), and best.xyz if the optimization did not converge
└── 02_frequencies/             # [IMPLEMENTED, P4]
    ├── frequencies.h5          # arrays with units, see below
    ├── frequencies.molden      # for viewers; also read by SHARC's wigner.py
    ├── frequencies.json        # see below
    └── tasks/                  # analytic Hessian: hessian/ (with hessian.npy);
                                #   finite differences: reference/, fd_000_plus/, fd_000_minus/, ... fd_<3N-1>_minus/
```

and, after `frequencies`:

```text
└── 03_sampling/                # [IMPLEMENTED, P5]
    ├── samples.h5              # positions, velocities, normal-mode coordinates, energies (with units)
    ├── samples.xyz             # one XYZ frame per sample, for a viewer
    ├── samples.json            # summary and check values
    └── tasks/samples/          # task.json, result.json, worker.log, and the task's own samples.h5 / samples.xyz
```

and, after `sampling`:

```text
├── 04_excitations/             # [IMPLEMENTED, P6]
│   ├── excitations.h5          # energies, excitation energies, transition dipoles, oscillator strengths (with units)
│   ├── excitations.json        # per-state statistics, reference values, failed samples
│   ├── spectrum.csv            # nuclear-ensemble absorption spectrum
│   └── tasks/                  # ref/ (the reference geometry) and s00001/, s00002/, ... (task id = number of its first
│                               #   sample); each with task.json, result.json, worker.log, electronic.log, scratch/
└── 05_selection/               # [IMPLEMENTED, P6]
    ├── trajectories.json       # the selected initial conditions
    ├── selection_report.json   # numbers behind the selection
    ├── selection_report.md     # the same, to read
    └── tasks/select/           # task.json, result.json, worker.log
```

A disabled `optimize` stage still occupies number 01, so `frequencies` stays in `02_frequencies/`, `sampling` in `03_sampling/`, `excitations` in `04_excitations/` and `selection` in `05_selection/`. In `fd_<k>_plus` and `fd_<k>_minus`, $k$ counts the Cartesian coordinates ($x,y,z$ of atom 1, then atom 2, ...) from 0. PySCF's temporary files of a task are written to its own `scratch/` directory inside the project, never to the system temporary directory.

**`optimize.json`** [IMPLEMENTED]:

| Field | Meaning | Unit |
|---|---|---|
| `schema_version` | File-format version (1) | — |
| `energy_hartree` | S0 energy at the optimized geometry | $E_h$ |
| `converged` | Always `true` (a failed optimization writes no `optimize.json`) | — |
| `n_gradients` | Number of energy+gradient evaluations used by the optimizer. One more gradient is computed afterwards for the next field and is not counted | — |
| `max_gradient_hartree_per_bohr` | Largest absolute Cartesian gradient component at the final geometry | $E_h/a_0$ |
| `convergence`, `max_steps` | The `[optimize]` values used | — |
| `symbols` | Element symbols, in input order | — |
| `coords_bohr`, `coords_angstrom` | Optimized coordinates, one `[x, y, z]` per atom | $a_0$, Å |

**`frequencies.json`** [IMPLEMENTED]:

| Field | Meaning | Unit |
|---|---|---|
| `schema_version` | File-format version (1) | — |
| `method` | `"analytic"` or `"finite_difference"` (the resolved choice) | — |
| `geometry_source` | `"optimize"` or `"prepare"`: where the geometry came from | — |
| `energy_hartree` | S0 energy at that geometry | $E_h$ |
| `max_gradient_hartree_per_bohr` | Largest absolute gradient component there (stationarity check, §8.7) | $E_h/a_0$ |
| `linear` | `true` if the molecule was treated as linear ($I_{\min}/I_{\max} < 10^{-3}$, 5 external motions, §8.7) | — |
| `n_external` | Number of translations and rotations projected out: 6, 5 (linear) or 3 (one atom) | — |
| `fd_consistency_residual_hartree_per_bohr` | Largest $\lvert(E_{+k}-E_{-k})/2h - g_k\rvert$ over the coordinates (finite differences; `null` for the analytic route, §8.7) | $E_h/a_0$ |
| `optimized_energy_difference_hartree` | $\lvert E_{\text{frequencies}} - E_{\text{optimize}}\rvert$ at the optimized geometry (`null` if the geometry did not come from `optimize`) | $E_h$ |
| `consistency_problems` | The messages of failed consistency checks; non-empty only when the stage failed | — |
| `external_residual_cm1` | Largest $\lvert\tilde\nu\rvert$ of the projected-out motions (§8.7) | cm⁻¹ |
| `frequencies_cm1` | Vibrational wavenumbers, ascending; **imaginary modes are negative** | cm⁻¹ |
| `reduced_masses_amu` | Reduced mass of each mode (§8.7 convention) | amu |
| `imaginary_modes` | `[{"mode": i, "frequency_cm1": f}, ...]` for $\tilde\nu < -$ `imaginary_cutoff_cm1` (modes are numbered from 1; non-empty only when the stage failed) | cm⁻¹ |
| `small_imaginary_modes` | The same for $-$cutoff $\le \tilde\nu < 0$ | cm⁻¹ |
| `warnings` | The warning messages issued (§8.7); empty if none | — |

**`frequencies.h5`** [IMPLEMENTED] (HDF5; file attributes `schema_version` = 1, `gqteashmd_version`, `linear` (bool) and `n_external` (int)). Every numeric dataset has a string attribute `units`. $N$ is the number of atoms and $n$ the number of vibrational modes ($3N - n_{\text{ext}}$):

| Dataset | Shape | Unit | Meaning |
|---|---|---|---|
| `symbols` | $N$ | — | Element symbols |
| `masses` | $N$ | amu | Masses used (§8.7) |
| `coordinates` | $N\times3$ | bohr | The geometry analysed |
| `hessian` | $3N\times3N$ | $E_h/a_0^2$ | Cartesian Hessian of S0 (symmetrized) |
| `frequencies` | $n$ | cm⁻¹ | Wavenumbers, ascending; imaginary modes negative |
| `modes_mass_weighted` | $3N\times n$ | 1 (dimensionless) | Column $k$ is the mass-weighted normal mode $\mathbf l_k$ (unit norm) |
| `modes_cartesian` | $3N\times n$ | 1 (dimensionless) | Column $k$ is the Cartesian displacement of mode $k$, scaled to unit length |
| `reduced_masses` | $n$ | amu | Reduced masses |

**`frequencies.molden`** [IMPLEMENTED]: Molden format with `[Atoms] AU`, `[FREQ]` (cm⁻¹), `[FR-COORD]` (bohr) and `[FR-NORM-COORD]` (the `modes_cartesian` columns, one `vibration k` block per mode). Imaginary modes appear as negative wavenumbers.

**`samples.json`** [IMPLEMENTED]:

| Field | Meaning | Unit |
|---|---|---|
| `schema_version` | File-format version (1) | — |
| `n_samples` | Number of samples $N_s$ | — |
| `n_vibrations` | Number $n$ of vibrational modes in `frequencies.h5` | — |
| `n_sampled_modes` | Number of modes that were sampled (wavenumber $\ge$ `low_frequency_cutoff_cm1`) | — |
| `unsampled_frequencies_cm1` | Wavenumbers of the modes that were not sampled (imaginary ones negative); empty if none | cm⁻¹ |
| `temperature_K`, `low_frequency_cutoff_cm1`, `project_seed` | The `[sampling]` and `[project]` values used | K, cm⁻¹, — |
| `mean_energy_hartree` | Mean over the samples of the harmonic energy $E_{\text{harm}}$ (§8.8) | $E_h$ |
| `expected_mean_energy_hartree` | $\langle E\rangle$, the exact mean of the distribution (§8.8) | $E_h$ |
| `expected_energy_std_hartree` | $\sigma_E$, the exact standard deviation of $E_{\text{harm}}$ of one sample (§8.8) | $E_h$ |
| `mode_energy_ratio_sums` | For each sampled mode $k$, $S_k=\sum_s E_{k,s}/\langle E_k\rangle$ over the samples (§8.8); a list with one entry per sampled mode, empty for a single atom | — |
| `mode_energy_min_p_value`, `mode_energy_p_value_threshold` | Smallest two-sided $p$-value of the $S_k$ and the limit $10^{-9}/n_{\text{sampled}}$ below which the stage fails (§8.8); `null` if no mode is sampled | — |
| `mean_kinetic_energy_hartree` | Mean kinetic energy of the final velocities (after the momentum removal) | $E_h$ |
| `mean_removed_kinetic_energy_hartree` | Mean of the kinetic energy removed with the translation and rotation | $E_h$ |
| `max_linear_momentum_au`, `max_angular_momentum_au` | Largest $\lvert\mathbf P\rvert$ and $\lvert\mathbf L\rvert$ over the samples, from the final velocities | atomic units |

**`samples.h5`** [IMPLEMENTED] (HDF5; file attributes `schema_version` = 1, `gqteashmd_version`, `temperature_K`, `low_frequency_cutoff_cm1`, `project_seed`, `n_samples` and `expected_mean_energy_hartree`). Every numeric dataset has a string attribute `units`. $N$ is the number of atoms, $n$ the number of vibrational modes and $N_s$ the number of samples. Dynamics will start from `positions` and `velocities`; `q` and `p` are kept for checks:

| Dataset | Shape | Unit | Meaning |
|---|---|---|---|
| `symbols` | $N$ | — | Element symbols |
| `masses` | $N$ | amu | Masses used (the same as in `frequencies.h5`) |
| `equilibrium_coordinates` | $N\times3$ | bohr | The geometry that was sampled around |
| `frequencies` | $n$ | cm⁻¹ | Wavenumbers of all modes |
| `sampled_modes` | $n$ | 1 = sampled, 0 = not | Which modes were sampled |
| `positions` | $N_s\times N\times3$ | bohr | Geometry of each sample |
| `velocities` | $N_s\times N\times3$ | bohr per atomic unit of time | Velocity of each sample, translation and rotation removed |
| `q` | $N_s\times n_s$ | $\sqrt{m_e}\,a_0$ | Mass-weighted normal coordinates $Q_k$ of the sampled modes ($n_s$ = `n_sampled_modes`) |
| `p` | $N_s\times n_s$ | $\sqrt{m_e}\,a_0$ per atomic unit of time | Their conjugate momenta $P_k$ |
| `harmonic_energy` | $N_s$ | $E_h$ | $E_{\text{harm}}$ of each sample, from $Q$ and $P$ before the momentum removal (§8.8) |
| `kinetic_energy` | $N_s$ | $E_h$ | Kinetic energy of the final velocities |
| `removed_kinetic_energy` | $N_s$ | $E_h$ | Kinetic energy taken away with the translation and rotation |

**`samples.xyz`** [IMPLEMENTED]: the $N_s$ geometries in Å, one XYZ frame each; the comment line is `sample k  E_harm = … Eh`.

**`excitations.json`** [IMPLEMENTED]:

| Field | Meaning | Unit |
|---|---|---|
| `schema_version` | File-format version (1) | — |
| `n_samples`, `n_ok` | Number of samples $N_s$, and of those that succeeded | — |
| `failed` | `[{"sample": k, "error": message}, ...]` for the samples that failed (empty if none), including unusable results (non-finite values, $\Delta E\le0$, §8.9.1) | — |
| `reference_error` | The error of the reference task `ref`, or `null` (also `null` with `include_reference = false`) | — |
| `error` | `null` if the stage succeeded; otherwise the message of the stop (too many failed samples, or the reference failed). The file is written before the stop | — |
| `state_labels` | `["S0", "S1", ...]` | — |
| `states` | `null` if the stage stopped; otherwise, for every excited state, an object with `mean_excitation_eV` and `std_excitation_eV` (mean and **population** standard deviation, divisor $N_{\text{ok}}$, over the successful samples), `mean_oscillator_strength`, `std_oscillator_strength`, and `reference_excitation_eV`, `reference_oscillator_strength` (`null` if `include_reference = false`) | eV, 1 |
| `include_reference`, `broadening_fwhm_eV`, `spectrum_step_eV` | The `[excitations]` values used | —, eV |

**`excitations.h5`** [IMPLEMENTED] (HDF5; file attributes `schema_version` = 1, `gqteashmd_version`, `n_samples`, `n_ok` and `project_seed`). Every numeric dataset has a string attribute `units`. $N_s$ is the number of samples, $n_s$ the number of states including S0 (`electronic.singlets`) and $N$ the number of atoms. **The rows of failed samples are `NaN`**; `ok` says which rows are valid:

| Dataset | Shape | Unit | Meaning |
|---|---|---|---|
| `symbols` | $N$ | — | Element symbols |
| `state_labels` | $n_s$ | — | `S0`, `S1`, ... (the $k$-th root at each geometry, in order of energy) |
| `sample_index` | $N_s$ | 1-based | Sample number, as in `samples.h5` |
| `ok` | $N_s$ | `int8 (1 = ok)` | Whether the sample succeeded (stored as int8: 1 = ok, 0 = failed) |
| `energies` | $N_s\times n_s$ | $E_h$ | Total energies of S0, S1, ... |
| `excitation_energies` | $N_s\times(n_s-1)$ | eV | $\Delta E_{0I}=E_I-E_0$, I = S1, S2, ... |
| `transition_dipoles` | $N_s\times(n_s-1)\times3$ | $e\,a_0$ | $\boldsymbol\mu_{0I}$ (§8.6.5); the overall sign of each vector is arbitrary |
| `oscillator_strengths` | $N_s\times(n_s-1)$ | 1 | $f_{0I}$ (§8.6.6) |
| `reference/…` | one row each | | Group written only if `include_reference = true`: `energies`, `excitation_energies`, `transition_dipoles`, `oscillator_strengths` as above, and `coordinates` ($N\times3$, bohr), the geometry that `sampling` sampled around |

**`spectrum.csv`** [IMPLEMENTED]: comment lines that start with `#` (title, units, `n_ok`, `fwhm_eV`, `step_eV`), a header line, then one line per grid point with values in `%.6e` format. Columns: `energy_eV` (photon energy $E$), `cross_section_A2` (total $\sigma(E)$, Å² per molecule), `epsilon_M-1cm-1` (molar absorption coefficient $\varepsilon(E)$), and one column `S1_A2`, `S2_A2`, ... per excited state (that state's contribution to $\sigma$; the columns add up to `cross_section_A2`). The formulas are in §8.9.2.

**`trajectories.json`** [IMPLEMENTED]:

| Field | Meaning |
|---|---|
| `schema_version`, `n_trajectories`, `mode`, `project_seed` | Format version (1), the number selected (can be below the request with `allow_fewer = true`), `"bright"` or `"fixed"`, and `project.seed` |
| `trajectories[].id` | `t00001`, `t00002`, ... |
| `trajectories[].index` | 0-based number $k$ of the trajectory |
| `trajectories[].sample` | 1-based sample number: the geometry and velocity in `samples.h5` (`positions[sample-1]`, `velocities[sample-1]`) |
| `trajectories[].state` | Initial electronic state label, e.g. `S2` |
| `trajectories[].excitation_energy_eV`, `oscillator_strength` | $\Delta E$ (eV) and $f$ of that pair |
| `trajectories[].rng` | `{"stage": "dynamics", "task_index": k}`: the random stream of the future `dynamics` task of this trajectory, derived from `project.seed`, the stage name and $k$ (§8.9.3). It is not an integer seed |

**`selection_report.json`** [IMPLEMENTED]:

| Field | Meaning |
|---|---|
| `schema_version`, `mode`, `weighting` | Format version (1), `"bright"` or `"fixed"`, and the `[selection] weighting` value (recorded for `"fixed"` too, where it has no effect) |
| `states`, `window_eV` | Candidate states and the window (`null` if none) |
| `n_requested`, `n_selected` | `n_trajectories` and the number selected |
| `stage_failed` | `true` if the stage stopped on too few trajectories (O20); the report is written before the stop, and `trajectories.json` is then not written |
| `candidates`, `accepted` | Number of candidate pairs, and of pairs accepted by the rule **before** the cut to `n_trajectories` |
| `f_max` | Largest oscillator strength among the candidates |
| `weight_max` | $w_{\max}$ of the weighting (in $\mathrm{eV^{-2}}$ for `f_over_dE2`, dimensionless for `f`; for `"fixed"` it is $f_{\max}$) |
| `weight_max_pair` | Bright mode: the pair that sets $w_{\max}$, `{"sample", "state", "excitation_energy_eV", "oscillator_strength", "weight"}`; `null` in fixed mode or without candidates |
| `weight_p99` | Bright mode: the 99th percentile of the candidate weights (linear interpolation); `null` in fixed mode or without candidates |
| `weight_outlier` | `true` if `weight_max_pair.weight` $> 10\times$ `weight_p99`: one pair dominates the normalization (a warning is logged, §8.9.3) |
| `per_state` | For every state: `candidates`, `accepted`, `selected`, `mean_excitation_energy_eV` and `mean_oscillator_strength` of the selected pairs (`null` if none) |

**`selection_report.md`** [IMPLEMENTED]: the same numbers as a page to read: the settings (with the $w_{\max}$ pair in bright mode), the outlier warning and the failure note if they apply, a per-state table, the first 20 trajectories, and the paths of `spectrum.csv` and `excitations.json`. All JSON files of both stages contain only valid JSON: missing values are `null`, never `NaN`.

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
- Each stage is split into **tasks**. `prepare` and `optimize` have one task each. `frequencies` has one task with the analytic Hessian, or $1+6N_{\text{at}}$ tasks with finite differences (§8.7). `sampling` and `selection` have one task each. `excitations` has $1+\lceil N_s/\text{samples\_per\_task}\rceil$ tasks (the reference `ref`, then chunks of `samples_per_task` consecutive samples; $N_s$ = `sampling.n_samples`), which run in parallel up to `max_parallel_tasks`. Planned stages have many tasks; for example, `dynamics` will have one task per trajectory.
- Stages run in order. A stage starts only when the stages it depends on are done.
- Within a stage, up to `max_parallel_tasks` tasks run at the same time. Each task runs as a separate process, so a crashing calculation cannot bring down the controller.

### 7.2 First run
Run `gqteashmd run input.toml`. It finishes with exit code 0 and prints `Workflow complete: …`.

### 7.3 Resuming after a crash, reboot or kill
Simply **run the same command again**:
- Stages that are already done are skipped ("up to date").
- An interrupted stage continues where it stopped.
- A stage whose results directory you deleted by hand is **computed again**, with a warning (`results directory … is missing; recomputing`). This holds for every stage that is not skipped and that is either done or has finished tasks (so a failed or interrupted stage with finished tasks is reset too, not only a done one). Downstream stages that are done are kept, and a recomputed stage listed in `workflow.pause_after` pauses again.
- Temporary files (`.…tmp`) left in `.gqteashmd/` by a crashed controller are removed when the next run starts.
- Tasks that were running when the controller died are checked. Finished ones are kept. Left-over processes are terminated and the tasks are run again.

### 7.4 Stopping and resuming
`gqteashmd stop input.toml`, or Ctrl-C in the terminal running the controller, does the following:
1. No new tasks start.
2. Running tasks get SIGTERM, then SIGKILL after `stop_grace_period_s`. The grace period applies to the worker (the group leader); as soon as the leader has exited, the rest of its process group is killed at once.
3. The run ends with exit code 5 and the message `Stopped. Run the same command again to resume.`

A stop (`gqteashmd stop`, SIGTERM or Ctrl-C) that arrives while the command is starting, after the project lock is taken, is handled the same way: no stage starts, the lock is released, and the run ends with `Stopped. Run the same command again to resume.` and exit code 5. Ctrl-C at any other moment prints `gqteashmd: interrupted` and also exits with code 5.

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

**A refused run changes nothing.** The check is made before anything in the project is written: the input snapshot `.gqteashmd/config.json` and the records of all stages are exactly as before. The only trace is the provenance record of the attempt (§6).

**Optional stages.** `[optimize] enabled = false` skips `optimize`, and `frequencies` then uses the geometry from `prepare`. A disabled stage that already has results keeps them (log line `Stage optimize: skipped (disabled in the input file; earlier results kept)`). Downstream stages see it as skipped, so a `frequencies` stage that was computed from the optimized geometry is stale: the run stops with exit code 3 and the message of §5.4. Enabling `optimize` again with unchanged inputs makes `optimize` and `frequencies` up to date again. To use the unoptimized geometry anyway, run `gqteashmd run INPUT --rerun-from frequencies`.

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

The lock is taken before anything else is created. A run that the lock refuses therefore creates nothing except the `.gqteashmd/` directory (for the lock file), and writes no log and no provenance record. A worker's whole process group is killed when it is stopped or when the worker ends: children that stay in the worker's process group are killed while the controller is running. Children that call `setsid`/`setpgid`, and strays of a worker that finished while the controller was dead, are not.

### 7.8 Review pauses
**[IMPLEMENTED]** `[workflow] pause_after` lists stages after which the run stops so that you can inspect their results:

```toml
[workflow]
pause_after = ["optimize"]         # example: stop after the geometry optimization
```

- When a listed stage **finishes in this run**, the run stops, prints the message below, and exits with code **5**:

  ```text
  Paused after stage 'optimize' as requested (workflow.pause_after). Review its results, then run the same command again to continue.
  ```

- Review the results in the stage directory, then run **the same command again**. The finished stage is "up to date", so the run continues with the next stage and does not pause again.
- A stage that was **skipped** (disabled in the input) never pauses. A stage that was already up to date from an earlier run does not pause either.
- The **last** stage never pauses, because nothing follows it. In this version the stages are `prepare`, `optimize`, `frequencies`, `sampling`, `excitations` and `selection`, so `pause_after = ["optimize"]` lets you inspect `optimized.xyz` before the frequencies are computed, `pause_after = ["frequencies"]` lets you inspect the frequencies and the Molden file before any samples are drawn, `pause_after = ["sampling"]` lets you look at `samples.xyz` before the expensive excitations, and `pause_after = ["excitations"]` lets you look at `spectrum.csv` and `excitations.json` before the initial states are chosen. `pause_after = ["selection"]` **has no effect until `dynamics` exists (P7)**, because `selection` is then the last stage.
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
| An enabled stage needs a newer version of a module | `Stage '…' needs pyscf >= 2.6; version … is installed. Update the conda environment (environment.yml).` | Update the environment |

**Required modules per stage:** `optimize` needs `pyscf>=2.6` and `geometric>=1.0`; `frequencies` needs `pyscf>=2.6` and `h5py>=3.9`. A stage that is disabled is not checked (with `optimize.enabled = false`, geomeTRIC is not needed). If the installed version cannot be read, the version is not checked.

**Scratch space.** PySCF's temporary files are written to `scratch/` inside each task directory, in the project. The check that the project directory is writable therefore also covers the scratch space; there is no separate scratch check.

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
- **§8.1–§8.9** describe **what the current version does** for molecules.
- **§8.10–§8.12 and §8.14** describe the **dynamics core**, implemented in P7a-1 and validated on analytic models; it is not yet connected to molecules (P7b).
- **§8.13** describes the **planned** analysis. Those formulas document the intended implementation; they are not available yet, and details may change after validation against SHARC 4.1 and the literature.

### 8.1 Units and constants
**[IMPLEMENTED]** for input and output:
- Coordinates in XYZ files are in **Ångström (Å)**.
- Charges are in units of the elementary charge $e$.
- Times in `[execution]` are in seconds.

**[IMPLEMENTED]** in the electronic-structure layer (§8.6): coordinates are converted from Å to bohr once, when the XYZ file is read, using the CODATA 2018 values below. Energies are in $E_h$, gradients in $E_h/a_0$ and transition dipoles in $e\,a_0$; the only conversion on output is $E_h \to$ eV for the excitation energies. The vibrational analysis (§8.7) takes masses in amu and Hessians in $E_h/a_0^2$, and reports wavenumbers in cm⁻¹ with the conversion factors of the table below.

**[IMPLEMENTED]** in the `sampling` stage (§8.8) and **[PLANNED]** for the other calculations:
- All internal quantities will be in **Hartree atomic units**, where $\hbar = m_e = e = 4\pi\varepsilon_0 = 1$.
- Conversions happen only when files are read or written, using CODATA values:

| Quantity | Atomic unit | Conversion |
|---|---|---|
| Length | bohr $a_0$ | 1 $a_0$ = 0.529177210903 Å |
| Energy | hartree $E_h$ | 1 $E_h$ = 27.211386245988 eV = 219474.6313632 cm⁻¹ |
| Time | $\hbar/E_h$ | 1 a.u. = 0.0241888432658 fs |
| Mass | electron mass $m_e$ | 1 u (Da) = 1822.888486 $m_e$ |
| Velocity | $a_0 E_h/\hbar$ | 1 a.u. = 2.18769126 × 10⁶ m s⁻¹ |
| Temperature | $E_h/k_B$ | $k_B$ = 3.1668115634556 × 10⁻⁶ $E_h$/K (used by the thermal Wigner distribution, §8.8) |

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
- For `optimize` and `frequencies`: only what determines the S0 surface, namely the `[electronic]` keys that enter the ground-state calculation, plus the stage's own section. The keys that matter only for excited states are left out: `td_conv_tol` and `overlap_norm_kept` always, and `singlets` for `tda` and `tddft` (for `casscf`, `singlets` defines the state average and **is** included). Changing them therefore does not redo `optimize` or `frequencies`. For `frequencies` the key `imaginary_cutoff_cm1` is also left out, because it is only checked after the calculation: if the stage stopped on a too-large imaginary mode and you change the cutoff, the next run repeats only the finalization from the stored task results. Changing `basis`, `functional`, the active space or `fd_step_bohr` does recompute. These stages read no upstream file when the fingerprint is computed; they depend on `prepare` (and on `optimize`) only through the upstream fingerprints. A disabled `optimize` counts as "skipped" in the fingerprint of `frequencies`.
- For `sampling`: the whole `[sampling]` section and `project.seed`, plus the fingerprint of `frequencies` (and so, through it, everything upstream). Any change of these after results exist is a stale-input stop (§7.5).
- For `excitations`: the `[electronic]` section **without** `overlap_norm_kept`, the `[excitations]` keys `include_reference` and `samples_per_task`, `sampling.n_samples`, and the fingerprint of `sampling`. The keys `max_failed_fraction`, `broadening_fwhm_eV` and `spectrum_step_eV` are left out because they act only after the calculation: after changing one of them, the next run repeats only the finalization from the stored task results (new spectrum, new failure check).
- For `selection`: the `[selection]` section **without `allow_fewer`**, and without `weighting` when `mode = "fixed"`; `project.seed`; and the fingerprint of `excitations`. `allow_fewer` acts only in the finalization (the O20 check, §8.9.3): after changing it, the next run repeats only the finalization from the stored task result. `weighting` has no effect in fixed mode. Changing any other `[selection]` key after results exist is a stale-input stop (§7.5): use `--rerun-from selection`.
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

> **§8.6, §8.7 and §8.8 are [IMPLEMENTED].** **Everything after them (§8.9–§8.13) is [PLANNED].** The stages, keys and outputs there do not exist in the current version.

### 8.6 Electronic structure (PySCF)
**[IMPLEMENTED, P3a]** Energies, analytic gradients and transition dipoles. **[IMPLEMENTED, P3b, library only]** SA-CASSCF nonadiabatic couplings, wavefunction overlaps with phase tracking, `compute_more`, restarts (§8.6.8). **[PLANNED]** use of couplings and overlaps in dynamics (P7), TDDFT couplings, triplets and spin–orbit coupling. **[IMPLEMENTED, P4]** the S0 Hessian (analytic for RHF/RKS; SA-CASSCF uses finite differences of gradients in the `frequencies` stage, §8.7).

**What is computed.** For a geometry $\mathbf R$ (nuclear coordinates, bohr) and the states $I = S_0, S_1, \dots$ requested by `electronic.singlets`:

| Quantity | Symbol | Unit | Status |
|---|---|---|---|
| Adiabatic energy | $E_I(\mathbf R)$ | $E_h$ | **[IMPLEMENTED]** |
| Excitation energy | $\Delta E_{0I} = E_I - E_0$ | $E_h$ (eV on output) | **[IMPLEMENTED]** |
| Gradient | $\nabla_{\mathbf R} E_I$, a $3N_{\text{at}}$-vector | $E_h/a_0$ | **[IMPLEMENTED]** (used by `optimize` and `frequencies`; `single-point` does not request it) |
| Transition dipole | $\boldsymbol\mu_{0I}$ | $e\,a_0$ | **[IMPLEMENTED]** |
| Oscillator strength | $f_{0I}$ | — | **[IMPLEMENTED]** |
| Nonadiabatic couplings $\mathbf d_{IJ} = \langle \Psi_I \vert \nabla_{\mathbf R} \Psi_J \rangle$ (CASSCF); wavefunction overlaps $S_{IJ}(t) = \langle \Psi_I(t-\Delta t) \vert \Psi_J(t) \rangle$ (TDA/TDDFT) | $\mathbf d_{IJ}$, $S_{IJ}$ | $a_0^{-1}$, — | **[IMPLEMENTED, P3b]** $\mathbf d_{IJ}$ for CASSCF only; $S_{IJ}$ for all methods; library only, not reachable from the CLI (§8.6.8) |
| Hessian $\partial^2 E_0/\partial x_i\partial x_j$ | $H_{ij}$ | $E_h/a_0^2$ | **[IMPLEMENTED, P4]** S0 only: analytic for `tda`/`tddft` (SCF Hessian, backend method `ground_state_hessian`); not for `casscf` (finite differences, §8.7) |

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
**[IMPLEMENTED, P4]** The `frequencies` stage computes the Cartesian Hessian of the S0 energy at the geometry of `optimize` (or of `prepare` when `optimize` is disabled) and analyses it in the harmonic approximation. The analysis (`gqteashmd.vib.harmonic`) is our own implementation of the standard treatment of Wilson, Decius and Cross [8] in the Eckart frame [9]. It needs only the Hessian, the geometry and the masses.

| Symbol | Meaning | Unit |
|---|---|---|
| $N_{\text{at}}$ | Number of atoms | — |
| $x_i$ ($i = 1,\dots,3N_{\text{at}}$) | Cartesian coordinates of the geometry $\mathbf R$ | $a_0$ |
| $m_i$ | Mass of the atom that owns coordinate $i$ (§ masses below) | amu |
| $E_0$, $g_i = \partial E_0/\partial x_i$ | S0 energy and its gradient | $E_h$, $E_h/a_0$ |
| $H_{ij} = \partial^2 E_0/\partial x_i\partial x_j$ | Cartesian Hessian | $E_h/a_0^2$ |
| $h$ | Finite-difference step (`frequencies.fd_step_bohr`) | $a_0$ |
| $\mathbf e_k$ | Unit vector along Cartesian coordinate $k$ | — |
| $H^{\text{mw}}$ | Mass-weighted Hessian | $E_h\,a_0^{-2}\,\text{amu}^{-1}$ |
| $\mathbf D$, $n_{\text{ext}}$ | Orthonormal translations and rotations ($3N_{\text{at}}\times n_{\text{ext}}$), and their number | —, — |
| $\mathbf Q$ | Orthonormal basis of the complement of $\mathbf D$ ($3N_{\text{at}}\times(3N_{\text{at}}-n_{\text{ext}})$) | — |
| $\lambda_k$, $\mathbf v_k$ | Eigenvalues and eigenvectors of $\mathbf Q^{\mathsf T}H^{\text{mw}}\mathbf Q$ | $E_h\,a_0^{-2}\,\text{amu}^{-1}$, — |
| $u$ | One amu in electron masses, $1822.888486$ | $m_e$/amu |
| $\tilde\nu_k$ | Wavenumber of mode $k$ | cm⁻¹ |
| $\mathbf l_k$ | Mass-weighted normal mode, unit norm | — |
| $\mathbf x_k$ | Cartesian displacement of mode $k$, scaled to unit length | — |
| $\mu_k$ | Reduced mass of mode $k$ | amu |

**Step 1: the Hessian.** Two ways, chosen by `frequencies.method`:

- *Analytic* (one task `hessian`): PySCF's analytic SCF Hessian (RHF or RKS) at $\mathbf R$.
- *Finite differences* (SA-CASSCF, which has no analytic Hessian in PySCF; or `method = "finite_difference"`): central differences of the S0 gradient. For each Cartesian coordinate $k$, the gradient is computed at $\mathbf R + h\,\mathbf e_k$ and $\mathbf R - h\,\mathbf e_k$:

$$
H_{ik} \;\approx\; \frac{g_i(\mathbf R + h\,\mathbf e_k) - g_i(\mathbf R - h\,\mathbf e_k)}{2h},
\qquad
\mathbf H \leftarrow \tfrac12\left(\mathbf H + \mathbf H^{\mathsf T}\right).
$$

  The error of the differences is of order $h^2$, and gradient noise is divided by $2h$; the default $h = 0.005\,a_0$ is a compromise between the two. The stage has $1 + 6N_{\text{at}}$ independent tasks: one `reference` task (the energy and gradient at $\mathbf R$, used for the stationarity check and `energy_hartree`) and $2\times3N_{\text{at}}$ displaced gradients. Every task builds a **fresh** electronic-structure object, so the orbitals of one displacement cannot influence another. The tasks run in parallel up to `max_parallel_tasks` (§8.4).

In both cases the Hessian is symmetrized before the analysis (for the analytic Hessian this removes only rounding noise).

**Step 2: mass weighting.**

$$
H^{\text{mw}}_{ij} \;=\; \frac{H_{ij}}{\sqrt{m_i\,m_j}} .
$$

**Step 3: removing translations and rotations (Eckart frame).** With atom positions $\mathbf r_A$, the centre of mass $\mathbf r_{\text{cm}} = \sum_A m_A\mathbf r_A/\sum_A m_A$, and the three Cartesian axes $a$, the mass-weighted vectors of infinitesimal translation and rotation have, for atom $A$, the components

$$
\left(\mathbf T_a\right)_A = \sqrt{m_A}\;\hat{\mathbf e}_a ,
\qquad
\left(\mathbf R_a\right)_A = \sqrt{m_A}\;\hat{\mathbf e}_a \times (\mathbf r_A - \mathbf r_{\text{cm}}) .
$$

Whether the molecule is linear is decided physically, from the principal moments of inertia $I_1\le I_2\le I_3$ of the mass-weighted, centred coordinates: the molecule is treated as **linear when $I_1/I_3 < 10^{-3}$**, and as a single atom when all moments vanish. For a linear (or nearly linear) molecule the rotation about the principal axis with the small moment is dropped, and the rotations about the other two axes are kept; for a nonlinear molecule all three are kept. This gives $n_{\text{ext}} = 6$ vectors for a nonlinear molecule, 5 for a linear one and 3 for a single atom. The vectors are orthonormalized by a singular-value decomposition and form the columns of $\mathbf D$. A slightly bent triatomic such as CO₂ at 179.9° therefore keeps its 4 vibrational modes (measured with RHF/STO-3G: 566.4, 566.4, 1435.1 and 2535.4 cm⁻¹, external residual 11 cm⁻¹), where a rank threshold on the rotation vectors would have lost the second bending mode. An orthonormal basis $\mathbf Q$ of the complement of $\mathbf D$ is built by QR decomposition. The vibrational problem is then

$$
\left(\mathbf Q^{\mathsf T} H^{\text{mw}} \mathbf Q\right)\mathbf v_k = \lambda_k \mathbf v_k ,
\qquad k = 1,\dots, 3N_{\text{at}} - n_{\text{ext}} .
$$

**Step 4: frequencies.** With the mass in amu and the Hessian in atomic units, the angular frequency in atomic units is $\omega_k = \sqrt{\lambda_k/u}$, and the wavenumber follows from $1\,E_h = 219474.6313632$ cm⁻¹:

$$
\tilde\nu_k \;=\; \operatorname{sign}(\lambda_k)\,\sqrt{\frac{\lvert\lambda_k\rvert}{u}}\;\times\;219474.6313632\ \text{cm}^{-1} .
$$

A negative $\lambda_k$ is an **imaginary frequency**; it is reported as a **negative** wavenumber, in the JSON, HDF5 and Molden files. Modes are numbered from 1 in ascending order of $\tilde\nu$, so the most negative (most unstable) mode comes first.

**Step 5: normal modes and reduced masses.** The mass-weighted mode is $\mathbf l_k = \mathbf Q\mathbf v_k$ (unit norm). Its Cartesian displacement is $\mathbf x_k \propto M^{-1/2}\mathbf l_k$, with $M = \operatorname{diag}(m_i)$, scaled to unit length (`modes_cartesian`, and the Molden file). The reduced mass uses the Gaussian/PySCF convention:

$$
\mu_k \;=\; \frac{1}{\sum_i l_{k,i}^2/m_i} .
$$

For example, the stretch of H₂ gets $\mu = 1.0078$ amu, not the textbook $m/2 = 0.504$ amu, so reduced masses can be compared directly with Gaussian and PySCF output. The water modes in §5.4 have $\mu \approx 1.05$–$1.08$ amu.

**External residual.** The motions removed in Step 3 would have zero frequency for an exact Hessian at a stationary point. The program also diagonalizes $\mathbf D^{\mathsf T}H^{\text{mw}}\mathbf D$ and reports the largest $\lvert\tilde\nu\rvert$ (same conversion) as `external_residual_cm1`. It measures how far the Hessian is from rotational and translational invariance. At optimized geometries we measured 25–41 cm⁻¹ with RHF and about 22 cm⁻¹ with B3LYP on the default grid (`grid_level` 3). The external motions are projected out before diagonalization, so a small residual does not contaminate the frequencies. A large residual means the Hessian or the geometry is inaccurate. The number of modes itself depends on the linearity decision above: for a molecule between the two cases (a bent molecule with $I_1/I_3$ just above $10^{-3}$) one of the modes is a genuine, very soft bending vibration.

**Masses [IMPLEMENTED, P4; open item O5 resolved].** The masses $m_i$ are those of the **most abundant isotope** of each element, in amu, for H to Rn (module `gqteashmd.chem.masses`; values from the NIST atomic weights and isotopic compositions as tabulated in PySCF; for Tc, Pm, Po, At and Rn, which have no stable isotope, the isotope listed in the module). Examples: H 1.007825, C 12.000000, N 14.003074, O 15.994915. They are stored in `frequencies.h5` (`masses`). There is no input key for isotopic substitution yet. The Wigner sampling of P5 will use the same masses.

**Why only S0, and how the surface is defined.** The Hessian is that of the ground-state surface on which the dynamics will start.
- For `tda` and `tddft` the S0 energy is the SCF energy of the restricted reference. Excited states do not enter it, so the stages ask the backend for S0 only (`singlets` is set to 1 internally for these calculations) and the Hessian is the analytic SCF Hessian (RHF for `functional = "hf"`, RKS otherwise).
- For `casscf` the S0 energy is the lowest root of the **state-averaged** calculation. The state average defines the orbitals and therefore the surface, so the stages keep your `singlets` (an equal-weight state average). PySCF has no analytic SA-CASSCF Hessian, so the Hessian comes from finite differences of the SA-CASSCF S0 gradient (Step 1), which costs $1 + 6N_{\text{at}}$ gradient evaluations (37 for ethylene).
- `optimize` uses the same S0 settings, with one electronic-structure object for the whole optimization path (successive geometries start from the previous orbitals).

**Stopping and warning policy.** `frequencies` writes all its files first and then checks the result:

| Condition | What happens |
|---|---|
| Some $\tilde\nu_k < -$ `imaginary_cutoff_cm1` (default 20 cm⁻¹) | The stage **fails** (exit code 4). The message lists every such mode, for example `mode 1: -2329.0 cm^-1, mode 2: -2329.0 cm^-1`. The files are on disk, so you can look at the modes in `frequencies.molden`. The structure is not a minimum: re-optimize (for example `convergence = "GAU_TIGHT"`), check the starting structure, or raise the cutoff if the mode is a known numerical artefact |
| $-$cutoff $\le \tilde\nu_k < 0$ | **Warning** `small imaginary modes (within the cutoff): …`; the modes are recorded in `small_imaginary_modes` |
| `external_residual_cm1` > 100 cm⁻¹ | **Warning** `translation/rotation contamination up to … cm^-1 (Hessian or geometry inaccurate)` |
| Largest absolute gradient component > $10^{-3}\,E_h/a_0$ | **Warning** `max |gradient| = … Eh/bohr: not a stationary point; frequencies are approximate` |

Warnings reach the console, `logs/gqteashmd.log` and the `warnings` list of `frequencies.json`; the run continues. The master plan's earlier idea of stopping on a wrong number of near-zero modes was replaced by the external-residual warning, together with the physical linearity criterion above, which fixes the mode count.

**Console form of the imaginary-mode stop.** The controller prefixes the stage message, so the line you see is `gqteashmd: error: Stage 'frequencies' could not finalize its results: The structure is not a minimum: 2 imaginary mode(s) beyond 20 cm^-1 (mode 1: -2329.0 cm^-1, mode 2: -2329.0 cm^-1). Re-optimize …` (exit code 4).

**Consistency checks of the finite-difference route.** For `finite_difference` the stage returns the energy of every displaced calculation as well, and checks, before writing the vibrational files, that the energies and the gradients describe the same surface: for every Cartesian coordinate $k$ it compares $(E_{+k}-E_{-k})/2h$ with the reference gradient $g_k$. If the largest difference exceeds $10^{-4}\,E_h/a_0$ the stage fails (exit code 4) and the message names the worst coordinate (atom and axis). The usual cause is that a displaced SA-CASSCF calculation converged to a different electronic solution; check the active space, or reduce `fd_step_bohr`. When the geometry comes from `optimize`, the stage also requires the S0 energy at that geometry to agree with the energy recorded by `optimize` to within $10^{-6}\,E_h$ (for both the analytic and the finite-difference route); otherwise it fails because the frequencies would not belong to the optimized surface. In both failure cases only `frequencies.json` is written, with the measured numbers. The measured values are always recorded in `frequencies.json` (`fd_consistency_residual_hartree_per_bohr`, `optimized_energy_difference_hartree`, `consistency_problems`). Measured: water RHF/STO-3G, $h = 0.005\,a_0$, $2.8\times10^{-6}\,E_h/a_0$; SA-CASSCF(2,2)/STO-3G ethylene, $4.6\times10^{-6}\,E_h/a_0$.

**One atom.** `optimize` writes its outputs without calling geomeTRIC (the geometry is unchanged, `n_gradients` is 1); `frequencies` returns 0 modes (`n_external` = 3) without an error.

**Cost.** Analytic: one SCF plus one Hessian. Finite differences: $1 + 6N_{\text{at}}$ SA-CASSCF gradient tasks; in the test suite, SA-CASSCF(2,2)/STO-3G ethylene (37 tasks) takes about 84 s. `optimize` for water at HF/STO-3G needed 5 gradient evaluations (about 0.9 s of computation).

**Assumptions.** Harmonic approximation; the Born–Oppenheimer S0 surface of §8.6; the geometry is a stationary point (checked by the gradient warning above). No anharmonicity, no thermochemistry (zero-point energy, entropy) and no frequency scaling are computed.

**Validation (measured).**

| Comparison | Result |
|---|---|
| Our harmonic analysis vs PySCF `thermo.harmonic_analysis` for the same Hessian, geometry and masses (water; linear CO₂) | Frequencies agree to $10^{-6}$ cm⁻¹ |
| Analytic SCF Hessian vs finite differences of analytic gradients ($h = 10^{-3}$; water, HF/STO-3G) | Largest difference $6.0\times10^{-7}\,E_h/a_0^2$ |
| `frequencies` stage, water RHF/STO-3G at a tightly optimized geometry (largest gradient $1.7\times10^{-7}\,E_h/a_0$): analytic vs finite differences ($h = 0.005\,a_0$) | Analytic 2170.05, 4140.00, 4391.07 cm⁻¹; finite differences 2170.00, 4140.02, 4391.08 cm⁻¹ (test tolerance 0.3 cm⁻¹; measured difference 0.05 cm⁻¹) |
| ORCA 6.1.1, water RHF/6-31G\*, fixed geometry, ORCA's masses | Ours 1825.981, 4055.472, 4173.836 cm⁻¹; ORCA 1826.02, 4055.42, 4173.83 cm⁻¹ (differences −0.039, +0.052, +0.006 cm⁻¹) |
| Linear water (a saddle point of the bend) | Two imaginary bending modes at −2329.0 cm⁻¹; the stage stops as designed |
| SHARC 4.1 `wigner.py` (run as a black box) reading our `frequencies.molden` | Accepted unchanged |

The PySCF comparison used the same masses and so tests the projection and the diagonalization alone; the ORCA comparison used ORCA's own masses and tests the whole chain. The example of §5.4 is not tightly converged (largest gradient $8.6\times10^{-5}\,E_h/a_0$), so its frequencies differ by up to 0.5 cm⁻¹ from the tightly converged values in the table.

### 8.8 Wigner sampling of initial conditions
**[IMPLEMENTED]** (P5). The stage `sampling` reads `frequencies.h5` (§6) and draws $N_s$ = `n_samples` initial geometries with velocities. Atomic units are used inside the calculation ($\hbar = 1$, Hartree energies, bohr, electron masses $m_e$, §8.1); the files carry the units of §6.

**Symbols.** $N_{\text{at}}$ atoms with masses $m_A$ (in $m_e$; the masses of `frequencies.h5` in amu times the conversion of §8.1) and equilibrium Cartesian geometry $\mathbf x_{\text{eq}}$ (the geometry that `frequencies` analysed, a $3N_{\text{at}}$-vector). $\mathbf M$ is the diagonal $3N_{\text{at}}\times3N_{\text{at}}$ matrix of the masses, each repeated three times. For every vibrational mode $k$: $\tilde\nu_k$ is its wavenumber in cm⁻¹ from `frequencies.h5`, $\omega_k = \tilde\nu_k / 219474.63\ldots$ is its angular frequency in hartree (the conversion of §8.1; $\hbar = 1$), and $\mathbf L_k$ is its mass-weighted normal mode (a $3N_{\text{at}}$-vector of unit length, dataset `modes_mass_weighted`). $T$ is `temperature_K` and $k_B$ the Boltzmann constant ($3.1668\times10^{-6}\,E_h/\text{K}$).

**Modes that are sampled.** Mode $k$ is sampled if $\tilde\nu_k \ge$ `low_frequency_cutoff_cm1`. Every other mode, including the small imaginary modes (negative $\tilde\nu$ within the `frequencies` cutoff, §8.7), is **not sampled**: its coordinate stays at the equilibrium value and its velocity component is zero. The stage logs a warning that lists the wavenumbers of the unsampled modes and writes them to `samples.json`. If *no* mode is sampled, the stage fails (§9). A floppy mode below the cutoff would otherwise have a huge amplitude $\sigma_Q \propto \omega^{-1/2}$ in a harmonic picture that is not valid for it; with the default 10 cm⁻¹ a mode is sampled from about 10 cm⁻¹ upwards.

**Distribution.** Each sampled mode is an independent harmonic oscillator in the mass-weighted normal coordinate $Q_k$ (unit $\sqrt{m_e}\,a_0$) and its conjugate momentum $P_k$ (unit $\sqrt{m_e}\,a_0$ per atomic unit of time). Their thermal Wigner distribution (Hillery et al., 1984) is a product of independent Gaussians with zero mean:

$$
W_k(Q_k, P_k) \;\propto\; \exp\!\left(-\frac{Q_k^2}{2\sigma_{Q,k}^2} - \frac{P_k^2}{2\sigma_{P,k}^2}\right),
\qquad
\sigma_{Q,k}^2 = \frac{\hbar\,c_k}{2\omega_k},
\qquad
\sigma_{P,k}^2 = \frac{\hbar\,\omega_k\,c_k}{2},
\qquad
c_k = \coth\!\left(\frac{\hbar\omega_k}{2k_BT}\right).
$$

The factor $c_k \ge 1$ measures the thermal occupation of mode $k$. At $T = 0$, $\coth \to 1$, $c_k = 1$ and $W_k$ is the Wigner function of the vibrational ground state, $\sigma_Q^2 = \hbar/2\omega$, $\sigma_P^2 = \hbar\omega/2$. Unlike a classical Boltzmann distribution, the Wigner distribution keeps the zero-point motion at every temperature, and it is exact for a harmonic oscillator (open item O4, decided on 2026-11-08: the exact harmonic thermal Wigner function, not a classical or a rescaled approximation).

**Harmonic energy, its mean and variance.** The harmonic energy of a sample is $E_{\text{harm}} = \sum_k \bigl(\tfrac12 P_k^2 + \tfrac12\omega_k^2 Q_k^2\bigr)$ (sum over the sampled modes; it is computed from $Q$ and $P$ before the momentum removal below). Each mode contributes with mean $\tfrac12\hbar\omega_k c_k$, and the contribution of one mode is exponentially distributed in the energy, so its variance is the square of its mean. Therefore

$$
\langle E\rangle = \sum_k \tfrac12\hbar\omega_k c_k,
\qquad
\operatorname{Var}(E) = \sum_k \bigl(\tfrac12\hbar\omega_k c_k\bigr)^2 .
$$

At $T = 0$, $\langle E\rangle = \tfrac12\sum_k\hbar\omega_k$ is the zero-point energy. **Check performed by the stage (exact for every $N_s$, $T$ and molecule).** For a sampled mode $k$, $E_k=\tfrac12P_k^2+\tfrac12\omega_k^2Q_k^2$ with Gaussian $Q_k$, $P_k$ of the widths above, so $E_k/\langle E_k\rangle$ follows the exponential distribution with mean 1, with $\langle E_k\rangle=\tfrac12\hbar\omega_k c_k$. Over the $N_s$ samples, $S_k=\sum_s E_{k,s}/\langle E_k\rangle$ is therefore exactly Gamma distributed with shape $N_s$ and scale 1, independently for each mode. The stage computes the two-sided tail probability $p_k = 2\min\bigl(F(S_k), 1-F(S_k)\bigr)$ ($F$ the Gamma($N_s$, 1) distribution function) and fails if $\min_k p_k < 10^{-9}/n_{\text{sampled}}$ (Bonferroni over the sampled modes). A correct program therefore fails this check with a probability of at most $10^{-9}$ per run, whatever $N_s$ (even $N_s=1$) and whatever the number of modes; a mode whose widths are wrong by 10% is detected at $N_s=2000$. A failure makes the stage fail with exit code 4; the message gives the mode, $S_k$, $N_s$ and $p_k$ and says that it indicates a program error. `samples.json` is written first, with the $S_k$, the smallest $p$ and the limit.

**Cartesian geometry and velocities.**

$$
\mathbf x = \mathbf x_{\text{eq}} + \mathbf M^{-1/2}\sum_k \mathbf L_k Q_k,
\qquad
\mathbf v = \mathbf M^{-1/2}\sum_k \mathbf L_k P_k ,
$$

with the sums over the sampled modes. The geometries are not moved or rotated afterwards: all samples keep the orientation of the equilibrium geometry.

**Removal of linear and angular momentum.** The $\mathbf v$ above contain a small net translation and rotation, because the harmonic modes are projected from the translations and rotations only at the equilibrium geometry. For each sample, with $\mathbf r_A$ the position of atom $A$ relative to the centre of mass:

1. the centre-of-mass velocity $\sum_A m_A\mathbf v_A / \sum_A m_A$ is subtracted from every $\mathbf v_A$;
2. the angular momentum $\mathbf L_{\text{ang}} = \sum_A m_A\,\mathbf r_A\times\mathbf v_A$ and the inertia tensor $\mathbf I = \sum_A m_A\bigl(r_A^2\,\mathbf 1 - \mathbf r_A\mathbf r_A^{\mathsf T}\bigr)$ give the angular velocity $\boldsymbol\Omega = \mathbf I^{-1}\mathbf L_{\text{ang}}$ (pseudo-inverse, so linear molecules work), and $\boldsymbol\Omega\times\mathbf r_A$ is subtracted from every $\mathbf v_A$.

The velocities are **not rescaled** afterwards. The kinetic energy that the removal took away, $T_{\text{removed}} = T_{\text{before}} - T_{\text{after}}$, is stored for every sample (`removed_kinetic_energy`). Because of the removal, the mean kinetic energy of the samples is slightly below the value $\tfrac14\sum_k\hbar\omega_k c_k$ that the distribution alone gives (water: $1.5\times10^{-5}\,E_h$ of $0.0122\,E_h$ in §5.5; the stage does not add the energy back). **Linear molecules:** for a bent sample of a linear molecule (e.g. CO₂), the axial component of the removed angular velocity takes out the vibrational angular momentum of the degenerate bending pair. This is what zero total angular momentum ($J=0$) requires, but it can be a sizeable fraction of the bending kinetic energy, much more than the small removal seen for water, so `removed_kinetic_energy` can be large for linear molecules. **Check performed by the stage:** after the removal, the largest $\lvert\mathbf P\rvert$ over the samples must be below $10^{-8}$ and the largest $\lvert\mathbf L_{\text{ang}}\rvert$ below $10^{-6}$ (atomic units); otherwise the stage fails with exit code 4 and says that this indicates a program error.

**Seed hierarchy.** All random numbers derive from `project.seed` $s$ in a fixed order: the stage generator is a NumPy `SeedSequence` built from $s$ and the CRC-32 checksum of the stage name (`sampling`); task number $i$ of the stage uses the $i$-th child of that sequence. `sampling` has one task, number 0, which draws first all $Q$ (a $N_s\times n_s$ array of standard normal numbers, scaled by $\sigma_Q$) and then all $P$. Consequences: the samples depend only on `project.seed`, `n_samples`, `temperature_K`, `low_frequency_cutoff_cm1` and the contents of `frequencies.h5` (frequencies, masses, equilibrium geometry and normal modes); the result is the same after a crash and a restart, on another day, and whatever other stages ran; and other stages have independent random streams.

**Assumptions and limits.**
- *Harmonic approximation:* the potential is quadratic around the geometry of `frequencies`. Anharmonic effects (stretches of X–H bonds at high temperature, torsions, ring puckering) are not described; large-amplitude motions can give geometries that are far from the equilibrium one.
- *Independent modes:* there are no correlations between modes. The Wigner function of the harmonic Hamiltonian factorizes exactly in normal modes, but not for an anharmonic one.
- *Cutoff:* modes below `low_frequency_cutoff_cm1` are not sampled, so their zero-point motion and thermal energy are missing from the sample.
- *Classical nuclei afterwards:* each sample is a classical point in phase space; its energy is not an eigenvalue and, on the real surface, the total energy of a sample differs from $E_{\text{harm}}$.
- The samples are drawn on the **S0 surface** only; the excited-state surfaces are used from `excitations` on (P6).

**Validation (measured).** The sampler was compared with SHARC 4.1 `wigner.py` (run as a black box) on water RHF/STO-3G (frequencies 2170.05, 4140.00, 4391.07 cm⁻¹ at the tightly optimized geometry of §8.7, analytic Hessian, the same Molden file for both programs), at $T = 0$: SHARC with 2000 samples (`-r 7`), `gqteashmd` with 20000 samples (seed 1). The differences of the means and of the standard deviations are in units of the combined standard error (for the standard deviations from the sample fourth moment, $\mathrm{SE}(s)^2\approx(m_4-s^4)/(4ns^2)$); $p$ is the two-sample Kolmogorov–Smirnov probability that both samples come from one distribution. The test asserts $\lvert\Delta\text{mean}\rvert\le4$ standard errors, $\lvert\Delta\text{std}\rvert\le4$ standard errors and $p>10^{-3}$ for every row.

| Quantity | `gqteashmd` mean / std | SHARC mean / std | mean difference / SE | std difference / SE (ratio) | KS $p$ |
|---|---|---|---|---|---|
| O–H₁ distance (bohr) | 1.873199 / 0.121838 | 1.872796 / 0.122546 | +0.14 | −0.36 (0.994) | 0.966 |
| O–H₂ distance (bohr) | 1.875603 / 0.120766 | 1.875007 / 0.122018 | +0.21 | −0.63 (0.990) | 0.275 |
| H–O–H angle (°) | 100.013 / 7.442 | 100.136 / 7.265 | −0.72 | +1.42 (1.024) | 0.280 |
| Kinetic energy after momentum removal ($E_h$) | 0.012225 / 0.010392 | 0.011957 / 0.010325 | +1.10 | +0.20 (1.006) | 0.443 |
| Kinetic energy before momentum removal ($E_h$) | 0.012250 / 0.010414 | 0.011983 / 0.010347 | +1.10 | +0.20 (1.006) | 0.439 |

The harmonic energy of SHARC's output, $\tfrac12\sum_k\hbar\omega_k$ = 0.0243789326 $E_h$, equals ours to better than $10^{-8}\,E_h$; the atomic masses agree to better than $10^{-6}$ amu; our maximum $\lvert\mathbf P\rvert$ and $\lvert\mathbf L_{\text{ang}}\rvert$ are $4\times10^{-15}$ and $5\times10^{-15}$ atomic units (SHARC's printed $\lvert\mathbf L_{\text{ang}}\rvert$ is below $5.4\times10^{-5}$, the rounding of its 8-decimal output). All five comparisons are consistent with sampling noise (largest mean difference 1.1 standard errors, largest standard-deviation difference 1.4). **A note on SHARC's kinetic-energy line:** the `Ekin` line in each SHARC `initconds` entry is the kinetic energy *before* the momentum removal; it differs from the energy of the velocities printed next to it by up to $6.4\times10^{-4}\,E_h$ (mean $2.5\times10^{-5}\,E_h$), so the comparison above is "before with before" and "after with after". **Temperature 1000 K.** The same comparison with SHARC `wigner.py -t 1000` (2000 samples, `-r 7`) against `gqteashmd` with `temperature_K = 1000` (20000 samples) is asserted as well (second parameter of `tests/validation/test_sharc_wigner_statistics.py`). Measured differences in standard errors (std ratio): O–H₁ distance mean −0.81, std −1.10 (0.982); O–H₂ +0.78, −0.79 (0.987); angle +0.28, +0.80 (1.014); kinetic energy after removal +1.57, +0.82 (1.028); before removal +1.56, +0.81 (1.028); KS $p$ between 0.13 and 0.36. The `Eharm` header of SHARC's file is the zero-point energy $\tfrac12\sum_k\hbar\omega_k$ (0.0243789326 $E_h$) at 1000 K as well, so it is not the thermal mean (ours: 0.0249198 $E_h$ at 1000 K); the test asserts it only against the zero-point energy.

**SHARC at higher temperature differs from the exact thermal Wigner function.** At 1000 K the thermal factors of water are small ($c_k$ = 1.09, 1.005, 1.004), so the comparison above cannot resolve width differences of a few per cent. At 3000 K ($c_k$ = 2.09, 1.32, 1.28) it can, and SHARC's samples are **not** the exact thermal Wigner distribution of §8.8. Projected onto our normal modes, $Q_k = \mathbf l_k\cdot\mathbf M^{1/2}(\mathbf x-\mathbf x_0)$, 10000 SHARC samples (`-t 3000`, seeds 11 and 12) give $\mathrm{var}(Q_k)/\sigma_{Q,k}^2$ = 0.92–0.99 (standard error 0.014; the bend 0.956 and 0.923) and a kurtosis of 3.04–3.12 (Gaussian: 3, standard error 0.05). The same projection at $T = 0$ gives 0.997–1.003 and 2.94–2.97, i.e. SHARC matches the exact Gaussian there. In the comparison test, the 3000 K case therefore fails (H–O–H angle KS $p = 7\times10^{-4}$, std ratio 1.042, +2.5 standard errors; SHARC's mean kinetic energy before removal, 0.017131 $E_h$, is 2.1 standard errors below the exact $\tfrac14\sum_k\hbar\omega_k c_k = 0.017775\,E_h$, ours 0.017830 $E_h$ is 0.5 above). It is kept as an expected failure (`xfail`, strict, deterministic seeds) that documents the difference. `gqteashmd` samples the exact distribution, as decided (O4): its widths and the thermal mean energy are verified by unit tests, and every run checks the per-mode Gamma statistic. Practical consequence: thermal samples of `gqteashmd` are slightly broader than SHARC's at high temperature; at room temperature, where $c_k \approx 1$ for all but the softest modes, the two agree.

Other checks of the sampler (exact Gaussian widths, the $T > 0$ energy, zero total momenta, linear molecules, reproducibility) are unit tests of the package.

### 8.9 Initial excitation
**[IMPLEMENTED]** (P6). The stage `excitations` computes the vertical excitations at the reference geometry and at every Wigner sample (§8.9.1) and builds the absorption spectrum (§8.9.2). The stage `selection` then chooses the initial electronic state of every trajectory (§8.9.3). §8.9.4 gives the comparison with SHARC 4.1. Atomic units are used inside the calculation and the files carry the units of §6, except that excitation energies are in eV (the only conversion is $E_h \to$ eV with the CODATA factor of §8.1).

#### 8.9.1 Vertical excitations (`excitations`)
At a geometry $\mathbf R$ the electronic-structure backend (§8.6) gives the energies $E_0, E_1, \dots, E_{n-1}$ of the $n$ = `electronic.singlets` singlet states, in order of energy, and the transition dipoles $\boldsymbol\mu_{0I}$. With the formulas of §8.6.5 and §8.6.6 this gives, for I = S1, ..., S($n-1$),

$$
\Delta E_{0I} = E_I - E_0,
\qquad
f_{0I} = \tfrac23\,\Delta E_{0I}\,\lvert\boldsymbol\mu_{0I}\rvert^2 \quad(\text{atomic units}).
$$

**Procedure.**

1. The stage plans one task `ref` (if `include_reference = true`) for the equilibrium geometry of `samples.h5`, and tasks `s00001`, `s00002`, ... of `samples_per_task` consecutive samples each (the task name is the number of its first sample).
2. A task builds **one backend** with the **full `[electronic]` settings** (the same method, basis, functional, active space and number of states as `single-point`; the logs are in its `electronic.log`, the scratch files in its `scratch/` directory), and calls `reset()` **before every sample**. A reset removes everything the backend remembers of the previous geometry (orbitals, density, phases), so a sample's result does not depend on which samples were computed before it in the same task. The $k$-th sample uses `positions[k-1]` of `samples.h5`. State $S_n$ is the $n$-th root **at that geometry**; no state is tracked between samples.
3. **A sample whose electronic structure fails** (an SCF, CASSCF or linear-response calculation that exhausts its retry ladder, §8.6; an `ElectronicStructureError`) is **recorded as failed, with the message**, and the task goes on with the next sample. The task itself then still counts as done. **A sample whose results cannot be used is recorded as failed in the same way:** a non-finite energy, transition dipole or oscillator strength (*non-finite energy of S1 (nan Eh)*, *non-finite transition dipole of S1 …*, *non-finite oscillator strength of S1 …*), or an excited state at or below S0 (*excitation energy of S2 is not positive (-2.72114 eV)*; for example an unstable reference in TDA, or a collapsed root). So every successful sample has finite values and $\Delta E_{0I}>0$ for every state, which the spectrum (§8.9.2) and the selection weights (§8.9.3) rely on. The same rule applies to the reference, whose failure stops the stage (step 4). Retryable errors (a `MemoryError` or `OSError` inside PySCF) and crashes fail the task instead; the task is retried up to `execution.max_attempts` times and **repeats all its samples** (§8.4).
4. **Failure tolerance.** After all tasks are done, `finalize` counts the failed samples, including all samples of a task that failed completely. If more than `max_failed_fraction` $\times N_s$ samples failed (equality is allowed), the stage stops with exit code 4 and lists the first five failures. The same fraction is applied to the tasks by the orchestrator (§8.4). **If the reference geometry failed (and `include_reference = true`), the stage always stops**, because the reference is the check on the sampled values. Samples that failed are excluded from the spectrum and from the selection; they stay in `excitations.h5` as `NaN` rows with `ok = 0`. **Before either stop, `finalize` writes `excitations.json`** with `n_ok`, the list of failed samples, `reference_error` and the message in `error` (`states` is then `null`; `excitations.h5` and `spectrum.csv` are not written), so the failures can be inspected. Raising `max_failed_fraction` repeats only the finalization (no `--rerun-from` needed); `include_reference = false` changes the fingerprint and needs `--rerun-from excitations`, **which recomputes every sample**.
   Before computing, every task checks that `samples.h5` holds `sampling.n_samples` samples; otherwise it fails (not retried) with *samples.h5 holds K samples but [sampling] n_samples = N; rerun the sampling stage with --rerun-from sampling*.
5. `finalize` writes `excitations.h5`, `spectrum.csv` and `excitations.json` (§6), and logs for each state the mean and standard deviation of $\Delta E$ and $f$ over the successful samples.

The sample is computed at the geometry that `sampling` produced; the velocities are not used. The reference is not part of the spectrum and not a candidate for the selection.

#### 8.9.2 The absorption spectrum (nuclear-ensemble method)
**Symbols.** $i$: a successful sample, $i = 1,\dots,N_{\text{ok}}$. $n$: an excited state (S1, S2, ...). $\Delta E_{in}$: its excitation energy at sample $i$, in eV. $f_{in}$: its oscillator strength. $E$: the photon energy of the spectrum, in eV. $\delta$: `broadening_fwhm_eV` (full width at half maximum, eV). $s = \delta/(2\sqrt{2\ln2})$: the corresponding standard deviation. $g(x)=\dfrac{1}{s\sqrt{2\pi}}\exp\!\Bigl(-\dfrac{x^2}{2s^2}\Bigr)$: the normalized Gaussian (unit area, in eV$^{-1}$). $e$, $\hbar$, $m_e$, $c$, $\varepsilon_0$: elementary charge, reduced Planck constant, electron mass, speed of light, vacuum permittivity.

**Formula** (Crespo-Otero and Barbatti, *Theor. Chem. Acc.* **131**, 1237 (2012); refractive index 1). The absorption cross section per molecule is

$$
\sigma(E)=\sum_n\sigma_n(E),
\qquad
\sigma_n(E)=\frac{\pi e^2\hbar}{2\,m_e\,c\,\varepsilon_0}\;\frac1E\;\frac1{N_{\text{ok}}}\sum_{i=1}^{N_{\text{ok}}}\Delta E_{in}\,f_{in}\;g\bigl(E-\Delta E_{in}\bigr).
$$

**The prefactor.** With $E$ and $\Delta E$ in eV the units of $\sigma$ are Å². In atomic units ($e=\hbar=m_e=1$, $4\pi\varepsilon_0=1$) the constant is $2\pi^2/c$, so

$$
\frac{\pi e^2\hbar}{2\,m_e\,c\,\varepsilon_0}=\frac{2\pi^2}{c_{\text{au}}}\;E_h[\mathrm{eV}]\;a_0[\text{Å}]^2 = 1.09761\ \mathrm{eV\,\text{Å}^2},
$$

with $c_{\text{au}}=137.036$, $E_h = 27.2114$ eV and $a_0 = 0.529177$ Å (§8.1). This is the same number as the SI expression. The molar absorption coefficient follows from the unit change Å² $\to$ cm² ($10^{-16}$), the Avogadro constant $N_A$, litres ($1/1000$) and the decadic logarithm:

$$
\varepsilon(E)=\sigma(E)\,\frac{10^{-16}\,N_A}{1000\ln10}=26153.8\ \mathrm{M^{-1}cm^{-1}\,\text{Å}^{-2}}\;\sigma(E).
$$

**The grid.** $E_j=e_0+j\,\Delta$, with $\Delta$ = `spectrum_step_eV`, where
$e_0=\max\bigl(\Delta,\ \Delta\lfloor(\min\Delta E - 4s)/\Delta\rfloor\bigr)$, and $j$ runs up to the first grid point at or above $\max\Delta E+4s$ (minimum and maximum over all successful samples and all states). The Gaussians are evaluated at every grid point, so a single line is integrated to within $2\times10^{-4}$ (relative) of the exact value (the grid ends at $\pm4s$ from the lowest and highest line). The columns of `spectrum.csv` are $\sigma(E_j)$, $\varepsilon(E_j)$ and the single terms $\sigma_n(E_j)$ (§6).

**Remarks.** The spectrum has no vibronic structure, only the width that the Wigner distribution of the ground-state nuclei gives plus the broadening $\delta$ (which smooths the finite number of samples; with few samples, a smaller $\delta$ shows the individual samples). The result depends on $\delta$ and on $N_{\text{ok}}$; check convergence with more samples.
#### 8.9.3 Selection of the initial states (`selection`)
**Candidates.** A *candidate* is a pair (sample $i$, state $n$) such that sample $i$ succeeded, state $n$ is in `selection.states` (default: all excited singlets), and, if `energy_window_eV` $=[lo,hi]$ is given, $lo \le \Delta E_{in} \le hi$ (both limits included, $\Delta E$ in eV). Candidates are ordered **sample by sample, and within a sample S1, S2, ...** (sample-major). A sample can give several candidates, and so it can start more than one trajectory; a given pair is used at most once.

**Bright mode** (`mode = "bright"`, open item O3, decided 2026-11-08). Every candidate has a weight $w_{in}$, and $w_{\max}$ is the largest weight **over the candidates** (so over the pairs inside the window; a narrower window therefore raises the acceptance of the remaining pairs). A candidate is accepted if

$$
r_{in} < \frac{w_{in}}{w_{\max}},
\qquad r_{in}\sim U[0,1) \text{ independent, one per candidate}.
$$

The weighting is chosen by `selection.weighting`:

| `weighting` | Weight $w_{in}$ | Notes |
|---|---|---|
| `"f_over_dE2"` (default) | $f_{in}/\Delta E_{in}^2$ | **SHARC 4.1 `excite.py`-compatible.** The rule was determined from SHARC's `excite.py` run as a black box: maximum-likelihood fits of the exponents of $w=f^a\,\Delta E^k$ to its per-pair selections give $k=-1.99\pm0.05$ and $a=0.99\pm0.03$ (§8.9.4); it is *not* taken from a publication. Since $f=\tfrac23\Delta E\lvert\mu\rvert^2$, $w\propto\lvert\mu\rvert^2/\Delta E$: it **favours low-energy pairs** (see the limits below). The ratio $w/w_{\max}$ does not depend on the energy unit |
| `"f"` | $f_{in}$ | The oscillator-strength convention: the acceptance probability is $f/f_{\max}$ (cf. Barbatti et al., *J. Photochem. Photobiol. A* **190**, 228 (2007)) |

If $w_{\max}\le0$ (no candidate, or all $f=0$), nothing is accepted. The accepted pairs are taken **in candidate order, up to `n_trajectories`**. All accepted pairs are counted first (the per-state numbers in the report are exact), then the list is cut.

**Fixed mode** (`mode = "fixed"`). `selection.states` holds exactly one state. The trajectories are the **first** `n_trajectories` eligible samples in ascending order (successful, and inside the window if one is given) in that state; no random numbers and no weights are used. `weighting` is ignored (and not part of the fingerprint, §8.3), but it is still written to the reports.

**Too few trajectories (O20, decided 2026-11-08).** If fewer than `n_trajectories` pairs are selected, the stage stops with exit code 4 and a message with the numbers (candidates, accepted, $f_{\max}$, the weighting, the window, the states) and the advice, each with the rerun it needs:

- widen `energy_window_eV`, add states or lower `n_trajectories`, **then run with `--rerun-from selection`** (these keys change the selection);
- increase `[sampling] n_samples`, **then run with `--rerun-from sampling`**, which recomputes the samples and the excitations of every sample;
- or set `allow_fewer = true` and run again: **no `--rerun-from` is needed**, because `allow_fewer` acts only in the finalization and is not part of the fingerprint (§8.3); the stored task result is reused.

With `allow_fewer = true` the stage accepts the smaller number, logs a warning, and writes `trajectories.json` with the number actually found. **Zero selected pairs is always an error**, also with `allow_fewer = true`. `selection_report.json` and `selection_report.md` are written **before** the stop (with `stage_failed = true`; `trajectories.json` is not written), so the numbers behind the stop can be read there. Because `allow_fewer` is not part of the fingerprint, setting it back to `false` after a finished run does **not** make `selection` stale: the smaller `trajectories.json` is kept. To apply the stricter setting, run with `--rerun-from selection`.

**The $w_{\max}$ pair and outliers.** In bright mode the reports name the pair that sets $w_{\max}$ (sample, state, $\Delta E$, $f$, $w$) and give the 99th percentile $w_{99}$ of the candidate weights. If $w_{\max} > 10\,w_{99}$, one pair dominates the normalization and suppresses every other acceptance; the stage then logs a **warning** (*the w_max pair (sample k, S1, dE = 0.05 eV, …) is … times the 99th percentile of the candidate weights …; set [selection] energy_window_eV to exclude it (then run with --rerun-from selection)*), marks `weight_outlier = true` in `selection_report.json` and repeats the warning in `selection_report.md` and, if the stage stops on O20, in its message. Example (measured by a unit test): 200 samples with S1 at 5 eV ($f=0.05$) and S2 at 7 eV ($f=0.3$), plus **one** sample with S1 at 0.05 eV and $\lvert\mu\rvert=1$ a.u.: with `f_over_dE2` and no window, fewer than 10 of the 402 candidates are accepted; with `energy_window_eV = [1.0, 10.0]` more than 200 are, and there is no warning.

**Random streams (seeds, D12).** The random numbers $r_{in}$ come from the generator of the stage `selection`, task 0: a NumPy `SeedSequence` built from `project.seed` and the CRC-32 checksum of the stage name, as in §8.8. Exactly **one number is drawn per candidate**, in candidate order, so the stream is a fixed function of the candidate list. Trajectory $k$ (0-based) will use the stream of the stage `dynamics`, task $k$, derived the same way; `trajectories.json` records it as `rng = {"stage": "dynamics", "task_index": k}` instead of an integer seed, so the random numbers of trajectory $k$ do not depend on how many other trajectories exist.

**Assumptions and limits.**
- The *vertical* excitation energy at a sample is used as the energy of the initial state; the nuclei start on the excited-state surface with the sampled geometry and velocities, with no further adjustment of the energy.
- The rule treats every pair as independent: the same sample may supply S1 and S2 trajectories.
- The adiabatic ordering is used at each geometry; if two states are close and cross between samples, the labels S1 and S2 of different samples may not describe the same electronic character.
- With the TDA/TDDFT description of a very bright state the oscillator strength can be unreliable (§8.6.4); in that case the selection inherits the error.
- **Low-energy bias of `f_over_dE2`.** $f/\Delta E^2=\tfrac23\lvert\mu\rvert^2/\Delta E$ (atomic units) grows without bound as $\Delta E\to0$, while $f$ itself goes to 0. A single sample near an S1/S0 seam (SA-CASSCF) or with a collapsed TDDFT root can therefore take over $w_{\max}$ and nearly empty the selection (see "The $w_{\max}$ pair and outliers" above). **Set `energy_window_eV` around the absorption band you want to excite** (read it from `spectrum.csv`): the window is applied before $w_{\max}$, so it removes such pairs. The `f` weighting does not have this bias.
- Non-finite pairs and pairs with $\Delta E\le0$ cannot reach the selection (the `excitations` stage records such samples as failed, §8.9.1); the selection also skips them defensively.

#### 8.9.4 Validation (measured)
Both stages were compared with SHARC 4.1 as a black box (SHARC is only executed; its source is not read). The tools are in `tools/sharc_compare/` (`sharc_formats.py` writes and reads the SHARC text formats; `run_excite.py` drives `excite.py`; `run_sharc_pyscf.py` runs `SHARC_PYSCF.py`), the tests in `tests/validation/test_sharc_excitation_energies.py` and `tests/validation/test_sharc_excite.py` (marker `sharc`, run with `SHARC_HOME` and `SHARC_PYTHON`, never in CI).

**Energies, dipoles and oscillator strengths against `SHARC_PYSCF.py`.** Water, SA3-CASSCF(4,4)/STO-3G (a state-averaged singlet CAS with three roots), the equilibrium geometry and two Wigner samples (RHF/STO-3G Hessian, $T=0$, seed 5), tight settings on both sides, a fresh backend per geometry. The same `QM.in` was given to SHARC's PySCF interface:

| geometry | state | $\Delta E$ SHARC ($E_h$) | $\Delta E$ ours ($E_h$) | difference ($E_h$) | difference of $\lvert\boldsymbol\mu\rvert$ ($e\,a_0$) | difference of $f$ |
|---|---|---|---|---|---|---|
| 0 (reference) | S1 | 0.42680104 | 0.42680105 | $1.1\times10^{-8}$ | $3.2\times10^{-9}$ | $2.2\times10^{-10}$ |
| 0 | S2 | 0.49404230 | 0.49404228 | $2.0\times10^{-8}$ | $2.2\times10^{-9}$ | $3.5\times10^{-12}$ |
| 1 | S1 | 0.36747180 | 0.36747175 | $5.4\times10^{-8}$ | $8.2\times10^{-8}$ | $3.5\times10^{-9}$ |
| 1 | S2 | 0.42178738 | 0.42178741 | $2.7\times10^{-8}$ | $3.8\times10^{-9}$ | $3.0\times10^{-11}$ |
| 2 | S1 | 0.47953815 | 0.47953824 | $8.7\times10^{-8}$ | $1.2\times10^{-7}$ | $7.4\times10^{-9}$ |
| 2 | S2 | 0.56282512 | 0.56282507 | $5.1\times10^{-8}$ | $2.8\times10^{-8}$ | $3.6\times10^{-11}$ |

The worst differences are $8.7\times10^{-8}\ E_h$ in $\Delta E$ (limit $10^{-6}$), $1.2\times10^{-7}\ e\,a_0$ in $\lvert\boldsymbol\mu\rvert$ (the S1 value is about 0.09, so $1.3\times10^{-6}$ relative) and $7.4\times10^{-9}$ in $f$. The dipole vectors agree up to an overall sign. The SHARC-format files that the tools write (`QM.in`, `QM.out`, the `initconds` file) are read back by SHARC's own `excite.py`, which echoes our energies to $1.4\times10^{-7}$ eV and our oscillator strengths to $1.4\times10^{-8}$ (SHARC prints $f$ with 8 decimals).

**The weighting exponents, measured.** Under a weight $w=f^a\,\Delta E^k$ normalized by the in-window maximum, a pair is selected with probability $p_i=w_i/\max_j w_j$. Two synthetic data sets of $N=3000$ samples isolate the exponents. In both, only S1 matters (S2 is dark, $f=0$, so it is never selected) and every pair lies inside the window $[0,20]$ eV; SHARC `excite.py` runs with seed 7.

- **$\Delta E$ exponent:** constant $f_1=0.1$, $\Delta E_1\sim U(2,10)$ eV (a factor 5). SHARC selected 598 of the 3000 pairs.
- **$f$ exponent:** constant $\Delta E_1=5$ eV, $f_1\sim U(0.01,0.4)$. SHARC selected 1527 pairs.

The exponent is fitted by **maximum likelihood** from SHARC's per-pair `selected` flags $y_i$: the binomial log-likelihood $\ln L=\sum_i\bigl[y_i\ln p_i+(1-y_i)\ln(1-p_i)\bigr]$ is maximized (a grid scan, then a bounded one-dimensional optimization), and the standard error is $1/\sqrt{-\partial^2\ln L/\partial k^2}$ (the observed information, by a central finite difference). Results:

| Exponent | Fitted value ± standard error | Expected for $f/\Delta E^2$ | $z$ |
|---|---|---|---|
| $k$ ($\Delta E$) | $-1.989\pm0.050$ | $-2$ | +0.21 |
| $a$ ($f$) | $0.993\pm0.028$ | $1$ | −0.25 |

The tests assert $\lvert\hat k+2\rvert<4\,\mathrm{SE}$ and $\lvert\hat a-1\rvert<4\,\mathrm{SE}$. So SHARC 4.1 uses $(f/\Delta E^2)/\max$; the alternatives $k=-1$ ($w\propto f/\Delta E\propto\lvert\mu\rvert^2$, about 20 standard errors from $\hat k$), $k=-1.5$ (about 10), $k=0$ ($w=f$) and $k=+2$ are excluded. For comparison, the expected counts of the $\Delta E$ data set are 357.9 ($k=-3$), 596.3 ($k=-2$), 825.0 ($k=-1.5$) and 1203.1 ($k=-1$), against the 598 observed.

**Window semantics against `excite.py`.** 3000 synthetic samples with two excited states, an energy window $[4,8]$ eV, seed 7. The data include pairs *outside* the window with a larger $f/\Delta E^2$ than any pair inside it, so that the window maximum and the all-pairs maximum can be told apart. SHARC selected **370 (S1) and 699 (S2)**. The numbers each rule predicts, and the difference in standard deviations of the binomial distribution ($z$):

| Rule | Expected S1 | Expected S2 | $z$ (S1) | $z$ (S2) |
|---|---|---|---|---|
| $f/\Delta E^2$, maximum over the **window** (our default) | 349.8 | 654.6 | +1.19 | +2.36 |
| $f/\Delta E^2$, maximum over all pairs | 126.7 | 237.1 | +22.4 | +32.8 |
| $f/\Delta E$ ($\propto\lvert\mu\rvert^2$), maximum over the window | 338.9 | 812.6 | +1.85 | −5.86 |
| $f$, maximum over the window | 332.5 | 1026.7 | +2.25 | −17.8 |

The test asserts only what these counts can decide: the default rule within 4 standard deviations, and the all-pairs maximum rejected (at least one state with $\lvert z\rvert>6$), so **the maximum is taken over the in-window pairs**. The other rows are reported only, because the counts of two states cannot pin an exponent down (the exponent is settled by the fits above). Our `select_bright` with the default rule on the same data gives 353 (S1) and 666 (S2), $z=+0.19$ and $+0.61$ against its own expectation. A two-sample Kolmogorov–Smirnov test of the $\Delta E$ of the selected S1 pairs, ours against SHARC's, gives $p=0.96$ (statistic 0.036). The comparison cannot say *why* SHARC uses $f/\Delta E^2$; hence the wording "determined by black-box comparison".

### 8.10 Nuclear motion (velocity Verlet)
**[IMPLEMENTED]** (P7a-1, `gqteashmd.dynamics`, analytic models; **[PLANNED, P7b]** for molecules with PySCF). All of §8.10–§8.12 and §8.14 use **Hartree atomic units** ($\hbar = 1$): positions in bohr, masses in electron masses $m_e$, time in $\hbar/E_h$ (1 a.u. = 0.024188843265857 fs), velocities in bohr per atomic unit of time, energies in $E_h$, nonadiabatic couplings in 1/bohr. The core works on a flat coordinate vector $\mathbf q$ of length $f$ (the number of degrees of freedom) with masses $\mathbf m$ of length $f$; a molecule of $N_{\text{at}}$ atoms has $f = 3N_{\text{at}}$ (P7b).

**Symbols.** $n$ is the number of electronic states, $\lambda$ the active state, $E_k(\mathbf q)$ the energies, $\mathbf g_k = \partial E_k/\partial\mathbf q$ the gradients (length $f$), $\mathbf d_{jk} = \langle j|\partial k/\partial\mathbf q\rangle$ the nonadiabatic coupling vector (NAC, length $f$; real and antisymmetric for real states), $\mathbf v$ the velocity, $\mathbf M = \mathrm{diag}(\mathbf m)$, $K = \tfrac12\mathbf v^{\mathsf T}\mathbf M\mathbf v$ the kinetic energy and $\Delta t$ = `dt_au`.

The nuclei move classically on the active surface:

$$
\mathbf a = -\mathbf M^{-1}\mathbf g_\lambda(\mathbf q),\qquad
\mathbf q' = \mathbf q + \mathbf v\,\Delta t + \tfrac12\mathbf a\,\Delta t^2,\qquad
\mathbf v' = \mathbf v + \tfrac12\big[\mathbf a + \mathbf a'\big]\Delta t ,\quad \mathbf a' = -\mathbf M^{-1}\mathbf g'_\lambda(\mathbf q') .
$$

The time step is fixed (velocity Verlet; no adaptive step). The order of all operations of one step, including where the hop happens, is in §8.12.

**Total energy.** $E_{\text{tot}} = E_\lambda(\mathbf q) + K$. The trajectory ends with the outcome `energy_drift` when $|E_{\text{tot}}(t) - E_{\text{tot}}(0)|$ exceeds `energy_drift_limit_au` (default 0.5 eV = 0.0183746 $E_h$, in the library and in `model-dynamics`; a NaN also counts as drift). Because the velocity is completed before a hop and rescaled exactly, $E_{\text{tot}}$ is continuous across hops.

**Measured:** Tully SAC with hopping off, $\Delta t = 1$ a.u.: $\max|E_{\text{tot}}-E_{\text{tot}}(0)| = 3.2\times10^{-7}\,E_h$ (and $8.1\times10^{-8}$ at $\Delta t = 0.5$, the expected $\Delta t^2$ scaling); at a hop step the jump is at most $7.9\times10^{-9}\,E_h$ ($\text{sharc}$, both velocity adjustments).

### 8.11 Electronic motion
**[IMPLEMENTED]** (P7a-1; molecules **[PLANNED, P7b]**). The electronic wavefunction $\Psi(t) = \sum_j c_j(t)\,\Psi_j$ is expanded in the **propagation basis**, the diagonal basis, which in P7a-1 equals the adiabatic (MCH) basis. It obeys

$$
\dot c_j = -i E_j\, c_j - \sum_k T_{jk}\, c_k,\qquad T_{jk} = \mathbf v\cdot\mathbf d_{jk}
\qquad\Longleftrightarrow\qquad
i\,\dot{\mathbf c} = \mathbf H_{\text{eff}}\,\mathbf c,\quad \mathbf H_{\text{eff}} = \mathrm{diag}(\mathbf E) - i\,\mathbf T .
$$

| Symbol | Meaning |
|---|---|
| $c_j$ | Complex amplitude of state $j$ (dimensionless); $\lvert c_j\rvert^2$ is the quantum population |
| $E_j$ | Energy of state $j$, in $E_h$ |
| $T_{jk}$ | Time-derivative coupling $\mathbf v\cdot\mathbf d_{jk}$ (real, antisymmetric), in 1/(atomic unit of time) |
| $\mathbf H_{\text{eff}}$ | Effective Hamiltonian; Hermitian, so the norm is conserved |

**Propagator over one nuclear step.** With $N_s$ = `nsubsteps` (default 25) and $\delta = \Delta t/N_s$, the step propagator is $\mathbf P = \mathbf P_{N_s}\cdots\mathbf P_1$ with $\mathbf P_m = \exp[-i\,\mathbf H_{\text{eff}}(s_m)\,\delta]$, computed from the eigendecomposition of the (explicitly Hermitized) $\mathbf H_{\text{eff}}$. $\mathbf P$ is unitary, so $\sum_j|c_j|^2 = 1$ is conserved to rounding. The option `substep_interpolation` sets how $\mathbf E$, $\mathbf d$ and $\mathbf v$ are interpolated between their values at $t$ and $t+\Delta t$ ($s\in[0,1]$ is the fraction of the step):

| `substep_interpolation` | Points $s_m$ | Interpolation | Accuracy |
|---|---|---|---|
| `"midpoint"` (default) | $s_m = (m-\tfrac12)/N_s$ | $\mathbf E$, $\mathbf d$, $\mathbf v$ linear, each separately; $\mathbf T(s) = \mathbf v(s)\cdot\mathbf d(s)$ | second order in $\delta$ |
| `"endpoint"` | $s_m = m/N_s$ | $\mathbf E$ and $\mathbf T = \mathbf v\cdot\mathbf d$ linear (between $\mathbf v(t)\cdot\mathbf d(t)$ and $\mathbf v(t+\Delta t)\cdot\mathbf d(t+\Delta t)$) | first order in $\delta$ |

`"endpoint"` is the scheme of the public SHARC 4.1 manual (§8.33.1, eqs. 8.191–8.194), offered for comparison with SHARC. The two schemes have slightly different continuous limits when $\mathbf v$ varies during the step (one is linear and the other quadratic in $s$ for $\mathbf T$). The check against SHARC is in §8.14.

**Not yet available:** local diabatization from wavefunction overlaps, the `overlap` coupling type, and spin–orbit coupling. **[PLANNED, P7a-2 and P7b]** What P7a-1 prepares: every electronic snapshot carries the complex MCH Hamiltonian, and every step calls `diagonal_basis` on it as a guard. That call computes the MCH→diagonal matrix $\mathbf U$; $\mathbf U$ must be diagonal up to phases (no SOC), and a non-diagonal one (largest off-diagonal element above $10^{-10}$) raises `NotImplementedError` pointing to P7a-2. **$\mathbf U$ itself is not stored:** no snapshot, trajectory state or restart file holds it, and each step uses the identity as the reference. Carrying $\mathbf U$ from step to step with phase continuity (a new field in the snapshot or state, and restart format version 2) is P7a-2 work.

### 8.12 Surface hopping, energy adjustment, decoherence
**[IMPLEMENTED]** (P7a-1, analytic models; **[PLANNED, P7b]** for molecules). Options and their defaults:

| Option (`[dynamics]` key) | Values (default first) | Meaning |
|---|---|---|
| `hopping` | `"sharc"`, `"fssh"`, `"none"` | Hopping probability: propagator-based (SHARC type), Tully fewest switches, or off (for tests and "hopping off" comparisons) |
| `ekin_correction` | `"parallel_vel"`, `"parallel_nac"` | Direction of the velocity adjustment at a hop |
| `frustrated` | `"keep"`, `"reflect"` | What happens to the velocity at a frustrated hop |
| `decoherence` | `"edc"`, `"none"` | Energy-based decoherence correction on or off |
| `decoherence_param_Eh` | 0.1 | $C$ of the EDC, in $E_h$ ($\ge 0$) |
| `edc_convention` | `"sharc"`, `"granucci_persico"` | Which damping rate the EDC uses (below) |
| `substep_interpolation` | `"midpoint"`, `"endpoint"` | §8.11 |
| `dt_au`, `nsubsteps`, `energy_drift_limit_au` | 1.0 (`model-dynamics`; required in the library), 25, 0.5 eV = 0.0183746 $E_h$ | §8.10, §8.11 |

Only **`nacdr`** couplings (NAC vectors) exist. Deferred (not in the first release): A-FSSH, NPI, GFSH, `parallel_diff`, thermostats, adaptive time steps.

**Hopping probability, Tully FSSH** (`hopping = "fssh"`; Tully 1990, adiabatic basis). The flux out of the active state $\lambda$ into $j$ follows from the electronic equation, $\mathrm d|c_\lambda|^2/\mathrm dt = -\sum_j 2\,\mathrm{Re}(c_\lambda^{*}\,T_{\lambda j}\,c_j)$. It is accumulated over the substeps, with the amplitudes at the start of each substep $m$ and $\mathbf T$ at the same points $s_m$ as the propagator:

$$
g_{\lambda\to j} = \max\!\left(0,\;\sum_{m=1}^{N_s}\delta\,\frac{2\,\mathrm{Re}\big(c_\lambda^{*}\,T_{\lambda j}\,c_j\big)_m}{|c_\lambda|^2_m}\right),\qquad j\neq\lambda .
$$

(A tiny floor on $|c_\lambda|^2$ avoids a division by zero.) If $\sum_j g_{\lambda\to j} > 1$ the probabilities are scaled down proportionally so that the sum is 1.

**Hopping probability, SHARC type** (`hopping = "sharc"`, the default; Mai, Marquetand & González, *Int. J. Quantum Chem.* **115**, 1215 (2015)). With $\mathbf c = \mathbf c(t)$, $\mathbf c' = \mathbf c(t+\Delta t)$ and $\mathbf P$ the step propagator of §8.11:

$$
h_{\lambda\to j} = \left(1 - \frac{|c'_\lambda|^2}{|c_\lambda|^2}\right)
\frac{\mathrm{Re}\big(c'_j\,P_{j\lambda}^{*}\,c_\lambda^{*}\big)}{|c_\lambda|^2 - \mathrm{Re}\big(c'_\lambda\,P_{\lambda\lambda}^{*}\,c_\lambda^{*}\big)},\qquad j\neq\lambda .
$$

Negative values are set to 0, and if $|c'_\lambda|^2 \ge |c_\lambda|^2$ all are 0 (no population is leaving). The same proportional cap as above applies. If the denominator is $\le 10^{-14}$, all are 0. Both probabilities are invariant under a global phase of $\mathbf c$ (tested).

**Hop decision.** One uniform random number $\xi\in[0,1)$ is drawn in **every** step, used or not (§8.14). With the states $j\ne\lambda$ in index order and the cumulative sums $S_j$ of the probabilities, the trajectory hops to the first $j$ with $\xi < S_j$; if there is none, it does not hop.

**Velocity adjustment** for a hop $\lambda\to j$ at fixed $\mathbf q'$, with $\Delta E = E_j - E_\lambda$ (at $\mathbf q'$) and kinetic energy $K$ (after the velocity update of the step). Energy conservation requires $K_{\text{new}} = K - \Delta E$.

- **`parallel_vel`** (adjust along the velocity): if $\Delta E \le K$, $\mathbf v' = \alpha\,\mathbf v$ with $\alpha = \sqrt{1 - \Delta E/K}$; otherwise the hop is **frustrated**. $K = 0$ and $\Delta E > 0$ is frustrated. $K = 0$ and $\Delta E < 0$ has no direction and raises a `ValueError` (unreachable from a moving trajectory; use `parallel_nac`).
- **`parallel_nac`** (adjust along the NAC): with $\mathbf u = \mathbf d_{\lambda j}$ (taken at $\mathbf q'$), $\mathbf v' = \mathbf v + \gamma\,\mathbf M^{-1}\mathbf u$, where energy conservation gives $a\gamma^2 + b\gamma + \Delta E = 0$ with $a = \tfrac12\mathbf u^{\mathsf T}\mathbf M^{-1}\mathbf u$ and $b = \mathbf v\cdot\mathbf u$. If the discriminant $D = b^2 - 4a\Delta E < 0$, the hop is frustrated. Otherwise $\gamma = (-b\pm\sqrt D)/(2a)$ with the root of **smaller $|\gamma|$** (computed as $\gamma = \Delta E/q$, $q = -(b + \mathrm{sign}(b)\sqrt D)/2$, which has no cancellation). If $a \le 10^{-30}$ the `parallel_vel` rule is used. (At rest, with a vanishing NAC and $\Delta E < 0$, there is no adjustment direction at all: a `ValueError` naming the zero NAC.) The test `test_hop_uses_the_nac_at_the_new_geometry` checks on the 2D model near its conical intersection that the NAC at $\mathbf q'$, not at $\mathbf q$, is used.
- **Frustrated hop**, `frustrated = "keep"`: the velocity is unchanged and there is no hop. `"reflect"`: no hop, and the momentum component along $\mathbf u$ is reversed in the mass-weighted metric, $\mathbf v' = \mathbf v - 2\,\dfrac{\mathbf v\cdot\mathbf u}{\mathbf u^{\mathsf T}\mathbf M^{-1}\mathbf u}\,\mathbf M^{-1}\mathbf u$ (for `parallel_vel`, $\mathbf u = \mathbf M\mathbf v$, which gives $\mathbf v' = -\mathbf v$; for `parallel_nac`, $\mathbf u = \mathbf d_{\lambda j}$). Every frustrated event is recorded in the step record.

**Energy-based decoherence correction (EDC)** (`decoherence = "edc"`), applied after the hop step to $\mathbf c'$, with $K$ the kinetic energy after the hop, $C$ = `decoherence_param_Eh` and, for every inactive state $j\neq\lambda$, the decoherence time $\tau_j = \dfrac{1}{|E_j - E_\lambda|}\Big(1 + \dfrac{C}{K}\Big)$:

| `edc_convention` | Update of the inactive amplitudes | Source |
|---|---|---|
| `"sharc"` (default) | $c_j \leftarrow c_j\,\exp\!\big[-\tfrac12\,\Delta t\,|E_j-E_\lambda|\,(1 + C/K)^{-1}\big] = c_j\,e^{-\Delta t/(2\tau_j)}$ | SHARC 4.1 manual §8.7.1, eq. (8.25) |
| `"granucci_persico"` | $c_j \leftarrow c_j\,e^{-\Delta t/\tau_j}$ | Granucci & Persico, *J. Chem. Phys.* **126**, 134114 (2007), as reproduced in Runeson, arXiv:2507.18381 (2025), Sec. III.3, eqs. (44a), (44b), (45) |

In both, the active amplitude is then rescaled keeping its phase, $c_\lambda \leftarrow c_\lambda\,\sqrt{\big(1 - \sum_{j\neq\lambda}|c_j|^2\big)/|c_\lambda|^2}$ (manual eq. 8.26; Runeson eq. 45), so the norm is exactly 1. If $|E_j-E_\lambda| = 0$ or $K = 0$, $c_j$ is unchanged. **The `"sharc"` convention damps the amplitudes at half the rate of `"granucci_persico"`** (measured against SHARC 4.1: damping exponent $\kappa$ = 0.500000 for SHARC, 0.500000 for `"sharc"`, 1.000000 for `"granucci_persico"`). The default follows SHARC (owner decision 2026-11-08).

> **Caveat on EDC.** EDC with $C = 0.1\,E_h$ **biases** simple one-dimensional benchmarks: with the Granucci–Persico rate it raised the upper-state yield on SAC at low momentum and destroyed the Stückelberg interference of DAC at high energy (numbers in §8.14). The time $\tau_j \propto 1/|\Delta E|$ becomes shorter than the time between two crossings when the gap is large. Runeson (arXiv:2507.18381, 2025) criticises the EDC for depending on the *total* kinetic energy of the system. Use `decoherence = "none"` (plain Tully FSSH) for such benchmarks, and validate the choice for your molecule.

**Step order** (ours, documented; one nuclear step from $(\mathbf q,\mathbf v,\mathbf a = -\mathbf M^{-1}\mathbf g_\lambda(\mathbf q),\mathbf c,\lambda)$, with the snapshot $\text{snap}$ = energies, gradients, NACs at $\mathbf q$):

1. $\mathbf q' = \mathbf q + \mathbf v\Delta t + \tfrac12\mathbf a\Delta t^2$.
2. $\text{snap}' = $ model evaluated at $\mathbf q'$ (with the previous snapshot for sign continuity).
3. $\mathbf v' = \mathbf v + \tfrac12(\mathbf a - \mathbf M^{-1}\mathbf g'_\lambda)\Delta t$ (with the old active state).
4. Propagate $\mathbf c$ over $[t, t+\Delta t]$, with $\mathbf E$, $\mathbf d$ interpolated between $\text{snap}$ and $\text{snap}'$ and $\mathbf v$ between $\mathbf v$ and $\mathbf v'$; this gives $\mathbf c'$ and $\mathbf P$.
5. Hopping probabilities of the chosen scheme; draw $\xi$.
6. If there is a hop: velocity adjustment of $\mathbf v'$ (frustrated: keep or reflect, $\lambda$ unchanged); if it succeeds, $\lambda\leftarrow j$. The NAC used by `parallel_nac` is $\mathbf d_{\lambda j}$ of $\text{snap}'$.
7. Decoherence on $\mathbf c'$, with $K$ from the final $\mathbf v'$ and $\mathbf E$ from $\text{snap}'$.
8. $\mathbf a'$ is the gradient of the (new) active state at $\mathbf q'$, divided by $-\mathbf m$ (recomputed from the stored snapshot, not stored).
9. Record the step; check the energy drift.

The hop thus happens **after** the velocity update of the step and the rescaling is done at fixed $\mathbf q'$. Where SHARC's order differs, the difference is measured by black-box comparison (§8.14), not read from its source.

**Outcomes of a trajectory:** `stopped` (a stop rule you gave was met), `finished` (`max_steps` reached without stopping) and `energy_drift`. Physical endings are outcomes, not errors; invalid options are a `ValueError` naming the option.

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

### 8.14 Dynamics core: models, determinism, restart, performance and validation
**[IMPLEMENTED]** (P7a-1, `gqteashmd.dynamics`, command `gqteashmd model-dynamics`, §5.7). The package imports nothing from `electronic/`, `stages/`, the orchestrator or `config/`; it takes its electronic data through a small protocol (`evaluate(q, previous) -> snapshot`) that the analytic models implement now and a PySCF adapter will implement in P7b. It uses NumPy only (SciPy only in tests and tools).

**Models.** Tully's three one-dimensional two-state models (Tully 1990), mass 2000 $m_e$ by default, atomic units. $V_{ij}$ is the diabatic matrix; the adiabatic energies, gradients and NACs $d_{jk} = G_{jk}/(E_k - E_j)$ with $\mathbf G = \mathbf W^{\mathsf T}(\partial\mathbf V/\partial x)\mathbf W$ come from its eigendecomposition (checked against finite differences). A two-dimensional two-state test model also exists in the library.

| Model | Parameters (default) | Diabatic matrix |
|---|---|---|
| `tully_sac` (simple avoided crossing) | $A$ 0.01, $B$ 1.6, $C$ 0.005, $D$ 1.0 | $V_{11} = \mathrm{sign}(x)\,A(1 - e^{-B|x|})$, $V_{22} = -V_{11}$, $V_{12} = C e^{-Dx^2}$ |
| `tully_dac` (dual avoided crossing) | $A$ 0.1, $B$ 0.28, $C$ 0.015, $D$ 0.06, $E_0$ 0.05 | $V_{11} = 0$, $V_{22} = -A e^{-Bx^2} + E_0$, $V_{12} = C e^{-Dx^2}$ |
| `tully_ecr` (extended coupling with reflection) | $A$ 6e-4, $B$ 0.1, $C$ 0.9 | $V_{11} = A$, $V_{22} = -A$, $V_{12} = B e^{Cx}$ for $x<0$ and $B(2 - e^{-Cx})$ for $x>0$ |

**Initial conditions of an ensemble.** The packet $\psi(x)\propto\exp[-(x-x_0)^2/\sigma^2 + ikx]$ on the lower adiabatic state. Its density has standard deviation $\sigma/2$ and its Wigner function $s_p = 1/\sigma$ ($\hbar = 1$), so `sampling = "wigner"` draws $x\sim N(x_0,\sigma/2)$ and $p\sim N(k,1/\sigma)$; `"fixed"` gives every trajectory $x_0$ and $k/m$. The default width is $\sigma = 20/|k|$. A trajectory runs until $|x| > $ `x_exit` while moving outward (the stop rule); the branch is **transmitted** ($x>0$) or **reflected** ($x<0$) and **lower** or **upper** (the active state). Probabilities are fractions over the stopped trajectories with 95 % Wilson score intervals.

**Determinism.** Exactly **one** uniform random number is drawn per nuclear step, whether it is used or not (also with `hopping = "none"`), and it is logged in the step record. Ensemble trajectory $i$ uses `default_rng(SeedSequence([seed, crc32(b"model-dynamics")]).spawn(i + 1)[i])`; the initial conditions use their own stream, `SeedSequence([seed, crc32(b"model-initial")])`. A `ReplaySource` fed with the logged numbers reproduces a trajectory exactly. The counts of an ensemble do **not** depend on `workers` (tested for 1, 2 and 4 workers; a 200-trajectory benchmark gave identical counts for 1, 2, 4, 8 and 16).

**Restart (library).** `dynamics.restart.save_state(path, state, rng)` writes one atomic `.npz` and `load_state(path, model)` reads it; continuing gives a result **bit-identical** to an uninterrupted run (tested: 50 steps + save + load + 50 steps equals 100 steps, for both hopping schemes with EDC and with hops in the window). It stores $\mathbf q$, $\mathbf v$, $\mathbf c$, the electronic snapshot (energies, gradients, NACs, Hamiltonian, model eigenvectors), time, step, active state, $E_{\text{tot}}(0)$ and the random-source state (a JSON `meta` entry; floats round-trip exactly). It does **not** store the masses, the model parameters, the options or $\mathbf U$ (with no SOC $\mathbf U$ is trivial; P7a-2 must store it); you must supply the same model and options. `load_state` never evaluates the model. A missing, foreign or corrupt file, or a shape mismatch with the model, is a `ValueError`. The random-source state is treated as untrusted input: only the NumPy bit generators PCG64, PCG64DXSM, MT19937, Philox and SFC64 are accepted (checked before anything is called), and a malformed state or a replay position outside the stored sequence is also a `ValueError`. The `model-dynamics` command does not write restart files (its trajectories are short); the `dynamics` stage of P7b will.

**Performance** (this workstation, 16 logical CPUs, code on `/mnt/c`; Tully SAC, $k = 20$, $\Delta t = 1$ a.u., $N_s = 25$, about 2040 steps per trajectory): about 0.30 s per trajectory with `hopping = "sharc"` and 0.35 s with `"fssh"` (about 150 µs per step; vectorised over the substeps, 4–5 times faster than the first version). 200 trajectories: 65 s with `workers = 1`, 18 s with `workers = 8` (measured again in §5.7: 17.98 s). Eight workers give 3.6 times the speed of one; more than 8 does not help on this machine (4 performance cores). Workers are processes started with `spawn`; a script that calls `run_model_ensemble(..., workers > 1)` must guard its entry point with `if __name__ == "__main__":`.

#### 8.14.1 Validation (measured)
**Three independent checks, all measured.** Where a tolerance is given, it is the rule "$|\text{FSSH} - \text{exact}| \le 4\,\mathrm{SE} + 0.02$" with $\mathrm{SE} = \sqrt{P_{\text{ex}}(1-P_{\text{ex}})/n}$, set before the runs.

**Asserted tests and report outputs.** Every number below is one of two kinds. **Asserted** numbers come from slow tests that fail if the rule (or the stated bound) is broken: `tests/validation/test_tully_exact.py` (`pytest -m slow`) and the single-trajectory SHARC tests of `tests/validation/test_sharc_tully.py` (`pytest -m sharc`). Numbers marked **(report)** are outputs of two scripts that are run by hand, assert nothing and write every number to a JSON file: `tools/validation/tully_report.py` (the unasserted FSSH-vs-exact ensembles) and `tools/sharc_compare/tully_ensembles.py` (the SHARC ensembles). The report numbers were measured with the same code while it was still part of the slow tests (Tasks 8 and 9); the final fix wave moved it out, so that the asserted set contains only checks that discriminate.

**(a) Against the exact quantum solution.** The reference is our own exact solver (a test helper, `tests/validation/wavepacket.py`): a two-state split-operator FFT propagation in the diabatic basis (symmetric Strang step, $2\times2$ exponential from the eigendecomposition of $\mathbf V$ at every grid point), starting from the same packet, with an absorbing boundary near the grid edge whose absorbed density is projected on the local adiabatic states and added to that side's branch, so the bookkeeping is exact. Convergence (the 10 asserted points, asserted; DAC $\ln E = -2$, report): halving $\Delta x$ and $\Delta t$ changes the branches by at most $9.8\times10^{-6}$, a wider grid and absorber by at most $9.4\times10^{-6}$, and the norm is $1$ to $3\times10^{-12}$. SAC at $k = 30$: exact 0.714 against Landau–Zener 0.72.

FSSH settings: `hopping = "sharc"` (except the `"fssh"` runs below), `parallel_vel`, frustrated hops kept, $\Delta t = 0.5$ a.u., 25 substeps, 8 workers, seed 20261108, x0 = −10 (SAC, DAC) or −15 (ECR), 95 % Wilson intervals. "A" marks asserted branches.

*SAC, no decoherence, n = 2000* (asserted at $k$ = 10, 15, 20, 30):

| $k$ | transmitted lower | exact | transmitted upper | exact |
|---|---|---|---|---|
| 10 | 0.8530 | 0.8456 | 0.1450 [0.130, 0.161] | 0.1533 |
| 15 | 0.6785 | 0.6770 | 0.3215 [0.301, 0.342] | 0.3230 |
| 20 | 0.5245 | 0.5077 | 0.4755 [0.454, 0.497] | 0.4923 |
| 30 | 0.3030 | 0.2857 | 0.6970 [0.677, 0.717] | 0.7143 |

The largest difference is 0.017; the reflected branches are at most 0.002 (exact 0.001 at $k$ = 10).

*DAC, no decoherence, n = 1000* ($\ln E$ = −1, 0, 1 asserted; $\ln E = -2$ report; mean energy $E = k^2/2m$ in $E_h$, so $k = \sqrt{2mE}$):

| $\ln E$ ($k$) | transmitted lower | exact | transmitted upper | exact |
|---|---|---|---|---|
| −2 (23.27) | 0.799 | 0.9068 | 0.201 [0.177, 0.227] | 0.0932 |
| −1 (38.36) | 0.617 | 0.6078 | 0.383 [0.353, 0.414] | 0.3922 |
| 0 (63.25) | 0.753 | 0.7449 | 0.247 [0.221, 0.275] | 0.2551 |
| 1 (104.27) | 0.151 | 0.1315 | 0.849 [0.826, 0.870] | 0.8685 |

The three asserted points differ by at most 0.020. **The point $\ln E = -2$ is not asserted: FSSH 0.201 against exact 0.093 (difference 0.108, tolerance 0.057).** It is the low-energy Stückelberg region, where FSSH without decoherence is known to deviate; an independent from-scratch FSSH (own adiabatic formulas, RK4 electrons, no `gqteashmd.dynamics` code, same streams) gives the same value at fixed $k$, so this is a property of the method. **Stated plainly: this point was classified as reported-only after the preliminary numbers had been seen**, not before. There is no reflection in any DAC run.

*Tully's flux probability (`hopping = "fssh"`), no decoherence* (asserted since the final fix wave; transmitted lower and upper, same rule; n = 2000 for SAC, 1000 for DAC; measured 2026-11-10):

| point | transmitted lower | exact | transmitted upper | exact | tolerance |
|---|---|---|---|---|---|
| SAC $k = 10$ | 0.8530 | 0.8456 | 0.1450 [0.130, 0.161] | 0.1533 | 0.052 |
| SAC $k = 20$ | 0.5235 | 0.5077 | 0.4765 [0.455, 0.498] | 0.4923 | 0.065 |
| DAC $\ln E = 0$ | 0.753 | 0.7449 | 0.247 [0.221, 0.275] | 0.2551 | 0.075 |

The largest difference is 0.016. The trajectories use the same random streams as the `"sharc"` runs above, and at SAC $k = 10$ and DAC $\ln E = 0$ the counts are identical to them (at SAC $k = 20$ they differ by 2 of 2000): the two probabilities estimate the same population flux and differ only at higher order in the time step, so the same random number rarely decides differently. This agreement is expected, not an independent statistical check of the two schemes; each still has to meet the rule against the exact result on its own.

*ECR, Granucci–Persico EDC, n = 1000* (asserted: transmitted upper and reflected lower):

| $k$ | transmitted upper (A) | exact | reflected lower (A) | exact | tolerance (R lower) |
|---|---|---|---|---|---|
| 10 | 0.000 | 0.0000 | 0.069 | 0.0898 | 0.056 |
| 20 | 0.000 | 0.0000 | 0.104 | 0.1570 | 0.066 |
| 30 | 0.342 | 0.3742 | 0.013 | 0.0238 | 0.039 |

The tightest margin is $k = 20$, reflected lower: difference −0.053 against 0.066. **At $k$ = 10 and 20 the upper channel is closed** (the kinetic energy, 0.025 and 0.1 $E_h$, is below the upper asymptote, about 0.2 $E_h$), so exact and FSSH both give 0 for transmitted upper and those two assertions are met trivially; only $k = 30$ tests that channel. The other ECR branches at these momenta are printed by the test but not asserted (transmitted lower 0.701 / 0.647 / 0.606 against exact 0.7004 / 0.6038 / 0.5697; reflected upper 0.230 / 0.249 / 0.039 against 0.2098 / 0.2392 / 0.0323). *ECR with `edc_convention = "sharc"`, n = 1000*, run first as report-only and then asserted on the same branches because every branch at all three momenta met the rule (**this assertion was added after the numbers were seen**): reflected lower 0.0830 / 0.1200 / 0.0150 (differences −0.007 / −0.037 / −0.009; at $k = 20$ closer to exact than the Granucci–Persico −0.053), transmitted upper 0.000 / 0.000 / 0.3590.

*Why SAC and DAC are asserted without decoherence.* The accuracy claim for FSSH on these two models (Tully 1990) is for fewest-switches surface hopping **without decoherence** (Tully's plain-FSSH setting). The asserted runs use that setting with the default SHARC-type (propagator-based) hopping probability, and at three points also with Tully's flux probability (`hopping = "fssh"`, below); without decoherence both probabilities implement the same fewest-switches method. The EDC runs (report; Granucci–Persico, $C$ = 0.1, n = 500) are biased: SAC transmitted upper 0.210 / 0.416 / 0.530 / 0.680 at $k$ = 10 / 15 / 20 / 30 (exact 0.1533 / 0.3230 / 0.4923 / 0.7143), and DAC transmitted upper 0.324 / 0.430 / 0.488 / 0.524 at $\ln E$ = −2 / −1 / 0 / 1 (exact 0.0932 / 0.3922 / 0.2551 / 0.8685). The damage was traced to the EDC, not to a bug: with decoherence off the core reproduces an independent plain-FSSH implementation (Tully's flux formula, no `gqteashmd.dynamics` code) count for count (SAC $k = 10$: 57/400 and DAC $k$ = 23.27: 66/400, identical).

**(b) Against published values.** **Not done: skipped by design.** No open-access source tabulates the branching probabilities of Tully's models (the figures of Tully 1990, Mannouch & Richardson, *J. Chem. Phys.* **158**, 104111 (2023) and He et al., *J. Phys. Chem. Lett.* **13**, 6966 (2022) are not tabulated, and those papers use other conventions). No value was invented or read off a plot; the test skips with this explanation. The exact solver of (a) takes its place. Qualitatively our exact ECR values lie on the trend of Fig. 8 of Mannouch & Richardson.

**(c) Against SHARC 4.1 as a black box** (`pytest -m sharc`, needs `SHARC_HOME` and `SHARC_PYTHON`; never in CI). Clean room: SHARC is **executed only**; its input formats and equations come from the public SHARC 4.1 manual, and nothing in the SHARC installation was read. The harness is `tools/sharc_compare/run_sharc_model.py` (a one-atom model, He, with the Tully potentials; `sharc.x` with `SHARC_ANALYTICAL`; for long ensembles our own provider writes `QM.out` in the documented format, shown equal to `SHARC_ANALYTICAL.py` to better than $10^{-9}$ on SAC, DAC and ECR). SHARC's 4.1 `driver.py` ANALYTICAL mode did not work on this installation, so `sharc.x` is used.

- **Nuclear motion and hopping probabilities** (SAC, $x$ from −1 to 1, $k = 40$, $\Delta t = 0.5$ a.u., 200 steps, hopping off): positions agree to $7.5\times10^{-11}$ bohr and velocities to $2.1\times10^{-12}$ (a.u.). The populations agree **only with `substep_interpolation = "endpoint"`**: $\max\big||c_0|^2_{\text{ours}} - |c_0|^2_{\text{SHARC}}\big|$ is $1.4\times10^{-8}$ over the first 50 steps (asserted $< 10^{-6}$) and $6.15\times10^{-7}$ over all 200 (asserted $< 2\times10^{-6}$). With the default `"midpoint"` it is $2.1\times10^{-5}$ over 50 steps and $2.5\times10^{-4}$ over 200 (peak at the crossing), the difference halves when the number of substeps doubles (ratio 2.003), and the final populations differ by $2\times10^{-7}$. The remaining $6\times10^{-7}$ is SHARC against its own documented scheme (propagating SHARC's own energies, couplings and velocities with the manual's scheme reproduces SHARC to $6.2\times10^{-7}$ and our core to $6.0\times10^{-9}$).
- **A replayed hop** (SAC, $x$ from −2 to 2, $k = 40$, $\Delta t = 1$ a.u., SHARC's own random numbers fed to our core): the same hop at step 136 (S0→S1), the same active-state sequence, velocities after the `parallel_vel` rescaling to $7.8\times10^{-11}$, and hop probabilities to $6.5\times10^{-7}$ with `"endpoint"` ($8.5\times10^{-5}$ with `"midpoint"`).
- **EDC rate:** SHARC's inactive amplitudes decay with $\kappa = 0.500000$ (relative to $e^{-\Delta t/\tau}$); our `"sharc"` convention gives 0.500000 and `"granucci_persico"` 1.000000. Inactive populations at steps 150 / 200 / 300: SHARC 0.250656 / 0.121657 / 0.026135, ours (`"sharc"`) 0.250688 / 0.121669 / 0.026138, ours (`"granucci_persico"`) 0.302264 / 0.066121 / 0.002998.
- **Ensembles (report, `tools/sharc_compare/tully_ensembles.py`)** ($n = 500$, the same Wigner initial conditions fed to both programs, SHARC hopping, `parallel_vel`, frustrated hops kept, $\Delta t = 0.5$ a.u., 25 substeps; two-proportion $z$; until the final fix wave a slow test asserted $|z| < 4$, a criterion that could not detect the factor 2 of the EDC rate, which the single-trajectory EDC test above measures directly): SAC $k = 20$ without decoherence gives identical counts, 261 / 239 / 0 / 0 (transmitted lower / transmitted upper / reflected lower / reflected upper; $z = 0.00$ on every branch). ECR $k = 20$ with EDC `"sharc"` ($C$ = 0.1): transmitted lower 318 against SHARC 304 ($z = +0.91$), transmitted upper 0 against 0, reflected lower 55 against 72 ($z = -1.61$), reflected upper 127 against 124 ($z = +0.22$); the largest $|z|$ is 1.61. With the Granucci–Persico convention the largest was 2.44 (reflected lower, 47 against 72), which is what led to the two options. The identical SAC counts are a coincidence of the statistics, not a shared random stream (for 40 trajectories only 20 labels agreed).

**Run times of the validation** (measured 2026-11-10, code on `/mnt/c`, 8 workers). Asserted: `pytest -m slow tests/validation/test_tully_exact.py` (26 tests: the exact-solver convergence and every asserted FSSH-vs-exact point above) takes about **94 minutes**, the two SAC $k = 10$ ensembles alone 29 minutes; `pytest -m sharc` (all SHARC comparisons, 16 passed and the documented 3000 K `wigner.py` xfail) takes about **4 minutes**, of which the single-trajectory Tully tests are 3.3 minutes. Report scripts (run by hand): `tools/validation/tully_report.py` about 20–30 minutes and `tools/sharc_compare/tully_ensembles.py` about 1 hour (SHARC is about 15 times slower per trajectory than our core: a file round trip and an awk process per step). None of them is part of the fast tier or of CI.

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
| `Stage '…' needs pyscf >= 2.6; version … is installed. Update the conda environment (environment.yml).` (also `geometric >= 1.0`, `h5py >= 3.9`) | Preflight: a module is too old (exit 2, §7.9) | Update the conda environment |
| `Stage 'optimize' needs an [electronic] section in the input file` (also for `frequencies`; several stages are joined by `; `) | `validate` or `run` with no `[electronic]` section (exit 2) | Add the section (§4.2) |
| `frequencies.method = 'analytic': there is no analytic SA-CASSCF Hessian in PySCF; use 'auto' or 'finite_difference'` | `analytic` together with `electronic.method = "casscf"` (exit 2) | Use `"auto"` or `"finite_difference"` |
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
| `The geometry optimization did not converge in N steps (convergence = '…'). Increase [optimize] max_steps, start from a better geometry, or loosen [optimize] convergence; the lowest-energy geometry found is in …/best.xyz` | The `optimize` task failed (not retried); the stage stops with `Stage 'optimize' failed: 1 of 1 tasks did not succeed` (exit 4) | Follow the advice. `optimization_path.xyz` and `best.xyz` (the lowest-energy geometry evaluated, not necessarily the last) are in the task directory |
| `The geometry optimizer (geomeTRIC) failed: … Check the starting geometry …` | geomeTRIC itself failed (internal coordinates, structure); not retried (exit 4). Errors from the electronic-structure calculation are reported as such, not with this message | Check the starting geometry, or disable `[optimize]` and provide an optimized geometry |
| `The finite-difference energies disagree with the reference gradient by … Eh/bohr` / `the energy at the optimized geometry … differs from the optimize stage's energy` | The consistency checks of `frequencies` (§8.7) failed (exit 4, `Stage 'frequencies' could not finalize its results: …`) | Check the active space, reduce `fd_step_bohr`, or re-run from `optimize` |
| `The structure is not a minimum: N imaginary mode(s) beyond 20 cm^-1 (mode 1: -2329.0 cm^-1, …). Re-optimize …` | `frequencies` found imaginary modes beyond the cutoff (exit 4, §8.7). The files are written first | Re-optimize with a tighter `convergence`, check the starting structure, or raise `imaginary_cutoff_cm1` |
| `The analytic SCF Hessian failed: …` | PySCF failed in the Hessian step (exit 4; retried only for `MemoryError`/`OSError`) | Read `electronic.log` in the task directory; try `frequencies.method = "finite_difference"` |
| `Invalid input file …:` with `sampling.n_samples` / `sampling.temperature_K` / `sampling.low_frequency_cutoff_cm1` text (for example `Input should be greater than 0`) | A `[sampling]` value out of range, or an unknown key (exit 2, §4.2) | Correct the key |
| `No vibrational mode is sampled: every mode is below the cutoff 10 cm^-1 (frequencies: …). Please lower [sampling] low_frequency_cutoff_cm1.` | All modes of the molecule are below `low_frequency_cutoff_cm1` (exit 4; the stage's files are not written) | Lower the cutoff (it must stay above 0) |
| `Stage sampling: N mode(s) below the cutoff are not sampled (frequencies in cm^-1: …)` | **Warning**: modes below the cutoff, including small imaginary modes, stay at equilibrium with zero velocity (§8.8); the run continues | Lower `low_frequency_cutoff_cm1` if you want those modes sampled |
| `The sampled harmonic energies are inconsistent with the Wigner distribution (mode …: sum of E/<E> over … samples = …, expected Gamma(…, 1); two-sided p = … < …); this indicates a program error — please report it` | The energy of one mode is outside the exact Gamma tail limit $10^{-9}/n_{\text{sampled}}$ (exit 4, §8.8). `samples.json` is written for diagnosis | Report it with the log |
| `Total momentum was not removed from the samples (max |P| = …, max |L| = … a.u.); this indicates a program error — please report it` | The momentum check of §8.8 failed (exit 4) | Report it with the log |
| `the excitations stage needs at least one excited state: [electronic] singlets must be >= 2` | `singlets = 1` (only S0) with the `excitations` stage (exit 2, found by `validate`) | Set `singlets` to at least 2 (§4.2) |
| `Invalid input file …:` with `selection.…` text (for example `selection.states: S5 does not exist with [electronic] singlets = 3 (S0..S2)`, `selection.mode = 'fixed' needs exactly one state in selection.states`, `S0 cannot be selected: initial states are excited singlets`, `energy_window_eV needs finite values with 0 <= lo < hi`) | An inconsistent or out-of-range `[selection]` key; `[excitations]` keys out of range are reported the same way (exit 2, §4.2) | Correct the keys |
| `Stage 'excitations' could not finalize its results: N of M samples failed, more than max_failed_fraction = 0.05: sample 7: …` | Too many samples failed (exit 4, §8.9.1); the first five failures are listed, and `excitations.json` (written before the stop) lists all | Read the messages and the `electronic.log` of the tasks (`04_excitations/tasks/`); fix the cause (geometry, active space, convergence; a changed input then needs the `--rerun-from` stage that the next run names), or raise `max_failed_fraction` and run again (it acts in the finalization only: no `--rerun-from` needed) |
| `Stage excitations: sample N failed: …` | **Warning**: one sample's electronic structure failed; the task goes on (§8.9.1) | None, unless the final count exceeds `max_failed_fraction` |
| `the equilibrium geometry itself failed: … Fix the electronic-structure settings first (then run with the --rerun-from stage that the next run names); … Or set [excitations] include_reference = false and run with --rerun-from excitations, which recomputes every sample.` | The reference calculation failed, or its result is unusable (exit 4; `excitations.json` is written first, with `reference_error`) | Fix the `[electronic]` settings (test with `single-point`) and rerun as the stale-stage message says, or set `include_reference = false` and run with `--rerun-from excitations` (all samples are recomputed) |
| `Stage excitations: sample k failed: excitation energy of Sn is not positive (… eV)` / `… non-finite energy of Sn …` / `… non-finite transition dipole of Sn …` / `… non-finite oscillator strength of Sn …` | **Warning**: the sample's result cannot be used (an excited state at or below S0, or a NaN/inf value), so it counts as a failed sample (§8.9.1) | A few are tolerated by `max_failed_fraction`; many point to an unstable reference (TDA/TDDFT) or a poor active space |
| `samples.h5 holds K samples but [sampling] n_samples = N; rerun the sampling stage with --rerun-from sampling` | `samples.h5` does not match the input (exit 4; not retried) | Run with `--rerun-from sampling` |
| `samples.h5 is missing or unreadable …; rerun the sampling stage with --rerun-from sampling` / `samples.h5 is inconsistent: dataset … has shape …` / `excitations.h5 is missing or unreadable …; rerun the excitations stage with --rerun-from excitations` | An upstream HDF5 file was deleted, damaged or written by another version (exit 4; not retried) | Run the suggested `--rerun-from` command |
| `Only K of N requested trajectories can be selected (candidates: …, accepted: …, f_max: …, weighting: …, window: …, states: …). Widen energy_window_eV, add states or lower n_trajectories (then run with --rerun-from selection); or increase [sampling] n_samples (then run with --rerun-from sampling, which recomputes the excitations of every sample); or set allow_fewer = true and run again (no --rerun-from needed). See …/selection_report.md.` | Fewer trajectories than requested can be selected (exit 4, §8.9.3). With zero selected the advice does not mention `allow_fewer`, because zero is always an error. If one pair dominates $w_{\max}$, a `Note: the w_max pair …` sentence is added | Follow the advice; read `selection_report.md`, which is written before the stop |
| `Stage selection: the w_max pair (sample k, Sn, dE = … eV, f = …, w = …) is … times the 99th percentile of the candidate weights (…), so it suppresses the acceptance of every other pair; set [selection] energy_window_eV to exclude it (then run with --rerun-from selection).` | **Warning**: one low-gap pair sets the normalization of `f_over_dE2` (§8.9.3) | Set an `energy_window_eV` around the band you want to excite, then `--rerun-from selection` |
| `excitations.h5 does not match [selection]: state Sn is not in the excitation data (excited singlets S1..Sm)` | `excitations.h5` has fewer states than `selection.states` asks for (exit 4; not retried; normally caught earlier by `validate`) | Run with `--rerun-from excitations` |
| `Stage selection: only K of N requested trajectories were selected (allow_fewer = true)` | **Warning**: `allow_fewer = true` accepted a smaller number (§8.9.3) | Check that the number is enough for your study |
| `Stage … results directory … is missing; recomputing` | **Warning**: you deleted a stage directory (§7.3) | None; the stage is computed again |
| `Stage frequencies: small imaginary modes …` / `translation/rotation contamination …` / `not a stationary point …` | **Warnings** from `frequencies` (§8.7); the run continues | Check the geometry and the Hessian |
| `internal error: …` | A program bug | Report it with `logs/gqteashmd.log` and the task's `result.json` |

---

## 10. Known limitations

- **Feature scope:** the stages `prepare`, `optimize`, `frequencies`, `sampling`, `excitations` and `selection` exist. The `dynamics` stage and analysis are planned (P7b, P8), so the workflow ends with the selected initial conditions (`trajectories.json`) today. For that reason `pause_after = ["selection"]` has no effect yet: the last stage never pauses (§7.8); `pause_after = ["excitations"]` works.
- **`sampling`:** harmonic and independent modes only (§8.8); no anharmonic or quasi-classical sampling, no sampling of an isotopically substituted molecule (the masses come from `frequencies`), samples are drawn on the S0 geometry of `frequencies` and not yet used by any later stage; the samples are not checked for chemical sense (very large amplitudes at high temperature or for floppy modes just above the cutoff are not detected).
- **Electronic structure:** closed-shell singlet molecules with a restricted reference only; no triplet states; nonadiabatic couplings (SA-CASSCF only) and wavefunction overlaps exist only as a library and are not used by any command or stage yet (P7); the analytic Hessian exists for RHF/RKS only (SA-CASSCF uses finite differences, §8.7); TDA/TDDFT have no analytic couplings, and their overlaps use the excitation amplitudes X only (§8.6.8); the cost of the overlap engine grows with the number of determinants kept (`overlap_norm_kept`); overlap-based couplings carry noise of order $10^{-7}/\Delta t$ at default CASSCF convergence; no density fitting; no scratch-directory input key yet (the backend supports one, and writes no checkpoint file); TDDFT/TDA describe the S₁/S₀ crossing seam incorrectly (§8.6.4); DFT gradients omit the grid response, so use `grid_level` ≥ 5 for dynamics (§8.6.4). The SCF, CASSCF and TD calculations each have a retry ladder (§8.6.1, §8.6.8); a calculation that fails every rung stops.
- **`excitations` and `selection`:** the spectrum is the vertical (nuclear-ensemble) spectrum with a Gaussian broadening: no vibronic structure, no refractive-index correction, no plot (`spectrum.csv` only), and it is not compared with an experiment. Each sample is computed from scratch (a full SCF and excited-state calculation per sample), with no parallelism inside a sample; the cost is $N_s+1$ calculations at the `[electronic]` level. The states are the adiabatic roots in order of energy at each geometry (no tracking between samples). Only singlets (S1, S2, ...) can be selected; no triplets. The bright-mode default `f_over_dE2` follows SHARC 4.1 `excite.py` as measured by black-box maximum likelihood (§8.9.4), not a published formula; it favours low-energy pairs, so one near-degenerate sample can dominate the normalization without an `energy_window_eV` (a warning is logged, §8.9.3). **SA-CASSCF:** every sample starts from SCF orbitals (`reset()` before each sample), so a displaced geometry can converge to a different active space than the reference, and nothing checks for this yet (a natural-occupation or energy-continuity check against `ref` is a P9 candidate); compare the per-state means in `excitations.json` with the reference values. The sample's total energy is not adjusted to the vertical excitation energy. Failed samples are excluded, which can bias the ensemble if failures cluster at certain geometries (the number of failures is in `excitations.json`).
- **`optimize` and `frequencies`:** they work on the S0 surface of `[electronic]` only (no excited-state optimization, no constraints, no transition-state search). There is no input key for isotope masses, and no thermochemistry (zero-point energy, entropy), anharmonicity or frequency scaling. A finite-difference Hessian costs $1+6N_{\text{at}}$ gradients, which is large for SA-CASSCF with many atoms. The check of the project directory covers the scratch space; there is no separate check that the disk has room for PySCF's scratch files beyond `min_free_disk_mb`. A refused stale run still appends a provenance record (§7.5).
- **Dynamics core (P7a-1):** it runs only on analytic models (Tully's three 1D models and a 2D test model) through `gqteashmd model-dynamics`; no molecule, no PySCF, no project, no `dynamics` stage yet (P7b). Only NAC-vector couplings (`nacdr`) exist: no overlap-based couplings or local diabatization, no spin–orbit coupling (a non-diagonal MCH→diagonal matrix $\mathbf U$ raises `NotImplementedError`), no LVC model (all P7a-2). Fixed time step, no thermostat, no A-FSSH/NPI/GFSH/`parallel_diff`. **EDC at $C = 0.1\,E_h$ biases the Tully benchmarks** (§8.12, §8.14.1), and the defaults mix conventions: the EDC follows SHARC but the substep interpolation is `"midpoint"` (SHARC's is `"endpoint"`), so populations differ from SHARC at the $10^{-4}$ level during a crossing unless you choose `"endpoint"`. The restart is a library function (not used by `model-dynamics`); it stores no model identity, options or $\mathbf U$, and the caller must supply the same ones. Published branching values were not available in tabulated form (§8.14.1(b)); DAC at $\ln E = -2$ deviates from the exact result (a property of FSSH). The asserted validation takes about 94 minutes (exact comparison) and 4 minutes (`pytest -m sharc`), the report scripts about 20–30 minutes and 1 hour (§8.14.1); none of it is in the fast test tier.
- **Platform:** Linux only, with no batch-scheduler support. Runs are designed for a single workstation.
- **Project-folder check:** the check for Windows drives and OneDrive (§7.9) looks at the text of the path. An absolute path that is a symbolic link into a Windows drive or a OneDrive folder is not detected.
- **SHARC comparison is not automated:** the tests against SHARC 4.1 need a SHARC installation and run only on the development machine (`pytest -m sharc`); the continuous-integration workflow (first run 2026-11-07, passing) runs everything else.

## 11. Roadmap

P1 foundation (done) → P2 input schema and pre-run checks (done) → P3a PySCF backend (done: energies, gradients, transition dipoles, `single-point`) → P3b couplings, overlaps, phase tracking and restarts (done, library only) → P3c hardening sweep (done) → P4 optimization, frequencies and Hessian (done: `optimize` and `frequencies` stages) → P5 Wigner sampling (done: `sampling` stage, validated against SHARC `wigner.py`) → P6 excitations and initial states (done: `excitations` and `selection` stages, compared with SHARC `SHARC_PYSCF.py` and `excite.py`) → P7a-1 dynamics core on analytic models (done on its branch: velocity Verlet, propagators, Tully/SHARC hopping, velocity adjustment, EDC, restarts, `model-dynamics`, validated against exact wavepackets and SHARC 4.1) → **P7a-2 (next): LVC model, local diabatization from overlaps, spin–orbit coupling, SHARC LVC comparisons** → P7b `dynamics` stage with PySCF → P8 analysis → P9 systematic validation against SHARC 4.1. This manual is updated with each step. Sections marked [PLANNED] become [IMPLEMENTED] when their sub-project is finished and validated.

## 12. References
1. J. C. Tully, *J. Chem. Phys.* **93**, 1061 (1990): fewest-switches surface hopping.
2. G. Granucci, M. Persico, A. Toniolo, *J. Chem. Phys.* **114**, 10608 (2001): local diabatization.
3. G. Granucci, M. Persico, *J. Chem. Phys.* **126**, 134114 (2007): energy-based decoherence correction.
4. S. Mai, P. Marquetand, L. González, *WIREs Comput. Mol. Sci.* **8**, e1370 (2018): the SHARC approach.
5. E. Wigner, *Phys. Rev.* **40**, 749 (1932); M. Barbatti, K. Sen, *Int. J. Quantum Chem.* **116**, 762 (2016): Wigner sampling for initial conditions.
   M. Hillery, R. F. O'Connell, M. O. Scully, E. P. Wigner, *Phys. Rep.* **106**, 121 (1984): distribution functions in physics (the thermal Wigner function of the harmonic oscillator); L. Sun, W. L. Hase, *J. Chem. Phys.* **133**, 044313 (2010): classical and Wigner sampling of vibrational energy levels.
6. Q. Sun et al., *J. Chem. Phys.* **153**, 024109 (2020): PySCF.
7. CODATA 2018 recommended values of the fundamental physical constants.
8. E. B. Wilson, J. C. Decius, P. C. Cross, *Molecular Vibrations* (McGraw-Hill, 1955): normal-coordinate analysis.
9. C. Eckart, *Phys. Rev.* **47**, 552 (1935): the translation and rotation conditions.
10. R. Crespo-Otero, M. Barbatti, *Theor. Chem. Acc.* **131**, 1237 (2012): spectrum simulation and decomposition with the nuclear ensemble (§8.9.2).
11. M. Barbatti et al., *J. Photochem. Photobiol. A* **190**, 228 (2007): initial conditions for surface-hopping dynamics and the oscillator-strength criterion for the choice of the initial state (§8.9.3).
12. J. C. Tully, *J. Chem. Phys.* **93**, 1061 (1990) (reference 1) also defines the three one-dimensional models of §8.14.
13. S. Mai, P. Marquetand, L. González, *Int. J. Quantum Chem.* **115**, 1215 (2015): the propagator-based (SHARC-type) hopping probability of §8.12.
14. J. E. Runeson, arXiv:2507.18381 (2025): a statement of the energy-based decoherence correction (Sec. III.3, eqs. 44-45) and its critique (dependence on the total kinetic energy).
15. SHARC 4.1 manual (public user documentation, sharc-md.org): §8.7.1 (EDC, eqs. 8.25-8.26) and §8.33.1 (electronic propagation, eqs. 8.191-8.194), and the input formats used to run SHARC as a black box (§8.14.1(c)). SHARC's source code was not read.
16. S. N. Mannouch, J. O. Richardson, *J. Chem. Phys.* **158**, 104111 (2023), and L. He et al., *J. Phys. Chem. Lett.* **13**, 6966 (2022): published FSSH results on Tully's models (figures only; not used as numerical references, §8.14.1(b)).
