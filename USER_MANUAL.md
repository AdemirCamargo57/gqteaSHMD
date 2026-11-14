# gqteashmd — User Manual

**Version:** 0.1.0.dev0 (development snapshot, 2026-11-13)
**Program:** `gqteashmd`, automated surface-hopping molecular dynamics with PySCF
**Licence:** MIT

> **Read this first: what works today.** This is an early development version. The **workflow engine is complete**: input file (with hardware and per-stage resource settings), checks before a run (preflight), project management, parallel tasks, crash-safe resume, graceful stop, review pauses, change detection, and a record of the software used (provenance). Eight stages exist: **`prepare`** (validates and stores the molecule), **`optimize`** (optional S0 geometry optimization with geomeTRIC), **`frequencies`** (S0 Hessian, harmonic frequencies, normal modes, Molden file), **`sampling`** (harmonic Wigner sampling of initial geometries and velocities, from the frequencies), **`excitations`** (vertical excitation energies and oscillator strengths at every sample, and the absorption spectrum) **`selection`** (the initial excited state of every trajectory), since P7b **`dynamics`** (one surface-hopping trajectory per selected initial condition, with PySCF; it runs only if the input has a `[dynamics]` table, §4.2, §5.8, §8.16) and, since P8, **`analysis`** (state populations with 95 % intervals, a delayed mono-exponential lifetime fit with bootstrap uncertainties, hop statistics and energy diagnostics over the trajectories; it runs after `dynamics` by default, §4.2, §5.9, §8.13). `optimize`, `frequencies`, `excitations` and `dynamics` work at the level of the `[electronic]` input section. The **electronic-structure layer** (PySCF: SCF, SA-CASSCF, TDA, TDDFT, CIS, TDHF; energies, analytic gradients, transition dipoles, the S0 Hessian) is **implemented** and validated against SHARC 4.1 and other programs; you can also use it directly with **`gqteashmd single-point`** (§5.3). The **surface-hopping dynamics core** (velocity Verlet, electronic propagation, Tully and SHARC-type hopping, velocity adjustment, decoherence, restarts) is **implemented and validated on Tully's analytic models** (P7a-1) and, since P7a-2, on **linear vibronic coupling (LVC) models with singlets, triplets and spin–orbit coupling** (propagation and hopping in the diagonal representation, local diabatization from overlaps, energy-based selection of gradients), validated against exact wavepackets and against SHARC 4.1 run as a black box: run it with **`gqteashmd model-dynamics`** (§5.7, §8.10–§8.12, §8.14, §8.15). **The `dynamics` stage (P7b) runs the core on molecules with PySCF, for singlets only**: local diabatization from wavefunction overlaps (every method) or SA-CASSCF nonadiabatic couplings, step rejection with step halving, a restart after every accepted step, outcomes per trajectory, and the extension of finished trajectories by raising `tmax_fs` (§8.16). **The `analysis` stage (P8) turns the trajectories into populations and a lifetime** with a fixed inclusion rule (only trajectories that ran to the end enter the averages; the others are listed, §8.13); it is validated on synthetic ensembles with known answers, not against SHARC (P9). The documented real study of §5.9 (ethylene SA2-CASSCF(2,2)/6-31G\*, 20 trajectories) shows that the workflow runs unattended and survives a kill, **and that its physics at these settings is not usable** (2 of 20 trajectories reach the end; the P9a diagnosis of these losses is in §8.16.9). **Since P9a plan 2** the `dynamics` stage rejects a step whose total energy jumps by more than `energy_jump_limit_eV` (default 0.1 eV) and ends a persistent jump with the outcome `electronic_discontinuity` (§8.16.3); with SA3-CASSCF(2,2) the ethylene acceptance study kept 13 of 20 trajectories, short of its target of 16 (§5.10), and P9a was closed with that result documented. **Since the CASSCF start rule** every SA-CASSCF calculation that has no previous orbitals starts from a fixed reference solution computed at the stage's reference geometry, projected onto the new geometry, instead of from that geometry's SCF orbitals (§8.6.2). **If you have a `casscf` project computed before the start rule, run `--rerun-from optimize`** (or `--rerun-from frequencies` if `optimize` is disabled): the fingerprints of the CASSCF stages changed, so the run stops with exit 3 until you do (§7.5); TDA/TDDFT projects are not affected. **If you have a project with `dynamics` results computed before P9a plan 2, run `--rerun-from dynamics`:** the stage version is now 2, so the run stops with exit 3 until you do (§7.5). **Triplets and spin–orbit matrix elements from PySCF are *planned*** (PySCF provides none yet, so spin–orbit dynamics runs only on LVC models). **If you have a project that was computed before P7b, run `--rerun-from excitations`:** the TDA/TDDFT root fix (§8.6.3) changed the version of `excitations`, so the run stops with exit 3 until you do (§7.5). Every section is marked:
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

`gqteashmd` runs a complete photodynamics study from **one input file**, from the XYZ geometry to state populations and a lifetime; all eight stages below exist today. The stages run in this order:

| Order | Stage | What it does | Status |
|---|---|---|---|
| 0 | `prepare` | Reads and checks the molecule (XYZ syntax, elements, charge and multiplicity); stores a normalized copy | **[IMPLEMENTED]** |
| 1 | `optimize` (optional) | Ground-state (S0) geometry optimization with geomeTRIC, on the surface of the `[electronic]` section | **[IMPLEMENTED, P4]** |
| 2 | `frequencies` | S0 Hessian, harmonic frequencies and normal modes (analytic Hessian or finite differences of gradients); Molden, HDF5 and JSON output | **[IMPLEMENTED, P4]** |
| 3 | `sampling` | Harmonic Wigner sampling of initial geometries and velocities, at zero or finite temperature | **[IMPLEMENTED, P5]** |
| 4 | `excitations` | Vertical excitation energies and oscillator strengths at the reference and at each sampled geometry; nuclear-ensemble absorption spectrum | **[IMPLEMENTED, P6]** |
| 5 | `selection` | Choice of the initial electronic state for each trajectory (bright-state rule or one fixed state) | **[IMPLEMENTED, P6]** |
| 6 | `dynamics` (optional: runs only if the input has a `[dynamics]` table) | Surface-hopping trajectories, one task per trajectory, run in parallel; PySCF, singlets only | **[IMPLEMENTED, P7b]** (§4.2, §5.8, §8.16; the core: §5.7, §8.10–§8.15) |
| 7 | `analysis` (runs after `dynamics` unless `[analysis] enabled = false`) | Classical and quantum state populations with 95 % intervals, a delayed mono-exponential fit ($t_0$, $\tau$) with bootstrap intervals, hop statistics, energy diagnostics; CSV, HDF5, JSON and Markdown (no plots) | **[IMPLEMENTED, P8]** (§4.2, §5.9, §6, §8.13) |

In SHARC 4.1 these steps are separate manual tools: `wigner.py`, `setup_init.py`, `excite.py`, `setup_traj.py`, `populations.py` and `make_fit.py`. `gqteashmd` runs them automatically, one after another. You only intervene when the program asks you to fix something.

The electronic-structure program is **PySCF** **[IMPLEMENTED]**: the `[electronic]` input section (§4.2), the command `gqteashmd single-point` (§5.3) and the stages `optimize`, `frequencies` and `excitations` use it today. The stage `dynamics` (P7b) uses it too, with wavefunction overlaps (every method) or SA-CASSCF nonadiabatic couplings (§8.6.8, §8.16). ORCA, Gaussian and others may be added later.

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

# [dynamics]                         # remove the leading '# ' to run trajectories
# tmax_fs = 100.0                    # must be a multiple of dt_fs
# dt_fs = 0.5
# coupling = "overlap"               # "nacdr" needs method = "casscf"
# hopping = "sharc"
# decoherence = "edc"
# energy_drift_limit_eV = 0.5
# energy_jump_limit_eV = 0.1         # reject a step whose E_tot jumps by more

# [analysis]                         # optional; runs after dynamics by default
# fit_states = ["S1", "S2"]          # fitted population (default: every excited state)
# fit_population = "classical"       # or "quantum"
# n_bootstrap = 1000
# energy_jump_eV = 0.1               # flag trajectories with a larger E_tot jump
# exclude_energy_jumps = false
```

The template's `[electronic]` block is active (TDA-B3LYP/def2-SVP with S0 and two excited states), because `optimize` and `frequencies` need it, and so does `single-point`; the CASSCF alternative is given as comments. `[optimize]`, `[frequencies]`, `[sampling]`, `[excitations]` and `[selection]` show their defaults (the commented `states` and `energy_window_eV` have no default value: the default is all excited singlets and no window). **`[dynamics]` is commented out on purpose:** the stage runs only if the input has the table (`tmax_fs` has no default), so a template run ends at `selection` as before. `[analysis]` is commented out too: it is optional (every key has a default) and the stage runs after `dynamics` whenever the input has a `[dynamics]` table; the template's commented values are the defaults except `fit_states` (default: every excited state). The template omits `execution.poll_interval_s` (default 0.5), most `[dynamics]` keys and the `[analysis]` keys `enabled` and `warn_excluded_fraction`. The table lists every key:

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
| `dynamics.tmax_fs` | float | — (**required**; the stage runs only if the `[dynamics]` table exists) | > 0, `dt_fs` < `tmax_fs`, an integer multiple of `dt_fs` (relative tolerance 1e-9) | End time of every trajectory (fs). **Not part of the fingerprint:** raising it after the stage is done continues the `finished` trajectories (§7.5, §8.16.6) |
| `dynamics.dt_fs` | float | 0.5 | > 0 | Nuclear time step (fs). Every trajectory stays on this time grid, also after a step rejection (§8.16.3) |
| `dynamics.nsubsteps` | int | 25 | ≥ 1 | Electronic substeps per nuclear (sub)step (§8.11) |
| `dynamics.coupling` | str | `"overlap"` | `"overlap"`, `"nacdr"` | `"overlap"`: local diabatization from wavefunction overlaps, for every method. `"nacdr"`: SA-CASSCF nonadiabatic coupling vectors; **only with `[electronic] method = "casscf"`** (§8.11, §8.15.5) |
| `dynamics.hopping` | str | `"sharc"` | `"sharc"`, `"fssh"`, `"none"` | Hopping probabilities (§8.12). `"fssh"` needs `coupling = "nacdr"` |
| `dynamics.ekin_correction` | str | `"parallel_vel"` | `"parallel_vel"`, `"parallel_nac"` | Velocity adjustment at a hop (§8.12). `"parallel_nac"` needs `coupling = "nacdr"` |
| `dynamics.frustrated` | str | `"keep"` | `"keep"`, `"reflect"` | Frustrated hops (§8.12) |
| `dynamics.decoherence` | str | `"edc"` | `"edc"`, `"none"` | Energy-based decoherence (§8.12) |
| `dynamics.decoherence_param_Eh` | float | 0.1 | > 0 | EDC parameter $C$ (Eh) |
| `dynamics.edc_convention` | str | `"sharc"` | `"sharc"`, `"granucci_persico"` | EDC convention (§8.12) |
| `dynamics.substep_interpolation` | str | `"midpoint"` | `"midpoint"`, `"endpoint"` | Interpolation inside a nuclear step (§8.11) |
| `dynamics.energy_drift_limit_eV` | float | 0.5 | > 0 | A trajectory ends with the outcome `energy_drift` when $\lvert E_{\text{tot}} - E_{\text{tot}}(0)\rvert$ exceeds this (eV) after an accepted step (§8.16.4) |
| `dynamics.energy_jump_limit_eV` | float | 0.1 | finite, > 0 | **[IMPLEMENTED, P9a plan 2]** A (sub)step whose $\lvert E_{\text{tot}} - E_{\text{tot}}(\text{previous record})\rvert$ exceeds this (eV) is rejected and halved like the other rejection signals; a jump that persists at the smallest step ends the trajectory with the outcome `electronic_discontinuity` (§8.16.3). It catches a CASSCF solution change that the overlap and rung signals miss. **To switch it off, set a large value** (e.g. `1.0e6`; 0 is not allowed). Part of the fingerprint |
| `dynamics.stop_after_s0_fs` | float | none (off) | > 0 | End the trajectory with `stopped_in_s0` when the active state has been S0 for this long without interruption (fs) |
| `dynamics.overlap_threshold` | float | 0.5 | 0 < x < 1 | A step whose smallest singular value of the raw state-overlap matrix is below this is rejected (§8.16.3) |
| `dynamics.max_step_halvings` | int | 3 | 0–6 | A rejected step is redone with $\Delta t/2$, ..., down to $\Delta t/2^{k}$ with $k$ = this value; 0 switches the halving off |
| `dynamics.output_every` | int | 1 | ≥ 1 | Keep every $n$-th full step in `output.h5`; every hop, flag and rejection and the last record are always kept (§8.16.5). Part of the fingerprint |
| `dynamics.max_failed_fraction` | float | 0.1 | $0 \le x < 1$ | The stage fails if more than this fraction of the trajectory *tasks* failed (crash, bad input files; **not** an outcome such as `energy_drift`). A check on the finished results: not part of the fingerprint |
| `analysis.enabled` | bool | `true` | `true`, `false` | `false` skips the `analysis` stage. The stage runs only if `dynamics` runs (a `[dynamics]` table); without one, an `[analysis]` table is accepted and has no effect. Not part of the fingerprint |
| `analysis.fit_states` | list of str | every excited state | excited singlet labels `"S1"`, `"S2"`, ... (case is ignored; `S0`, triplets, duplicates and an empty list are rejected; each must exist with `[electronic] singlets`) | States whose summed population is fitted (§8.13.4); the default fits $1 - P(S_0)$ |
| `analysis.fit_population` | str | `"classical"` | `"classical"`, `"quantum"` | Fit the classical (active-state) populations $P_J$ or the quantum populations $\bar\rho_J$ (§8.13.3) |
| `analysis.n_bootstrap` | int | 1000 | ≥ 100 | Number $B$ of bootstrap resamples for the quantum-population intervals and the fit intervals (§8.13.3) |
| `analysis.energy_jump_eV` | float | 0.1 | finite, > 0 | A trajectory whose largest jump of $E_{\text{tot}}$ between consecutive kept records exceeds this (eV) is **flagged** (§8.13.6). The default is from a measurement (§8.13.6) |
| `analysis.exclude_energy_jumps` | bool | `false` | `true`, `false` | `true`: flagged trajectories are excluded from the averages (reason `energy_jump`); `false`: they are only reported |
| `analysis.warn_excluded_fraction` | float | 0.2 | $0 \le x \le 1$ | Warn (possible selection bias) when more than this fraction of the selected trajectories is excluded (§8.13.1). A check on the finished results: not part of the fingerprint |

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
- **Stage names:** an override for a stage that does not exist is an input error (`[resources.stages] names unknown stage(s): …`). The stages in this version are `prepare`, `optimize`, `frequencies`, `sampling`, `excitations`, `selection`, `dynamics` and `analysis`.

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
- **CASSCF start rule (no input key) [IMPLEMENTED].** For `method = "casscf"` every fresh solve starts from a reference solution computed at the stage's reference geometry (§8.6.2). There is nothing to set; the rule is part of the fingerprint of every CASSCF stage, so a `casscf` project computed before it needs `--rerun-from optimize` (or `frequencies`) once (§7.5).
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

#### The `[dynamics]` section
**[IMPLEMENTED]** (P7b). The keys are in the table above. A small, complete example (water, CIS/STO-3G, the run of §5.8):

```toml
[electronic]
method = "tda"
basis = "sto-3g"
functional = "hf"
singlets = 3

[dynamics]
tmax_fs = 1.0
dt_fs = 0.25
```

- **The stage runs only if the input has a `[dynamics]` table** (`tmax_fs` has no default). Without the table the workflow ends at `selection`, as before P7b, and the log says `Stage dynamics: skipped (disabled in the input file)`. A user who forgets the table therefore gets no dynamics and no error.
- **`[electronic]` is required, with `singlets >= 2`** (S0 and at least one excited state): `Stage 'dynamics' needs [electronic] singlets >= 2 ...`, exit 2. The stage runs at the level of the `[electronic]` section **with all its settings**, one PySCF backend per trajectory. `[electronic] triplets` must stay 0. **Compute one or two states above the highest initial state:** a state outside the computed set that crosses the tracked one cannot be followed, and the trajectory ends with `overlap_failure` (§8.16.4; example in §5.8, where `singlets = 3` and a trajectory that starts in S2 meets an uncomputed S3).
- **Rules checked when the input is loaded** (`validate`, `run`; exit 2; the violations are listed together, separated by `; `, **only within one rule group**: the rules of the `[dynamics]` table itself (the first three below) are checked first, and the rule `coupling = "nacdr"` needs `method = "casscf"` is checked with the other cross-section rules only after every table is valid, so an input that breaks both groups shows the second group on the next `validate`): `dt_fs` < `tmax_fs`; `tmax_fs` an integer multiple of `dt_fs`; `hopping = "fssh"` and `ekin_correction = "parallel_nac"` need `coupling = "nacdr"`; `coupling = "nacdr"` needs `method = "casscf"` (TDA/TDDFT have no analytic couplings). An unknown key is an error, so `eselect_Eh` and the spin–orbit options of `model-dynamics` are **not** accepted (no triplets or SOC from PySCF).
- **Warning (not an error):** for `tda`/`tddft` with a DFT functional (anything but `"hf"`) and `grid_level < 5` the stage logs that PySCF DFT gradients omit the grid response; use `grid_level >= 5` for dynamics (§8.6.4).
- **Fingerprint (§8.3):** `dynamics` depends on the whole `[electronic]` section (including `overlap_norm_kept`, which changes the overlaps), on `[dynamics]` **without `tmax_fs` and `max_failed_fraction`**, on `project.seed` and on the fingerprints of `prepare`, `sampling` and `selection`. Changing `tmax_fs` upward **extends** the finished trajectories on the next `run` (§7.5); changing any other hashed key after results exist is a stale-input stop, and you need `--rerun-from dynamics`. `max_failed_fraction` only acts in the finalization. `energy_jump_limit_eV` is hashed like the other keys.
- **Stage version 2 (P9a plan 2).** The energy-jump signal (§8.16.3) changed the stage's behaviour, so its version was raised from 1 to 2: a project whose `dynamics` stage was computed before is stale, `run` stops with exit 3, and `--rerun-from dynamics` recomputes every trajectory (§7.5). The stages before it are not affected.
- **Random numbers:** trajectory $k$ draws from the stream `task_rng(project.seed, "dynamics", k)` that `trajectories.json` records (§8.9.3); exactly one number per nuclear (sub)step (§8.12).
- **`pause_after = ["selection"]` now pauses** when a `[dynamics]` table exists, because `dynamics` follows (§7.8).

#### The `[analysis]` section
**[IMPLEMENTED]** (P8). The section is optional; without it every key takes its default, and the stage runs after `dynamics`. The keys are in the table above; the methods are in §8.13.

```toml
[analysis]
fit_states = ["S1"]                # fit P(S1) only (default: every excited state, i.e. 1 - P(S0))
fit_population = "classical"
n_bootstrap = 1000
energy_jump_eV = 0.1
exclude_energy_jumps = false
warn_excluded_fraction = 0.2
```

- **The stage runs only if `dynamics` runs**, i.e. if the input has a `[dynamics]` table (§4.2), and `enabled` is not `false`. An `[analysis]` table without `[dynamics]` is **not** an error: the stage is skipped (`Stage analysis: skipped (disabled in the input file)`), like `pause_after` without a later stage.
- **Rules checked when the input is loaded** (`validate`, `run`; exit 2): `fit_states` must hold excited singlet labels (`S0`: *S0 cannot be in analysis.fit_states (fit excited states)*; a triplet: *T1 is a triplet; analysis.fit_states takes singlets*; a duplicate; an empty list: *analysis.fit_states must not be empty (omit it to use all)*), and each label must exist with `[electronic] singlets` (*analysis.fit_states: S3 does not exist with [electronic] singlets = 3 (S0..S2)*, reported together with the other cross-section rules, separated by `; `); `n_bootstrap < 100`; `energy_jump_eV` not finite or $\le 0$; `warn_excluded_fraction` outside $[0, 1]$; `fit_population` other than `"classical"`/`"quantum"`; an unknown key.
- **Fingerprint (§8.3):** `analysis` depends on `[analysis]` **without `enabled` and `warn_excluded_fraction`**, on `project.seed` (the bootstrap resamples) and on the fingerprints of `selection` and `dynamics`. Changing a hashed key after the analysis is done is a stale-input stop: run `--rerun-from analysis`, which recomputes only the analysis (the trajectories stay). `warn_excluded_fraction` acts only in the finalization: change it and run again. `dynamics.tmax_fs` is in no fingerprint, but **a change of `tmax_fs` recomputes the analysis automatically** on a plain `run` (§7.5).
- **Random numbers:** the bootstrap draws from the stream `task_rng(project.seed, "analysis", 0)` (§8.13.7), so the same input and trajectories give the same intervals.
- **`pause_after = ["dynamics"]` pauses before the analysis** (§7.8).

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
**[PLANNED]** Their key names and meanings are in the master plan §4.3 (`[optimize]`, `[frequencies]`, `[sampling]`, `[excitations]`, `[selection]`, `[dynamics]` and `[analysis]` are implemented, see above). Each sub-project adds the sections it needs, and this manual is updated each time. (The small input file of `gqteashmd model-dynamics`, §5.7, is separate from the project input; its `[dynamics]` table has the options of §8.12 in atomic units and more keys, such as `eselect_Eh`, than the project's `[dynamics]` of §4.2.) These keys are not part of the input yet, because they depend on decisions that are still open:
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
| `meta.casscf` | CASSCF only: the rung of the CASSCF ladder that converged (§8.6.8); a single point has no previous orbitals, so normally `"reference orbitals"` | — |
| `meta.casscf_reference` | CASSCF only **[IMPLEMENTED, start rule]**: the reference solution (§8.6.2), `{"guess": [...], "sa_energy_Eh": ..., "n_guesses": ..., "n_solved": ..., "n_counted": ...}` (`n_solved` = guesses actually solved, 1 when the default counts), or `"unavailable"` if none was found | —, $E_h$ |
| `meta.reference_active_min_sv` | CASSCF only, when the reference rung produced the solution: smallest singular value of the active-orbital overlap with the projected reference (§8.6.2) | — |
| `meta.wall_time_s` | Wall time of the calculation | s |

`single-point` uses its own geometry as the reference geometry. Measured under the earlier lowest-SA rule on ethylene SA-CASSCF(2,2)/STO-3G (the `single-point` test of the start rule): `casscf = "reference orbitals"`, `casscf_reference = {"guess": [6, 8], "n_guesses": 4, "n_counted": 4, "sa_energy_Eh": -76.86134594}`, `reference_active_min_sv` 0.99999999991. The guess `[6, 8]` is the STO-3G case of §8.6.2 ("Why the default guess comes first"): it is not PySCF's default active space `[7, 8]`, which the rule now takes (so the current output has guess `[7, 8]` and `n_solved` 1; not re-measured). The example output below was produced before P3b and the start rule, so its `meta` lacks `casscf`, `casscf_reference`, `reference_active_min_sv` and `overlaps`.

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
**[IMPLEMENTED]** (P6). Real output from a development machine: the run of §3 (water, HF/STO-3G, that is CIS, `singlets = 3`: S0, S1, S2), with the **small settings** `[sampling] n_samples = 20`, `[selection] n_trajectories = 5` and `allow_fewer = true`, otherwise the defaults (`project.seed = 12345`, broadening 0.1 eV, grid step 0.01 eV, bright mode with `weighting = "f_over_dE2"`, no window). The excitation energies are far too high (12-17 eV) because CIS/STO-3G is a poor description of water; the example shows the workflow and the files, not chemistry. **This output was produced before the TDA/TDDFT root fix of P7b (§8.6.3)**: with the current program the S2 numbers can differ, and a project computed before P7b stops with exit 3 at `excitations` and needs `gqteashmd run INPUT --rerun-from excitations` (§7.5). All six stages took 64 s with `max_parallel_tasks = 1`, 45 s of them in `excitations` (21 tasks, one per sample plus the reference); `selection` took about 2 s, mostly Python start-up. The console output is in §3.

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
**[IMPLEMENTED]** (P7a-1: Tully models; P7a-2: LVC models, §5.7.1). It runs an ensemble of surface-hopping trajectories on one of Tully's one-dimensional two-state models (§8.14) and prints the branching probabilities, or on a linear vibronic coupling (LVC) model with singlets, triplets and spin–orbit coupling and writes populations against time (§5.7.1). It is the way to try the dynamics core (§8.10–§8.12, §8.15) and to validate your settings against a known answer. `[model] name` selects the kind: `tully_sac`, `tully_dac` and `tully_ecr` (the tables of this section) or `lvc` (§5.7.1). It needs no project directory, no molecule and no electronic-structure program (NumPy only), does not run the preflight checks (§7.9) and writes no files unless you ask for JSON. The input file is a **separate small TOML file** (not the project input of §4); its `[dynamics]` table holds the dynamics options of §8.12 (in atomic units) and is **not** the `[dynamics]` section of a project (§4.2, which has its own keys in fs and eV and runs the `dynamics` stage on molecules, §8.16).

```text
gqteashmd model-dynamics INPUT [--json FILE]
```

| Option | Meaning |
|---|---|
| `--json FILE` | Also write the results to a JSON file (fields below). Refuses an existing directory or an unwritable path before the run |

Exit codes: **0** on success; **2** for an input problem (unknown key, invalid value, unknown model parameter; the message names the key, e.g. `dynamics.edc_convention`); **1** when a trajectory fails although the input is valid (`model-dynamics run failed inside a trajectory: …`, e.g. two states degenerate at the start because `parameters` removed the coupling), when every trajectory of an LVC ensemble ended with `energy_drift` (§5.7.1), and also for an internal error (`internal error: …`); **5** when interrupted with Ctrl-C. Every table is strict: unknown keys are rejected. Everything is in atomic units (§8.10).

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
| | `coupling` | `"nacdr"` | `"nacdr"` (NAC vectors) or `"overlap"` (local diabatization from wavefunction overlaps, §8.15.5; also runs on the Tully models; not with `hopping = "fssh"`, exit 2) |
| | `gradient_transformation` | `"ngt"` | `"ngt"` or `"none"` (§8.15.4); only matters with spin–orbit coupling |
| | `eselect_Eh` | none | Energy window for gradients and couplings in $E_h$ (> 0), or unset for everything (§8.15.6) |
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

### 5.7.1 LVC models with spin–orbit coupling
**[IMPLEMENTED]** (P7a-2; real molecules with PySCF run through the `dynamics` stage, §8.16, for singlets only: **[PLANNED]** triplets and spin–orbit matrix elements from PySCF, O7). With `[model] name = "lvc"` the command reads a **linear vibronic coupling model** (§8.15.2) in SHARC's file formats, `LVC.template` and the `V0.txt` it names, and runs an ensemble from Wigner-sampled initial conditions of the model's ground-state vibrations. The states are singlets and triplets; each triplet is expanded into its three $M_s$ components and spin–orbit coupling (SOC) mixes them, so the dynamics runs in the diagonal representation (§8.15). The model is a **separate model file set, not a molecule**: nothing is computed with PySCF. The output is the populations against time, not branching probabilities.

Keys (every table strict; a Tully key in an LVC input, or an LVC key in a Tully input, is rejected with the key named):

| Table | Key | Default | Meaning |
|---|---|---|---|
| `[model]` | `name` | required | `"lvc"` |
| | `template` | required | Path of the `LVC.template` file, relative to the input file (`~` expanded). The `V0.txt` it names on its first line is resolved against the template's directory |
| | `spin_orbit` | `true` | `false` sets $\boldsymbol\Sigma = 0$ (SHARC's `nospinorbit`) |
| `[initial]` | `state` | required | `"S0"`, `"S1"`, ... or `"T1"`, ... in the labels of the model's state counts. The state is an **MCH** state (§8.15.3); the active diagonal state is the one with the largest weight |
| | `ms` | none | $M_s$ of a triplet initial state, −1, 0 or +1 (default 0); an error for a singlet |
| | `sampling` | `"wigner"` | `"wigner"` (ground-state Wigner function of the model's vibrations at T = 0, §8.15.2) or `"fixed"` (every trajectory starts at the reference geometry with zero velocity) |
| `[dynamics]` | all keys of §5.7 | as there | `dt_au` is 1.0 unless set; `hopping = "fssh"` is refused with spin–orbit coupling (§8.15.3), `ekin_correction = "parallel_nac"` stops with an error at a hop whose diagonal coupling vector is complex, which it is with spin–orbit coupling (§8.15.3), `coupling = "overlap"`, `gradient_transformation` and `eselect_Eh` as in §5.7 |
| `[ensemble]` | `n_trajectories` | 100 | Number of trajectories |
| | `seed` | 12345 | Seed (≥ 0); results are reproducible and do not depend on `workers` (§8.14) |
| | `n_steps` | 1000 | Nuclear steps per trajectory; every trajectory takes exactly `n_steps` |
| | `record_every` | 10 | Populations are recorded at time 0 and after every `record_every`-th step |
| | `workers` | 1 | Worker processes |

There is no `x0`, `k`, `sigma`, `x_exit` or `max_steps` for LVC models. A trajectory that ends with `energy_drift` is **excluded from every average** and counted (§8.15.7). A missing or unreadable template or `V0.txt` is exit 2 and names the file; a template keyword or section that the reader does not support is exit 2 with the file name and line number (§8.15.2); so is a near-zero frequency in `V0.txt` (`Cannot use the LVC template …: LVC reference data: near-zero frequency for mode k (… Eh); …`, §8.15.2). Exit 1: a failure inside a trajectory (for example `hopping = "fssh"` with spin–orbit coupling, or `parallel_nac` at a hop, with the message from §8.15.3), and **every trajectory ending with `energy_drift`** (`gqteashmd: error: all N trajectories ended with energy_drift (reduce dynamics.dt_au or raise dynamics.energy_drift_limit_au)`; no table and no JSON file are written). A Tully input in the same situation still exits 0: its JSON counts the drift and stays valid (probabilities 0.0 over zero finished trajectories).

The console output lists, for the final recorded time, the spin-free MCH populations (classical: $\sum_{M_s}|U_{\alpha\lambda}|^2$ averaged over trajectories; quantum: $\sum_{M_s}|(U\mathbf c)_\alpha|^2$, §8.15.7), then the number of hops, frustrated hops and excluded energy-drift trajectories, and the wall time. With `--json FILE` it also writes `kind: "lvc_populations"`, `schema_version` (1), `model`, `initial`, `dynamics`, `ensemble` (as run; the template path is the resolved absolute path), `times_au`, `labels_expanded`, `labels_spin_free`, `populations` (six arrays of shape (times, states): `diag_classical`, `diag_quantum`, `mch_classical`, `mch_quantum`, `mch_spin_free_classical`, `mch_spin_free_quantum`), `counts` (`n_trajectories`, `n_energy_drift`, `n_included`, `n_hops`, `n_frustrated`) and `elapsed_s`. The JSON file is **not byte-reproducible** because `elapsed_s` is a wall time; every other field is. If all trajectories ended with energy drift there is nothing to average: the run exits 1 and writes no JSON (above), so the JSON file never contains `NaN`. (The library function `run_population_ensemble` itself returns NaN averages in that case.)

**Example.** The model fixture `lvc_soc` of the repository (`tests/data/lvc/`: a 3-atom model with 9 coordinates of which three are vibrational modes (the other six have zero frequency), two singlets and two triplets, so $N_{\text{exp}} = 2 + 3\cdot 2 = 8$ expanded states; the parameters are invented for tests, not a real molecule), 100 trajectories started in S1, 100 steps of 20 a.u. (2000 a.u. = 48.4 fs). The input (`lvc_s1.toml`, with `lvc_soc.template` and `V0.txt` copied next to it):

```toml
[model]
name = "lvc"
template = "lvc_soc.template"

[initial]
state = "S1"
sampling = "wigner"

[dynamics]
dt_au = 20.0

[ensemble]
n_trajectories = 100
seed = 20261110
n_steps = 100
record_every = 10
workers = 8
```

and the real output (WSL, this workstation, 2026-11-10; the command took 3.7 s wall including start-up, of which 2.7 s is the ensemble itself):

```text
$ gqteashmd model-dynamics lvc_s1.toml --json out.json
LVC dynamics: S1, 100 trajectories, 100 steps of 20.0 a.u.
final spin-free MCH populations at t = 2000.0 a.u.:
state      classical     quantum
S0            0.0000      0.0001
S1            0.9059      0.8926
T1            0.0941      0.1073
T2            0.0000      0.0000
hops: 69, frustrated hops: 2, energy_drift (excluded): 0, elapsed: 2.7 s
Results written to out.json
```

The S1 → T1 transfer (about 0.09–0.11) comes from the spin–orbit coupling (with `spin_orbit = false` the singlet and triplet blocks do not couple); the classical (hopping) and quantum (amplitude) populations differ by about 0.013 for this ensemble size. A time step of 20 a.u. is much larger than the 0.5 a.u. of the validation runs (§8.15.9); for numbers you rely on, check the result at a smaller `dt_au`. The default `n_steps = 1000` with `dt_au = 1.0` is a short run (24 fs).

### 5.8 Example: `dynamics` on water
**[IMPLEMENTED]** (P7b). Real output from a development machine (code on `/mnt/c`, one core per task, `max_parallel_tasks = 1`): water, HF/STO-3G (CIS, `singlets = 3`: S0, S1, S2), the **small settings** `[sampling] n_samples = 4`, `[excitations] samples_per_task = 2`, `[selection] n_trajectories = 2` with `allow_fewer = true`, and a dynamics of `tmax_fs = 1.0`, `dt_fs = 0.25`, everything else at its default (`project.seed = 12345`, `coupling = "overlap"`, `hopping = "sharc"`, `decoherence = "edc"`). It is the input of the end-to-end test (§8.16.8 B). CIS/STO-3G is a poor description of water (the excitation energies are 11-17 eV, S1 is dissociative); the example shows the workflow and the files, not chemistry. The whole run of the seven stages took **36 s** (the `dynamics` stage 5 s of it), starting from `input.toml` in an empty directory next to `water.xyz` (O at (0, 0, 0.1173), H at (0, ±0.7572, −0.4692) Å, the geometry of the test suite):

```toml
[project]
name = "water-dynamics"
directory = "run"

[molecule]
geometry = "water.xyz"

[resources]
total_cores = 4
total_memory_mb = 4000

[electronic]
method = "tda"
basis = "sto-3g"
functional = "hf"
singlets = 3

[sampling]
n_samples = 4

[excitations]
samples_per_task = 2

[selection]
n_trajectories = 2
allow_fewer = true

[dynamics]
tmax_fs = 1.0
dt_fs = 0.25
```

The end of the console output of the first run was

```text
... INFO    gqteashmd.orchestrator: Stage selection: done (1/1 tasks)
... INFO    gqteashmd.orchestrator: Stage dynamics: starting
... INFO    gqteashmd.orchestrator: Stage dynamics: done (2/2 tasks)
Workflow complete: 7 stage(s) run, 0 already up to date, 0 skipped.
```

**Extension.** Then `tmax_fs` was changed to `1.5` in the same file and the same command run again (5 s; the other six stages are up to date because `tmax_fs` is not part of any fingerprint, §8.3):

```text
... INFO    gqteashmd.orchestrator: Stage selection: up to date
... INFO    gqteashmd.orchestrator: Stage dynamics: reopening 1 task(s): traj_00000
... INFO    gqteashmd.orchestrator: Stage dynamics: starting
... INFO    gqteashmd.orchestrator: Stage dynamics: done (2/2 tasks)
Workflow complete: 1 stage(s) run, 6 already up to date, 0 skipped.
```

Only `traj_00000` (outcome `finished`) was reopened and continued from its restart at 1.0 fs to 1.5 fs; `traj_00001` had ended with `overlap_failure` and is final. `06_dynamics/dynamics_summary.md` after the extension (the first run's summary, at 1.0 fs, was overwritten):

```text
# Dynamics summary

- trajectories: 2
- tmax: 1.5 fs, dt: 0.25 fs
- outcomes: finished 1, energy_drift 0, stopped_in_s0 0, overlap_failure 1, electronic_failure 0
- totals: n_hops 0, n_frustrated 0, n_rejections 4, n_flagged 0
- largest energy drift (eV): 0.0275

| task | sample | state | outcome | t_final (fs) | hops | rejections | flagged | max drift (eV) | message |
|---|---|---|---|---|---|---|---|---|---|
| traj_00000 | 2 | S2 | finished | 1.500 | 0 | 0 | 0 | 0.0275 |  |
| traj_00001 | 3 | S2 | overlap_failure | 0.094 | 0 | 4 | 0 | 7.979e-06 | t = 0.0938 fs, step dt/8: smallest singular value of the state overlap matrix 7.34e-13 is below overlap_threshold = 0.5 |
```

How to read it. Both trajectories start in S2 (the bright-state selection of §8.9.3 picked S2 for samples 2 and 3). `traj_00000` runs the 6 steps of 0.25 fs without a rejection, a hop or a flag; the drift of 0.0275 eV ($1.0\times10^{-3}\,E_h$) is the integration error of a strongly dissociative surface at $\Delta t = 0.25$ fs (the $\Delta t^2$ scaling of this system is in §8.16.8 B, test (d)). Its `output.h5` holds 7 records (`time_fs`, $E_h$):

```text
time_fs  active  E(S0), E(S1), E(S2) (Eh)               E_tot (Eh)    smallest singular value
0.000    S2      -74.941063 -74.467370 -74.323336       -74.30195952   (no previous geometry)
0.250    S2      -74.931923 -74.452232 -74.308226       -74.30170072   0.9994
0.500    S2      -74.926162 -74.453856 -74.308856       -74.30167561   0.9996
0.750    S2      -74.924972 -74.472554 -74.325619       -74.30191137   0.9992
1.000    S2      -74.926837 -74.504375 -74.354839       -74.30230765   0.9983
1.250    S2      -74.928409 -74.542630 -74.390228       -74.30270190   0.9972
1.500    S2      -74.926214 -74.580365 -74.425281       -74.30297016   0.9959
```

`traj_00001` shows the failure outcome that §8.16.4 explains: after 0.094 fs S2 crosses an S3 state that is not among the computed states (`singlets = 3`), the smallest singular value of the state overlap falls to $7\times10^{-13}$, halving the step to $\Delta t/8$ does not help (4 rejections), and the trajectory ends with `overlap_failure`; the stage itself succeeds (exit 0) because the task is done. The remedy is to compute more states (`singlets = 4`, §8.16.4); that was not run here. The same input with `singlets = 3` and `tmax_fs = 2.0` and a kill of the controller in the middle is the test of §8.16.8 B, test (f).

**Since P8** the same input also runs the `analysis` stage after `dynamics` (eight stages; the console output above was captured before P8 and does not show it). It is the end-to-end test of §8.13.8: `traj_00000` is the one included trajectory, `traj_00001` is excluded (`overlap_failure`), so 50 % are excluded and the stage logs the selection-bias warning (more than `warn_excluded_fraction = 0.2`) and still succeeds (exit 0); the horizon is 1.0 fs, and after raising `tmax_fs` to 1.5 the analysis is recomputed in the same run with the horizon 1.5 fs (§7.5). A population average over one trajectory means nothing; §5.9 shows the files on a larger study.

### 5.9 Study: ethylene SA2-CASSCF(2,2)/6-31G\*, 20 trajectories (milestone M4)
**[IMPLEMENTED]** (P8). A real study through every stage, run unattended from the XYZ file, with one deliberate kill of the controller during the dynamics. **Report-only:** no number of this section is asserted by a test. Machine: 16 cores, 7754 MB RAM (WSL Oracle Linux 9.5), project on the WSL native filesystem (`~/p8_m4`), code at commit `2631eff` (before the CASSCF start rule of §8.6.2, so the same input now needs `--rerun-from optimize` on that project; the study was not re-run). The input, fixed before the run (the geometry is `tests/data/ethylene.xyz`):

```toml
[project]
name = "ethylene_m4"
directory = "run"
seed = 2026

[molecule]
geometry = "ethylene.xyz"
charge = 0
multiplicity = 1

[resources]
total_cores = 16
total_memory_mb = 7000
max_parallel_tasks = 8
threads_per_task = 2
memory_mb_per_task = 800

[electronic]
method = "casscf"
basis = "6-31g*"
singlets = 2

[electronic.casscf]
ncas = 2
nelecas = 2

[sampling]
n_samples = 100

[selection]
n_trajectories = 20

[dynamics]
tmax_fs = 100.0
dt_fs = 0.5
```

Everything else is at its default: `optimize` enabled, finite-difference frequencies (37 gradient tasks), $T = 0$ K sampling, bright-state selection (all 20 trajectories start in S1), `coupling = "overlap"`, `hopping = "sharc"`, EDC, `energy_drift_limit_eV = 0.5`, `overlap_threshold = 0.5`, and the `[analysis]` defaults (fit of P(S1), classical, 1000 resamples, jump threshold 0.1 eV).

**Wall time (run A, from `logs/gqteashmd.log`):** `prepare` 5 s, `optimize` 12 s, `frequencies` 25 s, `sampling` 2 s, `excitations` 43 s (101 tasks, 99 of 100 samples ok; S1 mean $\Delta E$ = 9.362 ± 0.628 eV, mean $f$ = 0.3542 ± 0.2178), `selection` 2 s, `dynamics` about 11 min, `analysis` 18 s; **13 min 10 s in total**, including the 13 s between the kill and the restart.

**The kill and the resume.** In run A the controller's process group was killed (SIGKILL) 11 min 04 s into the dynamics, which was too late: 19 of 20 trajectories had already ended, and the orphan worker of the last one (`traj_00018`) finished by itself, so the second run only reconciled its `result.json` (§7.3). The study was therefore **repeated once with the identical input** in a fresh project (run B, `~/p8_m4b`), with the kill 2 min 54 s after the dynamics started: 5 trajectories had ended, 8 were running at 0-27 fs with a restart generation, 7 not started. The second run terminated the **7 orphan workers** that were still alive (`Terminating orphaned worker ...`), requeued them, and they **resumed from their restart generations** (generations 0-54; their final generations are beyond those; the code of this run had no INFO log line for a resume, P9a plan 2 added one, §8.16.6); it exited 0 (`Workflow complete: 2 stage(s) run, 6 already up to date, 0 skipped.`) after 13 min 01 s in total. **Every trajectory's outcome, `t_final_fs`, hops and rejections in run B equal those of run A**, and the analysis agrees to round-off (the PySCF resume is not bit-identical, §8.16.6): `rho_S1(100 fs)` $6.270982\times10^{-4}$ (B) against $6.270995\times10^{-4}$ (A), the largest drift 0.92626253543 against 0.92626253544 eV, the consistency 0.378485199 against 0.378485201. The numbers below are those of run A.

**Outcomes (`06_dynamics/dynamics_summary.json`):** `finished` 2, `overlap_failure` 8, `energy_drift` 7, `electronic_failure` 3, `stopped_in_s0` 0; 13 hops, 0 frustrated, 52 rejections, 3 flagged steps (`n_flagged`, §8.16.3); largest drift 0.926 eV (`traj_00009`, `energy_drift` at 10.0 fs).

- The 8 `overlap_failure` trajectories each end **after their first S1 → S0 hop**, at 17.7-85.4 fs, with smallest singular values of 0.08-0.44 at $\Delta t/8$. **Corrected by the P9a diagnosis (§8.16.9):** they end 5.8-42.9 fs after the hop, not "right after" it, and no state outside the computed set was found. In 3 of the 8 the cause was reproduced as the same SA2-CASSCF(2,2) solution switch as for the energy drifts; for the other 5 the evidence is indirect. The earlier reading, "the two computed states are not enough to follow the wavefunction", is not supported.
- The 7 `energy_drift` trajectories end at 6.5-11 fs with drifts of 0.61-0.93 eV, **with no hop**, and 6 of them with no rejection (one has 4): a jump of $E_{\text{tot}}$ that passes every rejection signal, consistent with the CASSCF solution change of §8.16.3.
- The 3 `electronic_failure` trajectories (at 0.25, 17.0 and 48.75 fs) are SA-CASSCF non-convergence on every rung.

**Analysis (`07_analysis/analysis.json`, `report.md`):** 20 selected, **2 included** (`traj_00003` and `traj_00010`, the two `finished` ones), 18 excluded, so **90 % are excluded** and the stage warns: *90% of the selected trajectories are excluded (more than warn_excluded_fraction = 0.2): possible selection bias, see 'excluded'*. Horizon 100 fs, grid step 0.5 fs, no trajectory cut. Populations from `populations.csv` (with 2 trajectories $P$ can only be 0, 0.5 or 1):

| $t$ (fs) | $P_{S_0}$ | $P_{S_1}$ | $\bar\rho_{S_0}$ | $\bar\rho_{S_1}$ |
|---|---|---|---|---|
| 0 | 0 | 1 | 0 | 1 |
| 25 | 0 | 1 | $3.42\times10^{-4}$ | 0.99966 |
| 50 | 0.5 | 0.5 | 0.4663 | 0.5337 |
| 75 | 0.5 | 0.5 | 0.5002 | 0.4998 |
| 100 | 1 | 0 | 0.99937 | $6.27\times10^{-4}$ |

The 95 % Wilson intervals of $P$ at 50 and 75 fs are [0.0945, 0.9055]; $P_{S_0}(100)$ lies in [0.342, 1.0]. **Fit** of $P_{S_1}$ (classical, 1000 resamples, 0 bootstrap failures): $A = 1$, $t_0$ = 39.80 fs, interval [39.80, 88.00], standard deviation 20.46 fs; $\tau$ = 32.61 fs, interval [0.036, 32.61], standard deviation 16.28 fs; rms 0.1511; **no flag**. Hops of the included trajectories: S1 → S0 3, S0 → S1 1; first hop and first S0 arrival at 55.25 fs (mean and median, 2 arrived, 0 censored). Energy: drift of the included trajectories median 0.300, 95th percentile 0.342, maximum 0.347 eV; **13 trajectories are flagged for a jump above 0.1 eV** (0.122-0.932 eV), among them both included ones (0.326 and 0.317 eV; they stay in the averages because `exclude_energy_jumps = false`). Consistency $\max_t\lvert P_J - \bar\rho_J\rvert$ = 0.3785 for S0 and S1.

**What the study shows, honestly.** The workflow runs unattended from the XYZ file to populations and a fit in 13 min on this machine and survives a kill of the controller in the middle of the dynamics, with the same outcomes after the resume. **The physics of this setup is not usable:** at these defaults SA2-CASSCF(2,2)/6-31G\* ethylene loses 18 of 20 trajectories (overlap failures 5.8-42.9 fs after the S1 → S0 hop, energy drifts of 0.6-0.9 eV within 11 fs that, in 6 of 7 cases, raised no rejection signal, three CASSCF non-convergences), so **the fitted $\tau$ and $t_0$ from 2 trajectories mean nothing**: the intervals are degenerate (the estimate of $\tau$ is on the upper edge of its interval, that of $t_0$ on the lower edge) and, in this run, no fit flag reported it. The analysis did what it is designed to do: it excluded the early-ended trajectories, said so in the warning, and flagged the energy jumps (13 trajectories). **Since the P8 final fix wave** the same analysis would carry two more warnings (the M4 analysis was not re-run, so they are not in the files above): the small-ensemble warning *only 2 trajectories are included (fewer than 20): populations, intervals and the fit are not statistically reliable*, with `small_ensemble: true` in `analysis.json` (§8.13.4), and the included-jump warning *2 included trajectories have an energy jump above energy_jump_eV = 0.1 eV (see energy.flagged); consider exclude_energy_jumps = true* (§8.13.6). With $N = 2$ the bootstrap intervals are still computed; only $N = 1$ sets them to `null`. The setup was **not tuned** in P8 (a smaller `dt_fs`, more states or another active space are P9 work); a large excluded fraction is a finding, not a target. **P9a plan 1 measured these alternatives on the same 20 initial conditions** (§8.16.9). The diagnosis report gives the result as a count of kept trajectories: those that end `finished` or `stopped_in_s0` with no $E_{\text{tot}}$ jump above 0.1 eV. Kept counts: M4 as run 0, $\Delta t$ = 0.25 fs 1, SA3-CASSCF(2,2) 11, and the others 0. The fixes of P9a plan 2 and the acceptance study that followed are in §5.10.

### 5.10 Ethylene SA-CASSCF: findings and guidance (P9a)
**[IMPLEMENTED]** for the program behaviour named here (the energy-jump signal, P9a plan 2; the CASSCF start rule, after the P9a closure); the rest is measured guidance, **report-only**: no number of this section is asserted by a test. The full record, with the file each number comes from, is in `docs/superpowers/reports/2026-11-12-p9a-diagnosis.md` (the diagnosis, P9a plan 1) and `docs/superpowers/reports/2026-11-13-p9a-acceptance.md` (the fixes and the acceptance run, plan 2). Machine and filesystem as in §5.9.

**Why the M4 study of §5.9 lost its trajectories (the root causes).**

- **`energy_drift`** (7 of 20): the SA2-CASSCF(2,2)/6-31G\* problem has several solutions along these paths, and a trajectory jumped from one to another by 0.6-0.9 eV in one step. SHARC 4.1 (`SHARC_PYSCF.py`) makes the same jump at the same step in 6 of 7. No rejection signal of the code of that time saw it. **The energy-jump signal of P9a plan 2 now does** (`energy_jump_limit_eV`, §4.2, §8.16.3): with it, the P7b validation run that jumped by 0.976 eV at 12 fs ends inside that step (at 11.8125 fs, $\Delta t/8$) as `electronic_discontinuity` instead of accepting the jump (§8.16.8 C).
- **`overlap_failure`** (8): 5.8-42.9 fs after the S1 → S0 hop. Replays reproduced the same mechanism in 7 of the 8: after the hop, a second solution appears, crosses the run's solution in SA energy, and the run, now on the higher one, loses it at the failed step. The overlap signal ends the trajectory as designed; no state outside the computed set was found. The eighth (00005) could not be replayed from its hop.
- **`electronic_failure`** (3): CASSCF stops at its iteration limit at a stationary energy. Looser tolerances (`conv_tol` 1e-8, `conv_tol_grad` 1e-4) converge at all three failure geometries, but they move the state energies by up to $7.1\times10^{-5}\,E_h$ and the gradients by up to $2.1\times10^{-3}\,E_h/a_0$ against the nominal tolerances, so they are not used (the two extra CASSCF rungs that plan 2 added to the backend for this test were enabled by no stage and were removed on 2026-11-13, owner decision). In two cases SHARC's interface converges at the nominal tolerances where our solver does not; no defect of ours was identified.

**The recommended setup** (owner decisions after the diagnosis; it is the input of the acceptance run): the M4 input of §5.9 with **`[electronic] singlets = 3`** (SA3-CASSCF(2,2), S2 in the state average) and **`[selection] states = ["S1"]`**, every other key at its default, `energy_jump_limit_eV` 0.1 eV included. SA3 removes the early solution switches and the convergence failures of M4 on the seed-2026 initial conditions and costs about the same per step (2.84 against 3.07 s per accepted (sub)step: the plan 1 seed-2026 variant runs, per-trajectory wall time summed over the trajectories and divided by the accepted (sub)steps, table of section 4 of the diagnosis report; the per-trajectory `s_per_step` values are in `~/p9a/variants/sa3_cas22/summary.json` and `~/p9a/variants/base/summary.json`).

**Measured kept counts.** "Kept" means `finished` or `stopped_in_s0` **and** no $E_{\text{tot}}$ jump above 0.1 eV between records (`energy.flagged` of `analysis.json`):

| Run | Initial conditions | Code | Kept of 20 | Losses |
|---|---|---|---|---|
| M4 (SA2), as run | seed 2026 | P8 | 0 (2 finished, both with jumps) | 8 `overlap_failure`, 7 `energy_drift`, 3 `electronic_failure` |
| SA3, plan 1 | the same, fixed | before plan 2 | 11 | 5 `overlap_failure` and 2 `energy_drift` after a hop, 2 finished with a jump at the hop |
| SA3, plan 2 (final code) | the same, fixed | `8c1e6f8` | 12 | 8 `overlap_failure`, all after a hop |
| **SA3, acceptance** | **seed 2027, regenerated** | `8c1e6f8` | **13** | 5 `overlap_failure`, 2 `electronic_discontinuity`, all after a hop |

The seed-2026 runs use the M4 initial conditions (SA2-based sampling) with changed electronic settings; the acceptance run is a full `gqteashmd run` from the XYZ file with seed 2027, so its optimization, frequencies, samples and excitations are SA3 ones. **The acceptance target, 16 of 20, was missed.** It was not lowered, and nothing was re-run with other settings. All four runs predate the CASSCF start rule (below).

**Owner decisions after the acceptance run (2026-11-13).** (1) **P9a is closed with the miss documented**; the losses after a hop are not a work item now. (2) **A start-solution rule was added**, the CASSCF start rule of §8.6.2 (below). (3) The two extra CASSCF rungs of plan 2 were removed. (4) `energy_jump_limit_eV` keeps its default of 0.1 eV.

**The acceptance run** (`~/p9a/acceptance`, 27 min 43 s in total, `dynamics` 26 min 11 s with 8 tasks x 2 threads): `optimize` converged; 12 real frequencies (887.4-3425.6 cm⁻¹) and no warning; 100 of 100 samples OK (S1 9.979 ± 0.318 eV, $f$ 0.516; S2 15.064 ± 0.759 eV, $f$ $1.3\times10^{-4}$); 20 trajectories selected, all in S1. Outcomes: `finished` 13, `overlap_failure` 5, `electronic_discontinuity` 2; 45 hops, 35 rejections (overlap 28, energy 7); largest drift 0.272 eV. Analysis: 13 included, none flagged for a jump, the small-ensemble warning ($N$ < 20) and the excluded-fraction warning (35 %); fit of $1 - P(S_0)$: $t_0$ = 11.22 fs [3.38, 57.89], $\tau$ = 75.20 fs [25.34, 135.30] (with $N = 13$ these intervals are wide, and `small_ensemble` is `true`).

**Why the target was missed.** Measured: all 7 losses come after a hop (6 on S0 after S1 → S0), each at the smallest step $\Delta t/8$, 9.9-62.9 fs after the last hop, with S1 5.9-7.4 eV above S0 in 5 of the 6 S0 cases (a hot ground state, not a crossing region). *Inference, not measured on seed 2027:* this is the post-hop solution switch that the seed-2026 replays reproduced in SA3 as in SA2. Plan 2 has no fix for it: the energy-jump signal only detects it, and re-solving from the previous orbitals does not avoid it (the continued solve itself lands on the other solution).

**Guidance for SA-CASSCF dynamics of ethylene with this program:**

- Use SA3-CASSCF(2,2)/6-31G\* rather than SA2 (12-13 of 20 kept against 0), keep `energy_jump_limit_eV` at its default (the owner kept 0.1 eV after the acceptance run), and expect losses after the S1 → S0 hop in the hot ground state at `dt_fs` 0.5 (7-8 of 20 in the two measurements above).
- Read the excluded fraction before the populations (§8.13.1): the losses after the hop remove decaying trajectories, so the populations and the fit can be biased.
- `dt_fs` 0.25 did not help with SA2 (1 of 20); SA3-CAS(2,3) and SA2-CAS(4,3) kept none (plan 1).
- **The starting solution is now fixed by a rule [IMPLEMENTED].** SA3-CAS(2,2) has several solutions at the initial geometries (4 at every one of the 20 seed-2026 starts from the 4 guesses of §8.6.2). Before the rule our runs happened to start on the one that is lowest in SA energy and most like the equilibrium solution (20 of 20); SHARC 4.1 starts on another solution, none of the four, in 13 of 20. The CASSCF start rule (§8.6.2) makes the first choice deterministic. **Measured** on the same 20 starts (SA3-CAS(2,2), the M4 optimized geometry as the reference geometry; report-only, nothing tuned): the rule's step-0 solution is that lowest-SA solution in **20 of 20** starts (energies within $3.5\times10^{-8}\,E_h$), every start used the rung `"reference orbitals"`, the reference was the default HOMO/LUMO active space (all 4 guesses converged when measured under the lowest-SA rule; the default won, so the result is the same under the default-first rule), and `reference_active_min_sv` was 0.9498-0.9973 (no warning). So the rule keeps the start our runs already had and does not look for SHARC's solution. The trajectories were not re-run with the rule. *Inference, not measured:* the start being the same, the rule itself should not change these kept counts; but a full rerun recomputes `optimize`, `frequencies` and `excitations` too, whose results move at the round-off level (in the cost run of §8.6.2 the excitation energies by up to $3.3\times10^{-6}\,E_h$), and over 100 fs such differences can change individual paths.

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

and, with a `[dynamics]` table, after `selection`:

```text
└── 06_dynamics/                # [IMPLEMENTED, P7b]
    ├── dynamics_summary.json   # outcomes and per-trajectory numbers, see below
    ├── dynamics_summary.md     # the same, to read
    └── tasks/traj_00000/       # one directory per trajectory; the number is the `index` in trajectories.json
        ├── output.h5           # the trajectory, thinned by output_every (HDF5, with units), see below
        ├── steps.log           # binary log of every accepted (sub)step (not for reading; output.h5 is built from it)
        ├── progress.json       # grid step, time (fs), active state and outcome, rewritten after every accepted step
        ├── restart/            # gen_<g>/ (state.npz, electronic.npz, meta.json) and `current` (names the newest complete generation)
        └── task.json, result.json, worker.log, electronic.log (PySCF output), scratch/
```

and, after `dynamics` (unless `[analysis] enabled = false`):

```text
└── 07_analysis/                # [IMPLEMENTED, P8]
    ├── analysis.json           # every number of the analysis, see below
    ├── report.md               # the same, to read
    ├── populations.h5          # populations with intervals, fit target and curve (HDF5, with units)
    ├── populations.csv         # the same as a table
    ├── hops.csv                # one row per hop or frustrated hop
    └── tasks/analysis/         # task.json, result.json, worker.log and the task's own files (copied to the stage directory)
```

A disabled `optimize` stage still occupies number 01, so `frequencies` stays in `02_frequencies/`, `sampling` in `03_sampling/`, `excitations` in `04_excitations/`, `selection` in `05_selection/`, `dynamics` in `06_dynamics/` and `analysis` in `07_analysis/`. In `fd_<k>_plus` and `fd_<k>_minus`, $k$ counts the Cartesian coordinates ($x,y,z$ of atom 1, then atom 2, ...) from 0. PySCF's temporary files of a task are written to its own `scratch/` directory inside the project, never to the system temporary directory.

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

**`dynamics_summary.json`** [IMPLEMENTED, P7b]: `schema_version` (1), `n_results` (trajectories with a result), `outcomes` (a count for each of `finished`, `energy_drift`, `stopped_in_s0`, `overlap_failure`, `electronic_failure` and, since P9a plan 2, `electronic_discontinuity`), `tmax_fs`, `dt_fs`, `totals` (`n_hops`, `n_frustrated`, `n_rejections`, `n_flagged`, summed over the trajectories, and since P9a plan 2 `rejections_by_signal`, the summed counts per signal), `max_drift_eV` (largest $\lvert E_{\text{tot}} - E_{\text{tot}}(0)\rvert$ of any trajectory, `null` if none is finite), `missing_output` (trajectories whose `output.h5` is missing: the stage only lists them and logs a warning, it does not build them) and `trajectories`, one row per trajectory: `task`, `sample`, `initial_state`, `outcome`, `t_final_fs`, `n_hops`, `n_rejections`, `n_flagged`, `rejections_by_signal`, `n_resumes`, `max_drift_eV`, `message` (the reason of a failure outcome, else `null`). `dynamics_summary.md` shows the same as a table, with a line `rejections by signal`. Both are rewritten whenever the stage finalizes (also after an extension). A trajectory that was stopped by `tmax_fs` is `finished`; the other five outcomes are final (§8.16.4). `schema_version` stays 1: the P9a plan 2 keys were added, none was changed.

- **`rejections_by_signal`** [IMPLEMENTED, P9a plan 2]: the rejected attempts of the trajectory counted by the signal that rejected them (§8.16.3): `electronic`, `overlap`, `energy`, `rung`; only signals that occurred are listed (`{}` for none). The sum equals `n_rejections`. Which attempt had which signal is not recorded, only the counts.
- **`n_resumes`** [IMPLEMENTED, P9a plan 2]: how often the trajectory continued from a restart generation (the task's `result.json` lists the generation names in `resumes`). A resume after a kill or a retry counts, and so does the continuation of a `finished` trajectory after `tmax_fs` was raised, or any new invocation of an ended task. For every trajectory with a resume the stage logs one INFO line, `Stage dynamics: traj_… resumed from restart generation(s) gen_…, …`, in `logs/gqteashmd.log` and on the console. (The trajectory runner also logs `resumed from restart generation gen_… at t = … fs` on the logger `gqteashmd.trajectory.runner`; a worker process configures no log handler, so that line is not shown in a stage run.) **Limit:** a resume is recorded in the next restart generation; if the task is killed again before it writes one, that resume is lost, and the next resume records only its own generation.

**`output.h5`** [IMPLEMENTED, P7b] (one per trajectory, `06_dynamics/tasks/traj_<k>/output.h5`; HDF5; file attributes `schema_version` = 1, `state_labels` (`S0`, `S1`, ...), `output_every`, `dt_fs`, `outcome`, `message` (only if there is one), `symbols`, `sample` (1-based index in `samples.h5`), `initial_state`, `task`). It is **built when the trajectory ends**, from `steps.log`, with every outcome; it does **not** exist while the trajectory runs (§8.16.5). Every dataset has a string attribute `units`. $M$ is the number of records kept, $n$ the number of states and $N$ the number of atoms:

| Dataset | Shape | Units | Content |
|---|---|---|---|
| `time_fs` | $M$ | fs | time of the record |
| `step`, `grid_step`, `ticks`, `depth` | $M$ | — | counter of accepted (sub)steps; completed full steps; position inside the current full step in units of $\Delta t/2^{k}$ ($k$ = `max_step_halvings`); the halving depth of the step (§8.16.3) |
| `active` | $M$ | state index (0 = S0) | active state after the step |
| `hop_from`, `hop_to` | $M$ | state index (−1 = no hop) | the hop of this step |
| `frustrated`, `rung_changed` | $M$ | bool | a frustrated hop; the CASSCF solution left its first ladder rung (flag, §8.16.3) |
| `rejections` | $M$ | — | rejected attempts before this record (§8.16.3) |
| `random_number` | $M$ | — | the uniform number of this step |
| `min_singular_value` | $M$ | — | smallest singular value of the raw state-overlap matrix |
| `energies` | $M\times n$ | Eh | adiabatic energies |
| `e_total`, `e_kinetic` | $M$ | Eh | $E_{\text{tot}}$ (active-state energy plus kinetic) and the kinetic energy |
| `populations` | $M\times n$ | — | $\lvert c_j\rvert^2$ |
| `coefficients_real`, `coefficients_imag` | $M\times n$ | — | electronic amplitudes |
| `probabilities` | $M\times n$ | — | hopping probabilities of the step |
| `positions` | $M\times N\times 3$ | bohr | |
| `velocities` | $M\times N\times 3$ | bohr per atomic unit of time | |

Which records are kept: with `output_every = n`, the records at full steps (`ticks` = 0) whose `grid_step` is a multiple of $n$, **every record with a hop, a frustrated hop, a rung flag or a preceding rejection**, and the last record. With `output_every = 1` every accepted (sub)step is in the file; the records of a halved step sit between the full steps.

**`steps.log`, `progress.json`, `restart/`** [IMPLEMENTED, P7b]: `steps.log` is a binary file of fixed-size records (one per accepted (sub)step; do not edit); `progress.json` has `grid_step`, `time_fs`, `active` (label) and `outcome` (`null` while the trajectory runs) and is rewritten after every accepted step, so you can read it while the run goes on (`gqteashmd status` does not show it). `restart/gen_<g>/` holds the core restart (`state.npz`), the electronic snapshot (`electronic.npz`) and `meta.json` (counters and the position inside a halved interval); `restart/current` names the newest complete generation (§8.16.6).

**`analysis.json`** [IMPLEMENTED, P8] (`07_analysis/`; the methods are in §8.13). It is written also when the stage stops (§8.13.7): then `stage_failed` is `true`, `error` holds the message and the fields that were not computed are `null`. It never holds `NaN` (`null` instead). Times in fs, energies in eV, populations as fractions:

| Field | Meaning |
|---|---|
| `schema_version` | Format version (1) |
| `n_selected`, `n_included` | Trajectories in `trajectories.json`, and those that enter the averages ($N$, §8.13.1) |
| `small_ensemble` | `true` when $N$ < 20 (`SMALL_ENSEMBLE`, a program constant, not an input key), also when the stage stops; then the populations, intervals and the fit are not statistically reliable (§8.13.4) |
| `outcomes` | Counts of the six outcomes of §8.16.4 (five before P9a plan 2) over the trajectories with an `output.h5`, plus `task_failed` (no result in `dynamics_summary.json` or no `output.h5`) |
| `included` | The task names of the included trajectories |
| `excluded` | `[{"task", "reason", "t_final_fs", "energy_jump_eV"}, ...]`: `reason` is the outcome, `task_failed` (then `t_final_fs` is `null`) or `energy_jump` (with `exclude_energy_jumps = true`; then `energy_jump_eV` holds the jump, otherwise it is `null`). A trajectory that failed at step 0 has `t_final_fs` 0 |
| `cut` | Included trajectories with records after the horizon (after `tmax_fs` was lowered, §8.13.2) |
| `horizon_fs`, `dt_out_fs` | The end $T$ of the common grid and its spacing `output_every` × `dt_fs` |
| `fit` | `states`, `population` (`classical`/`quantum`), `A`, `t0_fs`, `tau_fs`, `t0_interval_fs`, `tau_interval_fs` (`[lo, hi]`, 95 %), `t0_std_fs`, `tau_std_fs`, `rms` (root-mean-square residual), `flags` (§8.13.4), `bootstrap_failures` |
| `hops` | Over the included trajectories up to the horizon: `pairs` (`{"S1->S0": n, ...}`), `frustrated_from` (frustrated hops per **originating** state, `{"S1": n}`), `per_trajectory` (`min`, `median`, `max` hops and `zero`, the number of trajectories without a hop), `first_hop_fs` (`n`, `mean`, `median`), `first_s0_arrival_fs` (`n_arrived`, `n_censored`, `mean`, `median`); and `totals_all` (`hops`, `frustrated` over every trajectory with an `output.h5`) (§8.13.5) |
| `energy` | `drift_eV` (`median`, `p95`, `max` over the included trajectories), `jump_threshold_eV`, `flagged` (`[{"task", "energy_jump_eV"}, ...]`, every trajectory whose jump exceeds the threshold, included or not) (§8.13.6) |
| `consistency` | Per state, $\max_t\lvert P_J - \bar\rho_J\rvert$ (§8.13.3) |
| `table` | The rows of the population table of `report.md`: `[{"time_fs", "P": {...}, "rho": {...}}, ...]` |
| `settings` | The `[analysis]` values used and `project_seed` |
| `warnings` | Every warning of the stage (a small ensemble, fit flags, bootstrap failures, included trajectories with an energy jump, the excluded fraction) |
| `stage_failed`, `error` | Whether the stage stopped, and why (§8.13.7) |

**`populations.h5`** [IMPLEMENTED, P8] (HDF5; file attributes `schema_version` = 1, `gqteashmd_version`, `n_included`, `n_selected`, `horizon_fs`, `dt_out_fs`, `fit_population`, `fit_states`). Every dataset has a string attribute `units`. $M$ is the number of grid points and $n$ the number of states. Not written when the stage stops:

| Dataset | Shape | Units | Content |
|---|---|---|---|
| `state_labels` | $n$ | — | `S0`, `S1`, ... |
| `time_fs` | $M$ | fs | grid times $t_m$ |
| `classical`, `classical_lower`, `classical_upper` | $M\times n$ | — | $P_J(t_m)$ and its 95 % Wilson interval |
| `quantum`, `quantum_lower`, `quantum_upper` | $M\times n$ | — | $\bar\rho_J(t_m)$ and its 95 % bootstrap interval |
| `fit_target` | $M$ | — | the fitted population $y(t_m)$ (§8.13.4) |
| `fit_curve` | $M$ | — | the fitted model $f(t_m)$; `NaN` everywhere if there is no fit |

**`populations.csv`** [IMPLEMENTED, P8]: comment lines that start with `#` (title, units, `n_included`, `horizon_fs`, `fit_population`, `fit_states`), a header line, then one line per grid point in `%.6e` format. Columns: `time_fs`, then for every state `P_<L>`, `P_<L>_lower`, `P_<L>_upper`, then for every state `rho_<L>`, `rho_<L>_lower`, `rho_<L>_upper`, then `fit_target` and `fit_curve` (empty when there is no fit).

**`hops.csv`** [IMPLEMENTED, P8]: one row per hop or frustrated hop of **every** trajectory with an `output.h5` (included or not, before or after the horizon), columns `task`, `time_fs` (`%.6f`), `from`, `to`, `frustrated` (0/1), `included` (0/1), `after_horizon` (0/1). For a frustrated hop `to` is **empty**: `output.h5` does not record the target of a frustrated hop, and `from` is the active state of that record.

**`report.md`** [IMPLEMENTED, P8]: the content of `analysis.json` to read: the settings, the outcome counts, the table of excluded trajectories, the cut trajectories, a population table ($P$ and $\bar\rho$ at every $\lceil M/25\rceil$-th grid point and the last, so at most about 26 rows), the fit, the hop and energy statistics (without `totals_all`), the consistency, the warnings, the stop message if the stage stopped, and the names of the other files; `small_ensemble` is a line of the summary after `n_included`. Integers (the seed, the counts) are printed exactly (a seed of 12345 appears as `12345`), other numbers with four significant digits (`.4g`), and a missing or non-finite value as `-`; `analysis.json` holds the full-precision values. When the stage stops, `07_analysis/` holds no `populations.h5`, `populations.csv` or `hops.csv`, also if an earlier successful analysis had written them there (they are removed, so the directory never mixes results of two runs).

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
- Each stage is split into **tasks**. `prepare` and `optimize` have one task each. `frequencies` has one task with the analytic Hessian, or $1+6N_{\text{at}}$ tasks with finite differences (§8.7). `sampling` and `selection` have one task each. `excitations` has $1+\lceil N_s/\text{samples\_per\_task}\rceil$ tasks (the reference `ref`, then chunks of `samples_per_task` consecutive samples; $N_s$ = `sampling.n_samples`), which run in parallel up to `max_parallel_tasks`. `dynamics` has one task per selected trajectory (`traj_00000`, ..., the `index` of `trajectories.json`), which run in parallel up to `max_parallel_tasks`; a task is **done for every outcome** of its trajectory (even `energy_drift`), and only crashes and unreadable input files make a task fail (§8.16.4). `analysis` has one task, `analysis`.
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

**Extending the dynamics (`tmax_fs`).** `dynamics.tmax_fs` is not part of the fingerprint (§8.3). If you raise it after `dynamics` is done, the next `gqteashmd run` does not stop: it reopens the trajectories whose outcome is `finished` with an end time below the new `tmax_fs` and continues them from their last restart; other outcomes are final, and their trajectories are not touched. Examples: water at `tmax_fs = 1.0`, then `1.5` (§5.8): the finished trajectory continues from 1.0 to 1.5 fs. Lowering `tmax_fs` reopens no trajectory and keeps the longer results (the summary reports their times); the analysis is then redone at the new horizon (below). **The same applies after a stop or a failure:** if the `dynamics` stage was stopped (`gqteashmd stop`, Ctrl-C) or failed and you raise `tmax_fs` before the next run, that run continues the unfinished trajectories and also reopens the finished ones that end below the new `tmax_fs`, so when the stage is done every `finished` trajectory ends at the new `tmax_fs` (one run, no second run needed). Rules: (a) the reopening also requeues tolerated *failed* tasks of the stage, with their attempts reset; (b) if `dynamics` is in `workflow.pause_after`, the reopened stage pauses again when it finishes (§7.8); (c) any other `[dynamics]` key, `[electronic]` and `project.seed` are hashed: change one and the run stops with exit 3, and `--rerun-from dynamics` recomputes every trajectory from its initial condition (the old results go to `superseded/`); (d) the stage after `dynamics`, `analysis`, follows the change automatically (next paragraph).

**The analysis after a change of `tmax_fs` [IMPLEMENTED, P8].** `tmax_fs` is in no fingerprint, so the analysis is kept up to date by two rules instead of a stale-input stop; both happen on a plain `gqteashmd run`, and neither needs `--rerun-from`:
- **Raising `tmax_fs`:** when `dynamics` reopens trajectories, the program **first** recomputes every later stage that depends on it and has results, with the warning `Stage analysis is recomputed because dynamics was extended`: its directory is moved to `.gqteashmd/superseded/` (not deleted) and its record is reset; then the trajectories continue, and the analysis runs again at the end of the same run with the new horizon. A kill at any point is safe: the next run makes the same decision again. This also holds for a disabled `analysis` that still has results (its results are moved aside, and it stays disabled).
- **Lowering (or raising) `tmax_fs` without reopening a trajectory** (for example when no trajectory is `finished`, or after lowering it): the `analysis` stage notices that its recorded end of the grid, `round(tmax_fs / dt_fs)`, differs from the current one and redoes its own task (`Stage analysis: reopening 1 task(s): analysis`). After lowering, the trajectories that ran longer are listed as `cut` (§8.13.2).

A changed `[analysis]` key (other than `enabled` and `warn_excluded_fraction`) is a stale-input stop (exit 3) that names `--rerun-from analysis`; it recomputes only the analysis. `warn_excluded_fraction` acts in the finalization only: change it and run again.

**A project computed before P7b (TDA/TDDFT root fix).** Since P7b the TDA/TDDFT calculation solves for three more roots than requested and keeps the lowest (§8.6.3), because PySCF's Davidson solver could skip a root at a symmetric geometry and return the next one as the requested state. This can change the excitation energies of `excitations`, so the stage version was raised: a project whose `excitations` stage is already done stops with exit 3 (*Stage 'excitations' has results produced from different inputs ...*). **Run `gqteashmd run INPUT --rerun-from excitations`.** It recomputes `excitations` and every stage after it, and `sampling`, `frequencies` and the geometry stay. `casscf` projects are affected by the stage version too (the stale stop is the same), although their numbers do not change.

**A `casscf` project computed before the CASSCF start rule.** The start rule (§8.6.2) changes how a fresh SA-CASSCF solve starts, so its marker is part of the fingerprint of every CASSCF stage (`optimize`, `frequencies`, `excitations`, `dynamics`; §8.3). A `method = "casscf"` project with results from before it stops with exit 3 at its first CASSCF stage (*Stage 'optimize' has results produced from different inputs ...*, or `'frequencies'` when `optimize` is disabled). **Run `gqteashmd run INPUT --rerun-from optimize`** (or `--rerun-from frequencies` without `optimize`): it recomputes that stage and every later one (`prepare` stays). The numbers can change slightly: in the measured cost run of §8.6.2 (ethylene SA2-CASSCF(2,2)/6-31G\*, 10 samples) the optimized geometry moved by $7.0\times10^{-6}$ bohr and the excitation energies by at most $3.3\times10^{-6}\,E_h$, and the same trajectories were selected; a sample that landed on another CASSCF solution before can now land on the reference one, which is the purpose of the rule. **`tda` and `tddft` projects are not affected** (their fingerprints are unchanged).

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
- The **last enabled** stage never pauses, because nothing follows it: a stage pauses only if a later stage is enabled. `dynamics` is enabled only with a `[dynamics]` table (§4.2), so without it the last enabled stage is `selection`; with it, and with `analysis` enabled (the default), the last one is `analysis`. In this version the stages are `prepare`, `optimize`, `frequencies`, `sampling`, `excitations`, `selection`, `dynamics` and `analysis`, so `pause_after = ["optimize"]` lets you inspect `optimized.xyz` before the frequencies are computed, `pause_after = ["frequencies"]` lets you inspect the frequencies and the Molden file before any samples are drawn, `pause_after = ["sampling"]` lets you look at `samples.xyz` before the expensive excitations, and `pause_after = ["excitations"]` lets you look at `spectrum.csv` and `excitations.json` before the initial states are chosen. `pause_after = ["selection"]` **pauses before the dynamics if the input has a `[dynamics]` table** (exit 5; the rerun continues with `dynamics`), so you can read `selection_report.md` before the expensive trajectories, and has no effect without that table. `pause_after = ["dynamics"]` **pauses before the analysis** (exit 5), so you can read `dynamics_summary.md` and the outcomes before the populations are computed; with `[analysis] enabled = false` it has no effect. A stage that is reopened by an extension (§7.5) and listed in `pause_after` pauses again after it finishes.
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

**Required modules per stage:** `optimize` needs `pyscf>=2.6` and `geometric>=1.0`; `frequencies` needs `pyscf>=2.6` and `h5py>=3.9`; `dynamics` needs `pyscf>=2.6` and `h5py>=3.9` (checked only if the input has a `[dynamics]` table); `analysis` needs `h5py>=3.9` and `scipy>=1.11` (checked only when the stage is enabled). A stage that is disabled is not checked (with `optimize.enabled = false`, geomeTRIC is not needed). If the installed version cannot be read, the version is not checked.

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
- **§8.10–§8.12, §8.14 and §8.15** describe the **dynamics core**, implemented in P7a-1 and P7a-2 and validated on analytic models; since P7b it is connected to molecules by the `dynamics` stage, which is described in **§8.16**.
- **§8.13** describes the **`analysis` stage** (P8): which trajectories enter the averages, the populations and their intervals, the lifetime fit, the hop statistics and the energy diagnostics. It is validated on synthetic ensembles with known answers; a comparison with SHARC 4.1's analysis tools is planned (P9).

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
- For `dynamics`: the whole `[electronic]` section (including `overlap_norm_kept`, which changes the overlaps), the `[dynamics]` section **without `tmax_fs` and `max_failed_fraction`**, `project.seed`, and the fingerprints of `prepare`, `sampling` and `selection`. `tmax_fs` is left out so that raising it extends the finished trajectories instead of being a stale-input stop (§7.5, §8.16.6); `max_failed_fraction` acts only in the finalization. `output_every` is hashed although it does not change the physics, because changing it on a resumed trajectory would mix record spacings in one log. A disabled `dynamics` (no `[dynamics]` table) gets the fingerprint `"skipped"` like any optional stage.
- For `analysis`: the `[analysis]` section **without `enabled` and `warn_excluded_fraction`**, `project.seed` (the bootstrap resamples) and the fingerprints of `selection` and `dynamics`. `warn_excluded_fraction` acts only in the finalization. `dynamics.tmax_fs` is not hashed here either; the analysis follows a change of it by the reopen rules of §7.5 (not by its fingerprint), so after an extension or a lowered `tmax_fs` it is recomputed without a stale-input stop.
- **The CASSCF start-rule marker [IMPLEMENTED, start rule]:** for `method = "casscf"` the `[electronic]` data hashed by `optimize`, `frequencies`, `excitations` and `dynamics` also holds `start_rule = "reference-1"`, the version of the start rule of §8.6.2 (not an input key). It made every CASSCF project computed before the rule stale (§7.5); a future change of the rule would do the same. `tda`/`tddft` fingerprints do not carry it. No stage version was raised for it.
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
**[IMPLEMENTED]** On stop, each running task's process group receives SIGTERM at time $t_0$ and SIGKILL at $t_0 + t_g$, with $t_g$ = `stop_grace_period_s`. A task that finishes within $t_g$ keeps its result. **[IMPLEMENTED, P7b]** Dynamics tasks write a restart generation after every accepted (sub)step, so a stop or a kill loses at most the step in progress (§8.16.6); the worker has no signal handler of its own.

---

> **§8.6 to §8.16 are [IMPLEMENTED]** (§8.13, the analysis, since P8). Each section is marked at its head.

### 8.6 Electronic structure (PySCF)
**[IMPLEMENTED, P3a]** Energies, analytic gradients and transition dipoles. **[IMPLEMENTED, P3b, library only]** SA-CASSCF nonadiabatic couplings, wavefunction overlaps with phase tracking, `compute_more`, restarts (§8.6.8). **[IMPLEMENTED, P7b]** use of couplings (CASSCF) and overlaps (all methods) in the `dynamics` stage (§8.16). **[PLANNED]** TDDFT couplings, triplets and spin–orbit coupling. **[IMPLEMENTED, P4]** the S0 Hessian (analytic for RHF/RKS; SA-CASSCF uses finite differences of gradients in the `frequencies` stage, §8.7). **[IMPLEMENTED, start rule]** the CASSCF start rule: fresh SA-CASSCF solves start from a reference solution (§8.6.2).

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
- **Orbital reuse by projection.** A backend object that computes several geometries one after another (as the dynamics stage will) starts each CASSCF from the previous geometry's converged orbitals. Those orbitals belong to the old nuclear positions, so they are not orthonormal in the new basis; they are first projected onto the new geometry's atomic-orbital basis (PySCF `project_init_guess`) and re-orthonormalized. A calculation with no previous orbitals (the first one of a backend object, every one after `reset()`) starts from the **reference solution** of the start rule below when the caller gave a reference geometry (every stage and `single-point` do), and from the SCF orbitals otherwise. The reuse changes the starting point; with several SA-CASSCF solutions at one geometry the starting point can decide which one is found, which is why the start rule exists.
- **Active space.** The program stops with a message if the active electrons do not fit in the molecule, or if the basis has fewer orbitals than the core plus active orbitals require (§9).

**The CASSCF start rule [IMPLEMENTED, start rule; no input key].** SA-CASSCF can have several solutions (local minima of $E_{\mathrm{SA}}$) at one geometry, and which one a calculation finds depends on its starting orbitals. The P9a survey found 4 distinct SA3-CASSCF(2,2) solutions at every one of 20 ethylene starting geometries (§5.10). The rule makes the start of a **fresh** solve (one without usable previous orbitals) deterministic and continuous with one reference solution:

1. **Reference geometry.** Each caller names one:

   | Caller | Reference geometry |
   |---|---|
   | `optimize` | the prepared geometry (the start of the optimization) |
   | `frequencies` | its working geometry: the optimized geometry, or the prepared one when `optimize` is disabled |
   | `excitations`, `dynamics` | the equilibrium geometry stored in `samples.h5` (`equilibrium_coordinates`, the geometry `frequencies` used) |
   | `single-point` | its own geometry |

   A library caller that gives none gets the behaviour before the rule (fresh solves start from the SCF orbitals).
2. **Guess set.** With $n_o = n_{e,\text{cas}}/2$ active occupied and $n_v = n_{\text{cas}} - n_o$ active virtual orbitals, the occupied window is the $n_o + 1$ highest doubly occupied SCF orbitals and the virtual window the $n_v + 1$ lowest virtual ones (each cut to the orbitals that exist). A guess chooses $n_o$ orbitals from the first window and $n_v$ from the second as the active space. PySCF's default active space (the $n_o$ highest occupied and $n_v$ lowest virtual orbitals) comes first, then the others in lexicographic order of their 0-based orbital indices. CAS(2,2) has 4 guesses, CAS(4,4) 9, CAS(6,6) 16.
3. **Selection, default guess first (owner decision 2026-11-14).** At the reference geometry the default guess (the first in the order) is solved with the input's settings (state average, `conv_tol`, `conv_tol_grad`, the singlet-only solver, no spin penalty). A guess counts if it converges and every root is a singlet. **If the default guess counts, its solution is the reference solution and no other guess is solved.** Only if it does not count are the other guesses solved, and the reference solution is the counted one with the lowest $E_{\mathrm{SA}}$; guesses within $10^{-8}\,E_h$ of the lowest are ties, won by the first in the order. The reason for putting the default first is given below.
4. **Use.** A fresh solve at any geometry starts from the reference orbitals projected onto that geometry (`project_init_guess`, as for the previous orbitals): this is rung 2, `"reference orbitals"`, of the CASSCF ladder (§8.6.8). Inside a trajectory a solve normally starts from the previous orbitals (rung 1); only if that fails does it fall to rung 2, which the `dynamics` stage treats as a rung change (the step is rejected and halved; at the smallest step it is accepted and flagged if the overlap is good, §8.16.3).
5. **When and how often.** The reference is computed lazily, the first time rung 2 is needed, once per backend object (one per `optimize` task, per `frequencies` finite-difference task, per `excitations` task and per trajectory), and kept by that object across `reset()` and `load_state()`; it is not stored in restart files (a resumed trajectory, in a new process, continues from its saved previous orbitals and recomputes the reference only if it needs it; 20 independent searches, one per backend in 8 worker processes, gave reference energies within $2\times10^{-13}\,E_h$ of each other in the measurement below).
6. **Fallback.** If the SCF at the reference geometry fails, or no guess counts, the backend logs *No CASSCF reference solution at the reference geometry (…): fresh solves start from the SCF orbitals.*, records `"unavailable"` and does not try again on that backend; the solve continues with the SCF orbitals (rung 3). A transient error (out of memory, I/O) is retried as usual.
7. **Mismatch diagnostic.** When rung 2 produced the solution, the smallest singular value of $\mathbf C_{\text{ref}}^T\mathbf S\,\mathbf C_{\text{act}}$ (the projected reference active orbitals against the converged active orbitals, AO overlap $\mathbf S$ at the current geometry) is recorded as `reference_active_min_sv`. A value near 1 means the same active-orbital character; below 0.9 the backend logs *The CASSCF solution started from the reference orbitals has a different active-orbital character (smallest singular value x < 0.9).* The solution is **accepted** either way: nothing depends on the value (owner decision).

**Metadata** of every CASSCF result: `casscf` (the rung, §8.6.8); `casscf_reference`: `"none"` (no reference geometry), `"not computed"` (a reference geometry exists but rung 1 has succeeded on every call so far), `"unavailable"` (the fallback) or `{"guess": [0-based orbital indices], "sa_energy_Eh": …, "n_guesses": …, "n_solved": …, "n_counted": …}` (`n_guesses` is the size of the guess set, `n_solved` the guesses actually solved, 1 when the default counts, `n_counted` the solved guesses that counted); and `reference_active_min_sv` when rung 2 produced the solution. **Where you see it:** `single-point --json` writes them in `meta` (§5.3). **In a project run the metadata is not recorded** (the stages do not store the result metadata), **but the two warnings above, like the backend's older small-overlap warning, are written to the task's `worker.log`** (the worker's stderr; `single-point` prints them on stderr). Only warnings and above are routed; `electronic.log` holds PySCF's own output only. Inside `optimize`, where geomeTRIC installs log handlers during the optimization, the routing was not checked.

**Measured** (report only, nothing tuned; ethylene SA3-CASSCF(2,2)/6-31G\*, the 20 seed-2026 starting geometries of §5.10 with the M4 optimized geometry as the reference):

| Quantity | Value |
|---|---|
| Starts on the lowest-SA solution of the P9a survey | **20 of 20** (energies within $6.5\times10^{-10}$–$3.5\times10^{-8}\,E_h$ of it) |
| Rung used at step 0 | `"reference orbitals"` at all 20 |
| Reference | guess `[7, 8]` (the default HOMO/LUMO active space), 4 of 4 guesses counted when measured under the lowest-SA rule (the default won, so the result is the same under the default-first rule, which solves and counts 1), $E_{\mathrm{SA}} = -77.73693580389885\,E_h$ |
| `reference_active_min_sv` | 0.9498–0.9973 (no warning) |
| One reference search (SCF + 4 guesses, under the lowest-SA rule; with the default counting only 1 guess is solved now, not re-measured) | 0.516 s |
| First calculation at a start (search, SCF, projected solve; 8 parallel tasks × 2 threads) | 0.893–1.653 s |

**Cost in a project** (ethylene SA2-CASSCF(2,2)/6-31G\*, the M4 input of §5.9 with 10 samples and no `[dynamics]`, 8 tasks × 2 threads; before and after the rule, each ±1 s): `optimize` 9 s and 9 s; `frequencies` (37 tasks, each with its own reference search) 26 s and 29 s; `excitations` (11 tasks of one sample) 4 s and 6 s; the whole run 42 s and 48 s. The cost grows with the number of guesses (16 for CAS(6,6), 25 for CAS(8,8)), which was measured for CAS(2,2) only. A larger `samples_per_task` shares one search between more samples.

**Why the default guess comes first (owner decision 2026-11-14).** The first version of the rule took the lowest state-averaged energy among all guesses, which can leave the default HOMO/LUMO space. Measured in the final review (ethylene, `tests/data/ethylene.xyz`, CAS(2,2), report only):

| Setting | Chosen guess | $E_{\mathrm{SA}}$ ($E_h$) | S0 ($E_h$) |
|---|---|---|---|
| STO-3G SA2, default | `[7, 8]` | -76.855342 | -77.115327 |
| STO-3G SA2, **the choice of the earlier lowest-SA rule** | **`[6, 8]`** | **-76.861346** | **-77.068750** |
| STO-3G SA3 | `[7, 8]` | -76.705619 | -77.115117 |
| 6-31G\* SA2 | `[7, 8]` (runner-up `[7, 9]`, -77.850121) | -77.860766 | -78.049024 |
| 6-31G\* SA3 | `[7, 8]` | -77.734636 | -78.047782 |

For SA2-CAS(2,2)/STO-3G the earlier rule took the (HOMO-1, LUMO) space: $E_{\mathrm{SA}}$ is 6.0 mE$_h$ lower but **S0 is 46.6 mE$_h$ (1.27 eV) higher** than with the default $\pi,\pi^*$ space (the $\sigma,\pi^*$ solution of §8.16.8). For 6-31G\* (SA2 and SA3) and STO-3G SA3 the default `[7, 8]` won by at least 10 mE$_h$. **Under the rule now in force STO-3G SA2 takes `[7, 8]`.**

The full suite at the earlier rule (2026-11-13) also failed one test, `test_casscf_frequencies_by_finite_differences` (ethylene SA2-CAS(2,2)/STO-3G, `optimize` off): the stage's finite-difference consistency check refused, 1.86e-1 Eh/bohr against the tolerance 1e-4 (worst C1 y), with a gradient of 6.65e-2 Eh/bohr and an imaginary mode of -747.8 cm$^{-1}$. A probe (fresh backend per geometry) showed the cause: with the reference `[6, 8]` (S0 -77.068750 $E_h$), the displaced solves at C1 y $+h$ and $-h$, both from rung "reference orbitals", landed on two different solutions (-77.068758 and -77.070599 $E_h$), so the finite-difference slope was 0.1841 against a gradient of -0.0023 Eh/bohr. Without a reference (the behaviour before the rule) the two agree to 7e-7 on the first 5 coordinates; with the reference limited to the default guess `[7, 8]` all 37 solves start from "reference orbitals" and the maximum |FD - gradient| over the 18 coordinates is 4.67e-6 Eh/bohr (P4 measured 4.6e-6). The owner therefore decided on 2026-11-14 that the default guess comes first. The chosen guess is still in the result metadata only (`single-point --json`); it is not logged and the stages do not record it, and there is no input key to change it.

**Found while fixing:** ethylene SA2-CAS(2,2)/STO-3G loses the default solution at small distortions **whatever the start**. Under the distortion $0.0x\cdot\mathrm{linspace}(-1, 1)$ of the 18 coordinates (bohr), at 0.005 bohr the reference start keeps it (S0 -77.115168 $E_h$, active singular value 0.9994); at 0.01 bohr both the reference start and the SCF start (the behaviour before the rule) land on the other solution (S0 -77.070432 $E_h$, singular value 0.039, and the warning of item 7 fires); at 0.02 bohr likewise (S0 -77.069939, singular value 0.078). At 6-31G\* the default solution persists to 0.02 bohr (singular value 0.9996). This is the STO-3G metastability already documented for the dynamics (§8.16.8: the $\pi,\pi^*$ solution is lost at about 1.06 fs); the finite-difference frequencies pass because each displacement (0.005 bohr along one coordinate) stays on it. Two unit tests moved from STO-3G to 6-31G\* for that reason, and a new test asserts the warning at the STO-3G 0.02 bohr distortion (same S0 as the SCF start to 1e-6).

**What the rule does not do.** It does not look for the solution with the globally lowest $E_{\mathrm{SA}}$ (SHARC 4.1 started on a solution outside the guess set in 13 of the 20 starts, §5.10); a projected start can still converge to another solution at a strongly distorted geometry (warned, accepted); the default-first choice (the default guess, else the lowest $E_{\mathrm{SA}}$ among the others) is a heuristic that was measured on ethylene only. TDA and TDDFT have no orbital optimization and no rule.

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
**[IMPLEMENTED, P3b]** as a Python library (`gqteashmd.electronic.pyscf_backend.PySCFBackend`, built by `gqteashmd.electronic.factory.make_backend`; `gqteashmd.overlaps.determinants` and `gqteashmd.overlaps.tracking`; the packages re-export nothing). **Use:** `single-point` does not request couplings or overlaps; the `dynamics` stage (P7b, §8.16) requests the phased overlaps at every geometry (and the NACs for `coupling = "nacdr"`) through `BackendModel`, which Löwdin-orthonormalises the overlaps before the core sees them. **[PLANNED]** triplet states and spin–orbit coupling, and TDA/TDDFT couplings.

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
| | 2 | **[IMPLEMENTED, start rule]** The reference solution of §8.6.2, projected onto this geometry, nominal settings (skipped without a reference geometry, or when no reference was found) | `"reference orbitals"` |
| | 3 | SCF orbitals | `"SCF orbitals"` |
| | 4 | SCF orbitals plus a spin penalty (`fix_spin_`, $\langle S^2\rangle=0$, shift 0.5) and 3× the macro iterations | `"SCF orbitals + spin penalty"` |
| `tda`, `tddft` | 1 | $N_{\text{states}}+3$ roots solved (the lowest $N_{\text{states}}$ kept), PySCF defaults otherwise | `td = "default"` |
| | 2 | $N_{\text{states}}+6$ roots solved (the lowest kept) and 3× `max_cycle` | `"extra roots"` |

**Why extra roots on the first rung (P7b, a fix of a P3a/P6 defect).** Asked for exactly $N_{\text{states}}$ roots, PySCF's Davidson solver can **skip a root without any error** at a symmetric geometry (its initial guess can lack a vector of one irreducible representation) and return the next higher root as the requested state. Measured on water CIS/STO-3G at the $C_{2v}$ geometry of the test suite: all roots are 13.1877, **15.1445**, 16.7707, ... eV, and a request for two states returned 13.1877 and 16.7707 eV, so "S2" was the third root (a request for three or more states returned 15.1445 eV as the second root). The number of roots solved is capped at the number of single excitations. The fix makes a skipped root much less likely, **but it is a heuristic, not a guarantee**: nothing detects a skip that survives three extra roots. The excitation energies of existing projects can change, which is why the stage version of `excitations` was raised (a project computed earlier needs `--rerun-from excitations`, §7.5). After the fix the roots agreed with a six-root reference calculation at every point probed along the path of the trajectory that had failed (§8.16.8).

A CASSCF rung fails if it does not converge or if a root has $\lvert\langle S^2\rangle\rvert > 10^{-3}$; a PySCF exception stops the ladder. On ethylene the spin-penalty rung reproduces the plain rung (energies within $10^{-6}\,E_h$, S1 gradient within $10^{-5}\,E_h/a_0$). The number of linear-response roots is limited to the number of single excitations; asking for more is an error. If all rungs fail the run stops (exit code 4, §9).

A fresh solve (no previous orbitals) therefore starts at rung 2 when the caller gave a reference geometry, as every stage and `single-point` do (§8.6.2); `casscf = "SCF orbitals"` then means that no reference was found or that the projected reference did not converge.

**The two extra CASSCF rungs of P9a plan 2 were removed (owner decision 2026-11-13).** Plan 2 had added, behind a backend option that no stage enabled, a rung with more iterations and a rung with loose tolerances (`conv_tol` $10^{-8}$, `conv_tol_grad` $10^{-4}$). A probe on the three ethylene failure geometries of §5.10 found that the loose tolerances change the state energies by up to $7.1\times10^{-5}\,E_h$ and the gradients by up to $2.1\times10^{-3}\,E_h/a_0$ against the nominal ones, so they were never enabled, and they and the probe tool are gone; passing the old option `casscf_extra_rungs` is now an error. The measurement is kept in the acceptance report of P9a.

**`compute_more(gradients, nacs)`.** After a `compute()` at a geometry, further gradients and NACs of that same geometry can be requested without repeating the SCF, CASSCF or TD calculation: the PySCF objects of the last successful `compute()` are reused and the same phases applied. It raises an error before any `compute()`, after `load_state()` (the live objects are not stored) and after a `compute()` that failed (a failed `compute()` discards the previous live objects).

**Restarts.** `save_state(path)` writes the minimal cross-geometry state (geometry, orbitals, density, truncated and phase-aligned determinant expansion, and a fingerprint of the method, basis, functional, atoms, charge, multiplicity, states, CASSCF active space (`ncas`, `nelecas`) and `overlap_norm_kept`) to an `.npz` file, atomically; `load_state(path)` reads it into a new backend object, also in a new process. A file written with different settings is refused (the error names the file). A restarted run reproduces the energies, gradients, NACs, dipoles and overlaps of the uninterrupted run to $2\times10^{-11}$ or better. Restarting from an older snapshot is safe for the signs: the snapshot carries the aligned sign frame, so the recomputed steps are aligned to it like the original ones. Continuity is lost only when a state that the next step needs was never saved (a new backend object then draws a new arbitrary sign frame), so save the state at every step you may restart from. A state file that is missing, damaged, of another format version, or has inconsistent array shapes is refused with *… is not a readable gqteashmd electronic state*. The fingerprint holds only the CASSCF active space (`ncas`, `nelecas`), not the CASSCF tolerances, so changing `conv_tol` does not invalidate a saved state.

**One backend per trajectory.** A backend object remembers the previous geometry (orbital guess, projection, state tracking, phases). Call `reset()` (or use a new backend) before computing an unrelated geometry; the next `compute()` is then a first geometry (`meta.overlaps = no previous geometry`). `reset()` keeps the CASSCF reference solution of the start rule (§8.6.2), which belongs to the backend's fixed reference geometry, and `save_state()` does not write it, so the state-file format and its fingerprint are unchanged by the rule.

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
**[IMPLEMENTED]** (P6). The stage `excitations` computes the vertical excitations at the reference geometry and at every Wigner sample (§8.9.1) and builds the absorption spectrum (§8.9.2). The stage `selection` then chooses the initial electronic state of every trajectory (§8.9.3). §8.9.4 gives the comparison with SHARC 4.1. Atomic units are used inside the calculation and the files carry the units of §6, except that excitation energies are in eV (the only conversion is $E_h \to$ eV with the CODATA factor of §8.1). **Since P7b** the TDA/TDDFT root fix of §8.6.3 raised the version of `excitations`: a project whose `excitations` stage was computed before stops with exit 3 and needs `gqteashmd run INPUT --rerun-from excitations` (§7.5), which recomputes `excitations` and the stages after it (also for `casscf`, whose numbers do not change).

#### 8.9.1 Vertical excitations (`excitations`)
At a geometry $\mathbf R$ the electronic-structure backend (§8.6) gives the energies $E_0, E_1, \dots, E_{n-1}$ of the $n$ = `electronic.singlets` singlet states, in order of energy, and the transition dipoles $\boldsymbol\mu_{0I}$. With the formulas of §8.6.5 and §8.6.6 this gives, for I = S1, ..., S($n-1$),

$$
\Delta E_{0I} = E_I - E_0,
\qquad
f_{0I} = \tfrac23\,\Delta E_{0I}\,\lvert\boldsymbol\mu_{0I}\rvert^2 \quad(\text{atomic units}).
$$

**Procedure.**

1. The stage plans one task `ref` (if `include_reference = true`) for the equilibrium geometry of `samples.h5`, and tasks `s00001`, `s00002`, ... of `samples_per_task` consecutive samples each (the task name is the number of its first sample).
2. A task builds **one backend** with the **full `[electronic]` settings** (the same method, basis, functional, active space and number of states as `single-point`; the logs are in its `electronic.log`, the scratch files in its `scratch/` directory), and calls `reset()` **before every sample**. A reset removes everything the backend remembers of the previous geometry (orbitals, density, phases), so a sample's result does not depend on which samples were computed before it in the same task. With `casscf`, every sample starts from the reference solution of the start rule (§8.6.2), computed once per task at the equilibrium geometry of `samples.h5` and kept across the resets. The $k$-th sample uses `positions[k-1]` of `samples.h5`. State $S_n$ is the $n$-th root **at that geometry**; no state is tracked between samples.
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
**[IMPLEMENTED]** (P7a-1, `gqteashmd.dynamics`, analytic models; **[IMPLEMENTED, P7b]** for molecules with PySCF through the `dynamics` stage, §8.16, which calls this core unchanged). All of §8.10–§8.12 and §8.14 use **Hartree atomic units** ($\hbar = 1$): positions in bohr, masses in electron masses $m_e$, time in $\hbar/E_h$ (1 a.u. = 0.024188843265857 fs), velocities in bohr per atomic unit of time, energies in $E_h$, nonadiabatic couplings in 1/bohr. The core works on a flat coordinate vector $\mathbf q$ of length $f$ (the number of degrees of freedom) with masses $\mathbf m$ of length $f$; a molecule of $N_{\text{at}}$ atoms has $f = 3N_{\text{at}}$ (P7b).

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
**[IMPLEMENTED]** (P7a-1; molecules since P7b, §8.16). The electronic wavefunction $\Psi(t) = \sum_j c_j(t)\,\Psi_j$ is expanded in the **propagation basis**, the diagonal basis, which without spin–orbit coupling equals the adiabatic (MCH) basis (with it, see the end of this section). It obeys

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

**With spin–orbit coupling and with overlaps (P7a-2).** The equations above are the form without spin–orbit coupling (the diagonal basis equals the MCH basis, $\mathbf U = \mathbf 1$), and the code runs exactly that form, bit for bit as in P7a-1, whenever the MCH Hamiltonian has no off-diagonal element. With spin–orbit coupling the electronic wavefunction is propagated in the **diagonal basis** (eigenstates of the complex MCH Hamiltonian), with the propagator $\mathbf R^{\text{diag}} = \mathbf U(t+\Delta t)^\dagger\,\mathbf R^{\text{MCH}}\,\mathbf U(t)$, and the `overlap` coupling type (local diabatization) replaces the NAC term. Both are in §8.15.3 and §8.15.5. What is **not** available from PySCF: spin–orbit matrix elements and triplets, so spin–orbit dynamics cannot yet be run on a molecule **[PLANNED]**; local diabatization from PySCF overlaps is used by the `dynamics` stage (§8.16).

### 8.12 Surface hopping, energy adjustment, decoherence
**[IMPLEMENTED]** (P7a-1, analytic models; since P7b also for molecules, through the `[dynamics]` keys of §4.2). Options and their defaults:

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

Three more options came with P7a-2 (§8.15): `coupling` (`"nacdr"`, the default, or `"overlap"`, local diabatization), `gradient_transformation` (`"ngt"` or `"none"`) and `eselect_Eh` (unset or a window in $E_h$). Deferred (not in the first release): A-FSSH, NPI, GFSH, `parallel_diff`, thermostats, adaptive time steps.

**Restrictions with the diagonal representation.** `hopping = "fssh"` (Tully's flux formula) exists only in the MCH basis with NAC couplings: `DynamicsOptions` rejects it together with `coupling = "overlap"`, and the driver raises `ValueError("hopping = 'fssh' needs the MCH basis (no spin-orbit coupling); use 'sharc'")` when the model has spin–orbit coupling (at the start and at every step). `ekin_correction = "parallel_nac"` needs a real coupling vector (§8.15.3).

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

The hop thus happens **after** the velocity update of the step and the rescaling is done at fixed $\mathbf q'$. Where SHARC's order differs, the difference is measured by black-box comparison (§8.14), not read from its source. **With spin–orbit coupling, `coupling = "overlap"` or `eselect_Eh` the same order is used with the diagonal basis and the masked gradients; the full form is in §8.15.3 and reduces to the list above for an MCH Hamiltonian without off-diagonal elements.**

**Outcomes of a trajectory:** `stopped` (a stop rule you gave was met), `finished` (`max_steps` reached without stopping) and `energy_drift`. Physical endings are outcomes, not errors; invalid options are a `ValueError` naming the option.

**Assumptions:**
- Classical nuclei.
- Independent trajectories.
- The surface-hopping ansatz, which approximates the quantum branching of the nuclear wavepacket.

### 8.13 Analysis: populations and lifetimes
**[IMPLEMENTED]** (P8: the packages `gqteashmd.analysis` and `gqteashmd.stages.analysis`; the input in §4.2, the files in §6, the study in §5.9). The stage has one task. It reads `trajectories.json` (the selected set, §8.9.3), `dynamics_summary.json` and every trajectory's `output.h5` (§6), applies the inclusion rule and the common grid, computes the observables and writes the files of `07_analysis/`. It does not change the trajectories. Every rule of this section (which trajectories count, the horizon, the fit model, the intervals) was **fixed in the design before any data was seen** (owner decisions of 2026-11-11), so that it cannot be tuned to a result. Times are in fs, energies in eV; NumPy computes everything, SciPy only the fit.

#### 8.13.1 Which trajectories enter the averages
| Trajectory | Status | Reason recorded in `excluded` |
|---|---|---|
| outcome `finished` | included | — |
| outcome `stopped_in_s0` | included; after its end it is **held in S0** (active state S0, $\lvert c_{S_0}\rvert^2 = 1$, all others 0) | — |
| outcome `energy_drift`, `overlap_failure`, `electronic_failure` or (since P9a plan 2) `electronic_discontinuity` | excluded | the outcome and `t_final_fs` (0 for a failure at step 0, which leaves no record, §8.16.3) |
| no row in `dynamics_summary.json` (a failed `dynamics` task tolerated by `max_failed_fraction`) or no `output.h5` | excluded | `task_failed` |
| `exclude_energy_jumps = true` and flagged by the jump test (§8.13.6) | excluded | `energy_jump`, with the jump in eV |

$N$ is the number of included trajectories; the **excluded fraction** is $(N_{\text{selected}} - N)/N_{\text{selected}}$, and above `warn_excluded_fraction` (default 0.2) the stage warns of a **possible selection bias**. Why the rule: a trajectory that ended early has no data after its end. Holding its last state, or dropping it from the later times only, would mix ensembles of different composition at different times; excluding it entirely keeps one ensemble for every time point, but it is a bias of its own when the failures depend on the dynamics (for example, when trajectories fail after a hop to S0, as in §5.9, exactly the decaying ones are removed). The excluded list and the warning make that visible; read the excluded fraction before you read the populations.

#### 8.13.2 The common time grid and the horizon
With $n_{\text{tot}} = \mathrm{round}(t_{\max}/\Delta t)$ (`tmax_fs`, `dt_fs`) and $h$ = the smaller of $n_{\text{tot}}$ and the last full step of every included **`finished`** trajectory (`stopped_in_s0` trajectories do not shorten it), the **horizon** is $T = h\,\Delta t$ and the grid is $t_m = m\,\Delta t_{\text{out}}$, $\Delta t_{\text{out}}$ = `output_every` × `dt_fs`, for every $m$ with $m \cdot$ `output_every` $\le h$; all comparisons are on the integer step counters (`grid_step`, `ticks`, §8.16.3), never on floating-point times.
- The value at $t_m$ is that of the record with `ticks` = 0 and `grid_step` = $m\cdot$`output_every`; `dynamics` keeps every such record (§6), so the **active state at $t_m$ is exact**. Records of halved steps (`ticks` > 0) are not grid points.
- A `stopped_in_s0` trajectory past its end is in S0 (§8.13.1).
- A grid record missing before a trajectory's end is a file error (§8.13.7).
- Records after $T$ are ignored, and the trajectories that have such records are listed as **`cut`**. This happens after `tmax_fs` was lowered (§7.5); since every `finished` trajectory ends at `tmax_fs`, the horizon is otherwise `tmax_fs` itself.

#### 8.13.3 Populations and their intervals
With $N_J(t_m)$ the number of included trajectories whose active state is $J$ at $t_m$, the **classical population** and its 95 % **Wilson score interval** [18] are

$$
P_J = \frac{N_J}{N},\qquad
\text{centre} = \frac{P_J + z^2/(2N)}{1 + z^2/N},\qquad
\text{half} = \frac{z}{1 + z^2/N}\sqrt{\frac{P_J(1-P_J)}{N} + \frac{z^2}{4N^2}},
$$

$z = 1.959963984540054$, the interval [centre − half, centre + half] clipped to [0, 1]. Unlike the normal approximation $\sqrt{P(1-P)/N}$, it does not collapse to zero width at $P = 0$ or 1 (with $N = 2$ and $P_{S_0}(100\,\text{fs}) = 1$ in §5.9 it is [0.342, 1.0]). At $N = 100$ and $P = 0.5$ the half-width is 0.096: differences below about that are not significant.

The **quantum population** is the ensemble average of the amplitudes, $\bar\rho_J(t_m) = \frac1N\sum_n \lvert c_J^{(n)}(t_m)\rvert^2$ (the `populations` dataset of `output.h5`), with a 95 % **percentile bootstrap interval** [19]: $B$ = `n_bootstrap` resamples of the $N$ trajectories with replacement, the mean of each, and the 2.5th and 97.5th percentiles of those means (NumPy `percentile`, linear interpolation). The resample indices are drawn **once**, as `integers(0, N, size=(B, N))` from the stream `task_rng(project.seed, "analysis", 0)`, and used for both the quantum intervals and the fit (§8.13.4).

**Consistency** (reported, not a check): $\max_m\lvert P_J(t_m) - \bar\rho_J(t_m)\rvert$ per state. Agreement means that the hopping follows the amplitudes; with few trajectories it can be large (0.38 in §5.9, with $N = 2$).

#### 8.13.4 The lifetime fit
The **target** is the summed population of the states in `fit_states` (default every excited state, so $y = 1 - P_{S_0}$): $y(t_m) = \sum_{J\in\text{fit\_states}} P_J(t_m)$ with `fit_population = "classical"` (the fraction of trajectories whose active state is in `fit_states`) or $\sum_J \bar\rho_J(t_m)$ with `"quantum"`. The model is a **delayed mono-exponential**

$$
f(t) = A \ \ (t < t_0),\qquad f(t) = A\,e^{-(t - t_0)/\tau}\ \ (t \ge t_0),
$$

with $A = y(0)$ fixed ($A = 1$ when every trajectory starts in a state of `fit_states`). The fit is an unweighted least-squares fit over every grid point (`scipy.optimize.least_squares`) in the parameters $t_0 \in [0, t_{\text{last}}]$ ($t_{\text{last}}$ the last grid time) and $\ln\tau$ (free, so $\tau > 0$); it is started from $t_0 \in \{0, t_{\text{last}}/4, t_{\text{last}}/2\}$, each with $\tau_{\text{start}} = \max(t_e - t_{0,\text{start}}, \Delta t_{\text{out}})$, where $t_e$ is the first time at which $y < A/e$ (or $t_{\text{last}}$ if it never is), and the converged start with the smallest cost wins. `rms` is the root-mean-square residual. **Uncertainty:** the same fit on the target of every bootstrap resample (§8.13.3); the intervals are the 2.5th and 97.5th percentiles and `t0_std_fs`, `tau_std_fs` the standard deviations (divisor $n-1$) of the successful resample fits. A resample without a fit (a failure, or no decay in that resample) is counted in `bootstrap_failures` and left out; with fewer than two successful resamples the intervals are `null`. With **one** included trajectory ($N = 1$) every resample is that trajectory, so no resample is fitted and `t0_interval_fs`, `tau_interval_fs`, `t0_std_fs` and `tau_std_fs` are `null` (the quantum-population intervals of §8.13.3 are still computed; at $N = 1$ they have zero width, which is degenerate but not wrong). The lifetime $\tau$ is bounded internally to $e^{\pm 700}$ fs, so it never overflows or underflows to 0 during the fit.

**Flags** (never a stop; each one is in `fit.flags` and gives a warning `fit: <flag> (<meaning>)`):

| Flag | Condition | Consequence |
|---|---|---|
| `no_population` | $y(0) \le 0$ (no trajectory starts in `fit_states`) | no fit; `t0_fs`, `tau_fs` `null` |
| `no_decay` | $y$ never drops below $0.99\,A$ | no fit; `t0_fs`, `tau_fs` `null` |
| `fit_failed` | `least_squares` failed from every start | no fit |
| `tau_beyond_horizon` | $\tau > t_{\text{last}}$ | the lifetime is an extrapolation |
| `little_decay` | $y(t_{\text{last}}) > 0.9\,A$ | the lifetime is poorly determined |
| `t0_at_upper_bound` | $t_0 > t_{\text{last}} - \Delta t_{\text{out}}$ | the decay starts at the end of the analysed time; a delay of 0 is a legitimate result and is not flagged |

`bootstrap_failures` > 0 adds the warning `fit: bootstrap_failures (k of B bootstrap resamples gave no fit and are left out of the intervals)`. **Small ensembles:** with fewer than 20 included trajectories (`SMALL_ENSEMBLE` = 20, a program constant, not an input key; ruling P8-5 after the final review) the stage adds the warning `only N trajectories are included (fewer than 20): populations, intervals and the fit are not statistically reliable` and sets `small_ensemble: true` in `analysis.json` (also on a stop, where no warning is added). With a few trajectories the percentile intervals are degenerate (in §5.9, with $N = 2$, the estimate of $\tau$ lies on the upper edge of its interval [0.036, 32.61] fs). The threshold 20 is guidance, not a measured coverage limit; the stage still fits and reports (a minimum-$N$ rule that refuses the fit is an owner decision for P9). The model is the only kinetic model; populations per initial state are not computed.

#### 8.13.5 Hop statistics
Over the included trajectories and the records up to $T$: the number of hops per pair (`"S1->S0"`); the number of **frustrated hops per originating state** (the active state of the record; `output.h5` does not record the target of a frustrated hop, so a per-pair count is not possible); the accepted hops per trajectory (minimum, median, maximum and the number of trajectories without one); the time of the first hop (mean and median over the trajectories that hop); and the **first arrival in S0**, the time of the first record with active state S0 at or before $T$ (a trajectory with no such record, also a `stopped_in_s0` one whose S0 records all lie after $T$ because `tmax_fs` was lowered, is **censored**), as mean and median over the trajectories that arrive, with the number that has not arrived by $T$ (`n_censored`). These first-arrival numbers are **reported, not fitted** (a censored mean is biased low). Beside them, `totals_all` counts the hops and frustrated hops of every trajectory with an `output.h5`, included or not; `hops.csv` lists every event (§6).

#### 8.13.6 Energy diagnostics
- **Drift** of an included trajectory: $\max\lvert E_{\text{tot}} - E_{\text{tot}}(0)\rvert$ over its records up to $T$; the median, the 95th percentile and the maximum over the included trajectories.
- **Jump** of a trajectory: $\max_r\lvert E_{\text{tot}}(r+1) - E_{\text{tot}}(r)\rvert$ over consecutive **kept records** of `output.h5` with `grid_step` $\le n_{\text{tot}}$ (up to the current `tmax_fs`, not the horizon: the jump can decide the inclusion, and the inclusion decides the horizon). Records of halved steps count as records. With `output_every` > 1, two consecutive full-step records are several steps apart, so the jump then includes the dynamics and the integration error of several steps; lower `energy_jump_eV` accordingly or use `output_every = 1`.
- A trajectory is **flagged** when its jump exceeds `energy_jump_eV`. Flagged trajectories stay in the averages unless `exclude_energy_jumps = true` (§8.13.1). The `flagged` list covers every trajectory with an `output.h5`, included or not. When $k > 0$ of the **included** trajectories are flagged, the stage warns `k included trajectories have an energy jump above energy_jump_eV = x eV (see energy.flagged); consider exclude_energy_jumps = true` (with `exclude_energy_jumps = true` no flagged trajectory is included, so the warning cannot appear).
- **The default 0.1 eV** comes from a rule fixed before the measurement: take the largest per-step $\lvert\Delta E_{\text{tot}}\rvert$ over the accepted steps of the P7b PySCF validation runs; if it is at most 0.01 eV the default is 0.1 eV, otherwise ten times the value. Measured over the 8 `output.h5` files of `tests/validation/test_pyscf_dynamics.py` (ethylene SA2-CASSCF(2,2)/6-31G\* and water CIS/STO-3G, `output_every = 1`, 9-21 records each, one file with 6 hops): **$7.71\times10^{-3}$ eV** ($7.706923\times10^{-3}$, water CIS at $\Delta t = 0.25$ fs), so the default is 0.1 eV.
- This flag was P8's answer to the open question of §8.16.3 (a CASSCF solution change that passes every rejection signal). Since P9a plan 2 (F1) the `dynamics` stage also rejects a (sub)step whose $\lvert\Delta E_{\text{tot}}\rvert$ exceeds `energy_jump_limit_eV` (default 0.1 eV); a jump that persists at the smallest step ends the trajectory as `electronic_discontinuity` (§8.16.3). The analysis flag remains as a diagnostic. In §5.9 it flagged 13 of 20 trajectories (0.12-0.93 eV).

#### 8.13.7 Checks, stops and reproducibility
| Situation | Result |
|---|---|
| `trajectories.json` missing or malformed | the task fails, not retried (exit 4): `… is missing or malformed (…); rerun with --rerun-from selection` |
| `dynamics_summary.json` missing or malformed | the same, with `rerun with --rerun-from dynamics` |
| an `output.h5` unreadable, or inconsistent with the others or with `[dynamics]`: `state_labels` differ from the first file's, `output_every` differs, `dt_fs` differs by more than $10^{-9}$ relative, the `task` or `initial_state` attribute is wrong, the populations have the wrong width; for an **included** trajectory also no record at $t = 0$, a non-finite `e_total`, `populations` or `active`, or a missing grid record | the task fails, not retried (exit 4): `<path>: <field> …; rerun with --rerun-from dynamics` |
| $\lvert\sum_J\lvert c_J\rvert^2 - 1\rvert > 10^{-6}$ in an included record | the stage fails (exit 4): `populations of traj_… sum to 1 within … only (tolerance 1e-6): this indicates a program error`, after `analysis.json` and `report.md` are written |
| no trajectory included | the stage fails (exit 4): `No trajectory can be analysed: outcomes …; only 'finished' and 'stopped_in_s0' trajectories enter the averages (USER_MANUAL 8.13)`, plus advice on `exclude_energy_jumps` if trajectories were excluded for a jump; `analysis.json` and `report.md` are written first, with `stage_failed: true` |
| excluded fraction > `warn_excluded_fraction` | a warning (log, `report.md`, `analysis.json`) |
| fewer than 20 included trajectories | a warning and `small_ensemble: true` (§8.13.4) |
| fit flags, bootstrap failures | warnings (§8.13.4) |
| an included trajectory with an energy jump above `energy_jump_eV` | a warning (§8.13.6) |

The stage tolerates no failed task (its one task is the stage). On a stop, `populations.h5`, `populations.csv` and `hops.csv` are not written; stale copies from an earlier attempt are removed, in the task directory and in `07_analysis/` itself.

**Reproducibility.** The analysis is a deterministic function of the `output.h5` files, the input and `project.seed`: the bootstrap resamples come from `task_rng(project.seed, "analysis", 0)` (§8.8, seed hierarchy), so the same trajectories and input give the same files. After a kill and resume through the whole workflow on the analytic test backend, `populations.csv` is **byte-identical** to that of an uninterrupted run (asserted, §8.13.8); with PySCF the trajectories themselves are reproduced only to round-off (§8.16.6), and so are the populations (§5.9).

#### 8.13.8 Validation (measured)
Every number is **asserted** (a bound fixed in the plan before the first run) unless marked **report**. The detailed reports are in the git-ignored ledger `.superpowers/sdd/2026-11-11-p8-analysis-stage/`. No comparison with SHARC 4.1's `populations.py` or `make_fit.py` exists (P9).
- *Wilson interval:* equal to the scalar formula for a hand-computed case.
- *Bootstrap coverage:* the 95 % interval of the mean of $N = 100$ draws from Beta(2, 5) with $B = 500$, repeated $R = 400$ times, covered the true mean $2/7$ in a fraction **0.93** (band [0.90, 0.99]).
- *Fit recovery:* 2000 synthetic first-passage times $t = 10\,\text{fs} + \text{Exp}(40\,\text{fs})$ on a grid of 0-150 fs at 0.5 fs, $B = 200$: $t_0$ = **9.747 fs** (standard deviation 0.372, interval [8.99, 10.45]) and $\tau$ = **38.971 fs** (standard deviation 0.983, interval [36.84, 40.64]) against 10 and 40 fs, rms 0.00344, no flag, no bootstrap failure (bounds $\lvert\tau - 40\rvert \le 4\sigma_\tau$, $\lvert t_0 - 10\rvert \le 4\sigma_{t_0} + 0.5$ fs, $\sigma_\tau < 3$ fs). A noise-free model is recovered; a zero delay is not flagged; `test_flags` triggers `no_decay` (a constant population), `no_population` (zero population), `tau_beyond_horizon` together with `little_decay` (one slow decay, $\tau$ = 5000 fs) and `t0_at_upper_bound` (a decay at the last grid point); **`fit_failed` is not exercised by any test**.
- *Rules:* the inclusion table, the horizon, `cut`, holding in S0, halved-step records off the grid, a missing grid record, a step-0 failure (no records) excluded with `t_final_fs` 0, the jump flag, the hop statistics and the reproducibility of the bootstrap from the seed have unit tests. Final fix wave (tests): a positive S0 arrival and a `stopped_in_s0` trajectory censored at a lowered horizon; integers printed exactly in `report.md` (seed 12345); $N = 1$ (`small_ensemble` true, the warning, `null` fit uncertainties) and $N = 20$ (`small_ensemble` false, no warning); the included-jump warning; $\tau$ at $\ln\tau = -1000$ positive with no floating-point warning; a stop after a success leaves no stale `populations.*`/`hops.csv` in `07_analysis/`.
- *Stage and pipeline* (canned LVC upstream, the analytic fake backend, the real `dynamics` and `analysis` stages): the stage's classical populations equal a direct recomputation from the `output.h5` files exactly and the quantum populations to $10^{-15}$; raising `tmax_fs` recomputes the analysis in the same run (the old one moved to `superseded/`); lowering it re-analyses at the new horizon with every trajectory `cut` and without running the dynamics again, and a second run changes nothing; `pause_after = ["dynamics"]` pauses before the analysis; a changed `[analysis]` key is stale and `--rerun-from analysis` recomputes it; `warn_excluded_fraction` is not stale; a disabled analysis and an `[analysis]` table without `[dynamics]` are skipped; the orchestrator rule (§7.5) resets the dependants with results, also a disabled one, and leaves the others (fake stages). Slow: a kill of the controller during the dynamics and a resume through the analysis give a byte-identical `populations.csv`.
- *PySCF end to end* (`tests/test_e2e_dynamics.py`, marker `pyscf`; the water run of §5.8): the analysis runs at the end, `traj_00000` is included, the horizon is 1.0 fs and 1.5 fs after raising `tmax_fs`; after a kill and resume the analysis does not stop. These files and the pipeline tests ran 11 passed in 369.9 s (water end to end 53.8 s, water kill 50.1 s, canned kill 32.3 s).
- *The M4 study* (§5.9) is a **report**.

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

**Restart (library).** `dynamics.restart.save_state(path, state, rng)` writes one atomic `.npz` and `load_state(path, model)` reads it; continuing gives a result **bit-identical** to an uninterrupted run (tested: 50 steps + save + load + 50 steps equals 100 steps, for both hopping schemes with EDC and with hops in the window). It stores $\mathbf q$, $\mathbf v$, $\mathbf c$, the electronic snapshot (energies, gradients, NACs, Hamiltonian, model eigenvectors, and since P7a-2 the overlaps with the previous geometry), time, step, active state, $E_{\text{tot}}(0)$, **$\mathbf U$ (since P7a-2, format version 2, §8.15.8)** and the random-source state (a JSON `meta` entry; floats round-trip exactly). It does **not** store the masses, the model parameters or the options; you must supply the same model and options. `load_state` never evaluates the model. A missing, foreign or corrupt file, or a shape mismatch with the model, is a `ValueError`. The random-source state is treated as untrusted input: only the NumPy bit generators PCG64, PCG64DXSM, MT19937, Philox and SFC64 are accepted (checked before anything is called), and a malformed state or a replay position outside the stored sequence is also a `ValueError`. The `model-dynamics` command does not write restart files (its trajectories are short); the `dynamics` stage of P7b will.

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

**Run times of the validation** (measured 2026-11-10, code on `/mnt/c`, 8 workers). Asserted: `pytest -m slow tests/validation/test_tully_exact.py` (26 tests: the exact-solver convergence and every asserted FSSH-vs-exact point above) takes about **94 minutes**, the two SAC $k = 10$ ensembles alone 29 minutes; `pytest -m sharc` (all SHARC comparisons, 16 passed and the documented 3000 K `wigner.py` xfail, **at the end of P7a-1; with the P7a-2 tests it is longer, §8.15.9**) takes about **4 minutes**, of which the single-trajectory Tully tests are 3.3 minutes. Report scripts (run by hand): `tools/validation/tully_report.py` about 20–30 minutes and `tools/sharc_compare/tully_ensembles.py` about 1 hour (SHARC is about 15 times slower per trajectory than our core: a file round trip and an awk process per step). None of them is part of the fast tier or of CI.

### 8.15 LVC models, spin–orbit coupling and the diagonal representation
**[IMPLEMENTED]** (P7a-2, `gqteashmd.dynamics`, command `gqteashmd model-dynamics` with `[model] name = "lvc"`, §5.7.1; **[PLANNED]** for molecules: the PySCF backend provides no triplets or spin–orbit matrix elements yet, so the `dynamics` stage (§8.16) is singlets only and never has $\mathbf U \ne \mathbf 1$). The package keeps the dependency rule of §8.14 (NumPy only; nothing imported from `electronic/`, `stages/`, the orchestrator or `config/`). Units are those of §8.10. The methods are implemented from the literature and the public SHARC 4.1 manual (the clean-room rule of §12 reference 15): Köppel, Domcke & Cederbaum, *Adv. Chem. Phys.* **57**, 59 (1984) (vibronic coupling models), Granucci, Persico & Toniolo, *J. Chem. Phys.* **114**, 10608 (2001) (local diabatization), Mai, Marquetand & González, *Int. J. Quantum Chem.* **115**, 1215 (2015) and *WIREs Comput. Mol. Sci.* **8**, e1370 (2018) (hopping in arbitrary representations, the diagonal representation).

**Symbols.** $N_{\text{exp}}$ is the number of spin-expanded states; $\alpha,\beta$ label **MCH** states (eigenstates of the spin-free electronic Hamiltonian, with the spin components expanded), $a,b$ label **diagonal** states (eigenstates of the complex MCH Hamiltonian $\mathbf H^{\text{MCH}}$, which contains the spin–orbit coupling); $\lambda$ is the active **diagonal** state. $\mathbf U$ is the unitary $N_{\text{exp}}\times N_{\text{exp}}$ matrix with $\mathbf H^{\text{MCH}} = \mathbf U\,\mathrm{diag}(E^{\text{diag}})\,\mathbf U^\dagger$ (column $a$ = diagonal state $a$ in the MCH basis). $\mathbf c = \mathbf c^{\text{diag}}$ are the amplitudes in the diagonal basis and $\mathbf c^{\text{MCH}} = \mathbf U\mathbf c^{\text{diag}}$.

#### 8.15.1 Spin expansion and canonical order
A model has state counts $(n_1, n_2, n_3)$ (singlets, doublets, triplets; the template's second line, e.g. `2 0 2`). **Only singlets and triplets are allowed**: a doublet count, more than three counts, a negative or non-integer count, or no state at all is a `ValueError`. A triplet state contributes three components $M_s = -1, 0, +1$, so $N_{\text{exp}} = n_1 + 3n_3$. The **canonical order** puts the state index innermost, then $M_s$, then the multiplicity: $S_0,\dots,S_{n_1-1}$, then $T_1,\dots,T_{n_3}$ with $M_s=-1$, then the same with $M_s = 0$, then with $M_s=+1$. For `2 0 2` it is $S_0, S_1, T_1(-1), T_2(-1), T_1(0), T_2(0), T_1(+1), T_2(+1)$. Expanded labels are written `T1(-1)`, `T1(0)`, `T1(+1)`; the spin-free labels (`S0`, `S1`, `T1`, `T2`) are the sums over $M_s$ (§8.15.7). SHARC's gradient and coupling labels follow exactly this order (checked, §8.15.9).

#### 8.15.2 The LVC model and its files
**Normal coordinates.** With the masses $M_A$ in electron masses and $\mathbf K$ the orthogonal $3N\times 3N$ matrix of mass-weighted normal modes of the file (one mode per column; orthogonality is not checked), the dimensionless coordinates are
$Q_i = \sqrt{\omega_i}\sum_{A,k}K_{(A,k),i}\sqrt{M_A}\,(r_{A,k} - r^{\text{ref}}_{A,k})$, with $\partial Q_i/\partial r_{A,k} = \sqrt{\omega_i}K_{(A,k),i}\sqrt{M_A}$. A mode with $\omega_i = 0$ exactly (translation, rotation) has $Q_i \equiv 0$ and is ignored everywhere; **only $\omega = 0$ exactly is treated as a zero mode**. [IMPLEMENTED, final fix wave] A frequency $0 < \omega_i <$ `ZERO_MODE_THRESHOLD_EH` $= 10^{-6}\,E_h$ (about 0.22 cm⁻¹, far below any real vibration) is **refused** when the model is built: the error names the mode (1-based, as in the file), its frequency and the threshold, and asks you to set the external modes to exactly 0 in `V0.txt` (`model-dynamics`: exit 2). Without this check the coordinate transformation and the Wigner sampling would divide by $\sqrt{\omega_i}$ (Wigner widths of about 48 bohr for a carbon atom at $10^{-8}\,E_h$). External modes with larger residual frequencies (for example the 1–40 cm⁻¹ of a numerical Hessian) are **not** detected and are treated as vibrations: write them as exact zeros. Negative frequencies are refused as before. The reference potential is $V_0 = \sum_i\omega_iQ_i^2/2$. The model works on the flat coordinate vector $\mathbf q$ (bohr, length $3N$, atom order of `V0.txt`) and the masses $M_A$ repeated three times.

**Spin-free diabatic block** of multiplicity $m$ (one real symmetric $n_m\times n_m$ matrix, shared by its $M_s$ components): $V^{(m)}_{\alpha\alpha} = V_0 + \varepsilon^{(m)}_\alpha + \sum_i\kappa^{(m)}_{\alpha,i}Q_i$ and $V^{(m)}_{\alpha\beta} = \sum_i\lambda^{(m)}_{\alpha\beta,i}Q_i$ ($\alpha\ne\beta$; a $\lambda$ entry for $(\alpha,\beta)$ also sets $(\beta,\alpha)$). **Diabatic spin–orbit coupling** $\boldsymbol\Sigma$ is a constant complex Hermitian $N_{\text{exp}}\times N_{\text{exp}}$ matrix in the canonical order, `SOC R` $+ i\,$`SOC I` of the template (absent: $\boldsymbol\Sigma = 0$). A $\boldsymbol\Sigma$ with $\max|\boldsymbol\Sigma-\boldsymbol\Sigma^\dagger| > 10^{-8}\,E_h$ is a `ValueError` (this check also runs with `spin_orbit = false`); smaller deviations are removed by $(\boldsymbol\Sigma+\boldsymbol\Sigma^\dagger)/2$.

**MCH quantities.** For each multiplicity $\mathbf V^{(m)} = \mathbf W^{(m)}\mathrm{diag}(\mathbf E^{(m)})\mathbf W^{(m)\mathsf T}$, eigenvalues ascending; the sign of each column is aligned to the previous geometry's vectors (largest-component positive at the first geometry). $\mathbf T$ is the block-diagonal real orthogonal matrix with $\mathbf W^{(m)}$ on every $(m, M_s)$ block. The snapshot holds the spin-free energies $E_\alpha$, $\mathbf H^{\text{MCH}} = \mathrm{diag}(E) + \mathbf T^{\mathsf T}\boldsymbol\Sigma\mathbf T$ (complex Hermitian), the gradients $g_\alpha = (\mathbf W^{\mathsf T}\partial\mathbf V\mathbf W)_{\alpha\alpha}$ (the same for every $M_s$ component), the NACs $\mathbf K_{\alpha\beta} = (\mathbf W^{\mathsf T}\partial\mathbf V\mathbf W)_{\alpha\beta}/(E_\beta - E_\alpha)$ within one $(m, M_s)$ block and **exactly zero between blocks**, $\mathbf T$ itself and the **overlaps** $S_{\beta\alpha} = \langle\psi_\beta(t)|\psi_\alpha(t+\Delta t)\rangle = (\mathbf T_{\text{prev}}^{\mathsf T}\mathbf T)_{\beta\alpha}$ (§8.15.5). $\partial\mathbf V^{(m)}/\partial r = \sum_i(\omega_iQ_i\mathbf 1 + \mathrm{diag}(\boldsymbol\kappa_{\cdot,i}) + \boldsymbol\Lambda_i)\,\partial Q_i/\partial r$ with $\boldsymbol\Lambda_i$ the symmetric matrix of the $\lambda$ entries of mode $i$. A degeneracy $|E_\beta - E_\alpha| < 10^{-12}\,E_h$ inside one $(m, M_s)$ block raises a `ValueError` (the NAC is undefined); degeneracies between blocks (for example the three $M_s$ components of a triplet) are allowed and are the normal case. The Tully models and the 2D test model also return overlaps ($\mathbf S = \mathbf W_{\text{prev}}^{\mathsf T}\mathbf W$ after the sign alignment), so `coupling = "overlap"` also runs on them.

**Wigner initial conditions** (T = 0; `sampling = "wigner"`). For each mode with $\omega_i > 0$, $Q_i, P_i \sim N(0,\tfrac12)$ independently (the ground-state Wigner function of $\omega_i(Q_i^2+P_i^2)/2$), then $\mathbf r = \mathbf r^{\text{ref}} + \sum_i\mathbf K_{\cdot,i}Q_i/(\sqrt{\omega_i}\sqrt{\mathbf M})$ and $\mathbf v = \sum_i\mathbf K_{\cdot,i}\sqrt{\omega_i}P_i/\sqrt{\mathbf M}$; zero modes get $Q = P = 0$ and draw nothing. The draws are `rng.standard_normal((n, n_vib, 2))` (per sample, modes in increasing index, $Q$ then $P$), from the `model-initial` stream of §8.14.

**Files (SHARC 4.1 formats, §6.4.1 of its manual; read and written natively).** `LVC.template`: line 1 the path of the `V0.txt` file (relative to the template's directory), line 2 the state counts, then the sections `epsilon`, `kappa`, `lambda`, `lambda_soc`, `gamma` (each a count line followed by the entries, 1-based indices: `epsilon`: multiplicity, state, value; `kappa`: multiplicity, state, mode, value; `lambda`: multiplicity, state 1, state 2, mode, value), the $N_{\text{exp}}\times N_{\text{exp}}$ matrices `SOC R`, `SOC I`, `DMX R`, … `DMZ I`, and `Multipolar Density Fit` (skipped by its count). Energies and frequencies are in $E_h$, coordinates in bohr, masses in amu. `V0.txt`: the sections `Geometry` (element, nuclear charge, $x\,y\,z$ in bohr, mass in amu), `Frequencies` ($3N$ values) and `Mass-weighted normal modes` (a $3N\times3N$ matrix). Header matching ignores case and extra white space. **Supported scope:** `gamma` and `lambda_soc` with a count above 0 are refused (the manual shows only the count line); the dipole matrices are read and stored but not used by the dynamics; SHARC's `LVC.resources` is not used (our model always diagonalises, does no Kabsch alignment and no atom permutation). A parse error is a `ValueError` naming the file and line number: an unknown keyword, a duplicate section or entry, an index of 0 (indices are 1-based) or out of range, a short row, a doublet count. `format_lvc_template` and `format_v0` write the same formats (`%.12e` for templates, `%.16e` for `V0.txt`) and reading what was written gives the same model.

**The fixtures.** `tests/data/lvc/` holds our own invented model, `V0.txt` and three templates: a C atom with two O atoms (bond 2.4 bohr, angle 120°, force constants 0.45 $E_h/\text{bohr}^2$ for the stretches and 0.18 $E_h/\text{rad}^2$ for the bend; modes 7–9 = bend, symmetric stretch and asymmetric stretch, 458.5, 1156.9 and 1493.3 cm⁻¹; six zero-frequency modes). `lvc_singlets` (3 singlets), `lvc_soc` (`2 0 2`, with $\boldsymbol\Sigma$ of $10^{-4}$ to $6\cdot10^{-4}\,E_h$ between singlets and triplets, and dipole matrices) and `lvc_twomode` (`2 0 1`, couplings on modes 7 and 9 only). `tests/data/lvc/make_fixtures.py` regenerates them. **The numbers of SHARC's own manual examples were not used.**

#### 8.15.3 The diagonal representation, $\mathbf U$ and the step order
**Identity case.** If $\mathbf H^{\text{MCH}}$ has no non-zero off-diagonal element and there is no previous $\mathbf U$, then $\mathbf U$ is `None` ($\mathbf U\equiv\mathbf 1$ in MCH order) and $E^{\text{diag}} = \mathrm{Re}\,\mathrm{diag}(\mathbf H^{\text{MCH}})$ with no arithmetic. Every code path then falls back to the P7a-1 code, so **the trajectories of every model without off-diagonal MCH elements (the Tully models, singlet LVC models, triplets without spin–orbit coupling) are bit-identical to P7a-1** (a golden-trajectory test, §8.15.9). The diagonal states are energy-ordered whenever $\mathbf U$ exists; an exactly diagonal $\mathbf H$ keeps the MCH order.

**Otherwise** $\mathbf U$ comes from `eigh` (eigenvalues ascending) and is made continuous in time. Group the ascending eigenvalues into **degenerate blocks** $B$ (consecutive eigenvalues closer than $10^{-10}\,E_h$ are chained); for each block form $\mathbf M = \mathbf U_{:,B}^\dagger\mathbf U^{\text{prev}}_{:,B}$ with the singular value decomposition $\mathbf M = \mathbf X\boldsymbol\Sigma_M\mathbf Y^\dagger$ and set $\mathbf U_{:,B}\leftarrow\mathbf U_{:,B}\mathbf X\mathbf Y^\dagger$ (the polar factor: the unitary correction closest to continuity; a $1\times1$ block with $|M| < 10^{-14}$ is left unchanged). This is the exactly unitary form of the projection of the manual (eq. 8.138), and a pure phase would fail inside a block because $\mathbf U$ is only defined up to a unitary mixing there (the three $M_s$ components of a triplet are exactly degenerate without spin–orbit coupling). At the first step (no previous $\mathbf U$) each column is made to have a real, positive largest component instead. $E^{\text{diag}}_a = \mathrm{Re}(\mathbf U^\dagger\mathbf H^{\text{MCH}}\mathbf U)_{aa}$. The active MCH state is $\arg\max_\alpha|U_{\alpha\lambda}|$ (lowest index on ties).

**Propagation** is in the diagonal basis, §8.11: $\mathbf c^{\text{diag}}(t+\Delta t) = \mathbf R^{\text{diag}}\mathbf c^{\text{diag}}(t)$ with $\mathbf R^{\text{diag}} = \mathbf U(t+\Delta t)^\dagger\mathbf R^{\text{MCH}}\mathbf U(t)$ and, for `coupling = "nacdr"`, $\mathbf R^{\text{MCH}} = \prod_m\exp[-i(\mathbf H(s_m) - i\mathbf T(s_m))\delta]$ with $\mathbf H(s) = (1-s)\mathbf H^{\text{MCH}}(t) + s\,\mathbf H^{\text{MCH}}(t+\Delta t)$ and $\mathbf T = \mathbf v\cdot\mathbf K$ in the chosen interpolation (§8.11). The $s_m$ and the products are those of §8.11 (later substeps on the left). If $\mathbf U$ is `None` at both ends and the coupling is `nacdr`, the P7a-1 propagator is called unchanged. **Hopping** uses the SHARC-type probability of §8.12 with $\mathbf R^{\text{diag}}$ and $\mathbf c^{\text{diag}}$ (the Tully flux formula is MCH-only, §8.12). The energy difference of a hop is $\Delta E = E^{\text{diag}}_j - E^{\text{diag}}_\lambda$ at $\mathbf q'$. **`parallel_vel`** is as in §8.12. **`parallel_nac`** uses $\mathbf u = \mathrm{Re}\,\mathbf K^{\text{diag}}_{\lambda j}$ (§8.15.4) from the masked snapshot at $\mathbf q'$, and **stops with `ValueError("parallel_nac needs a real coupling vector; with spin-orbit coupling use parallel_vel")`** when $\max|\mathrm{Im}\,\mathbf K^{\text{diag}}_{\lambda j}| > 10^{-10}\max(1,\max|\mathbf K^{\text{diag}}_{\lambda j}|)$ (the SHARC manual does not say how a complex vector is made real; `parallel_vel` is SHARC's standard with spin–orbit coupling). With $\mathbf U$ = `None` the vector is the snapshot's NAC exactly (P7a-1). Frustrated hops (keep/reflect) and the EDC are as in §8.12, applied to $\mathbf c^{\text{diag}}$ with $E^{\text{diag}}$ and the kinetic energy after the hop.

**Initial state.** The state given by the user is an **MCH** state index $\alpha_0$ (and, optionally, an MCH vector $\mathbf c^{\text{MCH}}$). $\mathbf U_0$ is built without a previous $\mathbf U$, $\mathbf c^{\text{diag}} = \mathbf U_0^\dagger\mathbf c^{\text{MCH}}$ (with $\mathbf c^{\text{MCH}} = \mathbf e_{\alpha_0}$ when none is given), and $\lambda = \arg\max_a|c^{\text{diag}}_a|^2$ (lowest index on ties). This is the mapping SHARC uses for `state N mch` with `coeff auto` (measured, §8.15.9). With $\mathbf U_0$ = `None` it is the P7a-1 behaviour. **A model whose Hamiltonian is diagonal at the start but acquires off-diagonal elements later** (so that $\mathbf U$ would appear in mid-trajectory with the active state already chosen) **is refused** with `the MCH Hamiltonian became non-diagonal during the trajectory (U was the identity); start from a geometry where the spin-orbit coupling is present`; the reverse (non-`None` $\mathbf U$, diagonal $\mathbf H$) keeps tracking $\mathbf U$. No model in scope behaves so, because the diabatic spin–orbit matrix is constant.

**Step order with the diagonal representation** (ours; replaces the list of §8.12, to which it reduces when $\mathbf U$ is `None` and `eselect_Eh` is unset; the hop is still after the velocity completion, so $E_{\text{tot}}$ is continuous across hops). From $(\mathbf q,\mathbf v,\mathbf c^{\text{diag}},\lambda,\text{snapshot},\mathbf U)$:

1. Mask the snapshot around $E^{\text{diag}}_\lambda(\mathbf q)$ (§8.15.6), then $\mathbf a = -\mathbf M^{-1}\mathbf g^{\text{diag}}_\lambda$ (§8.15.4).
2. $\mathbf q' = \mathbf q + \mathbf v\Delta t + \tfrac12\mathbf a\Delta t^2$.
3. $\text{snap}'$ = model evaluated at $\mathbf q'$ with the previous snapshot (sign continuity, overlaps), **unmasked**.
4. $(E^{\text{diag}\prime},\mathbf U')$ = diagonal basis of $\text{snap}'.\mathbf H^{\text{MCH}}$ against $\mathbf U$.
5. Mask $\text{snap}'$ around $E^{\text{diag}\prime}_\lambda$ (old $\lambda$); then $\mathbf G^{\text{diag}\prime}$ and $\mathbf v' = \mathbf v + \tfrac12(\mathbf a - \mathbf M^{-1}\mathbf g^{\text{diag}\prime}_\lambda)\Delta t$.
6. Propagate $\mathbf c^{\text{diag}}$ over $[t, t+\Delta t]$ with the masked snapshots, $\mathbf v$, $\mathbf v'$, $\mathbf U$, $\mathbf U'$ and the chosen coupling (§8.11, §8.15.5): $\mathbf c'$ and $\mathbf R^{\text{diag}}$.
7. Hopping probabilities from $\lambda$; draw $\xi$ (**always exactly one number**, §8.14).
8. If there is a target $j$: velocity adjustment (above); if accepted, $\lambda\leftarrow j$. With `eselect_Eh`, $\text{snap}'$ is **not** masked again inside the step: the new $\lambda$ takes effect at step 1 of the next step, which masks the stored (unmasked) snapshot around it and so plays the part of the "second electronic-structure call" of a real backend after a hop.
9. EDC on $\mathbf c'$.
10. $\mathbf a'$ is recomputed from the stored state at the next step's (1), not stored.
11. Record (with $E_{\text{tot}} = E^{\text{diag}\prime}_\lambda + \tfrac12\mathbf v'^{\mathsf T}\mathbf M\mathbf v'$) and check the energy drift as in §8.10. The record also carries $E^{\text{diag}}$, $\mathbf c^{\text{MCH}} = \mathbf U'\mathbf c'$, the active MCH state and a copy of $\mathbf U'$.

#### 8.15.4 Gradients in the diagonal basis
The force on a diagonal state is not the gradient of an MCH energy. The **MCH gradient tensor** is the real symmetric $N_{\text{exp}}\times N_{\text{exp}}\times f$ array $G^{\text{MCH}}_{\alpha\alpha} = \mathbf g_\alpha$ and $G^{\text{MCH}}_{\alpha\beta} = (E_\beta - E_\alpha)\mathbf K_{\alpha\beta}$, i.e. $\langle\alpha|\partial\hat H_{\text{SF}}|\beta\rangle$, **with the spin-free energies $E$** (not $\mathrm{diag}\,\mathbf H^{\text{MCH}}$; the two agree when the diagonal of $\mathbf T^{\mathsf T}\boldsymbol\Sigma\mathbf T$ vanishes, which holds for every fixture; with the spin-free energies the transformation is exact for a constant diabatic $\boldsymbol\Sigma$). With `gradient_transformation = "ngt"` (default) $\mathbf G^{\text{diag}} = \mathbf U^\dagger\mathbf G^{\text{MCH}}\mathbf U$ component by component; `"none"` (SHARC's `nogradcorrect`, for comparison only) sets the off-diagonal terms of $\mathbf G^{\text{MCH}}$ to zero first. The diagonal gradient is $\mathbf g^{\text{diag}}_a = \mathrm{Re}\,G^{\text{diag}}_{aa}$ and the diagonal NAC $\mathbf K^{\text{diag}}_{ab} = G^{\text{diag}}_{ab}/(E^{\text{diag}}_b - E^{\text{diag}}_a)$ for $a\ne b$ (the zero vector if $|E^{\text{diag}}_b - E^{\text{diag}}_a| < 10^{-12}\,E_h$). With $\mathbf U$ = `None`, $\mathbf G^{\text{diag}} = \mathbf G^{\text{MCH}}$ and no arithmetic is done. For a constant diabatic $\boldsymbol\Sigma$, $\mathbf g^{\text{diag}}_a = \partial E^{\text{diag}}_a/\partial\mathbf q$ exactly (Hellmann–Feynman), which the finite-difference tests assert (§8.15.9). Without the transformation the energy is not conserved: its drift is 151.6 times larger (§8.15.9).

#### 8.15.5 Local diabatization (`coupling = "overlap"`)
**[IMPLEMENTED]** for models with overlaps (Tully, 2D test and LVC) and, since P7b, for PySCF wavefunctions (the overlap engine of §8.6.8, used by the `dynamics` stage, §8.16; the overlaps are Löwdin-orthonormalised first). The method is that of Granucci, Persico & Toniolo (2001) as the SHARC manual states it (eqs. 8.185–8.199). With the overlap matrix $S_{\beta\alpha} = \langle\psi_\beta(t)|\psi_\alpha(t+\Delta t)\rangle$ of the MCH states at $t$ and $t+\Delta t$ (**no NACs are used**):

$$
\mathbf H_{\text{tra}} = \mathbf S\,\mathbf H^{\text{MCH}}(t+\Delta t)\,\mathbf S^\dagger,\qquad
\tilde{\mathbf H}(s) = (1-s)\,\mathbf H^{\text{MCH}}(t) + s\,\mathbf H_{\text{tra}},\qquad
\mathbf R^{\text{MCH}} = \mathbf S^\dagger\prod_m\exp[-i\tilde{\mathbf H}(s_m)\delta] .
$$

($\mathbf S^\dagger$ takes amplitudes from the basis at $t$ to the basis at $t+\Delta t$; $\mathbf H_{\text{tra}}$ is $\mathbf H^{\text{MCH}}(t+\Delta t)$ written in the basis at $t$.) $\mathbf R^{\text{diag}} = \mathbf U'^\dagger\mathbf R^{\text{MCH}}\mathbf U$ as above. The propagator is exactly unitary when $\mathbf S$ is; the model overlaps are exactly orthogonal, a PySCF overlap matrix is not, and needs Löwdin orthonormalisation first (P7b). The orientation of $\mathbf S$ and of $\mathbf U$, $\mathbf U'$ is pinned by exactness tests (a moving non-symmetric basis; frozen nuclei), which fail under every one of three deliberate orientation mistakes (§8.15.9). `coupling = "overlap"` is not allowed with `hopping = "fssh"`.

#### 8.15.6 Energy-based selection of gradients and couplings (`eselect_Eh`)
With $\varepsilon$ = `eselect_Eh` (a window in $E_h$; SHARC's `eselect` is in eV) and the reference energy $E^{\text{diag}}_\lambda$ of the active state at that geometry, the gradient of the MCH state $\alpha$ is kept if $|E_{\text{ref}} - E_\alpha| < \varepsilon$ (strictly less), a coupling $\mathbf K_{\alpha\beta}$ if both $\alpha$ and $\beta$ satisfy it; everything else is **set to zero** in one masked snapshot used throughout the step (the gradient tensor, the propagation couplings, the hop direction). This models a backend that does not compute the deselected quantities (P7b); at $\mathbf q$ the reference is the $\lambda$ after the previous hop, at $\mathbf q'$ the old $\lambda$ for the whole step, also after an accepted hop ($\text{snap}'$ is not masked again inside the step; the new $\lambda$ takes effect when the next step masks the same stored, unmasked snapshot around it, §8.15.3 steps 1 and 8). Unset (the default): nothing is masked and no energies are computed. A window larger than every gap gives results **bit-identical** to the unset case (asserted). **`eselect` cannot be compared with SHARC's through `SHARC_LVC.py`:** SHARC's own selection (`eselect 0.3`, `grad_select`, `nac_select`) asks the interface for the same set of states as our mask (measured), but `SHARC_LVC.py` returns **every** gradient and coupling whatever is requested, so SHARC's selected trajectory is bit-identical to its unselected one while ours differs from our unselected one by 2.5·10⁻⁵ bohr after 300 steps (report, §8.15.9). A like-for-like check needs an interface that honours partial requests (our swap provider could be made to; not done).

#### 8.15.7 Populations, ensembles and determinism
`gqteashmd model-dynamics` with an LVC input records, for every trajectory and recorded step, four views, each a vector over the $N_{\text{exp}}$ expanded states: diagonal classical $\delta_{a\lambda}$; diagonal quantum $|c^{\text{diag}}_a|^2$; MCH classical $|U_{\alpha\lambda}|^2$; MCH quantum $|(\mathbf U\mathbf c^{\text{diag}})_\alpha|^2$ ($\mathbf U$ = `None` means $\mathbf 1$); and the **spin-free** MCH populations, the sums over the $M_s$ components of each (multiplicity, state) (`S0`, `S1`, `T1`, `T2`). Time 0 is recorded before the first step, and a step is recorded when its number is a multiple of `record_every`; every trajectory takes exactly `n_steps`. The ensemble averages are taken over the trajectories that did **not** end with `energy_drift`, which are excluded and counted (a changing set would mix the populations of different ensembles). Trajectory $i$ uses child $i$ of `SeedSequence([seed, crc32(b"model-dynamics")])` and the Wigner conditions the stream `model-initial` (§8.14); one random number per step, used or not; the results do not depend on `workers` (asserted: 1 and 3 workers give equal arrays). The workers are `spawn` processes.

#### 8.15.8 Restart format 2
`restart.save_state`/`load_state` (§8.14) write format **version 2**, which adds $\mathbf U$ (flag `has_u`) and the snapshot overlaps (`has_overlaps`) to the P7a-1 contents (the consistency of flags and arrays and the shapes against the model are checked). **Version-1 files are still read**, with $\mathbf U$ = `None`, provided their Hamiltonian has no off-diagonal element (a P7a-1 file never has one; otherwise `ValueError` mentioning "off-diagonal"). Continuing is **bit-identical** also with spin–orbit coupling and hops: 60 steps + save + load + 60 steps equals 120 steps for both couplings (`lvc_soc`, EDC, hops in each half; asserted). A frozen P7a-1 file (`restart_v1.npz`) and frozen P7a-1 trajectories (`tully_golden.npz`, SAC, DAC and ECR with hops and a frustrated reflection) are kept in the tests; the trajectories were generated with the unchanged P7a-1 code and are compared bit for bit.

#### 8.15.9 Validation (measured)
Every number is marked **asserted** (a test fails if the stated bound is broken; the bound is in brackets) or **(report)** (printed or written to a JSON file by a script that asserts nothing). Where an assertion took its final form **after the numbers were seen**, that is said; nothing was loosened without being stated here.

**(a) Unit checks, all asserted in the fast tier** (measured by the implementers of Tasks 2, 4, 5 and 6; bounds in brackets):

| Check | Measured | Bound |
|---|---|---|
| $\mathbf U$ continuity along a smooth path (200 points, 4 states): max over steps and columns of $|\langle U^{\text{prev}}_a|U_a\rangle - 1|$ | 4.79e-8 | 1e-6 |
| Degenerate block, polar factor, max $|\mathbf U - \mathbf U^{\text{prev}}|$ (a pure phase rule gives 1.48) | 1.0e-13 | 1e-12 |
| Triplet group merged and re-split along a path (a pure phase rule gives 0.79, not asserted) | 1.9e-11 | 1e-6 |
| Singlet–triplet SOC, threefold triplet split 1+2: non-degenerate columns / degenerate pair | 7.7e-9 / 2.7e-13 | 1e-6 |
| Hellmann–Feynman: $\mathbf g^{\text{diag}}$ vs finite differences of $E^{\text{diag}}$ | 1.7e-11 | 1e-7 |
| Diagonal NAC vs finite differences of the tracked basis | 4.0e-11 | 1e-6 |
| LVC gradients vs finite differences (h = 1e-5; `lvc_singlets` / `lvc_soc` / `lvc_twomode`) | 3.8e-10 / 5.8e-11 / 2.5e-12 | 1e-8 |
| LVC NACs vs finite differences of $\mathbf T$ | 6.1e-8 / 1.8e-8 / 1.1e-11 | 1e-6 |
| LVC overlaps, $\mathbf S^{\mathsf T}\mathbf S - \mathbf 1$ over 50 steps | 1.6e-15 / 6.7e-16 / 5.6e-16 | 1e-13 |
| SOC: $\mathbf g^{\text{diag}}$ vs finite differences of $E^{\text{diag}}$ (3 geometries) | 2.1e-11 / 1.2e-11 | 1e-7 |
| One-mode LVC vs a hand-built diabatic model (all fields incl. overlaps) | 4.4e-16 | 1e-13 |
| Wigner sampling (n = 20000): largest deviation of a mean or variance, in standard errors | 2.95 | 4 |
| Diagonal propagator unitarity $\|\mathbf R^\dagger\mathbf R - \mathbf 1\|_{\max}$ (both couplings and interpolations) | ≤ 1.0e-14 | 1e-12 |
| Frozen nuclei vs $\exp(-i\mathbf Ht)$ (100 steps); Rabi oscillation (200 steps) | 2.8e-13; 9.6e-14 | 1e-10 |
| Orientation tests (moving non-symmetric basis; frozen nuclei with $\mathbf U'\ne\mathbf U$) | ≤ 2.2e-15 | 1e-12 |
| Trivial crossing (swap of two states): local diabatization keeps the diabatic population | 1.3e-15 from 1 | 1e-12 |
| The same crossing with `nacdr` (the failure that local diabatization fixes) | diabatic population 0.000 | asserted $<0.5$ |
| `nacdr` vs `overlap` on Tully SAC, differences at $\Delta t$ = 1, 0.5, 0.25 | 5.34e-6, 1.34e-6, 3.34e-7 (ratios 3.998, 3.9995) | ratios in [3, 5] |
| `lvc_soc`, hopping off: $\max|E_{\text{tot}}-E_{\text{tot}}(0)|$, 2000 steps $\Delta t = 0.5$ / 4000 steps $\Delta t = 0.25$ (ratio 4.0000) | 1.166e-8 / 2.915e-9 $E_h$ | $<10^{-5}$; ratio in [3, 5] |
| Hop jumps in $E_{\text{tot}}$ (17 hops, 1000 steps) | ≤ 5.55e-10 $E_h$ | $<10^{-8}$ |
| `gradient_transformation = "none"` drift relative to `"ngt"` | 151.6 times larger (1.767e-6 $E_h$) | $>10\times$ |
| $\mathbf g^{\text{diag}}_\lambda$ vs finite differences of $E^{\text{diag}}_\lambda$ along a trajectory (5 steps, h = 1e-5) | ≤ 2.5e-11 | 1e-6 |
| `eselect_Eh` = 10 (larger than every gap) vs unset, 300 steps with hops | bit-identical | identical |
| `eselect_Eh` = 0.05, 500 steps: drift; $\max|\Delta\mathbf q|$ against the unset run | 2.18e-7 $E_h$; 2.7e-6 bohr | drift $<10^{-3}$ |
| Singlet LVC, `parallel_nac` (10 hops, 439 frustrated): drift | 5.16e-7 $E_h$ | $<10^{-5}$ (chosen before the run, not given by the plan) |
| Restart 60 + 60 = 120 steps with $\mathbf U$, overlaps, hops, both couplings | bit-identical | identical |
| End-to-end orientation check (final fix wave): `lvc_soc`, hopping and decoherence off, 100 a.u.; $\max\lvert\mathbf T\mathbf U\mathbf c^{\text{diag}} - \mathbf c^{\text{dia}}_{\text{exact}}\rvert$ (up to a global phase) against a fourth-order Magnus integration of the diabatic Schrödinger equation along the same nuclear path, $\Delta t$ = 1 / 0.5 | `nacdr` 2.75e-8 / 6.87e-9 (ratio 4.00); `overlap` 3.80e-9 / 9.52e-10 (ratio 4.00) | $<10^{-6}$ at 0.5; ratio in [3, 5] |
| Tully SAC with a constant complex *diabatic* coupling (0.005i), $x = -6 \to +6$, 1200 steps $\Delta t = 1$, hopping off: drift (final fix wave) | 2.52e-7 $E_h$ (6.29e-8 at $\Delta t = 0.5$, ratio 4.00, not asserted) | $<10^{-5}$ (chosen before the run) |

Two notes on these tests. (1) The fixture's plain hop probabilities are tiny (at most $6.9\cdot10^{-6}$ per step), so the tests that need hops use a random stream in which every 10th number is forced to 0 (as the P7a-1 ECR golden stream does), which gives hops at steps 1, 21, 24, 61, 101. (2) **An extra constant-coupling test changed after a number was seen:** a Tully SAC trajectory with a *constant MCH-basis* coupling added drifted by 8.6·10⁻³ $E_h$ from $x = -3$ and was moved to start at $x = -10$ (drift 2.0·10⁻¹² $E_h$ there), after the number was seen. The cause, verified by the reviewer, is the model, not the driver: a constant MCH coupling is not a constant diabatic coupling where the NAC is large, so the spin-free gradient tensor misses $\partial H_{\text{SO}}$ and the model is not conservative (drift independent of $\Delta t$, finite-difference check failing by 1.4·10⁻²); a constant *diabatic* coupling crossing the same region drifts 6·10⁻⁸ $E_h$ with $\Delta t^2$ scaling. In the final fix wave that test was **replaced** by a constant *diabatic* coupling crossing the coupling region (the table row above; bound fixed before the run). (3) The orientation of the MCH populations ($|U_{\alpha\lambda}|^2$ from the column of $\lambda$, $\mathbf U\mathbf c$, the sums over $M_s$) is asserted against hand-computed values for a non-symmetric complex $\mathbf U$; each wrong variant (row instead of column, $\mathbf U^\dagger\mathbf c$, $\mathbf U^T\mathbf c$, a state-index-outermost $M_s$ sum) was checked to fail. The end-to-end orientation check catches $\mathbf U'^T$ for $\mathbf U'^\dagger$, the sign of $\mathbf T$, $\mathbf S$ for $\mathbf S^\dagger$ and a transposed $\mathbf H_{\text{tra}}$ (errors 0.96, 2.3e-3, 2.3e-3 and 8.8e-5 against the bound 1e-6). Its first version used a second-order reference with 100 substeps per a.u., whose own error moved the `overlap` ratio to 4.996; the reference was made fourth order **after that number was seen**, the bounds unchanged.

**(b) Local diabatization against the exact quantum solution** (asserted, `pytest -m slow tests/validation/test_tully_overlap_exact.py`, 4 tests, **677.5 s** in one run, measured 2026-11-10). Points and rule fixed in the plan before the run: `coupling = "overlap"`, `hopping = "sharc"`, `parallel_vel`, keep, $\Delta t = 0.5$ a.u., 25 substeps, seed 20261108, 8 workers, Wigner sampling, SAC and DAC without decoherence, ECR with the EDC of the `"sharc"` convention; rule $|P - P_{\text{exact}}| \le 4\,\mathrm{SE} + 0.02$ on the branches that P7a-1 asserts; for every ensemble also `energy_drift == 0` and at most 0.1 % unfinished (both met: 0). The exact runs had norm − 1 $\le 8.1\cdot10^{-13}$.

| Point (n) | Branch | FSSH, 95 % interval | Exact | Difference | Tolerance | Asserted |
|---|---|---|---|---|---|---|
| SAC $k$ = 20 (2000) | transmitted lower | 0.5245 [0.5026, 0.5463] | 0.5077 | +0.0168 | 0.0647 | yes |
| | transmitted upper | 0.4755 [0.4537, 0.4974] | 0.4923 | −0.0168 | 0.0647 | yes |
| DAC $\ln E$ = 0 (1000) | transmitted lower | 0.7530 [0.7253, 0.7787] | 0.7449 | +0.0081 | 0.0751 | yes |
| | transmitted upper | 0.2470 [0.2213, 0.2747] | 0.2551 | −0.0081 | 0.0751 | yes |
| DAC $\ln E$ = 1 (1000) | transmitted lower | 0.1510 [0.1301, 0.1745] | 0.1315 | +0.0195 | 0.0627 | yes |
| | transmitted upper | 0.8490 [0.8255, 0.8699] | 0.8685 | −0.0195 | 0.0627 | yes |
| ECR $k$ = 30, EDC `sharc` (1000) | transmitted upper | 0.3590 [0.3299, 0.3892] | 0.3742 | −0.0152 | 0.0812 | yes |
| | reflected lower | 0.0150 [0.0091, 0.0246] | 0.0238 | −0.0088 | 0.0393 | yes |
| | transmitted lower | 0.5860 [0.5552, 0.6161] | 0.5697 | +0.0163 | 0.0826 | no |
| | reflected upper | 0.0400 [0.0295, 0.0540] | 0.0323 | +0.0077 | 0.0424 | no |

All asserted differences are at most 0.0195; the largest $|\text{difference}|/\text{tolerance}$ is 0.31 (DAC $\ln E$ = 1). Wall times: SAC 340.3 s, DAC $\ln E$ = 0 51.7 s, DAC $\ln E$ = 1 34.5 s, ECR 231.6 s. The SAC reflected branches are 0 in both. The numbers agree with the `nacdr` runs of §8.14.1.

**(c) The exact 2D LVC solver and a population comparison** (`tests/validation/wavepacket2d.py`; the self-tests are asserted, the FSSH comparison is **(report)**, `tools/validation/lvc_exact_report.py`, nothing asserted: FSSH is not exact at a conical intersection). The model is `lvc_twomode` (`2 0 1`, couplings on $Q_7$ and $Q_9$, the other modes factor out exactly). $H = \sum_i\omega_iP_i^2/2 + \mathbf V(Q_7,Q_9)$ with $\mathbf V$ the complex spin-expanded diabatic matrix of §8.15.2 restricted to the two modes; Strang splitting with the kinetic factor in momentum space (2D FFT) and the potential exponential precomputed per grid point; initial state exp$(-(Q_7^2+Q_9^2)/2)/\sqrt\pi$ on the diabatic S1; MCH populations from the projection onto $\mathbf T(Q)$ at each point, summed over $M_s$; the potential is built independently of `lvc.py` and checked against `LVCModel`. **Grid: 256 × 128 points on $Q_7\in[-14,18)$, $Q_9\in[-8,8)$ ($\mathrm dQ = 0.125$), $\Delta t = 0.5$ a.u., the MCH projection on a 2× Fourier-interpolated grid.** *This replaced the grid of the plan* (128 × 128 on $[-8,8)^2$, a plan defect found by the implementer, ruled by the controller): the packet swings to $Q_7 = -2\kappa/\omega = -8.6$ on the diabatic S1 and, after reaching S0, to $+14.4$, so on the plan's grid it wraps around the periodic boundary (33 % of the density in the outer 3 %, S0 population at 2000 a.u. 0.0107 instead of 0.1370, energy error 5.1e-7 $E_h$); a test keeps that grid as a failing example and an edge-density guard (outer 3 % of either axis must stay below 1e-8 at every record) stops a run that wraps. The projection refinement was added **after** the pre-registered doubling-the-grid test failed (1.72e-4 against 1e-4): $\mathbf T(Q)$ has a conical singularity at the S0/S1 intersection ($Q_7 = -10$, $Q_9 = 0$), so the plain projected sum converges only as $\mathrm dQ^2$ while the wavefunction itself is converged (the diabatic populations changed by only 2.2e-11); the threshold was not changed. Self-tests (thresholds fixed before the run):

| Test | Marker | Measured | Bound |
|---|---|---|---|
| $\mathbf V$ and MCH energies equal `LVCModel` at 8 random points | fast | $\mathbf V$ exact (1e-15), $E$ to 1e-12 | asserted |
| A template coupling another mode is refused; the plan's grid trips the edge guard | fast | `ValueError`; `RuntimeError` | asserted |
| Norm over 2000 a.u. | fast | 7.4e-13 | 1e-10 |
| Energy over 2000 a.u. | fast | 7.9e-9 $E_h$ | 1e-8 (close; over the 4000 a.u. of the report 1.02e-8, not asserted) |
| Decoupled ($\lambda = 0$, $\boldsymbol\Sigma = 0$): diabatic S1 population stays 1 | fast | 7.0e-13 | 1e-12 |
| Halving $\Delta t$ (0.5 → 0.25) | slow | 1.23e-7 | 1e-4 |
| Doubling the grid, more points ($\mathrm dQ$ halved) | slow | 4.11e-5 | 1e-4 |
| Doubling the grid, wider box (same $\mathrm dQ$) | slow | 4.95e-11 | 1e-4 |

Edge density: 4.0e-11 over 2000 a.u., 4.8e-9 over 4000 a.u. (guard 1e-8; a longer run would need a wider $Q_7$ box). **FSSH against exact (report)**: `hopping = "sharc"`, EDC `sharc` ($C = 0.1$), `parallel_vel`, keep, `nacdr`, `ngt`, $\Delta t = 0.5$ a.u., 25 substeps, n = 1000 Wigner samples (seed 20261108), initial MCH S1, 8000 steps (4000 a.u.), recorded every 20 a.u., 8 workers. Wall time: **exact 34.6 s, FSSH 1216.9 s (20.3 min)**, total 1255 s. FSSH: 0 energy-drift trajectories, 2889 hops, 11 frustrated. Exact over 4000 a.u.: norm error 1.5e-12, energy error 1.02e-8 $E_h$, $E = 0.154447\,E_h$. Spin-summed MCH populations (exact / quantum / classical):

| $t$ (a.u.) | S0 | S1 | T1 |
|---|---|---|---|
| 0 | 0.0004 / 0.0000 / 0.0000 | 0.9996 / 1.0000 / 0.9998 | 0.0000 / 0.0000 / 0.0002 |
| 600 | 0.0004 / 0.0000 / 0.0000 | 0.9971 / 0.9977 / 0.9430 | 0.0025 / 0.0023 / 0.0570 |
| 1000 | 0.0005 / 0.0001 / 0.0000 | 0.9893 / 0.9913 / 0.9848 | 0.0101 / 0.0086 / 0.0152 |
| 1400 | 0.0444 / 0.0364 / 0.0320 | 0.9457 / 0.9551 / 0.9570 | 0.0099 / 0.0085 / 0.0110 |
| 2000 | 0.1370 / 0.1008 / 0.0950 | 0.8531 / 0.8912 / 0.8988 | 0.0099 / 0.0081 / 0.0062 |
| 3000 | 0.1369 / 0.0960 / 0.0960 | 0.8449 / 0.8825 / 0.8849 | 0.0182 / 0.0215 / 0.0191 |
| 4000 | 0.1374 / 0.0964 / 0.0960 | 0.8264 / 0.8754 / 0.8659 | 0.0362 / 0.0282 / 0.0381 |

Maximum $|\text{FSSH} - \text{exact}|$ over all times: quantum populations S0 0.0409 (at 3920 a.u.), S1 0.0489 (4000), T1 0.0080 (4000), overall **0.0489**; classical S0 0.0436 (1880), S1 0.0713 (640), T1 0.0717 (640), overall **0.0717**. Reading: the S1 → T1 (spin–orbit) transfer is reproduced closely by the quantum populations (at most 0.008); the S1 → S0 transfer at the conical intersection is under-estimated by FSSH (0.096 against 0.137, a stable 0.04 after 2000 a.u.); the classical T1 population has transient spikes (up to 0.072 above exact, at 600, 2400 and 3600 a.u.) while trajectories pass the S1/T1 crossing at $Q_7 = -3.3$, because $|U_{\alpha\lambda}|^2$ splits the active diagonal state between S1 and T1 there. At $t=0$ the exact packet (diabatic S1) has 3.6e-4 on MCH S0, FSSH starts in pure MCH S1 as specified.

**(d) Against SHARC 4.1 as a black box — single points** (asserted, `pytest -m sharc tests/validation/test_sharc_lvc.py`, 12 tests, about 20 s, 24 calls of `SHARC_LVC.py`, SHARC version string 4.0 dated 01.04.2025). Clean room: SHARC was only executed, from the public manual's calling convention (`$SHARC/SHARC_LVC.py QM.in`, with `QM.in`, `LVC.template` and `V0.txt` in the directory; first a step 0, then a step 1 with `OVERLAP` and the same `savedir`); nothing in the installation was read. Same fixtures (`lvc_singlets`, `lvc_soc`), 6 random geometries per fixture ($Q\sim N(0,1)$, seed 5) and their 6 displaced partners ($Q + 0.05\,d$), 12 points (the plan asked for 6; chosen before any number was seen). **Tolerance, fixed from SHARC's printed precision before any difference was computed:** every block of `QM.out` is printed in `%.12E` (13 significant digits), tol = max($10^{-9}$, 10 × the last printed digit of the largest value) = $10^{-9}$ for every block and point (all printed magnitudes are below 100). Compared up to the sign matrix $\mathbf D = \mathrm{diag}(\pm1)$ per spin-free state (the same for all $M_s$), fitted per point:

| Quantity | Maximum difference | Tolerance |
|---|---|---|
| diagonal of $\mathbf H$ (energies) and $\mathbf H$ (with SOC), absolute | 7.0e-13 $E_h$ | 1e-9 |
| gradients of every component, absolute, no sign | 5.7e-12 $E_h$/bohr | 1e-9 |
| NACs $D_\alpha D_\beta\mathbf K_{\alpha\beta}$, relative to max$|\mathbf K|$ (max$|\mathbf K|$ 0.8–3.3 1/bohr) | 2.5e-11 | 1e-9 |
| overlaps $\mathbf D\mathbf S\mathbf D$ (6 pairs) | 1.5e-13 | 1e-9 |

The margins are at least 40 times (NACs), 175 times (gradients). **Measured conventions:** SHARC's gradient and NAC labels are exactly our canonical order (asserted at every point); the elements of $\mathbf H$ match $\mathbf D\mathbf H\mathbf D$ and **not** its complex conjugate (the conjugate differs by 5.9e-4 $E_h$ for `lvc_soc`); the vectors between different multiplicities or $M_s$ components are printed and are exactly 0 (they are printed as a full 8 × 8 set of pairs); $S[\text{row}][\text{col}] = \langle\psi_{\text{row}}(t)|\psi_{\text{col}}(t+\Delta t)\rangle$ as ours (the transposed form misses by 4.8e-3 to 2.0e-2); $\mathbf D$ is the same at every point of both fixtures (S0+, S1−, S2+ for `lvc_singlets`; S0+, S1−, T1+, T2− for `lvc_soc`; a gauge difference to our column-sign rule, which makes the largest component positive). The differences are larger than the print rounding (about 5e-14) but far inside the tolerance; the relative size of 1e-11 to 4e-11 for gradients and NACs looks like a slightly different amu → $m_e$ constant inside SHARC (a hypothesis, not measured at this point). **Two SHARC findings:** (1) **`SHARC_LVC.py` (4.0) crashes with spin–orbit coupling when the template has dipole (`DMX`/`DMY`/`DMZ`) blocks**, even without a `DM` request (a casting error in its dipole `einsum`). The harness gives SHARC the same template **without the dipole blocks** (`without_dipoles`); **dipoles are therefore not compared with spin–orbit coupling** (our dynamics does not use them). (2) Coordinates written without a decimal point are rejected by SHARC (`line is not xyz`); the harness writes `%.16e`.

**(e) Against SHARC 4.1 as a black box — single trajectories** (asserted, markers `sharc` and `slow`, `pytest -m sharc tests/validation/test_sharc_lvc.py`, about 8 min for the 11 original tests; `sharc.x` with `QM/runQM.sh` calling `SHARC_LVC.py` on the template without the dipole blocks; input from the public manual §4.1–§4.3, `output.dat` from §5.3.1; no keyword was rejected). Common set-up: `lvc_soc`, one Wigner sample (seed 1), start `state 2 mch`, `coeff auto`, $\Delta t = 0.5$ a.u. (`stepsize` 0.5 × 0.0241888… fs), 25 substeps, `surf diagonal`, `gradcorrect ngt`, `ekincorrect parallel_vel`; our side with `substep_interpolation = "endpoint"` (the scheme of the manual, §8.11) and `edc_convention = "sharc"`.

*Initial state.* SHARC maps `state 2 mch` (our S1) to $\mathbf c^{\text{diag}} = \mathbf U^\dagger\mathbf c^{\text{MCH}}$ and the active diagonal state $\arg\max|c^{\text{diag}}_a|^2$, exactly as §8.15.3 (asserted: SHARC diagonal state 5 / MCH 2 (1-based), ours $\lambda$ = 4 / `mch_state` 1 (0-based); $|c^{\text{diag}}|^2$ identical, 0.996117 on the active state; populations at step 0 agree to 5.5e-12).

*Hopping off, 200 steps, both `nacdr` and `overlap`:*

| Quantity | Rule (plan) | `nacdr` | `overlap` | Status |
|---|---|---|---|---|
| $\max|\Delta\mathbf q|$ over steps 0–200 | ≤ 1e-8 bohr | 8.45e-10 | 8.45e-10 | asserted |
| $\max|\Delta\mathbf v|$ | ≤ 1e-10 a.u. | 2.32e-11 | 2.32e-11 | asserted |
| spin-summed MCH quantum populations, steps 0–50 (steps 0–200) | ≤ 1e-6 | 5.47e-12 (1.47e-10) | 3.12e-11 (9.47e-11) | asserted for 0–50; 0–200 **(report)** |
| $\mathbf D$ from $\mathbf H$ constant over the run | — | S0+ S1− T1+ T2− | same | asserted |
| $E^{\text{diag}}$ **along our own path** against the eigenvalues of SHARC's $\mathbf H$ | ≤ 1e-9 | **2.14e-8** | **2.14e-8** | **failed as specified; (report)** |
| $E^{\text{diag}}$ of our model **at SHARC's interface geometry** | ≤ 1e-9 | 3.20e-10 | 3.20e-10 | asserted (post hoc form) |
| our $\mathbf U$, brief's metric $\max|(\mathbf D\mathbf U_{\text{ours}})\boldsymbol\Phi - \mathbf U_{\text{SHARC}}|$, all columns | ≤ 1e-7 | **5.23e-5** | **5.23e-5** | **failed as specified; (report)** |
| (i) the same in the **separated columns** (0 and 4: no neighbour within $10^{-4}\,E_h$), best $\boldsymbol\Phi$ per step | ≤ 1e-7 | 1.39e-9 | 1.39e-9 | asserted (post hoc form) |
| (ii) **projectors** $\|\mathbf P_{\text{ours}} - \mathbf P_{\text{SHARC}}\|_{\max}$ on the cluster [1,2,3] (T1 group) / [5,6,7] (T2 group) | ≤ 1e-7 | 8.15e-11 / 1.38e-9 | same | asserted (post hoc form) |
| (iii) separated columns with $\boldsymbol\Phi$ fixed at step 0 (nacdr: our tracking against SHARC's) | ≤ 1e-7 | 1.39e-9 | **1.35e-5** | `nacdr` asserted; `overlap` **(report)** |
| SHARC's $\mathbf U$ against the eigenvectors of SHARC's own printed $\mathbf H$ (ordering convention only) | ≤ 1e-7 | 1.19e-8 | 1.19e-8 | asserted (does not test our $\mathbf U$) |

Nuclear motion is the same for both couplings (without hops it does not depend on the electronic propagation); $\mathrm dq$ grows as 4.8e-12 at step 10, 1.8e-10 at step 100, 8.4e-10 at step 200. **How the two failing comparisons were replaced, stated plainly.** The plan asked for $E^{\text{diag}}$ along our own path at 1e-9 and for the column metric on all columns at 1e-7; both **failed when first run** (2.14e-8 and 5.23e-5), and the assertions in the test file are different ones, **chosen after the numbers were seen**:

- *$E^{\text{diag}}$.* Diagnosis (measured): `sharc.x` gives its interface the geometry in ångström **rounded to 7 decimals** (the interface geometry is `round(q_SHARC · a0[Å], 7)` to 1.1e-16 Å, up to 9.1e-8 bohr from sharc.x's own positions, which it propagates unrounded), so the like-for-like point is our model evaluated at that rounded geometry, where the difference is 3.2e-10 (diagonal only; off-diagonal elements agree to 6.7e-13) against the unchanged tolerance 1e-9. The remaining 3.2e-10 looks like a slightly different Å → bohr constant in `SHARC_LVC.py` (with $a_0 = 0.5291772105$ instead of the CODATA 2018 value 0.529177210903 the residual falls to 3.9e-11; not pinned down). The positions agree to 8.5e-10 bohr, which bounds the error of the replacement. **$E^{\text{diag}}$ along our own path is reported, not asserted.**
- *$\mathbf U$.* `lvc_soc` has two diagonal states only 1.74e-8 $E_h$ apart at step 0 (two T2 components) and a T1 group with gaps of 1.0e-6 and 1.6e-5 $E_h$; moreover `sharc.x` adds **+1e-12 $E_h$ ($M_s = 0$) and +2e-12 $E_h$ ($M_s = +1$)** to the triplet diagonal of $\mathbf H$ (SHARC_LVC's `QM.out` has the three components exactly equal; `output.dat` does not). Divided by the 1.7e-8 gap these give 6e-5 to 1.2e-4 of mixing inside the pair, and the 3.2e-10 diagonal residual over the 1e-6 and 1.6e-5 gaps gives about 2.5e-7 in the T1 columns: $\mathbf U$ is not determined to 1e-7 in those columns by an $\mathbf H$ known to 1e-12 (the per-column maxima of the brief's metric are 4.8e-10, 8.7e-8 for the separated columns 0 and 4, up to 5.0e-5 and 5.2e-5 for the pair of columns 5 and 6). The first replacement (SHARC's $\mathbf U$ against the eigenvectors of SHARC's own $\mathbf H$, 1.19e-8) turned out, in the task review, **not to test our $\mathbf U$ at all**; it is kept, relabelled as the ordering check. The final form (after a second post-hoc ruling; thresholds fixed before (ii) was computed): our model's $\mathbf H$ at the interface geometry is tracked by our own `diagonal_basis` exactly as the core tracks it (first-step rule, then the previous $\mathbf U$) and compared with SHARC's $\mathbf U$ by (i), (ii) and (iii) above, all at the original 1e-7. The brief's metric for all columns is still printed per step (5.23e-5 at step 200). Interpretation: the 5e-5 is a rotation **inside** the near-degenerate clusters, which no comparison to this accuracy can resolve; outside them, and as projectors on them, our $\mathbf U$ matches SHARC's to about 1e-9. The populations, which are $\mathbf U$-invariant, agree to 1e-10. The diagnosis of the 5e-5 (the two effects above, summing to its size) rests on probes of the implementer, not on a test that adds SHARC's offsets to our $\mathbf H$; that confirmation was not done.
- *Phase drift with `overlap` (report).* With local diabatization the best per-step phase between our basis of SHARC's $\mathbf H$ and SHARC's $\mathbf U$ drifts by 1.35e-5 over 200 steps (linearly: 5.8e-7 at step 10, 6.8e-6 at step 100), against 2.3e-9 with `nacdr`: SHARC seems to track the phases of $\mathbf U$ differently under local diabatization. Not asserted, not investigated further.

*Replayed hop (asserted, one test, about 4 min, 300 SHARC steps).* The geometry was chosen by a search with our core and `sharc.x` driven by **our own `QM.out` provider** (the swap test below), then confirmed with the real `SHARC_LVC.py`: for Wigner seeds 1–100, the probability of at least one hop within 300 steps along the no-hop path (median 2.6e-4; seed 59 and seed 42 0.9955); seed 59 with `rngseed 1` hops at step 67 (the first `rngseed` tried; no other was tried). SHARC's own random numbers (from `output.dat`) are fed to a `ReplaySource` of our core:

| Quantity | Rule | Measured |
|---|---|---|
| SHARC hops (step, from → to; 0-based diagonal states) | — | [(67, 7 → 4)] |
| Our hops with SHARC's random numbers | same | [(67, 7 → 4)] |
| Active diagonal state, every step 0–300 | identical | identical |
| $\max|\Delta p|$ (hop probabilities, steps 1–300) | ≤ 1e-5 | 8.43e-7 |
| $\max|\Delta\mathbf v|$, every step including the rescaled one | ≤ 1e-9 a.u. | 2.95e-10 (2.4e-10 at step 67) |
| Frustrated hops (ours) | — | 0 |
| Positions (**(report)**, not asserted) | — | $\max|\Delta\mathbf q|$ 3.2e-8 bohr over 300 steps, 4.2e-9 over the first 100 |

**`eselect` (report, `tools/sharc_compare/lvc_ensembles.py`, 14 min).** The trajectory of the replayed hop (seed 59, `rngseed 1`, 300 steps, `nacdr`, no decoherence), SHARC with `grad_select`, `nac_select`, `eselect 0.3` (eV) against ours with `eselect_Eh = 0.3/27.211…` and SHARC's random numbers. SHARC asks its interface for exactly the states our mask keeps (`GRAD 2 4 6 8`, `NACDR SELECT` over S1 and the three T2 components). Result: the hop (67, 7 → 4) and the active sequence agree; but $\max|\Delta\mathbf q|$ over the first 100 steps is 2.5e-6 bohr (2.5e-5 over 300), $\max|\Delta\mathbf v|$ 3.0e-7, $\max|\Delta p|$ 1.4e-6 — **and SHARC's selected run is bit-identical to its unselected run** (final positions differ by 0.0; `select_directly` 204 s, `noselect_directly` 420 s, unselected 205 s), because `SHARC_LVC.py` returns every gradient (S0: max$|\mathbf g|$ 0.0353) and every coupling (S0–S1: max$|\mathbf K|$ 0.175) regardless. Our masking (§8.15.6) models a backend that does not compute them, so our selected run differs from our unselected run. **The `eselect` comparison is therefore not like-for-like and is not asserted.**

**Ensemble populations against SHARC (report, a swap test, `tools/sharc_compare/lvc_ensembles.py`).** *This is not a comparison with `SHARC_LVC.py`:* `sharc.x` was driven by **our own `QM.out` provider** (our `LVCModel` behind `QM/runQM.sh`; strategy 5.4 (c), SHARC's driver and our electronic structure), because `SHARC_LVC.py` costs 0.68 s per step alone (0.81 s with the detailed output) and 2.05 s per step for a batch of 14 parallel trajectories, so 4000 steps × n = 100 would take **about 16 hours** (n = 200 33 h, one at a time 76 h). So it tests the two **drivers** (hopping, EDC, velocity adjustment in the diagonal basis, with independent random streams), not the electronic structure of `SHARC_LVC.py` (covered by (d) and (e)). Settings: n = 200 shared Wigner initial conditions (the `model-initial` stream, seed 20261110), start in S1, EDC `sharc` ($C = 0.1$), `nacdr`, `parallel_vel`, keep, `ngt`, $\Delta t = 0.5$ a.u., 25 substeps, `endpoint` on our side, 4000 steps (2000 a.u.); SHARC's random seeds from `SeedSequence([seed, crc32(b"sharc-rngseed")])`. No trajectory drifted on either side (SHARC's largest $|E_{\text{tot}} - E_{\text{tot}}(0)|$ 3e-5 eV); our side made 149 hops (2 frustrated), SHARC at least 144 visible active-state changes (a lower bound). Run time: the whole script 57 min (the `eselect` part 14 min), ours 154 s on 8 workers, SHARC with our provider 2442 s on 12 threads (limited by the single Python process serving all of them, and partly concurrent with another test run). Spin-summed MCH classical populations, ours / SHARC (pooled two-proportion $z$):

| $t$ (a.u.) | S1 | T1 |
|---|---|---|
| 200 | 0.977 / 0.975 (+0.14) | 0.001 / 0.001 (+0.00) |
| 400 | 0.993 / 0.993 (+0.01) | 0.002 / 0.002 (+0.00) |
| 600 | 0.996 / 0.996 (−0.01) | 0.004 / 0.004 (+0.00) |
| 800 | 0.994 / 0.994 (−0.01) | 0.006 / 0.006 (+0.00) |
| 1000 | 0.992 / 0.992 (+0.01) | 0.008 / 0.008 (−0.01) |
| 1200 | 0.985 / 0.984 (+0.06) | 0.015 / 0.016 (−0.06) |
| 1400 | 0.920 / 0.926 (−0.23) | 0.080 / 0.074 (+0.23) |
| 1600 | 0.826 / 0.812 (+0.36) | 0.174 / 0.188 (−0.36) |
| 1800 | 0.850 / 0.848 (+0.06) | 0.150 / 0.152 (−0.06) |
| 2000 | 0.901 / 0.909 (−0.27) | 0.099 / 0.090 (+0.27) |

S0 is 0.000 on both sides at every time, and T2 is 0.022 / 0.024 at 200 a.u., 0.005 / 0.005 at 400 a.u. and 0.000 afterwards. The largest $|z|$ over all states and times is **0.36**. The quantum populations are in the JSON. The comparison used the same Wigner conditions, so it is sensitive to systematic differences of the drivers, not to the sampling.

**Black-box facts measured on the way** (not asserted): `sharc.x` writes `QM.in` in ångström with 7 decimals; SHARC prints 21874.66184217 $m_e$ for 12 amu (1822.8884868 $m_e$/amu against our 1822.888486209, relative difference 3.5e-10; its effect on positions is below 1e-12 bohr here); `SHARC_LVC.py` run in angstrom is 3.2e-10 $E_h$ off our $\mathbf H$ on the diagonal; `sharc.x` with our provider costs about 8 ms per step. These rest on the implementer's probes and on the passing interface-geometry assertions that depend on them.

**Run times and counts** (measured; the commit at which each was measured is stated; **final verification of the finished branch, 2026-11-11 at `19b43b6` with SHARC and ORCA: `pytest -W error` in one run, 1097 passed, 2 skipped, 1 xfailed in 2 h 33 min**; both skips are by design (the `overlap` phase-drift case, §8.15.9(e), and the published-values skip, §8.14.1(b)), the xfail is the 3000 K `wigner.py` case, §8.8; the fast tier at `19b43b6` is 922 passed, 20 skipped). Fast tier, earlier on the branch, `pytest -m "not slow and not pyscf" -W error`: **911 passed, 20 skipped** in 10 min 3 s (at `e00b6d8`, after Task 10; 931 collected, unchanged at `ca2597d`; the skips are the live-SHARC tests without `SHARC_HOME`, 8 from P7a-1 and 12 from the single-point tests). Slow tests of P7a-2, run alone: `test_tully_overlap_exact.py` 4 passed in 677.5 s; `test_wavepacket2d.py` (5 fast, 3 slow) 8 passed in 235.6 s. `pytest -m sharc` (live SHARC): **39 passed, 1 xfailed in 12 min 4 s at `e00b6d8`** (the xfail is the documented 3000 K `wigner.py` case); the fix round of the $\mathbf U$ assertions (`ca2597d`) then renamed one test function and added three, each parametrised over `nacdr` and `overlap` (40 tests collected with `-m sharc` at `e00b6d8`, 46 at `ca2597d`: the 2 renamed items plus 6 new ones; for (iii) with `overlap` the test is skipped by design, §8.15.9(e)): that file's targeted run gave 15 passed, 1 skipped in 4 min 35 s; 46 tests are collected with `-m sharc` at `ca2597d` (collected, not run as a whole). Collected over everything at `ca2597d`: 1089 tests, 62 of them `slow`. Report scripts (by hand): `tools/validation/lvc_exact_report.py` 1255 s, `tools/sharc_compare/lvc_ensembles.py` 57 min. A known intermittent live-SHARC test unrelated to P7a-2 is listed in §10.

### 8.16 The `dynamics` stage with PySCF (P7b)
**[IMPLEMENTED]** (P7b: the packages `gqteashmd.trajectory` and `gqteashmd.stages.dynamics`; singlets only; **[PLANNED]** triplets and spin–orbit coupling from PySCF, O7). The stage runs one trajectory per entry of `trajectories.json` with the dynamics core of §8.10–§8.15, **called unchanged** (the dependency rule of §8.14 holds: the core imports nothing from `electronic/`, `stages/` or `config/`, and it is bit-identical to P7a-2 on the analytic models, guarded by the golden test). What the stage adds is the connection to PySCF, the handling of failed steps, the files and the restart. Units are those of §8.10; fs and eV appear only in the input and in the reports.

#### 8.16.1 What a trajectory task does
1. It reads its entry of `trajectories.json` (§8.9.3): the sample index, the initial state (`S1` is state index 1 with the amplitudes $c = e_1$) and the random stream `task_rng(project.seed, "dynamics", k)`. It checks that the sample exists in `samples.h5`, that the atoms of `samples.h5` are those of the prepared molecule and that the recorded stream is the stage's own; otherwise it fails with a non-retryable error that names the `--rerun-from` stage (`sampling` or `selection`).
2. Positions and velocities are the sample's `positions` (bohr) and `velocities` (bohr per atomic unit of time) of `samples.h5`; the masses are the isotope masses of `samples.h5` (amu, converted to $m_e$, repeated for the three Cartesian components).
3. One PySCF backend is built **per trajectory** with the full `[electronic]` settings (`electronic.log` and `scratch/` in the task directory), and `reset()` is called before step 0, so a trajectory never depends on another one or on a previous attempt. With `casscf`, its reference geometry for the start rule (§8.6.2) is the equilibrium geometry of `samples.h5`, so step 0 starts from the reference solution (rung `"reference orbitals"`).
4. Every nuclear (sub)step is one call of the core's step (§8.12; the step order is not changed). The runner draws no random number itself: the core draws exactly one per call, and the number of a rejected attempt is returned (§8.16.3), so the stream depends only on the accepted (sub)steps.

#### 8.16.2 The electronic model
For each geometry $\mathbf q$ the adapter (`BackendModel`) asks the backend for the energies, **the gradient of the active state only**, the phased overlaps with the previous geometry (always) and, for `coupling = "nacdr"`, all NACs $\mathbf d_{IJ}$ with $I<J$. It hands the core a snapshot with:
- $\mathbf H = \operatorname{diag}(E_I)$, so $\mathbf U$ stays `None` and the core takes its MCH fast path (singlets only);
- the gradient rows of the **non-active states set to `NaN`** as a tripwire: if the core ever read one of them the energy would become `NaN` and the run would stop; a test asserts that no `NaN` reaches a record over trajectories with hops. This is why one gradient per step is enough: the core reads the gradient of the *old* active state at $\mathbf q'$ (velocity completion, step 5 of §8.12) and that of the *new* active state only at the start of the next step. After a hop the adapter therefore calls `compute_more` at the same geometry for the new active state's gradient;
- the NACs (antisymmetrised) for `nacdr`, zeros for `overlap`;
- the overlaps **Löwdin-orthonormalised** (the propagator of §8.15.5 assumes a unitary $\mathbf S$ and does not check it). The raw smallest singular value of the overlap matrix is kept for the rejection test and the log.

A non-finite energy, a non-finite or missing gradient of the active state, non-finite overlaps or, with `coupling = "nacdr"`, a non-finite NAC are raised as a non-retryable electronic-structure error and treated like any failed calculation (§8.16.3): they halve the step and, if persistent, end the trajectory as `electronic_failure`.

**Other errors inside a trajectory.** A restart or step log that cannot be used (an unreadable `restart/current`, a damaged generation, a `steps.log` shorter than the restart expects) fails the task, not retried, with *trajectory traj_…: cannot resume from the restart in …; the restart or step log is unusable, rerun with --rerun-from dynamics*. Any other error raised inside the dynamics (for example a numerical error of the propagator) fails only that task, not retried, with *trajectory traj_…: (the error message)* and **no** rerun advice: the other trajectories and their restarts are not affected, and a full `--rerun-from dynamics` is not the remedy.

#### 8.16.3 Step rejection
**Signals** (checked after each step at the new geometry):
1. the electronic calculation raised a non-retryable error (a failed SCF/CASSCF/TD ladder, non-finite data as above);
2. the smallest singular value of the raw state-overlap matrix is below `overlap_threshold` (default 0.5): the states cannot be matched reliably;
3. **[IMPLEMENTED, P9a plan 2]** the total energy jumped: $\lvert E_{\text{tot}} - E_{\text{tot}}(\text{previous record})\rvert$ > `energy_jump_limit_eV` (default 0.1 eV), where the previous record is the last accepted (sub)step (at the first step, $t = 0$). Velocity Verlet changes $E_{\text{tot}}$ by far less per step in a correct run (the largest per-step jump in the PySCF validation runs was $7.7\times10^{-3}$ eV, §8.13.6), so a larger jump means that the electronic solution changed;
4. CASSCF left its first ladder rung (`casscf` meta other than `"previous orbitals"`, at a geometry that has a previous one; since the start rule a fall to `"reference orbitals"` counts too, like one to the SCF orbitals), i.e. the solution may have jumped. **Only CASSCF** counts: TD and SCF rung changes are recorded in the log but are not signals;
5. the gradient refresh after a hop failed (`compute_more`; the step is rejected, and the hop with it).

They are tested in this order and the first that applies is the step's signal; it is counted in `rejections_by_signal` of the result and of `dynamics_summary.json` (§6) as `electronic` (1 and 5), `overlap` (2), `energy` (3) or `rung` (4).

A *retryable* error (out of memory, an I/O error inside PySCF) is **not** a rejection: it ends the task, the orchestrator retries it (up to `execution.max_attempts`) and the retry resumes from the restart.

**Procedure.** A restart generation is written after every accepted (sub)step (§8.16.6), so the current generation always holds the state at the time $t$ at which the step started. A rejected step is discarded; the backend snapshot of time $t$ and the random-number state are restored; and the interval $[t, t+\Delta t]$ is redone as **two steps of $\Delta t/2$**, each of which can be halved again, down to $\Delta t/2^{k}$ with $k$ = `max_step_halvings` (default 3, so $\Delta t/8$). After the interval the trajectory returns to the full $\Delta t$, so it stays on the grid of `dt_fs` and the records of different trajectories line up. Time is kept on an **integer grid** (`grid_step` full steps, `ticks` in units of $\Delta t/2^{k}$), never decided from floating-point sums; this is also why `tmax_fs` must be an integer multiple of `dt_fs`. The substeps use $\Delta t/2^{j}$ with `nsubsteps` per (sub)step.

**At the smallest step:** a CASSCF rung change with a good overlap **and no energy jump** is accepted and **flagged** (`rung_changed` in the log); an overlap still below the threshold ends the trajectory with `overlap_failure`; an energy jump still above `energy_jump_limit_eV` ends it with `electronic_discontinuity` (also when the rung changed in the same step, because the energy test comes first); a failed calculation ends it with `electronic_failure`. A failure at step 0, where there is nothing to roll back to, ends the trajectory as `electronic_failure` at $t = 0$ and writes no restart (a rerun of the task recomputes step 0). `n_rejections` counts every rejected attempt, including the last one that ends a trajectory. Every rejection that is followed by an accepted step is in the `rejections` column of that step's log record, and every flag is in the log; **the rejections that end a trajectory appear in no log record** (no step is accepted after them), so `output.h5` does not show them: they are counted in `n_rejections` of the result and `dynamics_summary.json`, and the outcome message says at which time and step size the trajectory ended.

**Why the energy signal exists (P9a plan 2): a CASSCF solution change can pass the other signals.** Measured in the validation of §8.16.8 C (ethylene SA2-CASSCF(2,2)/6-31G\*, step 24 at 12 fs): in one step E0 falls by 25 mEh and E1 rises by 37 mEh, the smallest singular value is 0.686 (above the threshold 0.5), and CASSCF stays on its first rung. Before P9a plan 2 **no step was rejected** there and only the cumulative energy-drift limit stopped the run, because the jump of $E_{\text{tot}}$ (0.976 eV) exceeded `energy_drift_limit_eV = 0.5`; a smaller jump, below the remaining drift margin, was accepted silently. The P9a diagnosis found the same kind of switch (0.6-0.9 eV) in 7 of the 20 M4 trajectories (§5.10). **With the energy signal** the same run rejects the jumping interval four times and ends at 11.8125 fs ($\Delta t/8$) as `electronic_discontinuity`, with the message `|E_tot - E_tot(previous record)| = 0.9431 eV exceeds energy_jump_limit_eV = 0.1` (asserted, §8.16.8 C). **What it still does not detect:** a solution change with a jump below `energy_jump_limit_eV` and a singular value above `overlap_threshold`. Read `min_singular_value` and the per-step `e_total` in `output.h5` (§6) if in doubt. The `analysis` stage still flags trajectories whose $E_{\text{tot}}$ jumps by more than `energy_jump_eV` (default 0.1 eV) between consecutive records (§8.13.6); in the acceptance run of §5.10 (both thresholds at their default, `output_every = 1`) that flag was empty; it can still fire with `output_every > 1` (kept records several steps apart), when the two thresholds differ, or across a halved interval (`output.h5` keeps only full steps, hops, flags, records after a rejection and the last record, so two consecutive kept records can be several substeps apart, each allowed up to `energy_jump_limit_eV`). To get the behaviour before P9a plan 2, set `energy_jump_limit_eV` to a large value (e.g. `1.0e6`).

#### 8.16.4 Outcomes
A task is **done for every outcome**; the outcome is in its result and in `dynamics_summary.json`. Only infrastructure failures (a crash, unreadable `samples.h5` or `trajectories.json`, a retryable error that exhausts its attempts) are failed tasks and count against `max_failed_fraction`.

| Outcome | When | Final? |
|---|---|---|
| `finished` | the time reached `tmax_fs` | no: raising `tmax_fs` continues it (§8.16.6) |
| `energy_drift` | $\lvert E_{\text{tot}} - E_{\text{tot}}(0)\rvert$ exceeds `energy_drift_limit_eV` after an accepted step (or is not finite) | yes |
| `stopped_in_s0` | `stop_after_s0_fs` is set and the active state was S0 for that long without interruption | yes |
| `overlap_failure` | a step was rejected at the smallest step because of the overlap | yes |
| `electronic_failure` | a step was rejected at the smallest step because the calculation failed, or the calculation failed at step 0 | yes |
| `electronic_discontinuity` | **[IMPLEMENTED, P9a plan 2]** a step was rejected at the smallest step because $E_{\text{tot}}$ jumped by more than `energy_jump_limit_eV` (§8.16.3); the `analysis` stage excludes it like the other early endings (§8.13.1) | yes |

**Guidance: compute one or two states above the highest initial state.** A state that is not among the computed ones cannot be tracked. In the example of §5.8 (`singlets = 3`) the trajectory that starts in S2 ends with `overlap_failure` at 0.094 fs, because S2 crosses an S3 state of the other irreducible representation of $C_s$ that is not computed: the overlap is zero by symmetry (the smallest singular value is $5.2\times10^{-12}$ at the probe point of §8.16.8 B and $7.3\times10^{-13}$ at $\Delta t/8$ in the run of §5.8) and no step size helps. That is the designed outcome for an intruder state, and the remedy is `singlets = 4`. In SA-CASSCF, `overlap_failure` can also mean a lost active space: the tracked solution of a metastable state-averaged problem disappears (§8.16.8 B).

#### 8.16.5 Output
`steps.log` records every accepted (sub)step as a binary record. `output.h5` (§6) is built from it **when the task ends, with every outcome**, and is thinned by `output_every` (every hop, frustrated hop, flag and preceding rejection and the last record are always kept). There is **no live `output.h5`**: an HDF5 file that is killed during a write can become unreadable, which is not acceptable for runs of days. To look at a running trajectory read `progress.json` (step, time, active state). `finalize` does not build missing `output.h5` files; it lists them in `dynamics_summary.json` and warns.

#### 8.16.6 Restart, resume and extension
- **Generations.** After every accepted (sub)step the task writes `restart/gen_<g>/`: `state.npz` (the core restart format 2 of §8.15.8: $\mathbf q$, $\mathbf v$, $c$, the snapshot, $\mathbf U$, time, step, active state, $E_{\text{tot}}(0)$, random-number state), `electronic.npz` (the backend snapshot: orbitals and the phase-aligned determinant expansion, §8.6.8) and `meta.json` (counters, the number of records of `steps.log` that belong to it and the position inside a halved interval: its end and the current depth). A generation is written to a temporary directory and renamed; `restart/current` is written atomically and names the newest complete generation; the older generation is removed only after `current` points to the new one. A kill at any instant leaves one complete generation.
- **Resume.** The task reads `current`, loads the core state and the backend snapshot, **truncates `steps.log` to the length in `meta.json`** (a record written after the last restart is dropped and later written again) and continues; a resume inside a halved interval finishes it with the same step size before returning to the full $\Delta t$. The core restart stores neither the model nor the options (§8.14); the **stage fingerprint** covers them, so a resumed trajectory always continues with the same input. **[IMPLEMENTED, P9a plan 2]** Every resume is recorded: the generation name is added to `resumes` in the task result (`n_resumes` in `dynamics_summary.json`) and the stage logs `Stage dynamics: traj_… resumed from restart generation(s) …` (§6). A resume is stored with the next generation the task writes, so a resume that is killed before it writes one leaves no record.
- **Stop and kill.** The worker has no signal handler; the executor's SIGTERM ends it and every file is written atomically, so at most the step in progress is lost.
- **Reproducibility.** With the analytic fake backend a resume is **bit-identical** (asserted). **With PySCF a resumed trajectory is not bit-identical** to an uninterrupted one: the iterative solvers restart from a saved snapshot. The difference measured (asserted bound, §8.16.8 B, test (c)) is round-off: $1.3\times10^{-13}$ bohr in position, $2.1\times10^{-13}\,E_h$ in energy and $4.6\times10^{-15}$ in the populations after five resumed steps.
- **Durability.** The restart writes are safe against a **process kill** at any instant (atomic renames, `steps.log` flushed and fsynced per record), **not against a power loss or an operating-system crash**: the generation files and their directory are not fsynced, so after a power loss the newest generation can be incomplete or missing (not tested).
- **Extension (O16).** `dynamics.tmax_fs` is not hashed. A stage that is done with an unchanged fingerprint is normally skipped; one orchestrator hook (`Stage.reopen_tasks`) lets `dynamics` name the tasks to continue: those with outcome `finished` whose number of full steps is below the new `tmax_fs`, read from the recorded task results (not from files). The orchestrator marks the stage RUNNING **before** it resets any task (a kill during the reopening leaves a stage that the next run resumes, and the hook is applied again to a RUNNING stage, which reopens exactly what still needs work), runs those tasks, and finalizes again. The restarted trajectory continues from its last generation to the new end; $E_{\text{tot}}(0)$ stays the value at $t=0$. The hook is applied to every recorded state of the stage except skipped: done, and also running (a killed controller), pending (a user stop) and failed, so a stop followed by a raised `tmax_fs` ends, in one run, with every `finished` trajectory at the new end (asserted with the fake stages and with the canned LVC pipeline). Only stages that override the hook pay the cost of reading their results on a plain rerun. Lowering `tmax_fs` reopens no trajectory. **[IMPLEMENTED, P8]** Reopening does not change the fingerprint, so the orchestrator resets every later stage that depends on `dynamics` and has results **before** it reopens the trajectories (the analysis is then recomputed in the same run), and the `analysis` stage redoes itself when `round(tmax_fs/dt_fs)` changed without a reopened trajectory (§7.5).

#### 8.16.7 Not implemented and limits
- Singlets only: no triplets, spin–orbit coupling or $\mathbf U$; `eselect` is not accepted (§4.2).
- No live `output.h5` (§8.16.5), no thermostat, no A-FSSH, no per-trajectory time limit other than `tmax_fs`.
- PySCF DFT gradients omit the grid response (warning for `tda`/`tddft` with a DFT functional and `grid_level < 5`, §8.6.4); TDA/TDDFT describe the S1/S0 seam incorrectly, so use SA-CASSCF for S1 to S0 decay (§8.6.4).
- A state outside the computed set cannot be followed (§8.16.4); the TDA/TDDFT root fix of §8.6.3 is a heuristic.
- **A CASSCF solution change with a small energy jump can pass every rejection signal.** Before P9a plan 2 a jump of any size passed (§8.16.3: ethylene step 24, 0.976 eV, smallest singular value 0.686, rung unchanged); since then a jump above `energy_jump_limit_eV` (0.1 eV) is rejected and, if it persists, ends the trajectory as `electronic_discontinuity`. A change with a smaller jump and a singular value above `overlap_threshold` is still accepted; inspect `min_singular_value` and the per-step `e_total` in `output.h5`.
- **Losses after a hop in SA-CASSCF.** In the ethylene studies of §5.10 every lost SA3 trajectory ends after a hop (`overlap_failure` or `electronic_discontinuity`), mostly in the hot ground state; the seed-2026 replays trace the overlap failures to a solution switch (the seed-2027 losses were not replayed). The stage has no remedy for it: re-solving from the previous orbitals lands on the same other solution.
- **The extra CASSCF rungs** (§8.6.8) exist in the backend, but no stage uses them.
- **`electronic.log` grows without limit:** PySCF writes its output (verbosity 4) at every (sub)step for the whole trajectory, about 2.4 kB per step for water CIS/STO-3G (the example of §5.8) and more for CASSCF. Watch the disk for long runs.
- `wall_time_s` in a task's `result.json` covers the **last invocation only**: after a resume or an extension it does not include the time of the earlier invocations.
- SA-CASSCF(2,2)/6-31G\* on ethylene does not converge at `conv_tol_grad` $\le 10^{-7}$ along the test path of §8.16.8 C on our side (and SHARC's interface does not converge with `conv-tol` $10^{-12}$ / `conv-tol-grad` $10^{-8}$ either); do not tighten it below the default $10^{-6}$ without testing.
- The stage is as expensive as its electronic structure: one SCF, one CASSCF/TD solve, one overlap per (sub)step. A rejected step costs the failed calculation as well. The water example of §5.8 took 5 s for its two short trajectories (including process start-up); ethylene SA2-CASSCF(2,2)/6-31G\* at about 1.4 s per step (our runs of §8.16.8 C: 33 s for 24 steps with `overlap`, 64 s with `nacdr`, one core each).

#### 8.16.8 Validation (measured)
Every number is marked **asserted** (a test fails if the bound in brackets is broken), **report** (printed or written by a script that asserts nothing) or **post hoc** (the system or the assertion took its final form after numbers were seen; the original rule and its numbers are given). Run times are from the development machine, code on `/mnt/c`. The detailed reports are in the git-ignored ledger `.superpowers/sdd/2026-11-11-p7b-dynamics-stage/`.

**A. Fast tier, no PySCF (asserted; `gqteashmd.testing.fake_backend.ModelBackend` over the analytic LVC singlet model).** The adapter reproduces the direct model runs of the core **bit for bit** for both couplings (`assert_array_equal` on every snapshot field); the runner equals a manual loop over the core's step; a crash at every call number, a kill between the log write and the restart write, and a kill inside a generation write all resume **bit-identically**, also inside a halved interval; one low overlap halves exactly one interval and the rejected step's random number is returned; persistent signals end as `overlap_failure` (low overlap) and `electronic_failure` (failed or non-finite calculation); a rung change that persists at the smallest step is accepted and flagged; `energy_drift`, `stopped_in_s0`, failure at step 0, a retryable error that propagates and resumes, and extension are each exercised; a damaged restart is a restart error with the `--rerun-from dynamics` advice and any other error inside a trajectory is reported without it; a non-finite NAC under `nacdr` is an electronic failure; the orchestrator-level tests cover the stage's plan, fingerprint rules, the reopening (including a kill during it, and after a stop or a failure of the stage), the pause rule and a kill and resume with the fake backend as a subprocess.

**B. PySCF validation (`tests/validation/test_pyscf_dynamics.py` and `tests/test_e2e_dynamics.py`, marker `pyscf`; the drift-ratio test (b) and the kill test (f) also `slow`; tests (a)-(f) as numbered in the plan). Post hoc: both systems of the plan changed after the numbers were seen.**
- *Ethylene SA2-CASSCF(2,2)/STO-3G (plan) is metastable.* With the planned initial conditions (T = 300 K velocities, `default_rng(2026)`, S1) the tracked $\pi,\pi^*$ solution is only a local minimum of the state-averaged energy (a $\sigma,\pi^*$ solution 11.3 mEh lower exists at the starting geometry) and is lost at about 1.06 fs along the S1 path: the step halving rejects at $\Delta t$, $\Delta t/2$, $\Delta t/4$ and $\Delta t/8$ (smallest singular value 0.03 at every depth, because the jump does not shrink with the step), the trajectory ends `overlap_failure` (correct behaviour), and there is no valid 10-step run. The tests (a)-(c) were moved to **SA2-CASSCF(2,2)/6-31G\*** (the P3a validation basis), same geometry, velocities, $\Delta t$ and lengths; bounds unchanged.
- *(a), asserted:* 6-31G\*, `hopping = decoherence = "none"`, `coupling = "overlap"`, $\Delta t = 0.5$ fs, 20 steps (10 fs): finished, 21 records, no rejection, smallest singular value $\ge 0.9$ (measured 0.972993).
- *(a), drift bound, strict xfail (post hoc):* the planned bound $\lvert E_{\text{tot}} - E_{\text{tot}}(0)\rvert \le 10^{-4}\,E_h$ was fixed before any number and was too tight: measured $2.682\times10^{-4}\,E_h$ (an excursion at 4.5-5.5 fs). It stays as an expected failure carrying that number. At $\Delta t = 0.25$ fs the drift is $9.92\times10^{-5}\,E_h$ over 10 fs (report). Energy conservation is validated by the $\Delta t^2$ scaling of (b), not by an absolute bound.
- *(b), asserted, slow:* drift over 5 fs, $\Delta t = 0.5$ and $0.25$ fs: $2.682\times10^{-4}$ and $9.916\times10^{-5}\,E_h$, ratio **2.705**, window [2.5, 6.0] (fixed in the plan; a second-order integrator gives 4, so the window is wide). **The margin to the lower edge is thin** (2.705 against 2.5). A further ratio $\Delta t = 0.25/0.125$ fs was 4.07 (report, probe). **Since P9a plan 2** the finer pair is asserted too (slow; bounds [3, 5] fixed in the plan, after the probe had been seen): drift $9.916\times10^{-5}$ and $2.439\times10^{-5}\,E_h$ over 5 fs, ratio **4.065**; and the non-slow `pyscf` tier has a CASSCF energy-consistency guard: the (a) drift must stay $\le 10^{-3}\,E_h$ (measured $2.682\times10^{-4}$; the bound was set after that number, a factor 3.7 above it, as a regression guard, not a validation).
- *(c), asserted (resume with PySCF):* a 10-step reference run against 5 steps followed by a new backend, model and runner on the same directory for 5 more; the second call is given **decoy** initial conditions (velocities scaled by 1.5, active state S0, another seed) that only a real resume ignores, and the test asserts that it made **5 compute calls** (a silent fresh start would need 11). Bounds fixed before: $\lvert\Delta\mathbf q\rvert \le 10^{-6}$ bohr, $\lvert\Delta E\rvert \le 10^{-7}\,E_h$, $\lvert\Delta P\rvert \le 10^{-6}$. Measured: $1.27\times10^{-13}$ bohr, $2.13\times10^{-13}\,E_h$ ($E_{\text{tot}}$ $1.99\times10^{-13}$, printed), $4.55\times10^{-15}$; 28 s. The first measurement before the decoy was added gave $4.3\times10^{-13}$ bohr, $1.6\times10^{-13}\,E_h$ and $2.7\times10^{-15}$.
- *(d), water CIS (`tda`, `functional = "hf"`, STO-3G, `singlets = 3`), start S1, `hopping = "none"` — post hoc.* The planned test (20 steps of 0.25 fs, drift $\le 10^{-4}\,E_h$) is a strict expected failure: with the root fix of §8.6.3 it measures drift $1.056\times10^{-3}\,E_h$ and then ends `overlap_failure` at 3.594 fs (smallest singular value $2.78\times10^{-5}$). CIS S1 of water is strongly dissociative (kinetic energy $0.0033 \to 0.145\,E_h$ within 2 fs; S1-S0 gap 13.2 → 3.5 eV within 3.5 fs), so an absolute bound is not reachable at this step. The asserted replacement: over 2 fs at $\Delta t = 0.25$ and $0.125$ fs both runs finish without a rejection and the drift ratio is in [2.5, 6.0]: $1.056\times10^{-3}$ and $2.636\times10^{-4}\,E_h$, ratio **4.007**. Probe to 3 fs at $\Delta t = 0.25$, $0.125$, $0.0625$ fs (report): $1.056\times10^{-3}$, $2.636\times10^{-4}$, $6.586\times10^{-5}\,E_h$, ratios 4.006 and 4.003, i.e. pure $\mathcal O(\Delta t^2)$ Verlet error and forces consistent with the energies. Before the root fix the test could not run at all: the skipped root made S2 at $t=0$ the 16.77 eV root and the overlap with the first displaced geometry $3.7\times10^{-5}$ (`overlap_failure` at $t=0$).
- *TDA root fix, asserted (`tests/electronic`):* water HF/STO-3G CIS at the $C_{2v}$ test geometry, `singlets = 3`: S2 − S0 = 15.1445 eV to $10^{-3}$ eV (was 16.7707 eV); the first rung solves 5 roots for 2 requested states and keeps the lowest 2.
- *Hops with PySCF, asserted (added in the final fix wave, review finding M1):* the other PySCF tests run with `hopping = "none"` and the end-to-end runs have no hop, so the gradient refresh after a hop (`compute_more` of `PySCFBackend` inside the runner) was not exercised. Water CIS/STO-3G as in (d), start S1, $\Delta t = 0.125$ fs, 8 steps, `hopping = "sharc"`, `coupling = "overlap"`, EDC, a replayed random stream of zeros (a hop whenever a probability is $> 0$): finished, at least one hop, every log record finite, $\lvert E_{\text{tot}} - E_{\text{tot}}(0)\rvert \le 5\times10^{-4}\,E_h$ (bound fixed before the run). Measured: **6 hops**, 0 frustrated, no rejection, drift $1.945\times10^{-4}\,E_h$.
- *(e) and (f), asserted, the whole workflow* (`tests/test_e2e_dynamics.py`: water CIS/STO-3G, `singlets = 3`, 4 samples, 2 trajectories, $\Delta t = 0.25$ fs, `allow_fewer = true`; the example of §5.8): the run exits 0; the summary has one row per selected trajectory and every outcome is one of the outcomes of §8.16.4 (six since P9a plan 2); a second run leaves every `steps.log` modification time and `dynamics_summary.json` byte-identical (nothing recomputed); raising `tmax_fs` from 1.0 to 1.5 continues the finished trajectory (`traj_00000` is asserted to be among the finished ones and every finished row to end at 1.5 fs; this rests on that one trajectory, because `traj_00001` ends as described in §8.16.4: `overlap_failure` at 0.09375 fs, S2/S3 crossing, smallest singular value $5.2\times10^{-12}$, roots checked against a six-root reference). (f) (slow): the controller's process group is killed with SIGKILL after `restart/current` of `traj_00000` appeared (workers become orphans); the next run exits 0 with all trajectories at an outcome (killed 40.5 s after the start, second run 3.3 s, `traj_00000` finished at 2.0 fs). **Since P9a plan 2** the test keeps the first run's output (`first_run.log`) and prints the recovery path from `n_resumes`; on the development machine: *reconciled from result.json (the orphaned workers finished)* (killed after 25.0 s, second run 7.5 s). The path is printed, not asserted: the workers survive the controller's kill, so both paths are correct.

**C. SHARC 4.1 as a black box (`tests/validation/test_sharc_pyscf_dynamics.py`, markers `sharc`, `pyscf`, `slow`; needs `SHARC_HOME`/`SHARC_PYTHON`; never in CI).** One hopping-off trajectory of `sharc.x` with `SHARC_PYSCF.py` against ours on ethylene SA2-CASSCF(2,2)/**6-31G\*** (the plan's STO-3G was replaced before any SHARC run, because of B), geometry `tests/data/ethylene.xyz` rounded to SHARC's 7-decimal ångström interface grid, the velocities of B, start S1, $\Delta t = 0.5$ fs, 25 substeps, `endpoint` interpolation, `parallel_vel`, no decoherence, CASSCF `conv_tol` $10^{-10}$ and `conv_tol_grad` $10^{-6}$ on both sides. The time of one SHARC step (5.3 s) set $n = 40$ steps (measured wall times of the 40-step runs: SHARC 205.7-224.5 s; ours, to their end at step 24, 31-33.6 s with `overlap` and 63-67.1 s with `nacdr`; in the report script, where our runs continue to step 40, 54.9 s and 112.7 s). **Both sides use PySCF as the electronic engine** (`SHARC_PYSCF.py` calls PySCF), so the "independent" CASSCF solutions below are separately run solves of the same library: this comparison validates our driver and our interface to PySCF against SHARC's driver and interface, **not the electronic structure itself**. Facts found on the way (measured before any comparison): SHARC 4.1's `QM.in` says `step 0` and `SHARC_PYSCF.py` (interface version 4.0) stops on it, so the harness's `runQM.sh` rewrites it to `init` (a harness shim; SHARC is not changed); **`SHARC_PYSCF.py` has no overlaps** (it stops at step 1 with `coupling overlap`), so the SHARC side runs `coupling nacdr` with `nophases_from_interface` and **ours runs twice, with `overlap` and with `nacdr`, and every tolerance is asserted against both** (with hopping off, positions, velocities and energies do not depend on the coupling). The keyword names come from the public manual only.
- *Window (post hoc).* Both programs see a CASSCF solution change at step 24 (12 fs): E0 falls by 25 mEh and E1 rises by 37 mEh in both (SHARC E0 −77.84680868, ours −77.84680900 $E_h$), $E_{\text{tot}}$ jumps by 0.976 eV (SHARC 0.97613 eV, ours 0.97612 eV; our smallest singular value there is 0.686, above the threshold 0.5, and CASSCF stays on its first rung, so there is no rejection: this solution change passes every rejection signal, §8.16.3). Our energy-drift stop ends our runs at step 24 (outcome `energy_drift`, as designed); SHARC, which applies no such stop, continues. **Asserted:** this change itself (our outcome `energy_drift` at grid step 24 with no rejection, SHARC's $E_{\text{tot}}$ jump above 0.5 eV at the same step and at no earlier one, equal $\Delta t$ and masses). The direct comparison is asserted over **steps 0-23**, the steps before it. **With the energy-jump signal (P9a plan 2)** our run no longer accepts the jump: both couplings end `electronic_discontinuity` at 11.8125 fs after 23 completed grid steps, with 4 rejections, all `energy`, and the message `|E_tot - E_tot(previous record)| = 0.9431 eV exceeds energy_jump_limit_eV = 0.1` (asserted: 23 grid steps and $11.5 < t_{\text{final}} < 12.0$ fs; **post hoc**: the plan asserted 24 grid steps, measured 23, because the jumping 24th interval is rejected and the run ends inside it; the assertion was changed after the number was seen). The `energy_drift` assertion above is that of the code before plan 2 and the comparison runs keep it by setting no energy-jump limit. The live file now has 18 tests: **18 passed in 596.97 s** (P9a plan 2, Task 2).
- *Direct comparison, steps 0-23, ours (`overlap`, `nacdr`) against SHARC (`nacdr`), asserted at the tolerances of the plan fixed before the run:* velocities $\le 10^{-7}$ bohr per a.u.: **$1.722\times10^{-8}$**; MCH energies $\le 10^{-6}\,E_h$: **$3.904\times10^{-7}$**; populations (bound $10^{-4}$): $6.070\times10^{-6}$ (`overlap`) and $3.488\times10^{-6}$ (`nacdr`). **The population assertion cannot discriminate (post hoc, found in review):** P(S0) stays at or below $2.4\times10^{-5}$ in SHARC, $2.13\times10^{-5}$ (`overlap`) and $2.08\times10^{-5}$ (`nacdr`) over these steps, below the bound. In relative terms the agreement is poor: $\lvert P_{\text{ours}} - P_{\text{SHARC}}\rvert/P_{\text{SHARC}}$ is 0.12 (step 3), 0.72 (step 7), 0.63 (step 13), 0.73 (step 14) and 0.30 (step 23) for `overlap`, up to 0.73 in the window (0.83 at step 8 for `nacdr`), i.e. **a factor of 3-6 in P(S0)** (report, table in the test output and in `~/p7b_reports/sharc_pyscf_dynamics.json`). Like-for-like coupling helps only partly (0.03 at step 3, 0.40 at step 13, 0.17 at step 23 for `nacdr`; 0.71 at step 14, 0.76 at step 7). It fits the explanation that at P(S0) of $10^{-8}$ to $10^{-5}$ the amplitude is an integral of a small coupling term that depends on two independent CASSCF solutions, not on the coupling scheme or the driver; the populations are validated by the swap test below.
- *Positions, report only (post hoc).* The planned assertion (positions $\le 10^{-6}$ bohr) fails: the maximum over steps 0-23 is $2.401\times10^{-6}$ bohr (max $2.5\times10^{-6}$ over 0-24) and first exceeds $10^{-6}$ at step 12, for both of our couplings. Diagnosis: the two programs converge CASSCF independently at `conv_tol_grad` $10^{-6}$ with different solvers (SHARC with `fix-spin-shift`, ours singlet-only), and the S1 gradients differ by up to $3.6\times10^{-6}\,E_h/a_0$ at steps 8-11 (the strongly anharmonic stretch); a velocity offset of about $8\times10^{-9}$ bohr per a.u. accumulates to the observed drift. The hypothesis was **tested and could not be confirmed**: tightening convergence on both sides is impossible (SHARC's interface stops at step 0 with `conv-tol` $10^{-12}$ / `conv-tol-grad` $10^{-8}$, "Calculator failed to converge"; ours fails at step 0 for $10^{-8}$ and at step 1, also after halving to $\Delta t/8$, for $10^{-7}$). So the position comparison is a report (maximum $2.401\times10^{-6}$ bohr); the swap test below is the asserted check of the drivers.
- *Swap test, asserted (promoted from a report after the numbers were seen).* `sharc.x` drives **our** `PySCFBackend` (our own `QM.out` provider behind `runQM.sh`, the P7a-2 pattern, with the Löwdin-orthonormalised overlaps) and is compared with our driver over all 41 steps (our reference run with an energy-drift limit of 100 eV so that it passes step 24; the test asserts that it finishes), at the tolerances of the plan: **$\Delta q = 2.134\times10^{-7}$ bohr, $\Delta v = 2.115\times10^{-9}$, $\Delta E = 4.550\times10^{-8}\,E_h$, $\Delta P = 8.629\times10^{-7}$** (not bit-identical between runs, third digit). With our electronic structure behind both drivers every tolerance holds at all 41 steps, including those after the solution change, so the step order, the Verlet integration and the local-diabatization propagation agree with SHARC's driver. The residual of $10^{-7}$ bohr is consistent with SHARC evaluating at its rounded geometry. Against the independent electronic structure of `SHARC_PYSCF.py` the same driver drifts away by $3.4\times10^{-6}$ bohr in position (report) and 0.35 in population after step 24 (report; steps 34-40 near a near-degeneracy 9.8 mEh at step 32 where NAC and overlap propagation give P(S0) 0.389 against 0.279).
- Run times: the live file `test_sharc_pyscf_dynamics.py` gave 14 passed in 459.9 s after the first fix round and **16 passed in 482.0 s at `188b20b`** (the two added items are the report-only `test_relative_s0_population_report[overlap|nacdr]`). The swap report script (`tools/sharc_compare/run_sharc_pyscf_dynamics.py --report`, 8 min 13 s, JSON `~/p7b_reports/sharc_pyscf_dynamics.json` on the development machine) holds every per-step difference.

**Summary of what the validation does and does not show.** The dynamics stage's file handling, restart, rejection and extension are validated with the fake backend (bit-identical) and, for PySCF, to round-off (resume) and by the end-to-end runs; the physical conservation is validated by the $\Delta t^2$ scaling of the drift (ratios 2.705, 4.007), not by an absolute bound; the match with SHARC's own electronic structure is closer than $10^{-7}$ in velocity and $4\times10^{-7}\,E_h$ in energy over 23 steps, but **not** closer than $2.4\times10^{-6}$ bohr in position and only to a factor of 3-6 in the tiny populations, for a reason traced to independent CASSCF convergence; the drivers agree to $2\times10^{-7}$ bohr in the swap test. No ensemble comparison with SHARC exists (P9).

**Run times and counts** (measured; commit stated): the fast tier `pytest -m "not slow and not pyscf" -W error` and `pytest -m sharc` of the finished branch are in the next paragraph; the full suite `pytest -W error` of the finished branch, after the final review and its fix wave, at `9431ce0` with SHARC and ORCA, is **1223 passed, 2 skipped, 3 xfailed in 2 h 39 min (9561.09 s)** (skips by design: the `overlap` phase-drift case of the SHARC LVC comparison and the published-values case of the Tully comparison; xfails: the two strict xfails of (a) and (d) and the 3000 K `wigner.py` case of §8.8). Measured before: fast tier at `8752591` 1001 passed, 1 failed (`tests/test_worker.py`, fixed in the same commit; the file re-run: 12 passed); PySCF files `test_pyscf_dynamics.py` + `test_e2e_dynamics.py` 6 passed, 2 xfailed in 166.4 s at `da8010c`.

Measured in Task 11 at `188b20b` (code on `/mnt/c`, the documents edited while the tests ran): `pytest -m "not slow and not pyscf" -W error`: **1013 passed, 20 skipped, 184 deselected in 832.76 s (13 min 53 s)** (the 20 skips are the live-SHARC tests without `SHARC_HOME`); `SHARC_HOME=… SHARC_PYTHON=… pytest -m sharc -W error`: **60 passed, 1 skipped, 1 xfailed in 1272.86 s (21 min 13 s)** (the skip is the `overlap` phase-drift case of `test_sharc_lvc.py`, reported and not asserted; the xfail is the 3000 K `wigner.py` case; the intermittent P6 test of §10 passed in this run).

Measured in the final fix wave at `368f1dc`: the fast tier **1023 passed, 20 skipped, 185 deselected in 908.76 s (15 min 9 s)**; `pytest tests/validation/test_pyscf_dynamics.py tests/test_e2e_dynamics.py tests/electronic/test_pyscf_linear_response.py -W error` (all markers) **20 passed, 2 xfailed in 225.51 s**.

#### 8.16.9 Diagnosis of the M4 losses (P9a plan 1: findings, no behaviour changed)
**Status: findings of P9a plan 1.** Plan 1 changed nothing in the program, and **nothing in this section is [IMPLEMENTED] program behaviour.** The owner approved the diagnosis report on 2026-11-13; plan 2 then added the energy-jump signal (§8.16.3), the resume and signal records (§6) and two dormant CASSCF rungs (removed again on 2026-11-13, §8.6.8), and ran the acceptance study: its results and the guidance are in §5.10, its record in `docs/superpowers/reports/2026-11-13-p9a-acceptance.md`. Where this section says that no signal sees a switch, that describes the code before plan 2.

**Where the results are.** The full report, with every number and the file each comes from, is `docs/superpowers/reports/2026-11-12-p9a-diagnosis.md`.

**The diagnosis tools.** They are report scripts: they write JSON and assert nothing, `gqteashmd` never imports them, and they run on the WSL native filesystem.

- `tools/diagnosis/trajectory_forensics.py` reads a project's trajectories and lists, for each one, every per-record $E_{\text{tot}}$ jump above 0.1 eV, the hops before the end and the last CASSCF log lines.
- `tools/diagnosis/replay_points.py` re-solves the geometries around a failure in three ways (continued, backward, fresh) and with method variants, and records the energies, the active-orbital and state overlaps, a determinism pair and convergence trials.
- `tools/diagnosis/run_variants.py` runs our trajectory runner on a project's fixed initial conditions with changed `[electronic]` and `[dynamics]` settings (`tools/diagnosis/variants_p9a.toml`).
- `tools/sharc_compare/run_sharc_pyscf_ensemble.py` runs SHARC 4.1 (`sharc.x` with `SHARC_PYSCF.py`, executed only) on the same initial conditions. Since plan 2 it refuses `[dynamics]` values it cannot translate to SHARC (and `hopping = "fssh"`).
- Added in plan 2: `tools/diagnosis/convergence_probe.py` (the extra CASSCF rungs at the failure geometries, with an accuracy check of the loose tolerances; **removed with those rungs on 2026-11-13**, its measurement is in the acceptance report), `tools/diagnosis/start_solutions.py` (the SA-CASSCF solutions found from a fixed set of starting guesses at the initial geometries), and the options `--trajectories`, `--base-variant` and `--from-first-hop` of the replay and forensics tools (`tools/diagnosis/variants_p9a2.toml` holds the plan-2 variants).
- Added with the CASSCF start rule: `tools/diagnosis/start_rule_check.py` (the rule's step-0 solution at a project's starting geometries against the survey of `start_solutions.py`, and the cost of one reference search; the measurement of §8.6.2).

Their helpers have unit tests in `tests/diagnosis/`.

**What was measured** on the 20 M4 initial conditions (seed 2026, start S1). "Kept" means `finished` or `stopped_in_s0` with no $E_{\text{tot}}$ jump above 0.1 eV between records.

| Setup | Kept of 20 |
|---|---|
| M4 as run | 0 |
| `coupling = "nacdr"` | 0 |
| `dt_fs = 0.25` | 1 |
| SA3-CASSCF(2,2) (`singlets = 3`) | 11 |
| SA3-CASSCF(2,3) | 0 |
| SA2-CASSCF(4,3) | 0 |
| SHARC 4.1, as M4 | 1 |
| SHARC 4.1, SA3-CASSCF(2,2) | 2 |

**No setup reached 16 of 20**, the target of the planned acceptance run.

**The root causes:**

- **`energy_drift`** (7 trajectories) is a switch between two solutions of the SA2-CASSCF(2,2)/6-31G\* problem. SHARC with `SHARC_PYSCF.py` makes the same jump at the same step in 6 of 7, so in those 6 it is a property of the method at these settings. In the seventh (00014) the two programs took different solution branches. The switch passes every rejection signal of §8.16.3: the gap is 3.4-5.1 eV, the smallest singular value 0.72-0.82 and the rung unchanged. A step of 0.25 fs does not avoid it. A third state in the average (SA3) gave a single solution in the replayed windows. In the SA3 runs, the 7 trajectories pass the times at which they drifted (6.5-11 fs), and none of the 20 ends with `energy_drift` before 40.25 fs. But SA3 drifts later, after a hop, in 2 trajectories (40.25 and 59.0 fs); those were not diagnosed.
- **`overlap_failure`** (8) comes 5.8-42.9 fs after the S1 → S0 hop. In 3 trajectories it was reproduced as the same kind of switch, which the overlap signal does see. The other 5 have indirect evidence only: up to the hop, our energies equal SHARC's to $5\times10^{-5}$ eV, and after it the paths differ. No state outside the computed set was found.
- **`electronic_failure`** (3): CASSCF stops at its 150-macro-iteration limit at a stationary energy.
  - In one trajectory, a fresh solve at the failed geometry fails at the run's settings, as the run did, and converges with `conv_tol_grad` = $10^{-4}$.
  - In the second (0.25 fs), our fresh solve at the failed geometry likewise fails at the run's settings and converges with $10^{-5}$ or $10^{-4}$. But at the geometry of the run's rejected first full step (0.5 fs), SHARC's interface returned a converged result where our solver fails on every rung, at the same nominal tolerances; our solve converges there only at $10^{-4}$, on SHARC's solution.
  - In the third, both programs converge at 17.0 fs and fail on the next calculation. Our solver also fails at SHARC's 17.0 fs geometry, where SHARC converged. Our run's own convergence at 17.0 fs is not reproduced by re-solves: unexplained, and convergence that depends on the path is not excluded.

**Not settled** (owner decisions in the report):

- Whether the two cases where SHARC converges and our solver does not, and the one trajectory in which the two programs part where ours halved a step, count as defects of ours.
- Whether a fresh solve that lands on another SA-CASSCF solution counts as a defect. The report reads it as several valid solutions, a reading made after the numbers were seen.

**What this means for ethylene SA2-CASSCF(2,2)/6-31G\* today:**

- Expect solution switches. (Plan 1 text: the `analysis` flag of §8.13.6 was then the only automatic sign of the smaller ones; since plan 2 the `dynamics` stage rejects jumps above `energy_jump_limit_eV`, §8.16.3.)
- SA3-CASSCF(2,2) kept the most trajectories in this measurement.
- **SA3-CASSCF(2,2) also has two solutions at the starting geometries:** in 13 of 20 trajectories our program and SHARC start on different ones. The measured yield therefore depends on the starting orbitals. (Since 2026-11-13 our start is fixed by the CASSCF start rule, §8.6.2.)

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
| `SA-CASSCF failed on every rung (conv_tol = …, conv_tol_grad = …): previous orbitals: not converged; reference orbitals: …; SCF orbitals: …` / `CASSCF (…) failed: …` (rungs that were skipped are not listed) | All rungs of the CASSCF ladder (§8.6.8) failed, by non-convergence or a non-singlet root (exit 4) | Choose another active space or geometry; loosen `casscf.conv_tol_grad` slightly |
| `CASSCF root S1 has <S^2> = …; it is not a singlet.` | A root of the active space is not a singlet (exit 4) | Check the active space and the number of states |
| `The CASSCF active space (… electrons) does not fit the molecule's … electrons.` / `The CASSCF active space needs … orbitals, but the basis '…' has only …` | `ncas`/`nelecas` do not fit the molecule or the basis (exit 4) | Reduce the active space or use a larger basis |
| `TDA did not converge for all … excited states (tried: default, extra roots; …)` (also `TDDFT`) | Both rungs of the linear-response ladder failed (exit 4) | Ask for fewer states or loosen `td_conv_tol` |
| `Stage 'dynamics' needs [electronic] singlets >= 2 (S0 and at least one excited state)` | The input has `[dynamics]` but only S0 is computed (exit 2) | Set `[electronic] singlets` to at least 2, better 1-2 above the highest initial state (§8.16.4) |
| `Invalid input file …: dynamics.tmax_fs = … must be an integer multiple of dt_fs = …`; `dynamics.dt_fs … must be smaller than tmax_fs`; `dynamics.hopping = 'fssh' needs coupling = 'nacdr'`; `dynamics.ekin_correction = 'parallel_nac' needs coupling = 'nacdr'`; `dynamics.coupling = 'nacdr' needs analytic NACs, which exist for [electronic] method = 'casscf' only` | A `[dynamics]` rule (§4.2; exit 2; the violations of one rule group are listed in one message, so a second run can show more) | Correct the keys |
| `Stage 'excitations' has results produced from different inputs …` right after updating the program (exit 3) | The TDA/TDDFT root fix raised the stage version (§8.6.3) | `gqteashmd run INPUT --rerun-from excitations` (§7.5) |
| `Stage 'optimize' has results produced from different inputs …` (or `'frequencies'` without `optimize`) for a `casscf` project right after updating the program (exit 3) | The CASSCF start rule is part of the fingerprint of every CASSCF stage (§8.3, §8.6.2) | `gqteashmd run INPUT --rerun-from optimize`, or `--rerun-from frequencies` when `optimize` is disabled (§7.5) |
| `No CASSCF reference solution at the reference geometry (…): fresh solves start from the SCF orbitals.` (a backend log warning, written to the task's `worker.log` in a project run and to stderr by `single-point`, §8.6.2) | The SCF or every guess of the start rule failed at the reference geometry (§8.6.2); the calculation goes on from the SCF orbitals | Check the active space at the reference geometry with `single-point` (its `meta.casscf_reference`) |
| `The CASSCF solution started from the reference orbitals has a different active-orbital character (smallest singular value … < 0.9).` (a backend log warning, written to the task's `worker.log` and, for `single-point`, to stderr, as above) | A fresh solve started from the reference converged to a solution of another active-orbital character (§8.6.2); it is accepted | Compare the energies of that geometry with the reference; consider another active space |
| `Stage 'dynamics' has results produced from different inputs …` (exit 3) | A hashed key changed (everything except `tmax_fs` and `max_failed_fraction`, §8.3), or the results were computed before P9a plan 2 (stage version 2, §4.2) | `--rerun-from dynamics`; to run longer, only raise `tmax_fs` (§7.5) |
| `trajectories.json is missing or malformed (…); rerun the selection stage with --rerun-from selection` | The `selection` results are damaged (exit 3) | `--rerun-from selection` |
| `sample N is outside 1..M of samples.h5; rerun with --rerun-from sampling` or `the atoms of samples.h5 … differ from the prepared molecule` (a failed task) | `samples.h5` does not belong to `trajectories.json` | `--rerun-from sampling` |
| `trajectory traj_…: …; the restart or step log is unusable, rerun with --rerun-from dynamics` (a failed task) | A restart generation or `steps.log` of that trajectory is damaged or inconsistent | `--rerun-from dynamics` (recomputes all trajectories from their initial conditions) |
| A trajectory with outcome `overlap_failure`, `electronic_failure`, `electronic_discontinuity` or `energy_drift` in `dynamics_summary.md` (the stage still succeeds) | The trajectory ended for a physical or electronic-structure reason (§8.16.4); the `message` column, `rejections_by_signal` and `output.h5` of that task say when and why | `overlap_failure`: add states above the highest initial state, or check for a lost CASSCF active space; `electronic_discontinuity`: the electronic solution changed (`\|E_tot - E_tot(previous record)\| = … eV exceeds energy_jump_limit_eV = …`): for SA-CASSCF try more states in the average or another active space (§5.10); `energy_drift`: reduce `dt_fs` or look at the geometry at the end; failures at step 0: test the setup with `single-point` |
| `Stage dynamics: no output.h5 for: …` (warning) | A task finished without `output.h5` (the stage does not build it, §8.16.5) | Look at the task's `worker.log`; `--rerun-from dynamics` recomputes the stage |
| `Invalid input file …:` with `analysis.…` text (for example `S0 cannot be in analysis.fit_states (fit excited states)`, `analysis.fit_states must not be empty (omit it to use all)`, `analysis.fit_states: S3 does not exist with [electronic] singlets = 3 (S0..S2)`, `Input should be greater than or equal to 100`) | An `[analysis]` key out of range or inconsistent (exit 2, §4.2) | Correct the key |
| `Stage 'analysis' has results produced from different inputs …` (exit 3) | A hashed `[analysis]` key changed (everything except `enabled` and `warn_excluded_fraction`) | `gqteashmd run INPUT --rerun-from analysis` (recomputes only the analysis) |
| `No trajectory can be analysed: outcomes …; only 'finished' and 'stopped_in_s0' trajectories enter the averages (USER_MANUAL 8.13)` | Every selected trajectory ended early or its task failed (exit 4, §8.13.7); `analysis.json` and `report.md` are written with `stage_failed: true` | Read `dynamics_summary.md` and fix the dynamics (more states, a smaller `dt_fs`, another active space; then the `--rerun-from` the next run names). If the message adds `set [analysis] exclude_energy_jumps = false`, the jump exclusion removed the rest: set it and run with `--rerun-from analysis` |
| `populations of traj_… sum to 1 within … only (tolerance 1e-6): this indicates a program error` | The amplitudes of an included trajectory are not normalized (exit 4, §8.13.7) | Report it with the log and that `output.h5` |
| `…/output.h5: … rerun with --rerun-from dynamics` (for example `state_labels … differ from …`, `dt_fs … differs from [dynamics] dt_fs = …`, `records have no record at t = 0 (outcome 'finished')`, `non-finite e_total in an included trajectory`, `… is not a readable HDF5 file`, `no record at grid step …`), also `dynamics_summary.json is missing or malformed (…); rerun with --rerun-from dynamics` | A `dynamics` output is damaged or does not match the input (the analysis task fails, not retried; exit 4) | `--rerun-from dynamics` (recomputes every trajectory) |
| `… trajectories.json is missing or malformed (…); rerun with --rerun-from selection` (from `analysis`) | The `selection` results are damaged (exit 4) | `--rerun-from selection` |
| `Stage analysis: N% of the selected trajectories are excluded (more than warn_excluded_fraction = 0.2): possible selection bias, see 'excluded'` | **Warning**: many trajectories ended early (§8.13.1); the populations describe only those that ran to the end | Read the `excluded` table in `report.md`; treat the populations and the fit with care; improve the dynamics (§5.9) |
| `Stage analysis: fit: no_decay (…)` / `fit: tau_beyond_horizon (…)` / `fit: little_decay (…)` / `fit: t0_at_upper_bound (…)` / `fit: fit_failed (…)` / `fit: no_population (…)` / `fit: bootstrap_failures (…)` | **Warnings**: the lifetime fit is missing or unreliable (§8.13.4); the stage succeeds | Run longer (raise `tmax_fs`), use more trajectories, or fit other states (`fit_states`, then `--rerun-from analysis`) |
| `Stage analysis is recomputed because dynamics was extended` | **Warning**: raising `tmax_fs` reopened trajectories, so the old analysis was moved to `superseded/` (§7.5) | None; the analysis is redone in the same run |
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

- **Feature scope:** the stages `prepare`, `optimize`, `frequencies`, `sampling`, `excitations`, `selection`, `dynamics` and `analysis` exist. `dynamics` runs only if the input has a `[dynamics]` table; without it the workflow ends at `selection`, `pause_after = ["selection"]` has no effect (§7.8) and `analysis` is skipped.
- **`analysis` (P8, §8.13):** no plots (CSV, HDF5, JSON and Markdown only); one kinetic model (the delayed mono-exponential of §8.13.4), no populations per initial state, no geometry analysis. **Excluding the trajectories that ended early can bias the populations** when the failures depend on the dynamics (for example failures after a hop): read the excluded fraction and the `excluded` table first (§8.13.1; 90 % in §5.9). **With few included trajectories the bootstrap intervals are degenerate** (in §5.9, $N = 2$: $\tau$ interval [0.036, 32.61] fs with the estimate on its edge); since the P8 final fix wave fewer than 20 included trajectories give a warning and `small_ensemble: true`, and $N = 1$ gives `null` fit uncertainties (§8.13.4), but the stage still fits: 20 is guidance, not a measured coverage limit, and the bootstrap coverage of the fit is measured only at $N = 2000$ (§8.13.8). The energy jump is measured between consecutive **kept** records, which are several steps apart with `output_every` > 1 (§8.13.6). The first arrival in S0 is reported, not fitted (a censored mean is biased low). The target of a frustrated hop is not recorded, so frustrated hops are counted per originating state and `to` is empty in `hops.csv`. No comparison with SHARC's analysis tools (P9).
- **`dynamics` (P7b, §8.16):** closed-shell singlets only (no triplets or spin–orbit coupling from PySCF, so no intersystem crossing, O7; no `eselect`); the electronic structure limits what can be followed: states outside the computed set that cross a tracked state end the trajectory with `overlap_failure` (compute one or two states above the highest initial state), a metastable SA-CASSCF solution can be lost (§8.16.8 B), and the TDA/TDDFT root fix is a heuristic (§8.6.3). There is **no live `output.h5`** (built when a trajectory ends; read `progress.json` during the run). **A PySCF resume is not bit-identical** to an uninterrupted run (round-off, §8.16.6). Raising `tmax_fs` extends finished trajectories (and recomputes the analysis, §7.5); any other `[dynamics]` change needs `--rerun-from dynamics`, and so do results computed before P9a plan 2 (stage version 2). A CASSCF solution change is rejected when $E_{\text{tot}}$ jumps by more than `energy_jump_limit_eV` (P9a plan 2, §8.16.3); one with a smaller jump and a good overlap still passes every signal. DFT gradients omit the grid response (use `grid_level >= 5`); TDA/TDDFT describe the S1/S0 seam incorrectly (use SA-CASSCF); SA-CASSCF(2,2)/6-31G\* on ethylene does not converge at `conv_tol_grad` $\le 10^{-7}$ on the test path. No thermostat and no ensemble comparison with SHARC (P9): the single-trajectory comparison of §8.16.8 C matches in velocity ($1.7\times10^{-8}$) and energy ($3.9\times10^{-7}\,E_h$) over 23 steps but not in position (report-only, $2.4\times10^{-6}$ bohr), and its population assertion cannot discriminate (relative differences of a factor 3-6 in P(S0) $\le 2.4\times10^{-5}$). The SHARC electronic-structure interface has no overlaps, so the `overlap` coupling is compared driver to driver (swap test) only. The PySCF validation systems were changed after the numbers were seen (STO-3G → 6-31G\*; water CIS drift bound → $\Delta t^2$ ratio; stated in §8.16.8 B), and the $\Delta t^2$-ratio test has a thin margin (2.705 against 2.5).
- **Ethylene SA-CASSCF trajectory losses** (P9a, §5.10, §8.16.9).
  - **Cause.** In SA2-CASSCF(2,2)/6-31G\* the active-space problem has several solutions along the paths. A trajectory can switch between them by 0.6-0.9 eV in one step (the M4 energy drifts), or later lose the overlap (`overlap_failure`, 5.8-42.9 fs after a hop). SHARC 4.1 makes the same switch at the same step in 6 of the 7 energy drifts. SA3-CASSCF(2,2) has the same post-hop mechanism (reproduced in 4 of 5 of its seed-2026 overlap failures).
  - **Detection: closed for jumps above 0.1 eV.** Since P9a plan 2 the `dynamics` stage rejects a step whose $E_{\text{tot}}$ jumps by more than `energy_jump_limit_eV` and ends a persistent jump as `electronic_discontinuity` (§8.16.3). A switch with a smaller jump and a good overlap is still not detected.
  - **Still open: the losses after a hop.** The acceptance study (SA3-CASSCF(2,2), seed 2027) kept **13 of 20**, short of the target of 16; all 7 losses come after a hop (5 `overlap_failure`, 2 `electronic_discontinuity`), and the program has no remedy for them. On the seed-2026 initial conditions the same setup keeps 12 of 20.
  - **Convergence.** CASSCF can stop at its iteration limit at a stationary energy (`electronic_failure`; none in the SA3 runs). Looser tolerances converge at the three M4 failure geometries but are too inaccurate to enable (§8.6.8; the rungs that tried them were removed). At two geometries, SHARC's interface converged where our solver fails at the same nominal tolerances (no defect of ours identified). In one case our run converged at a geometry where re-solves fail (unexplained).
  - **Starting solution: fixed by the CASSCF start rule** (§8.6.2). SA3-CASSCF(2,2) has several solutions at the starting geometries. The rule starts every fresh solve from the reference solution at the reference geometry: the solution of PySCF's default active space if it converges with all-singlet roots, otherwise the lowest-SA solution among the other guesses (default guess first, owner decision 2026-11-14); on the 20 seed-2026 starts it lands on the lowest-SA solution of the P9a survey in 20 of 20, the start our runs already had. SHARC starts on another solution, outside the guess set, in 13 of 20; the rule does not seek it (a wider guess set or a global search was out of scope), and it is measured on ethylene only. **Why the default comes first:** the earlier lowest-SA criterion took guess `[6, 8]` for ethylene SA2-CAS(2,2)/STO-3G at the test geometry (SA -76.861346 Eh against -76.855342 Eh for the default `[7, 8]`) with an S0 46.6 mEh (1.27 eV) higher (-77.068750 against -77.115327 Eh), and the full suite then failed the finite-difference frequencies test (§8.6.2); for 6-31G\* (SA2, SA3) and STO-3G SA3 the default won by at least 10 mEh (report only). The choice is visible only through `single-point --json`; no input key overrides it. **STO-3G SA2 loses the default solution at small distortions whatever the start** (0.01 bohr; §8.6.2).
  - **P9a is closed** (owner decision 2026-11-13) with the acceptance miss documented; the losses after a hop are not being worked on now.
- **`sampling`:** harmonic and independent modes only (§8.8); no anharmonic or quasi-classical sampling, no sampling of an isotopically substituted molecule (the masses come from `frequencies`), samples are drawn on the S0 geometry of `frequencies` and not yet used by any later stage; the samples are not checked for chemical sense (very large amplitudes at high temperature or for floppy modes just above the cutoff are not detected).
- **Electronic structure:** closed-shell singlet molecules with a restricted reference only; no triplet states; nonadiabatic couplings (SA-CASSCF only) and wavefunction overlaps are used by the `dynamics` stage only (not by any other command); the analytic Hessian exists for RHF/RKS only (SA-CASSCF uses finite differences, §8.7); TDA/TDDFT have no analytic couplings, and their overlaps use the excitation amplitudes X only (§8.6.8); the cost of the overlap engine grows with the number of determinants kept (`overlap_norm_kept`); overlap-based couplings carry noise of order $10^{-7}/\Delta t$ at default CASSCF convergence; no density fitting; no scratch-directory input key yet (the backend supports one, and writes no checkpoint file); TDDFT/TDA describe the S₁/S₀ crossing seam incorrectly (§8.6.4); DFT gradients omit the grid response, so use `grid_level` ≥ 5 for dynamics (§8.6.4). The SCF, CASSCF and TD calculations each have a retry ladder (§8.6.1, §8.6.8); a calculation that fails every rung stops. **CASSCF start rule (§8.6.2):** its diagnostics (`casscf_reference`, `reference_active_min_sv`) are not recorded in a project run (the stages store no result metadata; they are visible through `single-point --json`); its two warnings go to the task's `worker.log` (and to stderr for `single-point`). Its cost (one search of 4 guesses for CAS(2,2), 0.516 s for ethylene SA3/6-31G\*) is paid once per backend, so per `frequencies` finite-difference task and, with `samples_per_task = 1`, per `excitations` sample; it was measured for CAS(2,2) only.
- **`excitations` and `selection`:** the spectrum is the vertical (nuclear-ensemble) spectrum with a Gaussian broadening: no vibronic structure, no refractive-index correction, no plot (`spectrum.csv` only), and it is not compared with an experiment. Each sample is computed from scratch (a full SCF and excited-state calculation per sample), with no parallelism inside a sample; the cost is $N_s+1$ calculations at the `[electronic]` level. The states are the adiabatic roots in order of energy at each geometry (no tracking between samples). Only singlets (S1, S2, ...) can be selected; no triplets. The bright-mode default `f_over_dE2` follows SHARC 4.1 `excite.py` as measured by black-box maximum likelihood (§8.9.4), not a published formula; it favours low-energy pairs, so one near-degenerate sample can dominate the normalization without an `energy_window_eV` (a warning is logged, §8.9.3). **SA-CASSCF:** since the CASSCF start rule every sample starts from the reference solution at the equilibrium geometry, projected (§8.6.2), instead of from its own SCF orbitals. A displaced geometry can still converge to a different active space; the backend measures this (`reference_active_min_sv`, a warning below 0.9): the warning goes to the task's `worker.log`, but the value is not stored by the stage (it is visible only through `single-point --json`, §8.6.2), so compare the per-state means in `excitations.json` with the reference values. The sample's total energy is not adjusted to the vertical excitation energy. Failed samples are excluded, which can bias the ensemble if failures cluster at certain geometries (the number of failures is in `excitations.json`).
- **`optimize` and `frequencies`:** they work on the S0 surface of `[electronic]` only (no excited-state optimization, no constraints, no transition-state search). There is no input key for isotope masses, and no thermochemistry (zero-point energy, entropy), anharmonicity or frequency scaling. A finite-difference Hessian costs $1+6N_{\text{at}}$ gradients, which is large for SA-CASSCF with many atoms. The check of the project directory covers the scratch space; there is no separate check that the disk has room for PySCF's scratch files beyond `min_free_disk_mb`. A refused stale run still appends a provenance record (§7.5).
- **Dynamics core (P7a-1, P7a-2):** `gqteashmd model-dynamics` runs it on analytic models only (Tully's three 1D models, a 2D test model and LVC models read from SHARC's `LVC.template` and `V0.txt`), without a molecule, PySCF or a project; molecules run through the `dynamics` stage (P7b, above). Spin–orbit coupling, triplets and `eselect` exist for the analytic and LVC models only and cannot be run on a molecule (PySCF provides no triplets or spin–orbit matrix elements); local diabatization runs on both. Fixed time step, no thermostat, no A-FSSH/NPI/GFSH/`parallel_diff`. **EDC at $C = 0.1\,E_h$ biases the Tully benchmarks** (§8.12, §8.14.1), and the defaults mix conventions: the EDC follows SHARC but the substep interpolation is `"midpoint"` (SHARC's is `"endpoint"`), so populations differ from SHARC at the $10^{-4}$ level during a crossing unless you choose `"endpoint"`. The restart is a library function (not used by `model-dynamics`); it stores $\mathbf U$ since P7a-2 but no model identity or options, and the caller must supply the same ones. Published branching values were not available in tabulated form (§8.14.1(b)); DAC at $\ln E = -2$ deviates from the exact result (a property of FSSH). The asserted validation takes about 94 minutes (exact comparison of P7a-1), 11 minutes (local diabatization against the exact solver, §8.15.9(b)) and, for `pytest -m sharc`, about 4 minutes at the end of P7a-1 and 12 minutes at the end of Task 10 of P7a-2 (§8.15.9); the report scripts take about 20–30 minutes, 1 hour (§8.14.1), 21 minutes and 57 minutes (§8.15.9); none of it is in the fast test tier.
- **LVC models and spin–orbit coupling (P7a-2):** only singlets and triplets (a doublet count is refused); `gamma` and `lambda_soc` with a count above 0 are refused and the dipole matrices are read but not used; of the SHARC `V0.txt` only an exact `0.0` frequency is a zero mode (a frequency between 0 and $10^{-6}\,E_h$ is refused; larger residual external frequencies are not detected and are treated as vibrations, §8.15.2); the orthogonality of the mode matrix is not checked; the initial vibrational state is the ground-state Wigner function at T = 0 (no temperature, no other sampling). `hopping = "fssh"` and `parallel_nac` do not work with spin–orbit coupling (§8.15.3), and `coupling = "overlap"` is not allowed with `hopping = "fssh"`. `eselect` is not validated against SHARC (its `SHARC_LVC.py` ignores selection, §8.15.6) and masks to zero like a backend that computes less. A model that is diagonal at the start and non-diagonal later is refused (§8.15.3). A trajectory that ends with energy drift is excluded from the population averages and only counted; if all end so, `model-dynamics` exits 1 and writes no JSON. The JSON of an LVC run contains the wall time `elapsed_s` and is therefore not byte-reproducible. The `U` comparison with SHARC is not resolved inside near-degenerate clusters (a rotation of 5·10⁻⁵ in the pair 1.7·10⁻⁸ $E_h$ apart cannot be tested, §8.15.9(e)), the $E^{\text{diag}}$ along our own path differs from SHARC's by 2.1·10⁻⁸ $E_h$ because of the rounding of SHARC's interface geometry, and the phases of $\mathbf U$ under local diabatization drift by 1.35·10⁻⁵ against SHARC's over 200 steps (reported only). The ensemble comparison with SHARC is a swap test (SHARC's driver with our electronic structure), not a comparison with `SHARC_LVC.py`. The time step of the example in §5.7.1 (20 a.u.) is not the validated 0.5 a.u. The 2D exact comparison (§8.15.9(c)) is a report: its S1 → S0 transfer differs from FSSH by up to 0.04.
- **Known intermittent test (not from P7a-2):** `tests/validation/test_sharc_excitation_energies.py::test_excitations_match_sharc_pyscf` (live SHARC, never in CI) was seen failing twice during Task 9 (once in a full `-m sharc` run and once in three standalone repeats; it passed in the Task 10 runs): our CASSCF S1 excitation energy of water CAS(4,4)/STO-3G was 0.42827465 $E_h$ in the failing run and 0.42680143 or 0.42680098 $E_h$ in the others (SHARC 0.42680104 each time), so our side sometimes converges to another CASSCF solution. **Open**: it needs a ruling or a fix (convergence or start orbitals), and a check that the result does not depend on the process. The CASSCF start rule does not touch it: the test builds the backend without a reference geometry, so the rule does not apply there, and the test was not re-run for the rule.
- **Platform:** Linux only, with no batch-scheduler support. Runs are designed for a single workstation.
- **Project-folder check:** the check for Windows drives and OneDrive (§7.9) looks at the text of the path. An absolute path that is a symbolic link into a Windows drive or a OneDrive folder is not detected.
- **SHARC comparison is not automated:** the tests against SHARC 4.1 need a SHARC installation and run only on the development machine (`pytest -m sharc`); the continuous-integration workflow (first run 2026-11-07, passing) runs everything else.

## 11. Roadmap

P1 foundation (done) → P2 input schema and pre-run checks (done) → P3a PySCF backend (done: energies, gradients, transition dipoles, `single-point`) → P3b couplings, overlaps, phase tracking and restarts (done, library only) → P3c hardening sweep (done) → P4 optimization, frequencies and Hessian (done: `optimize` and `frequencies` stages) → P5 Wigner sampling (done: `sampling` stage, validated against SHARC `wigner.py`) → P6 excitations and initial states (done: `excitations` and `selection` stages, compared with SHARC `SHARC_PYSCF.py` and `excite.py`) → P7a-1 dynamics core on analytic models (done: velocity Verlet, propagators, Tully/SHARC hopping, velocity adjustment, EDC, restarts, `model-dynamics`, validated against exact wavepackets and SHARC 4.1) → P7a-2 LVC models, spin–orbit coupling in the diagonal representation, local diabatization from overlaps, `eselect` (done: `model-dynamics` for LVC models, validated against exact wavepackets and SHARC 4.1) → P7b `dynamics` stage with PySCF (done: singlets, local diabatization or CASSCF NACs, step rejection, per-step restart, extension by `tmax_fs`, validated against SHARC 4.1 as a black box on one trajectory; reviewed and merged on 2026-11-11) → P8 analysis (reviewed, full test suite green and merged on 2026-11-12: the `analysis` stage with populations, intervals, a lifetime fit, hop and energy statistics, automatic recomputation after a change of `tmax_fs`, validated on synthetic ensembles; the M4 study of §5.9) → **P9 systematic validation against SHARC 4.1 (in progress)**. It is split into two parts:
- **P9a**, the M4 losses: **done and closed** (owner decision 2026-11-13). The diagnosis (plan 1) is in §8.16.9; plan 2 added the energy-jump rejection signal (`electronic_discontinuity`, stage version 2) and the resume and signal records, and ran the acceptance study, which **missed its target** (13 of 20 trajectories kept, 16 needed; §5.10). P9a was closed with that miss documented; the two dormant CASSCF rungs of plan 2 were removed, and `energy_jump_limit_eV` stays 0.1 eV.
- **The CASSCF start rule** (after the P9a closure; implemented, merged into `main` on 2026-11-14): fresh SA-CASSCF solves start from a reference solution (§8.6.2).
- **P9b** (next), the ensemble comparison with SHARC, including the analysis (`populations.py`, `make_fit.py`).

This manual is updated with each step. Sections marked [PLANNED] become [IMPLEMENTED] when their sub-project is finished and validated.

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
15. SHARC 4.1 manual (public user documentation, sharc-md.org): §8.7.1 (EDC, eqs. 8.25-8.26) and §8.33.1 (electronic propagation, eqs. 8.191-8.194), and the input formats used to run SHARC as a black box (§8.14.1(c)). SHARC's source code was not read. P7a-2 used the same manual for the LVC file formats (§6.4.1), the diagonal representation (§8.28–§8.29, §8.33), the input and output files (§4.1, §5.3.1, §6.34) and the interface of `SHARC_LVC.py` (executed only), §8.15.
16. S. N. Mannouch, J. O. Richardson, *J. Chem. Phys.* **158**, 104111 (2023), and L. He et al., *J. Phys. Chem. Lett.* **13**, 6966 (2022): published FSSH results on Tully's models (figures only; not used as numerical references, §8.14.1(b)).
17. H. Köppel, W. Domcke, L. S. Cederbaum, *Adv. Chem. Phys.* **57**, 59 (1984): vibronic coupling models (the LVC model of §8.15.2).
18. E. B. Wilson, *J. Am. Stat. Assoc.* **22**, 209 (1927): the score interval for a binomial proportion (§8.13.3).
19. B. Efron, *Ann. Stat.* **7**, 1 (1979): the bootstrap (§8.13.3, §8.13.4).
