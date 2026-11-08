# P6: `excitations` + `selection` Stages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two new stages.
- `excitations` computes vertical excitation energies, transition dipoles and oscillator strengths at every Wigner sample, in parallel tasks, plus the equilibrium geometry. It writes `excitations.h5`, `excitations.json` and a nuclear-ensemble absorption spectrum, `spectrum.csv`.
- `selection` chooses the initial state of each trajectory from those results. It uses a reproducible stochastic "bright" rule or a "fixed" state, and writes `trajectories.json` plus a review report for `pause_after`.

Both stages are validated against SHARC 4.1, run as a black box: energies against `SHARC_PYSCF.py`, selection statistics against `excite.py`.

**Architecture:**
- **Pure functions** live in a new package `gqteashmd.excite`, independent of PySCF:
  - `spectrum.py`: the broadened cross-section;
  - `selection.py`: the selection rules;
  - `artifacts.py`: the HDF5 and CSV files.
- **Stages.** `stages/excitations.py` is one task per chunk of samples plus a `ref` task. Each sample's electronic-structure failure is recorded and does not kill its task. `finalize` enforces `max_failed_fraction` over samples. `stages/selection.py` is a single task using the D12 seed stream.
- **SHARC comparison.** `tools/sharc_compare` gains our own writers for SHARC's `QM.in`/`QM.out`/`initconds` text formats, a driver that runs `excite.py` through stdin, and a parser for `initconds.excited`.

**Tech Stack:** Python 3.12, NumPy, SciPy (tests), h5py, PySCF (through the existing backend), pytest, ruff 0.17.4, mypy.

**Spec:**
- Master plan [`2026-11-06-gqteashmd-master-plan.md`](2026-11-06-gqteashmd-master-plan.md):
  - §1.1 (stages 4 and 5);
  - §2 (review pauses);
  - §3 (seeds D12);
  - §4.3 `[excitations]` and `[selection]`;
  - §5, the `excitations` and `selection` rows;
  - §7, the `excitations` and `selection` validation;
  - §9 P6;
  - O3 and O20.
- Strategy [`SHMD_implementation_strategy.md`](../../../SHMD_implementation_strategy.md) §4.8 (initial conditions and excitation).

**Owner decisions (2026-11-08):**
1. **O3, the "bright" rule.** This is the Newton-X-style double criterion (Barbatti et al., *J. Photochem. Photobiol. A* **190**, 228 (2007)).
   - A pair (sample *i*, state *n*) is a **candidate** if three things hold:
     - the sample's calculation succeeded;
     - *n* is in `selection.states`;
     - ΔE_in lies in `energy_window_eV`.
   - A candidate is **accepted** if r_in < f_in / f_max. Here r_in ~ U[0, 1) is drawn from the selection stream, and f_max is the largest f over **all candidates**.
   - Accepted pairs are taken in order (sample index, then state) up to `n_trajectories`. One sample may start several trajectories in different states.
   - A `"fixed"` mode starts every eligible sample in one given state.
2. **O20.** If fewer pairs are accepted than `n_trajectories`, the stage stops with advice. `allow_fewer = true` accepts fewer, with a warning.
3. **Spectrum: CSV only.** The stage writes `spectrum.csv` and no PNG; there is no new dependency.
4. **Validation.** Black-box `excite.py`, driven through stdin and never read, on data we write in SHARC's formats. Excitation energies and oscillator strengths are compared with `SHARC_PYSCF.py` on a few sample geometries.

## Global Constraints

- **Clean room.** Never read, open, grep or copy anything under `/home/ajc/sharc4-4.1` (SHARC) or the ORCA install directory.
  - SHARC may only be *executed*, from `tools/sharc_compare/`, and its output files inspected.
  - Our `QM.in`/`QM.out`/`initconds` writers and the `initconds.excited` parser are written from the observed text formats in "Facts" below.
  - `gqteashmd` never imports anything from `tools/`. MIT licence.
- **Environment.** Linux only. From PowerShell:
  `wsl -d OracleLinux_9_5 -- bash -lc 'cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && conda activate gqteashmd-dev && <command>'`.
  - For anything that uses a scratch directory, write a `.sh` script and run it with `wsl -d OracleLinux_9_5 -- bash /path/script.sh`, because variables inside the quoted `-lc` string can arrive empty.
  - Check `pwd` before writing, and delete scratch by its exact path.
  - Run one pytest process at a time, and leave no background shells or wait loops running.
- **Dependencies.** No new runtime dependencies. h5py is imported lazily, inside functions. PySCF is imported only in `electronic/pyscf_*.py`. `import gqteashmd.cli` must not import pyscf, geometric or h5py.
- **Units.**
  - Internally: Hartree atomic units.
  - Excitation energies in files and in the input: **eV**, ΔE[eV] = ΔE[Eh] × `HARTREE_IN_EV`.
  - Transition dipoles: e·a0.
  - Oscillator strengths: f = (2/3)·ΔE[Eh]·|μ|², computed with `electronic.properties.oscillator_strength`.
  - Cross-section: Å² per molecule.
- **States.** Labels are `S0`, `S1`, … The states of `[electronic] singlets` include S0 (O19). Candidate states are excited singlets only, S1 … S(singlets−1).
- **Seeds (D12).** The selection stream is `gqteashmd.util.seeds.task_rng(project.seed, "selection", 0)`. The dynamics stream of trajectory *k* (0-based) will be `task_rng(project.seed, "dynamics", k)`. `trajectories.json` records that stream; it does not invent a new integer seed.
- **One backend per task, reset per sample.** A task builds one backend with the **full** `[electronic]` settings (`stage_backend(..., settings=ctx.config.electronic)`, *not* `ground_state_settings`). It calls `backend.reset()` before every sample, because independent geometries must not share orbitals (CLAUDE.md P4/P6 note).
- **Errors.**
  - An `ElectronicStructureError` on one sample is recorded as that sample's failure, and the task continues.
  - A retryable error (`TaskError(retryable=True)`, e.g. `MemoryError`/`OSError` wrapped by the backend) is re-raised, so the task is retried.
  - Statistical or consistency failures are `StageFailedError` (exit 4) with advice.
  - Input problems are `ConfigError` (exit 2).
- **Style.** ruff 0.17.4, line length 100, rules E, F, W, I, B, UP and SIM. mypy must stay clean on `src/`.
- **Git.** Branch `feat/p6-excitations-selection`, WSL Git only, hooks on, never `--no-verify`. Conventional subjects, and each commit ends with the `Co-Authored-By:` trailer of the model that wrote it. Do not push.
- **Documentation.** The last task updates CLAUDE.md, USER_MANUAL.md and the master plan. Mark [IMPLEMENTED] and [PLANNED], and define every symbol.

## Facts verified before writing this plan (WSL, 2026-11-08)

1. **Backend contract** (`electronic/base.py`).
   - `Request(coords_bohr, transition_dipoles=True)` returns an `ElectronicResult` with these fields:
     - `states`, a tuple of `StateLabel`: S0 … S(singlets−1);
     - `energies`, Eh per label;
     - `transition_dipoles`, keyed S1 … with `np.ndarray(3)` in e·a0, S0 → S_I;
     - `meta`.
   - `ElectronicResult.excitation_energy(label)` gives E_I − E_0.
   - `backend.reset()` forgets the previous geometry.
   - `stage_backend(ctx, stage_name, molecule, task_dir, *, settings=None)` (in `stages/geometry.py`) defaults to the S0 settings. Pass `settings=ctx.config.electronic`.
   - `prepared_molecule(ctx.upstream_dir("prepare"))` gives the `MoleculeSpec` (symbols, charge, multiplicity).
2. **Orchestrator** (`orchestrator.py:_run_stage`).
   - A stage fails when `failed_tasks / total_tasks > stage.max_failed_fraction`, a class or instance attribute.
   - `finalize(ctx, results)` receives only the **DONE** tasks' results, keyed by task id.
   - A FAILED stage rerun requeues its failed tasks.
   - `default_pipeline(config)` receives the config, so a stage can be constructed with config-derived attributes.
3. **`samples.h5`** (P5) holds these datasets:
   - `positions` (n, N, 3) bohr;
   - `velocities` (n, N, 3) bohr/au_time;
   - `equilibrium_coordinates` (N, 3) bohr;
   - `symbols`, `masses` amu.

   Its `n_samples` attribute is `config.sampling.n_samples`. Sample *s* (1-based in files and task ids) is row *s − 1*.
4. **SHARC `excite.py`** (4.1; executed only). It is interactive and reads its answers from stdin; blank lines accept defaults. Working answer sequence, from the observed `KEYSTROKES.excite`:

   | # | Prompt | Answer |
   |---|---|---|
   | 1 | `Use file "initconds"? [True]` | `y` |
   | 2 | `How should the excited-state lists be generated? [2]` | `2` (read QM.out from ICOND dirs) |
   | 3 | `Path to ICOND directories:` | `.` |
   | 4 | `Do you want to use the diagonal representation (True=diag, False=MCH)? [False]` | `false` |
   | 5 | `How should the excited states be flagged? [3]` | `3` ("Simulate delta-pulse excitation based on excitation energies and oscillator strengths"; `2` = "Provide a list of desired initial states", `1` = unselect all) |
   | 6 | `Range (eV): [0.0 10.0]` | `<lo> <hi>` |
   | 7 | `Lower state for excitation? [1]` | `1` |
   | 8 | `Do you want to include all states in the selection? [True]` | blank |
   | 9 | `RNG Seed:  [!]` | an integer |
   | 10 | `Do you want to continue? [True]` | blank |

   - **Inputs it needs** in the working directory:
     - `initconds`, in the P5 format (header, `Equilibrium`, `Index k` blocks);
     - `ICOND_00000/` (the reference: its QM.out gives `Eref`) and `ICOND_00001/` … `ICOND_<N>/`, each holding a `QM.in` and a `QM.out`.
     - Without a `QM.in` that contains a `states <n>` line it crashes with "TypeError: object of type 'NoneType' has no len()" at prompt 7.
   - **The `QM.in`** that works:
     `3\nref\nO x y z\nH ...\nH ...\nunit angstrom\nstates 3\nh\ndm\n`.
   - **The `QM.out`** (as written by SHARC_PYSCF): blocks `! 0 Basic information`, `! 1 Hamiltonian Matrix (3x3, complex)` and `! 2 Dipole Moment Matrices (3x3x3, complex)`.
     - Block 0 has the lines `states 3 `, `nmstates 3`, `natom 3`, `npc 0` and `charges 0 `.
     - Each matrix is preceded by `3 3`. Every row holds 2n numbers in `%17.9E` style (real, imaginary).
     - There are three dipole matrices (x, y, z).
     - Block `! 8 Runtime` is optional.
   - **The output `initconds.excited`:**
     - Header `SHARC Initial conditions file, version 4.0   <Excited>`, then the lines `Ninit`, `Natom`, `Repr MCH`, `Eref`, `Eharm` and `States 3`.
     - Each `Index k` block has `Atoms`, then `States`, then one line per state:
       `idx  E_abs(Eh)  E_ref(Eh)  Re μx  Im μx  Re μy  Im μy  Re μz  Im μz  ΔE(eV)  f  Selected(True|False)`.
     - The dipole columns are the 1→idx transition dipole. Observed: `002 -74.55839998 -74.98520102 0.08877556 0 -0 0 0 0 11.61384795 0.00224244 False`.
     - The reported f matches (2/3)·ΔE·|μ|² (0.0022424 recomputed).
     - A summary table `State Selected InRange Total` is printed to stdout.
5. **Cross-section** (Crespo-Otero & Barbatti, *Theor. Chem. Acc.* **131**, 1237 (2012)), with refractive index 1:

   σ(E) = (π e² ħ / (2 mₑ c ε₀)) · (1/E) · (1/N_ok) Σ_samples Σ_n ΔE_n f_n g(E − ΔE_n).

   - In atomic units the prefactor is 2π²/c with c = 137.035999084 (α⁻¹, CODATA 2018), i.e. 0.144049 Eh·a0². In eV·Å² it is **1.097633**, from HARTREE_IN_EV × BOHR_IN_ANGSTROM².
   - Check: ∫σ dE = prefactor·f for one line, the standard integrated absorption.
   - g is a normalized Gaussian of full width at half maximum δ = `broadening_fwhm_eV`, with standard deviation s = δ / (2√(2 ln 2)).
   - The molar decadic absorption coefficient is ε[M⁻¹cm⁻¹] = σ[Å²] × 10⁻¹⁶ × N_A / (1000 ln 10) = σ[Å²] × **26153.0** (N_A = 6.02214076e23).

## File Structure

```text
src/gqteashmd/core/units.py                 # Task 1: SPEED_OF_LIGHT_AU, AVOGADRO
src/gqteashmd/config/schema.py              # Task 1: ExcitationsSection, SelectionSection, cross checks
src/gqteashmd/cli.py                        # Task 1: TEMPLATE blocks
src/gqteashmd/excite/__init__.py            # Task 2 (new package)
src/gqteashmd/excite/spectrum.py            # Task 2: CROSS_SECTION_PREFACTOR_EV_A2, broadened_spectrum
src/gqteashmd/excite/selection.py           # Task 2: Candidate, select_bright, select_fixed
src/gqteashmd/excite/artifacts.py           # Task 3: write/read excitations.h5, format_spectrum_csv
src/gqteashmd/vib/artifacts.py              # Task 3: read_samples_h5
src/gqteashmd/stages/excitations.py         # Task 4 (new)
src/gqteashmd/stages/selection.py           # Task 5 (new)
src/gqteashmd/stages/registry.py            # Task 6
tools/sharc_compare/sharc_formats.py        # Task 7 (new): write_qm_in, write_qm_out, format_initconds, parse_initconds_excited
tools/sharc_compare/run_excite.py           # Task 7 (new): run_excite (stdin-driven)
tests/...                                   # per task
CLAUDE.md, USER_MANUAL.md, master plan      # Task 8
```

---

### Task 1: Input sections `[excitations]` and `[selection]`

**Files:**
- Modify: `src/gqteashmd/core/units.py`, `src/gqteashmd/config/schema.py`, `src/gqteashmd/cli.py` (TEMPLATE), `src/gqteashmd/stages/base.py` (`check_config` hook), `src/gqteashmd/stages/pipeline.py` (`check_stage_inputs` calls it)
- Test: `tests/config/test_excitations_selection.py` (new), `tests/core/test_units.py`, `tests/stages/test_base_pipeline.py` (hook)

**Interfaces (produces):**
- **Constants** in `core/units.py`: `SPEED_OF_LIGHT_AU = 137.035999084` and `AVOGADRO = 6.02214076e23`.
- **`ExcitationsSection`** (`extra="forbid"`, frozen, like the other sections). Each float field must be finite, as in `[sampling]`.

  | Key | Type | Default | Rule |
  |---|---|---|---|
  | `include_reference` | bool | True | |
  | `samples_per_task` | int | 1 | ≥ 1 |
  | `max_failed_fraction` | float | 0.05 | 0 ≤ x < 1 |
  | `broadening_fwhm_eV` | float | 0.1 | > 0 |
  | `spectrum_step_eV` | float | 0.01 | > 0 |

- **`SelectionSection`:**

  | Key | Type | Default | Rule |
  |---|---|---|---|
  | `mode` | `Literal["bright", "fixed"]` | `"bright"` | |
  | `states` | `list[str] \| None` | None | None = all excited singlets |
  | `energy_window_eV` | `tuple[float, float] \| None` | None | None = no window |
  | `n_trajectories` | int | 100 | ≥ 1 |
  | `allow_fewer` | bool | False | |

  - A field validator parses every label with `StateLabel.parse` (an invalid label is a ConfigError naming it), and rejects `S0`, triplets and duplicates.
  - The window needs 0 ≤ lo < hi, both finite.
- **`ShmdConfig`** gets `excitations: ExcitationsSection = Field(default_factory=ExcitationsSection)` and `selection: SelectionSection = Field(default_factory=SelectionSection)`.
- **`ShmdConfig._consistent`** additionally reports, joined with the others by `"; "`:
  - `selection.states` containing a state ≥ S(singlets), when `[electronic]` is present. Message: `selection.states: S5 does not exist with [electronic] singlets = 3 (S0..S2)`.
  - `mode = "fixed"` without exactly one entry in `states`. Message: `selection.mode = 'fixed' needs exactly one state in selection.states`.
  - The check `[electronic] singlets >= 2` is **not** a schema rule, because existing tests use `singlets = 1` for S0-only stages. It is a stage-level input check (Task 4): the `Stage` base gets `def check_config(self, config: ShmdConfig) -> list[str]: return []`, and `stages/pipeline.py:check_stage_inputs` appends the problems of every **enabled** stage to its existing message list. That function is called by `validate` and `Orchestrator.__init__`, so the check runs before anything is computed. Task 1 adds the hook and its call in `check_stage_inputs`, with a unit test using a fake stage. Task 4's `ExcitationsStage.check_config` returns `["the excitations stage needs at least one excited state: [electronic] singlets must be >= 2"]` when `singlets < 2`.
- **TEMPLATE:** add active `[excitations]` and `[selection]` blocks with the defaults and one-line comments, as for `[sampling]`. The template's `singlets = 3` already satisfies the check, and the existing init test validates the template.

- [ ] **Step 1: Write the failing tests** (`tests/config/test_excitations_selection.py`):

```python
import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import ConfigError

TDA = '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 3\n'


def test_defaults(tmp_path, write_input):
    cfg = load_config(write_input(tmp_path, extra=TDA)).config
    assert cfg.excitations.samples_per_task == 1 and cfg.excitations.include_reference
    assert cfg.excitations.max_failed_fraction == 0.05
    assert (cfg.selection.mode, cfg.selection.states, cfg.selection.energy_window_eV) == (
        "bright", None, None)
    assert cfg.selection.n_trajectories == 100 and not cfg.selection.allow_fewer


def test_values(tmp_path, write_input):
    extra = TDA + ('[selection]\nmode = "bright"\nstates = ["S1", "s2"]\n'
                   "energy_window_eV = [4.5, 12.0]\nn_trajectories = 7\n")
    cfg = load_config(write_input(tmp_path, extra=extra)).config
    assert cfg.selection.states == ["S1", "S2"]  # normalized
    assert cfg.selection.energy_window_eV == (4.5, 12.0)


@pytest.mark.parametrize("extra, message", [
    ('[selection]\nstates = ["S0"]\n', "S0"),
    ('[selection]\nstates = ["T1"]\n', "T1"),
    ('[selection]\nstates = ["S1", "S1"]\n', "duplicate"),
    ('[selection]\nstates = ["X9"]\n', "X9"),
    ('[selection]\nstates = ["S5"]\n', "S5 does not exist"),
    ('[selection]\nmode = "fixed"\nstates = ["S1", "S2"]\n', "exactly one state"),
    ('[selection]\nmode = "fixed"\n', "exactly one state"),
    ("[selection]\nenergy_window_eV = [6.0, 4.0]\n", "energy_window_eV"),
    ("[selection]\nenergy_window_eV = [-1.0, 4.0]\n", "energy_window_eV"),
    ("[selection]\nn_trajectories = 0\n", "n_trajectories"),
    ("[excitations]\nsamples_per_task = 0\n", "samples_per_task"),
    ("[excitations]\nmax_failed_fraction = 1.0\n", "max_failed_fraction"),
    ("[excitations]\nbroadening_fwhm_eV = 0.0\n", "broadening_fwhm_eV"),
    ("[excitations]\nspectrum_step_eV = inf\n", "spectrum_step_eV"),
])
def test_invalid(tmp_path, write_input, extra, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write_input(tmp_path, extra=TDA + extra))
```

  Append to `tests/core/test_units.py`:

```python
def test_speed_of_light_and_avogadro():
    from gqteashmd.core import units

    assert units.SPEED_OF_LIGHT_AU == 137.035999084
    assert units.AVOGADRO == 6.02214076e23
```

- [ ] **Step 2: Run the tests to verify they fail.** Run `pytest tests/config/test_excitations_selection.py tests/core/test_units.py -v -W error`.
- [ ] **Step 3: Implement.** Follow the existing section style in `schema.py`: `_Section` base, `Field(..., ge=, gt=, lt=, allow_inf_nan=False)`, and the field validators. The cross-field checks go into `ShmdConfig._consistent`, appending to `problems`.
- [ ] **Step 4: Run the tests to verify they pass.** Run `pytest tests/config tests/test_cli.py -v -W error && mypy`, then the fast tier `pytest -m "not slow and not pyscf" -W error` once.
- [ ] **Step 5: Commit**

```bash
ruff format src tests tools && ruff check src tests tools
git add src/gqteashmd/core/units.py src/gqteashmd/config/schema.py src/gqteashmd/cli.py tests
git commit -m "feat: add the [excitations] and [selection] input sections"
```

---

### Task 2: Pure spectrum and selection functions (`gqteashmd.excite`)

**Files:**
- Create: `src/gqteashmd/excite/__init__.py` (docstring only), `src/gqteashmd/excite/spectrum.py`, `src/gqteashmd/excite/selection.py`
- Test: `tests/excite/__init__.py` (if the test tree uses packages; check `tests/vib/`), `tests/excite/test_spectrum.py`, `tests/excite/test_selection.py`

**Interfaces (produces):**
- **`spectrum.py`:**
  - The constant is

    ```python
    CROSS_SECTION_PREFACTOR_EV_A2 = 2 * math.pi**2 / SPEED_OF_LIGHT_AU * HARTREE_IN_EV * BOHR_IN_ANGSTROM**2
    ```

    which is about 1.097633.
  - `EPSILON_PER_A2 = 1e-16 * AVOGADRO / (1000 * math.log(10))` is about 26153.0.
  - `broadened_spectrum(excitation_energies_eV, oscillator_strengths, *, fwhm_eV, step_eV) -> Spectrum`.
    - **Inputs:** arrays of shape (n_ok, n_exc), with NaN-free rows only (the caller drops failed samples).
    - **Energy grid** E_j = e0 + j·step, j = 0…, where:
      - e0 = max(step, step·floor((min ΔE − 4 s)/step));
      - the grid runs to the first point ≥ max ΔE + 4 s;
      - s = fwhm/(2√(2 ln 2)).
    - **Output** `Spectrum` (frozen dataclass) with:
      - `energies_eV` (m,);
      - `cross_section_A2` (m,), the total;
      - `cross_section_by_state_A2` (m, n_exc);
      - `epsilon_M1cm1` (m,).

      σ_n(E_j) = PREFACTOR/E_j · (1/n_ok) Σ_i ΔE_in f_in g(E_j − ΔE_in), with g the normalized Gaussian.
    - **Errors:** `ValueError` if n_ok = 0.
- **`selection.py`:**
  - `@dataclass(frozen=True) class Candidate: sample: int; state: str; excitation_energy_eV: float; oscillator_strength: float` (`sample` 1-based).
  - `@dataclass(frozen=True) class Selection: selected: tuple[Candidate, ...]; candidates: int; f_max: float; accepted: int`. Here `accepted` counts all accepted pairs before truncation to `n`.
  - `candidates_from(excitation_energies_eV, oscillator_strengths, ok, states, window) -> list[Candidate]`.
    - **Arguments:** arrays (n_samples, n_exc) over the excited states S1…; `ok` (n_samples,) bool; `states` is a list of labels (subset of S1…S(n_exc)); `window` is `(lo, hi)` or None.
    - **Order:** sample-major, then the state order S1, S2, …, regardless of the order of `states` in the input.
    - **Window:** inclusive, lo ≤ ΔE ≤ hi.
  - `select_bright(candidates, n, rng) -> Selection`.
    - Draws exactly one `rng.random()` per candidate, in candidate order, whether or not it is needed. The stream is therefore a fixed function of the candidate list.
    - Accepts if r < f/f_max. f_max = max over candidates. If f_max ≤ 0 (no candidate with f > 0), `selected = ()`.
    - Returns the first `n` accepted.
  - `select_fixed(candidates, n) -> Selection` takes the first `n` candidates; it is used when `states` has one entry. `f_max` is the maximum f, or 0 if there is none; `accepted = len(candidates)`.

- [ ] **Step 1: Write the failing tests.**

`tests/excite/test_spectrum.py`:

```python
import math

import numpy as np
import pytest

from gqteashmd.core.units import BOHR_IN_ANGSTROM, HARTREE_IN_EV
from gqteashmd.excite.spectrum import (
    CROSS_SECTION_PREFACTOR_EV_A2, EPSILON_PER_A2, broadened_spectrum)


def test_prefactor_values():
    assert CROSS_SECTION_PREFACTOR_EV_A2 == pytest.approx(1.097633, rel=1e-6)
    assert EPSILON_PER_A2 == pytest.approx(26153.0, rel=1e-5)
    # SI cross-check: pi e^2 hbar / (2 m_e c eps0) in eV*A^2
    e, hbar, me, c, eps0 = 1.602176634e-19, 1.054571817e-34, 9.1093837015e-31, 299792458.0, 8.8541878128e-12
    si = math.pi * e**2 * hbar / (2 * me * c * eps0)  # J*m^2
    assert si / e * 1e20 == pytest.approx(CROSS_SECTION_PREFACTOR_EV_A2, rel=1e-8)


def test_single_line_integrates_to_prefactor_times_f():
    s = broadened_spectrum(np.array([[5.0]]), np.array([[0.2]]), fwhm_eV=0.1, step_eV=0.001)
    integral = np.trapezoid(s.cross_section_A2 * s.energies_eV / 5.0, s.energies_eV)
    assert integral == pytest.approx(CROSS_SECTION_PREFACTOR_EV_A2 * 0.2, rel=1e-6)
    assert s.energies_eV[np.argmax(s.cross_section_A2)] == pytest.approx(5.0, abs=0.002)
    np.testing.assert_allclose(s.epsilon_M1cm1, s.cross_section_A2 * EPSILON_PER_A2)


def test_average_over_samples_and_state_split():
    e = np.array([[4.0, 6.0], [4.2, 6.1]])
    f = np.array([[0.1, 0.0], [0.3, 0.05]])
    s = broadened_spectrum(e, f, fwhm_eV=0.2, step_eV=0.01)
    np.testing.assert_allclose(s.cross_section_by_state_A2.sum(axis=1), s.cross_section_A2)
    one = broadened_spectrum(e[:1], f[:1], fwhm_eV=0.2, step_eV=0.01)
    assert s.energies_eV[0] <= one.energies_eV[0]
    with pytest.raises(ValueError):
        broadened_spectrum(np.zeros((0, 2)), np.zeros((0, 2)), fwhm_eV=0.1, step_eV=0.01)
```

  Use `np.trapz` if NumPy < 2; check the installed version.

`tests/excite/test_selection.py`:

```python
import numpy as np

from gqteashmd.excite.selection import candidates_from, select_bright, select_fixed

E = np.array([[5.0, 7.0], [5.5, 9.5], [4.0, 6.5]])
F = np.array([[0.2, 0.4], [0.1, 0.8], [0.0, 0.3]])
OK = np.array([True, True, False])


def test_candidates_window_states_and_failed_samples():
    c = candidates_from(E, F, OK, ["S1", "S2"], (4.5, 8.0))
    assert [(x.sample, x.state) for x in c] == [(1, "S1"), (1, "S2"), (2, "S1")]
    c = candidates_from(E, F, OK, ["S2"], None)
    assert [(x.sample, x.state) for x in c] == [(1, "S2"), (2, "S2")]


def test_bright_uses_fmax_over_candidates_only():
    # state S2 of sample 2 (f = 0.8) is outside the window, so f_max = 0.4
    c = candidates_from(E, F, OK, ["S1", "S2"], (4.5, 8.0))
    sel = select_bright(c, 10, np.random.default_rng(0))
    assert sel.f_max == 0.4 and sel.candidates == 3


def test_bright_acceptance_frequencies_match_f_over_fmax():
    e = np.full((20000, 2), 5.0)
    f = np.tile([0.1, 0.4], (20000, 1))
    c = candidates_from(e, f, np.ones(20000, bool), ["S1", "S2"], None)
    sel = select_bright(c, 10**9, np.random.default_rng(1))
    states = [x.state for x in sel.selected]
    p1, p2 = states.count("S1") / 20000, states.count("S2") / 20000
    assert abs(p1 - 0.25) < 4 * np.sqrt(0.25 * 0.75 / 20000)
    assert p2 == 1.0  # f = f_max is always accepted (r < 1)


def test_bright_is_reproducible_and_truncates_in_order():
    c = candidates_from(E, F, OK, ["S1", "S2"], None)
    a = select_bright(c, 1, np.random.default_rng(5))
    b = select_bright(c, 1, np.random.default_rng(5))
    assert a == b and len(a.selected) <= 1


def test_fixed_takes_first_n_in_order():
    c = candidates_from(E, F, OK, ["S1"], None)
    sel = select_fixed(c, 1)
    assert [(x.sample, x.state) for x in sel.selected] == [(1, "S1")]
```

- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement** both modules. Write module docstrings that state the formulas with every symbol defined, and cite Crespo-Otero & Barbatti 2012 and Barbatti et al. 2007.
- [ ] **Step 4: Run them to verify they pass**, plus `mypy`.
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add the nuclear-ensemble spectrum and the initial-state selection rules"
```

---

### Task 3: Artifacts — `read_samples_h5`, `excitations.h5`, `spectrum.csv`

**Files:**
- Modify: `src/gqteashmd/vib/artifacts.py` (add `read_samples_h5`)
- Create: `src/gqteashmd/excite/artifacts.py`
- Test: `tests/vib/test_artifacts.py`, `tests/excite/test_artifacts.py`

**Interfaces (produces):**
- **`read_samples_h5(path) -> dict`** with keys `symbols` (tuple[str]), `masses_amu`, `equilibrium_bohr` (N, 3), `positions_bohr` (n, N, 3), `velocities_au` (n, N, 3), `n_samples` (int) and `temperature_K`. A missing or corrupt file raises `ValueError` naming the path and advising `--rerun-from sampling`, like `read_frequencies_h5`.
- **`write_excitations_h5(path, *, symbols, state_labels, sample_indices, ok, energies_hartree, transition_dipoles_au, oscillator_strengths, reference=None, project_seed=None)`.**
  - Atomic write; h5py imported lazily.
  - Attributes: `schema_version=1`, `gqteashmd_version`, `n_samples`, `n_ok`.
  - Datasets, each numeric one with a `units` attribute:

    | Dataset | Shape | Units / notes |
    |---|---|---|
    | `state_labels` | | strings, S0 … |
    | `sample_index` | int32 (n,) | 1-based |
    | `ok` | int8 (n,) | |
    | `energies` | (n, n_states) | hartree; NaN for failed samples |
    | `excitation_energies` | (n, n_states−1) | eV; NaN for failed samples |
    | `transition_dipoles` | (n, n_states−1, 3) | e*a0 |
    | `oscillator_strengths` | (n, n_states−1) | `-` |

  - Optional group `reference` with the same per-state datasets for the equilibrium geometry (a single row) and `coordinates` (bohr).
- **`read_excitations_h5(path) -> dict`**, the inverse, used by selection. Same error style; advice `--rerun-from excitations`.
- **`format_spectrum_csv(spectrum, state_labels) -> str`.**
  - Header lines starting with `#`: units, n_ok, fwhm and step.
  - Then the column line `energy_eV,cross_section_A2,epsilon_M-1cm-1,S1_A2,S2_A2,...`.
  - Values in `%.6e`.

- [ ] **Step 1: Write the failing tests.**
  - Round-trip `write_excitations_h5` through `read_excitations_h5`: NaN rows for a failed sample; units attributes; the reference group present or absent; no temporary file left.
  - Round-trip `read_samples_h5` on a file written by `write_samples_h5`, using a small `sample_wigner` call as in `tests/vib/test_artifacts.py`.
  - A missing file gives `ValueError` matching `--rerun-from`.
  - `format_spectrum_csv`: the header and the column count.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.** Follow `write_frequencies_h5` and `write_samples_h5`.
- [ ] **Step 4: Run them to verify they pass**, plus `mypy`, plus `python -c "import gqteashmd.cli, sys; assert 'h5py' not in sys.modules"`.
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: read samples.h5 and write excitations.h5 and spectrum.csv"
```

---

### Task 4: The `excitations` stage

**Files:**
- Create: `src/gqteashmd/stages/excitations.py`
- Test: `tests/stages/test_excitations.py`. It needs no PySCF: monkeypatch `gqteashmd.stages.excitations.stage_backend` to return a fake backend, and build `prepare` and `sampling` upstream directories with the existing writers.

**Interfaces (produces):**
- **`class ExcitationsStage(Stage)`:**
  - `name = "excitations"`, `version = 1`;
  - `depends_on = ("prepare", "sampling")`;
  - `config_sections = ("electronic", "excitations")`;
  - `required_modules = ("pyscf>=2.6", "h5py>=3.9")`;
  - `__init__(self, max_failed_fraction: float = 0.05)` sets `self.max_failed_fraction`.
- **`fingerprint_inputs`:**
  - the `[electronic]` dump without `overlap_norm_kept`;
  - `[excitations]` without `max_failed_fraction`, `broadening_fwhm_eV` and `spectrum_step_eV`, which are finalize-only;
  - `config.sampling.n_samples`.

  It reads no artifacts.
- **`plan`:**
  - `TaskSpec("ref")` if `include_reference`;
  - then one task per chunk of `samples_per_task` consecutive samples. The id is `s{first:05d}` and the payload is `{"samples": [first, ..., last]}`, 1-based, up to `config.sampling.n_samples`.
- **`check_config(config)`:** returns the Task 1 message when `config.electronic` is present and `singlets < 2`. Test it through `check_stage_inputs`.
- **`run_task`.** Build the molecule from `prepared_molecule(ctx.upstream_dir("prepare"))`, the geometries from `read_samples_h5(ctx.upstream_dir("sampling") / "samples.h5")`, and one backend via `stage_backend(ctx, self.name, molecule, task_dir, settings=ctx.config.electronic)`. Then, for each sample (or for `equilibrium_bohr` in `ref`):
  1. Call `backend.reset()`.
  2. Call `result = backend.compute(Request(coords, transition_dipoles=True))`.
  3. Record per-state energies, transition dipoles (S1…) and f via `oscillator_strength`.
  4. On `ElectronicStructureError` (non-retryable), record `{"sample": s, "ok": False, "error": str(exc)}` and continue. A retryable `TaskError` is re-raised.
  5. Return `{"samples": [ {sample, ok, energies_hartree:[...], transition_dipoles_au:[[...]...], oscillator_strengths:[...]} | {sample, ok:False, error} ]}`.

     `ref` returns the same with `"sample": 0`.
- **`finalize`:**
  1. Collect the per-sample rows over the DONE results. Samples of planned tasks that are not in `results` (failed tasks) count as failed, with the error "task failed".
  2. If `failed/n_samples > max_failed_fraction`, raise `StageFailedError`, listing up to 5 failed samples with their errors. Advice: fix the cause, or raise `[excitations] max_failed_fraction`.
  3. If `include_reference` and the reference failed, raise `StageFailedError` ("the equilibrium geometry itself failed; …").
  4. Write `excitations.h5` and `spectrum.csv` (via `broadened_spectrum` over the OK samples).
  5. Write `excitations.json` with: `n_samples`, `n_ok`, `failed` (a list of `{sample, error}`), per state the mean and std of ΔE (eV) and f over the OK samples, the reference ΔE and f per state, `broadening_fwhm_eV`, `spectrum_step_eV`, and the state labels.
  6. Log a summary line per state.

- [ ] **Step 1: Write the failing tests.** The fake backend has `states` S0…S2, `reset()` counting calls, and `compute(request)` returning an `ElectronicResult` whose energies are a function of the coordinates (e.g. E1 = E0 + 0.2 + 0.01·x₀). It raises `ElectronicStructureError` for one chosen sample. Test:
  1. `plan`: ids `["ref", "s00001", "s00003", "s00005"]` for n_samples = 5 and samples_per_task = 2; without the reference there is no `ref`.
  2. `run_task` of `s00001`: two rows; `reset` called before each sample; f equals (2/3)ΔE|μ|².
  3. One failing sample is recorded and the other rows are present.
  4. A retryable `TaskError` propagates.
  5. `finalize` writes the three files with the right shapes. A failed sample gives NaN rows and `ok = 0`.
  6. A failure fraction above the limit gives `StageFailedError` matching `max_failed_fraction`. A missing task result (a simulated failed task) counts its samples as failed.
  7. A failed reference raises.
  8. The fingerprint changes with the basis and `samples_per_task`, but not with `broadening_fwhm_eV` or `max_failed_fraction`.
  9. The backend receives the full settings (`singlets = 3`), not the S0 settings.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.** The fixed module-level name `stage_backend` must be imported into `stages/excitations.py` so tests can monkeypatch it.
- [ ] **Step 4: Run them to verify they pass**, plus `mypy`.
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add the excitations stage"
```

---

### Task 5: The `selection` stage

**Files:**
- Create: `src/gqteashmd/stages/selection.py`
- Test: `tests/stages/test_selection.py`. It builds an `excitations` upstream directory with `write_excitations_h5` and synthetic data.

**Interfaces (produces):**
- **`class SelectionStage(Stage)`:**
  - `name = "selection"`, `version = 1`;
  - `depends_on = ("excitations",)`;
  - `config_sections = ("selection",)`;
  - `required_modules = ("h5py>=3.9",)`.
- **`fingerprint_inputs`:** `{"selection": section dump, "project_seed": project.seed}`.
- **`plan`:** `[TaskSpec("select")]`.
- **`run_task`:**
  1. Read `excitations.h5`.
  2. Build the candidates (`states` None → all excited states).
  3. Use `rng = task_rng(project.seed, "selection", 0)`.
  4. Call `select_bright` or `select_fixed`.
  5. Return the small summary plus the selected list: sample, state, ΔE and f per trajectory. Its size is n_trajectories rows, which is acceptable.
- **`finalize`:**
  1. If `len(selected) < n_trajectories` and not `allow_fewer`, raise `StageFailedError` (O20) with the numbers: candidates, accepted, requested, f_max, window and states. Advice: increase `[sampling] n_samples` (with `--rerun-from sampling`), widen `energy_window_eV`, add states, lower `n_trajectories`, or set `allow_fewer = true`. With `allow_fewer`, log a warning. Zero selected is always an error.
  2. Write `trajectories.json`:

     ```json
     {"schema_version": 1, "n_trajectories": k, "mode": ..., "project_seed": s,
      "trajectories": [{"id": "t00001", "index": 0, "sample": 12, "state": "S1",
                        "excitation_energy_eV": ..., "oscillator_strength": ...,
                        "rng": {"stage": "dynamics", "task_index": 0}}, ...]}
     ```

  3. Write `selection_report.json`: mode, states, window, n_trajectories requested and selected, candidates, accepted, f_max, and per state the counts of candidates, accepted and selected, with the mean ΔE and f of the selected.
  4. Write `selection_report.md`, a human-readable review for `pause_after`:
     - a title;
     - the settings;
     - a per-state table;
     - the first 20 trajectories as a table;
     - the pointers `../<excitations dir>/spectrum.csv` and `excitations.json`;
     - the line "Review, then run `gqteashmd run` again to continue".

- [ ] **Step 1: Write the failing tests.**
  1. With synthetic data (50 samples, S1 dim, S2 bright, one failed sample), bright mode selects at most `n` and only from OK samples. The same seed gives an identical `trajectories.json`, and a different `project.seed` gives a different selection.
  2. Fixed mode with `states = ["S2"]` selects the first `n` OK samples in S2.
  3. Too few gives `StageFailedError` matching `allow_fewer`. With `allow_fewer = true` the stage writes fewer and warns (caplog).
  4. The window excludes out-of-range pairs.
  5. `selection_report.md` contains the per-state table.
  6. The fingerprint includes the seed.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them to verify they pass**, plus `mypy`.
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add the selection stage"
```

---

### Task 6: Pipeline … → sampling → excitations → selection; end-to-end

**Files:**
- Modify: `src/gqteashmd/stages/registry.py`, `tests/stages/test_base_pipeline.py`, `tests/test_e2e_frequencies.py` (or a new `tests/test_e2e_excitations.py`)

**Interfaces:**
- `default_pipeline(config)` returns `[Prepare, Optimize, Frequencies, Sampling, ExcitationsStage(max_failed_fraction=config.excitations.max_failed_fraction), SelectionStage()]`.
- **E2E** (marker `pyscf`): water HF/STO-3G, `method = "tda"`, `functional = "hf"`, `singlets = 3`, with `[sampling] n_samples = 4`, `[excitations] samples_per_task = 2` and `[selection] n_trajectories = 2`, `allow_fewer = true`, no window. Check:
  - `run` exits 0;
  - `excitations.h5` has 4 OK samples;
  - `spectrum.csv` has a positive maximum;
  - `trajectories.json` has 1–2 entries;
  - a rerun is up to date;
  - `pause_after = ["excitations"]` on a fresh project exits 5 after `excitations`, and the next run finishes `selection`.

- [ ] **Step 1: Write the failing tests.** Update the order test to six stages and write the e2e test.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.** Run the fast tier and report every existing test that needed changing (e.g. ones that ran the default pipeline with `singlets = 1`, or the existing e2e tests that now run two more stages). The fix for those: give them `singlets >= 2`, or keep them, depending on what Task 1 ruled.
- [ ] **Step 4: Run the tests.** Run `pytest tests/stages tests/test_e2e_*.py -m "not slow" -W error`.
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: run excitations and selection after sampling"
```

---

### Task 7: Validation against SHARC 4.1 (black box)

**Files:**
- Create: `tools/sharc_compare/sharc_formats.py`, `tools/sharc_compare/run_excite.py`, `tests/sharc_compare/test_sharc_formats.py` (always run), `tests/validation/test_sharc_excite.py` (markers `sharc`), `tests/validation/test_sharc_excitation_energies.py` (markers `sharc`, `pyscf`)

**Interfaces:**
- **`sharc_formats.py`** (our own code, from Facts §4):
  - `format_qm_in(symbols, coords_angstrom, n_states) -> str`;
  - `format_qm_out(energies_hartree, dipole_matrices) -> str`, where `dipole_matrices` is (3, n, n) real; imaginary parts are written as 0;
  - `format_initconds(symbols, atomic_numbers, masses_amu, equilibrium_bohr, positions_bohr, velocities_au) -> str`. It writes the P5 format and leaves `Eharm` and the per-sample energy lines at 0. The existing `tools/sharc_compare/initconds.parse_initconds` must read it back; test that.
  - `parse_initconds_excited(text) -> dict`, with the keys `eref_hartree`, `n_states` and `states` (n, n_states), and per state the rows `{energy_hartree, dipole (3 complex → real parts), delta_e_eV, f, selected: bool}`.
- **`run_excite.py`:**
  - `run_excite(sharc_home, python, work_dir, *, window_eV, seed) -> Path`.
  - It writes the stdin answers of Facts §4 (prompts 1–10, with `window_eV` and `seed`), runs `excite.py` with the same environment handling as `run_wigner.py` (SHARC=<home>/bin; PYTHONPATH replaced), timeout 600, and returns the path of `initconds.excited`.
  - On a nonzero exit or a missing output it raises `SystemExit` with the tail of its output.
- **Selection statistics test** (`test_sharc_excite.py`; no PySCF needed):
  1. Make synthetic excitation data for N = 3000 samples and 2 excited states (n_states = 3), in a fixed-seed generator:
     - ΔE₁ ~ U(4, 6) eV, f₁ ~ U(0, 0.1);
     - ΔE₂ ~ U(5, 9) eV, f₂ ~ U(0, 0.4).
     - Make about 10 % of the S2 entries large (f up to 0.8) with ΔE₂ > 8.5 eV, i.e. **outside** the window [4.0, 8.0]. This makes f_max depend on the window semantics.
     - Build the transition dipoles along x with |μ| = √(3f/(2ΔE)), so that f is reproduced.
  2. Write `initconds` (dummy water positions, equilibrium for all samples), `ICOND_00000` (reference) and `ICOND_00001…ICOND_03000`, each with `QM.in` and `QM.out`, with `format_*`.
  3. Run `excite.py` with window [4.0, 8.0] and seed 7.
  4. Parse the flags. Our rule gives the exact acceptance probability of each pair, p_in = f_in/f_max,window for in-window pairs and 0 otherwise.
  5. Assert, per state, that SHARC's selected count lies within 4 standard deviations of Σ p_in, with variance Σ p(1−p).
  6. Also compute the alternative rules (f_max over all pairs regardless of the window; probability ∝ f/ΔE²) and report which one SHARC's counts are consistent with.
  7. If SHARC is **not** consistent with our rule, do not change our rule (owner decision 1). Mark the test `xfail(strict=True)` with the measured numbers in the reason, document it, and report it to the controller.
  8. Also run **our** `select_bright` on the same candidates with `default_rng(1)`. Assert its per-state counts are within 4σ of the same expectation, and that the ΔE distributions of SHARC's and our selected S1 pairs agree (KS p > 1e-3).
  9. Assert SHARC's ΔE and f columns equal our input (ΔE to 1e-6 eV, f to 1e-6 relative). This validates our writers and units.
- **Excitation energies test** (`test_sharc_excitation_energies.py`; `sharc` and `pyscf`):
  1. Water CAS(4,4)/STO-3G, 3 roots, at the reference geometry (Å) O (0, 0, 0.1501683345), H (0, ±0.7580805047, −0.4856341673), plus 2 displaced geometries. Use `samples.h5`-style geometries from `sample_wigner` with a fixed rng, from our analytic RHF frequencies as in the P5 validation test.
  2. Run `SHARC_PYSCF.py` (extend `run_sharc_pyscf.py` with a function that takes a geometry and a template and returns the parsed QM.out; keep the CLI). Use the same template as Facts §4: `ncas 4`, `nelecas 4`, `roots 3`, `basis sto-3g`, `method casscf`, `conv-tol 1e-10`, `conv-tol-grad 1e-6`.
  3. Compare with `PySCFBackend` using `method = "casscf"`, `ncas = 4`, `nelecas = 4`, `singlets = 3` and tight tolerances. Excitation energies must agree within 1e-5 Eh (strategy §5.3); |μ| and f within 1e-4 relative or 1e-7 absolute (signs are arbitrary).
- [ ] **Step 1: Write the always-run format tests.** Round-trip `format_initconds` through `parse_initconds`. Check the `format_qm_out` text against the observed layout (block headers and `3 3` lines), parsed back with `qmout.parse_qm_out`. Feed `parse_initconds_excited` the two-sample text of Facts §4.
- [ ] **Step 2: Implement** `sharc_formats.py` and `run_excite.py`.
- [ ] **Step 3: Write the two live tests.** Use the skip conditions and markers of the existing SHARC tests.
- [ ] **Step 4: Run.**
  - `pytest tests/sharc_compare -W error`.
  - `SHARC_HOME=/home/ajc/sharc4-4.1 SHARC_PYTHON=/home/ajc/miniforge3/envs/sharc4.1/bin/python pytest -m sharc -v -W error -s`.
  - Report all measured numbers.
  - Clean room: execute only; inspect outputs only in `tmp_path`.
- [ ] **Step 5: Commit**

```bash
git commit -m "test: compare excitations and selection with SHARC 4.1 (black box)"
```

---

### Task 8: Documentation current with P6

**Files:** `CLAUDE.md`, `USER_MANUAL.md`, `docs/superpowers/plans/2026-11-06-gqteashmd-master-plan.md`

- [ ] **Step 1: USER_MANUAL.md.**
  - §4: the `[excitations]` and `[selection]` key tables.
  - §5: a real e2e example, run from a scratch script outside the repository, through `selection`, with the `selection_report.md` excerpt.
  - §6: the artifacts (`excitations.h5` datasets with units, `excitations.json`, `spectrum.csv` columns, `trajectories.json`, `selection_report.json` and `.md`).
  - §8: new [IMPLEMENTED] subsections:
    - vertical excitations: the per-sample procedure, `reset`, failures and `max_failed_fraction`;
    - the nuclear-ensemble spectrum, with the formula, every symbol, the prefactor value, ε, the broadening and the grid;
    - the selection rules: the bright double criterion with f_max over candidates and the order; fixed mode; O20 and `allow_fewer`; seeds (D12);
    - validation numbers from Task 7.
  - §9: the error messages.
  - Known limitations: the workflow now ends at `selection`, so `pause_after = ["selection"]` has no effect until P7.
- [ ] **Step 2: CLAUDE.md.**
  - Status: P6 done on the branch, with the counts.
  - Architecture: `excite/`, the two stages, `tools/sharc_compare/sharc_formats.py` and `run_excite.py`.
  - Rules: one backend per task with full settings and reset per sample; per-sample failures; `trajectories.json` rng streams.
  - Decisions: O3 and O20 resolved.
  - Next: P7a.
- [ ] **Step 3: Master plan.** §4.3 keys as implemented; §5 rows; §9 P6 delivered; §11 O3 and O20 resolved.
- [ ] **Step 4: Commit**

```bash
git commit -m "docs: document excitations, the spectrum and initial-state selection (P6)"
```

---

## Self-review notes

- **Spec coverage:**

  | Spec item | Task |
  |---|---|
  | §1.1 stages 4 and 5 | Tasks 4–6 |
  | §2 review pause (`pause_after`) | report: Task 5; e2e: Task 6 |
  | §3 seeds | Tasks 2, 5 |
  | §4.3 keys | Task 1 |
  | §5 `excitations` row (excitations.h5, spectrum.csv, per-sample status, `max_failed_fraction`) | Tasks 3, 4 |
  | §5 `spectrum.png` | dropped by owner decision 3 |
  | §5 `selection` row (trajectories.json, selection_report.json/.md, O20 stop, labels must exist) | Tasks 1, 5 |
  | §7 `excitations` vs SHARC_PYSCF | Task 7 |
  | §7 `selection` vs excite.py | Task 7 |
  | O3, O20 | owner decisions, Tasks 2, 5 |
  | Strategy §4.8 | Tasks 2, 4, 5 |
  | Docs | Task 8 |

- **Type consistency:**
  - `Candidate` and `Selection` (Task 2) are used in Tasks 5 and 7.
  - `broadened_spectrum` (Task 2) is used in Task 4.
  - `read_samples_h5` and `read_excitations_h5` (Task 3) are used in Tasks 4 and 5.
  - `ExcitationsStage(max_failed_fraction=)` (Task 4) is used in Task 6.
  - `format_*` and `run_excite` are used within Task 7.
