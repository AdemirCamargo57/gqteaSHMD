# P3a: PySCF Electronic-Structure Backend (energies, gradients, transition dipoles) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give gqteashmd a validated electronic-structure layer:
- an `[electronic]` input section;
- a backend contract;
- a PySCF backend that computes energies, analytic gradients and S0→Sn transition dipoles with SA-CASSCF, TDA or TDDFT;
- a `gqteashmd single-point` command;
- validation of ethylene SA2-CASSCF(2,2)/6-31G\* against SHARC 4.1 (run as a black box) and finite differences.

**Architecture:**
- `gqteashmd.electronic` holds the backend contract (`ElectronicBackend`, `Request`, `ElectronicResult`). PySCF is imported only inside `gqteashmd.electronic.pyscf_backend`, and lazily, so the rest of the program never depends on it.
- All quantities are atomic units internally: bohr, hartree, e·a₀. Conversions happen at the input/output edge through `gqteashmd.core.units` (CODATA 2018).
- SA-CASSCF uses PySCF's singlet-only FCI solver plus a ⟨S²⟩ guard. Orbitals from a previous geometry are reused only through `mcscf.project_init_guess`.
- Nonadiabatic couplings, wavefunction overlaps and phase tracking are **P3b**. The Hessian moves to **P4**, where the frequencies stage uses it.

**Tech Stack:** Python 3.12, pydantic v2, NumPy, PySCF 2.14 (pinned in `environment.yml`), pytest, ruff 0.17.4, mypy.

**Spec:**
- Master plan [`2026-11-06-gqteashmd-master-plan.md`](2026-11-06-gqteashmd-master-plan.md): §2, §4, §5, §6, D4, D7, D13, D19, O18, O19.
- Strategy [`SHMD_implementation_strategy.md`](../../../SHMD_implementation_strategy.md): §2.3 backend contract, §3.1 PySCF, §3.4 method validity, §5.3 tolerances.
- [`CLAUDE.md`](../../../CLAUDE.md): rules and P2 deferred items.

## Global Constraints

- **Clean room:** no code from SHARC 4.1 is copied, translated or adapted. SHARC may only be *executed* as a black box from `tools/sharc_compare/`; its output files are parsed by our own code. The licence is MIT.
- **Runtime environment:** Linux only (target Oracle Linux 9). All commands run inside WSL Oracle Linux 9.5:
  `wsl -d OracleLinux_9_5 -- bash -lc 'cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && conda activate gqteashmd-dev && <command>'`.
  Create and edit files with the file tools at Windows paths under `C:\Users\ajc\OneDrive\gqteaSHMD\`, with LF line endings.
  Scratch work goes in `mktemp -d /tmp/...`, never in `$HOME` or the repository. Run one pytest process at a time.
- **Python and dependencies:**
  - Python 3.12. Runtime dependencies: `pydantic>=2.6`, `numpy>=1.26`, `pyscf>=2.6`. The development environment pins `pyscf=2.14.*`.
  - **PySCF is imported only inside functions and methods of `gqteashmd/electronic/pyscf_backend.py`.** No other module imports pyscf at module level.
- **Units:** coordinates are in bohr inside the program and in Å in XYZ files. Energies are in hartree (eV only for display). Gradients are in hartree/bohr. Dipoles are in e·a₀ (atomic units). Constants come from `gqteashmd.core.units`.
- **States:** labels are `S0`, `S1`, …, `T1`, …, 0-based for singlets. `electronic.singlets` counts states **including S0** (O19, decided by the owner on 2026-11-07). `electronic.triplets` must be 0 in P3a. The backend supports closed-shell singlet molecules only (`molecule.multiplicity = 1`).
- **Errors:**
  - User-caused input problems → `ConfigError` (exit 2).
  - Electronic-structure failures (no convergence, wrong spin, unusable basis or functional) → `ElectronicStructureError`, a non-retryable `TaskError` with exit code 4.
- **Style and checks:**
  - ruff 0.17.4: line length 100, rules E, F, W, I, B, UP, SIM.
  - mypy must stay clean on `src/`; pyscf has no type hints and is ignored through an override.
  - Type hints on all public functions.
- **Git:**
  - Branch `feat/p3a-pyscf-backend`, using WSL Git only.
  - Pre-commit hooks are installed; never use `--no-verify`.
  - Use conventional-commit subjects, and end each message with the `Co-Authored-By:` trailer for the model that wrote it.
  - Do not push.
- **Documentation:** the last task updates CLAUDE.md, USER_MANUAL.md and the master plan to match the code. USER_MANUAL.md keeps [IMPLEMENTED] and [PLANNED] apart and defines every symbol and unit.

## Facts verified before writing this plan (PySCF 2.14.0, WSL, 2026-11-07)

1. **The default solver finds a triplet.** `mcscf.CASSCF(mf, 2, 2).state_average_([0.5, 0.5])` with the default FCI solver converges to S0 plus the **triplet** ³ππ\* (⟨S²⟩ = 2.0) for ethylene.
2. **SA3 needs the singlet-only solver.** `fix_spin_(ss=0, shift=0.2)` fixes SA2 but not SA3, where the third root is still a triplet. `fci.solver(mol, singlet=True)` gives pure singlets for SA2 and SA3. The energies are identical to the `fix_spin_` SA2 result, and gradients and couplings work with it.
3. **SA2 ethylene reference values.** Ethylene SA2-CASSCF(2,2)/6-31G\*, singlet solver:
   - energies −78.04902446 and −77.67250833 Eh;
   - μ(S0→S1) = (−1.52976, −0.05399, ~0) a.u.;
   - max|∇E(S1)| = 0.12667 Eh/bohr.

   SHARC 4.1 `SHARC_PYSCF.py`, with `fix-spin-shift 0.2`, `conv-tol 1e-10` and `conv-tol-grad 1e-6`, gives −78.04902443 and −77.67250836 Eh and the same dipole and gradients. The reference file in Task 6 holds the full numbers.
4. **Analytic gradients agree with finite differences** (central differences, h = 10⁻⁴ bohr):
   - SA-CASSCF to ≈5·10⁻⁸ Eh/bohr, starting each displaced geometry from a fresh RHF guess;
   - TDA-B3LYP/STO-3G to 5·10⁻⁷.

   **Reusing the previous geometry's MO coefficients without projection gives wrong energies, with a first-order error.** `mcscf.project_init_guess(mc, mo_prev, mol_prev)` gives the same energies as a fresh start, to 10⁻⁸ Eh after a 0.05 bohr step.
5. **Other PySCF behaviour:**
   - An unknown basis raises `pyscf.lib.exceptions.BasisNotFoundError` (a `RuntimeError`), preceded by a `UserWarning` ("Basis may be available in basis-set-exchange"), which `-W error` turns into an exception.
   - `scf.RHF(...)` with `max_cycle = 1` does not converge, even with level shift, damping or `.newton()`.
   - The SA-CASSCF gradient object has `.converged`. The TDA gradient object has none.
   - `td.converged` is a bool array.
   - `td.oscillator_strength()` equals ⅔ ΔE |μ|².
   - `mol.stdout` set before `mol.build()` sends all PySCF output, including SCF and CASSCF, to that file.
6. **Timing** (4 threads): ethylene SA2-CASSCF/6-31G\* energy 0.3 s, plus each gradient ≈1 s. TDA-B3LYP/def2-SVP 2 roots ≈3 s, plus each gradient ≈2.6 s. STO-3G and 6-31G versions are about 10× faster and are used in unit tests.
7. **CI passed** on GitHub (Oracle Linux 9 container) at `e73609f`.

## File Structure

```text
src/gqteashmd/preflight.py                 # Task 1: inaccessible ancestor → ConfigError
pyproject.toml, environment.yml            # Task 2: numpy/pyscf deps, pin, mypy override, sharc marker (Task 10)
src/gqteashmd/core/__init__.py             # Task 2 (new, empty)
src/gqteashmd/core/units.py                # Task 2 (new)
src/gqteashmd/core/states.py               # Task 2 (new): StateLabel, states_from_counts
src/gqteashmd/chem/electrons.py            # Task 3 (new): count_electrons
src/gqteashmd/stages/prepare.py            # Task 3: use count_electrons
src/gqteashmd/config/schema.py             # Task 4: ElectronicSection, CasscfSection, closed-shell check
src/gqteashmd/cli.py                       # Tasks 4, 9: template block; single-point command
src/gqteashmd/errors.py                    # Task 5: ElectronicStructureError
src/gqteashmd/electronic/__init__.py       # Task 5 (new, empty)
src/gqteashmd/electronic/base.py           # Task 5 (new): MoleculeSpec, Request, ElectronicResult, ElectronicBackend
src/gqteashmd/electronic/properties.py     # Task 5 (new): oscillator_strength
src/gqteashmd/electronic/pyscf_backend.py  # Tasks 6, 7 (new)
src/gqteashmd/electronic/factory.py        # Task 9 (new): make_backend
tools/sharc_compare/qmout.py               # Task 10 (new): QM.out parser (our own code)
tools/sharc_compare/run_sharc_pyscf.py     # Task 10 (new): runs SHARC_PYSCF.py as a black box
tests/data/ethylene.xyz                    # Task 6 (new)
tests/data/reference/sharc41_pyscf_ethylene_sa2cas22_631gs.json   # Task 6 (new)
tests/test_preflight.py                    # Task 1
tests/core/test_units.py, tests/core/test_states.py               # Task 2 (new)
tests/chem/test_electrons.py               # Task 3 (new)
tests/config/test_electronic.py            # Task 4 (new)
tests/electronic/test_base.py              # Task 5 (new)
tests/electronic/conftest.py               # Task 6 (new): ethylene fixtures, settings helper
tests/electronic/test_pyscf_casscf.py      # Task 6 (new)
tests/electronic/test_pyscf_linear_response.py   # Task 7 (new)
tests/electronic/test_gradients_fd.py      # Task 8 (new, slow)
tests/test_cli_single_point.py             # Task 9 (new)
tests/sharc_compare/test_qmout.py, tests/sharc_compare/test_live_sharc.py   # Task 10 (new)
CLAUDE.md, USER_MANUAL.md, docs/superpowers/plans/2026-11-06-gqteashmd-master-plan.md   # Task 11
```

---

### Task 1: An inaccessible ancestor directory is a configuration error

The P2 residual: if a folder above `project.directory` has no search permission, `Path.exists()` raises `PermissionError` and preflight shows "internal error" (exit 1).

**Files:**
- Modify: `src/gqteashmd/preflight.py`
- Test: `tests/test_preflight.py`

**Interfaces:**
- Consumes: `run_preflight(config, stages, *, machine=None)` and `_with_directory(cfg, path)` (an existing helper in the test file).
- Produces: preflight never raises `OSError` for a project path. An unsearchable ancestor leads to the existing "Cannot create or write the project directory … (no write permission in …)" `ConfigError`.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_preflight.py`:

```python
import os

import pytest


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_unsearchable_ancestor_is_a_config_error(tmp_path, write_input):
    locked = tmp_path / "locked"
    locked.mkdir()
    cfg = _with_directory(load_config(write_input(tmp_path)).config, locked / "sub" / "run")
    locked.chmod(0o000)
    try:
        with pytest.raises(ConfigError, match="Cannot create or write the project directory"):
            run_preflight(cfg, [PrepareStage()], machine=MachineInfo(None, None))
    finally:
        locked.chmod(0o755)
```

(Merge the imports with the existing ones at the top of the file. `os` and `pytest` may already be imported.)

- [ ] **Step 2: Run it to verify it fails.**
  Run: `pytest tests/test_preflight.py -k unsearchable -v`
  Expected: FAIL with `PermissionError`.

- [ ] **Step 3: Implement.** In `src/gqteashmd/preflight.py`, add the helper below and use it in `_nearest_existing` and in the not-a-directory check:

```python
def _exists(path: Path) -> bool:
    """Path.exists() that treats a path inside an unsearchable directory as non-existent."""
    try:
        return path.exists()
    except OSError:
        return False


def _nearest_existing(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if _exists(candidate):
            return candidate
    return Path("/")
```

In `run_preflight`, replace `if project_dir.exists() and not project_dir.is_dir():` with `if _exists(project_dir) and not project_dir.is_dir():`.

- [ ] **Step 4: Run the tests to verify they pass.**
  Run: `pytest tests/test_preflight.py -v`
  Expected: all pass.

- [ ] **Step 5: Lint, type-check, commit**

```bash
ruff format src tests && ruff check --fix src tests && mypy
git add src/gqteashmd/preflight.py tests/test_preflight.py
git commit -m "fix: treat an inaccessible ancestor of the project directory as a configuration error"
```

---

### Task 2: Dependencies, physical constants and state labels

**Files:**
- Modify: `pyproject.toml`, `environment.yml`
- Create: `src/gqteashmd/core/__init__.py` (empty), `src/gqteashmd/core/units.py`, `src/gqteashmd/core/states.py`
- Test: `tests/core/test_units.py`, `tests/core/test_states.py`

**Interfaces:**
- Produces:
  - **Constants:** `BOHR_IN_ANGSTROM`, `ANGSTROM_TO_BOHR`, `HARTREE_IN_EV`, `HARTREE_IN_CM1`, `DALTON_IN_ME`, `AU_TIME_IN_FS`.
  - **`StateLabel(multiplicity: int, index: int)`:** frozen and ordered, so S states sort before T states. It provides `StateLabel.parse(text)` and `str(label)` → `"S1"`.
  - **`states_from_counts(singlets: int, triplets: int = 0) -> tuple[StateLabel, ...]`.**

- [ ] **Step 1: Dependencies.**
  - In `pyproject.toml`, set `dependencies = ["pydantic>=2.6", "numpy>=1.26", "pyscf>=2.6"]`.
  - In `environment.yml`, replace the `- pyscf` line with `- pyscf=2.14.*` and keep the comment.
  - Append to `pyproject.toml`:

```toml
[[tool.mypy.overrides]]
module = ["pyscf", "pyscf.*"]
ignore_missing_imports = true
```

- [ ] **Step 2: Write the failing tests.**

`tests/core/test_units.py`:

```python
import math

from gqteashmd.core import units


def test_codata_2018_values():
    assert units.BOHR_IN_ANGSTROM == 0.529177210903
    assert units.HARTREE_IN_EV == 27.211386245988
    assert units.HARTREE_IN_CM1 == 219474.6313632
    assert units.DALTON_IN_ME == 1822.888486209
    assert units.AU_TIME_IN_FS == 0.024188843265857


def test_angstrom_to_bohr_is_the_inverse():
    assert math.isclose(units.ANGSTROM_TO_BOHR * units.BOHR_IN_ANGSTROM, 1.0, rel_tol=1e-15)
```

`tests/core/test_states.py`:

```python
import pytest

from gqteashmd.core.states import StateLabel, states_from_counts

S0, S1, S2 = StateLabel(1, 0), StateLabel(1, 1), StateLabel(1, 2)
T1, T2 = StateLabel(3, 1), StateLabel(3, 2)


def test_parse_and_format():
    assert StateLabel.parse("S0") == S0
    assert StateLabel.parse(" t1 ") == T1
    assert str(T2) == "T2"
    assert str(S1) == "S1"


@pytest.mark.parametrize("bad", ["", "X1", "S", "S-1", "T0", "Q1", "S1.5", "S01x"])
def test_invalid_labels(bad):
    with pytest.raises(ValueError):
        StateLabel.parse(bad)


def test_invalid_construction():
    with pytest.raises(ValueError):
        StateLabel(2, 0)
    with pytest.raises(ValueError):
        StateLabel(3, 0)
    with pytest.raises(ValueError):
        StateLabel(1, -1)


def test_states_from_counts_include_s0():
    assert states_from_counts(3) == (S0, S1, S2)
    assert states_from_counts(2, 2) == (S0, S1, T1, T2)
    with pytest.raises(ValueError, match="S0"):
        states_from_counts(0)


def test_labels_sort_singlets_first():
    assert sorted([T1, S1, S0]) == [S0, S1, T1]
```

- [ ] **Step 3: Run the tests to verify they fail.**
  Run: `pytest tests/core -v`
  Expected: FAIL with `ModuleNotFoundError: No module named 'gqteashmd.core'`.

- [ ] **Step 4: Implement.**

`src/gqteashmd/core/units.py`:

```python
"""Physical constants and unit conversions (CODATA 2018).

gqteashmd works in Hartree atomic units internally (bohr, hartree, electron mass, e·a0,
hbar/E_h); conversions happen only when files are read or written.
"""

from __future__ import annotations

BOHR_IN_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_IN_ANGSTROM
HARTREE_IN_EV = 27.211386245988
HARTREE_IN_CM1 = 219474.6313632
DALTON_IN_ME = 1822.888486209
AU_TIME_IN_FS = 0.024188843265857
```

`src/gqteashmd/core/states.py`:

```python
"""Electronic state labels: S0, S1, ... (singlets, 0-based) and T1, T2, ... (triplets)."""

from __future__ import annotations

import re
from dataclasses import dataclass

_PREFIX = {1: "S", 3: "T"}
_MULTIPLICITY = {prefix: mult for mult, prefix in _PREFIX.items()}
_PATTERN = re.compile(r"([ST])(\d+)")


@dataclass(frozen=True, order=True)
class StateLabel:
    multiplicity: int
    index: int

    def __post_init__(self) -> None:
        if self.multiplicity not in _PREFIX:
            raise ValueError(f"Unsupported multiplicity {self.multiplicity}; use 1 (S) or 3 (T).")
        lowest = 0 if self.multiplicity == 1 else 1
        if self.index < lowest:
            raise ValueError(f"{_PREFIX[self.multiplicity]}{self.index} is not a valid state.")

    def __str__(self) -> str:
        return f"{_PREFIX[self.multiplicity]}{self.index}"

    @classmethod
    def parse(cls, text: str) -> StateLabel:
        match = _PATTERN.fullmatch(text.strip().upper())
        if match is None:
            raise ValueError(f"Invalid state label {text!r}; use e.g. S0, S1, T1.")
        return cls(_MULTIPLICITY[match.group(1)], int(match.group(2)))


def states_from_counts(singlets: int, triplets: int = 0) -> tuple[StateLabel, ...]:
    """States S0..S(singlets-1) followed by T1..T(triplets); `singlets` includes S0."""
    if singlets < 1:
        raise ValueError("At least one singlet state (S0) is required.")
    return tuple(StateLabel(1, i) for i in range(singlets)) + tuple(
        StateLabel(3, i) for i in range(1, triplets + 1)
    )
```

- [ ] **Step 5: Run the tests to verify they pass.**
  Run: `pytest tests/core -v && mypy`
  Expected: all pass; mypy prints `Success`.

- [ ] **Step 6: Lint, full suite, commit**

```bash
ruff format src tests && ruff check --fix src tests && pytest -m "not slow" -W error
git add pyproject.toml environment.yml src/gqteashmd/core tests/core
git commit -m "feat: add CODATA 2018 units, state labels and the numpy/pyscf dependencies"
```

---

### Task 3: One electron-count rule for `prepare` and the backend

**Files:**
- Create: `src/gqteashmd/chem/electrons.py`
- Modify: `src/gqteashmd/stages/prepare.py`
- Test: `tests/chem/test_electrons.py`

**Interfaces:**
- Produces: `count_electrons(atomic_numbers: Sequence[int], charge: int, multiplicity: int) -> int`. It raises `ValueError("Charge q and multiplicity M are inconsistent with N electrons.")`.
- `PrepareStage.run_task` keeps its exact message, which is that text followed by " Check [molecule] in the input file.", and stays non-retryable.

- [ ] **Step 1: Write the failing test** in `tests/chem/test_electrons.py`:

```python
import pytest

from gqteashmd.chem.electrons import count_electrons


@pytest.mark.parametrize(
    "numbers, charge, mult, expected",
    [([8, 1, 1], 0, 1, 10), ([8, 1, 1], 1, 2, 9), ([8, 8], 0, 3, 16), ([6, 6, 1, 1, 1, 1], 0, 1, 16)],
)
def test_valid_electron_counts(numbers, charge, mult, expected):
    assert count_electrons(numbers, charge, mult) == expected


@pytest.mark.parametrize(
    "numbers, charge, mult",
    [([8, 1, 1], 0, 2), ([8, 1, 1], 1, 1), ([8, 1, 1], 0, 12), ([8, 1, 1], 10, 1)],
)
def test_inconsistent_counts(numbers, charge, mult):
    with pytest.raises(ValueError, match="inconsistent with"):
        count_electrons(numbers, charge, mult)
```

- [ ] **Step 2: Run it to verify it fails.**
  Run: `pytest tests/chem/test_electrons.py -v`
  Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `src/gqteashmd/chem/electrons.py`:

```python
"""Electron count and its consistency with charge and spin multiplicity."""

from __future__ import annotations

from collections.abc import Sequence


def count_electrons(atomic_numbers: Sequence[int], charge: int, multiplicity: int) -> int:
    """N_e = sum(Z) - q; requires N_e > 0, M - 1 <= N_e and N_e ≡ M - 1 (mod 2)."""
    n_electrons = sum(atomic_numbers) - charge
    unpaired = multiplicity - 1
    if n_electrons <= 0 or unpaired > n_electrons or n_electrons % 2 != unpaired % 2:
        raise ValueError(
            f"Charge {charge} and multiplicity {multiplicity} are inconsistent with "
            f"{n_electrons} electrons."
        )
    return n_electrons
```

In `src/gqteashmd/stages/prepare.py`, replace the inline `n_electrons = …` / `unpaired = …` / `if …: raise TaskError(...)` block in `run_task` with:

```python
        try:
            n_electrons = count_electrons(numbers, molecule.charge, molecule.multiplicity)
        except ValueError as exc:
            raise TaskError(f"{exc} Check [molecule] in the input file.", retryable=False) from exc
```

Add `from gqteashmd.chem.electrons import count_electrons` to the imports.

- [ ] **Step 4: Run the tests to verify they pass.**
  Run: `pytest tests/chem tests/stages -v`
  Expected: all pass, including the existing prepare tests that match "inconsistent".

- [ ] **Step 5: Lint, type-check, commit**

```bash
ruff format src tests && ruff check --fix src tests && mypy
git add src/gqteashmd/chem/electrons.py src/gqteashmd/stages/prepare.py tests/chem/test_electrons.py
git commit -m "refactor: share the electron-count rule between prepare and the backend"
```

---

### Task 4: The `[electronic]` input section

**Files:**
- Modify: `src/gqteashmd/config/schema.py`, `src/gqteashmd/cli.py` (`TEMPLATE`)
- Test: `tests/config/test_electronic.py`

**Interfaces:**
- Produces:
  - **`CasscfSection(ncas, nelecas, conv_tol=1e-10, conv_tol_grad=1e-6)`.**
  - **`ElectronicSection`**, with fields:
    - `method` ("casscf" | "tda" | "tddft");
    - `basis`;
    - `functional` (str | None);
    - `singlets`;
    - `triplets = 0`;
    - `grid_level = 3`;
    - `scf_conv_tol = 1e-10`;
    - `scf_max_cycle = 100`;
    - `td_conv_tol = 1e-8`;
    - `casscf` (CasscfSection | None).
  - **`ShmdConfig.electronic: ElectronicSection | None = None`.**
  - **Validators:**
    - `casscf` requires `[electronic.casscf]` and no `functional`.
    - `tda`/`tddft` require `functional` and no `[electronic.casscf]`.
    - `triplets` must be 0.
    - `nelecas` must be even and ≤ 2·`ncas`.
    - With `[electronic]` present, `molecule.multiplicity` must be 1.

- [ ] **Step 1: Write the failing tests** in `tests/config/test_electronic.py`:

```python
import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import ConfigError

CAS = (
    '[electronic]\nmethod = "casscf"\nbasis = "6-31g*"\nsinglets = 2\n'
    "[electronic.casscf]\nncas = 2\nnelecas = 2\n"
)
TDA = '[electronic]\nmethod = "tda"\nbasis = "def2-svp"\nfunctional = "b3lyp"\nsinglets = 3\n'


def _load(tmp_path, write_input, extra, molecule=""):
    return load_config(write_input(tmp_path, extra=extra, molecule=molecule)).config


def test_section_is_optional(tmp_path, write_input):
    assert _load(tmp_path, write_input, "").electronic is None


def test_casscf_section_with_defaults(tmp_path, write_input):
    el = _load(tmp_path, write_input, CAS).electronic
    assert (el.method, el.basis, el.singlets, el.triplets) == ("casscf", "6-31g*", 2, 0)
    assert (el.casscf.ncas, el.casscf.nelecas) == (2, 2)
    assert el.casscf.conv_tol == 1e-10 and el.casscf.conv_tol_grad == 1e-6
    assert el.scf_conv_tol == 1e-10 and el.scf_max_cycle == 100


def test_tda_section(tmp_path, write_input):
    el = _load(tmp_path, write_input, TDA).electronic
    assert (el.method, el.functional, el.grid_level, el.td_conv_tol) == ("tda", "b3lyp", 3, 1e-8)


@pytest.mark.parametrize(
    "extra, message",
    [
        ('[electronic]\nmethod = "casscf"\nbasis = "6-31g*"\nsinglets = 2\n', r"\[electronic\.casscf\]"),
        (CAS.replace('singlets = 2\n', 'singlets = 2\nfunctional = "b3lyp"\n'), "functional is not used"),
        ('[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nsinglets = 2\n', "electronic.functional is required"),
        (TDA + "[electronic.casscf]\nncas = 2\nnelecas = 2\n", "only used by method = 'casscf'"),
        (TDA.replace("singlets = 3", "singlets = 3\ntriplets = 1"), "triplets"),
        (CAS.replace("nelecas = 2", "nelecas = 3"), "even"),
        (CAS.replace("nelecas = 2", "nelecas = 6"), "at most 2"),
        (TDA.replace('"tda"', '"cc2"'), "method"),
        (TDA.replace("singlets = 3", "singlets = 0"), "singlets"),
    ],
)
def test_invalid_sections(tmp_path, write_input, extra, message):
    with pytest.raises(ConfigError, match=message):
        _load(tmp_path, write_input, extra)


def test_open_shell_molecule_is_refused(tmp_path, write_input):
    with pytest.raises(ConfigError, match="closed-shell singlet"):
        _load(tmp_path, write_input, CAS, molecule="multiplicity = 3")
```

- [ ] **Step 2: Run them to verify they fail.**
  Run: `pytest tests/config/test_electronic.py -v`
  Expected: FAIL, because `electronic` is an unknown section.

- [ ] **Step 3: Implement** in `src/gqteashmd/config/schema.py`. Add these classes before `ShmdConfig`:

```python
class CasscfSection(_Section):
    ncas: int = Field(ge=1)
    nelecas: int = Field(ge=1)
    conv_tol: float = Field(default=1e-10, gt=0)
    conv_tol_grad: float = Field(default=1e-6, gt=0)

    @model_validator(mode="after")
    def _singlet_active_space(self) -> CasscfSection:
        if self.nelecas % 2:
            raise ValueError("electronic.casscf.nelecas must be even for singlet states")
        if self.nelecas > 2 * self.ncas:
            raise ValueError(
                f"electronic.casscf.nelecas = {self.nelecas} must be at most 2 × ncas = "
                f"{2 * self.ncas}"
            )
        return self


class ElectronicSection(_Section):
    method: Literal["casscf", "tda", "tddft"]
    basis: str = Field(min_length=1)
    functional: str | None = None
    singlets: int = Field(ge=1)
    triplets: int = Field(default=0, ge=0)
    grid_level: int = Field(default=3, ge=0, le=9)
    scf_conv_tol: float = Field(default=1e-10, gt=0)
    scf_max_cycle: int = Field(default=100, ge=1)
    td_conv_tol: float = Field(default=1e-8, gt=0)
    casscf: CasscfSection | None = None

    @model_validator(mode="after")
    def _method_options(self) -> ElectronicSection:
        if self.triplets:
            raise ValueError(
                "electronic.triplets > 0 is not supported yet (planned together with "
                "intersystem crossing); set triplets = 0"
            )
        if self.method == "casscf":
            if self.casscf is None:
                raise ValueError(
                    "method = 'casscf' needs an [electronic.casscf] table with ncas and nelecas"
                )
            if self.functional is not None:
                raise ValueError("electronic.functional is not used by method = 'casscf'")
        else:
            if self.functional is None:
                raise ValueError(
                    f"electronic.functional is required for method = '{self.method}' "
                    "(use 'hf' for CIS/TDHF)"
                )
            if self.casscf is not None:
                raise ValueError("[electronic.casscf] is only used by method = 'casscf'")
        return self
```

In `ShmdConfig`:
- add the field `electronic: ElectronicSection | None = None`;
- add the validator below;
- change the module docstring to say "Input file schema".

```python
    @model_validator(mode="after")
    def _closed_shell_for_electronic_structure(self) -> ShmdConfig:
        if self.electronic is not None and self.molecule.multiplicity != 1:
            raise ValueError(
                "the PySCF backend supports closed-shell singlet molecules only "
                "(molecule.multiplicity = 1)"
            )
        return self
```

In `src/gqteashmd/cli.py`, append to `TEMPLATE`, just before the closing `"""`:

```toml

# Electronic structure (needed by the quantum-chemistry stages and `gqteashmd single-point`):
# [electronic]
# method = "casscf"             # "casscf" | "tda" | "tddft"
# basis = "6-31g*"
# singlets = 2                  # number of singlet states INCLUDING S0
# [electronic.casscf]
# ncas = 2                      # active orbitals
# nelecas = 2                   # active electrons
```

- [ ] **Step 4: Run the tests to verify they pass.**
  Run: `pytest tests/config tests/test_cli.py -v && mypy`
  Expected: all pass. The template stays valid TOML.

- [ ] **Step 5: Lint, full suite, commit**

```bash
ruff format src tests && ruff check --fix src tests && pytest -m "not slow" -W error
git add src/gqteashmd/config/schema.py src/gqteashmd/cli.py tests/config/test_electronic.py
git commit -m "feat: add the [electronic] input section"
```

---

### Task 5: Backend contract, error type and oscillator strength

**Files:**
- Modify: `src/gqteashmd/errors.py`
- Create: `src/gqteashmd/electronic/__init__.py` (empty), `src/gqteashmd/electronic/base.py`, `src/gqteashmd/electronic/properties.py`
- Test: `tests/electronic/test_base.py`

**Interfaces:**
- Produces:
  - **`ElectronicStructureError(TaskError)`:** `exit_code = 4`, default `retryable=False`.
  - **`MoleculeSpec(symbols: tuple[str, ...], charge: int = 0, multiplicity: int = 1)`.**
  - **`Request(coords_bohr, gradients: frozenset[StateLabel] = frozenset(), transition_dipoles: bool = False)`:** `coords_bohr` is coerced to a float array of shape (natom, 3).
  - **`ElectronicResult(states, energies, gradients={}, transition_dipoles={}, spin_squares={}, meta={})`:** it provides `excitation_energy(label)` and `to_json()`.
  - **`ElectronicBackend`:** an ABC with the property `states` and the method `compute(request) -> ElectronicResult`.
  - **`oscillator_strength(excitation_energy, transition_dipole) -> float`.**

- [ ] **Step 1: Write the failing tests** in `tests/electronic/test_base.py`:

```python
import math

import numpy as np
import pytest

from gqteashmd.core.states import StateLabel
from gqteashmd.electronic.base import ElectronicResult, Request
from gqteashmd.electronic.properties import oscillator_strength
from gqteashmd.errors import ElectronicStructureError, TaskError

S0, S1 = StateLabel(1, 0), StateLabel(1, 1)


def test_error_type():
    err = ElectronicStructureError("SCF did not converge")
    assert isinstance(err, TaskError)
    assert err.retryable is False and err.exit_code == 4


def test_request_coerces_and_checks_shape():
    req = Request([[0, 0, 0], [0, 0, 1.4]])
    assert req.coords_bohr.dtype == float and req.coords_bohr.shape == (2, 3)
    with pytest.raises(ValueError, match="natom, 3"):
        Request([0.0, 0.0, 1.0])


def test_oscillator_strength_formula():
    assert math.isclose(oscillator_strength(0.5, np.array([1.0, 0.0, 0.0])), 1.0 / 3.0)
    assert oscillator_strength(0.3, np.zeros(3)) == 0.0


def test_result_json():
    result = ElectronicResult(
        states=(S0, S1),
        energies={S0: -1.0, S1: -0.5},
        gradients={S1: np.ones((2, 3))},
        transition_dipoles={S1: np.array([1.0, 0.0, 0.0])},
        spin_squares={S0: 0.0, S1: 0.0},
        meta={"method": "casscf"},
    )
    assert result.excitation_energy(S1) == 0.5
    data = result.to_json()
    assert data["schema_version"] == 1
    s0, s1 = data["states"]
    assert s0["label"] == "S0" and s0["oscillator_strength"] is None
    assert s1["excitation_ev"] == pytest.approx(0.5 * 27.211386245988)
    assert s1["oscillator_strength"] == pytest.approx(1.0 / 3.0)
    assert data["gradients_hartree_per_bohr"]["S1"] == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert data["meta"] == {"method": "casscf"}
```

- [ ] **Step 2: Run them to verify they fail.**
  Run: `pytest tests/electronic/test_base.py -v`
  Expected: FAIL with an `ImportError`.

- [ ] **Step 3: Implement.**

Append to `src/gqteashmd/errors.py`:

```python
class ElectronicStructureError(TaskError):
    """An electronic-structure calculation failed (no convergence, wrong spin, bad basis, ...)."""

    exit_code = 4

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message, retryable=retryable)
```

`src/gqteashmd/electronic/properties.py`:

```python
"""Derived electronic properties (atomic units)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def oscillator_strength(excitation_energy: float, transition_dipole: ArrayLike) -> float:
    """Length-gauge oscillator strength f = (2/3) ΔE |μ|² (ΔE in hartree, μ in e·a0)."""
    mu = np.asarray(transition_dipole, dtype=float)
    return float(2.0 / 3.0 * excitation_energy * float(np.dot(mu, mu)))
```

`src/gqteashmd/electronic/base.py`:

```python
"""Electronic-structure backend contract. All quantities in atomic units."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from gqteashmd.core.states import StateLabel
from gqteashmd.core.units import HARTREE_IN_EV
from gqteashmd.electronic.properties import oscillator_strength


@dataclass(frozen=True)
class MoleculeSpec:
    symbols: tuple[str, ...]
    charge: int = 0
    multiplicity: int = 1


@dataclass(frozen=True, eq=False)
class Request:
    """One electronic-structure evaluation at a geometry (coordinates in bohr)."""

    coords_bohr: np.ndarray
    gradients: frozenset[StateLabel] = frozenset()
    transition_dipoles: bool = False

    def __init__(
        self,
        coords_bohr: ArrayLike,
        gradients: frozenset[StateLabel] = frozenset(),
        transition_dipoles: bool = False,
    ) -> None:
        coords = np.asarray(coords_bohr, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords_bohr must have shape (natom, 3), got {coords.shape}")
        object.__setattr__(self, "coords_bohr", coords)
        object.__setattr__(self, "gradients", frozenset(gradients))
        object.__setattr__(self, "transition_dipoles", transition_dipoles)


@dataclass(frozen=True, eq=False)
class ElectronicResult:
    """Energies (Eh), gradients (Eh/bohr), S0→state transition dipoles (e·a0), ⟨S²⟩."""

    states: tuple[StateLabel, ...]
    energies: Mapping[StateLabel, float]
    gradients: Mapping[StateLabel, np.ndarray] = field(default_factory=dict)
    transition_dipoles: Mapping[StateLabel, np.ndarray] = field(default_factory=dict)
    spin_squares: Mapping[StateLabel, float] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)

    def excitation_energy(self, label: StateLabel) -> float:
        return self.energies[label] - self.energies[self.states[0]]

    def to_json(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for label in self.states:
            delta = self.excitation_energy(label)
            tdip = self.transition_dipoles.get(label)
            rows.append(
                {
                    "label": str(label),
                    "energy_hartree": float(self.energies[label]),
                    "excitation_ev": delta * HARTREE_IN_EV,
                    "transition_dipole_au": None if tdip is None else [float(x) for x in tdip],
                    "oscillator_strength": None if tdip is None else oscillator_strength(delta, tdip),
                    "spin_square": self.spin_squares.get(label),
                }
            )
        return {
            "schema_version": 1,
            "states": rows,
            "gradients_hartree_per_bohr": {
                str(label): np.asarray(g, dtype=float).tolist()
                for label, g in sorted(self.gradients.items())
            },
            "meta": dict(self.meta),
        }


class ElectronicBackend(ABC):
    @property
    @abstractmethod
    def states(self) -> tuple[StateLabel, ...]: ...

    @abstractmethod
    def compute(self, request: Request) -> ElectronicResult: ...
```

(`Request` defines its own `__init__` so that it can coerce the array. Keep `@dataclass(frozen=True, eq=False)` for the repr. If the decorator complains about the custom `__init__`, use `@dataclass(frozen=True, eq=False, init=False)`.)

- [ ] **Step 4: Run the tests to verify they pass.**
  Run: `pytest tests/electronic/test_base.py -v && mypy`
  Expected: all pass, and mypy prints `Success`.

- [ ] **Step 5: Lint, full suite, commit**

```bash
ruff format src tests && ruff check --fix src tests && pytest -m "not slow" -W error
git add src/gqteashmd/errors.py src/gqteashmd/electronic tests/electronic/test_base.py
git commit -m "feat: add the electronic-structure backend contract"
```

---

### Task 6: PySCF backend: SCF, SA-CASSCF, gradients, transition dipoles

**Files:**
- Create:
  - `src/gqteashmd/electronic/pyscf_backend.py`
  - `tests/data/ethylene.xyz`
  - `tests/data/reference/sharc41_pyscf_ethylene_sa2cas22_631gs.json`
  - `tests/electronic/conftest.py`
  - `tests/electronic/test_pyscf_casscf.py`

**Interfaces:**
- Consumes:
  - `MoleculeSpec`, `Request`, `ElectronicResult`, `ElectronicBackend` (Task 5);
  - `ElectronicSection` (Task 4);
  - `states_from_counts` (Task 2);
  - `ElectronicStructureError`, `ConfigError`.
- Produces:
  - `MIN_PYSCF_VERSION = (2, 6, 0)`;
  - `require_pyscf() -> str`, which returns the version or raises `ConfigError`;
  - `PySCFBackend(molecule, settings, *, log_file: Path | None = None, threads: int | None = None, memory_mb: int | None = None)`. Its `compute(request)` supports `method = "casscf"` in this task. `tda`/`tddft` come in Task 7 and raise `NotImplementedError` until then.
- Test fixtures (`tests/electronic/conftest.py`): `ethylene_symbols`, `ethylene_bohr` and `make_settings(**fields) -> ElectronicSection`.

- [ ] **Step 1: Add the data files.**

`tests/data/ethylene.xyz`. This is the owner's validation geometry, from `C:\AJCamargo\TOPICOS\Eteno\Eteno.xyz`; copy it exactly:

```text
6
Generated by the gQTEA molecular design tool (pre-optimized); coordinates in Angstrom
C   -0.66838406 -0.02359001 -0.00000006
C    0.66838372  0.02358989  0.00000005
H   -1.17036631 -0.96853122 -0.00000039
H   -1.23573513  0.88361004  0.00000043
H    1.23573623 -0.88360988  0.00000040
H    1.17036556  0.96853118 -0.00000042
```

`tests/data/reference/sharc41_pyscf_ethylene_sa2cas22_631gs.json`. These are the values from running SHARC 4.1 `SHARC_PYSCF.py` as a black box on 2026-11-07. Task 10's tool regenerates and checks this file.

```json
{
  "schema_version": 1,
  "source": "SHARC 4.1 SHARC_PYSCF.py, executed as a black box (QM.out parsed by gqteashmd)",
  "molecule": "tests/data/ethylene.xyz",
  "method": "SA-CASSCF",
  "basis": "6-31g*",
  "ncas": 2,
  "nelecas": 2,
  "roots": 2,
  "sharc_template": {"fix-spin-shift": 0.2, "conv-tol": 1e-10, "conv-tol-grad": 1e-06},
  "energies_hartree": [-78.04902443, -77.67250836],
  "transition_dipoles_au": {"S1": [-1.529760771, -0.05399139466, -2.380718969e-07]},
  "gradients_hartree_per_bohr": {
    "S0": [
      [-0.003889053964, -0.000136948009, -3.715872228e-08],
      [0.00388875951, 0.0001373424173, 3.172899634e-08],
      [0.006545164315, 0.0005754966567, -1.647271776e-07],
      [0.006569650127, -0.0001128933028, 1.863221695e-07],
      [-0.006569321885, 0.0001126094566, 1.674421018e-07],
      [-0.006545198103, -0.0005756072187, -1.836073677e-07]
    ],
    "S1": [
      [0.1266703636, 0.004471024236, 2.360564851e-08],
      [-0.1266706699, -0.004470625729, -1.876612445e-08],
      [0.006474015469, -0.002689478915, 3.082738396e-07],
      [0.006268505785, 0.003138938297, -3.169175757e-07],
      [-0.006268161377, -0.003139226823, -3.106935710e-07],
      [-0.006474053519, 0.002689368933, 3.144977830e-07]
    ]
  }
}
```

- [ ] **Step 2: Add the shared test fixtures** in `tests/electronic/conftest.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from gqteashmd.chem.xyz import parse_xyz
from gqteashmd.config.schema import ElectronicSection
from gqteashmd.core.units import ANGSTROM_TO_BOHR

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def ethylene_symbols() -> tuple[str, ...]:
    return parse_xyz((DATA / "ethylene.xyz").read_text()).symbols


@pytest.fixture
def ethylene_bohr() -> np.ndarray:
    geometry = parse_xyz((DATA / "ethylene.xyz").read_text())
    return np.array(geometry.coords_angstrom) * ANGSTROM_TO_BOHR


@pytest.fixture
def make_settings():
    def _make(**fields) -> ElectronicSection:
        base = {"method": "casscf", "basis": "6-31g*", "singlets": 2,
                "casscf": {"ncas": 2, "nelecas": 2}}
        base.update(fields)
        return ElectronicSection.model_validate(base)

    return _make


@pytest.fixture
def reference_dir() -> Path:
    return DATA / "reference"
```

- [ ] **Step 3: Write the failing tests** in `tests/electronic/test_pyscf_casscf.py`:

```python
import json

import numpy as np
import pytest

from gqteashmd.core.states import StateLabel
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.pyscf_backend import PySCFBackend, require_pyscf
from gqteashmd.errors import ConfigError, ElectronicStructureError

S0, S1 = StateLabel(1, 0), StateLabel(1, 1)


def _backend(symbols, settings, **kw):
    return PySCFBackend(MoleculeSpec(tuple(symbols)), settings, **kw)


def test_require_pyscf_reports_a_supported_version():
    major, minor = (int(x) for x in require_pyscf().split(".")[:2])
    assert (major, minor) >= (2, 6)


def test_sa2_casscf_matches_the_sharc_reference(ethylene_symbols, ethylene_bohr, make_settings,
                                                reference_dir):
    ref = json.loads((reference_dir / "sharc41_pyscf_ethylene_sa2cas22_631gs.json").read_text())
    backend = _backend(ethylene_symbols, make_settings())
    result = backend.compute(Request(ethylene_bohr, gradients=frozenset({S0, S1}),
                                     transition_dipoles=True))
    assert result.states == (S0, S1)
    for label, e_ref in zip((S0, S1), ref["energies_hartree"], strict=True):
        assert abs(result.energies[label] - e_ref) < 1e-6
    for name, g_ref in ref["gradients_hartree_per_bohr"].items():
        label = StateLabel.parse(name)
        assert np.abs(result.gradients[label] - np.array(g_ref)).max() < 1e-5
    mu_ref = np.array(ref["transition_dipoles_au"]["S1"])
    mu = result.transition_dipoles[S1]
    assert min(np.abs(mu - mu_ref).max(), np.abs(mu + mu_ref).max()) < 1e-4  # global sign is arbitrary
    assert all(abs(s) < 1e-6 for s in result.spin_squares.values())


def test_meta_and_log_file(tmp_path, ethylene_symbols, ethylene_bohr, make_settings):
    log = tmp_path / "logs" / "pyscf.log"
    result = _backend(ethylene_symbols, make_settings(basis="6-31g"), log_file=log).compute(
        Request(ethylene_bohr)
    )
    assert log.stat().st_size > 0
    assert result.meta["method"] == "casscf" and result.meta["basis"] == "6-31g"
    assert result.meta["scf"] == "default" and result.meta["pyscf_version"]
    assert result.gradients == {} and result.transition_dipoles == {}


def test_request_validation(ethylene_symbols, ethylene_bohr, make_settings):
    backend = _backend(ethylene_symbols, make_settings(basis="sto-3g"))
    with pytest.raises(ValueError, match="S5"):
        backend.compute(Request(ethylene_bohr, gradients=frozenset({StateLabel(1, 5)})))
    with pytest.raises(ValueError, match="atom"):
        backend.compute(Request(ethylene_bohr[:3]))


def test_unknown_basis_is_an_electronic_structure_error(ethylene_symbols, ethylene_bohr,
                                                       make_settings):
    with pytest.raises(ElectronicStructureError, match="basis 'no-such-basis'") as info:
        _backend(ethylene_symbols, make_settings(basis="no-such-basis")).compute(
            Request(ethylene_bohr)
        )
    assert info.value.retryable is False and info.value.exit_code == 4


def test_active_space_must_fit(ethylene_symbols, ethylene_bohr, make_settings):
    settings = make_settings(basis="sto-3g", casscf={"ncas": 40, "nelecas": 2})
    with pytest.raises(ElectronicStructureError, match="active space"):
        _backend(ethylene_symbols, settings).compute(Request(ethylene_bohr))


def test_scf_failure_after_the_ladder(ethylene_symbols, ethylene_bohr, make_settings):
    with pytest.raises(ElectronicStructureError, match="SCF did not converge"):
        _backend(ethylene_symbols, make_settings(basis="6-31g", scf_max_cycle=1)).compute(
            Request(ethylene_bohr)
        )


def test_open_shell_molecule_is_refused(ethylene_symbols, make_settings):
    with pytest.raises(ConfigError, match="closed-shell singlet"):
        PySCFBackend(MoleculeSpec(tuple(ethylene_symbols), 0, 3), make_settings())


def test_previous_orbitals_are_projected(ethylene_symbols, ethylene_bohr, make_settings):
    settings = make_settings(basis="6-31g")
    moved = ethylene_bohr + np.random.default_rng(7).normal(scale=0.05, size=ethylene_bohr.shape)
    reused = _backend(ethylene_symbols, settings)
    reused.compute(Request(ethylene_bohr))
    second = reused.compute(Request(moved))
    fresh = _backend(ethylene_symbols, settings).compute(Request(moved))
    for label in (S0, S1):
        assert abs(second.energies[label] - fresh.energies[label]) < 1e-7
```

- [ ] **Step 4: Run them to verify they fail.**
  Run: `pytest tests/electronic/test_pyscf_casscf.py -v`
  Expected: FAIL with `ModuleNotFoundError: gqteashmd.electronic.pyscf_backend`.

- [ ] **Step 5: Implement** `src/gqteashmd/electronic/pyscf_backend.py`:

```python
"""PySCF backend: restricted SCF, SA-CASSCF and TDA/TDDFT energies, gradients, dipoles.

PySCF is imported lazily inside functions so that the rest of gqteashmd never needs it.
Coordinates are passed to PySCF in bohr; all results are in atomic units.
"""

from __future__ import annotations

import re
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

import numpy as np

from gqteashmd.config.schema import ElectronicSection
from gqteashmd.core.states import StateLabel, states_from_counts
from gqteashmd.electronic.base import ElectronicBackend, ElectronicResult, MoleculeSpec, Request
from gqteashmd.errors import ConfigError, ElectronicStructureError

MIN_PYSCF_VERSION = (2, 6, 0)
SPIN_TOLERANCE = 1e-3  # |<S^2>| above this means a root is not a singlet

# (name, level shift / Eh, damping factor, second-order solver)
_SCF_LADDER: tuple[tuple[str, float, float, bool], ...] = (
    ("default", 0.0, 0.0, False),
    ("level shift + damping", 0.2, 0.5, False),
    ("second-order", 0.0, 0.0, True),
)


def require_pyscf() -> str:
    """Return the installed PySCF version; ConfigError if missing or too old."""
    try:
        import pyscf
    except ImportError as exc:
        raise ConfigError(
            "PySCF is not installed in this environment; install it with conda (pyscf)."
        ) from exc
    version: str = pyscf.__version__
    parts = tuple(int(p) for p in re.findall(r"\d+", version)[:3])
    if parts < MIN_PYSCF_VERSION:
        raise ConfigError(f"PySCF {version} is too old; gqteashmd needs PySCF >= 2.6.0.")
    return version


class PySCFBackend(ElectronicBackend):
    def __init__(
        self,
        molecule: MoleculeSpec,
        settings: ElectronicSection,
        *,
        log_file: Path | None = None,
        threads: int | None = None,
        memory_mb: int | None = None,
    ) -> None:
        self._version = require_pyscf()
        if molecule.multiplicity != 1:
            raise ConfigError(
                "The PySCF backend supports closed-shell singlet molecules only "
                "(molecule.multiplicity = 1)."
            )
        self._molecule = molecule
        self._settings = settings
        self._log_file = log_file
        self._threads = threads
        self._memory_mb = memory_mb
        self._states = states_from_counts(settings.singlets, settings.triplets)
        self._previous_mol: Any = None
        self._previous_mo: Any = None
        self._previous_dm: Any = None

    @property
    def states(self) -> tuple[StateLabel, ...]:
        return self._states

    # ------------------------------------------------------------------ public API
    def compute(self, request: Request) -> ElectronicResult:
        unknown = sorted(str(s) for s in set(request.gradients) - set(self._states))
        if unknown:
            raise ValueError(f"Gradients requested for unknown state(s): {', '.join(unknown)}")
        if len(request.coords_bohr) != len(self._molecule.symbols):
            raise ValueError(
                f"Got coordinates for {len(request.coords_bohr)} atoms; the molecule has "
                f"{len(self._molecule.symbols)} atoms."
            )
        if self._threads is not None:
            from pyscf import lib

            lib.num_threads(self._threads)
        started = time.perf_counter()
        with self._log() as stdout:
            mol = self._build_mol(request.coords_bohr, stdout)
            mf, scf_rung = self._run_scf(mol)
            if self._settings.method == "casscf":
                energies, gradients, dipoles, spins = self._run_casscf(mol, mf, request)
            else:
                energies, gradients, dipoles, spins = self._run_linear_response(mol, mf, request)
        meta: dict[str, Any] = {
            "backend": "pyscf",
            "pyscf_version": self._version,
            "method": self._settings.method,
            "basis": self._settings.basis,
            "functional": self._settings.functional,
            "scf": scf_rung,
            "scf_energy_hartree": float(mf.e_tot),
            "wall_time_s": time.perf_counter() - started,
        }
        return ElectronicResult(self._states, energies, gradients, dipoles, spins, meta)

    # ------------------------------------------------------------------ setup
    @contextmanager
    def _log(self) -> Iterator[IO[str] | None]:
        if self._log_file is None:
            yield None
        else:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_file, "a", encoding="utf-8") as fh:
                yield fh

    def _build_mol(self, coords: np.ndarray, stdout: IO[str] | None) -> Any:
        from pyscf import gto

        mol = gto.Mole()
        if stdout is not None:
            mol.stdout = stdout
            mol.verbose = 4
        else:
            mol.verbose = 0
        if self._memory_mb is not None:
            mol.max_memory = self._memory_mb
        atoms = [
            (symbol, (float(x), float(y), float(z)))
            for symbol, (x, y, z) in zip(self._molecule.symbols, coords, strict=True)
        ]
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Basis may be available", category=UserWarning)
                mol.build(
                    atom=atoms,
                    unit="Bohr",
                    basis=self._settings.basis,
                    charge=self._molecule.charge,
                    spin=self._molecule.multiplicity - 1,
                )
        except RuntimeError as exc:  # pyscf BasisNotFoundError is a RuntimeError
            raise ElectronicStructureError(
                f"PySCF cannot build basis '{self._settings.basis}': {exc}"
            ) from exc
        return mol

    def _make_scf(self, mol: Any) -> Any:
        from pyscf import dft, scf

        functional = self._settings.functional
        if self._settings.method == "casscf" or functional is None or functional.lower() == "hf":
            mf = scf.RHF(mol)
        else:
            mf = dft.RKS(mol)
            mf.xc = functional
            mf.grids.level = self._settings.grid_level
        mf.conv_tol = self._settings.scf_conv_tol
        mf.max_cycle = self._settings.scf_max_cycle
        return mf

    def _run_scf(self, mol: Any) -> tuple[Any, str]:
        dm0 = self._previous_dm
        if dm0 is not None and getattr(dm0, "shape", None) != (mol.nao, mol.nao):
            dm0 = None
        for name, shift, damp, second_order in _SCF_LADDER:
            mf = self._make_scf(mol)
            mf.level_shift = shift
            mf.damp = damp
            if second_order:
                mf = mf.newton()
            try:
                mf.kernel(dm0=dm0)
            except Exception as exc:  # PySCF raises many types; report them as SCF failures
                raise ElectronicStructureError(
                    f"PySCF SCF failed ({name}): {type(exc).__name__}: {exc}"
                ) from exc
            if mf.converged:
                self._previous_dm = mf.make_rdm1()
                return mf, name
        tried = ", ".join(name for name, *_ in _SCF_LADDER)
        raise ElectronicStructureError(
            f"SCF did not converge (tried: {tried}; scf_max_cycle = "
            f"{self._settings.scf_max_cycle}, scf_conv_tol = {self._settings.scf_conv_tol})."
        )

    # ------------------------------------------------------------------ SA-CASSCF
    def _run_casscf(
        self, mol: Any, mf: Any, request: Request
    ) -> tuple[
        dict[StateLabel, float],
        dict[StateLabel, np.ndarray],
        dict[StateLabel, np.ndarray],
        dict[StateLabel, float],
    ]:
        from pyscf import fci, mcscf

        cas = self._settings.casscf
        assert cas is not None  # guaranteed by ElectronicSection validation
        core_electrons = mol.nelectron - cas.nelecas
        if core_electrons < 0 or core_electrons % 2:
            raise ElectronicStructureError(
                f"The CASSCF active space ({cas.nelecas} electrons) does not fit the molecule's "
                f"{mol.nelectron} electrons."
            )
        if core_electrons // 2 + cas.ncas > mol.nao:
            raise ElectronicStructureError(
                f"The CASSCF active space needs {core_electrons // 2 + cas.ncas} orbitals, but the "
                f"basis '{self._settings.basis}' has only {mol.nao}."
            )
        n_states = len(self._states)
        mc = mcscf.CASSCF(mf, cas.ncas, cas.nelecas)
        mc.fcisolver = fci.solver(mol, singlet=True)  # singlet-only CI (the default finds triplets)
        mc = mc.state_average_([1.0 / n_states] * n_states)
        mc.conv_tol = cas.conv_tol
        mc.conv_tol_grad = cas.conv_tol_grad
        mo0 = mf.mo_coeff
        if self._previous_mo is not None and self._previous_mol is not None:
            # orbitals of another geometry are not orthonormal here: project them first
            mo0 = mcscf.project_init_guess(mc, self._previous_mo, self._previous_mol)
        try:
            mc.kernel(mo_coeff=mo0)
        except Exception as exc:
            raise ElectronicStructureError(f"CASSCF failed: {type(exc).__name__}: {exc}") from exc
        if not mc.converged:
            raise ElectronicStructureError(
                f"SA-CASSCF did not converge (conv_tol = {cas.conv_tol}, "
                f"conv_tol_grad = {cas.conv_tol_grad})."
            )
        spin_values, _ = mc.fcisolver.states_spin_square(mc.ci, mc.ncas, mc.nelecas)
        spins = {label: float(s) for label, s in zip(self._states, spin_values, strict=True)}
        for label, s in spins.items():
            if abs(s) > SPIN_TOLERANCE:
                raise ElectronicStructureError(
                    f"CASSCF root {label} has <S^2> = {s:.4f}; it is not a singlet."
                )
        energies = {
            label: float(e) for label, e in zip(self._states, mc.e_states, strict=True)
        }
        gradients: dict[StateLabel, np.ndarray] = {}
        if request.gradients:
            solver = mc.nuc_grad_method()
            for label in sorted(request.gradients):
                g = solver.kernel(state=label.index)
                if not getattr(solver, "converged", True):
                    raise ElectronicStructureError(f"The SA-CASSCF gradient of {label} did not converge.")
                gradients[label] = np.asarray(g, dtype=float)
        dipoles: dict[StateLabel, np.ndarray] = {}
        if request.transition_dipoles:
            r_ints = mol.intor("int1e_r", comp=3)
            mo_cas = mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas]
            for label in self._states[1:]:
                t_mo = mc.fcisolver.trans_rdm1(mc.ci[0], mc.ci[label.index], mc.ncas, mc.nelecas)
                t_ao = mo_cas @ t_mo @ mo_cas.T
                dipoles[label] = -np.einsum("xij,ji->x", r_ints, t_ao)  # μ = -Σ r γ (electrons)
        self._previous_mo, self._previous_mol = mc.mo_coeff, mol
        return energies, gradients, dipoles, spins

    # ------------------------------------------------------------------ TDA / TDDFT (Task 7)
    def _run_linear_response(
        self, mol: Any, mf: Any, request: Request
    ) -> tuple[
        dict[StateLabel, float],
        dict[StateLabel, np.ndarray],
        dict[StateLabel, np.ndarray],
        dict[StateLabel, float],
    ]:
        raise NotImplementedError("TDA/TDDFT is implemented in Task 7")
```

(The two lines over 100 characters, the `filterwarnings` call and the gradient `raise`, are wrapped by `ruff format`.)

- [ ] **Step 6: Run the tests to verify they pass.**
  Run: `pytest tests/electronic -v && mypy`
  Expected: all pass, in about 10–20 s. mypy prints `Success`.

- [ ] **Step 7: Lint, full suite, commit**

```bash
ruff format src tests && ruff check --fix src tests && pytest -m "not slow" -W error
git add src/gqteashmd/electronic/pyscf_backend.py tests/data tests/electronic/conftest.py tests/electronic/test_pyscf_casscf.py
git commit -m "feat: add the PySCF backend with SA-CASSCF energies, gradients and transition dipoles"
```

---

### Task 7: PySCF backend: TDA, TDDFT, CIS and TDHF

**Files:**
- Modify: `src/gqteashmd/electronic/pyscf_backend.py` (replace `_run_linear_response`)
- Test: `tests/electronic/test_pyscf_linear_response.py`

**Interfaces:**
- Consumes: `PySCFBackend` and the fixtures from Task 6.
- Produces the `tda`/`tddft` methods. Their properties:
  - S0 is the SCF energy.
  - Sk = E_SCF + ω_k.
  - Gradients come from `mf.nuc_grad_method()` for S0 and from `td.nuc_grad_method().kernel(state=k)` for Sk.
  - Transition dipoles come from `td.transition_dipole()`.
  - `spin_squares` is empty, because the reference is restricted and the response is singlet by construction.
  - `functional = "hf"` gives CIS (tda) or TDHF (tddft).

- [ ] **Step 1: Write the failing tests** in `tests/electronic/test_pyscf_linear_response.py`:

```python
import numpy as np
import pytest
from pyscf import dft, gto, scf, tddft

from gqteashmd.core.states import StateLabel
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.properties import oscillator_strength
from gqteashmd.electronic.pyscf_backend import PySCFBackend
from gqteashmd.errors import ElectronicStructureError

S0, S1, S2 = StateLabel(1, 0), StateLabel(1, 1), StateLabel(1, 2)


def _tda_settings(make_settings, **kw):
    fields = {"method": "tda", "basis": "sto-3g", "functional": "b3lyp", "singlets": 3,
              "casscf": None}
    fields.update(kw)
    return make_settings(**fields)


def _direct(symbols, coords, functional, method="tda"):
    mol = gto.M(atom=[(s, tuple(c)) for s, c in zip(symbols, coords, strict=True)],
                unit="Bohr", basis="sto-3g", verbose=0)
    if functional == "hf":
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol)
        mf.xc = functional
        mf.grids.level = 3
    mf.conv_tol = 1e-10
    mf.kernel()
    td = tddft.TDA(mf) if method == "tda" else tddft.TDDFT(mf)
    td.nstates = 2
    td.conv_tol = 1e-8
    td.kernel()
    return mf, td


def test_tda_matches_direct_pyscf(ethylene_symbols, ethylene_bohr, make_settings):
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), _tda_settings(make_settings))
    result = backend.compute(Request(ethylene_bohr, gradients=frozenset({S0, S1}),
                                     transition_dipoles=True))
    mf, td = _direct(ethylene_symbols, ethylene_bohr, "b3lyp")
    assert abs(result.energies[S0] - mf.e_tot) < 1e-8
    for label, omega in zip((S1, S2), td.e, strict=True):
        assert abs(result.excitation_energy(label) - omega) < 1e-6
    for label, f_ref in zip((S1, S2), td.oscillator_strength(), strict=True):
        f = oscillator_strength(result.excitation_energy(label), result.transition_dipoles[label])
        assert f == pytest.approx(f_ref, rel=1e-4, abs=1e-8)
    assert np.abs(result.gradients[S0] - mf.nuc_grad_method().kernel()).max() < 1e-6
    assert np.abs(result.gradients[S1] - td.nuc_grad_method().kernel(state=1)).max() < 1e-5
    assert result.spin_squares == {}


def test_cis_and_tdhf_with_hf(ethylene_symbols, ethylene_bohr, make_settings):
    for method in ("tda", "tddft"):
        settings = _tda_settings(make_settings, method=method, functional="hf")
        result = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings).compute(
            Request(ethylene_bohr, gradients=frozenset({S1}))
        )
        mf, td = _direct(ethylene_symbols, ethylene_bohr, "hf", method)
        assert abs(result.energies[S0] - mf.e_tot) < 1e-8
        assert abs(result.excitation_energy(S1) - td.e[0]) < 1e-6
        assert result.gradients[S1].shape == (6, 3)


def test_ground_state_only(ethylene_symbols, ethylene_bohr, make_settings):
    settings = _tda_settings(make_settings, singlets=1)
    result = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings).compute(
        Request(ethylene_bohr, gradients=frozenset({S0}), transition_dipoles=True)
    )
    assert result.states == (S0,)
    assert result.transition_dipoles == {} and result.gradients[S0].shape == (6, 3)


def test_unknown_functional_is_an_electronic_structure_error(ethylene_symbols, ethylene_bohr,
                                                            make_settings):
    settings = _tda_settings(make_settings, functional="no-such-functional")
    with pytest.raises(ElectronicStructureError, match="SCF failed"):
        PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings).compute(
            Request(ethylene_bohr)
        )
```

- [ ] **Step 2: Run them to verify they fail.**
  Run: `pytest tests/electronic/test_pyscf_linear_response.py -v`
  Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement.** Replace `_run_linear_response` in `src/gqteashmd/electronic/pyscf_backend.py` with:

```python
    def _run_linear_response(
        self, mol: Any, mf: Any, request: Request
    ) -> tuple[
        dict[StateLabel, float],
        dict[StateLabel, np.ndarray],
        dict[StateLabel, np.ndarray],
        dict[StateLabel, float],
    ]:
        from pyscf import tddft

        excited = self._states[1:]
        energies: dict[StateLabel, float] = {self._states[0]: float(mf.e_tot)}
        gradients: dict[StateLabel, np.ndarray] = {}
        dipoles: dict[StateLabel, np.ndarray] = {}
        td: Any = None
        if excited:
            td = tddft.TDA(mf) if self._settings.method == "tda" else tddft.TDDFT(mf)
            td.nstates = len(excited)
            td.conv_tol = self._settings.td_conv_tol
            try:
                td.kernel()
            except Exception as exc:
                raise ElectronicStructureError(
                    f"{self._settings.method.upper()} failed: {type(exc).__name__}: {exc}"
                ) from exc
            converged = np.atleast_1d(np.asarray(td.converged, dtype=bool))
            if len(td.e) < len(excited) or not bool(converged.all()):
                raise ElectronicStructureError(
                    f"{self._settings.method.upper()} did not converge for all {len(excited)} "
                    f"excited states (td_conv_tol = {self._settings.td_conv_tol})."
                )
            for label, omega in zip(excited, td.e, strict=False):
                energies[label] = float(mf.e_tot + omega)
            if request.transition_dipoles:
                for label, mu in zip(excited, td.transition_dipole(), strict=False):
                    dipoles[label] = np.asarray(mu, dtype=float)
        for label in sorted(request.gradients):
            if label.index == 0:
                g = mf.nuc_grad_method().kernel()
            else:
                g = td.nuc_grad_method().kernel(state=label.index)
            gradients[label] = np.asarray(g, dtype=float)
        return energies, gradients, dipoles, {}
```

- [ ] **Step 4: Run the tests to verify they pass.**
  Run: `pytest tests/electronic -v && mypy`
  Expected: all pass. mypy prints `Success`.

- [ ] **Step 5: Lint, full suite, commit**

```bash
ruff format src tests && ruff check --fix src tests && pytest -m "not slow" -W error
git add src/gqteashmd/electronic/pyscf_backend.py tests/electronic/test_pyscf_linear_response.py
git commit -m "feat: add TDA, TDDFT, CIS and TDHF to the PySCF backend"
```

---

### Task 8: Gradient validation by finite differences (slow tests)

**Files:**
- Create: `tests/electronic/test_gradients_fd.py`

**Interfaces:**
- Consumes: `PySCFBackend` and the fixtures from Task 6.
- Produces: slow tests (`@pytest.mark.slow`) that compare analytic gradients with central finite differences. Each displaced point is computed with a **fresh** backend; reused orbitals introduce convergence noise of about 10⁻⁵.

- [ ] **Step 1: Write the tests** in `tests/electronic/test_gradients_fd.py`:

```python
import numpy as np
import pytest

from gqteashmd.core.states import StateLabel
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.pyscf_backend import PySCFBackend

S0, S1 = StateLabel(1, 0), StateLabel(1, 1)
H = 1e-4  # bohr


def _fd(symbols, settings, coords, label, atom, axis):
    def energy(c):
        return PySCFBackend(MoleculeSpec(tuple(symbols)), settings).compute(Request(c)).energies[label]

    plus, minus = coords.copy(), coords.copy()
    plus[atom, axis] += H
    minus[atom, axis] -= H
    return (energy(plus) - energy(minus)) / (2 * H)


@pytest.mark.slow
def test_sa_casscf_gradients_match_finite_differences(ethylene_symbols, ethylene_bohr, make_settings):
    settings = make_settings(basis="6-31g", scf_conv_tol=1e-12,
                             casscf={"ncas": 2, "nelecas": 2, "conv_tol": 1e-12, "conv_tol_grad": 1e-7})
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings)
    result = backend.compute(Request(ethylene_bohr, gradients=frozenset({S0, S1})))
    for atom, axis in ((0, 0), (2, 1), (4, 2)):
        for label in (S0, S1):
            fd = _fd(ethylene_symbols, settings, ethylene_bohr, label, atom, axis)
            assert abs(fd - result.gradients[label][atom, axis]) < 1e-6, (label, atom, axis)


@pytest.mark.slow
def test_tda_gradients_match_finite_differences(ethylene_symbols, ethylene_bohr, make_settings):
    settings = make_settings(method="tda", basis="sto-3g", functional="b3lyp", singlets=2,
                             casscf=None, scf_conv_tol=1e-12, td_conv_tol=1e-10)
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings)
    result = backend.compute(Request(ethylene_bohr, gradients=frozenset({S1})))
    for atom, axis in ((0, 0), (2, 1)):
        fd = _fd(ethylene_symbols, settings, ethylene_bohr, S1, atom, axis)
        assert abs(fd - result.gradients[S1][atom, axis]) < 5e-6, (atom, axis)
```

- [ ] **Step 2: Run the tests.**
  Run: `pytest tests/electronic/test_gradients_fd.py -v -m slow`
  Expected: 2 passed, in under about 2 minutes. If a tolerance fails, report the measured differences. Do not loosen a tolerance without reporting it.

- [ ] **Step 3: Lint, commit**

```bash
ruff format src tests && ruff check --fix src tests
git add tests/electronic/test_gradients_fd.py
git commit -m "test: validate SA-CASSCF and TDA gradients against finite differences"
```

---

### Task 9: `gqteashmd single-point` and the backend factory

**Files:**
- Create: `src/gqteashmd/electronic/factory.py`
- Modify: `src/gqteashmd/cli.py`
- Test: `tests/test_cli_single_point.py`

**Interfaces:**
- Consumes:
  - `PySCFBackend`, `Request` and `MoleculeSpec`;
  - `parse_xyz`, `atomic_number`, `count_electrons` and `ANGSTROM_TO_BOHR`;
  - `ElectronicResult.to_json()`.
- Produces:
  - `make_backend(config, molecule, *, log_file=None, threads=None, memory_mb=None) -> ElectronicBackend`. It raises `ConfigError` when `[electronic]` is missing.
  - The CLI command `gqteashmd single-point INPUT [--json PATH] [--log PATH]`. It computes energies and S0→Sn transition dipoles at the input geometry and prints a table:
    - state;
    - energy in Eh;
    - ΔE in eV;
    - f.

    It returns 0 on success, 2 for input errors and 4 for electronic-structure failures.

- [ ] **Step 1: Write the failing tests** in `tests/test_cli_single_point.py`:

```python
import json
from pathlib import Path

from gqteashmd.cli import main

ETHYLENE = (Path(__file__).resolve().parent / "data" / "ethylene.xyz").read_text()
CAS = (
    '[electronic]\nmethod = "casscf"\nbasis = "6-31g"\nsinglets = 2\n'
    "[electronic.casscf]\nncas = 2\nnelecas = 2\n"
)


def _input(tmp_path, extra=CAS, xyz=ETHYLENE):
    (tmp_path / "ethylene.xyz").write_text(xyz)
    path = tmp_path / "input.toml"
    path.write_text(
        '[project]\nname = "sp"\ndirectory = "run"\n\n[molecule]\ngeometry = "ethylene.xyz"\n\n'
        "[resources]\ntotal_cores = 4\ntotal_memory_mb = 4000\n\n" + extra
    )
    return path


def test_single_point_prints_a_table_and_writes_json(tmp_path, capsys):
    out = tmp_path / "sp.json"
    log = tmp_path / "pyscf.log"
    assert main(["single-point", str(_input(tmp_path)), "--json", str(out), "--log", str(log)]) == 0
    text = capsys.readouterr().out
    assert "casscf / 6-31g" in text and "S0" in text and "S1" in text
    data = json.loads(out.read_text())
    assert [row["label"] for row in data["states"]] == ["S0", "S1"]
    assert data["states"][1]["oscillator_strength"] > 0.1
    assert log.stat().st_size > 0
    assert not (tmp_path / "run").exists()  # a single point does not create a project


def test_missing_electronic_section_is_exit_2(tmp_path, capsys):
    assert main(["single-point", str(_input(tmp_path, extra=""))]) == 2
    assert "[electronic]" in capsys.readouterr().err


def test_invalid_geometry_is_exit_2(tmp_path, capsys):
    assert main(["single-point", str(_input(tmp_path, xyz="2\n\nC 0 0 0\n"))]) == 2
    assert "Invalid geometry file" in capsys.readouterr().err


def test_unknown_basis_is_exit_4(tmp_path, capsys):
    path = _input(tmp_path, extra=CAS.replace("6-31g", "no-such-basis"))
    assert main(["single-point", str(path)]) == 4
    assert "basis 'no-such-basis'" in capsys.readouterr().err
```

- [ ] **Step 2: Run them to verify they fail.**
  Run: `pytest tests/test_cli_single_point.py -v`
  Expected: FAIL, because `single-point` is an invalid choice and argparse exits with 2. The first test fails.

- [ ] **Step 3: Implement.**

`src/gqteashmd/electronic/factory.py`:

```python
"""Choose and construct the electronic-structure backend for a configuration."""

from __future__ import annotations

from pathlib import Path

from gqteashmd.config.schema import ShmdConfig
from gqteashmd.electronic.base import ElectronicBackend, MoleculeSpec
from gqteashmd.errors import ConfigError


def make_backend(
    config: ShmdConfig,
    molecule: MoleculeSpec,
    *,
    log_file: Path | None = None,
    threads: int | None = None,
    memory_mb: int | None = None,
) -> ElectronicBackend:
    if config.electronic is None:
        raise ConfigError(
            "This calculation needs an [electronic] section in the input file "
            "(method, basis, singlets, ...)."
        )
    from gqteashmd.electronic.pyscf_backend import PySCFBackend

    return PySCFBackend(
        molecule, config.electronic, log_file=log_file, threads=threads, memory_mb=memory_mb
    )
```

In `src/gqteashmd/cli.py`, add these imports:

```python
import numpy as np

from gqteashmd.chem.electrons import count_electrons
from gqteashmd.chem.elements import atomic_number
from gqteashmd.chem.xyz import parse_xyz
from gqteashmd.core.units import ANGSTROM_TO_BOHR
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.factory import make_backend
from gqteashmd.util.fileio import write_json_atomic
```

Add the command function:

```python
def _cmd_single_point(args: argparse.Namespace) -> int:
    cfg = load_config(args.input).config
    path = cfg.molecule.geometry
    if not path.is_file():
        raise ConfigError(f"Geometry file not found or not a regular file: {path}")
    try:
        geometry = parse_xyz(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise ConfigError(f"Geometry file {path} is not valid UTF-8 text.") from exc
    except OSError as exc:
        raise ConfigError(f"Cannot read geometry file {path}: {exc.strerror or exc}") from exc
    except ValueError as exc:
        raise ConfigError(f"Invalid geometry file {path}: {exc}") from exc
    try:
        count_electrons(
            [atomic_number(s) for s in geometry.symbols],
            cfg.molecule.charge,
            cfg.molecule.multiplicity,
        )
    except ValueError as exc:
        raise ConfigError(f"{exc} Check [molecule] in the input file.") from exc
    molecule = MoleculeSpec(geometry.symbols, cfg.molecule.charge, cfg.molecule.multiplicity)
    backend = make_backend(
        cfg,
        molecule,
        log_file=args.log,
        threads=cfg.resources.threads_per_task,
        memory_mb=cfg.resources.memory_mb_per_task,
    )
    coords = np.array(geometry.coords_angstrom, dtype=float) * ANGSTROM_TO_BOHR
    result = backend.compute(Request(coords, transition_dipoles=True))
    data = result.to_json()
    settings = cfg.electronic
    assert settings is not None  # make_backend checked it
    name = f"{settings.method} / {settings.basis}"
    if settings.functional:
        name += f" / {settings.functional}"
    print(f"Single point: {name} (PySCF {result.meta['pyscf_version']})")
    print(f"{'state':<6}{'energy (Eh)':>18}{'dE (eV)':>10}{'f':>9}")
    for row in data["states"]:
        f = "-" if row["oscillator_strength"] is None else f"{row['oscillator_strength']:.4f}"
        print(f"{row['label']:<6}{row['energy_hartree']:>18.9f}{row['excitation_ev']:>10.4f}{f:>9}")
    if args.json is not None:
        write_json_atomic(args.json, data)
        print(f"Results written to {args.json}")
    return 0
```

Register the command in `build_parser`, before `return parser`:

```python
    p_sp = sub.add_parser(
        "single-point", help="compute energies and transition dipoles at the input geometry"
    )
    p_sp.add_argument("input", type=Path)
    p_sp.add_argument("--json", type=Path, help="write the results to this JSON file")
    p_sp.add_argument("--log", type=Path, help="write the PySCF output to this file")
    p_sp.set_defaults(func=_cmd_single_point)
```

- [ ] **Step 4: Run the tests to verify they pass.**
  Run: `pytest tests/test_cli_single_point.py tests/test_cli.py -v && mypy`
  Expected: all pass. mypy prints `Success`.

- [ ] **Step 5: Lint, full suite, commit**

```bash
ruff format src tests && ruff check --fix src tests && pytest -m "not slow" -W error
git add src/gqteashmd/electronic/factory.py src/gqteashmd/cli.py tests/test_cli_single_point.py
git commit -m "feat: add the gqteashmd single-point command"
```

---

### Task 10: SHARC comparison tool (black box) and the reference check

**Files:**
- Create: `tools/sharc_compare/qmout.py`, `tools/sharc_compare/run_sharc_pyscf.py`, `tests/sharc_compare/test_qmout.py`, `tests/sharc_compare/test_live_sharc.py`
- Modify: `pyproject.toml` (marker `sharc`)

**Interfaces:**
- Produces:
  - **`parse_qm_out(text) -> dict`**, our own parser of SHARC's `QM.out` text format. It returns the keys `hamiltonian` (complex n×n), `dipole` (3 real n×n) and `gradients` (n_states × natom × 3).
  - **`tools/sharc_compare/run_sharc_pyscf.py`**, which runs `SHARC_PYSCF.py` in a temporary directory with the template settings from Task 6. Usage: `--sharc-home --python --xyz --basis --ncas --nelecas --roots --out`. It writes a JSON file in the Task 6 reference schema.
  - **The `sharc` test**, which is skipped unless the `SHARC_HOME` and `SHARC_PYTHON` environment variables are set. It checks that a fresh SHARC run reproduces the committed reference JSON.

- [ ] **Step 1: Register the marker.** In `pyproject.toml` `[tool.pytest.ini_options] markers`, add:
  `"sharc: runs SHARC 4.1 as a black box (needs SHARC_HOME and SHARC_PYTHON)"`.

- [ ] **Step 2: Write the failing parser test** in `tests/sharc_compare/test_qmout.py`:

```python
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("qmout", ROOT / "tools" / "sharc_compare" / "qmout.py")
qmout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qmout)

SAMPLE = """! 0 Basic information
states 2
nmstates 2
natom 2

! 1 Hamiltonian Matrix (2x2, complex)
2 2
-7.804902443E+001  0.000000000E+000  0.000000000E+000  0.000000000E+000
 0.000000000E+000  0.000000000E+000 -7.767250836E+001  0.000000000E+000

! 2 Dipole Moment Matrices (3x2x2, complex)
2 2
 2.5E-008  0.0E+000 -1.529760771E+000  0.0E+000
-1.529760771E+000  0.0E+000  2.6E-007  0.0E+000
2 2
 1.0E-007  0.0E+000 -5.399139466E-002  0.0E+000
-5.399139466E-002  0.0E+000  1.1E-007  0.0E+000
2 2
 9.8E-009  0.0E+000 -2.380718969E-007  0.0E+000
-2.380718969E-007  0.0E+000  2.2E-008  0.0E+000

! 3 Gradient Vectors (2x2x3, real)
2 3 ! 1 1 0.0
-3.889053964E-003 -1.369480090E-004 -3.715872228E-008
 3.888759510E-003  1.373424173E-004  3.172899634E-008
2 3 ! 1 2 0.0
 1.266703636E-001  4.471024236E-003  2.360564851E-008
-1.266706699E-001 -4.470625729E-003 -1.876612445E-008

! 8 Runtime
 3.0E+000
"""


def test_parse_qm_out():
    data = qmout.parse_qm_out(SAMPLE)
    assert data["hamiltonian"][0][0] == complex(-78.04902443, 0.0)
    assert data["hamiltonian"][1][1].real == -77.67250836
    assert [m[0][1] for m in data["dipole"]] == [-1.529760771, -0.05399139466, -2.380718969e-07]
    assert len(data["gradients"]) == 2 and data["gradients"][1][0][0] == 0.1266703636
```

- [ ] **Step 3: Run it to verify it fails.**
  Run: `pytest tests/sharc_compare/test_qmout.py -v`
  Expected: FAIL (the file is not found).

- [ ] **Step 4: Implement** `tools/sharc_compare/qmout.py`:

```python
"""Parse the text format of SHARC's QM.out (our own parser; no SHARC code)."""

from __future__ import annotations

import re
from typing import Any


def _rows(lines: list[str], start: int, n: int) -> list[list[float]]:
    return [[float(x) for x in lines[start + k].split()] for k in range(n)]


def parse_qm_out(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    out: dict[str, Any] = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("! 1 Hamiltonian Matrix"):
            n = int(lines[i + 1].split()[0])
            rows = _rows(lines, i + 2, n)
            out["hamiltonian"] = [[complex(r[2 * j], r[2 * j + 1]) for j in range(n)] for r in rows]
            i += 2 + n
        elif line.startswith("! 2 Dipole Moment Matrices"):
            mats, j = [], i + 1
            for _ in range(3):
                n = int(lines[j].split()[0])
                rows = _rows(lines, j + 1, n)
                mats.append([[r[2 * c] for c in range(n)] for r in rows])
                j += 1 + n
            out["dipole"] = mats
            i = j
        elif line.startswith("! 3 Gradient Vectors"):
            match = re.search(r"\((\d+)x(\d+)x3", line)
            if match is None:
                raise ValueError(f"cannot read gradient block header: {line!r}")
            n_states, natom = int(match.group(1)), int(match.group(2))
            grads, j = [], i + 1
            for _ in range(n_states):
                grads.append(_rows(lines, j + 1, natom))
                j += 1 + natom
            out["gradients"] = grads
            i = j
        else:
            i += 1
    return out
```

`tools/sharc_compare/run_sharc_pyscf.py`:

```python
"""Run SHARC 4.1's SHARC_PYSCF.py as a black box and write its results as reference JSON.

SHARC is only *executed*; nothing from SHARC is imported or copied.

Example (WSL):
  python tools/sharc_compare/run_sharc_pyscf.py --sharc-home /home/ajc/sharc4-4.1 \
      --python /home/ajc/miniforge3/envs/sharc4.1/bin/python --xyz tests/data/ethylene.xyz \
      --basis 6-31g* --ncas 2 --nelecas 2 --roots 2 --out /tmp/ref.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qmout import parse_qm_out  # noqa: E402

TEMPLATE = """ncas {ncas}
nelecas {nelecas}
roots {roots}
basis {basis}
method casscf
fix-spin-shift 0.2
conv-tol 1e-10
conv-tol-grad 1e-6
verbose 3
"""


def run(args: argparse.Namespace) -> dict:
    xyz_lines = Path(args.xyz).read_text().splitlines()
    natom = int(xyz_lines[0].split()[0])
    atoms = "\n".join(xyz_lines[2 : 2 + natom])
    with tempfile.TemporaryDirectory(prefix="gq-sharc-") as tmp:
        work = Path(tmp)
        (work / "QM.in").write_text(
            f"{natom}\nreference\n{atoms}\ninit\nsavedir ./SAVE\nunit angstrom\n"
            f"states {args.roots}\nh\ndm\ngrad all\n"
        )
        (work / "PYSCF.template").write_text(TEMPLATE.format(**vars(args)))
        (work / "PYSCF.resources").write_text("scratchdir ./SCRATCH\nmemory 4000\nncpu 1\n")
        home = Path(args.sharc_home)
        env = dict(os.environ)
        env["SHARC"] = str(home / "bin")
        env["PYTHONPATH"] = f"{home / 'lib'}:{home / 'pysharc'}"
        proc = subprocess.run(
            [args.python, str(home / "bin" / "SHARC_PYSCF.py"), "QM.in"],
            cwd=work, env=env, capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0:
            raise SystemExit(f"SHARC_PYSCF.py failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
        data = parse_qm_out((work / "QM.out").read_text())
    n = args.roots
    return {
        "schema_version": 1,
        "source": "SHARC 4.1 SHARC_PYSCF.py, executed as a black box (QM.out parsed by gqteashmd)",
        "molecule": str(args.xyz),
        "method": "SA-CASSCF",
        "basis": args.basis,
        "ncas": args.ncas,
        "nelecas": args.nelecas,
        "roots": n,
        "sharc_template": {"fix-spin-shift": 0.2, "conv-tol": 1e-10, "conv-tol-grad": 1e-06},
        "energies_hartree": [data["hamiltonian"][k][k].real for k in range(n)],
        "transition_dipoles_au": {
            f"S{k}": [data["dipole"][x][0][k] for x in range(3)] for k in range(1, n)
        },
        "gradients_hartree_per_bohr": {f"S{k}": data["gradients"][k] for k in range(n)},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sharc-home", required=True)
    p.add_argument("--python", required=True, help="Python of the SHARC conda environment")
    p.add_argument("--xyz", required=True)
    p.add_argument("--basis", required=True)
    p.add_argument("--ncas", type=int, required=True)
    p.add_argument("--nelecas", type=int, required=True)
    p.add_argument("--roots", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    Path(args.out).write_text(json.dumps(run(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write the live test** in `tests/sharc_compare/test_live_sharc.py`:

```python
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "tests" / "data" / "reference" / "sharc41_pyscf_ethylene_sa2cas22_631gs.json"


@pytest.mark.sharc
@pytest.mark.skipif(
    not (os.environ.get("SHARC_HOME") and os.environ.get("SHARC_PYTHON")),
    reason="needs SHARC_HOME and SHARC_PYTHON",
)
def test_fresh_sharc_run_reproduces_the_committed_reference(tmp_path):
    out = tmp_path / "ref.json"
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "sharc_compare" / "run_sharc_pyscf.py"),
         "--sharc-home", os.environ["SHARC_HOME"], "--python", os.environ["SHARC_PYTHON"],
         "--xyz", str(ROOT / "tests" / "data" / "ethylene.xyz"), "--basis", "6-31g*",
         "--ncas", "2", "--nelecas", "2", "--roots", "2", "--out", str(out)],
        check=True, timeout=1800,
    )
    fresh, ref = json.loads(out.read_text()), json.loads(REFERENCE.read_text())
    assert np.allclose(fresh["energies_hartree"], ref["energies_hartree"], atol=1e-8, rtol=0)
    for name in ref["gradients_hartree_per_bohr"]:
        assert np.allclose(fresh["gradients_hartree_per_bohr"][name],
                           ref["gradients_hartree_per_bohr"][name], atol=1e-7, rtol=0)
    mu, mu_ref = np.array(fresh["transition_dipoles_au"]["S1"]), np.array(ref["transition_dipoles_au"]["S1"])
    assert min(np.abs(mu - mu_ref).max(), np.abs(mu + mu_ref).max()) < 1e-7
```

- [ ] **Step 6: Run the tests.**
  Run: `pytest tests/sharc_compare -v`
  Expected: the parser test passes and the live test is skipped.

  Then, on this machine, run:
  `SHARC_HOME=/home/ajc/sharc4-4.1 SHARC_PYTHON=/home/ajc/miniforge3/envs/sharc4.1/bin/python pytest -m sharc -v`
  Expected: 1 passed. Paste its output into the report. If it fails, report the differences. Do not edit the reference JSON to make it pass.

- [ ] **Step 7: Lint, full suite, commit.** The `tools/` directory is lint-checked like `src`.

```bash
ruff format src tests tools && ruff check --fix src tests tools && pytest -m "not slow" -W error && mypy
git add pyproject.toml tools/sharc_compare tests/sharc_compare
git commit -m "test: add the SHARC 4.1 black-box comparison tool for PySCF single points"
```

---

### Task 11: Documentation current with P3a

**Files:**
- Modify: `CLAUDE.md`, `USER_MANUAL.md`, `docs/superpowers/plans/2026-11-06-gqteashmd-master-plan.md`

**Interfaces:**
- Consumes: the final code of Tasks 1–10. Read the code, not this plan; fix rounds may have changed details.

- [ ] **Step 1: USER_MANUAL.md** (mark [IMPLEMENTED] and [PLANNED] precisely):
  - **§1:** the electronic-structure layer and `single-point` are implemented. The quantum-chemistry *stages* are still planned.
  - **§4.2 / §4.4:** the full `[electronic]` and `[electronic.casscf]` key tables (types, defaults, allowed values, meaning) with all validator rules. `singlets` includes S0. State labels are S0, S1, … Closed-shell singlet molecules only. Remove the implemented items from §4.4.
  - **§5:** the `gqteashmd single-point INPUT [--json] [--log]` command, an example run with real output (from a WSL scratch directory), the JSON fields, and exit codes 0, 2 and 4.
  - **§8.6 → [IMPLEMENTED].** Define every symbol and unit for each item below.
    - The SCF and its convergence ladder (default → level shift 0.2 Eh with damping 0.5 → second-order).
    - SA-CASSCF:
      - the state-averaged energy functional $E_{\mathrm{SA}} = \sum_{I} w_I E_I$ with $w_I = 1/N$;
      - singlet-only CI;
      - the spin check $|\langle S^2\rangle| \le 10^{-3}$;
      - orbital reuse by projection.
    - TDA/TDDFT/CIS/TDHF: $E_k = E_{\mathrm{SCF}} + \omega_k$.
    - Analytic gradients $\nabla_{\mathbf R} E_I$ in Eh/bohr.
    - The transition dipole $\boldsymbol\mu_{0I} = -\sum_{pq} \mathbf r_{pq}\,\gamma^{0I}_{pq}$ for CASSCF, and the linear-response transition dipoles.
    - The oscillator strength (moved from §8.9 to implemented).
    - The assumptions: TDDFT seam topology, restricted reference, no triplets yet.
    - **Validation results:** a table of ethylene vs SHARC 4.1 with the measured differences (energies, gradients, dipole), and the finite-difference gradient checks with their measured differences.
  - **§9 / §10:**
    - add the new error messages: missing `[electronic]`, basis, SCF/CASSCF convergence, spin, active space, open-shell refusal;
    - remove "CI has never run" (CI passed on 2026-11-07);
    - keep the other limitations.
- [ ] **Step 2: CLAUDE.md:**
  - **Status:** P3a done on branch `feat/p3a-pyscf-backend`, with the test count.
  - **Architecture:** `core/`, `electronic/`, `tools/sharc_compare/`.
  - **Rules:**
    - PySCF is imported lazily, only in `electronic/pyscf_backend.py`;
    - the singlet-only CASSCF solver and the spin guard;
    - orbitals are reused only via `project_init_guess`;
    - finite-difference tests use fresh backends.
  - **Decisions:** O18 (pyscf 2.14.\* pinned) and O19 (singlets include S0).
  - **Known limitations:** CI now passes; NACs, overlaps and the Hessian are not implemented yet.
  - **Remaining work:** P3b (NACs, wavefunction overlaps, phase tracking) and P4 (optimize and frequencies, including the Hessian).
- [ ] **Step 3: Master plan:**
  - O18 resolved (pyscf 2.14.\* pinned in `environment.yml`, runtime `pyscf>=2.6`), O19 resolved.
  - §9: split the P3 row into P3a (delivered) and P3b (NACs, overlaps, phase tracking; needed by P7b), and note that the Hessian moved to P4.
- [ ] **Step 4: Verify** the manual's new `single-point` example verbatim in `mktemp -d /tmp/gq-doc-XXXX`, with `conda activate gqteashmd-dev` first. Remove the scratch directory afterwards.
- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md USER_MANUAL.md docs/superpowers/plans/2026-11-06-gqteashmd-master-plan.md
git commit -m "docs: document the PySCF backend, single-point and validation results (P3a)"
```

---

## Self-review notes

- **Spec coverage:**
  - §2.3 backend contract → Task 5 (NAC and overlap fields come in P3b).
  - §3.1 PySCF SCF/TDA/TDDFT/SA-CASSCF energies, gradients and dipoles → Tasks 6 and 7.
  - Retry ladder → Task 6.
  - Threads and memory → Tasks 6 and 9 (scratch-directory handling is part of P3b, together with chkfiles).
  - §4.2 PySCF version preflight → `require_pyscf` (Task 6); stages declaring `required_modules=("pyscf",)` arrive with P4.
  - D19 ethylene validation → Tasks 6, 8 and 10.
  - O19 → Task 2.
  - The P2 residual → Task 1.
  - Documentation → Task 11.
- **Type consistency:**
  - `StateLabel(multiplicity, index)` is used identically everywhere.
  - `Request(coords_bohr, gradients, transition_dipoles)` and `ElectronicResult(states, energies, gradients, transition_dipoles, spin_squares, meta)` have the same fields in Tasks 5–9.
  - `PySCFBackend(molecule, settings, *, log_file, threads, memory_mb)` has the same signature in Tasks 6–9.
  - `make_settings` in `tests/electronic/conftest.py` passes `casscf=None` explicitly for TDA settings.
