# Triplet states in the PySCF backend: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `[electronic] triplets > 0` work for SA-CASSCF and TDA/TDDFT in the PySCF backend (energies, analytic gradients, CASSCF triplet-triplet NACs, overlaps and phases, `single-point`, the `excitations` stage), with no spin-orbit coupling yet, and with `triplets = 0` exactly as today.

**Architecture:** Spin-free triplet states `T1…Tm` join the singlets as their own labels (approach A). SA-CASSCF uses `mcscf.state_average_mix_` over a singlet and a triplet FCI solver with equal weights; TDA/TDDFT runs a second response solve with `singlet = False`. The determinant expansion becomes one expansion per multiplicity block, and the overlap matrix is block-diagonal with exactly zero singlet-triplet entries. Stages see the triplets only in `excitations` (a new dataset); `dynamics` refuses them until sub-project 3.

**Tech Stack:** Python 3.12, PySCF 2.14 (`mcscf.state_average_mix_`, `fci.addons.fix_spin_`, `tdscf` with `singlet = False`), NumPy, h5py, pytest; ORCA 6.1.1 and SHARC 4.1 as black boxes for validation; everything runs in WSL Oracle Linux 9.5.

**Spec:** `docs/superpowers/specs/2026-11-26-triplets-backend-design.md` (owner-approved 2026-11-26). Read it before any task; the plan rulings below are binding where they differ.

## Global Constraints

- Run every command in WSL: `wsl -d OracleLinux_9_5 -- bash -lc 'cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && conda activate gqteashmd-dev && <command>'`. Below, `W '<command>'` stands for that wrapper. ORCA: `ORCA_EXE=/home/ajc/bin/orca_6_1_1/orca`. SHARC: `SHARC_HOME=/home/ajc/sharc4-4.1 SHARC_PYTHON=/home/ajc/miniforge3/envs/sharc4.1/bin/python`.
- Git from WSL only (pre-commit hooks run there); never `--no-verify`. One commit per task, conventional subjects; end each commit message with the `Co-Authored-By:` trailer of the model that writes the commit (CLAUDE.md rule).
- Files with LF line endings. Lint: `ruff format . && ruff check .` (line length 100); `mypy` clean on `src/`.
- **Clean room:** SHARC is executed only (from `tools/sharc_compare/`), never read, copied or imported. ORCA is executed only.
- **PySCF is imported lazily, only inside `electronic/pyscf_*.py` functions.** `import gqteashmd.cli` must not import pyscf, geometric, h5py or scipy.
- **`triplets = 0` is unchanged:** the same PySCF calls (the singlet-only CI solver, no triplet response solve), the same numbers, the same saved-state bytes (format 1), the same stage outputs and fingerprints; no stage version changes. The existing test suite stays green without modification, except where a task below says otherwise.
- Tests: `pytest ... -W error`; a new test must not compare BLAS/LAPACK results bit for bit or rely on an eigenvector's sign; check new tests once with `OPENBLAS_CORETYPE=SandyBridge`. PySCF tests carry the marker `pyscf`; slow ones also `slow`; ORCA ones `orca` (skip without `ORCA_EXE`); SHARC ones `sharc` (skip without `SHARC_HOME`/`SHARC_PYTHON`).
- Fixed numbers (spec 4): ORCA energies 1e-6 Eh; FD gradients step 1e-3 bohr, |analytic - FD| <= 1e-5 Eh/bohr; triplet-triplet NACs 1e-4 1/bohr; overlap continuity: triplet-block smallest singular value >= 0.99 on 0.01-bohr steps, positive diagonal after tracking, singlet-triplet entries exactly 0, resumed vs uninterrupted phased overlaps 1e-10; SHARC (if available) 1e-8 Eh and 1e-7 Eh/bohr. Spin guard 1e-3 (`SPIN_TOLERANCE`): singlets `|<S^2>| <= 1e-3`, triplets `|<S^2> - 2| <= 1e-3`.
- Error messages for the singlet-only path keep their current text (existing tests assert them).
- The ledger is `.superpowers/sdd/2026-11-26-triplets-backend/progress.md` (git-ignored); every task writes `task-N-report.md` there with the commands run and their counts.

## Plan rulings (binding where they differ from the spec)

- **PR1, the TDA/TDDFT triplet expansion (spec 3.4).** TDA/TDDFT triplets are expanded in their **Ms = 0** component, `sum_ia X_ia (|i->a, alpha> - |i->a, beta>)`, in exactly the determinant layout of the singlets (whose excited part is `+`). The spec's Ms = +1 component would need a spin-flip determinant sign convention the code does not have; Ms = 0 reuses the validated singlet layout, and triplet-triplet overlaps are the same for every Ms component (Wigner-Eckart). CASSCF triplets keep their Ms = +1 CI vectors (the triplet solver's own determinant space). The cross blocks are still exactly zero, because each block's overlap is computed separately and the off-diagonal blocks are set to zero.
- **PR2, intermediate commits.** Task 2 lets the schema accept `triplets > 0` before the backend supports it (Tasks 3-5) and before `dynamics` refuses it (Task 6). The branch is merged only when complete.
- **PR3, the FD components (spec 4 item 2).** As in the existing FD tests (`tests/electronic/test_gradients_fd.py`), each FD test compares a fixed set of Cartesian components (three per state for the response tests, two per state for CASSCF), not every component; the bound applies to the largest difference over that set.
- **PR4, the NAC system (spec 4 item 3).** Ethylene CAS(2,2) has a single triplet, so the triplet-triplet NAC test uses water CAS(4,4)/STO-3G with 2 singlets and 2 triplets (15 triplet CSFs available).
- **PR5, the saved electronic state.** `save_state` writes format 1, byte-identical to today, when there is no triplet block; with triplets it writes format 2 (format 1's arrays plus `t_alpha`, `t_beta`, `t_coeffs`). `load_state` reads both; a format-1 file for a backend with triplets is unreadable ("missing the triplet expansion").
- **PR6, `single-point` table.** An `S^2` column is added only when the backend has triplets; the `triplets = 0` table is unchanged.
- **PR7, the excitations reference.** The `reference` group of `excitations.h5` carries no triplet data; `excitations.json` carries the reference triplet excitation energies.
- **PR8, a failing NAC.** If PySCF's `state_average_mix_` NACs fail the FD test for a triplet pair (Task 7), the implementer stops (BLOCKED) and reports the numbers; no workaround in the task. The controller rules (for example: no triplet-triplet NACs, so sub-project 3 would use overlaps with triplets).

## File structure

| File | Task | Responsibility |
|---|---|---|
| `src/gqteashmd/config/schema.py` | 2 | accept `triplets > 0`; `n_triplet_csfs` and the CASSCF check |
| `src/gqteashmd/electronic/pyscf_response.py` | 3 | the triplet response solve, gradients, zero dipoles, the Ms = 0 triplet expansion |
| `src/gqteashmd/electronic/pyscf_casscf.py` | 4 | the mixed solver, spin guard, ladder and reference, gradients/NACs by state position, dipoles, block expansions |
| `src/gqteashmd/electronic/pyscf_backend.py` | 5 | block snapshot, block overlaps and phases, format 2 save/load, triplet live objects |
| `src/gqteashmd/cli.py` | 5 | `single-point` `S^2` column with triplets |
| `src/gqteashmd/electronic/factory.py` | 6 | ground-state settings drop the triplets for SCF methods |
| `src/gqteashmd/stages/excitations.py`, `src/gqteashmd/excite/artifacts.py` | 6 | `triplet_energies`, triplet labels, triplet summary |
| `src/gqteashmd/stages/dynamics.py` | 6 | refuse `triplets > 0` |
| `tests/config/…`, `tests/electronic/…`, `tests/stages/…`, `tests/validation/…`, `tools/sharc_compare/…` | 2-7 | tests and harnesses |
| documents | 8 | USER_MANUAL, CLAUDE.md, CHANGELOG, report |

---

### Task 1: what `SHARC_PYSCF.py` supports for triplets (a measurement, no code)

**Files:** none in the repository; ledger `task-1-report.md` (and the raw `QM.in`, `PYSCF.template`, `QM.out` copied there).

**Interfaces:** Produces the facts Task 7 needs: whether `SHARC_PYSCF.py` returns triplet energies and gradients, and the exact `QM.in` and `QM.out` text (state counts line, triplet ordering and Ms components, units).

- [ ] **Step 1:** In a scratch directory under `~/triplets_sharc/` in WSL, write a `QM.in` like `tools/sharc_compare/run_sharc_pyscf.py:run_geometry` writes (ethylene, `tests/data/ethylene.xyz`, Angstrom), with the state line `states 2 0 1` (singlets doublets triplets, the SHARC convention for state counts) and the requests `h`, `grad all`; and the `PYSCF.template` of `run_sharc_pyscf.py` (`TEMPLATE`, ncas 2, nelecas 2, roots 2, basis 6-31g*). Run `SHARC_PYSCF.py QM.in` exactly as `run_geometry` does (the same `SHARC`, `PYTHONPATH` environment).
- [ ] **Step 2:** Record: exit status; whether `QM.out` has 2 + 3 x 1 = 5 states (the triplet in three Ms components) with energies on the diagonal of `h`; whether gradients exist for the triplet components; the triplet energy against our expectation (to be compared in Task 7); any template key the interface needs for triplets (if the run fails, try once with a template key that the error message names; do not read SHARC source). Also try water CAS(4,4)/STO-3G with `states 2 0 2` if ethylene works.
- [ ] **Step 3:** Write `task-1-report.md`: supported yes/no, the formats, the commands. If not supported, Task 7's SHARC test is written as a skip with the recorded reason.

---

### Task 2: input: accept triplets; the CASSCF triplet count

**Files:**
- Modify: `src/gqteashmd/config/schema.py` (`n_singlet_csfs` neighbour, `ElectronicSection._method_options`)
- Test: `tests/config/test_electronic_section.py` (or the file that holds the `[electronic]` validation tests; find it with `grep -rn "not supported yet" tests/config`)

**Interfaces:**
- Produces: `n_triplet_csfs(ncas: int, nelecas: int) -> int`; `ElectronicSection` accepts `triplets > 0` for every method; for CASSCF `triplets` is checked against `n_triplet_csfs`.

- [ ] **Step 1: Write the failing tests**

```python
from gqteashmd.config.schema import ElectronicSection, n_singlet_csfs, n_triplet_csfs


def test_triplet_csf_counts():
    assert n_triplet_csfs(2, 2) == 1
    assert n_triplet_csfs(3, 2) == 3
    assert n_triplet_csfs(4, 4) == 15
    assert n_singlet_csfs(4, 4) == 20  # unchanged


def test_triplets_are_accepted():
    for method, extra in (
        ("tda", {"functional": "hf"}),
        ("tddft", {"functional": "b3lyp"}),
        ("casscf", {"casscf": {"ncas": 2, "nelecas": 2}}),
    ):
        s = ElectronicSection.model_validate(
            {"method": method, "basis": "sto-3g", "singlets": 2, "triplets": 1, **extra}
        )
        assert s.triplets == 1


def test_casscf_triplets_beyond_the_active_space_are_refused():
    with pytest.raises(ValueError, match="electronic.triplets = 2 exceeds the 1 triplet states"):
        ElectronicSection.model_validate(
            {
                "method": "casscf",
                "basis": "sto-3g",
                "singlets": 2,
                "triplets": 2,
                "casscf": {"ncas": 2, "nelecas": 2},
            }
        )
```

Delete or rewrite the existing test that asserts the old refusal text (`electronic.triplets > 0 is not supported yet`); say which in the report.

- [ ] **Step 2: Run them to verify they fail**

Run: `W 'pytest tests/config -q -W error'`
Expected: the three new tests fail (`ImportError: n_triplet_csfs`, the refusal).

- [ ] **Step 3: Implement**

```python
def n_triplet_csfs(ncas: int, nelecas: int) -> int:
    """Number of triplet CSFs of a CAS(nelecas, ncas) space (Weyl formula with S = 1, nelecas
    even): 3/(n+1) C(n+1, N/2 - 1) C(n+1, N/2 + 2)."""
    n, half = ncas, nelecas // 2
    if half < 1:
        return 0
    return 3 * math.comb(n + 1, half - 1) * math.comb(n + 1, half + 2) // (n + 1)
```

In `ElectronicSection._method_options`: remove the `if self.triplets:` refusal; inside `if self.casscf is not None:` add, after the singlet check:

```python
                available_t = n_triplet_csfs(self.casscf.ncas, self.casscf.nelecas)
                if self.triplets > available_t:
                    problems.append(
                        f"electronic.triplets = {self.triplets} exceeds the {available_t} triplet "
                        f"states of CAS({self.casscf.nelecas},{self.casscf.ncas}) "
                        f"(nelecas = {self.casscf.nelecas}, ncas = {self.casscf.ncas})"
                    )
```

Update the `init` template comment for `triplets` if it says "must be 0" (`grep -rn "triplets" src/gqteashmd/cli.py src/gqteashmd/config`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `W 'pytest tests/config tests/test_cli.py -q -W error -m "not slow and not pyscf"'`
Expected: all pass.

- [ ] **Step 5: Lint, type-check, commit** (`feat: accept triplet states in [electronic]`).

---

### Task 3: TDA/TDDFT triplets (`pyscf_response.py`)

**Files:**
- Modify: `src/gqteashmd/electronic/pyscf_response.py`
- Test: `tests/electronic/test_pyscf_linear_response.py`

**Interfaces:**
- Produces:
  - `ResponseOutcome(td, rung, energies, td_triplet=None, triplet_rung=None)`;
  - `run_response(mf, settings, states)`: singlet excited states solved as today, triplets `T1…Tm` by `solve_triplets(mf, settings, n, n_single) -> (td, rung)` (module-level, so a test can replace it);
  - `response_gradients(mf, td, labels, method, td_triplet=None)`;
  - `response_dipoles(td, states, method)`: singlets as today, every triplet `np.zeros(3)`;
  - `response_expansion(mf, td, n_excited)` unchanged (the caller passes the number of excited **singlets**);
  - `response_triplet_expansion(mf, td_triplet, n_triplets) -> DeterminantExpansion` (PR1: Ms = 0, `+X` on the alpha replacements, `-X` on the beta ones).

- [ ] **Step 1: Write the failing tests** (marker `pyscf`; use the file's existing water fixtures or build water as in `tests/conftest.py:WATER_XYZ`)

```python
import numpy as np
import pytest
from pyscf import gto, scf, tdscf

from gqteashmd.config.schema import ElectronicSection
from gqteashmd.core.states import StateLabel, states_from_counts
from gqteashmd.electronic import pyscf_response
from gqteashmd.overlaps.determinants import state_overlaps, truncate

WATER = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"


def _water_rhf():
    mol = gto.M(atom=WATER, basis="sto-3g", verbose=0)
    return scf.RHF(mol).run(conv_tol=1e-11)


def _settings(singlets, triplets):
    return ElectronicSection.model_validate(
        {"method": "tda", "basis": "sto-3g", "functional": "hf", "singlets": singlets,
         "triplets": triplets, "td_conv_tol": 1e-9}
    )


def test_triplet_energies_equal_a_direct_triplet_tda():
    mf = _water_rhf()
    states = states_from_counts(2, 3)
    out = pyscf_response.run_response(mf, _settings(2, 3), states)
    td = tdscf.TDA(mf)
    td.singlet = False
    td.nstates = 6
    td.conv_tol = 1e-9
    td.kernel()
    for i in range(3):
        assert out.energies[StateLabel(3, i + 1)] == pytest.approx(mf.e_tot + td.e[i], abs=1e-9)
    assert out.td_triplet is not None and out.triplet_rung == "default"


def test_singlets_do_not_change_when_triplets_are_added():
    mf = _water_rhf()
    a = pyscf_response.run_response(mf, _settings(3, 0), states_from_counts(3, 0))
    b = pyscf_response.run_response(mf, _settings(3, 2), states_from_counts(3, 2))
    for label, e in a.energies.items():
        assert b.energies[label] == pytest.approx(e, abs=1e-12)


def test_triplet_transition_dipoles_are_zero_and_gradients_exist():
    mf = _water_rhf()
    states = states_from_counts(2, 1)
    out = pyscf_response.run_response(mf, _settings(2, 1), states)
    mus = pyscf_response.response_dipoles(out.td, states, "tda")
    np.testing.assert_array_equal(mus[StateLabel(3, 1)], np.zeros(3))
    assert np.linalg.norm(mus[StateLabel(1, 1)]) >= 0.0
    g = pyscf_response.response_gradients(
        mf, out.td, {StateLabel(3, 1)}, "tda", td_triplet=out.td_triplet
    )
    assert g[StateLabel(3, 1)].shape == (3, 3) and np.all(np.isfinite(g[StateLabel(3, 1)]))


def test_the_triplet_expansion_is_orthonormal_and_orthogonal_to_the_singlets():
    mf = _water_rhf()
    states = states_from_counts(3, 3)
    out = pyscf_response.run_response(mf, _settings(3, 3), states)
    singlet = truncate(pyscf_response.response_expansion(mf, out.td, 2), 1.0)
    triplet = truncate(pyscf_response.response_triplet_expansion(mf, out.td_triplet, 3), 1.0)
    eye = np.eye(mf.mo_coeff.shape[1])
    np.testing.assert_allclose(state_overlaps(triplet, triplet, eye), np.eye(3), atol=1e-8)
    np.testing.assert_allclose(state_overlaps(singlet, triplet, eye), 0.0, atol=1e-10)


def test_without_triplets_no_triplet_solve_runs(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("a triplet solve ran with triplets = 0")

    monkeypatch.setattr(pyscf_response, "solve_triplets", boom)
    mf = _water_rhf()
    out = pyscf_response.run_response(mf, _settings(3, 0), states_from_counts(3, 0))
    assert out.td_triplet is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `W 'pytest tests/electronic/test_pyscf_linear_response.py -q -W error'`
Expected: the new tests fail (`AttributeError: ... td_triplet`, `solve_triplets`, `response_triplet_expansion`).

- [ ] **Step 3: Implement**

```python
@dataclass(eq=False)
class ResponseOutcome:
    td: Any
    rung: str | None
    energies: dict[StateLabel, float]
    td_triplet: Any = None
    triplet_rung: str | None = None


def _solve(
    mf: Any, settings: ElectronicSection, n_roots: int, n_single: int, *, singlet: bool
) -> tuple[Any, str]:
    from pyscf import tddft

    method = settings.method.upper()
    label = method if singlet else f"{method} triplets"
    for name, extra, cycle_factor in _LADDER:
        td = tddft.TDA(mf) if settings.method == "tda" else tddft.TDDFT(mf)
        if not singlet:
            td.singlet = False
        td.nstates = min(n_roots + extra, n_single)
        td.conv_tol = settings.td_conv_tol
        td.max_cycle = td.max_cycle * cycle_factor
        try:
            td.kernel()
        except Exception as exc:
            raise wrap_failure(f"{label} ({name})", exc) from exc
        if _converged_roots(td, n_roots):
            return td, name
    tried = ", ".join(name for name, *_ in _LADDER)
    kind = "" if singlet else "triplet "
    raise ElectronicStructureError(
        f"{method} did not converge for all {n_roots} excited {kind}states (tried: {tried}; "
        f"td_conv_tol = {settings.td_conv_tol})."
    )


def solve_triplets(
    mf: Any, settings: ElectronicSection, n_roots: int, n_single: int
) -> tuple[Any, str]:
    """The triplet response solve (a seam for tests)."""
    return _solve(mf, settings, n_roots, n_single, singlet=False)


def run_response(
    mf: Any, settings: ElectronicSection, states: tuple[StateLabel, ...]
) -> ResponseOutcome:
    method = settings.method.upper()
    excited = [s for s in states[1:] if s.multiplicity == 1]
    triplets = [s for s in states if s.multiplicity == 3]
    energies: dict[StateLabel, float] = {states[0]: float(mf.e_tot)}
    if not excited and not triplets:
        return ResponseOutcome(None, None, energies)
    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    n_single = nocc * (int(mf.mo_coeff.shape[1]) - nocc)
    for group, kind in ((excited, ""), (triplets, "triplet ")):
        if len(group) > n_single:
            raise ElectronicStructureError(
                f"{method} cannot give {len(group)} excited {kind}states: the molecule has only "
                f"{n_single} single excitation{'s' if n_single != 1 else ''} in this basis."
            )
    td = rung = None
    if excited:
        td, rung = _solve(mf, settings, len(excited), n_single, singlet=True)
        for label, omega in zip(excited, td.e[: len(excited)], strict=True):
            energies[label] = float(mf.e_tot + omega)
    td_t = rung_t = None
    if triplets:
        td_t, rung_t = solve_triplets(mf, settings, len(triplets), n_single)
        for label, omega in zip(triplets, td_t.e[: len(triplets)], strict=True):
            energies[label] = float(mf.e_tot + omega)
    return ResponseOutcome(td, rung, energies, td_t, rung_t)
```

The singlet messages and the `wrap_failure` text are unchanged for the singlets. In `response_gradients` add the keyword `td_triplet: Any = None`; for a label with `multiplicity == 3` call `td_triplet.nuc_grad_method().kernel(state=label.index)` (failure text `f"{method.upper()} triplet gradient of {label}"`). `response_dipoles`: compute the singlet dipoles over `[s for s in states[1:] if s.multiplicity == 1]` exactly as today (no call when there is none), then set `np.zeros(3)` for every triplet. Add:

```python
def response_triplet_expansion(mf: Any, td: Any, n_triplets: int) -> DeterminantExpansion:
    """Triplets as their Ms = 0 component over mf.mo_coeff (plan ruling PR1):
    T_k = sum_ia X_k[i, a] (|i->a, alpha> - |i->a, beta>), the singlet layout of
    response_expansion without the ground determinant and with the beta part negated. X only
    (TDDFT's Y is dropped); truncate() renormalises."""
    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    nvir = int(mf.mo_coeff.shape[1]) - nocc
    occ = np.arange(nocc)
    n_single = nocc * nvir
    excited_occ = np.tile(occ, (n_single, 1))
    i_index, a_index = np.divmod(np.arange(n_single), nvir)
    excited_occ[np.arange(n_single), i_index] = nocc + a_index
    ground = np.tile(occ, (n_single, 1))
    alpha = np.vstack([excited_occ, ground])
    beta = np.vstack([ground, excited_occ])
    coeffs = np.zeros((n_triplets, 2 * n_single))
    for k in range(n_triplets):
        x = np.asarray(td.xy[k][0], dtype=float).reshape(-1)
        coeffs[k, :n_single] = x
        coeffs[k, n_single:] = -x
    return DeterminantExpansion(alpha, beta, coeffs)
```

Update the module docstring (triplets; PR1).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `W 'pytest tests/electronic/test_pyscf_linear_response.py tests/electronic/test_factory.py -q -W error'` and the new tests once with `OPENBLAS_CORETYPE=SandyBridge`.
Expected: all pass. If the orthonormality test fails at 1e-8, check PySCF's X normalisation for triplets (`sum x^2` per state) before touching the tolerance; report the measured value.

- [ ] **Step 5: Lint, type-check, commit** (`feat: add TDA/TDDFT triplet states to the response layer`).

---

### Task 4: SA-CASSCF triplets (`pyscf_casscf.py`)

**Files:**
- Modify: `src/gqteashmd/electronic/pyscf_casscf.py`
- Test: `tests/electronic/test_pyscf_casscf.py`, `tests/electronic/test_casscf_guesses.py`

**Interfaces:**
- Produces:
  - `state_counts(states) -> tuple[int, int]` (singlets, triplets);
  - `mixed_solver(mol, mf, settings, n_singlets, n_triplets, rung) -> Any` (module-level, a seam for tests);
  - `_solver(mol, mf, settings, states, rung)` (now takes the state labels; unchanged behaviour without triplets);
  - `_spins(mc, states)`: each root against its own multiplicity;
  - `casscf_gradients(mc, labels, states)`, `casscf_nacs(mc, pairs, states, natom)`, `casscf_dipoles(mol, mc, states)`: roots addressed by `states.index(label)`; singlet-triplet NACs and S0 -> T dipoles are zero arrays without a PySCF call;
  - `casscf_expansions(mc, states) -> tuple[DeterminantExpansion, DeterminantExpansion | None]` (singlet block, triplet block or `None`); `casscf_expansion(mc)` stays for the singlet-only case.
- Consumes: `find_reference`/`run_casscf` callers pass `states` (already the case).

- [ ] **Step 1: Write the failing tests** (marker `pyscf`; water CAS(4,4)/STO-3G, 2 singlets + 2 triplets; ethylene CAS(2,2)/6-31G\* 2 + 1 with the conftest fixtures)

```python
import numpy as np
import pytest

from gqteashmd.core.states import StateLabel, states_from_counts
from gqteashmd.electronic import pyscf_casscf
from gqteashmd.overlaps.determinants import state_overlaps


@pytest.fixture
def water_mol():
    """(Mole, converged RHF) for water at tests/conftest.py:WATER_XYZ, STO-3G (put it in
    tests/electronic/conftest.py if the tests of Task 5 need it too)."""
    from pyscf import gto, scf

    mol = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
                basis="sto-3g", verbose=0)
    return mol, scf.RHF(mol).run(conv_tol=1e-11)


def _water_cas44(make_settings, triplets):
    return make_settings(basis="sto-3g", singlets=2, triplets=triplets,
                         casscf={"ncas": 4, "nelecas": 4, "conv_tol": 1e-10, "conv_tol_grad": 1e-6})


def test_mixed_sa_casscf_has_the_right_spins_and_matches_a_direct_mix(water_mol, make_settings):
    mol, mf = water_mol  # an RHF on the conftest water geometry, STO-3G (add the fixture if absent)
    settings = _water_cas44(make_settings, 2)
    states = states_from_counts(2, 2)
    out = pyscf_casscf.run_casscf(mol, mf, settings, states, None)
    assert [round(out.spin_squares[s]) for s in states] == [0, 0, 2, 2]
    for s in states:
        target = 0.0 if s.multiplicity == 1 else 2.0
        assert abs(out.spin_squares[s] - target) <= pyscf_casscf.SPIN_TOLERANCE
    from pyscf import fci, mcscf
    mc = mcscf.CASSCF(mf, 4, 4)
    s1 = fci.solver(mol, singlet=True); s1.nroots = 2; s1.spin = 0
    s3 = fci.solver(mol, singlet=False); s3.nroots = 2; s3.spin = 2
    mc = mcscf.state_average_mix_(mc, [s1, s3], [0.25] * 4)
    mc.conv_tol = 1e-10; mc.conv_tol_grad = 1e-6
    mc.kernel()
    for s, e in zip(states, mc.e_states, strict=True):
        assert out.energies[s] == pytest.approx(e, abs=1e-8)


def test_singlet_triplet_nacs_and_triplet_dipoles_are_zero(water_mol, make_settings):
    mol, mf = water_mol
    settings = _water_cas44(make_settings, 2)
    states = states_from_counts(2, 2)
    out = pyscf_casscf.run_casscf(mol, mf, settings, states, None)
    s1, t1, t2 = StateLabel(1, 1), StateLabel(3, 1), StateLabel(3, 2)
    nacs = pyscf_casscf.casscf_nacs(out.mc, {(s1, t1)}, states, mol.natm)
    np.testing.assert_array_equal(nacs[(s1, t1)], np.zeros((mol.natm, 3)))
    mus = pyscf_casscf.casscf_dipoles(mol, out.mc, states)
    np.testing.assert_array_equal(mus[t1], np.zeros(3))
    g = pyscf_casscf.casscf_gradients(out.mc, {t2}, states)
    assert g[t2].shape == (mol.natm, 3) and np.all(np.isfinite(g[t2]))


def test_block_expansions(water_mol, make_settings):
    mol, mf = water_mol
    settings = _water_cas44(make_settings, 2)
    states = states_from_counts(2, 2)
    out = pyscf_casscf.run_casscf(mol, mf, settings, states, None)
    singlet, triplet = pyscf_casscf.casscf_expansions(out.mc, states)
    assert singlet.n_states == 2 and triplet is not None and triplet.n_states == 2
    assert triplet.alpha.shape[1] == singlet.alpha.shape[1] + 1  # Ms = +1
    assert triplet.beta.shape[1] == singlet.beta.shape[1] - 1
    eye = np.eye(np.asarray(out.mc.mo_coeff).shape[1])
    np.testing.assert_allclose(state_overlaps(triplet, triplet, eye), np.eye(2), atol=1e-8)


def test_without_triplets_the_singlet_solver_is_used(monkeypatch, water_mol, make_settings):
    def boom(*args, **kwargs):
        raise AssertionError("the mixed solver ran with triplets = 0")

    monkeypatch.setattr(pyscf_casscf, "mixed_solver", boom)
    mol, mf = water_mol
    settings = _water_cas44(make_settings, 0)
    out = pyscf_casscf.run_casscf(mol, mf, settings, states_from_counts(2, 0), None)
    assert len(out.energies) == 2


def test_the_reference_search_counts_guesses_with_the_right_spins(water_mol, make_settings):
    mol, mf = water_mol
    settings = _water_cas44(make_settings, 2)
    reference, problems = pyscf_casscf.find_reference(mol, mf, settings, states_from_counts(2, 2))
    assert reference is not None and reference.n_counted >= 1, problems
```

Add a `water_mol` fixture to `tests/electronic/conftest.py` if none exists (RHF/STO-3G on `tests/conftest.py:WATER_XYZ`, `conv_tol` 1e-11), and an ethylene mixed test (2 singlets + 1 triplet, 6-31G\*) asserting the spins and that S0 of the mixed SA differs from the singlet-only S0 (a sanity check that the state average includes the triplet).

- [ ] **Step 2: Run them to verify they fail**

Run: `W 'pytest tests/electronic/test_pyscf_casscf.py tests/electronic/test_casscf_guesses.py -q -W error -m "not slow"'`
Expected: the new tests fail (the singlet-only solver gives four singlet roots, `casscf_expansions` missing, ...).

- [ ] **Step 3: Implement**

```python
def state_counts(states: Sequence[StateLabel]) -> tuple[int, int]:
    return (
        sum(1 for s in states if s.multiplicity == 1),
        sum(1 for s in states if s.multiplicity == 3),
    )


def mixed_solver(
    mol: Any, mf: Any, settings: ElectronicSection, n_singlets: int, n_triplets: int, rung: Rung
) -> Any:
    """Equal-weight SA-CASSCF over singlets and triplets with one orbital set (spec 3.2)."""
    from pyscf import fci, mcscf

    cas = settings.casscf
    assert cas is not None
    singlet = fci.solver(mol, singlet=True)
    singlet.nroots = n_singlets
    singlet.spin = 0
    triplet = fci.solver(mol, singlet=False)
    triplet.nroots = n_triplets
    triplet.spin = 2  # Ms = +1 in the active space
    if rung.spin_shift:
        singlet = fci.addons.fix_spin_(singlet, shift=rung.spin_shift, ss=0)
        triplet = fci.addons.fix_spin_(triplet, shift=rung.spin_shift, ss=2)
    n = n_singlets + n_triplets
    mc = mcscf.CASSCF(mf, cas.ncas, cas.nelecas)
    mc = mcscf.state_average_mix_(mc, [singlet, triplet], [1.0 / n] * n)
    mc.conv_tol = cas.conv_tol
    mc.conv_tol_grad = cas.conv_tol_grad
    mc.max_cycle_macro = mc.max_cycle_macro * rung.macro_factor
    return mc


def _solver(
    mol: Any, mf: Any, settings: ElectronicSection, states: Sequence[StateLabel], rung: Rung
) -> Any:
    n_singlets, n_triplets = state_counts(states)
    if n_triplets:
        return mixed_solver(mol, mf, settings, n_singlets, n_triplets, rung)
    # ... the existing singlet-only body, unchanged, with n_states = n_singlets
```

`_spins(mc, states)`: for each `(label, ci)` use `nelec = (na, nb)` for singlets and `(na + 1, nb - 1)` for triplets, where `(na, nb) = mc.nelecas`, `fci.spin_op.spin_square0(ci, mc.ncas, nelec)[0]`, target 0 or 2; a root is wrong when `abs(s - target) > SPIN_TOLERANCE`; the problem text stays `not all roots are singlets (...)` when there is no triplet and is `not all roots have the expected spin (...)` otherwise (in `run_casscf` and `find_reference`). Pass `states` to `_solver` in `find_reference` and `run_casscf`. `Reference.n_counted`'s comment: "converged with every root of the expected spin".

```python
def casscf_gradients(
    mc: Any, labels: Iterable[StateLabel], states: Sequence[StateLabel]
) -> dict[StateLabel, np.ndarray]:
    # as today, with solver.kernel(state=states.index(label))


def casscf_nacs(
    mc: Any,
    pairs: Iterable[tuple[StateLabel, StateLabel]],
    states: Sequence[StateLabel],
    natom: int,
) -> dict[tuple[StateLabel, StateLabel], np.ndarray]:
    # as today for pairs of the same multiplicity, with
    # solver.kernel(state=(states.index(a), states.index(b))); the solver is built only if such
    # a pair exists; a singlet-triplet pair gives np.zeros((natom, 3)) without a call


def casscf_dipoles(mol: Any, mc: Any, states: Sequence[StateLabel]) -> dict[StateLabel, np.ndarray]:
    # singlets: as today, with ci = mc.ci[states.index(label)]; with triplets present the
    # transition density comes from pyscf.fci.direct_spin1.trans_rdm1(ci0, ci, ncas, nelecas)
    # (mc.fcisolver is the mixed solver); without triplets mc.fcisolver.trans_rdm1 as today;
    # every triplet: np.zeros(3)


def casscf_expansions(
    mc: Any, states: Sequence[StateLabel]
) -> tuple[DeterminantExpansion, DeterminantExpansion | None]:
    """The singlet block (Ms = 0, mc.nelecas) and the triplet block (Ms = +1, one more alpha and
    one fewer beta electron) of the SA-CASSCF states, each over mc.mo_coeff with the layout of
    casscf_expansion."""
```

Implement `casscf_expansions` with a helper `_expansion(ncore, ncas, n_alpha, n_beta, cis)` holding today's `casscf_expansion` body (occupation strings and the coefficient reshape), so that `casscf_expansion(mc)` calls the helper with `mc.nelecas` and all `mc.ci` (unchanged result) and `casscf_expansions` calls it once per block (`mc.ci[:n_singlets]`, `mc.ci[n_singlets:]`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `W 'pytest tests/electronic -q -W error -m "not slow"'` (expect about 13 min) and the new tests once with `OPENBLAS_CORETYPE=SandyBridge`.
Expected: all pass, the existing ones unchanged.

- [ ] **Step 5: Lint, type-check, commit** (`feat: add SA-CASSCF triplet states`).

---

### Task 5: backend integration: blocks, overlaps, phases, saved state, `single-point`

**Files:**
- Modify: `src/gqteashmd/electronic/pyscf_backend.py`, `src/gqteashmd/cli.py` (`_cmd_single_point`)
- Test: `tests/electronic/test_state.py`, new `tests/electronic/test_triplet_overlaps.py`, `tests/test_cli_single_point.py`

**Interfaces:**
- Consumes (Tasks 3-4): `ResponseOutcome.td_triplet/.triplet_rung`, `response_triplet_expansion`, `response_gradients(..., td_triplet=)`, `casscf_expansions`, `casscf_gradients(mc, labels, states)`, `casscf_nacs(mc, pairs, states, natom)`.
- Produces: `_Snapshot.triplet_expansion: DeterminantExpansion | None`; `_Live.triplet_solver: Any = None`; saved state format 2 (PR5); `ElectronicResult.overlaps` block-diagonal; `meta["td_triplet"]` with triplets.

- [ ] **Step 1: Write the failing tests**

`tests/electronic/test_triplet_overlaps.py` (marker `pyscf`; spec 4 item 4):

```python
import numpy as np
import pytest

from gqteashmd.core.units import ANGSTROM_TO_BOHR
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.pyscf_backend import PySCFBackend

pytestmark = pytest.mark.pyscf
PATH_STEP = 0.01  # bohr, every coordinate
WATER_SYMBOLS = ("O", "H", "H")


@pytest.fixture
def water_bohr():
    return np.array(
        [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]
    ) * ANGSTROM_TO_BOHR


def _settings(method, make_settings):
    if method == "tda":
        return make_settings(method="tda", basis="sto-3g", functional="hf", casscf=None,
                             singlets=2, triplets=2, scf_conv_tol=1e-11, td_conv_tol=1e-9)
    return make_settings(basis="sto-3g", singlets=2, triplets=2, scf_conv_tol=1e-11,
                         casscf={"ncas": 4, "nelecas": 4, "conv_tol": 1e-10,
                                 "conv_tol_grad": 1e-6})


def _path(coords, n=5):
    return [coords + PATH_STEP * k for k in range(n)]


@pytest.mark.parametrize("method", ["tda", "casscf"])
def test_triplet_overlaps_are_continuous_along_a_path(method, water_bohr, make_settings):
    settings = _settings(method, make_settings)  # TDA-HF/STO-3G 2 + 2, or CAS(4,4)/STO-3G 2 + 2
    backend = PySCFBackend(MoleculeSpec(WATER_SYMBOLS), settings)
    n_s = settings.singlets
    for k, coords in enumerate(_path(water_bohr)):
        result = backend.compute(Request(coords, overlaps=True))
        if k == 0:
            continue
        s = result.overlaps
        triplet = s[n_s:, n_s:]
        assert np.linalg.svd(triplet, compute_uv=False).min() >= 0.99
        assert np.all(np.diag(s) > 0)
        assert np.all(s[:n_s, n_s:] == 0.0) and np.all(s[n_s:, :n_s] == 0.0)


@pytest.mark.parametrize("method", ["tda", "casscf"])
def test_a_resumed_backend_gives_the_same_phased_overlaps(method, water_bohr, make_settings, tmp_path):
    settings = _settings(method, make_settings)
    path = _path(water_bohr)
    whole = PySCFBackend(MoleculeSpec(WATER_SYMBOLS), settings)
    reference = [whole.compute(Request(c, overlaps=True)).overlaps for c in path]
    first = PySCFBackend(MoleculeSpec(WATER_SYMBOLS), settings)
    for c in path[:3]:
        first.compute(Request(c, overlaps=True))
    first.save_state(tmp_path / "state.npz")
    resumed = PySCFBackend(MoleculeSpec(WATER_SYMBOLS), settings)
    resumed.load_state(tmp_path / "state.npz")
    for k in (3, 4):
        s = resumed.compute(Request(path[k], overlaps=True)).overlaps
        np.testing.assert_allclose(s, reference[k], atol=1e-10)
```

In `tests/electronic/test_state.py` add: a `triplets = 0` backend's saved file has `format == 1` and no `t_alpha` key (`np.load`), and a triplets backend's file has `format == 2` and the three `t_` arrays; a format-1 file loaded into a triplets backend raises `ValueError` (`missing the triplet expansion` or the fingerprint message, whichever the code reaches first; assert `ValueError`). In `tests/test_cli_single_point.py` add a run with `triplets = 1` (TDA-HF/STO-3G water): the table has an `S^2` column and a `T1` row with `f` `0.0000`; the JSON has a `T1` row with `oscillator_strength` 0.0; and a `triplets = 0` run whose table header is unchanged.

- [ ] **Step 2: Run them to verify they fail**

Run: `W 'pytest tests/electronic/test_triplet_overlaps.py tests/electronic/test_state.py tests/test_cli_single_point.py -q -W error'`
Expected: the new tests fail (overlaps are not block-diagonal or `compute` fails on the triplet labels).

- [ ] **Step 3: Implement** in `pyscf_backend.py`:

- `_Snapshot`: add `triplet_expansion: DeterminantExpansion | None = None` (last field). `_Live`: add `triplet_solver: Any = None`.
- `compute`:
  - CASSCF: `expansion, triplet_expansion = pyscf_casscf.casscf_expansions(solver, self._states)`.
  - Response: `n_s, n_t = pyscf_casscf.state_counts(self._states)`; `expansion = pyscf_response.response_expansion(mf, solver, n_s - 1)`; `triplet_expansion = pyscf_response.response_triplet_expansion(mf, outcome_r.td_triplet, n_t) if n_t else None`; `rung_meta = {"td": outcome_r.rung}` plus `rung_meta["td_triplet"] = outcome_r.triplet_rung` when `n_t`; `triplet_solver = outcome_r.td_triplet`.
  - Truncate both blocks (`truncate` inside the same `try`).
  - `self._state_overlaps(previous, mol, mo_coeff, expansion, triplet_expansion)` builds the block-diagonal matrix: the singlet block from `state_overlaps(previous.expansion, expansion, s_mo)`, the triplet block from `state_overlaps(previous.triplet_expansion, triplet_expansion, s_mo)` when present, zeros elsewhere.
  - After `track`: `expansion = expansion.with_signs(phases[:n_s])`, `triplet_expansion = triplet_expansion.with_signs(phases[n_s:])` when present.
  - `_Live(mol, mf, solver, energies, phases, triplet_solver)`; `_Snapshot(..., expansion, triplet_expansion)`.
- `_derivatives`: CASSCF `pyscf_casscf.casscf_gradients(live.solver, gradients, self._states)` and `pyscf_casscf.casscf_nacs(live.solver, nacs, self._states, live.mol.natm)`; response `response_gradients(live.mf, live.solver, gradients, method, td_triplet=live.triplet_solver)`.
- `_dipoles`: response `response_dipoles(live.solver, self._states, method)` (now zero for triplets); CASSCF unchanged call (the function handles triplets).
- `save_state`: `_STATE_FORMAT_TRIPLETS = 2`; without a triplet block write exactly today's file (format 1); with one, add `t_alpha`, `t_beta`, `t_coeffs` and `format=2`.
- `load_state`: accept formats 1 and 2; format 2 reads the `t_` arrays into a `DeterminantExpansion`; after the fingerprint check, a backend with triplets and a file without a triplet block raises `unreadable("missing the triplet expansion")`, and a backend without triplets and a file with one raises `unreadable("unexpected triplet expansion")`.

In `cli.py:_cmd_single_point`: when `settings.triplets > 0`, print the header with an extra `S^2` column (`f"{'S^2':>8}"`) and each row's `spin_square` (`-` when `None`); otherwise print exactly today's table.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `W 'pytest tests/electronic tests/test_cli_single_point.py tests/test_cli.py -q -W error -m "not slow"'` and the new tests once with `OPENBLAS_CORETYPE=SandyBridge`.
Expected: all pass, the existing ones unchanged.

- [ ] **Step 5: Lint, type-check, commit** (`feat: carry triplet states through the PySCF backend`).

---

### Task 6: stages: `excitations`, the `dynamics` refusal, the ground-state settings

**Files:**
- Modify: `src/gqteashmd/electronic/factory.py`, `src/gqteashmd/stages/excitations.py`, `src/gqteashmd/excite/artifacts.py`, `src/gqteashmd/stages/dynamics.py`
- Test: `tests/electronic/test_factory.py`, `tests/stages/test_excitations.py` (find the existing file with `grep -rln "ExcitationsStage" tests`), `tests/stages/test_dynamics.py`, `tests/excite/test_artifacts.py` (or where `write_excitations_h5` is tested)

**Interfaces:**
- Produces: `ground_state_settings` sets `singlets = 1` and `triplets = 0` for SCF methods; `write_excitations_h5(..., triplet_labels: Sequence[str] = (), triplet_energies_eV: ArrayLike | None = None)` writes the datasets `triplet_labels` and `triplet_energies` (units `eV`) only when triplets exist; `read_excitations_h5` returns `triplet_labels` (tuple) and `triplet_energies_eV` (or `None`); `excitations.json` gains `triplet_labels` and `triplet_states` (per triplet: mean and std excitation eV, reference excitation eV) only when triplets exist, so that `triplets = 0` outputs stay identical.

- [ ] **Step 1: Write the failing tests**

- `test_factory.py`: for `tda` with `singlets = 3, triplets = 2`, `ground_state_settings(s).triplets == 0` and `singlets == 1`; `ground_state_fingerprint` is equal for `triplets = 0` and `triplets = 2` (TDA); for CASSCF it differs and `ground_state_settings` keeps the triplets.
- `test_dynamics.py`: `DynamicsStage().check_config(cfg)` with `[electronic] triplets = 1` returns exactly `["Stage 'dynamics' with [electronic] triplets > 0 is planned (spin-orbit coupling, sub-project 3); set [electronic] triplets = 0"]` (in addition to nothing else for a valid config).
- `test_excitations.py` (PySCF, like the existing stage tests that run real water samples; if the existing ones use a canned backend, follow that file's pattern): with `triplets = 1` the stage writes `triplet_energies` of shape (n_samples, 1), finite and > 0 for OK samples, units `eV`, `triplet_labels == ("T1",)`; the singlet `energies`, `oscillator_strengths` and `spectrum.csv` equal those of a `triplets = 0` run on the same samples (allclose 1e-10; spectrum text equal); `excitations.json` has `triplet_states["T1"]`; a `triplets = 0` run's `excitations.h5` has no `triplet_energies` and its `excitations.json` no `triplet_states`.
- artifacts: a round trip of `write_excitations_h5`/`read_excitations_h5` with and without triplet data.

- [ ] **Step 2: Run them to verify they fail.** `W 'pytest tests/electronic/test_factory.py tests/stages/test_dynamics.py <the excitations and artifacts test files> -q -W error'`.

- [ ] **Step 3: Implement**

- `factory.ground_state_settings`: `return settings.model_copy(update={"singlets": 1, "triplets": 0})` for non-CASSCF (CASSCF unchanged). The fingerprint of existing projects is unchanged because their `triplets` is already 0.
- `stages/dynamics.py:check_config`: append the refusal when `config.electronic is not None and config.electronic.triplets > 0`.
- `stages/excitations.py`:
  - `run_task`: `singlets = [s for s in result.states if s.multiplicity == 1]`, `triplets = [s for s in result.states if s.multiplicity == 3]`; `energies`, `dipoles`, `strengths` over `singlets` exactly as today (with `singlets` in place of `states`); `triplet_energies = [float(result.energies[t]) for t in triplets]`; `_invalid_excitations` gets a keyword `triplet_energies: list[float] = ()` with the triplet labels and checks finite and `(E_T - E_S0) > 0` (`excitation energy of T1 is not positive (...)`, `non-finite energy of T1 (...)`); the row gains `"triplet_energies_hartree": triplet_energies` only when triplets exist.
  - `finalize`: `triplet_labels = [str(s) for s in states_from_counts(electronic.singlets, electronic.triplets) if s.multiplicity == 3]`; when non-empty, a `(n_samples, m)` NaN array filled with `(E_T - E_S0) * HARTREE_IN_EV` of the OK rows, passed to `write_excitations_h5`; `excitations.json` gains `triplet_labels` and `triplet_states` (mean/std over OK samples, ddof 0, and the reference value from the `ref` row when `include_reference`), and a log line per triplet like the singlet one.
- `excite/artifacts.py`: the two optional datasets as above (`units` attribute `eV` on `triplet_energies`; `triplet_labels` as fixed-length strings like `state_labels`), shape-checked on read.

- [ ] **Step 4: Run the tests to verify they pass.** `W 'pytest tests/electronic/test_factory.py tests/stages tests/excite -q -W error -m "not slow"'`. Expected: all pass, existing unchanged.

- [ ] **Step 5: Lint, type-check, commit** (`feat: record triplet energies in the excitations stage`).

---

### Task 7: validation: ORCA, finite differences, NACs, SHARC

**Files:**
- Create: `tests/validation/test_orca_triplets.py`, `tests/validation/test_sharc_triplets.py` (per Task 1's result)
- Modify: `tests/electronic/test_gradients_fd.py`, `tests/electronic/test_couplings_fd.py`; `tools/sharc_compare/run_sharc_pyscf.py` (a `states` argument for `run_geometry`, only if Task 1 found triplets supported)

**Interfaces:** Consumes the backend of Tasks 3-5 and Task 1's report.

- [ ] **Step 1: ORCA** (`tests/validation/test_orca_triplets.py`, markers `orca`, `pyscf`, skip without `ORCA_EXE`; the subprocess pattern of `tests/validation/test_orca_frequencies.py`). Two tests:
  - Water at `tests/conftest.py:WATER_XYZ`, RHF/STO-3G, CIS (TDA-HF) with 3 triplets: ORCA input `! RHF STO-3G NoRI TightSCF` plus `%tddft nroots 3 tda true triplets true end`. Parse ORCA's triplet excitation energies (read the output once by running ORCA by hand and write the parser against what it prints; keep the parser in the test file). Ours: `PySCFBackend` with `method = "tda"`, `functional = "hf"`, `singlets = 1`, `triplets = 3`, `scf_conv_tol = 1e-11`, `td_conv_tol = 1e-9`. **Assert: the SCF energies and every triplet excitation energy (Eh) within 1e-6.**
  - Ethylene (`tests/data/ethylene.xyz`) SA-CASSCF(2,2)/6-31G\* over 2 singlets + 1 triplet, equal weights: ORCA `%casscf nel 2 norb 2 mult 1,3 nroots 2,1` with block weights that give 1/3 per state (state the weights in the ORCA input explicitly; read ORCA's printed weights to confirm), `! NoRI TightSCF`, the same basis. Ours: `casscf` with `singlets = 2`, `triplets = 1`, `conv_tol = 1e-10`, `conv_tol_grad = 1e-6`. **Assert: every state energy within 1e-6 Eh.** If ORCA converges to another active space (energies off by more than 1e-3 Eh, a different solution), report it to the controller as a NEEDS_CONTEXT with both outputs instead of changing the ORCA guess on your own.
- [ ] **Step 2: Finite-difference gradients** (`tests/electronic/test_gradients_fd.py`, markers `slow`, `pyscf`; the file's `_fd` helper with step `H_TRIPLET = 1e-3` bohr; a fresh backend per displacement):
  - water TDA-HF/STO-3G `singlets = 1`, `triplets = 2`: `T1` and `T2` at components (0, 2), (1, 1), (2, 2) (atom, axis);
  - water TDA-B3LYP/STO-3G, `grid_level = 5`, `triplets = 1`: `T1` at the same components;
  - ethylene SA-CASSCF(2,2)/6-31G\* `singlets = 2`, `triplets = 1`: `S1` and `T1` at (0, 0), (2, 1).
  **Assert: max |analytic - FD| <= 1e-5 Eh/bohr over each test's components** (PR3). Print the measured maxima.
- [ ] **Step 3: Triplet-triplet NACs** (`tests/electronic/test_couplings_fd.py`, markers `slow`, `pyscf`; PR4): water CAS(4,4)/STO-3G, `singlets = 2`, `triplets = 2`, the existing FD-overlap method of that file applied to the pair `(T1, T2)`. **Assert: within 1e-4 1/bohr; the `(S1, T1)` NAC exactly zero.** If the T-T assertion fails, stop and report the numbers (PR8).
- [ ] **Step 4: SHARC** (`tests/validation/test_sharc_triplets.py`, markers `sharc`, `pyscf`): if Task 1 found triplets supported, extend `run_sharc_pyscf.run_geometry` with a `states: str = None` argument (the `states` line; the default keeps today's line) and compare the ethylene 2 + 1 triplet energy (each Ms component equal) and gradient: **assert within 1e-8 Eh and 1e-7 Eh/bohr**. If not supported: one test `pytest.skip(<Task 1's recorded reason>)` so that the decision is visible in the test list.
- [ ] **Step 5: Run** `W 'pytest tests/validation/test_orca_triplets.py tests/validation/test_sharc_triplets.py -q -W error'` with `ORCA_EXE`, `SHARC_HOME`, `SHARC_PYTHON` set, and `W 'pytest tests/electronic/test_gradients_fd.py tests/electronic/test_couplings_fd.py -q -W error -m slow -k "triplet or T1 or T2"'` (adjust the `-k` to the test names you chose); record every measured difference in the report. Lint, commit (`test: validate the triplet states against ORCA, SHARC and finite differences`).

---

### Task 7b (controller): the fast tier and the slow electronic tests

- [ ] `W 'pytest -m "not slow and not pyscf" -W error -q'` and `W 'pytest tests/electronic tests/stages/test_excitations.py tests/test_cli_single_point.py -W error -q'` (all markers) on the branch head; record counts and times in the ledger. Failures go back to the task that owns the code.

---

### Task 8: documents

**Files:**
- Create: `docs/superpowers/reports/2026-11-26-triplets-backend.md` (the validation report: every number marked asserted or report, the Task 1 SHARC finding, the plan rulings, what was decided after numbers were seen, the honest summary)
- Modify: `USER_MANUAL.md` (the "Read this first" summary; 4.2 `[electronic] triplets`: what it does now, for which methods, the CASSCF count, the state-average consequence for S0 and the `--rerun-from optimize` it implies when a CASSCF project turns triplets on; 5.3 `single-point` (the `S^2` column, triplet rows); 6 the excitations outputs (`triplet_energies`, `triplet_labels`, `triplet_states`); 8.6 the triplet methods (mixed SA-CASSCF, the triplet response solve, the Ms = 0 expansion of PR1, overlaps block-diagonal, NACs only within a multiplicity, dipoles zero), with the measured validation numbers; the limitations (no SOC, dynamics refuses triplets, TDA triplet NACs none); section 11 "Next": sub-projects 2 and 3), `CLAUDE.md` (a new sub-project block in the style of the others, the `electronic/` description, Current status, Known limitations 4, Remaining work (sub-projects 2 and 3 named), the Environment test counts, the deferred minors from the ledger), `CHANGELOG.md` `Unreleased` (Added: triplet states; no upgrade action for `triplets = 0` projects), the spec's erratum block (PR1-PR8 and anything else that differed).
- [ ] Check the site test (`W 'pytest tests/site -q -W error'`; do not rename USER_MANUAL headings) and commit (`docs: record the triplet states of the PySCF backend`).

After Task 8: the final whole-branch review, its fix wave, and superpowers:finishing-a-development-branch (the owner decides the merge).
