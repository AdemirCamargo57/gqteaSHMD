# The CASSCF Start Rule and the P9a Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every fresh SA-CASSCF solve starts from a reference solution (the lowest SA energy among a fixed guess set at the stage's reference geometry, projected onto the current geometry); the dormant F4 rescue rungs are removed; P9a is closed in the documents.

**Architecture:** `pyscf_casscf.py` gains a pure guess-set function, a reference search (`find_reference`) and a fourth ladder rung "reference orbitals" fed by a callable; `PySCFBackend` takes `reference_coords_bohr`, computes the reference lazily once per backend and records it in the result metadata; every CASSCF stage passes its reference geometry through `stage_backend`; a `start_rule` marker in the electronic fingerprint makes existing CASSCF projects stale.

**Tech Stack:** Python 3.12, NumPy, PySCF 2.14 (lazily imported), pytest, ruff, mypy; WSL Oracle Linux 9.5.

**Spec:** `docs/superpowers/specs/2026-11-13-casscf-start-rule-design.md` (owner-approved 2026-11-13). Read it before your task; the plan's rulings below are binding where they differ.

## Global Constraints

- **Run everything in WSL:** `wsl.exe -d OracleLinux_9_5 -- bash -lc 'cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && conda activate gqteashmd-dev && <command>'`. Git from WSL only (pre-commit hooks: ruff 0.17.4). **Never `--no-verify`.** Edit files with the normal file tools, LF line endings.
- **One commit per task**, conventional subject (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`), ending with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Clean room:** SHARC is never read, copied or imported; it may only be executed from `tools/sharc_compare/`.
- **PySCF is imported lazily, only inside functions of `electronic/pyscf_*.py`.** `import gqteashmd.cli` must not import pyscf, geometric, h5py or scipy (existing test). So `CASSCF_START_RULE` lives in `electronic/factory.py`, which must not import `pyscf_casscf` or `overlaps/` at module level.
- `tools/` is never imported by `src/gqteashmd`.
- Lint `ruff format . && ruff check .` (line length 100), `mypy` clean on `src/`.
- Spec values (verbatim): `CASSCF_START_RULE = "reference-1"`; `REFERENCE_TIE_EH = 1e-8`; `REFERENCE_OVERLAP_WARNING = 0.9`; the ladder names `"previous orbitals"`, `"reference orbitals"`, `"SCF orbitals"`, `"SCF orbitals + spin penalty"`; the metadata keys `casscf_reference` and `reference_active_min_sv`; the fingerprint key `start_rule`.
- Keep project runs and large outputs out of OneDrive and `/mnt/c` (use `~/start_rule/...` in WSL).

## Plan rulings (binding where they differ from the spec)

- **PR1 — the reference geometry of `excitations` and `dynamics` is `samples.h5`'s `equilibrium_coordinates`** (`read_samples_h5(...)["equilibrium_bohr"]`), not `working_geometry(ctx)`. Why: `ctx.upstream` holds direct dependencies only; `excitations` and `dynamics` do not depend on `optimize`, so `working_geometry` would return the *prepared* geometry there. `equilibrium_coordinates` is the geometry the `frequencies` stage used (the working geometry), which is the spec's intent. `optimize` passes the prepared geometry, `frequencies` its `working_geometry(ctx)` coordinates.
- **PR2 — `stage_backend` takes `reference_coords_bohr` as a required keyword** (no default), so every stage states its reference. The dynamics `BackendFactory` signature `(ctx, molecule, task_dir)` is unchanged: `_pyscf_backend` reads `samples.h5` itself.
- **PR3 — a fourth metadata value:** `casscf_reference = "not computed"` when the backend has a reference geometry but has not needed the reference yet (rung 1 succeeded on every call so far). The spec listed `"none"`, `"unavailable"` and the object.
- **PR4 — the flaky live-SHARC water test (CLAUDE.md limitation 3a) is not measured.** It builds `PySCFBackend` without a reference, so the rule does not apply to it; re-running it would measure nothing about the rule. The documents say so; limitation 3a stays open. (Spec section 8, third measurement, is dropped.)
- **PR5 — `pick_lowest` is the tie rule:** the index of the first SA energy `<= min + REFERENCE_TIE_EH`. This is the spec's "guesses within 1e-8 Eh of the lowest are ties, won by the first in the order".
- **PR6 — the reference `Mole` is rebuilt without a log stream** (`_build_mol(coords, None)`) before it is cached, as the previous-geometry `Mole` already is, so a cached object never holds a closed log file.

---

### Task 1: Remove the dormant rescue rungs

**Files:**
- Modify: `src/gqteashmd/electronic/pyscf_casscf.py` (lines 23-61, 93-134), `src/gqteashmd/electronic/pyscf_backend.py:95-106,152-159`, `src/gqteashmd/electronic/factory.py:34-62`, `src/gqteashmd/stages/geometry.py:33-55`
- Delete: `tools/diagnosis/convergence_probe.py`, `tests/diagnosis/test_convergence_probe.py`
- Modify tests: `tests/electronic/test_pyscf_casscf.py` (delete the five tests from `test_extra_rungs_run_after_the_ladder_in_order` to `test_loose_rung_never_tightens_the_user_tolerances`, lines 356-429; add one)

**Interfaces:**
- Produces: `pyscf_casscf.Rung(name: str, start: str, spin_shift: float, macro_factor: int)` with `start` in `"previous"`, `"scf"` (Task 2 adds `"reference"`); `run_casscf(mol, mf, settings, states, previous)` without `extra_rungs`; `PySCFBackend`, `make_backend` and `stage_backend` without `casscf_extra_rungs`.

- [ ] **Step 1: Write the failing test** (append to `tests/electronic/test_pyscf_casscf.py`)

```python
def test_the_rescue_rungs_are_gone(ethylene_symbols, make_settings):
    assert not hasattr(pyscf_casscf, "EXTRA_RUNGS")
    assert not hasattr(pyscf_casscf, "check_extra_rungs")
    assert [r.name for r in pyscf_casscf.LADDER] == [
        "previous orbitals",
        "SCF orbitals",
        "SCF orbitals + spin penalty",
    ]
    with pytest.raises(TypeError):
        _backend(ethylene_symbols, make_settings(basis="sto-3g"), casscf_extra_rungs=())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/electronic/test_pyscf_casscf.py::test_the_rescue_rungs_are_gone -q -W error`
Expected: FAIL (`EXTRA_RUNGS` exists).

- [ ] **Step 3: Remove the code**

In `pyscf_casscf.py`: delete the `EXTRA_RUNGS`/`EXTRA_RUNG_NAMES` block with its comment, `check_extra_rungs`, the `extra_rungs` parameter and its two lines in `run_casscf` (`enabled = ...`, `rungs = ...`; iterate `for rung in LADDER:`), and the `conv_tol`/`conv_tol_grad` fields of `Rung`. The `Rung` docstring and the loop become:

```python
@dataclass(frozen=True)
class Rung:
    """One attempt of the SA-CASSCF ladder.

    start: "previous" (the previous geometry's orbitals, skipped without them) or "scf" (the SCF
    orbitals).
    """

    name: str
    start: str
    spin_shift: float
    macro_factor: int
```

```python
    for rung in LADDER:
        name = rung.name
        if rung.start == "previous" and previous is None:
            continue
        mc = mcscf.CASSCF(mf, cas.ncas, cas.nelecas)
        mc.fcisolver = fci.solver(mol, singlet=True)  # singlet-only CI (excludes triplets)
        mc = mc.state_average_([1.0 / n_states] * n_states)
        if rung.spin_shift:
            mc.fix_spin_(ss=0, shift=rung.spin_shift)  # also penalise quintets and higher spins
        mc.conv_tol = cas.conv_tol
        mc.conv_tol_grad = cas.conv_tol_grad
        mc.max_cycle_macro = mc.max_cycle_macro * rung.macro_factor
        mo0 = mf.mo_coeff
        if rung.start == "previous":
            assert previous is not None
            previous_mo, previous_mol = previous
```

(the rest of the loop body is unchanged). Update the `run_casscf` docstring first line to `"""Equal-weight SA-CASSCF over `states` (singlets): the rungs of LADDER.`. Remove the now-unused `Iterable` import only if nothing else uses it (`casscf_gradients` does: keep it).

In `pyscf_backend.py`: delete the `casscf_extra_rungs` parameter, the line `self._casscf_extra_rungs = ...` and the `extra_rungs=self._casscf_extra_rungs,` argument. In `factory.py` and `stages/geometry.py`: delete the `casscf_extra_rungs` parameter and its pass-through.

Delete `tools/diagnosis/convergence_probe.py` and `tests/diagnosis/test_convergence_probe.py` (`git rm`). Delete the five rescue-rung tests listed under **Files**. Then `grep -rn "extra_rung\|EXTRA_RUNG\|convergence_probe" src tools tests` must print nothing.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/electronic/test_pyscf_casscf.py tests/diagnosis tests/stages/test_optimize.py tests/stages/test_frequencies.py -q -W error`
Expected: all pass. Then `ruff format . && ruff check . && mypy`.

- [ ] **Step 5: Commit**

```bash
git add -A src/gqteashmd/electronic src/gqteashmd/stages/geometry.py tests/electronic/test_pyscf_casscf.py tools/diagnosis tests/diagnosis
git commit -m "refactor: remove the dormant CASSCF rescue rungs" -m "Owner decision 2026-11-13 (P9a closure, decision 3)." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The guess set, the reference search and the reference rung

**Files:**
- Modify: `src/gqteashmd/electronic/pyscf_casscf.py`
- Create: `tests/electronic/test_casscf_guesses.py` (no PySCF)
- Modify tests: `tests/electronic/test_pyscf_casscf.py` (append)

**Interfaces:**
- Consumes: Task 1's `Rung`, `LADDER`, `run_casscf(mol, mf, settings, states, previous)`.
- Produces (Task 3 relies on these exact names):
  - `REFERENCE_RUNG = "reference orbitals"`, `REFERENCE_TIE_EH = 1e-8`, `REFERENCE_OVERLAP_WARNING = 0.9`;
  - `Orbitals = tuple[np.ndarray, Any]` (MO coefficients, Mole);
  - `guess_active_sets(n_occupied: int, n_mo: int, ncas: int, nelecas: int) -> list[tuple[int, ...]]`;
  - `pick_lowest(sa_energies: Sequence[float]) -> int`;
  - `Reference` (frozen dataclass, `eq=False`): `mo_coeff: np.ndarray`, `mol: Any`, `guess: tuple[int, ...]`, `sa_energy: float`, `n_guesses: int`, `n_counted: int`, method `meta() -> dict[str, Any]`;
  - `find_reference(mol, mf, settings, states) -> tuple[Reference | None, list[str]]`;
  - `active_min_singular_value(mol, mo_start, mo_final, ncore, ncas) -> float`;
  - `run_casscf(mol, mf, settings, states, previous, *, reference: Callable[[], Orbitals | None] | None = None) -> CasscfOutcome`;
  - `CasscfOutcome` gains `reference_active_min_sv: float | None = None` (set only when the reference rung produced the solution).

- [ ] **Step 1: Write the failing pure tests** (`tests/electronic/test_casscf_guesses.py`)

```python
import pytest

from gqteashmd.electronic.pyscf_casscf import guess_active_sets, pick_lowest


def test_cas22_has_four_guesses_with_the_default_first():
    # ethylene: 8 doubly occupied orbitals (0-based 0..7), plenty of virtuals
    assert guess_active_sets(8, 38, 2, 2) == [(7, 8), (6, 8), (6, 9), (7, 9)]


def test_cas44_has_nine_guesses():
    guesses = guess_active_sets(8, 38, 4, 4)
    assert len(guesses) == 9
    assert guesses[0] == (6, 7, 8, 9)
    assert guesses[1:] == sorted(guesses[1:])
    assert len(set(guesses)) == 9


def test_cas23_has_two_occupied_and_three_virtual_choices():
    guesses = guess_active_sets(8, 38, 3, 2)  # 1 occupied active, 2 virtual active
    assert guesses[0] == (7, 8, 9)
    assert len(guesses) == 2 * 3


def test_no_virtual_active_orbital():
    # nelecas = 2 ncas: only the occupied window matters
    assert guess_active_sets(8, 38, 1, 2) == [(7,), (6,)]


def test_the_occupied_window_is_clipped_at_the_first_orbital():
    # core = 0: the window would start at -1
    assert guess_active_sets(1, 10, 2, 2) == [(0, 1), (0, 2)]


def test_the_virtual_window_is_clipped_at_the_last_orbital():
    assert guess_active_sets(8, 9, 2, 2) == [(7, 8), (6, 8)]


@pytest.mark.parametrize(
    ("energies", "expected"),
    [
        ([1.0, 0.5, 0.5 + 5e-9], 1),
        ([0.5 + 5e-9, 0.5], 0),
        ([0.3, 0.1], 1),
        ([0.1, 0.1 + 2e-8], 0),
        ([0.1 + 2e-8, 0.1], 1),
    ],
)
def test_pick_lowest_takes_the_first_within_the_tie_tolerance(energies, expected):
    assert pick_lowest(energies) == expected


def test_pick_lowest_needs_candidates():
    with pytest.raises(ValueError, match="no candidate"):
        pick_lowest([])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/electronic/test_casscf_guesses.py -q -W error`
Expected: FAIL (ImportError: `guess_active_sets`).

- [ ] **Step 3: Implement the pure functions** (in `pyscf_casscf.py`, after `SPIN_TOLERANCE`; add `import itertools` and `from collections.abc import Callable, Iterable, Sequence`)

```python
REFERENCE_RUNG = "reference orbitals"
REFERENCE_TIE_EH = 1e-8  # SA energies within this of the lowest are ties: the first guess wins
REFERENCE_OVERLAP_WARNING = 0.9  # warn below this active-orbital singular value (no effect)

Orbitals = tuple[np.ndarray, Any]  # (MO coefficients, the Mole they belong to)


def guess_active_sets(n_occupied: int, n_mo: int, ncas: int, nelecas: int) -> list[tuple[int, ...]]:
    """0-based active-orbital choices around the Fermi level, PySCF's default first.

    The occupied window is the nelecas/2 + 1 highest doubly occupied orbitals, the virtual
    window the ncas - nelecas/2 + 1 lowest virtual ones, both clipped to the orbitals that
    exist. The other choices follow in lexicographic order of their sorted indices.
    """
    n_occ_act = nelecas // 2
    n_virt_act = ncas - n_occ_act
    occ_window = range(max(0, n_occupied - n_occ_act - 1), n_occupied)
    virt_window = range(n_occupied, min(n_mo, n_occupied + n_virt_act + 1))
    default = tuple(range(n_occupied - n_occ_act, n_occupied + n_virt_act))
    choices = {
        tuple(sorted((*o, *v)))
        for o in itertools.combinations(occ_window, n_occ_act)
        for v in itertools.combinations(virt_window, n_virt_act)
    }
    choices.discard(default)
    return [default, *sorted(choices)]


def pick_lowest(sa_energies: Sequence[float]) -> int:
    """Index of the first SA energy within REFERENCE_TIE_EH of the lowest."""
    if not sa_energies:
        raise ValueError("no candidate SA energies")
    lowest = min(sa_energies)
    return next(i for i, e in enumerate(sa_energies) if e <= lowest + REFERENCE_TIE_EH)
```

- [ ] **Step 4: Run the pure tests**

Run: `pytest tests/electronic/test_casscf_guesses.py -q -W error`
Expected: 12 passed.

- [ ] **Step 5: Write the failing PySCF tests** (append to `tests/electronic/test_pyscf_casscf.py`; the module is already `pytest.mark.pyscf`)

```python
def _displaced(coords, scale=0.02):
    """A fixed non-uniform distortion (not a translation) of the geometry, in bohr."""
    c = np.asarray(coords, dtype=float)
    return c + scale * np.linspace(-1.0, 1.0, c.size).reshape(c.shape)


def _scf(symbols, coords, settings):
    backend = _backend(symbols, settings)
    mol = backend._build_mol(np.asarray(coords), None)
    mf, _ = backend._run_scf(mol, None)
    return backend, mol, mf


def test_find_reference_is_the_lowest_sa_guess(ethylene_symbols, ethylene_bohr, make_settings):
    from pyscf import fci, mcscf

    settings = make_settings(singlets=3)  # SA3-CAS(2,2)/6-31G*, the P9a acceptance setup
    backend, mol, mf = _scf(ethylene_symbols, ethylene_bohr, settings)
    reference, problems = pyscf_casscf.find_reference(mol, mf, settings, backend.states)
    assert reference is not None and reference.n_guesses == 4
    independent = {}
    for guess in pyscf_casscf.guess_active_sets(mol.nelectron // 2, mf.mo_coeff.shape[1], 2, 2):
        mc = mcscf.CASSCF(mf, 2, 2)
        mc.fcisolver = fci.solver(mol, singlet=True)
        mc = mc.state_average_([1 / 3] * 3)
        mc.kernel(mo_coeff=mcscf.sort_mo(mc, mf.mo_coeff, [i + 1 for i in guess]))
        if mc.converged:  # not asserted: the conftest geometry is not the optimized one
            independent[guess] = float(np.mean(mc.e_states))
    assert reference.n_counted == len(independent) == 4 - len(problems)
    best = min(independent, key=independent.get)
    assert reference.guess == best
    assert abs(reference.sa_energy - independent[best]) < 1e-8
    assert reference.meta() == {
        "guess": list(best),
        "sa_energy_Eh": reference.sa_energy,
        "n_guesses": 4,
        "n_counted": len(independent),
    }


def test_find_reference_skips_unconverged_guesses(
    ethylene_symbols, ethylene_bohr, make_settings, monkeypatch
):
    settings = make_settings(basis="sto-3g")
    backend, mol, mf = _scf(ethylene_symbols, ethylene_bohr, settings)
    calls = []

    def reject_first(mc):
        calls.append(1)
        return len(calls) > 1 and bool(mc.converged)

    monkeypatch.setattr(pyscf_casscf, "_converged", reject_first)
    reference, problems = pyscf_casscf.find_reference(mol, mf, settings, backend.states)
    assert reference is not None
    assert problems[0] == "guess [7, 8]: not converged"  # the default guess is tried first
    assert reference.n_counted == 4 - len(problems)


def test_find_reference_without_a_converged_guess(
    ethylene_symbols, ethylene_bohr, make_settings, monkeypatch
):
    settings = make_settings(basis="sto-3g")
    backend, mol, mf = _scf(ethylene_symbols, ethylene_bohr, settings)
    monkeypatch.setattr(pyscf_casscf, "_converged", lambda mc: False)
    reference, problems = pyscf_casscf.find_reference(mol, mf, settings, backend.states)
    assert reference is None and len(problems) == 4


def test_the_reference_rung_comes_after_previous_and_before_scf(
    ethylene_symbols, ethylene_bohr, make_settings
):
    assert [r.name for r in pyscf_casscf.LADDER] == [
        "previous orbitals",
        "reference orbitals",
        "SCF orbitals",
        "SCF orbitals + spin penalty",
    ]
    settings = make_settings(basis="sto-3g")
    backend, mol, mf = _scf(ethylene_symbols, ethylene_bohr, settings)
    reference, _ = pyscf_casscf.find_reference(mol, mf, settings, backend.states)
    assert reference is not None
    asked = []

    def provider():
        asked.append(1)
        return reference.mo_coeff, reference.mol

    _, mol2, mf2 = _scf(ethylene_symbols, _displaced(ethylene_bohr), settings)
    outcome = pyscf_casscf.run_casscf(mol2, mf2, settings, backend.states, None, reference=provider)
    assert outcome.rung == "reference orbitals" and asked == [1]
    assert outcome.reference_active_min_sv is not None
    assert outcome.reference_active_min_sv >= pyscf_casscf.REFERENCE_OVERLAP_WARNING
    plain = pyscf_casscf.run_casscf(mol2, mf2, settings, backend.states, None)
    assert plain.rung == "SCF orbitals" and plain.reference_active_min_sv is None


def test_an_unavailable_reference_falls_through_to_scf(
    ethylene_symbols, ethylene_bohr, make_settings
):
    settings = make_settings(basis="sto-3g")
    backend, mol, mf = _scf(ethylene_symbols, ethylene_bohr, settings)
    outcome = pyscf_casscf.run_casscf(
        mol, mf, settings, backend.states, None, reference=lambda: None
    )
    assert outcome.rung == "SCF orbitals"
```

- [ ] **Step 6: Run them to verify they fail**

Run: `pytest tests/electronic/test_pyscf_casscf.py -q -W error -k "reference"`
Expected: FAIL (`find_reference` missing).

- [ ] **Step 7: Implement the reference search and the rung** (in `pyscf_casscf.py`)

Replace `LADDER` and add the helpers; `run_casscf` uses `_solver` and `_spins` so the ladder and the search share one solver set-up:

```python
LADDER: tuple[Rung, ...] = (
    Rung("previous orbitals", "previous", 0.0, 1),
    Rung(REFERENCE_RUNG, "reference", 0.0, 1),
    Rung("SCF orbitals", "scf", 0.0, 1),
    Rung("SCF orbitals + spin penalty", "scf", 0.5, 3),
)


@dataclass(frozen=True, eq=False)
class Reference:
    """The reference solution: the lowest-SA guess at the reference geometry."""

    mo_coeff: np.ndarray
    mol: Any
    guess: tuple[int, ...]
    sa_energy: float
    n_guesses: int
    n_counted: int

    def meta(self) -> dict[str, Any]:
        return {
            "guess": list(self.guess),
            "sa_energy_Eh": self.sa_energy,
            "n_guesses": self.n_guesses,
            "n_counted": self.n_counted,
        }


@dataclass(eq=False)
class CasscfOutcome:
    mc: Any
    rung: str
    energies: dict[StateLabel, float]
    spin_squares: dict[StateLabel, float]
    reference_active_min_sv: float | None = None


def _solver(mol: Any, mf: Any, settings: ElectronicSection, n_states: int, rung: Rung) -> Any:
    from pyscf import fci, mcscf

    cas = settings.casscf
    assert cas is not None
    mc = mcscf.CASSCF(mf, cas.ncas, cas.nelecas)
    mc.fcisolver = fci.solver(mol, singlet=True)  # singlet-only CI (excludes triplets)
    mc = mc.state_average_([1.0 / n_states] * n_states)
    if rung.spin_shift:
        mc.fix_spin_(ss=0, shift=rung.spin_shift)  # also penalise quintets and higher spins
    mc.conv_tol = cas.conv_tol
    mc.conv_tol_grad = cas.conv_tol_grad
    mc.max_cycle_macro = mc.max_cycle_macro * rung.macro_factor
    return mc


def _spins(mc: Any, states: tuple[StateLabel, ...]) -> tuple[dict[StateLabel, float], list[str]]:
    from pyscf import fci

    spins = {
        label: float(fci.spin_op.spin_square0(ci, mc.ncas, mc.nelecas)[0])
        for label, ci in zip(states, mc.ci, strict=True)
    }
    wrong = [f"{label} <S^2> = {s:.4f}" for label, s in spins.items() if abs(s) > SPIN_TOLERANCE]
    return spins, wrong


def active_min_singular_value(
    mol: Any, mo_start: np.ndarray, mo_final: np.ndarray, ncore: int, ncas: int
) -> float:
    """Smallest singular value of C_start^T S C_final over the active orbitals."""
    s_ao = mol.intor("int1e_ovlp")
    a = np.asarray(mo_start)[:, ncore : ncore + ncas]
    b = np.asarray(mo_final)[:, ncore : ncore + ncas]
    return float(np.linalg.svd(a.T @ s_ao @ b, compute_uv=False).min())


def find_reference(
    mol: Any, mf: Any, settings: ElectronicSection, states: tuple[StateLabel, ...]
) -> tuple[Reference | None, list[str]]:
    """Solve every guess of guess_active_sets fresh; the lowest SA energy is the reference.

    A guess counts if it converges and every root is a singlet. Returns (None, problems)
    when none counts. An exception (not a non-convergence) propagates, wrapped.
    """
    from pyscf import mcscf

    cas = settings.casscf
    assert cas is not None
    check_active_space(mol, settings)
    guesses = guess_active_sets(
        mol.nelectron // 2, int(np.asarray(mf.mo_coeff).shape[1]), cas.ncas, cas.nelecas
    )
    plain = Rung("reference guess", "scf", 0.0, 1)
    problems: list[str] = []
    counted: list[tuple[tuple[int, ...], float, np.ndarray]] = []
    for guess in guesses:
        mc = _solver(mol, mf, settings, len(states), plain)
        try:
            mo0 = mcscf.sort_mo(mc, mf.mo_coeff, [i + 1 for i in guess])  # 1-based
            mc.kernel(mo_coeff=mo0)
        except Exception as exc:
            raise wrap_failure(f"CASSCF (reference guess {list(guess)})", exc) from exc
        if not _converged(mc):
            problems.append(f"guess {list(guess)}: not converged")
            continue
        _, wrong = _spins(mc, states)
        if wrong:
            problems.append(f"guess {list(guess)}: not all roots are singlets ({', '.join(wrong)})")
            continue
        counted.append((guess, float(np.mean(mc.e_states)), np.asarray(mc.mo_coeff)))
    if not counted:
        return None, problems
    best = pick_lowest([sa for _, sa, _ in counted])
    guess, sa, mo = counted[best]
    return Reference(mo, mol, guess, sa, len(guesses), len(counted)), problems
```

`run_casscf` becomes (signature, docstring and loop; the final `raise` is unchanged):

```python
def run_casscf(
    mol: Any,
    mf: Any,
    settings: ElectronicSection,
    states: tuple[StateLabel, ...],
    previous: Orbitals | None,
    *,
    reference: Callable[[], Orbitals | None] | None = None,
) -> CasscfOutcome:
    """Equal-weight SA-CASSCF over `states` (singlets): the rungs of LADDER.

    `previous` = (MO coefficients, Mole) of the previous geometry; `reference` is called (at
    most once) when the reference rung is reached and returns the reference orbitals or None.
    Both are projected onto this geometry.
    """
    from pyscf import mcscf

    cas = settings.casscf
    assert cas is not None
    check_active_space(mol, settings)
    problems: list[str] = []
    for rung in LADDER:
        name = rung.name
        start: Orbitals | None = None
        if rung.start == "previous":
            if previous is None:
                continue
            start = previous
        elif rung.start == "reference":
            start = None if reference is None else reference()
            if start is None:
                continue
        mc = _solver(mol, mf, settings, len(states), rung)
        mo0 = mf.mo_coeff
        if start is not None:
            start_mo, start_mol = start
            what = "previous" if rung.start == "previous" else "reference"
            try:
                # orbitals of another geometry are not orthonormal here: project them first
                mo0 = mcscf.project_init_guess(mc, start_mo, start_mol)
            except Exception as exc:
                raise wrap_failure(f"Projecting the {what} orbitals", exc) from exc
        try:
            mc.kernel(mo_coeff=mo0)
        except Exception as exc:
            # an exception (unlike non-convergence) signals a broken setup: abort the ladder
            raise wrap_failure(f"CASSCF ({name})", exc) from exc
        if not _converged(mc):
            problems.append(f"{name}: not converged")
            continue
        spins, wrong = _spins(mc, states)
        if wrong:
            problems.append(f"{name}: not all roots are singlets ({', '.join(wrong)})")
            continue
        energies = {label: float(e) for label, e in zip(states, mc.e_states, strict=True)}
        min_sv = None
        if rung.start == "reference":
            min_sv = active_min_singular_value(mol, mo0, mc.mo_coeff, mc.ncore, mc.ncas)
        return CasscfOutcome(mc, name, energies, spins, min_sv)
```

Update the `Rung` docstring: `start: "previous" (the previous geometry's orbitals), "reference" (the backend's reference orbitals), both skipped when absent, or "scf" (the SCF orbitals).` Keep the existing text of the `"Projecting the previous orbitals"` error (now produced by `what = "previous"`). In `pyscf_backend.py` the call `pyscf_casscf.run_casscf(mol, mf, self._settings, self._states, prev_orbitals)` is unchanged by this task.

- [ ] **Step 8: Run the tests**

Run: `pytest tests/electronic/test_casscf_guesses.py tests/electronic/test_pyscf_casscf.py -q -W error`
Expected: all pass (the existing ladder tests are unchanged: without a reference the reference rung is skipped). Then `ruff format . && ruff check . && mypy`.

- [ ] **Step 9: Commit**

```bash
git add src/gqteashmd/electronic/pyscf_casscf.py tests/electronic/test_casscf_guesses.py tests/electronic/test_pyscf_casscf.py
git commit -m "feat: add the CASSCF reference search and the reference-orbitals rung" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The backend's reference (lazy, cached, metadata) and `single-point`

**Files:**
- Modify: `src/gqteashmd/electronic/pyscf_backend.py`, `src/gqteashmd/electronic/factory.py:34-62`, `src/gqteashmd/cli.py:290-310`
- Modify tests: `tests/electronic/test_pyscf_casscf.py` (append), `tests/test_cli.py` (one test)

**Interfaces:**
- Consumes: Task 2's `find_reference`, `Reference`, `run_casscf(..., reference=)`, `CasscfOutcome.reference_active_min_sv`, `REFERENCE_OVERLAP_WARNING`.
- Produces: `PySCFBackend(molecule, settings, *, log_file=None, threads=None, memory_mb=None, scratch_dir=None, reference_coords_bohr: ArrayLike | None = None)`; `make_backend(..., reference_coords_bohr: ArrayLike | None = None)`; result `meta["casscf_reference"]` in `"none" | "not computed" | "unavailable" | dict` and `meta["reference_active_min_sv"]` (CASSCF only).

- [ ] **Step 1: Write the failing tests** (append to `tests/electronic/test_pyscf_casscf.py`)

```python
import logging  # move to the top of the module with the other imports


def _ref_backend(symbols, coords, settings):
    return _backend(symbols, settings, reference_coords_bohr=np.asarray(coords))


def test_a_fresh_solve_starts_from_the_reference(ethylene_symbols, ethylene_bohr, make_settings):
    backend = _ref_backend(ethylene_symbols, ethylene_bohr, make_settings(basis="sto-3g"))
    result = backend.compute(Request(_displaced(ethylene_bohr)))
    assert result.meta["casscf"] == "reference orbitals"
    ref = result.meta["casscf_reference"]
    assert ref["n_guesses"] == 4 and ref["n_counted"] >= 1
    assert ref["guess"] in ([7, 8], [6, 8], [6, 9], [7, 9])
    assert result.meta["reference_active_min_sv"] >= 0.9


def test_without_a_reference_geometry_the_ladder_is_unchanged(
    ethylene_symbols, ethylene_bohr, make_settings
):
    result = _backend(ethylene_symbols, make_settings(basis="sto-3g")).compute(
        Request(ethylene_bohr)
    )
    assert result.meta["casscf"] == "SCF orbitals"
    assert result.meta["casscf_reference"] == "none"
    assert "reference_active_min_sv" not in result.meta


def test_the_reference_is_computed_once_and_survives_reset(
    ethylene_symbols, ethylene_bohr, make_settings, monkeypatch
):
    calls = []
    real = pyscf_casscf.find_reference

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(pyscf_casscf, "find_reference", counting)
    backend = _ref_backend(ethylene_symbols, ethylene_bohr, make_settings(basis="sto-3g"))
    first = backend.compute(Request(ethylene_bohr))
    backend.reset()
    second = backend.compute(Request(_displaced(ethylene_bohr, 0.01)))
    assert calls == [1]
    assert first.meta["casscf_reference"] == second.meta["casscf_reference"]
    assert second.meta["casscf"] == "reference orbitals"


def test_the_reference_is_not_computed_while_previous_orbitals_work(
    tmp_path, ethylene_symbols, ethylene_bohr, make_settings, monkeypatch
):
    settings = make_settings(basis="sto-3g")
    donor = _backend(ethylene_symbols, settings)
    donor.compute(Request(ethylene_bohr))
    donor.save_state(tmp_path / "state.npz")
    monkeypatch.setattr(
        pyscf_casscf, "find_reference", lambda *a, **k: pytest.fail("reference computed")
    )
    backend = _ref_backend(ethylene_symbols, ethylene_bohr, settings)
    backend.load_state(tmp_path / "state.npz")
    result = backend.compute(Request(_displaced(ethylene_bohr, 0.01)))
    assert result.meta["casscf"] == "previous orbitals"
    assert result.meta["casscf_reference"] == "not computed"


def test_a_failed_previous_rung_tries_the_reference_next(
    ethylene_symbols, ethylene_bohr, make_settings, monkeypatch
):
    backend = _ref_backend(ethylene_symbols, ethylene_bohr, make_settings(basis="sto-3g"))
    backend.compute(Request(ethylene_bohr))
    calls = []

    def fail_first(mc):
        calls.append(1)
        return len(calls) > 1 and bool(mc.converged)

    monkeypatch.setattr(pyscf_casscf, "_converged", fail_first)
    result = backend.compute(Request(_displaced(ethylene_bohr, 0.01)))
    assert result.meta["casscf"] == "reference orbitals" and len(calls) == 2


def test_no_reference_solution_falls_back_once_with_a_warning(
    ethylene_symbols, ethylene_bohr, make_settings, monkeypatch, caplog
):
    calls = []

    def none_found(*args, **kwargs):
        calls.append(1)
        return None, ["guess [7, 8]: not converged"]

    monkeypatch.setattr(pyscf_casscf, "find_reference", none_found)
    backend = _ref_backend(ethylene_symbols, ethylene_bohr, make_settings(basis="sto-3g"))
    with caplog.at_level(logging.WARNING, logger="gqteashmd.electronic.pyscf_backend"):
        first = backend.compute(Request(ethylene_bohr))
    backend.reset()
    second = backend.compute(Request(ethylene_bohr))
    assert calls == [1]
    assert first.meta["casscf"] == second.meta["casscf"] == "SCF orbitals"
    assert first.meta["casscf_reference"] == "unavailable"
    assert "No CASSCF reference solution at the reference geometry" in caplog.text
    assert "guess [7, 8]: not converged" in caplog.text


def test_a_failed_reference_scf_falls_back(
    ethylene_symbols, ethylene_bohr, make_settings, monkeypatch, caplog
):
    real = PySCFBackend._run_scf
    calls = []

    def second_fails(self, mol, dm):
        calls.append(1)
        if len(calls) == 2:  # 1 = the current geometry, 2 = the reference geometry
            raise ElectronicStructureError("SCF did not converge on any rung")
        return real(self, mol, dm)

    monkeypatch.setattr(PySCFBackend, "_run_scf", second_fails)
    backend = _ref_backend(ethylene_symbols, ethylene_bohr, make_settings(basis="sto-3g"))
    with caplog.at_level(logging.WARNING, logger="gqteashmd.electronic.pyscf_backend"):
        result = backend.compute(Request(ethylene_bohr))
    assert result.meta["casscf_reference"] == "unavailable"
    assert result.meta["casscf"] == "SCF orbitals"
    assert "the SCF failed" in caplog.text


def test_a_different_active_character_is_warned_and_recorded(
    ethylene_symbols, ethylene_bohr, make_settings, monkeypatch, caplog
):
    monkeypatch.setattr(pyscf_casscf, "active_min_singular_value", lambda *a: 0.5)
    backend = _ref_backend(ethylene_symbols, ethylene_bohr, make_settings(basis="sto-3g"))
    with caplog.at_level(logging.WARNING, logger="gqteashmd.electronic.pyscf_backend"):
        result = backend.compute(Request(ethylene_bohr))
    assert result.meta["reference_active_min_sv"] == 0.5
    assert "different active-orbital character" in caplog.text


def test_reference_coordinates_are_validated(ethylene_symbols, make_settings):
    with pytest.raises(ValueError, match="reference_coords_bohr"):
        _backend(
            ethylene_symbols, make_settings(basis="sto-3g"), reference_coords_bohr=np.zeros((2, 3))
        )
    bad = np.zeros((len(ethylene_symbols), 3))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="reference_coords_bohr"):
        _backend(ethylene_symbols, make_settings(basis="sto-3g"), reference_coords_bohr=bad)
```

Add to `tests/test_cli.py` (it has PySCF single-point tests; mark this one `@pytest.mark.pyscf` like them, and reuse their input-writing helper and argument pattern; read the existing `single-point` tests first):

```python
@pytest.mark.pyscf
def test_single_point_uses_its_own_geometry_as_the_reference(...):
    # a casscf single-point input (ethylene SA2-CAS(2,2)/STO-3G, as the existing casscf
    # single-point test builds it), run with --json OUT
    data = json.loads(out.read_text())
    assert data["meta"]["casscf"] == "reference orbitals"
    assert data["meta"]["casscf_reference"]["n_guesses"] == 4
```

(If the existing single-point tests have no CASSCF input, build the input with the `[electronic]` block `method = "casscf"`, `basis = "sto-3g"`, `singlets = 2`, `[electronic.casscf] ncas = 2, nelecas = 2` and the repository's `tests/data/ethylene.xyz`. The JSON key holding the meta is whatever `ElectronicResult.to_json()` writes; read it before asserting.)

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/electronic/test_pyscf_casscf.py -q -W error -k "reference"`
Expected: FAIL (`reference_coords_bohr` is not a parameter).

- [ ] **Step 3: Implement** (in `pyscf_backend.py`; add `import dataclasses`)

Constructor (after the `count_electrons` check):

```python
        self._reference_coords: np.ndarray | None = None
        if reference_coords_bohr is not None:
            ref = np.array(reference_coords_bohr, dtype=float)
            if ref.shape != (len(molecule.symbols), 3) or not np.all(np.isfinite(ref)):
                raise ValueError(
                    f"reference_coords_bohr must be a finite ({len(molecule.symbols)}, 3) "
                    f"array; got shape {ref.shape}"
                )
            self._reference_coords = ref
        self._reference: pyscf_casscf.Reference | None = None
        # "none" (no geometry), "not computed", "unavailable" (searched, none found), "ok"
        self._reference_status = "none" if self._reference_coords is None else "not computed"
```

In `compute`, the CASSCF branch:

```python
                outcome = pyscf_casscf.run_casscf(
                    mol,
                    mf,
                    self._settings,
                    self._states,
                    prev_orbitals,
                    reference=(
                        None
                        if self._reference_coords is None
                        else lambda: self._reference_orbitals(stdout)
                    ),
                )
                solver, energies, spins = outcome.mc, outcome.energies, outcome.spin_squares
                mo_coeff = np.asarray(solver.mo_coeff)
                expansion = pyscf_casscf.casscf_expansion(solver)
                rung_meta: dict[str, Any] = {
                    "casscf": outcome.rung,
                    "casscf_reference": self._reference_meta(),
                }
                if outcome.reference_active_min_sv is not None:
                    rung_meta["reference_active_min_sv"] = outcome.reference_active_min_sv
                    if outcome.reference_active_min_sv < pyscf_casscf.REFERENCE_OVERLAP_WARNING:
                        _LOG.warning(
                            "The CASSCF solution started from the reference orbitals has a "
                            "different active-orbital character (smallest singular value "
                            "%.3f < %.1f).",
                            outcome.reference_active_min_sv,
                            pyscf_casscf.REFERENCE_OVERLAP_WARNING,
                        )
```

New private methods (near `_run_scf`):

```python
    def _reference_meta(self) -> str | dict[str, Any]:
        if self._reference is not None:
            return self._reference.meta()
        return self._reference_status

    def _reference_orbitals(self, stdout: IO[str] | None) -> pyscf_casscf.Orbitals | None:
        """The reference orbitals, searched once per backend (kept across reset/load_state)."""
        if self._reference_status == "not computed":
            reference = self._find_reference(stdout)
            self._reference = reference
            self._reference_status = "unavailable" if reference is None else "ok"
        if self._reference is None:
            return None
        return self._reference.mo_coeff, self._reference.mol

    def _find_reference(self, stdout: IO[str] | None) -> pyscf_casscf.Reference | None:
        assert self._reference_coords is not None
        mol = self._build_mol(self._reference_coords, stdout)
        try:
            mf, _ = self._run_scf(mol, None)
        except ElectronicStructureError as exc:
            if exc.retryable:
                raise  # transient: the task is retried and searches again
            _LOG.warning(_NO_REFERENCE, f"the SCF failed: {exc}")
            return None
        reference, problems = pyscf_casscf.find_reference(mol, mf, self._settings, self._states)
        if reference is None:
            _LOG.warning(_NO_REFERENCE, "; ".join(problems))
            return None
        # a cached Mole must not hold the (closed) log stream: rebuild it without one
        return dataclasses.replace(reference, mol=self._build_mol(self._reference_coords, None))
```

Module constant (near `_LOG`):

```python
_NO_REFERENCE = (
    "No CASSCF reference solution at the reference geometry (%s): fresh solves start from "
    "the SCF orbitals."
)
```

A retryable error raised inside the search leaves the status `"not computed"`, so the retried task searches again. `reset()` and `load_state()` do not touch `_reference`/`_reference_status`; `save_state()` does not store them. For non-CASSCF methods nothing changes (the reference is ignored and no `casscf_reference` key is written).

`factory.make_backend`: add the keyword `reference_coords_bohr: ArrayLike | None = None` (import `ArrayLike` from `numpy.typing` under `TYPE_CHECKING` or directly; `numpy` is already a dependency) and pass it to `PySCFBackend`.

`cli.py` single-point: compute `coords` before `make_backend` and pass `reference_coords_bohr=coords`:

```python
    coords = np.array(geometry.coords_angstrom, dtype=float) * ANGSTROM_TO_BOHR
    backend = make_backend(
        cfg,
        molecule,
        log_file=args.log,
        threads=cfg.resources.threads_per_task,
        memory_mb=cfg.resources.memory_mb_per_task,
        reference_coords_bohr=coords,
    )
    result = backend.compute(Request(coords, transition_dipoles=True))
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/electronic -q -W error` and `pytest tests/test_cli.py -q -W error`
Expected: all pass. If an existing test asserted the exact `meta` key set of a CASSCF result, add `casscf_reference` to its expectation (and say so in the report). Then `ruff format . && ruff check . && mypy`.

- [ ] **Step 5: Commit**

```bash
git add src/gqteashmd/electronic/pyscf_backend.py src/gqteashmd/electronic/factory.py src/gqteashmd/cli.py tests/electronic/test_pyscf_casscf.py tests/test_cli.py
git commit -m "feat: start fresh CASSCF solves from a cached reference solution" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The stages pass their reference; the fingerprint marker

**Files:**
- Modify: `src/gqteashmd/stages/geometry.py:33-55`, `src/gqteashmd/stages/optimize.py:55`, `src/gqteashmd/stages/frequencies.py:74-75`, `src/gqteashmd/stages/excitations.py:81-95,117`, `src/gqteashmd/stages/dynamics.py:40-42,109-118`, `src/gqteashmd/electronic/factory.py:13-31`
- Modify tests: `tests/stages/test_fingerprints.py`, `tests/stages/test_excitations.py` (the `fake` fixture, one test), `tests/stages/test_optimize.py`, `tests/stages/test_frequencies.py`, `tests/stages/test_dynamics.py` (one test each), `tests/electronic/test_factory.py` (one `pyscf` test of `stage_backend` end to end)

**Interfaces:**
- Consumes: Task 3's `make_backend(..., reference_coords_bohr=)`.
- Produces: `factory.CASSCF_START_RULE = "reference-1"`; `factory.electronic_fingerprint(settings: ElectronicSection) -> dict[str, Any]`; `ground_state_fingerprint` with `start_rule` for CASSCF; `stage_backend(ctx, stage_name, molecule, task_dir, *, reference_coords_bohr: ArrayLike, settings: ElectronicSection | None = None)`.

- [ ] **Step 1: Write the failing tests**

`tests/stages/test_fingerprints.py` (append; add the imports `from gqteashmd.config.schema import ElectronicSection`, `from gqteashmd.electronic.factory import CASSCF_START_RULE, electronic_fingerprint`, `from gqteashmd.stages.excitations import ExcitationsStage`):

```python
def test_casscf_fingerprints_carry_the_start_rule(tmp_path, write_input):
    optimize, frequencies = _fingerprints(tmp_path, write_input, "a", CAS)
    assert optimize["electronic"]["start_rule"] == CASSCF_START_RULE == "reference-1"
    assert frequencies["electronic"]["start_rule"] == CASSCF_START_RULE


def test_tda_fingerprints_have_no_start_rule(tmp_path, write_input):
    optimize, frequencies = _fingerprints(tmp_path, write_input, "a", TDA)
    assert "start_rule" not in optimize["electronic"]
    assert "start_rule" not in frequencies["electronic"]


def test_electronic_fingerprint_is_the_section_plus_the_marker():
    cas = ElectronicSection.model_validate(
        {"method": "casscf", "basis": "sto-3g", "singlets": 2, "casscf": {"ncas": 2, "nelecas": 2}}
    )
    tda = ElectronicSection.model_validate(
        {"method": "tda", "basis": "sto-3g", "functional": "hf", "singlets": 2}
    )
    assert electronic_fingerprint(cas) == {
        **cas.model_dump(mode="json"),
        "start_rule": CASSCF_START_RULE,
    }
    assert electronic_fingerprint(tda) == tda.model_dump(mode="json")


def test_excitations_fingerprint_carries_the_start_rule_for_casscf_only(tmp_path, write_input):
    for name, electronic, expected in (("a", CAS, True), ("b", TDA, False)):
        directory = tmp_path / name
        directory.mkdir()
        cfg = load_config(write_input(directory, extra=electronic)).config
        ctx = StageContext(cfg, ProjectLayout(cfg.project.directory), directory, {})
        fp = ExcitationsStage().fingerprint_inputs(ctx)["electronic"]
        assert ("start_rule" in fp) is expected
        assert "overlap_norm_kept" not in fp
```

For `dynamics` add the same check in `tests/stages/test_dynamics.py` (its `_context` uses a TDA-or-CASSCF `ELECTRONIC` block; read it): `DynamicsStage(...).fingerprint_inputs(ctx)["electronic"]` equals `electronic_fingerprint(ctx.config.electronic)`.

`tests/stages/test_excitations.py`: change the `fake` fixture's factory to record the reference:

```python
    def factory(ctx, stage_name, molecule, task_dir, *, reference_coords_bohr, settings=None):
        backend = FakeBackend(settings, holder["fail_on"], holder["retryable_on"], holder["bad"])
        holder["backends"].append(backend)
        holder["settings"] = settings
        holder["reference"] = np.asarray(reference_coords_bohr)
        return backend
```

and add:

```python
def test_backend_reference_is_the_sampling_equilibrium(tmp_path, write_input, fake):
    ctx = _context(tmp_path, write_input)
    _run(ctx)
    np.testing.assert_array_equal(fake["reference"], _samples(ctx)["equilibrium_bohr"])
```

`tests/stages/test_optimize.py` and `tests/stages/test_frequencies.py`: a spy replacing the module's `stage_backend`, stopping the task with a sentinel after recording the reference:

```python
class _Stop(Exception):
    pass


def test_optimize_passes_the_prepared_geometry_as_reference(tmp_path, write_input, monkeypatch):
    from gqteashmd.stages import optimize as optimize_module
    from gqteashmd.stages.geometry import prepared_molecule

    cfg, octx = _contexts(tmp_path, write_input)
    seen = {}

    def spy(ctx, stage_name, molecule, task_dir, *, reference_coords_bohr, settings=None):
        seen["reference"] = np.asarray(reference_coords_bohr)
        raise _Stop

    monkeypatch.setattr(optimize_module, "stage_backend", spy)
    task_dir = octx.layout.task_dir(octx.stage_dir, "optimize")
    task_dir.mkdir(parents=True)
    with pytest.raises(_Stop):
        OptimizeStage().run_task(octx, TaskSpec("optimize"), task_dir)
    _, prepared = prepared_molecule(octx.upstream_dir("prepare"))
    np.testing.assert_array_equal(seen["reference"], prepared)
```

The frequencies test is the same with `frequencies_module`, a context built as `_run_stage` builds it (prepare only, so the working geometry is the prepared one; factor the context part of `_run_stage` into a `_context` helper if needed), a `TaskSpec("reference")` and the expectation `working_geometry(ctx)[1]`. The dynamics test calls `dynamics_module._pyscf_backend(ctx, molecule, task_dir)` with the spy on `dynamics_module.stage_backend` and `ctx = _context(tmp_path, write_input)`, and expects `read_samples_h5(ctx.upstream_dir("sampling") / "samples.h5")["equilibrium_bohr"]`.

`tests/electronic/test_factory.py` (module is `pyscf`): end to end through `stage_backend` for a CASSCF `frequencies`-like task: build a `StageContext` with a CASSCF ethylene STO-3G input (copy `tests/data/ethylene.xyz` next to the input; run the real `PrepareStage` as the stage tests do), call `stage_backend(ctx, "frequencies", molecule, task_dir, reference_coords_bohr=coords)` with `coords = working_geometry(ctx)[1]`, compute at `coords` with the first Cartesian component moved by +0.005 bohr (an FD displacement) and assert `meta["casscf"] == "reference orbitals"` and `isinstance(meta["casscf_reference"], dict)`.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/stages/test_fingerprints.py tests/stages/test_excitations.py tests/stages/test_optimize.py tests/stages/test_frequencies.py tests/stages/test_dynamics.py -q -W error -m "not slow and not pyscf"`
Expected: the new tests FAIL (no `start_rule`; `reference_coords_bohr` not passed).

- [ ] **Step 3: Implement**

`factory.py` (no new module-level imports besides `ArrayLike`):

```python
# The CASSCF start rule (fresh solves start from a reference solution). It is part of every
# CASSCF stage fingerprint: changing the rule makes existing CASSCF projects stale.
CASSCF_START_RULE = "reference-1"


def electronic_fingerprint(settings: ElectronicSection) -> dict[str, Any]:
    """The whole [electronic] section as JSON data, plus the start-rule marker for CASSCF."""
    dumped = settings.model_dump(mode="json")
    if settings.method == "casscf":
        dumped["start_rule"] = CASSCF_START_RULE
    return dumped


def ground_state_fingerprint(settings: ElectronicSection) -> dict[str, Any]:
    """The settings that determine the S0 surface, as JSON-compatible data (for fingerprints).

    Response-only keys (`td_conv_tol`, `overlap_norm_kept`, and `singlets` for the non-CASSCF
    methods) do not change S0 and are left out; CASSCF carries the start-rule marker."""
    dumped = electronic_fingerprint(ground_state_settings(settings))
    for key in ("td_conv_tol", "overlap_norm_kept"):
        dumped.pop(key, None)
    if settings.method != "casscf":
        dumped.pop("singlets", None)
    return dumped
```

`stages/geometry.py`:

```python
def stage_backend(
    ctx: StageContext,
    stage_name: str,
    molecule: MoleculeSpec,
    task_dir: Path,
    *,
    reference_coords_bohr: ArrayLike,
    settings: ElectronicSection | None = None,
) -> ElectronicBackend:
    """One backend for a stage task, using the stage's effective resources.

    `reference_coords_bohr` is the stage's reference geometry for the CASSCF start rule
    (ignored by the other methods)."""
    ...
        scratch_dir=task_dir / "scratch",
        reference_coords_bohr=reference_coords_bohr,
    )
```

Stages:
- `optimize.py:55`: `backend = stage_backend(ctx, self.name, molecule, task_dir, reference_coords_bohr=start)`.
- `frequencies.py:75`: `backend = stage_backend(ctx, self.name, molecule, task_dir, reference_coords_bohr=coords)`.
- `excitations.py:117`: `backend = stage_backend(ctx, self.name, molecule, task_dir, settings=ctx.config.electronic, reference_coords_bohr=np.asarray(data["equilibrium_bohr"], dtype=float))`; `fingerprint_inputs`: `electronic_dump = None if electronic is None else electronic_fingerprint(electronic)` (then the existing `pop("overlap_norm_kept", None)`).
- `dynamics.py`: `fingerprint_inputs` uses `"electronic": electronic_fingerprint(ctx.config.electronic)`; `_pyscf_backend` becomes

```python
def _pyscf_backend(ctx: StageContext, molecule: MoleculeSpec, task_dir: Path) -> ElectronicBackend:
    """The PySCF backend of one trajectory; the CASSCF reference is the sampling equilibrium."""
    try:
        samples = read_samples_h5(ctx.upstream_dir("sampling") / "samples.h5")
    except ValueError as exc:
        raise TaskError(str(exc), retryable=False) from exc
    return stage_backend(
        ctx,
        "dynamics",
        molecule,
        task_dir,
        settings=ctx.config.electronic,
        reference_coords_bohr=np.asarray(samples["equilibrium_bohr"], dtype=float),
    )
```

(`read_samples_h5`, `TaskError` and `np` are already imported in `dynamics.py`; check.) No stage `version` changes (spec 5).

- [ ] **Step 4: Run the tests**

Run: `pytest tests/stages tests/electronic/test_factory.py tests/test_orchestrator.py -q -W error` (all markers; the `pyscf` stage tests run CASSCF/TDA fast cases)
Expected: all pass. A fingerprint-golden test (if one exists) that pins a CASSCF fingerprint changes; update it and say so in the report. Then `ruff format . && ruff check . && mypy`.

- [ ] **Step 5: Commit**

```bash
git add src/gqteashmd/electronic/factory.py src/gqteashmd/stages tests/stages tests/electronic/test_factory.py
git commit -m "feat: pass each stage's CASSCF reference geometry and hash the start rule" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Measurements (report only)

**Files:**
- Create: `tools/diagnosis/start_rule_check.py`, `tests/diagnosis/test_start_rule_check.py`
- Report: `.superpowers/sdd/2026-11-13-casscf-start-rule/task-5-report.md` (git-ignored); JSON under `~/start_rule/` in WSL

**Interfaces:**
- Consumes: Tasks 2-4; `tools/diagnosis/start_solutions.py` (`match_solution`, `_equilibrium`, `N_STATES`), `tools/diagnosis/project_inputs.py:load_project_inputs`, `tools/diagnosis/run_variants.py:deep_merge`, the survey JSON `~/p9a/plan2/start_solutions.json`, the M4 project `~/p8_m4/run`.
- Produces: `start_rule_check.summarize(rows: list[dict]) -> dict` and the measured numbers.

- [ ] **Step 1: Write the failing helper test** (`tests/diagnosis/test_start_rule_check.py`; use the conftest loader as the other diagnosis tests do: `from tests.diagnosis.conftest import load_tool` or the import they use; read one first)

```python
def test_summarize_counts_the_rule_solutions():
    tool = load_tool("diagnosis", "start_rule_check")
    rows = [
        {"task": "traj_00000", "solution": 0},
        {"task": "traj_00001", "solution": 0},
        {"task": "traj_00002", "solution": 2},
        {"task": "traj_00003", "solution": None},
        {"task": "traj_00004", "error": "RuntimeError: x"},
    ]
    assert tool.summarize(rows) == {
        "n_starts": 5,
        "n_sol0": 2,
        "n_other": 1,
        "n_unmatched": 1,
        "n_errors": 1,
        "other": ["traj_00002"],
        "unmatched": ["traj_00003"],
        "errors": ["traj_00004"],
    }
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/diagnosis/test_start_rule_check.py -q -W error`
Expected: FAIL (no module).

- [ ] **Step 3: Write the tool**

```python
"""Start-rule check (CASSCF start rule, Task 5): the rule's step-0 solution at each start.

For the SA3 settings of a project, each start geometry is solved by a fresh PySCFBackend whose
reference is the project's equilibrium geometry, and the result is matched against the
start-solution survey (sol0 = the survey's lowest-SA solution). Report only.

Example (WSL):
  python tools/diagnosis/start_rule_check.py ~/p8_m4/run \
      --survey ~/p9a/plan2/start_solutions.json --out ~/start_rule/start_rule_check.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from forensics import jsonable  # noqa: E402
from project_inputs import load_project_inputs  # noqa: E402
from run_variants import deep_merge  # noqa: E402
from start_solutions import N_STATES, _equilibrium, match_solution  # noqa: E402

from gqteashmd.config.schema import ElectronicSection  # noqa: E402


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    errors = [r["task"] for r in rows if "error" in r]
    ok = [r for r in rows if "error" not in r]
    other = [r["task"] for r in ok if r.get("solution") not in (0, None)]
    unmatched = [r["task"] for r in ok if r.get("solution") is None]
    return {
        "n_starts": len(rows),
        "n_sol0": sum(1 for r in ok if r.get("solution") == 0),
        "n_other": len(other),
        "n_unmatched": len(unmatched),
        "n_errors": len(errors),
        "other": other,
        "unmatched": unmatched,
        "errors": errors,
    }


def check_start(job: tuple) -> dict[str, Any]:
    project, task, survey_solutions, memory_mb = job
    from gqteashmd.electronic.base import MoleculeSpec, Request
    from gqteashmd.electronic.pyscf_backend import PySCFBackend

    row: dict[str, Any] = {"task": task}
    try:
        inputs = load_project_inputs(Path(project))
        base = inputs.config.electronic
        assert base is not None
        electronic = ElectronicSection.model_validate(
            deep_merge(base.model_dump(mode="json"), {"singlets": N_STATES})
        )
        q_eq, _ = _equilibrium(Path(project))
        start = next(s for s in inputs.starts if s.task_id == task)
        q = np.asarray(start.q, dtype=float).reshape(-1, 3)
        backend = PySCFBackend(
            MoleculeSpec(tuple(inputs.symbols), inputs.charge, 1),
            electronic,
            memory_mb=memory_mb,
            reference_coords_bohr=q_eq,
        )
        started = time.perf_counter()
        result = backend.compute(Request(q))
        row["wall_time_s"] = time.perf_counter() - started
        energies = [float(result.energies[label]) for label in backend.states[:N_STATES]]
        row.update(
            energies_Eh=energies,
            rung=result.meta["casscf"],
            casscf_reference=result.meta["casscf_reference"],
            reference_active_min_sv=result.meta.get("reference_active_min_sv"),
            solution=match_solution(energies, survey_solutions),
        )
    except Exception as exc:  # noqa: BLE001 - a report tool records the failure
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def reference_cost(project: Path, memory_mb: int) -> dict[str, Any]:
    """Wall time of one reference search at the equilibrium geometry (SA3 settings)."""
    from gqteashmd.electronic import pyscf_casscf
    from gqteashmd.electronic.base import MoleculeSpec
    from gqteashmd.electronic.pyscf_backend import PySCFBackend

    inputs = load_project_inputs(project)
    base = inputs.config.electronic
    assert base is not None
    electronic = ElectronicSection.model_validate(
        deep_merge(base.model_dump(mode="json"), {"singlets": N_STATES})
    )
    q_eq, _ = _equilibrium(project)
    backend = PySCFBackend(
        MoleculeSpec(tuple(inputs.symbols), inputs.charge, 1), electronic, memory_mb=memory_mb
    )
    mol = backend._build_mol(q_eq, None)
    started = time.perf_counter()
    mf, _ = backend._run_scf(mol, None)
    reference, problems = pyscf_casscf.find_reference(mol, mf, electronic, backend.states)
    return {
        "wall_time_s": time.perf_counter() - started,
        "reference": None if reference is None else reference.meta(),
        "problems": problems,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("project", type=Path)
    ap.add_argument("--survey", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--memory-mb", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    project = args.project.expanduser()
    survey = json.loads(args.survey.expanduser().read_text(encoding="utf-8"))["tasks"]
    inputs = load_project_inputs(project)
    jobs = [
        (str(project), s.task_id, survey[s.task_id]["solutions"], args.memory_mb)
        for s in inputs.starts
    ]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    if args.workers > 1:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            rows = list(pool.map(check_start, jobs))
    else:
        rows = [check_start(job) for job in jobs]
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "summary": summarize(rows),
        "reference_cost": reference_cost(project, args.memory_mb),
        "rows": rows,
    }
    out.write_text(json.dumps(jsonable(report), indent=1, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Check the names this imports (`load_project_inputs(...).symbols/.charge/.starts/.config`, `start.task_id/.q`, `_equilibrium`, `match_solution`, `N_STATES`, `jsonable`, `deep_merge`) against `tools/diagnosis/start_solutions.py` and `project_inputs.py`; they are the ones `start_solutions.py` uses.

- [ ] **Step 4: Run the test, then commit the tool**

Run: `pytest tests/diagnosis -q -W error -m "not pyscf"` → pass; `ruff format . && ruff check .`.

```bash
git add tools/diagnosis/start_rule_check.py tests/diagnosis/test_start_rule_check.py
git commit -m "test: add the CASSCF start-rule check tool" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Measurement 1 — the rule at the 20 seed-2026 starts**

Run (WSL, from the repository): `python tools/diagnosis/start_rule_check.py ~/p8_m4/run --survey ~/p9a/plan2/start_solutions.json --out ~/start_rule/start_rule_check.json --workers 8 --threads 2`.
Record in the report: the summary (`n_sol0` of 20, the `other`/`unmatched`/`errors` lists), the rung of every start, the range of `reference_active_min_sv`, the reference's `guess` and `sa_energy_Eh`, `reference_cost.wall_time_s`, and the per-start wall time range. Whatever the count is, it is reported; **nothing is tuned**. A start that is not on sol0 gets its energies and singular value in the report.

- [ ] **Step 6: Measurement 2 — the cost in a real project**

Build `~/start_rule/cost/input.toml` from the M4 input (`~/p8_m4/run`'s input file: ethylene SA2-CAS(2,2)/6-31G\*) with `[sampling] n_samples = 10`, `[excitations] samples_per_task = 1`, `[selection] n_trajectories = 2`, `allow_fewer = true`, **no `[dynamics]` table**, `[resources]` 8 parallel tasks x 2 threads, a fresh `project.directory` per run. Run it twice, with the same input in two fresh directories:
- **before**: the `main` code (`git worktree add ~/start_rule/main-src main`; run `PYTHONPATH=~/start_rule/main-src/src python -m gqteashmd run input.toml`; confirm `.gqteashmd/provenance.json` names the `main` commit or that `python -c "import gqteashmd; print(gqteashmd.__file__)"` with that `PYTHONPATH` points into the worktree);
- **after**: the branch head (`python -m gqteashmd run input.toml`).

From each `logs/gqteashmd.log`, the wall time of `optimize`, `frequencies` and `excitations` (from `Stage <name>: starting` to `Stage <name>: done`). Report the six numbers and the ratios, and the `casscf` rung counts of the `excitations` samples if they are recorded anywhere (otherwise say they are not). Remove the worktree afterwards (`git worktree remove ~/start_rule/main-src`).

- [ ] **Step 7: Write the report**

`task-5-report.md`: what was run (commands, commit, times), the two measurements with every number, and nothing else. The flaky live-SHARC test is not run (ruling PR4).

---

### Task 6: Documents and the P9a closure

**Files:**
- Modify: `docs/superpowers/reports/2026-11-13-p9a-acceptance.md`, `CLAUDE.md`, `USER_MANUAL.md`, `docs/superpowers/specs/2026-11-13-casscf-start-rule-design.md` (an erratum block at its top)

**Interfaces:**
- Consumes: Task 5's report (every number quoted from it), the rulings PR1-PR6.

- [ ] **Step 1: The acceptance report**

Add a section `## 7. Owner decisions (2026-11-13)` after section 6: (1) P9a closes with the miss documented (13 of 20, rule 16); the post-hop losses are not a work item now. (2) A start-solution rule is added: the CASSCF start rule of `docs/superpowers/specs/2026-11-13-casscf-start-rule-design.md`, with Task 5's seed-2026 result (`n_sol0` of 20). (3) The dormant F4 rungs are removed (and `tools/diagnosis/convergence_probe.py`; the measurement of section 3.1 was made with it at commit `95fb881`). (4) `energy_jump_limit_eV` stays 0.1 eV.

- [ ] **Step 2: The spec erratum**

At the top of the spec, an `> **Erratum (implementation):**` block listing PR1 (the reference of `excitations`/`dynamics` is `samples.h5`'s `equilibrium_coordinates`), PR2 (`reference_coords_bohr` required in `stage_backend`), PR3 (`"not computed"`), PR4 (the flaky-test measurement dropped, why), PR6, and the measured Task 5 numbers that replace "expected 20 of 20".

- [ ] **Step 3: CLAUDE.md**

- "Read first": add the spec, this plan and the ledger `.superpowers/sdd/2026-11-13-casscf-start-rule/`.
- Workflow line: the merge state is set at the merge, not now.
- `electronic/` block (the "Ladders" bullet): replace the CASSCF ladder text with the four rungs (previous orbitals, reference orbitals, SCF orbitals, SCF orbitals + spin penalty), delete the `EXTRA_RUNGS`/F4 text, and add the start rule (reference geometry per stage: `optimize` prepared, `frequencies` working geometry, `excitations`/`dynamics` `samples.h5` equilibrium, `single-point` its own; guess set; `pick_lowest`; lazy cache across `reset`/`load_state`, not saved; fallback; metadata `casscf_reference`/`reference_active_min_sv`; `REFERENCE_OVERLAP_WARNING` 0.9; `CASSCF_START_RULE` in the fingerprints; existing CASSCF projects need `--rerun-from optimize`/`frequencies`).
- P7b block: `BackendModel.rung_changed` now also fires on a fall to "reference orbitals" (a sentence).
- P9a plan 2 bullets: "F4 did not enter ... stay in the backend, off" becomes "removed by owner decision 2026-11-13"; "Open for the owner" becomes the four decisions as taken; status line "P9a: DONE and closed (owner decision 2026-11-13: the miss documented)".
- Current status: a bullet for this sub-project (branch, commits, Task 5 numbers, test counts measured in Tasks 1-5).
- Environment tiers: the new tests' counts and times as measured.
- Known limitations 1a: "the start solution of SA3 ... no rule, owner decision 3" becomes the rule and its measured seed-2026 result; 3a: the rule does not touch that test (PR4). Deferred list: remove the F4/Task 3/Task 4 minors about the rungs and the probe; keep the others.
- Remaining work: P9a closed; P9b next.

- [ ] **Step 4: USER_MANUAL.md**

- The CASSCF section and 8.6.8: the four-rung ladder, the start rule (what the reference geometry is per stage, the guess set, the cost measured in Task 5), the two metadata keys, the warning text; mark it implemented. Remove the dormant-rung description.
- The stale-project advice: existing CASSCF projects get exit 3 and need `--rerun-from optimize` (or `frequencies`); TDA/TDDFT projects are not affected.
- 5.10: decision 1 (P9a closed with the miss), the rule replaces "no rule" (with the seed-2026 numbers), F1 default kept; the F4 rungs removed.
- Section 10: the start-solution limitation updated; 3a unchanged with the PR4 note.

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/test_cli.py -q -W error` and `ruff format . && ruff check .`.

```bash
git add docs/superpowers/reports/2026-11-13-p9a-acceptance.md docs/superpowers/specs/2026-11-13-casscf-start-rule-design.md CLAUDE.md USER_MANUAL.md
git commit -m "docs: document the CASSCF start rule and close P9a" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Self-review notes (plan author)

- Spec coverage: section 0 decisions 1, 4 -> Task 6; decision 3 / section 6 -> Task 1; sections 2-3 -> Task 2 (guess set, selection, rung) and Task 3 (lazy cache, fallback, `reset`/`load_state`, PR6); section 4 -> Task 3; section 5 -> Task 4 (marker, no version change); section 7 -> Task 6; section 8 unit and PySCF tests -> Tasks 1-4, measurements -> Task 5 (the flaky-test item dropped by PR4); the full suite -> the controller after Task 6.
- The `test_cli` single-point test and the `test_factory` end-to-end test name their inputs by behaviour (the repository's existing helpers are reused); every other test is given in full.
- Names used across tasks: `REFERENCE_RUNG`, `REFERENCE_TIE_EH`, `REFERENCE_OVERLAP_WARNING`, `Orbitals`, `guess_active_sets`, `pick_lowest`, `Reference.meta`, `find_reference`, `active_min_singular_value`, `run_casscf(..., reference=)`, `CasscfOutcome.reference_active_min_sv`, `reference_coords_bohr`, `CASSCF_START_RULE`, `electronic_fingerprint`; checked for consistency.
- Task 5 is a measurement with a small tool; by precedent (P9a1-3, P9a2-3) its numbers are checked by the Task 6 review and the final review; the tool itself gets a code review.
