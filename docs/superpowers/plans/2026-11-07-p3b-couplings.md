# P3b: Couplings for Dynamics (NACs, wavefunction overlaps, phase tracking, full backend contract) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the electronic-structure layer everything surface-hopping dynamics (P7b) needs from a backend:
- SA-CASSCF nonadiabatic coupling vectors;
- wavefunction overlaps between consecutive geometries, for every method;
- consistent state phases;
- extra gradients at the same geometry after a hop (`compute_more`);
- an explicit, serialisable reference state (`save_state`/`load_state`) so restarts are reproducible;
- a capabilities declaration;
- a scratch/chkfile policy;
- retry ladders for CASSCF and TDA/TDDFT.

**Architecture:**
- **A backend-independent `gqteashmd.overlaps` package (pure NumPy/SciPy).** It holds:
  - `DeterminantExpansion` (states as coefficients over Slater determinants);
  - the Löwdin determinant-overlap rule, which gives S_IJ = ⟨Ψ_I(R_prev)|Ψ_J(R_cur)⟩;
  - phase/root tracking.
- **The PySCF backend is split into three modules,** all of which import PySCF only inside functions:
  - `pyscf_backend.py` (orchestration, SCF, state);
  - `pyscf_casscf.py` (SA-CASSCF ladder, gradients, NACs, dipoles, determinant expansion);
  - `pyscf_response.py` (TDA/TDDFT ladder, gradients, dipoles, determinant expansion).
- **The backend keeps one explicit `_Snapshot`** of the previous geometry: coordinates, MO coefficients, SCF density and phase-aligned determinant expansion. This snapshot replaces the hidden `_previous_*` attributes. `save_state`/`load_state` write and read exactly this snapshot.

**Tech Stack:** Python 3.12, NumPy, SciPy (`linear_sum_assignment`), PySCF 2.14 (pinned), pydantic v2, pytest, ruff 0.17.4, mypy.

**Spec:**
- [`2026-11-06-gqteashmd-master-plan.md`](2026-11-06-gqteashmd-master-plan.md): §9 P3b row, §6.
- [`SHMD_implementation_strategy.md`](../../../SHMD_implementation_strategy.md):
  - §2.3 the backend contract;
  - §3.1 the PySCF table;
  - §3.3 wavefunction overlaps and phase/sign alignment;
  - §3.5 conformance checks (overlaps at Δt→0 ≈ identity, phase consistency);
  - §5.3 NAC tolerance (≤ 1·10⁻⁴ relative, up to a global sign per state pair);
  - R3 phase, sign and root-flip errors.
- [`CLAUDE.md`](../../../CLAUDE.md): rules.
- The P3a plan [`2026-11-07-p3a-pyscf-backend.md`](2026-11-07-p3a-pyscf-backend.md), for the existing backend.

## Global Constraints

- **Clean room:** no code from SHARC 4.1 is copied, translated or adapted. SHARC may only be *executed* as a black box from `tools/sharc_compare/`, and only its output files may be inspected. Never open files under `/home/ajc/sharc4-4.1`. Overlaps are implemented from the literature: Löwdin's determinant rule; Plasser et al., *JCTC* **12**, 1207 (2016) for the truncation idea. The licence is MIT.
- **Environment:** Linux only. All commands run inside WSL Oracle Linux 9.5:
  `wsl -d OracleLinux_9_5 -- bash -lc 'cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && conda activate gqteashmd-dev && <command>'`.
  Edit files with the file tools at the Windows paths, using LF line endings. Scratch work goes in `mktemp -d /tmp/gq-XXXX`, never in `$HOME` or the repository. Run one pytest process at a time.
- **Dependencies:** runtime dependencies are `pydantic>=2.6`, `numpy>=1.26`, `scipy>=1.11` (new; PySCF already requires SciPy) and `pyscf>=2.6`. The development environment pins `pyscf=2.14.*`.
- **PySCF imports:** PySCF is imported only inside functions and methods of the `gqteashmd/electronic/pyscf_*.py` modules. `import gqteashmd.cli` and `import gqteashmd.worker` must not import PySCF; an existing test checks this. `gqteashmd.overlaps` never imports PySCF.
- **Units:**
  - coordinates in bohr;
  - energies in hartree;
  - gradients in hartree/bohr;
  - NACs d_IJ = ⟨Ψ_I|∂Ψ_J/∂R⟩ in 1/bohr;
  - overlaps dimensionless;
  - dipoles in e·a₀.
- **States and couplings:**
  - Labels are `S0`, `S1`, …, and `electronic.singlets` includes S0.
  - NAC pairs are requested as `(I, J)` with `I < J` in `StateLabel` order. The result holds d_IJ, and d_JI = −d_IJ.
  - `overlaps[i, j] = ⟨Ψ_i(previous geometry)|Ψ_j(current geometry)⟩`, after phase alignment of the current states.
- **Errors:**
  - User input problems → `ConfigError` (exit 2).
  - Electronic-structure failures → `ElectronicStructureError` (exit 4), non-retryable except `MemoryError`/`OSError` raised inside PySCF.
  - Programming or contract misuse (unknown state, NACs requested from a method without them, `compute_more` before `compute`, incompatible saved state) → `ValueError`.
- **Style:** ruff 0.17.4 with line length 100 and rules E, F, W, I, B, UP, SIM. mypy must stay clean on `src/`. Type hints on all public functions.
- **Git:** branch `feat/p3b-couplings` in WSL Git. Pre-commit hooks run; never use `--no-verify`. Use conventional commit subjects and the `Co-Authored-By:` trailer of the model that wrote the commit. Do not push.
- **Documentation:** the last task updates CLAUDE.md, USER_MANUAL.md and the master plan to match the code. [IMPLEMENTED] and [PLANNED] stay apart, and every symbol and unit is defined.
- **Out of scope:** the orchestrator hardening sweep (CLAUDE.md limitation 4) is a separate small plan (P3c), by ruling. It is unrelated to couplings and would double this plan's review surface.

## Facts verified before writing this plan (PySCF 2.14.0, WSL, 2026-11-07)

1. **SA-CASSCF NACs exist.**
   - The class is `mc.nac_method()` → `pyscf.nac.sacasscf.NonAdiabaticCouplings`. `kernel(state=(i, j))` returns a (natom, 3) array, and it has `.converged`.
   - Calling with `state=(1, 0)` gives `−state=(0, 1)`, so the result is antisymmetric.
   - `mult_ediff=True` multiplies the result by E_j − E_i.
   - `use_etfs=True` drops the CSF term.
   - The default includes the CSF term and is the derivative coupling in 1/bohr.
2. **The default NAC matches SHARC.** Ethylene, SA2-CASSCF(2,2)/6-31G\*, SHARC 4.1 `SHARC_PYSCF.py` run as a black box with `nacdr`. SHARC's "! 5 Nonadiabatic couplings (ddr) (2x2x6x3)" block, pair (1,2), matches our default `state=(0,1)` to **9.1·10⁻⁸** up to an overall sign. With `use_etfs=True` the difference is 2.7·10⁻¹, so the default (full) convention is the one to use. SHARC's d₁₂ rows:
   - (−2.927020103023E−001, −1.033059334102E−002, −1.072930978298E−007)
   - (−2.927018076373E−001, −1.033064441499E−002, −7.552369004663E−008)
   - (1.120814888216E−002, 3.938780995342E−003, 4.662627690038E−007)
   - (1.145793557682E−002, −3.138817368786E−003, −4.088438886865E−007)
   - (1.145795755341E−002, −3.138796774753E−003, 4.682770179166E−007)
   - (1.120813522737E−002, 3.938776613639E−003, −4.068296702623E−007)

   The (2,1) block is the negative of the (1,2) block, and the diagonal blocks are zero.
3. **Overlaps confirm the NAC sign convention.** We built the Löwdin determinant overlap from the CASSCF CI strings and took central finite differences of ⟨0(R)|1(R±h)⟩, with h = 10⁻⁴ bohr and SA2-CAS(2,2)/6-31G:
   - for C1 x: FD 0.25668 vs NAC 0.25677;
   - for H3 y and H5 x the absolute differences are 4–6·10⁻⁵.

   `state=(0,1)` is therefore ⟨0|∂1⟩. The ~10⁻⁴ residual is the AO-derivative convention. The overlap FD differentiates the ket basis functions, while the analytic CSF term uses the antisymmetrised form ½(⟨χ|∂χ⟩ − ⟨∂χ|χ⟩), and the two differ by ½ Σ γ^{IJ}_{μν} ∂S_{μν}/∂R. It is not a bug, and the slow FD test tolerance (2·10⁻⁴ absolute) reflects it.
4. **TDA/TDDFT NACs are not available** in PySCF 2.14 (`pyscf.nac` has only `sacasscf`, and `td` has no `nac_method`). TDA/TDDFT couplings therefore go through overlaps.
5. **TDA amplitudes.** For RHF, `td.xy[k] = (X, Y)` with X of shape (n_occ, n_vir) and ΣX² = 1/2. The singlet state is Σ_ia X_ia (|Φ_i^a(α)⟩ + |Φ_i^a(β)⟩), with norm 1.
   - The determinant-overlap engine on this expansion gives the identity matrix at equal geometry, to 10⁻⁸.
   - After a random 0.01 bohr displacement of all atoms, the diagonal is |S_ii| ≈ 0.98, the singular values are 0.974–0.986, and one state came back with S₁₁ = −0.986. That is a phase flip, and tracking must fix it.
   - A naive Python double loop took 8.9 s for 4 states in 6-31G. The plan's vectorised engine computes determinants only for unique α and β strings.
6. **Files written by PySCF.** SCF writes a chkfile (`mf.chkfile`) into `lib.param.TMPDIR` (default `/tmp`, set when the SCF object is created), and nothing appears in the current directory. Setting `mf.chkfile = None` disables the file. CASSCF inherits that setting.
7. **Fix-wave facts from P3a that still hold:**
   - the SCF ladder (default → level shift 0.2 with damping 0.5 and `diis_start_cycle = 8` → second order);
   - `_wrap_failure` (MemoryError/OSError retryable);
   - `mol.build(..., dump_input=False, parse_arg=False)`.

## File Structure

```text
pyproject.toml, environment.yml            # Task 1: scipy dependency
src/gqteashmd/overlaps/__init__.py         # Task 1 (new, empty)
src/gqteashmd/overlaps/determinants.py     # Task 1 (new): DeterminantExpansion, truncate, state_overlaps
src/gqteashmd/overlaps/tracking.py         # Task 2 (new): track, TrackedOverlaps, lowdin_orthonormalize
src/gqteashmd/electronic/base.py           # Task 3: Capabilities; Request.nacs/overlaps; result nacs/overlaps/phases; ABC
src/gqteashmd/config/schema.py             # Task 3: electronic.overlap_norm_kept
src/gqteashmd/util/fileio.py               # Task 3: write_bytes_atomic
tools/sharc_compare/{qmout.py,run_sharc_pyscf.py}   # Task 4: NAC block, nacdr request
tests/data/reference/sharc41_pyscf_ethylene_sa2cas22_631gs.json   # Task 4: regenerated with "nacs_per_bohr"
src/gqteashmd/electronic/pyscf_errors.py   # Task 5 (new): wrap_failure (moved from pyscf_backend)
src/gqteashmd/electronic/pyscf_casscf.py   # Task 5 (new): run_casscf ladder, gradients, NACs, dipoles, expansion
src/gqteashmd/electronic/pyscf_response.py # Task 6 (new): run_response ladder, gradients, dipoles, expansion
src/gqteashmd/electronic/pyscf_backend.py  # Tasks 3, 5, 6, 7, 8
tests/overlaps/test_determinants.py, tests/overlaps/test_tracking.py   # Tasks 1, 2 (new)
tests/electronic/test_base.py              # Task 3
tests/config/test_electronic.py            # Task 3
tests/test_errors_fileio.py                # Task 3
tests/sharc_compare/test_qmout.py, tests/sharc_compare/test_live_sharc.py   # Task 4
tests/electronic/test_pyscf_casscf.py      # Task 5
tests/electronic/test_pyscf_linear_response.py   # Task 6
tests/electronic/test_couplings.py         # Task 7 (new): overlaps, phases, NAC requests, compute_more, scratch
tests/electronic/test_state.py             # Task 8 (new): save_state/load_state
tests/electronic/test_couplings_fd.py      # Task 9 (new, slow): NACs vs finite-difference overlaps
CLAUDE.md, USER_MANUAL.md, master plan     # Task 10
```

---

### Task 1: Determinant expansions and their overlaps (`gqteashmd.overlaps.determinants`)

**Files:**
- Modify: `pyproject.toml`, `environment.yml`
- Create: `src/gqteashmd/overlaps/__init__.py` (empty), `src/gqteashmd/overlaps/determinants.py`
- Test: `tests/overlaps/test_determinants.py`

**Interfaces:**
- Produces:
  - `DeterminantExpansion(alpha, beta, coeffs)`:
    - frozen;
    - `alpha` is an (n_det, n_α) int array, `beta` is (n_det, n_β) int, and `coeffs` is (n_states, n_det) float;
    - properties `n_states` and `n_dets`, and the method `with_signs(signs) -> DeterminantExpansion`.
  - `truncate(expansion, norm_kept: float) -> DeterminantExpansion`.
  - `state_overlaps(bra, ket, s_mo) -> np.ndarray` of shape (bra.n_states, ket.n_states).

- [ ] **Step 1: Dependencies.**
  - In `pyproject.toml`, set `dependencies = ["pydantic>=2.6", "numpy>=1.26", "scipy>=1.11", "pyscf>=2.6"]`.
  - In `environment.yml`, add `- scipy` next to `- numpy` if it is not already listed.

- [ ] **Step 2: Write the failing tests** in `tests/overlaps/test_determinants.py`:

```python
import math

import numpy as np
import pytest

from gqteashmd.overlaps.determinants import DeterminantExpansion, state_overlaps, truncate


def _rotation(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def test_single_determinant_overlap_is_the_determinant():
    rng = np.random.default_rng(1)
    s_mo = rng.normal(size=(5, 5))
    bra = DeterminantExpansion([[0, 2, 3]], [[0, 1]], [[1.0]])
    ket = DeterminantExpansion([[1, 2, 4]], [[0, 3]], [[1.0]])
    expected = np.linalg.det(s_mo[np.ix_([0, 2, 3], [1, 2, 4])]) * np.linalg.det(
        s_mo[np.ix_([0, 1], [0, 3])]
    )
    assert state_overlaps(bra, ket, s_mo)[0, 0] == pytest.approx(expected, rel=1e-12)


def test_rotated_orbital():
    wf = DeterminantExpansion([[0]], np.zeros((1, 0), dtype=int), [[1.0]])
    assert state_overlaps(wf, wf, _rotation(0.3))[0, 0] == pytest.approx(math.cos(0.3))


def test_orbital_order_defines_the_sign():
    a = DeterminantExpansion([[0, 1]], [[0]], [[1.0]])
    b = DeterminantExpansion([[1, 0]], [[0]], [[1.0]])
    assert state_overlaps(a, b, np.eye(3))[0, 0] == pytest.approx(-1.0)


def test_orthonormal_states_at_identity_give_identity():
    rng = np.random.default_rng(2)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    alpha = [[0, 1], [0, 2], [1, 2], [0, 3]]
    beta = [[0, 1]] * 4
    wf = DeterminantExpansion(alpha, beta, q[:3])
    np.testing.assert_allclose(state_overlaps(wf, wf, np.eye(4)), np.eye(3), atol=1e-12)


def test_matches_brute_force_with_mixed_spin_excitations():
    rng = np.random.default_rng(3)
    s_mo = np.eye(4) + 0.1 * rng.normal(size=(4, 4))
    alpha = [[0], [2], [0], [3]]
    beta = [[1], [1], [2], [1]]
    bra = DeterminantExpansion(alpha, beta, rng.normal(size=(2, 4)))
    ket = DeterminantExpansion(alpha[::-1], beta[::-1], rng.normal(size=(3, 4)))
    brute = np.zeros((2, 3))
    for s in range(2):
        for t in range(3):
            for k in range(4):
                for m in range(4):
                    da = s_mo[bra.alpha[k][0], ket.alpha[m][0]]
                    db = s_mo[bra.beta[k][0], ket.beta[m][0]]
                    brute[s, t] += bra.coeffs[s, k] * ket.coeffs[t, m] * da * db
    np.testing.assert_allclose(state_overlaps(bra, ket, s_mo), brute, rtol=1e-12, atol=1e-14)


def test_electron_counts_must_match():
    a = DeterminantExpansion([[0, 1]], [[0]], [[1.0]])
    b = DeterminantExpansion([[0]], [[0]], [[1.0]])
    with pytest.raises(ValueError, match="electron"):
        state_overlaps(a, b, np.eye(3))


def test_inconsistent_shapes_are_rejected():
    with pytest.raises(ValueError, match="determinant counts"):
        DeterminantExpansion([[0], [1]], [[0]], [[1.0, 0.0]])


def test_with_signs_flips_whole_states():
    wf = DeterminantExpansion([[0], [1]], [[0], [0]], [[1.0, 0.0], [0.0, 1.0]])
    flipped = wf.with_signs(np.array([1.0, -1.0]))
    np.testing.assert_array_equal(flipped.coeffs, [[1.0, 0.0], [0.0, -1.0]])


def test_truncate_keeps_the_dominant_determinants_and_renormalises():
    coeffs = np.array([[0.9, 0.4, 0.1, 0.0], [0.0, 0.1, 0.995, 0.0]])
    wf = DeterminantExpansion([[0], [1], [2], [3]], [[0]] * 4, coeffs)
    kept = truncate(wf, 0.95)
    assert kept.n_dets == 3  # determinant 3 is used by no state
    np.testing.assert_allclose(np.linalg.norm(kept.coeffs, axis=1), 1.0)
    assert kept.coeffs[0, 2] == 0.0  # 0.81 + 0.16 = 0.97 of state 0: the 0.01 determinant is dropped
    full = truncate(wf, 1.0)
    assert full.n_dets == 3
    with pytest.raises(ValueError, match="norm_kept"):
        truncate(wf, 0.0)
```

- [ ] **Step 3: Run the tests to verify they fail.**
  Run: `pytest tests/overlaps/test_determinants.py -v`
  Expected: FAIL with `ModuleNotFoundError: No module named 'gqteashmd.overlaps'`.

- [ ] **Step 4: Implement** `src/gqteashmd/overlaps/determinants.py`:

```python
"""Overlaps between many-electron states written as Slater-determinant expansions.

Backend-independent. A backend supplies, for each geometry, its MO coefficients and a
DeterminantExpansion; with s_mo = C_bra^T S_AO(R_bra, R_ket) C_ket (MO overlap across the two
geometries) Löwdin's rule gives <A|B> = det(s_mo[a, b]) for each spin, a and b being the ordered
occupied-orbital lists of the two determinants, and

    S_IJ = sum_kl c^bra_Ik c^ket_Jl det(s_mo[alpha_k, alpha_l]) det(s_mo[beta_k, beta_l]).

Determinants are evaluated once per unique alpha string and once per unique beta string.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

_MAX_BLOCK_ELEMENTS = 20_000_000  # float64 elements per batched-determinant block (~160 MB)


@dataclass(frozen=True, eq=False)
class DeterminantExpansion:
    """States as linear combinations of a shared list of Slater determinants.

    alpha[k] / beta[k]: MO indices occupied by the alpha / beta electrons of determinant k, in
    the order that defines its sign. coeffs[s, k]: coefficient of determinant k in state s.
    """

    alpha: np.ndarray
    beta: np.ndarray
    coeffs: np.ndarray

    def __init__(self, alpha: ArrayLike, beta: ArrayLike, coeffs: ArrayLike) -> None:
        a = np.asarray(alpha, dtype=np.int64)
        b = np.asarray(beta, dtype=np.int64)
        c = np.asarray(coeffs, dtype=float)
        if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
            raise ValueError("alpha, beta and coeffs must be 2-D arrays")
        if not a.shape[0] == b.shape[0] == c.shape[1]:
            raise ValueError(
                f"inconsistent determinant counts: alpha {a.shape[0]}, beta {b.shape[0]}, "
                f"coeffs {c.shape[1]}"
            )
        object.__setattr__(self, "alpha", a)
        object.__setattr__(self, "beta", b)
        object.__setattr__(self, "coeffs", c)

    @property
    def n_states(self) -> int:
        return int(self.coeffs.shape[0])

    @property
    def n_dets(self) -> int:
        return int(self.coeffs.shape[1])

    def with_signs(self, signs: ArrayLike) -> DeterminantExpansion:
        """The same states, state s multiplied by signs[s] (a phase change)."""
        return DeterminantExpansion(
            self.alpha, self.beta, self.coeffs * np.asarray(signs, dtype=float)[:, None]
        )


def truncate(expansion: DeterminantExpansion, norm_kept: float) -> DeterminantExpansion:
    """Keep, per state, the largest coefficients until their squared sum reaches `norm_kept` of
    the state's squared norm; drop determinants no state keeps; renormalise every state."""
    if not 0.0 < norm_kept <= 1.0:
        raise ValueError(f"norm_kept must be in (0, 1], got {norm_kept}")
    coeffs = expansion.coeffs.copy()
    for s in range(coeffs.shape[0]):
        weights = coeffs[s] ** 2
        total = float(weights.sum())
        if total == 0.0:
            raise ValueError(f"state {s} has only zero coefficients")
        order = np.argsort(-weights, kind="stable")
        cumulative = np.cumsum(weights[order])
        n_keep = int(np.searchsorted(cumulative, norm_kept * total * (1.0 - 1e-12))) + 1
        coeffs[s, order[n_keep:]] = 0.0
    used = np.any(coeffs != 0.0, axis=0)
    coeffs = coeffs[:, used]
    coeffs /= np.linalg.norm(coeffs, axis=1)[:, None]
    return DeterminantExpansion(expansion.alpha[used], expansion.beta[used], coeffs)


def _unique_rows(occ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if occ.shape[1] == 0:
        return occ[:1], np.zeros(occ.shape[0], dtype=np.int64)
    unique, inverse = np.unique(occ, axis=0, return_inverse=True)
    return unique, inverse.reshape(-1)


def _determinants(s_mo: np.ndarray, bra_rows: np.ndarray, ket_rows: np.ndarray) -> np.ndarray:
    """D[u, v] = det(s_mo[bra_rows[u]][:, ket_rows[v]]), batched in memory-bounded blocks."""
    n_bra, n = bra_rows.shape
    n_ket = ket_rows.shape[0]
    if n == 0:
        return np.ones((n_bra, n_ket))
    out = np.empty((n_bra, n_ket))
    step = max(1, _MAX_BLOCK_ELEMENTS // (n_ket * n * n))
    for start in range(0, n_bra, step):
        rows = bra_rows[start : start + step]
        block = s_mo[rows[:, None, :, None], ket_rows[None, :, None, :]]
        out[start : start + step] = np.linalg.det(block)
    return out


def state_overlaps(
    bra: DeterminantExpansion, ket: DeterminantExpansion, s_mo: ArrayLike
) -> np.ndarray:
    """S[I, J] = <bra_I | ket_J> given the MO overlap s_mo[p, q] = <phi^bra_p | phi^ket_q>."""
    if bra.alpha.shape[1] != ket.alpha.shape[1] or bra.beta.shape[1] != ket.beta.shape[1]:
        raise ValueError(
            f"different electron counts: bra ({bra.alpha.shape[1]} alpha, {bra.beta.shape[1]} "
            f"beta), ket ({ket.alpha.shape[1]} alpha, {ket.beta.shape[1]} beta)"
        )
    s = np.asarray(s_mo, dtype=float)
    ua, ia = _unique_rows(bra.alpha)
    va, ja = _unique_rows(ket.alpha)
    ub, ib = _unique_rows(bra.beta)
    vb, jb = _unique_rows(ket.beta)
    d = _determinants(s, ua, va)[np.ix_(ia, ja)] * _determinants(s, ub, vb)[np.ix_(ib, jb)]
    return bra.coeffs @ d @ ket.coeffs.T
```

- [ ] **Step 5: Run the tests to verify they pass.**
  Run: `pytest tests/overlaps -v -W error && mypy`
  Expected: all pass. mypy prints `Success`.

- [ ] **Step 6: Lint, full suite, commit**

```bash
ruff format src tests tools && ruff check src tests tools && pytest -m "not slow" -W error
git add pyproject.toml environment.yml src/gqteashmd/overlaps tests/overlaps
git commit -m "feat: add determinant-expansion wavefunction overlaps"
```

---

### Task 2: Phase alignment and state tracking (`gqteashmd.overlaps.tracking`)

**Files:**
- Create: `src/gqteashmd/overlaps/tracking.py`
- Test: `tests/overlaps/test_tracking.py`

**Interfaces:**
- Produces:
  - `TrackedOverlaps`, with these fields:
    - `overlaps` (S with columns multiplied by `phases`);
    - `phases` (±1.0 per current state);
    - `assignment` (`assignment[j]` = the previous state matched to current state j);
    - `min_singular_value`.

    It also has the property `root_flips -> list[tuple[int, int]]`.
  - `track(s: ArrayLike) -> TrackedOverlaps`.
  - `lowdin_orthonormalize(s: ArrayLike) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests** in `tests/overlaps/test_tracking.py`:

```python
import numpy as np
import pytest

from gqteashmd.overlaps.tracking import lowdin_orthonormalize, track


def test_identity_needs_no_correction():
    t = track(np.eye(3))
    np.testing.assert_array_equal(t.phases, [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(t.assignment, [0, 1, 2])
    assert t.root_flips == [] and t.min_singular_value == pytest.approx(1.0)


def test_a_sign_flip_is_corrected():
    s = np.diag([0.99, -0.98, 0.97])
    t = track(s)
    np.testing.assert_array_equal(t.phases, [1.0, -1.0, 1.0])
    assert np.all(np.diag(t.overlaps) > 0)


def test_a_root_flip_is_detected_and_phased_by_the_matched_element():
    s = np.array([[0.1, -0.99, 0.0], [0.98, 0.1, 0.0], [0.0, 0.0, 1.0]])
    t = track(s)
    np.testing.assert_array_equal(t.assignment, [1, 0, 2])
    assert t.root_flips == [(1, 0), (0, 1)]
    np.testing.assert_array_equal(t.phases, [1.0, -1.0, 1.0])
    assert t.overlaps[1, 0] > 0 and t.overlaps[0, 1] > 0


def test_small_singular_value_is_reported():
    s = np.array([[0.6, 0.0], [0.0, 0.4]])
    assert track(s).min_singular_value == pytest.approx(0.4)


def test_square_matrix_required():
    with pytest.raises(ValueError, match="square"):
        track(np.ones((2, 3)))


def test_lowdin_gives_the_nearest_orthogonal_matrix():
    rng = np.random.default_rng(4)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    s = 0.97 * q + 0.01 * rng.normal(size=(3, 3))
    u = lowdin_orthonormalize(s)
    np.testing.assert_allclose(u.T @ u, np.eye(3), atol=1e-12)
    assert np.abs(u - q).max() < 0.05
```

- [ ] **Step 2: Run them to verify they fail.**
  Run: `pytest tests/overlaps/test_tracking.py -v`
  Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `src/gqteashmd/overlaps/tracking.py`:

```python
"""State matching and phase alignment from S[i, j] = <Psi_i(previous)|Psi_j(current)>.

The current states are matched to the previous ones by the assignment that maximises
sum_j |S[a(j), j]| (Hungarian algorithm), and each current state gets the phase that makes its
matched overlap positive. A matched pair with a != j is a root flip (two adiabatic states swapped
their energy order between the geometries). A small singular value of S signals an intruder
state or too few computed states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True, eq=False)
class TrackedOverlaps:
    overlaps: np.ndarray  # S with column j multiplied by phases[j]
    phases: np.ndarray  # +1.0 / -1.0 for each current state
    assignment: np.ndarray  # assignment[j] = previous state matched to current state j
    min_singular_value: float

    @property
    def root_flips(self) -> list[tuple[int, int]]:
        """(previous, current) pairs of matched states whose indices differ."""
        return [(int(a), j) for j, a in enumerate(self.assignment) if a != j]


def _square(s: ArrayLike) -> np.ndarray:
    m = np.asarray(s, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"the overlap matrix must be square, got shape {m.shape}")
    return m


def track(s: ArrayLike) -> TrackedOverlaps:
    m = _square(s)
    n = m.shape[0]
    rows, cols = linear_sum_assignment(-np.abs(m))
    assignment = np.empty(n, dtype=np.int64)
    assignment[cols] = rows
    matched = m[assignment, np.arange(n)]
    phases = np.where(matched < 0.0, -1.0, 1.0)
    singular_values = np.linalg.svd(m, compute_uv=False)
    return TrackedOverlaps(m * phases[None, :], phases, assignment, float(singular_values.min()))


def lowdin_orthonormalize(s: ArrayLike) -> np.ndarray:
    """The orthogonal matrix nearest to S (Frobenius norm): U V^T from S = U Sigma V^T."""
    u, _, vt = np.linalg.svd(_square(s))
    return u @ vt
```

- [ ] **Step 4: Run the tests to verify they pass.**
  Run: `pytest tests/overlaps -v -W error && mypy`
  Expected: all pass. If mypy reports missing stubs for scipy, add `"scipy"` and `"scipy.*"` to the existing `[[tool.mypy.overrides]]` module list in `pyproject.toml`.

- [ ] **Step 5: Lint, full suite, commit**

```bash
ruff format src tests tools && ruff check src tests tools && pytest -m "not slow" -W error
git add src/gqteashmd/overlaps/tracking.py tests/overlaps/test_tracking.py pyproject.toml
git commit -m "feat: add state matching and phase alignment from overlap matrices"
```

---

### Task 3: The full backend contract, `overlap_norm_kept`, and atomic byte writes

**Files:**
- Modify: `src/gqteashmd/electronic/base.py`, `src/gqteashmd/config/schema.py`, `src/gqteashmd/util/fileio.py`, `src/gqteashmd/electronic/pyscf_backend.py`
- Test: `tests/electronic/test_base.py`, `tests/config/test_electronic.py`, `tests/test_errors_fileio.py`

**Interfaces:**
- Produces:
  - `Capabilities(gradients: bool, nacs: bool, overlaps: bool, transition_dipoles: bool)`, frozen.
  - `Request(coords_bohr, gradients=frozenset(), transition_dipoles=False, nacs=frozenset(), overlaps=False)`. A NAC pair `(a, b)` must satisfy `a < b`, otherwise `ValueError` is raised.
  - `ElectronicResult`, with three new fields and new JSON keys:
    - `nacs: Mapping[tuple[StateLabel, StateLabel], np.ndarray] = {}`;
    - `overlaps: np.ndarray | None = None`;
    - `phases: np.ndarray | None = None`;
    - `to_json()` gains `"nacs_per_bohr"` (keys like `"S0-S1"`), `"overlaps"` and `"phases"`.
  - `ElectronicBackend` gains four abstract members:
    - `capabilities` (property);
    - `compute_more(gradients=frozenset(), nacs=frozenset()) -> ElectronicResult`;
    - `save_state(path: Path) -> None`;
    - `load_state(path: Path) -> None`.
  - `ElectronicSection.overlap_norm_kept: float = 0.9999`, with `0.9 < x <= 1.0`.
  - `write_bytes_atomic(path: Path, data: bytes) -> None`.
  - `PySCFBackend` must stay instantiable:
    - `capabilities` returns `Capabilities(gradients=True, nacs=(method == "casscf"), overlaps=True, transition_dipoles=True)`;
    - `compute_more`, `save_state` and `load_state` raise `NotImplementedError("... arrives in P3b Task 7/8")` until Tasks 7 and 8 replace them;
    - `compute()` raises `ValueError("... not implemented yet")` for a request with `nacs` or `overlaps` until Tasks 5 and 7.

- [ ] **Step 1: Write the failing tests.**

Append to `tests/electronic/test_base.py`:

```python
import json

from gqteashmd.electronic.base import Capabilities


def test_request_carries_nac_pairs_and_overlaps():
    req = Request([[0, 0, 0]], nacs=frozenset({(S0, S1)}), overlaps=True)
    assert req.nacs == frozenset({(S0, S1)}) and req.overlaps is True
    with pytest.raises(ValueError, match="I < J"):
        Request([[0, 0, 0]], nacs=frozenset({(S1, S0)}))
    with pytest.raises(ValueError, match="I < J"):
        Request([[0, 0, 0]], nacs=frozenset({(S1, S1)}))


def test_capabilities_are_frozen():
    caps = Capabilities(gradients=True, nacs=False, overlaps=True, transition_dipoles=True)
    with pytest.raises(AttributeError):
        caps.nacs = True  # type: ignore[misc]


def test_result_json_with_couplings_and_overlaps():
    result = ElectronicResult(
        states=(S0, S1),
        energies={S0: -1.0, S1: -0.5},
        nacs={(S0, S1): np.full((2, 3), 0.25)},
        overlaps=np.array([[0.99, 0.01], [-0.01, 0.98]]),
        phases=np.array([1.0, -1.0]),
    )
    data = result.to_json()
    assert data["nacs_per_bohr"] == {"S0-S1": [[0.25] * 3, [0.25] * 3]}
    assert data["overlaps"] == [[0.99, 0.01], [-0.01, 0.98]]
    assert data["phases"] == [1.0, -1.0]
    json.dumps(data)
    empty = ElectronicResult(states=(S0,), energies={S0: -1.0}).to_json()
    assert empty["nacs_per_bohr"] == {} and empty["overlaps"] is None and empty["phases"] is None
```

Append to `tests/config/test_electronic.py`:

```python
def test_overlap_norm_kept(tmp_path, write_input):
    assert _load(tmp_path, write_input, CAS).electronic.overlap_norm_kept == 0.9999
    el = _load(tmp_path, write_input, CAS.replace("singlets = 2", "singlets = 2\noverlap_norm_kept = 1.0"))
    assert el.electronic.overlap_norm_kept == 1.0
    for bad in ("0.9", "1.5"):
        with pytest.raises(ConfigError, match="overlap_norm_kept"):
            _load(tmp_path, write_input, CAS.replace("singlets = 2", f"singlets = 2\noverlap_norm_kept = {bad}"))
```

Append to `tests/test_errors_fileio.py`:

```python
def test_write_bytes_atomic(tmp_path):
    from gqteashmd.util.fileio import write_bytes_atomic

    target = tmp_path / "sub" / "state.npz"
    write_bytes_atomic(target, b"\x00\x01")
    assert target.read_bytes() == b"\x00\x01"
    assert [p.name for p in target.parent.iterdir()] == ["state.npz"]
```

(If `pytest`, `np` or `Request` are not yet imported at the top of a file, add the imports there. Do not duplicate existing imports.)

- [ ] **Step 2: Run them to verify they fail.**
  Run: `pytest tests/electronic/test_base.py tests/config/test_electronic.py tests/test_errors_fileio.py -v`
  Expected: FAIL. The cause is an `ImportError` (no `Capabilities`, no `write_bytes_atomic`) and the unknown key `overlap_norm_kept`.

- [ ] **Step 3: Implement.**

`src/gqteashmd/util/fileio.py`: refactor so text and bytes share one atomic writer:

```python
def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write `data` to `path` so that readers never see a partially written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_text_atomic(path: Path, text: str) -> None:
    """Write `text` to `path` so that readers never see a partially written file."""
    write_bytes_atomic(path, text.encode("utf-8"))
```

`src/gqteashmd/config/schema.py`, in `ElectronicSection`, after `td_conv_tol`:

```python
    overlap_norm_kept: float = Field(default=0.9999, gt=0.9, le=1.0)
```

`src/gqteashmd/electronic/base.py`:
- Add `from pathlib import Path` to the imports.
- Add this class after `MoleculeSpec`:

```python
@dataclass(frozen=True)
class Capabilities:
    """What a backend can compute (strategy §2.3)."""

    gradients: bool
    nacs: bool  # analytic nonadiabatic coupling vectors d_IJ = <I|dJ/dR>
    overlaps: bool  # <Psi_I(previous geometry)|Psi_J(current geometry)>
    transition_dipoles: bool
```

- Replace `Request` with:

```python
@dataclass(frozen=True, eq=False)
class Request:
    """One electronic-structure evaluation at a geometry (coordinates in bohr).

    nacs: pairs (I, J) with I < J; the result holds d_IJ = <Psi_I|dPsi_J/dR> (1/bohr).
    overlaps: also compute the overlaps with the previous geometry and align the phases.
    """

    coords_bohr: np.ndarray
    gradients: frozenset[StateLabel] = frozenset()
    transition_dipoles: bool = False
    nacs: frozenset[tuple[StateLabel, StateLabel]] = frozenset()
    overlaps: bool = False

    def __init__(
        self,
        coords_bohr: ArrayLike,
        gradients: frozenset[StateLabel] = frozenset(),
        transition_dipoles: bool = False,
        nacs: frozenset[tuple[StateLabel, StateLabel]] = frozenset(),
        overlaps: bool = False,
    ) -> None:
        coords = np.asarray(coords_bohr, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords_bohr must have shape (natom, 3), got {coords.shape}")
        pairs = frozenset(nacs)
        bad = sorted(f"({a}, {b})" for a, b in pairs if not a < b)
        if bad:
            raise ValueError(f"NAC pairs must be ordered (I, J) with I < J, got {', '.join(bad)}")
        object.__setattr__(self, "coords_bohr", coords)
        object.__setattr__(self, "gradients", frozenset(gradients))
        object.__setattr__(self, "transition_dipoles", transition_dipoles)
        object.__setattr__(self, "nacs", pairs)
        object.__setattr__(self, "overlaps", overlaps)
```

- In `ElectronicResult`, add three fields after `spin_squares`, and update the docstring to mention NACs (1/bohr) and overlaps:

```python
    nacs: Mapping[tuple[StateLabel, StateLabel], np.ndarray] = field(default_factory=dict)
    overlaps: np.ndarray | None = None
    phases: np.ndarray | None = None
```

- In `to_json`'s returned dict, after `"gradients_hartree_per_bohr"`, add:

```python
            "nacs_per_bohr": {
                f"{a}-{b}": np.asarray(v, dtype=float).tolist()
                for (a, b), v in sorted(self.nacs.items())
            },
            "overlaps": None if self.overlaps is None else np.asarray(self.overlaps).tolist(),
            "phases": None if self.phases is None else [float(p) for p in self.phases],
```

- Replace `ElectronicBackend` with:

```python
class ElectronicBackend(ABC):
    @property
    @abstractmethod
    def states(self) -> tuple[StateLabel, ...]: ...

    @property
    @abstractmethod
    def capabilities(self) -> Capabilities: ...

    @abstractmethod
    def compute(self, request: Request) -> ElectronicResult: ...

    @abstractmethod
    def compute_more(
        self,
        gradients: frozenset[StateLabel] = frozenset(),
        nacs: frozenset[tuple[StateLabel, StateLabel]] = frozenset(),
    ) -> ElectronicResult:
        """Extra gradients/NACs at the geometry of the last compute() (e.g. after a hop)."""

    @abstractmethod
    def save_state(self, path: Path) -> None:
        """Write the reference wavefunction state the next compute() depends on."""

    @abstractmethod
    def load_state(self, path: Path) -> None:
        """Restore a state written by save_state() (same settings and molecule)."""
```

`src/gqteashmd/electronic/pyscf_backend.py`:
- Import `Capabilities`.
- At the start of `compute()`, add:

```python
        if request.nacs or request.overlaps:
            raise ValueError("NACs and overlaps are not implemented yet (P3b Tasks 5 and 7).")
```

- Add these temporary members. Tasks 5, 7 and 8 replace them.

```python
    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            gradients=True,
            nacs=self._settings.method == "casscf",
            overlaps=True,
            transition_dipoles=True,
        )

    def compute_more(
        self,
        gradients: frozenset[StateLabel] = frozenset(),
        nacs: frozenset[tuple[StateLabel, StateLabel]] = frozenset(),
    ) -> ElectronicResult:
        raise NotImplementedError("compute_more arrives in P3b Task 7")

    def save_state(self, path: Path) -> None:
        raise NotImplementedError("save_state arrives in P3b Task 8")

    def load_state(self, path: Path) -> None:
        raise NotImplementedError("load_state arrives in P3b Task 8")
```

- [ ] **Step 4: Run the tests to verify they pass.**
  Run: `pytest tests/electronic tests/config tests/test_errors_fileio.py -v -W error && mypy`
  Expected: all pass. mypy prints `Success`.

- [ ] **Step 5: Lint, full suite, commit**

```bash
ruff format src tests tools && ruff check src tests tools && pytest -m "not slow" -W error
git add src/gqteashmd/electronic/base.py src/gqteashmd/electronic/pyscf_backend.py src/gqteashmd/config/schema.py src/gqteashmd/util/fileio.py tests/electronic/test_base.py tests/config/test_electronic.py tests/test_errors_fileio.py
git commit -m "feat: extend the backend contract with couplings, overlaps, capabilities and state"
```

---

### Task 4: SHARC reference with nonadiabatic couplings

**Files:**
- Modify:
  - `tools/sharc_compare/qmout.py`
  - `tools/sharc_compare/run_sharc_pyscf.py`
  - `tests/sharc_compare/test_qmout.py`
  - `tests/sharc_compare/test_live_sharc.py`
  - `tests/data/reference/sharc41_pyscf_ethylene_sa2cas22_631gs.json` (regenerated by the tool, never edited by hand)

**Interfaces:**
- Produces:
  - `parse_qm_out(text)["nacs"]`, a list of rows per (i, j). `nacs[i][j]` is a natom × 3 list for the 0-based state pair.
  - The runner requests `nacdr` and writes `"nacs_per_bohr": {"S0-S1": nacs[0][1]}`.
  - The reference JSON gains the `"nacs_per_bohr"` key. Every other value must stay identical.

- [ ] **Step 1: Write the failing parser test.**
  - In `tests/sharc_compare/test_qmout.py`, add the following block to `SAMPLE`, between the gradient block and `! 8 Runtime`. The format matches what SHARC wrote on 2026-11-07:

```text
! 5 Nonadiabatic couplings (ddr) (2x2x2x3)
2 3 ! 1 1 0.0 1 1 0.0
 0.0E+000  0.0E+000  0.0E+000
 0.0E+000  0.0E+000  0.0E+000
2 3 ! 1 1 0.0 1 2 0.0
-2.927020103023E-001 -1.033059334102E-002 -1.072930978298E-007
-2.927018076373E-001 -1.033064441499E-002 -7.552369004663E-008
2 3 ! 1 2 0.0 1 1 0.0
 2.927020103023E-001  1.033059334102E-002  1.072930978298E-007
 2.927018076373E-001  1.033064441499E-002  7.552369004663E-008
2 3 ! 1 2 0.0 1 2 0.0
 0.0E+000  0.0E+000  0.0E+000
 0.0E+000  0.0E+000  0.0E+000
```

  - Add the test:

```python
def test_parse_nac_block():
    data = qmout.parse_qm_out(SAMPLE)
    assert data["nacs"][0][1][0][0] == -0.2927020103023
    assert data["nacs"][1][0][1][2] == 7.552369004663e-08
    assert data["nacs"][0][0] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
```

  - Add a malformed-block case in the style of the existing malformed tests: truncate the NAC block after its first header, and expect `ValueError` matching "malformed".

- [ ] **Step 2: Run it to verify it fails.**
  Run: `pytest tests/sharc_compare/test_qmout.py -v`
  Expected: FAIL with `KeyError: 'nacs'`.

- [ ] **Step 3: Implement.**
  - In `qmout.py`, parse `! 5 Nonadiabatic couplings`. Read n_states and natom from the `(NxNxAx3)` header, then read n_states × n_states blocks in row-major order (i, then j). Each block is one `"<natom> 3 ! ..."` header line followed by natom rows. Store the result as `out["nacs"][i][j]`.
  - Use the parser's existing malformed-block error helper, so truncation raises `ValueError("QM.out: malformed 'Nonadiabatic couplings' block near line <n>: ...")`.
  - Keep `nacs` optional in the "required blocks" check: older reference runs had no NACs.
  - In `run_sharc_pyscf.py`:
    - add the line `nacdr` after `grad all` in the generated `QM.in`;
    - add `"nacs_per_bohr": {f"S{i}-S{j}": data["nacs"][i][j] for i in range(n) for j in range(i + 1, n)}` to the returned dict.

- [ ] **Step 4: Regenerate the reference with the tool, then check that nothing else changed.**

```bash
python tools/sharc_compare/run_sharc_pyscf.py --sharc-home /home/ajc/sharc4-4.1 \
  --python /home/ajc/miniforge3/envs/sharc4.1/bin/python --xyz tests/data/ethylene.xyz \
  --basis "6-31g*" --ncas 2 --nelecas 2 --roots 2 \
  --out tests/data/reference/sharc41_pyscf_ethylene_sa2cas22_631gs.json
git diff --stat tests/data/reference
git diff tests/data/reference
```

  - **Expected:** the only change is the added `"nacs_per_bohr"` key. `"S0-S1"` row 0 is ≈ (−0.29270201, −0.01033059, −1.07e−07), as listed in the "Facts" section.
  - **Byte-identical values:** energies, gradients and dipoles must not change. A different `"molecule"` path string is the one exception: if the tool writes the path as given, pass the same relative path as before.
  - **If any other value changes,** stop and report. Do not commit.

- [ ] **Step 5: Extend the live test.** In `tests/sharc_compare/test_live_sharc.py`, after the dipole check, add:

```python
    for name, rows in ref["nacs_per_bohr"].items():
        fresh_rows = np.array(fresh["nacs_per_bohr"][name])
        rows = np.array(rows)
        assert min(np.abs(fresh_rows - rows).max(), np.abs(fresh_rows + rows).max()) < 1e-7
```

- [ ] **Step 6: Run.**
  - Run `pytest tests/sharc_compare -v -W error`. Expected: the parser tests pass and the live test is skipped.
  - Run `SHARC_HOME=/home/ajc/sharc4-4.1 SHARC_PYTHON=/home/ajc/miniforge3/envs/sharc4.1/bin/python pytest -m sharc -v -W error`. Expected: 1 passed.

- [ ] **Step 7: Lint, full suite, commit**

```bash
ruff format src tests tools && ruff check src tests tools && pytest -m "not slow" -W error
git add tools/sharc_compare tests/sharc_compare tests/data/reference/sharc41_pyscf_ethylene_sa2cas22_631gs.json
git commit -m "test: add SHARC 4.1 nonadiabatic couplings to the ethylene reference"
```

---

### Task 5: SA-CASSCF module: retry ladder, NACs, determinant expansion

**Files:**
- Create: `src/gqteashmd/electronic/pyscf_errors.py`, `src/gqteashmd/electronic/pyscf_casscf.py`
- Modify: `src/gqteashmd/electronic/pyscf_backend.py`, `tests/electronic/test_pyscf_casscf.py`

**Interfaces:**
- Consumes:
  - `DeterminantExpansion`, `state_overlaps` (Task 1);
  - `Request.nacs` (Task 3);
  - the reference JSON `"nacs_per_bohr"` (Task 4).
- Produces:
  - **`pyscf_errors.wrap_failure(what, exc) -> ElectronicStructureError`.** This is the old `_wrap_failure`, moved here. `pyscf_backend` imports it under the old name only if a test still needs it; update the test import instead.
  - **`pyscf_casscf.SPIN_TOLERANCE = 1e-3`.**
  - **`CasscfOutcome(mc, rung, energies, spin_squares)`.**
  - **`run_casscf(mol, mf, settings, states, previous: tuple[np.ndarray, Any] | None) -> CasscfOutcome`.** It runs this ladder:
    1. "previous orbitals", skipped when `previous` is None;
    2. "SCF orbitals";
    3. "SCF orbitals + spin penalty", with `fix_spin_(ss=0, shift=0.5)` and 3× macro iterations.

    A rung fails when its result is not converged or when a root has |⟨S²⟩| > 1e-3. An exception raised by PySCF aborts the ladder. The convergence test is the seam `_converged(mc) -> bool`.
  - **`casscf_gradients(mc, labels) -> dict[StateLabel, np.ndarray]`.**
  - **`casscf_nacs(mc, pairs) -> dict[tuple[StateLabel, StateLabel], np.ndarray]`.**
  - **`casscf_dipoles(mol, mc, states) -> dict[StateLabel, np.ndarray]`.**
  - **`casscf_expansion(mc) -> DeterminantExpansion`.** It lists core orbitals first, then the active orbitals of each FCI string in ascending order; α and β strings come from `pyscf.fci.cistring.make_strings`. Coefficients are `mc.ci[k].reshape(-1)`, with the α string as the slow index.
  - **The backend's `compute()`** uses these functions for `method = "casscf"`. NAC requests are allowed for casscf, and the backend raises `ValueError` for TDA/TDDFT. Result `meta["casscf"]` holds the rung name.

- [ ] **Step 1: Write the failing tests.**
  - In `tests/electronic/test_pyscf_casscf.py`, change the import of `_wrap_failure` to `from gqteashmd.electronic.pyscf_errors import wrap_failure as _wrap_failure`.
  - Append:

```python
from gqteashmd.electronic import pyscf_casscf
from gqteashmd.overlaps.determinants import state_overlaps


def test_sa2_casscf_nacs_match_the_sharc_reference(ethylene_symbols, ethylene_bohr, make_settings,
                                                   reference_dir):
    ref = json.loads((reference_dir / "sharc41_pyscf_ethylene_sa2cas22_631gs.json").read_text())
    result = _backend(ethylene_symbols, make_settings()).compute(
        Request(ethylene_bohr, nacs=frozenset({(S0, S1)}))
    )
    d = result.nacs[(S0, S1)]
    d_ref = np.array(ref["nacs_per_bohr"]["S0-S1"])
    assert d.shape == (6, 3)
    assert min(np.abs(d - d_ref).max(), np.abs(d + d_ref).max()) < 1e-6  # global sign arbitrary
    assert result.meta["casscf"] in ("SCF orbitals", "previous orbitals")


def test_nacs_need_known_states_and_a_capable_method(ethylene_symbols, ethylene_bohr,
                                                     make_settings):
    backend = _backend(ethylene_symbols, make_settings(basis="sto-3g"))
    with pytest.raises(ValueError, match="S5"):
        backend.compute(Request(ethylene_bohr, nacs=frozenset({(S0, StateLabel(1, 5))})))
    tda = _backend(ethylene_symbols, make_settings(method="tda", basis="sto-3g",
                                                   functional="hf", casscf=None))
    assert tda.capabilities.nacs is False
    with pytest.raises(ValueError, match="overlaps"):
        tda.compute(Request(ethylene_bohr, nacs=frozenset({(S0, S1)})))


def test_casscf_expansion_is_orthonormal_at_equal_geometry(ethylene_symbols, ethylene_bohr,
                                                          make_settings):
    from pyscf import gto, scf

    settings = make_settings(basis="6-31g")
    mol = gto.M(atom=[(s, tuple(c)) for s, c in zip(ethylene_symbols, ethylene_bohr, strict=True)],
                unit="Bohr", basis="6-31g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    outcome = pyscf_casscf.run_casscf(mol, mf, settings, (S0, S1), None)
    expansion = pyscf_casscf.casscf_expansion(outcome.mc)
    assert expansion.n_dets == 4 and expansion.n_states == 2
    s_mo = outcome.mc.mo_coeff.T @ mol.intor("int1e_ovlp") @ outcome.mc.mo_coeff
    np.testing.assert_allclose(state_overlaps(expansion, expansion, s_mo), np.eye(2), atol=1e-8)


def test_casscf_ladder_falls_back_to_later_rungs(ethylene_symbols, ethylene_bohr, make_settings,
                                                 monkeypatch):
    calls = []

    def flaky(mc):
        calls.append(1)
        return len(calls) > 1 and bool(mc.converged)  # the first rung "fails"

    monkeypatch.setattr(pyscf_casscf, "_converged", flaky)
    result = _backend(ethylene_symbols, make_settings(basis="sto-3g")).compute(Request(ethylene_bohr))
    assert result.meta["casscf"] == "SCF orbitals + spin penalty"  # no previous orbitals: rung 1 skipped
    assert len(calls) == 2


def test_casscf_ladder_spin_penalty_rung_gives_the_same_states(ethylene_symbols, ethylene_bohr,
                                                               make_settings, monkeypatch):
    plain = _backend(ethylene_symbols, make_settings(basis="sto-3g")).compute(
        Request(ethylene_bohr, gradients=frozenset({S1}))
    )
    calls = []

    def fail_first(mc):
        calls.append(1)
        return len(calls) > 1 and bool(mc.converged)

    monkeypatch.setattr(pyscf_casscf, "_converged", fail_first)
    penalised = _backend(ethylene_symbols, make_settings(basis="sto-3g")).compute(
        Request(ethylene_bohr, gradients=frozenset({S1}))
    )
    assert penalised.meta["casscf"] == "SCF orbitals + spin penalty"
    for label in (S0, S1):
        assert abs(penalised.energies[label] - plain.energies[label]) < 1e-6
    assert np.abs(penalised.gradients[S1] - plain.gradients[S1]).max() < 1e-5


def test_casscf_ladder_exhausted(ethylene_symbols, ethylene_bohr, make_settings, monkeypatch):
    monkeypatch.setattr(pyscf_casscf, "_converged", lambda mc: False)
    with pytest.raises(ElectronicStructureError, match="SA-CASSCF failed on every rung"):
        _backend(ethylene_symbols, make_settings(basis="sto-3g")).compute(Request(ethylene_bohr))
```

  - In `test_casscf_ladder_falls_back_to_later_rungs`: with no previous orbitals, rung 1 is skipped, so the first evaluated rung is "SCF orbitals". When the seam makes it fail, rung 3 runs. That is why `len(calls) == 2` and the rung is the spin-penalty one. If the implementer finds the spin-penalty rung cannot produce gradients in PySCF 2.14, report the measured exception. Do not silently drop the gradient assertion in the second test.

- [ ] **Step 2: Run them to verify they fail.**
  Run: `pytest tests/electronic/test_pyscf_casscf.py -v`
  Expected: FAIL with `ModuleNotFoundError: gqteashmd.electronic.pyscf_errors` (or the import of `pyscf_casscf`).

- [ ] **Step 3: Implement.**

`src/gqteashmd/electronic/pyscf_errors.py`:

```python
"""Error mapping shared by the PySCF modules."""

from __future__ import annotations

from gqteashmd.errors import ElectronicStructureError


def wrap_failure(what: str, exc: Exception) -> ElectronicStructureError:
    """Wrap an exception raised inside PySCF; only out-of-memory and I/O errors are retryable."""
    return ElectronicStructureError(
        f"{what} failed: {type(exc).__name__}: {exc}",
        retryable=isinstance(exc, (MemoryError, OSError)),
    )
```

`src/gqteashmd/electronic/pyscf_casscf.py`:

```python
"""SA-CASSCF with PySCF: retry ladder, gradients, NACs, transition dipoles, determinants.

PySCF is imported inside the functions only. All quantities are in atomic units.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from gqteashmd.config.schema import ElectronicSection
from gqteashmd.core.states import StateLabel
from gqteashmd.electronic.pyscf_errors import wrap_failure
from gqteashmd.errors import ElectronicStructureError
from gqteashmd.overlaps.determinants import DeterminantExpansion

SPIN_TOLERANCE = 1e-3  # |<S^2>| above this means a root is not a singlet

# (name, start from the previous geometry's orbitals, spin-penalty shift / Eh, macro-cycle factor)
_LADDER: tuple[tuple[str, bool, float, int], ...] = (
    ("previous orbitals", True, 0.0, 1),
    ("SCF orbitals", False, 0.0, 1),
    ("SCF orbitals + spin penalty", False, 0.5, 3),
)


@dataclass(eq=False)
class CasscfOutcome:
    mc: Any
    rung: str
    energies: dict[StateLabel, float]
    spin_squares: dict[StateLabel, float]


def _converged(mc: Any) -> bool:
    """Convergence test of a CASSCF run (a seam for tests)."""
    return bool(mc.converged)


def check_active_space(mol: Any, settings: ElectronicSection) -> None:
    cas = settings.casscf
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
            f"basis '{settings.basis}' has only {mol.nao}."
        )


def run_casscf(
    mol: Any,
    mf: Any,
    settings: ElectronicSection,
    states: tuple[StateLabel, ...],
    previous: tuple[np.ndarray, Any] | None,
) -> CasscfOutcome:
    """Equal-weight SA-CASSCF over `states` (singlets), trying the rungs of _LADDER in order.

    `previous` = (MO coefficients, Mole) of the previous geometry, projected onto this geometry
    for the first rung.
    """
    from pyscf import fci, mcscf

    cas = settings.casscf
    assert cas is not None
    check_active_space(mol, settings)
    n_states = len(states)
    problems: list[str] = []
    for name, use_previous, spin_shift, macro_factor in _LADDER:
        if use_previous and previous is None:
            continue
        mc = mcscf.CASSCF(mf, cas.ncas, cas.nelecas)
        mc.fcisolver = fci.solver(mol, singlet=True)  # singlet-only CI (excludes triplets)
        mc = mc.state_average_([1.0 / n_states] * n_states)
        if spin_shift:
            mc.fix_spin_(ss=0, shift=spin_shift)  # also penalise quintets and higher spins
        mc.conv_tol = cas.conv_tol
        mc.conv_tol_grad = cas.conv_tol_grad
        mc.max_cycle_macro = mc.max_cycle_macro * macro_factor
        mo0 = mf.mo_coeff
        if use_previous:
            assert previous is not None
            previous_mo, previous_mol = previous
            try:
                # orbitals of another geometry are not orthonormal here: project them first
                mo0 = mcscf.project_init_guess(mc, previous_mo, previous_mol)
            except Exception as exc:
                raise wrap_failure("Projecting the previous orbitals", exc) from exc
        try:
            mc.kernel(mo_coeff=mo0)
        except Exception as exc:
            # an exception (unlike non-convergence) signals a broken setup: abort the ladder
            raise wrap_failure(f"CASSCF ({name})", exc) from exc
        if not _converged(mc):
            problems.append(f"{name}: not converged")
            continue
        spins = {
            label: float(fci.spin_op.spin_square0(ci, mc.ncas, mc.nelecas)[0])
            for label, ci in zip(states, mc.ci, strict=True)
        }
        wrong = [f"{label} <S^2> = {s:.4f}" for label, s in spins.items() if abs(s) > SPIN_TOLERANCE]
        if wrong:
            problems.append(f"{name}: not all roots are singlets ({', '.join(wrong)})")
            continue
        energies = {label: float(e) for label, e in zip(states, mc.e_states, strict=True)}
        return CasscfOutcome(mc, name, energies, spins)
    raise ElectronicStructureError(
        f"SA-CASSCF failed on every rung (conv_tol = {cas.conv_tol}, conv_tol_grad = "
        f"{cas.conv_tol_grad}): {'; '.join(problems)}."
    )


def casscf_gradients(mc: Any, labels: Iterable[StateLabel]) -> dict[StateLabel, np.ndarray]:
    gradients: dict[StateLabel, np.ndarray] = {}
    wanted = sorted(labels)
    if not wanted:
        return gradients
    solver = mc.nuc_grad_method()
    for label in wanted:
        try:
            g = solver.kernel(state=label.index)
        except Exception as exc:
            raise wrap_failure(f"The SA-CASSCF gradient of {label}", exc) from exc
        if not getattr(solver, "converged", True):
            raise ElectronicStructureError(f"The SA-CASSCF gradient of {label} did not converge.")
        gradients[label] = np.asarray(g, dtype=float)
    return gradients


def casscf_nacs(
    mc: Any, pairs: Iterable[tuple[StateLabel, StateLabel]]
) -> dict[tuple[StateLabel, StateLabel], np.ndarray]:
    """d_IJ = <Psi_I|dPsi_J/dR> (1/bohr), including the CSF term (PySCF's default convention)."""
    nacs: dict[tuple[StateLabel, StateLabel], np.ndarray] = {}
    wanted = sorted(pairs)
    if not wanted:
        return nacs
    solver = mc.nac_method()
    for a, b in wanted:
        try:
            d = solver.kernel(state=(a.index, b.index))
        except Exception as exc:
            raise wrap_failure(f"The SA-CASSCF coupling {a}-{b}", exc) from exc
        if not getattr(solver, "converged", True):
            raise ElectronicStructureError(f"The SA-CASSCF coupling {a}-{b} did not converge.")
        nacs[(a, b)] = np.asarray(d, dtype=float)
    return nacs


def casscf_dipoles(
    mol: Any, mc: Any, states: tuple[StateLabel, ...]
) -> dict[StateLabel, np.ndarray]:
    """S0 -> state transition dipoles mu = -sum_pq r_pq gamma^{0I}_pq over active orbitals."""
    dipoles: dict[StateLabel, np.ndarray] = {}
    try:
        r_ints = mol.intor("int1e_r", comp=3)
        mo_cas = mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas]
        for label in states[1:]:
            t_mo = mc.fcisolver.trans_rdm1(mc.ci[0], mc.ci[label.index], mc.ncas, mc.nelecas)
            t_ao = mo_cas @ t_mo @ mo_cas.T
            dipoles[label] = -np.einsum("xij,ji->x", r_ints, t_ao)
    except Exception as exc:
        raise wrap_failure("The transition dipole calculation", exc) from exc
    return dipoles


def casscf_expansion(mc: Any) -> DeterminantExpansion:
    """SA-CASSCF states as determinants over mc.mo_coeff: core orbitals, then the active orbitals
    of each alpha/beta FCI string in ascending order (alpha string = slow CI index)."""
    from pyscf.fci import cistring

    ncore, ncas = int(mc.ncore), int(mc.ncas)
    n_alpha, n_beta = (int(n) for n in mc.nelecas)
    core = list(range(ncore))

    def occupations(n_el: int) -> list[list[int]]:
        return [
            core + [ncore + k for k in range(ncas) if int(s) >> k & 1]
            for s in cistring.make_strings(range(ncas), n_el)
        ]

    occ_alpha, occ_beta = occupations(n_alpha), occupations(n_beta)
    alpha = [a for a in occ_alpha for _ in occ_beta]
    beta = [b for _ in occ_alpha for b in occ_beta]
    coeffs = np.array([np.asarray(ci, dtype=float).reshape(-1) for ci in mc.ci])
    return DeterminantExpansion(alpha, beta, coeffs)
```

`src/gqteashmd/electronic/pyscf_backend.py`:
- Delete `_wrap_failure`, `SPIN_TOLERANCE` and `_run_casscf`. Import `wrap_failure` from `pyscf_errors`, and replace the remaining `_wrap_failure(` calls with `wrap_failure(`.
- Replace the temporary NAC guard from Task 3 with:

```python
        unknown = sorted(
            str(s) for s in {s for pair in request.nacs for s in pair} - set(self._states)
        )
        if unknown:
            raise ValueError(f"NACs requested for unknown state(s): {', '.join(unknown)}")
        if request.nacs and not self.capabilities.nacs:
            raise ValueError(
                f"Nonadiabatic coupling vectors are not available for method = "
                f"'{self._settings.method}'; request overlaps instead."
            )
        if request.overlaps:
            raise ValueError("Overlaps are not implemented yet (P3b Task 7).")
```

- In `compute()`, replace the casscf branch with the code below. Then add `**rung_meta` to `meta`, and pass `nacs=nacs` to `ElectronicResult`. For the TDA branch, set `nacs = {}` and `rung_meta = {}` for now (Task 6 changes it).

```python
            if self._settings.method == "casscf":
                previous = None
                if self._previous_mo is not None and self._previous_mol is not None:
                    previous = (self._previous_mo, self._previous_mol)
                outcome = pyscf_casscf.run_casscf(mol, mf, self._settings, self._states, previous)
                mc = outcome.mc
                energies, spins = outcome.energies, outcome.spin_squares
                gradients = pyscf_casscf.casscf_gradients(mc, request.gradients)
                nacs = pyscf_casscf.casscf_nacs(mc, request.nacs)
                dipoles = (
                    pyscf_casscf.casscf_dipoles(mol, mc, self._states)
                    if request.transition_dipoles
                    else {}
                )
                rung_meta = {"casscf": outcome.rung}
                self._previous_mo, self._previous_mol = mc.mo_coeff, mol
```

  Import the module with `from gqteashmd.electronic import pyscf_casscf`.
- Update `tests/electronic/test_pyscf_casscf.py` assertions that matched the old CASSCF messages ("SA-CASSCF did not converge", "not a singlet") to the new "SA-CASSCF failed on every rung" text. Report every assertion you change.

- [ ] **Step 4: Run the tests to verify they pass.**
  Run: `pytest tests/electronic -v -W error && mypy`
  Expected: all pass, including every P3a test.

- [ ] **Step 5: Lint, full suite, commit**

```bash
ruff format src tests tools && ruff check src tests tools && pytest -m "not slow" -W error
git add src/gqteashmd/electronic tests/electronic
git commit -m "feat: add SA-CASSCF couplings, a CASSCF retry ladder and determinant expansions"
```

---

### Task 6: TDA/TDDFT module: retry ladder and determinant expansion

**Files:**
- Create: `src/gqteashmd/electronic/pyscf_response.py`
- Modify: `src/gqteashmd/electronic/pyscf_backend.py`, `tests/electronic/test_pyscf_linear_response.py`

**Interfaces:**
- Produces:
  - `ResponseOutcome(td, rung, energies)`. For singlets = 1, `td` is None and `rung` is None.
  - `run_response(mf, settings, states) -> ResponseOutcome`. The ladder has two rungs:
    1. "default": `nstates = n`;
    2. "extra roots": `nstates = n + 3`, 3× `max_cycle`, and the lowest n roots are kept.

    The seam `_converged_roots(td, n) -> bool` checks that `len(td.e) >= n` and `converged[:n].all()`.
  - `response_gradients(mf, td, labels, method) -> dict`.
  - `response_dipoles(td, states, method) -> dict`.
  - `response_expansion(mf, td, n_excited) -> DeterminantExpansion`. Determinant 0 is the ground determinant. Next come the α single excitations i→a (row-major in (i, a)), placed in i's slot of the occupied list, then the β singles. State k+1 has coefficient `X_k[i, a]` on both its α and β determinants. For TDDFT the expansion uses X only (RPA's Y is dropped), and `truncate` renormalises it.
  - Result `meta["td"]` holds the rung name, or None when there are no excited states.

- [ ] **Step 1: Write the failing tests.** Append to `tests/electronic/test_pyscf_linear_response.py`:

```python
from gqteashmd.electronic import pyscf_response
from gqteashmd.overlaps.determinants import state_overlaps, truncate


def _scf(symbols, coords, functional):
    mol = gto.M(atom=[(s, tuple(c)) for s, c in zip(symbols, coords, strict=True)],
                unit="Bohr", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol) if functional == "hf" else dft.RKS(mol)
    if functional != "hf":
        mf.xc = functional
    mf.conv_tol = 1e-10
    mf.kernel()
    return mol, mf


def test_tda_expansion_is_orthonormal_at_equal_geometry(ethylene_symbols, ethylene_bohr,
                                                       make_settings):
    mol, mf = _scf(ethylene_symbols, ethylene_bohr, "b3lyp")
    settings = _tda_settings(make_settings, singlets=3)
    outcome = pyscf_response.run_response(mf, settings, (S0, S1, S2))
    expansion = pyscf_response.response_expansion(mf, outcome.td, 2)
    s_mo = mf.mo_coeff.T @ mol.intor("int1e_ovlp") @ mf.mo_coeff
    np.testing.assert_allclose(state_overlaps(expansion, expansion, s_mo), np.eye(3), atol=1e-8)
    kept = truncate(expansion, 0.9999)
    assert kept.n_dets < expansion.n_dets
    np.testing.assert_allclose(np.diag(state_overlaps(kept, kept, s_mo)), 1.0, atol=1e-10)


def test_tdhf_x_only_expansion_is_nearly_orthonormal(ethylene_symbols, ethylene_bohr,
                                                     make_settings):
    mol, mf = _scf(ethylene_symbols, ethylene_bohr, "hf")
    settings = _tda_settings(make_settings, method="tddft", functional="hf", singlets=3)
    outcome = pyscf_response.run_response(mf, settings, (S0, S1, S2))
    expansion = truncate(pyscf_response.response_expansion(mf, outcome.td, 2), 1.0)
    s_mo = mf.mo_coeff.T @ mol.intor("int1e_ovlp") @ mf.mo_coeff
    s = state_overlaps(expansion, expansion, s_mo)
    np.testing.assert_allclose(np.diag(s), 1.0, atol=1e-10)
    assert np.abs(s - np.eye(3)).max() < 0.05  # X-only RPA states: report the measured value


def test_response_ladder_retries_with_extra_roots(ethylene_symbols, ethylene_bohr, make_settings,
                                                  monkeypatch):
    calls = []
    original = pyscf_response._converged_roots

    def flaky(td, n):
        calls.append(td.nstates)
        return len(calls) > 1 and original(td, n)

    monkeypatch.setattr(pyscf_response, "_converged_roots", flaky)
    result = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), _tda_settings(make_settings)).compute(
        Request(ethylene_bohr)
    )
    assert calls == [2, 5] and result.meta["td"] == "extra roots"
    _, td = _direct(ethylene_symbols, ethylene_bohr, "b3lyp")
    for label, omega in zip((S1, S2), td.e, strict=True):
        assert abs(result.excitation_energy(label) - omega) < 1e-6


def test_response_ladder_exhausted(ethylene_symbols, ethylene_bohr, make_settings, monkeypatch):
    monkeypatch.setattr(pyscf_response, "_converged_roots", lambda td, n: False)
    with pytest.raises(ElectronicStructureError, match="did not converge for all 2 excited states"):
        PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), _tda_settings(make_settings)).compute(
            Request(ethylene_bohr)
        )
```

(`_tda_settings` defaults to 3 singlets, so the excited count is 2 and `calls == [2, 5]`. If the existing helper differs, adapt the test numbers and report the change. Add the imports `dft`, `gto` and `scf` from pyscf at the top if they are not already there.)

- [ ] **Step 2: Run them to verify they fail.**
  Run: `pytest tests/electronic/test_pyscf_linear_response.py -v`
  Expected: FAIL with `ImportError` for `pyscf_response`.

- [ ] **Step 3: Implement** `src/gqteashmd/electronic/pyscf_response.py`:

```python
"""TDA / TDDFT (CIS / TDHF with functional "hf") with PySCF: retry ladder, gradients, dipoles,
determinant expansion. PySCF is imported inside the functions only. Atomic units."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from gqteashmd.config.schema import ElectronicSection
from gqteashmd.core.states import StateLabel
from gqteashmd.electronic.pyscf_errors import wrap_failure
from gqteashmd.errors import ElectronicStructureError
from gqteashmd.overlaps.determinants import DeterminantExpansion

# (name, extra roots in the Davidson solver, max_cycle factor)
_LADDER: tuple[tuple[str, int, int], ...] = (("default", 0, 1), ("extra roots", 3, 3))


@dataclass(eq=False)
class ResponseOutcome:
    td: Any
    rung: str | None
    energies: dict[StateLabel, float]


def _converged_roots(td: Any, n: int) -> bool:
    """True when the lowest n roots exist and converged (a seam for tests)."""
    converged = np.atleast_1d(np.asarray(td.converged, dtype=bool))
    return len(td.e) >= n and bool(converged[:n].all())


def run_response(
    mf: Any, settings: ElectronicSection, states: tuple[StateLabel, ...]
) -> ResponseOutcome:
    from pyscf import tddft

    method = settings.method.upper()
    excited = states[1:]
    energies: dict[StateLabel, float] = {states[0]: float(mf.e_tot)}
    if not excited:
        return ResponseOutcome(None, None, energies)
    for name, extra, cycle_factor in _LADDER:
        td = tddft.TDA(mf) if settings.method == "tda" else tddft.TDDFT(mf)
        td.nstates = len(excited) + extra
        td.conv_tol = settings.td_conv_tol
        td.max_cycle = td.max_cycle * cycle_factor
        try:
            td.kernel()
        except Exception as exc:
            raise wrap_failure(f"{method} ({name})", exc) from exc
        if _converged_roots(td, len(excited)):
            for label, omega in zip(excited, td.e[: len(excited)], strict=True):
                energies[label] = float(mf.e_tot + omega)
            return ResponseOutcome(td, name, energies)
    tried = ", ".join(name for name, *_ in _LADDER)
    raise ElectronicStructureError(
        f"{method} did not converge for all {len(excited)} excited states (tried: {tried}; "
        f"td_conv_tol = {settings.td_conv_tol})."
    )


def response_gradients(
    mf: Any, td: Any, labels: Iterable[StateLabel], method: str
) -> dict[StateLabel, np.ndarray]:
    gradients: dict[StateLabel, np.ndarray] = {}
    for label in sorted(labels):
        try:
            if label.index == 0:
                g = mf.nuc_grad_method().kernel()
            else:
                g = td.nuc_grad_method().kernel(state=label.index)
        except Exception as exc:
            what = "SCF" if label.index == 0 else method.upper()
            raise wrap_failure(f"{what} gradient of {label}", exc) from exc
        gradients[label] = np.asarray(g, dtype=float)
    return gradients


def response_dipoles(
    td: Any, states: tuple[StateLabel, ...], method: str
) -> dict[StateLabel, np.ndarray]:
    excited = states[1:]
    if not excited:
        return {}
    try:
        mus = td.transition_dipole()[: len(excited)]
    except Exception as exc:
        raise wrap_failure(f"{method.upper()} transition dipole", exc) from exc
    return {label: np.asarray(mu, dtype=float) for label, mu in zip(excited, mus, strict=True)}


def response_expansion(mf: Any, td: Any, n_excited: int) -> DeterminantExpansion:
    """Ground determinant plus spin-adapted single excitations over mf.mo_coeff.

    State k+1 = sum_ia X_k[i, a] (|i->a, alpha> + |i->a, beta>), the excited orbital taking i's
    place in the occupied list (the sign convention of a_a^+ a_i). For TDDFT only X is used (the
    RPA de-excitation part Y is dropped); truncate() renormalises.
    """
    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    nvir = int(mf.mo_coeff.shape[1]) - nocc
    occ = np.arange(nocc)
    n_single = nocc * nvir
    excited_occ = np.tile(occ, (n_single, 1))
    i_index, a_index = np.divmod(np.arange(n_single), nvir)
    excited_occ[np.arange(n_single), i_index] = nocc + a_index
    ground = np.tile(occ, (n_single, 1))
    alpha = np.vstack([occ[None, :], excited_occ, ground])
    beta = np.vstack([occ[None, :], ground, excited_occ])
    coeffs = np.zeros((1 + n_excited, 1 + 2 * n_single))
    coeffs[0, 0] = 1.0
    for k in range(n_excited):
        x = np.asarray(td.xy[k][0], dtype=float).reshape(-1)
        coeffs[k + 1, 1 : 1 + n_single] = x
        coeffs[k + 1, 1 + n_single :] = x
    return DeterminantExpansion(alpha, beta, coeffs)
```

In `pyscf_backend.py`, delete `_run_linear_response` and use the module in the TDA/TDDFT branch:

```python
            else:
                outcome_r = pyscf_response.run_response(mf, self._settings, self._states)
                energies, spins = outcome_r.energies, {}
                gradients = pyscf_response.response_gradients(
                    mf, outcome_r.td, request.gradients, self._settings.method
                )
                nacs = {}
                dipoles = (
                    pyscf_response.response_dipoles(outcome_r.td, self._states, self._settings.method)
                    if request.transition_dipoles
                    else {}
                )
                rung_meta = {"td": outcome_r.rung}
```

Update any P3a test assertion that matched the old TD failure text, and report the change.

- [ ] **Step 4: Run the tests to verify they pass.**
  Run: `pytest tests/electronic -v -W error && mypy`
  Expected: all pass. Report the measured `max|S − I|` of the TDHF X-only test.

- [ ] **Step 5: Lint, full suite, commit**

```bash
ruff format src tests tools && ruff check src tests tools && pytest -m "not slow" -W error
git add src/gqteashmd/electronic tests/electronic
git commit -m "feat: add a TDA/TDDFT retry ladder and response determinant expansions"
```

---

### Task 7: Backend integration: snapshot, overlaps and phases, `compute_more`, scratch policy

**Files:**
- Modify: `src/gqteashmd/electronic/pyscf_backend.py`
- Create: `tests/electronic/test_couplings.py`

**Interfaces:**
- Consumes:
  - `truncate`, `state_overlaps` (Task 1);
  - `track` (Task 2);
  - the contract (Task 3);
  - `pyscf_casscf` (Task 5) and `pyscf_response` (Task 6).
- Produces:
  - **`PySCFBackend(molecule, settings, *, log_file=None, threads=None, memory_mb=None, scratch_dir: Path | None = None)`.**
  - **The `_Snapshot(coords, mo_coeff, density, expansion)` dataclass.** It replaces `_previous_mol`, `_previous_mo` and `_previous_dm`. `expansion` is truncated with `settings.overlap_norm_kept` and phase-aligned.
  - **`compute(request)`:**
    - When `request.overlaps` is set and a snapshot exists, it computes `S = state_overlaps(previous.expansion, current.expansion, C_prev^T S_AO(prev, cur) C_cur)` and calls `track(S)`.
    - It returns `overlaps` and `phases`, and multiplies the current expansion's states by the phases.
    - It multiplies NACs d_IJ by p_I p_J and dipoles μ_0I by p_0 p_I.
    - It sets `meta["overlaps"]` to one of `"computed"`, `"no previous geometry"` or `"not requested"`.
    - With overlaps computed, it also sets `meta["overlap_min_singular_value"]` and `meta["root_flips"]`, and logs a warning (`logging.getLogger(__name__)`) when the minimum singular value is below 0.5.
  - **`compute_more(gradients, nacs)`:**
    - It uses the cached PySCF objects of the last `compute()` and does not rerun the SCF or the correlated method.
    - Phases are applied to the NACs.
    - It returns energies, the requested gradients and NACs, and `meta["compute_more"] = True`.
    - Before any `compute()` it raises `ValueError`.
  - **Scratch policy:**
    - `mf.chkfile = None`, so no SCF chkfile is written.
    - During `compute`/`compute_more`, `pyscf.lib.param.TMPDIR` points to `scratch_dir` (created if missing), and the previous value is restored afterwards.

- [ ] **Step 1: Write the failing tests** in `tests/electronic/test_couplings.py`:

```python
import os

import numpy as np
import pytest

from gqteashmd.core.states import StateLabel
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.pyscf_backend import PySCFBackend

S0, S1, S2 = StateLabel(1, 0), StateLabel(1, 1), StateLabel(1, 2)


def _cas(make_settings):
    return make_settings(basis="6-31g")


def _tda(make_settings):
    return make_settings(method="tda", basis="sto-3g", functional="b3lyp", singlets=3,
                         casscf=None)


def _step(coords, seed=11, scale=0.01):
    return coords + np.random.default_rng(seed).normal(scale=scale, size=coords.shape)


@pytest.mark.parametrize("which", ["cas", "tda"])
def test_overlaps_at_equal_geometry_are_the_identity(which, ethylene_symbols, ethylene_bohr,
                                                     make_settings):
    settings = _cas(make_settings) if which == "cas" else _tda(make_settings)
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings)
    first = backend.compute(Request(ethylene_bohr, overlaps=True))
    assert first.overlaps is None and first.meta["overlaps"] == "no previous geometry"
    second = backend.compute(Request(ethylene_bohr, overlaps=True))
    np.testing.assert_allclose(second.overlaps, np.eye(len(backend.states)), atol=1e-6)
    assert second.meta["overlaps"] == "computed" and second.meta["root_flips"] == []
    assert backend.compute(Request(ethylene_bohr)).meta["overlaps"] == "not requested"


@pytest.mark.parametrize("which", ["cas", "tda"])
def test_small_step_overlaps_are_near_identity_with_positive_diagonal(which, ethylene_symbols,
                                                                      ethylene_bohr, make_settings):
    settings = _cas(make_settings) if which == "cas" else _tda(make_settings)
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings)
    backend.compute(Request(ethylene_bohr, overlaps=True))
    result = backend.compute(Request(_step(ethylene_bohr), overlaps=True))
    assert np.all(np.diag(result.overlaps) > 0.9)
    assert set(np.unique(result.phases)) <= {-1.0, 1.0}
    assert result.meta["overlap_min_singular_value"] > 0.9
    off = result.overlaps - np.diag(np.diag(result.overlaps))
    assert np.abs(off + off.T).max() < 0.02  # off-diagonal ~antisymmetric for small steps


def test_phases_keep_couplings_and_dipoles_continuous(ethylene_symbols, ethylene_bohr,
                                                      make_settings):
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), _cas(make_settings))
    req = dict(nacs=frozenset({(S0, S1)}), transition_dipoles=True, overlaps=True)
    a = backend.compute(Request(ethylene_bohr, **req))
    b = backend.compute(Request(_step(ethylene_bohr, scale=0.005), **req))
    da, db = a.nacs[(S0, S1)], b.nacs[(S0, S1)]
    assert np.abs(da - db).max() < 0.2 * np.abs(da + db).max()  # no sign jump between steps
    ma, mb = a.transition_dipoles[S1], b.transition_dipoles[S1]
    assert np.dot(ma, mb) > 0


def test_compute_more_reuses_the_last_geometry(ethylene_symbols, ethylene_bohr, make_settings,
                                               monkeypatch):
    settings = _cas(make_settings)
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings)
    with pytest.raises(ValueError, match="compute_more"):
        backend.compute_more(gradients=frozenset({S1}))
    backend.compute(Request(ethylene_bohr, gradients=frozenset({S0})))
    scf_calls = []
    original = PySCFBackend._run_scf
    monkeypatch.setattr(PySCFBackend, "_run_scf",
                        lambda self, *a, **k: scf_calls.append(1) or original(self, *a, **k))
    more = backend.compute_more(gradients=frozenset({S1}), nacs=frozenset({(S0, S1)}))
    assert scf_calls == [] and more.meta["compute_more"] is True
    fresh = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings).compute(
        Request(ethylene_bohr, gradients=frozenset({S1}), nacs=frozenset({(S0, S1)}))
    )
    assert np.abs(more.gradients[S1] - fresh.gradients[S1]).max() < 1e-7
    d_more, d_fresh = more.nacs[(S0, S1)], fresh.nacs[(S0, S1)]
    assert min(np.abs(d_more - d_fresh).max(), np.abs(d_more + d_fresh).max()) < 1e-7
    assert more.energies == backend.compute_more().energies


def test_compute_more_writes_to_the_log(tmp_path, ethylene_symbols, ethylene_bohr, make_settings):
    log = tmp_path / "pyscf.log"
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), _tda(make_settings), log_file=log)
    backend.compute(Request(ethylene_bohr))
    size = log.stat().st_size
    backend.compute_more(gradients=frozenset({S1}))
    assert log.stat().st_size > size


def test_scratch_policy(tmp_path, ethylene_symbols, ethylene_bohr, make_settings, monkeypatch):
    from pyscf import lib

    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    before = lib.param.TMPDIR
    scratch = tmp_path / "scratch"
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), _cas(make_settings),
                           scratch_dir=scratch)
    backend.compute(Request(ethylene_bohr, gradients=frozenset({S1})))
    assert os.listdir(work) == []
    assert scratch.is_dir() and lib.param.TMPDIR == before
    assert not any(p.name.startswith("tmp") for p in scratch.iterdir())  # no SCF chkfile
```

- [ ] **Step 2: Run them to verify they fail.**
  Run: `pytest tests/electronic/test_couplings.py -v`
  Expected: FAIL. Causes: overlaps not implemented (`ValueError`), an unknown `scratch_dir` keyword, and `compute_more` raising `NotImplementedError`.

- [ ] **Step 3: Implement** in `src/gqteashmd/electronic/pyscf_backend.py`.

  Add the imports (`import logging`, `from dataclasses import dataclass`, `truncate`, `state_overlaps`, `track`, `DeterminantExpansion`) and `_LOG = logging.getLogger(__name__)`.

  Add these dataclasses at module level:

```python
@dataclass(eq=False)
class _Snapshot:
    """What the next geometry needs from this one; save_state() writes exactly this."""

    coords: np.ndarray  # bohr
    mo_coeff: np.ndarray  # MOs the expansion refers to (CASSCF: mc.mo_coeff; else mf.mo_coeff)
    density: np.ndarray  # SCF density matrix, the next SCF guess
    expansion: DeterminantExpansion  # truncated and phase-aligned


@dataclass(eq=False)
class _Live:
    """PySCF objects of the last compute(), reused by compute_more() (never serialised)."""

    mol: Any
    mf: Any
    solver: Any  # CASSCF object, TD object or None
    energies: dict[StateLabel, float]
    phases: np.ndarray
```

  Constructor changes:
  - Add the keyword `scratch_dir: Path | None = None` and store it as `self._scratch_dir`.
  - Replace the three `_previous_*` attributes with `self._snapshot: _Snapshot | None = None` and `self._live: _Live | None = None`.

  Replace `_log` with an `_environment` context manager that covers threads, scratch and the log:

```python
    @contextmanager
    def _environment(self) -> Iterator[IO[str] | None]:
        from pyscf import lib

        if self._threads is not None:
            lib.num_threads(self._threads)
        old_tmpdir = lib.param.TMPDIR
        try:
            if self._scratch_dir is not None:
                try:
                    self._scratch_dir.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    raise wrap_failure(f"Preparing the scratch directory {self._scratch_dir}",
                                       exc) from exc
                lib.param.TMPDIR = str(self._scratch_dir)
            if self._log_file is None:
                yield None
            else:
                self._log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self._log_file, "a", encoding="utf-8") as fh:
                    yield fh
        finally:
            lib.param.TMPDIR = old_tmpdir
```

  In `_make_scf`, add `mf.chkfile = None` (the program keeps its own state; no SCF chkfile).

  Change `_run_scf(self, mol)` to `_run_scf(self, mol, previous_dm)`, using the argument in place of `self._previous_dm`. Remove the line `self._previous_dm = mf.make_rdm1()`.

  Rewrite `compute()`. Keep the existing request validation (states, atom count, NAC capability) and remove the temporary overlaps guard:

```python
    def compute(self, request: Request) -> ElectronicResult:
        self._validate(request)
        started = time.perf_counter()
        previous = self._snapshot
        with self._environment() as stdout:
            mol = self._build_mol(request.coords_bohr, stdout)
            mf, scf_rung = self._run_scf(mol, None if previous is None else previous.density)
            if self._settings.method == "casscf":
                prev_orbitals = None
                if previous is not None:
                    prev_orbitals = (previous.mo_coeff, self._build_mol(previous.coords, None))
                outcome = pyscf_casscf.run_casscf(mol, mf, self._settings, self._states,
                                                  prev_orbitals)
                solver, energies, spins = outcome.mc, outcome.energies, outcome.spin_squares
                mo_coeff = np.asarray(solver.mo_coeff)
                expansion = pyscf_casscf.casscf_expansion(solver)
                rung_meta: dict[str, Any] = {"casscf": outcome.rung}
            else:
                outcome_r = pyscf_response.run_response(mf, self._settings, self._states)
                solver, energies, spins = outcome_r.td, outcome_r.energies, {}
                mo_coeff = np.asarray(mf.mo_coeff)
                expansion = pyscf_response.response_expansion(mf, solver, len(self._states) - 1)
                rung_meta = {"td": outcome_r.rung}
            expansion = truncate(expansion, self._settings.overlap_norm_kept)
            phases = np.ones(len(self._states))
            overlaps = None
            if not request.overlaps:
                rung_meta["overlaps"] = "not requested"
            elif previous is None:
                rung_meta["overlaps"] = "no previous geometry"
            else:
                tracked = track(self._state_overlaps(previous, mol, mo_coeff, expansion))
                phases, overlaps = tracked.phases, tracked.overlaps
                expansion = expansion.with_signs(phases)
                rung_meta.update(
                    overlaps="computed",
                    overlap_min_singular_value=tracked.min_singular_value,
                    root_flips=[[i, j] for i, j in tracked.root_flips],
                )
                if tracked.min_singular_value < 0.5:
                    _LOG.warning(
                        "The state overlap matrix is far from unitary (smallest singular value "
                        "%.3f): an intruder state or too few computed states.",
                        tracked.min_singular_value,
                    )
            live = _Live(mol, mf, solver, energies, phases)
            gradients, nacs = self._derivatives(live, request.gradients, request.nacs)
            dipoles = self._dipoles(live, request.transition_dipoles)
        self._snapshot = _Snapshot(np.array(request.coords_bohr), mo_coeff,
                                   np.asarray(mf.make_rdm1()), expansion)
        self._live = live
        meta: dict[str, Any] = {
            "backend": "pyscf",
            "pyscf_version": self._version,
            "method": self._settings.method,
            "basis": self._settings.basis,
            "functional": self._settings.functional,
            "scf": scf_rung,
            "scf_energy_hartree": float(mf.e_tot),
            **rung_meta,
            "wall_time_s": time.perf_counter() - started,
        }
        return ElectronicResult(
            states=self._states,
            energies=energies,
            gradients=gradients,
            transition_dipoles=dipoles,
            spin_squares=spins,
            nacs=nacs,
            overlaps=overlaps,
            phases=None if overlaps is None else phases,
            meta=meta,
        )
```

  Add the helpers below. `_validate` holds the checks that used to sit at the top of `compute()`: unknown gradient states, unknown NAC states, NAC capability, and atom count.

```python
    def _validate(self, request: Request) -> None:
        self._check_labels(request.gradients, request.nacs)
        if len(request.coords_bohr) != len(self._molecule.symbols):
            raise ValueError(
                f"Got coordinates for {len(request.coords_bohr)} atoms; the molecule has "
                f"{len(self._molecule.symbols)} atoms."
            )

    def _check_labels(
        self,
        gradients: frozenset[StateLabel],
        nacs: frozenset[tuple[StateLabel, StateLabel]],
    ) -> None:
        unknown = sorted(str(s) for s in set(gradients) - set(self._states))
        if unknown:
            raise ValueError(f"Gradients requested for unknown state(s): {', '.join(unknown)}")
        unknown = sorted(str(s) for s in {s for pair in nacs for s in pair} - set(self._states))
        if unknown:
            raise ValueError(f"NACs requested for unknown state(s): {', '.join(unknown)}")
        if nacs and not self.capabilities.nacs:
            raise ValueError(
                f"Nonadiabatic coupling vectors are not available for method = "
                f"'{self._settings.method}'; request overlaps instead."
            )

    def _state_overlaps(
        self, previous: _Snapshot, mol: Any, mo_coeff: np.ndarray, expansion: DeterminantExpansion
    ) -> np.ndarray:
        from pyscf import gto

        try:
            previous_mol = self._build_mol(previous.coords, None)
            s_ao = gto.intor_cross("int1e_ovlp", previous_mol, mol)
        except Exception as exc:
            raise wrap_failure("The AO overlap between geometries", exc) from exc
        s_mo = previous.mo_coeff.T @ s_ao @ mo_coeff
        return state_overlaps(previous.expansion, expansion, s_mo)

    def _phase(self, live: _Live, label: StateLabel) -> float:
        return float(live.phases[self._states.index(label)])

    def _derivatives(
        self,
        live: _Live,
        gradients: frozenset[StateLabel],
        nacs: frozenset[tuple[StateLabel, StateLabel]],
    ) -> tuple[dict[StateLabel, np.ndarray], dict[tuple[StateLabel, StateLabel], np.ndarray]]:
        if self._settings.method == "casscf":
            grads = pyscf_casscf.casscf_gradients(live.solver, gradients)
            raw = pyscf_casscf.casscf_nacs(live.solver, nacs)
        else:
            grads = pyscf_response.response_gradients(live.mf, live.solver, gradients,
                                                      self._settings.method)
            raw = {}
        phased = {
            (a, b): d * self._phase(live, a) * self._phase(live, b) for (a, b), d in raw.items()
        }
        return grads, phased

    def _dipoles(self, live: _Live, wanted: bool) -> dict[StateLabel, np.ndarray]:
        if not wanted:
            return {}
        if self._settings.method == "casscf":
            raw = pyscf_casscf.casscf_dipoles(live.mol, live.solver, self._states)
        else:
            raw = pyscf_response.response_dipoles(live.solver, self._states, self._settings.method)
        p0 = self._phase(live, self._states[0])
        return {label: mu * p0 * self._phase(live, label) for label, mu in raw.items()}
```

  Replace the temporary `compute_more` with:

```python
    def compute_more(
        self,
        gradients: frozenset[StateLabel] = frozenset(),
        nacs: frozenset[tuple[StateLabel, StateLabel]] = frozenset(),
    ) -> ElectronicResult:
        live = self._live
        if live is None:
            raise ValueError("compute_more() needs a preceding compute() in this process.")
        self._check_labels(frozenset(gradients), frozenset(nacs))
        with self._environment() as stdout:
            out = stdout if stdout is not None else sys.stdout
            for obj in (live.mol, live.mf, live.solver):
                if obj is not None:
                    obj.stdout = out  # the file of the previous compute() is closed
            grads, phased = self._derivatives(live, frozenset(gradients), frozenset(nacs))
        return ElectronicResult(
            states=self._states,
            energies=dict(live.energies),
            gradients=grads,
            nacs=phased,
            meta={"backend": "pyscf", "method": self._settings.method, "compute_more": True},
        )
```

  Add `import sys`. Delete the old `_log` context manager and any `self._previous_*` references.

- [ ] **Step 4: Run the tests to verify they pass.**
  Run: `pytest tests/electronic -v -W error && mypy`
  Expected: all pass, including every P3a test. Report:
  - the measured overlap diagonals and minimum singular values;
  - the antisymmetry residual;
  - the `test_couplings.py` duration.

- [ ] **Step 5: Lint, full suite, commit**

```bash
ruff format src tests tools && ruff check src tests tools && pytest -m "not slow" -W error
git add src/gqteashmd/electronic/pyscf_backend.py tests/electronic/test_couplings.py
git commit -m "feat: track state phases with wavefunction overlaps and add compute_more"
```

---

### Task 8: Reproducible restarts: `save_state` / `load_state`

**Files:**
- Modify: `src/gqteashmd/electronic/pyscf_backend.py`
- Create: `tests/electronic/test_state.py`

**Interfaces:**
- Consumes: `_Snapshot` (Task 7), `write_bytes_atomic` (Task 3).
- Produces:
  - **`save_state(path)`** writes an `.npz` file containing `format`, `fingerprint`, `coords`, `mo_coeff`, `density`, `alpha`, `beta` and `coeffs`.
    - `fingerprint` is the JSON of the method, basis, functional, symbols, charge, multiplicity, singlets, triplets, casscf settings and overlap_norm_kept, with sorted keys.
    - The file is written atomically.
    - It raises `ValueError` before the first compute.
  - **`load_state(path)`** restores the snapshot and clears the live objects.
    - A fingerprint that differs from the current settings raises `ValueError`, naming the file.
    - A wrong `format` value also raises `ValueError`.
  - **Restart property:** a continued run and a restarted run give the same result at the next geometry (energies, gradients, NACs, overlaps and phases) to within 1e-9.

- [ ] **Step 1: Write the failing tests** in `tests/electronic/test_state.py`:

```python
import numpy as np
import pytest

from gqteashmd.core.states import StateLabel
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.pyscf_backend import PySCFBackend

S0, S1, S2 = StateLabel(1, 0), StateLabel(1, 1), StateLabel(1, 2)


def _step(coords):
    return coords + np.random.default_rng(5).normal(scale=0.02, size=coords.shape)


@pytest.mark.parametrize(
    "fields, request_kw",
    [
        ({"basis": "6-31g"}, {"gradients": frozenset({S1}), "nacs": frozenset({(S0, S1)})}),
        (
            {"method": "tda", "basis": "sto-3g", "functional": "b3lyp", "singlets": 3,
             "casscf": None},
            {"gradients": frozenset({S2}), "transition_dipoles": True},
        ),
    ],
)
def test_restart_reproduces_the_continued_run(tmp_path, ethylene_symbols, ethylene_bohr,
                                              make_settings, fields, request_kw):
    settings = make_settings(**fields)
    spec = MoleculeSpec(tuple(ethylene_symbols))
    nxt = Request(_step(ethylene_bohr), overlaps=True, **request_kw)

    continued = PySCFBackend(spec, settings)
    continued.compute(Request(ethylene_bohr, overlaps=True))
    a = continued.compute(nxt)

    first = PySCFBackend(spec, settings)
    first.compute(Request(ethylene_bohr, overlaps=True))
    first.save_state(tmp_path / "state.npz")
    restarted = PySCFBackend(spec, settings)
    restarted.load_state(tmp_path / "state.npz")
    b = restarted.compute(nxt)

    for label in a.states:
        assert abs(a.energies[label] - b.energies[label]) < 1e-9
    for label, g in a.gradients.items():
        assert np.abs(g - b.gradients[label]).max() < 1e-9
    for pair, d in a.nacs.items():
        assert np.abs(d - b.nacs[pair]).max() < 1e-9
    np.testing.assert_allclose(a.overlaps, b.overlaps, atol=1e-9)
    np.testing.assert_array_equal(a.phases, b.phases)
    for label, mu in a.transition_dipoles.items():
        assert np.abs(mu - b.transition_dipoles[label]).max() < 1e-9


def test_state_errors(tmp_path, ethylene_symbols, ethylene_bohr, make_settings):
    spec = MoleculeSpec(tuple(ethylene_symbols))
    backend = PySCFBackend(spec, make_settings(basis="sto-3g"))
    with pytest.raises(ValueError, match="compute"):
        backend.save_state(tmp_path / "s.npz")
    backend.compute(Request(ethylene_bohr))
    backend.save_state(tmp_path / "s.npz")
    other = PySCFBackend(spec, make_settings(basis="6-31g"))
    with pytest.raises(ValueError, match="different settings"):
        other.load_state(tmp_path / "s.npz")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s.npz"]  # atomic: no temp file left
```

- [ ] **Step 2: Run them to verify they fail.**
  Run: `pytest tests/electronic/test_state.py -v`
  Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement** in `pyscf_backend.py`. Add the imports `io` and `json`, plus `write_bytes_atomic`, and the constant `_STATE_FORMAT = 1`:

```python
    def _fingerprint(self) -> str:
        s = self._settings
        return json.dumps(
            {
                "method": s.method,
                "basis": s.basis,
                "functional": s.functional,
                "symbols": list(self._molecule.symbols),
                "charge": self._molecule.charge,
                "multiplicity": self._molecule.multiplicity,
                "singlets": s.singlets,
                "triplets": s.triplets,
                "casscf": None if s.casscf is None else s.casscf.model_dump(),
                "overlap_norm_kept": s.overlap_norm_kept,
            },
            sort_keys=True,
        )

    def save_state(self, path: Path) -> None:
        snap = self._snapshot
        if snap is None:
            raise ValueError("There is no electronic state to save: call compute() first.")
        buffer = io.BytesIO()
        np.savez(
            buffer,
            format=np.array(_STATE_FORMAT),
            fingerprint=np.array(self._fingerprint()),
            coords=snap.coords,
            mo_coeff=snap.mo_coeff,
            density=snap.density,
            alpha=snap.expansion.alpha,
            beta=snap.expansion.beta,
            coeffs=snap.expansion.coeffs,
        )
        write_bytes_atomic(Path(path), buffer.getvalue())

    def load_state(self, path: Path) -> None:
        with np.load(Path(path), allow_pickle=False) as data:
            if int(data["format"]) != _STATE_FORMAT:
                raise ValueError(f"{path}: unsupported electronic state format {int(data['format'])}.")
            if str(data["fingerprint"]) != self._fingerprint():
                raise ValueError(
                    f"The electronic state in {path} was saved with different settings or a "
                    "different molecule; it cannot be reused."
                )
            self._snapshot = _Snapshot(
                coords=np.array(data["coords"]),
                mo_coeff=np.array(data["mo_coeff"]),
                density=np.array(data["density"]),
                expansion=DeterminantExpansion(data["alpha"], data["beta"], data["coeffs"]),
            )
        self._live = None
```

- [ ] **Step 4: Run the tests to verify they pass.**
  Run: `pytest tests/electronic -v -W error && mypy`
  Expected: all pass. If a restarted value differs from the continued one by more than 1e-9, find the hidden state that is not in the snapshot. Do not loosen the tolerance. Report the cause.

- [ ] **Step 5: Lint, full suite, commit**

```bash
ruff format src tests tools && ruff check src tests tools && pytest -m "not slow" -W error
git add src/gqteashmd/electronic/pyscf_backend.py tests/electronic/test_state.py
git commit -m "feat: save and restore the backend reference state for reproducible restarts"
```

---

### Task 9: NACs against finite-difference overlaps (slow)

**Files:**
- Create: `tests/electronic/test_couplings_fd.py`

**Interfaces:**
- Consumes the backend API only.
- The comparison is d_01(R) against [S_01(R, R+h) − S_01(R, R−h)]/(2h), where S comes from `Request(..., overlaps=True)` on a fresh backend that first computed R.
- The tolerance is 2·10⁻⁴ absolute (Facts §3: AO-derivative convention). This task also records how the overlap-based time-derivative coupling compares, for the documentation.

- [ ] **Step 1: Write the test:**

```python
import numpy as np
import pytest

from gqteashmd.core.states import StateLabel
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.pyscf_backend import PySCFBackend

S0, S1 = StateLabel(1, 0), StateLabel(1, 1)
H = 1e-4  # bohr


@pytest.mark.slow
def test_casscf_nacs_match_finite_difference_overlaps(ethylene_symbols, ethylene_bohr,
                                                      make_settings):
    settings = make_settings(basis="6-31g", scf_conv_tol=1e-12,
                             casscf={"ncas": 2, "nelecas": 2, "conv_tol": 1e-10,
                                     "conv_tol_grad": 1e-6})
    spec = MoleculeSpec(tuple(ethylene_symbols))
    reference = PySCFBackend(spec, settings).compute(
        Request(ethylene_bohr, nacs=frozenset({(S0, S1)}))
    )
    d = reference.nacs[(S0, S1)]
    worst = 0.0
    for atom, axis in ((0, 0), (2, 1), (4, 0)):
        s01 = []
        for sign in (+1, -1):
            backend = PySCFBackend(spec, settings)
            backend.compute(Request(ethylene_bohr))
            displaced = ethylene_bohr.copy()
            displaced[atom, axis] += sign * H
            s01.append(backend.compute(Request(displaced, overlaps=True)).overlaps[0, 1])
        fd = (s01[0] - s01[1]) / (2 * H)
        worst = max(worst, abs(fd - d[atom, axis]))
        assert abs(fd - d[atom, axis]) < 2e-4, (atom, axis, fd, d[atom, axis])
    print(f"max |FD overlap - NAC| = {worst:.2e} 1/bohr")
```

  **Phases:** the reference NAC comes from a backend whose phases are its own raw phases. Each FD backend's first compute at R gives the same raw CI phases, because it is deterministic from the same SCF guess, and its second call aligns to those. The signs are therefore consistent. If the measured FD comes out with the opposite sign everywhere, report it; it would mean the raw phases are not deterministic. Do not take an absolute value to hide that.

- [ ] **Step 2: Run.**
  Run: `pytest tests/electronic/test_couplings_fd.py -v -m slow -W error -s`
  Expected: 1 passed. Report the printed maximum difference and the duration.

- [ ] **Step 3: Lint, commit**

```bash
ruff format src tests tools && ruff check src tests tools
git add tests/electronic/test_couplings_fd.py
git commit -m "test: validate SA-CASSCF NACs against finite-difference wavefunction overlaps"
```

---

### Task 10: Documentation current with P3b

**Files:**
- Modify: `CLAUDE.md`, `USER_MANUAL.md`, `docs/superpowers/plans/2026-11-06-gqteashmd-master-plan.md`

**Interfaces:**
- Consumes: the final code of Tasks 1–9. Read the code, not this plan, because fix rounds may change details.

- [ ] **Step 1: USER_MANUAL.md.** Keep [IMPLEMENTED] and [PLANNED] precise.
  - **§4:** add `electronic.overlap_norm_kept` (type, default 0.9999, range 0.9 < x ≤ 1, meaning).
  - **New subsection in §8.6, [IMPLEMENTED].** Define every symbol and unit.
    - The NAC definition $\mathbf d_{IJ} = \langle\Psi_I|\nabla_{\mathbf R}\Psi_J\rangle$ (1/bohr). State that it is antisymmetric and that it includes the CSF term. Explain the AO-derivative convention and the measured ~10⁻⁴ difference from finite-difference overlaps. Give the agreement with SHARC to ~10⁻⁷ up to sign, using the measured value.
    - The determinant expansion and Löwdin's rule:
      $S_{IJ} = \sum_{kl} c^{\mathrm{prev}}_{Ik} c^{\mathrm{cur}}_{Jl} \det(\mathbf s^{\alpha}_{kl})\det(\mathbf s^{\beta}_{kl})$, with $\mathbf s = \mathbf C_{\mathrm{prev}}^{T}\mathbf S_{\mathrm{AO}}(\mathbf R_{\mathrm{prev}},\mathbf R_{\mathrm{cur}})\mathbf C_{\mathrm{cur}}$.
    - Truncation by `overlap_norm_kept`, and renormalisation.
    - The CASSCF expansion (core plus FCI strings).
    - The TDA singlet expansion with $\sum X^2 = 1/2$, and the X-only approximation for TDDFT/TDHF, with its measured non-orthogonality.
    - Phase alignment and root-flip matching (assignment maximising $\sum_j|S_{a(j)j}|$), the smallest-singular-value diagnostic with its 0.5 warning, and Löwdin orthonormalisation (provided for P7; not applied by the backend).
    - How the phases change the outputs: $\mathbf d_{IJ}\to p_Ip_J\mathbf d_{IJ}$, $\boldsymbol\mu_{0I}\to p_0p_I\boldsymbol\mu_{0I}$.
    - The retry ladders for CASSCF and TDA/TDDFT.
    - `compute_more`.
    - Restarts via the saved state.
    - The scratch policy (no chkfile; TMPDIR).
  - **§8.6 validation table:** add the measured NAC numbers (vs SHARC, vs FD) and the overlap checks.
  - **Planned items** that must stay marked [PLANNED]: NAC and overlap use in dynamics (P7), triplets, the Hessian (P4), and the hardening sweep (P3c).
- [ ] **Step 2: CLAUDE.md.**
  - **Status:** P3b done on branch `feat/p3b-couplings`, with the test counts and times.
  - **Architecture:** `overlaps/`, plus `pyscf_errors.py`, `pyscf_casscf.py` and `pyscf_response.py`.
  - **Rules:**
    - PySCF is imported lazily only in `electronic/pyscf_*.py`, and `overlaps/` never imports it.
    - The backend's only cross-geometry state is `_Snapshot`, which `save_state` and `load_state` write and read.
    - Phases are tracked only when `overlaps=True`, so dynamics must request overlaps on every step.
    - NAC pairs are I < J.
    - `compute_more` uses the cached objects.
  - **Remaining work:** P3c (the hardening sweep), then P4.
  - **Known limitations:**
    - the overlap engine's cost grows with the number of determinants kept;
    - TDDFT overlaps use only X;
    - TDA/TDDFT has no analytic NACs.
- [ ] **Step 3: Master plan.**
  - §9: mark P3b delivered and add a P3c row for the hardening sweep.
  - §2.2: add the layout of `overlaps/` and the three `pyscf_*` modules.
  - §6.2: add the CASSCF and TD ladders.
- [ ] **Step 4: Verify.** Run every command or example you added to the manual in `mktemp -d /tmp/gq-doc-XXXX`, with `conda activate gqteashmd-dev` first. Remove the directory afterwards.
- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md USER_MANUAL.md docs/superpowers/plans/2026-11-06-gqteashmd-master-plan.md
git commit -m "docs: document couplings, overlaps, phase tracking and restarts (P3b)"
```

---

## Self-review notes

- **Spec coverage:**

  | Spec item | Where it is covered |
  |---|---|
  | Strategy §2.3 contract: `capabilities`, `nacdr`, overlaps, phases, `compute_more`, `save_state`/`load_state` | Tasks 3, 5, 7, 8 |
  | §3.1 SA-CASSCF NACs | Task 5 |
  | §3.1 TDDFT NACs | Not available in 2.14 (Fact 4); overlap route |
  | §3.3 CIS/TDA overlaps (own NumPy code), truncation, consistent normalisation | Tasks 1, 6 |
  | §3.3 phase alignment, root-flip detection, Löwdin, small-singular-value log | Tasks 2, 7 |
  | §3.5 conformance: overlaps at Δt→0 ≈ identity, phase consistency | Task 7 |
  | §5.3 NAC tolerance vs SHARC, up to sign | Tasks 4, 5 |
  | §5.1 NACs vs FD overlaps | Task 9 |
  | Master plan P3b row: scratch/chkfile | Task 7 |
  | Master plan P3b row: CASSCF/TDDFT retry ladders | Tasks 5, 6 |
  | Master plan P3b row: hardening sweep | Moved to P3c by ruling (Global Constraints) |
  | Documentation | Task 10 |

- **Type consistency:**
  - `DeterminantExpansion(alpha, beta, coeffs)`, `truncate(expansion, norm_kept)` and `state_overlaps(bra, ket, s_mo)` are used identically in Tasks 1, 5, 6, 7 and 8.
  - `track(s) -> TrackedOverlaps(overlaps, phases, assignment, min_singular_value)` is used in Task 7.
  - `Request(..., nacs, overlaps)` and `ElectronicResult(..., nacs, overlaps, phases)` are used in Tasks 3–9.
  - `compute_more(gradients, nacs)` has the same signature in the ABC (Task 3) and in PySCFBackend (Task 7).
  - `run_casscf(mol, mf, settings, states, previous)` and `run_response(mf, settings, states)` are used in Task 7 as defined in Tasks 5 and 6.
  - `_run_scf(mol, previous_dm)` is changed in Task 7, and the P3a SCF-ladder test (`_make_scf` monkeypatch) still applies.
