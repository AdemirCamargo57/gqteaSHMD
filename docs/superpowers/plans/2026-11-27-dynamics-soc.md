# The `dynamics` stage with triplets and spin-orbit coupling: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run surface hopping with triplets and spin-orbit coupling in the `dynamics` stage on the PySCF backend, analyse the spin-free populations and singlet-triplet transitions in the `analysis` stage, and validate the driver against SHARC 4.1 by a swap test.

**Architecture:** The backend stays spin-free; a new pure-NumPy module `trajectory/spin_expansion.py` expands its energies, gradients and block overlaps into the core's spin-expanded basis; `BackendModel` passes the backend's spin-orbit Hamiltonian as the snapshot's complex MCH Hamiltonian, so the core's existing diagonal-basis path (P7a-2) does the dynamics. The step log and `output.h5` gain the MCH quantities for triplet runs only; the analysis reads spin-free states.

**Tech Stack:** Python 3.12, NumPy, PySCF 2.14 (lazy), h5py (lazy), pydantic 2, pytest; SHARC 4.1 executed as a black box.

**Spec:** `docs/superpowers/specs/2026-11-27-dynamics-soc-design.md` (owner-approved 2026-11-27; owner decisions 1-8 in its section 0).

## Global Constraints

- Run code, tests and git inside WSL: `wsl -d OracleLinux_9_5 -- bash -lc 'cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && conda activate gqteashmd-dev && <command>'` (written `W '<command>'` below). Edit files with LF endings. Never `--no-verify`; commit messages in a file with `git commit -F`, each ending with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- One commit per task, conventional subjects (`feat:`, `fix:`, `test:`, `docs:`).
- ruff (line length 100, rules E, F, W, I, B, UP, SIM) and `mypy src` clean.
- PySCF, h5py, SciPy, geomeTRIC are imported inside functions only; `import gqteashmd.cli` must not import pyscf, scipy, h5py or geometric.
- Dependency rule: `dynamics/` imports nothing from `trajectory/`, `electronic/`, `stages/`, `config/`; `trajectory/` may import `dynamics/`, `electronic/base.py`, `overlaps/`, `core/`, `errors`, `util/`; `analysis/` imports only `core/`, `gqteashmd.__version__` and itself.
- No test compares BLAS/LAPACK results bit for bit or relies on an eigenvector's sign; check new tests with `OPENBLAS_CORETYPE=SandyBridge`. A test that checks a process ended counts a zombie as ended.
- Clean room: SHARC is executed only (from `tools/sharc_compare/`), its formats from the public manual and observed output; never read, copy or adapt SHARC source; never imported by `gqteashmd`.
- **Without triplets nothing changes:** the requests, U (`None`), `steps.log`, `output.h5` (`schema_version` 1, today's bytes), the analysis, every fingerprint and every stage version (`DynamicsStage.version` 3 and `AnalysisStage.version` 4 stay). The existing tests keep their assertions.
- Pre-registered bounds (spec 5.1), never loosened: swap (a) and (b) |dq| <= 1e-5 bohr, |dv| <= 1e-6 a.u., |dE| <= 1e-6 Eh, |dP| <= 1e-5, per step and component; (a) steps 0-40; (b) to the first singlet-triplet hop + 20 steps, at most 200 steps, same active spin-free state. A miss is BLOCKED with the numbers.
- Do not edit USER_MANUAL.md, CLAUDE.md or CHANGELOG.md before Task 12.

## Conventions (binding for every task)

- **Spin-free states** (the backend's `states`, the stage's labels, `output.h5` `state_labels`): `S0 ... S(n-1), T1 ... Tm` (n singlets including S0, m triplets).
- **Expanded (MCH) basis of the core:** size `N = n + 3m`; the expanded index of S_I is I, of (T_J, M) is `n + M_index * m + (J - 1)` with M_index 0, 1, 2 for Ms = -1, 0, +1 (`dynamics/spin.py:expanded_states((n, 0, m))`, labels `T1(-1)` ...). The backend's `ElectronicResult.hamiltonian` is in this order already (sub-project 2).
- **Diagonal basis:** the core's eigenbasis of H, tracked by U (`dynamics/representation.py:diagonal_basis`); `state.active` and `state.c` are diagonal; `mch_state(U, active)` is the expanded index with the largest |U[:, active]|; its spin-free state is `spin_expansion.spin_free_index(n, m, k)`.
- **The "active spin-free state"** of a record = `spin_free_index(n, m, mch_state(U', active))`.

## Plan rulings (binding where they refine the spec)

- **PR1** `output.h5` schema 2: `hop_from`/`hop_to` hold the **spin-free** states before and after an accepted hop (the active spin-free state of the previous and of this record; -1 when no hop), so `analysis/hops.py` works on spin-free labels unchanged; the diagonal indices are `hop_from_diag`/`hop_to_diag`. (Spec 4.2 said "the diagonal `hop_from`/`hop_to`" beside the MCH data; the spec erratum records this naming.)
- **PR2** `BackendModel` with triplets: `labels` = the spin-free labels (as today: the backend's `states`); `n_states` = N (the core's state count, used by the step log for the diagonal columns); new attributes `n_singlets`, `n_triplets`, `n_spin_free` = n + m; `u: np.ndarray | None` set by the runner before each step (like `active`).
- **PR3** The first evaluation (`previous is None`, the start of a trajectory) requests every spin-free gradient whatever `eselect_eV` (the diagonal active state is not known before U0).
- **PR4** `ModelBackend` over a model with `state_counts == (n, 0, m)` serves the spin-free view: states `S0..S(n-1), T1..Tm`; energies, gradients, overlaps of a triplet from its Ms = -1 component (M_index 0); `hamiltonian` (with `Request(soc=True)`) the model's expanded H; `Capabilities.soc = m > 0`. `soc=True` with m = 0 stays a `ValueError`.
- **PR5** The canned spin-orbit pipeline uses `tests/data/lvc/lvc_soc.template` (2 singlets + 2 triplets) through the environment variable `GQTEASHMD_CANNED_LVC_SOC=1` (`canned_lvc.canned_model` builds `LVCModel.from_template(..., spin_orbit=True)` then).
- **PR6** The progress file (`progress.json`), the summary rows and log messages name the active **spin-free** label for a triplet run.
- **PR7** The dropped-term report (spec 5.2) is `tools/validation/ngt_term_report.py` (a report script: JSON out, nothing asserted).
- **PR8** `stop_after_s0_fs` counts a record as in S0 when its active spin-free state is S0.

## File structure

| File | Responsibility |
|---|---|
| `src/gqteashmd/trajectory/spin_expansion.py` (new) | the expanded order; expand energies, gradients, overlaps; spin-free index and populations |
| `src/gqteashmd/config/schema.py` | `DynamicsStageSection.gradient_transformation`, `eselect_eV`; the triplet rules in `ShmdConfig._consistent` |
| `src/gqteashmd/stages/dynamics.py` | `check_config` (refusal replaced by the rules), `run_settings`, `fingerprint_inputs`, `run_task` |
| `src/gqteashmd/testing/fake_backend.py`, `testing/canned_lvc.py` | the spin-free view of an LVC spin-orbit model (PR4, PR5) |
| `src/gqteashmd/trajectory/model.py` | `BackendModel` with triplets, the `eselect` window, the refresh |
| `src/gqteashmd/trajectory/store.py`, `trajectory/runner.py` | the MCH log columns and `output.h5` schema 2 (triplet runs only) |
| `src/gqteashmd/analysis/hops.py`, `analysis/report.py`, `stages/analysis.py` | spin-free transitions by kind, the ISC block, the fit default |
| `tools/sharc_compare/run_sharc_soc_swap.py` (new) | the swap harness with spin-orbit coupling |
| `tools/validation/ngt_term_report.py` (new) | the dropped-term report |

---

### Task 1: facts (a measurement, no code in `src/`)

**Files:** none committed except, if needed, test fixtures under `tests/data/sharc/soc/` (raw `QM.out`/`output.dat` excerpts of the probe run).

- [ ] **Step 1: SHARC's `QM.out` with spin-orbit coupling.** From the public SHARC 4.1 manual (sections on `QM.out` and the `SOC` request; read the manual only) and a probe: write a 2S+2T `QM.out` with `tools/sharc_compare/run_sharc_lvc.py:format_qm_out` from the P7a-2 LVC spin-orbit model (it already writes a complex H with SOC, expanded gradients and overlaps) and run `sharc.x` for 3 steps with `coupling overlap`, `nogradcorrect`, `surf diagonal`, `ekincorrect parallel_vel`. Record: the state ordering SHARC expects (singlets, then each triplet's Ms components; confirm it equals our canonical order, or note the permutation), the requests SHARC sends in `QM.in` for this setup (`H`, `GRAD`, `OVERLAP`, `SOC`?), whether it asks for every gradient or a subset, the `output.dat` blocks needed for the comparison (positions, velocities, `U`, coefficients, `E_tot`, active state). If the order differs from ours, the harness permutes; nothing in `src/` changes.
- [ ] **Step 2: time per step for H2CS/6-31G**, SA-CASSCF(4,3) 2S+2T and CIS 3S+3T (`td_conv_tol` 1e-7): one `PySCFBackend.compute(Request(..., gradients=all, overlaps=True, soc=True))` at the committed geometry and at a 0.01 bohr displaced one (the second with a previous geometry), 8 threads; record the SOMF share (time `soc.soc_operator` alone).
- [ ] **Step 3: the start sample for swap (b)** (spec 5.1): project seed 2029; Wigner samples at T = 0 of H2CS RHF/6-31G\* frequencies, generated as the `sampling` stage would (a throwaway project through `sampling`, or `vib.wigner.sample_wigner` with `task_rng(2029, "sampling", 0)`); initial state S1 for both methods. Run SHARC-swap trajectories (sharc.x driving our backend through the provider of Task 9's design, or the Step 1 probe harness extended by hand in a throwaway script) on samples 0, 1, 2, ... until one has a singlet-triplet hop within 200 steps of 0.5 fs; record the index per method. If none of the first 20 has one, stop and report BLOCKED for (b) (spec 5.1).
- [ ] **Step 4:** write `task-1-report.md` in the ledger with every fact and number. Commit only fixtures (if any): `test: add the SHARC spin-orbit QM.out probe fixtures`.

### Task 2: `trajectory/spin_expansion.py`

**Files:**
- Create: `src/gqteashmd/trajectory/spin_expansion.py`
- Test: `tests/trajectory/test_spin_expansion.py`

**Interfaces:**
- Produces: `expanded_size(n: int, m: int) -> int`; `expanded_index(n: int, m: int, spin_free: int, m_index: int = 0) -> int`; `spin_free_index(n: int, m: int, k: int) -> int`; `expand_energies(e: np.ndarray, n: int, m: int) -> np.ndarray` (shape (N,)); `expand_gradients(g: np.ndarray, n: int, m: int) -> np.ndarray` ((n+m, f) -> (N, f)); `expand_overlaps(s: np.ndarray, n: int, m: int) -> np.ndarray` ((n+m)^2 -> N^2); `spin_free_populations(p: np.ndarray, n: int, m: int) -> np.ndarray` ((..., N) -> (..., n+m)).

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from gqteashmd.dynamics.spin import expanded_states
from gqteashmd.trajectory import spin_expansion as sx


@pytest.mark.parametrize(("n", "m"), [(2, 0), (2, 1), (3, 3), (1, 2)])
def test_indices_follow_the_canonical_order(n, m):
    labels = [str(s) for s in expanded_states((n, 0, m))]
    assert sx.expanded_size(n, m) == len(labels)
    for j in range(m):
        for mi, ms in enumerate(("-1", "0", "+1")):
            k = sx.expanded_index(n, m, n + j, mi)
            assert labels[k] == f"T{j + 1}({ms})"
            assert sx.spin_free_index(n, m, k) == n + j
    for i in range(n):
        assert sx.expanded_index(n, m, i) == i and sx.spin_free_index(n, m, i) == i


def test_energies_and_gradients_repeat_each_triplet_three_times():
    e = np.array([0.0, 0.1, 0.2, 0.3])  # S0 S1 T1 T2
    g = np.arange(8.0).reshape(4, 2)
    ee, gg = sx.expand_energies(e, 2, 2), sx.expand_gradients(g, 2, 2)
    assert ee.tolist() == [0.0, 0.1, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3]
    assert gg.shape == (8, 2) and np.array_equal(gg[[2, 4, 6]], np.repeat(g[[2]], 3, axis=0))


def test_overlaps_are_block_diagonal_with_kron_triplet_blocks():
    rng = np.random.default_rng(0)
    s = np.zeros((4, 4))
    s[:2, :2], s[2:, 2:] = rng.normal(size=(2, 2)), rng.normal(size=(2, 2))
    big = sx.expand_overlaps(s, 2, 2)
    assert np.array_equal(big[:2, :2], s[:2, :2])
    assert np.array_equal(big[2:, 2:], np.kron(np.eye(3), s[2:, 2:]))
    assert np.all(big[:2, 2:] == 0) and np.all(big[2:, :2] == 0)


def test_overlaps_refuse_nonzero_singlet_triplet_blocks():
    s = np.eye(4)
    s[0, 3] = 1e-3
    with pytest.raises(ValueError, match="singlet-triplet"):
        sx.expand_overlaps(s, 2, 2)


def test_spin_free_populations_sum_over_ms():
    p = np.array([[0.1, 0.2, 0.05, 0.1, 0.05, 0.2, 0.1, 0.2]])
    assert np.allclose(sx.spin_free_populations(p, 2, 2), [[0.1, 0.2, 0.2, 0.5]])


def test_no_triplets_is_the_identity():
    e, g, s = np.array([0.0, 0.1]), np.ones((2, 3)), np.eye(2)
    assert np.array_equal(sx.expand_energies(e, 2, 0), e)
    assert np.array_equal(sx.expand_gradients(g, 2, 0), g)
    assert np.array_equal(sx.expand_overlaps(s, 2, 0), s)
```

(the singlet-triplet tolerance of `expand_overlaps` is exact: the backend's cross blocks are exact zeros, sub-project 1).

- [ ] **Step 2:** `W 'pytest tests/trajectory/test_spin_expansion.py -q -W error'` → FAIL (ModuleNotFoundError).
- [ ] **Step 3: Implement**

```python
"""The core's spin-expanded (MCH) basis for spin-free backend data (dynamics with SOC, spec 3.1).

Spin-free states S0..S(n-1), T1..Tm; expanded order S0..S(n-1), then T1..Tm with Ms = -1, then
0, then +1 (`dynamics/spin.py`): index of S_I is I, of (T_J, M) n + M_index m + (J - 1).
"""

from __future__ import annotations

import numpy as np

M_COMPONENTS = 3


def expanded_size(n: int, m: int) -> int:
    return int(n) + M_COMPONENTS * int(m)


def expanded_index(n: int, m: int, spin_free: int, m_index: int = 0) -> int:
    if spin_free < n:
        return int(spin_free)
    return int(n + m_index * m + (spin_free - n))


def spin_free_index(n: int, m: int, k: int) -> int:
    return int(k) if k < n else int(n + (k - n) % m)


def _rows(n: int, m: int) -> np.ndarray:
    """Spin-free row of every expanded index."""
    return np.array([spin_free_index(n, m, k) for k in range(expanded_size(n, m))], dtype=int)


def expand_energies(e: np.ndarray, n: int, m: int) -> np.ndarray:
    return np.asarray(e, dtype=float)[_rows(n, m)]


def expand_gradients(g: np.ndarray, n: int, m: int) -> np.ndarray:
    return np.asarray(g, dtype=float)[_rows(n, m)]


def expand_overlaps(s: np.ndarray, n: int, m: int) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    if m and (np.any(s[:n, n:] != 0) or np.any(s[n:, :n] != 0)):
        raise ValueError("overlaps have nonzero singlet-triplet blocks")
    out = np.zeros((expanded_size(n, m),) * 2)
    out[:n, :n] = s[:n, :n]
    if m:
        out[n:, n:] = np.kron(np.eye(M_COMPONENTS), s[n:, n:])
    return out


def spin_free_populations(p: np.ndarray, n: int, m: int) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.zeros(p.shape[:-1] + (n + m,))
    np.add.at(out, (..., _rows(n, m)), p)
    return out
```

- [ ] **Step 4:** run → PASS (also SandyBridge); ruff, mypy.
- [ ] **Step 5:** commit `feat: add the spin expansion of spin-free electronic data`.

### Task 3: configuration

**Files:**
- Modify: `src/gqteashmd/config/schema.py` (`DynamicsStageSection`: `gradient_transformation: Literal["ngt", "none"] = "ngt"`, `eselect_eV: float | None = Field(default=None, gt=0, allow_inf_nan=False)`; `ShmdConfig._consistent`: the triplet rules), `src/gqteashmd/stages/dynamics.py` (`check_config`: replace the triplet refusal by the effective-charge element check; `run_settings`: pass `gradient_transformation` and `eselect_Eh = eselect_eV / HARTREE_IN_EV`; `fingerprint_inputs`: pop the two keys at their defaults), `src/gqteashmd/cli.py` `init` template (commented lines for the two keys)
- Test: `tests/config/test_dynamics_section.py`, `tests/stages/test_dynamics.py`

**Interfaces:** Produces `DynamicsStageSection.gradient_transformation`, `.eselect_eV`; `run_settings(section).options.gradient_transformation`, `.eselect_Eh`.

- [ ] **Step 1: Failing tests:**
  - defaults `"ngt"` and `None`; `eselect_eV` 0, negative, inf refused; `"none"` accepted;
  - with `[electronic] triplets = 2` (casscf, ncas 3, nelecas 4 or tda): `coupling = "nacdr"`, `gradient_transformation = "ngt"` (also the default), `hopping = "fssh"`, `ekin_correction = "parallel_nac"` each refused, **all at once** in one `ConfigError` (one test with all four), each message naming its key and `triplets`; the valid combination (`overlap`, `none`, `sharc`, `parallel_vel`) accepted;
  - `DynamicsStage.check_config` with triplets no longer contains "is planned"; with `soc_operator = "effective_charge"` returns a problem naming N for a molecule with nitrogen (a temporary NH3 `xyz` file in `tmp_path`) and none for `tests/data/thioformaldehyde.xyz` (H, C, S);
  - `fingerprint_inputs` of a default section has neither key and equals the dict of a section built without them; with `gradient_transformation = "none"` or `eselect_eV = 0.5` it contains them;
  - `run_settings` maps `eselect_eV = 0.5` to `eselect_Eh = 0.5 / HARTREE_IN_EV` and `None` to `None`.
- [ ] **Step 2:** see them fail. **Step 3:** implement (the cross-field rules in `ShmdConfig._consistent`, which already reports every violation joined by `; `; the element check in `check_config` reads the molecule's symbols from `config.molecule` via `chem.xyz.parse_xyz` as `prepare` does and calls `electronic.soc.require_effective_charges`, which imports no PySCF). **Step 4:** `W 'pytest tests/config tests/stages/test_dynamics.py tests/test_cli.py -q -W error -m "not slow and not pyscf"'`, ruff, mypy. **Step 5:** commit `feat: accept triplets in the dynamics input with the spin-orbit rules`.

### Task 4: the fake backend with spin-orbit coupling and `BackendModel` with triplets

**Files:**
- Modify: `src/gqteashmd/testing/fake_backend.py` (PR4), `src/gqteashmd/testing/canned_lvc.py` (PR5), `src/gqteashmd/trajectory/model.py`
- Test: `tests/trajectory/test_fake_backend.py`, `tests/trajectory/test_model.py`

**Interfaces:**
- Consumes: Task 2's `spin_expansion`; `dynamics.representation.diagonal_basis`, `diagonal_energies`, `selection_mask`; `electronic.base.Request(soc=)`, `Capabilities.soc`, `ElectronicResult.hamiltonian`.
- Produces: `BackendModel(backend, masses, *, coupling, eselect_Eh=None)`; attributes `labels`, `n_states` (= N), `n_singlets`, `n_triplets`, `n_spin_free`, `active`, `u`, `state_counts` (`(n, 0, m)` with triplets, `(n,)` without); `evaluate(q, previous)`; `refresh_active_gradient(state)`; `active_spin_free(u, active) -> int` (the spin-free state of `mch_state(u, active)`).

- [ ] **Step 1: Failing tests** (fast tier; model = `LVCModel.from_template(tests/data/lvc/lvc_soc.template, spin_orbit=True)`, 2S+2T, at its reference geometry and a displaced one):
  - `ModelBackend` over it: `states` are `S0 S1 T1 T2`; `capabilities.soc` True; a `compute` returns 4 energies, spin-free gradients, a 4x4 overlap with exact-zero cross blocks, and with `soc=True` a `hamiltonian` equal to the model's own snapshot `hamiltonian`; over `lvc_singlets.template` `soc=True` still raises;
  - `BackendModel` round trip: `evaluate(q, None)` and `evaluate(q1, snap0)` give snapshots whose `energies`, `gradients` and `hamiltonian` equal the model's own `evaluate` within 1e-12, and whose `overlaps` equal the model's overlaps within 1e-10 (Lowdin of an orthonormal-to-round-off matrix);
  - `run_trajectory` (the core's driver) for 30 steps with the LVC model directly and with `BackendModel(ModelBackend(model))`, same `ReplaySource`, `gradient_transformation = "none"`, `coupling = "overlap"`: identical `active`, and q, v, c within 1e-10;
  - the request: without `eselect_Eh` every spin-free gradient is requested in `compute` and `compute_more` is never called (a spy backend); with `eselect_Eh` set and a previous snapshot, the gradients fetched are exactly the spin-free states of `selection_mask(snap.energies, E_diag'[active], eselect)` with `E_diag'` from `diagonal_basis(H', model.u)` (equal to the core's own mask: assert by recomputing it as the driver does); at `previous is None` every gradient (PR3);
  - `refresh_active_gradient` after a hop with `eselect_Eh`: fetches the missing gradients of the new mask; a state kept by the mask without a gradient after the refresh raises `ElectronicStructureError` (retryable False) — provoke it with a spy backend whose `compute_more` drops one;
  - a missing `hamiltonian` with triplets is a non-retryable `ElectronicStructureError`; a non-finite one too;
  - `active_spin_free`: for U = identity and active = index of `T2(+1)` returns 3;
  - without triplets (`lvc_singlets.template`): requests and snapshots are byte-for-byte the same as before this task (compare against `git stash`-free expectations: the existing `tests/trajectory/test_model.py` stays green unmodified).
- [ ] **Step 2:** see them fail. **Step 3: implement:**
  - `ModelBackend.__init__`: `counts = tuple(model.state_counts)`; `n, m = counts[0], (counts[2] if len(counts) == 3 else 0)`; states `states_from_counts(n, m)` (the existing helper takes the triplet count as its second argument; check its signature); `compute` evaluates the model as today and, with m > 0, builds the spin-free view: rows `[0..n-1] + [expanded_index(n, m, n + j, 0) for j]` of energies and gradients and the same rows/columns of overlaps; NACs between spin-free states from the same rows; `hamiltonian` the model's expanded H when `request.soc`;
  - `canned_lvc.canned_model()`: `spin_orbit = os.environ.get("GQTEASHMD_CANNED_LVC_SOC") == "1"`;
  - `BackendModel`: with triplets refuse `coupling != "overlap"` (ValueError naming it); request `soc=True`; build the snapshot from `spin_expansion` (energies, gradients with NaN rows for spin-free states without a gradient, then expanded; `nacs` zeros (N, N, f); `overlaps` = `expand_overlaps(lowdin_orthonormalize(raw))` — Lowdin per spin-free matrix is block-preserving; `hamiltonian` = `result.hamiltonian`); `_check_finite` over the expanded data and H; the `eselect` window per PR3 and spec 3.3; `refresh_active_gradient` generalised.
- [ ] **Step 4:** `W 'pytest tests/trajectory tests/dynamics -q -W error -m "not slow"'` (the golden test unchanged and green), SandyBridge for the new tests, ruff, mypy. **Step 5:** commit `feat: run the dynamics adapter with triplets and spin-orbit coupling`.

### Task 5: the step log and `output.h5` schema 2

**Files:**
- Modify: `src/gqteashmd/trajectory/store.py` (`log_dtype(n_states, n_dof, *, n_spin_free=None)`; `write_output_h5(..., n_singlets=None, n_triplets=None)`), `src/gqteashmd/trajectory/runner.py` (`_append`, `_write_progress`, `_finish`, `stop` per PR8, setting `model.u` before each step)
- Test: `tests/trajectory/test_store.py`, `tests/trajectory/test_runner.py`

**Interfaces:**
- Consumes: Task 4's `BackendModel` attributes; `StepRecord.energies_diag`, `c_mch`, `active_mch`, `u`.
- Produces: log columns (triplet runs only) `energies_diag` (N), `active_mch` (spin-free index), `energies_sf` (n + m), `populations_mch` (n + m); `output.h5` schema 2 datasets `active`, `populations`, `energies` (spin-free MCH), `hop_from`/`hop_to` (spin-free, PR1), `active_diag`, `populations_diag`, `energies_diag`, `hop_from_diag`/`hop_to_diag`, attrs `n_singlets`, `n_triplets`, `soc_operator` (the last from the stage's `attrs`).

- [ ] **Step 1: Failing tests:**
  - `log_dtype(4, 6)` equals today's dtype exactly (fields and order); `log_dtype(8, 6, n_spin_free=4)` adds the four columns at the end;
  - a runner over the canned spin-orbit model (`BackendModel(ModelBackend(lvc_soc))`, initial state S1, 20 steps, `hop_energy = "current"`, a `ReplaySource([0.0] * 400)` so hops happen): `steps.log` rows have `populations_mch` summing to 1 within 1e-12, `active_mch` equal to `spin_free_index(n, m, record.active_mch)` of the `StepRecord`s collected through a spy on the runner's `_append` (U itself is not logged); `output.h5` `schema_version` 2, `state_labels` `["S0","S1","T1","T2"]`, `populations` equal to the summed `|c_mch|^2`, `active` the spin-free MCH state, `hop_from`/`hop_to` spin-free at hop records and -1 elsewhere, the `_diag` datasets present with N columns, every dataset with `units`;
  - restart: 10 + kill + 10 steps equal 20 uninterrupted steps bit for bit in `steps.log` and `output.h5` (the existing crash-point test pattern of `test_runner.py`, extended to the spin-orbit model);
  - `stop_after_s0_fs` stops when the active spin-free state is S0 (PR8);
  - without triplets the existing `test_store.py` and `test_runner.py` stay green unmodified, and `output.h5` keeps `schema_version` 1.
- [ ] **Step 2:** see them fail. **Step 3:** implement (the runner passes `n_spin_free` only when `model.n_triplets > 0`; `_append` fills the new columns from the `StepRecord`: `energies_diag`, `active_mch` = `spin_free_index(..., record.active_mch)`, `energies_sf` = the spin-free energies (the snapshot's expanded energies at the spin-free rows), `populations_mch` = `spin_free_populations(|record.c_mch|^2)`; for the t = 0 row from the state, the same quantities from `state.u` and `state.c` via `mch_coefficients`/`mch_state`; `write_output_h5` derives the spin-free `hop_from`/`hop_to` from consecutive kept-or-not rows of `active_mch` at rows with `hop_to >= 0` — derive from the full log before thinning, so the "previous record" is the previous accepted (sub)step). **Step 4:** `W 'pytest tests/trajectory -q -W error -m "not slow"'`, SandyBridge, ruff, mypy. **Step 5:** commit `feat: store the spin-free populations of spin-orbit trajectories`.

### Task 6: the stage and the canned spin-orbit pipeline

**Files:**
- Modify: `src/gqteashmd/stages/dynamics.py` (`run_task`: `BackendModel(..., eselect_Eh=...)`; the initial MCH index of the selected singlet; `attrs` with `soc_operator`; the summary rows' active label per PR6)
- Test: `tests/stages/test_dynamics.py`, `tests/stages/test_dynamics_pipeline.py`

**Interfaces:** Consumes Tasks 3-5. Produces a `dynamics` stage that runs triplet projects.

- [ ] **Step 1: Failing tests** (canned pipeline `canned_dynamics` with `GQTEASHMD_CANNED_LVC_SOC=1`, PR5; the config `[electronic]` with `triplets = 2` matching the model, the triplet rules satisfied; 4 trajectories, 20 fs):
  - the stage finishes; every `output.h5` is schema 2 with labels `S0 S1 T1 T2`; `dynamics_summary.json` rows name spin-free active states;
  - the summed `populations` of each trajectory equal a direct `run_trajectory` of the LVC model with the same initial conditions and the trajectory's `task_rng` stream (positions within 1e-10; populations within 1e-10);
  - kill and resume of the canned spin-orbit pipeline (the existing slow kill-test pattern, `slow`) gives byte-identical `output.h5` files;
  - extension (raise `tmax_fs`) works;
  - without the environment variable the existing pipeline tests stay green unmodified.
- [ ] **Step 2-5:** fail, implement, `W 'pytest tests/stages/test_dynamics.py tests/stages/test_dynamics_pipeline.py -q -W error'` (all markers), SandyBridge for the new tests, ruff, mypy; commit `feat: run the dynamics stage with triplets`.

### Task 7: the analysis

**Files:**
- Modify: `src/gqteashmd/analysis/hops.py` (`transitions(t, labels, n_singlets) -> list[Transition]`, `transition_statistics(...)`), `src/gqteashmd/analysis/report.py` (the ISC block of `analysis_markdown`), `src/gqteashmd/stages/analysis.py` (the fit default; reading `n_singlets`/`n_triplets` from the `output.h5` attributes; the new summary keys)
- Test: `tests/analysis/test_hops.py`, `tests/analysis/test_report.py`, `tests/stages/test_analysis.py`, `tests/stages/test_analysis_pipeline.py`

**Interfaces:**
- Produces: `Transition(task, time_fs, from_state, to_state, at_hop: bool, included, after_horizon)`; `transitions(t: Trajectory, *, included, horizon_fs) -> list[Transition]` (every change of `t.active` between consecutive records; `at_hop` when the later record has `hop_to >= 0`); `transition_statistics(events, included, labels, horizon_fs) -> dict` with `kinds` (`"S-S"`, `"S-T"`, `"T-S"`, `"T-T"`, each `{"at_hop": int, "without_hop": int}`), `pairs` (label pairs), `first_singlet_to_triplet_fs` (per trajectory, censored like the S0 arrival: `null` when none before the horizon), and their mean/median over the uncensored; `analysis.json` key `transitions` (present only with triplets).

- [ ] **Step 1: Failing tests:**
  - `transitions` on a hand-built `Trajectory` (labels S0 S1 T1 T2; active 1,1,2,2,0 with hops at records 2 and 4, plus a change 2 -> 3 without a hop at record 3) gives S-T at hop, T-T without hop, T-S at hop; the kind counts; `first_singlet_to_triplet_fs` and its censoring;
  - the fit default: labels with a T give `fit_states = ["S1", ...]` (the excited singlets only); without T unchanged (`labels[1:]`);
  - `analysis.json` of a triplet project has `transitions`; `report.md` has a `## Intersystem crossing` section (counts and the first S-to-T time); a singlet project's `analysis.json` and `report.md` are unchanged (the existing tests unmodified);
  - the canned spin-orbit pipeline through `canned_analysis` (`GQTEASHMD_CANNED_LVC_SOC=1`): the classical populations equal a direct recomputation from the `output.h5` files (the P8 Task 9 pattern), `transitions` present.
- [ ] **Step 2-5:** fail, implement (`analysis/` stays free of `trajectory/` and `electronic/` imports: the stage passes `n_singlets` from the h5 attribute), `W 'pytest tests/analysis tests/stages/test_analysis.py tests/stages/test_analysis_pipeline.py -q -W error -m "not slow"'`, SandyBridge, ruff, mypy; commit `feat: analyse singlet-triplet transitions`.

### Task 8: PySCF smoke tests

**Files:** Test: `tests/validation/test_pyscf_soc_dynamics.py` (marker `pyscf`; the CIS one `slow` if over 60 s)

- [ ] **Step 1:** for H2CS/6-31G at the committed geometry with zero velocities displaced by 0.01 bohr along a fixed random direction (seed 7), initial state S1, dt 0.25 fs, 8 steps, `ReplaySource([0.0] * 400)` (every step hops when a probability is positive), SA-CASSCF(4,3) 2S+2T and CIS 3S+3T (`td_conv_tol` 1e-7): the trajectory finishes; every record finite; `populations_mch` sum to 1 within 1e-10; at least one hop; the `hamiltonian` of the last snapshot Hermitian within 1e-12; `state_labels` right. Assert nothing about energy conservation (owner decision 4).
- [ ] **Step 2:** run (also SandyBridge); ruff; commit `test: run spin-orbit trajectories on thioformaldehyde`.

### Task 9: the SHARC swap harness and the asserted swap tests

**Files:**
- Create: `tools/sharc_compare/run_sharc_soc_swap.py` (the harness: a `SocSwapModel` serving `BackendModel`'s expanded snapshot to `sharc.x` through `run_sharc_lvc.format_qm_out`/`QmServer`, the SHARC input from Task 1's facts, `output.dat` parsing via `run_sharc_lvc.parse_lvc_output_dat` or an extension; `run_ours` through `trajectory.runner` with `ReplaySource` of SHARC's random numbers; `differences(a, b)` over the gauge-invariant quantities), `tests/sharc_compare/test_run_sharc_soc_swap.py` (unit tests without SHARC), `tests/validation/test_sharc_soc_swap.py` (markers `sharc`, `pyscf`, `slow`)
- [ ] **Step 1: unit tests** (no SHARC): the provider's `QM.out` for a canned snapshot equals the LVC writer's output; the permutation of Task 1 (if any) round-trips; `differences` on identical inputs gives zeros and on shifted inputs the shift; the replay source consumes SHARC's random numbers in step order.
- [ ] **Step 2: the asserted tests** (spec 5.1), parametrised over the two methods: (a) hopping off, 40 steps, the four bounds per step and component; (b) hopping on from Task 1's start sample, `hop_energy = "previous"`, window to the first singlet-triplet hop + 20 steps (at most 200), the four bounds and the same active spin-free state; each assertion names its quantity and step; the measured maxima printed.
- [ ] **Step 3:** run with `SHARC_HOME`, `SHARC_PYTHON`; record every maximum in `task-9-report.md`. A miss: BLOCKED with the numbers. **Step 4:** ruff; commit `test: compare spin-orbit trajectories with SHARC by a swap test`.

### Task 10: the dropped-term report

**Files:** Create: `tools/validation/ngt_term_report.py`, `tests/validation/test_ngt_term_report.py` (helper tests, fast)

- [ ] **Step 1:** helper test: for a synthetic snapshot with known singlet-singlet NACs and U, the reported term equals `Re[(U^dag G_offdiag U)]_{lambda lambda}` computed by hand.
- [ ] **Step 2:** the script: along the (a) SA-CASSCF trajectory of Task 9 (re-run ours), at every step compute the CASSCF singlet-singlet NACs (`compute_more(nacs=...)`), the off-diagonal NGT term projected into the active diagonal state, and its norm relative to the force used; JSON out (max, median, per step). Run it; record in `task-10-report.md`. Commit `test: report the spin-orbit force term left out by gradient_transformation none`.

### Task 11 (controller): the test tiers

- [ ] On the branch head: `W 'pytest -m "not slow and not pyscf" -W error -q'`; `W 'pytest tests/trajectory tests/stages tests/analysis -W error -q'` (all markers); `W 'pytest -m "pyscf or slow" -W error -q --ignore=tests/validation/test_tully_exact.py --ignore=tests/validation/test_tully_overlap_exact.py --ignore=tests/validation/test_wavepacket2d.py'`; with `SHARC_HOME`/`SHARC_PYTHON`/`ORCA_EXE`, `W 'pytest -m sharc -W error -q'`. Record counts and times in the ledger.

### Task 12: documents

**Files:** Create `docs/superpowers/reports/2026-11-27-dynamics-soc.md` (every number marked asserted, report or inference; the Task 1 facts; the swap maxima; the dropped-term numbers; what was decided after numbers were seen; the honest summary). Modify USER_MANUAL.md ("Read this first"; 4.2 the two keys and the triplet rules; 5.x a triplet dynamics example from Task 8's settings; 6 the `output.h5` schema 2 datasets and the `transitions` summary keys; 8.x the method: the expansion, the window, the dropped term, the cost; 10 limitations; 11 roadmap), CLAUDE.md (a sub-project block, status, limitations 1a and 4, the remaining work: O7 closed or what remains, the Environment counts, the deferred minors), CHANGELOG `Unreleased`, the spec erratum (PR1's naming and anything else that differed). Run `W 'pytest tests/site -q -W error'`; commit `docs: record the dynamics with spin-orbit coupling`.

After Task 12: the final whole-branch review, its fix wave, and superpowers:finishing-a-development-branch (the owner decides the merge).
