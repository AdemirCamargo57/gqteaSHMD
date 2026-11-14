# P9b SHARC Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare our dynamics driver and analysis with SHARC 4.1 at the ensemble level on ethylene, under pass rules fixed before any run; measure the tau coverage of the analysis and set the minimum-N warning from it; add a replayed single-trajectory SHARC check whose population part can fail.

**Architecture:** Everything except one analysis change lives in `tools/sharc_compare/`, `tools/validation/` and `tests/`. SHARC trajectories are read into the `analysis.grid.Trajectory` type, so our unchanged `analysis/` functions analyse both programs; the pass rules are pure functions in `ensemble_compare.py`, tested on synthetic ensembles; the long runs are measurements by hand whose verdicts the tool computes.

**Tech Stack:** Python 3.12, NumPy, SciPy (`scipy.stats.ks_2samp`, `scipy.optimize.least_squares`), h5py, PySCF 2.14, pytest; SHARC 4.1 executed as a black box (`sharc.x`, `SHARC_PYSCF.py`, `populations.py`, `make_fit.py`).

**Spec:** [docs/superpowers/specs/2026-11-14-p9b-sharc-comparison-design.md](../specs/2026-11-14-p9b-sharc-comparison-design.md) (owner-approved 2026-11-14). The spec is binding; where this plan differs, the "Plan rulings" below say why.

## Global Constraints

- **Clean room:** SHARC 4.1 (`/home/ajc/sharc4-4.1`) is only *executed*, from `tools/sharc_compare/`; inputs come from the public SHARC 4.1 manual (sharc-md.org, `SHARC_Manual.pdf`) and from prompts observed at execution; nothing in the installation is read, copied or adapted; `gqteashmd` never imports SHARC or `tools/`.
- **Run everything in WSL:** `wsl -d OracleLinux_9_5 -- bash -lc 'cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && conda activate gqteashmd-dev && <command>'`. Every `Run:` line below is such a `<command>`. Live SHARC needs `SHARC_HOME=/home/ajc/sharc4-4.1 SHARC_PYTHON=/home/ajc/miniforge3/envs/sharc4.1/bin/python`.
- **Git from WSL only**, never `--no-verify`; conventional subjects; one commit per task; every commit message ends with `Co-Authored-By: <the committing model's trailer>`.
- **Runs and large outputs in `~/p9b/`** (WSL native filesystem), never in OneDrive or `/mnt/c`.
- Lint `ruff format . && ruff check .` (line length 100); `mypy` clean on `src/`.
- **The only change under `src/`** is `src/gqteashmd/stages/analysis.py` (Task 6).
- System (spec 3.1): ethylene `tests/data/ethylene.xyz`, SA3-CASSCF(2,2)/6-31G\*, `[selection] states = ["S1"]`, `tmax_fs` 100, `dt_fs` 0.5, every other `[dynamics]` key at its default; **seed 2028**; `n_trajectories` 100; `n_samples` 300, raised to 500 before any dynamics run if fewer than 100 trajectories are selected.
- Kept (spec 3.2): ours = outcome `finished` or `stopped_in_s0` and record-to-record `|dE_tot|` <= 0.1 eV; SHARC = reaches `tmax_fs` and record-to-record `|dE_tot|` <= 0.1 eV. Paired set = kept by both. **Fewer than 30 paired: every swap verdict `inconclusive`.**
- Swap rules (spec 3.3), bootstrap over initial-condition indices jointly, **B = 1000**, stream `task_rng(2028, "p9b-compare", 0)`: (1) classical S0, S1, S2 at every 1 fs: `|dP| <= 4 SE + 0.02`; (2) `|dtau| <= 4 sqrt(s_a^2 + s_b^2)`, same for `t0`; (3) first-hop times two-sample KS `p >= 1e-3`.
- Analysis vs SHARC tools (spec 4): all 100 swap SHARC trajectories; classical exact; quantum within one unit of SHARC's last printed digit; fit within the combined uncertainties on a model both express.
- Replay (spec 5): positions <= 1e-5 bohr, velocities <= 1e-6 a.u., populations and hopping probabilities <= 1e-5, identical active state; end of comparability = first random number within 1e-5 of a cumulative hopping probability on either side, else 80 steps; at least one hop and `P(S0) >= 1e-3` in the window.
- Coverage (spec 6): `t0 + Exp(tau)`, t0 10 fs, tau 40 fs, grid 0.5 fs to 100 fs, **N = 20, 50, 100**, **400 repetitions**, **B = 500**, through `analysis.kinetics.bootstrap_fit`; N_min = smallest N with coverage >= 0.90, else 100.
- Failures (spec 8): nothing is tuned away (no new seed, no re-run with other settings, no changed bound); no behaviour fix beyond Task 6; everything decided after numbers were seen is stated.

## Plan rulings (binding where they differ from the spec)

- **PR1 — the SHARC reader produces `analysis.grid.Trajectory`**, not `TrajectoryData` (spec 7.2): every `analysis/` function takes `grid.Trajectory`; `TrajectoryData` is the HDF5 view of our own files. Cost if wrong: none (an internal type).
- **PR2 — ours-nacdr is a copy of the seed-2028 project rerun from `dynamics` with `coupling = "nacdr"`** (spec 3.1 named `run_variants.py`): `run_variants.py` builds its backend without `reference_coords_bohr`, so the CASSCF start rule would not apply there; the copied project keeps byte-identical upstream artifacts. Cost if wrong: if the copied project does not run, Task 7 falls back to a fresh project with an equality check of `trajectories.json` and the `samples.h5` positions/velocities.
- **PR3 — the ensemble runs keep `substep_interpolation = "midpoint"` (our default, spec 3.1) although `sharc.x` uses its endpoint scheme.** P7a-1 measured the effect at 2.5e-4 in populations over 200 Tully steps, far below the 0.02 floor of rule 1. The replay test (bounds 1e-5) uses `"endpoint"`. Cost if wrong: a small systematic difference attributed to the drivers; the report states it.
- **PR4 — SHARC's frustrated hops** are taken from whatever Task 1 finds in `output.lis`/`output.dat`; if neither records them, the SHARC column of the frustrated-hop table is `null`. Report-only. Cost: a missing report column.
- **PR5 — a fit that gives no estimate or no bootstrap std, and a KS sample with fewer than 2 hop times, give `inconclusive` with the reason** (the spec names `inconclusive` only for < 30 paired). Cost if wrong: a broken fit would not fail; the reason is in the JSON and the report.
- **PR6 — the comparison grid** is grid steps `0, 2, ..., n_total` (every 1 fs); all paired trajectories reach `tmax_fs` (`stop_after_s0_fs` is off by default), so no horizon cut is needed; `dt_out` = 1.0 fs in the fits.
- **PR7 — "exact" for the classical populations of `populations.py`** means `|ours - SHARC| <= 0.5 * 10^-d` with `d` the printed decimals: every value is a multiple of 1/100, so any real difference is >= 0.01. Cost: none.
- **PR8 — the swap provider already supplies overlaps** (`SwapModel` behind `QmServer`, `phases_from_interface = True`, P7b), so spec 7.1's "added if missing" needs no code. The swap jobs run in **separate processes** (`spawn`), because PySCF's thread setting is process-global and the work is Python-heavy.
- **PR9 — the replay test (spec 5) comes after the runs** (Task 8): its start is chosen from the seed-2028 swap ensemble, which Task 7 produces.

## File structure

- Create `tools/sharc_compare/sharc_trajectories.py` — SHARC trajectory directory and our `output.h5` into `grid.Trajectory`; the kept rule (Task 2).
- Create `tools/sharc_compare/ensemble_compare.py` — paired set, the three rules, report tables, CLI (Task 3).
- Modify `tools/sharc_compare/run_sharc_pyscf_ensemble.py` — `--provider swap` (Task 4).
- Modify `tools/diagnosis/project_inputs.py` — `ProjectInputs.equilibrium_bohr` (Task 4).
- Create `tools/sharc_compare/run_sharc_analysis.py` — drives `populations.py`, `make_fit.py`; parsers; the comparison of section 4 (Task 5).
- Create `tools/validation/tau_coverage.py` (Task 6). Modify `src/gqteashmd/stages/analysis.py` (Task 6).
- Create `tools/sharc_compare/p9b/ethylene_seed2028.toml`, `tools/sharc_compare/p9b/variants_p9b.toml` (Task 7).
- Create `tests/data/sharc/p9b/` fixtures (Task 1), `tests/data/ethylene_hop_start.json` (Task 8).
- Tests: `tests/sharc_compare/test_sharc_trajectories.py` (Task 2), `tests/sharc_compare/test_ensemble_compare.py` (Task 3), `tests/diagnosis/test_sharc_ensemble.py` (extended, Task 4), `tests/sharc_compare/test_run_sharc_analysis.py` (Task 5), `tests/validation/test_tau_coverage_tool.py` and `tests/stages/test_analysis.py` (Task 6), `tests/validation/test_sharc_pyscf_dynamics.py` (extended, Task 8).
- Create `docs/superpowers/reports/YYYY-MM-DD-p9b-sharc-comparison.md`; modify `CLAUDE.md`, `USER_MANUAL.md`, the spec's erratum (Task 10).

---

### Task 1: Observe SHARC's formats and analysis tools; record the facts; commit fixtures

A measurement task: no tool code. Its facts file is the source of every SHARC prompt, answer and format used by Tasks 2, 4, 5 and 8.

**Files:**
- Create: `tests/data/sharc/p9b/traj_00000/output.dat`, `tests/data/sharc/p9b/traj_00000/output.lis`, `tests/data/sharc/p9b/traj_00001/...` (the same files), plus the output files of `populations.py` and `make_fit.py` run on them (names as observed) under `tests/data/sharc/p9b/analysis/`.
- Create: `tests/data/sharc/p9b/README.md` (how each file was produced: command, SHARC version, date).
- Create (ledger, git-ignored): `.superpowers/sdd/2026-11-14-p9b-sharc-comparison/task-1-facts.md`.

**Interfaces:**
- Consumes: `run_sharc_pyscf_dynamics.run_sharc_trajectory`, `SwapModel`, `ethylene_initial_conditions`, `interface_geometry` (existing).
- Produces: the fixtures and the facts file with sections F1-F6 below (later tasks quote them by section).

- [ ] **Step 1: Two short swap trajectories with hopping on.** In a throwaway script in `~/p9b/task1/` (not committed), build our SA3 backend (`PySCFBackend(MoleculeSpec(symbols, 0, 1), ElectronicSection.model_validate({"method": "casscf", "basis": "6-31g*", "singlets": 3, "casscf": {"ncas": 2, "nelecas": 2}}), log_file=..., threads=2, memory_mb=1500, scratch_dir=..., reference_coords_bohr=q0)` with `q0, v0` from `ethylene_initial_conditions()`), and run `run_sharc_trajectory(interface_geometry(q0.reshape(-1, 3)), v0.reshape(-1, 3), n_steps=10, workdir=~/p9b/task1/traj_0000k, symbols=..., masses_amu=..., sharc_home=..., provider="ours", model=SwapModel(backend, coupling="overlap"), coupling="overlap", n_states=3, state_mch=1, hopping="sharc", decoherence="edc", seed=k + 1)` for k = 0, 1 (a fresh backend each). Copy `output.dat` and `output.lis` of both into `tests/data/sharc/p9b/traj_0000k/`.
- [ ] **Step 2: F1 — output formats.** Record in the facts file: the blocks present in `output.dat` at `printlevel 2` with `coupling overlap` (in particular "6 Hopping" and "9 Random"), the line format of a hop and of a frustrated hop in `output.lis` (and whether `output.dat` marks a frustrated hop at all; PR4), the file sizes.
- [ ] **Step 3: F2 — `populations.py`.** Read its section of the public manual (record the section number). Execute it in a directory holding the two trajectories (record the layout it requires, e.g. `TRAJ_*` names; if so, Task 5 builds a symlink tree), answering its prompts for (a) classical populations of the active MCH state and (b) quantum populations `|c|^2` in the MCH basis. Record every prompt verbatim with the answer given, the output file name, its columns and its print format (decimals). Copy the outputs to `tests/data/sharc/p9b/analysis/`.
- [ ] **Step 4: F3 — `make_fit.py`.** Read its manual section; record which kinetic models it supports and whether a delay `t0` can be expressed; whether it reports parameter uncertainties (and how); then execute it on the classical output of Step 3 with the model "S1 -> S0, one rate" (plus a delay if F3 says it can), recording every prompt, answer, the output file and its format. Copy the outputs to `tests/data/sharc/p9b/analysis/`.
- [ ] **Step 5: F4 — the fit model ruling.** Write in the facts file, before any comparison: **if** `make_fit.py` can express the delay, the shared model is our delayed mono-exponential of `1 - P(S0)`; **otherwise** the shared model is `y(t) = A exp(-t / tau)` with `A = y(0)` (our side fitted by Task 5's `fit_undelayed`). State SHARC's uncertainty (`s_sharc`), or 0 if `make_fit.py` reports none.
- [ ] **Step 6: F5 — the quantum print bound.** From F2's print format, record the bound "one unit in the last printed digit" as a number (e.g. `1e-8`). F6: the SHARC environment the tools need (`SHARC`, `PYTHONPATH`, as in `run_excite.py`) and their run times on the fixtures.
- [ ] **Step 7: Commit the fixtures.**

```bash
git add tests/data/sharc/p9b
git commit -m "test: SHARC swap-trajectory and analysis-tool fixtures for P9b" -m "Co-Authored-By: <trailer>"
```

---

### Task 2: SHARC and our trajectories as `grid.Trajectory`; the kept rule

**Files:**
- Create: `tools/sharc_compare/sharc_trajectories.py`
- Test: `tests/sharc_compare/test_sharc_trajectories.py`

**Interfaces:**
- Consumes: `run_sharc_lvc.parse_lvc_output_dat(path) -> LvcTrajectory` (fields `steps`, `hamiltonian`, `c`, `active`, `e_kinetic`, `dt_au`, `random_numbers`, `probabilities`); `gqteashmd.trajectory.store.read_output_h5`; `gqteashmd.analysis.grid.Trajectory`, `INCLUDED_OUTCOMES`; `gqteashmd.analysis.energy.max_jump_eV`.
- Produces: `from_parsed(task: str, traj, *, n_total: int) -> grid.Trajectory`; `read_sharc_trajectory(directory: Path, task: str, *, n_total: int) -> grid.Trajectory | None`; `read_our_trajectory(path: Path, task: str) -> grid.Trajectory | None`; `max_jump_up_to(t: grid.Trajectory, n_total: int) -> float`; `is_kept(t: grid.Trajectory | None, n_total: int) -> bool`; constants `KEPT_JUMP_EV = 0.1`, `STOPPED = "stopped"`.

- [ ] **Step 1: Write the failing tests.**

```python
"""The SHARC and our trajectory readers and the kept rule (P9b spec 3.2, plan Task 2)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tests.diagnosis.conftest import load_tool, make_rows, write_trajectory

from gqteashmd.core.units import AU_TIME_IN_FS, HARTREE_IN_EV

st = load_tool("sharc_compare", "sharc_trajectories")
FIXTURES = Path(__file__).resolve().parents[1] / "data" / "sharc" / "p9b"


def _parsed(active, e_pot_eV, *, n=3, dt_fs=0.5, ekin=0.0):
    m = len(active)
    h = np.zeros((m, n, n), dtype=complex)
    for k in range(m):
        h[k] = np.diag(np.asarray(e_pot_eV[k]) / HARTREE_IN_EV)
    c = np.zeros((m, n), dtype=complex)
    c[np.arange(m), active] = 1.0
    return SimpleNamespace(
        steps=np.arange(m),
        hamiltonian=h,
        c=c,
        active=np.asarray(active),
        e_kinetic=np.full(m, ekin),
        dt_au=dt_fs / AU_TIME_IN_FS,
    )


def test_from_parsed_builds_hops_times_and_total_energy():
    e = [[0.0, 5.0, 7.0]] * 4
    t = st.from_parsed("traj_00003", _parsed([1, 1, 0, 0], e, ekin=0.01), n_total=3)
    assert t.task == "traj_00003" and t.outcome == "finished" and t.initial_state == 1
    np.testing.assert_allclose(t.time_fs, [0.0, 0.5, 1.0, 1.5])
    assert t.hop_to.tolist() == [-1, -1, 0, -1] and t.hop_from.tolist() == [-1, -1, 1, -1]
    assert np.all(t.ticks == 0) and not t.frustrated.any()
    np.testing.assert_allclose(t.e_total[0], 5.0 / HARTREE_IN_EV + 0.01)
    np.testing.assert_allclose(t.populations[2], [1.0, 0.0, 0.0])


def test_a_short_sharc_run_is_stopped():
    t = st.from_parsed("traj_00000", _parsed([1, 1], [[0.0, 5.0, 7.0]] * 2), n_total=3)
    assert t.outcome == st.STOPPED and not st.is_kept(t, 3)


def test_kept_rule_uses_the_jump_up_to_n_total():
    e = [[0.0, 5.0, 7.0], [0.0, 5.05, 7.0], [0.0, 5.3, 7.0], [0.0, 9.0, 7.0]]
    t = st.from_parsed("traj_00000", _parsed([1, 1, 1, 1], e), n_total=2)
    # steps 0-2 jump at most 0.25 eV (> 0.1): not kept; the 3.7 eV jump at step 3 is beyond n_total
    assert abs(st.max_jump_up_to(t, 2) - 0.25) < 1e-9
    assert not st.is_kept(t, 2)
    small = st.from_parsed("traj_00000", _parsed([1, 1, 1], e[:2] + [e[1]]), n_total=2)
    assert st.is_kept(small, 2)


def test_none_is_not_kept():
    assert not st.is_kept(None, 10)


def test_our_output_h5_reads_back(tmp_path):
    rows = make_rows([5.0, 5.0, 5.05], [[0.0, 5.0], [0.0, 5.0], [0.0, 5.0]])
    write_trajectory(tmp_path / "traj_00000", rows, outcome="finished")
    t = st.read_our_trajectory(tmp_path / "traj_00000" / "output.h5", "traj_00000")
    assert t is not None and t.outcome == "finished" and t.active.tolist() == [1, 1, 1]
    assert st.is_kept(t, 2)
    assert st.read_our_trajectory(tmp_path / "missing" / "output.h5", "x") is None


def test_the_fixture_reads_back():
    t = st.read_sharc_trajectory(FIXTURES / "traj_00000", "traj_00000", n_total=10)
    assert t is not None and t.outcome == "finished"
    assert t.grid_step.tolist() == list(range(11)) and t.populations.shape == (11, 3)
    np.testing.assert_allclose(t.populations.sum(axis=1), 1.0, atol=1e-6)
    assert st.read_sharc_trajectory(FIXTURES / "nope", "x", n_total=10) is None
```

- [ ] **Step 2: Run the tests to see them fail.**
Run: `pytest tests/sharc_compare/test_sharc_trajectories.py -q`
Expected: FAIL (`No such file or directory` / module `sharc_trajectories` not found).

- [ ] **Step 3: Write the module.**

```python
"""SHARC trajectories and ours in the form our analysis reads (P9b spec 3.2, 4, 7.2; PR1).

``from_parsed`` turns a parsed SHARC ``output.dat`` (``run_sharc_lvc.parse_lvc_output_dat``)
into an ``analysis.grid.Trajectory``: one record per printed step (``ticks`` 0), hops from the
changes of the active state, no frustrated hops (PR4), ``E_tot`` = the active MCH energy plus
the kinetic energy. SHARC is only executed; nothing from it is imported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from gqteashmd.analysis.energy import max_jump_eV
from gqteashmd.analysis.grid import FINISHED, INCLUDED_OUTCOMES, Trajectory
from gqteashmd.core.units import AU_TIME_IN_FS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_sharc_lvc import parse_lvc_output_dat  # noqa: E402

KEPT_JUMP_EV = 0.1  # spec 3.2: the analysis default energy_jump_eV
STOPPED = "stopped"  # a SHARC run that did not reach tmax


def from_parsed(task: str, traj, *, n_total: int) -> Trajectory:
    steps = np.asarray(traj.steps, dtype=np.int64)
    active = np.asarray(traj.active, dtype=np.int64)
    n = traj.hamiltonian.shape[1]
    energies = traj.hamiltonian[:, np.arange(n), np.arange(n)].real
    k = np.arange(steps.size)
    hop_from = np.full(steps.size, -1, dtype=np.int64)
    hop_to = np.full(steps.size, -1, dtype=np.int64)
    changed = np.flatnonzero(active[1:] != active[:-1]) + 1
    hop_from[changed] = active[changed - 1]
    hop_to[changed] = active[changed]
    dt_fs = float(traj.dt_au) * AU_TIME_IN_FS
    finished = steps.size > 0 and int(steps[-1]) >= n_total
    return Trajectory(
        task=task,
        outcome=FINISHED if finished else STOPPED,
        initial_state=int(active[0]) if active.size else -1,
        grid_step=steps,
        ticks=np.zeros(steps.size, dtype=np.int64),
        time_fs=steps * dt_fs,
        active=active,
        hop_from=hop_from,
        hop_to=hop_to,
        frustrated=np.zeros(steps.size, dtype=bool),
        e_total=energies[k, active] + np.asarray(traj.e_kinetic, dtype=float),
        populations=np.abs(np.asarray(traj.c)) ** 2,
    )


def read_sharc_trajectory(directory: Path, task: str, *, n_total: int) -> Trajectory | None:
    """None for a directory without a readable output.dat (a run that never started)."""
    path = Path(directory) / "output.dat"
    if not path.is_file():
        return None
    try:
        return from_parsed(task, parse_lvc_output_dat(path), n_total=n_total)
    except (ValueError, IndexError, OSError):
        return None


def read_our_trajectory(path: Path, task: str) -> Trajectory | None:
    """Our output.h5 as the analysis stage reads it (stages/analysis.py:_trajectory)."""
    from gqteashmd.trajectory.store import read_output_h5

    if not Path(path).is_file():
        return None
    data = read_output_h5(Path(path))
    return Trajectory(
        task=task,
        outcome=data.outcome,
        initial_state=data.labels.index(data.initial_state),
        grid_step=np.asarray(data["grid_step"], dtype=np.int64),
        ticks=np.asarray(data["ticks"], dtype=np.int64),
        time_fs=np.asarray(data["time_fs"], dtype=float),
        active=np.asarray(data["active"], dtype=np.int64),
        hop_from=np.asarray(data["hop_from"], dtype=np.int64),
        hop_to=np.asarray(data["hop_to"], dtype=np.int64),
        frustrated=np.asarray(data["frustrated"]) != 0,
        e_total=np.asarray(data["e_total"], dtype=float),
        populations=np.asarray(data["populations"], dtype=float),
    )


def max_jump_up_to(t: Trajectory, n_total: int) -> float:
    """The largest record-to-record |dE_tot| (eV) over grid steps <= n_total (P8 PR3)."""
    return max_jump_eV(t.e_total[t.grid_step <= n_total])


def is_kept(t: Trajectory | None, n_total: int) -> bool:
    """Spec 3.2: an included outcome (SHARC: reached tmax) and no jump above 0.1 eV."""
    if t is None or t.outcome not in INCLUDED_OUTCOMES:
        return False
    return max_jump_up_to(t, n_total) <= KEPT_JUMP_EV
```

- [ ] **Step 4: Run the tests.** Run: `pytest tests/sharc_compare/test_sharc_trajectories.py -q -W error`. Expected: 6 passed. If the fixture test fails on the record count or on `outcome`, adjust the test to the Task 1 F1 facts (the fixture has 10 steps), not the reader.
- [ ] **Step 5: Lint and commit.** Run: `ruff format tools tests && ruff check tools tests`.

```bash
git add tools/sharc_compare/sharc_trajectories.py tests/sharc_compare/test_sharc_trajectories.py
git commit -m "feat: read SHARC and our trajectories for the P9b comparison" -m "Co-Authored-By: <trailer>"
```

---

### Task 3: The paired comparison and its verdicts (`ensemble_compare.py`)

**Files:**
- Create: `tools/sharc_compare/ensemble_compare.py`
- Test: `tests/sharc_compare/test_ensemble_compare.py`

**Interfaces:**
- Consumes: Task 2's `is_kept`, `read_sharc_trajectory`, `read_our_trajectory`, `max_jump_up_to`; `analysis.populations.classical_populations`, `quantum_populations`, `bootstrap_indices`; `analysis.kinetics.bootstrap_fit`; `analysis.energy.max_drift_eV`, `summary_eV`; `gqteashmd.util.seeds.task_rng`; `tools/diagnosis/project_inputs.load_project_inputs`; `tools/diagnosis/forensics.find_stage_dir`.
- Produces: constants `MIN_PAIRED = 30`, `N_BOOTSTRAP = 1000`, `SE_FACTOR = 4.0`, `POPULATION_FLOOR = 0.02`, `FIT_FACTOR = 4.0`, `KS_MIN_P = 1e-3`, `OUTPUT_DT_FS = 1.0`, `COMPARE_STREAM = "p9b-compare"`, `START_DIFFERENCE_EV = 1e-3`; `paired_tasks(ours, theirs, n_total) -> list[str]`; `on_steps(trajs, steps, n_states) -> tuple[np.ndarray, np.ndarray]` (active (K, T), populations (K, T, n)); `population_rule(active_a, active_b, n_states, indices) -> dict`; `fit_rule(t_fs, per_a, per_b, indices) -> dict`; `first_hop_times(trajs) -> list[float]`; `ks_rule(times_a, times_b) -> dict`; `compare(ours, theirs, *, tasks, n_total, dt_fs, n_states, seed, asserted) -> dict`; CLI `main(argv)`.

Each rule returns `{"rule": str, "verdict": "pass" | "fail" | "inconclusive", "reason": str | None, ...numbers}`.

- [ ] **Step 1: Write the failing tests.**

```python
"""The P9b paired comparison rules on synthetic ensembles (spec 3.3; plan Task 3)."""

import numpy as np
from tests.diagnosis.conftest import load_tool

from gqteashmd.analysis.grid import Trajectory
from gqteashmd.analysis.populations import bootstrap_indices

ec = load_tool("sharc_compare", "ensemble_compare")
N_TOTAL = 200  # 100 fs at 0.5 fs
DT = 0.5


def _traj(task, hop_step=None, *, n_total=N_TOTAL, outcome="finished", jump_eV=0.0):
    steps = np.arange(n_total + 1)
    active = np.ones(steps.size, dtype=np.int64)
    hop_to = np.full(steps.size, -1)
    hop_from = np.full(steps.size, -1)
    if hop_step is not None:
        active[hop_step:] = 0
        hop_to[hop_step], hop_from[hop_step] = 0, 1
    e = np.zeros(steps.size)
    e[-1] = jump_eV / 27.211386245988
    pops = np.zeros((steps.size, 3))
    pops[np.arange(steps.size), active] = 1.0
    return Trajectory(
        task, outcome, 1, steps, np.zeros(steps.size, np.int64), steps * DT, active,
        hop_from, hop_to, np.zeros(steps.size, bool), e, pops,
    )


def _ensemble(n, rng, *, shift=0):
    hops = (10 + rng.exponential(40.0, size=n)) / DT + shift
    return {
        f"traj_{k:05d}": _traj(f"traj_{k:05d}", int(h) if h <= N_TOTAL else None)
        for k, h in enumerate(hops)
    }


def _compare(a, b, **kw):
    return ec.compare(a, b, tasks=sorted(a), n_total=N_TOTAL, dt_fs=DT, n_states=3,
                      seed=2028, asserted=True, **kw)


def test_paired_tasks_needs_both_kept():
    a = {"t0": _traj("t0"), "t1": _traj("t1", outcome="overlap_failure"), "t2": _traj("t2")}
    b = {"t0": _traj("t0"), "t1": _traj("t1"), "t2": _traj("t2", jump_eV=0.5)}
    assert ec.paired_tasks(a, b, N_TOTAL) == ["t0"]


def test_identical_ensembles_pass_every_rule():
    a = _ensemble(60, np.random.default_rng(1))
    r = _compare(a, dict(a))
    assert [v["verdict"] for v in r["rules"]] == ["pass", "pass", "pass", "pass"]
    pop = r["rules"][0]
    assert pop["max_se"] == 0.0  # pairs resampled together: identical sides have zero spread


def test_a_shifted_population_fails():
    a = _ensemble(60, np.random.default_rng(2))
    b = {k: _traj(k, 4) for k in a}  # everyone hops at 2 fs
    r = _compare(a, b)
    assert r["rules"][0]["rule"] == "populations" and r["rules"][0]["verdict"] == "fail"


def test_fewer_than_thirty_paired_is_inconclusive():
    a = _ensemble(29, np.random.default_rng(3))
    r = _compare(a, dict(a))
    assert r["n_paired"] == 29
    assert {v["verdict"] for v in r["rules"]} == {"inconclusive"}


def test_population_rule_bound_is_4se_plus_floor():
    active_a = np.ones((40, 3), dtype=np.int64)
    active_b = active_a.copy()
    active_b[:1, :] = 0  # one of 40 differs everywhere: dP = 0.025
    idx = bootstrap_indices(np.random.default_rng(0), 40, 1000)
    r = ec.population_rule(active_a, active_b, 3, idx)
    assert abs(r["max_abs_difference"] - 0.025) < 1e-12
    assert r["verdict"] == ("pass" if 0.025 <= 4 * r["max_se"] + 0.02 else "fail")


def test_ks_rule():
    rng = np.random.default_rng(4)
    same = ec.ks_rule(list(rng.uniform(0, 10, 50)), list(rng.uniform(0, 10, 50)))
    assert same["verdict"] == "pass"
    apart = ec.ks_rule(list(rng.uniform(0, 10, 50)), list(rng.uniform(50, 60, 50)))
    assert apart["verdict"] == "fail" and apart["p_value"] < 1e-3
    assert ec.ks_rule([1.0], [1.0, 2.0])["verdict"] == "inconclusive"


def test_a_fit_without_decay_is_inconclusive():
    t = np.arange(0.0, 100.0 + 1e-9, 1.0)
    per = np.ones((40, t.size))
    idx = bootstrap_indices(np.random.default_rng(0), 40, 100)
    r = ec.fit_rule(t, per, per.copy(), idx)
    assert r["verdict"] == "inconclusive" and r["reason"]


def test_first_hop_times_skip_trajectories_without_hops():
    assert ec.first_hop_times([_traj("a", 20), _traj("b")]) == [10.0]


def test_report_tables_are_present():
    a = _ensemble(40, np.random.default_rng(5))
    r = _compare(a, dict(a))
    for key in ("kept_ours", "kept_theirs", "outcomes_ours", "outcomes_theirs",
                "quantum_max_abs_difference", "hop_pairs_ours", "hop_pairs_theirs",
                "drift_eV_ours", "drift_eV_theirs", "start_difference"):
        assert key in r["report"], key
```

- [ ] **Step 2: Run them to see them fail.** Run: `pytest tests/sharc_compare/test_ensemble_compare.py -q`. Expected: FAIL (module missing).

- [ ] **Step 3: Write the module.**

```python
"""The P9b paired ensemble comparison (spec 3.2-3.4): kept sets, the paired set, the three
swap rules and the report tables. Pure functions; ``main`` reads a gqteashmd project and a
SHARC ensemble directory (``run_sharc_pyscf_ensemble.py``) and writes one JSON.

Example (WSL):
  python tools/sharc_compare/ensemble_compare.py ~/p9b/ours/run ~/p9b/swap/project \\
      --label swap --asserted --out ~/p9b/compare/swap.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.analysis.energy import max_drift_eV, summary_eV
from gqteashmd.analysis.grid import Trajectory
from gqteashmd.analysis.kinetics import bootstrap_fit
from gqteashmd.analysis.populations import (
    bootstrap_indices,
    classical_populations,
    quantum_populations,
)
from gqteashmd.core.units import HARTREE_IN_EV
from gqteashmd.util.seeds import task_rng

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "diagnosis"))
from sharc_trajectories import (  # noqa: E402
    is_kept,
    read_our_trajectory,
    read_sharc_trajectory,
)

MIN_PAIRED = 30
N_BOOTSTRAP = 1000
SE_FACTOR = 4.0
POPULATION_FLOOR = 0.02
FIT_FACTOR = 4.0
KS_MIN_P = 1e-3
OUTPUT_DT_FS = 1.0
COMPARE_STREAM = "p9b-compare"
START_DIFFERENCE_EV = 1e-3  # report: a step-0 energy difference above this = another solution
RULES = ("populations", "tau", "t0", "first_hop_ks")


def _verdict(rule: str, ok: bool | None, reason: str | None = None, **numbers: Any) -> dict:
    verdict = "inconclusive" if ok is None else ("pass" if ok else "fail")
    return {"rule": rule, "verdict": verdict, "reason": reason, **numbers}


def paired_tasks(
    ours: Mapping[str, Trajectory | None], theirs: Mapping[str, Trajectory | None], n_total: int
) -> list[str]:
    return sorted(
        k for k in ours if is_kept(ours[k], n_total) and is_kept(theirs.get(k), n_total)
    )


def on_steps(
    trajs: Sequence[Trajectory], steps: np.ndarray, n_states: int
) -> tuple[np.ndarray, np.ndarray]:
    active = np.zeros((len(trajs), steps.size), dtype=np.int64)
    pops = np.zeros((len(trajs), steps.size, n_states))
    for i, t in enumerate(trajs):
        where = {int(s): j for j, s in enumerate(t.grid_step) if int(t.ticks[j]) == 0}
        for k, s in enumerate(steps):
            j = where.get(int(s))
            if j is None:
                raise ValueError(f"{t.task}: no full-step record at grid step {int(s)}")
            active[i, k] = int(t.active[j])
            pops[i, k] = t.populations[j]
    return active, pops


def population_rule(
    active_a: np.ndarray, active_b: np.ndarray, n_states: int, indices: np.ndarray
) -> dict:
    d = classical_populations(active_a, n_states)[0] - classical_populations(active_b, n_states)[0]
    boot = np.stack(
        [
            classical_populations(active_a[idx], n_states)[0]
            - classical_populations(active_b[idx], n_states)[0]
            for idx in indices
        ]
    )
    se = boot.std(axis=0, ddof=1)
    bound = SE_FACTOR * se + POPULATION_FLOOR
    ratio = np.abs(d) / bound
    t_worst, s_worst = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
    return _verdict(
        "populations",
        bool(np.all(np.abs(d) <= bound)),
        max_abs_difference=float(np.abs(d).max()),
        max_se=float(se.max()),
        worst={"time_index": int(t_worst), "state": int(s_worst),
               "difference": float(d[t_worst, s_worst]), "bound": float(bound[t_worst, s_worst])},
        n_points_failing=int(np.count_nonzero(np.abs(d) > bound)),
    )


def fit_rule(t_fs: np.ndarray, per_a: np.ndarray, per_b: np.ndarray, indices: np.ndarray) -> dict:
    fa = bootstrap_fit(t_fs, per_a, indices, dt_out=OUTPUT_DT_FS)
    fb = bootstrap_fit(t_fs, per_b, indices, dt_out=OUTPUT_DT_FS)
    out = {}
    for name, ea, eb, sa, sb in (
        ("tau", fa.fit.tau_fs, fb.fit.tau_fs, fa.tau_std_fs, fb.tau_std_fs),
        ("t0", fa.fit.t0_fs, fb.fit.t0_fs, fa.t0_std_fs, fb.t0_std_fs),
    ):
        numbers = {"ours": ea, "theirs": eb, "std_ours": sa, "std_theirs": sb,
                   "flags_ours": list(fa.fit.flags), "flags_theirs": list(fb.fit.flags)}
        if None in (ea, eb, sa, sb):
            out[name] = _verdict(name, None, "a fit gave no estimate or no bootstrap std (PR5)",
                                 **numbers)
            continue
        bound = FIT_FACTOR * float(np.hypot(sa, sb))
        out[name] = _verdict(name, abs(ea - eb) <= bound, difference=float(ea - eb),
                             bound=bound, **numbers)
    out["verdict"] = (
        "inconclusive"
        if "inconclusive" in (out["tau"]["verdict"], out["t0"]["verdict"])
        else ("pass" if out["tau"]["verdict"] == out["t0"]["verdict"] == "pass" else "fail")
    )
    out["reason"] = out["tau"]["reason"] or out["t0"]["reason"]
    return out


def first_hop_times(trajs: Sequence[Trajectory]) -> list[float]:
    times = []
    for t in trajs:
        hops = np.flatnonzero(t.hop_to >= 0)
        if hops.size:
            times.append(float(t.time_fs[hops[0]]))
    return times


def ks_rule(times_a: Sequence[float], times_b: Sequence[float]) -> dict:
    if len(times_a) < 2 or len(times_b) < 2:
        return _verdict("first_hop_ks", None, "fewer than 2 first-hop times on a side (PR5)",
                        n_ours=len(times_a), n_theirs=len(times_b))
    from scipy.stats import ks_2samp

    res = ks_2samp(times_a, times_b)
    p = float(res.pvalue)
    return _verdict("first_hop_ks", p >= KS_MIN_P, p_value=p, statistic=float(res.statistic),
                    n_ours=len(times_a), n_theirs=len(times_b))


def _hop_pairs(trajs: Sequence[Trajectory]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for t in trajs:
        for j in np.flatnonzero(t.hop_to >= 0):
            c[f"S{int(t.hop_from[j])}->S{int(t.hop_to[j])}"] += 1
    return dict(sorted(c.items()))


def _report(ours, theirs, tasks, paired, n_total, steps, n_states) -> dict[str, Any]:
    present_o = [ours[k] for k in tasks if ours.get(k) is not None]
    present_t = [theirs[k] for k in tasks if theirs.get(k) is not None]
    po = [ours[k] for k in paired]
    pt = [theirs[k] for k in paired]
    quantum = None
    if paired:
        qa = quantum_populations(on_steps(po, steps, n_states)[1])
        qb = quantum_populations(on_steps(pt, steps, n_states)[1])
        quantum = float(np.abs(qa - qb).max())
    start: dict[str, float] = {}  # |E_tot(0) ours - SHARC| per task (eV): the start solution
    for k in tasks:
        a, b = ours.get(k), theirs.get(k)
        if a is None or b is None or not a.e_total.size or not b.e_total.size:
            continue
        start[k] = float(abs(a.e_total[0] - b.e_total[0]) * HARTREE_IN_EV)
    return {
        "kept_ours": sum(is_kept(ours.get(k), n_total) for k in tasks),
        "kept_theirs": sum(is_kept(theirs.get(k), n_total) for k in tasks),
        "outcomes_ours": dict(Counter(t.outcome for t in present_o)),
        "outcomes_theirs": dict(Counter(t.outcome for t in present_t)),
        "missing_ours": [k for k in tasks if ours.get(k) is None],
        "missing_theirs": [k for k in tasks if theirs.get(k) is None],
        "quantum_max_abs_difference": quantum,
        "hop_pairs_ours": _hop_pairs(po),
        "hop_pairs_theirs": _hop_pairs(pt),
        "frustrated_ours": int(sum(int(t.frustrated.sum()) for t in po)),
        "frustrated_theirs": None,  # PR4: set by main from output.lis when Task 1 F1 allows
        "drift_eV_ours": summary_eV([max_drift_eV(t.e_total) for t in po]),
        "drift_eV_theirs": summary_eV([max_drift_eV(t.e_total) for t in pt]),
        "start_difference": {
            "per_task_eV": start,
            "n_above": sum(v > START_DIFFERENCE_EV for v in start.values()),
            "threshold_eV": START_DIFFERENCE_EV,
        },
    }


def compare(
    ours: Mapping[str, Trajectory | None],
    theirs: Mapping[str, Trajectory | None],
    *,
    tasks: Sequence[str],
    n_total: int,
    dt_fs: float,
    n_states: int,
    seed: int,
    asserted: bool,
) -> dict[str, Any]:
    paired = paired_tasks({k: ours.get(k) for k in tasks}, theirs, n_total)
    every = max(1, round(OUTPUT_DT_FS / dt_fs))
    steps = np.arange(0, n_total + 1, every, dtype=np.int64)
    t_fs = steps * dt_fs
    out: dict[str, Any] = {
        "asserted": asserted,
        "n_tasks": len(tasks),
        "n_paired": len(paired),
        "paired": paired,
        "constants": {"MIN_PAIRED": MIN_PAIRED, "N_BOOTSTRAP": N_BOOTSTRAP,
                      "SE_FACTOR": SE_FACTOR, "POPULATION_FLOOR": POPULATION_FLOOR,
                      "FIT_FACTOR": FIT_FACTOR, "KS_MIN_P": KS_MIN_P, "seed": seed,
                      "stream": COMPARE_STREAM},
        "report": _report(ours, theirs, tasks, paired, n_total, steps, n_states),
    }
    if len(paired) < MIN_PAIRED:
        reason = f"only {len(paired)} paired trajectories (fewer than {MIN_PAIRED})"
        out["rules"] = [_verdict(r, None, reason) for r in RULES]
        return out
    po = [ours[k] for k in paired]
    pt = [theirs[k] for k in paired]
    aa, _ = on_steps(po, steps, n_states)
    ab, _ = on_steps(pt, steps, n_states)
    idx = bootstrap_indices(task_rng(seed, COMPARE_STREAM, 0), len(paired), N_BOOTSTRAP)
    fits = fit_rule(t_fs, (aa != 0).astype(float), (ab != 0).astype(float), idx)
    out["rules"] = [
        population_rule(aa, ab, n_states, idx),
        fits["tau"],
        fits["t0"],
        ks_rule(first_hop_times(po), first_hop_times(pt)),
    ]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    from forensics import find_stage_dir, jsonable
    from project_inputs import load_project_inputs

    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("project", type=Path, help="our gqteashmd project directory")
    p.add_argument("sharc", type=Path, help="a SHARC ensemble variant directory (traj_*/)")
    p.add_argument("--label", required=True)
    p.add_argument("--asserted", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2028)
    args = p.parse_args(argv)
    inputs = load_project_inputs(args.project)
    dyn = inputs.config.dynamics
    assert dyn is not None and inputs.config.electronic is not None
    n_total = round(dyn.tmax_fs / dyn.dt_fs)
    tasks = [s.task_id for s in inputs.starts]
    dyn_dir = find_stage_dir(inputs.project_dir, "dynamics")
    ours = {k: read_our_trajectory(dyn_dir / k / "output.h5", k) for k in tasks}
    theirs = {k: read_sharc_trajectory(args.sharc / k, k, n_total=n_total) for k in tasks}
    result = compare(ours, theirs, tasks=tasks, n_total=n_total, dt_fs=dyn.dt_fs,
                     n_states=inputs.config.electronic.singlets, seed=args.seed,
                     asserted=args.asserted)
    result.update(label=args.label, project=str(args.project), sharc=str(args.sharc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(jsonable(result), indent=2))
    print(json.dumps({"label": args.label, "n_paired": result["n_paired"],
                      "verdicts": {r["rule"]: r["verdict"] for r in result["rules"]}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`fit_rule` returns the `tau` and `t0` verdicts (and an overall one); `compare` lists `tau` and `t0` as separate rules. The start-difference table compares the step-0 total energies; with the same initial geometry and velocities, a difference above 1e-3 eV means another SA-CASSCF solution at the start (spec 3.4; P9a measured 0.176-0.372 eV for the different starts and at most 1.35e-6 eV for equal ones).

- [ ] **Step 4: Run the tests.** Run: `pytest tests/sharc_compare/test_ensemble_compare.py -q -W error`. Expected: 10 passed. `test_identical_ensembles_pass_every_rule` relies on `ks_2samp` of identical samples giving p = 1.
- [ ] **Step 5: Lint and commit.**

```bash
git add tools/sharc_compare/ensemble_compare.py tests/sharc_compare/test_ensemble_compare.py
git commit -m "feat: the P9b paired ensemble comparison and its verdicts" -m "Co-Authored-By: <trailer>"
```

---

### Task 4: The swap provider in the ensemble harness

**Files:**
- Modify: `tools/sharc_compare/run_sharc_pyscf_ensemble.py` (`sharc_settings`, `run_ensemble`, `main`)
- Modify: `tools/diagnosis/project_inputs.py` (`ProjectInputs.equilibrium_bohr`)
- Test: `tests/diagnosis/test_sharc_ensemble.py` (extend)

**Interfaces:**
- Consumes: `run_sharc_pyscf_dynamics.SwapModel(backend, *, coupling)`, `run_sharc_trajectory(..., provider="ours", model=..., coupling=...)`; `PySCFBackend(..., reference_coords_bohr=)`.
- Produces: `sharc_settings(electronic, dynamics, *, provider="sharc_pyscf") -> dict`; `swap_backend(inputs, electronic, directory: Path, *, threads: int, memory_mb: int)`; `run_ensemble(..., provider: str = "sharc_pyscf", threads: int = 2, memory_mb: int = 800)`; CLI `--provider sharc_pyscf|swap`, `--threads`, `--memory-mb`; `ProjectInputs.equilibrium_bohr: np.ndarray | None` (last field, default `None`, filled from `samples.h5`).

- [ ] **Step 1: Write the failing tests** (append to `tests/diagnosis/test_sharc_ensemble.py`, reusing its `_fake_world`/`_sections` helpers; for `provider="swap"` also monkeypatch `ens.swap_backend` to return `SimpleNamespace(states=())` and `ens.SwapModel` to a recorder).

```python
def test_swap_settings_use_the_project_coupling_and_interface_phases():
    electronic, dynamics = _sections()
    s = ens.sharc_settings(electronic, dynamics, provider="swap")
    assert s["coupling"] == dynamics.coupling and s["phases_from_interface"] is True
    assert "template" not in s


def test_direct_settings_stay_nacdr():
    electronic, dynamics = _sections()
    assert ens.sharc_settings(electronic, dynamics)["coupling"] == "nacdr"


def test_a_swap_job_passes_our_model_and_provider(tmp_path, monkeypatch):
    calls = _fake_world(monkeypatch, lambda t, k: {"returncode": 0, "data": None, "stderr_tail": ""})
    made = []
    monkeypatch.setattr(ens, "swap_backend", lambda *a, **k: made.append(a[2]) or "backend")
    monkeypatch.setattr(ens, "SwapModel", lambda backend, coupling: ("model", backend, coupling))
    ens.run_ensemble(Path("proj"), ens.Variant("base"), tmp_path, sharc_home=Path("/x"),
                     sharc_python="py", workers=1, provider="swap")
    for task, kwargs, _ in calls:
        assert kwargs["provider"] == "ours"
        assert kwargs["model"][0] == "model" and kwargs["model"][2] == kwargs["coupling"]
    assert [d.name for d in made] == ["traj_00000.backend", "traj_00001.backend"]


def test_an_unknown_provider_is_refused(tmp_path):
    with pytest.raises(ValueError, match="provider"):
        ens.run_ensemble(Path("proj"), ens.Variant("base"), tmp_path, sharc_home=Path("/x"),
                         sharc_python="py", workers=1, provider="nope")
```

- [ ] **Step 2: Run them to see them fail.** Run: `pytest tests/diagnosis/test_sharc_ensemble.py -q -m "not pyscf"`. Expected: the four new tests FAIL (`unexpected keyword argument 'provider'`).

- [ ] **Step 3: Implement.**
  - `project_inputs.py`: add `equilibrium_bohr: np.ndarray | None = None` as the last `ProjectInputs` field and pass `np.asarray(samples["equilibrium_bohr"], dtype=float).reshape(-1, 3)` in `load_project_inputs`.
  - `run_sharc_pyscf_ensemble.py`:

```python
PROVIDERS = {"sharc_pyscf": "sharc_pyscf", "swap": "ours"}  # our name -> run_sharc_trajectory's


def sharc_settings(electronic, dynamics, *, provider: str = "sharc_pyscf") -> dict[str, Any]:
    """Keyword arguments of ``run_sharc_trajectory`` for one variant (CASSCF only). ``swap``
    runs sharc.x on our backend with the project's coupling (P9b spec 3.1, PR8)."""
    # ... the existing checks, unchanged ...
    common = {
        "n_states": electronic.singlets,
        "dt_fs": dynamics.dt_fs,
        "hopping": "off" if dynamics.hopping == "none" else "sharc",
        "decoherence": "edc" if dynamics.decoherence == "edc" else "none",
    }
    if provider == "swap":
        return {**common, "coupling": dynamics.coupling, "phases_from_interface": True}
    c = electronic.casscf
    return {
        **common,
        "coupling": "nacdr",
        "phases_from_interface": False,
        "template": pyscf_template(
            Convergence(c.conv_tol, c.conv_tol_grad),
            ncas=c.ncas, nelecas=c.nelecas, roots=electronic.singlets, basis=electronic.basis,
        ),
    }


def swap_backend(inputs, electronic, directory: Path, *, threads: int, memory_mb: int):
    """Our backend as the stage builds it, with the CASSCF start rule's reference geometry
    (samples.h5 equilibrium coordinates, as `dynamics` passes it)."""
    from gqteashmd.electronic.base import MoleculeSpec
    from gqteashmd.electronic.pyscf_backend import PySCFBackend

    directory.mkdir(parents=True, exist_ok=True)
    return PySCFBackend(
        MoleculeSpec(inputs.symbols, inputs.charge, inputs.multiplicity),
        electronic,
        log_file=directory / "electronic.log",
        threads=threads,
        memory_mb=memory_mb,
        scratch_dir=directory / "scratch",
        reference_coords_bohr=inputs.equilibrium_bohr,
    )
```

  - In `run_ensemble`: validate `provider in PROVIDERS` first (`ValueError` naming `provider`); compute `settings = sharc_settings(electronic, dynamics, provider=provider)`; inside `job`, for `provider == "swap"` build `model = SwapModel(swap_backend(inputs, electronic, base / f"{start.task_id}.backend", threads=threads, memory_mb=memory_mb), coupling=settings["coupling"])` and call `run_sharc_trajectory(..., provider=PROVIDERS[provider], model=model, sharc_python=sharc_python, ...)`; for `sharc_pyscf` unchanged. Import `SwapModel` from `run_sharc_pyscf_dynamics` at the top with the other names.
  - Execution (PR8): `workers <= 1` runs the jobs inline in a loop; `provider == "sharc_pyscf"` keeps the `ThreadPoolExecutor`; `provider == "swap"` uses `ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))` with a **module-level** job function `_swap_job(payload: dict) -> dict` (payload: project dir, variant name/overrides, out dir, task id, sharc home, python, n_steps, threads, memory_mb; the worker reloads the project inputs). Keep `THREAD_VARIABLES` for `sharc.x`; our backend's own `threads` sets PySCF's threads in its process.
  - `main`: `--provider` (choices `sharc_pyscf`, `swap`; default `sharc_pyscf`), `--threads` (2), `--memory-mb` (800); the docstring gains a swap example.

- [ ] **Step 4: Run the tests.** Run: `pytest tests/diagnosis/test_sharc_ensemble.py tests/diagnosis -q -W error -m "not pyscf"`. Expected: all pass (the old tests unchanged).
- [ ] **Step 5: One live swap trajectory.** Add to the same file a test marked `sharc`, `pyscf`, `slow`, skipped without `SHARC_HOME`/`SHARC_PYTHON`, mirroring `test_one_short_sharc_trajectory` but with `provider="swap"` and `--tmax-fs 1.0`; assert `ended == "finished"`, `n_steps == 2`, `error is None`. Run: `SHARC_HOME=... SHARC_PYTHON=... pytest tests/diagnosis/test_sharc_ensemble.py -q -k swap -W error`. Expected: pass.
- [ ] **Step 6: Lint and commit.**

```bash
git add tools/sharc_compare/run_sharc_pyscf_ensemble.py tools/diagnosis/project_inputs.py tests/diagnosis/test_sharc_ensemble.py
git commit -m "feat: run sharc.x on our backend over a project's initial conditions" -m "Co-Authored-By: <trailer>"
```

---

### Task 5: SHARC's `populations.py` and `make_fit.py` against our analysis

**Files:**
- Create: `tools/sharc_compare/run_sharc_analysis.py`
- Test: `tests/sharc_compare/test_run_sharc_analysis.py`

**Interfaces:**
- Consumes: Task 1 facts F2-F6 (prompts, answers, layout, formats, the fit-model ruling, the quantum bound); Task 2's `read_sharc_trajectory`; Task 3's `on_steps`; `analysis.populations`, `analysis.kinetics`.
- Produces: `sharc_tool_environment(sharc_home: Path) -> dict[str, str]`; `run_tool(sharc_home, python, script: str, work_dir: Path, answers: Sequence[str], *, timeout_s: float = 3600) -> subprocess.CompletedProcess`; `populations_answers(kind: str) -> list[str]` (kind `classical`|`quantum`, the F2 answers verbatim with one comment per prompt); `fit_answers(...) -> list[str]` (F3); `parse_populations(path: Path) -> tuple[np.ndarray, np.ndarray]` (times fs, (T, n)); `parse_fit(path: Path) -> dict` (`tau_fs`, `t0_fs` or None, `tau_std_fs` or None); `fit_undelayed(t, per, indices) -> dict` (`tau_fs`, `tau_std_fs`; least squares on `log tau` with `A = y(0)`, the bootstrap std over `indices`); `compare_tools(sharc_dir: Path, tasks, *, n_total, dt_fs, n_states, sharc_home, python, work_dir, quantum_bound: float) -> dict`; CLI `main`.

- [ ] **Step 1: Write the failing tests** (fast tier: the parsers on the Task 1 outputs, `fit_undelayed` on a noise-free curve, the comparison arithmetic; marker `sharc`: `run_tool` on the two-trajectory fixture reproduces the stored outputs).

```python
from pathlib import Path

import numpy as np
import pytest
from tests.diagnosis.conftest import load_tool

from gqteashmd.analysis.populations import bootstrap_indices

ra = load_tool("sharc_compare", "run_sharc_analysis")
FIX = Path(__file__).resolve().parents[1] / "data" / "sharc" / "p9b"


def test_parse_populations_reads_the_stored_classical_output():
    t, p = ra.parse_populations(FIX / "analysis" / ra.CLASSICAL_OUTPUT)
    assert t[0] == 0.0 and p.shape[1] == 3
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_fit_undelayed_recovers_tau():
    t = np.arange(0.0, 100.0 + 1e-9, 1.0)
    per = np.tile(np.exp(-t / 30.0), (20, 1))
    r = ra.fit_undelayed(t, per, bootstrap_indices(np.random.default_rng(0), 20, 50))
    assert abs(r["tau_fs"] - 30.0) < 1e-6 and r["tau_std_fs"] < 1e-6


def test_classical_comparison_is_exact_to_the_print_precision():
    ours = np.array([[0.31, 0.69, 0.0]])
    theirs = np.array([[0.3100000000, 0.6900000000, 0.0]])
    assert ra.classical_agree(ours, theirs, decimals=10)
    assert not ra.classical_agree(ours, theirs + 0.01, decimals=10)


@pytest.mark.sharc
def test_populations_py_on_the_fixture_matches_the_stored_output(tmp_path):
    ...  # copies FIX/traj_* into tmp_path in the F2 layout, runs run_tool with
    # populations_answers("classical"), and compares with FIX/analysis (skip without SHARC_HOME)
```

  The `sharc` test is written out completely by the implementer from F2 (the layout and file names are facts, not choices); `CLASSICAL_OUTPUT`, `QUANTUM_OUTPUT`, `FIT_OUTPUT` are module constants holding the F2/F3 file names, and `classical_agree(ours, theirs, *, decimals) -> bool` implements PR7 (`np.all(np.abs(ours - theirs) <= 0.5 * 10**-decimals)`).

- [ ] **Step 2: Run them to see them fail.** Run: `pytest tests/sharc_compare/test_run_sharc_analysis.py -q -m "not sharc"`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `run_sharc_analysis.py`:
  - `sharc_tool_environment` as in `run_excite.py` (`SHARC = <home>/bin`, `PYTHONPATH` **replaced** by `<home>/lib:<home>/pysharc`).
  - `run_tool` like `run_excite`: stdin = the answers joined by newlines plus ten blank lines, `capture_output`, the timeout; a non-zero exit or a missing output raises `RuntimeError` with the tails and a hint to check the answers against Task 1 F2/F3.
  - The answer builders hold F2/F3's answers verbatim, one comment per prompt (as in `run_excite.py`).
  - `fit_undelayed`: `scipy.optimize.least_squares` on `x = log tau` with residual `A exp(-t/exp(x)) - y`, `A = y[0]`, start `log(max(t[-1], 1.0))`; the std is `np.std(taus, ddof=1)` over the fits of `per[idx].mean(axis=0)` for `idx in indices`.
  - `compare_tools`: build the F2 layout in `work_dir` (symlinks to `sharc_dir/traj_*`), run `populations.py` twice and `make_fit.py` once, read our side with `read_sharc_trajectory` (all tasks, **no exclusion**, spec 4) and `on_steps` on SHARC's printed times; verdicts: classical `pass` iff `classical_agree` at every shared time; quantum `pass` iff `max |diff| <= quantum_bound`; fit per F4 (`delayed`: our `bootstrap_fit` with `task_rng(2028, "p9b-compare", 0)` and B = 1000 vs `parse_fit`; `undelayed`: `fit_undelayed`), `pass` iff `|tau_ours - tau_sharc| <= 4 sqrt(s_ours^2 + s_sharc^2)` with `s_sharc` from F4 (0 if none). Returns one dict with every number and the three verdicts.
  - `main`: `run_sharc_analysis.py SHARC_ENSEMBLE_DIR PROJECT --sharc-home --python --work ~/p9b/tools --out JSON --quantum-bound X`.
- [ ] **Step 4: Run the tests.** Run: `pytest tests/sharc_compare/test_run_sharc_analysis.py -q -W error -m "not sharc"`, then with `SHARC_HOME`/`SHARC_PYTHON`: `pytest tests/sharc_compare/test_run_sharc_analysis.py -q -W error`. Expected: all pass.
- [ ] **Step 5: Lint and commit.**

```bash
git add tools/sharc_compare/run_sharc_analysis.py tests/sharc_compare/test_run_sharc_analysis.py
git commit -m "feat: drive SHARC's populations.py and make_fit.py for the P9b comparison" -m "Co-Authored-By: <trailer>"
```

---

### Task 6: The tau coverage study and the minimum-N warning

**Files:**
- Create: `tools/validation/tau_coverage.py`
- Test: `tests/validation/test_tau_coverage_tool.py`
- Modify: `src/gqteashmd/stages/analysis.py` (`SMALL_ENSEMBLE`, a new `SMALL_ENSEMBLE_COVERAGE` text, the warning, `version = 2`)
- Modify: `tests/stages/test_analysis.py` (the warning text, the version), and any test that matches `"fewer than 20"` (find with `grep -rn "fewer than 20\|SMALL_ENSEMBLE" tests src`).

**Interfaces:**
- Consumes: `analysis.kinetics.bootstrap_fit`, `analysis.populations.bootstrap_indices`, `gqteashmd.util.seeds.task_rng`.
- Produces: `coverage_once(n: int, rng) -> dict` (`covered: bool`, `tau_fs`, `interval`); `coverage(n: int, repetitions: int, seed_index: int, *, workers: int) -> dict`; `n_min(results: Mapping[int, float], threshold: float = 0.9) -> int | None`; CLI writing `~/p9b/tau_coverage.json`. In `stages/analysis.py`: `SMALL_ENSEMBLE: int` (the measured N_min), `SMALL_ENSEMBLE_COVERAGE: str`.

- [ ] **Step 1: Write the failing tool tests.**

```python
import numpy as np
from tests.diagnosis.conftest import load_tool

tc = load_tool("validation", "tau_coverage")


def test_constants_are_the_spec_values():
    assert tc.T0_FS == 10.0 and tc.TAU_FS == 40.0 and tc.DT_FS == 0.5 and tc.T_END_FS == 100.0
    assert tc.SIZES == (20, 50, 100) and tc.REPETITIONS == 400 and tc.N_BOOTSTRAP == 500
    assert tc.THRESHOLD == 0.90


def test_one_repetition_reports_an_interval():
    r = tc.coverage_once(100, np.random.default_rng(0), n_bootstrap=50)
    assert set(r) >= {"covered", "tau_fs", "interval"}
    assert r["interval"] is None or r["interval"][0] <= r["interval"][1]


def test_n_min():
    assert tc.n_min({20: 0.85, 50: 0.91, 100: 0.95}) == 50
    assert tc.n_min({20: 0.92, 50: 0.91, 100: 0.95}) == 20
    assert tc.n_min({20: 0.80, 50: 0.85, 100: 0.89}) is None
```

- [ ] **Step 2: Run them to see them fail.** Run: `pytest tests/validation/test_tau_coverage_tool.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3: Write the tool.**

```python
"""Coverage of the analysis' 95 % tau interval on synthetic ensembles (P9b spec 6).

Each repetition draws N first-passage times t0 + Exp(tau) (t0 10 fs, tau 40 fs), builds the
per-trajectory excited-state indicator on the 0.5 fs grid to 100 fs, and runs the analysis'
own ``bootstrap_fit`` with B = 500. Coverage = the fraction of repetitions whose interval holds
40 fs (a repetition without an interval counts as not covered). Report script: nothing is
asserted; the JSON goes to --out.

Example (WSL): python tools/validation/tau_coverage.py --out ~/p9b/tau_coverage.json --workers 8
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from gqteashmd.analysis.kinetics import bootstrap_fit
from gqteashmd.analysis.populations import bootstrap_indices
from gqteashmd.util.seeds import task_rng

T0_FS = 10.0
TAU_FS = 40.0
DT_FS = 0.5
T_END_FS = 100.0
SIZES = (20, 50, 100)
REPETITIONS = 400
N_BOOTSTRAP = 500
THRESHOLD = 0.90
SEED = 2028
STREAM = "p9b-coverage"
GRID = np.arange(0.0, T_END_FS + 1e-9, DT_FS)


def coverage_once(n: int, rng: np.random.Generator, *, n_bootstrap: int = N_BOOTSTRAP) -> dict:
    times = T0_FS + rng.exponential(TAU_FS, size=n)
    per = (GRID[None, :] < times[:, None]).astype(float)
    r = bootstrap_fit(GRID, per, bootstrap_indices(rng, n, n_bootstrap), dt_out=DT_FS)
    interval = r.tau_interval_fs
    covered = interval is not None and interval[0] <= TAU_FS <= interval[1]
    return {"covered": bool(covered), "tau_fs": r.fit.tau_fs, "interval": interval,
            "bootstrap_failures": r.bootstrap_failures}


def _chunk(args: tuple[int, int, int]) -> list[dict]:
    n, size_index, rep = args
    rng = task_rng(SEED, STREAM, size_index * REPETITIONS + rep)
    return [coverage_once(n, rng)]


def coverage(n: int, repetitions: int, size_index: int, *, workers: int) -> dict:
    jobs = [(n, size_index, rep) for rep in range(repetitions)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        rows = [r for chunk in pool.map(_chunk, jobs, chunksize=8) for r in chunk]
    covered = sum(r["covered"] for r in rows)
    p = covered / repetitions
    return {"n": n, "repetitions": repetitions, "covered": covered, "coverage": p,
            "standard_error": float(np.sqrt(p * (1 - p) / repetitions)),
            "no_interval": sum(r["interval"] is None for r in rows),
            "tau_median_fs": float(np.median([r["tau_fs"] for r in rows if r["tau_fs"]]))}


def n_min(results: Mapping[int, float], threshold: float = THRESHOLD) -> int | None:
    ok = [n for n in sorted(results) if results[n] >= threshold]
    return ok[0] if ok else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    started = time.perf_counter()
    rows = [coverage(n, REPETITIONS, i, workers=args.workers) for i, n in enumerate(SIZES)]
    result = {
        "model": {"t0_fs": T0_FS, "tau_fs": TAU_FS, "dt_fs": DT_FS, "t_end_fs": T_END_FS},
        "n_bootstrap": N_BOOTSTRAP, "threshold": THRESHOLD, "seed": SEED, "stream": STREAM,
        "sizes": rows,
        "n_min": n_min({r["n"]: r["coverage"] for r in rows}),
        "wall_time_s": time.perf_counter() - started,
    }
    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.out.expanduser().write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "sizes"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tool tests.** Run: `pytest tests/validation/test_tau_coverage_tool.py -q -W error`. Expected: 3 passed.
- [ ] **Step 5: The measurement.** Run (WSL, in the background, on the native filesystem): `python tools/validation/tau_coverage.py --out ~/p9b/tau_coverage.json --workers 8`. Record in `task-6-report.md`: the coverage, standard error and `no_interval` per N, `n_min`, the wall time. **Do not change any constant after seeing the numbers.**
- [ ] **Step 6: Write the failing analysis tests.** In `tests/stages/test_analysis.py`, change the expected warning to use `analysis.SMALL_ENSEMBLE` and `analysis.SMALL_ENSEMBLE_COVERAGE`, and the version test to `stage.version == 2`; add:

```python
def test_the_small_ensemble_threshold_is_the_measured_n_min():
    from gqteashmd.stages import analysis

    assert analysis.SMALL_ENSEMBLE == <N_MIN>  # P9b Task 6 measurement, ~/p9b/tau_coverage.json
    assert "coverage" in analysis.SMALL_ENSEMBLE_COVERAGE
```

  where `<N_MIN>` is the measured `n_min`, or 100 if it is `null` (spec 6). Run: `pytest tests/stages/test_analysis.py -q -W error`. Expected: FAIL (old threshold, version 1).
- [ ] **Step 7: Change `stages/analysis.py`.**

```python
# P9b (spec 6): the smallest of N = 20, 50, 100 whose 95 % tau interval covered the true tau in
# at least 90 % of 400 synthetic repetitions (tools/validation/tau_coverage.py)
SMALL_ENSEMBLE = <N_MIN>
SMALL_ENSEMBLE_COVERAGE = "<measured text, e.g. 0.86 at N = 20, 0.91 at N = 50, 0.93 at N = 100>"
```

  and the warning:

```python
            warnings.append(
                f"only {n} trajectories are included (fewer than {SMALL_ENSEMBLE}): populations, "
                "intervals and the fit are not statistically reliable (measured coverage of the "
                f"95 % tau interval: {SMALL_ENSEMBLE_COVERAGE})"
            )
```

  If `n_min` is `null`: `SMALL_ENSEMBLE = 100` and the text begins `"below 0.90 at every tested N: "`. Set `version = 2`. Update the `small_ensemble` comment in `_base_summary` (it already uses `SMALL_ENSEMBLE`).
- [ ] **Step 8: Run the affected tests.** Run: `pytest tests/stages/test_analysis.py tests/stages/test_analysis_pipeline.py tests/analysis tests/test_orchestrator.py -q -W error`, then `grep -rn "fewer than 20" tests src` (expected: no match). Expected: all pass.
- [ ] **Step 9: Lint, mypy, commit.** Run: `ruff format . && ruff check . && mypy src`.

```bash
git add tools/validation/tau_coverage.py tests/validation/test_tau_coverage_tool.py src/gqteashmd/stages/analysis.py tests/stages
git commit -m "feat: set the small-ensemble warning from the measured tau coverage" -m "Co-Authored-By: <trailer>"
```

---

### Task 7: The four runs (measurement)

**Files:**
- Create: `tools/sharc_compare/p9b/ethylene_seed2028.toml`, `tools/sharc_compare/p9b/ethylene_seed2028_nacdr.toml`, `tools/sharc_compare/p9b/variants_p9b.toml`
- Report (ledger): `task-7-report.md`

**Interfaces:**
- Consumes: Tasks 4 and 6 (the swap provider; the analysis version 2 runs in our projects).
- Produces: `~/p9b/ours/run` (project), `~/p9b/ours_nacdr/run`, `~/p9b/swap/project/traj_*`, `~/p9b/direct/project/traj_*`, each with its summary; the input files in the repository.

- [ ] **Step 1: The inputs.** `ethylene_seed2028.toml` = the P9a acceptance input (`~/p9a/acceptance/input.toml`) with `[project] name = "ethylene_p9b"`, `directory = "run"`, `seed = 2028`; `[sampling] n_samples = 300`; `[selection] n_trajectories = 100`, `states = ["S1"]`; `[dynamics] tmax_fs = 100.0`, `dt_fs = 0.5`; the same `[resources]`, `[electronic]` (SA3-CAS(2,2)/6-31G\*). `ethylene_seed2028_nacdr.toml` = the same plus `[dynamics] coupling = "nacdr"`. `variants_p9b.toml`:

```toml
# P9b (plan Task 7): the project's own settings, no overrides.
[[variant]]
name = "project"
description = "the seed-2028 project as it is"
```

  Commit the three files: `git commit -m "test: the P9b ethylene inputs" ...`.
- [ ] **Step 2: Our run.** `mkdir -p ~/p9b/ours && cp tools/sharc_compare/p9b/ethylene_seed2028.toml ~/p9b/ours/input.toml && cp tests/data/ethylene.xyz ~/p9b/ours/`; `cd ~/p9b/ours && gqteashmd run input.toml > run.log 2>&1` (background). If `selection` stops with fewer than 100 trajectories (O20), set `n_samples = 500` in the input (in the repository file as well) and rerun with `--rerun-from sampling` **before any dynamics** (spec 3.1); record it. Record the wall times per stage, the outcomes, the analysis summary.
- [ ] **Step 3: Ours-nacdr (PR2).** `cp -a ~/p9b/ours ~/p9b/ours_nacdr`, replace its `input.toml` by `ethylene_seed2028_nacdr.toml`, `gqteashmd status input.toml` (expected: every stage up to `selection` DONE), then `gqteashmd run input.toml --rerun-from dynamics`. Check before it runs that `cmp ~/p9b/ours/run/05_selection/trajectories.json ~/p9b/ours_nacdr/run/05_selection/trajectories.json` is silent (the directory name as `find_stage_dir` finds it). **Fallback** if `status` fails on the copy: a fresh project from the nacdr input, then assert `trajectories.json` byte-equal and `samples.h5` `positions`/`velocities` `np.array_equal` before its dynamics; record which path was taken.
- [ ] **Step 4: The swap ensemble.** `python tools/sharc_compare/run_sharc_pyscf_ensemble.py ~/p9b/ours/run tools/sharc_compare/p9b/variants_p9b.toml --variant project --provider swap --out ~/p9b/swap --sharc-home /home/ajc/sharc4-4.1 --python /home/ajc/miniforge3/envs/sharc4.1/bin/python --workers 8 --threads 2 --memory-mb 800` (background; resumable: rerun the same command after an interruption). Record the wall time and `sharc_summary.json` counts.
- [ ] **Step 5: The direct ensemble.** The same command with `--provider sharc_pyscf --out ~/p9b/direct` (about 15 h). Record as above.
- [ ] **Step 6: The report.** `task-7-report.md`: every command, start and end time, wall time, the outcome counts, anything unexpected. No comparison yet.

---

### Task 8: The replayed single trajectory (asserted)

**Files:**
- Create: `tests/data/ethylene_hop_start.json`
- Modify: `tests/validation/test_sharc_pyscf_dynamics.py` (a new section at its end, its own module docstring paragraph)

**Interfaces:**
- Consumes: `~/p9b/swap/project/traj_*` (Task 7), `run_sharc_pyscf_dynamics.run_sharc_trajectory`, `SwapModel`; `gqteashmd.trajectory.runner.TrajectoryRunner`, `RunSettings`, `InitialConditions`; `trajectory.model.BackendModel`; `dynamics.options.DynamicsOptions`; `dynamics.randomness.ReplaySource`; `trajectory.store.StepLog`.
- Produces: `tests/data/ethylene_hop_start.json` with keys `source` (the swap run directory and its task), `task`, `seed` (the swap run's `rngseed` = index + 1), `symbols`, `masses_amu`, `q_bohr` (flat), `v_au` (flat), `state` (`"S1"`), `equilibrium_bohr` (flat), `electronic` (the project's `[electronic]` as JSON), `first_hop_fs`.

- [ ] **Step 1: Choose the start by the rule (spec 5).** A throwaway script: for the swap trajectories in index order, read `output.dat` (Task 2's `read_sharc_trajectory`), take the first with an accepted hop at `time_fs <= 20.0`; write its start (from `load_project_inputs(~/p9b/ours/run)`) to the JSON. Then run the test of Step 2; if its "must be able to fail" assertions (a hop and `P(S0) >= 1e-3` inside the compared window) fail for this start, take the next start in index order that meets the hop rule and repeat. Record every start tried and why in `task-8-report.md`.
- [ ] **Step 2: Write the test.**

```python
# --- P9b: a replayed hopping-on trajectory (spec 5; plan Task 8) ------------------------------
# sharc.x on our backend (SwapModel) with hopping on; our TrajectoryRunner replays its random
# numbers (output.dat block 9 of steps 1, 2, ...). Both: SA3-CASSCF(2,2)/6-31G* from
# tests/data/ethylene_hop_start.json, overlap coupling, EDC, endpoint interpolation (PR3).
# Pre-registered bounds (spec 5): positions <= 1e-5 bohr, velocities <= 1e-6 au, populations
# and hopping probabilities <= 1e-5, the same active state; the window ends at the first step
# whose random number lies within 1e-5 of a cumulative hopping probability on either side,
# at our first rejected step (a halving would shift the replayed stream), or after 80 steps.

REPLAY_STEPS = 80
REPLAY_BOUNDS = {"q": 1e-5, "v": 1e-6, "populations": 1e-5, "probabilities": 1e-5}
NEAR_TIE = 1e-5
START = Path(__file__).resolve().parents[1] / "data" / "ethylene_hop_start.json"


def _near_tie(xi: float, probabilities: np.ndarray) -> bool:
    return bool(np.any(np.abs(np.cumsum(probabilities) - xi) < NEAR_TIE))


def _replay_backend(start, scratch):
    from gqteashmd.config.schema import ElectronicSection
    from gqteashmd.electronic.base import MoleculeSpec
    from gqteashmd.electronic.pyscf_backend import PySCFBackend

    scratch.mkdir(parents=True, exist_ok=True)
    return PySCFBackend(
        MoleculeSpec(tuple(start["symbols"]), 0, 1),
        ElectronicSection.model_validate(start["electronic"]),
        log_file=scratch / "electronic.log",
        threads=2,
        memory_mb=1500,
        scratch_dir=scratch,
        reference_coords_bohr=np.asarray(start["equilibrium_bohr"]).reshape(-1, 3),
    )


def test_replayed_hopping_on_trajectory(tmp_path):
    from gqteashmd.core.units import AU_TIME_IN_FS, DALTON_IN_ME
    from gqteashmd.dynamics.options import DynamicsOptions
    from gqteashmd.dynamics.randomness import ReplaySource
    from gqteashmd.trajectory.model import BackendModel
    from gqteashmd.trajectory.runner import InitialConditions, RunSettings, TrajectoryRunner
    from gqteashmd.trajectory.store import StepLog

    dyn = _load("run_sharc_pyscf_dynamics")
    start = json.loads(START.read_text())
    q0 = np.asarray(start["q_bohr"]).reshape(-1, 3)
    v0 = np.asarray(start["v_au"]).reshape(-1, 3)
    n_states = int(start["electronic"]["singlets"])
    sharc = dyn.run_sharc_trajectory(
        dyn.interface_geometry(q0), v0, n_steps=REPLAY_STEPS, workdir=tmp_path / "sharc",
        symbols=start["symbols"], masses_amu=start["masses_amu"],
        sharc_home=Path(os.environ["SHARC_HOME"]), provider="ours",
        model=dyn.SwapModel(_replay_backend(start, tmp_path / "b1"), coupling="overlap"),
        coupling="overlap", n_states=n_states, state_mch=1, hopping="sharc",
        decoherence="edc", seed=int(start["seed"]),
    )
    traj = sharc["trajectory"]
    dt_au = 0.5 / AU_TIME_IN_FS
    options = DynamicsOptions(
        dt_au=dt_au, nsubsteps=25, hopping="sharc", decoherence="edc",
        ekin_correction="parallel_vel", coupling="overlap", gradient_transformation="none",
        substep_interpolation="endpoint",
    )
    masses = np.repeat(np.asarray(start["masses_amu"]) * DALTON_IN_ME, 3)
    model = BackendModel(_replay_backend(start, tmp_path / "b2"), masses, coupling="overlap")
    TrajectoryRunner(model, RunSettings(options, tmax_au=REPLAY_STEPS * dt_au),
                     tmp_path / "ours").run(
        InitialConditions(dyn.interface_geometry(q0).reshape(-1), v0.reshape(-1), 1),
        ReplaySource(list(traj.random_numbers[1:])),
    )
    rows = StepLog(tmp_path / "ours" / "steps.log", n_states, q0.size).read()
    # the window: steps 1..end-1 are compared; it ends at a near tie, a rejection or step 80
    end = len(rows)
    for k in range(1, min(len(rows), len(traj.steps))):
        if int(rows["rejections"][k]) > 0 or int(rows["ticks"][k]) != 0:
            end = k
            break
        if _near_tie(float(traj.random_numbers[k]), traj.probabilities[k]) or _near_tie(
            float(rows["random_number"][k]), rows["probabilities"][k]
        ):
            end = k
            break
    ours_p = rows["c_real"][:end] ** 2 + rows["c_imag"][:end] ** 2
    sharc_p = np.abs(traj.c[:end]) ** 2
    diffs = {
        "q": np.abs(rows["q"][:end] - traj.q[:end]).max(),
        "v": np.abs(rows["v"][:end] - traj.v[:end]).max(),
        "populations": np.abs(ours_p - sharc_p).max(),
        "probabilities": np.abs(rows["probabilities"][1:end] - traj.probabilities[1:end]).max(),
    }
    hops = int(np.count_nonzero(np.diff(traj.active[:end])))
    print(f"\n[P9b replay] window {end} steps, hops {hops}, max P(S0) {sharc_p[:, 0].max():.3e}, "
          + ", ".join(f"d{k} {v:.2e}" for k, v in diffs.items()))
    assert hops >= 1, "no hop inside the compared window: the check cannot fail (spec 5)"
    assert sharc_p[:, 0].max() >= 1e-3, "P(S0) < 1e-3 in the window: the check cannot fail"
    assert np.array_equal(rows["active"][:end], traj.active[:end])
    for key, bound in REPLAY_BOUNDS.items():
        assert diffs[key] <= bound, (key, diffs[key], bound)
```

  Note: `traj.q` from `parse_lvc_output_dat` is the geometry sharc.x prints (bohr, flat per step); our runner starts from the same 7-decimal interface geometry, as in P7b. If the array layouts differ from this sketch, align them as `step_differences` does in `run_sharc_pyscf_dynamics.py`; do not change a bound.
- [ ] **Step 3: Run it.** Run: `SHARC_HOME=... SHARC_PYTHON=... pytest tests/validation/test_sharc_pyscf_dynamics.py -k replayed -q -s -W error`. Expected: pass; record the printed line, the window, the run time in `task-8-report.md`. A failing bound is reported (spec 8), not changed.
- [ ] **Step 4: Lint and commit.**

```bash
git add tests/data/ethylene_hop_start.json tests/validation/test_sharc_pyscf_dynamics.py
git commit -m "test: a replayed hopping-on trajectory against sharc.x" -m "Co-Authored-By: <trailer>"
```

---

### Task 9: The comparisons (measurement)

**Files:** none in the repository. Report (ledger): `task-9-report.md`; JSON in `~/p9b/compare/`.

**Interfaces:**
- Consumes: Tasks 3, 5, 7.
- Produces: `~/p9b/compare/swap.json` (asserted verdicts), `direct_vs_ours.json`, `direct_vs_ours_nacdr.json` (report), `tools.json` (section 4 verdicts).

- [ ] **Step 1: Swap (asserted).** Run: `python tools/sharc_compare/ensemble_compare.py ~/p9b/ours/run ~/p9b/swap/project --label swap --asserted --out ~/p9b/compare/swap.json`.
- [ ] **Step 2: Direct (report).** Run: `python tools/sharc_compare/ensemble_compare.py ~/p9b/ours/run ~/p9b/direct/project --label direct_vs_ours --out ~/p9b/compare/direct_vs_ours.json` and the same against `~/p9b/ours_nacdr/run` (`--label direct_vs_ours_nacdr`).
- [ ] **Step 3: SHARC's tools.** Run: `python tools/sharc_compare/run_sharc_analysis.py ~/p9b/swap/project ~/p9b/ours/run --sharc-home /home/ajc/sharc4-4.1 --python /home/ajc/miniforge3/envs/sharc4.1/bin/python --work ~/p9b/tools --out ~/p9b/compare/tools.json --quantum-bound <F5>`.
- [ ] **Step 4: Diagnosis of any fail** (spec 8): for a failed rule, the per-time differences and the trajectories that dominate them, and, where it helps, `tools/diagnosis/replay_points.py` on those trajectories. No new run with other settings or another seed.
- [ ] **Step 5: The report.** `task-9-report.md`: every verdict with its numbers, the report tables, the start-solution count of the direct runs, what was decided after numbers were seen.

---

### Task 10: The comparison report and the documents

**Files:**
- Create: `docs/superpowers/reports/YYYY-MM-DD-p9b-sharc-comparison.md` (dated the day it is written)
- Modify: `CLAUDE.md`, `USER_MANUAL.md`, `docs/superpowers/specs/2026-11-14-p9b-sharc-comparison-design.md` (an erratum block at its top: PR1-PR9 and anything that changed during the work)

- [ ] **Step 1: The report.** Sections: 1 what was run (commands, versions, times, machine); 2 the swap verdicts with every number (asserted); 3 the direct ensemble (report), with the start-solution and coupling analysis; 4 our analysis against `populations.py`/`make_fit.py` (verdicts); 5 the replay test; 6 the coverage study and N_min; 7 decisions for the owner (each failed or inconclusive item, with its diagnosis; any fix that a failure would need); 8 the honest summary (every decision made after numbers were seen).
- [ ] **Step 2: USER_MANUAL.md.** A new subsection (next number in section 8, "SHARC ensemble comparison (P9b)") with every number marked asserted or report; the analysis section: the new `small_ensemble` threshold, the warning text, `AnalysisStage` version 2 (existing projects need `--rerun-from analysis`); sections 10 and 11 (what is validated, what is not).
- [ ] **Step 3: CLAUDE.md.** "Read first" (the P9b spec, plan, ledger, report); Environment (the P9b test counts measured in the tasks); a P9b block; the status bullet; Known limitations; the deferred items resolved (a SHARC population check that can fail; the analysis against `populations.py`/`make_fit.py`; tau coverage and the minimum-N rule; molecular ensembles against SHARC) and the TDA/TDDFT root-skip detection moved to "Later (robustness)"; Remaining work.
- [ ] **Step 4: Check and commit.** `grep -n "fewer than 20" USER_MANUAL.md CLAUDE.md` (expected: only historical mentions).

```bash
git add docs/superpowers/reports CLAUDE.md USER_MANUAL.md docs/superpowers/specs/2026-11-14-p9b-sharc-comparison-design.md
git commit -m "docs: the P9b SHARC comparison report and documents" -m "Co-Authored-By: <trailer>"
```

---

## Self-review (done while writing)

- Spec coverage: 3.1 runs -> Task 7; 3.2 kept/paired -> Tasks 2, 3; 3.3 rules -> Task 3, verdicts Task 9; 3.4 direct -> Tasks 3 (start-difference table), 9; 4 -> Tasks 1, 5, 9; 5 -> Task 8 (PR9); 6 -> Task 6; 7 components -> Tasks 2-6, 8; 8 failures -> Tasks 8, 9, 10; 9 testing -> each task, the full suite before the final review (controller); 10 constraints -> Global Constraints; 11 documents -> Task 10.
- Facts not knowable before Task 1 (SHARC tool prompts, output names and formats, `make_fit.py` models) are produced by Task 1 and named by section (F1-F6) where they are used; every rule that depends on them is fixed in this plan or by F4/F5 before any comparison.
- Names: `is_kept`, `read_sharc_trajectory`, `read_our_trajectory`, `on_steps`, `compare`, `sharc_settings(..., provider=)`, `swap_backend`, `run_ensemble(..., provider=, threads=, memory_mb=)`, `SMALL_ENSEMBLE`, `SMALL_ENSEMBLE_COVERAGE` are used consistently across Tasks 2-9.
