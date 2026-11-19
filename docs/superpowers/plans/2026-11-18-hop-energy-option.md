# Hop Energy Option Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `hop_energy = "current" | "previous"` to the dynamics: `"previous"` reproduces SHARC 4.1's measured hop energy convention. Then assert the P9b replay with it up to 20 steps after the hop, and report its effect on our P9b ensemble.

**Architecture:** One new `DynamicsOptions` field changes only the energy gap passed to the existing `adjust_velocity` at a hop (`driver.step`, step 8). The default is bit-identical to today. Both input schemas pass the key through. The stage fingerprint includes it only when it is not the default. The validation consists of:
- unit tests on analytic models;
- a new live-SHARC replay test that shares the P9b SHARC run;
- a report script and a 2 h ensemble run.

**Tech Stack:** Python 3.12, NumPy, pydantic 2, pytest; PySCF and SHARC 4.1 (executed only) for the live test and the run.

**Spec:** `docs/superpowers/specs/2026-11-18-hop-energy-option-design.md` (owner-approved 2026-11-18).

## Global Constraints

- Run everything in WSL: `wsl -d OracleLinux_9_5 -- bash -lc 'cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && conda activate gqteashmd-dev && <command>'`. Edit files with the normal file tools, using LF line endings.
- Git from WSL only (the pre-commit hooks run there). Never use `--no-verify`. One commit per task, with conventional subjects. End each commit message with your model's `Co-Authored-By:` trailer.
- Clean room: SHARC 4.1 is executed only (from `tools/sharc_compare/`), and nothing under `/home/ajc/sharc4-4.1` is read. `gqteashmd` never imports SHARC or `tools/`.
- `src/gqteashmd/dynamics/` imports nothing from `electronic/`, `stages/`, the orchestrator or `config/`.
- **The default `hop_energy = "current"` must be bit-identical to `main`:** `tests/dynamics/test_golden.py` stays green unchanged, and no golden data is regenerated.
- The stage fingerprint gets `hop_energy` only when it is not `"current"`. `DynamicsStage.version` stays 2.
- Lint: `ruff format . && ruff check .` (line length 100); `mypy src` clean.
- Project runs and large outputs go to `~/hop/` in WSL, never to OneDrive or `/mnt/c`.
- Replay pre-registration (spec 4.2, fixed before any run):
  - the window is steps 0 to h + 20, where h is the first step at which SHARC's active state changes;
  - the window ends earlier at a near tie (1e-5) or at our first rejected step;
  - must be able to fail: h <= 59, P(S0) >= 1e-3 in the window, and the same active state at every step;
  - bounds: positions <= 1e-5 bohr, velocities <= 1e-6 au, populations <= 1e-5, hopping probabilities <= 1e-5;
  - also `|E_tot(h) - E_tot(h - 1)| <= 1e-9` Eh.
  - A failing bound is reported, never tuned.
- While the Task 5 run is going (about 2 h), do not switch the repository's branch or edit `src/`: the run uses the editable install from this checkout.

---

### Task 1: The option and the rule in the dynamics core

**Files:**
- Modify: `src/gqteashmd/dynamics/options.py`
- Modify: `src/gqteashmd/dynamics/driver.py` (module docstring item 8; `step`, step 8; new function `hop_delta_e`)
- Test: `tests/dynamics/test_options_records.py`, `tests/dynamics/test_driver.py`, `tests/dynamics/test_restart.py`

**Interfaces:**
- Produces:
  - `gqteashmd.dynamics.options.HOP_ENERGIES = ("current", "previous")`;
  - the `DynamicsOptions.hop_energy: str = "current"` field (the last field);
  - `gqteashmd.dynamics.driver.hop_delta_e(hop_energy: str, e_new: float, e_old: float, kinetic: float, e_total_previous: float) -> float`.

- [ ] **Step 1: Write the failing tests.**

In `tests/dynamics/test_options_records.py`, add:

```python
def test_hop_energy_defaults_to_current_and_is_validated():
    from gqteashmd.dynamics.options import HOP_ENERGIES

    assert HOP_ENERGIES == ("current", "previous")
    assert DynamicsOptions(dt_au=1.0).hop_energy == "current"
    assert DynamicsOptions(dt_au=1.0, hop_energy="previous").hop_energy == "previous"
    with pytest.raises(ValueError, match="hop_energy"):
        DynamicsOptions(dt_au=1.0, hop_energy="sharc")
```

(Use the file's existing imports of `DynamicsOptions` and `pytest`; add them if missing.)

In `tests/dynamics/test_driver.py`, extend the import to `from gqteashmd.dynamics.driver import Outcome, hop_delta_e, initial_state, run_trajectory, step` and add:

```python
def test_hop_delta_e_for_both_rules():
    # exact binary fractions, so the comparisons are exact
    assert hop_delta_e("current", 0.25, 0.125, 0.5, 9.0) == 0.125
    assert hop_delta_e("previous", 0.25, 0.125, 0.5, 0.625) == 0.125
    assert hop_delta_e("previous", 0.25, 0.125, 0.5, 0.5) == 0.25
    with pytest.raises(ValueError, match="hop_energy"):
        hop_delta_e("sharc", 0.25, 0.125, 0.5, 0.5)


@pytest.mark.parametrize(("ekin", "tol"), [("parallel_vel", 1e-12), ("parallel_nac", 1e-10)])
def test_previous_hop_energy_keeps_the_previous_record_energy(ekin, tol):
    """xi = 0 every step: a hop whenever a probability is positive (as the continuity test)."""
    model = TullySimpleAvoidedCrossing()
    state = _start(model, x0=-4.0, k=20.0)
    runs = {}
    for rule in ("current", "previous"):
        options = DynamicsOptions(
            dt_au=1.0, hopping="sharc", ekin_correction=ekin, hop_energy=rule
        )
        _, _, runs[rule] = _run(model, state, options, ReplaySource([0.0] * 2000), max_steps=800)
    prev = runs["previous"]
    hops = [i for i, r in enumerate(prev) if r.hop_to is not None]
    assert len(hops) >= 2
    energies = [state.e_total_initial] + [r.e_total for r in prev]
    for i in hops:
        assert not prev[i].frustrated
        assert abs(prev[i].e_total - energies[i]) <= tol, (i, prev[i].e_total - energies[i])
    # before the first hop the two rules give identical records; the first hop is the same step
    first = next(i for i, r in enumerate(runs["current"]) if r.hop_to is not None)
    assert first == hops[0]
    _assert_records_equal(runs["current"][:first], prev[:first])
    # the rule changes the hop itself: "current" keeps the Verlet-completed energy of the step
    assert prev[first].e_total != runs["current"][first].e_total


@pytest.mark.parametrize("frustrated", ["keep", "reflect"])
def test_previous_hop_energy_frustrated_hop(frustrated):
    """Too little kinetic energy for the upper state under either rule (K 0.004 < gap 0.018)."""
    model = TullySimpleAvoidedCrossing()
    options = DynamicsOptions(
        dt_au=1.0, hopping="sharc", frustrated=frustrated, hop_energy="previous"
    )
    state = _start(model, x0=-1.5, k=4.0)
    _, _, records = _run(model, state, options, ReplaySource([0.0] * 3000), max_steps=200)
    first = next(r for r in records if r.frustrated)
    assert first.hop_from is None and first.hop_to is None and first.active == 0
    assert all(r.active == 0 for r in records)
    assert (first.v[0] < 0.0) if frustrated == "reflect" else (first.v[0] > 0.0)
```

In `tests/dynamics/test_restart.py`, add:

```python
def test_restart_is_bit_identical_with_the_previous_hop_energy(tmp_path):
    model = TullySimpleAvoidedCrossing()
    options = DynamicsOptions(dt_au=2.0, hopping="sharc", decoherence="edc", hop_energy="previous")
    start = _sac_state(model)
    hops: list[int] = []
    full, _ = run_trajectory(
        model,
        start,
        options,
        GeneratorSource(np.random.default_rng(5)),
        max_steps=100,
        on_step=lambda r: hops.extend([r.step] if r.hop_to is not None else []),
    )
    assert any(h <= 50 for h in hops) and any(h > 50 for h in hops), hops
    rng = GeneratorSource(np.random.default_rng(5))
    half, _ = run_trajectory(model, start, options, rng, max_steps=50)
    path = tmp_path / "restart.npz"
    save_state(path, half, rng)
    loaded, rng2 = load_state(path, model)
    resumed, _ = run_trajectory(model, loaded, options, rng2, max_steps=50)
    _assert_same(resumed, full)
    assert np.array_equal(resumed.snapshot.energies, full.snapshot.energies)
```

If the `hops` assertion fails for seed 5 (with the default rule the hops are at steps 32 and 89), use the first seed from 6 upward that has hops on both sides of step 50. Use it in both `default_rng` calls, and write the seed and the hop steps in the report.

- [ ] **Step 2: Run them and see them fail.**

Run: `pytest tests/dynamics/test_options_records.py tests/dynamics/test_driver.py tests/dynamics/test_restart.py -q -W error -k "hop_energy or hop_delta_e"`
Expected: FAIL. The `hop_energy` field and `hop_delta_e` do not exist yet (`TypeError` / `ImportError`).

- [ ] **Step 3: Implement the option** (`options.py`).

After `GRADIENT_TRANSFORMATIONS = ("ngt", "none")`, add:

```python
HOP_ENERGIES = ("current", "previous")
```

Add the last field of `DynamicsOptions`, after `eselect_Eh`:

```python
    hop_energy: str = "current"
```

In `__post_init__`, after `_choice("gradient_transformation", ...)`, add:

```python
        _choice("hop_energy", self.hop_energy, HOP_ENERGIES)
```

- [ ] **Step 4: Implement the rule** (`driver.py`).

Import `HOP_ENERGIES` beside `DynamicsOptions`: `from gqteashmd.dynamics.options import HOP_ENERGIES, DynamicsOptions`. After `kinetic_energy`, add:

```python
def hop_delta_e(
    hop_energy: str, e_new: float, e_old: float, kinetic: float, e_total_previous: float
) -> float:
    """The dE given to ``adjust_velocity`` at a hop (item 8).

    ``"current"``: E^diag'_j - E^diag'_lambda, which keeps the total energy of the
    Verlet-completed step. ``"previous"``: E^diag'_j + K(v') - E_tot,prev, so an accepted
    ``parallel_vel`` hop ends with E_tot = E_tot,prev, the previous record's total energy
    (SHARC 4.1's convention as measured in P9b; for ``parallel_nac`` the same dE is our own
    definition).
    """
    if hop_energy not in HOP_ENERGIES:
        allowed = ", ".join(HOP_ENERGIES)
        raise ValueError(f"hop_energy must be one of {allowed}; got {hop_energy!r}")
    if hop_energy == "current":
        return float(e_new - e_old)
    return float(e_new + kinetic - e_total_previous)
```

In `step`, replace the line `delta_e = float(e_diag1[target] - e_diag1[lam])` with:

```python
        if options.hop_energy == "previous":
            # E_tot of the step's starting state = the previous record's E_tot (item 11)
            e_total_previous = float(
                diagonal_energies(snap0.hamiltonian, u0)[state.active]
            ) + kinetic_energy(v0, masses)
            delta_e = hop_delta_e(
                "previous", e_diag1[target], e_diag1[lam], kinetic_energy(v1, masses),
                e_total_previous,
            )
        else:
            delta_e = hop_delta_e("current", e_diag1[target], e_diag1[lam], 0.0, 0.0)
```

(`e_diag1[target] - e_diag1[lam]` on NumPy float64 values followed by `float` is the same arithmetic as before, so the default stays bit-identical.)

In the module docstring, item 8, change `adjust v' for dE = E^diag'_j - E^diag'_lambda (item 9;` to:
```
adjust v' for dE = E^diag'_j - E^diag'_lambda (``hop_energy = "current"``) or
   dE = E^diag'_j + K(v') - E_tot,prev (``"previous"``: the previous record's E_tot, recomputed
   from the state; ``hop_delta_e``) (item 9;
```

- [ ] **Step 5: Run the dynamics tests.**

Run: `pytest tests/dynamics -q -W error -m "not slow"`
Expected: PASS, including `test_golden.py` unchanged. Then: `ruff format . && ruff check . && mypy src`, all clean.

- [ ] **Step 6: Commit.**

```bash
git add src/gqteashmd/dynamics/options.py src/gqteashmd/dynamics/driver.py tests/dynamics/test_options_records.py tests/dynamics/test_driver.py tests/dynamics/test_restart.py
git commit -m "feat: hop_energy option with SHARC's previous-record convention" -m "Co-Authored-By: <your model trailer>"
```

---

### Task 2: The input key in both schemas, the fingerprint and the template

**Files:**
- Modify: `src/gqteashmd/config/schema.py` (`DynamicsStageSection`), `src/gqteashmd/config/model_dynamics.py` (`DynamicsSection`), `src/gqteashmd/stages/dynamics.py` (`run_settings`, `DynamicsStage.fingerprint_inputs`), `src/gqteashmd/cli.py` (the `init` template's commented `[dynamics]` block)
- Test: `tests/config/test_dynamics_section.py`, `tests/stages/test_dynamics.py`, `tests/test_cli_model_dynamics.py`

**Interfaces:**
- Consumes: `DynamicsOptions.hop_energy` (Task 1).
- Produces: the `[dynamics] hop_energy` key, `Literal["current", "previous"]`, default `"current"`, in both schemas. `run_settings(section).options.hop_energy` equals the key.

- [ ] **Step 1: Write the failing tests.**

In `tests/config/test_dynamics_section.py`:
- in `test_defaults` add `assert d.hop_energy == "current"`;
- add `('tmax_fs = 10.0\nhop_energy = "sharc"\n', "hop_energy"),` to the `test_invalid_values_name_the_key` parameter list.

In `tests/stages/test_dynamics.py` add:

```python
MAIN_DYNAMICS_FINGERPRINT_KEYS = {
    "dt_fs", "nsubsteps", "coupling", "hopping", "ekin_correction", "frustrated",
    "decoherence", "decoherence_param_Eh", "edc_convention", "substep_interpolation",
    "energy_drift_limit_eV", "energy_jump_limit_eV", "stop_after_s0_fs", "overlap_threshold",
    "max_step_halvings", "output_every",
}


def test_hop_energy_reaches_the_options_and_the_fingerprint_only_when_set(tmp_path, write_input):
    default = _context(tmp_path / "a", write_input)
    assert run_settings(default.config.dynamics).options.hop_energy == "current"
    fp_default = _stage().fingerprint_inputs(default)["dynamics"]
    # the default leaves the fingerprint as on main: projects made before the key stay up to date
    assert set(fp_default) == MAIN_DYNAMICS_FINGERPRINT_KEYS
    lines = 'tmax_fs = 2.0\ndt_fs = 0.25\nhop_energy = "previous"\n'
    previous = _context(tmp_path / "b", write_input, dynamics=lines)
    assert run_settings(previous.config.dynamics).options.hop_energy == "previous"
    fp_previous = _stage().fingerprint_inputs(previous)["dynamics"]
    assert fp_previous["hop_energy"] == "previous"
    assert {k: v for k, v in fp_previous.items() if k != "hop_energy"} == fp_default
```

In `tests/test_cli_model_dynamics.py` add:

```python
def test_hop_energy_reaches_the_library_options():
    from pydantic import ValidationError

    from gqteashmd.config.model_dynamics import DynamicsSection
    from gqteashmd.dynamics.options import DynamicsOptions

    section = DynamicsSection(hop_energy="previous")
    assert DynamicsOptions(**section.model_dump()).hop_energy == "previous"
    with pytest.raises(ValidationError, match="hop_energy"):
        DynamicsSection(hop_energy="sharc")
```

(The existing `test_dynamics_section_defaults_are_the_library_defaults` fails as soon as Task 1's field exists without this task's schema field, which is part of the RED.)

- [ ] **Step 2: Run them and see them fail.**

Run: `pytest tests/config/test_dynamics_section.py tests/stages/test_dynamics.py tests/test_cli_model_dynamics.py -q -W error`
Expected: FAIL. The new assertions fail, and so does the defaults test of `test_cli_model_dynamics.py` (`hop_energy` is missing from the section).

- [ ] **Step 3: Implement.**

`config/schema.py`, `DynamicsStageSection`: after `substep_interpolation`, add:

```python
    hop_energy: Literal["current", "previous"] = "current"
```

`config/model_dynamics.py`, `DynamicsSection`: after `eselect_Eh`, add:

```python
    hop_energy: Literal["current", "previous"] = _OPTION_DEFAULTS["hop_energy"]
```

`stages/dynamics.py`, `run_settings`: add `hop_energy=section.hop_energy,` after `coupling=section.coupling,`. In `fingerprint_inputs`, after `dynamics.pop("max_failed_fraction")`, add:

```python
        if dynamics.get("hop_energy") == "current":
            dynamics.pop("hop_energy")  # hashed only when set: projects made before it stay current
```

`cli.py`, the `init` template: after the line `# decoherence = "edc"`, add:

```
# hop_energy = "current"             # or "previous": SHARC 4.1's convention at a hop
```

- [ ] **Step 4: Run the tests.**

Run: `pytest tests/config tests/stages/test_dynamics.py tests/test_cli_model_dynamics.py tests/test_cli.py -q -W error -m "not slow and not pyscf"`
Expected: PASS. Then `ruff format . && ruff check . && mypy src`, all clean.

- [ ] **Step 5: Commit.**

```bash
git add src/gqteashmd/config/schema.py src/gqteashmd/config/model_dynamics.py src/gqteashmd/stages/dynamics.py src/gqteashmd/cli.py tests/config/test_dynamics_section.py tests/stages/test_dynamics.py tests/test_cli_model_dynamics.py
git commit -m "feat: the hop_energy key in [dynamics], hashed only when set" -m "Co-Authored-By: <your model trailer>"
```

---

### Task 3: The replay with the previous-record convention (live SHARC)

**Files:**
- Modify: `tests/validation/test_sharc_pyscf_dynamics.py` (the P9b replay section at its end: split the `replay` fixture so that the SHARC run is shared; add the new fixture and test; extend the section comment)

**Interfaces:**
- Consumes: `DynamicsOptions(hop_energy="previous")` (Task 1); the existing `_load`, `_near_tie`, `_replay_backend`, `REPLAY_STEPS`, `REPLAY_BOUNDS`, `START` of the file.
- Produces: the module fixtures `replay_sharc` and `replay_previous`, and the test `test_replayed_trajectory_with_the_previous_hop_energy`. The existing `replay` fixture and its two tests keep their behaviour, prints and results.

- [ ] **Step 1: Refactor the fixture, unchanged in behaviour.**

Replace the body of the existing `replay` fixture with a shared SHARC fixture and two helpers:

```python
@pytest.fixture(scope="module")
def replay_sharc(tmp_path_factory):
    """sharc.x on our backend, hopping on, run once for every replay test of this module."""
    tmp_path = tmp_path_factory.mktemp("replay")
    dyn = _load("run_sharc_pyscf_dynamics")
    start = json.loads(START.read_text())
    q0 = np.asarray(start["q_bohr"]).reshape(-1, 3)
    v0 = np.asarray(start["v_au"]).reshape(-1, 3)
    n_states = int(start["electronic"]["singlets"])
    sharc = dyn.run_sharc_trajectory(
        dyn.interface_geometry(q0),
        v0,
        n_steps=REPLAY_STEPS,
        workdir=tmp_path / "sharc",
        symbols=start["symbols"],
        masses_amu=start["masses_amu"],
        sharc_home=Path(os.environ["SHARC_HOME"]),
        provider="ours",
        model=dyn.SwapModel(_replay_backend(start, tmp_path / "b1"), coupling="overlap"),
        coupling="overlap",
        n_states=n_states,
        state_mch=1,
        hopping="sharc",
        decoherence="edc",
        seed=int(start["seed"]),
    )
    return {
        "dyn": dyn, "start": start, "q0": q0, "v0": v0, "n_states": n_states,
        "traj": sharc["trajectory"], "tmp": tmp_path,
    }


def _replay_ours(shared, hop_energy, name):
    """Our runner on the same start, replaying SHARC's random numbers; returns the step log."""
    from gqteashmd.core.units import AU_TIME_IN_FS, DALTON_IN_ME
    from gqteashmd.dynamics.options import DynamicsOptions
    from gqteashmd.dynamics.randomness import ReplaySource
    from gqteashmd.trajectory.model import BackendModel
    from gqteashmd.trajectory.runner import InitialConditions, RunSettings, TrajectoryRunner
    from gqteashmd.trajectory.store import StepLog

    dyn, start, tmp = shared["dyn"], shared["start"], shared["tmp"]
    q0, v0 = shared["q0"], shared["v0"]
    dt_au = 0.5 / AU_TIME_IN_FS
    options = DynamicsOptions(
        dt_au=dt_au,
        nsubsteps=25,
        hopping="sharc",
        decoherence="edc",
        ekin_correction="parallel_vel",
        coupling="overlap",
        gradient_transformation="none",
        substep_interpolation="endpoint",
        hop_energy=hop_energy,
    )
    masses = np.repeat(np.asarray(start["masses_amu"]) * DALTON_IN_ME, 3)
    model = BackendModel(_replay_backend(start, tmp / f"{name}_backend"), masses, coupling="overlap")
    TrajectoryRunner(model, RunSettings(options, tmax_au=REPLAY_STEPS * dt_au), tmp / name).run(
        InitialConditions(dyn.interface_geometry(q0).reshape(-1), v0.reshape(-1), 1),
        ReplaySource(list(shared["traj"].random_numbers[1:])),
    )
    return StepLog(tmp / name / "steps.log", shared["n_states"], q0.size).read()


def _natural_end(rows, traj):
    """Spec 5 window end: a near tie, our first rejected step or substep record, or step 80."""
    end = min(len(rows), len(traj.steps))
    for k in range(1, end):
        if int(rows["rejections"][k]) > 0 or int(rows["ticks"][k]) != 0:
            return k
        if _near_tie(float(traj.random_numbers[k]), traj.probabilities[k]) or _near_tie(
            float(rows["random_number"][k]), rows["probabilities"][k]
        ):
            return k
    return end


def _per_record(rows, traj, end):
    ours_p = rows["c_real"][:end] ** 2 + rows["c_imag"][:end] ** 2
    sharc_p = np.abs(traj.c[:end]) ** 2
    dprob = np.abs(rows["probabilities"][1:end] - traj.probabilities[1:end]).max(axis=1)
    return {
        "q": np.abs(rows["q"][:end] - traj.q[:end]).max(axis=1),
        "v": np.abs(rows["v"][:end] - traj.v[:end]).max(axis=1),
        "populations": np.abs(ours_p - sharc_p).max(axis=1),
        # step 0 has no propagation, so the probabilities are compared from step 1 on
        "probabilities": np.concatenate([[0.0], dprob]),
    }, sharc_p
```

Then rewrite `replay(tmp_path_factory)` as `replay(replay_sharc)`. The new body:
- `rows = _replay_ours(replay_sharc, "current", "ours")`, `traj = replay_sharc["traj"]`;
- `end = _natural_end(rows, traj)`;
- `per_record, sharc_p = _per_record(rows, traj, end)`;
- `ours_p = rows["c_real"][:end] ** 2 + rows["c_imag"][:end] ** 2`, which the step print lines use;
- then the same `changes`, `hops`, `first_hop`, `diffs`, `before`, the same three `print` blocks and the same returned dict as today.

The P9b behaviour, numbers and prints must not change. Keep its docstring.

- [ ] **Step 2: The new fixture and test.**

Append after `test_replayed_trajectory_over_the_pre_registered_window`:

```python
# --- hop_energy = "previous" (spec 2026-11-18-hop-energy-option-design.md, 4.2) ---------------
# The same SHARC run; ours with SHARC's hop energy convention. Pre-registered before any run:
# the window is steps 0 .. h + 20 (h = the first step at which SHARC's active state changes), or
# shorter at a near tie or our first rejected step; the P9b bounds; h <= 59, P(S0) >= 1e-3 in
# the window, the same active state; |E_tot(h) - E_tot(h - 1)| <= 1e-9 Eh on our side.

HOP_WINDOW_AFTER = 20
HOP_ENERGY_STEP_BOUND_EH = 1e-9


@pytest.fixture(scope="module")
def replay_previous(replay_sharc):
    traj = replay_sharc["traj"]
    rows = _replay_ours(replay_sharc, "previous", "ours_previous")
    natural = _natural_end(rows, traj)
    changes = np.flatnonzero(np.diff(traj.active[:natural])) + 1
    h = int(changes[0]) if changes.size else None
    end = natural if h is None else min(natural, h + HOP_WINDOW_AFTER + 1)
    per_record, sharc_p = _per_record(rows, traj, end)
    diffs = {k: float(v.max()) for k, v in per_record.items()}
    full_end = min(len(rows), len(traj.steps))
    full, _ = _per_record(rows, traj, full_end)
    e_step = None if h is None else abs(float(rows["e_total"][h]) - float(rows["e_total"][h - 1]))
    print(
        f"\n[hop_energy previous] hop step {h}, window {end} records (natural end {natural}), "
        f"max P(S0) {sharc_p[:, 0].max():.3e}, |E_tot(h) - E_tot(h-1)| {e_step}, "
        + ", ".join(f"d{k} {v:.2e}" for k, v in diffs.items())
    )
    print(
        f"[hop_energy previous] all {full_end} records: "
        + ", ".join(f"d{k} {float(v.max()):.2e}" for k, v in full.items())
    )
    if h is not None:
        for k in range(h - 1, min(h + HOP_WINDOW_AFTER + 1, full_end)):
            print(
                f"[hop_energy previous] step {k}: "
                + " ".join(f"d{key} {float(v[k]):.2e}" for key, v in full.items())
                + f" E_tot ours {float(rows['e_total'][k]):.10f}"
            )
    return {
        "hop": h,
        "end": end,
        "max_p_s0": float(sharc_p[:, 0].max()),
        "active_ours": np.asarray(rows["active"][:end]),
        "active_sharc": np.asarray(traj.active[:end]),
        "diffs": diffs,
        "hop_energy_step_Eh": e_step,
    }


def test_replayed_trajectory_with_the_previous_hop_energy(replay_previous):
    """Spec 4.2: the P9b bounds from step 0 through 20 steps after the hop, with hop_energy =
    "previous"; the window was fixed from P9b numbers already seen (stated in the spec)."""
    r = replay_previous
    assert r["hop"] is not None and r["hop"] <= 59, f"no usable hop in the window: {r['hop']}"
    assert r["max_p_s0"] >= 1e-3, "P(S0) < 1e-3 in the window: the check cannot fail"
    assert np.array_equal(r["active_ours"], r["active_sharc"]), "active states differ"
    assert r["hop_energy_step_Eh"] <= HOP_ENERGY_STEP_BOUND_EH, r["hop_energy_step_Eh"]
    for key, bound in REPLAY_BOUNDS.items():
        value = r["diffs"][key]
        assert value <= bound, f"{key} up to h + {HOP_WINDOW_AFTER}: {value:.3e} > {bound:.0e}"
```

Add a sentence about the new test to the module docstring's P9b paragraph.

- [ ] **Step 3: Run the replay tests (live SHARC, about 10 min).**

Run: `SHARC_HOME=/home/ajc/sharc4-4.1 SHARC_PYTHON=/home/ajc/miniforge3/envs/sharc4.1/bin/python pytest tests/validation/test_sharc_pyscf_dynamics.py -k replayed -q -s -W error --basetemp=$HOME/hop/task3/tmp 2>&1 | tee ~/hop/task3/run.log`
Expected:
- `test_replayed_trajectory_agrees_before_the_hop` passes;
- `test_replayed_trajectory_over_the_pre_registered_window` xfails, with the same P9b prints;
- the new test passes.

Save the full output (`run.log`) and put every `[hop_energy previous]` line in the report.

**If the new test fails:** change nothing (no bound, window, start or setting). Report DONE_WITH_CONCERNS with the numbers. The controller then rules (spec 4.2: a strict xfail with the measured numbers).

- [ ] **Step 4: Lint and commit.**

`ruff format . && ruff check .`, then:

```bash
git add tests/validation/test_sharc_pyscf_dynamics.py
git commit -m "test: the replay with hop_energy = previous against sharc.x" -m "Co-Authored-By: <your model trailer>"
```

---

### Task 4: The ensemble report script

**Files:**
- Create: `tools/validation/hop_energy_report.py`
- Test: `tests/validation/test_hop_energy_report.py`

**Interfaces:**
- Consumes (tools, through `sys.path` as `ensemble_compare.py` does): `sharc_trajectories.is_kept`, `sharc_trajectories.read_our_trajectory`, `forensics.find_stage_dir`, `forensics.jsonable`; `gqteashmd.analysis.grid.INCLUDED_OUTCOMES`, `Trajectory`; `gqteashmd.core.units.HARTREE_IN_EV`.
- Produces: the CLI `python tools/validation/hop_energy_report.py CURRENT_RUN PREVIOUS_RUN --out JSON` and the functions `record_energy_changes_eV(t) -> tuple[np.ndarray, np.ndarray]`, `lost_after_hop(t) -> bool`, `quantiles(values) -> dict`, `summarize(run_dir) -> dict`, `outcome_changes(a_rows, b_rows) -> list[dict]`.

- [ ] **Step 1: Write the failing tests** (`tests/validation/test_hop_energy_report.py`).

```python
import numpy as np

from gqteashmd.analysis.grid import Trajectory
from gqteashmd.core.units import HARTREE_IN_EV
from tests.diagnosis.conftest import load_tool

hr = load_tool("validation", "hop_energy_report")


def _traj(outcome, e_total, hop_to):
    n = len(e_total)
    return Trajectory(
        task="traj_00000",
        outcome=outcome,
        initial_state=1,
        grid_step=np.arange(n),
        ticks=np.zeros(n, dtype=np.int64),
        time_fs=0.5 * np.arange(n),
        active=np.ones(n, dtype=np.int64),
        hop_from=np.full(n, -1),
        hop_to=np.asarray(hop_to),
        frustrated=np.zeros(n, dtype=bool),
        e_total=np.asarray(e_total, dtype=float),
        populations=np.zeros((n, 3)),
    )


def test_record_energy_changes_split_by_hop():
    t = _traj("finished", [0.0, 1e-3, 1e-3, 4e-3], [-1, -1, 0, -1])
    hop, other = hr.record_energy_changes_eV(t)
    assert np.allclose(hop, [0.0])
    assert np.allclose(other, [1e-3 * HARTREE_IN_EV, 3e-3 * HARTREE_IN_EV])


def test_lost_after_hop():
    assert hr.lost_after_hop(_traj("overlap_failure", [0.0, 0.0], [-1, 0]))
    assert not hr.lost_after_hop(_traj("overlap_failure", [0.0, 0.0], [-1, -1]))
    assert not hr.lost_after_hop(_traj("finished", [0.0, 0.0], [-1, 0]))


def test_quantiles():
    q = hr.quantiles([1.0, 2.0, 3.0, 4.0])
    assert q["n"] == 4 and q["median"] == 2.5 and q["max"] == 4.0
    assert q["p95"] == float(np.percentile([1.0, 2.0, 3.0, 4.0], 95))
    assert hr.quantiles([]) == {"n": 0, "median": None, "p95": None, "max": None}


def test_outcome_changes():
    a = [{"task": "t0", "outcome": "finished"}, {"task": "t1", "outcome": "overlap_failure"}]
    b = [{"task": "t0", "outcome": "finished"}, {"task": "t1", "outcome": "finished"}]
    assert hr.outcome_changes(a, b) == [
        {"task": "t1", "current": "overlap_failure", "previous": "finished"}
    ]
```

- [ ] **Step 2: Run them and see them fail.**

Run: `pytest tests/validation/test_hop_energy_report.py -q -W error`
Expected: FAIL (the tool does not exist).

- [ ] **Step 3: Write the tool.**

```python
"""Hop energy option report: our P9b ensemble with hop_energy = "current" against the same
ensemble rerun with "previous" (spec 2026-11-18-hop-energy-option-design.md, section 4.3).

Report only: every number goes to the JSON file and nothing is asserted. Never imported by
gqteashmd. Usage: hop_energy_report.py CURRENT_RUN PREVIOUS_RUN --out JSON (project dirs).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sharc_compare"))
sys.path.insert(0, str(HERE.parent / "diagnosis"))
from forensics import find_stage_dir, jsonable  # noqa: E402
from sharc_trajectories import is_kept, read_our_trajectory  # noqa: E402

from gqteashmd.analysis.grid import INCLUDED_OUTCOMES, Trajectory  # noqa: E402
from gqteashmd.core.units import HARTREE_IN_EV  # noqa: E402


def record_energy_changes_eV(t: Trajectory) -> tuple[np.ndarray, np.ndarray]:
    """|dE_tot| (eV) from each record to the next, split by whether the later one is a hop."""
    if t.e_total.size < 2:
        return np.empty(0), np.empty(0)
    d = np.abs(np.diff(t.e_total)) * HARTREE_IN_EV
    hop = np.asarray(t.hop_to[1:]) >= 0
    return d[hop], d[~hop]


def lost_after_hop(t: Trajectory) -> bool:
    """Not included by the analysis, with at least one accepted hop before its end."""
    return t.outcome not in INCLUDED_OUTCOMES and bool(np.any(np.asarray(t.hop_to) >= 0))


def quantiles(values: Sequence[float]) -> dict[str, Any]:
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return {"n": 0, "median": None, "p95": None, "max": None}
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "max": float(v.max()),
    }


def outcome_changes(
    current: Sequence[dict[str, Any]], previous: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    after = {r["task"]: r["outcome"] for r in previous}
    return [
        {"task": r["task"], "current": r["outcome"], "previous": after.get(r["task"])}
        for r in current
        if after.get(r["task"]) != r["outcome"]
    ]


def summarize(run_dir: Path) -> dict[str, Any]:
    dyn = find_stage_dir(run_dir, "dynamics")
    summary = json.loads((dyn / "dynamics_summary.json").read_text())
    analysis = json.loads((find_stage_dir(run_dir, "analysis") / "analysis.json").read_text())
    n_total = round(summary["tmax_fs"] / summary["dt_fs"])
    rows = summary["trajectories"]
    trajs = [read_our_trajectory(dyn / "tasks" / r["task"] / "output.h5", r["task"]) for r in rows]
    hop_d: list[float] = []
    other_d: list[float] = []
    for t in trajs:
        if t is not None:
            a, b = record_energy_changes_eV(t)
            hop_d.extend(a.tolist())
            other_d.extend(b.tolist())
    fit = analysis.get("fit") or {}
    return {
        "run": str(run_dir),
        "outcomes": summary["outcomes"],
        "totals": summary["totals"],
        "n_kept": sum(is_kept(t, n_total) for t in trajs),
        "n_missing_output": sum(t is None for t in trajs),
        "n_lost": sum(t is not None and t.outcome not in INCLUDED_OUTCOMES for t in trajs),
        "n_lost_after_hop": sum(t is not None and lost_after_hop(t) for t in trajs),
        "max_drift_eV": quantiles(
            [r["max_drift_eV"] for r in rows if r.get("max_drift_eV") is not None]
        ),
        "record_dE_eV_at_hops": quantiles(hop_d),
        "record_dE_eV_elsewhere": quantiles(other_d),
        "analysis": {
            "n_selected": analysis.get("n_selected"),
            "n_included": analysis.get("n_included"),
            "small_ensemble": analysis.get("small_ensemble"),
            "warnings": analysis.get("warnings"),
            "t0_fs": fit.get("t0_fs"),
            "t0_interval_fs": fit.get("t0_interval_fs"),
            "tau_fs": fit.get("tau_fs"),
            "tau_interval_fs": fit.get("tau_interval_fs"),
        },
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("current", type=Path, help='the run with hop_energy = "current" (P9b ours)')
    p.add_argument("previous", type=Path, help='the run with hop_energy = "previous"')
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    current, previous = summarize(args.current), summarize(args.previous)
    result = {
        "current": {k: v for k, v in current.items() if k != "rows"},
        "previous": {k: v for k, v in previous.items() if k != "rows"},
        "outcome_changes": outcome_changes(current["rows"], previous["rows"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(jsonable(result), indent=2))
    print(json.dumps({k: result[k]["n_kept"] for k in ("current", "previous")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests and try the tool on the P9b run against itself.**

Run: `pytest tests/validation/test_hop_energy_report.py -q -W error`. Expected: PASS.
Then: `python tools/validation/hop_energy_report.py ~/p9b/ours/run ~/p9b/ours/run --out ~/hop/task4_selftest.json`. Expected: `{"current": 67, "previous": 67}` (P9b: kept 67), with `outcome_changes` empty. Write the printed line and the `current` block's `max_drift_eV`, `record_dE_eV_at_hops` and `record_dE_eV_elsewhere` in the report.

- [ ] **Step 5: Lint and commit.**

`ruff format . && ruff check .`, then:

```bash
git add tools/validation/hop_energy_report.py tests/validation/test_hop_energy_report.py
git commit -m "test: the hop energy ensemble report script" -m "Co-Authored-By: <your model trailer>"
```

---

### Task 5: The ensemble run and the reports (measurement)

**Files:** none in the repository. Report (ledger): `task-5-report.md`; runs and JSON in `~/hop/` in WSL.

**Interfaces:**
- Consumes: Tasks 1, 2 and 4; `~/p9b/ours` (the P9b `overlap` run, input and project); `~/p9b/swap/project`; `tools/sharc_compare/ensemble_compare.py`.
- Produces: `~/hop/ours_previous/run` (the project), `~/hop/report.json`, `~/hop/swap_previous.json`.

- [ ] **Step 1: The copy.**
  1. `mkdir -p ~/hop && cp -a ~/p9b/ours ~/hop/ours_previous`.
  2. In `~/hop/ours_previous/input.toml`, add the line `hop_energy = "previous"` to the `[dynamics]` table and change nothing else.
  3. Check: `cd ~/hop/ours_previous && gqteashmd validate input.toml && gqteashmd status input.toml`. Both should succeed.
  4. `cmp ~/p9b/ours/input.toml ~/hop/ours_previous/input.toml` should report a difference, and `diff` should show exactly the one added line.

- [ ] **Step 2: The run (detached, about 2 h).**

`cd ~/hop/ours_previous && setsid nohup gqteashmd run input.toml --rerun-from dynamics > run.log 2>&1 < /dev/null &`

Wait in bounded stretches, then check that `run.log` ends with exit 0 (`echo $?` cannot be used on a detached run; read the log's final summary and `gqteashmd status input.toml`).
- Record the stage start and end times from `run.log`.
- Record that `06_dynamics` and `07_analysis` of the copy were moved to `.gqteashmd/superseded/`.
- On an interruption, resume with a plain `gqteashmd run input.toml` (P9b ruling P4) and record it.

- [ ] **Step 3: The reports.**

From the repository root:
1. `python tools/validation/hop_energy_report.py ~/p9b/ours/run ~/hop/ours_previous/run --out ~/hop/report.json`
2. `python tools/sharc_compare/ensemble_compare.py ~/hop/ours_previous/run ~/p9b/swap/project --label swap_previous --out ~/hop/swap_previous.json`

The second is a report: no `--asserted`, nothing decided on it.

- [ ] **Step 4: The report** (`task-5-report.md`).
  - The run: times, interruptions, the superseded directories.
  - A table, current against previous, with:
    - outcomes;
    - kept;
    - hops and frustrated hops;
    - `rejections_by_signal`;
    - `max_drift_eV` (median, p95, max);
    - record |dE| at hops and elsewhere (median, p95, max);
    - losses and losses after a hop;
    - the analysis (N, excluded fraction, t0 and tau with intervals, warnings).
  - The list of outcome changes.
  - The four swap-rule verdicts with their numbers, next to P9b's.
  - Every number from the JSON, with the files named. Nothing re-tuned, no other seed or settings.

---

### Task 6: The report and the documents

**Files:**
- Create: `docs/superpowers/reports/<date>-hop-energy-option.md` (dated the day it is written)
- Modify: `USER_MANUAL.md`, `CLAUDE.md`, `docs/superpowers/specs/2026-11-18-hop-energy-option-design.md` (an erratum block at its top for anything that changed during the work; "none" if nothing did)

- [ ] **Step 1: The report.** Sections:
  1. what was built (the rule, the key, the fingerprint rule);
  2. the replay (asserted), with the Task 3 numbers and the per-step lines from h - 1 to h + 20;
  3. the ensemble (report), with the Task 5 table and the swap-rule verdicts;
  4. what was decided after numbers were seen;
  5. the honest summary.

  Mark every number asserted or report.
- [ ] **Step 2: `USER_MANUAL.md`.**
  - The `[dynamics]` key table (section 4.2) and the `model-dynamics` `[dynamics]` keys (section 5.7): `hop_energy`, its two values, the default, and that it is hashed only when set. Changing it needs `--rerun-from dynamics`.
  - The dynamics method section that states the hop and velocity adjustment: the `"previous"` rule, and that it is our definition for `parallel_nac`.
  - A new subsection at the end of section 8, "The hop energy option", with every number marked asserted or report.
  - Section 8.17 (P9b): one sentence pointing to it from the replay paragraph.
  - Sections 10 and 11: what is now validated after the hop and what is not.
- [ ] **Step 3: `CLAUDE.md`.**
  - "Read first": the spec, the plan, the ledger and the report.
  - Environment: the test counts measured in Tasks 1-4.
  - The P7a-1 rules (one line on the option, which leaves the step order unchanged).
  - A short block for this sub-project; the status bullet; Known limitations item 1c (the hop energy line); Remaining work (the post-hop losses next).
- [ ] **Step 4: Check and commit.** Run `grep -n "hop_energy" USER_MANUAL.md CLAUDE.md` and confirm the key is in both.

```bash
git add docs/superpowers/reports CLAUDE.md USER_MANUAL.md docs/superpowers/specs/2026-11-18-hop-energy-option-design.md
git commit -m "docs: the hop energy option report and documents" -m "Co-Authored-By: <your model trailer>"
```

---

## Self-review (done while writing)

- **Spec coverage:**
  - spec 2 (the rule) -> Task 1;
  - spec 3 (input, fingerprint, model-dynamics, template) -> Task 2;
  - 4.1 -> Tasks 1 and 2;
  - 4.2 -> Task 3;
  - 4.3 -> Tasks 4 and 5;
  - 4.4: the full suite runs before the final review (controller);
  - 5 (components) -> Tasks 1-4;
  - 6 (documents) -> Task 6. The P9b report section 7 was already done in the spec commit `a9e70e5`.
- **Names used consistently across tasks:** `HOP_ENERGIES`, `hop_energy`, `hop_delta_e`, `replay_sharc`, `_replay_ours`, `_natural_end`, `_per_record`, `replay_previous`, `HOP_WINDOW_AFTER`, `HOP_ENERGY_STEP_BOUND_EH`, `record_energy_changes_eV`, `lost_after_hop`, `quantiles`, `outcome_changes`, `summarize`.
- **Facts checked against the code while writing:**
  - `diagonal_energies` is the definition the record's E^diag uses, so with U `None` E_tot,prev equals the previous record's `e_total` bit for bit;
  - `DynamicsSection` takes its defaults from `DynamicsOptions`, and `test_dynamics_section_defaults_are_the_library_defaults` enforces it;
  - `fingerprint_inputs` dumps the whole section;
  - the P9b summary has `outcomes`, `totals`, `trajectories[*].max_drift_eV`;
  - `analysis.json` has `fit`, `n_included`, `n_selected`.
