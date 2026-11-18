# The hop energy option: design

Status: written 2026-11-18; owner approval pending. Branch `feat/hop-energy-option` (from `main` `4e61daa`).

## 0. Owner decisions (2026-11-18)

These follow the P9b report, section 7. The owner took the recommendation of each item; they are recorded in that section.

1. **The hop energy convention:** option (c). The current rule stays the default; SHARC's convention is added as an option.
2. **The replay's pass rule:** the original P9b bounds, asserted from step 0 through **20 steps after the first hop**. The full 80 steps are reported, and the existing strict xfail on the default rule stays.
3. **An ensemble report:** our P9b ensemble is rerun with the option on, as a report (nothing asserted).
4. **The rest of P9b section 7:** the direct comparison is closed; the tau/t0 passes are accepted as measured; N_min stays 50; the losses after a hop become the **next sub-project**, after this one; the smaller items are accepted; `vmIdleTimeout=-1` was set in `.wslconfig` (2026-11-18).

## 1. Why

P9b measured how the two programs set the energy at a hop (P9b report, section 5):

- **SHARC 4.1:** the total energy after a hop equals the **previous record's** total energy, at all 221 swap hops (|dE_tot| <= 9.71e-12 Eh against a non-hop median of 8.31e-5). That this is SHARC's rule is an inference.
- **Ours:** the velocities are rescaled so that the total energy of the **current, Verlet-completed** step is kept: `K' = K(v') - dE`, `dE = E_new(q') - E_old(q')` (method reference 6).

The two differ by one step's integration error, about 1e-4 Eh per hop. At the replay's hop it was 1.71e-4 Eh in E_tot and 6.5e-4 in the velocity scale, and it is the whole difference after the hop. As a result, nothing checks our post-hop dynamics against SHARC at the single-trajectory level. Every loss of ours in P9b happens after a hop.

Neither rule is wrong. Ours is the textbook FSSH rule; SHARC's removes one step's integration error at each hop. The option exists so that the post-hop dynamics can be compared with SHARC, and so that the rule's effect on an ensemble can be measured.

## 2. The rule

A new field of `DynamicsOptions`: `hop_energy`, one of `HOP_ENERGIES = ("current", "previous")`, default `"current"`. Any other value is a `ValueError` naming the option.

At a hop attempt to state `j` from the active state `lambda` (step 8 of the step order, `driver.step`):

- `"current"` (unchanged): `delta_e = E_diag'_j - E_diag'_lambda`.
- `"previous"`: `delta_e = E_diag'_j + K(v') - E_tot,prev`.
  - `v'` is the Verlet-completed velocity before the adjustment.
  - `E_tot,prev` is the total energy of the step's starting state: `E_diag[lambda] + K(v)`, with `E_diag` the diagonal energies of the state's snapshot in its U (`diagonal_energies(snapshot.hamiltonian, u)`) and `lambda`, `v` the state's. It equals the previous record's `e_total` (exactly for U `None`, to round-off otherwise). At the first step it is E_tot(0).

`adjust_velocity` is called with that `delta_e` and is not changed. So with `parallel_vel` an accepted hop gives `K' = E_tot,prev - E_diag'_j`, and the record's `E_tot` equals `E_tot,prev`. A hop that would need `K' < 0` is frustrated and handled by `frustrated` (keep or reflect), as today.

- **`parallel_nac`** uses the same `delta_e`. That is our own definition: P9b measured SHARC with `parallel_vel` only. The documents say so.
- **SOC, `eselect`, both couplings:** the rule is stated in the diagonal basis, so it applies unchanged. No SOC case was compared with SHARC.
- **Unchanged:** the step order, the one random number per step, the restart format (options are not stored), the record fields, and every result with the default.

## 3. Input

- **`[dynamics]` of a project** (`config/schema.py:DynamicsStageSection`): `hop_energy = "current" | "previous"`, default `"current"`, passed through `stages/dynamics.py:run_settings`.
- **The stage fingerprint** keeps the key **only when it is not `"current"`** (`fingerprint_inputs` pops it otherwise). Existing projects stay up to date and a project with the default gets no new hash; changing the key is stale (exit 3, `--rerun-from dynamics`). The pattern is the one of `selection.weighting` in fixed mode. `DynamicsStage.version` stays 2.
- **`model-dynamics` `[dynamics]`** (`config/model_dynamics.py:DynamicsSection`, Tully and LVC): the same key, default from `DynamicsOptions`.
- The `init` template's commented `[dynamics]` block gets the key.

## 4. Validation

### 4.1 Unit tests (fast tier; analytic models)

- The default is bit-identical: the golden test (`tests/dynamics/test_golden.py`) stays green unchanged.
- With `"previous"` and `parallel_vel`, on a Tully model with hops: at every accepted hop, `|E_tot(record) - E_tot(previous record)| <= 1e-12` Eh. Away from hops nothing changes: before the first hop, the records of the two rules are identical.
- `delta_e` under `"previous"` at a forced hop equals `E_diag'_j + K(v') - E_tot,prev` (computed independently in the test).
- A frustrated hop under `"previous"` (keep and reflect), and a `parallel_nac` hop.
- Restart bit-identity with the option on (save and load in the middle, hops in both halves).
- The input: the key in both schemas, the invalid value, the fingerprint rule (the default leaves the fingerprint unchanged against `main`; `"previous"` changes it).

### 4.2 The replay (live SHARC; asserted; pre-registered here)

The P9b replay setup, unchanged:

- start `tests/data/ethylene_hop_start.json` (start 9, `rngseed` 10);
- `sharc.x` on our backend (`SwapModel`), ethylene SA3-CASSCF(2,2)/6-31G\*;
- `overlap` coupling, EDC C = 0.1, `endpoint` interpolation, `parallel_vel`, frustrated `keep`;
- 80 steps of 0.5 fs, the replayed random numbers.

The only change on our side is `hop_energy = "previous"`. The SHARC run is shared with the existing replay tests.

- **The window:** steps 0 to `h + 20`, where `h` is the first step at which SHARC's active state changes. It ends earlier at a near tie (1e-5) or at our first rejected step, as in P9b.
- **Must be able to fail:** a hop at a step `h <= 59` (so the window is complete within 80 steps), P(S0) >= 1e-3 in the window, and the same active state on both sides at every step of the window.
- **Bounds** (P9b, spec 5): positions <= 1e-5 bohr, velocities <= 1e-6 au, populations <= 1e-5, hopping probabilities <= 1e-5.
- **Also asserted:** at our hop step, `|E_tot(h) - E_tot(h - 1)| <= 1e-9` Eh.
- **Reported** (printed and in the report): the maxima over the window and over all 80 steps, and per step from `h - 1` to `h + 20`.

Why 20 steps: P9b measured a fast growth of small differences after this hop (dq about x30 per 25 steps on a hot S0). From the pre-hop agreement (1.3e-7 bohr), positions should stay inside 1e-5 for about 30 steps. **This estimate uses P9b numbers already seen; the window is fixed here, before any run with the option.**

If the test fails its bounds, it is reported, not tuned: it becomes a strict xfail with its measured numbers (the P9b ruling P5 precedent). The existing tests stay as they are: `test_replayed_trajectory_agrees_before_the_hop` (passing) and `test_replayed_trajectory_over_the_pre_registered_window` (strict xfail on the default rule).

### 4.3 The ensemble (report; nothing asserted)

- **The run:** a copy of `~/p9b/ours/run` to `~/hop/ours_previous` with `hop_energy = "previous"` in `[dynamics]`, then `gqteashmd run input.toml --rerun-from dynamics` (P9b PR2, copy path; about 2 h, detached).
- **Compared with the P9b run** (`~/p9b/ours/run`, the same initial conditions and random streams):
  - the outcomes;
  - the kept count (`is_kept`);
  - hops and frustrated hops;
  - `rejections_by_signal`;
  - `max_drift_eV` (median, p95, max);
  - the per-record |dE_tot| at hops and away from them;
  - the losses after a hop;
  - the analysis (N, the excluded fraction, t0 and tau with their intervals).
- **Also, as a report:** `tools/sharc_compare/ensemble_compare.py` of the new run against the P9b swap run (`~/p9b/swap/project`), the four P9b rules with their verdicts.
- Nothing is re-tuned on these numbers.

### 4.4 Not re-run

The asserted Tully, LVC and SHARC comparisons use the default, which is bit-identical, so they are not re-run inside the tasks. The full suite runs before the final review, as usual.

## 5. Components

- `src/gqteashmd/dynamics/options.py`: the field, `HOP_ENERGIES`, validation.
- `src/gqteashmd/dynamics/driver.py`: `delta_e` at step 8; the module docstring's step order mentions the option.
- `src/gqteashmd/config/schema.py`, `config/model_dynamics.py`, `stages/dynamics.py` (`run_settings`, `fingerprint_inputs`), the `init` template.
- Tests: `tests/dynamics/` (the rule, the restart), `tests/config/` (both schemas), `tests/stages/test_dynamics.py` (the fingerprint), `tests/validation/test_sharc_pyscf_dynamics.py` (the new replay test).
- A report script for 4.3 under `tools/validation/` (JSON out, nothing asserted, never imported by `gqteashmd`).
- Clean room: SHARC is executed only. The rule comes from our own black-box measurement (P9b); nothing in the SHARC installation is read.

## 6. Documents

- **`USER_MANUAL.md`:** the key (sections 4.2 and 5.7), the rule in the dynamics method section, and a new subsection in section 8 with every number marked asserted or report.
- **`CLAUDE.md`:** "Read first", the module block, status, the known limitations (item 1c), remaining work.
- **The P9b report, section 7:** the owner decisions of 2026-11-18 (done in the spec commit).
- **A short report** `docs/superpowers/reports/<date>-hop-energy-option.md`, dated the day it is written, with the replay and the ensemble numbers.
