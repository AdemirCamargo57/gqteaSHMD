# The CASSCF start rule and the P9a closure — design

**Status:** design approved in conversation by the owner on 2026-11-13 (both parts); this written spec awaits the owner's review.
**Branch:** `feat/casscf-start-rule` (from `main` at `e1bdc74`).
**Inputs:** the P9a acceptance report `docs/superpowers/reports/2026-11-13-p9a-acceptance.md` (section 3.2, the start-solution survey; section 6, the open decisions) and the diagnosis report `docs/superpowers/reports/2026-11-12-p9a-diagnosis.md`.

## 0. Owner decisions (2026-11-13)

On the four decisions of acceptance report section 6:

1. **The miss:** P9a closes with the miss documented (13 of 20 kept on seed 2027, 16 needed). The post-hop losses do not become a work item now.
2. **The start-solution rule:** add one.
3. **The dormant CASSCF rescue rungs** (F4, off by default): remove them.
4. **The F1 default:** keep `energy_jump_limit_eV = 0.1`.

On the design of the start rule (multiple choice, the recommended option each time):

5. **Criterion: continuity with a reference solution.** Solve once at a reference geometry, take the lowest SA energy among a fixed guess set there, and start each fresh solve from those orbitals projected onto its geometry.
6. **Scope: every CASSCF stage** (`optimize`, `frequencies`, `excitations`, `dynamics`), and `single-point`.
7. **Mismatch: record and warn.** When a start from the reference converges to another solution, it is accepted; the active-orbital overlap is recorded and a warning is logged below a fixed constant. No behaviour depends on it.

## 1. Goal and scope

Today a fresh SA-CASSCF solve (one with no previous orbitals: an `excitations` sample, a finite-difference `frequencies` task, `optimize`'s first geometry, a trajectory's step 0) starts from the SCF orbitals, and the solution it converges to is whatever PySCF finds from there. The P9a survey found 4 distinct SA3-CAS(2,2) solutions at every one of the 20 ethylene starts, and our runs happened to land on "sol0" (the lowest in SA energy among the guesses and the most like the equilibrium solution) in 20 of 20. Nothing guarantees it. The goal is a deterministic, documented rule: a fresh solve starts on the solution continuously connected to the reference geometry's lowest-SA solution.

In scope: the rule (sections 2-5), the removal of the rescue rungs (section 6), the P9a closure documents (section 7), the validation (section 8).

Out of scope: the post-hop losses (decision 1); a wider guess set or a global search for the lowest SA energy (SHARC's lower solution in 5 of 20 starts is not sought); TDA/TDDFT (no orbital optimisation, no rule); any change to the dynamics rejection signals, thresholds or `energy_jump_limit_eV`.

## 2. The reference solution

**Reference geometry.** `PySCFBackend` gains a keyword `reference_coords_bohr: ArrayLike | None = None` (passed through `make_backend`). `stage_backend` passes `working_geometry(ctx)`'s coordinates, so the reference is the prepared geometry for `optimize` (it is not upstream of itself) and the optimized geometry (or the prepared one without `optimize`) for `frequencies`, `excitations` and `dynamics`. `single-point` passes its own geometry. `None` (library callers, tests, the `tools/` scripts) means no reference: the ladder is today's.

**When it is computed.** Lazily, the first time the reference rung is reached (section 3), inside the backend's PySCF environment (same log file, threads, scratch). It is cached on the backend object: it survives `reset()` and `load_state()` and is never written to `save_state()` (a resumed trajectory starts from its saved previous orbitals, and a needed reference is recomputed deterministically).

**Guess set.** With `n_occ_act = nelecas / 2` and `n_virt_act = ncas - n_occ_act`, the occupied window is the `n_occ_act + 1` highest doubly occupied SCF orbitals and the virtual window the `n_virt_act + 1` lowest virtual SCF orbitals, each clipped to the orbitals that exist. A guess is one choice of `n_occ_act` orbitals from the occupied window and `n_virt_act` from the virtual window, applied with `mcscf.sort_mo`. Order: PySCF's default active space (the `n_occ_act` highest occupied and the `n_virt_act` lowest virtual orbitals) first, then the other choices in lexicographic order of their sorted 0-based indices. CAS(2,2): 4 guesses; CAS(4,4): 9; CAS(6,6): 16. This is the survey's set (`tools/diagnosis/start_solutions.py:guess_pairs`), reimplemented in `src/` (the package never imports `tools/`) and extended to the edge cases (`n_virt_act = 0` and an occupied window larger than the occupied orbitals).

**Selection.** Each guess is solved with the backend's own settings (state average, `conv_tol`, `conv_tol_grad`, `max_cycle_macro`, the singlet-only solver; no spin penalty). A guess counts if it converges and passes the spin guard (`|<S^2>| <= SPIN_TOLERANCE` for every root). The reference is the counted guess with the lowest SA energy; guesses within `1e-8` Eh of the lowest are ties, won by the first in the order. An exception inside a guess (not a non-convergence) is wrapped by `wrap_failure` and propagates, as in the ladder (a broken setup; `MemoryError`/`OSError` stay retryable).

**Fallback.** If the SCF at the reference geometry fails or no guess counts, the backend logs a warning (`no CASSCF reference solution at the reference geometry (...): fresh solves start from the SCF orbitals`), caches "unavailable" (no retry on the same backend) and the reference rung is skipped. This is today's behaviour, recorded in the metadata (section 4).

## 3. The ladder

`pyscf_casscf.LADDER` becomes:

1. previous orbitals (skipped without them);
2. **reference orbitals** (skipped without a reference): the reference MOs projected onto the current geometry with `mcscf.project_init_guess`, nominal settings;
3. SCF orbitals;
4. SCF orbitals + spin penalty (3 x macro iterations).

A fresh solve therefore starts with rung 2. Mid-trajectory, a failed rung 1 tries rung 2 before the SCF orbitals. For the `dynamics` stage, a rung other than "previous orbitals" is already the CASSCF rung signal (`BackendModel.rung_changed`), so a fall to rung 2 is rejected and halved like a fall to rung 3 today; at step 0 no signal applies. `run_casscf` takes the reference as an argument (`reference: tuple[np.ndarray, Any] | None`, MOs and Mole, like `previous`); the backend computes it (section 2) before calling `run_casscf` only when rung 1 cannot be used or fails. The cleanest split (a callable the ladder calls when it reaches rung 2, or a two-call sequence) is a plan decision; the observable behaviour is the ladder above, the lazy computation and the metadata.

## 4. Metadata and the mismatch diagnostic

The result `meta` of every CASSCF `compute` carries:

- `casscf_reference`: `"none"` (no reference geometry), `"unavailable"` (the fallback), or `{"guess": [...], "sa_energy_Eh": x, "n_guesses": k, "n_counted": m}` once computed (the same object on every later compute of that backend);
- when rung 2 produced the solution: `reference_active_min_sv`, the smallest singular value of `C_ref^T S C_act` (the projected reference active orbitals against the converged active orbitals, AO overlap metric `S` at the current geometry).

`REFERENCE_OVERLAP_WARNING = 0.9` (a module constant in `pyscf_casscf.py`, not an input key, not hashed; the survey measured 0.950-0.997 for sol0 and 0.0016-0.154 for the other solutions). Below it the backend logs `the CASSCF solution started from the reference orbitals has a different active-orbital character (smallest singular value x < 0.9)`. Nothing else depends on the value. The stages record no new field: the warning reaches `electronic.log`/the worker log, and the metadata reaches whatever already stores the result meta.

## 5. Fingerprints and user impact

No input key. A module constant `CASSCF_START_RULE = "reference-1"` is added to the fingerprint data of every stage that hashes `[electronic]` (`optimize` and `frequencies` through `ground_state_fingerprint`; `excitations` and `dynamics` through their own section dumps) when `method = "casscf"`. Consequences:

- an existing CASSCF project is stale (exit 3) at its first CASSCF stage and needs `--rerun-from optimize` (or `--rerun-from frequencies` without `optimize`), which recomputes everything downstream by the fingerprint chain;
- TDA/TDDFT projects stay up to date;
- stage versions are not raised (the marker does the job for CASSCF only);
- the dynamics restart format and the backend's `save_state` fingerprint are unchanged.

Cost: one set of guesses (4 for CAS(2,2)) per backend, at the reference geometry. A backend is built per `excitations` task (`samples_per_task` samples), per `frequencies` finite-difference task and per trajectory, so `frequencies` and `excitations` with `samples_per_task = 1` pay it per task. The cost is measured in section 8, not estimated in the documents.

## 6. Removing the rescue rungs

Removed: `EXTRA_RUNGS`, `EXTRA_RUNG_NAMES`, `check_extra_rungs`, `run_casscf(..., extra_rungs=)`, `casscf_extra_rungs` in `PySCFBackend`, `make_backend` and `stage_backend`, the `Rung` fields used only by them (`conv_tol`, `conv_tol_grad`, the `"previous_or_scf"` start), their tests, and `tools/diagnosis/convergence_probe.py` with its tests (it exists only to drive those rungs; its measurement stays in the acceptance report, cited by commit `95fb881`). `Rung` keeps `name`, `start` (`"previous"`, `"reference"`, `"scf"`), `spin_shift`, `macro_factor`.

## 7. Closing P9a

- The acceptance report gets a section "Owner decisions (2026-11-13)" with decisions 1-4 of section 0 (and a pointer to this spec for the rule).
- CLAUDE.md: P9a closed (status, Remaining work, the P9 table row); the ladder and the start rule in the electronic block; the rescue rungs removed wherever mentioned (P9a block, Known limitations, deferred list); P9b is the next sub-project.
- USER_MANUAL.md: the start rule and the ladder (the CASSCF section, 8.6.8 replaced), the stale-project advice (`--rerun-from optimize`/`frequencies`), the metadata keys, 5.10 updated (decision 1, the rule replaces "no rule"), the F1 default as decided; implemented vs planned marked.

## 8. Validation

**Unit tests** (no PySCF, through the `_converged` seam and fakes where the existing tests use them; PySCF tests where they need real solves):

- the guess set: counts and order for CAS(2,2), CAS(4,4), CAS(2,3), `n_virt_act = 0`, a clipped occupied window;
- the selection: lowest SA energy wins; a tie within 1e-8 Eh goes to the first guess; non-converged and spin-failed guesses do not count; none counting gives the fallback with its warning and `"unavailable"`, and no second attempt on the same backend;
- the ladder order with and without previous orbitals and a reference; the reference is computed lazily (not at all when rung 1 succeeds) and once per backend, also across `reset()`;
- the metadata keys and the warning below `REFERENCE_OVERLAP_WARNING`;
- the fingerprint marker: present for CASSCF in the four stages, absent for TDA/TDDFT (a TDA project's fingerprints are unchanged by this branch);
- the rescue rungs are gone (`casscf_extra_rungs` is not accepted).

**PySCF tests** (marker `pyscf`; `slow` where a run is long):

- ethylene SA3-CAS(2,2)/6-31G\* at the conftest equilibrium geometry: the reference equals the lowest-SA solution of the 4 guesses solved independently;
- a fresh solve at a displaced geometry (a fixed displacement in the test) uses rung 2 and records `reference_active_min_sv >= 0.9`;
- a backend built by `stage_backend` for a `frequencies` finite-difference task (ethylene, a canned prepare/optimize context) has the working geometry as its reference: its first `compute` reports `casscf: "reference orbitals"` and a `casscf_reference` object;
- the existing CASSCF tests (P3a/P3b SHARC comparisons, finite differences, the dynamics tests) stay green; any that changes is reported with its numbers.

**Measurements by hand** (report only, nothing tuned, numbers in the plan's ledger and the documents):

- seed 2026: the rule's step-0 solution at the 20 M4 starts (SA3-CAS(2,2), reference = the M4 optimized geometry) against the survey's sol0 (`start_solutions.json`, energy match 1e-4 Eh). Expected 20 of 20 (our runs already started on sol0); whatever is measured is reported, a mismatch is documented and left to the owner;
- the cost: the reference computation's wall time per backend and the `excitations` and `frequencies` stage wall times of the ethylene end-to-end test before and after;
- the flaky live-SHARC test of CLAUDE.md limitation 3a (`test_excitations_match_sharc_pyscf`, water CAS(4,4)/STO-3G): run 3 times with the rule; report whether our S1 stays on one solution. No claim beyond the measurement.

**Suites:** the fast tier during the work; the full suite `pytest -W error` with SHARC and ORCA at the branch head before the merge.

## 9. Risks

- **The reference solution may not be the physically wanted one** at a molecule other than ethylene (the lowest SA energy among a small guess set is a heuristic). The metadata records which guess won, and the warning shows a character change at a sample.
- **A projected start can still converge to another solution** at a strongly distorted geometry; by decision 7 this is accepted and warned, not prevented.
- **Results of existing CASSCF projects can change** after the rerun (a sample that landed on another solution today now starts from the reference). This is the purpose of the rule; the measurement in section 8 shows how often on seed 2026.
- **Cost** for large active spaces (16 guesses for CAS(6,6), 25 for CAS(8,8)) per backend; measured for CAS(2,2) only.
