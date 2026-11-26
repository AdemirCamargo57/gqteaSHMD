# Triplet states in the PySCF backend (triplets and SOC, sub-project 1): report

Status: **triplet states work in the PySCF backend, in `single-point` and in the `excitations` stage, for SA-CASSCF and TDA/TDDFT, without spin-orbit coupling.** Every asserted check that remained after the rulings passes within its spec-fixed bound: ORCA 6.1.1 energies (largest difference 1.07e-8 Eh, bound 1e-6), finite-difference triplet gradients (8.59e-7 Eh/bohr, bound 1e-5), the singlet NAC of a state average with a triplet (1.37e-5 1/bohr, bound 1e-4), triplet-block overlap continuity (0.992663 and 0.99304, bound 0.99) and the resumed overlaps (6e-15, bound 1e-10). **Three things differ from the owner-approved design.** (1) Triplet-triplet NACs are **not available**: PySCF 2.14 fails on them in a mixed-spin state average, so the program refuses them; the approved default was "checked by finite differences". (2) TDA/TDDFT triplets are expanded in their Ms = 0 component (plan ruling PR1), not Ms = +1. (3) SHARC 4.1's `SHARC_PYSCF.py` refuses non-singlet states, so there is no SHARC comparison. Four test choices were changed after numbers were seen (section 8); no spec-fixed bound was changed. The `dynamics` stage refuses `triplets > 0` until sub-project 3.

Date: 2026-11-26. Branch `feat/triplets-backend` (from `main` `478d3f2`). The spec is `docs/superpowers/specs/2026-11-26-triplets-backend-design.md` (owner-approved 2026-11-26; spec commit `6b68564`; erratum at its top). The plan is `docs/superpowers/plans/2026-11-26-triplets-backend.md` (plan commit `ad952e1`; its "Plan rulings" PR1-PR8 are binding where they differ from the spec). The code tasks are these commits:

- `949e6f5`: Task 2, the input (the refusal removed, the CASSCF triplet count);
- `0f689cf` and its fix `dcf2629`: Task 3, TDA/TDDFT triplets;
- `05e4ba0` and its fix `5d79f83`: Task 4, SA-CASSCF triplets (the fix refuses triplet-triplet NACs);
- `4ec82df`: Task 5, the backend (blocks, overlaps, phases, the saved state, `single-point`);
- `163dc57`: Task 6, the stages (`excitations`, the `dynamics` refusal, the ground-state settings);
- `bd043b8`: Task 7, the validation tests.

Task 1 (what `SHARC_PYSCF.py` supports) and Task 7b (the test tiers) are measurements by the controller, with no diff.

**Sources of every number:**

- the task reports in the git-ignored ledger `.superpowers/sdd/2026-11-26-triplets-backend/` (`task-1-report.md` to `task-7-report.md`, `progress.md`);
- the throwaway probes named there (in WSL `~/probe_t5.py`, `~/probe_t5_resume.py`, `~/probe_t5_eth.py`; the Task 4 and Task 7 scratchpad scripts; the ORCA directories `~/task7_orca/w` and `~/task7_orca/e`). None is in the repository.

**Labels:**

- *Asserted*: a pytest assertion.
- *Report*: measured, with no rule.
- *Inference*: not established by a measurement.

## 1. What was built

### 1.1 Input (Task 2)

- The load-time refusal of `[electronic] triplets > 0` is removed for `casscf`, `tda` and `tddft`.
- `config/schema.py:n_triplet_csfs(ncas, nelecas)` counts the triplet configuration state functions of CAS(`nelecas`, `ncas`) (the Weyl formula with S = 1: `3/(n+1) C(n+1, N/2-1) C(n+1, N/2+2)`). For CASSCF, `triplets` may not exceed it: *electronic.triplets = m exceeds the k triplet states of CAS(N,n) (nelecas = N, ncas = n)* (exit 2). CAS(2,2) has 1, CAS(4,4) 15 (asserted).
- `triplets` was already part of every electronic fingerprint, so a project with `triplets = 0` keeps its fingerprints.

### 1.2 SA-CASSCF (Task 4, `electronic/pyscf_casscf.py`)

- **The state average.** With `triplets > 0` the backend builds `mcscf.state_average_mix_` over a singlet solver (`fci.solver(mol, singlet=True)`, n roots) and a triplet solver (`singlet=False`, spin 2, so the CI vectors are Ms = +1 determinants of the active space; m roots), with equal weights 1/(n + m) (`mixed_solver`). With `triplets = 0` the singlet-only solver of before runs, unchanged (asserted: the mixed solver is not called).
- **The spin guard** checks every root against its own multiplicity: singlets `|<S^2>| <= 1e-3`, triplets `|<S^2> - 2| <= 1e-3` (`SPIN_TOLERANCE`). With triplets the message is *not all roots have the expected spin (...)*; without them it keeps its old text.
- **The ladder** keeps its four rungs. On the spin-penalty rung each solver gets its own `fix_spin_` (ss = 0 for the singlets, ss = 2 for the triplets).
- **The start rule.** The reference search (`find_reference`) solves the same mixed state average, and a guess counts only if every root has the right spin. The default guess still comes first. `CASSCF_START_RULE` stays `"reference-1"`.
- **Gradients:** `nuc_grad_method` of the mixed state average, at the state's position in the average.
- **NACs:** singlet-singlet pairs from the mixed state average's `nac_method`; singlet-triplet pairs are exact zero vectors, with no PySCF call; **triplet-triplet pairs are refused** (section 6, item 1).
- **Transition dipoles:** S0 -> Sk as before; with triplets present the transition density comes from `pyscf.fci.direct_spin1.trans_rdm1`. S0 -> Tk is zero.
- **Expansions:** `casscf_expansions(mc, states)` gives one determinant expansion per multiplicity (the singlet block in Ms = 0, the triplet block in Ms = +1).

### 1.3 TDA/TDDFT (Task 3, `electronic/pyscf_response.py`)

- **A separate triplet response solve** on the same SCF (`td.singlet = False`), with the same root ladder as the singlets (each rung solves `min(m + extra, available)` roots and keeps the lowest m). `td_conv_tol` applies to both. Non-convergence reads *TDA did not converge for all m excited triplet states (...)*. `meta["td_triplet"]` holds the triplet rung, only with triplets.
- **Gradients** from the triplet response object's `nuc_grad_method`. A triplet gradient without that object is a `ValueError` (a programming error, not an electronic failure).
- **NACs:** none, as for the singlets (overlaps only).
- **Transition dipoles:** S0 -> T zero.
- **Expansion** (PR1): the Ms = 0 component, `sum_ia X_ia (|i->a, alpha> - |i->a, beta>)`, renormalised, in the same determinant layout as the singlets (whose excited part has `+`). One private helper builds the layout for both (fix `dcf2629`), so the two cannot drift apart.

### 1.4 The backend (Task 5, `electronic/pyscf_backend.py`)

- `PySCFBackend.states` is `S0 ... S(n-1), T1 ... Tm`. Every per-state result is keyed by these labels.
- **Overlaps are block-diagonal by multiplicity.** Each block is computed separately into a zero matrix, so the singlet-triplet entries are exactly 0. Truncation (`overlap_norm_kept`) works within each block. Phase tracking runs on the whole matrix (a cross match has |S| = 0), and each block gets its slice of the phases.
- **The saved state** (PR5): without triplets `save_state` writes format 1, the same keys in the same order as before. With triplets it writes **format 2**: format 1's arrays plus `t_alpha`, `t_beta`, `t_coeffs`. `load_state` reads both. A format-1 file for a backend with triplets is refused ("missing the triplet expansion"), a format-2 file for one without them too ("unexpected triplet expansion"), and so is a triplet block with the wrong number of states.
- **`single-point`** (PR6): with `triplets > 0` the table gets an `S^2` column (`-` where there is no value: TDA/TDDFT give no `<S^2>`), and the triplet rows show f = 0. The JSON gets the triplet rows (transition dipole `[0, 0, 0]`, `oscillator_strength` 0.0, `spin_square` for CASSCF, `null` for TDA/TDDFT). Without triplets the table is unchanged.

### 1.5 The stages (Task 6)

- **`optimize`, `frequencies`:** S0 only, through `factory.ground_state_settings`. For TDA/TDDFT it now also sets `triplets = 0`, so the ground-state fingerprint does not change. For CASSCF the state average keeps the triplets, so S0 changes (section 4.6) and so does the fingerprint.
- **`excitations`:** with triplets, `excitations.h5` gains `triplet_labels` and `triplet_energies` (samples x triplets, S0 -> T excitation energies in eV, NaN rows for failed samples, `units` attribute). The `reference` group carries no triplet data (PR7). `excitations.json` gains `triplet_labels` and `triplet_states` (per triplet the mean and population standard deviation of the excitation energy and the reference value; `null` when the stage stopped). A non-finite triplet energy or a triplet at or below S0 fails the sample like a singlet one. The singlet arrays, the spectrum and the selection candidates are computed exactly as before.
- **`dynamics`:** `check_config` refuses `triplets > 0` (exit 2): *Stage 'dynamics' with [electronic] triplets > 0 is planned (spin-orbit coupling, sub-project 3); set [electronic] triplets = 0*.
- **`selection`, `analysis`:** unchanged (their state lists already reject triplet labels).

## 2. Task 1: `SHARC_PYSCF.py` refuses triplets (report)

The controller ran SHARC 4.1's `SHARC_PYSCF.py` (version 4.0, 01.04.25) on ethylene SA-CASSCF(2,2)/6-31G\* with `states 2 0 1`, `h`, `grad all`, the template's roots line `roots 2 0 1`, and the environment of `tools/sharc_compare/run_sharc_pyscf.run_geometry`. It exited with code 1 and wrote no `QM.out`; its output reads `NotImplementedError: Not implemented for states other than singlets!` while reading `QM.in` (seen in its output only; no SHARC source was read). So there is no SHARC comparison of the triplets. `tests/validation/test_sharc_triplets.py` is a skip with that reason (spec 4 item 5: "no substitute is invented"). Water `states 2 0 2` was not tried, because the refusal comes at input reading.

## 3. Tests (asserted)

| Task | Command | Result |
|---|---|---|
| 2 | `pytest tests/config tests/test_cli.py -q -W error -m "not slow and not pyscf"` | **147 passed**, 2 deselected |
| 3 (with its fix) | `pytest tests/electronic/test_pyscf_linear_response.py tests/electronic/test_factory.py -q -W error` | **24 passed** (92 s); the triplet tests with `OPENBLAS_CORETYPE=SandyBridge` 6 passed |
| 4 | `pytest tests/electronic -q -W error -m "not slow"` | **120 passed**, 3 deselected (21 min 22 s) |
| 4 (fix `5d79f83`) | `pytest tests/electronic/test_pyscf_casscf.py tests/electronic/test_casscf_guesses.py -q -W error -m "not slow"` | **64 passed** (61.33 s); with `OPENBLAS_CORETYPE=SandyBridge` 64 passed |
| 5 | `pytest tests/electronic tests/test_cli_single_point.py tests/test_cli.py -q -W error -m "not slow"` | **169 passed**, 5 deselected (16 min 54 s); `tests/electronic/test_triplet_overlaps.py` 4 passed twice; the new tests with `OPENBLAS_CORETYPE=SandyBridge` 9 passed |
| 6 | `pytest tests/electronic/test_factory.py tests/stages tests/excite -q -W error -m "not slow"` | 225 passed, 1 failed (a test fixture, fixed; the file re-run 6 passed); with `OPENBLAS_CORETYPE=SandyBridge` the factory, excitations, dynamics, artifacts, fingerprint and selection tests 73 passed |
| 7 | with `ORCA_EXE`, `SHARC_HOME`, `SHARC_PYTHON`: `pytest tests/validation/test_orca_triplets.py tests/validation/test_sharc_triplets.py -q -W error` | **2 passed, 1 skipped** (59.97 s; the skip is the SHARC refusal of section 2) |
| 7 | `pytest tests/electronic/test_gradients_fd.py tests/electronic/test_couplings_fd.py -q -W error -m slow -k "triplet or mixed"` | **4 passed**, 5 deselected (62.81 s) |
| 7 | with `OPENBLAS_CORETYPE=SandyBridge`, ORCA and SHARC set: `pytest tests/electronic/test_couplings_fd.py tests/validation/test_orca_triplets.py tests/validation/test_sharc_triplets.py -m "not slow" -q -W error` | 4 passed, 1 skipped, 2 deselected (32.03 s) |
| 7b (controller, at `bd043b8`) | fast tier `pytest -m "not slow and not pyscf" -W error -q` (on `/mnt/c`) | **1429 passed, 22 skipped, 261 deselected in 1299.51 s** (21 min 39 s) |
| 7b (controller, at `bd043b8`) | `pytest tests/electronic tests/stages/test_excitations.py tests/test_cli_single_point.py -W error -q` (all markers) | **170 passed in 414.66 s** (6 min 55 s) |

The full suite `pytest -W error` with SHARC and ORCA was **not run** on this branch. Every existing test stayed green without modification, except the one parametrised case that asserted the old refusal (removed in Task 2, as the plan said).

What the tests assert, among others:

- **Input:** `triplets > 0` is accepted for all three methods; CASSCF triplets beyond the active space are refused; the CSF counts.
- **TDA/TDDFT:** the triplet energies, a finite triplet gradient, zero S0 -> T dipoles, the Ms = 0 expansion orthonormal at 1e-8 and orthogonal to the singlets; with `triplets = 0` no triplet solve is made.
- **SA-CASSCF** (water CAS(4,4)/STO-3G 2S+2T, ethylene CAS(2,2)/6-31G\* 2S+1T): the spins; the state energies against a hand-built `state_average_mix_` (section 8.1); exact S-T zero NACs and zero T dipoles; finite triplet gradients; the block expansions; the mixed solver is not called with `triplets = 0`; the triplet-triplet NAC refusal, also at request time and before any PySCF call; the reference search counts guesses with the right spins; the ethylene S0 of the mixed average differs from the singlet-only S0 by more than 1e-5 Eh (bound chosen before the run).
- **Backend:** triplet-block continuity and the resumed overlaps (section 4.4); the saved-state formats 1 and 2 and the three refusals; the `single-point` S^2 column, the triplet rows, and the unchanged table without triplets.
- **Stages:** `excitations` records the triplets and leaves the singlet results alone; a triplet below S0 fails the sample; `dynamics` refuses triplets; the TDA ground-state settings drop the triplets; the `excitations.h5` round trip with triplets. **These stage tests use the fake backend**, not a real PySCF triplet run through the stage.

## 4. Validation numbers (asserted)

### 4.1 ORCA 6.1.1, black box (marker `orca`; rule: every state energy within 1e-6 Eh)

ORCA was run without RI or COSX, `TightSCF`, the same basis and geometry. Two input keywords were added after the first numbers were seen (section 8.4): `NoFrozenCore` for water, and `etol 1e-10` / `gtol 1e-6` in `%casscf` for ethylene.

| Quantity | ORCA | ours | difference (Eh) |
|---|---|---|---|
| water RHF/STO-3G SCF | -74.9630231390 | -74.9630231385 | +5.6e-10 |
| water CIS T1 excitation | 0.4074603 (89427.2 cm-1) | 0.40746039 | +8.2e-8 |
| water CIS T2 excitation | 0.49211519 (108006.8 cm-1) | 0.49211524 | +4.9e-8 |
| water CIS T3 excitation | 0.50784958 (111460.1 cm-1) | 0.50784949 | -8.9e-8 |
| ethylene SA-CASSCF(2,2)/6-31G\* 2S+1T, S0 | -78.0524948366 | -78.0524948362 | +4.2e-10 |
| same, S1 | -77.6658066565 | -77.6658066634 | -6.9e-9 |
| same, T1 | -77.8943984708 | -77.8943984601 | +1.07e-8 |

- The CIS comparison reads ORCA's cm-1 column (0.1 cm-1 printed, about 2.3e-7 Eh); against ORCA's full-precision `property.txt` values the triplet differences were 3.1e-9, 1.1e-8 and 2.7e-9 Eh (report).
- ORCA reorders the blocks and prints a weight of 1/3 per root, which the test parses and asserts: the same equal-weight average as ours.
- ORCA's ethylene energies differ between the test run and a hand run by up to 5.5e-9 Eh (ORCA's own convergence scatter at `gtol` 1e-6; report). The numbers are the same under `OPENBLAS_CORETYPE=SandyBridge` to the printed digits.

### 4.2 Finite-difference gradients (markers `slow`, `pyscf`; rule: max |analytic - FD| <= 1e-5 Eh/bohr)

Central differences, step 1e-3 bohr, a fresh backend for every displaced geometry, a fixed set of Cartesian components (PR3).

| System | States | Components | max \|analytic - FD\| (Eh/bohr) |
|---|---|---|---|
| water TDA-HF/STO-3G, 1S+2T | T1, T2 | (0,2), (1,1), (2,2) | 8.32e-7 (T2 at (2,2)) |
| water TDA-B3LYP/STO-3G, `grid_level` 5, 1S+1T | T1 | (0,2), (1,1), (2,2) | 8.59e-7 (T1 at (0,2)) |
| ethylene SA-CASSCF(2,2)/6-31G\*, 2S+1T | S1, T1 | (0,0), (2,1) | 3.40e-7 (T1 at (0,0)) |

A spot check before the test (report) gave S0, S1 and T1 of the ethylene mixed average within 3.4e-7 Eh/bohr at three components.

### 4.3 Nonadiabatic couplings

- **The singlet NAC of a state average with a triplet** (marker `slow`; rule: within 1e-4 1/bohr): ethylene SA-CASSCF(2,2)/6-31G\* 2S+1T, pair (S0, S1), `conv_tol` 1e-11, `conv_tol_grad` 1e-7, against central differences of the phased overlaps, step 1e-3 bohr, components (0,0), (2,1), (4,0): differences +1.37e-5, +5.58e-6, +8.65e-6, **max 1.37e-5 1/bohr**. The NAC magnitudes there are 0.285, 3.6e-3 and 9.7e-3. This system replaced the planned water system (section 8.3).
- **Triplet-triplet NACs are refused** (water CAS(4,4)/STO-3G 2S+2T, pair (T1, T2)): a `ValueError` with `TRIPLET_NAC_MESSAGE` at request time.
- **Singlet-triplet NACs are exactly zero** (same system, pairs (S1, T1) and (S0, T2)): `np.array_equal` with zeros, shape (3, 3).

### 4.4 Overlap continuity and the resumed overlaps (rules: triplet-block smallest singular value >= 0.99 on 0.01-bohr steps; positive diagonal after tracking; S-T entries exactly 0; resumed against uninterrupted 1e-10)

The path has 5 points, `coords + 0.01 k` bohr in every coordinate (a rigid translation).

| System | Triplet block, smallest singular value per step | Other |
|---|---|---|
| TDA, water TDA-HF/STO-3G 2S+2T | 0.99304 at every step | |
| CASSCF, ethylene SA-CASSCF(2,2)/6-31G\* 2S+1T | 0.992663 at every step (a 1x1 block) | singlet block 0.992656; smallest diagonal 0.992641; cross blocks exactly 0.0; rung "previous orbitals" at every step; the same under SandyBridge |

The resumed overlaps: one backend computes points 0-2, saves its state and continues to points 3 and 4; a new backend loads the saved state and computes points 3 and 4. Largest difference of the phased overlaps: ethylene 5.66e-15 / 4.22e-15 (default kernel) and 3.77e-15 / 5.36e-15 (SandyBridge); TDA water 1.67e-15 / 2.00e-15. The CASSCF system and the structure of the resume test were changed after numbers were seen (sections 8.2 and 8.3).

### 4.5 Spins (report from Task 4; asserted only against the 1e-3 guard)

Water CAS(4,4)/STO-3G 2S+2T, rung "SCF orbitals": S0 -74.97798376526 Eh (`<S^2>` 6.8e-24), S1 -74.53025054059 (2.6e-15), T1 -74.59534388539 (2.0000000000000004), T2 -74.48068916719 (2.0000000000000178). Its reference search: guess [3, 4, 5, 6], SA energy -74.64606683960 Eh, `n_guesses` 3 (STO-3G water has 7 MOs, so the virtual window is cut), `n_solved` 1, `n_counted` 1.

### 4.6 S0 of CASSCF changes with the triplets (asserted > 1e-5 Eh; value report)

Ethylene CAS(2,2)/6-31G\*: the mixed 2S+1T average gives S0 -78.05249483926, S1 -77.66580665354, T1 -77.89439846686 Eh; the singlet-only SA2 gives S0 -78.04902441636, S1 -77.67250837 Eh. **S0 moves by -3.47e-3 Eh.** So turning triplets on in a CASSCF project changes the S0 surface, the optimized geometry, the frequencies and the samples. This is why its fingerprint changes from `optimize` on (section 9).

## 5. Report items (measured, no rule)

- **Water CAS(4,4)/STO-3G overlap continuity** (the plan's first CASSCF system, Task 5; conv_tol_grad 1e-6, the same rigid path, steps 1-4):

  | Case | Smallest singular value per step |
  |---|---|
  | 2S+2T, triplet block | 0.99303, 0.98266, 0.99294, 0.98314 (another run: 0.99302, 0.99302, 0.99302, 0.98315) |
  | 2S+2T, singlet block | 0.99303, 0.97935, 0.99302, 0.97837 |
  | `triplets = 0`, singlet block | 0.9770, 0.9930, 0.9763, 0.9770 (step 3 fell back to "SCF orbitals") |

  At a low step the triplet block is a rotation of about 0.095 rad scaled by about 0.983; the energies along the path agree to about 1e-6 Eh. At `conv_tol_grad` 1e-7 and 1e-8 no rung converges. Which steps drop is not reproducible, even between two backends in one process.
- **Water CAS(4,4)/STO-3G S0-S1 NAC against FD overlaps** (the plan's NAC system, Task 7): with 2S+2T off by up to 1.6e-3 1/bohr (3.3e-3 at `conv_tol_grad` 5e-7 and h 1e-3), the same at h 1e-4 and 1e-3 and at `conv_tol_grad` 1e-6 to 3e-7 (systematic, not FD noise). **With `triplets = 0` off by 2.3e-2 1/bohr**, and the NAC is not symmetric between the two H atoms (-0.200 against -0.240 1/bohr; FD -0.214 against -0.217), although the molecule is C2v. CAS(4,4) in STO-3G water takes both virtual orbitals, so there is no external space. **The cause was not established.**
- **Ethylene with a triplet, by step size** (Task 7): the S0-S1 NAC against FD overlaps differs by 4.6e-4, 2.8e-5 and 1.4e-5 1/bohr at h = 1e-4, 3e-4 and 1e-3: it falls as 1/h, which is FD noise of the converged overlaps, not a NAC error (inference from the trend). At 6-31G with a triplet the difference was 8.4e-4 (h 1e-4) and 1.2e-3 (conv_tol_grad 1e-6); without the triplet the existing 6-31G test gives 8.6e-5.
- **The direct-mix per-state energies scatter with the BLAS kernel** (Task 4): our ladder against a hand-built `state_average_mix_`, `conv_tol` 1e-10, `conv_tol_grad` 1e-6: largest per-state difference 2.47e-7 (default kernel), 6.99e-8 (SandyBridge), 1.06e-8 (Nehalem); the SA energy differs by 7.0e-12, 3.7e-12 and 3.3e-13 Eh. The test as planned (1e-8 per state) failed once under SandyBridge (5.2e-8).
- **The resume test as first written** compared two independent backends. Their step-0 sign frames come from PySCF's raw signs, which are not reproducible between objects: TDA 2.78e-3 (all of it a sign; magnitudes within 1.7e-15), CASSCF 1.6e-3 and 0.19 (signs) plus 8.9e-5 in magnitude.
- **ORCA without the added keywords** (Task 7): frozen-core CIS in ORCA's default puts T2 2.5e-5 Eh away from ours; ORCA's default CASSCF orbital-gradient tolerance (2.5e-4) leaves the per-state energies off by up to 2.6e-6 Eh (S0 2.0e-6).

## 6. Where the work differs from the owner-approved design

1. **Triplet-triplet NACs are not available.** The approved default (owner decision 7) was "CASSCF triplet-triplet NACs checked by finite differences". PySCF 2.14's SA-CASSCF NAC code raises an `AssertionError` for a pair of triplets in a mixed-spin state average: it uses `mc.nelecas` (the Ms = 0 electron count) for every root, so the 4x4 triplet CI vectors do not fit the 6x6 singlet layout (`pyscf/nac/sacasscf.py:make_fcasscf_nacs` via `direct_spin1.trans_rdm12`; water 2S+2T, Task 4). By ruling PR8 nothing in PySCF was patched. **The program refuses the request with a `ValueError`**, in `casscf_nacs` and at request time in `PySCFBackend._check_labels`, before any PySCF call: *T1-T2: nonadiabatic coupling vectors between triplet states are not available: PySCF's SA-CASSCF couplings do not support a mixed-spin state average; use coupling = 'overlap'*. Singlet-singlet NACs of the mixed average are FD-checked (section 4.3); singlet-triplet NACs are exact zeros. **Consequence for sub-project 3:** `coupling = "nacdr"` with triplets is not possible unless this is solved; `coupling = "overlap"` works, because the triplet overlaps exist. Spec 3.2 (NAC bullet) and spec 4 item 3 change accordingly. **To be surfaced to the owner.**
2. **TDA/TDDFT triplets are expanded in their Ms = 0 component** (plan ruling PR1), not in Ms = +1 as spec 3.4 said. Ms = +1 would need a spin-flip determinant sign convention that the code does not have; Ms = 0 reuses the validated singlet layout, and triplet-triplet overlaps are the same for every Ms component (Wigner-Eckart). CASSCF triplets keep Ms = +1. The cross blocks are exactly zero either way, because each block is computed separately.
3. **No SHARC comparison of the triplets** (spec 4 item 5): `SHARC_PYSCF.py` refuses non-singlet states (section 2). The spec foresaw this branch.

Smaller differences:

- **`<S^2>` of a TDA/TDDFT triplet is not reported** (spec 2.3 listed it for every triplet row): the response solve gives none, so `spin_square` is `null` and the table shows `-`. The spin guard exists for SA-CASSCF only, as before.
- **Spec 4 item 4 (continuity)** named no system; the plan's water CAS(4,4) was replaced by ethylene 2S+1T (section 8.2).
- **Spec 4 item 3** used two triplets for the NAC test (PR4: water). With T-T NACs refused, the NAC test checks a singlet pair of the mixed average on ethylene (one triplet), the refusal and the exact zeros on water.

## 7. Plan rulings PR1-PR8

- **PR1** the TDA/TDDFT triplet expansion in Ms = 0 (section 6, item 2).
- **PR2** intermediate commits: Task 2 accepts `triplets > 0` before the backend supports it and before `dynamics` refuses it; the branch is merged only when complete.
- **PR3** each FD test compares a fixed set of Cartesian components (three per state for the response tests, two for CASSCF).
- **PR4** the NAC test system water CAS(4,4)/STO-3G 2S+2T; after PR8 it holds the refusal and the zero tests only (section 8.3).
- **PR5** the saved state: format 1 without triplets, format 2 with them.
- **PR6** the `S^2` column of `single-point` only with triplets.
- **PR7** no triplet data in the `reference` group of `excitations.h5`; the reference triplet energies in `excitations.json`.
- **PR8** a failing triplet NAC stops the task (BLOCKED) and the controller rules. It was triggered in Task 4 (section 6, item 1).

Other controller rulings (ledger): Task 4 updates the two backend call sites when it changes the signatures (so the backend stays importable between commits); one private helper for the single-excitation layout (Task 3 review); the triplet-gradient guard is a `ValueError`; the T-T refusal also in `_check_labels`.

## 8. What was decided after numbers were seen

No spec-fixed bound (1e-6 Eh, 1e-5 Eh/bohr, 1e-4 1/bohr, 0.99, 1e-10) was changed. These test choices were:

1. **Task 4, the direct-mix test.** The plan asserted each state energy against a hand-built `state_average_mix_` at 1e-8 Eh; it failed once under SandyBridge (5.2e-8; section 5). Ruling: the SA energy at 1e-8 Eh and each state at 1e-6 Eh. Reason: the per-state energies of an SA-CASSCF are first order in the orbital gradient (only the SA energy is variational), so at `conv_tol_grad` 1e-6 they scatter across BLAS kernels by up to 2.5e-7; 1e-7 does not converge. The 1e-8 bound was the plan's, not the spec's. Cost: a wrong root order is then caught only through the 1e-6 bound, still far below any state spacing.
2. **Task 5, the resume test.** As planned it compared two independent backends, whose sign frames differ (section 5). Ruling: one chain (backend A computes points 0-2, saves, continues to 3-4; backend C loads A's state and computes 3-4), bound unchanged at 1e-10. Reason: phases are defined relative to a backend's own first geometry; the spec's "one that was not interrupted" is the same chain continued.
3. **Task 5, the CASSCF continuity system.** Water CAS(4,4)/STO-3G 2S+2T gave a triplet block of 0.98266/0.98314 against 0.99; the singlet block also dropped to 0.978, and to 0.976-0.977 with `triplets = 0` (section 5). Ruling: ethylene SA-CASSCF(2,2)/6-31G\* 2S+1T, the same path and bounds (measured 0.992663). Reason: the drop is a property of that SA-CASSCF along the path, also without triplets, not of the triplet code. Cost: the CASSCF block is 1x1 (ethylene has one triplet), so a CASSCF block with two triplets is checked for continuity only through TDA, and for CASSCF only by the block orthonormality of Task 4.
4. **Task 7, the singlet NAC system and the ORCA inputs.** The S0-S1 NAC of water CAS(4,4) 2S+2T was off by 1.6e-3 1/bohr, and by 2.3e-2 without triplets, asymmetric between the H atoms (section 5). Ruling: the check runs on ethylene SA-CASSCF(2,2)/6-31G\* 2S+1T, pair (S0, S1), h 1e-3, bound 1e-4; the refusal and the exact zeros stay on water 2S+2T. Reason: the water SA-CASSCF fails the FD NAC check also without triplets, so it cannot test the triplet code. Cost: a mixed-average singlet NAC with two triplets present is not FD-checked. The ORCA inputs gained `NoFrozenCore` (water; ORCA freezes O 1s by default, a different method from our all-electron CIS) and `etol 1e-10` / `gtol 1e-6` (ethylene CASSCF; the spec asks for tight reference convergence, and ORCA's default orbital-gradient tolerance is 2.5e-4). The 1e-6 Eh bound is unchanged.

## 9. What a project sees

- **`triplets = 0` (every existing project):** nothing changes. The same PySCF calls, the same numbers, the same saved-state bytes (format 1), the same stage outputs and fingerprints; no stage version changed.
- **A CASSCF project that turns triplets on:** the state average now includes the triplets, so S0 changes (section 4.6). The `optimize` fingerprint changes, and the next run stops with exit 3 at `optimize` (or at `frequencies` when `optimize` is disabled): run `--rerun-from optimize` (or `--rerun-from frequencies`). Everything from there on is recomputed.
- **A TDA/TDDFT project that turns triplets on:** the ground-state settings drop the triplets, so `optimize`, `frequencies` and `sampling` stay up to date. `excitations` hashes the whole `[electronic]` section and is stale: run `--rerun-from excitations`.
- **With a `[dynamics]` table** the input is refused (exit 2) whatever the method, until sub-project 3.

## 10. Honest summary

- *Asserted:* triplet energies of CIS and of an SA-CASSCF average with a triplet agree with ORCA 6.1.1 to 1.07e-8 Eh; analytic triplet gradients agree with finite differences to 8.6e-7 Eh/bohr (TDA-HF, TDA-B3LYP, SA-CASSCF); the singlet NAC of an average with a triplet agrees with FD overlaps to 1.37e-5 1/bohr; the triplet overlaps are continuous (0.99304 TDA, 0.992663 CASSCF) and survive a save and load to 6e-15; singlet-triplet NACs and overlaps are exactly zero; `triplets = 0` is unchanged (the existing suite green, the fast tier 1429 passed).
- *Not available:* triplet-triplet NACs (PySCF 2.14), so `coupling = "nacdr"` with triplets is not possible for sub-project 3 unless this is solved; the SHARC comparison (SHARC 4.1 refuses triplets).
- *Narrow coverage:*
  - The CASSCF continuity test has a 1x1 triplet block on a rigid-translation path, which tests numerical continuity, not a real distortion.
  - A CASSCF block with two triplets is checked only for orthonormality (Task 4).
  - The singlet NAC check with triplets present covers one triplet.
  - The `excitations` stage tests use the fake backend; no PySCF triplet run went through the stage.
  - The full suite was not run on the branch.
- *Unexplained:* the water CAS(4,4)/STO-3G SA-CASSCF fails the FD NAC check and the 0.99 continuity bound **also without triplets** (2.3e-2 1/bohr, asymmetric between equivalent atoms; 0.976-0.977). It was set aside as a property of that system, not explained. It may be related to the open intermittent water CAS(4,4) test of CLAUDE.md limitation 3a (inference).
- *Decided after numbers were seen:* four test choices (section 8); no spec-fixed bound.

## 11. Deferred minors (ledger, "minor (deferred)" lines)

- Task 2: no loader-path (`load_config`, exit 2) test of the triplet-count refusal; no combined singlet and triplet violation test; the RED of the new tests not recorded separately.
- Task 3: a weak S1 dipole assertion (norm >= 0); the energy test mirrors the code's `nstates`; no tests of the triplet failure paths (non-convergence text, `wrap_failure` label, the `n_single` refusal, the extra-roots rung); the no-triplet-solve test guards only the `solve_triplets` route.
- Task 4: `casscf_dipoles` reads `mc.fcisolver.trans_rdm1` before replacing it in the mixed case; the reference-search test cannot tell the mixed search from the singlet one; no test that `direct_spin1.trans_rdm1` equals the singlet solver's on the same CI vectors; the T-T refusal test runs a full SA-CASSCF though the refusal comes first; `TRIPLET_NAC_MESSAGE` advises `coupling = 'overlap'`, a `[dynamics]` key not usable with triplets until sub-project 3 (a neutral "request overlaps instead" would fit `single-point` too); a teardown "unraisable FileIO" warning once in a RED run.
- Task 5: the CASSCF continuity block is 1x1 and the path a rigid translation (section 10); `state_counts` computed twice in `compute`, and the response branch imports it from `pyscf_casscf` (it could live in `core/states.py`); the triplet-count check of `load_state` untested; `_state_overlaps`' unreachable guard is an electronic error rather than an assert; `response_triplet_expansion` and `casscf_expansions` run outside `wrap_failure` (as the singlet expansions always did).
- Task 6: an unused `write` parameter of the test helper `_finalized`; the below-S0 triplet test passed before the change (it guards the old path only); no non-finite triplet energy test; `triplet_states` `null` on the failure branch (now documented in USER_MANUAL 6); `finalize` indexes `rows[0]["triplet_energies_hartree"]` directly (an unreachable `KeyError`); the stage tests use the fake backend.
- Task 7: the module `skipif` of `test_sharc_triplets.py` hides the measured reason when `SHARC_HOME` is unset; ORCA `check=True` hides ORCA's stdout on a failure; the parsers raise a bare `IndexError` on a missing header; the water coordinates are copied into three test files; the exact-zero NAC test does not assert that the triplets are present; the S0-S1 NAC is checked at three components, two of them small.
