# Triplet states in the PySCF backend (triplets and SOC, sub-project 1): design

Status: written 2026-11-26; owner-approved 2026-11-26. Branch `feat/triplets-backend` (from `main` `478d3f2`). The report is `docs/superpowers/reports/2026-11-26-triplets-backend.md`.

> **Erratum (2026-11-26, after the implementation).** Where the work differs from this spec:
>
> - **Plan rulings** (`docs/superpowers/plans/2026-11-26-triplets-backend.md`, binding where they differ):
>   - **PR1 (section 3.4):** TDA/TDDFT triplets are expanded in their **Ms = 0** component, `sum_ia X_ia (|i->a, alpha> - |i->a, beta>)`, in the singlets' determinant layout, not in Ms = +1. CASSCF triplets keep Ms = +1. The cross blocks are still exactly zero.
>   - **PR2:** Task 2 accepts `triplets > 0` before the backend supports it and before `dynamics` refuses it; the branch is merged only when complete.
>   - **PR3 (section 4 item 2):** each FD test compares a fixed set of Cartesian components (three per state for the response tests, two for CASSCF).
>   - **PR4 (section 4 item 3):** the NAC test system is water CAS(4,4)/STO-3G with 2 singlets and 2 triplets (ethylene has one triplet).
>   - **PR5 (section 3.4, snapshot):** `save_state` writes format 1, as before, without triplets, and format 2 (format 1 plus `t_alpha`, `t_beta`, `t_coeffs`) with them.
>   - **PR6 (section 2.3):** the `S^2` column of the `single-point` table appears only with triplets.
>   - **PR7 (section 2.2):** the `reference` group of `excitations.h5` has no triplet data; `excitations.json` has the reference triplet excitation energies.
>   - **PR8:** a failing triplet NAC stops the task and the controller rules. **It was triggered.**
> - **Triplet-triplet NACs are not available (sections 3.2 and 4 item 3; owner decision 7's default "checked by finite differences" is not reachable).** PySCF 2.14's SA-CASSCF NAC code raises an `AssertionError` for a pair of triplets in a mixed-spin state average (it uses `mc.nelecas` for every root). The program refuses such a pair with a `ValueError`, in `casscf_nacs` and at request time in the backend, before any PySCF call. Singlet-singlet NACs stay (FD-checked, below); singlet-triplet NACs are exact zeros. Section 4 item 3 is therefore: the singlet NAC of a state average with a triplet against FD overlaps (1e-4, unchanged), the refusal, and the exact zeros. **Consequence for sub-project 3:** `coupling = "nacdr"` with triplets is not possible unless this is solved; `coupling = "overlap"` works.
> - **Section 4 item 5:** `SHARC_PYSCF.py` refuses non-singlet states (`NotImplementedError: Not implemented for states other than singlets!`, Task 1), so the SHARC test is a skip with that reason.
> - **Section 2.3:** a TDA/TDDFT triplet row has no `<S^2>` (`null` in the JSON, `-` in the table); the response solve gives none.
> - **Decided after numbers were seen** (no spec-fixed bound changed; the report's section 8 has every number):
>   - the direct-mix test asserts the SA energy at 1e-8 Eh and each state at 1e-6 Eh (the plan had 1e-8 per state; per-state energies scatter across BLAS kernels by up to 2.5e-7 at `conv_tol_grad` 1e-6);
>   - the resume test of item 4 is one save-and-load chain (two independent backends have different sign frames);
>   - the CASSCF continuity test of item 4 uses ethylene SA-CASSCF(2,2)/6-31G\* with 2 singlets and 1 triplet (water CAS(4,4)/STO-3G gave 0.983 for the triplet block, and its singlet block drops to 0.977 also without triplets);
>   - the singlet NAC check uses ethylene 2S+1T (water CAS(4,4) is off by 1.6e-3 1/bohr with triplets and by 2.3e-2 without them), **with the FD step 1e-3 bohr instead of the existing FD-overlap step 1e-4 bohr**: at 1e-4 the ethylene check measured 4.6e-4 1/bohr and fails the bound 1e-4 (2.8e-5 at 3e-4, 1.37e-5 at 1e-3); at 1e-4 the triplet case is also worse than the singlet-only one (6-31G\* 4.6e-4 against 2.3e-4; 6-31G 8.4e-4 against 8.6e-5). That the difference is FD noise is an inference from its fall as h grows;
>   - the ORCA inputs of item 1 gained `NoFrozenCore` (water) and `etol 1e-10` / `gtol 1e-6` (ethylene CASSCF), so ORCA computes the same method at the same convergence.
> - **Measured** (asserted): ORCA largest difference 1.07e-8 Eh; FD gradients 8.59e-7 Eh/bohr; the singlet NAC 1.37e-5 1/bohr (at h 1e-3); triplet-block continuity 0.99304 (TDA) and 0.992663 (CASSCF, a 1x1 block); resumed overlaps 6e-15.
> - **Section 4 item 6 (`triplets = 0` unchanged):** the existing tests passed without modification of their assertions (two test helpers were extended) in the fast tier, the `pyscf`/`slow` tier without the three analytic-model validation files, and `-m sharc` (report section 3). The three analytic-model files, the ORCA frequency test and the full suite in one run were not run on the branch.

This is the first of three sub-projects that close O7 (triplets and spin-orbit coupling from PySCF): (1) triplet states in the backend, no spin-orbit coupling (this spec); (2) the spin-orbit coupling matrix elements, validated against ORCA; (3) the `dynamics` stage with spin-orbit coupling. Each has its own spec, plan, review and merge.

## 0. Owner decisions (2026-11-26, multiple choice and approved sections)

1. **Next sub-project:** triplets and spin-orbit coupling from PySCF (chosen over the post-hop losses, revisiting IPCW and robustness work).
2. **Methods:** SA-CASSCF and TDA/TDDFT, both in each sub-project.
3. **Split:** three sub-projects, as above.
4. **Stage reach of sub-project 1:** the backend, `single-point` and the `excitations` stage take triplets; `optimize` and `frequencies` use S0 only (for CASSCF from the state average that includes the triplets); `selection` and `analysis` are unchanged; `dynamics` refuses `triplets > 0` until sub-project 3.
5. **The SA-CASSCF state average:** one orbital set, equal weight 1/N per spin-free state (singlets and triplets), every triplet counted once.
6. **Asserted validation:** ORCA 6.1.1 energies (black box), finite-difference gradients, overlap continuity, and `SHARC_PYSCF.py` (black box) if it supports triplets.
7. **Representation (approach A):** spin-free triplet states `T1…Tm` beside the singlets in the backend's results; the spin expansion (Ms = -1, 0, +1) stays in the dynamics. With the controller's defaults, approved: the mixed-solver SA-CASSCF with a per-multiplicity spin guard and start rule, a separate triplet response solve on the same root ladder, CASSCF triplet-triplet NACs checked by finite differences, and `triplets = 0` exactly as today.
8. **Sections 1-3 of the design** (input and stages, the backend, validation and order) were approved as presented.

## 1. Why, and what PySCF provides

The dynamics core already propagates with spin-orbit coupling in the diagonal representation on LVC models (P7a-2), but the PySCF backend is singlet-only: `[electronic] triplets > 0` is refused at load time (`config/schema.py`), SA-CASSCF uses a singlet-only CI solver with a spin guard (`electronic/pyscf_casscf.py`), the CASSCF reference search counts only all-singlet roots, and the stages read the singlet labels.

### 1.1 Measured during the design (throwaway probe, PySCF 2.14.0, seen before the rules were fixed)

Water at the conftest geometry, STO-3G:

- RHF triplet TDA (`td.singlet = False`): energies 11.0876, 13.3911, 13.8193 eV; the analytic gradient of the first triplet runs (max |g| 0.298 Eh/bohr).
- RKS-B3LYP triplet TDDFT: the analytic gradient runs (max |g| 0.326).
- `mol.intor` gives the Breit-Pauli spin-orbit integrals `int1e_pnucxp` (3, n, n), `int2e_p1vxp1` (3, n, n, n, n) and `int1e_prinvxp` (for sub-project 2).
- `mcscf.state_average_mix_` over a singlet solver (2 roots) and a triplet solver (spin 2, 1 root), CAS(4,4), equal weights: converged (state energies -74.975944, -74.531925, -74.596721 Eh); the analytic gradient of the triplet root runs (max |g| 0.327); `nac_method` exists.
- `fci.direct_spin1.trans_rdm1s` exists (for sub-project 2).
- PySCF has no spin-orbit matrix-element module (`pyscf.prop.zfs` and the like are not installed); sub-project 2 builds them from the integrals, from the literature.

## 2. Input and stages

### 2.1 `[electronic] triplets`

- The load-time refusal of `triplets > 0` is removed, for `casscf`, `tda` and `tddft`.
- For CASSCF the cross-field check extends to the triplets: `triplets` must not exceed the number of triplet configuration state functions of CAS(`nelecas`, `ncas`) (the Weyl count with S = 1), as `singlets` is checked against the singlet count today.
- `triplets` is already part of the electronic settings that are fingerprinted (`electronic_fingerprint`, `ground_state_fingerprint`), so projects with `triplets = 0` keep their fingerprints.

### 2.2 Stages with `triplets > 0`

| Stage | Behaviour |
|---|---|
| `optimize`, `frequencies` | S0 only, through `factory.ground_state_settings`. For CASSCF the state average keeps the triplets, so S0 changes when triplets are added (the fingerprint changes with `triplets`). For TDA/TDDFT the ground-state settings set `singlets = 1` and `triplets = 0` (the SCF ground state is unchanged). |
| `excitations` | Records the triplet excitation energies of every sample in a new dataset `triplet_energies` (samples x triplets, eV, NaN rows for failed samples, `units` attribute) and the triplet labels; S0 -> T oscillator strengths are 0 (spin-forbidden without spin-orbit coupling) and are not written. The spectrum and the selection candidates stay singlet-only. A non-finite or non-positive triplet excitation energy fails the sample like a singlet one. `excitations.json` reports the triplet means and standard deviations beside the singlet ones. |
| `selection` | Unchanged (initial states are excited singlets). |
| `dynamics` | `check_config` refuses (exit 2): `Stage 'dynamics' with [electronic] triplets > 0 is planned (spin-orbit coupling, sub-project 3); set [electronic] triplets = 0`. |
| `analysis` | Unchanged. |

### 2.3 `single-point`

The table and the JSON (`--json`) gain the triplet rows: energy, excitation energy, gradient when requested, `<S^2>`; S0 -> T transition dipoles 0 and oscillator strengths 0.

### 2.4 Unchanged for `triplets = 0`

The same code path (the singlet-only CI solver, no triplet response solve) and the same numbers; no stage version changes and no existing project goes stale. `model-dynamics` is unchanged.

## 3. The backend (`electronic/`, `overlaps/`)

### 3.1 States

`PySCFBackend.states` is `S0…S(n-1), T1…Tm` (`core.states.states_from_counts`). Every per-state field of `ElectronicResult` (energies, gradients, phases, `<S^2>` in the metadata, NAC pairs, dipoles) is keyed by these labels.

### 3.2 SA-CASSCF (`pyscf_casscf.py`)

- **Solver:** with `triplets > 0`, `mcscf.state_average_mix_` over a singlet solver (`fci.solver(mol, singlet=True)`, n roots) and a triplet solver (`spin = 2`, m roots; Ms = +1 in the active space), weights 1/(n + m) each. With `triplets = 0` the existing singlet-only solver, unchanged.
- **Spin guard:** each root against its own multiplicity: singlets `|<S^2>| <= 1e-3`, triplets `|<S^2> - 2| <= 1e-3` (`SPIN_TOLERANCE`); a root of the wrong spin is an `ElectronicStructureError`.
- **Ladder:** the four rungs stay; on the spin-penalty rung each solver gets its own `fix_spin_` (ss = 0 for singlets, ss = 2 for triplets).
- **Start rule:** the reference search (`find_reference`) solves the same mixed state average; a guess counts only if it converges with every root of the right spin. The default guess still comes first (ruling C6 of the start rule). `CASSCF_START_RULE` stays `"reference-1"` (the fingerprint already carries `triplets`).
- **Gradients:** `nuc_grad_method` of the mixed state average, for the requested states.
- **NACs:** between states of the same multiplicity (singlet-singlet and triplet-triplet pairs, I < J within a block), from the mixed state average's `nac_method`; singlet-triplet NACs are zero vectors, with no PySCF call.
- **Transition dipoles:** S0 -> Sk as today; S0 -> Tk zero.

### 3.3 TDA/TDDFT (`pyscf_response.py`)

- **Solve:** a separate triplet response solve on the same SCF (`td.singlet = False`), with the same root ladder (each rung solves `min(m + extra, available)` roots and keeps the lowest m; `_LADDER` of P7b). `td_conv_tol` applies to both solves.
- **Gradients:** from the triplet response object's `nuc_grad_method`.
- **NACs:** none (overlaps only), as today.
- **Transition dipoles:** S0 -> T zero.

### 3.4 Overlaps and phases

- **Expansions:** singlets keep their current determinant expansion (Ms = 0). CASSCF triplets use the triplet solver's CI vectors, already Ms = +1 determinants of the active space. TDA/TDDFT triplets are expanded in their Ms = +1 component: the beta i -> alpha a single excitations, coefficient proportional to X_ia (X only, as for the singlets), renormalised.
- **The overlap matrix** is block-diagonal by multiplicity: each block from `overlaps.determinants.state_overlaps` with its own alpha/beta electron counts; the cross blocks are exactly zero. Truncation (`overlap_norm_kept`) keeps the union of the important determinants within each block separately.
- **Phase tracking:** per state, with the existing rules (`overlaps.tracking`).
- **Snapshot:** `_Snapshot` holds the expansions of both blocks; `save_state`/`load_state` store the triplet block too; `triplets` is already in the snapshot fingerprint.

### 3.5 Errors

Triplet failures follow `pyscf_errors.wrap_failure`: non-convergence and a wrong spin are non-retryable `ElectronicStructureError`; `MemoryError`/`OSError` raised inside PySCF are retryable.

## 4. Validation (asserted; tolerances fixed here, before any run)

1. **ORCA 6.1.1, black box** (marker `orca`, runs only with `ORCA_EXE`): water TDA-HF/STO-3G triplets (CIS triplets) and ethylene SA-CASSCF(2,2)/6-31G\* over two singlets and one triplet with equal weights (ORCA states the weights per multiplicity). ORCA runs without RI or COSX approximations, with tight SCF and CASSCF convergence, the same basis and geometry (the existing frequency comparison's harness pattern). **Rule: every state energy within 1e-6 Eh.**
2. **Finite-difference gradients** (markers `slow`, `pyscf`): central differences of each requested triplet energy, step 1e-3 bohr, a fresh backend per displacement; water triplet TDA-HF and TDA-B3LYP (`grid_level` 5), and the ethylene mixed SA-CASSCF (one singlet and one triplet gradient). **Rule: max |analytic - FD| <= 1e-5 Eh/bohr.**
3. **Triplet-triplet NACs** (CASSCF, two triplets): against finite differences of the phased overlaps, as the existing singlet NAC test. **Rule: within 1e-4 1/bohr; singlet-triplet NACs exactly zero.**
4. **Overlap continuity** along a 5-point path (steps of 0.01 bohr in every coordinate) for both methods. **Rules:** the triplet block's smallest singular value >= 0.99; after phase tracking the diagonal is positive; the singlet-triplet entries are exactly 0; a backend resumed through `save_state`/`load_state` gives the same phased overlaps (to 1e-10) as one that was not interrupted.
5. **`SHARC_PYSCF.py`, black box** (markers `sharc`, `pyscf`): Task 1 runs it with triplets requested. If it supports them: **triplet energies within 1e-8 Eh and gradients within 1e-7 Eh/bohr** (the singlet chain's tolerances), on the same systems as item 1 where its interface allows. If it does not: the test records "not available", is skipped with that reason, and the report says so; no substitute is invented.
6. **`triplets = 0` unchanged:** the existing singlet tests, the golden test and the SHARC singlet comparisons stay green unmodified; a test asserts that the `triplets = 0` path makes no triplet solve and uses the singlet-only solver.
7. **Stages and input:** the excitations stage writes `triplet_energies`, and its spectrum and selection equal those of a `triplets = 0` run on the same samples; the dynamics stage refuses `triplets > 0` (exit 2, the message of 2.2); `single-point --json` carries the triplet rows; the input validation covers the CASSCF triplet Weyl count and the removed refusal.

All new tests follow the project rule: no bit-for-bit BLAS comparison and no reliance on an eigenvector's sign; checked once with `OPENBLAS_CORETYPE=SandyBridge`.

## 5. Order of work (one commit per task)

1. Run `SHARC_PYSCF.py` with triplets requested and record what it supports and its formats (a measurement, no code).
2. Input: the refusal removed; the CASSCF triplet Weyl check.
3. TDA/TDDFT triplets: the triplet solve, gradients, the Ms = +1 expansion, overlaps.
4. SA-CASSCF triplets: the mixed state average, the spin guard, the ladder and the start rule, gradients, NACs, the expansion.
5. Backend integration: snapshot, phases, `save_state`/`load_state`, `single-point`.
6. Stages: `excitations`, the `dynamics` refusal, `optimize`/`frequencies` through the ground-state settings.
7. Validation: ORCA, finite-difference gradients and NACs, overlap continuity, SHARC if available.
8. Documents: USER_MANUAL, CLAUDE.md, CHANGELOG, and a validation report `docs/superpowers/reports/<date>-triplets-backend.md`.

## 6. Not in scope

Spin-orbit coupling matrix elements (sub-project 2); a complex Hamiltonian and spin-expanded states; dynamics with triplets (sub-project 3); open-shell references and triplet ground states; doublets; triplet initial states in `selection`.
