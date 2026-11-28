# The `dynamics` stage with triplets and spin-orbit coupling (triplets and SOC, sub-project 3): design

Status: written 2026-11-27; owner-approved 2026-11-27. Branch `feat/dynamics-soc` (from `main` `5f0f2c1`). The report is `docs/superpowers/reports/2026-11-27-dynamics-soc.md`.

> **Erratum (2026-11-28, after the implementation).** Where the work differs from this spec (the report's sections 9 and 11 have every number):
>
> - **`hopping = "none"` is allowed with triplets (section 2.3 said `hopping = "sharc"`):** the core refuses only `fssh` with spin-orbit coupling (P7a-2 ruling R4), `"none"` runs in the diagonal basis, and the swap test (a) runs hopping off. Every violation of the triplet rules is reported at once, but a rule of the `[dynamics]` table itself (`hopping = "fssh"` or `ekin_correction = "parallel_nac"` needs `coupling = "nacdr"`) is checked first and hides them (a deferred minor).
> - **`output.h5` naming (section 4.2; plan ruling PR1):** `hop_from`/`hop_to` hold the **spin-free** states before and after an accepted hop (the active spin-free state of the previous log row and of this row), and the diagonal indices are `hop_from_diag`/`hop_to_diag`, so `analysis/hops.py` works on spin-free labels unchanged. The other diagonal datasets are `active_diag`, `populations_diag`, `energies_diag`; `coefficients_*` and `probabilities` stay diagonal. A diagonal hop that keeps the spin-free state is stored as such (e.g. T1, T1).
> - **`steps.log` (section 4.1) keeps `energies` expanded** (N = n + 3m columns, the core's MCH energies) **and adds `energies_sf`** (n + m); `energies_diag`, `active_mch` (the spin-free index) and `populations_mch` are added as designed. `output.h5` `energies` is the spin-free one.
> - **The start search of swap (b) (sections 5.1, 6 item 1) was done in Task 9**, once the harness existed (pre-flight ruling); the rule was unchanged and fixed before any comparison. Every sample's SHARC run used `rngseed` 2029 (the project seed), so all 20 saw the same random stream.
> - **The schema-2 kept-row rule (not in the spec):** besides the rows kept today, every row whose active spin-free state differs from the previous row's is kept, so every spin-free change is a record at its own time (Task 7 fix, after the review showed a change on a dropped substep row being misclassified). Schema 1 is unchanged.
> - **`hops.csv` is unchanged (section 4.3 said "`hops.csv` the spin-free transitions"):** its accepted hops are spin-free by PR1; the changes of the active spin-free state without a hop are listed in `analysis.json` `transitions` only.
> - **Transitions (section 4.3) are counted over the complete trajectories up to the horizon**, like the hops; with no complete trajectory the counts are zeros. A same-state hop (T1 -> T1) is in the hop table but is not a transition. The analysis also checks that `n_singlets`/`n_triplets` give the file's labels and that every hop record follows the previous record (non-retryable errors naming `--rerun-from dynamics`).
> - **The `final_state` key** (the spin-free label of the last record) was added to the runner result and the `dynamics_summary.json` rows, with a `final state` column in `dynamics_summary.md`, triplet runs only (plan ruling PR6).
> - **A triplet initial state is refused by the stage** (a non-retryable error naming `--rerun-from selection`); `selection` already refuses triplets, so this is a guard (section 2.4).
> - **The CIS smoke test (section 5.3) asserts a triplet population, not a hop:** at those settings no CIS hop occurs (every probability is exactly 0, because the active diagonal population rises at every step); the test asserts the summed triplet population at 2 fs > 1e-4 (measured about 2.15e-3). Decided after the numbers were seen; no setting changed.
> - **The swap comparison settings (section 5.1):** our runner uses `overlap_threshold` 1e-9 and the energy-jump signal off, because SHARC has no step rejection and a halved step on our side would change the integrator and the replayed stream (decided after SHARC's search run of CIS sample 7 showed a raw overlap of 0.377, before any comparison of that sample); SA-CASSCF (a) starts from sample 0 because the (b) rule gave no start; the Wigner samples come from RHF/6-31G\* frequencies while the dynamics run in 6-31G.
> - **SA-CASSCF swap (b) is BLOCKED (section 5.1):** none of the 20 samples has a hop of any kind in SHARC's 200-step run (summed hopping probability 1.432 over the 20 runs; P(no hop of any kind) about 0.24 with independent streams). The SA-CASSCF hopping path is not compared with SHARC; the decision is the owner's. CIS (b) found a start (sample 7, the first singlet-triplet hop at step 64) and passes.
> - **The cost (section 3.5), measured in Task 1:** H2CS/6-31G with 8 threads, SA-CASSCF(4,3) 2S+2T 4.1-4.7 s per step (SOMF 15-16 %), CIS 3S+3T 2.2-2.3 s (SOMF about 30 %).

The third of three sub-projects that close O7 (triplets and spin-orbit coupling from PySCF). Sub-project 1 gave the backend triplet states (merged `d698271`); sub-project 2 gave the spin-orbit Hamiltonian between them (merged `3d5e402`). This one runs surface hopping on that Hamiltonian in the `dynamics` stage and analyses the result.

## 0. Owner decisions (2026-11-27, multiple choice and approved sections)

1. **Reach:** the `dynamics` stage and the `analysis` stage. The stage runs in the diagonal basis and stores spin-free (MCH) populations per state (triplets summed over Ms) beside the diagonal ones; the analysis reports singlet and triplet populations, singlet-triplet transitions and the fit.
2. **Gradients:** every spin-free state's gradient at every step by default; an optional `[dynamics] eselect_eV` limits them to the states within that energy of the active state, through the core's existing `eselect` masking (P7a-2).
3. **The NAC term of the diagonal-basis force:** with triplets the stage requires `gradient_transformation = "none"` and `coupling = "overlap"` (refused otherwise, exit 2). Exact for singlet-triplet mixing (singlet-triplet NACs are zero by spin); it drops only the terms in which one diagonal state mixes two singlets or two different triplets at once (second order in the spin-orbit coupling). The dropped term is measured on the test molecule as a report.
4. **Validation:** the SHARC swap test (`sharc.x` driving our PySCF backend through our own `QM.out` provider), asserted. The analytic fake-backend pipeline, restart and resume tests are development tests (they are written anyway), not counted as validation; no asserted PySCF energy-conservation test and no ISC ensemble report.
5. **System:** thioformaldehyde (H2CS), 6-31G, the geometry of sub-project 2 (`tests/data/thioformaldehyde.xyz`), with SA-CASSCF(4,3) 2 singlets + 2 triplets and CIS 3 singlets + 3 triplets (the CIS at `td_conv_tol` 1e-7: it does not converge at the default, a documented limitation of sub-project 2).
6. **The default fit with triplets:** the summed population of the excited singlets (S1, S2, ...); unchanged (every excited state) without triplets.
7. **Approach A:** `BackendModel` is extended (one adapter for both cases) and a new pure-NumPy module `trajectory/spin_expansion.py` expands the backend's spin-free data into the core's spin-expanded basis; the core's existing diagonal-basis path does the rest.
8. **Sections 1-4 of the design** were approved as presented.

## 1. Why, and what exists

- The dynamics core propagates with a complex MCH Hamiltonian in the diagonal basis (P7a-2): `diagonal_basis` tracks U, `gradient_tensor` builds G from spin-free energies, gradients and NACs (ruling R3), `propagate_diagonal` with `coupling = "overlap"` uses local diabatization with the overlap matrix S, `eselect_Eh` masks gradients, restart format 2 stores U. Validated on analytic and LVC models only.
- The backend (sub-projects 1 and 2) returns spin-free energies `S0..S(n-1), T1..Tm`, analytic gradients per spin-free state, block-diagonal spin-free overlaps (exact-zero singlet-triplet blocks, phases per block), no triplet-triplet NACs (PySCF 2.14; no NACs at all for TDA/TDDFT), and with `Request(soc=True)` the phased spin-expanded Hamiltonian `hamiltonian` in the canonical order of `dynamics/spin.py` (S0 ... S(n-1), then T1 ... Tm with Ms = -1, then 0, then +1), Hermitian, spin-free energies on the diagonal.
- `trajectory/model.py:BackendModel` is singlet-only today (U `None`, the active gradient only, the other rows NaN); `stages/dynamics.py:check_config` refuses `triplets > 0` (exit 2).

## 2. Input and rules

### 2.1 Turning it on

`[electronic] triplets > 0` together with a `[dynamics]` table, today refused, becomes allowed. The spin-orbit operator is `[electronic] soc_operator` (sub-project 2; `"somf"` by default).

### 2.2 New `[dynamics]` keys

- `gradient_transformation`: `"ngt"` (default, the `DynamicsOptions` default) or `"none"`. Without triplets the two are identical in effect (U is `None`).
- `eselect_eV`: off (`None`) by default; finite > 0 otherwise; converted to `eselect_Eh` for the core.
- Both are left out of the fingerprint at their defaults (the `hop_energy` precedent), so no singlet project changes its fingerprint.

### 2.3 Rules for a run with triplets (`DynamicsStage.check_config` and the schema validator; every violation reported at once, exit 2, each message naming the key)

- `coupling = "overlap"` (no triplet-triplet NACs);
- `gradient_transformation = "none"` (owner decision 3);
- `hopping = "sharc"` and `ekin_correction = "parallel_vel"` (the core's limits with spin-orbit coupling, ruling R4 of P7a-2);
- `soc_operator = "effective_charge"` only for molecules of H, C, O, S (the check moves before any trajectory; today it would come from inside `compute`).

### 2.4 Unchanged

- Initial states: trajectories start in the singlet chosen by `selection`, an MCH state (P7a-2 ruling R7).
- Without triplets: the requests, U (`None`), `steps.log`, `output.h5`, the analysis, every fingerprint and every stage version. No existing project becomes stale.

## 3. The adapter and the step

### 3.1 `trajectory/spin_expansion.py` (new; NumPy only; imports nothing but NumPy and `dynamics/spin.py`)

The one statement, for the stage, of the canonical expanded order:

- `expanded_size(n_s, n_t) = n_s + 3 n_t`; expanded index of (T_J, M) = `n_s + M_index * n_t + J` (M_index 0, 1, 2 for Ms = -1, 0, +1), of S_I = I;
- `expand_energies(e_sf)`, `expand_gradients(g_sf)`: each triplet's spin-free value repeated in its three Ms rows;
- `expand_overlaps(s_sf)`: block diagonal; the singlet block as is; the triplet block `kron(I_3, S_TT)` (M outermost); singlet-triplet blocks zero;
- `spin_free_index(k)`: the spin-free state of expanded index k;
- `spin_free_populations(p_exp)`: sums over Ms.

### 3.2 `BackendModel` with triplets

- `state_counts = (n, 0, m)` (`dynamics.spin` counts per multiplicity).
- Each evaluation requests: energies, `soc = True`, overlaps (always), the gradients (every spin-free state, or the `eselect` window of 3.3), no NACs.
- The snapshot: the expanded energies and gradients, zero NACs (n_exp, n_exp, f), `overlaps` = `expand_overlaps` of the Lowdin-orthonormalised spin-free blocks, `hamiltonian` = the backend's phased spin-orbit Hamiltonian. The core then finds an off-diagonal H and takes its diagonal-basis path (U tracked, restart format 2 stores it).
- The raw smallest singular value of the spin-free overlap matrix stays the overlap rejection signal.
- Every finiteness check of today applies to the expanded data and to `hamiltonian`; a missing `hamiltonian` is a non-retryable electronic error.

### 3.3 The `eselect` window, the core's mask exactly

- The runner gives the adapter the tracked U and the active state before each evaluation (as it gives `active` today).
- The adapter computes energies, H and overlaps at q' first; it then applies the core's own `diagonal_basis(H', U)` and `selection_mask` (the same functions, so the same mask) around the active state's diagonal energy, and fetches the gradients of the kept spin-free states with `compute_more`.
- After a hop, `refresh_active_gradient` generalises to "compute the missing gradients of the states the new active state's mask keeps".
- A state the mask keeps without a computed gradient is a non-retryable electronic error; the NaN rows stay as the tripwire.
- With `eselect_eV` off every spin-free gradient is requested in the first `compute` and no `compute_more` is needed.

### 3.4 Unchanged

- The step: the core's P7a-2 diagonal-basis order (plan Method reference 11 of P7a-2), `hop_energy` as today (`"current"` default, `"previous"` SHARC's rule).
- Rejection: the four signals (electronic; overlap on the raw smallest singular value of the spin-free overlaps; energy jump on the diagonal-basis E_tot; CASSCF rung), halving, `max_step_halvings`, restart generations, extension (O16), the outcomes.
- `stop_after_s0_fs`: "in S0" when the dominant MCH state of the active diagonal state (`mch_state(U, active)`) is S0.

### 3.5 Cost

Every step pays one SOMF build (sub-project 2: 10-14 s on benzene against 0.3-0.7 s for the SCF, measured by its final review) plus n + m gradients. Documented; `soc_operator = "effective_charge"` is the cheaper operator. Task 1 measures the time per step for the test system.

## 4. Outputs and analysis

### 4.1 `steps.log` with triplets

The diagonal-basis columns the core works in: `active`, `hop_from`/`hop_to` (diagonal indices), `c_real`/`c_imag` and `probabilities` (n_exp each), and new `energies_diag` (n_exp). New MCH columns: `active_mch` (the spin-free index of `mch_state(U', active)`), `energies` (the spin-free energies, n + m), `populations_mch` (the spin-free sums of |U' c|^2, n + m, computed when the record is made, so U is not logged). Without triplets the record layout is unchanged.

### 4.2 `output.h5` with triplets (`schema_version` 2)

- `state_labels`: the spin-free labels (S0, S1, T1, T2, ...).
- `active`: the spin-free MCH active state; `populations`: the spin-free MCH quantum populations; `energies`: the spin-free energies.
- Beside them: `active_diag`, `populations_diag`, `energies_diag`, the diagonal `hop_from`/`hop_to`; attribute `n_singlets`, `n_triplets`, `soc_operator`.
- Without triplets: `schema_version` 1 and today's bytes.

### 4.3 Analysis

- The estimators (complete, censored, hold-in-S0; bootstrap; the delayed mono-exponential fit) read the spin-free `active` and `populations`, so they work unchanged with triplet states.
- `fit_states` default: the excited singlets when triplets exist (owner decision 6); unchanged otherwise. `fit_states` still accepts excited singlet labels only.
- Transitions: spin-free changes of the active MCH state between records, by kind S-S, S-T, T-S, T-T, split into changes at an accepted hop and changes without a hop (the MCH character can change while the diagonal state stays); frustrated hops per originating spin-free state.
- `report.md`: an intersystem-crossing block (the S-T and T-S counts; the first S-to-T time per trajectory, censored like the S0 arrival); `hops.csv` the spin-free transitions.
- No stage version changes: singlet projects give identical results, and triplet projects did not exist.

## 5. Validation (bounds fixed here, before any run)

### 5.1 The SHARC swap test (asserted; markers `sharc`, `pyscf`, `slow`; needs `SHARC_HOME`, `SHARC_PYTHON`)

- `sharc.x` drives our PySCF backend through our own `QM.out` provider (the P7b/P9b swap harness of `tools/sharc_compare/`), extended to write the complex spin-orbit Hamiltonian, the expanded gradients and the expanded overlaps in SHARC's `QM.out` format (the format from the public SHARC 4.1 manual and a probe run, Task 1; clean room unchanged).
- SHARC settings: `coupling overlap`, `nogradcorrect`, the diagonal surface, `ekincorrect parallel_vel`, EDC (C = 0.1 Eh), SHARC-type hopping, `endpoint` substep interpolation; ours the matching options.
- Both methods (SA-CASSCF(4,3) 2S+2T; CIS 3S+3T at `td_conv_tol` 1e-7), dt 0.5 fs, starting from one Wigner sample of the test system (seed and sample chosen in Task 1 by the rule below, before any comparison).
- **(a) Hopping off**, steps 0-40: positions, velocities, total energy, the active state's diagonal energy and the spin-free MCH populations (all invariant under the gauge of U). **Bounds: |dq| <= 1e-5 bohr, |dv| <= 1e-6 a.u., |dE| <= 1e-6 Eh, |dP| <= 1e-5**, per step and component.
- **(b) Hopping on**, SHARC's random numbers replayed into ours, `hop_energy = "previous"` (SHARC's rule, P9b): the same bounds and the same active spin-free state, from step 0 to the first singlet-triplet hop plus 20 steps (at most 200 steps).
- **The start rule for (b):** the first sample (by index, under the project seed of Task 1) whose SHARC run has a singlet-triplet hop within 200 steps; if none of the first 20 samples has one, (b) is reported BLOCKED to the owner (no window or setting is changed).
- A miss is reported with the numbers (BLOCKED), not loosened.

### 5.2 Report (not asserted)

The force term that `gradient_transformation = "none"` drops, along the (a) trajectory of the SA-CASSCF case: the singlet-singlet NAC term of `gradient_tensor` (CASSCF singlet-singlet NACs computed for this measurement only) projected through U into the active diagonal state, relative to the force actually used.

### 5.3 Development tests (asserted, mostly fast tier)

`spin_expansion` (order, `kron` blocks, round trips); the adapter over a fake backend serving the LVC spin-orbit model of P7a-2 (spin-free view of its states); the `eselect` window equal to the core's mask, and the refresh after a hop; the analytic spin-orbit pipeline through dynamics and analysis (populations against a direct `model-dynamics` LVC run of the same model and seeds; restart and kill/resume bit-identical; extension); `output.h5` schema 2; the analysis statistics on canned data; the `check_config` refusals; a short PySCF smoke run per method on H2CS (finite records, labels, a hop present when a replayed zero random stream forces it); `triplets = 0` unchanged (the existing tests unmodified).

All new tests: no bit-for-bit BLAS comparison, no reliance on an eigenvector's sign; checked once with `OPENBLAS_CORETYPE=SandyBridge`.

## 6. Order of work (one commit per task)

1. Facts (a measurement): SHARC's `QM.out` format for a spin-orbit Hamiltonian, expanded gradients and overlaps (public manual, a probe run); the time per step for H2CS in both methods; the start sample of 5.1 (b).
2. `trajectory/spin_expansion.py`.
3. Configuration: the two keys, the rules of 2.3, the fingerprint rule.
4. The fake backend with spin-orbit coupling; `BackendModel` with triplets; the `eselect` window and the refresh.
5. The step log and `output.h5` schema 2.
6. The stage (the refusal removed) and the canned spin-orbit pipeline.
7. The analysis (fit default, transitions, report).
8. The PySCF smoke tests.
9. The swap harness and the asserted swap tests.
10. The dropped-term report.
11. The controller's test tiers.
12. Documents: USER_MANUAL, CLAUDE.md, CHANGELOG, a validation report.

## 7. Not in scope

- Triplet initial states (selection stays singlet), `fssh` and `parallel_nac` with spin-orbit coupling (R4), `coupling = "nacdr"` with triplets, the NGT term with triplets.
- Quintets and higher, doublets, open-shell references.
- A direct comparison with `SHARC_PYSCF.py` (it refuses triplets, sub-project 1).
- An ISC ensemble study (owner decision 4).
- Reusing the SOMF operator between steps.
