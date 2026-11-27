# Spin-orbit coupling matrix elements from PySCF (triplets and SOC, sub-project 2): design

Status: written 2026-11-27; owner-approved 2026-11-27. Branch `feat/soc-matrix-elements` (from `main` `0e269c9`). The report is `docs/superpowers/reports/2026-11-27-soc-matrix-elements.md`.

> **Erratum (2026-11-27, after the implementation).** Where the work differs from this spec (the report's sections 9 and 11 have every number):
>
> - **The effective nuclear charges (section 3.1; section 6 "beyond Ar"):** the table holds **only H 1.0, C 3.6, O 5.6, S 13.6**, the values ORCA 6.1.1 uses, measured in Task 1 by the ratio of its `SOCType 1` to its one-electron Breit-Pauli elements on single atoms (each within about 1e-4). They were **not checked against the tables of Koseki et al.**, which could not be read, and no value was taken from memory or another program. Any other element is a `ConfigError` (*no established effective nuclear charge for X (available: H, C, O, S); use soc_operator = 'somf'*), raised before any SCF; ghost atoms are refused too. Task 1's check "ORCA's effective charges equal the literature values" is therefore not established.
> - **The sign of i (plan Conventions):** fixed from the operator definition, `h[k] = +i (alpha^2/2) A[k]` with the integrals as PySCF returns them, not from ORCA's printed sign of `<T1|HSO|S0>`, which follows ORCA's arbitrary CI phase (ours `+102.25i` with T1's largest X coefficient positive, ORCA `-102.25i`). It is invisible to every root sum of squares; a committed test pins it and the x, y, z order.
> - **`soc.assemble_hamiltonian`** (section 3.3; the plan's Conventions) was built in Task 5 (no task listed it). Besides Hermiticity (1e-12 Eh, `AssertionError`) it checks that the triplet-triplet block has no spin-orbit element on its diagonal above 1e-12 Eh, then sets the diagonal to the spin-free energies.
> - **The resume bound (section 4 item 4) applies to the off-diagonal spin-orbit elements only** (owner decision 2026-11-27, asked after the numbers were seen): the TDA spin-free energies on the diagonal differed by 1.330e-8 to 1.336e-8 Eh after a resume in 9 of 14 runs (the SCF from the same saved density stops one cycle apart at the default gradient criterion), while the spin-orbit elements stayed within 2.17e-11 Eh and CASSCF passed. The diagonal difference is reported, not asserted, and the finding is a documented limitation.
> - **ORCA TDA `nroots 6`** (section 4 items 1-2): ORCA's `nroots 3` skips the H2CO S3 (Task 1), so ORCA is asked for 6 roots and the states are mapped by excitation energy (asserted within 1e-6 Eh). ORCA's TDA prints no triplet-triplet elements, so item 1's T-T comparison covers SA-CASSCF only.
> - **The ORCA test's CIS `td_conv_tol` is 1e-7** (decided after one run): our backend refused H2CS CIS 3S+3T/6-31G as unconverged at 1e-10, and PySCF's Davidson does not converge it at any value from 1e-10 to 1e-8 (the default), because it stops on linear dependency at `td.lindep` 1e-12. No H2CS coupling had been seen; no bound changed.
> - **The root sum of squares (section 4 item 1)** counts each distinct matrix element of a pair once: an S_I-T_J pair its 3 elements, a T_J-T_J' pair (J < J') its 9, a T_J-T_J pair the 3 with M < M'. ORCA's TDA side is its three Cartesian columns; its CASSCF side every printed `NONZERO SOC MATRIX ELEMENTS` line of the pair, keyed by the unordered pair of states.
> - **TDDFT (section 3.2): the X-only approximation as designed:** X rescaled to `x = X / sqrt(2 sum X^2)` (measured `sum X^2` 0.5001-0.5216 for TDDFT, 1/2 for TDA), Y dropped; checked against the brute-force reference with the same response object, not against another program.
> - **The SA-CASSCF SOMF density (section 3.1):** ORCA's is the state-averaged density with the CASSCF weights, like ours (Task 1; an inference from a measured linear weight dependence); the effect of the choice (about 0.1 %) is below the 0.2 % rule, so the comparison cannot tell choices apart.
> - **The input key (section 2.1):** the default `soc_operator` is left out of every dump of the section (a `model_serializer`), so `config.json` and every existing fingerprint stay byte-identical; `ground_state_fingerprint` always drops it; a non-default value is hashed by `excitations` and `dynamics` even with `triplets = 0`.
> - **The JSON key (sections 2.2-2.3) is `hamiltonian_hartree`** (renamed in the final fix wave so that the key names its unit, like the other keys of the file); the Python attribute stays `ElectronicResult.hamiltonian`. **`Capabilities.soc`** (default false) was added: true for `PySCFBackend` with triplets; the test backend `ModelBackend` refuses `Request(soc=True)`. The `single-point` table prints a second column header before the T-T rows.
> - **The energy bound 1e-6 Eh of the ORCA test** (energies first) is not in section 4: it was set in the plan (Task 6), before any task ran.
> - **The brute-force tests (section 4 item 3)** also use ethylene SA-CASSCF(2,2)/STO-3G 2S+1T (an odd core) and TDDFT-B3LYP/STO-3G 2S+2T, and two random operators besides SOMF.
> - **References (section 7), verified with Consensus:** Hess, Marian, Wahlgren, Gropen, *Chem. Phys. Lett.* 251, 365 (1996), doi:10.1016/0009-2614(96)00119-4; Neese, *J. Chem. Phys.* 122, 034107 (2005), doi:10.1063/1.1829047; Koseki, Schmidt, Gordon, *J. Phys. Chem.* 96, 10768 (1992), doi:10.1021/j100205a033; **Koseki, Gordon, Schmidt, Matsunaga**, *J. Phys. Chem.* 99, 12764 (1995), doi:10.1021/j100034a013 (the author order of section 7 was wrong). *J. Phys. Chem. A* 102, 10430 (1998) is Koseki, Schmidt, Gordon on the **transition metals** and has no H-Ar values; it is not cited for the table.
> - **Measured** (asserted): ORCA largest ratio |diff|/allowance 0.055 (CIS) and 0.004 (CASSCF) over 32 S-T and 4 T1-T2 pairs with real couplings (20 S-T and 8 T_J-T_J pairs zero on both sides); brute-force reference 9.0e-16 Eh; Hermiticity 2.71e-20 Eh; invariance 6.15e-11 relative; continuity 0.0155 of the allowance, no sign change; the resumed spin-orbit elements 2.18e-11 Eh. Not run on the branch: the three analytic-model validation files, `pytest -m sharc`, the full suite in one run.

The second of three sub-projects that close O7 (triplets and spin-orbit coupling from PySCF). Sub-project 1 (triplet states in the backend) is merged (`d698271`); sub-project 3 (the `dynamics` stage with triplets and SOC) follows.

## 0. Owner decisions (2026-11-27, multiple choice and approved sections)

1. **Operators:** both the spin-orbit mean-field operator (SOMF, with the two-electron mean field computed exactly) and the one-electron operator with effective nuclear charges; SOMF is the default.
2. **Reach:** the backend (a complex spin-expanded Hamiltonian on request) and `single-point` (a table and the JSON); no stage changes; `dynamics` still refuses triplets.
3. **Validation (all asserted):** ORCA 6.1.1 SOMF, ORCA effective charges, internal physics checks, phase continuity along a path and on resume.
4. **Systems:** formaldehyde (H2CO) and thioformaldehyde (H2CS), for SA-CASSCF and TDA.
5. **Approach A:** method-specific transition-density formulas (SA-CASSCF from spin-resolved CI transition densities, TDA/TDDFT from closed-form contractions of the CIS-like X), with a brute-force Slater-Condon evaluator used only in the tests.
6. **Sections 1-3 of the design** were approved as presented.

## 1. Why, and what was measured during the design

The dynamics core already propagates with SOC in the diagonal representation on LVC models (P7a-2): it takes a complex Hermitian MCH Hamiltonian in the canonical spin-expanded order of `dynamics/spin.py` (S0 ... S(n-1), then T1 ... Tm with Ms = -1, then Ms = 0, then Ms = +1). Sub-project 1 gave the backend triplet states without SOC. This sub-project computes the SOC matrix elements between them.

### 1.1 Measured (throwaway probes, PySCF 2.14.0 and ORCA 6.1.1, seen before the rules were fixed)

- PySCF: `mol.intor("int1e_pnucxp", comp=3)` is antisymmetric to 1.4e-14 (H2CO/6-31G, 22 basis functions); `scf.jk.get_jk(mol, [dm, dm], ["ijkl,kl->ij", "ijkl,jk->il"], intor="int2e_p1vxp1", comp=3)` contracts the two-electron spin-orbit integrals with a density directly (0.2 s); `fci.addons` has `des_a`, `des_b`, `cre_a`, `cre_b`.
- ORCA: TDA (CIS) with `%tddft ... triplets true dosoc true end` prints a block `CALCULATED SOCME BETWEEN TRIPLETS AND SINGLETS` with `<T|HSO|S>` as (Re, Im) in cm-1 for the Cartesian components Z, X, Y, to 0.01 cm-1 (H2CO CIS/6-31G at a test geometry: T1-S0 63.20 cm-1, T1-S2 49.06, T2-S1 40.26, T3-S0 60.00). Its default SOMF is `SOCType 3` with the flags 1-electron on, Coulomb exact (4), **exchange AMFI-type (3)**, DFT correlation off. `SOCFlags 1,4,4,0` gives exchange "Exact integrals". `SOCType 1` gives "effective nuclear charge". The SA-CASSCF run with `rel dosoc true end` uses SOMF and QDPT; its matrix-element print format is to be established in Task 1.

## 2. Input and output

### 2.1 Input

- New key `[electronic] soc_operator = "somf" | "effective_charge"`, default `"somf"`.
- `factory.electronic_fingerprint` leaves the key out when it has its default (the `hop_energy` precedent), so adding the field does not change the fingerprint of any existing project. No stage reads the key yet.

### 2.2 Backend

- `Request` gains `soc: bool = False`. With `soc = True` and triplets present, `ElectronicResult` gains `hamiltonian` (complex, N_exp x N_exp, Hermitian, Eh; N_exp = n + 3m, the canonical order above; the spin-free energies on the diagonal, each triplet energy repeated for its three components; S-T and T-T SOC elements off the diagonal; S-S elements exactly zero) and `hamiltonian_labels` (the `dynamics.spin` labels `S0`, ..., `T1(-1)`, ...).
- **Phases:** every element carries the tracked phases p_I p_J of its two spin-free states (the NAC rule), so the matrix is smooth along a path; the three components of a triplet share its phase.
- `soc = True` without triplets is a `ValueError` (like NACs where they are not available). `soc = False` (the default) computes no spin-orbit integral.
- The saved electronic state is unchanged (SOC is recomputed at every geometry, not stored).

### 2.3 `single-point`

With `triplets > 0` it requests the SOC and prints a table of the S-T and T-T matrix elements in cm-1 (one row per spin-free state pair: the Ms = -1, 0, +1 elements as complex numbers and the root sum of squares over Ms); `--json` gains `hamiltonian` (real and imaginary parts), `hamiltonian_labels` and `soc_operator`. Without triplets the output is unchanged.

### 2.4 Unchanged

Every stage, every `triplets = 0` path, the saved electronic state; no stage version or fingerprint changes.

## 3. The methods

### 3.1 The operators (`electronic/soc.py`, new; PySCF imported inside its functions)

- **SOMF** (Hess, Marian, Wahlgren, Gropen 1996; Neese 2005): the effective one-electron spatial operator (three Cartesian components, Eh), the one-electron Breit-Pauli term (`int1e_pnucxp`) plus the two-electron mean field of a reference density D, the Coulomb term minus the two exchange terms with the factor 3/2, both contracted directly from `int2e_p1vxp1` (`scf.jk.get_jk`), times alpha^2/2. D is the SCF density for TDA/TDDFT and the state-averaged density for SA-CASSCF. Task 1 records which density ORCA uses for its CASSCF SOMF; if it is another one, the CASSCF comparison names the difference and our choice is not changed to match.
- **Effective charges:** the one-electron term only, each nuclear charge replaced by its effective charge of Koseki, Schmidt and Gordon (1992, 1995) and Koseki, Gordon, Schmidt (1998) for H-Ar, built per atom from `int1e_prinvxp` (origin at the nucleus) scaled by Z_eff and alpha^2/2. An element outside the table is a `ConfigError`. Task 1 checks that ORCA's effective charges for H, C, O, S equal these literature values and records the result; the bound of section 4 applies either way.

### 3.2 The matrix elements

The SOC operator is a rank-1 spin tensor: one reduced matrix element is computed per spin-free state pair and every Ms component follows from the Wigner-Eckart theorem with Condon-Shortley phases. Singlet-singlet elements are zero by construction.

- **SA-CASSCF:** S-T from the spin-flip transition density <S| b+_p a_q |T, Ms = +1> over the active orbitals (from the CI vectors with `fci.addons.des_a`/`cre_b`); T-T from the spin-conserving (alpha minus beta) transition density between two Ms = +1 triplets; the doubly occupied core enters only through the SOMF density (its one-body part cancels between the spin states).
- **TDA/TDDFT:** the states as CIS-like expansions in X only (S0 = Phi0, Sm the singlet and Tn the triplet combinations, as for the overlaps); S0-T = sum X_ia h_ia; Sm-Tn and Tn-Tn' the occupied-occupied and virtual-virtual contractions, derived by the Slater-Condon rules. For TDDFT, X renormalised and Y dropped (the same approximation as the overlaps; documented; only TDA is compared with ORCA).

### 3.3 Assembly and errors

`PySCFBackend` assembles the matrix: the diagonal from the spin-free energies, the off-diagonal elements in the canonical order times p_I p_J. It is Hermitian by construction; a deviation above 1e-12 Eh is an internal error (no silent symmetrisation). The integral and SOMF steps go through `pyscf_errors.wrap_failure` like every other PySCF step.

### 3.4 Test-only reference

`tests/electronic/soc_reference.py`: a brute-force Slater-Condon evaluator of a spin-orbital one-body operator between determinant expansions, sharing no code with the production formulas.

## 4. Validation (asserted; bounds fixed here, before any run)

**Systems:** H2CO and H2CS, 6-31G, at geometries from our own RHF/6-31G\* optimisation, committed as `tests/data/formaldehyde.xyz` and `tests/data/thioformaldehyde.xyz`; ORCA at the same coordinates. SA-CASSCF CAS(4,3) (n, pi, pi\*; PySCF's default active space), 2 singlets + 2 triplets, equal weights; TDA: CIS (TDA-HF), 3 singlets + 3 triplets.

1. **ORCA 6.1.1, SOMF** (`SOCType 3`, `SOCFlags 1,4,4,0`, no RI; marker `orca`, runs only with `ORCA_EXE`): for every S-T pair the root sum of squares over its components (invariant under any choice of triplet basis). **Rule: |ours - ORCA| <= max(0.05 cm-1, 0.2 % of ORCA).** Both molecules, both methods; T-T pairs the same where ORCA prints them (Task 1).
2. **ORCA, effective charges** (`SOCType 1`): the same comparison and rule.
3. **Internal** (fast tier, PySCF): the Hamiltonian is Hermitian within 1e-12 Eh and S-S elements are exactly 0; the Ms components follow the Wigner-Eckart ratios within a relative 1e-10; the production elements equal the brute-force Slater-Condon reference within 1e-10 Eh for both methods (H2CO/STO-3G); the root sums of squares are invariant under a rigid rotation and translation within a relative 1e-7; SOMF with the two-electron part off equals the one-electron Breit-Pauli term within 1e-12, and so does the effective-charge operator with Z_eff = Z.
4. **Phase continuity:** 5 points stretching C=O by 0.005 bohr per step, both methods. **Rule:** no element larger than 1 cm-1 changes sign between steps, and each element changes by at most 10 % + 0.05 cm-1 per step. **Resume:** a save/load chain gives the same phased Hamiltonian as the uninterrupted chain within 1e-10 Eh.
5. **Unchanged:** `soc = False` and `triplets = 0` call no spin-orbit integral (a monkeypatch guard); the existing tests stay green.

All new tests follow the project rule: no bit-for-bit BLAS comparison and no reliance on an eigenvector's sign; checked once with `OPENBLAS_CORETYPE=SandyBridge`.

## 5. Order of work (one commit per task)

1. ORCA facts (a measurement): the SOC blocks ORCA prints for CASSCF and TDA (incl. T-T), the density of its CASSCF SOMF, its Z_eff values, the formats; the two geometries optimised.
2. `electronic/soc.py`: the SOMF and effective-charge operators, with their tests.
3. The SA-CASSCF elements, with the brute-force reference test.
4. The TDA/TDDFT elements, with the reference test.
5. Backend assembly: `Request.soc`, the phases, the `soc_operator` key and its fingerprint rule, `single-point`, the continuity and resume tests.
6. The ORCA validation tests.
7. The controller's test tiers (fast tier, the electronic tests, the PySCF tier) on the branch head.
8. Documents: USER_MANUAL, CLAUDE.md, CHANGELOG, a validation report.

## 6. Not in scope

Dynamics with SOC (sub-project 3); SOC inside the SCF or CASSCF optimisation; scalar-relativistic corrections; SOC for TDDFT beyond the X-only approximation; effective charges beyond Ar.

## 7. References (to be verified with Consensus before the documents cite them)

- B. A. Hess, C. M. Marian, U. Wahlgren, O. Gropen, "A mean-field spin-orbit method applicable to correlated wavefunctions", Chem. Phys. Lett. 251, 365 (1996).
- F. Neese, "Efficient and accurate approximations to the molecular spin-orbit coupling operator and their use in molecular g-tensor calculations", J. Chem. Phys. 122, 034107 (2005).
- S. Koseki, M. W. Schmidt, M. S. Gordon, J. Phys. Chem. 96, 10768 (1992); J. Phys. Chem. 99, 12764 (1995); S. Koseki, M. S. Gordon, M. W. Schmidt, N. Matsunaga, J. Phys. Chem. A 102, 10430 (1998) (effective nuclear charges).
