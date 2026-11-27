# Spin-orbit coupling matrix elements from PySCF (triplets and SOC, sub-project 2): report

Status: **the PySCF backend computes the spin-orbit coupling (SOC) matrix elements between its singlet and triplet states, for SA-CASSCF and TDA/TDDFT, with two operators (the spin-orbit mean field, SOMF, the default; and the one-electron operator with effective nuclear charges), and returns them as the complex spin-expanded Hamiltonian that the dynamics core uses; `single-point` prints them.** Every asserted check passes within its spec-fixed bound:

- ORCA 6.1.1 on formaldehyde (H2CO) and thioformaldehyde (H2CS), SA-CASSCF(4,3)/6-31G 2S+2T and CIS/6-31G 3S+3T, both operators (rule max(0.05 cm-1, 0.2 % of ORCA) per pair): largest ratio |diff|/allowance 0.055 (CIS), 0.004 (CASSCF);
- the brute-force Slater-Condon reference: largest difference 9.0e-16 Eh (bound 1e-10);
- Hermiticity of the assembled matrix: 2.71e-20 Eh (bound 1e-12);
- rotation and translation invariance: relative 6.15e-11 (bound 1e-7);
- continuity along a C=O stretch: largest per-step change 0.0155 of the allowance, no sign change;
- the resumed chain: the spin-orbit elements within 2.18e-11 Eh (bound 1e-10).

**One bound was narrowed after numbers were seen, by owner decision (2026-11-27):** the resume bound 1e-10 Eh applies to the off-diagonal (spin-orbit) elements only, because after a resume the TDA spin-free energies on the diagonal differed by 1.3e-8 Eh in most runs at the default SCF criterion (section 11, item 1). **Two things differ from the owner-approved design** (section 9): the effective-charge table holds only H, C, O and S, the values ORCA 6.1.1 uses (measured), not checked against the tables of Koseki et al.; and the overall sign of i is fixed from the operator definition, not from ORCA's printed sign, which follows ORCA's arbitrary CI phase. **Not covered:** signs and phases are not compared with ORCA (only root sums of squares); the CIS triplet-triplet elements have no external reference (ORCA's TDA prints none); the comparison cannot tell apart the choice of the SA-CASSCF SOMF density; there is no SHARC comparison. **Found on the way:** H2CS CIS 3S+3T/6-31G does not converge at the default `td_conv_tol` (1e-8), because PySCF's Davidson solver stops on linear dependency; the ORCA test uses 1e-7 (section 11, item 2). No stage changed; the `dynamics` stage still refuses `triplets > 0` (sub-project 3).

Date: 2026-11-27. Branch `feat/soc-matrix-elements` (from `main` `0e269c9`). The spec is `docs/superpowers/specs/2026-11-27-soc-matrix-elements-design.md` (owner-approved 2026-11-27; spec commit `31e50a2`; erratum at its top). The plan is `docs/superpowers/plans/2026-11-27-soc-matrix-elements.md` (plan commit `bcce9a5`; its "Plan rulings" PR1-PR5 and its "Conventions" are binding where they differ from the spec). The tasks are these commits:

- `55aa533`: Task 1, the two geometries (`tests/data/formaldehyde.xyz`, `tests/data/thioformaldehyde.xyz`); the ORCA facts are a measurement with no other diff;
- `48aa281` and its fix `16dd555`: Task 2, the operators (`electronic/soc.py`; the fix pins the sign and the x/y/z order in a test and refuses ghost atoms);
- `a20c6fa` and its fix `1c80436`: Task 3, the SA-CASSCF elements and the brute-force reference (the fix checks the reference singlets and a non-vacuous triplet-triplet comparison);
- `0cccfb8` and its fix `777e2c3`: Task 4, the TDA/TDDFT elements (the fix shares the test helpers and checks the amplitudes' shape and norm);
- `1083b40` and its fix `cb96020`: Task 5, the backend, the input key and `single-point` (the fix refuses effective-charge elements before any SCF and fixes one rss definition);
- `ded6a2f` and its fix `1c2fcdd`: Task 6, the ORCA validation (the fix refuses unreadable ORCA lines and reports ORCA's stderr).

Task 7 (the test tiers) is the controller's measurement, with no diff; Task 8 is this report and the documents.

**Sources of every number:**

- the task reports in the git-ignored ledger `.superpowers/sdd/2026-11-27-soc-matrix-elements/` (`task-1-report.md` to `task-6-report.md`, `progress.md`, the Task 1 inputs and excerpts in `task-1-files/`);
- the throwaway scripts and ORCA runs named there (in WSL `~/soc_task1/`, `~/soc_task2/`; the Task 5 probes `probe2.py` to `probe5.py` in the scratchpad; the Task 6 reviewer's probes). None is in the repository.

**Labels:**

- *Asserted*: a pytest assertion.
- *Report*: measured, with no rule.
- *Inference*: not established by a measurement.

## 1. What was built

### 1.1 The operators (Task 2, `electronic/soc.py`)

Every operator is returned as `h` with shape (3, nao, nao), complex, `h[k] = 1j (alpha^2/2) A[k]` with `A[k]` real antisymmetric (k = x, y, z) and alpha = 1/c, so that `H_SO = sum_pq sum_k h[k]_pq S^k_pq`, `S^k_pq = sum a+_{p sigma} (s_k)_{sigma sigma'} a_{q sigma'}`, `s = sigma_Pauli/2`. `h` is Hermitian and purely imaginary.

- **One-electron Breit-Pauli:** `A = P`, `P = int1e_pnucxp` (the nuclear term with the true charges).
- **SOMF** (Hess, Marian, Wahlgren, Gropen 1996; Neese 2005): `A = P + J - 3/2 (K1 + K2)`, with `J`, `K1`, `K2` contracted directly from `int2e_p1vxp1` and a reference density D through `scf.jk.get_jk` (scripts `"ijkl,kl->ij"`, `"ijkl,il->kj"`, `"ijkl,jk->il"`). The derivation (the mean field of the Breit-Pauli two-electron term over a closed-shell determinant; the spin-other-orbit Coulomb term vanishes; each exchange term carries 3/2) is in the module docstring. D is the SCF density for TDA/TDDFT and the state-averaged density `mc.make_rdm1()` for SA-CASSCF (plan ruling PR4).
- **Effective charges:** `A = -sum_A Z_eff,A prinvxp(R_A)` (`int1e_prinvxp` with the origin at each nucleus). `KOSEKI_ZEFF = {"H": 1.0, "C": 3.6, "O": 5.6, "S": 13.6}`: the values ORCA 6.1.1 uses, measured in Task 1 (section 2.4). Any other element is a `ConfigError`: *no established effective nuclear charge for X (available: H, C, O, S); use soc_operator = 'somf'*; ghost atoms are refused too.
- `soc_operator(mol, dm, kind)` dispatches on `"somf"` / `"effective_charge"` (`SOC_OPERATORS`); a PySCF failure in the integrals or the mean field is an `ElectronicStructureError` through `wrap_failure`. PySCF is imported inside the functions only; `import gqteashmd.cli` loads no pyscf, scipy, h5py or geometric (checked in Tasks 2-5).

### 1.2 The matrix elements (Tasks 3 and 4)

The SOC operator is a rank-1 spin tensor. With `h_plus = h_x + i h_y` and `h_minus = h_x - i h_y`, `H_SO = sum_pq [h_z S^z_pq + h_plus S^-_pq/2 + h_minus S^+_pq/2]`. The triplet components follow Condon-Shortley phases, `|T, M-1> = S_- |T, M> / sqrt 2`. Every element reduces to two densities between the Ms = +1 triplet components:

- singlet-triplet, `D_pq = <S| a+_{p beta} a_{q alpha} |T, +1>`: `<S|H|T, +1> = (1/2) sum h_plus D`, `<S|H|T, 0> = -(1/sqrt 2) sum h_z D`, `<S|H|T, -1> = -(1/2) sum h_minus D`;
- triplet-triplet, `Z_pq = <T, +1| S^z_pq |T', +1>`: with `z = sum h_z Z`, `p = sum h_plus Z`, `m = sum h_minus Z`, the 3 x 3 block (M, M' in the order -1, 0, +1) is `[[-z, p/sqrt2, 0], [m/sqrt2, 0, p/sqrt2], [0, m/sqrt2, z]]`.

`soc.SocBlocks(st, tt)` holds `st` (n_singlets, n_triplets, 3) and `tt` (n_triplets, n_triplets, 3, 3); there is no singlet-singlet part. The intra-triplet blocks `tt[J, J]` are computed like the others (plan ruling PR5).

- **SA-CASSCF** (`casscf_soc_blocks(mc, states, h_ao)`): D and Z from the CI vectors with `fci.addons.des_a`/`cre_b`/`cre_a`/`des_b`, over the active orbitals (`h` transformed with the active MOs). The core enters only through the SOMF density. The triplet CI vectors are taken by multiplicity from `states`.
- **TDA/TDDFT** (`response_soc_blocks(mf, td, td_triplet, states, h_ao)`): the states as CIS-like expansions in X only, `S0 = Phi0`, `S_m = sum x_ia (|ia, alpha> + |ia, beta>)`, `|T_n, 0> = sum x_ia (|ia, alpha> - |ia, beta>)`, with `x = X / sqrt(2 sum X^2)` (Y dropped, the overlaps' approximation). The densities are closed-form contractions: S0-T `D[i,a] = -sqrt2 x^T[i,a]`; S_m-T `D[b,a] = -sqrt2 (x^S^T x^T)[b,a]`, `D[i,j] = +sqrt2 (x^T x^S^T)[i,j]`; T-T' `Z[b,a] = (x^T^T x^T')[b,a]`, `Z[i,j] = (x^T' x^T^T)[i,j]`. So `<S0|H|T_n, 0> = sum_ia h_z,ia x_ia`, spec 3.2's "S0-T = sum X_ia h_ia" with the CIS normalisation. A wrong amplitude shape or a zero norm is an `ElectronicStructureError`; an excited singlet without its response object is a `ValueError`.
- The components are built by two helpers shared by both methods (`_st_components`, `_tt_block`), so one convention holds for both.

### 1.3 The assembly, the backend and the input (Task 5)

- **`soc.assemble_hamiltonian(energies_s, energies_t, blocks, phases_s, phases_t)`** (controller ruling: the plan's Conventions define it, no task's file list named it): the N_exp x N_exp complex matrix, N_exp = n + 3m, in the canonical order of `dynamics/spin.py` (S0 ... S(n-1), then T1 ... Tm with Ms = -1, then Ms = 0, then Ms = +1; index of T_J with Ms = M: `n + (M + 1) m + J`, 0-based J), labels from `dynamics.spin.expanded_states` (`hamiltonian_labels`): `p_I p_J st` in the S-T blocks, their conjugates in the T-S blocks, `p_J p_J' tt` in the T-T blocks; the S-S block exactly zero. It raises an `AssertionError` (an internal error, not a silent symmetrisation) if the T-T block deviates from Hermitian by more than 1e-12 Eh or carries a spin-orbit element on its diagonal above 1e-12 Eh, then sets the diagonal to the spin-free energies (each triplet at its three components). The three components of a triplet share its tracked phase.
- **`Request.soc: bool = False`**; with it `ElectronicResult` gains `hamiltonian` and `hamiltonian_labels`, and `to_json` writes the matrix as `hamiltonian_hartree` (`{"real", "imag"}`; the key was `hamiltonian` until the final fix wave, renamed so that it names its unit) and `hamiltonian_labels`, only when present. `Capabilities.soc` (default false; final fix wave) is true for `PySCFBackend` with `triplets > 0`; the test backend `ModelBackend` refuses `Request(soc=True)` with a `ValueError`. `soc = True` with `triplets = 0` is a `ValueError` (*Spin-orbit couplings need triplet states ...*). With `soc_operator = "effective_charge"` an element outside the table is refused in `_validate`, before any SCF (`soc.require_effective_charges`). `soc = False` (the default) and `triplets = 0` call no spin-orbit integral (monkeypatch guard, asserted for both methods). The saved electronic state is unchanged (SOC is recomputed at every geometry).
- **`[electronic] soc_operator = "somf" | "effective_charge"`**, default `"somf"`. A pydantic `model_serializer` leaves the key out of every dump when it has its default, so `.gqteashmd/config.json` and every fingerprint of an existing project are byte-identical (the Task 5 review checked the default fingerprints and dumps against an extracted `main` tree at `0e269c9`); `electronic_fingerprint` also pops it at its default; `ground_state_fingerprint` always drops it.
- **`single-point`** with `triplets > 0` requests the SOC, prints a table in cm-1 and writes `hamiltonian_hartree`, `hamiltonian_labels` and `soc_operator` to its JSON; without triplets the output is unchanged. One row per S_I-T_J pair (Ms = -1, 0, +1) and, under a second column header (final fix wave), one per T_J-T_J' pair with J <= J' (the elements (M, M') = (+1, 0), (+1, +1), (0, +1)), and the root sum of squares (rss) of the pair, `soc.pair_rss` (the one definition, also used by the ORCA test; final fix wave), which counts each distinct matrix element once: S_I-T_J its 3 elements; T_J-T_J' (J < J') its 9; T_J-T_J the 3 with M < M'. A value that rounds to zero prints as `0.00`, never `-0.00`. The table of the H2CO CASSCF example (default convergence, run for this report and re-run in the final fix wave):

```text
Spin-orbit coupling (cm-1, somf)
S-T rows: <S|H|T,Ms> for Ms = -1, 0, +1; T-T rows: <T,M|H|T',M'> for (M, M') = (+1, 0), (+1, +1), (0, +1); rss counts each distinct element of the pair once
pair                    -1                 0                +1       rss
S0-T1           0.00+0.00j       0.00+65.11j        0.00+0.00j     65.11
S0-T2           0.00+0.00j        0.00+0.00j        0.00+0.00j      0.00
S1-T1           0.00+0.00j        0.00+0.00j        0.00+0.00j      0.00
S1-T2           0.00+0.00j       0.00+57.69j        0.00+0.00j     57.69
pair                (+1,0)           (+1,+1)            (0,+1)       rss
T1-T1           0.00+0.00j        0.00+0.00j        0.00+0.00j      0.00
T1-T2           0.00+0.00j       0.00+55.94j        0.00+0.00j     79.11
T2-T2           0.00+0.00j        0.00+0.00j        0.00+0.00j      0.00
```

  (T1-T2: the (+1, 0) and (0, +1) elements are 0 by symmetry, the (+1, +1) element is +55.94i; its rss over the 9 elements is 55.94 sqrt 2 = 79.11. The table is the final fix wave's re-run, with the T-T column header; the first run printed -57.69i and -55.94i for S1-T2 and T1-T2: the signs follow the arbitrary CI signs, the magnitudes do not change.) A defect found by running the example and fixed in Task 5: the first version showed the energy in the middle column of the T_J-T_J rows.

## 2. Task 1: ORCA facts (report)

ORCA 6.1.1 and PySCF 2.14.0, in `~/soc_task1/` (WSL).

### 2.1 The geometries (plan ruling PR1)

H2CO and H2CS optimised at RHF/6-31G\* with `gqteashmd.geomopt.optimize_geometry` (`GAU_TIGHT`), committed in Angstrom with 10 decimals: H2CO E -113.8656852629 Eh, max |g| 1.006e-7 Eh/bohr (8 gradients); H2CS E -436.5050306150 Eh, max |g| 1.240e-6 (7). Planar in the yz plane, C=O / C=S along z; C2v only to about 4e-7 Angstrom (the optimiser's residual), so no test assumes exact symmetry zeros. ORCA's RHF/6-31G at the same coordinates agrees with PySCF to 1.9e-10 / 3.5e-10 Eh, and ORCA did not reorient the molecule. The Task 1 review checked H2CO C=O 1.1844 A, C-H 1.0916, HCH 115.7; H2CS C=S 1.5971, C-H 1.078, HCH 115.6.

### 2.2 The TDA (CIS) SOC block

`%tddft nroots 3 tda true triplets true dosoc true end` with `%rel SOCType 3 SOCFlags 1,4,4,0 end` (SOMF, one-electron, Coulomb and exchange with exact integrals) or `SOCType 1` (effective charges) prints `CALCULATED SOCME BETWEEN TRIPLETS AND SINGLETS`: one line per (T, S) pair, zeros included, `<T|HSO|S>` as (Re, Im) pairs in the Cartesian column order **Z, X, Y**, cm-1 to 0.01; S 0 is the SCF ground state. All elements are purely imaginary. **No triplet-triplet elements are printed** (whether ORCA's CIS-SOC matrix holds T-T terms internally is not established). The Cartesian reading of the columns is an inference from symmetry (H2CO T1-S0 appears in Z only, A2 x A1 = R_z); the rss over the three columns is the invariant compared.

**ORCA's TDA skips a root with `nroots 3` for H2CO:** its third singlet is 0.433511 Eh, while PySCF and ORCA with `nroots 6` find 0.391500 Eh as the third root. So Task 6 asks ORCA for 6 roots and maps the states by energy (controller ruling).

### 2.3 The SA-CASSCF SOC block

`%casscf ... mult 1,3 nroots 2,2 bweight 2,2 weights[0] = 1,1 weights[1] = 1,1 etol 1e-10 gtol 1e-6 rel dosoc true printlevel 3 end end`. ORCA re-orders the blocks (block 0 = triplets, block 1 = singlets); every state has weight 0.25. `printlevel 3` inside the `rel` sub-block prints `NONZERO SOC MATRIX ELEMENTS (cm**-1)`: one line per element `<bra|H|ket>` with (Block, Root, S, Ms), cm-1 to 0.001, **one triangle only, S-T and T-T** ("NONZERO" lets rounded zeros through, and which ones appear varies). The full `SOC MATRIX (A.U.)` is printed to 1e-6 Eh only, too coarse for the rule.

ORCA's SA-CASSCF energies agree with ours (PySCF SA-CASSCF(4,3)/6-31G 2S+2T, `state_average_mix_`, weights 0.25) within 2.8e-8 Eh for every state of both molecules: the same solutions. The ORCA elements (magnitudes, cm-1; the signs follow ORCA's CI phases and flip between runs):

| | H2CO SOMF | H2CO Z_eff | H2CS SOMF | H2CS Z_eff |
|---|---|---|---|---|
| S0-T1 (Ms 0) | 65.113 | 73.220 | 182.403 | 192.552 |
| S1-T2 (Ms 0) | 57.694 | 65.155 | 172.516 | 182.083 |
| T2-T1 (Ms +1 and -1) | 55.943 | 63.191 | 168.649 | 177.999 |

### 2.4 The density of ORCA's SA-CASSCF SOMF (plan ruling PR4)

A CASCI on the fixed equal-weight orbitals with different state weights (the CI vectors and energies do not change, only the averaged active density): the H2CO SOMF elements move by up to 0.06 cm-1 (about 0.1 %), antisymmetrically for antisymmetric density changes, i.e. linearly in the density; with `SOCType 1` (no density) they do not move. **ORCA's CASSCF SOMF uses the state-averaged density with the CASSCF weights**, the same choice as ours. That it is exactly the printed SA density is the simplest reading (*inference*, high confidence); the ORCA 6.1 manual does not state the choice. The effect (0.1 %) is below the 0.2 % rule, so the comparison could not tell density choices apart anyway.

### 2.5 The effective nuclear charges

ORCA prints no Z_eff values. For a single atom the effective-charge operator is Z_eff/Z times the one-electron Breit-Pauli operator, so the ratio of every `SOCType 1` element to the `SOCFlags 1,0,0,0` element is Z_eff/Z. Measured (atoms H 2S/2P, C 3P, O 3P, S 3P; the interval is the intersection of the 3-decimal rounding intervals):

| | Z | ORCA Z_eff | interval |
|---|---|---|---|
| H | 1 | 1.000 | [0.99974, 1.00026] |
| C | 6 | 3.600 | [3.59992, 3.60006] |
| O | 8 | 5.600 | [5.59993, 5.60006] |
| S | 16 | 13.600 | [13.59995, 13.60008] |

The ratio test also shows that ORCA's Z_eff operator has the prefactor of its one-electron term. **The literature values of Koseki et al. were not established:** the papers were located (section 12) but their tables could not be read, and no number was written from memory; a third-party program's table (pysisyphus) turned out to implement another set (Chiodo and Russo 2008: C 3.01, O 5.02, S 12.92) and was not used. So `KOSEKI_ZEFF` holds only these four values (controller ruling; spec erratum).

## 3. Task 2: the operator calibration (report; plan ruling PR2)

H2CO CIS/6-31G, S0-T1 rss in cm-1, ours against ORCA (`nroots 6`), one variant per ORCA operator part. Ours: PySCF RHF (E -113.80712844415 Eh, ORCA -113.80712844433), triplet TDA, T1 0.133196 Eh on both sides, element `sum_ia X_ia h_k[a,i]`.

| ORCA variant | ours | ORCA | ours - ORCA | bound max(0.05, 0.2 %) |
|---|---|---|---|---|
| `SOCType 3 SOCFlags 1,0,0,0` (one-electron) | 102.2500 | 102.25 | +0.00004 | 0.2045 |
| `SOCType 3 SOCFlags 1,4,0,0` (+ Coulomb) | 76.3618 | 76.36 | +0.0018 | 0.1527 |
| `SOCType 3 SOCFlags 1,4,4,0` (+ exact exchange) | 63.0769 | 63.08 | -0.0031 | 0.1262 |
| `SOCType 1` (effective charges) | 70.8098 | 70.81 | -0.0002 | 0.1416 |

Every difference is below ORCA's print rounding (0.005 cm-1). The Coulomb and the exchange parts, the prefactor alpha^2/2, the factor 3/2 and the relative sign of the mean field are thereby checked separately. The integral definitions behind the derivation were checked numerically (`int1e_pnucxp` against the antisymmetrised `int1e_ipnucip` to 1.1e-14; the explicit `einsum` contractions of `int2e_p1vxp1` against `get_jk` to 2.7e-15; `K1 = -K2^T` to 8.3e-16; report).

**The sign of i** (section 9, item 2): with T1's X phased so that its largest coefficient is positive, ours gives `<T1,z|H_SO|S0> = +102.25i` (one-electron) and `+63.08i` (SOMF); ORCA prints `(0.00, -102.25)` and `(0.00, -63.08)`. ORCA's printed sign cannot fix the sign of i, because flipping T1's CI vector flips it, and no phase-free cycle (S-T, T-T, T-S of the same states) exists for these systems. The sign is fixed from the operator definition, `h = +i (alpha^2/2) A` with the integrals as PySCF returns them, and a committed test pins it and the x, y, z order (fix `16dd555`).

## 4. The brute-force reference (asserted; Tasks 3 and 4)

`tests/electronic/soc_reference.py` (not collected; it imports nothing from `soc.py`) evaluates a spin-orbital one-body operator between explicit determinant expansions by the Slater-Condon rules, with explicit signs, reading PySCF's CI coefficients by a convention taken from PySCF 2.14's own conventions (the core as a leading block of pairs), building the Ms = 0, -1 components by applying `S_-/sqrt2` itself and the TDA states from their own excitations, each normalised by its own vector norm. Its self-checks (asserted): the triplet components orthonormal within 1e-12, `S_-|T, -1> = 0`, `S_-|S> = 0`, `S_+|T, +1> = 0`, `<T, 0|S_-|T, +1> = sqrt 2`, the Gram matrix of all response components equal to I within 1e-12.

Largest |production - reference| (Eh; bound 1e-10; the largest element in parentheses for the response methods):

| System | Operator | S-T | T-T |
|---|---|---|---|
| H2CO/STO-3G SA-CASSCF(4,3) 2S+2T | SOMF | 5.4e-20 | 8.1e-20 |
| same | random imaginary | 2.2e-16 | 3.3e-16 |
| same | random Hermitian | 8.9e-16 | 6.7e-16 |
| ethylene/STO-3G SA-CASSCF(2,2) 2S+1T (odd core) | SOMF | 1.6e-27 (elements 7.95e-12, zero by symmetry) | 2.6e-36 |
| same | random imaginary | 7.9e-17 | 7.4e-32 |
| same | random Hermitian | 7.0e-16 | 2.3e-16 |
| H2CO/STO-3G CIS 3S+3T | SOMF | S0-T 2.7e-20 (2.3e-4), Sm-T 2.7e-20 (1.4e-4) | 5.4e-20 (1.4e-4) |
| same | random imaginary | 2.2e-16 (1.3), 2.8e-16 (1.1) | 4.5e-16 (1.1) |
| same | random Hermitian | 7.0e-16 (2.5), 9.0e-16 (1.5) | 6.7e-16 (1.9) |
| H2CO/STO-3G TDDFT-B3LYP 2S+2T | SOMF | 5.4e-20 (2.3e-4), 2.7e-20 (1.4e-4) | 2.7e-20 (1.4e-4) |
| same | random imaginary | 4.4e-16 (0.77), 1.1e-16 (0.65) | 2.2e-16 (0.66) |
| same | random Hermitian | 4.6e-16 (2.5), 5.0e-16 (1.5) | 6.7e-16 (1.8) |

- **Non-vacuity** (asserted): the S0-T and excited S-T rows exceed 1e-6, and the T-T rows for the Hermitian operator. The ethylene SOMF S-T check is skipped (its elements vanish by symmetry, 7.95e-12; controller ruling). A throwaway mutation (the sign of the occupied-occupied S-T density flipped) gave a difference of 3.0 Eh (CIS) and 2.99 Eh (TDDFT): the comparison can fail (report).
- **Wigner-Eckart ratios** (asserted, relative 1e-10): one random complex matrix in a single Cartesian slot gives the explicit ratio tables `WE_ST` and `WE_TT` written out in the tests (controller ruling), on production and on the reference; the reference test passed before any production code existed. They caught a p/m transposition in the T-T block in Task 3's first green attempt.
- **TDDFT renormalisation** (asserted): `|2 sum X^2 - 1| > 1e-4` for TDDFT and `<= 1e-12` for TDA, so the rescaling is not vacuous. Measured `sum X^2` for TDDFT 0.5001-0.5216 (report). Both sides of the TDDFT comparison use the same `td`, so this checks the formulas, not the X-only approximation.
- The two reviews built the elements a third way (an FCI-vector construction with PySCF's `cre`/`des` and no Wigner-Eckart formulas): agreement 1.8e-14 (CASSCF) and 4.6e-16 (TDA); the CIS energies between the built states equal E_HF + omega within 4.1e-8 Eh, confirming the X -> determinant convention (report, review).

## 5. Task 5: the backend checks (asserted)

H2CO/6-31G, SA-CASSCF(4,3) 2S+2T and CIS 3S+3T; test settings fixed before any bound was measured: `scf_conv_tol` 1e-12, CASSCF `conv_tol` 1e-12 and `conv_tol_grad` 1e-7, `td_conv_tol` 1e-10 (a probe showed that `conv_tol_grad` 1e-8 and 1e-9 do not converge for this CASSCF).

| Check | Bound | CASSCF | CIS |
|---|---|---|---|
| Hermiticity of the assembled H (largest deviation) | 1e-12 Eh | 3.96e-22 | 2.71e-20 |
| largest \|T-T diagonal spin-orbit element\| | 1e-12 Eh | 5.3e-35 | 5.8e-34 |
| S-S block | exactly 0 | 0 | 0 |
| diagonal = spin-free energies | exact | exact | exact |
| invariance under a 30-degree rotation about (1, 2, 3) and a translation (0.3, -0.2, 0.5) bohr, per pair relative to its own rss (floor 1e-3 of the largest pair) | 1e-7 | 7.56e-14, 1.43e-13, 3.47e-13, 4.0e-14 (four runs) | 3.58e-11, 6.15e-11, 2.92e-11, 1.1e-11 |
| continuity, 5 points, C=O stretched 0.005 bohr per step: largest per-step change / (10 % + 0.05 cm-1) | <= 1 | 0.0144 | 0.0155 |
| continuity: sign change of an element > 1 cm-1 (Re(a conj b) <= 0) | none | none (smallest kept 55.94 cm-1) | none (22.28 cm-1) |
| resume (one chain: A computes points 0-4 and saves after point 2; C loads it and computes 3-4), off-diagonal elements | 1e-10 Eh | about 2e-18 to 3e-18 | 2.174e-11 to 2.18e-11 (6 of 8 runs; about 1e-16 in the other 2) |
| resume, diagonal (spin-free energies) | **reported, not asserted** (owner decision) | at most 3.1e-13 | 1.330e-8 to 1.336e-8 in 6 of 8 runs (9 of 14 before the decision); about 1e-13 otherwise |

The off-diagonal elements are above 10 cm-1, so the checks are not vacuous. The phases differ between A and C (the raw Davidson signs), but the phased elements agree, so the phase handling is correct.

**The TDA resume finding** (measured, `probe4.py`): A and C start the SCF at point 3 from the same saved density but do not always take the same number of SCF cycles (one repetition: 9 cycles, orbital gradient 2.04e-9, against 8 cycles, 3.36e-8). The SCF stops on `scf_conv_tol = 1e-12`, whose gradient criterion is 1e-6, and the run sits so close to the stopping test that the cycle count depends on round-off (the SCF energies differed by 8.5e-14). One cycle less leaves the orbital energies 1.9e-8 Eh away; the excitation energies are not variational in the orbitals and move by 1.3e-8 Eh. When both took 9 cycles the difference was 5.7e-14. The spin-orbit elements stay within 2.2e-11. Sub-project 1's resume test compared overlaps, not energies, which is why it did not see this.

The pair magnitudes of the CASSCF test system (cm-1): S0-T1 65.1127, S1-T2 57.6944, T1-T2 79.1147, the others zero by symmetry; of the CIS system: S0-T1 63.0769, S0-T3 60.7987, S1-T2 40.0313, S1-T3 49.1292, S2-T1 49.3403, S2-T2 31.6311, T1-T2 58.0713, T1-T3 69.611, T2-T3 46.6748 (report).

Also asserted in Task 5: `soc = False` and `triplets = 0` call no spin-orbit integral (both methods, triplets 0 and 2); `soc = True` without triplets is refused; the SOMF density passed to the operator is `mc.make_rdm1()` for SA-CASSCF (a spy; PR4); the effective-charge element check fires before any SCF (H2CN+, both methods); the default `soc_operator` is not dumped and round-trips; the fingerprints with the default are unchanged; the `single-point` table, its rss definition (a synthetic H with every intra-triplet element nonzero) and its JSON.

## 6. Task 6: ORCA 6.1.1 (asserted; `tests/validation/test_orca_soc.py`; the comparison test has the markers `orca` and `pyscf` and is skipped without `ORCA_EXE`, the parser tests have none since the final fix wave)

The ORCA inputs are Task 1's, with the coordinates of `tests/data/*.xyz` verbatim and TDA `nroots 6`. Ours: `PySCFBackend.compute(Request(coords, soc=True))`; CASSCF `ncas 3`, `nelecas 4`, `conv_tol 1e-11`, `conv_tol_grad 1e-6` (matching ORCA's `etol 1e-10`, `gtol 1e-6`), `scf_conv_tol 1e-12`; CIS `functional "hf"`, `scf_conv_tol 1e-12`, **`td_conv_tol 1e-7`** (section 11, item 2).

- **Energies first** (asserted, 1e-6 Eh): the absolute S0 energies, then every excited state mapped to the ORCA state of the same multiplicity with the nearest excitation energy, one to one, or the test fails with "different solutions". Largest difference 2.05e-7 Eh (CIS, ORCA's cm-1 column has a resolution of 2.3e-7 Eh); CASSCF 1.39e-8 to 4.93e-8 Eh; the SCF energies 1.83e-10 and 3.44e-10 Eh. Every map chose ORCA's states in index order.
- **The quantity** is the rss per pair: ours by the rss definition of section 1.3; ORCA TDA the three Cartesian columns; ORCA CASSCF every printed NONZERO line of the pair, keyed by the unordered pair of (block, root, Ms) states, so an element and its conjugate count once (a missing element is below ORCA's print threshold; a T_J-T_J pair has no line, so ORCA's value is 0). Every printed line must belong to a compared pair, and a duplicate or unreadable line is refused.
- **The rule:** `|ours - ORCA| <= max(0.05 cm-1, 0.002 x ORCA)` per pair; zero-by-symmetry pairs under the same rule (no exact zero assumed); the largest pair must exceed 10 cm-1.
- **Not compared:** signs and phases; the CASSCF SOMF density (section 2.4).

**Counts.** 8 combinations (2 molecules x 2 methods x 2 operators), 64 pairs: **32 S-T pairs and 4 T1-T2 pairs (CASSCF) carry a real coupling; 20 S-T pairs are zero on both sides (by symmetry); the 8 T_J-T_J pairs are zero.** Every pair passes. The CIS T-T pairs are not compared (ORCA prints none).

Every pair (cm-1; allowance max(0.05, 0.002 x ORCA); ratio |diff|/allowance; "0.0000" = below 5e-5 on our side and 0.00 or absent on ORCA's):

| molecule | method | operator | pair | ours | ORCA | \|diff\| | allowance | ratio |
|---|---|---|---|---|---|---|---|---|
| H2CO | CASSCF | somf | S0-T1 | 65.1127 | 65.1130 | 0.0003 | 0.1302 | 0.002 |
| H2CO | CASSCF | somf | S0-T2 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CO | CASSCF | somf | S1-T1 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CO | CASSCF | somf | S1-T2 | 57.6944 | 57.6940 | 0.0004 | 0.1154 | 0.003 |
| H2CO | CASSCF | somf | T1-T1 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CO | CASSCF | somf | T1-T2 | 79.1147 | 79.1153 | 0.0007 | 0.1582 | 0.004 |
| H2CO | CASSCF | somf | T2-T2 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CO | CASSCF | effective_charge | S0-T1 | 73.2196 | 73.2200 | 0.0004 | 0.1464 | 0.003 |
| H2CO | CASSCF | effective_charge | S0-T2 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CO | CASSCF | effective_charge | S1-T1 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CO | CASSCF | effective_charge | S1-T2 | 65.1549 | 65.1550 | 0.0001 | 0.1303 | 0.001 |
| H2CO | CASSCF | effective_charge | T1-T1 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CO | CASSCF | effective_charge | T1-T2 | 89.3662 | 89.3656 | 0.0006 | 0.1787 | 0.004 |
| H2CO | CASSCF | effective_charge | T2-T2 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CO | CIS | somf | S0-T1 | 63.0769 | 63.0800 | 0.0031 | 0.1262 | 0.025 |
| H2CO | CIS | somf | S0-T2 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CO | CIS | somf | S0-T3 | 60.7987 | 60.8000 | 0.0013 | 0.1216 | 0.011 |
| H2CO | CIS | somf | S1-T1 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CO | CIS | somf | S1-T2 | 40.0313 | 40.0300 | 0.0013 | 0.0801 | 0.017 |
| H2CO | CIS | somf | S1-T3 | 49.1292 | 49.1300 | 0.0008 | 0.0983 | 0.008 |
| H2CO | CIS | somf | S2-T1 | 49.3403 | 49.3400 | 0.0003 | 0.0987 | 0.003 |
| H2CO | CIS | somf | S2-T2 | 31.6311 | 31.6300 | 0.0011 | 0.0633 | 0.017 |
| H2CO | CIS | somf | S2-T3 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CO | CIS | effective_charge | S0-T1 | 70.8098 | 70.8100 | 0.0002 | 0.1416 | 0.001 |
| H2CO | CIS | effective_charge | S0-T2 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CO | CIS | effective_charge | S0-T3 | 68.9962 | 69.0000 | 0.0038 | 0.1380 | 0.028 |
| H2CO | CIS | effective_charge | S1-T1 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CO | CIS | effective_charge | S1-T2 | 45.0411 | 45.0400 | 0.0011 | 0.0901 | 0.012 |
| H2CO | CIS | effective_charge | S1-T3 | 54.1838 | 54.1800 | 0.0038 | 0.1084 | 0.035 |
| H2CO | CIS | effective_charge | S2-T1 | 54.4004 | 54.4000 | 0.0004 | 0.1088 | 0.004 |
| H2CO | CIS | effective_charge | S2-T2 | 36.2977 | 36.3000 | 0.0023 | 0.0726 | 0.032 |
| H2CO | CIS | effective_charge | S2-T3 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CS | CASSCF | somf | S0-T1 | 182.4033 | 182.4030 | 0.0003 | 0.3648 | 0.001 |
| H2CS | CASSCF | somf | S0-T2 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CS | CASSCF | somf | S1-T1 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CS | CASSCF | somf | S1-T2 | 172.5162 | 172.5160 | 0.0002 | 0.3450 | 0.001 |
| H2CS | CASSCF | somf | T1-T1 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CS | CASSCF | somf | T1-T2 | 238.5050 | 238.5057 | 0.0007 | 0.4770 | 0.001 |
| H2CS | CASSCF | somf | T2-T2 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CS | CASSCF | effective_charge | S0-T1 | 192.5518 | 192.5520 | 0.0002 | 0.3851 | 0.001 |
| H2CS | CASSCF | effective_charge | S0-T2 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CS | CASSCF | effective_charge | S1-T1 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CS | CASSCF | effective_charge | S1-T2 | 182.0832 | 182.0830 | 0.0002 | 0.3642 | 0.001 |
| H2CS | CASSCF | effective_charge | T1-T1 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CS | CASSCF | effective_charge | T1-T2 | 251.7291 | 251.7286 | 0.0005 | 0.5035 | 0.001 |
| H2CS | CASSCF | effective_charge | T2-T2 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CS | CIS | somf | S0-T1 | 177.1138 | 177.1100 | 0.0038 | 0.3542 | 0.011 |
| H2CS | CIS | somf | S0-T2 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CS | CIS | somf | S0-T3 | 137.9228 | 137.9200 | 0.0028 | 0.2758 | 0.010 |
| H2CS | CIS | somf | S1-T1 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CS | CIS | somf | S1-T2 | 126.3939 | 126.3800 | 0.0139 | 0.2528 | 0.055 |
| H2CS | CIS | somf | S1-T3 | 135.0621 | 135.0600 | 0.0021 | 0.2701 | 0.008 |
| H2CS | CIS | somf | S2-T1 | 134.5979 | 134.5900 | 0.0079 | 0.2692 | 0.030 |
| H2CS | CIS | somf | S2-T2 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CS | CIS | somf | S2-T3 | 126.9117 | 126.9200 | 0.0083 | 0.2538 | 0.033 |
| H2CS | CIS | effective_charge | S0-T1 | 187.1683 | 187.1700 | 0.0017 | 0.3743 | 0.005 |
| H2CS | CIS | effective_charge | S0-T2 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CS | CIS | effective_charge | S0-T3 | 147.0456 | 147.0500 | 0.0044 | 0.2941 | 0.015 |
| H2CS | CIS | effective_charge | S1-T1 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CS | CIS | effective_charge | S1-T2 | 133.3059 | 133.3000 | 0.0059 | 0.2666 | 0.022 |
| H2CS | CIS | effective_charge | S1-T3 | 142.6787 | 142.6800 | 0.0013 | 0.2854 | 0.005 |
| H2CS | CIS | effective_charge | S2-T1 | 141.8015 | 141.7900 | 0.0115 | 0.2836 | 0.041 |
| H2CS | CIS | effective_charge | S2-T2 | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.000 |
| H2CS | CIS | effective_charge | S2-T3 | 133.3913 | 133.4000 | 0.0087 | 0.2668 | 0.033 |

**Against ORCA's print precision** (report):

- CASSCF: the differences are at most 0.0007 cm-1, the scale of ORCA's 3-decimal print; ratios at most 0.004.
- CIS: the differences are mostly below ORCA's 2-decimal rounding (an rss of three components each rounded by up to 0.005 has a rounding error of up to about 0.009). Four H2CS pairs exceed it: S1-T2 0.0139 and S2-T1 0.0079 (SOMF); S2-T1 0.0115 and S2-T3 0.0087 (effective charges). Cause (the Task 6 reviewer's measurement): ORCA's default TDA convergence; with `%tddft etol 1e-10 rtol 1e-8` ORCA's S1-T2 moves from 126.38 to 126.39 (difference 0.0039) and S2-T1 from 134.59 to 134.60 (0.0021), both below print rounding. The committed ORCA inputs were not changed after this was seen (controller ruling). Our side is not the cause: for H2CO, `td_conv_tol` 1e-10 and 1e-7 give the same energies and rss to 4 decimals.
- The largest ratio is 0.055.

The test file has 15 tests (the 8 combinations, six missing-block and incomplete-block parser tests, and an unreadable-NONZERO-line test). Since the final fix wave only the 8 combinations carry `orca` and `pyscf` and need `ORCA_EXE`; the 7 parser tests run in the fast tier. Its slowest test (H2CO CASSCF SOMF) takes 4.8 s, so none is `slow`.

## 7. Tests (asserted)

| Task | Command | Result |
|---|---|---|
| 2 | `pytest tests/electronic/test_soc_operators.py -q -W error` | RED 5 failed, 8 errors (`ImportError`); GREEN **13 passed** in 3.31 s; with `OPENBLAS_CORETYPE=SandyBridge` 13 passed; after the controller's fix 15 passed |
| 3 | `pytest tests/electronic/test_soc_casscf.py -q -W error` | RED 12 failed, 4 passed (the reference self-tests); GREEN **18 passed** in 6.73 s; SandyBridge 18 passed; with the operators' file 31 passed |
| 4 | `pytest tests/electronic/test_soc_response.py tests/electronic/test_soc_casscf.py -q -W error` | RED 14 failed, 6 passed; GREEN **38 passed** in 28.23 s (53 with the operators' file); SandyBridge 38 passed; after the fix round the three files **57 passed** in 27.11 s, SandyBridge (two files) 42 passed |
| 5 | `pytest tests/electronic tests/config tests/test_cli_single_point.py tests/test_cli.py -q -W error -m "not slow"` | **378 passed**, 9 deselected in 294.44 s (before the table fix; `tests/test_cli_single_point.py` re-run alone 13 passed) |
| 5 | the resume test, 8 repeats after the owner decision | 8 of 8 passed |
| 5 | the two new files (`test_soc_assembly.py`, `test_soc_backend.py`), 3 repeats | 22 passed each time |
| 5 | SandyBridge, the new files with `test_base.py`, `test_factory.py`, `tests/config/test_electronic.py`, `tests/test_cli_single_point.py`, `-m "not slow"` | 77 passed in 30.86 s |
| 5 (fix `cb96020`) | the four SOC files, `tests/test_cli.py`, `tests/config/test_electronic.py`, `tests/electronic/test_factory.py`, `test_base.py`, `-m "not slow"` | **129 passed**, 2 deselected in 77.36 s; SandyBridge 57 passed |
| 6 | with `ORCA_EXE`: `pytest tests/validation/test_orca_soc.py -q -W error` | **14 passed** in 23.61 s; SandyBridge 14 passed in 25.58 s; without `ORCA_EXE` 14 skipped; after the fix round **15 passed** in 72.84 s |
| 7 (controller, at `1c2fcdd`) | fast tier `pytest -m "not slow and not pyscf" -W error -q` (on `/mnt/c`) | **1441 passed, 22 skipped, 360 deselected in 890.80 s** |
| 7 (controller, at `1c2fcdd`) | `pytest tests/electronic tests/test_cli_single_point.py -W error -q` (all markers) | **242 passed in 312.52 s** |
| 7 (controller, at `1c2fcdd`) | `pytest -m "pyscf or slow" -W error -q --ignore=tests/validation/test_tully_exact.py --ignore=tests/validation/test_tully_overlap_exact.py --ignore=tests/validation/test_wavepacket2d.py` (no SHARC/ORCA) | **256 passed, 69 skipped, 1454 deselected, 2 xfailed in 753.68 s** (the xfails are the two strict P7b ones of `test_pyscf_dynamics.py`) |
| 7 (controller, at `1c2fcdd`) | with `ORCA_EXE`: `pytest -m orca -W error -q` | **18 passed, 1805 deselected in 24.19 s** (the 15 SOC tests, the 2 triplet tests and the frequency test) |
| final fix wave | the SOC files, `test_base.py`, `test_factory.py`, `tests/test_cli.py`, `tests/test_cli_single_point.py`, `tests/trajectory/test_fake_backend.py`, `tests/site`, `tests/test_package.py` (all markers); `tests/validation/test_orca_soc.py` with and without `ORCA_EXE`; the changed files under SandyBridge | **192 passed** in 232.91 s; **15 passed** in 115.48 s with ORCA, **7 passed, 8 skipped** without; SandyBridge **88 passed** in 72.57 s; `-m orca` now collects 11 (the 7 parser tests left the `orca` mark) |

**What ran on the branch:** the four tiers of Task 7 and the task runs above. **Not run on the branch:** `tests/validation/test_tully_exact.py`, `test_tully_overlap_exact.py` and `test_wavepacket2d.py` (analytic models; their code is untouched), `pytest -m sharc`, and the full suite `pytest -W error` in one run. The existing tests passed with unchanged assertions in those runs, among them `tests/stages/test_fingerprints.py::test_electronic_fingerprint_is_the_section_plus_the_marker` (thanks to the serializer, section 1.3).

## 8. Report items and limits (measured, no rule)

- **The SA-CASSCF SOMF density is not tested.** Ours is the state average (PR4), ORCA's is the state average too (inference, section 2.4), and a different choice would move the elements by about 0.1 %, below the 0.2 % rule.
- **Signs and phases are not compared with ORCA**, only root sums of squares; the overall sign of i is a convention (section 3). The phases are tested internally: the continuity test (no sign change along the path) and the resume test (the phased elements agree although the raw phases differ).
- **The zero-by-symmetry pairs pass at the 0.05 cm-1 floor** trivially (20 S-T and 8 T_J-T_J pairs).
- **CIS triplet-triplet elements have no external reference** (ORCA's TDA prints none); they are checked by the brute-force reference, the Wigner-Eckart ratios, Hermiticity, invariance and continuity only.
- **TDDFT (non-TDA) SOC** uses X only, renormalised (the overlaps' approximation, as designed); it is checked against the reference with the same `td`, not against another program.
- **The CASSCF CI algebra holds O(ncas^2 n_det) per array** (`_excitations`), about 1 GB per array at CAS(12,12) (Task 3 review; an estimate, not measured on a large active space).
- **No SHARC comparison:** SHARC is not used in this sub-project (its `SHARC_PYSCF.py` refuses triplets, sub-project 1).
- **The cost of the SOMF operator** (measured by the final whole-branch review, not re-measured): benzene RHF with 4 threads, SOMF 12.8, 10.0 and 13.6 s for 6-31G, 6-31G\* and cc-pVDZ (66, 96 and 114 basis functions), against 0.3-0.7 s for the SCF. `single-point` with triplets always computes it, with no opt-out (an opt-out is the owner's decision). An input to sub-project 3's design: requesting the couplings at every step pays this at every step.
- **The single-point Hamiltonian in the dynamics core** (the final whole-branch review's independent check, not a test): H2CO and H2CS SA-CASSCF through `diagonal_basis`: Hermitian to 5.4e-20 / 4.3e-19 Eh, eigenvalues within 0.25 / 4.0 cm-1 of the spin-free energies, U tracked smoothly along a random 0.02-bohr path (smallest |diag(U_prev^H U)| 1.0000); the near-degenerate CIS gaps (1.1e-8 to 1.6e-8 Eh) lie above `DEGENERACY_BLOCK_TOLERANCE` (1e-10). HBr and Br2CO TDA/cc-pVDZ SOMF couplings about 1267 cm-1, Hermitian to 1.2e-18 Eh.
- **The TDA resume margin of the spin-orbit elements is thin:** 2.17e-11 against 1e-10 (factor 4.6), from the same one-cycle SCF difference (section 5).

## 9. Where the work differs from the owner-approved design

1. **The effective nuclear charges cover H, C, O and S only** (spec 3.1 said H-Ar from Koseki, Schmidt, Gordon (1992, 1995) and Koseki, Gordon, Schmidt (1998)). The papers' tables could not be read; the four values are those ORCA 6.1.1 uses, measured by atomic ratios (section 2.5), and they are not checked against the papers. Any other element is refused with a `ConfigError` before any SCF; `soc_operator = "somf"` works for every element. The 1998 paper (J. Phys. Chem. A 102, 10430) treats the transition metals only, so it has no H-Ar values; the spec's citations are corrected in section 12 and the spec erratum.
2. **The overall sign of i is fixed from the operator definition**, not "against ORCA's printed sign of `<T1|HSO|S0>`" (the plan's Conventions). ORCA's printed sign follows the arbitrary phase of its T1 CI vector, so it cannot fix a convention (section 3). The sign is invisible to every rss comparison; sub-project 3 sees only phase-consistent elements.
3. **The resume bound 1e-10 Eh applies to the off-diagonal (spin-orbit) elements only** (spec 4 item 4 said "the same phased Hamiltonian ... within 1e-10 Eh"). Owner decision 2026-11-27, after the numbers were seen (section 11, item 1). The diagonal difference is reported.
4. **ORCA prints no TDA triplet-triplet elements**, so spec 4 item 1's "T-T pairs the same where ORCA prints them" covers the CASSCF T-T pairs only (the spec foresaw this).

Smaller differences:

- `soc.assemble_hamiltonian` was built in Task 5 (the Conventions define it, no task listed it; controller ruling); besides Hermiticity it checks that the T-T block has no spin-orbit element on its diagonal (1e-12 Eh) before setting the energies there.
- The ORCA TDA runs use `nroots 6` and map the states by energy (ORCA's `nroots 3` skips the H2CO S3, section 2.2).
- The ORCA test's CIS uses `td_conv_tol` 1e-7 (section 11, item 2).
- The rss of a pair is defined as "each distinct matrix element once" (S-T 3 elements, T_J-T_J' 9, T_J-T_J the 3 with M < M'); spec 4 said "the root sum of squares over its components".
- The brute-force comparison has two systems beyond the spec's H2CO/STO-3G pair: ethylene SA-CASSCF(2,2) 2S+1T (an odd core, where the core-embedding sign matters, and a triplet with no active beta electron) and TDDFT-B3LYP 2S+2T, and two random operators besides SOMF.
- `ground_state_fingerprint` always drops `soc_operator` (no S0 calculation uses it), so a non-default value makes `excitations` and `dynamics` stale but not `optimize` or `frequencies`; and a non-default value is hashed even with `triplets = 0`, where it changes no number (section 13).
- The default `soc_operator` is left out of **every** dump (a `model_serializer`), not only out of the fingerprint, so `config.json` stays byte-identical.
- Ghost atoms are refused by the effective-charge operator (no table entry).

## 10. Plan rulings PR1-PR5 and the controller's rulings

- **PR1** the geometries optimised with our own geomopt at RHF/6-31G\* (section 2.1).
- **PR2** the operator calibration as a measurement (section 3).
- **PR3** all SOC code in `electronic/soc.py`; `pyscf_casscf.py` and `pyscf_response.py` unchanged.
- **PR4** the SA-CASSCF SOMF density is the state average `mc.make_rdm1()` (ORCA's too, inference; checked at the call site by a spy).
- **PR5** the intra-triplet elements computed and assembled like the others; `single-point` shows T_J-T_J rows.

Controller rulings (ledger): `assemble_hamiltonian` in Task 5; Task 3's Wigner-Eckart test with explicit ratio tables; the Z_eff table restricted to the measured four; ORCA TDA `nroots >= 6` with the states mapped by energy; the CASSCF density cannot be told apart (stated, not tested); the sign of i fixed from the operator; Task 3's additions accepted (ethylene, random operators); the CIS `td_conv_tol` 1e-7 in the ORCA test; the ORCA inputs kept as committed after the reviewer's tighter-ORCA measurement.

## 11. What was decided after numbers were seen

1. **The resume bound (owner decision 2026-11-27, asked as a multiple choice after the numbers were seen).** Task 5's resume test, as pre-registered (the whole phased Hamiltonian within 1e-10 Eh), failed in 9 of 14 runs on the TDA side by 1.330e-8 to 1.336e-8 Eh, on the S1 **diagonal** (a spin-free energy) at point 3; the spin-orbit elements were within 2.17e-11 even then, and CASSCF passed every run (at most 2.6e-13). The cause is the SCF's cycle count at the default gradient criterion (section 5). The owner chose: the 1e-10 bound applies to the off-diagonal spin-orbit elements only, for both methods; the diagonal difference is reported, not asserted; the TDA finding is a documented limitation. Nothing else changed (system, path, settings, other bounds). Cost if wrong: a resume that changed the spin-free energies by more than 1e-10 would not fail this test (the energies' own resume behaviour is the backend's, measured here at 1.3e-8 for TDA).
2. **The ORCA test's CIS `td_conv_tol` 1e-7 (decided after one run).** The first run, with 1e-10 copied from Task 5, gave 6 passed and 2 failed: H2CS CIS with both operators never reached a comparison, because our backend refused the TDA as unconverged. Probe: PySCF's Davidson leaves residuals of about 2.5e-8 to 2.7e-8 on S2, T2 and T3, at every `td_conv_tol` from 1e-10 to 1e-8 (including the program default 1e-8), at `nstates` 5 and 8 and with `max_cycle` up to 1000. The mechanism (the Task 6 reviewer): in PySCF 2.14 `td_conv_tol` is the residual tolerance, and `_lr_eig.eigh` stops with "Linear dependency in trial subspace" once no new trial vector has a squared norm above `td.lindep` (default 1e-12); the H2CS triplets at tol 1e-8 converged `[T, F, T]` at lindep 1e-12 and `[T, T, T]` at 1e-16. 1e-7 is the tightest value that converges. No H2CS coupling had been seen when it was chosen; the H2CO numbers had been seen and passed, and H2CO at 1e-10 and 1e-7 gives the same numbers to 4 decimals. The energy error is second order in the residual, the coupling error first order (about 1e-6 relative). No bound, system or method changed. A remedy in the product (a lower `lindep` in `pyscf_response._solve`) is the owner's decision; it is documented as a limitation.
3. **ORCA TDA `nroots 6`** (after Task 1 saw `nroots 3` skip the H2CO S3; before any comparison of ours with ORCA's TDA elements).
4. **The Z_eff table restricted to H, C, O, S** (after Task 1 could not establish the literature values; before any comparison).
5. **The sign of i from the operator definition** (after Task 2 found that ORCA's printed sign cannot fix it).
6. **Task 3: the ethylene SOMF non-vacuity check skipped** (after its S-T elements were seen to be zero by symmetry, 7.95e-12); the other two operators keep the check on ethylene.
7. **The ORCA inputs kept as committed** after the reviewer showed that tighter ORCA TDA convergence brings the four H2CS CIS excesses below print rounding (the bound passes either way).
8. **When the bounds not fixed by the spec were fixed** (final review; neither was changed after its own numbers were seen):
   - **The energy bound 1e-6 Eh of the ORCA test** (energies first) is not in the spec. It was written into the plan (Task 6 Step 1, plan commit `bcce9a5`) before any task ran, after the design probes of spec 1.1, and the controller restated it after Task 1 (with the `nroots 6` ruling), when Task 1's SA-CASSCF energies of the two programs (within 3e-8 Eh) had been seen; before Task 6 ran.
   - **Task 5's readings of the spec bounds** (task-5-report, "Decisions the brief did not fix"): the invariance per pair relative to its own rss with a floor of 1e-3 of the largest pair; a sign change as Re(a conj b) <= 0 for an element above 1 cm-1 at both points; the continuity allowance on the off-diagonal elements only. They were fixed by the Task 5 implementer when writing the tests, which were run red first, so before any Task 5 number was measured, but after the numbers of Tasks 1-4 (among them ORCA's couplings of the test system, Task 1); neither the spec nor the brief fixed them; no later change is recorded; the Task 5 review accepted them.

No spec-fixed bound (0.05 cm-1 / 0.2 %, 1e-12, 1e-10, 1e-7, the continuity rule) was changed; the resume bound's scope was narrowed by the owner (item 1).

## 12. References (verified with Consensus by the controller, 2026-11-27)

- B. A. Hess, C. M. Marian, U. Wahlgren, O. Gropen, "A mean-field spin-orbit method applicable to correlated wavefunctions", *Chem. Phys. Lett.* **251**, 365 (1996), doi:10.1016/0009-2614(96)00119-4 (title, journal, year and DOI confirmed; the page 365 as in the spec, not shown by Consensus).
- F. Neese, "Efficient and accurate approximations to the molecular spin-orbit coupling operator and their use in molecular g-tensor calculations", *J. Chem. Phys.* **122**, 034107 (2005), doi:10.1063/1.1829047 (its abstract: the SOMF spin-other-orbit exchange is twice the spin-same-orbit one, i.e. the factor 3/2; effective nuclear charges within about 10 % of SOMF on average).
- S. Koseki, M. W. Schmidt, M. S. Gordon, "MCSCF/6-31G(d,p) calculations of one-electron spin-orbit coupling constants in diatomic molecules", *J. Phys. Chem.* **96**, 10768 (1992), doi:10.1021/j100205a033 (effective charges for the first- and second-row elements, all-electron 6-31G(d,p)).
- S. Koseki, M. S. Gordon, M. W. Schmidt, N. Matsunaga, "Main group effective nuclear charges for spin-orbit calculations", *J. Phys. Chem.* **99**, 12764 (1995), doi:10.1021/j100034a013 (periods 2-5, with ECP basis sets). **The spec had this paper's author order wrong.**
- **The spec's 1998 reference is wrong for this table:** *J. Phys. Chem. A* **102**, 10430 (1998), doi:10.1021/jp983453n, is S. Koseki, M. W. Schmidt, M. S. Gordon, "Effective nuclear charges for the first- through third-row transition metal elements in spin-orbit calculations": transition metals, not H-Ar. It is cited nowhere for our table.

The papers' tables were not read: our four Z_eff values are ORCA's (measured), not checked against Koseki 1992.

## 13. What a project sees

- **Every existing project:** nothing changes. `soc_operator` at its default is left out of every dump, so `.gqteashmd/config.json` and every fingerprint are byte-identical (checked against `main` in the Task 5 review); no stage version changed; no stage reads the key or requests SOC. `triplets = 0` calls no spin-orbit integral.
- **`single-point` with `triplets > 0`** prints the SOC table and writes `hamiltonian_hartree`, `hamiltonian_labels` and `soc_operator` to its JSON, and always pays the SOMF cost of section 8 (with `soc_operator = "effective_charge"` and an element outside H, C, O, S it stops with exit 2 before any calculation).
- **Setting `soc_operator = "effective_charge"` in a project** changes the `excitations` and `dynamics` fingerprints (they hash the whole `[electronic]` section), also with `triplets = 0`, where it changes no number: the next run stops with exit 3 and needs `--rerun-from excitations`. `optimize` and `frequencies` are not affected (`ground_state_fingerprint` drops the key). The input is accepted silently in that case (deferred minor).
- **With a `[dynamics]` table** triplets are still refused (exit 2) until sub-project 3.

## 14. Honest summary

- *Asserted:* the SOC matrix elements of SA-CASSCF and CIS agree with ORCA 6.1.1 on H2CO and H2CS for both operators: 32 S-T and 4 T1-T2 pairs with real couplings (largest ratio 0.055 of the allowance; CASSCF at most 0.0007 cm-1), 20 S-T pairs zero on both sides, the T_J-T_J pairs zero; the production formulas equal a brute-force Slater-Condon reference to 9.0e-16 Eh for SA-CASSCF, CIS and TDDFT and three operators; the assembled Hamiltonian is Hermitian (2.71e-20 Eh), invariant under rotation and translation (6.15e-11 relative), continuous along a stretch (0.0155 of the allowance, no sign change); the spin-orbit elements survive a save and load (2.18e-11 Eh); `soc = False` and `triplets = 0` call no spin-orbit integral and every existing fingerprint is unchanged.
- *Report:* the SOMF cost (benzene 10-14 s against 0.3-0.7 s for the SCF, the final review's measurement); the reviewer's check that the Hamiltonian goes through `diagonal_basis`; the operator calibration (each part of the SOMF operator against ORCA, below print rounding); ORCA's Z_eff for H, C, O, S (measured, 1e-4); the TDA resume energies differ by 1.3e-8 Eh at the default SCF criterion; H2CS CIS 3S+3T/6-31G does not converge at the default `td_conv_tol` (PySCF's linear-dependency stop).
- *Inference:* ORCA's SA-CASSCF SOMF uses the state-averaged density, as ours.
- *Not covered:* signs and phases against another program; the CASSCF density choice; CIS triplet-triplet elements against another program; TDDFT (non-TDA) against another program; effective charges beyond H, C, O, S; a SHARC comparison; any stage (no stage requests SOC; dynamics with triplets is sub-project 3).
- *Decided after numbers were seen:* the scope of the resume bound (owner decision) and six test or setting choices (section 11); no spec-fixed bound value changed.
- *Not run on the branch:* the three analytic-model validation files, `pytest -m sharc`, the full suite in one run.

## 15. Deferred minors (ledger, "minor (deferred)" lines)

- Task 1: the `nroots 6` ORCA run proving the skipped H2CO S3 has no committed excerpt (Task 6 re-runs ORCA with 6 roots and checks the energies against ours); the PySCF TDA comparison energies have no script excerpt; the "no T-T in TDA" negative claim lacks the grep; not every table cell was spot-checked.
- Task 2: the pure-imaginarity assertion is vacuous by construction (kept, plan-mandated); the effective-charge error names `soc_operator`, the key name Task 5 kept.
- Task 3: `_excitations` memory O(ncas^2 n_det) per array (about 1 GB at CAS(12,12)); hoist `des_a` out of the p loop or use `trans_rdm1s` for T-T (a later robustness item); `except Exception` around the CI algebra turns programming bugs into `ElectronicStructureError` (the existing pattern). (Task 5's PR4 call-site check was done at its review, with a spy.)
- Task 5: the TDA resume off-diagonal margin is thin (2.17e-11 against 1e-10, factor 4.6, from the same one-cycle SCF difference); a non-default `soc_operator` with `triplets = 0` is accepted silently and makes `excitations`/`dynamics` stale (section 13).
- Task 6: the CASSCF density choice untested (0.1 % < 0.2 %); signs and phases not compared; the symmetry-zero pairs pass at the floor; CIS T-T has no ORCA reference; the H2CS CIS differences of 0.008-0.014 cm-1 above ORCA's print rounding (cause: ORCA's default TDA convergence, the reviewer's measurement).
