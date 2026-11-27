# Spin-orbit coupling matrix elements from PySCF: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute spin-orbit coupling (SOC) matrix elements between the singlet and triplet states of the PySCF backend (SA-CASSCF and TDA/TDDFT) with two operators (SOMF, the default, and one-electron effective nuclear charges), return them as the complex spin-expanded Hamiltonian the dynamics core uses, print them in `single-point`, and validate them against ORCA 6.1.1, a brute-force reference and physical invariances.

**Architecture:** A new `electronic/soc.py` builds the effective one-electron spin-orbit operator in the AO basis (three Cartesian components). Method-specific functions turn it into "SOC blocks" (singlet-triplet and triplet-triplet elements for every Ms component): SA-CASSCF from spin-resolved CI transition densities, TDA/TDDFT from closed-form contractions of the X amplitudes. The backend assembles the blocks with the spin-free energies into the canonical spin-expanded Hamiltonian, with the tracked phases applied. A brute-force Slater-Condon evaluator, test-only, is the independent arbiter of the formulas; ORCA is the arbiter of the operators.

**Tech Stack:** Python 3.12, PySCF 2.14 (`int1e_pnucxp`, `int1e_prinvxp`, `int2e_p1vxp1`, `scf.jk.get_jk`, `fci.addons.des_a/des_b/cre_a/cre_b`, `fci.cistring`), NumPy, pytest; ORCA 6.1.1 as a black box; WSL Oracle Linux 9.5.

**Spec:** `docs/superpowers/specs/2026-11-27-soc-matrix-elements-design.md` (owner-approved 2026-11-27). Read it before any task; the plan rulings below are binding where they differ.

## Global Constraints

- Run every command in WSL: `wsl -d OracleLinux_9_5 -- bash -lc 'cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && conda activate gqteashmd-dev && <command>'`. Below, `W '<command>'` stands for that wrapper. ORCA: `ORCA_EXE=/home/ajc/bin/orca_6_1_1/orca`; run ORCA in a temp directory on the WSL filesystem, never in the repository.
- Git from WSL only; never `--no-verify`. One commit per task, conventional subjects; end each commit message with the `Co-Authored-By:` trailer of the model that writes the commit.
- Files with LF line endings. Lint: `ruff format . && ruff check .` (line length 100); `mypy` clean on `src/`.
- **Clean room:** SHARC is not used in this sub-project. ORCA is executed only (its manual may be read).
- **PySCF is imported lazily, only inside functions in `electronic/pyscf_*.py` and `electronic/soc.py`.** `import gqteashmd.cli` must not import pyscf, geometric, h5py or scipy.
- **Unchanged paths:** `Request(soc=False)` (the default) and `triplets = 0` call no spin-orbit integral; the saved electronic state is unchanged; no stage version or fingerprint changes (the new key is left out of the fingerprint at its default).
- Units: Eh internally; cm-1 only in `single-point` output (`core.units.HARTREE_IN_CM1`); alpha = 1/`core.units.SPEED_OF_LIGHT_AU`.
- **Canonical spin-expanded order** (`dynamics/spin.py`): S0 ... S(n-1), then T1 ... Tm with Ms = -1, then with Ms = 0, then with Ms = +1. Labels from `dynamics.spin.expanded_states(...)[k].label()`.
- Fixed numbers (spec 4): ORCA rule |ours - ORCA| <= max(0.05 cm-1, 0.2 % of ORCA) on the root sum of squares over components per pair; Hermitian within 1e-12 Eh; S-S exactly 0; Wigner-Eckart ratios within a relative 1e-10; brute-force reference within 1e-10 Eh; rotation/translation invariance within a relative 1e-7; operator reductions within 1e-12; continuity: no sign change of an element > 1 cm-1, per-step change <= 10 % + 0.05 cm-1; resume within 1e-10 Eh.
- Tests: `pytest ... -W error`; no bit-for-bit BLAS comparison or eigenvector sign reliance; new tests checked once with `OPENBLAS_CORETYPE=SandyBridge`. Markers: `pyscf` (and `slow` if over about 30 s), `orca` (skip without `ORCA_EXE`).
- The ledger is `.superpowers/sdd/2026-11-27-soc-matrix-elements/progress.md` (git-ignored); every task writes `task-N-report.md` there.

## Conventions (binding; every task uses them)

- **The spatial operator** is three real antisymmetric AO matrices `A[k]` (k = x, y, z) such that the effective one-electron spin-orbit operator is `H_SO = sum_{pq} sum_k (i A[k]_pq) S^k_pq` with `S^k_pq = sum_{sigma sigma'} a+_{p sigma} (s_k)_{sigma sigma'} a_{q sigma'}`, `s = sigma_Pauli / 2`. So `h[k] = 1j * A[k]` is Hermitian and purely imaginary. `soc.py` returns `h` with shape (3, nao, nao), complex. The overall sign of `i` is a phase convention that the root sums of squares cannot see; Task 2 fixes it once against ORCA's printed sign of `<T1|HSO|S0>` and records it, and it does not change afterwards.
- **Spherical components:** `h_plus = h[0] + 1j h[1]`, `h_minus = h[0] - 1j h[1]`, so that `sum_k h_k s_k = h_z s_z + (h_plus s_minus + h_minus s_plus) / 2`.
- **SOC blocks** (`soc.SocBlocks`, frozen dataclass): `st` complex (n_singlets, n_triplets, 3) = `<S_I | H_SO | T_J, M>` for M = -1, 0, +1 (index 0, 1, 2); `tt` complex (n_triplets, n_triplets, 3, 3) = `<T_J, M | H_SO | T_J', M'>`. Triplet components follow Condon-Shortley phases: `|T, M-1> = S_minus |T, M> / sqrt(2)` for M = +1, 0 (`S_minus = sum_p a+_{p beta} a_{p alpha}`).
- **Assembly** (`soc.assemble_hamiltonian(energies_s, energies_t, blocks, phases_s, phases_t) -> np.ndarray`): the N_exp x N_exp matrix with the spin-free energies on the diagonal (each triplet energy at its three components), `st` in the S-T blocks times `p_I p_J`, `st` conjugated in the T-S blocks, `tt` in the T-T blocks times `p_J p_J'`; it asserts Hermiticity within 1e-12 Eh (an `AssertionError` is an internal error).

## Plan rulings (binding where they differ from the spec)

- **PR1, the geometries (spec 4).** Task 1 optimises H2CO and H2CS at RHF/6-31G\* with PySCF + geomeTRIC (`gqteashmd.geomopt.optimize_geometry` with an RHF energy-and-gradient callable, convergence `GAU_TIGHT`) and commits the results as XYZ files in Angstrom with 10 decimals; the ORCA inputs use those exact coordinates.
- **PR2, the operator calibration (spec 3.1).** Task 2 records, as a measurement (not a pre-registered test), our one-electron-only and SOMF S0-T1 root sums of squares for H2CO CIS/6-31G against ORCA with `SOCFlags 1,0,0,0` (1-electron only), `1,4,0,0` (plus Coulomb) and `1,4,4,0` (plus exact exchange). This catches a prefactor or contraction-pattern error before Tasks 3-4 build on it. A mismatch beyond the spec-4 rule at this stage is a BLOCKED report, not a tolerance change.
- **PR3, where the method functions live.** `casscf_soc_blocks` and `response_soc_blocks` live in `electronic/soc.py` (one module for all SOC code; it imports PySCF only inside functions). `pyscf_casscf.py` and `pyscf_response.py` are not changed except where a task says so.
- **PR4, the SOMF density for SA-CASSCF.** Ours is the state-averaged one-particle density (`mc.make_rdm1()`, AO basis). Task 1 measures ORCA's choice (by running its CASSCF SOC with different weights, or from its printout); if it differs, Task 6's CASSCF comparison states the difference in its report and the rule still applies.
- **PR5, intra-triplet elements.** `tt[J, J]` (the SOC within one spin-free triplet, M != M') is computed and assembled like every other T-T element; the `single-point` table shows T_J-T_J rows too.

## File structure

| File | Task | Responsibility |
|---|---|---|
| `tests/data/formaldehyde.xyz`, `tests/data/thioformaldehyde.xyz` | 1 | the optimised geometries |
| `src/gqteashmd/electronic/soc.py` (new) | 2-5 | operators, `SocBlocks`, CASSCF and TDA blocks, assembly |
| `tests/electronic/soc_reference.py` (new, not collected) | 3 | brute-force Slater-Condon reference |
| `tests/electronic/test_soc_operators.py`, `test_soc_casscf.py`, `test_soc_response.py`, `test_soc_backend.py` (new) | 2-5 | tests |
| `src/gqteashmd/config/schema.py`, `electronic/factory.py` | 5 | `soc_operator`, fingerprint rule |
| `src/gqteashmd/electronic/base.py`, `pyscf_backend.py` | 5 | `Request.soc`, `ElectronicResult.hamiltonian(_labels)`, assembly with phases |
| `src/gqteashmd/cli.py` | 5 | `single-point` SOC table and JSON |
| `tests/validation/test_orca_soc.py` (new) | 6 | the ORCA comparisons |
| documents | 8 | USER_MANUAL, CLAUDE.md, CHANGELOG, report |

---

### Task 1: ORCA facts and the two geometries

**Files:**
- Create: `tests/data/formaldehyde.xyz`, `tests/data/thioformaldehyde.xyz`
- Ledger: `task-1-report.md` (the facts below) and `task-1-files/` (ORCA inputs and the relevant output excerpts)

**Interfaces:** Produces the facts Task 2 (calibration), Task 6 (parsers, T-T availability, CASSCF density, Z_eff) need.

- [ ] **Step 1: Geometries (PR1).** Write a throwaway script (not committed) that builds the molecule, runs `geomopt.optimize_geometry(symbols, coords_bohr, energy_and_gradient, max_steps=100, convergence="GAU_TIGHT", work_dir=<tmp>)` with an RHF/6-31G\* energy and gradient from PySCF, for H2CO (planar, C=O along z, start C 0 0 0; O 0 0 1.21; H 0 ±0.94 -0.59 Angstrom) and H2CS (start C 0 0 0; S 0 0 1.61; H 0 ±0.93 -0.58). Write the converged coordinates to the two XYZ files (comment line: `RHF/6-31G* optimised (gqteashmd geomopt, GAU_TIGHT)`, 10 decimals). Record the final energies and max gradients.
- [ ] **Step 2: ORCA TDA (CIS) SOC** for both molecules at those geometries, 6-31G, `! RHF 6-31G NoRI NoFrozenCore TightSCF`, `%tddft nroots 3 tda true triplets true dosoc true end`, and `%rel SOCType 3 SOCFlags 1,4,4,0 end`; again with `SOCType 1` (effective charges). Record: the printed blocks and their units/precision; whether T-T elements are printed (search the output); the Cartesian component convention (`Z X Y` columns of `<T|HSO|S>`).
- [ ] **Step 3: ORCA SA-CASSCF SOC:** `%casscf nel 4 norb 3 mult 1,3 nroots 2,2 bweight 2,2 weights[0]=1,1 weights[1]=1,1 etol 1e-10 gtol 1e-6 rel dosoc true end end` with `%rel SOCType 3 SOCFlags 1,4,4,0 end` (and `SOCType 1`); raise the print level if needed to get the SOC matrix elements between the spin-free states (`printlevel` or the documented keyword; find it in ORCA's output or manual). Record: the format of the S-T and T-T elements; which density ORCA's CASSCF SOMF uses (PR4: run once with unequal weights, e.g. `weights[0]=3,1`, and see whether the SOC changes beyond the state change; or find it printed); the CASSCF energies (to compare the solution with ours later: same active space).
- [ ] **Step 4: Z_eff:** record the effective nuclear charges ORCA prints or uses for H, C, O, S with `SOCType 1` (if printed), and the literature values (Koseki 1992/1995/1998) for H-Ar; state whether they agree.
- [ ] **Step 5:** commit the two XYZ files (`test: add formaldehyde and thioformaldehyde geometries`); write `task-1-report.md`.

---

### Task 2: the operators (`electronic/soc.py`)

**Files:**
- Create: `src/gqteashmd/electronic/soc.py`, `tests/electronic/test_soc_operators.py`

**Interfaces:**
- Produces:
  - `KOSEKI_ZEFF: dict[str, float]` (H-Ar, from the literature; Task 1's table);
  - `one_electron_operator(mol) -> np.ndarray` (3, nao, nao) complex: `h = 1j * (alpha^2 / 2) * A` from `int1e_pnucxp` with the sign fixed by Task 2's calibration (Conventions);
  - `effective_charge_operator(mol, zeff: Mapping[str, float] | None = None) -> np.ndarray`: per atom `int1e_prinvxp` with the origin at the nucleus (`mol.with_rinv_origin`), scaled by Z_eff (default `KOSEKI_ZEFF`), same prefactor and sign; a missing element raises `ConfigError` naming it;
  - `somf_operator(mol, dm, *, two_electron: bool = True) -> np.ndarray`: the one-electron term plus, when `two_electron`, the SOMF mean field of the AO density `dm` (Coulomb minus the two exchange terms with the factor 3/2; contracted with `scf.jk.get_jk(mol, ..., intor="int2e_p1vxp1", comp=3)`; the exact index patterns are derived in this task from the integral's definition and checked against ORCA by PR2);
  - `soc_operator(mol, dm, kind: str) -> np.ndarray` dispatching `"somf"` / `"effective_charge"`;
  - every PySCF call inside `wrap_failure`.

- [ ] **Step 1: Failing tests** (`tests/electronic/test_soc_operators.py`, marker `pyscf`; H2CO/6-31G from `tests/data/formaldehyde.xyz`):
  - `h` has shape (3, nao, nao), is Hermitian within 1e-12 and purely imaginary (`abs(h.real).max() <= 1e-14`) for all three operators;
  - `somf_operator(mol, dm, two_electron=False)` equals `one_electron_operator(mol)` within 1e-12;
  - `effective_charge_operator(mol, zeff={s: Z(s)})` with the true nuclear charges equals `one_electron_operator(mol)` within 1e-12;
  - the two-electron part reduces the one-electron term (screening): for H2CO the Frobenius norm of `somf - one_electron` projected on the O atom's basis functions is negative relative to the one-electron term (i.e. `norm(somf) < norm(one_electron)` for the O-centred block) — a sanity check of the sign of the mean field;
  - `effective_charge_operator` with an atom missing from the table (e.g. `zeff={"H": 1.0}` for H2CO) raises `ConfigError` naming `C`;
  - `KOSEKI_ZEFF` has H, C, O, S (and every element H-Ar) as floats.
- [ ] **Step 2:** run them, see them fail (`ModuleNotFoundError`).
- [ ] **Step 3: Implement** the module. Derive the SOMF mean-field contraction from the definition of `int2e_p1vxp1` (libcint: the spin-orbit operator on electron 1 of the bra pair) and the SOMF formula (Neese 2005: `h^SOMF_pq = h^1e_pq + sum_rs D_rs [ (pq|rs)' - 3/2 (pr|sq)' - 3/2 (sq|pr)' ]`, primes marking the operator component); document the derivation in the docstring.
- [ ] **Step 4: Calibration (PR2):** with a temporary script, compute the H2CO CIS/6-31G S0-T1 SOC root sum of squares with the three operator variants (one-electron only, plus Coulomb, plus exact exchange) using a minimal throwaway S0-T formula (`sum X_ia h_ia` with the CIS-like normalisation; the full TDA formulas come in Task 4) and compare with ORCA `SOCFlags 1,0,0,0`, `1,4,0,0`, `1,4,4,0`. Record the numbers and the sign convention (Conventions) in the report. If any variant misses the spec-4 rule, stop and report BLOCKED with the numbers.
- [ ] **Step 5:** run the tests (also with SandyBridge), lint, mypy, commit (`feat: add the spin-orbit operators`).

---

### Task 3: SA-CASSCF SOC blocks and the brute-force reference

**Files:**
- Create: `tests/electronic/soc_reference.py` (not collected), `tests/electronic/test_soc_casscf.py`
- Modify: `src/gqteashmd/electronic/soc.py`

**Interfaces:**
- Consumes: `soc_operator` (Task 2); `pyscf_casscf.state_counts`, the mixed SA-CASSCF `mc` (sub-project 1: `mc.ci` singlet roots then triplet Ms = +1 roots; triplet `nelec = (na + 1, nb - 1)`).
- Produces: `SocBlocks(st, tt)`; `casscf_soc_blocks(mc, states, h_ao) -> SocBlocks` (`h_ao` from `soc_operator(mol, mc.make_rdm1(), kind)`; transforms to the active MOs; S-T from the spin-flip transition density `<S| a+_{p beta} a_{q alpha} |T, +1>` built with `fci.addons.des_a`/`cre_b`; the other Ms components by Wigner-Eckart (Conventions); T-T from the Ms = +1 alpha-minus-beta transition densities and Wigner-Eckart; the core contributes nothing directly).
- Test-only `soc_reference.py`: `ReferenceState` (a vector over explicit spin-orbital determinants: alpha and beta occupation tuples over all MOs), `casscf_reference_states(mc, states) -> dict[(label, M), ReferenceState]` (singlets from their CI vectors; triplets: the Ms = +1 CI vector, then `S_minus / sqrt(2)` applied twice, with explicit determinants), `one_body_element(bra, ket, h_mo) -> complex` (Slater-Condon with fermionic signs in the order all-alpha-then-all-beta spin orbitals; the operator `sum_k h_k s_k` in spin-orbital form). It shares no code with `soc.py`.

- [ ] **Step 1: Failing tests** (`pyscf`; H2CO/STO-3G CAS(4,3), 2 singlets + 2 triplets, and the 6-31G system of the spec for the invariance checks):
  - `casscf_soc_blocks` equals the reference element by element (every `st[I, J, M]` and `tt[J, J', M, M']`) within 1e-10 Eh;
  - Wigner-Eckart: for each S-T pair, `st[I, J, :]` has the ratio structure of a rank-1 tensor between S = 0 and S = 1 (one independent reduced element; with the Conventions, |st[I,J,0]| relates to the M = ±1 elements through the h_z and h_± components — assert the relation derived in the docstring within a relative 1e-10); for T-T, `tt[J, J', M, M']` follows the 3j-symbol structure of a rank-1 tensor between S = 1 states within a relative 1e-10;
  - the singlet-singlet elements do not exist in the blocks (the assembly later puts exact zeros);
  - with `h_ao = 0` all blocks are exactly zero.
- [ ] **Step 2:** see them fail.
- [ ] **Step 3:** implement `SocBlocks` and `casscf_soc_blocks` in `soc.py`, and the reference in `tests/electronic/soc_reference.py`.
- [ ] **Step 4:** run (also with SandyBridge), lint, mypy, commit (`feat: add the SA-CASSCF spin-orbit matrix elements`).

---

### Task 4: TDA/TDDFT SOC blocks

**Files:**
- Create: `tests/electronic/test_soc_response.py`
- Modify: `src/gqteashmd/electronic/soc.py`, `tests/electronic/soc_reference.py` (add `response_reference_states`)

**Interfaces:**
- Consumes: `ResponseOutcome(td, rung, energies, td_triplet, triplet_rung)` (sub-project 1), `soc_operator`.
- Produces: `response_soc_blocks(mf, td, td_triplet, states, h_ao) -> SocBlocks` (MO transform; S0-T = sum X^T_ia h_ia with the CIS-like normalisation; Sm-Tn and Tn-Tn' from the occupied-occupied and virtual-virtual contractions by Slater-Condon; X only, renormalised per state; Wigner-Eckart for the components); test-only `response_reference_states(mf, td, td_triplet, states)` (CIS determinants over all MOs: singlets `sum X_ia (|i->a alpha> + |i->a beta>)` normalised, triplets built at Ms = 0 as `sum X_ia (|i->a alpha> - |i->a beta>)` normalised, then Ms = ±1 by `S_plus`/`S_minus` with the Conventions).

- [ ] **Step 1: Failing tests** (`pyscf`; H2CO/STO-3G CIS, 3 singlets + 3 triplets; TDDFT-B3LYP/STO-3G 2 + 2 for the TDDFT path):
  - `response_soc_blocks` equals the reference element by element within 1e-10 Eh (CIS and TDDFT, with the same X-only states in the reference);
  - the Wigner-Eckart ratios within a relative 1e-10 (as in Task 3);
  - with `h_ao = 0` all blocks exactly zero.
- [ ] **Step 2:** see them fail. **Step 3:** implement. **Step 4:** run (also SandyBridge), lint, mypy, commit (`feat: add the TDA/TDDFT spin-orbit matrix elements`).

---

### Task 5: the backend, the input key, `single-point`

**Files:**
- Modify: `src/gqteashmd/config/schema.py` (`ElectronicSection.soc_operator: Literal["somf", "effective_charge"] = "somf"`), `src/gqteashmd/electronic/factory.py` (`electronic_fingerprint` pops `soc_operator` when it is `"somf"`), `src/gqteashmd/electronic/base.py` (`Request.soc: bool = False` with the constructor argument; `ElectronicResult.hamiltonian: np.ndarray | None = None`, `hamiltonian_labels: tuple[str, ...] | None = None`; `to_json` writes `hamiltonian` as `{"real": ..., "imag": ...}` and the labels when present), `src/gqteashmd/electronic/pyscf_backend.py` (request validation: `soc=True` without triplets is a `ValueError`; in `compute`, when `request.soc`, build `h_ao = soc.soc_operator(mol, D, settings.soc_operator)` with D the SCF density (TDA/TDDFT) or `mc.make_rdm1()` (CASSCF), the blocks, and `soc.assemble_hamiltonian` with the tracked phases split by multiplicity; `soc=False` touches nothing), `src/gqteashmd/cli.py` (`single-point` requests `soc=True` when `triplets > 0`; prints a `Spin-orbit coupling (cm-1, <soc_operator>)` table: one row per (S_I, T_J) and per (T_J, T_J') with J <= J', the three Ms elements as `re+imj` with 2 decimals and the root sum of squares; the JSON gains `hamiltonian`, `hamiltonian_labels`, `soc_operator`; without triplets the output is unchanged)
- Test: `tests/electronic/test_soc_backend.py`, `tests/config/test_electronic.py`, `tests/electronic/test_factory.py`, `tests/test_cli_single_point.py`

**Interfaces:** Consumes Tasks 2-4. Produces `Request(..., soc=True)` -> `result.hamiltonian`, `result.hamiltonian_labels`.

- [ ] **Step 1: Failing tests:**
  - config: `soc_operator` defaults to `"somf"`, accepts `"effective_charge"`, rejects others; `electronic_fingerprint` of a default section has no `soc_operator` key and equals the dict of a section built without the key (so existing fingerprints are unchanged), and contains it for `"effective_charge"`;
  - backend (`pyscf`, H2CO/6-31G, CASSCF 2 + 2 and CIS 3 + 3): `hamiltonian` has shape (n + 3m)^2, is Hermitian within 1e-12 Eh, its S-S off-diagonal block is exactly zero, its diagonal equals the spin-free energies (each triplet three times) exactly; `hamiltonian_labels` equal `[s.label() for s in dynamics.spin.expanded_states((n, 0, m))]`; `soc=True` without triplets raises `ValueError`; with `soc=False` (monkeypatch `soc.soc_operator` to raise) a triplets compute runs and `hamiltonian` is `None`;
  - invariance: the per-pair root sums of squares are unchanged under a rigid rotation (e.g. 30 degrees about an axis not along a symmetry element) plus a translation of the molecule within a relative 1e-7;
  - continuity (spec 4 item 4): 5 points stretching C=O by 0.005 bohr per step (move O along the C-O axis), one backend per chain: between consecutive steps no element with |H_ij| > 1 cm-1 changes sign, and every element changes by at most 10 % + 0.05 cm-1;
  - resume: backend A computes points 0-2, `save_state`, continues to 3-4; backend C loads A's file and computes 3-4; the phased `hamiltonian` of C equals A's within 1e-10 Eh at points 3 and 4;
  - `single-point` (`pyscf`): with `triplets > 0` the SOC table appears and the JSON has `hamiltonian`, `hamiltonian_labels`, `soc_operator`; with `triplets = 0` the output is unchanged (header and rows as before).
- [ ] **Step 2:** see them fail. **Step 3:** implement. **Step 4:** run `W 'pytest tests/electronic tests/config tests/test_cli_single_point.py tests/test_cli.py -q -W error -m "not slow"'` (about 20 min) and the new tests with SandyBridge, lint, mypy, commit (`feat: return the spin-orbit Hamiltonian from the PySCF backend`).

---

### Task 6: the ORCA validation

**Files:**
- Create: `tests/validation/test_orca_soc.py` (markers `orca`, `pyscf`; skip without `ORCA_EXE`)

**Interfaces:** Consumes Task 1's facts (formats, T-T availability, the CASSCF density) and Tasks 2-5.

- [ ] **Step 1:** write the tests: for H2CO and H2CS (Task 1 geometries, 6-31G), for SA-CASSCF CAS(4,3) 2 + 2 and CIS 3 + 3, for `soc_operator = "somf"` (ORCA `SOCType 3`, `SOCFlags 1,4,4,0`) and `"effective_charge"` (ORCA `SOCType 1`): run ORCA in `tmp_path` with the Task 1 inputs, parse its S-T elements (and T-T where printed), compute ours through `PySCFBackend.compute(Request(coords, soc=True))`, and assert for every pair the root sum of squares over its components: **|ours - ORCA| <= max(0.05 cm-1, 0.002 x ORCA)**. Map ORCA's state numbering to ours by energy order (assert the spin-free excitation energies agree within 1e-6 Eh first, so the comparison is between the same states; a solution mismatch is a NEEDS_CONTEXT report, not a tolerance change). Write each parser against the real ORCA output and fail loudly (a clear message) on a missing block.
- [ ] **Step 2:** run them with `ORCA_EXE` set; record every pair's difference in the report. A miss is reported (BLOCKED) with the numbers; no bound changes. Lint, commit (`test: validate the spin-orbit couplings against ORCA`).

---

### Task 7 (controller): the test tiers

- [ ] On the branch head: `W 'pytest -m "not slow and not pyscf" -W error -q'`; `W 'pytest tests/electronic tests/test_cli_single_point.py -W error -q'` (all markers); `W 'pytest -m "pyscf or slow" -W error -q --ignore=tests/validation/test_tully_exact.py --ignore=tests/validation/test_tully_overlap_exact.py --ignore=tests/validation/test_wavepacket2d.py'`; with `ORCA_EXE`, `W 'pytest -m orca -W error -q'`. Record counts and times in the ledger.

---

### Task 8: documents

**Files:**
- Create: `docs/superpowers/reports/2026-11-27-soc-matrix-elements.md` (every number marked asserted, report or inference; the Task 1 facts; the calibration of Task 2; the ORCA table; what was decided after numbers were seen; the honest summary)
- Modify: `USER_MANUAL.md` (the "Read this first" summary; 4.2 `[electronic] soc_operator`; 5.3 the `single-point` SOC table and JSON; 8.6 a new subsection on SOC: operators, formulas, conventions, the canonical order and phases, the validation numbers; limitations; section 11), `CLAUDE.md` (a sub-project block in the style of the others, the `electronic/` description, Current status, Known limitations 4, Remaining work, Environment counts, the deferred minors), `CHANGELOG.md` `Unreleased`, the spec's erratum.
- [ ] Verify the three references of spec 7 with Consensus before citing them (the controller can do it if the implementer has no access); run `W 'pytest tests/site -q -W error'`; commit (`docs: record the spin-orbit coupling matrix elements`).

After Task 8: the final whole-branch review, its fix wave, and superpowers:finishing-a-development-branch (the owner decides the merge).
