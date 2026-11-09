# Implementation Strategy for a Surface-Hopping MD Program (PySCF first, more backends later)

**Working name:** `gqteashmd` (rename as you like)
**Reference code:** SHARC 4.1, local copy at `\\wsl.localhost\OracleLinux_9_5\home\ajc\sharc4-4.1`
**Date:** 2026-11-06 (revised 2026-11-06: PySCF-first decision)

> **Decision (2026-11-06): PySCF is the only electronic-structure backend for the first release.**
> Gaussian and other programs (ORCA, OpenMolcas, OpenQP) will be added later as separate backends behind the same interface (§2.3, §3.5, Phase 7).
> Reasons:
>
> - PySCF runs inside the program, so orbitals, CI amplitudes and overlap integrals are available directly, with no file parsing.
> - It is free and installs with Conda, so it can run in automated tests.
> - It provides SA-CASSCF gradients and nonadiabatic couplings.
> - Gaussian would be the hardest interface to build (undocumented binary records, one gradient per job, no TDDFT couplings or spin–orbit coupling) while adding the least new capability.
>
> Sections on Gaussian are kept as design notes for that later phase.

---

## 0. Executive summary

- **Build a Python-first code.** Write a pure-Python/NumPy dynamics core (Numba only where profiling shows it's needed). Put electronic-structure backends behind one narrow, typed interface. In ab initio surface hopping, more than 99 % of wall time goes to the electronic-structure call, so a compiled core like SHARC's Fortran `sharc.x` buys almost nothing and costs a lot of development speed.
- **Make SHARC's file protocol (`QM.in`/`QM.out`) a first-class adapter, not the internal design.** This one decision makes validation tractable. You can run *your* driver with *SHARC's* interfaces, and *SHARC's* driver with *your* backends. That lets you test dynamics and electronic structure separately.
- **Validate in layers:** analytic models → Linear Vibronic Coupling (LVC) models (cheap, with spin–orbit coupling (SOC), and SHARC has them) → single-geometry electronic-structure comparisons → short deterministic trajectory comparisons → statistical ensemble comparisons. Treat any difference outside the error bars as a bug until proven otherwise.
- **Scope "comparable to SHARC" carefully.** SHARC 4.1 ships about 40 electronic-structure interfaces and a very large option space (see §1). A realistic target is SHARC's *core dynamics feature set* with two well-tested backends, not feature parity.
- **Start with PySCF only.** The first release uses PySCF alone (TDA/TDDFT and SA-CASSCF). The backend interface is designed from day 1 so that Gaussian, ORCA, OpenMolcas or OpenQP can be added later without touching the dynamics core (§3.5).
- **Biggest technical risks:**
  1. SOC with PySCF. SHARC's own PySCF interface is singlet-only CASSCF/PDFT, and PySCF has no ready-made TDDFT state SOC. Intersystem crossing, which is the "SH" in SHARC, is therefore the hardest part of a PySCF-only stack. The most direct later fix is an ORCA backend.
  2. Phase and state tracking.
  3. PySCF-only performance limits for larger molecules.
  4. GPL-3 licensing if any SHARC code is reused.
  5. Later: Gaussian overlaps depend on undocumented RWF binary records.

---

## 1. What SHARC 4.1 actually contains (scope baseline)

This comes from inspecting the local installation, not from memory of the manual.

| Layer | SHARC 4.1 location | Notes |
|---|---|---|
| Dynamics core | `source/*.f90` (Fortran 90; `main.F90`, `nuclear.F90`, `electronic.f90`, `tsh_tu.f90`, `decoherence_*.f90`, `pointer_basis.f90`, `zpe.f90`, `restart.f90`, …) | Monolithic executable `sharc.x`; also `libsharc.so` for in-memory use |
| In-memory driver | `pysharc/` (C extension + Python) | Used for fast models (LVC) without file I/O |
| Interface framework | `lib/SHARC_INTERFACE.py` → `SHARC_ABINITIO.py`, `SHARC_HYBRID.py`, `SHARC_FAST.py`; `qmin.py`, `qmout.py` | Abstract base class with `read_template`, `read_resources`, `setup_interface`, `run`, `getQMout`, `create_restart_files` |
| Interfaces | `bin/SHARC_*.py` (~40: GAUSSIAN, PYSCF, ORCA, MOLCAS, LVC, ANALYTICAL, QMMM, SCHNARC/SPAINN ML, …) | |
| Overlaps | `wfoverlap/` (Fortran, Plasser et al.) | Determinant-based overlaps; used by the Gaussian interface |
| Pre/post-processing | `bin/wigner.py`, `excite.py`, `setup_traj.py`, `data_extractor.x`, `populations.py`, `diagnostics.py`, `make_fit.py`, … | |
| Tests | `tests/INPUT/*` (+ `RESULTS/`), `tests/interface/test_*.py` (pytest) | Includes **`PYSCF_ddr`** and **`GAUSSIAN_overlap`**, both directly reusable for you |

**Dynamics option space** (keywords parsed in `source/input.F90`):

- **Methods:** `method tsh | scp` (scp = self-consistent potential: Ehrenfest, CSDM).
- **Representations:** `surf sharc | diagonal | fish | mch | ehrenfest`.
- **Couplings:** `coupling ddt | ddr | overlap | ktdc` (the last is curvature-driven/Baeck–An style). `ktdc_method gradient | energy`.
- **Electronic EOM** (`eeom`): constant interpolation, linear interpolation, local diabatization, norm-preserving interpolation (NPI).
- **Nuclear integrator:** `integrator fvv` (default) `| avv` (adaptive) `| bsh` (Bulirsch–Stoer–Hairer).
- **Hopping:** `hopping_procedure standard | sharc | gfsh | off`. Also `boltzmann_hop`, `force_hop_to_gs`, `time_uncertainty`.
- **Kinetic-energy correction:** `ekincorrect none | parallel_vel | parallel_pvel | parallel_nac | parallel_diff | parallel_pnac | parallel_pdiff | parallel_enac | parallel_penac`. `reflect_frustrated` takes the same choices.
- **Decoherence:** `decoherence_scheme none | edc | afssh | dom`; SCP `switching_procedure csdm | scdm | ndm`.
- **Other:** `gradcorrect` (with nac/ngt/kmatrix/tdm/enac variants), `pointer_basis`, `zpe_correction pumping | lp`, `nac_projection`, `grad_select`/`nac_select` with `eselect`, `phase_correction_algo`, `track_phase`.
- **Environment:** `thermostat langevin` (with regions), `rattle` constraints, `laser` fields with `dipole_gradient`, `atommask`, `freeze`, `dampeddyn`, `restart`, `output_format ascii | netcdf`.

**What SHARC's interfaces for PySCF (current focus) and Gaussian (planned later) support:**

| | `SHARC_PYSCF.py` | `SHARC_GAUSSIAN.py` |
|---|---|---|
| Methods | SA-CASSCF, L-PDFT, MC-PDFT, CMS-PDFT | CIS/TDA, TDDFT (single reference) |
| Multiplicities | **Singlets only** (`NotImplementedError` otherwise) | Singlets + triplets (triplets from closed-shell or separate UKS job) |
| SOC | No | **No** (the example `QM.in` states this explicitly) |
| Gradients | Yes (analytic) | Yes (`Force`, one root per run) |
| NACs (`nacdr`) | Yes (CASSCF; not for L-PDFT) | **No**; couplings come from **overlaps** |
| Overlaps | No (in this interface version) | Yes: MO coefficients from `.fchk`, CI vectors from `rwfdump` record **635R**, AO overlaps from a doubled-geometry job, then `wfoverlap.x` |
| Other | Dipoles | Dipoles, Dyson norms (`ion`), TheoDORE, multipolar fits, point charges |
| Architecture | Legacy procedural script wrapped by `SHARC_OLD` | Modern class on `SHARC_ABINITIO` |

**Takeaway:** SHARC itself handles ISC/SOC through *other* backends (MOLCAS, MOLPRO, ORCA, LVC, …). With PySCF alone, plan for ISC explicitly (§4.6, Risk R1).

**Interfaces relevant to later backends** (feature lists from SHARC 4.1's `bin/SHARC_*.py`):

| Interface | Methods | Features |
|---|---|---|
| `SHARC_ORCA.py` (ORCA 5–6) | CIS/TDDFT | h, dm, **soc**, grad, overlap, phases, ion, theodore, point charges |
| `SHARC_MOLCAS.py` (OpenMolcas > v23) | CASSCF/RASSCF, (X/RMS-)CASPT2, CMS-PDFT | h, dm, **soc**, **nacdr**, grad, overlap, phases, ion, point charges |
| `SHARC_OPENQP.py` (OpenQP) | MRSF-TDDFT | h, dm, grad, overlap, phases (no soc/nacdr yet) |
| `SHARC_GAUSSIAN.py` (G16) | CIS/TDDFT | h, dm, grad, overlap, phases, ion (no soc/nacdr) |

---

## 2. Recommended software architecture

### 2.1 Guiding principles

1. **Python core, compiled kernels only on evidence.** Per-step dynamics cost scales as O(N_states³ · N_substeps) plus O(N_atoms · N_states²). That is trivial next to a TDDFT gradient. Use Numba only for the analytic/LVC backends and for overlap kernels if profiling demands it.
2. **Strategy objects for every algorithmic choice.** Every SHARC keyword family in §1 becomes an interface with registered implementations: `Integrator`, `ElectronicPropagator`, `HoppingScheme`, `VelocityAdjuster`, `DecoherenceScheme`, `PhaseTracker`. Configuration selects implementations by name. This makes "same as SHARC except X" experiments trivial, and those experiments are your main debugging tool.
3. **Atomic units internally, everywhere.** Convert only at I/O boundaries. Use one `constants.py`. Check its values against CODATA *and* against `sharc4-4.1/lib/constants.py`: small differences in amu→a.u. or Bohr→Å conversions show up as drift in trajectory comparisons.
4. **An immutable per-step record plus a mutable trajectory state.** Every step yields a frozen `StepRecord` (geometry, velocities, energies, matrices, coefficients, active state, random number used, hop decision). That serves output, restart *and* testing.
5. **The backend owns the wavefunction; the driver owns the dynamics.** The driver never touches MO coefficients. It asks for quantities (energies, gradients, NACs, overlaps, dipoles, SOC) and receives arrays.
6. **Determinism by construction.** Use one seeded `numpy.random.Generator` per trajectory, serialized into the restart file. Log every random number consumed. Also support a "replay" mode that reads random numbers from a file, which is useful for SHARC comparisons (§5.4).

### 2.2 Package layout

```
gqteashmd/
├── core/
│   ├── constants.py        # CODATA + unit conversions (a.u. internal)
│   ├── states.py           # StateLabel(mult, index, ms); MCH <-> spin-expanded index maps
│   ├── molecule.py         # symbols, masses, charge, frozen/masked atoms
│   └── records.py          # Request, ElectronicResult, StepRecord, TrajectoryState (dataclasses)
├── backends/
│   ├── base.py             # ElectronicStructureBackend ABC + Capabilities
│   ├── analytic/           # Tully models, spin-boson, user-defined Python potentials
│   ├── lvc.py              # linear vibronic coupling (incl. SOC) — reads SHARC LVC.template
│   ├── pyscf_backend/      # FIRST RELEASE — in-process: tddft.py, casscf.py, soc.py, overlaps.py
│   ├── sharc_external.py   # runs ANY SHARC_*.py interface via QM.in/QM.out (validation + stopgap)
│   ├── orca_backend/       # LATER (Phase 7) — TDDFT + SOC
│   ├── gaussian_backend/   # LATER (Phase 7) — subprocess: inputgen, runner, parsers (log/fchk/rwf)
│   └── molcas_backend/     # LATER (Phase 7, on demand) — multireference + SOC + NACs
├── overlaps/
│   ├── ao_cross.py         # <χ_μ(R_t)|χ_ν(R_{t+dt})>  (via PySCF intor_cross)
│   ├── cis_overlap.py      # CIS/TDA-type state overlaps (own NumPy implementation)
│   ├── wfoverlap.py        # optional wrapper around external wfoverlap.x (validation)
│   └── tracking.py         # phase alignment, root-flip detection, Löwdin orthogonalization
├── dynamics/
│   ├── representation.py   # MCH ↔ diagonal (U matrix), gradient transformation, gradcorrect
│   ├── integrators.py      # velocity Verlet (fvv); adaptive later
│   ├── propagators.py      # exp(-iH dt) with ddr/ddt, local diabatization, NPI
│   ├── hopping.py          # Tully FSSH, SHARC-type (propagator-based), GFSH
│   ├── velocity.py         # rescaling (vel/nac/gdiff/…), frustrated-hop reflection
│   ├── decoherence.py      # EDC, A-FSSH, (DoM later)
│   ├── selection.py        # grad_select/nac_select by energy window (eselect)
│   ├── thermostat.py       # Langevin (later)
│   ├── driver.py           # the main time loop (§4.1)
│   └── restart.py          # full-state checkpoint (HDF5) incl. RNG and backend state
├── initconds/
│   ├── wigner.py           # harmonic Wigner sampling from Hessians (Gaussian fchk, PySCF hessian)
│   └── excite.py           # state selection (oscillator strength / energy window / stochastic)
├── io/
│   ├── config.py           # TOML/YAML input + pydantic schema validation
│   ├── sharc_compat/       # QM.in/QM.out reader+writer, SHARC 'input' importer, output.dat reader
│   └── h5output.py         # per-trajectory HDF5 (h5py), chunked, append-per-step
├── ensemble/
│   ├── setup.py            # directory tree / config per trajectory
│   ├── submit.py           # SLURM/PBS job arrays (template-based)
│   └── collect.py          # merge HDF5 → ensemble dataset (xarray-friendly)
├── analysis/
│   ├── populations.py      # classical/quantum/Wigner populations + bootstrap CIs
│   ├── kinetics.py         # fitting (exp/sequential models)
│   └── geometry.py         # internal coordinates, hop geometries, clustering
└── cli/                    # `gqteashmd run`, `gqteashmd setup`, `gqteashmd collect`, `gqteashmd compare-sharc`
tests/                      # mirrors package; plus validation/ (slow, statistical)
```

### 2.3 The backend contract (the most important interface)

```python
@dataclass(frozen=True)
class Capabilities:
    multiplicities: tuple[int, ...]      # e.g. (1,) or (1, 3)
    gradients: bool
    nacdr: bool                          # analytic derivative couplings
    overlaps: bool                       # <Ψ_I(t-dt)|Ψ_J(t)>
    soc: bool
    dipoles: bool
    dipole_gradients: bool = False
    incremental: bool = True             # can return extra gradients later w/o redoing SCF

@dataclass(frozen=True)
class Request:
    step: int
    coords: np.ndarray                   # (natom, 3) bohr
    states: dict[int, int]               # {multiplicity: nstates}
    energies: bool = True
    gradients: frozenset[int] = frozenset()      # MCH state indices
    nacdr: frozenset[tuple[int, int]] = frozenset()
    overlaps: bool = False
    soc: bool = False
    dipoles: bool = False

@dataclass(frozen=True)
class ElectronicResult:
    H: np.ndarray            # (nmstates, nmstates) complex, MCH basis, incl. SOC off-diagonals
    dipoles: np.ndarray | None        # (3, nmstates, nmstates)
    gradients: dict[int, np.ndarray]  # state -> (natom, 3)
    nacdr: dict[tuple[int, int], np.ndarray]
    overlaps: np.ndarray | None       # (nmstates, nmstates) S_IJ = <Ψ_I(t-dt)|Ψ_J(t)>
    phases: np.ndarray | None
    meta: dict                        # timings, convergence info, warnings

class ElectronicStructureBackend(ABC):
    capabilities: Capabilities
    def compute(self, req: Request) -> ElectronicResult: ...
    def compute_more(self, req: Request) -> ElectronicResult: ...  # extra gradients after a hop
    def save_state(self, path) -> None: ...   # orbitals/CI for restart & overlaps
    def load_state(self, path) -> None: ...
```

Key design points:

- **`compute_more`** handles the classic surface-hopping case. You only computed the active-state gradient, a hop happens, and now you need the new state's gradient at the *same* geometry. SHARC handles this through `grad_select`/`eselect` plus a second call. With PySCF in-process this is cheap because SCF/TDDFT are cached. With Gaussian it means a second job reusing the `.chk`.
- **Spin expansion lives in `core/states.py`, not in backends.** Backends return spin-free (MCH) multiplets. The driver expands triplets to M_s = −1, 0, +1, as SHARC does (`nstates` versus `nmstates`). This prevents each backend from re-implementing the expansion with its own bugs.
- **The `sharc_external` backend** writes `QM.in`, runs `SHARC_<X>.py` in a working directory, and parses `QM.out` with your own parser. It gives you SHARC's 40 interfaces, including the LVC and ANALYTICAL ones used for validation, for free and at arm's length (see §7, licensing).

### 2.4 Configuration and input

- Use a native **TOML (or YAML) input validated by `pydantic`**. That gives typed, documented, defaulted options and readable errors.
- Add a **SHARC `input` importer** (`gqteashmd import-sharc input`). It maps `surf sharc`, `coupling overlap`, `ekincorrect parallel_vel`, `decoherence_scheme edc`, and so on to your config. Every validation case can then start from the *same* SHARC input file. Write a mapping table and warn loudly on unsupported keywords instead of ignoring them.

### 2.5 Output, restart, and data

- **Per-trajectory HDF5** (one file per trajectory, appended each step, chunked datasets) covering everything in `StepRecord`. HDF5 is robust, self-describing, partially readable while running, and works with xarray/pandas.
- **Restart = one HDF5 group** with the full `TrajectoryState`: geometry, velocities, coefficients, active state, decoherence auxiliaries, RNG bit-generator state, step counter, previous-step matrices needed for interpolation. Add a pointer to the backend's saved state (`.chk`, PySCF `chkfile`, previous MO/CI vectors). Test it by checking that *restart + N steps* is bit-identical to *N continuous steps*.
- **Converter `gqteashmd export-sharc`** writes a minimal SHARC-like `output.dat` *only if needed*. Comparing via your own reader of SHARC's `output.dat`/`output.lis` is usually simpler and avoids chasing SHARC's output-version changes.

### 2.6 Execution model on clusters

- **One trajectory = one process = one job-array element.** Trajectories are independent, so don't build MPI into the core. Parallelism *within* a trajectory belongs to the backend (OpenMP/BLAS threads for PySCF, `%NProcShared` for Gaussian, and concurrent Gaussian jobs for several gradients).
- Thread hygiene: set `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and PySCF `lib.num_threads()` from one resource setting, and record them in the HDF5 metadata.
- **Do not run trajectories inside the OneDrive-synced folder.** Thousands of small, frequently rewritten scratch files (Gaussian `.rwf`/`.chk`) cause sync conflicts and slow I/O. Use local or cluster scratch, and copy results back.

---

## 3. Interfacing with the electronic-structure packages

### 3.0 Backend strategy: PySCF now, others later

| Order | Backend | Status | Main reason to add it |
|---|---|---|---|
| 1 | **PySCF** (TDA/TDDFT, SA-CASSCF, MC-PDFT via pyscf-forge) | **First release (Phases 3–4)** | In-process access to wavefunctions, Conda-installable, CI-testable, CASSCF NACs |
| — | `sharc_external` (any SHARC interface via `QM.in`/`QM.out`) | First release (Phase 2) | Validation; temporary access to ORCA/MOLCAS/Gaussian before native backends exist |
| 2 | **ORCA** | Later (Phase 7a) | TDDFT with **singlet–triplet SOC** → production ISC; free for academics |
| 3 | **Gaussian** | Later (Phase 7b) | Continuity with the group's existing Gaussian workflows/benchmarks |
| 4 | OpenMolcas | On demand | Multireference dynamics (CASPT2, SOC, analytic NACs) |
| 5 | OpenQP (MRSF-TDDFT) | On demand | Correct S1/S0 crossing topology at near-TDDFT cost |

PySCF-only also limits the first release:

- **Spin–orbit coupling:** only via LVC models or own development (§4.6).
- **System size:** PySCF TDDFT gradients are slower than ORCA/Gaussian for larger molecules. Use density fitting, and evaluate `gpu4pyscf`.
- **Methods:** no CASPT2-quality multireference.

Accept these limits explicitly for the first release. For a study that needs more before Phase 7, use the `sharc_external` backend with SHARC's ORCA/MOLCAS interfaces as a stopgap.

### 3.1 PySCF backend (in-process; the backend of the first release)

**Why in-process:** no file parsing, direct access to MO coefficients and CI/TDDFT amplitudes, and cheap AO cross-overlaps (`gto.intor_cross('int1e_ovlp', mol_prev, mol_curr)`). You can also reuse previous orbitals as the SCF guess, and request extra gradients without recomputing SCF/TDDFT.

| Capability | PySCF route | Notes |
|---|---|---|
| Ground state | `scf.RHF/RKS/UKS`, optional density fitting (`.density_fit()`) | Guess from previous step's density; stability analysis optional |
| Excited states (TDA/TDDFT) | `tddft.TDA` / `tddft.TDDFT` (singlets; triplets via `singlet=False` or spin-flip) | Store X (and Y) amplitudes for overlaps |
| Excited-state gradients | `td.nuc_grad_method().kernel(state=i)` | One call per state; cache Z-vector solvers if possible |
| SA-CASSCF energies/gradients/NACs | `mcscf.CASSCF(...).state_average_(...)`, `mc.nuc_grad_method()`, `mc.nac_method()` | This is the route SHARC_PYSCF uses; also MC-PDFT via `pyscf-forge` (`mrh`) |
| TDDFT NACs | Recent PySCF releases include TDDFT nonadiabatic-coupling modules | **Verify availability and version** in your pinned PySCF. Otherwise use overlaps (works for any method) |
| Dipoles / transition dipoles | `mf.dip_moment()`, `td.transition_dipole()` | For oscillator strengths in `excite` and laser coupling |
| Hessian (for Wigner sampling) | `pyscf.hessian` (RHF/RKS) | Or read a Gaussian `.fchk` Hessian |
| SOC | **Not a turnkey feature for TDDFT** (see §4.6) | Breit–Pauli / SOMF one-electron integrals (`int1e_pnucxp`, …) exist, but you must assemble state-pair SOC matrix elements yourself |
| GPU | `gpu4pyscf` (optional) | Worth evaluating for DFT/TDDFT throughput |

**Robustness measures:**

- Run each PySCF calculation in a guarded context: catch convergence failures, retry with level shift/damping or a fresh guess, and only then fail the trajectory with a clear status code.
- Keep a **persistent worker-process option**: driver ↔ PySCF via `multiprocessing` with a pipe. It protects the driver from native crashes or memory growth over thousands of steps, while keeping most of the in-process speed.
- Pin the PySCF version. SHARC's own interface enforces `>= 2.6.0`, and PySCF APIs do change between minor versions.

### 3.2 Gaussian backend (subprocess, file-based): **deferred to Phase 7b; design notes**

*Not part of the first release. These notes record how to build it later.* Gaussian is proprietary and cannot be installed via Conda. Treat it as an external executable (`$g16root`, `GAUSS_SCRDIR`) configured in a resources file, as SHARC's `GAUSSIAN.resources` does.

**Per-step workflow (mirrors SHARC_GAUSSIAN's proven approach):**

1. **Input generation:** `%Chk`, `%Mem`, `%NProcShared`. Route: `#P <functional>/<basis> TD(NStates=n, Root=k[, TDA]) Force NoSymm Guess=Read [SCRF] [EmpiricalDispersion]` plus any IOps. Use **`NoSymm`** and fixed orientation so states, orbitals and coordinates stay comparable between steps. Use `Guess=Read` from the previous step's `.chk`.
2. **Several gradients = several jobs.** Gaussian computes one excited-state gradient (`Root=k`) per run. Run the other roots as separate jobs from the converged `.chk`, concurrently if cores allow.
3. **Parsing:** energies and forces from the `.log` file ("Forces (Hartrees/Bohr)"; mind the **sign**, since forces = −gradient), transition dipoles from the log, MO coefficients from `formchk` → `.fchk`.
4. **Overlaps** (Gaussian 16 gives no excited–excited NACs for TDDFT, so overlaps are the coupling route):
   - CI amplitudes: `rwfdump <file>.rwf <out> 635R`, as SHARC does. This is an **undocumented binary layout** that depends on version (G16 A/B/C), TDA versus RPA, and restricted versus unrestricted. Isolate it in one parser with fixture-based tests.
   - AO cross-overlap S(R_{t−dt}, R_t): either (a) SHARC's approach, a "doubled molecule" Gaussian job printing the overlap matrix, or (b) rebuilding the basis in PySCF and calling `intor_cross`. Option (b) is faster and simpler, **but** Gaussian's 5D/7F versus 6D/10F convention and its AO ordering/normalization must match exactly. Validate (b) against (a) before trusting it.
   - State overlaps: your own CIS/TDA overlap code (§3.3), cross-checked against `wfoverlap.x`.
5. **Restart/guess handling:** keep the `.chk` (and, when overlaps are needed, the CI vectors and MOs) of step t−1 only, plus an optional periodic backup.

### 3.3 Wavefunction overlaps (written for PySCF first, reused by later backends)

- Implement **CIS/TDA-type overlaps** directly in NumPy. Truncate the determinant expansion by amplitude (SHARC's `wfthres`), use frozen-core consistently, and normalize consistently.
- For full TDDFT (RPA), using only X (or X+Y renormalized) is an *approximation*. Document which one you use, because this choice changes the computed couplings.
- **Phase/sign alignment:** after computing S, align the signs of the current wavefunctions to the previous ones (S_II > 0), detect root flips (permutation from max |S_IJ|), and Löwdin-orthogonalize S before using it in local diabatization. Log any step where the overlap matrix is far from unitary (e.g. min singular value < 0.5). That usually signals an intruder state or too small a basis of states.

### 3.4 Electronic-structure method validity

Linear-response TDDFT has the wrong dimensionality of the S1/S0 intersection seam. Treat S1→S0 hops in TDDFT as a known methodological limitation, and offer SHARC-like handling (`force_hop_to_gs` below an energy gap, or stopping the trajectory). CASSCF via PySCF is the correct choice when S1/S0 crossings are central. Later, OpenQP's MRSF-TDDFT is an alternative.

### 3.5 Adding further backends later

The first release must make adding a backend a self-contained task. Keep these rules from the start:

1. **The dynamics core never imports a backend module.** Backends register by name (entry points or a registry dict) and are selected in the config.
2. **Backends report what they can do through `Capabilities`.** The driver checks requested features against it at start-up and fails early with a clear message (e.g. "SOC requested but backend 'pyscf-tddft' has soc=False").
3. **Shared pieces are backend-independent.** These include `overlaps/` (AO cross-overlaps, CIS overlaps, phase tracking), spin expansion (`core/states.py`) and unit handling. A new backend only supplies MO coefficients, CI amplitudes and basis information in a common format (`WavefunctionData` dataclass).
4. **Every backend goes through the same conformance test suite** (`tests/backends/conformance.py`):
   - gradients vs finite differences;
   - overlaps at Δt → 0 are ≈ identity;
   - phase consistency;
   - restart round-trip;
   - single-point agreement with the corresponding SHARC interface (§5.3).

   A backend counts as "done" only when it passes the suite.
5. **The `sharc_external` backend is the template and fallback.** Before writing a native ORCA/Gaussian backend, run the same molecule through `sharc_external` with SHARC's interface. That gives the reference data the native backend must reproduce.

Estimated effort per additional native backend once the core is mature:

- ORCA: about 4–6 weeks, including SOC.
- Gaussian: about 6–8 weeks, mostly the overlap pipeline (§3.2).
- OpenMolcas: about 6–8 weeks.

---

## 4. Key surface-hopping and nuclear-dynamics components

### 4.1 Main loop (per nuclear step Δt)

```
given: R(t), v(t), a(t), c(t), active state λ, previous electronic data E(t), U(t), S, ...

1.  R(t+Δt) = R + vΔt + ½aΔt²                          # velocity-Verlet position update
2.  result  = backend.compute(Request(R(t+Δt), ...))    # energies, (selected) grads, NACs/overlaps, SOC, DM
3.  track phases / root flips; build H_MCH(t+Δt) (+SOC), dipoles
4.  diagonalize H_MCH -> U(t+Δt), H_diag; align phases of U to U(t)          (representation.py)
5.  propagate c from t to t+Δt with N substeps, interpolating H and couplings  (propagators.py)
6.  compute hopping probabilities from the propagator / density matrix        (hopping.py)
7.  draw ξ ~ U(0,1) [logged]; select new state; if hop: energy check ->
       rescale velocities along chosen direction, or frustrated -> reflect/ignore   (velocity.py)
       if gradient of new state missing -> backend.compute_more(...)
8.  apply decoherence correction (EDC / A-FSSH)                                (decoherence.py)
9.  gradient of active state in propagation representation (+gradcorrect terms)
10. v(t+Δt) = v + ½(a(t) + a(t+Δt))Δt                   # velocity-Verlet velocity update
11. checks: total-energy drift, norm of c, overlap sanity; thermostat; output; restart
```

The order of steps 6–10 (hop *before* the velocity half-step, and when to recompute gradients) differs between codes. **Clean-room note (P7a-1):** SHARC's source is not read (never copy or adapt it). The order used is **our documented step order**, written in USER_MANUAL.md §8.12 and the P7a-1 plan's Method reference: positions, evaluate, velocity completed with the old active gradient, electronic propagation, hopping probabilities and one random number, hop with the velocity adjustment at fixed positions, decoherence, new active gradient. **Differences from SHARC are measured by black-box comparison** (executing SHARC and comparing its output; USER_MANUAL.md §8.14.1(c)), e.g. the end-point substep interpolation and the EDC convention found that way. Otherwise trajectory-level comparisons could diverge for trivial reasons.

### 4.2 Representation handling (MCH versus diagonal): design for it from day 1

- Without SOC or fields, diagonal = MCH, and everything reduces to standard FSSH.
- With SOC, SHARC propagates and hops in the **diagonal** basis (eigenstates of H_MCH + H_SOC). This requires:
  - U(t) with phase continuity between steps (otherwise there are spurious couplings).
  - Transforming gradients to the diagonal basis. That transform needs NAC-type terms (`gradcorrect`), or approximations (`kmatrix`, `tdm`, `ngt`, `enac` in SHARC 4.1) when NACs aren't available, as with overlap-only Gaussian runs.
  - Hopping and rescaling expressed in the diagonal basis, including *spin-mixed* states where an "energy-difference-gradient" direction must be built in the diagonal basis.
- Even if SOC comes late, keep `U` in the data model and in the loop from the start. Retrofitting it is a common cause of rewrites.

### 4.3 Electronic propagation

Implement in this order:

1. **Exponential propagator with ddr (NAC·v) couplings**, linear interpolation of H and couplings over substeps. Simplest; needs NACs (CASSCF/PySCF, analytic models).
2. **Local diabatization** (Granucci, Persico & Toniolo, JCP 2001) using the overlap matrix S. It is robust for trivially unavoided crossings and is SHARC's default for `coupling overlap`. It is needed for PySCF TDDFT if analytic TDDFT NACs are unavailable, and is mandatory for a later Gaussian backend.
3. **NPI** (Meek & Levine, JPCL 2014) as an alternative overlap-based time-derivative coupling.
4. Later: curvature-driven couplings (`ktdc`, energy-only, so no NACs and no overlaps). They are cheap and useful when overlaps are unreliable.

Unit tests: norm conservation to machine precision, time-reversibility for the exponential propagator, exact two-level Rabi oscillations, and the Landau–Zener limit.

### 4.4 Hopping, velocity adjustment, frustrated hops

- **Hopping probabilities:** standard Tully FSSH from the density matrix and couplings, and SHARC's propagator-based probability (Mai, Marquetand & González, IJQC 2015), which works with any representation and with local diabatization. GFSH is optional.
- **Velocity adjustment:** along v (`parallel_vel`, the simplest and SHARC's common default), along NAC (`parallel_nac`, needs NACs), along the gradient difference (`parallel_diff`, the overlap-only fallback), and their "projected" variants that remove translation/rotation. All of them must conserve total energy exactly and, for the projected variants, linear and angular momentum. Write property tests for that.
- **Frustrated hops:** none / reflect along the chosen vector (`reflect_frustrated`). Log every frustrated event: it is a key diagnostic and a key comparison quantity against SHARC.
- **Energy-window selection** (`eselect`): request gradients/NACs only for states within ΔE of the active state. This is the main lever on Gaussian cost.

### 4.5 Decoherence

- **EDC** (energy-based decoherence; Granucci & Persico, JCP 2007), with parameter C = 0.1 Eh as in SHARC's `decoherence_param`. Simple, robust, and the most common in SHARC studies.
- **A-FSSH** (Jain, Alguire & Subotnik, JCTC 2016): auxiliary position/momentum moments plus collapse/reset. Higher payoff but more state and more bugs. Implement only after EDC is validated.
- Decoherence on/off must be a toggle in every validation study. Agreement *with* EDC but not without it (or vice versa) localizes bugs quickly.

### 4.6 Spin–orbit coupling: the strategic gap

You need SOC matrix elements between singlet and triplet *states*, consistent with the states (and phases) used for dynamics:

- **Option A (recommended start): use LVC models.** SHARC's `SHARC_LVC.py` and `setup_LVCparam.py`/`create_LVCparam.py` parametrize LVC models, including SOC, from a few ab initio calculations. Implementing LVC natively in your code gives SOC-capable dynamics immediately, cheaply, and in a form directly comparable to SHARC.
- **Option B: SOC for TDDFT/TDA in PySCF.** Assemble ⟨S_i|Ĥ_SO|T_j,M_s⟩ from TDA amplitudes using a one-electron Breit–Pauli/SOMF operator (effective-charge or mean-field two-electron treatment). This is a well-defined but real project: about 1–2 months, including validation against ORCA or another established SOC-TDDFT implementation on small molecules (e.g. thioformaldehyde, thiophene, nucleobases). Community codes exist and should be evaluated before writing your own.
- **Option C: SOC from a different code.** Short term, use the `sharc_external` backend with SHARC's ORCA/MOLCAS interfaces. Long term, a **native ORCA backend** (Phase 7a) is the most direct production route to TDDFT-level ISC. Keep PySCF for singlet-only and CASSCF work.
- **Gaussian cannot give you TDDFT SOC matrix elements in a usable way**, so adding Gaussian later does not solve the SOC gap.

### 4.7 Nuclear dynamics extras (by priority)

1. Velocity Verlet with fixed Δt (SHARC default `fvv`). Total-energy monitoring plus a `killafter`-style stop.
2. Atom masks and frozen atoms (needed for any QM/MM-like or solvated work later).
3. Removal of translation/rotation (`remove_trans_rot`).
4. Langevin thermostat with regions (for condensed-phase work).
5. RATTLE constraints, adaptive step size, ZPE corrections, laser fields: only if research needs demand them.

### 4.8 Initial conditions and excitation

- **Wigner sampling** from harmonic frequencies and normal modes, from a Gaussian `.fchk`/`freq` log or a PySCF Hessian. Pay attention to mass-weighting, linear molecules, and imaginary frequencies, and handle low-frequency modes the way `wigner.py` does (temperature option). This is an area where reading SHARC's `wigner.py` *behaviour* (not copying code) is useful.
- **Excitation selection** (`excite.py` equivalent): compute vertical excitation energies and oscillator strengths for each initial geometry. Select the starting state stochastically with acceptance w/w_max over the in-window pairs, where w = f/ΔE² (SHARC 4.1 `excite.py`-compatible; measured in P6 by black-box maximum likelihood: ΔE exponent −1.99 ± 0.05, f exponent 0.99 ± 0.03) or w = f (the oscillator-strength convention of Barbatti et al. 2007); optionally within an energy window, or delta-pulse style, and write the chosen state and coefficients into each trajectory's config.
- Compare your sampled distributions (bond-length, kinetic-energy, and excitation-energy histograms) against `wigner.py` + `excite.py` output from the *same* frequency file before running any dynamics.

---

## 5. Testing and validation against SHARC

### 5.1 Test pyramid

| Level | What | Tools | Runtime |
|---|---|---|---|
| Unit | Pure functions: unit conversions, propagators, rescaling, phase alignment, state index maps, parsers (Gaussian log/fchk/rwf fixtures; SHARC `QM.out` fixtures from `tests/interface/inputs`) | `pytest`, `hypothesis` (property tests: energy and momentum conservation, unitarity, norm) | seconds |
| Component | Backend vs finite differences: gradients vs FD energies; NACs vs FD overlaps; overlap matrix unitarity at small Δt | `pytest -m slowish` | minutes |
| Cross-code single point | Same `QM.in` → your backend vs `SHARC_PYSCF.py` / `SHARC_GAUSSIAN.py` | `sharc_external` + comparison script | minutes |
| Short-trajectory | 10–50 fs, same initial conditions, comparison per step | `gqteashmd compare-sharc` | min–hours |
| Ensemble/statistical | Populations, hop times, branching ratios with confidence intervals | job arrays + `analysis/` | hours–days |
| Regression | SHARC's own `tests/INPUT/PYSCF_ddr`, `GAUSSIAN_overlap`, `LVC_*`, `ANALYTICAL_overlap` inputs | CI (cheap ones) + nightly (ab initio) | varies |

### 5.2 Model-system benchmarks (no electronic-structure noise)

1. **Tully's three 1D models** (simple avoided crossing, dual avoided crossing, extended coupling with reflection). Reproduce transmission/reflection probabilities versus incoming momentum, against the published FSSH curves and against exact wavepacket results. Use ≥ 2000 trajectories per momentum. These catch most propagator, hopping and rescaling bugs.
2. **SHARC's `SHARC_ANALYTICAL` interface** (`tests/INPUT/ANALYTICAL_overlap`, `examples/SHARC_ANALYTICAL`). Run the same analytic potential in SHARC and in your code.
3. **LVC models** (`tests/INPUT/LVC_overlap`, `LVC_pysharc`, `examples/SHARC_LVC`, plus the LVC in the SHARC tutorial). These are cheap, multi-dimensional, and include SOC, so they are the **primary SHARC-equivalence benchmark** for the diagonal representation, gradcorrect, phase tracking, and ISC.

### 5.3 Electronic-structure equivalence (single geometries)

Compare on the geometries in `tests/INPUT/PYSCF_ddr/geom` and `GAUSSIAN_overlap/geom`, plus 2–3 of your group's molecules:

| Quantity | Target tolerance (same method/basis/grid) |
|---|---|
| Absolute energies | ≤ 1 × 10⁻⁷ Eh (PySCF↔PySCF); ≤ 1 × 10⁻⁶ Eh (Gaussian runs, same settings) |
| Excitation energies | ≤ 1 × 10⁻⁵ Eh |
| Gradients | ≤ 1 × 10⁻⁵ Eh/bohr (max abs component) |
| NACs | ≤ 1 × 10⁻⁴ relative, **up to a global sign per state pair** |
| Overlap matrix | ≤ 1 × 10⁻⁴ per element, **up to state signs**; same determinant-threshold settings |
| Transition dipoles | ≤ 1 × 10⁻⁴ a.u., up to sign |

Tolerances are starting points. Tighten them for PySCF↔PySCF (same library) and loosen them only with a documented reason (e.g. different DFT grids).

### 5.4 Trajectory-level comparison (dynamics equivalence)

Surface-hopping trajectories are chaotic and stochastic, so per-trajectory agreement only holds for a short time and only under controlled conditions:

1. **Hopping off** (`hopping_procedure off` in SHARC): compare R(t), v(t), E(t) and |c(t)|² step by step for ~20–50 fs. The difference should grow from ~10⁻¹⁰ slowly. A jump at a specific step points to a specific component.
2. **Hopping on with injected random numbers.** If SHARC's high-printlevel `output.log` records the per-step random number and hop probabilities (check `printlevel` 4–5 output in your build), feed the same sequence through `replay` mode. Hops should then occur at identical steps until numerical noise flips a decision.
3. **Swap tests via the file protocol:**
   - (a) SHARC driver + SHARC interface,
   - (b) **your driver + SHARC interface** (via `sharc_external`),
   - (c) **SHARC driver + your backend** (wrapped as a `QM.in`→`QM.out` script),
   - (d) your driver + your backend.

   (a)≈(b) validates your dynamics. (a)≈(c) validates your electronic structure. This 2×2 design is the single most effective debugging tool in this plan.

### 5.5 Ensemble/statistical equivalence

- Compare **populations versus time** with bootstrap 95 % confidence intervals. "Agreement" means the difference curve is consistent with zero at every time.
- Compare **lifetimes** from identical kinetic models (e.g. mono-/bi-exponential or sequential fits), with fit uncertainties. Also compare **hop-time and hop-geometry distributions** (two-sample Kolmogorov–Smirnov or Anderson–Darling), **frustrated-hop counts**, and **total-energy drift distributions**.
- Ensemble size: population errors scale as √(p(1−p)/N). N = 100 gives about ±0.05 at p = 0.5, which is too coarse for subtle bugs. Use N ≥ 1000 on LVC/analytic models; for ab initio, N ≈ 100–200 is usually the realistic maximum.

### 5.6 Continuous integration

- GitHub/GitLab CI on Linux: unit tests plus Tully models with small ensembles (fast, fixed seeds) on every push.
- Nightly or weekly runs on a group workstation or cluster: PySCF regression and LVC ensemble comparison. Gaussian tests run only where licensed. Mark them `@pytest.mark.gaussian` and skip automatically when `g16` is absent.
- Store SHARC reference outputs (generated once, versioned with the SHARC version and command lines) in `tests/reference/sharc-4.1/`, so CI doesn't depend on a SHARC install.

---

## 6. Incremental development roadmap

Effort estimates assume **one experienced PhD-/postdoc-level developer full-time, plus part-time review** from a second group member. They are rough guides, not commitments.

| Phase | Deliverables | Exit criterion (milestone) | Est. |
|---|---|---|---|
| **0. Foundations** | Repo, Conda env + lock file, CI, `core/` data model & units, config schema, `QM.in`/`QM.out` reader/writer, HDF5 writer, logging | Round-trip of SHARC `QM.out` fixtures; CI green | 3–4 wk |
| **1. Minimal FSSH on models** | Analytic backend (Tully models), velocity Verlet, exponential propagator (ddr), Tully hopping, `parallel_vel`/`parallel_nac` rescaling, frustrated-hop handling, RNG/replay, restart | Tully 1990 figures reproduced within statistical error; restart bit-identical | 4–6 wk |
| **2. SHARC dynamics core on LVC** | LVC backend (reads SHARC `LVC.template`), spin expansion, SOC, diagonal representation + U-phase tracking, gradcorrect, local diabatization (overlap coupling), EDC decoherence, `eselect`, `sharc_external` backend | LVC ensemble populations (with and without SOC) match SHARC within CIs; swap test (a)≈(b) | 6–8 wk |
| **3. PySCF backend: core** | SCF/TDA/TDDFT singlets, gradients, AO cross-overlaps, CIS overlaps, phase/root tracking, SA-CASSCF energies/gradients/NACs; failure recovery; backend conformance suite (§3.5) | Single-point tolerances (§5.3) vs `SHARC_PYSCF.py`; `PYSCF_ddr` regression; 100-traj ensemble on a small molecule vs SHARC | 6–8 wk |
| **4. PySCF backend: production hardening** | Triplets (closed-shell reference and spin-flip/UKS options), TDDFT NACs if available in the pinned PySCF (else overlap route), MC-PDFT via pyscf-forge, density fitting, optional `gpu4pyscf`, worker-process isolation, performance profiling | 1–2 ps trajectories on a group-relevant molecule without crashes or memory growth; ≥ 95 % of trajectories finish; cost per step documented | 4–6 wk |
| **5. Production workflow** | Wigner sampling (PySCF Hessian), excitation selection, ensemble setup, SLURM templates, collection, populations/kinetics/geometry analysis, docs + tutorial | One complete group-relevant study reproduced end-to-end, compared with SHARC on the same initial conditions | 4–6 wk |
| **6. ISC capability (PySCF route)** | Option A (LVC parametrization from PySCF) and/or Option B (PySCF TDA SOC), SOC-aware rescaling in diagonal basis | SOC matrix elements vs reference code; ISC rates on a benchmark (e.g. thiophene/SO₂-type systems) vs SHARC | 6–10 wk |
| **7. Additional electronic-structure backends** | 7a **ORCA** (TDDFT + SOC); 7b **Gaussian** (§3.2 design notes); 7c OpenMolcas / OpenQP on demand. Each one passes the conformance suite and matches its SHARC interface | Per backend: single-point agreement with `SHARC_<X>.py`; swap test (a)≈(c); one short ensemble vs SHARC | 4–8 wk each |
| **8. Dynamics extensions (on demand)** | A-FSSH, NPI, ktdc, Langevin thermostat, atom masks, adaptive Δt, laser fields, ZPE corrections, ML potentials, QM/MM | Each feature with its own SHARC comparison test | ongoing |

**Total to a usable PySCF-based research tool (phases 0–5): about 7–10 months. Adding ISC (phase 6): about 9–12 months. Each further backend (phase 7): 1–2 months.** If ISC is urgent, consider doing **7a (ORCA) before 6**: a native ORCA backend is likely faster to deliver than PySCF TDDFT-SOC.

**Sequencing rationale:**

- Phases 1–2 remove all electronic-structure noise, so the dynamics core is validated *before* any ab initio backend exists.
- Focusing on a single backend (PySCF) in phases 3–5 keeps the first release small and fully testable in CI.
- Phase 2's `sharc_external` backend means you can run production-like ab initio dynamics with your own driver (through SHARC's ORCA/MOLCAS/Gaussian interfaces) even before any native backend beyond PySCF exists.

---

## 7. Environment and package management (Conda)

Conda is a good choice. Recommendations:

- **Use conda-forge only** (as SHARC 4.1 does), via `mamba`/`micromamba` for speed. Keep a hand-written `environment.yml` *and* a generated **lock file** (`conda-lock`) per platform, so cluster and workstation environments are reproducible.
- **Platform:** PySCF is built for Linux/macOS (no native Windows builds), and Gaussian runs on Linux. **Develop and run inside WSL2 (your Oracle Linux 9.5 instance) or on the cluster**, not in native Windows Python. VS Code's WSL remote works well for this.
- **The first release needs only Conda packages** (PySCF). Later backends (ORCA, Gaussian) are outside Conda: configure their install paths (`orcadir`, `g16root`) and scratch locations in a resources file, detect them at runtime, and skip their tests when they are absent.
- **Install your own package editable** (`pip install -e .` with `pyproject.toml`) into the Conda env. Don't publish to conda-forge until the API stabilizes.
- **BLAS:** keep a single BLAS (OpenBLAS or MKL) consistent with PySCF's build. Mixing them causes subtle threading and performance problems.
- **Keep a separate env for SHARC 4.1** (its `conda_env_sharc4.1.yml` pins Python 3.12, and its `_compat` variant pins NumPy 2.2/SciPy 1.15), so that comparison runs aren't perturbed by your env's upgrades.

Starting `environment.yml`:

```yaml
name: gqteashmd
channels: [conda-forge]
dependencies:
  - python=3.12
  - numpy
  - scipy
  - h5py
  - pyscf            # pin exact version after Phase 3 starts, e.g. pyscf=2.x.y
  - pydantic>=2
  - tomli-w
  - numba            # optional, for analytic/LVC kernels
  - xarray
  - matplotlib
  - pytest
  - hypothesis
  - pytest-xdist
  - ruff
  - mypy
  - pip
  - pip:
      - -e .
```

---

## 8. Major technical risks

| # | Risk | Likelihood / impact | Mitigation |
|---|---|---|---|
| **R1** | **No production SOC path with PySCF** (SHARC's PySCF interface is singlet-only; PySCF has no turnkey TDDFT state SOC) | High / High if ISC is a research goal | Decide early whether ISC is required. Use LVC first (Option A); budget phase 6 for TDA-SOC in PySCF, or bring the ORCA backend (7a) forward; `sharc_external` as stopgap |
| **R2** | **PySCF-only limits:** cost of TDDFT gradients for larger molecules; TDDFT NAC availability depends on PySCF version; no CASPT2-level methods | Medium / Medium | Density fitting, `gpu4pyscf`, `eselect` to limit gradients; overlap-based couplings as fallback; additional backends in phase 7 |
| **R2b** | *(Phase 7b only)* **Gaussian overlaps rely on undocumented RWF record 635R** (layout varies by version and TDA/RPA/unrestricted) | Medium / High | One isolated parser; fixture tests per Gaussian revision; cross-check against SHARC_GAUSSIAN + wfoverlap; pin the Gaussian revision per project |
| **R2c** | **Backend lock-in:** PySCF-specific assumptions leak into the core, making later backends expensive | Medium / High | Rules in §3.5; a second "backend" (analytic/LVC + `sharc_external`) exists from Phase 2, so the interface is exercised by more than one implementation from the start |
| **R3** | **Phase, sign and root-flip errors** (in overlaps, NACs, U matrix) cause spurious hops that look "physical" | High / High | Dedicated `tracking.py` with diagnostics; unit tests with synthetic permuted/sign-flipped states; log the S-matrix condition; LVC benchmarks with SOC |
| **R4** | **Representation subtleties** (diagonal basis, gradcorrect without NACs, triplet M_s expansion) | High / Medium–High | Design `U` into the data model from day 1; validate on LVC + SOC against SHARC before any ab initio work |
| **R5** | **Statistical validation is weak at ab initio ensemble sizes**: subtle bugs hide in error bars | High / Medium | Do the heavy statistical validation on LVC/analytic models (N ≥ 1000); for ab initio rely on deterministic short-time and swap tests |
| **R6** | **Electronic-structure failures mid-trajectory** (SCF non-convergence, TDDFT instabilities near S1/S0, intruder states) | High / Medium | Retry ladder; explicit trajectory status codes; `force_hop_to_gs`-style handling; report and exclude rules decided *before* analysis (avoid selection bias) |
| **R7** | **TDDFT methodological limits** (S1/S0 seam topology; RPA overlap approximations) | Certain / Medium | Document clearly; use CASSCF/MC-PDFT (PySCF) for S1→S0 studies; record which approximation each run used |
| **R8** | **Scope creep toward full SHARC parity** | High / High | Freeze a feature list per phase; every new feature needs a research use case *and* a validation test |
| **R9** | **GPL-3 licensing**: all SHARC sources (including `wfoverlap`) are GPL-3; copying or translating code makes your program a derivative work | Medium / High (legal) | Either (a) adopt GPL-3 yourselves (simplest if you ever want to reuse code), or (b) clean-room implementation from the literature, using SHARC only as an external black box via files/processes. Decide **before** writing code |
| **R10** | **PySCF API drift and native-library instability** over long runs | Medium / Medium | Pin versions; optional worker-process isolation; regression tests on each upgrade |
| **R11** | **Unit/constant mismatches** cause slow drift versus SHARC (amu, Bohr, fs ↔ a.u.) | Medium / Low–Medium | Single constants module; explicit comparison against SHARC's `constants.py` values; "SHARC-constants" mode for comparisons |
| **R12** | **Infrastructure:** OneDrive-synced working directory, Windows vs Linux paths, cluster scratch quotas | Medium / Medium | Run on WSL/cluster scratch only; keep OneDrive for documents and code (git), not trajectory data |
| **R13** | **Developer continuity** in an academic group (students leave) | High / High | Typed, documented, tested code; ADRs for key decisions; tutorials; code review as a group habit |

---

## 9. Assumptions

1. **"Comparable in scope to SHARC 4.1"** means the core TSH feature set: MCH/diagonal representations, SOC-ready design, ddr and overlap couplings, local diabatization, SHARC-type hopping, the main rescaling variants, EDC (later A-FSSH), Wigner initial conditions, excitation selection, ensemble tooling, and analysis. It does *not* mean SHARC's ~40 interfaces, laser fields, ZPE corrections, QM/MM, ML potentials, or SCP/Ehrenfest/CSDM in the first release.
2. **Target systems** are small to medium organic or main-group molecules (≤ ~50 atoms in the QM region), with typical dynamics of 0.1–1 ps at Δt ≈ 0.5 fs, and ensembles of 50–500 trajectories.
3. **Methods (first release)** are TDA/TDDFT and SA-CASSCF/MC-PDFT, **all via PySCF**, singlets first, with triplets/ISC as a second-stage goal. Other electronic-structure programs come later (phase 7).
4. **For the later Gaussian backend:** Gaussian 16 (revision to be pinned) is licensed and available on the group's Linux cluster. Gaussian provides analytic excited-state gradients but no usable excited–excited NACs or SOC for TDDFT in that version, which is consistent with how SHARC_GAUSSIAN works. ORCA (if adopted) is licensed per user under its academic licence.
5. **PySCF ≥ 2.6** from conda-forge. Availability of TDDFT NAC modules depends on the pinned version and must be checked (§3.1).
6. **Compute:** a Linux cluster with a batch scheduler (SLURM assumed) for ensembles, plus WSL2 for development.
7. **Development language:** Python 3.12. The group is comfortable with Python and Git. Fortran/C++ is not required.
8. **SHARC 4.1 remains installed** (WSL) as the reference implementation and is used as a black box (input files, executables, outputs) during validation.
9. **Licensing:** the group has not yet chosen a licence. The recommendation in R9 depends on that choice.
10. **Effort estimates** assume one dedicated developer and exclude time for science production runs.

---

## 10. Immediate next steps (first 2–3 weeks)

1. **Decide two things now:** licence (GPL-3 vs clean-room/permissive, R9) and whether ISC is a must-have (R1). Both shape the architecture and the roadmap.
2. **Create the environment** in WSL (`micromamba create -f environment.yml`), the repo skeleton (§2.2), and CI.
3. **Generate the SHARC reference set**, versioned, from the SHARC 4.1 install:
   - run `tests/INPUT/{ANALYTICAL_overlap, LVC_overlap, LVC_pysharc, PYSCF_ddr}` and archive their outputs (`GAUSSIAN_overlap` and `ORCA_overlap` can wait until phase 7);
   - run a 1000-trajectory LVC ensemble (with and without SOC) and archive the populations;
   - produce `QM.in`/`QM.out` pairs from `SHARC_PYSCF.py` at 3–5 geometries.
4. **Write the data model and `QM.out` parser first.** Everything else plugs into it, and it is immediately testable against the `tests/interface/inputs` fixtures.
5. **Read (not copy) these SHARC sources** to pin down the step order and conventions you must match in compatibility mode: `source/main.F90`, `nuclear.F90`, `electronic.f90`, `tsh_tu.f90`, `decoherence_afssh.f90`, `lib/SHARC_INTERFACE.py` (request logic), `bin/SHARC_PYSCF.py` (PySCF calls and conventions). Leave `bin/SHARC_GAUSSIAN.py` / `SHARC_ORCA.py` for phase 7. Also read `doc/SHARC_Manual.pdf` (dynamics theory chapter) and `doc/SHARC_interface_development_guide.pdf`.

---

## 11. Key references (for the methods to implement)

- J. C. Tully, *J. Chem. Phys.* **93**, 1061 (1990): fewest-switches surface hopping; model problems.
- G. Granucci, M. Persico, A. Toniolo, *J. Chem. Phys.* **114**, 10608 (2001): local diabatization.
- G. Granucci, M. Persico, *J. Chem. Phys.* **126**, 134114 (2007): energy-based decoherence (EDC).
- G. A. Meek, B. G. Levine, *J. Phys. Chem. Lett.* **5**, 2351 (2014): norm-preserving interpolation.
- A. Jain, E. Alguire, J. E. Subotnik, *J. Chem. Theory Comput.* **12**, 5256 (2016): A-FSSH.
- S. Mai, P. Marquetand, L. González, *Int. J. Quantum Chem.* **115**, 1215 (2015): SHARC hopping in arbitrary representations.
- S. Mai, P. Marquetand, L. González, *WIREs Comput. Mol. Sci.* **8**, e1370 (2018): SHARC overview.
- F. Plasser et al., *J. Chem. Theory Comput.* **12**, 1207 (2016): efficient wavefunction overlaps (wfoverlap).
- SHARC 4.1 manual and interface development guide: `sharc4-4.1/doc/`.

*(Check citation details against the SHARC bibliography in `sharc4-4.1/doc/SHARC_papers.bib` before you cite them in group documents.)*
