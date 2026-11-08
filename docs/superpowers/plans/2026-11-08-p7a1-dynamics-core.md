# P7a-1: Surface-Hopping Dynamics Core on Analytic Models — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pure-Python trajectory surface-hopping core in `gqteashmd.dynamics`, validated on Tully's three one-dimensional models. It has these parts:
- a velocity-Verlet integrator;
- an exponential electronic propagator with NAC·v couplings and substeps;
- Tully FSSH and SHARC-type hopping;
- `parallel_vel`/`parallel_nac` velocity adjustment, with frustrated hops kept or reflected;
- EDC decoherence;
- logged and replayable random numbers;
- per-step records and bit-identical restarts;
- an ensemble runner with the command `gqteashmd model-dynamics`.

It is validated against an exact wavepacket solver, published values and SHARC 4.1 run as a black box.

**Architecture:**
- **No dependencies on the workflow.** `gqteashmd.dynamics` imports nothing from `electronic/`, `stages/`, the orchestrator or `config/`. It takes its electronic data through a small protocol (`ElectronicModel.evaluate`), which the analytic models implement now and a PySCF adapter will implement in P7b.
- **Strategy objects.** Every algorithmic choice is a small strategy selected by name in `DynamicsOptions`, following strategy §2.1.
- **Ready for spin–orbit coupling.** The data model carries the complex MCH Hamiltonian and the MCH→diagonal matrix U from day 1 (strategy §4.2). In P7a-1, U must be diagonal up to phases, i.e. there is no SOC; non-diagonal U raises `NotImplementedError` pointing to P7a-2.

**Tech Stack:** Python 3.12, NumPy, SciPy (tests: `scipy.stats`, `scipy.linalg.expm` as a reference), pydantic (runner input only), pytest, ruff 0.17.4, mypy.

**Spec:**
- Master plan [`2026-11-06-gqteashmd-master-plan.md`](2026-11-06-gqteashmd-master-plan.md): §9 P7a, O6 and O7, and §7 (the `dynamics` validation row).
- Strategy [`SHMD_implementation_strategy.md`](../../../SHMD_implementation_strategy.md):
  - §2.1 (strategy objects, atomic units, StepRecord, determinism and replay);
  - §2.5 (restart is bit-identical);
  - §4.1–§4.5 (main loop, representations, propagation, hopping, velocity adjustment, decoherence);
  - §5.1–§5.2 (unit and property tests, Tully models);
  - §6, phase 1.

**Owner decisions (2026-11-08):**
1. **Scope split.** P7a becomes **P7a-1** (this plan: the core on analytic models) followed by **P7a-2**: the LVC backend, overlap-based local diabatization, triplet spin expansion, SOC with the diagonal representation and gradient corrections, and the SHARC LVC comparisons.
2. **O6, first-release options.**

   | Option | Values |
   |---|---|
   | Hopping | `"sharc"` (propagator-based, the default) and `"fssh"` (Tully) |
   | Velocity adjustment (`ekin_correction`) | `"parallel_vel"` (default) and `"parallel_nac"` |
   | Frustrated hops | `"keep"` (default) and `"reflect"` |
   | Decoherence | `"edc"` (default, C = 0.1 Eh) and `"none"` |
   | Integrator | velocity Verlet, fixed step |
   | Coupling | `"nacdr"` only; `"overlap"` comes in P7a-2 |

   Deferred: A-FSSH, NPI, GFSH, `parallel_diff` and thermostats.
3. **Validation, all three.**
   - Our own exact 1D split-operator wavepacket solver.
   - Published values, cited.
   - SHARC 4.1 run as a black box on the same analytic model, with its input formats taken only from the **public SHARC manual** (user documentation, never source code).
4. **O7, SOC.** Designed into the data model now and built in P7a-2.

## Global Constraints

- **Clean room.** Never read, open, grep or copy anything under `/home/ajc/sharc4-4.1` (SHARC), the SHARC conda environment, or the ORCA install. SHARC may only be *executed*, from `tools/sharc_compare/`, and its output files inspected.
  - SHARC's **input formats** may be learned only from the public SHARC manual and tutorial (https://sharc-md.org), which are user documentation.
  - The methods are implemented from the literature:
    - Tully, *J. Chem. Phys.* **93**, 1061 (1990): FSSH and the three models;
    - Mai, Marquetand & González, *Int. J. Quantum Chem.* **115**, 1215 (2015): propagator-based hopping;
    - Granucci & Persico, *J. Chem. Phys.* **126**, 134114 (2007): EDC.
  - Strategy §4.1's suggestion to read SHARC's Fortran source is **overridden** by this rule. Our step order is our own and documented (Task 7).
  - `gqteashmd` never imports from `tools/`.
- **Environment.** Linux only. From PowerShell:
  `wsl -d OracleLinux_9_5 -- bash -lc 'cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && conda activate gqteashmd-dev && <command>'`.
  - For anything that uses a scratch directory or long quoting, write a `.sh` script and run it with `wsl -d OracleLinux_9_5 -- bash /path/script.sh`.
  - Check `pwd` before writing, and delete scratch by its exact path.
  - Run one pytest process at a time, and leave no background shells or wait loops running.
- **Dependencies.** No new runtime dependencies. `gqteashmd.dynamics` uses NumPy only (SciPy only in tests and tools). `import gqteashmd.cli` must not import pyscf, geometric or h5py.
- **Units.** Hartree atomic units throughout the dynamics core:
  - positions in bohr, masses in electron masses, time in ħ/Eh (1 a.u. = 0.024188843265857 fs), velocities in bohr per a.u. of time, energies in Eh, NACs in 1/bohr;
  - ħ = 1.

  The model runner's input is also in atomic units (keys end in `_au`).
- **Coordinates.** The core works on a flat coordinate vector `q` of length `f` (degrees of freedom), with masses `m` of length `f`. A molecule has f = 3N with masses repeated three times; P7b adapts to that.
- **Determinism.** One random source per trajectory, `RandomSource`, drawing exactly **one** uniform number per nuclear step, used or not, logged in the step record and stored in the restart.
  - Ensemble trajectory *i* uses `np.random.default_rng(SeedSequence([seed, zlib.crc32(b"model-dynamics")]).spawn(i + 1)[i])`, the D12 pattern.
  - The sampling of initial conditions uses a separate stream, `crc32(b"model-initial")`.
- **Errors.** Invalid options give `ValueError` with the option name (the runner's CLI maps input problems to `ConfigError`, exit 2). Physics endings such as energy drift are trajectory **outcomes**, not exceptions.
- **Style.** ruff 0.17.4, line length 100, rules E, F, W, I, B, UP and SIM. mypy must stay clean on `src/`.
- **Git.** Branch `feat/p7a1-dynamics-core`, WSL Git only, hooks on, never `--no-verify`. Conventional subjects; each commit ends with the `Co-Authored-By:` trailer of the model that wrote it. Do not push.
- **Documentation.** The last task updates CLAUDE.md, USER_MANUAL.md, the master plan and the strategy's step-order note. Mark [IMPLEMENTED] and [PLANNED], and define every symbol.

## Method reference (binding definitions; every task uses these)

Notation:
- n is the number of electronic states and f the number of degrees of freedom.
- c ∈ ℂⁿ are the electronic amplitudes in the **propagation basis**: the diagonal basis, which in P7a-1 equals the adiabatic/MCH basis.
- λ is the active state, E_k(q) the energies, and g_k = ∂E_k/∂q the gradients (length f).
- d_jk = ⟨j|∂k/∂q⟩ is the NAC vector (length f). It is real and antisymmetric for real states.
- v is the velocity and M = diag(m).

1. **Electronic equation:**

   dc_j/dt = −i E_j c_j − Σ_k T_jk c_k, with T_jk = v·d_jk (real, antisymmetric).

   The effective Hamiltonian H_eff = diag(E) − i T is Hermitian.
2. **Propagator over one nuclear step Δt** with N_s = `nsubsteps`, δ = Δt/N_s.
   - E, d and v are linearly interpolated between t and t+Δt at the substep midpoints, s_m = (m − ½)/N_s for m = 1…N_s.
   - Each substep factor is P_m = exp(−i H_eff(s_m) δ), from the eigendecomposition of the Hermitian H_eff.
   - c(t+Δt) = P c(t), with P = P_{N_s} ⋯ P_1, which is unitary.
3. **Tully FSSH probability** (Tully 1990, adiabatic basis), accumulated over the substeps:

   g_{λ→j} = max(0, Σ_m δ · 2 Re(c_λ^* T_λj c_j)|_m / |c_λ|²|_m),

   evaluated with the amplitudes at the start of each substep.

   *Sign check:* from (1), d|c_λ|²/dt = Σ_j (−2 Re(c_λ^* T_λj c_j)), so the flux out of λ into j is +2 Re(c_λ^* T_λj c_j). The probabilities are capped so that their sum is ≤ 1, by scaling them down if necessary.
4. **SHARC-type probability** (Mai, Marquetand & González 2015, eq. for h_{β→α}). With β = λ the active state, c = c(t), c′ = c(t+Δt) and P the step propagator:

   h_{λ→j} = (1 − |c′_λ|²/|c_λ|²) · Re(c′_j P_jλ^* c_λ^*) / (|c_λ|² − Re(c′_λ P_λλ^* c_λ^*)), for j ≠ λ.

   Negative values are set to 0; if |c′_λ|² ≥ |c_λ|², all are 0. The probabilities are capped so that their sum is ≤ 1 as in (3). If the denominator is ≤ 1e-14, all are 0.
5. **Hop decision.** Draw ξ ~ U[0, 1) (always one per step). With the states j ≠ λ in index order and cumulative sums S_j of the probabilities, hop to the first j with ξ < S_j; otherwise there is no hop.
6. **Velocity adjustment** for a hop λ→j at fixed q, with ΔE = E_j − E_λ and kinetic energy K = ½ vᵀMv.
   - **`parallel_vel`:** if ΔE ≤ K, v′ = α v with α = √(1 − ΔE/K); otherwise the hop is frustrated. K = 0 and ΔE > 0 is frustrated.
   - **`parallel_nac`:** u = d_λj and v′ = v + γ M⁻¹u. Energy conservation gives a γ² + b γ + ΔE = 0 with a = ½ uᵀM⁻¹u and b = v·u.
     - If D = b² − 4aΔE < 0, the hop is frustrated.
     - Otherwise γ = (−b ± √D)/(2a), taking the root with the smaller |γ|.
     - If a ≤ 1e-30, use `parallel_vel`'s rule.
   - **Frustrated hop**, `frustrated = "keep"`: v unchanged, no hop. `"reflect"`: no hop, and v′ = v − 2 (v·u / uᵀM⁻¹u) M⁻¹u, which reverses the momentum component along u in the mass-weighted metric (u = v for `parallel_vel`, giving v′ = −v; u = d_λj for `parallel_nac`).
   - Every frustrated event is recorded.
7. **EDC** (Granucci & Persico 2007), after the hop step. With K the kinetic energy and C = `decoherence_param_Eh` (0.1):
   - for j ≠ λ, τ_jλ = (1/|E_j − E_λ|) · (1 + C/K), and c_j ← c_j exp(−Δt/τ_jλ);
   - then c_λ ← c_λ · √((1 − Σ_{j≠λ}|c_j|²)/|c_λ|²).
   - If |E_j − E_λ| = 0 or K = 0, c_j is unchanged for that j.
   - The norm is 1 afterwards.
8. **Step order** (ours, documented). From (q, v, a = −M⁻¹g_λ(q), c, λ):
   1. q′ = q + vΔt + ½aΔt².
   2. snap′ = model.evaluate(q′, previous=snap).
   3. v′ = v + ½(a − M⁻¹g′_λ)Δt.
   4. Propagate c over [t, t+Δt] with E, d interpolated between snap and snap′ and v between v and v′; obtain c′ and P.
   5. Hopping probabilities (scheme); draw ξ.
   6. If there is a hop: velocity adjustment of v′ (frustrated → keep/reflect, λ unchanged); if it succeeds, λ ← j.
   7. Decoherence on c′ with K from v′ and E from snap′.
   8. a′ = −M⁻¹g′_λ (the new active state's gradient at q′).
   9. Record; check the energy drift.

   Because the velocity is completed before the hop and rescaled exactly, the total energy E_λ + K is continuous across hops.
9. **Total energy** is E_tot = E_λ(q) + ½vᵀMv. The `energy_drift` outcome is |E_tot − E_tot(0)| > `energy_drift_limit_au`.

## File Structure

```text
src/gqteashmd/dynamics/__init__.py        # Task 1 (docstring: no workflow imports)
src/gqteashmd/dynamics/options.py         # Task 1: DynamicsOptions
src/gqteashmd/dynamics/records.py         # Task 1: ElectronicSnapshot, TrajectoryState, StepRecord
src/gqteashmd/dynamics/randomness.py      # Task 1: RandomSource, GeneratorSource, ReplaySource
src/gqteashmd/dynamics/models.py          # Task 2: ElectronicModel, DiabaticModel, Tully models, 2D model
src/gqteashmd/dynamics/representation.py  # Task 2: diagonal_basis (U, phase continuity; SOC guard)
src/gqteashmd/dynamics/propagators.py     # Task 3
src/gqteashmd/dynamics/integrators.py     # Task 3
src/gqteashmd/dynamics/hopping.py         # Task 4
src/gqteashmd/dynamics/velocity.py        # Task 5
src/gqteashmd/dynamics/decoherence.py     # Task 5
src/gqteashmd/dynamics/driver.py          # Task 6: step, run_trajectory, Outcome
src/gqteashmd/dynamics/restart.py         # Task 6: save/load TrajectoryState (.npz, atomic)
src/gqteashmd/dynamics/ensemble.py        # Task 7: wavepacket initial conditions, run_model_ensemble, Branching
src/gqteashmd/config/model_dynamics.py    # Task 7: pydantic input for the model runner
src/gqteashmd/cli.py                      # Task 7: `model-dynamics` command
tests/dynamics/...                        # per task
tests/validation/wavepacket.py            # Task 8: exact split-operator solver (test helper)
tests/validation/test_tully_exact.py      # Task 8
tools/sharc_compare/run_sharc_model.py    # Task 9
tests/validation/test_sharc_tully.py      # Task 9 (marker sharc)
CLAUDE.md, USER_MANUAL.md, master plan, strategy   # Task 10
```

---

### Task 1: Options, records, random sources

**Files:** create `dynamics/__init__.py`, `options.py`, `records.py`, `randomness.py`; test `tests/dynamics/test_options_records.py`. Add `tests/dynamics/__init__.py` only if the test tree uses packages (check `tests/vib`).

**Interfaces (produce):**

```python
# options.py
@dataclass(frozen=True)
class DynamicsOptions:
    dt_au: float                      # > 0
    nsubsteps: int = 25               # >= 1
    hopping: str = "sharc"            # "sharc" | "fssh" | "none"
    ekin_correction: str = "parallel_vel"   # "parallel_vel" | "parallel_nac"
    frustrated: str = "keep"          # "keep" | "reflect"
    decoherence: str = "edc"          # "edc" | "none"
    decoherence_param_Eh: float = 0.1 # >= 0
    energy_drift_limit_au: float = 0.5 / 27.211386245988   # > 0 (0.5 eV)
    # __post_init__ validates every field; ValueError naming the field and the allowed values
```

`"none"` hopping is for tests and for the "hopping off" comparisons of strategy §5.4.

```python
# records.py
@dataclass(frozen=True, eq=False)
class ElectronicSnapshot:
    energies: np.ndarray              # (n,) Eh, ascending not required
    gradients: np.ndarray             # (n, f) dE_k/dq
    nacs: np.ndarray                  # (n, n, f) d_jk, antisymmetric in (j, k)
    hamiltonian: np.ndarray           # (n, n) complex MCH Hamiltonian (diag(E) for adiabatic models)
    vectors: np.ndarray | None = None # model-internal eigenvectors for phase continuity (opaque)

@dataclass(eq=False)
class TrajectoryState:
    time_au: float; step: int
    q: np.ndarray; v: np.ndarray; c: np.ndarray  # c complex (n,)
    active: int
    snapshot: ElectronicSnapshot
    e_total_initial: float

@dataclass(frozen=True, eq=False)
class StepRecord:
    step: int; time_au: float
    q: np.ndarray; v: np.ndarray; c: np.ndarray; active: int
    energies: np.ndarray; e_total: float; e_kinetic: float
    probabilities: np.ndarray          # (n,) hop probabilities from the old active state (0 at index λ)
    random_number: float
    hop_from: int | None; hop_to: int | None; frustrated: bool
```

```python
# randomness.py
class RandomSource(Protocol):
    def uniform(self) -> float: ...
    def state(self) -> dict[str, Any]: ...          # JSON-serialisable
class GeneratorSource:  # wraps np.random.Generator; state() = bit_generator.state; from_state()
class ReplaySource:     # draws from a given sequence in order; IndexError -> ValueError("replay sequence exhausted at draw k")
    # state() = {"kind": "replay", "position": k, "values": [...]}
def source_from_state(state: dict[str, Any]) -> RandomSource
```

- [ ] **Step 1: Write the failing tests:**
  - option validation, one test per field with the message containing the field name;
  - the defaults;
  - `GeneratorSource` round-trips through `state()`/`source_from_state` and continues the identical sequence;
  - `ReplaySource` returns the given values in order and raises `ValueError` when exhausted;
  - the records are constructible.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them to verify they pass**, plus `mypy`.
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add the dynamics options, records and random sources"
```

---

### Task 2: Models and representation

**Files:** create `dynamics/models.py`, `dynamics/representation.py`; tests `tests/dynamics/test_models.py`, `tests/dynamics/test_representation.py`.

**Interfaces (produce):**

```python
class ElectronicModel(Protocol):
    n_states: int; n_dof: int; masses: np.ndarray    # (f,) electron masses
    def evaluate(self, q: np.ndarray, previous: ElectronicSnapshot | None) -> ElectronicSnapshot: ...

class DiabaticModel(ABC):     # implements ElectronicModel from a diabatic matrix
    @abstractmethod
    def diabatic(self, q) -> tuple[np.ndarray, np.ndarray]: ...  # V (n,n) real symmetric, dV (f,n,n)
    def evaluate(self, q, previous):
        # E, W = eigh(V) ascending; align each column of W to previous.vectors by sign (W_k <- -W_k if
        # W_k . W_prev_k < 0); G_f = W^T dV_f W; gradients[k, f] = G_f[k, k];
        # nacs[j, k, f] = G_f[j, k] / (E_k - E_j) for j != k (0 on the diagonal);
        # if |E_k - E_j| < 1e-12 raise ValueError("degenerate states ...")
        # hamiltonian = diag(E) as complex; vectors = W
```

- **The models** (Tully 1990), with mass default 2000 (`masses = np.array([mass])`):
  - `TullySimpleAvoidedCrossing(A=0.01, B=1.6, C=0.005, D=1.0)`:
    - V11 = A(1 − e^{−Bx}) for x ≥ 0, and −A(1 − e^{Bx}) for x < 0;
    - V22 = −V11;
    - V12 = C e^{−Dx²}.
  - `TullyDualAvoidedCrossing(A=0.1, B=0.28, C=0.015, D=0.06, E0=0.05)`:
    - V11 = 0;
    - V22 = −A e^{−Bx²} + E0;
    - V12 = C e^{−Dx²}.
  - `TullyExtendedCoupling(A=6e-4, B=0.1, C=0.9)`:
    - V11 = A, V22 = −A;
    - V12 = B e^{Cx} for x < 0, and B(2 − e^{−Cx}) for x ≥ 0.
  - `TwoDimensionalCrossing(k=0.02, a=1.0, delta=0.01, c=0.005, mass=2000.0)`, a 2-state, f = 2 model for the multi-dimensional tests:
    - V11 = ½k((x − a)² + y²);
    - V22 = ½k((x + a)² + y²) + delta;
    - V12 = c·y.

    It has a conical intersection, so tests must stay away from it.
- **`representation.py`:** `diagonal_basis(hamiltonian, previous_u) -> tuple[np.ndarray, np.ndarray]`.
  - It diagonalizes the Hermitian MCH H and returns (E_diag, U). The phase of each column of U is aligned to `previous_u` (U_k ← U_k · conj(phase of ⟨U_prev,k|U_k⟩)), and columns are reordered to maximum overlap only if needed.
  - **P7a-1 guard:** if U is not diagonal up to phases (max off-diagonal |U| > 1e-10), raise `NotImplementedError("spin-orbit coupling / non-diagonal MCH Hamiltonians need P7a-2 (diagonal representation)")`.
  - The driver calls it every step, so the code path exists.

- [ ] **Step 1: Write the failing tests:**
  - for each model at several q: V is symmetric; dV against central finite differences (h = 1e-5, atol 1e-8); the gradients against FD of the adiabatic energies; the NACs against FD of the aligned eigenvectors, ⟨j| (|k(q+h)⟩ − |k(q−h)⟩)/2h, to 1e-6;
  - NAC antisymmetry;
  - sign continuity along a path: walk x from −10 to 10 in 0.01 steps passing `previous`, and check that ⟨W(x)|W(x+δ)⟩_kk > 0 throughout;
  - the Tully SAC asymptotic energies ±0.01 at |x| = 10;
  - `diagonal_basis` on diag(E) gives U = I and E sorted;
  - the guard raises on a Hamiltonian with an off-diagonal element 0.01;
  - degenerate states give `ValueError`.
- [ ] **Step 2–4:** red, implement, green, `mypy`.
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add analytic Tully models and the diagonal-basis guard"
```

---

### Task 3: Electronic propagator and velocity Verlet

**Files:** create `dynamics/propagators.py`, `dynamics/integrators.py`; tests `tests/dynamics/test_propagators.py`, `tests/dynamics/test_integrators.py`.

**Interfaces (produce):**

```python
def propagate(c: np.ndarray, e0, e1, nac0, nac1, v0, v1, dt: float, nsubsteps: int
              ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Method reference (2). Returns c(t+dt), the step propagator P (n,n) and the list of the
    amplitudes at the start of each substep (for FSSH), all complex."""

def coupling_matrix(nacs: np.ndarray, v: np.ndarray) -> np.ndarray:  # T_jk = v . d_jk  (n,n)

def verlet_positions(q, v, a, dt) -> np.ndarray       # q + v dt + 0.5 a dt^2
def verlet_velocities(v, a_old, a_new, dt) -> np.ndarray   # v + 0.5 (a_old + a_new) dt
def acceleration(gradient: np.ndarray, masses: np.ndarray) -> np.ndarray  # -g / m
```

- [ ] **Step 1: Write the failing tests:**
  - **Unitarity and norm:** for a random Hermitian/antisymmetric setup, P†P = I to 1e-12 and |c′| = 1.
  - **Agreement with `scipy.linalg.expm`:** for constant E and T, P equals expm(−i(diag E − iT)Δt) to 1e-12, for N_s = 1 and N_s = 25.
  - **Rabi:** for two states with E = (0, 0) and a constant coupling T_12 = −T_21 = κ, the populations follow cos²(κt) and sin²(κt) exactly.
  - **Time reversal:** propagating forward and then backward (v → −v, Δt → −Δt) returns c to 1e-12.
  - **Adiabatic limit:** with T = 0, only phases change, e^{−iE_kΔt}.
  - **Landau–Zener:** the linear diabatic crossing V11 = αx, V22 = −αx, V12 = Γ, with x(t) = x0 + u t at constant u. Electronic propagation only, along the prescribed path from x = −L to x = L, L = 40Γ/α, with adiabatic E and NACs from a `DiabaticModel` subclass in the test. The final probability of **leaving** the initial adiabatic state, which is the Landau–Zener diabatic passage, must equal P_LZ = exp(−2πΓ²/(u·|d(V11 − V22)/dx|)) = exp(−πΓ²/(uα)) within 1e-3 for Γ = 0.005, α = 0.01 and u in {0.002, 0.005, 0.01}. The expected values are 0.0197, 0.208 and 0.456. If the finite path length leaves residual oscillations above 1e-3, increase L and report it.
  - **Verlet:** the harmonic oscillator conserves energy to O(Δt²) over 1000 steps with relative drift < 1e-4 at Δt = 0.05·period/(2π), and it is time-reversible.
- [ ] **Step 2–4:** red, implement, green, `mypy`.
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add the exponential electronic propagator and velocity Verlet"
```

---

### Task 4: Hopping probabilities and the hop decision

**Files:** create `dynamics/hopping.py`; test `tests/dynamics/test_hopping.py`.

**Interfaces (produce):**

```python
def fssh_probabilities(active: int, substep_amplitudes: list[np.ndarray], t0: np.ndarray,
                       t1: np.ndarray, dt: float) -> np.ndarray
    """Method reference (3); coupling matrices T interpolated at substep midpoints like the
    propagator; returns (n,) with 0 at `active`, sum <= 1."""
def sharc_probabilities(active: int, c_old: np.ndarray, c_new: np.ndarray,
                        propagator: np.ndarray) -> np.ndarray      # method reference (4)
def choose_state(active: int, probabilities: np.ndarray, xi: float) -> int | None  # (5)
def hopping_probabilities(scheme: str, ...) -> np.ndarray   # dispatch on "sharc" | "fssh" | "none"
```

- [ ] **Step 1: Write the failing tests:**
  - Probabilities are ≥ 0, sum to ≤ 1, and are 0 at the active index.
  - Without coupling (T = 0) both schemes give 0.
  - **Small-Δt flux:** for a two-state system where the population flows monotonically out of λ, both schemes give Σ_j h ≈ (|c_λ|² − |c′_λ|²)/|c_λ|² to O(Δt²) (relative 1e-3 at small Δt). The two schemes agree with each other to the same order.
  - **SHARC conservation property:** when all flux leaves λ, Σ h = 1 − |c′_λ|²/|c_λ|² exactly (two-state case).
  - **`choose_state`:** ξ just below the cumulative sum hops, ξ at or above it does not; states are ordered by index; with all probabilities 0, `None`.
  - **Statistical:** 20000 draws with fixed probabilities (0.1, 0.3) reproduce the frequencies within 4σ.
  - `"none"` returns zeros.
- [ ] **Step 2–4:** red, implement, green, `mypy`.
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add FSSH and SHARC-type hopping probabilities"
```

---

### Task 5: Velocity adjustment, frustrated hops, EDC decoherence

**Files:** create `dynamics/velocity.py`, `dynamics/decoherence.py`; tests `tests/dynamics/test_velocity.py`, `tests/dynamics/test_decoherence.py`.

**Interfaces (produce):**

```python
@dataclass(frozen=True)
class Adjustment:
    accepted: bool
    velocity: np.ndarray       # new v (adjusted, or reflected / unchanged if frustrated)
def adjust_velocity(v, masses, delta_e, *, method: str, nac: np.ndarray | None, frustrated: str
                    ) -> Adjustment       # method reference (6)
def apply_decoherence(c, active, energies, kinetic_energy, dt, *, scheme: str, param: float
                      ) -> np.ndarray     # method reference (7); "none" returns c unchanged
```

- [ ] **Step 1: Write the failing tests.** Use property tests over 200 random draws with a fixed rng, no new dependency, for f = 1, 2 and 6:
  - **Energy conservation:** an accepted adjustment conserves K + ΔE exactly (relative 1e-12).
  - **`parallel_nac`** changes v only along M⁻¹u, and picks the smaller |γ|, which for a small ΔE is the root continuously connected to γ = 0.
  - **`parallel_vel`** keeps the direction.
  - **Frustrated:** ΔE > K for `parallel_vel`, or D < 0, gives `accepted = False`; `"keep"` leaves v unchanged; `"reflect"` reverses the component along u in the mass-weighted metric and conserves K exactly.
  - **Downward hops** (ΔE < 0) are always accepted.
  - **EDC:** the norm is 1 after the step; |c_λ| does not decrease; the coherences decay with the τ formula, checked on a two-state example by hand, e.g. E = (0, 0.02), K = 0.01, C = 0.1, Δt = 1 → τ = (1/0.02)(1 + 10) = 550, factor e^{−1/550}. `"none"` is the identity; K = 0 or degenerate energies leave c_j unchanged.
- [ ] **Step 2–4:** red, implement, green, `mypy`.
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add velocity adjustment, frustrated hops and EDC decoherence"
```

---

### Task 6: Driver, outcomes, restart

**Files:** create `dynamics/driver.py`, `dynamics/restart.py`; tests `tests/dynamics/test_driver.py`, `tests/dynamics/test_restart.py`.

**Interfaces (produce):**

```python
class Outcome(str, Enum): FINISHED = "finished"; STOPPED = "stopped"; ENERGY_DRIFT = "energy_drift"

def initial_state(model, q0, v0, active: int, c0: np.ndarray | None = None) -> TrajectoryState
    # c0 default: unit vector on `active`; evaluates the model at q0; records E_tot(0)

def step(model, state: TrajectoryState, options: DynamicsOptions, rng: RandomSource,
         u_previous: np.ndarray | None = None) -> tuple[TrajectoryState, StepRecord]
    # method reference (8) exactly; calls representation.diagonal_basis every step

def run_trajectory(model, state, options, rng, *, max_steps: int,
                   stop: Callable[[TrajectoryState], bool] | None = None,
                   on_step: Callable[[StepRecord], None] | None = None
                   ) -> tuple[TrajectoryState, Outcome]
    # stops on: stop(state) True -> STOPPED; energy drift -> ENERGY_DRIFT; max_steps -> FINISHED

# restart.py
def save_state(path: Path, state: TrajectoryState, rng: RandomSource) -> None   # atomic .npz + JSON rng state
def load_state(path: Path, model) -> tuple[TrajectoryState, RandomSource]       # re-evaluates nothing:
    # the snapshot (energies, gradients, nacs, hamiltonian, vectors) is stored, so continuing is bit-identical
```

- [ ] **Step 1: Write the failing tests:**
  - **Energy conservation:** with `hopping = "none"` on Tully SAC, E_tot is conserved to < 1e-6 Eh over the passage, at Δt = 1.0 and m = 2000, starting at x = −10 with k = 20 (v = k/m).
  - **Hops:** with hopping on, E_tot stays continuous across hops (|ΔE_tot| per hop step < 1e-6). The `StepRecord` of a hop step has `hop_from`/`hop_to` set, and a frustrated step has `frustrated = True`.
  - **Determinism:** the same seed gives identical records; `ReplaySource` with the logged random numbers reproduces the same trajectory.
  - **Restart:** saving at step 50 and continuing 50 steps equals 100 continuous steps **bit-identically** (`np.array_equal` on q, v, c and the active state) for both hopping schemes with EDC on.
  - **Outcomes:** the drift outcome is triggered by a tiny `energy_drift_limit_au`; the stop callback works.
  - **The SOC guard propagates:** a model returning a non-diagonal H raises `NotImplementedError`.
- [ ] **Step 2–4:** red, implement, green, `mypy`.
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add the surface-hopping driver and bit-identical restarts"
```

---

### Task 7: Ensemble runner and `gqteashmd model-dynamics`

**Files:** create `dynamics/ensemble.py`, `config/model_dynamics.py`; modify `cli.py`; tests `tests/dynamics/test_ensemble.py`, `tests/test_cli_model_dynamics.py`.

**Interfaces (produce):**
- **`wavepacket_initial_conditions(n, x0, k, sigma, mass, rng, sampling)`.**
  - The packet is ψ(x) ∝ exp(−(x − x0)²/σ² + ikx), so |ψ|² has standard deviation s_x = σ/2 and the Wigner function has s_p = 1/σ.
  - `"wigner"` draws x ~ N(x0, s_x), p ~ N(k, s_p) and returns q (n, 1) and v = p/mass. `"fixed"` returns x0 and k/mass.
- **`run_model_ensemble(model, *, n_trajectories, x0, k, sigma, initial_state, options, seed, x_exit, max_steps, sampling) -> Branching`.**
  - Trajectory i uses the D12-pattern stream (Global Constraints).
  - The stop condition is |x| > x_exit while moving outward.
  - **`Branching`:** counts of `transmitted_lower`, `transmitted_upper`, `reflected_lower`, `reflected_upper`, `unfinished` and `energy_drift`; `probabilities()` (fractions over the finished trajectories); and 95 % Wilson intervals.
  - "Lower/upper" is the adiabatic state index at the end (0 or 1); transmitted means x > 0.
- **`config/model_dynamics.py`:** a pydantic `ModelDynamicsInput` with these sections:

  ```toml
  [model]
  name = "tully_sac"     # | "tully_dac" | "tully_ecr"
  mass = 2000.0
  # optional parameter overrides

  [initial]
  x0 = -10.0
  k = 20.0
  sigma = 1.0            # default 20/k
  state = 0
  sampling = "wigner"    # | "fixed"

  [dynamics]
  dt_au = 1.0
  nsubsteps = 25
  hopping = "sharc"
  ekin_correction = "parallel_vel"
  frustrated = "keep"
  decoherence = "edc"
  decoherence_param_Eh = 0.1
  energy_drift_limit_au = 0.0184

  [ensemble]
  n_trajectories = 1000
  seed = 12345
  x_exit = 10.0
  max_steps = 200000
  ```

  `extra="forbid"`, finite floats, and every violation reported at once, as in `schema.py`.
- **CLI:** `gqteashmd model-dynamics INPUT.toml [--json OUT]`. It prints a table of branching probabilities with intervals; exit 0, or 2 on a `ConfigError`. It is a separate command with no project directory, like `single-point`.

- [ ] **Step 1: Write the failing tests.**
  - The initial-condition moments match s_x and s_p (5σ statistical).
  - Branching on Tully SAC at k = 30: hops are rare, so P(transmitted lower) ≈ 1 − P_LZ-like; assert only that the probabilities sum to 1 and that the run is deterministic for a seed (n = 50).
  - The CLI writes a JSON file with the documented keys; an invalid input exits 2 with a message naming the key.
  - `import gqteashmd.cli` still loads no h5py, pyscf or geometric.
- [ ] **Step 2–4:** red, implement, green, `mypy`.
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add the model-dynamics ensemble runner and command"
```

---

### Task 8: Validation — exact wavepacket and published values

**Files:** create `tests/validation/wavepacket.py` (helper, our own code) and `tests/validation/test_tully_exact.py` (marker `slow`). The fast smoke test is unmarked.

**Interfaces:**
- **`exact_branching(model, *, mass, x0, k, sigma, x_min, x_max, n_grid, dt, t_final) -> dict`.**
  - A two-state split-operator FFT propagation of ψ(x) = (ψ₁, ψ₂) in the **diabatic** basis: exp(−iVδt/2) exp(−iTδt) exp(−iVδt/2), with the 2×2 exp(−iV) at each grid point from the eigendecomposition. The initial packet is on the lower **adiabatic** state, i.e. the projection uses the local adiabatic eigenvectors at each grid point.
  - **End:** propagate until the norm inside |x| < x_int is below 1e-4. Then project ψ onto the local adiabatic eigenvectors at each grid point and integrate x > 0 → transmitted, x < 0 → reflected, per adiabatic state.
  - **Convergence:** the helper must have a self-test where halving dx and dt changes the probabilities by < 1e-3, and the total norm is 1 to 1e-8.
- **Comparison test**, per model and per k, with n = 2000 FSSH trajectories (Wigner initial conditions, SHARC hopping, `parallel_vel`, EDC on, Δt = 0.5 a.u., 25 substeps):
  - **SAC** at k ∈ {10, 15, 20, 30}: |P_FSSH − P_exact| ≤ 4 · SE_binomial + 0.02 for transmitted lower/upper. FSSH is known to be accurate here.
  - **DAC** at ln(E_kin) ∈ {−2, −1, 0, 1} (E = k²/2m): the same rule for transmitted lower/upper. **Known FSSH deviations** (e.g. the Stückelberg oscillations at low energy) are *not* asserted; report them.
  - **ECR** at k ∈ {10, 20, 30}: assert only the reflection-on-lower and transmission-on-upper probabilities, with the same rule. The upper-state reflection is a known decoherence-sensitive quantity, so report it with and without EDC.
  - Every measured number goes into the test output (`-s`) and the report.
- **Published values.** Find tabulated FSSH and/or exact branching probabilities for Tully's models in an **open-access** source; Tully 1990 itself gives figures only.
  - Cite each value in the test with DOI and table or figure, and compare at up to 6 points, with tolerance 0.05 absolute (digitization) unless the source tabulates numbers.
  - If no open tabulated source is found, write that in the report and skip this part with an explanatory `pytest.skip`. Do **not** invent numbers.
- [ ] **Step 1:** Write the helper with its convergence self-test, then the comparison tests.
- [ ] **Step 2:** Run `pytest tests/validation/test_tully_exact.py -m slow -v -W error -s`. Expected: pass, with every number reported. If an FSSH-vs-exact assertion fails where FSSH is supposed to be accurate, debug the core (`superpowers:systematic-debugging`); do not loosen the tolerance.
- [ ] **Step 3: Commit**

```bash
git commit -m "test: validate the dynamics core against exact wavepacket results on Tully's models"
```

---

### Task 9: Validation — SHARC 4.1 driver as a black box on the same model (feasibility-gated)

**Files:** create `tools/sharc_compare/run_sharc_model.py` and `tests/validation/test_sharc_tully.py` (markers `sharc`, `slow`).

**Procedure:**
1. **Feasibility probe (document it in the report).** From the **public** SHARC 4.1 manual and tutorial (sharc-md.org, read with the web-fetch tool; never the files under `/home/ajc/sharc4-4.1`), learn:
   - how to run the SHARC driver (`sharc.x`) on a single initial condition with the analytical-PES interface `SHARC_ANALYTICAL.py`: the `input` keywords, the `ANALYTICAL.template` format, and the geometry/velocity files;
   - which keywords set Δt, the substeps, the coupling (`nacdr`), the hopping (`sharc`), `ekincorrect parallel_vel`, frustrated-hop handling, decoherence (`edc`, `decoherence_param 0.1`) and the random seed;
   - how SHARC writes its per-step output (`output.lis`/`output.dat`), to parse the final state and position.

   A 1D Tully model can be expressed as one atom of mass 2000 mₑ moving along x (the potentials depend on x only; y and z velocities are 0). Translation/rotation removal must be off.
2. **If it is feasible:** `run_sharc_model.run(sharc_home, python, model_spec, x0, v0, n_steps, dt_fs, options, seed, work_dir)` writes the inputs with our own writer, executes SHARC (subprocess, timeout, environment as in `run_wigner.py`), and parses the final state with our own parser.
   - Run an ensemble of N = 500 for Tully SAC at k = 20 and ECR at k = 20, with Wigner initial conditions drawn by **our** sampler and fed to both programs.
   - Compare the branching probabilities with ours under the same options (two-proportion z-test, |z| < 4).
   - Also compare one trajectory with **hopping off** step by step (positions, velocities, |c|²) over 200 steps: the differences must stay small (report the growth; assert < 1e-6 in |c|² for the first 50 steps, or document the step-order difference if they differ systematically).
3. **If it is not feasible** (the analytical interface cannot express the model, or the driver cannot be run non-interactively), stop and report **BLOCKED** with the findings. The controller decides whether to move this comparison to P7a-2, which needs the same SHARC-driver harness for LVC.
- [ ] Commit (if feasible):

```bash
git commit -m "test: compare the dynamics core with the SHARC 4.1 driver on Tully models (black box)"
```

---

### Task 10: Documentation

**Files:** `CLAUDE.md`, `USER_MANUAL.md`, `docs/superpowers/plans/2026-11-06-gqteashmd-master-plan.md`, `SHMD_implementation_strategy.md` (§4.1 note).

- [ ] **USER_MANUAL.md:**
  - a new [IMPLEMENTED] section on the dynamics core: every formula of the Method reference with all symbols, the step order, the options, the outcomes and the restart;
  - the `model-dynamics` command, its input keys and a real example run on Tully SAC (run it for real from a scratch script);
  - the validation numbers from Tasks 8 and 9;
  - [PLANNED] markers for LVC, overlaps/local diabatization, SOC and the `dynamics` stage (P7a-2, P7b).
- [ ] **CLAUDE.md:** the status (P7a-1 on the branch, with counts), the architecture (`dynamics/` package and its dependency rule), the rules (step order, one random number per step, restart bit-identity), decision O6 resolved and the split of P7a, and next step P7a-2.
- [ ] **Master plan:** §9 (P7a split into P7a-1 and P7a-2; P7a-1 delivered); §11 (O6 resolved; O7 noted as designed but not built).
- [ ] **Strategy §4.1:** replace "read SHARC's main.F90" with "our documented step order; differences from SHARC are measured by black-box comparison (clean room)".
- [ ] **Commit**

```bash
git commit -m "docs: document the surface-hopping dynamics core (P7a-1)"
```

---

## Self-review notes

- **Spec coverage:**

  | Spec item | Task |
  |---|---|
  | Integrator | Task 3 |
  | Propagators | Task 3 |
  | Hopping | Task 4 |
  | Rescaling and frustrated hops | Task 5 |
  | EDC | Task 5 |
  | MCH/diagonal data model and SOC guard | Tasks 1, 2 |
  | Analytic backend | Task 2 |
  | Standalone runner | Task 7 |
  | RNG and replay | Tasks 1, 6 |
  | Restart, bit-identical | Task 6 |
  | Tully validation: exact | Task 8 |
  | Tully validation: literature | Task 8 |
  | Tully validation: SHARC | Task 9 |
  | LVC, local diabatization, SOC, `eselect`, `sharc_external` | P7a-2 by owner decision 1 |

- **Type consistency:**
  - `ElectronicSnapshot`, `TrajectoryState`, `StepRecord` and `DynamicsOptions` (Task 1) are used in Tasks 2–9.
  - `propagate` (Task 3) returns the substep amplitudes that `fssh_probabilities` (Task 4) consumes.
  - `adjust_velocity` and `apply_decoherence` (Task 5) are used by `step` (Task 6).
  - `run_trajectory` (Task 6) is used by `run_model_ensemble` (Task 7).
  - `Branching` (Task 7) is used by Tasks 8 and 9.
