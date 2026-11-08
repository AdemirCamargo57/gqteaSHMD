# P4: `optimize` and `frequencies` Stages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `gqteashmd run` take a molecule from the input file to an optimized S0 geometry and its harmonic vibrational analysis. That analysis means frequencies, normal modes, a Molden file and `frequencies.h5`. The run uses the `[electronic]` level of theory and stops on significant imaginary modes. The plan also closes the orchestrator items that P3c moved to P4.

**Architecture:**
- **Two new optional input sections.** `[optimize]` (with `enabled`) and `[frequencies]`.
- **Two new stages.**
  - `OptimizeStage` runs one task: geomeTRIC with our own custom engine wrapped around the `ElectronicBackend`.
  - `FrequenciesStage` gets its Hessian one of two ways. The analytic SCF Hessian takes one task. Finite differences of S0 gradients take 1 + 6N parallel tasks. Its `finalize` runs our own harmonic analysis (`gqteashmd.vib.harmonic`) and writes the artifacts.
- **No PySCF dependency outside the backend.**
  - Masses come from our own table, `gqteashmd.chem.masses`.
  - geomeTRIC is imported only inside `gqteashmd.geomopt`.
  - PySCF stays inside `electronic/pyscf_*.py`.

**Tech Stack:** Python 3.12, NumPy, h5py, geomeTRIC 1.1, PySCF 2.14, pydantic v2, pytest, ruff 0.17.4, mypy.

**Spec:**
- [`2026-11-06-gqteashmd-master-plan.md`](2026-11-06-gqteashmd-master-plan.md):
  - §1.1: stages 1–2.
  - §3: orchestration, optional stages.
  - §4.3: `[optimize]`, `[frequencies]`.
  - §5: stage table rows `optimize` and `frequencies`, artifacts.
  - §6.1.
  - §7: validation of `frequencies`.
  - §9: P4 row.
  - D6, D11, O5.
- The P3c plan "Scope rulings" (items moved to P4).
- [`CLAUDE.md`](../../../CLAUDE.md).

**Owner decisions (2026-11-07):**
1. Optimization and frequencies use the **same level as `[electronic]`**. This puts them on the dynamics S0 surface.
2. Add an **optional ORCA comparison test**. ORCA 6.1.1 is installed at `/home/ajc/bin/orca_6_1_1/orca`.

## Global Constraints

- **Clean room.** No SHARC code may be copied or read. SHARC may only be *executed*, and its output inspected. The same applies to ORCA: execute it and parse its output.
  - The harmonic analysis (mass weighting, Eckart projection) is implemented from the literature: Wilson, Decius and Cross, *Molecular Vibrations*; and the standard Eckart/Sayvetz conditions.
  - The licence is MIT.
- **Environment.** Linux only. Run every command inside WSL Oracle Linux 9.5, from **PowerShell**:
  `wsl -d OracleLinux_9_5 -- bash -lc 'cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && conda activate gqteashmd-dev && <command>'`.
  - Scratch goes in `mktemp -d /tmp/gq-XXXX`. Before writing scratch output, check with `pwd` that you are really in it. A `cd` can fail silently when the command passes through Git Bash.
  - Delete scratch by its exact path.
  - Run one pytest process at a time.
- **Dependencies.** Runtime: `pydantic>=2.6`, `numpy>=1.26`, `scipy>=1.11`, `pyscf>=2.6`, plus **new** `h5py>=3.9` and `geometric>=1.0`. `environment.yml` pins `pyscf=2.14.*` and **new** `geometric=1.1.*`.
- **Imports.**
  - PySCF is imported only inside functions in `electronic/pyscf_*.py`.
  - geomeTRIC is imported only inside functions in `gqteashmd/geomopt.py`.
  - h5py is imported only inside functions in `gqteashmd/vib/artifacts.py`.
  - `import gqteashmd.cli` must stay free of all three. Extend the existing lazy-import test.
- **Units.** Coordinates are in bohr inside the program and Å in XYZ files. Masses are in amu (dalton). The Hessian is in Eh/bohr². Frequencies are wavenumbers in cm⁻¹, with imaginary modes written as **negative** numbers. Constants come from `gqteashmd.core.units`.
- **Masses (O5, resolved here).** Use the mass of the most abundant isotope. Values are in amu, from `gqteashmd.chem.masses`.
- **States.** Optimization and frequencies are for **S0** on the `[electronic]` surface.
  - For SCF methods (`tda`/`tddft`), S0 is the SCF ground state, so the stages compute only S0 (`singlets = 1` internally).
  - For SA-CASSCF, S0 is the state-averaged root 0. The state average defines that surface, so the full `[electronic]` settings are kept.
- **Errors.**

  | Problem | Raised as |
  |---|---|
  | User input problems | `ConfigError` (exit 2) |
  | Non-converged optimization | non-retryable `TaskError`, so the stage fails (exit 4) with advice |
  | Imaginary mode with \|ν\| > cutoff | `StageFailedError` (exit 4), listing the modes |
  | Electronic-structure failures | `ElectronicStructureError` (as in P3) |
- **Backends.** A backend keeps the previous geometry (P3b). The optimizer reuses **one** backend for its whole path. Every finite-difference Hessian task builds a **fresh** backend.
- **Style.** ruff 0.17.4, line length 100, rules E, F, W, I, B, UP, SIM. mypy must stay clean on `src/`.
- **Git.** Branch `feat/p4-optimize-frequencies`, WSL Git only, hooks on, never `--no-verify`. Use conventional subjects and the `Co-Authored-By:` trailer of the model that wrote the change. Do not push.
- **Documentation.** The last task updates CLAUDE.md, USER_MANUAL.md and the master plan. Keep [IMPLEMENTED] and [PLANNED] apart, and define every symbol and unit.

## Facts verified before writing this plan (WSL, 2026-11-07)

1. **geomeTRIC 1.1.1 with a custom engine.**
   - A subclass of `geometric.engine.Engine` overrides `calc_new(self, coords, dirname)`. `coords` is a flat array in **bohr**. The method returns `{"energy": float, "gradient": flat ndarray}`.
   - The optimizer is started with `geometric.optimize.run_optimizer(customengine=engine, check=1, input=<path prefix>, maxiter=..., convergence_set=...)`. It returns a `geometric.molecule.Molecule`; the final geometry is `res.xyzs[-1]` (Å) and the final energy is `res.qm_energies[-1]`.
   - The engine's `Molecule` is built with `.elem = [...]` and `.xyzs = [Å array]`.
   - Water RHF/6-31G* converged in 3 gradient calls and 0.3 s.
   - Convergence sets: `GAU` (the default), `GAU_LOOSE`, `GAU_TIGHT`, `GAU_VERYTIGHT`, `NWCHEM_LOOSE`, `TURBOMOLE`, `INTERFRAG_TIGHT`.
   - On failure geomeTRIC raises `geometric.errors.GeomOptNotConvergedError`. It prints its progress to stdout, which ends up in `worker.log`.
2. **PySCF analytic Hessians.** `mf.Hessian().kernel()` works for RHF and RKS (B3LYP checked) and returns an array of shape (natm, natm, 3, 3). `transpose(0, 2, 1, 3).reshape(3N, 3N)` gives the Cartesian Hessian. SA-CASSCF has no analytic Hessian, so finite differences are used.
3. **Harmonic analysis matches PySCF.** The algorithm of Task 1 (mass weighting, then an orthonormal translation/rotation basis by SVD, then diagonalization in the orthogonal complement) reproduces `pyscf.hessian.thermo.harmonic_analysis` with the same masses. Frequencies and reduced masses match to all printed digits:

   | Molecule (level) | Frequencies (cm⁻¹) | Note |
   |---|---|---|
   | Water, RHF/6-31G* | 1826.144, 4055.833, 4174.213 | |
   | Water, B3LYP/6-31G* | 1710.84, 3721.07, 3844.93 | |
   | CO₂, RHF/STO-3G | 566.065 (×2), 1435.438, 2536.18 | linear: 5 external modes detected |

   - The finite-difference Hessian (h = 0.005 bohr) agrees with the analytic one to 0.064 cm⁻¹.
4. **Residual translation/rotation contamination.** Even at optimized geometries, the projected translation/rotation block shows ν up to 25–41 cm⁻¹ (RHF) and 22 cm⁻¹ (B3LYP, grid 3). This comes from residual gradient and the DFT grid.
   - **Ruling:** the master plan's "wrong number of near-zero modes → stop" becomes a **warning** when the largest |ν| in that block exceeds 100 cm⁻¹. The complement-basis method separates internal from external modes exactly, so the count of modes is always right.
   - The value is recorded in `frequencies.json`.
5. **ORCA 6.1.1 matches.** `! RHF 6-31G* Freq TightSCF` on the optimized water gave 1826.02, 4055.42 and 4173.83 cm⁻¹. ORCA uses averaged atomic masses, and PySCF with averaged masses gave 1825.98, 4055.47 and 4173.84.
   - **Output to parse:** a line "VIBRATIONAL FREQUENCIES", then lines like `     6:    1826.02 cm**-1`. The first 6 (or 5) entries are 0.00.
   - **Masses:** the atom table has the header `  NO LB      ZA    FRAG     MASS         X           Y           Z`.
6. **Most-abundant-isotope masses.** The values for H–Rn (Z = 1–86, from PySCF's `COMMON_ISOTOPE_MASSES`, which come from NIST) are given in Task 1, verbatim.

## File Structure

```text
src/gqteashmd/chem/masses.py              # Task 1 (new): most-abundant isotope masses
src/gqteashmd/vib/__init__.py             # Task 1 (new, empty)
src/gqteashmd/vib/harmonic.py             # Task 1 (new): harmonic_analysis, HarmonicAnalysis
src/gqteashmd/vib/molden.py               # Task 2 (new): format_molden
src/gqteashmd/vib/artifacts.py            # Task 2 (new): write_frequencies_h5 (lazy h5py)
pyproject.toml, environment.yml           # Tasks 2, 5: h5py, geometric deps; markers orca (Task 9)
src/gqteashmd/config/schema.py            # Task 3: OptimizeSection, FrequenciesSection, all-violations validators
src/gqteashmd/cli.py                      # Tasks 3, 6: TEMPLATE; validate calls check_stage_inputs
src/gqteashmd/electronic/base.py          # Task 4: Capabilities.hessian, ground_state_hessian
src/gqteashmd/electronic/pyscf_backend.py # Task 4: analytic SCF Hessian
src/gqteashmd/electronic/factory.py       # Task 4: settings=, scratch_dir=; ground_state_settings
src/gqteashmd/geomopt.py                  # Task 5 (new): optimize_geometry (geomeTRIC custom engine)
src/gqteashmd/stages/geometry.py          # Task 6 (new): prepared_molecule, working_geometry, stage_backend
src/gqteashmd/stages/optimize.py          # Task 6 (new): OptimizeStage
src/gqteashmd/stages/frequencies.py       # Task 7 (new): FrequenciesStage
src/gqteashmd/stages/registry.py          # Task 8: prepare → optimize → frequencies
src/gqteashmd/stages/pipeline.py          # Task 8: check_stage_inputs
src/gqteashmd/testing/pipelines.py        # Task 8 (new): prepare_only (for orchestration tests)
src/gqteashmd/orchestrator.py             # Tasks 8, 10: required sections; refuse stale runs before writing
src/gqteashmd/preflight.py                # Task 11: "name>=version" requirements
tests/...                                 # per task
CLAUDE.md, USER_MANUAL.md, master plan    # Task 12
```

---

### Task 1: Isotope masses and the harmonic analysis

**Files:**
- Create: `src/gqteashmd/chem/masses.py`, `src/gqteashmd/vib/__init__.py` (empty), `src/gqteashmd/vib/harmonic.py`
- Test: `tests/chem/test_masses.py`, `tests/vib/test_harmonic.py` (pure tests), `tests/vib/test_harmonic_pyscf.py` (cross-check against PySCF, with `pytestmark = pytest.mark.pyscf`)

**Interfaces:**
- Produces:
  - `isotope_mass(symbol: str) -> float` (amu); unknown symbols raise `ValueError`.
  - `HarmonicAnalysis`, a frozen dataclass:
    - `frequencies_cm1`: (n_vib,), sorted ascending, imaginary values negative;
    - `modes_mass_weighted`: (3N, n_vib), orthonormal columns;
    - `modes_cartesian`: (3N, n_vib), unit-length columns;
    - `reduced_masses_amu`: (n_vib,);
    - `linear`: bool;
    - `n_external`: 3 (atom), 5 (linear) or 6;
    - `external_residual_cm1`: largest |ν| of the translation/rotation block.
  - `harmonic_analysis(masses_amu, coords_bohr, hessian) -> HarmonicAnalysis`. The Hessian is (3N, 3N) in Eh/bohr²; it is symmetrized, and a non-finite or wrongly shaped Hessian raises `ValueError`.

- [ ] **Step 1: Write the failing tests.**

`tests/chem/test_masses.py`:

```python
import pytest

from gqteashmd.chem.masses import isotope_mass


def test_most_abundant_isotope_masses():
    assert isotope_mass("H") == 1.007825
    assert isotope_mass("C") == 12.0
    assert isotope_mass("O") == 15.994915
    assert isotope_mass("Cl") == 34.968853
    assert isotope_mass("Rn") == 222.01757


def test_unknown_symbol():
    with pytest.raises(ValueError, match="Xx"):
        isotope_mass("Xx")
```

`tests/vib/test_harmonic.py`:

```python
import math

import numpy as np
import pytest

from gqteashmd.core.units import DALTON_IN_ME, HARTREE_IN_CM1
from gqteashmd.vib.harmonic import harmonic_analysis


def _diatomic(k=0.5, r=1.4, m1=1.007825, m2=1.007825):
    coords = np.array([[0.0, 0.0, -r / 2], [0.0, 0.0, r / 2]])
    e = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 1.0]) / math.sqrt(2.0)
    hess = 2.0 * k * np.outer(e, e)  # E = k/2 (z2 - z1 - r)^2
    return np.array([m1, m2]), coords, hess


def test_diatomic_frequency_and_reduced_mass():
    masses, coords, hess = _diatomic()
    result = harmonic_analysis(masses, coords, hess)
    mu = masses[0] * masses[1] / masses.sum()
    expected = math.sqrt(0.5 / (mu * DALTON_IN_ME)) * HARTREE_IN_CM1
    assert result.linear and result.n_external == 5
    assert result.frequencies_cm1.shape == (1,)
    assert result.frequencies_cm1[0] == pytest.approx(expected, rel=1e-10)
    assert result.reduced_masses_amu[0] == pytest.approx(mu, rel=1e-10)
    assert result.external_residual_cm1 < 1e-6


def test_rotation_and_translation_do_not_change_frequencies():
    rng = np.random.default_rng(1)
    masses = np.array([15.994915, 1.007825, 1.007825])
    coords = np.array([[0.0, 0.0, 0.22], [0.0, 1.43, -0.88], [0.0, -1.43, -0.88]])
    a = rng.normal(size=(9, 9))
    hess = a @ a.T * 0.05
    # make it translation/rotation free by projecting in mass-weighted space is not needed:
    # compare the analysis of (coords, H) with that of rotated/translated (coords', R H R^T)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    big = np.kron(np.eye(3), q)
    first = harmonic_analysis(masses, coords, hess)
    second = harmonic_analysis(masses, coords @ q.T + 0.7, big @ hess @ big.T)
    np.testing.assert_allclose(first.frequencies_cm1, second.frequencies_cm1, rtol=1e-9)


def test_modes_are_orthonormal_and_unit_cartesian():
    masses, coords, hess = _diatomic()
    result = harmonic_analysis(masses, coords, hess)
    l = result.modes_mass_weighted
    np.testing.assert_allclose(l.T @ l, np.eye(l.shape[1]), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(result.modes_cartesian, axis=0), 1.0)


def test_imaginary_modes_are_negative():
    masses, coords, hess = _diatomic(k=-0.5)
    assert harmonic_analysis(masses, coords, hess).frequencies_cm1[0] < 0


def test_single_atom_has_no_vibrations():
    result = harmonic_analysis(np.array([4.002603]), np.zeros((1, 3)), np.zeros((3, 3)))
    assert result.n_external == 3 and result.frequencies_cm1.shape == (0,)


@pytest.mark.parametrize(
    "hess, message",
    [(np.zeros((5, 5)), "shape"), (np.full((6, 6), np.nan), "finite")],
)
def test_bad_hessians(hess, message):
    masses, coords, _ = _diatomic()
    with pytest.raises(ValueError, match=message):
        harmonic_analysis(masses, coords, hess)
```

`tests/vib/test_harmonic_pyscf.py`:

```python
import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.hessian import thermo

from gqteashmd.chem.masses import isotope_mass
from gqteashmd.vib.harmonic import harmonic_analysis

pytestmark = pytest.mark.pyscf


@pytest.mark.parametrize(
    "atoms",
    [
        "O 0 0 0.108140699; H 0 0.754903478 -0.464620350; H 0 -0.754903478 -0.464620350",
        "C 0 0 0; O 0 0 1.16; O 0 0 -1.16",
    ],
)
def test_matches_pyscf_harmonic_analysis(atoms):
    mol = gto.M(atom=atoms, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    h4 = mf.Hessian().kernel()
    n = mol.natm
    masses = np.array([isotope_mass(mol.atom_symbol(i)) for i in range(n)])
    ours = harmonic_analysis(masses, mol.atom_coords(), h4.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n))
    ref = thermo.harmonic_analysis(mol, h4, mass=masses)
    np.testing.assert_allclose(ours.frequencies_cm1, np.sort(ref["freq_wavenumber"].real), atol=1e-6)
    np.testing.assert_allclose(np.sort(ours.reduced_masses_amu),
                               np.sort(ref["reduced_mass"]), rtol=1e-6)
```

(If PySCF returns imaginary frequencies as complex numbers, compare against `np.where(f.imag != 0, -abs(f), f.real)`. Report what you found.)

- [ ] **Step 2: Run the tests to verify they fail.**
  Run `pytest tests/chem/test_masses.py tests/vib -v -W error`.
  Expected: they FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement.**

`src/gqteashmd/chem/masses.py`:

```python
"""Atomic masses of the most abundant isotope, in amu (dalton), for H..Rn (master plan O5).

Values: NIST atomic weights and isotopic compositions (as tabulated by PySCF's
COMMON_ISOTOPE_MASSES). Tc, Pm, Po, At and Rn have no stable isotope; the listed isotope is used.
"""

from __future__ import annotations

from gqteashmd.chem.elements import atomic_number

_MASSES = (
    1.007825, 4.002603, 7.016004, 9.012182, 11.009305, 12.0, 14.003074, 15.994915, 18.998403,
    19.99244, 22.98977, 23.985042, 26.981538, 27.976927, 30.973762, 31.972071, 34.968853,
    39.962383, 38.963707, 39.962591, 44.95591, 47.947947, 50.943964, 51.940512, 54.93805,
    55.934942, 58.9332, 57.935348, 62.929601, 63.929147, 68.925581, 73.921178, 74.921596,
    79.916522, 78.918338, 83.911507, 84.911789, 87.905614, 88.905848, 89.904704, 92.906378,
    97.905408, 98.907216, 101.90435, 102.905504, 105.903483, 106.905093, 113.903358, 114.903878,
    119.902197, 120.903818, 129.906223, 126.904468, 131.904154, 132.905447, 137.905241,
    138.906348, 139.905435, 140.907648, 141.907719, 144.912744, 151.919729, 152.921227,
    157.924101, 158.925343, 163.929171, 164.930319, 165.93029, 168.934211, 173.938858,
    174.940768, 179.946549, 180.947996, 183.950933, 186.955751, 191.961479, 192.962924,
    194.964774, 196.966552, 201.970626, 204.974412, 207.976636, 208.980383, 208.982416,
    209.987131, 222.01757,
)


def isotope_mass(symbol: str) -> float:
    """Mass (amu) of the most abundant isotope of `symbol`; ValueError for unknown symbols."""
    z = atomic_number(symbol)  # raises ValueError for unknown symbols
    return _MASSES[z - 1]
```

(Check that `atomic_number` raises `ValueError` and that its message names the symbol. If it raises something else, wrap it here so that the test's `match="Xx"` holds.)

`src/gqteashmd/vib/harmonic.py`:

```python
"""Harmonic vibrational analysis of a Cartesian Hessian (Wilson-Decius-Cross; Eckart frame).

    H_mw = M^-1/2 H M^-1/2,   M = diag(m_1, m_1, m_1, m_2, ...)  (amu)
    D    = orthonormal mass-weighted translations (sqrt(m_i) e_a) and infinitesimal rotations
           (sqrt(m_i) e_a x (r_i - R_com)), by SVD; 5 vectors for linear molecules, 3 for an atom
    Q    = orthonormal basis of the complement of D;  H_int = Q^T H_mw Q;  H_int v = lambda v
    omega = sqrt(lambda / m_e-per-amu) (atomic units);  nu~ = omega * HARTREE_IN_CM1  (cm^-1)
    l = Q v (mass-weighted, orthonormal);  x = M^-1/2 l;  reduced mass mu = 1 / |x|^2 (amu)
Imaginary frequencies (lambda < 0) are reported as negative wavenumbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from gqteashmd.core.units import DALTON_IN_ME, HARTREE_IN_CM1

_RANK_TOLERANCE = 1e-6  # relative singular value below which a rotation vector is dependent


@dataclass(frozen=True, eq=False)
class HarmonicAnalysis:
    frequencies_cm1: np.ndarray
    modes_mass_weighted: np.ndarray
    modes_cartesian: np.ndarray
    reduced_masses_amu: np.ndarray
    linear: bool
    n_external: int
    external_residual_cm1: float


def _wavenumbers(eigenvalues: np.ndarray) -> np.ndarray:
    return np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues) / DALTON_IN_ME) * HARTREE_IN_CM1


def _external_basis(masses: np.ndarray, coords: np.ndarray) -> np.ndarray:
    n = len(masses)
    root = np.sqrt(masses)
    centre = (masses[:, None] * coords).sum(axis=0) / masses.sum()
    x = coords - centre
    vectors = []
    for axis in range(3):
        v = np.zeros((n, 3))
        v[:, axis] = root
        vectors.append(v.ravel())
    for axis in range(3):
        e = np.zeros(3)
        e[axis] = 1.0
        vectors.append((np.cross(e, x) * root[:, None]).ravel())
    d = np.array(vectors).T
    u, s, _ = np.linalg.svd(d, full_matrices=False)
    return u[:, s > _RANK_TOLERANCE * s.max()]


def harmonic_analysis(
    masses_amu: ArrayLike, coords_bohr: ArrayLike, hessian: ArrayLike
) -> HarmonicAnalysis:
    masses = np.asarray(masses_amu, dtype=float)
    coords = np.asarray(coords_bohr, dtype=float)
    h = np.asarray(hessian, dtype=float)
    n3 = 3 * len(masses)
    if coords.shape != (len(masses), 3) or h.shape != (n3, n3):
        raise ValueError(
            f"inconsistent shape: {len(masses)} masses, coords {coords.shape}, hessian {h.shape}"
        )
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(coords))):
        raise ValueError("the Hessian and coordinates must be finite")
    if np.any(masses <= 0):
        raise ValueError("masses must be positive")
    h = 0.5 * (h + h.T)
    root3 = np.sqrt(np.repeat(masses, 3))
    h_mw = h / np.outer(root3, root3)
    d = _external_basis(masses, coords)
    n_external = d.shape[1]
    q_full, _ = np.linalg.qr(np.hstack([d, np.eye(n3)]))
    q = q_full[:, n_external:n3]
    eigenvalues, vectors = np.linalg.eigh(q.T @ h_mw @ q)
    modes_mw = q @ vectors
    cartesian = modes_mw / root3[:, None]
    norms = np.linalg.norm(cartesian, axis=0)
    residual = _wavenumbers(np.linalg.eigvalsh(d.T @ h_mw @ d)) if n_external else np.zeros(0)
    return HarmonicAnalysis(
        frequencies_cm1=_wavenumbers(eigenvalues),
        modes_mass_weighted=modes_mw,
        modes_cartesian=cartesian / norms if norms.size else cartesian,
        reduced_masses_amu=1.0 / norms**2 if norms.size else norms,
        linear=n_external == 5,
        n_external=n_external,
        external_residual_cm1=float(np.abs(residual).max()) if residual.size else 0.0,
    )
```

The single-atom case: `_external_basis` returns 3 translations, because the rotation vectors are zero. `s.max()` comes from the translations, so the 3 zero rotations are dropped. Check this.

- [ ] **Step 4: Run the tests to verify they pass.**
  Run `pytest tests/chem tests/vib -v -W error && mypy`.
  Expected: all pass.

- [ ] **Step 5: Lint, run the fast tier, commit**

```bash
ruff format src tests tools && ruff check src tests tools && pytest -m "not slow and not pyscf" -W error
git add src/gqteashmd/chem/masses.py src/gqteashmd/vib tests/chem/test_masses.py tests/vib
git commit -m "feat: add isotope masses and a harmonic vibrational analysis"
```

---

### Task 2: Molden output and `frequencies.h5`

**Files:**
- Create: `src/gqteashmd/vib/molden.py`, `src/gqteashmd/vib/artifacts.py`
- Modify: `pyproject.toml` (dependency `h5py>=3.9`)
- Test: `tests/vib/test_molden.py`, `tests/vib/test_artifacts.py`

**Interfaces:**
- `format_molden(symbols, coords_bohr, frequencies_cm1, modes_cartesian) -> str`. The file has these sections in order:
  1. `[Molden Format]`
  2. `[Atoms] AU`: one line per atom, `symbol index Z x y z`, in bohr.
  3. `[FREQ]`: one value per line, in cm⁻¹; imaginary modes negative.
  4. `[FR-COORD]`: one line per atom, `symbol x y z`, in bohr.
  5. `[FR-NORM-COORD]`: for each mode, a `vibration k` line followed by N lines of Cartesian displacement components.
- `write_frequencies_h5(path, *, symbols, masses_amu, coords_bohr, hessian, analysis)`:
  - h5py is imported lazily inside the function.
  - The file is written to a temporary name, then moved into place with `os.replace`.
  - File attributes: `schema_version = 1`, `gqteashmd_version`.
  - Every dataset carries a `units` attribute. The datasets are:

    | Dataset | Units |
    |---|---|
    | `symbols` (string) | — |
    | `masses` | `amu` |
    | `coordinates` | `bohr` |
    | `hessian` | `hartree/bohr^2` |
    | `frequencies` | `cm^-1` |
    | `modes_mass_weighted` | `1` |
    | `modes_cartesian` | `1` |
    | `reduced_masses` | `amu` |

- [ ] **Step 1: Write the failing tests.**

`tests/vib/test_molden.py`:

```python
import numpy as np

from gqteashmd.vib.molden import format_molden


def test_molden_sections_and_values():
    coords = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    modes = np.array([[0.0], [0.0], [-0.70710678], [0.0], [0.0], [0.70710678]])
    text = format_molden(("H", "H"), coords, np.array([-12.5]), modes)
    lines = text.splitlines()
    order = [l for l in lines if l.startswith("[")]
    assert order == ["[Molden Format]", "[Atoms] AU", "[FREQ]", "[FR-COORD]", "[FR-NORM-COORD]"]
    freq_at = lines.index("[FREQ]")
    assert float(lines[freq_at + 1]) == -12.5
    fr = lines.index("[FR-COORD]")
    assert lines[fr + 1].split()[0] == "H" and float(lines[fr + 2].split()[3]) == 0.7
    vib = lines.index("[FR-NORM-COORD]")
    assert lines[vib + 1].split() == ["vibration", "1"]
    assert float(lines[vib + 3].split()[2]) == 0.70710678
    assert lines[1 + 1].split()[:3] == ["H", "1", "1"]
```

`tests/vib/test_artifacts.py`:

```python
import h5py
import numpy as np

from gqteashmd.vib.artifacts import write_frequencies_h5
from gqteashmd.vib.harmonic import harmonic_analysis


def test_frequencies_h5_round_trip(tmp_path):
    masses = np.array([1.007825, 1.007825])
    coords = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    e = np.array([0, 0, -1, 0, 0, 1]) / np.sqrt(2)
    hess = np.outer(e, e)
    analysis = harmonic_analysis(masses, coords, hess)
    path = tmp_path / "frequencies.h5"
    write_frequencies_h5(path, symbols=("H", "H"), masses_amu=masses, coords_bohr=coords,
                         hessian=hess, analysis=analysis)
    with h5py.File(path, "r") as f:
        assert f.attrs["schema_version"] == 1
        assert [s.decode() if isinstance(s, bytes) else s for s in f["symbols"][()]] == ["H", "H"]
        np.testing.assert_allclose(f["frequencies"][()], analysis.frequencies_cm1)
        assert f["frequencies"].attrs["units"] == "cm^-1"
        assert f["hessian"].attrs["units"] == "hartree/bohr^2"
        assert f["coordinates"].attrs["units"] == "bohr"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["frequencies.h5"]
```

- [ ] **Step 2: Run the tests to verify they fail.**
  Run `pytest tests/vib -v -W error`.
  Expected: they FAIL with `ImportError`.

- [ ] **Step 3: Implement.**

`src/gqteashmd/vib/molden.py`:

```python
"""Molden vibrational file ([FREQ], [FR-COORD], [FR-NORM-COORD]); coordinates in bohr."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from gqteashmd.chem.elements import atomic_number


def format_molden(
    symbols: Sequence[str], coords_bohr: ArrayLike, frequencies_cm1: ArrayLike,
    modes_cartesian: ArrayLike,
) -> str:
    coords = np.asarray(coords_bohr, dtype=float)
    freqs = np.asarray(frequencies_cm1, dtype=float)
    modes = np.asarray(modes_cartesian, dtype=float)
    lines = ["[Molden Format]", "[Atoms] AU"]
    for index, (symbol, (x, y, z)) in enumerate(zip(symbols, coords, strict=True), start=1):
        lines.append(f"{symbol:<2} {index:5d} {atomic_number(symbol):4d} {x:18.10f} {y:18.10f} {z:18.10f}")
    lines.append("[FREQ]")
    lines += [f"{f:16.6f}" for f in freqs]
    lines.append("[FR-COORD]")
    for symbol, (x, y, z) in zip(symbols, coords, strict=True):
        lines.append(f"{symbol:<2} {x:18.10f} {y:18.10f} {z:18.10f}")
    lines.append("[FR-NORM-COORD]")
    for k in range(modes.shape[1]):
        lines.append(f"vibration {k + 1:6d}")
        for dx, dy, dz in modes[:, k].reshape(-1, 3):
            lines.append(f"{dx:18.10f} {dy:18.10f} {dz:18.10f}")
    return "\n".join(lines) + "\n"
```

(The test parses `lines[vib + 1].split() == ["vibration", "1"]`. That works because `split()` drops the padding.)

`src/gqteashmd/vib/artifacts.py`:

```python
"""frequencies.h5 (HDF5, h5py imported lazily). Every dataset carries a 'units' attribute."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from gqteashmd import __version__
from gqteashmd.vib.harmonic import HarmonicAnalysis


def write_frequencies_h5(
    path: Path, *, symbols: Sequence[str], masses_amu: ArrayLike, coords_bohr: ArrayLike,
    hessian: ArrayLike, analysis: HarmonicAnalysis,
) -> None:
    import h5py

    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["schema_version"] = 1
            f.attrs["gqteashmd_version"] = __version__
            f.create_dataset("symbols", data=np.array(list(symbols), dtype=h5py.string_dtype()))
            for name, data, units in (
                ("masses", masses_amu, "amu"),
                ("coordinates", coords_bohr, "bohr"),
                ("hessian", hessian, "hartree/bohr^2"),
                ("frequencies", analysis.frequencies_cm1, "cm^-1"),
                ("modes_mass_weighted", analysis.modes_mass_weighted, "1"),
                ("modes_cartesian", analysis.modes_cartesian, "1"),
                ("reduced_masses", analysis.reduced_masses_amu, "amu"),
            ):
                f.create_dataset(name, data=np.asarray(data, dtype=float)).attrs["units"] = units
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
```

In `pyproject.toml`, add `"h5py>=3.9"` to the dependencies. If mypy has no stubs for h5py, add `"h5py"` and `"h5py.*"` to the mypy overrides list.

- [ ] **Step 4: Run the tests to verify they pass.**
  Run `pytest tests/vib -v -W error && mypy`.
  Expected: all pass.

- [ ] **Step 5: Lint, fast tier, commit**

```bash
ruff format src tests tools && ruff check src tests tools && pytest -m "not slow and not pyscf" -W error
git add pyproject.toml src/gqteashmd/vib tests/vib
git commit -m "feat: write Molden vibration files and frequencies.h5"
```

---

### Task 3: `[optimize]`, `[frequencies]` and validators that report every violation

**Files:**
- Modify: `src/gqteashmd/config/schema.py`, `src/gqteashmd/cli.py` (`TEMPLATE`)
- Test: `tests/config/test_optimize_frequencies.py` (new) and `tests/config/test_resources.py`

**Interfaces:**
- `OptimizeSection`:
  - `enabled: bool = True`;
  - `max_steps: int = 100`, with `ge=1`;
  - `convergence: Literal["GAU", "GAU_LOOSE", "GAU_TIGHT", "GAU_VERYTIGHT", "NWCHEM_LOOSE", "TURBOMOLE", "INTERFRAG_TIGHT"] = "GAU"`.
- `FrequenciesSection`:
  - `method: Literal["auto", "analytic", "finite_difference"] = "auto"`;
  - `fd_step_bohr: float = 0.005`, with `0 < x ≤ 0.1`;
  - `imaginary_cutoff_cm1: float = 20.0`, with `ge=0`.
- `ShmdConfig` gains `optimize: OptimizeSection` and `frequencies: FrequenciesSection`, both default-constructed when absent.
- The validators of `ShmdConfig` and `ResourcesSection._fits_the_workstation` now **collect every violation** and raise one `ValueError` that joins them with "; ". Previously only the first was reported.
- A new `ShmdConfig` rule: `frequencies.method = "analytic"` with `electronic.method = "casscf"` is rejected with "no analytic SA-CASSCF Hessian in PySCF; use 'auto' or 'finite_difference'".

- [ ] **Step 1: Write the failing tests.**

`tests/config/test_optimize_frequencies.py`:

```python
import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import ConfigError

CAS = (
    '[electronic]\nmethod = "casscf"\nbasis = "sto-3g"\nsinglets = 2\n'
    "[electronic.casscf]\nncas = 2\nnelecas = 2\n"
)


def _load(tmp_path, write_input, extra):
    return load_config(write_input(tmp_path, extra=extra)).config


def test_defaults(tmp_path, write_input):
    cfg = _load(tmp_path, write_input, "")
    assert (cfg.optimize.enabled, cfg.optimize.max_steps, cfg.optimize.convergence) == (True, 100, "GAU")
    assert (cfg.frequencies.method, cfg.frequencies.fd_step_bohr,
            cfg.frequencies.imaginary_cutoff_cm1) == ("auto", 0.005, 20.0)


def test_values(tmp_path, write_input):
    cfg = _load(tmp_path, write_input,
                '[optimize]\nenabled = false\nmax_steps = 50\nconvergence = "GAU_TIGHT"\n'
                '[frequencies]\nmethod = "finite_difference"\nfd_step_bohr = 0.01\n'
                "imaginary_cutoff_cm1 = 50.0\n")
    assert not cfg.optimize.enabled and cfg.optimize.convergence == "GAU_TIGHT"
    assert cfg.frequencies.method == "finite_difference" and cfg.frequencies.fd_step_bohr == 0.01


@pytest.mark.parametrize(
    "extra, message",
    [
        ('[optimize]\nconvergence = "VERY_GOOD"\n', "convergence"),
        ("[optimize]\nmax_steps = 0\n", "max_steps"),
        ("[frequencies]\nfd_step_bohr = 0.5\n", "fd_step_bohr"),
        ('[frequencies]\nmethod = "numerical"\n', "method"),
        (CAS + '[frequencies]\nmethod = "analytic"\n', "no analytic SA-CASSCF Hessian"),
    ],
)
def test_invalid(tmp_path, write_input, extra, message):
    with pytest.raises(ConfigError, match=message):
        _load(tmp_path, write_input, extra)


def test_all_model_violations_are_reported_together(tmp_path, write_input):
    text = CAS + '[frequencies]\nmethod = "analytic"\n'
    with pytest.raises(ConfigError) as info:
        load_config(write_input(tmp_path, extra=text, molecule="multiplicity = 3"))
    assert "closed-shell" in str(info.value) and "analytic SA-CASSCF" in str(info.value)
```

Add this test to `tests/config/test_resources.py`, reusing the module's helpers:

```python
def test_every_resource_violation_is_reported(tmp_path, write_input):
    from gqteashmd.config.loader import load_config
    from gqteashmd.errors import ConfigError

    path = write_input(tmp_path, total_cores=2, total_memory_mb=1000,
                       resources="max_parallel_tasks = 4\nmemory_mb_per_task = 500")
    with pytest.raises(ConfigError) as info:
        load_config(path)
    text = str(info.value)
    assert "exceeds total_cores" in text and "exceeds total_memory_mb" in text
```

- [ ] **Step 2: Run them to verify they fail.**
  Run `pytest tests/config -v -W error`.
  Expected: FAIL. The sections are unknown, and only the first violation is reported.

- [ ] **Step 3: Implement** in `schema.py`.

  - **New sections.** Add `OptimizeSection` and `FrequenciesSection` per the Interfaces. Give `fd_step_bohr` `Field(default=0.005, gt=0, le=0.1)` and `imaginary_cutoff_cm1` `Field(default=20.0, ge=0)`.
  - **New `ShmdConfig` fields:**
    `optimize: OptimizeSection = Field(default_factory=OptimizeSection)` and
    `frequencies: FrequenciesSection = Field(default_factory=FrequenciesSection)`.
  - **Collect every `ShmdConfig` violation.** Replace `_closed_shell_for_electronic_structure` with:

```python
    @model_validator(mode="after")
    def _consistent(self) -> ShmdConfig:
        problems: list[str] = []
        if self.electronic is not None and self.molecule.multiplicity != 1:
            problems.append(
                "the PySCF backend supports closed-shell singlet molecules only "
                "(molecule.multiplicity = 1)"
            )
        if (
            self.electronic is not None
            and self.electronic.method == "casscf"
            and self.frequencies.method == "analytic"
        ):
            problems.append(
                "frequencies.method = 'analytic': there is no analytic SA-CASSCF Hessian in "
                "PySCF; use 'auto' or 'finite_difference'"
            )
        if problems:
            raise ValueError("; ".join(problems))
        return self
```

  - **Collect every resource violation.** In `ResourcesSection._fits_the_workstation`, collect the messages into `problems` the same way, and raise once with `"; ".join(problems)`.
  - **Check the multi-violation messages.** If any other model validator in the file (`ElectronicSection._method_options`, `CasscfSection`) raises on its first problem, convert it to collect as well. Keep every existing message text, because tests match it.

In `cli.py`, update `TEMPLATE` so a new user gets a runnable workflow:
- replace the commented `[electronic]` block with an **active** one: `method = "tda"`, `basis = "def2-svp"`, `functional = "b3lyp"`, `singlets = 3`;
- keep a commented CASSCF alternative;
- add `[optimize]` and `[frequencies]` blocks, showing the defaults with one-line comments.

Existing tests check that the template validates. Report any that you adapt.

- [ ] **Step 4: Run the tests to verify they pass.**
  Run `pytest tests/config tests/test_cli.py -v -W error && mypy`.
  Expected: all pass.

- [ ] **Step 5: Lint, run the fast tier, commit**

```bash
ruff format src tests tools && ruff check src tests tools && pytest -m "not slow and not pyscf" -W error
git add src/gqteashmd/config/schema.py src/gqteashmd/cli.py tests/config
git commit -m "feat: add the [optimize] and [frequencies] input sections and report every violation"
```

---

### Task 4: Ground-state Hessian in the backend; factory options

**Files:**
- Modify: `src/gqteashmd/electronic/base.py`, `src/gqteashmd/electronic/pyscf_backend.py`, `src/gqteashmd/electronic/factory.py`
- Test: `tests/electronic/test_hessian.py` (new, `pytestmark = pytest.mark.pyscf`), `tests/electronic/test_factory.py`

**Interfaces:**
- **`Capabilities.hessian: bool = False`**: a new field with a default, so existing callers still work.
- **`ElectronicBackend.ground_state_hessian(coords_bohr) -> np.ndarray`**: (3N, 3N) in Eh/bohr². It is a non-abstract method whose default raises `NotImplementedError(f"{type(self).__name__} has no analytic Hessian")`.
- **`PySCFBackend`:**
  - `capabilities.hessian` is `method != "casscf"`.
  - `ground_state_hessian` runs a fresh SCF (ladder, no previous density) at `coords` and calls `mf.Hessian().kernel()`, wrapped with `wrap_failure("The analytic SCF Hessian", exc)`.
  - It does **not** touch `_snapshot` or `_live`.
  - For CASSCF it raises `ValueError` ("no analytic Hessian for method = 'casscf'").
- **`make_backend(config, molecule, *, settings=None, log_file=None, threads=None, memory_mb=None, scratch_dir=None)`**: `settings` defaults to `config.electronic`, and `scratch_dir` is passed to `PySCFBackend`.
- **`ground_state_settings(settings: ElectronicSection) -> ElectronicSection`**: for `casscf` it returns `settings` unchanged. Otherwise it returns `settings.model_copy(update={"singlets": 1})`.

- [ ] **Step 1: Write the failing tests** in `tests/electronic/test_hessian.py`:

```python
import numpy as np
import pytest

from gqteashmd.core.states import StateLabel
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.factory import ground_state_settings
from gqteashmd.electronic.pyscf_backend import PySCFBackend

pytestmark = pytest.mark.pyscf
S0 = StateLabel(1, 0)
WATER = ("O", "H", "H")
WATER_BOHR = np.array([[0.0, 0.0, 0.2043], [0.0, 1.4266, -0.8780], [0.0, -1.4266, -0.8780]])


def _settings(make_settings, **kw):
    fields = {"method": "tda", "basis": "sto-3g", "functional": "hf", "singlets": 3, "casscf": None}
    fields.update(kw)
    return make_settings(**fields)


def test_analytic_hessian_matches_finite_differences_of_gradients(make_settings):
    settings = ground_state_settings(_settings(make_settings, scf_conv_tol=1e-12))
    assert settings.singlets == 1
    backend = PySCFBackend(MoleculeSpec(WATER), settings)
    assert backend.capabilities.hessian is True
    h = backend.ground_state_hessian(WATER_BOHR)
    assert h.shape == (9, 9)
    step = 1e-3
    fd = np.zeros((9, 9))
    for k in range(9):
        plus, minus = WATER_BOHR.copy().ravel(), WATER_BOHR.copy().ravel()
        plus[k] += step
        minus[k] -= step
        g = []
        for c in (plus, minus):
            fresh = PySCFBackend(MoleculeSpec(WATER), settings)
            g.append(fresh.compute(Request(c.reshape(3, 3), gradients=frozenset({S0}))).gradients[S0].ravel())
        fd[:, k] = (g[0] - g[1]) / (2 * step)
    assert np.abs(h - 0.5 * (fd + fd.T)).max() < 1e-4


def test_hessian_does_not_disturb_the_snapshot(make_settings):
    backend = PySCFBackend(MoleculeSpec(WATER), ground_state_settings(_settings(make_settings)))
    backend.compute(Request(WATER_BOHR))
    snapshot = backend._snapshot
    backend.ground_state_hessian(WATER_BOHR + 0.05)
    assert backend._snapshot is snapshot


def test_casscf_has_no_analytic_hessian(ethylene_symbols, ethylene_bohr, make_settings):
    settings = make_settings(basis="sto-3g")
    assert ground_state_settings(settings) is settings  # the SA defines S0: keep all states
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings)
    assert backend.capabilities.hessian is False
    with pytest.raises(ValueError, match="casscf"):
        backend.ground_state_hessian(ethylene_bohr)
```

Extend `tests/electronic/test_factory.py` with three checks:
- `make_backend(cfg, spec, settings=other)` uses `other`, for example a different basis, visible in `meta` after a compute;
- `scratch_dir` reaches the backend: `backend._scratch_dir == path`;
- the default `ElectronicBackend.ground_state_hessian` raises `NotImplementedError`. Use a minimal subclass with no-op abstract methods.

- [ ] **Step 2: Run them to verify they fail.**
  Run: `pytest tests/electronic/test_hessian.py tests/electronic/test_factory.py -v -W error`
  Expected: FAIL. `ground_state_settings`, `capabilities.hessian` and `ground_state_hessian` are missing.

- [ ] **Step 3: Implement.**
  - **`base.py`.** Add the field to `Capabilities`: `hessian: bool = False`. It must come after the existing fields. Add the non-abstract `ground_state_hessian` to `ElectronicBackend`, with a docstring: "Cartesian Hessian of S0 (Eh/bohr²) at `coords_bohr`; independent of the step history."
  - **`pyscf_backend.py`.** Add `hessian=self._settings.method != "casscf"` in `capabilities`. Add this method:

```python
    def ground_state_hessian(self, coords_bohr: ArrayLike) -> np.ndarray:
        if self._settings.method == "casscf":
            raise ValueError(
                "There is no analytic Hessian for method = 'casscf'; use finite differences of "
                "S0 gradients (frequencies.method = 'finite_difference')."
            )
        coords = np.asarray(coords_bohr, dtype=float)
        if coords.shape != (len(self._molecule.symbols), 3):
            raise ValueError(f"coords_bohr must have shape ({len(self._molecule.symbols)}, 3)")
        with self._environment() as stdout:
            mol = self._build_mol(coords, stdout)
            mf, _ = self._run_scf(mol, None)
            try:
                h4 = mf.Hessian().kernel()
            except Exception as exc:
                raise wrap_failure("The analytic SCF Hessian", exc) from exc
        n = len(self._molecule.symbols)
        return np.asarray(h4, dtype=float).transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
```

  (Import `ArrayLike` from `numpy.typing` if it is not already imported.)
  - **`factory.py`.** Add `settings` and `scratch_dir` to `make_backend`, pass `settings or config.electronic`, and keep the `ConfigError` when both are `None`. Add `ground_state_settings` with the docstring "S0 only. SCF methods need no excited states for S0. SA-CASSCF keeps its state average, which defines the S0 surface."

- [ ] **Step 4: Run the tests to verify they pass.**
  Run: `pytest tests/electronic -v -W error -m "not slow" && mypy`
  Expected: all pass. Report the measured `max|H − FD|`.

- [ ] **Step 5: Lint, commit**

```bash
ruff format src tests tools && ruff check src tests tools
git add src/gqteashmd/electronic tests/electronic
git commit -m "feat: add the analytic SCF ground-state Hessian and backend factory options"
```

---

### Task 5: geomeTRIC driver

**Files:**
- Create: `src/gqteashmd/geomopt.py`
- Modify: `pyproject.toml` (dependency `geometric>=1.0`), `environment.yml` (`- geometric=1.1.*`)
- Test: `tests/test_geomopt.py`. These are pure tests that use a model potential; they need geomeTRIC, which is installed.

**Interfaces:**
- `OptimizationResult` is a frozen dataclass with these fields:
  - `coords_bohr` (N, 3);
  - `energy_hartree`;
  - `converged: bool`;
  - `n_gradients: int`;
  - `path_bohr: list[np.ndarray]`, one entry per gradient call;
  - `energies_hartree: list[float]`.
- `optimize_geometry(symbols, coords_bohr, energy_and_gradient, *, max_steps, convergence, work_dir) -> OptimizationResult`:
  - `energy_and_gradient(coords (N, 3) bohr)` returns `(E in Eh, gradient (N, 3) in Eh/bohr)`.
  - geomeTRIC is imported inside the function.
  - Its files go to `work_dir` (prefix `work_dir / "opt"`).
  - Non-convergence (`GeomOptNotConvergedError`) returns `converged=False` with the last evaluated geometry.
  - Exceptions raised by `energy_and_gradient` propagate unchanged.

- [ ] **Step 1: Write the failing tests** in `tests/test_geomopt.py`:

```python
import numpy as np
import pytest

from gqteashmd.geomopt import optimize_geometry

R0, K = 1.8, 0.4  # bohr, Eh/bohr^2


def _bonds(coords):
    """Harmonic bonds 0-1 and 0-2 plus a weak 1-2 bond (a bent triatomic model)."""
    energy, grad = 0.0, np.zeros_like(coords)
    for (i, j), (r0, k) in {(0, 1): (R0, K), (0, 2): (R0, K), (1, 2): (2.9, 0.05)}.items():
        d = coords[i] - coords[j]
        r = np.linalg.norm(d)
        energy += 0.5 * k * (r - r0) ** 2
        g = k * (r - r0) * d / r
        grad[i] += g
        grad[j] -= g
    return energy, grad


START = np.array([[0.0, 0.0, 0.0], [0.0, 1.6, 0.9], [0.0, -1.5, 1.0]])


def test_converges_to_the_model_minimum(tmp_path):
    result = optimize_geometry(("O", "H", "H"), START, _bonds, max_steps=100, convergence="GAU",
                               work_dir=tmp_path / "geo")
    assert result.converged
    d01 = np.linalg.norm(result.coords_bohr[0] - result.coords_bohr[1])
    d02 = np.linalg.norm(result.coords_bohr[0] - result.coords_bohr[2])
    assert abs(d01 - R0) < 0.02 and abs(d02 - R0) < 0.02
    assert result.energy_hartree < _bonds(START)[0]
    assert result.n_gradients == len(result.path_bohr) == len(result.energies_hartree) >= 2
    assert np.abs(_bonds(result.coords_bohr)[1]).max() < 1e-3


def test_reports_non_convergence(tmp_path):
    result = optimize_geometry(("O", "H", "H"), START, _bonds, max_steps=1, convergence="GAU_VERYTIGHT",
                               work_dir=tmp_path / "geo")
    assert result.converged is False
    np.testing.assert_allclose(result.coords_bohr, result.path_bohr[-1])


def test_energy_errors_propagate(tmp_path):
    def broken(coords):
        raise RuntimeError("electronic structure exploded")

    with pytest.raises(RuntimeError, match="exploded"):
        optimize_geometry(("O", "H", "H"), START, broken, max_steps=10, convergence="GAU",
                          work_dir=tmp_path / "geo")
```

- [ ] **Step 2: Run the tests to verify they fail.**
  Run `pytest tests/test_geomopt.py -v -W error`.
  Expected: they FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `src/gqteashmd/geomopt.py`:

```python
"""Geometry optimization with geomeTRIC (imported lazily) driving any energy+gradient function."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from gqteashmd.core.units import ANGSTROM_TO_BOHR, BOHR_IN_ANGSTROM

EnergyGradient = Callable[[np.ndarray], tuple[float, np.ndarray]]


@dataclass(frozen=True, eq=False)
class OptimizationResult:
    coords_bohr: np.ndarray
    energy_hartree: float
    converged: bool
    n_gradients: int
    path_bohr: list[np.ndarray] = field(default_factory=list)
    energies_hartree: list[float] = field(default_factory=list)


def optimize_geometry(
    symbols: Sequence[str], coords_bohr: ArrayLike, energy_and_gradient: EnergyGradient, *,
    max_steps: int, convergence: str, work_dir: Path,
) -> OptimizationResult:
    import geometric.engine
    import geometric.molecule
    import geometric.optimize
    from geometric.errors import GeomOptNotConvergedError

    path: list[np.ndarray] = []
    energies: list[float] = []

    class _Engine(geometric.engine.Engine):  # type: ignore[misc]
        def calc_new(self, coords: Any, dirname: Any) -> dict[str, Any]:
            c = np.asarray(coords, dtype=float).reshape(-1, 3)
            energy, gradient = energy_and_gradient(c)
            path.append(c.copy())
            energies.append(float(energy))
            return {"energy": float(energy), "gradient": np.asarray(gradient, dtype=float).ravel()}

    molecule = geometric.molecule.Molecule()
    molecule.elem = list(symbols)
    molecule.xyzs = [np.asarray(coords_bohr, dtype=float) * BOHR_IN_ANGSTROM]
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # geomeTRIC internals
            result = geometric.optimize.run_optimizer(
                customengine=_Engine(molecule), check=1, input=str(work_dir / "opt"),
                maxiter=max_steps, convergence_set=convergence,
            )
    except GeomOptNotConvergedError:
        return OptimizationResult(path[-1], energies[-1], False, len(path), path, energies)
    final = np.asarray(result.xyzs[-1], dtype=float) * ANGSTROM_TO_BOHR
    return OptimizationResult(final, float(result.qm_energies[-1]), True, len(path), path, energies)
```

Verify these points, and report if any of them is different:
- geomeTRIC raises `GeomOptNotConvergedError` when `maxiter` is reached.
- geomeTRIC does not swallow the engine's exception. If it wraps it in `EngineError`, re-raise the original `__cause__`/`__context__`.
- `-W error` is clean.
- If geomeTRIC changes the logging configuration of the root logger (`logIni`), protect gqteashmd's handlers.

Also:
- In `pyproject.toml`, add `"geometric>=1.0"`. In `environment.yml`, add `- geometric=1.1.*`.
- mypy: add `"geometric"` and `"geometric.*"` to the ignore-missing-imports override.
- Extend the lazy-import subprocess test in `tests/test_package.py` so that `geometric` and `h5py` must not appear in `sys.modules` after `import gqteashmd.cli, gqteashmd.worker`.

- [ ] **Step 4: Run the tests to verify they pass.**
  Run `pytest tests/test_geomopt.py tests/test_package.py -v -W error && mypy`.
  Expected: all pass.

- [ ] **Step 5: Lint, run the fast tier, commit**

```bash
ruff format src tests tools && ruff check src tests tools && pytest -m "not slow and not pyscf" -W error
git add pyproject.toml environment.yml src/gqteashmd/geomopt.py tests/test_geomopt.py tests/test_package.py
git commit -m "feat: add a geomeTRIC geometry-optimization driver for any energy and gradient"
```

---

### Task 6: The `optimize` stage

**Files:**
- Create: `src/gqteashmd/stages/geometry.py`, `src/gqteashmd/stages/optimize.py`
- Test: `tests/stages/test_optimize.py` (`pytestmark = pytest.mark.pyscf`)

**Interfaces:**
- `stages/geometry.py`:
  - **`prepared_molecule(prepare_dir: Path) -> tuple[MoleculeSpec, np.ndarray]`** reads `molecule.json`: symbols, charge and multiplicity, plus `coords_angstrom`, which it converts to bohr.
  - **`working_geometry(ctx) -> tuple[MoleculeSpec, np.ndarray, str]`** returns the optimized geometry from `ctx.upstream["optimize"] / "optimize.json"` (`coords_bohr`), with source `"optimize"`. If the optimize stage is not upstream, which means it was disabled or skipped, it returns the geometry from `prepare`, with source `"prepare"`.
  - **`stage_backend(ctx, stage_name, molecule, task_dir, *, settings=None) -> ElectronicBackend`** wraps `make_backend`. It uses:
    - the stage's effective resources (threads, memory);
    - `log_file = task_dir / "electronic.log"`;
    - `scratch_dir = task_dir / "scratch"`;
    - `settings = ground_state_settings(config.electronic)` unless `settings` is given.
- **`OptimizeStage`** has these attributes:
  - `name = "optimize"`, `version = 1`;
  - `depends_on = ("prepare",)`;
  - `config_sections = ("electronic", "optimize")`;
  - `required_modules = ("pyscf>=2.6", "geometric>=1.0")`;
  - `enabled(config)` returns `config.optimize.enabled`.

  It runs one task, `"optimize"`. That task:
  - optimizes S0 using **one** backend for the whole path;
  - writes `optimized.xyz` (Å), `optimization_path.xyz` (Å, every evaluated geometry with its energy in the comment) and `optimize.json` into the task directory;
  - returns `{"energy_hartree", "converged", "n_gradients", "max_gradient_hartree_per_bohr"}`.

  If the optimization does not converge, the task first writes `last.xyz`. It then raises a non-retryable `TaskError` whose message reads: "The geometry optimization did not converge in N steps (convergence = 'GAU'). Increase [optimize] max_steps, start from a better geometry, or loosen [optimize] convergence; the last geometry is in …/last.xyz".

  `finalize` copies `optimized.xyz`, `optimization_path.xyz` and `optimize.json` into `ctx.stage_dir`. The JSON contains:
  - `schema_version = 1`;
  - `energy_hartree`, `converged`, `n_gradients`, `max_gradient_hartree_per_bohr`;
  - `convergence`, `max_steps`;
  - `symbols`, `coords_bohr`, `coords_angstrom`.

- [ ] **Step 1: Write the failing tests** in `tests/stages/test_optimize.py`. The stage is driven directly with a hand-built `StageContext`, with no controller and no subprocess.

```python
import json

import numpy as np
import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import TaskError
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.stages.base import StageContext, TaskSpec
from gqteashmd.stages.optimize import OptimizeStage
from gqteashmd.stages.prepare import PrepareStage

pytestmark = pytest.mark.pyscf
ELECTRONIC = '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 2\n'


def _contexts(tmp_path, write_input, extra=""):
    cfg = load_config(write_input(tmp_path, extra=ELECTRONIC + extra)).config
    layout = ProjectLayout(cfg.project.directory)
    prepare_dir = layout.stage_dir(0, "prepare")
    prepare_dir.mkdir(parents=True)
    prepare = PrepareStage()
    pctx = StageContext(cfg, layout, prepare_dir, {})
    prepare.finalize(pctx, {"molecule": prepare.run_task(pctx, TaskSpec("molecule"), prepare_dir)})
    octx = StageContext(cfg, layout, layout.stage_dir(1, "optimize"), {"prepare": prepare_dir})
    octx.stage_dir.mkdir(parents=True)
    return cfg, octx


def test_optimize_stage_produces_a_minimum(tmp_path, write_input):
    cfg, ctx = _contexts(tmp_path, write_input)
    stage = OptimizeStage()
    assert stage.enabled(cfg) and [t.task_id for t in stage.plan(ctx)] == ["optimize"]
    task_dir = ctx.stage_dir / "tasks" / "optimize"
    task_dir.mkdir(parents=True)
    result = stage.run_task(ctx, TaskSpec("optimize"), task_dir)
    assert result["converged"] and result["max_gradient_hartree_per_bohr"] < 1e-3
    json.dumps(result)
    stage.finalize(ctx, {"optimize": result})
    data = json.loads((ctx.stage_dir / "optimize.json").read_text())
    assert data["schema_version"] == 1 and len(data["coords_bohr"]) == 3
    assert (ctx.stage_dir / "optimized.xyz").read_text().splitlines()[0].strip() == "3"
    path_lines = (ctx.stage_dir / "optimization_path.xyz").read_text().splitlines()
    assert path_lines[0].strip() == "3" and "energy" in path_lines[1].lower()


def test_non_convergence_is_a_non_retryable_task_error(tmp_path, write_input):
    _, ctx = _contexts(tmp_path, write_input,
                       '[optimize]\nmax_steps = 1\nconvergence = "GAU_VERYTIGHT"\n')
    task_dir = ctx.stage_dir / "tasks" / "optimize"
    task_dir.mkdir(parents=True)
    with pytest.raises(TaskError, match="did not converge in 1 steps") as info:
        OptimizeStage().run_task(ctx, TaskSpec("optimize"), task_dir)
    assert info.value.retryable is False and (task_dir / "last.xyz").exists()


def test_disabled_optimize(tmp_path, write_input):
    cfg, _ = _contexts(tmp_path, write_input, "[optimize]\nenabled = false\n")
    assert OptimizeStage().enabled(cfg) is False
```

(If PrepareStage's `run_task` and `finalize` signatures differ, adapt the helper, but keep its purpose: make a real prepare output, then run optimize. `write_input` writes a water geometry. If the 1-step run happens to converge, adjust the start geometry or the setting so that it does not, and report the change.)

- [ ] **Step 2: Run the tests to verify they fail.**
  Run `pytest tests/stages/test_optimize.py -v -W error`.
  Expected: they FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement.**

`src/gqteashmd/stages/geometry.py`:

```python
"""Geometry and backend helpers shared by the quantum-chemistry stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gqteashmd.config.schema import ElectronicSection
from gqteashmd.core.units import ANGSTROM_TO_BOHR
from gqteashmd.electronic.base import ElectronicBackend, MoleculeSpec
from gqteashmd.electronic.factory import ground_state_settings, make_backend
from gqteashmd.stages.base import StageContext
from gqteashmd.util.fileio import read_json


def prepared_molecule(prepare_dir: Path) -> tuple[MoleculeSpec, np.ndarray]:
    data = read_json(Path(prepare_dir) / "molecule.json")
    spec = MoleculeSpec(tuple(data["symbols"]), int(data["charge"]), int(data["multiplicity"]))
    return spec, np.asarray(data["coords_angstrom"], dtype=float) * ANGSTROM_TO_BOHR


def working_geometry(ctx: StageContext) -> tuple[MoleculeSpec, np.ndarray, str]:
    """The optimized geometry if `optimize` ran, else the prepared input geometry (bohr)."""
    spec, coords = prepared_molecule(ctx.upstream_dir("prepare"))
    if "optimize" in ctx.upstream:
        data = read_json(ctx.upstream["optimize"] / "optimize.json")
        return spec, np.asarray(data["coords_bohr"], dtype=float), "optimize"
    return spec, coords, "prepare"


def stage_backend(
    ctx: StageContext, stage_name: str, molecule: MoleculeSpec, task_dir: Path, *,
    settings: ElectronicSection | None = None,
) -> ElectronicBackend:
    electronic = ctx.config.electronic
    assert electronic is not None  # guaranteed by check_stage_inputs
    res = ctx.config.resources.for_stage(stage_name)
    return make_backend(
        ctx.config, molecule, settings=settings or ground_state_settings(electronic),
        log_file=task_dir / "electronic.log", threads=res.threads_per_task,
        memory_mb=res.memory_mb_per_task, scratch_dir=task_dir / "scratch",
    )
```

`src/gqteashmd/stages/optimize.py`:

```python
"""Stage 1 (optional): optimize the S0 geometry on the [electronic] surface with geomeTRIC."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.chem.xyz import Geometry, format_xyz
from gqteashmd.config.schema import ShmdConfig
from gqteashmd.core.units import BOHR_IN_ANGSTROM
from gqteashmd.electronic.base import Request
from gqteashmd.errors import TaskError
from gqteashmd.geomopt import optimize_geometry
from gqteashmd.stages.base import Stage, StageContext, TaskSpec
from gqteashmd.stages.geometry import prepared_molecule, stage_backend
from gqteashmd.util.fileio import write_json_atomic, write_text_atomic

ARTIFACTS = ("optimized.xyz", "optimization_path.xyz", "optimize.json")


def _xyz(symbols: tuple[str, ...], coords_bohr: np.ndarray, comment: str) -> str:
    angstrom = np.asarray(coords_bohr) * BOHR_IN_ANGSTROM
    return format_xyz(Geometry(symbols, tuple((float(x), float(y), float(z)) for x, y, z in angstrom),
                               comment=comment))


class OptimizeStage(Stage):
    name = "optimize"
    version = 1
    depends_on = ("prepare",)
    config_sections = ("electronic", "optimize")
    required_modules = ("pyscf>=2.6", "geometric>=1.0")

    def enabled(self, config: ShmdConfig) -> bool:
        return config.optimize.enabled

    def plan(self, ctx: StageContext) -> list[TaskSpec]:
        return [TaskSpec("optimize")]

    def run_task(self, ctx: StageContext, task: TaskSpec, task_dir: Path) -> dict[str, Any]:
        options = ctx.config.optimize
        molecule, start = prepared_molecule(ctx.upstream_dir("prepare"))
        backend = stage_backend(ctx, self.name, molecule, task_dir)
        s0 = backend.states[0]

        def energy_and_gradient(coords: np.ndarray) -> tuple[float, np.ndarray]:
            result = backend.compute(Request(coords, gradients=frozenset({s0})))
            return result.energies[s0], result.gradients[s0]

        opt = optimize_geometry(
            molecule.symbols, start, energy_and_gradient, max_steps=options.max_steps,
            convergence=options.convergence, work_dir=task_dir / "geometric",
        )
        path_text = "".join(
            _xyz(molecule.symbols, c, f"step {i} energy {e:.10f} Eh")
            for i, (c, e) in enumerate(zip(opt.path_bohr, opt.energies_hartree, strict=True))
        )
        write_text_atomic(task_dir / "optimization_path.xyz", path_text)
        if not opt.converged:
            write_text_atomic(task_dir / "last.xyz", _xyz(molecule.symbols, opt.coords_bohr, "last"))
            raise TaskError(
                f"The geometry optimization did not converge in {options.max_steps} steps "
                f"(convergence = '{options.convergence}'). Increase [optimize] max_steps, start from "
                "a better geometry, or loosen [optimize] convergence; the last geometry is in "
                f"{task_dir / 'last.xyz'}",
                retryable=False,
            )
        _, final_gradient = energy_and_gradient(opt.coords_bohr)
        max_gradient = float(np.abs(final_gradient).max())
        write_text_atomic(task_dir / "optimized.xyz",
                          _xyz(molecule.symbols, opt.coords_bohr, f"energy {opt.energy_hartree:.10f} Eh"))
        write_json_atomic(task_dir / "optimize.json", {
            "schema_version": 1,
            "energy_hartree": opt.energy_hartree,
            "converged": True,
            "n_gradients": opt.n_gradients,
            "max_gradient_hartree_per_bohr": max_gradient,
            "convergence": options.convergence,
            "max_steps": options.max_steps,
            "symbols": list(molecule.symbols),
            "coords_bohr": opt.coords_bohr.tolist(),
            "coords_angstrom": (opt.coords_bohr * BOHR_IN_ANGSTROM).tolist(),
        })
        return {"energy_hartree": opt.energy_hartree, "converged": True,
                "n_gradients": opt.n_gradients, "max_gradient_hartree_per_bohr": max_gradient}

    def finalize(self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]) -> None:
        task_dir = ctx.layout.task_dir(ctx.stage_dir, "optimize")
        for name in ARTIFACTS:
            shutil.copyfile(task_dir / name, ctx.stage_dir / name)
```

Notes:
- The final gradient check makes one extra gradient call at the converged geometry. It is the same backend and the next point in the path, so it is cheap.
- If `format_xyz`'s `Geometry` signature differs, adapt `_xyz`.
- Copy the files atomically: copy to a temporary name, then `os.replace`, or use `write_text_atomic` with the read text.

- [ ] **Step 4: Run the tests to verify they pass.**
  Run `pytest tests/stages/test_optimize.py -v -W error && mypy`.
  Expected: all pass. Report the number of gradients and the time.

- [ ] **Step 5: Lint and commit**

```bash
ruff format src tests tools && ruff check src tests tools
git add src/gqteashmd/stages/geometry.py src/gqteashmd/stages/optimize.py tests/stages/test_optimize.py
git commit -m "feat: add the optimize stage"
```

---

### Task 7: The `frequencies` stage

**Files:**
- Create: `src/gqteashmd/stages/frequencies.py`
- Test: `tests/stages/test_frequencies.py` (`pytestmark = pytest.mark.pyscf`)

**Interfaces:**
- **Attributes** of `FrequenciesStage`:
  - `name = "frequencies"`, `version = 1`;
  - `depends_on = ("prepare", "optimize")`;
  - `config_sections = ("electronic", "frequencies")`;
  - `required_modules = ("pyscf>=2.6", "h5py>=3.9")`.
- **Method selection.** `resolved_method(config)` returns `"analytic"` when `frequencies.method == "analytic"`, or when it is `"auto"` and `electronic.method != "casscf"`. Otherwise it returns `"finite_difference"`.
- **Plan:**
  - Analytic: one task, `TaskSpec("hessian")`.
  - Finite differences: `TaskSpec("reference")` plus, for k = 0 … 3N−1, `TaskSpec(f"fd_{k:03d}_plus", {"coordinate": k, "sign": 1})` and `TaskSpec(f"fd_{k:03d}_minus", {"coordinate": k, "sign": -1})`. N comes from the prepared `molecule.json`.
- **`run_task`.** Every task uses the `working_geometry` and a **fresh** backend, built with `stage_backend`.
  - `hessian`: computes the S0 energy and gradient at the geometry, then `ground_state_hessian`. It saves `hessian.npy` and returns `{"energy_hartree", "gradient": list}`.
  - `reference`: returns `{"energy_hartree", "gradient": list}`.
  - Displaced tasks: compute the gradient at `coords.ravel()[k] += sign·fd_step_bohr` and return `{"gradient": list}`.
- **`finalize`:**
  - Build the Hessian:
    - analytic: load `hessian.npy`;
    - finite differences: `H[:, k] = (g⁺ − g⁻) / (2h)`, then symmetrize `½ (H + Hᵀ)`.
  - Masses come from `isotope_mass`. Run `harmonic_analysis`.
  - **Always write** `frequencies.h5`, `frequencies.molden` and `frequencies.json`. Then apply the policy:
    - ν < −cutoff: raise `StageFailedError`, listing each such mode as "mode k: ν cm⁻¹". Add advice: re-optimize with `[optimize] convergence = "GAU_TIGHT"`, check the structure, or raise `[frequencies] imaginary_cutoff_cm1` if the mode is known to be a harmless numerical artefact.
    - −cutoff ≤ ν < 0: log a warning and record the mode in `small_imaginary_modes`.
    - `external_residual_cm1 > 100`: warning.
    - Maximum |gradient| > 1e-3 Eh/bohr: warning, "not a stationary point; frequencies are approximate".
  - **`frequencies.json` contents:**
    - `schema_version`, `method`, `geometry_source`;
    - `energy_hartree`, `max_gradient_hartree_per_bohr`;
    - `linear`, `n_external`, `external_residual_cm1`;
    - `frequencies_cm1`, `reduced_masses_amu`;
    - `imaginary_modes`, `small_imaginary_modes`, `warnings`.

- [ ] **Step 1: Write the failing tests** in `tests/stages/test_frequencies.py`. Build the prepare stage directly, as in Task 6; the shared helper may go in a `tests/stages/conftest.py`.

```python
import json

import h5py
import numpy as np
import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import StageFailedError
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.stages.base import StageContext, TaskSpec
from gqteashmd.stages.frequencies import FrequenciesStage, resolved_method
from gqteashmd.stages.prepare import PrepareStage

pytestmark = pytest.mark.pyscf
HF = '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 2\n'
# RHF/STO-3G water minimum (from a geomeTRIC GAU optimization), Angstrom
WATER_MIN = """3
water RHF/STO-3G minimum
O    0.000000000   0.000000000   0.127368896
H    0.000000000   0.751652800  -0.503684448
H    0.000000000  -0.751652800  -0.503684448
"""


def _run_stage(tmp_path, write_input, extra, xyz=None):
    path = write_input(tmp_path, extra=HF + extra)
    if xyz is not None:
        (tmp_path / "water.xyz").write_text(xyz)
    cfg = load_config(path).config
    layout = ProjectLayout(cfg.project.directory)
    prep_dir = layout.stage_dir(0, "prepare")
    prep_dir.mkdir(parents=True)
    prepare = PrepareStage()
    pctx = StageContext(cfg, layout, prep_dir, {})
    prepare.finalize(pctx, {"molecule": prepare.run_task(pctx, TaskSpec("molecule"), prep_dir)})
    ctx = StageContext(cfg, layout, layout.stage_dir(2, "frequencies"), {"prepare": prep_dir})
    ctx.stage_dir.mkdir(parents=True)
    stage = FrequenciesStage()
    results = {}
    for task in stage.plan(ctx):
        task_dir = layout.task_dir(ctx.stage_dir, task.task_id)
        task_dir.mkdir(parents=True)
        results[task.task_id] = stage.run_task(ctx, task, task_dir)
    return cfg, ctx, stage, results


def test_analytic_and_finite_difference_agree(tmp_path, write_input):
    cfg_a, ctx_a, stage, res_a = _run_stage(tmp_path / "a", write_input, "", WATER_MIN)
    assert resolved_method(cfg_a) == "analytic" and list(res_a) == ["hessian"]
    stage.finalize(ctx_a, res_a)
    cfg_f, ctx_f, _, res_f = _run_stage(tmp_path / "f", write_input,
                                        '[frequencies]\nmethod = "finite_difference"\n', WATER_MIN)
    assert len(res_f) == 1 + 6 * 3
    stage.finalize(ctx_f, res_f)
    fa = json.loads((ctx_a.stage_dir / "frequencies.json").read_text())
    ff = json.loads((ctx_f.stage_dir / "frequencies.json").read_text())
    assert len(fa["frequencies_cm1"]) == 3 and fa["imaginary_modes"] == []
    np.testing.assert_allclose(fa["frequencies_cm1"], ff["frequencies_cm1"], atol=1.0)
    assert fa["geometry_source"] == "prepare" and fa["method"] == "analytic"
    with h5py.File(ctx_a.stage_dir / "frequencies.h5", "r") as f:
        np.testing.assert_allclose(f["frequencies"][()], fa["frequencies_cm1"])
    assert "[FR-NORM-COORD]" in (ctx_a.stage_dir / "frequencies.molden").read_text()


def test_imaginary_mode_beyond_the_cutoff_stops_the_stage(tmp_path, write_input):
    linear_water = "3\nlinear water (a saddle)\nO 0 0 0\nH 0 0 0.96\nH 0 0 -0.96\n"
    _, ctx, stage, results = _run_stage(tmp_path, write_input, "", linear_water)
    with pytest.raises(StageFailedError, match=r"mode \d+: -\d"):
        stage.finalize(ctx, results)
    assert (ctx.stage_dir / "frequencies.json").exists()  # written before stopping


def test_casscf_resolves_to_finite_differences(tmp_path, write_input):
    cas = ('[electronic]\nmethod = "casscf"\nbasis = "sto-3g"\nsinglets = 2\n'
           "[electronic.casscf]\nncas = 2\nnelecas = 2\n")
    cfg = load_config(write_input(tmp_path, extra=cas)).config
    assert resolved_method(cfg) == "finite_difference"
```

**About `WATER_MIN`:** this geometry is the expected RHF/STO-3G minimum. If the measured maximum gradient there is above 1e-3, replace the coordinates with an optimized geometry from Task 6's stage, and report it. Analytic and finite-difference results agree to within 0.1 cm⁻¹ with h = 0.005 (Facts §3), so the test uses 1 cm⁻¹ as a safe tolerance.

**About linear water:** the linear geometry is a saddle point with an imaginary bend. If its imaginary mode were smaller than the 20 cm⁻¹ cutoff, use `imaginary_cutoff_cm1 = 0` instead. It is about −1000 cm⁻¹, so this is very unlikely.

- [ ] **Step 2: Run the tests to verify they fail.**
  Run `pytest tests/stages/test_frequencies.py -v -W error`.
  Expected: they fail with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `src/gqteashmd/stages/frequencies.py`:

```python
"""Stage 2: S0 Hessian (analytic or finite differences of gradients) and harmonic analysis."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.chem.masses import isotope_mass
from gqteashmd.config.schema import ShmdConfig
from gqteashmd.electronic.base import Request
from gqteashmd.errors import StageFailedError
from gqteashmd.stages.base import Stage, StageContext, TaskSpec
from gqteashmd.stages.geometry import prepared_molecule, stage_backend, working_geometry
from gqteashmd.util.fileio import write_json_atomic, write_text_atomic
from gqteashmd.vib.artifacts import write_frequencies_h5
from gqteashmd.vib.harmonic import harmonic_analysis
from gqteashmd.vib.molden import format_molden

log = logging.getLogger("gqteashmd.stages.frequencies")
EXTERNAL_RESIDUAL_WARNING_CM1 = 100.0
STATIONARY_GRADIENT = 1e-3  # Eh/bohr


def resolved_method(config: ShmdConfig) -> str:
    requested = config.frequencies.method
    electronic = config.electronic
    if requested == "auto":
        return "finite_difference" if electronic is not None and electronic.method == "casscf" else "analytic"
    return requested


def _task_id(k: int, sign: int) -> str:
    return f"fd_{k:03d}_{'plus' if sign > 0 else 'minus'}"


class FrequenciesStage(Stage):
    name = "frequencies"
    version = 1
    depends_on = ("prepare", "optimize")
    config_sections = ("electronic", "frequencies")
    required_modules = ("pyscf>=2.6", "h5py>=3.9")

    def plan(self, ctx: StageContext) -> list[TaskSpec]:
        if resolved_method(ctx.config) == "analytic":
            return [TaskSpec("hessian")]
        spec, _ = prepared_molecule(ctx.upstream_dir("prepare"))
        tasks = [TaskSpec("reference")]
        for k in range(3 * len(spec.symbols)):
            tasks += [TaskSpec(_task_id(k, s), {"coordinate": k, "sign": s}) for s in (1, -1)]
        return tasks

    def run_task(self, ctx: StageContext, task: TaskSpec, task_dir: Path) -> dict[str, Any]:
        molecule, coords, _ = working_geometry(ctx)
        backend = stage_backend(ctx, self.name, molecule, task_dir)
        s0 = backend.states[0]
        if task.task_id in ("hessian", "reference"):
            result = backend.compute(Request(coords, gradients=frozenset({s0})))
            out: dict[str, Any] = {"energy_hartree": result.energies[s0],
                                   "gradient": result.gradients[s0].ravel().tolist()}
            if task.task_id == "hessian":
                np.save(task_dir / "hessian.npy", backend.ground_state_hessian(coords))
            return out
        displaced = coords.copy().ravel()
        displaced[int(task.payload["coordinate"])] += int(task.payload["sign"]) * ctx.config.frequencies.fd_step_bohr
        result = backend.compute(Request(displaced.reshape(-1, 3), gradients=frozenset({s0})))
        return {"gradient": result.gradients[s0].ravel().tolist()}

    def finalize(self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]) -> None:
        options = ctx.config.frequencies
        molecule, coords, source = working_geometry(ctx)
        method = resolved_method(ctx.config)
        n3 = 3 * len(molecule.symbols)
        if method == "analytic":
            hessian = np.load(ctx.layout.task_dir(ctx.stage_dir, "hessian") / "hessian.npy")
            centre = results["hessian"]
        else:
            hessian = np.zeros((n3, n3))
            step = options.fd_step_bohr
            for k in range(n3):
                plus = np.asarray(results[_task_id(k, 1)]["gradient"])
                minus = np.asarray(results[_task_id(k, -1)]["gradient"])
                hessian[:, k] = (plus - minus) / (2.0 * step)
            hessian = 0.5 * (hessian + hessian.T)
            centre = results["reference"]
        masses = np.array([isotope_mass(s) for s in molecule.symbols])
        analysis = harmonic_analysis(masses, coords, hessian)
        freqs = analysis.frequencies_cm1
        cutoff = options.imaginary_cutoff_cm1
        big = [(i + 1, float(f)) for i, f in enumerate(freqs) if f < -cutoff]
        small = [(i + 1, float(f)) for i, f in enumerate(freqs) if -cutoff <= f < 0]
        max_gradient = float(np.abs(np.asarray(centre["gradient"])).max())
        warnings: list[str] = []
        if small:
            warnings.append("small imaginary modes (within the cutoff): "
                            + ", ".join(f"mode {i}: {f:.1f} cm^-1" for i, f in small))
        if analysis.external_residual_cm1 > EXTERNAL_RESIDUAL_WARNING_CM1:
            warnings.append(f"translation/rotation contamination up to "
                            f"{analysis.external_residual_cm1:.0f} cm^-1 (Hessian or geometry inaccurate)")
        if max_gradient > STATIONARY_GRADIENT:
            warnings.append(f"max |gradient| = {max_gradient:.2e} Eh/bohr: not a stationary point; "
                            "frequencies are approximate")
        for message in warnings:
            log.warning("Stage frequencies: %s", message)
        write_frequencies_h5(ctx.stage_dir / "frequencies.h5", symbols=molecule.symbols,
                             masses_amu=masses, coords_bohr=coords, hessian=hessian, analysis=analysis)
        write_text_atomic(ctx.stage_dir / "frequencies.molden",
                          format_molden(molecule.symbols, coords, freqs, analysis.modes_cartesian))
        write_json_atomic(ctx.stage_dir / "frequencies.json", {
            "schema_version": 1, "method": method, "geometry_source": source,
            "energy_hartree": centre.get("energy_hartree"),
            "max_gradient_hartree_per_bohr": max_gradient,
            "linear": analysis.linear, "n_external": analysis.n_external,
            "external_residual_cm1": analysis.external_residual_cm1,
            "frequencies_cm1": freqs.tolist(),
            "reduced_masses_amu": analysis.reduced_masses_amu.tolist(),
            "imaginary_modes": [{"mode": i, "frequency_cm1": f} for i, f in big],
            "small_imaginary_modes": [{"mode": i, "frequency_cm1": f} for i, f in small],
            "warnings": warnings,
        })
        if big:
            listing = ", ".join(f"mode {i}: {f:.1f} cm^-1" for i, f in big)
            raise StageFailedError(
                f"The structure is not a minimum: {len(big)} imaginary mode(s) beyond "
                f"{cutoff:g} cm^-1 ({listing}). Re-optimize (e.g. [optimize] convergence = "
                "\"GAU_TIGHT\"), check the starting structure, or raise [frequencies] "
                "imaginary_cutoff_cm1 if the mode is a known numerical artefact. The modes are in "
                f"{ctx.stage_dir / 'frequencies.molden'}."
            )
```

Format the long lines with `ruff format` and keep them under 100 characters. `finalize` runs in the controller, and its `ctx.upstream` holds the DONE upstream directories, so `working_geometry` works there too.

- [ ] **Step 4: Run the tests to verify they pass.**
  Run `pytest tests/stages/test_frequencies.py -v -W error && mypy`.
  Expected: all pass. Report:
  - the analytic and finite-difference frequencies;
  - the maximum gradient at `WATER_MIN`;
  - the imaginary mode of linear water.

- [ ] **Step 5: Lint and commit**

```bash
ruff format src tests tools && ruff check src tests tools
git add src/gqteashmd/stages/frequencies.py tests/stages
git commit -m "feat: add the frequencies stage (analytic or finite-difference Hessian)"
```

---

### Task 8: Pipeline: prepare → optimize → frequencies; required input sections

**Files:**
- Modify: `src/gqteashmd/stages/registry.py`, `src/gqteashmd/stages/pipeline.py`, `src/gqteashmd/orchestrator.py`, `src/gqteashmd/cli.py` (`_cmd_validate`)
- Create: `src/gqteashmd/testing/pipelines.py`
- Test: `tests/stages/test_base_pipeline.py`, plus the existing tests that ran the default pipeline

**Interfaces:**
- `default_pipeline(config)` returns `[PrepareStage(), OptimizeStage(), FrequenciesStage()]`.
- `check_stage_inputs(config, stages)` raises a single `ConfigError` listing every enabled stage that names a section in `config_sections` which is `None` in the config. The message reads: "Stage 'frequencies' needs an [electronic] section in the input file (method, basis, singlets, …)." It is called next to `check_stage_names`, both in `Orchestrator.__init__` and in `cli._cmd_validate`.
- `gqteashmd.testing.pipelines:prepare_only(config)` returns `[PrepareStage()]`. It is used by orchestration and CLI tests that do not need quantum chemistry.

- [ ] **Step 1: Write the failing tests** in `tests/stages/test_base_pipeline.py`. Merge the imports with the existing ones.

```python
def test_default_pipeline_order(tmp_path, write_input):
    from gqteashmd.config.loader import load_config
    from gqteashmd.stages.registry import default_pipeline

    cfg = load_config(write_input(tmp_path)).config
    assert [s.name for s in default_pipeline(cfg)] == ["prepare", "optimize", "frequencies"]


def test_missing_electronic_section_is_reported(tmp_path, write_input):
    from gqteashmd.config.loader import load_config
    from gqteashmd.errors import ConfigError
    from gqteashmd.stages.pipeline import check_stage_inputs
    from gqteashmd.stages.registry import default_pipeline

    cfg = load_config(write_input(tmp_path)).config  # no [electronic]
    with pytest.raises(ConfigError) as info:
        check_stage_inputs(cfg, default_pipeline(cfg))
    text = str(info.value)
    assert "'optimize'" in text and "'frequencies'" in text and "[electronic]" in text
```

Add a CLI test: `gqteashmd validate` on an input without `[electronic]` returns exit 2 and prints the message above.

- [ ] **Step 2: Run them to verify they fail.**
  Run: `pytest tests/stages tests/test_cli.py -v -W error -k "pipeline_order or electronic_section or validate"`
  Expected: FAIL.

- [ ] **Step 3: Implement.**
  - **`registry.py`.** Return the three stages and update the module docstring.
  - **`pipeline.py`:**

```python
def check_stage_inputs(config: ShmdConfig, stages: Sequence[Stage]) -> None:
    """Every enabled stage's input sections must be present (optional sections default)."""
    missing = [
        f"Stage '{stage.name}' needs an [{section}] section in the input file"
        for stage in stages
        if stage.enabled(config)
        for section in stage.config_sections
        if getattr(config, section, None) is None
    ]
    if missing:
        raise ConfigError("; ".join(missing) + ". See USER_MANUAL.md §4.")
```

  - **Wiring.** Call `check_stage_inputs` in `Orchestrator.__init__` after `check_stage_names`, and in `cli._cmd_validate`.
  - **`testing/pipelines.py`:**

```python
"""Pipelines for tests that exercise orchestration without quantum chemistry."""

from __future__ import annotations

from gqteashmd.config.schema import ShmdConfig
from gqteashmd.stages.base import Stage
from gqteashmd.stages.prepare import PrepareStage


def prepare_only(config: ShmdConfig) -> list[Stage]:
    return [PrepareStage()]
```

- **Adapt the existing tests.** Some tests run `gqteashmd run` or `validate` on the default pipeline only to exercise `prepare`, the controller or the CLI. Pass them `--pipeline gqteashmd.testing.pipelines:prepare_only`, or the module's existing fake pipeline. Find them with `grep -rn "\"run\"\|'run'\|\"validate\"" tests`, then check each one.
  - Do not add `[electronic]` to those tests: that would make them run PySCF.
  - `test_package` and the template tests must still pass.
  - List every test you changed in the report.

- [ ] **Step 4: Run the tests to verify they pass.**
  Run: `pytest -m "not slow and not pyscf" -W error && mypy`
  Expected: all pass, and the fast tier takes about the same time as before. Report the time.

- [ ] **Step 5: Lint and commit**

```bash
ruff format src tests tools && ruff check src tests tools
git add src/gqteashmd tests
git commit -m "feat: run prepare, optimize and frequencies; require the stages' input sections"
```

---

### Task 9: End-to-end and external validation tests

**Files:**
- Create:
  - `tests/test_e2e_frequencies.py` (pyscf; the CASSCF case is also `slow`)
  - `tests/validation/test_orca_frequencies.py` (marker `orca`)
  - `tests/validation/test_sharc_wigner_reads_molden.py` (marker `sharc`)
- Modify: `pyproject.toml` (register the marker `"orca: runs ORCA as a black box (needs ORCA_EXE)"`)

**Interfaces:** none new. These tests run `gqteashmd run` through `cli.main`, and run ORCA and SHARC as black boxes.

- [ ] **Step 1: Write the tests.**

`tests/test_e2e_frequencies.py`:

```python
import json

import pytest

from gqteashmd.cli import main
from gqteashmd.config.loader import load_config
from gqteashmd.project.layout import ProjectLayout

pytestmark = pytest.mark.pyscf
HF = '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 2\n'


def test_run_optimizes_and_computes_frequencies(tmp_path, write_input):
    path = write_input(tmp_path, extra=HF)
    assert main(["run", str(path)]) == 0
    layout = ProjectLayout(load_config(path).config.project.directory)
    opt = json.loads((layout.stage_dir(1, "optimize") / "optimize.json").read_text())
    freq = json.loads((layout.stage_dir(2, "frequencies") / "frequencies.json").read_text())
    assert opt["converged"] and freq["geometry_source"] == "optimize"
    assert len(freq["frequencies_cm1"]) == 3 and freq["imaginary_modes"] == []
    assert all(1500 < f < 4500 for f in freq["frequencies_cm1"])
    assert main(["run", str(path)]) == 0  # up to date: nothing recomputed


def test_disabling_optimize_after_a_run_is_refused_without_changes(tmp_path, write_input):
    path = write_input(tmp_path, extra=HF)
    assert main(["run", str(path)]) == 0
    path.write_text(path.read_text() + "\n[optimize]\nenabled = false\n")
    assert main(["run", str(path)]) == 3  # frequencies was computed from the optimized geometry


@pytest.mark.slow
def test_casscf_frequencies_by_finite_differences(tmp_path, write_input):
    from pathlib import Path

    ethylene = (Path(__file__).parent / "data" / "ethylene.xyz").read_text()
    (tmp_path / "water.xyz").write_text(ethylene)  # write_input points [molecule] at water.xyz
    cas = ('[electronic]\nmethod = "casscf"\nbasis = "sto-3g"\nsinglets = 2\n'
           "[electronic.casscf]\nncas = 2\nnelecas = 2\n[optimize]\nenabled = false\n"
           "[frequencies]\nimaginary_cutoff_cm1 = 10000.0\n")
    path = write_input(tmp_path, extra=cas)
    (tmp_path / "water.xyz").write_text(ethylene)
    assert main(["run", str(path)]) == 0
    layout = ProjectLayout(load_config(path).config.project.directory)
    freq = json.loads((layout.stage_dir(2, "frequencies") / "frequencies.json").read_text())
    assert freq["method"] == "finite_difference" and len(freq["frequencies_cm1"]) == 12
```

Two notes on this file:
- **Stopping after refusal.** In the second test, the refusal comes from Task 10. Until Task 10 lands the exit code is still 3, because `StaleStageError` is already raised. The "without changes" part is asserted in Task 10, so do not add a state assertion here.
- **The water fixture.** `write_input` writes `water.xyz`. Rewrite it after `write_input` with the ethylene geometry, as shown, or use the fixture's geometry option if it has one. The CASSCF frequencies are at the unoptimized geometry, so imaginary modes are possible; the cutoff is set high so the test checks the mechanics. Report the measured duration.

`tests/validation/test_orca_frequencies.py`:

```python
import os
import re
import subprocess

import numpy as np
import pytest

from gqteashmd.core.units import ANGSTROM_TO_BOHR
from gqteashmd.electronic.base import MoleculeSpec
from gqteashmd.electronic.pyscf_backend import PySCFBackend
from gqteashmd.vib.harmonic import harmonic_analysis

pytestmark = [pytest.mark.orca, pytest.mark.pyscf,
              pytest.mark.skipif(not os.environ.get("ORCA_EXE"), reason="needs ORCA_EXE")]
GEOMETRY = [("O", (0.0, 0.0, 0.108140699)), ("H", (0.0, 0.754903478, -0.464620350)),
            ("H", (0.0, -0.754903478, -0.464620350))]  # RHF/6-31G* minimum (Angstrom)


def _run_orca(tmp_path):
    lines = "\n".join(f"{s} {x:.9f} {y:.9f} {z:.9f}" for s, (x, y, z) in GEOMETRY)
    (tmp_path / "w.inp").write_text(f"! RHF 6-31G* Freq TightSCF\n* xyz 0 1\n{lines}\n*\n")
    out = subprocess.run([os.environ["ORCA_EXE"], "w.inp"], cwd=tmp_path, capture_output=True,
                         text=True, timeout=600, check=True).stdout
    block = out.split("VIBRATIONAL FREQUENCIES", 1)[1]
    freqs = [float(m) for m in re.findall(r"^\s*\d+:\s+(-?\d+\.\d+) cm\*\*-1", block, re.M)]
    table = out.split("NO LB      ZA    FRAG     MASS", 1)[1].splitlines()[1:]
    masses = [float(row.split()[4]) for row in table[: len(GEOMETRY)]]
    return np.array(freqs), np.array(masses)


def test_frequencies_match_orca(tmp_path, make_settings):
    orca_freqs, orca_masses = _run_orca(tmp_path)
    settings = make_settings(method="tda", basis="6-31g*", functional="hf", singlets=1,
                             casscf=None, scf_conv_tol=1e-11)
    coords = np.array([c for _, c in GEOMETRY]) * ANGSTROM_TO_BOHR
    backend = PySCFBackend(MoleculeSpec(tuple(s for s, _ in GEOMETRY)), settings)
    ours = harmonic_analysis(orca_masses, coords, backend.ground_state_hessian(coords))
    np.testing.assert_allclose(ours.frequencies_cm1, orca_freqs[orca_freqs != 0.0], atol=0.5)
```

(`make_settings` is a fixture in `tests/electronic/conftest.py`. Either move this file under `tests/electronic/`, or copy the small settings builder into the file. Choose one and report which.)

`tests/validation/test_sharc_wigner_reads_molden.py`. It runs SHARC's `wigner.py` as a black box on our Molden file:

```python
import os
import subprocess
import sys

import numpy as np
import pytest

from gqteashmd.vib.harmonic import harmonic_analysis
from gqteashmd.vib.molden import format_molden

pytestmark = [pytest.mark.sharc,
              pytest.mark.skipif(not (os.environ.get("SHARC_HOME") and os.environ.get("SHARC_PYTHON")),
                                 reason="needs SHARC_HOME and SHARC_PYTHON")]


def test_sharc_wigner_accepts_our_molden(tmp_path):
    masses = np.array([15.994915, 1.007825, 1.007825])
    coords = np.array([[0.0, 0.0, 0.2043], [0.0, 1.4266, -0.8780], [0.0, -1.4266, -0.8780]])
    rng = np.random.default_rng(3)
    a = rng.normal(size=(9, 9))
    analysis = harmonic_analysis(masses, coords, a @ a.T * 0.1)
    molden = tmp_path / "freq.molden"
    molden.write_text(format_molden(("O", "H", "H"), coords, analysis.frequencies_cm1,
                                    analysis.modes_cartesian))
    home = os.environ["SHARC_HOME"]
    env = dict(os.environ, SHARC=f"{home}/bin", PYTHONPATH=f"{home}/lib:{home}/pysharc")
    before = set(os.listdir(tmp_path))
    proc = subprocess.run([os.environ["SHARC_PYTHON"], f"{home}/bin/wigner.py", "-n", "3",
                           str(molden)], cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert set(os.listdir(tmp_path)) - before  # wigner.py wrote its initial conditions
```

- **Clean room:** never open the SHARC sources. Only run them and look at their outputs.
- **If `wigner.py` rejects the file:** report its stderr verbatim. Adjust `format_molden` only from that error message and the documented Molden format.

- [ ] **Step 2: Run the tests.**
  - `pytest tests/test_e2e_frequencies.py -v -W error -m "not slow"`. Expected: 2 passed.
  - `pytest tests/test_e2e_frequencies.py -v -W error -m slow`. Expected: 1 passed; report the time.
  - `ORCA_EXE=/home/ajc/bin/orca_6_1_1/orca pytest -m orca -v -W error`. Expected: 1 passed; report the differences.
  - `SHARC_HOME=/home/ajc/sharc4-4.1 SHARC_PYTHON=/home/ajc/miniforge3/envs/sharc4.1/bin/python pytest -m sharc -v -W error`. Expected: the existing reference test and the wigner test pass.
- [ ] **Step 3: Lint and commit**

```bash
ruff format src tests tools && ruff check src tests tools
git add pyproject.toml tests
git commit -m "test: validate optimize and frequencies end to end and against ORCA and SHARC wigner.py"
```

---

### Task 10: Refuse stale runs before writing anything; deleted stage directories

**Files:**
- Modify: `src/gqteashmd/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- **Stale check up front.** `Orchestrator._check_up_to_date(state)` runs before `write_snapshot`, `_reset_from` and any stage-state write. It computes every stage's fingerprint, giving disabled stages `SKIPPED_FINGERPRINT`, and raises `StaleStageError` (the same message as today) for the first stale stage. Stages from `--rerun-from` onward are exempt, because they will be reset.
  - **The new order in `run()`:**
    1. session
    2. StateStore
    3. `append_provenance` (kept, as documented)
    4. `_check_up_to_date`
    5. `write_snapshot`
    6. `_reset_from`
    7. `_run_pipeline`
  - **Result:** a refused run leaves `state.db` and `config.json` byte-identical, and no record changes. For example, a disabled optional stage no longer overwrites DONE with SKIPPED.
- **Shared helpers.** `_stage_fingerprint(stage, ctx, fingerprints) -> str` and `_stale_error(stage) -> StaleStageError` are shared by `_check_up_to_date` and `_run_pipeline`.
- **Rule: `fingerprint_inputs` reads no upstream artifacts.** The pre-pass passes an empty `upstream`. State this rule in the `Stage` docstring.
- **Deleted directory rule.** It now covers every non-SKIPPED stage record whose directory is missing, provided the stage is DONE or has DONE tasks. The stage is reset with a warning and recomputed. Previously only DONE stages were covered (P3c ledger).

- [ ] **Step 1: Write the failing tests** in `tests/test_orchestrator.py`. Use the module's fake pipeline, `behaviour` fixtures and helpers. Its fake pipeline has an optional `optimize` that the behaviour key `skip_optimize` disables.

```python
def test_a_refused_run_changes_nothing(project, behaviour):
    """Run fully; then disable the optional stage: the run is refused and nothing is written."""
    # use the module's helper to run the fake pipeline once (all stages DONE)
    first = run_fake(project)                                   # adapt to the real helper name
    assert first.completed
    db = project.layout.state_db.read_bytes()
    snapshot = project.layout.config_snapshot.read_bytes()
    behaviour({"skip_optimize": True})
    with pytest.raises(StaleStageError):
        run_fake(project)
    assert project.layout.config_snapshot.read_bytes() == snapshot
    store = StateStore(project.layout.state_db, readonly=True)
    try:
        assert store.get_stage("optimize").status == StageStatus.DONE
    finally:
        store.close()


def test_a_pending_stage_with_done_tasks_and_a_deleted_directory_is_recomputed(project, behaviour):
    # stop a stage part-way (some tasks DONE), delete its directory, run again → recomputed
    ...
```

The second test sketch shows the intent. Write it with the module's existing stop and partial-run helpers: the slow kill/resume tests show how to stop part-way. If writing it needs more than a stop flag after the first finished task, use a `should_stop` callable that returns True after one `on_finish`, through `Orchestrator.run(should_stop=...)`.

Compare `state.db` by its records, not its bytes, because SQLite WAL can touch the file. `config.json` must be byte-identical.

- [ ] **Step 2: Run them to verify they fail.**
  Run `pytest tests/test_orchestrator.py -v -W error -k "refused_run or deleted_directory"`.
  Expected: FAIL. Today the snapshot is rewritten and optimize becomes SKIPPED. The partial stage ends in "missing or corrupt".

- [ ] **Step 3: Implement** in `orchestrator.py`.
  - Factor the fingerprint computation and the stale message out of `_run_pipeline` into `_stage_fingerprint` and `_stale_error`.
  - Add the pre-pass:

```python
    def _check_up_to_date(self, state: StateStore) -> None:
        """Refuse a stale run before anything in the project is changed."""
        names = [s.name for s in self.stages]
        exempt = set(names[names.index(self.rerun_from):]) if self.rerun_from else set()
        fingerprints: dict[str, str] = {}
        for index, stage in enumerate(self.stages):
            if not stage.enabled(self.config):
                fingerprints[stage.name] = SKIPPED_FINGERPRINT
                continue
            ctx = StageContext(self.config, self.layout, self.layout.stage_dir(index, stage.name), {})
            fp = self._stage_fingerprint(stage, ctx, fingerprints)
            fingerprints[stage.name] = fp
            if stage.name in exempt:
                continue
            record = state.get_stage(stage.name)
            if record is not None and record.fingerprint != fp:
                done_tasks = state.task_counts(stage.name)[TaskStatus.DONE] > 0
                if done_tasks or record.status in (StageStatus.DONE, StageStatus.RUNNING):
                    raise self._stale_error(stage)
```

  - Reorder `run()` as listed in the Interfaces.
  - Extend the deleted-directory rule in `_run_pipeline`:

```python
            if (
                record is not None
                and record.status != StageStatus.SKIPPED
                and not stage_dir.exists()
                and (record.status == StageStatus.DONE
                     or state.task_counts(stage.name)[TaskStatus.DONE] > 0)
            ):
```

  - Keep the warning and the `reset_stage` call. Update the `Stage` docstring in `stages/base.py` with the `fingerprint_inputs` rule.

- [ ] **Step 4: Run the tests to verify they pass.**
  Run `pytest tests/test_orchestrator.py tests/test_cli.py -v -W error && pytest -m slow -W error tests/test_cli.py`.
  Expected: all pass.

- [ ] **Step 5: Lint, run the fast tier, commit**

```bash
ruff format src tests tools && ruff check src tests tools && pytest -m "not slow and not pyscf" -W error
git add src/gqteashmd/orchestrator.py src/gqteashmd/stages/base.py tests/test_orchestrator.py
git commit -m "fix: refuse a stale run before writing anything and recompute partial deleted stages"
```

---

### Task 11: Preflight: module versions

**Files:**
- Modify: `src/gqteashmd/preflight.py`
- Test: `tests/test_preflight.py`

**Interfaces:**
- `required_modules` entries may be `"name"` or `"name>=X.Y[.Z]"`.
- Preflight imports nothing. It checks two things:
  - **Presence**, with `importlib.util.find_spec`, as today.
  - **Version**, with `importlib.metadata.version(name)`. If the version cannot be read (`PackageNotFoundError`), the check is skipped.
- A version that is too old raises `ConfigError`: "Stage 'optimize' needs geometric >= 1.0; version 0.9.7 is installed. Update the conda environment (environment.yml)."
- An entry that does not parse raises `ValueError`. That is a programming error.
- **Ruling (writable scratch).** Stages put PySCF's scratch in `task_dir / "scratch"`, inside the project directory, whose writability preflight already checks. No separate scratch check is added, and the manual must say this.

- [ ] **Step 1: Write the failing tests** in `tests/test_preflight.py`:

```python
def test_module_versions_are_checked(tmp_path, write_input, monkeypatch):
    from importlib import metadata

    from gqteashmd.config.loader import load_config
    from gqteashmd.errors import ConfigError
    from gqteashmd.preflight import MachineInfo, run_preflight
    from gqteashmd.stages.base import Stage

    class Needy(Stage):
        name = "needy"
        required_modules = ("numpy>=999.0",)

        def plan(self, ctx):
            return []

        def run_task(self, ctx, task, task_dir):
            return {}

        def finalize(self, ctx, results):
            pass

    cfg = _with_directory(load_config(write_input(tmp_path)).config, tmp_path / "run")
    with pytest.raises(ConfigError, match=r"needs numpy >= 999.0; version .* is installed"):
        run_preflight(cfg, [Needy()], machine=MachineInfo(None, None))
    Needy.required_modules = ("numpy>=1.0",)
    run_preflight(cfg, [Needy()], machine=MachineInfo(None, None))
    Needy.required_modules = ("numpy>>1",)
    with pytest.raises(ValueError, match="requirement"):
        run_preflight(cfg, [Needy()], machine=MachineInfo(None, None))
    monkeypatch.setattr(metadata, "version",
                        lambda name: (_ for _ in ()).throw(metadata.PackageNotFoundError(name)))
    Needy.required_modules = ("numpy>=999.0",)
    run_preflight(cfg, [Needy()], machine=MachineInfo(None, None))  # unknown version: skip
```

- [ ] **Step 2: Run it to verify it fails.**
  Run `pytest tests/test_preflight.py -v -W error -k versions`.
  Expected: FAIL. The entry `numpy>=999.0` is not recognised as a module name.

- [ ] **Step 3: Implement.** In `preflight.py`, add:

```python
from importlib import metadata

_REQUIREMENT = re.compile(r"([A-Za-z0-9_.\-]+)(?:>=([0-9]+(?:\.[0-9]+)*))?")


def _parse_requirement(text: str) -> tuple[str, str | None]:
    match = _REQUIREMENT.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"invalid module requirement {text!r}; use 'name' or 'name>=X.Y'")
    return match.group(1), match.group(2)


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])
```

and replace the module loop with:

```python
    for stage in stages:
        if not stage.enabled(config):
            continue
        for requirement in stage.required_modules:
            name, minimum = _parse_requirement(requirement)
            if not _module_available(name):
                raise ConfigError(
                    f"Stage '{stage.name}' needs the Python module '{name}', which is not "
                    "installed in this environment."
                )
            if minimum is None:
                continue
            try:
                installed = metadata.version(name)
            except metadata.PackageNotFoundError:
                continue
            if _version_tuple(installed) < _version_tuple(minimum):
                raise ConfigError(
                    f"Stage '{stage.name}' needs {name} >= {minimum}; version {installed} is "
                    "installed. Update the conda environment (environment.yml)."
                )
```

`validate_pipeline` may check `required_modules` entries. If it does, make it accept the `>=` form.

- [ ] **Step 4: Run the tests to verify they pass.**
  Run `pytest tests/test_preflight.py tests/stages -v -W error && mypy`.
  Expected: all pass.

- [ ] **Step 5: Lint, run the fast tier, commit**

```bash
ruff format src tests tools && ruff check src tests tools && pytest -m "not slow and not pyscf" -W error
git add src/gqteashmd/preflight.py tests/test_preflight.py
git commit -m "feat: check minimum module versions in the preflight"
```

---

### Task 12: Documentation current with P4

**Files:**
- Modify: `CLAUDE.md`, `USER_MANUAL.md`, `docs/superpowers/plans/2026-11-06-gqteashmd-master-plan.md`

- [ ] **Step 1: USER_MANUAL.md.**
  - **§1 and the workflow description:** `prepare → optimize → frequencies` is implemented. `sampling` and later stages are planned.
  - **§4: input keys.** Add the `[optimize]` and `[frequencies]` key tables (type, default, allowed values, meaning). State that `[electronic]` is required by `optimize` and `frequencies`. Document the template.
  - **§5:** an example `gqteashmd run` with real output (water HF/STO-3G), run in `mktemp -d /tmp/gq-doc-XXXX` with `conda activate gqteashmd-dev` and deleted afterwards. List the artifacts with their contents and units: `optimized.xyz`, `optimization_path.xyz`, `optimize.json`, `frequencies.h5`, `frequencies.molden`, `frequencies.json`.
  - **§8: formulas [IMPLEMENTED].** Define every symbol and unit:
    - mass weighting;
    - the Eckart translation and rotation vectors, and the complement projection;
    - ω and ν̃ (cm⁻¹ conversion);
    - normal modes, Cartesian displacements and reduced mass;
    - imaginary modes as negative numbers;
    - the finite-difference Hessian (central differences, symmetrization, 1 + 6N gradients);
    - isotope masses (O5);
    - why S0 is computed with SCF only for TDA/TDDFT, and with the state average for SA-CASSCF.
  - **§8: the stopping and warning policy.** Imaginary modes beyond the cutoff stop the run; smaller ones warn. Also cover the external-residual warning (100 cm⁻¹) and the stationarity warning (1e-3 Eh/bohr).
  - **§8: validation.** Include the measured comparisons:
    - harmonic analysis vs PySCF;
    - analytic vs finite-difference Hessian;
    - the ORCA comparison;
    - SHARC `wigner.py` reading our Molden file.
  - **§7 and §9:**
    - a refused stale run now changes nothing;
    - the deleted-directory rule;
    - preflight version checks;
    - the scratch location;
    - the new error messages (missing `[electronic]` for a stage, no convergence, imaginary modes, module version).
- [ ] **Step 2: CLAUDE.md.**
  - **Status:** P4 is done on `feat/p4-optimize-frequencies` and not merged. Record the test counts and the tier times.
  - **Architecture:** add `chem/masses`, `vib/`, `geomopt`, `stages/geometry|optimize|frequencies`, `testing/pipelines`, and the orchestrator pre-pass.
  - **Rules:**
    - geomeTRIC and h5py are imported lazily;
    - optimize uses one backend, and FD tasks use a fresh backend each;
    - S0 settings come from `ground_state_settings`;
    - `fingerprint_inputs` reads no upstream artifacts.
  - **Decisions:** O5 (masses) and the owner's decisions 1–2.
  - **Remaining work:** P5 (`sampling`) is next.
- [ ] **Step 3: Master plan.**
  - §9: mark P4 delivered.
  - §11: mark O5 resolved.
  - §4.3: `convergence` is `"GAU"`, and the keys are as implemented.
  - §5: the stage table rows match the code, including the warning rulings.
- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md USER_MANUAL.md docs/superpowers/plans/2026-11-06-gqteashmd-master-plan.md
git commit -m "docs: document the optimize and frequencies stages (P4)"
```

---

## Self-review notes

- **Spec coverage:**

  | Spec item | Task |
  |---|---|
  | §1.1 stages 1–2 | Tasks 6, 7, 8 |
  | §4.3 keys | Task 3 (`convergence` uses geomeTRIC's set names; "default" maps to `GAU`) |
  | §5 artifacts and stop conditions | Tasks 6, 7 (the near-zero-mode stop became a warning, Facts §4) |
  | §6.1 errors | Tasks 6, 7 |
  | §7 frequencies validation vs an independent program and SHARC `wigner.py` readability | Task 9 |
  | D11 HDF5 with units | Task 2 |
  | O5 | Task 1 |
  | P3c items moved to P4: refused runs writing state, SKIPPED over DONE, `_reset_from` before the stale check | Task 10 |
  | P3c items moved to P4: validator first violation | Task 3 |
  | P3c items moved to P4: version and scratch preflight | Task 11 (ruling: scratch lives in the project) |
  | Parked P3c item: deleted directory with partial tasks | Task 10 |
  | Owner decisions | Tasks 4 and 7 (decision 1); Task 9 (decision 2) |

- **Type consistency:**
  - `harmonic_analysis(masses_amu, coords_bohr, hessian) -> HarmonicAnalysis` is used in Tasks 2, 7 and 9.
  - `format_molden(symbols, coords_bohr, frequencies_cm1, modes_cartesian)` is used in Tasks 2, 7 and 9.
  - `write_frequencies_h5(path, *, symbols, masses_amu, coords_bohr, hessian, analysis)` is used in Tasks 2 and 7.
  - `optimize_geometry(...) -> OptimizationResult` is used in Tasks 5 and 6.
  - `make_backend(..., settings=, scratch_dir=)` and `ground_state_settings` are used in Tasks 4 and 6.
  - `working_geometry` and `stage_backend` are used in Tasks 6 and 7.
  - `check_stage_inputs` is used in Task 8.
  - `resolved_method` is used in Task 7.
