"""PySCF validation of the trajectory code (P7b Task 9, bounds fixed in the plan before any run).

(a) energy conservation, ethylene SA2-CASSCF(2,2)/STO-3G on S1 with local diabatization;
(b) the dt^2 scaling of the drift (slow); (c) resume from a restart with a new backend, model and
runner (O16 extension) against one uninterrupted run; (d) energy conservation for water CIS (TDA).
Initial velocities: v = N(0, 1) * sqrt(k_B T / m), T = 300 K, seed 2026, no momentum removal.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from gqteashmd.chem.masses import isotope_mass
from gqteashmd.chem.xyz import parse_xyz
from gqteashmd.config.loader import load_config
from gqteashmd.core.units import (
    ANGSTROM_TO_BOHR,
    AU_TIME_IN_FS,
    BOLTZMANN_HARTREE_PER_K,
    DALTON_IN_ME,
)
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.randomness import GeneratorSource
from gqteashmd.electronic.base import MoleculeSpec
from gqteashmd.electronic.factory import make_backend
from gqteashmd.trajectory.model import BackendModel
from gqteashmd.trajectory.runner import InitialConditions, RunSettings, TrajectoryRunner
from gqteashmd.trajectory.store import StepLog

pytestmark = pytest.mark.pyscf

DATA = Path(__file__).resolve().parents[1] / "data"
ETHYLENE = (
    '[electronic]\nmethod = "casscf"\nbasis = "sto-3g"\nsinglets = 2\n'
    "[electronic.casscf]\nncas = 2\nnelecas = 2\n"
)
WATER_CIS = '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 3\n'
TEMPERATURE_K = 300.0
VELOCITY_SEED = 2026


class System:
    """An electronic configuration, a molecule and its initial conditions (start S1)."""

    def __init__(self, tmp_path: Path, write_input, electronic: str, xyz: str) -> None:
        path = write_input(tmp_path, extra=electronic)
        (tmp_path / "water.xyz").write_text(xyz)  # write_input points [molecule] at water.xyz
        self.config = load_config(path).config
        geometry = parse_xyz(xyz)
        self.molecule = MoleculeSpec(geometry.symbols, 0, 1)
        self.q = (np.array(geometry.coords_angstrom) * ANGSTROM_TO_BOHR).reshape(-1)
        masses = np.array([isotope_mass(s) for s in geometry.symbols]) * DALTON_IN_ME
        rng = np.random.default_rng(VELOCITY_SEED)
        v = rng.standard_normal((len(masses), 3)) * np.sqrt(
            BOLTZMANN_HARTREE_PER_K * TEMPERATURE_K / masses[:, None]
        )
        self.v = v.reshape(-1)
        self.masses = np.repeat(masses, 3)
        self.tmp_path = tmp_path

    def run(self, name: str, *, dt_fs: float, n_steps: int, **options) -> tuple[dict, np.ndarray]:
        """Run (or resume) trajectory `name` to n_steps steps of dt_fs; a new backend each call."""
        directory = self.tmp_path / name
        backend = make_backend(
            self.config,
            self.molecule,
            settings=self.config.electronic,
            scratch_dir=self.tmp_path / "scratch",
            log_file=self.tmp_path / "e.log",
            threads=2,
            memory_mb=1000,
        )
        coupling = options.setdefault("coupling", "overlap")
        dt_au = dt_fs / AU_TIME_IN_FS
        settings = RunSettings(DynamicsOptions(dt_au=dt_au, **options), tmax_au=n_steps * dt_au)
        model = BackendModel(backend, self.masses, coupling=coupling)
        runner = TrajectoryRunner(model, settings, directory)
        result = runner.run(
            InitialConditions(self.q, self.v, 1), GeneratorSource(np.random.default_rng(0))
        )
        rows = StepLog(directory / "steps.log", model.n_states, model.n_dof).read()
        return result, rows


def _ethylene(tmp_path, write_input) -> System:
    return System(tmp_path, write_input, ETHYLENE, (DATA / "ethylene.xyz").read_text())


def _drift(rows: np.ndarray) -> float:
    return float(np.max(np.abs(rows["e_total"] - rows["e_total"][0])))


def _brief(result: dict) -> str:
    keys = ("outcome", "t_final_fs", "n_grid_steps", "n_steps", "n_rejections", "max_drift_eV")
    return ", ".join(f"{k} {result[k]}" for k in keys) + f"; {result['message']}"


def _smallest(rows: np.ndarray) -> str:
    """The smallest singular value over the accepted steps (record 0 has none)."""
    sv = rows["min_singular_value"][1:]
    return f"{sv.min():.6f}" if sv.size else "- (no accepted step)"


def _timed(label: str, call):
    start = time.perf_counter()
    out = call()
    print(f"{label}: wall time {time.perf_counter() - start:.1f} s")
    return out


CONSERVING = {"hopping": "none", "decoherence": "none"}


def test_ethylene_casscf_conserves_energy_with_local_diabatization(tmp_path, write_input):
    system = _ethylene(tmp_path, write_input)
    result, rows = _timed(
        "(a) ethylene 20 steps of 0.5 fs",
        lambda: system.run("a", dt_fs=0.5, n_steps=20, **CONSERVING),
    )
    drift = _drift(rows)
    sv = rows["min_singular_value"][1:]
    rejections = int(rows["rejections"].sum())
    numbers = (
        f"max |E_tot - E_tot(0)| = {drift:.3e} Eh; smallest singular value {_smallest(rows)} "
        f"(per record {np.array2string(sv, precision=6)}); rejections {rejections} "
        f"(result {result['n_rejections']}); outcome {result['outcome']} ({result['message']}); "
        f"{result['n_grid_steps']} grid steps"
    )
    print("(a)", numbers)
    assert result["outcome"] == "finished" and result["n_grid_steps"] == 20, numbers
    assert rows.shape == (21,), numbers
    assert drift <= 1e-4, numbers
    assert np.all(sv >= 0.9), numbers
    assert rejections == 0 and result["n_rejections"] == 0, numbers


@pytest.mark.slow
def test_ethylene_casscf_drift_scales_as_dt_squared(tmp_path, write_input):
    system = _ethylene(tmp_path, write_input)
    drifts = {}
    for dt_fs, n_steps in ((0.5, 10), (0.25, 20)):
        result, rows = _timed(
            f"(b) ethylene 5 fs at dt {dt_fs} fs",
            lambda dt_fs=dt_fs, n_steps=n_steps: system.run(
                f"b_{dt_fs}", dt_fs=dt_fs, n_steps=n_steps, **CONSERVING
            ),
        )
        drifts[dt_fs] = _drift(rows)
        assert result["outcome"] == "finished" and result["n_rejections"] == 0, _brief(result)
    ratio = drifts[0.5] / drifts[0.25]
    numbers = (
        f"drift(0.5 fs) = {drifts[0.5]:.3e} Eh, drift(0.25 fs) = {drifts[0.25]:.3e} Eh, "
        f"ratio {ratio:.3f}"
    )
    print("(b)", numbers)
    if drifts[0.5] < 1e-6:
        pytest.skip(f"drift(0.5 fs) below 1e-6 Eh, convergence noise dominates: {numbers}")
    assert 2.5 <= ratio <= 6.0, numbers


def test_ethylene_resume_from_a_restart_matches_one_run(tmp_path, write_input):
    system = _ethylene(tmp_path, write_input)
    ref_result, ref = _timed(
        "(c) reference 10 steps", lambda: system.run("ref", dt_fs=0.5, n_steps=10, **CONSERVING)
    )
    first, _ = _timed(
        "(c) first part 5 steps", lambda: system.run("cut", dt_fs=0.5, n_steps=5, **CONSERVING)
    )
    assert first["outcome"] == "finished" and first["n_grid_steps"] == 5, _brief(first)
    second, resumed = _timed(
        "(c) resumed to 10 steps",
        lambda: system.run("cut", dt_fs=0.5, n_steps=10, **CONSERVING),
    )
    assert ref_result["outcome"] == "finished", _brief(ref_result)
    assert second["outcome"] == "finished", _brief(second)
    assert second["n_grid_steps"] == 10 and second["n_steps"] == 10, _brief(second)
    assert resumed.shape == ref.shape == (11,)

    def populations(rows):
        return rows["c_real"] ** 2 + rows["c_imag"] ** 2

    dq = float(np.max(np.abs(resumed["q"] - ref["q"])))
    de = float(np.max(np.abs(resumed["energies"] - ref["energies"])))
    de_tot = float(np.max(np.abs(resumed["e_total"] - ref["e_total"])))
    dp = float(np.max(np.abs(populations(resumed) - populations(ref))))
    numbers = (
        f"max |dq| = {dq:.3e} bohr, max |dE| = {de:.3e} Eh (E_tot {de_tot:.3e} Eh), "
        f"max |dP| = {dp:.3e} over the 11 records"
    )
    print("(c)", numbers)
    assert dq <= 1e-6, numbers
    assert de <= 1e-7, numbers
    assert dp <= 1e-6, numbers


def test_water_cis_conserves_energy(tmp_path, write_input, water_xyz):
    system = System(tmp_path, write_input, WATER_CIS, water_xyz)
    result, rows = _timed(
        "(d) water CIS 20 steps of 0.25 fs",
        lambda: system.run("d", dt_fs=0.25, n_steps=20, hopping="none"),
    )
    drift = _drift(rows)
    numbers = (
        f"max |E_tot - E_tot(0)| = {drift:.3e} Eh; smallest singular value "
        f"{_smallest(rows)}; rejections {result['n_rejections']}; outcome {result['outcome']} "
        f"({result['message']}); {len(rows)} records"
    )
    print("(d)", numbers)
    assert result["outcome"] == "finished" and result["n_grid_steps"] == 20, numbers
    assert drift <= 1e-4, numbers
    assert result["n_rejections"] == 0 and int(rows["rejections"].sum()) == 0, numbers
