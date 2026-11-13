"""PySCF validation of the trajectory code (P7b Task 9, bounds fixed in the plan before any run).

(a) energy conservation, ethylene SA2-CASSCF(2,2) on S1 with local diabatization; (b) the dt^2
scaling of the drift (slow); (c) resume from a restart with a new backend, model and runner (O16
extension) against one uninterrupted run; (d) energy conservation for water CIS (TDA).
Initial velocities: v = N(0, 1) * sqrt(k_B T / m), T = 300 K, seed 2026, no momentum removal.

Changed after the measured numbers were seen (P7b Task 9, controller ruling; each changed test
says what and why): (a)-(c) use 6-31G* instead of STO-3G, because the STO-3G SA2-CASSCF pi,pi*
solution is metastable (a sigma,pi* solution 11.3 mEh lower in SA energy exists at the start; the
pi,pi* one is lost after 1.06 fs on S1, smallest singular value 0.035 at every halving). The
1e-4 Eh drift bounds of (a) and (d) are kept as strict xfails with the measured numbers; (d)
asserts the dt^2 ratio over 2 fs instead.
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
from gqteashmd.dynamics.randomness import GeneratorSource, RandomSource, ReplaySource
from gqteashmd.electronic.base import MoleculeSpec
from gqteashmd.electronic.factory import make_backend
from gqteashmd.trajectory.model import BackendModel
from gqteashmd.trajectory.runner import InitialConditions, RunSettings, TrajectoryRunner
from gqteashmd.trajectory.store import StepLog

pytestmark = pytest.mark.pyscf

DATA = Path(__file__).resolve().parents[1] / "data"
# 6-31G* instead of the plan's STO-3G: changed after the measured numbers were seen (P7b Task 9,
# controller ruling), see the module docstring
ETHYLENE = (
    '[electronic]\nmethod = "casscf"\nbasis = "6-31g*"\nsinglets = 2\n'
    "[electronic.casscf]\nncas = 2\nnelecas = 2\n"
)
WATER_CIS = '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 3\n'
WATER_XYZ = """3
water
O    0.000000    0.000000    0.117300
H    0.000000    0.757200   -0.469200
H    0.000000   -0.757200   -0.469200
"""
TEMPERATURE_K = 300.0
VELOCITY_SEED = 2026
CONSERVING = {"hopping": "none", "decoherence": "none"}


def _write_input(directory: Path, electronic: str, xyz: str) -> Path:
    """The input of tests/conftest.py:write_input (callable from a module-scoped fixture)."""
    (directory / "molecule.xyz").write_text(xyz)
    path = directory / "input.toml"
    path.write_text(
        '[project]\nname = "test"\ndirectory = "run"\n\n[molecule]\ngeometry = "molecule.xyz"\n\n'
        "[resources]\ntotal_cores = 4\ntotal_memory_mb = 4000\n\n" + electronic
    )
    return path


class System:
    """An electronic configuration, a molecule and its initial conditions (start S1)."""

    def __init__(self, directory: Path, electronic: str, xyz: str) -> None:
        self.config = load_config(_write_input(directory, electronic, xyz)).config
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
        self.directory = directory

    def run(
        self,
        name: str,
        *,
        dt_fs: float,
        n_steps: int,
        initial: InitialConditions | None = None,
        seed: int = 0,
        source: RandomSource | None = None,
        **options,
    ) -> tuple[dict, np.ndarray]:
        """Run (or resume) trajectory `name` to n_steps steps of dt_fs; a new backend each call.

        `initial` defaults to (q, v, S1); `seed` seeds the hopping stream unless `source` is
        given. The result also holds `compute_calls`, the number of backend.compute calls of
        this call."""
        directory = self.directory / name
        backend = make_backend(
            self.config,
            self.molecule,
            settings=self.config.electronic,
            scratch_dir=self.directory / "scratch",
            log_file=self.directory / "e.log",
            threads=2,
            memory_mb=1000,
        )
        calls = 0
        compute = backend.compute

        def counted(request):
            nonlocal calls
            calls += 1
            return compute(request)

        backend.compute = counted  # type: ignore[method-assign]
        coupling = options.setdefault("coupling", "overlap")
        dt_au = dt_fs / AU_TIME_IN_FS
        settings = RunSettings(DynamicsOptions(dt_au=dt_au, **options), tmax_au=n_steps * dt_au)
        model = BackendModel(backend, self.masses, coupling=coupling)
        runner = TrajectoryRunner(model, settings, directory)
        if initial is None:
            initial = InitialConditions(self.q, self.v, 1)
        if source is None:
            source = GeneratorSource(np.random.default_rng(seed))
        result = runner.run(initial, source)
        rows = StepLog(directory / "steps.log", model.n_states, model.n_dof).read()
        return {**result, "compute_calls": calls}, rows


def _ethylene(directory: Path) -> System:
    directory.mkdir(parents=True, exist_ok=True)
    return System(directory, ETHYLENE, (DATA / "ethylene.xyz").read_text())


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


@pytest.fixture(scope="module")
def ethylene_run(tmp_path_factory):
    """Check (a)'s run, computed once for its two tests and reused by (b)."""
    system = _ethylene(tmp_path_factory.mktemp("ethylene_a"))
    result, rows = _timed(
        "(a) ethylene 6-31G* 20 steps of 0.5 fs",
        lambda: system.run("a", dt_fs=0.5, n_steps=20, **CONSERVING),
    )
    sv = rows["min_singular_value"][1:]
    numbers = (
        f"max |E_tot - E_tot(0)| = {_drift(rows):.3e} Eh; smallest singular value "
        f"{_smallest(rows)} (per record {np.array2string(sv, precision=4)}); rejections "
        f"{int(rows['rejections'].sum())} (result {result['n_rejections']}); {_brief(result)}"
    )
    print("(a)", numbers)
    return result, rows, numbers


def test_ethylene_casscf_runs_without_rejection_with_local_diabatization(ethylene_run):
    """(a) without its drift bound, on 6-31G*: changed after the measured numbers were seen (P7b
    Task 9, controller ruling): STO-3G lost its pi,pi* CASSCF solution after 1.06 fs (sv 0.035);
    the drift bound is the strict xfail below."""
    # changed after the measured numbers were seen (P7b Task 9, controller ruling): 6-31G*, split
    result, rows, numbers = ethylene_run
    assert result["outcome"] == "finished" and result["n_grid_steps"] == 20, numbers
    assert rows.shape == (21,), numbers
    assert np.all(rows["min_singular_value"][1:] >= 0.9), numbers
    assert int(rows["rejections"].sum()) == 0 and result["n_rejections"] == 0, numbers


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "changed after the measured numbers were seen (P7b Task 9, controller ruling): the plan's "
        "1e-4 Eh bound at dt 0.5 fs was fixed before any number and is too tight; measured "
        "2.68e-4 Eh (6-31G*, an excursion at 4.5-5.5 fs); energy conservation is validated by "
        "the dt^2 scaling of (b)"
    ),
)
def test_ethylene_casscf_drift_bound_at_half_a_femtosecond(ethylene_run):
    """(a)'s drift bound as a strict xfail: changed after the measured numbers were seen (P7b
    Task 9, controller ruling): measured 2.68e-4 Eh against 1e-4 Eh at dt 0.5 fs (6-31G*)."""
    # changed after the measured numbers were seen (P7b Task 9, controller ruling): strict xfail
    _, rows, numbers = ethylene_run
    assert _drift(rows) <= 1e-4, numbers


@pytest.mark.slow
def test_ethylene_casscf_drift_scales_as_dt_squared(ethylene_run, tmp_path):
    """(b) on 6-31G*: changed after the measured numbers were seen (P7b Task 9, controller
    ruling): STO-3G ended with overlap_failure at 1.06 fs. The dt 0.5 fs run to 5 fs is the first
    10 steps of (a)'s run (the steps do not depend on tmax); the window is the plan's."""
    # changed after the measured numbers were seen (P7b Task 9, controller ruling): 6-31G*
    result_a, rows_a, _ = ethylene_run
    assert result_a["outcome"] == "finished", _brief(result_a)
    half = rows_a[(rows_a["ticks"] == 0) & (rows_a["grid_step"] <= 10)]
    assert half.shape == (11,)
    result, rows = _timed(
        "(b) ethylene 6-31G* 5 fs at dt 0.25 fs",
        lambda: _ethylene(tmp_path).run("b", dt_fs=0.25, n_steps=20, **CONSERVING),
    )
    drifts = {0.5: _drift(half), 0.25: _drift(rows)}
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


ETHYLENE_DRIFT_GUARD_EH = 1e-3


def test_ethylene_casscf_energy_stays_consistent(ethylene_run):
    """A regression guard in the non-slow pyscf tier (P9a plan 2, spec 5): the bound 1e-3 Eh was
    set after the measured 2.68e-4 Eh of (a) (P7b Task 9), a factor 3.7 above it; the strict
    1e-4 xfail and the dt^2 ratio of (b) stay the validation."""
    _, rows, numbers = ethylene_run
    assert _drift(rows) <= ETHYLENE_DRIFT_GUARD_EH, numbers


@pytest.mark.slow
def test_ethylene_casscf_drift_ratio_at_the_finer_dt_pair(tmp_path):
    """(b) at dt 0.25 / 0.125 fs over 5 fs (P9a plan 2, spec 5; P7b measured the ratio 4.07 as a
    probe before this test existed; the bounds [3, 5] around the exact 4 are fixed in the plan)."""
    drifts = {}
    for dt in (0.25, 0.125):
        result, rows = _timed(
            f"(b') ethylene 6-31G* 5 fs at dt {dt} fs",
            lambda dt=dt: _ethylene(tmp_path / str(dt)).run(
                "b", dt_fs=dt, n_steps=int(round(5.0 / dt)), **CONSERVING
            ),
        )
        assert result["outcome"] == "finished" and result["n_rejections"] == 0, _brief(result)
        drifts[dt] = _drift(rows)
    ratio = drifts[0.25] / drifts[0.125]
    numbers = (
        f"drift(0.25) = {drifts[0.25]:.3e} Eh, drift(0.125) = {drifts[0.125]:.3e} Eh, "
        f"ratio {ratio:.3f}"
    )
    print("(b')", numbers)
    if drifts[0.25] < 1e-6:
        pytest.skip(f"drift(0.25 fs) below 1e-6 Eh, convergence noise dominates: {numbers}")
    assert 3.0 <= ratio <= 5.0, numbers


def test_ethylene_resume_from_a_restart_matches_one_run(tmp_path):
    """(c) on 6-31G*: changed after the measured numbers were seen (P7b Task 9, controller
    ruling): with STO-3G the 5-step first part already ended with overlap_failure at 1.06 fs."""
    # changed after the measured numbers were seen (P7b Task 9, controller ruling): 6-31G*
    system = _ethylene(tmp_path)
    ref_result, ref = _timed(
        "(c) reference 10 steps", lambda: system.run("ref", dt_fs=0.5, n_steps=10, **CONSERVING)
    )
    first, _ = _timed(
        "(c) first part 5 steps", lambda: system.run("cut", dt_fs=0.5, n_steps=5, **CONSERVING)
    )
    assert first["outcome"] == "finished" and first["n_grid_steps"] == 5, _brief(first)
    # Initial conditions and a seed the restart must override: a silent fresh start would
    # diverge from `ref` (other velocities, S0, another stream) instead of matching it.
    decoy = InitialConditions(system.q, 1.5 * system.v, 0)
    second, resumed = _timed(
        "(c) resumed to 10 steps",
        lambda: system.run("cut", dt_fs=0.5, n_steps=10, initial=decoy, seed=1, **CONSERVING),
    )
    assert ref_result["outcome"] == "finished", _brief(ref_result)
    assert second["outcome"] == "finished", _brief(second)
    assert second["n_grid_steps"] == 10 and second["n_steps"] == 10, _brief(second)
    print(
        f"(c) backend.compute calls: first part {first['compute_calls']}, resumed part "
        f"{second['compute_calls']}"
    )
    assert second["compute_calls"] == 5, second["compute_calls"]  # steps 6-10 only
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


def _water(directory: Path) -> System:
    return System(directory, WATER_CIS, WATER_XYZ)


def test_water_cis_drift_scales_as_dt_squared(tmp_path):
    """(d) re-targeted: changed after the measured numbers were seen (P7b Task 9, controller
    ruling): water CIS S1 is dissociative (E_kin 0.003 -> 0.145 Eh in 2 fs), so the drift is the
    O(dt^2) Verlet error (to 3 fs: 1.056e-3 / 2.636e-4 / 6.586e-5 Eh at dt 0.25 / 0.125 /
    0.0625 fs, ratios 4.006 and 4.003). Over 2 fs both runs must finish without rejection and
    drift(0.25)/drift(0.125) must lie in the plan's window [2.5, 6.0]."""
    # changed after the measured numbers were seen (P7b Task 9, controller ruling): ratio over 2 fs
    system = _water(tmp_path)
    drifts = {}
    for dt_fs, n_steps in ((0.25, 8), (0.125, 16)):
        result, rows = _timed(
            f"(d) water CIS 2 fs at dt {dt_fs} fs",
            lambda dt_fs=dt_fs, n_steps=n_steps: system.run(
                f"d_{dt_fs}", dt_fs=dt_fs, n_steps=n_steps, hopping="none"
            ),
        )
        drifts[dt_fs] = _drift(rows)
        print(f"(d) dt {dt_fs} fs: drift {drifts[dt_fs]:.3e} Eh; {_brief(result)}")
        assert result["outcome"] == "finished" and result["n_grid_steps"] == n_steps, _brief(result)
        assert result["n_rejections"] == 0 and int(rows["rejections"].sum()) == 0, _brief(result)
    ratio = drifts[0.25] / drifts[0.125]
    numbers = (
        f"drift(0.25 fs) = {drifts[0.25]:.3e} Eh, drift(0.125 fs) = {drifts[0.125]:.3e} Eh, "
        f"ratio {ratio:.3f}"
    )
    print("(d)", numbers)
    assert 2.5 <= ratio <= 6.0, numbers


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "changed after the measured numbers were seen (P7b Task 9, controller ruling): the plan's "
        "1e-4 Eh bound over 20 steps of 0.25 fs is not reachable on the dissociative CIS S1; "
        "measured at 28d8260: max |E_tot - E_tot(0)| = 1.056e-3 Eh, then overlap_failure at "
        "t = 3.594 fs (step dt/8, smallest singular value 2.78e-5; 4 rejections, 14 grid steps)"
    ),
)
def test_water_cis_conserves_energy_over_twenty_steps(tmp_path):
    """The plan's original (d) as a strict xfail: changed after the measured numbers were seen
    (P7b Task 9, controller ruling); the measured numbers are in the xfail reason."""
    # changed after the measured numbers were seen (P7b Task 9, controller ruling): strict xfail
    result, rows = _timed(
        "(d) water CIS 20 steps of 0.25 fs",
        lambda: _water(tmp_path).run("d", dt_fs=0.25, n_steps=20, hopping="none"),
    )
    drift = _drift(rows)
    numbers = (
        f"max |E_tot - E_tot(0)| = {drift:.3e} Eh; smallest singular value "
        f"{_smallest(rows)}; rejections {int(rows['rejections'].sum())}; {_brief(result)}; "
        f"{len(rows)} records"
    )
    print("(d original)", numbers)
    assert result["outcome"] == "finished" and result["n_grid_steps"] == 20, numbers
    assert drift <= 1e-4, numbers
    assert result["n_rejections"] == 0 and int(rows["rejections"].sum()) == 0, numbers


def test_water_cis_hops_with_pyscf(tmp_path):
    """Hops through the runner with PySCF (final review M1): xi = 0 hops whenever a probability
    is > 0, so refresh_active_gradient -> PySCFBackend.compute_more runs. The drift bound 5e-4 Eh
    was fixed before the run (the reviewer's probe saw 1.2e-4 Eh)."""
    result, rows = _timed(
        "(M1) water CIS 8 steps of 0.125 fs with hops",
        lambda: _water(tmp_path).run(
            "hops",
            dt_fs=0.125,
            n_steps=8,
            hopping="sharc",
            coupling="overlap",
            source=ReplaySource([0.0] * 400),
        ),
    )
    drift = _drift(rows)
    numbers = (
        f"n_hops {result['n_hops']}, n_frustrated {result['n_frustrated']}, max |E_tot - E_tot(0)| "
        f"= {drift:.3e} Eh; {_brief(result)}"
    )
    print("(M1)", numbers)
    assert result["outcome"] == "finished", numbers
    assert result["n_hops"] >= 1, numbers
    for name in rows.dtype.names:
        # record 0 has no random number and no overlap by design (NaN)
        values = rows[name][1:] if name in ("random_number", "min_singular_value") else rows[name]
        assert np.all(np.isfinite(values)), f"{name}: {numbers}"
    assert drift <= 5e-4, numbers
