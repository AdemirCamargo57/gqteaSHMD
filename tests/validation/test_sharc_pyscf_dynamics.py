"""One hopping-off PySCF trajectory against SHARC 4.1 (sharc.x + SHARC_PYSCF.py, black box).

P7b Task 10. SHARC is only executed, from ``tools/sharc_compare/run_sharc_pyscf_dynamics.py``
(our writers and parser; formats from the public SHARC 4.1 manual). System: ethylene
SA2-CASSCF(2,2)/6-31G* (controller decision before any SHARC run, from the Task 9 diagnosis:
the STO-3G pi,pi* solution is metastable and lost after about 1 fs), ``tests/data/ethylene.xyz``
rounded to SHARC's 7-decimal angstrom grid, the Task 9 velocities (T = 300 K,
``default_rng(2026)``, isotope masses), start in S1, hopping off, no decoherence,
``ekincorrect parallel_vel``, dt 0.5 fs, 25 substeps; ours with ``substep_interpolation =
"endpoint"`` (SHARC's scheme, measured in P7a-1).

Coupling (deviation from the brief, forced by the black box and measured before any comparison):
SHARC_PYSCF.py has no overlaps ("Within the SHARC-PySCF interface, 'overlap' is not supported"
at step 1, also for the ``PHASES`` request), so the SHARC run uses ``coupling nacdr`` with
``nophases_from_interface``. Our trajectory is run with the brief's ``coupling = "overlap"`` and
also with ``coupling = "nacdr"`` (like for like). With hopping off, positions, velocities and
energies do not depend on the coupling; every tolerance is asserted against both of our runs.

Length: the timing probe (``--steps 2``, coupling nacdr) took 15.8 s for 3 printed steps
(5.3 s per interface call), so n = 40 (the largest of 10, 20, 40 under 15 min; estimated
3.6 min, measured 207 s for the 40-step SHARC run).

Tolerances (fixed in the brief before any run): positions <= 1e-6 bohr, velocities <= 1e-7 bohr
per atomic time unit, MCH energies <= 1e-6 Eh, populations <= 1e-4, at every step.

Run with ``-s`` to see the per-step differences.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "sharc_compare"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pytestmark = [
    pytest.mark.sharc,
    pytest.mark.pyscf,
    pytest.mark.slow,
    pytest.mark.skipif(
        not (os.environ.get("SHARC_HOME") and os.environ.get("SHARC_PYTHON")),
        reason="needs SHARC_HOME and SHARC_PYTHON",
    ),
]

N_STEPS = 40
TOLERANCES = {"q": 1e-6, "v": 1e-7, "energies": 1e-6, "populations": 1e-4}
UNITS = {"q": "bohr", "v": "bohr/au", "energies": "Eh", "populations": ""}


@pytest.fixture(scope="module")
def harness():
    return _load("run_sharc_pyscf_dynamics")


@pytest.fixture(scope="module")
def initial(harness):
    return harness.ethylene_initial_conditions()


@pytest.fixture(scope="module")
def sharc_run(harness, initial, tmp_path_factory):
    data = harness.run_sharc_trajectory(
        initial.q_bohr,
        initial.v_au,
        n_steps=N_STEPS,
        workdir=tmp_path_factory.mktemp("sharc") / "traj",
        symbols=initial.symbols,
        masses_amu=initial.masses_amu,
        sharc_home=Path(os.environ["SHARC_HOME"]),
        sharc_python=os.environ["SHARC_PYTHON"],
        provider="sharc_pyscf",
        coupling="nacdr",
    )
    print(f"sharc.x + SHARC_PYSCF.py: {N_STEPS} steps, wall time {data['wall_time_s']:.1f} s")
    return data


@pytest.fixture(scope="module", params=["overlap", "nacdr"])
def our_run(request, harness, initial, tmp_path_factory):
    data = harness.run_our_trajectory(
        initial,
        n_steps=N_STEPS,
        workdir=tmp_path_factory.mktemp(f"ours_{request.param}"),
        coupling=request.param,
    )
    print(f"ours ({request.param}): {data['result']}")
    return request.param, data


def test_both_trajectories_ran_every_step(sharc_run, our_run):
    coupling, ours = our_run
    result = ours["result"]
    assert result["outcome"] == "finished" and result["n_rejections"] == 0, result
    assert ours["steps"].tolist() == list(range(N_STEPS + 1)), coupling
    assert sharc_run["steps"].tolist() == list(range(N_STEPS + 1))
    assert np.all(sharc_run["active"] == 1) and np.all(ours["active"] == 1)
    # the same time step and masses (SHARC converts fs and amu with its own constants)
    assert sharc_run["dt_au"] == pytest.approx(ours["dt_au"], rel=1e-9)
    np.testing.assert_allclose(sharc_run["masses_me"], ours["masses_me"], rtol=1e-9)


@pytest.mark.parametrize("quantity", ["q", "v", "energies", "populations"])
def test_trajectory_matches_sharc_at_every_step(harness, sharc_run, our_run, quantity):
    """Every step both programs printed (a shorter run is the failure of the test above)."""
    coupling, ours = our_run
    diffs = harness.step_differences(ours, sharc_run)
    assert diffs["steps"].size and diffs["steps"][0] == 0
    table = harness.format_differences(diffs)
    tolerance = TOLERANCES[quantity]
    above = np.flatnonzero(diffs[quantity] > tolerance)
    first = int(diffs["steps"][above[0]]) if above.size else None
    numbers = (
        f"ours ({coupling}) vs SHARC (nacdr), {quantity}, steps 0-{int(diffs['steps'][-1])}: "
        f"max {diffs[quantity].max():.3e} "
        f"{UNITS[quantity]} (tolerance {tolerance:g}); first step above: {first}\n{table}"
    )
    print(numbers)
    assert above.size == 0, numbers
