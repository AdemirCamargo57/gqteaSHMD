"""One hopping-off PySCF trajectory against SHARC 4.1 (sharc.x + SHARC_PYSCF.py, black box),
and the swap test (sharc.x with our PySCFBackend behind QM.out).

P7b Task 10. SHARC is only executed, from ``tools/sharc_compare/run_sharc_pyscf_dynamics.py``
(our writers and parser; formats from the public SHARC 4.1 manual). System: ethylene
SA2-CASSCF(2,2)/6-31G* (controller decision before any SHARC run, from the Task 9 diagnosis:
the STO-3G pi,pi* solution is metastable and lost after about 1 fs), ``tests/data/ethylene.xyz``
rounded to SHARC's 7-decimal angstrom grid, the Task 9 velocities (T = 300 K,
``default_rng(2026)``, isotope masses), start in S1, hopping off, no decoherence,
``ekincorrect parallel_vel``, dt 0.5 fs, 25 substeps; ours with ``substep_interpolation =
"endpoint"`` (SHARC's scheme, measured in P7a-1). CASSCF convergence on both sides: conv-tol
1e-10, conv-tol-grad 1e-6 (the P3a template).

Coupling (deviation from the brief, forced by the black box and measured before any comparison):
SHARC_PYSCF.py has no overlaps ("Within the SHARC-PySCF interface, 'overlap' is not supported"
at step 1, also for the ``PHASES`` request), so the SHARC run uses ``coupling nacdr`` with
``nophases_from_interface``. Our trajectory is run with the brief's ``coupling = "overlap"`` and
also with ``coupling = "nacdr"`` (like for like). With hopping off, positions, velocities and
energies do not depend on the coupling.

Length: the timing probe (``--steps 2``, coupling nacdr) took 15.8 s for 3 printed steps
(5.3 s per interface call), so n = 40 (the largest of 10, 20, 40 under 15 min; estimated
3.6 min, measured 206-207 s for the 40-step SHARC run).

Tolerances (fixed in the brief before any run): positions <= 1e-6 bohr, velocities <= 1e-7 bohr
per atomic time unit, MCH energies <= 1e-6 Eh, populations <= 1e-4, at every compared step.

Changed after the measured numbers were seen (P7b Task 10, controller ruling), with the numbers
of the first runs (both runs identical to every printed digit):

- The compared window is steps 0-23. At step 24 both programs pass through the same CASSCF
  solution change (E0 -77.82183 -> -77.84681 Eh, E1 -77.73994 -> -77.70310 Eh, E_tot +0.976 eV
  in both); our energy-drift stop (0.5 eV) ends our runs there, which is the designed outcome
  and is asserted (``test_casscf_solution_change_at_step_24``).
- Positions are report-only: they exceed 1e-6 bohr from step 12 (max 2.5e-6 bohr over steps
  0-24). The swap test shows the drivers agree to 2.1e-7 bohr; the difference comes from the
  two independent CASSCF solutions (S1 gradients differ by up to 3.6e-6 Eh/bohr at steps 8-11
  with conv-tol-grad 1e-6). Tighter CASSCF convergence (keywords documented in manual table
  6.36) cannot be run: with conv-tol / conv-tol-grad = 1e-12 / 1e-8 (the ruling), 1e-12 / 1e-7
  and 1e-10 / 1e-7, SHARC_PYSCF.py stops at step 0 ("Calculator failed to converge!", "Gradient
  failed to converge"), and our SA-CASSCF fails on every rung at step 0 (1e-8) or at the first
  step after it, also at dt/8 (1e-7). So the comparison keeps the P3a convergence.
- The swap test is promoted from report to assertion after its numbers were seen (max over 41
  steps: dq 2.1e-7 bohr, dv 2.1e-9, dE 4.6e-8 Eh, dP 8.7e-7), at the brief's tolerances. Its
  reference trajectory of ours runs with ``energy_drift_limit_eV = 100`` so that it covers the
  same 40 steps as sharc.x.
- The direct population bound does not discriminate (review finding, P7b Task 10 fix round 2):
  over steps 0-23 P(S0) <= 2.4e-5 in both programs, so the 1e-4 bound cannot fail, while the
  relative P(S0) differences against SHARC reach a factor of about 3 (e.g. 9.6e-9 vs 3.4e-8 at
  step 7, ours overlap). The populations are validated by the swap test (the same electronic
  structure under both drivers, agreement to about 4 digits); the relative P(S0) differences of
  both our couplings are printed, report-only (``test_relative_s0_population_report``).

Run with ``-s`` to see the per-step differences.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from gqteashmd.core.units import HARTREE_IN_EV

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
# changed after the measured numbers were seen (P7b Task 10, controller ruling): the CASSCF
# solution change at step 24 (E_tot +0.976 eV in both programs) ends the compared window
CHANGE_STEP = 24
LAST_COMPARED = CHANGE_STEP - 1
SOLUTION_CHANGE_EV = 0.5
SWAP_DRIFT_LIMIT_EV = 100.0


@pytest.fixture(scope="module")
def harness():
    return _load("run_sharc_pyscf_dynamics")


@pytest.fixture(scope="module")
def initial(harness):
    return harness.ethylene_initial_conditions()


def _sharc(harness, initial, workdir, **kwargs):
    return harness.run_sharc_trajectory(
        initial.q_bohr,
        initial.v_au,
        n_steps=N_STEPS,
        workdir=workdir,
        symbols=initial.symbols,
        masses_amu=initial.masses_amu,
        sharc_home=Path(os.environ["SHARC_HOME"]),
        **kwargs,
    )


@pytest.fixture(scope="module")
def sharc_run(harness, initial, tmp_path_factory):
    data = _sharc(
        harness,
        initial,
        tmp_path_factory.mktemp("sharc") / "traj",
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


def _window(data, last):
    keep = data["steps"] <= last
    return {key: np.asarray(data[key])[keep] for key in ("steps", *TOLERANCES)}


def _compare(harness, ours, theirs, quantity, *, last, label):
    diffs = harness.step_differences(_window(ours, last), _window(theirs, last))
    assert diffs["steps"].tolist() == list(range(last + 1)), label
    tolerance = TOLERANCES[quantity]
    above = np.flatnonzero(diffs[quantity] > tolerance)
    first = int(diffs["steps"][above[0]]) if above.size else None
    numbers = (
        f"{label}, {quantity}, steps 0-{last}: max {diffs[quantity].max():.3e} "
        f"{UNITS[quantity]} (tolerance {tolerance:g}); first step above: {first}\n"
        f"{harness.format_differences(diffs)}"
    )
    print(numbers)
    return above, numbers


def test_casscf_solution_change_at_step_24(sharc_run, our_run):
    """Changed after the measured numbers were seen (P7b Task 10, controller ruling): instead of
    40 finished steps, both programs must show the CASSCF solution change at step 24 (measured:
    E_tot +0.976 eV in both) and our runs must end there with ``energy_drift``."""
    # changed after the measured numbers were seen (P7b Task 10, controller ruling)
    # brittle across processes and PySCF versions (see CLAUDE.md Known limitation 3a);
    # re-checked in P9a plan 2
    coupling, ours = our_run
    result = ours["result"]
    assert result["outcome"] == "energy_drift", result
    assert result["n_grid_steps"] == CHANGE_STEP and result["n_rejections"] == 0, result
    assert ours["steps"].tolist() == list(range(CHANGE_STEP + 1)), coupling
    assert sharc_run["steps"].tolist() == list(range(N_STEPS + 1))
    assert np.all(sharc_run["active"] == 1) and np.all(ours["active"] == 1)
    jumps = {
        name: (data["e_total"][CHANGE_STEP] - data["e_total"][LAST_COMPARED]) * HARTREE_IN_EV
        for name, data in (("SHARC", sharc_run), (f"ours ({coupling})", ours))
    }
    print(f"E_tot(24) - E_tot(23): {jumps}")
    assert all(abs(jump) > SOLUTION_CHANGE_EV for jump in jumps.values()), jumps
    before = np.abs(np.diff(sharc_run["e_total"][:CHANGE_STEP])).max() * HARTREE_IN_EV
    assert before < SOLUTION_CHANGE_EV, before  # the change is the first jump
    # the same time step and masses (SHARC converts fs and amu with its own constants)
    assert sharc_run["dt_au"] == pytest.approx(ours["dt_au"], rel=1e-9)
    np.testing.assert_allclose(sharc_run["masses_me"], ours["masses_me"], rtol=1e-9)


@pytest.mark.parametrize("quantity", ["v", "energies", "populations"])
def test_trajectory_matches_sharc_before_the_solution_change(harness, sharc_run, our_run, quantity):
    """Changed after the measured numbers were seen (P7b Task 10, controller ruling): compared
    over steps 0-23, before the CASSCF solution change at step 24 (the overlap populations
    differed by 6.4e-3 at step 24 itself).

    The populations case does not discriminate (review finding, P7b Task 10 fix round 2): over
    steps 0-23 P(S0) <= 2.4e-5 in both programs, so |dP| <= 1e-4 cannot fail, and the relative
    P(S0) differences reach a factor of about 3; the populations are validated by the swap test
    (``test_swap_matches_our_trajectory``), the relative numbers are reported by
    ``test_relative_s0_population_report``."""
    # changed after the measured numbers were seen (P7b Task 10, controller ruling): steps 0-23
    coupling, ours = our_run
    above, numbers = _compare(
        harness,
        ours,
        sharc_run,
        quantity,
        last=LAST_COMPARED,
        label=f"ours ({coupling}) vs SHARC (nacdr)",
    )
    assert above.size == 0, numbers


def test_positions_before_the_solution_change(harness, sharc_run, our_run):
    """Report-only. Changed after the measured numbers were seen (P7b Task 10, controller
    ruling): positions exceeded 1e-6 bohr from step 12 (max 2.4e-6 over steps 0-23); the swap
    test shows the drivers agree, the difference is CASSCF convergence (conv-tol-grad 1e-6 on
    both sides), and no tighter setting converges in SHARC_PYSCF.py (1e-12 / 1e-8, 1e-12 / 1e-7,
    1e-10 / 1e-7: all stop at step 0; see the module docstring)."""
    # changed after the measured numbers were seen (P7b Task 10, controller ruling): report only
    coupling, ours = our_run
    diffs = harness.step_differences(
        _window(ours, LAST_COMPARED), _window(sharc_run, LAST_COMPARED)
    )
    _compare(
        harness,
        ours,
        sharc_run,
        "q",
        last=LAST_COMPARED,
        label=f"REPORT ONLY: ours ({coupling}) vs SHARC (nacdr)",
    )
    report_dir = os.environ.get("GQTEASHMD_REPORT_DIR")
    if report_dir:
        path = Path(report_dir) / "sharc_pyscf_positions.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            report = json.loads(path.read_text())
        except (OSError, ValueError):
            report = {}
        report[coupling] = {
            "steps": [int(s) for s in diffs["steps"]],
            "max_abs_q_bohr": [float(x) for x in diffs["q"]],
        }
        path.write_text(json.dumps(report, indent=1) + "\n")


def test_relative_s0_population_report(harness, sharc_run, our_run):
    """Report-only (review finding, P7b Task 10 fix round 2): P(S0) of ours and SHARC and
    |P_ours - P_SHARC| / P_SHARC per step over steps 0-23, for both our couplings (SHARC's side
    uses nacdr). Only the window and the bound P(S0) <= 1e-4 are checked: the absolute
    population bound is meaningless when P(S0) itself stays below it."""
    # review finding, P7b Task 10 fix round 2: report only
    coupling, ours = our_run
    data = harness.relative_population_differences(ours, sharc_run, last=LAST_COMPARED)
    assert data["steps"] == list(range(LAST_COMPARED + 1))
    relative = [r for r in data["relative_difference"] if r is not None]
    print(
        f"REPORT ONLY: P(S0), ours ({coupling}) vs SHARC (nacdr), steps 0-{LAST_COMPARED}: "
        f"max P(S0) ours {max(data['ours']):.3e}, SHARC {max(data['reference']):.3e}; "
        f"max relative difference {max(relative):.3f}\n"
        f"{harness.format_relative_populations(data)}"
    )
    assert max(data["ours"]) < TOLERANCES["populations"]  # the reason the bound is blind


# --- the swap test: sharc.x with our PySCFBackend behind QM.out ------------------------------


@pytest.fixture(scope="module")
def swap_runs(harness, initial, tmp_path_factory):
    """Our overlap trajectory over the full 40 steps (energy-drift stop lifted) and sharc.x
    with our PySCFBackend (``SwapModel``, coupling overlap) behind QM.out."""
    work = tmp_path_factory.mktemp("swap")
    ours = harness.run_our_trajectory(
        initial,
        n_steps=N_STEPS,
        workdir=work / "ours",
        coupling="overlap",
        energy_drift_limit_eV=SWAP_DRIFT_LIMIT_EV,
    )
    model = harness.SwapModel(harness.make_ethylene_backend(work / "scratch"), coupling="overlap")
    swap = _sharc(harness, initial, work / "swap", provider="ours", coupling="overlap", model=model)
    print(
        f"swap: ours {ours['wall_time_s']:.1f} s ({ours['result']['outcome']}), sharc.x + ours "
        f"{swap['wall_time_s']:.1f} s"
    )
    return ours, swap


@pytest.mark.parametrize("quantity", ["q", "v", "energies", "populations"])
def test_swap_matches_our_trajectory(harness, swap_runs, quantity):
    """Promoted after its numbers were seen (P7b Task 10, controller ruling): the swap report
    measured max dq 2.1e-7 bohr, dv 2.1e-9, dE 4.6e-8 Eh, dP 8.7e-7 over steps 0-40; asserted
    at the brief's tolerances over all 41 steps."""
    # changed after the measured numbers were seen (P7b Task 10, controller ruling): promoted
    ours, swap = swap_runs
    assert ours["result"]["outcome"] == "finished", ours["result"]
    above, numbers = _compare(
        harness, swap, ours, quantity, last=N_STEPS, label="sharc.x + our PySCFBackend vs ours"
    )
    assert above.size == 0, numbers


F1_EV = 0.1  # the [dynamics] default of energy_jump_limit_eV (P9a plan 2, Task 2)


@pytest.fixture(scope="module", params=["overlap", "nacdr"])
def our_run_f1(request, harness, initial, tmp_path_factory):
    data = harness.run_our_trajectory(
        initial,
        n_steps=N_STEPS,
        workdir=tmp_path_factory.mktemp(f"ours_f1_{request.param}"),
        coupling=request.param,
        energy_jump_limit_eV=F1_EV,
    )
    print(f"ours with F1 ({request.param}): {data['result']}")
    return request.param, data


def test_the_energy_check_ends_the_run_at_the_step_24_solution_change(harness, our_run_f1):
    """P9a plan 2: with the F1 default, the CASSCF solution change at step 24 (E_tot +0.976 eV)
    ends our run as electronic_discontinuity inside the 24th interval.

    changed after the measured numbers were seen (P9a plan 2 Task 2, controller ruling P9a2-4):
    the plan asserted n_grid_steps == 24; with F1 the jumping 24th interval is rejected, so 23
    grid steps complete and the run ends inside the 24th interval (measured: t_final 11.8125 fs,
    rejections {'energy': 4}, jump 0.9431 eV, both couplings)."""
    coupling, ours = our_run_f1
    result = ours["result"]
    assert result["outcome"] == "electronic_discontinuity", (coupling, result)
    assert result["n_grid_steps"] == CHANGE_STEP - 1, (coupling, result)
    assert (CHANGE_STEP - 1) * harness.DT_FS < result["t_final_fs"] < CHANGE_STEP * harness.DT_FS
    assert result["rejections_by_signal"].get("energy", 0) >= 1, (coupling, result)
