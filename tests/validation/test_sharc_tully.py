"""The dynamics core against the SHARC 4.1 driver (sharc.x, run as a black box) on Tully models.

SHARC is only executed, from ``tools/sharc_compare/run_sharc_model.py`` (our writers/parsers;
formats from the public SHARC 4.1 manual). The model is one atom of mass 2000 m_e moving
along x. Two providers stand behind SHARC's QM/runQM.sh: SHARC's own SHARC_ANALYTICAL.py, and
our awk implementation of the same diabatic matrices (``fast``, about 100x faster), which
``test_fast_provider_matches_sharc_analytical`` shows gives the same SHARC trajectory.

Findings (details in the Task 9 report):

- Nuclear motion (velocity Verlet) agrees to 1e-10 bohr.
- SHARC interpolates H and v.K at the substep *ends*, s = i/N (manual eqs. 8.191-8.194). With
  ``substep_interpolation = "endpoint"`` our core follows it (|c|^2 to < 1e-6); our default
  midpoints, s = (i - 1/2)/N, differ by a first-order term in the substep length (reported).
- SHARC's EDC damps the amplitudes by exp(-dt/(2 tau)) (manual eq. 8.25), which is our
  ``edc_convention = "sharc"`` (the default); ``"granucci_persico"`` damps by exp(-dt/tau).
  ``test_edc_rate_definition`` measures both against SHARC.

Every test here compares single trajectories and is asserted. The ensemble comparison of
branching ratios (SAC k = 20, ECR k = 20, n = 500, about 1 h) is a report script,
``tools/sharc_compare/tully_ensembles.py``: its z-test could not detect the factor 2 of the
EDC rate that ``test_edc_rate_definition`` measures directly.

Run with ``-s`` to see every number.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from gqteashmd.dynamics.driver import initial_state, run_trajectory
from gqteashmd.dynamics.models import (
    TullyDualAvoidedCrossing,
    TullyExtendedCoupling,
    TullySimpleAvoidedCrossing,
)
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.randomness import ReplaySource

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


rsm = _load("run_sharc_model")

pytestmark = [
    pytest.mark.sharc,
    pytest.mark.slow,
    pytest.mark.skipif(
        not (os.environ.get("SHARC_HOME") and os.environ.get("SHARC_PYTHON")),
        reason="needs SHARC_HOME and SHARC_PYTHON",
    ),
]

AU_TIME_IN_FS = rsm.AU_TIME_IN_FS
MODELS = {
    "SAC": TullySimpleAvoidedCrossing,
    "DAC": TullyDualAvoidedCrossing,
    "ECR": TullyExtendedCoupling,
}


def _home() -> Path:
    return Path(os.environ["SHARC_HOME"])


def _python() -> str:
    return os.environ["SHARC_PYTHON"]


def _sharc(cls, x0, v0, n_steps, options: DynamicsOptions, seed, work_dir, **kw):
    return rsm.run(
        _home(),
        _python(),
        rsm.model_spec(cls()),
        x0,
        v0,
        n_steps,
        options.dt_au * AU_TIME_IN_FS,
        rsm.SharcOptions.from_dynamics_options(options),
        seed,
        work_dir,
        **kw,
    )


def _ours(cls, sharc_traj, x0, v0, n_steps, options, random_numbers=None):
    """Our core with SHARC's mass (header of output.dat) and, optionally, its random numbers."""
    model = cls(mass=float(sharc_traj.masses_me[0]))
    values = random_numbers if random_numbers is not None else [0.5] * n_steps
    records = []
    state = initial_state(model, np.array([x0]), np.array([v0]), 0)
    run_trajectory(
        model, state, options, ReplaySource(values), max_steps=n_steps, on_step=records.append
    )
    return records


def _documented_propagation(traj, nsubsteps: int, scheme: str) -> np.ndarray:
    """c(t) from SHARC's own E, K and v (output.dat) with an exponential substep propagator.

    Substep i (1..N) uses R_i = exp(-(i H(s) + T(s)) dt/N), applied in time order, with
    - ``"sharc"`` (manual eqs. 8.191-8.194): s = i/N, H and T = v.K interpolated linearly;
    - ``"ours"`` (method reference 2): s = (i - 1/2)/N, H, K and v interpolated separately.
    """
    delta = traj.dt_au / nsubsteps
    c = traj.c[0].astype(complex)
    out = [c]
    for k in range(len(traj.steps) - 1):
        e0, e1 = traj.energies[k], traj.energies[k + 1]
        k0, k1 = traj.nacs[k, :, :, 0], traj.nacs[k + 1, :, :, 0]
        v0, v1 = traj.v[k, 0], traj.v[k + 1, 0]
        for i in range(1, nsubsteps + 1):
            if scheme == "sharc":
                s = i / nsubsteps
                t = v0 * k0 + s * (v1 * k1 - v0 * k0)
            else:
                s = (i - 0.5) / nsubsteps
                t = (v0 + s * (v1 - v0)) * (k0 + s * (k1 - k0))
            h = np.diag(e0 + s * (e1 - e0))
            c = expm(-(1j * h + t) * delta) @ c
        out.append(c)
    return np.array(out)


# --- 1. the fast provider is SHARC's analytical interface, faster -----------------------------


@pytest.mark.parametrize("name", ("SAC", "DAC", "ECR"))
def test_fast_provider_matches_sharc_analytical(name, tmp_path):
    """Same SHARC run (hopping on, seed 7) with SHARC_ANALYTICAL.py and with our awk QM.out.

    x from -0.4 to +0.4 crosses both branches of the piecewise SAC/ECR potentials.
    """
    options = DynamicsOptions(dt_au=1.0, nsubsteps=25, hopping="sharc", decoherence="none")
    runs = {}
    for provider in ("analytical", "fast"):
        start = time.perf_counter()
        runs[provider] = _sharc(
            MODELS[name], -0.4, 0.02, 40, options, 7, tmp_path / provider, provider=provider
        )
        print(f"\n[{name}] {provider}: {time.perf_counter() - start:.1f} s for 40 steps")
    a, f = runs["analytical"], runs["fast"]
    assert len(a.steps) == len(f.steps) == 41
    # The sign of the upper adiabatic state is a gauge: it may differ (it does for DAC). It
    # flips the NAC and c_2 together and changes no observable.
    sign = float(np.sign(np.sum(a.nacs[:, 0, 1, 0] * f.nacs[:, 0, 1, 0])))
    gauge = np.array([1.0, sign])
    diffs = {
        "q": np.max(np.abs(a.q - f.q)),
        "v": np.max(np.abs(a.v - f.v)),
        "|c|^2": np.max(np.abs(np.abs(a.c) ** 2 - np.abs(f.c) ** 2)),
        "c": np.max(np.abs(a.c - f.c * gauge)),
        "energies": np.max(np.abs(a.energies - f.energies)),
        "gradients": np.max(np.abs(a.gradients - f.gradients)),
        "nacs": np.max(np.abs(a.nacs - sign * f.nacs)),
        "probabilities": np.max(np.abs(a.probabilities - f.probabilities)),
    }
    print(f"  gauge sign {sign:+.0f}  " + "  ".join(f"{k} {v:.1e}" for k, v in diffs.items()))
    assert np.array_equal(a.active, f.active)
    assert np.array_equal(a.random_numbers, f.random_numbers)
    for key, value in diffs.items():
        assert value < 1e-8, key


# --- 2. one trajectory with hopping off, step by step -----------------------------------------

STEP_X0, STEP_V0, STEP_N = -1.0, 0.02, 200  # SAC, k = 40: x from -1 to +1 through the crossing


def _compare_steps(label, sharc, ours):
    """Print and return (max|dx|, max|dv|, |d|c0|^2| per step) of our records vs SHARC."""
    q = np.array([r.q[0] for r in ours])
    v = np.array([r.v[0] for r in ours])
    pop = np.array([abs(r.c[0]) ** 2 for r in ours])
    s_pop = np.abs(sharc.c[1:, 0]) ** 2
    dq, dv, dp = np.abs(q - sharc.q[1:, 0]), np.abs(v - sharc.v[1:, 0]), np.abs(pop - s_pop)
    print(
        f"\n[SAC hopping off, {label}] x {STEP_X0} -> {q[-1]:.4f}, mass "
        f"{sharc.masses_me[0]!r} m_e; final |c_0|^2 ours {pop[-1]:.10f} SHARC {s_pop[-1]:.10f}"
    )
    for k in (1, 10, 25, 50, 100, 150, 200):
        print(
            f"  steps 1-{k:3d}: max|dx| {dq[:k].max():.2e}  max|dv| {dv[:k].max():.2e}  "
            f"max|d|c0|^2| {dp[:k].max():.2e}"
        )
    print(f"  |d|c0|^2| per step 1..10: {' '.join(f'{d:.1e}' for d in dp[:10])}")
    return dq, dv, dp, pop, s_pop


def test_hopping_off_step_by_step(tmp_path):
    """SHARC_ANALYTICAL.py run against our core with the end-point substep interpolation
    (SHARC's scheme; asserted) and with our midpoint default (the documented difference)."""
    endpoint = DynamicsOptions(
        dt_au=0.5,
        nsubsteps=25,
        hopping="none",
        decoherence="none",
        substep_interpolation="endpoint",
    )
    sharc = _sharc(
        TullySimpleAvoidedCrossing, STEP_X0, STEP_V0, STEP_N, endpoint, 1, tmp_path / "sharc"
    )
    assert sharc.dt_au == 0.5 and len(sharc.steps) == STEP_N + 1
    assert np.all(sharc.active == 0)

    # End-point interpolation: our core follows SHARC.
    ours = _ours(TullySimpleAvoidedCrossing, sharc, STEP_X0, STEP_V0, STEP_N, endpoint)
    dq, dv, dp, pop, s_pop = _compare_steps("endpoint", sharc, ours)
    end = _documented_propagation(sharc, 25, "sharc")
    d_end_ours = np.max(np.abs(np.abs(end[1:, 0]) ** 2 - pop))
    d_end = np.max(np.abs(np.abs(end[:, 0]) ** 2 - np.abs(sharc.c[:, 0]) ** 2))
    print(
        f"  on SHARC's E, K, v: manual scheme vs SHARC max|d|c0|^2| {d_end:.2e}; "
        f"vs our endpoint core {d_end_ours:.2e}"
    )
    assert dq.max() < 1e-8 and dv.max() < 1e-10  # the same velocity Verlet
    assert dp[:50].max() < 1e-6
    assert dp.max() < 2e-6
    assert d_end_ours < 1e-8  # the manual's scheme on SHARC's data is our endpoint core
    assert d_end < 2e-6  # and reproduces SHARC (a residual ~ dt^2, not ~ 1/N, remains)

    # Midpoint (our default): the documented first-order substep difference, reported.
    midpoint = replace(endpoint, substep_interpolation="midpoint")
    ours_mid = _ours(TullySimpleAvoidedCrossing, sharc, STEP_X0, STEP_V0, STEP_N, midpoint)
    _, _, dp_mid, pop_mid, _ = _compare_steps("midpoint", sharc, ours_mid)
    mid = _documented_propagation(sharc, 25, "ours")
    d_mid_ours = np.max(np.abs(np.abs(mid[1:, 0]) ** 2 - pop_mid))
    print(f"  our midpoint scheme on SHARC's E, K, v vs our core max|d|c0|^2| {d_mid_ours:.2e}")
    assert d_mid_ours < 1e-8
    assert abs(pop_mid[-1] - s_pop[-1]) < 1e-5  # after the crossing the populations agree

    # The midpoint difference is first order in the substep: N = 25 -> 50 halves it.
    peaks = {}
    for nsub in (25, 50):
        opt = replace(midpoint, nsubsteps=nsub)
        s = _sharc(
            TullySimpleAvoidedCrossing,
            STEP_X0,
            STEP_V0,
            STEP_N,
            opt,
            1,
            tmp_path / f"fast{nsub}",
            provider="fast",
        )
        o = _ours(TullySimpleAvoidedCrossing, s, STEP_X0, STEP_V0, STEP_N, opt)
        p_ours = np.array([abs(r.c[0]) ** 2 for r in o])
        peaks[nsub] = np.max(np.abs(p_ours - np.abs(s.c[1:, 0]) ** 2))
    ratio = peaks[25] / peaks[50]
    print(f"  midpoint max|d|c0|^2| N=25 {peaks[25]:.3e}, N=50 {peaks[50]:.3e}, ratio {ratio:.3f}")
    assert 1.8 < ratio < 2.2
    assert peaks[25] < 1e-3


# --- 3. hopping on, SHARC's random numbers replayed in our core --------------------------------


def test_hopping_with_replayed_random_numbers(tmp_path):
    """SAC (x -2 -> 2, k = 40, dt = 1, end-point substeps): with SHARC's random number of every
    step our core hops at the same steps to the same states, with the same velocities."""
    options = DynamicsOptions(
        dt_au=1.0,
        nsubsteps=25,
        hopping="sharc",
        decoherence="none",
        substep_interpolation="endpoint",
    )
    s = _sharc(
        TullySimpleAvoidedCrossing, -2.0, 0.02, 200, options, 9, tmp_path / "s", provider="fast"
    )
    ours = _ours(
        TullySimpleAvoidedCrossing, s, -2.0, 0.02, 200, options, list(s.random_numbers[1:])
    )
    hops = [(r.step, r.hop_from, r.hop_to) for r in ours if r.hop_to is not None]
    active = np.array([r.active for r in ours])
    dv = np.max(np.abs(np.array([r.v[0] for r in ours]) - s.v[1:, 0]))
    p_ours = np.array([r.probabilities for r in ours])
    dprob = np.max(np.abs(p_ours - s.probabilities[1:]))
    print(f"\n[SAC hopping] our hops {hops}; SHARC: {s.hop_lines}")
    print(f"  max|dv| {dv:.2e}; max|d probability| {dprob:.2e}")
    assert hops and len(hops) == len(s.hop_lines)
    assert np.array_equal(active, s.active[1:])
    assert dv < 1e-8
    assert dprob < 1e-5  # measured 6.5e-7 with end-point substeps


# --- 4. EDC: SHARC damps at half our rate -------------------------------------------------------


def _edc_kappa(c, active, energies, e_kin, dt, steps, param=0.1):
    """kappa with |c_j(t+dt)| = |c_j(t)| exp(-kappa dt/tau), tau = (1 + C/K)/|dE| (no coupling)."""
    values = []
    for k in steps:
        j = 1 - active[k]
        tau = (1.0 + param / e_kin[k + 1]) / abs(energies[k + 1][1] - energies[k + 1][0])
        values.append(-math.log(abs(c[k + 1][j]) / abs(c[k][j])) * tau / dt)
    return np.array(values)


def test_edc_rate_definition(tmp_path):
    """Far from the coupling (SAC, x > 5) only EDC changes |c_inactive|: SHARC kappa = 1/2
    (manual eq. 8.25); ours kappa = 1/2 with ``edc_convention = "sharc"`` and 1 with
    ``"granucci_persico"``. Same random numbers, same hop."""
    options = DynamicsOptions(
        dt_au=1.0,
        nsubsteps=25,
        hopping="sharc",
        decoherence="edc",
        edc_convention="sharc",
        substep_interpolation="endpoint",
    )
    s = _sharc(
        TullySimpleAvoidedCrossing, -2.0, 0.02, 500, options, 9, tmp_path / "s", provider="fast"
    )
    steps = [k for k in range(len(s.steps) - 1) if s.q[k, 0] > 5.0][:20]
    assert len(steps) == 20
    k_sharc = _edc_kappa(s.c, s.active, s.energies, s.e_kinetic, 1.0, steps)
    print(f"\n[EDC] kappa SHARC {k_sharc.min():.6f}..{k_sharc.max():.6f}; hops {s.hop_lines}")
    assert np.allclose(k_sharc, 0.5, atol=0.01)
    pop_inactive = {}
    for convention, expected in (("sharc", 0.5), ("granucci_persico", 1.0)):
        opt = replace(options, edc_convention=convention)
        ours = _ours(
            TullySimpleAvoidedCrossing, s, -2.0, 0.02, 500, opt, list(s.random_numbers[1:])
        )
        c_ours = [s.c[0]] + [r.c for r in ours]
        a_ours = [s.active[0]] + [r.active for r in ours]
        e_ours = [s.energies[0]] + [r.energies for r in ours]
        kin = [s.e_kinetic[0]] + [r.e_kinetic for r in ours]
        k_ours = _edc_kappa(c_ours, a_ours, e_ours, kin, 1.0, steps)
        pop_inactive[convention] = [abs(c_ours[k][1 - a_ours[k]]) ** 2 for k in (150, 200, 300)]
        print(f"  ours ({convention}): kappa {k_ours.min():.6f}..{k_ours.max():.6f}")
        if convention == "sharc":
            assert np.array_equal(np.array(a_ours[1:]), s.active[1:])
        assert np.allclose(k_ours, expected, atol=0.01)
    s_inactive = [abs(s.c[k][1 - s.active[k]]) ** 2 for k in (150, 200, 300)]
    print(
        "  |c_inactive|^2 at steps 150/200/300: SHARC "
        + " ".join(f"{p:.6f}" for p in s_inactive)
        + "; ours sharc "
        + " ".join(f"{p:.6f}" for p in pop_inactive["sharc"])
        + "; ours granucci_persico "
        + " ".join(f"{p:.6f}" for p in pop_inactive["granucci_persico"])
    )
    # Agreement to ~1e-4 relative (measured 3.2e-5 at step 150; dt = 1 au, so the dt^2
    # propagator residual of section 2 is ~4x larger here).
    assert np.allclose(pop_inactive["sharc"], s_inactive, rtol=5e-4, atol=0.0)
