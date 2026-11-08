"""The dynamics core against the SHARC 4.1 driver (sharc.x, run as a black box) on Tully models.

SHARC is only executed, from ``tools/sharc_compare/run_sharc_model.py`` (our writers/parsers;
formats from the public SHARC 4.1 manual). The model is one atom of mass 2000 m_e moving
along x. Two providers stand behind SHARC's QM/runQM.sh: SHARC's own SHARC_ANALYTICAL.py, and
our awk implementation of the same diabatic matrices (``fast``, about 100x faster), which
``test_fast_provider_matches_sharc_analytical`` shows gives the same SHARC trajectory.

Findings (details in the Task 9 report):

- Nuclear motion (velocity Verlet) agrees to 1e-10 bohr.
- The amplitudes differ systematically by a first-order term in the substep length: SHARC
  interpolates H and v.K at the substep *ends*, s = i/N (manual eqs. 8.191-8.194); we use the
  *midpoints*, s = (i - 1/2)/N (method reference 2). The difference halves when N doubles, and
  propagating SHARC's own E, K, v with the manual's formula reproduces SHARC's |c|^2.
- SHARC's EDC damps the amplitudes by exp(-dt/(2 tau)) (manual eq. 8.25); ours by
  exp(-dt/tau) (method reference 7, Granucci & Persico 2007). SHARC's decoherence rate is half
  ours; ``test_edc_rate_definition`` measures it.

Run with ``-s`` to see every number.
"""

from __future__ import annotations

import importlib.util
import math
import os
import shutil
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from gqteashmd.dynamics import ensemble
from gqteashmd.dynamics.driver import initial_state, run_trajectory
from gqteashmd.dynamics.ensemble import BRANCHES, run_model_ensemble, wavepacket_initial_conditions
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

SEED = 20261108
SHARC_WORKERS = 14
OUR_WORKERS = 8
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


def test_hopping_off_step_by_step(tmp_path):
    options = DynamicsOptions(dt_au=0.5, nsubsteps=25, hopping="none", decoherence="none")
    sharc = _sharc(
        TullySimpleAvoidedCrossing, STEP_X0, STEP_V0, STEP_N, options, 1, tmp_path / "sharc"
    )
    assert sharc.dt_au == 0.5 and len(sharc.steps) == STEP_N + 1
    assert np.all(sharc.active == 0)
    ours = _ours(TullySimpleAvoidedCrossing, sharc, STEP_X0, STEP_V0, STEP_N, options)
    q = np.array([r.q[0] for r in ours])
    v = np.array([r.v[0] for r in ours])
    pop = np.array([abs(r.c[0]) ** 2 for r in ours])
    s_pop = np.abs(sharc.c[1:, 0]) ** 2
    dq, dv, dp = np.abs(q - sharc.q[1:, 0]), np.abs(v - sharc.v[1:, 0]), np.abs(pop - s_pop)
    print(
        f"\n[SAC hopping off] x {STEP_X0} -> {q[-1]:.4f}, mass {sharc.masses_me[0]!r} m_e; "
        f"final |c_0|^2 ours {pop[-1]:.10f} SHARC {s_pop[-1]:.10f}"
    )
    for k in (1, 10, 25, 50, 100, 150, 200):
        print(
            f"  steps 1-{k:3d}: max|dx| {dq[:k].max():.2e}  max|dv| {dv[:k].max():.2e}  "
            f"max|d|c0|^2| {dp[:k].max():.2e}"
        )
    print(f"  |d|c0|^2| per step 1..10: {' '.join(f'{d:.1e}' for d in dp[:10])}")

    # Nuclear motion: the same velocity Verlet.
    assert dq.max() < 1e-8 and dv.max() < 1e-10
    # Amplitudes: the documented substep-quadrature difference (module docstring), so the
    # brief's 1e-6 for the first 50 steps is not met (measured ~2e-5); asserted instead:
    end = _documented_propagation(sharc, 25, "sharc")
    mid = _documented_propagation(sharc, 25, "ours")
    d_end = np.max(np.abs(np.abs(end[:, 0]) ** 2 - np.abs(sharc.c[:, 0]) ** 2))
    d_mid_ours = np.max(np.abs(np.abs(mid[1:, 0]) ** 2 - pop))
    print(
        f"  on SHARC's E, K, v: manual scheme vs SHARC max|d|c0|^2| {d_end:.2e}; "
        f"our scheme vs our core max|d|c0|^2| {d_mid_ours:.2e}"
    )
    assert d_end < 2e-6  # the manual's scheme reproduces SHARC (residual ~ dt^2, not ~ 1/N)
    assert d_mid_ours < 1e-8  # our scheme on SHARC's data reproduces our core
    assert abs(pop[-1] - s_pop[-1]) < 1e-5  # after the crossing the populations agree

    # The difference is first order in the substep: N = 25 -> 50 halves it (fast provider).
    peaks = {}
    for nsub in (25, 50):
        opt = DynamicsOptions(dt_au=0.5, nsubsteps=nsub, hopping="none", decoherence="none")
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
    print(f"  max|d|c0|^2| N=25 {peaks[25]:.3e}, N=50 {peaks[50]:.3e}, ratio {ratio:.3f}")
    assert 1.8 < ratio < 2.2
    assert peaks[25] < 1e-3


# --- 3. hopping on, SHARC's random numbers replayed in our core --------------------------------


def test_hopping_with_replayed_random_numbers(tmp_path):
    """SAC (x -2 -> 2, k = 40, dt = 1): with SHARC's random number of every step our core hops
    at the same steps to the same states, and conserves energy across the hop as SHARC does."""
    options = DynamicsOptions(dt_au=1.0, nsubsteps=25, hopping="sharc", decoherence="none")
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
    (manual eq. 8.25), ours kappa = 1 (method reference 7). Same random numbers, same hop."""
    options = DynamicsOptions(dt_au=1.0, nsubsteps=25, hopping="sharc", decoherence="edc")
    s = _sharc(
        TullySimpleAvoidedCrossing, -2.0, 0.02, 500, options, 9, tmp_path / "s", provider="fast"
    )
    ours = _ours(
        TullySimpleAvoidedCrossing, s, -2.0, 0.02, 500, options, list(s.random_numbers[1:])
    )
    steps = [k for k in range(len(s.steps) - 1) if s.q[k, 0] > 5.0][:20]
    assert len(steps) == 20
    k_sharc = _edc_kappa(s.c, s.active, s.energies, s.e_kinetic, 1.0, steps)
    c_ours = [s.c[0]] + [r.c for r in ours]
    a_ours = [s.active[0]] + [r.active for r in ours]
    e_ours = [s.energies[0]] + [r.energies for r in ours]
    k_ours_kin = [s.e_kinetic[0]] + [r.e_kinetic for r in ours]
    k_ours = _edc_kappa(c_ours, a_ours, e_ours, k_ours_kin, 1.0, steps)
    print(
        f"\n[EDC] kappa SHARC {k_sharc.min():.6f}..{k_sharc.max():.6f}, "
        f"ours {k_ours.min():.6f}..{k_ours.max():.6f}; SHARC hops {s.hop_lines}"
    )
    print(
        "  |c_inactive|^2 at steps 150/200/300: SHARC "
        + " ".join(f"{abs(s.c[k][1 - s.active[k]]) ** 2:.4f}" for k in (150, 200, 300))
        + ", ours "
        + " ".join(f"{abs(c_ours[k][1 - a_ours[k]]) ** 2:.4f}" for k in (150, 200, 300))
    )
    assert np.allclose(k_sharc, 0.5, atol=0.01)
    assert np.allclose(k_ours, 1.0, atol=0.01)


# --- 5. ensembles: branching ratios with the same initial conditions ---------------------------


@dataclass(frozen=True)
class Point:
    model: str
    k: float
    x0: float
    x_exit: float
    decoherence: str  # Task 8's asserted setting of this model
    n: int = 500

    @property
    def sigma(self) -> float:
        return 20.0 / self.k


POINTS = (Point("SAC", 20.0, -10.0, 10.0, "none"), Point("ECR", 20.0, -15.0, 12.0, "edc"))


def two_proportion_z(c1: int, n1: int, c2: int, n2: int) -> float:
    pooled = (c1 + c2) / (n1 + n2)
    var = pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2)
    return 0.0 if var == 0.0 else (c1 / n1 - c2 / n2) / math.sqrt(var)


def _sharc_ensemble(point: Point, options: DynamicsOptions, base: Path) -> dict[str, int]:
    cls = MODELS[point.model]
    mass = float(cls().masses[0])
    tag = zlib.crc32(b"model-initial")
    assert tag == ensemble._INITIAL_TAG  # the stream run_model_ensemble draws from
    rng = np.random.default_rng(np.random.SeedSequence([SEED, tag]))
    q0, v0 = wavepacket_initial_conditions(
        point.n, point.x0, point.k, point.sigma, mass, rng, "wigner"
    )
    seeds = np.random.SeedSequence([SEED, zlib.crc32(b"sharc-rngseed")]).generate_state(point.n)
    max_steps = 200_000

    def one(i: int) -> str:
        work = base / f"traj{i:04d}"
        traj = _sharc(
            cls,
            float(q0[i, 0]),
            float(v0[i, 0]),
            max_steps,
            options,
            int(seeds[i]) % 2_000_000_000 + 1,
            work,
            provider="fast",
            detailed=False,
            x_exit=point.x_exit,
            timeout_s=3600.0,
        )
        shutil.rmtree(work)
        if not traj.stopped:
            return "unfinished"
        side = "transmitted" if traj.final_x_bohr > 0 else "reflected"
        return f"{side}_{'lower' if traj.final_state == 0 else 'upper'}"

    with ThreadPoolExecutor(max_workers=SHARC_WORKERS) as pool:
        labels = list(pool.map(one, range(point.n)))
    return {name: labels.count(name) for name in (*BRANCHES, "unfinished")}


@pytest.mark.parametrize("point", POINTS, ids=lambda p: f"{p.model}-k{p.k:g}-{p.decoherence}")
def test_branching_matches_sharc(point, tmp_path):
    options = DynamicsOptions(
        dt_au=0.5,
        nsubsteps=25,
        hopping="sharc",
        ekin_correction="parallel_vel",
        frustrated="keep",
        decoherence=point.decoherence,
    )
    start = time.perf_counter()
    ours = run_model_ensemble(
        MODELS[point.model](),
        n_trajectories=point.n,
        x0=point.x0,
        k=point.k,
        sigma=point.sigma,
        initial_state=0,
        options=options,
        seed=SEED,
        x_exit=point.x_exit,
        max_steps=1_000_000,
        sampling="wigner",
        workers=OUR_WORKERS,
    )
    t_ours = time.perf_counter() - start
    start = time.perf_counter()
    sharc = _sharc_ensemble(point, options, tmp_path)
    t_sharc = time.perf_counter() - start
    n_ours, n_sharc = ours.n_finished, sum(sharc[b] for b in BRANCHES)
    print(
        f"\n[{point.model} k={point.k:g} {point.decoherence}] n={point.n}; ours {t_ours:.0f} s "
        f"(unfinished {ours.unfinished}, drift {ours.energy_drift}); SHARC {t_sharc:.0f} s "
        f"(unfinished {sharc['unfinished']})"
    )
    failures = []
    for branch in BRANCHES:
        c_ours, c_sharc = getattr(ours, branch), sharc[branch]
        z = two_proportion_z(c_ours, n_ours, c_sharc, n_sharc)
        print(
            f"  {branch:18s} ours {c_ours:4d} ({c_ours / n_ours:.4f})  SHARC {c_sharc:4d} "
            f"({c_sharc / n_sharc:.4f})  z {z:+.2f}"
        )
        if not abs(z) < 4.0:
            failures.append(f"{branch}: z = {z:+.2f}")
    assert ours.unfinished == 0 and ours.energy_drift == 0
    assert sharc["unfinished"] == 0
    assert failures == []
