"""Ensembles of 1D model trajectories: wavepacket initial conditions and branching statistics."""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

import numpy as np

from gqteashmd.dynamics.driver import Outcome, run_trajectory
from gqteashmd.dynamics.driver import initial_state as make_initial_state
from gqteashmd.dynamics.models import ElectronicModel
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.randomness import GeneratorSource
from gqteashmd.dynamics.records import TrajectoryState

SAMPLING_MODES = ("wigner", "fixed")
BRANCHES = ("transmitted_lower", "transmitted_upper", "reflected_lower", "reflected_upper")
Z_95 = 1.959963984540054
_TRAJECTORY_TAG = zlib.crc32(b"model-dynamics")
_INITIAL_TAG = zlib.crc32(b"model-initial")


def wavepacket_initial_conditions(
    n: int,
    x0: float,
    k: float,
    sigma: float,
    mass: float,
    rng: np.random.Generator,
    sampling: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Positions q (n, 1) and velocities v (n, 1) for psi(x) ~ exp(-(x - x0)^2/sigma^2 + ikx).

    |psi|^2 has standard deviation sigma/2 and the Wigner function has s_p = 1/sigma
    (hbar = 1). ``"wigner"`` draws x ~ N(x0, sigma/2), p ~ N(k, 1/sigma); ``"fixed"`` gives
    every trajectory x0 and k/mass.
    """
    if sampling not in SAMPLING_MODES:
        raise ValueError(f"sampling must be one of {', '.join(SAMPLING_MODES)}; got {sampling!r}")
    if sampling == "fixed":
        return np.full((n, 1), float(x0)), np.full((n, 1), k / mass)
    x = rng.normal(x0, sigma / 2.0, size=(n, 1))
    p = rng.normal(k, 1.0 / sigma, size=(n, 1))
    return x, p / mass


def wilson_interval(count: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for ``count`` successes in ``n`` trials."""
    if n <= 0:
        return 0.0, 1.0
    p = count / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class Branching:
    transmitted_lower: int = 0
    transmitted_upper: int = 0
    reflected_lower: int = 0
    reflected_upper: int = 0
    unfinished: int = 0
    energy_drift: int = 0
    n_trajectories: int = field(default=0)

    @property
    def n_finished(self) -> int:
        return sum(getattr(self, name) for name in BRANCHES)

    def counts(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in (*BRANCHES, "unfinished", "energy_drift")}

    def probabilities(self) -> dict[str, float]:
        """Fractions over the finished (stopped) trajectories; all zero if there are none."""
        n = self.n_finished
        return {name: (getattr(self, name) / n if n else 0.0) for name in BRANCHES}

    def intervals_95(self) -> dict[str, tuple[float, float]]:
        n = self.n_finished
        return {name: wilson_interval(getattr(self, name), n) for name in BRANCHES}


def run_model_ensemble(
    model: ElectronicModel,
    *,
    n_trajectories: int,
    x0: float,
    k: float,
    sigma: float,
    initial_state: int,
    options: DynamicsOptions,
    seed: int,
    x_exit: float,
    max_steps: int,
    sampling: str = "wigner",
) -> Branching:
    """Run ``n_trajectories`` trajectories until |x| > x_exit while moving outward."""
    mass = float(model.masses[0])
    init_rng = np.random.default_rng(np.random.SeedSequence([seed, _INITIAL_TAG]))
    q0, v0 = wavepacket_initial_conditions(n_trajectories, x0, k, sigma, mass, init_rng, sampling)
    children = np.random.SeedSequence([seed, _TRAJECTORY_TAG]).spawn(n_trajectories)

    def stop(state: TrajectoryState) -> bool:
        x, v = float(state.q[0]), float(state.v[0])
        return abs(x) > x_exit and v * math.copysign(1.0, x) > 0

    result = Branching(n_trajectories=n_trajectories)
    for i in range(n_trajectories):
        rng = GeneratorSource(np.random.default_rng(children[i]))
        state = make_initial_state(model, q0[i], v0[i], initial_state)
        final, outcome = run_trajectory(model, state, options, rng, max_steps=max_steps, stop=stop)
        if outcome is Outcome.ENERGY_DRIFT:
            result.energy_drift += 1
        elif outcome is Outcome.FINISHED:
            result.unfinished += 1
        else:
            side = "transmitted" if final.q[0] > 0 else "reflected"
            level = "lower" if final.active == 0 else "upper"
            name = f"{side}_{level}"
            setattr(result, name, getattr(result, name) + 1)
    return result
