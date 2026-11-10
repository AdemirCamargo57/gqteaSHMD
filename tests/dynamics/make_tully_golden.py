"""Record the golden trajectories that pin the P7a-1 results (run once, on e76c83a).

Usage: python tests/dynamics/make_tully_golden.py [--restart-fixture]

``tully_golden.npz`` holds, for three trajectories, every step's q, v, c, active state,
hop probabilities, random number, hop_from/hop_to (-1 for none) and frustrated flag.
``test_golden.py`` reruns them with ``np.array_equal``; they must stay bit-identical for every
model without off-diagonal MCH Hamiltonian elements.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from gqteashmd.dynamics.driver import initial_state, run_trajectory
from gqteashmd.dynamics.models import (
    TullyDualAvoidedCrossing,
    TullyExtendedCoupling,
    TullySimpleAvoidedCrossing,
)
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.randomness import GeneratorSource, RandomSource, ReplaySource
from gqteashmd.dynamics.restart import save_state

MASS = 2000.0
DATA = Path(__file__).parent / "data"

# ECR starts in a superposition (active state 0, amplitudes 0.8 and 0.6i). At k = 10 the hop
# probability to the upper state is ~1e-5 per step, and 400 seeds with plain draws never
# produced a frustrated attempt. So the ECR stream is the seed-13 generator's first 600
# numbers with every 25th number replaced by 1e-300 (a forced attempt wherever p > 0): the
# attempts at x > -2.2 bohr (gap > E_kin = 0.025 Eh) are frustrated and reflected.
INITIAL_AMPLITUDES = {"ecr": np.array([0.8, 0.6j])}
FORCED_EVERY = {"ecr": 25}


def make_source(name: str, steps: int, seed: int) -> RandomSource:
    generator = np.random.default_rng(seed)
    if name not in FORCED_EVERY:
        return GeneratorSource(generator)
    values = generator.random(steps)
    values[:: FORCED_EVERY[name]] = 1e-300
    return ReplaySource(values)


# name -> (model factory, q0, k, active, options, steps, seed)
CASES = {
    "sac": (
        TullySimpleAvoidedCrossing,
        -2.0,  # from x = -5 the packet does not reach the coupling region in 400 steps
        20.0,
        0,
        dict(dt_au=1.0, hopping="sharc", edc_convention="sharc", ekin_correction="parallel_vel"),
        400,
        11,
    ),
    "dac": (
        TullyDualAvoidedCrossing,
        -3.0,
        25.0,
        0,
        dict(dt_au=1.0, hopping="fssh", decoherence="none", substep_interpolation="midpoint"),
        300,
        12,
    ),
    "ecr": (
        TullyExtendedCoupling,
        -3.0,
        10.0,
        0,
        dict(
            dt_au=1.0,
            ekin_correction="parallel_nac",
            frustrated="reflect",
            edc_convention="granucci_persico",
            substep_interpolation="endpoint",
        ),
        600,
        13,
    ),
}


def run_case(name: str) -> dict[str, np.ndarray]:
    factory, x0, k, active, opts, steps, seed = CASES[name]
    model = factory()
    options = DynamicsOptions(**opts)
    c0 = INITIAL_AMPLITUDES.get(name)
    state = initial_state(model, np.array([x0]), np.array([k / MASS]), active, c0)
    records = []
    run_trajectory(
        model,
        state,
        options,
        make_source(name, steps, seed),
        max_steps=steps,
        on_step=records.append,
    )
    return {
        "q": np.array([r.q for r in records]),
        "v": np.array([r.v for r in records]),
        "c": np.array([r.c for r in records]),
        "active": np.array([r.active for r in records]),
        "probabilities": np.array([r.probabilities for r in records]),
        "random_number": np.array([r.random_number for r in records]),
        "hop_from": np.array([-1 if r.hop_from is None else r.hop_from for r in records]),
        "hop_to": np.array([-1 if r.hop_to is None else r.hop_to for r in records]),
        "frustrated": np.array([r.frustrated for r in records]),
    }


def all_cases() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in CASES:
        for key, value in run_case(name).items():
            out[f"{name}_{key}"] = value
    return out


def main() -> None:
    data = all_cases()
    assert data["sac_hop_to"].max() >= 0, "SAC: no hop"
    assert (data["dac_hop_to"] >= 0).any(), "DAC: no hop"
    assert data["ecr_frustrated"].any(), "ECR: no frustrated reflection"
    for name in CASES:
        n_hops = int((data[f"{name}_hop_to"] >= 0).sum())
        n_frus = int(data[f"{name}_frustrated"].sum())
        print(f"{name}: hops={n_hops} frustrated={n_frus}")
    DATA.mkdir(exist_ok=True)
    np.savez(DATA / "tully_golden.npz", **data)  # type: ignore[arg-type]
    if "--restart-fixture" in sys.argv:
        model = TullySimpleAvoidedCrossing()
        options = DynamicsOptions(dt_au=1.0)
        state = initial_state(model, np.array([-5.0]), np.array([20.0 / MASS]), 0)
        rng = GeneratorSource(np.random.default_rng(3))
        state, _ = run_trajectory(model, state, options, rng, max_steps=40)
        save_state(DATA / "restart_v1.npz", state, rng)
        print("wrote restart_v1.npz at step", state.step)


if __name__ == "__main__":
    main()
