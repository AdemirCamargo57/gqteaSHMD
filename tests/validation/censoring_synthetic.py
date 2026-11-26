"""An in-memory project for tools/validation/censoring_report.py (plan ruling PR5).

Three states, start S2, decay S2 -> S1 -> S0 with back hops; the kinetic energy in S0 is a
per-trajectory level; every third trajectory that reaches S0 early is lost after its arrival.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gqteashmd.analysis.grid import Trajectory

TRANSITIONS = np.array([[0.93, 0.07, 0.0], [0.10, 0.90, 0.0], [0.0, 0.10, 0.90]])


def _trajectory(task: str, outcome: str, active: np.ndarray) -> Trajectory:
    n = active.size
    hop_from = np.full(n, -1)
    hop_to = np.full(n, -1)
    changed = np.flatnonzero(np.diff(active) != 0) + 1
    hop_from[changed] = active[changed - 1]
    hop_to[changed] = active[changed]
    return Trajectory(
        task,
        outcome,
        int(active[0]),
        np.arange(n, dtype=np.int64),
        np.zeros(n, dtype=np.int64),
        0.5 * np.arange(n),
        active.astype(np.int64),
        hop_from,
        hop_to,
        np.zeros(n, dtype=bool),
        np.zeros(n),
        np.eye(3)[active],
    )


def synthetic_project(seed: int = 11, n: int = 30, n_steps: int = 41) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    trajectories: dict[str, Trajectory | None] = {}
    kinetic: dict[str, np.ndarray] = {}
    for i in range(n):
        task = f"traj_{i:05d}"
        active = [2]
        for _ in range(n_steps - 1):
            active.append(int(rng.choice(3, p=TRANSITIONS[active[-1]])))
        a = np.asarray(active)
        heat = rng.uniform(0.005, 0.02)
        kin = np.where(a == 0, heat, 0.004) + rng.normal(0.0, 1e-4, n_steps)
        outcome, end = "finished", n_steps
        s0 = np.flatnonzero(a == 0)
        if i % 3 == 0 and s0.size and s0[0] < n_steps - 3:
            end = int(rng.integers(s0[0] + 1, n_steps - 1))
            outcome = "overlap_failure"
        trajectories[task] = _trajectory(task, outcome, a[:end])
        kinetic[task] = kin[:end]
    return {
        "trajectories": trajectories,
        "e_kinetic": kinetic,
        "dt_fs": 0.5,
        "output_every": 1,
        "n_total": n_steps - 1,
        "n_states": 3,
        "seed": 2028,
        "n_bootstrap": 50,
    }
