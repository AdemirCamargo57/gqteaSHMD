"""Inclusion rule and common time grid (P8 spec sections 3.1 and 3.2)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

FINISHED = "finished"
STOPPED_IN_S0 = "stopped_in_s0"
INCLUDED_OUTCOMES = (FINISHED, STOPPED_IN_S0)


class InconsistentData(ValueError):
    """A trajectory's records do not fit the common grid (a file or program error)."""


@dataclass(frozen=True)
class Trajectory:
    task: str
    outcome: str
    initial_state: int
    grid_step: np.ndarray
    ticks: np.ndarray
    time_fs: np.ndarray
    active: np.ndarray
    hop_from: np.ndarray
    hop_to: np.ndarray
    frustrated: np.ndarray
    e_total: np.ndarray
    populations: np.ndarray

    @property
    def t_final_fs(self) -> float:
        return float(self.time_fs[-1])

    @property
    def last_full_step(self) -> int:
        full = self.grid_step[self.ticks == 0]
        return int(full.max()) if full.size else 0


@dataclass(frozen=True)
class Excluded:
    task: str
    reason: str
    t_final_fs: float | None
    energy_jump_eV: float | None = None


def select(
    tasks: Sequence[str],
    trajectories: Mapping[str, Trajectory | None],
    jumps_eV: Mapping[str, float],
    *,
    energy_jump_eV: float,
    exclude_energy_jumps: bool,
) -> tuple[list[Trajectory], list[Excluded]]:
    included: list[Trajectory] = []
    excluded: list[Excluded] = []
    for task in tasks:
        t = trajectories.get(task)
        if t is None:
            excluded.append(Excluded(task, "task_failed", None))
        elif t.outcome not in INCLUDED_OUTCOMES:
            excluded.append(Excluded(task, t.outcome, t.t_final_fs))
        elif exclude_energy_jumps and jumps_eV.get(task, 0.0) > energy_jump_eV:
            jump = float(jumps_eV[task])
            excluded.append(Excluded(task, "energy_jump", t.t_final_fs, jump))
        else:
            included.append(t)
    return included, excluded


def horizon_steps(included: Sequence[Trajectory], n_total: int) -> int:
    ends = [t.last_full_step for t in included if t.outcome == FINISHED]
    return int(min([n_total, *ends]))


@dataclass(frozen=True)
class GridData:
    grid_steps: np.ndarray
    time_fs: np.ndarray
    active: np.ndarray
    populations: np.ndarray
    tasks: tuple[str, ...]
    horizon_steps: int
    horizon_fs: float
    cut: tuple[str, ...]


def on_grid(
    included: Sequence[Trajectory],
    *,
    output_every: int,
    dt_fs: float,
    n_total: int,
    n_states: int,
) -> GridData:
    h = horizon_steps(included, n_total)
    steps = np.arange(0, h + 1, output_every, dtype=np.int64)
    active = np.zeros((len(included), steps.size), dtype=np.int64)
    populations = np.zeros((len(included), steps.size, n_states))
    ground = np.zeros(n_states)
    ground[0] = 1.0
    cut: list[str] = []
    for i, t in enumerate(included):
        full = np.flatnonzero(t.ticks == 0)
        where = {int(t.grid_step[j]): int(j) for j in full}
        end = t.last_full_step
        for k, s in enumerate(steps):
            j = where.get(int(s))
            if j is not None:
                active[i, k] = int(t.active[j])
                populations[i, k] = t.populations[j]
            elif t.outcome == STOPPED_IN_S0 and s > end:
                populations[i, k] = ground
            else:
                raise InconsistentData(f"{t.task}: no record at grid step {int(s)}")
        if np.any((t.grid_step > h) | ((t.grid_step == h) & (t.ticks > 0))):
            cut.append(t.task)
        if not np.all(np.isfinite(populations[i])):
            raise InconsistentData(f"{t.task}: non-finite populations")
    return GridData(
        steps,
        steps * float(dt_fs),
        active,
        populations,
        tuple(t.task for t in included),
        h,
        h * float(dt_fs),
        tuple(cut),
    )
