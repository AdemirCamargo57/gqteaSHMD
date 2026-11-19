"""Estimators for trajectories that end early (censored analysis spec, section 2.2).

A trajectory is *observed* at a grid time while it has data there; once its data ends it is
censored and leaves the risk set. The discrete-time Aalen-Johansen estimator multiplies the
interval transition matrices estimated from the trajectories at risk at both ends of each
interval. With no censoring it equals the plain fractions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from gqteashmd.analysis.energy import first_jump_index
from gqteashmd.analysis.grid import INCLUDED_OUTCOMES, Excluded, Trajectory


@dataclass(frozen=True)
class AalenJohansen:
    populations: np.ndarray  # (G, S)
    empty_rows: int  # (interval, state) pairs with nobody at risk but probability > 0 (PR2)


def transition_indicators(active: np.ndarray, observed: np.ndarray, n_states: int) -> np.ndarray:
    """(n, G-1, S, S): 1 where trajectory i is at risk over [t_k, t_k+1] and goes a -> b."""
    active = np.asarray(active, dtype=np.int64)
    observed = np.asarray(observed, dtype=bool)
    n, g = active.shape
    x = np.zeros((n, max(g - 1, 0), n_states, n_states))
    if g < 2:
        return x
    risk = observed[:, :-1] & observed[:, 1:]
    i, k = np.nonzero(risk)
    x[i, k, active[i, k], active[i, k + 1]] = 1.0
    return x


def _product(initial: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """initial (B, S), counts (B, G-1, S, S) -> populations (B, G, S), empty rows (B,)."""
    b, steps, s, _ = counts.shape
    p = np.empty((b, steps + 1, s))
    p[:, 0] = initial
    empty = np.zeros(b, dtype=np.int64)
    eye = np.eye(s)
    for k in range(steps):
        n_from = counts[:, k].sum(axis=-1)  # (B, S): at risk in each state at t_k
        missing = n_from == 0
        with np.errstate(invalid="ignore", divide="ignore"):
            rows = counts[:, k] / n_from[..., None]
        rows = np.where(missing[..., None], eye, rows)
        empty += np.sum(missing & (p[:, k] > 0.0), axis=-1)
        p[:, k + 1] = np.einsum("ba,bac->bc", p[:, k], rows)
    return p, empty


def aalen_johansen(active: np.ndarray, observed: np.ndarray, n_states: int) -> AalenJohansen:
    active = np.asarray(active, dtype=np.int64)
    n = active.shape[0]
    if n == 0:
        raise ValueError("aalen_johansen needs at least one trajectory")
    initial = np.bincount(active[:, 0], minlength=n_states)[None, :] / n
    counts = transition_indicators(active, observed, n_states).sum(axis=0)[None]
    p, empty = _product(initial, counts)
    return AalenJohansen(p[0], int(empty[0]))


def resample_weights(indices: np.ndarray, n: int) -> np.ndarray:
    """(B, n): how often each trajectory appears in each bootstrap resample."""
    return np.stack([np.bincount(np.asarray(idx), minlength=n) for idx in indices]).astype(float)


def bootstrap_aalen_johansen(
    active: np.ndarray, observed: np.ndarray, n_states: int, indices: np.ndarray
) -> np.ndarray:
    """(B, G, S): the estimator on every resample (rows of `indices`), vectorised."""
    active = np.asarray(active, dtype=np.int64)
    n = active.shape[0]
    w = resample_weights(indices, n)
    x = transition_indicators(active, observed, n_states)
    counts = np.tensordot(w, x, axes=(1, 0))  # (B, G-1, S, S)
    initial = w @ np.eye(n_states)[active[:, 0]] / n
    p, _ = _product(initial, counts)
    return p


def at_risk_average(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """(G, ...): the mean of `values` (n, G, ...) over the trajectories observed at each time."""
    w = np.asarray(observed, dtype=float)
    v = np.asarray(values, dtype=float)
    num = np.einsum("ng,ng...->g...", w, v)
    den = w.sum(axis=0).reshape((-1,) + (1,) * (v.ndim - 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        result: np.ndarray = num / den
    return result


def bootstrap_at_risk_average(
    values: np.ndarray, observed: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """(B, G, ...): `at_risk_average` on every resample."""
    v = np.asarray(values, dtype=float)
    w = resample_weights(indices, v.shape[0])[:, :, None] * np.asarray(observed, dtype=float)[None]
    num = np.einsum("bng,ng...->bg...", w, v)
    den = w.sum(axis=1).reshape(w.shape[0], -1, *((1,) * (v.ndim - 2)))
    with np.errstate(invalid="ignore", divide="ignore"):
        result: np.ndarray = num / den
    return result


@dataclass(frozen=True)
class Censored:
    task: str
    outcome: str
    reason: str  # "outcome" (ended early) or "energy_jump" (exclude_energy_jumps = true)
    grid_step: int  # the last full-step grid point used
    time_fs: float


def censoring_step(t: Trajectory, *, output_every: int, before_record: int | None = None) -> int:
    """The last grid step (a multiple of output_every) with a full-step record, among the
    records before index `before_record` (all records if None); 0 if there is none."""
    stop = t.grid_step.size if before_record is None else int(before_record)
    steps = t.grid_step[:stop]
    full = steps[(t.ticks[:stop] == 0) & (steps % output_every == 0)]
    return int(full.max()) if full.size else 0


def select_censored(
    tasks: Sequence[str],
    trajectories: Mapping[str, Trajectory | None],
    *,
    output_every: int,
    n_total: int,
    dt_fs: float,
    energy_jump_eV: float | None,
) -> tuple[list[Trajectory], dict[str, int], list[Censored], list[Excluded]]:
    """Spec section 2.1: complete and censored trajectories are used, the others excluded.

    `energy_jump_eV` is the threshold when exclude_energy_jumps = true, else None: a
    trajectory is then censored before its first jump over grid steps <= n_total (P8 PR3).
    """
    used: list[Trajectory] = []
    at: dict[str, int] = {}
    censored: list[Censored] = []
    excluded: list[Excluded] = []
    for task in tasks:
        t = trajectories.get(task)
        if t is None:
            excluded.append(Excluded(task, "task_failed", None))
            continue
        if t.time_fs.size == 0:
            excluded.append(Excluded(task, t.outcome, t.t_final_fs))
            continue
        jump_at = None
        if energy_jump_eV is not None:
            window = int(np.count_nonzero(t.grid_step <= n_total))  # records are time-ordered
            jump_at = first_jump_index(t.e_total[:window], energy_jump_eV)
        if jump_at is not None:
            reason = "energy_jump"
            step = censoring_step(t, output_every=output_every, before_record=jump_at)
        elif t.outcome in INCLUDED_OUTCOMES:
            used.append(t)
            continue
        else:
            reason = "outcome"
            step = censoring_step(t, output_every=output_every)
        used.append(t)
        at[task] = step
        censored.append(Censored(task, t.outcome, reason, step, step * float(dt_fs)))
    return used, at, censored, excluded
