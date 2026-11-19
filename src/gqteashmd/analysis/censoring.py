"""Estimators for trajectories that end early (censored analysis spec, section 2.2).

A trajectory is *observed* at a grid time while it has data there; once its data ends it is
censored and leaves the risk set. The discrete-time Aalen-Johansen estimator multiplies the
interval transition matrices estimated from the trajectories at risk at both ends of each
interval. With no censoring it equals the plain fractions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
