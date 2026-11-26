"""IPCW estimator of state occupation (spec 2026-11-26-ipcw-trial-analysis-v4-design.md, 2).

Report tool, never imported by gqteashmd. The censoring hazard is a discrete-time logistic
model, logit h_i(k) = alpha_{a,k} + beta z_i(k) 1{a = 0}, with one intercept per grid step and
state (nonparametric baseline) and z the standardised running mean of the S0 kinetic energy.
The populations are (1/n) sum_i w_i(k) 1{active_i(k) = s}, w_i the inverse probability of
remaining uncensored (not normalised).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from gqteashmd.analysis.grid import Trajectory
from gqteashmd.core.units import HARTREE_IN_EV

BETA_BOUND = 10.0
BOUND_TOLERANCE = 1e-3


@dataclass(frozen=True)
class CensoringFit:
    status: str  # "ok" or "fit_failed"
    reason: str | None
    beta: float | None  # per SD of the covariate; 0.0 without it
    sd_eV: float | None  # SD of the running S0 kinetic energy over the S0 cells at risk
    hazard: np.ndarray | None  # (n, G-1) fitted per-interval censoring probability
    n_events_s0: int
    n_informative: int  # S0 cells with 0 < d < r

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def beta_per_eV(self) -> float | None:
        if self.beta is None or not self.sd_eV:
            return None
        return self.beta / self.sd_eV

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "beta_per_sd": self.beta,
            "sd_eV": self.sd_eV,
            "beta_per_eV": self.beta_per_eV,
            "n_events_s0": self.n_events_s0,
            "n_informative": self.n_informative,
        }


def _failed(reason: str, n_events_s0: int, n_informative: int = 0) -> CensoringFit:
    return CensoringFit("fit_failed", reason, None, None, None, n_events_s0, n_informative)


def kinetic_on_grid(
    trajectories: Sequence[Trajectory],
    kinetic: Mapping[str, np.ndarray],
    grid_steps: np.ndarray,
) -> np.ndarray:
    """(n, G): e_kinetic at each trajectory's full-step record of every grid step (NaN if none)."""
    out = np.full((len(trajectories), len(grid_steps)), np.nan)
    for i, t in enumerate(trajectories):
        values = np.asarray(kinetic[t.task], dtype=float)
        where = {int(t.grid_step[j]): int(j) for j in np.flatnonzero(t.ticks == 0)}
        for m, s in enumerate(grid_steps):
            j = where.get(int(s))
            if j is not None:
                out[i, m] = values[j]
    return out


def running_s0_mean(active: np.ndarray, kinetic: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """(n, G): the mean kinetic energy over the observed S0 records up to each grid step."""
    in_s0 = (np.asarray(active) == 0) & np.asarray(observed, dtype=bool)
    total = np.cumsum(np.where(in_s0, np.asarray(kinetic, dtype=float), 0.0), axis=1)
    count = np.cumsum(in_s0, axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def risk_and_events(observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, G-1) each: at risk over [k, k+1] (observed at k); censored there (not at k+1)."""
    obs = np.asarray(observed, dtype=bool)
    risk = obs[:, :-1]
    return risk, risk & ~obs[:, 1:]


def _event_cells(
    active: np.ndarray, risk: np.ndarray, event: np.ndarray, n_states: int
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """(k, state, members, censored) of every cell with at least one event."""
    a = np.asarray(active)[:, :-1]
    cells = []
    for k in range(risk.shape[1]):
        for s in range(n_states):
            members = np.flatnonzero(risk[:, k] & (a[:, k] == s))
            if members.size and event[members, k].any():
                cells.append((k, s, members, event[members, k]))
    return cells


def _hazard(
    beta: float,
    z: np.ndarray | None,
    cells: Sequence[tuple[int, int, np.ndarray, np.ndarray]],
    shape: tuple[int, int],
) -> tuple[np.ndarray, float]:
    """The profile hazard for one beta and its log-likelihood (cells without events: h = 0)."""
    from scipy.optimize import brentq
    from scipy.special import expit

    h = np.zeros(shape)
    loglik = 0.0
    for k, s, members, censored in cells:
        r, d = members.size, int(censored.sum())
        if d == r:
            h[members, k] = 1.0
            continue
        if s != 0 or beta == 0.0 or z is None:
            hc = np.full(r, d / r)
        else:
            x = beta * z[members, k]
            base = math.log(d / (r - d))
            spread = float(np.max(np.abs(x))) + 1.0  # plan ruling PR7
            alpha = brentq(
                lambda al, x=x, d=d: float(expit(al + x).sum()) - d,
                base - spread,
                base + spread,
                xtol=1e-14,
            )
            hc = expit(alpha + x)
        h[members, k] = hc
        with np.errstate(divide="ignore"):  # expit can round to 1.0 near the beta bounds
            loglik += float(np.log(hc[censored]).sum() + np.log1p(-hc[~censored]).sum())
    return h, loglik


def fit_censoring_hazard(
    active: np.ndarray,
    observed: np.ndarray,
    kinetic: np.ndarray,
    n_states: int,
    *,
    covariate: bool = True,
) -> CensoringFit:
    from scipy.optimize import minimize_scalar

    active = np.asarray(active, dtype=np.int64)
    risk, event = risk_and_events(observed)
    in_s0 = risk & (active[:, :-1] == 0)
    n_events_s0 = int((event & in_s0).sum())
    cells = _event_cells(active, risk, event, n_states)
    informative = sum(1 for _, s, m, c in cells if s == 0 and 0 < int(c.sum()) < m.size)
    if not covariate:
        h, _ = _hazard(0.0, None, cells, risk.shape)
        return CensoringFit("ok", None, 0.0, None, h, n_events_s0, informative)
    if n_events_s0 == 0:
        return _failed("no censoring event in S0", n_events_s0)
    if informative == 0:
        return _failed("no informative S0 interval", n_events_s0)  # plan ruling PR3
    kbar = running_s0_mean(active, kinetic, observed)[:, :-1]
    values = kbar[in_s0]
    sd = float(values.std()) if values.size else 0.0
    if not np.all(np.isfinite(values)) or sd <= 0.0:
        return _failed("the covariate has no spread", n_events_s0, informative)
    z = np.where(in_s0, (kbar - float(values.mean())) / sd, 0.0)
    result = minimize_scalar(
        lambda b: -_hazard(float(b), z, cells, risk.shape)[1],
        bounds=(-BETA_BOUND, BETA_BOUND),
        method="bounded",
        options={"xatol": 1e-8},
    )
    beta = float(result.x)
    if not result.success or abs(beta) > BETA_BOUND - BOUND_TOLERANCE:
        return _failed("beta on a bound", n_events_s0, informative)
    h, _ = _hazard(beta, z, cells, risk.shape)
    return CensoringFit("ok", None, beta, sd * HARTREE_IN_EV, h, n_events_s0, informative)


def censoring_weights(hazard: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """(n, G): w(0) = 1, w(k+1) = w(k) / (1 - h(k)); 0 where a trajectory is not observed."""
    obs = np.asarray(observed, dtype=bool)
    w = np.ones(obs.shape)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        w[:, 1:] = np.cumprod(1.0 / (1.0 - np.asarray(hazard, dtype=float)), axis=1)
    return np.where(obs, w, 0.0)


def ipcw_populations(active: np.ndarray, weights: np.ndarray, n_states: int) -> np.ndarray:
    """(G, S): (1/n) sum_i w_i(k) 1{active_i(k) = s}."""
    a = np.asarray(active, dtype=np.int64)
    onehot = np.eye(n_states)[a]
    return np.einsum("ng,ngs->gs", np.asarray(weights, dtype=float), onehot) / a.shape[0]


def ipcw_estimate(
    active: np.ndarray,
    observed: np.ndarray,
    kinetic: np.ndarray,
    n_states: int,
    *,
    covariate: bool = True,
) -> tuple[np.ndarray | None, CensoringFit]:
    fit = fit_censoring_hazard(active, observed, kinetic, n_states, covariate=covariate)
    if not fit.ok or fit.hazard is None:
        return None, fit
    weights = censoring_weights(fit.hazard, observed)
    return ipcw_populations(active, weights, n_states), fit
