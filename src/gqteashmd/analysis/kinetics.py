"""Delayed mono-exponential fit with bootstrap uncertainties (P8 spec section 5)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def model_curve(t: np.ndarray, a: float, t0: float, tau: float) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore"):
        curve: np.ndarray = a * np.exp(-np.maximum(np.asarray(t) - t0, 0.0) / tau)
    return curve


@dataclass(frozen=True)
class FitResult:
    a: float
    t0_fs: float | None
    tau_fs: float | None
    rms: float | None
    success: bool
    flags: tuple[str, ...]


def fit_delayed_exponential(t: np.ndarray, y: np.ndarray, *, dt_out: float) -> FitResult:
    from scipy.optimize import least_squares

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    a = float(y[0])
    if a <= 0.0:
        return FitResult(a, None, None, None, True, ("no_population",))
    if float(y.min()) >= 0.99 * a:
        return FitResult(a, None, None, None, True, ("no_decay",))
    horizon = float(t[-1])
    below = np.flatnonzero(y < a / math.e)
    t_e = float(t[below[0]]) if below.size else horizon

    def residual(x: np.ndarray) -> np.ndarray:
        tau = float(np.exp(min(x[1], 700.0)))
        res: np.ndarray = model_curve(t, a, float(x[0]), tau) - y
        return res

    best = None
    for t0_start in (0.0, horizon / 4, horizon / 2):
        x0 = np.array([t0_start, math.log(max(t_e - t0_start, dt_out))])
        try:
            r = least_squares(residual, x0, bounds=([0.0, -np.inf], [horizon, np.inf]))
        except (ValueError, FloatingPointError):
            continue
        if not r.success or not np.all(np.isfinite(r.x)):
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return FitResult(a, None, None, None, False, ("fit_failed",))
    t0 = float(best.x[0])
    tau = float(np.exp(min(best.x[1], 700.0)))
    rms = float(np.sqrt(np.mean(residual(best.x) ** 2)))
    flags: list[str] = []
    if tau > horizon:
        flags.append("tau_beyond_horizon")
    if float(y[-1]) > 0.9 * a:
        flags.append("little_decay")
    if t0 > horizon - dt_out:
        flags.append("t0_at_upper_bound")
    return FitResult(a, t0, tau, rms, True, tuple(flags))


@dataclass(frozen=True)
class FitWithUncertainty:
    fit: FitResult
    t0_interval_fs: tuple[float, float] | None
    tau_interval_fs: tuple[float, float] | None
    t0_std_fs: float | None
    tau_std_fs: float | None
    bootstrap_failures: int


def bootstrap_fit(
    t: np.ndarray, per_trajectory: np.ndarray, indices: np.ndarray, *, dt_out: float
) -> FitWithUncertainty:
    fit = fit_delayed_exponential(t, per_trajectory.mean(axis=0), dt_out=dt_out)
    if fit.tau_fs is None:
        return FitWithUncertainty(fit, None, None, None, None, 0)
    taus: list[float] = []
    t0s: list[float] = []
    failures = 0
    for idx in indices:
        r = fit_delayed_exponential(t, per_trajectory[idx].mean(axis=0), dt_out=dt_out)
        if r.tau_fs is None or r.t0_fs is None:
            failures += 1
            continue
        taus.append(r.tau_fs)
        t0s.append(r.t0_fs)
    if len(taus) < 2:
        return FitWithUncertainty(fit, None, None, None, None, failures)
    t0_lo, t0_hi = (float(v) for v in np.percentile(t0s, [2.5, 97.5]))
    tau_lo, tau_hi = (float(v) for v in np.percentile(taus, [2.5, 97.5]))
    return FitWithUncertainty(
        fit,
        (t0_lo, t0_hi),
        (tau_lo, tau_hi),
        float(np.std(t0s, ddof=1)),
        float(np.std(taus, ddof=1)),
        failures,
    )
