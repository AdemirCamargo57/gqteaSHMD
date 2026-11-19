"""The populations, intervals and fit of each estimator (censored analysis spec, section 2).

`complete_estimate` is the P8 arithmetic unchanged (the stage's numbers before this
sub-project); `censored_estimate` is the Aalen-Johansen curve with bootstrap intervals and the
fit to its excited-state sum. The caller passes the random generator (seed rule D12).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from gqteashmd.analysis import censoring, kinetics, populations
from gqteashmd.analysis.grid import GridData


@dataclass(frozen=True)
class Estimate:
    name: str
    n: int
    classical: np.ndarray
    classical_lower: np.ndarray
    classical_upper: np.ndarray
    quantum: np.ndarray
    quantum_lower: np.ndarray
    quantum_upper: np.ndarray
    target: np.ndarray
    fit: kinetics.FitWithUncertainty
    empty_rows: int | None


def _percentiles(boot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2.5/97.5 % over the resamples; NaN where a resample is not finite (nobody at risk)."""
    lo = np.full(boot.shape[1:], np.nan)
    hi = np.full(boot.shape[1:], np.nan)
    ok = np.all(np.isfinite(boot), axis=0)
    if np.any(ok):
        lo[ok], hi[ok] = np.percentile(boot[:, ok], [2.5, 97.5], axis=0)
    return lo, hi


def complete_estimate(
    g: GridData,
    fit_index: Sequence[int],
    *,
    fit_population: str,
    rng: np.random.Generator,
    n_bootstrap: int,
    dt_out: float,
) -> Estimate:
    n = int(g.active.shape[0])
    n_states = int(g.populations.shape[-1])
    p_classical, n_counts = populations.classical_populations(g.active, n_states)
    p_lo, p_hi = populations.wilson_intervals(n_counts, n)
    rho = populations.quantum_populations(g.populations)
    idx = populations.bootstrap_indices(rng, n, n_bootstrap)
    rho_lo, rho_hi = populations.bootstrap_intervals(g.populations, idx)
    if fit_population == "classical":
        per = np.isin(g.active, list(fit_index)).astype(float)
    else:
        per = g.populations[:, :, list(fit_index)].sum(axis=-1)
    fit = kinetics.bootstrap_fit(g.time_fs, per, idx, dt_out=dt_out)
    return Estimate(
        "complete", n, p_classical, p_lo, p_hi, rho, rho_lo, rho_hi, per.mean(axis=0), fit, None
    )


def censored_estimate(
    g: GridData,
    fit_index: Sequence[int],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    dt_out: float,
) -> Estimate:
    n = int(g.active.shape[0])
    n_states = int(g.populations.shape[-1])
    aj = censoring.aalen_johansen(g.active, g.observed, n_states)
    idx = populations.bootstrap_indices(rng, n, n_bootstrap)
    boot = censoring.bootstrap_aalen_johansen(g.active, g.observed, n_states, idx)
    p_lo, p_hi = _percentiles(boot)
    rho = censoring.at_risk_average(g.populations, g.observed)
    rho_lo, rho_hi = _percentiles(
        censoring.bootstrap_at_risk_average(g.populations, g.observed, idx)
    )
    columns = list(fit_index)
    target = aj.populations[:, columns].sum(axis=1)
    fit = kinetics.bootstrap_fit_curves(
        g.time_fs, target, boot[:, :, columns].sum(axis=-1), n_trajectories=n, dt_out=dt_out
    )
    return Estimate(
        "censored", n, aj.populations, p_lo, p_hi, rho, rho_lo, rho_hi, target, fit, aj.empty_rows
    )
