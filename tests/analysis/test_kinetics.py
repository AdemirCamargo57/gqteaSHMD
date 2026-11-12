import math
import warnings

import numpy as np
import pytest

from gqteashmd.analysis.kinetics import (
    bootstrap_fit,
    fit_delayed_exponential,
    model_curve,
    tau_from_log,
)
from gqteashmd.analysis.populations import bootstrap_indices

T = np.arange(0.0, 150.0 + 1e-9, 0.5)


def test_noise_free_model_is_recovered():
    y = model_curve(T, 1.0, 12.0, 35.0)
    r = fit_delayed_exponential(T, y, dt_out=0.5)
    assert r.success and r.flags == ()
    assert r.t0_fs == pytest.approx(12.0, rel=1e-6) and r.tau_fs == pytest.approx(35.0, rel=1e-6)
    assert r.rms < 1e-8


def test_zero_delay_is_not_flagged():
    r = fit_delayed_exponential(T, model_curve(T, 1.0, 0.0, 30.0), dt_out=0.5)
    assert r.t0_fs == pytest.approx(0.0, abs=1e-6) and "t0_at_upper_bound" not in r.flags


def test_flags():
    assert fit_delayed_exponential(T, np.ones_like(T), dt_out=0.5).flags == ("no_decay",)
    assert fit_delayed_exponential(T, np.zeros_like(T), dt_out=0.5).flags == ("no_population",)
    slow = fit_delayed_exponential(T, model_curve(T, 1.0, 0.0, 5000.0), dt_out=0.5)  # y(T) 0.97
    assert "tau_beyond_horizon" in slow.flags and "little_decay" in slow.flags
    late = fit_delayed_exponential(T, model_curve(T, 1.0, 149.8, 0.2), dt_out=0.5)
    assert "t0_at_upper_bound" in late.flags  # only the last point decays; if the optimizer
    # cannot place t0 there from the fixed starts, report it (do not loosen the rule)


def test_synthetic_ensemble_recovers_tau_and_t0_within_the_bootstrap_error():
    rng = np.random.default_rng(20261111)
    times = 10.0 + rng.exponential(40.0, size=2000)
    per_trajectory = (T[None, :] < times[:, None]).astype(float)
    result = bootstrap_fit(T, per_trajectory, bootstrap_indices(rng, 2000, 200), dt_out=0.5)
    fit = result.fit
    assert fit.success and result.tau_std_fs is not None and result.t0_std_fs is not None
    assert result.tau_std_fs < 3.0
    assert abs(fit.tau_fs - 40.0) <= 4 * result.tau_std_fs
    assert abs(fit.t0_fs - 10.0) <= 4 * result.t0_std_fs + 0.5
    lo, hi = result.tau_interval_fs
    assert lo < fit.tau_fs < hi and result.bootstrap_failures == 0


def test_one_trajectory_has_no_bootstrap_uncertainty():
    per = (T[None, :] < 20.0).astype(float)  # one trajectory, a step at 20 fs
    result = bootstrap_fit(T, per, bootstrap_indices(np.random.default_rng(1), 1, 50), dt_out=0.5)
    assert result.fit.tau_fs is not None
    assert result.t0_interval_fs is None and result.tau_interval_fs is None
    assert result.t0_std_fs is None and result.tau_std_fs is None
    assert result.bootstrap_failures == 0


def test_an_extreme_log_tau_does_not_underflow_to_zero():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tau = tau_from_log(-1000.0)
        assert tau > 0.0
        curve = model_curve(T, 1.0, 10.0, tau)
        assert np.all(np.isfinite(curve)) and curve[0] == 1.0 and curve[-1] == 0.0
        assert tau_from_log(1000.0) == tau_from_log(700.0) < math.inf


def test_no_fit_means_no_bootstrap():
    per = np.ones((10, T.size))
    result = bootstrap_fit(T, per, bootstrap_indices(np.random.default_rng(1), 10, 100), dt_out=0.5)
    assert result.fit.flags == ("no_decay",) and result.tau_interval_fs is None
    assert result.bootstrap_failures == 0
