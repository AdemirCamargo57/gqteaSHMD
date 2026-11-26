"""Per-estimator results (censored analysis spec, sections 2.2-2.4)."""

import numpy as np

from gqteashmd.analysis import estimators, grid, kinetics, populations
from gqteashmd.analysis.grid import GridData


def _grid(active, observed):
    active = np.asarray(active, dtype=np.int64)
    n, g = active.shape
    return GridData(
        np.arange(g),
        0.5 * np.arange(g),
        active,
        np.eye(3)[active],
        tuple(f"t{i}" for i in range(n)),
        g - 1,
        0.5 * (g - 1),
        (),
        np.asarray(observed, dtype=bool),
    )


def _decaying(rng, n=40, g=81):
    times = 10 + rng.exponential(10.0, size=n)
    active = np.where(0.5 * np.arange(g)[None, :] < times[:, None], 1, 0)
    return active


def test_complete_estimate_equals_the_old_stage_arithmetic():
    rng = np.random.default_rng(7)
    g = _grid(_decaying(rng), np.ones((40, 81), dtype=bool))
    est = estimators.complete_estimate(
        g,
        [1, 2],
        fit_population="classical",
        rng=np.random.default_rng(9),
        n_bootstrap=100,
        dt_out=0.5,
    )
    p, counts = populations.classical_populations(g.active, 3)
    np.testing.assert_array_equal(est.classical, p)
    idx = populations.bootstrap_indices(np.random.default_rng(9), 40, 100)
    per = np.isin(g.active, [1, 2]).astype(float)
    assert est.fit == kinetics.bootstrap_fit(g.time_fs, per, idx, dt_out=0.5)
    assert est.n == 40 and est.empty_rows is None


def test_censored_estimate_without_censoring_has_the_same_curve():
    rng = np.random.default_rng(7)
    g = _grid(_decaying(rng), np.ones((40, 81), dtype=bool))
    est = estimators.censored_estimate(
        g, [1, 2], rng=np.random.default_rng(10), n_bootstrap=100, dt_out=0.5
    )
    p, _ = populations.classical_populations(g.active, 3)
    assert np.abs(est.classical - p).max() <= 1e-12
    assert est.empty_rows == 0
    np.testing.assert_allclose(est.target, 1.0 - p[:, 0], atol=1e-12)
    assert est.fit.fit.tau_fs is not None and est.fit.tau_interval_fs is not None


def test_censored_quantum_is_the_at_risk_average():
    active = np.array([[1, 1, 0], [1, 0, 0]])
    observed = np.array([[True, True, True], [True, True, False]])
    est = estimators.censored_estimate(
        _grid(active, observed), [1, 2], rng=np.random.default_rng(1), n_bootstrap=100, dt_out=0.5
    )
    np.testing.assert_allclose(est.quantum[2], [1.0, 0.0, 0.0])  # only trajectory 0 at t_2


def _held_example():
    active = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]])
    observed = np.array([[True, True, True, False], [True, True, False, False], [True] * 4])
    pops = np.eye(2)[active].astype(float)
    return grid.GridData(
        np.arange(4),
        0.5 * np.arange(4),
        active,
        pops,
        ("a", "b", "c"),
        3,
        1.5,
        (),
        observed,
    )


def test_hold_in_s0_grid_puts_the_unobserved_steps_in_s0():
    held = estimators.hold_in_s0_grid(_held_example())
    np.testing.assert_array_equal(held.active, [[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]])
    assert held.observed.all()
    np.testing.assert_array_equal(held.populations[1, 3], [1.0, 0.0])


def test_hold_in_s0_estimate_is_the_complete_arithmetic_on_the_held_grid():
    g = _held_example()
    kw = {"fit_population": "classical", "n_bootstrap": 100, "dt_out": 0.5}
    hold = estimators.hold_in_s0_estimate(g, [1], rng=np.random.default_rng(3), **kw)
    direct = estimators.complete_estimate(
        estimators.hold_in_s0_grid(g), [1], rng=np.random.default_rng(3), **kw
    )
    assert hold.name == "hold_in_s0" and direct.name == "complete"
    np.testing.assert_array_equal(hold.classical, direct.classical)
    np.testing.assert_array_equal(hold.classical_lower, direct.classical_lower)
    assert hold.fit.fit.tau_fs == direct.fit.fit.tau_fs
    np.testing.assert_allclose(hold.classical[:, 0], [0.0, 1 / 3, 2 / 3, 2 / 3])
