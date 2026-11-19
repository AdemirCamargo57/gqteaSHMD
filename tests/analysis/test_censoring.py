"""Censored estimators (censored analysis spec, sections 2.2 and 4.1 items 1-2)."""

import numpy as np
import pytest

from gqteashmd.analysis import censoring, populations

T = np.array([[0.98, 0.02, 0.0], [0.04, 0.95, 0.01], [0.0, 0.05, 0.95]])  # rows: from S0, S1, S2
STEPS = 100
N = 100
R = 400
CENSOR = 0.02  # per step, in S0 only
POINTS = range(10, STEPS + 1, 10)  # plan ruling PR1
BOUND_SE = 4.0


def _truth():
    p = np.zeros((STEPS + 1, 3))
    p[0, 2] = 1.0
    for k in range(STEPS):
        p[k + 1] = p[k] @ T
    return p


def _simulate(rng, n=N, censor=CENSOR):
    cum = np.cumsum(T, axis=1)
    active = np.empty((n, STEPS + 1), dtype=np.int64)
    observed = np.ones((n, STEPS + 1), dtype=bool)
    active[:, 0] = 2
    alive = np.ones(n, dtype=bool)
    for k in range(STEPS):
        u = rng.random(n)
        active[:, k + 1] = (u[:, None] > cum[active[:, k]]).sum(axis=1)
        # a trajectory in S0 at step k is lost before step k + 1 with probability `censor`
        lost = alive & (active[:, k] == 0) & (rng.random(n) < censor)
        alive &= ~lost
        observed[:, k + 1] = alive
    return active, observed


def test_without_censoring_it_equals_the_classical_fractions():
    rng = np.random.default_rng(1)
    active, observed = _simulate(rng, censor=0.0)
    assert observed.all()
    aj = censoring.aalen_johansen(active, observed, 3)
    fractions, _ = populations.classical_populations(active, 3)
    assert np.abs(aj.populations - fractions).max() <= 1e-12
    assert aj.empty_rows == 0


def test_a_lost_trajectory_leaves_the_risk_set_of_its_interval():
    # two trajectories: one stays in S1, one goes S1 -> S0 and is lost after step 1
    active = np.array([[1, 1, 1], [1, 0, 0]])
    observed = np.array([[True, True, True], [True, True, False]])
    aj = censoring.aalen_johansen(active, observed, 2)
    np.testing.assert_allclose(aj.populations[1], [0.5, 0.5])
    # interval 1: only trajectory 0 at risk (S1 -> S1); S0 has nobody at risk but p = 0.5:
    # its identity row keeps it (one empty row, ruling PR2)
    np.testing.assert_allclose(aj.populations[2], [0.5, 0.5])
    assert aj.empty_rows == 1


def test_the_estimator_recovers_a_markov_truth_and_the_other_rules_do_not():
    rng = np.random.default_rng(20261119)
    truth = _truth()
    est = {"censored": [], "still_running": [], "complete_only": []}
    for _ in range(R):
        active, observed = _simulate(rng)
        est["censored"].append(censoring.aalen_johansen(active, observed, 3).populations)
        onehot = np.eye(3)[active]
        est["still_running"].append(censoring.at_risk_average(onehot, observed))
        never = observed[:, -1]
        est["complete_only"].append(populations.classical_populations(active[never], 3)[0])
    worst = {}
    for name, values in est.items():
        v = np.asarray(values)  # (R, G, S)
        mean = v.mean(axis=0)
        se = v.std(axis=0, ddof=1) / np.sqrt(R)
        z = [
            abs(mean[k, s] - truth[k, s]) / se[k, s] if se[k, s] > 0 else 0.0
            for k in POINTS
            for s in range(3)
        ]
        worst[name] = max(z)
    assert worst["censored"] <= BOUND_SE, worst
    assert worst["still_running"] > BOUND_SE, worst  # the test can fail
    assert worst["complete_only"] > BOUND_SE, worst


@pytest.mark.slow
def test_the_bootstrap_interval_covers_the_truth():
    rng = np.random.default_rng(20261120)
    truth = _truth()[50, 0]
    covered = 0
    for _ in range(R):
        active, observed = _simulate(rng)
        idx = populations.bootstrap_indices(rng, N, 500)
        boot = censoring.bootstrap_aalen_johansen(active, observed, 3, idx)
        lo, hi = np.percentile(boot[:, 50, 0], [2.5, 97.5])
        covered += int(lo <= truth <= hi)
    assert 0.90 <= covered / R <= 0.99, covered / R


def test_the_bootstrap_of_the_identity_resample_is_the_estimate():
    rng = np.random.default_rng(3)
    active, observed = _simulate(rng, n=20)
    idx = np.arange(20)[None, :]
    boot = censoring.bootstrap_aalen_johansen(active, observed, 3, idx)
    aj = censoring.aalen_johansen(active, observed, 3)
    assert np.abs(boot[0] - aj.populations).max() <= 1e-12


def test_at_risk_average_and_its_bootstrap():
    values = np.array([[[1.0], [1.0]], [[0.0], [5.0]]])  # (n=2, G=2, 1)
    observed = np.array([[True, True], [True, False]])
    np.testing.assert_allclose(censoring.at_risk_average(values, observed)[:, 0], [0.5, 1.0])
    idx = np.array([[0, 1], [1, 1]])
    boot = censoring.bootstrap_at_risk_average(values, observed, idx)
    np.testing.assert_allclose(boot[0, :, 0], [0.5, 1.0])
    assert boot[1, 0, 0] == 0.0 and np.isnan(boot[1, 1, 0])  # nobody at risk: NaN


def test_resample_weights_count_multiplicities():
    w = censoring.resample_weights(np.array([[0, 0, 2], [1, 2, 2]]), 3)
    np.testing.assert_array_equal(w, [[2, 0, 1], [0, 1, 2]])
