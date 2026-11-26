"""tools/validation/ipcw.py (spec 2026-11-26-ipcw-trial-analysis-v4-design.md, 4.1 items 1-7)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "validation"))
import ipcw  # noqa: E402

from gqteashmd.analysis import censoring  # noqa: E402
from gqteashmd.analysis.grid import Trajectory  # noqa: E402
from gqteashmd.core.units import HARTREE_IN_EV  # noqa: E402

T_MARKOV = np.array([[0.97, 0.02, 0.01], [0.05, 0.90, 0.05], [0.0, 0.08, 0.92]])


def markov_sample(rng, n=300, g=60, hazard=(0.06, 0.01, 0.01)):
    """Spec 1.1: start S2, censoring hazard by the current state."""
    act = np.zeros((n, g), dtype=np.int64)
    act[:, 0] = 2
    for k in range(1, g):
        cum = np.cumsum(T_MARKOV[act[:, k - 1]], axis=1)
        act[:, k] = (rng.random(n)[:, None] > cum).sum(axis=1)
    obs = np.ones((n, g), dtype=bool)
    hz = np.asarray(hazard)
    for k in range(g - 1):
        cut = obs[:, k] & (rng.random(n) < hz[act[:, k]])
        obs[cut, k + 1 :] = False
    return act, obs


def per_cell_hazard(act, obs, n_states, *, by_state=True):
    risk, event = ipcw.risk_and_events(obs)
    h = np.zeros(risk.shape)
    for k in range(risk.shape[1]):
        groups = range(n_states) if by_state else [None]
        for s in groups:
            m = risk[:, k] & ((act[:, k] == s) if s is not None else True)
            if m.any():
                h[m, k] = event[m, k].sum() / m.sum()
    return h


def test_marginal_weights_give_the_still_running_average():  # item 1
    act, obs = markov_sample(np.random.default_rng(1))
    w = ipcw.censoring_weights(per_cell_hazard(act, obs, 3, by_state=False), obs)
    np.testing.assert_allclose(
        ipcw.ipcw_populations(act, w, 3),
        censoring.at_risk_average(np.eye(3)[act], obs),
        rtol=0,
        atol=1e-12,
    )


def test_without_the_covariate_the_hazard_is_d_over_r_per_cell():  # item 2
    act, obs = markov_sample(np.random.default_rng(2), n=200, g=40)
    fit = ipcw.fit_censoring_hazard(act, obs, np.zeros(act.shape), 3, covariate=False)
    assert fit.ok and fit.beta == 0.0
    expected = per_cell_hazard(act, obs, 3)
    np.testing.assert_allclose(fit.hazard, expected, rtol=0, atol=1e-12)
    w = np.ones(act.shape)
    with np.errstate(divide="ignore", invalid="ignore"):  # h = 1 where every one is censored
        for k in range(act.shape[1] - 1):
            w[:, k + 1] = w[:, k] / (1.0 - expected[:, k])
    w[~obs] = 0.0
    hand = np.stack([((act == s) * w).sum(axis=0) for s in range(3)], axis=-1) / act.shape[0]
    curve, _ = ipcw.ipcw_estimate(act, obs, np.zeros(act.shape), 3, covariate=False)
    np.testing.assert_allclose(curve, hand, rtol=0, atol=1e-12)


def test_ipcw_is_unbiased_under_state_dependent_censoring():  # item 3, the PR1 form
    rng = np.random.default_rng(20261126)
    g = 61  # grid points 0..60, so the spec's points 10, 20, ..., 60 exist
    truth = np.zeros((g, 3))
    truth[0, 2] = 1.0
    for k in range(1, g):
        truth[k] = truth[k - 1] @ T_MARKOV
    curves = []
    for _ in range(100):
        act, obs = markov_sample(rng, g=g)
        curve, fit = ipcw.ipcw_estimate(act, obs, np.zeros(act.shape), 3, covariate=False)
        assert fit.ok
        curves.append(curve)
    c = np.asarray(curves)
    se = c.std(axis=0, ddof=1) / np.sqrt(len(curves))
    for k in (10, 20, 30, 40, 50, 60):
        for s in range(3):
            if se[k, s] > 0:
                assert abs(c[:, k, s].mean() - truth[k, s]) <= 4 * se[k, s], (k, s)


def hot_censoring_sample(rng, n=400, g=40, a=-3.0, beta=1.0):
    """Everyone in S0; a constant kinetic level per trajectory; logit h = a + beta z."""
    act = np.zeros((n, g), dtype=np.int64)
    level = rng.normal(0.01, 0.002, n)
    kin = np.repeat(level[:, None], g, axis=1)
    z = (level - level.mean()) / level.std()
    h = 1.0 / (1.0 + np.exp(-(a + beta * z)))
    obs = np.ones((n, g), dtype=bool)
    for k in range(g - 1):
        cut = obs[:, k] & (rng.random(n) < h)
        obs[cut, k + 1 :] = False
    return act, obs, kin


def test_the_fit_recovers_a_known_beta():  # item 4
    # compared per eV: the fit standardises over the S0 cells at risk (hot trajectories leave
    # early), the simulation over the trajectories, so the per-SD values differ by design
    rng = np.random.default_rng(7)
    act, obs, kin = hot_censoring_sample(rng)
    true_per_eV = 1.0 / (np.std(kin[:, 0]) * HARTREE_IN_EV)
    fit = ipcw.fit_censoring_hazard(act, obs, kin, 1)
    assert fit.ok and fit.sd_eV > 0
    assert fit.beta_per_eV == pytest.approx(fit.beta / fit.sd_eV)
    boots = []
    for _ in range(40):
        r = rng.integers(0, act.shape[0], act.shape[0])
        f = ipcw.fit_censoring_hazard(act[r], obs[r], kin[r], 1)
        assert f.ok
        boots.append(f.beta_per_eV)
    assert abs(fit.beta_per_eV - true_per_eV) <= 4 * np.std(boots, ddof=1)


def test_without_censoring_the_populations_are_the_plain_fractions():  # item 5
    act, _ = markov_sample(np.random.default_rng(3), n=50, g=20)
    obs = np.ones(act.shape, dtype=bool)
    curve, fit = ipcw.ipcw_estimate(act, obs, np.zeros(act.shape), 3, covariate=False)
    assert fit.ok and not fit.hazard.any()
    plain = np.stack([(act == s).mean(axis=0) for s in range(3)], axis=-1)
    np.testing.assert_array_equal(curve, plain)


def test_running_s0_mean():  # item 6
    active = np.array([[1, 0, 0, 1, 0]])
    kinetic = np.array([[5.0, 1.0, 3.0, 7.0, 2.0]])
    observed = np.ones((1, 5), dtype=bool)
    out = ipcw.running_s0_mean(active, kinetic, observed)
    assert np.isnan(out[0, 0])
    np.testing.assert_allclose(out[0, 1:], [1.0, 2.0, 2.0, 2.0])


def test_fit_failures():  # item 7
    act = np.zeros((4, 5), dtype=np.int64)
    obs = np.ones((4, 5), dtype=bool)
    fit = ipcw.fit_censoring_hazard(act, obs, np.ones((4, 5)), 1)
    assert not fit.ok and fit.reason == "no censoring event in S0"
    # complete separation: the two hot trajectories are lost at step 1, the cold ones never
    kin = np.repeat(np.array([[2.0], [2.0], [1.0], [1.0]]), 5, axis=1)
    obs2 = obs.copy()
    obs2[:2, 2:] = False
    fit = ipcw.fit_censoring_hazard(act, obs2, kin, 1)
    assert not fit.ok and fit.reason == "beta on a bound"
    # every S0 event cell has d = r: no informative interval (plan ruling PR3)
    obs3 = obs.copy()
    obs3[:, 2:] = False
    fit = ipcw.fit_censoring_hazard(act, obs3, kin, 1)
    assert not fit.ok and fit.reason == "no informative S0 interval"


def test_kinetic_on_grid_reads_full_step_records():
    t = Trajectory(
        "a",
        "finished",
        1,
        np.array([0, 1, 1, 2]),
        np.array([0, 0, 1, 0]),
        np.array([0.0, 0.5, 0.75, 1.0]),
        np.array([1, 0, 0, 0]),
        np.full(4, -1),
        np.full(4, -1),
        np.zeros(4, dtype=bool),
        np.zeros(4),
        np.eye(2)[[1, 0, 0, 0]],
    )
    out = ipcw.kinetic_on_grid([t], {"a": np.array([1.0, 2.0, 9.0, 3.0])}, np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(out[0, :3], [1.0, 2.0, 3.0])
    assert np.isnan(out[0, 3])
