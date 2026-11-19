"""Censored estimators (censored analysis spec, sections 2.2 and 4.1 items 1-2)."""

import numpy as np
import pytest

from gqteashmd.analysis import censoring, populations
from gqteashmd.analysis.grid import Trajectory, on_grid
from gqteashmd.core.units import HARTREE_IN_EV

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


def _traj(task, outcome, active, *, e_total=None, every=1):
    n = len(active)
    return Trajectory(
        task=task,
        outcome=outcome,
        initial_state=int(active[0]) if n else 1,
        grid_step=np.arange(n, dtype=np.int64),
        ticks=np.zeros(n, dtype=np.int64),
        time_fs=0.5 * np.arange(n),
        active=np.asarray(active, dtype=np.int64),
        hop_from=np.full(n, -1),
        hop_to=np.full(n, -1),
        frustrated=np.zeros(n, dtype=bool),
        e_total=np.zeros(n) if e_total is None else np.asarray(e_total, dtype=float),
        populations=np.eye(2)[np.asarray(active, dtype=np.int64)],
    )


def test_select_censored_sorts_every_ending():
    trajs = {
        "traj_00000": _traj("traj_00000", "finished", [1, 1, 0, 0, 0]),
        "traj_00001": _traj("traj_00001", "overlap_failure", [1, 0, 0]),
        "traj_00002": _traj("traj_00002", "electronic_failure", []),
        "traj_00003": None,
        "traj_00004": _traj(
            "traj_00004", "finished", [1, 1, 1, 1, 1], e_total=[0, 0, 0, 0.2 / HARTREE_IN_EV, 0]
        ),
    }
    tasks = list(trajs)
    used, at, cens, excl = censoring.select_censored(
        tasks, trajs, output_every=1, n_total=4, dt_fs=0.5, energy_jump_eV=0.1
    )
    assert [t.task for t in used] == ["traj_00000", "traj_00001", "traj_00004"]
    assert at == {"traj_00001": 2, "traj_00004": 2}  # the jump is at record 3: censored at 2
    assert [(c.task, c.reason, c.grid_step, c.time_fs) for c in cens] == [
        ("traj_00001", "outcome", 2, 1.0),
        ("traj_00004", "energy_jump", 2, 1.0),
    ]
    assert [(e.task, e.reason) for e in excl] == [
        ("traj_00002", "electronic_failure"),
        ("traj_00003", "task_failed"),
    ]
    _, at_off, _, _ = censoring.select_censored(
        tasks, trajs, output_every=1, n_total=4, dt_fs=0.5, energy_jump_eV=None
    )
    assert at_off == {"traj_00001": 2}  # exclude_energy_jumps = false: no jump censoring


def test_on_grid_marks_the_censored_part_as_unobserved():
    a = _traj("a", "finished", [1, 1, 0, 0, 0])
    b = _traj("b", "overlap_failure", [1, 0, 0])
    g = on_grid([a, b], output_every=1, dt_fs=0.5, n_total=4, n_states=2, censored_at={"b": 2})
    np.testing.assert_array_equal(g.observed, [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    assert g.cut == () and g.horizon_steps == 4
    g2 = on_grid([a], output_every=1, dt_fs=0.5, n_total=4, n_states=2, horizon=2)
    assert g2.horizon_steps == 2 and g2.observed.shape == (1, 3) and g2.observed.all()
