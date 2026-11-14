"""The P9b paired comparison rules on synthetic ensembles (spec 3.3; plan Task 3)."""

import numpy as np
from tests.diagnosis.conftest import load_tool

from gqteashmd.analysis.grid import Trajectory
from gqteashmd.analysis.populations import bootstrap_indices

ec = load_tool("sharc_compare", "ensemble_compare")
N_TOTAL = 200  # 100 fs at 0.5 fs
DT = 0.5


def _traj(task, hop_step=None, *, n_total=N_TOTAL, outcome="finished", jump_eV=0.0):
    steps = np.arange(n_total + 1)
    active = np.ones(steps.size, dtype=np.int64)
    hop_to = np.full(steps.size, -1)
    hop_from = np.full(steps.size, -1)
    if hop_step is not None:
        active[hop_step:] = 0
        hop_to[hop_step], hop_from[hop_step] = 0, 1
    e = np.zeros(steps.size)
    e[-1] = jump_eV / 27.211386245988
    pops = np.zeros((steps.size, 3))
    pops[np.arange(steps.size), active] = 1.0
    return Trajectory(
        task,
        outcome,
        1,
        steps,
        np.zeros(steps.size, np.int64),
        steps * DT,
        active,
        hop_from,
        hop_to,
        np.zeros(steps.size, bool),
        e,
        pops,
    )


def _ensemble(n, rng, *, shift=0):
    hops = (10 + rng.exponential(40.0, size=n)) / DT + shift
    return {
        f"traj_{k:05d}": _traj(f"traj_{k:05d}", int(h) if h <= N_TOTAL else None)
        for k, h in enumerate(hops)
    }


def _compare(a, b, **kw):
    return ec.compare(
        a, b, tasks=sorted(a), n_total=N_TOTAL, dt_fs=DT, n_states=3, seed=2028, asserted=True, **kw
    )


def test_paired_tasks_needs_both_kept():
    a = {"t0": _traj("t0"), "t1": _traj("t1", outcome="overlap_failure"), "t2": _traj("t2")}
    b = {"t0": _traj("t0"), "t1": _traj("t1"), "t2": _traj("t2", jump_eV=0.5)}
    assert ec.paired_tasks(a, b, N_TOTAL) == ["t0"]


def test_identical_ensembles_pass_every_rule():
    a = _ensemble(60, np.random.default_rng(1))
    r = _compare(a, dict(a))
    assert [v["verdict"] for v in r["rules"]] == ["pass", "pass", "pass", "pass"]
    pop = r["rules"][0]
    assert pop["max_se"] == 0.0  # pairs resampled together: identical sides have zero spread


def test_a_shifted_population_fails():
    a = _ensemble(60, np.random.default_rng(2))
    b = {k: _traj(k, 4) for k in a}  # everyone hops at 2 fs
    r = _compare(a, b)
    assert r["rules"][0]["rule"] == "populations" and r["rules"][0]["verdict"] == "fail"


def test_fewer_than_thirty_paired_is_inconclusive():
    a = _ensemble(29, np.random.default_rng(3))
    r = _compare(a, dict(a))
    assert r["n_paired"] == 29
    assert {v["verdict"] for v in r["rules"]} == {"inconclusive"}


def test_population_rule_bound_is_4se_plus_floor():
    active_a = np.ones((40, 3), dtype=np.int64)
    active_b = active_a.copy()
    active_b[:1, :] = 0  # one of 40 differs everywhere: dP = 0.025
    idx = bootstrap_indices(np.random.default_rng(0), 40, 1000)
    r = ec.population_rule(active_a, active_b, 3, idx)
    assert abs(r["max_abs_difference"] - 0.025) < 1e-12
    assert r["verdict"] == ("pass" if 4 * r["max_se"] + 0.02 >= 0.025 else "fail")


def test_ks_rule():
    rng = np.random.default_rng(4)
    same = ec.ks_rule(list(rng.uniform(0, 10, 50)), list(rng.uniform(0, 10, 50)))
    assert same["verdict"] == "pass"
    apart = ec.ks_rule(list(rng.uniform(0, 10, 50)), list(rng.uniform(50, 60, 50)))
    assert apart["verdict"] == "fail" and apart["p_value"] < 1e-3
    assert ec.ks_rule([1.0], [1.0, 2.0])["verdict"] == "inconclusive"


def test_a_fit_without_decay_is_inconclusive():
    t = np.arange(0.0, 100.0 + 1e-9, 1.0)
    per = np.ones((40, t.size))
    idx = bootstrap_indices(np.random.default_rng(0), 40, 100)
    r = ec.fit_rule(t, per, per.copy(), idx)
    assert r["verdict"] == "inconclusive" and r["reason"]


def test_first_hop_times_skip_trajectories_without_hops():
    assert ec.first_hop_times([_traj("a", 20), _traj("b")]) == [10.0]


def test_report_tables_are_present():
    a = _ensemble(40, np.random.default_rng(5))
    r = _compare(a, dict(a))
    for key in (
        "kept_ours",
        "kept_theirs",
        "outcomes_ours",
        "outcomes_theirs",
        "quantum_max_abs_difference",
        "hop_pairs_ours",
        "hop_pairs_theirs",
        "drift_eV_ours",
        "drift_eV_theirs",
        "start_difference",
    ):
        assert key in r["report"], key
