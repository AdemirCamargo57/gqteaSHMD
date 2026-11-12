import numpy as np

from gqteashmd.analysis.grid import Trajectory
from gqteashmd.analysis.hops import HopEvent, hop_events, hop_statistics

LABELS = ("S0", "S1", "S2")


def traj(task, outcome, active, hop_from, hop_to, frustrated, dt=0.5):
    n = len(active)
    return Trajectory(
        task,
        outcome,
        int(active[0]),
        np.arange(n),
        np.zeros(n, int),
        np.arange(n) * dt,
        np.asarray(active),
        np.asarray(hop_from),
        np.asarray(hop_to),
        np.asarray(frustrated, bool),
        np.zeros(n),
        np.zeros((n, 3)),
    )


A = traj("a", "finished", [2, 2, 1, 1, 0], [-1, -1, 2, -1, 1], [-1, -1, 1, -1, 0], [0, 1, 0, 0, 0])
B = traj("b", "finished", [1, 1, 1, 1, 1], [-1] * 5, [-1] * 5, [0] * 5)


def test_events_and_after_horizon():
    events = hop_events(A, included=True, horizon_fs=1.5)
    assert events == [
        HopEvent("a", 0.5, 2, None, True, True, False),
        HopEvent("a", 1.0, 2, 1, False, True, False),
        HopEvent("a", 2.0, 1, 0, False, True, True),
    ]


def test_statistics():
    events = hop_events(A, included=True, horizon_fs=1.5) + hop_events(
        B, included=True, horizon_fs=1.5
    )
    s = hop_statistics(events, [A, B], LABELS, 1.5)
    assert s["pairs"] == {"S2->S1": 1}
    assert s["frustrated_from"] == {"S2": 1}
    assert s["per_trajectory"] == {"min": 0, "median": 0.5, "max": 1, "zero": 1}
    assert s["first_hop_fs"] == {"n": 1, "mean": 1.0, "median": 1.0}
    assert s["first_s0_arrival_fs"] == {
        "n_arrived": 0,
        "n_censored": 2,
        "mean": None,
        "median": None,
    }
    assert s["totals_all"] == {"hops": 2, "frustrated": 1}


def test_first_s0_arrival_is_the_first_active_0_record():
    c = traj("c", "finished", [1, 1, 0, 0, 0], [-1, -1, 1, -1, -1], [-1, -1, 0, -1, -1], [0] * 5)
    events = hop_events(c, included=True, horizon_fs=2.0)
    s = hop_statistics(events, [c], LABELS, 2.0)
    assert s["first_s0_arrival_fs"] == {"n_arrived": 1, "n_censored": 0, "mean": 1.0, "median": 1.0}


def test_stopped_in_s0_after_the_horizon_is_censored():
    # S0 is reached at 2.5 fs, after the horizon 2.0 fs (a lowered tmax_fs): censored (spec 6)
    d = traj(
        "d",
        "stopped_in_s0",
        [1, 1, 1, 1, 1, 0, 0],
        [-1, -1, -1, -1, -1, 1, -1],
        [-1, -1, -1, -1, -1, 0, -1],
        [0] * 7,
    )
    events = hop_events(d, included=True, horizon_fs=2.0)
    s = hop_statistics(events, [d], LABELS, 2.0)
    assert s["first_s0_arrival_fs"] == {
        "n_arrived": 0,
        "n_censored": 1,
        "mean": None,
        "median": None,
    }
