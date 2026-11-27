import numpy as np

from gqteashmd.analysis.grid import Trajectory
from gqteashmd.analysis.hops import (
    HopEvent,
    Transition,
    hop_events,
    hop_statistics,
    transition_statistics,
    transitions,
)

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


# ------------------------------------------------- spin-free transitions (triplets, spec 4.3)
SOC_LABELS = ("S0", "S1", "T1", "T2")


def soc_traj(task, active, hop_from, hop_to, dt=0.5):
    n = len(active)
    return Trajectory(
        task,
        "finished",
        int(active[0]),
        np.arange(n),
        np.zeros(n, int),
        np.arange(n) * dt,
        np.asarray(active),
        np.asarray(hop_from),
        np.asarray(hop_to),
        np.zeros(n, bool),
        np.zeros(n),
        np.zeros((n, len(SOC_LABELS))),
    )


# S1 -> T1 at a hop (record 2), T1 -> T2 without a hop (record 3), T2 -> S0 at a hop (record 4)
ISC = soc_traj("isc", [1, 1, 2, 3, 0], [-1, -1, 1, -1, 3], [-1, -1, 2, -1, 0])
STAY = soc_traj("stay", [1, 1, 1, 1, 1], [-1] * 5, [-1] * 5)
NO_KIND = {"at_hop": 0, "without_hop": 0}


def test_transitions_are_the_changes_of_the_active_spin_free_state():
    assert transitions(ISC, included=True, horizon_fs=1.5) == [
        Transition("isc", 1.0, 1, 2, True, True, False),
        Transition("isc", 1.5, 2, 3, False, True, False),
        Transition("isc", 2.0, 3, 0, True, True, True),
    ]
    assert transitions(STAY, included=True, horizon_fs=2.0) == []


def test_a_hop_within_one_spin_free_state_is_not_a_transition():
    # an accepted diagonal hop between Ms components of T1 is stored as (T1, T1) (ledger
    # ruling): the hop table keeps it, the transitions do not
    t = soc_traj("ms", [1, 2, 2, 2], [-1, 1, 2, -1], [-1, 2, 2, -1])
    assert transitions(t, included=True, horizon_fs=2.0) == [
        Transition("ms", 0.5, 1, 2, True, True, False)
    ]
    s = hop_statistics(hop_events(t, included=True, horizon_fs=2.0), [t], SOC_LABELS, 2.0)
    assert s["pairs"] == {"S1->T1": 1, "T1->T1": 1}


def test_transition_statistics_by_kind():
    events = transitions(ISC, included=True, horizon_fs=2.0)
    events += transitions(STAY, included=True, horizon_fs=2.0)
    s = transition_statistics(events, [ISC, STAY], SOC_LABELS, 2.0)
    assert s["kinds"] == {
        "S-S": NO_KIND,
        "S-T": {"at_hop": 1, "without_hop": 0},
        "T-S": {"at_hop": 1, "without_hop": 0},
        "T-T": {"at_hop": 0, "without_hop": 1},
    }
    assert s["pairs"] == {"S1->T1": 1, "T1->T2": 1, "T2->S0": 1}
    assert s["first_singlet_to_triplet_fs"] == {
        "per_trajectory": {"isc": 1.0, "stay": None},
        "n_crossed": 1,
        "n_censored": 1,
        "mean": 1.0,
        "median": 1.0,
    }


def test_transitions_after_the_horizon_or_not_included_are_not_counted():
    # horizon 0.5 fs: the S1 -> T1 change at 1.0 fs comes after it, so the first S-to-T time
    # is censored like the S0 arrival
    events = transitions(ISC, included=True, horizon_fs=0.5)
    events += transitions(soc_traj("x", [1, 2], [-1, 1], [-1, 2]), included=False, horizon_fs=0.5)
    s = transition_statistics(events, [ISC], SOC_LABELS, 0.5)
    assert all(v == NO_KIND for v in s["kinds"].values())
    assert s["pairs"] == {}
    assert s["first_singlet_to_triplet_fs"] == {
        "per_trajectory": {"isc": None},
        "n_crossed": 0,
        "n_censored": 1,
        "mean": None,
        "median": None,
    }


def test_the_first_singlet_to_triplet_time_counts_a_change_without_a_hop():
    a = soc_traj("a", [1, 1, 1, 3, 3], [-1] * 5, [-1] * 5)  # S1 -> T2 at 1.5 fs, no hop
    b = soc_traj("b", [1, 2, 2, 2, 2], [-1, 1, -1, -1, -1], [-1, 2, -1, -1, -1])  # at 0.5 fs
    events = transitions(a, included=True, horizon_fs=2.0)
    events += transitions(b, included=True, horizon_fs=2.0)
    s = transition_statistics(events, [a, b], SOC_LABELS, 2.0)
    assert s["kinds"]["S-T"] == {"at_hop": 1, "without_hop": 1}
    first = s["first_singlet_to_triplet_fs"]
    assert first["per_trajectory"] == {"a": 1.5, "b": 0.5}
    assert first["n_crossed"] == 2 and first["n_censored"] == 0
    assert first["mean"] == 1.0 and first["median"] == 1.0
