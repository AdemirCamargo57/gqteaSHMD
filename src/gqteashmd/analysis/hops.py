"""Hop statistics (P8 spec section 6; plan ruling PR2 for frustrated hops)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from gqteashmd.analysis.grid import Trajectory

_TOL = 1e-9


@dataclass(frozen=True)
class HopEvent:
    task: str
    time_fs: float
    from_state: int
    to_state: int | None
    frustrated: bool
    included: bool
    after_horizon: bool


def hop_events(t: Trajectory, *, included: bool, horizon_fs: float) -> list[HopEvent]:
    events: list[HopEvent] = []
    for j in range(t.time_fs.size):
        time = float(t.time_fs[j])
        after = time > horizon_fs + _TOL
        if int(t.hop_to[j]) >= 0:
            events.append(
                HopEvent(t.task, time, int(t.hop_from[j]), int(t.hop_to[j]), False, included, after)
            )
        elif bool(t.frustrated[j]):
            events.append(HopEvent(t.task, time, int(t.active[j]), None, True, included, after))
    return events


def _stats(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.median(values))


def hop_statistics(
    events: Sequence[HopEvent],
    included: Sequence[Trajectory],
    labels: Sequence[str],
    horizon_fs: float,
) -> dict[str, Any]:
    counted = [e for e in events if e.included and not e.after_horizon]
    pairs = Counter(
        f"{labels[e.from_state]}->{labels[e.to_state]}" for e in counted if e.to_state is not None
    )
    frustrated = Counter(labels[e.from_state] for e in counted if e.frustrated)
    per = [sum(1 for e in counted if e.task == t.task and not e.frustrated) for t in included]
    first_hop = [
        min(e.time_fs for e in counted if e.task == t.task and not e.frustrated)
        for t in included
        if any(e.task == t.task and not e.frustrated for e in counted)
    ]
    arrivals: list[float] = []
    for t in included:
        hits = np.flatnonzero((t.active == 0) & (t.time_fs <= horizon_fs + _TOL))
        if hits.size:  # no active-0 record up to the horizon: censored (spec section 6)
            arrivals.append(float(t.time_fs[hits[0]]))
    hop_mean, hop_median = _stats(first_hop)
    s0_mean, s0_median = _stats(arrivals)
    return {
        "pairs": dict(sorted(pairs.items())),
        "frustrated_from": dict(sorted(frustrated.items())),
        "per_trajectory": (
            {
                "min": int(min(per)),
                "median": float(np.median(per)),
                "max": int(max(per)),
                "zero": int(sum(1 for p in per if p == 0)),
            }
            if per
            else {"min": None, "median": None, "max": None, "zero": 0}
        ),
        "first_hop_fs": {"n": len(first_hop), "mean": hop_mean, "median": hop_median},
        "first_s0_arrival_fs": {
            "n_arrived": len(arrivals),
            "n_censored": len(included) - len(arrivals),
            "mean": s0_mean,
            "median": s0_median,
        },
        "totals_all": {
            "hops": sum(1 for e in events if not e.frustrated),
            "frustrated": sum(1 for e in events if e.frustrated),
        },
    }


# --------------------------------------------- spin-free transitions (triplets, spec 4.3)
KINDS = ("S-S", "S-T", "T-S", "T-T")


@dataclass(frozen=True)
class Transition:
    """A change of the active spin-free state between consecutive records; `at_hop` when the
    later record carries an accepted hop (otherwise the MCH character of the active diagonal
    state changed without a hop)."""

    task: str
    time_fs: float
    from_state: int
    to_state: int
    at_hop: bool
    included: bool
    after_horizon: bool


def transitions(t: Trajectory, *, included: bool, horizon_fs: float) -> list[Transition]:
    """Every change of `t.active` between consecutive records. An accepted hop that keeps the
    spin-free state (between the Ms components of one triplet) is not a transition.

    For a schema-2 trajectory (triplets) only: output.h5 keeps every row where the spin-free
    state changes (Task 7 fix ruling), so a hop record's `hop_from`/`hop_to` must be the
    previous record's and its own `active`; otherwise ValueError naming the record (an
    inconsistent file; the missing change is not reconstructed).
    """
    for j in np.flatnonzero(t.hop_to[1:] >= 0) + 1:
        seen = (int(t.hop_from[j]), int(t.hop_to[j]))
        if seen != (int(t.active[j - 1]), int(t.active[j])):
            raise ValueError(
                f"{t.task}: record {j} (t = {float(t.time_fs[j]):g} fs) has a hop "
                f"{seen[0]} -> {seen[1]} but the active spin-free states are "
                f"{int(t.active[j - 1])} -> {int(t.active[j])}: the file is inconsistent"
            )
    changed = np.flatnonzero(t.active[1:] != t.active[:-1]) + 1
    out: list[Transition] = []
    for j in changed:
        time = float(t.time_fs[j])
        out.append(
            Transition(
                t.task,
                time,
                int(t.active[j - 1]),
                int(t.active[j]),
                bool(int(t.hop_to[j]) >= 0),
                included,
                time > horizon_fs + _TOL,
            )
        )
    return out


def _multiplicity(label: str) -> str:
    return "T" if label.startswith("T") else "S"


def transition_statistics(
    events: Sequence[Transition],
    included: Sequence[Trajectory],
    labels: Sequence[str],
    horizon_fs: float,
) -> dict[str, Any]:
    """Counts by kind (S-S, S-T, T-S, T-T; at a hop or without one) and by label pair over the
    included trajectories up to the horizon, and the first singlet-to-triplet time of every
    included trajectory: `None` (censored, like the S0 arrival) when it has none by then."""
    counted = [e for e in events if e.included and not e.after_horizon]
    kinds = {k: {"at_hop": 0, "without_hop": 0} for k in KINDS}
    for e in counted:
        kind = f"{_multiplicity(labels[e.from_state])}-{_multiplicity(labels[e.to_state])}"
        kinds[kind]["at_hop" if e.at_hop else "without_hop"] += 1
    pairs = Counter(f"{labels[e.from_state]}->{labels[e.to_state]}" for e in counted)
    first: dict[str, float | None] = {}
    for t in included:
        times = [
            e.time_fs
            for e in counted
            if e.task == t.task
            and _multiplicity(labels[e.from_state]) == "S"
            and _multiplicity(labels[e.to_state]) == "T"
        ]
        first[t.task] = min(times) if times else None
    crossed = [v for v in first.values() if v is not None]
    mean, median = _stats(crossed)
    return {
        "kinds": kinds,
        "pairs": dict(sorted(pairs.items())),
        "first_singlet_to_triplet_fs": {
            "per_trajectory": first,
            "n_crossed": len(crossed),
            "n_censored": len(first) - len(crossed),
            "mean": mean,
            "median": median,
        },
    }
