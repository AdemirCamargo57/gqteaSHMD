"""Hop statistics (P8 spec section 6; plan ruling PR2 for frustrated hops)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from gqteashmd.analysis.grid import STOPPED_IN_S0, Trajectory

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
        if hits.size:
            arrivals.append(float(t.time_fs[hits[0]]))
        elif t.outcome == STOPPED_IN_S0:
            arrivals.append(float(t.time_fs[-1]))
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
