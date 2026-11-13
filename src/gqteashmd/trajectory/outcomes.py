"""How a trajectory of the dynamics stage ended (spec section 5); every outcome is a DONE task."""

from __future__ import annotations

from enum import StrEnum


class TrajectoryOutcome(StrEnum):
    FINISHED = "finished"
    ENERGY_DRIFT = "energy_drift"
    STOPPED_IN_S0 = "stopped_in_s0"
    OVERLAP_FAILURE = "overlap_failure"
    ELECTRONIC_FAILURE = "electronic_failure"
    ELECTRONIC_DISCONTINUITY = "electronic_discontinuity"
