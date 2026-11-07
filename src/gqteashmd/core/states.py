"""Electronic state labels: S0, S1, ... (singlets, 0-based) and T1, T2, ... (triplets)."""

from __future__ import annotations

import re
from dataclasses import dataclass

_PREFIX = {1: "S", 3: "T"}
_MULTIPLICITY = {prefix: mult for mult, prefix in _PREFIX.items()}
_PATTERN = re.compile(r"([ST])([0-9]+)")


@dataclass(frozen=True, order=True)
class StateLabel:
    multiplicity: int
    index: int

    def __post_init__(self) -> None:
        if self.multiplicity not in _PREFIX:
            raise ValueError(f"Unsupported multiplicity {self.multiplicity}; use 1 (S) or 3 (T).")
        lowest = 0 if self.multiplicity == 1 else 1
        if self.index < lowest:
            raise ValueError(f"{_PREFIX[self.multiplicity]}{self.index} is not a valid state.")

    def __str__(self) -> str:
        return f"{_PREFIX[self.multiplicity]}{self.index}"

    @classmethod
    def parse(cls, text: str) -> StateLabel:
        match = _PATTERN.fullmatch(text.strip().upper())
        if match is None:
            raise ValueError(f"Invalid state label {text!r}; use e.g. S0, S1, T1.")
        return cls(_MULTIPLICITY[match.group(1)], int(match.group(2)))


def states_from_counts(singlets: int, triplets: int = 0) -> tuple[StateLabel, ...]:
    """States S0..S(singlets-1) followed by T1..T(triplets); `singlets` includes S0."""
    if singlets < 1:
        raise ValueError("At least one singlet state (S0) is required.")
    if triplets < 0:
        raise ValueError(f"The number of triplets cannot be negative, got {triplets}.")
    return tuple(StateLabel(1, i) for i in range(singlets)) + tuple(
        StateLabel(3, i) for i in range(1, triplets + 1)
    )
