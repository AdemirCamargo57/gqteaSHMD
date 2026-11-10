"""Spin layout of spin-orbit-coupled models: the canonical order of the expanded MCH states.

A model has state counts ``(n_1, n_2, n_3)``: singlets, doublets, triplets. Only singlets and
triplets are supported (``n_2 = 0``). A triplet contributes three components, M_s = -1, 0, +1.
The canonical order has the state index innermost, then M_s, then the multiplicity:
S0 ... S(n_1-1), then T1 ... T(n_3) with M_s = -1, the same with M_s = 0, the same with
M_s = +1. The number of expanded states is ``n_1 + 3 n_3``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ExpandedState:
    """One spin component of an MCH state; ``index`` is 0-based within the multiplicity."""

    multiplicity: int
    index: int
    ms: int

    def label(self) -> str:
        """``S0``, ``S1`` for singlets; ``T1(-1)``, ``T1(0)``, ``T1(+1)`` for triplets."""
        if self.multiplicity == 1:
            return f"S{self.index}"
        return f"T{self.index + 1}({self.ms:+d})".replace("(+0)", "(0)")


def check_counts(counts: Sequence[int]) -> tuple[int, ...]:
    """Validate the state counts ``(n_1,)``, ``(n_1, 0)`` or ``(n_1, 0, n_3)``."""
    values = tuple(counts)
    if not 1 <= len(values) <= 3:
        raise ValueError(
            f"State counts must list singlets, doublets and triplets (1 to 3 numbers), "
            f"got {len(values)}: {values}."
        )
    for n in values:
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"State counts must be non-negative integers, got {values}.")
    if len(values) >= 2 and values[1] != 0:
        raise ValueError(
            f"Doublet states are not supported (counts {values}); use singlets and triplets."
        )
    if sum(values) == 0:
        raise ValueError(f"State counts {values} contain no state.")
    return values


def _n1_n3(counts: Sequence[int]) -> tuple[int, int]:
    values = check_counts(counts)
    return values[0], values[2] if len(values) == 3 else 0


def expanded_states(counts: Sequence[int]) -> tuple[ExpandedState, ...]:
    """The expanded states in canonical order."""
    n1, n3 = _n1_n3(counts)
    states = [ExpandedState(1, i, 0) for i in range(n1)]
    for ms in (-1, 0, 1):
        states.extend(ExpandedState(3, i, ms) for i in range(n3))
    return tuple(states)


def n_expanded(counts: Sequence[int]) -> int:
    """Number of expanded states, ``n_1 + 3 n_3``."""
    n1, n3 = _n1_n3(counts)
    return n1 + 3 * n3


def spin_free_label(state: ExpandedState) -> str:
    """Label without M_s: ``S1``, ``T1``."""
    return f"S{state.index}" if state.multiplicity == 1 else f"T{state.index + 1}"
