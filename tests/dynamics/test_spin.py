"""Canonical order of the spin-expanded states."""

from __future__ import annotations

import pytest

from gqteashmd.dynamics.spin import (
    ExpandedState,
    check_counts,
    expanded_states,
    n_expanded,
    spin_free_label,
)


def labels(counts):
    return [s.label() for s in expanded_states(counts)]


def test_singlets_only():
    assert labels((3,)) == ["S0", "S1", "S2"]
    assert n_expanded((3,)) == 3


def test_two_singlets_one_triplet():
    assert labels((2, 0, 1)) == ["S0", "S1", "T1(-1)", "T1(0)", "T1(+1)"]
    assert n_expanded((2, 0, 1)) == 5


def test_one_singlet_two_triplets_order_is_state_then_ms():
    assert labels((1, 0, 2)) == [
        "S0",
        "T1(-1)",
        "T2(-1)",
        "T1(0)",
        "T2(0)",
        "T1(+1)",
        "T2(+1)",
    ]
    states = expanded_states((1, 0, 2))
    assert states[2] == ExpandedState(3, 1, -1)
    assert [spin_free_label(s) for s in states] == ["S0", "T1", "T2", "T1", "T2", "T1", "T2"]


def test_count_forms_accepted():
    assert check_counts((2,)) == (2,)
    assert check_counts([2, 0]) == (2, 0)
    assert check_counts((2, 0, 1)) == (2, 0, 1)


@pytest.mark.parametrize("counts", [(2, 1), (2, 1, 1), (1, 0, 0, 1), (), (-1,), (0, 0, 0)])
def test_invalid_counts_rejected(counts):
    with pytest.raises(ValueError):
        check_counts(counts)
    with pytest.raises(ValueError):
        expanded_states(counts)
