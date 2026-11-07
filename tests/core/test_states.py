import pytest

from gqteashmd.core.states import StateLabel, states_from_counts

S0, S1, S2 = StateLabel(1, 0), StateLabel(1, 1), StateLabel(1, 2)
T1, T2 = StateLabel(3, 1), StateLabel(3, 2)


def test_parse_and_format():
    assert StateLabel.parse("S0") == S0
    assert StateLabel.parse(" t1 ") == T1
    assert str(T2) == "T2"
    assert str(S1) == "S1"


@pytest.mark.parametrize("bad", ["", "X1", "S", "S-1", "T0", "Q1", "S1.5", "S01x", "S٣"])
def test_invalid_labels(bad):
    with pytest.raises(ValueError):
        StateLabel.parse(bad)


def test_invalid_construction():
    with pytest.raises(ValueError):
        StateLabel(2, 0)
    with pytest.raises(ValueError):
        StateLabel(3, 0)
    with pytest.raises(ValueError):
        StateLabel(1, -1)


def test_states_from_counts_include_s0():
    assert states_from_counts(3) == (S0, S1, S2)
    assert states_from_counts(2, 2) == (S0, S1, T1, T2)
    with pytest.raises(ValueError, match="S0"):
        states_from_counts(0)


def test_labels_sort_singlets_first():
    assert sorted([T1, S1, S0]) == [S0, S1, T1]


def test_negative_triplets_are_rejected():
    with pytest.raises(ValueError, match="triplets"):
        states_from_counts(2, -1)
