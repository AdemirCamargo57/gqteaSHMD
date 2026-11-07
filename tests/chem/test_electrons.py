import pytest

from gqteashmd.chem.electrons import count_electrons


@pytest.mark.parametrize(
    "numbers, charge, mult, expected",
    [
        ([8, 1, 1], 0, 1, 10),
        ([8, 1, 1], 1, 2, 9),
        ([8, 8], 0, 3, 16),
        ([6, 6, 1, 1, 1, 1], 0, 1, 16),
    ],
)
def test_valid_electron_counts(numbers, charge, mult, expected):
    assert count_electrons(numbers, charge, mult) == expected


@pytest.mark.parametrize(
    "numbers, charge, mult",
    [([8, 1, 1], 0, 2), ([8, 1, 1], 1, 1), ([8, 1, 1], 0, 12), ([8, 1, 1], 10, 1)],
)
def test_inconsistent_counts(numbers, charge, mult):
    with pytest.raises(ValueError, match="inconsistent with"):
        count_electrons(numbers, charge, mult)


def test_inconsistent_message_is_exact():
    with pytest.raises(ValueError) as info:
        count_electrons([8, 1, 1], 0, 2)
    assert str(info.value) == "Charge 0 and multiplicity 2 are inconsistent with 10 electrons."
