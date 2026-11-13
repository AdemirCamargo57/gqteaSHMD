import pytest

from gqteashmd.electronic.pyscf_casscf import guess_active_sets, pick_lowest


def test_cas22_has_four_guesses_with_the_default_first():
    # ethylene: 8 doubly occupied orbitals (0-based 0..7), plenty of virtuals
    assert guess_active_sets(8, 38, 2, 2) == [(7, 8), (6, 8), (6, 9), (7, 9)]


def test_cas44_has_nine_guesses():
    guesses = guess_active_sets(8, 38, 4, 4)
    assert len(guesses) == 9
    assert guesses[0] == (6, 7, 8, 9)
    assert guesses[1:] == sorted(guesses[1:])
    assert len(set(guesses)) == 9


def test_cas23_has_two_occupied_and_three_virtual_choices():
    guesses = guess_active_sets(8, 38, 3, 2)  # 1 occupied active, 2 virtual active
    assert guesses[0] == (7, 8, 9)
    assert len(guesses) == 2 * 3


def test_no_virtual_active_orbital():
    # nelecas = 2 ncas: only the occupied window matters
    assert guess_active_sets(8, 38, 1, 2) == [(7,), (6,)]


def test_the_occupied_window_is_clipped_at_the_first_orbital():
    # core = 0: the window would start at -1
    assert guess_active_sets(1, 10, 2, 2) == [(0, 1), (0, 2)]


def test_the_virtual_window_is_clipped_at_the_last_orbital():
    assert guess_active_sets(8, 9, 2, 2) == [(7, 8), (6, 8)]


@pytest.mark.parametrize(
    ("energies", "expected"),
    [
        ([1.0, 0.5, 0.5 + 5e-9], 1),
        ([0.5 + 5e-9, 0.5], 0),
        ([0.3, 0.1], 1),
        ([0.1, 0.1 + 2e-8], 0),
        ([0.1 + 2e-8, 0.1], 1),
    ],
)
def test_pick_lowest_takes_the_first_within_the_tie_tolerance(energies, expected):
    assert pick_lowest(energies) == expected


def test_pick_lowest_needs_candidates():
    with pytest.raises(ValueError, match="no candidate"):
        pick_lowest([])
