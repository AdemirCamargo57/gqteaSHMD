import pytest

from gqteashmd.chem.masses import isotope_mass


def test_most_abundant_isotope_masses():
    assert isotope_mass("H") == 1.007825
    assert isotope_mass("C") == 12.0
    assert isotope_mass("O") == 15.994915
    assert isotope_mass("Cl") == 34.968853
    assert isotope_mass("Rn") == 222.01757


def test_unknown_symbol():
    with pytest.raises(ValueError, match="Xx"):
        isotope_mass("Xx")
