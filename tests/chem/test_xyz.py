import pytest

from gqteashmd.chem.elements import atomic_number, normalize_symbol
from gqteashmd.chem.xyz import Geometry, format_xyz, parse_xyz


def test_atomic_numbers_and_normalization():
    assert atomic_number("H") == 1
    assert atomic_number("cl") == 17
    assert atomic_number("RN") == 86
    assert normalize_symbol("br") == "Br"
    with pytest.raises(ValueError, match="Xx"):
        atomic_number("Xx")


def test_parse_water(water_xyz):
    geom = parse_xyz(water_xyz)
    assert geom.symbols == ("O", "H", "H")
    assert geom.coords_angstrom[1] == (0.0, 0.7572, -0.4692)
    assert geom.comment == "water"


def test_parse_normalizes_symbols_and_ignores_extra_columns_and_trailing_blank_lines():
    geom = parse_xyz("1\n\ncl 1.0 2.0 3.0 0.5\n\n\n")
    assert geom.symbols == ("Cl",)
    assert geom.coords_angstrom == ((1.0, 2.0, 3.0),)


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "line 1"),
        ("two\n\nH 0 0 0\n", "line 1"),
        ("0\n\n", "at least one atom"),
        ("2\n\nH 0 0 0\n", "expected 2 atoms"),
        ("1\n\nH 0 0\n", "line 3"),
        ("1\n\nH 0 x 0\n", "line 3"),
        ("1\n\nQq 0 0 0\n", "line 3"),
        ("1\n\nH 0 0 0\nH 1 1 1\n", "more lines"),
    ],
)
def test_parse_errors_report_location(text, message):
    with pytest.raises(ValueError, match=message):
        parse_xyz(text)


def test_format_roundtrip(water_xyz):
    geom = parse_xyz(water_xyz)
    again = parse_xyz(format_xyz(geom))
    assert again == geom
    assert format_xyz(Geometry(("H",), ((0.0, 0.0, 0.0),))).splitlines()[0] == "1"
