import math

from gqteashmd.core import units


def test_codata_2018_values():
    assert units.BOHR_IN_ANGSTROM == 0.529177210903
    assert units.HARTREE_IN_EV == 27.211386245988
    assert units.HARTREE_IN_CM1 == 219474.6313632
    assert units.DALTON_IN_ME == 1822.888486209
    assert units.AU_TIME_IN_FS == 0.024188843265857


def test_angstrom_to_bohr_is_the_inverse():
    assert math.isclose(units.ANGSTROM_TO_BOHR * units.BOHR_IN_ANGSTROM, 1.0, rel_tol=1e-15)
