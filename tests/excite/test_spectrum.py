import math

import numpy as np
import pytest

from gqteashmd.excite.spectrum import (
    CROSS_SECTION_PREFACTOR_EV_A2,
    EPSILON_PER_A2,
    broadened_spectrum,
)


def test_prefactor_values():
    assert pytest.approx(1.09761, rel=1e-5) == CROSS_SECTION_PREFACTOR_EV_A2
    assert pytest.approx(26153.8, rel=1e-5) == EPSILON_PER_A2
    # SI cross-check: pi e^2 hbar / (2 m_e c eps0) in eV*A^2
    e, hbar, me, c, eps0 = (
        1.602176634e-19,
        1.054571817e-34,
        9.1093837015e-31,
        299792458.0,
        8.8541878128e-12,
    )
    si = math.pi * e**2 * hbar / (2 * me * c * eps0)  # J*m^2
    assert si / e * 1e20 == pytest.approx(CROSS_SECTION_PREFACTOR_EV_A2, rel=1e-8)


def test_single_line_integrates_to_prefactor_times_f():
    s = broadened_spectrum(np.array([[5.0]]), np.array([[0.2]]), fwhm_eV=0.1, step_eV=0.001)
    integral = np.trapezoid(s.cross_section_A2 * s.energies_eV / 5.0, s.energies_eV)
    # the grid ends at +-4 s, which drops 6.3e-5 of a Gaussian
    assert integral == pytest.approx(CROSS_SECTION_PREFACTOR_EV_A2 * 0.2, rel=2e-4)
    assert s.energies_eV[np.argmax(s.cross_section_A2)] == pytest.approx(5.0, abs=0.002)
    np.testing.assert_allclose(s.epsilon_M1cm1, s.cross_section_A2 * EPSILON_PER_A2)


def test_average_over_samples_and_state_split():
    e = np.array([[4.0, 6.0], [4.2, 6.1]])
    f = np.array([[0.1, 0.0], [0.3, 0.05]])
    s = broadened_spectrum(e, f, fwhm_eV=0.2, step_eV=0.01)
    np.testing.assert_allclose(s.cross_section_by_state_A2.sum(axis=1), s.cross_section_A2)
    one = broadened_spectrum(e[:1], f[:1], fwhm_eV=0.2, step_eV=0.01)
    assert s.energies_eV[0] <= one.energies_eV[0]
    with pytest.raises(ValueError):
        broadened_spectrum(np.zeros((0, 2)), np.zeros((0, 2)), fwhm_eV=0.1, step_eV=0.01)


@pytest.mark.parametrize(
    "de, first, last",
    [
        # s = 0.1 / 2.35482 = 0.042466, 4 s = 0.169864: e0 = 0.01 floor(483.01) = 4.83,
        # last = first grid point >= 7.169864
        ([[5.0, 7.0]], 4.83, 7.17),
        # (0.1 - 4 s) / step < 0, so the floor max(step, ...) gives e0 = step (never E = 0)
        ([[0.1]], 0.01, 0.27),
    ],
)
def test_grid_rule(de, first, last):
    de = np.array(de)
    spec = broadened_spectrum(de, np.full_like(de, 0.1), fwhm_eV=0.1, step_eV=0.01)
    e = spec.energies_eV
    assert e[0] == pytest.approx(first, abs=1e-12)
    assert e[-1] == pytest.approx(last, abs=1e-9)
    assert np.allclose(np.diff(e), 0.01)
    assert e[-2] < de.max() + 4 * 0.1 / (2 * math.sqrt(2 * math.log(2))) <= e[-1]
