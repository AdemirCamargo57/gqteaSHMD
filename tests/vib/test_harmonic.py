import math

import numpy as np
import pytest

from gqteashmd.core.units import DALTON_IN_ME, HARTREE_IN_CM1
from gqteashmd.vib.harmonic import harmonic_analysis


def _diatomic(k=0.5, r=1.4, m1=1.007825, m2=1.007825):
    coords = np.array([[0.0, 0.0, -r / 2], [0.0, 0.0, r / 2]])
    e = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 1.0]) / math.sqrt(2.0)
    hess = 2.0 * k * np.outer(e, e)  # E = k/2 (z2 - z1 - r)^2
    return np.array([m1, m2]), coords, hess


def test_diatomic_frequency_and_reduced_mass():
    masses, coords, hess = _diatomic()
    result = harmonic_analysis(masses, coords, hess)
    mu = masses[0] * masses[1] / masses.sum()
    expected = math.sqrt(0.5 / (mu * DALTON_IN_ME)) * HARTREE_IN_CM1
    assert result.linear and result.n_external == 5
    assert result.frequencies_cm1.shape == (1,)
    assert result.frequencies_cm1[0] == pytest.approx(expected, rel=1e-10)
    # Reduced mass in the PySCF/Gaussian convention, 1/sum(l_i^2/m_i) with l the unit
    # mass-weighted mode; for a diatomic this is M m1 m2 / (m1^2 + m2^2) (= m for homonuclear),
    # not the textbook mu = m1 m2 / M.
    convention = masses.sum() * masses[0] * masses[1] / (masses[0] ** 2 + masses[1] ** 2)
    assert result.reduced_masses_amu[0] == pytest.approx(convention, rel=1e-10)
    assert result.external_residual_cm1 < 1e-6


def test_rotation_and_translation_do_not_change_frequencies():
    rng = np.random.default_rng(1)
    masses = np.array([15.994915, 1.007825, 1.007825])
    coords = np.array([[0.0, 0.0, 0.22], [0.0, 1.43, -0.88], [0.0, -1.43, -0.88]])
    a = rng.normal(size=(9, 9))
    hess = a @ a.T * 0.05
    # compare the analysis of (coords, H) with that of rotated/translated (coords', R H R^T)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    big = np.kron(np.eye(3), q)
    first = harmonic_analysis(masses, coords, hess)
    second = harmonic_analysis(masses, coords @ q.T + 0.7, big @ hess @ big.T)
    np.testing.assert_allclose(first.frequencies_cm1, second.frequencies_cm1, rtol=1e-9)


def test_modes_are_orthonormal_and_unit_cartesian():
    masses, coords, hess = _diatomic()
    result = harmonic_analysis(masses, coords, hess)
    modes = result.modes_mass_weighted
    np.testing.assert_allclose(modes.T @ modes, np.eye(modes.shape[1]), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(result.modes_cartesian, axis=0), 1.0)


def test_imaginary_modes_are_negative():
    masses, coords, hess = _diatomic(k=-0.5)
    assert harmonic_analysis(masses, coords, hess).frequencies_cm1[0] < 0


def test_single_atom_has_no_vibrations():
    result = harmonic_analysis(np.array([4.002603]), np.zeros((1, 3)), np.zeros((3, 3)))
    assert result.n_external == 3 and result.frequencies_cm1.shape == (0,)


@pytest.mark.parametrize(
    "hess, message",
    [(np.zeros((5, 5)), "shape"), (np.full((6, 6), np.nan), "finite")],
)
def test_bad_hessians(hess, message):
    masses, coords, _ = _diatomic()
    with pytest.raises(ValueError, match=message):
        harmonic_analysis(masses, coords, hess)


def _bent_co2(angle_deg, bond=2.19):
    half = np.radians(angle_deg) / 2.0
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [bond * np.sin(half), 0.0, bond * np.cos(half)],
            [-bond * np.sin(half), 0.0, bond * np.cos(half)],
        ]
    )
    return np.array([12.0, 15.994915, 15.994915]), coords


def test_near_linear_triatomic_is_treated_as_linear():
    masses, coords = _bent_co2(179.9)
    result = harmonic_analysis(masses, coords, np.zeros((9, 9)))
    assert result.linear and result.n_external == 5
    assert result.frequencies_cm1.shape == (4,)


def test_clearly_bent_triatomic_is_nonlinear():
    masses, coords = _bent_co2(120.0)
    result = harmonic_analysis(masses, coords, np.zeros((9, 9)))
    assert not result.linear and result.n_external == 6
    assert result.frequencies_cm1.shape == (3,)
