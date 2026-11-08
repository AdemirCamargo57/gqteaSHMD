import math

import numpy as np
import pytest

from gqteashmd.core.units import BOLTZMANN_HARTREE_PER_K, DALTON_IN_ME, HARTREE_IN_CM1
from gqteashmd.vib.harmonic import harmonic_analysis
from gqteashmd.vib.wigner import mode_widths, remove_rigid_motion, sample_wigner

WATER_MASSES = np.array([15.994915, 1.007825, 1.007825])
WATER_BOHR = np.array(
    [[0.0, 0.0, 0.28377702], [0.0, 1.43256453, -0.91771557], [0.0, -1.43256453, -0.91771557]]
)


def _model_water():
    """A synthetic, translation/rotation-free Hessian projected into the vibrational space."""
    rng = np.random.default_rng(5)
    a = rng.normal(size=(9, 9))
    analysis = harmonic_analysis(WATER_MASSES, WATER_BOHR, a @ a.T * 0.2 + np.eye(9) * 0.3)
    return analysis


def test_widths_at_zero_and_high_temperature():
    omega = np.array([0.01, 0.02])
    sq, sp = mode_widths(omega, 0.0)
    np.testing.assert_allclose(sq**2, 1 / (2 * omega))
    np.testing.assert_allclose(sp**2, omega / 2)
    t = 1e5  # kT >> omega: classical limit <E> = kT
    sq, sp = mode_widths(omega, t)
    kt = BOLTZMANN_HARTREE_PER_K * t
    np.testing.assert_allclose(0.5 * sp**2 + 0.5 * omega**2 * sq**2, kt, rtol=1e-3)


def test_mean_energy_matches_zero_point_energy():
    analysis = _model_water()
    samples = sample_wigner(
        WATER_MASSES,
        WATER_BOHR,
        analysis.frequencies_cm1,
        analysis.modes_mass_weighted,
        n_samples=40000,
        temperature_K=0.0,
        low_frequency_cutoff_cm1=10.0,
        rng=np.random.default_rng(1),
    )
    zpe = 0.5 * (analysis.frequencies_cm1 / HARTREE_IN_CM1).sum()
    assert samples.expected_mean_energy_hartree == pytest.approx(zpe, rel=1e-12)
    se = samples.expected_energy_std_hartree / math.sqrt(40000)
    assert abs(samples.harmonic_energy_hartree.mean() - zpe) < 5 * se
    assert samples.positions_bohr.shape == (40000, 3, 3)


def test_thermal_mean_energy():
    analysis = _model_water()
    t = 3000.0
    samples = sample_wigner(
        WATER_MASSES,
        WATER_BOHR,
        analysis.frequencies_cm1,
        analysis.modes_mass_weighted,
        n_samples=40000,
        temperature_K=t,
        low_frequency_cutoff_cm1=10.0,
        rng=np.random.default_rng(2),
    )
    omega = analysis.frequencies_cm1 / HARTREE_IN_CM1
    expected = (0.5 * omega / np.tanh(omega / (2 * BOLTZMANN_HARTREE_PER_K * t))).sum()
    assert samples.expected_mean_energy_hartree == pytest.approx(expected, rel=1e-12)
    se = samples.expected_energy_std_hartree / math.sqrt(40000)
    assert abs(samples.harmonic_energy_hartree.mean() - expected) < 5 * se


def test_momenta_are_removed():
    analysis = _model_water()
    samples = sample_wigner(
        WATER_MASSES,
        WATER_BOHR,
        analysis.frequencies_cm1,
        analysis.modes_mass_weighted,
        n_samples=500,
        temperature_K=0.0,
        low_frequency_cutoff_cm1=10.0,
        rng=np.random.default_rng(3),
    )
    m = WATER_MASSES * DALTON_IN_ME
    v, x = samples.velocities_au, samples.positions_bohr
    p_total = (m[None, :, None] * v).sum(axis=1)
    com = (m[None, :, None] * x).sum(axis=1) / m.sum()
    l_total = (m[None, :, None] * np.cross(x - com[:, None, :], v)).sum(axis=1)
    assert np.abs(p_total).max() < 1e-10 and np.abs(l_total).max() < 1e-8
    assert np.all(samples.removed_kinetic_energy_hartree >= -1e-15)


def test_kinetic_energy_bookkeeping_matches_mode_momenta():
    analysis = _model_water()
    samples = sample_wigner(
        WATER_MASSES,
        WATER_BOHR,
        analysis.frequencies_cm1,
        analysis.modes_mass_weighted,
        n_samples=50,
        temperature_K=300.0,
        low_frequency_cutoff_cm1=10.0,
        rng=np.random.default_rng(4),
    )
    before = 0.5 * (samples.p**2).sum(axis=1)
    np.testing.assert_allclose(
        samples.kinetic_energy_hartree + samples.removed_kinetic_energy_hartree, before, rtol=1e-10
    )


def test_low_frequency_and_imaginary_modes_are_not_sampled():
    analysis = _model_water()
    freqs = analysis.frequencies_cm1.copy()
    freqs[0] = -15.0  # a small imaginary mode kept by the frequencies stage
    samples = sample_wigner(
        WATER_MASSES,
        WATER_BOHR,
        freqs,
        analysis.modes_mass_weighted,
        n_samples=200,
        temperature_K=0.0,
        low_frequency_cutoff_cm1=10.0,
        rng=np.random.default_rng(4),
    )
    assert samples.sampled_modes.tolist() == [False, True, True]
    assert samples.q.shape == (200, 2)
    mode0 = analysis.modes_mass_weighted[:, 0]
    root = np.sqrt(np.repeat(WATER_MASSES * DALTON_IN_ME, 3))
    displacement = (samples.positions_bohr - WATER_BOHR).reshape(200, 9) * root
    assert np.abs(displacement @ mode0).max() < 1e-10  # nothing along the unsampled mode


def test_same_rng_same_samples():
    analysis = _model_water()
    kwargs = dict(n_samples=10, temperature_K=0.0, low_frequency_cutoff_cm1=10.0)
    a = sample_wigner(
        WATER_MASSES,
        WATER_BOHR,
        analysis.frequencies_cm1,
        analysis.modes_mass_weighted,
        rng=np.random.default_rng(9),
        **kwargs,
    )
    b = sample_wigner(
        WATER_MASSES,
        WATER_BOHR,
        analysis.frequencies_cm1,
        analysis.modes_mass_weighted,
        rng=np.random.default_rng(9),
        **kwargs,
    )
    np.testing.assert_array_equal(a.positions_bohr, b.positions_bohr)
    np.testing.assert_array_equal(a.velocities_au, b.velocities_au)


def test_linear_molecule_and_single_atom():
    co2 = np.array([[0, 0, 0], [0, 0, 2.2], [0, 0, -2.2]], dtype=float)
    masses = np.array([12.0, 15.994915, 15.994915])
    e = np.zeros(9)
    e[[5, 8]] = [1, -1]
    analysis = harmonic_analysis(masses, co2, 0.5 * np.outer(e, e) + 0.01 * np.eye(9))
    s = sample_wigner(
        masses,
        co2,
        analysis.frequencies_cm1,
        analysis.modes_mass_weighted,
        n_samples=50,
        temperature_K=0.0,
        low_frequency_cutoff_cm1=10.0,
        rng=np.random.default_rng(6),
    )
    assert np.all(np.isfinite(s.velocities_au))
    assert s.sampled_modes.sum() == 4  # 3N - 5
    m_me = masses * DALTON_IN_ME
    linear_p = np.linalg.norm((m_me[None, :, None] * s.velocities_au).sum(axis=1), axis=1)
    centre = (m_me[None, :, None] * s.positions_bohr).sum(axis=1, keepdims=True) / m_me.sum()
    ang = np.linalg.norm(
        (m_me[None, :, None] * np.cross(s.positions_bohr - centre, s.velocities_au)).sum(axis=1),
        axis=1,
    )
    assert linear_p.max() < 1e-10 and ang.max() < 1e-8
    assert np.all(s.removed_kinetic_energy_hartree >= -1e-14)
    atom = sample_wigner(
        np.array([4.002603]),
        np.zeros((1, 3)),
        np.zeros(0),
        np.zeros((3, 0)),
        n_samples=3,
        temperature_K=0.0,
        low_frequency_cutoff_cm1=10.0,
        rng=np.random.default_rng(7),
    )
    assert np.all(atom.positions_bohr == 0) and np.all(atom.velocities_au == 0)
    assert atom.expected_mean_energy_hartree == 0.0


def test_remove_rigid_motion_on_a_spinning_molecule():
    m = WATER_MASSES * DALTON_IN_ME
    omega = np.array([0.001, -0.002, 0.0005])
    v = np.cross(omega, WATER_BOHR) + np.array([1e-3, 0, 0])
    out = remove_rigid_motion(m, WATER_BOHR[None], v[None])
    assert np.abs(out).max() < 1e-12
