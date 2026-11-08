"""Harmonic Wigner sampling of initial conditions (independent normal modes).

For a harmonic mode of angular frequency omega (hartree, hbar = 1) the thermal Wigner function
is a Gaussian in the mass-weighted normal coordinate Q (sqrt(m_e) bohr) and momentum P:
    sigma_Q^2 = c / (2 omega),   sigma_P^2 = c omega / 2,   c = coth(omega / 2 k_B T)
(c = 1 at T = 0)
(Hillery et al., Phys. Rep. 106, 121 (1984)). Then <E> = (omega/2) c and Var(E) = <E>^2.
Cartesian displacements and velocities: dx = M^-1/2 L Q, v = M^-1/2 L P, M in electron masses.
Modes below the low-frequency cutoff (including imaginary ones) are not sampled. The total linear
and angular momentum of each sample is removed afterwards (velocities are not rescaled).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from gqteashmd.core.units import BOLTZMANN_HARTREE_PER_K, DALTON_IN_ME, HARTREE_IN_CM1


@dataclass(frozen=True, eq=False)
class WignerSamples:
    positions_bohr: np.ndarray
    velocities_au: np.ndarray
    q: np.ndarray
    p: np.ndarray
    harmonic_energy_hartree: np.ndarray
    kinetic_energy_hartree: np.ndarray
    removed_kinetic_energy_hartree: np.ndarray
    sampled_modes: np.ndarray
    expected_mean_energy_hartree: float
    expected_energy_std_hartree: float
    omega_au: np.ndarray  # angular frequencies of the sampled modes
    expected_mode_energy_hartree: np.ndarray  # <E_k> of the sampled modes

    def mode_energy_ratio_sums(self) -> np.ndarray:
        """S_k = sum over samples of E_k / <E_k>, per sampled mode.

        For a correct sampler E_k / <E_k> is Exp(1) and independent between modes and samples,
        so S_k follows Gamma(shape = n_samples, scale = 1) exactly, whatever T and omega.
        """
        energy = 0.5 * (self.p**2 + (self.omega_au * self.q) ** 2)
        return np.asarray((energy / self.expected_mode_energy_hartree).sum(axis=0))


def _occupation_factor(omega: np.ndarray, temperature_K: float) -> np.ndarray:
    if temperature_K <= 0.0:
        return np.ones_like(omega)
    return 1.0 / np.tanh(omega / (2.0 * BOLTZMANN_HARTREE_PER_K * temperature_K))


def mode_widths(omega_au: ArrayLike, temperature_K: float) -> tuple[np.ndarray, np.ndarray]:
    omega = np.asarray(omega_au, dtype=float)
    c = _occupation_factor(omega, temperature_K)
    return np.sqrt(c / (2.0 * omega)), np.sqrt(c * omega / 2.0)


def _kinetic(masses_me: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    return np.asarray(0.5 * (masses_me[None, :, None] * velocities**2).sum(axis=(1, 2)))


def remove_rigid_motion(
    masses_me: ArrayLike, positions_bohr: ArrayLike, velocities_au: ArrayLike
) -> np.ndarray:
    m = np.asarray(masses_me, dtype=float)
    x = np.asarray(positions_bohr, dtype=float)
    v = np.asarray(velocities_au, dtype=float)
    total = m.sum()
    v = v - ((m[None, :, None] * v).sum(axis=1) / total)[:, None, :]
    r = x - ((m[None, :, None] * x).sum(axis=1) / total)[:, None, :]
    angular = (m[None, :, None] * np.cross(r, v)).sum(axis=1)
    r2 = (r**2).sum(axis=2)
    inertia = (m[None, :] * r2).sum(axis=1)[:, None, None] * np.eye(3) - np.einsum(
        "a,nai,naj->nij", m, r, r
    )
    omega = np.einsum("nij,nj->ni", np.linalg.pinv(inertia, rcond=1e-10), angular)
    return np.asarray(v - np.cross(omega[:, None, :], r))


def sample_wigner(
    masses_amu: ArrayLike,
    coords_bohr: ArrayLike,
    frequencies_cm1: ArrayLike,
    modes_mass_weighted: ArrayLike,
    *,
    n_samples: int,
    temperature_K: float,
    low_frequency_cutoff_cm1: float,
    rng: np.random.Generator,
) -> WignerSamples:
    masses = np.asarray(masses_amu, dtype=float) * DALTON_IN_ME
    coords = np.asarray(coords_bohr, dtype=float)
    freqs = np.asarray(frequencies_cm1, dtype=float)
    modes = np.asarray(modes_mass_weighted, dtype=float)
    natom = len(masses)
    sampled = freqs >= low_frequency_cutoff_cm1
    omega = freqs[sampled] / HARTREE_IN_CM1
    vectors = modes[:, sampled]
    sigma_q, sigma_p = mode_widths(omega, temperature_K)
    q = rng.standard_normal((n_samples, omega.size)) * sigma_q
    p = rng.standard_normal((n_samples, omega.size)) * sigma_p
    # `modes_mass_weighted` from `harmonic_analysis` is orthonormal in amu-weighted space. A unit
    # vector in mass-weighted space is the same whether the masses are in amu or in electron
    # masses, because only the norm scales. So using `root` in electron masses is consistent with
    # omega in a.u.
    root = np.sqrt(np.repeat(masses, 3))
    positions = coords[None] + ((q @ vectors.T) / root).reshape(n_samples, natom, 3)
    raw_velocities = ((p @ vectors.T) / root).reshape(n_samples, natom, 3)
    velocities = remove_rigid_motion(masses, positions, raw_velocities)
    kinetic_before = _kinetic(masses, raw_velocities)
    kinetic_after = _kinetic(masses, velocities)
    mean_per_mode = 0.5 * omega * _occupation_factor(omega, temperature_K)
    return WignerSamples(
        positions_bohr=positions,
        velocities_au=velocities,
        q=q,
        p=p,
        harmonic_energy_hartree=0.5 * (p**2 + (omega * q) ** 2).sum(axis=1),
        kinetic_energy_hartree=kinetic_after,
        removed_kinetic_energy_hartree=kinetic_before - kinetic_after,
        sampled_modes=sampled,
        expected_mean_energy_hartree=float(mean_per_mode.sum()),
        expected_energy_std_hartree=float(np.sqrt((mean_per_mode**2).sum())),
        omega_au=omega,
        expected_mode_energy_hartree=mean_per_mode,
    )
