"""Nuclear-ensemble absorption spectrum (Crespo-Otero and Barbatti, Theor. Chem. Acc. 131,
1237 (2012)).

For n samples i of the ground-state nuclear distribution and excited states n, with vertical
excitation energies ``dE_in`` (eV) and oscillator strengths ``f_in``, the photoabsorption cross
section per molecule at photon energy E (eV) is::

    sigma_n(E) = PREFACTOR / E * (1/n_ok) * sum_i dE_in * f_in * g(E - dE_in)
    sigma(E)   = sum_n sigma_n(E)

where ``g`` is the normalized Gaussian of standard deviation ``s = FWHM / (2 sqrt(2 ln 2))``
and ``n_ok`` is the number of samples. ``PREFACTOR = 2 pi^2 / c * E_h[eV] * a0[A]^2`` (c in
atomic units) equals ``pi e^2 hbar / (2 m_e c eps0)`` in eV A^2, about 1.09761. The molar
absorption coefficient is ``epsilon = sigma * EPSILON_PER_A2`` in M^-1 cm^-1, with
``EPSILON_PER_A2 = 1e-16 N_A / (1000 ln 10)`` (A^2 -> cm^2, then /1000 L, then log10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gqteashmd.core.units import AVOGADRO, BOHR_IN_ANGSTROM, HARTREE_IN_EV, SPEED_OF_LIGHT_AU

CROSS_SECTION_PREFACTOR_EV_A2 = (
    2 * math.pi**2 / SPEED_OF_LIGHT_AU * HARTREE_IN_EV * BOHR_IN_ANGSTROM**2
)
EPSILON_PER_A2 = 1e-16 * AVOGADRO / (1000 * math.log(10))

_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


@dataclass(frozen=True)
class Spectrum:
    energies_eV: NDArray[np.float64]  # (m,)
    cross_section_A2: NDArray[np.float64]  # (m,) total
    cross_section_by_state_A2: NDArray[np.float64]  # (m, n_exc)
    epsilon_M1cm1: NDArray[np.float64]  # (m,)


def broadened_spectrum(
    excitation_energies_eV: NDArray[np.float64],
    oscillator_strengths: NDArray[np.float64],
    *,
    fwhm_eV: float,
    step_eV: float,
) -> Spectrum:
    """Gaussian-broadened ensemble spectrum; inputs have shape (n_ok, n_exc), no NaN rows.

    The grid is E_j = e0 + j*step with e0 = max(step, step*floor((min dE - 4 s)/step)), running
    to the first point >= max dE + 4 s.
    """
    de = np.asarray(excitation_energies_eV, dtype=float)
    f = np.asarray(oscillator_strengths, dtype=float)
    if de.ndim != 2 or de.shape != f.shape:
        raise ValueError(
            "energies and oscillator strengths must be arrays of equal shape (n, n_exc)"
        )
    n_ok = de.shape[0]
    if n_ok == 0 or de.shape[1] == 0:
        raise ValueError("no successful samples: cannot build a spectrum")
    s = fwhm_eV * _FWHM_TO_SIGMA
    e0 = max(step_eV, step_eV * math.floor((float(de.min()) - 4 * s) / step_eV))
    n_points = max(1, math.ceil((float(de.max()) + 4 * s - e0) / step_eV - 1e-9) + 1)
    energies = np.asarray(e0 + step_eV * np.arange(n_points), dtype=np.float64)
    by_state: NDArray[np.float64] = np.zeros((n_points, de.shape[1]))
    norm = 1.0 / (s * math.sqrt(2 * math.pi))
    for i in range(n_ok):
        gauss = norm * np.exp(-0.5 * ((energies[:, None] - de[i][None, :]) / s) ** 2)
        by_state += gauss * (de[i] * f[i])[None, :]
    by_state *= CROSS_SECTION_PREFACTOR_EV_A2 / (n_ok * energies[:, None])
    total: NDArray[np.float64] = by_state.sum(axis=1)
    epsilon: NDArray[np.float64] = total * EPSILON_PER_A2
    return Spectrum(energies, total, by_state, epsilon)
