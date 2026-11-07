"""Derived electronic properties (atomic units)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def oscillator_strength(excitation_energy: float, transition_dipole: ArrayLike) -> float:
    """Length-gauge oscillator strength f = (2/3) ΔE |μ|² (ΔE in hartree, μ in e·a0)."""
    mu = np.asarray(transition_dipole, dtype=float)
    return float(2.0 / 3.0 * excitation_energy * float(np.dot(mu, mu)))
