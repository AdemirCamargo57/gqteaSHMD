"""Velocity-Verlet helpers."""

from __future__ import annotations

import numpy as np


def acceleration(gradient: np.ndarray, masses: np.ndarray) -> np.ndarray:
    a: np.ndarray = -np.asarray(gradient, dtype=float) / np.asarray(masses, dtype=float)
    return a


def verlet_positions(q: np.ndarray, v: np.ndarray, a: np.ndarray, dt: float) -> np.ndarray:
    out: np.ndarray = q + v * dt + 0.5 * a * dt * dt
    return out


def verlet_velocities(v: np.ndarray, a_old: np.ndarray, a_new: np.ndarray, dt: float) -> np.ndarray:
    out: np.ndarray = v + 0.5 * (a_old + a_new) * dt
    return out
