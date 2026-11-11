"""Classical and quantum populations with 95 % intervals (P8 spec section 4)."""

from __future__ import annotations

import numpy as np

Z_95 = 1.959963984540054


def classical_populations(active: np.ndarray, n_states: int) -> tuple[np.ndarray, np.ndarray]:
    n = active.shape[0]
    counts = np.stack([(active == j).sum(axis=0) for j in range(n_states)], axis=-1)
    return counts / n, counts


def wilson_intervals(counts: np.ndarray, n: int, z: float = Z_95) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(counts, dtype=float) / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return np.maximum(0.0, centre - half), np.minimum(1.0, centre + half)


def quantum_populations(populations: np.ndarray) -> np.ndarray:
    result: np.ndarray = populations.mean(axis=0)
    return result


def bootstrap_indices(rng: np.random.Generator, n: int, b: int) -> np.ndarray:
    return rng.integers(0, n, size=(b, n))


def bootstrap_intervals(
    per_trajectory: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    means = np.stack([per_trajectory[idx].mean(axis=0) for idx in indices])
    lo, hi = np.percentile(means, [2.5, 97.5], axis=0)
    return lo, hi


def consistency(classical: np.ndarray, quantum: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.abs(classical - quantum).max(axis=0)
    return result
