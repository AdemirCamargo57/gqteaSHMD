"""Test-only helpers shared by the spin-orbit block tests (not collected): the random spatial
matrix, the single-Cartesian-slot operator and the Wigner-Eckart checker. The expected ratio
tables are not here: each test file writes its own out (controller ruling)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def random_matrix(nao: int, seed: int) -> np.ndarray:
    """A random complex (nao, nao) matrix."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((nao, nao)) + 1j * rng.standard_normal((nao, nao))


def in_slot(matrix: np.ndarray, k: int) -> np.ndarray:
    """A (3, n, n) operator: `matrix` in Cartesian slot k (0, 1, 2 = x, y, z), zero elsewhere."""
    h = np.zeros((3, *matrix.shape), dtype=complex)
    h[k] = matrix
    return h


def check_wigner_eckart(
    st_by_slot: Mapping[int, np.ndarray],
    tt_by_slot: Mapping[int, np.ndarray],
    we_st: Mapping[int, np.ndarray],
    we_tt: Mapping[int, np.ndarray],
    bound: float,
) -> int:
    """Assert the we_st / we_tt ratio tables for every pair; return the number of pairs checked.

    st_by_slot[k] (n_s, n_t, 3) and tt_by_slot[k] (n_t, n_t, 3, 3) are the blocks of the operator
    with one spatial matrix in slot k. The independent element is z0 = st_by_slot[2][i, j, 1]
    (S-T) or tt_by_slot[2][j, j', 2, 2] (T-T); every component must equal z0 times the table entry
    within `bound` relative to |z0|.
    """
    checked = 0
    st_z = st_by_slot[2]
    for i in range(st_z.shape[0]):
        for j in range(st_z.shape[1]):
            z0 = st_z[i, j, 1]
            assert abs(z0) > 1e-8
            for k in range(3):
                expected = z0 * we_st[k]
                assert np.abs(st_by_slot[k][i, j] - expected).max() <= bound * abs(z0)
            checked += 1
    tt_z = tt_by_slot[2]
    for j in range(tt_z.shape[0]):
        for jp in range(tt_z.shape[1]):
            z0 = tt_z[j, jp, 2, 2]
            assert abs(z0) > 1e-8
            for k in range(3):
                expected = z0 * we_tt[k]
                assert np.abs(tt_by_slot[k][j, jp] - expected).max() <= bound * abs(z0)
            checked += 1
    return checked
