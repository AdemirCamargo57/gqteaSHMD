"""Diagonal (propagation) basis of the MCH Hamiltonian.

In P7a-1 the Hamiltonian is real-diagonal, so the diagonal basis equals the MCH basis.
"""

from __future__ import annotations

import numpy as np

SOC_MESSAGE = (
    "spin-orbit coupling / non-diagonal MCH Hamiltonians need P7a-2 (diagonal representation)"
)
OFF_DIAGONAL_TOLERANCE = 1e-10


def diagonal_basis(
    hamiltonian: np.ndarray, previous_u: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalise the Hermitian MCH Hamiltonian; return (E_diag, U) with H U = U diag(E).

    The columns of U are ordered by maximum overlap with ``previous_u`` (the identity when
    absent) and each column's phase is fixed so that its overlap with the reference is real
    and positive. ``NotImplementedError`` if U is not diagonal up to phases.
    """
    h = np.asarray(hamiltonian, dtype=complex)
    n = h.shape[0]
    ref = np.eye(n, dtype=complex) if previous_u is None else np.asarray(previous_u, complex)
    energies, u = np.linalg.eigh(h)
    overlap = ref.conj().T @ u  # [i, k] = <ref_i | U_k>
    order = _assign(np.abs(overlap))  # order[i] = column of U assigned to reference i
    if order != list(range(n)):
        energies = energies[order]
        u = u[:, order]
        overlap = ref.conj().T @ u
    diag = np.diag(overlap)
    phase = np.where(np.abs(diag) > 0, diag / np.where(np.abs(diag) > 0, np.abs(diag), 1), 1.0)
    u = u * phase.conj()[None, :]
    off = np.abs(u - np.diag(np.diag(u)))
    if off.size and np.max(off) > OFF_DIAGONAL_TOLERANCE:
        raise NotImplementedError(SOC_MESSAGE)
    return energies, u


def _assign(weights: np.ndarray) -> list[int]:
    """Greedy one-to-one assignment of columns to rows by largest weight (small n)."""
    n = weights.shape[0]
    w = weights.copy()
    order = [0] * n
    for _ in range(n):
        i, k = np.unravel_index(np.argmax(w), w.shape)
        order[int(i)] = int(k)
        w[i, :] = -1.0
        w[:, k] = -1.0
    return order
