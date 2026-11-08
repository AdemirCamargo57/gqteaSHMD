import numpy as np
import pytest

from gqteashmd.dynamics.representation import diagonal_basis


def test_diagonal_hamiltonian_gives_identity():
    e, u = diagonal_basis(np.diag([-0.1, 0.0, 0.3]).astype(complex), None)
    np.testing.assert_allclose(e, [-0.1, 0.0, 0.3])
    np.testing.assert_allclose(u, np.eye(3), atol=1e-14)


def test_unsorted_diagonal_keeps_mch_order():
    e, u = diagonal_basis(np.diag([0.5, 0.1]).astype(complex), None)
    np.testing.assert_allclose(e, [0.5, 0.1])
    np.testing.assert_allclose(u, np.eye(2), atol=1e-14)


def test_phase_alignment_with_previous():
    prev = np.diag([1j, 1.0])
    e, u = diagonal_basis(np.diag([0.0, 0.2]).astype(complex), prev)
    np.testing.assert_allclose(np.diag(prev.conj().T @ u).imag, 0, atol=1e-14)
    assert np.all(np.diag(prev.conj().T @ u).real > 0)


def test_soc_guard():
    h = np.array([[0.0, 0.01], [0.01, 0.1]], dtype=complex)
    with pytest.raises(NotImplementedError, match="P7a-2"):
        diagonal_basis(h, None)
