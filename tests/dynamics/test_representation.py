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


@pytest.mark.parametrize("diag", [[0.5, 0.1], [-0.1, 0.0, 0.3], [0.2, 0.2 + 1e-9, -0.4]])
def test_diagonal_fast_path_equals_the_general_path(diag):
    h = np.diag(diag).astype(complex)
    e_fast, u_fast = diagonal_basis(h, None)
    e_gen, u_gen = diagonal_basis(h, np.eye(len(diag)))  # identity reference: general path
    np.testing.assert_allclose(e_fast, e_gen, rtol=0, atol=1e-15)
    np.testing.assert_allclose(u_fast, u_gen, rtol=0, atol=1e-15)
    assert e_fast.dtype == float and u_fast.dtype == complex
