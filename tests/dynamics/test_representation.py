"""Diagonal representation: U tracking, gradient tensor, selection (method reference 4-6)."""

import numpy as np
import pytest

from gqteashmd.dynamics.records import ElectronicSnapshot
from gqteashmd.dynamics.representation import (
    DEGENERACY_BLOCK_TOLERANCE,
    diagonal_basis,
    diagonal_energies,
    diagonal_gradient,
    diagonal_nac,
    diagonal_tensor,
    gradient_tensor,
    masked_snapshot,
    mch_coefficients,
    mch_state,
    selection_mask,
)


def _hermitian(rng, n, scale=1.0):
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return scale * (a + a.conj().T) / 2


def _unitary(rng, n):
    q, r = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    return q * (np.diag(r) / np.abs(np.diag(r)))[None, :]


# ---- identity case (R1) ----


def test_diagonal_hamiltonian_gives_none():
    h = np.diag([-0.1, 0.0, 0.3]).astype(complex)
    e, u = diagonal_basis(h, None)
    assert u is None
    np.testing.assert_array_equal(e, [-0.1, 0.0, 0.3])
    assert e.dtype == float


def test_unsorted_diagonal_keeps_mch_order():
    e, u = diagonal_basis(np.diag([0.5, 0.1]).astype(complex), None)
    assert u is None
    np.testing.assert_array_equal(e, [0.5, 0.1])


def test_diagonal_hamiltonian_first_step_gives_none():
    e, u = diagonal_basis(np.diag([0.5, 0.1, 0.1]).astype(complex), None, first_step=True)
    assert u is None
    np.testing.assert_array_equal(e, [0.5, 0.1, 0.1])


def test_identity_case_is_the_real_part_of_the_diagonal_bit_for_bit():
    d = np.array([0.123456789012345, -1.0 / 3.0, 2.0 / 7.0])
    h = np.diag(d).astype(complex)
    e, u = diagonal_basis(h, None)
    assert u is None
    assert e.tobytes() == d.tobytes()
    assert diagonal_energies(h, None).tobytes() == d.tobytes()


def test_diagonal_h_with_a_previous_u_is_energy_ordered():
    # R1: with a previous U the general path is taken and the states are energy-ordered
    e, u = diagonal_basis(np.diag([0.5, 0.1]).astype(complex), np.eye(2, dtype=complex))
    assert u is not None
    np.testing.assert_allclose(e, [0.1, 0.5], rtol=0, atol=1e-15)


@pytest.mark.parametrize("diag", [[-0.1, 0.0, 0.3], [-0.4, 0.2, 0.2 + 1e-9]])
def test_sorted_diagonal_identity_reference_equals_the_identity_case(diag):
    h = np.diag(diag).astype(complex)
    e_none, u_none = diagonal_basis(h, None)
    e_gen, u_gen = diagonal_basis(h, np.eye(len(diag), dtype=complex))
    assert u_none is None
    np.testing.assert_allclose(e_gen, e_none, rtol=0, atol=1e-15)
    np.testing.assert_allclose(u_gen, np.eye(len(diag)), rtol=0, atol=1e-15)
    assert e_gen.dtype == float and u_gen.dtype == complex


def test_phase_alignment_with_previous():
    prev = np.diag([1j, 1.0])
    e, u = diagonal_basis(np.diag([0.0, 0.2]).astype(complex), prev)
    np.testing.assert_allclose(np.diag(prev.conj().T @ u).imag, 0, atol=1e-14)
    assert np.all(np.diag(prev.conj().T @ u).real > 0)


def test_off_diagonal_hamiltonian_is_no_longer_refused():
    # the SOC guard moved to the driver (Task 2, ruling B)
    h = np.array([[0.0, 0.01], [0.01, 0.1]], dtype=complex)
    e, u = diagonal_basis(h, None)
    assert u is not None and e.shape == (2,)


# ---- general case ----


def test_complex_2x2():
    h = np.array([[0.1, 0.02 - 0.03j], [0.02 + 0.03j, -0.05]])
    e, u = diagonal_basis(h, None)
    assert u is not None
    assert e[0] < e[1]
    np.testing.assert_allclose(u.conj().T @ u, np.eye(2), rtol=0, atol=1e-13)
    np.testing.assert_allclose(u @ np.diag(e) @ u.conj().T, h, rtol=0, atol=1e-13)
    np.testing.assert_allclose(e, np.linalg.eigvalsh(h), rtol=0, atol=1e-15)


def test_identity_reference_without_first_step():
    # previous_u None and first_step False: the identity is the reference
    rng = np.random.default_rng(11)
    h = np.diag([0.0, 0.3, 0.7]) + _hermitian(rng, 3, 0.01)
    _, u = diagonal_basis(h, None)
    d = np.diag(u)
    np.testing.assert_allclose(d.imag, 0, atol=1e-15)
    assert np.all(d.real > 0.9)


def test_first_step_makes_the_largest_component_real_and_positive():
    rng = np.random.default_rng(5)
    h = _hermitian(rng, 5)
    e, u = diagonal_basis(h, None, first_step=True)
    assert u is not None
    np.testing.assert_allclose(u.conj().T @ u, np.eye(5), rtol=0, atol=1e-13)
    np.testing.assert_allclose(u @ np.diag(e) @ u.conj().T, h, rtol=0, atol=1e-13)
    for a in range(5):
        k = int(np.argmax(np.abs(u[:, a])))
        assert abs(u[k, a].imag) <= 1e-15
        assert u[k, a].real > 0


def test_first_step_with_a_previous_u_is_an_error():
    with pytest.raises(ValueError, match="first_step"):
        diagonal_basis(np.eye(2, dtype=complex), np.eye(2, dtype=complex), first_step=True)


def test_phase_continuity_along_a_smooth_path():
    rng = np.random.default_rng(2)
    h0 = np.diag([0.0, 0.3, 0.6, 0.9]) + _hermitian(rng, 4, 0.02)
    h1 = _hermitian(rng, 4, 0.02)
    s_values = np.linspace(0.0, 1.0, 200)
    _, u_prev = diagonal_basis(h0, None, first_step=True)
    worst = 0.0
    for s in s_values[1:]:
        h = h0 + s * h1
        e, u = diagonal_basis(h, u_prev)
        assert np.all(np.diff(e) > 1e-3)  # non-degenerate path
        overlaps = np.einsum("ia,ia->a", u_prev.conj(), u)
        worst = max(worst, float(np.max(np.abs(overlaps - 1.0))))
        assert np.all(overlaps.real > 0.99)  # no column flips
        u_prev = u
    assert worst < 1e-6, worst


def test_degenerate_block_follows_the_previous_subspace_basis():
    rng = np.random.default_rng(7)
    h = np.diag([1.0, 1.0, 1.0, 2.0]).astype(complex)
    h[0, 3] = 1e-13j  # tiny perturbation: non-diagonal H, no splitting of the block
    h[3, 0] = -1e-13j
    u_prev = np.zeros((4, 4), dtype=complex)
    u_prev[:3, :3] = _unitary(rng, 3)
    u_prev[3, 3] = 1.0
    e, u = diagonal_basis(h, u_prev)
    assert np.max(e[:3]) - np.min(e[:3]) <= DEGENERACY_BLOCK_TOLERANCE
    np.testing.assert_allclose(u[:, :3], u_prev[:, :3], rtol=0, atol=1e-12)
    np.testing.assert_allclose(u[:, 3], u_prev[:, 3], rtol=0, atol=1e-12)
    np.testing.assert_allclose(u.conj().T @ u, np.eye(4), rtol=0, atol=1e-13)
    # a pure phase rule on the raw eigenvectors cannot reach U_prev inside the block
    _, raw = np.linalg.eigh(h)
    d = np.einsum("ia,ia->a", u_prev.conj(), raw)
    phased = raw * np.where(np.abs(d) > 0, d / np.where(np.abs(d) > 0, np.abs(d), 1), 1)
    assert np.max(np.abs(phased[:, :3] - u_prev[:, :3])) > 1e-2


def _triplet_path_hamiltonian(s, sigma_tt):
    """1 singlet + 1 triplet (canonical order S0, T1(-1), T1(0), T1(+1))."""
    e_s = -0.1 + 0.02 * s
    e_t = 0.2 - 0.05 * s
    h = np.diag([e_s, e_t, e_t, e_t]).astype(complex)
    h[1:, 1:] += (1.0 - 2.0 * s) ** 2 * sigma_tt
    return h


def test_triplet_block_keeps_continuity_through_merge_and_resplitting():
    # triplet-triplet SOC (1 - 2s)^2 Sigma_TT: split, exactly degenerate at s = 1/2, split again
    rng = np.random.default_rng(13)
    sigma_tt = _hermitian(rng, 3, 0.01)
    s_values = np.linspace(0.0, 1.0, 201)
    _, u_prev = diagonal_basis(_triplet_path_hamiltonian(0.0, sigma_tt), None, first_step=True)
    assert u_prev is not None
    worst = 0.0
    merged = False
    for s in s_values[1:]:
        h = _triplet_path_hamiltonian(s, sigma_tt)
        e, u = diagonal_basis(h, u_prev)
        merged = merged or np.max(e[1:]) - np.min(e[1:]) <= DEGENERACY_BLOCK_TOLERANCE
        np.testing.assert_allclose(u @ np.diag(e) @ u.conj().T, h, rtol=0, atol=1e-13)
        worst = max(worst, float(np.max(np.abs(u - u_prev))))
        u_prev = u
    assert merged  # the path passes the exact triplet degeneracy
    assert worst < 1e-6, worst


def test_triplet_block_split_by_singlet_triplet_soc_keeps_continuity():
    # S-T SOC splits the threefold triplet into 1 + 2; the 2-fold block stays exactly degenerate
    rng = np.random.default_rng(17)
    v = rng.normal(size=3) + 1j * rng.normal(size=3)
    v *= 0.01 / np.linalg.norm(v)
    worst_column = 0.0
    worst_block = 0.0

    def hamiltonian(s):
        e_s = -0.1 + 0.05 * s
        e_t = 0.2 - 0.05 * s
        h = np.diag([e_s, e_t, e_t, e_t]).astype(complex)
        h[0, 1:] = v
        h[1:, 0] = v.conj()
        return h

    _, u_prev = diagonal_basis(hamiltonian(0.0), None, first_step=True)
    for s in np.linspace(0.0, 1.0, 200)[1:]:
        h = hamiltonian(s)
        e, u = diagonal_basis(h, u_prev)
        # ascending: S-like, the degenerate pair, the T-like state mixed with S
        assert abs(e[1] - e[2]) <= DEGENERACY_BLOCK_TOLERANCE
        np.testing.assert_allclose(u @ np.diag(e) @ u.conj().T, h, rtol=0, atol=1e-13)
        overlaps = np.einsum("ia,ia->a", u_prev.conj(), u)
        assert np.all(overlaps.real > 0.99)
        worst_column = max(worst_column, float(np.max(np.abs(overlaps[[0, 3]] - 1.0))))
        worst_block = max(worst_block, float(np.max(np.abs(u[:, 1:3] - u_prev[:, 1:3]))))
        u_prev = u
    assert worst_column < 1e-6, worst_column
    assert worst_block < 1e-6, worst_block


def test_diagonal_energies():
    rng = np.random.default_rng(3)
    h = _hermitian(rng, 4)
    u = _unitary(rng, 4)
    np.testing.assert_allclose(
        diagonal_energies(h, u), np.real(np.diag(u.conj().T @ h @ u)), rtol=0, atol=1e-15
    )
    e, u_basis = diagonal_basis(h, None)
    assert e.tobytes() == diagonal_energies(h, u_basis).tobytes()


# ---- gradient tensor (item 5) ----


def _snapshot(n=3, f=2, seed=0):
    rng = np.random.default_rng(seed)
    e = np.sort(rng.normal(size=n))
    g = rng.normal(size=(n, f))
    k = rng.normal(size=(n, n, f))
    k = (k - k.transpose(1, 0, 2)) / 2
    return ElectronicSnapshot(
        energies=e, gradients=g, nacs=k, hamiltonian=np.diag(e).astype(complex)
    )


def test_gradient_tensor_structure():
    snap = _snapshot()
    g = gradient_tensor(snap)
    assert g.shape == (3, 3, 2) and g.dtype == float
    np.testing.assert_array_equal(g, g.transpose(1, 0, 2))
    for a in range(3):
        np.testing.assert_array_equal(g[a, a], snap.gradients[a])
        for b in range(3):
            if a != b:
                expected = (snap.energies[b] - snap.energies[a]) * snap.nacs[a, b]
                np.testing.assert_array_equal(g[a, b], expected)


def test_gradient_tensor_none_zeroes_the_off_diagonal():
    snap = _snapshot()
    g = gradient_tensor(snap, "none")
    for a in range(3):
        np.testing.assert_array_equal(g[a, a], snap.gradients[a])
        for b in range(3):
            if a != b:
                np.testing.assert_array_equal(g[a, b], 0.0)


def test_gradient_tensor_rejects_an_unknown_transformation():
    with pytest.raises(ValueError, match="gradient_transformation"):
        gradient_tensor(_snapshot(), "bogus")


def test_diagonal_tensor_identity_and_trace_invariance():
    g = gradient_tensor(_snapshot())
    assert diagonal_tensor(g, None) is g
    u = _unitary(np.random.default_rng(1), 3)
    gd = diagonal_tensor(g, u)
    assert gd.shape == g.shape
    for i in range(g.shape[2]):
        np.testing.assert_allclose(np.trace(gd[:, :, i]), np.trace(g[:, :, i]), rtol=0, atol=1e-13)
        np.testing.assert_allclose(gd[:, :, i], u.conj().T @ g[:, :, i] @ u, rtol=0, atol=1e-14)


def test_diagonal_gradient_with_identity_is_the_mch_gradient():
    snap = _snapshot()
    g = gradient_tensor(snap)
    for a in range(3):
        assert diagonal_gradient(diagonal_tensor(g, None), a).tobytes() == (
            snap.gradients[a].tobytes()
        )


# ---- Hellmann-Feynman checks (ruling C) ----


def _symmetric(rng, n, scale):
    a = rng.normal(size=(n, n))
    return scale * (a + a.T)


_HF_RNG = np.random.default_rng(23)
_A = np.diag([0.0, 0.25, 0.6]) + _symmetric(_HF_RNG, 3, 0.03)
_B = _symmetric(_HF_RNG, 3, 0.05)
_SIGMA = _hermitian(_HF_RNG, 3, 0.01)


def _mch(q, w_previous=None):
    """Spin-free MCH data of V(q) = A + q B with sign continuity; H_MCH = diag(E) + W^T Sigma W."""
    e, w = np.linalg.eigh(_A + q * _B)
    if w_previous is None:
        k = np.argmax(np.abs(w), axis=0)
        signs = np.sign(w[k, np.arange(3)])
    else:
        signs = np.sign(np.einsum("ia,ia->a", w_previous, w))
    w = w * signs[None, :]
    b_mch = w.T @ _B @ w
    gaps = e[None, :] - e[:, None]
    nacs = np.zeros((3, 3, 1))
    off = ~np.eye(3, dtype=bool)
    nacs[off, 0] = b_mch[off] / gaps[off]
    snap = ElectronicSnapshot(
        energies=e,
        gradients=np.diag(b_mch)[:, None].copy(),
        nacs=nacs,
        hamiltonian=np.diag(e) + w.T @ _SIGMA @ w,
    )
    return snap, w


Q0 = 0.3
H_FD = 1e-5


def test_hellmann_feynman_diagonal_gradient():
    snap, _ = _mch(Q0)
    e_diag, u = diagonal_basis(snap.hamiltonian, None, first_step=True)
    g_diag = diagonal_tensor(gradient_tensor(snap), u)
    e_plus, _ = diagonal_basis(_mch(Q0 + H_FD)[0].hamiltonian, None, first_step=True)
    e_minus, _ = diagonal_basis(_mch(Q0 - H_FD)[0].hamiltonian, None, first_step=True)
    fd = (e_plus - e_minus) / (2 * H_FD)
    analytic = np.array([diagonal_gradient(g_diag, a)[0] for a in range(3)])
    np.testing.assert_allclose(analytic, fd, rtol=0, atol=1e-7)


def test_diagonal_nac_against_finite_differences_of_the_tracked_basis():
    snap, w0 = _mch(Q0)
    e_diag, u0 = diagonal_basis(snap.hamiltonian, None, first_step=True)
    g_diag = diagonal_tensor(gradient_tensor(snap), u0)

    def phi(q):
        # the diagonal states in the fixed diabatic basis, tracked against q0
        s, w = _mch(q, w0)
        _, u = diagonal_basis(s.hamiltonian, u0)
        return w @ u

    phi0 = w0 @ u0
    dphi = (phi(Q0 + H_FD) - phi(Q0 - H_FD)) / (2 * H_FD)
    fd = phi0.conj().T @ dphi  # <a|d b>
    for a in range(3):
        for b in range(3):
            if a != b:
                k = diagonal_nac(g_diag, e_diag, a, b)
                assert k.shape == (1,)
                np.testing.assert_allclose(k[0], fd[a, b], rtol=0, atol=1e-6)


def test_diagonal_nac_degenerate_pair_is_zero():
    g = np.ones((2, 2, 3), dtype=complex)
    e = np.array([0.1, 0.1 + 1e-13])
    np.testing.assert_array_equal(diagonal_nac(g, e, 0, 1), np.zeros(3))
    np.testing.assert_array_equal(diagonal_nac(g, e, 0, 0), np.zeros(3))
    e2 = np.array([0.1, 0.3])
    np.testing.assert_allclose(diagonal_nac(g, e2, 0, 1), np.ones(3) / 0.2, rtol=1e-15)


# ---- energy-based selection (item 6) ----


def test_selection_mask_is_strict():
    e = np.array([-0.5, 0.0, 0.25, 0.5])
    mask = selection_mask(e, 0.0, 0.5)
    assert mask is not None and mask.dtype == bool
    np.testing.assert_array_equal(mask, [False, True, True, False])
    assert selection_mask(e, 0.0, None) is None


def test_masked_snapshot():
    snap = _snapshot(n=4, f=2, seed=4)
    before_g = snap.gradients.copy()
    before_k = snap.nacs.copy()
    mask = np.array([True, False, True, False])
    masked = masked_snapshot(snap, mask)
    assert isinstance(masked, ElectronicSnapshot) and masked is not snap
    np.testing.assert_array_equal(snap.gradients, before_g)
    np.testing.assert_array_equal(snap.nacs, before_k)
    for a in range(4):
        expected = before_g[a] if mask[a] else 0.0
        np.testing.assert_array_equal(masked.gradients[a], expected)
        for b in range(4):
            expected = before_k[a, b] if mask[a] and mask[b] else 0.0
            np.testing.assert_array_equal(masked.nacs[a, b], expected)
    assert masked.energies is snap.energies and masked.hamiltonian is snap.hamiltonian
    assert masked_snapshot(snap, None) is snap
    assert masked_snapshot(snap, np.ones(4, dtype=bool)) is snap


# ---- MCH quantities of the diagonal state ----


def test_mch_state():
    assert mch_state(None, 2) == 2
    u = np.array([[0.6, 1.0], [0.8, 0.0]], dtype=complex)
    assert mch_state(u, 0) == 1
    r = 1 / np.sqrt(2)
    tie = np.array([[r, r], [r, -r]], dtype=complex)
    assert mch_state(tie, 0) == 0 and mch_state(tie, 1) == 0


def test_mch_coefficients():
    c = np.array([0.6, 0.8j])
    assert mch_coefficients(None, c) is c
    u = _unitary(np.random.default_rng(9), 2)
    np.testing.assert_allclose(mch_coefficients(u, c), u @ c, rtol=0, atol=1e-15)
