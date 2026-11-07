import math

import numpy as np
import pytest

from gqteashmd.overlaps import determinants
from gqteashmd.overlaps.determinants import DeterminantExpansion, state_overlaps, truncate


def _rotation(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def test_single_determinant_overlap_is_the_determinant():
    rng = np.random.default_rng(1)
    s_mo = rng.normal(size=(5, 5))
    bra = DeterminantExpansion([[0, 2, 3]], [[0, 1]], [[1.0]])
    ket = DeterminantExpansion([[1, 2, 4]], [[0, 3]], [[1.0]])
    expected = np.linalg.det(s_mo[np.ix_([0, 2, 3], [1, 2, 4])]) * np.linalg.det(
        s_mo[np.ix_([0, 1], [0, 3])]
    )
    assert state_overlaps(bra, ket, s_mo)[0, 0] == pytest.approx(expected, rel=1e-12)


def test_rotated_orbital():
    wf = DeterminantExpansion([[0]], np.zeros((1, 0), dtype=int), [[1.0]])
    assert state_overlaps(wf, wf, _rotation(0.3))[0, 0] == pytest.approx(math.cos(0.3))


def test_orbital_order_defines_the_sign():
    a = DeterminantExpansion([[0, 1]], [[0]], [[1.0]])
    b = DeterminantExpansion([[1, 0]], [[0]], [[1.0]])
    assert state_overlaps(a, b, np.eye(3))[0, 0] == pytest.approx(-1.0)


def test_orthonormal_states_at_identity_give_identity():
    rng = np.random.default_rng(2)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    alpha = [[0, 1], [0, 2], [1, 2], [0, 3]]
    beta = [[0, 1]] * 4
    wf = DeterminantExpansion(alpha, beta, q[:3])
    np.testing.assert_allclose(state_overlaps(wf, wf, np.eye(4)), np.eye(3), atol=1e-12)


def test_matches_brute_force_with_mixed_spin_excitations():
    rng = np.random.default_rng(3)
    s_mo = np.eye(4) + 0.1 * rng.normal(size=(4, 4))
    alpha = [[0], [2], [0], [3]]
    beta = [[1], [1], [2], [1]]
    bra = DeterminantExpansion(alpha, beta, rng.normal(size=(2, 4)))
    ket = DeterminantExpansion(alpha[::-1], beta[::-1], rng.normal(size=(3, 4)))
    brute = np.zeros((2, 3))
    for s in range(2):
        for t in range(3):
            for k in range(4):
                for m in range(4):
                    da = s_mo[bra.alpha[k][0], ket.alpha[m][0]]
                    db = s_mo[bra.beta[k][0], ket.beta[m][0]]
                    brute[s, t] += bra.coeffs[s, k] * ket.coeffs[t, m] * da * db
    np.testing.assert_allclose(state_overlaps(bra, ket, s_mo), brute, rtol=1e-12, atol=1e-14)


def test_electron_counts_must_match():
    a = DeterminantExpansion([[0, 1]], [[0]], [[1.0]])
    b = DeterminantExpansion([[0]], [[0]], [[1.0]])
    with pytest.raises(ValueError, match="electron"):
        state_overlaps(a, b, np.eye(3))
    c = DeterminantExpansion([[0, 1]], [[0, 1]], [[1.0]])
    with pytest.raises(ValueError, match="electron"):
        state_overlaps(a, c, np.eye(3))


def test_inconsistent_shapes_are_rejected():
    with pytest.raises(ValueError, match="determinant counts"):
        DeterminantExpansion([[0], [1]], [[0]], [[1.0, 0.0]])


def test_with_signs_flips_whole_states():
    wf = DeterminantExpansion([[0], [1]], [[0], [0]], [[1.0, 0.0], [0.0, 1.0]])
    flipped = wf.with_signs(np.array([1.0, -1.0]))
    np.testing.assert_array_equal(flipped.coeffs, [[1.0, 0.0], [0.0, -1.0]])


@pytest.mark.parametrize("tiny_blocks", [False, True])
def test_multi_electron_expansions_match_double_sum(monkeypatch, tiny_blocks):
    if tiny_blocks:
        monkeypatch.setattr(determinants, "_MAX_BLOCK_ELEMENTS", 1)
    rng = np.random.default_rng(7)
    s_mo = rng.normal(size=(6, 6))  # non-symmetric
    bra_a = [[0, 2, 4], [1, 3, 5], [0, 2, 4], [2, 0, 4], [1, 2, 3]]
    bra_b = [[0, 1], [2, 3], [0, 1], [1, 0], [4, 5]]
    ket_a = [[1, 2, 5], [0, 2, 4], [1, 2, 5], [3, 4, 0], [5, 1, 2], [0, 1, 2]]
    ket_b = [[3, 4], [0, 1], [3, 4], [5, 2], [3, 4], [0, 5]]
    bra = DeterminantExpansion(bra_a, bra_b, rng.normal(size=(3, 5)))
    ket = DeterminantExpansion(ket_a, ket_b, rng.normal(size=(2, 6)))
    expected = np.zeros((3, 2))
    for k in range(5):
        for m in range(6):
            d = np.linalg.det(s_mo[np.ix_(bra_a[k], ket_a[m])]) * np.linalg.det(
                s_mo[np.ix_(bra_b[k], ket_b[m])]
            )
            expected += np.outer(bra.coeffs[:, k], ket.coeffs[:, m]) * d
    np.testing.assert_allclose(state_overlaps(bra, ket, s_mo), expected, rtol=1e-12, atol=1e-12)


def test_truncate_keeps_the_dominant_determinants_and_renormalises():
    coeffs = np.array([[0.9, 0.4, 0.1, 0.0], [0.0, 0.1, 0.995, 0.0]])
    wf = DeterminantExpansion([[0], [1], [2], [3]], [[0]] * 4, coeffs)
    kept = truncate(wf, 0.95)
    assert kept.n_dets == 3  # determinant 3 is used by no state
    np.testing.assert_allclose(np.linalg.norm(kept.coeffs, axis=1), 1.0)
    # state 0's own cut (0.95 of 0.98) keeps determinants 0 and 1; determinant 2 is important for
    # state 1, so state 0 keeps its 0.1 coefficient there too (cross-state coefficients are kept)
    norm0 = math.sqrt(0.9**2 + 0.4**2 + 0.1**2)
    np.testing.assert_allclose(kept.coeffs[0], [0.9 / norm0, 0.4 / norm0, 0.1 / norm0])
    full = truncate(wf, 1.0)
    assert full.n_dets == 3
    with pytest.raises(ValueError, match="norm_kept"):
        truncate(wf, 0.0)
    with pytest.raises(ValueError, match="norm_kept"):
        truncate(wf, 1.5)


def test_truncate_rejects_zero_and_non_finite_states():
    zero = DeterminantExpansion([[0], [1]], [[0], [0]], [[1.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="zero coefficients"):
        truncate(zero, 0.9)
    bad = DeterminantExpansion([[0], [1]], [[0], [0]], [[1.0, np.nan]])
    with pytest.raises(ValueError, match="finite"):
        truncate(bad, 0.9)


def test_truncate_keeps_tiny_cross_state_coefficients():
    eps = 1e-6
    coeffs = np.array([[1.0, eps, 0.0], [0.0, 1.0, eps]])
    wf = DeterminantExpansion([[0], [1], [2]], [[0]] * 3, coeffs)
    kept = truncate(wf, 0.9999)  # determinant 2 is dropped; state 0 keeps its eps on determinant 1
    s = state_overlaps(kept, kept, np.eye(3))
    assert s[0, 1] == pytest.approx(eps / math.sqrt(1 + eps**2), rel=1e-12)


def test_beta_mo_overlap_defaults_to_the_alpha_one():
    rng = np.random.default_rng(11)
    s_mo = rng.normal(size=(5, 5))
    bra = DeterminantExpansion([[0, 1], [0, 2]], [[0], [1]], rng.normal(size=(2, 2)))
    ket = DeterminantExpansion([[1, 2], [0, 3]], [[2], [3]], rng.normal(size=(2, 2)))
    np.testing.assert_array_equal(
        state_overlaps(bra, ket, s_mo, s_mo_beta=s_mo), state_overlaps(bra, ket, s_mo)
    )


def test_beta_mo_overlap_changes_only_the_beta_determinants():
    rng = np.random.default_rng(12)
    s_alpha = rng.normal(size=(5, 5))
    s_beta = rng.normal(size=(5, 5))
    alpha_b, beta_b = [[0, 1], [0, 2], [1, 2]], [[0], [1], [3]]
    alpha_k, beta_k = [[1, 2], [0, 3], [0, 1]], [[2], [3], [0]]
    bra = DeterminantExpansion(alpha_b, beta_b, rng.normal(size=(2, 3)))
    ket = DeterminantExpansion(alpha_k, beta_k, rng.normal(size=(3, 3)))
    expected = np.zeros((2, 3))
    for k in range(3):
        for m in range(3):
            d = np.linalg.det(s_alpha[np.ix_(alpha_b[k], alpha_k[m])]) * np.linalg.det(
                s_beta[np.ix_(beta_b[k], beta_k[m])]
            )
            expected += np.outer(bra.coeffs[:, k], ket.coeffs[:, m]) * d
    got = state_overlaps(bra, ket, s_alpha, s_mo_beta=s_beta)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
    assert not np.allclose(got, state_overlaps(bra, ket, s_alpha))


def test_truncate_at_full_norm_keeps_every_nonzero_coefficient():
    # the squared 1e-200 underflows to 0.0; the coefficient itself is nonzero and must survive
    coeffs = np.array([[1.0, 1e-200, 0.0, 0.0], [0.0, 0.0, 1.0, 1e-7]])
    wf = DeterminantExpansion([[0], [1], [2], [3]], [[0]] * 4, coeffs)
    kept = truncate(wf, 1.0)
    assert kept.n_dets == 4
    assert kept.coeffs[0, 1] == 1e-200
    assert kept.coeffs[1, 3] == pytest.approx(1e-7, rel=1e-12)


def test_overlap_memory_guard_raises_before_allocating(monkeypatch):
    monkeypatch.setattr(determinants, "MAX_OVERLAP_BYTES", 10)
    wf = DeterminantExpansion([[0], [1]], [[0], [1]], [[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(MemoryError, match="estimated .* GiB; lower overlap_norm_kept"):
        state_overlaps(wf, wf, np.eye(2))
