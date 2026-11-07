import math

import numpy as np
import pytest

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


def test_inconsistent_shapes_are_rejected():
    with pytest.raises(ValueError, match="determinant counts"):
        DeterminantExpansion([[0], [1]], [[0]], [[1.0, 0.0]])


def test_with_signs_flips_whole_states():
    wf = DeterminantExpansion([[0], [1]], [[0], [0]], [[1.0, 0.0], [0.0, 1.0]])
    flipped = wf.with_signs(np.array([1.0, -1.0]))
    np.testing.assert_array_equal(flipped.coeffs, [[1.0, 0.0], [0.0, -1.0]])


def test_truncate_keeps_the_dominant_determinants_and_renormalises():
    coeffs = np.array([[0.9, 0.4, 0.1, 0.0], [0.0, 0.1, 0.995, 0.0]])
    wf = DeterminantExpansion([[0], [1], [2], [3]], [[0]] * 4, coeffs)
    kept = truncate(wf, 0.95)
    assert kept.n_dets == 3  # determinant 3 is used by no state
    np.testing.assert_allclose(np.linalg.norm(kept.coeffs, axis=1), 1.0)
    assert (
        kept.coeffs[0, 2] == 0.0
    )  # 0.81 + 0.16 = 0.97 of state 0: the 0.01 determinant is dropped
    full = truncate(wf, 1.0)
    assert full.n_dets == 3
    with pytest.raises(ValueError, match="norm_kept"):
        truncate(wf, 0.0)
