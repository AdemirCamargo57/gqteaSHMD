import numpy as np
import pytest

from gqteashmd.overlaps.tracking import lowdin_orthonormalize, track


def test_identity_needs_no_correction():
    t = track(np.eye(3))
    np.testing.assert_array_equal(t.phases, [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(t.assignment, [0, 1, 2])
    assert t.root_flips == [] and t.min_singular_value == pytest.approx(1.0)


def test_a_sign_flip_is_corrected():
    s = np.diag([0.99, -0.98, 0.97])
    t = track(s)
    np.testing.assert_array_equal(t.phases, [1.0, -1.0, 1.0])
    assert np.all(np.diag(t.overlaps) > 0)


def test_a_root_flip_is_detected_and_phased_by_the_matched_element():
    s = np.array([[0.1, -0.99, 0.0], [0.98, 0.1, 0.0], [0.0, 0.0, 1.0]])
    t = track(s)
    np.testing.assert_array_equal(t.assignment, [1, 0, 2])
    assert t.root_flips == [(1, 0), (0, 1)]
    np.testing.assert_array_equal(t.phases, [1.0, -1.0, 1.0])
    assert t.overlaps[1, 0] > 0 and t.overlaps[0, 1] > 0


def test_small_singular_value_is_reported():
    s = np.array([[0.6, 0.0], [0.0, 0.4]])
    assert track(s).min_singular_value == pytest.approx(0.4)


def test_square_matrix_required():
    with pytest.raises(ValueError, match="square"):
        track(np.ones((2, 3)))


def test_lowdin_gives_the_nearest_orthogonal_matrix():
    rng = np.random.default_rng(4)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    s = 0.97 * q + 0.01 * rng.normal(size=(3, 3))
    u = lowdin_orthonormalize(s)
    np.testing.assert_allclose(u.T @ u, np.eye(3), atol=1e-12)
    assert np.abs(u - q).max() < 0.05
