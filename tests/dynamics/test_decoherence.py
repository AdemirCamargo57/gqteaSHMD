import numpy as np
import pytest

from gqteashmd.dynamics.decoherence import apply_decoherence


def _rand_c(rng, n):
    c = rng.normal(size=n) + 1j * rng.normal(size=n)
    return c / np.linalg.norm(c)


@pytest.mark.parametrize("n", [2, 3, 6])
def test_norm_and_active_amplitude(n):
    rng = np.random.default_rng(n)
    for _ in range(200):
        c = _rand_c(rng, n)
        e = rng.uniform(0, 0.2, n)
        act = int(rng.integers(n))
        out = apply_decoherence(c, act, e, rng.uniform(0.001, 0.1), 20.0, scheme="edc", param=0.1)
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-14)
        assert abs(out[act]) >= abs(c[act]) - 1e-15
        assert np.angle(out[act] * np.conj(c[act])) == pytest.approx(0.0, abs=1e-12)


def test_two_state_by_hand():
    e = np.array([0.0, 0.02])
    c = np.array([0.6, 0.8j])
    out = apply_decoherence(c, 0, e, 0.01, 1.0, scheme="edc", param=0.1)
    tau = (1 / 0.02) * (1 + 0.1 / 0.01)
    assert tau == pytest.approx(550.0)
    cj = 0.8 * np.exp(-1.0 / 550.0)
    assert out[1] == pytest.approx(1j * cj, rel=1e-14)
    assert abs(out[0]) == pytest.approx(np.sqrt(1 - cj**2), rel=1e-14)


def test_none_input_untouched_and_edges():
    c = np.array([0.6, 0.8], dtype=complex)
    c0 = c.copy()
    e = np.array([0.0, 0.02])
    out = apply_decoherence(c, 0, e, 0.01, 1.0, scheme="none", param=0.1)
    np.testing.assert_array_equal(out, c0)
    out = apply_decoherence(c, 0, e, 0.01, 1.0, scheme="edc", param=0.1)
    np.testing.assert_array_equal(c, c0)
    assert out is not c
    np.testing.assert_allclose(apply_decoherence(c, 0, e, 0.0, 1.0, scheme="edc", param=0.1), c0)
    np.testing.assert_allclose(
        apply_decoherence(c, 0, np.zeros(2), 0.01, 1.0, scheme="edc", param=0.1), c0
    )
    c1 = np.array([0.0, 1.0], dtype=complex)
    np.testing.assert_array_equal(
        apply_decoherence(c1, 0, e, 0.01, 1.0, scheme="edc", param=0.1), c1
    )
    with pytest.raises(ValueError, match="scheme"):
        apply_decoherence(c, 0, e, 0.01, 1.0, scheme="x", param=0.1)
