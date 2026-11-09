import numpy as np
import pytest

from gqteashmd.dynamics.decoherence import EDC_CONVENTIONS, apply_decoherence

GP = "granucci_persico"


def _rand_c(rng, n):
    c = rng.normal(size=n) + 1j * rng.normal(size=n)
    return c / np.linalg.norm(c)


@pytest.mark.parametrize("convention", EDC_CONVENTIONS)
@pytest.mark.parametrize("n", [2, 3, 6])
def test_norm_and_active_amplitude(n, convention):
    rng = np.random.default_rng(n)
    for _ in range(200):
        c = _rand_c(rng, n)
        e = rng.uniform(0, 0.2, n)
        act = int(rng.integers(n))
        out = apply_decoherence(
            c, act, e, rng.uniform(0.001, 0.1), 20.0, scheme="edc", param=0.1, convention=convention
        )
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-14)
        assert abs(out[act]) >= abs(c[act]) - 1e-15
        assert np.angle(out[act] * np.conj(c[act])) == pytest.approx(0.0, abs=1e-12)


def test_conventions_are_the_two_documented_ones():
    assert EDC_CONVENTIONS == ("sharc", GP)


def test_two_state_by_hand_granucci_persico():
    """c_j <- c_j exp(-dt/tau), tau = (1/|dE|)(1 + C/K) (Granucci & Persico 2007)."""
    e = np.array([0.0, 0.02])
    c = np.array([0.6, 0.8j])
    out = apply_decoherence(c, 0, e, 0.01, 1.0, scheme="edc", param=0.1, convention=GP)
    tau = (1 / 0.02) * (1 + 0.1 / 0.01)
    assert tau == pytest.approx(550.0)
    cj = 0.8 * np.exp(-1.0 / 550.0)
    assert out[1] == pytest.approx(1j * cj, rel=1e-14)
    assert abs(out[0]) == pytest.approx(np.sqrt(1 - cj**2), rel=1e-14)


def test_two_state_by_hand_sharc():
    """SHARC 4.1 manual eq. (8.25): c_j <- c_j exp[-(1/2) dt |dE| (1 + C/K)^-1]."""
    e = np.array([0.0, 0.02])
    c = np.array([0.6, 0.8j])
    out = apply_decoherence(c, 0, e, 0.01, 1.0, scheme="edc", param=0.1, convention="sharc")
    cj = 0.8 * np.exp(-0.5 * 1.0 * 0.02 / (1 + 0.1 / 0.01))
    assert cj == pytest.approx(0.8 * np.exp(-1.0 / 1100.0), rel=1e-15)
    assert out[1] == pytest.approx(1j * cj, rel=1e-14)
    assert abs(out[0]) == pytest.approx(np.sqrt(1 - cj**2), rel=1e-14)
    assert np.angle(out[0]) == pytest.approx(0.0, abs=1e-15)


def test_sharc_damping_exponent_is_half_the_granucci_persico_one():
    rng = np.random.default_rng(11)
    c = _rand_c(rng, 4)
    e = rng.uniform(0, 0.3, 4)
    kw = {"scheme": "edc", "param": 0.1}
    gp = apply_decoherence(c, 2, e, 0.05, 3.0, convention=GP, **kw)
    sh = apply_decoherence(c, 2, e, 0.05, 3.0, convention="sharc", **kw)
    for j in (0, 1, 3):
        assert np.log(abs(sh[j]) / abs(c[j])) == pytest.approx(
            0.5 * np.log(abs(gp[j]) / abs(c[j])), rel=1e-12
        )


@pytest.mark.parametrize("convention", EDC_CONVENTIONS)
def test_none_input_untouched_and_edges(convention):
    kw = {"param": 0.1, "convention": convention}
    c = np.array([0.6, 0.8], dtype=complex)
    c0 = c.copy()
    e = np.array([0.0, 0.02])
    out = apply_decoherence(c, 0, e, 0.01, 1.0, scheme="none", **kw)
    np.testing.assert_array_equal(out, c0)
    out = apply_decoherence(c, 0, e, 0.01, 1.0, scheme="edc", **kw)
    np.testing.assert_array_equal(c, c0)
    assert out is not c
    np.testing.assert_allclose(apply_decoherence(c, 0, e, 0.0, 1.0, scheme="edc", **kw), c0)
    np.testing.assert_allclose(
        apply_decoherence(c, 0, np.zeros(2), 0.01, 1.0, scheme="edc", **kw), c0
    )
    c1 = np.array([0.0, 1.0], dtype=complex)
    np.testing.assert_array_equal(apply_decoherence(c1, 0, e, 0.01, 1.0, scheme="edc", **kw), c1)
    with pytest.raises(ValueError, match="scheme"):
        apply_decoherence(c, 0, e, 0.01, 1.0, scheme="x", **kw)


def test_unknown_convention():
    c = np.array([0.6, 0.8], dtype=complex)
    with pytest.raises(ValueError, match="convention"):
        apply_decoherence(
            c, 0, np.array([0.0, 0.02]), 0.01, 1.0, scheme="edc", param=0.1, convention="x"
        )
