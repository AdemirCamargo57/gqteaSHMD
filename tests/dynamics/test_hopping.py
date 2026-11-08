import numpy as np
import pytest

from gqteashmd.dynamics.hopping import (
    choose_state,
    fssh_probabilities,
    hopping_probabilities,
    sharc_probabilities,
)
from gqteashmd.dynamics.propagators import propagate


def _two_state(kappa=0.5, dt=0.01, ns=4, angle=0.7, e=(0.0, 0.05)):
    nac = np.zeros((2, 2, 1))
    nac[0, 1, 0], nac[1, 0, 0] = kappa, -kappa
    c = np.array([np.cos(angle), np.sin(angle)], dtype=complex)
    ev = np.array(e)
    v = np.ones(1)
    c2, p, starts = propagate(c, ev, ev, nac, nac, v, v, dt, ns)
    return c, c2, p, starts, nac, v, dt


def test_basic_properties():
    rng = np.random.default_rng(0)
    n, f = 4, 2
    nac = rng.normal(size=(n, n, f))
    nac = nac - nac.transpose(1, 0, 2)
    e = rng.normal(size=n)
    v = rng.normal(size=f)
    c = rng.normal(size=n) + 1j * rng.normal(size=n)
    c /= np.linalg.norm(c)
    c2, p, starts = propagate(c, e, e, nac, nac, v, v, 0.3, 5)
    for active in range(n):
        for h in (
            fssh_probabilities(active, starts, nac, nac, v, v, 0.3),
            sharc_probabilities(active, c, c2, p),
        ):
            assert np.all(h >= 0) and h.sum() <= 1.0 + 1e-12 and h[active] == 0


def test_no_coupling_gives_zero():
    c, c2, p, starts, nac, v, dt = _two_state(kappa=0.0)
    assert np.all(fssh_probabilities(0, starts, nac, nac, v, v, dt) == 0)
    assert np.all(sharc_probabilities(0, c, c2, p) == 0)


def test_small_dt_flux_and_agreement():
    c, c2, p, starts, nac, v, dt = _two_state(dt=0.002, ns=2)
    exact = (abs(c[0]) ** 2 - abs(c2[0]) ** 2) / abs(c[0]) ** 2
    assert exact > 0
    hf = fssh_probabilities(0, starts, nac, nac, v, v, dt).sum()
    hs = sharc_probabilities(0, c, c2, p).sum()
    assert hf == pytest.approx(exact, rel=1e-3)
    assert hs == pytest.approx(exact, rel=1e-3)
    assert hf == pytest.approx(hs, rel=1e-3)


def test_sharc_conservation_exact_two_state():
    c, c2, p, _, _, _, _ = _two_state(dt=0.4, ns=3)
    assert abs(c2[0]) < abs(c[0])
    h = sharc_probabilities(0, c, c2, p)
    assert h.sum() == pytest.approx(1 - abs(c2[0]) ** 2 / abs(c[0]) ** 2, rel=1e-12)


def test_population_gain_gives_zero():
    c, c2, p, _, _, _, _ = _two_state(angle=-0.7, dt=0.05)
    # amplitudes flow into state 0 here
    if abs(c2[0]) >= abs(c[0]):
        assert np.all(sharc_probabilities(0, c, c2, p) == 0)


def test_choose_state():
    h = np.array([0.0, 0.125, 0.25])
    assert choose_state(0, h, 0.1249) == 1
    assert choose_state(0, h, 0.125) == 2
    assert choose_state(0, h, 0.3749) == 2
    assert choose_state(0, h, 0.375) is None
    assert choose_state(1, np.array([0.2, 0.0, 0.1]), 0.25) == 2
    assert choose_state(0, np.zeros(3), 0.0) is None


def test_choose_state_statistics():
    h = np.array([0.0, 0.1, 0.3])
    rng = np.random.default_rng(5)
    n = 20000
    picks = [choose_state(0, h, rng.random()) for _ in range(n)]
    for j, pj in ((1, 0.1), (2, 0.3)):
        freq = sum(1 for x in picks if x == j) / n
        assert abs(freq - pj) < 4 * np.sqrt(pj * (1 - pj) / n)


def test_dispatch():
    c, c2, p, starts, nac, v, dt = _two_state(dt=0.2)
    args = (0, c, c2, p, starts, nac, nac, v, v, dt)
    assert np.all(hopping_probabilities("none", *args) == 0)
    assert np.allclose(hopping_probabilities("sharc", *args), sharc_probabilities(0, c, c2, p))
    assert np.allclose(
        hopping_probabilities("fssh", *args), fssh_probabilities(0, starts, nac, nac, v, v, dt)
    )
    with pytest.raises(ValueError, match="hopping"):
        hopping_probabilities("x", *args)
