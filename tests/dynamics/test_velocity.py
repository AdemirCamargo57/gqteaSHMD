import numpy as np
import pytest

from gqteashmd.dynamics.velocity import adjust_velocity

METHODS = ("parallel_vel", "parallel_nac")


def _draw(rng, f):
    m = rng.uniform(1.0, 2000.0, f)
    v = rng.normal(0.0, 0.01, f)
    u = rng.normal(0.0, 1.0, f)
    return m, v, u


def _kin(m, v):
    return 0.5 * float(np.sum(m * v * v))


@pytest.mark.parametrize("f", [1, 2, 6])
@pytest.mark.parametrize("method", METHODS)
def test_energy_conservation_and_frustration(f, method):
    rng = np.random.default_rng(100 + f)
    n_acc = 0
    for _ in range(200):
        m, v, u = _draw(rng, f)
        k = _kin(m, v)
        de = rng.uniform(-2.0, 2.0) * k
        adj = adjust_velocity(v, m, de, method=method, nac=u, frustrated="keep")
        if de <= 0:
            assert adj.accepted
        if adj.accepted:
            n_acc += 1
            assert _kin(m, adj.velocity) + de == pytest.approx(k, rel=1e-12, abs=1e-18)
        else:
            assert de > 0
            np.testing.assert_array_equal(adj.velocity, v)
    assert n_acc > 20


@pytest.mark.parametrize("f", [1, 2, 6])
def test_parallel_vel_keeps_direction(f):
    rng = np.random.default_rng(f)
    for _ in range(200):
        m, v, _ = _draw(rng, f)
        k = _kin(m, v)
        de = rng.uniform(-1.0, 0.9) * k
        adj = adjust_velocity(v, m, de, method="parallel_vel", nac=None, frustrated="keep")
        assert adj.accepted
        alpha = np.sqrt(1 - de / k)
        np.testing.assert_allclose(adj.velocity, alpha * v, rtol=1e-13)
    # frustrated exactly when dE > K
    adj = adjust_velocity(v, m, 1.01 * k, method="parallel_vel", nac=None, frustrated="keep")
    assert not adj.accepted


@pytest.mark.parametrize("f", [1, 2, 6])
def test_parallel_nac_along_minv_u_and_smaller_root(f):
    rng = np.random.default_rng(7 + f)
    for _ in range(200):
        m, v, u = _draw(rng, f)
        k = _kin(m, v)
        de = rng.uniform(-1.0, 1.0) * k
        adj = adjust_velocity(v, m, de, method="parallel_nac", nac=u, frustrated="keep")
        if not adj.accepted:
            continue
        dv = adj.velocity - v
        minv_u = u / m
        gamma = float(dv @ minv_u / (minv_u @ minv_u))
        np.testing.assert_allclose(dv, gamma * minv_u, atol=1e-13 * (1 + np.abs(dv).max()))
        a = 0.5 * float(np.sum(u * u / m))
        b = float(v @ u)
        d = b * b - 4 * a * de
        roots = [(-b + s * np.sqrt(d)) / (2 * a) for s in (1, -1)]
        other = max(roots, key=abs)
        assert abs(gamma) <= abs(other) + 1e-12
    # small dE: continuously connected to gamma = 0
    m, v, u = _draw(rng, f)
    adj = adjust_velocity(v, m, 1e-12, method="parallel_nac", nac=u, frustrated="keep")
    np.testing.assert_allclose(adj.velocity, v, atol=1e-8)


@pytest.mark.parametrize("f", [1, 2, 6])
def test_negative_discriminant_frustrated_and_reflect(f):
    rng = np.random.default_rng(55 + f)
    for _ in range(200):
        m, v, u = _draw(rng, f)
        a = 0.5 * float(np.sum(u * u / m))
        b = float(v @ u)
        de = 2.0 * b * b / (4 * a) + 1e-6  # D < 0
        keep = adjust_velocity(v, m, de, method="parallel_nac", nac=u, frustrated="keep")
        assert not keep.accepted
        np.testing.assert_array_equal(keep.velocity, v)
        ref = adjust_velocity(v, m, de, method="parallel_nac", nac=u, frustrated="reflect")
        assert not ref.accepted
        assert _kin(m, ref.velocity) == pytest.approx(_kin(m, v), rel=1e-12)
        minv_u = u / m
        # component along u in the mass-weighted metric reversed, the rest unchanged
        pv = float(v @ u) / (2 * a)
        pr = float(ref.velocity @ u) / (2 * a)
        assert pr == pytest.approx(-pv, rel=1e-10, abs=1e-14)
        perp = v - pv * minv_u
        perp_r = ref.velocity - pr * minv_u
        np.testing.assert_allclose(perp_r, perp, atol=1e-13)


def test_reflect_parallel_vel_reverses():
    m = np.array([1.0, 3.0])
    v = np.array([0.1, -0.2])
    adj = adjust_velocity(v, m, 10.0, method="parallel_vel", nac=None, frustrated="reflect")
    assert not adj.accepted
    np.testing.assert_allclose(adj.velocity, -v)


def test_edge_cases_and_errors():
    m = np.array([1.0])
    z = np.zeros(1)
    adj = adjust_velocity(z, m, 0.1, method="parallel_vel", nac=None, frustrated="keep")
    assert not adj.accepted
    adj = adjust_velocity(z, m, 0.0, method="parallel_vel", nac=None, frustrated="keep")
    assert adj.accepted
    np.testing.assert_array_equal(adj.velocity, z)
    with pytest.raises(ValueError, match="parallel_nac"):
        adjust_velocity(z, m, -0.1, method="parallel_vel", nac=None, frustrated="keep")
    with pytest.raises(ValueError, match="nac"):
        adjust_velocity(z, m, 0.1, method="parallel_nac", nac=None, frustrated="keep")
    with pytest.raises(ValueError, match="method"):
        adjust_velocity(z, m, 0.1, method="x", nac=None, frustrated="keep")
    with pytest.raises(ValueError, match="frustrated"):
        adjust_velocity(z, m, 0.1, method="parallel_vel", nac=None, frustrated="x")
    # zero NAC falls back to the parallel_vel rule
    v = np.array([0.2])
    adj = adjust_velocity(v, m, 0.01, method="parallel_nac", nac=z, frustrated="keep")
    assert adj.accepted
    np.testing.assert_allclose(adj.velocity, v * np.sqrt(1 - 0.01 / 0.02))
    # K = 0 with parallel_nac, downhill: works along the NAC
    adj = adjust_velocity(z, m, -0.01, method="parallel_nac", nac=np.ones(1), frustrated="keep")
    assert adj.accepted
    assert _kin(m, adj.velocity) == pytest.approx(0.01)
