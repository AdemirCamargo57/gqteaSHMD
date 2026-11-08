import numpy as np
import pytest
from scipy.linalg import expm

from gqteashmd.dynamics.models import DiabaticModel
from gqteashmd.dynamics.propagators import coupling_matrix, propagate


def _rand_setup(rng, n=3, f=2):
    e = rng.normal(size=n)
    nac = rng.normal(size=(n, n, f))
    nac = nac - nac.transpose(1, 0, 2)
    v = rng.normal(size=f)
    c = rng.normal(size=n) + 1j * rng.normal(size=n)
    return e, nac, v, c / np.linalg.norm(c)


def test_coupling_matrix_antisymmetric():
    e, nac, v, _ = _rand_setup(np.random.default_rng(0))
    t = coupling_matrix(nac, v)
    assert np.allclose(t, -t.T)
    assert np.isclose(t[0, 1], nac[0, 1] @ v)


@pytest.mark.parametrize("ns", [1, 7])
def test_unitarity_and_norm(ns):
    rng = np.random.default_rng(1)
    e0, nac0, v0, c = _rand_setup(rng)
    e1, nac1, v1, _ = _rand_setup(rng)
    c2, p, starts = propagate(c, e0, e1, nac0, nac1, v0, v1, 0.3, ns)
    assert np.allclose(p.conj().T @ p, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.norm(c2), 1.0, atol=1e-12)
    assert np.allclose(p @ c, c2)
    assert len(starts) == ns and np.allclose(starts[0], c)


@pytest.mark.parametrize("ns", [1, 25])
def test_matches_expm_for_constant_hamiltonian(ns):
    e, nac, v, c = _rand_setup(np.random.default_rng(2))
    dt = 0.4
    _, p, _ = propagate(c, e, e, nac, nac, v, v, dt, ns)
    ref = expm(-1j * (np.diag(e) - 1j * coupling_matrix(nac, v)) * dt)
    assert np.allclose(p, ref, atol=1e-12)


@pytest.mark.parametrize("ns", [1, 2])
def test_midpoint_interpolation(ns):
    rng = np.random.default_rng(4)
    e0, nac0, v0, c = _rand_setup(rng)
    e1, nac1, v1, _ = _rand_setup(rng)
    dt = 0.35
    _, p, _ = propagate(c, e0, e1, nac0, nac1, v0, v1, dt, ns)
    ref = np.eye(3, dtype=complex)
    for m in range(1, ns + 1):
        s = (m - 0.5) / ns
        h = np.diag((1 - s) * e0 + s * e1) - 1j * coupling_matrix(
            (1 - s) * nac0 + s * nac1, (1 - s) * v0 + s * v1
        )
        ref = expm(-1j * h * dt / ns) @ ref
    assert np.allclose(p, ref, atol=1e-12)


def test_rabi():
    kappa, t = 0.3, 2.0
    nac = np.zeros((2, 2, 1))
    nac[0, 1, 0], nac[1, 0, 0] = kappa, -kappa
    e = np.zeros(2)
    v = np.ones(1)
    c, _, _ = propagate(np.array([1, 0], complex), e, e, nac, nac, v, v, t, 4)
    # dc0/dt = -T01 c1, dc1/dt = -T10 c0 = kappa c0 -> c0 = cos(kappa t)
    assert np.allclose(abs(c) ** 2, [np.cos(kappa * t) ** 2, np.sin(kappa * t) ** 2], atol=1e-12)


def test_time_reversal():
    rng = np.random.default_rng(3)
    e, nac, v, c = _rand_setup(rng)
    c1, _, _ = propagate(c, e, e, nac, nac, v, v, 0.5, 5)
    # Running the same flow backwards in time: dt -> -dt at the same v (T dt flips sign).
    c0, _, _ = propagate(c1, e, e, nac, nac, v, v, -0.5, 5)
    assert np.allclose(c0, c, atol=1e-12)


def test_adiabatic_limit_phases_only():
    e = np.array([0.1, 0.3])
    nac = np.zeros((2, 2, 1))
    c = np.array([0.6, 0.8j])
    c2, _, _ = propagate(c, e, e, nac, nac, np.ones(1), np.ones(1), 1.7, 3)
    assert np.allclose(c2, c * np.exp(-1j * e * 1.7), atol=1e-14)


class _LinearCrossing(DiabaticModel):
    n_states = 2
    n_dof = 1

    def __init__(self, alpha, gamma):
        self.alpha, self.gamma = alpha, gamma
        self.masses = np.ones(1)

    def diabatic(self, q):
        a, g = self.alpha, self.gamma
        v = np.array([[a * q[0], g], [g, -a * q[0]]])
        dv = np.array([[[a, 0.0], [0.0, -a]]])
        return v, dv


@pytest.mark.parametrize("u", [0.002, 0.005, 0.01])
def test_landau_zener(u):
    gamma, alpha = 0.005, 0.01
    model = _LinearCrossing(alpha, gamma)
    length = 40 * gamma / alpha
    dt, ns = 2.0, 2
    nsteps = int(round(2 * length / u / dt))
    dt = 2 * length / u / nsteps
    xs = -length + u * dt * np.arange(nsteps + 1)
    snaps = []
    prev = None
    for x in xs:
        prev = model.evaluate(np.array([x]), prev)
        snaps.append(prev)
    # adiabatic state continuity: the initial state is the lower one
    c = np.array([1, 0], dtype=complex)
    vel = np.array([u])
    for s0, s1 in zip(snaps[:-1], snaps[1:], strict=True):
        c, _, _ = propagate(c, s0.energies, s1.energies, s0.nacs, s1.nacs, vel, vel, dt, ns)
    leave = 1.0 - abs(c[0]) ** 2
    expected = np.exp(-np.pi * gamma**2 / (u * alpha))
    assert abs(leave - expected) < 1e-3, (leave, expected)


def _propagate_loop(c, e0, e1, nac0, nac1, v0, v1, dt, nsubsteps):
    """Straightforward reference: one substep at a time (the pre-vectorization code)."""
    c = np.asarray(c, dtype=complex)
    n = c.shape[0]
    delta = dt / nsubsteps
    p = np.eye(n, dtype=complex)
    current = c.copy()
    starts = []
    for m in range(1, nsubsteps + 1):
        s = (m - 0.5) / nsubsteps
        e = (1 - s) * e0 + s * e1
        t = coupling_matrix((1 - s) * nac0 + s * nac1, (1 - s) * v0 + s * v1)
        w, u = np.linalg.eigh(np.diag(e).astype(complex) - 1j * t)
        p_m = (u * np.exp(-1j * w * delta)) @ u.conj().T
        starts.append(current.copy())
        current = p_m @ current
        p = p_m @ p
    return current, p, starts


@pytest.mark.parametrize("ns", [1, 2, 7, 25])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vectorized_propagate_matches_the_loop_reference(ns, seed):
    rng = np.random.default_rng(100 + seed)
    e0, nac0, v0, c = _rand_setup(rng, n=3, f=2)
    e1, nac1, v1, _ = _rand_setup(rng, n=3, f=2)
    c2, p, starts = propagate(c, e0, e1, nac0, nac1, v0, v1, 0.7, ns)
    c2_ref, p_ref, starts_ref = _propagate_loop(c, e0, e1, nac0, nac1, v0, v1, 0.7, ns)
    assert np.max(np.abs(c2 - c2_ref)) <= 1e-13
    assert np.max(np.abs(p - p_ref)) <= 1e-13
    assert starts.shape == (ns, 3)
    assert np.max(np.abs(starts - np.array(starts_ref))) <= 1e-13


def test_vectorized_propagate_matches_the_loop_on_the_sac_model():
    from gqteashmd.dynamics.models import TullySimpleAvoidedCrossing

    model = TullySimpleAvoidedCrossing()
    s0 = model.evaluate(np.array([-0.2]), None)
    s1 = model.evaluate(np.array([-0.19]), s0)
    v0, v1 = np.array([0.01]), np.array([0.0101])
    c = np.array([0.6, 0.8j])
    got = propagate(c, s0.energies, s1.energies, s0.nacs, s1.nacs, v0, v1, 1.0, 25)
    ref = _propagate_loop(c, s0.energies, s1.energies, s0.nacs, s1.nacs, v0, v1, 1.0, 25)
    for a, b in zip(got, ref, strict=True):
        assert np.max(np.abs(np.asarray(a) - np.array(b))) <= 1e-13


def test_non_hermitian_input_is_hermitized():
    """A slightly non-antisymmetric d (rounding in a model) gives the Hermitian-part result."""
    rng = np.random.default_rng(7)
    e, nac, v, c = _rand_setup(rng, n=3, f=2)
    skew = nac + 1e-9 * rng.normal(size=nac.shape)
    _, p, _ = propagate(c, e, e, skew, skew, v, v, 0.5, 3)
    anti = 0.5 * (skew - skew.transpose(1, 0, 2))
    _, p_ref, _ = propagate(c, e, e, anti, anti, v, v, 0.5, 3)
    assert np.allclose(p.conj().T @ p, np.eye(3), atol=1e-12)
    assert np.max(np.abs(p - p_ref)) <= 1e-13
