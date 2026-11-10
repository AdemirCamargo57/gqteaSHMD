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


def _propagate_endpoint_loop(c, e0, e1, nac0, nac1, v0, v1, dt, nsubsteps):
    """SHARC 4.1 manual eqs. (8.191)-(8.194): substep i = 1..N uses s = i/N, with H and the
    coupling T = v.K each interpolated linearly between the step ends."""
    c = np.asarray(c, dtype=complex)
    n = c.shape[0]
    delta = dt / nsubsteps
    t0, t1 = coupling_matrix(nac0, v0), coupling_matrix(nac1, v1)
    p = np.eye(n, dtype=complex)
    current = c.copy()
    starts = []
    for i in range(1, nsubsteps + 1):
        s = i / nsubsteps
        h = np.diag(e0 + s * (e1 - e0)) - 1j * (t0 + s * (t1 - t0))
        p_i = expm(-1j * h * delta)
        starts.append(current.copy())
        current = p_i @ current
        p = p_i @ p
    return current, p, starts


@pytest.mark.parametrize("ns", [1, 2, 7, 25])
@pytest.mark.parametrize("seed", [0, 1])
def test_endpoint_matches_the_manual_loop_reference(ns, seed):
    rng = np.random.default_rng(200 + seed)
    e0, nac0, v0, c = _rand_setup(rng, n=3, f=2)
    e1, nac1, v1, _ = _rand_setup(rng, n=3, f=2)
    got = propagate(c, e0, e1, nac0, nac1, v0, v1, 0.7, ns, interpolation="endpoint")
    ref = _propagate_endpoint_loop(c, e0, e1, nac0, nac1, v0, v1, 0.7, ns)
    for a, b in zip(got, ref, strict=True):
        assert np.max(np.abs(np.asarray(a) - np.array(b))) <= 1e-12
    p = got[1]
    assert np.allclose(p.conj().T @ p, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.norm(got[0]), 1.0, atol=1e-12)


def test_midpoint_is_the_default():
    rng = np.random.default_rng(5)
    e0, nac0, v0, c = _rand_setup(rng)
    e1, nac1, v1, _ = _rand_setup(rng)
    a = propagate(c, e0, e1, nac0, nac1, v0, v1, 0.5, 4)
    b = propagate(c, e0, e1, nac0, nac1, v0, v1, 0.5, 4, interpolation="midpoint")
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)
    with pytest.raises(ValueError, match="interpolation"):
        propagate(c, e0, e1, nac0, nac1, v0, v1, 0.5, 4, interpolation="left")


def test_endpoint_and_midpoint_converge_to_the_same_limit():
    """With v constant both schemes integrate the same linear H(s) and T(s); against a fine
    reference the end-point error falls as 1/N_s, the midpoint error as 1/N_s^2."""
    rng = np.random.default_rng(9)
    e0, nac0, v, c = _rand_setup(rng, n=3, f=2)
    e1, nac1, _, _ = _rand_setup(rng, n=3, f=2)
    dt = 1.5
    ref, _, _ = propagate(c, e0, e1, nac0, nac1, v, v, dt, 4096)
    errors = {}
    for scheme in ("midpoint", "endpoint"):
        errors[scheme] = [
            np.max(
                np.abs(
                    propagate(c, e0, e1, nac0, nac1, v, v, dt, ns, interpolation=scheme)[0] - ref
                )
            )
            for ns in (16, 32, 64)
        ]
    end, mid = errors["endpoint"], errors["midpoint"]
    for k in range(2):
        assert 1.8 < end[k] / end[k + 1] < 2.2, end
        assert 3.6 < mid[k] / mid[k + 1] < 4.4, mid
    assert mid[-1] < end[-1]


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


@pytest.mark.parametrize("interpolation", ["midpoint", "endpoint"])
def test_energies_are_added_for_a_non_contiguous_coupling_stack(monkeypatch, interpolation):
    """The energies reach the diagonal of H_eff whatever the memory layout of the coupling
    stack (an in-place add through a reshape would silently drop them on a copy)."""
    from gqteashmd.dynamics import propagators

    rng = np.random.default_rng(7)
    e0, nac0, v0, c = _rand_setup(rng, n=3, f=2)
    e1, nac1, v1, _ = _rand_setup(rng, n=3, f=2)
    args = (c, e0, e1, nac0, nac1, v0, v1, 0.7, 5)
    reference = propagate(*args, interpolation=interpolation)
    original = propagators.substep_couplings

    def fortran_ordered(*a, **kw):
        t = np.asfortranarray(original(*a, **kw))
        assert not t.flags.c_contiguous
        return t

    monkeypatch.setattr(propagators, "substep_couplings", fortran_ordered)
    result = propagate(*args, interpolation=interpolation)
    for got, ref in zip(result, reference, strict=True):
        assert np.max(np.abs(got - ref)) <= 1e-13


# --- P7a-2 Task 5: propagation in the diagonal basis (Method reference 7) ---------------------


def _random_unitary(rng, n):
    q, r = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _random_orthogonal(rng, n):
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return q


def _random_hermitian(rng, n, scale=0.1):
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return scale * 0.5 * (a + a.conj().T)


def _diag_kw(coupling, overlaps=None, interpolation="midpoint"):
    return {"coupling": coupling, "overlaps": overlaps, "interpolation": interpolation}


@pytest.mark.parametrize("interpolation", ["midpoint", "endpoint"])
def test_diagonal_fast_path_is_the_p7a1_propagate(interpolation):
    from gqteashmd.dynamics.propagators import propagate_diagonal

    rng = np.random.default_rng(500)
    for _ in range(20):
        n = int(rng.integers(2, 5))
        f = int(rng.integers(1, 4))
        ns = int(rng.integers(1, 30))
        dt = float(rng.uniform(0.1, 2.0))
        e0, nac0, v0, c = _rand_setup(rng, n=n, f=f)
        e1, nac1, v1, _ = _rand_setup(rng, n=n, f=f)
        h0, h1 = np.diag(e0).astype(complex), np.diag(e1).astype(complex)
        kw = _diag_kw("nacdr", interpolation=interpolation)
        got = propagate_diagonal(c, None, None, h0, h1, nac0, nac1, v0, v1, dt, ns, **kw)
        ref = propagate(c, e0, e1, nac0, nac1, v0, v1, dt, ns, interpolation=interpolation)
        assert len(got) == 3
        for a, b in zip(got, ref, strict=True):
            assert np.array_equal(a, b)


@pytest.mark.parametrize("interpolation", ["midpoint", "endpoint"])
@pytest.mark.parametrize("coupling", ["nacdr", "overlap"])
def test_diagonal_propagator_is_unitary(coupling, interpolation):
    from gqteashmd.dynamics.propagators import propagate_diagonal

    rng = np.random.default_rng(510)
    for _ in range(10):
        n, f = 4, 3
        _, nac0, v0, c = _rand_setup(rng, n=n, f=f)
        _, nac1, v1, _ = _rand_setup(rng, n=n, f=f)
        h0, h1 = _random_hermitian(rng, n), _random_hermitian(rng, n)
        u0, u1 = _random_unitary(rng, n), _random_unitary(rng, n)
        s = _random_orthogonal(rng, n) if coupling == "overlap" else None
        kw = _diag_kw(coupling, s, interpolation)
        c2, r, starts = propagate_diagonal(c, u0, u1, h0, h1, nac0, nac1, v0, v1, 0.8, 7, **kw)
        assert starts is None
        assert np.max(np.abs(r.conj().T @ r - np.eye(n))) <= 1e-12
        assert np.max(np.abs(r @ c - c2)) <= 1e-14
        assert abs(np.linalg.norm(c2) - 1.0) <= 1e-12


@pytest.mark.parametrize("interpolation", ["midpoint", "endpoint"])
def test_general_path_with_identity_u_matches_the_fast_path(interpolation):
    """Ruling F: explicit identity matrices for U take the general path."""
    from gqteashmd.dynamics.propagators import propagate_diagonal

    rng = np.random.default_rng(520)
    eye = np.eye(3, dtype=complex)
    kw = _diag_kw("nacdr", interpolation=interpolation)
    for _ in range(10):
        e0, nac0, v0, c = _rand_setup(rng, n=3, f=2)
        e1, nac1, v1, _ = _rand_setup(rng, n=3, f=2)
        h0, h1 = np.diag(e0).astype(complex), np.diag(e1).astype(complex)
        fast = propagate_diagonal(c, None, None, h0, h1, nac0, nac1, v0, v1, 0.6, 9, **kw)
        general = propagate_diagonal(c, eye, eye, h0, h1, nac0, nac1, v0, v1, 0.6, 9, **kw)
        assert fast[2] is not None and general[2] is None
        assert np.max(np.abs(general[0] - fast[0])) <= 1e-13
        assert np.max(np.abs(general[1] - fast[1])) <= 1e-13


@pytest.mark.parametrize("interpolation", ["midpoint", "endpoint"])
@pytest.mark.parametrize("coupling", ["nacdr", "overlap"])
def test_frozen_nuclei_follow_the_exact_exponential(coupling, interpolation):
    """v = 0 and a constant H with spin-orbit coupling: c^MCH(t) = exp(-iHt) c^MCH(0)."""
    from gqteashmd.dynamics.propagators import propagate_diagonal

    rng = np.random.default_rng(530)
    n, f, dt = 4, 2, 0.5
    h = _random_hermitian(rng, n)
    _, u = np.linalg.eigh(h)
    u = u.astype(complex)
    nac = rng.normal(size=(n, n, f))
    nac = nac - nac.transpose(1, 0, 2)
    v = np.zeros(f)
    s = np.eye(n) if coupling == "overlap" else None
    kw = _diag_kw(coupling, s, interpolation)
    c0_mch = rng.normal(size=n) + 1j * rng.normal(size=n)
    c0_mch /= np.linalg.norm(c0_mch)
    c = u.conj().T @ c0_mch
    worst = 0.0
    for k in range(1, 101):
        c, _, _ = propagate_diagonal(c, u, u, h, h, nac, nac, v, v, dt, 5, **kw)
        ref = expm(-1j * h * dt * k) @ c0_mch
        worst = max(worst, float(np.max(np.abs(u @ c - ref))))
    assert worst <= 1e-10, worst


@pytest.mark.parametrize("coupling", ["nacdr", "overlap"])
def test_rabi_oscillation_in_the_diagonal_basis(coupling):
    from gqteashmd.dynamics.propagators import propagate_diagonal

    big_v, delta, dt = 0.01 * np.exp(0.7j), 0.02, 2.0
    h = np.array([[0.0, big_v], [np.conj(big_v), delta]], dtype=complex)
    _, u = np.linalg.eigh(h)
    u = u.astype(complex)
    omega = np.sqrt(delta**2 + 4 * abs(big_v) ** 2)
    nac = np.zeros((2, 2, 1))
    v = np.zeros(1)
    s = np.eye(2) if coupling == "overlap" else None
    kw = _diag_kw(coupling, s)
    c = u.conj().T @ np.array([1.0, 0.0], dtype=complex)
    worst = 0.0
    for k in range(1, 201):
        c, _, _ = propagate_diagonal(c, u, u, h, h, nac, nac, v, v, dt, 3, **kw)
        t = k * dt
        expected = 4 * abs(big_v) ** 2 / omega**2 * np.sin(omega * t / 2) ** 2
        worst = max(worst, abs(abs((u @ c)[1]) ** 2 - expected))
    assert worst <= 1e-10, worst


def _trivial_crossing_snapshot(alpha, x, previous_vectors, sign):
    """MCH (energy-ordered) data of the uncoupled diabatic H = diag(+alpha x, -alpha x).

    The adiabatic states swap character at x = 0; ``sign`` flips the phase of the lower state
    after the crossing, so the overlap there is a signed permutation."""
    from gqteashmd.dynamics.records import ElectronicSnapshot

    if x < 0:  # lower adiabatic state = diabatic state 0
        energies = np.array([alpha * x, -alpha * x])
        w = np.eye(2)
    else:  # lower adiabatic state = diabatic state 1
        energies = np.array([-alpha * x, alpha * x])
        w = np.array([[0.0, 1.0], [sign, 0.0]])
    overlaps = None if previous_vectors is None else previous_vectors.T @ w
    return ElectronicSnapshot(
        energies=energies,
        gradients=np.zeros((2, 1)),
        nacs=np.zeros((2, 2, 1)),  # G_01 = 0 everywhere: no coupling off the crossing point
        hamiltonian=np.diag(energies).astype(complex),
        vectors=w,
        overlaps=overlaps,
    )


def _run_trivial_crossing(coupling, sign=1.0):
    from gqteashmd.dynamics.propagators import propagate_diagonal

    alpha, vel, dt, ns = 0.01, 0.01, 1.0, 5
    xs = -0.2 + 0.0013 + vel * dt * np.arange(41)  # x = 0 falls between two steps
    snaps = []
    prev = None
    for x in xs:
        prev = _trivial_crossing_snapshot(alpha, x, None if prev is None else prev.vectors, sign)
        snaps.append(prev)
    v = np.array([vel])
    c = np.array([1.0, 0.0], dtype=complex)  # lower adiabatic state = diabatic state 0
    diabatic = [abs((snaps[0].vectors @ c)[0]) ** 2]
    swaps = 0
    for s0, s1 in zip(snaps[:-1], snaps[1:], strict=True):
        if not np.array_equal(s1.overlaps, np.eye(2)):
            swaps += 1
        kw = _diag_kw(coupling, s1.overlaps if coupling == "overlap" else None)
        h0, h1 = s0.hamiltonian, s1.hamiltonian
        c, _, _ = propagate_diagonal(c, None, None, h0, h1, s0.nacs, s1.nacs, v, v, dt, ns, **kw)
        diabatic.append(abs((s1.vectors @ c)[0]) ** 2)
    return np.array(diabatic), swaps


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_local_diabatization_follows_a_trivial_crossing(sign):
    diabatic, swaps = _run_trivial_crossing("overlap", sign)
    assert swaps == 1
    assert np.max(np.abs(diabatic - 1.0)) <= 1e-12
    # NAC propagation with finite steps never sees the delta-function coupling: the population
    # stays adiabatic and leaves the diabatic state (number in the Task 5 report).
    nac_diabatic, _ = _run_trivial_crossing("nacdr", sign)
    assert abs(nac_diabatic[-1] - 1.0) > 0.5, nac_diabatic[-1]


def _sac_populations(coupling, dt, ns=25):
    """|c|^2 after a fixed classical path x = -4 + v t to x = +4 (k = 20) on Tully SAC."""
    from gqteashmd.dynamics.models import TullySimpleAvoidedCrossing
    from gqteashmd.dynamics.propagators import propagate_diagonal

    model = TullySimpleAvoidedCrossing()
    vel = 20.0 / float(model.masses[0])
    nsteps = int(round(8.0 / vel / dt))
    v = np.array([vel])
    prev = model.evaluate(np.array([-4.0]), None)
    c = np.array([1.0, 0.0], dtype=complex)
    for k in range(1, nsteps + 1):
        snap = model.evaluate(np.array([-4.0 + vel * dt * k]), prev)
        kw = _diag_kw(coupling, snap.overlaps if coupling == "overlap" else None)
        h0, h1 = prev.hamiltonian, snap.hamiltonian
        c, _, _ = propagate_diagonal(
            c, None, None, h0, h1, prev.nacs, snap.nacs, v, v, dt, ns, **kw
        )
        prev = snap
    return np.abs(c) ** 2


def test_nacdr_and_overlap_agree_to_second_order_on_tully_sac():
    """Pre-registered: the ratio of successive |c|^2 differences lies in [3, 5] (midpoint)."""
    diffs = []
    for dt in (1.0, 0.5, 0.25):
        nac, ovl = _sac_populations("nacdr", dt), _sac_populations("overlap", dt)
        diffs.append(float(np.max(np.abs(nac - ovl))))
    ratios = [diffs[0] / diffs[1], diffs[1] / diffs[2]]
    for r in ratios:
        assert 3.0 <= r <= 5.0, (diffs, ratios)


def test_overlap_coupling_needs_overlaps():
    from gqteashmd.dynamics.propagators import propagate_diagonal

    e, nac, v, c = _rand_setup(np.random.default_rng(540))
    h = np.diag(e).astype(complex)
    args = (c, None, None, h, h, nac, nac, v, v, 0.5, 3)
    with pytest.raises(ValueError, match="overlaps"):
        propagate_diagonal(*args, coupling="overlap", overlaps=None, interpolation="midpoint")
    with pytest.raises(ValueError, match="coupling"):
        propagate_diagonal(*args, coupling="ddr", overlaps=None, interpolation="midpoint")
