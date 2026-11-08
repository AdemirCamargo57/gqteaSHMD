import numpy as np

from gqteashmd.dynamics.integrators import acceleration, verlet_positions, verlet_velocities


def _run(q, v, steps, dt, k=1.0, m=1.0):
    masses = np.array([m])
    a = acceleration(k * q, masses)
    traj = []
    for _ in range(steps):
        q = verlet_positions(q, v, a, dt)
        a_new = acceleration(k * q, masses)
        v = verlet_velocities(v, a, a_new, dt)
        a = a_new
        traj.append(0.5 * m * v @ v + 0.5 * k * q @ q)
    return q, v, traj


def test_acceleration():
    assert np.allclose(acceleration(np.array([2.0, 4.0]), np.array([2.0, 4.0])), [-1.0, -1.0])


def test_energy_conservation():
    period = 2 * np.pi
    dt = 0.05 * period / (2 * np.pi)
    q0, v0 = np.array([1.0]), np.array([0.0])
    _, _, energies = _run(q0, v0, 1000, dt)
    e0 = 0.5
    per = int(round(period / dt))
    # bounded O(dt^2) oscillation (max ~ (w dt)^2/8) and no secular drift
    assert max(abs(e - e0) for e in energies) / e0 < 1e-3
    drift = abs(np.mean(energies[-per:]) - np.mean(energies[:per])) / e0
    assert drift < 1e-4


def test_time_reversible():
    q0, v0 = np.array([1.0]), np.array([0.3])
    q1, v1, _ = _run(q0, v0, 300, 0.05)
    q2, v2, _ = _run(q1, -v1, 300, 0.05)
    assert np.allclose(q2, q0, atol=1e-12) and np.allclose(-v2, v0, atol=1e-12)
