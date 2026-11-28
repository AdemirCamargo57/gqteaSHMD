"""Helpers of tools/validation/ngt_term_report.py (spec 5.2, plan ruling PR7): the singlet-singlet
NAC term of the diagonal-basis force that ``gradient_transformation = "none"`` leaves out,
checked on synthetic snapshots against a hand computation (no PySCF, no run)."""

import numpy as np
from tests.diagnosis.conftest import load_tool

from gqteashmd.dynamics.records import ElectronicSnapshot
from gqteashmd.trajectory.spin_expansion import (
    expand_energies,
    expand_gradients,
    expanded_size,
)

ngt = load_tool("validation", "ngt_term_report")


def _unitary(n_states, rng):
    """A unitary U from a random complex Hermitian matrix (any U works for the formula)."""
    a = rng.normal(size=(n_states, n_states)) + 1j * rng.normal(size=(n_states, n_states))
    return np.linalg.eigh(a + a.conj().T)[1]


def _snapshot(n, m, f, rng):
    """Expanded energies and gradients as BackendModel builds them (zero NACs, as with triplets)."""
    energies_sf = np.sort(rng.normal(size=n + m))
    gradients_sf = rng.normal(size=(n + m, f))
    big_n = expanded_size(n, m)
    return ElectronicSnapshot(
        energies=expand_energies(energies_sf, n, m),
        gradients=expand_gradients(gradients_sf, n, m),
        nacs=np.zeros((big_n, big_n, f)),
        hamiltonian=np.diag(expand_energies(energies_sf, n, m)).astype(complex),
        overlaps=None,
    )


def _singlet_nacs(n, f, rng):
    return {(i, j): rng.normal(size=f) for i in range(n) for j in range(i + 1, n)}


def _hand_dropped(snapshot, nacs, u, active, n):
    """Re[(U^dagger G_offdiag U)]_{lambda lambda}, G_ab = (E_b - E_a) K_ab, singlet pairs only,
    written out element by element."""
    e = snapshot.energies
    total = np.zeros(snapshot.gradients.shape[1])
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            k_ab = nacs[(a, b)] if a < b else -nacs[(b, a)]
            weight = np.conj(u[a, active]) * u[b, active]
            total += np.real(weight) * (e[b] - e[a]) * k_ab
    return total


def test_the_dropped_term_equals_the_hand_computation():
    rng = np.random.default_rng(20261128)
    n, m, f = 3, 2, 6
    snapshot = _snapshot(n, m, f, rng)
    nacs = _singlet_nacs(n, f, rng)
    u = _unitary(expanded_size(n, m), rng)
    g_off = ngt.singlet_nac_tensor(snapshot.energies, nacs, n, m)
    for active in range(expanded_size(n, m)):
        expected = _hand_dropped(snapshot, nacs, u, active, n)
        assert np.allclose(ngt.dropped_term(g_off, u, active), expected, rtol=1e-12, atol=1e-13)


def test_two_singlets_give_the_closed_form():
    # n = 2: Re[U^dag G U]_ll = 2 Re(conj(U_0l) U_1l) (E_1 - E_0) K_01
    rng = np.random.default_rng(7)
    n, m, f = 2, 2, 6
    snapshot = _snapshot(n, m, f, rng)
    nacs = _singlet_nacs(n, f, rng)
    u = _unitary(expanded_size(n, m), rng)
    g_off = ngt.singlet_nac_tensor(snapshot.energies, nacs, n, m)
    e = snapshot.energies
    for active in range(expanded_size(n, m)):
        coherence = 2.0 * np.real(np.conj(u[0, active]) * u[1, active])
        expected = coherence * (e[1] - e[0]) * nacs[(0, 1)]
        assert np.allclose(ngt.dropped_term(g_off, u, active), expected, rtol=1e-12, atol=1e-13)


def test_the_tensor_has_singlet_entries_only_and_is_symmetric():
    rng = np.random.default_rng(3)
    n, m, f = 3, 2, 6
    snapshot = _snapshot(n, m, f, rng)
    nacs = _singlet_nacs(n, f, rng)
    g_off = ngt.singlet_nac_tensor(snapshot.energies, nacs, n, m)
    big_n = expanded_size(n, m)
    assert g_off.shape == (big_n, big_n, f)
    assert np.array_equal(g_off, np.swapaxes(g_off, 0, 1))  # (E_b - E_a) K_ab is symmetric
    assert not np.any(g_off[n:, :])
    assert not np.any(g_off[:, n:])
    assert not np.any(g_off[np.arange(big_n), np.arange(big_n)])
    e = snapshot.energies
    assert np.array_equal(g_off[0, 2], (e[2] - e[0]) * nacs[(0, 2)])


def test_the_used_force_is_the_population_weighted_gradient():
    # with "none" G^MCH is diagonal: -Re[U^dag G U]_ll = -sum_k |U_kl|^2 g_k
    rng = np.random.default_rng(11)
    n, m, f = 2, 2, 6
    snapshot = _snapshot(n, m, f, rng)
    u = _unitary(expanded_size(n, m), rng)
    for active in range(expanded_size(n, m)):
        weights = np.abs(u[:, active]) ** 2
        expected = -(weights[:, None] * snapshot.gradients).sum(axis=0)
        assert np.allclose(ngt.used_force(snapshot, u, active), expected, rtol=1e-12, atol=1e-13)


def test_without_u_nothing_is_dropped():
    rng = np.random.default_rng(5)
    n, m, f = 2, 2, 6
    snapshot = _snapshot(n, m, f, rng)
    g_off = ngt.singlet_nac_tensor(snapshot.energies, _singlet_nacs(n, f, rng), n, m)
    assert not np.any(ngt.dropped_term(g_off, None, 1))


def test_step_entry_reports_norms_and_their_ratio():
    rng = np.random.default_rng(13)
    n, m, f = 2, 2, 6
    snapshot = _snapshot(n, m, f, rng)
    nacs = _singlet_nacs(n, f, rng)
    u = _unitary(expanded_size(n, m), rng)
    active = 1
    entry = ngt.step_entry(3, snapshot, u, active, nacs, n, m)
    dropped = _hand_dropped(snapshot, nacs, u, active, n)
    used = -(np.abs(u[:, active])[:, None] ** 2 * snapshot.gradients).sum(axis=0)
    assert entry["step"] == 3
    assert entry["active"] == active
    assert np.isclose(entry["dropped_norm"], np.linalg.norm(dropped), rtol=1e-12)
    assert np.isclose(entry["force_norm"], np.linalg.norm(used), rtol=1e-12)
    assert np.isclose(entry["ratio"], np.linalg.norm(dropped) / np.linalg.norm(used), rtol=1e-12)
    assert np.isclose(sum(entry["singlet_weights"]), np.sum(np.abs(u[:n, active]) ** 2))


def test_summary_gives_max_median_and_the_step_of_the_max():
    entries = [{"step": k, "ratio": r} for k, r in enumerate([0.1, 0.4, 0.2, 0.3])]
    summary = ngt.summarise(entries)
    assert summary["max_ratio"] == 0.4
    assert summary["step_of_max"] == 1
    assert np.isclose(summary["median_ratio"], 0.25)
    assert summary["n_steps"] == 4
