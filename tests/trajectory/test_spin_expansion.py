import numpy as np
import pytest

from gqteashmd.dynamics.spin import expanded_states
from gqteashmd.trajectory import spin_expansion as sx


@pytest.mark.parametrize(("n", "m"), [(2, 0), (2, 1), (3, 3), (1, 2)])
def test_indices_follow_the_canonical_order(n, m):
    labels = [s.label() for s in expanded_states((n, 0, m))]
    assert sx.expanded_size(n, m) == len(labels)
    for j in range(m):
        for mi, ms in enumerate(("-1", "0", "+1")):
            k = sx.expanded_index(n, m, n + j, mi)
            assert labels[k] == f"T{j + 1}({ms})"
            assert sx.spin_free_index(n, m, k) == n + j
    for i in range(n):
        assert sx.expanded_index(n, m, i) == i and sx.spin_free_index(n, m, i) == i


def test_energies_and_gradients_repeat_each_triplet_three_times():
    e = np.array([0.0, 0.1, 0.2, 0.3])  # S0 S1 T1 T2
    g = np.arange(8.0).reshape(4, 2)
    ee, gg = sx.expand_energies(e, 2, 2), sx.expand_gradients(g, 2, 2)
    assert ee.tolist() == [0.0, 0.1, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3]
    assert gg.shape == (8, 2) and np.array_equal(gg[[2, 4, 6]], np.repeat(g[[2]], 3, axis=0))


def test_overlaps_are_block_diagonal_with_kron_triplet_blocks():
    rng = np.random.default_rng(0)
    s = np.zeros((4, 4))
    s[:2, :2], s[2:, 2:] = rng.normal(size=(2, 2)), rng.normal(size=(2, 2))
    big = sx.expand_overlaps(s, 2, 2)
    assert np.array_equal(big[:2, :2], s[:2, :2])
    assert np.array_equal(big[2:, 2:], np.kron(np.eye(3), s[2:, 2:]))
    assert np.all(big[:2, 2:] == 0) and np.all(big[2:, :2] == 0)


def test_overlaps_refuse_nonzero_singlet_triplet_blocks():
    s = np.eye(4)
    s[0, 3] = 1e-3
    with pytest.raises(ValueError, match="singlet-triplet"):
        sx.expand_overlaps(s, 2, 2)


def test_spin_free_populations_sum_over_ms():
    p = np.array([[0.1, 0.2, 0.05, 0.1, 0.05, 0.2, 0.1, 0.2]])
    assert np.allclose(sx.spin_free_populations(p, 2, 2), [[0.1, 0.2, 0.2, 0.5]])


def test_no_triplets_is_the_identity():
    e, g, s = np.array([0.0, 0.1]), np.ones((2, 3)), np.eye(2)
    assert np.array_equal(sx.expand_energies(e, 2, 0), e)
    assert np.array_equal(sx.expand_gradients(g, 2, 0), g)
    assert np.array_equal(sx.expand_overlaps(s, 2, 0), s)
