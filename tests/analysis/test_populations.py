import numpy as np

from gqteashmd.analysis.populations import (
    bootstrap_indices,
    bootstrap_intervals,
    classical_populations,
    consistency,
    quantum_populations,
    wilson_intervals,
)
from gqteashmd.dynamics.ensemble import wilson_interval


def test_classical_populations_count_active_states():
    active = np.array([[1, 1, 0], [1, 2, 0]])
    p, counts = classical_populations(active, 3)
    np.testing.assert_array_equal(counts, [[0, 2, 0], [0, 1, 1], [2, 0, 0]])
    np.testing.assert_allclose(p, counts / 2)


def test_wilson_matches_the_scalar_formula():
    counts = np.array([[0, 3, 10], [7, 1, 2]])
    lo, hi = wilson_intervals(counts, 10)
    for idx in np.ndindex(counts.shape):
        assert (lo[idx], hi[idx]) == wilson_interval(int(counts[idx]), 10)


def test_bootstrap_is_reproducible_and_brackets_the_mean():
    rng = np.random.default_rng(1)
    x = rng.random((50, 4, 2))
    i1 = bootstrap_indices(np.random.default_rng(5), 50, 300)
    i2 = bootstrap_indices(np.random.default_rng(5), 50, 300)
    np.testing.assert_array_equal(i1, i2)
    lo, hi = bootstrap_intervals(x, i1)
    mean = quantum_populations(x)
    assert lo.shape == mean.shape and np.all(lo <= mean) and np.all(mean <= hi)


def test_bootstrap_interval_coverage_on_synthetic_ensembles():
    rng = np.random.default_rng(20261111)
    true_mean = 2.0 / 7.0  # Beta(2, 5)
    covered = 0
    repetitions = 400
    for _ in range(repetitions):
        x = rng.beta(2.0, 5.0, size=(100, 1, 1))
        lo, hi = bootstrap_intervals(x, bootstrap_indices(rng, 100, 500))
        covered += bool(lo[0, 0] <= true_mean <= hi[0, 0])
    print(f"observed coverage {covered / repetitions}")
    assert 0.90 <= covered / repetitions <= 0.99, covered


def test_consistency_is_the_largest_difference_per_state():
    p = np.array([[1.0, 0.0], [0.5, 0.5]])
    rho = np.array([[0.9, 0.1], [0.5, 0.5]])
    np.testing.assert_allclose(consistency(p, rho), [0.1, 0.1])
