import numpy as np

from gqteashmd.util.seeds import stage_seed_sequence, task_rng


def test_same_inputs_same_numbers():
    a = task_rng(20261106, "sampling", 0).standard_normal(5)
    b = task_rng(20261106, "sampling", 0).standard_normal(5)
    np.testing.assert_array_equal(a, b)


def test_streams_differ_by_stage_task_and_seed():
    base = task_rng(1, "sampling", 0).standard_normal(4)
    others = (task_rng(2, "sampling", 0), task_rng(1, "selection", 0), task_rng(1, "sampling", 1))
    for other in others:
        assert not np.allclose(base, other.standard_normal(4))


def test_task_streams_are_stable_children():
    # task i's stream must not depend on how many tasks exist
    seq = stage_seed_sequence(7, "sampling")
    child = seq.spawn(3)[1]
    np.testing.assert_array_equal(
        np.random.default_rng(child).standard_normal(3),
        task_rng(7, "sampling", 1).standard_normal(3),
    )
