import numpy as np
import pytest

from gqteashmd.analysis.grid import (
    Excluded,
    InconsistentData,
    Trajectory,
    horizon_steps,
    on_grid,
    select,
)


def traj(task, outcome, active, *, every=1, ticks=None, extra_grid=None, initial=1, n=3):
    """A trajectory with one record per grid step 0..len(active)-1 (ticks 0)."""
    k = len(active)
    grid = np.arange(k) if extra_grid is None else np.asarray(extra_grid)
    ticks = np.zeros(len(grid), dtype=int) if ticks is None else np.asarray(ticks)
    act = np.asarray(active)
    pops = np.zeros((len(grid), n))
    pops[np.arange(len(grid)), act] = 1.0
    return Trajectory(
        task,
        outcome,
        initial,
        grid,
        ticks,
        grid * 0.5,
        act,
        np.full(len(grid), -1),
        np.full(len(grid), -1),
        np.zeros(len(grid), bool),
        np.full(len(grid), -1.0),
        pops,
    )


def test_inclusion_rule():
    t = {
        "a": traj("a", "finished", [1, 1]),
        "b": traj("b", "stopped_in_s0", [1, 0]),
        "c": traj("c", "energy_drift", [1]),
        "d": None,
        "e": traj("e", "finished", [1, 1]),
    }
    included, excluded = select(
        ["a", "b", "c", "d", "e"], t, {"e": 0.5}, energy_jump_eV=0.1, exclude_energy_jumps=False
    )
    assert [x.task for x in included] == ["a", "b", "e"]
    assert excluded == [Excluded("c", "energy_drift", 0.0), Excluded("d", "task_failed", None)]
    included, excluded = select(["e"], t, {"e": 0.5}, energy_jump_eV=0.1, exclude_energy_jumps=True)
    assert included == [] and excluded == [Excluded("e", "energy_jump", 0.5, 0.5)]


def test_horizon_is_the_shortest_finished_trajectory_and_at_most_n_total():
    a = traj("a", "finished", [1] * 9)
    b = traj("b", "finished", [1] * 6)
    s = traj("s", "stopped_in_s0", [1, 0])
    assert horizon_steps([a, b, s], n_total=8) == 5
    assert horizon_steps([a], n_total=4) == 4
    assert horizon_steps([s], n_total=4) == 4


def test_grid_holds_stopped_trajectories_in_s0_and_lists_cut_ones():
    a = traj("a", "finished", [1, 1, 2, 2, 2])
    s = traj("s", "stopped_in_s0", [1, 0])
    g = on_grid([a, s], output_every=2, dt_fs=0.5, n_total=3, n_states=3)
    np.testing.assert_array_equal(g.grid_steps, [0, 2])
    np.testing.assert_allclose(g.time_fs, [0.0, 1.0])
    np.testing.assert_array_equal(g.active, [[1, 2], [1, 0]])
    np.testing.assert_array_equal(g.populations[1, 1], [1.0, 0.0, 0.0])
    assert g.horizon_steps == 3 and g.horizon_fs == 1.5 and g.cut == ("a",)
    assert g.tasks == ("a", "s")


def test_substep_records_are_not_grid_points():
    a = traj("a", "finished", [1, 1, 2], extra_grid=[0, 0, 1], ticks=[0, 4, 0])
    g = on_grid([a], output_every=1, dt_fs=0.5, n_total=1, n_states=3)
    np.testing.assert_array_equal(g.active, [[1, 2]])


def test_a_missing_grid_record_is_inconsistent():
    a = traj("a", "finished", [1, 1], extra_grid=[0, 2])
    with pytest.raises(InconsistentData, match="a"):
        on_grid([a], output_every=1, dt_fs=0.5, n_total=2, n_states=3)


def test_non_finite_populations_are_inconsistent():
    a = traj("a", "finished", [1, 1])
    a.populations[1, 0] = np.nan
    with pytest.raises(InconsistentData, match="non-finite"):
        on_grid([a], output_every=1, dt_fs=0.5, n_total=1, n_states=3)
