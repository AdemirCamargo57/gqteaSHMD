"""Helpers of tools/validation/censoring_report.py on synthetic trajectories."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "validation"))
import censoring_report as cr  # noqa: E402

from gqteashmd.analysis.grid import Trajectory  # noqa: E402


def _traj(task, outcome, active):
    n = len(active)
    return Trajectory(
        task,
        outcome,
        int(active[0]),
        np.arange(n),
        np.zeros(n, dtype=np.int64),
        0.5 * np.arange(n),
        np.asarray(active),
        np.full(n, -1),
        np.full(n, -1),
        np.zeros(n, dtype=bool),
        np.zeros(n),
        np.eye(2)[np.asarray(active)],
    )


def test_loss_delays_are_measured_from_the_first_s0_record():
    lost = _traj("a", "overlap_failure", [1, 1, 0, 0, 0, 0])  # S0 from 1.0 fs, ends 2.5 fs
    kept = _traj("b", "finished", [1, 0, 0])
    assert cr.loss_delays_fs([lost, kept]) == [1.5]
    assert cr.reaching_s0_fraction([lost, kept]) == (1, 2)


def test_impose_censoring_cuts_after_the_first_s0_arrival():
    t = _traj("a", "finished", [1, 1, 0, 0, 0, 0, 0])
    assert cr.censor_step_for(t, delay_fs=1.0, dt_fs=0.5, output_every=1) == 4  # 1.0 + 1.0 fs
    assert cr.censor_step_for(t, delay_fs=99.0, dt_fs=0.5, output_every=1) is None
    assert cr.censor_step_for(_traj("b", "finished", [1, 1, 1]), 1.0, 0.5, 1) is None


def test_hold_in_s0_fills_the_censored_part_with_s0():
    active = np.array([[1, 0, 0, 1]])
    observed = np.array([[True, True, False, False]])
    np.testing.assert_array_equal(cr.hold_in_s0(active, observed), [[1, 0, 0, 0]])


def test_bias_ratio_skips_zero_units():
    mean = np.array([[0.5, 0.5], [0.6, 0.4]])
    truth = np.array([[0.5, 0.5], [0.5, 0.5]])
    unit = np.array([[0.0, 0.0], [0.05, 0.05]])
    assert cr.max_bias_ratio(mean, truth, unit) == pytest.approx(2.0)


def test_poisson_interval_brackets_the_rate():
    lo, hi = cr.poisson_rate_interval(4, 2.0)
    assert lo < 2.0 < hi
    assert cr.poisson_rate_interval(0, 2.0)[0] == 0.0
