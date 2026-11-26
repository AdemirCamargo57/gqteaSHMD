"""Helpers of tools/validation/censoring_report.py on synthetic trajectories."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "validation"))
import censoring_report as cr  # noqa: E402

sys.path.insert(0, str(ROOT / "tests" / "validation"))
from censoring_synthetic import synthetic_project  # noqa: E402

from gqteashmd.analysis.grid import Trajectory  # noqa: E402

BEFORE = json.loads(
    (ROOT / "tests" / "validation" / "data" / "censoring_report_before.json").read_text()
)
OLD = ("censored", "still_running", "complete_only", "hold_in_s0")


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


def _close(a, b, rel):
    if a is None or b is None:
        assert a is None and b is None
    else:
        assert a == pytest.approx(b, rel=rel, abs=1e-12)


@pytest.mark.parametrize("hot,key", [(False, "artificial"), (True, "sensitivity")])
def test_the_four_estimators_are_unchanged_by_the_ipcw_addition(hot, key):  # spec 4.1 item 8
    new = cr.artificial(synthetic_project(), repetitions=4, bootstrap=20, seed=5, hot=hot)
    old = BEFORE[key]
    for name in OLD:
        a, b = new["estimators"][name], old["estimators"][name]
        _close(a["max_bias_over_se"], b["max_bias_over_se"], 1e-9)
        _close(a["max_abs_bias"], b["max_abs_bias"], 1e-9)
        _close(a["tau_bias_fs"], b["tau_bias_fs"], 1e-6)
        _close(a["t0_bias_fs"], b["t0_bias_fs"], 1e-6)
        assert a["fits_failed"] == b["fits_failed"]
    assert "ipcw" in new["estimators"] and new["ipcw_fit"]["failures"] == 0
    assert ("ipcw_wins" in new) is hot and ("verdict_good" in new) is (not hot)


def test_the_reanalysis_is_unchanged_and_gains_ipcw():
    new = cr.reanalysis(synthetic_project(), seed=5)
    for name in ("complete", "censored"):
        _close(new[name]["tau_fs"], BEFORE["reanalysis"][name]["tau_fs"], 1e-6)
        _close(new[name]["t0_fs"], BEFORE["reanalysis"][name]["t0_fs"], 1e-6)
    block = new["ipcw"]
    assert block["fit"]["status"] == "ok"
    assert set(block) >= {"populations", "tau_fs", "tau_interval_fs", "max_weight"}
    assert block["max_weight"] >= 1.0


def test_control_and_gate_report_their_rules():
    p = synthetic_project()
    c = cr.control(p, repetitions=4, bootstrap=20, seed=5)
    assert set(c["estimators"]) == set(cr.ESTIMATORS)
    assert c["target_fraction"] == pytest.approx(c["loss_fraction"][0] / c["loss_fraction"][1])
    assert c["expected_lost_fraction"] == pytest.approx(c["target_fraction"], abs=1e-9)
    assert isinstance(c["passed"], bool) and c["beta_true"] == 1.0
    g = cr.gate(p, seed=5, resamples=40)
    assert g["fit"]["status"] == "ok" and g["successes"] <= 40
    assert isinstance(g["passed"], bool)


def test_ipcw_go_needs_all_three_parts():
    ok = {"control": {"passed": True}, "sensitivity": {"ipcw_wins": True}, "gate": {"passed": True}}
    assert cr.ipcw_go(ok)
    for part, key in (("control", "passed"), ("sensitivity", "ipcw_wins"), ("gate", "passed")):
        bad = json.loads(json.dumps(ok))
        bad[part][key] = False
        assert not cr.ipcw_go(bad)
