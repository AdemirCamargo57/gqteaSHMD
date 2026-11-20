import numpy as np
from tests.diagnosis.conftest import load_tool

from gqteashmd.analysis.grid import Trajectory
from gqteashmd.core.units import HARTREE_IN_EV

hr = load_tool("validation", "hop_energy_report")


def _traj(outcome, e_total, hop_to):
    n = len(e_total)
    return Trajectory(
        task="traj_00000",
        outcome=outcome,
        initial_state=1,
        grid_step=np.arange(n),
        ticks=np.zeros(n, dtype=np.int64),
        time_fs=0.5 * np.arange(n),
        active=np.ones(n, dtype=np.int64),
        hop_from=np.full(n, -1),
        hop_to=np.asarray(hop_to),
        frustrated=np.zeros(n, dtype=bool),
        e_total=np.asarray(e_total, dtype=float),
        populations=np.zeros((n, 3)),
    )


def test_record_energy_changes_split_by_hop():
    t = _traj("finished", [0.0, 1e-3, 1e-3, 4e-3], [-1, -1, 0, -1])
    hop, other = hr.record_energy_changes_eV(t)
    assert np.allclose(hop, [0.0])
    assert np.allclose(other, [1e-3 * HARTREE_IN_EV, 3e-3 * HARTREE_IN_EV])


def test_lost_after_hop():
    assert hr.lost_after_hop(_traj("overlap_failure", [0.0, 0.0], [-1, 0]))
    assert not hr.lost_after_hop(_traj("overlap_failure", [0.0, 0.0], [-1, -1]))
    assert not hr.lost_after_hop(_traj("finished", [0.0, 0.0], [-1, 0]))


def test_quantiles():
    q = hr.quantiles([1.0, 2.0, 3.0, 4.0])
    assert q["n"] == 4 and q["median"] == 2.5 and q["max"] == 4.0
    assert q["p95"] == float(np.percentile([1.0, 2.0, 3.0, 4.0], 95))
    assert hr.quantiles([]) == {"n": 0, "median": None, "p95": None, "max": None}


def test_outcome_changes():
    a = [{"task": "t0", "outcome": "finished"}, {"task": "t1", "outcome": "overlap_failure"}]
    b = [{"task": "t0", "outcome": "finished"}, {"task": "t1", "outcome": "finished"}]
    assert hr.outcome_changes(a, b) == [
        {"task": "t1", "current": "overlap_failure", "previous": "finished"}
    ]


def test_the_analysis_numbers_follow_the_complete_rule():
    # final review m1: under schema 2 the top level belongs to the estimator in use (censored by
    # default); the report compares complete-rule numbers, so it reads estimators["complete"]
    fit_c = {"t0_fs": 1.0, "t0_interval_fs": [0.5, 2.0], "tau_fs": 80.0, "tau_interval_fs": [9, 99]}
    fit_x = {"t0_fs": 3.0, "t0_interval_fs": [2.0, 4.0], "tau_fs": 50.0, "tau_interval_fs": [4, 60]}
    schema2 = {
        "schema_version": 2,
        "estimator": "censored",
        "n_selected": 100,
        "n_included": 100,
        "small_ensemble": False,
        "warnings": ["w"],
        "fit": fit_x,
        "estimators": {
            "complete": {"n": 67, "small_ensemble": False, "fit": fit_c},
            "censored": {"n": 100, "small_ensemble": False, "fit": fit_x},
        },
    }
    a = hr.analysis_numbers(schema2)
    assert a["n_included"] == 67 and a["tau_fs"] == 80.0 and a["t0_interval_fs"] == [0.5, 2.0]
    assert a["n_selected"] == 100 and a["warnings"] == ["w"] and a["small_ensemble"] is False
    assert a["estimator_read"] == "complete" and a["schema_version"] == 2
    schema1 = {
        "schema_version": 1,
        "n_selected": 100,
        "n_included": 67,
        "small_ensemble": False,
        "warnings": [],
        "fit": fit_c,
    }
    b = hr.analysis_numbers(schema1)
    assert b["n_included"] == 67 and b["tau_fs"] == 80.0 and b["estimator_read"] == "complete"
    censored = schema2["estimators"]["censored"]
    nothing = {**schema2, "estimators": {"complete": None, "censored": censored}}
    c = hr.analysis_numbers(nothing)  # no complete trajectory
    assert c["n_included"] == 0 and c["tau_fs"] is None
