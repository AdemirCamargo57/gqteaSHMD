import pytest
from tests.diagnosis.conftest import load_tool

tool = load_tool("diagnosis", "convergence_probe")


def _probe(converged_rungs, dE=1e-8, dg=1e-6):
    names = ("traj_00000_failed_step", "traj_00007_full_step", "traj_00016_sharc_17fs")
    return {
        "geometries": {
            name: {
                "more_iterations": {"rung": converged_rungs.get(("more_iterations", name))},
                "loose": {"rung": converged_rungs.get(("loose", name))},
            }
            for name in names
        },
        "accuracy": [{"max_abs_dE_Eh": dE, "max_abs_dg_Eh_per_bohr": dg}] * 5,
    }


def test_nothing_converges_enables_nothing():
    decision = tool.f4_decision(_probe({}))
    assert decision["enabled"] == [] and not decision["more_iterations"]


def test_more_iterations_needs_one_geometry():
    rungs = {("more_iterations", "traj_00007_full_step"): "previous orbitals, more iterations"}
    assert tool.f4_decision(_probe(rungs))["enabled"] == ["previous orbitals, more iterations"]


def test_a_lower_rung_does_not_count():
    rungs = {("more_iterations", "traj_00007_full_step"): "SCF orbitals"}
    assert tool.f4_decision(_probe(rungs))["enabled"] == []


def test_loose_needs_two_geometries_and_accuracy():
    two = {
        ("loose", "traj_00000_failed_step"): "loose tolerances",
        ("loose", "traj_00007_full_step"): "loose tolerances",
    }
    assert tool.f4_decision(_probe(two))["enabled"] == ["loose tolerances"]
    assert tool.f4_decision(_probe(two, dg=2e-4))["enabled"] == []
    assert tool.f4_decision(_probe(two, dE=2e-6))["enabled"] == []
    one = {("loose", "traj_00007_full_step"): "loose tolerances"}
    assert tool.f4_decision(_probe(one))["enabled"] == []


def test_both_enabled_in_ladder_order():
    rungs = {
        ("more_iterations", "traj_00016_sharc_17fs"): "previous orbitals, more iterations",
        ("loose", "traj_00000_failed_step"): "loose tolerances",
        ("loose", "traj_00016_sharc_17fs"): "loose tolerances",
    }
    assert tool.f4_decision(_probe(rungs))["enabled"] == [
        "previous orbitals, more iterations",
        "loose tolerances",
    ]


def test_max_abs_diff():
    assert tool.max_abs_diff([1.0, 2.0], [1.5, 1.0]) == pytest.approx(1.0)
