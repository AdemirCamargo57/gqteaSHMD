import json

import numpy as np
import pytest
from tests.diagnosis.conftest import load_tool, make_rows, write_trajectory

fx = load_tool("diagnosis", "forensics")


def test_jump_without_hop_names_the_state_and_the_gap():
    e_total = [0.0, 0.01, 0.02, 0.32, 0.33]  # jump of 0.30 eV into record 3
    energies = [[0.0, 9.0], [0.0, 9.0], [0.0, 9.0], [0.05, 9.3], [0.05, 9.3]]
    rows = make_rows(e_total, energies)
    jumps = fx.log_jumps(rows)
    assert [j.record for j in jumps] == [3]
    j = jumps[0]
    assert j.delta_e_total_eV == pytest.approx(0.30, abs=1e-9)
    assert j.state == 1 and j.delta_e_state_eV == pytest.approx(0.3, abs=1e-9)
    assert j.gap_eV == pytest.approx(9.25, abs=1e-9)
    assert j.time_fs == pytest.approx(1.5, abs=1e-9)
    assert fx.classify_onset("energy_drift", jumps, fx.log_hops(rows)) == "jump_without_hop"


def test_threshold_is_strict():
    rows = make_rows([0.0, 0.1], [[0.0, 9.0], [0.0, 9.1]])
    assert fx.log_jumps(rows, threshold_eV=0.1 + 1e-9) == []
    assert len(fx.log_jumps(rows, threshold_eV=0.05)) == 1


def test_overlap_failure_after_a_hop():
    active = [1, 1, 0, 0]
    hop_to = [-1, -1, 0, -1]
    rows = make_rows(
        [0, 0, 0, 0], [[0, 2], [0, 1], [0, 0.5], [0, 0.4]], active=active, hop_to=hop_to
    )
    hops = fx.log_hops(rows)
    assert [(h.record, h.from_state, h.to_state) for h in hops] == [(2, 1, 0)]
    assert hops[0].gap_eV == pytest.approx(0.5, abs=1e-9)
    assert fx.classify_onset("overlap_failure", [], hops) == "failure_after_hop"
    assert fx.classify_onset("overlap_failure", [], []) == "overlap_failure_without_hop"


def test_onset_order():
    hop = fx.Hop(2, 1.0, 1, 0, 0.3)
    early = fx.Jump(1, 0.5, 0.2, 1, 0.2, 5.0, 0.9, False, 0, 1)
    late = fx.Jump(3, 1.5, 0.2, 0, 0.2, 0.3, 0.9, False, 0, 0)
    assert fx.classify_onset("electronic_failure", [early], [hop]) == "convergence_failure"
    assert fx.classify_onset("energy_drift", [early, late], [hop]) == "jump_without_hop"
    assert fx.classify_onset("finished", [late], [hop]) == "jump_after_hop"
    assert fx.classify_onset("finished", [], [hop]) == "none"
    assert set(fx.ONSETS) >= {"none", "jump_after_hop"}


def test_failed_geometry_is_the_verlet_position_update():
    q = np.array([0.0, 1.0])
    v = np.array([0.5, -0.5])
    g = np.array([2.0, -4.0])
    m = np.array([2.0, 4.0])
    out = fx.failed_geometry(q, v, g, m, 0.1)
    np.testing.assert_allclose(out, q + v * 0.1 - 0.5 * (g / m) * 0.01)


def test_electronic_log_tail_keeps_the_convergence_lines():
    text = "a\nSCF not converged\nb\nCASSCF failed\nc\nmacro iter 5 converged\n"
    assert fx.electronic_log_tail(text, n=2) == ["CASSCF failed", "macro iter 5 converged"]


def test_jsonable_replaces_non_finite_values():
    data = fx.jsonable(
        {"a": np.float64("nan"), "b": [np.int64(3), np.inf], "c": fx.Hop(0, 0.0, 1, 0, 0.1)}
    )
    assert data == {
        "a": None,
        "b": [3, None],
        "c": {"record": 0, "time_fs": 0.0, "from_state": 1, "to_state": 0, "gap_eV": 0.1},
    }
    json.dumps(data, allow_nan=False)


def test_trajectory_and_project_report(tmp_path):
    project = tmp_path / "run"
    tasks = project / "06_dynamics" / "tasks"
    rows = make_rows([0.0, 0.0, 0.5], [[0.0, 9.0], [0.0, 9.0], [0.0, 9.5]])
    write_trajectory(tasks / "traj_00000", rows, outcome="energy_drift", message="drift")
    write_trajectory(tasks / "traj_00001", make_rows([0, 0], [[0, 9], [0, 9]]), outcome="finished")
    (tasks / "traj_00000" / "electronic.log").write_text("SCF not converged\n")
    report = fx.trajectory_report(tasks / "traj_00000")
    assert report["task"] == "traj_00000"
    assert report["outcome"] == "energy_drift"
    assert report["onset"] == "jump_without_hop"
    assert report["n_records"] == 3
    assert report["jumps"][0]["record"] == 2
    assert report["electronic_log"] == ["SCF not converged"]
    assert len(report["last_records"]) == 3
    whole = fx.project_report(project)
    assert whole["n_trajectories"] == 2
    assert whole["onsets"]["jump_without_hop"] == 1 and whole["onsets"]["none"] == 1
    assert whole["outcomes"] == {"energy_drift": 1, "finished": 1}
    json.dumps(fx.jsonable(whole), allow_nan=False)


def test_find_stage_dir(tmp_path):
    (tmp_path / "03_sampling").mkdir()
    assert fx.find_stage_dir(tmp_path, "sampling") == tmp_path / "03_sampling"
    with pytest.raises(ValueError, match="dynamics"):
        fx.find_stage_dir(tmp_path, "dynamics")


def test_cli_writes_strict_json(tmp_path):
    cli = load_tool("diagnosis", "trajectory_forensics")
    project = tmp_path / "run"
    write_trajectory(
        project / "06_dynamics" / "tasks" / "traj_00000",
        make_rows([0, 0], [[0, 9], [0, 9]]),
        outcome="finished",
    )
    out = tmp_path / "f.json"
    assert cli.main([str(project), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["n_trajectories"] == 1


def test_trajectory_dirs_of_an_explicit_directory(tmp_path):
    for name in ("traj_00002", "traj_00000", "scratch"):
        (tmp_path / name).mkdir()
    (tmp_path / "summary.json").write_text("{}")
    dirs = fx.trajectory_dirs(tmp_path / "no_project", trajectories=tmp_path)
    assert [d.name for d in dirs] == ["traj_00000", "traj_00002"]
