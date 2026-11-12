"""The workflow through `dynamics` with PySCF: water HF/STO-3G (CIS), P7b Task 9 checks (e), (f)."""

import contextlib
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from gqteashmd.cli import main
from gqteashmd.config.loader import load_config
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.trajectory.outcomes import TrajectoryOutcome

pytestmark = pytest.mark.pyscf
# the workflow input of tests/test_e2e_excitations.py (EXTRA there), followed by [dynamics]
EXTRA = (
    '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 3\n'
    "[sampling]\nn_samples = 4\n[excitations]\nsamples_per_task = 2\n"
    "[selection]\nn_trajectories = 2\nallow_fewer = true\n"
)
OUTCOMES = {o.value for o in TrajectoryOutcome}


def _input(tmp_path, write_input, tmax_fs):
    return write_input(tmp_path, extra=EXTRA + f"[dynamics]\ntmax_fs = {tmax_fs}\ndt_fs = 0.25\n")


def _layout(path):
    return ProjectLayout(load_config(path).config.project.directory)


def _selected(layout):
    data = json.loads((layout.stage_dir(5, "selection") / "trajectories.json").read_text())
    return sorted(f"traj_{int(e['index']):05d}" for e in data["trajectories"])


def _summary(layout):
    return json.loads((layout.stage_dir(6, "dynamics") / "dynamics_summary.json").read_text())


def _check_rows(layout):
    summary = _summary(layout)
    rows = summary["trajectories"]
    report = [(r["task"], r["outcome"], r["t_final_fs"], r["max_drift_eV"]) for r in rows]
    print("dynamics_summary rows (task, outcome, t_final_fs, max_drift_eV):", report)
    assert sorted(r["task"] for r in rows) == _selected(layout), report
    assert summary["n_results"] == len(rows) and all(r["outcome"] in OUTCOMES for r in rows), report
    return rows


def test_workflow_runs_dynamics_resumes_and_extends(tmp_path, write_input):
    start = time.perf_counter()
    path = _input(tmp_path, write_input, "1.0")
    assert main(["run", str(path)]) == 0
    print(f"first run: {time.perf_counter() - start:.1f} s")
    layout = _layout(path)
    stage = layout.stage_dir(6, "dynamics")
    _check_rows(layout)
    analysis = json.loads((layout.stage_dir(7, "analysis") / "analysis.json").read_text())
    included = {
        r["task"]
        for r in _summary(layout)["trajectories"]
        if r["outcome"] in ("finished", "stopped_in_s0")
    }
    assert set(analysis["included"]) == included and analysis["stage_failed"] is False
    assert analysis["horizon_fs"] == pytest.approx(1.0)
    assert (layout.stage_dir(7, "analysis") / "populations.csv").exists()
    logs = sorted(stage.glob("tasks/traj_*/steps.log"))
    assert [p.parent.name for p in logs] == _selected(layout)
    before = {p: p.stat().st_mtime_ns for p in logs}
    summary_bytes = (stage / "dynamics_summary.json").read_bytes()

    assert main(["run", str(path)]) == 0  # up to date: nothing recomputed or rewritten
    assert {p: p.stat().st_mtime_ns for p in logs} == before
    assert (stage / "dynamics_summary.json").read_bytes() == summary_bytes

    start = time.perf_counter()
    path = _input(tmp_path, write_input, "1.5")  # O16: raising tmax reopens finished trajectories
    assert main(["run", str(path)]) == 0
    print(f"extension run: {time.perf_counter() - start:.1f} s")
    rows = _check_rows(layout)
    finished = [r for r in rows if r["outcome"] == TrajectoryOutcome.FINISHED]
    assert finished, rows  # the extension check below must not be vacuous
    assert "traj_00000" in {r["task"] for r in finished}, rows  # measured: finished at 1.5 fs
    assert all(r["t_final_fs"] == pytest.approx(1.5) for r in finished), rows
    analysis = json.loads((layout.stage_dir(7, "analysis") / "analysis.json").read_text())
    assert analysis["horizon_fs"] == pytest.approx(1.5)  # recomputed after the reopen (P8)


@pytest.mark.slow
def test_a_killed_controller_resumes_the_dynamics(tmp_path, write_input):
    path = _input(tmp_path, write_input, "2.0")
    layout = _layout(path)
    marker = layout.task_dir(layout.stage_dir(6, "dynamics"), "traj_00000") / "restart" / "current"
    start = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-m", "gqteashmd", "run", str(path)],
        env=dict(os.environ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 900
        while not marker.exists():
            if proc.poll() is not None:
                pytest.fail(f"the run ended (exit {proc.returncode}) before {marker} existed")
            if time.monotonic() > deadline:
                pytest.fail(f"{marker} did not appear within 900 s")
            time.sleep(0.05)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)  # the controller's group; workers have their own
        proc.wait()
    print(f"killed after {time.perf_counter() - start:.1f} s")
    start = time.perf_counter()
    assert main(["run", str(path)]) == 0
    print(f"second run: {time.perf_counter() - start:.1f} s")
    _check_rows(layout)
    analysis = json.loads((layout.stage_dir(7, "analysis") / "analysis.json").read_text())
    assert analysis["stage_failed"] is False
