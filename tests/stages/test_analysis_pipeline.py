"""The analysis stage under the orchestrator: canned LVC upstream, real dynamics and analysis."""

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

from gqteashmd.cli import main
from gqteashmd.testing import canned_lvc
from gqteashmd.trajectory.store import read_output_h5

LVC = Path(__file__).resolve().parents[1] / "data" / "lvc" / "lvc_singlets.template"
CANNED = "gqteashmd.testing.pipelines:canned_analysis"
FAST = "poll_interval_s = 0.05\nstop_grace_period_s = 2.0"
ELECTRONIC = '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 3\n'


@pytest.fixture(autouse=True)
def _template(monkeypatch):
    monkeypatch.setenv(canned_lvc.LVC_TEMPLATE_ENV, str(LVC))


def _input(tmp_path, write_input, tmax="2.0", extra="", extra_top=""):
    return write_input(
        tmp_path,
        execution=FAST,
        extra=extra_top + ELECTRONIC + f"[dynamics]\ntmax_fs = {tmax}\ndt_fs = 0.25\n" + extra,
    )


def _main(path, *args):
    return main(["run", str(path), "--pipeline", CANNED, *args])


def _dyn(tmp_path):
    return tmp_path / "run" / "03_dynamics"


def _ana(tmp_path):
    return tmp_path / "run" / "04_analysis"


def _analysis(tmp_path):
    return json.loads((_ana(tmp_path) / "analysis.json").read_text())


def test_stage_populations_equal_a_direct_recomputation(tmp_path, write_input):
    path = _input(tmp_path, write_input)  # tmax 2.0, dt 0.25
    assert _main(path) == 0
    trajs = [read_output_h5(p) for p in sorted(_dyn(tmp_path).glob("tasks/traj_*/output.h5"))]
    assert len(trajs) == 3
    full = [t["ticks"] == 0 for t in trajs]
    active = np.stack([t["active"][f] for t, f in zip(trajs, full, strict=True)])
    pops = np.stack([t["populations"][f] for t, f in zip(trajs, full, strict=True)])
    with h5py.File(_ana(tmp_path) / "populations.h5") as f:
        for j in range(pops.shape[2]):
            np.testing.assert_array_equal(f["classical"][()][:, j], (active == j).mean(axis=0))
        np.testing.assert_allclose(f["quantum"][()], pops.mean(axis=0), rtol=0, atol=1e-15)
    summary = _analysis(tmp_path)
    assert summary["n_included"] == 3 and summary["stage_failed"] is False
    for name in ("populations.csv", "hops.csv", "report.md"):
        assert (_ana(tmp_path) / name).exists()


def test_without_losses_the_censored_default_equals_the_complete_rule(tmp_path, write_input):
    # censored analysis spec 4.1 item 1, on the canned pipeline (every trajectory finishes)
    assert _main(_input(tmp_path, write_input)) == 0
    summary = _analysis(tmp_path)
    assert summary["estimator"] == "censored" and summary["schema_version"] == 3
    assert summary["n_censored"] == 0 and summary["n_complete"] == 3
    complete = summary["estimators"]["complete"]["table"]
    assert len(summary["table"]) == len(complete) > 1
    for row, other in zip(summary["table"], complete, strict=True):
        assert row["time_fs"] == other["time_fs"]
        for state, value in row["P"].items():
            assert abs(value - other["P"][state]) <= 1e-12, (row["time_fs"], state)


def test_raising_tmax_recomputes_the_analysis_in_one_run(tmp_path, write_input):
    assert _main(_input(tmp_path, write_input)) == 0
    assert _analysis(tmp_path)["horizon_fs"] == pytest.approx(2.0)
    assert _main(_input(tmp_path, write_input, tmax="3.0")) == 0
    assert _analysis(tmp_path)["horizon_fs"] == pytest.approx(3.0)
    superseded = tmp_path / "run" / ".gqteashmd" / "superseded"
    assert any(p.name.endswith("_04_analysis") for p in superseded.iterdir())


def test_pause_after_dynamics_pauses_before_the_analysis(tmp_path, write_input):
    path = _input(tmp_path, write_input, extra_top='[workflow]\npause_after = ["dynamics"]\n')
    assert _main(path) == 5 and not (_ana(tmp_path) / "analysis.json").exists()
    assert _main(path) == 0 and (_ana(tmp_path) / "analysis.json").exists()


def test_disabled_analysis_is_skipped(tmp_path, write_input):
    assert _main(_input(tmp_path, write_input, extra="[analysis]\nenabled = false\n")) == 0
    assert not (_ana(tmp_path) / "analysis.json").exists()


def test_analysis_without_dynamics_is_skipped(tmp_path, write_input):
    path = write_input(tmp_path, execution=FAST, extra=ELECTRONIC + "[analysis]\n")
    assert _main(path) == 0
    assert not (_ana(tmp_path) / "analysis.json").exists()


def test_changing_an_analysis_key_is_stale_and_rerun_from_analysis_recomputes(
    tmp_path, write_input
):
    assert _main(_input(tmp_path, write_input)) == 0
    changed = _input(tmp_path, write_input, extra="[analysis]\nn_bootstrap = 200\n")
    assert _main(changed) == 3
    assert _main(changed, "--rerun-from", "analysis") == 0
    assert _analysis(tmp_path)["settings"]["n_bootstrap"] == 200


def test_warn_excluded_fraction_is_finalize_only(tmp_path, write_input):
    assert _main(_input(tmp_path, write_input)) == 0
    changed = _input(tmp_path, write_input, extra="[analysis]\nwarn_excluded_fraction = 0.5\n")
    assert _main(changed) == 0  # not stale


def _logs(tmp_path):
    return {p: p.stat().st_mtime_ns for p in sorted(_dyn(tmp_path).glob("tasks/traj_*/steps.log"))}


def test_lowering_tmax_reopens_the_analysis_and_cuts_every_trajectory(tmp_path, write_input):
    assert _main(_input(tmp_path, write_input, tmax="3.0")) == 0
    assert _analysis(tmp_path)["horizon_fs"] == pytest.approx(3.0)
    logs = _logs(tmp_path)
    assert len(logs) == 3
    lowered = _input(tmp_path, write_input, tmax="2.0")
    assert _main(lowered) == 0
    summary = _analysis(tmp_path)
    assert summary["horizon_fs"] == pytest.approx(2.0)
    assert summary["cut"] == ["traj_00000", "traj_00001", "traj_00002"]
    assert _logs(tmp_path) == logs  # the dynamics did not run again
    before = (_ana(tmp_path) / "analysis.json").stat().st_mtime_ns
    assert _main(lowered) == 0  # a second run at 2.0 is a no-op for the analysis
    assert (_ana(tmp_path) / "analysis.json").stat().st_mtime_ns == before


@pytest.mark.slow
def test_a_killed_controller_resumes_through_the_analysis(tmp_path, write_input):
    ref, run = tmp_path / "ref", tmp_path / "kill"
    ref.mkdir()
    run.mkdir()
    start = time.perf_counter()
    assert _main(_input(ref, write_input, tmax="25.0")) == 0
    print(f"reference canned run: {time.perf_counter() - start:.1f} s")
    path = _input(run, write_input, tmax="25.0")
    marker = run / "run" / "03_dynamics" / "tasks" / "traj_00000" / "restart" / "current"
    env = dict(os.environ)
    proc = subprocess.Popen(
        [sys.executable, "-m", "gqteashmd", "run", str(path), "--pipeline", CANNED],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 120
        while not marker.exists():
            if proc.poll() is not None:
                pytest.fail(f"the run ended (exit {proc.returncode}) before {marker} existed")
            if time.monotonic() > deadline:
                pytest.fail(f"{marker} did not appear within 120 s")
            time.sleep(0.01)
        assert proc.poll() is None, "the controller finished before it could be killed"
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    assert _main(path) == 0
    a, b = (r / "run" / "04_analysis" / "populations.csv" for r in (ref, run))
    assert a.read_bytes() == b.read_bytes()


# ---------------------------------------------------------------- triplets (sub-project 3)
LVC_SOC = Path(__file__).resolve().parents[1] / "data" / "lvc" / "lvc_soc.template"
ELECTRONIC_SOC = (
    '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 2\n'
    "triplets = 2\n"
)
SOC_LABELS = ["S0", "S1", "T1", "T2"]


def _kind(i, j):
    return f"{SOC_LABELS[i][0]}-{SOC_LABELS[j][0]}"


def test_the_canned_spin_orbit_pipeline_through_the_analysis(tmp_path, write_input, monkeypatch):
    # plan ruling PR5: lvc_soc.template, 2 singlets + 2 triplets, four trajectories from S1
    monkeypatch.setenv(canned_lvc.LVC_TEMPLATE_ENV, str(LVC_SOC))
    monkeypatch.setenv(canned_lvc.LVC_SOC_ENV, "1")
    path = write_input(
        tmp_path,
        execution=FAST,
        extra=ELECTRONIC_SOC
        + '[dynamics]\ntmax_fs = 20.0\ndt_fs = 0.25\ngradient_transformation = "none"\n',
    )
    assert _main(path) == 0
    trajs = [read_output_h5(p) for p in sorted(_dyn(tmp_path).glob("tasks/traj_*/output.h5"))]
    assert len(trajs) == 4 and all(list(t.labels) == SOC_LABELS for t in trajs)
    full = [t["ticks"] == 0 for t in trajs]
    active = np.stack([t["active"][f] for t, f in zip(trajs, full, strict=True)])
    pops = np.stack([t["populations"][f] for t, f in zip(trajs, full, strict=True)])
    with h5py.File(_ana(tmp_path) / "populations.h5") as f:
        for j in range(len(SOC_LABELS)):
            np.testing.assert_array_equal(f["classical"][()][:, j], (active == j).mean(axis=0))
        np.testing.assert_allclose(f["quantum"][()], pops.mean(axis=0), rtol=0, atol=1e-15)
        assert list(f.attrs["fit_states"]) == ["S1"]
        np.testing.assert_array_equal(f["fit_target"][()], f["classical"][()][:, 1])
    assert pops[:, :, 2:].sum(axis=2).max() > 1e-3  # the triplets are populated
    summary = _analysis(tmp_path)
    assert summary["n_included"] == 4 and summary["stage_failed"] is False
    assert summary["fit"]["states"] == ["S1"]
    # the transitions equal a recount of the changes of `active` over every record
    expected = {k: {"at_hop": 0, "without_hop": 0} for k in ("S-S", "S-T", "T-S", "T-T")}
    for t in trajs:
        a, hop_to = t["active"], t["hop_to"]
        for r in range(1, a.size):
            if a[r] != a[r - 1]:
                expected[_kind(a[r - 1], a[r])]["at_hop" if hop_to[r] >= 0 else "without_hop"] += 1
    transitions = summary["transitions"]
    assert transitions["kinds"] == expected
    first = transitions["first_singlet_to_triplet_fs"]
    assert sorted(first["per_trajectory"]) == [f"traj_{k:05d}" for k in range(4)]
    assert first["n_crossed"] + first["n_censored"] == 4
    assert "\n## Intersystem crossing\n" in (_ana(tmp_path) / "report.md").read_text()
