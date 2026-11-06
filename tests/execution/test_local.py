import json
import os
import sys
import time

import pytest

from gqteashmd.execution.local import Job, LocalExecutor

SCRIPT = """
import json, os, pathlib, sys, time
spec = json.loads(pathlib.Path(sys.argv[1]).read_text())
print("hello from", spec["name"], flush=True)
time.sleep(spec.get("sleep", 0))
pathlib.Path(spec["out"]).write_text(
    os.environ.get("OMP_NUM_THREADS", "") + " " + os.environ.get("PYSCF_MAX_MEMORY", "")
)
sys.exit(spec.get("exit", 0))
"""


def _jobs(tmp_path, specs):
    script = tmp_path / "script.py"
    script.write_text(SCRIPT)
    jobs = []
    for i, spec in enumerate(specs):
        task_dir = tmp_path / f"job{i}"
        task_dir.mkdir()
        task_file = task_dir / "task.json"
        task_file.write_text(json.dumps({"name": f"job{i}", "out": str(task_dir / "out"), **spec}))
        jobs.append(Job(f"job{i}", task_file, task_dir / "worker.log"))
    return script, jobs


def _executor(script, **kw):
    defaults = dict(
        max_parallel=2,
        threads_per_task=3,
        memory_mb_per_task=1500,
        poll_interval=0.02,
        stop_grace_period=5.0,
    )
    defaults.update(kw)
    return LocalExecutor(command=[sys.executable, str(script)], **defaults)


def test_runs_all_jobs_and_reports_exit_codes(tmp_path):
    script, jobs = _jobs(tmp_path, [{}, {"exit": 3}, {}])
    outcomes = {}
    finished = _executor(script).run(
        jobs,
        on_start=lambda job, pid: None,
        on_finish=lambda o: outcomes.__setitem__(o.job.key, o.returncode),
        should_stop=lambda: False,
    )
    assert finished is True
    assert outcomes == {"job0": 0, "job1": 3, "job2": 0}


def test_never_exceeds_max_parallel(tmp_path):
    script, jobs = _jobs(tmp_path, [{"sleep": 0.3}] * 5)
    active, peak = set(), []

    def start(job, pid):
        active.add(job.key)
        peak.append(len(active))

    _executor(script, max_parallel=2).run(
        jobs,
        on_start=start,
        on_finish=lambda o: active.discard(o.job.key),
        should_stop=lambda: False,
    )
    assert max(peak) == 2


def test_sets_thread_and_memory_environment_and_captures_log(tmp_path):
    script, jobs = _jobs(tmp_path, [{}])
    _executor(script).run(
        jobs, on_start=lambda j, p: None, on_finish=lambda o: None, should_stop=lambda: False
    )
    assert (tmp_path / "job0" / "out").read_text() == "3 1500"
    assert "hello from job0" in (tmp_path / "job0" / "worker.log").read_text()


def test_stop_terminates_running_jobs(tmp_path):
    script, jobs = _jobs(tmp_path, [{"sleep": 60}] * 3)
    started, outcomes = [], []
    t0 = time.monotonic()
    finished = _executor(script).run(
        jobs,
        on_start=lambda job, pid: started.append(job.key),
        on_finish=outcomes.append,
        should_stop=lambda: len(started) >= 2,
    )
    assert finished is False
    assert sorted(o.job.key for o in outcomes) == ["job0", "job1"]
    assert all(o.returncode is None for o in outcomes)
    assert time.monotonic() - t0 < 20


def test_exception_in_callback_cleans_up_running_workers(tmp_path):
    script, jobs = _jobs(tmp_path, [{"sleep": 60}] * 2)
    pids = []

    def start(job, pid):
        pids.append(pid)
        if len(pids) == 2:
            raise RuntimeError("db full")

    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="db full"):
        _executor(script).run(
            jobs, on_start=start, on_finish=lambda o: None, should_stop=lambda: False
        )
    assert len(pids) == 2
    for pid in pids:
        assert not os.path.exists(f"/proc/{pid}")
    assert time.monotonic() - t0 < 20
