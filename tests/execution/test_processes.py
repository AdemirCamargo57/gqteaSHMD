import subprocess
import sys
import time

from gqteashmd.execution.processes import is_worker_process, terminate_process_group


def _wait_until(predicate, timeout):
    """Poll until `predicate()` is true (the child may not have exec'd yet) or time runs out."""
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.02)


def test_detects_and_terminates_worker_process(tmp_path):
    task_file = tmp_path / "task.json"
    task_file.write_text("{}")
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", str(task_file)],
        start_new_session=True,
    )
    try:
        _wait_until(lambda: is_worker_process(proc.pid, task_file), timeout=5.0)
        assert is_worker_process(proc.pid, task_file) is True
        assert is_worker_process(proc.pid, tmp_path / "other.json") is False
        terminate_process_group(proc.pid, grace=5.0)
        assert proc.wait(timeout=5) != 0
    finally:
        if proc.poll() is None:
            proc.kill()


def test_unknown_pid_is_not_a_worker(tmp_path):
    assert is_worker_process(2**22 + 12345, tmp_path / "task.json") is False
    terminate_process_group(2**22 + 12345, grace=0.1)  # must not raise


def test_terminate_process_group_kills_members_after_the_leader_exits(tmp_path):
    import os
    import textwrap

    from gqteashmd.execution.processes import _alive

    pid_file = tmp_path / "child.pid"
    leader = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(f"""
            import subprocess, sys, time
            code = (
                "import os, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                f"open({str(pid_file)!r} + '.tmp', 'w').write(str(os.getpid())); "
                f"os.rename({str(pid_file)!r} + '.tmp', {str(pid_file)!r}); time.sleep(60)"
            )
            child = subprocess.Popen([sys.executable, "-c", code])
            time.sleep(60)
            """),
        ],
        start_new_session=True,
    )
    child = None
    try:
        _wait_until(pid_file.exists, 10)
        assert pid_file.exists()
        child = int(pid_file.read_text())
        terminate_process_group(leader.pid, grace=2.0)  # the leader dies; the child ignores SIGTERM
        leader.wait(timeout=5)
        _wait_until(lambda: not _alive(child), 5)
        assert not _alive(child)
    finally:
        if leader.poll() is None:
            leader.kill()
            leader.wait()
        if child is not None and _alive(child):
            os.kill(child, 9)
