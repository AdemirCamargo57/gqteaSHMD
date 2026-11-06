import subprocess
import sys

from gqteashmd.execution.processes import is_worker_process, terminate_process_group


def test_detects_and_terminates_worker_process(tmp_path):
    task_file = tmp_path / "task.json"
    task_file.write_text("{}")
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", str(task_file)],
        start_new_session=True,
    )
    try:
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
