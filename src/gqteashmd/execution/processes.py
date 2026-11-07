"""Linux process helpers used to clean up orphaned workers after a controller crash."""

from __future__ import annotations

import contextlib
import os
import signal
import time
from pathlib import Path


def is_worker_process(pid: int, task_file: Path) -> bool:
    """True if `pid` is alive and its command line contains `task_file`."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return str(task_file).encode() in cmdline


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    state = stat.rsplit(")", 1)[1].split()[0]
    return state != "Z"  # a zombie has already exited


def terminate_process_group(pid: int, grace: float) -> None:
    """SIGTERM the process group led by `pid`; SIGKILL whatever remains of it after the leader
    exits or after `grace` seconds."""
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and _alive(pid):
        time.sleep(0.05)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)
