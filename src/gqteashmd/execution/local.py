"""Run task subprocesses on the local machine with a concurrency limit."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

log = logging.getLogger(__name__)

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@dataclass(frozen=True)
class Job:
    key: str
    task_file: Path
    log_file: Path


@dataclass(frozen=True)
class JobOutcome:
    job: Job
    returncode: int | None  # None: terminated because a stop was requested


@dataclass
class _Running:
    job: Job
    process: subprocess.Popen[bytes]
    log: IO[bytes]


def _signal_group(pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        # A stray member with other credentials must not abort the stage.
        log.debug("No permission to signal process group %d with %s", pid, sig.name)


class LocalExecutor:
    def __init__(
        self,
        *,
        max_parallel: int,
        threads_per_task: int,
        memory_mb_per_task: int,
        poll_interval: float = 0.5,
        stop_grace_period: float = 30.0,
        command: Sequence[str] | None = None,
    ) -> None:
        if max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        self.max_parallel = max_parallel
        self.threads_per_task = threads_per_task
        self.memory_mb_per_task = memory_mb_per_task
        self.poll_interval = poll_interval
        self.stop_grace_period = stop_grace_period
        self.command = (
            list(command) if command is not None else [sys.executable, "-m", "gqteashmd.worker"]
        )

    def environment(self) -> dict[str, str]:
        env = dict(os.environ)
        for name in THREAD_ENV_VARS:
            env[name] = str(self.threads_per_task)
        env["PYSCF_MAX_MEMORY"] = str(self.memory_mb_per_task)
        return env

    def run(
        self,
        jobs: Sequence[Job],
        *,
        on_start: Callable[[Job, int], None],
        on_finish: Callable[[JobOutcome], None],
        should_stop: Callable[[], bool],
    ) -> bool:
        """Run all jobs. Returns False if a stop was requested before all jobs finished."""
        queue = deque(jobs)
        running: dict[str, _Running] = {}
        env = self.environment()
        try:
            while queue or running:
                if should_stop():
                    items = list(running.values())
                    self._reap_all(running)
                    for item in items:
                        on_finish(JobOutcome(item.job, None))
                    return False
                while queue and len(running) < self.max_parallel:
                    job = queue.popleft()
                    item = self._launch(job, env)
                    running[job.key] = item
                    on_start(job, item.process.pid)
                for key in list(running):
                    item = running[key]
                    code = item.process.poll()
                    if code is not None:
                        _signal_group(item.process.pid, signal.SIGKILL)  # stray children
                        item.log.close()
                        del running[key]
                        on_finish(JobOutcome(item.job, code))
                if queue or running:
                    time.sleep(self.poll_interval)
        except BaseException:
            self._reap_all(running)  # no on_finish: the caller is already failing
            raise
        return True

    def _launch(self, job: Job, env: dict[str, str]) -> _Running:
        job.log_file.parent.mkdir(parents=True, exist_ok=True)
        log = open(job.log_file, "ab")  # noqa: SIM115 - closed when the job ends
        try:
            process = subprocess.Popen(
                [*self.command, str(job.task_file)],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                cwd=job.task_file.parent,
                start_new_session=True,  # Ctrl-C on the controller does not hit workers
            )
        except OSError:
            log.close()
            raise
        return _Running(job, process, log)

    def _reap_all(self, running: dict[str, _Running]) -> None:
        """SIGTERM every running group, wait out the grace period, SIGKILL, reap, close logs."""
        for item in running.values():
            _signal_group(item.process.pid, signal.SIGTERM)
        deadline = time.monotonic() + self.stop_grace_period
        for item in running.values():
            try:
                item.process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                _signal_group(item.process.pid, signal.SIGKILL)
                item.process.wait()
            _signal_group(item.process.pid, signal.SIGKILL)  # stray children
            item.log.close()
        running.clear()
