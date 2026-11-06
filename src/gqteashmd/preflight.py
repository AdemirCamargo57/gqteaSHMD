"""Checks that run before any stage starts and before the project directory is touched."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from gqteashmd.config.schema import ShmdConfig
from gqteashmd.errors import ConfigError
from gqteashmd.stages.base import Stage


@dataclass(frozen=True)
class MachineInfo:
    cores: int | None
    memory_mb: int | None


def detect_machine() -> MachineInfo:
    try:
        cores: int | None = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = os.cpu_count()
    memory_mb: int | None = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                memory_mb = int(line.split()[1]) // 1024
                break
    except (OSError, ValueError, IndexError):
        memory_mb = None
    return MachineInfo(cores=cores, memory_mb=memory_mb)


def _nearest_existing(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate
    return Path("/")


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


_WSL_DRIVE = re.compile(r"^/mnt/[A-Za-z](/|$)")


def _unsafe_location(path: Path) -> str | None:
    """Reason the path is unsuitable for a project directory, or None."""
    if _WSL_DRIVE.match(path.as_posix()):
        return "a Windows drive mounted in WSL"
    for part in path.parts:
        lowered = part.lower()
        if lowered == "onedrive" or lowered.startswith("onedrive - "):
            return "a OneDrive folder"
    return None


def run_preflight(
    config: ShmdConfig, stages: Sequence[Stage], *, machine: MachineInfo | None = None
) -> list[str]:
    """Raise ConfigError for problems that must stop the run; return warnings."""
    machine = machine or detect_machine()
    warnings: list[str] = []
    project_dir = config.project.directory

    reason = _unsafe_location(project_dir)
    if reason is not None:
        raise ConfigError(
            f"Project directory {project_dir} is on {reason}, where file "
            "locks and many small files are unreliable. Set project.directory to a local Linux "
            f"disk, e.g. ~/gqteashmd-runs/{config.project.name}."
        )

    try:
        free_mb = shutil.disk_usage(_nearest_existing(project_dir)).free // (1024 * 1024)
    except OSError as exc:
        raise ConfigError(
            f"Cannot determine free disk space for {project_dir}: {exc.strerror or exc}"
        ) from exc
    minimum = config.execution.min_free_disk_mb
    if free_mb < minimum:
        raise ConfigError(
            f"Only {free_mb} MB are free on the disk holding {project_dir}; at least {minimum} MB "
            "are required (execution.min_free_disk_mb)."
        )

    res = config.resources
    if machine.cores is not None and res.total_cores > machine.cores:
        warnings.append(
            f"resources.total_cores = {res.total_cores} exceeds the {machine.cores} cores "
            "detected on this machine; tasks may run slower than expected."
        )
    if machine.memory_mb is not None and res.total_memory_mb > machine.memory_mb:
        warnings.append(
            f"resources.total_memory_mb = {res.total_memory_mb} exceeds the {machine.memory_mb} MB "
            "of RAM detected on this machine; tasks may be killed for lack of memory."
        )

    for stage in stages:
        if not stage.enabled(config):
            continue
        for module in stage.required_modules:
            if not _module_available(module):
                raise ConfigError(
                    f"Stage '{stage.name}' needs the Python module '{module}', which is not "
                    "installed in this environment."
                )
    return warnings
