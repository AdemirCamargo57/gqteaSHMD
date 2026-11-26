"""Record which software produced a project's results (one entry per `gqteashmd run`)."""

from __future__ import annotations

import os
import platform
import socket
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from gqteashmd import __version__
from gqteashmd.build_info import commit_info
from gqteashmd.util.fileio import read_json, write_json_atomic

PACKAGES = ("pydantic", "numpy", "scipy", "h5py", "pyscf", "geometric")


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip() or None
    except OSError:
        return None
    return None


def collect_provenance() -> dict[str, Any]:
    info = commit_info()
    return {
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "gqteashmd": __version__,
        "git_commit": info.commit,
        "git_dirty": info.dirty,
        "commit_source": info.source,
        "build_time": info.build_time,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "cpu_model": _cpu_model(),
        "packages": {name: _package_version(name) for name in PACKAGES},
    }


def append_provenance(path: Path, entry: dict[str, Any]) -> None:
    runs: list[Any] = []
    if path.exists():
        try:
            data = read_json(path)
            if isinstance(data, dict) and isinstance(data.get("runs"), list):
                runs = list(data["runs"])
        except (OSError, ValueError, KeyError, TypeError):
            runs = []
    runs.append(entry)
    write_json_atomic(path, {"schema_version": 1, "runs": runs})
