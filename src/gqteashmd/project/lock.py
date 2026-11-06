"""Ensure only one controller runs per project (Linux flock)."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import IO

from gqteashmd.errors import ProjectError


class ProjectLock:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fh: IO[str] | None = None

    def __enter__(self) -> ProjectLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+", encoding="utf-8")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            fh.close()
            holder = locked_by(self.path)
            raise ProjectError(
                f"Another gqteashmd controller is running for this project "
                f"(pid {holder}, lock file {self.path}). "
                "Use `gqteashmd status` or `gqteashmd stop`."
            ) from exc
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()}\n")
        fh.flush()
        self._fh = fh
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._fh is not None:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None


def locked_by(path: Path) -> int | None:
    """PID of the controller holding the lock, or None if nobody holds it."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "a+", encoding="utf-8") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.seek(0)
            text = fh.read().strip()
            return int(text) if text.isdigit() else None
        fcntl.flock(fh, fcntl.LOCK_UN)
        return None
