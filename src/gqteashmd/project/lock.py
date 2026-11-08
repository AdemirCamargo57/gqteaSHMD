"""Ensure only one controller runs per project (Linux flock)."""

from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path
from typing import IO

from gqteashmd.errors import ProjectError


class ProjectLock:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fh: IO[str] | None = None

    def __enter__(self) -> ProjectLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+", encoding="utf-8")  # noqa: SIM115 - kept open while locked
        try:
            for attempt in range(2):
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    if attempt == 0:
                        time.sleep(0.2)  # `locked_by` may be probing the lock for an instant
                        continue
                    holder = locked_by(self.path)
                    raise ProjectError(
                        "Another gqteashmd controller is running for this project "
                        f"(pid {holder if holder is not None else 'unknown'}, "
                        f"lock file {self.path}). "
                        "Use `gqteashmd status` or `gqteashmd stop`."
                    ) from exc
                except OSError as exc:
                    raise ProjectError(
                        f"File locking is not available for {self.path} "
                        f"({exc.strerror or exc}); put the project directory on a local Linux "
                        "filesystem."
                    ) from exc
            try:
                fh.seek(0)
                fh.truncate()
                fh.write(f"{os.getpid()}\n")
                fh.flush()
            except BaseException:
                fcntl.flock(fh, fcntl.LOCK_UN)
                raise
        except BaseException:
            fh.close()
            raise
        self._fh = fh
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._fh is not None:
            fh, self._fh = self._fh, None
            try:
                fcntl.flock(fh, fcntl.LOCK_UN)
            finally:
                fh.close()


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
        except OSError as exc:
            raise ProjectError(
                f"File locking is not available for {path} "
                f"({exc.strerror or exc}); put the project directory on a local Linux filesystem."
            ) from exc
        fcntl.flock(fh, fcntl.LOCK_UN)
        return None
