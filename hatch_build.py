"""Hatchling build hook: write the build stamp into the wheel (release spec 2026-11-25, 2.2).

A wheel gets `gqteashmd/_build.py` with the commit, whether the tree was dirty, and the build
time (read by `gqteashmd.build_info`). Editable builds get nothing: they read the live checkout.
The source tree is never written; the stamp is a temporary file added with `force_include`.
"""

from __future__ import annotations

import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

STAMP_TARGET = "gqteashmd/_build.py"


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def stamp_text(root: Path, now: datetime) -> str:
    """The stamp module's text: `root` must be the top of a git checkout to get a commit."""
    commit: str | None = None
    dirty: bool | None = None
    top = _git(root, "rev-parse", "--show-toplevel")
    if top and Path(top).resolve() == root.resolve():
        commit = _git(root, "rev-parse", "HEAD") or None
        if commit is not None:
            status = _git(root, "status", "--porcelain")
            dirty = None if status is None else bool(status)
    return (
        '"""Build stamp, written by hatch_build.py when this wheel was built."""\n\n'
        f"COMMIT = {commit!r}\n"
        f"DIRTY = {dirty!r}\n"
        f"BUILD_TIME = {now.isoformat(timespec='seconds')!r}\n"
    )


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel" or version == "editable":
            return
        self._stamp_dir = tempfile.TemporaryDirectory(prefix="gqteashmd-stamp-")
        path = Path(self._stamp_dir.name) / "_build.py"
        path.write_text(stamp_text(Path(self.root), datetime.now(UTC)), encoding="utf-8")
        build_data["force_include"][str(path)] = STAMP_TARGET

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        stamp_dir = getattr(self, "_stamp_dir", None)
        if stamp_dir is not None:
            stamp_dir.cleanup()
            self._stamp_dir = None
