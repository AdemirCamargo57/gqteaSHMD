"""Which commit this installation of gqteashmd comes from (spec 2026-11-25-release-0.1.0, 2.3-2.5).

A source checkout (an editable install) answers from git directly. A wheel answers from the
build stamp `gqteashmd/_build.py` that `hatch_build.py` writes into it. Standard library only:
`gqteashmd.cli` imports this module.
"""

from __future__ import annotations

import importlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from gqteashmd import __version__

CommitSource = Literal["git", "build"]


@dataclass(frozen=True)
class CommitInfo:
    commit: str | None
    dirty: bool | None
    source: CommitSource | None
    build_time: str | None


UNKNOWN = CommitInfo(None, None, None, None)


def _in_src_layout(module_file: Path) -> bool:
    """True for `<root>/src/gqteashmd/<module>`: a source checkout, not e.g. `pip --target`."""
    return module_file.resolve().parents[1].name == "src"


def _checkout_root() -> Path | None:
    """The repository root if this module is imported from a source checkout (src layout)."""
    here = Path(__file__)
    return here.resolve().parents[2] if _in_src_layout(here) else None


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _live_checkout() -> CommitInfo | None:
    root = _checkout_root()
    if root is None or not (root / ".git").exists():
        return None
    commit = _git(root, "rev-parse", "HEAD")
    if not commit:
        return None
    status = _git(root, "status", "--porcelain")
    return CommitInfo(commit, None if status is None else bool(status), "git", None)


def build_stamp() -> dict[str, Any] | None:
    """The stamp written into a wheel at build time; `None` if the package carries none."""
    try:
        module = importlib.import_module("gqteashmd._build")
    except ImportError:
        return None
    return {
        "commit": getattr(module, "COMMIT", None),
        "dirty": getattr(module, "DIRTY", None),
        "build_time": getattr(module, "BUILD_TIME", None),
    }


def commit_info() -> CommitInfo:
    """The live checkout first, then the build stamp, else nothing known."""
    live = _live_checkout()
    if live is not None:
        return live
    stamp = build_stamp()
    if stamp is None or not stamp["commit"]:
        return UNKNOWN
    dirty = stamp["dirty"] if isinstance(stamp["dirty"], bool) else None
    build_time = stamp["build_time"] if isinstance(stamp["build_time"], str) else None
    return CommitInfo(str(stamp["commit"]), dirty, "build", build_time)


def version_text() -> str:
    """The text of `gqteashmd --version`."""
    info = commit_info()
    if info.commit is None:
        return f"gqteashmd {__version__}"
    dirty = ", dirty" if info.dirty else ""
    return f"gqteashmd {__version__} (commit {info.commit[:7]}{dirty})"
