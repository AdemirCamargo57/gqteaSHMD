"""The hatchling build hook writes the build stamp into a wheel (release spec 2.2; plan PR2).

Each test copies the working tree's build inputs into tmp_path and builds there with hatchling's
PEP 517 functions, so the hook under test is the working tree's, committed or not.
"""

import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
STAMP = "gqteashmd/_build.py"


def _tree(tmp_path: Path) -> Path:
    # the tree and the build output are siblings, so a build never makes the tree dirty
    dest = tmp_path / "tree"
    dest.mkdir()
    for name in ("pyproject.toml", "hatch_build.py", "README.md", "LICENSE"):
        shutil.copy2(ROOT / name, dest / name)
    shutil.copytree(
        ROOT / "src",
        dest / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "_build.py"),
    )
    return dest


def _commit(dest: Path) -> str:
    git = ["git", "-C", str(dest), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(["git", "init", "-q", str(dest)], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "c"], check=True)
    out = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def _build(tree: Path, kind: str) -> Path:
    out_dir = tree.parent / "out"
    code = f"from hatchling.build import build_{kind}; print(build_{kind}({str(out_dir)!r}))"
    # no bytecode: importing hatch_build.py would leave an untracked __pycache__ (a dirty tree)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=tree, capture_output=True, text=True, check=True, env=env
    )
    return out_dir / out.stdout.strip().splitlines()[-1]


def _stamp(wheel: Path) -> dict:
    with zipfile.ZipFile(wheel) as archive:
        text = archive.read(STAMP).decode("utf-8")
    namespace: dict = {}
    exec(text, namespace)
    return {k: namespace[k] for k in ("COMMIT", "DIRTY", "BUILD_TIME")}


def test_a_wheel_from_a_clean_checkout_carries_its_commit(tmp_path):
    tree = _tree(tmp_path)
    head = _commit(tree)
    stamp = _stamp(_build(tree, "wheel"))
    assert stamp["COMMIT"] == head and stamp["DIRTY"] is False
    assert datetime.fromisoformat(stamp["BUILD_TIME"]).utcoffset().total_seconds() == 0
    assert not (tree / "src" / "gqteashmd" / "_build.py").exists()  # source tree untouched


def test_a_wheel_from_a_dirty_checkout_says_so(tmp_path):
    tree = _tree(tmp_path)
    head = _commit(tree)
    (tree / "README.md").write_text("changed\n")
    stamp = _stamp(_build(tree, "wheel"))
    assert stamp["COMMIT"] == head and stamp["DIRTY"] is True


def test_a_wheel_without_git_has_no_commit(tmp_path):
    stamp = _stamp(_build(_tree(tmp_path), "wheel"))
    assert stamp["COMMIT"] is None and stamp["DIRTY"] is None


def test_an_editable_build_has_no_stamp(tmp_path):
    tree = _tree(tmp_path)
    _commit(tree)
    with zipfile.ZipFile(_build(tree, "editable")) as archive:
        assert STAMP not in archive.namelist()
