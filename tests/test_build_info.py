"""Which commit an installation was built from (spec 2026-11-25-release-0.1.0, 2.3-2.5)."""

import subprocess
import sys
import types

import pytest

from gqteashmd import __version__, build_info
from gqteashmd.build_info import CommitInfo

STAMP_TIME = "2026-11-25T10:00:00+00:00"


def _stamp(monkeypatch, **values):
    module = types.ModuleType("gqteashmd._build")
    for name, value in values.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "gqteashmd._build", module)


def _no_stamp(monkeypatch):
    # a None entry in sys.modules makes the import raise ModuleNotFoundError
    monkeypatch.setitem(sys.modules, "gqteashmd._build", None)


def _repo(path):
    git = ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "f.txt").write_text("x\n")
    subprocess.run([*git, "add", "f.txt"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "c"], check=True)
    out = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


@pytest.fixture
def no_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(build_info, "_checkout_root", lambda: tmp_path)


def test_no_stamp_module_gives_none(monkeypatch):
    _no_stamp(monkeypatch)
    assert build_info.build_stamp() is None


def test_the_stamp_is_read(monkeypatch):
    _stamp(monkeypatch, COMMIT="a" * 40, DIRTY=False, BUILD_TIME=STAMP_TIME)
    assert build_info.build_stamp() == {
        "commit": "a" * 40,
        "dirty": False,
        "build_time": STAMP_TIME,
    }


def test_a_live_checkout_comes_first(monkeypatch, tmp_path):
    head = _repo(tmp_path)
    monkeypatch.setattr(build_info, "_checkout_root", lambda: tmp_path)
    _stamp(monkeypatch, COMMIT="b" * 40, DIRTY=False, BUILD_TIME=STAMP_TIME)
    assert build_info.commit_info() == CommitInfo(head, False, "git", None)
    (tmp_path / "f.txt").write_text("changed\n")
    assert build_info.commit_info() == CommitInfo(head, True, "git", None)


def test_the_stamp_is_used_without_a_checkout(no_checkout, monkeypatch):
    _stamp(monkeypatch, COMMIT="c" * 40, DIRTY=False, BUILD_TIME=STAMP_TIME)
    assert build_info.commit_info() == CommitInfo("c" * 40, False, "build", STAMP_TIME)


def test_nothing_known_without_checkout_or_stamp(no_checkout, monkeypatch):
    _no_stamp(monkeypatch)
    assert build_info.commit_info() == CommitInfo(None, None, None, None)


def test_a_stamp_without_a_commit_is_unknown(no_checkout, monkeypatch):
    _stamp(monkeypatch, COMMIT=None, DIRTY=None, BUILD_TIME=STAMP_TIME)
    assert build_info.commit_info() == CommitInfo(None, None, None, None)


@pytest.mark.parametrize(
    ("info", "text"),
    [
        (CommitInfo("0123456789" * 4, False, "build", STAMP_TIME), "(commit 0123456)"),
        (CommitInfo("0123456789" * 4, True, "git", None), "(commit 0123456, dirty)"),
        (CommitInfo("0123456789" * 4, None, "git", None), "(commit 0123456)"),
    ],
)
def test_version_text_names_the_commit(monkeypatch, info, text):
    monkeypatch.setattr(build_info, "commit_info", lambda: info)
    assert build_info.version_text() == f"gqteashmd {__version__} {text}"


def test_version_text_without_a_commit(monkeypatch):
    monkeypatch.setattr(build_info, "commit_info", lambda: CommitInfo(None, None, None, None))
    assert build_info.version_text() == f"gqteashmd {__version__}"


def test_build_info_imports_only_the_standard_library():
    code = (
        "import sys; import gqteashmd.build_info; "
        "assert not any(m.split('.')[0] in ('pyscf', 'geometric', 'h5py', 'scipy', 'numpy', "
        "'pydantic') for m in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
