"""The checks of the release gate (tools/release/release_checks.py; release spec 5.1)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "release"))
import release_checks as rc  # noqa: E402

from gqteashmd.cli import main as cli_main  # noqa: E402

VERSION = "0.1.0"
COMMIT = "0123456789abcdef" * 2 + "01234567"  # 40 hex
CONDA = [
    {"name": "numpy", "channel": "conda-forge"},
    {"name": "gqteashmd", "channel": "pypi"},
]


def _provenance(**changes):
    entry = {
        "gqteashmd": VERSION,
        "git_commit": COMMIT,
        "git_dirty": False,
        "commit_source": "build",
        "build_time": "2026-11-25T10:00:00+00:00",
    }
    entry.update(changes)
    return {"schema_version": 1, "runs": [{"gqteashmd": "old"}, entry]}


def _evaluate(**changes):
    kwargs = {
        "version": VERSION,
        "commit": COMMIT,
        "version_line": f"gqteashmd {VERSION} (commit 0123456)\n",
        "conda_list": CONDA,
        "dev_conda_list": CONDA,
        "provenance": _provenance(),
        "analysis_exists": True,
        "run_exit": 0,
    }
    kwargs.update(changes)
    return {c["check"]: c["passed"] for c in rc.evaluate(**kwargs)}


def test_the_expected_version_line():
    assert rc.expected_version_line(VERSION, COMMIT) == "gqteashmd 0.1.0 (commit 0123456)"


def test_only_runtime_packages_from_pip_count():
    listing = [*CONDA, {"name": "scipy", "channel": "pypi"}]
    assert rc.pip_installed_runtime(CONDA) == []
    assert rc.pip_installed_runtime(listing) == ["scipy"]


def test_a_good_install_passes_every_check():
    results = _evaluate()
    assert all(results.values()), results
    assert set(results) == {
        "version_line",
        "runtime_from_conda",
        "dev_runtime_from_conda",
        "run_exit_code",
        "analysis_json",
        "provenance_version",
        "provenance_commit_source",
        "provenance_commit",
        "provenance_clean",
    }


@pytest.mark.parametrize(
    ("changes", "failing"),
    [
        ({"version_line": "gqteashmd 0.1.0 (commit 0123456, dirty)"}, "version_line"),
        ({"conda_list": [*CONDA, {"name": "h5py", "channel": "pypi"}]}, "runtime_from_conda"),
        (
            {"dev_conda_list": [*CONDA, {"name": "h5py", "channel": "pypi"}]},
            "dev_runtime_from_conda",
        ),
        ({"run_exit": 4}, "run_exit_code"),
        ({"analysis_exists": False}, "analysis_json"),
        ({"provenance": _provenance(gqteashmd="0.1.0.dev0")}, "provenance_version"),
        ({"provenance": _provenance(commit_source="git")}, "provenance_commit_source"),
        ({"provenance": _provenance(git_commit="f" * 40)}, "provenance_commit"),
        ({"provenance": _provenance(git_dirty=None)}, "provenance_clean"),
    ],
)
def test_each_check_can_fail(changes, failing):
    results = _evaluate(**changes)
    assert [name for name, ok in results.items() if not ok] == [failing]


def test_no_provenance_fails_the_provenance_checks():
    results = _evaluate(provenance=None)
    failed = sorted(name for name, ok in results.items() if not ok)
    assert failed == [
        "provenance_clean",
        "provenance_commit",
        "provenance_commit_source",
        "provenance_version",
    ]


def test_main_writes_the_report_and_returns_the_verdict(tmp_path):
    conda = tmp_path / "conda.json"
    conda.write_text(json.dumps(CONDA))
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps(CONDA))
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps(_provenance()))
    analysis = tmp_path / "analysis.json"
    analysis.write_text("{}")
    out = tmp_path / "report.json"
    args = [
        "--version", VERSION, "--commit", COMMIT,
        "--version-line", f"gqteashmd {VERSION} (commit 0123456)",
        "--conda-list", str(conda), "--dev-conda-list", str(dev), "--provenance", str(provenance),
        "--analysis", str(analysis), "--run-exit", "0", "--out", str(out),
    ]  # fmt: skip
    assert rc.main(args) == 0
    report = json.loads(out.read_text())
    assert report["passed"] is True and report["version"] == VERSION
    assert report["commit"] == COMMIT and len(report["checks"]) == 9
    analysis.unlink()
    assert rc.main(args) == 1
    assert json.loads(out.read_text())["passed"] is False


def test_the_gate_input_validates():
    assert cli_main(["validate", str(ROOT / "tools" / "release" / "water_check.toml")]) == 0
