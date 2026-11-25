"""Checks of the release clean-install gate (release spec 2026-11-25, section 5.1; plan PR1).

Run by tools/release/clean_install_check.sh with the development environment's Python. Standard
library only; never imported by gqteashmd. Every check is evaluated and written to report.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

RUNTIME_PACKAGES = ("pydantic", "numpy", "scipy", "pyscf", "h5py", "geometric")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def expected_version_line(version: str, commit: str) -> str:
    return f"gqteashmd {version} (commit {commit[:7]})"


def pip_installed_runtime(conda_list: list[dict[str, Any]]) -> list[str]:
    """Runtime dependencies that pip, not conda, installed (the lock file would miss them)."""
    return sorted(
        str(p["name"])
        for p in conda_list
        if p.get("name") in RUNTIME_PACKAGES and p.get("channel") == "pypi"
    )


def _last_run(provenance: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        return {}
    runs = provenance.get("runs")
    if not isinstance(runs, list) or not runs or not isinstance(runs[-1], dict):
        return {}
    return runs[-1]


def evaluate(
    *,
    version: str,
    commit: str,
    version_line: str,
    conda_list: list[dict[str, Any]],
    dev_conda_list: list[dict[str, Any]],
    provenance: dict[str, Any] | None,
    analysis_exists: bool,
    run_exit: int,
) -> list[dict[str, Any]]:
    entry = _last_run(provenance)
    pip = pip_installed_runtime(conda_list)
    dev_pip = pip_installed_runtime(dev_conda_list)
    line = version_line.strip()
    items = [
        ("version_line", line == expected_version_line(version, commit), line),
        # the fresh environment, built from the lock file: a lock that pulled pip packages
        ("runtime_from_conda", not pip, pip),
        # the tested environment (gqteashmd-dev), which the lock file is exported from: a runtime
        # dependency installed there by pip is missing from the lock
        ("dev_runtime_from_conda", not dev_pip, dev_pip),
        ("run_exit_code", run_exit == 0, run_exit),
        ("analysis_json", analysis_exists, analysis_exists),
        ("provenance_version", entry.get("gqteashmd") == version, entry.get("gqteashmd")),
        (
            "provenance_commit_source",
            entry.get("commit_source") == "build",
            entry.get("commit_source"),
        ),
        (
            "provenance_commit",
            bool(COMMIT_RE.fullmatch(commit)) and entry.get("git_commit") == commit,
            entry.get("git_commit"),
        ),
        ("provenance_clean", entry.get("git_dirty") is False, entry.get("git_dirty")),
    ]
    return [{"check": name, "passed": bool(ok), "value": value} for name, ok, value in items]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--version-line", required=True)
    parser.add_argument("--conda-list", type=Path, required=True)
    parser.add_argument("--dev-conda-list", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--run-exit", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    conda_list = _read_json(args.conda_list)
    dev_conda_list = _read_json(args.dev_conda_list)
    checks = evaluate(
        version=args.version,
        commit=args.commit,
        version_line=args.version_line,
        conda_list=conda_list if isinstance(conda_list, list) else [],
        dev_conda_list=dev_conda_list if isinstance(dev_conda_list, list) else [],
        provenance=_read_json(args.provenance),
        analysis_exists=args.analysis.is_file(),
        run_exit=args.run_exit,
    )
    passed = all(c["passed"] for c in checks)
    report = {"version": args.version, "commit": args.commit, "passed": passed, "checks": checks}
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for c in checks:
        print(f"{'ok  ' if c['passed'] else 'FAIL'} {c['check']}: {c['value']}")
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
