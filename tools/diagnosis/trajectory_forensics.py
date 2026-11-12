"""Forensics JSON of one gqteashmd project (P9a spec 3.1).

Example (WSL):
  python tools/diagnosis/trajectory_forensics.py ~/p8_m4/run --out ~/p9a/forensics_m4.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forensics import JUMP_THRESHOLD_EV, jsonable, project_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("project", type=Path, help="the project directory (holds 06_dynamics)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold-eV", type=float, default=JUMP_THRESHOLD_EV)
    args = parser.parse_args(argv)
    report = project_report(args.project.expanduser(), threshold_eV=args.threshold_eV)
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsonable(report), indent=1, allow_nan=False) + "\n")
    print(f"{report['n_trajectories']} trajectories; onsets {report['onsets']}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
