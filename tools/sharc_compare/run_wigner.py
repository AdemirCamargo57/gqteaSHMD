"""Run SHARC 4.1's wigner.py on a Molden file as a black box and list what it wrote.

SHARC is only *executed*; nothing from SHARC is imported or copied.

Example (WSL):
  python tools/sharc_compare/run_wigner.py --sharc-home /home/ajc/sharc4-4.1 \
      --python /home/ajc/miniforge3/envs/sharc4.1/bin/python --molden freq.molden --n 3 \
      --out-dir /tmp/wigner
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def run_wigner(sharc_home: Path, python: str, molden: Path, n: int, out_dir: Path) -> list[Path]:
    """Execute ``wigner.py -n N molden`` in ``out_dir``; return the files it created."""
    out_dir.mkdir(parents=True, exist_ok=True)
    before = set(os.listdir(out_dir))
    env = dict(os.environ)
    env["SHARC"] = str(sharc_home / "bin")
    # PYTHONPATH is replaced, not extended: SHARC's own conda environment must not see ours.
    env["PYTHONPATH"] = f"{sharc_home / 'lib'}:{sharc_home / 'pysharc'}"
    proc = subprocess.run(
        [python, str(sharc_home / "bin" / "wigner.py"), "-n", str(n), str(molden.resolve())],
        cwd=out_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise SystemExit(f"wigner.py failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
    return sorted(out_dir / name for name in set(os.listdir(out_dir)) - before)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sharc-home", required=True)
    p.add_argument("--python", required=True, help="Python of the SHARC conda environment")
    p.add_argument("--molden", required=True)
    p.add_argument("--n", type=int, required=True, help="number of geometries to sample")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    created = run_wigner(
        Path(args.sharc_home), args.python, Path(args.molden), args.n, Path(args.out_dir)
    )
    for path in created:
        print(path)
    if not created:
        raise SystemExit("wigner.py exited with status 0 but wrote no files")


if __name__ == "__main__":
    main()
