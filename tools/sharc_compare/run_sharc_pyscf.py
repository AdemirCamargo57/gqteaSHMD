"""Run SHARC 4.1's SHARC_PYSCF.py as a black box and write its results as reference JSON.

SHARC is only *executed*; nothing from SHARC is imported or copied.

Example (WSL):
  python tools/sharc_compare/run_sharc_pyscf.py --sharc-home /home/ajc/sharc4-4.1 \
      --python /home/ajc/miniforge3/envs/sharc4.1/bin/python --xyz tests/data/ethylene.xyz \
      --basis 6-31g* --ncas 2 --nelecas 2 --roots 2 --out /tmp/ref.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qmout import parse_qm_out  # noqa: E402

TEMPLATE = """ncas {ncas}
nelecas {nelecas}
roots {roots}
basis {basis}
method casscf
fix-spin-shift 0.2
conv-tol 1e-10
conv-tol-grad 1e-6
verbose 3
"""


def run(args: argparse.Namespace) -> dict:
    xyz_lines = Path(args.xyz).read_text().splitlines()
    natom = int(xyz_lines[0].split()[0])
    atoms = "\n".join(xyz_lines[2 : 2 + natom])
    with tempfile.TemporaryDirectory(prefix="gq-sharc-") as tmp:
        work = Path(tmp)
        (work / "QM.in").write_text(
            f"{natom}\nreference\n{atoms}\ninit\nsavedir ./SAVE\nunit angstrom\n"
            f"states {args.roots}\nh\ndm\ngrad all\n"
        )
        (work / "PYSCF.template").write_text(TEMPLATE.format(**vars(args)))
        (work / "PYSCF.resources").write_text("scratchdir ./SCRATCH\nmemory 4000\nncpu 1\n")
        home = Path(args.sharc_home)
        env = dict(os.environ)
        env["SHARC"] = str(home / "bin")
        env["PYTHONPATH"] = f"{home / 'lib'}:{home / 'pysharc'}"
        proc = subprocess.run(
            [args.python, str(home / "bin" / "SHARC_PYSCF.py"), "QM.in"],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode != 0:
            raise SystemExit(f"SHARC_PYSCF.py failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
        data = parse_qm_out((work / "QM.out").read_text())
    n = args.roots
    return {
        "schema_version": 1,
        "source": "SHARC 4.1 SHARC_PYSCF.py, executed as a black box (QM.out parsed by gqteashmd)",
        "molecule": str(args.xyz),
        "method": "SA-CASSCF",
        "basis": args.basis,
        "ncas": args.ncas,
        "nelecas": args.nelecas,
        "roots": n,
        "sharc_template": {"fix-spin-shift": 0.2, "conv-tol": 1e-10, "conv-tol-grad": 1e-06},
        "energies_hartree": [data["hamiltonian"][k][k].real for k in range(n)],
        "transition_dipoles_au": {
            f"S{k}": [data["dipole"][x][0][k] for x in range(3)] for k in range(1, n)
        },
        "gradients_hartree_per_bohr": {f"S{k}": data["gradients"][k] for k in range(n)},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sharc-home", required=True)
    p.add_argument("--python", required=True, help="Python of the SHARC conda environment")
    p.add_argument("--xyz", required=True)
    p.add_argument("--basis", required=True)
    p.add_argument("--ncas", type=int, required=True)
    p.add_argument("--nelecas", type=int, required=True)
    p.add_argument("--roots", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    Path(args.out).write_text(json.dumps(run(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
