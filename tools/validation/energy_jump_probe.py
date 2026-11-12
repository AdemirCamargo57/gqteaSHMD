"""Largest |E_tot(r+1) - E_tot(r)| over consecutive records of output.h5 files (P8 Task 1).

Run: python tools/validation/energy_jump_probe.py ROOT [--json OUT]
Scans ROOT recursively for output.h5, prints one line per file and the overall maximum.
Not collected by pytest; nothing is asserted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

HARTREE_IN_EV = 27.211386245988


def scan(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("output.h5")):
        with h5py.File(path, "r") as f:
            e = np.asarray(f["e_total"][()], dtype=float)
            hops = int(np.count_nonzero(np.asarray(f["hop_to"][()]) >= 0))
            outcome = f.attrs.get("outcome", "")
            if isinstance(outcome, bytes):
                outcome = outcome.decode()
            outcome = str(outcome)
            output_every = int(f.attrs.get("output_every", 1))
        jumps = np.abs(np.diff(e)) * HARTREE_IN_EV
        rows.append(
            {
                "file": str(path),
                "records": int(e.size),
                "output_every": output_every,
                "outcome": outcome,
                "hops": hops,
                "max_jump_eV": float(jumps.max()) if jumps.size else 0.0,
                "p99_jump_eV": float(np.percentile(jumps, 99)) if jumps.size else 0.0,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    rows = scan(args.root)
    for r in rows:
        print(
            f"{r['max_jump_eV']:.3e} eV max, {r['p99_jump_eV']:.3e} eV p99, "
            f"{r['records']} records, {r['hops']} hops, {r['outcome']}: {r['file']}"
        )
    overall = max((r["max_jump_eV"] for r in rows), default=0.0)
    print(f"overall max jump: {overall:.6e} eV over {len(rows)} files")
    if args.json:
        args.json.write_text(json.dumps({"files": rows, "overall_max_jump_eV": overall}, indent=1))


if __name__ == "__main__":
    main()
