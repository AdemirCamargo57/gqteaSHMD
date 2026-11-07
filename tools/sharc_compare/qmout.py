"""Parse the text format of SHARC's QM.out (our own parser; no SHARC code)."""

from __future__ import annotations

import re
from typing import Any


def _rows(lines: list[str], start: int, n: int) -> list[list[float]]:
    return [[float(x) for x in lines[start + k].split()] for k in range(n)]


def parse_qm_out(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    out: dict[str, Any] = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("! 1 Hamiltonian Matrix"):
            n = int(lines[i + 1].split()[0])
            rows = _rows(lines, i + 2, n)
            out["hamiltonian"] = [[complex(r[2 * j], r[2 * j + 1]) for j in range(n)] for r in rows]
            i += 2 + n
        elif line.startswith("! 2 Dipole Moment Matrices"):
            mats, j = [], i + 1
            for _ in range(3):
                n = int(lines[j].split()[0])
                rows = _rows(lines, j + 1, n)
                mats.append([[r[2 * c] for c in range(n)] for r in rows])
                j += 1 + n
            out["dipole"] = mats
            i = j
        elif line.startswith("! 3 Gradient Vectors"):
            match = re.search(r"\((\d+)x(\d+)x3", line)
            if match is None:
                raise ValueError(f"cannot read gradient block header: {line!r}")
            n_states, natom = int(match.group(1)), int(match.group(2))
            grads, j = [], i + 1
            for _ in range(n_states):
                grads.append(_rows(lines, j + 1, natom))
                j += 1 + natom
            out["gradients"] = grads
            i = j
        else:
            i += 1
    return out
