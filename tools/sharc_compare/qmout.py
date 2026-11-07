"""Parse the text format of SHARC's QM.out (our own parser; no SHARC code)."""

from __future__ import annotations

import re
from typing import Any

REQUIRED_BLOCKS = ("hamiltonian", "dipole", "gradients")


def _rows(lines: list[str], start: int, n: int, block: str) -> list[list[float]]:
    rows = []
    for k in range(n):
        index = start + k
        if index >= len(lines):
            raise ValueError(
                f"QM.out: malformed '{block}' block near line {start + 1}: "
                f"expected {n} data lines, the file ends after {k}"
            )
        try:
            rows.append([float(x) for x in lines[index].split()])
        except ValueError as exc:
            raise ValueError(
                f"QM.out: malformed '{block}' block near line {index + 1}: {exc}"
            ) from exc
    return rows


def _count(lines: list[str], index: int, block: str) -> int:
    try:
        return int(lines[index].split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"QM.out: malformed '{block}' block near line {index + 1}: "
            "expected the matrix dimension"
        ) from exc


def parse_qm_out(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    out: dict[str, Any] = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("! 1 Hamiltonian Matrix"):
            n = _count(lines, i + 1, "hamiltonian")
            rows = _rows(lines, i + 2, n, "hamiltonian")
            if any(len(r) < 2 * n for r in rows):
                raise ValueError(
                    f"QM.out: malformed 'hamiltonian' block near line {i + 3}: "
                    f"each row needs {2 * n} numbers (real and imaginary parts)"
                )
            out["hamiltonian"] = [[complex(r[2 * j], r[2 * j + 1]) for j in range(n)] for r in rows]
            i += 2 + n
        elif line.startswith("! 2 Dipole Moment Matrices"):
            mats, j = [], i + 1
            for _ in range(3):
                n = _count(lines, j, "dipole")
                rows = _rows(lines, j + 1, n, "dipole")
                if any(len(r) < 2 * n for r in rows):
                    raise ValueError(
                        f"QM.out: malformed 'dipole' block near line {j + 2}: "
                        f"each row needs {2 * n} numbers (real and imaginary parts)"
                    )
                mats.append([[r[2 * c] for c in range(n)] for r in rows])
                j += 1 + n
            out["dipole"] = mats
            i = j
        elif line.startswith("! 3 Gradient Vectors"):
            match = re.search(r"\((\d+)x(\d+)x3", line)
            if match is None:
                raise ValueError(
                    f"QM.out: malformed 'gradients' block near line {i + 1}: "
                    f"cannot read the header {line!r}"
                )
            n_states, natom = int(match.group(1)), int(match.group(2))
            grads, j = [], i + 1
            for _ in range(n_states):
                grads.append(_rows(lines, j + 1, natom, "gradients"))
                j += 1 + natom
            out["gradients"] = grads
            i = j
        else:
            i += 1
    for name in REQUIRED_BLOCKS:
        if name not in out:
            raise ValueError(f"QM.out: the '{name}' block is missing")
    return out
