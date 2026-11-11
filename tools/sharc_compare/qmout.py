"""Parse the text format of SHARC's QM.out (our own parser; no SHARC code).

Layout from the public SHARC 4.1 manual, section 6.34.2. Blocks read: 1 Hamiltonian (complex),
2 dipoles, 3 gradients, 5 non-adiabatic couplings (``Nonadiabatic`` or ``Non-adiabatic`` in the
title), 6 overlap matrix (complex). Gradient and NAC vectors whose dimension line carries
``! m1 <m> s1 <s> ms1 <ms> [m2 <m> s2 <s> ms2 <ms>]`` labels get them in ``gradient_labels`` and
``nac_labels`` (1-based state index ``s`` within multiplicity ``m``); other headers give ``None``.
"""

from __future__ import annotations

import re
from typing import Any

REQUIRED_BLOCKS = ("hamiltonian", "dipole", "gradients")
_NAC_TITLE = re.compile(r"^! 5 Non-?adiabatic couplings")
_LABEL = re.compile(r"m(?P<k>[12])\s+(-?\d+)\s+s(?P=k)\s+(-?\d+)\s+ms(?P=k)\s+(-?\d+)")
Label = tuple[int, int, int]


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


def _labels(line: str) -> list[Label]:
    """The (m, s, ms) labels written after ``!`` on a vector's dimension line."""
    _, _, comment = line.partition("!")
    return [(int(a), int(b), int(c)) for _, a, b, c in _LABEL.findall(comment)]


def _complex_matrix(lines: list[str], i: int, block: str) -> tuple[list[list[complex]], int]:
    """The square complex matrix whose dimension line is ``lines[i]``; returns (matrix, next)."""
    n = _count(lines, i, block)
    rows = _rows(lines, i + 1, n, block)
    if any(len(r) < 2 * n for r in rows):
        raise ValueError(
            f"QM.out: malformed '{block}' block near line {i + 2}: "
            f"each row needs {2 * n} numbers (real and imaginary parts)"
        )
    return [[complex(r[2 * j], r[2 * j + 1]) for j in range(n)] for r in rows], i + 1 + n


def parse_qm_out(text: str, required: tuple[str, ...] = REQUIRED_BLOCKS) -> dict[str, Any]:
    """The blocks of a QM.out text; ``ValueError`` if one of ``required`` is missing."""
    lines = text.splitlines()
    out: dict[str, Any] = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("! 1 Hamiltonian Matrix"):
            out["hamiltonian"], i = _complex_matrix(lines, i + 1, "hamiltonian")
        elif line.startswith("! 6 Overlap matrix"):
            out["overlap"], i = _complex_matrix(lines, i + 1, "overlap")
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
            grads, grad_labels, j = [], [], i + 1
            for _ in range(n_states):
                if j >= len(lines):
                    raise ValueError(
                        f"QM.out: malformed 'gradients' block near line {j + 1}: "
                        f"expected {n_states} vectors, the file ends"
                    )
                found = _labels(lines[j])
                grad_labels.append(found[0] if len(found) == 1 else None)
                grads.append(_rows(lines, j + 1, natom, "gradients"))
                j += 1 + natom
            out["gradients"] = grads
            out["gradient_labels"] = grad_labels
            i = j
        elif _NAC_TITLE.match(line):
            match = re.search(r"\((\d+)x(\d+)x(\d+)x3", line)
            if match is None:
                raise ValueError(
                    f"QM.out: malformed 'Nonadiabatic couplings' block near line {i + 1}: "
                    f"cannot read the header {line!r}"
                )
            n_states, natom = int(match.group(1)), int(match.group(3))
            nacs: list[list[Any]] = []
            nac_labels: list[list[tuple[Label, Label] | None]] = []
            j = i + 1
            for _ in range(n_states):
                row, label_row = [], []
                for _ in range(n_states):
                    if j >= len(lines):
                        raise ValueError(
                            f"QM.out: malformed 'Nonadiabatic couplings' block near line "
                            f"{j + 1}: expected {n_states}x{n_states} vectors, the file ends"
                        )
                    found = _labels(lines[j])
                    label_row.append((found[0], found[1]) if len(found) == 2 else None)
                    row.append(_rows(lines, j + 1, natom, "Nonadiabatic couplings"))
                    j += 1 + natom
                nacs.append(row)
                nac_labels.append(label_row)
            out["nacs"] = nacs
            out["nac_labels"] = nac_labels
            i = j
        else:
            i += 1
    for name in required:
        if name not in out:
            raise ValueError(f"QM.out: the '{name}' block is missing")
    return out
