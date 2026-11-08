"""Parse the text format of SHARC's ``initconds`` file (our own parser; no SHARC code).

The format was observed from a black-box run: a header (``Ninit``, ``Natom``, ``Temp``,
``Eharm``), an ``Equilibrium`` block and one ``Index`` block per sample. Atom rows hold
``symbol charge x y z mass vx vy vz`` (positions in bohr, mass in amu, velocities in a.u.).
"""

from __future__ import annotations

from typing import Any

import numpy as np

_HEADER_KEYS = ("Ninit", "Natom", "Temp", "Eharm")


def _atom_rows(lines: list[str], start: int, natom: int, block: str) -> list[list[str]]:
    rows = []
    for k in range(natom):
        index = start + k
        fields = lines[index].split() if index < len(lines) else []
        if len(fields) != 9:
            raise ValueError(
                f"initconds: malformed '{block}' block near line {index + 1}: "
                f"expected an atom row with 9 fields"
            )
        rows.append(fields)
    return rows


def _floats(fields: list[str], lo: int, hi: int, line_hint: str) -> list[float]:
    try:
        return [float(x) for x in fields[lo:hi]]
    except ValueError as exc:
        raise ValueError(f"initconds: malformed number in {line_hint}: {exc}") from exc


def parse_initconds(text: str) -> dict[str, Any]:
    """Return header values, the equilibrium geometry and all samples as NumPy arrays."""
    lines = text.splitlines()
    header: dict[str, float] = {}
    for line in lines:
        fields = line.split()
        if fields and fields[0] in _HEADER_KEYS and len(fields) >= 2:
            try:
                header[fields[0]] = float(fields[1])
            except ValueError as exc:
                raise ValueError(f"initconds: bad value in header line '{line}'") from exc
    missing = [key for key in _HEADER_KEYS if key not in header]
    if missing:
        raise ValueError(f"initconds: header is missing {', '.join(missing)}")
    natom = int(header["Natom"])
    ninit = int(header["Ninit"])

    try:
        eq_start = next(i for i, line in enumerate(lines) if line.strip() == "Equilibrium") + 1
    except StopIteration:
        raise ValueError("initconds: no 'Equilibrium' block") from None
    eq_rows = _atom_rows(lines, eq_start, natom, "Equilibrium")
    symbols = [row[0] for row in eq_rows]
    masses = [float(row[5]) for row in eq_rows]
    equilibrium = [_floats(row, 2, 5, "Equilibrium") for row in eq_rows]

    positions, velocities, ekin = [], [], []
    i = eq_start + natom
    while i < len(lines):
        fields = lines[i].split()
        if len(fields) == 2 and fields[0] == "Index":
            if i + 1 >= len(lines) or lines[i + 1].strip() != "Atoms":
                raise ValueError(f"initconds: expected 'Atoms' after line {i + 1}")
            rows = _atom_rows(lines, i + 2, natom, f"Index {fields[1]}")
            positions.append([_floats(r, 2, 5, f"Index {fields[1]}") for r in rows])
            velocities.append([_floats(r, 6, 9, f"Index {fields[1]}") for r in rows])
            j = i + 2 + natom
            while j < len(lines) and lines[j].split()[:1] != ["Ekin"]:
                if lines[j].split()[:1] == ["Index"]:
                    break
                j += 1
            if j >= len(lines) or lines[j].split()[:1] != ["Ekin"]:
                raise ValueError(f"initconds: no 'Ekin' line for Index {fields[1]}")
            try:
                ekin.append(float(lines[j].split()[1]))
            except (IndexError, ValueError) as exc:
                raise ValueError(f"initconds: bad 'Ekin' line '{lines[j]}'") from exc
            i = j
        i += 1
    if len(positions) != ninit:
        raise ValueError(
            f"initconds: header says Ninit = {ninit} but {len(positions)} samples found"
        )
    return {
        "natom": natom,
        "temperature_K": header["Temp"],
        "eharm_hartree": header["Eharm"],
        "symbols": symbols,
        "masses_amu": np.array(masses),
        "equilibrium_bohr": np.array(equilibrium).reshape(natom, 3),
        "positions_bohr": np.array(positions).reshape(ninit, natom, 3),
        "velocities_au": np.array(velocities).reshape(ninit, natom, 3),
        "ekin_hartree": np.array(ekin),
    }
