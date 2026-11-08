"""Writers and parsers for the text formats SHARC 4.1 tools read and write (our own code).

Everything here was written from formats observed in black-box runs (``QM.in``, ``QM.out``,
``initconds`` and ``initconds.excited``); no SHARC source was read or copied.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def format_qm_in(symbols: Sequence[str], coords_angstrom: np.ndarray, n_states: int) -> str:
    """A minimal ``QM.in``: geometry in angstrom, ``states n`` singlets, Hamiltonian and dipoles."""
    coords = np.asarray(coords_angstrom, dtype=float).reshape(len(symbols), 3)
    lines = [str(len(symbols)), "ref"]
    lines += [
        f"{s} {x:.10f} {y:.10f} {z:.10f}" for s, (x, y, z) in zip(symbols, coords, strict=True)
    ]
    lines += ["unit angstrom", f"states {n_states}", "h", "dm"]
    return "\n".join(lines) + "\n"


def _fortran_e(value: float) -> str:
    """``%17.9E`` with the three-digit exponent of the observed files (``-7.4E+001``)."""
    mantissa, exponent = f"{value:.9E}".split("E")
    return f"{mantissa}E{exponent[0]}{int(exponent[1:]):03d}".rjust(17)


def _matrix_block(matrix: np.ndarray) -> list[str]:
    n = matrix.shape[0]
    lines = [f"{n} {n}"]
    for row in matrix:
        lines.append(" ".join(f"{_fortran_e(float(v))} {_fortran_e(0.0)}" for v in row) + " ")
    return lines


def format_qm_out(
    energies_hartree: Sequence[float], dipole_matrices: np.ndarray, natom: int = 3
) -> str:
    """``QM.out`` with the blocks 0 (basic information), 1 (Hamiltonian) and 2 (dipoles).

    ``dipole_matrices`` has shape (3, n, n) and is real; imaginary parts are written as 0.
    """
    n = len(energies_hartree)
    dip = np.asarray(dipole_matrices, dtype=float)
    if dip.shape != (3, n, n):
        raise ValueError(f"dipole_matrices must have shape (3, {n}, {n}), got {dip.shape}")
    lines = [
        "! 0 Basic information",
        f"states {n} ",
        f"nmstates {n}",
        f"natom {natom}",
        "npc 0",
        "charges 0 ",
        "",
        f"! 1 Hamiltonian Matrix ({n}x{n}, complex)",
    ]
    lines += _matrix_block(np.diag(np.asarray(energies_hartree, dtype=float)))
    lines += ["", "", f"! 2 Dipole Moment Matrices (3x{n}x{n}, complex)"]
    for x in range(3):
        lines += _matrix_block(dip[x])
    return "\n".join(lines) + "\n"


def format_initconds(
    symbols: Sequence[str],
    atomic_numbers: Sequence[float],
    masses_amu: Sequence[float],
    equilibrium_bohr: np.ndarray,
    positions_bohr: np.ndarray,
    velocities_au: np.ndarray,
) -> str:
    """The ``initconds`` file (P5 format); ``Eharm`` and the per-sample energies are 0."""
    natom = len(symbols)
    eq = np.asarray(equilibrium_bohr, dtype=float).reshape(natom, 3)
    pos = np.asarray(positions_bohr, dtype=float).reshape(-1, natom, 3)
    vel = np.asarray(velocities_au, dtype=float).reshape(-1, natom, 3)
    n = pos.shape[0]

    def row(k: int, xyz: np.ndarray, v: np.ndarray) -> str:
        return (
            f" {symbols[k]:2s} {atomic_numbers[k]:5.1f} "
            + " ".join(f"{c:13.8f}" for c in xyz)
            + f" {masses_amu[k]:13.8f} "
            + " ".join(f"{c:13.8f}" for c in v)
        )

    lines = [
        "SHARC Initial conditions file, version 4.0",
        f"Ninit     {n}",
        f"Natom     {natom}",
        "Repr      None",
        f"Temp      {0.0:.10f}",
        f"Eref      {0.0:.10f}",
        f"Eharm     {0.0:.10f}",
        "",
        "Equilibrium",
    ]
    lines += [row(k, eq[k], np.zeros(3)) for k in range(natom)]
    lines += ["", ""]
    for s in range(n):
        lines += [f"Index     {s + 1}", "Atoms"]
        lines += [row(k, pos[s, k], vel[s, k]) for k in range(natom)]
        lines += [
            "States",
            f"Ekin        {0.0:.12f} a.u.",
            f"Epot_harm   {0.0:.12f} a.u.",
            f"Epot        {0.0:.12f} a.u.",
            f"Etot_harm   {0.0:.12f} a.u.",
            f"Etot        {0.0:.12f} a.u.",
            "",
            "",
        ]
    return "\n".join(lines) + "\n"


def _header_value(lines: list[str], key: str) -> str:
    for line in lines:
        fields = line.split()
        if fields and fields[0] == key and len(fields) >= 2:
            return fields[1]
    raise ValueError(f"initconds.excited: header line '{key}' is missing")


def parse_initconds_excited(text: str) -> dict[str, Any]:
    """Parse ``initconds.excited`` into ``eref_hartree``, ``n_states`` and ``states``.

    ``states`` is a list over samples (in file order); each holds one dict per state with
    ``energy_hartree``, ``dipole`` (real parts of the three complex components), ``delta_e_eV``,
    ``f`` and ``selected``.
    """
    lines = text.splitlines()
    n_states = int(_header_value(lines, "States"))
    eref = float(_header_value(lines, "Eref"))
    ninit = int(_header_value(lines, "Ninit"))
    natom = int(_header_value(lines, "Natom"))
    samples: list[list[dict[str, Any]]] = []
    i = 0
    while i < len(lines):
        fields = lines[i].split()
        if len(fields) == 2 and fields[0] == "Index":
            if lines[i + 1].strip() != "Atoms":
                raise ValueError(f"initconds.excited: expected 'Atoms' after line {i + 1}")
            j = i + 2 + natom
            if lines[j].split()[:1] != ["States"]:
                raise ValueError(f"initconds.excited: expected 'States' at line {j + 1}")
            rows = []
            for k in range(n_states):
                f = lines[j + 1 + k].split()
                if len(f) != 12 or f[-1] not in ("True", "False"):
                    raise ValueError(
                        f"initconds.excited: malformed state line {j + 2 + k}: {lines[j + 1 + k]!r}"
                    )
                rows.append(
                    {
                        "energy_hartree": float(f[1]),
                        "dipole": np.array([float(f[3]), float(f[5]), float(f[7])]),
                        "delta_e_eV": float(f[9]),
                        "f": float(f[10]),
                        "selected": f[11] == "True",
                    }
                )
            samples.append(rows)
            i = j + 1 + n_states
        else:
            i += 1
    if len(samples) != ninit:
        raise ValueError(f"initconds.excited: Ninit = {ninit} but {len(samples)} blocks found")
    return {"eref_hartree": eref, "n_states": n_states, "states": samples}
