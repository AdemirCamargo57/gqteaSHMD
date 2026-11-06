"""XYZ geometry files (coordinates in Angstrom)."""

from __future__ import annotations

from dataclasses import dataclass

from gqteashmd.chem.elements import normalize_symbol

Coord = tuple[float, float, float]


@dataclass(frozen=True)
class Geometry:
    symbols: tuple[str, ...]
    coords_angstrom: tuple[Coord, ...]
    comment: str = ""


def parse_xyz(text: str) -> Geometry:
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        raise ValueError("line 1: expected the number of atoms, file is empty")
    try:
        n_atoms = int(lines[0].split()[0])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"line 1: expected the number of atoms, got {lines[0]!r}") from exc
    if n_atoms < 1:
        raise ValueError("line 1: the geometry must contain at least one atom")
    comment = lines[1].strip() if len(lines) > 1 else ""
    atom_lines = lines[2:]
    if len(atom_lines) < n_atoms:
        raise ValueError(f"expected {n_atoms} atoms but found {len(atom_lines)} atom lines")
    if len(atom_lines) > n_atoms:
        raise ValueError(f"file contains more lines than the {n_atoms} atoms declared on line 1")
    symbols: list[str] = []
    coords: list[Coord] = []
    for offset, line in enumerate(atom_lines, start=3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"line {offset}: expected 'symbol x y z', got {line!r}")
        try:
            symbols.append(normalize_symbol(fields[0]))
            coords.append((float(fields[1]), float(fields[2]), float(fields[3])))
        except ValueError as exc:
            raise ValueError(f"line {offset}: {exc}") from exc
    return Geometry(tuple(symbols), tuple(coords), comment)


def format_xyz(geometry: Geometry) -> str:
    rows = [str(len(geometry.symbols)), geometry.comment]
    for symbol, (x, y, z) in zip(geometry.symbols, geometry.coords_angstrom, strict=True):
        rows.append(f"{symbol:<3}{x:16.10f}{y:16.10f}{z:16.10f}")
    return "\n".join(rows) + "\n"
