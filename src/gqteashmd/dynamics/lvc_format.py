"""Reader and writer of SHARC-style LVC model files (``V0.txt`` and ``LVC.template``).

The formats are those published in the SHARC 4.1 manual (section 6.4.1). Indices in the files are
1-based; in memory they are 0-based (``state0`` within its multiplicity, ``mode0`` over all 3N
modes). Energies and frequencies are in Eh, coordinates in bohr, masses in amu (they stay in amu
here). Every parse error is a ``ValueError`` naming the file, the line number and the expectation.

Scope: ``gamma`` and ``lambda_soc`` with a count above 0 are refused (the manual shows only the
count line); the ``Multipolar Density Fit`` section is skipped by its count; the dipole matrices
are stored but unused by the dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gqteashmd.dynamics.spin import check_counts, n_expanded


@dataclass(frozen=True, eq=False)
class V0Data:
    """Reference geometry and normal modes (``V0.txt``)."""

    symbols: tuple[str, ...]
    nuclear_charges: np.ndarray  # (N,)
    coords_bohr: np.ndarray  # (N, 3)
    masses_amu: np.ndarray  # (N,)
    frequencies_Eh: np.ndarray  # (3N,)
    modes: np.ndarray  # (3N, 3N), column i = mode i, rows (atom, xyz)


@dataclass(frozen=True, eq=False)
class LVCParameters:
    """Parameters of ``LVC.template`` (all indices 0-based)."""

    v0: V0Data
    counts: tuple[int, ...]  # as in the file, e.g. (2, 0, 1)
    epsilon: dict[tuple[int, int], float]  # (mult, state0) -> Eh
    kappa: dict[tuple[int, int, int], float]  # (mult, state0, mode0) -> Eh
    lam: dict[tuple[int, int, int, int], float]  # (mult, s0, t0, mode0), s0 < t0
    soc: np.ndarray | None  # complex (N_exp, N_exp), Eh
    dipoles: np.ndarray | None  # complex (3, N_exp, N_exp), e a0


def _same(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and bool(np.array_equal(a, b))


def v0_equal(a: V0Data, b: V0Data) -> bool:
    """Exact equality of two ``V0Data``."""
    return (
        a.symbols == b.symbols
        and _same(a.nuclear_charges, b.nuclear_charges)
        and _same(a.coords_bohr, b.coords_bohr)
        and _same(a.masses_amu, b.masses_amu)
        and _same(a.frequencies_Eh, b.frequencies_Eh)
        and _same(a.modes, b.modes)
    )


def lvc_equal(a: LVCParameters, b: LVCParameters) -> bool:
    """Exact equality of two ``LVCParameters``."""
    return (
        v0_equal(a.v0, b.v0)
        and a.counts == b.counts
        and a.epsilon == b.epsilon
        and a.kappa == b.kappa
        and a.lam == b.lam
        and _same(a.soc, b.soc)
        and _same(a.dipoles, b.dipoles)
    )


# --------------------------------------------------------------------------- reading helpers


def _collapse(text: str) -> str:
    return " ".join(text.split()).lower()


class _Lines:
    """Non-blank lines of a file with their 1-based line numbers."""

    def __init__(self, text: str, name: str) -> None:
        self.name = name
        self._items = [
            (i, line) for i, line in enumerate(text.splitlines(), start=1) if line.strip()
        ]
        self._last = len(text.splitlines())
        self._pos = 0

    def error(self, lineno: int, message: str) -> ValueError:
        return ValueError(f"{self.name}, line {lineno}: {message}")

    def peek(self) -> tuple[int, str] | None:
        return self._items[self._pos] if self._pos < len(self._items) else None

    def take(self, expected: str) -> tuple[int, str]:
        item = self.peek()
        if item is None:
            raise self.error(self._last + 1, f"unexpected end of file, expected {expected}")
        self._pos += 1
        return item

    def at_end(self) -> bool:
        return self._pos >= len(self._items)


def _float(lines: _Lines, lineno: int, token: str, what: str) -> float:
    try:
        value = float(token.replace("D", "E").replace("d", "e"))
    except ValueError:
        raise lines.error(lineno, f"expected {what}, got {token!r}") from None
    if not math.isfinite(value):
        raise lines.error(lineno, f"expected a finite {what}, got {token!r}")
    return value


def _int(lines: _Lines, lineno: int, token: str, what: str) -> int:
    try:
        return int(token)
    except ValueError:
        raise lines.error(lineno, f"expected {what} (an integer), got {token!r}") from None


def _count(lines: _Lines, what: str) -> tuple[int, int]:
    lineno, line = lines.take(f"the number of {what} entries")
    tokens = line.split()
    if len(tokens) != 1:
        raise lines.error(lineno, f"expected a single count for {what}, got {line.strip()!r}")
    n = _int(lines, lineno, tokens[0], f"the number of {what} entries")
    if n < 0:
        raise lines.error(lineno, f"the number of {what} entries must not be negative, got {n}")
    return lineno, n


# --------------------------------------------------------------------------- V0.txt

_V0_HEADERS = ("geometry", "frequencies", "mass-weighted normal modes")


def read_v0(text: str, name: str = "V0.txt") -> V0Data:
    """Parse the text of a ``V0.txt`` file."""
    lines = _Lines(text, name)

    def header(expected: str) -> None:
        lineno, line = lines.take(f"the header {expected!r}")
        if _collapse(line) != expected:
            raise lines.error(lineno, f"expected the header {expected!r}, got {line.strip()!r}")

    def block_until_header() -> list[tuple[int, str]]:
        rows: list[tuple[int, str]] = []
        while (item := lines.peek()) is not None and _collapse(item[1]) not in _V0_HEADERS:
            rows.append(lines.take("a data line"))
        return rows

    header("geometry")
    symbols: list[str] = []
    numbers: list[list[float]] = []
    for lineno, line in block_until_header():
        tokens = line.split()
        if len(tokens) != 6:
            raise lines.error(
                lineno, f"expected 'element charge x y z mass' (6 fields), got {len(tokens)}"
            )
        symbols.append(tokens[0])
        numbers.append(
            [_float(lines, lineno, t, "a number") for t in tokens[1:]],
        )
    if not symbols:
        raise lines.error(lines._last + 1, "the Geometry section is empty")
    n_atoms = len(symbols)
    geometry = np.array(numbers, dtype=float)

    header("frequencies")
    freq_values: list[float] = []
    last = lines._last + 1
    for lineno, line in block_until_header():
        last = lineno
        freq_values.extend(_float(lines, lineno, t, "a frequency") for t in line.split())
    if len(freq_values) != 3 * n_atoms:
        raise lines.error(
            last, f"expected {3 * n_atoms} frequencies (3N), found {len(freq_values)}"
        )

    header("mass-weighted normal modes")
    modes = np.empty((3 * n_atoms, 3 * n_atoms))
    for row in range(3 * n_atoms):
        lineno, line = lines.take(f"mode matrix row {row + 1} of {3 * n_atoms}")
        tokens = line.split()
        if len(tokens) != 3 * n_atoms:
            raise lines.error(
                lineno,
                f"expected {3 * n_atoms} values in mode matrix row {row + 1}, got {len(tokens)}",
            )
        modes[row] = [_float(lines, lineno, t, "a matrix element") for t in tokens]
    extra = lines.peek()
    if extra is not None:
        raise lines.error(extra[0], "unexpected data after the mode matrix")

    return V0Data(
        symbols=tuple(symbols),
        nuclear_charges=geometry[:, 0].copy(),
        coords_bohr=geometry[:, 1:4].copy(),
        masses_amu=geometry[:, 4].copy(),
        frequencies_Eh=np.array(freq_values, dtype=float),
        modes=modes,
    )


def format_v0(v0: V0Data) -> str:
    """Text of a ``V0.txt`` file (17 significant digits, so that the round trip is exact)."""
    out = ["Geometry"]
    for s, z, xyz, m in zip(
        v0.symbols, v0.nuclear_charges, v0.coords_bohr, v0.masses_amu, strict=True
    ):
        out.append(f" {s:<2s} {z:.16e} {xyz[0]:.16e} {xyz[1]:.16e} {xyz[2]:.16e} {m:.16e}")
    out.append("Frequencies")
    out.append(" ".join(f"{w:.16e}" for w in v0.frequencies_Eh))
    out.append("Mass-weighted  normal  modes")
    out.extend(" ".join(f"{x:.16e}" for x in row) for row in v0.modes)
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- LVC.template

_SECTIONS_COUNTED = ("epsilon", "kappa", "lambda", "lambda_soc", "gamma", "multipolar density fit")
_MATRIX_SECTIONS = (
    "soc r",
    "soc i",
    "dmx r",
    "dmx i",
    "dmy r",
    "dmy i",
    "dmz r",
    "dmz i",
)


def template_v0_path(text: str, name: str = "LVC.template") -> str:
    """The V0 path on the first line of a template."""
    lines = _Lines(text, name)
    _, line = lines.take("the V0 file path on line 1")
    return line.strip()


def parse_lvc_template(text: str, v0: V0Data, name: str = "LVC.template") -> LVCParameters:
    """Parse the text of an ``LVC.template`` for the given reference data."""
    lines = _Lines(text, name)
    lines.take("the V0 file path on line 1")
    lineno, line = lines.take("the state counts (singlets doublets triplets)")
    counts_in = [_int(lines, lineno, t, "a state count") for t in line.split()]
    try:
        counts = check_counts(counts_in)
    except ValueError as exc:
        raise lines.error(lineno, str(exc)) from None
    n_exp = n_expanded(counts)
    n_modes = v0.modes.shape[0]

    def check_state(lineno: int, mult: int, state1: int, what: str) -> int:
        if mult not in (1, 3):
            raise lines.error(lineno, f"{what}: multiplicity must be 1 or 3, got {mult}")
        n = counts[mult - 1] if mult - 1 < len(counts) else 0
        if not 1 <= state1 <= n:
            raise lines.error(
                lineno,
                f"{what}: state index {state1} is outside 1..{n} for multiplicity {mult} "
                f"(indices in the file are 1-based)",
            )
        return state1 - 1

    def check_mode(lineno: int, mode1: int) -> int:
        if not 1 <= mode1 <= n_modes:
            raise lines.error(
                lineno,
                f"mode index {mode1} is outside 1..{n_modes} (indices in the file are 1-based)",
            )
        return mode1 - 1

    epsilon: dict[tuple[int, int], float] = {}
    kappa: dict[tuple[int, int, int], float] = {}
    lam: dict[tuple[int, int, int, int], float] = {}
    matrices: dict[str, np.ndarray] = {}
    seen: set[str] = set()

    while not lines.at_end():
        lineno, line = lines.take("a section keyword")
        key = _collapse(line)
        if key not in _SECTIONS_COUNTED and key not in _MATRIX_SECTIONS:
            raise lines.error(lineno, f"unknown section keyword {line.strip()!r}")
        if key in seen:
            raise lines.error(lineno, f"duplicate section {key!r}")
        seen.add(key)

        if key in _MATRIX_SECTIONS:
            matrix = np.empty((n_exp, n_exp))
            for row in range(n_exp):
                rl, rt = lines.take(f"row {row + 1} of {n_exp} of section {key!r}")
                tokens = rt.split()
                if len(tokens) != n_exp:
                    raise lines.error(
                        rl,
                        f"expected {n_exp} values in row {row + 1} of section {key!r} "
                        f"(N_exp = {n_exp}), got {len(tokens)}",
                    )
                matrix[row] = [_float(lines, rl, t, "a matrix element") for t in tokens]
            matrices[key] = matrix
            continue

        count_line, n = _count(lines, key)
        if key in ("gamma", "lambda_soc"):
            if n > 0:
                raise lines.error(
                    count_line,
                    f"section {key!r} with {n} entries is not supported "
                    f"(only a count of 0 is accepted)",
                )
            continue
        for _ in range(n):
            el, et = lines.take(f"an entry of section {key!r}")
            tokens = et.split()
            if key == "multipolar density fit":
                continue
            width = {"epsilon": 3, "kappa": 4, "lambda": 5}[key]
            if len(tokens) != width:
                raise lines.error(
                    el, f"expected {width} fields in an entry of {key!r}, got {len(tokens)}"
                )
            ints = [_int(lines, el, t, "an index") for t in tokens[:-1]]
            value = _float(lines, el, tokens[-1], "a value")
            mult = ints[0]
            if key == "epsilon":
                e_key = (mult, check_state(el, mult, ints[1], "epsilon"))
                if e_key in epsilon:
                    raise lines.error(el, f"duplicate epsilon entry {tuple(ints)}")
                epsilon[e_key] = value
            elif key == "kappa":
                k_key = (
                    mult,
                    check_state(el, mult, ints[1], "kappa"),
                    check_mode(el, ints[2]),
                )
                if k_key in kappa:
                    raise lines.error(el, f"duplicate kappa entry {tuple(ints)}")
                kappa[k_key] = value
            else:
                s0 = check_state(el, mult, ints[1], "lambda")
                t0 = check_state(el, mult, ints[2], "lambda")
                if s0 == t0:
                    raise lines.error(
                        el, "lambda couples two different states, got a diagonal entry"
                    )
                a, b = sorted((s0, t0))
                l_key = (mult, a, b, check_mode(el, ints[3]))
                if l_key in lam:
                    raise lines.error(el, f"duplicate lambda entry {tuple(ints)}")
                lam[l_key] = value

    soc: np.ndarray | None = None
    if "soc r" in matrices or "soc i" in matrices:
        soc = (
            matrices.get("soc r", np.zeros((n_exp, n_exp)))
            + 1j * matrices.get("soc i", np.zeros((n_exp, n_exp)))
        ).astype(complex)
    dipoles: np.ndarray | None = None
    if any(k.startswith("dm") for k in matrices):
        dipoles = np.zeros((3, n_exp, n_exp), dtype=complex)
        for axis, letter in enumerate("xyz"):
            re = matrices.get(f"dm{letter} r")
            im = matrices.get(f"dm{letter} i")
            if re is not None:
                dipoles[axis] += re
            if im is not None:
                dipoles[axis] += 1j * im

    return LVCParameters(
        v0=v0, counts=counts, epsilon=epsilon, kappa=kappa, lam=lam, soc=soc, dipoles=dipoles
    )


def read_lvc_template(path: Path) -> LVCParameters:
    """Read a template and the ``V0.txt`` it names (a relative path is resolved against it)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    v0_ref = Path(template_v0_path(text, str(path)))
    v0_path = v0_ref if v0_ref.is_absolute() else path.parent / v0_ref
    try:
        v0_text = v0_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{path}, line 1: cannot read the V0 file {v0_path}: {exc}") from None
    return parse_lvc_template(text, read_v0(v0_text, str(v0_path)), str(path))


def _matrix_lines(title: str, matrix: np.ndarray) -> list[str]:
    out = [title]
    out.extend("  ".join(f"{x + 0.0:.12e}" for x in row) for row in matrix)
    return out


def format_lvc_template(params: LVCParameters, v0_name: str = "V0.txt") -> str:
    """Text of an ``LVC.template`` (values with ``%.12e``; entries sorted)."""
    out = [v0_name, " ".join(str(n) for n in params.counts)]
    out.append("epsilon")
    out.append(str(len(params.epsilon)))
    for (mult, s), value in sorted(params.epsilon.items()):
        out.append(f"{mult:4d} {s + 1:6d} {value:.12e}")
    out.append("kappa")
    out.append(str(len(params.kappa)))
    for (mult, s, mode), value in sorted(params.kappa.items()):
        out.append(f"{mult:4d} {s + 1:6d} {mode + 1:6d} {value:.12e}")
    out.append("lambda")
    out.append(str(len(params.lam)))
    for (mult, s, t, mode), value in sorted(params.lam.items()):
        out.append(f"{mult:4d} {s + 1:6d} {t + 1:6d} {mode + 1:6d} {value:.12e}")
    out.extend(["lambda_soc", "0", "gamma", "0"])
    if params.soc is not None:
        out.extend(_matrix_lines("SOC  R", params.soc.real))
        out.extend(_matrix_lines("SOC  I", params.soc.imag))
    if params.dipoles is not None:
        for axis, letter in enumerate("XYZ"):
            out.extend(_matrix_lines(f"DM{letter}  R", params.dipoles[axis].real))
            out.extend(_matrix_lines(f"DM{letter}  I", params.dipoles[axis].imag))
    return "\n".join(out) + "\n"
