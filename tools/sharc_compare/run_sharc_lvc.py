"""Run SHARC 4.1 with its LVC interface as a black box: single points of SHARC_LVC.py, and
sharc.x trajectories with SHARC_LVC.py (or our own QM.out provider) behind ``QM/runQM.sh``.

SHARC is only *executed*; nothing from SHARC is imported or copied. Formats come from the
public SHARC 4.1 manual: running an interface (``$SHARC/SHARC_<NAME>.py QM.in`` in the directory
with its input files), the LVC files V0.txt and LVC.template (6.4.1), QM.in (6.34.1) and QM.out
(6.34.2, parsed by our ``qmout.py``), the sharc.x ``input`` keywords (4.1), the ``geom`` and
``veloc`` files (4.2, 4.3), ``QM/runQM.sh`` (3.1.1) and output.dat (5.3.1). Every writer and
parser below is our own code.

Measured behaviour of this installation (the Task 9 and Task 10 reports of P7a-2 have the
details):

- QM.in coordinates need a decimal point (SHARC rejected ``C 0 0.05 ...`` with "line is not
  xyz"), so ``format_qm_in`` writes ``%.16e``.
- ``init`` is reported as deprecated ("Calculation control via STEP file and 'step' keyword"),
  so the first call of a sequence uses ``step 0`` and the next ``step 1`` with the same
  ``savedir`` (needed for ``OVERLAP``).
- With ``SOC`` requested and DMX/DMY/DMZ blocks in the template, SHARC_LVC.py stops with
  "Cannot cast ufunc 'matmul' input 1 from dtype('float64') to dtype('complex128') with
  casting rule 'no'" even when ``DM`` is not requested; ``without_dipoles`` removes those
  blocks (the dipoles are not used by the dynamics and are not compared).
- sharc.x writes the QM.in of its interface in angstrom with 7 decimals (positions and
  velocities), so the interface evaluates the model at a geometry rounded to about 5e-8
  angstrom; sharc.x itself propagates the unrounded geometry. It always requests ``DM``.
- output.dat prints every number with 13 significant digits (``0.1234567890123E+000``).

The swap test (strategy 5.4 (c): SHARC's driver with our electronic structure) puts our
``LVCModel`` behind ``QM/runQM.sh``: ``QmServer`` answers in a thread of the calling process,
and the shell script written by ``write_server_runqm`` hands each request to it through two
named pipes, so no Python starts per step.

Example (WSL), one point of a template from tests/data/lvc:
  python tools/sharc_compare/run_sharc_lvc.py --sharc-home /home/ajc/sharc4-4.1 \\
      --python /home/ajc/miniforge3/envs/sharc4.1/bin/python \\
      --template tests/data/lvc/lvc_soc.template --states "2 0 2" --soc \\
      --geometry geom.xyz --work-dir /tmp/sharc-lvc
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import os
import re
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.chem.elements import atomic_number
from gqteashmd.core.units import HARTREE_IN_EV

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qmout import parse_qm_out  # noqa: E402
from run_sharc_model import AU_TIME_IN_FS as AU_TIME_IN_FS  # noqa: E402  (re-exported)
from run_sharc_model import (  # noqa: E402
    BOHR_IN_ANGSTROM,
    SHARC_X_COMMAND,
    run_in_session,
    sharc_environment,
)

LVC_BLOCKS = ("hamiltonian", "gradients")  # always present; dipoles only with DM
_DM_HEADER = re.compile(r"^DM[XYZ]\s+[RI]$")
PROVIDERS = ("sharc_lvc", "ours")
# one thread per process for SHARC's Python and NumPy (trajectories run in parallel)
SINGLE_THREAD = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


# --- single points of SHARC_LVC.py (Task 9) ------------------------------------------------------


def format_qm_in(
    symbols: Sequence[str],
    coords_bohr: np.ndarray,
    *,
    states: Sequence[int],
    savedir: str,
    step: int,
    soc: bool = False,
    grad: bool = True,
    nacdr: bool = True,
    overlap: bool = False,
    comment: str = "gqteashmd",
) -> str:
    """QM.in (manual 6.34.1) in bohr with full precision; ``SOC`` or else ``H`` is requested."""
    coords = np.asarray(coords_bohr, dtype=float)
    if coords.shape != (len(symbols), 3):
        raise ValueError(
            f"coordinates must have shape ({len(symbols)}, 3) for {len(symbols)} atoms; got "
            f"{coords.shape}"
        )
    lines = [str(len(symbols)), comment]
    lines += [
        f"{s} " + " ".join(f"{x + 0.0:.16e}" for x in row)
        for s, row in zip(symbols, coords, strict=True)
    ]
    lines += [
        f"step {int(step)}",
        f"savedir {savedir}",
        "unit bohr",
        "states " + " ".join(str(int(n)) for n in states),
        "SOC" if soc else "H",
    ]
    if grad:
        lines.append("GRAD")
    if nacdr:
        lines.append("NACDR")
    if overlap:
        lines.append("OVERLAP")
    return "\n".join(lines) + "\n"


def without_dipoles(template_text: str) -> str:
    """The LVC.template text without its DMX/DMY/DMZ R/I blocks (header + square matrix)."""
    lines = template_text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        if _DM_HEADER.match(" ".join(lines[i].split())):
            n = len(lines[i + 1].split()) if i + 1 < len(lines) else 0
            i += 1 + n
            continue
        out.append(lines[i])
        i += 1
    return "".join(out)


def run_single_point(
    work_dir: Path,
    qm_in_text: str,
    template_text: str,
    v0_text: str,
    *,
    sharc_home: Path,
    sharc_python: str,
    timeout: float = 600.0,
    required: tuple[str, ...] = LVC_BLOCKS,
) -> dict[str, Any]:
    """Write QM.in, LVC.template and V0.txt in ``work_dir``, run ``$SHARC/SHARC_LVC.py QM.in``
    there and return the parsed QM.out blocks (``qmout.parse_qm_out``).

    ``work_dir`` may exist (a later step of the same savedir). The interface runs in its own
    session (``run_sharc_model.run_in_session``); on a timeout its whole process group is
    killed and ``RuntimeError`` is raised, as for a non-zero exit or a missing QM.out (both
    with the interface's stderr and stdout).
    """
    sharc_home = Path(sharc_home)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    for name, text in (("QM.in", qm_in_text), ("LVC.template", template_text), ("V0.txt", v0_text)):
        (work_dir / name).write_bytes(text.encode())
    qm_out = work_dir / "QM.out"
    qm_out.unlink(missing_ok=True)
    proc = run_in_session(
        [sharc_python, str(sharc_home / "bin" / "SHARC_LVC.py"), "QM.in"],
        cwd=work_dir,
        env=sharc_environment(sharc_home),
        timeout=timeout,
        what="SHARC_LVC.py",
    )
    output = f"\n{proc.stderr[-3000:]}\n{proc.stdout[-2000:]}"
    if proc.returncode != 0:
        raise RuntimeError(f"SHARC_LVC.py failed ({proc.returncode}) in {work_dir}:{output}")
    if not qm_out.is_file():
        raise RuntimeError(f"SHARC_LVC.py wrote no QM.out in {work_dir}:{output}")
    return parse_qm_out(qm_out.read_text(), required=required)


# --- sharc.x trajectories: input, geom, veloc (manual 4.1-4.3) ----------------------------------


@dataclass(frozen=True)
class LvcSharcOptions:
    """The sharc.x settings we compare; ``from_dynamics_options`` maps our DynamicsOptions."""

    coupling: str = "nacdr"  # nacdr | overlap
    hopping: str = "sharc"  # sharc | off
    ekincorrect: str = "parallel_vel"  # parallel_vel | parallel_nac
    reflect_frustrated: str = "none"  # none | parallel_vel | parallel_nac
    decoherence: str = "none"  # none | edc
    decoherence_param: float = 0.1
    gradcorrect: str = "ngt"  # ngt | none (nogradcorrect)
    nsubsteps: int = 25
    eselect_eV: float | None = None  # None: grad_all, nac_all

    @classmethod
    def from_dynamics_options(cls, options) -> LvcSharcOptions:
        """Map our options to SHARC's. SHARC has neither our ``"fssh"`` hopping nor the
        Granucci-Persico EDC (its EDC is our ``"sharc"`` convention): both are ``ValueError``.
        SHARC interpolates at the substep end points (our ``"endpoint"``);
        ``substep_interpolation`` is not passed. ``eselect_Eh`` becomes eV."""
        hopping = {"sharc": "sharc", "none": "off"}.get(options.hopping)
        if hopping is None:
            raise ValueError(f"SHARC has no {options.hopping!r} hopping (use sharc or none)")
        if options.decoherence == "edc" and options.edc_convention != "sharc":
            raise ValueError(
                f"SHARC's EDC is the 'sharc' convention; got edc_convention = "
                f"{options.edc_convention!r}"
            )
        reflect = "none" if options.frustrated == "keep" else options.ekin_correction
        eselect = None if options.eselect_Eh is None else options.eselect_Eh * HARTREE_IN_EV
        return cls(
            coupling=options.coupling,
            hopping=hopping,
            ekincorrect=options.ekin_correction,
            reflect_frustrated=reflect,
            decoherence=options.decoherence,
            decoherence_param=float(options.decoherence_param_Eh),
            gradcorrect=options.gradient_transformation,
            nsubsteps=int(options.nsubsteps),
            eselect_eV=eselect,
        )


def lvc_input(
    n_steps: int,
    dt_fs: float,
    options: LvcSharcOptions,
    seed: int,
    *,
    states: Sequence[int],
    state_mch: int,
    detailed: bool = True,
    output_stride: int | None = None,
) -> str:
    """The sharc.x ``input`` (manual 4.1) for a spin-orbit trajectory in the diagonal basis.

    ``state_mch`` is our 0-based MCH (canonical) index; SHARC's ``state`` is 1-based. The run
    length is set with ``tmax`` (``nsteps`` was ignored by this sharc.x, P7a-1). ``detailed``
    writes the gradients, NACs and overlaps of every printed step; ``output_stride`` sets
    ``output_dat_steps``.
    """
    lines = [
        f"printlevel {2 if detailed else 0}",
        'geomfile "geom"',
        "veloc external",
        'velocfile "veloc"',
        "nstates " + " ".join(str(int(n)) for n in states),
        "charge " + " ".join("0" for _ in states),
        f"state {int(state_mch) + 1} mch",
        "coeff auto",
        f"stepsize {dt_fs!r}",
        f"tmax {(n_steps + 0.25) * dt_fs!r}",
        f"nsubsteps {options.nsubsteps}",
        "method tsh",
        "surf diagonal",
        "spinorbit",
        f"coupling {options.coupling}",
        f"eeom {'ld' if options.coupling == 'overlap' else 'li'}",
        "nogradcorrect" if options.gradcorrect == "none" else f"gradcorrect {options.gradcorrect}",
        f"ekincorrect {options.ekincorrect}",
        f"reflect_frustrated {options.reflect_frustrated}",
        f"decoherence_scheme {options.decoherence}",
        f"decoherence_param {options.decoherence_param!r}",
        f"hopping_procedure {options.hopping}",
    ]
    if options.eselect_eV is None:
        lines += ["grad_all", "nac_all"]
    else:
        lines += ["grad_select", "nac_select", f"eselect {options.eselect_eV!r}"]
    lines += [f"rngseed {int(seed)}", "output_format ascii", "nowrite_restart_files"]
    lines += ["write_grad", "write_nacdr", "write_overlap"] if detailed else ["nowrite_overlap"]
    if output_stride is not None:
        lines.append(f"output_dat_steps {int(output_stride)}")
    return "\n".join(lines) + "\n"


def format_geom(symbols: Sequence[str], masses_amu: Sequence[float], coords_bohr) -> str:
    """The ``geom`` file (manual 4.2): symbol, nuclear charge, x y z in bohr, mass in amu; every
    number with a decimal point and full precision."""
    coords = np.asarray(coords_bohr, dtype=float)
    if coords.shape != (len(symbols), 3) or len(masses_amu) != len(symbols):
        raise ValueError(
            f"geom: coordinates must have shape ({len(symbols)}, 3) and {len(symbols)} masses; "
            f"got {coords.shape} and {len(masses_amu)}"
        )
    return "".join(
        f"{s} {float(atomic_number(s))!r} "
        + " ".join(f"{x + 0.0:.16e}" for x in row)
        + f" {float(m)!r}\n"
        for s, row, m in zip(symbols, coords, masses_amu, strict=True)
    )


def format_veloc(velocities_au) -> str:
    """The ``veloc`` file (manual 4.3): one atom per line, bohr per atomic time unit."""
    v = np.asarray(velocities_au, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"veloc: velocities must have shape (natom, 3); got {v.shape}")
    return "".join(" ".join(f"{x + 0.0:.16e}" for x in row) + "\n" for row in v)


# --- output.dat (manual 5.3.1) --------------------------------------------------------------------


@dataclass
class LvcTrajectory:
    """The printed steps of one sharc.x output.dat (atomic units; 0-based state indices).

    ``u[k][alpha, a]``: row = MCH state, column = diagonal state. ``overlaps[k]`` is None at a
    step without an overlap block. ``gradients`` (k, n, 3N) and ``nacs`` (k, n, n, 3N) are None
    unless every printed step has them.
    """

    dt_au: float
    nsteps_header: int
    nstates_m: tuple[int, ...]
    masses_me: np.ndarray
    steps: np.ndarray
    hamiltonian: np.ndarray
    u: np.ndarray
    overlaps: list[np.ndarray | None]
    c: np.ndarray
    probabilities: np.ndarray
    e_kinetic: np.ndarray
    active: np.ndarray
    active_mch: np.ndarray
    random_numbers: np.ndarray
    q: np.ndarray
    v: np.ndarray
    gradients: np.ndarray | None
    nacs: np.ndarray | None


def _numbers(line: str) -> list[float]:
    return [float(t.replace("D", "E")) for t in line.split()]


def _block_rows(title: str, body: list[str], n_rows: int, n_cols: int, step: Any) -> np.ndarray:
    """``n_rows`` rows of exactly ``n_cols`` numbers; ``ValueError`` naming the block."""
    where = f"output.dat: the '{title}' block of step {step}"
    if len(body) != n_rows:
        raise ValueError(f"{where} has {len(body)} lines, expected {n_rows}")
    try:
        rows = [_numbers(line) for line in body]
    except ValueError as exc:
        raise ValueError(f"{where} is malformed: {exc}") from None
    if any(len(r) != n_cols for r in rows):
        raise ValueError(f"{where} needs {n_cols} numbers per line")
    return np.array(rows, dtype=float)


def _complex_rows(rows: np.ndarray) -> np.ndarray:
    return rows[:, 0::2] + 1j * rows[:, 1::2]


def parse_lvc_output_dat(path: Path) -> LvcTrajectory:
    """Parse output.dat: keyword header, header arrays, then the blocks of every printed step
    (manual 5.3.1). Unknown blocks are skipped; a known block with the wrong shape is a
    ``ValueError`` naming it."""
    lines = Path(path).read_text().splitlines()
    header: dict[str, list[str]] = {}
    i = 0
    while i < len(lines) and not lines[i].startswith("****"):
        tokens = lines[i].split()
        if len(tokens) >= 2:
            header[tokens[0].lower()] = tokens[1:]
        i += 1
    try:
        nstates_m = tuple(int(x) for x in header["nstates_m"])
        natom = int(header["natom"][0])
        dt_au = float(header["dtstep"][0])
        nsteps = int(header["nsteps"][0])
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"{path}: incomplete output.dat header ({exc})") from None
    n = sum((m + 1) * count for m, count in enumerate(nstates_m))
    f = 3 * natom

    blocks: list[tuple[str, list[str]]] = []
    for line in lines[i:]:
        if line.startswith("!"):
            blocks.append((line[1:].strip(), []))
        elif line.startswith("****"):
            continue
        elif blocks:
            blocks[-1][1].append(line)

    masses = np.zeros(0)
    steps: list[dict[str, Any]] = []
    for title, body in blocks:
        number = title.split()[0] if title.split() else ""
        if title == "Atomic masses":
            masses = _block_rows(title, body, natom, 1, "header")[:, 0]
            continue
        if number == "0" and title.startswith("0 Step"):
            if not body or not body[0].split():
                raise ValueError(f"output.dat: the '{title}' block has no step number")
            steps.append({"step": int(body[0].split()[0]), "grad": [], "nac": [], "S": None})
            continue
        if not steps or not number.isdigit():
            continue
        s = steps[-1]
        label = s["step"]
        if number == "1":
            s["H"] = _complex_rows(_block_rows(title, body, n, 2 * n, label))
        elif number == "2":
            s["U"] = _complex_rows(_block_rows(title, body, n, 2 * n, label))
        elif number == "3":
            _block_rows(title, body, n, 2 * n, label)  # dipoles: checked, not used
        elif number == "4":
            s["S"] = _complex_rows(_block_rows(title, body, n, 2 * n, label))
        elif number == "5":
            s["c"] = _complex_rows(_block_rows(title, body, n, 2, label))[:, 0]
        elif number == "6":
            s["p"] = _block_rows(title, body, n, 1, label)[:, 0]
        elif number == "7":
            s["ekin"] = float(_block_rows(title, body, 1, 1, label)[0, 0])
        elif number == "8":
            row = _block_rows(title, body, 1, 2, label)[0]
            s["active"], s["active_mch"] = int(row[0]) - 1, int(row[1]) - 1
        elif number == "9":
            s["xi"] = float(_block_rows(title, body, 1, 1, label)[0, 0])
        elif number == "11":
            s["q"] = _block_rows(title, body, natom, 3, label).reshape(-1)
        elif number == "12":
            s["v"] = _block_rows(title, body, natom, 3, label).reshape(-1)
        elif number == "15":
            s["grad"].append(_block_rows(title, body, natom, 3, label).reshape(-1))
        elif number == "16":
            s["nac"].append(_block_rows(title, body, natom, 3, label).reshape(-1))

    needed = ("H", "U", "c", "p", "ekin", "active", "xi", "q", "v")
    for s in steps:
        missing = [key for key in needed if key not in s]
        if missing:
            raise ValueError(f"output.dat: step {s['step']} lacks the blocks {missing}")
    k = len(steps)
    gradients = nacs = None
    if steps and all(len(s["grad"]) == n for s in steps):
        gradients = np.array([s["grad"] for s in steps])
    if steps and all(len(s["nac"]) == n * n for s in steps):
        nacs = np.array([s["nac"] for s in steps]).reshape(k, n, n, f)
    return LvcTrajectory(
        dt_au=dt_au,
        nsteps_header=nsteps,
        nstates_m=nstates_m,
        masses_me=masses,
        steps=np.array([s["step"] for s in steps], dtype=int),
        hamiltonian=np.array([s["H"] for s in steps]).reshape(k, n, n),
        u=np.array([s["U"] for s in steps]).reshape(k, n, n),
        overlaps=[s["S"] for s in steps],
        c=np.array([s["c"] for s in steps]).reshape(k, n),
        probabilities=np.array([s["p"] for s in steps]).reshape(k, n),
        e_kinetic=np.array([s["ekin"] for s in steps]),
        active=np.array([s["active"] for s in steps], dtype=int),
        active_mch=np.array([s["active_mch"] for s in steps], dtype=int),
        random_numbers=np.array([s["xi"] for s in steps]),
        q=np.array([s["q"] for s in steps]).reshape(k, f),
        v=np.array([s["v"] for s in steps]).reshape(k, f),
        gradients=gradients,
        nacs=nacs,
    )


# --- our own QM.out provider (swap test) ---------------------------------------------------------

_CONTROL_KEYS = frozenset({"unit", "states", "charge", "dt", "step", "savedir", "retain"})
_REQUEST_KEYS = frozenset({"h", "soc", "dm", "grad", "nacdr", "overlap", "phases"})


@dataclass(frozen=True)
class QmRequest:
    """What sharc.x asked for in one QM.in (manual 6.34.1)."""

    symbols: tuple[str, ...]
    coords_bohr: np.ndarray  # (natom, 3)
    step: int
    states: tuple[int, ...]
    requests: frozenset[str]  # lower-case request keywords


def parse_sharc_qm_in(text: str) -> QmRequest:
    """Read a QM.in written by sharc.x: atom lines (symbol, x y z, optional velocities), the
    control keywords and the request keywords. ``select`` lists (``... select`` up to ``end``)
    are skipped: our provider always computes everything. ``ValueError`` names QM.in."""
    lines = text.splitlines()
    try:
        natom = int(lines[0].split()[0])
        atoms = [lines[2 + a].split() for a in range(natom)]
        symbols = tuple(t[0] for t in atoms)
        coords = np.array([[float(x) for x in t[1:4]] for t in atoms], dtype=float)
    except (IndexError, ValueError) as exc:
        raise ValueError(f"QM.in: cannot read the geometry ({exc})") from None
    if coords.shape != (natom, 3):
        raise ValueError(f"QM.in: expected {natom} atoms with 3 coordinates each")
    unit, step, states = "angstrom", 0, ()
    requests: set[str] = set()
    k = 2 + natom
    while k < len(lines):
        tokens = lines[k].split()
        k += 1
        if not tokens:
            continue
        key = tokens[0].lower()
        try:
            if key == "unit":
                unit = tokens[1].lower()
            elif key == "step":
                step = int(tokens[1])
            elif key == "states":
                states = tuple(int(x) for x in tokens[1:])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"QM.in: cannot read {lines[k - 1]!r} ({exc})") from None
        if key in _REQUEST_KEYS:
            requests.add(key)
            if len(tokens) > 1 and tokens[1].lower() == "select":
                while k < len(lines) and lines[k].strip().lower() != "end":
                    k += 1
                k += 1
        elif key not in _CONTROL_KEYS:
            raise ValueError(f"QM.in: unknown keyword {tokens[0]!r}")
    if unit not in ("angstrom", "bohr"):
        raise ValueError(f"QM.in: unknown unit {unit!r}")
    if unit == "angstrom":
        coords = coords / BOHR_IN_ANGSTROM
    return QmRequest(symbols, coords, step, states, frozenset(requests))


def _complex_lines(matrix: np.ndarray) -> list[str]:
    m = np.asarray(matrix, dtype=complex)
    return [" ".join(f"{z.real + 0.0:.16e} {z.imag + 0.0:.16e}" for z in row) for row in m]


def format_qm_out(snapshot, states, requests: frozenset[str]) -> str:
    """QM.out (manual 6.34.2) of one of our ``ElectronicSnapshot`` objects.

    ``states`` are the model's ``ExpandedState`` objects in canonical order. ``soc`` gives the
    full MCH Hamiltonian, ``h`` alone its diagonal; ``dm`` zero dipoles (not compared);
    ``grad``/``nacdr`` every vector with its ``m s ms`` labels; ``overlap`` the snapshot
    overlaps (the model aligns signs to the previous step, so ``phases`` are all 1). Numbers
    are written with 17 significant digits.
    """
    n = len(states)
    f = np.asarray(snapshot.gradients).shape[1]
    natom = f // 3
    counts = [sum(1 for s in states if s.multiplicity == 1)]
    n3 = sum(1 for s in states if s.multiplicity == 3) // 3
    if n3:
        counts += [0, n3]

    def label(k: int, s) -> str:
        return f"m{k} {s.multiplicity} s{k} {s.index + 1} ms{k} {s.ms}"

    bras = [label(1, s) for s in states]
    kets = [label(2, s) for s in states]
    out = [
        "! 0 Basic information",
        "states " + " ".join(str(c) for c in counts),
        f"nmstates {n}",
        f"natom {natom}",
        "npc 0",
        "charges " + " ".join("0" for _ in counts),
    ]
    h = np.asarray(snapshot.hamiltonian, dtype=complex)
    if "soc" not in requests:
        h = np.diag(np.asarray(snapshot.energies, dtype=float)).astype(complex)
    out += [f"! 1 Hamiltonian Matrix ({n}x{n}, complex)", f"{n} {n}", *_complex_lines(h)]
    if "dm" in requests:
        out.append(f"! 2 Dipole Moment Matrices (3x{n}x{n}, complex)")
        for _ in range(3):
            out += [f"{n} {n}", *_complex_lines(np.zeros((n, n)))]
    if "grad" in requests:
        out.append(f"! 3 Gradient Vectors ({n}x{natom}x3, real)")
        for a in range(n):
            out.append(f"{natom} 3 ! {bras[a]}")
            out += _vector_lines(snapshot.gradients[a])
    if "overlap" in requests:
        if snapshot.overlaps is None:
            raise ValueError("OVERLAP requested but the snapshot has no previous step")
        out += [f"! 6 Overlap matrix ({n}x{n}, complex)", f"{n} {n}"]
        out += _complex_lines(snapshot.overlaps)
    if "nacdr" in requests:
        out.append(f"! 5 Non-adiabatic couplings (ddr) ({n}x{n}x{natom}x3, real)")
        for a in range(n):
            for b in range(n):
                out.append(f"{natom} 3 ! {bras[a]}   {kets[b]}")
                out += _vector_lines(snapshot.nacs[a, b])
    if "phases" in requests:
        out += [f"! 7 Wave function phases ({n}x1, complex)", str(n)]
        out += [f"{1.0:.16e} {0.0:.16e}"] * n
    out += ["! 8 Runtime", f"{0.0:.16e}"]
    return "\n".join(out) + "\n"


def _vector_lines(vector) -> list[str]:
    rows = np.asarray(vector, dtype=float).reshape(-1, 3)
    return [" ".join(f"{x + 0.0:.16e}" for x in row) for row in rows]


REQUEST_FIFO = "gq_request.fifo"
REPLY_FIFO = "gq_reply.fifo"


def write_server_runqm(qm_dir: Path) -> Path:
    """``QM/runQM.sh`` for ``QmServer``: send one request line, wait for the reply line, exit 0
    only on ``ok``. Paths are absolute, so the script works from any directory."""
    qm_dir = Path(qm_dir).resolve()
    script = qm_dir / "runQM.sh"
    script.write_text(
        "#!/bin/sh\n"
        "# our own QM.out provider (swap test): the gqteashmd QmServer answers this request\n"
        f'echo request > "{qm_dir / REQUEST_FIFO}" || exit 1\n'
        f'read status < "{qm_dir / REPLY_FIFO}" || exit 1\n'
        '[ "$status" = ok ]\n'
    )
    script.chmod(0o755)
    return script


class QmServer:
    """Answers the requests of ``write_server_runqm`` scripts in ``qm_dir`` with our model.

    Each request: read ``qm_dir/QM.in`` (``parse_sharc_qm_in``), evaluate the model at its
    geometry with the snapshot of the previous step number as ``previous`` (so signs are
    continuous and the overlaps refer to the previous step), write ``qm_dir/QM.out``
    (``format_qm_out``) and reply ``ok``; any error is counted, written to
    ``qm_dir/server_errors.log`` and answered with ``error``. A context manager: the named pipes
    are created on entry, the thread is stopped and joined on exit.
    """

    REPLY_TIMEOUT_S = 30.0

    def __init__(self, model, qm_dir: Path) -> None:
        self.model = model
        self.qm_dir = Path(qm_dir).resolve()
        self.calls = 0
        self.errors = 0
        self._snapshots: dict[int, Any] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="QmServer", daemon=True)

    def __enter__(self) -> QmServer:
        for name in (REQUEST_FIFO, REPLY_FIFO):
            path = self.qm_dir / name
            path.unlink(missing_ok=True)
            os.mkfifo(path)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def close(self) -> None:
        """Stop and join the thread: it may be blocked opening the request pipe, so a writer
        of our own wakes it (repeated until it has ended, at most 10 s)."""
        self._stop.set()
        deadline = time.monotonic() + 10.0
        while self._thread.is_alive() and time.monotonic() < deadline:
            with contextlib.suppress(OSError):  # ENXIO while no reader has the pipe open
                fd = os.open(self.qm_dir / REQUEST_FIFO, os.O_WRONLY | os.O_NONBLOCK)
                try:
                    os.write(fd, b"quit\n")
                finally:
                    os.close(fd)
            self._thread.join(timeout=0.05)

    def _answer(self) -> None:
        request = parse_sharc_qm_in((self.qm_dir / "QM.in").read_text())
        previous = self._snapshots.get(request.step - 1)
        snapshot = self.model.evaluate(request.coords_bohr.reshape(-1), previous)
        text = format_qm_out(snapshot, self.model.states, request.requests)
        self._snapshots = {k: v for k, v in self._snapshots.items() if k == request.step - 1}
        self._snapshots[request.step] = snapshot
        tmp = self.qm_dir / "QM.out.tmp"
        tmp.write_text(text)
        os.replace(tmp, self.qm_dir / "QM.out")

    def _reply(self, status: str) -> None:
        deadline = time.monotonic() + self.REPLY_TIMEOUT_S
        while True:
            try:
                fd = os.open(self.qm_dir / REPLY_FIFO, os.O_WRONLY | os.O_NONBLOCK)
                break
            except OSError as exc:  # ENXIO: the script has not opened its end yet
                if exc.errno != errno.ENXIO or time.monotonic() > deadline:
                    return
                time.sleep(0.0005)
        try:
            os.write(fd, f"{status}\n".encode())
        finally:
            os.close(fd)

    def _serve(self) -> None:
        while not self._stop.is_set():
            with open(self.qm_dir / REQUEST_FIFO) as pipe:
                messages = pipe.read().split()
            if self._stop.is_set() or not messages or "quit" in messages:
                continue
            self.calls += 1
            try:
                self._answer()
                status = "ok"
            except Exception as exc:  # reported to sharc.x as a failed interface call
                self.errors += 1
                with open(self.qm_dir / "server_errors.log", "a") as log:
                    log.write(f"call {self.calls}: {type(exc).__name__}: {exc}\n")
                status = "error"
            self._reply(status)


# --- one sharc.x trajectory ----------------------------------------------------------------------


def write_lvc_trajectory(
    work_dir: Path,
    *,
    symbols: Sequence[str],
    masses_amu: Sequence[float],
    q0: np.ndarray,
    v0: np.ndarray,
    input_text: str,
    provider: str,
    template_text: str | None = None,
    v0_text: str | None = None,
    sharc_python: str | None = None,
) -> None:
    """Write ``input``, ``geom``, ``veloc`` and ``QM/`` in ``work_dir`` (which must not exist).

    ``"sharc_lvc"``: ``QM/runQM.sh`` runs SHARC_LVC.py (manual 3.1.1) on the template (give
    it without its DM blocks) and V0.txt; ``"ours"``: ``QM/runQM.sh`` talks to a ``QmServer``.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {', '.join(PROVIDERS)}; got {provider!r}")
    work_dir = Path(work_dir)
    qm = work_dir / "QM"
    qm.mkdir(parents=True, exist_ok=False)
    natom = len(symbols)
    (work_dir / "geom").write_text(
        format_geom(symbols, masses_amu, np.asarray(q0, dtype=float).reshape(natom, 3))
    )
    (work_dir / "veloc").write_text(format_veloc(np.asarray(v0, dtype=float).reshape(natom, 3)))
    (work_dir / "input").write_text(input_text)
    if provider == "ours":
        write_server_runqm(qm)
        return
    if template_text is None or v0_text is None or sharc_python is None:
        raise ValueError("the sharc_lvc provider needs template_text, v0_text and sharc_python")
    (qm / "LVC.template").write_text(template_text)
    (qm / "V0.txt").write_text(v0_text)
    script = qm / "runQM.sh"
    script.write_text(f'cd QM/ || exit 1\nexec "{sharc_python}" "$SHARC/SHARC_LVC.py" QM.in\n')
    script.chmod(0o755)


def run_lvc_trajectory(
    work_dir: Path,
    *,
    sharc_home: Path,
    provider: str,
    model=None,
    timeout_s: float = 7200.0,
    **files: Any,
) -> LvcTrajectory:
    """``write_lvc_trajectory(work_dir, provider=provider, **files)``, run ``sharc.x input``
    there (its own session, SHARC's environment with one thread per process) and parse
    output.dat. ``"ours"`` needs ``model`` and serves it for the duration of the run."""
    sharc_home = Path(sharc_home)
    work_dir = Path(work_dir)
    write_lvc_trajectory(work_dir, provider=provider, **files)
    env = {**sharc_environment(sharc_home), **SINGLE_THREAD}
    server = None
    if provider == "ours":
        if model is None:
            raise ValueError("provider 'ours' needs the model")
        server = QmServer(model, work_dir / "QM")
    with server if server is not None else contextlib.nullcontext():
        proc = run_in_session(
            ["bash", "-c", SHARC_X_COMMAND],
            cwd=work_dir,
            env=env,
            timeout=timeout_s,
            what="sharc.x",
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"sharc.x failed ({proc.returncode}) in {work_dir}:\n{proc.stderr[-2000:]}"
            f"\n{proc.stdout[-2000:]}"
        )
    return parse_lvc_output_dat(work_dir / "output.dat")


def run_model_trajectory(
    work_dir: Path,
    model,
    template_path: Path,
    q0: np.ndarray,
    v0: np.ndarray,
    options,
    n_steps: int,
    rngseed: int,
    *,
    sharc_home: Path,
    sharc_python: str | None = None,
    provider: str = "sharc_lvc",
    state_mch: int = 1,
    detailed: bool = True,
    output_stride: int | None = None,
    timeout_s: float = 7200.0,
) -> LvcTrajectory:
    """One sharc.x trajectory of our ``LVCModel`` read from ``template_path`` (which names its
    V0.txt), with the settings of our ``DynamicsOptions`` (``LvcSharcOptions``), started in
    the MCH state ``state_mch`` (0-based) with ``coeff auto``. ``"sharc_lvc"`` gives SHARC the
    template without its DM blocks; ``"ours"`` serves ``model`` itself (the swap test)."""
    template_path = Path(template_path)
    template = template_path.read_text()
    v0_path = template_path.parent / template.splitlines()[0].strip()
    text = lvc_input(
        n_steps,
        options.dt_au * AU_TIME_IN_FS,
        LvcSharcOptions.from_dynamics_options(options),
        rngseed,
        states=model.state_counts,
        state_mch=state_mch,
        detailed=detailed,
        output_stride=output_stride,
    )
    v0_data = model.parameters.v0
    return run_lvc_trajectory(
        work_dir,
        sharc_home=sharc_home,
        provider=provider,
        model=model,
        timeout_s=timeout_s,
        symbols=v0_data.symbols,
        masses_amu=v0_data.masses_amu,
        q0=q0,
        v0=v0,
        input_text=text,
        template_text=without_dipoles("V0.txt\n" + template.split("\n", 1)[1]),
        v0_text=v0_path.read_text(),
        sharc_python=sharc_python,
    )


_LIS_ROW = re.compile(r"^\s*\d+\s+[-+0-9.Ee]+\s+\d+\s+\d+\s")


def parse_output_lis(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(steps, E_tot in eV) of every step line of output.lis (manual 5.2: step, time, diag
    state, MCH state, then E_kin, E_pot, E_tot in eV)."""
    steps, totals = [], []
    for line in Path(path).read_text().splitlines():
        if _LIS_ROW.match(line):
            tokens = line.split()
            steps.append(int(tokens[0]))
            totals.append(float(tokens[6]))
    return np.array(steps, dtype=int), np.array(totals)


# --- CLI: one single point ------------------------------------------------------------------------


def _read_xyz_bohr(path: Path) -> tuple[list[str], np.ndarray]:
    rows = path.read_text().splitlines()
    n = int(rows[0])
    symbols = [r.split()[0] for r in rows[2 : 2 + n]]
    coords = np.array([[float(x) for x in r.split()[1:4]] for r in rows[2 : 2 + n]])
    return symbols, coords


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sharc-home", required=True)
    p.add_argument("--python", required=True, help="Python of the SHARC conda environment")
    p.add_argument("--template", required=True, help="LVC.template naming V0.txt next to it")
    p.add_argument("--geometry", required=True, help="xyz file in bohr")
    p.add_argument("--states", required=True, help='e.g. "2 0 2"')
    p.add_argument("--soc", action="store_true")
    p.add_argument("--work-dir", required=True)
    args = p.parse_args()
    template = Path(args.template)
    v0 = template.parent / template.read_text().splitlines()[0].strip()
    symbols, coords = _read_xyz_bohr(Path(args.geometry))
    work = Path(args.work_dir)
    qm_in = format_qm_in(
        symbols,
        coords,
        states=[int(n) for n in args.states.split()],
        savedir=str(work.resolve() / "SAVE"),
        step=0,
        soc=args.soc,
    )
    text = template.read_text().splitlines(keepends=True)
    text[0] = "V0.txt\n"
    data = run_single_point(
        work,
        qm_in,
        without_dipoles("".join(text)),
        v0.read_text(),
        sharc_home=Path(args.sharc_home),
        sharc_python=args.python,
    )
    print(json.dumps(data, default=lambda z: [z.real, z.imag], indent=1))


if __name__ == "__main__":
    main()
