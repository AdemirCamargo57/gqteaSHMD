"""Run SHARC 4.1's LVC interface (SHARC_LVC.py) as a black box at single points; parse QM.out.

SHARC is only *executed*; nothing from SHARC is imported or copied. Formats come from the
public SHARC 4.1 manual: running an interface (``$SHARC/SHARC_<NAME>.py QM.in`` in the directory
with its input files), the LVC files V0.txt and LVC.template (6.4.1), QM.in (6.34.1) and QM.out
(6.34.2, parsed by our ``qmout.py``). Every writer below is our own code.

Measured behaviour of this installation (the Task 9 report of P7a-2 has the details):

- QM.in coordinates need a decimal point (SHARC rejected ``C 0 0.05 ...`` with "line is not
  xyz"), so ``format_qm_in`` writes ``%.16e``.
- ``init`` is reported as deprecated ("Calculation control via STEP file and 'step' keyword"),
  so the first call of a sequence uses ``step 0`` and the next ``step 1`` with the same
  ``savedir`` (needed for ``OVERLAP``).
- With ``SOC`` requested and DMX/DMY/DMZ blocks in the template, SHARC_LVC.py stops with
  "Cannot cast ufunc 'matmul' input 1 from dtype('float64') to dtype('complex128') with
  casting rule 'no'" even when ``DM`` is not requested; ``without_dipoles`` removes those
  blocks (the dipoles are not used by the dynamics and are not compared).

Example (WSL), one point of a template from tests/data/lvc:
  python tools/sharc_compare/run_sharc_lvc.py --sharc-home /home/ajc/sharc4-4.1 \\
      --python /home/ajc/miniforge3/envs/sharc4.1/bin/python \\
      --template tests/data/lvc/lvc_soc.template --states "2 0 2" --soc \\
      --geometry geom.xyz --work-dir /tmp/sharc-lvc
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qmout import parse_qm_out  # noqa: E402
from run_sharc_model import _environment  # noqa: E402

LVC_BLOCKS = ("hamiltonian", "gradients")  # always present; dipoles only with DM
_DM_HEADER = re.compile(r"^DM[XYZ]\s+[RI]$")


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
    session; on a timeout its whole process group is killed and ``RuntimeError`` is raised, as
    for a non-zero exit or a missing QM.out.
    """
    sharc_home = Path(sharc_home)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    for name, text in (("QM.in", qm_in_text), ("LVC.template", template_text), ("V0.txt", v0_text)):
        (work_dir / name).write_bytes(text.encode())
    qm_out = work_dir / "QM.out"
    qm_out.unlink(missing_ok=True)
    proc = subprocess.Popen(
        [sharc_python, str(sharc_home / "bin" / "SHARC_LVC.py"), "QM.in"],
        cwd=work_dir,
        env=_environment(sharc_home),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise RuntimeError(
            f"SHARC_LVC.py did not finish within {timeout} s in {work_dir}"
        ) from None
    finally:
        if proc.poll() is None:  # an exception other than the timeout (e.g. KeyboardInterrupt)
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"SHARC_LVC.py failed ({proc.returncode}) in {work_dir}:\n{stderr[-3000:]}"
            f"\n{stdout[-2000:]}"
        )
    if not qm_out.is_file():
        raise RuntimeError(f"SHARC_LVC.py wrote no QM.out in {work_dir}:\n{stdout[-2000:]}")
    return parse_qm_out(qm_out.read_text(), required=required)


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
