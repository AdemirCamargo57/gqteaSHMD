"""Run SHARC 4.1's excite.py through stdin as a black box and return ``initconds.excited``.

SHARC is only *executed*; nothing from SHARC is imported or copied. The working directory must
hold ``initconds`` and ``ICOND_00000`` ... ``ICOND_<N>`` (each with ``QM.in`` and ``QM.out``).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

TIMEOUT_S = 600


def run_excite(
    sharc_home: Path, python: str, work_dir: Path, *, window_eV: tuple[float, float], seed: int
) -> Path:
    """Answer the ten prompts of excite.py (delta-pulse excitation) and return its output file."""
    answers = [
        "y",  # use file "initconds"
        "2",  # excited-state lists from QM.out in the ICOND directories
        ".",  # path to the ICOND directories
        "False",  # MCH representation
        "3",  # flag by excitation energies and oscillator strengths
        f"{window_eV[0]} {window_eV[1]}",  # range (eV)
        "1",  # lower state
        "",  # include all states
        str(seed),  # RNG seed
        "",  # continue
    ]
    env = dict(os.environ)
    env["SHARC"] = str(sharc_home / "bin")
    # PYTHONPATH is replaced, not extended: SHARC's own conda environment must not see ours.
    env["PYTHONPATH"] = f"{sharc_home / 'lib'}:{sharc_home / 'pysharc'}"
    try:
        proc = subprocess.run(
            [python, str(sharc_home / "bin" / "excite.py")],
            input="\n".join(answers) + "\n" + "\n" * 10,
            cwd=work_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            f"excite.py did not finish within {TIMEOUT_S} s in {work_dir} "
            "(a prompt may have changed: check the answers in run_excite.py)"
        ) from exc
    out = work_dir / "initconds.excited"
    if proc.returncode != 0 or not out.is_file():
        tail = (proc.stdout[-1500:] + "\n" + proc.stderr[-1500:]).strip()
        raise SystemExit(f"excite.py failed ({proc.returncode}):\n{tail}")
    return out
