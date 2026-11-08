import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gqteashmd.vib.harmonic import harmonic_analysis
from gqteashmd.vib.molden import format_molden

RUNNER = Path(__file__).resolve().parents[2] / "tools" / "sharc_compare" / "run_wigner.py"

pytestmark = [
    pytest.mark.sharc,
    pytest.mark.skipif(
        not (os.environ.get("SHARC_HOME") and os.environ.get("SHARC_PYTHON")),
        reason="needs SHARC_HOME and SHARC_PYTHON",
    ),
]


def test_sharc_wigner_accepts_our_molden(tmp_path):
    masses = np.array([15.994915, 1.007825, 1.007825])
    coords = np.array([[0.0, 0.0, 0.2043], [0.0, 1.4266, -0.8780], [0.0, -1.4266, -0.8780]])
    rng = np.random.default_rng(3)
    a = rng.normal(size=(9, 9))
    analysis = harmonic_analysis(masses, coords, a @ a.T * 0.1)
    molden = tmp_path / "freq.molden"
    molden.write_text(
        format_molden(("O", "H", "H"), coords, analysis.frequencies_cm1, analysis.modes_cartesian)
    )
    out_dir = tmp_path / "wigner"
    proc = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--sharc-home",
            os.environ["SHARC_HOME"],
            "--python",
            os.environ["SHARC_PYTHON"],
            "--molden",
            str(molden),
            "--n",
            "3",
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (proc.stderr or proc.stdout)[-2000:]
    assert proc.stdout.strip()  # wigner.py wrote its initial conditions (paths listed)
    assert any(out_dir.iterdir())
