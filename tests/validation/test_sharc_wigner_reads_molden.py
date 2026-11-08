import os
import subprocess

import numpy as np
import pytest

from gqteashmd.vib.harmonic import harmonic_analysis
from gqteashmd.vib.molden import format_molden

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
    home = os.environ["SHARC_HOME"]
    env = dict(os.environ, SHARC=f"{home}/bin", PYTHONPATH=f"{home}/lib:{home}/pysharc")
    before = set(os.listdir(tmp_path))
    proc = subprocess.run(
        [os.environ["SHARC_PYTHON"], f"{home}/bin/wigner.py", "-n", "3", str(molden)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert set(os.listdir(tmp_path)) - before  # wigner.py wrote its initial conditions
