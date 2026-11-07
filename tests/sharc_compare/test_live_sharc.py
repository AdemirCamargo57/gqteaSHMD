import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "tests" / "data" / "reference" / "sharc41_pyscf_ethylene_sa2cas22_631gs.json"


@pytest.mark.sharc
@pytest.mark.skipif(
    not (os.environ.get("SHARC_HOME") and os.environ.get("SHARC_PYTHON")),
    reason="needs SHARC_HOME and SHARC_PYTHON",
)
def test_fresh_sharc_run_reproduces_the_committed_reference(tmp_path):
    out = tmp_path / "ref.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "sharc_compare" / "run_sharc_pyscf.py"),
            "--sharc-home",
            os.environ["SHARC_HOME"],
            "--python",
            os.environ["SHARC_PYTHON"],
            "--xyz",
            str(ROOT / "tests" / "data" / "ethylene.xyz"),
            "--basis",
            "6-31g*",
            "--ncas",
            "2",
            "--nelecas",
            "2",
            "--roots",
            "2",
            "--out",
            str(out),
        ],
        check=True,
        timeout=1800,
    )
    fresh, ref = json.loads(out.read_text()), json.loads(REFERENCE.read_text())
    assert np.allclose(fresh["energies_hartree"], ref["energies_hartree"], atol=1e-8, rtol=0)
    for name in ref["gradients_hartree_per_bohr"]:
        assert np.allclose(
            fresh["gradients_hartree_per_bohr"][name],
            ref["gradients_hartree_per_bohr"][name],
            atol=1e-7,
            rtol=0,
        )
    mu = np.array(fresh["transition_dipoles_au"]["S1"])
    mu_ref = np.array(ref["transition_dipoles_au"]["S1"])
    assert min(np.abs(mu - mu_ref).max(), np.abs(mu + mu_ref).max()) < 1e-7
    for name, rows in ref["nacs_per_bohr"].items():
        fresh_rows = np.array(fresh["nacs_per_bohr"][name])
        rows = np.array(rows)
        assert min(np.abs(fresh_rows - rows).max(), np.abs(fresh_rows + rows).max()) < 1e-7
