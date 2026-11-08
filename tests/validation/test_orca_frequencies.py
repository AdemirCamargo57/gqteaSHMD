import os
import re
import subprocess

import numpy as np
import pytest

from gqteashmd.config.schema import ElectronicSection
from gqteashmd.core.units import ANGSTROM_TO_BOHR
from gqteashmd.electronic.base import MoleculeSpec
from gqteashmd.electronic.pyscf_backend import PySCFBackend
from gqteashmd.vib.harmonic import harmonic_analysis

pytestmark = [
    pytest.mark.orca,
    pytest.mark.pyscf,
    pytest.mark.skipif(not os.environ.get("ORCA_EXE"), reason="needs ORCA_EXE"),
]
GEOMETRY = [
    ("O", (0.0, 0.0, 0.108140699)),
    ("H", (0.0, 0.754903478, -0.464620350)),
    ("H", (0.0, -0.754903478, -0.464620350)),
]  # RHF/6-31G* minimum (Angstrom)


def _run_orca(tmp_path):
    lines = "\n".join(f"{s} {x:.9f} {y:.9f} {z:.9f}" for s, (x, y, z) in GEOMETRY)
    (tmp_path / "w.inp").write_text(f"! RHF 6-31G* Freq TightSCF\n* xyz 0 1\n{lines}\n*\n")
    out = subprocess.run(
        [os.environ["ORCA_EXE"], "w.inp"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    ).stdout
    block = out.split("VIBRATIONAL FREQUENCIES", 1)[1]
    freqs = [float(m) for m in re.findall(r"^\s*\d+:\s+(-?\d+\.\d+) cm\*\*-1", block, re.M)]
    table = out.split("NO LB      ZA    FRAG     MASS", 1)[1].splitlines()[1:]
    masses = [float(row.split()[4]) for row in table[: len(GEOMETRY)]]
    return np.array(freqs), np.array(masses)


def test_frequencies_match_orca(tmp_path):
    orca_freqs, orca_masses = _run_orca(tmp_path)
    settings = ElectronicSection.model_validate(
        {
            "method": "tda",
            "basis": "6-31g*",
            "functional": "hf",
            "singlets": 1,
            "scf_conv_tol": 1e-11,
        }
    )
    coords = np.array([c for _, c in GEOMETRY]) * ANGSTROM_TO_BOHR
    backend = PySCFBackend(MoleculeSpec(tuple(s for s, _ in GEOMETRY)), settings)
    ours = harmonic_analysis(orca_masses, coords, backend.ground_state_hessian(coords))
    theirs = orca_freqs[orca_freqs != 0.0]
    print("ORCA", theirs, "ours", ours.frequencies_cm1, "diff", ours.frequencies_cm1 - theirs)
    np.testing.assert_allclose(ours.frequencies_cm1, theirs, atol=0.2)
