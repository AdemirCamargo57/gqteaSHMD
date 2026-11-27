"""Triplet energies against ORCA 6.1.1, run as a black box (spec 4 item 1; bound 1e-6 Eh)."""

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gqteashmd.chem.xyz import parse_xyz
from gqteashmd.config.schema import ElectronicSection
from gqteashmd.core.states import StateLabel
from gqteashmd.core.units import ANGSTROM_TO_BOHR, HARTREE_IN_CM1
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.pyscf_backend import PySCFBackend

pytestmark = [
    pytest.mark.orca,
    pytest.mark.pyscf,
    pytest.mark.skipif(not os.environ.get("ORCA_EXE"), reason="needs ORCA_EXE"),
]
TOLERANCE = 1e-6  # Eh, per state (spec 4)
S0, S1, T1 = StateLabel(1, 0), StateLabel(1, 1), StateLabel(3, 1)
WATER = [
    ("O", (0.0, 0.0, 0.1173)),
    ("H", (0.0, 0.7572, -0.4692)),
    ("H", (0.0, -0.7572, -0.4692)),
]  # tests/conftest.py:WATER_XYZ (Angstrom)
ETHYLENE = Path(__file__).resolve().parents[1] / "data" / "ethylene.xyz"


def _run_orca(tmp_path, name, keywords, blocks, atoms):
    lines = "\n".join(f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, (x, y, z) in atoms)
    (tmp_path / f"{name}.inp").write_text(f"{keywords}\n{blocks}\n* xyz 0 1\n{lines}\n*\n")
    return subprocess.run(
        [os.environ["ORCA_EXE"], f"{name}.inp"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    ).stdout


def _scf_energy(out):
    return float(re.search(r"^Total Energy\s+:\s+(-?\d+\.\d+) Eh", out, re.M).group(1))


def _cis_triplets(out):
    """Triplet excitation energies (Eh) from the `CIS EXCITED STATES (TRIPLETS)` block.

    ORCA prints `STATE  4:  E=   0.407460 au     11.088 eV    89427.2 cm**-1 ... Mult 3`; the au
    column has 6 decimals (5e-7 Eh), so the cm**-1 column (0.05 cm-1 = 2.3e-7 Eh) is parsed and
    divided by HARTREE_IN_CM1.
    """
    block = out.split("CIS EXCITED STATES (TRIPLETS)", 1)[1]
    pattern = r"^STATE\s+\d+:\s+E=\s+\S+ au\s+\S+ eV\s+(\d+\.\d+) cm\*\*-1.*Mult 3"
    return np.array([float(m) / HARTREE_IN_CM1 for m in re.findall(pattern, block, re.M)])


def _casscf_states(out):
    """{multiplicity: [state energies (Eh)]} and the printed root weights, from ORCA's CASSCF.

    The final energies are the `ROOT   i:  E=     -77.8943984731 Eh` lines (10 decimals) under
    `CAS-SCF STATES FOR BLOCK  k MULT= m` after `CASSCF RESULTS` (an earlier copy of these blocks
    is printed inside the iterations). The weights are the `ROOT=i WEIGHT=    0.333333` lines.
    """
    weights = [float(w) for w in re.findall(r"^\s+ROOT=\d+ WEIGHT=\s+(\d+\.\d+)", out, re.M)]
    final = out.split("CASSCF RESULTS", 1)[1]
    energies = {}
    for part in re.split(r"CAS-SCF STATES FOR BLOCK\s+\d+ MULT=", final)[1:]:
        mult = int(part.split()[0])
        body = part.split("\n\n\n", 1)[0]
        energies[mult] = [
            float(e) for e in re.findall(r"^ROOT\s+\d+:\s+E=\s+(-?\d+\.\d+) Eh", body, re.M)
        ]
    return energies, weights


def test_water_cis_triplets_match_orca(tmp_path):
    out = _run_orca(
        tmp_path,
        "water",
        # NoFrozenCore: ORCA freezes O 1s in CIS by default; PySCF's TDA is all-electron
        "! RHF STO-3G NoRI NoFrozenCore TightSCF",
        "%tddft nroots 3 tda true triplets true end",
        WATER,
    )
    orca_scf, orca_triplets = _scf_energy(out), _cis_triplets(out)
    settings = ElectronicSection.model_validate(
        {
            "method": "tda",
            "basis": "sto-3g",
            "functional": "hf",
            "singlets": 1,
            "triplets": 3,
            "scf_conv_tol": 1e-11,
            "td_conv_tol": 1e-9,
        }
    )
    coords = np.array([c for _, c in WATER]) * ANGSTROM_TO_BOHR
    result = PySCFBackend(MoleculeSpec(tuple(s for s, _ in WATER)), settings).compute(
        Request(coords)
    )
    ours_scf = result.energies[S0]
    ours = np.array([result.energies[StateLabel(3, k)] - ours_scf for k in (1, 2, 3)])
    print(f"SCF: ORCA {orca_scf:.10f}  ours {ours_scf:.10f}  diff {ours_scf - orca_scf:+.2e}")
    print("triplet excitation energies: ORCA", orca_triplets, "ours", ours)
    print("diff", ours - orca_triplets)
    assert orca_triplets.shape == (3,)
    assert abs(ours_scf - orca_scf) <= TOLERANCE
    np.testing.assert_allclose(ours, orca_triplets, rtol=0, atol=TOLERANCE)


def test_ethylene_sa_casscf_with_a_triplet_matches_orca(tmp_path):
    geometry = parse_xyz(ETHYLENE.read_text())
    atoms = list(zip(geometry.symbols, geometry.coords_angstrom, strict=True))
    out = _run_orca(
        tmp_path,
        "ethylene",
        "! 6-31G* NoRI TightSCF",
        # bweight/weights: 1/3 per state; etol/gtol: ORCA's default orbital gradient (2.5e-4)
        # leaves per-state energies off by 2e-6 Eh, so converge as tightly as ours
        "%casscf\n  nel 2\n  norb 2\n  mult 1,3\n  nroots 2,1\n  bweight 2,1\n"
        "  weights[0] = 0.5,0.5\n  weights[1] = 1.0\n  etol 1e-10\n  gtol 1e-6\nend",
        atoms,
    )
    energies, weights = _casscf_states(out)
    assert weights == pytest.approx([1 / 3] * 3, abs=1e-6)
    assert sorted(energies) == [1, 3] and len(energies[1]) == 2 and len(energies[3]) == 1
    orca = {S0: energies[1][0], S1: energies[1][1], T1: energies[3][0]}
    settings = ElectronicSection.model_validate(
        {
            "method": "casscf",
            "basis": "6-31g*",
            "singlets": 2,
            "triplets": 1,
            "casscf": {"ncas": 2, "nelecas": 2, "conv_tol": 1e-10, "conv_tol_grad": 1e-6},
        }
    )
    coords = np.array(geometry.coords_angstrom) * ANGSTROM_TO_BOHR
    backend = PySCFBackend(MoleculeSpec(geometry.symbols), settings, reference_coords_bohr=coords)
    result = backend.compute(Request(coords))
    for label, theirs in orca.items():
        ours = result.energies[label]
        print(f"{label}: ORCA {theirs:.10f}  ours {ours:.10f}  diff {ours - theirs:+.2e}")
    for label, theirs in orca.items():
        assert abs(result.energies[label] - theirs) <= TOLERANCE, label
