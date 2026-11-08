"""Vertical excitation energies, transition dipoles and oscillator strengths against SHARC 4.1.

SHARC_PYSCF.py is executed as a black box on water CAS(4,4)/STO-3G (3 singlets) at the reference
geometry and two Wigner-displaced geometries; PySCFBackend must agree.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from gqteashmd.chem.masses import isotope_mass
from gqteashmd.config.schema import ElectronicSection
from gqteashmd.core.states import StateLabel
from gqteashmd.core.units import ANGSTROM_TO_BOHR, BOHR_IN_ANGSTROM
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.properties import oscillator_strength
from gqteashmd.electronic.pyscf_backend import PySCFBackend
from gqteashmd.vib.harmonic import harmonic_analysis
from gqteashmd.vib.wigner import sample_wigner

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "sharc_compare"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pytestmark = [
    pytest.mark.sharc,
    pytest.mark.pyscf,
    pytest.mark.skipif(
        not (os.environ.get("SHARC_HOME") and os.environ.get("SHARC_PYTHON")),
        reason="needs SHARC_HOME and SHARC_PYTHON",
    ),
]

SYMBOLS = ("O", "H", "H")
REFERENCE_ANGSTROM = np.array(
    [
        [0.0, 0.0, 0.1501683345],
        [0.0, 0.7580805047, -0.4856341673],
        [0.0, -0.7580805047, -0.4856341673],
    ]
)
TEMPLATE = (
    "ncas 4\nnelecas 4\nroots 3\nbasis sto-3g\nmethod casscf\nconv-tol 1e-10\nconv-tol-grad 1e-6\n"
)


def _geometries():
    """Reference geometry plus two Wigner samples from our analytic RHF/STO-3G frequencies."""
    scf = ElectronicSection.model_validate(
        {
            "method": "tda",
            "basis": "sto-3g",
            "functional": "hf",
            "singlets": 1,
            "scf_conv_tol": 1e-11,
        }
    )
    coords = REFERENCE_ANGSTROM * ANGSTROM_TO_BOHR
    masses = np.array([isotope_mass(s) for s in SYMBOLS])
    analysis = harmonic_analysis(
        masses, coords, PySCFBackend(MoleculeSpec(SYMBOLS), scf).ground_state_hessian(coords)
    )
    samples = sample_wigner(
        masses,
        coords,
        analysis.frequencies_cm1,
        analysis.modes_mass_weighted,
        n_samples=2,
        temperature_K=0.0,
        low_frequency_cutoff_cm1=10.0,
        rng=np.random.default_rng(5),
    )
    return [coords, *samples.positions_bohr]


def test_excitations_match_sharc_pyscf():
    run_sharc = _load("run_sharc_pyscf")
    settings = ElectronicSection.model_validate(
        {
            "method": "casscf",
            "basis": "sto-3g",
            "singlets": 3,
            "scf_conv_tol": 1e-11,
            "casscf": {"ncas": 4, "nelecas": 4, "conv_tol": 1e-10, "conv_tol_grad": 1e-6},
        }
    )
    worst_de = worst_mu = worst_f = 0.0
    for k, coords in enumerate(_geometries()):
        sharc = run_sharc.run_geometry(
            Path(os.environ["SHARC_HOME"]),
            os.environ["SHARC_PYTHON"],
            SYMBOLS,
            coords * BOHR_IN_ANGSTROM,
            TEMPLATE,
            3,
        )
        e = [sharc["hamiltonian"][i][i].real for i in range(3)]
        # A fresh backend per geometry: no orbital reuse between the three calculations.
        backend = PySCFBackend(MoleculeSpec(SYMBOLS), settings)
        result = backend.compute(Request(coords, transition_dipoles=True))
        for i in (1, 2):
            label = StateLabel(1, i)
            de_sharc = e[i] - e[0]
            mu_sharc = np.array([sharc["dipole"][x][0][i] for x in range(3)])
            f_sharc = oscillator_strength(de_sharc, mu_sharc)
            de_ours = result.excitation_energy(label)
            mu_ours = result.transition_dipoles[label]
            f_ours = oscillator_strength(de_ours, mu_ours)
            d_de = abs(de_ours - de_sharc)
            d_mu = abs(np.linalg.norm(mu_ours) - np.linalg.norm(mu_sharc))
            d_f = abs(f_ours - f_sharc)
            print(
                f"geometry {k} S{i}: dE SHARC {de_sharc:.8f} ours {de_ours:.8f}"
                f" (diff {d_de:.2e} Eh)"
                f" | |mu| SHARC {np.linalg.norm(mu_sharc):.6f} ours {np.linalg.norm(mu_ours):.6f}"
                f" (diff {d_mu:.2e}) | f SHARC {f_sharc:.6f} ours {f_ours:.6f} (diff {d_f:.2e})"
            )
            worst_de, worst_mu, worst_f = (
                max(worst_de, d_de),
                max(worst_mu, d_mu),
                max(worst_f, d_f),
            )
            assert d_de < 1e-5
            assert d_mu <= 1e-4 * np.linalg.norm(mu_sharc) + 1e-7
            assert d_f <= 1e-4 * f_sharc + 1e-7
            # the transition dipole vectors agree up to an overall sign
            sign = np.sign(np.dot(mu_ours, mu_sharc)) or 1.0
            np.testing.assert_allclose(
                sign * mu_ours, mu_sharc, atol=1e-4 * np.linalg.norm(mu_sharc) + 1e-7
            )
    print(f"worst: dE {worst_de:.2e} Eh, |mu| {worst_mu:.2e}, f {worst_f:.2e}")
