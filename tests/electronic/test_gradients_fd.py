import numpy as np
import pytest

from gqteashmd.core.states import StateLabel
from gqteashmd.core.units import ANGSTROM_TO_BOHR
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.pyscf_backend import PySCFBackend

pytestmark = pytest.mark.pyscf

S0, S1 = StateLabel(1, 0), StateLabel(1, 1)
T1, T2 = StateLabel(3, 1), StateLabel(3, 2)
H = 1e-4  # bohr
H_TRIPLET = 1e-3  # bohr (spec 4 item 2)
TRIPLET_TOLERANCE = 1e-5  # Eh/bohr, max over each test's components (PR3)
WATER_SYMBOLS = ("O", "H", "H")
WATER_BOHR = (
    np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
    * ANGSTROM_TO_BOHR
)  # tests/conftest.py:WATER_XYZ
WATER_COMPONENTS = ((0, 2), (1, 1), (2, 2))  # (atom, axis)


def _fd(symbols, settings, coords, label, atom, axis, step=H):
    def energy(c):
        return (
            PySCFBackend(MoleculeSpec(tuple(symbols)), settings).compute(Request(c)).energies[label]
        )

    plus, minus = coords.copy(), coords.copy()
    plus[atom, axis] += step
    minus[atom, axis] -= step
    return (energy(plus) - energy(minus)) / (2 * step)


@pytest.mark.slow
def test_sa_casscf_gradients_match_finite_differences(
    ethylene_symbols, ethylene_bohr, make_settings
):
    settings = make_settings(
        basis="6-31g",
        scf_conv_tol=1e-12,
        casscf={"ncas": 2, "nelecas": 2, "conv_tol": 1e-10, "conv_tol_grad": 1e-6},
    )
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings)
    result = backend.compute(Request(ethylene_bohr, gradients=frozenset({S0, S1})))
    for atom, axis in ((0, 0), (2, 1), (4, 2)):
        for label in (S0, S1):
            fd = _fd(ethylene_symbols, settings, ethylene_bohr, label, atom, axis)
            assert abs(fd - result.gradients[label][atom, axis]) < 1e-6, (label, atom, axis)


@pytest.mark.slow
def test_tda_gradients_match_finite_differences(ethylene_symbols, ethylene_bohr, make_settings):
    settings = make_settings(
        method="tda",
        basis="sto-3g",
        functional="b3lyp",
        singlets=2,
        casscf=None,
        scf_conv_tol=1e-12,
        td_conv_tol=1e-9,
        grid_level=5,
    )
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings)
    result = backend.compute(Request(ethylene_bohr, gradients=frozenset({S1})))
    for atom, axis in ((0, 0), (2, 1)):
        fd = _fd(ethylene_symbols, settings, ethylene_bohr, S1, atom, axis)
        assert abs(fd - result.gradients[S1][atom, axis]) < 5e-6, (atom, axis)


def _worst_triplet_difference(symbols, settings, coords, labels, components):
    """max |analytic - FD| over the labels and components, with a fresh backend per point."""
    backend = PySCFBackend(MoleculeSpec(tuple(symbols)), settings)
    result = backend.compute(Request(coords, gradients=frozenset(labels)))
    worst = 0.0
    for label in labels:
        for atom, axis in components:
            fd = _fd(symbols, settings, coords, label, atom, axis, step=H_TRIPLET)
            analytic = result.gradients[label][atom, axis]
            print(f"{label} atom {atom} axis {axis}: FD {fd:+.8e} analytic {analytic:+.8e}")
            worst = max(worst, abs(fd - analytic))
    print(f"max |analytic - FD| = {worst:.2e} Eh/bohr")
    return worst


@pytest.mark.slow
def test_tda_hf_triplet_gradients_match_finite_differences(make_settings):
    settings = make_settings(
        method="tda",
        basis="sto-3g",
        functional="hf",
        singlets=1,
        triplets=2,
        casscf=None,
        scf_conv_tol=1e-12,
        td_conv_tol=1e-9,
    )
    worst = _worst_triplet_difference(
        WATER_SYMBOLS, settings, WATER_BOHR, (T1, T2), WATER_COMPONENTS
    )
    assert worst <= TRIPLET_TOLERANCE


@pytest.mark.slow
def test_tda_b3lyp_triplet_gradient_matches_finite_differences(make_settings):
    settings = make_settings(
        method="tda",
        basis="sto-3g",
        functional="b3lyp",
        singlets=1,
        triplets=1,
        casscf=None,
        scf_conv_tol=1e-12,
        td_conv_tol=1e-9,
        grid_level=5,
    )
    worst = _worst_triplet_difference(WATER_SYMBOLS, settings, WATER_BOHR, (T1,), WATER_COMPONENTS)
    assert worst <= TRIPLET_TOLERANCE


@pytest.mark.slow
def test_mixed_sa_casscf_s1_and_t1_gradients_match_finite_differences(
    ethylene_symbols, ethylene_bohr, make_settings
):
    settings = make_settings(
        basis="6-31g*",
        singlets=2,
        triplets=1,
        scf_conv_tol=1e-12,
        casscf={"ncas": 2, "nelecas": 2, "conv_tol": 1e-10, "conv_tol_grad": 1e-6},
    )
    worst = _worst_triplet_difference(
        ethylene_symbols, settings, ethylene_bohr, (S1, T1), ((0, 0), (2, 1))
    )
    assert worst <= TRIPLET_TOLERANCE
