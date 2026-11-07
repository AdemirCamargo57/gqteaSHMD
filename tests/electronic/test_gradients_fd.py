import pytest

from gqteashmd.core.states import StateLabel
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.pyscf_backend import PySCFBackend

S0, S1 = StateLabel(1, 0), StateLabel(1, 1)
H = 1e-4  # bohr


def _fd(symbols, settings, coords, label, atom, axis):
    def energy(c):
        return (
            PySCFBackend(MoleculeSpec(tuple(symbols)), settings).compute(Request(c)).energies[label]
        )

    plus, minus = coords.copy(), coords.copy()
    plus[atom, axis] += H
    minus[atom, axis] -= H
    return (energy(plus) - energy(minus)) / (2 * H)


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
