import numpy as np
import pytest
from pyscf import dft, gto, scf, tddft

from gqteashmd.core.states import StateLabel
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.properties import oscillator_strength
from gqteashmd.electronic.pyscf_backend import PySCFBackend
from gqteashmd.errors import ElectronicStructureError

S0, S1, S2 = StateLabel(1, 0), StateLabel(1, 1), StateLabel(1, 2)


def _tda_settings(make_settings, **kw):
    fields = {
        "method": "tda",
        "basis": "sto-3g",
        "functional": "b3lyp",
        "singlets": 3,
        "casscf": None,
    }
    fields.update(kw)
    return make_settings(**fields)


def _direct(symbols, coords, functional, method="tda"):
    mol = gto.M(
        atom=[(s, tuple(c)) for s, c in zip(symbols, coords, strict=True)],
        unit="Bohr",
        basis="sto-3g",
        verbose=0,
    )
    if functional == "hf":
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol)
        mf.xc = functional
        mf.grids.level = 3
    mf.conv_tol = 1e-10
    mf.kernel()
    td = tddft.TDA(mf) if method == "tda" else tddft.TDDFT(mf)
    td.nstates = 2
    td.conv_tol = 1e-8
    td.kernel()
    return mf, td


def test_tda_matches_direct_pyscf(ethylene_symbols, ethylene_bohr, make_settings):
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), _tda_settings(make_settings))
    result = backend.compute(
        Request(ethylene_bohr, gradients=frozenset({S0, S1}), transition_dipoles=True)
    )
    mf, td = _direct(ethylene_symbols, ethylene_bohr, "b3lyp")
    assert abs(result.energies[S0] - mf.e_tot) < 1e-8
    for label, omega in zip((S1, S2), td.e, strict=True):
        assert abs(result.excitation_energy(label) - omega) < 1e-6
    for label, f_ref in zip((S1, S2), td.oscillator_strength(), strict=True):
        f = oscillator_strength(result.excitation_energy(label), result.transition_dipoles[label])
        assert f == pytest.approx(f_ref, rel=1e-4, abs=1e-8)
    assert np.abs(result.gradients[S0] - mf.nuc_grad_method().kernel()).max() < 1e-6
    assert np.abs(result.gradients[S1] - td.nuc_grad_method().kernel(state=1)).max() < 1e-5
    assert result.spin_squares == {}


def test_cis_and_tdhf_with_hf(ethylene_symbols, ethylene_bohr, make_settings):
    for method in ("tda", "tddft"):
        settings = _tda_settings(make_settings, method=method, functional="hf")
        result = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings).compute(
            Request(ethylene_bohr, gradients=frozenset({S1}))
        )
        mf, td = _direct(ethylene_symbols, ethylene_bohr, "hf", method)
        assert abs(result.energies[S0] - mf.e_tot) < 1e-8
        assert abs(result.excitation_energy(S1) - td.e[0]) < 1e-6
        assert result.gradients[S1].shape == (6, 3)


def test_ground_state_only(ethylene_symbols, ethylene_bohr, make_settings):
    settings = _tda_settings(make_settings, singlets=1)
    result = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings).compute(
        Request(ethylene_bohr, gradients=frozenset({S0}), transition_dipoles=True)
    )
    assert result.states == (S0,)
    assert result.transition_dipoles == {} and result.gradients[S0].shape == (6, 3)


def test_unknown_functional_is_an_electronic_structure_error(
    ethylene_symbols, ethylene_bohr, make_settings
):
    settings = _tda_settings(make_settings, functional="no-such-functional")
    with pytest.raises(ElectronicStructureError, match="SCF failed"):
        PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings).compute(
            Request(ethylene_bohr)
        )
