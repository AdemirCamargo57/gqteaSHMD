"""Triplet-block overlaps along a path (spec 4 item 4): TDA-HF/STO-3G water 2 + 2 and
SA-CASSCF(2,2)/6-31G* ethylene 2 + 1 (controller ruling: water CAS(4,4)/STO-3G scatters between
converged SA-CASSCF points by 1-2 % in its overlaps, with or without triplets)."""

import numpy as np
import pytest

from gqteashmd.core.units import ANGSTROM_TO_BOHR
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.pyscf_backend import PySCFBackend

pytestmark = pytest.mark.pyscf
PATH_STEP = 0.01  # bohr, every coordinate
WATER_SYMBOLS = ("O", "H", "H")
WATER_BOHR = (
    np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
    * ANGSTROM_TO_BOHR
)


@pytest.fixture
def system(request, make_settings, ethylene_symbols, ethylene_bohr):
    """(symbols, coordinates in bohr, settings) of the TDA or the CASSCF case."""
    if request.param == "tda":
        settings = make_settings(
            method="tda",
            basis="sto-3g",
            functional="hf",
            casscf=None,
            singlets=2,
            triplets=2,
            scf_conv_tol=1e-11,
            td_conv_tol=1e-9,
        )
        return WATER_SYMBOLS, WATER_BOHR, settings
    settings = make_settings(
        basis="6-31g*",
        singlets=2,
        triplets=1,
        scf_conv_tol=1e-11,
        casscf={"ncas": 2, "nelecas": 2, "conv_tol": 1e-10, "conv_tol_grad": 1e-6},
    )
    return tuple(ethylene_symbols), ethylene_bohr, settings


def _path(coords, n=5):
    return [coords + PATH_STEP * k for k in range(n)]


@pytest.mark.parametrize("system", ["tda", "casscf"], indirect=True)
def test_triplet_overlaps_are_continuous_along_a_path(system):
    symbols, coords0, settings = system
    backend = PySCFBackend(MoleculeSpec(symbols), settings)
    n_s = settings.singlets
    for k, coords in enumerate(_path(coords0)):
        result = backend.compute(Request(coords, overlaps=True))
        if k == 0:
            continue
        s = result.overlaps
        triplet = s[n_s:, n_s:]
        assert np.linalg.svd(triplet, compute_uv=False).min() >= 0.99
        assert np.all(np.diag(s) > 0)
        assert np.all(s[:n_s, n_s:] == 0.0) and np.all(s[n_s:, :n_s] == 0.0)


@pytest.mark.parametrize("system", ["tda", "casscf"], indirect=True)
def test_a_resumed_backend_gives_the_same_phased_overlaps(system, tmp_path):
    # Phases are defined relative to a backend's own first geometry, so the uninterrupted
    # reference is the same chain continued (controller ruling), not an independent backend.
    symbols, coords0, settings = system
    path = _path(coords0)
    chain = PySCFBackend(MoleculeSpec(symbols), settings)
    for c in path[:3]:
        chain.compute(Request(c, overlaps=True))
    chain.save_state(tmp_path / "state.npz")
    reference = {k: chain.compute(Request(path[k], overlaps=True)).overlaps for k in (3, 4)}
    resumed = PySCFBackend(MoleculeSpec(symbols), settings)
    resumed.load_state(tmp_path / "state.npz")
    for k in (3, 4):
        s = resumed.compute(Request(path[k], overlaps=True)).overlaps
        np.testing.assert_allclose(s, reference[k], atol=1e-10)
