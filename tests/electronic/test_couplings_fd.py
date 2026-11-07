import numpy as np
import pytest

from gqteashmd.core.states import StateLabel
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.pyscf_backend import PySCFBackend

pytestmark = pytest.mark.pyscf

S0, S1 = StateLabel(1, 0), StateLabel(1, 1)
H = 1e-4  # bohr


@pytest.mark.slow
def test_casscf_nacs_match_finite_difference_overlaps(
    ethylene_symbols, ethylene_bohr, make_settings, tmp_path
):
    settings = make_settings(
        basis="6-31g",
        scf_conv_tol=1e-12,
        # conv_tol_grad = 1e-6 leaves ~1e-3 noise in the FD; 1e-7 is the tightest that converges
        casscf={"ncas": 2, "nelecas": 2, "conv_tol": 1e-11, "conv_tol_grad": 1e-7},
    )
    spec = MoleculeSpec(tuple(ethylene_symbols))
    backend = PySCFBackend(spec, settings)
    reference = backend.compute(Request(ethylene_bohr, nacs=frozenset({(S0, S1)})))
    d = reference.nacs[(S0, S1)]
    # One reference frame: every displaced overlap is phase-aligned to the snapshot at R.
    snapshot = tmp_path / "R.npz"
    backend.save_state(snapshot)
    worst = 0.0
    for atom, axis in ((0, 0), (2, 1), (4, 0)):
        s01 = []
        for sign in (+1, -1):
            backend.load_state(snapshot)
            displaced = np.array(ethylene_bohr, copy=True)
            displaced[atom, axis] += sign * H
            s01.append(backend.compute(Request(displaced, overlaps=True)).overlaps[0, 1])
        fd = (s01[0] - s01[1]) / (2 * H)
        print(f"atom {atom} axis {axis}: FD = {fd:+.6e}  NAC = {d[atom, axis]:+.6e}")
        worst = max(worst, abs(fd - d[atom, axis]))
        assert abs(fd - d[atom, axis]) < 2e-4, (atom, axis, fd, d[atom, axis])
    print(f"max |FD overlap - NAC| = {worst:.2e} 1/bohr")
