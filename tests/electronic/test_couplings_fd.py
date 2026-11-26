import numpy as np
import pytest

from gqteashmd.core.states import StateLabel
from gqteashmd.core.units import ANGSTROM_TO_BOHR
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


# --- Mixed-spin state average (singlets + triplets) ---

T1, T2 = StateLabel(3, 1), StateLabel(3, 2)
H_TRIPLET = 1e-3  # bohr; at 1e-4 the FD-overlap noise of the mixed SA reaches 4.6e-4 1/bohr
WATER_SYMBOLS = ("O", "H", "H")
WATER_BOHR = (
    np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
    * ANGSTROM_TO_BOHR
)  # tests/conftest.py:WATER_XYZ


def _water_cas44_with_triplets(make_settings):
    return make_settings(
        basis="sto-3g",
        singlets=2,
        triplets=2,
        scf_conv_tol=1e-12,
        casscf={"ncas": 4, "nelecas": 4, "conv_tol": 1e-10, "conv_tol_grad": 1e-6},
    )


@pytest.mark.slow
def test_singlet_nac_of_the_mixed_sa_matches_finite_difference_overlaps(
    ethylene_symbols, ethylene_bohr, make_settings, tmp_path
):
    """S0-S1 NAC of SA-CASSCF(2,2)/6-31G* over 2 singlets + 1 triplet (controller ruling: water
    CAS(4,4)/STO-3G is off by 1.6e-3 with triplets and 2.3e-2 without, independent of h)."""
    settings = make_settings(
        basis="6-31g*",
        singlets=2,
        triplets=1,
        scf_conv_tol=1e-12,
        casscf={"ncas": 2, "nelecas": 2, "conv_tol": 1e-11, "conv_tol_grad": 1e-7},
    )
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings)
    reference = backend.compute(Request(ethylene_bohr, nacs=frozenset({(S0, S1)})))
    d = reference.nacs[(S0, S1)]
    snapshot = tmp_path / "R.npz"
    backend.save_state(snapshot)
    worst = 0.0
    for atom, axis in ((0, 0), (2, 1), (4, 0)):
        s01 = []
        for sign in (+1, -1):
            backend.load_state(snapshot)
            displaced = np.array(ethylene_bohr, copy=True)
            displaced[atom, axis] += sign * H_TRIPLET
            s01.append(backend.compute(Request(displaced, overlaps=True)).overlaps[0, 1])
        fd = (s01[0] - s01[1]) / (2 * H_TRIPLET)
        print(f"atom {atom} axis {axis}: FD = {fd:+.6e}  NAC = {d[atom, axis]:+.6e}")
        worst = max(worst, abs(fd - d[atom, axis]))
    print(f"max |FD overlap - NAC| = {worst:.2e} 1/bohr")
    assert worst <= 1e-4


def test_a_triplet_triplet_nac_request_is_refused(make_settings):
    backend = PySCFBackend(MoleculeSpec(WATER_SYMBOLS), _water_cas44_with_triplets(make_settings))
    with pytest.raises(ValueError, match="between triplet states are not available"):
        backend.compute(Request(WATER_BOHR, nacs=frozenset({(T1, T2)})))


def test_singlet_triplet_nacs_are_exactly_zero(make_settings):
    backend = PySCFBackend(MoleculeSpec(WATER_SYMBOLS), _water_cas44_with_triplets(make_settings))
    result = backend.compute(Request(WATER_BOHR, nacs=frozenset({(S1, T1), (S0, T2)})))
    for pair in ((S1, T1), (S0, T2)):
        assert result.nacs[pair].shape == (3, 3)
        assert np.array_equal(result.nacs[pair], np.zeros((3, 3))), pair
