"""SA-CASSCF spin-orbit blocks of `electronic/soc.py` (spec 3.2, 4 item 3): element by element
against the brute-force Slater-Condon reference of `soc_reference.py`, the Wigner-Eckart ratios
of the Ms components, the shapes and the zero operator.

Systems: H2CO/STO-3G SA-CASSCF(4,3) 2 singlets + 2 triplets (the plan's system) and ethylene/
STO-3G SA-CASSCF(2,2) 2 singlets + 1 triplet (an odd number of core orbitals and a triplet with
no active beta electron).
"""

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from tests.electronic import soc_reference, soc_testing

from gqteashmd.config.schema import ElectronicSection
from gqteashmd.core.states import states_from_counts

pytestmark = pytest.mark.pyscf
DATA = Path(__file__).resolve().parents[1] / "data"
REFERENCE_BOUND = 1e-10  # Eh, spec 4 item 3
RATIO_BOUND = 1e-10  # relative, spec 4 item 3
SQRT2 = np.sqrt(2.0)
SYSTEMS = {
    "formaldehyde": ("formaldehyde.xyz", 4, 3, 2, 2),
    "ethylene": ("ethylene.xyz", 2, 2, 2, 1),
}


@pytest.fixture(scope="module", params=list(SYSTEMS))
def casscf(request):
    """(Mole, converged mixed SA-CASSCF object, states)."""
    from pyscf import gto, scf

    from gqteashmd.electronic import pyscf_casscf

    xyz, nelecas, ncas, singlets, triplets = SYSTEMS[request.param]
    mol = gto.M(atom=str(DATA / xyz), basis="sto-3g", verbose=0)
    mf = scf.RHF(mol)
    mf.chkfile = None
    mf._chkfile = None  # PySCF's temporary chkfile would leak an open file into -W error
    mf.run(conv_tol=1e-11)
    settings = ElectronicSection.model_validate(
        {
            "method": "casscf",
            "basis": "sto-3g",
            "singlets": singlets,
            "triplets": triplets,
            "casscf": {"ncas": ncas, "nelecas": nelecas, "conv_tol": 1e-10, "conv_tol_grad": 1e-6},
        }
    )
    states = states_from_counts(singlets, triplets)
    outcome = pyscf_casscf.run_casscf(mol, mf, settings, states, None)
    return mol, outcome.mc, states


def _operator(mol, mc, kind: str) -> np.ndarray:
    """h_ao (3, nao, nao): the SOMF operator, a random purely imaginary Hermitian one, or a
    random complex Hermitian one."""
    from gqteashmd.electronic import soc

    if kind == "somf":
        return soc.soc_operator(mol, mc.make_rdm1(), "somf")
    rng = np.random.default_rng(20261127)
    if kind == "imaginary":
        a = rng.standard_normal((3, mol.nao, mol.nao))
        return 1j * (a - a.transpose(0, 2, 1))
    b = rng.standard_normal((3, mol.nao, mol.nao)) + 1j * rng.standard_normal((3, mol.nao, mol.nao))
    return b + b.conj().transpose(0, 2, 1)


def _to_mo(mc, h_ao: np.ndarray) -> np.ndarray:
    c = mc.mo_coeff
    return np.einsum("pi,kpq,qj->kij", c, h_ao, c)


def _reference_blocks(mc, states, h_ao):
    """(st, tt) of the brute-force reference, with the M index 0, 1, 2 = Ms -1, 0, +1."""
    ref = soc_reference.casscf_reference_states(mc, states)
    h_mo = _to_mo(mc, h_ao)
    singlets = [s for s in states if s.multiplicity == 1]
    triplets = [s for s in states if s.multiplicity == 3]
    st = np.zeros((len(singlets), len(triplets), 3), dtype=complex)
    tt = np.zeros((len(triplets), len(triplets), 3, 3), dtype=complex)
    for i, s in enumerate(singlets):
        for j, t in enumerate(triplets):
            for m in range(3):
                st[i, j, m] = soc_reference.one_body_element(ref[(s, 0)], ref[(t, m - 1)], h_mo)
    for j, t in enumerate(triplets):
        for jp, tp in enumerate(triplets):
            for m in range(3):
                for mp in range(3):
                    tt[j, jp, m, mp] = soc_reference.one_body_element(
                        ref[(t, m - 1)], ref[(tp, mp - 1)], h_mo
                    )
    return st, tt


def test_reference_triplet_components_are_orthonormal_and_complete(casscf):
    _, mc, states = casscf
    ref = soc_reference.casscf_reference_states(mc, states)
    triplets = [s for s in states if s.multiplicity == 3]
    keys = [(t, m) for t in triplets for m in (-1, 0, 1)]
    gram = np.array([[soc_reference.overlap(ref[a], ref[b]) for b in keys] for a in keys])
    assert np.abs(gram - np.eye(len(keys))).max() <= 1e-12
    for t in triplets:  # S_minus |T, -1> = 0
        assert soc_reference.lowering(ref[(t, -1)]).norm() <= 1e-12
    for s in (s for s in states if s.multiplicity == 1):  # S_minus |S> = 0, |S> normalised
        assert abs(soc_reference.overlap(ref[(s, 0)], ref[(s, 0)]) - 1.0) <= 1e-12
        assert soc_reference.lowering(ref[(s, 0)]).norm() <= 1e-12


def test_the_blocks_have_only_singlet_triplet_and_triplet_triplet_parts(casscf):
    from gqteashmd.electronic import soc

    mol, mc, states = casscf
    n_s = sum(1 for s in states if s.multiplicity == 1)
    n_t = len(states) - n_s
    blocks = soc.casscf_soc_blocks(mc, states, _operator(mol, mc, "somf"))
    assert [f.name for f in fields(blocks)] == ["st", "tt"]
    assert blocks.st.shape == (n_s, n_t, 3)
    assert blocks.tt.shape == (n_t, n_t, 3, 3)
    assert np.iscomplexobj(blocks.st) and np.iscomplexobj(blocks.tt)


def test_a_zero_operator_gives_exactly_zero_blocks(casscf):
    from gqteashmd.electronic import soc

    mol, mc, states = casscf
    blocks = soc.casscf_soc_blocks(mc, states, np.zeros((3, mol.nao, mol.nao), dtype=complex))
    assert np.all(blocks.st == 0)
    assert np.all(blocks.tt == 0)


@pytest.mark.parametrize("kind", ["somf", "imaginary", "hermitian"])
def test_the_blocks_equal_the_brute_force_reference(request, casscf, kind):
    from gqteashmd.electronic import soc

    mol, mc, states = casscf
    h_ao = _operator(mol, mc, kind)
    blocks = soc.casscf_soc_blocks(mc, states, h_ao)
    st, tt = _reference_blocks(mc, states, h_ao)
    d_st = np.abs(blocks.st - st).max()
    d_tt = np.abs(blocks.tt - tt).max()
    print(f"{kind}: max |st| {np.abs(st).max():.3e} diff {d_st:.3e}; ", end="")
    print(f"max |tt| {np.abs(tt).max():.3e} diff {d_tt:.3e}")
    if kind != "somf" or request.node.callspec.params["casscf"] == "formaldehyde":
        # not a vacuous comparison (ethylene pi-pi* S-T SOMF elements vanish by symmetry)
        assert np.abs(st).max() > 1e-6
    if kind == "hermitian":  # the T-T comparison is not vacuous either
        assert np.abs(tt).max() > 1e-6
    assert d_st <= REFERENCE_BOUND
    assert d_tt <= REFERENCE_BOUND


# Wigner-Eckart with the Conventions: H_SO = h_z s_z + (h_plus s_minus + h_minus s_plus) / 2,
# h_plus = h_x + i h_y, h_minus = h_x - i h_y, and |T, M - 1> = S_minus |T, M> / sqrt(2). Put one
# spatial matrix B into a single Cartesian slot. For a singlet-triplet pair, with
# z0 = <S| H(B in z) |T, 0> (the one independent element), the M = -1, 0, +1 components are
#   B in z: (0, z0, 0)
#   B in x: (z0 / sqrt(2), 0, -z0 / sqrt(2))
#   B in y: (-i z0 / sqrt(2), 0, -i z0 / sqrt(2))
# For a triplet-triplet pair, with z0 = <T, +1| H(B in z) |T', +1>, the 3 x 3 blocks [M, M']
# (rows and columns Ms = -1, 0, +1) are z0 times the matrices of WE_TT below.
WE_ST = {
    2: np.array([0.0, 1.0, 0.0]),
    0: np.array([1.0, 0.0, -1.0]) / SQRT2,
    1: np.array([-1j, 0.0, -1j]) / SQRT2,
}
WE_TT = {
    2: np.diag([-1.0, 0.0, 1.0]).astype(complex),
    0: np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex) / SQRT2,
    1: np.array([[0, 1j, 0], [-1j, 0, 1j], [0, -1j, 0]], dtype=complex) / SQRT2,
}


# On the production code this test checks the assembled formula against the hand-written tables
# (it caught a transposition); the physical evidence is the same test on the reference and the
# element-by-element reference comparison.
def test_the_ms_components_follow_wigner_eckart(casscf):
    from gqteashmd.electronic import soc

    mol, mc, states = casscf
    b = soc_testing.random_matrix(mol.nao, 7)
    st = {k: soc.casscf_soc_blocks(mc, states, soc_testing.in_slot(b, k)).st for k in range(3)}
    tt = {k: soc.casscf_soc_blocks(mc, states, soc_testing.in_slot(b, k)).tt for k in range(3)}
    n_s = sum(1 for s in states if s.multiplicity == 1)
    n_t = len(states) - n_s
    assert (
        soc_testing.check_wigner_eckart(st, tt, WE_ST, WE_TT, RATIO_BOUND) == n_s * n_t + n_t * n_t
    )


def test_the_reference_follows_the_same_wigner_eckart_ratios(casscf):
    mol, mc, states = casscf
    b = soc_testing.random_matrix(mol.nao, 7)
    blocks = {k: _reference_blocks(mc, states, soc_testing.in_slot(b, k)) for k in range(3)}
    n_s = sum(1 for s in states if s.multiplicity == 1)
    n_t = len(states) - n_s
    checked = soc_testing.check_wigner_eckart(
        {k: v[0] for k, v in blocks.items()},
        {k: v[1] for k, v in blocks.items()},
        WE_ST,
        WE_TT,
        RATIO_BOUND,
    )
    assert checked == n_s * n_t + n_t * n_t


def test_a_failure_inside_the_ci_algebra_is_an_electronic_structure_error(casscf):
    from types import SimpleNamespace

    from gqteashmd.electronic import soc
    from gqteashmd.errors import ElectronicStructureError

    mol, mc, states = casscf
    broken = SimpleNamespace(
        ncore=mc.ncore,
        ncas=mc.ncas,
        nelecas=mc.nelecas,
        mo_coeff=mc.mo_coeff,
        ci=[np.zeros(np.asarray(ci).size + 1) for ci in mc.ci],  # wrong CI length
    )
    with pytest.raises(ElectronicStructureError, match="spin-orbit matrix elements"):
        soc.casscf_soc_blocks(broken, states, _operator(mol, mc, "imaginary"))
