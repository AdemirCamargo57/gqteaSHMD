"""TDA/TDDFT spin-orbit blocks of `electronic/soc.py` (spec 3.2, 4 item 3): element by element
against the brute-force Slater-Condon reference of `soc_reference.py` (CIS-like states over all
MOs, X only), the Wigner-Eckart ratios of the Ms components, the shapes and the zero operator.

Systems: H2CO/STO-3G CIS (TDA-HF) 3 singlets + 3 triplets (the plan's system) and H2CO/STO-3G
TDDFT-B3LYP 2 singlets + 2 triplets (the TDDFT path: X renormalised, Y dropped).
"""

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from tests.electronic import soc_reference

from gqteashmd.config.schema import ElectronicSection
from gqteashmd.core.states import states_from_counts

pytestmark = pytest.mark.pyscf
DATA = Path(__file__).resolve().parents[1] / "data"
REFERENCE_BOUND = 1e-10  # Eh, spec 4 item 3
RATIO_BOUND = 1e-10  # relative, spec 4 item 3
SQRT2 = np.sqrt(2.0)
SYSTEMS = {
    "cis": ("tda", "hf", 3, 3),
    "tddft": ("tddft", "b3lyp", 2, 2),
}


@pytest.fixture(scope="module", params=list(SYSTEMS))
def response(request):
    """(Mole, SCF object, ResponseOutcome, states, method)."""
    from pyscf import dft, gto, scf

    from gqteashmd.electronic import pyscf_response

    method, functional, singlets, triplets = SYSTEMS[request.param]
    mol = gto.M(atom=str(DATA / "formaldehyde.xyz"), basis="sto-3g", verbose=0)
    if functional == "hf":
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol)
        mf.xc = functional
    mf.chkfile = None
    mf._chkfile = None  # PySCF's temporary chkfile would leak an open file into -W error
    mf.run(conv_tol=1e-11)
    settings = ElectronicSection.model_validate(
        {
            "method": method,
            "basis": "sto-3g",
            "functional": functional,
            "singlets": singlets,
            "triplets": triplets,
            "td_conv_tol": 1e-9,
        }
    )
    states = states_from_counts(singlets, triplets)
    outcome = pyscf_response.run_response(mf, settings, states)
    return mol, mf, outcome, states, method


def _random_matrix(nao: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((nao, nao)) + 1j * rng.standard_normal((nao, nao))


def _operator(mol, mf, kind: str) -> np.ndarray:
    """h_ao (3, nao, nao): the SOMF operator of the SCF density, a random purely imaginary
    Hermitian one, or a random complex Hermitian one."""
    from gqteashmd.electronic import soc

    if kind == "somf":
        return soc.soc_operator(mol, mf.make_rdm1(), "somf")
    rng = np.random.default_rng(20261127)
    if kind == "imaginary":
        a = rng.standard_normal((3, mol.nao, mol.nao))
        return 1j * (a - a.transpose(0, 2, 1))
    b = rng.standard_normal((3, mol.nao, mol.nao)) + 1j * rng.standard_normal((3, mol.nao, mol.nao))
    return b + b.conj().transpose(0, 2, 1)


def _blocks(mf, outcome, states, h_ao):
    from gqteashmd.electronic import soc

    return soc.response_soc_blocks(mf, outcome.td, outcome.td_triplet, states, h_ao)


def _reference_blocks(mf, outcome, states, h_ao):
    """(st, tt) of the brute-force reference, with the M index 0, 1, 2 = Ms -1, 0, +1."""
    ref = soc_reference.response_reference_states(mf, outcome.td, outcome.td_triplet, states)
    c = mf.mo_coeff
    h_mo = np.einsum("pi,kpq,qj->kij", c, h_ao, c)
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


def _in_slot(matrix: np.ndarray, k: int) -> np.ndarray:
    h = np.zeros((3, *matrix.shape), dtype=complex)
    h[k] = matrix
    return h


def test_the_tddft_amplitudes_need_renormalising(response):
    """Not vacuous: TDDFT's X alone is not normalised to sum X^2 = 1/2 (TDA's is)."""
    _, _, outcome, states, method = response
    norms = [2 * np.sum(np.asarray(x) ** 2) for x, _ in outcome.td_triplet.xy]
    if outcome.td is not None:
        norms += [2 * np.sum(np.asarray(x) ** 2) for x, _ in outcome.td.xy]
    deviation = max(abs(n - 1.0) for n in norms)
    if method == "tda":
        assert deviation <= 1e-12
    else:
        assert deviation > 1e-4


def test_reference_states_are_orthonormal_and_complete(response):
    _, mf, outcome, states, _ = response
    ref = soc_reference.response_reference_states(mf, outcome.td, outcome.td_triplet, states)
    keys = sorted(ref, key=lambda k: (k[0], k[1]))
    assert len(keys) == sum(1 if s.multiplicity == 1 else 3 for s in states)
    gram = np.array([[soc_reference.overlap(ref[a], ref[b]) for b in keys] for a in keys])
    assert np.abs(gram - np.eye(len(keys))).max() <= 1e-12
    for s in (s for s in states if s.multiplicity == 1):  # S_minus |S> = 0
        assert soc_reference.lowering(ref[(s, 0)]).norm() <= 1e-12
    for t in (s for s in states if s.multiplicity == 3):
        assert soc_reference.raising(ref[(t, 1)]).norm() <= 1e-12
        assert soc_reference.lowering(ref[(t, -1)]).norm() <= 1e-12
        # Condon-Shortley: |T, 0> = S_minus |T, +1> / sqrt(2)
        down = soc_reference.lowering(ref[(t, 1)])
        assert abs(soc_reference.overlap(ref[(t, 0)], down) - SQRT2) <= 1e-12


def test_the_blocks_have_only_singlet_triplet_and_triplet_triplet_parts(response):
    mol, mf, outcome, states, _ = response
    n_s = sum(1 for s in states if s.multiplicity == 1)
    n_t = len(states) - n_s
    blocks = _blocks(mf, outcome, states, _operator(mol, mf, "somf"))
    assert [f.name for f in fields(blocks)] == ["st", "tt"]
    assert blocks.st.shape == (n_s, n_t, 3)
    assert blocks.tt.shape == (n_t, n_t, 3, 3)
    assert np.iscomplexobj(blocks.st) and np.iscomplexobj(blocks.tt)


def test_a_zero_operator_gives_exactly_zero_blocks(response):
    mol, mf, outcome, states, _ = response
    blocks = _blocks(mf, outcome, states, np.zeros((3, mol.nao, mol.nao), dtype=complex))
    assert np.all(blocks.st == 0)
    assert np.all(blocks.tt == 0)


@pytest.mark.parametrize("kind", ["somf", "imaginary", "hermitian"])
def test_the_blocks_equal_the_brute_force_reference(response, kind):
    mol, mf, outcome, states, method = response
    h_ao = _operator(mol, mf, kind)
    blocks = _blocks(mf, outcome, states, h_ao)
    st, tt = _reference_blocks(mf, outcome, states, h_ao)
    d_st = np.abs(blocks.st - st).max()
    d_tt = np.abs(blocks.tt - tt).max()
    print(f"{method} {kind}: max |st| {np.abs(st).max():.3e} diff {d_st:.3e}; ", end="")
    print(f"max |tt| {np.abs(tt).max():.3e} diff {d_tt:.3e}")
    # not vacuous: the S0-T rows and the excited S-T rows are both non-zero
    assert np.abs(st[0]).max() > 1e-6
    assert np.abs(st[1:]).max() > 1e-6
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


def _check_wigner_eckart(st_by_slot, tt_by_slot):
    """Assert the WE_ST / WE_TT ratios for every pair; return the number of pairs checked."""
    checked = 0
    st_z = st_by_slot[2]
    for i in range(st_z.shape[0]):
        for j in range(st_z.shape[1]):
            z0 = st_z[i, j, 1]
            assert abs(z0) > 1e-8
            for k in range(3):
                expected = z0 * WE_ST[k]
                assert np.abs(st_by_slot[k][i, j] - expected).max() <= RATIO_BOUND * abs(z0)
            checked += 1
    tt_z = tt_by_slot[2]
    for j in range(tt_z.shape[0]):
        for jp in range(tt_z.shape[1]):
            z0 = tt_z[j, jp, 2, 2]
            assert abs(z0) > 1e-8
            for k in range(3):
                expected = z0 * WE_TT[k]
                assert np.abs(tt_by_slot[k][j, jp] - expected).max() <= RATIO_BOUND * abs(z0)
            checked += 1
    return checked


def test_the_ms_components_follow_wigner_eckart(response):
    mol, mf, outcome, states, _ = response
    b = _random_matrix(mol.nao, 7)
    blocks = {k: _blocks(mf, outcome, states, _in_slot(b, k)) for k in range(3)}
    n_s = sum(1 for s in states if s.multiplicity == 1)
    n_t = len(states) - n_s
    checked = _check_wigner_eckart(
        {k: v.st for k, v in blocks.items()}, {k: v.tt for k, v in blocks.items()}
    )
    assert checked == n_s * n_t + n_t * n_t


def test_the_reference_follows_the_same_wigner_eckart_ratios(response):
    mol, mf, outcome, states, _ = response
    b = _random_matrix(mol.nao, 7)
    blocks = {k: _reference_blocks(mf, outcome, states, _in_slot(b, k)) for k in range(3)}
    n_s = sum(1 for s in states if s.multiplicity == 1)
    n_t = len(states) - n_s
    checked = _check_wigner_eckart(
        {k: v[0] for k, v in blocks.items()}, {k: v[1] for k, v in blocks.items()}
    )
    assert checked == n_s * n_t + n_t * n_t


def test_a_failure_inside_the_contractions_is_an_electronic_structure_error(response):
    from types import SimpleNamespace

    from gqteashmd.electronic import soc
    from gqteashmd.errors import ElectronicStructureError

    mol, mf, outcome, states, _ = response
    broken = SimpleNamespace(  # wrong X shape
        xy=[(np.zeros((1, 1)), 0) for _ in outcome.td_triplet.xy],
    )
    with pytest.raises(ElectronicStructureError, match="spin-orbit matrix elements"):
        soc.response_soc_blocks(mf, outcome.td, broken, states, _operator(mol, mf, "imaginary"))
