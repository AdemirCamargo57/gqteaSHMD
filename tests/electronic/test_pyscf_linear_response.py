import numpy as np
import pytest
from pyscf import dft, gto, scf, tddft
from scipy.linalg import block_diag

from gqteashmd.core.states import StateLabel
from gqteashmd.electronic import pyscf_response
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.properties import oscillator_strength
from gqteashmd.electronic.pyscf_backend import PySCFBackend
from gqteashmd.errors import ElectronicStructureError
from gqteashmd.overlaps.determinants import state_overlaps, truncate

pytestmark = pytest.mark.pyscf

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


def _check_dipoles(result, td, labels):
    """Compare transition dipoles and oscillator strengths with direct PySCF.

    The dipole sign is arbitrary per state; at least one compared state must be bright, so the
    check cannot pass on all-zero dipoles.
    """
    mu_ref = td.transition_dipole()
    f_ref = td.oscillator_strength()
    assert max(np.linalg.norm(m) for m in mu_ref) > 0.1
    for i, label in enumerate(labels):
        a, b = result.transition_dipoles[label], mu_ref[i]
        assert min(np.abs(a - b).max(), np.abs(a + b).max()) < 1e-6
        f = oscillator_strength(result.excitation_energy(label), a)
        assert f == pytest.approx(f_ref[i], rel=1e-6, abs=1e-12)


def _direct(symbols, coords, functional, method="tda", nstates=2):
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
    td.nstates = nstates
    td.conv_tol = 1e-8
    td.kernel()
    return mf, td


def test_tda_matches_direct_pyscf(ethylene_symbols, ethylene_bohr, make_settings):
    # STO-3G B3LYP: S1 and S2 are dark, the third excited state (S3) is bright (|mu| = 1.5)
    settings = _tda_settings(make_settings, singlets=4)
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings)
    result = backend.compute(
        Request(ethylene_bohr, gradients=frozenset({S0, S1}), transition_dipoles=True)
    )
    mf, td = _direct(ethylene_symbols, ethylene_bohr, "b3lyp", nstates=3)
    labels = (S1, S2, StateLabel(1, 3))
    assert abs(result.energies[S0] - mf.e_tot) < 1e-8
    for label, omega in zip(labels, td.e, strict=True):
        assert abs(result.excitation_energy(label) - omega) < 1e-6
    _check_dipoles(result, td, labels)
    assert np.abs(result.gradients[S0] - mf.nuc_grad_method().kernel()).max() < 1e-6
    assert np.abs(result.gradients[S1] - td.nuc_grad_method().kernel(state=1)).max() < 1e-5
    assert result.spin_squares == {}


def test_cis_and_tdhf_with_hf(ethylene_symbols, ethylene_bohr, make_settings):
    for method in ("tda", "tddft"):
        settings = _tda_settings(make_settings, method=method, functional="hf")
        result = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings).compute(
            Request(ethylene_bohr, gradients=frozenset({S1, S2}), transition_dipoles=True)
        )
        mf, td = _direct(ethylene_symbols, ethylene_bohr, "hf", method)
        assert abs(result.energies[S0] - mf.e_tot) < 1e-8
        for label, omega in zip((S1, S2), td.e, strict=True):
            assert abs(result.excitation_energy(label) - omega) < 1e-6
        _check_dipoles(result, td, (S1, S2))  # S1 is bright here (|mu| about 1.4-1.6)
        for k, label in ((1, S1), (2, S2)):
            ref = td.nuc_grad_method().kernel(state=k)
            assert result.gradients[label].shape == (6, 3)
            assert np.abs(result.gradients[label] - ref).max() < 1e-5


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
    with pytest.raises(ElectronicStructureError, match=r"SCF (.*) failed"):
        PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings).compute(
            Request(ethylene_bohr)
        )


def _scf(symbols, coords, functional):
    mol = gto.M(
        atom=[(s, tuple(c)) for s, c in zip(symbols, coords, strict=True)],
        unit="Bohr",
        basis="sto-3g",
        verbose=0,
    )
    mf = scf.RHF(mol) if functional == "hf" else dft.RKS(mol)
    if functional != "hf":
        mf.xc = functional
    mf.conv_tol = 1e-10
    mf.kernel()
    return mol, mf


def test_tda_expansion_is_orthonormal_at_equal_geometry(
    ethylene_symbols, ethylene_bohr, make_settings
):
    mol, mf = _scf(ethylene_symbols, ethylene_bohr, "b3lyp")
    settings = _tda_settings(make_settings, singlets=3)
    outcome = pyscf_response.run_response(mf, settings, (S0, S1, S2))
    expansion = pyscf_response.response_expansion(mf, outcome.td, 2)
    s_mo = mf.mo_coeff.T @ mol.intor("int1e_ovlp") @ mf.mo_coeff
    np.testing.assert_allclose(state_overlaps(expansion, expansion, s_mo), np.eye(3), atol=1e-8)
    kept = truncate(expansion, 0.9999)
    assert kept.n_dets < expansion.n_dets
    np.testing.assert_allclose(np.diag(state_overlaps(kept, kept, s_mo)), 1.0, atol=1e-10)


def test_response_ladder_retries_with_extra_roots(
    ethylene_symbols, ethylene_bohr, make_settings, monkeypatch
):
    calls = []
    original = pyscf_response._converged_roots

    def flaky(td, n):
        calls.append(td.nstates)
        return len(calls) > 1 and original(td, n)

    monkeypatch.setattr(pyscf_response, "_converged_roots", flaky)
    result = PySCFBackend(
        MoleculeSpec(tuple(ethylene_symbols)), _tda_settings(make_settings)
    ).compute(Request(ethylene_bohr))
    assert calls == [5, 8] and result.meta["td"] == "extra roots"
    _, td = _direct(ethylene_symbols, ethylene_bohr, "b3lyp")
    for label, omega in zip((S1, S2), td.e, strict=True):
        assert abs(result.excitation_energy(label) - omega) < 1e-6


def test_response_ladder_exhausted(ethylene_symbols, ethylene_bohr, make_settings, monkeypatch):
    monkeypatch.setattr(pyscf_response, "_converged_roots", lambda td, n: False)
    with pytest.raises(ElectronicStructureError, match="did not converge for all 2 excited states"):
        PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), _tda_settings(make_settings)).compute(
            Request(ethylene_bohr)
        )


def _rotated_overlap_reference(xs, q):
    """Explicit overlaps of the spin-adapted singles when the virtuals are rotated by q."""
    n = len(xs)
    ref = np.zeros((n + 1, n + 1))
    ref[0, 0] = 1.0
    for k in range(n):
        for m in range(n):
            ref[k + 1, m + 1] = 2.0 * np.einsum("ia,ib,ab->", xs[k], xs[m], q)
    return ref


@pytest.mark.parametrize("functional, method", [("b3lyp", "tda"), ("hf", "tddft")])
def test_expansion_layout_with_rotated_virtuals(
    ethylene_symbols, ethylene_bohr, make_settings, functional, method
):
    mol, mf = _scf(ethylene_symbols, ethylene_bohr, functional)
    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    nvir = mf.mo_coeff.shape[1] - nocc
    assert nocc != nvir
    settings = _tda_settings(make_settings, method=method, functional=functional, singlets=3)
    outcome = pyscf_response.run_response(mf, settings, (S0, S1, S2))
    expansion = pyscf_response.response_expansion(mf, outcome.td, 2)
    q, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(nvir, nvir)))
    s_mo = block_diag(np.eye(nocc), q)
    xs = [np.asarray(outcome.td.xy[k][0]) for k in range(2)]
    np.testing.assert_allclose(
        state_overlaps(expansion, expansion, s_mo), _rotated_overlap_reference(xs, q), atol=1e-12
    )


def test_tdhf_x_only_norm_exceeds_one_by_twice_sum_y_squared(
    ethylene_symbols, ethylene_bohr, make_settings
):
    """Dropping Y from an RPA state inflates its norm: 2 sum X^2 = 1 + 2 sum Y^2 > 1.

    Measured for ethylene/STO-3G TDHF, S1 has sum Y^2 of about 1.1e-2, so the X-only state has
    a norm^2 of about 1.023 before renormalisation (a 2.3% change).
    """
    mol, mf = _scf(ethylene_symbols, ethylene_bohr, "hf")
    settings = _tda_settings(make_settings, method="tddft", functional="hf", singlets=3)
    outcome = pyscf_response.run_response(mf, settings, (S0, S1, S2))
    expansion = pyscf_response.response_expansion(mf, outcome.td, 2)
    s_mo = mf.mo_coeff.T @ mol.intor("int1e_ovlp") @ mf.mo_coeff
    diag = np.diag(state_overlaps(expansion, expansion, s_mo))
    for k in range(2):
        sum_y2 = float(np.sum(np.asarray(outcome.td.xy[k][1]) ** 2))
        assert sum_y2 > 0
        assert diag[k + 1] > 1.0
        assert abs(diag[k + 1] - (1.0 + 2.0 * sum_y2)) < 1e-10


def test_too_many_excited_states_for_the_single_excitation_space(make_settings):
    mol, mf = _scf(["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]), "hf")
    settings = _tda_settings(make_settings, functional="hf", singlets=3)
    with pytest.raises(ElectronicStructureError, match="only 1 single excitation"):
        pyscf_response.run_response(mf, settings, (S0, S1, S2))


_WATER_ATOMS = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"


def test_symmetric_geometry_does_not_skip_a_root(make_settings):
    """Water CIS at the C2v test geometry: asking for 2 roots used to return the third
    (16.7707 eV) as S2 because the Davidson guess lacked one irrep (measured, Task 9)."""
    from gqteashmd.core.units import ANGSTROM_TO_BOHR

    coords = (
        np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
        * ANGSTROM_TO_BOHR
    )
    settings = _tda_settings(make_settings, functional="hf", singlets=3)
    result = PySCFBackend(MoleculeSpec(("O", "H", "H")), settings).compute(Request(coords))
    ev = 27.211386245988
    assert abs(result.excitation_energy(S1) * ev - 13.1877) < 1e-3
    assert abs(result.excitation_energy(S2) * ev - 15.1445) < 1e-3


def test_first_rung_requests_three_extra_roots_and_keeps_the_lowest(make_settings, monkeypatch):
    calls = []
    original = pyscf_response._converged_roots

    def spy(td, n):
        calls.append((td.nstates, n, len(td.e)))
        return original(td, n)

    monkeypatch.setattr(pyscf_response, "_converged_roots", spy)
    mol = gto.M(atom=_WATER_ATOMS, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    settings = _tda_settings(make_settings, functional="hf")
    out = pyscf_response.run_response(mf, settings, (S0, S1, S2))
    assert calls == [(5, 2, 5)] and out.rung == "default"
    assert set(out.energies) == {S0, S1, S2}
    assert abs(out.energies[S2] - (mf.e_tot + out.td.e[1])) < 1e-12
