"""The effective one-electron spin-orbit operators of `electronic/soc.py` (spec 3.1 and 4 item 3):
H2CO/6-31G at the committed RHF/6-31G* geometry."""

from pathlib import Path

import numpy as np
import pytest

from gqteashmd.errors import ConfigError

pytestmark = pytest.mark.pyscf
DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def formaldehyde():
    """(Mole, RHF AO density) of H2CO/6-31G."""
    from pyscf import gto, scf

    mol = gto.M(atom=str(DATA / "formaldehyde.xyz"), basis="6-31g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    return mol, mf.make_rdm1()


@pytest.fixture(scope="module")
def operators(formaldehyde):
    from gqteashmd.electronic import soc

    mol, dm = formaldehyde
    return {
        "one_electron": soc.one_electron_operator(mol),
        "somf": soc.somf_operator(mol, dm),
        "effective_charge": soc.effective_charge_operator(mol),
    }


@pytest.mark.parametrize("name", ["one_electron", "somf", "effective_charge"])
def test_operator_is_hermitian_and_purely_imaginary(formaldehyde, operators, name):
    mol, _ = formaldehyde
    h = operators[name]
    assert h.shape == (3, mol.nao, mol.nao)
    assert np.iscomplexobj(h)
    assert np.abs(h - h.conj().transpose(0, 2, 1)).max() <= 1e-12
    assert np.abs(h.real).max() <= 1e-14
    assert np.abs(h.imag).max() > 1e-6


def test_somf_without_the_two_electron_part_is_the_one_electron_operator(formaldehyde, operators):
    from gqteashmd.electronic import soc

    mol, dm = formaldehyde
    h = soc.somf_operator(mol, dm, two_electron=False)
    assert np.abs(h - operators["one_electron"]).max() <= 1e-12


def test_effective_charges_equal_to_the_nuclear_charges_give_the_one_electron_operator(
    formaldehyde, operators
):
    from gqteashmd.electronic import soc

    mol, _ = formaldehyde
    zeff = {mol.atom_pure_symbol(i): float(mol.atom_charge(i)) for i in range(mol.natm)}
    h = soc.effective_charge_operator(mol, zeff=zeff)
    assert np.abs(h - operators["one_electron"]).max() <= 1e-12


def test_the_two_electron_mean_field_screens_the_one_electron_term_on_oxygen(
    formaldehyde, operators
):
    mol, _ = formaldehyde
    oxygen = next(i for i in range(mol.natm) if mol.atom_pure_symbol(i) == "O")
    _, _, start, stop = mol.aoslice_by_atom()[oxygen]
    block = np.s_[:, start:stop, start:stop]
    one = np.linalg.norm(operators["one_electron"][block])
    somf = np.linalg.norm(operators["somf"][block])
    assert somf < one


def test_the_effective_charges_screen_the_one_electron_term(operators):
    assert np.linalg.norm(operators["effective_charge"]) < np.linalg.norm(operators["one_electron"])


def test_the_mean_field_equals_the_explicit_contraction_of_the_integrals():
    """The three index patterns of the docstring, against the full integral tensor (STO-3G)."""
    from pyscf import gto, scf

    from gqteashmd.core.units import SPEED_OF_LIGHT_AU
    from gqteashmd.electronic import soc

    mol = gto.M(atom=str(DATA / "formaldehyde.xyz"), basis="sto-3g", verbose=0)
    dm = scf.RHF(mol).run(conv_tol=1e-11).make_rdm1()
    g = mol.intor("int2e_p1vxp1", comp=3)
    coulomb = np.einsum("xpqrs,rs->xpq", g, dm)
    exchange = np.einsum("xpqrs,ps->xrq", g, dm) + np.einsum("xpqrs,rq->xps", g, dm)
    expected = 1j * (0.5 / SPEED_OF_LIGHT_AU**2) * (coulomb - 1.5 * exchange)
    mean_field = soc.somf_operator(mol, dm) - soc.one_electron_operator(mol)
    assert np.abs(mean_field - expected).max() <= 1e-12


def test_the_one_electron_integral_is_the_curl_of_the_nuclear_potential():
    """Pins the sign and the x, y, z order the convention rests on (STO-3G):
    int1e_pnucxp[k] = eps_kab <d_a p| V_N |d_b q>, with V_N = -sum Z/r negative."""
    from pyscf import gto

    mol = gto.M(atom=str(DATA / "formaldehyde.xyz"), basis="sto-3g", verbose=0)
    grad_grad = mol.intor("int1e_ipnucip", comp=9).reshape(3, 3, mol.nao, mol.nao)
    eps = np.zeros((3, 3, 3))
    for k, a, b in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        eps[k, a, b], eps[k, b, a] = 1.0, -1.0
    curl = np.einsum("kab,abpq->kpq", eps, grad_grad)
    pnucxp = mol.intor("int1e_pnucxp", comp=3)
    assert np.abs(pnucxp - curl).max() <= 1e-10 * np.abs(pnucxp).max()
    assert np.diag(mol.intor("int1e_nuc")).max() < 0.0


def test_a_ghost_atom_is_refused_by_the_effective_charge_operator():
    from pyscf import gto

    from gqteashmd.electronic import soc

    mol = gto.M(atom="O 0 0 0; ghost-H 0 0 1.8", basis="sto-3g", verbose=0, spin=0, charge=0)
    with pytest.raises(ConfigError, match="ghost"):
        soc.effective_charge_operator(mol)


def test_a_missing_effective_charge_is_a_config_error_naming_the_element(formaldehyde):
    from gqteashmd.electronic import soc

    mol, _ = formaldehyde
    with pytest.raises(ConfigError, match=r"no established effective nuclear charge for C\b"):
        soc.effective_charge_operator(mol, zeff={"H": 1.0})


def test_an_element_outside_the_table_is_a_config_error():
    from pyscf import gto

    from gqteashmd.electronic import soc

    mol = gto.M(atom="Cl 0 0 0; H 0 0 1.27", basis="sto-3g", unit="angstrom", verbose=0)
    with pytest.raises(ConfigError, match="no established effective nuclear charge for Cl"):
        soc.effective_charge_operator(mol)


def test_the_effective_charge_table():
    from gqteashmd.electronic import soc

    assert {"H", "C", "O", "S"} <= set(soc.KOSEKI_ZEFF)
    assert soc.KOSEKI_ZEFF["H"] == 1.0
    assert soc.KOSEKI_ZEFF["C"] == 3.6
    assert soc.KOSEKI_ZEFF["O"] == 5.6
    assert soc.KOSEKI_ZEFF["S"] == 13.6
    assert all(isinstance(value, float) for value in soc.KOSEKI_ZEFF.values())


def test_soc_operator_dispatches_on_the_kind(formaldehyde, operators):
    from gqteashmd.electronic import soc

    mol, dm = formaldehyde
    assert np.abs(soc.soc_operator(mol, dm, "somf") - operators["somf"]).max() <= 1e-12
    assert (
        np.abs(soc.soc_operator(mol, dm, "effective_charge") - operators["effective_charge"]).max()
        <= 1e-12
    )
    with pytest.raises(ValueError, match="unknown spin-orbit operator 'amfi'"):
        soc.soc_operator(mol, dm, "amfi")


def test_a_pyscf_failure_is_an_electronic_structure_error(formaldehyde, monkeypatch):
    from gqteashmd.electronic import soc
    from gqteashmd.errors import ElectronicStructureError

    mol, dm = formaldehyde

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(mol, "intor", broken)
    with pytest.raises(ElectronicStructureError, match="boom"):
        soc.one_electron_operator(mol)
