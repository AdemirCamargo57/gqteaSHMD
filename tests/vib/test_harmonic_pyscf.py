import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.hessian import thermo

from gqteashmd.chem.masses import isotope_mass
from gqteashmd.vib.harmonic import harmonic_analysis

pytestmark = pytest.mark.pyscf


@pytest.mark.parametrize(
    "atoms",
    [
        "O 0 0 0.108140699; H 0 0.754903478 -0.464620350; H 0 -0.754903478 -0.464620350",
        "C 0 0 0; O 0 0 1.16; O 0 0 -1.16",
    ],
)
def test_matches_pyscf_harmonic_analysis(atoms):
    mol = gto.M(atom=atoms, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    h4 = mf.Hessian().kernel()
    n = mol.natm
    masses = np.array([isotope_mass(mol.atom_symbol(i)) for i in range(n)])
    hess = h4.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    ours = harmonic_analysis(masses, mol.atom_coords(), hess)
    ref = thermo.harmonic_analysis(mol, h4, mass=masses)
    freq = np.asarray(ref["freq_wavenumber"])
    freq = np.where(freq.imag != 0, -np.abs(freq), freq.real)
    np.testing.assert_allclose(ours.frequencies_cm1, np.sort(freq), atol=1e-6)
    np.testing.assert_allclose(
        np.sort(ours.reduced_masses_amu), np.sort(ref["reduced_mass"]), rtol=1e-6
    )
