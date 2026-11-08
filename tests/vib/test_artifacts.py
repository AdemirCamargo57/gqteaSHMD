import h5py
import numpy as np

from gqteashmd.vib.artifacts import write_frequencies_h5
from gqteashmd.vib.harmonic import harmonic_analysis


def test_frequencies_h5_round_trip(tmp_path):
    masses = np.array([1.007825, 1.007825])
    coords = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    e = np.array([0, 0, -1, 0, 0, 1]) / np.sqrt(2)
    hess = np.outer(e, e)
    analysis = harmonic_analysis(masses, coords, hess)
    path = tmp_path / "frequencies.h5"
    write_frequencies_h5(
        path,
        symbols=("H", "H"),
        masses_amu=masses,
        coords_bohr=coords,
        hessian=hess,
        analysis=analysis,
    )
    with h5py.File(path, "r") as f:
        assert f.attrs["schema_version"] == 1
        assert [s.decode() if isinstance(s, bytes) else s for s in f["symbols"][()]] == ["H", "H"]
        np.testing.assert_allclose(f["frequencies"][()], analysis.frequencies_cm1)
        assert f["frequencies"].attrs["units"] == "cm^-1"
        assert f["hessian"].attrs["units"] == "hartree/bohr^2"
        assert f["coordinates"].attrs["units"] == "bohr"
        assert bool(f.attrs["linear"]) is True
        assert int(f.attrs["n_external"]) == 5
    assert sorted(p.name for p in tmp_path.iterdir()) == ["frequencies.h5"]
