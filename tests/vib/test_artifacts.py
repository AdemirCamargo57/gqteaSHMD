import h5py
import numpy as np
import pytest

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


def test_read_frequencies_h5_round_trip(tmp_path):
    from gqteashmd.vib.artifacts import read_frequencies_h5

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
    data = read_frequencies_h5(path)
    assert data["symbols"] == ("H", "H")
    assert all(isinstance(s, str) for s in data["symbols"])
    np.testing.assert_allclose(data["masses_amu"], masses)
    np.testing.assert_allclose(data["coords_bohr"], coords)
    np.testing.assert_allclose(data["frequencies_cm1"], analysis.frequencies_cm1)
    np.testing.assert_allclose(data["modes_mass_weighted"], analysis.modes_mass_weighted)
    assert data["linear"] is True and data["n_external"] == analysis.n_external
    assert isinstance(data["n_external"], int)


def test_samples_h5_round_trip(tmp_path):
    from gqteashmd.vib.artifacts import write_samples_h5
    from gqteashmd.vib.wigner import sample_wigner

    masses = np.array([1.007825, 1.007825])
    coords = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    e = np.array([0, 0, -1, 0, 0, 1]) / np.sqrt(2)
    a = harmonic_analysis(masses, coords, np.outer(e, e))
    s = sample_wigner(
        masses,
        coords,
        a.frequencies_cm1,
        a.modes_mass_weighted,
        n_samples=10,
        temperature_K=0.0,
        low_frequency_cutoff_cm1=10.0,
        rng=np.random.default_rng(1),
    )
    path = tmp_path / "samples.h5"
    write_samples_h5(
        path,
        symbols=("H", "H"),
        masses_amu=masses,
        coords_bohr=coords,
        frequencies_cm1=a.frequencies_cm1,
        samples=s,
        temperature_K=0.0,
        low_frequency_cutoff_cm1=10.0,
        project_seed=42,
    )
    with h5py.File(path, "r") as f:
        assert f.attrs["schema_version"] == 1
        assert f.attrs["n_samples"] == 10 and f.attrs["project_seed"] == 42
        assert f.attrs["temperature_K"] == 0.0 and f.attrs["low_frequency_cutoff_cm1"] == 10.0
        assert "gqteashmd_version" in f.attrs
        assert f.attrs["expected_mean_energy_hartree"] == s.expected_mean_energy_hartree
        assert f["positions"].shape == (10, 2, 3)
        assert f["velocities"].attrs["units"] == "bohr/au_time"
        assert f["sampled_modes"].dtype == np.int8 and f["q"].shape == (10, 1)
        assert f["q"].attrs["units"] == "sqrt(m_e)*bohr"
        assert f["p"].attrs["units"] == "sqrt(m_e)*bohr/au_time"
        assert f["masses"].attrs["units"] == "amu"
        assert f["equilibrium_coordinates"].attrs["units"] == "bohr"
        assert f["frequencies"].attrs["units"] == "cm^-1"
        for name in ("harmonic_energy", "kinetic_energy", "removed_kinetic_energy"):
            assert f[name].attrs["units"] == "hartree"
        np.testing.assert_allclose(f["harmonic_energy"][()], s.harmonic_energy_hartree)
        np.testing.assert_array_equal(f["sampled_modes"][()], s.sampled_modes.astype(np.int8))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["samples.h5"]


def test_format_samples_xyz():
    from gqteashmd.core.units import BOHR_IN_ANGSTROM
    from gqteashmd.vib.artifacts import format_samples_xyz

    pos = (
        np.array([[[0.0, 0.0, 0.2], [0.0, 1.4, -0.9], [0.0, -1.4, -0.9]]] * 2)
        + np.array([0.0, 0.1])[:, None, None]
    )
    text = format_samples_xyz(("O", "H", "H"), pos, np.array([0.01, 0.02]))
    lines = text.splitlines()
    assert len(lines) == 10
    assert lines[0].strip() == "3" and "sample 1" in lines[1] and "sample 2" in lines[6]
    assert "E_harm" in lines[1] and "Eh" in lines[1]
    assert float(lines[3].split()[2]) == pytest.approx(1.4 * BOHR_IN_ANGSTROM, abs=1e-9)
    assert float(lines[9].split()[3]) == pytest.approx(pos[1, 2, 2] * BOHR_IN_ANGSTROM, abs=1e-9)


def _write_h2(path):
    masses = np.array([1.007825, 1.007825])
    coords = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    e = np.array([0, 0, -1, 0, 0, 1]) / np.sqrt(2)
    hess = np.outer(e, e)
    write_frequencies_h5(
        path,
        symbols=("H", "H"),
        masses_amu=masses,
        coords_bohr=coords,
        hessian=hess,
        analysis=harmonic_analysis(masses, coords, hess),
    )


def test_read_frequencies_h5_missing_file_names_the_path_and_the_remedy(tmp_path):
    from gqteashmd.vib.artifacts import read_frequencies_h5

    path = tmp_path / "frequencies.h5"
    with pytest.raises(ValueError, match="--rerun-from frequencies") as info:
        read_frequencies_h5(path)
    assert str(path) in str(info.value)


def test_read_frequencies_h5_missing_dataset_and_bad_shape(tmp_path):
    from gqteashmd.vib.artifacts import read_frequencies_h5

    path = tmp_path / "frequencies.h5"
    _write_h2(path)
    with h5py.File(path, "r+") as f:
        del f["modes_mass_weighted"]
    with pytest.raises(ValueError, match="missing or unreadable"):
        read_frequencies_h5(path)
    _write_h2(path)
    with h5py.File(path, "r+") as f:
        del f["masses"]
        f.create_dataset("masses", data=np.ones(3))
    with pytest.raises(ValueError, match="inconsistent.*masses"):
        read_frequencies_h5(path)
    path.write_bytes(b"not an hdf5 file")
    with pytest.raises(ValueError, match="missing or unreadable"):
        read_frequencies_h5(path)


def test_read_samples_h5_round_trip_and_errors(tmp_path):
    from gqteashmd.vib.artifacts import read_samples_h5, write_samples_h5
    from gqteashmd.vib.wigner import sample_wigner

    masses = np.array([1.007825, 1.007825])
    coords = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    e = np.array([0, 0, -1, 0, 0, 1]) / np.sqrt(2)
    a = harmonic_analysis(masses, coords, np.outer(e, e))
    s = sample_wigner(
        masses,
        coords,
        a.frequencies_cm1,
        a.modes_mass_weighted,
        n_samples=4,
        temperature_K=0.0,
        low_frequency_cutoff_cm1=10.0,
        rng=np.random.default_rng(1),
    )
    path = tmp_path / "samples.h5"
    write_samples_h5(
        path,
        symbols=("H", "H"),
        masses_amu=masses,
        coords_bohr=coords,
        frequencies_cm1=a.frequencies_cm1,
        samples=s,
        temperature_K=300.0,
        low_frequency_cutoff_cm1=10.0,
        project_seed=42,
    )
    d = read_samples_h5(path)
    assert d["symbols"] == ("H", "H") and d["n_samples"] == 4 and d["temperature_K"] == 300.0
    np.testing.assert_allclose(d["masses_amu"], masses)
    np.testing.assert_allclose(d["equilibrium_bohr"], coords)
    np.testing.assert_allclose(d["positions_bohr"], s.positions_bohr)
    np.testing.assert_allclose(d["velocities_au"], s.velocities_au)
    with pytest.raises(ValueError, match="--rerun-from sampling"):
        read_samples_h5(tmp_path / "missing.h5")
    bad = tmp_path / "bad.h5"
    bad.write_bytes(b"not hdf5")
    with pytest.raises(ValueError, match="--rerun-from sampling"):
        read_samples_h5(bad)
