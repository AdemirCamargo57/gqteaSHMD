import h5py
import numpy as np
import pytest

from gqteashmd.core.units import HARTREE_IN_EV
from gqteashmd.excite.artifacts import (
    format_spectrum_csv,
    read_excitations_h5,
    write_excitations_h5,
)
from gqteashmd.excite.spectrum import broadened_spectrum

LABELS = ("S0", "S1", "S2")


def _data(n=3):
    rng = np.random.default_rng(0)
    e = -1.0 + np.cumsum(rng.random((n, 3)) * 0.1, axis=1)
    e[:, 0] = -1.0
    mu = rng.random((n, 2, 3))
    f = rng.random((n, 2))
    ok = np.ones(n, bool)
    ok[1] = False
    e[1], mu[1], f[1] = np.nan, np.nan, np.nan
    return e, mu, f, ok


def _write(path, reference=None):
    e, mu, f, ok = _data()
    write_excitations_h5(
        path,
        symbols=("O", "H"),
        state_labels=LABELS,
        sample_indices=np.array([1, 2, 3]),
        ok=ok,
        energies_hartree=e,
        transition_dipoles_au=mu,
        oscillator_strengths=f,
        reference=reference,
        project_seed=7,
    )
    return e, mu, f, ok


def test_round_trip_with_failed_sample_and_reference(tmp_path):
    ref = {
        "energies_hartree": np.array([[-1.0, -0.9, -0.8]]),
        "transition_dipoles_au": np.ones((1, 2, 3)),
        "oscillator_strengths": np.array([[0.1, 0.2]]),
        "coordinates_bohr": np.zeros((2, 3)),
    }
    path = tmp_path / "excitations.h5"
    e, mu, f, ok = _write(path, ref)
    d = read_excitations_h5(path)
    assert d["symbols"] == ("O", "H") and d["state_labels"] == LABELS
    np.testing.assert_array_equal(d["sample_index"], [1, 2, 3])
    np.testing.assert_array_equal(d["ok"], ok)
    np.testing.assert_allclose(d["energies_hartree"], e)
    np.testing.assert_allclose(d["excitation_energies_eV"], (e[:, 1:] - e[:, :1]) * HARTREE_IN_EV)
    assert np.isnan(d["excitation_energies_eV"][1]).all()
    np.testing.assert_allclose(d["transition_dipoles_au"], mu)
    np.testing.assert_allclose(d["oscillator_strengths"], f)
    r = d["reference"]
    np.testing.assert_allclose(r["energies_hartree"], ref["energies_hartree"])
    np.testing.assert_allclose(
        r["excitation_energies_eV"], [[0.1 * HARTREE_IN_EV, 0.2 * HARTREE_IN_EV]]
    )
    np.testing.assert_allclose(r["coordinates_bohr"], np.zeros((2, 3)))
    with h5py.File(path, "r") as h:
        assert h.attrs["schema_version"] == 1 and h.attrs["n_samples"] == 3
        assert h.attrs["n_ok"] == 2 and "gqteashmd_version" in h.attrs
        assert h["energies"].attrs["units"] == "hartree"
        assert h["excitation_energies"].attrs["units"] == "eV"
        assert h["transition_dipoles"].attrs["units"] == "e*a0"
        assert h["oscillator_strengths"].attrs["units"] == "-"
        assert h["reference/coordinates"].attrs["units"] == "bohr"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["excitations.h5"]


def test_reference_absent(tmp_path):
    path = tmp_path / "excitations.h5"
    _write(path)
    assert read_excitations_h5(path)["reference"] is None
    with h5py.File(path, "r") as h:
        assert "reference" not in h


def test_read_errors_name_rerun(tmp_path):
    with pytest.raises(ValueError, match="--rerun-from excitations"):
        read_excitations_h5(tmp_path / "missing.h5")
    bad = tmp_path / "bad.h5"
    bad.write_bytes(b"junk")
    with pytest.raises(ValueError, match="--rerun-from excitations"):
        read_excitations_h5(bad)


def test_format_spectrum_csv():
    de = np.array([[5.0, 7.0], [5.2, 7.1]])
    f = np.array([[0.1, 0.2], [0.1, 0.3]])
    spec = broadened_spectrum(de, f, fwhm_eV=0.3, step_eV=0.05)
    text = format_spectrum_csv(spec, ("S1", "S2"), n_ok=2, fwhm_eV=0.3, step_eV=0.05)
    lines = text.splitlines()
    header = [x for x in lines if x.startswith("#")]
    assert any("n_ok" in x and "2" in x for x in header)
    assert any("fwhm" in x for x in header) and any("step" in x for x in header)
    body = [x for x in lines if not x.startswith("#")]
    assert body[0] == "energy_eV,cross_section_A2,epsilon_M-1cm-1,S1_A2,S2_A2"
    assert len(body) == 1 + len(spec.energies_eV)
    assert all(len(r.split(",")) == 5 for r in body)
    assert body[1].split(",")[0] == f"{spec.energies_eV[0]:.6e}"
