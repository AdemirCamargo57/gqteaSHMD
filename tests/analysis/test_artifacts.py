import h5py
import numpy as np

from gqteashmd.analysis.artifacts import (
    format_hops_csv,
    format_populations_csv,
    write_populations_h5,
)
from gqteashmd.analysis.hops import HopEvent

LABELS = ("S0", "S1")
T = np.array([0.0, 0.5])
P = np.array([[0.0, 1.0], [0.5, 0.5]])
ARRAYS = dict(
    time_fs=T,
    labels=LABELS,
    classical=P,
    classical_lower=P - 0.1,
    classical_upper=P + 0.1,
    quantum=P,
    quantum_lower=P,
    quantum_upper=P,
    fit_target=np.array([1.0, 0.5]),
    fit_curve=np.array([1.0, np.nan]),
)


def test_populations_h5_has_units_and_attributes(tmp_path):
    path = tmp_path / "populations.h5"
    write_populations_h5(path, **ARRAYS, attrs={"n_included": 2, "horizon_fs": 0.5})
    with h5py.File(path, "r") as f:
        assert f.attrs["schema_version"] == 1 and f.attrs["n_included"] == 2
        assert "gqteashmd_version" in f.attrs
        for name in f:
            assert "units" in f[name].attrs, name
        np.testing.assert_array_equal(f["classical"][()], P)
        assert f["time_fs"].attrs["units"] == "fs"
    assert not list(tmp_path.glob(".*tmp"))


def test_populations_csv():
    text = format_populations_csv(**ARRAYS, comments=["populations", "n_included = 2"])
    lines = text.splitlines()
    assert lines[0] == "# populations" and lines[1] == "# n_included = 2"
    header = lines[2].split(",")
    assert header[:4] == ["time_fs", "P_S0", "P_S0_lower", "P_S0_upper"]
    assert header[-2:] == ["fit_target", "fit_curve"] and "rho_S1_upper" in header
    assert lines[3].split(",")[0] == "0.000000e+00"
    assert lines[4].endswith(",")  # fit_curve NaN -> empty field


def test_hops_csv():
    text = format_hops_csv(
        [
            HopEvent("a", 1.0, 1, 0, False, True, False),
            HopEvent("a", 0.5, 1, None, True, True, False),
        ],
        LABELS,
    )
    assert text.splitlines() == [
        "task,time_fs,from,to,frustrated,included,after_horizon",
        "a,1.000000,S1,S0,0,1,0",
        "a,0.500000,S1,,1,1,0",
    ]
