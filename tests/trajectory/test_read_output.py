import h5py
import numpy as np
import pytest

from gqteashmd.trajectory.store import (
    OUTPUT_DATASETS,
    log_dtype,
    log_row,
    read_output_h5,
    write_output_h5,
)

ATTRS = {
    "dt_fs": 0.5,
    "outcome": "finished",
    "task": "traj_00000",
    "initial_state": "S1",
    "sample": 3,
    "symbols": ["H", "H"],
}


def _rows(n=5):
    dtype = log_dtype(3, 6)
    rows = np.stack(
        [
            log_row(
                dtype,
                step=k,
                grid_step=k,
                time_au=20.0 * k,
                active=1,
                hop_from=-1,
                hop_to=-1,
                e_total=-1.0 + 1e-6 * k,
                c_real=[0.0, 1.0, 0.0],
            )
            for k in range(n)
        ]
    )
    return rows


def test_round_trip(tmp_path):
    path = tmp_path / "output.h5"
    write_output_h5(
        path, _rows(), labels=["S0", "S1", "S2"], n_atoms=2, output_every=1, attrs=ATTRS
    )
    data = read_output_h5(path)
    assert set(OUTPUT_DATASETS) == set(data.data)
    assert data.labels == ("S0", "S1", "S2") and data.output_every == 1
    assert data.dt_fs == 0.5 and data.outcome == "finished" and data.task == "traj_00000"
    assert data.initial_state == "S1" and data.n_records == 5
    np.testing.assert_array_equal(data["grid_step"], np.arange(5))
    np.testing.assert_array_equal(data["populations"][:, 1], np.ones(5))
    assert data["positions"].shape == (5, 2, 3)


def test_missing_file_names_the_path(tmp_path):
    with pytest.raises(ValueError, match="output.h5"):
        read_output_h5(tmp_path / "output.h5")


def test_garbage_file_names_the_path(tmp_path):
    path = tmp_path / "output.h5"
    path.write_bytes(b"not hdf5")
    with pytest.raises(ValueError, match=str(path.name)):
        read_output_h5(path)


def test_missing_dataset_is_an_error(tmp_path):
    path = tmp_path / "output.h5"
    write_output_h5(
        path, _rows(), labels=["S0", "S1", "S2"], n_atoms=2, output_every=1, attrs=ATTRS
    )
    with h5py.File(path, "a") as f:
        del f["e_total"]
    with pytest.raises(ValueError, match="e_total"):
        read_output_h5(path)


def test_inconsistent_lengths_are_an_error(tmp_path):
    path = tmp_path / "output.h5"
    write_output_h5(
        path, _rows(), labels=["S0", "S1", "S2"], n_atoms=2, output_every=1, attrs=ATTRS
    )
    with h5py.File(path, "a") as f:
        del f["active"]
        f.create_dataset("active", data=np.zeros(4, dtype=int)).attrs["units"] = "-"
    with pytest.raises(ValueError, match="active"):
        read_output_h5(path)
