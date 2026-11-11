"""Files of one trajectory (spec section 6)."""

import numpy as np
import pytest

from gqteashmd.trajectory.store import RestartStore, StepLog, log_dtype, log_row, write_output_h5


def _row(dtype, step, *, grid_step=None, ticks=0, hop_to=-1, rejections=0):
    return log_row(
        dtype,
        step=step,
        grid_step=step if grid_step is None else grid_step,
        ticks=ticks,
        time_au=10.0 * step,
        hop_from=-1,
        hop_to=hop_to,
        rejections=rejections,
        energies=[0.0, 0.1],
        c_real=[1.0, 0.0],
        q=np.arange(6.0) + step,
        v=np.ones(6),
    )


def test_log_append_read_and_truncate(tmp_path):
    log = StepLog(tmp_path / "steps.log", 2, 6)
    assert len(log) == 0
    for k in range(4):
        log.append(_row(log.dtype, k))
    assert len(log) == 4
    rows = log.read()
    assert list(rows["step"]) == [0, 1, 2, 3]
    np.testing.assert_array_equal(rows["q"][2], np.arange(6.0) + 2)
    with open(log.path, "ab") as fh:  # a record cut by a kill
        fh.write(b"\x01" * 5)
    assert len(log) == 4
    log.truncate(3)
    assert len(log) == 3 and log.path.stat().st_size == 3 * log.dtype.itemsize
    with pytest.raises(ValueError, match="records"):
        log.truncate(4)


def test_append_refuses_a_log_with_a_cut_record(tmp_path):
    log = StepLog(tmp_path / "steps.log", 2, 6)
    for k in range(2):
        log.append(_row(log.dtype, k))
    with open(log.path, "ab") as fh:  # a record cut by a kill
        fh.write(b"\x01" * 5)
    size = log.path.stat().st_size
    with pytest.raises(ValueError, match="truncate"):
        log.append(_row(log.dtype, 2))
    assert log.path.stat().st_size == size
    log.truncate(2)
    log.append(_row(log.dtype, 2))
    assert len(log.read()) == 3
    assert list(log.read()["step"]) == [0, 1, 2]


def test_dtype_fields():
    dtype = log_dtype(3, 9)
    assert dtype["energies"].shape == (3,) and dtype["q"].shape == (9,)
    assert dtype.names[:4] == ("step", "grid_step", "ticks", "depth")


def test_restart_generations(tmp_path):
    store = RestartStore(tmp_path / "restart")
    assert store.current() is None
    first = store.write(lambda d: (d / "a.txt").write_text("1"))
    second = store.write(lambda d: (d / "a.txt").write_text("2"))
    assert first.name == "gen_00000000" and second.name == "gen_00000001"
    assert store.current() == second and not first.exists()
    assert (second / "a.txt").read_text() == "2"


def test_a_kill_inside_a_generation_keeps_the_previous_one(tmp_path):
    store = RestartStore(tmp_path / "restart")
    good = store.write(lambda d: (d / "a.txt").write_text("1"))

    def broken(directory):
        (directory / "a.txt").write_text("half")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        store.write(broken)
    assert store.current() == good and (good / "a.txt").read_text() == "1"
    assert list(store.directory.glob(".gen_*.tmp"))
    store.clean_temporary()
    assert not list(store.directory.glob(".gen_*.tmp"))
    (store.directory / "gen_00000001").mkdir()  # renamed, but `current` was never updated
    again = store.write(lambda d: (d / "a.txt").write_text("3"))
    assert again.name == "gen_00000001" and (again / "a.txt").read_text() == "3"


def test_a_bad_current_marker_is_an_error(tmp_path):
    store = RestartStore(tmp_path / "restart")
    store.directory.mkdir()
    (store.directory / "current").write_text("gen_00000009\n")
    with pytest.raises(ValueError, match="restart generation"):
        store.current()


def test_output_h5_selection_and_units(tmp_path):
    h5py = pytest.importorskip("h5py")
    dtype = log_dtype(2, 6)
    rows = np.stack(
        [
            _row(dtype, 0),
            _row(dtype, 1),
            _row(dtype, 2, hop_to=1),
            _row(dtype, 3, grid_step=2, ticks=4, rejections=1),
            _row(dtype, 4, grid_step=3),
            _row(dtype, 5, grid_step=4),
            _row(dtype, 6, grid_step=5),
        ]
    )
    path = tmp_path / "output.h5"
    write_output_h5(
        path, rows, labels=["S0", "S1"], n_atoms=2, output_every=2, attrs={"outcome": "finished"}
    )
    with h5py.File(path, "r") as f:
        # grid points 0, 2, 4 kept; the hop (step 2) and the rejected interval (step 3) kept;
        # the last record (step 6) kept
        assert list(f["step"][()]) == [0, 2, 3, 5, 6]
        assert f["positions"].shape == (5, 2, 3)
        for name, dataset in f.items():
            assert "units" in dataset.attrs, name
        assert f["time_fs"].attrs["units"] == "fs"
        assert f.attrs["outcome"] == "finished"
        assert list(f.attrs["state_labels"]) == ["S0", "S1"]
        np.testing.assert_allclose(f["populations"][0], [1.0, 0.0])


def test_output_h5_of_an_empty_log(tmp_path):
    h5py = pytest.importorskip("h5py")
    write_output_h5(
        tmp_path / "o.h5",
        np.zeros(0, log_dtype(2, 6)),
        labels=["S0", "S1"],
        n_atoms=2,
        output_every=1,
        attrs={},
    )
    with h5py.File(tmp_path / "o.h5", "r") as f:
        assert f["step"].shape == (0,) and f["positions"].shape == (0, 2, 3)
