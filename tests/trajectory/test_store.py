"""Files of one trajectory (spec section 6)."""

from pathlib import Path

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


# ---------------------------------------------------------------- triplets (spec 4.1-4.2, PR1)

TODAY = [
    ("step", "<i8", ()),
    ("grid_step", "<i8", ()),
    ("ticks", "<i8", ()),
    ("depth", "<i8", ()),
    ("time_au", "<f8", ()),
    ("active", "<i8", ()),
    ("hop_from", "<i8", ()),
    ("hop_to", "<i8", ()),
    ("frustrated", "i1", ()),
    ("rung_changed", "i1", ()),
    ("rejections", "<i8", ()),
    ("random_number", "<f8", ()),
    ("min_singular_value", "<f8", ()),
    ("e_total", "<f8", ()),
    ("e_kinetic", "<f8", ()),
    ("energies", "<f8", (4,)),
    ("c_real", "<f8", (4,)),
    ("c_imag", "<f8", (4,)),
    ("probabilities", "<f8", (4,)),
    ("q", "<f8", (6,)),
    ("v", "<f8", (6,)),
]


def _fields(dtype):
    return [(name, dtype[name].base.str, dtype[name].shape) for name in dtype.names]


def test_the_singlet_log_dtype_is_unchanged():
    dtype = log_dtype(4, 6)
    assert _fields(dtype) == [(n, np.dtype(t).str, s) for n, t, s in TODAY]
    assert log_dtype(4, 6, n_spin_free=None) == dtype


def test_the_triplet_log_dtype_adds_four_columns_at_the_end():
    dtype = log_dtype(8, 6, n_spin_free=4)
    expected = [(n, np.dtype(t).str, (8,) if s == (4,) else s) for n, t, s in TODAY]
    expected += [
        ("energies_diag", "<f8", (8,)),
        ("active_mch", "<i8", ()),
        ("energies_sf", "<f8", (4,)),
        ("populations_mch", "<f8", (4,)),
    ]
    assert _fields(dtype) == expected
    log = StepLog(Path("unused.log"), 8, 6, n_spin_free=4)
    assert log.dtype == dtype


def _soc_row(dtype, step, *, active_mch, hop_from=-1, hop_to=-1, frustrated=0):
    c = np.zeros(8)
    c[step % 8] = 1.0
    return log_row(
        dtype,
        step=step,
        grid_step=step,
        time_au=10.0 * step,
        active=step % 8,
        hop_from=hop_from,
        hop_to=hop_to,
        frustrated=frustrated,
        energies=np.arange(8.0),
        c_real=c,
        q=np.arange(6.0) + step,
        v=np.ones(6),
        energies_diag=np.arange(8.0) + 0.5,
        active_mch=active_mch,
        energies_sf=[0.0, 0.1, 0.2, 0.3],
        populations_mch=[0.1, 0.2, 0.3, 0.4],
    )


def test_output_h5_schema_2_with_triplets(tmp_path):
    h5py = pytest.importorskip("h5py")
    dtype = log_dtype(8, 6, n_spin_free=4)
    rows = np.stack(
        [
            _soc_row(dtype, 0, active_mch=1),
            _soc_row(dtype, 1, active_mch=3),  # not kept (output_every = 2)
            _soc_row(dtype, 2, active_mch=2, hop_from=4, hop_to=3),  # T2 -> T1: from row 1
            _soc_row(dtype, 3, active_mch=2),  # not kept
            _soc_row(dtype, 4, active_mch=2),
            _soc_row(dtype, 5, active_mch=0, hop_from=3, hop_to=0),  # T1 -> S0: from row 4
            _soc_row(dtype, 6, active_mch=0, frustrated=1),
            _soc_row(dtype, 7, active_mch=0),
        ]
    )
    path = tmp_path / "output.h5"
    write_output_h5(
        path,
        rows,
        labels=["S0", "S1", "T1", "T2"],
        n_atoms=2,
        output_every=2,
        attrs={"outcome": "finished", "soc_operator": "somf"},
        n_singlets=2,
        n_triplets=2,
    )
    with h5py.File(path, "r") as f:
        assert f.attrs["schema_version"] == 2
        assert (f.attrs["n_singlets"], f.attrs["n_triplets"]) == (2, 2)
        assert f.attrs["soc_operator"] == "somf"
        assert list(f.attrs["state_labels"]) == ["S0", "S1", "T1", "T2"]
        index = [0, 2, 4, 5, 6, 7]
        assert list(f["step"][()]) == index
        kept = rows[index]
        np.testing.assert_array_equal(f["active"][()], kept["active_mch"])
        np.testing.assert_array_equal(f["populations"][()], kept["populations_mch"])
        np.testing.assert_array_equal(f["energies"][()], kept["energies_sf"])
        # spin-free hops (PR1): the active spin-free state of the previous log row (kept or
        # not) and of this one
        assert list(f["hop_from"][()]) == [-1, 3, -1, 2, -1, -1]
        assert list(f["hop_to"][()]) == [-1, 2, -1, 0, -1, -1]
        assert list(f["hop_from_diag"][()]) == [-1, 4, -1, 3, -1, -1]
        assert list(f["hop_to_diag"][()]) == [-1, 3, -1, 0, -1, -1]
        np.testing.assert_array_equal(f["active_diag"][()], kept["active"])
        np.testing.assert_array_equal(f["energies_diag"][()], kept["energies_diag"])
        np.testing.assert_array_equal(
            f["populations_diag"][()], kept["c_real"] ** 2 + kept["c_imag"] ** 2
        )
        assert f["populations_diag"].shape == (6, 8) and f["populations"].shape == (6, 4)
        assert f["coefficients_real"].shape == (6, 8) and f["frustrated"][4] == 1
        for name, dataset in f.items():
            assert "units" in dataset.attrs, name


def test_output_h5_without_triplets_keeps_schema_1(tmp_path):
    h5py = pytest.importorskip("h5py")
    dtype = log_dtype(2, 6)
    rows = np.stack([_row(dtype, 0), _row(dtype, 1, hop_to=1)])
    for counts in ({}, {"n_singlets": 2, "n_triplets": 0}):
        path = tmp_path / f"o{len(counts)}.h5"
        write_output_h5(
            path, rows, labels=["S0", "S1"], n_atoms=2, output_every=1, attrs={}, **counts
        )
        with h5py.File(path, "r") as f:
            assert f.attrs["schema_version"] == 1
            assert "n_triplets" not in f.attrs and "active_diag" not in f
