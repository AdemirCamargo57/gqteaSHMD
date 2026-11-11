"""Files of one trajectory (spec section 6, rulings R2 and R3).

steps.log: append-only fixed-size binary records, one per accepted (sub)step and one for t = 0;
the restart's meta.json stores how many records belong to it, so a resume truncates the records
written after the last restart (and any record cut by a kill). restart/: generations gen_<n>,
each written into a temporary directory and renamed; the small file `current` (written
atomically) names the newest complete one; older ones are deleted only after `current` points
to the new one, so a kill at any instant leaves one complete generation. output.h5 is built from
the log when a trajectory ends (h5py imported inside the function).
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.core.units import AU_TIME_IN_FS
from gqteashmd.util.fileio import write_text_atomic

_GENERATION = re.compile(r"gen_(\d{8})")


def log_dtype(n_states: int, n_dof: int) -> np.dtype:
    n, f = int(n_states), int(n_dof)
    return np.dtype(
        [
            ("step", "<i8"),
            ("grid_step", "<i8"),
            ("ticks", "<i8"),
            ("depth", "<i8"),
            ("time_au", "<f8"),
            ("active", "<i8"),
            ("hop_from", "<i8"),
            ("hop_to", "<i8"),
            ("frustrated", "i1"),
            ("rung_changed", "i1"),
            ("rejections", "<i8"),
            ("random_number", "<f8"),
            ("min_singular_value", "<f8"),
            ("e_total", "<f8"),
            ("e_kinetic", "<f8"),
            ("energies", "<f8", (n,)),
            ("c_real", "<f8", (n,)),
            ("c_imag", "<f8", (n,)),
            ("probabilities", "<f8", (n,)),
            ("q", "<f8", (f,)),
            ("v", "<f8", (f,)),
        ]
    )


def log_row(dtype: np.dtype, **values: Any) -> np.ndarray:
    row = np.zeros((), dtype=dtype)
    for name, value in values.items():
        row[name] = value
    return row


class StepLog:
    def __init__(self, path: Path, n_states: int, n_dof: int) -> None:
        self.path = Path(path)
        self.dtype = log_dtype(n_states, n_dof)

    def __len__(self) -> int:
        if not self.path.exists():
            return 0
        return self.path.stat().st_size // self.dtype.itemsize

    def append(self, row: np.ndarray) -> None:
        data = np.asarray(row, dtype=self.dtype).reshape(1).tobytes()
        with open(self.path, "ab") as fh:
            size = os.fstat(fh.fileno()).st_size
            if size % self.dtype.itemsize:
                raise ValueError(
                    f"{self.path} ends with a cut record ({size} bytes is not a whole number "
                    f"of {self.dtype.itemsize}-byte records); truncate the log to the restart's "
                    "record count before appending"
                )
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())

    def truncate(self, n_records: int) -> None:
        have = len(self)
        if n_records > have:
            raise ValueError(
                f"{self.path} holds {have} complete records, but the restart expects {n_records}"
            )
        with open(self.path, "ab"):
            pass
        os.truncate(self.path, n_records * self.dtype.itemsize)

    def read(self) -> np.ndarray:
        n = len(self)
        if n == 0:
            return np.zeros(0, dtype=self.dtype)
        return np.fromfile(self.path, dtype=self.dtype, count=n)


class RestartStore:
    CURRENT = "current"

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def current(self) -> Path | None:
        marker = self.directory / self.CURRENT
        if not marker.exists():
            return None
        name = marker.read_text(encoding="utf-8").strip()
        path = self.directory / name
        if not _GENERATION.fullmatch(name) or not path.is_dir():
            raise ValueError(f"{marker} names {name!r}, which is not a complete restart generation")
        return path

    def write(self, fill: Callable[[Path], None]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        current = self.current()
        number = 0 if current is None else int(current.name[4:]) + 1
        name = f"gen_{number:08d}"
        tmp = self.directory / f".{name}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        fill(tmp)
        final = self.directory / name
        shutil.rmtree(final, ignore_errors=True)  # renamed by a run killed before `current`
        os.replace(tmp, final)
        write_text_atomic(self.directory / self.CURRENT, name + "\n")
        for old in self.directory.glob("gen_*"):
            if old.name != name:
                shutil.rmtree(old, ignore_errors=True)
        return final

    def clean_temporary(self) -> None:
        for tmp in self.directory.glob(".gen_*.tmp"):
            shutil.rmtree(tmp, ignore_errors=True)


def _selected(rows: np.ndarray, output_every: int) -> np.ndarray:
    keep = (rows["ticks"] == 0) & (rows["grid_step"] % output_every == 0)
    keep |= rows["hop_to"] >= 0
    keep |= rows["frustrated"] != 0
    keep |= rows["rung_changed"] != 0
    keep |= rows["rejections"] > 0
    if rows.size:
        keep[-1] = True
    selected: np.ndarray = rows[keep]
    return selected


def write_output_h5(
    path: Path,
    rows: np.ndarray,
    *,
    labels: Sequence[str],
    n_atoms: int,
    output_every: int,
    attrs: Mapping[str, Any],
) -> None:
    import h5py

    path = Path(path)
    kept = _selected(rows, output_every)
    m = kept.shape[0]
    populations = kept["c_real"] ** 2 + kept["c_imag"] ** 2
    datasets: list[tuple[str, np.ndarray, str]] = [
        ("time_fs", kept["time_au"] * AU_TIME_IN_FS, "fs"),
        ("step", kept["step"], "-"),
        ("grid_step", kept["grid_step"], "-"),
        ("ticks", kept["ticks"], "-"),
        ("depth", kept["depth"], "-"),
        ("active", kept["active"], "state index (0 = S0)"),
        ("hop_from", kept["hop_from"], "state index (-1 = no hop)"),
        ("hop_to", kept["hop_to"], "state index (-1 = no hop)"),
        ("frustrated", kept["frustrated"], "bool"),
        ("rung_changed", kept["rung_changed"], "bool"),
        ("rejections", kept["rejections"], "-"),
        ("random_number", kept["random_number"], "-"),
        ("min_singular_value", kept["min_singular_value"], "-"),
        ("energies", kept["energies"], "Eh"),
        ("e_total", kept["e_total"], "Eh"),
        ("e_kinetic", kept["e_kinetic"], "Eh"),
        ("populations", populations, "-"),
        ("coefficients_real", kept["c_real"], "-"),
        ("coefficients_imag", kept["c_imag"], "-"),
        ("probabilities", kept["probabilities"], "-"),
        ("positions", kept["q"].reshape(m, n_atoms, 3), "bohr"),
        ("velocities", kept["v"].reshape(m, n_atoms, 3), "bohr per atomic unit of time"),
    ]
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["schema_version"] = 1
            f.attrs["state_labels"] = [str(label) for label in labels]
            f.attrs["output_every"] = int(output_every)
            for key, value in attrs.items():
                f.attrs[key] = value
            for name, data, units in datasets:
                f.create_dataset(name, data=np.asarray(data)).attrs["units"] = units
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
