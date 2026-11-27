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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.core.units import AU_TIME_IN_FS
from gqteashmd.util.fileio import write_text_atomic

_GENERATION = re.compile(r"gen_(\d{8})")


def log_dtype(n_states: int, n_dof: int, *, n_spin_free: int | None = None) -> np.dtype:
    """The record of steps.log; ``n_states`` columns in the core's (diagonal) basis.

    With triplets (``n_spin_free`` = n + m given; spec 4.1) four MCH columns follow at the end:
    ``energies_diag`` (n_states), ``active_mch`` (the active spin-free state), ``energies_sf``
    and ``populations_mch`` (n + m each). Without it the layout is unchanged.
    """
    n, f = int(n_states), int(n_dof)
    mch: list[tuple[Any, ...]] = []
    if n_spin_free is not None:
        k = int(n_spin_free)
        mch = [
            ("energies_diag", "<f8", (n,)),
            ("active_mch", "<i8"),
            ("energies_sf", "<f8", (k,)),
            ("populations_mch", "<f8", (k,)),
        ]
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
            *mch,
        ]
    )


def log_row(dtype: np.dtype, **values: Any) -> np.ndarray:
    row = np.zeros((), dtype=dtype)
    for name, value in values.items():
        row[name] = value
    return row


class StepLog:
    def __init__(
        self, path: Path, n_states: int, n_dof: int, *, n_spin_free: int | None = None
    ) -> None:
        self.path = Path(path)
        self.dtype = log_dtype(n_states, n_dof, n_spin_free=n_spin_free)

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


def _kept(rows: np.ndarray, output_every: int) -> np.ndarray:
    """The boolean mask of the log rows that output.h5 keeps (ruling R2)."""
    keep = (rows["ticks"] == 0) & (rows["grid_step"] % output_every == 0)
    keep |= rows["hop_to"] >= 0
    keep |= rows["frustrated"] != 0
    keep |= rows["rung_changed"] != 0
    keep |= rows["rejections"] > 0
    if rows.size:
        keep[-1] = True
    mask: np.ndarray = keep
    return mask


def spin_free_hops(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spin-free ``hop_from``/``hop_to`` of every log row (plan ruling PR1).

    At an accepted hop (``hop_to >= 0``) they are the active spin-free state (``active_mch``)
    of the previous log row, i.e. the previous accepted (sub)step whether output.h5 keeps it
    or not, and of this row; -1 elsewhere.
    """
    active = np.asarray(rows["active_mch"], dtype=np.int64)
    hop_from = np.full(active.shape, -1, dtype=np.int64)
    hop_to = np.full(active.shape, -1, dtype=np.int64)
    hops = np.flatnonzero(rows["hop_to"] >= 0)
    hops = hops[hops > 0]  # row 0 is t = 0, never a hop
    hop_from[hops] = active[hops - 1]
    hop_to[hops] = active[hops]
    return hop_from, hop_to


def write_output_h5(
    path: Path,
    rows: np.ndarray,
    *,
    labels: Sequence[str],
    n_atoms: int,
    output_every: int,
    attrs: Mapping[str, Any],
    n_singlets: int | None = None,
    n_triplets: int | None = None,
) -> None:
    """output.h5 from the step log; ``schema_version`` 1, or 2 with triplets (spec 4.2).

    With ``n_triplets > 0`` the log must have the MCH columns (``log_dtype(n_spin_free=)``):
    ``active``, ``populations``, ``energies``, ``hop_from``/``hop_to`` are then spin-free
    (PR1), the diagonal ones are ``active_diag``, ``populations_diag``, ``energies_diag``,
    ``hop_from_diag``/``hop_to_diag``, and the attributes ``n_singlets``, ``n_triplets`` are
    added. ``coefficients_*`` and ``probabilities`` stay in the diagonal basis.
    """
    import h5py

    path = Path(path)
    triplets = bool(n_triplets)
    if triplets and not n_singlets:
        raise ValueError("write_output_h5 with n_triplets > 0 needs n_singlets >= 1")
    keep = _kept(rows, output_every)
    kept = rows[keep]
    m = kept.shape[0]
    populations = kept["c_real"] ** 2 + kept["c_imag"] ** 2
    if triplets:
        hop_from_sf, hop_to_sf = (a[keep] for a in spin_free_hops(rows))
        index = "spin-free state index"
        active = ("active", kept["active_mch"], f"{index} (0 = S0)")
        hop_from = ("hop_from", hop_from_sf, f"{index} (-1 = no hop)")
        hop_to = ("hop_to", hop_to_sf, f"{index} (-1 = no hop)")
        energies = ("energies", kept["energies_sf"], "Eh")
        spin_free_populations = ("populations", kept["populations_mch"], "-")
    else:
        active = ("active", kept["active"], "state index (0 = S0)")
        hop_from = ("hop_from", kept["hop_from"], "state index (-1 = no hop)")
        hop_to = ("hop_to", kept["hop_to"], "state index (-1 = no hop)")
        energies = ("energies", kept["energies"], "Eh")
        spin_free_populations = ("populations", populations, "-")
    datasets: list[tuple[str, np.ndarray, str]] = [
        ("time_fs", kept["time_au"] * AU_TIME_IN_FS, "fs"),
        ("step", kept["step"], "-"),
        ("grid_step", kept["grid_step"], "-"),
        ("ticks", kept["ticks"], "-"),
        ("depth", kept["depth"], "-"),
        active,
        hop_from,
        hop_to,
        ("frustrated", kept["frustrated"], "bool"),
        ("rung_changed", kept["rung_changed"], "bool"),
        ("rejections", kept["rejections"], "-"),
        ("random_number", kept["random_number"], "-"),
        ("min_singular_value", kept["min_singular_value"], "-"),
        energies,
        ("e_total", kept["e_total"], "Eh"),
        ("e_kinetic", kept["e_kinetic"], "Eh"),
        spin_free_populations,
        ("coefficients_real", kept["c_real"], "-"),
        ("coefficients_imag", kept["c_imag"], "-"),
        ("probabilities", kept["probabilities"], "-"),
        ("positions", kept["q"].reshape(m, n_atoms, 3), "bohr"),
        ("velocities", kept["v"].reshape(m, n_atoms, 3), "bohr per atomic unit of time"),
    ]
    if triplets:
        diagonal = "diagonal state index"
        datasets += [
            ("active_diag", kept["active"], f"{diagonal} (0 = lowest)"),
            ("populations_diag", populations, "-"),
            ("energies_diag", kept["energies_diag"], "Eh"),
            ("hop_from_diag", kept["hop_from"], f"{diagonal} (-1 = no hop)"),
            ("hop_to_diag", kept["hop_to"], f"{diagonal} (-1 = no hop)"),
        ]
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["schema_version"] = 2 if triplets else 1
            f.attrs["state_labels"] = [str(label) for label in labels]
            f.attrs["output_every"] = int(output_every)
            if triplets:
                f.attrs["n_singlets"] = int(n_singlets or 0)
                f.attrs["n_triplets"] = int(n_triplets or 0)
            for key, value in attrs.items():
                f.attrs[key] = value
            for name, data, units in datasets:
                f.create_dataset(name, data=np.asarray(data)).attrs["units"] = units
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


OUTPUT_DATASETS = (
    "time_fs",
    "step",
    "grid_step",
    "ticks",
    "depth",
    "active",
    "hop_from",
    "hop_to",
    "frustrated",
    "rung_changed",
    "rejections",
    "random_number",
    "min_singular_value",
    "energies",
    "e_total",
    "e_kinetic",
    "populations",
    "coefficients_real",
    "coefficients_imag",
    "probabilities",
    "positions",
    "velocities",
)
_REQUIRED_ATTRS = ("state_labels", "output_every", "dt_fs", "outcome", "task", "initial_state")


def _plain(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class TrajectoryData:
    """The contents of one output.h5 (USER_MANUAL section 6)."""

    path: Path
    attrs: Mapping[str, Any]
    data: Mapping[str, np.ndarray]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.data[name]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(str(x) for x in self.attrs["state_labels"])

    @property
    def output_every(self) -> int:
        return int(self.attrs["output_every"])

    @property
    def dt_fs(self) -> float:
        return float(self.attrs["dt_fs"])

    @property
    def outcome(self) -> str:
        return str(self.attrs["outcome"])

    @property
    def task(self) -> str:
        return str(self.attrs["task"])

    @property
    def initial_state(self) -> str:
        return str(self.attrs["initial_state"])

    @property
    def n_records(self) -> int:
        return int(self.data["time_fs"].shape[0])


def read_output_h5(path: Path) -> TrajectoryData:
    """Read one trajectory's output.h5; ValueError naming the file for any problem."""
    import h5py

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path} does not exist")
    try:
        with h5py.File(path, "r") as f:
            missing = [n for n in OUTPUT_DATASETS if n not in f]
            missing += [a for a in _REQUIRED_ATTRS if a not in f.attrs]
            if missing:
                raise ValueError(f"{path} lacks {', '.join(missing)}")
            data = {n: np.asarray(f[n][()]) for n in OUTPUT_DATASETS}
            attrs = {k: _plain(v) for k, v in f.attrs.items()}
    except OSError as exc:
        raise ValueError(f"{path} is not a readable HDF5 file: {exc}") from exc
    lengths = {n: a.shape[0] if a.ndim else -1 for n, a in data.items()}
    m = lengths["time_fs"]
    wrong = sorted(n for n, k in lengths.items() if k != m)
    if wrong:
        raise ValueError(f"{path}: datasets {', '.join(wrong)} do not have {m} records")
    return TrajectoryData(path, attrs, data)
