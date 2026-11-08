"""frequencies.h5 (HDF5, h5py imported lazily). Every dataset carries a 'units' attribute."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from gqteashmd import __version__
from gqteashmd.vib.harmonic import HarmonicAnalysis


def write_frequencies_h5(
    path: Path,
    *,
    symbols: Sequence[str],
    masses_amu: ArrayLike,
    coords_bohr: ArrayLike,
    hessian: ArrayLike,
    analysis: HarmonicAnalysis,
) -> None:
    import h5py

    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["schema_version"] = 1
            f.attrs["gqteashmd_version"] = __version__
            f.attrs["linear"] = bool(analysis.linear)
            f.attrs["n_external"] = int(analysis.n_external)
            f.create_dataset("symbols", data=np.array(list(symbols), dtype=h5py.string_dtype()))
            for name, data, units in (
                ("masses", masses_amu, "amu"),
                ("coordinates", coords_bohr, "bohr"),
                ("hessian", hessian, "hartree/bohr^2"),
                ("frequencies", analysis.frequencies_cm1, "cm^-1"),
                ("modes_mass_weighted", analysis.modes_mass_weighted, "1"),
                ("modes_cartesian", analysis.modes_cartesian, "1"),
                ("reduced_masses", analysis.reduced_masses_amu, "amu"),
            ):
                f.create_dataset(name, data=np.asarray(data, dtype=float)).attrs["units"] = units
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
