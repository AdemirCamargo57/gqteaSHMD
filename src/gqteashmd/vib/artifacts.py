"""frequencies.h5 and samples.h5 (HDF5, h5py imported lazily).

Every dataset carries a 'units' attribute.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from gqteashmd import __version__
from gqteashmd.chem.xyz import Geometry, format_xyz
from gqteashmd.core.units import BOHR_IN_ANGSTROM
from gqteashmd.vib.harmonic import HarmonicAnalysis
from gqteashmd.vib.wigner import WignerSamples


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


def read_frequencies_h5(path: Path) -> dict[str, Any]:
    """Read the parts of ``frequencies.h5`` that the sampler needs.

    A missing, unreadable or inconsistent file raises ``ValueError`` whose message names the
    path and says how to recompute it.
    """
    import h5py

    path = Path(path)
    advice = f"; rerun the frequencies stage with --rerun-from frequencies ({path})"
    try:
        with h5py.File(path, "r") as f:
            symbols = tuple(
                s.decode() if isinstance(s, bytes) else str(s) for s in f["symbols"][()]
            )
            data: dict[str, Any] = {
                "symbols": symbols,
                "masses_amu": np.asarray(f["masses"][()], dtype=float),
                "coords_bohr": np.asarray(f["coordinates"][()], dtype=float),
                "frequencies_cm1": np.asarray(f["frequencies"][()], dtype=float),
                "modes_mass_weighted": np.asarray(f["modes_mass_weighted"][()], dtype=float),
                "linear": bool(f.attrs["linear"]),
                "n_external": int(f.attrs["n_external"]),
            }
    except (OSError, KeyError) as exc:
        raise ValueError(f"frequencies.h5 is missing or unreadable ({exc!r}){advice}") from exc
    n = len(data["symbols"])
    k = data["frequencies_cm1"].size
    expected = {
        "masses": (data["masses_amu"].shape, (n,)),
        "coordinates": (data["coords_bohr"].shape, (n, 3)),
        "frequencies": (data["frequencies_cm1"].ndim, 1),
        "modes_mass_weighted": (data["modes_mass_weighted"].shape, (3 * n, k)),
    }
    for name, (got, want) in expected.items():
        if got != want:
            raise ValueError(
                f"frequencies.h5 is inconsistent: dataset '{name}' has shape/rank {got}, "
                f"expected {want}{advice}"
            )
    return data


def write_samples_h5(
    path: Path,
    *,
    symbols: Sequence[str],
    masses_amu: ArrayLike,
    coords_bohr: ArrayLike,
    frequencies_cm1: ArrayLike,
    samples: WignerSamples,
    temperature_K: float,
    low_frequency_cutoff_cm1: float,
    project_seed: int,
) -> None:
    import h5py

    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["schema_version"] = 1
            f.attrs["gqteashmd_version"] = __version__
            f.attrs["temperature_K"] = float(temperature_K)
            f.attrs["low_frequency_cutoff_cm1"] = float(low_frequency_cutoff_cm1)
            f.attrs["project_seed"] = int(project_seed)
            f.attrs["n_samples"] = int(samples.positions_bohr.shape[0])
            f.attrs["expected_mean_energy_hartree"] = float(samples.expected_mean_energy_hartree)
            f.create_dataset("symbols", data=np.array(list(symbols), dtype=h5py.string_dtype()))
            f["symbols"].attrs["units"] = "-"
            f.create_dataset(
                "sampled_modes", data=np.asarray(samples.sampled_modes, dtype=np.int8)
            ).attrs["units"] = "int8 (1 = sampled)"
            for name, data, units in (
                ("masses", masses_amu, "amu"),
                ("equilibrium_coordinates", coords_bohr, "bohr"),
                ("frequencies", frequencies_cm1, "cm^-1"),
                ("positions", samples.positions_bohr, "bohr"),
                ("velocities", samples.velocities_au, "bohr/au_time"),
                ("q", samples.q, "sqrt(m_e)*bohr"),
                ("p", samples.p, "sqrt(m_e)*bohr/au_time"),
                ("harmonic_energy", samples.harmonic_energy_hartree, "hartree"),
                ("kinetic_energy", samples.kinetic_energy_hartree, "hartree"),
                ("removed_kinetic_energy", samples.removed_kinetic_energy_hartree, "hartree"),
            ):
                f.create_dataset(name, data=np.asarray(data, dtype=float)).attrs["units"] = units
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def format_samples_xyz(
    symbols: Sequence[str], positions_bohr: ArrayLike, energies_hartree: ArrayLike
) -> str:
    """One XYZ frame per sample, in angstrom; the comment line is ``sample k  E_harm = ... Eh``."""
    positions = np.asarray(positions_bohr, dtype=float)
    energies = np.asarray(energies_hartree, dtype=float)
    frames = []
    for k, (xyz, energy) in enumerate(zip(positions, energies, strict=True), start=1):
        geometry = Geometry(
            tuple(symbols),
            tuple((float(x), float(y), float(z)) for x, y, z in xyz * BOHR_IN_ANGSTROM),
            f"sample {k}  E_harm = {energy:.10f} Eh",
        )
        frames.append(format_xyz(geometry))
    return "".join(frames)
