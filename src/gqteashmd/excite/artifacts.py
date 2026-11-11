"""excitations.h5 and spectrum.csv (h5py imported lazily)."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gqteashmd import __version__
from gqteashmd.core.units import HARTREE_IN_EV
from gqteashmd.excite.spectrum import Spectrum


def _excitation_energies_eV(energies_hartree: NDArray[np.float64]) -> NDArray[np.float64]:
    return (energies_hartree[:, 1:] - energies_hartree[:, :1]) * HARTREE_IN_EV


def write_excitations_h5(
    path: Path,
    *,
    symbols: Sequence[str],
    state_labels: Sequence[str],
    sample_indices: ArrayLike,
    ok: ArrayLike,
    energies_hartree: ArrayLike,
    transition_dipoles_au: ArrayLike,
    oscillator_strengths: ArrayLike,
    reference: dict[str, Any] | None = None,
    project_seed: int | None = None,
) -> None:
    """Write vertical excitations of all samples (NaN rows for failed samples).

    ``reference`` (optional) holds ``energies_hartree`` (1, n_states), ``transition_dipoles_au``
    (1, n_states-1, 3), ``oscillator_strengths`` (1, n_states-1) and ``coordinates_bohr``.
    """
    import h5py

    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    ok_arr = np.asarray(ok, dtype=bool)
    energies = np.asarray(energies_hartree, dtype=float)
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["schema_version"] = 1
            f.attrs["gqteashmd_version"] = __version__
            f.attrs["n_samples"] = int(ok_arr.size)
            f.attrs["n_ok"] = int(ok_arr.sum())
            if project_seed is not None:
                f.attrs["project_seed"] = int(project_seed)
            string = h5py.string_dtype()
            f.create_dataset("symbols", data=np.array(list(symbols), dtype=string))
            f.create_dataset("state_labels", data=np.array(list(state_labels), dtype=string))
            f["symbols"].attrs["units"] = "-"
            f["state_labels"].attrs["units"] = "-"
            f.create_dataset("sample_index", data=np.asarray(sample_indices, dtype=np.int32)).attrs[
                "units"
            ] = "1-based"
            f.create_dataset("ok", data=ok_arr.astype(np.int8)).attrs["units"] = "int8 (1 = ok)"
            for name, data, units in (
                ("energies", energies, "hartree"),
                ("excitation_energies", _excitation_energies_eV(energies), "eV"),
                ("transition_dipoles", transition_dipoles_au, "e*a0"),
                ("oscillator_strengths", oscillator_strengths, "-"),
            ):
                f.create_dataset(name, data=np.asarray(data, dtype=float)).attrs["units"] = units
            if reference is not None:
                g = f.create_group("reference")
                ref_e = np.asarray(reference["energies_hartree"], dtype=float)
                for name, data, units in (
                    ("energies", ref_e, "hartree"),
                    ("excitation_energies", _excitation_energies_eV(ref_e), "eV"),
                    ("transition_dipoles", reference["transition_dipoles_au"], "e*a0"),
                    ("oscillator_strengths", reference["oscillator_strengths"], "-"),
                    ("coordinates", reference["coordinates_bohr"], "bohr"),
                ):
                    ds = g.create_dataset(name, data=np.asarray(data, dtype=float))
                    ds.attrs["units"] = units
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _text(values: Any) -> tuple[str, ...]:
    return tuple(s.decode() if isinstance(s, bytes) else str(s) for s in values)


def _floats(dataset: Any) -> NDArray[np.float64]:
    return np.asarray(dataset[()], dtype=float)


def read_excitations_h5(path: Path) -> dict[str, Any]:
    """Read ``excitations.h5``; a missing or inconsistent file raises ``ValueError``."""
    import h5py

    path = Path(path)
    advice = f"; rerun the excitations stage with --rerun-from excitations ({path})"
    try:
        with h5py.File(path, "r") as f:
            data: dict[str, Any] = {
                "symbols": _text(f["symbols"][()]),
                "state_labels": _text(f["state_labels"][()]),
                "sample_index": np.asarray(f["sample_index"][()], dtype=int),
                "ok": np.asarray(f["ok"][()]).astype(bool),
                "energies_hartree": _floats(f["energies"]),
                "excitation_energies_eV": _floats(f["excitation_energies"]),
                "transition_dipoles_au": _floats(f["transition_dipoles"]),
                "oscillator_strengths": _floats(f["oscillator_strengths"]),
                "reference": None,
            }
            if "reference" in f:
                g = f["reference"]
                data["reference"] = {
                    "energies_hartree": _floats(g["energies"]),
                    "excitation_energies_eV": _floats(g["excitation_energies"]),
                    "transition_dipoles_au": _floats(g["transition_dipoles"]),
                    "oscillator_strengths": _floats(g["oscillator_strengths"]),
                    "coordinates_bohr": _floats(g["coordinates"]),
                }
    except (OSError, KeyError) as exc:
        raise ValueError(f"excitations.h5 is missing or unreadable ({exc!r}){advice}") from exc
    n, ns = data["ok"].size, len(data["state_labels"])
    expected = {
        "sample_index": (data["sample_index"].shape, (n,)),
        "energies": (data["energies_hartree"].shape, (n, ns)),
        "excitation_energies": (data["excitation_energies_eV"].shape, (n, ns - 1)),
        "transition_dipoles": (data["transition_dipoles_au"].shape, (n, ns - 1, 3)),
        "oscillator_strengths": (data["oscillator_strengths"].shape, (n, ns - 1)),
    }
    for name, (got, want) in expected.items():
        if got != want:
            raise ValueError(
                f"excitations.h5 is inconsistent: dataset '{name}' has shape {got}, "
                f"expected {want}{advice}"
            )
    return data


def format_spectrum_csv(
    spectrum: Spectrum,
    state_labels: Sequence[str],
    *,
    n_ok: int | None = None,
    fwhm_eV: float | None = None,
    step_eV: float | None = None,
) -> str:
    """CSV text of the spectrum.

    ``state_labels`` are the excited states (a leading S0 is dropped). ``n_ok``, ``fwhm_eV`` and
    ``step_eV`` only appear in the ``#`` header lines (``step_eV`` defaults to the grid step).
    """
    n_exc = spectrum.cross_section_by_state_A2.shape[1]
    labels = list(state_labels)
    if len(labels) == n_exc + 1:
        labels = labels[1:]
    if len(labels) != n_exc:
        raise ValueError(f"{len(labels)} state labels for {n_exc} excited states")
    if step_eV is None and spectrum.energies_eV.size > 1:
        step_eV = float(spectrum.energies_eV[1] - spectrum.energies_eV[0])

    def show(value: object) -> str:
        return "unknown" if value is None else str(value)

    columns = ["energy_eV", "cross_section_A2", "epsilon_M-1cm-1", *(f"{s}_A2" for s in labels)]
    lines = [
        "# gqteashmd nuclear-ensemble absorption spectrum",
        "# units: energy_eV = eV, cross sections = angstrom^2 per molecule, epsilon = M^-1 cm^-1",
        f"# n_ok = {show(n_ok)}",
        f"# fwhm_eV = {show(fwhm_eV)}",
        f"# step_eV = {show(step_eV)}",
        ",".join(columns),
    ]
    for i, e in enumerate(spectrum.energies_eV):
        row = [e, spectrum.cross_section_A2[i], spectrum.epsilon_M1cm1[i]]
        row.extend(spectrum.cross_section_by_state_A2[i])
        lines.append(",".join(f"{float(x):.6e}" for x in row))
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class SelectedTrajectory:
    """One entry of trajectories.json (USER_MANUAL section 6)."""

    index: int
    sample: int
    state: str
    rng: Mapping[str, Any]

    @property
    def task_id(self) -> str:
        return f"traj_{self.index:05d}"


def read_trajectories_json(path: Path) -> list[SelectedTrajectory]:
    """The selected trajectories; ValueError naming the file if it is missing or malformed."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [
            SelectedTrajectory(int(e["index"]), int(e["sample"]), str(e["state"]), e["rng"])
            for e in data["trajectories"]
        ]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{path} is missing or malformed ({exc!r})") from exc
