"""Canned upstream data and a fake backend for dynamics-stage tests (no PySCF needed).

The LVC template comes from the environment variable ``LVC_TEMPLATE_ENV``; the model has no
spin-orbit coupling. Faults for the fake backend come from ``FAULTS_ENV`` as JSON
``{"traj_00001": {"5": "kill"}}`` (task id, then 1-based compute call number, then fault kind).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from gqteashmd.core.units import BOHR_IN_ANGSTROM
from gqteashmd.dynamics.lvc import LVCModel, lvc_wigner_samples
from gqteashmd.electronic.base import MoleculeSpec
from gqteashmd.testing.fake_backend import ModelBackend
from gqteashmd.util.fileio import write_json_atomic

LVC_TEMPLATE_ENV = "GQTEASHMD_CANNED_LVC_TEMPLATE"
FAULTS_ENV = "GQTEASHMD_CANNED_FAULTS"


def canned_model() -> LVCModel:
    return LVCModel.from_template(Path(os.environ[LVC_TEMPLATE_ENV]), spin_orbit=False)


def write_canned_molecule(stage_dir: Path, model: LVCModel) -> None:
    v0 = model.parameters.v0
    coords = model.reference_q.reshape(-1, 3) * BOHR_IN_ANGSTROM
    write_json_atomic(
        Path(stage_dir) / "molecule.json",
        {
            "schema_version": 1,
            "symbols": list(v0.symbols),
            "charge": 0,
            "multiplicity": 1,
            "coords_angstrom": coords.tolist(),
        },
    )


def write_canned_samples(stage_dir: Path, model: LVCModel, n_samples: int, seed: int = 0) -> None:
    import h5py

    v0 = model.parameters.v0
    natom = len(v0.symbols)
    q, v = lvc_wigner_samples(model, n_samples, np.random.default_rng(seed))
    with h5py.File(Path(stage_dir) / "samples.h5", "w") as f:
        f.attrs["n_samples"] = int(n_samples)
        f.attrs["temperature_K"] = 0.0
        f.create_dataset("symbols", data=np.array(list(v0.symbols), dtype=h5py.string_dtype()))
        f["symbols"].attrs["units"] = "-"
        for name, data, units in (
            ("masses", np.asarray(v0.masses_amu, dtype=float), "amu"),
            ("equilibrium_coordinates", model.reference_q.reshape(natom, 3), "bohr"),
            ("positions", q.reshape(n_samples, natom, 3), "bohr"),
            ("velocities", v.reshape(n_samples, natom, 3), "bohr/au_time"),
        ):
            f.create_dataset(name, data=data).attrs["units"] = units


def write_canned_trajectories(
    stage_dir: Path, entries: list[tuple[int, str]], project_seed: int
) -> None:
    trajectories = [
        {
            "id": f"t{k + 1:05d}",
            "index": k,
            "sample": int(sample),
            "state": state,
            "excitation_energy_eV": 0.0,
            "oscillator_strength": 0.0,
            "rng": {"stage": "dynamics", "task_index": k},
        }
        for k, (sample, state) in enumerate(entries)
    ]
    write_json_atomic(
        Path(stage_dir) / "trajectories.json",
        {
            "schema_version": 1,
            "n_trajectories": len(trajectories),
            "mode": "fixed",
            "project_seed": project_seed,
            "trajectories": trajectories,
        },
    )


def lvc_backend_factory(ctx: object, molecule: MoleculeSpec, task_dir: Path) -> ModelBackend:
    raw = os.environ.get(FAULTS_ENV)
    faults = json.loads(raw).get(Path(task_dir).name, {}) if raw else {}
    return ModelBackend(
        canned_model(),
        faults={int(c): kind for c, kind in faults.items()},
        marker_dir=Path(task_dir),
    )
