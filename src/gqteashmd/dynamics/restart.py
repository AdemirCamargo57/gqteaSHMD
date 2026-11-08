"""Restart files of one trajectory (``.npz``, written atomically).

The file holds everything the next step needs, so continuing after ``load_state`` is
bit-identical to never having stopped: q, v, c, the active state, time, step, E_tot(0), the full
electronic snapshot (energies, gradients, NACs, Hamiltonian, model vectors) and the random
source's state (JSON). ``load_state`` evaluates nothing; it checks the array shapes against the
model (the masses and model parameters are not stored, so the caller must pass the same model).
U is not stored: in P7a-1 the driver uses the identity reference (see ``driver``).
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.dynamics.models import ElectronicModel
from gqteashmd.dynamics.randomness import RandomSource, source_from_state
from gqteashmd.dynamics.records import ElectronicSnapshot, TrajectoryState
from gqteashmd.util.fileio import write_bytes_atomic

FORMAT = "gqteashmd-dynamics-restart"
VERSION = 1
_ARRAYS = (
    "q",
    "v",
    "c",
    "energies",
    "gradients",
    "nacs",
    "hamiltonian",
)


def save_state(path: Path, state: TrajectoryState, rng: RandomSource) -> None:
    """Write ``state`` and ``rng.state()`` to ``path`` atomically (temporary file + rename)."""
    snap = state.snapshot
    meta: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "time_au": float(state.time_au),
        "step": int(state.step),
        "active": int(state.active),
        "e_total_initial": float(state.e_total_initial),
        "has_vectors": snap.vectors is not None,
        "rng": rng.state(),
    }
    arrays: dict[str, np.ndarray] = {
        "q": np.asarray(state.q),
        "v": np.asarray(state.v),
        "c": np.asarray(state.c),
        "energies": np.asarray(snap.energies),
        "gradients": np.asarray(snap.gradients),
        "nacs": np.asarray(snap.nacs),
        "hamiltonian": np.asarray(snap.hamiltonian),
        "meta": np.array(json.dumps(meta)),
    }
    if snap.vectors is not None:
        arrays["vectors"] = np.asarray(snap.vectors)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)  # type: ignore[arg-type]  # stubs type **kwds as allow_pickle
    write_bytes_atomic(Path(path), buffer.getvalue())


def load_state(path: Path, model: ElectronicModel) -> tuple[TrajectoryState, RandomSource]:
    """Read a restart file written by ``save_state``; ``model`` is only checked, not evaluated."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: np.array(data[name]) for name in _ARRAYS}
            vectors = np.array(data["vectors"]) if "vectors" in data.files else None
            meta = json.loads(str(data["meta"]))
        if meta.get("format") != FORMAT or meta.get("version") != VERSION:
            raise ValueError("unknown format or version")
        if bool(meta["has_vectors"]) != (vectors is not None):
            raise ValueError("vectors flag inconsistent")
        rng = source_from_state(meta["rng"])
        time_au = float(meta["time_au"])
        step = int(meta["step"])
        active = int(meta["active"])
        e_total_initial = float(meta["e_total_initial"])
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        zipfile.BadZipFile,
        EOFError,
    ) as exc:
        raise ValueError(
            f"{path} is not a readable gqteashmd dynamics restart file: {exc}"
        ) from exc
    n, f = int(model.n_states), int(model.n_dof)
    expected = {
        "q": (f,),
        "v": (f,),
        "c": (n,),
        "energies": (n,),
        "gradients": (n, f),
        "nacs": (n, n, f),
        "hamiltonian": (n, n),
    }
    mismatch = [k for k, shape in expected.items() if arrays[k].shape != shape]
    if vectors is not None and vectors.shape[1:] != (n,):
        mismatch.append("vectors")
    if not 0 <= active < n:
        mismatch.append("active")
    if mismatch:
        raise ValueError(
            f"restart file {path} does not match the model ({', '.join(mismatch)} differ)"
        )
    snapshot = ElectronicSnapshot(
        energies=arrays["energies"],
        gradients=arrays["gradients"],
        nacs=arrays["nacs"],
        hamiltonian=arrays["hamiltonian"],
        vectors=vectors,
    )
    state = TrajectoryState(
        time_au=time_au,
        step=step,
        q=np.asarray(arrays["q"], dtype=float),
        v=np.asarray(arrays["v"], dtype=float),
        c=np.asarray(arrays["c"], dtype=complex),
        active=active,
        snapshot=snapshot,
        e_total_initial=e_total_initial,
    )
    return state, rng
