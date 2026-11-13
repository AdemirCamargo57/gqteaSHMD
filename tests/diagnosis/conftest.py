"""Helpers shared by the tools/diagnosis tests: load a script by path, write a trajectory."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from gqteashmd.core.units import AU_TIME_IN_FS, HARTREE_IN_EV
from gqteashmd.trajectory.store import StepLog, log_dtype, log_row, write_output_h5

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def load_tool(subdir: str, name: str):
    """Import tools/<subdir>/<name>.py once (shared module object, like tests/sharc_compare)."""
    key = f"{subdir}_{name}"
    if key in sys.modules:
        return sys.modules[key]
    path = TOOLS / subdir / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    sys.modules.setdefault(name, module)  # sibling imports (e.g. `import forensics`)
    spec.loader.exec_module(module)
    return module


def make_rows(e_total_eV, energies_eV, *, active=None, hop_to=None, dt_fs=0.5, sv=None):
    """steps.log rows for a 2-atom, n-state toy trajectory (energies given in eV)."""
    e_total_eV = np.asarray(e_total_eV, dtype=float)
    energies_eV = np.asarray(energies_eV, dtype=float)
    m, n = energies_eV.shape
    dtype = log_dtype(n, 6)
    rows = np.zeros(m, dtype=dtype)
    for k in range(m):
        rows[k] = log_row(
            dtype,
            step=k,
            grid_step=k,
            ticks=0,
            depth=0,
            time_au=k * dt_fs / AU_TIME_IN_FS,
            active=1 if active is None else active[k],
            hop_from=-1 if hop_to is None or hop_to[k] < 0 else (active[k - 1]),
            hop_to=-1 if hop_to is None else hop_to[k],
            frustrated=0,
            rung_changed=0,
            rejections=0,
            random_number=0.5,
            min_singular_value=0.99 if sv is None else sv[k],
            e_total=e_total_eV[k] / HARTREE_IN_EV,
            e_kinetic=0.0,
            energies=energies_eV[k] / HARTREE_IN_EV,
            c_real=np.eye(n)[1],
            c_imag=np.zeros(n),
            probabilities=np.zeros(n),
            q=np.arange(6, dtype=float),
            v=np.zeros(6),
        )
    return rows


def write_trajectory(traj_dir: Path, rows, *, outcome: str, message: str | None = None):
    """Write steps.log, output.h5 and a worker-format result.json for `rows`."""
    traj_dir.mkdir(parents=True, exist_ok=True)
    n = rows["energies"].shape[1]
    log = StepLog(traj_dir / "steps.log", n, 6)
    for row in rows:
        log.append(row)
    labels = [f"S{i}" for i in range(n)]
    write_output_h5(
        traj_dir / "output.h5",
        rows,
        labels=labels,
        n_atoms=2,
        output_every=1,
        attrs={
            "state_labels": labels,
            "output_every": 1,
            "dt_fs": 0.5,
            "outcome": outcome,
            "task": traj_dir.name,
            "initial_state": "S1",
        },
    )
    result = {
        "outcome": outcome,
        "t_final_fs": float(rows["time_au"][-1] * AU_TIME_IN_FS),
        "message": message,
    }
    (traj_dir / "result.json").write_text(json.dumps({"status": "ok", "result": result}))
