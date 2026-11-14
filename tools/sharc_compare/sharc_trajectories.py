"""SHARC trajectories and ours in the form our analysis reads (P9b spec 3.2, 4, 7.2; PR1).

``from_parsed`` turns a parsed SHARC ``output.dat`` (``run_sharc_lvc.parse_lvc_output_dat``)
into an ``analysis.grid.Trajectory``: one record per printed step (``ticks`` 0), hops from the
changes of the active state, no frustrated hops (PR4), ``E_tot`` = the active MCH energy plus
the kinetic energy. SHARC is only executed; nothing from it is imported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from gqteashmd.analysis.energy import max_jump_eV
from gqteashmd.analysis.grid import FINISHED, INCLUDED_OUTCOMES, Trajectory
from gqteashmd.core.units import AU_TIME_IN_FS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_sharc_lvc import parse_lvc_output_dat  # noqa: E402

KEPT_JUMP_EV = 0.1  # spec 3.2: the analysis default energy_jump_eV
STOPPED = "stopped"  # a SHARC run that did not reach tmax


def from_parsed(task: str, traj, *, n_total: int) -> Trajectory:
    steps = np.asarray(traj.steps, dtype=np.int64)
    active = np.asarray(traj.active, dtype=np.int64)
    n = traj.hamiltonian.shape[1]
    energies = traj.hamiltonian[:, np.arange(n), np.arange(n)].real
    k = np.arange(steps.size)
    hop_from = np.full(steps.size, -1, dtype=np.int64)
    hop_to = np.full(steps.size, -1, dtype=np.int64)
    changed = np.flatnonzero(active[1:] != active[:-1]) + 1
    hop_from[changed] = active[changed - 1]
    hop_to[changed] = active[changed]
    dt_fs = float(traj.dt_au) * AU_TIME_IN_FS
    finished = steps.size > 0 and int(steps[-1]) >= n_total
    return Trajectory(
        task=task,
        outcome=FINISHED if finished else STOPPED,
        initial_state=int(active[0]) if active.size else -1,
        grid_step=steps,
        ticks=np.zeros(steps.size, dtype=np.int64),
        time_fs=steps * dt_fs,
        active=active,
        hop_from=hop_from,
        hop_to=hop_to,
        frustrated=np.zeros(steps.size, dtype=bool),
        e_total=energies[k, active] + np.asarray(traj.e_kinetic, dtype=float),
        populations=np.abs(np.asarray(traj.c)) ** 2,
    )


def read_sharc_trajectory(directory: Path, task: str, *, n_total: int) -> Trajectory | None:
    """None for a directory without a readable output.dat (a run that never started)."""
    path = Path(directory) / "output.dat"
    if not path.is_file():
        return None
    try:
        return from_parsed(task, parse_lvc_output_dat(path), n_total=n_total)
    except (ValueError, IndexError, OSError):
        return None


def read_our_trajectory(path: Path, task: str) -> Trajectory | None:
    """Our output.h5 as the analysis stage reads it (stages/analysis.py:_trajectory)."""
    from gqteashmd.trajectory.store import read_output_h5

    if not Path(path).is_file():
        return None
    data = read_output_h5(Path(path))
    return Trajectory(
        task=task,
        outcome=data.outcome,
        initial_state=data.labels.index(data.initial_state),
        grid_step=np.asarray(data["grid_step"], dtype=np.int64),
        ticks=np.asarray(data["ticks"], dtype=np.int64),
        time_fs=np.asarray(data["time_fs"], dtype=float),
        active=np.asarray(data["active"], dtype=np.int64),
        hop_from=np.asarray(data["hop_from"], dtype=np.int64),
        hop_to=np.asarray(data["hop_to"], dtype=np.int64),
        frustrated=np.asarray(data["frustrated"]) != 0,
        e_total=np.asarray(data["e_total"], dtype=float),
        populations=np.asarray(data["populations"], dtype=float),
    )


def max_jump_up_to(t: Trajectory, n_total: int) -> float:
    """The largest record-to-record |dE_tot| (eV) over grid steps <= n_total (P8 PR3)."""
    return max_jump_eV(t.e_total[t.grid_step <= n_total])


def is_kept(t: Trajectory | None, n_total: int) -> bool:
    """Spec 3.2: an included outcome (SHARC: reached tmax) and no jump above 0.1 eV."""
    if t is None or t.outcome not in INCLUDED_OUTCOMES:
        return False
    return max_jump_up_to(t, n_total) <= KEPT_JUMP_EV
