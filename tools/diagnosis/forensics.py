"""Forensics of finished gqteashmd trajectories (P9a spec section 3.1).

Report helpers only: they read steps.log, output.h5, result.json and electronic.log of a
trajectory directory and change nothing. Energies in the files are in hartree; every number
returned here is in eV or fs.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.core.units import AU_TIME_IN_FS, HARTREE_IN_EV
from gqteashmd.trajectory.store import StepLog, read_output_h5

JUMP_THRESHOLD_EV = 0.1
ONSETS = (
    "convergence_failure",
    "failure_after_hop",
    "overlap_failure_without_hop",
    "jump_without_hop",
    "jump_after_hop",
    "none",
)
LAST_RECORDS = 5
_LOG_KEYWORDS = re.compile(r"converge|casscf|scf|fail|rung", re.IGNORECASE)


@dataclass(frozen=True)
class Jump:
    record: int
    time_fs: float
    delta_e_total_eV: float
    state: int
    delta_e_state_eV: float
    gap_eV: float
    min_singular_value: float
    rung_changed: bool
    depth: int
    active: int


@dataclass(frozen=True)
class Hop:
    record: int
    time_fs: float
    from_state: int
    to_state: int
    gap_eV: float


def energy_jumps(
    time_fs: np.ndarray,
    e_total_Eh: np.ndarray,
    energies_Eh: np.ndarray,
    *,
    threshold_eV: float = JUMP_THRESHOLD_EV,
    min_singular_value: np.ndarray | None = None,
    rung_changed: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    active: np.ndarray | None = None,
) -> list[Jump]:
    """Consecutive records whose |E_tot difference| exceeds threshold_eV (strictly)."""
    e = np.asarray(e_total_Eh, dtype=float)
    energies = np.asarray(energies_Eh, dtype=float)
    if e.size < 2:
        return []
    de = np.diff(e) * HARTREE_IN_EV
    out: list[Jump] = []
    for k in np.flatnonzero(np.abs(de) > threshold_eV):
        i = int(k) + 1
        d_state = (energies[i] - energies[i - 1]) * HARTREE_IN_EV
        s = int(np.argmax(np.abs(d_state)))
        gap = (
            (energies[i, 1] - energies[i, 0]) * HARTREE_IN_EV if energies.shape[1] > 1 else math.nan
        )
        out.append(
            Jump(
                record=i,
                time_fs=float(time_fs[i]),
                delta_e_total_eV=float(de[k]),
                state=s,
                delta_e_state_eV=float(d_state[s]),
                gap_eV=float(gap),
                min_singular_value=math.nan
                if min_singular_value is None
                else float(min_singular_value[i]),
                rung_changed=False if rung_changed is None else bool(rung_changed[i]),
                depth=0 if depth is None else int(depth[i]),
                active=-1 if active is None else int(active[i]),
            )
        )
    return out


def log_jumps(rows: np.ndarray, threshold_eV: float = JUMP_THRESHOLD_EV) -> list[Jump]:
    return energy_jumps(
        rows["time_au"] * AU_TIME_IN_FS,
        rows["e_total"],
        rows["energies"],
        threshold_eV=threshold_eV,
        min_singular_value=rows["min_singular_value"],
        rung_changed=rows["rung_changed"],
        depth=rows["depth"],
        active=rows["active"],
    )


def log_hops(rows: np.ndarray) -> list[Hop]:
    out: list[Hop] = []
    for i in np.flatnonzero(rows["hop_to"] >= 0):
        i = int(i)
        a, b = int(rows["hop_from"][i]), int(rows["hop_to"][i])
        gap = abs(float(rows["energies"][i][b] - rows["energies"][i][a])) * HARTREE_IN_EV
        out.append(Hop(i, float(rows["time_au"][i] * AU_TIME_IN_FS), a, b, gap))
    return out


def classify_onset(outcome: str, jumps: Sequence[Jump], hops: Sequence[Hop]) -> str:
    """One label of ONSETS: what happened first on the way to the trajectory's end."""
    if outcome == "electronic_failure":
        return "convergence_failure"
    if outcome == "overlap_failure":
        return "failure_after_hop" if hops else "overlap_failure_without_hop"
    if jumps:
        first_hop = hops[0].time_fs if hops else math.inf
        return "jump_without_hop" if jumps[0].time_fs < first_hop else "jump_after_hop"
    return "none"


def failed_geometry(
    q: np.ndarray, v: np.ndarray, gradient: np.ndarray, masses_me: np.ndarray, dt_au: float
) -> np.ndarray:
    """The velocity-Verlet position of the step after (q, v): q + v dt - g/(2m) dt^2."""
    q, v, g, m = (np.asarray(x, dtype=float) for x in (q, v, gradient, masses_me))
    return q + v * dt_au - 0.5 * (g / m) * dt_au**2


def electronic_log_tail(text: str, n: int = 20) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if _LOG_KEYWORDS.search(line)]
    return lines[-n:]


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def find_stage_dir(project_dir: Path, name: str) -> Path:
    found = sorted(p for p in Path(project_dir).glob(f"[0-9][0-9]_{name}") if p.is_dir())
    if len(found) != 1:
        raise ValueError(f"{project_dir} has {len(found)} directories for stage {name!r}")
    return found[0]


def read_rows(traj_dir: Path) -> tuple[np.ndarray, list[str], int]:
    data = read_output_h5(Path(traj_dir) / "output.h5")
    labels = list(data.labels)
    n_atoms = int(data["positions"].shape[1])
    rows = StepLog(Path(traj_dir) / "steps.log", len(labels), 3 * n_atoms).read()
    return rows, labels, n_atoms


def _result(traj_dir: Path) -> dict[str, Any]:
    path = Path(traj_dir) / "result.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return dict(raw.get("result", raw)) if isinstance(raw, dict) else {}


def trajectory_report(traj_dir: Path, *, threshold_eV: float = JUMP_THRESHOLD_EV) -> dict[str, Any]:
    traj_dir = Path(traj_dir)
    rows, labels, _ = read_rows(traj_dir)
    result = _result(traj_dir)
    outcome = str(result.get("outcome", read_output_h5(traj_dir / "output.h5").outcome))
    jumps = log_jumps(rows, threshold_eV)
    hops = log_hops(rows)
    log_path = traj_dir / "electronic.log"
    tail = electronic_log_tail(log_path.read_text(errors="replace")) if log_path.is_file() else []
    last = [
        {
            "record": int(i),
            "time_fs": float(rows["time_au"][i] * AU_TIME_IN_FS),
            "depth": int(rows["depth"][i]),
            "active": int(rows["active"][i]),
            "min_singular_value": float(rows["min_singular_value"][i]),
            "rung_changed": bool(rows["rung_changed"][i]),
            "energies_eV": (rows["energies"][i] * HARTREE_IN_EV).tolist(),
            "e_total_eV": float(rows["e_total"][i] * HARTREE_IN_EV),
        }
        for i in range(max(0, rows.size - LAST_RECORDS), rows.size)
    ]
    return {
        "task": traj_dir.name,
        "labels": labels,
        "outcome": outcome,
        "t_final_fs": result.get("t_final_fs"),
        "message": result.get("message"),
        "n_records": int(rows.size),
        "jumps": jsonable(jumps),
        "hops": jsonable(hops),
        "onset": classify_onset(outcome, jumps, hops),
        "last_records": last,
        "electronic_log": tail,
    }


def project_report(project_dir: Path, *, threshold_eV: float = JUMP_THRESHOLD_EV) -> dict[str, Any]:
    tasks = find_stage_dir(project_dir, "dynamics") / "tasks"
    reports = [
        trajectory_report(d, threshold_eV=threshold_eV)
        for d in sorted(tasks.glob("traj_*"))
        if (d / "output.h5").is_file()
    ]
    onsets = Counter(r["onset"] for r in reports)
    return {
        "project": str(project_dir),
        "threshold_eV": threshold_eV,
        "n_trajectories": len(reports),
        "outcomes": dict(sorted(Counter(r["outcome"] for r in reports).items())),
        "onsets": {o: int(onsets.get(o, 0)) for o in ONSETS},
        "trajectories": reports,
    }
