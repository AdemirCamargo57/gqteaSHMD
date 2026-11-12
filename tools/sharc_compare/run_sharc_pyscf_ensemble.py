"""Run sharc.x + SHARC_PYSCF.py on the initial conditions of a gqteashmd project (P9a spec 3.4).

The diagnosis of the M4 trajectory losses compares how SHARC's trajectories end when started
from the very same selected geometries and velocities. SHARC is only *executed* (``sharc.x``,
``SHARC_PYSCF.py``); nothing from its installation is read or imported. The ``geom``/``veloc``
formats (manual 4.2, 4.3) are those written by ``run_sharc_lvc.format_geom``/``format_veloc``.

Example (WSL):
  python tools/sharc_compare/run_sharc_pyscf_ensemble.py ~/p8_m4/run \\
      tools/diagnosis/variants_p9a.toml --variant base --out ~/p9a/sharc \\
      --sharc-home /home/ajc/sharc4-4.1 \\
      --python /home/ajc/miniforge3/envs/sharc4.1/bin/python --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.analysis.energy import max_jump_eV
from gqteashmd.core.states import StateLabel
from gqteashmd.core.units import HARTREE_IN_EV

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "diagnosis"))
from forensics import JUMP_THRESHOLD_EV, energy_jumps, jsonable  # noqa: E402
from project_inputs import load_project_inputs  # noqa: E402
from run_sharc_lvc import parse_lvc_output_dat  # noqa: E402
from run_sharc_pyscf_dynamics import (  # noqa: E402
    Convergence,
    interface_geometry,
    pyscf_template,
    run_sharc_trajectory,
    trajectory_data,
)
from run_variants import Variant, read_variants, variant_sections  # noqa: E402

GEOM_FIELDS = 6  # symbol, Z, x, y, z, mass


def parse_geom(text: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Symbols, masses (amu) and coordinates (bohr) of a ``geom`` file (``symbol Z x y z mass``)."""
    symbols: list[str] = []
    masses: list[float] = []
    coords: list[list[float]] = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        f = line.split()
        if len(f) != GEOM_FIELDS:
            raise ValueError(f"geom line {n}: expected {GEOM_FIELDS} fields, got {len(f)}")
        symbols.append(f[0])
        coords.append([float(x) for x in f[2:5]])
        masses.append(float(f[5]))
    return symbols, np.array(masses), np.array(coords).reshape(-1, 3)


def parse_veloc(text: str) -> np.ndarray:
    """The (natom, 3) velocities of a ``veloc`` file (three floats per line)."""
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        f = line.split()
        if len(f) != 3:
            raise ValueError(f"veloc line {n}: expected 3 fields, got {len(f)}")
        rows.append([float(x) for x in f])
    return np.array(rows).reshape(-1, 3)


def sharc_settings(electronic, dynamics) -> dict[str, Any]:
    """Keyword arguments of ``run_sharc_trajectory`` for one variant (CASSCF only)."""
    if electronic.method != "casscf" or electronic.casscf is None:
        raise ValueError(
            f"SHARC_PYSCF.py runs casscf only; the variant has method = {electronic.method!r}"
        )
    c = electronic.casscf
    return {
        "n_states": electronic.singlets,
        "dt_fs": dynamics.dt_fs,
        "hopping": "off" if dynamics.hopping == "none" else "sharc",
        "decoherence": "edc" if dynamics.decoherence == "edc" else "none",
        "coupling": "nacdr",
        "phases_from_interface": False,
        "template": pyscf_template(
            Convergence(c.conv_tol, c.conv_tol_grad),
            ncas=c.ncas,
            nelecas=c.nelecas,
            roots=electronic.singlets,
            basis=electronic.basis,
        ),
    }


def trajectory_summary(
    task_id: str, out: dict[str, Any], *, dt_fs: float, tmax_fs: float
) -> dict[str, Any]:
    """Per-trajectory summary of a ``run_sharc_trajectory`` result (success or failure)."""
    data = out if out.get("returncode") == 0 and "steps" in out else out.get("data")
    summary: dict[str, Any] = {
        "task": task_id,
        "returncode": out.get("returncode"),
        "stderr_tail": out.get("stderr_tail"),
        "error": out.get("error"),
        "t_final_fs": None,
        "n_steps": 0,
        "n_hops": 0,
        "max_drift_eV": None,
        "max_jump_eV": None,
        "jumps": [],
        "ended": "stopped",
    }
    if data is None or len(data["steps"]) == 0:
        return summary
    steps = np.asarray(data["steps"])
    t = steps * dt_fs
    e_total = np.asarray(data["e_total"], dtype=float)
    active = np.asarray(data["active"])
    summary.update(
        t_final_fs=float(t[-1]),
        n_steps=int(len(steps) - 1),
        n_hops=int(np.count_nonzero(np.diff(active))),
        max_drift_eV=float(np.abs(e_total - e_total[0]).max() * HARTREE_IN_EV),
        max_jump_eV=max_jump_eV(e_total),
        jumps=jsonable(
            energy_jumps(
                t,
                e_total,
                np.asarray(data["energies"], dtype=float),
                threshold_eV=JUMP_THRESHOLD_EV,
                active=active,
            )
        ),
        ended="finished"
        if out.get("returncode") == 0 and t[-1] >= tmax_fs - dt_fs / 2
        else "stopped",
    )
    return summary


THREAD_VARIABLES = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}  # PYSCF.resources has ncpu 1: several sharc.x in parallel must not oversubscribe the cores
STATUS_FILE = "status.json"


def _write_status(workdir: Path, out: dict[str, Any], error: str | None, wall_time_s: float):
    """status.json: what the run ended with (written atomically after every run)."""
    status = {
        "returncode": out.get("returncode"),
        "stderr_tail": out.get("stderr_tail"),
        "error": error,
        "wall_time_s": wall_time_s,
    }
    workdir.mkdir(parents=True, exist_ok=True)
    tmp = workdir / f".{STATUS_FILE}.tmp"
    tmp.write_text(json.dumps(status))
    os.replace(tmp, workdir / STATUS_FILE)


def _read_existing(workdir: Path) -> dict[str, Any]:
    """A finished run directory (it has status.json): its recorded ending and its output.dat."""
    status = json.loads((workdir / STATUS_FILE).read_text())
    out: dict[str, Any] = {
        "returncode": status.get("returncode"),
        "stderr_tail": status.get("stderr_tail"),
        "error": status.get("error"),
        "wall_time_s": status.get("wall_time_s"),
        "data": None,
    }
    if (workdir / "output.dat").is_file():
        try:
            data = trajectory_data(parse_lvc_output_dat(workdir / "output.dat"))
        except (ValueError, IndexError, OSError):
            return out
        out["data"] = data
        if out["returncode"] == 0:
            out.update(data)
    return out


def _move_aside(workdir: Path) -> Path:
    """Rename a work dir that has no status.json to <task>.stale-<k> (first free k)."""
    k = 0
    while (target := workdir.with_name(f"{workdir.name}.stale-{k}")).exists():
        k += 1
    workdir.rename(target)
    return target


def run_ensemble(
    project_dir: Path,
    variant: Variant,
    out_dir: Path,
    *,
    sharc_home: Path,
    sharc_python: str,
    workers: int,
    tmax_fs: float | None = None,
    only: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run SHARC from every (or the ``only``) selected start of the project."""
    inputs = load_project_inputs(project_dir)
    electronic, dynamics = variant_sections(inputs.config, variant)
    tmax = dynamics.tmax_fs if tmax_fs is None else tmax_fs
    n_steps = round(tmax / dynamics.dt_fs)
    settings = sharc_settings(electronic, dynamics)
    base = Path(out_dir) / variant.name
    wanted = None if only is None else set(only)
    starts = [s for s in inputs.starts if wanted is None or s.task_id in wanted]

    def job(start) -> dict[str, Any]:
        workdir = base / start.task_id
        if workdir.exists() and not (workdir / STATUS_FILE).is_file():
            _move_aside(workdir)
        reused = workdir.exists()
        if reused:
            out = _read_existing(workdir)
        else:
            t0 = time.perf_counter()
            error = None
            try:
                out = run_sharc_trajectory(
                    interface_geometry(start.q.reshape(-1, 3)),
                    start.v.reshape(-1, 3),
                    n_steps=n_steps,
                    workdir=workdir,
                    symbols=inputs.symbols,
                    masses_amu=inputs.masses_amu,
                    sharc_home=Path(sharc_home),
                    sharc_python=sharc_python,
                    provider="sharc_pyscf",
                    seed=start.index + 1,
                    state_mch=StateLabel.parse(start.state).index,
                    check=False,
                    env_extra=dict(THREAD_VARIABLES),
                    **settings,
                )
            except Exception as exc:  # noqa: BLE001 - recorded, never aborts the ensemble
                error = f"{type(exc).__name__}: {exc}"
                out = {"returncode": None, "stderr_tail": None, "data": None}
            _write_status(workdir, out, error, time.perf_counter() - t0)
            out = {**out, "error": error}
        s = trajectory_summary(start.task_id, out, dt_fs=dynamics.dt_fs, tmax_fs=tmax)
        s["reused"] = reused
        return s

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        trajectories = list(pool.map(job, starts))
    ended: dict[str, int] = {}
    for s in trajectories:
        ended[s["ended"]] = ended.get(s["ended"], 0) + 1
    summary = {
        "variant": variant.name,
        "project": str(inputs.project_dir),
        "n_trajectories": len(trajectories),
        "tmax_fs": tmax,
        "ended": ended,
        "n_errors": sum(1 for s in trajectories if s["error"]),
        "n_with_jump": sum(1 for s in trajectories if s["jumps"]),
        "n_finished_without_jump": sum(
            1 for s in trajectories if s["ended"] == "finished" and not s["jumps"]
        ),
        "trajectories": trajectories,
        "wall_time_s": time.perf_counter() - started,
    }
    base.mkdir(parents=True, exist_ok=True)
    (base / "sharc_summary.json").write_text(json.dumps(jsonable(summary), indent=2))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("project", type=Path)
    p.add_argument("variants", type=Path)
    p.add_argument("--variant", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sharc-home", type=Path, required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tmax-fs", type=float, default=None)
    p.add_argument("--task", action="append", default=None)
    args = p.parse_args(argv)
    variants = {v.name: v for v in read_variants(args.variants)}
    if args.variant not in variants:
        p.error(f"unknown variant {args.variant!r}; the file has {sorted(variants)}")
    summary = run_ensemble(
        args.project,
        variants[args.variant],
        args.out.expanduser(),
        sharc_home=args.sharc_home,
        sharc_python=args.python,
        workers=args.workers,
        tmax_fs=args.tmax_fs,
        only=args.task,
    )
    print(json.dumps({k: v for k, v in jsonable(summary).items() if k != "trajectories"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
