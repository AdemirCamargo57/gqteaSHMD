"""Replayed single points around a trajectory's failure (P9a spec 3.2).

For the records in a window before the failure (the first energy jump, else the end) and, for
overlap and electronic failures, the geometry of the failed step (the Verlet position at the
smallest step), solve SA-CASSCF in three modes and four method variants and record energies,
<S^2>, natural occupations, active-orbital and state overlaps with the previous point,
the convergence rung and the wall time. Reads PySCFBackend._live (diagnostic only, ruling P9a1-1).

Example (WSL):
  python tools/diagnosis/replay_points.py ~/p8_m4/run traj_00004 --out ~/p9a/replay/traj_00004.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forensics import (  # noqa: E402
    failed_geometry,
    jsonable,
    read_rows,
    trajectory_dirs,
    trajectory_report,
)
from project_inputs import load_project_inputs  # noqa: E402
from run_variants import Variant, deep_merge  # noqa: E402

from gqteashmd.config.schema import ElectronicSection  # noqa: E402
from gqteashmd.core.units import AU_TIME_IN_FS, DALTON_IN_ME, HARTREE_IN_EV  # noqa: E402

MODES = ("continued", "backward", "fresh")
REPLAY_VARIANTS = (
    Variant("base", {}, {}),
    Variant("sa3_cas22", {"singlets": 3}, {}),
    Variant("sa3_cas23", {"singlets": 3, "casscf": {"ncas": 3, "nelecas": 2}}, {}),
    Variant("sa2_cas43", {"casscf": {"ncas": 3, "nelecas": 4}}, {}),
)
CONVERGENCE_TRIALS = (  # (conv_tol, conv_tol_grad) of [electronic.casscf]
    (1e-10, 1e-6),
    (1e-8, 1e-5),
    (1e-8, 1e-4),
)
LOST_EXCLUDED = ("finished", "stopped_in_s0")


@dataclass(frozen=True)
class PointResult:
    label: str
    mode: str
    variant: str
    energies_eV: list[float] | None
    spin_squares: list[float] | None
    occupations: list[list[float]] | None
    active_overlap: list[float] | None
    state_overlap_min_sv: float | None
    rung: str | None
    wall_time_s: float
    error: str | None
    segment: int = 0


def active_orbital_overlap(
    c_prev: np.ndarray, c_cur: np.ndarray, s_cross: np.ndarray
) -> np.ndarray:
    """Singular values (descending) of c_prev^T S c_cur."""
    sv = np.linalg.svd(np.asarray(c_prev).T @ s_cross @ np.asarray(c_cur), compute_uv=False)
    return np.sort(sv)[::-1]


def natural_occupations(rdm1: np.ndarray) -> np.ndarray:
    rdm1 = np.asarray(rdm1, dtype=float)
    return np.sort(np.linalg.eigvalsh((rdm1 + rdm1.T) / 2))[::-1]


def probe_window(n_records: int, end: int, size: int) -> list[int]:
    if not 0 <= end < n_records or size < 1:
        raise ValueError(f"bad window: n_records={n_records}, end={end}, size={size}")
    return list(range(max(0, end - size + 1), end + 1))


def failure_record(report: dict[str, Any]) -> int:
    jumps = report["jumps"]
    if jumps:
        first = jumps[0]
        return int(first["record"] if isinstance(first, dict) else first.record)
    return int(report["n_records"]) - 1


def segments(
    n_records: int, jump_record: int, end_record: int, window: int, with_failed_step: bool
) -> list[list[int | str]]:
    """Contiguous runs of records to replay, each solved as its own path.

    The first run is the window ending at the jump (or the end); for an overlap or electronic
    failure a second run is the window ending at the last record, followed by "failed_step".
    Runs that overlap or touch are merged into one.
    """
    first = probe_window(n_records, jump_record, window)
    if not with_failed_step:
        return [list(first)]
    tail = probe_window(n_records, end_record, window)
    if tail[0] <= first[-1] + 1:
        merged: list[int | str] = list(range(first[0], max(first[-1], tail[-1]) + 1))
        return [[*merged, "failed_step"]]
    return [list(first), [*tail, "failed_step"]]


def hop_to_end_segment(n_records: int, hop_record: int, with_failed_step: bool) -> list[int | str]:
    """One run from the first hop record to the last record, plus "failed_step" if asked."""
    if not 0 <= hop_record < n_records:
        raise ValueError(f"hop record {hop_record} is outside 0..{n_records - 1}")
    run: list[int | str] = list(range(hop_record, n_records))
    if with_failed_step:
        run.append("failed_step")
    return run


def _failed(
    label: str, mode: str, variant: str, wall: float, exc: Exception, segment: int
) -> PointResult:
    return PointResult(
        label, mode, variant, None, None, None, None, None, None, wall, str(exc), segment
    )


def solve_path(
    symbols: Sequence[str],
    charge: int,
    geometries: Sequence[np.ndarray],
    labels: Sequence[str],
    electronic: ElectronicSection,
    *,
    mode: str,
    variant: str,
    threads: int,
    memory_mb: int,
    scratch: Path,
    segment: int = 0,
) -> list[PointResult]:
    """Solve every geometry; results come back in the order of `geometries` for every mode."""
    import pyscf.fci.direct_spin1
    import pyscf.gto

    from gqteashmd.electronic.base import MoleculeSpec, Request
    from gqteashmd.electronic.pyscf_backend import PySCFBackend
    from gqteashmd.errors import ElectronicStructureError

    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {MODES}")
    if len(geometries) != len(labels):
        raise ValueError("geometries and labels differ in length")
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    backend = PySCFBackend(
        MoleculeSpec(tuple(symbols), charge, 1),
        electronic,
        log_file=scratch / "electronic.log",
        threads=threads,
        memory_mb=memory_mb,
        scratch_dir=scratch,
    )
    order = list(range(len(geometries)))
    if mode == "backward":
        order.reverse()
    results: dict[int, PointResult] = {}
    previous: tuple[Any, np.ndarray] | None = None  # (mol, active orbitals) of the last success
    for i in order:
        if mode == "fresh":
            backend.reset()
        started = time.perf_counter()
        try:
            res = backend.compute(
                Request(
                    np.asarray(geometries[i], dtype=float), gradients=frozenset(), overlaps=True
                )
            )
        except ElectronicStructureError as exc:
            results[i] = _failed(
                labels[i], mode, variant, time.perf_counter() - started, exc, segment
            )
            continue
        live = backend._live
        mc = live.solver
        c_act = np.asarray(mc.mo_coeff)[:, mc.ncore : mc.ncore + mc.ncas]
        occupations = [
            natural_occupations(pyscf.fci.direct_spin1.make_rdm1(ci, mc.ncas, mc.nelecas)).tolist()
            for ci in mc.ci
        ]
        overlap = None
        if previous is not None and mode != "fresh":
            s_cross = pyscf.gto.intor_cross("int1e_ovlp", previous[0], live.mol)
            overlap = active_orbital_overlap(previous[1], c_act, s_cross).tolist()
        previous = (live.mol, c_act)
        spins = [res.spin_squares.get(s) for s in backend.states]
        sv = res.meta.get("overlap_min_singular_value")
        results[i] = PointResult(
            label=labels[i],
            mode=mode,
            variant=variant,
            energies_eV=[float(res.energies[s]) * HARTREE_IN_EV for s in backend.states],
            spin_squares=[None if x is None else float(x) for x in spins],  # type: ignore[misc]
            occupations=occupations,
            active_overlap=overlap,
            state_overlap_min_sv=None if sv is None else float(sv),
            rung=res.meta.get("casscf"),
            wall_time_s=time.perf_counter() - started,
            error=None,
            segment=segment,
        )
    return [results[i] for i in range(len(geometries))]


def _fresh_energies(args: tuple) -> list[float]:
    symbols, charge, q, electronic, threads, memory_mb, scratch = args
    point = solve_path(
        symbols,
        charge,
        [np.asarray(q)],
        ["d"],
        ElectronicSection.model_validate(electronic),
        mode="fresh",
        variant="base",
        threads=threads,
        memory_mb=memory_mb,
        scratch=Path(scratch),
    )[0]
    if point.error is not None or point.energies_eV is None:
        raise RuntimeError(point.error)
    return point.energies_eV


def determinism(
    symbols, charge, geometry, electronic, *, threads, memory_mb, scratch
) -> dict[str, Any]:
    """Two fresh solves in two separate (spawned) processes."""
    outputs: list[list[float]] = []
    try:
        for k in range(2):
            args = (
                tuple(symbols),
                charge,
                np.asarray(geometry),
                electronic.model_dump(mode="json"),
                threads,
                memory_mb,
                str(Path(scratch) / f"determinism_{k}"),
            )
            with ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                max_tasks_per_child=1,
            ) as pool:
                outputs.append(pool.submit(_fresh_energies, args).result())
    except Exception as exc:  # noqa: BLE001 - a report tool records the failure
        return {"error": f"{type(exc).__name__}: {exc}"}
    a, b = np.asarray(outputs[0]), np.asarray(outputs[1])
    return {"energies_eV": outputs, "max_abs_diff_eV": float(np.max(np.abs(a - b)))}


def convergence_trials(
    symbols, charge, geometry, electronic, *, threads, memory_mb, scratch
) -> list[dict[str, Any]]:
    out = []
    for k, (tol, tol_grad) in enumerate(CONVERGENCE_TRIALS):
        merged = deep_merge(
            electronic.model_dump(mode="json"),
            {"casscf": {"conv_tol": tol, "conv_tol_grad": tol_grad}},
        )
        entry: dict[str, Any] = {"conv_tol": tol, "conv_tol_grad": tol_grad}
        try:
            section = ElectronicSection.model_validate(merged)
            p = solve_path(
                symbols,
                charge,
                [np.asarray(geometry)],
                ["trial"],
                section,
                mode="fresh",
                variant="base",
                threads=threads,
                memory_mb=memory_mb,
                scratch=Path(scratch) / f"trial_{k}",
            )[0]
            entry.update(
                error=p.error, energies_eV=p.energies_eV, rung=p.rung, wall_time_s=p.wall_time_s
            )
        except ValueError as exc:
            entry.update(error=str(exc), energies_eV=None, rung=None, wall_time_s=0.0)
        out.append(entry)
    return out


def _active_gradient(inputs, records, rows, labels, electronic, *, threads, memory_mb, scratch):
    """Active-state gradient (Eh/bohr, flat) at the last of `records`, one continued backend."""
    from gqteashmd.core.states import StateLabel
    from gqteashmd.electronic.base import MoleculeSpec, Request
    from gqteashmd.electronic.pyscf_backend import PySCFBackend

    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    backend = PySCFBackend(
        MoleculeSpec(inputs.symbols, inputs.charge, 1),
        electronic,
        log_file=scratch / "electronic.log",
        threads=threads,
        memory_mb=memory_mb,
        scratch_dir=scratch,
    )
    label = StateLabel.parse(labels[int(rows["active"][records[-1]])])
    for i in records:
        last = i == records[-1]
        res = backend.compute(
            Request(
                np.asarray(rows["q"][i]).reshape(-1, 3),
                gradients=frozenset({label}) if last else frozenset(),
                overlaps=True,
            )
        )
    return np.asarray(res.gradients[label], dtype=float).reshape(-1)


def replay_trajectory(
    project_dir: Path,
    task_id: str,
    *,
    window: int,
    threads: int,
    memory_mb: int,
    scratch: Path,
    trajectories: Path | None = None,
    base_variant: str = "base",
    from_first_hop: bool = False,
) -> dict[str, Any]:
    inputs = load_project_inputs(project_dir)
    assert inputs.config.electronic is not None and inputs.config.dynamics is not None
    project_electronic = inputs.config.electronic.model_dump(mode="json")
    by_name = {v.name: v for v in REPLAY_VARIANTS}
    if base_variant not in by_name:
        raise ValueError(f"base variant {base_variant!r} is not one of {sorted(by_name)}")
    base = ElectronicSection.model_validate(
        deep_merge(project_electronic, by_name[base_variant].electronic)
    )
    traj = next((d for d in trajectory_dirs(project_dir, trajectories) if d.name == task_id), None)
    if traj is None:
        raise ValueError(f"{task_id}: no such trajectory directory")
    report = trajectory_report(traj)
    rows, labels, _ = read_rows(traj)
    end = failure_record(report)
    scratch = Path(scratch)
    outcome = report["outcome"]
    last = int(rows.size) - 1
    failing = outcome in ("overlap_failure", "electronic_failure")
    if from_first_hop:
        if not report["hops"]:
            raise ValueError(f"{task_id}: the trajectory has no hop")
        hop = report["hops"][0]
        hop_record = int(hop["record"] if isinstance(hop, dict) else hop.record)
        runs = [hop_to_end_segment(int(rows.size), hop_record, failing)]
    else:
        runs = segments(int(rows.size), end, last, window, failing)
    gradient_error = None
    q_failed = None
    if failing:
        try:
            g = _active_gradient(
                inputs,
                probe_window(int(rows.size), last, window),
                rows,
                labels,
                base,
                threads=threads,
                memory_mb=memory_mb,
                scratch=scratch / task_id / "gradient",
            )
            dyn = inputs.config.dynamics
            dt_min = dyn.dt_fs / AU_TIME_IN_FS / 2**dyn.max_step_halvings
            masses = np.repeat(inputs.masses_amu * DALTON_IN_ME, 3)
            q_failed = failed_geometry(
                np.asarray(rows["q"][last], dtype=float),
                np.asarray(rows["v"][last], dtype=float),
                g,
                masses,
                dt_min,
            ).reshape(-1, 3)
        except Exception as exc:  # noqa: BLE001 - recorded, the other points still run
            gradient_error = f"{type(exc).__name__}: {exc}"
    seg_geoms: list[list[np.ndarray]] = []
    seg_labels: list[list[str]] = []
    for run in runs:
        geoms, labs = [], []
        for item in run:
            if item == "failed_step":
                if q_failed is None:
                    continue
                geoms.append(q_failed)
                labs.append("failed_step")
            else:
                geoms.append(np.asarray(rows["q"][item], dtype=float).reshape(-1, 3))
                labs.append(f"r{item}")
        seg_geoms.append(geoms)
        seg_labels.append(labs)
    records = [i for run in runs for i in run if isinstance(i, int)]
    point_labels = [lab for labs in seg_labels for lab in labs]
    points: dict[str, Any] = {}
    for variant in REPLAY_VARIANTS:
        try:
            section = ElectronicSection.model_validate(
                base.model_dump(mode="json")
                if variant.name == "base"
                else deep_merge(project_electronic, variant.electronic)
            )
        except ValueError as exc:
            points[variant.name] = {"error": str(exc)}
            continue
        points[variant.name] = {
            mode: [
                point
                for k, (geoms, labs) in enumerate(zip(seg_geoms, seg_labels, strict=True))
                for point in solve_path(
                    inputs.symbols,
                    inputs.charge,
                    geoms,
                    labs,
                    section,
                    mode=mode,
                    variant=variant.name,
                    threads=threads,
                    memory_mb=memory_mb,
                    scratch=scratch / task_id / variant.name / mode / f"seg{k}",
                    segment=k,
                )
            ]
            for mode in MODES
        }
    det = determinism(
        inputs.symbols,
        inputs.charge,
        np.asarray(rows["q"][end], dtype=float).reshape(-1, 3),
        base,
        threads=threads,
        memory_mb=memory_mb,
        scratch=scratch / task_id / "determinism",
    )
    trials: list[dict[str, Any]] = []
    if outcome == "electronic_failure" and q_failed is not None:
        trials = convergence_trials(
            inputs.symbols,
            inputs.charge,
            q_failed,
            base,
            threads=threads,
            memory_mb=memory_mb,
            scratch=scratch / task_id / "trials",
        )
    return {
        "task": task_id,
        "outcome": outcome,
        "onset": report["onset"],
        "records": records,
        "labels": point_labels,
        "segments": [[x for x in run] for run in runs],
        "stored_energies_eV": [(rows["energies"][i] * HARTREE_IN_EV).tolist() for i in records],
        "failed_geometry_error": gradient_error,
        "points": points,
        "determinism": det,
        "convergence_trials": trials,
    }


def _replay_one(args: tuple) -> tuple[str, dict[str, Any]]:
    project, task, window, threads, memory_mb, scratch, trajectories, base_variant, hop = args
    try:
        return task, replay_trajectory(
            Path(project),
            task,
            window=window,
            threads=threads,
            memory_mb=memory_mb,
            scratch=Path(scratch),
            trajectories=None if trajectories is None else Path(trajectories),
            base_variant=base_variant,
            from_first_hop=hop,
        )
    except ValueError as exc:
        if not hop:
            raise
        return task, {"task": task, "error": str(exc)}  # e.g. no hop in the trajectory


def _lost_tasks(project: Path, trajectories: Path | None = None) -> list[str]:
    out = []
    for d in trajectory_dirs(project, trajectories):
        if (d / "output.h5").is_file() and trajectory_report(d)["outcome"] not in LOST_EXCLUDED:
            out.append(d.name)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("project", type=Path)
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--all-lost", action="store_true")
    ap.add_argument("--out", type=Path, required=True, help="directory, one <task>.json each")
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--memory-mb", type=int, default=800)
    ap.add_argument("--scratch", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument(
        "--trajectories",
        type=Path,
        default=None,
        help="trajectory directories of a run_variants output, in place of the project's tasks",
    )
    ap.add_argument(
        "--base-variant",
        default="base",
        help="the REPLAY_VARIANTS entry whose [electronic] is the run's own (default: base)",
    )
    ap.add_argument(
        "--from-first-hop",
        action="store_true",
        help="replay one segment from the first hop record to the end (and the failed step)",
    )
    args = ap.parse_args(argv)
    project = args.project.expanduser()
    trajectories = args.trajectories.expanduser() if args.trajectories else None
    tasks = list(args.tasks) + (_lost_tasks(project, trajectories) if args.all_lost else [])
    if not tasks:
        ap.error("give task ids or --all-lost")
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    scratch = (args.scratch or out / "scratch").expanduser()
    jobs = [
        (
            str(project),
            t,
            args.window,
            args.threads,
            args.memory_mb,
            str(scratch),
            None if trajectories is None else str(trajectories),
            args.base_variant,
            args.from_first_hop,
        )
        for t in tasks
    ]

    def write(task: str, result: dict[str, Any]) -> None:
        (out / f"{task}.json").write_text(
            json.dumps(jsonable(result), indent=1, allow_nan=False), encoding="utf-8"
        )
        print(f"wrote {out / (task + '.json')}", flush=True)

    if args.workers > 1:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for task, result in pool.map(_replay_one, jobs):
                write(task, result)
    else:
        for job in jobs:
            write(*_replay_one(job))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
