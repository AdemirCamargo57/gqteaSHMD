"""Our TrajectoryRunner on a project's fixed initial conditions, for setup variants (P9a 3.3).

The orchestrator is bypassed on purpose: a changed [electronic] section would change the S0
surface and so the samples. Each trajectory uses the stage's random stream
task_rng(project.seed, "dynamics", index), so the variant "base" reproduces the project.

Example (WSL):
  python tools/diagnosis/run_variants.py ~/p8_m4/run tools/diagnosis/variants_p9a.toml \\
      --out ~/p9a/variants --workers 8 --threads 2 --memory-mb 800 --variant base
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import sys
import time
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forensics import ONSETS, jsonable, trajectory_report  # noqa: E402
from project_inputs import ProjectInputs, load_project_inputs  # noqa: E402

from gqteashmd.config.schema import (  # noqa: E402
    DynamicsStageSection,
    ElectronicSection,
    ShmdConfig,
)

KEPT_OUTCOMES = ("finished", "stopped_in_s0")
KEPT_JUMP_EV = 0.1
_NAME = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class Variant:
    name: str
    electronic: Mapping[str, Any] = field(default_factory=dict)
    dynamics: Mapping[str, Any] = field(default_factory=dict)


def deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def read_variants(path: Path) -> list[Variant]:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for entry in data.get("variant", []):
        name = str(entry.get("name", ""))
        if not _NAME.fullmatch(name):
            raise ValueError(f"{path}: variant name {name!r} must match [a-z0-9_]+")
        unknown = set(entry) - {"name", "electronic", "dynamics", "description"}
        if unknown:
            raise ValueError(f"{path}: variant {name!r} has unknown keys {sorted(unknown)}")
        out.append(
            Variant(name, dict(entry.get("electronic", {})), dict(entry.get("dynamics", {})))
        )
    if len({v.name for v in out}) != len(out):
        raise ValueError(f"{path}: duplicate variant names")
    return out


def variant_sections(
    config: ShmdConfig, variant: Variant
) -> tuple[ElectronicSection, DynamicsStageSection]:
    assert config.electronic is not None and config.dynamics is not None
    electronic = ElectronicSection.model_validate(
        deep_merge(config.electronic.model_dump(mode="json"), variant.electronic)
    )
    dynamics = DynamicsStageSection.model_validate(
        deep_merge(config.dynamics.model_dump(mode="json"), variant.dynamics)
    )
    return electronic, dynamics


def is_kept(outcome: str, max_jump: float) -> bool:
    return outcome in KEPT_OUTCOMES and max_jump <= KEPT_JUMP_EV


@dataclass(frozen=True)
class Job:
    directory: str
    task_id: str
    index: int
    sample: int
    state: str
    q: np.ndarray
    v: np.ndarray
    symbols: tuple[str, ...]
    charge: int
    multiplicity: int
    masses_amu: np.ndarray
    electronic: dict[str, Any]
    dynamics: dict[str, Any]
    seed: int
    threads: int
    memory_mb: int
    backend: str


def _backend(job: Job, directory: Path):
    if job.backend == "pyscf":
        from gqteashmd.electronic.base import MoleculeSpec
        from gqteashmd.electronic.pyscf_backend import PySCFBackend

        return PySCFBackend(
            MoleculeSpec(job.symbols, job.charge, job.multiplicity),
            ElectronicSection.model_validate(job.electronic),
            log_file=directory / "electronic.log",
            threads=job.threads,
            memory_mb=job.memory_mb,
            scratch_dir=directory / "scratch",
        )
    if job.backend.startswith("lvc:"):
        from gqteashmd.dynamics.lvc import LVCModel
        from gqteashmd.testing.fake_backend import ModelBackend

        return ModelBackend(LVCModel.from_template(Path(job.backend[4:]), spin_orbit=False))
    raise ValueError(f"unknown backend {job.backend!r}")


def run_job(job: Job) -> dict[str, Any]:
    """One trajectory, as DynamicsStage.run_task would run it (top level: picklable)."""
    from gqteashmd.analysis.energy import max_jump_eV
    from gqteashmd.core.states import StateLabel
    from gqteashmd.core.units import DALTON_IN_ME
    from gqteashmd.dynamics.randomness import GeneratorSource
    from gqteashmd.stages.dynamics import run_settings
    from gqteashmd.trajectory.model import BackendModel
    from gqteashmd.trajectory.runner import OUTPUT_FILE, InitialConditions, TrajectoryRunner
    from gqteashmd.trajectory.store import read_output_h5
    from gqteashmd.util.seeds import task_rng

    directory = Path(job.directory)
    directory.mkdir(parents=True, exist_ok=True)
    section = DynamicsStageSection.model_validate(job.dynamics)
    settings = run_settings(section)
    started = time.perf_counter()
    row: dict[str, Any] = {"task": job.task_id, "sample": job.sample, "index": job.index}
    try:
        model = BackendModel(
            _backend(job, directory),
            np.repeat(np.asarray(job.masses_amu) * DALTON_IN_ME, 3),
            coupling=section.coupling,
        )
        active = model.labels.index(StateLabel.parse(job.state))
        result = TrajectoryRunner(
            model,
            settings,
            directory,
            attrs={
                "symbols": list(job.symbols),
                "sample": job.sample,
                "initial_state": job.state,
                "task": job.task_id,
            },
        ).run(
            InitialConditions(np.asarray(job.q), np.asarray(job.v), active),
            GeneratorSource(task_rng(job.seed, "dynamics", job.index)),
        )
    except ValueError as exc:  # what the stage turns into a non-retryable TaskError
        result = {"outcome": "error", "message": f"{type(exc).__name__}: {exc}"}
    result["wall_time_s"] = time.perf_counter() - started
    (directory / "result.json").write_text(json.dumps({"status": "ok", "result": jsonable(result)}))
    row.update(outcome=str(result["outcome"]), message=result.get("message"))
    for key in ("t_final_fs", "n_steps", "n_hops", "n_rejections", "n_flagged", "max_drift_eV"):
        row[key] = result.get(key)
    row["wall_time_s"] = result["wall_time_s"]
    row["s_per_step"] = result["wall_time_s"] / max(1, int(result.get("n_steps") or 0))
    output = directory / OUTPUT_FILE
    if output.is_file():
        data = read_output_h5(output)
        mask = np.asarray(data["grid_step"]) <= settings.n_grid_steps
        row["max_jump_eV"] = max_jump_eV(np.asarray(data["e_total"])[mask])
        forensic = trajectory_report(directory)
        row["onset"] = forensic["onset"]
        row["first_jump_fs"] = forensic["jumps"][0]["time_fs"] if forensic["jumps"] else None
        row["first_hop_fs"] = forensic["hops"][0]["time_fs"] if forensic["hops"] else None
    else:
        row.update(max_jump_eV=None, onset="none", first_jump_fs=None, first_hop_fs=None)
    row["kept"] = row["max_jump_eV"] is not None and is_kept(row["outcome"], row["max_jump_eV"])
    return row


def run_variant(
    inputs: ProjectInputs,
    variant: Variant,
    out_dir: Path,
    *,
    workers: int,
    threads: int,
    memory_mb: int,
    backend: str = "pyscf",
    only: Sequence[str] | None = None,
) -> dict[str, Any]:
    electronic, dynamics = variant_sections(inputs.config, variant)
    base = Path(out_dir).expanduser() / variant.name
    base.mkdir(parents=True, exist_ok=True)
    jobs = [
        Job(
            str(base / s.task_id),
            s.task_id,
            s.index,
            s.sample,
            s.state,
            s.q,
            s.v,
            inputs.symbols,
            inputs.charge,
            inputs.multiplicity,
            inputs.masses_amu,
            electronic.model_dump(mode="json"),
            dynamics.model_dump(mode="json"),
            inputs.config.project.seed,
            threads,
            memory_mb,
            backend,
        )
        for s in inputs.starts
        if only is None or s.task_id in only
    ]
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(threads)  # inherited by the spawned workers
    os.environ["PYSCF_MAX_MEMORY"] = str(memory_mb)
    started = time.perf_counter()
    if workers <= 1:
        rows = [run_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            rows = list(pool.map(run_job, jobs))
    rows.sort(key=lambda r: r["task"])
    onsets = Counter(r["onset"] for r in rows)
    summary = {
        "variant": variant.name,
        "electronic": electronic.model_dump(mode="json"),
        "dynamics": dynamics.model_dump(mode="json"),
        "project": str(inputs.project_dir),
        "seed": inputs.config.project.seed,
        "backend": backend,
        "n_trajectories": len(rows),
        "outcomes": dict(sorted(Counter(r["outcome"] for r in rows).items())),
        "onsets": {o: int(onsets.get(o, 0)) for o in ONSETS},
        "n_kept": sum(1 for r in rows if r["kept"]),
        "wall_time_s": time.perf_counter() - started,
        "trajectories": rows,
    }
    (base / "summary.json").write_text(
        json.dumps(jsonable(summary), indent=1, allow_nan=False) + "\n"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("project", type=Path)
    parser.add_argument("variants", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", action="append", help="run only these variants (repeatable)")
    parser.add_argument("--task", action="append", help="run only these tasks (repeatable)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-mb", type=int, default=800)
    args = parser.parse_args(argv)
    inputs = load_project_inputs(args.project)
    variants = read_variants(args.variants)
    if args.variant:
        unknown = sorted(set(args.variant) - {v.name for v in variants})
        if unknown:
            parser.error(f"unknown variants {unknown}")
        variants = [v for v in variants if v.name in args.variant]
    for variant in variants:
        summary = run_variant(
            inputs,
            variant,
            args.out,
            workers=args.workers,
            threads=args.threads,
            memory_mb=args.memory_mb,
            only=args.task,
        )
        print(
            f"{variant.name}: kept {summary['n_kept']}/{summary['n_trajectories']} "
            f"outcomes {summary['outcomes']} in {summary['wall_time_s']:.0f} s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
