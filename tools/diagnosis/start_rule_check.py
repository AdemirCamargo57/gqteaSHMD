"""Start-rule check (CASSCF start rule, Task 5): the rule's step-0 solution at each start.

For the SA3 settings of a project, each start geometry is solved by a fresh PySCFBackend whose
reference is the project's equilibrium geometry, and the result is matched against the
start-solution survey (sol0 = the survey's lowest-SA solution). Report only.

Example (WSL):
  python tools/diagnosis/start_rule_check.py ~/p8_m4/run \
      --survey ~/p9a/plan2/start_solutions.json --out ~/start_rule/start_rule_check.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from forensics import jsonable  # noqa: E402
from project_inputs import load_project_inputs  # noqa: E402
from run_variants import deep_merge  # noqa: E402
from start_solutions import N_STATES, _equilibrium, match_solution  # noqa: E402

from gqteashmd.config.schema import ElectronicSection  # noqa: E402


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    errors = [r["task"] for r in rows if "error" in r]
    ok = [r for r in rows if "error" not in r]
    other = [r["task"] for r in ok if r.get("solution") not in (0, None)]
    unmatched = [r["task"] for r in ok if r.get("solution") is None]
    return {
        "n_starts": len(rows),
        "n_sol0": sum(1 for r in ok if r.get("solution") == 0),
        "n_other": len(other),
        "n_unmatched": len(unmatched),
        "n_errors": len(errors),
        "other": other,
        "unmatched": unmatched,
        "errors": errors,
    }


def check_start(job: tuple) -> dict[str, Any]:
    project, task, survey_solutions, memory_mb = job
    from gqteashmd.electronic.base import MoleculeSpec, Request
    from gqteashmd.electronic.pyscf_backend import PySCFBackend

    row: dict[str, Any] = {"task": task}
    try:
        inputs = load_project_inputs(Path(project))
        base = inputs.config.electronic
        assert base is not None
        electronic = ElectronicSection.model_validate(
            deep_merge(base.model_dump(mode="json"), {"singlets": N_STATES})
        )
        q_eq, _ = _equilibrium(Path(project))
        start = next(s for s in inputs.starts if s.task_id == task)
        q = np.asarray(start.q, dtype=float).reshape(-1, 3)
        backend = PySCFBackend(
            MoleculeSpec(tuple(inputs.symbols), inputs.charge, 1),
            electronic,
            memory_mb=memory_mb,
            reference_coords_bohr=q_eq,
        )
        started = time.perf_counter()
        result = backend.compute(Request(q))
        row["wall_time_s"] = time.perf_counter() - started
        energies = [float(result.energies[label]) for label in backend.states[:N_STATES]]
        row.update(
            energies_Eh=energies,
            rung=result.meta["casscf"],
            casscf_reference=result.meta["casscf_reference"],
            reference_active_min_sv=result.meta.get("reference_active_min_sv"),
            solution=match_solution(energies, survey_solutions),
        )
    except Exception as exc:  # noqa: BLE001 - a report tool records the failure
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def reference_cost(project: Path, memory_mb: int) -> dict[str, Any]:
    """Wall time of one reference search at the equilibrium geometry (SA3 settings)."""
    from gqteashmd.electronic import pyscf_casscf
    from gqteashmd.electronic.base import MoleculeSpec
    from gqteashmd.electronic.pyscf_backend import PySCFBackend

    inputs = load_project_inputs(project)
    base = inputs.config.electronic
    assert base is not None
    electronic = ElectronicSection.model_validate(
        deep_merge(base.model_dump(mode="json"), {"singlets": N_STATES})
    )
    q_eq, _ = _equilibrium(project)
    backend = PySCFBackend(
        MoleculeSpec(tuple(inputs.symbols), inputs.charge, 1), electronic, memory_mb=memory_mb
    )
    mol = backend._build_mol(q_eq, None)
    started = time.perf_counter()
    mf, _ = backend._run_scf(mol, None)
    reference, problems = pyscf_casscf.find_reference(mol, mf, electronic, backend.states)
    return {
        "wall_time_s": time.perf_counter() - started,
        "reference": None if reference is None else reference.meta(),
        "problems": problems,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("project", type=Path)
    ap.add_argument("--survey", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--memory-mb", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    project = args.project.expanduser()
    survey = json.loads(args.survey.expanduser().read_text(encoding="utf-8"))["tasks"]
    inputs = load_project_inputs(project)
    jobs = [
        (str(project), s.task_id, survey[s.task_id]["solutions"], args.memory_mb)
        for s in inputs.starts
    ]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    if args.workers > 1:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            rows = list(pool.map(check_start, jobs))
    else:
        rows = [check_start(job) for job in jobs]
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "summary": summarize(rows),
        "reference_cost": reference_cost(project, args.memory_mb),
        "rows": rows,
    }
    out.write_text(json.dumps(jsonable(report), indent=1, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
