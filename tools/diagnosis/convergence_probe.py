"""F4 convergence probe (P9a plan 2, Task 4, owner decision 5): do the two extra CASSCF rungs help?

At three M4 failure geometries (traj_00000 at the failed step, traj_00007 at its first full
step, traj_00016 at SHARC's 17.0 fs record) each is solved along a short path (so "previous
orbitals" has orbitals) in four configurations: default, more_iterations, loose, both. The
accuracy of the loose rung is measured at five geometries where the default ladder converges
(energies and the active-state gradient at the nominal tolerances against conv_tol 1e-8 and
conv_tol_grad 1e-4). The conditions, fixed before any run:
  - more_iterations is enabled if its configuration converges at >= 1 of the 3 geometries with
    the rung "previous orbitals, more iterations";
  - loose is enabled if its configuration converges at >= 2 of 3 with the rung "loose
    tolerances" and over the 5 accuracy points max |dE| <= 1e-6 Eh and max |dg| <= 1e-4 Eh/bohr.
A report tool: failures are recorded, nothing is asserted.

Example (WSL):
  python tools/diagnosis/convergence_probe.py ~/p8_m4/run --sharc ~/p9a/sharc/base \
      --out ~/p9a/plan2/convergence_probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sharc_compare"))
from forensics import failed_geometry, find_stage_dir, jsonable, read_rows  # noqa: E402
from project_inputs import load_project_inputs  # noqa: E402
from replay_points import _active_gradient  # noqa: E402
from run_variants import deep_merge  # noqa: E402

from gqteashmd.config.schema import ElectronicSection  # noqa: E402
from gqteashmd.core.units import AU_TIME_IN_FS, DALTON_IN_ME, HARTREE_IN_EV  # noqa: E402
from gqteashmd.electronic.pyscf_casscf import EXTRA_RUNG_NAMES  # noqa: E402

MORE_ITERATIONS_RUNG, LOOSE_RUNG = EXTRA_RUNG_NAMES
GEOMETRY_NAMES = ("traj_00000_failed_step", "traj_00007_full_step", "traj_00016_sharc_17fs")
CONFIGURATIONS = {
    "default": (),
    "more_iterations": (MORE_ITERATIONS_RUNG,),
    "loose": (LOOSE_RUNG,),
    "both": tuple(EXTRA_RUNG_NAMES),
}
LOOSE_TOLERANCES = {"conv_tol": 1e-8, "conv_tol_grad": 1e-4}
MAX_DE_EH = 1e-6
MAX_DG_EH_PER_BOHR = 1e-4
ACCURACY_POINTS = (  # (trajectory, record), record -1 = the last
    ("traj_00000", -1),
    ("traj_00007", -1),
    ("traj_00016", -1),
    ("traj_00001", 0),
    ("traj_00002", 0),
)
SHARC_TIMES_FS = (16.0, 16.5, 17.0)


def max_abs_diff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def f4_decision(probe: dict[str, Any]) -> dict[str, Any]:
    """Apply the owner-decision-5 conditions to a probe result."""
    geoms = probe["geometries"]

    def count(config: str, rung: str) -> int:
        return sum(1 for g in geoms.values() if g[config].get("rung") == rung)

    n_more = count("more_iterations", MORE_ITERATIONS_RUNG)
    more = n_more >= 1
    n_loose = count("loose", LOOSE_RUNG)
    acc = probe.get("accuracy", [])
    dE = max((a["max_abs_dE_Eh"] for a in acc), default=float("nan"))
    dg = max((a["max_abs_dg_Eh_per_bohr"] for a in acc), default=float("nan"))
    accurate = bool(acc) and dE <= MAX_DE_EH and dg <= MAX_DG_EH_PER_BOHR
    loose = n_loose >= 2 and accurate
    reasons = [
        f"more_iterations converged with its rung at {n_more} of {len(geoms)} (need 1)",
        f"loose converged with its rung at {n_loose} of {len(geoms)} (need 2)",
        f"loose accuracy over {len(acc)} points: max |dE| {dE:.3g} Eh (<= {MAX_DE_EH:g}), "
        f"max |dg| {dg:.3g} Eh/bohr (<= {MAX_DG_EH_PER_BOHR:g}): {accurate}",
    ]
    enabled = [name for name, on in ((MORE_ITERATIONS_RUNG, more), (LOOSE_RUNG, loose)) if on]
    return {"more_iterations": more, "loose": loose, "enabled": enabled, "reasons": reasons}


def _sharc_geometries(sharc_dir: Path, task: str) -> list[np.ndarray]:
    from run_sharc_lvc import parse_lvc_output_dat
    from run_sharc_pyscf_dynamics import trajectory_data

    data = trajectory_data(parse_lvc_output_dat(Path(sharc_dir) / task / "output.dat"))
    t_fs = np.asarray(data["steps"], dtype=float) * float(data["dt_au"]) * AU_TIME_IN_FS
    out = []
    for target in SHARC_TIMES_FS:
        k = int(np.argmin(np.abs(t_fs - target)))
        if abs(t_fs[k] - target) > 1e-3:
            raise ValueError(f"{task}: no SHARC record at {target} fs (nearest {t_fs[k]:.4f} fs)")
        out.append(np.asarray(data["q"][k], dtype=float).reshape(-1, 3))
    return out


def probe_geometries(
    project: Path, sharc_dir: Path, *, threads: int, memory_mb: int, scratch: Path
) -> dict[str, tuple[list[np.ndarray], np.ndarray]]:
    """name -> (path geometries, probe geometry), all (n_atoms, 3) bohr."""
    inputs = load_project_inputs(project)
    dyn, base = inputs.config.dynamics, inputs.config.electronic
    assert dyn is not None and base is not None
    stage = find_stage_dir(project, "dynamics") / "tasks"
    masses = np.repeat(inputs.masses_amu * DALTON_IN_ME, 3)
    dt = dyn.dt_fs / AU_TIME_IN_FS
    dt_min = dt / 2**dyn.max_step_halvings
    out: dict[str, tuple[list[np.ndarray], np.ndarray]] = {}

    def verlet(task: str, records: list[int], step: float) -> tuple[list[np.ndarray], np.ndarray]:
        rows, labels, _ = read_rows(stage / task)
        g = _active_gradient(
            inputs,
            records,
            rows,
            labels,
            base,
            threads=threads,
            memory_mb=memory_mb,
            scratch=scratch / task / "gradient",
        )
        last = records[-1]
        q_probe = failed_geometry(
            np.asarray(rows["q"][last], dtype=float),
            np.asarray(rows["v"][last], dtype=float),
            g,
            masses,
            step,
        )
        path = [np.asarray(rows["q"][i], dtype=float).reshape(-1, 3) for i in records]
        return path, q_probe.reshape(-1, 3)

    rows0, _, _ = read_rows(stage / "traj_00000")
    last = int(rows0.size) - 1
    out["traj_00000_failed_step"] = verlet("traj_00000", [last - 2, last - 1, last], dt_min)
    out["traj_00007_full_step"] = verlet("traj_00007", [0], dt)
    sharc = _sharc_geometries(sharc_dir, "traj_00016")
    out["traj_00016_sharc_17fs"] = (sharc[:2], sharc[2])
    return out


def solve_configuration(
    inputs, electronic, path, probe_q, rungs, *, threads, memory_mb, scratch
) -> dict[str, Any]:
    """A fresh backend, the path continued, then the probe geometry."""
    from gqteashmd.electronic.base import MoleculeSpec, Request
    from gqteashmd.electronic.pyscf_backend import PySCFBackend

    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        backend = PySCFBackend(
            MoleculeSpec(tuple(inputs.symbols), inputs.charge, 1),
            electronic,
            log_file=scratch / "electronic.log",
            threads=threads,
            memory_mb=memory_mb,
            scratch_dir=scratch,
            casscf_extra_rungs=tuple(rungs),
        )
        res = None
        for q in [*path, probe_q]:
            res = backend.compute(
                Request(np.asarray(q, dtype=float), gradients=frozenset(), overlaps=True)
            )
        assert res is not None
        return {
            "rung": res.meta.get("casscf"),
            "energies_eV": [float(res.energies[s]) * HARTREE_IN_EV for s in backend.states],
            "wall_time_s": time.perf_counter() - started,
        }
    except Exception as exc:  # noqa: BLE001 - a report tool records the failure
        return {
            "rung": None,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_time_s": time.perf_counter() - started,
        }


def _energies_and_gradient(inputs, electronic, q, label, *, threads, memory_mb, scratch):
    from gqteashmd.core.states import StateLabel
    from gqteashmd.electronic.base import MoleculeSpec, Request
    from gqteashmd.electronic.pyscf_backend import PySCFBackend

    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    backend = PySCFBackend(
        MoleculeSpec(tuple(inputs.symbols), inputs.charge, 1),
        electronic,
        log_file=scratch / "electronic.log",
        threads=threads,
        memory_mb=memory_mb,
        scratch_dir=scratch,
    )
    state = StateLabel.parse(label)
    res = backend.compute(
        Request(np.asarray(q, dtype=float), gradients=frozenset({state}), overlaps=False)
    )
    energies = np.array([float(res.energies[s]) for s in backend.states])
    return energies, np.asarray(res.gradients[state], dtype=float).reshape(-1)


def accuracy_point(
    inputs, electronic, q, label, *, threads, memory_mb, scratch, name: str = ""
) -> dict[str, Any]:
    """Energies and the active gradient, fresh, at nominal and at the loose tolerances."""
    entry: dict[str, Any] = {"point": name}
    try:
        loose = ElectronicSection.model_validate(
            deep_merge(electronic.model_dump(mode="json"), {"casscf": LOOSE_TOLERANCES})
        )
        kw = {"threads": threads, "memory_mb": memory_mb}
        e0, g0 = _energies_and_gradient(
            inputs, electronic, q, label, scratch=Path(scratch) / "nominal", **kw
        )
        e1, g1 = _energies_and_gradient(
            inputs, loose, q, label, scratch=Path(scratch) / "loose", **kw
        )
        entry.update(
            max_abs_dE_Eh=max_abs_diff(e0, e1), max_abs_dg_Eh_per_bohr=max_abs_diff(g0, g1)
        )
    except Exception as exc:  # noqa: BLE001 - recorded; the decision sees a missing point
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def run_probe(
    project: Path, sharc_dir: Path, scratch: Path, *, threads: int, memory_mb: int
) -> dict[str, Any]:
    inputs = load_project_inputs(project)
    electronic = inputs.config.electronic
    assert electronic is not None
    kw = {"threads": threads, "memory_mb": memory_mb}
    scratch = Path(scratch)
    geometries: dict[str, Any] = {}
    try:
        planned = probe_geometries(project, sharc_dir, scratch=scratch, **kw)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "geometries": {}, "accuracy": []}
    for name, (path, probe_q) in planned.items():
        geometries[name] = {
            config: solve_configuration(
                inputs, electronic, path, probe_q, rungs, scratch=scratch / name / config, **kw
            )
            for config, rungs in CONFIGURATIONS.items()
        }
    stage = find_stage_dir(project, "dynamics") / "tasks"
    accuracy = []
    for task, record in ACCURACY_POINTS:
        name = f"{task}[{record}]"
        try:
            rows, labels, _ = read_rows(stage / task)
            i = record if record >= 0 else int(rows.size) + record
            q = np.asarray(rows["q"][i], dtype=float).reshape(-1, 3)
            label = labels[int(rows["active"][i])]
        except Exception as exc:  # noqa: BLE001
            accuracy.append({"point": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        accuracy.append(
            accuracy_point(
                inputs, electronic, q, label, scratch=scratch / "accuracy" / name, name=name, **kw
            )
        )
    probe: dict[str, Any] = {
        "geometries": geometries,
        "accuracy": [a for a in accuracy if "error" not in a],
        "accuracy_all": accuracy,
    }
    probe["decision"] = f4_decision(probe)
    return probe


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("project", type=Path, help="the M4 project directory")
    ap.add_argument("--sharc", type=Path, required=True, help="SHARC base run directory")
    ap.add_argument("--out", type=Path, required=True, help="JSON output file")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--memory-mb", type=int, default=1500)
    args = ap.parse_args(argv)
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    result = run_probe(
        args.project.expanduser(),
        args.sharc.expanduser(),
        out.parent / "convergence_probe_scratch",
        threads=args.threads,
        memory_mb=args.memory_mb,
    )
    out.write_text(json.dumps(jsonable(result), indent=2, allow_nan=False))
    print(json.dumps(jsonable(result.get("decision", {})), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
