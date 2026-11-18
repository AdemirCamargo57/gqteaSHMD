"""The P9b paired ensemble comparison (spec 3.2-3.4): kept sets, the paired set, the three
swap rules and the report tables. Pure functions; ``main`` reads a gqteashmd project and a
SHARC ensemble directory (``run_sharc_pyscf_ensemble.py``) and writes one JSON.

Example (WSL):
  python tools/sharc_compare/ensemble_compare.py ~/p9b/ours/run ~/p9b/swap/project \\
      --label swap --asserted --out ~/p9b/compare/swap.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.analysis.energy import max_drift_eV, summary_eV
from gqteashmd.analysis.grid import Trajectory
from gqteashmd.analysis.kinetics import bootstrap_fit
from gqteashmd.analysis.populations import (
    bootstrap_indices,
    classical_populations,
    quantum_populations,
)
from gqteashmd.core.units import HARTREE_IN_EV
from gqteashmd.util.seeds import task_rng

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "diagnosis"))
from sharc_trajectories import (  # noqa: E402
    is_kept,
    read_our_trajectory,
    read_sharc_trajectory,
)

MIN_PAIRED = 30
N_BOOTSTRAP = 1000
SE_FACTOR = 4.0
POPULATION_FLOOR = 0.02
FIT_FACTOR = 4.0
KS_MIN_P = 1e-3
OUTPUT_DT_FS = 1.0
COMPARE_STREAM = "p9b-compare"
START_DIFFERENCE_EV = 1e-3  # report: a step-0 energy difference above this = another solution
RULES = ("populations", "tau", "t0", "first_hop_ks")


def _verdict(rule: str, ok: bool | None, reason: str | None = None, **numbers: Any) -> dict:
    verdict = "inconclusive" if ok is None else ("pass" if ok else "fail")
    return {"rule": rule, "verdict": verdict, "reason": reason, **numbers}


def paired_tasks(
    ours: Mapping[str, Trajectory | None], theirs: Mapping[str, Trajectory | None], n_total: int
) -> list[str]:
    return sorted(k for k in ours if is_kept(ours[k], n_total) and is_kept(theirs.get(k), n_total))


def on_steps(
    trajs: Sequence[Trajectory], steps: np.ndarray, n_states: int
) -> tuple[np.ndarray, np.ndarray]:
    active = np.zeros((len(trajs), steps.size), dtype=np.int64)
    pops = np.zeros((len(trajs), steps.size, n_states))
    for i, t in enumerate(trajs):
        where = {int(s): j for j, s in enumerate(t.grid_step) if int(t.ticks[j]) == 0}
        for k, s in enumerate(steps):
            j = where.get(int(s))
            if j is None:
                raise ValueError(f"{t.task}: no full-step record at grid step {int(s)}")
            active[i, k] = int(t.active[j])
            pops[i, k] = t.populations[j]
    return active, pops


def population_rule(
    active_a: np.ndarray, active_b: np.ndarray, n_states: int, indices: np.ndarray
) -> dict:
    d = classical_populations(active_a, n_states)[0] - classical_populations(active_b, n_states)[0]
    boot = np.stack(
        [
            classical_populations(active_a[idx], n_states)[0]
            - classical_populations(active_b[idx], n_states)[0]
            for idx in indices
        ]
    )
    se = boot.std(axis=0, ddof=1)
    bound = SE_FACTOR * se + POPULATION_FLOOR
    ratio = np.abs(d) / bound
    t_worst, s_worst = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
    return _verdict(
        "populations",
        bool(np.all(np.abs(d) <= bound)),
        max_abs_difference=float(np.abs(d).max()),
        max_se=float(se.max()),
        worst={
            "time_index": int(t_worst),
            "state": int(s_worst),
            "difference": float(d[t_worst, s_worst]),
            "bound": float(bound[t_worst, s_worst]),
        },
        n_points_failing=int(np.count_nonzero(np.abs(d) > bound)),
    )


def fit_rule(t_fs: np.ndarray, per_a: np.ndarray, per_b: np.ndarray, indices: np.ndarray) -> dict:
    fa = bootstrap_fit(t_fs, per_a, indices, dt_out=OUTPUT_DT_FS)
    fb = bootstrap_fit(t_fs, per_b, indices, dt_out=OUTPUT_DT_FS)
    out = {}
    for name, ea, eb, sa, sb in (
        ("tau", fa.fit.tau_fs, fb.fit.tau_fs, fa.tau_std_fs, fb.tau_std_fs),
        ("t0", fa.fit.t0_fs, fb.fit.t0_fs, fa.t0_std_fs, fb.t0_std_fs),
    ):
        numbers = {
            "ours": ea,
            "theirs": eb,
            "std_ours": sa,
            "std_theirs": sb,
            "flags_ours": list(fa.fit.flags),
            "flags_theirs": list(fb.fit.flags),
        }
        if None in (ea, eb, sa, sb):
            out[name] = _verdict(
                name, None, "a fit gave no estimate or no bootstrap std (PR5)", **numbers
            )
            continue
        bound = FIT_FACTOR * float(np.hypot(sa, sb))
        out[name] = _verdict(
            name, abs(ea - eb) <= bound, difference=float(ea - eb), bound=bound, **numbers
        )
    out["verdict"] = (
        "inconclusive"
        if "inconclusive" in (out["tau"]["verdict"], out["t0"]["verdict"])
        else ("pass" if out["tau"]["verdict"] == out["t0"]["verdict"] == "pass" else "fail")
    )
    out["reason"] = out["tau"]["reason"] or out["t0"]["reason"]
    return out


def first_hop_times(trajs: Sequence[Trajectory]) -> list[float]:
    times = []
    for t in trajs:
        hops = np.flatnonzero(t.hop_to >= 0)
        if hops.size:
            times.append(float(t.time_fs[hops[0]]))
    return times


def ks_rule(times_a: Sequence[float], times_b: Sequence[float]) -> dict:
    if len(times_a) < 2 or len(times_b) < 2:
        return _verdict(
            "first_hop_ks",
            None,
            "fewer than 2 first-hop times on a side (PR5)",
            n_ours=len(times_a),
            n_theirs=len(times_b),
        )
    from scipy.stats import ks_2samp

    res = ks_2samp(times_a, times_b)
    p = float(res.pvalue)
    return _verdict(
        "first_hop_ks",
        p >= KS_MIN_P,
        p_value=p,
        statistic=float(res.statistic),
        n_ours=len(times_a),
        n_theirs=len(times_b),
    )


def _hop_pairs(trajs: Sequence[Trajectory]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for t in trajs:
        for j in np.flatnonzero(t.hop_to >= 0):
            c[f"S{int(t.hop_from[j])}->S{int(t.hop_to[j])}"] += 1
    return dict(sorted(c.items()))


def _report(ours, theirs, tasks, paired, n_total, steps, n_states) -> dict[str, Any]:
    present_o = [ours[k] for k in tasks if ours.get(k) is not None]
    present_t = [theirs[k] for k in tasks if theirs.get(k) is not None]
    po = [ours[k] for k in paired]
    pt = [theirs[k] for k in paired]
    quantum = None
    if paired:
        qa = quantum_populations(on_steps(po, steps, n_states)[1])
        qb = quantum_populations(on_steps(pt, steps, n_states)[1])
        quantum = float(np.abs(qa - qb).max())
    start: dict[str, float] = {}  # |E_tot(0) ours - SHARC| per task (eV): the start solution
    for k in tasks:
        a, b = ours.get(k), theirs.get(k)
        if a is None or b is None or not a.e_total.size or not b.e_total.size:
            continue
        start[k] = float(abs(a.e_total[0] - b.e_total[0]) * HARTREE_IN_EV)
    return {
        "kept_ours": sum(is_kept(ours.get(k), n_total) for k in tasks),
        "kept_theirs": sum(is_kept(theirs.get(k), n_total) for k in tasks),
        "outcomes_ours": dict(Counter(t.outcome for t in present_o)),
        "outcomes_theirs": dict(Counter(t.outcome for t in present_t)),
        "missing_ours": [k for k in tasks if ours.get(k) is None],
        "missing_theirs": [k for k in tasks if theirs.get(k) is None],
        "quantum_max_abs_difference": quantum,
        "hop_pairs_ours": _hop_pairs(po),
        "hop_pairs_theirs": _hop_pairs(pt),
        "frustrated_ours": int(sum(int(t.frustrated.sum()) for t in po)),
        "frustrated_theirs": None,  # PR4: output.dat does not record them, so always null
        "drift_eV_ours": summary_eV([max_drift_eV(t.e_total) for t in po]),
        "drift_eV_theirs": summary_eV([max_drift_eV(t.e_total) for t in pt]),
        "start_difference": {
            "per_task_eV": start,
            "n_above": sum(v > START_DIFFERENCE_EV for v in start.values()),
            "threshold_eV": START_DIFFERENCE_EV,
        },
    }


def compare(
    ours: Mapping[str, Trajectory | None],
    theirs: Mapping[str, Trajectory | None],
    *,
    tasks: Sequence[str],
    n_total: int,
    dt_fs: float,
    n_states: int,
    seed: int,
    asserted: bool,
) -> dict[str, Any]:
    paired = paired_tasks({k: ours.get(k) for k in tasks}, theirs, n_total)
    every = max(1, round(OUTPUT_DT_FS / dt_fs))
    steps = np.arange(0, n_total + 1, every, dtype=np.int64)
    t_fs = steps * dt_fs
    out: dict[str, Any] = {
        "asserted": asserted,
        "n_tasks": len(tasks),
        "n_paired": len(paired),
        "paired": paired,
        "constants": {
            "MIN_PAIRED": MIN_PAIRED,
            "N_BOOTSTRAP": N_BOOTSTRAP,
            "SE_FACTOR": SE_FACTOR,
            "POPULATION_FLOOR": POPULATION_FLOOR,
            "FIT_FACTOR": FIT_FACTOR,
            "KS_MIN_P": KS_MIN_P,
            "seed": seed,
            "stream": COMPARE_STREAM,
        },
        "report": _report(ours, theirs, tasks, paired, n_total, steps, n_states),
    }
    if len(paired) < MIN_PAIRED:
        reason = f"only {len(paired)} paired trajectories (fewer than {MIN_PAIRED})"
        out["rules"] = [_verdict(r, None, reason) for r in RULES]
        return out
    po = [ours[k] for k in paired]
    pt = [theirs[k] for k in paired]
    aa, _ = on_steps(po, steps, n_states)
    ab, _ = on_steps(pt, steps, n_states)
    idx = bootstrap_indices(task_rng(seed, COMPARE_STREAM, 0), len(paired), N_BOOTSTRAP)
    fits = fit_rule(t_fs, (aa != 0).astype(float), (ab != 0).astype(float), idx)
    out["rules"] = [
        population_rule(aa, ab, n_states, idx),
        fits["tau"],
        fits["t0"],
        ks_rule(first_hop_times(po), first_hop_times(pt)),
    ]
    return out


def our_output_path(dyn_dir: Path, task: str) -> Path:
    """`<dynamics stage dir>/tasks/<task>/output.h5` (ProjectLayout.task_dir)."""
    return Path(dyn_dir) / "tasks" / task / "output.h5"


def read_ours(dyn_dir: Path, tasks: Sequence[str]) -> dict[str, Trajectory | None]:
    """Our trajectories by task; a partial absence is None, none at all is an error."""
    ours = {k: read_our_trajectory(our_output_path(dyn_dir, k), k) for k in tasks}
    if tasks and all(t is None for t in ours.values()):
        raise ValueError(
            f"none of our {len(tasks)} selected trajectories was found; "
            f"looked for {our_output_path(dyn_dir, tasks[0])} (and the same under tasks/ "
            "for the other trajectories)"
        )
    return ours


def main(argv: Sequence[str] | None = None) -> int:
    from forensics import find_stage_dir, jsonable
    from project_inputs import load_project_inputs

    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("project", type=Path, help="our gqteashmd project directory")
    p.add_argument("sharc", type=Path, help="a SHARC ensemble variant directory (traj_*/)")
    p.add_argument("--label", required=True)
    p.add_argument("--asserted", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2028)
    args = p.parse_args(argv)
    inputs = load_project_inputs(args.project)
    dyn = inputs.config.dynamics
    assert dyn is not None and inputs.config.electronic is not None
    n_total = round(dyn.tmax_fs / dyn.dt_fs)
    tasks = [s.task_id for s in inputs.starts]
    dyn_dir = find_stage_dir(inputs.project_dir, "dynamics")
    try:
        ours = read_ours(dyn_dir, tasks)
    except ValueError as error:
        print(f"ensemble_compare: error: {error}", file=sys.stderr)
        return 1
    theirs = {k: read_sharc_trajectory(args.sharc / k, k, n_total=n_total) for k in tasks}
    result = compare(
        ours,
        theirs,
        tasks=tasks,
        n_total=n_total,
        dt_fs=dyn.dt_fs,
        n_states=inputs.config.electronic.singlets,
        seed=args.seed,
        asserted=args.asserted,
    )
    result.update(label=args.label, project=str(args.project), sharc=str(args.sharc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(jsonable(result), indent=2))
    print(
        json.dumps(
            {
                "label": args.label,
                "n_paired": result["n_paired"],
                "verdicts": {r["rule"]: r["verdict"] for r in result["rules"]},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
