"""Hop energy option report: our P9b ensemble with hop_energy = "current" against the same
ensemble rerun with "previous" (spec 2026-11-18-hop-energy-option-design.md, section 4.3).

Report only: every number goes to the JSON file and nothing is asserted. Never imported by
gqteashmd. Usage: hop_energy_report.py CURRENT_RUN PREVIOUS_RUN --out JSON (project dirs).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sharc_compare"))
sys.path.insert(0, str(HERE.parent / "diagnosis"))
from forensics import find_stage_dir, jsonable  # noqa: E402
from sharc_trajectories import is_kept, read_our_trajectory  # noqa: E402

from gqteashmd.analysis.grid import INCLUDED_OUTCOMES, Trajectory  # noqa: E402
from gqteashmd.core.units import HARTREE_IN_EV  # noqa: E402


def record_energy_changes_eV(t: Trajectory) -> tuple[np.ndarray, np.ndarray]:
    """|dE_tot| (eV) from each record to the next, split by whether the later one is a hop."""
    if t.e_total.size < 2:
        return np.empty(0), np.empty(0)
    d = np.abs(np.diff(t.e_total)) * HARTREE_IN_EV
    hop = np.asarray(t.hop_to[1:]) >= 0
    return d[hop], d[~hop]


def lost_after_hop(t: Trajectory) -> bool:
    """Not included by the analysis, with at least one accepted hop before its end."""
    return t.outcome not in INCLUDED_OUTCOMES and bool(np.any(np.asarray(t.hop_to) >= 0))


def quantiles(values: Sequence[float]) -> dict[str, Any]:
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return {"n": 0, "median": None, "p95": None, "max": None}
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "max": float(v.max()),
    }


def outcome_changes(
    current: Sequence[dict[str, Any]], previous: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    after = {r["task"]: r["outcome"] for r in previous}
    return [
        {"task": r["task"], "current": r["outcome"], "previous": after.get(r["task"])}
        for r in current
        if after.get(r["task"]) != r["outcome"]
    ]


def analysis_numbers(analysis: dict[str, Any]) -> dict[str, Any]:
    """The complete-rule numbers of an analysis.json (final review m1).

    Under schema_version 2 the top-level `n_included` and `fit` belong to the estimator in use
    (`censored` by default), so the complete rule's are read from `estimators["complete"]`
    (null when no trajectory is complete); schema 1 has the complete rule at the top level.
    """
    schema = analysis.get("schema_version", 1)
    if schema >= 2:
        block = (analysis.get("estimators") or {}).get("complete") or {}
        n_included = block.get("n", 0)
        small = block.get("small_ensemble")
        fit = block.get("fit") or {}
    else:
        n_included = analysis.get("n_included")
        small = analysis.get("small_ensemble")
        fit = analysis.get("fit") or {}
    return {
        "schema_version": schema,
        "estimator_read": "complete",
        "n_selected": analysis.get("n_selected"),
        "n_included": n_included,
        "small_ensemble": small,
        "warnings": analysis.get("warnings"),
        "t0_fs": fit.get("t0_fs"),
        "t0_interval_fs": fit.get("t0_interval_fs"),
        "tau_fs": fit.get("tau_fs"),
        "tau_interval_fs": fit.get("tau_interval_fs"),
    }


def summarize(run_dir: Path) -> dict[str, Any]:
    dyn = find_stage_dir(run_dir, "dynamics")
    summary = json.loads((dyn / "dynamics_summary.json").read_text())
    analysis = json.loads((find_stage_dir(run_dir, "analysis") / "analysis.json").read_text())
    n_total = round(summary["tmax_fs"] / summary["dt_fs"])
    rows = summary["trajectories"]
    trajs = [read_our_trajectory(dyn / "tasks" / r["task"] / "output.h5", r["task"]) for r in rows]
    hop_d: list[float] = []
    other_d: list[float] = []
    for t in trajs:
        if t is not None:
            a, b = record_energy_changes_eV(t)
            hop_d.extend(a.tolist())
            other_d.extend(b.tolist())
    return {
        "run": str(run_dir),
        "outcomes": summary["outcomes"],
        "totals": summary["totals"],
        "n_kept": sum(is_kept(t, n_total) for t in trajs),
        "n_missing_output": sum(t is None for t in trajs),
        "n_lost": sum(t is not None and t.outcome not in INCLUDED_OUTCOMES for t in trajs),
        "n_lost_after_hop": sum(t is not None and lost_after_hop(t) for t in trajs),
        "max_drift_eV": quantiles(
            [r["max_drift_eV"] for r in rows if r.get("max_drift_eV") is not None]
        ),
        "record_dE_eV_at_hops": quantiles(hop_d),
        "record_dE_eV_elsewhere": quantiles(other_d),
        "analysis": analysis_numbers(analysis),
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("current", type=Path, help='the run with hop_energy = "current" (P9b ours)')
    p.add_argument("previous", type=Path, help='the run with hop_energy = "previous"')
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    current, previous = summarize(args.current), summarize(args.previous)
    result = {
        "current": {k: v for k, v in current.items() if k != "rows"},
        "previous": {k: v for k, v in previous.items() if k != "rows"},
        "outcome_changes": outcome_changes(current["rows"], previous["rows"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(jsonable(result), indent=2))
    print(json.dumps({k: result[k]["n_kept"] for k in ("current", "previous")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
