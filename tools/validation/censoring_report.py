"""Censored analysis report (spec 2026-11-19-censored-analysis-design.md, section 4.2).

Report only: every number goes to the JSON file; the one verdict ("good", item 5) was fixed in
the spec before any run. Never imported by gqteashmd. It calls gqteashmd.analysis on existing
output.h5 files and does not run the stage (DynamicsStage.version 3 makes those projects stale).

Usage: censoring_report.py --artificial RUN [--project RUN ...] --out JSON
       [--repetitions 200] [--bootstrap 200] [--seed 20261119]
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
from sharc_trajectories import read_our_trajectory  # noqa: E402

from gqteashmd.analysis import censoring, estimators, grid, populations  # noqa: E402
from gqteashmd.analysis.grid import INCLUDED_OUTCOMES, Trajectory  # noqa: E402
from gqteashmd.analysis.kinetics import fit_delayed_exponential  # noqa: E402
from gqteashmd.util.seeds import task_rng  # noqa: E402

REPORT_TIMES_FS = (25.0, 50.0, 75.0, 100.0)


# ------------------------------------------------------------------------- helpers
def first_s0_time(t: Trajectory) -> float | None:
    hits = np.flatnonzero(t.active == 0)
    return float(t.time_fs[hits[0]]) if hits.size else None


def loss_delays_fs(trajs: Sequence[Trajectory]) -> list[float]:
    """t_final - first S0 arrival of every lost trajectory that reached S0 (spec 1.1)."""
    out = []
    for t in trajs:
        if t.outcome in INCLUDED_OUTCOMES or t.time_fs.size == 0:
            continue
        s0 = first_s0_time(t)
        if s0 is not None:
            out.append(float(t.time_fs[-1]) - s0)
    return out


def reaching_s0_fraction(trajs: Sequence[Trajectory]) -> tuple[int, int]:
    """(lost among those reaching S0, those reaching S0)."""
    reach = [t for t in trajs if t.time_fs.size and first_s0_time(t) is not None]
    lost = [t for t in reach if t.outcome not in INCLUDED_OUTCOMES]
    return len(lost), len(reach)


def censor_step_for(t: Trajectory, delay_fs: float, dt_fs: float, output_every: int) -> int | None:
    """The grid step at first S0 arrival + delay (rounded down to the output grid), or None if
    the trajectory never reaches S0 or that time is beyond its last record."""
    s0 = first_s0_time(t)
    if s0 is None or s0 + delay_fs > float(t.time_fs[-1]):
        return None
    step = int(np.floor((s0 + delay_fs) / dt_fs + 1e-9))
    return (step // output_every) * output_every


def hold_in_s0(active: np.ndarray, observed: np.ndarray) -> np.ndarray:
    held = np.array(active, copy=True)
    held[~np.asarray(observed, dtype=bool)] = 0
    return held


def max_bias_ratio(mean: np.ndarray, truth: np.ndarray, unit: np.ndarray) -> float:
    ok = unit > 0
    return float(np.max(np.abs(mean - truth)[ok] / unit[ok])) if np.any(ok) else 0.0


def poisson_rate_interval(k: int, exposure: float) -> tuple[float, float]:
    from scipy.stats import chi2

    lo = 0.0 if k == 0 else float(chi2.ppf(0.025, 2 * k) / 2 / exposure)
    hi = float(chi2.ppf(0.975, 2 * k + 2) / 2 / exposure)
    return lo, hi


# --------------------------------------------------------------------------- input
def load_project(run: Path) -> dict[str, Any]:
    config = json.loads((run / ".gqteashmd" / "config.json").read_text())  # ruling PR9
    dyn = config["dynamics"]
    dyn_dir = find_stage_dir(run, "dynamics")
    trajs: dict[str, Trajectory | None] = {}
    kinetic: dict[str, np.ndarray] = {}
    from gqteashmd.trajectory.store import read_output_h5

    for task_dir in sorted((dyn_dir / "tasks").glob("traj_*")):
        path = task_dir / "output.h5"
        trajs[task_dir.name] = read_our_trajectory(path, task_dir.name)
        if path.is_file():
            kinetic[task_dir.name] = np.asarray(read_output_h5(path).data["e_kinetic"], dtype=float)
    first = next(t for t in trajs.values() if t is not None)
    return {
        "trajectories": trajs,
        "e_kinetic": kinetic,
        "dt_fs": float(dyn["dt_fs"]),
        "output_every": int(dyn.get("output_every", 1)),
        "n_total": round(float(dyn["tmax_fs"]) / float(dyn["dt_fs"])),
        "n_states": int(first.populations.shape[1]),
        "seed": int(config["project"]["seed"]),
        "n_bootstrap": int(config.get("analysis", {}).get("n_bootstrap", 1000)),
    }


# -------------------------------------------------------------- the four estimators
def four_estimates(
    complete: Sequence[Trajectory], steps: dict[str, int], p: dict[str, Any]
) -> tuple[dict[str, np.ndarray], grid.GridData]:
    # "complete" holds only `finished` trajectories. In P9b none is `stopped_in_s0`
    # (stop_after_s0_fs is off), which is as the spec asks.
    h = grid.horizon_steps(complete, p["n_total"])
    g = grid.on_grid(
        complete,
        output_every=p["output_every"],
        dt_fs=p["dt_fs"],
        n_total=p["n_total"],
        n_states=p["n_states"],
        censored_at={k: v for k, v in steps.items() if v < h},
        horizon=h,
    )
    s = p["n_states"]
    never = g.observed[:, -1]
    out = {
        "censored": censoring.aalen_johansen(g.active, g.observed, s).populations,
        "still_running": censoring.at_risk_average(np.eye(s)[g.active], g.observed),
        "complete_only": populations.classical_populations(g.active[never], s)[0]
        if never.any()
        else np.full((g.active.shape[1], s), np.nan),
        "hold_in_s0": populations.classical_populations(hold_in_s0(g.active, g.observed), s)[0],
    }
    return out, g


def fit_excited(t: np.ndarray, p: np.ndarray, dt_out: float) -> tuple[float | None, float | None]:
    r = fit_delayed_exponential(t, 1.0 - p[:, 0], dt_out=dt_out)
    return r.tau_fs, r.t0_fs


def artificial(
    p: dict[str, Any], *, repetitions: int, bootstrap: int, seed: int, hot: bool
) -> dict[str, Any]:
    trajs = [t for t in p["trajectories"].values() if t is not None]
    complete = [t for t in trajs if t.outcome == "finished"]
    delays = loss_delays_fs(trajs)
    n_lost, n_reach = reaching_s0_fraction(trajs)
    fraction = n_lost / n_reach
    dt_out = p["output_every"] * p["dt_fs"]
    truth_est, g0 = four_estimates(complete, {}, p)
    truth = truth_est["censored"]  # no censoring: the classical fractions of the complete set
    tau_true, t0_true = fit_excited(g0.time_fs, truth, dt_out)
    rng = np.random.default_rng(seed + (1 if hot else 0))
    reach = [t for t in complete if first_s0_time(t) is not None]
    hot_set: set[str] = set()
    if hot:  # spec 4.2 item 6: the fraction with the highest mean kinetic energy in S0

        def mean_k_s0(t: Trajectory) -> float:
            k = p["e_kinetic"][t.task]
            return float(k[t.active == 0].mean())

        ranked = sorted(reach, key=mean_k_s0, reverse=True)
        hot_set = {t.task for t in ranked[: round(fraction * len(reach))]}
    curves: dict[str, list[np.ndarray]] = {k: [] for k in truth_est}
    fits: dict[str, list[tuple[float | None, float | None]]] = {k: [] for k in truth_est}
    units = []
    for _ in range(repetitions):
        steps: dict[str, int] = {}
        for t in reach:
            chosen = (t.task in hot_set) if hot else (rng.random() < fraction)
            if not chosen:
                continue
            step = censor_step_for(t, float(rng.choice(delays)), p["dt_fs"], p["output_every"])
            if step is not None:
                steps[t.task] = step
        est, g = four_estimates(complete, steps, p)
        for k, v in est.items():
            curves[k].append(v)
            fits[k].append(fit_excited(g.time_fs, v, dt_out))
        idx = populations.bootstrap_indices(rng, len(complete), bootstrap)
        boot = censoring.bootstrap_aalen_johansen(g.active, g.observed, p["n_states"], idx)
        units.append(boot.std(axis=0, ddof=1))
    unit = np.mean(units, axis=0)
    result: dict[str, Any] = {
        "n_complete": len(complete),
        "loss_fraction": [n_lost, n_reach],
        "delays_fs": delays,
        "repetitions": repetitions,
        "truth": {"tau_fs": tau_true, "t0_fs": t0_true},
        "estimators": {},
    }
    for k in truth_est:
        mean = np.nanmean(np.asarray(curves[k]), axis=0)
        taus = [f[0] for f in fits[k] if f[0] is not None]
        t0s = [f[1] for f in fits[k] if f[1] is not None]
        result["estimators"][k] = {
            "max_bias_over_se": max_bias_ratio(mean, truth, unit),
            "max_abs_bias": float(np.nanmax(np.abs(mean - truth))),
            "tau_bias_fs": (float(np.mean(taus)) - tau_true) if taus and tau_true else None,
            "t0_bias_fs": (float(np.mean(t0s)) - t0_true) if t0s and t0_true is not None else None,
            "fits_failed": sum(1 for f in fits[k] if f[0] is None),
        }
    if not hot:  # the verdict fixed in spec 4.2 item 5
        r = {k: v["max_bias_over_se"] for k, v in result["estimators"].items()}
        result["verdict_good"] = bool(
            r["censored"] <= 1.0
            and r["censored"] < r["complete_only"]
            and r["censored"] < r["still_running"]
        )
    return result


def back_hop_rates(p: dict[str, Any]) -> dict[str, Any]:
    """Hops S0 -> excited per fs of S0 time, lost (before the loss) and complete (spec 4.2 6)."""
    out = {}
    for name, keep in (
        ("lost", lambda t: t.outcome not in INCLUDED_OUTCOMES),
        ("complete", lambda t: t.outcome == "finished"),
    ):
        hops = 0
        exposure = 0.0
        for t in p["trajectories"].values():
            if t is None or t.time_fs.size < 2 or not keep(t) or first_s0_time(t) is None:
                continue
            dt = np.diff(t.time_fs)
            in_s0 = t.active[:-1] == 0
            exposure += float(dt[in_s0].sum())
            hops += int(np.sum((t.hop_from[1:] == 0) & (t.hop_to[1:] > 0)))
        lo, hi = poisson_rate_interval(hops, exposure) if exposure > 0 else (None, None)
        out[name] = {
            "hops": hops,
            "s0_time_fs": exposure,
            "rate_per_fs": hops / exposure if exposure > 0 else None,
            "interval_95": [lo, hi],
        }
    return out


def reanalysis(p: dict[str, Any]) -> dict[str, Any]:
    trajs = p["trajectories"]
    tasks = list(trajs)
    complete, _ = grid.select(tasks, trajs, {}, energy_jump_eV=0.1, exclude_energy_jumps=False)
    used, steps, cens, excl = censoring.select_censored(
        tasks,
        trajs,
        output_every=p["output_every"],
        n_total=p["n_total"],
        dt_fs=p["dt_fs"],
        energy_jump_eV=None,
    )
    h = grid.horizon_steps(complete, p["n_total"])
    steps = {k: v for k, v in steps.items() if v < h}
    kw = {
        "output_every": p["output_every"],
        "dt_fs": p["dt_fs"],
        "n_total": p["n_total"],
        "n_states": p["n_states"],
        "horizon": h,
    }
    g_c = grid.on_grid(complete, **kw)
    g_u = grid.on_grid(used, censored_at=steps, **kw)
    dt_out = p["output_every"] * p["dt_fs"]
    fit_index = list(range(1, p["n_states"]))
    ests = {
        "complete": estimators.complete_estimate(
            g_c,
            fit_index,
            fit_population="classical",
            rng=task_rng(p["seed"], "analysis", 0),
            n_bootstrap=p["n_bootstrap"],
            dt_out=dt_out,
        ),
        "censored": estimators.censored_estimate(
            g_u,
            fit_index,
            rng=task_rng(p["seed"], "analysis", 1),
            n_bootstrap=p["n_bootstrap"],
            dt_out=dt_out,
        ),
    }
    out: dict[str, Any] = {
        "n_complete": len(complete),
        "n_censored": len(steps),
        "n_excluded": len(excl),
        "horizon_fs": h * p["dt_fs"],
    }
    for name, e in ests.items():
        g = g_u if name == "censored" else g_c
        rows = {}
        for when in REPORT_TIMES_FS:
            k = int(np.argmin(np.abs(g.time_fs - when)))
            rows[f"{g.time_fs[k]:.2f}"] = {
                "P": e.classical[k].tolist(),
                "lower": e.classical_lower[k].tolist(),
                "upper": e.classical_upper[k].tolist(),
            }
        f = e.fit
        out[name] = {
            "populations": rows,
            "tau_fs": f.fit.tau_fs,
            "tau_interval_fs": f.tau_interval_fs,
            "t0_fs": f.fit.t0_fs,
            "t0_interval_fs": f.t0_interval_fs,
            "empty_rows": e.empty_rows,
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--artificial", type=Path, required=True)
    ap.add_argument("--project", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repetitions", type=int, default=200)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261119)
    args = ap.parse_args(argv)
    base = load_project(args.artificial)
    report = {
        "artificial": artificial(
            base, repetitions=args.repetitions, bootstrap=args.bootstrap, seed=args.seed, hot=False
        ),
        "sensitivity": artificial(
            base, repetitions=args.repetitions, bootstrap=args.bootstrap, seed=args.seed, hot=True
        ),
        "back_hop_rates": back_hop_rates(base),
        "reanalysis": {
            str(r): reanalysis(load_project(r)) for r in [args.artificial, *args.project]
        },
    }
    args.out.write_text(json.dumps(jsonable(report), indent=2) + "\n")
    print(json.dumps({"verdict_good": report["artificial"]["verdict_good"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
