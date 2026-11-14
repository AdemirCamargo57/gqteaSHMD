"""Coverage of the analysis' 95 % tau interval on synthetic ensembles (P9b spec 6).

Each repetition draws N first-passage times t0 + Exp(tau) (t0 10 fs, tau 40 fs), builds the
per-trajectory excited-state indicator on the 0.5 fs grid to 100 fs, and runs the analysis'
own ``bootstrap_fit`` with B = 500. Coverage = the fraction of repetitions whose interval holds
40 fs (a repetition without an interval counts as not covered). Report script: nothing is
asserted; the JSON goes to --out.

Example (WSL): python tools/validation/tau_coverage.py --out ~/p9b/tau_coverage.json --workers 8
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from gqteashmd.analysis.kinetics import bootstrap_fit
from gqteashmd.analysis.populations import bootstrap_indices
from gqteashmd.util.seeds import task_rng

T0_FS = 10.0
TAU_FS = 40.0
DT_FS = 0.5
T_END_FS = 100.0
SIZES = (20, 50, 100)
REPETITIONS = 400
N_BOOTSTRAP = 500
THRESHOLD = 0.90
SEED = 2028
STREAM = "p9b-coverage"
GRID = np.arange(0.0, T_END_FS + 1e-9, DT_FS)


def coverage_once(n: int, rng: np.random.Generator, *, n_bootstrap: int = N_BOOTSTRAP) -> dict:
    times = T0_FS + rng.exponential(TAU_FS, size=n)
    per = (GRID[None, :] < times[:, None]).astype(float)
    r = bootstrap_fit(GRID, per, bootstrap_indices(rng, n, n_bootstrap), dt_out=DT_FS)
    interval = r.tau_interval_fs
    covered = interval is not None and interval[0] <= TAU_FS <= interval[1]
    return {
        "covered": bool(covered),
        "tau_fs": r.fit.tau_fs,
        "interval": interval,
        "bootstrap_failures": r.bootstrap_failures,
    }


def _chunk(args: tuple[int, int, int]) -> list[dict]:
    n, size_index, rep = args
    rng = task_rng(SEED, STREAM, size_index * REPETITIONS + rep)
    return [coverage_once(n, rng)]


def coverage(n: int, repetitions: int, size_index: int, *, workers: int) -> dict:
    jobs = [(n, size_index, rep) for rep in range(repetitions)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        rows = [r for chunk in pool.map(_chunk, jobs, chunksize=8) for r in chunk]
    covered = sum(r["covered"] for r in rows)
    p = covered / repetitions
    return {
        "n": n,
        "repetitions": repetitions,
        "covered": covered,
        "coverage": p,
        "standard_error": float(np.sqrt(p * (1 - p) / repetitions)),
        "no_interval": sum(r["interval"] is None for r in rows),
        "tau_median_fs": float(np.median([r["tau_fs"] for r in rows if r["tau_fs"]])),
    }


def n_min(results: Mapping[int, float], threshold: float = THRESHOLD) -> int | None:
    ok = [n for n in sorted(results) if results[n] >= threshold]
    return ok[0] if ok else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    started = time.perf_counter()
    rows = [coverage(n, REPETITIONS, i, workers=args.workers) for i, n in enumerate(SIZES)]
    result = {
        "model": {"t0_fs": T0_FS, "tau_fs": TAU_FS, "dt_fs": DT_FS, "t_end_fs": T_END_FS},
        "n_bootstrap": N_BOOTSTRAP,
        "threshold": THRESHOLD,
        "seed": SEED,
        "stream": STREAM,
        "sizes": rows,
        "n_min": n_min({r["n"]: r["coverage"] for r in rows}),
        "wall_time_s": time.perf_counter() - started,
    }
    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.out.expanduser().write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "sizes"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
