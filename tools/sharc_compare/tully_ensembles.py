"""Branching ratios of the dynamics core against SHARC 4.1 (sharc.x, black box): a report.

Not a test; nothing is asserted. The asserted SHARC comparisons are the single-trajectory
tests in ``tests/validation/test_sharc_tully.py`` (step by step, replayed hops, the EDC rate),
which discriminate far better: a two-proportion z-test at n = 500 could not see the factor 2
between SHARC's EDC and Granucci-Persico's (max |z| 2.44). The numbers this script produced
are documented in USER_MANUAL.md section 8.14.1(c).

SHARC is only executed, through ``run_sharc_model.py`` next to this file (our writers and
parsers; formats from the public SHARC 4.1 manual), with our awk ``fast`` provider behind
``QM/runQM.sh``. Both programs get the same Wigner initial conditions (the ``model-initial``
stream of ``gqteashmd.dynamics.ensemble``); SHARC gets its own seeds. Points (n = 500):

- SAC k = 20, no decoherence;
- ECR k = 20, EDC with SHARC's convention (``edc_convention = "sharc"``, C = 0.1 Eh).

SHARC hopping, ``parallel_vel``, frustrated hops kept, dt = 0.5 au, 25 substeps. Each branch
gets a two-proportion z value. Runs in WSL (about 1 h; SHARC is about 15 times slower per
trajectory than the core), for example:

    SHARC_HOME=/home/ajc/sharc4-4.1 SHARC_PYTHON=/home/ajc/miniforge3/envs/sharc4.1/bin/python \\
        python tools/sharc_compare/tully_ensembles.py --out ~/tully_ensembles.json

``--n 3`` gives a smoke run. The SHARC work directories go to a temporary directory outside
the repository (``--work-dir`` to choose one) and are removed afterwards.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from gqteashmd.dynamics import ensemble
from gqteashmd.dynamics.ensemble import BRANCHES, run_model_ensemble, wavepacket_initial_conditions
from gqteashmd.dynamics.models import TullyExtendedCoupling, TullySimpleAvoidedCrossing
from gqteashmd.dynamics.options import DynamicsOptions

HERE = Path(__file__).resolve().parent
SEED = 20261108
SHARC_WORKERS = 14
OUR_WORKERS = 8
MODELS = {"SAC": TullySimpleAvoidedCrossing, "ECR": TullyExtendedCoupling}


def _load(name: str):
    if name in sys.modules:  # shared with the tests that load the harness by path
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rsm = _load("run_sharc_model")


@dataclass(frozen=True)
class Point:
    model: str
    k: float
    x0: float
    x_exit: float
    decoherence: str
    edc_convention: str = "sharc"  # the convention SHARC itself uses

    @property
    def sigma(self) -> float:
        return 20.0 / self.k

    @property
    def name(self) -> str:
        return f"{self.model}-k{self.k:g}-{self.decoherence}"


POINTS = (Point("SAC", 20.0, -10.0, 10.0, "none"), Point("ECR", 20.0, -15.0, 12.0, "edc"))


def two_proportion_z(c1: int, n1: int, c2: int, n2: int) -> float:
    """Pooled two-proportion z statistic of c1/n1 against c2/n2 (0 if the pooled variance is 0)."""
    pooled = (c1 + c2) / (n1 + n2)
    var = pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2)
    return 0.0 if var == 0.0 else (c1 / n1 - c2 / n2) / math.sqrt(var)


def options_for(point: Point) -> DynamicsOptions:
    return DynamicsOptions(
        dt_au=0.5,
        nsubsteps=25,
        hopping="sharc",
        ekin_correction="parallel_vel",
        frustrated="keep",
        decoherence=point.decoherence,
        edc_convention=point.edc_convention,
    )


def sharc_ensemble(
    point: Point, n: int, sharc_home: Path, python: str, base: Path, workers: int
) -> dict[str, int]:
    cls = MODELS[point.model]
    options = options_for(point)
    mass = float(cls().masses[0])
    tag = zlib.crc32(b"model-initial")
    assert tag == ensemble._INITIAL_TAG  # the stream run_model_ensemble draws from
    rng = np.random.default_rng(np.random.SeedSequence([SEED, tag]))
    q0, v0 = wavepacket_initial_conditions(n, point.x0, point.k, point.sigma, mass, rng, "wigner")
    seeds = np.random.SeedSequence([SEED, zlib.crc32(b"sharc-rngseed")]).generate_state(n)
    sharc_options = rsm.SharcOptions.from_dynamics_options(options)

    def one(i: int) -> str:
        work = base / f"traj{i:04d}"
        try:
            traj = rsm.run(
                sharc_home,
                python,
                rsm.model_spec(cls()),
                float(q0[i, 0]),
                float(v0[i, 0]),
                200_000,
                options.dt_au * rsm.AU_TIME_IN_FS,
                sharc_options,
                int(seeds[i]) % 2_000_000_000 + 1,
                work,
                provider="fast",
                detailed=False,
                x_exit=point.x_exit,
                timeout_s=3600.0,
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if not traj.stopped:
            return "unfinished"
        side = "transmitted" if traj.final_x_bohr > 0 else "reflected"
        return f"{side}_{'lower' if traj.final_state == 0 else 'upper'}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        labels = list(pool.map(one, range(n)))
    return {name: labels.count(name) for name in (*BRANCHES, "unfinished")}


def compare_point(
    point: Point, n: int, sharc_home: Path, python: str, base: Path, sharc_workers: int
) -> dict:
    start = time.perf_counter()
    ours = run_model_ensemble(
        MODELS[point.model](),
        n_trajectories=n,
        x0=point.x0,
        k=point.k,
        sigma=point.sigma,
        initial_state=0,
        options=options_for(point),
        seed=SEED,
        x_exit=point.x_exit,
        max_steps=1_000_000,
        sampling="wigner",
        workers=OUR_WORKERS,
    )
    t_ours = time.perf_counter() - start
    start = time.perf_counter()
    sharc = sharc_ensemble(point, n, sharc_home, python, base / point.name, sharc_workers)
    t_sharc = time.perf_counter() - start
    n_ours, n_sharc = ours.n_finished, sum(sharc[b] for b in BRANCHES)
    print(
        f"\n[{point.name} ({point.edc_convention})] n={n}; ours {t_ours:.0f} s "
        f"(unfinished {ours.unfinished}, drift {ours.energy_drift}); SHARC {t_sharc:.0f} s "
        f"(unfinished {sharc['unfinished']})"
    )
    branches = {}
    for branch in BRANCHES:
        c_ours, c_sharc = getattr(ours, branch), sharc[branch]
        z = two_proportion_z(c_ours, n_ours, c_sharc, n_sharc) if n_ours and n_sharc else 0.0
        print(
            f"  {branch:18s} ours {c_ours:4d} ({c_ours / max(n_ours, 1):.4f})  SHARC "
            f"{c_sharc:4d} ({c_sharc / max(n_sharc, 1):.4f})  z {z:+.2f}"
        )
        branches[branch] = {"ours": c_ours, "sharc": c_sharc, "z": z}
    return {
        "point": point.name,
        "model": point.model,
        "k": point.k,
        "x0": point.x0,
        "x_exit": point.x_exit,
        "sigma": point.sigma,
        "decoherence": point.decoherence,
        "edc_convention": point.edc_convention,
        "n_trajectories": n,
        "ours": {**ours.counts(), "wall_s": t_ours},
        "sharc": {**sharc, "wall_s": t_sharc},
        "branches": branches,
        "max_abs_z": max(abs(b["z"]) for b in branches.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="JSON file to write")
    parser.add_argument("--sharc-home", default=os.environ.get("SHARC_HOME"))
    parser.add_argument("--python", default=os.environ.get("SHARC_PYTHON"))
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--sharc-workers", type=int, default=SHARC_WORKERS)
    parser.add_argument("--points", nargs="+", choices=[p.name for p in POINTS])
    parser.add_argument("--work-dir", type=Path, help="parent of the SHARC run directories")
    args = parser.parse_args(argv)
    if not args.sharc_home or not args.python:
        parser.error("needs --sharc-home and --python (or SHARC_HOME and SHARC_PYTHON)")
    points = [p for p in POINTS if args.points is None or p.name in args.points]

    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    base = Path(tempfile.mkdtemp(prefix="gqteashmd-sharc-ens-", dir=args.work_dir))
    start = time.perf_counter()
    try:
        results = [
            compare_point(p, args.n, Path(args.sharc_home), args.python, base, args.sharc_workers)
            for p in points
        ]
    finally:
        shutil.rmtree(base, ignore_errors=True)
    report = {
        "schema_version": 1,
        "script": "tools/sharc_compare/tully_ensembles.py",
        "started_at": started_at,
        "seed": SEED,
        "settings": "sharc hopping, parallel_vel, frustrated keep, dt 0.5 au, 25 substeps, "
        "wigner initial conditions (shared), fast provider",
        "points": results,
        "wall_s": time.perf_counter() - start,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport written to {args.out} ({report['wall_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
