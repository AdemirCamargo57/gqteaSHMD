"""Shared set-up of the FSSH-versus-exact comparisons on Tully's three 1D models.

Used by the asserted slow test ``tests/validation/test_tully_exact.py`` and by the report
script ``tools/validation/tully_report.py`` (the unasserted ensembles). Not collected by pytest
(no ``test_`` prefix); load it by path with ``load_common()``.

Reference: our own split-operator solver (``wavepacket.py`` next to this file), started from
the same packet psi ~ exp(-(x - x0)^2/sigma^2 + ikx), sigma = 20/k, on the lower adiabatic
state, whose Wigner function the FSSH ensemble samples. Model definitions: Tully, J. Chem.
Phys. 93, 1061 (1990), mass 2000 me.

FSSH settings: Wigner initial conditions, ``parallel_vel``, frustrated hops kept, dt = 0.5 au,
25 substeps, seed 20261108; ``hopping`` (default ``"sharc"``), ``decoherence`` and
``edc_convention`` are chosen per run. The rule is |P_FSSH - P_exact| <= 4 SE + 0.02 with
SE = sqrt(P_exact (1 - P_exact) / n_finished).
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from gqteashmd.dynamics.ensemble import run_model_ensemble
from gqteashmd.dynamics.models import (
    TullyDualAvoidedCrossing,
    TullyExtendedCoupling,
    TullySimpleAvoidedCrossing,
)
from gqteashmd.dynamics.options import DynamicsOptions

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_common():
    """This module, loaded once under a fixed name (shared by the test and the report)."""
    return _load("gqteashmd_validation_tully_common", HERE / "tully_common.py")


wavepacket = _load("gqteashmd_validation_wavepacket", HERE / "wavepacket.py")

MASS = 2000.0
WORKERS = 8
SEED = 20261108
BRANCHES = wavepacket.BRANCHES
MODELS = {
    "SAC": TullySimpleAvoidedCrossing,
    "DAC": TullyDualAvoidedCrossing,
    "ECR": TullyExtendedCoupling,
}
HALF_WIDTH = 60.0


@dataclass(frozen=True)
class Point:
    model: str
    k: float
    label: str
    x0: float
    x_exit: float  # FSSH stop |x| > x_exit; also the exact solver's interaction region
    n_grid: int  # exact solver grid on [-60, 60) bohr
    dt_exact: float
    asserted: tuple[str, ...]  # branches asserted in the run with the point's own decoherence
    decoherence: str  # decoherence setting of the asserted run
    n_trajectories: int

    @property
    def sigma(self) -> float:
        return 20.0 / self.k


def dac_k(ln_e: float) -> float:
    return math.sqrt(2.0 * MASS * math.exp(ln_e))


def ids(point: Point) -> str:
    return f"{point.model}-{point.label}"


TRANSMITTED = ("transmitted_lower", "transmitted_upper")
ECR_ASSERTED = ("reflected_lower", "transmitted_upper")

# The asserted points. SAC k = 30 has the smallest margin of the exact solver's grid guard
# (wavepacket.py: p_fast = 42.0 against the limit 0.8 pi/dx = 42.9 of the 2048-point grid). Its
# n_grid is deliberately not changed: the documented numbers (USER_MANUAL.md section 8.14.1)
# were obtained with it and its convergence test passes (halving dx and dt changes < 1e-3). A
# larger k or a narrower packet on this grid raises ValueError ("grid too coarse").
SAC_POINTS = tuple(
    Point("SAC", k, f"k={k}", -10.0, 10.0, 2048, 1.0, TRANSMITTED, "none", 2000)
    for k in (10, 15, 20, 30)
)
DAC_POINTS = tuple(
    Point("DAC", dac_k(e), f"lnE={e}", -10.0, 10.0, 8192, 0.5, TRANSMITTED, "none", 1000)
    for e in (-1, 0, 1)
)
ECR_POINTS = tuple(
    Point("ECR", k, f"k={k}", -15.0, 12.0, 4096, 1.0, ECR_ASSERTED, "edc", 1000)
    for k in (10, 20, 30)
)
POINTS = (*SAC_POINTS, *DAC_POINTS, *ECR_POINTS)

# DAC ln E = -2: the low-energy Stueckelberg region, classified report-only after the numbers
# were seen (USER_MANUAL.md section 8.14.1(a)); run only by tools/validation/tully_report.py.
DAC_LOW = Point("DAC", dac_k(-2), "lnE=-2", -10.0, 10.0, 8192, 0.5, (), "none", 1000)
REPORT_POINTS = (*SAC_POINTS, DAC_LOW, *DAC_POINTS, *ECR_POINTS)


@cache
def exact(point: Point, refine: int = 1, half_width: float = HALF_WIDTH) -> dict:
    """Exact branching; ``refine = 2`` halves dx and dt."""
    return wavepacket.exact_branching(
        MODELS[point.model](),
        mass=MASS,
        x0=point.x0,
        k=point.k,
        sigma=point.sigma,
        x_min=-half_width,
        x_max=half_width,
        n_grid=int(point.n_grid * refine * half_width / HALF_WIDTH),
        dt=point.dt_exact / refine,
        t_final=1.0e6,
        x_int=point.x_exit,
    )


def convergence(point: Point) -> dict:
    """Changes of the exact branches when dx and dt are halved, and the checks on both runs."""
    base, fine = exact(point), exact(point, 2)
    return {
        "base": base,
        "fine": fine,
        "max_change": max(abs(fine[b] - base[b]) for b in BRANCHES),
    }


def fssh(
    point: Point,
    *,
    n: int,
    decoherence: str,
    workers: int = WORKERS,
    edc_convention: str = "granucci_persico",
    hopping: str = "sharc",
):
    """The FSSH ensemble; returns (Branching, wall seconds).

    EDC defaults to the Granucci-Persico form the first ECR runs were validated with (Task 8);
    ``edc_convention = "sharc"`` gives the SHARC 4.1 manual's eq. (8.25). ``hopping`` is the
    probability scheme: ``"sharc"`` (Mai, Marquetand & Gonzalez 2015) or ``"fssh"`` (Tully's
    flux formula); with ``decoherence = "none"`` both are fewest-switches surface hopping.
    """
    options = DynamicsOptions(
        dt_au=0.5,
        nsubsteps=25,
        hopping=hopping,
        ekin_correction="parallel_vel",
        frustrated="keep",
        decoherence=decoherence,
        edc_convention=edc_convention,
    )
    start = time.perf_counter()
    result = run_model_ensemble(
        MODELS[point.model](),
        n_trajectories=n,
        x0=point.x0,
        k=point.k,
        sigma=point.sigma,
        initial_state=0,
        options=options,
        seed=SEED,
        x_exit=point.x_exit,
        max_steps=1_000_000,
        sampling="wigner",
        workers=workers,
    )
    return result, time.perf_counter() - start


def tolerance(p_exact: float, n: int) -> float:
    return 4.0 * math.sqrt(max(p_exact * (1.0 - p_exact), 0.0) / n) + 0.02


def comparison(point: Point, ref: dict, result, asserted: tuple[str, ...] = ()) -> dict:
    """Per-branch numbers of one ensemble against exact; ``ok`` is the rule's verdict."""
    probs, intervals, n = result.probabilities(), result.intervals_95(), result.n_finished
    rows = {}
    for branch in BRANCHES:
        diff = probs[branch] - ref[branch]
        tol = tolerance(ref[branch], n)
        rows[branch] = {
            "fssh": probs[branch],
            "interval_95": list(intervals[branch]),
            "exact": ref[branch],
            "diff": diff,
            "tolerance": tol,
            "ok": abs(diff) <= tol,
            "asserted": branch in asserted,
        }
    return rows


def compare(point: Point, ref: dict, result, seconds: float, tag: str, *, asserted: bool) -> list:
    """Print the comparison table; return the asserted branches that fail.

    With ``asserted = True`` the branches in ``point.asserted`` are asserted, otherwise none.
    """
    rows = comparison(point, ref, result, point.asserted if asserted else ())
    print(
        f"\n[{point.model} {point.label} k={point.k:.3f} {tag}] n_finished={result.n_finished} "
        f"unfinished={result.unfinished} energy_drift={result.energy_drift} "
        f"wall={seconds:.1f}s exact t_end={ref['t_end']:.0f} norm-1={ref['norm'] - 1:.1e}"
    )
    failures = []
    for branch, row in rows.items():
        lo, hi = row["interval_95"]
        print(
            f"  {branch:18s} FSSH {row['fssh']:.4f} [{lo:.4f}, {hi:.4f}]  exact "
            f"{row['exact']:.4f}  diff {row['diff']:+.4f}  tol {row['tolerance']:.4f}  "
            f"{'ASSERTED' if row['asserted'] else 'reported'} {'ok' if row['ok'] else 'OUTSIDE'}"
        )
        if row["asserted"] and not row["ok"]:
            failures.append(f"{branch}: FSSH {row['fssh']:.4f} vs exact {row['exact']:.4f}")
    return failures
