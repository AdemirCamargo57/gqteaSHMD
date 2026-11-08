"""The surface-hopping core against exact wavepacket scattering on Tully's three 1D models.

Reference: our own split-operator solver (``tests/validation/wavepacket.py``), started from the
same packet psi ~ exp(-(x - x0)^2/sigma^2 + ikx), sigma = 20/k, on the lower adiabatic state,
whose Wigner function the FSSH ensemble samples. Model definitions: Tully, J. Chem. Phys. 93,
1061 (1990), mass 2000 me.

FSSH settings: Wigner initial conditions, SHARC hopping, ``parallel_vel``, frustrated hops kept,
dt = 0.5 au, 25 substeps. The asserted run of each point uses the decoherence setting under
which FSSH is known to be accurate there; the other setting is run with n = 500 and reported:
- SAC and DAC: plain FSSH (no decoherence), as in Tully 1990. EDC (Granucci & Persico 2007)
  acts throughout the coupling region and destroys the coherent back-transfer of a slow single
  passage: at SAC k = 10 (fixed k, n = 400) EDC gives P(T, upper) = 0.22 +- 0.02, plain FSSH
  0.14 +- 0.02, the exact packet 0.15. Plain FSSH from an independent minimal implementation
  (same random numbers) gave the identical count.
- ECR: EDC (C = 0.1 Eh), since the reflection channels need decoherence (Tully 1990; Subotnik &
  Shenvi 2011).
n = 2000 (SAC) and 1000 (DAC, ECR; controller ruling on wall time) trajectories.

Assertions, only where FSSH is known to be accurate: |P_FSSH - P_exact| <= 4 SE + 0.02, with
SE = sqrt(P_exact (1 - P_exact) / n_finished):
- SAC (k = 10, 15, 20, 30): transmitted lower and upper.
- DAC (ln E = -1, 0, 1): transmitted lower and upper. ln E = -2 lies in the low-energy
  Stueckelberg region where FSSH is known to deviate (Tully 1990): reported only
  (plain FSSH at fixed k = 23.27, n = 400: 0.165 +- 0.019 upper, identical in the core and
  in the independent implementation; exact packet 0.093).
- ECR (k = 10, 20, 30): reflected lower and transmitted upper. The upper-state reflection is
  decoherence-sensitive: it is printed, with and without EDC, but not asserted.

Every number is printed (run with ``-s``).
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest

from gqteashmd.dynamics.ensemble import run_model_ensemble
from gqteashmd.dynamics.models import (
    TullyDualAvoidedCrossing,
    TullyExtendedCoupling,
    TullySimpleAvoidedCrossing,
)
from gqteashmd.dynamics.options import DynamicsOptions


def _load_wavepacket():
    name = "gqteashmd_validation_wavepacket"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("wavepacket.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wavepacket = _load_wavepacket()

MASS = 2000.0
WORKERS = 8
SEED = 20261108
BRANCHES = wavepacket.BRANCHES
MODELS = {
    "SAC": TullySimpleAvoidedCrossing,
    "DAC": TullyDualAvoidedCrossing,
    "ECR": TullyExtendedCoupling,
}


@dataclass(frozen=True)
class Point:
    model: str
    k: float
    label: str
    x0: float
    x_exit: float  # FSSH stop |x| > x_exit; also the exact solver's interaction region
    n_grid: int  # exact solver grid on [-60, 60) bohr
    dt_exact: float
    asserted: tuple[str, ...]
    decoherence: str  # setting of the asserted FSSH run
    n_trajectories: int

    @property
    def sigma(self) -> float:
        return 20.0 / self.k


def _dac_k(ln_e: float) -> float:
    return math.sqrt(2.0 * MASS * math.exp(ln_e))


TRANSMITTED = ("transmitted_lower", "transmitted_upper")
ECR_ASSERTED = ("reflected_lower", "transmitted_upper")
POINTS = (
    *(
        Point("SAC", k, f"k={k}", -10.0, 10.0, 2048, 1.0, TRANSMITTED, "none", 2000)
        for k in (10, 15, 20, 30)
    ),
    *(
        Point("DAC", _dac_k(e), f"lnE={e}", -10.0, 10.0, 8192, 0.5, a, "none", 1000)
        for e, a in ((-2, ()), (-1, TRANSMITTED), (0, TRANSMITTED), (1, TRANSMITTED))
    ),
    *(
        Point("ECR", k, f"k={k}", -15.0, 12.0, 4096, 1.0, ECR_ASSERTED, "edc", 1000)
        for k in (10, 20, 30)
    ),
)
N_REPORTED = 500
SMOKE = POINTS[2]  # SAC k = 20
HALF_WIDTH = 60.0


def _ids(point: Point) -> str:
    return f"{point.model}-{point.label}"


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


def fssh(point: Point, *, n: int, decoherence: str, workers: int = WORKERS):
    options = DynamicsOptions(
        dt_au=0.5,
        nsubsteps=25,
        hopping="sharc",
        ekin_correction="parallel_vel",
        frustrated="keep",
        decoherence=decoherence,
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


def compare(point: Point, ref: dict, result, seconds: float, tag: str) -> list[str]:
    """Print the comparison table; return the asserted branches that fail."""
    probs, intervals, n = result.probabilities(), result.intervals_95(), result.n_finished
    print(
        f"\n[{point.model} {point.label} k={point.k:.3f} {tag}] n_finished={n} "
        f"unfinished={result.unfinished} energy_drift={result.energy_drift} "
        f"wall={seconds:.1f}s exact t_end={ref['t_end']:.0f} norm-1={ref['norm'] - 1:.1e}"
    )
    failures = []
    for branch in BRANCHES:
        lo, hi = intervals[branch]
        diff = probs[branch] - ref[branch]
        tol = tolerance(ref[branch], n)
        asserted = branch in point.asserted and tag == point.decoherence
        ok = abs(diff) <= tol
        print(
            f"  {branch:18s} FSSH {probs[branch]:.4f} [{lo:.4f}, {hi:.4f}]  exact "
            f"{ref[branch]:.4f}  diff {diff:+.4f}  tol {tol:.4f}  "
            f"{'ASSERTED' if asserted else 'reported'} {'ok' if ok else 'OUTSIDE'}"
        )
        if asserted and not ok:
            failures.append(f"{branch}: FSSH {probs[branch]:.4f} vs exact {ref[branch]:.4f}")
    return failures


# --- exact solver self-tests ----------------------------------------------------------------


@pytest.mark.parametrize("point", (POINTS[2], POINTS[9]), ids=_ids)
def test_exact_solver_is_converged_quick(point):
    """Halving dx and dt changes no branch by 1e-3; the norm is 1 to 1e-8 (SAC k=20, ECR k=20)."""
    _check_convergence(point)


@pytest.mark.slow
@pytest.mark.parametrize("point", POINTS, ids=_ids)
def test_exact_solver_is_converged(point):
    """As the quick test, for every point; also a wider grid and absorber change < 1e-3."""
    _check_convergence(point)
    wide = exact(point, 1, 90.0)
    base = exact(point)
    shift = max(abs(wide[b] - base[b]) for b in BRANCHES)
    print(f"\n[{point.model} {point.label}] wider grid/absorber: max change {shift:.2e}")
    assert shift < 1e-3


def _check_convergence(point: Point) -> None:
    base, fine = exact(point), exact(point, 2)
    change = max(abs(fine[b] - base[b]) for b in BRANCHES)
    print(
        f"\n[{point.model} {point.label}] exact "
        + " ".join(f"{b}={base[b]:.5f}" for b in BRANCHES)
        + f"; halved dx, dt: max change {change:.2e}; t_end {base['t_end']:.0f}"
    )
    for result in (base, fine):
        assert result["converged"]
        assert abs(result["norm"] - 1.0) < 1e-8
        assert abs(sum(result[b] for b in BRANCHES) - 1.0) < 1e-8
        assert result["edge_norm"] < 1e-8
    assert change < 1e-3


# --- FSSH against exact ----------------------------------------------------------------------


def test_fssh_smoke_sac():
    """Small unmarked run (n = 20, SAC k = 20): every trajectory finishes, within the rule."""
    ref = exact(SMOKE)
    result, seconds = fssh(SMOKE, n=20, decoherence=SMOKE.decoherence, workers=2)
    assert result.unfinished == 0 and result.energy_drift == 0
    assert compare(SMOKE, ref, result, seconds, SMOKE.decoherence) == []


@pytest.mark.slow
@pytest.mark.parametrize("point", POINTS, ids=_ids)
def test_fssh_matches_exact(point):
    ref = exact(point)
    result, seconds = fssh(point, n=point.n_trajectories, decoherence=point.decoherence)
    failures = compare(point, ref, result, seconds, point.decoherence)
    assert result.energy_drift == 0
    assert result.unfinished <= 0.001 * point.n_trajectories
    assert failures == []


@pytest.mark.slow
@pytest.mark.parametrize("point", POINTS, ids=_ids)
def test_fssh_other_decoherence_reported(point):
    """The other decoherence setting (n = 500): printed for the report, not compared."""
    other = "edc" if point.decoherence == "none" else "none"
    result, seconds = fssh(point, n=N_REPORTED, decoherence=other)
    compare(point, exact(point), result, seconds, other)
    assert result.energy_drift == 0


# --- published values ------------------------------------------------------------------------


def test_published_values():
    pytest.skip(
        "No open-access source tabulating branching probabilities for Tully's models was found "
        "(Tully 1990 and the open-access comparisons, e.g. Mannouch & Richardson, J. Chem. Phys. "
        "158, 104111 (2023), doi:10.1063/5.0139734, arXiv:2212.11773, Figs. 6 and 8, give "
        "figures only, mostly at fixed momentum). No digitized numbers are used; see the Task 8 "
        "report."
    )
