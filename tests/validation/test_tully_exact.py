"""The surface-hopping core against exact wavepacket scattering on Tully's three 1D models.

Set-up (packet, exact solver, FSSH settings, the rule |P_FSSH - P_exact| <= 4 SE + 0.02):
``tully_common.py`` next to this file. Every run here is **asserted**; every number is printed
(run with ``-s``). The unasserted ensembles (the other decoherence setting of every point,
DAC ln E = -2) are produced by ``tools/validation/tully_report.py`` and documented in
USER_MANUAL.md section 8.14.1(a).

Asserted runs (marker ``slow``; 8 workers):

- The exact solver's convergence at every asserted point (halving dx and dt, a wider grid and
  absorber: changes < 1e-3; norm 1 to 1e-8).
- SHARC-type hopping (``hopping = "sharc"``), no decoherence, the fewest-switches setting of
  Tully 1990:
  - SAC k = 10, 15, 20, 30 (n = 2000): transmitted lower and upper;
  - DAC ln E = -1, 0, 1 (n = 1000): transmitted lower and upper.
- Tully's flux probability (``hopping = "fssh"``), no decoherence: SAC k = 10, 20 (n = 2000)
  and DAC ln E = 0 (n = 1000), the same branches and rule.
- ECR k = 10, 20, 30 (n = 1000) with EDC (C = 0.1 Eh), since the reflection channels need
  decoherence (Tully 1990; Subotnik & Shenvi 2011): reflected lower and transmitted upper,
  once with ``edc_convention = "granucci_persico"`` and once with ``"sharc"`` (the default; this
  second assertion was added after its numbers were seen). At k = 10 and 20 the upper channel
  is closed (both exact and FSSH give 0), so only k = 30 tests transmitted upper.

SAC and DAC are asserted without decoherence because EDC biases them (it destroys the
coherent back-transfer of a slow single passage and the Stueckelberg interference; numbers in
USER_MANUAL.md section 8.14.1(a)).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_common():
    name = "gqteashmd_validation_tully_common"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("tully_common.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = _load_common()
POINTS = common.POINTS
SAC_K20 = POINTS[2]
ECR_K20 = POINTS[8]
ids = common.ids
assert (SAC_K20.model, SAC_K20.k, ECR_K20.model, ECR_K20.k) == ("SAC", 20, "ECR", 20)

# Tully's flux probability, no decoherence: n as for the SHARC-hopping run of the same point.
FSSH_HOPPING_POINTS = tuple(p for p in POINTS if ids(p) in ("SAC-k=10", "SAC-k=20", "DAC-lnE=0"))
assert [ids(p) for p in FSSH_HOPPING_POINTS] == ["SAC-k=10", "SAC-k=20", "DAC-lnE=0"]


def _check_ensemble(point, result, failures):
    assert result.energy_drift == 0
    assert result.unfinished <= 0.001 * point.n_trajectories
    assert failures == []


# --- exact solver self-tests ----------------------------------------------------------------


@pytest.mark.parametrize("point", (SAC_K20, ECR_K20), ids=ids)
def test_exact_solver_is_converged_quick(point):
    """Halving dx and dt changes no branch by 1e-3; the norm is 1 to 1e-8 (SAC k=20, ECR k=20)."""
    _check_convergence(point)


@pytest.mark.slow
@pytest.mark.parametrize("point", POINTS, ids=ids)
def test_exact_solver_is_converged(point):
    """As the quick test, for every point; also a wider grid and absorber change < 1e-3."""
    _check_convergence(point)
    wide = common.exact(point, 1, 90.0)
    base = common.exact(point)
    shift = max(abs(wide[b] - base[b]) for b in common.BRANCHES)
    print(f"\n[{point.model} {point.label}] wider grid/absorber: max change {shift:.2e}")
    assert shift < 1e-3


def _check_convergence(point) -> None:
    conv = common.convergence(point)
    base = conv["base"]
    print(
        f"\n[{point.model} {point.label}] exact "
        + " ".join(f"{b}={base[b]:.5f}" for b in common.BRANCHES)
        + f"; halved dx, dt: max change {conv['max_change']:.2e}; t_end {base['t_end']:.0f}"
    )
    for result in (base, conv["fine"]):
        assert result["converged"]
        assert abs(result["norm"] - 1.0) < 1e-8
        assert abs(sum(result[b] for b in common.BRANCHES) - 1.0) < 1e-8
        assert result["edge_norm"] < 1e-8
    assert conv["max_change"] < 1e-3


# --- FSSH against exact ----------------------------------------------------------------------


def test_fssh_smoke_sac():
    """Small unmarked run (n = 20, SAC k = 20): every trajectory finishes, within the rule."""
    ref = common.exact(SAC_K20)
    result, seconds = common.fssh(SAC_K20, n=20, decoherence="none", workers=2)
    assert result.unfinished == 0 and result.energy_drift == 0
    assert common.compare(SAC_K20, ref, result, seconds, "none", asserted=True) == []


@pytest.mark.slow
@pytest.mark.parametrize("point", POINTS, ids=ids)
def test_sharc_hopping_matches_exact(point):
    """SHARC-type hopping with the point's decoherence (none for SAC/DAC; Granucci-Persico EDC
    for ECR, the form these points were first validated with)."""
    result, seconds = common.fssh(point, n=point.n_trajectories, decoherence=point.decoherence)
    failures = common.compare(
        point, common.exact(point), result, seconds, point.decoherence, asserted=True
    )
    _check_ensemble(point, result, failures)


@pytest.mark.slow
@pytest.mark.parametrize("point", FSSH_HOPPING_POINTS, ids=ids)
def test_tully_flux_hopping_matches_exact(point):
    """``hopping = "fssh"`` (Tully's flux formula), no decoherence: the ensemble-level check of
    that code path (substep accumulation, cap, floor), with the rule and branches of the
    SHARC-hopping run of the same point."""
    result, seconds = common.fssh(point, n=point.n_trajectories, decoherence="none", hopping="fssh")
    failures = common.compare(
        point, common.exact(point), result, seconds, "none hopping=fssh", asserted=True
    )
    _check_ensemble(point, result, failures)


@pytest.mark.slow
@pytest.mark.parametrize("point", common.ECR_POINTS, ids=ids)
def test_ecr_with_sharc_edc_matches_exact(point):
    """ECR with the SHARC 4.1 manual's EDC (eq. 8.25; half the Granucci-Persico rate), n = 1000,
    under the same rule and asserted branches as the Granucci-Persico run (Task 9b). It was first
    run report-only; every branch at all three momenta met the rule, so it is asserted."""
    result, seconds = common.fssh(
        point, n=point.n_trajectories, decoherence="edc", edc_convention="sharc"
    )
    failures = common.compare(
        point, common.exact(point), result, seconds, "edc-sharc", asserted=True
    )
    _check_ensemble(point, result, failures)


# --- published values ------------------------------------------------------------------------


def test_published_values():
    pytest.skip(
        "No open-access source tabulating branching probabilities for Tully's models was found "
        "(Tully 1990 and the open-access comparisons, e.g. Mannouch & Richardson, J. Chem. Phys. "
        "158, 104111 (2023), doi:10.1063/5.0139734, arXiv:2212.11773, Figs. 6 and 8, give "
        "figures only, mostly at fixed momentum). No digitized numbers are used; see the Task 8 "
        "report."
    )
