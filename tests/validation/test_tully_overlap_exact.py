"""Local diabatization (``coupling = "overlap"``) against exact wavepacket scattering (P7a-2).

The propagation uses the snapshot overlaps instead of the NACs (Granucci, Persico & Toniolo,
J. Chem. Phys. 114, 10608 (2001); Method reference item 7). Set-up, exact solver and rule as in
``test_tully_exact.py`` (shared through ``tully_common.py``): Wigner initial conditions,
``hopping = "sharc"``, ``parallel_vel``, frustrated hops kept, dt = 0.5 au, 25 substeps, seed
20261108, 8 workers, and |P_FSSH - P_exact| <= 4 SE + 0.02 on the P7a-1 asserted branches.

Pre-registered points (marker ``slow``, all asserted; run with ``-s`` for the numbers):

- SAC k = 20 (n = 2000), no decoherence: transmitted lower and upper;
- DAC ln E = 0 and 1 (n = 1000), no decoherence: transmitted lower and upper;
- ECR k = 30 (n = 1000), EDC with ``edc_convention = "sharc"``: reflected lower and transmitted
  upper.
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
ids = common.ids

# (point, decoherence, EDC convention); n is the point's own n_trajectories
OVERLAP_RUNS = (
    (common.SAC_POINTS[2], "none", "sharc"),
    (common.DAC_POINTS[1], "none", "sharc"),
    (common.DAC_POINTS[2], "none", "sharc"),
    (common.ECR_POINTS[2], "edc", "sharc"),
)
assert [(ids(p), p.n_trajectories) for p, _, _ in OVERLAP_RUNS] == [
    ("SAC-k=20", 2000),
    ("DAC-lnE=0", 1000),
    ("DAC-lnE=1", 1000),
    ("ECR-k=30", 1000),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    ("point", "decoherence", "edc_convention"), OVERLAP_RUNS, ids=[ids(r[0]) for r in OVERLAP_RUNS]
)
def test_local_diabatization_matches_exact(point, decoherence, edc_convention):
    result, seconds = common.fssh(
        point,
        n=point.n_trajectories,
        decoherence=decoherence,
        edc_convention=edc_convention,
        coupling="overlap",
    )
    tag = "overlap " + ("none" if decoherence == "none" else f"edc-{edc_convention}")
    failures = common.compare(point, common.exact(point), result, seconds, tag, asserted=True)
    assert result.energy_drift == 0
    assert result.unfinished <= 0.001 * point.n_trajectories
    assert failures == []
