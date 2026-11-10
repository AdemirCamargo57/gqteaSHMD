"""The two validation report scripts stay importable and keep their point sets (no runs here).

``tools/validation/tully_report.py`` (unasserted FSSH-vs-exact ensembles) and
``tools/sharc_compare/tully_ensembles.py`` (SHARC ensemble branching ratios) are run by hand;
they are not collected by pytest.
"""

import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


report = _load("gqteashmd_tully_report", ROOT / "tools" / "validation" / "tully_report.py")
ensembles = _load(
    "gqteashmd_tully_ensembles", ROOT / "tools" / "sharc_compare" / "tully_ensembles.py"
)


def test_tully_report_covers_every_point_and_the_low_energy_dac_point():
    common = report.load_common()
    names = [common.ids(p) for p in common.REPORT_POINTS]
    assert names == [common.ids(p) for p in common.SAC_POINTS] + ["DAC-lnE=-2"] + [
        common.ids(p) for p in (*common.DAC_POINTS, *common.ECR_POINTS)
    ]
    assert common.DAC_LOW not in common.POINTS  # report-only, never asserted
    assert callable(report.main)


def test_sharc_ensemble_script_points_and_z_statistic():
    assert [p.name for p in ensembles.POINTS] == ["SAC-k20-none", "ECR-k20-edc"]
    assert all(p.edc_convention == "sharc" for p in ensembles.POINTS)
    assert ensembles.two_proportion_z(5, 10, 5, 10) == 0.0
    assert ensembles.two_proportion_z(0, 10, 0, 10) == 0.0
    z = ensembles.two_proportion_z(60, 100, 40, 100)
    assert math.isclose(z, 0.2 / math.sqrt(0.25 * 0.02), rel_tol=1e-12)
    assert callable(ensembles.main)
