import numpy as np
import pytest

from gqteashmd.analysis.energy import max_drift_eV, max_jump_eV, summary_eV
from gqteashmd.core.units import HARTREE_IN_EV


def test_jump_and_drift():
    e = np.array([-1.0, -1.0 + 1e-4, -1.0 + 3e-4, -1.0 - 1e-3])
    assert max_jump_eV(e) == pytest.approx(1.3e-3 * HARTREE_IN_EV)
    assert max_drift_eV(e) == pytest.approx(1e-3 * HARTREE_IN_EV)
    assert max_jump_eV(e[:1]) == 0.0


def test_summary():
    assert summary_eV([]) == {"median": None, "p95": None, "max": None}
    s = summary_eV([1.0, 2.0, 3.0])
    assert s["median"] == 2.0 and s["max"] == 3.0 and 2.0 < s["p95"] <= 3.0
