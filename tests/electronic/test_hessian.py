import numpy as np
import pytest

from gqteashmd.core.states import StateLabel
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.factory import ground_state_settings
from gqteashmd.electronic.pyscf_backend import PySCFBackend

pytestmark = pytest.mark.pyscf
S0 = StateLabel(1, 0)
WATER = ("O", "H", "H")
WATER_BOHR = np.array([[0.0, 0.0, 0.2043], [0.0, 1.4266, -0.8780], [0.0, -1.4266, -0.8780]])


def _settings(make_settings, **kw):
    fields = {"method": "tda", "basis": "sto-3g", "functional": "hf", "singlets": 3, "casscf": None}
    fields.update(kw)
    return make_settings(**fields)


def test_analytic_hessian_matches_finite_differences_of_gradients(make_settings):
    settings = ground_state_settings(_settings(make_settings, scf_conv_tol=1e-12))
    assert settings.singlets == 1
    backend = PySCFBackend(MoleculeSpec(WATER), settings)
    assert backend.capabilities.hessian is True
    h = backend.ground_state_hessian(WATER_BOHR)
    assert h.shape == (9, 9)
    step = 1e-3
    fd = np.zeros((9, 9))
    for k in range(9):
        plus, minus = WATER_BOHR.copy().ravel(), WATER_BOHR.copy().ravel()
        plus[k] += step
        minus[k] -= step
        g = []
        for c in (plus, minus):
            fresh = PySCFBackend(MoleculeSpec(WATER), settings)
            request = Request(c.reshape(3, 3), gradients=frozenset({S0}))
            g.append(fresh.compute(request).gradients[S0].ravel())
        fd[:, k] = (g[0] - g[1]) / (2 * step)
    error = np.abs(h - 0.5 * (fd + fd.T)).max()
    print(f"max|H - FD| = {error:.3e}")
    assert error < 1e-4


def test_hessian_does_not_disturb_the_snapshot(make_settings):
    backend = PySCFBackend(MoleculeSpec(WATER), ground_state_settings(_settings(make_settings)))
    backend.compute(Request(WATER_BOHR))
    snapshot = backend._snapshot
    backend.ground_state_hessian(WATER_BOHR + 0.05)
    assert backend._snapshot is snapshot


def test_casscf_has_no_analytic_hessian(ethylene_symbols, ethylene_bohr, make_settings):
    settings = make_settings(basis="sto-3g")
    assert ground_state_settings(settings) is settings  # the SA defines S0: keep all states
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings)
    assert backend.capabilities.hessian is False
    with pytest.raises(ValueError, match="casscf"):
        backend.ground_state_hessian(ethylene_bohr)
