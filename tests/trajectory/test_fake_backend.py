"""The fake backend used by the fast tests of the dynamics stage."""

from pathlib import Path

import numpy as np
import pytest

from gqteashmd.core.states import StateLabel
from gqteashmd.dynamics.lvc import LVCModel
from gqteashmd.electronic.base import Request
from gqteashmd.errors import ElectronicStructureError
from gqteashmd.testing.fake_backend import ModelBackend, SimulatedCrash

LVC = Path(__file__).resolve().parents[1] / "data" / "lvc" / "lvc_singlets.template"
S0, S1, S2 = StateLabel(1, 0), StateLabel(1, 1), StateLabel(1, 2)


def _model():
    return LVCModel.from_template(LVC, spin_orbit=False)


def _coords(model, shift=0.0):
    q = np.array(model.reference_q, dtype=float)
    q[0] += shift
    return q.reshape(-1, 3)


def test_energies_gradients_and_nacs_are_the_models():
    model = _model()
    backend = ModelBackend(model)
    result = backend.compute(
        Request(_coords(model), gradients=frozenset({S1}), nacs=frozenset({(S0, S2)}))
    )
    snap = model.evaluate(_coords(model).reshape(-1), None)
    assert result.states == (S0, S1, S2)
    assert [result.energies[s] for s in (S0, S1, S2)] == list(snap.energies)
    np.testing.assert_array_equal(result.gradients[S1].reshape(-1), snap.gradients[1])
    np.testing.assert_array_equal(result.nacs[(S0, S2)].reshape(-1), snap.nacs[0, 2])
    assert set(result.gradients) == {S1}
    assert result.overlaps is None and result.meta["casscf"] == "SCF orbitals"


def test_a_spin_orbit_request_is_refused():
    """ModelBackend has no spin-orbit couplings: a request for them is an error, not a result
    without a Hamiltonian (sub-project 3 gets a signal)."""
    model = _model()
    backend = ModelBackend(model)
    assert backend.capabilities.soc is False
    with pytest.raises(ValueError, match="spin-orbit"):
        backend.compute(Request(_coords(model), soc=True))
    assert backend.calls == 0  # refused before anything is evaluated
    assert backend.compute(Request(_coords(model))).hamiltonian is None


def test_overlaps_follow_the_previous_geometry_and_state_round_trips(tmp_path):
    model = _model()
    backend = ModelBackend(model)
    backend.compute(Request(_coords(model)))
    backend.save_state(tmp_path / "e.npz")
    second = backend.compute(Request(_coords(model, 0.01), overlaps=True))
    other = ModelBackend(model)
    other.load_state(tmp_path / "e.npz")
    again = other.compute(Request(_coords(model, 0.01), overlaps=True))
    np.testing.assert_array_equal(second.overlaps, again.overlaps)
    assert second.meta["casscf"] == "previous orbitals"
    assert 0.9 < second.meta["overlap_min_singular_value"] <= 1.0 + 1e-12


def test_compute_more_returns_extra_gradients_at_the_last_geometry():
    model = _model()
    backend = ModelBackend(model)
    with pytest.raises(ValueError):
        backend.compute_more(gradients=frozenset({S2}))
    backend.compute(Request(_coords(model), gradients=frozenset({S1})))
    more = backend.compute_more(gradients=frozenset({S2}))
    snap = model.evaluate(_coords(model).reshape(-1), None)
    np.testing.assert_array_equal(more.gradients[S2].reshape(-1), snap.gradients[2])


@pytest.mark.parametrize("kind", ["error", "retryable"])
def test_error_faults(kind):
    model = _model()
    backend = ModelBackend(model, faults={2: kind})
    backend.compute(Request(_coords(model)))
    with pytest.raises(ElectronicStructureError) as info:
        backend.compute(Request(_coords(model, 0.01)))
    assert info.value.retryable is (kind == "retryable")
    with pytest.raises(ValueError):
        backend.compute_more(gradients=frozenset({S1}))  # a failed compute discards the last


def test_meta_faults_and_calls_offset():
    model = _model()
    backend = ModelBackend(model, faults={3: "low_overlap", 4: "rung", 5: "no_overlaps"}, calls=1)
    backend.compute(Request(_coords(model)))  # call 2
    r3 = backend.compute(Request(_coords(model, 0.01), overlaps=True))
    r4 = backend.compute(Request(_coords(model, 0.02), overlaps=True))
    r5 = backend.compute(Request(_coords(model, 0.03), overlaps=True))
    assert backend.calls == 5
    assert r3.meta["overlap_min_singular_value"] == 0.1 and r3.overlaps is not None
    assert r4.meta["casscf"] == "SCF orbitals"
    assert r5.overlaps is None


def test_crash_fault_and_capabilities():
    model = _model()
    backend = ModelBackend(model, faults={1: "crash"}, nacs=False)
    assert backend.capabilities.nacs is False and backend.capabilities.overlaps is True
    with pytest.raises(SimulatedCrash):
        backend.compute(Request(_coords(model)))


def test_reset_forgets_the_previous_geometry():
    model = _model()
    backend = ModelBackend(model)
    backend.compute(Request(_coords(model)))
    backend.reset()
    assert backend.compute(Request(_coords(model), overlaps=True)).overlaps is None


def test_nan_and_more_error_faults():
    model = _model()
    backend = ModelBackend(model, faults={1: "nan_energy", 2: "nan_gradient", 3: "more_error"})
    r1 = backend.compute(Request(_coords(model), gradients=frozenset({S1})))
    assert all(np.isnan(e) for e in r1.energies.values())
    assert np.all(np.isfinite(r1.gradients[S1]))
    r2 = backend.compute(Request(_coords(model, 0.01), gradients=frozenset({S1})))
    assert all(np.isfinite(e) for e in r2.energies.values())
    assert np.all(np.isnan(r2.gradients[S1]))
    backend.compute(Request(_coords(model, 0.02)))
    with pytest.raises(ElectronicStructureError) as info:
        backend.compute_more(gradients=frozenset({S1}))
    assert info.value.retryable is False
    backend.compute(Request(_coords(model, 0.03)))  # no fault: compute_more works again
    backend.compute_more(gradients=frozenset({S1}))


def test_energy_jump_fault_shifts_every_energy():
    from gqteashmd.testing.fake_backend import ENERGY_JUMP_SHIFT_EH

    model = _model()
    q = _coords(model)
    plain = ModelBackend(model).compute(Request(q))
    shifted = ModelBackend(model, faults={1: "energy_jump"}).compute(Request(q))
    for label, e in plain.energies.items():
        assert shifted.energies[label] == pytest.approx(e + ENERGY_JUMP_SHIFT_EH, abs=1e-15)


# ---------------------------------------------------------------- spin-orbit models (PR4)

LVC_SOC = LVC.parent / "lvc_soc.template"
T1, T2 = StateLabel(3, 1), StateLabel(3, 2)
SPIN_FREE = (S0, S1, T1, T2)
ROWS = (0, 1, 2, 3)  # the Ms = -1 expanded rows of S0, S1, T1, T2 (n = 2, m = 2)


def _soc_model():
    return LVCModel.from_template(LVC_SOC, spin_orbit=True)


def test_a_spin_orbit_model_is_served_as_its_spin_free_view():
    model = _soc_model()
    backend = ModelBackend(model)
    assert backend.states == SPIN_FREE
    assert backend.capabilities.soc is True
    q = _coords(model)
    result = backend.compute(Request(q, gradients=frozenset(SPIN_FREE), soc=True))
    snap = model.evaluate(q.reshape(-1), None)
    assert result.states == SPIN_FREE
    np.testing.assert_allclose(
        [result.energies[s] for s in SPIN_FREE], snap.energies[list(ROWS)], rtol=0, atol=1e-12
    )
    for label, row in zip(SPIN_FREE, ROWS, strict=True):
        np.testing.assert_allclose(
            result.gradients[label].reshape(-1), snap.gradients[row], rtol=0, atol=1e-12
        )
    assert result.hamiltonian is not None and result.hamiltonian.shape == (8, 8)
    np.testing.assert_allclose(result.hamiltonian, snap.hamiltonian, rtol=0, atol=1e-12)
    assert result.hamiltonian_labels == (
        "S0", "S1", "T1(-1)", "T2(-1)", "T1(0)", "T2(0)", "T1(+1)", "T2(+1)"
    )  # fmt: skip
    assert backend.compute(Request(q)).hamiltonian is None  # only when requested


def test_the_spin_free_overlaps_and_nacs_have_exact_zero_cross_blocks():
    model = _soc_model()
    backend = ModelBackend(model)
    first = model.evaluate(_coords(model).reshape(-1), None)
    backend.compute(Request(_coords(model)))
    result = backend.compute(
        Request(
            _coords(model, 0.01),
            overlaps=True,
            nacs=frozenset({(S0, S1), (S0, T1), (T1, T2)}),
            soc=True,
        )
    )
    snap = model.evaluate(_coords(model, 0.01).reshape(-1), first)
    assert result.overlaps.shape == (4, 4)
    assert np.all(result.overlaps[:2, 2:] == 0.0) and np.all(result.overlaps[2:, :2] == 0.0)
    np.testing.assert_allclose(
        result.overlaps, snap.overlaps[np.ix_(ROWS, ROWS)], rtol=0, atol=1e-12
    )
    assert np.all(result.nacs[(S0, T1)] == 0.0)
    np.testing.assert_allclose(result.nacs[(T1, T2)].reshape(-1), snap.nacs[2, 3], atol=1e-12)
    np.testing.assert_allclose(result.nacs[(S0, S1)].reshape(-1), snap.nacs[0, 1], atol=1e-12)
    assert 0.9 < result.meta["overlap_min_singular_value"] <= 1.0 + 1e-12
    np.testing.assert_allclose(result.hamiltonian, snap.hamiltonian, rtol=0, atol=1e-12)


def test_compute_more_serves_triplet_gradients():
    model = _soc_model()
    backend = ModelBackend(model)
    backend.compute(Request(_coords(model), soc=True))
    more = backend.compute_more(gradients=frozenset({T2}))
    snap = model.evaluate(_coords(model).reshape(-1), None)
    assert set(more.gradients) == {T2}
    np.testing.assert_allclose(more.gradients[T2].reshape(-1), snap.gradients[3], atol=1e-12)


@pytest.mark.parametrize(("flag", "spin_orbit"), [(None, False), ("0", False), ("1", True)])
def test_the_canned_model_switches_on_spin_orbit_coupling(monkeypatch, flag, spin_orbit):
    from gqteashmd.testing import canned_lvc

    monkeypatch.setenv(canned_lvc.LVC_TEMPLATE_ENV, str(LVC_SOC))
    if flag is None:
        monkeypatch.delenv(canned_lvc.LVC_SOC_ENV, raising=False)
    else:
        monkeypatch.setenv(canned_lvc.LVC_SOC_ENV, flag)
    assert canned_lvc.LVC_SOC_ENV == "GQTEASHMD_CANNED_LVC_SOC"
    model = canned_lvc.canned_model()
    assert model.spin_orbit is spin_orbit and model.state_counts == (2, 0, 2)
