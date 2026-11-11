"""BackendModel: the dynamics core over an electronic backend (spec section 3.1)."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from gqteashmd.dynamics.driver import initial_state, step
from gqteashmd.dynamics.lvc import LVCModel, lvc_wigner_samples
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.randomness import ReplaySource
from gqteashmd.overlaps.tracking import lowdin_orthonormalize
from gqteashmd.testing.fake_backend import ModelBackend
from gqteashmd.trajectory.model import BackendModel

LVC = Path(__file__).resolve().parents[1] / "data" / "lvc" / "lvc_singlets.template"
N_STEPS = 120


def _model():
    return LVCModel.from_template(LVC, spin_orbit=False)


class _Lowdin:
    """The direct model as BackendModel presents it with coupling = "overlap": the same Lowdin
    step on the overlaps, and zero NACs (BackendModel requests none for "overlap")."""

    def __init__(self, model):
        self._m = model
        self.n_states, self.n_dof, self.masses = model.n_states, model.n_dof, model.masses

    @property
    def state_counts(self):
        return self._m.state_counts

    def evaluate(self, q, previous):
        snap = self._m.evaluate(q, previous)
        snap = dataclasses.replace(snap, nacs=np.zeros_like(snap.nacs))
        if snap.overlaps is None:
            return snap
        return dataclasses.replace(snap, overlaps=lowdin_orthonormalize(snap.overlaps))


def _start(model):
    q, v = lvc_wigner_samples(model, 1, np.random.default_rng(11))
    return q[0], v[0]


def _direct(model, options, active, stepper=None):
    # The samples come from the bare LVC model (lvc_wigner_samples reads its private state); the
    # stepping uses ``stepper`` (the Lowdin wrapper for coupling = "overlap").
    q0, v0 = _start(model)
    stepper = model if stepper is None else stepper
    state = initial_state(stepper, q0, v0, active, options=options)
    source = ReplaySource([0.0] * N_STEPS)  # xi = 0: hop whenever a probability is > 0
    records = []
    for _ in range(N_STEPS):
        state, record = step(stepper, state, options, source)
        records.append(record)
    return records


def _adapted(model, options, active):
    q0, v0 = _start(model)
    adapter = BackendModel(ModelBackend(model), model.masses, coupling=options.coupling)
    adapter.active = active
    state = initial_state(adapter, q0, v0, active, options=options)
    source = ReplaySource([0.0] * N_STEPS)
    records = []
    for _ in range(N_STEPS):
        adapter.active = state.active
        state, record = step(adapter, state, options, source)
        if record.hop_to is not None:
            state = adapter.refresh_active_gradient(state)
        records.append(record)
    return records


@pytest.mark.parametrize("coupling", ["nacdr", "overlap"])
def test_backend_model_reproduces_the_direct_model_bit_for_bit(coupling):
    options = DynamicsOptions(dt_au=10.0, coupling=coupling)
    model = _model()
    stepper = None if coupling == "nacdr" else _Lowdin(model)
    direct = _direct(model, options, 2, stepper)
    adapted = _adapted(model, options, 2)
    assert any(r.hop_to is not None for r in direct)
    for a, b in zip(direct, adapted, strict=True):
        for name in ("q", "v", "c", "energies", "probabilities"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        assert (a.active, a.e_total, a.hop_from, a.hop_to, a.frustrated) == (
            b.active,
            b.e_total,
            b.hop_from,
            b.hop_to,
            b.frustrated,
        )


def test_only_the_active_gradient_is_computed_and_no_nan_reaches_a_record():
    records = _adapted(_model(), DynamicsOptions(dt_au=10.0, coupling="overlap"), 2)
    assert any(r.hop_to is not None for r in records)
    for r in records:
        for name in ("q", "v", "c", "energies", "probabilities"):
            assert np.all(np.isfinite(getattr(r, name)))
        assert np.isfinite(r.e_total)


def test_snapshot_layout():
    model = _model()
    adapter = BackendModel(ModelBackend(model), model.masses, coupling="nacdr")
    adapter.active = 1
    snap = adapter.evaluate(np.array(model.reference_q), None)
    assert np.all(np.isfinite(snap.gradients[1]))
    assert np.all(np.isnan(snap.gradients[[0, 2]]))
    np.testing.assert_array_equal(snap.nacs, -snap.nacs.transpose(1, 0, 2))
    np.testing.assert_array_equal(snap.hamiltonian, np.diag(snap.energies).astype(complex))
    assert snap.overlaps is None and adapter.min_singular_value is None
    assert adapter.rung_changed is False  # a first geometry is never a rung change


def test_meta_signals():
    model = _model()
    backend = ModelBackend(model, faults={2: "low_overlap", 3: "rung"})
    adapter = BackendModel(backend, model.masses, coupling="overlap")
    q = np.array(model.reference_q)
    first = adapter.evaluate(q, None)
    adapter.evaluate(q + 0.001, first)
    assert adapter.min_singular_value == 0.1 and adapter.rung_changed is False
    adapter.evaluate(q + 0.002, first)
    assert adapter.rung_changed is True


def test_missing_overlaps_with_a_previous_geometry_is_an_error():
    model = _model()
    adapter = BackendModel(
        ModelBackend(model, faults={2: "no_overlaps"}), model.masses, coupling="overlap"
    )
    q = np.array(model.reference_q)
    first = adapter.evaluate(q, None)
    with pytest.raises(ValueError, match="no overlaps"):
        adapter.evaluate(q + 0.001, first)


def test_construction_errors():
    model = _model()
    with pytest.raises(ValueError, match="nacdr"):
        BackendModel(ModelBackend(model, nacs=False), model.masses, coupling="nacdr")
    with pytest.raises(ValueError, match="coupling"):
        BackendModel(ModelBackend(model), model.masses, coupling="ddr")
    with pytest.raises(ValueError, match="masses"):
        BackendModel(ModelBackend(model), model.masses[:-1], coupling="overlap")
