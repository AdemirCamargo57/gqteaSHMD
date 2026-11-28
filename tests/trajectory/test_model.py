"""BackendModel: the dynamics core over an electronic backend (spec section 3.1)."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from gqteashmd.core.states import StateLabel
from gqteashmd.dynamics.driver import initial_state, run_trajectory, step
from gqteashmd.dynamics.lvc import LVCModel, lvc_wigner_samples
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.randomness import ReplaySource
from gqteashmd.dynamics.representation import (
    diagonal_basis,
    diagonal_energies,
    mch_state,
    selection_mask,
)
from gqteashmd.dynamics.spin import expanded_states
from gqteashmd.electronic.base import Request
from gqteashmd.errors import ElectronicStructureError
from gqteashmd.overlaps.tracking import lowdin_orthonormalize
from gqteashmd.testing.fake_backend import ModelBackend
from gqteashmd.trajectory.model import BackendModel
from gqteashmd.trajectory.spin_expansion import expand_overlaps, spin_free_index

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


@pytest.mark.parametrize("kind", ["nan_energy", "nan_gradient"])
def test_non_finite_data_is_an_electronic_failure(kind):
    model = _model()
    adapter = BackendModel(ModelBackend(model, faults={1: kind}), model.masses, coupling="overlap")
    adapter.active = 1
    with pytest.raises(ElectronicStructureError) as info:
        adapter.evaluate(model.reference_q, None)
    assert info.value.retryable is False


def test_non_finite_nacs_are_an_electronic_failure_with_nacdr():
    model = _model()
    adapter = BackendModel(
        ModelBackend(model, faults={1: "nan_nac"}), model.masses, coupling="nacdr"
    )
    adapter.active = 1
    with pytest.raises(ElectronicStructureError, match="non-finite NAC") as info:
        adapter.evaluate(model.reference_q, None)
    assert info.value.retryable is False
    # coupling = "overlap" requests no NACs, so the fault leaves nothing to check
    adapter = BackendModel(
        ModelBackend(model, faults={1: "nan_nac"}), model.masses, coupling="overlap"
    )
    adapter.active = 1
    adapter.evaluate(model.reference_q, None)


def test_a_failed_gradient_refresh_is_an_electronic_failure():
    model = _model()
    adapter = BackendModel(
        ModelBackend(model, faults={1: "more_error"}), model.masses, coupling="overlap"
    )
    adapter.active = 1
    q0, v0 = _start(model)
    state = initial_state(
        adapter, q0, v0, 1, options=DynamicsOptions(dt_au=10.0, coupling="overlap")
    )
    with pytest.raises(ElectronicStructureError) as info:
        adapter.refresh_active_gradient(state)
    assert info.value.retryable is False


# ---------------------------------------------------------------- triplets and SOC (spec 3.2)

LVC_SOC = LVC.parent / "lvc_soc.template"
S0, S1, T1, T2 = StateLabel(1, 0), StateLabel(1, 1), StateLabel(3, 1), StateLabel(3, 2)
SPIN_FREE = (S0, S1, T1, T2)
ESELECT_EH = 0.05
SOC_STEPS = 30


def _soc_model():
    return LVCModel.from_template(LVC_SOC, spin_orbit=True)


def _soc_start(model):
    q, v = lvc_wigner_samples(model, 1, np.random.default_rng(2))
    return q[0], v[0]


def _soc_options(eselect=None):
    return DynamicsOptions(
        dt_au=10.0, coupling="overlap", gradient_transformation="none", eselect_Eh=eselect
    )


def _xi():
    # small numbers: from this start they give a singlet -> triplet hop at step 22 (p 0.144)
    return ReplaySource(list(0.1 * np.random.default_rng(2).random(SOC_STEPS)))


class _Spy(ModelBackend):
    """Records the gradients of every compute and compute_more; ``drop`` removes one label from
    the next compute_more result (a backend that loses a requested gradient)."""

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.computed: list[frozenset] = []
        self.more: list[frozenset] = []
        self.drop = None

    def compute(self, request):
        self.computed.append(request.gradients)
        assert request.soc is True and request.overlaps is True and not request.nacs
        return super().compute(request)

    def compute_more(self, gradients=frozenset(), nacs=frozenset()):
        self.more.append(frozenset(gradients))
        result = super().compute_more(gradients, nacs)
        if self.drop is not None:
            kept = {k: g for k, g in result.gradients.items() if k != self.drop}
            result = dataclasses.replace(result, gradients=kept)
        return result


class _Hamiltonian(ModelBackend):
    """Replaces the returned Hamiltonian: ``None`` (missing) or with a NaN element."""

    def __init__(self, model, kind):
        super().__init__(model)
        self.kind = kind

    def compute(self, request):
        result = super().compute(request)
        if self.kind == "missing":
            return dataclasses.replace(result, hamiltonian=None)
        h = np.array(result.hamiltonian, copy=True)
        h[2, 4] = complex(np.nan, 0.0)
        return dataclasses.replace(result, hamiltonian=h)


def test_the_triplet_adapter_attributes():
    model = _soc_model()
    adapter = BackendModel(ModelBackend(model), model.masses, coupling="overlap")
    assert adapter.labels == SPIN_FREE
    assert (adapter.n_states, adapter.n_singlets, adapter.n_triplets) == (8, 2, 2)
    assert adapter.n_spin_free == 4 and adapter.state_counts == (2, 0, 2)
    assert adapter.u is None and adapter.eselect_Eh is None
    with pytest.raises(ValueError, match="coupling"):
        BackendModel(ModelBackend(model), model.masses, coupling="nacdr")
    singlets = _model()
    plain = BackendModel(ModelBackend(singlets), singlets.masses, coupling="overlap")
    assert plain.state_counts == (3,) and plain.n_triplets == 0 and plain.n_spin_free == 3


def test_the_triplet_snapshots_round_trip_the_model():
    model = _soc_model()
    adapter = BackendModel(ModelBackend(model), model.masses, coupling="overlap")
    q0, v0 = _soc_start(model)
    q1 = q0 + 5.0 * v0
    mine0 = adapter.evaluate(q0, None)
    mine1 = adapter.evaluate(q1, mine0)
    theirs0 = model.evaluate(q0, None)
    theirs1 = model.evaluate(q1, theirs0)
    for mine, theirs in ((mine0, theirs0), (mine1, theirs1)):
        for name in ("energies", "gradients", "hamiltonian"):
            np.testing.assert_allclose(
                getattr(mine, name), getattr(theirs, name), rtol=0, atol=1e-12
            )
        assert mine.nacs.shape == (8, 8, model.n_dof) and np.all(mine.nacs == 0.0)
    assert mine0.overlaps is None
    np.testing.assert_allclose(mine1.overlaps, theirs1.overlaps, rtol=0, atol=1e-10)
    assert np.all(mine1.overlaps[:2, 2:] == 0.0) and np.all(mine1.overlaps[2:, :2] == 0.0)
    assert 0.5 < adapter.min_singular_value <= 1.0 + 1e-12


@pytest.mark.parametrize("eselect", [None, ESELECT_EH])
def test_a_triplet_trajectory_through_the_adapter_follows_the_model(eselect):
    model = _soc_model()
    options = _soc_options(eselect)
    q0, v0 = _soc_start(model)
    direct: list = []
    state = initial_state(model, q0, v0, 1, options=options)
    run_trajectory(model, state, options, _xi(), max_steps=SOC_STEPS, on_step=direct.append)

    adapter = BackendModel(
        ModelBackend(model), model.masses, coupling="overlap", eselect_Eh=eselect
    )
    adapted: list = []
    state = initial_state(adapter, q0, v0, 1, options=options)
    if eselect is None:  # the core's own loop: no active, U or refresh bookkeeping is needed
        run_trajectory(adapter, state, options, _xi(), max_steps=SOC_STEPS, on_step=adapted.append)
    else:  # the bookkeeping of the runner: active and U before every step, refresh after a hop
        source = _xi()
        for _ in range(SOC_STEPS):
            adapter.active, adapter.u = state.active, state.u
            state, record = step(adapter, state, options, source)
            if record.hop_to is not None:
                state = adapter.refresh_active_gradient(state)
            adapted.append(record)
    spin_free = [1] + [spin_free_index(2, 2, mch_state(r.u, r.active)) for r in direct]
    hops = [k for k, r in enumerate(direct) if r.hop_to is not None]
    assert len(hops) >= 2
    assert any(spin_free[k] < 2 <= spin_free[k + 1] for k in hops)  # a singlet -> triplet hop
    assert len(direct) == len(adapted) == SOC_STEPS
    for a, b in zip(direct, adapted, strict=True):
        assert (a.active, a.hop_from, a.hop_to) == (b.active, b.hop_from, b.hop_to)
        for name in ("q", "v", "c"):
            np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=0, atol=1e-10)


def test_without_eselect_every_spin_free_gradient_comes_with_compute():
    model = _soc_model()
    spy = _Spy(model)
    adapter = BackendModel(spy, model.masses, coupling="overlap")
    options = _soc_options()
    q0, v0 = _soc_start(model)
    state = initial_state(adapter, q0, v0, 1, options=options)
    hops = 0
    source = _xi()
    for _ in range(SOC_STEPS):
        adapter.active, adapter.u = state.active, state.u
        state, record = step(adapter, state, options, source)
        if record.hop_to is not None:
            hops += 1
            state = adapter.refresh_active_gradient(state)
    assert hops >= 1
    assert spy.computed == [frozenset(SPIN_FREE)] * (SOC_STEPS + 1)
    assert spy.more == []


def _kept_labels(energies, hamiltonian, u, active):
    """The spin-free states the core's mask keeps, recomputed as the driver does (steps 4-5)."""
    e_diag, _ = diagonal_basis(hamiltonian, u)
    mask = selection_mask(energies, e_diag[active], ESELECT_EH)
    return frozenset(SPIN_FREE[spin_free_index(2, 2, k)] for k in np.flatnonzero(mask))


def test_with_eselect_the_gradients_fetched_are_the_cores_mask():
    model = _soc_model()
    spy = _Spy(model)
    adapter = BackendModel(spy, model.masses, coupling="overlap", eselect_Eh=ESELECT_EH)
    options = _soc_options(ESELECT_EH)
    q0, v0 = _soc_start(model)
    state = initial_state(adapter, q0, v0, 1, options=options)
    assert spy.computed == [frozenset(SPIN_FREE)] and spy.more == []  # PR3: no U0 yet
    dropped = 0
    for _ in range(SOC_STEPS):
        adapter.active, adapter.u = state.active, state.u
        spy.computed.clear()
        spy.more.clear()
        new_state, _ = step(adapter, state, options, ReplaySource([0.99]))
        snap = new_state.snapshot
        expected = _kept_labels(snap.energies, snap.hamiltonian, state.u, state.active)
        assert spy.computed == [frozenset()]
        assert spy.more == ([expected] if expected else [])
        missing = [s for s in SPIN_FREE if s not in expected]
        dropped += len(missing)
        for label in missing:  # the tripwire stays on the rows the mask drops
            assert np.all(np.isnan(snap.gradients[SPIN_FREE.index(label)]))
        for label in expected:
            assert np.all(np.isfinite(snap.gradients[SPIN_FREE.index(label)]))
        state = new_state
    assert dropped > 0  # the window dropped at least one state somewhere


def _state_with_window(adapter, model):
    """A state after one step on the S1-dominant diagonal state (the window drops S0), then
    moved to the S0-dominant diagonal state as an accepted hop would."""
    options = _soc_options(ESELECT_EH)
    q0, v0 = _soc_start(model)
    state = initial_state(adapter, q0, v0, 1, options=options)
    adapter.active, adapter.u = state.active, state.u
    state, _ = step(adapter, state, options, ReplaySource([0.99]))
    assert np.all(np.isnan(state.snapshot.gradients[0]))  # S0 is outside the window
    low = int(np.argmin(diagonal_energies(state.snapshot.hamiltonian, state.u)))
    return dataclasses.replace(state, active=low)


def test_the_refresh_after_a_hop_fetches_the_new_windows_missing_gradients():
    model = _soc_model()
    spy = _Spy(model)
    adapter = BackendModel(spy, model.masses, coupling="overlap", eselect_Eh=ESELECT_EH)
    hopped = _state_with_window(adapter, model)
    snap = hopped.snapshot
    expected = _kept_labels(snap.energies, snap.hamiltonian, hopped.u, hopped.active)
    have = {s for i, s in enumerate(SPIN_FREE) if np.all(np.isfinite(snap.gradients[i]))}
    spy.more.clear()
    refreshed = adapter.refresh_active_gradient(hopped)
    assert spy.more == [frozenset(expected - have)] and S0 in spy.more[0]
    for label in expected:
        assert np.all(np.isfinite(refreshed.snapshot.gradients[SPIN_FREE.index(label)]))
    theirs = model.evaluate(np.asarray(hopped.q), None)
    np.testing.assert_allclose(
        refreshed.snapshot.gradients[0], theirs.gradients[0], rtol=0, atol=1e-12
    )


def test_a_gradient_missing_after_the_refresh_is_an_electronic_failure():
    model = _soc_model()
    spy = _Spy(model)
    adapter = BackendModel(spy, model.masses, coupling="overlap", eselect_Eh=ESELECT_EH)
    hopped = _state_with_window(adapter, model)
    spy.drop = S0
    with pytest.raises(ElectronicStructureError, match="S0") as info:
        adapter.refresh_active_gradient(hopped)
    assert info.value.retryable is False


@pytest.mark.parametrize("kind", ["missing", "nan"])
def test_a_missing_or_non_finite_hamiltonian_is_an_electronic_failure(kind):
    model = _soc_model()
    adapter = BackendModel(_Hamiltonian(model, kind), model.masses, coupling="overlap")
    with pytest.raises(ElectronicStructureError, match="Hamiltonian") as info:
        adapter.evaluate(np.array(model.reference_q), None)
    assert info.value.retryable is False


def test_the_active_spin_free_state():
    model = _soc_model()
    adapter = BackendModel(ModelBackend(model), model.masses, coupling="overlap")
    t2_plus = 2 + 2 * 2 + 1  # n + M_index m + (J - 1) for T2(+1)
    assert adapter.active_spin_free(np.eye(8, dtype=complex), t2_plus) == 3
    assert adapter.active_spin_free(None, 1) == 1
    singlets = _model()
    plain = BackendModel(ModelBackend(singlets), singlets.masses, coupling="overlap")
    assert plain.active_spin_free(None, 2) == 2


def test_the_runner_hands_the_tracked_u_to_the_adapter(tmp_path, monkeypatch):
    from gqteashmd.trajectory import runner as runner_module
    from gqteashmd.trajectory.runner import Progress, RunSettings, TrajectoryRunner

    model = _model()
    adapter = BackendModel(ModelBackend(model), model.masses, coupling="overlap")
    options = DynamicsOptions(dt_au=10.0, coupling="overlap")
    runner = TrajectoryRunner(adapter, RunSettings(options, tmax_au=100.0), tmp_path / "r")
    q0, v0 = _start(model)
    state = initial_state(adapter, q0, v0, 1, options=options)
    tracked = dataclasses.replace(state, u=np.eye(3, dtype=complex), active=2)
    seen = {}

    class _Stop(BaseException):
        pass

    def spy_step(m, s, o, r):
        seen["u"], seen["active"] = m.u, m.active
        raise _Stop

    monkeypatch.setattr(runner_module, "step", spy_step)
    with pytest.raises(_Stop):
        runner._advance(tracked, ReplaySource([0.5]), Progress())
    assert seen["u"] is tracked.u and seen["active"] == 2


# ---------------------------------------------------------------- fix round (Task 4 review)


class _Replaced(ModelBackend):
    """Replaces fields of every compute result (``fields`` maps a field to a function of the
    result giving its new value); ``states`` and ``soc`` override the backend's own."""

    def __init__(self, model, *, states=None, soc=None, **fields):
        super().__init__(model)
        self._override_states, self._soc, self._fields = states, soc, fields

    @property
    def states(self):
        return super().states if self._override_states is None else self._override_states

    @property
    def capabilities(self):
        own = super().capabilities
        return own if self._soc is None else dataclasses.replace(own, soc=self._soc)

    def compute(self, request):
        result = super().compute(request)
        changes = {name: f(result) for name, f in self._fields.items()}
        return dataclasses.replace(result, **changes) if changes else result


def test_the_hamiltonian_labels_must_be_the_canonical_order():
    model = _soc_model()
    canonical = tuple(s.label() for s in expanded_states((2, 0, 2)))
    assert (
        ModelBackend(model)
        .compute(Request(np.array(model.reference_q).reshape(-1, 3), soc=True))
        .hamiltonian_labels
        == canonical
    )
    swapped = canonical[:2] + (canonical[4], canonical[3], canonical[2]) + canonical[5:]
    backend = _Replaced(model, hamiltonian_labels=lambda r: swapped)
    adapter = BackendModel(backend, model.masses, coupling="overlap")
    with pytest.raises(ElectronicStructureError, match="Hamiltonian") as info:
        adapter.evaluate(np.array(model.reference_q), None)
    assert info.value.retryable is False
    assert ", ".join(swapped) in str(info.value) and ", ".join(canonical) in str(info.value)
    unlabelled = _Replaced(model, hamiltonian_labels=lambda r: None)  # no labels: not checked
    BackendModel(unlabelled, model.masses, coupling="overlap").evaluate(model.reference_q, None)


def test_a_non_hermitian_hamiltonian_is_an_electronic_failure():
    model = _soc_model()

    def skewed(result):
        h = np.array(result.hamiltonian, copy=True)
        h[1, 2] += 1e-9  # its partner h[2, 1] is unchanged
        return h

    adapter = BackendModel(_Replaced(model, hamiltonian=skewed), model.masses, coupling="overlap")
    with pytest.raises(ElectronicStructureError, match="Hermitian") as info:
        adapter.evaluate(np.array(model.reference_q), None)
    assert info.value.retryable is False

    def nearly(result):  # 1e-13 Eh is round-off: accepted
        h = np.array(result.hamiltonian, copy=True)
        h[1, 2] += 1e-13
        return h

    BackendModel(_Replaced(model, hamiltonian=nearly), model.masses, coupling="overlap").evaluate(
        np.array(model.reference_q), None
    )


def test_the_constructor_refusals_with_triplets():
    model = _soc_model()
    with pytest.raises(ValueError, match=r"S0\.\.S\(n-1\) followed by T1\.\.Tm; got S0, T1, S1"):
        BackendModel(_Replaced(model, states=(S0, T1, S1, T2)), model.masses, coupling="overlap")
    with pytest.raises(ValueError, match="Capabilities.soc"):
        BackendModel(_Replaced(model, soc=False), model.masses, coupling="overlap")
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="eselect_Eh"):
            BackendModel(ModelBackend(model), model.masses, coupling="overlap", eselect_Eh=bad)


def test_a_kept_gradient_missing_from_the_window_is_an_electronic_failure():
    model = _soc_model()
    spy = _Spy(model)
    adapter = BackendModel(spy, model.masses, coupling="overlap", eselect_Eh=ESELECT_EH)
    options = _soc_options(ESELECT_EH)
    q0, v0 = _soc_start(model)
    state = initial_state(adapter, q0, v0, 1, options=options)
    adapter.active, adapter.u = state.active, state.u
    spy.drop = S1  # the window around the S1-dominant state keeps S1
    with pytest.raises(ElectronicStructureError, match="no gradient for the state S1") as info:
        step(adapter, state, options, ReplaySource([0.99]))
    assert info.value.retryable is False
    assert spy.more and S1 in spy.more[-1]


def test_non_orthogonal_raw_triplet_overlaps_are_lowdin_orthonormalised():
    model = _soc_model()
    rng = np.random.default_rng(7)

    def distorted(result):
        s = np.array(result.overlaps, copy=True)
        s[:2, :2] += 0.05 * rng.standard_normal((2, 2))
        s[2:, 2:] += 0.05 * rng.standard_normal((2, 2))  # the cross blocks stay exactly 0
        distorted.raw = s
        return s

    backend = _Replaced(model, overlaps=lambda r: None if r.overlaps is None else distorted(r))
    adapter = BackendModel(backend, model.masses, coupling="overlap")
    q0, v0 = _soc_start(model)
    first = adapter.evaluate(q0, None)
    snap = adapter.evaluate(q0 + 5.0 * v0, first)
    raw = distorted.raw
    assert np.max(np.abs(raw.T @ raw - np.eye(4))) > 1e-2  # really non-orthogonal
    expected = lowdin_orthonormalize(expand_overlaps(raw, 2, 2))
    np.testing.assert_allclose(snap.overlaps, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(snap.overlaps.T @ snap.overlaps, np.eye(8), rtol=0, atol=1e-12)
    assert np.all(snap.overlaps[:2, 2:] == 0.0) and np.all(snap.overlaps[2:, :2] == 0.0)
