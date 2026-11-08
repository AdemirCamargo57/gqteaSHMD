from dataclasses import fields, replace

import numpy as np
import pytest

from gqteashmd.dynamics.driver import Outcome, initial_state, run_trajectory, step
from gqteashmd.dynamics.models import TullySimpleAvoidedCrossing
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.randomness import GeneratorSource, ReplaySource
from gqteashmd.dynamics.records import StepRecord

MASS = 2000.0


def _start(model, x0=-10.0, k=20.0, active=0):
    return initial_state(model, np.array([x0]), np.array([k / MASS]), active)


def _run(model, state, options, rng, max_steps, stop=None):
    records: list[StepRecord] = []
    final, outcome = run_trajectory(
        model, state, options, rng, max_steps=max_steps, stop=stop, on_step=records.append
    )
    return final, outcome, records


def _exit(state):
    return abs(state.q[0]) > 10.0 and state.q[0] * state.v[0] > 0


def _assert_records_equal(a: list[StepRecord], b: list[StepRecord]) -> None:
    assert len(a) == len(b)
    for ra, rb in zip(a, b, strict=True):
        for fld in fields(StepRecord):
            xa, xb = getattr(ra, fld.name), getattr(rb, fld.name)
            if isinstance(xa, np.ndarray):
                assert np.array_equal(xa, xb), fld.name
            else:
                assert xa == xb, fld.name


def test_initial_state_defaults():
    model = TullySimpleAvoidedCrossing()
    state = _start(model, active=1)
    assert state.step == 0 and state.time_au == 0.0
    assert state.active == 1
    assert np.array_equal(state.c, np.array([0.0, 1.0], dtype=complex))
    assert state.c.dtype == complex
    snap = model.evaluate(np.array([-10.0]), None)
    assert np.array_equal(state.snapshot.energies, snap.energies)
    kinetic = 0.5 * MASS * (20.0 / MASS) ** 2
    assert state.e_total_initial == pytest.approx(snap.energies[1] + kinetic, abs=1e-15)


def test_initial_state_rejects_bad_input():
    model = TullySimpleAvoidedCrossing()
    q, v = np.array([-10.0]), np.array([0.01])
    with pytest.raises(ValueError, match="active"):
        initial_state(model, q, v, 2)
    with pytest.raises(ValueError, match="c0"):
        initial_state(model, q, v, 0, np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="q0"):
        initial_state(model, np.array([0.0, 1.0]), v, 0)


def test_energy_conservation_without_hopping():
    """Tully SAC, hopping off, dt = 1: E_tot conserved to < 1e-6 Eh over the whole passage."""
    model = TullySimpleAvoidedCrossing()
    options = DynamicsOptions(dt_au=1.0, hopping="none")
    state = _start(model)
    final, outcome, records = _run(
        model, state, options, GeneratorSource(np.random.default_rng(0)), max_steps=5000, stop=_exit
    )
    assert outcome is Outcome.STOPPED
    assert final.q[0] > 10.0  # transmitted on the ground state
    assert all(r.active == 0 for r in records)
    drift = max(abs(r.e_total - state.e_total_initial) for r in records)
    assert drift < 1e-6


def test_step_follows_the_documented_order():
    """One step reproduced by hand from the components (method reference 8, no hop)."""
    from gqteashmd.dynamics.decoherence import apply_decoherence
    from gqteashmd.dynamics.hopping import hopping_probabilities
    from gqteashmd.dynamics.integrators import acceleration, verlet_positions, verlet_velocities
    from gqteashmd.dynamics.propagators import propagate

    model = TullySimpleAvoidedCrossing()
    options = DynamicsOptions(dt_au=2.0, hopping="sharc")
    state = _start(model, x0=-1.0)
    state.c = np.array([0.8, 0.6j], dtype=complex)
    snap = state.snapshot
    m = model.masses
    a = acceleration(snap.gradients[0], m)
    q1 = verlet_positions(state.q, state.v, a, 2.0)
    snap1 = model.evaluate(q1, snap)
    v1 = verlet_velocities(state.v, a, acceleration(snap1.gradients[0], m), 2.0)
    c1, p, amps = propagate(
        state.c, snap.energies, snap1.energies, snap.nacs, snap1.nacs, state.v, v1, 2.0, 25
    )
    probs = hopping_probabilities(
        "sharc", 0, state.c, c1, p, amps, snap.nacs, snap1.nacs, state.v, v1, 2.0
    )
    kin = 0.5 * float(np.sum(m * v1 * v1))
    c2 = apply_decoherence(c1, 0, snap1.energies, kin, 2.0, scheme="edc", param=0.1)

    new, rec = step(model, state, options, ReplaySource([0.999999]))
    assert np.array_equal(new.q, q1)
    assert np.array_equal(new.v, v1)
    assert np.array_equal(new.c, c2)
    assert np.array_equal(rec.probabilities, probs)
    assert rec.random_number == 0.999999
    assert rec.hop_from is None and rec.hop_to is None and not rec.frustrated
    assert new.step == 1 and new.time_au == 2.0 and rec.step == 1 and rec.time_au == 2.0
    assert rec.e_kinetic == kin
    assert rec.e_total == snap1.energies[0] + kin
    assert np.array_equal(rec.energies, snap1.energies)
    assert new.snapshot is not snap
    assert new.e_total_initial == state.e_total_initial
    # the input state is not modified
    assert state.step == 0 and np.array_equal(state.q, np.array([-1.0]))


def test_one_random_number_per_step_even_without_hopping():
    model = TullySimpleAvoidedCrossing()
    options = DynamicsOptions(dt_au=1.0, hopping="none")
    rng = ReplaySource([0.1, 0.2, 0.3])
    _, outcome, records = _run(model, _start(model), options, rng, max_steps=3)
    assert outcome is Outcome.FINISHED
    assert [r.random_number for r in records] == [0.1, 0.2, 0.3]
    assert rng.state()["position"] == 3


@pytest.mark.parametrize("scheme", ["sharc", "fssh"])
@pytest.mark.parametrize("ekin", ["parallel_vel", "parallel_nac"])
def test_hops_keep_total_energy_continuous(scheme, ekin):
    """xi = 0 every step: a hop happens whenever a probability is positive."""
    model = TullySimpleAvoidedCrossing()
    options = DynamicsOptions(dt_au=1.0, hopping=scheme, ekin_correction=ekin)
    state = _start(model, x0=-4.0, k=20.0)
    _, _, records = _run(model, state, options, ReplaySource([0.0] * 2000), max_steps=800)
    hops = [i for i, r in enumerate(records) if r.hop_to is not None]
    assert len(hops) >= 2
    previous = [state.e_total_initial] + [r.e_total for r in records]
    for i in hops:
        r = records[i]
        assert r.hop_from is not None and r.hop_from != r.hop_to
        assert r.active == r.hop_to
        assert r.probabilities[r.hop_from] == 0.0
        assert not r.frustrated
        assert abs(r.e_total - previous[i]) < 1e-6
    assert max(abs(e - state.e_total_initial) for e in previous) < 1e-6


@pytest.mark.parametrize("frustrated", ["keep", "reflect"])
def test_frustrated_hop_is_recorded(frustrated):
    """Too little kinetic energy for the upper state: the attempted hops are frustrated."""
    model = TullySimpleAvoidedCrossing()
    options = DynamicsOptions(dt_au=1.0, hopping="sharc", frustrated=frustrated)
    state = _start(model, x0=-1.5, k=4.0)  # K = 0.004 Eh < gap (about 0.018 Eh)
    _, _, records = _run(model, state, options, ReplaySource([0.0] * 3000), max_steps=200)
    frustrated_steps = [r for r in records if r.frustrated]
    assert frustrated_steps
    first = frustrated_steps[0]
    assert first.hop_from is None and first.hop_to is None
    assert first.active == 0
    assert all(r.active == 0 for r in records)
    if frustrated == "reflect":
        assert first.v[0] < 0.0  # parallel_vel reflection: v' = -v
    else:
        assert first.v[0] > 0.0


def test_same_seed_gives_identical_records_and_replay_reproduces():
    model = TullySimpleAvoidedCrossing()
    options = DynamicsOptions(dt_au=1.0)
    state = _start(model, x0=-4.0, k=20.0)
    _, out_a, rec_a = _run(model, state, options, GeneratorSource(np.random.default_rng(11)), 800)
    _, out_b, rec_b = _run(model, state, options, GeneratorSource(np.random.default_rng(11)), 800)
    assert out_a is out_b
    _assert_records_equal(rec_a, rec_b)
    logged = [r.random_number for r in rec_a]
    _, out_c, rec_c = _run(model, state, options, ReplaySource(logged), 800)
    assert out_c is out_a
    _assert_records_equal(rec_a, rec_c)


def test_energy_drift_outcome():
    model = TullySimpleAvoidedCrossing()
    options = DynamicsOptions(dt_au=1.0, hopping="none", energy_drift_limit_au=1e-15)
    final, outcome, records = _run(
        model, _start(model, x0=-1.0), options, GeneratorSource(np.random.default_rng(0)), 500
    )
    assert outcome is Outcome.ENERGY_DRIFT
    assert final.step == len(records) < 500
    assert abs(records[-1].e_total - final.e_total_initial) > 1e-15


def test_stop_callback_and_finished():
    model = TullySimpleAvoidedCrossing()
    options = DynamicsOptions(dt_au=1.0)
    rng = GeneratorSource(np.random.default_rng(3))
    final, outcome, records = _run(
        model, _start(model), options, rng, 1000, stop=lambda s: s.q[0] > -9.95
    )
    assert outcome is Outcome.STOPPED
    assert final.q[0] > -9.95 and records[-2].q[0] <= -9.95
    assert final.step == len(records)

    final2, outcome2, records2 = _run(model, _start(model), options, rng, 7)
    assert outcome2 is Outcome.FINISHED
    assert final2.step == 7 and len(records2) == 7
    assert final2.time_au == 7.0

    # a state that already satisfies stop takes no step
    final3, outcome3, records3 = _run(model, final, options, rng, 10, stop=lambda s: True)
    assert outcome3 is Outcome.STOPPED and records3 == [] and final3 is final


class _SocModel(TullySimpleAvoidedCrossing):
    """Returns a non-diagonal MCH Hamiltonian once a previous snapshot exists."""

    def __init__(self, always: bool = False) -> None:
        super().__init__()
        self.always = always

    def evaluate(self, q, previous):
        snap = super().evaluate(q, previous)
        if previous is None and not self.always:
            return snap
        h = snap.hamiltonian.copy()
        h[0, 1] = h[1, 0] = 0.01
        return replace(snap, hamiltonian=h)


def test_soc_guard_propagates():
    options = DynamicsOptions(dt_au=1.0)
    model = _SocModel()
    state = _start(model)
    with pytest.raises(NotImplementedError, match="P7a-2"):
        step(model, state, options, ReplaySource([0.5]))
    with pytest.raises(NotImplementedError, match="P7a-2"):
        _start(_SocModel(always=True))
