import numpy as np
import pytest

from gqteashmd.dynamics.driver import initial_state, run_trajectory
from gqteashmd.dynamics.models import TullySimpleAvoidedCrossing, TwoDimensionalCrossing
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.randomness import GeneratorSource, ReplaySource
from gqteashmd.dynamics.restart import load_state, save_state

MASS = 2000.0


def _sac_state(model):
    return initial_state(model, np.array([-1.5]), np.array([30.0 / MASS]), 0)


def _assert_same(a, b):
    assert np.array_equal(a.q, b.q)
    assert np.array_equal(a.v, b.v)
    assert np.array_equal(a.c, b.c)
    assert a.active == b.active
    assert a.step == b.step and a.time_au == b.time_au
    assert a.e_total_initial == b.e_total_initial


@pytest.mark.parametrize("scheme", ["sharc", "fssh"])
def test_restart_is_bit_identical(tmp_path, scheme):
    model = TullySimpleAvoidedCrossing()
    options = DynamicsOptions(dt_au=2.0, hopping=scheme, decoherence="edc")
    start = _sac_state(model)

    full, _ = run_trajectory(
        model, start, options, GeneratorSource(np.random.default_rng(5)), max_steps=100
    )

    rng = GeneratorSource(np.random.default_rng(5))
    half, _ = run_trajectory(model, start, options, rng, max_steps=50)
    path = tmp_path / "restart.npz"
    save_state(path, half, rng)
    loaded, rng2 = load_state(path, model)
    _assert_same(loaded, half)
    resumed, _ = run_trajectory(model, loaded, options, rng2, max_steps=50)

    _assert_same(resumed, full)
    assert np.array_equal(resumed.snapshot.energies, full.snapshot.energies)
    assert np.array_equal(resumed.snapshot.vectors, full.snapshot.vectors)
    assert resumed.step == 100


def test_restart_round_trips_the_snapshot_and_replay_source(tmp_path):
    model = TwoDimensionalCrossing()
    options = DynamicsOptions(dt_au=1.0)
    state = initial_state(
        model, np.array([1.0, 0.5]), np.array([0.0, 0.002]), 1, np.array([0.6, 0.8j])
    )
    rng = ReplaySource([0.3] * 20 + [0.7] * 5)
    state, _ = run_trajectory(model, state, options, rng, max_steps=20)
    path = tmp_path / "sub" / "r.npz"
    save_state(path, state, rng)
    loaded, rng2 = load_state(path, model)
    for name in ("energies", "gradients", "nacs", "hamiltonian", "vectors"):
        a, b = getattr(state.snapshot, name), getattr(loaded.snapshot, name)
        assert a.dtype == b.dtype and np.array_equal(a, b), name
    assert loaded.c.dtype == complex
    assert rng2.state() == rng.state()
    assert [rng2.uniform() for _ in range(5)] == [0.7] * 5
    assert [p.name for p in path.parent.iterdir()] == ["r.npz"]  # no temporary file left


class _CountingModel(TullySimpleAvoidedCrossing):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def evaluate(self, q, previous):
        self.calls += 1
        return super().evaluate(q, previous)


def test_load_state_evaluates_nothing(tmp_path):
    model = TullySimpleAvoidedCrossing()
    state = _sac_state(model)
    path = tmp_path / "r.npz"
    save_state(path, state, GeneratorSource(np.random.default_rng(1)))
    counting = _CountingModel()
    load_state(path, counting)
    assert counting.calls == 0


def test_snapshot_without_vectors(tmp_path):
    from dataclasses import replace

    model = TullySimpleAvoidedCrossing()
    state = _sac_state(model)
    state.snapshot = replace(state.snapshot, vectors=None)
    path = tmp_path / "r.npz"
    save_state(path, state, ReplaySource([0.5]))
    loaded, _ = load_state(path, model)
    assert loaded.snapshot.vectors is None


def test_save_overwrites_atomically(tmp_path):
    model = TullySimpleAvoidedCrossing()
    rng = ReplaySource([0.5] * 10)
    state = _sac_state(model)
    path = tmp_path / "r.npz"
    save_state(path, state, rng)
    state2, _ = run_trajectory(model, state, DynamicsOptions(dt_au=1.0), rng, max_steps=3)
    save_state(path, state2, rng)
    loaded, _ = load_state(path, model)
    assert loaded.step == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == ["r.npz"]


def test_unreadable_or_mismatched_restart(tmp_path):
    model = TullySimpleAvoidedCrossing()
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a restart")
    with pytest.raises(ValueError, match="not a readable gqteashmd dynamics restart"):
        load_state(bad, model)
    with pytest.raises(ValueError, match="not a readable gqteashmd dynamics restart"):
        load_state(tmp_path / "missing.npz", model)
    other = tmp_path / "other.npz"
    np.savez(other, x=np.zeros(3))
    with pytest.raises(ValueError, match="not a readable gqteashmd dynamics restart"):
        load_state(other, model)

    path = tmp_path / "r.npz"
    save_state(path, _sac_state(model), ReplaySource([0.5]))
    with pytest.raises(ValueError, match="does not match the model"):
        load_state(path, TwoDimensionalCrossing())
