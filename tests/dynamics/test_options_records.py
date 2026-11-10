import json

import numpy as np
import pytest

from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.randomness import GeneratorSource, ReplaySource, source_from_state
from gqteashmd.dynamics.records import ElectronicSnapshot, StepRecord, TrajectoryState


def test_defaults():
    o = DynamicsOptions(dt_au=1.0)
    assert (o.nsubsteps, o.hopping, o.ekin_correction) == (25, "sharc", "parallel_vel")
    assert (o.frustrated, o.decoherence, o.decoherence_param_Eh) == ("keep", "edc", 0.1)
    assert o.energy_drift_limit_au == pytest.approx(0.5 / 27.211386245988)
    assert (o.edc_convention, o.substep_interpolation) == ("sharc", "midpoint")
    o = DynamicsOptions(
        dt_au=1.0, edc_convention="granucci_persico", substep_interpolation="endpoint"
    )
    assert (o.edc_convention, o.substep_interpolation) == ("granucci_persico", "endpoint")


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("dt_au", 0.0),
        ("dt_au", -1.0),
        ("dt_au", float("nan")),
        ("nsubsteps", 0),
        ("nsubsteps", 2.5),
        ("hopping", "tully"),
        ("ekin_correction", "x"),
        ("frustrated", "x"),
        ("decoherence", "x"),
        ("decoherence_param_Eh", -0.1),
        ("energy_drift_limit_au", 0.0),
        ("edc_convention", "gp"),
        ("substep_interpolation", "left"),
    ],
)
def test_validation_names_field(field, bad):
    kwargs = {"dt_au": 1.0, field: bad}
    with pytest.raises(ValueError, match=field):
        DynamicsOptions(**kwargs)


def test_generator_roundtrip():
    src = GeneratorSource(np.random.default_rng(5))
    src.uniform()
    st = json.loads(json.dumps(src.state()))
    other = source_from_state(st)
    assert [src.uniform() for _ in range(5)] == [other.uniform() for _ in range(5)]


def test_replay():
    r = ReplaySource([0.1, 0.2])
    assert (r.uniform(), r.uniform()) == (0.1, 0.2)
    with pytest.raises(ValueError, match="exhausted at draw 2"):
        r.uniform()
    r2 = source_from_state({"kind": "replay", "position": 1, "values": [0.1, 0.2]})
    assert r2.uniform() == 0.2
    assert r2.state()["position"] == 2


def test_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        source_from_state({"kind": "nope"})


@pytest.mark.parametrize("name", ["PCG64", "PCG64DXSM", "MT19937", "Philox", "SFC64"])
def test_every_whitelisted_bit_generator_round_trips(name):
    src = GeneratorSource(np.random.Generator(getattr(np.random, name)(11)))
    src.uniform()
    other = source_from_state(json.loads(json.dumps(src.state())))
    assert [src.uniform() for _ in range(3)] == [other.uniform() for _ in range(3)]


@pytest.mark.parametrize("name", ["seed", "default_rng", "RandomState", "nope", 3])
def test_a_restart_cannot_name_an_arbitrary_numpy_attribute(name):
    """Restart files are untrusted input: only the bit generators are accepted, and nothing is
    called before the check (``np.random.seed`` would reseed the legacy global RNG)."""
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    state = {"kind": "generator", "bit_generator": {"bit_generator": name, "state": {}}}
    with pytest.raises(ValueError, match="bit generator"):
        source_from_state(state)
    assert np.random.random() == expected  # the global RNG was not touched


@pytest.mark.parametrize(
    "state",
    [
        {"kind": "generator"},
        {"kind": "generator", "bit_generator": {}},
        {"kind": "generator", "bit_generator": {"bit_generator": "PCG64", "state": "x"}},
        {"kind": "replay", "values": [0.1]},
        {"kind": "replay", "position": 0},
    ],
)
def test_incomplete_or_malformed_source_states_are_value_errors(state):
    with pytest.raises(ValueError, match="random source state"):
        source_from_state(state)


@pytest.mark.parametrize("position", [-1, 3, 1.5, True])
def test_replay_position_must_be_inside_the_sequence(position):
    with pytest.raises(ValueError, match="position"):
        ReplaySource([0.1, 0.2], position)
    with pytest.raises(ValueError, match="position"):
        source_from_state({"kind": "replay", "position": position, "values": [0.1, 0.2]})
    ReplaySource([0.1, 0.2], 2)  # at the end: valid, the next draw reports exhaustion


def test_records():
    snap = ElectronicSnapshot(
        np.zeros(2), np.zeros((2, 1)), np.zeros((2, 2, 1)), np.zeros((2, 2), complex)
    )
    assert snap.vectors is None
    st = TrajectoryState(0.0, 0, np.zeros(1), np.zeros(1), np.ones(2, complex), 0, snap, 0.0)
    assert st.active == 0
    rec = StepRecord(
        1, 1.0, np.zeros(1), np.zeros(1), np.ones(2, complex), 0, np.zeros(2), 0.0, 0.0,
        np.zeros(2), 0.5, None, None, False,
    )  # fmt: skip
    assert rec.hop_to is None
