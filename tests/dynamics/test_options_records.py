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
