import json
import math

import numpy as np
import pytest

from gqteashmd.core.states import StateLabel
from gqteashmd.electronic.base import Capabilities, ElectronicResult, Request
from gqteashmd.electronic.properties import oscillator_strength
from gqteashmd.errors import ElectronicStructureError, TaskError

S0, S1, T1 = StateLabel(1, 0), StateLabel(1, 1), StateLabel(3, 1)


def test_error_type():
    err = ElectronicStructureError("SCF did not converge")
    assert isinstance(err, TaskError)
    assert err.retryable is False and err.exit_code == 4


def test_request_coerces_and_checks_shape():
    req = Request([[0, 0, 0], [0, 0, 1.4]])
    assert req.coords_bohr.dtype == float and req.coords_bohr.shape == (2, 3)
    with pytest.raises(ValueError, match="natom, 3"):
        Request([0.0, 0.0, 1.0])


def test_oscillator_strength_formula():
    assert math.isclose(oscillator_strength(0.5, np.array([1.0, 0.0, 0.0])), 1.0 / 3.0)
    assert oscillator_strength(0.3, np.zeros(3)) == 0.0


def test_result_json():
    result = ElectronicResult(
        states=(S0, S1),
        energies={S0: -1.0, S1: -0.5},
        gradients={S1: np.ones((2, 3))},
        transition_dipoles={S1: np.array([1.0, 0.0, 0.0])},
        spin_squares={S0: 0.0, S1: 0.0},
        meta={"method": "casscf"},
    )
    assert result.excitation_energy(S1) == 0.5
    data = result.to_json()
    assert data["schema_version"] == 1
    s0, s1 = data["states"]
    assert s0["label"] == "S0" and s0["oscillator_strength"] is None
    assert s1["excitation_ev"] == pytest.approx(0.5 * 27.211386245988)
    assert s1["oscillator_strength"] == pytest.approx(1.0 / 3.0)
    assert data["gradients_hartree_per_bohr"]["S1"] == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert data["meta"] == {"method": "casscf"}


def test_result_json_is_serialisable_and_orders_gradients():
    result = ElectronicResult(
        states=(S0, S1, T1),
        energies={S0: -1.0, S1: -0.5, T1: np.float64(-0.6)},
        gradients={T1: np.ones((1, 3)), S1: np.zeros((1, 3))},
        transition_dipoles={S1: np.array([1.0, 0.0, 0.0], dtype=np.float32)},
        spin_squares={S0: np.float32(0.0), S1: np.float64(0.0), T1: np.float32(2.0)},
    )
    data = result.to_json()
    json.dumps(data)  # numpy scalars would raise here
    assert list(data["gradients_hartree_per_bohr"]) == ["S1", "T1"]
    for row in data["states"]:
        for key in ("excitation_ev", "oscillator_strength", "spin_square"):
            assert row[key] is None or type(row[key]) is float


def test_request_carries_nac_pairs_and_overlaps():
    req = Request([[0, 0, 0]], nacs=frozenset({(S0, S1)}), overlaps=True)
    assert req.nacs == frozenset({(S0, S1)}) and req.overlaps is True
    with pytest.raises(ValueError, match="I < J"):
        Request([[0, 0, 0]], nacs=frozenset({(S1, S0)}))
    with pytest.raises(ValueError, match="I < J"):
        Request([[0, 0, 0]], nacs=frozenset({(S1, S1)}))


def test_capabilities_are_frozen():
    caps = Capabilities(gradients=True, nacs=False, overlaps=True, transition_dipoles=True)
    with pytest.raises(AttributeError):
        caps.nacs = True  # type: ignore[misc]


def test_result_json_with_couplings_and_overlaps():
    result = ElectronicResult(
        states=(S0, S1),
        energies={S0: -1.0, S1: -0.5},
        nacs={(S0, S1): np.full((2, 3), 0.25)},
        overlaps=np.array([[0.99, 0.01], [-0.01, 0.98]]),
        phases=np.array([1.0, -1.0]),
    )
    data = result.to_json()
    assert data["nacs_per_bohr"] == {"S0-S1": [[0.25] * 3, [0.25] * 3]}
    assert data["overlaps"] == [[0.99, 0.01], [-0.01, 0.98]]
    assert data["phases"] == [1.0, -1.0]
    json.dumps(data)
    empty = ElectronicResult(states=(S0,), energies={S0: -1.0}).to_json()
    assert empty["nacs_per_bohr"] == {} and empty["overlaps"] is None and empty["phases"] is None
