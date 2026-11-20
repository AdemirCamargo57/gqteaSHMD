"""Identity guard: the P7a-1 trajectories (recorded at e76c83a) must never change.

The events (active state, hops, frustrated hops) are compared exactly. The floating-point
records are compared to an absolute 1e-12: they were bit-identical on the development machine,
but OpenBLAS picks its kernels by CPU, and with other kernels (OPENBLAS_CORETYPE=SandyBridge,
Nehalem, Prescott; measured 2026-11-20) the amplitudes and hop probabilities differ by up to
3.5e-15 while positions, velocities and every event stay the same. CI runs on other CPUs.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("make_tully_golden", HERE / "make_tully_golden.py")
assert _spec is not None and _spec.loader is not None
golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden)


FLOAT_TOLERANCE = 1e-12


@pytest.mark.parametrize("name", ["sac", "dac", "ecr"])
def test_trajectory_is_identical_to_p7a1(name):
    with np.load(HERE / "data" / "tully_golden.npz") as data:
        recorded = {k.removeprefix(f"{name}_"): data[k] for k in data.files if k.startswith(name)}
    current = golden.run_case(name)
    assert set(current) == set(recorded)
    for key, value in current.items():
        expected = recorded[key]
        assert value.shape == expected.shape, f"{name}: {key} shape changed"
        if np.issubdtype(value.dtype, np.inexact):
            assert np.allclose(value, expected, rtol=0.0, atol=FLOAT_TOLERANCE), (
                f"{name}: {key} changed by {np.max(np.abs(value - expected)):.3e}"
            )
        else:
            assert np.array_equal(value, expected), f"{name}: {key} changed"


@pytest.mark.parametrize("name", ["sac", "dac", "ecr"])
def test_tully_records_have_no_diagonal_basis(name):
    """U stays None (the identity) for every Tully step; the MCH view equals the diagonal one."""
    records: list = []
    golden.run_case(name, on_step=records.append)
    assert records
    for r in records:
        assert r.u is None
        assert np.array_equal(r.energies_diag, r.energies)
        assert np.array_equal(r.c_mch, r.c) and r.c_mch is not r.c
        assert r.active_mch == r.active


def test_golden_trajectories_contain_the_events():
    with np.load(HERE / "data" / "tully_golden.npz") as data:
        assert (data["sac_hop_to"] >= 0).any()
        assert (data["dac_hop_to"] >= 0).any()
        assert (data["ecr_hop_to"] >= 0).any()
        assert data["ecr_frustrated"].any()
