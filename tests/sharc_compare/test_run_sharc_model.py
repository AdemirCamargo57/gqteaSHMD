"""The option mapping of the SHARC model harness (no SHARC needed)."""

import importlib.util
import sys
from pathlib import Path

import pytest

from gqteashmd.dynamics.options import DynamicsOptions

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "sharc_compare"


def _load(name):
    if name in sys.modules:  # shared with tests/validation/test_sharc_tully.py
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve their module while the class is built
    spec.loader.exec_module(module)
    return module


rsm = _load("run_sharc_model")


def test_options_are_mapped():
    options = DynamicsOptions(
        dt_au=0.5, nsubsteps=10, frustrated="reflect", ekin_correction="parallel_nac"
    )
    mapped = rsm.SharcOptions.from_dynamics_options(options)
    assert mapped == rsm.SharcOptions(
        hopping="sharc",
        ekincorrect="parallel_nac",
        reflect_frustrated="parallel_nac",
        decoherence="edc",
        decoherence_param=0.1,
        nsubsteps=10,
    )
    off = rsm.SharcOptions.from_dynamics_options(
        DynamicsOptions(dt_au=0.5, hopping="none", decoherence="none")
    )
    assert (off.hopping, off.reflect_frustrated, off.decoherence) == ("off", "none", "none")


def test_options_sharc_cannot_reproduce_are_rejected():
    with pytest.raises(ValueError, match="fssh"):
        rsm.SharcOptions.from_dynamics_options(DynamicsOptions(dt_au=0.5, hopping="fssh"))
    with pytest.raises(ValueError, match="granucci_persico"):
        rsm.SharcOptions.from_dynamics_options(
            DynamicsOptions(dt_au=0.5, edc_convention="granucci_persico")
        )
    # Without EDC the convention is irrelevant, and the substep interpolation is allowed to
    # differ from SHARC's (always end points) so that the difference can be measured.
    rsm.SharcOptions.from_dynamics_options(
        DynamicsOptions(dt_au=0.5, decoherence="none", edc_convention="granucci_persico")
    )
    rsm.SharcOptions.from_dynamics_options(
        DynamicsOptions(dt_au=0.5, substep_interpolation="midpoint")
    )
