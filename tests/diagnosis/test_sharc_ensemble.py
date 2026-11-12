import os
from pathlib import Path

import numpy as np
import pytest
from tests.diagnosis.conftest import load_tool

from gqteashmd.config.schema import DynamicsStageSection, ElectronicSection

ens = load_tool("sharc_compare", "run_sharc_pyscf_ensemble")
lvc = load_tool("sharc_compare", "run_sharc_lvc")
dyn = load_tool("sharc_compare", "run_sharc_pyscf_dynamics")


def test_geom_and_veloc_read_back():
    symbols = ["C", "C", "H", "H", "H", "H"]
    masses = np.array([12.0, 12.0, 1.00782503223, 1.00782503223, 1.00782503223, 1.00782503223])
    rng = np.random.default_rng(5)
    q = rng.standard_normal((6, 3))
    v = rng.standard_normal((6, 3)) * 1e-4
    s2, m2, q2 = ens.parse_geom(lvc.format_geom(symbols, masses, q))
    assert s2 == symbols
    np.testing.assert_allclose(m2, masses, rtol=1e-10)
    np.testing.assert_allclose(q2, q, atol=1e-10)
    np.testing.assert_allclose(ens.parse_veloc(lvc.format_veloc(v)), v, atol=1e-14)


def _sections(**dyn_keys):
    electronic = ElectronicSection.model_validate(
        {"method": "casscf", "basis": "6-31g*", "singlets": 3, "casscf": {"ncas": 3, "nelecas": 2}}
    )
    return electronic, DynamicsStageSection.model_validate({"tmax_fs": 100.0, **dyn_keys})


def test_sharc_settings_follow_the_variant():
    s = ens.sharc_settings(*_sections())
    assert s["n_states"] == 3 and s["dt_fs"] == 0.5
    assert s["hopping"] == "sharc" and s["decoherence"] == "edc" and s["coupling"] == "nacdr"
    assert s["phases_from_interface"] is False
    assert "ncas 3" in s["template"] and "roots 3" in s["template"]
    assert "basis 6-31g*" in s["template"]
    off = ens.sharc_settings(*_sections(hopping="none", decoherence="none"))
    assert off["hopping"] == "off" and off["decoherence"] == "none"


def test_sharc_settings_refuse_tda():
    electronic = ElectronicSection.model_validate(
        {"method": "tda", "basis": "sto-3g", "functional": "hf", "singlets": 2}
    )
    with pytest.raises(ValueError, match="casscf"):
        ens.sharc_settings(electronic, DynamicsStageSection.model_validate({"tmax_fs": 1.0}))


def test_trajectory_summary_finished_and_jump():
    eh = 1 / 27.211386245988
    out = {
        "returncode": 0,
        "steps": np.arange(5),
        "e_total": np.array([0.0, 0.0, 0.3, 0.3, 0.3]) * eh,
        "energies": np.array([[0.0, 9.0]] * 2 + [[0.0, 9.3]] * 3) * eh,
        "active": np.array([1, 1, 1, 0, 0]),
        "dt_au": 0.5 / 0.024188843265857,
    }
    s = ens.trajectory_summary("traj_00000", out, dt_fs=0.5, tmax_fs=2.0)
    assert s["ended"] == "finished" and s["n_hops"] == 1
    assert s["max_jump_eV"] == pytest.approx(0.3, abs=1e-9)
    assert [j["record"] for j in s["jumps"]] == [2]
    failed = ens.trajectory_summary(
        "traj_00001", {"returncode": 3, "data": None, "stderr_tail": "x"}, dt_fs=0.5, tmax_fs=2.0
    )
    assert failed["ended"] == "stopped" and failed["t_final_fs"] is None
    assert failed["stderr_tail"] == "x"
    partial = ens.trajectory_summary(
        "traj_00002", {"returncode": 1, "data": out, "stderr_tail": "y"}, dt_fs=0.5, tmax_fs=20.0
    )
    assert partial["ended"] == "stopped" and partial["n_steps"] == 4


@pytest.mark.sharc
@pytest.mark.pyscf
@pytest.mark.skipif(
    not (os.environ.get("SHARC_HOME") and os.environ.get("SHARC_PYTHON")),
    reason="needs SHARC_HOME and SHARC_PYTHON",
)
def test_one_short_sharc_trajectory(tmp_path):
    """Two steps of sharc.x + SHARC_PYSCF.py on ethylene SA2-CAS(2,2)/6-31G*."""
    initial = dyn.ethylene_initial_conditions()
    electronic = ElectronicSection.model_validate(
        {"method": "casscf", "basis": "6-31g*", "singlets": 2, "casscf": {"ncas": 2, "nelecas": 2}}
    )
    settings = ens.sharc_settings(electronic, DynamicsStageSection.model_validate({"tmax_fs": 1.0}))
    out = dyn.run_sharc_trajectory(
        initial.q_bohr,
        initial.v_au,
        n_steps=2,
        workdir=tmp_path / "traj",
        symbols=initial.symbols,
        masses_amu=initial.masses_amu,
        sharc_home=Path(os.environ["SHARC_HOME"]),
        sharc_python=os.environ["SHARC_PYTHON"],
        provider="sharc_pyscf",
        check=False,
        **settings,
    )
    assert out["returncode"] == 0
    assert list(out["steps"]) == [0, 1, 2]
    summary = ens.trajectory_summary("t", out, dt_fs=0.5, tmax_fs=1.0)
    assert summary["ended"] == "finished"
