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


# --- run_ensemble with a stubbed run_sharc_trajectory (no SHARC) -------------------------------


def _fake_world(monkeypatch, behaviour):
    """Stub the project and sharc.x: behaviour(task_id, kwargs) -> result dict (or raises)."""
    from types import SimpleNamespace

    starts = [
        SimpleNamespace(
            task_id=f"traj_{k:05d}",
            index=k,
            state="S1",
            q=np.arange(6.0) + 10 * k,
            v=1e-3 * (np.arange(6.0) + 1 + k),
        )
        for k in range(2)
    ]
    inputs = SimpleNamespace(
        config=None,
        project_dir=Path("proj"),
        symbols=("H", "H"),
        masses_amu=np.ones(2),
        starts=starts,
    )
    sections = _sections()
    monkeypatch.setattr(ens, "load_project_inputs", lambda p: inputs)
    monkeypatch.setattr(ens, "variant_sections", lambda c, v: sections)
    calls = []

    def stub(*args, workdir, **kwargs):
        calls.append((workdir.name, kwargs, args))
        workdir.mkdir(parents=True, exist_ok=False)
        (workdir / "output.dat").write_text("garbage")
        return behaviour(workdir.name, kwargs)

    monkeypatch.setattr(ens, "run_sharc_trajectory", stub)
    return calls


def _run(tmp_path):
    from gqteashmd.config.schema import DynamicsStageSection  # noqa: F401

    variant = ens.Variant("base")
    return ens.run_ensemble(
        Path("proj"), variant, tmp_path, sharc_home=Path("/x"), sharc_python="py", workers=1
    )


def test_a_raising_job_is_recorded(tmp_path, monkeypatch):
    def behaviour(task, kwargs):
        if task == "traj_00000":
            raise RuntimeError("sharc.x did not finish")
        return {"returncode": 0, "data": None, "stderr_tail": "", "wall_time_s": 1.0}

    _fake_world(monkeypatch, behaviour)
    summary = _run(tmp_path)
    assert summary["n_errors"] == 1
    bad = summary["trajectories"][0]
    assert bad["ended"] == "stopped" and bad["returncode"] is None
    assert "did not finish" in bad["error"]
    assert (tmp_path / "base" / "sharc_summary.json").is_file()


def test_status_json_is_reused_with_its_failure(tmp_path, monkeypatch):
    def behaviour(task, kwargs):
        return {"returncode": 3, "data": None, "stderr_tail": "boom", "wall_time_s": 2.0}

    calls = _fake_world(monkeypatch, behaviour)
    _run(tmp_path)
    assert len(calls) == 2
    second = _run(tmp_path)
    assert len(calls) == 2  # nothing rerun
    t = second["trajectories"][0]
    assert t["reused"] is True and t["returncode"] == 3 and t["stderr_tail"] == "boom"


def test_a_dir_without_status_is_moved_aside(tmp_path, monkeypatch):
    calls = _fake_world(
        monkeypatch, lambda t, k: {"returncode": 0, "data": None, "stderr_tail": ""}
    )
    stale = tmp_path / "base" / "traj_00000"
    stale.mkdir(parents=True)
    (stale / "output.dat").write_text("old")
    summary = _run(tmp_path)
    assert (tmp_path / "base" / "traj_00000.stale-0" / "output.dat").read_text() == "old"
    assert tuple(c[0] for c in calls[:1]) == ("traj_00000",)
    assert summary["trajectories"][0]["reused"] is False


def test_ended_needs_returncode_zero():
    eh = 1 / 27.211386245988
    out = {
        "returncode": 1,
        "steps": np.arange(5),
        "e_total": np.zeros(5),
        "energies": np.zeros((5, 2)) * eh,
        "active": np.ones(5, dtype=int),
    }
    s = ens.trajectory_summary("t", {**out, "data": out}, dt_fs=0.5, tmax_fs=2.0)
    assert s["ended"] == "stopped" and s["t_final_fs"] == 2.0


def test_thread_variables_reach_sharc(tmp_path, monkeypatch):
    seen = []

    def behaviour(task, kwargs):
        seen.append(kwargs["env_extra"])
        return {"returncode": 0, "data": None, "stderr_tail": ""}

    _fake_world(monkeypatch, behaviour)
    _run(tmp_path)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert seen[0][name] == "1"


def test_each_job_gets_its_own_start_state_masses_and_seed(tmp_path, monkeypatch):
    calls = _fake_world(
        monkeypatch, lambda t, k: {"returncode": 0, "data": None, "stderr_tail": ""}
    )
    _run(tmp_path)
    for k, (task, kwargs, args) in enumerate(calls):
        q = (np.arange(6.0) + 10 * k).reshape(-1, 3)
        v = (1e-3 * (np.arange(6.0) + 1 + k)).reshape(-1, 3)
        assert task == f"traj_{k:05d}"
        np.testing.assert_array_equal(args[0], ens.interface_geometry(q))
        np.testing.assert_array_equal(args[1], v)
        assert kwargs["state_mch"] == 1 and kwargs["seed"] == k + 1
        np.testing.assert_array_equal(kwargs["masses_amu"], np.ones(2))
        assert tuple(kwargs["symbols"]) == ("H", "H")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("nsubsteps", 10),
        ("decoherence_param_Eh", 0.2),
        ("ekin_correction", "parallel_nac"),
        ("frustrated", "reflect"),
        ("edc_convention", "granucci_persico"),
    ],
)
def test_an_untranslated_dynamics_value_is_refused(key, value):
    extra = {"coupling": "nacdr"} if key == "ekin_correction" else {}
    with pytest.raises(ValueError, match=key):
        ens.sharc_settings(*_sections(**{key: value}, **extra))


def test_hopping_fssh_is_refused():
    with pytest.raises(ValueError, match="fssh"):
        ens.sharc_settings(*_sections(hopping="fssh", coupling="nacdr"))


def test_the_m4_defaults_are_translated():
    assert ens.sharc_settings(*_sections())["hopping"] == "sharc"


def test_a_timed_out_run_builds_its_row_from_the_partial_output_dat(tmp_path, monkeypatch):
    """The row of the run that timed out equals the row a rerun builds from the same files."""
    data = {
        "steps": np.arange(4),
        "e_total": np.zeros(4),
        "energies": np.zeros((4, 3)),
        "active": np.ones(4, dtype=int),
    }
    monkeypatch.setattr(ens, "parse_lvc_output_dat", lambda path: "parsed")
    monkeypatch.setattr(ens, "trajectory_data", lambda parsed: dict(data))

    def behaviour(task, kwargs):
        raise TimeoutError("sharc.x timed out")

    _fake_world(monkeypatch, behaviour)
    first = _run(tmp_path)["trajectories"][0]
    assert first["error"] and "timed out" in first["error"] and first["reused"] is False
    assert first["n_steps"] == 3 and first["t_final_fs"] == pytest.approx(1.5)
    assert first["ended"] == "stopped"
    again = _run(tmp_path)["trajectories"][0]
    assert again["reused"] is True
    assert {k: v for k, v in again.items() if k != "reused"} == {
        k: v for k, v in first.items() if k != "reused"
    }
