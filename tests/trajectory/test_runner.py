"""The trajectory runner on the fake backend (fast tier; spec sections 3.2-6)."""

import json
from pathlib import Path

import numpy as np
import pytest

from gqteashmd.core.units import AU_TIME_IN_FS, HARTREE_IN_EV
from gqteashmd.dynamics.driver import initial_state, step
from gqteashmd.dynamics.lvc import LVCModel, lvc_wigner_samples
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.randomness import GeneratorSource, ReplaySource
from gqteashmd.dynamics.restart import load_state
from gqteashmd.errors import ElectronicStructureError
from gqteashmd.testing.fake_backend import ModelBackend, SimulatedCrash
from gqteashmd.trajectory import runner as runner_module
from gqteashmd.trajectory.model import BackendModel
from gqteashmd.trajectory.runner import (
    InitialConditions,
    RestartError,
    RunSettings,
    TrajectoryRunner,
)
from gqteashmd.trajectory.store import RestartStore, StepLog

LVC = Path(__file__).resolve().parents[1] / "data" / "lvc" / "lvc_singlets.template"
DT = 10.0
N = 30


def _model():
    return LVCModel.from_template(LVC, spin_orbit=False)


def _initial(model, active=2):
    q, v = lvc_wigner_samples(model, 1, np.random.default_rng(11))
    return InitialConditions(q=q[0], v=v[0], active=active)


def _settings(n=N, **kwargs):
    options = DynamicsOptions(dt_au=DT, coupling="overlap", **kwargs.pop("options", {}))
    return RunSettings(options, tmax_au=n * DT, **kwargs)


def _run(tmp_path, *, name="a", settings=None, faults=None, calls=0, source=None, active=2):
    model = _model()
    adapter = BackendModel(
        ModelBackend(model, faults=faults, calls=calls), model.masses, coupling="overlap"
    )
    runner = TrajectoryRunner(adapter, settings or _settings(), tmp_path / name)
    if source is None:
        source = GeneratorSource(np.random.default_rng(5))
    return runner.run(_initial(model, active), source), tmp_path / name


def _log(directory):
    return StepLog(directory / "steps.log", 3, 9).read()


def test_a_run_finishes_and_writes_its_files(tmp_path):
    h5py = pytest.importorskip("h5py")
    result, d = _run(tmp_path)
    assert result["outcome"] == "finished" and result["n_grid_steps"] == N
    assert result["t_final_fs"] == pytest.approx(N * DT * AU_TIME_IN_FS)
    rows = _log(d)
    assert len(rows) == N + 1 and list(rows["grid_step"]) == list(range(N + 1))
    with h5py.File(d / "output.h5", "r") as f:
        assert f["step"].shape == (N + 1,)
    assert json.loads((d / "progress.json").read_text())["outcome"] == "finished"


def test_the_runner_equals_the_manual_core_loop(tmp_path):
    model = _model()
    adapter = BackendModel(ModelBackend(model), model.masses, coupling="overlap")
    init = _initial(model)
    options = _settings().options
    adapter.active = init.active
    state = initial_state(adapter, init.q, init.v, init.active, options=options)
    source = GeneratorSource(np.random.default_rng(5))
    expected = []
    for _ in range(N):
        adapter.active = state.active
        state, record = step(adapter, state, options, source)
        if record.hop_to is not None:
            state = adapter.refresh_active_gradient(state)
        expected.append(record)
    _, d = _run(tmp_path)
    rows = _log(d)[1:]
    for row, record in zip(rows, expected, strict=True):
        np.testing.assert_array_equal(row["q"], record.q)
        np.testing.assert_array_equal(row["c_real"] + 1j * row["c_imag"], record.c)
        assert (row["active"], row["e_total"], row["random_number"]) == (
            record.active,
            record.e_total,
            record.random_number,
        )


@pytest.mark.parametrize("crash_call", [2, 11, 25])
def test_a_crash_and_resume_is_bit_identical(tmp_path, crash_call):
    _, reference = _run(tmp_path, name="ref")
    with pytest.raises(SimulatedCrash):
        _run(tmp_path, name="b", faults={crash_call: "crash"})
    result, d = _run(tmp_path, name="b", calls=crash_call - 1)
    assert result["outcome"] == "finished"
    assert _log(d).tobytes() == _log(reference).tobytes()


def test_a_kill_between_the_log_and_the_restart_is_bit_identical(tmp_path, monkeypatch):
    _, reference = _run(tmp_path, name="ref")
    original = RestartStore.write
    count = {"n": 0}

    def flaky(self, fill):
        count["n"] += 1
        if count["n"] == 10:
            raise SimulatedCrash("killed after the log record, before the restart")
        return original(self, fill)

    monkeypatch.setattr(RestartStore, "write", flaky)
    with pytest.raises(SimulatedCrash):
        _run(tmp_path, name="b")
    monkeypatch.setattr(RestartStore, "write", original)
    calls = 10  # compute calls made before the kill: step 0 and steps 1..9
    _run(tmp_path, name="b", calls=calls)
    assert _log(tmp_path / "b").tobytes() == _log(reference).tobytes()


def test_extension_continues_a_finished_trajectory(tmp_path):
    _, reference = _run(tmp_path, name="ref", settings=_settings(2 * N))
    first, d = _run(tmp_path, name="b", settings=_settings(N))
    assert first["outcome"] == "finished"
    again, _ = _run(tmp_path, name="b", settings=_settings(N))
    assert again["n_steps"] == first["n_steps"]  # nothing recomputed
    longer, _ = _run(tmp_path, name="b", settings=_settings(2 * N))
    assert longer["outcome"] == "finished" and longer["n_grid_steps"] == 2 * N
    assert _log(d).tobytes() == _log(reference).tobytes()


def test_energy_drift_outcome(tmp_path):
    result, d = _run(tmp_path, settings=_settings(options={"energy_drift_limit_au": 1e-14}))
    assert result["outcome"] == "energy_drift" and result["n_grid_steps"] == 1
    again, _ = _run(tmp_path, settings=_settings(options={"energy_drift_limit_au": 1e-14}))
    # a final outcome stays; the second call resumed from the final generation, and says so
    assert again["resumes"] == ["gen_00000001"]
    same = {"wall_time_s": again["wall_time_s"], "resumes": again["resumes"]}
    assert again == {**result, **same}


def test_stopped_in_s0_outcome(tmp_path):
    settings = _settings(stop_after_s0_au=3 * DT, options={"hopping": "none"})  # stays in S0
    result, _ = _run(tmp_path, settings=settings, active=0)
    assert result["outcome"] == "stopped_in_s0" and result["n_grid_steps"] == 3


def test_a_failure_at_step_0(tmp_path):
    h5py = pytest.importorskip("h5py")
    result, d = _run(tmp_path, faults={1: "error"})
    assert result["outcome"] == "electronic_failure" and result["t_final_fs"] == 0.0
    assert "step 0" in result["message"]
    assert RestartStore(d / "restart").current() is None
    with h5py.File(d / "output.h5", "r") as f:
        assert f["step"].shape == (0,)


def test_a_retryable_error_propagates_and_resumes(tmp_path):
    _, reference = _run(tmp_path, name="ref")
    with pytest.raises(ElectronicStructureError) as info:
        _run(tmp_path, name="b", faults={4: "retryable"})
    assert info.value.retryable
    _run(tmp_path, name="b", calls=3)
    assert _log(tmp_path / "b").tobytes() == _log(reference).tobytes()


def test_one_low_overlap_halves_one_interval(tmp_path):
    result, d = _run(tmp_path, faults={6: "low_overlap"})
    assert result["outcome"] == "finished" and result["n_rejections"] == 1
    rows = _log(d)
    assert len(rows) == N + 2  # one interval in two halves
    half, whole = rows[5], rows[6]
    assert (half["grid_step"], half["ticks"], half["depth"], half["rejections"]) == (4, 4, 1, 1)
    assert (whole["grid_step"], whole["ticks"], whole["depth"]) == (5, 0, 1)
    assert half["time_au"] == 4.5 * DT and whole["time_au"] == 5 * DT
    assert rows[7]["depth"] == 0


def test_a_rejected_step_returns_its_random_number(tmp_path):
    values = list(np.linspace(0.01, 0.99, 200))
    result, d = _run(tmp_path, faults={6: "low_overlap"}, source=ReplaySource(values))
    rows = _log(d)
    np.testing.assert_array_equal(rows["random_number"][1:], values[: result["n_steps"]])


@pytest.mark.parametrize(
    ("kind", "outcome"), [("low_overlap", "overlap_failure"), ("error", "electronic_failure")]
)
def test_a_persistent_signal_ends_the_trajectory(tmp_path, kind, outcome):
    result, d = _run(tmp_path, faults={c: kind for c in (6, 7, 8, 9)})
    assert result["outcome"] == outcome and result["n_rejections"] == 4
    assert result["n_grid_steps"] == 4 and "fs" in result["message"]
    state, _ = load_state(
        RestartStore(d / "restart").current() / "state.npz",
        BackendModel(ModelBackend(_model()), _model().masses, coupling="overlap"),
    )
    assert state.step == 4  # the restart holds the last accepted state


def test_a_persistent_rung_change_is_accepted_at_the_smallest_step(tmp_path):
    result, d = _run(tmp_path, faults={c: "rung" for c in (6, 7, 8, 9)})
    assert result["outcome"] == "finished"
    assert (result["n_rejections"], result["n_flagged"]) == (3, 1)
    rows = _log(d)
    flagged = rows[rows["rung_changed"] != 0]
    assert len(flagged) == 1 and flagged[0]["depth"] == 3 and flagged[0]["ticks"] == 1


def test_a_resume_inside_a_halved_interval_is_bit_identical(tmp_path):
    _, reference = _run(tmp_path, name="ref", faults={6: "low_overlap"})
    with pytest.raises(SimulatedCrash):
        _run(tmp_path, name="b", faults={6: "low_overlap", 8: "crash"})
    _run(tmp_path, name="b", faults={6: "low_overlap"}, calls=7)
    assert _log(tmp_path / "b").tobytes() == _log(reference).tobytes()


def test_hops_leave_a_finite_active_gradient_in_the_restart(tmp_path):
    result, d = _run(tmp_path, source=ReplaySource([0.0] * 200))
    assert result["n_hops"] > 0
    rows = _log(d)
    assert np.all(np.isfinite(rows["e_total"])) and np.all(np.isfinite(rows["q"]))
    model = BackendModel(ModelBackend(_model()), _model().masses, coupling="overlap")
    state, _ = load_state(RestartStore(d / "restart").current() / "state.npz", model)
    assert np.all(np.isfinite(state.snapshot.gradients[state.active]))


@pytest.mark.parametrize("damage", ["current", "log", "meta"])
def test_an_unusable_restart_is_a_restart_error(tmp_path, damage):
    _, d = _run(tmp_path, settings=_settings(n=5))
    if damage == "current":
        (d / "restart" / "current").write_text("garbage\n")
    elif damage == "log":
        (d / "steps.log").write_bytes(b"")  # fewer records than the restart's count
    else:
        generation = RestartStore(d / "restart").current()
        meta = json.loads((generation / "meta.json").read_text())
        (generation / "meta.json").write_text(json.dumps({**meta, "unknown_key": 1}))
    with pytest.raises(RestartError, match="cannot resume"):
        _run(tmp_path, settings=_settings(n=5))


def test_settings_validation():
    options = DynamicsOptions(dt_au=DT)
    with pytest.raises(ValueError, match="multiple"):
        RunSettings(options, tmax_au=10.5 * DT)
    with pytest.raises(ValueError, match="max_step_halvings"):
        RunSettings(options, tmax_au=10 * DT, max_step_halvings=7)
    assert RunSettings(options, tmax_au=10 * DT).ticks_per_step == 8
    assert runner_module.LOG_FILE == "steps.log"


def test_non_finite_energies_halve_the_interval(tmp_path):
    result, d = _run(tmp_path, faults={6: "nan_energy"})
    assert result["outcome"] == "finished" and result["n_rejections"] == 1
    rows = _log(d)
    assert len(rows) == N + 2
    assert np.all(np.isfinite(rows["e_total"])) and np.all(np.isfinite(rows["q"]))
    assert (rows[5]["grid_step"], rows[5]["ticks"], rows[5]["depth"]) == (4, 4, 1)


def test_persistent_non_finite_energies_end_the_trajectory(tmp_path):
    result, _ = _run(tmp_path, faults={c: "nan_energy" for c in (6, 7, 8, 9)})
    assert result["outcome"] == "electronic_failure" and result["n_rejections"] == 4
    assert "fs" in result["message"] and "non-finite" in result["message"]


def test_a_failed_gradient_refresh_after_a_hop_rejects_the_step(tmp_path):
    _, reference = _run(tmp_path, name="ref", source=ReplaySource([0.0] * 200))
    rows = _log(reference)
    k = int(rows["grid_step"][np.flatnonzero(rows["hop_to"] >= 0)[0]])
    result, d = _run(tmp_path, faults={k + 1: "more_error"}, source=ReplaySource([0.0] * 200))
    assert result["n_rejections"] >= 1
    assert result["outcome"] in {"finished", "electronic_failure", "energy_drift"}
    rows = _log(d)
    assert np.all(np.isfinite(rows["e_total"])) and np.all(np.isfinite(rows["q"]))


def test_rejections_are_counted_by_signal(tmp_path):
    result, _ = _run(tmp_path, faults={6: "low_overlap", 12: "rung", 18: "error"})
    assert result["outcome"] == "finished" and result["n_rejections"] == 3
    assert result["rejections_by_signal"] == {"overlap": 1, "rung": 1, "electronic": 1}


def test_a_persistent_signal_counts_every_rejected_attempt(tmp_path):
    result, _ = _run(tmp_path, faults={c: "low_overlap" for c in (6, 7, 8, 9)})
    assert result["rejections_by_signal"] == {"overlap": 4}


def test_a_resume_is_recorded_with_its_generation(tmp_path):
    with pytest.raises(SimulatedCrash):
        _run(tmp_path, name="b", faults={8: "crash"})
    result, d = _run(tmp_path, name="b", calls=7)
    assert result["outcome"] == "finished"
    assert len(result["resumes"]) == 1 and result["resumes"][0].startswith("gen_")
    fresh, _ = _run(tmp_path, name="c")
    assert fresh["resumes"] == [] and fresh["rejections_by_signal"] == {}


F1 = 0.1 / HARTREE_IN_EV  # the stage default, in Eh


def test_without_faults_the_energy_check_rejects_nothing(tmp_path):
    result, _ = _run(tmp_path, settings=_settings(energy_jump_limit_au=F1))
    assert result["outcome"] == "finished" and result["n_rejections"] == 0


def test_one_energy_jump_halves_one_interval(tmp_path):
    result, d = _run(
        tmp_path, settings=_settings(energy_jump_limit_au=F1), faults={6: "energy_jump"}
    )
    assert result["outcome"] == "finished" and result["rejections_by_signal"] == {"energy": 1}
    rows = _log(d)
    assert len(rows) == N + 2
    assert (rows[5]["grid_step"], rows[5]["ticks"], rows[5]["depth"]) == (4, 4, 1)


def test_a_persistent_energy_jump_ends_the_trajectory(tmp_path):
    result, _ = _run(
        tmp_path,
        settings=_settings(energy_jump_limit_au=F1),
        faults={c: "energy_jump" for c in range(6, 200)},
    )
    assert result["outcome"] == "electronic_discontinuity"
    assert result["n_rejections"] == 4 and result["n_grid_steps"] == 4
    assert "energy_jump_limit_eV" in result["message"] and "fs" in result["message"]


def test_an_energy_jump_with_a_rung_change_at_the_smallest_step_is_not_accepted(tmp_path):
    faults = {c: "energy_jump_rung" for c in range(6, 200)}
    result, _ = _run(tmp_path, settings=_settings(energy_jump_limit_au=F1), faults=faults)
    assert result["outcome"] == "electronic_discontinuity" and result["n_flagged"] == 0


def test_the_energy_check_is_off_by_default_in_run_settings(tmp_path):
    result, _ = _run(tmp_path, faults={6: "energy_jump"})
    assert result["outcome"] == "finished" and result["n_rejections"] == 0


def test_energy_jump_limit_validation():
    options = DynamicsOptions(dt_au=DT)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="energy_jump_limit_au"):
            RunSettings(options, tmax_au=10 * DT, energy_jump_limit_au=bad)
