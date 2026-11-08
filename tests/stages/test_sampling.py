import dataclasses
import json

import h5py
import numpy as np
import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import StageFailedError
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.stages import sampling as sampling_module
from gqteashmd.stages.base import StageContext
from gqteashmd.stages.sampling import SamplingStage
from gqteashmd.vib.artifacts import write_frequencies_h5
from gqteashmd.vib.harmonic import harmonic_analysis

SYMBOLS = ("O", "H", "H")
MASSES = np.array([15.9949146, 1.00782503, 1.00782503])
COORDS = np.array([[0.0, 0.0, 0.2837], [0.0, 1.4309, -0.9196], [0.0, -1.4309, -0.9196]])


def _water_hessian():
    """Valence force field: springs on the two O-H distances and the H-H distance."""
    hessian = np.zeros((9, 9))
    for (a, b), k in (((0, 1), 0.5), ((0, 2), 0.5), ((1, 2), 0.1)):
        u = COORDS[a] - COORDS[b]
        u = u / np.linalg.norm(u)
        g = np.zeros(9)
        g[3 * a : 3 * a + 3] = u
        g[3 * b : 3 * b + 3] = -u
        hessian += k * np.outer(g, g)
    return hessian


def _write_frequencies(directory):
    directory.mkdir(parents=True)
    hessian = _water_hessian()
    analysis = harmonic_analysis(MASSES, COORDS, hessian)
    write_frequencies_h5(
        directory / "frequencies.h5",
        symbols=SYMBOLS,
        masses_amu=MASSES,
        coords_bohr=COORDS,
        hessian=hessian,
        analysis=analysis,
    )
    return analysis


def _context(tmp_path, write_input, extra="", seed=7, symbols_dir=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = write_input(tmp_path, extra=extra)
    path.write_text(path.read_text().replace('name = "test"', f'name = "test"\nseed = {seed}'))
    cfg = load_config(path).config
    layout = ProjectLayout(cfg.project.directory)
    freq_dir = layout.stage_dir(2, "frequencies")
    _write_frequencies(freq_dir)
    stage_dir = layout.stage_dir(3, "sampling")
    stage_dir.mkdir(parents=True)
    return StageContext(cfg, layout, stage_dir, {"frequencies": freq_dir})


def _run(ctx):
    stage = SamplingStage()
    results = {}
    for task in stage.plan(ctx):
        task_dir = ctx.layout.task_dir(ctx.stage_dir, task.task_id)
        task_dir.mkdir(parents=True)
        results[task.task_id] = stage.run_task(ctx, task, task_dir)
    return stage, results


def test_stage_attributes(tmp_path, write_input):
    stage = SamplingStage()
    assert stage.name == "sampling" and stage.depends_on == ("frequencies",)
    assert stage.config_sections == ("sampling",) and stage.required_modules == ("h5py>=3.9",)
    ctx = _context(tmp_path, write_input, seed=11)
    fp = stage.fingerprint_inputs(ctx)
    assert fp["project_seed"] == 11 and fp["sampling"]["n_samples"] == 200


def test_plan_run_finalize_writes_artifacts(tmp_path, write_input):
    ctx = _context(tmp_path, write_input, "[sampling]\nn_samples = 400\ntemperature_K = 300\n")
    stage, results = _run(ctx)
    assert list(results) == ["samples"]
    assert len(json.dumps(results)) < 5000  # small result, no per-sample arrays
    stage.finalize(ctx, results)
    for name in ("samples.h5", "samples.xyz", "samples.json"):
        assert (ctx.stage_dir / name).exists()
    with h5py.File(ctx.stage_dir / "samples.h5", "r") as f:
        assert f["positions"].shape == (400, 3, 3)
        assert f["velocities"].shape == (400, 3, 3)
    summary = json.loads((ctx.stage_dir / "samples.json").read_text())
    for key in ("temperature_K", "low_frequency_cutoff_cm1", "project_seed", "n_samples"):
        assert key in summary
    assert summary["project_seed"] == 7 and summary["n_samples"] == 400
    assert abs(summary["mean_energy_hartree"] - summary["expected_mean_energy_hartree"]) < (
        5 * summary["expected_energy_std_hartree"] / np.sqrt(400)
    )
    assert summary["max_linear_momentum_au"] < 1e-8
    assert summary["max_angular_momentum_au"] < 1e-6
    assert (ctx.stage_dir / "samples.xyz").read_text().count("sample ") == 400


def _positions(ctx):
    with h5py.File(ctx.stage_dir / "samples.h5", "r") as f:
        return f["positions"][()]


def test_seed_reproducibility(tmp_path, write_input):
    ctx_a = _context(tmp_path / "a", write_input, seed=5)
    ctx_b = _context(tmp_path / "b", write_input, seed=5)
    ctx_c = _context(tmp_path / "c", write_input, seed=6)
    for ctx in (ctx_a, ctx_b, ctx_c):
        stage, results = _run(ctx)
        stage.finalize(ctx, results)
    np.testing.assert_array_equal(_positions(ctx_a), _positions(ctx_b))
    assert not np.array_equal(_positions(ctx_a), _positions(ctx_c))


def test_all_modes_below_the_cutoff_stops_with_advice(tmp_path, write_input):
    ctx = _context(tmp_path, write_input, "[sampling]\nlow_frequency_cutoff_cm1 = 1e6\n")
    stage, results = _run(ctx)
    with pytest.raises(StageFailedError, match="lower"):
        stage.finalize(ctx, results)


def test_single_atom_samples_the_equilibrium(tmp_path, write_input):
    ctx = _context(tmp_path, write_input, "[sampling]\nn_samples = 5\n")
    freq_dir = ctx.upstream["frequencies"]
    (freq_dir / "frequencies.h5").unlink()
    analysis = harmonic_analysis(MASSES[:1], COORDS[:1], np.zeros((3, 3)))
    write_frequencies_h5(
        freq_dir / "frequencies.h5",
        symbols=("O",),
        masses_amu=MASSES[:1],
        coords_bohr=COORDS[:1],
        hessian=np.zeros((3, 3)),
        analysis=analysis,
    )
    stage, results = _run(ctx)
    stage.finalize(ctx, results)
    with h5py.File(ctx.stage_dir / "samples.h5", "r") as f:
        np.testing.assert_allclose(f["positions"][()], np.broadcast_to(COORDS[:1], (5, 1, 3)))
        assert not np.any(f["velocities"][()])


def test_inconsistent_energy_is_reported_as_a_program_error(tmp_path, write_input, monkeypatch):
    """Inject a +1 Eh shift into the harmonic energies returned by `sample_wigner` (patched in
    the stage module), so run_task's summary carries the wrong mean and finalize must catch it."""
    real = sampling_module.sample_wigner

    def shifted(*args, **kwargs):
        samples = real(*args, **kwargs)
        return dataclasses.replace(
            samples, harmonic_energy_hartree=samples.harmonic_energy_hartree + 1.0
        )

    monkeypatch.setattr(sampling_module, "sample_wigner", shifted)
    ctx = _context(tmp_path, write_input)
    stage, results = _run(ctx)
    with pytest.raises(StageFailedError, match="program error"):
        stage.finalize(ctx, results)
