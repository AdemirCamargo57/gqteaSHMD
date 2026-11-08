import dataclasses
import json

import h5py
import numpy as np
import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import StageFailedError, TaskError
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.stages import sampling as sampling_module
from gqteashmd.stages.base import StageContext
from gqteashmd.stages.sampling import SamplingStage
from gqteashmd.vib.artifacts import write_frequencies_h5
from gqteashmd.vib.harmonic import harmonic_analysis
from gqteashmd.vib.wigner import sample_wigner

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


def _context(tmp_path, write_input, extra="", seed=7):
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
    assert summary["mode_energy_min_p_value"] >= summary["mode_energy_p_value_threshold"]
    assert len(summary["mode_energy_ratio_sums"]) == 3
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
    """Scale the coordinates and momenta of the sampled modes by 2 in `sample_wigner` (patched in
    the stage module), so the per-mode energy sums are wrong and finalize must catch it."""
    real = sampling_module.sample_wigner

    def too_wide(*args, **kwargs):
        samples = real(*args, **kwargs)
        return dataclasses.replace(
            samples,
            q=2.0 * samples.q,
            p=2.0 * samples.p,
            harmonic_energy_hartree=4.0 * samples.harmonic_energy_hartree,
        )

    monkeypatch.setattr(sampling_module, "sample_wigner", too_wide)
    ctx = _context(tmp_path, write_input)
    stage, results = _run(ctx)
    with pytest.raises(StageFailedError, match="program error"):
        stage.finalize(ctx, results)
    summary = json.loads((ctx.stage_dir / "samples.json").read_text())
    assert summary["mode_energy_min_p_value"] < summary["mode_energy_p_value_threshold"]


def _diatomic_samples(seed, n, scale=1.0, temperature_K=0.0):
    masses = np.array([12.0, 15.9949146])
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.13]])
    h = np.zeros((6, 6))
    h[2, 2] = h[5, 5] = 0.5
    h[2, 5] = h[5, 2] = -0.5
    analysis = harmonic_analysis(masses, coords, h)
    s = sample_wigner(
        masses,
        coords,
        analysis.frequencies_cm1,
        analysis.modes_mass_weighted,
        n_samples=n,
        temperature_K=temperature_K,
        low_frequency_cutoff_cm1=10.0,
        rng=np.random.default_rng(seed),
    )
    return dataclasses.replace(s, q=scale * s.q, p=scale * s.p)


def test_correct_sampler_never_trips_the_energy_check_at_one_sample():
    threshold = sampling_module.ENERGY_FALSE_ALARM_PROBABILITY
    for seed in range(200):
        s = _diatomic_samples(seed, 1, temperature_K=300.0 if seed % 2 else 0.0)
        p = sampling_module.mode_energy_p_values(s.mode_energy_ratio_sums(), 1)
        assert p.min() >= threshold, f"seed {seed}"


def test_a_ten_percent_width_error_in_one_mode_is_detected():
    s = _diatomic_samples(3, 2000, scale=1.1)
    p = sampling_module.mode_energy_p_values(s.mode_energy_ratio_sums(), 2000)
    assert p.min() < sampling_module.ENERGY_FALSE_ALARM_PROBABILITY
    ok = _diatomic_samples(3, 2000)
    p_ok = sampling_module.mode_energy_p_values(ok.mode_energy_ratio_sums(), 2000)
    assert p_ok.min() >= sampling_module.ENERGY_FALSE_ALARM_PROBABILITY


def test_non_finite_energy_sums_fail_the_check():
    p = sampling_module.mode_energy_p_values([float("nan")], 10)
    assert p[0] == 0.0


def test_missing_frequencies_file_is_a_non_retryable_task_error(tmp_path, write_input):
    ctx = _context(tmp_path, write_input)
    (ctx.upstream["frequencies"] / "frequencies.h5").unlink()
    stage = SamplingStage()
    task_dir = ctx.layout.task_dir(ctx.stage_dir, "samples")
    task_dir.mkdir(parents=True)
    with pytest.raises(TaskError, match="--rerun-from frequencies") as info:
        stage.run_task(ctx, stage.plan(ctx)[0], task_dir)
    assert info.value.retryable is False


def test_small_imaginary_mode_is_listed_and_warned_about(tmp_path, write_input, caplog):
    ctx = _context(tmp_path, write_input)
    with h5py.File(ctx.upstream["frequencies"] / "frequencies.h5", "r+") as f:
        f["frequencies"][0] = -7.5
    stage, results = _run(ctx)
    with caplog.at_level("WARNING", logger="gqteashmd.stages.sampling"):
        stage.finalize(ctx, results)
    summary = json.loads((ctx.stage_dir / "samples.json").read_text())
    assert summary["unsampled_frequencies_cm1"] == [-7.5]
    assert summary["n_sampled_modes"] == 2
    assert "-7.5" in caplog.text and "not sampled" in caplog.text


def test_residual_momentum_is_reported_as_a_program_error(tmp_path, write_input):
    ctx = _context(tmp_path, write_input)
    stage, results = _run(ctx)
    for result in results.values():
        result["max_linear_momentum_au"] = 1.0
    with pytest.raises(StageFailedError, match="Total momentum was not removed"):
        stage.finalize(ctx, results)
