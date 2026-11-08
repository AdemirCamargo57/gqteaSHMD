import json
import logging

import numpy as np
import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import StageFailedError
from gqteashmd.excite.artifacts import write_excitations_h5
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.stages.base import StageContext
from gqteashmd.stages.selection import SelectionStage

N = 50
FAILED = 3  # 0-based index of the failed sample


def _write_excitations(directory):
    directory.mkdir(parents=True)
    rng = np.random.default_rng(1)
    energies = np.zeros((N, 3))
    energies[:, 1] = 0.25 + 0.01 * rng.random(N)  # S1 ~ 6.8-7.1 eV
    energies[:, 2] = 0.30 + 0.01 * rng.random(N)  # S2 ~ 8.2-8.4 eV
    energies -= 75.0
    energies[:, 0] = -75.0 - 0.0
    f = np.stack([0.01 * rng.random(N), 0.1 + 0.4 * rng.random(N)], axis=1)
    ok = np.ones(N, dtype=bool)
    ok[FAILED] = False
    energies[FAILED] = np.nan
    f[FAILED] = np.nan
    write_excitations_h5(
        directory / "excitations.h5",
        symbols=("O", "H", "H"),
        state_labels=("S0", "S1", "S2"),
        sample_indices=np.arange(1, N + 1),
        ok=ok,
        energies_hartree=energies,
        transition_dipoles_au=np.zeros((N, 2, 3)),
        oscillator_strengths=f,
    )


def _context(tmp_path, write_input, selection="", seed=7):
    tmp_path.mkdir(parents=True, exist_ok=True)
    text = (
        '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 3\n'
        + selection
    )
    path = write_input(tmp_path, extra=text)
    path.write_text(path.read_text().replace('name = "test"', f'name = "test"\nseed = {seed}'))
    cfg = load_config(path).config
    layout = ProjectLayout(cfg.project.directory)
    exc_dir = layout.stage_dir(4, "excitations")
    _write_excitations(exc_dir)
    stage_dir = layout.stage_dir(5, "selection")
    stage_dir.mkdir(parents=True)
    return StageContext(cfg, layout, stage_dir, {"excitations": exc_dir})


def _run(ctx):
    stage = SelectionStage()
    results = {}
    for task in stage.plan(ctx):
        task_dir = ctx.layout.task_dir(ctx.stage_dir, task.task_id)
        task_dir.mkdir(parents=True)
        results[task.task_id] = stage.run_task(ctx, task, task_dir)
    return stage, results


def _full(ctx):
    stage, results = _run(ctx)
    stage.finalize(ctx, results)
    return json.loads((ctx.stage_dir / "trajectories.json").read_text())


def test_attributes_and_fingerprint_includes_seed(tmp_path, write_input):
    stage = SelectionStage()
    assert stage.name == "selection" and stage.depends_on == ("excitations",)
    assert stage.config_sections == ("selection",) and stage.required_modules == ("h5py>=3.9",)
    a = _context(tmp_path / "a", write_input, seed=1)
    b = _context(tmp_path / "b", write_input, seed=2)
    assert stage.fingerprint_inputs(a)["project_seed"] == 1
    assert stage.fingerprint_inputs(a) != stage.fingerprint_inputs(b)


def test_bright_is_reproducible_and_seed_dependent(tmp_path, write_input):
    sel = "[selection]\nn_trajectories = 10\n"
    a = _full(_context(tmp_path / "a", write_input, sel, seed=7))
    a2 = _full(_context(tmp_path / "a2", write_input, sel, seed=7))
    b = _full(_context(tmp_path / "b", write_input, sel, seed=8))
    assert a["trajectories"] == a2["trajectories"]
    assert a["trajectories"] != b["trajectories"]
    assert a["n_trajectories"] == len(a["trajectories"]) == 10
    assert all(t["sample"] != FAILED + 1 for t in a["trajectories"])
    first = a["trajectories"][0]
    assert first["id"] == "t00001" and first["index"] == 0
    assert first["rng"] == {"stage": "dynamics", "task_index": 0}
    assert (tmp_path / "a").exists()


def test_fixed_selects_first_ok_samples(tmp_path, write_input):
    sel = '[selection]\nmode = "fixed"\nstates = ["S2"]\nn_trajectories = 5\n'
    data = _full(_context(tmp_path, write_input, sel))
    assert [t["sample"] for t in data["trajectories"]] == [1, 2, 3, 5, 6]
    assert {t["state"] for t in data["trajectories"]} == {"S2"}


def test_too_few_fails_then_allow_fewer_warns(tmp_path, write_input, caplog):
    sel = '[selection]\nmode = "fixed"\nstates = ["S2"]\nn_trajectories = 500\n'
    ctx = _context(tmp_path / "a", write_input, sel)
    stage, results = _run(ctx)
    with pytest.raises(StageFailedError, match="allow_fewer"):
        stage.finalize(ctx, results)
    ctx = _context(tmp_path / "b", write_input, sel + "allow_fewer = true\n")
    stage, results = _run(ctx)
    with caplog.at_level(logging.WARNING):
        stage.finalize(ctx, results)
    assert any("allow_fewer" in r.getMessage() or "fewer" in r.getMessage() for r in caplog.records)
    assert len(json.loads((ctx.stage_dir / "trajectories.json").read_text())["trajectories"]) == 49


def test_zero_selected_always_fails(tmp_path, write_input):
    sel = "[selection]\nenergy_window_eV = [20.0, 30.0]\nallow_fewer = true\n"
    ctx = _context(tmp_path, write_input, sel)
    stage, results = _run(ctx)
    with pytest.raises(StageFailedError):
        stage.finalize(ctx, results)


def test_window_excludes_pairs(tmp_path, write_input):
    sel = "[selection]\nenergy_window_eV = [8.0, 9.0]\nn_trajectories = 20\n"
    data = _full(_context(tmp_path, write_input, sel))
    assert {t["state"] for t in data["trajectories"]} == {"S2"}
    assert all(8.0 <= t["excitation_energy_eV"] <= 9.0 for t in data["trajectories"])


def test_report_files(tmp_path, write_input):
    ctx = _context(tmp_path, write_input, "[selection]\nn_trajectories = 10\n")
    _full(ctx)
    report = json.loads((ctx.stage_dir / "selection_report.json").read_text())
    assert report["n_selected"] == 10 and report["n_requested"] == 10
    assert set(report["per_state"]) == {"S1", "S2"}
    md = (ctx.stage_dir / "selection_report.md").read_text()
    assert "| state | candidates | accepted | selected | mean" in md
    assert "../04_excitations/spectrum.csv" in md
    assert "t00001" in md
    assert md.rstrip().endswith("Review, then run `gqteashmd run` again to continue.")


def test_weighting_reaches_the_rule_and_the_reports(tmp_path, write_input):
    out = {}
    for w in ("f_over_dE2", "f"):
        sel = f'[selection]\nn_trajectories = 5\nweighting = "{w}"\nallow_fewer = true\n'
        ctx = _context(tmp_path / w, write_input, sel)
        stage, results = _run(ctx)
        stage.finalize(ctx, results)
        summary = results["select"]
        assert summary["weighting"] == w
        report = json.loads((ctx.stage_dir / "selection_report.json").read_text())
        assert report["weighting"] == w and report["weight_max"] == summary["weight_max"]
        assert f"- weighting: {w}" in (ctx.stage_dir / "selection_report.md").read_text()
        out[w] = summary
    assert out["f"]["weight_max"] == pytest.approx(out["f"]["f_max"])
    assert out["f_over_dE2"]["weight_max"] < out["f"]["weight_max"]
    assert out["f"]["accepted"] != out["f_over_dE2"]["accepted"]
