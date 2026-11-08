import json

import h5py
import numpy as np
import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import StageFailedError
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.stages.base import StageContext, TaskSpec
from gqteashmd.stages.frequencies import FrequenciesStage, resolved_method
from gqteashmd.stages.prepare import PrepareStage

pytestmark = pytest.mark.pyscf
HF = '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 2\n'
# RHF/STO-3G water minimum (our optimize stage, GAU_VERYTIGHT; max |g| 2e-7), Angstrom
WATER_MIN = """3
water RHF/STO-3G minimum
O    0.000000000   0.000000000   0.1501683345
H    0.000000000   0.7580805047  -0.4856341673
H    0.000000000  -0.7580805047  -0.4856341673
"""


def _run_stage(tmp_path, write_input, extra, xyz=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = write_input(tmp_path, extra=HF + extra)
    if xyz is not None:
        (tmp_path / "water.xyz").write_text(xyz)
    cfg = load_config(path).config
    layout = ProjectLayout(cfg.project.directory)
    prep_dir = layout.stage_dir(0, "prepare")
    prep_dir.mkdir(parents=True)
    prepare = PrepareStage()
    pctx = StageContext(cfg, layout, prep_dir, {})
    prepare.finalize(pctx, {"molecule": prepare.run_task(pctx, TaskSpec("molecule"), prep_dir)})
    ctx = StageContext(cfg, layout, layout.stage_dir(2, "frequencies"), {"prepare": prep_dir})
    ctx.stage_dir.mkdir(parents=True)
    stage = FrequenciesStage()
    results = {}
    for task in stage.plan(ctx):
        task_dir = layout.task_dir(ctx.stage_dir, task.task_id)
        task_dir.mkdir(parents=True)
        results[task.task_id] = stage.run_task(ctx, task, task_dir)
    return cfg, ctx, stage, results


def test_analytic_and_finite_difference_agree(tmp_path, write_input):
    cfg_a, ctx_a, stage, res_a = _run_stage(tmp_path / "a", write_input, "", WATER_MIN)
    assert resolved_method(cfg_a) == "analytic" and list(res_a) == ["hessian"]
    stage.finalize(ctx_a, res_a)
    cfg_f, ctx_f, _, res_f = _run_stage(
        tmp_path / "f", write_input, '[frequencies]\nmethod = "finite_difference"\n', WATER_MIN
    )
    assert len(res_f) == 1 + 6 * 3
    stage.finalize(ctx_f, res_f)
    fa = json.loads((ctx_a.stage_dir / "frequencies.json").read_text())
    ff = json.loads((ctx_f.stage_dir / "frequencies.json").read_text())
    assert len(fa["frequencies_cm1"]) == 3 and fa["imaginary_modes"] == []
    np.testing.assert_allclose(fa["frequencies_cm1"], ff["frequencies_cm1"], atol=0.3)
    assert fa["geometry_source"] == "prepare" and fa["method"] == "analytic"
    assert fa["fd_consistency_residual_hartree_per_bohr"] is None
    residual = ff["fd_consistency_residual_hartree_per_bohr"]
    print("water FD consistency residual (Eh/bohr):", residual)
    assert residual < 1e-5
    with h5py.File(ctx_a.stage_dir / "frequencies.h5", "r") as f:
        np.testing.assert_allclose(f["frequencies"][()], fa["frequencies_cm1"])
    assert "[FR-NORM-COORD]" in (ctx_a.stage_dir / "frequencies.molden").read_text()


def test_imaginary_mode_beyond_the_cutoff_stops_the_stage(tmp_path, write_input):
    linear_water = "3\nlinear water (a saddle)\nO 0 0 0\nH 0 0 0.96\nH 0 0 -0.96\n"
    _, ctx, stage, results = _run_stage(tmp_path, write_input, "", linear_water)
    with pytest.raises(StageFailedError, match=r"mode \d+: -\d"):
        stage.finalize(ctx, results)
    assert (ctx.stage_dir / "frequencies.json").exists()  # written before stopping


def test_casscf_resolves_to_finite_differences(tmp_path, write_input):
    cas = (
        '[electronic]\nmethod = "casscf"\nbasis = "sto-3g"\nsinglets = 2\n'
        "[electronic.casscf]\nncas = 2\nnelecas = 2\n"
    )
    cfg = load_config(write_input(tmp_path, extra=cas)).config
    assert resolved_method(cfg) == "finite_difference"


def test_inconsistent_finite_difference_energy_stops_the_stage(tmp_path, write_input):
    _, ctx, stage, results = _run_stage(
        tmp_path, write_input, '[frequencies]\nmethod = "finite_difference"\n', WATER_MIN
    )
    broken = {name: dict(value) for name, value in results.items()}
    broken["fd_004_plus"]["energy_hartree"] += 1e-3  # another electronic solution
    with pytest.raises(StageFailedError, match=r"coordinate 4 \(atom H2, y\)"):
        stage.finalize(ctx, broken)
    data = json.loads((ctx.stage_dir / "frequencies.json").read_text())
    assert data["fd_consistency_residual_hartree_per_bohr"] > 1e-4
    assert not (ctx.stage_dir / "frequencies.h5").exists()


def _optimize_dir(ctx, energy, coords_bohr):
    directory = ctx.layout.stage_dir(1, "optimize")
    directory.mkdir(parents=True)
    (directory / "optimize.json").write_text(
        json.dumps({"energy_hartree": energy, "coords_bohr": np.asarray(coords_bohr).tolist()})
    )
    return directory


def test_energy_at_the_optimized_geometry_is_checked(tmp_path, write_input):
    from gqteashmd.stages.geometry import prepared_molecule

    _, ctx, stage, results = _run_stage(tmp_path, write_input, "", WATER_MIN)
    _, coords = prepared_molecule(ctx.upstream_dir("prepare"))
    energy = results["hessian"]["energy_hartree"]
    good = StageContext(
        ctx.config,
        ctx.layout,
        ctx.stage_dir,
        {**ctx.upstream, "optimize": _optimize_dir(ctx, energy + 2e-8, coords)},
    )
    stage.finalize(good, results)
    data = json.loads((ctx.stage_dir / "frequencies.json").read_text())
    assert data["geometry_source"] == "optimize"
    assert data["optimized_energy_difference_hartree"] == pytest.approx(2e-8, abs=1e-9)
    bad_dir = ctx.layout.stage_dir(1, "optimize")
    (bad_dir / "optimize.json").write_text(
        json.dumps({"energy_hartree": energy + 1e-3, "coords_bohr": coords.tolist()})
    )
    with pytest.raises(StageFailedError, match="not the optimized one"):
        stage.finalize(good, results)


def test_single_atom_has_no_modes(tmp_path, write_input):
    _, ctx, stage, results = _run_stage(tmp_path, write_input, "", "1\nhelium\nHe 0.0 0.0 0.0\n")
    stage.finalize(ctx, results)
    data = json.loads((ctx.stage_dir / "frequencies.json").read_text())
    assert data["frequencies_cm1"] == [] and data["n_external"] == 3
