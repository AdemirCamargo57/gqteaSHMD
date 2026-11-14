import json

import numpy as np
import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import TaskError
from gqteashmd.geomopt import OptimizationFailedError
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.stages.base import StageContext, TaskSpec
from gqteashmd.stages.optimize import OptimizeStage
from gqteashmd.stages.prepare import PrepareStage

pytestmark = pytest.mark.pyscf
ELECTRONIC = '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 2\n'


def _contexts(tmp_path, write_input, extra=""):
    cfg = load_config(write_input(tmp_path, extra=ELECTRONIC + extra)).config
    layout = ProjectLayout(cfg.project.directory)
    prepare_dir = layout.stage_dir(0, "prepare")
    prepare_dir.mkdir(parents=True)
    prepare = PrepareStage()
    pctx = StageContext(cfg, layout, prepare_dir, {})
    prepare.finalize(pctx, {"molecule": prepare.run_task(pctx, TaskSpec("molecule"), prepare_dir)})
    octx = StageContext(cfg, layout, layout.stage_dir(1, "optimize"), {"prepare": prepare_dir})
    octx.stage_dir.mkdir(parents=True)
    return cfg, octx


def test_optimize_stage_produces_a_minimum(tmp_path, write_input):
    cfg, ctx = _contexts(tmp_path, write_input)
    stage = OptimizeStage()
    assert stage.enabled(cfg) and [t.task_id for t in stage.plan(ctx)] == ["optimize"]
    task_dir = ctx.stage_dir / "tasks" / "optimize"
    task_dir.mkdir(parents=True)
    result = stage.run_task(ctx, TaskSpec("optimize"), task_dir)
    assert result["converged"] and result["max_gradient_hartree_per_bohr"] < 1e-3
    json.dumps(result)
    stage.finalize(ctx, {"optimize": result})
    data = json.loads((ctx.stage_dir / "optimize.json").read_text())
    assert data["schema_version"] == 1 and len(data["coords_bohr"]) == 3
    assert (ctx.stage_dir / "optimized.xyz").read_text().splitlines()[0].strip() == "3"
    path_lines = (ctx.stage_dir / "optimization_path.xyz").read_text().splitlines()
    assert path_lines[0].strip() == "3" and "energy" in path_lines[1].lower()


def test_non_convergence_is_a_non_retryable_task_error(tmp_path, write_input):
    _, ctx = _contexts(
        tmp_path, write_input, '[optimize]\nmax_steps = 1\nconvergence = "GAU_VERYTIGHT"\n'
    )
    task_dir = ctx.stage_dir / "tasks" / "optimize"
    task_dir.mkdir(parents=True)
    with pytest.raises(TaskError, match="did not converge in 1 steps") as info:
        OptimizeStage().run_task(ctx, TaskSpec("optimize"), task_dir)
    assert info.value.retryable is False and (task_dir / "best.xyz").exists()


def test_disabled_optimize(tmp_path, write_input):
    cfg, _ = _contexts(tmp_path, write_input, "[optimize]\nenabled = false\n")
    assert OptimizeStage().enabled(cfg) is False


def test_non_convergence_message_names_the_best_geometry(tmp_path, write_input):
    _, ctx = _contexts(
        tmp_path, write_input, '[optimize]\nmax_steps = 1\nconvergence = "GAU_VERYTIGHT"\n'
    )
    task_dir = ctx.stage_dir / "tasks" / "optimize"
    task_dir.mkdir(parents=True)
    with pytest.raises(TaskError, match=r"lowest-energy geometry found is in .*best\.xyz"):
        OptimizeStage().run_task(ctx, TaskSpec("optimize"), task_dir)


def test_geometric_failure_is_a_non_retryable_task_error(tmp_path, write_input, monkeypatch):
    from gqteashmd.stages import optimize

    def failing(*args, **kwargs):
        raise OptimizationFailedError("LinearTorsionError: linear torsion")

    monkeypatch.setattr(optimize, "optimize_geometry", failing)
    _, ctx = _contexts(tmp_path, write_input)
    task_dir = ctx.stage_dir / "tasks" / "optimize"
    task_dir.mkdir(parents=True)
    with pytest.raises(TaskError, match="geomeTRIC") as info:
        OptimizeStage().run_task(ctx, TaskSpec("optimize"), task_dir)
    assert info.value.retryable is False


def test_single_atom_needs_no_optimizer(tmp_path, write_input, monkeypatch):
    from gqteashmd.stages import optimize

    def forbidden(*args, **kwargs):
        raise AssertionError("geomeTRIC must not be called for one atom")

    monkeypatch.setattr(optimize, "optimize_geometry", forbidden)
    path = write_input(tmp_path, extra=ELECTRONIC.replace("singlets = 2", "singlets = 1"))
    (tmp_path / "water.xyz").write_text("1\nhelium\nHe 0.0 0.0 0.0\n")
    cfg = load_config(path).config
    layout = ProjectLayout(cfg.project.directory)
    prepare_dir = layout.stage_dir(0, "prepare")
    prepare_dir.mkdir(parents=True)
    prepare = PrepareStage()
    pctx = StageContext(cfg, layout, prepare_dir, {})
    prepare.finalize(pctx, {"molecule": prepare.run_task(pctx, TaskSpec("molecule"), prepare_dir)})
    ctx = StageContext(cfg, layout, layout.stage_dir(1, "optimize"), {"prepare": prepare_dir})
    ctx.stage_dir.mkdir(parents=True)
    task_dir = ctx.stage_dir / "tasks" / "optimize"
    task_dir.mkdir(parents=True)
    result = OptimizeStage().run_task(ctx, TaskSpec("optimize"), task_dir)
    assert result["converged"] is True and result["n_gradients"] == 1
    data = json.loads((task_dir / "optimize.json").read_text())
    assert data["symbols"] == ["He"] and data["coords_bohr"] == [[0.0, 0.0, 0.0]]


class _Stop(Exception):
    pass


def test_optimize_passes_the_prepared_geometry_as_reference(tmp_path, write_input, monkeypatch):
    from gqteashmd.stages import optimize as optimize_module
    from gqteashmd.stages.geometry import prepared_molecule

    cfg, octx = _contexts(tmp_path, write_input)
    seen = {}

    def spy(ctx, stage_name, molecule, task_dir, *, reference_coords_bohr, settings=None):
        seen["reference"] = np.asarray(reference_coords_bohr)
        raise _Stop

    monkeypatch.setattr(optimize_module, "stage_backend", spy)
    task_dir = octx.layout.task_dir(octx.stage_dir, "optimize")
    task_dir.mkdir(parents=True)
    with pytest.raises(_Stop):
        OptimizeStage().run_task(octx, TaskSpec("optimize"), task_dir)
    _, prepared = prepared_molecule(octx.upstream_dir("prepare"))
    np.testing.assert_array_equal(seen["reference"], prepared)
