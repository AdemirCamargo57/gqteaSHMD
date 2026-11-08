import json

import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import TaskError
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
    assert info.value.retryable is False and (task_dir / "last.xyz").exists()


def test_disabled_optimize(tmp_path, write_input):
    cfg, _ = _contexts(tmp_path, write_input, "[optimize]\nenabled = false\n")
    assert OptimizeStage().enabled(cfg) is False
