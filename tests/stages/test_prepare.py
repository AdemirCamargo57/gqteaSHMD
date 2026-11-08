import json

import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import ConfigError, TaskError
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.stages.base import StageContext, TaskSpec
from gqteashmd.stages.prepare import PrepareStage
from gqteashmd.stages.registry import default_pipeline


def _ctx(path) -> StageContext:
    cfg = load_config(path).config
    layout = ProjectLayout(cfg.project.directory)
    return StageContext(cfg, layout, layout.stage_dir(0, "prepare"), {})


def test_plan_has_single_task(tmp_path, write_input):
    ctx = _ctx(write_input(tmp_path))
    assert PrepareStage().plan(ctx) == [TaskSpec("molecule")]


def test_run_task_counts_electrons(tmp_path, write_input):
    ctx = _ctx(write_input(tmp_path))
    result = PrepareStage().run_task(ctx, TaskSpec("molecule"), tmp_path / "task")
    assert result["n_electrons"] == 10
    assert result["atomic_numbers"] == [8, 1, 1]
    assert result["symbols"] == ["O", "H", "H"]


def test_inconsistent_multiplicity_is_non_retryable(tmp_path, write_input):
    ctx = _ctx(write_input(tmp_path, molecule="multiplicity = 2"))
    with pytest.raises(TaskError, match="inconsistent") as info:
        PrepareStage().run_task(ctx, TaskSpec("molecule"), tmp_path / "task")
    assert info.value.retryable is False


def test_bad_geometry_is_non_retryable(tmp_path, write_input):
    path = write_input(tmp_path)
    (tmp_path / "water.xyz").write_text("3\n\nO 0 0 0\n")
    with pytest.raises(TaskError, match="Invalid geometry") as info:
        PrepareStage().run_task(_ctx(path), TaskSpec("molecule"), tmp_path / "task")
    assert info.value.retryable is False


def test_finalize_writes_artifacts(tmp_path, write_input):
    ctx = _ctx(write_input(tmp_path))
    stage = PrepareStage()
    result = stage.run_task(ctx, TaskSpec("molecule"), tmp_path / "task")
    stage.finalize(ctx, {"molecule": result})
    data = json.loads((ctx.stage_dir / "molecule.json").read_text())
    assert data["schema_version"] == 1
    assert data["n_electrons"] == 10
    assert (ctx.stage_dir / "geometry.xyz").read_text().splitlines()[0] == "3"


def test_fingerprint_tracks_geometry_content_not_path(tmp_path, write_input):
    path = write_input(tmp_path)
    stage = PrepareStage()
    first = stage.fingerprint_inputs(_ctx(path))
    assert set(first) == {"charge", "multiplicity", "geometry_sha256"}
    (tmp_path / "water.xyz").write_text((tmp_path / "water.xyz").read_text() + "\n")
    assert stage.fingerprint_inputs(_ctx(path)) != first


def test_fingerprint_with_missing_geometry_is_config_error(tmp_path, write_input):
    path = write_input(tmp_path)
    (tmp_path / "water.xyz").unlink()
    with pytest.raises(ConfigError, match="Geometry file not found"):
        PrepareStage().fingerprint_inputs(_ctx(path))


def test_default_pipeline_starts_with_prepare(tmp_path, write_input):
    cfg = load_config(write_input(tmp_path)).config
    assert [s.name for s in default_pipeline(cfg)][0] == "prepare"


def test_geometry_that_is_a_directory_is_a_config_error(tmp_path, write_input):
    path = write_input(tmp_path)
    (tmp_path / "water.xyz").unlink()
    (tmp_path / "water.xyz").mkdir()
    ctx = _ctx(path)
    with pytest.raises(ConfigError, match="Cannot read geometry file"):
        PrepareStage().fingerprint_inputs(ctx)
    with pytest.raises(TaskError, match="Cannot read geometry file") as info:
        PrepareStage().run_task(ctx, TaskSpec("molecule"), tmp_path / "task")
    assert info.value.retryable is False


O2_XYZ = "2\noxygen\nO 0.0 0.0 0.0\nO 0.0 0.0 1.21\n"


@pytest.mark.parametrize(
    "molecule, ok",
    [
        ("charge = 1\nmultiplicity = 2", True),  # H2O+ : 9 electrons, doublet
        ("charge = 1\nmultiplicity = 1", False),  # 9 electrons cannot be a singlet
        ("multiplicity = 12", False),  # 11 unpaired > 10 electrons
        ("charge = 10", False),  # 0 electrons
    ],
)
def test_electron_count_rules(tmp_path, write_input, molecule, ok):
    ctx = _ctx(write_input(tmp_path, molecule=molecule))
    if ok:
        assert (
            PrepareStage().run_task(ctx, TaskSpec("molecule"), tmp_path / "t")["n_electrons"] == 9
        )
    else:
        with pytest.raises(TaskError, match="inconsistent"):
            PrepareStage().run_task(ctx, TaskSpec("molecule"), tmp_path / "t")


def test_triplet_oxygen_is_accepted(tmp_path, write_input):
    path = write_input(tmp_path, molecule="multiplicity = 3")
    (tmp_path / "water.xyz").write_text(O2_XYZ)
    result = PrepareStage().run_task(_ctx(path), TaskSpec("molecule"), tmp_path / "t")
    assert (result["n_electrons"], result["multiplicity"]) == (16, 3)


def test_fingerprint_does_not_depend_on_location(tmp_path, write_input):
    first = PrepareStage().fingerprint_inputs(_ctx(write_input(tmp_path)))
    other = tmp_path / "elsewhere"
    other.mkdir()
    second = PrepareStage().fingerprint_inputs(_ctx(write_input(other)))
    assert first == second
