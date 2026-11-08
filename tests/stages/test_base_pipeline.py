import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import ProjectError
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.stages.base import Stage, StageContext, TaskSpec, is_safe_name
from gqteashmd.stages.pipeline import import_pipeline, validate_pipeline


class _Dummy(Stage):
    name = "dummy"
    config_sections = ("molecule", "resources")

    def plan(self, ctx):
        return [TaskSpec("only")]

    def run_task(self, ctx, task, task_dir):
        return {}

    def finalize(self, ctx, results):
        return None


def _stage(name: str, deps: tuple[str, ...] = ()) -> Stage:
    stage = _Dummy()
    stage.name = name
    stage.depends_on = deps
    return stage


def _ctx(tmp_path, write_input) -> StageContext:
    cfg = load_config(write_input(tmp_path)).config
    layout = ProjectLayout(cfg.project.directory)
    return StageContext(cfg, layout, layout.stage_dir(0, "dummy"), {})


def test_task_id_must_be_filesystem_safe():
    TaskSpec("ok_1.x-2")
    with pytest.raises(ValueError, match="a/b"):
        TaskSpec("a/b")
    with pytest.raises(ValueError):
        TaskSpec("")


def test_default_fingerprint_uses_declared_sections(tmp_path, write_input):
    fp = _Dummy().fingerprint_inputs(_ctx(tmp_path, write_input))
    assert set(fp) == {"molecule", "resources"}
    assert fp["molecule"]["charge"] == 0


def test_upstream_dir_lookup(tmp_path, write_input):
    ctx = _ctx(tmp_path, write_input)
    ctx_with = StageContext(ctx.config, ctx.layout, ctx.stage_dir, {"prepare": tmp_path / "p"})
    assert ctx_with.upstream_dir("prepare") == tmp_path / "p"
    with pytest.raises(ProjectError, match="optimize"):
        ctx_with.upstream_dir("optimize")


def test_import_pipeline_resolves_module_attribute():
    factory = import_pipeline("os.path:join")
    assert callable(factory)


@pytest.mark.parametrize(
    "ref, message",
    [
        ("nocolon", "expected 'module:function'"),
        ("no_such_module_xyz:f", "Cannot import"),
        ("os:no_such_attr", "has no attribute"),
    ],
)
def test_import_pipeline_reports_bad_references(ref, message):
    with pytest.raises(ProjectError, match=message):
        import_pipeline(ref)


def test_validate_pipeline_accepts_ordered_dependencies():
    validate_pipeline([_stage("a"), _stage("b", ("a",)), _stage("c", ("a", "b"))])


def test_validate_pipeline_rejects_duplicates():
    with pytest.raises(ProjectError, match="Duplicate stage name 'a'"):
        validate_pipeline([_stage("a"), _stage("a")])


def test_validate_pipeline_rejects_forward_dependencies():
    with pytest.raises(ProjectError, match="'a' depends on 'b'"):
        validate_pipeline([_stage("a", ("b",)), _stage("b")])


def test_validate_pipeline_rejects_unsafe_names():
    with pytest.raises(ProjectError, match="Invalid stage name"):
        validate_pipeline([_stage("bad name")])


@pytest.mark.parametrize("name", [".", ".."])
def test_dot_names_are_not_safe(name):
    assert is_safe_name(name) is False
    with pytest.raises(ValueError):
        TaskSpec(name)
    with pytest.raises(ProjectError, match="Invalid stage name"):
        validate_pipeline([_stage(name)])


def test_ordinary_names_are_safe():
    assert is_safe_name("t001") and is_safe_name("03_sampling.v2")


def test_default_pipeline_order(tmp_path, write_input):
    from gqteashmd.stages.registry import default_pipeline

    cfg = load_config(write_input(tmp_path)).config
    assert [s.name for s in default_pipeline(cfg)] == [
        "prepare",
        "optimize",
        "frequencies",
        "sampling",
        "excitations",
        "selection",
    ]


def test_missing_electronic_section_is_reported(tmp_path, write_input):
    from gqteashmd.errors import ConfigError
    from gqteashmd.stages.pipeline import check_stage_inputs
    from gqteashmd.stages.registry import default_pipeline

    cfg = load_config(write_input(tmp_path)).config  # no [electronic]
    with pytest.raises(ConfigError) as info:
        check_stage_inputs(cfg, default_pipeline(cfg))
    text = str(info.value)
    assert "'optimize'" in text and "'frequencies'" in text and "[electronic]" in text


def test_prepare_only_pipeline(tmp_path, write_input):
    from gqteashmd.testing.pipelines import prepare_only

    cfg = load_config(write_input(tmp_path)).config
    assert [s.name for s in prepare_only(cfg)] == ["prepare"]


def test_check_config_hook_problems_are_reported_for_enabled_stages(tmp_path, write_input):
    from gqteashmd.errors import ConfigError
    from gqteashmd.stages.pipeline import check_stage_inputs

    class Picky(_Dummy):
        def check_config(self, config):
            return ["problem one", "problem two"]

    class Off(Picky):
        def enabled(self, config):
            return False

    cfg = load_config(write_input(tmp_path)).config
    assert _Dummy().check_config(cfg) == []
    check_stage_inputs(cfg, [_Dummy(), Off()])
    with pytest.raises(ConfigError, match="problem one; problem two"):
        check_stage_inputs(cfg, [Picky()])
