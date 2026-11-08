import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.config.schema import EffectiveResources
from gqteashmd.errors import ConfigError


def test_hardware_keys_are_required(tmp_path, write_input):
    path = write_input(tmp_path)
    path.write_text(path.read_text().replace("total_cores = 4\n", ""))
    with pytest.raises(ConfigError, match=r"resources\.total_cores: Field required"):
        load_config(path)


def test_missing_resources_section_is_an_error(tmp_path):
    (tmp_path / "w.xyz").write_text("1\n\nH 0 0 0\n")
    path = tmp_path / "input.toml"
    path.write_text('[project]\nname = "x"\ndirectory = "run"\n[molecule]\ngeometry = "w.xyz"\n')
    with pytest.raises(ConfigError, match="resources: Field required"):
        load_config(path)


def test_cores_must_fit(tmp_path, write_input):
    path = write_input(tmp_path, resources="max_parallel_tasks = 3\nthreads_per_task = 2")
    with pytest.raises(ConfigError, match="= 6 exceeds total_cores = 4"):
        load_config(path)


def test_memory_must_fit(tmp_path, write_input):
    path = write_input(tmp_path, resources="max_parallel_tasks = 3\nmemory_mb_per_task = 2000")
    with pytest.raises(ConfigError, match="= 6000 MB exceeds total_memory_mb = 4000"):
        load_config(path)


def test_stage_override_must_fit_and_is_named(tmp_path, write_input):
    path = write_input(tmp_path, resources="[resources.stages.frequencies]\nthreads_per_task = 8")
    with pytest.raises(ConfigError, match=r"\[resources\.stages\.frequencies\]"):
        load_config(path)


def test_for_stage_merges_overrides(tmp_path, write_input):
    path = write_input(
        tmp_path,
        resources="max_parallel_tasks = 2\n[resources.stages.frequencies]\n"
        "max_parallel_tasks = 1\nthreads_per_task = 4",
    )
    res = load_config(path).config.resources
    assert res.for_stage("frequencies") == EffectiveResources(1, 4, 2000)
    assert res.for_stage("sampling") == EffectiveResources(2, 1, 2000)


def test_unknown_override_key_is_rejected(tmp_path, write_input):
    path = write_input(tmp_path, resources="[resources.stages.sampling]\nthreds = 2")
    with pytest.raises(ConfigError, match="threds"):
        load_config(path)


def test_empty_override_name_is_rejected(tmp_path, write_input):
    path = write_input(tmp_path, resources='[resources.stages.""]\nthreads_per_task = 1')
    with pytest.raises(ConfigError, match="not a valid stage name"):
        load_config(path)


def test_global_oversubscription_is_caught_with_an_override_present(tmp_path, write_input):
    path = write_input(
        tmp_path,
        resources="max_parallel_tasks = 3\nthreads_per_task = 2\n"
        "[resources.stages.prepare]\nthreads_per_task = 1",
    )
    with pytest.raises(ConfigError, match=r"\[resources\]: .*= 3 × 2 = 6 exceeds total_cores = 4"):
        load_config(path)


def test_memory_message_shows_the_factors(tmp_path, write_input):
    path = write_input(tmp_path, resources="max_parallel_tasks = 3\nmemory_mb_per_task = 2000")
    with pytest.raises(ConfigError, match="= 3 × 2000 = 6000 MB exceeds total_memory_mb = 4000"):
        load_config(path)


def test_every_resource_violation_is_reported(tmp_path, write_input):
    path = write_input(
        tmp_path,
        total_cores=2,
        total_memory_mb=1000,
        resources="max_parallel_tasks = 4\nmemory_mb_per_task = 500",
    )
    with pytest.raises(ConfigError) as info:
        load_config(path)
    text = str(info.value)
    assert "exceeds total_cores" in text and "exceeds total_memory_mb" in text
