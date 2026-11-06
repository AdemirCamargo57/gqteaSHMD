import pytest

from gqteashmd.config.loader import load_config, load_snapshot, section_fingerprint, write_snapshot
from gqteashmd.errors import ConfigError


def test_minimal_input_gets_defaults_and_absolute_paths(tmp_path, write_input):
    loaded = load_config(write_input(tmp_path))
    cfg = loaded.config
    assert loaded.source == (tmp_path / "input.toml").resolve()
    assert cfg.project.directory == (tmp_path / "run").resolve()
    assert cfg.molecule.geometry == (tmp_path / "water.xyz").resolve()
    assert cfg.molecule.charge == 0
    assert cfg.molecule.multiplicity == 1
    assert cfg.project.seed == 12345
    assert cfg.resources.max_parallel_tasks == 1
    assert cfg.execution.max_attempts == 2


def test_unknown_section_is_rejected(tmp_path, write_input):
    path = write_input(tmp_path, extra="[molecule_typo]\ncharge = 0")
    with pytest.raises(ConfigError, match="molecule_typo"):
        load_config(path)


def test_unknown_key_is_rejected_with_location(tmp_path, write_input):
    path = write_input(tmp_path, molecule="chrage = 1")
    with pytest.raises(ConfigError, match=r"molecule\.chrage"):
        load_config(path)


def test_invalid_value_reports_field(tmp_path, write_input):
    path = write_input(tmp_path, resources="max_parallel_tasks = 0")
    with pytest.raises(ConfigError, match=r"resources\.max_parallel_tasks"):
        load_config(path)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_bad_toml(tmp_path):
    path = tmp_path / "input.toml"
    path.write_text("project = [")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(path)


def test_snapshot_roundtrip(tmp_path, write_input):
    loaded = load_config(write_input(tmp_path, molecule="charge = 2"))
    snap = tmp_path / "snap.json"
    write_snapshot(loaded, snap)
    again = load_snapshot(snap)
    assert again.config == loaded.config
    assert again.source == snap.resolve()


def test_snapshot_missing_is_config_error(tmp_path):
    with pytest.raises(ConfigError, match="snapshot"):
        load_snapshot(tmp_path / "missing.json")


def test_section_fingerprint_selects_sections(tmp_path, write_input):
    cfg = load_config(write_input(tmp_path)).config
    fp = section_fingerprint(cfg, ["molecule", "resources"])
    assert set(fp) == {"molecule", "resources"}
    assert fp["molecule"]["charge"] == 0
    assert isinstance(fp["molecule"]["geometry"], str)
