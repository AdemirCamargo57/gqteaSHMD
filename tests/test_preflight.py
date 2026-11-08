import json
import os
import shutil
from pathlib import Path

import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import ConfigError
from gqteashmd.orchestrator import Orchestrator
from gqteashmd.preflight import MachineInfo, detect_machine, run_preflight
from gqteashmd.stages.prepare import PrepareStage
from gqteashmd.testing.fake_stages import BEHAVIOUR_ENV

FAKE = "gqteashmd.testing.fake_stages:make_pipeline"


def _with_directory(cfg, directory):
    return cfg.model_copy(
        update={"project": cfg.project.model_copy(update={"directory": directory})}
    )


def test_detect_machine_reports_positive_values():
    info = detect_machine()
    assert info.cores is not None and info.cores >= 1
    assert info.memory_mb is not None and info.memory_mb >= 1


def test_no_warnings_when_hardware_fits(tmp_path, write_input):
    cfg = load_config(write_input(tmp_path)).config
    assert (
        run_preflight(cfg, [PrepareStage()], machine=MachineInfo(cores=64, memory_mb=10**6)) == []
    )


def test_warns_when_user_hardware_exceeds_machine(tmp_path, write_input):
    cfg = load_config(write_input(tmp_path)).config  # total_cores 4, total_memory_mb 4000
    warnings = run_preflight(cfg, [PrepareStage()], machine=MachineInfo(cores=2, memory_mb=1000))
    assert len(warnings) == 2
    assert "total_cores = 4 exceeds the 2 cores" in warnings[0]
    assert "total_memory_mb = 4000 exceeds the 1000 MB" in warnings[1]


@pytest.mark.parametrize(
    "directory",
    ["/mnt/c/Users/x/run", "/mnt/D/run", "/home/x/OneDrive/run", "/home/x/OneDrive - UEG/run"],
)
def test_windows_or_onedrive_project_directory_is_refused(tmp_path, write_input, directory):
    cfg = _with_directory(load_config(write_input(tmp_path)).config, Path(directory))
    with pytest.raises(ConfigError, match="local Linux disk"):
        run_preflight(cfg, [PrepareStage()], machine=MachineInfo(None, None))


@pytest.mark.parametrize(
    "directory",
    ["/mnt/data/run", "/mnt/nvme0/run", "/home/x/onedrive-backup/run", "/home/x/run"],
)
def test_ordinary_linux_project_directory_is_accepted(
    tmp_path, write_input, directory, monkeypatch
):
    # The directories are fictitious: this test is about the location rule, not permissions.
    monkeypatch.setattr(os, "access", lambda *_a, **_k: True)
    cfg = _with_directory(load_config(write_input(tmp_path)).config, Path(directory))
    assert run_preflight(cfg, [PrepareStage()], machine=MachineInfo(None, None)) == []


def test_too_little_free_disk_is_refused(tmp_path, write_input):
    cfg = load_config(write_input(tmp_path, execution="min_free_disk_mb = 1000000000000")).config
    with pytest.raises(ConfigError, match="min_free_disk_mb"):
        run_preflight(cfg, [PrepareStage()], machine=MachineInfo(None, None))


def test_missing_module_is_refused_before_the_project_is_created(
    tmp_path, write_input, monkeypatch
):
    monkeypatch.setenv(
        BEHAVIOUR_ENV, json.dumps({"requires": {"sampling": ["no_such_module_xyz"]}})
    )
    orchestrator = Orchestrator(load_config(write_input(tmp_path)), pipeline_ref=FAKE)
    with pytest.raises(ConfigError, match="no_such_module_xyz"):
        orchestrator.run()
    assert not (tmp_path / "run").exists()


def test_project_directory_that_is_a_file_is_a_config_error(tmp_path, write_input):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    cfg = _with_directory(load_config(write_input(tmp_path)).config, blocker)
    with pytest.raises(ConfigError, match="not a directory"):
        run_preflight(cfg, [PrepareStage()], machine=MachineInfo(None, None))


def test_project_directory_below_a_file_is_a_config_error(tmp_path, write_input):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    cfg = _with_directory(load_config(write_input(tmp_path)).config, blocker / "run")
    with pytest.raises(ConfigError, match="Cannot create or write"):
        run_preflight(cfg, [PrepareStage()], machine=MachineInfo(None, None))


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_read_only_parent_is_a_config_error(tmp_path, write_input):
    cfg = _with_directory(load_config(write_input(tmp_path)).config, tmp_path / "ro" / "run")
    (tmp_path / "ro").mkdir()
    (tmp_path / "ro").chmod(0o555)
    try:
        with pytest.raises(ConfigError, match="no write permission"):
            run_preflight(cfg, [PrepareStage()], machine=MachineInfo(None, None))
    finally:
        (tmp_path / "ro").chmod(0o755)


def test_zero_minimum_disk_skips_the_disk_query(tmp_path, write_input, monkeypatch):
    def boom(_path):
        raise OSError("no disk query")

    monkeypatch.setattr(shutil, "disk_usage", boom)
    cfg = load_config(write_input(tmp_path, execution="min_free_disk_mb = 0")).config
    assert run_preflight(cfg, [PrepareStage()], machine=MachineInfo(None, None)) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_unsearchable_ancestor_is_a_config_error(tmp_path, write_input):
    locked = tmp_path / "locked"
    locked.mkdir()
    cfg = _with_directory(load_config(write_input(tmp_path)).config, locked / "sub" / "run")
    locked.chmod(0o000)
    try:
        with pytest.raises(ConfigError, match="Cannot create or write the project directory"):
            run_preflight(cfg, [PrepareStage()], machine=MachineInfo(None, None))
    finally:
        locked.chmod(0o755)


def test_module_check_treats_import_errors_as_missing(monkeypatch):
    import importlib.util

    from gqteashmd import preflight

    def broken(name):
        raise ImportError("broken parent package")

    monkeypatch.setattr(importlib.util, "find_spec", broken)
    assert preflight._module_available("anything") is False


def _needy_stage(requirement):
    from gqteashmd.stages.base import Stage

    class Needy(Stage):
        name = "needy"
        required_modules = (requirement,)

        def plan(self, ctx):
            return []

        def run_task(self, ctx, task, task_dir):
            return {}

        def finalize(self, ctx, results):
            pass

    return Needy()


def test_module_versions_are_checked(tmp_path, write_input, monkeypatch):
    from importlib import metadata

    cfg = _with_directory(load_config(write_input(tmp_path)).config, tmp_path / "run")
    machine = MachineInfo(None, None)
    with pytest.raises(ConfigError, match=r"needs numpy >= 999.0; version .* is installed"):
        run_preflight(cfg, [_needy_stage("numpy>=999.0")], machine=machine)
    run_preflight(cfg, [_needy_stage("numpy>=1.0")], machine=machine)
    with pytest.raises(ValueError, match="requirement"):
        run_preflight(cfg, [_needy_stage("numpy>>1")], machine=machine)

    def unknown(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", unknown)
    run_preflight(cfg, [_needy_stage("numpy>=999.0")], machine=machine)  # unknown version: skip


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2.6", (2, 6, 0)),
        ("2.6rc1", (2, 6, 0)),
        ("1.0", (1, 0, 0)),
        ("1.0.0", (1, 0, 0)),
        ("2.10.1.post3", (2, 10, 1)),
        ("3.12.0.dev1", (3, 12, 0)),
        ("7", (7, 0, 0)),
    ],
)
def test_version_tuple_is_padded_and_stops_at_non_numeric(text, expected):
    from gqteashmd.preflight import _version_tuple

    assert _version_tuple(text) == expected
