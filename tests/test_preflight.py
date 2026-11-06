import json
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
def test_ordinary_linux_project_directory_is_accepted(tmp_path, write_input, directory):
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
