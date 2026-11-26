import json
import subprocess
import sys
import types

import pytest

import gqteashmd
from gqteashmd.config.loader import load_config
from gqteashmd.orchestrator import Orchestrator
from gqteashmd.project.provenance import append_provenance, collect_provenance
from gqteashmd.testing.fake_stages import BEHAVIOUR_ENV
from gqteashmd.util.fileio import read_json

FAKE = "gqteashmd.testing.fake_stages:make_pipeline"
FAST = "poll_interval_s = 0.05\nstop_grace_period_s = 2.0"


def test_collect_provenance_has_the_documented_keys():
    entry = collect_provenance()
    assert set(entry) == {
        "started_at",
        "gqteashmd",
        "git_commit",
        "git_dirty",
        "commit_source",
        "build_time",
        "python",
        "platform",
        "hostname",
        "cpu_count",
        "cpu_model",
        "packages",
    }
    assert entry["gqteashmd"] == gqteashmd.__version__
    assert entry["packages"]["pydantic"]


def test_append_keeps_history_and_survives_corruption(tmp_path):
    path = tmp_path / "provenance.json"
    append_provenance(path, {"n": 1})
    append_provenance(path, {"n": 2})
    assert [r["n"] for r in read_json(path)["runs"]] == [1, 2]
    path.write_text("{broken")
    append_provenance(path, {"n": 3})
    assert read_json(path) == {"schema_version": 1, "runs": [{"n": 3}]}


def test_each_run_appends_one_record(tmp_path, write_input, monkeypatch):
    monkeypatch.setenv(BEHAVIOUR_ENV, json.dumps({}))
    path = write_input(tmp_path, execution=FAST)
    Orchestrator(load_config(path), pipeline_ref=FAKE).run()
    Orchestrator(load_config(path), pipeline_ref=FAKE).run()
    runs = read_json(tmp_path / "run" / ".gqteashmd" / "provenance.json")["runs"]
    assert len(runs) == 2


@pytest.mark.parametrize(
    "content", [b"\xff\xfe\x00binary", b'{"runs": "abc"}', b"[1, 2]", b'{"runs": 5}']
)
def test_damaged_provenance_is_replaced(tmp_path, content):
    path = tmp_path / "provenance.json"
    path.write_bytes(content)
    append_provenance(path, {"n": 1})
    assert read_json(path) == {"schema_version": 1, "runs": [{"n": 1}]}


def test_missing_git_gives_null_commit(monkeypatch):
    def no_git(*_a, **_k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    monkeypatch.setitem(sys.modules, "gqteashmd._build", None)
    entry = collect_provenance()
    assert entry["git_commit"] is None and entry["git_dirty"] is None
    assert entry["commit_source"] is None and entry["build_time"] is None


def test_provenance_takes_the_build_stamp_without_a_checkout(monkeypatch, tmp_path):
    from gqteashmd import build_info

    stamp = types.ModuleType("gqteashmd._build")
    stamp.COMMIT, stamp.DIRTY, stamp.BUILD_TIME = "d" * 40, False, "2026-11-25T10:00:00+00:00"
    monkeypatch.setattr(build_info, "_checkout_root", lambda: tmp_path)
    monkeypatch.setitem(sys.modules, "gqteashmd._build", stamp)
    entry = collect_provenance()
    assert (entry["git_commit"], entry["git_dirty"]) == ("d" * 40, False)
    assert entry["commit_source"] == "build"
    assert entry["build_time"] == "2026-11-25T10:00:00+00:00"


def test_provenance_from_a_live_checkout_has_no_build_time(monkeypatch):
    from gqteashmd import build_info
    from gqteashmd.build_info import CommitInfo

    monkeypatch.setattr(
        build_info, "_live_checkout", lambda: CommitInfo("e" * 40, True, "git", None)
    )
    entry = collect_provenance()
    assert (entry["git_commit"], entry["git_dirty"]) == ("e" * 40, True)
    assert entry["commit_source"] == "git" and entry["build_time"] is None


def test_provenance_records_the_cpu_model():
    from gqteashmd.project.provenance import collect_provenance

    entry = collect_provenance()
    assert "cpu_model" in entry
    assert entry["cpu_model"] is None or isinstance(entry["cpu_model"], str)
