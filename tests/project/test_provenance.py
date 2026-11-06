import json

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
        "python",
        "platform",
        "hostname",
        "cpu_count",
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
