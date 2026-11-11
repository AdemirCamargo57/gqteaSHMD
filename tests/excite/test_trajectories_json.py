import json

import pytest

from gqteashmd.excite.artifacts import SelectedTrajectory, read_trajectories_json


def _write(path, trajectories):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "n_trajectories": len(trajectories),
                "trajectories": trajectories,
            }
        )
    )


def test_reads_the_entries(tmp_path):
    path = tmp_path / "trajectories.json"
    _write(
        path,
        [{"index": 0, "sample": 4, "state": "S2", "rng": {"stage": "dynamics", "task_index": 0}}],
    )
    (entry,) = read_trajectories_json(path)
    assert entry == SelectedTrajectory(0, 4, "S2", {"stage": "dynamics", "task_index": 0})
    assert entry.task_id == "traj_00000"


@pytest.mark.parametrize("content", ["not json", '{"trajectories": [{"index": 0}]}', "{}"])
def test_malformed_file_is_a_value_error_naming_it(tmp_path, content):
    path = tmp_path / "trajectories.json"
    path.write_text(content)
    with pytest.raises(ValueError, match="trajectories.json"):
        read_trajectories_json(path)


def test_missing_file_is_a_value_error(tmp_path):
    with pytest.raises(ValueError, match="trajectories.json"):
        read_trajectories_json(tmp_path / "trajectories.json")
