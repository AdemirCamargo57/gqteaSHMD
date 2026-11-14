from tests.diagnosis.conftest import load_tool


def test_summarize_counts_the_rule_solutions():
    tool = load_tool("diagnosis", "start_rule_check")
    rows = [
        {"task": "traj_00000", "solution": 0},
        {"task": "traj_00001", "solution": 0},
        {"task": "traj_00002", "solution": 2},
        {"task": "traj_00003", "solution": None},
        {"task": "traj_00004", "error": "RuntimeError: x"},
    ]
    assert tool.summarize(rows) == {
        "n_starts": 5,
        "n_sol0": 2,
        "n_other": 1,
        "n_unmatched": 1,
        "n_errors": 1,
        "other": ["traj_00002"],
        "unmatched": ["traj_00003"],
        "errors": ["traj_00004"],
    }
