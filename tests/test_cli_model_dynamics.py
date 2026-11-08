import json

from gqteashmd.cli import main

INPUT = """\
[model]
name = "tully_sac"

[initial]
k = 30.0

[ensemble]
n_trajectories = 5
max_steps = 5000
"""


def _write(tmp_path, text=INPUT):
    path = tmp_path / "md.toml"
    path.write_text(text)
    return path


def test_model_dynamics_prints_a_table_and_writes_json(tmp_path, capsys):
    out = tmp_path / "out.json"
    assert main(["model-dynamics", str(_write(tmp_path)), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "transmitted_lower" in text and "95% interval" in text
    data = json.loads(out.read_text())
    for key in (
        "schema_version",
        "model",
        "initial",
        "dynamics",
        "ensemble",
        "counts",
        "probabilities",
        "intervals_95",
        "n_finished",
    ):
        assert key in data
    assert data["schema_version"] == 1
    assert data["n_finished"] == 5
    assert data["initial"]["sigma"] == 20 / 30
    assert len(data["intervals_95"]["transmitted_lower"]) == 2
    assert not (tmp_path / "run").exists()


def test_all_violations_are_reported_and_name_the_key(tmp_path, capsys):
    bad = INPUT.replace("k = 30.0", "k = 0.0\nbogus = 1").replace(
        'name = "tully_sac"', 'name = "tully_sac"\n[model.parameters]\nZ = 1.0'
    )
    assert main(["model-dynamics", str(_write(tmp_path, bad))]) == 2
    err = capsys.readouterr().err
    assert "initial.k" in err and "initial.bogus" in err and "model.parameters" in err
    assert "Z" in err


def test_unknown_model_and_missing_file_are_exit_2(tmp_path, capsys):
    assert main(["model-dynamics", str(_write(tmp_path, INPUT.replace("tully_sac", "x")))]) == 2
    assert "model.name" in capsys.readouterr().err
    assert main(["model-dynamics", str(tmp_path / "none.toml")]) == 2


def test_state_out_of_range_is_exit_2(tmp_path, capsys):
    bad = INPUT.replace("k = 30.0", "k = 30.0\nstate = 2")
    assert main(["model-dynamics", str(_write(tmp_path, bad))]) == 2
    assert "initial.state" in capsys.readouterr().err
