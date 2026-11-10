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


def test_edc_convention_and_substep_interpolation_keys(tmp_path, capsys):
    out = tmp_path / "out.json"
    text = (
        INPUT
        + '[dynamics]\nedc_convention = "granucci_persico"\nsubstep_interpolation = "endpoint"\n'
    )
    assert main(["model-dynamics", str(_write(tmp_path, text)), "--json", str(out)]) == 0
    capsys.readouterr()
    dynamics = json.loads(out.read_text())["dynamics"]
    assert dynamics["edc_convention"] == "granucci_persico"
    assert dynamics["substep_interpolation"] == "endpoint"
    bad = INPUT + '[dynamics]\nedc_convention = "gp"\nsubstep_interpolation = "left"\n'
    assert main(["model-dynamics", str(_write(tmp_path, bad))]) == 2
    err = capsys.readouterr().err
    assert "dynamics.edc_convention" in err and "dynamics.substep_interpolation" in err


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


def test_workers_give_the_same_counts_and_must_be_positive(tmp_path, capsys):
    outs = []
    for workers in (1, 2):
        out = tmp_path / f"out{workers}.json"
        text = INPUT + f"workers = {workers}\n"
        assert main(["model-dynamics", str(_write(tmp_path, text)), "--json", str(out)]) == 0
        outs.append(json.loads(out.read_text()))
    assert outs[0]["counts"] == outs[1]["counts"]
    assert outs[1]["ensemble"]["workers"] == 2
    capsys.readouterr()
    assert main(["model-dynamics", str(_write(tmp_path, INPUT + "workers = 0\n"))]) == 2
    assert "ensemble.workers" in capsys.readouterr().err


def test_dynamics_section_defaults_are_the_library_defaults():
    """[dynamics] takes every default from DynamicsOptions (dt_au: 1.0 here, required there)."""
    import dataclasses

    from gqteashmd.config.model_dynamics import DynamicsSection
    from gqteashmd.dynamics.options import DynamicsOptions

    section = DynamicsSection().model_dump()
    library = dataclasses.asdict(DynamicsOptions(dt_au=1.0))
    # P7a-2 Task 1 added options the driver does not use yet; the runner exposes them when the
    # driver implements them (a later task of P7a-2 removes this exemption).
    not_yet_exposed = {"coupling", "gradient_transformation", "eselect_Eh"}
    assert set(library) - set(section) == not_yet_exposed
    assert section == {k: v for k, v in library.items() if k not in not_yet_exposed}
    assert section["energy_drift_limit_au"] == 0.5 / 27.211386245988


def test_model_parameters_reach_the_model(tmp_path, capsys):
    """C = 0 removes the coupling: every trajectory stays on the lower state."""
    out = tmp_path / "out.json"
    text = INPUT.replace('name = "tully_sac"', 'name = "tully_sac"\n[model.parameters]\nC = 0.0')
    assert main(["model-dynamics", str(_write(tmp_path, text)), "--json", str(out)]) == 0
    capsys.readouterr()
    data = json.loads(out.read_text())
    assert data["model"]["parameters"] == {"C": 0.0}
    assert data["counts"]["transmitted_lower"] == 5


def test_explicit_sigma_is_passed_to_the_ensemble(tmp_path, capsys, monkeypatch):
    from gqteashmd import cli
    from gqteashmd.dynamics.ensemble import Branching

    seen = {}

    def fake(model, **kwargs):
        seen.update(kwargs)
        return Branching(transmitted_lower=kwargs["n_trajectories"], n_trajectories=5)

    monkeypatch.setattr(cli, "run_model_ensemble", fake)
    text = INPUT.replace("k = 30.0", "k = 30.0\nsigma = 0.25")
    out = tmp_path / "out.json"
    assert main(["model-dynamics", str(_write(tmp_path, text)), "--json", str(out)]) == 0
    capsys.readouterr()
    assert seen["sigma"] == 0.25 and seen["k"] == 30.0
    assert json.loads(out.read_text())["initial"]["sigma"] == 0.25


def test_max_steps_reached_counts_as_unfinished(tmp_path, capsys):
    out = tmp_path / "out.json"
    text = INPUT.replace("max_steps = 5000", "max_steps = 10")
    assert main(["model-dynamics", str(_write(tmp_path, text)), "--json", str(out)]) == 0
    capsys.readouterr()
    data = json.loads(out.read_text())
    assert data["counts"]["unfinished"] == 5
    assert data["n_finished"] == 0
    assert all(p == 0.0 for p in data["probabilities"].values())


def test_an_error_inside_a_trajectory_is_not_reported_as_invalid_input(tmp_path, capsys):
    """C = 0 at x0 = 0 makes the two states degenerate at the start: the input is valid, the
    run fails (exit 1, a distinct message), not 'Invalid model-dynamics input' (exit 2)."""
    text = INPUT.replace(
        'name = "tully_sac"', 'name = "tully_sac"\n[model.parameters]\nC = 0.0'
    ).replace("k = 30.0", 'k = 30.0\nx0 = 0.0\nsampling = "fixed"')
    assert main(["model-dynamics", str(_write(tmp_path, text))]) == 1
    err = capsys.readouterr().err
    assert "run failed inside a trajectory" in err and "degenerate" in err
    assert "Invalid model-dynamics input" not in err and "internal error" not in err
