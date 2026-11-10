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
    assert section == library
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


LVC_INPUT = """[model]
name = "lvc"
template = "{template}"

[initial]
state = "S1"

[dynamics]
dt_au = 20.0

[ensemble]
n_trajectories = 8
n_steps = 50
record_every = 10
seed = 3
"""


def _lvc(tmp_path, extra="", template="lvc/lvc_soc.template", sub="in"):
    import shutil
    from pathlib import Path

    src = Path(__file__).parent / "data" / "lvc"
    shutil.copytree(src, tmp_path / "lvc", dirs_exist_ok=True)
    folder = tmp_path / sub
    folder.mkdir(exist_ok=True)
    text = LVC_INPUT.format(template="../" + template) + extra
    path = folder / "lvc.toml"
    path.write_text(text)
    return path


def test_lvc_example_runs_and_writes_the_documented_json(tmp_path, capsys):
    out = tmp_path / "out.json"
    path = _lvc(tmp_path)  # the template is relative to the input file, not to the cwd
    assert main(["model-dynamics", str(path), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "spin-free MCH populations" in text and "elapsed" in text and "T2" in text
    data = json.loads(out.read_text())
    assert data["kind"] == "lvc_populations" and data["schema_version"] == 1
    assert data["times_au"] == [0.0, 200.0, 400.0, 600.0, 800.0, 1000.0]
    assert data["labels_spin_free"] == ["S0", "S1", "T1", "T2"]
    assert len(data["labels_expanded"]) == 8
    pops = data["populations"]
    assert set(pops) == {
        "diag_classical",
        "diag_quantum",
        "mch_classical",
        "mch_quantum",
        "mch_spin_free_classical",
        "mch_spin_free_quantum",
    }
    assert len(pops["mch_quantum"]) == 6 and len(pops["mch_quantum"][0]) == 8
    assert data["counts"]["n_trajectories"] == 8
    assert data["model"]["template"] == str((tmp_path / "lvc" / "lvc_soc.template").resolve())
    assert data["ensemble"]["n_steps"] == 50


def test_lvc_triplet_ms_and_errors(tmp_path, capsys):
    path = _lvc(tmp_path, sub="a")
    text = path.read_text().replace('state = "S1"', 'state = "T2"\nms = 1')
    path.write_text(text)
    assert main(["model-dynamics", str(path)]) == 0
    capsys.readouterr()
    for old, new, key in (
        ('state = "S1"', 'state = "S1"\nms = 0', "initial.ms"),
        ('state = "S1"', 'state = "S9"', "S9"),
        ('state = "S1"', 'state = "x"', "initial.state"),
        ("n_steps = 50", "n_steps = 0", "ensemble.n_steps"),
        ("n_steps = 50", "n_steps = 5\nmax_steps = 5", "ensemble.max_steps"),
        ("[initial]", "[initial]\nk = 5.0", "initial.k"),
    ):
        bad = _lvc(tmp_path, sub="b")
        bad.write_text(bad.read_text().replace(old, new))
        assert main(["model-dynamics", str(bad)]) == 2, key
        assert key in capsys.readouterr().err


def test_lvc_missing_template_and_keys_across_models(tmp_path, capsys):
    path = _lvc(tmp_path, template="lvc/missing.template", sub="c")
    assert main(["model-dynamics", str(path)]) == 2
    assert "missing.template" in capsys.readouterr().err
    tully = _write(tmp_path, INPUT + "n_steps = 5\n")
    assert main(["model-dynamics", str(tully)]) == 2
    assert "ensemble.n_steps" in capsys.readouterr().err
    tully = _write(
        tmp_path, INPUT.replace('name = "tully_sac"', 'name = "tully_sac"\nspin_orbit = true')
    )
    assert main(["model-dynamics", str(tully)]) == 2
    assert "model.spin_orbit" in capsys.readouterr().err


def test_overlap_coupling_runs_on_a_tully_input(tmp_path, capsys):
    out = tmp_path / "out.json"
    text = INPUT + '[dynamics]\ncoupling = "overlap"\n'
    assert main(["model-dynamics", str(_write(tmp_path, text)), "--json", str(out)]) == 0
    capsys.readouterr()
    assert json.loads(out.read_text())["dynamics"]["coupling"] == "overlap"
