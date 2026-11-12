import json
import math

import numpy as np

from gqteashmd.analysis.report import analysis_markdown, json_safe


def test_json_safe_replaces_non_finite_values():
    value = {"a": math.nan, "b": [1.0, math.inf], "c": np.float64(2.5), "d": np.arange(2)}
    safe = json_safe(value)
    assert safe == {"a": None, "b": [1.0, None], "c": 2.5, "d": [0, 1]}
    json.dumps(safe, allow_nan=False)


def test_json_safe_tuples_and_bools():
    safe = json_safe({"t": (1.0, math.nan), "flag": True, "n": np.bool_(False)})
    assert safe == {"t": [1.0, None], "flag": True, "n": False}
    assert safe["flag"] is True and safe["n"] is False


def test_markdown_mentions_the_main_numbers(summary_example):
    text = analysis_markdown(summary_example)
    assert text.startswith("# Analysis")
    for needle in ("S0", "tau", "excluded", "horizon", "traj_00001"):
        assert needle in text, needle


def test_markdown_prints_integers_as_integers(summary_example):
    s = dict(summary_example)
    s["settings"] = {**s["settings"], "project_seed": 12345}
    s["outcomes"] = {**s["outcomes"], "finished": 10000}
    s["fit"] = {**s["fit"], "tau_fs": 40.123456, "rms": math.nan}
    text = analysis_markdown(s)
    assert "- project_seed: 12345\n" in text
    assert "- finished: 10000\n" in text
    assert "- tau_fs: 40.12\n" in text  # floats keep four significant digits
    assert "- rms: -\n" in text  # a non-finite float is printed as '-'
    assert "- exclude_energy_jumps: True\n" in text


def test_markdown_shows_the_small_ensemble_flag(summary_example):
    assert "- small_ensemble: True\n" in analysis_markdown(summary_example)
    s = {**summary_example, "small_ensemble": False}
    assert "- small_ensemble: False\n" in analysis_markdown(s)


def test_markdown_survives_a_failed_fit(summary_example):
    s = dict(summary_example)
    s["fit"] = {k: None for k in s["fit"]}
    s["hops"] = None
    s["energy"] = None
    s["table"] = []
    s["stage_failed"] = True
    s["error"] = "too many excluded"
    text = analysis_markdown(s)
    assert "**Stage failed: too many excluded**" in text
    assert "tau_fs: -" in text
