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
