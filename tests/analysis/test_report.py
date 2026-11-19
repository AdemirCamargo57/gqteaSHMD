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


def _censored_summary(summary_example):
    s = dict(summary_example)
    s["settings"] = {**s["settings"], "estimator": "censored"}
    s["estimator"] = "censored"
    s["n_complete"] = 1
    s["n_censored"] = 1
    s["censored"] = [
        {
            "task": "traj_00002",
            "outcome": "overlap_failure",
            "reason": "outcome",
            "censored_at_fs": 4.5,
        }
    ]
    s["estimators"] = {
        "censored": {"n": 2, "fit": dict(s["fit"])},
        "complete": {
            "n": 1,
            "fit": {
                **s["fit"],
                "tau_fs": 61.25,
                "tau_interval_fs": [50.0, 70.0],
                "t0_fs": 3.5,
                "t0_interval_fs": [2.0, 4.0],
            },
        },
    }
    return s


def test_markdown_shows_the_estimator_and_its_counts(summary_example):
    text = analysis_markdown(_censored_summary(summary_example))
    assert "- estimator: censored\n" in text
    assert "- n_included: 2\n- n_complete: 1\n- n_censored: 1\n" in text
    old = analysis_markdown(summary_example)  # a summary without the keys still renders
    assert "- estimator: -\n" in old and "- n_complete: -\n" in old


def test_markdown_shows_the_empty_rows_of_the_censored_estimator(summary_example):
    # final review I2: the count of held state probabilities is printed
    s = _censored_summary(summary_example)
    s["estimators"]["censored"]["empty_rows"] = 3
    text = analysis_markdown(s)
    assert "- n_censored: 1\n- empty_rows: 3\n" in text
    assert "- empty_rows: -\n" in analysis_markdown(summary_example)  # no estimators block


def test_markdown_lists_the_censored_trajectories(summary_example):
    text = analysis_markdown(_censored_summary(summary_example))
    section = text.split("## Censored trajectories\n", 1)[1].split("\n## ", 1)[0]
    assert "| task | outcome | reason | censored_at_fs |" in section
    assert "| traj_00002 | overlap_failure | outcome | 4.5 |" in section
    assert text.index("## Excluded trajectories") < text.index("## Censored trajectories")
    s = {**_censored_summary(summary_example), "censored": []}
    section = analysis_markdown(s).split("## Censored trajectories\n", 1)[1]
    assert section.startswith("none censored\n")


def test_markdown_shows_the_other_estimator(summary_example):
    text = analysis_markdown(_censored_summary(summary_example))
    section = text.split("## Other estimator\n", 1)[1].split("\n## ", 1)[0]
    assert "- estimator: complete\n" in section
    assert "- tau_fs: 61.25\n" in section and "- tau_interval_fs: [50, 70]\n" in section
    assert "- t0_fs: 3.5\n" in section and "- t0_interval_fs: [2, 4]" in section
    assert text.index("## Fit") < text.index("## Other estimator") < text.index("## Hops")
    s = _censored_summary(summary_example)
    s["estimators"] = {**s["estimators"], "complete": None}
    section = analysis_markdown(s).split("## Other estimator\n", 1)[1].split("\n## ", 1)[0]
    assert "- tau_fs: -\n" in section and "- t0_interval_fs: -" in section
    section = analysis_markdown(summary_example).split("## Other estimator\n", 1)[1]
    assert section.startswith("- estimator: -\n- tau_fs: -\n")
