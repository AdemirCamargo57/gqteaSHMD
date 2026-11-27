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
    assert (
        text.index("## Fit")
        < text.index("## Other estimator")
        < text.index("## Hops (complete trajectories)")
    )
    s = _censored_summary(summary_example)
    s["estimators"] = {**s["estimators"], "complete": None}
    section = analysis_markdown(s).split("## Other estimator\n", 1)[1].split("\n## ", 1)[0]
    assert "- tau_fs: -\n" in section and "- t0_interval_fs: -" in section
    section = analysis_markdown(summary_example).split("## Other estimator\n", 1)[1]
    assert section.startswith("- estimator: -\n- tau_fs: -\n")


def test_markdown_says_hops_and_energy_cover_the_complete_trajectories(summary_example):
    # final review m4: under the censored default these sections cover the complete ones only
    text = analysis_markdown(_censored_summary(summary_example))
    assert "\n## Hops (complete trajectories)\n- pairs: S1->S0 2\n" in text
    assert "\n## Energy (complete trajectories)\n- drift_eV: " in text
    assert "\n## Hops\n" not in text and "\n## Energy\n" not in text


def test_markdown_shows_the_lifetime_range(summary_example):
    s = dict(summary_example)
    fit = {
        "tau_fs": 50.0,
        "tau_interval_fs": [40.0, 60.0],
        "t0_fs": 5.0,
        "t0_interval_fs": [4.0, 6.0],
    }
    s["estimators"] = {
        "complete": {"fit": dict(fit, tau_fs=80.0)},
        "censored": {"fit": dict(fit, tau_fs=62.5)},
        "hold_in_s0": {"fit": fit},
    }
    s["tau_range_fs"] = {
        "min": 50.0,
        "max": 80.0,
        "min_estimator": "hold_in_s0",
        "max_estimator": "complete",
    }
    text = analysis_markdown(s)
    section = text.split("## Lifetime range\n", 1)[1].split("\n## ", 1)[0]
    assert "- complete: tau 80 fs [40, 60], t0 5 fs [4, 6]" in section
    assert "- range: 50-80 fs (hold_in_s0 to complete)" in section
    assert "Which estimate is closest to the truth is not known (USER_MANUAL 8.19)." in section
    assert text.index("## Other estimator") < text.index("## Lifetime range")
    empty = analysis_markdown(summary_example).split("## Lifetime range\n", 1)[1]
    assert "- range: -" in empty


TRANSITIONS = {
    "kinds": {
        "S-S": {"at_hop": 0, "without_hop": 0},
        "S-T": {"at_hop": 3, "without_hop": 1},
        "T-S": {"at_hop": 2, "without_hop": 0},
        "T-T": {"at_hop": 0, "without_hop": 4},
    },
    "pairs": {"S1->T1": 3, "T1->S0": 2},
    "first_singlet_to_triplet_fs": {
        "per_trajectory": {"traj_00000": 12.5, "traj_00001": None},
        "n_crossed": 1,
        "n_censored": 1,
        "mean": 12.5,
        "median": 12.5,
    },
}


def test_markdown_has_an_intersystem_crossing_block_with_triplets(summary_example):
    text = analysis_markdown({**summary_example, "transitions": TRANSITIONS})
    section = text.split("\n## Intersystem crossing\n", 1)[1].split("\n## ", 1)[0]
    assert section.startswith(
        "Changes of the active spin-free state (complete trajectories, up to the horizon)\n"
    )
    assert "- S-T: at hop 3, without hop 1\n" in section
    assert "- T-S: at hop 2, without hop 0\n" in section
    assert "- S-S: at hop 0, without hop 0\n" in section
    assert "- T-T: at hop 0, without hop 4\n" in section
    assert "- pairs: S1->T1 3, T1->S0 2\n" in section
    assert "- first S-to-T time (fs): n_crossed 1, n_censored 1, mean 12.5, median 12.5\n" in (
        section
    )
    assert "- first S-to-T time per trajectory (fs): traj_00000 12.5, traj_00001 -" in section
    assert text.index("## Hops (complete trajectories)") < text.index("## Intersystem crossing")
    assert text.index("## Intersystem crossing") < text.index("## Energy (complete trajectories)")


def test_markdown_of_an_uncomputed_transitions_block(summary_example):
    # a triplet project whose analysis stopped: the key exists with null
    text = analysis_markdown({**summary_example, "transitions": None})
    section = text.split("\n## Intersystem crossing\n", 1)[1].split("\n## ", 1)[0]
    assert section == "not computed\n"


def test_markdown_without_triplets_has_no_intersystem_crossing_block(summary_example):
    assert "Intersystem crossing" not in analysis_markdown(summary_example)
