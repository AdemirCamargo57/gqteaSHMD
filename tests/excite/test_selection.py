import numpy as np
import pytest

from gqteashmd.excite.selection import (
    Candidate,
    candidate_weight,
    candidates_from,
    select_bright,
    select_fixed,
)

E = np.array([[5.0, 7.0], [5.5, 9.5], [4.0, 6.5]])
F = np.array([[0.2, 0.4], [0.1, 0.8], [0.0, 0.3]])
OK = np.array([True, True, False])


def test_candidates_window_states_and_failed_samples():
    c = candidates_from(E, F, OK, ["S1", "S2"], (4.5, 8.0))
    assert [(x.sample, x.state) for x in c] == [(1, "S1"), (1, "S2"), (2, "S1")]
    c = candidates_from(E, F, OK, ["S2"], None)
    assert [(x.sample, x.state) for x in c] == [(1, "S2"), (2, "S2")]
    c = candidates_from(E, F, OK, ["S2", "S1"], None)
    assert [(x.sample, x.state) for x in c] == [(1, "S1"), (1, "S2"), (2, "S1"), (2, "S2")]


def test_bright_uses_fmax_over_candidates_only():
    # state S2 of sample 2 (f = 0.8) is outside the window, so f_max = 0.4
    c = candidates_from(E, F, OK, ["S1", "S2"], (4.5, 8.0))
    sel = select_bright(c, 10, np.random.default_rng(0))
    assert sel.f_max == 0.4 and sel.candidates == 3
    # f / dE^2 (default): weights 0.2/25, 0.4/49, 0.1/30.25
    assert sel.weight_max == pytest.approx(0.4 / 49.0)
    sel_f = select_bright(c, 10, np.random.default_rng(0), weighting="f")
    assert sel_f.weight_max == 0.4


def test_bright_acceptance_frequencies_match_f_over_fmax():
    e = np.full((20000, 2), 5.0)
    f = np.tile([0.1, 0.4], (20000, 1))
    c = candidates_from(e, f, np.ones(20000, bool), ["S1", "S2"], None)
    sel = select_bright(c, 10**9, np.random.default_rng(1))
    states = [x.state for x in sel.selected]
    p1, p2 = states.count("S1") / 20000, states.count("S2") / 20000
    assert abs(p1 - 0.25) < 4 * np.sqrt(0.25 * 0.75 / 20000)
    assert p2 == 1.0  # f = f_max is always accepted (r < 1)
    assert sel.accepted == len(sel.selected)


@pytest.mark.parametrize(
    "weighting, p1",
    [
        ("f", 0.1 / 0.4),  # S1 (5 eV, f 0.1) against S2 (7 eV, f 0.4)
        ("f_over_dE2", (0.1 / 25.0) / (0.4 / 49.0)),
    ],
)
def test_bright_acceptance_frequencies_per_weighting(weighting, p1):
    n = 20000
    e = np.tile([5.0, 7.0], (n, 1))
    f = np.tile([0.1, 0.4], (n, 1))
    c = candidates_from(e, f, np.ones(n, bool), ["S1", "S2"], None)
    sel = select_bright(c, 10**9, np.random.default_rng(3), weighting=weighting)
    states = [x.state for x in sel.selected]
    got = states.count("S1") / n
    assert abs(got - p1) < 4 * np.sqrt(p1 * (1 - p1) / n)
    assert states.count("S2") == n  # the maximum is always accepted
    assert sel.f_max == 0.4


def test_weightings_differ():
    n = 2000
    e = np.tile([5.0, 7.0], (n, 1))
    f = np.tile([0.1, 0.4], (n, 1))
    c = candidates_from(e, f, np.ones(n, bool), ["S1", "S2"], None)
    a = select_bright(c, 10**9, np.random.default_rng(3), weighting="f")
    b = select_bright(c, 10**9, np.random.default_rng(3))
    assert a.accepted != b.accepted


def test_bright_is_reproducible_and_truncates_in_order():
    c = candidates_from(E, F, OK, ["S1", "S2"], None)
    a = select_bright(c, 1, np.random.default_rng(5))
    b = select_bright(c, 1, np.random.default_rng(5))
    assert a == b and len(a.selected) <= 1


def test_bright_without_bright_candidate_selects_nothing():
    c = candidates_from(np.array([[5.0]]), np.array([[0.0]]), np.array([True]), ["S1"], None)
    sel = select_bright(c, 3, np.random.default_rng(0))
    assert sel.selected == () and sel.f_max == 0.0 and sel.accepted == 0


def test_fixed_takes_first_n_in_order():
    c = candidates_from(E, F, OK, ["S1"], None)
    sel = select_fixed(c, 1)
    assert [(x.sample, x.state) for x in sel.selected] == [(1, "S1")]
    assert sel.accepted == 2 and sel.candidates == 2 and sel.f_max == 0.2


def test_bright_consumes_one_draw_per_candidate():
    c = candidates_from(E, F, OK, ["S1", "S2"], None)
    rng = np.random.default_rng(11)
    select_bright(c, 2, rng)
    ref = np.random.default_rng(11)
    ref.random(len(c))
    assert rng.random() == ref.random()


def test_candidates_skip_non_finite_and_non_positive_pairs():
    e = np.array([[5.0, 0.0], [-0.3, np.nan], [np.inf, 6.0], [5.0, 6.0]])
    f = np.array([[0.1, 0.2], [0.1, 0.2], [0.1, np.nan], [0.1, 0.2]])
    c = candidates_from(e, f, np.ones(4, bool), ["S1", "S2"], None)
    assert [(x.sample, x.state) for x in c] == [(1, "S1"), (4, "S1"), (4, "S2")]
    sel = select_bright(c, 10, np.random.default_rng(0))  # no ZeroDivisionError, finite w_max
    assert np.isfinite(sel.weight_max) and sel.weight_max == pytest.approx(0.2 / 36.0)


@pytest.mark.parametrize(
    "states, match",
    [
        (["S3"], "S3 is not in the excitation data"),
        (["S1", "S1"], "duplicate state"),
        (["T1"], "T1 is not in the excitation data"),
    ],
)
def test_candidates_reject_unknown_or_duplicate_states(states, match):
    with pytest.raises(ValueError, match=match):
        candidates_from(E, F, OK, states, None)


def test_candidates_reject_inconsistent_shapes():
    with pytest.raises(ValueError, match="inconsistent shapes"):
        candidates_from(E, F[:, :1], OK, ["S1"], None)


@pytest.mark.parametrize("de", [0.0, -1.0, float("nan")])
def test_candidate_weight_rejects_bad_energies(de):
    bad = Candidate(1, "S1", de, 0.1)
    for weighting in ("f_over_dE2", "f"):
        with pytest.raises(ValueError, match="finite and positive"):
            candidate_weight(bad, weighting)
    with pytest.raises(ValueError, match="sample 1 S1"):
        select_bright([bad], 1, np.random.default_rng(0))
