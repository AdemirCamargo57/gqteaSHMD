import numpy as np

from gqteashmd.excite.selection import candidates_from, select_bright, select_fixed

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
