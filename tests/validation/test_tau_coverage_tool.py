import numpy as np
from tests.diagnosis.conftest import load_tool

tc = load_tool("validation", "tau_coverage")


def test_constants_are_the_spec_values():
    assert tc.T0_FS == 10.0 and tc.TAU_FS == 40.0 and tc.DT_FS == 0.5 and tc.T_END_FS == 100.0
    assert tc.SIZES == (20, 50, 100) and tc.REPETITIONS == 400 and tc.N_BOOTSTRAP == 500
    assert tc.THRESHOLD == 0.90


def test_one_repetition_reports_an_interval():
    r = tc.coverage_once(100, np.random.default_rng(0), n_bootstrap=50)
    assert set(r) >= {"covered", "tau_fs", "interval"}
    assert r["interval"] is None or r["interval"][0] <= r["interval"][1]


def test_n_min():
    assert tc.n_min({20: 0.85, 50: 0.91, 100: 0.95}) == 50
    assert tc.n_min({20: 0.92, 50: 0.91, 100: 0.95}) == 20
    assert tc.n_min({20: 0.80, 50: 0.85, 100: 0.89}) is None
