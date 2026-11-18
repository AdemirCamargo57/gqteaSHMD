"""The SHARC and our trajectory readers and the kept rule (P9b spec 3.2, plan Task 2)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tests.diagnosis.conftest import load_tool, make_rows, write_trajectory

from gqteashmd.core.units import AU_TIME_IN_FS, HARTREE_IN_EV

st = load_tool("sharc_compare", "sharc_trajectories")
FIXTURES = Path(__file__).resolve().parents[1] / "data" / "sharc" / "p9b"


def _parsed(active, e_pot_eV, *, n=3, dt_fs=0.5, ekin=0.0):
    m = len(active)
    h = np.zeros((m, n, n), dtype=complex)
    for k in range(m):
        h[k] = np.diag(np.asarray(e_pot_eV[k]) / HARTREE_IN_EV)
    c = np.zeros((m, n), dtype=complex)
    c[np.arange(m), active] = 1.0
    return SimpleNamespace(
        steps=np.arange(m),
        hamiltonian=h,
        c=c,
        active=np.asarray(active),
        e_kinetic=np.full(m, ekin),
        dt_au=dt_fs / AU_TIME_IN_FS,
    )


def test_from_parsed_builds_hops_times_and_total_energy():
    e = [[0.0, 5.0, 7.0]] * 4
    t = st.from_parsed("traj_00003", _parsed([1, 1, 0, 0], e, ekin=0.01), n_total=3)
    assert t.task == "traj_00003" and t.outcome == "finished" and t.initial_state == 1
    np.testing.assert_allclose(t.time_fs, [0.0, 0.5, 1.0, 1.5])
    assert t.hop_to.tolist() == [-1, -1, 0, -1] and t.hop_from.tolist() == [-1, -1, 1, -1]
    assert np.all(t.ticks == 0) and not t.frustrated.any()
    np.testing.assert_allclose(t.e_total[0], 5.0 / HARTREE_IN_EV + 0.01)
    np.testing.assert_allclose(t.populations[2], [1.0, 0.0, 0.0])


def test_a_short_sharc_run_is_stopped():
    t = st.from_parsed("traj_00000", _parsed([1, 1], [[0.0, 5.0, 7.0]] * 2), n_total=3)
    assert t.outcome == st.STOPPED and not st.is_kept(t, 3)


def test_kept_rule_uses_the_jump_up_to_n_total():
    e = [[0.0, 5.0, 7.0], [0.0, 5.05, 7.0], [0.0, 5.3, 7.0], [0.0, 9.0, 7.0]]
    t = st.from_parsed("traj_00000", _parsed([1, 1, 1, 1], e), n_total=2)
    # steps 0-2 jump at most 0.25 eV (> 0.1): not kept; the 3.7 eV jump at step 3 is beyond n_total
    assert abs(st.max_jump_up_to(t, 2) - 0.25) < 1e-9
    assert not st.is_kept(t, 2)
    small = st.from_parsed("traj_00000", _parsed([1, 1, 1], e[:2] + [e[1]]), n_total=2)
    assert st.is_kept(small, 2)


def test_none_is_not_kept():
    assert not st.is_kept(None, 10)


def test_our_output_h5_reads_back(tmp_path):
    rows = make_rows([5.0, 5.0, 5.05], [[0.0, 5.0], [0.0, 5.0], [0.0, 5.0]])
    write_trajectory(tmp_path / "traj_00000", rows, outcome="finished")
    t = st.read_our_trajectory(tmp_path / "traj_00000" / "output.h5", "traj_00000")
    assert t is not None and t.outcome == "finished" and t.active.tolist() == [1, 1, 1]
    assert st.is_kept(t, 2)
    assert st.read_our_trajectory(tmp_path / "missing" / "output.h5", "x") is None


def test_the_fixture_reads_back():
    t = st.read_sharc_trajectory(FIXTURES / "traj_00000", "traj_00000", n_total=10)
    assert t is not None and t.outcome == "finished"
    assert t.grid_step.tolist() == list(range(11)) and t.populations.shape == (11, 3)
    np.testing.assert_allclose(t.populations.sum(axis=1), 1.0, atol=1e-6)
    assert st.read_sharc_trajectory(FIXTURES / "nope", "x", n_total=10) is None
