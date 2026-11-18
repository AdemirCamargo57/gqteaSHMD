"""SHARC's populations.py / make_fit.py driver and the comparison arithmetic (P9b Task 5)."""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from tests.diagnosis.conftest import load_tool

from gqteashmd.analysis.populations import bootstrap_indices

ra = load_tool("sharc_compare", "run_sharc_analysis")
FIX = Path(__file__).resolve().parents[1] / "data" / "sharc" / "p9b"
TASKS = ["traj_00000", "traj_00001"]


def test_parse_populations_reads_the_stored_classical_output():
    t, p = ra.parse_populations(FIX / "analysis" / ra.CLASSICAL_OUTPUT)
    assert t[0] == 0.0 and p.shape[1] == 3
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_parse_populations_reads_the_stored_quantum_output():
    t, p = ra.parse_populations(FIX / "analysis" / ra.QUANTUM_OUTPUT)
    assert t.shape == (11,) and p.shape == (11, 3)
    assert p[1, 1] == pytest.approx(0.999999779)


def test_parse_fit_reads_normal_and_bootstrap_outputs():
    normal = ra.parse_fit(FIX / "analysis" / "transcripts" / "make_fit_classical.transcript.txt")
    assert normal["tau_fs"] == pytest.approx(104172.5466)
    assert normal["t0_fs"] is None and normal["tau_std_fs"] is None  # the intrinsic error is unused
    boot = ra.parse_fit(FIX / "analysis" / ra.FIT_OUTPUT)
    assert boot["tau_fs"] == pytest.approx(104172.5466) and boot["tau_std_fs"] == 0.0


def test_fit_undelayed_recovers_tau():
    t = np.arange(0.0, 100.0 + 1e-9, 1.0)
    per = np.tile(np.exp(-t / 30.0), (20, 1))
    r = ra.fit_undelayed(t, per, bootstrap_indices(np.random.default_rng(0), 20, 50))
    assert abs(r["tau_fs"] - 30.0) < 1e-6 and r["tau_std_fs"] < 1e-6


def test_fit_undelayed_without_population_gives_no_tau():
    t = np.arange(0.0, 10.0)
    r = ra.fit_undelayed(t, np.zeros((4, 10)), bootstrap_indices(np.random.default_rng(0), 4, 5))
    assert r["tau_fs"] is None and r["tau_std_fs"] is None


def test_fit_verdict_without_a_bootstrap_std_is_inconclusive():
    # PR5: a missing std is not taken as 0 (P9b final review, M4)
    r = ra.fit_verdict(78.5, 6.5, 78.6, None)
    assert r["verdict"] == "inconclusive" and "make_fit.py" in r["reason"]
    r = ra.fit_verdict(78.5, None, 78.6, 6.6)
    assert r["verdict"] == "inconclusive" and "our fit" in r["reason"]
    assert ra.fit_verdict(None, None, 78.6, 6.6)["verdict"] == "inconclusive"


def test_fit_verdict_compares_within_the_combined_uncertainty():
    ok = ra.fit_verdict(78.5, 6.0, 78.6, 8.0)
    assert ok["verdict"] == "pass" and abs(ok["bound"] - 40.0) < 1e-12
    assert ra.fit_verdict(30.0, 1.0, 78.6, 1.0)["verdict"] == "fail"


def test_classical_comparison_is_exact_to_the_print_precision():
    ours = np.array([[0.31, 0.69, 0.0]])
    theirs = np.array([[0.3100000000, 0.6900000000, 0.0]])
    assert ra.classical_agree(ours, theirs, decimals=10)
    assert not ra.classical_agree(ours, theirs + 0.01, decimals=10)
    assert ra.classical_agree(ours, theirs + 4e-10, decimals=9)
    assert not ra.classical_agree(ours, theirs + 6e-10, decimals=9)


def test_answers_follow_the_task_1_prompts():
    assert ra.populations_answers("classical", maxtime_fs=5.0) == [
        "Singlet_1", "end", "2", "3", "", "5.0", "", "", "",
    ]  # fmt: skip
    assert ra.populations_answers("quantum") == [
        "Singlet_1", "end", "8", "False", "3", "", "5.0", "", "", "",
    ]  # fmt: skip
    boot = ra.populations_answers("classical", bootstrap=True)
    assert boot[boot.index("5.0") + 1 :] == ["True", "", "", ""]
    with pytest.raises(ValueError, match="kind"):
        ra.populations_answers("mixed")
    assert ra.fit_answers("pop.out")[6:8] == ["", "pop.out"]
    assert "S1 = 3 4" in ra.fit_answers("pop.out") and "S1 = 3 4 5" in ra.fit_answers(
        "pop.out", n_states=4
    )
    b = ra.fit_answers("pop.out", bootstrap_samples=1000, bootstrap_dir="bd")
    assert b[6:10] == ["True", "1000", "bd", ""]


def test_tool_environment_replaces_pythonpath_and_sets_the_f6_variables(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/ours")
    monkeypatch.delenv("HOSTNAME", raising=False)
    monkeypatch.delenv("USER", raising=False)
    env = ra.sharc_tool_environment(Path("/h"), "/c/envs/s/bin/python")
    assert env["SHARC"] == "/h/bin" and env["PYTHONPATH"] == "/h/lib:/h/pysharc"
    assert env["LD_LIBRARY_PATH"].startswith("/c/envs/s/lib")
    assert env["HOSTNAME"] and env["USER"]


def test_build_tree_links_the_files_in_the_f2_layout(tmp_path):
    (tmp_path / "ens" / "traj_00003").mkdir(parents=True)
    for n in ("input", "output.dat", "output.lis"):
        (tmp_path / "ens" / "traj_00003" / n).write_text("x")
    (tmp_path / "ens" / "traj_00004").mkdir()
    (tmp_path / "ens" / "traj_00004" / "output.dat").write_text("x")  # incomplete
    skipped = ra.build_tree(tmp_path / "ens", ["traj_00003", "traj_00004"], tmp_path / "S")
    assert skipped == ["traj_00004"]
    assert (tmp_path / "S" / "TRAJ_00003" / "output.lis").read_text() == "x"
    assert not (tmp_path / "S" / "TRAJ_00004").exists()
    with pytest.raises(ValueError, match="trajectory name"):
        ra.sharc_dir_name("run_3")


def test_run_tool_reports_a_failing_script_with_the_hint(tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "x.py").write_text("import sys; sys.exit(3)\n")
    with pytest.raises(RuntimeError, match=r"x\.py failed \(exit 3.*Task 1 facts F2/F3"):
        ra.run_tool(tmp_path, "python3", "x.py", tmp_path, ["a"])


def test_held_on_steps_continues_an_early_end_with_its_last_values():
    from tests.sharc_compare.test_sharc_trajectories import _parsed

    st = load_tool("sharc_compare", "sharc_trajectories")
    full = st.from_parsed("traj_00000", _parsed([1, 1, 0, 0], [[0.0, 5.0, 7.0]] * 4), n_total=3)
    short = st.from_parsed("traj_00001", _parsed([1, 1], [[0.0, 5.0, 7.0]] * 2), n_total=3)
    active, pops = ra.held_on_steps([full, short], np.arange(4), 3)
    assert active.tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]
    assert pops[1, 3].tolist() == [0.0, 1.0, 0.0]


needs_sharc = pytest.mark.skipif(
    not (os.environ.get("SHARC_HOME") and os.environ.get("SHARC_PYTHON")),
    reason="needs SHARC_HOME and SHARC_PYTHON",
)


def _copy_fixture(tmp_path):
    ens = tmp_path / "ens"
    for k in TASKS:
        shutil.copytree(FIX / k, ens / k, ignore=shutil.ignore_patterns("output_data"))
    return ens


@pytest.mark.sharc
@needs_sharc
def test_populations_py_on_the_fixture_matches_the_stored_output(tmp_path):
    home, py = Path(os.environ["SHARC_HOME"]), os.environ["SHARC_PYTHON"]
    ens = _copy_fixture(tmp_path)
    work = tmp_path / "work"
    assert ra.build_tree(ens, TASKS, work / ra.PARENT) == []
    for k in TASKS:
        ra.run_extractor(home, py, work / ra.PARENT / ra.sharc_dir_name(k))
    for kind, name in (("classical", ra.CLASSICAL_OUTPUT), ("quantum", ra.QUANTUM_OUTPUT)):
        ra.run_tool(home, py, "populations.py", work, ra.populations_answers(kind), outputs=[name])
        t, p = ra.parse_populations(work / name)
        t_ref, p_ref = ra.parse_populations(FIX / "analysis" / name)
        np.testing.assert_array_equal(t, t_ref)
        assert ra.classical_agree(p, p_ref, decimals=ra.PRINT_DECIMALS), kind
        assert np.abs(p - p_ref).max() <= 1e-9


@pytest.mark.sharc
@needs_sharc
def test_compare_tools_on_the_fixture_agrees_with_both_populations(tmp_path):
    home, py = Path(os.environ["SHARC_HOME"]), os.environ["SHARC_PYTHON"]
    result = ra.compare_tools(
        _copy_fixture(tmp_path),
        TASKS,
        n_total=10,
        dt_fs=0.5,
        n_states=3,
        sharc_home=home,
        python=py,
        work_dir=tmp_path / "work",
        n_bootstrap=20,
    )
    assert result["n_trajectories"] == 2 and result["skipped"] == []
    assert result["classical"]["verdict"] == "pass", result["classical"]
    assert result["quantum"]["verdict"] == "pass", result["quantum"]
    assert result["fit"]["tau_sharc_fs"] == pytest.approx(104172.5466)  # no decay: documents format
