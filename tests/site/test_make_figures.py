"""tools/site/make_figures.py on small written files."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "site"))
import make_figures as mf  # noqa: E402

from gqteashmd.core.units import AU_TIME_IN_FS, BOHR_IN_ANGSTROM  # noqa: E402
from gqteashmd.trajectory.store import log_dtype, log_row, write_output_h5  # noqa: E402

LABELS = ("S0", "S1", "S2")


def _output_h5(path: Path, active: list[int]) -> Path:
    dtype = log_dtype(len(LABELS), 6)
    rows = []
    for k, a in enumerate(active):
        previous = active[k - 1] if k else a
        rows.append(
            log_row(
                dtype,
                step=k,
                grid_step=k,
                time_au=k * 0.5 / AU_TIME_IN_FS,
                active=a,
                hop_from=previous if a != previous else -1,
                hop_to=a if a != previous else -1,
                q=np.array([0.0, 0.0, 0.0, 1.0 + k, 0.0, 0.0]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_output_h5(
        path,
        np.array(rows, dtype=dtype),
        labels=LABELS,
        n_atoms=2,
        output_every=1,
        attrs={
            "dt_fs": 0.5,
            "outcome": "finished",
            "task": "traj_00000",
            "initial_state": "S1",
            "sample": 1,
            "symbols": ["C", "H"],
        },
    )
    return path


def _columns(n: int = 5) -> dict[str, np.ndarray]:
    t = 25.0 * np.arange(n)
    s1 = np.linspace(1.0, 0.4, n)
    cols = {"time_fs": t, "fit_curve": s1 + 0.01}
    for name, p in (("S0", 1.0 - s1), ("S1", s1), ("S2", np.zeros(n))):
        cols[f"P_{name}"] = p
        cols[f"P_{name}_lower"] = np.clip(p - 0.05, 0.0, 1.0)
        cols[f"P_{name}_upper"] = np.clip(p + 0.05, 0.0, 1.0)
    return cols


ANALYSIS = {
    "n_included": 67,
    "n_selected": 100,
    "fit": {
        "tau_fs": 80.13602924,
        "t0_fs": 12.8353,
        "tau_interval_fs": [57.62081651, 110.05550736],
        "states": ["S1", "S2"],
    },
}


def test_trajectory_record(tmp_path):
    rec = mf.trajectory_record(
        _output_h5(tmp_path / "output.h5", [1, 1, 0, 0]), source="src", method="m"
    )
    assert set(rec) == {
        "source",
        "method",
        "dt_fs",
        "symbols",
        "state_labels",
        "time_fs",
        "active",
        "hop_index",
        "positions_angstrom",
    }
    assert rec["hop_index"] == 2
    assert rec["active"] == [1, 1, 0, 0]
    assert rec["symbols"] == ["C", "H"]
    assert rec["state_labels"] == list(LABELS)
    assert rec["time_fs"] == pytest.approx([0.0, 0.5, 1.0, 1.5])
    pos = np.array(rec["positions_angstrom"])
    assert pos.shape == (4, 2, 3)
    assert pos[3, 1, 0] == pytest.approx(round(4.0 * BOHR_IN_ANGSTROM, 3))
    json.dumps(rec)  # plain JSON types only


def test_trajectory_without_a_hop_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no hop"):
        mf.trajectory_record(_output_h5(tmp_path / "o.h5", [1, 1, 1]), source="s", method="m")


def test_read_populations(tmp_path):
    path = tmp_path / "populations.csv"
    path.write_text("# comment\ntime_fs,P_S0\n0.0,0.0\n0.5,0.25\n", encoding="utf-8")
    cols = mf.read_populations(path)
    assert list(cols) == ["time_fs", "P_S0"]
    assert cols["P_S0"] == pytest.approx([0.0, 0.25])


def test_population_svg():
    svg = mf.population_svg(_columns(), ANALYSIS)
    assert svg.startswith("<svg ") and svg.endswith("</svg>")
    assert 'data-n-included="67"' in svg and 'data-n-selected="100"' in svg
    assert 'data-tau-fs="80.136"' in svg and 'data-t0-fs="12.835"' in svg
    assert 'data-tau-interval-fs="57.621 110.056"' in svg
    assert "67 of 100 trajectories" in svg
    for state in ("s0", "s1", "s2"):
        assert f'class="pop-band pop-{state}"' in svg
        assert f'class="pop-line pop-{state}"' in svg
    assert 'class="pop-fit"' in svg
    assert 'role="img"' in svg and '<title id="pop-title">' in svg


def test_population_svg_refuses_another_estimator():
    with pytest.raises(ValueError, match="estimator = 'censored'"):
        mf.population_svg(_columns(), {**ANALYSIS, "estimator": "censored"})


def test_population_svg_refuses_other_fit_states():
    fit = {**ANALYSIS["fit"], "states": ["S1"]}
    with pytest.raises(ValueError, match=r"\['S1'\]"):
        mf.population_svg(_columns(), {**ANALYSIS, "fit": fit})


def test_splice_replaces_between_the_markers():
    html = f"a\n{mf.BEGIN}\nold\n{mf.END}\nb\n"
    once = mf.splice(html, "<svg>new</svg>")
    assert once == f"a\n{mf.BEGIN}\n<svg>new</svg>\n{mf.END}\nb\n"
    assert mf.splice(once, "<svg>new</svg>") == once


def test_splice_needs_both_markers_once():
    with pytest.raises(ValueError, match="marker"):
        mf.splice("no markers", "<svg/>")
    with pytest.raises(ValueError, match="marker"):
        mf.splice(f"{mf.BEGIN}{mf.END}{mf.BEGIN}{mf.END}", "<svg/>")


def test_main_writes_both_figures(tmp_path):
    project = tmp_path / "run"
    _output_h5(project / "06_dynamics" / "tasks" / "traj_00007" / "output.h5", [1, 0, 0])
    analysis = project / "07_analysis"
    analysis.mkdir(parents=True)
    cols = _columns()
    header = list(cols)
    lines = ["# gqteashmd analysis", ",".join(header)]
    lines += [",".join(f"{cols[h][i]:.6e}" for h in header) for i in range(len(cols["time_fs"]))]
    (analysis / "populations.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (analysis / "analysis.json").write_text(json.dumps(ANALYSIS), encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "index.html").write_text(f"x\n{mf.BEGIN}\n{mf.END}\ny\n", encoding="utf-8")

    assert mf.main([str(project), "--task", "traj_00007", "--docs", str(docs)]) == 0

    rec = json.loads((docs / "trajectory.json").read_text(encoding="utf-8"))
    assert rec["hop_index"] == 1
    assert str(tmp_path) not in json.dumps(rec)
    assert 'data-tau-fs="80.136"' in (docs / "index.html").read_text(encoding="utf-8")
