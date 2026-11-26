"""Build the two data figures of the project website (docs/) from a gqteashmd project.

Usage, in WSL from the repository root:

    python tools/site/make_figures.py PROJECT --task traj_00033 --docs docs

PROJECT is a gqteashmd project directory. The script reads
PROJECT/06_dynamics/tasks/TASK/output.h5, PROJECT/07_analysis/populations.csv and
PROJECT/07_analysis/analysis.json; it writes DOCS/trajectory.json and replaces the populations
figure between the two markers of DOCS/index.html. The project is only read.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.core.units import BOHR_IN_ANGSTROM
from gqteashmd.trajectory.store import read_output_h5

BEGIN = "<!-- populations-svg:begin -->"
END = "<!-- populations-svg:end -->"
DYNAMICS_DIR = "06_dynamics"
ANALYSIS_DIR = "07_analysis"
DEFAULT_METHOD = "SA3-CASSCF(2,2)/6-31G*"

WIDTH, HEIGHT = 640, 420
LEFT, RIGHT, TOP, BOTTOM = 56, 16, 44, 52
TICK_FS = 20.0
STATES = ("S0", "S1", "S2")
LEGEND = (
    ("pop-line pop-s0", "S0"),
    ("pop-line pop-s1", "S1"),
    ("pop-line pop-s2", "S2"),
    ("pop-fit", "fit to S1 + S2"),
)


def trajectory_record(output_h5: Path, *, source: str, method: str) -> dict[str, Any]:
    """The hero trajectory: positions in angstrom (3 decimals), time, active state, first hop."""
    t = read_output_h5(Path(output_h5))
    active = [int(a) for a in t["active"]]
    hops = [k for k in range(1, len(active)) if active[k] != active[k - 1]]
    if not hops:
        raise ValueError(f"{output_h5} has no hop")
    positions = np.round(np.asarray(t["positions"], dtype=float) * BOHR_IN_ANGSTROM, 3)
    return {
        "source": source,
        "method": method,
        "dt_fs": float(t.attrs["dt_fs"]),
        "symbols": [str(s) for s in t.attrs["symbols"]],
        "state_labels": list(t.labels),
        "time_fs": [round(float(x), 4) for x in t["time_fs"]],
        "active": active,
        "hop_index": hops[0],
        "positions_angstrom": positions.tolist(),
    }


def read_populations(path: Path) -> dict[str, np.ndarray]:
    """The columns of an analysis populations.csv, by header name (comment lines skipped)."""
    lines = [
        line
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    header = lines[0].split(",")
    values = np.array([[float(x) for x in line.split(",")] for line in lines[1:]], dtype=float)
    return {name: values[:, i] for i, name in enumerate(header)}


def population_svg(columns: dict[str, np.ndarray], analysis: dict[str, Any]) -> str:
    """The populations figure: bands, lines, the fitted curve, axes, a legend row on top."""
    estimator = analysis.get("estimator", "complete")
    if estimator != "complete":
        raise ValueError(
            "the figure shows the complete-trajectory populations; "
            f"this analysis used estimator = {estimator!r}"
        )
    states = analysis["fit"]["states"]
    if states != ["S1", "S2"]:
        raise ValueError(f"the legend says 'fit to S1 + S2'; this analysis fitted {states}")
    t = np.asarray(columns["time_fs"], dtype=float)
    t_max = float(t[-1])
    plot_w = WIDTH - LEFT - RIGHT
    plot_h = HEIGHT - TOP - BOTTOM

    def x(v: float) -> float:
        return LEFT + plot_w * v / t_max

    def y(p: float) -> float:
        return TOP + plot_h * (1.0 - p)

    def points(ts: np.ndarray, ps: np.ndarray) -> str:
        return " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in zip(ts, ps, strict=True))

    fit = analysis["fit"]
    n, total = int(analysis["n_included"]), int(analysis["n_selected"])
    lo, hi = fit["tau_interval_fs"]
    title = (
        f"Classical populations of S0, S1 and S2 from 0 to {t_max:g} fs for {n} of {total} "
        f"trajectories, with 95 % intervals; dashed: the fitted delayed exponential of "
        f"P(S1) + P(S2), tau {fit['tau_fs']:.0f} fs"
    )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        f'aria-labelledby="pop-title" data-n-included="{n}" data-n-selected="{total}" '
        f'data-tau-fs="{fit["tau_fs"]:.3f}" data-t0-fs="{fit["t0_fs"]:.3f}" '
        f'data-tau-interval-fs="{lo:.3f} {hi:.3f}">',
        f'<title id="pop-title">{html.escape(title)}</title>',
    ]
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        out.append(
            f'<line class="pop-grid" x1="{LEFT}" y1="{y(p):.1f}" x2="{WIDTH - RIGHT}" '
            f'y2="{y(p):.1f}"/>'
        )
        out.append(
            f'<text class="pop-label" x="{LEFT - 8}" y="{y(p) + 4:.1f}" '
            f'text-anchor="end">{p:g}</text>'
        )
    for tick in np.arange(0.0, t_max + 1e-9, TICK_FS):
        out.append(
            f'<text class="pop-label" x="{x(tick):.1f}" y="{HEIGHT - BOTTOM + 20}" '
            f'text-anchor="middle">{tick:g}</text>'
        )
    out.append(
        f'<line class="pop-axis" x1="{LEFT}" y1="{y(0.0):.1f}" x2="{WIDTH - RIGHT}" '
        f'y2="{y(0.0):.1f}"/>'
    )
    out.append(f'<line class="pop-axis" x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{y(0.0):.1f}"/>')
    out.append(
        f'<text class="pop-label" x="{LEFT + plot_w / 2:.1f}" y="{HEIGHT - 10}" '
        f'text-anchor="middle">time (fs)</text>'
    )
    out.append(
        f'<text class="pop-label" transform="translate(16 {TOP + plot_h / 2:.1f}) rotate(-90)" '
        f'text-anchor="middle">population</text>'
    )
    for s in STATES:
        hi, lo = columns[f"P_{s}_upper"], columns[f"P_{s}_lower"]
        band = points(t, hi) + " " + points(t[::-1], lo[::-1])
        out.append(f'<polygon class="pop-band pop-{s.lower()}" points="{band}"/>')
    for s in STATES:
        line = points(t, columns[f"P_{s}"])
        out.append(f'<polyline class="pop-line pop-{s.lower()}" points="{line}"/>')
    out.append(f'<polyline class="pop-fit" points="{points(t, columns["fit_curve"])}"/>')
    lx = LEFT
    for cls, label in LEGEND:
        out.append(f'<line class="{cls}" x1="{lx}" y1="18" x2="{lx + 24}" y2="18"/>')
        out.append(f'<text class="pop-label" x="{lx + 30}" y="22">{html.escape(label)}</text>')
        lx += 70 if len(label) < 4 else 130
    out.append("</svg>")
    return "\n".join(out)


def splice(html_text: str, svg: str) -> str:
    """Replace the text between BEGIN and END (each exactly once, BEGIN first)."""
    if html_text.count(BEGIN) != 1 or html_text.count(END) != 1:
        raise ValueError(f"index.html needs the markers {BEGIN} and {END} exactly once each")
    head, rest = html_text.split(BEGIN)
    _, tail = rest.split(END)
    if END in head:
        raise ValueError(f"the marker {END} comes before {BEGIN}")
    return f"{head}{BEGIN}\n{svg}\n{END}{tail}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("project", type=Path, help="a gqteashmd project directory")
    parser.add_argument("--task", required=True, help="the trajectory, e.g. traj_00033")
    parser.add_argument("--docs", type=Path, default=Path("docs"), help="the site directory")
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument(
        "--source",
        default=None,
        help="the source text in trajectory.json (default: names the ensemble and the task)",
    )
    args = parser.parse_args(argv)
    source = args.source or (
        f"gqteashmd ethylene ensemble of the SHARC comparison (USER_MANUAL 8.17), {args.task}"
    )

    output_h5 = args.project / DYNAMICS_DIR / "tasks" / args.task / "output.h5"
    record = trajectory_record(output_h5, source=source, method=args.method)
    (args.docs / "trajectory.json").write_text(
        json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8"
    )

    analysis_dir = args.project / ANALYSIS_DIR
    columns = read_populations(analysis_dir / "populations.csv")
    analysis = json.loads((analysis_dir / "analysis.json").read_text(encoding="utf-8"))
    svg = population_svg(columns, analysis)
    index = args.docs / "index.html"
    index.write_text(splice(index.read_text(encoding="utf-8"), svg), encoding="utf-8")

    hop = record["hop_index"]
    print(
        f"trajectory.json: {len(record['time_fs'])} frames, hop at {record['time_fs'][hop]:g} fs; "
        f"populations: {analysis['n_included']} of {analysis['n_selected']}, "
        f"tau {analysis['fit']['tau_fs']:.1f} fs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
