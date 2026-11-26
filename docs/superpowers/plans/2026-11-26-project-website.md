# The gqteashmd project website Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A GitHub Pages site for gqteashmd in `docs/`, in the style and structure of the gqteaWinToga site, with a real ethylene trajectory in the hero, the workflow, the real input and populations of the P9b ensemble, the validation with its markings, the known limitations, installation and credits.

**Architecture:** Static files in `docs/` (`index.html`, `style.css`, `app.js`, `icon.svg`, `trajectory.json`, `.nojekyll`), no build step and no external script. A committed script, `tools/site/make_figures.py`, turns a gqteashmd project's `output.h5`, `populations.csv` and `analysis.json` into `docs/trajectory.json` and the inline populations SVG of `index.html`. A fast-tier test, `tests/site/test_site.py`, checks links, manual anchors, the sample input, the figure data and the captions' numbers.

**Tech Stack:** HTML, CSS, plain JavaScript (no libraries), Python 3.12 with NumPy and h5py (through `gqteashmd.trajectory.store.read_output_h5`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-26-project-website-design.md` (owner-approved 2026-11-26; owner decisions in its section 0).

## Global Constraints

- Run code, tests and git inside WSL: `wsl -d OracleLinux_9_5 -- bash -lc 'cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && conda activate gqteashmd-dev && <command>'`. Git from WSL only (the pre-commit hooks, ruff 0.17.4, run there); never `--no-verify`. LF line endings.
- One commit per task, conventional subjects; every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Branch `feat/project-website` (from `main` `280a612`; the spec is commit `20da77c`).
- The site's files are `docs/index.html`, `docs/style.css`, `docs/app.js`, `docs/icon.svg`, `docs/trajectory.json`, `docs/.nojekyll`. No other file under `docs/` is created or changed by the site.
- No external script; the only outside resources are Google Fonts (STIX Two Text, Public Sans, JetBrains Mono, the same `<link>` as gqteaWinToga) and the GitHub API call of the release button (`https://api.github.com/repos/AdemirCamargo57/gqteaSHMD/releases/latest`).
- No local path, host name or user name in a committed site file or in `make_figures.py` (every path is an argument). The contact `ajc@ueg.br` and the six names are public (owner decision 4).
- Every number on the page comes from `CHANGELOG.md` 0.1.0, from the spec, or from the files `make_figures.py` reads. Do not round a number differently from the spec.
- `tools/` is never imported by `gqteashmd`. ruff (line length 100, rules E, F, W, I, B, UP, SIM) on every new Python file.
- Tests must not use the network and must not compare BLAS results bit for bit (CLAUDE.md rule).
- Data source (WSL): the P9b "ours" project `~/p9b/ours/run` (layout `06_dynamics`, `07_analysis`), trajectory `traj_00033` (finished, one hop S1→S0 at 28.0 fs, no rejection, max drift 0.069 eV). It is read only.

## Plan rulings (binding where they differ from the spec)

- **PR1 (spec 5, "no local path appears in `docs/`"):** the check covers the site's own files (the six of the Global Constraints). `docs/superpowers/` and `docs/development.md` hold the project's documents, which name WSL paths by design and are not part of the page.
- **PR2 (spec 5, the manual browser check):** no JavaScript engine is installed in WSL (`node` absent), so `app.js` is not syntax-checked by a tool. The browser check by the owner (Task 3, Step 6) is the check; the implementer keeps `app.js` to ES5-style code like gqteaWinToga's.
- **PR3 (spec 3.2):** the populations figure puts its legend in a row above the plot, so it never covers a curve.

## File structure

| File | Task | Responsibility |
|---|---|---|
| `tools/site/make_figures.py` (new) | 1 | `trajectory_record`, `read_populations`, `population_svg`, `splice`, `main` |
| `tests/site/__init__.py` (new, empty) | 1 | test package |
| `tests/site/test_make_figures.py` (new) | 1 | unit tests of the script on written files |
| `docs/index.html`, `docs/style.css`, `docs/app.js`, `docs/icon.svg`, `docs/.nojekyll` (new) | 2 | the page |
| `docs/trajectory.json` (new, generated) | 2 | the hero trajectory |
| `tests/site/test_site.py` (new) | 2 | links, anchors, sample input, figure data, captions |
| `README.md`, `USER_MANUAL.md`, `CLAUDE.md` | 3 | the documents |

---

### Task 1: The figure script

**Files:**
- Create: `tools/site/make_figures.py`
- Create: `tests/site/__init__.py` (empty)
- Test: `tests/site/test_make_figures.py`

**Interfaces:**
- Consumes: `gqteashmd.trajectory.store.read_output_h5(path) -> TrajectoryData` (`t["active"]`, `t["time_fs"]`, `t["positions"]` in bohr with shape (n, atoms, 3), `t.attrs["dt_fs"]`, `t.attrs["symbols"]`, `t.labels`); `gqteashmd.core.units.BOHR_IN_ANGSTROM`.
- Produces (Task 2 uses them):
  - `trajectory_record(output_h5: Path, *, source: str, method: str) -> dict[str, Any]` with exactly the keys `source, method, dt_fs, symbols, state_labels, time_fs, active, hop_index, positions_angstrom`;
  - `read_populations(path: Path) -> dict[str, np.ndarray]` (the CSV's columns by header name);
  - `population_svg(columns: dict[str, np.ndarray], analysis: dict[str, Any]) -> str` (one `<svg>` element with `data-n-included`, `data-n-selected`, `data-tau-fs` (3 decimals), `data-t0-fs` (3 decimals));
  - `splice(html: str, svg: str) -> str` (replaces the text between `BEGIN = "<!-- populations-svg:begin -->"` and `END = "<!-- populations-svg:end -->"`);
  - `main(argv: list[str] | None = None) -> int`: `make_figures.py PROJECT --task TASK [--docs DIR] [--method TEXT] [--source TEXT]`, writes `DIR/trajectory.json` and splices the SVG into `DIR/index.html`.

- [ ] **Step 1: Write the failing tests**

`tests/site/__init__.py` is an empty file. `tests/site/test_make_figures.py`:

```python
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


ANALYSIS = {"n_included": 67, "n_selected": 100, "fit": {"tau_fs": 80.13602924, "t0_fs": 12.8353}}


def test_trajectory_record(tmp_path):
    rec = mf.trajectory_record(
        _output_h5(tmp_path / "output.h5", [1, 1, 0, 0]), source="src", method="m"
    )
    assert set(rec) == {
        "source", "method", "dt_fs", "symbols", "state_labels", "time_fs", "active",
        "hop_index", "positions_angstrom",
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
    assert "67 of 100 trajectories" in svg
    for state in ("s0", "s1", "s2"):
        assert f'class="pop-band pop-{state}"' in svg
        assert f'class="pop-line pop-{state}"' in svg
    assert 'class="pop-fit"' in svg
    assert 'role="img"' in svg and '<title id="pop-title">' in svg


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/site/test_make_figures.py -q -W error`
Expected: collection error, `ModuleNotFoundError: No module named 'make_figures'`.

- [ ] **Step 3: Write the script**

`tools/site/make_figures.py`:

```python
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
    title = (
        f"Classical populations of S0, S1 and S2 from 0 to {t_max:g} fs for {n} of {total} "
        f"trajectories, with 95 % intervals; dashed: the fitted delayed exponential of "
        f"P(S1) + P(S2), tau {fit['tau_fs']:.0f} fs"
    )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        f'aria-labelledby="pop-title" data-n-included="{n}" data-n-selected="{total}" '
        f'data-tau-fs="{fit["tau_fs"]:.3f}" data-t0-fs="{fit["t0_fs"]:.3f}">',
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/site/test_make_figures.py -q -W error && ruff format tools/site tests/site && ruff check tools/site tests/site`
Expected: 7 passed; ruff clean (`ruff format` may rewrap the long `set(rec) ==` literal; keep its result).

- [ ] **Step 5: Commit**

```bash
git add tools/site/make_figures.py tests/site/__init__.py tests/site/test_make_figures.py
git commit -m "feat: add the figure script of the project website"
```
(with the `Co-Authored-By` trailer of the Global Constraints)

---

### Task 2: The page

**Files:**
- Create: `docs/index.html`, `docs/style.css`, `docs/app.js`, `docs/icon.svg`, `docs/.nojekyll`
- Create (generated): `docs/trajectory.json`
- Test: `tests/site/test_site.py`

**Interfaces:**
- Consumes: `tools/site/make_figures.py` of Task 1 (`BEGIN`, `END`, `main`); `trajectory.json` keys of Task 1; the SVG `data-*` attributes and classes of Task 1 (`pop-grid`, `pop-axis`, `pop-label`, `pop-band`, `pop-line`, `pop-s0`/`pop-s1`/`pop-s2`, `pop-fit`).
- Produces: the published site; `tests/site/test_site.py` (`github_slugs(markdown: str) -> set[str]`).

- [ ] **Step 1: Write the failing test**

`tests/site/test_site.py`:

```python
"""The project website in docs/: links, manual anchors, the sample input, the figure data."""

import html
import json
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path

import pytest

from gqteashmd.cli import main as cli_main

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
INDEX = DOCS / "index.html"
SAMPLE_INPUT = ROOT / "tools" / "sharc_compare" / "p9b" / "ethylene_seed2028.toml"
SITE_FILES = ("index.html", "style.css", "app.js", "icon.svg", "trajectory.json", ".nojekyll")
BEGIN = "<!-- populations-svg:begin -->"
END = "<!-- populations-svg:end -->"
FONT_HOSTS = ("https://fonts.googleapis.com", "https://fonts.gstatic.com")


class _Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str, str]] = []  # (tag, attribute, value)
        self.ids: set[str] = set()
        self.scripts: list[str | None] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        for name in ("href", "src"):
            if a.get(name):
                self.links.append((tag, name, a[name]))
        if tag == "script":
            self.scripts.append(a.get("src"))


@pytest.fixture(scope="module")
def page() -> _Page:
    p = _Page()
    p.feed(INDEX.read_text(encoding="utf-8"))
    return p


def github_slugs(markdown: str) -> set[str]:
    """GitHub's heading anchors: lower case, punctuation dropped, spaces to hyphens, -n repeats."""
    slugs: set[str] = set()
    seen: dict[str, int] = {}
    in_code = False
    for line in markdown.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            continue
        m = None if in_code else re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not m:
            continue
        slug = re.sub(r"[^\w\- ]", "", m.group(1).strip().lower()).replace(" ", "-")
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        slugs.add(slug if n == 0 else f"{slug}-{n}")
    return slugs


def test_the_slug_rule():
    md = "## 5.3 `gqteashmd single-point`\n```\n## not a heading\n```\n## A\n## A\n"
    assert github_slugs(md) == {"53-gqteashmd-single-point", "a", "a-1"}


def test_every_site_file_exists():
    for name in SITE_FILES:
        assert (DOCS / name).is_file(), name


def test_local_links_exist(page):
    for _tag, _attr, value in page.links:
        if value.startswith(("http://", "https://", "mailto:", "#")):
            continue
        assert (DOCS / value.split("#")[0]).is_file(), value


def test_fragment_links_have_targets(page):
    for _tag, _attr, value in page.links:
        if value.startswith("#"):
            assert value[1:] in page.ids, value


def test_manual_anchors_exist(page):
    slugs = github_slugs((ROOT / "USER_MANUAL.md").read_text(encoding="utf-8"))
    anchors = [
        v.split("#", 1)[1] for _t, _a, v in page.links if re.search(r"USER_MANUAL\.md#", v)
    ]
    assert anchors, "the page links to no manual section"
    missing = [a for a in anchors if a not in slugs]
    assert not missing, missing


def test_only_app_js_and_the_fonts_are_loaded(page):
    assert page.scripts == ["app.js"]
    for tag, attr, value in page.links:
        if tag == "link" and value.startswith("http"):
            assert value.startswith(FONT_HOSTS), value
        if tag in ("script", "img") and attr == "src":
            assert not value.startswith("http"), value


def _sample_input_text() -> str:
    text = INDEX.read_text(encoding="utf-8")
    m = re.search(r'<figure class="feature-figure input-sample">\s*<pre><code>(.*?)</code></pre>',
                  text, re.S)
    assert m, "no sample input figure"
    return html.unescape(m.group(1))


def test_the_sample_input_is_the_ensemble_input():
    assert _sample_input_text() == SAMPLE_INPUT.read_text(encoding="utf-8")


def test_the_sample_input_validates(tmp_path):
    shutil.copy(SAMPLE_INPUT, tmp_path / SAMPLE_INPUT.name)
    shutil.copy(ROOT / "tests" / "data" / "ethylene.xyz", tmp_path / "ethylene.xyz")
    assert cli_main(["validate", str(tmp_path / SAMPLE_INPUT.name)]) == 0


def test_the_trajectory_data():
    rec = json.loads((DOCS / "trajectory.json").read_text(encoding="utf-8"))
    assert set(rec) == {
        "source", "method", "dt_fs", "symbols", "state_labels", "time_fs", "active",
        "hop_index", "positions_angstrom",
    }
    n, atoms = len(rec["time_fs"]), len(rec["symbols"])
    assert len(rec["active"]) == n
    assert [len(f) for f in rec["positions_angstrom"]] == [atoms] * n
    assert all(len(xyz) == 3 for f in rec["positions_angstrom"] for xyz in f)
    h = rec["hop_index"]
    assert 0 < h < n and rec["active"][h] != rec["active"][h - 1]
    assert rec["time_fs"] == sorted(rec["time_fs"])


def test_the_hero_caption_names_the_hop_time():
    rec = json.loads((DOCS / "trajectory.json").read_text(encoding="utf-8"))
    hop_fs = rec["time_fs"][rec["hop_index"]]
    m = re.search(r"hops to S0 at (\d+) fs", INDEX.read_text(encoding="utf-8"))
    assert m and int(m.group(1)) == round(hop_fs)


def test_the_population_figure_and_its_caption():
    text = INDEX.read_text(encoding="utf-8")
    block = text.split(BEGIN)[1].split(END)[0]
    assert block.count("<svg ") == 1
    n = int(re.search(r'data-n-included="(\d+)"', block).group(1))
    total = int(re.search(r'data-n-selected="(\d+)"', block).group(1))
    tau = float(re.search(r'data-tau-fs="([\d.]+)"', block).group(1))
    assert f"Populations of the {n} of {total} trajectories" in text
    assert f"tau {round(tau)} fs" in text


def test_no_local_path_in_the_site_files():
    for name in SITE_FILES:
        text = (DOCS / name).read_text(encoding="utf-8")
        # The page names "/mnt/c" and "a OneDrive folder" as places to avoid; a real path has a
        # separator after them.
        for pattern in (r"/home/\w", r"/mnt/[a-z]/", r"[A-Za-z]:\\", r"OneDrive[/\\]"):
            assert not re.search(pattern, text), (name, pattern)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/site/test_site.py -q -W error`
Expected: `test_the_slug_rule` passes; every other test fails or errors (`docs/index.html` does not exist).

- [ ] **Step 3: The stylesheet, the icon, `.nojekyll`**

1. Copy gqteaWinToga's stylesheet verbatim and check it is the reviewed version:
   ```bash
   curl -sL https://ademircamargo57.github.io/gqteaWinToga/style.css -o docs/style.css
   sha256sum docs/style.css   # 1f0d2fa3ca23c2d1ef62e93f63e5a22208d81f2fc06041889463a49e6b682edd
   ```
   If the hash differs, the sister site changed: keep the downloaded file only if `diff` against the hash's version is understood, and report it (the controller rules on it).
2. Replace its comment line `/* Tabs: echo the program's own main window */` by `/* Tabs: the stages of the workflow */` and `/* Feature spotlights (ESP, Fukui) */` by `/* Feature spotlights */`, `/* Sample input file shown as a figure (Gaussian 16 builder) */` by `/* Sample input file shown as a figure */`, `/* 3D viewer tabs as cards */` by `/* Cards */`. Change nothing else in the copied part.
3. Append:

```css

/* Hero trajectory player */
.player {
  display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 6px 12px; align-items: center;
  font-family: var(--sans); font-size: 0.9rem; margin-top: 6px;
}
.player[hidden] { display: none; }
.player-toggle {
  font: inherit; font-weight: 600; color: var(--on-honey); background: var(--honey);
  border: 0; border-radius: 8px; padding: 6px 14px; cursor: pointer; min-width: 5.5em;
}
.player-toggle:hover { background: var(--honey-hover); }
.player-track { position: relative; padding-top: 18px; }
.player-track input { display: block; width: 100%; margin: 0; accent-color: var(--link); }
.player-hop {
  position: absolute; top: 0; transform: translateX(-50%);
  font-size: 0.75rem; color: var(--muted); white-space: nowrap; pointer-events: none;
}
.player-hop::after {
  content: ""; display: block; width: 2px; height: 7px; margin: 1px auto 0; background: var(--honey);
}
.player-readout {
  grid-column: 1 / -1; text-align: center; color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.player-readout[data-state="S1"] { color: var(--link); font-weight: 600; }
.player-readout[data-state="S0"] { color: var(--ink); font-weight: 600; }
.player-fallback { font-family: var(--sans); font-size: 0.9rem; color: var(--muted); text-align: center; }

/* Populations figure (inline SVG written by tools/site/make_figures.py) */
.pop-figure svg {
  display: block; width: 100%; height: auto;
  background: var(--surface); border: 1px solid var(--rule); border-radius: 10px;
}
.pop-grid { stroke: var(--rule); stroke-width: 1; }
.pop-axis { stroke: var(--muted); stroke-width: 1; }
.pop-label { fill: var(--muted); font-family: var(--sans); font-size: 13px; }
.pop-band { stroke: none; opacity: 0.2; }
.pop-line { fill: none; stroke-width: 2.5; }
.pop-band.pop-s0 { fill: var(--ink); }
.pop-band.pop-s1 { fill: var(--link); }
.pop-band.pop-s2 { fill: var(--honey); }
.pop-line.pop-s0 { stroke: var(--ink); }
.pop-line.pop-s1 { stroke: var(--link); }
.pop-line.pop-s2 { stroke: var(--honey); }
.pop-fit { fill: none; stroke: var(--muted); stroke-width: 2; stroke-dasharray: 6 5; }

/* Result badges (How it was checked) */
.badge {
  display: inline-block; padding: 1px 9px; border-radius: 999px;
  font-family: var(--sans); font-size: 0.75rem; font-weight: 600;
  vertical-align: 0.1em; white-space: nowrap;
}
.badge-test { background: var(--honey); color: var(--on-honey); }
.badge-pre { border: 1px solid var(--link); color: var(--link); }
.badge-report { border: 1px solid var(--muted); color: var(--muted); }
.badge-key {
  list-style: none; margin: 0 0 24px; padding: 0; max-width: 80ch;
  display: grid; gap: 8px; font-family: var(--sans); font-size: 0.95rem;
}
.result-list {
  list-style: none; margin: 0 0 12px; padding: 0;
  font-family: var(--sans); font-size: 0.95rem; line-height: 1.5;
}
.result-list li { padding: 7px 0; border-top: 1px solid var(--rule); }

/* Known limitations */
.limits {
  margin-top: 32px; background: var(--surface); border: 1px solid var(--rule);
  border-left: 4px solid var(--honey); border-radius: 12px; padding: 22px 24px;
}
.limits h3 { margin-bottom: 10px; }
.limits ul { margin: 0 0 10px; padding-left: 1.1em; font-family: var(--sans); font-size: 1rem; }
.limits li { margin-bottom: 8px; max-width: 80ch; }
.limits p { margin: 0; font-family: var(--sans); font-size: 0.95rem; }
```

4. `docs/icon.svg` (an ethylene glyph in the site's colours):

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64">
  <rect width="64" height="64" rx="14" fill="#1a2b3f"/>
  <g stroke="#e6ecf1" stroke-width="3" stroke-linecap="round">
    <line x1="24" y1="29" x2="40" y2="29"/>
    <line x1="24" y1="35" x2="40" y2="35"/>
    <line x1="22" y1="32" x2="12" y2="20"/>
    <line x1="22" y1="32" x2="12" y2="44"/>
    <line x1="42" y1="32" x2="52" y2="20"/>
    <line x1="42" y1="32" x2="52" y2="44"/>
  </g>
  <circle cx="22" cy="32" r="7" fill="#f2b21b"/>
  <circle cx="42" cy="32" r="7" fill="#f2b21b"/>
  <g fill="#f4f6f7">
    <circle cx="12" cy="20" r="4.5"/>
    <circle cx="12" cy="44" r="4.5"/>
    <circle cx="52" cy="20" r="4.5"/>
    <circle cx="52" cy="44" r="4.5"/>
  </g>
</svg>
```

5. `docs/.nojekyll`: an empty file (`: > docs/.nojekyll`). `test_no_local_path_in_the_site_files` reads it as empty text.

- [ ] **Step 4: The script, `docs/app.js`**

```js
/* gqteashmd site: latest-release button, tabs, and the hero trajectory player. */

/* ---------- Latest release: point the button at the wheel ---------- */
(function () {
  var REPO = "AdemirCamargo57/gqteaSHMD";
  var btn = document.getElementById("download-btn");
  var label = document.getElementById("download-label");
  var meta = document.getElementById("download-meta");
  if (!btn || !window.fetch) return;

  function sizeKB(bytes) { return (bytes / 1024).toFixed(0) + " kB"; }
  function niceDate(iso) {
    try {
      return new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
    } catch (e) { return ""; }
  }

  fetch("https://api.github.com/repos/" + REPO + "/releases/latest", {
    headers: { Accept: "application/vnd.github+json" }
  })
    .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(function (rel) {
      var assets = rel.assets || [];
      var wheel = assets.find(function (a) { return /\.whl$/i.test(a.name); });
      if (!wheel) return;
      var version = (rel.tag_name || "").replace(/^v/i, "");
      btn.href = rel.html_url || btn.href;
      label.textContent = "Get gqteashmd " + version;
      meta.textContent = "Linux only. Wheel " + sizeKB(wheel.size) + " and the lock file " +
        "environment.lock.txt, released " + niceDate(rel.published_at) +
        ". Free and open source, MIT license.";
    })
    .catch(function () { /* keep the fallback link to the latest release */ });
})();

/* ---------- Tabs (WAI-ARIA tabs pattern) ---------- */
(function () {
  var root = document.querySelector("[data-tabs]");
  if (!root) return;
  var tabs = Array.prototype.slice.call(root.querySelectorAll('[role="tab"]'));

  function select(tab, focus) {
    tabs.forEach(function (t) {
      var on = t === tab;
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
    });
    if (focus) tab.focus();
  }

  tabs.forEach(function (tab, i) {
    tab.addEventListener("click", function () { select(tab, false); });
    tab.addEventListener("keydown", function (e) {
      var j = null;
      if (e.key === "ArrowRight") j = (i + 1) % tabs.length;
      else if (e.key === "ArrowLeft") j = (i - 1 + tabs.length) % tabs.length;
      else if (e.key === "Home") j = 0;
      else if (e.key === "End") j = tabs.length - 1;
      if (j !== null) { e.preventDefault(); select(tabs[j], true); }
    });
  });
})();

/* ---------- Hero trajectory: ethylene along one gqteashmd trajectory ---------- */
(function () {
  var canvas = document.getElementById("trajectory");
  var panel = document.getElementById("player");
  var fallback = document.getElementById("trajectory-fallback");
  var slider = document.getElementById("player-time");
  var button = document.getElementById("player-toggle");
  var readout = document.getElementById("player-readout");
  var marker = document.getElementById("player-hop");
  if (!canvas || !canvas.getContext || !window.fetch || !panel) return;
  var ctx = canvas.getContext("2d");

  var style = {
    C: { r: 0.42, c: "#5d6873" },
    H: { r: 0.26, c: "#f4f6f7" }
  };
  var cov = { C: 0.76, H: 0.31 };
  var FS_PER_SECOND = 10;  // playback speed: about 10 s per 100 fs
  var CAM = 30;            // camera distance for a gentle perspective
  var W = 560, H = 520;
  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = W * dpr; canvas.height = H * dpr;

  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var data = null, frame = 0, clock = 0, tEnd = 1, maxR = 1;
  var playing = !reduce, visible = true, last = 0;
  var rotY = -0.5, rotX = 0.35, dragging = false, lastX = 0, lastY = 0;

  function isDark() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t) return t === "dark";
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function centred(k) {
    var f = data.positions_angstrom[k], n = f.length, cx = 0, cy = 0, cz = 0;
    f.forEach(function (a) { cx += a[0]; cy += a[1]; cz += a[2]; });
    cx /= n; cy /= n; cz /= n;
    return f.map(function (a, i) {
      return { el: data.symbols[i], x: a[0] - cx, y: a[1] - cy, z: a[2] - cz };
    });
  }

  function project(P) {
    var cy_ = Math.cos(rotY), sy = Math.sin(rotY), cx_ = Math.cos(rotX), sx = Math.sin(rotX);
    return P.map(function (p) {
      var x = p.x * cy_ + p.z * sy;
      var z = -p.x * sy + p.z * cy_;
      var y = p.y * cx_ - z * sx;
      z = p.y * sx + z * cx_;
      var persp = CAM / (CAM - z);
      return { x: x * persp, y: y * persp, z: z, s: persp, el: p.el };
    });
  }

  function shade(hex, f) {
    var n = parseInt(hex.slice(1), 16);
    var r = Math.round(((n >> 16) & 255) * f), g = Math.round(((n >> 8) & 255) * f), b = Math.round((n & 255) * f);
    return "rgb(" + r + "," + g + "," + b + ")";
  }

  function draw() {
    if (!data) return;
    var dark = isDark();
    var P = centred(frame);
    var Q = project(P);
    var scale = (Math.min(W, H) / 2) / (maxR * CAM / (CAM - maxR));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.save();
    ctx.translate(W / 2, H / 2);

    // covalent bonds by distance, recomputed every frame (bonds may stretch)
    var items = [];
    for (var i = 0; i < P.length; i++) for (var j = i + 1; j < P.length; j++) {
      var dx = P[i].x - P[j].x, dy = P[i].y - P[j].y, dz = P[i].z - P[j].z;
      if (Math.sqrt(dx * dx + dy * dy + dz * dz) < 1.2 * (cov[P[i].el] + cov[P[j].el])) {
        items.push({ t: "b", z: (Q[i].z + Q[j].z) / 2 - 0.3, a: Q[i], c: Q[j] });
      }
    }
    Q.forEach(function (q) { items.push({ t: "a", z: q.z, q: q }); });
    items.sort(function (m, n) { return m.z - n.z; });

    items.forEach(function (it) {
      if (it.t === "b") {
        var w = 0.17 * scale * (it.a.s + it.c.s) / 2;
        ctx.lineCap = "round";
        ctx.lineWidth = w + 2.5;
        ctx.strokeStyle = dark ? "#0b1520" : "#3a4652";
        ctx.beginPath();
        ctx.moveTo(it.a.x * scale, -it.a.y * scale);
        ctx.lineTo(it.c.x * scale, -it.c.y * scale);
        ctx.stroke();
        ctx.lineWidth = w;
        ctx.strokeStyle = dark ? "#8796a6" : "#aab5bf";
        ctx.stroke();
      } else {
        var q = it.q, st = style[q.el] || style.C;
        var r = st.r * scale * q.s;
        var x = q.x * scale, y = -q.y * scale;
        var g = ctx.createRadialGradient(x - r * 0.35, y - r * 0.4, r * 0.1, x, y, r);
        g.addColorStop(0, "#ffffff");
        g.addColorStop(0.28, st.c);
        g.addColorStop(1, shade(st.c, 0.55));
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = g;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = dark ? "#0b1520" : "#33404d";
        ctx.stroke();
      }
    });
    ctx.restore();
  }

  function frameAt(t) {
    var k = 0;
    while (k + 1 < data.time_fs.length && data.time_fs[k + 1] <= t) k++;
    return k;
  }

  function show(k) {
    frame = k;
    slider.value = String(k);
    var labels = data.state_labels, h = data.hop_index;
    var state = labels[data.active[k]];
    var text = state;
    if (k === h) text = labels[data.active[h - 1]] + " → " + state;
    readout.textContent = "t = " + data.time_fs[k].toFixed(1) + " fs · " + text;
    readout.setAttribute("data-state", state);
    draw();
  }

  function setPlaying(on) {
    playing = on;
    button.textContent = on ? "Pause" : "Play";
    button.setAttribute("aria-label", on ? "Pause the trajectory" : "Play the trajectory");
    if (on) { last = 0; requestAnimationFrame(tick); }
  }

  function tick(t) {
    if (!playing || !visible) { last = 0; return; }
    var dt = last ? Math.min(t - last, 50) : 16;
    last = t;
    clock += dt * FS_PER_SECOND / 1000;
    if (clock > tEnd) clock = 0;
    var k = frameAt(clock);
    if (k !== frame) show(k); else if (dragging) draw();
    requestAnimationFrame(tick);
  }

  function setup(d) {
    data = d;
    tEnd = data.time_fs[data.time_fs.length - 1];
    for (var k = 0; k < data.positions_angstrom.length; k++) {
      centred(k).forEach(function (p) {
        maxR = Math.max(maxR, Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z) + 0.45);
      });
    }
    slider.max = String(data.time_fs.length - 1);
    marker.style.left = (100 * data.time_fs[data.hop_index] / tEnd) + "%";
    if (fallback) fallback.hidden = true;
    panel.hidden = false;
    show(0);
    setPlaying(playing);
  }

  fetch("trajectory.json")
    .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(setup)
    .catch(function () { /* the fallback text stays */ });

  slider.addEventListener("input", function () {
    if (playing) setPlaying(false);
    var k = parseInt(slider.value, 10);
    clock = data.time_fs[k];
    show(k);
  });
  button.addEventListener("click", function () { setPlaying(!playing); });

  if ("IntersectionObserver" in window) {
    new IntersectionObserver(function (entries) {
      var was = visible;
      visible = entries[0].isIntersecting;
      if (visible && !was && playing) { last = 0; requestAnimationFrame(tick); }
    }).observe(canvas);
  }

  canvas.addEventListener("pointerdown", function (e) {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", function (e) {
    if (!dragging) return;
    rotY += (e.clientX - lastX) * 0.01;
    rotX += (e.clientY - lastY) * 0.01;
    rotX = Math.max(-1.4, Math.min(1.4, rotX));
    lastX = e.clientX; lastY = e.clientY;
    draw();
  });
  function end() { dragging = false; }
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);

  if (window.matchMedia) {
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    if (mq.addEventListener) mq.addEventListener("change", draw);
  }
})();
```

- [ ] **Step 5: The page, `docs/index.html`**

Write exactly this file. The sample input inside `<pre><code>` is `tools/sharc_compare/p9b/ethylene_seed2028.toml` verbatim (it holds no `<`, `>` or `&`); the populations SVG is filled in Step 6.

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>gqteashmd</title>
<meta name="description" content="gqteashmd is a free program for surface-hopping molecular dynamics with PySCF on Linux: from one input file it optimizes a molecule, computes its frequencies, samples initial conditions, computes the excitations, runs the trajectories and analyses the populations and lifetimes, in one resumable run.">
<link rel="icon" type="image/svg+xml" href="icon.svg">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=STIX+Two+Text:ital,wght@0,400;0,600;0,700;1,400&family=Public+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
<link rel="stylesheet" href="style.css">
</head>
<body>

<header class="site-header">
  <div class="wrap header-row">
    <a class="brand" href="#top" aria-label="gqteashmd home">
      <img src="icon.svg" alt="" width="36" height="36">
      <span>gqteashmd</span>
    </a>
    <nav aria-label="Main">
      <a href="#workflow">Workflow</a>
      <a href="#new">What's new</a>
      <a href="#checked">Validation</a>
      <a href="#install">Install</a>
      <a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md">Manual</a>
      <a href="https://github.com/AdemirCamargo57/gqteaSHMD">Source code</a>
    </nav>
  </div>
</header>

<main id="top">

  <section class="hero">
    <div class="wrap hero-grid">
      <div class="hero-text">
        <h1>gqteashmd</h1>
        <p class="lede">Automated surface-hopping molecular dynamics with PySCF.</p>
        <p>Write one TOML input file and run one command. gqteashmd optimizes the molecule, computes its vibrational frequencies, samples initial geometries and velocities from the Wigner distribution, computes the vertical excitations, selects the initial excited states, runs the surface-hopping trajectories and turns them into state populations and an excited-state lifetime. The whole run resumes where it stopped after a crash, a reboot or a deliberate stop.</p>

        <div class="download">
          <a id="download-btn" class="btn-download" href="https://github.com/AdemirCamargo57/gqteaSHMD/releases/latest">
            <svg aria-hidden="true" viewBox="0 0 24 24" width="22" height="22"><path d="M12 3v12m0 0-5-5m5 5 5-5M4 20h16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
            <span id="download-label">Get gqteashmd 0.1.0</span>
          </a>
          <p class="download-meta" id="download-meta">Linux only. Free and open source, MIT license.</p>
          <p class="download-links"><a href="https://github.com/AdemirCamargo57/gqteaSHMD/releases">Release notes and earlier versions</a></p>
        </div>
      </div>

      <figure class="hero-figure">
        <canvas id="trajectory" width="560" height="520" role="img" aria-label="Ball-and-stick model of ethylene moving along one surface-hopping trajectory: the carbon-carbon bond twists and the molecule hops from the S1 to the S0 state at 28 femtoseconds"></canvas>
        <p id="trajectory-fallback" class="player-fallback">The trajectory player needs JavaScript and the file <code>trajectory.json</code>.</p>
        <div id="player" class="player" hidden>
          <button type="button" id="player-toggle" class="player-toggle">Pause</button>
          <div class="player-track">
            <input type="range" id="player-time" min="0" max="200" step="1" value="0" aria-label="Time along the trajectory">
            <span id="player-hop" class="player-hop" aria-hidden="true">hop</span>
          </div>
          <output id="player-readout" class="player-readout" for="player-time">t = 0.0 fs · S1</output>
        </div>
        <figcaption>Ethylene after excitation to S1: one trajectory of a 100-trajectory gqteashmd ensemble (SA3-CASSCF(2,2)/6-31G*, 0.5 fs steps). The C=C bond twists and the molecule hops to S0 at 28 fs. Drag to rotate.</figcaption>
      </figure>
    </div>
  </section>

  <section id="workflow" class="section">
    <div class="wrap">
      <h2>What it does</h2>
      <p class="section-intro">One input file drives eight stages, from the geometry to the lifetime. Each stage is split into tasks that run in parallel, and every finished task is kept, so a run that stops resumes where it stopped.</p>

      <div class="tabs" data-tabs>
        <div class="tab-list" role="tablist" aria-label="Workflow stages">
          <button role="tab" id="tab-control" aria-controls="panel-control" aria-selected="true">Input &amp; control</button>
          <button role="tab" id="tab-ground" aria-controls="panel-ground" aria-selected="false" tabindex="-1">Ground state</button>
          <button role="tab" id="tab-initial" aria-controls="panel-initial" aria-selected="false" tabindex="-1">Initial conditions</button>
          <button role="tab" id="tab-dynamics" aria-controls="panel-dynamics" aria-selected="false" tabindex="-1">Dynamics</button>
          <button role="tab" id="tab-analysis" aria-controls="panel-analysis" aria-selected="false" tabindex="-1">Analysis</button>
        </div>

        <div role="tabpanel" id="panel-control" aria-labelledby="tab-control" class="tab-panel">
          <p class="panel-lead">A strict input file, checks before anything runs, and a run that can be stopped, resumed and changed safely.</p>
          <ul class="tool-list">
            <li>One strict TOML input file; units in the key names (<code>dt_fs</code>, <code>temperature_K</code>); <code>gqteashmd validate</code> lists every problem at once</li>
            <li>Checks before a run: the project location, free disk space, the required Python packages and the hardware you declared</li>
            <li>Parallel tasks on one workstation, with cores, memory and threads set per stage</li>
            <li>Crash-safe resume: a killed or rebooted run continues from its finished tasks and the last restart of each trajectory</li>
            <li><code>gqteashmd stop</code> and <code>gqteashmd status</code>; pauses for review after chosen stages (<code>pause_after</code>)</li>
            <li>Change detection: a changed input never mixes with old results; <code>--rerun-from STAGE</code> says what to recompute</li>
            <li>A provenance record of every run (versions, packages, machine) and, since 0.1.0, the commit of the installed build</li>
          </ul>
        </div>

        <div role="tabpanel" id="panel-ground" aria-labelledby="tab-ground" class="tab-panel" hidden>
          <p class="panel-lead">The molecule's ground state at the level of theory of the dynamics.</p>
          <ul class="tool-list">
            <li><code>prepare</code>: checks and stores the molecule, its charge and multiplicity</li>
            <li><code>optimize</code> (optional): the S0 geometry with geomeTRIC</li>
            <li><code>frequencies</code>: the S0 Hessian, analytic for SCF methods or by finite differences for SA-CASSCF; normal modes and a Molden file</li>
            <li>Electronic structure from PySCF: SCF, state-averaged CASSCF, TDA, TDDFT, CIS and TDHF; energies, analytic gradients and transition dipoles</li>
            <li>A fixed start for SA-CASSCF: every fresh solve begins from a reference solution at the stage's reference geometry</li>
          </ul>
        </div>

        <div role="tabpanel" id="panel-initial" aria-labelledby="tab-initial" class="tab-panel" hidden>
          <p class="panel-lead">Initial geometries, velocities and excited states for the trajectories.</p>
          <ul class="tool-list">
            <li><code>sampling</code>: harmonic Wigner sampling with exact thermal widths; linear and angular momentum removed; an exact energy check per mode</li>
            <li><code>excitations</code>: vertical excitation energies and oscillator strengths at every sample, and the absorption spectrum as a CSV file</li>
            <li><code>selection</code>: the initial state of every trajectory, by bright-state selection (weight f/dE&sup2; or f) in an energy window, or a fixed state</li>
            <li>Reproducible: every random number derives from <code>project.seed</code>, so the same input gives the same samples, also after a restart</li>
          </ul>
        </div>

        <div role="tabpanel" id="panel-dynamics" aria-labelledby="tab-dynamics" class="tab-panel" hidden>
          <p class="panel-lead">One surface-hopping trajectory per selected initial condition, each a resumable task.</p>
          <ul class="tool-list">
            <li>SHARC-type or Tully (FSSH) hopping, with velocity adjustment and the energy-based decoherence correction</li>
            <li>Couplings from wavefunction overlaps (local diabatization, every method) or SA-CASSCF nonadiabatic couplings</li>
            <li>Step rejection: an unreliable step (a failed calculation, a small overlap, an energy jump, a changed CASSCF start) is redone with halved steps</li>
            <li>A restart after every accepted step; an outcome for every trajectory (finished, energy drift, overlap or electronic failure, electronic discontinuity)</li>
            <li>Raise <code>tmax_fs</code> and run again to extend the finished trajectories</li>
            <li>An optional hop energy rule that keeps the previous step's total energy, as SHARC does (<code>hop_energy = "previous"</code>)</li>
          </ul>
        </div>

        <div role="tabpanel" id="panel-analysis" aria-labelledby="tab-analysis" class="tab-panel" hidden>
          <p class="panel-lead">From trajectories to populations and a lifetime, with uncertainties.</p>
          <ul class="tool-list">
            <li>Classical and quantum state populations with 95 % intervals</li>
            <li>The censored estimator by default: a trajectory that ended early still counts up to its end (Aalen-Johansen), and the complete-trajectory rule is reported beside it</li>
            <li>A delayed mono-exponential fit (delay t0 and lifetime tau) with bootstrap intervals</li>
            <li>Hop statistics, first arrival in S0, energy drift and energy-jump diagnostics</li>
            <li>A warning below 50 trajectories, the measured size for reliable lifetime intervals</li>
            <li>Results as CSV, HDF5, JSON and a Markdown report</li>
          </ul>
        </div>
      </div>
    </div>
  </section>

  <section id="new" class="section section-alt">
    <div class="wrap feature-grid">
      <div class="feature-text">
        <p class="feature-tag">First release, 0.1.0</p>
        <h2>One input file, one command</h2>
        <p>The input on the right is the real input of the 100-trajectory ethylene ensemble shown on this page. Everything not written in it takes its documented default. <code>gqteashmd run input.toml</code> then runs every stage to the end.</p>
        <ul class="feature-list">
          <li><b>Survives crashes.</b> Kill the run, reboot the machine, run the same command again: finished tasks are kept and every trajectory continues from its last restart.</li>
          <li><b>Stops when asked.</b> <code>gqteashmd stop</code> ends a run gracefully, and <code>pause_after</code> stops it after a stage you want to check first.</li>
          <li><b>Never mixes old and new.</b> Each stage records a fingerprint of its inputs. A changed input stops the run and names the <code>--rerun-from</code> stage that recomputes what it affects.</li>
          <li><b>Records what ran.</b> Every run appends the program version, the commit of the installed build, the Python packages and the machine to <code>provenance.json</code>.</li>
        </ul>
        <p><a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md#4-the-input-file">The input file</a> &middot; <a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md#5-commands">The commands</a> &middot; <a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md#73-resuming-after-a-crash-reboot-or-kill">Resuming</a></p>
      </div>

      <figure class="feature-figure input-sample">
<pre><code>[project]
name = "ethylene_p9b"
directory = "run"
seed = 2028

[molecule]
geometry = "ethylene.xyz"
charge = 0
multiplicity = 1

[resources]
total_cores = 16
total_memory_mb = 7000
max_parallel_tasks = 8
threads_per_task = 2
memory_mb_per_task = 800

[electronic]
method = "casscf"
basis = "6-31g*"
singlets = 3

[electronic.casscf]
ncas = 2
nelecas = 2

[sampling]
n_samples = 300

[selection]
n_trajectories = 100
states = ["S1"]

[dynamics]
tmax_fs = 100.0
dt_fs = 0.5
</code></pre>
        <figcaption>The input of the ethylene ensemble: SA3-CASSCF(2,2)/6-31G*, 300 Wigner samples, 100 trajectories started in S1, 100 fs in steps of 0.5 fs, run on a 16-core workstation in 8 parallel tasks.</figcaption>
      </figure>
    </div>
  </section>

  <section id="dynamics" class="section">
    <div class="wrap feature-grid">
      <div class="feature-text">
        <p class="feature-tag"><code>dynamics</code> and <code>analysis</code> stages</p>
        <h2>Dynamics that run to the end, or say why they stopped</h2>
        <p>Every trajectory is its own task with its own restart. At every step the program checks the new electronic structure before it accepts the step, and when a trajectory cannot go on it ends with an outcome that says why, instead of producing numbers nobody can trust.</p>
        <ul class="feature-list">
          <li><b>Overlaps every step.</b> Wavefunction overlaps between consecutive steps carry the state couplings and track the states through crossings.</li>
          <li><b>Rejected, then halved.</b> A step with a failed calculation, a small overlap, an energy jump or a changed CASSCF start is redone as two half steps, down to an eighth of the step.</li>
          <li><b>Honest outcomes.</b> Finished, energy drift, overlap failure, electronic failure or electronic discontinuity, each with its time and its message.</li>
          <li><b>Lost trajectories still count.</b> The default censored estimator uses an early-ended trajectory up to its end; the complete-trajectory rule is reported beside it.</li>
        </ul>
        <p><a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md#816-the-dynamics-stage-with-pyscf-p7b">The dynamics stage</a> &middot; <a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md#813-analysis-populations-and-lifetimes">The analysis</a> &middot; <a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md#510-ethylene-sa-casscf-findings-and-guidance-p9a">Ethylene: findings and guidance</a></p>
      </div>

      <figure class="feature-figure pop-figure">
<!-- populations-svg:begin -->
<!-- populations-svg:end -->
        <figcaption>Populations of the 67 of 100 trajectories that ran to 100 fs, with 95 % intervals; the dashed line is the fitted delayed exponential (tau 80 fs). The other 33 ended early after a hop. The default censored estimator also uses them and gives a shorter lifetime (62 fs here); which value is closer to the truth is not known (see <a href="#limits">Known limitations</a>).</figcaption>
      </figure>
    </div>
  </section>

  <section id="commands" class="section section-alt">
    <div class="wrap">
      <h2>Other commands</h2>
      <p class="section-intro">Besides <code>run</code>, a few commands prepare, inspect and control a project, and two run calculations without one.</p>
      <div class="card-grid">
        <article class="card">
          <h3><code>init</code> and <code>validate</code></h3>
          <p>Write a commented template input, then check an input file and its geometry before anything runs, with every problem listed at once.</p>
          <p class="card-link"><a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md#3-quick-start">Manual</a></p>
        </article>
        <article class="card">
          <h3><code>status</code> and <code>stop</code></h3>
          <p>See which stages are done and how many tasks run, and stop a running project gracefully from another terminal.</p>
          <p class="card-link"><a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md#5-commands">Manual</a></p>
        </article>
        <article class="card">
          <h3><code>single-point</code></h3>
          <p>One PySCF calculation at the input geometry, outside any project: energies, gradients, transition dipoles and oscillator strengths, as a table or JSON.</p>
          <p class="card-link"><a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md#53-gqteashmd-single-point">Manual</a></p>
        </article>
        <article class="card">
          <h3><code>model-dynamics</code></h3>
          <p>Surface-hopping ensembles on Tully's one-dimensional models, and on linear vibronic coupling models with singlets, triplets and spin-orbit coupling, without PySCF.</p>
          <p class="card-link"><a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md#57-gqteashmd-model-dynamics">Manual</a></p>
        </article>
      </div>
    </div>
  </section>

  <section id="checked" class="section">
    <div class="wrap">
      <h2>How it was checked</h2>
      <p class="section-intro">gqteashmd is research software at version 0.1. Each result below is marked by how it was checked, and each card links to the manual section with the numbers.</p>
      <ul class="badge-key">
        <li><span class="badge badge-test">Test</span> an automated test fails when the bound is missed</li>
        <li><span class="badge badge-pre">Pre-registered</span> a pass rule fixed before the run, checked by a tool run by hand</li>
        <li><span class="badge badge-report">Report</span> a measurement recorded without a pass rule</li>
      </ul>

      <div class="card-grid">
        <article class="card">
          <h3>Electronic structure</h3>
          <ul class="result-list">
            <li>Energies, gradients and nonadiabatic couplings against SHARC 4.1 (ethylene SA2-CASSCF(2,2)/6-31G*): 1.4e-8 Eh, 1.4e-8 Eh/bohr, 9.1e-8 1/bohr up to sign <span class="badge badge-test">Test</span></li>
            <li>Analytic gradients against finite differences <span class="badge badge-test">Test</span></li>
          </ul>
          <p class="card-link"><a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md#86-electronic-structure-pyscf">Manual</a></p>
        </article>
        <article class="card">
          <h3>Frequencies and sampling</h3>
          <ul class="result-list">
            <li>Harmonic frequencies against ORCA 6.1.1 (water RHF/6-31G*): within 0.052 cm<sup>&minus;1</sup> <span class="badge badge-test">Test</span></li>
            <li>Wigner sampling against SHARC's <code>wigner.py</code> at 0 and 1000 K: means within 1.6 and standard deviations within 1.4 standard errors <span class="badge badge-test">Test</span></li>
          </ul>
          <p class="card-link"><a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md#87-harmonic-analysis-frequencies">Manual</a> &middot; <a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md#88-wigner-sampling-of-initial-conditions">Sampling</a></p>
        </article>
        <article class="card">
          <h3>Excitations and selection</h3>
          <ul class="result-list">
            <li>Excitation energies against SHARC's <code>SHARC_PYSCF.py</code>, and the selection against <code>excite.py</code>: dE &le; 8.7e-8 Eh <span class="badge badge-test">Test</span></li>
          </ul>
          <p class="card-link"><a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md#89-initial-excitation">Manual</a></p>
        </article>
        <article class="card">
          <h3>Dynamics core</h3>
          <ul class="result-list">
            <li>FSSH and local diabatization against exact wavepackets on Tully's models <span class="badge badge-test">Test</span></li>
            <li>FSSH against an exact 2D LVC solver: largest differences 0.049 (quantum) and 0.072 (classical populations) <span class="badge badge-report">Report</span></li>
            <li>Single trajectories against SHARC 4.1: nuclear motion within 7.5e-11 bohr <span class="badge badge-test">Test</span></li>
          </ul>
          <p class="card-link"><a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md#814-dynamics-core-models-determinism-restart-performance-and-validation">Manual</a> &middot; <a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md#815-lvc-models-spinorbit-coupling-and-the-diagonal-representation">LVC</a></p>
        </article>
        <article class="card">
          <h3>The dynamics stage</h3>
          <ul class="result-list">
            <li>Restart and extension bit-identical with an analytic backend, and to round-off with PySCF <span class="badge badge-test">Test</span></li>
            <li>SHARC's driver on our electronic structure: positions within 2.1e-7 bohr over 41 steps <span class="badge badge-test">Test</span></li>
          </ul>
          <p class="card-link"><a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md#816-the-dynamics-stage-with-pyscf-p7b">Manual</a></p>
        </article>
        <article class="card">
          <h3>Ensembles against SHARC</h3>
          <ul class="result-list">
            <li>Ethylene, 100 trajectories, SHARC driving our electronic structure: all four comparison rules pass on the 48 paired trajectories <span class="badge badge-pre">Pre-registered</span></li>
            <li>Our population functions against SHARC's <code>populations.py</code> on 100 SHARC trajectories <span class="badge badge-pre">Pre-registered</span></li>
            <li>Against SHARC with its own electronic-structure interface: inconclusive, 13 paired <span class="badge badge-report">Report</span></li>
          </ul>
          <p class="card-link"><a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md#817-sharc-ensemble-comparison-p9b">Manual</a></p>
        </article>
        <article class="card">
          <h3>The analysis</h3>
          <ul class="result-list">
            <li>The censored estimator is unbiased and its intervals cover on a synthetic Markov model <span class="badge badge-test">Test</span></li>
            <li>The least biased of three estimators under artificial random losses <span class="badge badge-pre">Pre-registered</span></li>
          </ul>
          <p class="card-link"><a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md#819-the-censored-analysis-and-the-hop-step-energy-check-measurements">Manual</a></p>
        </article>
      </div>

      <aside id="limits" class="limits" aria-labelledby="limits-title">
        <h3 id="limits-title">Known limitations</h3>
        <ul>
          <li>Closed-shell singlets only: no triplets or spin-orbit coupling from PySCF (they exist only for the analytic and LVC models of <code>model-dynamics</code>). Linux only, on one workstation, without a batch scheduler.</li>
          <li>33 of the 100 ethylene trajectories end early after a hop, when the CASSCF calculation switches to another solution.</li>
          <li>The lifetime is uncertain: the censored estimate is 62 fs, the complete-trajectory estimate 80 fs, and which is closer to the truth is not known.</li>
          <li>The comparison with SHARC's own electronic-structure interface is inconclusive; a replayed trajectory disagrees with SHARC after its hop under the default hop energy rule; the lifetime fit has not been compared with SHARC's <code>make_fit.py</code>.</li>
          <li>No plots: the results come as CSV, HDF5, JSON and Markdown files.</li>
        </ul>
        <p>The full list is in the <a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/CHANGELOG.md">release notes</a> and the <a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md#10-known-limitations">manual</a>.</p>
      </aside>
    </div>
  </section>

  <section id="install" class="section section-alt">
    <div class="wrap install-grid">
      <div>
        <h2>Install</h2>
        <ol class="steps">
          <li>
            <h3>Download the release</h3>
            <p>From the <a href="https://github.com/AdemirCamargo57/gqteaSHMD/releases/latest">latest release</a>, download <code>environment.lock.txt</code> and the wheel, <code>gqteashmd-0.1.0-py3-none-any.whl</code>.</p>
          </li>
          <li>
            <h3>Create the environment</h3>
            <p>In a Linux shell: <code>conda create -n gqteashmd-0.1.0 --file environment.lock.txt</code>, then <code>conda activate gqteashmd-0.1.0</code>. The lock file gives the exact packages the release was tested with.</p>
          </li>
          <li>
            <h3>Install the program</h3>
            <p><code>pip install --no-deps gqteashmd-0.1.0-py3-none-any.whl</code>. Check it with <code>gqteashmd --version</code>, which names the version and the commit it was built from.</p>
          </li>
          <li>
            <h3>Run a project</h3>
            <p><code>gqteashmd init input.toml</code> writes a commented template. Edit it, check it with <code>gqteashmd validate input.toml</code>, then start it with <code>gqteashmd run input.toml</code>.</p>
          </li>
        </ol>
      </div>

      <aside class="requirements" aria-labelledby="req-title">
        <h3 id="req-title">Requirements</h3>
        <ul>
          <li>Linux, 64-bit. Tested on Oracle Linux 9.5 under WSL2.</li>
          <li>conda (for example Miniforge). PySCF, NumPy, SciPy, h5py, geomeTRIC and pydantic all come from the lock file.</li>
          <li>On Windows, use WSL2, and keep projects on the Linux filesystem, not under <code>/mnt/c</code> or a OneDrive folder.</li>
          <li>SHARC 4.1 and ORCA are needed only for the comparison tests, not to run gqteashmd.</li>
        </ul>

        <details class="source-install">
          <summary>Install for development instead</summary>
          <p>From a clone of the repository, in WSL or Linux:</p>
<pre><code>git clone https://github.com/AdemirCamargo57/gqteaSHMD.git
cd gqteaSHMD
conda env create -f environment.yml
conda activate gqteashmd-dev
pip install -e . --no-deps</code></pre>
          <p>Details are in the <a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md#221-install-for-development">user manual</a>.</p>
        </details>
      </aside>
    </div>
  </section>

  <section id="docs" class="section">
    <div class="wrap docs-grid">
      <div>
        <h2>Learn to use it</h2>
        <p>The <a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/USER_MANUAL.md">user manual</a> describes every input key, command, stage and output file, with worked examples on water (optimization, frequencies, sampling, excitations, dynamics) and the ethylene studies of the program's validation. Each method has a section on its formulas and on how it was checked.</p>
        <p>Found a bug or have a request? <a href="https://github.com/AdemirCamargo57/gqteaSHMD/issues">Open an issue on GitHub</a>.</p>
      </div>
      <div>
        <h2>Who makes it</h2>
        <p>gqteashmd is developed by the <span lang="pt-BR">Grupo de Química Teórica e Estrutural de Anápolis</span> (gQTEA) at the State University of Goiás (UEG), Brazil, for research and graduate teaching in computational chemistry.</p>
        <ul class="team">
          <li>Ademir J. Camargo</li>
          <li>Valter H. C. Silva</li>
          <li>Solemar S. Oliveira</li>
          <li>Hamilton B. Napolitano</li>
          <li>Luciano Ribeiro</li>
          <li>Flavio O. Sanches</li>
        </ul>
        <p>gqteashmd was written by the gQTEA team with the help of an AI coding assistant (Claude, Anthropic); every result on this page comes from the program's tests and recorded runs.</p>
        <p>Contact: <a href="mailto:ajc@ueg.br">ajc@ueg.br</a></p>
      </div>
    </div>
  </section>

</main>

<footer class="site-footer">
  <div class="wrap">
    <p>gqteashmd is free software released under the <a href="https://github.com/AdemirCamargo57/gqteaSHMD/blob/main/LICENSE">MIT License</a>. © 2026 the gqteashmd developers, State University of Goiás.</p>
  </div>
</footer>

<script src="app.js"></script>
</body>
</html>
```

Before writing, check two names the page assumes: the licence file is `LICENSE` at the repository root (`ls LICENSE*`; if it is named otherwise, use that name in the footer link) and `tests/data/ethylene.xyz` exists.

- [ ] **Step 6: Generate the figures from the ensemble**

In WSL, from the repository root:

```bash
python tools/site/make_figures.py ~/p9b/ours/run --task traj_00033 --docs docs
```

Expected output: `trajectory.json: 201 frames, hop at 28 fs; populations: 67 of 100, tau 80.1 fs`. If the hop time, the counts or tau differ, stop and report (the captions name 28 fs, 67 of 100 and 80 fs).

- [ ] **Step 7: Run the tests**

Run: `pytest tests/site -q -W error`
Expected: all pass (7 from Task 1, 12 here). A failing `test_manual_anchors_exist` lists the anchors that do not match a manual heading: fix the link in `index.html`, never the manual.

- [ ] **Step 8: Look at it**

`python -m http.server -d docs 8000` in WSL serves the page at `http://localhost:8000/` (WSL2 forwards localhost to Windows). The implementer cannot open a browser: record in the report that the visual check is left to the controller's Task 3 step. Check with `curl -s localhost:8000/trajectory.json | head -c 200` that the server serves the data, then stop the server.

- [ ] **Step 9: Commit**

```bash
git add docs/index.html docs/style.css docs/app.js docs/icon.svg docs/.nojekyll docs/trajectory.json tests/site/test_site.py
git commit -m "feat: add the project website"
```
(with the `Co-Authored-By` trailer)

---

### Task 3: The documents and the owner's check

**Files:**
- Modify: `README.md` (top), `USER_MANUAL.md` (section 1), `CLAUDE.md` (the read-first list, a new block after "Release 0.1.0", the status, the table)

**Interfaces:**
- Consumes: the file names of Tasks 1-2; the site address `https://ademircamargo57.github.io/gqteaSHMD/`.

- [ ] **Step 1: README.md**

After the first paragraph (ending "Linux only (Oracle Linux 9; WSL2 works).") add a line:

```markdown
**Website:** https://ademircamargo57.github.io/gqteaSHMD/
```

and change the status line to `**Status:** latest release 0.1.0, the first release ([CHANGELOG.md](CHANGELOG.md)); \`main\` is the development snapshot after it.` (keep its second sentence).

- [ ] **Step 2: USER_MANUAL.md**

At the end of section "1. What the program does" add a paragraph: `The program's website, https://ademircamargo57.github.io/gqteaSHMD/, gives an overview, the installation steps of the latest release and how each part was checked; this manual is the reference.`

- [ ] **Step 3: CLAUDE.md**

1. In "Read first when resuming", after the release 0.1.0 bullet: `- **The project website: spec and plan:** [docs/superpowers/specs/2026-11-26-project-website-design.md](docs/superpowers/specs/2026-11-26-project-website-design.md) (owner decisions in section 0) and [docs/superpowers/plans/2026-11-26-project-website.md](docs/superpowers/plans/2026-11-26-project-website.md) (rulings PR1-PR3); ledger \`.superpowers/sdd/2026-11-26-project-website/\` (git-ignored, local).`
2. After the "Release 0.1.0" block in Architecture, a block: `- **Project website (branch \`feat/project-website\`).** GitHub Pages from \`docs/\` on \`main\` (the owner enables it: Settings → Pages → \`main\`, \`/docs\`), address https://ademircamargo57.github.io/gqteaSHMD/. Files \`docs/index.html\`, \`style.css\` (gqteaWinToga's, plus the player, populations, badge and limitations rules), \`app.js\` (release button, tabs, trajectory player), \`icon.svg\`, \`trajectory.json\`, \`.nojekyll\`. The two figures come from the P9b "ours" ensemble (\`~/p9b/ours/run\` in WSL): \`tools/site/make_figures.py PROJECT --task traj_00033 --docs docs\` writes \`trajectory.json\` and the inline populations SVG between \`<!-- populations-svg:begin -->\` and \`<!-- populations-svg:end -->\`. \`tests/site/test_site.py\` (fast tier) checks links, manual anchors (GitHub's slug rule), the sample input (equal to \`tools/sharc_compare/p9b/ethylene_seed2028.toml\` and valid), the figure data and the captions' numbers; \`tests/site/test_make_figures.py\` the script. **Rule: a renamed USER_MANUAL heading can break a site link; the site test catches it.** Every number on the page comes from CHANGELOG 0.1.0 or from the figure files; update the page with each release.`
3. In "Current status", a bullet `- **Project website: in progress on \`feat/project-website\`** (spec \`20da77c\`, owner decisions 2026-11-26).`
4. In the table of "Remaining work", a row after "Release 0.1.0": `| Website | The project website on GitHub Pages | in progress on \`feat/project-website\` |`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/site tests/test_package.py -q -W error`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add README.md USER_MANUAL.md CLAUDE.md
git commit -m "docs: record the project website"
```
(with the `Co-Authored-By` trailer)

- [ ] **Step 6: The visual check (controller, then owner)**

The controller serves `docs/` (`python -m http.server -d docs 8000` in WSL) and asks the owner to open `http://localhost:8000/` and check, against spec section 5: the trajectory plays, pauses, and the slider moves it, with the hop marked; the drag rotation; the tabs with the mouse and the arrow keys; the populations figure; light and dark mode; a narrow window (about 375 px, no horizontal scroll); the release button text (filled from the GitHub API). Findings are fixed on the branch before the merge; a different trajectory (spec 8: `traj_00004`) is a ruling.

---

## After the plan

Final review, then the owner decides the merge; after the merge the owner enables GitHub Pages and the CLAUDE.md status is updated (website DONE, the address live).
