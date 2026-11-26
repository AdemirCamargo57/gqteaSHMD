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
    anchors = [v.split("#", 1)[1] for _t, _a, v in page.links if re.search(r"USER_MANUAL\.md#", v)]
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
    m = re.search(
        r'<figure class="feature-figure input-sample">\s*<pre><code>(.*?)</code></pre>',
        text,
        re.S,
    )
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
