"""Initial-state selection statistics against SHARC 4.1 excite.py (run as a black box).

Synthetic excitation data are written in SHARC's text formats by our own writers; excite.py is
driven through stdin. Nothing from SHARC is read or imported.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import ks_2samp

from gqteashmd.core.units import HARTREE_IN_EV
from gqteashmd.excite.selection import candidates_from, select_bright

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "sharc_compare"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pytestmark = [
    pytest.mark.sharc,
    pytest.mark.skipif(
        not (os.environ.get("SHARC_HOME") and os.environ.get("SHARC_PYTHON")),
        reason="needs SHARC_HOME and SHARC_PYTHON",
    ),
]

N = 3000
WINDOW = (4.0, 8.0)
SYMBOLS = ("O", "H", "H")
EQ_BOHR = np.array(
    [[0.0, 0.0, 0.28377702], [0.0, 1.43256453, -0.91771557], [0.0, -1.43256453, -0.91771557]]
)
EQ_ANGSTROM = np.array(
    [
        [0.0, 0.0, 0.1501683345],
        [0.0, 0.7580805047, -0.4856341673],
        [0.0, -0.7580805047, -0.4856341673],
    ]
)
E_REF = -75.0


def _synthetic():
    """Excitation energies (eV) and oscillator strengths, columns S1 and S2."""
    rng = np.random.default_rng(20261108)
    de = np.column_stack([rng.uniform(4, 6, N), rng.uniform(5, 9, N)])
    f = np.column_stack([rng.uniform(0, 0.1, N), rng.uniform(0, 0.4, N)])
    big = rng.random(N) < 0.10  # bright S2 pairs above the window: f_max depends on semantics
    de[big, 1] = rng.uniform(8.5, 9.0, big.sum())
    f[big, 1] = rng.uniform(0.4, 0.8, big.sum())
    # S1 pairs below the window with a larger f/dE^2 than every in-window pair: the window
    # maximum and the all-pairs maximum then differ under f/dE^2 as well.
    low = rng.random(N) < 0.10
    de[low, 0] = rng.uniform(3.0, 3.8, low.sum())
    f[low, 0] = rng.uniform(0.3, 0.4, low.sum())
    return de, f


def _write_tree(root, formats, de_ev, f):
    n = de_ev.shape[0]
    masses = [15.994915, 1.007825, 1.007825]
    root.joinpath("initconds").write_text(
        formats.format_initconds(
            SYMBOLS,
            [8.0, 1.0, 1.0],
            masses,
            EQ_BOHR,
            np.tile(EQ_BOHR, (n, 1, 1)),
            np.zeros((n, 3, 3)),
        )
    )
    qm_in = formats.format_qm_in(SYMBOLS, EQ_ANGSTROM, 3)

    def write(k, energies, dipoles):
        d = root / f"ICOND_{k:05d}"
        d.mkdir()
        (d / "QM.in").write_text(qm_in)
        (d / "QM.out").write_text(formats.format_qm_out(energies, dipoles))

    write(0, [E_REF, E_REF + 0.5, E_REF + 0.6], np.zeros((3, 3, 3)))
    for s in range(n):
        de_eh = de_ev[s] / HARTREE_IN_EV
        dip = np.zeros((3, 3, 3))
        for col in range(2):
            mu = np.sqrt(3.0 * f[s, col] / (2.0 * de_eh[col]))
            dip[0, 0, col + 1] = dip[0, col + 1, 0] = mu
        write(s + 1, [E_REF, E_REF + de_eh[0], E_REF + de_eh[1]], dip)


def _expectations(de, f):
    """Per-state acceptance probabilities under the four candidate rules."""
    inside = (de >= WINDOW[0]) & (de <= WINDOW[1])
    w = f / de**2
    out = {}
    for name, weight in (("f/dE^2", w), ("f", f)):
        out[f"{name} (window max)"] = np.where(inside, weight / weight[inside].max(), 0.0)
        out[f"{name} (all-pairs max)"] = np.where(inside, weight / weight.max(), 0.0)
    return out


def _z(count, p):
    return (count - p.sum()) / np.sqrt((p * (1.0 - p)).sum())


@pytest.fixture(scope="module")
def sharc_run(tmp_path_factory):
    formats = _load("sharc_formats")
    run_excite = _load("run_excite")
    root = tmp_path_factory.mktemp("excite")
    de, f = _synthetic()
    _write_tree(root, formats, de, f)
    out = run_excite.run_excite(
        Path(os.environ["SHARC_HOME"]), os.environ["SHARC_PYTHON"], root, window_eV=WINDOW, seed=7
    )
    parsed = formats.parse_initconds_excited(out.read_text())
    rows = parsed["states"]
    return {
        "eref": parsed["eref_hartree"],
        "n_states": parsed["n_states"],
        "n": len(rows),
        "de_in": de,
        "f_in": f,
        "de": np.array([[r[1]["delta_e_eV"], r[2]["delta_e_eV"]] for r in rows]),
        "f": np.array([[r[1]["f"], r[2]["f"]] for r in rows]),
        "selected": np.array([[r[1]["selected"], r[2]["selected"]] for r in rows]),
    }


def test_sharc_reports_our_input_back(sharc_run):
    """Our writers and units: excite.py reads our files and reports the same numbers."""
    run = sharc_run
    assert run["n_states"] == 3 and run["n"] == N
    assert run["eref"] == pytest.approx(E_REF, abs=1e-8)
    d_de = np.abs(run["de"] - run["de_in"]).max()
    d_f_abs = np.abs(run["f"] - run["f_in"]).max()
    d_f_rel = np.max(np.abs(run["f"] - run["f_in"]) / run["f_in"])
    print(f"\nmax |dE_SHARC - dE_input| = {d_de:.3e} eV")
    print(f"max |f_SHARC - f_input| = {d_f_abs:.3e}; max relative f difference = {d_f_rel:.3e}")
    assert d_de < 1e-6
    # excite.py prints f with 8 decimals, so small f need an absolute floor of 1e-8.
    assert np.all(np.abs(run["f"] - run["f_in"]) <= 1e-6 * run["f_in"] + 1e-8)
    outside = (run["de_in"] < WINDOW[0]) | (run["de_in"] > WINDOW[1])
    assert not run["selected"][outside].any(), "SHARC selected a pair outside the window"


DEFAULT_RULE = "f/dE^2 (window max)"


def test_our_select_bright_follows_the_default_rule_and_resembles_sharc(sharc_run):
    de, f = sharc_run["de_in"], sharc_run["f_in"]
    cands = candidates_from(de, f, np.ones(N, dtype=bool), ["S1", "S2"], WINDOW)
    ours = select_bright(cands, len(cands), np.random.default_rng(1))
    counts = [sum(c.state == s for c in ours.selected) for s in ("S1", "S2")]
    p = _expectations(de, f)[DEFAULT_RULE]
    z = [_z(counts[c], p[:, c]) for c in range(2)]
    print(f"\nour counts S1, S2 = {counts}, expected {p[:, 0].sum():.1f}, {p[:, 1].sum():.1f}")
    print(f"z = ({z[0]:+.2f}, {z[1]:+.2f}); our weight_max = {ours.weight_max:.6g}")
    assert all(abs(x) < 4 for x in z)
    sel = sharc_run["selected"]
    de_ours = np.array([c.excitation_energy_eV for c in ours.selected if c.state == "S1"])
    ks = ks_2samp(de_ours, de[sel[:, 0], 0])
    print(
        f"KS of selected S1 dE (ours vs SHARC): statistic {ks.statistic:.4f}, p = {ks.pvalue:.4f}"
    )
    assert ks.pvalue > 1e-3


def test_sharc_counts_follow_f_over_de_squared_with_the_window_maximum(sharc_run):
    """SHARC matches f/dE^2 normalised by the in-window maximum, and none of the alternatives."""
    de, f, sel = sharc_run["de_in"], sharc_run["f_in"], sharc_run["selected"]
    counts = sel.sum(axis=0)
    print(f"\nSHARC selected counts S1, S2 = {counts.tolist()}")
    zs = {}
    for name, p in _expectations(de, f).items():
        zs[name] = [float(_z(counts[c], p[:, c])) for c in range(2)]
        print(
            f"  {name:24s} expected S1 {p[:, 0].sum():8.2f} S2 {p[:, 1].sum():8.2f} "
            f"z = ({zs[name][0]:+.2f}, {zs[name][1]:+.2f})"
        )
    assert all(abs(z) < 4 for z in zs[DEFAULT_RULE]), f"z = {zs[DEFAULT_RULE]}"
    for name, z in zs.items():
        if name != DEFAULT_RULE:
            assert max(abs(x) for x in z) > 6, f"{name} is not discriminated: z = {z}"
