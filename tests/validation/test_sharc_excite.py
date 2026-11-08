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
    big = rng.random(N) < 0.10  # bright S2 pairs outside the window: f_max depends on semantics
    de[big, 1] = rng.uniform(8.5, 9.0, big.sum())
    f[big, 1] = rng.uniform(0.4, 0.8, big.sum())
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
    """Per-state acceptance probabilities under our rule and two alternatives."""
    inside = (de >= WINDOW[0]) & (de <= WINDOW[1])
    out = {}
    out["f/f_max (window)"] = np.where(inside, f / f[inside].max(), 0.0)
    out["f/f_max (all pairs)"] = np.where(inside, f / f.max(), 0.0)
    w = f / de**2
    out["f/dE^2 (window)"] = np.where(inside, w / w[inside].max(), 0.0)
    return out


def _z(count, p):
    return (count - p.sum()) / np.sqrt((p * (1.0 - p)).sum())


OBSERVED_RULE_DIFFERENCE = (
    "SHARC 4.1 excite.py (delta-pulse excitation, option 3) does not accept a pair with "
    "probability f/f_max. Measured on N = 3000 synthetic samples, window [4, 8] eV, seed 7: "
    "SHARC selected [435, 674] pairs (S1, S2); our rule f/f_max(window) expects [368.0, 1026.7] "
    "(z = +3.82, -19.11) and f/f_max(all pairs) expects [184.6, 515.1] (z = +19.24, +8.59). "
    "SHARC is consistent with a probability proportional to f/dE^2, normalised by the largest "
    "f/dE^2 over the in-window pairs: expected [387.4, 654.6] (z = +2.68, +1.03). Owner decision 1 "
    "(Newton-X-style f/f_max) is kept; the controller decides whether to change it."
)


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


def test_our_select_bright_follows_our_rule_and_resembles_sharc(sharc_run):
    de, f = sharc_run["de_in"], sharc_run["f_in"]
    cands = candidates_from(de, f, np.ones(N, dtype=bool), ["S1", "S2"], WINDOW)
    ours = select_bright(cands, len(cands), np.random.default_rng(1))
    counts = [sum(c.state == s for c in ours.selected) for s in ("S1", "S2")]
    p = _expectations(de, f)["f/f_max (window)"]
    z = [_z(counts[c], p[:, c]) for c in range(2)]
    print(f"\nour counts S1, S2 = {counts}, expected {p[:, 0].sum():.1f}, {p[:, 1].sum():.1f}")
    print(f"z = ({z[0]:+.2f}, {z[1]:+.2f}); our f_max = {ours.f_max:.6f}")
    assert all(abs(x) < 4 for x in z)
    sel = sharc_run["selected"]
    de_ours = np.array([c.excitation_energy_eV for c in ours.selected if c.state == "S1"])
    ks = ks_2samp(de_ours, de[sel[:, 0], 0])
    print(
        f"KS of selected S1 dE (ours vs SHARC): statistic {ks.statistic:.4f}, p = {ks.pvalue:.4f}"
    )
    assert ks.pvalue > 1e-3


def _report(sharc_run):
    de, f, sel = sharc_run["de_in"], sharc_run["f_in"], sharc_run["selected"]
    counts = sel.sum(axis=0)
    print(f"\nSHARC selected counts S1, S2 = {counts.tolist()}")
    result = {}
    for name, p in _expectations(de, f).items():
        zs = [_z(counts[c], p[:, c]) for c in range(2)]
        result[name] = zs
        print(
            f"  {name:22s} expected S1 {p[:, 0].sum():8.2f} S2 {p[:, 1].sum():8.2f} "
            f"z = ({zs[0]:+.2f}, {zs[1]:+.2f})"
        )
    return result


@pytest.mark.xfail(strict=True, reason=OBSERVED_RULE_DIFFERENCE)
def test_sharc_counts_match_f_over_fmax(sharc_run):
    zs = _report(sharc_run)["f/f_max (window)"]
    assert all(abs(z) < 4 for z in zs), f"z = {zs}"


def test_sharc_counts_match_f_over_de_squared(sharc_run):
    """Documents the rule SHARC was observed to follow (probability ~ f/dE^2, window maximum)."""
    zs = _report(sharc_run)["f/dE^2 (window)"]
    assert all(abs(z) < 4 for z in zs), f"z = {zs}"
