"""The LVC model (with triplets and SOC) against SHARC 4.1's SHARC_LVC.py at single points.

SHARC is only executed, from ``tools/sharc_compare/run_sharc_lvc.py`` (our writers; QM.out read
by our ``qmout.py``; formats from the public SHARC 4.1 manual, sections 6.4 and 6.34). Both
programs read the same V0.txt and LVC.template (the SHARC copy without the DM blocks, which
SHARC_LVC.py cannot combine with SOC on this installation; dipoles are not compared).

Points: for each fixture, 6 geometries q0 = cartesian(Q) with Q ~ N(0, 1) from
``default_rng(5)`` (all 6 drawn first), and partners q1 = cartesian(Q + 0.05 d) with d a unit
random direction of the vibrational normal coordinates (drawn next from the same generator).
SHARC runs ``step 0`` at q0 and ``step 1`` at q1 with ``OVERLAP`` and the same savedir; we
evaluate snap0 = evaluate(q0, None) and snap1 = evaluate(q1, snap0).

Pre-registered tolerances (fixed from SHARC's print format before any difference was looked
at): every QM.out block is printed with ``%.12E`` (13 significant digits), so the last printed
digit of a block is 10^(e - 12), e the decimal exponent of its largest |value|, and
tol = max(1e-9, 10 * 10^(e - 12)) per block and point (the NAC tolerance applies to
max |dK| / max |K|). The comparisons are up to one sign per spin-free MCH state, D = diag(+-1)
the same for every M_s component, chosen per point as the D that minimises the larger of the
H and NAC residuals (each in units of its tolerance); the global sign of D, which H, NACs and
gradients cannot fix, is + on S0. Gradients are compared without signs; overlaps as
D(q0) S D(q1).

Run with ``-s`` to see every number.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from gqteashmd.dynamics.lvc import LVCModel

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "sharc_compare"
DATA = ROOT / "tests" / "data" / "lvc"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lvc_tool = _load("run_sharc_lvc")

pytestmark = [
    pytest.mark.sharc,
    pytest.mark.skipif(
        not (os.environ.get("SHARC_HOME") and os.environ.get("SHARC_PYTHON")),
        reason="needs SHARC_HOME and SHARC_PYTHON",
    ),
]

FIXTURES = {
    "lvc_singlets": {"states": (3,), "soc": False},
    "lvc_soc": {"states": (2, 0, 2), "soc": True},
}
N_GEOMETRIES = 6
SEED = 5
DISPLACEMENT_Q = 0.05
FLOOR = 1e-9
PRINTED_DECIMALS = 12  # %.12E


def print_tolerance(values: np.ndarray) -> float:
    """max(1e-9, 10 x the last printed digit) of a block printed with %.12E."""
    largest = float(np.max(np.abs(values))) if np.size(values) else 0.0
    if largest == 0.0:
        return FLOOR
    exponent = math.floor(math.log10(largest))
    return max(FLOOR, 10.0 * 10.0 ** (exponent - PRINTED_DECIMALS))


def geometries(model: LVCModel) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(SEED)
    big_q = rng.standard_normal((N_GEOMETRIES, model.n_dof))
    directions = rng.standard_normal((N_GEOMETRIES, model.n_dof))
    vibrational = np.asarray(model.parameters.v0.frequencies_Eh) > 0
    pairs = []
    for k in range(N_GEOMETRIES):
        d = np.where(vibrational, directions[k], 0.0)
        d /= np.linalg.norm(d)
        pairs.append((model.cartesian(big_q[k]), model.cartesian(big_q[k] + DISPLACEMENT_Q * d)))
    return pairs


def _sharc_arrays(data: dict, n: int, f: int) -> dict[str, np.ndarray]:
    out = {
        "hamiltonian": np.array(data["hamiltonian"], dtype=complex),
        "gradients": np.array(data["gradients"], dtype=float).reshape(n, f),
        "nacs": np.array(data["nacs"], dtype=float).reshape(n, n, f),
        "gradient_labels": data["gradient_labels"],
        "nac_labels": data["nac_labels"],
    }
    if "overlap" in data:
        out["overlap"] = np.array(data["overlap"], dtype=complex)
    return out


def _sign_choices(model: LVCModel) -> tuple[list[str], np.ndarray, list[np.ndarray]]:
    """Spin-free labels, the map expanded state -> spin-free index, and every D (S0 sign +)."""
    keys: list[tuple[int, int]] = []
    for s in model.states:
        if (s.multiplicity, s.index) not in keys:
            keys.append((s.multiplicity, s.index))
    labels = [f"S{i}" if m == 1 else f"T{i + 1}" for m, i in keys]
    index = np.array([keys.index((s.multiplicity, s.index)) for s in model.states])
    choices = [
        np.array((1.0, *signs))[index]
        for signs in itertools.product((1.0, -1.0), repeat=len(keys) - 1)
    ]
    return labels, index, choices


def _compare_point(model: LVCModel, snap, sharc: dict) -> dict:
    """Residuals at one point for the D that fits H and NACs best."""
    h_ours = np.asarray(snap.hamiltonian, dtype=complex)
    h_sharc = sharc["hamiltonian"]
    k_sharc = sharc["nacs"]
    tol = {
        "H": print_tolerance(np.concatenate([h_sharc.real.ravel(), h_sharc.imag.ravel()])),
        "E": print_tolerance(np.diag(h_sharc).real),
        "grad": print_tolerance(sharc["gradients"]),
        "nac": print_tolerance(k_sharc),
    }
    k_max = float(np.max(np.abs(k_sharc)))
    labels, index, choices = _sign_choices(model)
    best = None
    for d in choices:
        dd = np.outer(d, d)
        h_res = float(np.max(np.abs(dd * h_ours - h_sharc)))
        k_res = float(np.max(np.abs(dd[:, :, None] * snap.nacs - k_sharc))) / k_max
        score = max(h_res / tol["H"], k_res / tol["nac"])
        if best is None or score < best[0]:
            best = (score, d, h_res, k_res)
    assert best is not None
    _, d, h_res, k_res = best
    d_spin_free = [int(d[list(index).index(j)]) for j in range(len(labels))]
    return {
        "D": d,
        "D_label": " ".join(
            f"{lab}{'+' if s > 0 else '-'}" for lab, s in zip(labels, d_spin_free, strict=True)
        ),
        "tol": tol,
        "E": float(np.max(np.abs(np.diag(h_ours).real - np.diag(h_sharc).real))),
        "H": h_res,
        "H_conjugated": float(np.max(np.abs(np.outer(d, d) * h_ours.conj() - h_sharc))),
        "grad": float(np.max(np.abs(snap.gradients - sharc["gradients"]))),
        "nac": k_res,
        "nac_max": k_max,
    }


_CACHE: dict[str, dict] = {}


def _run_fixture(name: str, base: Path) -> dict:
    if name in _CACHE:
        return _CACHE[name]
    spec = FIXTURES[name]
    template_path = DATA / f"{name}.template"
    model = LVCModel.from_template(template_path)
    assert model.state_counts == spec["states"], model.state_counts
    template = lvc_tool.without_dipoles(template_path.read_text())
    assert template.splitlines()[0].strip() == "V0.txt"
    v0_text = (DATA / "V0.txt").read_text()
    symbols = model.parameters.v0.symbols
    n, f = model.n_states, model.n_dof
    home, python = Path(os.environ["SHARC_HOME"]), os.environ["SHARC_PYTHON"]
    points = []
    for k, (q0, q1) in enumerate(geometries(model)):
        work = base / f"{name}-{k}"
        runs = []
        for step, q in enumerate((q0, q1)):
            qm_in = lvc_tool.format_qm_in(
                symbols,
                q.reshape(-1, 3),
                states=spec["states"],
                savedir=str(work / "SAVE"),
                step=step,
                soc=spec["soc"],
                overlap=step == 1,
            )
            data = lvc_tool.run_single_point(
                work,
                qm_in,
                template,
                v0_text,
                sharc_home=home,
                sharc_python=python,
                timeout=300,
            )
            runs.append(_sharc_arrays(data, n, f))
        snap0 = model.evaluate(q0, None)
        snap1 = model.evaluate(q1, snap0)
        at0 = _compare_point(model, snap0, runs[0])
        at1 = _compare_point(model, snap1, runs[1])
        s_sharc = runs[1]["overlap"]
        s_ours = np.asarray(snap1.overlaps, dtype=float)
        dd = np.outer(at0["D"], at1["D"])
        points.append(
            {
                "q0": at0,
                "q1": at1,
                "S": float(np.max(np.abs(dd * s_ours - s_sharc))),
                "S_transposed": float(np.max(np.abs(dd * s_ours.T - s_sharc))),
                "S_tol": print_tolerance(
                    np.concatenate([s_sharc.real.ravel(), s_sharc.imag.ravel()])
                ),
                "labels": runs[0]["gradient_labels"],
                "nac_labels": runs[0]["nac_labels"],
            }
        )
    result = {"model": model, "points": points}
    _CACHE[name] = result
    _report(name, points)
    return result


def _report(name: str, points: list[dict]) -> None:
    print(f"\n{name}: max differences per point (tolerance in brackets)")
    for k, p in enumerate(points):
        for where in ("q0", "q1"):
            r = p[where]
            print(
                f"  g{k} {where}: D = {r['D_label']:<16} |dE| {r['E']:.2e} [{r['tol']['E']:.0e}]"
                f"  |dH| {r['H']:.2e} [{r['tol']['H']:.0e}] (conj {r['H_conjugated']:.2e})"
                f"  |dg| {r['grad']:.2e} [{r['tol']['grad']:.0e}]"
                f"  |dK|/max|K| {r['nac']:.2e} [{r['tol']['nac']:.0e}] (max|K| {r['nac_max']:.3f})"
            )
        print(
            f"  g{k} overlap D(q0) S D(q1): {p['S']:.2e} [{p['S_tol']:.0e}]"
            f" (transposed {p['S_transposed']:.2e})"
        )


@pytest.fixture(scope="module", params=list(FIXTURES))
def compared(request, tmp_path_factory):
    name = request.param
    return name, _run_fixture(name, tmp_path_factory.mktemp(f"sharc-{name}"))


def _all_points(result: dict):
    for p in result["points"]:
        yield p["q0"]
        yield p["q1"]


def test_state_order_is_canonical(compared):
    """SHARC prints the gradients and NACs in our canonical order (S, then T with M_s = -1,
    0, +1, the state index innermost) and gives every M_s component and every pair."""
    _, result = compared
    model = result["model"]
    expected = [(s.multiplicity, s.index + 1, s.ms) for s in model.states]
    for p in result["points"]:
        assert p["labels"] == expected
        assert [[pair for pair in row] for row in p["nac_labels"]] == [
            [(a, b) for b in expected] for a in expected
        ]


def test_energies(compared):
    _, result = compared
    for r in _all_points(result):
        assert r["E"] <= r["tol"]["E"], (r["E"], r["tol"]["E"])


def test_hamiltonian_up_to_signs(compared):
    _, result = compared
    for r in _all_points(result):
        assert r["H"] <= r["tol"]["H"], (r["D_label"], r["H"], r["tol"]["H"])


def test_gradients_of_every_component(compared):
    _, result = compared
    for r in _all_points(result):
        assert r["grad"] <= r["tol"]["grad"], (r["grad"], r["tol"]["grad"])


def test_nacs_up_to_signs(compared):
    _, result = compared
    for r in _all_points(result):
        assert r["nac"] <= r["tol"]["nac"], (r["D_label"], r["nac"], r["tol"]["nac"])


def test_overlaps_up_to_signs(compared):
    _, result = compared
    for p in result["points"]:
        assert p["S"] <= p["S_tol"], (p["q0"]["D_label"], p["q1"]["D_label"], p["S"], p["S_tol"])
