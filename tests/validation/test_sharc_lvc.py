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

The second part (Task 10, marker ``slow``, about 8 min) runs sharc.x trajectories with
SHARC_LVC.py behind ``QM/runQM.sh`` and compares them with our core step by step; its rules
are in the comment block before it.

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

from gqteashmd.dynamics.driver import initial_state, run_trajectory
from gqteashmd.dynamics.ensemble import lvc_initial_conditions
from gqteashmd.dynamics.lvc import LVCModel
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.randomness import ReplaySource
from gqteashmd.dynamics.representation import diagonal_basis, diagonal_energies, mch_state

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


# --- Task 10: sharc.x trajectories with spin-orbit coupling --------------------------------------
#
# sharc.x runs SHARC_LVC.py behind QM/runQM.sh (``run_sharc_lvc.run_lvc_trajectory``); both
# programs start from the same Wigner sample of ``lvc_soc`` (``lvc_initial_conditions``) in S1
# (SHARC ``state 2 mch``, ``coeff auto``), dt = 0.5 au, 25 substeps, ``surf diagonal``,
# ``gradcorrect ngt``, ``parallel_vel``, frustrated hops kept, no decoherence; our side uses
# ``substep_interpolation = "endpoint"``. Our random numbers are SHARC's (output.dat block 9 of
# steps 1, 2, ...) through a ``ReplaySource``.
#
# Pre-registered (P7a-2 plan, Task 10): positions <= 1e-8 bohr and velocities <= 1e-10 au at
# every step; spin-summed MCH quantum populations |(U c)|^2 <= 1e-6 over the first 50 steps;
# E^diag <= the Task 9 print tolerance; U up to D and the best diagonal phase <= 1e-7; with a
# hop: the same active diagonal states, probabilities <= 1e-5, velocities <= 1e-9.
#
# Measured black-box behaviour that the plan did not foresee (Task 10 report), and the two
# comparisons changed AFTER the numbers were seen (marked below):
# - sharc.x hands its interface the geometry in angstrom rounded to 7 decimals (up to 9e-8
#   bohr off its own positions), so SHARC's H is H(rounded q): our E^diag along our own path
#   differs by up to 2.1e-8 Eh. The E^diag assertion uses our model at that interface
#   geometry, round(q_SHARC [angstrom], 7) (measured 3.2e-10); the core's E^diag along our own
#   path is reported, not asserted.
# - sharc.x adds 1e-12 Eh (M_s = 0) and 2e-12 Eh (M_s = +1) to the triplet diagonal of H, and
#   lvc_soc has diagonal states only 1.7e-8 Eh apart (two T2 components), so U is not
#   determined to 1e-7 by H in near-degenerate columns: the brief's metric
#   |(D U_ours) Phi - U_SHARC| is 1.4e-5 to 5.2e-5 there. It is printed per step, not asserted.
#   The U assertions (post hoc, rules fixed by the controller after the first numbers were
#   seen, fix round 1) use OUR model's H at the interface geometry, tracked by our
#   ``diagonal_basis`` over the run exactly as the core does (first-step rule, then the
#   previous U), with D applied: (i) the brief's column metric <= 1e-7 in the separated
#   columns (no neighbour within 1e-4 Eh, from SHARC's step-0 eigenvalues); (ii) for each
#   near-degenerate cluster, max |P_ours - P_SHARC| <= 1e-7 with P = sum of u u^dagger over the
#   cluster's columns; (iii) for ``nacdr`` only, the separated columns with Phi fixed at step 0
#   (not re-fitted) <= 1e-7 over the run (for ``overlap`` this drift is reported only).
# - A separate check: SHARC's U against the eigenvectors of SHARC's own printed H (our
#   ``diagonal_basis`` of that H, phase re-fitted per step). It only checks SHARC's ordering
#   convention (ascending eigenvectors of its H), not our U.

DT_AU = 0.5
INITIAL_MCH = 1  # S1
OFF_SEED = 1  # Wigner sample of Step 1
OFF_STEPS = 200
POP_STEPS = 50
HOP_SEED = 59  # Step 2: found by a search; sharc.x hops within 300 steps (first at step 67)
HOP_RNGSEED = 1
HOP_STEPS = 300
TOL_Q = 1e-8
TOL_V = 1e-10
TOL_POP = 1e-6
TOL_U = 1e-7
TOL_PROBABILITY = 1e-5
TOL_V_HOP = 1e-9


def _soc_model() -> LVCModel:
    return LVCModel.from_template(DATA / "lvc_soc.template")


def _options(coupling: str, hopping: str) -> DynamicsOptions:
    return DynamicsOptions(
        dt_au=DT_AU,
        nsubsteps=25,
        hopping=hopping,
        decoherence="none",
        edc_convention="sharc",
        substep_interpolation="endpoint",
        coupling=coupling,
    )


def _wigner_sample(model: LVCModel, seed: int) -> tuple[np.ndarray, np.ndarray]:
    q, v = lvc_initial_conditions(model, 1, seed, "wigner")
    return q[0], v[0]


def _sharc_trajectory(work, model, q0, v0, options, n_steps, rngseed):
    traj = lvc_tool.run_model_trajectory(
        work,
        model,
        DATA / "lvc_soc.template",
        q0,
        v0,
        options,
        n_steps,
        rngseed,
        sharc_home=Path(os.environ["SHARC_HOME"]),
        sharc_python=os.environ["SHARC_PYTHON"],
        state_mch=INITIAL_MCH,
        timeout_s=3600.0,
    )
    assert traj.steps.tolist() == list(range(n_steps + 1)), traj.steps[-3:]
    return traj


def _our_trajectory(model, q0, v0, options, random_numbers):
    state = initial_state(model, q0, v0, INITIAL_MCH, options=options)
    records: list = []
    run_trajectory(
        model,
        state,
        options,
        ReplaySource(list(random_numbers)),
        max_steps=len(random_numbers),
        on_step=records.append,
    )
    return state, records


def _interface_geometry(q_bohr: np.ndarray) -> np.ndarray:
    """The geometry sharc.x writes to QM.in: angstrom with 7 decimals (measured)."""
    angstrom = lvc_tool.BOHR_IN_ANGSTROM
    return np.round(np.asarray(q_bohr) * angstrom, 7) / angstrom


def _best_phases(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The diagonal phases Phi minimising |a Phi - b| column by column."""
    z = np.sum(a.conj() * b, axis=0)
    return z / np.abs(z)


CLUSTER_GAP_EH = 1e-4  # diagonal states closer than this to a neighbour form one cluster


def _clusters(energies: np.ndarray) -> list[list[int]]:
    """Index lists of consecutive ascending energies chained by gaps < CLUSTER_GAP_EH."""
    groups = [[0]]
    for b in range(1, len(energies)):
        if energies[b] - energies[b - 1] < CLUSTER_GAP_EH:
            groups[-1].append(b)
        else:
            groups.append([b])
    return groups


def _compare_hopping_off(model: LVCModel, sharc, state, records) -> dict:
    n_out = len(records) + 1
    q = np.vstack([state.q] + [r.q for r in records])
    v = np.vstack([state.v] + [r.v for r in records])
    c = [state.c] + [r.c for r in records]
    u = [state.u] + [r.u for r in records]
    e_diag = [diagonal_energies(state.snapshot.hamiltonian, state.u)]
    e_diag += [r.energies_diag for r in records]
    _, index, choices = _sign_choices(model)
    spin_sum = np.zeros((model.n_states, int(index.max()) + 1))
    spin_sum[np.arange(model.n_states), index] = 1.0
    gaps = np.diff(np.linalg.eigvalsh(sharc.hamiltonian[0]))
    near = np.zeros(model.n_states, dtype=bool)  # columns with a neighbour closer than 1e-4 Eh
    near[:-1] |= gaps < 1e-4
    near[1:] |= gaps < 1e-4
    out: dict[str, list] = {key: [] for key in ("dpop", "dE_path", "dE", "dU", "dU_sep")}
    out.update({"dU_near": [], "dU_from_sharc_H": [], "phase_drift": [], "D": []})
    out.update({"iU_sep": [], "iU_fixed_phase": [], "clusters_same": []})
    clusters = _clusters(np.linalg.eigvalsh(sharc.hamiltonian[0]))
    near_clusters = [cl for cl in clusters if len(cl) > 1]
    out["projector"] = {str(cl): [] for cl in near_clusters}
    snap = None
    u_iface = None  # our U at the interface geometry, tracked as the core does
    u_sharc_h = None
    phi0 = None
    phi0_iface = None
    for k in range(n_out):
        h_s = sharc.hamiltonian[k]
        e_s = np.linalg.eigvalsh(h_s)
        snap = model.evaluate(_interface_geometry(sharc.q[k]), snap)
        residuals = [
            float(np.max(np.abs(np.outer(d, d) * snap.hamiltonian - h_s))) for d in choices
        ]
        d = choices[int(np.argmin(residuals))]
        out["D"].append(tuple(d.tolist()))
        _, u_iface = diagonal_basis(snap.hamiltonian, u_iface, first_step=(k == 0))
        di = d[:, None] * u_iface
        phi_i = _best_phases(di, sharc.u[k])
        phi0_iface = phi_i if phi0_iface is None else phi0_iface
        sep = ~near
        out["iU_sep"].append(
            float(np.max(np.abs(di[:, sep] * phi_i[None, sep] - sharc.u[k][:, sep])))
        )
        out["iU_fixed_phase"].append(
            float(np.max(np.abs(di[:, sep] * phi0_iface[None, sep] - sharc.u[k][:, sep])))
        )
        for cl in near_clusters:
            p_ours = di[:, cl] @ di[:, cl].conj().T
            p_sh = sharc.u[k][:, cl] @ sharc.u[k][:, cl].conj().T
            out["projector"][str(cl)].append(float(np.max(np.abs(p_ours - p_sh))))
        out["clusters_same"].append(_clusters(e_s) == clusters)
        out["dE"].append(float(np.max(np.abs(np.linalg.eigvalsh(snap.hamiltonian) - e_s))))
        out["dE_path"].append(float(np.max(np.abs(e_diag[k] - e_s))))
        p_ours = (np.abs(u[k] @ c[k]) ** 2) @ spin_sum
        p_sharc = (np.abs(sharc.u[k] @ sharc.c[k]) ** 2) @ spin_sum
        out["dpop"].append(float(np.max(np.abs(p_ours - p_sharc))))
        du = d[:, None] * u[k]
        per_column = np.max(np.abs(du * _best_phases(du, sharc.u[k])[None, :] - sharc.u[k]), axis=0)
        out["dU"].append(float(per_column.max()))
        out["dU_sep"].append(float(per_column[~near].max()))
        out["dU_near"].append(float(per_column[near].max()) if near.any() else 0.0)
        _, u_sharc_h = diagonal_basis(h_s, u_sharc_h, first_step=(k == 0))
        phi = _best_phases(u_sharc_h, sharc.u[k])
        phi0 = phi if phi0 is None else phi0
        out["dU_from_sharc_H"].append(float(np.max(np.abs(u_sharc_h * phi[None, :] - sharc.u[k]))))
        out["phase_drift"].append(float(np.max(np.abs(phi - phi0))))
    out["dq"] = np.max(np.abs(q - sharc.q), axis=1).tolist()
    out["dv"] = np.max(np.abs(v - sharc.v), axis=1).tolist()
    out["tol_E"] = [print_tolerance(np.linalg.eigvalsh(h)) for h in sharc.hamiltonian]
    out["near_columns"] = np.flatnonzero(near).tolist()
    out["min_gap"] = float(gaps.min())
    return out


def _report_hopping_off(coupling: str, r: dict, sharc, state) -> None:
    print(f"\n[lvc_soc hopping off, {coupling}] {len(r['dq']) - 1} steps")
    print(
        f"  start: SHARC diag {sharc.active[0]} / MCH {sharc.active_mch[0]}; ours lambda "
        f"{state.active} / MCH {mch_state(state.u, state.active)} (0-based)"
    )
    print(f"  max|dq| {max(r['dq']):.2e} bohr, max|dv| {max(r['dv']):.2e} au")
    print(
        f"  max|d pop| first {POP_STEPS} steps {max(r['dpop'][: POP_STEPS + 1]):.2e}, "
        f"all {max(r['dpop']):.2e}"
    )
    print(
        f"  max|dE_diag| at the interface geometry {max(r['dE']):.2e} (tol "
        f"{min(r['tol_E']):.0e}); along our path {max(r['dE_path']):.2e}"
    )
    print(
        f"  D constant: {len(set(r['D'])) == 1}; near-degenerate columns {r['near_columns']} "
        f"(smallest gap {r['min_gap']:.2e} Eh)"
    )
    print(
        f"  U (brief): max {max(r['dU']):.2e}; separated columns {max(r['dU_sep']):.2e}; "
        f"near-degenerate {max(r['dU_near']):.2e}"
    )
    print(
        f"  SHARC's U vs eigenvectors of its printed H: max {max(r['dU_from_sharc_H']):.2e}; "
        f"phase drift {max(r['phase_drift']):.2e}"
    )
    print(
        f"  our U at the interface geometry (tracked): (i) separated columns "
        f"{max(r['iU_sep']):.2e}; (iii) Phi fixed at step 0 {max(r['iU_fixed_phase']):.2e}; "
        f"clusters as at step 0 at every step: {all(r['clusters_same'])}"
    )
    for cluster, values in r["projector"].items():
        print(f"  (ii) projector, cluster {cluster}: max {max(values):.2e}")
    print("  per step (k: dU brief / brief separated / (i) / (ii) per cluster / (iii)):")
    for k in range(0, len(r["dU"]), 10):
        proj = " ".join(f"{v[k]:.2e}" for v in r["projector"].values())
        print(
            f"    {k:4d}: {r['dU'][k]:.2e} / {r['dU_sep'][k]:.2e} / {r['iU_sep'][k]:.2e} / "
            f"{proj} / {r['iU_fixed_phase'][k]:.2e}"
        )


_TRAJ_CACHE: dict[str, tuple] = {}


@pytest.fixture(scope="module", params=["nacdr", "overlap"])
def hopping_off(request, tmp_path_factory):
    coupling = request.param
    if coupling not in _TRAJ_CACHE:
        model = _soc_model()
        q0, v0 = _wigner_sample(model, OFF_SEED)
        options = _options(coupling, "none")
        work = tmp_path_factory.mktemp(f"sharc-off-{coupling}") / "traj"
        sharc = _sharc_trajectory(work, model, q0, v0, options, OFF_STEPS, 1)
        state, records = _our_trajectory(model, q0, v0, options, sharc.random_numbers[1:])
        result = _compare_hopping_off(model, sharc, state, records)
        result["coupling"] = coupling
        _report_hopping_off(coupling, result, sharc, state)
        _TRAJ_CACHE[coupling] = (sharc, state, records, result)
    return _TRAJ_CACHE[coupling]


@pytest.mark.slow
def test_initial_state_follows_the_mch_state(hopping_off):
    """SHARC's ``state 2 mch`` + ``coeff auto``: c^diag = U^dagger e_S1 and the active diagonal
    state is the one with the largest weight, as our initial_state (ruling R7)."""
    sharc, state, _, result = hopping_off
    assert sharc.active[0] == state.active
    assert sharc.active_mch[0] == INITIAL_MCH == mch_state(state.u, state.active)
    assert result["dpop"][0] <= TOL_POP


@pytest.mark.slow
def test_hopping_off_positions_and_velocities(hopping_off):
    _, _, _, r = hopping_off
    assert max(r["dq"]) <= TOL_Q, max(r["dq"])
    assert max(r["dv"]) <= TOL_V, max(r["dv"])


@pytest.mark.slow
def test_hopping_off_mch_populations(hopping_off):
    _, _, _, r = hopping_off
    assert max(r["dpop"][: POP_STEPS + 1]) <= TOL_POP, max(r["dpop"][: POP_STEPS + 1])


@pytest.mark.slow
def test_hopping_off_diagonal_energies(hopping_off):
    """Changed after the numbers were seen: at the geometry SHARC's interface received. The
    core's E^diag along our own path is reported, not asserted."""
    _, _, _, r = hopping_off
    for k, (de, tol) in enumerate(zip(r["dE"], r["tol_E"], strict=True)):
        assert de <= tol, (k, de, tol)


@pytest.mark.slow
def test_sharc_u_is_the_eigenbasis_of_its_printed_h(hopping_off):
    """SHARC's U against the ascending eigenvectors of SHARC's own printed H (phase re-fitted
    per step): checks SHARC's ordering convention only, not our U. Also: the sign matrix D
    fitted from H is the same at every step."""
    _, _, _, r = hopping_off
    assert len(set(r["D"])) == 1, set(r["D"])
    assert max(r["dU_from_sharc_H"]) <= TOL_U, max(r["dU_from_sharc_H"])


@pytest.mark.slow
def test_our_u_in_the_separated_columns(hopping_off):
    """Post hoc (fix round 1, (i)): our U (our H at the interface geometry, tracked by our
    diagonal_basis as the core does, D applied) against SHARC's U in the columns without a
    neighbour within 1e-4 Eh, best diagonal phase per step."""
    _, _, _, r = hopping_off
    assert max(r["iU_sep"]) <= TOL_U, max(r["iU_sep"])


@pytest.mark.slow
def test_our_cluster_projectors(hopping_off):
    """Post hoc (fix round 1, (ii)): for each near-degenerate cluster, the projector onto our
    columns (D applied) against SHARC's, max-abs element, every step; the cluster structure is
    that of SHARC's step-0 eigenvalues at every step."""
    _, _, _, r = hopping_off
    assert all(r["clusters_same"])
    for cluster, values in r["projector"].items():
        assert max(values) <= TOL_U, (cluster, max(values))


@pytest.mark.slow
def test_our_u_phase_tracking(hopping_off):
    """Post hoc (fix round 1, (iii)): with ``nacdr``, our separated columns with Phi fixed at
    step 0 (not re-fitted) stay within 1e-7 of SHARC's U over the run. With ``overlap`` the
    drift is reported only."""
    _, _, _, r = hopping_off
    if r["coupling"] != "nacdr":
        pytest.skip("for coupling = overlap the phase drift is reported, not asserted")
    assert max(r["iU_fixed_phase"]) <= TOL_U, max(r["iU_fixed_phase"])


@pytest.mark.slow
def test_replayed_hop(tmp_path):
    """Hopping on (nacdr): with SHARC's random numbers our core takes the same hops."""
    model = _soc_model()
    q0, v0 = _wigner_sample(model, HOP_SEED)
    options = _options("nacdr", "sharc")
    sharc = _sharc_trajectory(tmp_path / "traj", model, q0, v0, options, HOP_STEPS, HOP_RNGSEED)
    state, records = _our_trajectory(model, q0, v0, options, sharc.random_numbers[1:])
    active = np.array([state.active] + [r.active for r in records])
    v = np.vstack([state.v] + [r.v for r in records])
    p_ours = np.array([r.probabilities for r in records])
    sharc_hops = [
        (int(sharc.steps[k]), int(sharc.active[k - 1]), int(sharc.active[k]))
        for k in range(1, len(sharc.steps))
        if sharc.active[k] != sharc.active[k - 1]
    ]
    our_hops = [(r.step, r.hop_from, r.hop_to) for r in records if r.hop_to is not None]
    dv = np.max(np.abs(v - sharc.v), axis=1)
    dp = float(np.max(np.abs(p_ours - sharc.probabilities[1:])))
    print(f"\n[lvc_soc replayed hop] Wigner seed {HOP_SEED}, SHARC rngseed {HOP_RNGSEED}")
    print(
        f"  SHARC hops {sharc_hops}; ours {our_hops}; frustrated (ours) "
        f"{sum(r.frustrated for r in records)}"
    )
    print(
        f"  max|dp| {dp:.2e}; max|dv| {dv.max():.2e} (at the hops: "
        f"{[f'{dv[s]:.1e}' for s, _, _ in our_hops]})"
    )
    assert sharc_hops, "SHARC did not hop within the run"
    assert our_hops == sharc_hops
    assert np.array_equal(active, sharc.active)
    assert dp <= TOL_PROBABILITY
    assert dv.max() <= TOL_V_HOP
