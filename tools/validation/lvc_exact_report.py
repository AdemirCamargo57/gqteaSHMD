"""Report-only FSSH-versus-exact populations on the two-mode LVC model (not a test; nothing is
asserted).

Model: ``tests/data/lvc/lvc_twomode.template`` (2 singlets + 1 triplet; modes 7 and 9 coupled,
constant S1-T1 spin-orbit coupling). Exact reference: the split-operator solver of
``tests/validation/wavepacket2d.py`` (self-tested in ``tests/validation/test_wavepacket2d.py``),
started from the V0 ground state on the diabatic S1. Its grid (``wavepacket2d.GRID``) is
256 x 128 points on Q_7 in [-14, 18), Q_9 in [-8, 8) (dQ = 0.125). The box is asymmetric because
the packet swings to Q_7 = -8.6 on S1 and out to +14.4 on S0. The MCH projection is made on a
2x Fourier-interpolated grid because near the conical intersection it converges only as dQ^2,
while the wavefunction is converged (diabatic populations to 2e-11).

FSSH: ``run_population_ensemble`` on the full Cartesian model with ``hopping = "sharc"``, EDC
with ``edc_convention = "sharc"`` (C = 0.1 Eh), ``parallel_vel``, frustrated hops kept,
``coupling = "nacdr"``, dt = 0.5 au, 25 substeps; n T = 0 Wigner samples from the runner's
``model-initial`` stream (``lvc_initial_conditions``, seed 20261108); initial MCH state S1.
Populations every 40 steps (20 au) to 4000 au.

Compared: the spin-summed MCH populations of S0, S1 and T1, FSSH quantum (|U c|^2) and
classical (|U_alpha,lambda|^2) against exact, at every output time; the JSON holds every time
series and the maximum absolute differences. FSSH is not exact here (the S0/S1 conical
intersection at Q_7 = -10, Q_9 = 0), so nothing is asserted. Runs in WSL from the repository
root (about 21 min with 8 workers on the development workstation):

    python tools/validation/lvc_exact_report.py --out ~/p7a2_reports/lvc_exact.json --workers 8

``--n`` and ``--t-final`` shrink the run for a smoke test.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from gqteashmd.dynamics.ensemble import lvc_initial_conditions, run_population_ensemble
from gqteashmd.dynamics.lvc import LVCModel
from gqteashmd.dynamics.lvc_format import read_lvc_template
from gqteashmd.dynamics.options import DynamicsOptions

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "tests" / "data" / "lvc" / "lvc_twomode.template"
SOLVER = ROOT / "tests" / "validation" / "wavepacket2d.py"
SEED = 20261108
DT = 0.5
RECORD_EVERY = 40
INITIAL_STATE = 1  # MCH S1 (expanded index); the exact packet starts on the diabatic S1


def load_solver():
    name = "gqteashmd_validation_wavepacket2d"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SOLVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def exact_populations(t_final: float) -> dict:
    wp = load_solver()
    model = wp.two_mode_lvc(read_lvc_template(TEMPLATE))
    start = time.perf_counter()
    result = wp.propagate(model, t_final=t_final, dt=DT, record_every=RECORD_EVERY, **wp.GRID)
    seconds = time.perf_counter() - start
    return {
        "labels_spin_free": list(model.labels_spin_free),
        "labels_expanded": list(model.labels_expanded),
        "grid": {key: list(value) for key, value in wp.GRID.items()},
        "dt_au": DT,
        "times_au": result["times"].tolist(),
        "mch_spin_free": result["mch_spin_free"].tolist(),
        "mch_expanded": result["mch"].tolist(),
        "diabatic_expanded": result["diabatic"].tolist(),
        "max_norm_error": float(np.max(np.abs(result["norm"] - 1.0))),
        "max_energy_error_Eh": float(np.max(np.abs(result["energy"] - result["energy"][0]))),
        "energy_Eh": float(result["energy"][0]),
        "edge_norm": float(result["edge_norm"]),
        "wall_s": seconds,
    }


def fssh_populations(n: int, t_final: float, workers: int) -> dict:
    model = LVCModel.from_template(TEMPLATE)
    options = DynamicsOptions(
        dt_au=DT,
        nsubsteps=25,
        hopping="sharc",
        ekin_correction="parallel_vel",
        frustrated="keep",
        decoherence="edc",
        edc_convention="sharc",
    )
    q0, v0 = lvc_initial_conditions(model, n, SEED, "wigner")
    start = time.perf_counter()
    result = run_population_ensemble(
        model,
        q0,
        v0,
        initial_state=INITIAL_STATE,
        options=options,
        seed=SEED,
        n_steps=round(t_final / DT),
        record_every=RECORD_EVERY,
        workers=workers,
    )
    seconds = time.perf_counter() - start
    return {
        "labels_spin_free": list(result.labels_spin_free),
        "options": {
            "dt_au": DT,
            "nsubsteps": 25,
            "hopping": "sharc",
            "ekin_correction": "parallel_vel",
            "frustrated": "keep",
            "decoherence": "edc",
            "edc_convention": "sharc",
            "decoherence_param_Eh": options.decoherence_param_Eh,
            "coupling": options.coupling,
            "gradient_transformation": options.gradient_transformation,
        },
        "sampling": "wigner (T = 0, model-initial stream)",
        "seed": SEED,
        "initial_state": "S1 (MCH)",
        "n_trajectories": result.n_trajectories,
        "n_energy_drift": result.n_energy_drift,
        "n_hops": result.n_hops,
        "n_frustrated": result.n_frustrated,
        "times_au": result.times_au.tolist(),
        "mch_spin_free_quantum": result.mch_spin_free_quantum.tolist(),
        "mch_spin_free_classical": result.mch_spin_free_classical.tolist(),
        "diag_quantum": result.diag_quantum.tolist(),
        "diag_classical": result.diag_classical.tolist(),
        "labels_expanded": list(result.labels_expanded),
        "workers": workers,
        "wall_s": seconds,
    }


def differences(exact: dict, fssh: dict) -> dict:
    if not np.allclose(exact["times_au"], fssh["times_au"]):
        raise RuntimeError("the exact and FSSH output times differ")
    times = np.asarray(exact["times_au"])
    reference = np.asarray(exact["mch_spin_free"])
    out: dict = {}
    for kind in ("quantum", "classical"):
        diff = np.asarray(fssh[f"mch_spin_free_{kind}"]) - reference
        rows = {}
        for j, label in enumerate(exact["labels_spin_free"]):
            i = int(np.argmax(np.abs(diff[:, j])))
            rows[label] = {
                "max_abs_diff": float(abs(diff[i, j])),
                "at_time_au": float(times[i]),
                "fssh": float(np.asarray(fssh[f"mch_spin_free_{kind}"])[i, j]),
                "exact": float(reference[i, j]),
                "final_diff": float(diff[-1, j]),
            }
        rows["overall_max_abs_diff"] = float(np.max(np.abs(diff)))
        out[kind] = rows
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="JSON file to write")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--t-final", type=float, default=4000.0)
    args = parser.parse_args(argv)

    start = time.perf_counter()
    report: dict = {
        "schema_version": 1,
        "script": "tools/validation/lvc_exact_report.py",
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "model": "tests/data/lvc/lvc_twomode.template",
        "asserted": False,
    }
    report["exact"] = exact_populations(args.t_final)
    print(f"exact: {report['exact']['wall_s']:.0f} s, edge {report['exact']['edge_norm']:.1e}")
    report["fssh"] = fssh_populations(args.n, args.t_final, args.workers)
    print(
        f"FSSH: {report['fssh']['wall_s']:.0f} s, n = {args.n}, energy drift "
        f"{report['fssh']['n_energy_drift']}, hops {report['fssh']['n_hops']}"
    )
    report["differences"] = differences(report["exact"], report["fssh"])
    times = report["exact"]["times_au"]
    labels = report["exact"]["labels_spin_free"]
    for i in range(0, len(times), max(1, len(times) // 20)):
        ex = report["exact"]["mch_spin_free"][i]
        qu = report["fssh"]["mch_spin_free_quantum"][i]
        cl = report["fssh"]["mch_spin_free_classical"][i]
        print(
            f"t = {times[i]:6.0f}  "
            + "  ".join(
                f"{lab}: exact {ex[j]:.4f} quantum {qu[j]:.4f} classical {cl[j]:.4f}"
                for j, lab in enumerate(labels)
            )
        )
    for kind, rows in report["differences"].items():
        print(
            f"max |FSSH {kind} - exact|: {rows['overall_max_abs_diff']:.4f}  "
            + "  ".join(
                f"{lab} {rows[lab]['max_abs_diff']:.4f} (t = {rows[lab]['at_time_au']:.0f})"
                for lab in labels
            )
        )
    report["wall_s"] = time.perf_counter() - start
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport written to {args.out} ({report['wall_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
