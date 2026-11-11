"""LVC dynamics with spin-orbit coupling against SHARC 4.1 (sharc.x, black box): a report.

Not a test; nothing is asserted. The asserted comparisons are the single trajectories of
``tests/validation/test_sharc_lvc.py`` (hopping off with both couplings, a replayed hop).
SHARC is only executed, through ``run_sharc_lvc.py`` next to this file (our writers and
parsers; formats from the public SHARC 4.1 manual). Every number goes to the JSON file.

Two reports (model ``tests/data/lvc/lvc_soc.template``, start in S1, dt = 0.5 au, 25
substeps, ``surf diagonal``, ``gradcorrect ngt``, ``parallel_vel``, frustrated hops kept; our
side with ``substep_interpolation = "endpoint"``):

- ``eselect``: the trajectory of the replayed-hop test (Wigner seed 59, SHARC rngseed 1, 300
  steps, ``nacdr``, no decoherence) with SHARC_LVC.py and SHARC's ``grad_select``,
  ``nac_select``, ``eselect 0.3`` (eV; ``select_directly``, SHARC's default, and
  ``noselect_directly``), against our core with ``eselect_Eh = 0.3 / HARTREE_IN_EV`` and
  SHARC's random numbers. Also the same SHARC run without selection, for scale.
- ``ensemble``: the swap test (strategy 5.4 (c)): sharc.x with our own QM.out provider (our
  ``LVCModel`` behind ``QM/runQM.sh``) against our core, n Wigner initial conditions shared
  by both programs (the ``model-initial`` stream of ``gqteashmd.dynamics.ensemble``), EDC
  with SHARC's convention (C = 0.1 Eh), ``nacdr``, 2000 au. SHARC gets its own seeds.
  Spin-summed MCH classical populations (sum over M_s of |U_alpha,lambda|^2) at 10 times with
  a pooled two-proportion z each. Trajectories whose |E_tot - E_tot(0)| exceeds 0.5 eV are
  excluded from the averages on both sides (ours: every step; SHARC: every line of
  output.lis) and counted. SHARC_LVC.py itself was not used for the ensemble: it costs
  0.68 s per step alone and 2.05 s per step for a batch of 14 parallel trajectories (P7a-2
  Task 10 report), so n = 100 would take about 16 h.

Runs in WSL in about 1 h (measured: eselect 14 min; ensemble 43 min, of which sharc.x with
12 threads 41 min, limited by the one Python process serving all of them), for example:

    SHARC_HOME=/home/ajc/sharc4-4.1 SHARC_PYTHON=/home/ajc/miniforge3/envs/sharc4.1/bin/python \\
        python tools/sharc_compare/lvc_ensembles.py --out ~/p7a2_reports/lvc_ensembles.json

``--n 4`` gives a smoke run of the ensemble; ``--reports eselect`` or ``ensemble`` runs one
part. The SHARC work directories go to a temporary directory outside the repository
(``--work-dir`` to choose one) and are removed afterwards.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from gqteashmd.core.units import HARTREE_IN_EV
from gqteashmd.dynamics import ensemble
from gqteashmd.dynamics.driver import initial_state, run_trajectory
from gqteashmd.dynamics.ensemble import lvc_initial_conditions, run_population_ensemble
from gqteashmd.dynamics.lvc import LVCModel
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.randomness import ReplaySource

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parents[1] / "tests" / "data" / "lvc" / "lvc_soc.template"
SEED = 20261110
INITIAL_MCH = 1  # S1
DT_AU = 0.5
ENSEMBLE_STEPS = 4000  # 2000 au
RECORD_EVERY = 400  # 10 time points after t = 0
HOP_SEED, HOP_RNGSEED, HOP_STEPS = 59, 1, 300  # the replayed-hop test of test_sharc_lvc.py
ESELECT_EV = 0.3
DRIFT_LIMIT_EV = 0.5
SHARC_WORKERS = 12
OUR_WORKERS = 8


def _load(name: str):
    if name in sys.modules:  # shared with the tests that load the harness by path
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lvc = _load("run_sharc_lvc")


def two_proportion_z(p1: float, n1: int, p2: float, n2: int) -> float:
    """Pooled two-proportion z of the fractions p1 (of n1) and p2 (of n2); 0 if the pooled
    variance is 0."""
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    var = pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2)
    return 0.0 if var <= 0.0 else (p1 - p2) / math.sqrt(var)


def _spin_sum(model: LVCModel) -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    for s in model.states:
        label = f"S{s.index}" if s.multiplicity == 1 else f"T{s.index + 1}"
        if label not in labels:
            labels.append(label)
    matrix = np.zeros((model.n_states, len(labels)))
    for i, s in enumerate(model.states):
        label = f"S{s.index}" if s.multiplicity == 1 else f"T{s.index + 1}"
        matrix[i, labels.index(label)] = 1.0
    return labels, matrix


# --- eselect -------------------------------------------------------------------------------------


def _hops(active: np.ndarray) -> list[tuple[int, int, int]]:
    return [
        (k, int(active[k - 1]), int(active[k]))
        for k in range(1, len(active))
        if active[k] != active[k - 1]
    ]


def eselect_report(sharc_home: Path, python: str, base: Path) -> dict:
    model = LVCModel.from_template(TEMPLATE)
    q0, v0 = lvc_initial_conditions(model, 1, HOP_SEED, "wigner")
    q0, v0 = q0[0], v0[0]
    plain = DynamicsOptions(
        dt_au=DT_AU,
        hopping="sharc",
        decoherence="none",
        substep_interpolation="endpoint",
    )
    selected = replace(plain, eselect_Eh=ESELECT_EV / HARTREE_IN_EV)
    runs = {}
    for name, options, extra in (
        ("no_selection", plain, None),
        ("select_directly", selected, None),
        ("noselect_directly", selected, "noselect_directly"),
    ):
        start = time.perf_counter()
        work = base / f"eselect-{name}"
        if extra is None:
            traj = lvc.run_model_trajectory(
                work,
                model,
                TEMPLATE,
                q0,
                v0,
                options,
                HOP_STEPS,
                HOP_RNGSEED,
                sharc_home=sharc_home,
                sharc_python=python,
            )
        else:  # the same input plus one keyword
            text = (
                lvc.lvc_input(
                    HOP_STEPS,
                    DT_AU * lvc.AU_TIME_IN_FS,
                    lvc.LvcSharcOptions.from_dynamics_options(options),
                    HOP_RNGSEED,
                    states=model.state_counts,
                    state_mch=INITIAL_MCH,
                )
                + f"{extra}\n"
            )
            template = TEMPLATE.read_text()
            traj = lvc.run_lvc_trajectory(
                work,
                sharc_home=sharc_home,
                provider="sharc_lvc",
                timeout_s=7200.0,
                symbols=model.parameters.v0.symbols,
                masses_amu=model.parameters.v0.masses_amu,
                q0=q0,
                v0=v0,
                input_text=text,
                template_text=lvc.without_dipoles("V0.txt\n" + template.split("\n", 1)[1]),
                v0_text=(TEMPLATE.parent / "V0.txt").read_text(),
                sharc_python=python,
            )
        wall = time.perf_counter() - start
        state = initial_state(model, q0, v0, INITIAL_MCH, options=options)
        records: list = []
        run_trajectory(
            model,
            state,
            options,
            ReplaySource(list(traj.random_numbers[1:])),
            max_steps=HOP_STEPS,
            on_step=records.append,
        )
        active = np.array([state.active] + [r.active for r in records])
        q = np.vstack([state.q] + [r.q for r in records])
        v = np.vstack([state.v] + [r.v for r in records])
        p = np.array([r.probabilities for r in records])
        n_common = (
            int(np.argmax(active != traj.active)) if np.any(active != traj.active) else len(active)
        )
        runs[name] = {
            "sharc_wall_s": wall,
            "sharc_hops": _hops(traj.active),
            "our_hops": _hops(active),
            "same_active_sequence": bool(np.array_equal(active, traj.active)),
            "steps_with_the_same_active_state_from_0": n_common,
            "max_abs_dq_bohr": float(np.max(np.abs(q - traj.q))),
            "max_abs_dv_au": float(np.max(np.abs(v - traj.v))),
            "max_abs_dprobability": float(np.max(np.abs(p - traj.probabilities[1:]))),
            "max_abs_dq_first_100_steps": float(np.max(np.abs(q[:101] - traj.q[:101]))),
            "sharc_q_final": traj.q[-1].tolist(),
        }
        print(
            f"[eselect {name}] SHARC {wall:.0f} s; "
            + json.dumps({k: runs[name][k] for k in runs[name] if k != "sharc_q_final"})
        )
    ref = np.array(runs["no_selection"]["sharc_q_final"])
    for name in ("select_directly", "noselect_directly"):
        runs[name]["sharc_final_q_vs_no_selection_bohr"] = float(
            np.max(np.abs(np.array(runs[name]["sharc_q_final"]) - ref))
        )
    return {
        "wigner_seed": HOP_SEED,
        "sharc_rngseed": HOP_RNGSEED,
        "n_steps": HOP_STEPS,
        "eselect_eV": ESELECT_EV,
        "eselect_Eh": ESELECT_EV / HARTREE_IN_EV,
        "runs": runs,
    }


# --- ensemble (swap test) ------------------------------------------------------------------------


def _ensemble_options() -> DynamicsOptions:
    return DynamicsOptions(
        dt_au=DT_AU,
        nsubsteps=25,
        hopping="sharc",
        ekin_correction="parallel_vel",
        frustrated="keep",
        decoherence="edc",
        decoherence_param_Eh=0.1,
        edc_convention="sharc",
        substep_interpolation="endpoint",
        coupling="nacdr",
    )


def sharc_swap_ensemble(model, q0, v0, sharc_home: Path, base: Path, workers: int) -> dict:
    options = _ensemble_options()
    n = q0.shape[0]
    seeds = np.random.SeedSequence([SEED, zlib.crc32(b"sharc-rngseed")]).generate_state(n)
    labels, spin_sum = _spin_sum(model)
    n_out = ENSEMBLE_STEPS // RECORD_EVERY + 1

    def one(i: int):
        work = base / f"traj{i:04d}"
        try:
            traj = lvc.run_model_trajectory(
                work,
                model,
                TEMPLATE,
                q0[i],
                v0[i],
                options,
                ENSEMBLE_STEPS,
                int(seeds[i]) % 2_000_000_000 + 1,
                sharc_home=sharc_home,
                provider="ours",
                detailed=False,
                output_stride=RECORD_EVERY,
            )
            _, e_tot = lvc.parse_output_lis(work / "output.lis")
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if traj.steps.tolist() != list(range(0, ENSEMBLE_STEPS + 1, RECORD_EVERY)):
            raise RuntimeError(f"trajectory {i}: unexpected printed steps {traj.steps.tolist()}")
        pops = np.zeros((n_out, 2, model.n_states))
        for k in range(n_out):
            lam = traj.active[k]
            pops[k, 0] = np.abs(traj.u[k][:, lam]) ** 2
            pops[k, 1] = np.abs(traj.u[k] @ traj.c[k]) ** 2
        drift = float(np.max(np.abs(e_tot - e_tot[0])))
        return pops, drift, len(_hops(np.array(traj.active)))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, range(n)))
    kept = [r for r in results if r[1] <= DRIFT_LIMIT_EV]
    mean = (
        np.mean([r[0] for r in kept], axis=0)
        if kept
        else np.full((n_out, 2, model.n_states), np.nan)
    )
    return {
        "labels": labels,
        "classical": mean[:, 0] @ spin_sum,
        "quantum": mean[:, 1] @ spin_sum,
        "n_included": len(kept),
        "n_energy_drift": len(results) - len(kept),
        "max_drift_eV": max(r[1] for r in results),
        "printed_hops_lower_bound": sum(r[2] for r in kept),
    }


def ensemble_report(n: int, sharc_home: Path, base: Path, sharc_workers: int) -> dict:
    model = LVCModel.from_template(TEMPLATE)
    assert zlib.crc32(b"model-initial") == ensemble._INITIAL_TAG
    q0, v0 = lvc_initial_conditions(model, n, SEED, "wigner")
    options = _ensemble_options()
    start = time.perf_counter()
    ours = run_population_ensemble(
        model,
        q0,
        v0,
        initial_state=INITIAL_MCH,
        options=options,
        seed=SEED,
        n_steps=ENSEMBLE_STEPS,
        record_every=RECORD_EVERY,
        workers=OUR_WORKERS,
    )
    t_ours = time.perf_counter() - start
    start = time.perf_counter()
    sharc = sharc_swap_ensemble(model, q0, v0, sharc_home, base / "swap", sharc_workers)
    t_sharc = time.perf_counter() - start
    labels = list(ours.labels_spin_free)
    assert labels == sharc["labels"], (labels, sharc["labels"])
    n_ours = ours.n_trajectories - ours.n_energy_drift
    n_sharc = sharc["n_included"]
    rows = []
    print(
        f"\n[ensemble, swap test] n = {n}; ours {t_ours:.0f} s (drift {ours.n_energy_drift}), "
        f"SHARC {t_sharc:.0f} s (drift {sharc['n_energy_drift']})"
    )
    print("  t/au   " + "   ".join(f"{lab}: ours SHARC z" for lab in labels))
    for k, t in enumerate(ours.times_au):
        if k == 0:
            continue
        cells = []
        for j, lab in enumerate(labels):
            p1 = float(ours.mch_spin_free_classical[k, j])
            p2 = float(sharc["classical"][k, j])
            z = two_proportion_z(p1, n_ours, p2, n_sharc)
            cells.append(
                {
                    "state": lab,
                    "ours": p1,
                    "sharc": p2,
                    "z": z,
                    "ours_quantum": float(ours.mch_spin_free_quantum[k, j]),
                    "sharc_quantum": float(sharc["quantum"][k, j]),
                }
            )
        rows.append({"time_au": float(t), "states": cells})
        print(
            f"  {t:6.0f} "
            + "  ".join(f"{c['ours']:.3f} {c['sharc']:.3f} {c['z']:+.2f}" for c in cells)
        )
    max_z = max(abs(c["z"]) for row in rows for c in row["states"])
    print(f"  max |z| {max_z:.2f}")
    return {
        "variant": "swap test: sharc.x with our QM.out provider (our LVCModel)",
        "n_trajectories": n,
        "seed": SEED,
        "t_max_au": ENSEMBLE_STEPS * DT_AU,
        "settings": "sharc hopping, EDC (sharc convention, C = 0.1 Eh), nacdr, parallel_vel, "
        "frustrated keep, dt 0.5 au, 25 substeps, ngt; ours endpoint",
        "ours": {
            "wall_s": t_ours,
            "n_energy_drift": ours.n_energy_drift,
            "n_hops": ours.n_hops,
            "n_frustrated": ours.n_frustrated,
        },
        "sharc": {
            "wall_s": t_sharc,
            "workers": sharc_workers,
            "n_energy_drift": sharc["n_energy_drift"],
            "max_drift_eV": sharc["max_drift_eV"],
            "active_changes_between_printed_steps": sharc["printed_hops_lower_bound"],
        },
        "rows": rows,
        "max_abs_z": max_z,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="JSON file to write")
    parser.add_argument("--sharc-home", default=os.environ.get("SHARC_HOME"))
    parser.add_argument("--python", default=os.environ.get("SHARC_PYTHON"))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--sharc-workers", type=int, default=SHARC_WORKERS)
    parser.add_argument(
        "--reports", nargs="+", choices=("eselect", "ensemble"), default=["eselect", "ensemble"]
    )
    parser.add_argument("--work-dir", type=Path, help="parent of the SHARC run directories")
    args = parser.parse_args(argv)
    if not args.sharc_home or not args.python:
        parser.error("needs --sharc-home and --python (or SHARC_HOME and SHARC_PYTHON)")
    sharc_home = Path(args.sharc_home)
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    base = Path(tempfile.mkdtemp(prefix="gqteashmd-sharc-lvc-", dir=args.work_dir))
    start = time.perf_counter()
    report: dict = {
        "schema_version": 1,
        "script": "tools/sharc_compare/lvc_ensembles.py",
        "started_at": started_at,
        "model": "tests/data/lvc/lvc_soc.template",
    }
    try:
        if "eselect" in args.reports:
            report["eselect"] = eselect_report(sharc_home, args.python, base)
        if "ensemble" in args.reports:
            report["ensemble"] = ensemble_report(args.n, sharc_home, base, args.sharc_workers)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    report["wall_s"] = time.perf_counter() - start
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport written to {args.out} ({report['wall_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
