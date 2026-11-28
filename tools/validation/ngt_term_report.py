"""The force term that ``gradient_transformation = "none"`` leaves out (dynamics with spin-orbit
coupling, spec 5.2, owner decision 3; plan ruling PR7).

With triplets the stage runs with ``gradient_transformation = "none"``: the core's MCH gradient
tensor (``dynamics/representation.py:gradient_tensor``) keeps only its diagonal (the spin-free
gradients, expanded over Ms), and the force on the active diagonal state lambda is
``-Re[U^dagger G U]_{lambda lambda}``. The ``"ngt"`` tensor adds ``G_ab = (E_b - E_a) K_ab``
off the diagonal, with the spin-free energies (ruling R3). Singlet-triplet NACs are zero by spin,
and triplet-triplet NACs are not available (PySCF 2.14), so this report measures the
**singlet-singlet** part only:

    d = Re[(U^dagger G_offdiag U)]_{lambda lambda},  G_offdiag = (E_b - E_a) K_ab, a, b singlets,

along our SA-CASSCF(4,3) 2S + 2T trajectory of the spin-orbit swap test (a) (thioformaldehyde,
6-31G, hopping off, steps 0-40, sample 0; ``tools/sharc_compare/run_sharc_soc_swap.py``), with
the CASSCF singlet-singlet NACs computed at every step for this measurement only
(``backend.compute_more(nacs=...)`` after each step's compute; the trajectory itself does not use
them). Per step: ``|d|``, the norm of the force actually used, and their ratio; then the maximum
and median of the ratio.

Report only: every number goes to the JSON file, nothing is asserted. Never imported by
gqteashmd. Usage (WSL):

  python tools/validation/ngt_term_report.py \\
      --samples ~/dynsoc_task9/samples/run/03_sampling/samples.h5 --sample 0 \\
      --work-dir ~/dynsoc_task10/run --out ~/dynsoc_task10/ngt_term.json \\
      [--reference ~/dynsoc_task9/reports/fix1/swap_a_casscf.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sharc_compare"))
from run_sharc_soc_swap import (  # noqa: E402
    INITIAL_STATE,
    STEPS_HOPPING_OFF,
    dynamics_options,
    jsonable,
    make_backend,
    masses_per_coordinate,
    read_start,
)

from gqteashmd.dynamics.records import ElectronicSnapshot  # noqa: E402
from gqteashmd.dynamics.representation import (  # noqa: E402
    diagonal_gradient,
    diagonal_tensor,
    gradient_tensor,
)
from gqteashmd.trajectory.spin_expansion import expanded_size  # noqa: E402

METHOD = "casscf"  # spec 5.2: the SA-CASSCF case (the only method with analytic NACs)


# --- the formula (helper tests: tests/validation/test_ngt_term_report.py) ----------------------


def singlet_nac_tensor(
    energies: np.ndarray, nacs: Mapping[tuple[int, int], np.ndarray], n: int, m: int
) -> np.ndarray:
    """The singlet-singlet part of the ``"ngt"`` off-diagonal tensor in the expanded basis, real
    (N, N, f): ``G_ab = (E_b - E_a) K_ab`` for singlets a != b (``K_ba = -K_ab``, so G is
    symmetric), every other entry 0. ``energies`` are the expanded (or spin-free) energies, whose
    first n entries are the singlets; ``nacs`` maps singlet index pairs (i, j), i < j, to K_ij."""
    e = np.asarray(energies, dtype=float)
    if not nacs:
        raise ValueError("no singlet-singlet NAC given")
    f = np.asarray(next(iter(nacs.values()))).size
    big_n = expanded_size(n, m)
    tensor = np.zeros((big_n, big_n, f))
    for (i, j), k in nacs.items():
        if not 0 <= i < j < n:
            raise ValueError(f"NAC pairs must be singlet indices i < j < {n}; got {(i, j)}")
        term = (e[j] - e[i]) * np.asarray(k, dtype=float).reshape(-1)
        tensor[i, j] = term
        tensor[j, i] = term  # (E_i - E_j) K_ji = (E_j - E_i) K_ij
    return tensor


def dropped_term(g_offdiag: np.ndarray, u: np.ndarray | None, active: int) -> np.ndarray:
    """``Re[(U^dagger G_offdiag U)]_{lambda lambda}`` (f,): the gradient term that "none" drops
    (the dropped force is its negative). The core's own ``diagonal_tensor``; with U None the
    diagonal of the off-diagonal tensor, i.e. zero."""
    return diagonal_gradient(diagonal_tensor(g_offdiag, u), active)


def used_force(snapshot: ElectronicSnapshot, u: np.ndarray | None, active: int) -> np.ndarray:
    """The force the core applies with ``"none"``: ``-Re[U^dagger G^MCH U]_{lambda lambda}``."""
    return -diagonal_gradient(diagonal_tensor(gradient_tensor(snapshot, "none"), u), active)


def step_entry(
    step: int,
    snapshot: ElectronicSnapshot,
    u: np.ndarray | None,
    active: int,
    nacs: Mapping[tuple[int, int], np.ndarray],
    n: int,
    m: int,
) -> dict[str, Any]:
    """One step's numbers: the norms of the dropped term and of the force used, their ratio,
    the singlet weights ``|U_a lambda|^2`` of the active diagonal state, and per NAC pair the
    spin-free gap and ``|K|``."""
    d = dropped_term(singlet_nac_tensor(snapshot.energies, nacs, n, m), u, active)
    force = used_force(snapshot, u, active)
    dropped_norm = float(np.linalg.norm(d))
    force_norm = float(np.linalg.norm(force))
    column = np.eye(expanded_size(n, m))[:, active] if u is None else np.asarray(u)[:, active]
    weights = np.abs(column[:n]) ** 2
    e = np.asarray(snapshot.energies, dtype=float)
    return {
        "step": int(step),
        "active": int(active),
        "dropped_norm": dropped_norm,
        "force_norm": force_norm,
        "ratio": dropped_norm / force_norm if force_norm > 0 else float("nan"),
        "singlet_weights": [float(w) for w in weights],
        "pairs": [
            {
                "pair": [int(i), int(j)],
                "gap_Eh": float(e[j] - e[i]),
                "nac_norm": float(np.linalg.norm(np.asarray(k, dtype=float))),
            }
            for (i, j), k in sorted(nacs.items())
        ],
    }


def summarise(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Max, median and the step of the max of the per-step ratio."""
    ratios = np.array([float(e["ratio"]) for e in entries])
    k = int(np.nanargmax(ratios))
    return {
        "n_steps": int(ratios.size),
        "max_ratio": float(ratios[k]),
        "step_of_max": int(entries[k]["step"]),
        "median_ratio": float(np.nanmedian(ratios)),
    }


# --- the run ------------------------------------------------------------------------------------


def _singlet_pairs(labels) -> frozenset:
    singlets = [s for s in labels if s.multiplicity == 1]
    return frozenset((a, b) for i, a in enumerate(singlets) for b in singlets[i + 1 :])


def _nacs(backend, labels) -> dict[tuple[int, int], np.ndarray]:
    """The phased singlet-singlet NACs at the geometry of the backend's last compute."""
    result = backend.compute_more(nacs=_singlet_pairs(labels))
    index = {label: i for i, label in enumerate(labels)}
    out = {}
    for (a, b), k in result.nacs.items():
        i, j = index[a], index[b]
        k = np.asarray(k, dtype=float).reshape(-1)
        if not np.all(np.isfinite(k)):
            raise ValueError(f"non-finite NAC {a}-{b}")
        out[(i, j) if i < j else (j, i)] = k if i < j else -k
    return out


def run(
    samples: Path, sample: int, n_steps: int, work_dir: Path, *, threads: int
) -> dict[str, Any]:
    """Our (a) trajectory (the driver step by step, the options of the swap test with hopping
    off) with the dropped term measured at steps 0..n_steps."""
    from gqteashmd.dynamics.driver import initial_state, step
    from gqteashmd.dynamics.randomness import ReplaySource
    from gqteashmd.dynamics.representation import diagonal_energies
    from gqteashmd.trajectory.model import BackendModel

    start = read_start(Path(samples), sample)
    options = dynamics_options(hopping="none")
    backend = make_backend(METHOD, start, Path(work_dir) / "scratch", threads=threads)
    model = BackendModel(backend, masses_per_coordinate(start), coupling="overlap")
    n, m = model.n_singlets, model.n_triplets
    labels = model.labels

    def measure(k: int, state, extra: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        nacs = _nacs(backend, labels)
        entry = step_entry(k, state.snapshot, state.u, state.active, nacs, n, m)
        e_kin = 0.5 * float(np.sum(model.masses * state.v**2))
        e_diag = diagonal_energies(state.snapshot.hamiltonian, state.u)
        entry.update(
            extra,
            nac_time_s=time.perf_counter() - started,
            active_spin_free=model.active_spin_free(state.u, state.active),
            e_total=float(e_diag[state.active]) + e_kin,
            q=state.q.tolist(),
            v=state.v.tolist(),
        )
        return entry

    started = time.perf_counter()
    backend.reset()
    model.active = INITIAL_STATE
    t0 = time.perf_counter()
    state = initial_state(
        model, start.q_bohr.reshape(-1), start.v_au.reshape(-1), INITIAL_STATE, options=options
    )
    entries = [measure(0, state, {"compute_time_s": time.perf_counter() - t0})]
    source = ReplaySource([0.0] * n_steps)  # one number per step; hopping off never uses it
    for k in range(1, n_steps + 1):
        model.active = state.active
        model.u = state.u
        t0 = time.perf_counter()
        state, _record = step(model, state, options, source)
        extra = {
            "compute_time_s": time.perf_counter() - t0,
            "min_singular_value": model.min_singular_value,
            "rung_changed": model.rung_changed,
            "casscf_rung": model.last_meta.get("casscf"),
        }
        entries.append(measure(k, state, extra))
        print(
            f"step {k:3d}: |d| {entries[-1]['dropped_norm']:.3e}, "
            f"|F| {entries[-1]['force_norm']:.3e}, ratio {entries[-1]['ratio']:.3e}",
            flush=True,
        )
    return {
        "method": METHOD,
        "sample": int(sample),
        "n_singlets": n,
        "n_triplets": m,
        "steps": [0, n_steps],
        "units": {"dropped_norm": "Eh/bohr", "force_norm": "Eh/bohr", "gap_Eh": "Eh"},
        "wall_time_s": time.perf_counter() - started,
        "summary": summarise(entries),
        "per_step": entries,
    }


def compare_with(report: dict[str, Any], reference: Path) -> dict[str, Any]:
    """Max |dq|, |dv|, |dE_tot| against the ``ours`` records of a swap (a) report (Task 9)."""
    ours = json.loads(Path(reference).read_text())["ours"]
    steps = list(ours["steps"])
    out: dict[str, float] = {"q": 0.0, "v": 0.0, "e_total": 0.0}
    for entry in report["per_step"]:
        i = steps.index(entry["step"])
        out["q"] = max(out["q"], float(np.max(np.abs(np.subtract(entry["q"], ours["q"][i])))))
        out["v"] = max(out["v"], float(np.max(np.abs(np.subtract(entry["v"], ours["v"][i])))))
        out["e_total"] = max(out["e_total"], abs(entry["e_total"] - float(ours["e_total"][i])))
    return {"reference": str(reference), "max_abs_difference": out}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--samples", type=Path, required=True, help="samples.h5 of Task 9")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--steps", type=int, default=STEPS_HOPPING_OFF)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--reference", type=Path, help="a swap (a) report JSON, to compare q, v")
    args = parser.parse_args(argv)
    report = run(args.samples, args.sample, args.steps, args.work_dir, threads=args.threads)
    if args.reference is not None:
        report["reference"] = compare_with(report, args.reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(jsonable(report), indent=1) + "\n")
    print(json.dumps(jsonable({"summary": report["summary"], **report.get("reference", {})})))


if __name__ == "__main__":
    main()
