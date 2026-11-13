"""Start-solution survey (P9a plan 2, Task 5): which SA-CASSCF solutions exist at each start.

For the SA3 settings of a project, every choice of the active orbitals from a window around the
Fermi level is solved fresh at each selected start geometry and at the equilibrium geometry, and
the distinct solutions are listed with the one our run used, the one SHARC used, the lowest in
energy and the one most like the equilibrium solution. No rule is chosen: the output feeds the
owner decision on a start-solution rule.

Example (WSL):
  python tools/diagnosis/start_solutions.py ~/p8_m4/run --variant-dir ~/p9a/variants/sa3_cas22 \
      --sharc ~/p9a/sharc/sa3_cas22 --out ~/p9a/plan2/start_solutions.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import os
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sharc_compare"))
from forensics import find_stage_dir, jsonable, read_rows  # noqa: E402
from project_inputs import load_project_inputs  # noqa: E402
from replay_points import active_orbital_overlap, natural_occupations  # noqa: E402
from run_variants import deep_merge  # noqa: E402

from gqteashmd.config.schema import ElectronicSection  # noqa: E402

REFERENCE_GUESS_ACTIVE = (7, 8)  # HOMO, LUMO of ethylene (0-based)
N_STATES = 3


def guess_pairs(n_occupied: int, ncas: int, nelecas: int) -> list[tuple[int, ...]]:
    """Sorted 0-based active-orbital choices from a window around the Fermi level."""
    n_occ_act = nelecas // 2
    n_virt_act = ncas - n_occ_act
    if n_occ_act < 1 or n_virt_act < 1 or n_occupied < n_occ_act + 1:
        raise ValueError(
            f"bad active space: n_occupied={n_occupied}, ncas={ncas}, nelecas={nelecas}"
        )
    occ_window = range(n_occupied - n_occ_act - 1, n_occupied)
    virt_window = range(n_occupied, n_occupied + n_virt_act + 1)
    return [
        tuple(sorted((*o, *v)))
        for o in itertools.combinations(occ_window, n_occ_act)
        for v in itertools.combinations(virt_window, n_virt_act)
    ]


def distinct_solutions(entries: Sequence[dict[str, Any]], tol_Eh: float = 1e-6) -> list[dict]:
    """Cluster the converged guesses by their state energies (max |dE| <= tol_Eh)."""
    clusters: list[dict[str, Any]] = []
    for e in entries:
        en = e.get("energies_Eh")
        if en is None:
            continue
        en = np.asarray(en, dtype=float)
        for c in clusters:
            if en.shape == c["_e"].shape and float(np.max(np.abs(en - c["_e"]))) <= tol_Eh:
                c["guesses"].append(e["guess"])
                break
        else:
            clusters.append({"guesses": [e["guess"]], "_e": en})
    out = [
        {
            "guesses": c["guesses"],
            "energies_Eh": c["_e"].tolist(),
            "sa_energy_Eh": float(np.mean(c["_e"])),
        }
        for c in clusters
    ]
    return sorted(out, key=lambda s: s["sa_energy_Eh"])


def match_solution(
    energies_Eh: Sequence[float], solutions: Sequence[dict], tol_Eh: float = 1e-4
) -> int | None:
    """Index of the solution whose state energies equal `energies_Eh` within tol, else None."""
    en = np.asarray(energies_Eh, dtype=float)
    best, best_d = None, tol_Eh
    for i, s in enumerate(solutions):
        ref = np.asarray(s["energies_Eh"], dtype=float)
        if ref.shape != en.shape:
            continue
        d = float(np.max(np.abs(en - ref)))
        if d <= best_d:
            best, best_d = i, d
    return best


def _build(symbols, charge, q, electronic: ElectronicSection, memory_mb: int):
    from pyscf import gto, scf

    mol = gto.Mole()
    mol.verbose = 0
    mol.max_memory = memory_mb
    mol.build(
        atom=[(s, tuple(float(x) for x in xyz)) for s, xyz in zip(symbols, q, strict=True)],
        unit="Bohr",
        basis=electronic.basis,
        charge=charge,
        spin=0,
        dump_input=False,
        parse_arg=False,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = electronic.scf_conv_tol
    mf.max_cycle = electronic.scf_max_cycle
    mf.chkfile = None
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    return mol, mf


def _solve(mol, mf, electronic: ElectronicSection, pair: tuple[int, ...]):
    from pyscf import fci, mcscf

    cas = electronic.casscf
    assert cas is not None
    mc = mcscf.CASSCF(mf, cas.ncas, cas.nelecas)
    mc.fcisolver = fci.solver(mol, singlet=True)
    mc = mc.state_average_([1.0 / N_STATES] * N_STATES)
    mc.conv_tol = cas.conv_tol
    mc.conv_tol_grad = cas.conv_tol_grad
    mo0 = mcscf.sort_mo(mc, mf.mo_coeff, [i + 1 for i in pair])  # 1-based
    mc.kernel(mo_coeff=mo0)
    return mc


def survey_geometry(
    symbols,
    charge,
    q,
    electronic: ElectronicSection,
    *,
    reference: tuple[Any, Any] | None,
    memory_mb: int,
) -> tuple[dict[str, Any], tuple[Any, Any] | None]:
    """Solve every guess at q: the record and the (MOs, mol) of the (7, 8) solution."""
    import pyscf.fci.direct_spin1
    from pyscf import mcscf

    cas = electronic.casscf
    assert cas is not None
    mol, mf = _build(symbols, charge, q, electronic, memory_mb)
    guesses: list[dict[str, Any]] = []
    own_ref = None
    ref_active = None
    if reference is not None:
        ref_mo, ref_mol = reference
        probe = mcscf.CASSCF(mf, cas.ncas, cas.nelecas)
        proj = mcscf.project_init_guess(probe, ref_mo, ref_mol)
        ncore = (mol.nelectron - cas.nelecas) // 2
        ref_active = proj[:, ncore : ncore + cas.ncas]
    s_ao = mol.intor("int1e_ovlp")
    for pair in guess_pairs(mol.nelectron // 2, cas.ncas, cas.nelecas):
        entry: dict[str, Any] = {"guess": pair, "converged": False, "energies_Eh": None}
        started = time.perf_counter()
        try:
            mc = _solve(mol, mf, electronic, pair)
            entry["converged"] = bool(mc.converged)
            if mc.converged:
                entry["energies_Eh"] = [float(e) for e in mc.e_states]
                entry["sa_energy_Eh"] = float(np.mean(mc.e_states))
                entry["occupations"] = [
                    natural_occupations(
                        pyscf.fci.direct_spin1.make_rdm1(ci, mc.ncas, mc.nelecas)
                    ).tolist()
                    for ci in mc.ci
                ]
                c_act = np.asarray(mc.mo_coeff)[:, mc.ncore : mc.ncore + mc.ncas]
                if ref_active is not None:
                    entry["active_overlap_sv"] = active_orbital_overlap(
                        ref_active, c_act, s_ao
                    ).tolist()
                if pair == REFERENCE_GUESS_ACTIVE:
                    own_ref = (np.asarray(mc.mo_coeff), mol)
        except Exception as exc:  # noqa: BLE001 - a report tool records the failure
            entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["wall_time_s"] = time.perf_counter() - started
        guesses.append(entry)
    solutions = distinct_solutions(guesses)
    for s in solutions:
        svs = [
            g["active_overlap_sv"]
            for g in guesses
            if g["guess"] in s["guesses"] and "active_overlap_sv" in g
        ]
        s["min_active_sv"] = float(min(sv[-1] for sv in svs)) if svs else None
    with_sv = [i for i, s in enumerate(solutions) if s["min_active_sv"] is not None]
    record = {
        "guesses": guesses,
        "solutions": solutions,
        "lowest_energy_solution": 0 if solutions else None,
        "most_like_equilibrium_solution": (
            max(with_sv, key=lambda i: solutions[i]["min_active_sv"]) if with_sv else None
        ),
    }
    return record, own_ref


def _our_step0(variant_dir: Path, task: str) -> list[float]:
    rows, _, _ = read_rows(variant_dir / task)
    return [float(x) for x in rows["energies"][0][:N_STATES]]


def _sharc_step0(sharc_dir: Path, task: str) -> list[float]:
    from run_sharc_lvc import parse_lvc_output_dat
    from run_sharc_pyscf_dynamics import trajectory_data

    data = trajectory_data(parse_lvc_output_dat(sharc_dir / task / "output.dat"))
    return [float(x) for x in data["energies"][0][:N_STATES]]


def _equilibrium(project: Path) -> tuple[np.ndarray, str]:
    """The geometry the frequencies stage used (working_geometry's rule), in bohr."""
    from gqteashmd.stages.geometry import prepared_molecule

    _, prepared = prepared_molecule(find_stage_dir(project, "prepare"))
    try:
        opt = find_stage_dir(project, "optimize")
    except ValueError:  # no optimize stage directory
        return prepared, "prepare"
    data = json.loads((opt / "optimize.json").read_text())
    return np.asarray(data["coords_bohr"], dtype=float), "optimize"


def survey_start(job: tuple) -> tuple[str, dict[str, Any]]:
    project, task, variant_dir, sharc_dir, memory_mb = job
    result: dict[str, Any] = {"task": task}
    try:
        inputs = load_project_inputs(Path(project))
        base = inputs.config.electronic
        assert base is not None
        electronic = ElectronicSection.model_validate(
            deep_merge(base.model_dump(mode="json"), {"singlets": N_STATES})
        )
        q_eq, eq_stage = _equilibrium(Path(project))
        eq_record, ref = survey_geometry(
            inputs.symbols, inputs.charge, q_eq, electronic, reference=None, memory_mb=memory_mb
        )
        if ref is None:
            raise RuntimeError("the equilibrium (7, 8) guess did not converge")
        if task == "equilibrium":
            record, _ = survey_geometry(
                inputs.symbols, inputs.charge, q_eq, electronic, reference=ref, memory_mb=memory_mb
            )
            result.update(geometry=eq_stage, **record)
            return task, result
        start = next(s for s in inputs.starts if s.task_id == task)
        q = np.asarray(start.q, dtype=float).reshape(-1, 3)
        record, _ = survey_geometry(
            inputs.symbols, inputs.charge, q, electronic, reference=ref, memory_mb=memory_mb
        )
        result.update(initial_state=start.state, **record)
        for key, fn, directory in (
            ("ours", _our_step0, variant_dir),
            ("sharc", _sharc_step0, sharc_dir),
        ):
            try:
                en = fn(Path(directory), task)
                result[key] = {
                    "energies_Eh": en,
                    "solution": match_solution(en, record["solutions"]),
                }
            except Exception as exc:  # noqa: BLE001
                result[key] = {"error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - recorded, the other starts still run
        result["error"] = f"{type(exc).__name__}: {exc}"
    return task, result


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("project", type=Path)
    ap.add_argument("--variant-dir", type=Path, required=True)
    ap.add_argument("--sharc", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--memory-mb", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    project = args.project.expanduser()
    inputs = load_project_inputs(project)
    tasks = ["equilibrium", *[s.task_id for s in inputs.starts]]
    jobs = [
        (
            str(project),
            t,
            str(args.variant_dir.expanduser()),
            str(args.sharc.expanduser()),
            args.memory_mb,
        )
        for t in tasks
    ]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    if args.workers > 1:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            results = dict(pool.map(survey_start, jobs))
    else:
        results = dict(map(survey_start, jobs))
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(jsonable({"tasks": results}), indent=1, allow_nan=False), encoding="utf-8"
    )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
