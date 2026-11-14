"""SHARC 4.1's ``populations.py`` and ``make_fit.py`` against our analysis (P9b spec 3.3, Task 5).

SHARC is only *executed* (stdin answers, as in ``run_excite.py``); nothing from it is imported.
The prompts and answers are the Task 1 facts F2/F3, the environment F6, the fit-model ruling F4
(``make_fit.py`` cannot express a delay, so both sides fit ``y(t) = A exp(-t/tau)``, ``A = y(0)``,
to the summed excited-state population ``1 - P(S0)`` of the classical populations on SHARC's
printed time grid) and the quantum print bound F5 (``1e-9``, one unit of ``%17.9f``).

"Ours" here means our analysis functions applied to the SHARC trajectories read by
``sharc_trajectories.read_sharc_trajectory`` (all trajectories, no exclusion).

Example (WSL):
  python tools/sharc_compare/run_sharc_analysis.py ~/p9b/sharc/run PROJECT \\
      --sharc-home ~/sharc4-4.1 --python ~/miniforge3/envs/sharc4.1/bin/python \\
      --work ~/p9b/tools --out ~/p9b/compare/tools.json
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import socket
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.analysis.populations import (
    bootstrap_indices,
    classical_populations,
    quantum_populations,
)
from gqteashmd.util.seeds import task_rng

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_compare import COMPARE_STREAM, N_BOOTSTRAP, on_steps  # noqa: E402
from sharc_trajectories import read_sharc_trajectory  # noqa: E402

TIMEOUT_S = 3600.0
PARENT = "Singlet_1"  # the parent directory populations.py is asked for (F2)
CLASSICAL_OUTPUT = "pop_2.out"  # F2: mode 2, active MCH state
QUANTUM_OUTPUT = "pop_8.out"  # F2: mode 8, |c|^2 in the MCH basis (eq. 8.18)
FIT_OUTPUT = "fit_bootstrap.txt"  # F3: bootstrap analysis file
PRINT_DECIMALS = 9  # F2: %17.9f
QUANTUM_BOUND = 1e-9  # F5
FIT_FACTOR = 4.0
FIT_SEED = 2028
HINT = "check the answers in run_sharc_analysis.py against Task 1 facts F2/F3"


def sharc_tool_environment(sharc_home: Path, python: str | None = None) -> dict[str, str]:
    """F6: SHARC and PYTHONPATH as for excite.py; the analysis tools also need HOSTNAME/USER
    and ``data_extractor.x`` needs the SHARC conda env's library directory."""
    env = dict(os.environ)
    env["SHARC"] = str(sharc_home / "bin")
    # PYTHONPATH is replaced, not extended: SHARC's own conda environment must not see ours.
    env["PYTHONPATH"] = f"{sharc_home / 'lib'}:{sharc_home / 'pysharc'}"
    env.setdefault("HOSTNAME", socket.gethostname())
    env.setdefault("USER", getpass.getuser())
    if python is not None:
        lib = Path(python).parent.parent / "lib"
        old = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = f"{lib}:{old}" if old else str(lib)
    return env


def populations_answers(
    kind: str, *, n_states: int = 3, maxtime_fs: float = 5.0, bootstrap: bool = False
) -> list[str]:
    """F2: the stdin answers of populations.py (``classical`` = mode 2, ``quantum`` = mode 8)."""
    if kind not in ("classical", "quantum"):
        raise ValueError(f"unknown kind {kind!r} (classical or quantum)")
    answers = [
        PARENT,  # Path: the parent directory of the TRAJ_0XXXX directories
        "end",  # Path: no more directories
        "2" if kind == "classical" else "8",  # Analyze mode
    ]
    if kind == "quantum":
        answers.append("False")  # Run data_extractor.x? (we ran it ourselves, F6)
    answers += [
        str(n_states),  # Number of states
        "",  # Normalize the populations? (default True)
        str(maxtime_fs),  # Simulation time (fs)
        "True" if bootstrap else "",  # Save data for bootstrapping?
    ]
    if bootstrap:
        answers.append("")  # Directory for data? (default bootstrap_data/)
    answers += [
        "",  # Gnuplot script? (default False)
        "",  # Do you want to do the specified analysis? (default True)
    ]
    return answers


def fit_answers(
    populations_file: str,
    *,
    n_states: int = 3,
    bootstrap_samples: int | None = None,
    bootstrap_dir: str = "bootstrap_data/",
) -> list[str]:
    """F3: make_fit.py for the network S1 -> S0 (rate k10), S1(0) = 1 fixed, fitted to the sum of
    the excited-state columns (columns count the time column as 1, S0 = 2, S1 = 3, ...)."""
    excited = " ".join(str(c) for c in range(3, n_states + 2))
    answers = [
        "+ S0 S1",  # species
        "end",
        "+ S1 S0 k10",  # reaction S1 -> S0 with rate label k10
        "end",
        "+ S1",  # S1 has the non-zero initial population
        "end",
    ]
    if bootstrap_samples is None:
        answers.append("")  # Do you want to use bootstrap data? (default False)
        answers.append(populations_file)  # Populations file
    else:
        answers += [
            "True",  # Do you want to use bootstrap data?
            str(bootstrap_samples),  # How many bootstrap samples?
            bootstrap_dir,  # Bootstrap data directory
            "",  # Write fitting curves for all bootstrap cycles? (default False)
        ]
    answers += [
        f"S1 = {excited}",  # species-to-column mapping (S0 left unmapped)
        "end",
        "end",  # initial guesses: keep the defaults
        "",  # optimize the initial populations? (default False)
        "",  # restrict to non-negative? (default True)
        "",  # Do you want to continue? (default True)
    ]
    return answers


def run_tool(
    sharc_home: Path,
    python: str,
    script: str,
    work_dir: Path,
    answers: Sequence[str],
    *,
    timeout_s: float = TIMEOUT_S,
    outputs: Sequence[str] = (),
    stdout_name: str | None = None,
) -> subprocess.CompletedProcess:
    """Run ``<sharc_home>/bin/<script>`` in ``work_dir`` with the answers on stdin."""
    work_dir = Path(work_dir)
    stdin = "\n".join(answers) + "\n" + "\n" * 10
    try:
        proc = subprocess.run(
            [python, str(sharc_home / "bin" / script)],
            input=stdin,
            cwd=work_dir,
            env=sharc_tool_environment(sharc_home, python),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"{script} did not finish within {timeout_s} s in {work_dir}; {HINT}"
        ) from exc
    if stdout_name:
        (work_dir / stdout_name).write_text(proc.stdout)
    missing = [o for o in outputs if not (work_dir / o).exists()]
    if proc.returncode != 0 or missing:
        tail = (proc.stdout[-1500:] + "\n" + proc.stderr[-1500:]).strip()
        raise RuntimeError(
            f"{script} failed (exit {proc.returncode}, missing outputs {missing}) in "
            f"{work_dir}; {HINT}:\n{tail}"
        )
    return proc


def run_extractor(
    sharc_home: Path, python: str, traj_dir: Path, *, timeout_s: float = 600.0
) -> None:
    """F6: ``data_extractor.x output.dat`` by hand (the script's own call fails here)."""
    proc = subprocess.run(
        [str(sharc_home / "bin" / "data_extractor.x"), "output.dat"],
        input="\n" * 10,
        cwd=traj_dir,
        env=sharc_tool_environment(sharc_home, python),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0 or not (traj_dir / "output_data" / "coeff_MCH.out").is_file():
        tail = (proc.stdout[-800:] + "\n" + proc.stderr[-800:]).strip()
        raise RuntimeError(f"data_extractor.x failed ({proc.returncode}) in {traj_dir}:\n{tail}")


def sharc_dir_name(task: str) -> str:
    """``traj_00007`` -> ``TRAJ_00007`` (F2: populations.py picks TRAJ_0XXXX directories)."""
    m = re.fullmatch(r"traj_(\d{5})", task)
    if not m:
        raise ValueError(f"unexpected trajectory name {task!r} (traj_ and five digits)")
    return f"TRAJ_{m.group(1)}"


def build_tree(sharc_dir: Path, tasks: Sequence[str], parent: Path) -> list[str]:
    """The F2 layout: ``parent/TRAJ_0000k/`` real directories holding symlinks to ``input``,
    ``output.dat`` and ``output.lis`` (so the extractor's ``output_data/`` lands in ``parent``,
    not in the ensemble). Tasks without those files are left out and returned as skipped."""
    parent.mkdir(parents=True, exist_ok=True)
    skipped = []
    for task in tasks:
        src = Path(sharc_dir) / task
        names = ("input", "output.dat", "output.lis")
        if not all((src / n).is_file() for n in names):
            skipped.append(task)
            continue
        dst = parent / sharc_dir_name(task)
        dst.mkdir(exist_ok=True)
        for n in names:
            link = dst / n
            if not link.is_symlink() and not link.exists():
                link.symlink_to((src / n).resolve())
    return skipped


def parse_populations(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``pop_<mode>.out``: times (fs) and populations (T, n); ``#`` lines skipped."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            rows.append([float(x) for x in line.split()])
    if not rows:
        raise ValueError(f"{path}: no data rows")
    data = np.asarray(rows)
    return data[:, 0], data[:, 1:]


_FINAL = re.compile(r"time constant \(\s*k10\s*\):\s*(?P<tau>\S+) fs\s*\+/-")
_ARITH = re.compile(
    r'Analysis for time constant "k10".*?'
    r"Arithmetic analysis:\s*(?P<mean>\S+)\s*\+/-\s*(?P<std>\S+)",
    re.DOTALL,
)


def parse_fit(path: Path) -> dict[str, Any]:
    """tau (fs) from the last ``Final parameters`` line (or, failing that, the arithmetic mean of
    a bootstrap analysis) and, only from a bootstrap analysis block (F4), its arithmetic std. The
    "intrinsic" error of the normal mode is ignored. ``t0_fs`` is always None (no delay).
    Nothing depends on the header lines (they were scrubbed in the fixtures)."""
    text = Path(path).read_text()
    arith = _ARITH.search(text)
    finals = _FINAL.findall(text)
    if finals:
        tau = float(finals[-1])
    elif arith:
        tau = float(arith.group("mean"))
    else:
        raise ValueError(f"{path}: no time constant k10 found")
    std = float(arith.group("std")) if arith else None
    return {"tau_fs": tau, "t0_fs": None, "tau_std_fs": std}


def classical_agree(ours: np.ndarray, theirs: np.ndarray, *, decimals: int) -> bool:
    """PR7: equal to the print precision, |diff| <= half a unit of the last printed digit."""
    return bool(np.all(np.abs(np.asarray(ours) - np.asarray(theirs)) <= 0.5 * 10.0**-decimals))


def fit_undelayed(t: np.ndarray, per: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    """F4: ``y = A exp(-t/tau)``, ``A = y(0)``, on the mean of ``per`` (K, T); the std is the
    ddof = 1 std of the same fit over the resamples ``per[idx].mean(axis=0)`` (None below two)."""
    from scipy.optimize import least_squares

    t = np.asarray(t, dtype=float)

    def fit(y: np.ndarray) -> float | None:
        a = float(y[0])
        if a <= 0.0:
            return None
        res = least_squares(
            lambda x: a * np.exp(-t / np.exp(x[0])) - y,
            [np.log(max(float(t[-1]), 1.0))],
            bounds=([-50.0], [50.0]),
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
        )
        return float(np.exp(res.x[0]))

    per = np.asarray(per, dtype=float)
    tau = fit(per.mean(axis=0))
    taus = [r for r in (fit(per[idx].mean(axis=0)) for idx in indices) if r is not None]
    std = float(np.std(taus, ddof=1)) if len(taus) >= 2 else None
    return {"tau_fs": tau, "tau_std_fs": std, "n_resamples_fitted": len(taus)}


def held_on_steps(trajs, steps: np.ndarray, n_states: int) -> tuple[np.ndarray, np.ndarray]:
    """``on_steps``, with a trajectory that ends early continued by its last values (as
    populations.py does, F2)."""
    active = np.zeros((len(trajs), steps.size), dtype=np.int64)
    pops = np.zeros((len(trajs), steps.size, n_states))
    for i, t in enumerate(trajs):
        last = int(max(s for s, k in zip(t.grid_step, t.ticks, strict=True) if k == 0))
        cut = int(np.searchsorted(steps, last, side="right"))
        a, p = on_steps([t], steps[:cut], n_states)
        active[i, :cut], pops[i, :cut] = a[0], p[0]
        active[i, cut:], pops[i, cut:] = a[0, -1], p[0, -1]
    return active, pops


def _verdict(ok: bool, **numbers: Any) -> dict[str, Any]:
    return {"verdict": "pass" if ok else "fail", **numbers}


def compare_tools(
    sharc_dir: Path,
    tasks: Sequence[str],
    *,
    n_total: int,
    dt_fs: float,
    n_states: int,
    sharc_home: Path,
    python: str,
    work_dir: Path,
    quantum_bound: float = QUANTUM_BOUND,
    seed: int = FIT_SEED,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict[str, Any]:
    """Run populations.py (modes 2 and 8) and make_fit.py on a SHARC ensemble and compare with
    our analysis of the same trajectories (all of them, no exclusion; spec 4).

    The fit target on both sides is the classical excited population ``1 - P(S0)`` on SHARC's
    time grid (F4), ``y = A exp(-t/tau)`` with ``A = y(0)``."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    skipped = build_tree(sharc_dir, tasks, work_dir / PARENT)
    used = [t for t in tasks if t not in skipped]
    if not used:
        raise RuntimeError(f"no trajectory of {sharc_dir} has input, output.dat and output.lis")
    for task in used:
        run_extractor(sharc_home, python, work_dir / PARENT / sharc_dir_name(task))
    kw = {"n_states": n_states, "maxtime_fs": n_total * dt_fs}
    run_tool(
        sharc_home, python, "populations.py", work_dir,
        populations_answers("classical", bootstrap=True, **kw),
        outputs=[CLASSICAL_OUTPUT, "bootstrap_data"],
        stdout_name="populations_classical.stdout.txt",
    )  # fmt: skip
    run_tool(
        sharc_home, python, "populations.py", work_dir,
        populations_answers("quantum", **kw),
        outputs=[QUANTUM_OUTPUT],
        stdout_name="populations_quantum.stdout.txt",
    )  # fmt: skip
    t_c, p_c = parse_populations(work_dir / CLASSICAL_OUTPUT)
    t_q, p_q = parse_populations(work_dir / QUANTUM_OUTPUT)
    steps = np.rint(t_c / dt_fs).astype(np.int64)
    if np.max(np.abs(steps * dt_fs - t_c)) > 1e-6 or not np.array_equal(t_c, t_q):
        raise ValueError("SHARC's printed times are not on our dt_fs grid (or differ by mode)")
    trajs = [read_sharc_trajectory(Path(sharc_dir) / k, k, n_total=n_total) for k in used]
    if any(t is None for t in trajs):
        raise RuntimeError("a trajectory has files but no readable output.dat")
    active, pops = held_on_steps(trajs, steps, n_states)
    ours_c = classical_populations(active, n_states)[0]
    ours_q = quantum_populations(pops)
    classical = _verdict(
        classical_agree(ours_c, p_c, decimals=PRINT_DECIMALS),
        max_abs_difference=float(np.abs(ours_c - p_c).max()),
        decimals=PRINT_DECIMALS,
    )
    qdiff = float(np.abs(ours_q - p_q).max())
    quantum = _verdict(qdiff <= quantum_bound, max_abs_difference=qdiff, bound=quantum_bound)

    # The fit: SHARC on what populations.py printed (pop_2.out), ours on the per-trajectory
    # indicator 1 - [active == S0] at the same steps (the same curve, F4). make_fit.py runs
    # twice: normal mode for tau, bootstrap mode for its std (the bootstrap run's "final
    # parameters" are not documented as the full-data fit).
    fit_dir = work_dir / "fit"
    fit_dir.mkdir(exist_ok=True)
    rel_pop = os.path.relpath(work_dir / CLASSICAL_OUTPUT, fit_dir)
    rel_boot = os.path.relpath(work_dir / "bootstrap_data", fit_dir)
    run_tool(
        sharc_home, python, "make_fit.py", fit_dir,
        fit_answers(rel_pop, n_states=n_states),
        outputs=["fit_results.txt"],
        stdout_name="make_fit_classical.stdout.txt",
    )  # fmt: skip
    sharc_fit = parse_fit(fit_dir / "make_fit_classical.stdout.txt")
    note = None
    s_sharc: float | None
    try:
        run_tool(
            sharc_home, python, "make_fit.py", fit_dir,
            fit_answers(
                rel_pop, n_states=n_states, bootstrap_samples=n_bootstrap, bootstrap_dir=rel_boot
            ),
            outputs=[FIT_OUTPUT],
            stdout_name="make_fit_bootstrap.stdout.txt",
        )  # fmt: skip
        s_sharc = parse_fit(fit_dir / FIT_OUTPUT)["tau_std_fs"]
    except (RuntimeError, ValueError) as exc:
        s_sharc, note = None, f"make_fit.py bootstrap failed, s_sharc taken as 0: {exc}"
    if s_sharc is None:
        s_sharc, note = 0.0, note or "no bootstrap std reported, s_sharc taken as 0"
    indicator = (active != 0).astype(float)
    idx = bootstrap_indices(task_rng(seed, COMPARE_STREAM, 0), len(used), n_bootstrap)
    ours_fit = fit_undelayed(steps * dt_fs, indicator, idx)
    tau_o, s_o = ours_fit["tau_fs"], ours_fit["tau_std_fs"] or 0.0
    fit: dict[str, Any]
    if tau_o is None:
        fit = {"verdict": "inconclusive", "reason": "no excited population at t = 0 on our side"}
    else:
        bound = FIT_FACTOR * float(np.hypot(s_o, s_sharc))
        fit = _verdict(
            abs(tau_o - sharc_fit["tau_fs"]) <= bound,
            difference=tau_o - sharc_fit["tau_fs"],
            bound=bound,
        )
    fit.update(
        tau_ours_fs=tau_o,
        tau_std_ours_fs=s_o,
        tau_sharc_fs=sharc_fit["tau_fs"],
        tau_std_sharc_fs=s_sharc,
        note=note,
        model="y = A exp(-t/tau), A = y(0); target 1 - P(S0), classical, SHARC's time grid (F4)",
        sharc_bootstrap_reproducible=False,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return {
        "n_trajectories": len(used),
        "skipped": skipped,
        "times_fs": t_c.tolist(),
        "classical": classical,
        "quantum": quantum,
        "fit": fit,
        "verdicts": {
            "classical": classical["verdict"],
            "quantum": quantum["verdict"],
            "fit": fit["verdict"],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    from forensics import jsonable
    from project_inputs import load_project_inputs

    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("sharc", type=Path, help="a SHARC ensemble variant directory (traj_*/)")
    p.add_argument("project", type=Path, help="our gqteashmd project directory")
    p.add_argument("--sharc-home", type=Path, required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--quantum-bound", type=float, default=QUANTUM_BOUND)
    args = p.parse_args(argv)
    inputs = load_project_inputs(args.project)
    dyn = inputs.config.dynamics
    assert dyn is not None and inputs.config.electronic is not None
    result = compare_tools(
        args.sharc,
        [s.task_id for s in inputs.starts],
        n_total=round(dyn.tmax_fs / dyn.dt_fs),
        dt_fs=dyn.dt_fs,
        n_states=inputs.config.electronic.singlets,
        sharc_home=args.sharc_home,
        python=args.python,
        work_dir=args.work,
        quantum_bound=args.quantum_bound,
    )
    result.update(sharc=str(args.sharc), project=str(args.project))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(jsonable(result), indent=2))
    print(json.dumps(result["verdicts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
