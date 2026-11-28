"""The spin-orbit swap test (dynamics with SOC, spec 5.1): sharc.x drives our PySCF backend.

SHARC 4.1 is only *executed*; nothing from SHARC is imported or copied. The sharc.x ``input``,
``geom``/``veloc``, ``QM/runQM.sh``, the QM.out writer, the QM.in reader and the output.dat
parser are those of ``run_sharc_lvc.py`` (our own code from the public manual and observed
output). The facts of the plan's Task 1 (probe of the LVC spin-orbit model through our
provider; ``tests/data/sharc/soc/``):

- SHARC's state order equals ours (S0..S(n-1), then T1..Tm with Ms = -1, 0, +1, the state index
  innermost), so nothing is permuted;
- every step sharc.x asks for ``SOC``, ``DM``, ``GRAD all`` and, from step 1 on, ``OVERLAP``
  (one call per step; never ``H``, ``NACDR`` or ``PHASES``); the dipoles are answered with zeros;
- output.dat does not print E_tot: it is ``Re(U^dagger H U)[a, a] + E_kin``; the step-0
  overlap block is a -123 placeholder (not used here);
- sharc.x adds about +1e-12 / +2e-12 Eh to the Ms = 0 / +1 triplet diagonal of H (so its U
  inside a triplet cluster is fixed by these offsets); every compared quantity is invariant
  under the gauge of U;
- sharc.x writes the interface geometry with 7 decimals in angstrom, so our provider evaluates
  at the rounded geometry while sharc.x propagates the unrounded one (P7a-2, P7b). Both runs
  start from the start geometry rounded the same way (``interface_geometry``).

The provider (``SocSwapModel``) is our ``BackendModel`` over a ``PySCFBackend``: the expanded
snapshot it hands to sharc.x (complex spin-orbit H, the n + 3m expanded gradients, the
Lowdin-orthonormalised block overlaps expanded over Ms) is the snapshot our runner's core sees.
Our side (``run_ours``) is ``TrajectoryRunner`` over a second ``BackendModel`` with the same
settings, replaying SHARC's random numbers (output.dat block 9, steps 1, 2, ...).

The compared quantities (spec 5.1), per printed step: positions, velocities, the total energy,
the active state's diagonal energy and the spin-free MCH populations (``|U c|^2`` summed over Ms).

System (spec, section 0 item 5): thioformaldehyde, 6-31G, SA-CASSCF(4,3) 2S + 2T or CIS
3S + 3T (``td_conv_tol`` 1e-7), the electronic settings of tests/validation/test_orca_soc.py;
starts: Wigner samples at T = 0 of the RHF/6-31G* frequencies (project seed 2029, a throwaway
project through ``sampling``, plan Task 1 Step 3), initial state S1 (MCH), dt 0.5 fs, 25
substeps, ``coupling overlap``, ``nogradcorrect``, ``ekincorrect parallel_vel``, EDC (0.1 Eh),
SHARC hopping, ``endpoint`` interpolation; (b) ours with ``hop_energy = "previous"``.

Examples (WSL):
  # the start search of swap (b): SHARC's own runs, samples 0, 1, ... until a singlet-triplet hop
  python tools/sharc_compare/run_sharc_soc_swap.py search --sharc-home /home/ajc/sharc4-4.1 \\
      --samples ~/dynsoc_task9/samples/run/03_sampling/samples.h5 --method casscf \\
      --work-dir ~/dynsoc_task9/search/casscf --out ~/dynsoc_task9/search/casscf.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.core.units import AU_TIME_IN_FS, DALTON_IN_ME

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sharc_lvc import (  # noqa: E402
    LvcSharcOptions,
    LvcTrajectory,
    lvc_input,
    run_lvc_trajectory,
)
from run_sharc_pyscf_dynamics import interface_geometry  # noqa: E402

METHODS = ("casscf", "cis")
SETTINGS: dict[str, dict[str, Any]] = {
    "casscf": {
        "method": "casscf",
        "basis": "6-31g",
        "singlets": 2,
        "triplets": 2,
        "scf_conv_tol": 1e-12,
        "casscf": {"ncas": 3, "nelecas": 4, "conv_tol": 1e-11, "conv_tol_grad": 1e-6},
    },
    "cis": {
        "method": "tda",
        "basis": "6-31g",
        "functional": "hf",
        "singlets": 3,
        "triplets": 3,
        "scf_conv_tol": 1e-12,
        "td_conv_tol": 1e-7,  # the CIS states do not converge at the default (sub-project 2)
    },
}
PROJECT_SEED = 2029  # the Wigner samples (plan Task 1 Step 3)
SHARC_RNGSEED = 2029  # sharc.x's own random numbers in the start search and in (b)
DT_FS = 0.5
NSUBSTEPS = 25
INITIAL_STATE = 1  # S1, the MCH index
STEPS_HOPPING_OFF = 40  # (a): steps 0-40
MAX_STEPS = 200  # (b): at most 200 steps
AFTER_HOP = 20  # (b): the first singlet-triplet hop + 20 steps
SEARCH_SAMPLES = 20  # (b): samples 0..19, else BLOCKED
TIMEOUT_PER_STEP_S = 60.0
# SHARC has no step rejection: our runner's overlap signal is moved out of reach (the smallest
# raw singular value of the CIS start search reached 0.377, below the stage default 0.5) and the
# energy-jump signal stays off (the library default); the CASSCF rung signal cannot be switched off
NO_REJECTION_OVERLAP = 1e-9
BOUNDS = {"q": 1e-5, "v": 1e-6, "e_total": 1e-6, "e_active": 1e-6, "populations": 1e-5}
UNITS = {"q": "bohr", "v": "a.u.", "e_total": "Eh", "e_active": "Eh", "populations": ""}
QUANTITIES = tuple(BOUNDS)


# --- the start ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Start:
    """One Wigner sample: symbols, masses (amu), positions (natom, 3; bohr, on SHARC's 7-decimal
    angstrom grid), velocities (natom, 3; bohr per atomic time unit) and the equilibrium geometry
    (the CASSCF reference, as the dynamics stage passes it)."""

    sample: int
    symbols: tuple[str, ...]
    masses_amu: np.ndarray
    q_bohr: np.ndarray
    v_au: np.ndarray
    equilibrium_bohr: np.ndarray

    def to_json(self) -> dict[str, Any]:
        return {
            "sample": self.sample,
            "symbols": list(self.symbols),
            "masses_amu": [float(x) for x in self.masses_amu],
            "q_bohr": np.asarray(self.q_bohr).tolist(),
            "v_au": np.asarray(self.v_au).tolist(),
            "equilibrium_bohr": np.asarray(self.equilibrium_bohr).tolist(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Start:
        return cls(
            sample=int(data["sample"]),
            symbols=tuple(data["symbols"]),
            masses_amu=np.asarray(data["masses_amu"], dtype=float),
            q_bohr=np.asarray(data["q_bohr"], dtype=float),
            v_au=np.asarray(data["v_au"], dtype=float),
            equilibrium_bohr=np.asarray(data["equilibrium_bohr"], dtype=float),
        )


def read_start(samples_h5: Path, sample: int) -> Start:
    """Sample ``sample`` of a ``samples.h5`` written by the sampling stage; the positions are
    rounded as sharc.x writes them for its interface."""
    from gqteashmd.vib.artifacts import read_samples_h5

    data = read_samples_h5(Path(samples_h5))
    return Start(
        sample=int(sample),
        symbols=tuple(data["symbols"]),
        masses_amu=data["masses_amu"],
        q_bohr=interface_geometry(data["positions_bohr"][sample]),
        v_au=data["velocities_au"][sample],
        equilibrium_bohr=data["equilibrium_bohr"],
    )


def masses_per_coordinate(start: Start) -> np.ndarray:
    return np.repeat(np.asarray(start.masses_amu, dtype=float) * DALTON_IN_ME, 3)


# --- settings -----------------------------------------------------------------------------------


def dynamics_options(*, hopping: str, hop_energy: str = "current"):
    """Our options matching the SHARC settings of spec 5.1 (``hopping`` "sharc" or "none")."""
    from gqteashmd.dynamics.options import DynamicsOptions

    return DynamicsOptions(
        dt_au=DT_FS / AU_TIME_IN_FS,
        nsubsteps=NSUBSTEPS,
        hopping=hopping,
        coupling="overlap",
        gradient_transformation="none",
        ekin_correction="parallel_vel",
        frustrated="keep",
        decoherence="edc",
        decoherence_param_Eh=0.1,
        edc_convention="sharc",
        substep_interpolation="endpoint",
        hop_energy=hop_energy,
    )


def make_backend(method: str, start: Start, scratch: Path, *, threads: int = 8):
    """A fresh ``PySCFBackend`` of ``method`` with the CASSCF reference at the equilibrium."""
    from gqteashmd.config.schema import ElectronicSection
    from gqteashmd.electronic.base import MoleculeSpec
    from gqteashmd.electronic.pyscf_backend import PySCFBackend

    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}; got {method!r}")
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    return PySCFBackend(
        MoleculeSpec(start.symbols),
        ElectronicSection.model_validate(SETTINGS[method]),
        log_file=scratch / "electronic.log",
        threads=threads,
        scratch_dir=scratch,
        reference_coords_bohr=np.asarray(start.equilibrium_bohr).reshape(-1, 3),
    )


# --- the provider -------------------------------------------------------------------------------


class SocSwapModel:
    """Our ``BackendModel`` (with triplets) as the model of ``run_sharc_lvc.QmServer``.

    ``evaluate`` is the adapter's own: the snapshot holds the complex spin-orbit Hamiltonian,
    every expanded gradient (no ``eselect``) and the expanded overlaps, the data our runner's
    core sees. A call without a previous step resets the backend (a fresh start). ``states``
    are the expanded states in canonical order (Task 1: SHARC's order, no permutation)."""

    def __init__(self, model) -> None:
        from gqteashmd.dynamics.spin import expanded_states

        if not model.n_triplets:
            raise ValueError("the spin-orbit swap model needs a BackendModel with triplets")
        if model.eselect_Eh is not None:
            raise ValueError("sharc.x asks for every gradient: build the model without eselect")
        self.model = model
        self.state_counts = model.state_counts
        self.states = tuple(expanded_states(model.state_counts))
        self.min_singular_values: list[float] = []

    def evaluate(self, q: np.ndarray, previous: Any):
        if previous is None:
            self.model.backend.reset()
        snapshot = self.model.evaluate(np.asarray(q, dtype=float).reshape(-1), previous)
        value = self.model.min_singular_value
        if value is not None:
            self.min_singular_values.append(value)
        return snapshot


# --- the two runs -------------------------------------------------------------------------------


def sharc_input(options, n_steps: int, state_counts: Sequence[int], rngseed: int) -> str:
    """The sharc.x ``input`` (``run_sharc_lvc.lvc_input``) for our ``options``: the diagonal
    surface with spin-orbit coupling, start in the MCH state S1 (``coeff auto``), every
    gradient (``grad_all``) and every printed step with its gradients and overlaps."""
    return lvc_input(
        n_steps,
        options.dt_au * AU_TIME_IN_FS,
        LvcSharcOptions.from_dynamics_options(options),
        rngseed,
        states=state_counts,
        state_mch=INITIAL_STATE,
    )


def run_sharc(
    model: SocSwapModel,
    start: Start,
    options,
    *,
    n_steps: int,
    work_dir: Path,
    sharc_home: Path,
    rngseed: int = SHARC_RNGSEED,
) -> LvcTrajectory:
    """sharc.x on ``model`` (``SocSwapModel``) from ``start`` in the MCH state S1, with the
    SHARC settings mapped from our ``options``; returns the parsed output.dat."""
    text = sharc_input(options, n_steps, model.state_counts, rngseed)
    return run_lvc_trajectory(
        Path(work_dir),
        sharc_home=Path(sharc_home),
        provider="ours",
        model=model,
        timeout_s=max(600.0, TIMEOUT_PER_STEP_S * n_steps),
        symbols=start.symbols,
        masses_amu=start.masses_amu,
        q0=start.q_bohr,
        v0=start.v_au,
        input_text=text,
    )


def replayed_numbers(trajectory: LvcTrajectory) -> list[float]:
    """SHARC's random numbers in the order our runner draws them: one per step from step 1 on
    (step 0 prints 0.0 and draws nothing); ``ValueError`` if the printed steps are not 0, 1, ..."""
    steps = np.asarray(trajectory.steps)
    if not np.array_equal(steps, np.arange(steps.size)):
        raise ValueError("SHARC's printed steps must be 0, 1, 2, ... to replay its random numbers")
    return [float(x) for x in np.asarray(trajectory.random_numbers)[1:]]


def run_ours(
    backend,
    start: Start,
    options,
    *,
    n_steps: int,
    work_dir: Path,
    random_numbers: Sequence[float],
) -> dict[str, Any]:
    """Our ``TrajectoryRunner`` over ``BackendModel(backend)`` from ``start`` in S1, replaying
    ``random_numbers`` (one per step). SHARC has no step rejection, so neither the overlap signal
    (``overlap_threshold`` ``NO_REJECTION_OVERLAP``) nor the energy-jump signal (off, the library
    default) can halve a step; the energy-drift stop keeps its default (it ends a run, it does
    not change one). Returns ``rows`` (all of steps.log), ``result`` (the runner's summary),
    ``wall_time_s``, ``n_singlets`` and ``n_triplets``."""
    from gqteashmd.dynamics.randomness import ReplaySource
    from gqteashmd.trajectory.model import BackendModel
    from gqteashmd.trajectory.runner import InitialConditions, RunSettings, TrajectoryRunner
    from gqteashmd.trajectory.store import StepLog

    work_dir = Path(work_dir)
    model = BackendModel(backend, masses_per_coordinate(start), coupling="overlap")
    runner = TrajectoryRunner(
        model,
        RunSettings(
            options, tmax_au=n_steps * options.dt_au, overlap_threshold=NO_REJECTION_OVERLAP
        ),
        work_dir / "trajectory",
    )
    started = time.perf_counter()
    result = runner.run(
        InitialConditions(
            np.asarray(start.q_bohr).reshape(-1),
            np.asarray(start.v_au).reshape(-1),
            INITIAL_STATE,
        ),
        ReplaySource(list(random_numbers)),
    )
    wall = time.perf_counter() - started
    rows = StepLog(
        work_dir / "trajectory" / "steps.log",
        model.n_states,
        model.n_dof,
        n_spin_free=model.n_spin_free,
    ).read()
    return {
        "rows": rows,
        "result": result,
        "wall_time_s": wall,
        "n_singlets": model.n_singlets,
        "n_triplets": model.n_triplets,
    }


# --- the compared records -----------------------------------------------------------------------


def sharc_records(trajectory: LvcTrajectory, n: int, m: int) -> dict[str, np.ndarray]:
    """The compared quantities of every printed SHARC step (atomic units): ``steps``, ``q``,
    ``v``, ``e_total`` (``Re(U^dagger H U)[a, a] + E_kin``), ``e_active`` (the active diagonal
    energy), ``populations`` (spin-free MCH, ``|U c|^2`` summed over Ms), ``active`` (diagonal,
    0-based), ``active_sf`` (the spin-free state of SHARC's printed MCH state), ``probabilities``
    and ``random_numbers``."""
    from gqteashmd.dynamics.representation import diagonal_energies
    from gqteashmd.trajectory.spin_expansion import spin_free_index, spin_free_populations

    k = np.arange(len(trajectory.steps))
    e_diag = np.array(
        [diagonal_energies(h, u) for h, u in zip(trajectory.hamiltonian, trajectory.u, strict=True)]
    )
    e_active = e_diag[k, trajectory.active]
    c_mch = np.einsum("kab,kb->ka", trajectory.u, trajectory.c)
    return {
        "steps": np.asarray(trajectory.steps, dtype=int),
        "q": np.asarray(trajectory.q),
        "v": np.asarray(trajectory.v),
        "e_total": e_active + np.asarray(trajectory.e_kinetic),
        "e_active": e_active,
        "populations": spin_free_populations(np.abs(c_mch) ** 2, n, m),
        "active": np.asarray(trajectory.active, dtype=int),
        "active_sf": np.array([spin_free_index(n, m, int(a)) for a in trajectory.active_mch]),
        "probabilities": np.asarray(trajectory.probabilities),
        "random_numbers": np.asarray(trajectory.random_numbers),
    }


def our_records(rows: np.ndarray) -> dict[str, np.ndarray]:
    """The same quantities from our steps.log, on the dt grid (records with ``ticks == 0``)."""
    grid = rows[rows["ticks"] == 0]
    k = np.arange(len(grid))
    return {
        "steps": np.asarray(grid["grid_step"], dtype=int),
        "q": np.asarray(grid["q"]),
        "v": np.asarray(grid["v"]),
        "e_total": np.asarray(grid["e_total"]),
        "e_active": np.asarray(grid["energies_diag"])[k, grid["active"]],
        "populations": np.asarray(grid["populations_mch"]),
        "active": np.asarray(grid["active"], dtype=int),
        "active_sf": np.asarray(grid["active_mch"], dtype=int),
        "probabilities": np.asarray(grid["probabilities"]),
        "random_numbers": np.asarray(grid["random_number"]),
    }


def differences(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per common step (aligned by step number), max over components of ``|a - b|`` for every
    compared quantity; ``steps`` the common steps; ``same_active_sf`` per step; and, when both
    have them, ``probabilities`` (the hopping probabilities, reported only)."""
    for name, side in (("first", a), ("second", b)):
        if not np.all(np.diff(side["steps"]) > 0):
            raise ValueError(f"the {name} records' steps are not strictly increasing")
    common = np.intersect1d(a["steps"], b["steps"])
    ia = np.searchsorted(a["steps"], common)
    ib = np.searchsorted(b["steps"], common)
    out: dict[str, np.ndarray] = {"steps": common}
    for key in QUANTITIES:
        diff = np.abs(np.asarray(a[key])[ia] - np.asarray(b[key])[ib])
        out[key] = diff.reshape(len(common), -1).max(axis=1)
    out["same_active_sf"] = np.asarray(a["active_sf"])[ia] == np.asarray(b["active_sf"])[ib]
    if "probabilities" in a and "probabilities" in b:  # reported, not bounded
        diff = np.abs(np.asarray(a["probabilities"])[ia] - np.asarray(b["probabilities"])[ib])
        out["probabilities"] = diff.reshape(len(common), -1).max(axis=1)
    return out


def maxima(diffs: dict[str, np.ndarray], last: int) -> dict[str, dict[str, Any]]:
    """For every quantity, the largest difference over the steps 0..``last`` and its step."""
    keep = diffs["steps"] <= last
    steps = diffs["steps"][keep]
    out = {}
    for key in QUANTITIES:
        values = diffs[key][keep]
        i = int(np.argmax(values))
        out[key] = {"max": float(values[i]), "step": int(steps[i]), "bound": BOUNDS[key]}
    return out


def violations(diffs: dict[str, np.ndarray], last: int) -> list[str]:
    """Every (quantity, step) over 0..``last`` outside its bound, and every step where the active
    spin-free states differ, as messages naming the quantity and the step."""
    keep = diffs["steps"] <= last
    found = []
    for key in QUANTITIES:
        for step, value in zip(diffs["steps"][keep], diffs[key][keep], strict=True):
            if not value <= BOUNDS[key]:
                found.append(
                    f"{key} at step {int(step)}: {value:.3e} {UNITS[key]} > {BOUNDS[key]:.0e}"
                )
    for step, same in zip(diffs["steps"][keep], diffs["same_active_sf"][keep], strict=True):
        if not same:
            found.append(f"active spin-free state at step {int(step)} differs")
    return found


def singlet_triplet_hops(records: dict[str, np.ndarray], n: int) -> list[int]:
    """The steps k at which the active (diagonal) state changed and its spin-free state went from
    a singlet to a triplet or back (spin-free index < n is a singlet)."""
    active, sf = np.asarray(records["active"]), np.asarray(records["active_sf"])
    return [
        int(records["steps"][k])
        for k in range(1, active.size)
        if active[k] != active[k - 1] and (sf[k] < n) != (sf[k - 1] < n)
    ]


def near_ties(records: dict[str, np.ndarray], last: int, tolerance: float = 1e-5) -> list[int]:
    """The steps 1..``last`` at which the random number lies within ``tolerance`` of a partial
    sum of the hopping probabilities (a hop decision that round-off could flip; reported)."""
    out = []
    for k, step in enumerate(records["steps"]):
        if 1 <= step <= last:
            sums = np.cumsum(records["probabilities"][k])
            if np.any(np.abs(sums - records["random_numbers"][k]) < tolerance):
                out.append(int(step))
    return out


def window_end(first_hop: int) -> int:
    """(b): the last compared step, the first singlet-triplet hop + 20, at most 200."""
    return min(int(first_hop) + AFTER_HOP, MAX_STEPS)


def format_table(diffs: dict[str, np.ndarray]) -> str:
    """The per-step differences and whether the active spin-free states agree."""
    header = "step  " + "  ".join(f"{k:>11}" for k in QUANTITIES) + "  same_sf"
    lines = [header]
    for i, step in enumerate(diffs["steps"]):
        values = "  ".join(f"{float(diffs[k][i]):11.3e}" for k in QUANTITIES)
        lines.append(f"{int(step):4d}  {values}  {bool(diffs['same_active_sf'][i])}")
    return "\n".join(lines)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating | float):
        x = float(value)
        return x if math.isfinite(x) else None
    return value


# --- the start search of (b) (plan Task 1 Step 3, moved to Task 9 by the controller) -----------


def search(
    *,
    samples_h5: Path,
    method: str,
    sharc_home: Path,
    work_dir: Path,
    out: Path,
    first: int = 0,
    threads: int = 8,
) -> dict[str, Any]:
    """SHARC's own runs (sharc.x on our provider, hopping on, ``rngseed`` 2029, 200 steps) on
    samples ``first``, ``first + 1``, ... (at most sample 19) until one has a singlet-triplet hop;
    every sample's result is written to ``out`` (JSON) as soon as it is known. A sample whose run
    failed, or printed fewer than 201 steps without a singlet-triplet hop, is undetermined: it is
    recorded with ``error``, ``stopped`` names it, and the search stops (it counts neither as "no
    hop" nor as a reason to go on, because the rule takes the first sample by index)."""
    work_dir, out = Path(work_dir), Path(out)
    work_dir.mkdir(parents=True, exist_ok=True)
    options = dynamics_options(hopping="sharc", hop_energy="previous")
    n, m = SETTINGS[method]["singlets"], SETTINGS[method]["triplets"]
    report: dict[str, Any] = {
        "method": method,
        "samples_h5": str(samples_h5),
        "project_seed": PROJECT_SEED,
        "rngseed": SHARC_RNGSEED,
        "max_steps": MAX_STEPS,
        "runs": [],
        "found": None,
    }
    from gqteashmd.trajectory.model import BackendModel

    for sample in range(first, SEARCH_SAMPLES):
        start = read_start(samples_h5, sample)
        backend = make_backend(
            method, start, work_dir / f"sample_{sample:02d}_backend", threads=threads
        )
        model = SocSwapModel(
            BackendModel(backend, masses_per_coordinate(start), coupling="overlap")
        )
        started = time.perf_counter()
        entry: dict[str, Any] = {"sample": sample}
        try:
            traj = run_sharc(
                model,
                start,
                options,
                n_steps=MAX_STEPS,
                work_dir=work_dir / f"sample_{sample:02d}",
                sharc_home=sharc_home,
            )
            records = sharc_records(traj, n, m)
            hops = singlet_triplet_hops(records, n)
            changes = [int(s) for s in records["steps"][1:][np.diff(records["active"]) != 0]]
            entry.update(
                {
                    "printed_steps": int(len(records["steps"])),
                    "hops": changes,
                    "active_sf": [int(x) for x in records["active_sf"]],
                    "singlet_triplet_hops": hops,
                    "min_singular_value": min(model.min_singular_values, default=None),
                }
            )
            if not hops and len(records["steps"]) < MAX_STEPS + 1:
                entry["error"] = (
                    f"SHARC printed {len(records['steps'])} of {MAX_STEPS + 1} steps and no "
                    "singlet-triplet hop: the sample is undetermined"
                )
        except Exception as exc:  # recorded; the search stops (below)
            entry["error"] = f"{type(exc).__name__}: {exc}"
            hops = []
        entry["wall_time_s"] = time.perf_counter() - started
        report["runs"].append(entry)
        if hops:
            report["found"] = {"sample": sample, "first_singlet_triplet_hop": hops[0]}
        elif "error" in entry:
            # the rule takes the first sample by index, so an undetermined sample ends the search
            report["stopped"] = {"sample": sample, "reason": entry["error"]}
        out.write_text(json.dumps(jsonable(report), indent=1) + "\n")
        print(json.dumps(jsonable(entry)), flush=True)
        if hops or "error" in entry:
            break
    return report


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("search", help="the start search of swap (b)")
    s.add_argument("--sharc-home", required=True)
    s.add_argument("--samples", required=True, help="samples.h5 of the sampling stage")
    s.add_argument("--method", choices=METHODS, required=True)
    s.add_argument("--work-dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--first", type=int, default=0)
    s.add_argument("--threads", type=int, default=8)
    args = p.parse_args(argv)
    search(
        samples_h5=Path(args.samples).expanduser(),
        method=args.method,
        sharc_home=Path(args.sharc_home),
        work_dir=Path(args.work_dir).expanduser(),
        out=Path(args.out).expanduser(),
        first=args.first,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
