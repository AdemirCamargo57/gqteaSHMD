"""Run one sharc.x trajectory with SHARC_PYSCF.py (or our own QM.out provider) as a black box.

SHARC is only *executed*; nothing from SHARC is imported or copied. The ``input`` keywords
(manual 4.1), the ``geom``/``veloc`` files (4.2, 4.3), ``QM/runQM.sh`` (3.1.1) and output.dat
(5.3.1) come from the public SHARC 4.1 manual, as in ``run_sharc_lvc.py``, whose writers and
parser are reused; the PYSCF.template keywords are those of ``run_sharc_pyscf.py`` (P3a/P6).

System (P7b Task 10): ethylene SA2-CASSCF(2,2)/6-31G* (``tests/data/ethylene.xyz``), the
velocities of ``tests/validation/test_pyscf_dynamics.py`` (T = 300 K, ``default_rng(2026)``,
isotope masses), start in S1, hopping off, no decoherence, dt 0.5 fs, 25 substeps. The start
geometry is rounded to 7 decimals in angstrom (sharc.x writes its interface's QM.in that way),
so both programs evaluate step 0 at the same geometry.

Providers behind ``QM/runQM.sh``:

- ``"sharc_pyscf"``: SHARC's SHARC_PYSCF.py with our PYSCF.template;
- ``"ours"`` (the swap test): our ``PySCFBackend`` answers through ``run_sharc_lvc.QmServer``.

Measured behaviour of this installation (SHARC_PYSCF.py interface version 4.0; P7b Task 10):

- ``step 0`` (what sharc.x writes) stops the interface ("File .../pyscf.chk not found, cannot
  move to old!"); ``init`` works, so ``QM/runQM.sh`` rewrites a ``step 0`` line to ``init``.
- It has no overlaps: with ``coupling overlap`` (sharc.x requests ``OVERLAP``) and with
  ``PHASES`` at step 1 it stops with "Within the SHARC-PySCF interface, 'overlap' is not
  supported". The SHARC_PYSCF.py trajectory therefore uses ``coupling nacdr`` with
  ``nophases_from_interface`` (manual table 4.1: "Try to recover phase information from QM
  data").

Examples (WSL):
  # timing probe: two steps of SHARC_PYSCF.py
  python tools/sharc_compare/run_sharc_pyscf_dynamics.py --sharc-home /home/ajc/sharc4-4.1 \\
      --python /home/ajc/miniforge3/envs/sharc4.1/bin/python --steps 2 --coupling nacdr \\
      --work-dir ~/p7b_probe
  # the swap-test report (ours, sharc.x + SHARC_PYSCF.py, sharc.x + our provider)
  python tools/sharc_compare/run_sharc_pyscf_dynamics.py --sharc-home /home/ajc/sharc4-4.1 \\
      --python /home/ajc/miniforge3/envs/sharc4.1/bin/python --steps 40 \\
      --work-dir ~/p7b_reports/swap --report ~/p7b_reports/sharc_pyscf_dynamics.json
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.chem.masses import isotope_mass
from gqteashmd.chem.xyz import parse_xyz
from gqteashmd.config.schema import ElectronicSection
from gqteashmd.core.units import (
    ANGSTROM_TO_BOHR,
    AU_TIME_IN_FS,
    BOHR_IN_ANGSTROM,
    BOLTZMANN_HARTREE_PER_K,
    DALTON_IN_ME,
    HARTREE_IN_EV,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sharc_lvc import (  # noqa: E402
    QmServer,
    format_geom,
    format_veloc,
    parse_lvc_output_dat,
    write_server_runqm,
)
from run_sharc_model import SHARC_X_COMMAND, run_in_session, sharc_environment  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ETHYLENE_XYZ = ROOT / "tests" / "data" / "ethylene.xyz"
BASIS = "6-31g*"  # P7b Task 9 diagnosis: the STO-3G pi,pi* CASSCF solution is metastable
N_STATES = 2
TEMPERATURE_K = 300.0
VELOCITY_SEED = 2026
DT_FS = 0.5
NSUBSTEPS = 25
INITIAL_STATE = 1  # S1, 0-based MCH index


@dataclass(frozen=True)
class Convergence:
    """CASSCF convergence on both sides: our ``[electronic.casscf] conv_tol`` /
    ``conv_tol_grad`` and the PYSCF.template ``conv-tol`` / ``conv-tol-grad`` (manual table
    6.36, "Convergence criterion for the CASSCF solver")."""

    conv_tol: float
    conv_tol_grad: float


DEFAULT_CONVERGENCE = Convergence(1e-10, 1e-6)  # the P3a template and our defaults
# P7b Task 10 fix round 1 (controller ruling); measured: SHARC_PYSCF.py stops at step 0
# ("Calculator failed to converge!") and our SA-CASSCF fails on every rung at step 0
TIGHT_CONVERGENCE = Convergence(1e-12, 1e-8)


def pyscf_template(
    convergence: Convergence = DEFAULT_CONVERGENCE,
    *,
    ncas: int = 2,
    nelecas: int = 2,
    roots: int = N_STATES,
    basis: str = BASIS,
) -> str:
    """PYSCF.template (manual 6.24, table 6.36): the keywords of ``run_sharc_pyscf.TEMPLATE``
    (P3a/P6) with the given CASSCF convergence and active space."""
    return (
        f"ncas {ncas}\nnelecas {nelecas}\nroots {roots}\nbasis {basis}\nmethod casscf\n"
        f"fix-spin-shift 0.2\nconv-tol {_power(convergence.conv_tol)}\n"
        f"conv-tol-grad {_power(convergence.conv_tol_grad)}\nverbose 3\n"
    )


def _power(x: float) -> str:
    """1e-06 -> "1e-6", as run_sharc_pyscf.TEMPLATE writes it."""
    mantissa, exponent = f"{x:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


PYSCF_TEMPLATE = pyscf_template()
PYSCF_RESOURCES = "scratchdir ./SCRATCH\nmemory 2000\nncpu 1\n"
PROVIDERS = ("sharc_pyscf", "ours")
COUPLINGS = ("overlap", "nacdr")
HOPPING = ("off", "sharc")
DECOHERENCE = ("none", "edc")
MIN_TIMEOUT_S = 600.0
TIMEOUT_PER_STEP_S = 60.0


def timeout_for(n_steps: int) -> float:
    """The sharc.x timeout for ``n_steps`` steps (seconds): 60 s per step, at least 600 s."""
    return max(MIN_TIMEOUT_S, TIMEOUT_PER_STEP_S * n_steps)


DECIMALS_ANGSTROM = 7  # sharc.x writes the interface geometry with 7 decimals in angstrom


# --- the system -----------------------------------------------------------------------------


@dataclass(frozen=True)
class InitialConditions:
    """Symbols, masses (amu), positions (natom, 3; bohr, on SHARC's 7-decimal angstrom grid)
    and velocities (natom, 3; bohr per atomic time unit)."""

    symbols: tuple[str, ...]
    masses_amu: np.ndarray
    q_bohr: np.ndarray
    v_au: np.ndarray


def interface_geometry(q_bohr: np.ndarray) -> np.ndarray:
    """``q`` rounded as sharc.x writes it for its interface: 7 decimals in angstrom."""
    return np.round(np.asarray(q_bohr, dtype=float) * BOHR_IN_ANGSTROM, DECIMALS_ANGSTROM) / (
        BOHR_IN_ANGSTROM
    )


def ethylene_initial_conditions() -> InitialConditions:
    """The Task 9 start: ``tests/data/ethylene.xyz``, v = N(0, 1) sqrt(k_B T / m) with
    ``default_rng(2026)`` and T = 300 K (no momentum removal), geometry on SHARC's grid."""
    geometry = parse_xyz(ETHYLENE_XYZ.read_text())
    symbols = tuple(geometry.symbols)
    masses_amu = np.array([isotope_mass(s) for s in symbols])
    masses_me = masses_amu * DALTON_IN_ME
    q = np.array(geometry.coords_angstrom, dtype=float) * ANGSTROM_TO_BOHR
    rng = np.random.default_rng(VELOCITY_SEED)
    v = rng.standard_normal((len(symbols), 3)) * np.sqrt(
        BOLTZMANN_HARTREE_PER_K * TEMPERATURE_K / masses_me[:, None]
    )
    return InitialConditions(symbols, masses_amu, interface_geometry(q), v)


def electronic_settings(convergence: Convergence = DEFAULT_CONVERGENCE) -> ElectronicSection:
    """Our SA2-CASSCF(2,2)/6-31G* with the CASSCF tolerances of the PYSCF.template."""
    return ElectronicSection.model_validate(
        {
            "method": "casscf",
            "basis": BASIS,
            "singlets": N_STATES,
            "casscf": {
                "ncas": 2,
                "nelecas": 2,
                "conv_tol": convergence.conv_tol,
                "conv_tol_grad": convergence.conv_tol_grad,
            },
        }
    )


def make_ethylene_backend(
    scratch: Path, *, threads: int = 2, convergence: Convergence = DEFAULT_CONVERGENCE
):
    from gqteashmd.electronic.base import MoleculeSpec
    from gqteashmd.electronic.pyscf_backend import PySCFBackend

    symbols = ethylene_initial_conditions().symbols
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    return PySCFBackend(
        MoleculeSpec(symbols, 0, 1),
        electronic_settings(convergence),
        log_file=scratch / "electronic.log",
        threads=threads,
        memory_mb=2000,
        scratch_dir=scratch,
    )


# --- sharc.x input ----------------------------------------------------------------------------


def sharc_input(
    n_steps: int,
    *,
    coupling: str,
    dt_fs: float = DT_FS,
    nsubsteps: int = NSUBSTEPS,
    state_mch: int = INITIAL_STATE,
    n_states: int = N_STATES,
    seed: int = 1,
    phases_from_interface: bool = True,
    hopping: str = "off",
    decoherence: str = "none",
    decoherence_param_Eh: float = 0.1,
) -> str:
    """The sharc.x ``input`` (manual 4.1) for a singlet trajectory (``surf sharc``, no
    spin-orbit coupling, ``ekincorrect parallel_vel``), by default with hopping off and no
    decoherence (``hopping_procedure``, ``decoherence_scheme``, ``decoherence_param``). Our
    ``state_mch`` is 0-based; SHARC's ``state`` is 1-based. The length is set with ``tmax``
    (``nsteps`` was ignored by this sharc.x, P7a-1). Every step is printed with its gradients
    and its overlaps (``overlap``) or NACs (``nacdr``)."""
    if coupling not in COUPLINGS:
        raise ValueError(f"coupling must be one of {', '.join(COUPLINGS)}; got {coupling!r}")
    if hopping not in HOPPING:
        raise ValueError(f"hopping must be one of {', '.join(HOPPING)}; got {hopping!r}")
    if decoherence not in DECOHERENCE:
        raise ValueError(
            f"decoherence must be one of {', '.join(DECOHERENCE)}; got {decoherence!r}"
        )
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1; got {n_steps}")
    lines = [
        "printlevel 2",
        'geomfile "geom"',
        "veloc external",
        'velocfile "veloc"',
        f"nstates {int(n_states)}",
        "charge 0",
        f"state {int(state_mch) + 1} mch",
        "coeff auto",
        f"stepsize {float(dt_fs)!r}",
        f"tmax {(n_steps + 0.25) * dt_fs!r}",
        f"nsubsteps {int(nsubsteps)}",
        "method tsh",
        "surf sharc",
        "nospinorbit",
        f"coupling {coupling}",
        f"eeom {'ld' if coupling == 'overlap' else 'li'}",
        "nogradcorrect",
        "ekincorrect parallel_vel",
        "reflect_frustrated none",
        f"decoherence_scheme {decoherence}",
    ]
    if decoherence == "edc":
        lines.append(f"decoherence_param {float(decoherence_param_Eh)!r}")
    lines += [f"hopping_procedure {hopping}", "grad_all"]
    if coupling == "nacdr":
        lines.append("nac_all")
    if not phases_from_interface:
        lines.append("nophases_from_interface")
    lines += [f"rngseed {int(seed)}", "output_format ascii", "nowrite_restart_files", "write_grad"]
    lines.append("write_overlap" if coupling == "overlap" else "write_nacdr")
    return "\n".join(lines) + "\n"


def write_trajectory_files(
    work_dir: Path,
    *,
    symbols: Sequence[str],
    masses_amu: Sequence[float],
    q0: np.ndarray,
    v0: np.ndarray,
    input_text: str,
    provider: str,
    sharc_python: str | None = None,
    template: str = PYSCF_TEMPLATE,
) -> None:
    """Write ``input``, ``geom``, ``veloc`` and ``QM/`` in ``work_dir`` (must not exist).
    ``"sharc_pyscf"``: ``QM/runQM.sh`` runs SHARC_PYSCF.py on our PYSCF.template;
    ``"ours"``: ``QM/runQM.sh`` talks to a ``QmServer``."""
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {', '.join(PROVIDERS)}; got {provider!r}")
    work_dir = Path(work_dir)
    qm = work_dir / "QM"
    qm.mkdir(parents=True, exist_ok=False)
    natom = len(symbols)
    (work_dir / "geom").write_text(
        format_geom(symbols, masses_amu, np.asarray(q0, dtype=float).reshape(natom, 3))
    )
    (work_dir / "veloc").write_text(format_veloc(np.asarray(v0, dtype=float).reshape(natom, 3)))
    (work_dir / "input").write_text(input_text)
    if provider == "ours":
        write_server_runqm(qm)
        return
    if sharc_python is None:
        raise ValueError("the sharc_pyscf provider needs sharc_python")
    (qm / "PYSCF.template").write_text(template)
    (qm / "PYSCF.resources").write_text(PYSCF_RESOURCES)
    script = qm / "runQM.sh"
    script.write_text(
        "cd QM/ || exit 1\n"
        "# measured (P7b Task 10): with 'step 0' SHARC_PYSCF.py stops with "
        "\"File .../pyscf.chk not found, cannot move to old!\"; with 'init' it runs\n"
        "sed -i 's/^step[[:space:]]*0[[:space:]]*$/init/' QM.in || exit 1\n"
        f'exec "{sharc_python}" "$SHARC/SHARC_PYSCF.py" QM.in\n'
    )
    script.chmod(0o755)


# --- our electronic structure behind QM.out (swap test) -------------------------------------


@dataclass(frozen=True)
class _QmState:
    multiplicity: int
    index: int  # 0-based within the multiplicity (S0 = 0)
    ms: int = 0


class SwapModel:
    """Our ``PySCFBackend`` as the model of ``run_sharc_lvc.QmServer``: every gradient (sharc.x
    asks for all), the Lowdin-orthonormalised overlaps with the previous call (the matrix our
    ``BackendModel`` hands to our propagator) and, for ``nacdr``, every NAC. A call without a
    previous step resets the backend (a fresh start)."""

    def __init__(self, backend, *, coupling: str) -> None:
        from gqteashmd.core.states import StateLabel

        if coupling not in COUPLINGS:
            raise ValueError(f"coupling must be one of {', '.join(COUPLINGS)}; got {coupling!r}")
        self.backend = backend
        self.labels: tuple[StateLabel, ...] = tuple(backend.states)
        self.states = tuple(_QmState(s.multiplicity, s.index) for s in self.labels)
        self.coupling = coupling
        self.min_singular_values: list[float] = []

    def evaluate(self, q: np.ndarray, previous: Any):
        from gqteashmd.dynamics.records import ElectronicSnapshot
        from gqteashmd.electronic.base import Request
        from gqteashmd.overlaps.tracking import lowdin_orthonormalize

        if previous is None:
            self.backend.reset()
        labels = self.labels
        pairs = (
            frozenset((a, b) for i, a in enumerate(labels) for b in labels[i + 1 :])
            if self.coupling == "nacdr"
            else frozenset()
        )
        result = self.backend.compute(
            Request(
                np.asarray(q, dtype=float).reshape(-1, 3),
                gradients=frozenset(labels),
                nacs=pairs,
                overlaps=True,
            )
        )
        n, f = len(labels), int(np.asarray(q).size)
        index = {label: i for i, label in enumerate(labels)}
        energies = np.array([float(result.energies[s]) for s in labels])
        gradients = np.array([np.asarray(result.gradients[s], float).reshape(-1) for s in labels])
        nacs = np.zeros((n, n, f))
        for (a, b), d in result.nacs.items():
            nacs[index[a], index[b]] = np.asarray(d, dtype=float).reshape(-1)
            nacs[index[b], index[a]] = -nacs[index[a], index[b]]
        if not np.all(np.isfinite(energies)):
            raise ValueError(f"the backend returned non-finite energies {energies.tolist()}")
        if not np.all(np.isfinite(gradients)):
            raise ValueError("the backend returned non-finite gradients")
        if self.coupling == "nacdr" and not np.all(np.isfinite(nacs)):
            raise ValueError("the backend returned non-finite NACs")
        overlaps = None
        if result.overlaps is not None:
            if not np.all(np.isfinite(result.overlaps)):
                raise ValueError(
                    "the backend returned non-finite overlaps with the previous geometry"
                )
            self.min_singular_values.append(
                float(np.linalg.svd(np.asarray(result.overlaps), compute_uv=False).min())
            )
            overlaps = lowdin_orthonormalize(result.overlaps)
        return ElectronicSnapshot(
            energies=energies,
            gradients=gradients,
            nacs=nacs,
            hamiltonian=np.diag(energies).astype(complex),
            overlaps=overlaps,
        )


# --- one sharc.x trajectory -------------------------------------------------------------------


def run_sharc_trajectory(
    geometry_bohr: np.ndarray,
    velocities_au: np.ndarray,
    *,
    n_steps: int,
    workdir: Path,
    symbols: Sequence[str],
    masses_amu: Sequence[float],
    sharc_home: Path,
    provider: str,
    coupling: str,
    sharc_python: str | None = None,
    model: SwapModel | None = None,
    phases_from_interface: bool | None = None,
    convergence: Convergence = DEFAULT_CONVERGENCE,
    timeout_s: float | None = None,
    template: str | None = None,
    n_states: int = N_STATES,
    dt_fs: float = DT_FS,
    state_mch: int = INITIAL_STATE,
    hopping: str = "off",
    decoherence: str = "none",
    seed: int = 1,
    check: bool = True,
    env_extra: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run ``sharc.x input`` in ``workdir`` (must not exist; its own session, a whole-group
    kill on a timeout) and return the per-step data of output.dat in atomic units:
    ``steps`` (k,), ``q`` and ``v`` (k, 3N), ``energies`` (k, n) (the MCH diagonal),
    ``populations`` (k, n) (``|c|^2``), ``active`` (k,) (0-based), ``e_kinetic`` (k,), plus
    ``dt_au``, ``masses_me``, ``wall_time_s`` and the parsed ``trajectory``.
    ``"ours"`` needs ``model`` (a ``SwapModel``) and serves it for the duration of the run.
    ``phases_from_interface`` defaults to False for SHARC_PYSCF.py (measured: it stops at
    step 1 with "'overlap' is not supported" when sharc.x requests PHASES) and True for ours.
    ``convergence`` sets the PYSCF.template CASSCF tolerances (``"sharc_pyscf"`` only);
    ``template`` replaces the whole PYSCF.template (default: ``pyscf_template(convergence,
    roots=n_states)``). ``timeout_s`` None is ``timeout_for(n_steps)``. With ``check=False`` a
    failed sharc.x does not raise: the result has ``returncode``, ``stderr_tail``,
    ``stdout_tail``, ``wall_time_s`` and ``data`` (the reduced partial output.dat, else None);
    a timeout still raises. ``env_extra`` adds variables to sharc.x's environment. A
    successful run has ``returncode`` 0."""
    workdir = Path(workdir)
    if template is None:
        template = pyscf_template(convergence, roots=n_states)
    if timeout_s is None:
        timeout_s = timeout_for(n_steps)
    if phases_from_interface is None:
        phases_from_interface = provider == "ours"
    write_trajectory_files(
        workdir,
        symbols=symbols,
        masses_amu=masses_amu,
        q0=geometry_bohr,
        v0=velocities_au,
        input_text=sharc_input(
            n_steps,
            coupling=coupling,
            dt_fs=dt_fs,
            state_mch=state_mch,
            n_states=n_states,
            seed=seed,
            phases_from_interface=phases_from_interface,
            hopping=hopping,
            decoherence=decoherence,
        ),
        provider=provider,
        sharc_python=sharc_python,
        template=template,
    )
    server = None
    if provider == "ours":
        if model is None:
            raise ValueError("provider 'ours' needs the model")
        server = QmServer(model, workdir / "QM")
    started = time.perf_counter()
    with server if server is not None else contextlib.nullcontext():
        proc = run_in_session(
            ["bash", "-c", SHARC_X_COMMAND],
            cwd=workdir,
            env={**sharc_environment(Path(sharc_home)), **(env_extra or {})},
            timeout=timeout_s,
            what="sharc.x",
        )
    wall = time.perf_counter() - started
    output = workdir / "output.dat"
    if proc.returncode != 0 or not output.is_file():
        if check:
            raise RuntimeError(
                f"sharc.x failed ({proc.returncode}) in {workdir}:\n{proc.stderr[-2000:]}"
                f"\n{proc.stdout[-2000:]}"
            )
        data = None
        if output.is_file():
            try:
                data = trajectory_data(parse_lvc_output_dat(output), wall)
            except (ValueError, IndexError, OSError):
                data = None
        return {
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-2000:],
            "stdout_tail": proc.stdout[-2000:],
            "wall_time_s": wall,
            "data": data,
        }
    return {**trajectory_data(parse_lvc_output_dat(output), wall), "returncode": 0}


def trajectory_data(trajectory, wall_time_s: float = math.nan) -> dict[str, Any]:
    """The per-step arrays of a parsed output.dat (``run_sharc_lvc.LvcTrajectory``)."""
    n = trajectory.hamiltonian.shape[1]
    energies = trajectory.hamiltonian[:, np.arange(n), np.arange(n)].real
    k = np.arange(len(trajectory.steps))
    active_gradient = None
    if trajectory.gradients is not None:
        active_gradient = trajectory.gradients[k, trajectory.active]
    return {
        "steps": trajectory.steps,
        "q": trajectory.q,
        "v": trajectory.v,
        "energies": energies,
        "populations": np.abs(trajectory.c) ** 2,
        "active": trajectory.active,
        "e_kinetic": trajectory.e_kinetic,
        "e_total": energies[k, trajectory.active] + trajectory.e_kinetic,
        "active_gradient": active_gradient,
        "dt_au": trajectory.dt_au,
        "masses_me": trajectory.masses_me,
        "wall_time_s": wall_time_s,
        "trajectory": trajectory,
    }


# --- our trajectory ---------------------------------------------------------------------------


def run_our_trajectory(
    initial: InitialConditions,
    *,
    n_steps: int,
    workdir: Path,
    coupling: str = "overlap",
    threads: int = 2,
    energy_drift_limit_eV: float | None = None,
    energy_jump_limit_eV: float | None = None,
    convergence: Convergence = DEFAULT_CONVERGENCE,
) -> dict[str, Any]:
    """Our ``TrajectoryRunner`` on the same system with the SHARC settings (hopping off, no
    decoherence, ``substep_interpolation = "endpoint"``); the same keys as
    ``run_sharc_trajectory`` (the records on the dt grid) plus ``result`` (the runner's
    summary) and ``rows`` (all of steps.log). ``energy_drift_limit_eV`` None keeps our
    default (0.5 eV). ``energy_jump_limit_eV`` None keeps the library default (the energy-jump
    rejection is off); the ``[dynamics]`` stage default is 0.1 eV."""
    from gqteashmd.dynamics.options import DynamicsOptions
    from gqteashmd.dynamics.randomness import GeneratorSource
    from gqteashmd.trajectory.model import BackendModel
    from gqteashmd.trajectory.runner import InitialConditions as RunnerInitial
    from gqteashmd.trajectory.runner import RunSettings, TrajectoryRunner
    from gqteashmd.trajectory.store import StepLog

    workdir = Path(workdir)
    backend = make_ethylene_backend(workdir / "scratch", threads=threads, convergence=convergence)
    masses = np.repeat(np.asarray(initial.masses_amu) * DALTON_IN_ME, 3)
    dt_au = DT_FS / AU_TIME_IN_FS
    options = DynamicsOptions(
        dt_au=dt_au,
        nsubsteps=NSUBSTEPS,
        hopping="none",
        decoherence="none",
        ekin_correction="parallel_vel",
        coupling=coupling,
        gradient_transformation="none",
        substep_interpolation="endpoint",
    )
    if energy_drift_limit_eV is not None:
        options = dataclasses.replace(
            options, energy_drift_limit_au=energy_drift_limit_eV / HARTREE_IN_EV
        )
    model = BackendModel(backend, masses, coupling=coupling)
    jump_au = None if energy_jump_limit_eV is None else energy_jump_limit_eV / HARTREE_IN_EV
    runner = TrajectoryRunner(
        model,
        RunSettings(options, tmax_au=n_steps * dt_au, energy_jump_limit_au=jump_au),
        workdir / "traj",
    )
    started = time.perf_counter()
    result = runner.run(
        RunnerInitial(initial.q_bohr.reshape(-1), initial.v_au.reshape(-1), INITIAL_STATE),
        GeneratorSource(np.random.default_rng(0)),
    )
    wall = time.perf_counter() - started
    rows = StepLog(workdir / "traj" / "steps.log", model.n_states, model.n_dof).read()
    grid = rows[rows["ticks"] == 0]  # the records on the dt grid (halved steps in between)
    return {
        "steps": grid["grid_step"],
        "q": grid["q"],
        "v": grid["v"],
        "energies": grid["energies"],
        "populations": grid["c_real"] ** 2 + grid["c_imag"] ** 2,
        "active": grid["active"],
        "e_kinetic": grid["e_kinetic"],
        "e_total": grid["e_total"],
        "dt_au": dt_au,
        "masses_me": np.asarray(initial.masses_amu) * DALTON_IN_ME,
        "wall_time_s": wall,
        "result": result,
        "rows": rows,
    }


# --- comparison -------------------------------------------------------------------------------

QUANTITIES = ("q", "v", "energies", "populations")


def step_differences(a: dict[str, Any], b: dict[str, Any]) -> dict[str, np.ndarray]:
    """Per printed step, max |a - b| of positions, velocities, MCH energies and populations,
    over the steps both have (aligned by step number)."""
    for name, side in (("first", a), ("second", b)):
        if not np.all(np.diff(side["steps"]) > 0):
            raise ValueError(f"the {name} trajectory's steps are not strictly increasing")
    common = np.intersect1d(a["steps"], b["steps"])
    ia = np.searchsorted(a["steps"], common)
    ib = np.searchsorted(b["steps"], common)
    out: dict[str, np.ndarray] = {"steps": common}
    for key in QUANTITIES:
        diff = np.abs(np.asarray(a[key])[ia] - np.asarray(b[key])[ib])
        out[key] = diff.reshape(len(common), -1).max(axis=1)
    return out


def format_differences(diffs: dict[str, np.ndarray]) -> str:
    """The per-step table of ``step_differences``."""
    lines = ["step  max|dq| bohr  max|dv| au  max|dE| Eh  max|dP|"]
    for k, step in enumerate(diffs["steps"]):
        values = "  ".join(f"{float(diffs[key][k]):10.3e}" for key in QUANTITIES)
        lines.append(f"{int(step):4d}  {values}")
    return "\n".join(lines)


def _jsonable(diffs: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "steps": [int(s) for s in diffs["steps"]],
        **{f"max_abs_{k}": [float(x) for x in diffs[k]] for k in QUANTITIES},
        **{f"overall_max_abs_{k}": float(np.max(diffs[k])) for k in QUANTITIES},
    }


def active_gradient_differences(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Per common step, max |g_a - g_b| of the active-state gradients (both from output.dat)
    and the max |q_a - q_b| at which they were evaluated."""
    common = np.intersect1d(a["steps"], b["steps"])
    ia = np.searchsorted(a["steps"], common)
    ib = np.searchsorted(b["steps"], common)
    dg = np.abs(a["active_gradient"][ia] - b["active_gradient"][ib]).max(axis=1)
    dq = np.abs(np.asarray(a["q"])[ia] - np.asarray(b["q"])[ib]).max(axis=1)
    return {
        "steps": [int(s) for s in common],
        "max_abs_gradient": [float(x) for x in dg],
        "max_abs_q": [float(x) for x in dq],
        "overall_max_abs_gradient": float(dg.max()),
    }


def relative_population_differences(
    ours: dict[str, Any], reference: dict[str, Any], *, state: int = 0, last: int | None = None
) -> dict[str, Any]:
    """Per common step (up to ``last``), the population of ``state`` in both runs and
    ``|P_ours - P_ref| / P_ref`` (None where P_ref = 0, e.g. at step 0)."""
    common = np.intersect1d(ours["steps"], reference["steps"])
    if last is not None:
        common = common[common <= last]
    p_ours = np.asarray(ours["populations"])[np.searchsorted(ours["steps"], common), state]
    p_ref = np.asarray(reference["populations"])[np.searchsorted(reference["steps"], common), state]
    relative = [
        None if r == 0 else float(abs(o - r) / r) for o, r in zip(p_ours, p_ref, strict=True)
    ]
    return {
        "steps": [int(x) for x in common],
        "state": state,
        "ours": [float(x) for x in p_ours],
        "reference": [float(x) for x in p_ref],
        "relative_difference": relative,
    }


def format_relative_populations(data: dict[str, Any]) -> str:
    """The per-step table of ``relative_population_differences``."""
    lines = [f"step  P(S{data['state']}) ours    P(S{data['state']}) ref     |dP|/P_ref"]
    for step, o, r, rel in zip(
        data["steps"], data["ours"], data["reference"], data["relative_difference"], strict=True
    ):
        lines.append(f"{step:4d}  {o:13.4e}  {r:13.4e}  {'-' if rel is None else f'{rel:.3f}':>10}")
    return "\n".join(lines)


REPORT_DRIFT_LIMIT_EV = 100.0  # the report follows our trajectory past the energy-drift stop


def swap_report(
    *, sharc_home: Path, sharc_python: str, n_steps: int, work_dir: Path, out: Path
) -> dict[str, Any]:
    """The swap-test report (not asserted): our trajectory and sharc.x + our ``PySCFBackend``
    (``QmServer``), both with ``coupling overlap``, our ``coupling nacdr`` trajectory, and
    sharc.x + SHARC_PYSCF.py with ``coupling nacdr`` (measured: SHARC_PYSCF.py has no
    overlaps); every pairwise per-step difference and the relative P(S0) differences of both
    our couplings against SHARC_PYSCF go to ``out`` (JSON)."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=False)
    initial = ethylene_initial_conditions()
    common = {
        "n_steps": n_steps,
        "symbols": initial.symbols,
        "masses_amu": initial.masses_amu,
        "sharc_home": sharc_home,
    }
    ours = run_our_trajectory(
        initial,
        n_steps=n_steps,
        workdir=work_dir / "ours",
        energy_drift_limit_eV=REPORT_DRIFT_LIMIT_EV,
    )
    ours_nacdr = run_our_trajectory(
        initial,
        n_steps=n_steps,
        workdir=work_dir / "ours_nacdr",
        coupling="nacdr",
        energy_drift_limit_eV=REPORT_DRIFT_LIMIT_EV,
    )
    sharc = run_sharc_trajectory(
        initial.q_bohr,
        initial.v_au,
        workdir=work_dir / "sharc_pyscf",
        sharc_python=sharc_python,
        provider="sharc_pyscf",
        coupling="nacdr",
        **common,
    )
    model = SwapModel(make_ethylene_backend(work_dir / "swap_scratch"), coupling="overlap")
    swap = run_sharc_trajectory(
        initial.q_bohr,
        initial.v_au,
        workdir=work_dir / "swap",
        provider="ours",
        coupling="overlap",
        model=model,
        **common,
    )
    pairs = {
        "swap_vs_ours": (swap, ours),
        "sharc_pyscf_vs_ours": (sharc, ours),
        "sharc_pyscf_vs_swap": (sharc, swap),
    }
    report = {
        "schema_version": 1,
        "kind": "sharc_pyscf_dynamics_swap_report",
        "asserted": False,
        "system": (
            f"ethylene SA2-CASSCF(2,2)/{BASIS}, start S1, T = {TEMPERATURE_K} K velocities "
            f"(default_rng({VELOCITY_SEED})), dt {DT_FS} fs, {NSUBSTEPS} substeps, hopping off, "
            "no decoherence; ours and swap: coupling overlap (ours: substep_interpolation "
            "endpoint); sharc_pyscf: coupling nacdr, nophases_from_interface; ours with "
            f"energy_drift_limit_eV = {REPORT_DRIFT_LIMIT_EV} (the asserted test keeps 0.5)"
        ),
        "n_steps": n_steps,
        "wall_time_s": {
            "ours": ours["wall_time_s"],
            "ours_nacdr": ours_nacdr["wall_time_s"],
            "sharc_pyscf": sharc["wall_time_s"],
            "swap": swap["wall_time_s"],
        },
        "our_result": ours["result"],
        "our_nacdr_result": ours_nacdr["result"],
        "swap_min_singular_values": model.min_singular_values,
        "dt_au": {"ours": ours["dt_au"], "sharc": sharc["dt_au"]},
        "masses_me": {
            "ours": [float(m) for m in ours["masses_me"]],
            "sharc": [float(m) for m in sharc["masses_me"]],
        },
        **{name: _jsonable(step_differences(*pair)) for name, pair in pairs.items()},
        "active_gradient_sharc_pyscf_vs_swap": active_gradient_differences(sharc, swap),
        "sharc_pyscf_vs_ours_nacdr": _jsonable(step_differences(sharc, ours_nacdr)),
        "relative_population_s0_vs_sharc_pyscf": {
            "ours": relative_population_differences(ours, sharc),
            "ours_nacdr": relative_population_differences(ours_nacdr, sharc),
        },
        "e_total_Eh": {
            "ours": [float(x) for x in ours["e_total"]],
            "sharc_pyscf": [float(x) for x in sharc["e_total"]],
            "swap": [float(x) for x in swap["e_total"]],
        },
        "energies_Eh": {
            "ours": np.asarray(ours["energies"]).tolist(),
            "sharc_pyscf": np.asarray(sharc["energies"]).tolist(),
            "swap": np.asarray(swap["energies"]).tolist(),
        },
        "populations": {
            "ours": np.asarray(ours["populations"]).tolist(),
            "ours_nacdr": np.asarray(ours_nacdr["populations"]).tolist(),
            "sharc_pyscf": np.asarray(sharc["populations"]).tolist(),
            "swap": np.asarray(swap["populations"]).tolist(),
        },
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    for name, (first, second) in pairs.items():
        print(f"{name}:\n{format_differences(step_differences(first, second))}")
        data = report[name]
        print(
            f"{name}: " + ", ".join(f"{k} {data[f'overall_max_abs_{k}']:.3e}" for k in QUANTITIES)
        )
    for name, data in report["relative_population_s0_vs_sharc_pyscf"].items():
        print(f"P(S0), {name} vs sharc_pyscf:\n{format_relative_populations(data)}")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sharc-home", required=True)
    p.add_argument("--python", required=True, help="Python of the SHARC conda environment")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--work-dir", required=True, help="must not exist")
    p.add_argument(
        "--coupling",
        choices=COUPLINGS,
        default="nacdr",
        help="SHARC_PYSCF.py has no overlaps (measured), so 'overlap' stops at step 1",
    )
    p.add_argument("--report", help="run the swap-test report and write its JSON here")
    args = p.parse_args(argv)
    work = Path(args.work_dir).expanduser()
    if args.report:
        swap_report(
            sharc_home=Path(args.sharc_home),
            sharc_python=args.python,
            n_steps=args.steps,
            work_dir=work,
            out=Path(args.report).expanduser(),
        )
        return
    initial = ethylene_initial_conditions()
    data = run_sharc_trajectory(
        initial.q_bohr,
        initial.v_au,
        n_steps=args.steps,
        workdir=work,
        symbols=initial.symbols,
        masses_amu=initial.masses_amu,
        sharc_home=Path(args.sharc_home),
        sharc_python=args.python,
        provider="sharc_pyscf",
        coupling=args.coupling,
    )
    n_printed = len(data["steps"])
    print(
        f"sharc.x + SHARC_PYSCF.py: {args.steps} steps of {DT_FS} fs ({args.coupling}), "
        f"{n_printed} printed steps, wall time {data['wall_time_s']:.1f} s "
        f"({data['wall_time_s'] / max(1, n_printed):.1f} s per printed step)"
    )
    for k, step in enumerate(data["steps"]):
        print(
            f"{int(step):4d} E = {np.array2string(data['energies'][k], precision=10)} "
            f"P = {np.array2string(data['populations'][k], precision=8)}"
        )


if __name__ == "__main__":
    main()
