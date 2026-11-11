"""Command-line interface: gqteashmd init | validate | run | status | stop."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd import __version__
from gqteashmd.chem.xyz import parse_xyz
from gqteashmd.config.loader import load_config
from gqteashmd.config.model_dynamics import LvcDynamicsInput, load_model_dynamics
from gqteashmd.core.units import ANGSTROM_TO_BOHR
from gqteashmd.dynamics.ensemble import (
    lvc_initial_conditions,
    run_model_ensemble,
    run_population_ensemble,
)
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.factory import make_backend
from gqteashmd.errors import ConfigError, ProjectError, ShmdError
from gqteashmd.logging_setup import FILE_ONLY, close_logging, configure_logging
from gqteashmd.orchestrator import Orchestrator
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.project.lock import locked_by
from gqteashmd.project.state import StateStore, TaskStatus
from gqteashmd.stages.pipeline import (
    DEFAULT_PIPELINE,
    check_stage_inputs,
    check_stage_names,
    import_pipeline,
)
from gqteashmd.util.fileio import write_json_atomic

EXIT_STOPPED = 5

TEMPLATE = """\
# gqteashmd input file. Paths are relative to this file.
schema_version = 1

[project]
name = "my_molecule"          # letters, digits, _ . -
directory = "run"             # results go here; use a local Linux disk (not OneDrive, not /mnt/c)
seed = 12345                  # master random seed for the whole workflow

[molecule]
geometry = "molecule.xyz"     # XYZ file, Angstrom
charge = 0
multiplicity = 1

[resources]
total_cores = 8               # REQUIRED: cores of this workstation you allow gqteashmd to use
total_memory_mb = 16000       # REQUIRED: RAM (MB) you allow gqteashmd to use
max_parallel_tasks = 1        # tasks running at the same time
threads_per_task = 1          # OpenMP/BLAS threads per task
memory_mb_per_task = 2000     # passed to PySCF as max_memory (MB)
# Per-stage overrides (unset keys use the values above), e.g.:
# [resources.stages.prepare]
# threads_per_task = 2

[execution]
max_attempts = 2              # attempts per task for retryable failures
stop_grace_period_s = 30.0    # time running tasks get to stop after `gqteashmd stop`
min_free_disk_mb = 1000      # refuse to start with less free disk space (MB)

[workflow]
pause_after = []              # stage names to pause after for review, e.g. ["selection"]

# Electronic structure (used by optimize, frequencies and the later quantum-chemistry stages):
[electronic]
method = "tda"                # "casscf" | "tda" | "tddft"
basis = "def2-svp"
functional = "b3lyp"          # not used by method = "casscf"; use "hf" for CIS/TDHF
singlets = 3                  # number of singlet states INCLUDING S0
# CASSCF alternative (replace the block above):
# [electronic]
# method = "casscf"
# basis = "6-31g*"
# singlets = 2
# [electronic.casscf]
# ncas = 2                      # active orbitals
# nelecas = 2                   # active electrons

[optimize]
enabled = true                # false: use the input geometry as it is
max_steps = 100
convergence = "GAU"           # geomeTRIC set: GAU, GAU_LOOSE, GAU_TIGHT, GAU_VERYTIGHT, ...

[frequencies]
method = "auto"               # "auto" | "analytic" | "finite_difference" (no analytic for casscf)
fd_step_bohr = 0.005          # displacement for finite differences
imaginary_cutoff_cm1 = 20.0   # imaginary modes above this wavenumber are errors

[sampling]
n_samples = 200               # number of Wigner-sampled initial geometries and velocities
temperature_K = 0.0           # 0 = zero-point (vibrational ground state) distribution
low_frequency_cutoff_cm1 = 10.0   # modes below this wavenumber are not sampled

[excitations]
include_reference = true      # also compute the vertical excitations at the reference geometry
samples_per_task = 1          # samples per task (raise for cheap methods: start-up dominates)
max_failed_fraction = 0.05    # stop if more samples than this fraction fail
broadening_fwhm_eV = 0.1      # Gaussian width (FWHM) of the nuclear-ensemble spectrum
spectrum_step_eV = 0.01       # energy grid step of the spectrum

[selection]
# n_samples x (number of states) x acceptance (typically 0.1-0.3) must reach n_trajectories,
# or the stage stops (set allow_fewer = true to accept fewer). f_over_dE2 favours low-energy
# pairs: set energy_window_eV around the band of interest.
mode = "bright"               # "bright": accept by oscillator strength; "fixed": one state
weighting = "f_over_dE2"      # "f_over_dE2" (SHARC excite.py-compatible) or "f"
# states = ["S1", "S2"]       # candidate excited states (default: all excited singlets)
# energy_window_eV = [4.5, 6.0]   # accepted excitation energy range (default: none)
n_trajectories = 100          # number of trajectories to start
allow_fewer = false           # accept fewer trajectories if too few pass the selection

# [dynamics]                         # remove the leading '# ' to run trajectories
# tmax_fs = 100.0                    # must be a multiple of dt_fs
# dt_fs = 0.5
# coupling = "overlap"               # "nacdr" needs method = "casscf"
# hopping = "sharc"
# decoherence = "edc"
# energy_drift_limit_eV = 0.5"""


def _cmd_init(args: argparse.Namespace) -> int:
    path: Path = args.path
    if path.exists() and not args.force:
        raise ConfigError(f"{path} already exists; use --force to overwrite it.")
    try:
        path.write_text(TEMPLATE, encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot write {path}: {exc.strerror or exc}") from exc
    print(f"Wrote template input file {path}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    loaded = load_config(args.input)
    cfg = loaded.config
    if not cfg.molecule.geometry.is_file():
        raise ConfigError(f"Geometry file not found or not a regular file: {cfg.molecule.geometry}")
    stages = import_pipeline(args.pipeline)(cfg)
    check_stage_names(cfg, [s.name for s in stages])
    check_stage_inputs(cfg, stages)
    print(f"Input file {loaded.source} is valid.")
    print(f"  project directory: {cfg.project.directory}")
    print(f"  geometry:          {cfg.molecule.geometry}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    loaded = load_config(args.input)
    orchestrator = Orchestrator(loaded, pipeline_ref=args.pipeline, rerun_from=args.rerun_from)
    warnings = orchestrator.preflight()
    stop_requested = threading.Event()
    logger = logging.getLogger("gqteashmd.cli")

    def _on_signal(signum: int, _frame: object) -> None:
        if not stop_requested.is_set():
            logger.warning("Stop requested (signal %d): stopping running tasks...", signum)
        stop_requested.set()

    # Installed before the lock is taken and restored after it is released, so a
    # `gqteashmd stop` during start-up is a graceful stop (the pipeline checks should_stop first).
    previous: dict[int, Any] = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, _on_signal)
        with ExitStack() as stack:
            try:
                stack.enter_context(orchestrator.session())
                configure_logging(orchestrator.layout.logs_dir / "gqteashmd.log")
            except OSError as exc:
                raise ConfigError(
                    f"Cannot create project directory {orchestrator.layout.root}: "
                    f"{exc.strerror or exc}"
                ) from exc
            for message in warnings:
                logger.warning("Preflight: %s", message)
            summary = orchestrator.run(should_stop=stop_requested.is_set)
    finally:
        for prev_sig, handler in previous.items():
            signal.signal(prev_sig, handler)
    if summary.stopped:
        print("Stopped. Run the same command again to resume.")
        return EXIT_STOPPED
    if summary.paused_after is not None:
        print(
            f"Paused after stage '{summary.paused_after}' as requested (workflow.pause_after). "
            "Review its results, then run the same command again to continue."
        )
        return EXIT_STOPPED
    print(
        f"Workflow complete: {len(summary.completed)} stage(s) run, "
        f"{len(summary.up_to_date)} already up to date, {len(summary.skipped)} skipped."
    )
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    cfg = load_config(args.input).config
    layout = ProjectLayout(cfg.project.directory)
    if not layout.state_db.exists():
        print(f"Project '{cfg.project.name}' has not been started (no state in {layout.root}).")
        return 0
    pid = locked_by(layout.lock_file)
    print(f"Project: {cfg.project.name}  ({layout.root})")
    print(f"Controller: {f'running (pid {pid})' if pid else 'not running'}")
    print(
        f"{'stage':<16}{'status':<13}{'done':>6}{'failed':>8}{'pending':>9}{'running':>9}  message"
    )
    with StateStore(layout.state_db, readonly=True) as state:
        for stage in import_pipeline(args.pipeline)(cfg):
            record = state.get_stage(stage.name)
            counts = state.task_counts(stage.name)
            status = record.status.value if record else "not started"
            message = (record.message or "").splitlines()[0] if record and record.message else ""
            print(
                f"{stage.name:<16}{status:<13}{counts[TaskStatus.DONE]:>6}"
                f"{counts[TaskStatus.FAILED]:>8}{counts[TaskStatus.PENDING]:>9}"
                f"{counts[TaskStatus.RUNNING]:>9}  {message}"
            )
    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    cfg = load_config(args.input).config
    pid = locked_by(ProjectLayout(cfg.project.directory).lock_file)
    if pid is None:
        print("No running controller for this project.")
        return 0
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        raise ProjectError(
            f"Cannot signal controller pid {pid}: {exc.strerror or exc}. "
            "It may have just exited, or belong to another user."
        ) from exc
    print(f"Stop requested for controller pid {pid}; running tasks are being stopped.")
    return 0


def _check_json_output(path: Path) -> None:
    """The parent of the --json file must be an existing writable directory (nothing is created)."""
    parent = path.parent
    if path.is_dir():
        raise ConfigError(f"--json {path}: this is a directory, not a file.")
    if not parent.is_dir():
        raise ConfigError(f"--json {path}: the directory {parent} does not exist.")
    if not os.access(parent, os.W_OK | os.X_OK):
        raise ConfigError(f"--json {path}: the directory {parent} is not writable.")


def _check_log_output(path: Path) -> None:
    """Create the parent of the --log file (as the backend would) and check that it opens."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8"):
            pass
    except OSError as exc:
        raise ConfigError(f"--log {path}: cannot be used: {exc.strerror or exc}") from exc


def _cmd_single_point(args: argparse.Namespace) -> int:
    cfg = load_config(args.input).config
    if args.json is not None:
        _check_json_output(args.json)
    if args.log is not None:
        _check_log_output(args.log)
    path = cfg.molecule.geometry
    if not path.is_file():
        raise ConfigError(f"Geometry file not found or not a regular file: {path}")
    try:
        geometry = parse_xyz(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise ConfigError(f"Geometry file {path} is not valid UTF-8 text.") from exc
    except OSError as exc:
        raise ConfigError(f"Cannot read geometry file {path}: {exc.strerror or exc}") from exc
    except ValueError as exc:
        raise ConfigError(f"Invalid geometry file {path}: {exc}") from exc
    # The backend validates charge/multiplicity against the elements in its constructor.
    molecule = MoleculeSpec(geometry.symbols, cfg.molecule.charge, cfg.molecule.multiplicity)
    backend = make_backend(
        cfg,
        molecule,
        log_file=args.log,
        threads=cfg.resources.threads_per_task,
        memory_mb=cfg.resources.memory_mb_per_task,
    )
    coords = np.array(geometry.coords_angstrom, dtype=float) * ANGSTROM_TO_BOHR
    result = backend.compute(Request(coords, transition_dipoles=True))
    data = result.to_json()
    settings = cfg.electronic
    assert settings is not None  # make_backend checked it
    name = f"{settings.method} / {settings.basis}"
    if settings.functional:
        name += f" / {settings.functional}"
    print(f"Single point: {name} (PySCF {result.meta['pyscf_version']})")
    print(f"{'state':<6}{'energy (Eh)':>18}{'dE (eV)':>10}{'f':>9}")
    for row in data["states"]:
        f = "-" if row["oscillator_strength"] is None else f"{row['oscillator_strength']:.4f}"
        print(f"{row['label']:<6}{row['energy_hartree']:>18.9f}{row['excitation_ev']:>10.4f}{f:>9}")
    if args.json is not None:
        write_json_atomic(args.json, data)
        print(f"Results written to {args.json}")
    return 0


def _cmd_model_dynamics(args: argparse.Namespace) -> int:
    cfg = load_model_dynamics(args.input)
    if args.json is not None:
        _check_json_output(args.json)
    if isinstance(cfg, LvcDynamicsInput):
        return _run_lvc_dynamics(args, cfg)
    try:
        model = cfg.build_model()
        options = cfg.build_options()
    except ValueError as exc:
        raise ConfigError(f"Invalid model-dynamics input {args.input}: {exc}") from exc
    # Every scalar argument below was validated by the input schema, so a ValueError from the
    # run is raised inside a trajectory (e.g. degenerate states for chosen model parameters).
    try:
        result = run_model_ensemble(
            model,
            n_trajectories=cfg.ensemble.n_trajectories,
            x0=cfg.initial.x0,
            k=cfg.initial.k,
            sigma=cfg.sigma,
            initial_state=cfg.initial.state,
            options=options,
            seed=cfg.ensemble.seed,
            x_exit=cfg.ensemble.x_exit,
            max_steps=cfg.ensemble.max_steps,
            sampling=cfg.initial.sampling,
            workers=cfg.ensemble.workers,
        )
    except ValueError as exc:
        raise ShmdError(
            f"model-dynamics run failed inside a trajectory: {exc} "
            "(check the model parameters and the [dynamics] settings)"
        ) from exc
    counts, probs, intervals = result.counts(), result.probabilities(), result.intervals_95()
    print(
        f"Model dynamics: {cfg.model.name}, {cfg.ensemble.n_trajectories} trajectories, "
        f"{result.n_finished} finished"
    )
    print(f"{'branch':<20}{'count':>8}{'P':>9}{'95% interval':>22}")
    for name, p in probs.items():
        lo, hi = intervals[name]
        interval = f"[{lo:.4f}, {hi:.4f}]"
        print(f"{name:<20}{counts[name]:>8}{p:>9.4f}{interval:>22}")
    print(f"unfinished: {counts['unfinished']}, energy_drift: {counts['energy_drift']}")
    if args.json is not None:
        data = {
            "schema_version": 1,
            "model": cfg.model.model_dump(),
            "initial": {**cfg.initial.model_dump(), "sigma": cfg.sigma},
            "dynamics": cfg.dynamics.model_dump(),
            "ensemble": cfg.ensemble.model_dump(),
            "counts": counts,
            "probabilities": probs,
            "intervals_95": {name: list(v) for name, v in intervals.items()},
            "n_finished": result.n_finished,
        }
        write_json_atomic(args.json, data)
        print(f"Results written to {args.json}")
    return 0


def _run_lvc_dynamics(args: argparse.Namespace, cfg: LvcDynamicsInput) -> int:
    try:
        model = cfg.build_model()
        options = cfg.build_options()
        initial = cfg.initial_index(model)
    except ValueError as exc:
        raise ConfigError(f"Invalid model-dynamics input {args.input}: {exc}") from exc
    ens = cfg.ensemble
    q0, v0 = lvc_initial_conditions(model, ens.n_trajectories, ens.seed, cfg.initial.sampling)
    started = time.perf_counter()
    try:
        result = run_population_ensemble(
            model,
            q0,
            v0,
            initial_state=initial,
            options=options,
            seed=ens.seed,
            n_steps=ens.n_steps,
            record_every=ens.record_every,
            workers=ens.workers,
        )
    except ValueError as exc:
        raise ShmdError(
            f"model-dynamics run failed inside a trajectory: {exc} "
            "(check the model and the [dynamics] settings)"
        ) from exc
    if result.n_energy_drift == result.n_trajectories:
        # every average would be NaN (not valid JSON): no populations, no JSON file
        raise ShmdError(
            f"all {result.n_trajectories} trajectories ended with energy_drift "
            "(reduce dynamics.dt_au or raise dynamics.energy_drift_limit_au)"
        )
    elapsed = time.perf_counter() - started
    print(
        f"LVC dynamics: {cfg.initial.state}, {ens.n_trajectories} trajectories, "
        f"{ens.n_steps} steps of {options.dt_au} a.u."
    )
    print(f"final spin-free MCH populations at t = {result.times_au[-1]:.1f} a.u.:")
    print(f"{'state':<8}{'classical':>12}{'quantum':>12}")
    for i, label in enumerate(result.labels_spin_free):
        print(
            f"{label:<8}{result.mch_spin_free_classical[-1, i]:>12.4f}"
            f"{result.mch_spin_free_quantum[-1, i]:>12.4f}"
        )
    print(
        f"hops: {result.n_hops}, frustrated hops: {result.n_frustrated}, "
        f"energy_drift (excluded): {result.n_energy_drift}, elapsed: {elapsed:.1f} s"
    )
    if args.json is not None:
        data = {
            "kind": "lvc_populations",
            "schema_version": 1,
            "model": cfg.model.model_dump(),
            "initial": cfg.initial.model_dump(),
            "dynamics": cfg.dynamics.model_dump(),
            "ensemble": ens.model_dump(),
            "times_au": result.times_au.tolist(),
            "labels_expanded": list(result.labels_expanded),
            "labels_spin_free": list(result.labels_spin_free),
            "populations": {
                name: getattr(result, name).tolist()
                for name in (
                    "diag_classical",
                    "diag_quantum",
                    "mch_classical",
                    "mch_quantum",
                    "mch_spin_free_classical",
                    "mch_spin_free_quantum",
                )
            },
            "counts": {
                "n_trajectories": result.n_trajectories,
                "n_energy_drift": result.n_energy_drift,
                "n_included": result.n_trajectories - result.n_energy_drift,
                "n_hops": result.n_hops,
                "n_frustrated": result.n_frustrated,
            },
            "elapsed_s": elapsed,
        }
        write_json_atomic(args.json, data)
        print(f"Results written to {args.json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gqteashmd", description="Surface-hopping MD workflow")
    parser.add_argument("--version", action="version", version=f"gqteashmd {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="write a commented template input file")
    p_init.add_argument("path", type=Path)
    p_init.add_argument("--force", action="store_true", help="overwrite an existing file")
    p_init.set_defaults(func=_cmd_init)

    p_validate = sub.add_parser("validate", help="check an input file without running anything")
    p_validate.add_argument("input", type=Path)
    p_validate.add_argument("--pipeline", default=DEFAULT_PIPELINE, help=argparse.SUPPRESS)
    p_validate.set_defaults(func=_cmd_validate)

    p_run = sub.add_parser("run", help="run (or resume) the workflow")
    p_run.add_argument("input", type=Path)
    p_run.add_argument("--rerun-from", metavar="STAGE", help="discard results from STAGE onward")
    p_run.add_argument("--pipeline", default=DEFAULT_PIPELINE, help=argparse.SUPPRESS)
    p_run.set_defaults(func=_cmd_run)

    p_status = sub.add_parser("status", help="show progress of a project")
    p_status.add_argument("input", type=Path)
    p_status.add_argument("--pipeline", default=DEFAULT_PIPELINE, help=argparse.SUPPRESS)
    p_status.set_defaults(func=_cmd_status)

    p_stop = sub.add_parser("stop", help="ask the running controller to stop gracefully")
    p_stop.add_argument("input", type=Path)
    p_stop.set_defaults(func=_cmd_stop)

    p_sp = sub.add_parser(
        "single-point", help="compute energies and transition dipoles at the input geometry"
    )
    p_sp.add_argument("input", type=Path)
    p_sp.add_argument("--json", type=Path, help="write the results to this JSON file")
    p_sp.add_argument("--log", type=Path, help="write the PySCF output to this file")
    p_sp.set_defaults(func=_cmd_single_point)

    p_md = sub.add_parser(
        "model-dynamics",
        help="run a surface-hopping ensemble on an analytic model (Tully 1D or LVC)",
    )
    p_md.add_argument("input", type=Path)
    p_md.add_argument("--json", type=Path, help="write the branching results to this JSON file")
    p_md.set_defaults(func=_cmd_model_dynamics)
    return parser


def main(argv: list[str] | None = None) -> int:
    log = logging.getLogger("gqteashmd.cli")
    try:
        args = build_parser().parse_args(argv)
        code: int = args.func(args)
        return code
    except KeyboardInterrupt:
        print("gqteashmd: interrupted", file=sys.stderr)
        return EXIT_STOPPED
    except ShmdError as exc:
        log.error("%s", exc, extra=FILE_ONLY)
        print(f"gqteashmd: error: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:
        log.error("Unexpected error", exc_info=True, extra=FILE_ONLY)
        print(
            f"gqteashmd: error: internal error: {type(exc).__name__}: {exc} "
            "(please report this; details in the project log if one was configured)",
            file=sys.stderr,
        )
        return 1
    finally:
        close_logging()
