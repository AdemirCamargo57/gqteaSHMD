"""PySCF backend: restricted SCF, SA-CASSCF and TDA/TDDFT energies, gradients, dipoles.

PySCF is imported lazily inside functions so that the rest of gqteashmd never needs it.
Coordinates are passed to PySCF in bohr; all results are in atomic units.
"""

from __future__ import annotations

import io
import json
import logging
import re
import sys
import time
import warnings
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import numpy as np
from numpy.typing import ArrayLike

from gqteashmd.chem.electrons import count_electrons
from gqteashmd.chem.elements import atomic_number
from gqteashmd.config.schema import ElectronicSection
from gqteashmd.core.states import StateLabel, states_from_counts
from gqteashmd.electronic import pyscf_casscf, pyscf_response
from gqteashmd.electronic.base import (
    Capabilities,
    ElectronicBackend,
    ElectronicResult,
    MoleculeSpec,
    Request,
)
from gqteashmd.electronic.pyscf_errors import wrap_failure
from gqteashmd.errors import ConfigError, ElectronicStructureError
from gqteashmd.overlaps.determinants import DeterminantExpansion, state_overlaps, truncate
from gqteashmd.overlaps.tracking import track
from gqteashmd.util.fileio import write_bytes_atomic

_LOG = logging.getLogger(__name__)

MIN_PYSCF_VERSION = (2, 6, 0)
_STATE_FORMAT = 1
_STATE_ARRAYS = ("coords", "mo_coeff", "density", "alpha", "beta", "coeffs")

# (name, level shift / Eh, damping factor, second-order solver)
_SCF_LADDER: tuple[tuple[str, float, float, bool], ...] = (
    ("default", 0.0, 0.0, False),
    ("level shift + damping", 0.2, 0.5, False),
    ("second-order", 0.0, 0.0, True),
)


@dataclass(eq=False)
class _Snapshot:
    """What the next geometry needs from this one; save_state() writes exactly this."""

    coords: np.ndarray  # bohr
    mo_coeff: np.ndarray  # MOs the expansion refers to (CASSCF: mc.mo_coeff; else mf.mo_coeff)
    density: np.ndarray  # SCF density matrix, the next SCF guess
    expansion: DeterminantExpansion  # truncated and phase-aligned


@dataclass(eq=False)
class _Live:
    """PySCF objects of the last compute(), reused by compute_more() (never serialised)."""

    mol: Any
    mf: Any
    solver: Any  # CASSCF object, TD object or None
    energies: dict[StateLabel, float]
    phases: np.ndarray


def require_pyscf() -> str:
    """Return the installed PySCF version; ConfigError if missing or too old."""
    try:
        import pyscf
    except ImportError as exc:
        raise ConfigError(
            "PySCF is not installed in this environment; install it with conda (pyscf)."
        ) from exc
    version: str = pyscf.__version__
    parts = tuple(int(p) for p in re.findall(r"\d+", version)[:3])
    if parts < MIN_PYSCF_VERSION:
        raise ConfigError(f"PySCF {version} is too old; gqteashmd needs PySCF >= 2.6.0.")
    return version


class PySCFBackend(ElectronicBackend):
    def __init__(
        self,
        molecule: MoleculeSpec,
        settings: ElectronicSection,
        *,
        log_file: Path | None = None,
        threads: int | None = None,
        memory_mb: int | None = None,
        scratch_dir: Path | None = None,
    ) -> None:
        self._version = require_pyscf()
        if molecule.multiplicity != 1:
            raise ConfigError(
                "The PySCF backend supports closed-shell singlet molecules only "
                "(molecule.multiplicity = 1)."
            )
        try:
            count_electrons(
                [atomic_number(s) for s in molecule.symbols], molecule.charge, molecule.multiplicity
            )
        except ValueError as exc:
            raise ConfigError(f"{exc} Check [molecule] in the input file.") from exc
        self._molecule = molecule
        self._settings = settings
        self._log_file = log_file
        self._threads = threads
        self._memory_mb = memory_mb
        self._states = states_from_counts(settings.singlets, settings.triplets)
        self._scratch_dir = scratch_dir
        self._snapshot: _Snapshot | None = None
        self._live: _Live | None = None

    @property
    def states(self) -> tuple[StateLabel, ...]:
        return self._states

    # ------------------------------------------------------------------ public API
    def compute(self, request: Request) -> ElectronicResult:
        """Evaluate the request at the given geometry.

        Whenever a previous geometry exists the states are tracked against it and the phases are
        applied to the stored expansion, the NACs and the dipoles; `request.overlaps` only
        decides whether `overlaps` and `phases` are returned.
        """
        self._validate(request)
        self._live = None  # a failing compute() must not leave compute_more() on a stale geometry
        started = time.perf_counter()
        previous = self._snapshot
        with self._environment() as stdout:
            mol = self._build_mol(request.coords_bohr, stdout)
            mf, scf_rung = self._run_scf(mol, None if previous is None else previous.density)
            if self._settings.method == "casscf":
                prev_orbitals = None
                if previous is not None:
                    prev_orbitals = (previous.mo_coeff, self._build_mol(previous.coords, None))
                outcome = pyscf_casscf.run_casscf(
                    mol, mf, self._settings, self._states, prev_orbitals
                )
                solver, energies, spins = outcome.mc, outcome.energies, outcome.spin_squares
                mo_coeff = np.asarray(solver.mo_coeff)
                expansion = pyscf_casscf.casscf_expansion(solver)
                rung_meta: dict[str, Any] = {"casscf": outcome.rung}
            else:
                outcome_r = pyscf_response.run_response(mf, self._settings, self._states)
                solver, energies, spins = outcome_r.td, outcome_r.energies, {}
                mo_coeff = np.asarray(mf.mo_coeff)
                expansion = pyscf_response.response_expansion(mf, solver, len(self._states) - 1)
                rung_meta = {"td": outcome_r.rung}
            try:
                expansion = truncate(expansion, self._settings.overlap_norm_kept)
            except (ValueError, MemoryError) as exc:
                raise wrap_failure("The wavefunction overlap", exc) from exc
            phases = np.ones(len(self._states))
            overlaps = None
            if previous is None:
                rung_meta["overlaps"] = "no previous geometry"
            else:
                try:
                    tracked = track(self._state_overlaps(previous, mol, mo_coeff, expansion))
                except (ValueError, MemoryError) as exc:  # LinAlgError is a ValueError
                    raise wrap_failure("The wavefunction overlap", exc) from exc
                phases = tracked.phases
                expansion = expansion.with_signs(phases)
                if request.overlaps:
                    overlaps = tracked.overlaps
                rung_meta.update(
                    overlaps="computed" if request.overlaps else "tracked, not returned",
                    overlap_min_singular_value=tracked.min_singular_value,
                    root_flips=[[i, j] for i, j in tracked.root_flips],
                )
                if tracked.min_singular_value < 0.5:
                    _LOG.warning(
                        "The state overlap matrix is far from unitary (smallest singular value "
                        "%.3f): an intruder state or too few computed states.",
                        tracked.min_singular_value,
                    )
            live = _Live(mol, mf, solver, energies, phases)
            gradients, nacs = self._derivatives(live, request.gradients, request.nacs)
            dipoles = self._dipoles(live, request.transition_dipoles)
        self._snapshot = _Snapshot(
            np.array(request.coords_bohr), mo_coeff, np.asarray(mf.make_rdm1()), expansion
        )
        self._live = live
        meta: dict[str, Any] = {
            "backend": "pyscf",
            "pyscf_version": self._version,
            "method": self._settings.method,
            "basis": self._settings.basis,
            "functional": self._settings.functional,
            "scf": scf_rung,
            "scf_energy_hartree": float(mf.e_tot),
            **rung_meta,
            "wall_time_s": time.perf_counter() - started,
        }
        return ElectronicResult(
            states=self._states,
            energies=energies,
            gradients=gradients,
            transition_dipoles=dipoles,
            spin_squares=spins,
            nacs=nacs,
            overlaps=overlaps,
            phases=None if overlaps is None else phases,
            meta=meta,
        )

    def reset(self) -> None:
        """Forget the previous geometry: the next compute() is a first geometry."""
        self._snapshot = None
        self._live = None

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            gradients=True,
            nacs=self._settings.method == "casscf",
            overlaps=True,
            transition_dipoles=True,
            hessian=self._settings.method != "casscf",
        )

    def ground_state_hessian(self, coords_bohr: ArrayLike) -> np.ndarray:
        """Analytic RHF/RKS Hessian (Eh/bohr²) from a fresh SCF; leaves the snapshot alone."""
        if self._settings.method == "casscf":
            raise ValueError(
                "There is no analytic Hessian for method = 'casscf'; use finite differences of "
                "S0 gradients (frequencies.method = 'finite_difference')."
            )
        coords = np.asarray(coords_bohr, dtype=float)
        n = len(self._molecule.symbols)
        if coords.shape != (n, 3):
            raise ValueError(f"coords_bohr must have shape ({n}, 3)")
        with self._environment() as stdout:
            mol = self._build_mol(coords, stdout)
            mf, _ = self._run_scf(mol, None)
            try:
                h4 = mf.Hessian().kernel()
            except Exception as exc:
                raise wrap_failure("The analytic SCF Hessian", exc) from exc
        return np.asarray(h4, dtype=float).transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)

    def compute_more(
        self,
        gradients: frozenset[StateLabel] = frozenset(),
        nacs: frozenset[tuple[StateLabel, StateLabel]] = frozenset(),
    ) -> ElectronicResult:
        """Add gradients and NACs; uses the PySCF objects of the last successful compute()."""
        live = self._live
        if live is None:
            raise ValueError("compute_more() needs a preceding compute() in this process.")
        self._check_labels(frozenset(gradients), frozenset(nacs))
        with self._environment() as stdout:
            out = stdout if stdout is not None else sys.stdout
            for obj in (live.mol, live.mf, live.solver):
                if obj is not None:
                    obj.stdout = out  # the file of the previous compute() is closed
            grads, phased = self._derivatives(live, frozenset(gradients), frozenset(nacs))
        return ElectronicResult(
            states=self._states,
            energies=dict(live.energies),
            gradients=grads,
            nacs=phased,
            meta={"backend": "pyscf", "method": self._settings.method, "compute_more": True},
        )

    def _fingerprint(self) -> str:
        s = self._settings
        return json.dumps(
            {
                "method": s.method,
                "basis": s.basis,
                "functional": s.functional,
                "symbols": list(self._molecule.symbols),
                "charge": self._molecule.charge,
                "multiplicity": self._molecule.multiplicity,
                "singlets": s.singlets,
                "triplets": s.triplets,
                # only the active space; tolerances do not invalidate a saved state
                "casscf": None
                if s.casscf is None
                else {"ncas": s.casscf.ncas, "nelecas": s.casscf.nelecas},
                "overlap_norm_kept": s.overlap_norm_kept,
            },
            sort_keys=True,
        )

    def save_state(self, path: Path) -> None:
        snap = self._snapshot
        if snap is None:
            raise ValueError("There is no electronic state to save: call compute() first.")
        buffer = io.BytesIO()
        np.savez(
            buffer,
            format=np.array(_STATE_FORMAT),
            fingerprint=np.array(self._fingerprint()),
            coords=snap.coords,
            mo_coeff=snap.mo_coeff,
            density=snap.density,
            alpha=snap.expansion.alpha,
            beta=snap.expansion.beta,
            coeffs=snap.expansion.coeffs,
        )
        try:
            write_bytes_atomic(Path(path), buffer.getvalue())
        except OSError as exc:
            raise wrap_failure(f"Writing the electronic state {path}", exc) from exc

    def load_state(self, path: Path) -> None:
        def unreadable(reason: str) -> ValueError:
            return ValueError(f"{path} is not a readable gqteashmd electronic state: {reason}")

        body: dict[str, np.ndarray] | None = None
        fingerprint = ""
        try:
            with np.load(Path(path), allow_pickle=False) as data:
                version = int(data["format"])  # read first: a newer format may lack other fields
                if version == _STATE_FORMAT:
                    fingerprint = str(data["fingerprint"])
                    body = {k: np.array(data[k]) for k in _STATE_ARRAYS}
        except (OSError, EOFError, zipfile.BadZipFile, KeyError, TypeError, ValueError) as exc:
            raise unreadable(f"{type(exc).__name__}: {exc}") from exc
        if body is None:
            raise ValueError(f"{path}: unsupported electronic state format {version}.")
        if fingerprint != self._fingerprint():
            raise ValueError(
                f"The electronic state in {path} was saved with different settings or a "
                "different molecule; it cannot be reused."
            )
        natom = len(self._molecule.symbols)
        if body["coords"].shape != (natom, 3):
            raise unreadable(f"coords has shape {body['coords'].shape}, expected ({natom}, 3)")
        if body["mo_coeff"].ndim != 2:
            raise unreadable(f"mo_coeff must be 2-D, got shape {body['mo_coeff'].shape}")
        nao = body["mo_coeff"].shape[0]
        if body["density"].shape != (nao, nao):
            raise unreadable(f"density has shape {body['density'].shape}, expected ({nao}, {nao})")
        try:
            expansion = DeterminantExpansion(body["alpha"], body["beta"], body["coeffs"])
        except ValueError as exc:
            raise unreadable(str(exc)) from exc
        self._snapshot = _Snapshot(body["coords"], body["mo_coeff"], body["density"], expansion)
        self._live = None

    # ------------------------------------------------------------------ setup
    @contextmanager
    def _environment(self) -> Iterator[IO[str] | None]:
        from pyscf import lib

        if self._threads is not None:
            lib.num_threads(self._threads)
        old_tmpdir = lib.param.TMPDIR
        try:
            if self._scratch_dir is not None:
                try:
                    self._scratch_dir.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    raise wrap_failure(
                        f"Preparing the scratch directory {self._scratch_dir}", exc
                    ) from exc
                lib.param.TMPDIR = str(self._scratch_dir)
            if self._log_file is None:
                yield None
            else:
                self._log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self._log_file, "a", encoding="utf-8") as fh:
                    yield fh
        finally:
            lib.param.TMPDIR = old_tmpdir

    def _validate(self, request: Request) -> None:
        self._check_labels(request.gradients, request.nacs)
        if len(request.coords_bohr) != len(self._molecule.symbols):
            raise ValueError(
                f"Got coordinates for {len(request.coords_bohr)} atoms; the molecule has "
                f"{len(self._molecule.symbols)} atoms."
            )

    def _check_labels(
        self,
        gradients: frozenset[StateLabel],
        nacs: frozenset[tuple[StateLabel, StateLabel]],
    ) -> None:
        unknown = sorted(str(s) for s in {s for pair in nacs for s in pair} - set(self._states))
        if unknown:
            raise ValueError(f"NACs requested for unknown state(s): {', '.join(unknown)}")
        if nacs and not self.capabilities.nacs:
            raise ValueError(
                f"Nonadiabatic coupling vectors are not available for method = "
                f"'{self._settings.method}'; request overlaps instead."
            )
        unknown = sorted(str(s) for s in set(gradients) - set(self._states))
        if unknown:
            raise ValueError(f"Gradients requested for unknown state(s): {', '.join(unknown)}")

    def _state_overlaps(
        self, previous: _Snapshot, mol: Any, mo_coeff: np.ndarray, expansion: DeterminantExpansion
    ) -> np.ndarray:
        from pyscf import gto

        try:
            previous_mol = self._build_mol(previous.coords, None)
            s_ao = gto.intor_cross("int1e_ovlp", previous_mol, mol)
        except Exception as exc:
            raise wrap_failure("The AO overlap between geometries", exc) from exc
        s_mo = previous.mo_coeff.T @ s_ao @ mo_coeff
        return state_overlaps(previous.expansion, expansion, s_mo)

    def _phase(self, live: _Live, label: StateLabel) -> float:
        return float(live.phases[self._states.index(label)])

    def _derivatives(
        self,
        live: _Live,
        gradients: frozenset[StateLabel],
        nacs: frozenset[tuple[StateLabel, StateLabel]],
    ) -> tuple[dict[StateLabel, np.ndarray], dict[tuple[StateLabel, StateLabel], np.ndarray]]:
        if self._settings.method == "casscf":
            grads = pyscf_casscf.casscf_gradients(live.solver, gradients)
            raw = pyscf_casscf.casscf_nacs(live.solver, nacs)
        else:
            grads = pyscf_response.response_gradients(
                live.mf, live.solver, gradients, self._settings.method
            )
            raw = {}
        phased = {
            (a, b): d * self._phase(live, a) * self._phase(live, b) for (a, b), d in raw.items()
        }
        return grads, phased

    def _dipoles(self, live: _Live, wanted: bool) -> dict[StateLabel, np.ndarray]:
        if not wanted:
            return {}
        if self._settings.method == "casscf":
            raw = pyscf_casscf.casscf_dipoles(live.mol, live.solver, self._states)
        else:
            raw = pyscf_response.response_dipoles(live.solver, self._states, self._settings.method)
        p0 = self._phase(live, self._states[0])
        return {label: mu * p0 * self._phase(live, label) for label, mu in raw.items()}

    def _build_mol(self, coords: np.ndarray, stdout: IO[str] | None) -> Any:
        from pyscf import gto

        mol = gto.Mole()
        if stdout is not None:
            mol.stdout = stdout
            mol.verbose = 4
        else:
            mol.verbose = 0
        if self._memory_mb is not None:
            mol.max_memory = self._memory_mb
        atoms = [
            (symbol, (float(x), float(y), float(z)))
            for symbol, (x, y, z) in zip(self._molecule.symbols, coords, strict=True)
        ]
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Basis may be available", category=UserWarning
                )
                mol.build(
                    atom=atoms,
                    unit="Bohr",
                    basis=self._settings.basis,
                    charge=self._molecule.charge,
                    spin=self._molecule.multiplicity - 1,
                    dump_input=False,  # do not copy the running script into the PySCF log
                    parse_arg=False,  # do not let PySCF read our command line
                )
        except RuntimeError as exc:  # pyscf BasisNotFoundError is a RuntimeError
            raise ElectronicStructureError(
                f"PySCF could not build the molecule with basis '{self._settings.basis}': {exc}"
            ) from exc
        return mol

    def _make_scf(self, mol: Any) -> Any:
        from pyscf import dft, scf

        functional = self._settings.functional
        if self._settings.method == "casscf" or functional is None or functional.lower() == "hf":
            mf = scf.RHF(mol)
        else:
            mf = dft.RKS(mol)
            mf.xc = functional
            mf.grids.level = self._settings.grid_level
        mf.conv_tol = self._settings.scf_conv_tol
        mf.max_cycle = self._settings.scf_max_cycle
        mf.chkfile = None  # the program keeps its own state; no SCF chkfile
        if hasattr(mf, "_chkfile"):
            # PySCF 2.14 SCF.__init__ creates lib.NamedTemporaryFile in TMPDIR; dropping it
            # deletes the file
            mf._chkfile = None
        return mf

    def _run_scf(self, mol: Any, previous_dm: Any) -> tuple[Any, str]:
        if previous_dm is not None and getattr(previous_dm, "shape", None) != (mol.nao, mol.nao):
            previous_dm = None
        for rung, (name, shift, damp, second_order) in enumerate(_SCF_LADDER):
            # only the first rung uses the previous geometry's density: a stale guess is the
            # likeliest reason rung 1 fails, so later rungs start from PySCF's default guess
            dm0 = previous_dm if rung == 0 else None
            mf = self._make_scf(mol)
            mf.level_shift = shift
            mf.damp = damp
            if damp:
                # PySCF damps only while cycle < diis_start_cycle - 1 (default 1, i.e. never):
                # damp the first cycles, then let DIIS finish the convergence
                mf.diis_start_cycle = 8
            if second_order:
                mf = mf.newton()
            try:
                mf.kernel(dm0=dm0)
            # An exception (unlike non-convergence) signals a broken setup that another rung
            # would not fix, so it aborts the ladder.
            except Exception as exc:  # PySCF raises many types; report them as SCF failures
                raise wrap_failure(f"PySCF SCF ({name})", exc) from exc
            if mf.converged:
                return mf, name
        tried = ", ".join(name for name, *_ in _SCF_LADDER)
        raise ElectronicStructureError(
            f"SCF did not converge (tried: {tried}; scf_max_cycle = "
            f"{self._settings.scf_max_cycle}, scf_conv_tol = {self._settings.scf_conv_tol})."
        )
