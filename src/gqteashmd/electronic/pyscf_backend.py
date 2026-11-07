"""PySCF backend: restricted SCF, SA-CASSCF and TDA/TDDFT energies, gradients, dipoles.

PySCF is imported lazily inside functions so that the rest of gqteashmd never needs it.
Coordinates are passed to PySCF in bohr; all results are in atomic units.
"""

from __future__ import annotations

import re
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

import numpy as np

from gqteashmd.chem.electrons import count_electrons
from gqteashmd.chem.elements import atomic_number
from gqteashmd.config.schema import ElectronicSection
from gqteashmd.core.states import StateLabel, states_from_counts
from gqteashmd.electronic import pyscf_casscf
from gqteashmd.electronic.base import (
    Capabilities,
    ElectronicBackend,
    ElectronicResult,
    MoleculeSpec,
    Request,
)
from gqteashmd.electronic.pyscf_errors import wrap_failure
from gqteashmd.errors import ConfigError, ElectronicStructureError

MIN_PYSCF_VERSION = (2, 6, 0)

# (name, level shift / Eh, damping factor, second-order solver)
_SCF_LADDER: tuple[tuple[str, float, float, bool], ...] = (
    ("default", 0.0, 0.0, False),
    ("level shift + damping", 0.2, 0.5, False),
    ("second-order", 0.0, 0.0, True),
)


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
        self._previous_mol: Any = None
        self._previous_mo: Any = None
        self._previous_dm: Any = None

    @property
    def states(self) -> tuple[StateLabel, ...]:
        return self._states

    # ------------------------------------------------------------------ public API
    def compute(self, request: Request) -> ElectronicResult:
        unknown_nac = sorted(
            str(s) for s in {s for pair in request.nacs for s in pair} - set(self._states)
        )
        if unknown_nac:
            raise ValueError(f"NACs requested for unknown state(s): {', '.join(unknown_nac)}")
        if request.nacs and not self.capabilities.nacs:
            raise ValueError(
                f"Nonadiabatic coupling vectors are not available for method = "
                f"'{self._settings.method}'; request overlaps instead."
            )
        if request.overlaps:
            raise ValueError("Overlaps are not implemented yet (P3b Task 7).")
        unknown = sorted(str(s) for s in set(request.gradients) - set(self._states))
        if unknown:
            raise ValueError(f"Gradients requested for unknown state(s): {', '.join(unknown)}")
        if len(request.coords_bohr) != len(self._molecule.symbols):
            raise ValueError(
                f"Got coordinates for {len(request.coords_bohr)} atoms; the molecule has "
                f"{len(self._molecule.symbols)} atoms."
            )
        if self._threads is not None:
            from pyscf import lib

            lib.num_threads(self._threads)
        started = time.perf_counter()
        with self._log() as stdout:
            mol = self._build_mol(request.coords_bohr, stdout)
            mf, scf_rung = self._run_scf(mol)
            nacs: dict[tuple[StateLabel, StateLabel], np.ndarray] = {}
            rung_meta: dict[str, Any] = {}
            if self._settings.method == "casscf":
                previous = None
                if self._previous_mo is not None and self._previous_mol is not None:
                    previous = (self._previous_mo, self._previous_mol)
                outcome = pyscf_casscf.run_casscf(mol, mf, self._settings, self._states, previous)
                mc = outcome.mc
                energies, spins = outcome.energies, outcome.spin_squares
                gradients = pyscf_casscf.casscf_gradients(mc, request.gradients)
                nacs = pyscf_casscf.casscf_nacs(mc, request.nacs)
                dipoles = (
                    pyscf_casscf.casscf_dipoles(mol, mc, self._states)
                    if request.transition_dipoles
                    else {}
                )
                rung_meta = {"casscf": outcome.rung}
                self._previous_mo, self._previous_mol = mc.mo_coeff, mol
            else:
                energies, gradients, dipoles, spins = self._run_linear_response(mol, mf, request)
        meta: dict[str, Any] = {
            "backend": "pyscf",
            "pyscf_version": self._version,
            "method": self._settings.method,
            "basis": self._settings.basis,
            "functional": self._settings.functional,
            "scf": scf_rung,
            "scf_energy_hartree": float(mf.e_tot),
            "wall_time_s": time.perf_counter() - started,
            **rung_meta,
        }
        return ElectronicResult(
            states=self._states,
            energies=energies,
            gradients=gradients,
            nacs=nacs,
            transition_dipoles=dipoles,
            spin_squares=spins,
            meta=meta,
        )

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            gradients=True,
            nacs=self._settings.method == "casscf",
            overlaps=True,
            transition_dipoles=True,
        )

    def compute_more(
        self,
        gradients: frozenset[StateLabel] = frozenset(),
        nacs: frozenset[tuple[StateLabel, StateLabel]] = frozenset(),
    ) -> ElectronicResult:
        raise NotImplementedError("compute_more arrives in P3b Task 7")

    def save_state(self, path: Path) -> None:
        raise NotImplementedError("save_state arrives in P3b Task 8")

    def load_state(self, path: Path) -> None:
        raise NotImplementedError("load_state arrives in P3b Task 8")

    # ------------------------------------------------------------------ setup
    @contextmanager
    def _log(self) -> Iterator[IO[str] | None]:
        if self._log_file is None:
            yield None
        else:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_file, "a", encoding="utf-8") as fh:
                yield fh

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
        return mf

    def _run_scf(self, mol: Any) -> tuple[Any, str]:
        previous_dm = self._previous_dm
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
                self._previous_dm = mf.make_rdm1()
                return mf, name
        tried = ", ".join(name for name, *_ in _SCF_LADDER)
        raise ElectronicStructureError(
            f"SCF did not converge (tried: {tried}; scf_max_cycle = "
            f"{self._settings.scf_max_cycle}, scf_conv_tol = {self._settings.scf_conv_tol})."
        )

    # ------------------------------------------------------------------ TDA / TDDFT (Task 7)
    def _run_linear_response(
        self, mol: Any, mf: Any, request: Request
    ) -> tuple[
        dict[StateLabel, float],
        dict[StateLabel, np.ndarray],
        dict[StateLabel, np.ndarray],
        dict[StateLabel, float],
    ]:
        from pyscf import tddft

        method = self._settings.method.upper()
        excited = self._states[1:]
        energies: dict[StateLabel, float] = {self._states[0]: float(mf.e_tot)}
        gradients: dict[StateLabel, np.ndarray] = {}
        dipoles: dict[StateLabel, np.ndarray] = {}
        td: Any = None
        if excited:
            td = tddft.TDA(mf) if self._settings.method == "tda" else tddft.TDDFT(mf)
            td.nstates = len(excited)
            td.conv_tol = self._settings.td_conv_tol
            try:
                td.kernel()
            except Exception as exc:
                raise wrap_failure(method, exc) from exc
            converged = np.atleast_1d(np.asarray(td.converged, dtype=bool))
            if len(td.e) < len(excited) or not bool(converged.all()):
                raise ElectronicStructureError(
                    f"{method} did not converge for all {len(excited)} "
                    f"excited states (td_conv_tol = {self._settings.td_conv_tol})."
                )
            for label, omega in zip(excited, td.e[: len(excited)], strict=True):
                energies[label] = float(mf.e_tot + omega)
            if request.transition_dipoles:
                try:
                    mus = td.transition_dipole()
                except Exception as exc:
                    raise wrap_failure(f"{method} transition dipole", exc) from exc
                for label, mu in zip(excited, mus, strict=True):
                    dipoles[label] = np.asarray(mu, dtype=float)
        for label in sorted(request.gradients):
            try:
                if label.index == 0:
                    g = mf.nuc_grad_method().kernel()
                else:
                    g = td.nuc_grad_method().kernel(state=label.index)
            except Exception as exc:
                what = "SCF" if label.index == 0 else method
                raise wrap_failure(f"{what} gradient of {label}", exc) from exc
            gradients[label] = np.asarray(g, dtype=float)
        return energies, gradients, dipoles, {}
