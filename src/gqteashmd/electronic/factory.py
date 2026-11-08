"""Choose and construct the electronic-structure backend for a configuration."""

from __future__ import annotations

from pathlib import Path

from gqteashmd.config.schema import ElectronicSection, ShmdConfig
from gqteashmd.electronic.base import ElectronicBackend, MoleculeSpec
from gqteashmd.errors import ConfigError


def ground_state_settings(settings: ElectronicSection) -> ElectronicSection:
    """S0 only. SCF methods need no excited states for S0. SA-CASSCF keeps its state average,
    which defines the S0 surface."""
    if settings.method == "casscf":
        return settings
    return settings.model_copy(update={"singlets": 1})


def make_backend(
    config: ShmdConfig,
    molecule: MoleculeSpec,
    *,
    settings: ElectronicSection | None = None,
    log_file: Path | None = None,
    threads: int | None = None,
    memory_mb: int | None = None,
    scratch_dir: Path | None = None,
) -> ElectronicBackend:
    chosen = settings if settings is not None else config.electronic
    if chosen is None:
        raise ConfigError(
            "This calculation needs an [electronic] section in the input file "
            "(method, basis, singlets, ...)."
        )
    from gqteashmd.electronic.pyscf_backend import PySCFBackend

    return PySCFBackend(
        molecule,
        chosen,
        log_file=log_file,
        threads=threads,
        memory_mb=memory_mb,
        scratch_dir=scratch_dir,
    )
