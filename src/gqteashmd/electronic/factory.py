"""Choose and construct the electronic-structure backend for a configuration."""

from __future__ import annotations

from pathlib import Path

from gqteashmd.config.schema import ShmdConfig
from gqteashmd.electronic.base import ElectronicBackend, MoleculeSpec
from gqteashmd.errors import ConfigError


def make_backend(
    config: ShmdConfig,
    molecule: MoleculeSpec,
    *,
    log_file: Path | None = None,
    threads: int | None = None,
    memory_mb: int | None = None,
    scratch_dir: Path | None = None,
) -> ElectronicBackend:
    if config.electronic is None:
        raise ConfigError(
            "This calculation needs an [electronic] section in the input file "
            "(method, basis, singlets, ...)."
        )
    from gqteashmd.electronic.pyscf_backend import PySCFBackend

    return PySCFBackend(
        molecule,
        config.electronic,
        log_file=log_file,
        threads=threads,
        memory_mb=memory_mb,
        scratch_dir=scratch_dir,
    )
