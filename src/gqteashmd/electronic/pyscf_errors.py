"""Error mapping shared by the PySCF modules."""

from __future__ import annotations

from gqteashmd.errors import ElectronicStructureError


def wrap_failure(what: str, exc: Exception) -> ElectronicStructureError:
    """Wrap an exception raised inside PySCF; only out-of-memory and I/O errors are retryable."""
    return ElectronicStructureError(
        f"{what} failed: {type(exc).__name__}: {exc}",
        retryable=isinstance(exc, (MemoryError, OSError)),
    )
