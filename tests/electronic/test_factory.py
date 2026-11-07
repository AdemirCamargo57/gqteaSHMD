from types import SimpleNamespace

import pytest

from gqteashmd.electronic.base import MoleculeSpec
from gqteashmd.electronic.factory import make_backend

pytestmark = pytest.mark.pyscf


def test_make_backend_forwards_the_scratch_directory(ethylene_symbols, make_settings, tmp_path):
    config = SimpleNamespace(electronic=make_settings(basis="sto-3g"))
    spec = MoleculeSpec(tuple(ethylene_symbols))
    scratch = tmp_path / "scratch"
    backend = make_backend(config, spec, scratch_dir=scratch, threads=1, memory_mb=500)
    assert backend._scratch_dir == scratch
    assert make_backend(config, spec)._scratch_dir is None
