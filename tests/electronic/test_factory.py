from types import SimpleNamespace

import pytest

from gqteashmd.electronic.base import Capabilities, ElectronicBackend, MoleculeSpec
from gqteashmd.electronic.factory import make_backend

pytestmark = pytest.mark.pyscf


def test_make_backend_forwards_the_scratch_directory(ethylene_symbols, make_settings, tmp_path):
    config = SimpleNamespace(electronic=make_settings(basis="sto-3g"))
    spec = MoleculeSpec(tuple(ethylene_symbols))
    scratch = tmp_path / "scratch"
    backend = make_backend(config, spec, scratch_dir=scratch, threads=1, memory_mb=500)
    assert backend._scratch_dir == scratch
    assert make_backend(config, spec)._scratch_dir is None


def test_make_backend_uses_explicit_settings(ethylene_symbols, make_settings):
    config = SimpleNamespace(electronic=make_settings(basis="sto-3g"))
    other = make_settings(basis="6-31g")
    spec = MoleculeSpec(tuple(ethylene_symbols))
    assert make_backend(config, spec, settings=other)._settings is other
    assert make_backend(config, spec)._settings is config.electronic


def test_make_backend_with_explicit_settings_needs_no_electronic_section(
    ethylene_symbols, make_settings
):
    config = SimpleNamespace(electronic=None)
    spec = MoleculeSpec(tuple(ethylene_symbols))
    assert make_backend(config, spec, settings=make_settings(basis="sto-3g")) is not None


def test_default_hessian_is_not_implemented():
    class Minimal(ElectronicBackend):
        states = ()  # type: ignore[assignment]
        capabilities = Capabilities(False, False, False, False)  # type: ignore[assignment]

        def compute(self, request):
            raise NotImplementedError

        def compute_more(self, gradients=frozenset(), nacs=frozenset()):
            raise NotImplementedError

        def reset(self):
            pass

        def save_state(self, path):
            pass

        def load_state(self, path):
            pass

    assert Capabilities(False, False, False, False).hessian is False
    with pytest.raises(NotImplementedError, match="Minimal has no analytic Hessian"):
        Minimal().ground_state_hessian([[0.0, 0.0, 0.0]])
