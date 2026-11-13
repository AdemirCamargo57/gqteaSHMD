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


def test_stage_backend_starts_a_displaced_casscf_solve_from_the_reference(tmp_path, write_input):
    """End to end through , like a finite-difference task of ."""
    import shutil
    from pathlib import Path

    from gqteashmd.config.loader import load_config
    from gqteashmd.core.states import StateLabel
    from gqteashmd.electronic.base import Request
    from gqteashmd.project.layout import ProjectLayout
    from gqteashmd.stages.base import StageContext, TaskSpec
    from gqteashmd.stages.geometry import stage_backend, working_geometry
    from gqteashmd.stages.prepare import PrepareStage

    cas = (
        '[electronic]\nmethod = "casscf"\nbasis = "sto-3g"\nsinglets = 2\n'
        "[electronic.casscf]\nncas = 2\nnelecas = 2\n"
    )
    path = write_input(tmp_path, extra=cas)
    shutil.copy(
        Path(__file__).resolve().parents[1] / "data" / "ethylene.xyz", tmp_path / "water.xyz"
    )
    cfg = load_config(path).config
    layout = ProjectLayout(cfg.project.directory)
    prep_dir = layout.stage_dir(0, "prepare")
    prep_dir.mkdir(parents=True)
    prepare = PrepareStage()
    pctx = StageContext(cfg, layout, prep_dir, {})
    prepare.finalize(pctx, {"molecule": prepare.run_task(pctx, TaskSpec("molecule"), prep_dir)})
    ctx = StageContext(cfg, layout, layout.stage_dir(2, "frequencies"), {"prepare": prep_dir})
    ctx.stage_dir.mkdir(parents=True)
    molecule, coords, _ = working_geometry(ctx)
    task_dir = layout.task_dir(ctx.stage_dir, "fd_0_plus")
    task_dir.mkdir(parents=True)
    backend = stage_backend(ctx, "frequencies", molecule, task_dir, reference_coords_bohr=coords)
    displaced = coords.copy()
    displaced.flat[0] += 0.005
    result = backend.compute(Request(displaced, gradients=frozenset({StateLabel(1, 0)})))
    assert result.meta["casscf"] == "reference orbitals"
    assert isinstance(result.meta["casscf_reference"], dict)
