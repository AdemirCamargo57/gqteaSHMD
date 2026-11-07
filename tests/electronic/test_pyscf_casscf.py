import json

import numpy as np
import pytest

from gqteashmd.core.states import StateLabel
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.pyscf_backend import PySCFBackend, require_pyscf
from gqteashmd.errors import ConfigError, ElectronicStructureError

S0, S1 = StateLabel(1, 0), StateLabel(1, 1)


def _backend(symbols, settings, **kw):
    return PySCFBackend(MoleculeSpec(tuple(symbols)), settings, **kw)


def test_require_pyscf_reports_a_supported_version():
    major, minor = (int(x) for x in require_pyscf().split(".")[:2])
    assert (major, minor) >= (2, 6)


def test_sa2_casscf_matches_the_sharc_reference(
    ethylene_symbols, ethylene_bohr, make_settings, reference_dir
):
    ref = json.loads((reference_dir / "sharc41_pyscf_ethylene_sa2cas22_631gs.json").read_text())
    backend = _backend(ethylene_symbols, make_settings())
    result = backend.compute(
        Request(ethylene_bohr, gradients=frozenset({S0, S1}), transition_dipoles=True)
    )
    assert result.states == (S0, S1)
    for label, e_ref in zip((S0, S1), ref["energies_hartree"], strict=True):
        assert abs(result.energies[label] - e_ref) < 1e-6
    for name, g_ref in ref["gradients_hartree_per_bohr"].items():
        label = StateLabel.parse(name)
        assert np.abs(result.gradients[label] - np.array(g_ref)).max() < 1e-5
    mu_ref = np.array(ref["transition_dipoles_au"]["S1"])
    mu = result.transition_dipoles[S1]
    # global sign is arbitrary
    assert min(np.abs(mu - mu_ref).max(), np.abs(mu + mu_ref).max()) < 1e-4
    assert all(abs(s) < 1e-6 for s in result.spin_squares.values())


def test_meta_and_log_file(tmp_path, ethylene_symbols, ethylene_bohr, make_settings):
    log = tmp_path / "logs" / "pyscf.log"
    result = _backend(ethylene_symbols, make_settings(basis="6-31g"), log_file=log).compute(
        Request(ethylene_bohr)
    )
    assert log.stat().st_size > 0
    assert result.meta["method"] == "casscf" and result.meta["basis"] == "6-31g"
    assert result.meta["scf"] == "default" and result.meta["pyscf_version"]
    assert result.gradients == {} and result.transition_dipoles == {}


def test_request_validation(ethylene_symbols, ethylene_bohr, make_settings):
    backend = _backend(ethylene_symbols, make_settings(basis="sto-3g"))
    with pytest.raises(ValueError, match="S5"):
        backend.compute(Request(ethylene_bohr, gradients=frozenset({StateLabel(1, 5)})))
    with pytest.raises(ValueError, match="atom"):
        backend.compute(Request(ethylene_bohr[:3]))


def test_unknown_basis_is_an_electronic_structure_error(
    ethylene_symbols, ethylene_bohr, make_settings
):
    with pytest.raises(ElectronicStructureError, match="basis 'no-such-basis'") as info:
        _backend(ethylene_symbols, make_settings(basis="no-such-basis")).compute(
            Request(ethylene_bohr)
        )
    assert info.value.retryable is False and info.value.exit_code == 4


def test_active_space_must_fit(ethylene_symbols, ethylene_bohr, make_settings):
    settings = make_settings(basis="sto-3g", casscf={"ncas": 40, "nelecas": 2})
    with pytest.raises(ElectronicStructureError, match="active space"):
        _backend(ethylene_symbols, settings).compute(Request(ethylene_bohr))


def test_scf_failure_after_the_ladder(ethylene_symbols, ethylene_bohr, make_settings):
    with pytest.raises(ElectronicStructureError, match="SCF did not converge"):
        _backend(ethylene_symbols, make_settings(basis="6-31g", scf_max_cycle=1)).compute(
            Request(ethylene_bohr)
        )


def test_open_shell_molecule_is_refused(ethylene_symbols, make_settings):
    with pytest.raises(ConfigError, match="closed-shell singlet"):
        PySCFBackend(MoleculeSpec(tuple(ethylene_symbols), 0, 3), make_settings())


def test_inconsistent_charge_is_a_config_error(ethylene_symbols, make_settings):
    with pytest.raises(ConfigError, match="inconsistent with 15 electrons"):
        PySCFBackend(MoleculeSpec(tuple(ethylene_symbols), 1, 1), make_settings())


def test_previous_orbitals_are_projected(
    ethylene_symbols, ethylene_bohr, make_settings, monkeypatch
):
    from pyscf import mcscf

    calls = []
    original = mcscf.project_init_guess

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(mcscf, "project_init_guess", spy)
    settings = make_settings(basis="6-31g")
    moved = ethylene_bohr + np.random.default_rng(7).normal(scale=0.05, size=ethylene_bohr.shape)
    reused = _backend(ethylene_symbols, settings)
    reused.compute(Request(ethylene_bohr))
    assert calls == []  # nothing to project on the first call
    second = reused.compute(Request(moved))
    assert calls == [1]  # the second call projects the first call orbitals
    fresh = _backend(ethylene_symbols, settings).compute(Request(moved))
    assert calls == [1]  # a new backend has no previous orbitals
    for label in (S0, S1):
        assert abs(second.energies[label] - fresh.energies[label]) < 1e-7


def test_a_later_scf_rung_rescues_a_failed_first_rung(
    ethylene_symbols, ethylene_bohr, make_settings, monkeypatch
):
    original = PySCFBackend._make_scf
    made = []

    def make_scf(self, mol):
        mf = original(self, mol)
        if not made:
            mf.max_cycle = 1  # the first rung cannot converge in one cycle
        made.append(mf)
        return mf

    monkeypatch.setattr(PySCFBackend, "_make_scf", make_scf)
    result = _backend(ethylene_symbols, make_settings(basis="sto-3g")).compute(
        Request(ethylene_bohr)
    )
    assert result.meta["scf"] == "level shift + damping"
    assert len(made) == 2
    # damping only acts before DIIS starts, so the damping rung must delay DIIS
    assert made[1].diis_start_cycle > 1 and made[1].damp == 0.5


def test_failure_wrapper_marks_only_transient_errors_retryable():
    from gqteashmd.electronic.pyscf_backend import _wrap_failure

    oom = _wrap_failure("CASSCF", MemoryError("no memory"))
    assert isinstance(oom, ElectronicStructureError) and oom.retryable is True
    assert oom.exit_code == 4
    assert _wrap_failure("scratch", OSError("disk full")).retryable is True
    err = _wrap_failure("CASSCF", ValueError("bad"))
    assert err.retryable is False
    assert str(err) == "CASSCF failed: ValueError: bad"


def test_importing_the_cli_and_worker_does_not_import_pyscf():
    import subprocess
    import sys

    code = (
        "import gqteashmd.cli, gqteashmd.worker, sys; "
        "assert not any(m == 'pyscf' or m.startswith('pyscf.') for m in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
