import os

import numpy as np
import pytest

from gqteashmd.core.states import StateLabel
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.pyscf_backend import PySCFBackend

pytestmark = pytest.mark.pyscf

S0, S1, S2 = StateLabel(1, 0), StateLabel(1, 1), StateLabel(1, 2)


def _cas(make_settings):
    return make_settings(basis="6-31g")


def _tda(make_settings):
    return make_settings(method="tda", basis="sto-3g", functional="b3lyp", singlets=3, casscf=None)


def _step(coords, seed=11, scale=0.01):
    return coords + np.random.default_rng(seed).normal(scale=scale, size=coords.shape)


@pytest.mark.parametrize("which", ["cas", "tda"])
def test_overlaps_at_equal_geometry_are_the_identity(
    which, ethylene_symbols, ethylene_bohr, make_settings
):
    settings = _cas(make_settings) if which == "cas" else _tda(make_settings)
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings)
    first = backend.compute(Request(ethylene_bohr, overlaps=True))
    assert first.overlaps is None and first.meta["overlaps"] == "no previous geometry"
    second = backend.compute(Request(ethylene_bohr, overlaps=True))
    np.testing.assert_allclose(second.overlaps, np.eye(len(backend.states)), atol=1e-6)
    assert second.meta["overlaps"] == "computed" and second.meta["root_flips"] == []
    assert backend.compute(Request(ethylene_bohr)).meta["overlaps"] == "tracked, not returned"


@pytest.mark.parametrize("which", ["cas", "tda"])
def test_small_step_overlaps_are_near_identity_with_positive_diagonal(
    which, ethylene_symbols, ethylene_bohr, make_settings
):
    settings = _cas(make_settings) if which == "cas" else _tda(make_settings)
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings)
    backend.compute(Request(ethylene_bohr, overlaps=True))
    result = backend.compute(Request(_step(ethylene_bohr), overlaps=True))
    assert np.all(np.diag(result.overlaps) > 0.9)
    assert set(np.unique(result.phases)) <= {-1.0, 1.0}
    assert result.meta["overlap_min_singular_value"] > 0.9
    off = result.overlaps - np.diag(np.diag(result.overlaps))
    assert np.abs(off + off.T).max() < 0.02  # off-diagonal ~antisymmetric for small steps


def test_phases_keep_couplings_and_dipoles_continuous(
    ethylene_symbols, ethylene_bohr, make_settings
):
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), _cas(make_settings))
    req = dict(nacs=frozenset({(S0, S1)}), transition_dipoles=True, overlaps=True)
    a = backend.compute(Request(ethylene_bohr, **req))
    b = backend.compute(Request(_step(ethylene_bohr, scale=0.005), **req))
    da, db = a.nacs[(S0, S1)], b.nacs[(S0, S1)]
    assert np.abs(da - db).max() < 0.2 * np.abs(da + db).max()  # no sign jump between steps
    ma, mb = a.transition_dipoles[S1], b.transition_dipoles[S1]
    assert np.dot(ma, mb) > 0


def test_compute_more_reuses_the_last_geometry(
    ethylene_symbols, ethylene_bohr, make_settings, monkeypatch
):
    settings = _cas(make_settings)
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings)
    with pytest.raises(ValueError, match="compute_more"):
        backend.compute_more(gradients=frozenset({S1}))
    backend.compute(Request(ethylene_bohr, gradients=frozenset({S0})))
    scf_calls = []
    original = PySCFBackend._run_scf
    monkeypatch.setattr(
        PySCFBackend,
        "_run_scf",
        lambda self, *a, **k: scf_calls.append(1) or original(self, *a, **k),
    )
    more = backend.compute_more(gradients=frozenset({S1}), nacs=frozenset({(S0, S1)}))
    assert scf_calls == [] and more.meta["compute_more"] is True
    fresh = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), settings).compute(
        Request(ethylene_bohr, gradients=frozenset({S1}), nacs=frozenset({(S0, S1)}))
    )
    assert np.abs(more.gradients[S1] - fresh.gradients[S1]).max() < 1e-7
    d_more, d_fresh = more.nacs[(S0, S1)], fresh.nacs[(S0, S1)]
    assert min(np.abs(d_more - d_fresh).max(), np.abs(d_more + d_fresh).max()) < 1e-7
    for label in fresh.energies:
        assert abs(more.energies[label] - fresh.energies[label]) < 1e-8
    assert more.energies == backend.compute_more().energies


def test_compute_more_writes_to_the_log(tmp_path, ethylene_symbols, ethylene_bohr, make_settings):
    log = tmp_path / "pyscf.log"
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), _tda(make_settings), log_file=log)
    backend.compute(Request(ethylene_bohr))
    size = log.stat().st_size
    backend.compute_more(gradients=frozenset({S1}))
    assert log.stat().st_size > size


def test_scratch_policy(tmp_path, ethylene_symbols, ethylene_bohr, make_settings, monkeypatch):
    from pyscf import lib

    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    before = lib.param.TMPDIR
    scratch = tmp_path / "scratch"
    backend = PySCFBackend(
        MoleculeSpec(tuple(ethylene_symbols)), _cas(make_settings), scratch_dir=scratch
    )
    backend.compute(Request(ethylene_bohr, gradients=frozenset({S1})))
    assert os.listdir(work) == []
    assert scratch.is_dir() and before == lib.param.TMPDIR
    assert not any(p.name.startswith("tmp") for p in scratch.iterdir())  # no SCF chkfile


def test_a_forced_sign_flip_is_undone_in_nacs_dipoles_and_diagonal(
    ethylene_symbols, ethylene_bohr, make_settings, monkeypatch
):
    # PySCF's raw CI signs differ from process to process, so the raw vectors of the second
    # geometry are pinned: root 0 and (for the reference) root 1 keep the reference sign, and the
    # flipped run gets the opposite sign for root 1. Both runs start from the same snapshot.
    from gqteashmd.electronic import pyscf_casscf

    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), _cas(make_settings))
    backend.compute(Request(ethylene_bohr, overlaps=True))
    snapshot = backend._snapshot
    req = Request(
        _step(ethylene_bohr, scale=0.005),
        nacs=frozenset({(S0, S1)}),
        transition_dipoles=True,
        overlaps=True,
    )
    original = pyscf_casscf.run_casscf
    pinned: dict[str, object] = {}

    def pinning(*args, **kwargs):
        outcome = original(*args, **kwargs)
        ci = outcome.mc.ci
        if "ref" not in pinned:
            pinned["ref"] = [np.array(v, copy=True) for v in ci]
        for k, want in ((0, 1.0), (1, pinned["flip"])):
            if np.vdot(ci[k], pinned["ref"][k]) * want < 0:
                ci[k] = -ci[k]
        return outcome

    monkeypatch.setattr(pyscf_casscf, "run_casscf", pinning)
    pinned["flip"] = 1.0
    ref = backend.compute(req)
    backend._snapshot = snapshot
    pinned["flip"] = -1.0
    got = backend.compute(req)
    assert got.phases[0] == ref.phases[0] and got.phases[1] == -ref.phases[1]
    assert np.all(np.diag(got.overlaps) > 0)
    np.testing.assert_allclose(got.nacs[(S0, S1)], ref.nacs[(S0, S1)], atol=1e-6)
    np.testing.assert_allclose(got.transition_dipoles[S1], ref.transition_dipoles[S1], atol=1e-6)


def test_overlap_orientation_follows_the_nac(ethylene_symbols, ethylene_bohr, make_settings):
    # <0(R)|1(R+dR)> = d_01 . dR to first order with d_01 = <0|d1/dR>; a transposed s_mo
    # (or swapped rows/columns) would give the opposite sign here
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), _cas(make_settings))
    first = backend.compute(Request(ethylene_bohr, nacs=frozenset({(S0, S1)}), overlaps=True))
    d = first.nacs[(S0, S1)]
    eps = 1e-3
    step = eps * d / np.linalg.norm(d)
    second = backend.compute(Request(ethylene_bohr + step, overlaps=True))
    expected = eps * np.linalg.norm(d)
    assert second.overlaps[0, 1] == pytest.approx(expected, rel=0.1)
    assert second.overlaps[1, 0] * expected < 0


def test_compute_more_writes_to_the_log_for_casscf(
    tmp_path, ethylene_symbols, ethylene_bohr, make_settings
):
    log = tmp_path / "pyscf.log"
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), _cas(make_settings), log_file=log)
    backend.compute(Request(ethylene_bohr))
    size = log.stat().st_size
    backend.compute_more(gradients=frozenset({S1}), nacs=frozenset({(S0, S1)}))
    assert log.stat().st_size > size


def test_a_failed_compute_leaves_no_live_objects_for_compute_more(
    ethylene_symbols, ethylene_bohr, make_settings, monkeypatch
):
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), _tda(make_settings))
    backend.compute(Request(ethylene_bohr))
    assert backend._live is not None

    def boom(self, *args, **kwargs):
        raise RuntimeError("gradient path failed")

    monkeypatch.setattr(PySCFBackend, "_derivatives", boom)
    with pytest.raises(RuntimeError, match="gradient path failed"):
        backend.compute(Request(_step(ethylene_bohr), gradients=frozenset({S1})))
    with pytest.raises(ValueError, match="needs a preceding"):
        backend.compute_more(gradients=frozenset({S1}))


def test_reset_makes_the_next_compute_a_first_geometry(
    ethylene_symbols, ethylene_bohr, make_settings, monkeypatch
):
    from pyscf import mcscf

    calls = []
    original = mcscf.project_init_guess
    monkeypatch.setattr(
        mcscf, "project_init_guess", lambda *a, **k: calls.append(1) or original(*a, **k)
    )
    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), make_settings(basis="sto-3g"))
    backend.compute(Request(ethylene_bohr))
    backend.compute(Request(_step(ethylene_bohr), overlaps=True))
    assert calls == [1]
    backend.reset()
    assert backend._snapshot is None and backend._live is None
    result = backend.compute(Request(_step(ethylene_bohr, seed=3), overlaps=True))
    assert calls == [1]  # no projection from the forgotten geometry
    assert result.meta["overlaps"] == "no previous geometry" and result.overlaps is None


def test_overlap_failures_are_electronic_structure_errors(
    ethylene_symbols, ethylene_bohr, make_settings, monkeypatch
):
    from gqteashmd.electronic import pyscf_backend
    from gqteashmd.errors import ElectronicStructureError

    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), _tda(make_settings))
    backend.compute(Request(ethylene_bohr))
    request = Request(_step(ethylene_bohr), overlaps=True)

    def failing(exc):
        def raiser(*args, **kwargs):
            raise exc

        return raiser

    for exc in (ValueError("bad shapes"), np.linalg.LinAlgError("singular")):
        monkeypatch.setattr(pyscf_backend, "state_overlaps", failing(exc))
        with pytest.raises(ElectronicStructureError, match="wavefunction overlap failed") as info:
            backend.compute(request)
        assert info.value.retryable is False
    monkeypatch.setattr(pyscf_backend, "state_overlaps", failing(MemoryError("estimated 9 GiB")))
    with pytest.raises(ElectronicStructureError, match="estimated 9 GiB") as info:
        backend.compute(request)
    assert info.value.retryable is True
    monkeypatch.undo()
    monkeypatch.setattr(pyscf_backend, "truncate", failing(ValueError("zero coefficients")))
    with pytest.raises(ElectronicStructureError, match="wavefunction overlap failed"):
        backend.compute(request)


def test_phase_tracking_continues_across_a_step_without_returned_overlaps(
    ethylene_symbols, ethylene_bohr, make_settings, monkeypatch
):
    # R0 (NAC) -> R1 (overlaps=False, nothing returned) -> R2 (overlaps=True, NAC). The raw CI sign
    # of state 1 at R1 is pinned once as it comes and once flipped. If the R1 step did not apply
    # its phases to the stored expansion, R2 would be aligned to the raw R1 frame and its NAC
    # would change sign with the pinned R1 sign. It must instead stay in the frame of R0.
    from gqteashmd.electronic import pyscf_casscf

    backend = PySCFBackend(MoleculeSpec(tuple(ethylene_symbols)), _cas(make_settings))
    pair = frozenset({(S0, S1)})
    d0 = backend.compute(Request(ethylene_bohr, nacs=pair, overlaps=True)).nacs[(S0, S1)]
    snapshot = backend._snapshot
    r1 = _step(ethylene_bohr, seed=21, scale=0.005)
    r2 = _step(r1, seed=22, scale=0.005)

    original = pyscf_casscf.run_casscf
    refs: dict[int, list[np.ndarray]] = {}
    state = {"call": 0, "flip_r1": 1.0}

    def pinning(*args, **kwargs):
        outcome = original(*args, **kwargs)
        ci = outcome.mc.ci
        call = state["call"]
        state["call"] += 1
        if call not in refs:
            refs[call] = [np.array(v, copy=True) for v in ci]
        want = (1.0, state["flip_r1"] if call == 0 else 1.0)
        for k in (0, 1):
            if np.vdot(ci[k], refs[call][k]) * want[k] < 0:
                ci[k] = -ci[k]
        return outcome

    monkeypatch.setattr(pyscf_casscf, "run_casscf", pinning)
    nacs_at_r2 = []
    for flip in (1.0, -1.0):
        backend._snapshot = snapshot
        state.update(call=0, flip_r1=flip)
        silent = backend.compute(Request(r1, overlaps=False))
        assert silent.overlaps is None and silent.meta["overlaps"] == "tracked, not returned"
        nacs_at_r2.append(backend.compute(Request(r2, nacs=pair, overlaps=True)).nacs[(S0, S1)])
    reference, flipped = nacs_at_r2
    np.testing.assert_allclose(flipped, reference, atol=1e-6)
    assert np.vdot(reference, d0) > 0  # still in the frame of R0
