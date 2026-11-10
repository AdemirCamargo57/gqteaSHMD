import numpy as np
import pytest

from gqteashmd.dynamics.models import (
    DiabaticModel,
    TullyDualAvoidedCrossing,
    TullyExtendedCoupling,
    TullySimpleAvoidedCrossing,
    TwoDimensionalCrossing,
)

MODELS = [
    (
        TullySimpleAvoidedCrossing(),
        [-3.0, -0.7, 0.2, 0.4, 2.5],
    ),  # not 0: V11 has a kink in V" at x = 0, which spoils finite differences
    (TullyDualAvoidedCrossing(), [-8.0, -2.0, 0.5, 4.0, 9.0]),
    (TullyExtendedCoupling(), [-6.0, -1.0, 0.3, 2.0, 7.0]),
    (TwoDimensionalCrossing(), [[0.3, 0.8], [-0.4, -0.6], [1.2, 0.5], [-1.5, 1.0]]),
]
IDS = ["sac", "dac", "ecr", "2d"]


def _qs(model, pts):
    return [np.atleast_1d(np.array(p, float)) for p in pts]


@pytest.mark.parametrize(("model", "pts"), MODELS, ids=IDS)
def test_diabatic_symmetric_and_derivatives(model, pts):
    h = 1e-5
    for q in _qs(model, pts):
        v, dv = model.diabatic(q)
        np.testing.assert_allclose(v, v.T, atol=0)
        assert dv.shape == (model.n_dof, model.n_states, model.n_states)
        for f in range(model.n_dof):
            e = np.zeros(model.n_dof)
            e[f] = h
            fd = (model.diabatic(q + e)[0] - model.diabatic(q - e)[0]) / (2 * h)
            np.testing.assert_allclose(dv[f], fd, atol=1e-8)


@pytest.mark.parametrize(("model", "pts"), MODELS, ids=IDS)
def test_gradients_and_nacs_against_finite_differences(model, pts):
    h = 1e-5
    for q in _qs(model, pts):
        snap = model.evaluate(q, None)
        assert snap.energies.shape == (model.n_states,)
        for f in range(model.n_dof):
            e = np.zeros(model.n_dof)
            e[f] = h
            sp = model.evaluate(q + e, snap)
            sm = model.evaluate(q - e, snap)
            np.testing.assert_allclose(
                snap.gradients[:, f], (sp.energies - sm.energies) / (2 * h), atol=1e-8
            )
            dw = (sp.vectors - sm.vectors) / (2 * h)  # columns: d|k>/dq_f
            d_fd = snap.vectors.T @ dw  # [j, k] = <j| d k>
            np.testing.assert_allclose(snap.nacs[:, :, f], d_fd, atol=1e-6)
        np.testing.assert_allclose(snap.nacs, -np.swapaxes(snap.nacs, 0, 1), atol=1e-14)
        np.testing.assert_allclose(np.diagonal(snap.nacs, axis1=0, axis2=1), 0.0)
        np.testing.assert_allclose(snap.hamiltonian, np.diag(snap.energies))
        assert snap.hamiltonian.dtype == complex


def test_sign_continuity_along_path():
    for model in (
        TullySimpleAvoidedCrossing(),
        TullyDualAvoidedCrossing(),
        TullyExtendedCoupling(),
    ):
        prev = None
        for x in np.arange(-10.0, 10.0 + 1e-9, 0.01):
            snap = model.evaluate(np.array([x]), prev)
            if prev is not None:
                assert np.all(np.einsum("ik,ik->k", prev.vectors, snap.vectors) > 0)
            prev = snap


def test_no_previous_is_deterministic_frame():
    snap = TullySimpleAvoidedCrossing().evaluate(np.array([0.3]), None)
    idx = np.argmax(np.abs(snap.vectors), axis=0)
    assert np.all(snap.vectors[idx, np.arange(2)] > 0)


def test_sac_asymptotic_energies():
    m = TullySimpleAvoidedCrossing()
    for x in (-10.0, 10.0):
        np.testing.assert_allclose(
            m.evaluate(np.array([x]), None).energies, [-0.01, 0.01], atol=1e-6
        )
    assert m.masses.tolist() == [2000.0]


def test_degenerate_states_raise():
    class Degenerate(DiabaticModel):
        n_states = 2
        n_dof = 1
        masses = np.array([1.0])

        def diabatic(self, q):
            return np.zeros((2, 2)), np.zeros((1, 2, 2))

    with pytest.raises(ValueError, match="degenerate"):
        Degenerate().evaluate(np.array([0.0]), None)


@pytest.mark.parametrize(("model", "pts"), MODELS, ids=IDS)
def test_coordinates_of_the_wrong_length_are_rejected(model, pts):
    good = _qs(model, pts)[0]
    for bad in (np.append(good, 0.1), good[:-1], np.zeros((1, model.n_dof))):
        with pytest.raises(ValueError, match="n_dof"):
            model.evaluate(bad, None)


@pytest.mark.parametrize(("model", "pts"), MODELS, ids=IDS)
def test_state_counts_are_singlets_only(model, pts):
    assert model.state_counts == (model.n_states,)


@pytest.mark.parametrize(("model", "pts"), MODELS, ids=IDS)
def test_overlaps_are_none_without_previous_vectors(model, pts):
    from dataclasses import replace

    q = _qs(model, pts)[0]
    snap = model.evaluate(q, None)
    assert snap.overlaps is None
    assert model.evaluate(q, replace(snap, vectors=None)).overlaps is None


def test_sac_overlaps_along_a_path_are_continuous():
    """R10: S = W_prev^T W after the sign alignment, orthogonal with a positive diagonal."""
    model = TullySimpleAvoidedCrossing()
    xs = np.linspace(-4.0, 4.0, 401)
    snap = model.evaluate(np.array([xs[0]]), None)
    min_diag, worst_orth = np.inf, 0.0
    for x in xs[1:]:
        new = model.evaluate(np.array([x]), snap)
        s = new.overlaps
        assert s is not None and s.shape == (2, 2)
        np.testing.assert_array_equal(s, snap.vectors.T @ new.vectors)
        min_diag = min(min_diag, float(np.min(np.diagonal(s))))
        worst_orth = max(worst_orth, float(np.max(np.abs(s.T @ s - np.eye(2)))))
        snap = new
    assert min_diag > 0.0, min_diag
    assert worst_orth <= 1e-13, worst_orth
