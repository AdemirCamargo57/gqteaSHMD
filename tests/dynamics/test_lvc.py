"""LVC models (P7a-2 Method reference items 1-3, 15; rulings R6, R10)."""

from __future__ import annotations

import importlib.util
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gqteashmd.core.units import DALTON_IN_ME
from gqteashmd.dynamics.lvc import LVCModel, lvc_wigner_samples
from gqteashmd.dynamics.lvc_format import LVCParameters, V0Data, read_lvc_template
from gqteashmd.dynamics.models import DiabaticModel
from gqteashmd.dynamics.representation import (
    diagonal_basis,
    diagonal_gradient,
    diagonal_tensor,
    gradient_tensor,
)
from gqteashmd.dynamics.spin import ExpandedState

DATA = Path(__file__).resolve().parents[1] / "data" / "lvc"
TEMPLATES = ["lvc_singlets.template", "lvc_soc.template", "lvc_twomode.template"]
SOC_TEMPLATES = ["lvc_soc.template", "lvc_twomode.template"]
H_FD = 1e-5


def _load_fixtures():
    name = "gqteashmd_lvc_make_fixtures"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, DATA / "make_fixtures.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fx = _load_fixtures()


def _model(name: str, **kwargs) -> LVCModel:
    return LVCModel.from_template(DATA / name, **kwargs)


def _random_qs(model: LVCModel, seed: int = 11, n: int = 5) -> list[np.ndarray]:
    """The reference and n geometries with Q ~ N(0, 1) on the vibrational modes."""
    rng = np.random.default_rng(seed)
    out = [model.reference_q.copy()]
    for _ in range(n):
        out.append(model.cartesian(rng.standard_normal(model.n_dof)))
    return out


def _omega(model: LVCModel) -> np.ndarray:
    return model.parameters.v0.frequencies_Eh


# ------------------------------------------------------------------ construction


@pytest.mark.parametrize("name", TEMPLATES)
def test_basic_attributes(name):
    params = read_lvc_template(DATA / name)
    model = _model(name)
    n_atoms = len(params.v0.symbols)
    assert model.n_dof == 3 * n_atoms
    assert model.state_counts == params.counts
    n1 = params.counts[0]
    n3 = params.counts[2] if len(params.counts) == 3 else 0
    assert model.n_states == n1 + 3 * n3 == len(model.states)
    assert all(isinstance(s, ExpandedState) for s in model.states)
    np.testing.assert_array_equal(model.masses, np.repeat(params.v0.masses_amu * DALTON_IN_ME, 3))
    np.testing.assert_array_equal(model.reference_q, params.v0.coords_bohr.reshape(-1))


@pytest.mark.parametrize("name", TEMPLATES)
def test_picklable(name):
    model = _model(name)
    clone = pickle.loads(pickle.dumps(model))
    q = _random_qs(model, n=1)[1]
    a, b = model.evaluate(q, None), clone.evaluate(q, None)
    for field in ("energies", "gradients", "nacs", "hamiltonian", "vectors"):
        assert np.array_equal(getattr(a, field), getattr(b, field)), field


def test_soc_must_be_hermitian():
    params = fx.soc_params()
    soc = params.soc.copy()
    soc[0, 2] += 1e-6
    with pytest.raises(ValueError, match="SOC.*Hermitian"):
        LVCModel(replace(params, soc=soc))
    # below the 1e-8 Eh tolerance: accepted and symmetrised
    soc = params.soc.copy()
    soc[0, 2] += 5e-9
    model = LVCModel(replace(params, soc=soc))
    sym = model.diabatic_hamiltonian(np.zeros(model.n_dof))
    assert np.array_equal(sym, sym.conj().T)


def test_soc_shape_is_checked():
    params = fx.soc_params()
    with pytest.raises(ValueError, match="SOC"):
        LVCModel(replace(params, soc=np.zeros((5, 5), dtype=complex)))


def _with_frequency(params: LVCParameters, mode: int, omega: float) -> LVCParameters:
    freqs = np.array(params.v0.frequencies_Eh, dtype=float)
    freqs[mode] = omega
    return replace(params, v0=replace(params.v0, frequencies_Eh=freqs))


@pytest.mark.parametrize("omega", [1e-8, 5e-7, -1e-9])
def test_near_zero_frequency_is_refused(omega):
    """A frequency 0 < |omega| < ZERO_MODE_THRESHOLD_EH is an unzeroed external mode: refused,
    naming the 1-based mode, its value and the constant (a negative one is refused anyway)."""
    from gqteashmd.dynamics.lvc import ZERO_MODE_THRESHOLD_EH

    params = _with_frequency(fx.soc_params(), 2, omega)  # mode 3 of V0.txt
    expected = "mode 3" if omega > 0 else ">= 0"
    with pytest.raises(ValueError, match=expected) as info:
        LVCModel(params)
    if omega > 0:
        text = str(info.value)
        assert f"{omega:.3e}" in text and f"{ZERO_MODE_THRESHOLD_EH:.0e}" in text
        assert "exactly 0" in text and "V0.txt" in text


def test_exact_zero_and_threshold_frequencies_are_accepted():
    from gqteashmd.dynamics.lvc import ZERO_MODE_THRESHOLD_EH

    params = fx.soc_params()
    assert int(np.sum(np.asarray(params.v0.frequencies_Eh) == 0.0)) == 6  # unchanged fixture
    LVCModel(params)
    LVCModel(_with_frequency(params, 2, ZERO_MODE_THRESHOLD_EH))  # the bound itself is a mode


def test_q_shape_is_checked():
    model = _model("lvc_singlets.template")
    with pytest.raises(ValueError, match="n_dof"):
        model.evaluate(np.zeros(4), None)


def test_degenerate_states_within_a_block_raise():
    params = fx.singlets_params()
    params = replace(params, epsilon={(1, 0): 0.0, (1, 1): 0.0, (1, 2): 0.19}, kappa={}, lam={})
    model = LVCModel(params)
    with pytest.raises(ValueError, match="degenerate states"):
        model.evaluate(model.reference_q, None)


def test_degeneracy_between_blocks_is_allowed():
    params = fx.twomode_params()
    params = replace(params, epsilon={(1, 0): 0.0, (1, 1): 0.15, (3, 0): 0.15}, kappa={})
    model = LVCModel(params, spin_orbit=False)
    snap = model.evaluate(model.reference_q, None)
    assert snap.energies[1] == snap.energies[2]


# ------------------------------------------------------------------ normal coordinates and V0


@pytest.mark.parametrize("name", TEMPLATES)
def test_normal_coordinates_round_trip(name):
    model = _model(name)
    active = _omega(model) > 0
    q_ref = model.normal_coordinates(model.reference_q)
    assert q_ref.shape == (model.n_dof,)
    assert np.all(q_ref == 0.0)
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(5):
        big_q = rng.standard_normal(model.n_dof)
        back = model.normal_coordinates(model.cartesian(big_q))
        assert np.all(back[~active] == 0.0)
        worst = max(worst, float(np.max(np.abs(back[active] - big_q[active]))))
    assert worst <= 1e-12, worst


@pytest.mark.parametrize("name", TEMPLATES)
def test_v0_is_the_harmonic_energy_of_the_force_field(name):
    params = read_lvc_template(DATA / name)
    bare = LVCModel(replace(params, epsilon={}, kappa={}, lam={}, soc=None))
    hessian = fx.force_field_hessian()
    worst_e = worst_g = 0.0
    for q in _random_qs(bare):
        dr = q - bare.reference_q
        v0_ff = 0.5 * dr @ hessian @ dr
        # every diagonal element of the bare diabatic matrix is V0
        h_diab = bare.diabatic_hamiltonian(bare.normal_coordinates(q))
        worst_e = max(worst_e, float(np.max(np.abs(np.diagonal(h_diab).real - v0_ff))))
    assert worst_e <= 1e-12, worst_e
    # the energy and gradient of a single-state bare model are V0 and H dr
    one = LVCModel(replace(params, counts=(1,), epsilon={}, kappa={}, lam={}, soc=None))
    for q in _random_qs(one):
        dr = q - one.reference_q
        snap = one.evaluate(q, None)
        worst_e = max(worst_e, abs(float(snap.energies[0]) - 0.5 * dr @ hessian @ dr))
        worst_g = max(worst_g, float(np.max(np.abs(snap.gradients[0] - hessian @ dr))))
    assert worst_e <= 1e-12, worst_e
    assert worst_g <= 1e-12, worst_g


# ------------------------------------------------------------------ MCH Hamiltonian


@pytest.mark.parametrize("name", TEMPLATES)
def test_mch_hamiltonian_hermitian_and_same_spectrum(name):
    model = _model(name)
    worst_h = worst_s = 0.0
    for q in _random_qs(model):
        snap = model.evaluate(q, None)
        h = snap.hamiltonian
        assert h.dtype == complex and h.shape == (model.n_states, model.n_states)
        worst_h = max(worst_h, float(np.max(np.abs(h - h.conj().T))))
        diab = model.diabatic_hamiltonian(model.normal_coordinates(q))
        assert diab.dtype == complex
        spec = np.linalg.eigvalsh(h)
        worst_s = max(worst_s, float(np.max(np.abs(spec - np.linalg.eigvalsh(diab)))))
        np.testing.assert_array_equal(np.diagonal(h).real, snap.energies)
    assert worst_h <= 1e-14, worst_h
    assert worst_s <= 1e-12, worst_s


@pytest.mark.parametrize("name", TEMPLATES)
def test_spin_orbit_false_gives_diagonal_hamiltonian(name):
    model = _model(name, spin_orbit=False)
    for q in _random_qs(model, n=2):
        h = model.evaluate(q, None).hamiltonian
        assert np.count_nonzero(h - np.diag(np.diagonal(h))) == 0
        assert np.count_nonzero(h.imag) == 0


def test_spin_orbit_true_gives_off_diagonal_hamiltonian():
    model = _model("lvc_soc.template")
    h = model.evaluate(model.reference_q, None).hamiltonian
    assert np.count_nonzero(h - np.diag(np.diagonal(h))) > 0


# ------------------------------------------------------------------ gradients and NACs


@pytest.mark.parametrize("name", TEMPLATES)
def test_gradients_against_finite_differences(name):
    model = _model(name)
    worst = 0.0
    for q in _random_qs(model):
        snap = model.evaluate(q, None)
        for f in range(model.n_dof):
            e = np.zeros(model.n_dof)
            e[f] = H_FD
            ep = model.evaluate(q + e, snap).energies
            em = model.evaluate(q - e, snap).energies
            fd = (ep - em) / (2 * H_FD)
            worst = max(worst, float(np.max(np.abs(snap.gradients[:, f] - fd))))
    assert worst <= 1e-8, worst


@pytest.mark.parametrize("name", TEMPLATES)
def test_nacs_against_finite_differences(name):
    model = _model(name)
    worst = 0.0
    for q in _random_qs(model):
        snap = model.evaluate(q, None)
        for f in range(model.n_dof):
            e = np.zeros(model.n_dof)
            e[f] = H_FD
            tp = model.evaluate(q + e, snap).vectors
            tm = model.evaluate(q - e, snap).vectors
            d_fd = snap.vectors.T @ ((tp - tm) / (2 * H_FD))  # [a, b] = <a| d b>
            worst = max(worst, float(np.max(np.abs(snap.nacs[:, :, f] - d_fd))))
        np.testing.assert_array_equal(snap.nacs, -np.swapaxes(snap.nacs, 0, 1))
    assert worst <= 1e-6, worst


def _block_of(state: ExpandedState) -> tuple[int, int]:
    return (state.multiplicity, state.ms)


@pytest.mark.parametrize("name", TEMPLATES)
def test_nacs_vanish_between_blocks(name):
    model = _model(name)
    for q in _random_qs(model, n=2):
        snap = model.evaluate(q, None)
        for a, sa in enumerate(model.states):
            for b, sb in enumerate(model.states):
                if _block_of(sa) != _block_of(sb):
                    assert np.all(snap.nacs[a, b] == 0.0)
                    assert np.all(snap.vectors[a, b] == 0.0)


@pytest.mark.parametrize("name", SOC_TEMPLATES)
def test_triplet_components_are_identical(name):
    model = _model(name)
    for q in _random_qs(model, n=2):
        snap = model.evaluate(q, None)
        index = {(s.multiplicity, s.index, s.ms): k for k, s in enumerate(model.states)}
        n3 = model.state_counts[2]
        for i in range(n3):
            ref = index[(3, i, -1)]
            for ms in (0, 1):
                k = index[(3, i, ms)]
                assert snap.energies[k] == snap.energies[ref]
                assert np.array_equal(snap.gradients[k], snap.gradients[ref])
                for j in range(n3):
                    assert np.array_equal(
                        snap.nacs[k, index[(3, j, ms)]], snap.nacs[ref, index[(3, j, -1)]]
                    )


# ------------------------------------------------------------------ overlaps


@pytest.mark.parametrize("name", TEMPLATES)
def test_overlaps_along_a_path(name):
    model = _model(name)
    rng = np.random.default_rng(5)
    start, stop = rng.standard_normal(model.n_dof), rng.standard_normal(model.n_dof)
    snap = model.evaluate(model.cartesian(start), None)
    assert snap.overlaps is None
    worst_orth, min_diag = 0.0, np.inf
    for k in range(1, 51):
        big_q = start + (stop - start) * k / 50
        new = model.evaluate(model.cartesian(big_q), snap)
        s = new.overlaps
        assert s is not None and s.shape == (model.n_states, model.n_states)
        np.testing.assert_array_equal(s, snap.vectors.T @ new.vectors)
        worst_orth = max(worst_orth, float(np.max(np.abs(s.T @ s - np.eye(model.n_states)))))
        min_diag = min(min_diag, float(np.min(np.diagonal(s))))
        snap = new
    assert worst_orth <= 1e-13, worst_orth
    assert min_diag > 0.9, min_diag


# ------------------------------------------------------------------ SOC: diagonal gradients


@pytest.mark.parametrize("name", SOC_TEMPLATES)
def test_diagonal_gradient_is_the_derivative_of_the_diagonal_energy(name):
    model = _model(name)
    worst = 0.0
    for q in _random_qs(model, seed=17, n=3)[1:]:
        snap = model.evaluate(q, None)
        e_diag, u = diagonal_basis(snap.hamiltonian, None, first_step=True)
        assert u is not None
        g_diag = diagonal_tensor(gradient_tensor(snap), u)
        for f in range(model.n_dof):
            e = np.zeros(model.n_dof)
            e[f] = H_FD
            ep = np.linalg.eigvalsh(model.evaluate(q + e, snap).hamiltonian)
            em = np.linalg.eigvalsh(model.evaluate(q - e, snap).hamiltonian)
            fd = (ep - em) / (2 * H_FD)
            for a in range(model.n_states):
                worst = max(worst, abs(float(diagonal_gradient(g_diag, a)[f]) - fd[a]))
    assert worst <= 1e-7, worst


# ------------------------------------------------------------------ one mode = DiabaticModel

OMEGA, MASS_AMU = 0.012, 1.5
EPS1, KAPPA0, KAPPA1, LAMBDA = 0.02, -0.004, 0.006, 0.003


def _one_mode_params() -> LVCParameters:
    v0 = V0Data(
        symbols=("X",),
        nuclear_charges=np.array([1.0]),
        coords_bohr=np.array([[0.3, -0.2, 0.1]]),
        masses_amu=np.array([MASS_AMU]),
        frequencies_Eh=np.array([OMEGA, 0.0, 0.0]),
        modes=np.eye(3),
    )
    return LVCParameters(
        v0=v0,
        counts=(2,),
        epsilon={(1, 0): 0.0, (1, 1): EPS1},
        kappa={(1, 0, 0): KAPPA0, (1, 1, 0): KAPPA1},
        lam={(1, 0, 1, 0): LAMBDA},
        soc=None,
        dipoles=None,
    )


class _HandBuilt(DiabaticModel):
    """V(q) of _one_mode_params written out by hand: Q = sqrt(omega M) (x - x_ref)."""

    n_states = 2
    n_dof = 3

    def __init__(self) -> None:
        self.mass = MASS_AMU * DALTON_IN_ME
        self.masses = np.full(3, self.mass)
        self.scale = np.sqrt(OMEGA * self.mass)

    def diabatic(self, q):
        big_q = self.scale * (q[0] - 0.3)
        v0 = 0.5 * OMEGA * big_q**2
        v = np.array(
            [[v0 + KAPPA0 * big_q, LAMBDA * big_q], [LAMBDA * big_q, v0 + EPS1 + KAPPA1 * big_q]]
        )
        dq = np.array([[OMEGA * big_q + KAPPA0, LAMBDA], [LAMBDA, OMEGA * big_q + KAPPA1]])
        dv = np.zeros((3, 2, 2))
        dv[0] = self.scale * dq
        return v, dv


def test_one_mode_lvc_reproduces_a_hand_built_diabatic_model():
    lvc, hand = LVCModel(_one_mode_params()), _HandBuilt()
    assert lvc.state_counts == (2,) and hand.state_counts == (2,)
    np.testing.assert_allclose(lvc.masses, hand.masses, rtol=0, atol=1e-13)
    xs = [0.3, 0.27, 0.31, 0.33, 0.25, 0.35]
    prev_l = prev_h = None
    for x in xs:
        q = np.array([x, -0.2 + 0.05 * x, 0.1 - x])
        a, b = lvc.evaluate(q, prev_l), hand.evaluate(q, prev_h)
        for field in ("energies", "gradients", "nacs", "hamiltonian", "vectors"):
            np.testing.assert_allclose(
                getattr(a, field), getattr(b, field), rtol=0, atol=1e-13, err_msg=field
            )
        if prev_l is None:
            assert a.overlaps is None and b.overlaps is None
        else:
            np.testing.assert_allclose(a.overlaps, b.overlaps, rtol=0, atol=1e-13)
        prev_l, prev_h = a, b


# ------------------------------------------------------------------ Wigner samples


@pytest.mark.parametrize("name", ["lvc_soc.template", "lvc_twomode.template"])
def test_wigner_samples_statistics(name):
    model = _model(name)
    n = 20000
    q, v = lvc_wigner_samples(model, n, np.random.default_rng(2024))
    assert q.shape == (n, model.n_dof) and v.shape == (n, model.n_dof)
    omega = _omega(model)
    active = omega > 0
    k = model.parameters.v0.modes
    sqrt_m = np.sqrt(model.masses)
    disp = (q - model.reference_q) * sqrt_m  # mass-weighted displacement
    mom = v * sqrt_m
    proj_q = disp @ k  # (n, 3N): K^T sqrt(M) dr
    proj_p = mom @ k
    assert float(np.max(np.abs(proj_q[:, ~active]))) <= 1e-12
    assert float(np.max(np.abs(proj_p[:, ~active]))) <= 1e-12
    big_q = proj_q[:, active] * np.sqrt(omega[active])
    big_p = proj_p[:, active] / np.sqrt(omega[active])
    np.testing.assert_allclose(model.normal_coordinates(q[0])[active], big_q[0], atol=1e-12)
    se_mean = np.sqrt(0.5 / n)
    se_var = 0.5 * np.sqrt(2.0 / n)
    for x in (big_q, big_p):
        assert np.all(np.abs(x.mean(axis=0)) < 4 * se_mean), x.mean(axis=0)
        assert np.all(np.abs(x.var(axis=0) - 0.5) < 4 * se_var), x.var(axis=0)


def test_wigner_samples_reproducible_and_draw_order():
    model = _model("lvc_twomode.template")
    q1, v1 = lvc_wigner_samples(model, 7, np.random.default_rng(99))
    q2, v2 = lvc_wigner_samples(model, 7, np.random.default_rng(99))
    assert np.array_equal(q1, q2) and np.array_equal(v1, v2)
    # the documented order: per sample, the vibrational modes in increasing index, Q then P,
    # each one standard normal times sqrt(1/2)
    rng = np.random.default_rng(99)
    omega = _omega(model)
    active = np.flatnonzero(omega > 0)
    k = model.parameters.v0.modes
    sqrt_m = np.sqrt(model.masses)
    for s in range(7):
        r = model.reference_q.copy()
        vel = np.zeros(model.n_dof)
        for i in active:
            big_q = rng.standard_normal() * np.sqrt(0.5)
            big_p = rng.standard_normal() * np.sqrt(0.5)
            r = r + k[:, i] * big_q / (np.sqrt(omega[i]) * sqrt_m)
            vel = vel + k[:, i] * np.sqrt(omega[i]) * big_p / sqrt_m
        np.testing.assert_allclose(q1[s], r, rtol=0, atol=1e-14)
        np.testing.assert_allclose(v1[s], vel, rtol=0, atol=1e-16)


def test_wigner_samples_rejects_negative_n():
    model = _model("lvc_singlets.template")
    with pytest.raises(ValueError, match="n"):
        lvc_wigner_samples(model, -1, np.random.default_rng(0))
