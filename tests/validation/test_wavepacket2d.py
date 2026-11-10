"""Self-tests of the exact two-mode LVC solver ``wavepacket2d.py`` (P7a-2, Task 8).

Pre-registered thresholds: norm 1 to 1e-10; energy conserved to 1e-8 Eh; halving dt and
doubling the grid change no population by 1e-4 over 2000 au; with lambda = 0 and Sigma = 0 the
packet stays on the diabatic S1 exactly (to 1e-12). Model: ``tests/data/lvc/lvc_twomode``.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from gqteashmd.dynamics.lvc import LVCModel
from gqteashmd.dynamics.lvc_format import read_lvc_template

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "data" / "lvc" / "lvc_twomode.template"
T_TEST = 2000.0


def _load():
    name = "gqteashmd_validation_wavepacket2d"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / "wavepacket2d.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wp = _load()


@cache
def params():
    return read_lvc_template(TEMPLATE)


@cache
def run(
    dt: float = 0.5,
    refine: int = 1,
    widen: int = 1,
    vibronic_coupling: bool = True,
    spin_orbit: bool = True,
):
    """``refine`` multiplies the points per axis on the same box; ``widen`` multiplies the box
    about its centre and the points (dQ unchanged)."""
    model = wp.two_mode_lvc(params(), vibronic_coupling=vibronic_coupling, spin_orbit=spin_orbit)
    lo, hi = np.array(wp.GRID["q_min"]), np.array(wp.GRID["q_max"])
    centre, half = (lo + hi) / 2, (hi - lo) / 2
    return wp.propagate(
        model,
        t_final=T_TEST,
        dt=dt,
        record_every=round(20.0 / dt),
        n_grid=tuple(refine * widen * n for n in wp.GRID["n_grid"]),
        q_min=tuple(float(x) for x in centre - widen * half),
        q_max=tuple(float(x) for x in centre + widen * half),
    )


def test_hamiltonian_is_the_lvc_model_restricted_to_two_modes():
    """V(Q_7, Q_9) and the MCH energies agree with ``LVCModel`` (other Q = 0)."""
    lvc = LVCModel(params())
    model = wp.two_mode_lvc(params())
    assert model.labels_expanded == ("S0", "S1", "T1(-1)", "T1(0)", "T1(+1)")
    assert model.labels_spin_free == ("S0", "S1", "T1")
    rng = np.random.default_rng(5)
    for qa, qb in rng.uniform(-6.0, 6.0, size=(8, 2)):
        big_q = np.zeros(lvc.n_dof)
        big_q[[6, 8]] = qa, qb
        np.testing.assert_allclose(
            model.diabatic(np.array(qa), np.array(qb)),
            lvc.diabatic_hamiltonian(big_q),
            rtol=0,
            atol=1e-15,
        )
        snap = lvc.evaluate(lvc.cartesian(big_q), None)
        t = model.mch_vectors(np.array(qa), np.array(qb))
        h_sf = model.diabatic(np.array(qa), np.array(qb))
        if model.soc is not None:
            h_sf = h_sf - model.soc
        energies = np.real(np.diag(t.T @ h_sf @ t))
        np.testing.assert_allclose(energies, snap.energies, rtol=0, atol=1e-12)


def test_a_template_coupling_another_mode_is_refused():
    p = params()
    other = dataclasses.replace(p, kappa={**p.kappa, (1, 1, 7): 1e-3})
    with pytest.raises(ValueError, match=r"couples modes \[7\]"):
        wp.two_mode_lvc(other)


def test_the_first_grid_is_too_small_and_the_guard_says_so():
    """On 128 x 128 points on [-8, 8)^2 the packet reaches the grid edge (``BRIEF_GRID``)."""
    model = wp.two_mode_lvc(params())
    with pytest.raises(RuntimeError, match="reached the grid edge"):
        wp.propagate(model, t_final=T_TEST, **wp.BRIEF_GRID)


def test_norm_energy_and_population_sums():
    r = run()
    norm_err = float(np.max(np.abs(r["norm"] - 1.0)))
    energy_err = float(np.max(np.abs(r["energy"] - r["energy"][0])))
    print(
        f"\nnorm - 1: {norm_err:.2e}; max |E(t) - E(0)|: {energy_err:.2e} Eh; "
        f"edge {r['edge_norm']:.1e}"
    )
    assert norm_err < 1e-10
    assert energy_err < 1e-8
    for key in ("diabatic", "mch", "mch_spin_free"):
        np.testing.assert_allclose(r[key].sum(axis=1), r["norm"], rtol=0, atol=1e-12)
    assert r["diabatic"][0, 1] == pytest.approx(1.0, abs=1e-14)


def test_decoupled_packet_stays_on_the_diabatic_s1():
    r = run(vibronic_coupling=False, spin_orbit=False)
    others = np.delete(r["diabatic"], 1, axis=1)
    print(
        f"\ndecoupled: max |P_S1 - 1| {np.max(np.abs(r['diabatic'][:, 1] - 1)):.2e}, "
        f"max other {np.max(others):.2e}"
    )
    np.testing.assert_allclose(r["diabatic"][:, 1], 1.0, rtol=0, atol=1e-12)
    assert np.max(others) < 1e-12


def _max_change(a: dict, b: dict) -> float:
    return max(float(np.max(np.abs(a[k] - b[k]))) for k in ("diabatic", "mch", "mch_spin_free"))


@pytest.mark.slow
def test_converged_under_halving_dt():
    base, fine = run(), run(dt=0.25)
    assert np.allclose(base["times"], fine["times"])
    change = _max_change(base, fine)
    print(f"\nhalving dt: max population change {change:.2e} over {T_TEST:g} au")
    assert change < 1e-4


@pytest.mark.slow
@pytest.mark.parametrize(("refine", "widen"), [(2, 1), (1, 2)], ids=["points", "box"])
def test_converged_under_doubling_the_grid(refine, widen):
    """Twice the points per axis on the same box (dQ halved), and twice the box per axis with
    the same dQ."""
    base, fine = run(), run(refine=refine, widen=widen)
    change = _max_change(base, fine)
    print(f"\ndoubling the grid ({refine=}, {widen=}): max population change {change:.2e}")
    assert change < 1e-4
