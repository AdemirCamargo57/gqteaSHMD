"""LVC population ensembles (P7a-2 Task 7; Method reference item 13, ruling R9)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gqteashmd.dynamics.ensemble import PopulationResult, run_population_ensemble
from gqteashmd.dynamics.lvc import LVCModel, lvc_wigner_samples
from gqteashmd.dynamics.options import DynamicsOptions

DATA = Path(__file__).resolve().parents[1] / "data" / "lvc"
FIELDS = (
    "diag_classical",
    "diag_quantum",
    "mch_classical",
    "mch_quantum",
    "mch_spin_free_classical",
    "mch_spin_free_quantum",
)


def _run(n=6, n_steps=20, record_every=1, workers=1, options=None, state=3, soc=True):
    model = LVCModel.from_template(DATA / "lvc_soc.template", spin_orbit=soc)
    q0, v0 = lvc_wigner_samples(model, n, np.random.default_rng(7))
    options = options or DynamicsOptions(dt_au=20.0)
    return run_population_ensemble(
        model,
        q0,
        v0,
        initial_state=state,
        options=options,
        seed=11,
        n_steps=n_steps,
        record_every=record_every,
        workers=workers,
    )


def test_populations_are_identical_for_one_and_three_workers():
    a, b = _run(workers=1), _run(workers=3)
    for name in FIELDS:
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert (a.n_hops, a.n_frustrated, a.n_energy_drift) == (
        b.n_hops,
        b.n_frustrated,
        b.n_energy_drift,
    )


def test_every_population_row_sums_to_one():
    r = _run()
    assert r.n_energy_drift == 0
    for name in FIELDS:
        assert np.allclose(getattr(r, name).sum(axis=1), 1.0, atol=1e-10), name
    assert r.times_au[0] == 0.0 and r.times_au.size == 21
    assert r.labels_spin_free == ("S0", "S1", "T1", "T2")
    assert r.labels_expanded[2] == "T1(-1)"
    # time 0: expanded index 3 is T2(-1); the classical MCH population is 1 there (U = I at q0
    # would only be exact without mixing, so check the quantum spin-free sum instead)
    assert r.labels_expanded[3] == "T2(-1)"
    assert r.mch_spin_free_quantum[0].argmax() == 3
    assert np.isclose(r.mch_spin_free_classical[0].sum(), 1.0)


def test_forced_drift_is_counted_and_excluded():
    r = _run(options=DynamicsOptions(dt_au=20.0, energy_drift_limit_au=1e-300))
    assert r.n_energy_drift == r.n_trajectories == 6
    assert r.n_hops == 0 and np.isnan(r.diag_classical).all()
    ok = _run(n=4)
    assert ok.n_energy_drift == 0


def test_record_every_subsamples_the_full_record():
    full, sub = _run(record_every=1), _run(record_every=5)
    assert sub.times_au.size == 5
    for name in FIELDS:
        assert np.array_equal(getattr(sub, name), getattr(full, name)[::5]), name


def test_without_soc_a_fixed_s0_ensemble_stays_in_s0():
    model = LVCModel.from_template(DATA / "lvc_soc.template", spin_orbit=False)
    q = np.tile(model.reference_q, (3, 1))
    r = run_population_ensemble(
        model,
        q,
        np.zeros_like(q),
        initial_state=0,
        options=DynamicsOptions(dt_au=20.0),
        seed=1,
        n_steps=20,
    )
    assert np.allclose(r.mch_spin_free_quantum[:, 0], 1.0, atol=1e-6)
    assert np.allclose(r.mch_spin_free_classical[:, 0], 1.0)
    assert r.n_hops == 0


def test_invalid_arguments_are_value_errors():
    model = LVCModel.from_template(DATA / "lvc_soc.template")
    q = np.tile(model.reference_q, (2, 1))
    for kw in ({"n_steps": 0}, {"record_every": 0}, {"workers": 0}):
        args = {"n_steps": 3, "record_every": 1, "workers": 1, **kw}
        with pytest.raises(ValueError):
            run_population_ensemble(
                model,
                q,
                np.zeros_like(q),
                initial_state=0,
                options=replace(DynamicsOptions(dt_au=20.0)),
                seed=1,
                **args,
            )
    assert isinstance(_run(n=2, n_steps=2), PopulationResult)


# ---- orientation of the MCH populations (final review, Important 2) ----

# M = R_12(0.6, 0.8) R_23(0.6, 0.8), real orthogonal with |M| not symmetric:
# [[0.6, -0.48, 0.64], [0.8, 0.36, -0.48], [0.0, 0.8, 0.6]]; U = diag(1, i, -1) M (complex).
_M = np.array([[0.6, -0.48, 0.64], [0.8, 0.36, -0.48], [0.0, 0.8, 0.6]])
_U = np.diag([1.0, 1.0j, -1.0]) @ _M
_C = np.array([0.6, 0.0, 0.8j])  # diagonal amplitudes, |c|^2 = (0.36, 0, 0.64)


def test_population_row_orientation_with_a_non_symmetric_unitary():
    """Hand-computed: active diagonal state 1. Classical MCH = |U[:, 1]|^2 = |(-0.48, 0.36 i,
    -0.8)|^2 = (0.2304, 0.1296, 0.64), not the row |U[1, :]|^2 = (0.64, 0.1296, 0.2304).
    Quantum MCH = |U c|^2 with M c = (0.36 + 0.512 i, 0.48 - 0.384 i, 0.48 i), so
    (0.391744, 0.377856, 0.2304), not |U^dagger c|^2 = (0.1296, 0.492544, 0.377856)."""
    from gqteashmd.dynamics.ensemble import _population_row

    assert np.allclose(_U.conj().T @ _U, np.eye(3), atol=1e-15)
    row = _population_row(1, _C, _U)
    assert row.shape == (4, 3)
    np.testing.assert_allclose(row[0], [0.0, 1.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(row[1], [0.36, 0.0, 0.64], atol=1e-15)
    np.testing.assert_allclose(row[2], [0.2304, 0.1296, 0.64], atol=1e-14)
    np.testing.assert_allclose(row[3], [0.391744, 0.377856, 0.2304], atol=1e-14)
    # U = None: the MCH populations are the diagonal ones
    plain = _population_row(2, _C, None)
    np.testing.assert_array_equal(plain[2], plain[0])
    np.testing.assert_array_equal(plain[3], plain[1])


def test_spin_free_populations_sum_over_ms():
    """Counts (2, 0, 2): expanded order S0, S1, T1(-1), T2(-1), T1(0), T2(0), T1(+1), T2(+1);
    T1 sums the expanded columns 2, 4, 6 and T2 the columns 3, 5, 7."""
    from gqteashmd.dynamics.ensemble import _spin_free_matrix

    labels, free, matrix = _spin_free_matrix((2, 0, 2))
    assert labels == ("S0", "S1", "T1(-1)", "T2(-1)", "T1(0)", "T2(0)", "T1(+1)", "T2(+1)")
    assert free == ("S0", "S1", "T1", "T2")
    expected = np.zeros((8, 4))
    for i, j in ((0, 0), (1, 1), (2, 2), (4, 2), (6, 2), (3, 3), (5, 3), (7, 3)):
        expected[i, j] = 1.0
    np.testing.assert_array_equal(matrix, expected)
    r = _run()  # lvc_soc has counts (2, 0, 2)
    for kind in ("classical", "quantum"):
        mch = getattr(r, f"mch_{kind}")
        free_pops = getattr(r, f"mch_spin_free_{kind}")
        for j, columns in enumerate(([0], [1], [2, 4, 6], [3, 5, 7])):
            np.testing.assert_allclose(free_pops[:, j], mch[:, columns].sum(axis=1), atol=1e-15)
