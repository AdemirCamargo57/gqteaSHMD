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
