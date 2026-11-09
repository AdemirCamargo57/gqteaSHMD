import multiprocessing

import numpy as np
import pytest

from gqteashmd.dynamics.ensemble import (
    Branching,
    run_model_ensemble,
    wavepacket_initial_conditions,
    wilson_interval,
)
from gqteashmd.dynamics.models import TullySimpleAvoidedCrossing
from gqteashmd.dynamics.options import DynamicsOptions


def test_wigner_moments_match_the_packet():
    n, x0, k, sigma, mass = 20000, -10.0, 20.0, 1.0, 2000.0
    q, v = wavepacket_initial_conditions(n, x0, k, sigma, mass, np.random.default_rng(1), "wigner")
    assert q.shape == (n, 1) and v.shape == (n, 1)
    sx, sp = sigma / 2, 1 / sigma
    assert abs(q.mean() - x0) < 5 * sx / np.sqrt(n)
    assert abs(v.mean() * mass - k) < 5 * sp / np.sqrt(n)
    assert q.std() == pytest.approx(sx, rel=5 * (1 / np.sqrt(2 * n)))
    assert (v * mass).std() == pytest.approx(sp, rel=5 * (1 / np.sqrt(2 * n)))


def test_fixed_sampling_is_exact():
    q, v = wavepacket_initial_conditions(
        3, -4.0, 10.0, 2.0, 2000.0, np.random.default_rng(1), "fixed"
    )
    assert np.all(q == -4.0) and np.allclose(v, 0.005)


def test_unknown_sampling_is_an_error():
    with pytest.raises(ValueError, match="sampling"):
        wavepacket_initial_conditions(1, 0.0, 1.0, 1.0, 1.0, np.random.default_rng(1), "x")


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(0, 50)[0] == pytest.approx(0.0, abs=1e-12)


def test_branching_probabilities():
    b = Branching(transmitted_lower=3, reflected_upper=1, unfinished=5)
    assert b.n_finished == 4
    assert b.probabilities()["transmitted_lower"] == 0.75
    assert Branching().probabilities()["reflected_lower"] == 0.0


def _run(seed, n=50, workers=1, initial_state=0):
    return run_model_ensemble(
        TullySimpleAvoidedCrossing(),
        n_trajectories=n,
        x0=-10.0,
        k=30.0,
        sigma=20 / 30,
        initial_state=initial_state,
        options=DynamicsOptions(dt_au=1.0, energy_drift_limit_au=0.0184),
        seed=seed,
        x_exit=10.0,
        max_steps=5000,
        sampling="wigner",
        workers=workers,
    )


def test_sac_ensemble_probabilities_sum_to_one_and_are_deterministic():
    a, b = _run(7, 12), _run(7, 12)
    assert a == b
    assert a.n_finished == 12 and a.unfinished == 0 and a.energy_drift == 0
    assert sum(a.probabilities().values()) == pytest.approx(1.0)
    # Not an invariant: at k = 30 (E = 0.225 Eh, far above the SAC asymptotes at +-0.01 Eh)
    # reflection is energetically possible but negligible; this seed gives none.
    assert a.reflected_lower == 0 and a.reflected_upper == 0


def test_counts_do_not_depend_on_the_number_of_workers():
    serial = _run(5, 9)
    assert _run(5, 9, workers=2) == serial
    assert _run(5, 9, workers=4) == serial  # more chunks than trajectories per worker
    assert serial.n_trajectories == 9 and serial.n_finished == 9
    assert multiprocessing.active_children() == []


@pytest.mark.parametrize("workers", [0, -1, 1.5, True, "2"])
def test_invalid_workers_is_an_error(workers):
    with pytest.raises(ValueError, match="workers"):
        _run(1, 2, workers=workers)


def test_a_worker_error_reaches_the_caller_and_no_worker_is_left():
    with pytest.raises(ValueError, match="active"):
        _run(1, 4, workers=2, initial_state=5)
    assert multiprocessing.active_children() == []
