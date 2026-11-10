from dataclasses import fields, replace
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from gqteashmd.dynamics.driver import Outcome, initial_state, run_trajectory, step
from gqteashmd.dynamics.lvc import LVCModel, lvc_wigner_samples
from gqteashmd.dynamics.models import TullySimpleAvoidedCrossing
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.randomness import GeneratorSource, ReplaySource
from gqteashmd.dynamics.records import StepRecord

MASS = 2000.0


def _start(model, x0=-10.0, k=20.0, active=0):
    return initial_state(model, np.array([x0]), np.array([k / MASS]), active)


def _run(model, state, options, rng, max_steps, stop=None):
    records: list[StepRecord] = []
    final, outcome = run_trajectory(
        model, state, options, rng, max_steps=max_steps, stop=stop, on_step=records.append
    )
    return final, outcome, records


def _exit(state):
    return abs(state.q[0]) > 10.0 and state.q[0] * state.v[0] > 0


def _assert_records_equal(a: list[StepRecord], b: list[StepRecord]) -> None:
    assert len(a) == len(b)
    for ra, rb in zip(a, b, strict=True):
        for fld in fields(StepRecord):
            xa, xb = getattr(ra, fld.name), getattr(rb, fld.name)
            if isinstance(xa, np.ndarray):
                assert np.array_equal(xa, xb), fld.name
            else:
                assert xa == xb, fld.name


def test_initial_state_defaults():
    model = TullySimpleAvoidedCrossing()
    state = _start(model, active=1)
    assert state.step == 0 and state.time_au == 0.0
    assert state.active == 1
    assert np.array_equal(state.c, np.array([0.0, 1.0], dtype=complex))
    assert state.c.dtype == complex
    snap = model.evaluate(np.array([-10.0]), None)
    assert np.array_equal(state.snapshot.energies, snap.energies)
    kinetic = 0.5 * MASS * (20.0 / MASS) ** 2
    assert state.e_total_initial == pytest.approx(snap.energies[1] + kinetic, abs=1e-15)


def test_initial_state_rejects_bad_input():
    model = TullySimpleAvoidedCrossing()
    q, v = np.array([-10.0]), np.array([0.01])
    with pytest.raises(ValueError, match="active"):
        initial_state(model, q, v, 2)
    with pytest.raises(ValueError, match="c0"):
        initial_state(model, q, v, 0, np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="q0"):
        initial_state(model, np.array([0.0, 1.0]), v, 0)


def test_energy_conservation_without_hopping():
    """Tully SAC, hopping off, dt = 1: E_tot conserved to < 1e-6 Eh over the whole passage."""
    model = TullySimpleAvoidedCrossing()
    options = DynamicsOptions(dt_au=1.0, hopping="none")
    state = _start(model)
    final, outcome, records = _run(
        model, state, options, GeneratorSource(np.random.default_rng(0)), max_steps=5000, stop=_exit
    )
    assert outcome is Outcome.STOPPED
    assert final.q[0] > 10.0  # transmitted on the ground state
    assert all(r.active == 0 for r in records)
    drift = max(abs(r.e_total - state.e_total_initial) for r in records)
    assert drift < 1e-6


def test_step_follows_the_documented_order():
    """One step reproduced by hand from the components (method reference 8, no hop)."""
    from gqteashmd.dynamics.decoherence import apply_decoherence
    from gqteashmd.dynamics.hopping import hopping_probabilities
    from gqteashmd.dynamics.integrators import acceleration, verlet_positions, verlet_velocities
    from gqteashmd.dynamics.propagators import propagate

    model = TullySimpleAvoidedCrossing()
    options = DynamicsOptions(dt_au=2.0, hopping="sharc")
    state = _start(model, x0=-1.0)
    state.c = np.array([0.8, 0.6j], dtype=complex)
    snap = state.snapshot
    m = model.masses
    a = acceleration(snap.gradients[0], m)
    q1 = verlet_positions(state.q, state.v, a, 2.0)
    snap1 = model.evaluate(q1, snap)
    v1 = verlet_velocities(state.v, a, acceleration(snap1.gradients[0], m), 2.0)
    c1, p, amps = propagate(
        state.c, snap.energies, snap1.energies, snap.nacs, snap1.nacs, state.v, v1, 2.0, 25
    )
    probs = hopping_probabilities(
        "sharc", 0, state.c, c1, p, amps, snap.nacs, snap1.nacs, state.v, v1, 2.0
    )
    kin = 0.5 * float(np.sum(m * v1 * v1))
    c2 = apply_decoherence(
        c1, 0, snap1.energies, kin, 2.0, scheme="edc", param=0.1, convention="sharc"
    )

    new, rec = step(model, state, options, ReplaySource([0.999999]))
    assert np.array_equal(new.q, q1)
    assert np.array_equal(new.v, v1)
    assert np.array_equal(new.c, c2)
    assert np.array_equal(rec.probabilities, probs)
    assert rec.random_number == 0.999999
    assert rec.hop_from is None and rec.hop_to is None and not rec.frustrated
    assert new.step == 1 and new.time_au == 2.0 and rec.step == 1 and rec.time_au == 2.0
    assert rec.e_kinetic == kin
    assert rec.e_total == snap1.energies[0] + kin
    assert np.array_equal(rec.energies, snap1.energies)
    assert new.snapshot is not snap
    assert new.e_total_initial == state.e_total_initial
    # the input state is not modified
    assert state.step == 0 and np.array_equal(state.q, np.array([-1.0]))


@pytest.mark.parametrize("hopping", ["sharc", "fssh"])
def test_step_passes_the_interpolation_and_edc_convention(hopping):
    """substep_interpolation reaches propagate and the FSSH flux; edc_convention reaches EDC."""
    from gqteashmd.dynamics.decoherence import apply_decoherence
    from gqteashmd.dynamics.hopping import hopping_probabilities
    from gqteashmd.dynamics.integrators import acceleration, verlet_positions, verlet_velocities
    from gqteashmd.dynamics.propagators import propagate

    model = TullySimpleAvoidedCrossing()
    options = DynamicsOptions(
        dt_au=2.0,
        hopping=hopping,
        substep_interpolation="endpoint",
        edc_convention="granucci_persico",
    )
    state = _start(model, x0=-0.5)
    state.c = np.array([0.8, 0.6j], dtype=complex)
    snap, m = state.snapshot, model.masses
    a = acceleration(snap.gradients[0], m)
    q1 = verlet_positions(state.q, state.v, a, 2.0)
    snap1 = model.evaluate(q1, snap)
    v1 = verlet_velocities(state.v, a, acceleration(snap1.gradients[0], m), 2.0)
    c1, p, amps = propagate(
        state.c, snap.energies, snap1.energies, snap.nacs, snap1.nacs, state.v, v1, 2.0, 25,
        interpolation="endpoint",
    )  # fmt: skip
    probs = hopping_probabilities(
        hopping, 0, state.c, c1, p, amps, snap.nacs, snap1.nacs, state.v, v1, 2.0,
        interpolation="endpoint",
    )  # fmt: skip
    kin = 0.5 * float(np.sum(m * v1 * v1))
    c2 = apply_decoherence(
        c1, 0, snap1.energies, kin, 2.0, scheme="edc", param=0.1, convention="granucci_persico"
    )
    new, rec = step(model, state, options, ReplaySource([0.999999]))
    assert np.array_equal(new.c, c2)
    assert np.array_equal(rec.probabilities, probs)
    default, _ = step(model, state, replace(options, substep_interpolation="midpoint"),
                      ReplaySource([0.999999]))  # fmt: skip
    assert np.max(np.abs(default.c - new.c)) > 1e-7


def test_one_random_number_per_step_even_without_hopping():
    model = TullySimpleAvoidedCrossing()
    options = DynamicsOptions(dt_au=1.0, hopping="none")
    rng = ReplaySource([0.1, 0.2, 0.3])
    _, outcome, records = _run(model, _start(model), options, rng, max_steps=3)
    assert outcome is Outcome.FINISHED
    assert [r.random_number for r in records] == [0.1, 0.2, 0.3]
    assert rng.state()["position"] == 3


@pytest.mark.parametrize("scheme", ["sharc", "fssh"])
@pytest.mark.parametrize("ekin", ["parallel_vel", "parallel_nac"])
def test_hops_keep_total_energy_continuous(scheme, ekin):
    """xi = 0 every step: a hop happens whenever a probability is positive."""
    model = TullySimpleAvoidedCrossing()
    options = DynamicsOptions(dt_au=1.0, hopping=scheme, ekin_correction=ekin)
    state = _start(model, x0=-4.0, k=20.0)
    _, _, records = _run(model, state, options, ReplaySource([0.0] * 2000), max_steps=800)
    hops = [i for i, r in enumerate(records) if r.hop_to is not None]
    assert len(hops) >= 2
    previous = [state.e_total_initial] + [r.e_total for r in records]
    for i in hops:
        r = records[i]
        assert r.hop_from is not None and r.hop_from != r.hop_to
        assert r.active == r.hop_to
        assert r.probabilities[r.hop_from] == 0.0
        assert not r.frustrated
        assert abs(r.e_total - previous[i]) < 1e-6
    assert max(abs(e - state.e_total_initial) for e in previous) < 1e-6


@pytest.mark.parametrize("frustrated", ["keep", "reflect"])
def test_frustrated_hop_is_recorded(frustrated):
    """Too little kinetic energy for the upper state: the attempted hops are frustrated."""
    model = TullySimpleAvoidedCrossing()
    options = DynamicsOptions(dt_au=1.0, hopping="sharc", frustrated=frustrated)
    state = _start(model, x0=-1.5, k=4.0)  # K = 0.004 Eh < gap (about 0.018 Eh)
    _, _, records = _run(model, state, options, ReplaySource([0.0] * 3000), max_steps=200)
    frustrated_steps = [r for r in records if r.frustrated]
    assert frustrated_steps
    first = frustrated_steps[0]
    assert first.hop_from is None and first.hop_to is None
    assert first.active == 0
    assert all(r.active == 0 for r in records)
    if frustrated == "reflect":
        assert first.v[0] < 0.0  # parallel_vel reflection: v' = -v
    else:
        assert first.v[0] > 0.0


@pytest.mark.parametrize("scheme", ["sharc", "fssh"])
def test_same_seed_gives_identical_records_and_replay_reproduces(scheme):
    """The window contains hops (SAC from x = -1.5, k = 30, dt = 2, seed 5: steps 32 and 89),
    so the replay also reproduces the hop decisions and the velocity adjustments."""
    model = TullySimpleAvoidedCrossing()
    options = DynamicsOptions(dt_au=2.0, hopping=scheme)
    state = _start(model, x0=-1.5, k=30.0)
    _, out_a, rec_a = _run(model, state, options, GeneratorSource(np.random.default_rng(5)), 100)
    _, out_b, rec_b = _run(model, state, options, GeneratorSource(np.random.default_rng(5)), 100)
    assert out_a is out_b
    _assert_records_equal(rec_a, rec_b)
    assert [r.step for r in rec_a if r.hop_to is not None] == [32, 89]
    logged = [r.random_number for r in rec_a]
    _, out_c, rec_c = _run(model, state, options, ReplaySource(logged), 100)
    assert out_c is out_a
    _assert_records_equal(rec_a, rec_c)


def test_hop_uses_the_nac_at_the_new_geometry():
    """parallel_nac adjusts v' along d'_{lambda j} of snap' (at q'), not along d of snap (q).
    2D model near its conical intersection (-0.25, 0), where the NAC direction turns by about
    45 degrees over one step; xi = 0 forces the hop 1 -> 0."""
    from gqteashmd.dynamics.integrators import acceleration, verlet_positions, verlet_velocities
    from gqteashmd.dynamics.models import TwoDimensionalCrossing
    from gqteashmd.dynamics.velocity import adjust_velocity

    model = TwoDimensionalCrossing()
    m, dt = model.masses, 5.0
    options = DynamicsOptions(
        dt_au=dt, hopping="sharc", ekin_correction="parallel_nac", decoherence="none"
    )
    state = initial_state(model, np.array([-0.24, 0.01]), np.array([-2e-3, 0.0]), 1)
    state.c = np.array([-0.6, 0.8], dtype=complex)
    snap0 = state.snapshot
    a = acceleration(snap0.gradients[1], m)
    q1 = verlet_positions(state.q, state.v, a, dt)
    snap1 = model.evaluate(q1, snap0)
    v1 = verlet_velocities(state.v, a, acceleration(snap1.gradients[1], m), dt)
    d0, d1 = snap0.nacs[1, 0], snap1.nacs[1, 0]
    cosine = d0 @ d1 / (np.linalg.norm(d0) * np.linalg.norm(d1))
    assert cosine < 0.9  # the NAC really turns over this step
    delta_e = float(snap1.energies[0] - snap1.energies[1])
    with_new = adjust_velocity(v1, m, delta_e, method="parallel_nac", nac=d1, frustrated="keep")
    with_old = adjust_velocity(v1, m, delta_e, method="parallel_nac", nac=d0, frustrated="keep")

    new, rec = step(model, state, options, ReplaySource([0.0]))
    assert (rec.hop_from, rec.hop_to) == (1, 0)
    assert np.array_equal(new.v, with_new.velocity)
    assert np.max(np.abs(new.v - with_old.velocity)) > 1e-6


def test_energy_drift_outcome():
    model = TullySimpleAvoidedCrossing()
    options = DynamicsOptions(dt_au=1.0, hopping="none", energy_drift_limit_au=1e-15)
    final, outcome, records = _run(
        model, _start(model, x0=-1.0), options, GeneratorSource(np.random.default_rng(0)), 500
    )
    assert outcome is Outcome.ENERGY_DRIFT
    assert final.step == len(records) < 500
    assert abs(records[-1].e_total - final.e_total_initial) > 1e-15


def test_stop_callback_and_finished():
    model = TullySimpleAvoidedCrossing()
    options = DynamicsOptions(dt_au=1.0)
    rng = GeneratorSource(np.random.default_rng(3))
    final, outcome, records = _run(
        model, _start(model), options, rng, 1000, stop=lambda s: s.q[0] > -9.95
    )
    assert outcome is Outcome.STOPPED
    assert final.q[0] > -9.95 and records[-2].q[0] <= -9.95
    assert final.step == len(records)

    final2, outcome2, records2 = _run(model, _start(model), options, rng, 7)
    assert outcome2 is Outcome.FINISHED
    assert final2.step == 7 and len(records2) == 7
    assert final2.time_au == 7.0

    # a state that already satisfies stop takes no step
    final3, outcome3, records3 = _run(model, final, options, rng, 10, stop=lambda s: True)
    assert outcome3 is Outcome.STOPPED and records3 == [] and final3 is final


class _SocModel(TullySimpleAvoidedCrossing):
    """Tully SAC with a constant off-diagonal MCH Hamiltonian element 0.01 Eh.

    ``when = "later"``: only once a previous snapshot exists; ``"always"``: at every geometry;
    ``"first"``: only at the first geometry (previous is None).
    """

    def __init__(self, when: str = "later") -> None:
        super().__init__()
        self.when = when

    def evaluate(self, q, previous):
        snap = super().evaluate(q, previous)
        coupled = {"later": previous is not None, "always": True, "first": previous is None}
        if not coupled[self.when]:
            return snap
        h = snap.hamiltonian.copy()
        h[0, 1] = h[1, 0] = 0.01
        return replace(snap, hamiltonian=h)


def test_hamiltonian_becoming_non_diagonal_is_refused():
    """Ruling J: U was None (the identity) and H^MCH gains an off-diagonal element."""
    options = DynamicsOptions(dt_au=1.0)
    model = _SocModel("later")
    state = _start(model)
    assert state.u is None
    with pytest.raises(ValueError, match="became non-diagonal during the trajectory"):
        step(model, state, options, ReplaySource([0.5]))


def test_hamiltonian_becoming_diagonal_keeps_tracking_u():
    """Ruling J, reverse: a U other than None stays in use when H^MCH turns diagonal."""
    from gqteashmd.dynamics.representation import diagonal_basis

    options = DynamicsOptions(dt_au=1.0)
    model = _SocModel("first")
    state = _start(model, x0=-1.0)
    assert state.u is not None
    new, rec = step(model, state, options, ReplaySource([0.5]))
    assert new.u is not None and rec.u is not None
    _, expected = diagonal_basis(new.snapshot.hamiltonian, state.u)
    assert np.array_equal(new.u, expected)
    assert abs(np.linalg.norm(new.c) - 1.0) <= 1e-12


def test_constant_off_diagonal_coupling_runs_in_the_diagonal_basis():
    """A constant MCH-basis coupling is a constant diabatic SOC only where the NAC vanishes, so
    the run stays at x < -7 (through the coupling region from x = -3 this model is not
    conservative: measured drift 8.6e-3 Eh; here 2e-12 Eh)."""
    model = _SocModel("always")
    options = DynamicsOptions(dt_au=1.0, hopping="none")
    state = _start(model, x0=-10.0)
    assert state.u is not None
    _, outcome, records = _run(
        model, state, options, GeneratorSource(np.random.default_rng(0)), max_steps=300
    )
    assert outcome is Outcome.FINISHED
    assert all(r.u is not None for r in records)
    assert max(abs(r.e_total - state.e_total_initial) for r in records) < 1e-5


# ---- P7a-2: the diagonal representation on the LVC fixtures (Method reference 4-12) ----

LVC = Path(__file__).resolve().parents[1] / "data" / "lvc"
FSSH_SOC_MESSAGE = "hopping = 'fssh' needs the MCH basis"
PARALLEL_NAC_SOC_MESSAGE = "parallel_nac needs a real coupling vector"
H_FD = 1e-5


@cache
def _lvc(name: str = "lvc_soc.template") -> LVCModel:
    return LVCModel.from_template(LVC / name)


def _sample(model, seed=7):
    q, v = lvc_wigner_samples(model, 1, np.random.default_rng(seed))
    return q[0], v[0]


def _lvc_start(model, active=1, seed=7, c0=None):
    q, v = _sample(model, seed)
    return initial_state(model, q, v, active, c0)


def _lvc_run(options, steps, *, rng=None, name="lvc_soc.template", active=1):
    model = _lvc(name)
    state = _lvc_start(model, active)
    rng = GeneratorSource(np.random.default_rng(3)) if rng is None else rng
    final, outcome, records = _run(model, state, options, rng, max_steps=steps)
    return state, final, outcome, records


@cache
def _soc_drift(dt: float, steps: int, transformation: str = "ngt") -> float:
    options = DynamicsOptions(dt_au=dt, hopping="none", gradient_transformation=transformation)
    state, _, outcome, records = _lvc_run(options, steps)
    assert outcome is Outcome.FINISHED
    assert all(r.u is not None for r in records)  # the spin-orbit path is really used
    return max(abs(r.e_total - state.e_total_initial) for r in records)


def test_soc_energy_conservation_scales_as_dt_squared():
    """lvc_soc from a Wigner sample in S1, hopping off, 2000 steps at dt = 0.5 (and 4000 at
    0.25): max |E_tot - E_tot(0)| < 1e-5 Eh, ratio of the two drifts in [3, 5]."""
    coarse = _soc_drift(0.5, 2000)
    fine = _soc_drift(0.25, 4000)
    assert coarse < 1e-5, coarse
    assert 3.0 <= coarse / fine <= 5.0, (coarse, fine)


def test_soc_hops_keep_total_energy_continuous():
    """hopping = "sharc" with xi = 0 every step: |E_tot(step) - E_tot(step - 1)| < 1e-8 Eh at
    every hop step."""
    options = DynamicsOptions(dt_au=0.5, hopping="sharc")
    state, _, _, records = _lvc_run(options, 1000, rng=ReplaySource([0.0] * 1000))
    previous = [state.e_total_initial] + [r.e_total for r in records]
    hops = [i for i, r in enumerate(records) if r.hop_to is not None]
    assert len(hops) >= 2
    for i in hops:
        r = records[i]
        assert r.active == r.hop_to and not r.frustrated
        assert abs(r.e_total - previous[i]) < 1e-8, (i, r.e_total - previous[i])


def test_without_the_gradient_transformation_energy_is_not_conserved():
    """gradient_transformation = "none" drops the off-diagonal G^MCH terms: drift > 10 x ngt."""
    assert _soc_drift(0.5, 2000, "none") > 10.0 * _soc_drift(0.5, 2000)


def test_diagonal_gradient_matches_finite_differences_along_the_trajectory():
    """At 5 recorded steps, g^diag_lambda = d E^diag_lambda / dq (central differences of
    model.evaluate + diagonal_basis at q +- h e_k) to 1e-6."""
    from gqteashmd.dynamics.driver import active_gradient
    from gqteashmd.dynamics.representation import diagonal_basis

    model = _lvc()
    options = DynamicsOptions(dt_au=0.5, hopping="sharc")
    state = _lvc_start(model)
    rng = GeneratorSource(np.random.default_rng(3))
    worst = 0.0
    for i in range(1, 501):
        state, _ = step(model, state, options, rng)
        if i % 100:
            continue
        assert state.u is not None
        g = active_gradient(state, options)
        fd = np.empty(model.n_dof)
        for k in range(model.n_dof):
            e = []
            for sign in (1.0, -1.0):
                q = state.q.copy()
                q[k] += sign * H_FD
                snap = model.evaluate(q, state.snapshot)
                e.append(diagonal_basis(snap.hamiltonian, state.u)[0][state.active])
            fd[k] = (e[0] - e[1]) / (2.0 * H_FD)
        worst = max(worst, float(np.max(np.abs(g - fd))))
    assert worst <= 1e-6, worst


def _forced_stream(n: int) -> ReplaySource:
    """The seed-21 generator's first n numbers with every 10th replaced by 0: the hop
    probabilities on lvc_soc are small (max about 7e-6 per step from this sample), so the
    forced attempts make the window contain hops (measured: steps 1, 21, 24, 61, 101)."""
    values = np.random.default_rng(21).random(n)
    values[::10] = 0.0
    return ReplaySource([float(x) for x in values])


def test_eselect_wider_than_every_gap_is_bit_identical():
    base = _lvc_run(DynamicsOptions(dt_au=0.5), 300, rng=_forced_stream(300))[3]
    wide = _lvc_run(DynamicsOptions(dt_au=0.5, eselect_Eh=10.0), 300, rng=_forced_stream(300))[3]
    assert any(r.hop_to is not None for r in base)
    _assert_records_equal(base, wide)


def test_eselect_narrow_changes_the_trajectory_and_conserves_energy():
    base = _lvc_run(DynamicsOptions(dt_au=0.5), 500)[3]
    state, _, outcome, narrow = _lvc_run(DynamicsOptions(dt_au=0.5, eselect_Eh=0.05), 500)
    assert outcome is Outcome.FINISHED
    assert max(float(np.max(np.abs(a.q - b.q))) for a, b in zip(base, narrow, strict=True)) > 0
    drift = max(abs(r.e_total - state.e_total_initial) for r in narrow)
    assert drift < 1e-3, drift


def test_fssh_with_spin_orbit_coupling_is_refused():
    model = _lvc()
    q, v = _sample(model)
    fssh = DynamicsOptions(dt_au=0.5, hopping="fssh")
    with pytest.raises(ValueError, match=FSSH_SOC_MESSAGE):
        initial_state(model, q, v, 1, options=fssh)
    state = initial_state(model, q, v, 1)  # without options the check is made at every step
    with pytest.raises(ValueError, match=FSSH_SOC_MESSAGE):
        step(model, state, fssh, ReplaySource([0.5]))
    # without spin-orbit coupling fssh is fine
    singlets = _lvc("lvc_singlets.template")
    q, v = _sample(singlets)
    initial_state(singlets, q, v, 1, options=fssh)


def test_parallel_nac_with_spin_orbit_coupling_raises_at_the_first_hop():
    """xi = 0: a target exists exactly when a probability is positive; every step before the
    one that raises had none."""
    model = _lvc()
    options = DynamicsOptions(dt_au=0.5, ekin_correction="parallel_nac")
    state = _lvc_start(model)
    rng = ReplaySource([0.0] * 500)
    for _ in range(500):
        try:
            state, rec = step(model, state, options, rng)
        except ValueError as exc:
            assert PARALLEL_NAC_SOC_MESSAGE in str(exc)
            break
        assert not np.any(rec.probabilities > 0)
        assert rec.hop_to is None and not rec.frustrated
    else:
        pytest.fail("no hop was attempted")


def test_parallel_nac_on_the_singlet_lvc_hops_and_conserves_energy():
    """No SOC (U = None): parallel_nac uses snapshot.nacs; xi = 0 forces the hops."""
    options = DynamicsOptions(dt_au=0.5, ekin_correction="parallel_nac")
    state, _, outcome, records = _lvc_run(
        options, 1000, rng=ReplaySource([0.0] * 1000), name="lvc_singlets.template", active=2
    )
    assert outcome is Outcome.FINISHED
    assert state.u is None and all(r.u is None for r in records)
    assert sum(r.hop_to is not None for r in records) >= 1
    drift = max(abs(r.e_total - state.e_total_initial) for r in records)
    assert drift < 1e-5, drift


def test_initial_state_maps_the_mch_state_to_the_diagonal_state():
    """Item 12: c^diag = U0^dagger c^MCH, lambda = argmax |c^diag|^2 and
    E_tot(0) = E^diag_lambda + K."""
    from gqteashmd.dynamics.driver import kinetic_energy
    from gqteashmd.dynamics.representation import diagonal_basis

    model = _lvc()
    q, v = _sample(model)
    e_diag, u = diagonal_basis(model.evaluate(q, None).hamiltonian, None, first_step=True)
    assert u is not None
    state = initial_state(model, q, v, 1)
    assert np.array_equal(state.u, u)
    expected = u.conj().T @ np.eye(model.n_states)[1]
    assert np.allclose(state.c, expected, atol=1e-15, rtol=0)
    assert state.active == int(np.argmax(np.abs(state.c) ** 2))
    assert state.active != 1  # the diagonal states are energy-ordered (triplets below S1)
    assert state.e_total_initial == float(e_diag[state.active]) + kinetic_energy(v, model.masses)

    c0 = np.zeros(model.n_states, dtype=complex)
    c0[0], c0[1] = 0.6, 0.8j
    mixed = initial_state(model, q, v, 1, c0)
    assert np.allclose(mixed.c, u.conj().T @ c0, atol=1e-15, rtol=0)
    assert mixed.active == int(np.argmax(np.abs(mixed.c) ** 2))
    with pytest.raises(ValueError, match="c0 must have norm 1"):
        initial_state(model, q, v, 1, 2.0 * c0)
    with pytest.raises(ValueError, match="active"):
        initial_state(model, q, v, model.n_states)


def test_records_carry_the_mch_view():
    """Items 4 and 13: c_mch = U' c', populations sum to 1, active_mch = argmax |U'_{., lambda}|."""
    _, _, _, records = _lvc_run(DynamicsOptions(dt_au=0.5), 300, rng=_forced_stream(300))
    assert any(r.hop_to is not None for r in records)
    for r in records:
        assert r.u is not None and r.c_mch is not None and r.energies_diag is not None
        assert abs(float(np.sum(np.abs(r.c_mch) ** 2)) - 1.0) <= 1e-12
        assert np.array_equal(r.c_mch, r.u @ r.c)
        assert r.active_mch == int(np.argmax(np.abs(r.u[:, r.active])))
        assert np.all(np.diff(r.energies_diag) >= 0)
        assert r.e_total == float(r.energies_diag[r.active]) + r.e_kinetic


def test_soc_step_follows_the_documented_order():
    """One spin-orbit step reproduced by hand from the components (item 11, no hop)."""
    from gqteashmd.dynamics.decoherence import apply_decoherence
    from gqteashmd.dynamics.hopping import sharc_probabilities
    from gqteashmd.dynamics.integrators import acceleration, verlet_positions, verlet_velocities
    from gqteashmd.dynamics.propagators import propagate_diagonal
    from gqteashmd.dynamics.representation import (
        diagonal_basis,
        diagonal_gradient,
        diagonal_tensor,
        gradient_tensor,
    )

    model = _lvc()
    options = DynamicsOptions(dt_au=0.5)
    state = _lvc_start(model)
    snap, m, lam = state.snapshot, model.masses, state.active
    g0 = diagonal_gradient(diagonal_tensor(gradient_tensor(snap), state.u), lam)
    a = acceleration(g0, m)
    q1 = verlet_positions(state.q, state.v, a, 0.5)
    snap1 = model.evaluate(q1, snap)
    e1, u1 = diagonal_basis(snap1.hamiltonian, state.u)
    g1 = diagonal_gradient(diagonal_tensor(gradient_tensor(snap1), u1), lam)
    v1 = verlet_velocities(state.v, a, acceleration(g1, m), 0.5)
    c1, r, _ = propagate_diagonal(
        state.c, state.u, u1, snap.hamiltonian, snap1.hamiltonian, snap.nacs, snap1.nacs,
        state.v, v1, 0.5, 25, coupling="nacdr", overlaps=snap1.overlaps,
        interpolation="midpoint",
    )  # fmt: skip
    probs = sharc_probabilities(lam, state.c, c1, r)
    kin = 0.5 * float(np.sum(m * v1 * v1))
    c2 = apply_decoherence(c1, lam, e1, kin, 0.5, scheme="edc", param=0.1, convention="sharc")

    new, rec = step(model, state, options, ReplaySource([0.999999]))
    assert rec.hop_to is None and not rec.frustrated
    assert np.array_equal(new.q, q1)
    assert np.array_equal(new.v, v1)
    assert np.array_equal(new.u, u1) and np.array_equal(rec.u, u1) and rec.u is not new.u
    assert np.array_equal(new.c, c2)
    assert np.array_equal(rec.probabilities, probs)
    assert np.array_equal(rec.energies, snap1.energies)
    assert np.array_equal(rec.energies_diag, e1)
    assert rec.e_total == float(e1[lam]) + kin
