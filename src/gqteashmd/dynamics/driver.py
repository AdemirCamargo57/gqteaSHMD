"""Surface-hopping driver: one nuclear step in the documented order (method reference 8).

Step order, from (q, v, a = -M^-1 g_lambda(q), c, lambda) at time t:

1. q' = q + v dt + a dt^2 / 2.
2. snap' = model.evaluate(q', previous=snap); a U other than ``None`` from
   ``diagonal_basis(snap'.hamiltonian, u_previous)`` (a non-diagonal, spin-orbit Hamiltonian)
   raises ``NotImplementedError(SOC_MESSAGE)`` until the driver propagates in the diagonal basis.
3. v' = v + (a - M^-1 g'_lambda) dt / 2, with the gradient of the *old* active state at q'.
4. Propagate c over [t, t + dt] with E, d interpolated between snap and snap' and v between
   v and v' (``options.substep_interpolation``); this gives c' and the step propagator P.
5. Hopping probabilities from lambda (``options.hopping``); draw exactly one xi ~ U[0, 1),
   also when hopping is ``"none"``.
6. If a target j is chosen: adjust v' for dE = E'_j - E'_lambda (NAC d'_lambda j for
   ``parallel_nac``). Accepted: lambda <- j. Frustrated: lambda unchanged, v' kept or reflected.
7. Decoherence on c' with K = v'^T M v' / 2 and the energies of snap'
   (``options.edc_convention``).
8. a' = -M^-1 g'_lambda for the (possibly new) active state; it is recomputed from the stored
   snapshot at the start of the next step, so it is not part of the state.
9. Record; ``run_trajectory`` checks the energy drift |E_tot - E_tot(0)|.

The propagation basis is the MCH basis in P7a-1. U is not stored in the state: ``run_trajectory``
passes ``u_previous=None`` (the identity reference), which is exact while U is diagonal.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

import numpy as np

from gqteashmd.dynamics.decoherence import apply_decoherence
from gqteashmd.dynamics.hopping import choose_state, hopping_probabilities
from gqteashmd.dynamics.integrators import acceleration, verlet_positions, verlet_velocities
from gqteashmd.dynamics.models import ElectronicModel
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.propagators import propagate
from gqteashmd.dynamics.randomness import RandomSource
from gqteashmd.dynamics.records import StepRecord, TrajectoryState
from gqteashmd.dynamics.representation import SOC_MESSAGE, diagonal_basis
from gqteashmd.dynamics.velocity import adjust_velocity

NORM_TOLERANCE = 1e-8


class Outcome(StrEnum):
    """Why a trajectory run ended (a str subclass: ``Outcome.FINISHED == "finished"``)."""

    FINISHED = "finished"
    STOPPED = "stopped"
    ENERGY_DRIFT = "energy_drift"


def _require_mch(u: np.ndarray | None) -> None:
    """The driver still propagates in the MCH basis: any U other than None is refused."""
    if u is not None:
        raise NotImplementedError(SOC_MESSAGE)


def kinetic_energy(v: np.ndarray, masses: np.ndarray) -> float:
    """K = v^T M v / 2."""
    return 0.5 * float(np.sum(np.asarray(masses) * v * v))


def initial_state(
    model: ElectronicModel,
    q0: np.ndarray,
    v0: np.ndarray,
    active: int,
    c0: np.ndarray | None = None,
) -> TrajectoryState:
    """Evaluate the model at q0 and return the state at t = 0 with E_tot(0) recorded.

    ``c0`` defaults to the unit vector on ``active``; a given ``c0`` must have norm 1.
    """
    f, n = int(model.n_dof), int(model.n_states)
    q = np.array(q0, dtype=float).reshape(-1)
    v = np.array(v0, dtype=float).reshape(-1)
    if q.shape != (f,):
        raise ValueError(f"q0 must have {f} components; got {q.size}")
    if v.shape != (f,):
        raise ValueError(f"v0 must have {f} components; got {v.size}")
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(v))):
        raise ValueError("q0 and v0 must be finite")
    if isinstance(active, bool) or not isinstance(active, int | np.integer):
        raise ValueError(f"active must be an integer state index; got {active!r}")
    active = int(active)
    if not 0 <= active < n:
        raise ValueError(f"active must be in 0..{n - 1}; got {active}")
    if c0 is None:
        c = np.zeros(n, dtype=complex)
        c[active] = 1.0
    else:
        c = np.array(c0, dtype=complex).reshape(-1)
        if c.shape != (n,):
            raise ValueError(f"c0 must have {n} components; got {c.size}")
        norm = float(np.linalg.norm(c))
        if not abs(norm - 1.0) <= NORM_TOLERANCE:
            raise ValueError(f"c0 must have norm 1; got {norm!r}")
    snap = model.evaluate(q, None)
    _require_mch(diagonal_basis(snap.hamiltonian, None, first_step=True)[1])
    e_total = float(snap.energies[active]) + kinetic_energy(v, model.masses)
    return TrajectoryState(
        time_au=0.0,
        step=0,
        q=q,
        v=v,
        c=c,
        active=active,
        snapshot=snap,
        e_total_initial=e_total,
    )


def step(
    model: ElectronicModel,
    state: TrajectoryState,
    options: DynamicsOptions,
    rng: RandomSource,
    u_previous: np.ndarray | None = None,
) -> tuple[TrajectoryState, StepRecord]:
    """Advance one nuclear step (module docstring); ``state`` is not modified."""
    dt = options.dt_au
    masses = np.asarray(model.masses, dtype=float)
    lam = state.active
    snap0 = state.snapshot
    q0, v0, c0 = state.q, state.v, state.c

    # 1-3: positions, electronic structure at q', velocities with the old active gradient
    a0 = acceleration(snap0.gradients[lam], masses)
    q1 = verlet_positions(q0, v0, a0, dt)
    snap1 = model.evaluate(q1, snap0)
    _require_mch(diagonal_basis(snap1.hamiltonian, u_previous)[1])
    v1 = verlet_velocities(v0, a0, acceleration(snap1.gradients[lam], masses), dt)

    # 4: electronic propagation over [t, t + dt]
    c1, propagator, substeps = propagate(
        c0,
        snap0.energies,
        snap1.energies,
        snap0.nacs,
        snap1.nacs,
        v0,
        v1,
        dt,
        options.nsubsteps,
        interpolation=options.substep_interpolation,
    )

    # 5: probabilities and the one random number of this step
    probabilities = hopping_probabilities(
        options.hopping,
        lam,
        c0,
        c1,
        propagator,
        substeps,
        snap0.nacs,
        snap1.nacs,
        v0,
        v1,
        dt,
        interpolation=options.substep_interpolation,
    )
    xi = float(rng.uniform())

    # 6: hop with velocity adjustment at fixed q'
    hop_from: int | None = None
    hop_to: int | None = None
    frustrated = False
    target = choose_state(lam, probabilities, xi)
    if target is not None:
        delta_e = float(snap1.energies[target] - snap1.energies[lam])
        adjustment = adjust_velocity(
            v1,
            masses,
            delta_e,
            method=options.ekin_correction,
            nac=snap1.nacs[lam, target],
            frustrated=options.frustrated,
        )
        v1 = adjustment.velocity
        if adjustment.accepted:
            hop_from, hop_to, lam = lam, target, target
        else:
            frustrated = True

    # 7: decoherence with K(v') and E(q')
    e_kin = kinetic_energy(v1, masses)
    c1 = apply_decoherence(
        c1,
        lam,
        snap1.energies,
        e_kin,
        dt,
        scheme=options.decoherence,
        param=options.decoherence_param_Eh,
        convention=options.edc_convention,
    )

    # 8-9: a' follows from snap1.gradients[lam] at the next step; record
    new_step = state.step + 1
    time = state.time_au + dt
    new_state = TrajectoryState(
        time_au=time,
        step=new_step,
        q=q1,
        v=v1,
        c=c1,
        active=lam,
        snapshot=snap1,
        e_total_initial=state.e_total_initial,
    )
    record = StepRecord(
        step=new_step,
        time_au=time,
        q=q1.copy(),
        v=v1.copy(),
        c=c1.copy(),
        active=lam,
        energies=np.array(snap1.energies, copy=True),
        e_total=float(snap1.energies[lam]) + e_kin,
        e_kinetic=e_kin,
        probabilities=probabilities,
        random_number=xi,
        hop_from=hop_from,
        hop_to=hop_to,
        frustrated=frustrated,
    )
    return new_state, record


def run_trajectory(
    model: ElectronicModel,
    state: TrajectoryState,
    options: DynamicsOptions,
    rng: RandomSource,
    *,
    max_steps: int,
    stop: Callable[[TrajectoryState], bool] | None = None,
    on_step: Callable[[StepRecord], None] | None = None,
) -> tuple[TrajectoryState, Outcome]:
    """Take up to ``max_steps`` steps (counted from ``state``, not from t = 0).

    Before every step, ``stop(state)`` True ends the run with ``STOPPED`` (so a state that
    already satisfies it takes no step). After every step, ``on_step`` receives the record and
    |E_tot - E_tot(0)| > ``energy_drift_limit_au`` (or a non-finite E_tot) ends the run with
    ``ENERGY_DRIFT``. Otherwise the run ends with ``FINISHED`` after ``max_steps`` steps.
    """
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError(f"max_steps must be an integer >= 0; got {max_steps!r}")
    limit = options.energy_drift_limit_au
    for _ in range(max_steps):
        if stop is not None and stop(state):
            return state, Outcome.STOPPED
        state, record = step(model, state, options, rng)
        if on_step is not None:
            on_step(record)
        if not abs(record.e_total - state.e_total_initial) <= limit:
            return state, Outcome.ENERGY_DRIFT
    if stop is not None and stop(state):
        return state, Outcome.STOPPED
    return state, Outcome.FINISHED
