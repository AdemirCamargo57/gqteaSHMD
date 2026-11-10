"""Surface-hopping driver in the diagonal representation (P7a-2 Method reference 4-12).

The state holds q, v, the diagonal amplitudes c^diag, the active diagonal state lambda, the
(unmasked) MCH snapshot at q and the diagonal basis U at q (``None`` = the identity in MCH order,
which is what every model without an off-diagonal MCH Hamiltonian element has, so those
trajectories are bit-identical to P7a-1). One step, in the order of item 11:

1. Mask the state's snapshot around E^diag_lambda(q) (``eselect_Eh``, item 6; ``None`` keeps
   everything and copies nothing), then a = -M^-1 g^diag_lambda (item 5).
2. q' = q + v dt + a dt^2 / 2.
3. snap' = model.evaluate(q', previous=snap) (the unmasked snapshot of the state).
4. (E^diag', U') = ``diagonal_basis(snap'.hamiltonian, U)``. With U = ``None`` a non-diagonal
   H^MCH at q' is a ``ValueError`` (ruling J: U cannot appear mid-trajectory); with a U other than
   ``None`` the basis is tracked against it also when H^MCH turns diagonal.
5. Mask snap' around E^diag'_lambda (old lambda), then G^diag' and
   v' = v + (a - M^-1 g^diag'_lambda) dt / 2.
6. Propagate c^diag over [t, t + dt] with ``propagate_diagonal`` (item 7): the full H^MCH at t
   and t + dt, the masked NACs, v and v', U and U' and, for ``coupling = "overlap"``, the
   overlaps of snap'. This gives c' and R^diag.
7. Hopping probabilities from lambda (``"sharc"`` with R^diag and c^diag; ``"fssh"`` only with
   U = ``None``, item 8); draw exactly one xi ~ U[0, 1), also when hopping is ``"none"``.
8. If a target j is chosen: adjust v' for dE = E^diag'_j - E^diag'_lambda (item 9; for
   ``parallel_nac`` the direction is ``snapshot.nacs[lambda, j]`` of the masked snap' when U' is
   ``None``, else Re K^diag'_lambda j, refused when it is complex). Accepted: lambda <- j.
   Frustrated: lambda unchanged, v' kept or reflected.
9. Decoherence on c' with K = v'^T M v' / 2 and E^diag' (``options.edc_convention``, item 10).
10. a' = -M^-1 g^diag'_lambda of snap' masked around the (possibly new) lambda; it is recomputed
    from the stored state at the next step's (1), which is exactly that mask, so it is not part
    of the state (the "second electronic-structure call" of an accepted hop with ``eselect``).
11. Record with E_tot = E^diag'_lambda + K; ``run_trajectory`` checks the energy drift
    |E_tot - E_tot(0)|. The record also carries E^diag', c^MCH = U' c', the active MCH state
    argmax_alpha |U'_alpha,lambda| and a copy of U' (``None`` for the identity).
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
from gqteashmd.dynamics.propagators import propagate_diagonal
from gqteashmd.dynamics.randomness import RandomSource
from gqteashmd.dynamics.records import ElectronicSnapshot, StepRecord, TrajectoryState
from gqteashmd.dynamics.representation import (
    diagonal_basis,
    diagonal_energies,
    diagonal_gradient,
    diagonal_nac,
    diagonal_tensor,
    gradient_tensor,
    masked_snapshot,
    mch_coefficients,
    mch_state,
    selection_mask,
)
from gqteashmd.dynamics.velocity import adjust_velocity

NORM_TOLERANCE = 1e-8
COMPLEX_NAC_TOLERANCE = 1e-10  # relative: max |Im K| > this * max(1, max |K|) is complex
FSSH_MESSAGE = "hopping = 'fssh' needs the MCH basis (no spin-orbit coupling); use 'sharc'"
COMPLEX_NAC_MESSAGE = (
    "parallel_nac needs a real coupling vector; with spin-orbit coupling use parallel_vel"
)
APPEARING_U_MESSAGE = (
    "the MCH Hamiltonian became non-diagonal during the trajectory (U was the identity); "
    "start from a geometry where the spin-orbit coupling is present"
)


class Outcome(StrEnum):
    """Why a trajectory run ended (a str subclass: ``Outcome.FINISHED == "finished"``)."""

    FINISHED = "finished"
    STOPPED = "stopped"
    ENERGY_DRIFT = "energy_drift"


def kinetic_energy(v: np.ndarray, masses: np.ndarray) -> float:
    """K = v^T M v / 2."""
    return 0.5 * float(np.sum(np.asarray(masses) * v * v))


def _check_fssh(options: DynamicsOptions | None, u: np.ndarray | None) -> None:
    """Item 8: ``"fssh"`` is defined only in the MCH basis (U = None)."""
    if options is not None and options.hopping == "fssh" and u is not None:
        raise ValueError(FSSH_MESSAGE)


def _masked(
    snapshot: ElectronicSnapshot,
    energies_diag: np.ndarray | None,
    active: int,
    eselect: float | None,
) -> ElectronicSnapshot:
    """The snapshot masked around E^diag_active (item 6); itself when ``eselect`` is None."""
    if eselect is None or energies_diag is None:
        return snapshot
    mask = selection_mask(snapshot.energies, float(energies_diag[active]), eselect)
    return masked_snapshot(snapshot, mask)


def _diagonal_tensor(
    snapshot: ElectronicSnapshot, u: np.ndarray | None, transformation: str
) -> np.ndarray | None:
    """G^diag = U^dagger G^MCH U (item 5); None when U is None (the MCH data are used as is)."""
    if u is None:
        return None
    return diagonal_tensor(gradient_tensor(snapshot, transformation), u)


def _gradient(snapshot: ElectronicSnapshot, g_diag: np.ndarray | None, active: int) -> np.ndarray:
    """g^diag_active: Re G^diag_aa, or the MCH gradient itself when U is None."""
    if g_diag is None:
        return np.asarray(snapshot.gradients[active])
    return diagonal_gradient(g_diag, active)


def active_gradient(state: TrajectoryState, options: DynamicsOptions) -> np.ndarray:
    """g^diag_lambda of the state, from its snapshot masked around E^diag_lambda(q) (item 11,
    step 1); the force on the nuclei is -g. A copy."""
    snap = state.snapshot
    e_diag = None
    if options.eselect_Eh is not None:
        e_diag = diagonal_energies(snap.hamiltonian, state.u)
    masked = _masked(snap, e_diag, state.active, options.eselect_Eh)
    g_diag = _diagonal_tensor(masked, state.u, options.gradient_transformation)
    return np.array(_gradient(masked, g_diag, state.active), dtype=float, copy=True)


def _hop_direction(
    snapshot: ElectronicSnapshot,
    g_diag: np.ndarray | None,
    energies_diag: np.ndarray,
    active: int,
    target: int,
) -> np.ndarray:
    """Item 9: the parallel_nac direction at q' from the masked snapshot."""
    if g_diag is None:
        return np.asarray(snapshot.nacs[active, target])
    k = diagonal_nac(g_diag, energies_diag, active, target)
    scale = max(1.0, float(np.max(np.abs(k))) if k.size else 0.0)
    if k.size and float(np.max(np.abs(np.imag(k)))) > COMPLEX_NAC_TOLERANCE * scale:
        raise ValueError(COMPLEX_NAC_MESSAGE)
    return np.array(np.real(k), dtype=float)


def initial_state(
    model: ElectronicModel,
    q0: np.ndarray,
    v0: np.ndarray,
    active: int,
    c0: np.ndarray | None = None,
    *,
    options: DynamicsOptions | None = None,
) -> TrajectoryState:
    """Evaluate the model at q0 and return the state at t = 0 with E_tot(0) recorded (item 12).

    ``active`` is an MCH state index (expanded, canonical order) and ``c0`` (default: the unit
    vector on ``active``; a given ``c0`` must have norm 1) is in the MCH basis. With U0 from
    ``diagonal_basis(H^MCH(q0), None, first_step=True)`` the state holds c^diag = U0^dagger c^MCH
    and lambda = argmax_a |c^diag_a|^2 (lowest index on ties); with U0 = ``None`` (no off-diagonal
    H^MCH element) c^diag = c^MCH and lambda = ``active``, the P7a-1 behaviour. E_tot(0) =
    E^diag_lambda + K. ``options`` (optional) lets the ``"fssh"`` check of item 8 be made here.
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
    e_diag, u = diagonal_basis(snap.hamiltonian, None, first_step=True)
    _check_fssh(options, u)
    lam = active
    if u is not None:
        c = u.conj().T @ c
        lam = int(np.argmax(np.abs(c) ** 2))
    e_total = float(e_diag[lam]) + kinetic_energy(v, model.masses)
    return TrajectoryState(
        time_au=0.0,
        step=0,
        q=q,
        v=v,
        c=c,
        active=lam,
        snapshot=snap,
        e_total_initial=e_total,
        u=u,
    )


def step(
    model: ElectronicModel,
    state: TrajectoryState,
    options: DynamicsOptions,
    rng: RandomSource,
) -> tuple[TrajectoryState, StepRecord]:
    """Advance one nuclear step (module docstring); ``state`` is not modified."""
    dt = options.dt_au
    eselect = options.eselect_Eh
    transformation = options.gradient_transformation
    masses = np.asarray(model.masses, dtype=float)
    lam = state.active
    snap0, u0 = state.snapshot, state.u
    q0, v0, c0 = state.q, state.v, state.c
    _check_fssh(options, u0)

    # 1: the masked snapshot at q and the acceleration of the active diagonal state
    e_diag0 = diagonal_energies(snap0.hamiltonian, u0) if eselect is not None else None
    masked0 = _masked(snap0, e_diag0, lam, eselect)
    a0 = acceleration(
        _gradient(masked0, _diagonal_tensor(masked0, u0, transformation), lam), masses
    )

    # 2-4: positions, electronic structure at q', diagonal basis at q'
    q1 = verlet_positions(q0, v0, a0, dt)
    snap1 = model.evaluate(q1, snap0)
    e_diag1, u1 = diagonal_basis(snap1.hamiltonian, u0)
    if u0 is None and u1 is not None:
        raise ValueError(APPEARING_U_MESSAGE)

    # 5: the masked snapshot at q' (old lambda) and the velocities
    masked1 = _masked(snap1, e_diag1, lam, eselect)
    g_diag1 = _diagonal_tensor(masked1, u1, transformation)
    v1 = verlet_velocities(v0, a0, acceleration(_gradient(masked1, g_diag1, lam), masses), dt)

    # 6: electronic propagation over [t, t + dt] in the diagonal basis
    c1, propagator, substeps = propagate_diagonal(
        c0,
        u0,
        u1,
        snap0.hamiltonian,
        snap1.hamiltonian,
        masked0.nacs,
        masked1.nacs,
        v0,
        v1,
        dt,
        options.nsubsteps,
        coupling=options.coupling,
        overlaps=snap1.overlaps,
        interpolation=options.substep_interpolation,
    )

    # 7: probabilities and the one random number of this step
    if substeps is None:  # only the fast path (U = U' = None, "nacdr") returns them; fssh needs it
        substeps = np.empty((0, c0.shape[0]), dtype=complex)
    probabilities = hopping_probabilities(
        options.hopping,
        lam,
        c0,
        c1,
        propagator,
        substeps,
        masked0.nacs,
        masked1.nacs,
        v0,
        v1,
        dt,
        interpolation=options.substep_interpolation,
    )
    xi = float(rng.uniform())

    # 8: hop with velocity adjustment at fixed q'
    hop_from: int | None = None
    hop_to: int | None = None
    frustrated = False
    target = choose_state(lam, probabilities, xi)
    if target is not None:
        delta_e = float(e_diag1[target] - e_diag1[lam])
        direction = None
        if options.ekin_correction == "parallel_nac":
            direction = _hop_direction(masked1, g_diag1, e_diag1, lam, target)
        adjustment = adjust_velocity(
            v1,
            masses,
            delta_e,
            method=options.ekin_correction,
            nac=direction,
            frustrated=options.frustrated,
        )
        v1 = adjustment.velocity
        if adjustment.accepted:
            hop_from, hop_to, lam = lam, target, target
        else:
            frustrated = True

    # 9: decoherence with K(v') and E^diag(q')
    e_kin = kinetic_energy(v1, masses)
    c1 = apply_decoherence(
        c1,
        lam,
        e_diag1,
        e_kin,
        dt,
        scheme=options.decoherence,
        param=options.decoherence_param_Eh,
        convention=options.edc_convention,
    )

    # 10-11: a' follows from the state at the next step's (1); record
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
        u=u1,
    )
    record = StepRecord(
        step=new_step,
        time_au=time,
        q=q1.copy(),
        v=v1.copy(),
        c=c1.copy(),
        active=lam,
        energies=np.array(snap1.energies, copy=True),
        e_total=float(e_diag1[lam]) + e_kin,
        e_kinetic=e_kin,
        probabilities=probabilities,
        random_number=xi,
        hop_from=hop_from,
        hop_to=hop_to,
        frustrated=frustrated,
        energies_diag=np.array(e_diag1, copy=True),
        c_mch=np.array(mch_coefficients(u1, c1), copy=True),
        active_mch=mch_state(u1, lam),
        u=None if u1 is None else np.array(u1, copy=True),
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
