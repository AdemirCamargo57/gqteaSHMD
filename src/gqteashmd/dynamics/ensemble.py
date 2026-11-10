"""Ensembles of 1D model trajectories: wavepacket initial conditions and branching statistics."""

from __future__ import annotations

import math
import multiprocessing
import zlib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from gqteashmd.dynamics.driver import Outcome, run_trajectory
from gqteashmd.dynamics.driver import initial_state as make_initial_state
from gqteashmd.dynamics.lvc import LVCModel, lvc_wigner_samples
from gqteashmd.dynamics.models import ElectronicModel
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.randomness import GeneratorSource
from gqteashmd.dynamics.records import StepRecord, TrajectoryState
from gqteashmd.dynamics.representation import mch_coefficients
from gqteashmd.dynamics.spin import expanded_states, spin_free_label

SAMPLING_MODES = ("wigner", "fixed")
BRANCHES = ("transmitted_lower", "transmitted_upper", "reflected_lower", "reflected_upper")
Z_95 = 1.959963984540054
_TRAJECTORY_TAG = zlib.crc32(b"model-dynamics")
_INITIAL_TAG = zlib.crc32(b"model-initial")


def wavepacket_initial_conditions(
    n: int,
    x0: float,
    k: float,
    sigma: float,
    mass: float,
    rng: np.random.Generator,
    sampling: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Positions q (n, 1) and velocities v (n, 1) for psi(x) ~ exp(-(x - x0)^2/sigma^2 + ikx).

    |psi|^2 has standard deviation sigma/2 and the Wigner function has s_p = 1/sigma
    (hbar = 1). ``"wigner"`` draws x ~ N(x0, sigma/2), p ~ N(k, 1/sigma); ``"fixed"`` gives
    every trajectory x0 and k/mass.
    """
    if sampling not in SAMPLING_MODES:
        raise ValueError(f"sampling must be one of {', '.join(SAMPLING_MODES)}; got {sampling!r}")
    if sampling == "fixed":
        return np.full((n, 1), float(x0)), np.full((n, 1), k / mass)
    x = rng.normal(x0, sigma / 2.0, size=(n, 1))
    p = rng.normal(k, 1.0 / sigma, size=(n, 1))
    return x, p / mass


def wilson_interval(count: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for ``count`` successes in ``n`` trials."""
    if n <= 0:
        return 0.0, 1.0
    p = count / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class Branching:
    transmitted_lower: int = 0
    transmitted_upper: int = 0
    reflected_lower: int = 0
    reflected_upper: int = 0
    unfinished: int = 0
    energy_drift: int = 0
    n_trajectories: int = field(default=0)

    @property
    def n_finished(self) -> int:
        return sum(getattr(self, name) for name in BRANCHES)

    def counts(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in (*BRANCHES, "unfinished", "energy_drift")}

    def probabilities(self) -> dict[str, float]:
        """Fractions over the finished (stopped) trajectories; all zero if there are none."""
        n = self.n_finished
        return {name: (getattr(self, name) / n if n else 0.0) for name in BRANCHES}

    def intervals_95(self) -> dict[str, tuple[float, float]]:
        n = self.n_finished
        return {name: wilson_interval(getattr(self, name), n) for name in BRANCHES}


def _leaving(x_exit: float, state: TrajectoryState) -> bool:
    """Stop rule: |x| > x_exit while moving outward."""
    x, v = float(state.q[0]), float(state.v[0])
    return abs(x) > x_exit and v * math.copysign(1.0, x) > 0


def _run_chunk(
    model: ElectronicModel,
    q0: np.ndarray,
    v0: np.ndarray,
    seeds: Sequence[np.random.SeedSequence],
    initial_state: int,
    options: DynamicsOptions,
    x_exit: float,
    max_steps: int,
) -> list[str]:
    """Run the trajectories of one chunk; return the counter name of each, in order."""
    stop = partial(_leaving, x_exit)
    labels: list[str] = []
    for qi, vi, seq in zip(q0, v0, seeds, strict=True):
        rng = GeneratorSource(np.random.default_rng(seq))
        state = make_initial_state(model, qi, vi, initial_state, options=options)
        final, outcome = run_trajectory(model, state, options, rng, max_steps=max_steps, stop=stop)
        if outcome is Outcome.ENERGY_DRIFT:
            labels.append("energy_drift")
        elif outcome is Outcome.FINISHED:
            labels.append("unfinished")
        else:
            side = "transmitted" if final.q[0] > 0 else "reflected"
            level = "lower" if final.active == 0 else "upper"
            labels.append(f"{side}_{level}")
    return labels


def run_model_ensemble(
    model: ElectronicModel,
    *,
    n_trajectories: int,
    x0: float,
    k: float,
    sigma: float,
    initial_state: int,
    options: DynamicsOptions,
    seed: int,
    x_exit: float,
    max_steps: int,
    sampling: str = "wigner",
    workers: int = 1,
) -> Branching:
    """Run ``n_trajectories`` trajectories until |x| > x_exit while moving outward.

    Trajectory i always uses its own stream (child i of the ``model-dynamics`` seed sequence)
    and initial condition i, so the counts are identical for any ``workers``. With
    ``workers > 1`` the trajectories run in chunks of consecutive indices on a pool of
    ``spawn``-started worker processes, which are all joined before returning (also on an
    error). ``model`` and ``options`` must be picklable, and a calling script must guard its
    entry point with ``if __name__ == "__main__":``. Nothing about threads is set: BLAS
    threading is irrelevant for these 2 x 2 problems.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"workers must be an integer >= 1; got {workers!r}")
    mass = float(model.masses[0])
    init_rng = np.random.default_rng(np.random.SeedSequence([seed, _INITIAL_TAG]))
    q0, v0 = wavepacket_initial_conditions(n_trajectories, x0, k, sigma, mass, init_rng, sampling)
    children = np.random.SeedSequence([seed, _TRAJECTORY_TAG]).spawn(n_trajectories)
    common = (initial_state, options, x_exit, max_steps)

    if workers == 1 or n_trajectories <= 1:
        labels = _run_chunk(model, q0, v0, children, *common)
    else:
        size = max(1, math.ceil(n_trajectories / (4 * workers)))
        bounds = [(i, min(i + size, n_trajectories)) for i in range(0, n_trajectories, size)]
        labels = []
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(workers, len(bounds)), mp_context=context) as pool:
            futures = [
                pool.submit(_run_chunk, model, q0[a:b], v0[a:b], children[a:b], *common)
                for a, b in bounds
            ]
            try:
                for future in futures:  # submission order = trajectory order
                    labels.extend(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    result = Branching(n_trajectories=n_trajectories)
    for name in labels:
        setattr(result, name, getattr(result, name) + 1)
    return result


@dataclass
class PopulationResult:
    """Ensemble-averaged populations at ``times_au`` (energy-drift trajectories excluded, R9)."""

    times_au: np.ndarray  # (n_out,)
    labels_expanded: tuple[str, ...]
    labels_spin_free: tuple[str, ...]
    diag_classical: np.ndarray  # (n_out, n_exp)
    diag_quantum: np.ndarray
    mch_classical: np.ndarray
    mch_quantum: np.ndarray
    mch_spin_free_classical: np.ndarray  # (n_out, n_sf)
    mch_spin_free_quantum: np.ndarray
    n_trajectories: int  # requested
    n_energy_drift: int
    n_hops: int  # over the included trajectories
    n_frustrated: int


def _spin_free_matrix(counts: Sequence[int]) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    """Labels of the expanded and spin-free states and the (n_exp, n_sf) summing matrix."""
    states = expanded_states(counts)
    free: list[str] = []
    for s in states:
        label = spin_free_label(s)
        if label not in free:
            free.append(label)
    matrix = np.zeros((len(states), len(free)))
    for i, s in enumerate(states):
        matrix[i, free.index(spin_free_label(s))] = 1.0
    return tuple(s.label() for s in states), tuple(free), matrix


def _population_row(active: int, c: np.ndarray, u: np.ndarray | None) -> np.ndarray:
    """(4, n_exp): diagonal classical/quantum, MCH classical/quantum (item 13)."""
    n = c.size
    diag_c = np.zeros(n)
    diag_c[active] = 1.0
    mch_c = np.abs(u[:, active]) ** 2 if u is not None else diag_c
    return np.vstack([diag_c, np.abs(c) ** 2, mch_c, np.abs(mch_coefficients(u, c)) ** 2])


def _run_population_chunk(
    model: ElectronicModel,
    q0: np.ndarray,
    v0: np.ndarray,
    seeds: Sequence[np.random.SeedSequence],
    initial_state: int,
    options: DynamicsOptions,
    n_steps: int,
    record_every: int,
) -> list[tuple[np.ndarray, bool, int, int]]:
    """Per trajectory: populations (n_out, 4, n_exp), drift flag, hops, frustrated hops."""
    n_out = n_steps // record_every + 1
    results: list[tuple[np.ndarray, bool, int, int]] = []
    for qi, vi, seq in zip(q0, v0, seeds, strict=True):
        rng = GeneratorSource(np.random.default_rng(seq))
        state = make_initial_state(model, qi, vi, initial_state, options=options)
        n = state.c.size
        pops = np.zeros((n_out, 4, n))
        pops[0] = _population_row(state.active, state.c, state.u)
        hops = [0, 0]

        def on_step(rec: StepRecord, pops: np.ndarray = pops, hops: list[int] = hops) -> None:
            if rec.hop_to is not None:
                hops[0] += 1
            if rec.frustrated:
                hops[1] += 1
            if rec.step % record_every == 0:
                pops[rec.step // record_every] = _population_row(rec.active, rec.c, rec.u)

        _, outcome = run_trajectory(model, state, options, rng, max_steps=n_steps, on_step=on_step)
        results.append((pops, outcome is Outcome.ENERGY_DRIFT, hops[0], hops[1]))
    return results


def run_population_ensemble(
    model: ElectronicModel,
    q0: np.ndarray,
    v0: np.ndarray,
    *,
    initial_state: int,
    options: DynamicsOptions,
    seed: int,
    n_steps: int,
    record_every: int = 1,
    workers: int = 1,
) -> PopulationResult:
    """Run ``len(q0)`` trajectories of exactly ``n_steps`` steps and average the populations.

    ``q0`` and ``v0`` are (n, f); ``initial_state`` is an MCH (expanded) index. Populations are
    recorded at step 0 and every ``record_every`` steps (item 13). Trajectory i uses child i of
    the ``model-dynamics`` seed sequence, so the result does not depend on ``workers``.
    A trajectory ending with ``energy_drift`` is excluded from every average and counted (R9);
    if all drift, the averages are NaN. Workers return only the populations at the output times.
    """
    for name, value in (("workers", workers), ("n_steps", n_steps), ("record_every", record_every)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1; got {value!r}")
    q0 = np.atleast_2d(np.asarray(q0, dtype=float))
    v0 = np.atleast_2d(np.asarray(v0, dtype=float))
    n_traj = q0.shape[0]
    if v0.shape != q0.shape or n_traj < 1:
        raise ValueError("q0 and v0 must have the same shape (n, f) with n >= 1")
    labels_exp, labels_free, matrix = _spin_free_matrix(model.state_counts)
    children = np.random.SeedSequence([seed, _TRAJECTORY_TAG]).spawn(n_traj)
    common = (initial_state, options, n_steps, record_every)

    if workers == 1 or n_traj <= 1:
        results = _run_population_chunk(model, q0, v0, children, *common)
    else:
        size = max(1, math.ceil(n_traj / (4 * workers)))
        bounds = [(i, min(i + size, n_traj)) for i in range(0, n_traj, size)]
        results = []
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(workers, len(bounds)), mp_context=context) as pool:
            futures = [
                pool.submit(_run_population_chunk, model, q0[a:b], v0[a:b], children[a:b], *common)
                for a, b in bounds
            ]
            try:
                for future in futures:
                    results.extend(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    kept = [r for r in results if not r[1]]
    n_out = n_steps // record_every + 1
    if kept:
        mean = np.mean([r[0] for r in kept], axis=0)  # (n_out, 4, n_exp)
    else:
        mean = np.full((n_out, 4, matrix.shape[0]), np.nan)
    return PopulationResult(
        times_au=np.arange(n_out) * record_every * options.dt_au,
        labels_expanded=labels_exp,
        labels_spin_free=labels_free,
        diag_classical=mean[:, 0],
        diag_quantum=mean[:, 1],
        mch_classical=mean[:, 2],
        mch_quantum=mean[:, 3],
        mch_spin_free_classical=mean[:, 2] @ matrix,
        mch_spin_free_quantum=mean[:, 3] @ matrix,
        n_trajectories=n_traj,
        n_energy_drift=len(results) - len(kept),
        n_hops=sum(r[2] for r in kept),
        n_frustrated=sum(r[3] for r in kept),
    )


def lvc_initial_conditions(
    model: LVCModel, n: int, seed: int, sampling: str
) -> tuple[np.ndarray, np.ndarray]:
    """(q, v), each (n, f): T = 0 Wigner samples (the ``model-initial`` stream) or ``"fixed"``
    (the reference geometry at rest for every trajectory)."""
    if sampling not in SAMPLING_MODES:
        raise ValueError(f"sampling must be one of {', '.join(SAMPLING_MODES)}; got {sampling!r}")
    if sampling == "fixed":
        q = np.tile(model.reference_q, (n, 1))
        return q, np.zeros_like(q)
    rng = np.random.default_rng(np.random.SeedSequence([seed, _INITIAL_TAG]))
    return lvc_wigner_samples(model, n, rng)
