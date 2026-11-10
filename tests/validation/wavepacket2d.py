"""Exact quantum dynamics of a two-mode LVC model with triplets and SOC (split-operator FFT; test
helper).

Our own reference solver for the P7a-2 LVC validation (Method reference items 2, 3 and 13).
Atomic units, hbar = 1.

- **Coordinates.** The dimensionless normal coordinates Q = (Q_a, Q_b) of the two modes the
  template couples (``lvc_twomode``: modes 7 and 9, 0-based 6 and 8). Every other mode enters
  the Hamiltonian only through its own oscillator omega_i Q_i^2 / 2 on the diagonal (no epsilon,
  kappa or lambda), so it factors out exactly and is left out; ``two_mode_lvc`` raises
  ``ValueError`` if the template couples any other mode.
- **Hamiltonian.** H = sum_i omega_i P_i^2 / 2 + V(Q), P_i = -i d/dQ_i, with V the complex
  (N_exp, N_exp) spin-expanded diabatic matrix of Method reference item 2 restricted to the two
  modes: on every (m, M_s) block of the canonical order the spin-free block
  V^(m)_aa = V0 + epsilon_a + sum_i kappa_a,i Q_i, V^(m)_ab = sum_i lambda_ab,i Q_i, with
  V0 = sum_i omega_i Q_i^2 / 2, plus the constant diabatic SOC Sigma (symmetrised,
  (Sigma + Sigma^dagger) / 2). The parameters are read from the template's ``LVCParameters``.
- **Propagation.** Uniform periodic grid; one step is the Strang splitting
  exp(-iV dt/2) exp(-iT dt) exp(-iV dt/2). T is applied in momentum space (2D FFT); the factor
  exp(-iV dt/2) is precomputed at every grid point from the eigendecomposition of the small
  Hermitian matrix V(Q) = W diag(E) W^dagger.
- **Initial state.** psi = exp(-(Q_a^2 + Q_b^2)/2) / sqrt(pi) (the ground state of V0) on one
  diabatic component (default: the expanded index of S1), normalised on the grid.
- **Populations.** Diabatic: sum over the grid of |psi_alpha|^2 dQ. MCH: at every grid point
  psi is projected on the local spin-free MCH states, c(Q) = T(Q)^T psi(Q), with T the block
  diagonal matrix of the per-multiplicity eigenvectors W^(m)(Q) on every (m, M_s) block
  (Method reference item 3), and |c_alpha(Q)|^2 dQ is summed over the grid; the spin-summed
  MCH populations add the M_s components of each (multiplicity, state).
  T(Q) has a conical singularity at the S0/S1 intersection (Q_7 = -10, Q_9 = 0 for
  ``lvc_twomode``), so this projection sum converges only as dQ^2, while psi itself (and the
  diabatic populations) converges spectrally. The projection is therefore made on a grid
  ``PROJECTION_REFINE`` = 2 times finer per axis, onto which psi is Fourier-interpolated
  (zero-padded FFT; this keeps the norm exactly). Measured without the refinement: halving dQ
  changes the diabatic populations by 2e-11 but the MCH ones by 1.7e-4.
- **Energy.** <T> from the momentum-space density, <V> = sum psi^dagger V psi dQ.
- **Guard.** The density in the outer ``EDGE_FRACTION`` of the grid along either axis (where
  the periodic grid would wrap the packet around) must stay below ``edge_tolerance`` at every
  record; else ``RuntimeError``.
- **Grid.** ``GRID``: 256 x 128 points on Q_a in [-14, 18), Q_b in [-8, 8) (dQ = 0.125 on both
  axes). The box is asymmetric in Q_7 because the packet reaches Q_7 = -8.6 on S1 and +14.4 on
  S0. The 128 x 128 grid on [-8, 8)^2 first specified (``BRIEF_GRID``) is too small for
  ``lvc_twomode``: on the diabatic S1 the packet swings to Q_7 = -2 kappa/omega = -8.6, and the
  part that reaches S0 there swings out to Q_7 = +14.4 (measured: 33 % of the density in the
  outer 3 % at t <= 4000 au).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.fft

from gqteashmd.dynamics.lvc_format import LVCParameters
from gqteashmd.dynamics.spin import check_counts, expanded_states, spin_free_label

EDGE_FRACTION = 0.03
FFT_WORKERS = 4  # threads of scipy.fft (independent transforms: results do not depend on it)
MODES = (6, 8)  # lvc_twomode: modes 7 and 9 of the template (1-based)
GRID = {"n_grid": (256, 128), "q_min": (-14.0, -8.0), "q_max": (18.0, 8.0)}
BRIEF_GRID = {"n_grid": (128, 128), "q_min": (-8.0, -8.0), "q_max": (8.0, 8.0)}
PROJECTION_REFINE = 2


@dataclass(frozen=True, eq=False)
class TwoModeLVC:
    """The two-mode diabatic Hamiltonian; ``spans`` are (multiplicity block, start, stop)."""

    omega: np.ndarray  # (2,) Eh
    labels_expanded: tuple[str, ...]
    labels_spin_free: tuple[str, ...]
    spin_free_matrix: np.ndarray  # (n_exp, n_sf) 0/1
    spans: tuple[tuple[int, int, int], ...]
    epsilon: tuple[np.ndarray, ...]  # per multiplicity block (n_m,)
    kappa: tuple[np.ndarray, ...]  # (n_m, 2)
    lam: tuple[np.ndarray, ...]  # (2, n_m, n_m) symmetric
    soc: np.ndarray | None  # (n_exp, n_exp) complex Hermitian
    n_states: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_states", len(self.labels_expanded))

    def spin_free_blocks(self, qa: np.ndarray, qb: np.ndarray) -> list[np.ndarray]:
        """Real symmetric (..., n_m, n_m) spin-free blocks at the points (qa, qb)."""
        q = np.stack(np.broadcast_arrays(qa, qb), axis=-1)  # (..., 2)
        v0 = 0.5 * np.sum(self.omega * q**2, axis=-1)
        blocks = []
        for eps, kap, lam in zip(self.epsilon, self.kappa, self.lam, strict=True):
            v = np.einsum("...i,ijk->...jk", q, lam)
            diag = np.arange(eps.size)
            v[..., diag, diag] += (v0[..., None] + eps) + q @ kap.T
            blocks.append(v)
        return blocks

    def diabatic(self, qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
        """Complex (..., N_exp, N_exp) diabatic matrix at the points (qa, qb)."""
        blocks = self.spin_free_blocks(qa, qb)
        shape = blocks[0].shape[:-2]
        v = np.zeros((*shape, self.n_states, self.n_states), dtype=complex)
        for b, start, stop in self.spans:
            v[..., start:stop, start:stop] = blocks[b]
        if self.soc is not None:
            v = v + self.soc
        return v

    def mch_vectors(self, qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
        """Real (..., N_exp, N_exp) T(Q): the spin-free eigenvectors (ascending) on every block."""
        per_block = [np.linalg.eigh(v)[1] for v in self.spin_free_blocks(qa, qb)]
        shape = per_block[0].shape[:-2]
        t = np.zeros((*shape, self.n_states, self.n_states))
        for b, start, stop in self.spans:
            t[..., start:stop, start:stop] = per_block[b]
        return t


def two_mode_lvc(
    params: LVCParameters,
    modes: tuple[int, int] = MODES,
    *,
    vibronic_coupling: bool = True,
    spin_orbit: bool = True,
) -> TwoModeLVC:
    """The two-mode Hamiltonian of ``params`` on the 0-based ``modes``.

    ``vibronic_coupling = False`` sets every lambda to 0 and ``spin_orbit = False`` sets
    Sigma = 0 (the decoupled self-test). ``ValueError`` if a kappa or lambda acts on another
    mode, or if a chosen mode has omega <= 0.
    """
    counts = check_counts(params.counts)
    omega_all = np.asarray(params.v0.frequencies_Eh, dtype=float)
    omega = omega_all[list(modes)]
    if not np.all(omega > 0):
        raise ValueError(f"modes {modes} must have omega > 0; got {omega.tolist()}")
    others = sorted(
        {
            key[-1]
            for key, value in (*params.kappa.items(), *params.lam.items())
            if value != 0.0 and key[-1] not in modes
        }
    )
    if others:
        raise ValueError(f"the template couples modes {others} besides {modes}: not separable")
    states = expanded_states(counts)
    n1 = counts[0]
    n3 = counts[2] if len(counts) == 3 else 0
    mults = [(m, n) for m, n in ((1, n1), (3, n3)) if n > 0]
    eps_l, kap_l, lam_l = [], [], []
    for mult, n_m in mults:
        eps = np.zeros(n_m)
        kap = np.zeros((n_m, 2))
        lam = np.zeros((2, n_m, n_m))
        for (m, s), value in params.epsilon.items():
            if m == mult:
                eps[s] = value
        for (m, s, i), value in params.kappa.items():
            if m == mult and i in modes:
                kap[s, modes.index(i)] = value
        if vibronic_coupling:
            for (m, s, t, i), value in params.lam.items():
                if m == mult and i in modes:
                    lam[modes.index(i), s, t] = lam[modes.index(i), t, s] = value
        eps_l.append(eps)
        kap_l.append(kap)
        lam_l.append(lam)
    spans = []
    start = 0
    for b, (mult, n_m) in enumerate(mults):
        for _ in range(1 if mult == 1 else 3):
            spans.append((b, start, start + n_m))
            start += n_m
    soc = None
    if spin_orbit and params.soc is not None:
        s = np.asarray(params.soc, dtype=complex)
        soc = 0.5 * (s + s.conj().T)
    free: list[str] = []
    for st in states:
        if spin_free_label(st) not in free:
            free.append(spin_free_label(st))
    matrix = np.zeros((len(states), len(free)))
    for i, st in enumerate(states):
        matrix[i, free.index(spin_free_label(st))] = 1.0
    return TwoModeLVC(
        omega=omega,
        labels_expanded=tuple(st.label() for st in states),
        labels_spin_free=tuple(free),
        spin_free_matrix=matrix,
        spans=tuple(spans),
        epsilon=tuple(eps_l),
        kappa=tuple(kap_l),
        lam=tuple(lam_l),
        soc=soc,
    )


def propagate(
    model: TwoModeLVC,
    *,
    t_final: float,
    dt: float = 0.5,
    record_every: int = 40,
    n_grid: tuple[int, int] = GRID["n_grid"],
    q_min: tuple[float, float] = GRID["q_min"],
    q_max: tuple[float, float] = GRID["q_max"],
    initial_state: int = 1,
    edge_tolerance: float = 1e-8,
    projection_refine: int = PROJECTION_REFINE,
) -> dict:
    """Propagate the V0 ground state on diabatic ``initial_state`` to ``t_final``.

    The MCH projection is made on a grid ``projection_refine`` times finer per axis, onto which
    psi is Fourier-interpolated (exact for the band-limited grid wavefunction; 1 = no
    refinement).

    Records every ``record_every`` steps (and at t = 0) and returns NumPy arrays: ``times``
    (n_out,), ``norm``, ``energy``, ``diabatic`` (n_out, N_exp), ``mch`` (n_out, N_exp),
    ``mch_spin_free`` (n_out, n_sf); and ``edge_norm`` (the largest density seen in the outer
    region), ``dq`` (2,), ``steps``. ``t_final / dt`` must be a multiple of ``record_every``.
    """
    n_steps = round(t_final / dt)
    if abs(n_steps * dt - t_final) > 1e-9 * max(1.0, t_final) or n_steps % record_every:
        raise ValueError("t_final / dt must be an integer multiple of record_every")
    axes = [
        np.linspace(lo, hi, n, endpoint=False)
        for lo, hi, n in zip(q_min, q_max, n_grid, strict=True)
    ]
    dq = np.array([a[1] - a[0] for a in axes])
    weight = float(dq[0] * dq[1])
    qa, qb = np.meshgrid(*axes, indexing="ij")
    momenta = [2.0 * math.pi * np.fft.fftfreq(n, d=d) for n, d in zip(n_grid, dq, strict=True)]
    pa, pb = np.meshgrid(*momenta, indexing="ij")
    t_kin = 0.5 * (model.omega[0] * pa**2 + model.omega[1] * pb**2)
    kinetic = np.exp(-1j * dt * t_kin)

    v = model.diabatic(qa, qb)  # (Na, Nb, n, n)
    e, w = np.linalg.eigh(v)
    w_dag = np.conj(np.swapaxes(w, -1, -2))
    # state-first layouts (n, n, Na, Nb) for the pointwise products
    half = np.moveaxis((w * np.exp(-0.5j * dt * e)[..., None, :]) @ w_dag, (-2, -1), (0, 1))
    full = np.moveaxis((w * np.exp(-1j * dt * e)[..., None, :]) @ w_dag, (-2, -1), (0, 1))
    v_sf = np.ascontiguousarray(np.moveaxis(v, (-2, -1), (0, 1)))
    refine = int(projection_refine)
    if refine < 1:
        raise ValueError("projection_refine must be an integer >= 1")
    fine_axes = [
        lo + (hi - lo) * np.arange(refine * m) / (refine * m)
        for lo, hi, m in zip(q_min, q_max, n_grid, strict=True)
    ]
    fa, fb = np.meshgrid(*fine_axes, indexing="ij")
    t_mch = np.ascontiguousarray(np.moveaxis(model.mch_vectors(fa, fb), (-2, -1), (0, 1)))
    pad_a, pad_b = (slice((refine - 1) * m // 2, (refine - 1) * m // 2 + m) for m in n_grid)
    half, full = np.ascontiguousarray(half), np.ascontiguousarray(full)

    n = model.n_states
    psi = np.zeros((n, *qa.shape), dtype=complex)
    psi[initial_state] = np.exp(-0.5 * (qa**2 + qb**2)) / math.sqrt(math.pi)
    psi /= math.sqrt(float(np.sum(np.abs(psi) ** 2)) * weight)

    edge = np.zeros(qa.shape, dtype=bool)
    for axis, size in enumerate(n_grid):
        k = max(1, int(EDGE_FRACTION * size))
        index = [slice(None), slice(None)]
        index[axis] = np.r_[0:k, size - k : size]
        edge[tuple(index)] = True
    n_cells = qa.size

    def apply(m: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Pointwise matrix times wavefunction: m (n, n, Na, Nb), x (n, Na, Nb)."""
        out = m[:, 0] * x[0]
        for b in range(1, n):
            out += m[:, b] * x[b]
        return out

    def record() -> tuple[float, float, np.ndarray, np.ndarray, float]:
        rho = np.abs(psi) ** 2
        norm = float(rho.sum()) * weight
        phi = scipy.fft.fft2(psi, axes=(1, 2), workers=FFT_WORKERS)
        kin = float(np.sum(t_kin * np.abs(phi) ** 2)) * weight / n_cells
        pot = float(np.real(np.sum(np.conj(psi) * apply(v_sf, psi)))) * weight
        if refine > 1:  # Fourier interpolation of psi onto the projection grid
            padded = np.zeros((n, refine * n_grid[0], refine * n_grid[1]), dtype=complex)
            padded[:, pad_a, pad_b] = np.fft.fftshift(phi, axes=(1, 2))
            psi_fine = scipy.fft.ifft2(
                np.fft.ifftshift(padded, axes=(1, 2)), axes=(1, 2), workers=FFT_WORKERS
            ) * (refine * refine)
        else:
            psi_fine = psi
        c = np.einsum("abxy,axy->bxy", t_mch, psi_fine)  # c_b = sum_a T_ab psi_a
        mch = np.sum(np.abs(c) ** 2, axis=(1, 2)) * weight / (refine * refine)
        return (
            norm,
            kin + pot,
            rho.sum(axis=(1, 2)) * weight,
            mch,
            float(rho[:, edge].sum()) * weight,
        )

    rows = [record()]
    edge_norm = rows[0][4]
    # Strang steps; between two steps without a record the two half factors are merged
    psi = apply(half, psi)
    for step in range(1, n_steps + 1):
        psi = scipy.fft.ifft2(
            kinetic * scipy.fft.fft2(psi, axes=(1, 2), workers=FFT_WORKERS),
            axes=(1, 2),
            workers=FFT_WORKERS,
        )
        if step % record_every:
            psi = apply(full, psi)
            continue
        psi = apply(half, psi)
        rows.append(record())
        edge_norm = max(edge_norm, rows[-1][4])
        if edge_norm > edge_tolerance:
            raise RuntimeError(
                f"the wavepacket reached the grid edge (density {edge_norm:.2e} in the outer "
                f"{EDGE_FRACTION:.0%} of an axis) at t = {step * dt:g} au; widen the grid"
            )
        if step < n_steps:
            psi = apply(half, psi)
    mch = np.array([r[3] for r in rows])
    return {
        "times": np.arange(len(rows)) * record_every * dt,
        "norm": np.array([r[0] for r in rows]),
        "energy": np.array([r[1] for r in rows]),
        "diabatic": np.array([r[2] for r in rows]),
        "mch": mch,
        "mch_spin_free": mch @ model.spin_free_matrix,
        "edge_norm": edge_norm,
        "dq": dq,
        "steps": n_steps,
    }
