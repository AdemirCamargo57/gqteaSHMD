"""Exact two-state wavepacket scattering on a 1D diabatic model (split-operator FFT; test helper).

Our own reference solver for the Tully models (Tully 1990), used to validate the surface-hopping
core. Atomic units, hbar = 1.

- psi(x) = (psi_1, psi_2) lives on a uniform periodic grid in the **diabatic** basis of
  ``model.diabatic`` (the same V as the dynamics core uses).
- One step is the symmetric (Strang) splitting exp(-iV dt/2) exp(-iT dt) exp(-iV dt/2), with
  T = p^2/2m applied in momentum space and the 2 x 2 factor exp(-iV dt/2) at each grid point
  built from the eigendecomposition V = W diag(E) W^T.
- The initial packet is psi(x) ~ exp(-(x - x0)^2/sigma^2 + ikx) times the local lower
  **adiabatic** eigenvector W[:, 0](x) (signs made continuous along the grid), normalised on
  the grid. This is the packet whose Wigner function the FSSH sampler draws from.
- **Outgoing flux is absorbed and counted.** The outer ``absorber_width`` of the grid on each
  side carries a quadratic absorbing potential, applied after every step as the factor
  exp(-W(x) dt). The density it removes is projected onto the local adiabatic states (the
  absorber lies far outside the coupling region) and added to the transmitted (x > 0) or
  reflected (x < 0) count of that state. Without it the slowest components (near a channel
  threshold) would force a grid hundreds of bohr wide. Because the absorption is a pointwise
  factor, ``norm`` (grid + absorbed) is conserved exactly by the bookkeeping, and checks that
  the propagation is unitary.
- The run ends once the packet has had time to reach the centre (t >= |x0| m/k) and the norm
  inside |x| < x_int has dropped below ``interaction_tolerance`` (1e-4). psi on the grid is then
  projected onto the local adiabatic eigenvectors at every grid point, and |<phi_j(x)|psi(x)>|^2
  is integrated over x > 0 (transmitted) and x < 0 (reflected) and added to the absorbed counts.
- Guard: the density in the outermost 1 % of the grid (beyond which the periodic grid would wrap
  the packet around) must stay below ``edge_tolerance``; else ``RuntimeError``.
"""

from __future__ import annotations

import math

import numpy as np

EDGE_FRACTION = 0.01
BRANCHES = ("transmitted_lower", "transmitted_upper", "reflected_lower", "reflected_upper")


def adiabatic_frames(model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """V (N, 2, 2), ascending adiabatic energies E (N, 2) and eigenvectors W (N, 2, 2).

    W[i, :, j] is adiabatic state j at x[i]; its sign is chosen continuous along the grid.
    """
    v = np.array([model.diabatic(np.array([xi]))[0] for xi in x])
    energies, w = np.linalg.eigh(v)
    for i in range(1, len(x)):
        signs = np.sign(np.einsum("aj,aj->j", w[i], w[i - 1]))
        signs[signs == 0] = 1.0
        w[i] *= signs
    return v, energies, w


def _apply(m: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Pointwise 2 x 2 matrix times the two-component wavefunction; m (N, 2, 2), psi (2, N)."""
    return np.stack(
        (m[:, 0, 0] * psi[0] + m[:, 0, 1] * psi[1], m[:, 1, 0] * psi[0] + m[:, 1, 1] * psi[1])
    )


def exact_branching(
    model,
    *,
    mass: float,
    x0: float,
    k: float,
    sigma: float,
    x_min: float,
    x_max: float,
    n_grid: int,
    dt: float,
    t_final: float,
    x_int: float = 10.0,
    absorber_width: float = 20.0,
    absorber_strength: float | None = None,
    interaction_tolerance: float = 1e-4,
    edge_tolerance: float = 1e-8,
    check_every: int = 20,
) -> dict:
    """Exact branching probabilities for a packet starting on the lower adiabatic state.

    Returns the four adiabatic branches (``transmitted_lower``, ``transmitted_upper``,
    ``reflected_lower``, ``reflected_upper``), ``norm`` (grid + absorbed, at the end),
    ``absorbed`` (the absorbed part of it), ``interaction_norm`` (left inside |x| < x_int),
    ``edge_norm`` (largest density seen in the outermost grid region), ``t_end``, ``steps``,
    ``dx`` and ``converged`` (False if ``t_final`` was reached first).

    ``absorber_strength`` (Eh) defaults to 60 v_fast / absorber_width, with v_fast the speed of
    the fastest relevant component (k + 8 s_p, plus the largest possible energy gain), so that
    its density is damped by about exp(-40) on the way through the absorber.
    """
    x = np.linspace(x_min, x_max, n_grid, endpoint=False)
    dx = x[1] - x[0]
    p = 2.0 * math.pi * np.fft.fftfreq(n_grid, d=dx)
    _, energies, w = adiabatic_frames(model, x)
    e_start = float(np.interp(x0, x, energies[:, 0]))
    p_fast = math.sqrt((abs(k) + 8.0 / sigma) ** 2 + 2.0 * mass * (e_start - energies.min()))
    if p_fast > 0.8 * math.pi / dx:
        raise ValueError(f"grid too coarse: p_max = {math.pi / dx:.1f} for p_fast = {p_fast:.1f}")
    if absorber_strength is None:
        absorber_strength = 60.0 * (p_fast / mass) / absorber_width
    depth = np.clip(
        np.maximum(x_min + absorber_width - x, x - (x_max - absorber_width)) / absorber_width,
        0.0,
        None,
    )
    absorbing = np.flatnonzero(depth > 0)
    if np.any(np.abs(x[absorbing]) < x_int + 5.0) or abs(x0) + 6 * sigma > min(-x_min, x_max) - (
        absorber_width
    ):
        raise ValueError("the absorber overlaps the interaction region or the initial packet")
    damping = np.exp(-absorber_strength * depth[absorbing] ** 2 * dt)
    removed = 1.0 - damping**2
    w_abs = w[absorbing]
    right_abs = x[absorbing] > 0

    half = np.einsum("iaj,ij,ibj->iab", w, np.exp(-0.5j * dt * energies), w)
    kinetic = np.exp(-0.5j * dt * p * p / mass)

    envelope = np.exp(-((x - x0) ** 2) / sigma**2 + 1j * k * x)
    psi = w[:, :, 0].T * envelope  # (2, N): lower adiabatic state in the diabatic basis
    psi /= math.sqrt(float(np.sum(np.abs(psi) ** 2)) * dx)

    absorbed = dict.fromkeys(BRANCHES, 0.0)
    n_edge = max(1, int(EDGE_FRACTION * n_grid))
    inside = np.abs(x) < x_int
    t_min = abs(x0) * mass / abs(k)

    def density() -> np.ndarray:
        return np.sum(np.abs(psi) ** 2, axis=0) * dx

    edge_norm = 0.0
    steps = 0
    converged = False
    max_steps = int(math.ceil(t_final / dt))
    while steps < max_steps:
        for _ in range(check_every):
            psi = _apply(half, psi)
            psi = np.fft.ifft(kinetic * np.fft.fft(psi, axis=1), axis=1)
            psi = _apply(half, psi)
            local = psi[:, absorbing]
            ad = np.abs(np.einsum("iaj,ai->ji", w_abs, local)) ** 2 * removed * dx
            absorbed["transmitted_lower"] += float(ad[0, right_abs].sum())
            absorbed["transmitted_upper"] += float(ad[1, right_abs].sum())
            absorbed["reflected_lower"] += float(ad[0, ~right_abs].sum())
            absorbed["reflected_upper"] += float(ad[1, ~right_abs].sum())
            psi[:, absorbing] = local * damping
        steps += check_every
        rho = density()
        edge_norm = max(edge_norm, float(rho[:n_edge].sum() + rho[-n_edge:].sum()))
        if edge_norm > edge_tolerance:
            raise RuntimeError(
                f"the packet reached the grid edge (density {edge_norm:.2e} in the outer "
                f"{EDGE_FRACTION:.0%}) at t = {steps * dt:.0f}; strengthen or widen the absorber"
            )
        if steps * dt >= t_min and float(rho[inside].sum()) < interaction_tolerance:
            converged = True
            break

    rho = density()
    pops = np.abs(np.einsum("iaj,ai->ji", w, psi)) ** 2 * dx  # (2, N) adiabatic densities
    right = x > 0
    result = {
        "transmitted_lower": absorbed["transmitted_lower"] + float(pops[0, right].sum()),
        "transmitted_upper": absorbed["transmitted_upper"] + float(pops[1, right].sum()),
        "reflected_lower": absorbed["reflected_lower"] + float(pops[0, ~right].sum()),
        "reflected_upper": absorbed["reflected_upper"] + float(pops[1, ~right].sum()),
    }
    total_absorbed = sum(absorbed.values())
    result.update(
        norm=float(rho.sum()) + total_absorbed,
        absorbed=total_absorbed,
        interaction_norm=float(rho[inside].sum()),
        edge_norm=edge_norm,
        t_end=steps * dt,
        steps=steps,
        dx=float(dx),
        converged=converged,
    )
    return result
