"""Linear vibronic coupling (LVC) models with triplets and spin-orbit coupling (P7a-2).

Method reference items 2, 3 and 15 (SHARC 4.1 manual section 8.18, eqs. 8.107 ff.; Koeppel,
Domcke & Cederbaum, Adv. Chem. Phys. 57, 59 (1984)). Atomic units throughout; the masses of
``V0.txt`` (amu) are converted with ``DALTON_IN_ME``.

Normal coordinates: Q_i = sqrt(omega_i) sum_j K_{j,i} sqrt(M_j) (r_j - r_ref,j), with K the
orthogonal (3N, 3N) matrix of mass-weighted normal modes (columns = modes) and M_j the mass of
the atom of Cartesian coordinate j. A mode with omega_i = 0 has Q_i = 0 exactly. The reference
potential is V0 = sum_i omega_i Q_i^2 / 2.

Diabatic block of multiplicity m (real symmetric, n_m x n_m, shared by its M_s components):
V_aa = V0 + epsilon_a + sum_i kappa_a,i Q_i, V_ab = sum_i lambda_ab,i Q_i. The diabatic SOC
Sigma is constant, complex Hermitian, in the canonical spin-expanded order (``spin.py``).

MCH data (``evaluate``): V^(m) = W^(m) diag(E^(m)) W^(m)^T per multiplicity (ascending; column
signs aligned to the previous vectors, else largest-|component| positive); T is block diagonal
with W^(m) on every (m, M_s) block; H^MCH = diag(E) + T^T Sigma T; gradients and NACs from
W^T dV W within each block (NACs are 0 between blocks); overlaps S = T_prev^T T.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gqteashmd.core.units import DALTON_IN_ME
from gqteashmd.dynamics.lvc_format import LVCParameters, read_lvc_template
from gqteashmd.dynamics.models import DEGENERACY_THRESHOLD
from gqteashmd.dynamics.records import ElectronicSnapshot
from gqteashmd.dynamics.spin import ExpandedState, check_counts, expanded_states

SOC_HERMITICITY_TOLERANCE = 1e-8  # Eh: max |Sigma - Sigma^dagger| accepted (then symmetrised)


class _Block:
    """Spin-free parameters of one multiplicity: epsilon (n,), kappa (n, 3N), lam (3N, n, n)."""

    def __init__(
        self, multiplicity: int, epsilon: np.ndarray, kappa: np.ndarray, lam: np.ndarray
    ) -> None:
        self.multiplicity = multiplicity
        self.epsilon = epsilon
        self.kappa = kappa
        self.lam = lam


class LVCModel:
    """An ``ElectronicModel`` returning the MCH quantities of an LVC model.

    ``spin_orbit = False`` sets Sigma = 0 (SHARC's ``nospinorbit``). The SOC matrix of the
    parameters must have shape (N_exp, N_exp) and be Hermitian to ``SOC_HERMITICITY_TOLERANCE``
    (``ValueError`` otherwise); it is symmetrised as (Sigma + Sigma^dagger) / 2.
    """

    def __init__(self, parameters: LVCParameters, *, spin_orbit: bool = True) -> None:
        v0 = parameters.v0
        counts = check_counts(parameters.counts)
        n_atoms = len(v0.symbols)
        f = 3 * n_atoms
        coords = np.asarray(v0.coords_bohr, dtype=float)
        masses_amu = np.asarray(v0.masses_amu, dtype=float)
        omega = np.asarray(v0.frequencies_Eh, dtype=float)
        modes = np.asarray(v0.modes, dtype=float)
        if coords.shape != (n_atoms, 3) or masses_amu.shape != (n_atoms,):
            raise ValueError(
                f"LVC reference data: expected coordinates ({n_atoms}, 3) and {n_atoms} masses, "
                f"got {coords.shape} and {masses_amu.shape}"
            )
        if omega.shape != (f,) or modes.shape != (f, f):
            raise ValueError(
                f"LVC reference data: expected {f} frequencies and a ({f}, {f}) mode matrix, "
                f"got {omega.shape} and {modes.shape}"
            )
        if not np.all(np.isfinite(omega)) or np.any(omega < 0):
            raise ValueError(
                f"LVC reference data: the frequencies must be finite and >= 0 Eh, got "
                f"{omega.tolist()}"
            )
        if not np.all(masses_amu > 0):
            raise ValueError(f"LVC reference data: the masses must be > 0, got {masses_amu}")

        self.parameters = parameters
        self.spin_orbit = bool(spin_orbit)
        self.state_counts: tuple[int, ...] = counts
        self.states: tuple[ExpandedState, ...] = expanded_states(counts)
        self.n_states = len(self.states)
        self.n_dof = f
        self.masses = np.repeat(masses_amu * DALTON_IN_ME, 3)
        self.reference_q = coords.reshape(-1).copy()

        self._omega = omega.copy()
        self._active = omega > 0
        self._modes = modes.copy()
        sqrt_m = np.sqrt(self.masses)
        self._sqrt_m = sqrt_m
        # dQ_i/dr_j = sqrt(omega_i) K_{j,i} sqrt(M_j); rows of zero-frequency modes are exactly 0
        self._dq_dr = np.sqrt(omega)[:, None] * modes.T * sqrt_m[None, :]
        self._dq_dr[~self._active] = 0.0
        # Cartesian columns of the vibrational modes: dr_j/dQ_i = K_{j,i} / (sqrt(omega_i) M_j^1/2)
        safe = np.where(self._active, omega, 1.0)
        self._dr_dq = modes / (np.sqrt(safe)[None, :] * sqrt_m[:, None])
        self._dr_dq[:, ~self._active] = 0.0

        self._blocks = tuple(self._block(parameters, mult, n_m, f) for mult, n_m in self._mults())
        # the (m, M_s) blocks in canonical order: (multiplicity block index, start, stop)
        spans: list[tuple[int, int, int]] = []
        start = 0
        for b, block in enumerate(self._blocks):
            n_m = block.epsilon.size
            for _ in range(1 if block.multiplicity == 1 else 3):
                spans.append((b, start, start + n_m))
                start += n_m
        self._spans = tuple(spans)  # singlets, then all triplets with M_s = -1, 0, +1
        soc = self._soc_matrix(parameters)  # validated also when spin_orbit is False
        self._soc = soc if self.spin_orbit else None

    # ------------------------------------------------------------------ construction helpers

    def _mults(self) -> list[tuple[int, int]]:
        n1 = self.state_counts[0]
        n3 = self.state_counts[2] if len(self.state_counts) == 3 else 0
        return [(m, n) for m, n in ((1, n1), (3, n3)) if n > 0]

    @staticmethod
    def _block(params: LVCParameters, mult: int, n_m: int, f: int) -> _Block:
        epsilon = np.zeros(n_m)
        kappa = np.zeros((n_m, f))
        lam = np.zeros((f, n_m, n_m))
        for (m, s), value in params.epsilon.items():
            if m == mult:
                epsilon[s] = value
        for (m, s, i), value in params.kappa.items():
            if m == mult:
                kappa[s, i] = value
        for (m, s, t, i), value in params.lam.items():
            if m == mult:
                lam[i, s, t] = lam[i, t, s] = value
        return _Block(mult, epsilon, kappa, lam)

    def _soc_matrix(self, params: LVCParameters) -> np.ndarray | None:
        if params.soc is None:
            return None
        soc = np.asarray(params.soc, dtype=complex)
        n = self.n_states
        if soc.shape != (n, n):
            raise ValueError(
                f"the SOC matrix of the LVC template must have shape ({n}, {n}) (N_exp for "
                f"the state counts {self.state_counts}), got {soc.shape}"
            )
        deviation = float(np.max(np.abs(soc - soc.conj().T)))
        if deviation > SOC_HERMITICITY_TOLERANCE:
            raise ValueError(
                f"the SOC matrix of the LVC template is not Hermitian: max |SOC - SOC^dagger| = "
                f"{deviation:.3e} Eh exceeds {SOC_HERMITICITY_TOLERANCE:.0e} Eh"
            )
        return np.asarray(0.5 * (soc + soc.conj().T))

    @classmethod
    def from_template(cls, path: Path, *, spin_orbit: bool = True) -> LVCModel:
        """Read ``LVC.template`` (and the ``V0.txt`` it names) and build the model."""
        return cls(read_lvc_template(Path(path)), spin_orbit=spin_orbit)

    # ------------------------------------------------------------------ coordinates

    def _check_q(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape != (self.n_dof,):
            raise ValueError(
                f"q must be a flat array of length n_dof = {self.n_dof}; got shape {q.shape}"
            )
        return q

    def normal_coordinates(self, q: np.ndarray) -> np.ndarray:
        """Q (3N,) of the flat Cartesian q; exactly 0 for the modes with omega = 0."""
        q = self._check_q(q)
        big_q: np.ndarray = self._dq_dr @ (q - self.reference_q)
        big_q[~self._active] = 0.0
        return big_q

    def cartesian(self, Q: np.ndarray) -> np.ndarray:
        """Flat Cartesian r_ref + sum_i K_{.,i} Q_i / (sqrt(omega_i) sqrt(M)) over omega_i > 0;
        the Q of zero-frequency modes are ignored. The inverse of ``normal_coordinates`` on the
        vibrational modes (K orthogonal)."""
        big_q = np.asarray(Q, dtype=float)
        if big_q.shape != (self.n_dof,):
            raise ValueError(f"Q must have length {self.n_dof}; got shape {big_q.shape}")
        big_q = np.where(self._active, big_q, 0.0)
        return np.asarray(self.reference_q + self._dr_dq @ big_q)

    # ------------------------------------------------------------------ Hamiltonians

    def _clean_q(self, Q: np.ndarray) -> np.ndarray:
        big_q = np.asarray(Q, dtype=float)
        if big_q.shape != (self.n_dof,):
            raise ValueError(f"Q must have length {self.n_dof}; got shape {big_q.shape}")
        return np.where(self._active, big_q, 0.0)

    def _spin_free(self, block: _Block, big_q: np.ndarray, v0: float) -> np.ndarray:
        v: np.ndarray = np.einsum("ijk,i->jk", block.lam, big_q)
        v[np.diag_indices_from(v)] = v0 + block.epsilon + block.kappa @ big_q
        return v

    def diabatic_hamiltonian(self, Q: np.ndarray) -> np.ndarray:
        """Complex (N_exp, N_exp): the spin-free blocks on every (m, M_s) block plus Sigma.
        The Q of zero-frequency modes are ignored (taken as 0)."""
        big_q = self._clean_q(Q)
        v0 = 0.5 * float(np.sum(self._omega * big_q**2))
        h = np.zeros((self.n_states, self.n_states), dtype=complex)
        spin_free = [self._spin_free(block, big_q, v0) for block in self._blocks]
        for b, start, stop in self._spans:
            h[start:stop, start:stop] = spin_free[b]
        if self._soc is not None:
            h = h + self._soc
        return h

    # ------------------------------------------------------------------ MCH data

    def evaluate(self, q: np.ndarray, previous: ElectronicSnapshot | None) -> ElectronicSnapshot:
        """MCH snapshot at q (Method reference item 3); ``previous`` fixes the column signs and
        gives the overlaps T_prev^T T."""
        q = self._check_q(q)
        big_q = self.normal_coordinates(q)
        v0 = 0.5 * float(np.sum(self._omega * big_q**2))
        n, f = self.n_states, self.n_dof
        previous_t = None
        if previous is not None and previous.vectors is not None:
            previous_t = np.asarray(previous.vectors, dtype=float)
            if previous_t.shape != (n, n):
                raise ValueError(
                    f"previous.vectors must have shape ({n}, {n}) for this LVC model; got "
                    f"{previous_t.shape}"
                )
        energies = np.zeros(n)
        gradients = np.zeros((n, f))
        nacs = np.zeros((n, n, f))
        t = np.zeros((n, n))
        first_span: dict[int, int] = {}
        for b, start, _ in self._spans:
            first_span.setdefault(b, start)
        per_block = []
        for b, block in enumerate(self._blocks):
            n_m = block.epsilon.size
            e_m, w = np.linalg.eigh(self._spin_free(block, big_q, v0))
            if previous_t is not None:
                s0 = first_span[b]
                w_prev = previous_t[s0 : s0 + n_m, s0 : s0 + n_m]
                signs = np.sign(np.einsum("ik,ik->k", w, w_prev))
                signs[signs == 0] = 1.0
                w = w * signs
            else:
                idx = np.argmax(np.abs(w), axis=0)
                w = w * np.where(w[idx, np.arange(n_m)] < 0, -1.0, 1.0)
            gaps = e_m[None, :] - e_m[:, None]  # E_b - E_a at [a, b]
            off = ~np.eye(n_m, dtype=bool)
            if np.any(np.abs(gaps[off]) < DEGENERACY_THRESHOLD):
                raise ValueError(
                    f"degenerate states of multiplicity {block.multiplicity} at q = "
                    f"{q.tolist()}: the non-adiabatic coupling is undefined"
                )
            # dV/dQ_i = omega_i Q_i 1 + diag(kappa_.,i) + Lambda_i, rotated: W^T dV/dQ_i W
            dv_dq = np.array(block.lam, copy=True)
            diag = np.arange(n_m)
            dv_dq[:, diag, diag] += (self._omega * big_q)[:, None] + block.kappa.T
            g_modes = np.einsum("ja,ijk,kb->iab", w, dv_dq, w)
            g_modes = 0.5 * (g_modes + g_modes.transpose(0, 2, 1))  # exactly symmetric
            g = np.einsum("iab,ir->abr", g_modes, self._dq_dr)  # (n_m, n_m, f)
            grads = np.array(g[diag, diag, :])
            safe = np.where(off, gaps, 1.0)
            block_nacs = np.where(off[:, :, None], g / safe[:, :, None], 0.0)
            per_block.append((e_m, w, grads, block_nacs))
        for b, start, stop in self._spans:
            e_m, w, grads, block_nacs = per_block[b]
            energies[start:stop] = e_m
            gradients[start:stop] = grads
            nacs[start:stop, start:stop] = block_nacs
            t[start:stop, start:stop] = w
        hamiltonian = np.diag(energies).astype(complex)
        if self._soc is not None:
            coupled = t.T @ self._soc @ t
            hamiltonian = hamiltonian + 0.5 * (coupled + coupled.conj().T)
        overlaps = previous_t.T @ t if previous_t is not None else None
        return ElectronicSnapshot(
            energies=energies,
            gradients=gradients,
            nacs=nacs,
            hamiltonian=hamiltonian,
            vectors=t,
            overlaps=overlaps,
        )


def lvc_wigner_samples(
    model: LVCModel, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """n ground-state (T = 0) Wigner samples of the LVC reference potential (item 15).

    For each vibrational mode (omega_i > 0), Q_i and P_i are independent N(0, 1/2). The draws are
    exactly ``z = rng.standard_normal((n, n_vib, 2))`` with Q = sqrt(1/2) z[..., 0] and
    P = sqrt(1/2) z[..., 1]: per sample, the vibrational modes in increasing index, Q then P
    (the same stream as successive scalar ``rng.standard_normal()`` calls in that order). Then
    r = r_ref + sum_i K_{.,i} Q_i / (sqrt(omega_i) sqrt(M)) and
    v = sum_i K_{.,i} sqrt(omega_i) P_i / sqrt(M). Zero-frequency modes have Q = P = 0 and draw
    nothing. Returns q (n, f) in bohr and v (n, f) in bohr per atomic time unit.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError(f"the number of samples n must be a non-negative integer; got {n!r}")
    active = np.flatnonzero(model._active)
    z = rng.standard_normal((int(n), active.size, 2))
    big_q = np.sqrt(0.5) * z[:, :, 0]
    big_p = np.sqrt(0.5) * z[:, :, 1]
    omega = model._omega[active]
    k = model._modes[:, active]  # (f, n_vib)
    sqrt_m = model._sqrt_m
    q = model.reference_q[None, :] + (big_q / np.sqrt(omega)[None, :]) @ k.T / sqrt_m[None, :]
    v = (big_p * np.sqrt(omega)[None, :]) @ k.T / sqrt_m[None, :]
    return q, v
