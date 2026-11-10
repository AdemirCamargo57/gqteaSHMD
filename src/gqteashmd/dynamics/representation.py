"""Diagonal (propagation) representation of the MCH Hamiltonian (P7a-2 method reference 4-6).

The diagonal states are the eigenvectors of the complex Hermitian MCH Hamiltonian H^MCH,
H^MCH U = U diag(E^diag), ordered by energy (ruling R1). Their phases and, inside degenerate
blocks, their subspace basis are tracked against the previous U by a polar factor (ruling R2).
``U = None`` means U = I in MCH order: it is returned exactly when H^MCH has no non-zero
off-diagonal element and there is no previous U, so the adiabatic models of P7a-1 never
diagonalise and keep the MCH order.

Gradients: the real symmetric MCH tensor G^MCH (g_alpha on the diagonal, (E_beta - E_alpha)
K_alpha,beta off it, with the spin-free energies E, ruling R3) is transformed to G^diag =
U^dagger G^MCH U per component; g^diag_a = Re G^diag_aa and K^diag_ab = G^diag_ab / (E^diag_b -
E^diag_a). Energy-based selection (``eselect``, ruling R5) masks one snapshot.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from gqteashmd.dynamics.options import GRADIENT_TRANSFORMATIONS
from gqteashmd.dynamics.records import ElectronicSnapshot

SOC_MESSAGE = (
    "spin-orbit coupling / non-diagonal MCH Hamiltonians need P7a-2 (diagonal representation)"
)
DEGENERACY_BLOCK_TOLERANCE = 1e-10  # Eh: consecutive eigenvalues this close form one block
NAC_DEGENERACY_TOLERANCE = 1e-12  # Eh: |E_b - E_a| below this gives a zero diagonal NAC
SINGULAR_OVERLAP_TOLERANCE = 1e-14  # a 1 x 1 block with |M| below this is left unchanged


def diagonal_basis(
    hamiltonian: np.ndarray, previous_u: np.ndarray | None, *, first_step: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """(E_diag, U) per Method reference 4. U is None iff H has no non-zero off-diagonal element
    and previous_u is None. first_step=True (no previous U at all) uses the largest-component
    phase rule; otherwise a None previous_u means the identity reference.

    Otherwise U comes from ``eigh`` (ascending eigenvalues). Each block B of consecutive
    eigenvalues with E_{b+1} - E_b <= ``DEGENERACY_BLOCK_TOLERANCE`` is rotated by the polar
    factor of M = U[:, B]^dagger U_prev[:, B] = X S Y^H: U[:, B] <- U[:, B] X Y^H, which makes
    U[:, B]^dagger U_prev[:, B] Hermitian positive semi-definite (a 1 x 1 block: a real positive
    overlap; left unchanged when |M| < 1e-14). With ``first_step`` the largest-|component| of
    each column (lowest index on ties) is made real and positive instead. E_diag is
    ``diagonal_energies(H, U)``. ``first_step`` together with a ``previous_u`` is a
    ``ValueError``.
    """
    h = np.asarray(hamiltonian, dtype=complex)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError(f"the MCH Hamiltonian must be a square matrix; got shape {h.shape}")
    n = h.shape[0]
    if first_step and previous_u is not None:
        raise ValueError("first_step=True means there is no previous U; got a previous_u")
    if previous_u is None and np.count_nonzero(h) == np.count_nonzero(np.diagonal(h)):
        return diagonal_energies(h, None), None
    eigenvalues, u = np.linalg.eigh(h)
    if first_step:
        rows = np.argmax(np.abs(u), axis=0)
        top = u[rows, np.arange(n)]
        u = u * (top.conj() / np.abs(top))[None, :]
    else:
        ref = np.eye(n, dtype=complex) if previous_u is None else np.asarray(previous_u, complex)
        if ref.shape != (n, n):
            raise ValueError(f"previous_u must have shape {(n, n)}; got {ref.shape}")
        for start, stop in _degenerate_blocks(eigenvalues):
            block = u[:, start:stop]
            m = block.conj().T @ ref[:, start:stop]
            if stop - start == 1 and abs(m[0, 0]) < SINGULAR_OVERLAP_TOLERANCE:
                continue
            x, _, yh = np.linalg.svd(m)
            u[:, start:stop] = block @ (x @ yh)
    return diagonal_energies(h, u), u


def _degenerate_blocks(energies: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) ranges of consecutive ascending energies chained by the block tolerance."""
    blocks = []
    start = 0
    for b in range(1, energies.size):
        if energies[b] - energies[b - 1] > DEGENERACY_BLOCK_TOLERANCE:
            blocks.append((start, b))
            start = b
    blocks.append((start, energies.size))
    return blocks


def diagonal_energies(hamiltonian: np.ndarray, u: np.ndarray | None) -> np.ndarray:
    """Re diag(U^dagger H U), or Re diag(H) when u is None; the one definition of E^diag that
    diagonal_basis returns and the driver recomputes from a stored state (item 11, step 1)."""
    h = np.asarray(hamiltonian)
    if u is None:
        return np.diagonal(h).real.copy()
    return np.real(np.einsum("ia,ij,ja->a", u.conj(), h, u))


def gradient_tensor(snapshot: ElectronicSnapshot, transformation: str = "ngt") -> np.ndarray:
    """G^MCH, real (n, n, f), item 5.

    G_aa = g_a and G_ab = (E_b - E_a) K_ab with the spin-free energies E (ruling R3); with
    ``transformation = "none"`` (SHARC's ``nogradcorrect``) the off-diagonal terms are 0.
    """
    if transformation not in GRADIENT_TRANSFORMATIONS:
        raise ValueError(
            f"gradient_transformation must be one of {', '.join(GRADIENT_TRANSFORMATIONS)}; "
            f"got {transformation!r}"
        )
    e = np.asarray(snapshot.energies, dtype=float)
    g = np.asarray(snapshot.gradients, dtype=float)
    n = e.size
    if transformation == "none":
        tensor = np.zeros((n, n, g.shape[1]))
    else:
        gaps = e[None, :] - e[:, None]  # [a, b] = E_b - E_a
        tensor = gaps[:, :, None] * np.asarray(snapshot.nacs, dtype=float)
    index = np.arange(n)
    tensor[index, index, :] = g
    return tensor


def diagonal_tensor(g_mch: np.ndarray, u: np.ndarray | None) -> np.ndarray:
    """G^diag = U^dagger G^MCH U per component; returns g_mch itself when u is None."""
    if u is None:
        return g_mch
    per_component = np.moveaxis(g_mch, 2, 0)  # (f, n, n)
    return np.moveaxis(u.conj().T @ per_component @ u, 0, 2)


def diagonal_gradient(g_diag: np.ndarray, state: int) -> np.ndarray:
    """g^diag_a = Re G^diag_aa, shape (f,) (a copy)."""
    return np.array(np.real(g_diag[state, state]), copy=True)


def diagonal_nac(g_diag: np.ndarray, e_diag: np.ndarray, a: int, b: int) -> np.ndarray:
    """K^diag_ab = G^diag_ab / (E^diag_b - E^diag_a); the zero vector when the gap is below
    ``NAC_DEGENERACY_TOLERANCE`` (also for a == b)."""
    gap = float(e_diag[b] - e_diag[a])
    if abs(gap) < NAC_DEGENERACY_TOLERANCE:
        return np.zeros_like(g_diag[a, b])
    return np.asarray(g_diag[a, b] / gap)


def selection_mask(
    energies_mch: np.ndarray, e_reference: float, eselect_Eh: float | None
) -> np.ndarray | None:
    """Bool (n,): |E_ref - E_alpha| < eselect_Eh (strict); None (keep everything) for None."""
    if eselect_Eh is None:
        return None
    return np.abs(e_reference - np.asarray(energies_mch, dtype=float)) < eselect_Eh


def masked_snapshot(snapshot: ElectronicSnapshot, mask: np.ndarray | None) -> ElectronicSnapshot:
    """Gradients of deselected states and NACs with a deselected index set to 0; the snapshot
    itself when mask is None or all True (no copy)."""
    if mask is None:
        return snapshot
    keep = np.asarray(mask, dtype=bool)
    if np.all(keep):
        return snapshot
    drop = ~keep
    gradients = np.array(snapshot.gradients, copy=True)
    gradients[drop] = 0.0
    nacs = np.array(snapshot.nacs, copy=True)
    nacs[drop, :, :] = 0.0
    nacs[:, drop, :] = 0.0
    return replace(snapshot, gradients=gradients, nacs=nacs)


def mch_state(u: np.ndarray | None, active: int) -> int:
    """argmax_alpha |U_alpha,active| (lowest index on ties); ``active`` when u is None."""
    if u is None:
        return int(active)
    return int(np.argmax(np.abs(u[:, active])))


def mch_coefficients(u: np.ndarray | None, c_diag: np.ndarray) -> np.ndarray:
    """c^MCH = U c^diag, or c_diag itself when u is None."""
    if u is None:
        return c_diag
    return np.asarray(u @ c_diag)
