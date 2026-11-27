"""The spin-expanded Hamiltonian of `soc.assemble_hamiltonian` (constraints, Assembly; ledger
ruling: implemented in Task 5): plain NumPy, no PySCF."""

import numpy as np
import pytest

from gqteashmd.dynamics.spin import expanded_states
from gqteashmd.electronic import soc
from gqteashmd.errors import ConfigError

N_S, N_T = 3, 2


def _blocks(seed: int = 7) -> soc.SocBlocks:
    """Random S-T elements and a Hermitian T-T block (the 3 x 3 blocks of a rank-1 tensor need
    not be modelled here: assembly only places and phases them)."""
    rng = np.random.default_rng(seed)
    st = rng.standard_normal((N_S, N_T, 3)) + 1j * rng.standard_normal((N_S, N_T, 3))
    flat = rng.standard_normal((3 * N_T, 3 * N_T)) + 1j * rng.standard_normal((3 * N_T, 3 * N_T))
    flat = flat + flat.conj().T
    flat[np.diag_indices(3 * N_T)] = 0.0  # no SOC on the diagonal (real spin-free states)
    # flat is ordered (J, M); tt[J, J', M, M'] = flat[(J, M), (J', M')]
    tt = flat.reshape(N_T, 3, N_T, 3).transpose(0, 2, 1, 3)
    return soc.SocBlocks(st, tt)


def _index(j: int, m: int) -> int:
    """Position of T_(j+1) with M index m (0, 1, 2 = Ms -1, 0, +1) in the canonical order."""
    return N_S + m * N_T + j


ENERGIES_S = np.array([-1.0, -0.7, -0.6])
ENERGIES_T = np.array([-0.8, -0.65])


def test_the_assembled_matrix_is_hermitian_with_the_energies_on_the_diagonal():
    blocks = _blocks()
    h = soc.assemble_hamiltonian(ENERGIES_S, ENERGIES_T, blocks, np.ones(N_S), np.ones(N_T))
    assert h.shape == (N_S + 3 * N_T, N_S + 3 * N_T) and h.dtype == complex
    assert np.max(np.abs(h - h.conj().T)) == 0.0  # T-S is the conjugate of S-T, T-T Hermitian
    assert np.all(h[:N_S, :N_S] == np.diag(ENERGIES_S))  # S-S off-diagonal exactly zero
    assert np.all(np.diag(h).real == np.concatenate([ENERGIES_S, np.tile(ENERGIES_T, 3)]))
    assert np.all(np.diag(h).imag == 0.0)


def test_the_blocks_are_placed_in_the_canonical_order_with_the_phases():
    blocks = _blocks()
    p_s = np.array([1.0, -1.0, 1.0])
    p_t = np.array([-1.0, 1.0])
    h = soc.assemble_hamiltonian(ENERGIES_S, ENERGIES_T, blocks, p_s, p_t)
    for i in range(N_S):
        for j in range(N_T):
            for m in range(3):
                expected = p_s[i] * p_t[j] * blocks.st[i, j, m]
                assert h[i, _index(j, m)] == expected
                assert h[_index(j, m), i] == np.conj(expected)
    for j in range(N_T):
        for jb in range(N_T):
            for m in range(3):
                for mb in range(3):
                    if (j, m) == (jb, mb):
                        continue
                    expected = p_t[j] * p_t[jb] * blocks.tt[j, jb, m, mb]
                    assert h[_index(j, m), _index(jb, mb)] == expected


def test_a_non_hermitian_triplet_block_is_an_internal_error():
    blocks = _blocks()
    tt = blocks.tt.copy()
    tt[0, 1, 2, 2] += 1e-9  # Eh, above the 1e-12 bound
    with pytest.raises(AssertionError, match="not Hermitian"):
        soc.assemble_hamiltonian(
            ENERGIES_S, ENERGIES_T, soc.SocBlocks(blocks.st, tt), np.ones(N_S), np.ones(N_T)
        )


def test_a_spin_orbit_element_on_the_diagonal_is_an_internal_error():
    blocks = _blocks()
    tt = blocks.tt.copy()
    tt[1, 1, 0, 0] = 1e-9j  # Eh: an imaginary diagonal is not Hermitian
    with pytest.raises(AssertionError, match="diagonal"):
        soc.assemble_hamiltonian(
            ENERGIES_S, ENERGIES_T, soc.SocBlocks(blocks.st, tt), np.ones(N_S), np.ones(N_T)
        )


def test_inconsistent_shapes_are_refused():
    blocks = _blocks()
    with pytest.raises(ValueError, match="shape"):
        soc.assemble_hamiltonian(ENERGIES_S[:2], ENERGIES_T, blocks, np.ones(2), np.ones(N_T))
    with pytest.raises(ValueError, match="phases"):
        soc.assemble_hamiltonian(ENERGIES_S, ENERGIES_T, blocks, np.ones(N_S), np.ones(1))


def test_the_labels_are_those_of_the_dynamics_spin_layout():
    assert soc.hamiltonian_labels(N_S, N_T) == tuple(
        s.label() for s in expanded_states((N_S, 0, N_T))
    )
    assert soc.hamiltonian_labels(2, 1) == ("S0", "S1", "T1(-1)", "T1(0)", "T1(+1)")


def test_the_effective_charge_table_check_names_the_missing_elements():
    soc.require_effective_charges(["C", "O", "H", "H", "S"])
    with pytest.raises(ConfigError, match="for N, Br .*use soc_operator = 'somf'"):
        soc.require_effective_charges(["C", "N", "Br", "N"])
