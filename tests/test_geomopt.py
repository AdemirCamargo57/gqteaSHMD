import numpy as np
import pytest

from gqteashmd.geomopt import optimize_geometry

R0, K = 1.8, 0.4  # bohr, Eh/bohr^2


def _bonds(coords):
    """Harmonic bonds 0-1 and 0-2 plus a weak 1-2 bond (a bent triatomic model)."""
    energy, grad = 0.0, np.zeros_like(coords)
    for (i, j), (r0, k) in {(0, 1): (R0, K), (0, 2): (R0, K), (1, 2): (2.9, 0.05)}.items():
        d = coords[i] - coords[j]
        r = np.linalg.norm(d)
        energy += 0.5 * k * (r - r0) ** 2
        g = k * (r - r0) * d / r
        grad[i] += g
        grad[j] -= g
    return energy, grad


START = np.array([[0.0, 0.0, 0.0], [0.0, 1.6, 0.9], [0.0, -1.5, 1.0]])


def test_converges_to_the_model_minimum(tmp_path):
    result = optimize_geometry(
        ("O", "H", "H"), START, _bonds, max_steps=100, convergence="GAU", work_dir=tmp_path / "geo"
    )
    assert result.converged
    d01 = np.linalg.norm(result.coords_bohr[0] - result.coords_bohr[1])
    d02 = np.linalg.norm(result.coords_bohr[0] - result.coords_bohr[2])
    assert abs(d01 - R0) < 0.02 and abs(d02 - R0) < 0.02
    assert result.energy_hartree < _bonds(START)[0]
    assert result.n_gradients == len(result.path_bohr) == len(result.energies_hartree) >= 2
    assert np.abs(_bonds(result.coords_bohr)[1]).max() < 1e-3


def test_reports_non_convergence(tmp_path):
    result = optimize_geometry(
        ("O", "H", "H"),
        START,
        _bonds,
        max_steps=1,
        convergence="GAU_VERYTIGHT",
        work_dir=tmp_path / "geo",
    )
    assert result.converged is False
    np.testing.assert_allclose(result.coords_bohr, result.path_bohr[-1])


def test_energy_errors_propagate(tmp_path):
    def broken(coords):
        raise RuntimeError("electronic structure exploded")

    with pytest.raises(RuntimeError, match="exploded"):
        optimize_geometry(
            ("O", "H", "H"),
            START,
            broken,
            max_steps=10,
            convergence="GAU",
            work_dir=tmp_path / "geo",
        )


def test_logging_configuration_is_preserved(tmp_path):
    import logging

    root = logging.getLogger()
    marker = logging.NullHandler()
    root.addHandler(marker)
    before, level = list(root.handlers), root.level
    try:
        optimize_geometry(
            ("O", "H", "H"),
            START,
            _bonds,
            max_steps=100,
            convergence="GAU",
            work_dir=tmp_path / "g",
        )
        assert root.handlers == before and root.level == level
    finally:
        root.removeHandler(marker)
