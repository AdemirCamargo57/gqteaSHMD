import numpy as np
import pytest

from gqteashmd.geomopt import OptimizationFailedError, optimize_geometry

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


def test_non_convergence_returns_the_lowest_energy_geometry(tmp_path):
    state = {"n": 0}

    def noisy(coords):
        energy, grad = _bonds(coords)
        state["n"] += 1
        if state["n"] == 1:
            return energy - 1.0, grad  # an artificially low first point that is never beaten
        return energy, grad

    result = optimize_geometry(
        ("O", "H", "H"),
        START,
        noisy,
        max_steps=3,
        convergence="GAU_VERYTIGHT",
        work_dir=tmp_path / "geo",
    )
    assert result.converged is False
    best = int(np.argmin(result.energies_hartree))
    assert best == 0
    np.testing.assert_allclose(result.coords_bohr, result.path_bohr[best])
    assert result.energy_hartree == min(result.energies_hartree)


def test_geometric_failures_are_wrapped_but_energy_errors_are_not(tmp_path, monkeypatch):
    import geometric.optimize

    def fail(*args, **kwargs):
        raise RuntimeError("internal coordinate failure")

    monkeypatch.setattr(geometric.optimize, "run_optimizer", fail)
    with pytest.raises(OptimizationFailedError, match="internal coordinate failure"):
        optimize_geometry(
            ("O", "H", "H"),
            START,
            _bonds,
            max_steps=10,
            convergence="GAU",
            work_dir=tmp_path / "geo",
        )


def test_energy_error_is_not_wrapped_even_when_geometric_raises_something_else(
    tmp_path, monkeypatch
):
    import geometric.optimize

    real = geometric.optimize.run_optimizer

    def swallowing(**kwargs):
        try:
            return real(**kwargs)
        except RuntimeError as exc:
            raise ValueError("geomeTRIC re-raised") from exc

    monkeypatch.setattr(geometric.optimize, "run_optimizer", swallowing)

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
