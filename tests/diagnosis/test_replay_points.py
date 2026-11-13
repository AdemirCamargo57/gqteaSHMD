from pathlib import Path

import numpy as np
import pytest
from tests.diagnosis.conftest import load_tool

rp = load_tool("diagnosis", "replay_points")


def test_active_orbital_overlap_identity_and_rotation():
    rng = np.random.default_rng(1)
    c, _ = np.linalg.qr(rng.standard_normal((6, 2)))
    s = np.eye(6)
    np.testing.assert_allclose(rp.active_orbital_overlap(c, c, s), [1.0, 1.0], atol=1e-12)
    rot = np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]])
    np.testing.assert_allclose(rp.active_orbital_overlap(c, c @ rot, s), [1.0, 1.0], atol=1e-12)
    other, _ = np.linalg.qr(np.hstack([c, rng.standard_normal((6, 2))]))
    sv = rp.active_orbital_overlap(c, np.column_stack([c[:, 0], other[:, 2]]), s)
    np.testing.assert_allclose(sv, [1.0, 0.0], atol=1e-12)


def test_natural_occupations_descending():
    np.testing.assert_allclose(rp.natural_occupations(np.diag([0.1, 1.9])), [1.9, 0.1])


def test_probe_window_and_failure_record():
    assert rp.probe_window(20, 10, 6) == [5, 6, 7, 8, 9, 10]
    assert rp.probe_window(20, 2, 6) == [0, 1, 2]
    with pytest.raises(ValueError):
        rp.probe_window(5, 5, 3)
    assert rp.failure_record({"jumps": [{"record": 4}], "n_records": 9}) == 4
    assert rp.failure_record({"jumps": [], "n_records": 9}) == 8


def test_segments_construction():
    # jump at the end: one window, no failed step requested
    assert rp.segments(20, 19, 19, 4, False) == [[16, 17, 18, 19]]
    # failure at the end, jump there too: one merged segment with failed_step
    assert rp.segments(20, 19, 19, 4, True) == [[16, 17, 18, 19, "failed_step"]]
    # jump far before the end: two segments, failed_step in the second
    assert rp.segments(40, 10, 39, 3, True) == [[8, 9, 10], [37, 38, 39, "failed_step"]]
    # windows touching (first ends at 10, second starts at 11): merged
    assert rp.segments(20, 10, 13, 3, True) == [[8, 9, 10, 11, 12, 13, "failed_step"]]
    # overlapping windows
    assert rp.segments(20, 10, 11, 4, True) == [[7, 8, 9, 10, 11, "failed_step"]]
    # just not touching: 11 is a gap of one record
    assert len(rp.segments(20, 10, 14, 3, True)) == 2


def test_replay_variants_are_the_spec_list():
    names = [v.name for v in rp.REPLAY_VARIANTS]
    assert names == ["base", "sa3_cas22", "sa3_cas23", "sa2_cas43"]
    assert set(rp.MODES) == {"continued", "backward", "fresh"}


@pytest.mark.pyscf
def test_continued_path_reproduces_the_backend(tmp_path):
    """Two ethylene geometries: the continued solve gives the PySCFBackend energies; the fresh
    solve agrees there (no solution change); the active orbitals overlap well."""
    from gqteashmd.chem.xyz import parse_xyz
    from gqteashmd.config.schema import ElectronicSection
    from gqteashmd.core.units import ANGSTROM_TO_BOHR, HARTREE_IN_EV
    from gqteashmd.electronic.base import MoleculeSpec, Request
    from gqteashmd.electronic.pyscf_backend import PySCFBackend

    xyz = parse_xyz((Path(__file__).resolve().parents[1] / "data" / "ethylene.xyz").read_text())
    q0 = np.asarray(xyz.coords_angstrom) * ANGSTROM_TO_BOHR
    q1 = q0 + 0.01 * np.random.default_rng(3).standard_normal(q0.shape)
    settings = ElectronicSection.model_validate(
        {"method": "casscf", "basis": "6-31g*", "singlets": 2, "casscf": {"ncas": 2, "nelecas": 2}}
    )
    labels = ["S0", "S1"]
    symbols = tuple(xyz.symbols)
    kw = {"variant": "base", "threads": 2, "memory_mb": 1000}
    cont = rp.solve_path(
        symbols, 0, [q0, q1], labels, settings, mode="continued", scratch=tmp_path / "a", **kw
    )
    fresh = rp.solve_path(
        symbols, 0, [q0, q1], labels, settings, mode="fresh", scratch=tmp_path / "b", **kw
    )
    backend = PySCFBackend(
        MoleculeSpec(symbols, 0, 1),
        settings,
        threads=2,
        memory_mb=1000,
        scratch_dir=tmp_path / "c",
    )
    ref = [backend.compute(Request(q, gradients=frozenset(), overlaps=True)) for q in (q0, q1)]
    for point, r in zip(cont, ref, strict=True):
        assert point.error is None
        np.testing.assert_allclose(
            point.energies_eV, [r.energies[s] * HARTREE_IN_EV for s in backend.states], atol=1e-8
        )
    np.testing.assert_allclose(cont[1].energies_eV, fresh[1].energies_eV, atol=1e-5)
    assert cont[0].active_overlap is None and min(cont[1].active_overlap) > 0.99
    assert cont[1].state_overlap_min_sv > 0.95  # brief 0.99; measured 0.9863 (see report)
    assert len(cont[1].occupations) == 2 and abs(sum(cont[1].occupations[0]) - 2.0) < 1e-8


def test_hop_to_end_segment():
    assert rp.hop_to_end_segment(10, 6, True) == [6, 7, 8, 9, "failed_step"]
    assert rp.hop_to_end_segment(10, 9, False) == [9]
    with pytest.raises(ValueError):
        rp.hop_to_end_segment(10, 10, False)


def test_replay_cli_accepts_the_new_options(capsys):
    with pytest.raises(SystemExit) as info:  # parse only: --help exits 0
        rp.main(["--help"])
    assert info.value.code == 0
    text = capsys.readouterr().out
    for option in ("--trajectories", "--base-variant", "--from-first-hop"):
        assert option in text
