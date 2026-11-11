"""SHARC LVC file formats and the synthetic fixtures."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from gqteashmd.core.units import DALTON_IN_ME
from gqteashmd.dynamics.lvc_format import (
    format_lvc_template,
    format_v0,
    lvc_equal,
    parse_lvc_template,
    read_lvc_template,
    read_v0,
    v0_equal,
)
from gqteashmd.dynamics.spin import n_expanded

DATA = Path(__file__).resolve().parents[1] / "data" / "lvc"
TEMPLATES = ["lvc_singlets.template", "lvc_soc.template", "lvc_twomode.template"]


def _load_fixtures():
    name = "gqteashmd_lvc_make_fixtures"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, DATA / "make_fixtures.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fx = _load_fixtures()


def V0():
    return read_v0((DATA / "V0.txt").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", TEMPLATES)
def test_round_trip_exact(name, tmp_path):
    first = read_lvc_template(DATA / name)
    (tmp_path / "V0.txt").write_text(format_v0(first.v0), encoding="utf-8")
    text = format_lvc_template(first)
    (tmp_path / name).write_text(text, encoding="utf-8")
    second = read_lvc_template(tmp_path / name)
    assert lvc_equal(first, second)
    assert v0_equal(first.v0, second.v0)
    assert format_lvc_template(second) == text  # idempotent


def test_committed_files_are_idempotent():
    for name in TEMPLATES:
        text = (DATA / name).read_text(encoding="utf-8")
        assert format_lvc_template(read_lvc_template(DATA / name)) == text
    v0_text = (DATA / "V0.txt").read_text(encoding="utf-8")
    assert format_v0(read_v0(v0_text)) == v0_text


def test_regeneration_matches_committed_files():
    texts = fx.fixture_texts()
    for name in TEMPLATES:
        assert texts[name] == (DATA / name).read_text(encoding="utf-8")
    # V0.txt holds eigenvectors: compare numerically (last-digit LAPACK differences).
    a = read_v0(texts["V0.txt"])
    b = V0()
    assert a.symbols == b.symbols
    for x, y in [
        (a.coords_bohr, b.coords_bohr),
        (a.masses_amu, b.masses_amu),
        (a.frequencies_Eh, b.frequencies_Eh),
        (a.modes, b.modes),
    ]:
        assert np.max(np.abs(x - y)) < 1e-12


def test_modes_orthogonal_and_six_zero_frequencies():
    v0 = V0()
    assert np.max(np.abs(v0.modes.T @ v0.modes - np.eye(9))) < 1e-12
    assert np.max(np.abs(v0.modes @ v0.modes.T - np.eye(9))) < 1e-12
    assert int(np.sum(v0.frequencies_Eh == 0.0)) == 6
    assert np.all(v0.frequencies_Eh[6:] > 0)
    assert np.all(np.diff(v0.frequencies_Eh) >= 0)
    assert v0.masses_amu.tolist() == [12.0, 15.99491462, 15.99491462]


def test_force_field_hessian_and_modes_consistent():
    h = fx.force_field_hessian()
    assert np.max(np.abs(h - h.T)) < 1e-14
    v0 = V0()
    for k in range(3):  # translations are exact null vectors of the Hessian
        t = np.zeros((3, 3))
        t[:, k] = 1.0
        assert np.max(np.abs(h @ t.ravel())) < 1e-12
    inv = 1.0 / np.sqrt(np.repeat(v0.masses_amu * DALTON_IN_ME, 3))
    mw = inv[:, None] * h * inv[None, :]
    for i in range(6, 9):
        k = v0.modes[:, i]
        assert np.max(np.abs(mw @ k - v0.frequencies_Eh[i] ** 2 * k)) < 1e-12
    for i in range(6):
        assert np.max(np.abs(mw @ v0.modes[:, i])) < 1e-10


def test_wilson_b_matches_finite_differences():
    c = fx.geometry_bohr()

    def internals(x):
        a, b, d = x.reshape(3, 3)
        r1, r2 = np.linalg.norm(b - a), np.linalg.norm(d - a)
        cos_t = (b - a) @ (d - a) / (r1 * r2)
        return np.array([r1, r2, np.arccos(cos_t)])

    b = fx.wilson_b(c)
    h = 1e-6
    for k in range(9):
        dx = np.zeros(9)
        dx[k] = h
        fd = (internals(c.ravel() + dx) - internals(c.ravel() - dx)) / (2 * h)
        assert np.max(np.abs(fd - b[:, k])) < 1e-8


def test_fixture_contents():
    s = read_lvc_template(DATA / "lvc_singlets.template")
    assert s.counts == (3,) and s.soc is None
    assert s.epsilon == {(1, 0): 0.0, (1, 1): 0.16, (1, 2): 0.19}
    assert all(abs(v) <= 0.01 for v in s.kappa.values())
    assert set(s.lam) == {(1, 1, 2, 8), (1, 0, 1, 8)}

    p = read_lvc_template(DATA / "lvc_soc.template")
    assert p.counts == (2, 0, 2) and p.soc.shape == (8, 8)
    assert p.dipoles.shape == (3, 8, 8)
    assert np.array_equal(p.soc, p.soc.conj().T)
    assert n_expanded(p.counts) == 8
    mags = np.abs(p.soc[p.soc != 0])
    assert mags.min() >= 1e-4 and mags.max() <= 6e-4
    assert np.any(p.soc.imag != 0)
    assert not np.any(p.soc[:2, :2]) and not np.any(p.soc[2:, 2:])  # only singlet-triplet
    assert p.soc[0, 2] != 0 and p.soc[1, 2] != 0 and p.soc[1, 3] != 0

    t = read_lvc_template(DATA / "lvc_twomode.template")
    assert t.counts == (2, 0, 1) and t.soc.shape == (5, 5)
    assert t.kappa == {(1, 0, 6): -0.006, (1, 1, 6): 0.009}
    assert t.lam == {(1, 0, 1, 8): 0.004}
    assert np.array_equal(t.soc, t.soc.conj().T)
    assert np.allclose(np.abs(t.soc[1, 2:]), 2e-4)


def test_relative_v0_path_resolved_against_template(tmp_path):
    sub = tmp_path / "model"
    sub.mkdir()
    (sub / "geom.txt").write_text((DATA / "V0.txt").read_text(encoding="utf-8"), encoding="utf-8")
    text = format_lvc_template(read_lvc_template(DATA / "lvc_singlets.template"), "geom.txt")
    (sub / "LVC.template").write_text(text, encoding="utf-8")
    params = read_lvc_template(sub / "LVC.template")
    assert params.v0.modes.shape == (9, 9)


def test_missing_v0_file_error_names_template(tmp_path):
    (tmp_path / "a.template").write_text("nothere.txt\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"a\.template, line 1"):
        read_lvc_template(tmp_path / "a.template")


# ------------------------------------------------------------------ errors


def parse(text):
    return parse_lvc_template(text, V0(), "T.template")


def test_only_first_lines_mandatory():
    p = parse("V0.txt\n2 0 1\n")
    assert p.epsilon == {} and p.soc is None and p.counts == (2, 0, 1)


def test_missing_count_has_line_number():
    with pytest.raises(ValueError, match=r"T\.template, line 4.*integer"):
        parse("V0.txt\n2\nepsilon\nkappa\n0\n")
    with pytest.raises(ValueError, match=r"T\.template, line 4.*end of file"):
        parse("V0.txt\n2\nepsilon\n")


def test_short_soc_row():
    rows = "\n".join(["0 " * 5] * 4 + ["0 0 0"])  # N_exp = 5 for (2, 0, 1)
    with pytest.raises(ValueError, match=r"T\.template, line 8.*row 5.*got 3"):
        parse("V0.txt\n2 0 1\nSOC  R\n" + rows + "\n")


def test_short_v0_row():
    lines = (DATA / "V0.txt").read_text(encoding="utf-8").splitlines()
    lines[-1] = " ".join(lines[-1].split()[:4])
    with pytest.raises(ValueError, match=r"V0\.txt, line 16.*row 9"):
        read_v0("\n".join(lines))


def test_zero_based_index_rejected():
    with pytest.raises(ValueError, match=r"line 5.*1-based"):
        parse("V0.txt\n2\nepsilon\n1\n1 0 0.0\n")
    with pytest.raises(ValueError, match=r"line 5.*1-based"):
        parse("V0.txt\n2\nkappa\n1\n1 1 0 0.01\n")


def test_gamma_with_entries_refused():
    with pytest.raises(ValueError, match=r"line 4.*'gamma'.*not supported"):
        parse("V0.txt\n2\ngamma\n1\n1 1 1 1 0.0\n")
    with pytest.raises(ValueError, match="lambda_soc"):
        parse("V0.txt\n2\nlambda_soc\n3\n")


def test_unknown_keyword():
    with pytest.raises(ValueError, match=r"line 3.*unknown section keyword 'foo'"):
        parse("V0.txt\n2\nfoo\n0\n")


def test_duplicate_kappa_and_lambda():
    with pytest.raises(ValueError, match=r"line 6.*duplicate kappa"):
        parse("V0.txt\n2\nkappa\n2\n1 1 7 0.01\n1 1 7 0.02\n")
    with pytest.raises(ValueError, match=r"line 6.*duplicate lambda"):
        parse("V0.txt\n2\nlambda\n2\n1 1 2 9 0.01\n1 2 1 9 0.02\n")


def test_lambda_normalised_and_multipolar_skipped():
    text = (
        "V0.txt\n2 0 1\nlambda\n1\n1 2 1 9 0.5\n"
        "Multipolar  Density  Fit\n2\n1 1 1 0 0.1 0.2\ngarbage here\n"
        "epsilon\n1\n3 1 0.25\n"
    )
    p = parse(text)
    assert p.lam == {(1, 0, 1, 8): 0.5}
    assert p.epsilon == {(3, 0): 0.25}


def test_doublets_rejected_with_line():
    with pytest.raises(ValueError, match=r"line 2.*[Dd]oublet"):
        parse("V0.txt\n2 1 0\n")
