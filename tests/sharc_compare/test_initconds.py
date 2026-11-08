import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "initconds", ROOT / "tools" / "sharc_compare" / "initconds.py"
)
initconds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(initconds)

SAMPLE = """SHARC Initial conditions file, version 4.0
Ninit     2
Natom     3
Repr      None
Temp            0.0000000000
Eref            0.0000000000
Eharm           0.0243789326

Equilibrium
 O   8.0   0.00000000   0.00000000   0.28377702  15.99491500   0.00000000   0.00000000   0.00000000
 H   1.0   0.00000000   1.43256453  -0.91771557   1.00782500   0.00000000   0.00000000   0.00000000
 H   1.0   0.00000000  -1.43256453  -0.91771557   1.00782500   0.00000000   0.00000000   0.00000000


Index     1
Atoms
 O   8.0   0.00000000   0.00369579   0.27985374  15.99491500   0.00000000   0.00003799   0.00023850
 H   1.0   0.00000000   1.59236886  -0.86198590   1.00782500   0.00000000   0.00103808  -0.00169402
 H   1.0   0.00000000  -1.65102377  -0.91117981   1.00782500   0.00000000  -0.00164097  -0.00209121
States
Ekin        0.010970157418 a.u.
Epot_harm   0.015700335920 a.u.
Epot        0.015700335920 a.u.
Etot_harm   0.026670493337 a.u.
Etot        0.026670493337 a.u.


Index     2
Atoms
 O   8.0   0.00000000   0.02069568   0.27020237  15.99491500   0.00000000   0.00033799   0.00019773
 H   1.0   0.00000000   1.16859151  -0.67225788   1.00782500   0.00000000  -0.00432307   0.00021733
 H   1.0   0.00000000  -1.49704705  -0.94773370   1.00782500   0.00000000  -0.00104108  -0.00335552
States
Ekin        0.031031060634 a.u.
Epot_harm   0.050923813657 a.u.
Epot        0.050923813657 a.u.
Etot_harm   0.081954874292 a.u.
Etot        0.081954874292 a.u.
"""


def test_parse_two_samples():
    d = initconds.parse_initconds(SAMPLE)
    assert d["natom"] == 3
    assert d["temperature_K"] == 0.0
    assert d["eharm_hartree"] == pytest.approx(0.0243789326)
    assert d["symbols"] == ["O", "H", "H"]
    np.testing.assert_allclose(d["masses_amu"], [15.994915, 1.007825, 1.007825])
    assert d["equilibrium_bohr"].shape == (3, 3)
    assert d["positions_bohr"].shape == (2, 3, 3)
    assert d["velocities_au"].shape == (2, 3, 3)
    assert d["ekin_hartree"].shape == (2,)
    assert d["velocities_au"][0, 1, 1] == pytest.approx(0.00103808)
    assert d["positions_bohr"][1, 2, 1] == pytest.approx(-1.49704705)
    assert d["ekin_hartree"][1] == pytest.approx(0.031031060634)


def test_wrong_sample_count_is_an_error():
    with pytest.raises(ValueError, match="Ninit = 3"):
        initconds.parse_initconds(SAMPLE.replace("Ninit     2", "Ninit     3"))


def test_missing_header_and_bad_atom_row():
    with pytest.raises(ValueError, match="header is missing"):
        initconds.parse_initconds(SAMPLE.replace("Eharm", "Eh"))
    with pytest.raises(ValueError, match="malformed"):
        initconds.parse_initconds(SAMPLE.replace("0.28377702  15.99491500", "0.28377702"))


def test_file_ending_right_after_an_index_line_is_a_value_error():
    truncated = SAMPLE.split("Index     1")[0] + "Index     1"
    with pytest.raises(ValueError, match="Atoms"):
        initconds.parse_initconds(truncated)


def test_run_wigner_rejects_a_bare_string_as_extra_args(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "run_wigner", ROOT / "tools" / "sharc_compare" / "run_wigner.py"
    )
    run_wigner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_wigner)
    with pytest.raises(TypeError, match="sequence"):
        run_wigner.run_wigner(
            tmp_path, "python", tmp_path / "x.molden", 1, tmp_path / "out", extra_args="-r 7"
        )
