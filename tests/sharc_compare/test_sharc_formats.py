import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "sharc_compare"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


formats = _load("sharc_formats")
initconds = _load("initconds")
qmout = _load("qmout")

SYMBOLS = ("O", "H", "H")
EQ = np.array(
    [[0.0, 0.0, 0.28377702], [0.0, 1.43256453, -0.91771557], [0.0, -1.43256453, -0.91771557]]
)


def test_initconds_round_trip():
    rng = np.random.default_rng(3)
    pos = EQ + 0.05 * rng.standard_normal((4, 3, 3))
    vel = 1e-3 * rng.standard_normal((4, 3, 3))
    masses = [15.994915, 1.007825, 1.007825]
    text = formats.format_initconds(SYMBOLS, [8.0, 1.0, 1.0], masses, EQ, pos, vel)
    parsed = initconds.parse_initconds(text)
    assert parsed["symbols"] == list(SYMBOLS)
    np.testing.assert_allclose(parsed["masses_amu"], masses, atol=1e-7)
    np.testing.assert_allclose(parsed["equilibrium_bohr"], EQ, atol=1e-8)
    np.testing.assert_allclose(parsed["positions_bohr"], pos, atol=1e-8)
    np.testing.assert_allclose(parsed["velocities_au"], vel, atol=1e-8)
    assert parsed["eharm_hartree"] == 0.0


def test_qm_in_layout():
    text = formats.format_qm_in(
        SYMBOLS, np.array([[0, 0, 0.15], [0, 0.75, -0.48], [0, -0.75, -0.48]]), 3
    )
    lines = text.splitlines()
    assert lines[0] == "3" and lines[1] == "ref"
    assert lines[2].startswith("O ")
    assert lines[-4:] == ["unit angstrom", "states 3", "h", "dm"]


def test_qm_out_layout_and_round_trip():
    energies = [-74.98520102, -74.55839998, -74.49115872]
    dip = np.zeros((3, 3, 3))
    dip[0, 0, 1] = dip[0, 1, 0] = 0.08877556
    dip[1, 0, 2] = dip[1, 2, 0] = -0.9547377
    text = formats.format_qm_out(energies, dip)
    lines = text.splitlines()
    assert lines[0] == "! 0 Basic information"
    assert "states 3 " in lines and "nmstates 3" in lines and "natom 3" in lines
    assert "! 1 Hamiltonian Matrix (3x3, complex)" in lines
    assert "! 2 Dipole Moment Matrices (3x3x3, complex)" in lines
    assert lines.count("3 3") == 4  # one Hamiltonian + three dipole matrices
    assert "-7.498520102E+001" in lines[9]
    gradient_block = "! 3 Gradient Vectors (3x3x3, real)\n" + "".join(
        "0 ! state\n" + "0.0 0.0 0.0\n" * 3 for _ in range(3)
    )
    parsed = qmout.parse_qm_out(text + "\n" + gradient_block)
    for k in range(3):
        assert parsed["hamiltonian"][k][k].real == np.float64(energies[k]).round(9)
    np.testing.assert_allclose(np.array(parsed["dipole"]), dip, atol=1e-9)


EXCITED = """SHARC Initial conditions file, version 4.0   <Excited>
Ninit     2
Natom     3
Repr      MCH
Temp            0.0000000000
Eref          -74.9852010200
Eharm           0.0000000000
States    3

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
001 -74.98520102 -74.98520102 0.00000000 0 -0 0 0 0 0.00000000 0.00000000 False
002 -74.55839998 -74.98520102 0.08877556 0 -0 0 0 0 11.61384795 0.00224244 False
003 -74.49115872 -74.98520102 0.00000000 0 -0 0 0.5 0 13.44000000 0.03 True

Index     2
Atoms
 O   8.0   0.00000000   0.00369579   0.27985374  15.99491500   0.00000000   0.00003799   0.00023850
 H   1.0   0.00000000   1.59236886  -0.86198590   1.00782500   0.00000000   0.00103808  -0.00169402
 H   1.0   0.00000000  -1.65102377  -0.91117981   1.00782500   0.00000000  -0.00164097  -0.00209121
States
001 -74.98520102 -74.98520102 0.00000000 0 -0 0 0 0 0.00000000 0.00000000 False
002 -74.55839998 -74.98520102 0.08877556 0 -0 0 0 0 11.61384795 0.00224244 True
003 -74.49115872 -74.98520102 0.00000000 0 -0 0 0.5 0 13.44000000 0.03 False
"""


def test_parse_initconds_excited():
    parsed = formats.parse_initconds_excited(EXCITED)
    assert parsed["eref_hartree"] == -74.98520102 and parsed["n_states"] == 3
    assert len(parsed["states"]) == 2
    row = parsed["states"][0][1]
    assert row["delta_e_eV"] == 11.61384795 and row["f"] == 0.00224244 and not row["selected"]
    assert row["energy_hartree"] == -74.55839998
    np.testing.assert_allclose(row["dipole"], [0.08877556, 0.0, 0.0])
    assert parsed["states"][0][2]["selected"] and parsed["states"][1][1]["selected"]
    assert parsed["states"][0][2]["dipole"][2] == 0.5
