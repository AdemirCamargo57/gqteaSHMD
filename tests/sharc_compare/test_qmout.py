import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "qmout", ROOT / "tools" / "sharc_compare" / "qmout.py"
)
qmout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qmout)

SAMPLE = """! 0 Basic information
states 2
nmstates 2
natom 2

! 1 Hamiltonian Matrix (2x2, complex)
2 2
-7.804902443E+001  0.000000000E+000  0.000000000E+000  0.000000000E+000
 0.000000000E+000  0.000000000E+000 -7.767250836E+001  0.000000000E+000

! 2 Dipole Moment Matrices (3x2x2, complex)
2 2
 2.5E-008  0.0E+000 -1.529760771E+000  0.0E+000
-1.529760771E+000  0.0E+000  2.6E-007  0.0E+000
2 2
 1.0E-007  0.0E+000 -5.399139466E-002  0.0E+000
-5.399139466E-002  0.0E+000  1.1E-007  0.0E+000
2 2
 9.8E-009  0.0E+000 -2.380718969E-007  0.0E+000
-2.380718969E-007  0.0E+000  2.2E-008  0.0E+000

! 3 Gradient Vectors (2x2x3, real)
2 3 ! 1 1 0.0
-3.889053964E-003 -1.369480090E-004 -3.715872228E-008
 3.888759510E-003  1.373424173E-004  3.172899634E-008
2 3 ! 1 2 0.0
 1.266703636E-001  4.471024236E-003  2.360564851E-008
-1.266706699E-001 -4.470625729E-003 -1.876612445E-008

! 8 Runtime
 3.0E+000
"""


def test_parse_qm_out():
    data = qmout.parse_qm_out(SAMPLE)
    assert data["hamiltonian"][0][0] == complex(-78.04902443, 0.0)
    assert data["hamiltonian"][1][1].real == -77.67250836
    assert [m[0][1] for m in data["dipole"]] == [-1.529760771, -0.05399139466, -2.380718969e-07]
    assert len(data["gradients"]) == 2 and data["gradients"][1][0][0] == 0.1266703636
