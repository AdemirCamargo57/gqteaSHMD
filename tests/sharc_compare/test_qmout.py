import importlib.util
from pathlib import Path

import pytest

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

! 5 Nonadiabatic couplings (ddr) (2x2x2x3)
2 3 ! 1 1 0.0 1 1 0.0
 0.0E+000  0.0E+000  0.0E+000
 0.0E+000  0.0E+000  0.0E+000
2 3 ! 1 1 0.0 1 2 0.0
-2.927020103023E-001 -1.033059334102E-002 -1.072930978298E-007
-2.927018076373E-001 -1.033064441499E-002 -7.552369004663E-008
2 3 ! 1 2 0.0 1 1 0.0
 2.927020103023E-001  1.033059334102E-002  1.072930978298E-007
 2.927018076373E-001  1.033064441499E-002  7.552369004663E-008
2 3 ! 1 2 0.0 1 2 0.0
 0.0E+000  0.0E+000  0.0E+000
 0.0E+000  0.0E+000  0.0E+000

! 8 Runtime
 3.0E+000
"""


def test_parse_qm_out():
    data = qmout.parse_qm_out(SAMPLE)
    assert data["hamiltonian"][0][0] == complex(-78.04902443, 0.0)
    assert data["hamiltonian"][1][1].real == -77.67250836
    assert [m[0][1] for m in data["dipole"]] == [-1.529760771, -0.05399139466, -2.380718969e-07]
    assert len(data["gradients"]) == 2 and data["gradients"][1][0][0] == 0.1266703636


def test_truncated_gradient_block_names_the_block():
    cut = SAMPLE[: SAMPLE.index("2 3 ! 1 2 0.0") + len("2 3 ! 1 2 0.0\n")]
    with pytest.raises(ValueError, match=r"QM\.out: malformed 'gradients' block near line \d+"):
        qmout.parse_qm_out(cut)


def test_garbled_number_names_the_block():
    bad = SAMPLE.replace("-7.804902443E+001", "not-a-number", 1)
    with pytest.raises(ValueError, match=r"malformed 'hamiltonian' block near line 8"):
        qmout.parse_qm_out(bad)


def test_missing_block_is_named():
    start = SAMPLE.index("! 2 Dipole")
    end = SAMPLE.index("! 3 Gradient")
    with pytest.raises(ValueError, match="'dipole' block is missing"):
        qmout.parse_qm_out(SAMPLE[:start] + SAMPLE[end:])


def test_parse_nac_block():
    data = qmout.parse_qm_out(SAMPLE)
    assert data["nacs"][0][1][0][0] == -0.2927020103023
    assert data["nacs"][1][0][1][2] == 7.552369004663e-08
    assert data["nacs"][0][0] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_truncated_nac_block_names_the_block():
    marker = "! 5 Nonadiabatic couplings (ddr) (2x2x2x3)\n"
    cut = SAMPLE[: SAMPLE.index(marker) + len(marker)]
    with pytest.raises(ValueError, match=r"malformed 'Nonadiabatic couplings' block near line \d+"):
        qmout.parse_qm_out(cut)


# A QM.out in the layout of manual section 6.34.2 with SOC, as SHARC_LVC.py writes it: states
# 1 0 1 (4 spin components), 1 atom; gradient and NAC headers carry m, s, ms labels; the NAC
# title is spelled "Non-adiabatic"; no dipole block (not requested).
SOC_SAMPLE = (
    """! 0 Basic information
states 1 0 1
nmstates 4
natom 1
npc 0
charges 0 1 0
! 1 Hamiltonian Matrix (4x4, complex)
4 4
 1.0E-003  0.0E+000  2.0E-004  3.0E-004  0.0E+000 -1.0E-004 -2.0E-004  3.0E-004
 2.0E-004 -3.0E-004  1.5E-001  0.0E+000  0.0E+000  0.0E+000  0.0E+000  0.0E+000
 0.0E+000  1.0E-004  0.0E+000  0.0E+000  1.5E-001  0.0E+000  0.0E+000  0.0E+000
-2.0E-004 -3.0E-004  0.0E+000  0.0E+000  0.0E+000  0.0E+000  1.5E-001  0.0E+000

! 3 Gradient Vectors (4x1x3, real)
1 3 ! m1 1 s1 1 ms1 0
 1.0E-002  2.0E-002  3.0E-002
1 3 ! m1 3 s1 1 ms1 -1
 4.0E-002  5.0E-002  6.0E-002
1 3 ! m1 3 s1 1 ms1 0
 4.0E-002  5.0E-002  6.0E-002
1 3 ! m1 3 s1 1 ms1 1
 4.0E-002  5.0E-002  6.0E-002

! 6 Overlap matrix (4x4, complex)
4 4
 9.9E-001  0.0E+000  1.0E-002  0.0E+000  0.0E+000  0.0E+000  0.0E+000  0.0E+000
-1.0E-002  0.0E+000  9.9E-001  0.0E+000  0.0E+000  0.0E+000  0.0E+000  0.0E+000
 0.0E+000  0.0E+000  0.0E+000  0.0E+000  1.0E+000  0.0E+000  0.0E+000  0.0E+000
 0.0E+000  0.0E+000  0.0E+000  0.0E+000  0.0E+000  0.0E+000  1.0E+000  2.0E-003

! 5 Non-adiabatic couplings (ddr) (4x4x1x3, real)
"""
    + "".join(
        f"1 3 ! m1 {m1} s1 1 ms1 {ms1}   m2 {m2} s2 1 ms2 {ms2}\n"
        f" {a}.0E-001  0.0E+000 -{b}.0E-001\n"
        for a, (m1, ms1) in enumerate([(1, 0), (3, -1), (3, 0), (3, 1)])
        for b, (m2, ms2) in enumerate([(1, 0), (3, -1), (3, 0), (3, 1)])
    )
    + """
! 8 Runtime
 0.000000000E+000
"""
)

LVC_BLOCKS = ("hamiltonian", "gradients")


def test_soc_hamiltonian_is_complex():
    data = qmout.parse_qm_out(SOC_SAMPLE, required=LVC_BLOCKS)
    h = data["hamiltonian"]
    assert len(h) == 4 and h[0][1] == complex(2e-4, 3e-4) and h[1][0] == complex(2e-4, -3e-4)
    assert h[3][3] == complex(0.15, 0.0)


def test_gradient_labels_cover_every_ms_component():
    data = qmout.parse_qm_out(SOC_SAMPLE, required=LVC_BLOCKS)
    assert data["gradient_labels"] == [(1, 1, 0), (3, 1, -1), (3, 1, 0), (3, 1, 1)]
    assert data["gradients"][2] == [[0.04, 0.05, 0.06]]


def test_hyphenated_nac_block_with_all_pairs():
    data = qmout.parse_qm_out(SOC_SAMPLE, required=LVC_BLOCKS)
    assert len(data["nacs"]) == 4 and all(len(row) == 4 for row in data["nacs"])
    # bra outer, ket inner (manual 6.34.2)
    assert data["nacs"][2][1] == [[0.2, 0.0, -0.1]]
    assert data["nac_labels"][2][1] == ((3, 1, 0), (3, 1, -1))


def test_overlap_block_is_complex():
    data = qmout.parse_qm_out(SOC_SAMPLE, required=LVC_BLOCKS)
    s = data["overlap"]
    assert s[0][1] == complex(0.01, 0.0) and s[1][0] == complex(-0.01, 0.0)
    assert s[3][3] == complex(1.0, 0.002)


def test_default_still_requires_the_dipole_block():
    with pytest.raises(ValueError, match="'dipole' block is missing"):
        qmout.parse_qm_out(SOC_SAMPLE)


def test_old_gradient_headers_have_no_labels():
    data = qmout.parse_qm_out(SAMPLE)
    assert data["gradient_labels"] == [None, None]


def test_truncated_overlap_block_names_the_block():
    marker = "! 6 Overlap matrix (4x4, complex)\n4 4\n"
    cut = SOC_SAMPLE[: SOC_SAMPLE.index(marker) + len(marker)]
    with pytest.raises(ValueError, match=r"malformed 'overlap' block near line \d+"):
        qmout.parse_qm_out(cut, required=LVC_BLOCKS)
