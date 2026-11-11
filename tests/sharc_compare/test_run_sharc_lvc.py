"""The SHARC_LVC harness without SHARC: QM.in writer, template filter, and the subprocess driver
against a fake interface script (our own, written into tmp_path); for the sharc.x trajectories
(Task 10) the input/geom/veloc writers, the output.dat parser (on snippets in the layout of the
public manual, section 5.3.1) and our own QM.out provider for the swap test."""

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from gqteashmd.core.units import HARTREE_IN_EV
from gqteashmd.dynamics.lvc import LVCModel
from gqteashmd.dynamics.options import DynamicsOptions

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "sharc_compare"


def _load(name):
    if name in sys.modules:  # shared with tests/validation/test_sharc_lvc.py
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lvc = _load("run_sharc_lvc")

COORDS = np.array([[0.0, 0.1, -0.2], [1.5e-17, -2.08, 1.245], [0.0, 2.0444482780307709, 1.2]])


def test_qm_in_with_soc_and_overlap():
    text = lvc.format_qm_in(
        ("C", "O", "O"),
        COORDS,
        states=(2, 0, 2),
        savedir="/tmp/x/SAVE",
        step=1,
        soc=True,
        overlap=True,
    )
    lines = text.splitlines()
    assert lines[0] == "3"
    # every coordinate with a decimal point and 17 significant digits (SHARC rejects "0")
    assert lines[2] == "C 0.0000000000000000e+00 1.0000000000000001e-01 -2.0000000000000001e-01"
    assert lines[4].split()[2] == "2.0444482780307709e+00"
    assert [float(x) for x in lines[4].split()[1:]] == COORDS[2].tolist()
    assert lines[5:] == [
        "step 1",
        "savedir /tmp/x/SAVE",
        "unit bohr",
        "states 2 0 2",
        "SOC",
        "GRAD",
        "NACDR",
        "OVERLAP",
    ]
    assert text.endswith("\n")


def test_qm_in_without_soc_requests_h():
    text = lvc.format_qm_in(
        ("C", "O", "O"), COORDS, states=(3,), savedir="S", step=0, nacdr=False, grad=False
    )
    assert text.splitlines()[5:] == ["step 0", "savedir S", "unit bohr", "states 3", "H"]


def test_qm_in_checks_shapes():
    with pytest.raises(ValueError, match="coordinates"):
        lvc.format_qm_in(("C", "O"), COORDS, states=(1,), savedir="S", step=0)


TEMPLATE = """V0.txt
1 0 1
epsilon
0
kappa
0
lambda
0
lambda_soc
0
gamma
0
SOC  R
0.0  1.0  2.0  3.0
1.0  0.0  0.0  0.0
2.0  0.0  0.0  0.0
3.0  0.0  0.0  0.0
SOC  I
0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0
DMX  R
1.0  0.0  0.0  0.0
0.0  1.0  0.0  0.0
0.0  0.0  1.0  0.0
0.0  0.0  0.0  1.0
DMX  I
0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0
"""


def test_without_dipoles_drops_only_the_dm_blocks():
    out = lvc.without_dipoles(TEMPLATE)
    assert "DMX" not in out
    assert out == TEMPLATE[: TEMPLATE.index("DMX  R")]
    assert lvc.without_dipoles(out) == out


FAKE_QM_OUT = """! 0 Basic information
states 1
nmstates 1
natom 1
! 1 Hamiltonian Matrix (1x1, complex)
1 1
 1.250000000000E-001  0.000000000000E+000
! 3 Gradient Vectors (1x1x3, real)
1 3 ! m1 1 s1 1 ms1 0
 1.0E-002  2.0E-002  3.0E-002
! 8 Runtime
 0.000000000E+000
"""


def _fake_sharc(tmp_path: Path, body: str) -> Path:
    home = tmp_path / "fake_sharc"
    (home / "bin").mkdir(parents=True)
    (home / "bin" / "SHARC_LVC.py").write_text(body)
    return home


def test_run_single_point_writes_inputs_and_parses(tmp_path):
    body = f"""import os, sys
from pathlib import Path
assert sys.argv[1:] == ["QM.in"], sys.argv
for name in ("QM.in", "LVC.template", "V0.txt"):
    assert Path(name).is_file(), name
assert os.environ["SHARC"].endswith("bin")
Path("QM.out").write_text({FAKE_QM_OUT!r})
"""
    home = _fake_sharc(tmp_path, body)
    work = tmp_path / "work"
    data = lvc.run_single_point(
        work,
        "qm in text\n",
        "template\n",
        "v0\n",
        sharc_home=home,
        sharc_python=sys.executable,
        timeout=60,
    )
    assert data["hamiltonian"] == [[complex(0.125, 0.0)]]
    assert data["gradient_labels"] == [(1, 1, 0)]
    assert (work / "QM.in").read_bytes() == b"qm in text\n"
    assert (work / "LVC.template").read_text() == "template\n"


def test_run_single_point_does_not_parse_a_stale_qm_out(tmp_path):
    home = _fake_sharc(tmp_path, "pass\n")
    work = tmp_path / "work"
    work.mkdir()
    (work / "QM.out").write_text(FAKE_QM_OUT)
    with pytest.raises(RuntimeError, match="wrote no QM.out"):
        lvc.run_single_point(
            work, "x\n", "t\n", "v\n", sharc_home=home, sharc_python=sys.executable, timeout=60
        )


def test_run_single_point_reports_a_failure(tmp_path):
    home = _fake_sharc(
        tmp_path, "import sys\nprint('boom from the interface', file=sys.stderr)\nsys.exit(43)\n"
    )
    with pytest.raises(RuntimeError, match=r"(?s)failed \(43\).*boom from the interface"):
        lvc.run_single_point(
            tmp_path / "w",
            "x\n",
            "t\n",
            "v\n",
            sharc_home=home,
            sharc_python=sys.executable,
            timeout=60,
        )


def test_timeout_kills_the_process_group(tmp_path):
    body = """import subprocess, time
from pathlib import Path
child = subprocess.Popen(["sleep", "300"])
Path("child.pid").write_text(str(child.pid))
time.sleep(300)
"""
    home = _fake_sharc(tmp_path, body)
    work = tmp_path / "w"
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="did not finish within 2"):
        lvc.run_single_point(
            work, "x\n", "t\n", "v\n", sharc_home=home, sharc_python=sys.executable, timeout=2
        )
    assert time.monotonic() - start < 60
    pid = int((work / "child.pid").read_text())
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"the interface's child process {pid} survived the timeout")


def test_missing_qm_out_after_success_shows_stderr(tmp_path):
    home = _fake_sharc(tmp_path, "import sys\nprint('only on stderr', file=sys.stderr)\n")
    with pytest.raises(RuntimeError, match=r"(?s)wrote no QM.out.*only on stderr"):
        lvc.run_single_point(
            tmp_path / "w", "x\n", "t\n", "v\n", sharc_home=home, sharc_python=sys.executable
        )


# --- Task 10: sharc.x trajectories (inputs: manual 4.1-4.3; output.dat: manual 5.3.1) -----------

qmout = _load("qmout")
DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "lvc"


def test_lvc_input_hopping_off_nacdr():
    options = lvc.LvcSharcOptions(coupling="nacdr", hopping="off")
    text = lvc.lvc_input(200, 0.0125, options, 17, states=(2, 0, 2), state_mch=1)
    lines = text.splitlines()
    for expected in (
        "nstates 2 0 2",
        "charge 0 0 0",
        "state 2 mch",
        "coeff auto",
        "veloc external",
        "stepsize 0.0125",
        f"tmax {(200 + 0.25) * 0.0125!r}",
        "nsubsteps 25",
        "method tsh",
        "surf diagonal",
        "spinorbit",
        "coupling nacdr",
        "eeom li",
        "gradcorrect ngt",
        "ekincorrect parallel_vel",
        "reflect_frustrated none",
        "decoherence_scheme none",
        "decoherence_param 0.1",
        "hopping_procedure off",
        "grad_all",
        "nac_all",
        "rngseed 17",
        "write_grad",
        "write_nacdr",
        "write_overlap",
    ):
        assert expected in lines, expected
    assert not any(line.startswith("output_dat_steps") for line in lines)
    assert text.endswith("\n")


def test_lvc_input_overlap_eselect_and_stride():
    options = lvc.LvcSharcOptions(
        coupling="overlap", hopping="sharc", decoherence="edc", gradcorrect="none", eselect_eV=0.3
    )
    text = lvc.lvc_input(
        4000, 0.0125, options, 3, states=(2, 0, 2), state_mch=0, detailed=False, output_stride=400
    )
    lines = text.splitlines()
    for expected in (
        "state 1 mch",
        "coupling overlap",
        "eeom ld",
        "nogradcorrect",
        "decoherence_scheme edc",
        "hopping_procedure sharc",
        "grad_select",
        "nac_select",
        "eselect 0.3",
        "output_dat_steps 400",
    ):
        assert expected in lines, expected
    for absent in ("grad_all", "nac_all", "write_grad", "write_nacdr", "gradcorrect ngt"):
        assert absent not in lines


def test_lvc_options_from_dynamics_options():
    mapped = lvc.LvcSharcOptions.from_dynamics_options(
        DynamicsOptions(
            dt_au=0.5,
            coupling="overlap",
            hopping="none",
            decoherence="none",
            gradient_transformation="none",
            eselect_Eh=0.3 / HARTREE_IN_EV,
            nsubsteps=20,
            substep_interpolation="endpoint",
        )
    )
    assert mapped.coupling == "overlap"
    assert mapped.hopping == "off"
    assert mapped.gradcorrect == "none"
    assert mapped.nsubsteps == 20
    assert mapped.eselect_eV == pytest.approx(0.3, rel=1e-15)
    default = lvc.LvcSharcOptions.from_dynamics_options(DynamicsOptions(dt_au=0.5))
    assert (default.hopping, default.decoherence, default.reflect_frustrated) == (
        "sharc",
        "edc",
        "none",
    )
    reflect = lvc.LvcSharcOptions.from_dynamics_options(
        DynamicsOptions(dt_au=0.5, frustrated="reflect")
    )
    assert reflect.reflect_frustrated == "parallel_vel"
    with pytest.raises(ValueError, match="fssh"):
        lvc.LvcSharcOptions.from_dynamics_options(
            DynamicsOptions(dt_au=0.5, hopping="fssh", decoherence="none")
        )
    with pytest.raises(ValueError, match="granucci_persico"):
        lvc.LvcSharcOptions.from_dynamics_options(
            DynamicsOptions(dt_au=0.5, edc_convention="granucci_persico")
        )


def test_geom_and_veloc_files_have_full_precision():
    coords = np.array([[0.0, -0.06336696516478459, -0.1324889529605209], [1e-17, -2.0, 1.2]])
    geom = lvc.format_geom(("C", "O"), (12.0, 15.99491462), coords)
    rows = [line.split() for line in geom.splitlines()]
    assert rows[0][:2] == ["C", "6.0"] and rows[1][:2] == ["O", "8.0"]
    assert rows[1][-1] == "15.99491462"
    assert rows[0][3] == "-6.3366965164784586e-02"  # %.16e: 17 significant digits
    assert np.array_equal(np.array([[float(x) for x in r[2:5]] for r in rows]), coords)
    assert all("." in x for r in rows for x in r[1:])
    veloc = lvc.format_veloc(coords * 1e-4)
    back = np.array([[float(x) for x in r.split()] for r in veloc.splitlines()])
    assert np.array_equal(back, coords * 1e-4)
    with pytest.raises(ValueError, match="shape"):
        lvc.format_geom(("C",), (12.0,), coords)


# output.dat in the layout of manual 5.3.1 as this sharc.x writes it: 2 states, 1 atom, %.13E
# mantissas, two steps; only the second has an overlap block (coupling overlap prints it at
# every step; this checks that a step without one is not misread).
_STEP0 = """! 0 Step
           0
! 1 Hamiltonian (MCH) in a.u.
 0.1000000000000E+000  0.0000000000000E+000     0.2000000000000E-003  0.1000000000000E-003
 0.2000000000000E-003 -0.1000000000000E-003     0.3000000000000E+000  0.0000000000000E+000
! 2 U matrix
 0.6000000000000E+000  0.0000000000000E+000     0.0000000000000E+000  0.8000000000000E+000
 0.8000000000000E+000  0.0000000000000E+000     0.0000000000000E+000 -0.6000000000000E+000
! 3 Dipole moments X (MCH) in a.u.
 0.0000000000000E+000  0.0000000000000E+000     0.0000000000000E+000  0.0000000000000E+000
 0.0000000000000E+000  0.0000000000000E+000     0.0000000000000E+000  0.0000000000000E+000
! 5 Coefficients (diag)
 0.6000000000000E+000  0.0000000000000E+000
 0.0000000000000E+000 -0.8000000000000E+000
! 6 Hopping Probabilities (diag)
 0.0000000000000E+000
 0.0000000000000E+000
! 7 Ekin (a.u.)
 0.1250000000000E-002
! 8 states (diag, MCH)
           2            1
! 9 Random number
 0.0000000000000E+000
! 10 Runtime (sec)
           1
! 11 Geometry in a.u.
-0.1500000000000E+001  0.0000000000000E+000  0.2500000000000E-016
! 12 Velocities in a.u.
 0.2000000000000E-001  0.0000000000000E+000  0.0000000000000E+000
! 15 Gradients (MCH) State  1
 0.1000000000000E-001  0.0000000000000E+000  0.0000000000000E+000
! 15 Gradients (MCH) State  2
-0.1000000000000E-001  0.0000000000000E+000  0.0000000000000E+000
! 16 NACdr matrix element (MCH)  1   1
 0.0000000000000E+000  0.0000000000000E+000  0.0000000000000E+000
! 16 NACdr matrix element (MCH)  1   2
 0.5000000000000E+000  0.0000000000000E+000  0.0000000000000E+000
! 16 NACdr matrix element (MCH)  2   1
-0.5000000000000E+000  0.0000000000000E+000  0.0000000000000E+000
! 16 NACdr matrix element (MCH)  2   2
 0.0000000000000E+000  0.0000000000000E+000  0.0000000000000E+000
"""
_STEP1 = """! 0 Step
           1
! 1 Hamiltonian (MCH) in a.u.
 0.1100000000000E+000  0.0000000000000E+000     0.2000000000000E-003  0.1000000000000E-003
 0.2000000000000E-003 -0.1000000000000E-003     0.2900000000000E+000  0.0000000000000E+000
! 2 U matrix
 0.6000000000000E+000  0.0000000000000E+000     0.0000000000000E+000  0.8000000000000E+000
 0.8000000000000E+000  0.0000000000000E+000     0.0000000000000E+000 -0.6000000000000E+000
! 4 Overlap matrix (MCH)
 0.9900000000000E+000  0.0000000000000E+000     0.1000000000000E-001  0.0000000000000E+000
-0.1000000000000E-001  0.0000000000000E+000     0.9900000000000E+000  0.0000000000000E+000
! 5 Coefficients (diag)
 0.5000000000000E+000  0.1000000000000E+000
 0.0000000000000E+000 -0.7000000000000E+000
! 6 Hopping Probabilities (diag)
 0.2500000000000E-001
 0.0000000000000E+000
! 7 Ekin (a.u.)
 0.1300000000000E-002
! 8 states (diag, MCH)
           1            2
! 9 Random number
 0.7500000000000E+000
! 10 Runtime (sec)
           0
! 11 Geometry in a.u.
-0.1490000000000E+001  0.0000000000000E+000  0.0000000000000E+000
! 12 Velocities in a.u.
 0.2100000000000E-001  0.0000000000000E+000  0.0000000000000E+000
! 15 Gradients (MCH) State  1
 0.2000000000000E-001  0.0000000000000E+000  0.0000000000000E+000
! 15 Gradients (MCH) State  2
-0.2000000000000E-001  0.0000000000000E+000  0.0000000000000E+000
! 16 NACdr matrix element (MCH)  1   1
 0.0000000000000E+000  0.0000000000000E+000  0.0000000000000E+000
! 16 NACdr matrix element (MCH)  1   2
 0.6000000000000E+000  0.0000000000000E+000  0.0000000000000E+000
! 16 NACdr matrix element (MCH)  2   1
-0.6000000000000E+000  0.0000000000000E+000  0.0000000000000E+000
! 16 NACdr matrix element (MCH)  2   2
 0.0000000000000E+000  0.0000000000000E+000  0.0000000000000E+000
"""
OUTPUT_DAT = (
    """SHARC_version   4.0
 method           0
 maxmult           1
 nstates_m    2
 natom           1
 dtstep  0.50000000000000000
 nsteps           1
 nsubsteps          25
 ezero   0.0000000000000000
 write_overlap           1
 write_grad           1
 write_nacdr           1
*************************** End of settings ***************************
! Atomic numbers
 0.2000000000000E+001
! Elements
 He
! Atomic masses
 0.2000000000000E+004
************************ End of header array data *********************
"""
    + _STEP0
    + _STEP1
)


def test_parse_lvc_output_dat(tmp_path):
    path = tmp_path / "output.dat"
    path.write_text(OUTPUT_DAT)
    t = lvc.parse_lvc_output_dat(path)
    assert t.dt_au == 0.5 and t.nsteps_header == 1 and t.nstates_m == (2,)
    assert t.masses_me.tolist() == [2000.0]
    assert t.steps.tolist() == [0, 1]
    assert t.hamiltonian.shape == (2, 2, 2)
    assert t.hamiltonian[0, 0, 1] == complex(2e-4, 1e-4)
    assert t.hamiltonian[1, 1, 1] == complex(0.29, 0.0)
    assert t.u[0, 0, 1] == complex(0.0, 0.8)  # printed row = MCH state, column = diagonal state
    assert t.u[0, 1, 1] == complex(0.0, -0.6)
    assert t.overlaps[0] is None and t.overlaps[1][0, 1] == complex(0.01, 0.0)
    assert t.c[1].tolist() == [complex(0.5, 0.1), complex(0.0, -0.7)]
    assert t.probabilities[1].tolist() == [0.025, 0.0]
    assert t.e_kinetic.tolist() == [0.00125, 0.0013]
    assert t.active.tolist() == [1, 0]  # 0-based diagonal states
    assert t.active_mch.tolist() == [0, 1]
    assert t.random_numbers.tolist() == [0.0, 0.75]
    assert t.q.shape == (2, 3) and t.q[0, 2] == 2.5e-17
    assert t.v[1, 0] == 0.021
    assert t.gradients is not None and t.gradients.shape == (2, 2, 3)
    assert t.gradients[1, 1, 0] == -0.02
    assert t.nacs is not None and t.nacs.shape == (2, 2, 2, 3)
    assert t.nacs[1, 0, 1, 0] == 0.6


def test_parse_lvc_output_dat_without_vectors(tmp_path):
    kept, skip = [], False
    for line in OUTPUT_DAT.splitlines():
        if line.startswith("!"):
            skip = line.startswith(("! 15", "! 16"))
        if not skip:
            kept.append(line)
    path = tmp_path / "output.dat"
    path.write_text("\n".join(kept) + "\n")
    t = lvc.parse_lvc_output_dat(path)
    assert t.gradients is None and t.nacs is None
    assert t.steps.tolist() == [0, 1]


OUTPUT_LIS = """#=====================================================
#     Step |         Time |         State |    Energy |
#          |              |  diag |   MCH |   kin |   pot |   tot |
#=====================================================
         0         0.0000       5       2       0.029456       4.502414       4.531870
         1         0.0121       5       2       0.029995       4.501875       4.531871
    # Surface Hop/Pointer State Switch: new state= 4
         2         0.0242       4       2       0.030540       4.501331       4.531869
"""


def test_parse_output_lis_total_energies(tmp_path):
    path = tmp_path / "output.lis"
    path.write_text(OUTPUT_LIS)
    steps, e_tot = lvc.parse_output_lis(path)
    assert steps.tolist() == [0, 1, 2]
    assert e_tot.tolist() == [4.531870, 4.531871, 4.531869]


def test_parse_lvc_output_dat_rejects_a_truncated_block(tmp_path):
    path = tmp_path / "output.dat"
    path.write_text(OUTPUT_DAT[: OUTPUT_DAT.rindex("! 2 U matrix") + 40])
    with pytest.raises(ValueError, match="U matrix"):
        lvc.parse_lvc_output_dat(path)


# QM.in as sharc.x writes it for its interface (manual 6.34.1): angstrom, positions + velocities.
SHARC_QM_IN = """     3
            846172133
C    0.0000000   -0.0332850   -0.0699962       0.0000000    0.0001882    0.0000885
O    0.0000000   -1.0775260    0.6540609      -0.0000000    0.0000556    0.0000075
O    0.0000000    1.1024978    0.6684783       0.0000000   -0.0001968   -0.0000740
unit angstrom
states   2  0  2
charge   0  0  0
dt     0.500000
step       5
savedir /tmp/x/restart
retain       2
SOC
DM
GRAD all
PHASES
NACDR
"""


def test_parse_sharc_qm_in():
    request = lvc.parse_sharc_qm_in(SHARC_QM_IN)
    assert request.symbols == ("C", "O", "O")
    assert request.step == 5
    assert request.states == (2, 0, 2)
    assert request.requests == frozenset({"soc", "dm", "grad", "phases", "nacdr"})
    assert request.coords_bohr.shape == (3, 3)
    assert request.coords_bohr[1, 1] == pytest.approx(-1.0775260 / lvc.BOHR_IN_ANGSTROM, 1e-15)
    bohr = lvc.parse_sharc_qm_in(SHARC_QM_IN.replace("unit angstrom", "unit bohr"))
    assert bohr.coords_bohr[1, 1] == -1.0775260
    with pytest.raises(ValueError, match="QM.in"):
        lvc.parse_sharc_qm_in("3\ncomment\nC 0 0\n")


def test_qm_out_from_our_model_reads_back():
    model = LVCModel.from_template(DATA / "lvc_soc.template")
    q0 = model.reference_q + 0.01 * np.arange(model.n_dof)
    snap0 = model.evaluate(q0, None)
    snap1 = model.evaluate(q0 + 0.02, snap0)
    requests = frozenset({"soc", "dm", "grad", "nacdr", "overlap", "phases"})
    text = lvc.format_qm_out(snap1, model.states, requests)
    data = qmout.parse_qm_out(
        text, required=("hamiltonian", "dipole", "gradients", "nacs", "overlap")
    )
    n, f = model.n_states, model.n_dof
    assert np.array_equal(np.array(data["hamiltonian"]), snap1.hamiltonian)
    assert np.array_equal(np.array(data["gradients"]).reshape(n, f), snap1.gradients)
    assert np.array_equal(np.array(data["nacs"]).reshape(n, n, f), snap1.nacs)
    assert np.array_equal(np.array(data["overlap"]).real, snap1.overlaps)
    expected = [(s.multiplicity, s.index + 1, s.ms) for s in model.states]
    assert data["gradient_labels"] == expected
    assert data["nac_labels"][2][5] == (expected[2], expected[5])
    assert "! 7 Wave function phases (8x1, complex)" in text
    plain = lvc.format_qm_out(snap0, model.states, frozenset({"h", "grad"}))
    for absent in ("Overlap", "couplings", "phases", "Dipole"):
        assert absent not in plain


def _qm_in_text(model: LVCModel, step: int, extra: str) -> str:
    q = model.reference_q.reshape(-1, 3) * lvc.BOHR_IN_ANGSTROM
    rows = "".join(
        f"{s} {x:.7f} {y:.7f} {z:.7f} 0.0 0.0 0.0\n"
        for s, (x, y, z) in zip(model.parameters.v0.symbols, q, strict=True)
    )
    return f"3\nx\n{rows}unit angstrom\nstates 2 0 2\nstep {step}\nSOC\nGRAD all\nNACDR\n{extra}"


def test_qm_server_answers_runqm(tmp_path):
    model = LVCModel.from_template(DATA / "lvc_soc.template")
    work = tmp_path / "traj"
    qm = work / "QM"
    qm.mkdir(parents=True)
    script = lvc.write_server_runqm(qm)

    def call() -> subprocess.CompletedProcess:
        return subprocess.run(
            ["sh", str(script)], cwd=work, capture_output=True, text=True, timeout=30
        )

    with lvc.QmServer(model, qm) as server:
        for step, extra in ((0, "PHASES\n"), (1, "OVERLAP\n")):
            (qm / "QM.in").write_text(_qm_in_text(model, step, extra))
            proc = call()
            assert proc.returncode == 0, proc.stderr
            data = qmout.parse_qm_out((qm / "QM.out").read_text(), required=("hamiltonian",))
            assert ("overlap" in data) == (step == 1)
        request = lvc.parse_sharc_qm_in((qm / "QM.in").read_text())
        expected = model.evaluate(request.coords_bohr.reshape(-1), None)
        assert np.array_equal(np.array(data["hamiltonian"]), expected.hamiltonian)
        (qm / "QM.in").write_text("garbage\n")
        assert call().returncode != 0
        assert (server.calls, server.errors) == (3, 1)
    assert not server.is_alive()
