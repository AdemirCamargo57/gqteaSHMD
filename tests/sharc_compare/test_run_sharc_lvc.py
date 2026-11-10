"""The SHARC_LVC single-point harness without SHARC: QM.in writer, template filter, and the
subprocess driver against a fake interface script (our own, written into tmp_path)."""

import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

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
