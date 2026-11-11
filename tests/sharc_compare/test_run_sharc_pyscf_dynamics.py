"""The SHARC_PYSCF trajectory harness without SHARC or PySCF: the sharc.x input, the trajectory
files and runQM.sh (run against a fake interface of our own), the output.dat reduction (on a
snippet in the layout of the public manual, section 5.3.1), the step alignment and the swap
model on a fake backend."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gqteashmd.core.states import StateLabel
from gqteashmd.core.units import BOHR_IN_ANGSTROM

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "sharc_compare"


def _load(name):
    if name in sys.modules:  # shared with tests/validation/test_sharc_pyscf_dynamics.py
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dyn = _load("run_sharc_pyscf_dynamics")


def test_sharc_input_nacdr_without_interface_phases():
    lines = dyn.sharc_input(40, coupling="nacdr", phases_from_interface=False).splitlines()
    for expected in (
        "nstates 2",
        "state 2 mch",  # S1, 1-based
        "coeff auto",
        "stepsize 0.5",
        "tmax 20.125",
        "nsubsteps 25",
        "surf sharc",
        "nospinorbit",
        "coupling nacdr",
        "eeom li",
        "nogradcorrect",
        "ekincorrect parallel_vel",
        "decoherence_scheme none",
        "hopping_procedure off",
        "grad_all",
        "nac_all",
        "nophases_from_interface",
        "write_grad",
        "write_nacdr",
    ):
        assert expected in lines, expected
    assert "write_overlap" not in lines


def test_sharc_input_overlap_with_interface_phases():
    lines = dyn.sharc_input(10, coupling="overlap").splitlines()
    assert {"coupling overlap", "eeom ld", "write_overlap"} <= set(lines)
    assert "nac_all" not in lines and "nophases_from_interface" not in lines
    with pytest.raises(ValueError, match="coupling"):
        dyn.sharc_input(10, coupling="ddr")
    with pytest.raises(ValueError, match="n_steps"):
        dyn.sharc_input(0)


def test_initial_conditions_lie_on_the_seven_decimal_grid():
    a = dyn.ethylene_initial_conditions()
    b = dyn.ethylene_initial_conditions()
    assert a.symbols == ("C", "C", "H", "H", "H", "H")
    assert a.q_bohr.shape == a.v_au.shape == (6, 3)
    assert np.array_equal(a.v_au, b.v_au)  # default_rng(2026): reproducible
    angstrom = a.q_bohr * BOHR_IN_ANGSTROM
    assert np.max(np.abs(angstrom - np.round(angstrom, 7))) < 1e-12
    assert np.array_equal(dyn.interface_geometry(a.q_bohr), a.q_bohr)  # idempotent
    # xyz: C -0.66838406 -> -0.6683841 angstrom
    assert angstrom[0, 0] == pytest.approx(-0.6683841, abs=1e-12)


def _fake_interface(directory: Path) -> Path:
    """A stand-in for the SHARC Python: it records the QM.in it was given (our own script)."""
    script = directory / "fake_python"
    script.write_text('#!/bin/sh\ncp "$2" QM.seen\necho "$1" > interface.seen\n')
    script.chmod(0o755)
    return script


def test_runqm_turns_step_zero_into_init(tmp_path):
    work = tmp_path / "traj"
    initial = dyn.ethylene_initial_conditions()
    dyn.write_trajectory_files(
        work,
        symbols=initial.symbols,
        masses_amu=initial.masses_amu,
        q0=initial.q_bohr,
        v0=initial.v_au,
        input_text=dyn.sharc_input(2, coupling="nacdr"),
        provider="sharc_pyscf",
        sharc_python=str(_fake_interface(tmp_path)),
    )
    assert (work / "QM" / "PYSCF.template").read_text() == dyn.PYSCF_TEMPLATE
    assert "basis 6-31g*" in dyn.PYSCF_TEMPLATE and "roots 2" in dyn.PYSCF_TEMPLATE
    assert (work / "geom").read_text().splitlines()[0].split()[:2] == ["C", "6.0"]
    env = {**os.environ, "SHARC": "/opt/sharc/bin"}
    for step, expected in (("step       0", "init"), ("step       1", "step       1")):
        (work / "QM" / "QM.in").write_text(f"6\nx\nunit angstrom\n{step}\nH\nGRAD all\n")
        proc = subprocess.run(
            ["sh", "QM/runQM.sh"], cwd=work, env=env, capture_output=True, text=True, timeout=30
        )
        assert proc.returncode == 0, proc.stderr
        assert (work / "QM" / "QM.seen").read_text().splitlines()[3] == expected
    assert (work / "QM" / "interface.seen").read_text().strip() == "/opt/sharc/bin/SHARC_PYSCF.py"


def test_write_trajectory_files_checks_the_provider(tmp_path):
    initial = dyn.ethylene_initial_conditions()
    files = dict(
        symbols=initial.symbols,
        masses_amu=initial.masses_amu,
        q0=initial.q_bohr,
        v0=initial.v_au,
        input_text="x\n",
    )
    with pytest.raises(ValueError, match="provider"):
        dyn.write_trajectory_files(tmp_path / "a", provider="gaussian", **files)
    with pytest.raises(ValueError, match="sharc_python"):
        dyn.write_trajectory_files(tmp_path / "b", provider="sharc_pyscf", **files)
    dyn.write_trajectory_files(tmp_path / "c", provider="ours", **files)
    assert "gq_request.fifo" in (tmp_path / "c" / "QM" / "runQM.sh").read_text()


# output.dat of a singlet nacdr run in the layout of manual 5.3.1 (2 states, 1 atom, %.13E)
def _step(step: int, e0: str, e1: str, c: tuple[str, str], x: str, vx: str) -> str:
    zero = "0.0000000000000E+000"
    dipole = f" {zero}  {zero}     {zero}  {zero}\n" * 2
    return (
        f"! 0 Step\n           {step}\n"
        "! 1 Hamiltonian (MCH) in a.u.\n"
        f" {e0}  {zero}     {zero}  {zero}\n"
        f" {zero}  {zero}     {e1}  {zero}\n"
        "! 2 U matrix\n"
        f" 0.1000000000000E+001  {zero}     {zero}  {zero}\n"
        f" {zero}  {zero}     0.1000000000000E+001  {zero}\n"
        f"! 3 Dipole moments X (MCH) in a.u.\n{dipole}"
        f"! 3 Dipole moments Y (MCH) in a.u.\n{dipole}"
        f"! 3 Dipole moments Z (MCH) in a.u.\n{dipole}"
        f"! 5 Coefficients (diag)\n {c[0]}  {zero}\n {zero}  {c[1]}\n"
        f"! 6 Hopping Probabilities (diag)\n {zero}\n {zero}\n"
        "! 7 Ekin (a.u.)\n 0.4100000000000E-002\n"
        "! 8 states (diag, MCH)\n           2            2\n"
        f"! 9 Random number\n {zero}\n"
        "! 10 Runtime (sec)\n           5\n"
        f"! 11 Geometry in a.u.\n {x}  {zero}  {zero}\n"
        f"! 12 Velocities in a.u.\n {vx}  {zero}  {zero}\n"
    )


OUTPUT_DAT = (
    """SHARC_version   4.1
 method           0
 maxmult           1
 nstates_m    2
 natom           1
 dtstep   20.670686930000000
 nsteps           1
 nsubsteps          25
*************************** End of settings ***************************
! Atomic masses
 0.2187466000000E+005
************************ End of header array data *********************
"""
    + _step(
        0,
        "-0.7804902443000E+002",
        "-0.7767250837000E+002",
        ("0.0000000000000E+000", "0.1000000000000E+001"),
        "-0.1263100000000E+001",
        "-0.1653000000000E-003",
    )
    + _step(
        1,
        "-0.7804885394000E+002",
        "-0.7767391390000E+002",
        ("0.6000000000000E-003", "-0.9999998200000E+000"),
        "-0.1267700000000E+001",
        "-0.1660000000000E-003",
    )
)


def test_trajectory_data_reduces_output_dat(tmp_path):
    path = tmp_path / "output.dat"
    path.write_text(OUTPUT_DAT)
    lvc = _load("run_sharc_lvc")
    data = dyn.trajectory_data(lvc.parse_lvc_output_dat(path), 12.5)
    assert data["steps"].tolist() == [0, 1]
    assert data["dt_au"] == 20.67068693 and data["masses_me"].tolist() == [21874.66]
    assert data["energies"].tolist() == [[-78.04902443, -77.67250837], [-78.04885394, -77.6739139]]
    assert data["populations"][0].tolist() == [0.0, 1.0]
    assert data["populations"][1].tolist() == [0.0006**2, 0.99999982**2]  # |c|^2
    assert data["q"][1].tolist() == [-1.2677, 0.0, 0.0]
    assert data["v"][0, 0] == -1.653e-4
    assert data["active"].tolist() == [1, 1]  # 0-based diagonal state
    assert data["wall_time_s"] == 12.5
    assert data["e_total"].tolist() == [-77.67250837 + 0.0041, -77.6739139 + 0.0041]
    assert data["active_gradient"] is None  # the snippet has no gradient blocks


def test_step_differences_align_by_step_number():
    a = {
        "steps": np.array([0, 1, 2]),
        "q": np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 2.0]]),
        "v": np.zeros((3, 2)),
        "energies": np.zeros((3, 2)),
        "populations": np.zeros((3, 2)),
    }
    b = {key: value[1:] for key, value in a.items()}
    b["q"] = np.array([[1.0, 1.5], [2.0, 2.25]])
    diffs = dyn.step_differences(a, b)
    assert diffs["steps"].tolist() == [1, 2]
    assert diffs["q"].tolist() == [0.5, 0.25]
    assert diffs["v"].tolist() == [0.0, 0.0]
    table = dyn.format_differences(diffs).splitlines()
    assert len(table) == 3 and table[1].split()[:2] == ["1", "5.000e-01"]


class _FakeBackend:
    """Two singlets; records the requests and the resets."""

    def __init__(self):
        self.states = (StateLabel(1, 0), StateLabel(1, 1))
        self.requests = []
        self.resets = 0
        self.overlaps = None

    def reset(self):
        self.resets += 1

    def compute(self, request):
        self.requests.append(request)
        s0, s1 = self.states
        return SimpleNamespace(
            energies={s0: -78.0, s1: -77.6},
            gradients={s0: np.full((1, 3), 0.1), s1: np.full((1, 3), -0.2)},
            nacs={(s0, s1): np.full((1, 3), 0.5)} if request.nacs else {},
            overlaps=self.overlaps,
        )


def test_swap_model_requests_everything_and_orthonormalises():
    backend = _FakeBackend()
    model = dyn.SwapModel(backend, coupling="nacdr")
    assert [(s.multiplicity, s.index, s.ms) for s in model.states] == [(1, 0, 0), (1, 1, 0)]
    snap0 = model.evaluate(np.zeros(3), None)
    assert backend.resets == 1
    request = backend.requests[0]
    assert request.gradients == frozenset(backend.states) and request.overlaps
    assert request.nacs == frozenset({tuple(backend.states)})
    assert snap0.overlaps is None and snap0.gradients[1].tolist() == [-0.2] * 3
    assert snap0.nacs[0, 1].tolist() == [0.5] * 3 and snap0.nacs[1, 0].tolist() == [-0.5] * 3
    assert np.array_equal(snap0.hamiltonian, np.diag([-78.0, -77.6]).astype(complex))
    backend.overlaps = np.array([[0.9, 0.1], [-0.1, 0.9]])
    snap1 = model.evaluate(np.full(3, 0.01), snap0)
    assert backend.resets == 1  # a continuation does not reset the backend
    s = snap1.overlaps
    np.testing.assert_allclose(s.T @ s, np.eye(2), atol=1e-14)  # Lowdin-orthonormalised
    assert model.min_singular_values == [pytest.approx(np.hypot(0.9, 0.1))]
    overlap_model = dyn.SwapModel(_FakeBackend(), coupling="overlap")
    overlap_model.evaluate(np.zeros(3), None)
    assert overlap_model.backend.requests[0].nacs == frozenset()
    with pytest.raises(ValueError, match="coupling"):
        dyn.SwapModel(_FakeBackend(), coupling="ddr")
