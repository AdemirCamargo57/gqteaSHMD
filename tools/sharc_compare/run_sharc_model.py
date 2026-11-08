"""Run the SHARC 4.1 driver (sharc.x) as a black box on a 1D Tully model; parse what it wrote.

SHARC is only *executed*; nothing from SHARC is imported or copied. The input formats used here
come from the public SHARC 4.1 manual (sharc-md.org): the ``input`` keywords (section 4.1),
the ``geom``/``veloc`` files (4.2, 4.3), ``QM/runQM.sh`` and the file-based interface QM.in /
QM.out (3.1.1, 6.34), the ANALYTICAL.template format (6.3.2) and the output files (5.2, 5.3,
5.6). Every writer and parser below is our own code.

The 1D model is one He atom (a 2-electron singlet, so the interface accepts charge 0) whose
mass is set to the model mass in the ``geom`` file; the potentials depend on x only, and
y = z = 0 with zero y/z velocities. Without a thermostat SHARC removes no translation or
rotation (manual 4.1.3, ``remove_trans_rot``).

Two electronic-structure providers can be put behind ``QM/runQM.sh``:

- ``"analytical"``: SHARC's own SHARC_ANALYTICAL.py with an ANALYTICAL.template that we write
  (sympy expressions of the diabatic matrix). It reloads Python and sympy at every step
  (about 0.5 s per step), so it is used for short runs.
- ``"fast"``: our own awk program implementing the same diabatic matrices, writing QM.out as
  specified in manual 6.34.2 (a few ms per step). It can also end a trajectory with SHARC's
  documented ``STOP`` file once |x| > x_exit while moving outward, which is the ensemble stop
  rule of ``gqteashmd.dynamics.ensemble``. The validation test checks that both providers give
  the same SHARC trajectory.

The pysharc driver (driver.py) was not usable: on this installation it starts with an empty
input (the log echoes no keywords), so sharc.x is used.

Example (WSL):
  python tools/sharc_compare/run_sharc_model.py --sharc-home /home/ajc/sharc4-4.1 \
      --python /home/ajc/miniforge3/envs/sharc4.1/bin/python --model sac --x0 -2 --v0 0.02 \
      --steps 50 --dt-au 1.0 --work-dir /tmp/sharc-sac
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BOHR_IN_ANGSTROM = 0.529177210903
DALTON_IN_ME = 1822.888486209
AU_TIME_IN_FS = 0.024188843265857
PROVIDERS = ("analytical", "fast")


@dataclass(frozen=True)
class ModelSpec:
    """A Tully model: ``name`` in sac, dac, ecr; ``params`` its constants; mass in m_e."""

    name: str
    params: dict[str, float]
    mass_me: float = 2000.0

    def __post_init__(self) -> None:
        expected = {"sac": "ABCD", "dac": ("A", "B", "C", "D", "E0"), "ecr": "ABC"}
        if self.name not in expected:
            raise ValueError(f"model name must be sac, dac or ecr; got {self.name!r}")
        if set(self.params) != set(expected[self.name]):
            raise ValueError(f"{self.name} needs parameters {sorted(expected[self.name])}")


def model_spec(model) -> ModelSpec:
    """The spec of one of our gqteashmd.dynamics.models Tully objects (duck-typed)."""
    names = {
        "TullySimpleAvoidedCrossing": ("sac", ("A", "B", "C", "D")),
        "TullyDualAvoidedCrossing": ("dac", ("A", "B", "C", "D", "E0")),
        "TullyExtendedCoupling": ("ecr", ("A", "B", "C")),
    }
    kind, keys = names[type(model).__name__]
    return ModelSpec(kind, {k: float(getattr(model, k)) for k in keys}, float(model.masses[0]))


# --- writers --------------------------------------------------------------------------------


def analytical_template(spec: ModelSpec) -> str:
    """ANALYTICAL.template (manual 6.3.2): header, a Variables block, the Hamiltonian block."""
    if spec.name == "sac":
        v11 = "Piecewise((A*(1-exp(-B*x)), x >= 0), (-A*(1-exp(B*x)), True))"
        rows = [f"{v11};", f"C*exp(-D*x**2); -{v11};"]
    elif spec.name == "dac":
        rows = ["0;", "C*exp(-D*x**2); -A*exp(-B*x**2)+E0;"]
    else:
        v12 = "Piecewise((B*exp(C*x), x < 0), (B*(2-exp(-C*x)), True))"
        rows = ["A;", f"{v12}; -A;"]
    variables = "\n".join(f"{k} {v!r}" for k, v in spec.params.items())
    body = "\n".join(rows)
    return f"1\n2\nHe x 0 0\nVariables\n{variables}\nEnd\nHamiltonian\n{body}\n"


_AWK_MODELS = {
    # diabatic V11, V22, V12 and their x-derivatives (same definitions as gqteashmd.dynamics)
    "sac": """
  if (x >= 0) { e = exp(-B * x); v11 = A * (1 - e); d11 = A * B * e }
  else { e = exp(B * x); v11 = -A * (1 - e); d11 = A * B * e }
  v22 = -v11; d22 = -d11
  v12 = C * exp(-D * x * x); d12 = -2 * D * x * v12""",
    "dac": """
  g = exp(-B * x * x)
  v11 = 0; d11 = 0
  v22 = -A * g + E0; d22 = 2 * A * B * x * g
  v12 = C * exp(-D * x * x); d12 = -2 * D * x * v12""",
    "ecr": """
  v11 = A; d11 = 0; v22 = -A; d22 = 0
  if (x < 0) { v12 = B * exp(C * x); d12 = B * C * exp(C * x) }
  else { v12 = B * (2 - exp(-C * x)); d12 = B * C * exp(-C * x) }""",
}

_AWK_PROGRAM = """# Our own file-based SHARC interface (manual 6.34) for a 1D two-state model.
# sharc.x runs QM/runQM.sh (which execs this program) from the trajectory directory.
BEGIN {{
  BOHR = {bohr!r}; XEXIT = {x_exit!r}
{constants}
  unit = "bohr"; nr = 0
  while ((getline line < "QM/QM.in") > 0) {{
    nr++
    n = split(line, t)
    if (nr == 3) {{ x = t[2] + 0; vx = t[5] + 0 }}
    else if (nr > 3 && n > 0) {{
      key = tolower(t[1])
      if (key == "unit") unit = tolower(t[2])
      else req[key] = 1
    }}
  }}
  close("QM/QM.in")
  if (nr < 3) {{ print "cannot read QM/QM.in" > "/dev/stderr"; exit 1 }}
  if (unit == "angstrom") x = x / BOHR
  if (("overlap" in req) || ("soc" in req)) {{
    print "unsupported request" > "/dev/stderr"; exit 1
  }}
{model}
  # adiabatic states of [[v11, v12], [v12, v22]]: mixing angle th = atan2(v12, (v11-v22)/2)/2,
  # continuous along the paths of these models; u1 = (-sin th, cos th), u2 = (cos th, sin th)
  # (the sign of a state is a gauge; this one matched SHARC_ANALYTICAL.py for SAC and ECR, and
  # differed in the sign of state 2 for DAC: no observable changes)
  m = (v11 + v22) / 2; h = (v11 - v22) / 2; r = sqrt(h * h + v12 * v12)
  e1 = m - r; e2 = m + r
  th = atan2(v12, h) / 2
  a1 = -sin(th); b1 = cos(th); a2 = cos(th); b2 = sin(th)
  g1 = a1 * a1 * d11 + 2 * a1 * b1 * d12 + b1 * b1 * d22
  g2 = a2 * a2 * d11 + 2 * a2 * b2 * d12 + b2 * b2 * d22
  g12 = a1 * a2 * d11 + (a1 * b2 + b1 * a2) * d12 + b1 * b2 * d22
  n12 = g12 / (e2 - e1)
  f = "%.15E"
  out = "QM/QM.out"
  printf "! 0 Basic information\\nstates 2\\nnmstates 2\\nnatom 1\\nnpc 0\\ncharges 0\\n" > out
  printf "! 1 Hamiltonian Matrix (2x2, complex)\\n2 2\\n" > out
  printf f " 0.0 0.0 0.0\\n0.0 0.0 " f " 0.0\\n", e1, e2 > out
  if ("dm" in req) {{
    printf "! 2 Dipole Moment Matrices (3x2x2, complex)\\n" > out
    for (p = 0; p < 3; p++) printf "2 2\\n0.0 0.0 0.0 0.0\\n0.0 0.0 0.0 0.0\\n" > out
  }}
  if ("grad" in req) {{
    printf "! 3 Gradient Vectors (2x1x3, real)\\n" > out
    printf "1 3 ! m1 1 s1 1 ms1 0\\n" f " 0.0 0.0\\n", g1 > out
    printf "1 3 ! m1 1 s1 2 ms1 0\\n" f " 0.0 0.0\\n", g2 > out
  }}
  if ("nacdr" in req) {{
    printf "! 5 Non-adiabatic couplings (ddr) (2x2x1x3, real)\\n" > out
    printf "1 3 ! m1 1 s1 1 ms1 0 m2 1 s2 1 ms2 0\\n0.0 0.0 0.0\\n" > out
    printf "1 3 ! m1 1 s1 1 ms1 0 m2 1 s2 2 ms2 0\\n" f " 0.0 0.0\\n", n12 > out
    printf "1 3 ! m1 1 s1 2 ms1 0 m2 1 s2 1 ms2 0\\n" f " 0.0 0.0\\n", -n12 > out
    printf "1 3 ! m1 1 s1 2 ms1 0 m2 1 s2 2 ms2 0\\n0.0 0.0 0.0\\n" > out
  }}
  if ("phases" in req) {{
    printf "! 7 Wave function phases (2x1, complex)\\n2\\n1.0 0.0\\n1.0 0.0\\n" > out
  }}
  printf "! 8 Runtime\\n0.0\\n" > out
  close(out)
  if (XEXIT > 0 && ((x > XEXIT && vx > 0) || (x < -XEXIT && vx < 0))) printf "" > "STOP"
  exit 0
}}
"""


def fast_interface(spec: ModelSpec, x_exit: float | None) -> str:
    """The awk program of the ``"fast"`` provider (x_exit None or <= 0: never writes STOP)."""
    constants = "\n".join(f"  {k} = {v!r}" for k, v in spec.params.items())
    return _AWK_PROGRAM.format(
        bohr=BOHR_IN_ANGSTROM,
        x_exit=float(x_exit) if x_exit else 0.0,
        constants=constants,
        model=_AWK_MODELS[spec.name],
    )


@dataclass(frozen=True)
class SharcOptions:
    """The SHARC settings we compare; ``from_dynamics_options`` maps our DynamicsOptions."""

    hopping: str = "sharc"  # sharc | off
    ekincorrect: str = "parallel_vel"  # parallel_vel | parallel_nac
    reflect_frustrated: str = "none"  # none | parallel_vel | parallel_nac
    decoherence: str = "none"  # none | edc
    decoherence_param: float = 0.1
    nsubsteps: int = 25

    @classmethod
    def from_dynamics_options(cls, options) -> SharcOptions:
        hopping = {"sharc": "sharc", "none": "off"}.get(options.hopping)
        if hopping is None:
            raise ValueError(f"SHARC has no {options.hopping!r} hopping (use sharc or none)")
        reflect = "none" if options.frustrated == "keep" else options.ekin_correction
        return cls(
            hopping=hopping,
            ekincorrect=options.ekin_correction,
            reflect_frustrated=reflect,
            decoherence=options.decoherence,
            decoherence_param=float(options.decoherence_param_Eh),
            nsubsteps=int(options.nsubsteps),
        )


def sharc_input(
    n_steps: int,
    dt_fs: float,
    options: SharcOptions,
    seed: int,
    *,
    detailed: bool,
) -> str:
    """The main ``input`` file (manual 4.1). The run length is set with ``tmax``.

    (On this installation sharc.x ignored ``nsteps`` and ran the default 5 fs; ``tmax`` works,
    and the header of output.dat reports the resulting number of steps.)
    """
    lines = [
        f"printlevel {2 if detailed else 0}",
        'geomfile "geom"',
        "veloc external",
        'velocfile "veloc"',
        "nstates 2",
        "charge 0",
        "state 1 mch",
        "coeff auto",
        f"stepsize {dt_fs!r}",
        f"tmax {(n_steps + 0.25) * dt_fs!r}",
        f"nsubsteps {options.nsubsteps}",
        "method tsh",
        "surf mch",
        "coupling nacdr",
        "eeom li",
        "nogradcorrect",
        f"ekincorrect {options.ekincorrect}",
        f"reflect_frustrated {options.reflect_frustrated}",
        f"decoherence_scheme {options.decoherence}",
        f"decoherence_param {options.decoherence_param!r}",
        f"hopping_procedure {options.hopping}",
        "grad_all",
        "nac_all",
        "nospinorbit",
        f"rngseed {int(seed)}",
        "output_format ascii",
        "nowrite_restart_files",
    ]
    if detailed:
        lines += ["write_grad", "write_nacdr"]
    else:
        lines += [f"output_dat_steps {max(1, n_steps)}"]
    return "\n".join(lines) + "\n"


def write_trajectory(
    work_dir: Path,
    spec: ModelSpec,
    x0: float,
    v0: float,
    n_steps: int,
    dt_fs: float,
    options: SharcOptions,
    seed: int,
    *,
    provider: str,
    python: str,
    detailed: bool,
    x_exit: float | None = None,
) -> None:
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {', '.join(PROVIDERS)}; got {provider!r}")
    if provider == "analytical" and x_exit:
        raise ValueError("x_exit needs the fast provider")
    qm = work_dir / "QM"
    qm.mkdir(parents=True, exist_ok=False)
    weight = spec.mass_me / DALTON_IN_ME
    (work_dir / "geom").write_text(f"He 2.0 {float(x0)!r} 0.0 0.0 {weight!r}\n")
    (work_dir / "veloc").write_text(f"{float(v0)!r} 0.0 0.0\n")
    (work_dir / "input").write_text(sharc_input(n_steps, dt_fs, options, seed, detailed=detailed))
    script = qm / "runQM.sh"
    if provider == "analytical":
        (qm / "ANALYTICAL.template").write_text(analytical_template(spec))
        script.write_text(
            f'#!/bin/bash\ncd QM || exit 1\nexec "{python}" "$SHARC/SHARC_ANALYTICAL.py" QM.in\n'
        )
    else:
        # sharc.x runs the script with sh from the trajectory directory; exec avoids a fork.
        awk = shutil.which("awk")
        if awk is None:
            raise RuntimeError("the fast provider needs awk")
        (qm / "model.awk").write_text(fast_interface(spec, x_exit))
        script.write_text(f'#!/bin/sh\nexec "{awk}" -f QM/model.awk\n')
    script.chmod(0o755)


def _environment(sharc_home: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["SHARC"] = str(sharc_home / "bin")
    # PYTHONPATH is replaced, not extended: SHARC's own conda environment must not see ours.
    env["PYTHONPATH"] = f"{sharc_home / 'lib'}:{sharc_home / 'pysharc'}"
    env.pop("SHARC_LOG", None)
    env.pop("SHARCLOG", None)
    return env


# --- parsers --------------------------------------------------------------------------------


@dataclass
class SharcTrajectory:
    """What one sharc.x run wrote. Per-step arrays come from output.dat (all steps written only
    with ``detailed=True``); ``final_*`` from the last lines of output.lis and output.xyz."""

    dt_au: float
    nsteps_header: int
    masses_me: np.ndarray
    steps: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    q: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    v: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    c: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=complex))
    energies: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    active: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    probabilities: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    random_numbers: np.ndarray = field(default_factory=lambda: np.zeros(0))
    e_kinetic: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gradients: np.ndarray = field(default_factory=lambda: np.zeros((0, 2, 3)))
    nacs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2, 2, 3)))
    final_step: int = -1
    final_state: int = -1  # 0-based diagonal state from output.lis
    final_x_bohr: float = math.nan  # from output.xyz
    hop_lines: list[str] = field(default_factory=list)
    stopped: bool = False  # the STOP file ended the run


def _floats(line: str) -> list[float]:
    return [float(t.replace("D", "E")) for t in line.split()]


def parse_output_dat(path: Path) -> SharcTrajectory:
    """Parse output.dat (manual 5.3): header keywords, header arrays, then per-step blocks."""
    lines = path.read_text().splitlines()
    header: dict[str, str] = {}
    i = 0
    while i < len(lines) and not lines[i].startswith("****"):
        parts = lines[i].split(None, 1)
        if len(parts) == 2:
            header[parts[0].lower()] = parts[1].strip()
        i += 1
    blocks: list[tuple[str, list[str]]] = []
    for line in lines[i:]:
        if line.startswith("!"):
            blocks.append((line[1:].strip(), []))
        elif blocks and not line.startswith("****"):
            blocks[-1][1].append(line)
    masses = np.array([])
    steps: list[dict] = []
    for title, body in blocks:
        if title == "Atomic masses":
            masses = np.array([v for row in body for v in _floats(row)])
        elif title == "0 Step":
            steps.append({"step": int(body[0].split()[0]), "grad": [], "nac": []})
        elif not steps:
            continue
        elif title.startswith("1 Hamiltonian"):
            rows = [_floats(r) for r in body]
            steps[-1]["energies"] = [rows[k][2 * k] for k in range(len(rows))]
        elif title.startswith("5 Coefficients"):
            steps[-1]["c"] = [complex(*_floats(r)[:2]) for r in body]
        elif title.startswith("6 Hopping"):
            steps[-1]["p"] = [_floats(r)[0] for r in body]
        elif title.startswith("7 Ekin"):
            steps[-1]["ekin"] = _floats(body[0])[0]
        elif title.startswith("8 states"):
            steps[-1]["active"] = int(body[0].split()[0]) - 1
        elif title.startswith("9 Random"):
            steps[-1]["xi"] = _floats(body[0])[0]
        elif title.startswith("11 Geometry"):
            steps[-1]["q"] = _floats(body[0])
        elif title.startswith("12 Velocities"):
            steps[-1]["v"] = _floats(body[0])
        elif title.startswith("15 Gradients"):
            steps[-1]["grad"].append(_floats(body[0]))
        elif title.startswith("16 NACdr"):
            steps[-1]["nac"].append(_floats(body[0]))
    traj = SharcTrajectory(
        dt_au=float(header["dtstep"]),
        nsteps_header=int(header["nsteps"]),
        masses_me=masses,
    )
    if steps:
        traj.steps = np.array([s["step"] for s in steps])
        traj.q = np.array([s["q"] for s in steps])
        traj.v = np.array([s["v"] for s in steps])
        traj.c = np.array([s["c"] for s in steps])
        traj.energies = np.array([s["energies"] for s in steps])
        traj.active = np.array([s["active"] for s in steps])
        traj.probabilities = np.array([s["p"] for s in steps])
        traj.random_numbers = np.array([s["xi"] for s in steps])
        traj.e_kinetic = np.array([s["ekin"] for s in steps])
        if all(len(s["grad"]) == 2 for s in steps):
            traj.gradients = np.array([s["grad"] for s in steps])
        if all(len(s["nac"]) == 4 for s in steps):
            traj.nacs = np.array([s["nac"] for s in steps]).reshape(len(steps), 2, 2, 3)
    return traj


_LIS_ROW = re.compile(r"^\s*\d+\s+[-+0-9.Ee]+\s+\d+\s+\d+\s")


def parse_final(work_dir: Path, traj: SharcTrajectory) -> SharcTrajectory:
    """Final step and state from output.lis (manual 5.2), final x from output.xyz (5.6)."""
    rows, hops = [], []
    for line in (work_dir / "output.lis").read_text().splitlines():
        if _LIS_ROW.match(line):
            rows.append(line)
        elif line.startswith("#") and "|" not in line and not set(line) <= {"#", "="}:
            hops.append(line)  # e.g. "# Surface Hop/Pointer State Switch: new state= ..."
    if not rows:
        raise ValueError(f"{work_dir / 'output.lis'} has no step lines")
    last = rows[-1].split()
    traj.final_step, traj.final_state = int(last[0]), int(last[2]) - 1
    traj.hop_lines = hops
    xyz = (work_dir / "output.xyz").read_text().splitlines()
    atom_lines = [ln for ln in xyz if ln.split() and ln.split()[0] == "He"]
    traj.final_x_bohr = float(atom_lines[-1].split()[1]) / BOHR_IN_ANGSTROM
    traj.stopped = (work_dir / "STOP").exists()
    return traj


# --- the run --------------------------------------------------------------------------------


def run(
    sharc_home: Path,
    python: str,
    model_spec: ModelSpec,
    x0: float,
    v0: float,
    n_steps: int,
    dt_fs: float,
    options: SharcOptions,
    seed: int,
    work_dir: Path,
    *,
    provider: str = "analytical",
    detailed: bool = True,
    x_exit: float | None = None,
    timeout_s: float = 3600.0,
) -> SharcTrajectory:
    """Write the inputs in ``work_dir`` (must not exist), run ``sharc.x input``, parse.

    ``detailed`` writes every step with gradients and NACs to output.dat; otherwise only the
    final state/position are parsed (output.lis, output.xyz). ``x_exit`` (fast provider only)
    ends the run through SHARC's STOP file once |x| > x_exit while moving outward.
    """
    sharc_home = Path(sharc_home)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=False)
    write_trajectory(
        work_dir,
        model_spec,
        x0,
        v0,
        n_steps,
        dt_fs,
        options,
        seed,
        provider=provider,
        python=python,
        detailed=detailed,
        x_exit=x_exit,
    )
    # The environment is set up as the manual recommends (2.2.2): source bin/sharcvars.sh.
    command = 'source "$SHARC/sharcvars.sh" > /dev/null && exec "$SHARC/sharc.x" input'
    proc = subprocess.run(
        ["bash", "-c", command],
        cwd=work_dir,
        env=_environment(sharc_home),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"sharc.x failed ({proc.returncode}) in {work_dir}:\n{proc.stderr[-2000:]}"
            f"\n{proc.stdout[-2000:]}"
        )
    traj = parse_output_dat(work_dir / "output.dat")
    return parse_final(work_dir, traj)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sharc-home", required=True)
    p.add_argument("--python", required=True, help="Python of the SHARC conda environment")
    p.add_argument("--model", choices=("sac", "dac", "ecr"), required=True)
    p.add_argument("--x0", type=float, required=True)
    p.add_argument("--v0", type=float, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--dt-au", type=float, default=0.5)
    p.add_argument("--hopping", choices=("sharc", "off"), default="sharc")
    p.add_argument("--decoherence", choices=("none", "edc"), default="none")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--provider", choices=PROVIDERS, default="analytical")
    p.add_argument("--work-dir", required=True)
    args = p.parse_args()
    defaults = {
        "sac": {"A": 0.01, "B": 1.6, "C": 0.005, "D": 1.0},
        "dac": {"A": 0.1, "B": 0.28, "C": 0.015, "D": 0.06, "E0": 0.05},
        "ecr": {"A": 6e-4, "B": 0.1, "C": 0.9},
    }
    traj = run(
        Path(args.sharc_home),
        args.python,
        ModelSpec(args.model, defaults[args.model]),
        args.x0,
        args.v0,
        args.steps,
        args.dt_au * AU_TIME_IN_FS,
        SharcOptions(hopping=args.hopping, decoherence=args.decoherence),
        args.seed,
        Path(args.work_dir),
        provider=args.provider,
    )
    for k in range(len(traj.steps)):
        print(
            f"{traj.steps[k]:6d} x={traj.q[k, 0]: .10f} v={traj.v[k, 0]: .10e} "
            f"|c|^2={abs(traj.c[k, 0]) ** 2:.12f} state={traj.active[k]}"
        )


if __name__ == "__main__":
    main()
