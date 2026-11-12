# P9a plan 1: diagnosis of the M4 trajectory losses — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure why 18 of the 20 M4 ethylene trajectories were lost and write a diagnosis report that labels each loss class as ours, method or settings, without changing the behaviour of `gqteashmd`.

**Architecture:** Report scripts in `tools/diagnosis/` (forensics of finished trajectories, replayed single points, our `TrajectoryRunner` on fixed initial conditions with setup variants) and one SHARC harness in `tools/sharc_compare/` (sharc.x + `SHARC_PYSCF.py` on the same initial conditions). Their helpers are unit-tested in the fast tier; the measurement runs are done by hand on the WSL native filesystem and their numbers go to JSON files and to a committed report.

**Tech Stack:** Python 3.12, NumPy, h5py, PySCF 2.14 (lazily, in the scripts), the existing `gqteashmd.trajectory`, `gqteashmd.analysis.energy`, `gqteashmd.electronic` APIs, SHARC 4.1 executed as a black box.

**Spec:** `docs/superpowers/specs/2026-11-12-p9a-trajectory-losses-design.md` (owner-approved 2026-11-12; sections 3 and 5 are this plan's scope).

## Global Constraints

- **No behaviour change of `gqteashmd`:** plan 1 modifies nothing under `src/gqteashmd/` (spec section 3). `tools/` is never imported by `gqteashmd`; the scripts may import `gqteashmd`, PySCF, h5py and SciPy.
- **Clean room:** SHARC 4.1 is only executed (`sharc.x`, `SHARC_PYSCF.py`) from `tools/sharc_compare/`; input keywords and file formats come from the existing harness and the public SHARC 4.1 manual; nothing in the SHARC installation is read.
- **Jump rule:** an energy jump is `|ΔE_tot| > 0.1 eV` between consecutive records (`JUMP_THRESHOLD_EV = 0.1`).
- **Kept rule (spec 4.3):** kept = outcome `finished` or `stopped_in_s0` **and** `max_jump_eV(e_total over records with grid_step <= n_total) <= 0.1` (`gqteashmd.analysis.energy.max_jump_eV`).
- **Seeds:** the seed-2026 initial conditions and random streams exactly as the M4 stage used them: trajectory k uses `GeneratorSource(task_rng(2026, "dynamics", k))` with `k` = the `index` of `trajectories.json`.
- **Measurements:** on the WSL native filesystem under `~/p9a/`; the M4 projects `~/p8_m4/run` and `~/p8_m4b/run` are read only, never written.
- **Commands:** everything runs in WSL: `wsl -d OracleLinux_9_5 -- bash -lc 'cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && conda activate gqteashmd-dev && <command>'`. Git from WSL only; never `--no-verify`. Commit subjects are conventional (`feat:`, `test:`, `docs:`, `fix:`); every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (or the trailer of the model that wrote it).
- **Lint:** `ruff format . && ruff check .` (line length 100, rules E, F, W, I, B, UP, SIM); `mypy` stays clean on `src/` (unchanged).
- **Tests:** tools tests live in `tests/diagnosis/` (new) and `tests/sharc_compare/` and load the scripts with `importlib` (the pattern of `tests/sharc_compare/test_run_sharc_pyscf_dynamics.py:_load`); PySCF tests carry `@pytest.mark.pyscf`, live-SHARC tests `@pytest.mark.sharc` (skipped without `SHARC_HOME`/`SHARC_PYTHON`). Every new test must pass under `-W error`.
- **Private access ruling (P9a1-1):** `tools/diagnosis/replay_points.py` may read `PySCFBackend._live` (the last PySCF objects) for orbital diagnostics only; it never writes it. Nothing in `src/` changes to expose it.
- **Variant names** are `[a-z0-9_]+`; directories under `--out` are named after them.

## File structure

| File | Responsibility |
|---|---|
| `tools/diagnosis/__init__.py` | empty (marks the directory; scripts are still run by path) |
| `tools/diagnosis/forensics.py` | pure helpers on `steps.log` rows: jumps, hops, onset, failed geometry, JSON safety; `trajectory_report`, `project_report`, `find_stage_dir` |
| `tools/diagnosis/trajectory_forensics.py` | CLI: forensics JSON of one project |
| `tools/diagnosis/project_inputs.py` | the fixed initial conditions and settings of a finished project |
| `tools/diagnosis/run_variants.py` | our runner on those initial conditions for a list of setup variants |
| `tools/diagnosis/variants_p9a.toml` | the variant list of spec 3.3 |
| `tools/diagnosis/replay_points.py` | replayed single points around each failure, method variants, determinism, convergence trials |
| `tools/sharc_compare/run_sharc_pyscf_dynamics.py` | modified: harness minors, hopping/decoherence/active-space parameters, explicit coupling/provider, scaled timeout, `check=False` |
| `tools/sharc_compare/run_sharc_pyscf_ensemble.py` | sharc.x + SHARC_PYSCF.py on a project's initial conditions |
| `tests/diagnosis/{__init__,conftest,test_forensics,test_project_inputs,test_run_variants,test_replay_points,test_sharc_ensemble}.py` | tests |
| `docs/superpowers/reports/2026-11-12-p9a-diagnosis.md` | the diagnosis report (Task 8) |

---

### Task 1: Forensics of finished trajectories

**Files:**
- Create: `tools/diagnosis/__init__.py`, `tools/diagnosis/forensics.py`, `tools/diagnosis/trajectory_forensics.py`
- Create: `tests/diagnosis/__init__.py`, `tests/diagnosis/conftest.py`, `tests/diagnosis/test_forensics.py`

**Interfaces:**
- Consumes: `gqteashmd.trajectory.store.log_dtype`, `StepLog`, `read_output_h5`, `write_output_h5`; `gqteashmd.core.units.AU_TIME_IN_FS`, `HARTREE_IN_EV`; `gqteashmd.util.fileio.read_json`.
- Produces (used by Tasks 2, 3, 5, 6):
  - `JUMP_THRESHOLD_EV: float = 0.1`
  - `ONSETS: tuple[str, ...] = ("convergence_failure", "failure_after_hop", "overlap_failure_without_hop", "jump_without_hop", "jump_after_hop", "none")`
  - `@dataclass(frozen=True) class Jump: record: int; time_fs: float; delta_e_total_eV: float; state: int; delta_e_state_eV: float; gap_eV: float; min_singular_value: float; rung_changed: bool; depth: int; active: int`
  - `@dataclass(frozen=True) class Hop: record: int; time_fs: float; from_state: int; to_state: int; gap_eV: float`
  - `energy_jumps(time_fs: np.ndarray, e_total_Eh: np.ndarray, energies_Eh: np.ndarray, *, threshold_eV: float = JUMP_THRESHOLD_EV, min_singular_value: np.ndarray | None = None, rung_changed: np.ndarray | None = None, depth: np.ndarray | None = None, active: np.ndarray | None = None) -> list[Jump]` (array form, reused for SHARC data in Task 5)
  - `log_jumps(rows: np.ndarray, threshold_eV: float = JUMP_THRESHOLD_EV) -> list[Jump]` (steps.log rows)
  - `log_hops(rows: np.ndarray) -> list[Hop]`
  - `classify_onset(outcome: str, jumps: Sequence[Jump], hops: Sequence[Hop]) -> str`
  - `failed_geometry(q: np.ndarray, v: np.ndarray, gradient: np.ndarray, masses_me: np.ndarray, dt_au: float) -> np.ndarray`
  - `electronic_log_tail(text: str, n: int = 20) -> list[str]`
  - `jsonable(value: Any) -> Any` (NaN/inf -> None, NumPy -> Python, dataclasses -> dict)
  - `find_stage_dir(project_dir: Path, name: str) -> Path`
  - `read_rows(traj_dir: Path) -> tuple[np.ndarray, list[str], int]` (rows of steps.log, state labels, n_atoms)
  - `trajectory_report(traj_dir: Path, *, threshold_eV: float = JUMP_THRESHOLD_EV) -> dict[str, Any]`
  - `project_report(project_dir: Path, *, threshold_eV: float = JUMP_THRESHOLD_EV) -> dict[str, Any]`

- [ ] **Step 1: Write the test helpers and the failing tests**

`tests/diagnosis/__init__.py`: empty. `tests/diagnosis/conftest.py`:

```python
"""Helpers shared by the tools/diagnosis tests: load a script by path, write a trajectory."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from gqteashmd.core.units import AU_TIME_IN_FS, HARTREE_IN_EV
from gqteashmd.trajectory.store import StepLog, log_dtype, log_row, write_output_h5

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def load_tool(subdir: str, name: str):
    """Import tools/<subdir>/<name>.py once (shared module object, like tests/sharc_compare)."""
    key = f"{subdir}_{name}"
    if key in sys.modules:
        return sys.modules[key]
    path = TOOLS / subdir / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    sys.modules.setdefault(name, module)  # sibling imports (e.g. `import forensics`)
    spec.loader.exec_module(module)
    return module


def make_rows(e_total_eV, energies_eV, *, active=None, hop_to=None, dt_fs=0.5, sv=None):
    """steps.log rows for a 2-atom, n-state toy trajectory (energies given in eV)."""
    e_total_eV = np.asarray(e_total_eV, dtype=float)
    energies_eV = np.asarray(energies_eV, dtype=float)
    m, n = energies_eV.shape
    dtype = log_dtype(n, 6)
    rows = np.zeros(m, dtype=dtype)
    for k in range(m):
        rows[k] = log_row(
            dtype,
            step=k,
            grid_step=k,
            ticks=0,
            depth=0,
            time_au=k * dt_fs / AU_TIME_IN_FS,
            active=1 if active is None else active[k],
            hop_from=-1 if hop_to is None or hop_to[k] < 0 else (active[k - 1]),
            hop_to=-1 if hop_to is None else hop_to[k],
            frustrated=0,
            rung_changed=0,
            rejections=0,
            random_number=0.5,
            min_singular_value=0.99 if sv is None else sv[k],
            e_total=e_total_eV[k] / HARTREE_IN_EV,
            e_kinetic=0.0,
            energies=energies_eV[k] / HARTREE_IN_EV,
            c_real=np.eye(n)[1],
            c_imag=np.zeros(n),
            probabilities=np.zeros(n),
            q=np.arange(6, dtype=float),
            v=np.zeros(6),
        )
    return rows


def write_trajectory(traj_dir: Path, rows, *, outcome: str, message: str | None = None):
    """Write steps.log, output.h5 and a worker-format result.json for `rows`."""
    traj_dir.mkdir(parents=True, exist_ok=True)
    n = rows["energies"].shape[1]
    log = StepLog(traj_dir / "steps.log", n, 6)
    for row in rows:
        log.append(row)
    labels = [f"S{i}" for i in range(n)]
    write_output_h5(
        traj_dir / "output.h5",
        rows,
        labels=labels,
        n_atoms=2,
        output_every=1,
        attrs={
            "state_labels": labels,
            "output_every": 1,
            "dt_fs": 0.5,
            "outcome": outcome,
            "task": traj_dir.name,
            "initial_state": "S1",
        },
    )
    result = {
        "outcome": outcome,
        "t_final_fs": float(rows["time_au"][-1] * AU_TIME_IN_FS),
        "message": message,
    }
    (traj_dir / "result.json").write_text(json.dumps({"status": "ok", "result": result}))
```

Note: `write_output_h5` may require more `attrs` than listed (check `src/gqteashmd/trajectory/store.py:write_output_h5` and `_REQUIRED_ATTRS`); pass what it requires.

`tests/diagnosis/test_forensics.py`:

```python
import json

import numpy as np
import pytest

from tests.diagnosis.conftest import load_tool, make_rows, write_trajectory

fx = load_tool("diagnosis", "forensics")


def test_jump_without_hop_names_the_state_and_the_gap():
    e_total = [0.0, 0.01, 0.02, 0.32, 0.33]  # jump of 0.30 eV into record 3
    energies = [[0.0, 9.0], [0.0, 9.0], [0.0, 9.0], [0.05, 9.3], [0.05, 9.3]]
    rows = make_rows(e_total, energies)
    jumps = fx.log_jumps(rows)
    assert [j.record for j in jumps] == [3]
    j = jumps[0]
    assert j.delta_e_total_eV == pytest.approx(0.30, abs=1e-9)
    assert j.state == 1 and j.delta_e_state_eV == pytest.approx(0.3, abs=1e-9)
    assert j.gap_eV == pytest.approx(9.25, abs=1e-9)
    assert j.time_fs == pytest.approx(1.5, abs=1e-9)
    assert fx.classify_onset("energy_drift", jumps, fx.log_hops(rows)) == "jump_without_hop"


def test_threshold_is_strict():
    rows = make_rows([0.0, 0.1], [[0.0, 9.0], [0.0, 9.1]])
    assert fx.log_jumps(rows, threshold_eV=0.1 + 1e-9) == []
    assert len(fx.log_jumps(rows, threshold_eV=0.05)) == 1


def test_overlap_failure_after_a_hop():
    active = [1, 1, 0, 0]
    hop_to = [-1, -1, 0, -1]
    rows = make_rows([0, 0, 0, 0], [[0, 2], [0, 1], [0, 0.5], [0, 0.4]], active=active, hop_to=hop_to)
    hops = fx.log_hops(rows)
    assert [(h.record, h.from_state, h.to_state) for h in hops] == [(2, 1, 0)]
    assert hops[0].gap_eV == pytest.approx(0.5, abs=1e-9)
    assert fx.classify_onset("overlap_failure", [], hops) == "failure_after_hop"
    assert fx.classify_onset("overlap_failure", [], []) == "overlap_failure_without_hop"


def test_onset_order():
    hop = fx.Hop(2, 1.0, 1, 0, 0.3)
    early = fx.Jump(1, 0.5, 0.2, 1, 0.2, 5.0, 0.9, False, 0, 1)
    late = fx.Jump(3, 1.5, 0.2, 0, 0.2, 0.3, 0.9, False, 0, 0)
    assert fx.classify_onset("electronic_failure", [early], [hop]) == "convergence_failure"
    assert fx.classify_onset("energy_drift", [early, late], [hop]) == "jump_without_hop"
    assert fx.classify_onset("finished", [late], [hop]) == "jump_after_hop"
    assert fx.classify_onset("finished", [], [hop]) == "none"
    assert set(fx.ONSETS) >= {"none", "jump_after_hop"}


def test_failed_geometry_is_the_verlet_position_update():
    q = np.array([0.0, 1.0])
    v = np.array([0.5, -0.5])
    g = np.array([2.0, -4.0])
    m = np.array([2.0, 4.0])
    out = fx.failed_geometry(q, v, g, m, 0.1)
    np.testing.assert_allclose(out, q + v * 0.1 - 0.5 * (g / m) * 0.01)


def test_electronic_log_tail_keeps_the_convergence_lines():
    text = "a\nSCF not converged\nb\nCASSCF failed\nc\nmacro iter 5 converged\n"
    assert fx.electronic_log_tail(text, n=2) == ["CASSCF failed", "macro iter 5 converged"]


def test_jsonable_replaces_non_finite_values():
    data = fx.jsonable({"a": np.float64("nan"), "b": [np.int64(3), np.inf], "c": fx.Hop(0, 0.0, 1, 0, 0.1)})
    assert data == {"a": None, "b": [3, None], "c": {"record": 0, "time_fs": 0.0, "from_state": 1, "to_state": 0, "gap_eV": 0.1}}
    json.dumps(data, allow_nan=False)


def test_trajectory_and_project_report(tmp_path):
    project = tmp_path / "run"
    tasks = project / "06_dynamics" / "tasks"
    rows = make_rows([0.0, 0.0, 0.5], [[0.0, 9.0], [0.0, 9.0], [0.0, 9.5]])
    write_trajectory(tasks / "traj_00000", rows, outcome="energy_drift", message="drift")
    write_trajectory(tasks / "traj_00001", make_rows([0, 0], [[0, 9], [0, 9]]), outcome="finished")
    (tasks / "traj_00000" / "electronic.log").write_text("SCF not converged\n")
    report = fx.trajectory_report(tasks / "traj_00000")
    assert report["task"] == "traj_00000"
    assert report["outcome"] == "energy_drift"
    assert report["onset"] == "jump_without_hop"
    assert report["n_records"] == 3
    assert report["jumps"][0]["record"] == 2
    assert report["electronic_log"] == ["SCF not converged"]
    assert len(report["last_records"]) == 3
    whole = fx.project_report(project)
    assert whole["n_trajectories"] == 2
    assert whole["onsets"]["jump_without_hop"] == 1 and whole["onsets"]["none"] == 1
    assert whole["outcomes"] == {"energy_drift": 1, "finished": 1}
    json.dumps(fx.jsonable(whole), allow_nan=False)


def test_find_stage_dir(tmp_path):
    (tmp_path / "03_sampling").mkdir()
    assert fx.find_stage_dir(tmp_path, "sampling") == tmp_path / "03_sampling"
    with pytest.raises(ValueError, match="dynamics"):
        fx.find_stage_dir(tmp_path, "dynamics")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/diagnosis/test_forensics.py -q -W error`
Expected: FAIL (`FileNotFoundError` / module not found for `tools/diagnosis/forensics.py`).

- [ ] **Step 3: Implement `tools/diagnosis/forensics.py`**

```python
"""Forensics of finished gqteashmd trajectories (P9a spec section 3.1).

Report helpers only: they read steps.log, output.h5, result.json and electronic.log of a
trajectory directory and change nothing. Energies in the files are in hartree; every number
returned here is in eV or fs.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.core.units import AU_TIME_IN_FS, HARTREE_IN_EV
from gqteashmd.trajectory.store import StepLog, read_output_h5

JUMP_THRESHOLD_EV = 0.1
ONSETS = (
    "convergence_failure",
    "failure_after_hop",
    "overlap_failure_without_hop",
    "jump_without_hop",
    "jump_after_hop",
    "none",
)
LAST_RECORDS = 5
_LOG_KEYWORDS = re.compile(r"converge|casscf|scf|fail|rung", re.IGNORECASE)


@dataclass(frozen=True)
class Jump:
    record: int
    time_fs: float
    delta_e_total_eV: float
    state: int
    delta_e_state_eV: float
    gap_eV: float
    min_singular_value: float
    rung_changed: bool
    depth: int
    active: int


@dataclass(frozen=True)
class Hop:
    record: int
    time_fs: float
    from_state: int
    to_state: int
    gap_eV: float


def energy_jumps(
    time_fs: np.ndarray,
    e_total_Eh: np.ndarray,
    energies_Eh: np.ndarray,
    *,
    threshold_eV: float = JUMP_THRESHOLD_EV,
    min_singular_value: np.ndarray | None = None,
    rung_changed: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    active: np.ndarray | None = None,
) -> list[Jump]:
    """Consecutive records whose |E_tot difference| exceeds threshold_eV (strictly)."""
    e = np.asarray(e_total_Eh, dtype=float)
    energies = np.asarray(energies_Eh, dtype=float)
    if e.size < 2:
        return []
    de = np.diff(e) * HARTREE_IN_EV
    out: list[Jump] = []
    for k in np.flatnonzero(np.abs(de) > threshold_eV):
        i = int(k) + 1
        d_state = (energies[i] - energies[i - 1]) * HARTREE_IN_EV
        s = int(np.argmax(np.abs(d_state)))
        gap = (energies[i, 1] - energies[i, 0]) * HARTREE_IN_EV if energies.shape[1] > 1 else math.nan
        out.append(
            Jump(
                record=i,
                time_fs=float(time_fs[i]),
                delta_e_total_eV=float(de[k]),
                state=s,
                delta_e_state_eV=float(d_state[s]),
                gap_eV=float(gap),
                min_singular_value=math.nan if min_singular_value is None else float(min_singular_value[i]),
                rung_changed=False if rung_changed is None else bool(rung_changed[i]),
                depth=0 if depth is None else int(depth[i]),
                active=-1 if active is None else int(active[i]),
            )
        )
    return out


def log_jumps(rows: np.ndarray, threshold_eV: float = JUMP_THRESHOLD_EV) -> list[Jump]:
    return energy_jumps(
        rows["time_au"] * AU_TIME_IN_FS,
        rows["e_total"],
        rows["energies"],
        threshold_eV=threshold_eV,
        min_singular_value=rows["min_singular_value"],
        rung_changed=rows["rung_changed"],
        depth=rows["depth"],
        active=rows["active"],
    )


def log_hops(rows: np.ndarray) -> list[Hop]:
    out: list[Hop] = []
    for i in np.flatnonzero(rows["hop_to"] >= 0):
        i = int(i)
        a, b = int(rows["hop_from"][i]), int(rows["hop_to"][i])
        gap = abs(float(rows["energies"][i][b] - rows["energies"][i][a])) * HARTREE_IN_EV
        out.append(Hop(i, float(rows["time_au"][i] * AU_TIME_IN_FS), a, b, gap))
    return out


def classify_onset(outcome: str, jumps: Sequence[Jump], hops: Sequence[Hop]) -> str:
    """One label of ONSETS: what happened first on the way to the trajectory's end."""
    if outcome == "electronic_failure":
        return "convergence_failure"
    if outcome == "overlap_failure":
        return "failure_after_hop" if hops else "overlap_failure_without_hop"
    if jumps:
        first_hop = hops[0].time_fs if hops else math.inf
        return "jump_without_hop" if jumps[0].time_fs < first_hop else "jump_after_hop"
    return "none"


def failed_geometry(
    q: np.ndarray, v: np.ndarray, gradient: np.ndarray, masses_me: np.ndarray, dt_au: float
) -> np.ndarray:
    """The velocity-Verlet position of the step after (q, v): q + v dt - g/(2m) dt^2."""
    q, v, g, m = (np.asarray(x, dtype=float) for x in (q, v, gradient, masses_me))
    return q + v * dt_au - 0.5 * (g / m) * dt_au**2


def electronic_log_tail(text: str, n: int = 20) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if _LOG_KEYWORDS.search(line)]
    return lines[-n:]


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def find_stage_dir(project_dir: Path, name: str) -> Path:
    found = sorted(p for p in Path(project_dir).glob(f"[0-9][0-9]_{name}") if p.is_dir())
    if len(found) != 1:
        raise ValueError(f"{project_dir} has {len(found)} directories for stage {name!r}")
    return found[0]


def read_rows(traj_dir: Path) -> tuple[np.ndarray, list[str], int]:
    data = read_output_h5(Path(traj_dir) / "output.h5")
    labels = list(data.labels)
    n_atoms = int(data["positions"].shape[1])
    rows = StepLog(Path(traj_dir) / "steps.log", len(labels), 3 * n_atoms).read()
    return rows, labels, n_atoms


def _result(traj_dir: Path) -> dict[str, Any]:
    path = Path(traj_dir) / "result.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return dict(raw.get("result", raw)) if isinstance(raw, dict) else {}


def trajectory_report(traj_dir: Path, *, threshold_eV: float = JUMP_THRESHOLD_EV) -> dict[str, Any]:
    traj_dir = Path(traj_dir)
    rows, labels, _ = read_rows(traj_dir)
    result = _result(traj_dir)
    outcome = str(result.get("outcome", read_output_h5(traj_dir / "output.h5").outcome))
    jumps = log_jumps(rows, threshold_eV)
    hops = log_hops(rows)
    log_path = traj_dir / "electronic.log"
    tail = electronic_log_tail(log_path.read_text(errors="replace")) if log_path.is_file() else []
    last = [
        {
            "record": int(i),
            "time_fs": float(rows["time_au"][i] * AU_TIME_IN_FS),
            "depth": int(rows["depth"][i]),
            "active": int(rows["active"][i]),
            "min_singular_value": float(rows["min_singular_value"][i]),
            "rung_changed": bool(rows["rung_changed"][i]),
            "energies_eV": (rows["energies"][i] * HARTREE_IN_EV).tolist(),
            "e_total_eV": float(rows["e_total"][i] * HARTREE_IN_EV),
        }
        for i in range(max(0, rows.size - LAST_RECORDS), rows.size)
    ]
    return {
        "task": traj_dir.name,
        "labels": labels,
        "outcome": outcome,
        "t_final_fs": result.get("t_final_fs"),
        "message": result.get("message"),
        "n_records": int(rows.size),
        "jumps": jumps,
        "hops": hops,
        "onset": classify_onset(outcome, jumps, hops),
        "last_records": last,
        "electronic_log": tail,
    }


def project_report(project_dir: Path, *, threshold_eV: float = JUMP_THRESHOLD_EV) -> dict[str, Any]:
    tasks = find_stage_dir(project_dir, "dynamics") / "tasks"
    reports = [
        trajectory_report(d, threshold_eV=threshold_eV)
        for d in sorted(tasks.glob("traj_*"))
        if (d / "output.h5").is_file()
    ]
    onsets = Counter(r["onset"] for r in reports)
    return {
        "project": str(project_dir),
        "threshold_eV": threshold_eV,
        "n_trajectories": len(reports),
        "outcomes": dict(sorted(Counter(r["outcome"] for r in reports).items())),
        "onsets": {o: int(onsets.get(o, 0)) for o in ONSETS},
        "trajectories": reports,
    }
```

- [ ] **Step 4: Implement the CLI `tools/diagnosis/trajectory_forensics.py`**

```python
"""Forensics JSON of one gqteashmd project (P9a spec 3.1).

Example (WSL):
  python tools/diagnosis/trajectory_forensics.py ~/p8_m4/run --out ~/p9a/forensics_m4.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forensics import JUMP_THRESHOLD_EV, jsonable, project_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("project", type=Path, help="the project directory (holds 06_dynamics)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold-eV", type=float, default=JUMP_THRESHOLD_EV)
    args = parser.parse_args(argv)
    report = project_report(args.project.expanduser(), threshold_eV=args.threshold_eV)
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsonable(report), indent=1, allow_nan=False) + "\n")
    print(f"{report['n_trajectories']} trajectories; onsets {report['onsets']}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Add to `tests/diagnosis/test_forensics.py`:

```python
def test_cli_writes_strict_json(tmp_path):
    cli = load_tool("diagnosis", "trajectory_forensics")
    project = tmp_path / "run"
    write_trajectory(project / "06_dynamics" / "tasks" / "traj_00000", make_rows([0, 0], [[0, 9], [0, 9]]), outcome="finished")
    out = tmp_path / "f.json"
    assert cli.main([str(project), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["n_trajectories"] == 1
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/diagnosis -q -W error`, then `ruff format . && ruff check .`
Expected: all pass, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add tools/diagnosis tests/diagnosis
git commit -m "feat: add trajectory forensics for the P9a diagnosis" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Fixed initial conditions and our runner on setup variants

**Files:**
- Create: `tools/diagnosis/project_inputs.py`, `tools/diagnosis/run_variants.py`, `tools/diagnosis/variants_p9a.toml`
- Create: `tests/diagnosis/test_project_inputs.py`, `tests/diagnosis/test_run_variants.py`

**Interfaces:**
- Consumes: Task 1 `forensics.find_stage_dir`, `forensics.trajectory_report`, `forensics.jsonable`, `forensics.JUMP_THRESHOLD_EV`; `gqteashmd.config.loader.load_snapshot` (returns `LoadedConfig` with `.config: ShmdConfig`); `gqteashmd.config.schema.ElectronicSection`, `DynamicsStageSection`, `ShmdConfig`; `gqteashmd.stages.geometry.prepared_molecule`; `gqteashmd.vib.artifacts.read_samples_h5` (keys `symbols`, `masses_amu`, `positions_bohr`, `velocities_au`, `n_samples`); `gqteashmd.excite.artifacts.read_trajectories_json` (`SelectedTrajectory.task_id`, `index`, `sample`, `state`, `rng`); `gqteashmd.stages.dynamics.run_settings`; `gqteashmd.trajectory.runner.TrajectoryRunner`, `InitialConditions`, `OUTPUT_FILE`; `gqteashmd.trajectory.model.BackendModel`; `gqteashmd.dynamics.randomness.GeneratorSource`; `gqteashmd.util.seeds.task_rng`; `gqteashmd.analysis.energy.max_jump_eV`; `gqteashmd.trajectory.store.read_output_h5`; `gqteashmd.electronic.pyscf_backend.PySCFBackend`; for tests `gqteashmd.testing.fake_backend.ModelBackend`, `gqteashmd.dynamics.lvc.LVCModel`, `gqteashmd.testing.canned_lvc.write_canned_molecule`, `write_canned_samples`, `write_canned_trajectories`.
- Produces (used by Tasks 3, 5, 7):
  - `project_inputs.TrajectoryStart` (frozen: `task_id: str, index: int, sample: int, state: str, q: np.ndarray, v: np.ndarray`; flat, bohr and bohr per atomic time unit)
  - `project_inputs.ProjectInputs` (frozen: `project_dir: Path, config: ShmdConfig, symbols: tuple[str, ...], charge: int, multiplicity: int, masses_amu: np.ndarray, starts: tuple[TrajectoryStart, ...]`)
  - `project_inputs.load_project_inputs(project_dir: Path) -> ProjectInputs`
  - `run_variants.Variant` (frozen: `name: str, electronic: Mapping[str, Any], dynamics: Mapping[str, Any]`)
  - `run_variants.deep_merge(base: Mapping, overrides: Mapping) -> dict`
  - `run_variants.read_variants(path: Path) -> list[Variant]`
  - `run_variants.variant_sections(config: ShmdConfig, variant: Variant) -> tuple[ElectronicSection, DynamicsStageSection]`
  - `run_variants.is_kept(outcome: str, max_jump: float) -> bool`
  - `run_variants.run_variant(inputs: ProjectInputs, variant: Variant, out_dir: Path, *, workers: int, threads: int, memory_mb: int, backend: str = "pyscf", only: Sequence[str] | None = None) -> dict[str, Any]` (writes `out_dir/<variant.name>/<task_id>/` and `out_dir/<variant.name>/summary.json`; returns the summary)

- [ ] **Step 1: Write the failing tests**

`tests/diagnosis/test_project_inputs.py` (builds a canned LVC project; the input text below must validate — compare with `tests/stages/test_dynamics_pipeline.py` if a field is missing):

```python
from pathlib import Path

import numpy as np
import pytest

from gqteashmd.config.loader import load_config, write_snapshot
from gqteashmd.dynamics.lvc import LVCModel
from gqteashmd.testing.canned_lvc import (
    write_canned_molecule,
    write_canned_samples,
    write_canned_trajectories,
)
from gqteashmd.vib.artifacts import read_samples_h5
from tests.diagnosis.conftest import load_tool

LVC = Path(__file__).resolve().parents[1] / "data" / "lvc" / "lvc_singlets.template"

INPUT = """
[project]
name = "canned"
directory = "run"
seed = {seed}

[molecule]
geometry = "mol.xyz"
charge = 0
multiplicity = 1

[resources]
total_cores = 2
total_memory_mb = 2000
max_parallel_tasks = 1
threads_per_task = 1
memory_mb_per_task = 500

[electronic]
method = "casscf"
basis = "sto-3g"
singlets = {singlets}

[electronic.casscf]
ncas = 2
nelecas = 2

[dynamics]
tmax_fs = 2.0
dt_fs = 0.5
"""


def make_project(tmp_path: Path, seed: int = 7, n_traj: int = 2) -> Path:
    """A finished-upstream project from the canned LVC model (no PySCF)."""
    model = LVCModel.from_template(LVC, spin_orbit=False)
    singlets = sum(1 for s in model.labels if str(s).startswith("S"))
    (tmp_path / "mol.xyz").write_text("1\n\nH 0 0 0\n")
    (tmp_path / "input.toml").write_text(INPUT.format(seed=seed, singlets=singlets))
    project = tmp_path / "run"
    for d in ("00_prepare", "03_sampling", "05_selection", ".gqteashmd"):
        (project / d).mkdir(parents=True)
    write_snapshot(load_config(tmp_path / "input.toml"), project / ".gqteashmd" / "config.json")
    write_canned_molecule(project / "00_prepare", model)
    write_canned_samples(project / "03_sampling", model, n_samples=5)
    write_canned_trajectories(project / "05_selection", [(k + 1, "S1") for k in range(n_traj)], seed)
    return project


def test_load_project_inputs(tmp_path):
    pi = load_tool("diagnosis", "project_inputs")
    project = make_project(tmp_path)
    inputs = pi.load_project_inputs(project)
    samples = read_samples_h5(project / "03_sampling" / "samples.h5")
    assert inputs.config.project.seed == 7
    assert len(inputs.starts) == 2
    first = inputs.starts[0]
    assert (first.task_id, first.index, first.sample, first.state) == ("traj_00000", 0, 1, "S1")
    np.testing.assert_array_equal(first.q, samples["positions_bohr"][0].reshape(-1))
    np.testing.assert_array_equal(first.v, samples["velocities_au"][0].reshape(-1))
    np.testing.assert_array_equal(inputs.masses_amu, samples["masses_amu"])
    assert inputs.symbols == tuple(samples["symbols"])


def test_load_project_inputs_refuses_a_foreign_rng_stream(tmp_path):
    pi = load_tool("diagnosis", "project_inputs")
    project = make_project(tmp_path)
    path = project / "05_selection" / "trajectories.json"
    path.write_text(path.read_text().replace('"dynamics"', '"other"', 1))
    with pytest.raises(ValueError, match="random stream"):
        pi.load_project_inputs(project)
```

Note: `write_canned_trajectories` / `write_canned_molecule` signatures are in `src/gqteashmd/testing/canned_lvc.py`; if the molecule written there has more atoms than `mol.xyz`, that is fine (the tools read `00_prepare/molecule.json`, never the input geometry). If `load_config` refuses the config because `singlets` exceeds the CAS(2,2) singlet count (Weyl count 3), set `ncas`/`nelecas` so the count fits (e.g. `ncas = 4`, `nelecas = 4`); the LVC backend ignores `[electronic]`.

`tests/diagnosis/test_run_variants.py`:

```python
import json

import numpy as np

from gqteashmd.config.schema import DynamicsStageSection
from gqteashmd.dynamics.lvc import LVCModel
from gqteashmd.dynamics.randomness import GeneratorSource
from gqteashmd.core.units import DALTON_IN_ME
from gqteashmd.stages.dynamics import run_settings
from gqteashmd.testing.fake_backend import ModelBackend
from gqteashmd.trajectory.model import BackendModel
from gqteashmd.trajectory.runner import InitialConditions, TrajectoryRunner
from gqteashmd.trajectory.store import StepLog
from gqteashmd.util.seeds import task_rng
from tests.diagnosis.conftest import load_tool
from tests.diagnosis.test_project_inputs import LVC, make_project

rv = load_tool("diagnosis", "run_variants")
pi = load_tool("diagnosis", "project_inputs")


def test_deep_merge_and_variant_sections(tmp_path):
    assert rv.deep_merge({"a": 1, "c": {"x": 1, "y": 2}}, {"c": {"y": 3}, "d": 4}) == {
        "a": 1,
        "c": {"x": 1, "y": 3},
        "d": 4,
    }
    inputs = pi.load_project_inputs(make_project(tmp_path))
    variant = rv.Variant("dt_half", {"casscf": {"conv_tol_grad": 1e-5}}, {"dt_fs": 0.25})
    electronic, dynamics = rv.variant_sections(inputs.config, variant)
    assert electronic.casscf.conv_tol_grad == 1e-5 and electronic.casscf.ncas == inputs.config.electronic.casscf.ncas
    assert dynamics.dt_fs == 0.25 and dynamics.tmax_fs == inputs.config.dynamics.tmax_fs


def test_read_variants(tmp_path):
    path = tmp_path / "v.toml"
    path.write_text(
        '[[variant]]\nname = "base"\n\n[[variant]]\nname = "sa3"\n[variant.electronic]\nsinglets = 3\n'
        '[variant.dynamics]\ndt_fs = 0.25\n'
    )
    variants = rv.read_variants(path)
    assert [v.name for v in variants] == ["base", "sa3"]
    assert dict(variants[1].electronic) == {"singlets": 3} and dict(variants[1].dynamics) == {"dt_fs": 0.25}
    assert dict(variants[0].electronic) == {} and dict(variants[0].dynamics) == {}


def test_read_variants_refuses_bad_names(tmp_path):
    import pytest

    path = tmp_path / "v.toml"
    path.write_text('[[variant]]\nname = "Bad Name"\n')
    with pytest.raises(ValueError, match="name"):
        rv.read_variants(path)


def test_the_kept_rule():
    assert rv.is_kept("finished", 0.1)
    assert rv.is_kept("stopped_in_s0", 0.0)
    assert not rv.is_kept("finished", 0.1000001)
    assert not rv.is_kept("energy_drift", 0.0)


def test_run_variant_reproduces_the_stage_stream(tmp_path):
    """The variant runner gives, bit for bit, the steps.log of a direct runner with the
    stage's random stream task_rng(seed, "dynamics", index)."""
    project = make_project(tmp_path)
    inputs = pi.load_project_inputs(project)
    out = tmp_path / "out"
    summary = rv.run_variant(
        inputs, rv.Variant("base", {}, {}), out, workers=1, threads=1, memory_mb=500, backend=f"lvc:{LVC}"
    )
    assert summary["n_trajectories"] == 2
    assert sum(summary["outcomes"].values()) == 2
    assert 0 <= summary["n_kept"] <= 2
    assert (out / "base" / "summary.json").is_file()
    json.loads((out / "base" / "summary.json").read_text())
    row = summary["trajectories"][0]
    assert row["task"] == "traj_00000" and row["onset"] in rv.ONSETS

    # direct runner with the same stream
    model = LVCModel.from_template(LVC, spin_orbit=False)
    start = inputs.starts[0]
    masses = np.repeat(inputs.masses_amu * DALTON_IN_ME, 3)
    section = DynamicsStageSection.model_validate(inputs.config.dynamics.model_dump())
    bm = BackendModel(ModelBackend(model), masses, coupling=section.coupling)
    direct = tmp_path / "direct"
    TrajectoryRunner(bm, run_settings(section), direct).run(
        InitialConditions(start.q, start.v, bm.labels.index(bm.labels[1])),
        GeneratorSource(task_rng(inputs.config.project.seed, "dynamics", start.index)),
    )
    n, f = bm.n_states, bm.n_dof
    a = StepLog(out / "base" / "traj_00000" / "steps.log", n, f).read()
    b = StepLog(direct / "steps.log", n, f).read()
    assert a.tobytes() == b.tobytes()


def test_run_variant_only_runs_the_named_tasks(tmp_path):
    inputs = pi.load_project_inputs(make_project(tmp_path))
    summary = rv.run_variant(
        inputs, rv.Variant("base", {}, {}), tmp_path / "out", workers=1, threads=1,
        memory_mb=500, backend=f"lvc:{LVC}", only=["traj_00001"],
    )
    assert [r["task"] for r in summary["trajectories"]] == ["traj_00001"]
```

(The direct runner's initial active state: use `StepLog`/the label `S1` exactly as `run_job` does — `bm.labels.index(StateLabel.parse("S1"))`; replace `bm.labels.index(bm.labels[1])` by that form if `labels[1]` is not S1 in the template.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/diagnosis/test_project_inputs.py tests/diagnosis/test_run_variants.py -q -W error`
Expected: FAIL (modules not found).

- [ ] **Step 3: Implement `tools/diagnosis/project_inputs.py`**

```python
"""The fixed initial conditions and settings of a gqteashmd project (P9a spec 3.3).

Reads the config snapshot, 00_prepare/molecule.json, <nn>_sampling/samples.h5 and
<nn>_selection/trajectories.json, exactly the inputs the dynamics stage reads.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gqteashmd.config.loader import load_snapshot
from gqteashmd.config.schema import ShmdConfig
from gqteashmd.excite.artifacts import read_trajectories_json
from gqteashmd.stages.geometry import prepared_molecule
from gqteashmd.vib.artifacts import read_samples_h5

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forensics import find_stage_dir  # noqa: E402


@dataclass(frozen=True)
class TrajectoryStart:
    task_id: str
    index: int
    sample: int
    state: str
    q: np.ndarray
    v: np.ndarray


@dataclass(frozen=True)
class ProjectInputs:
    project_dir: Path
    config: ShmdConfig
    symbols: tuple[str, ...]
    charge: int
    multiplicity: int
    masses_amu: np.ndarray
    starts: tuple[TrajectoryStart, ...]


def load_project_inputs(project_dir: Path) -> ProjectInputs:
    project_dir = Path(project_dir).expanduser()
    config = load_snapshot(project_dir / ".gqteashmd" / "config.json").config
    molecule, _ = prepared_molecule(find_stage_dir(project_dir, "prepare"))
    samples = read_samples_h5(find_stage_dir(project_dir, "sampling") / "samples.h5")
    selected = read_trajectories_json(find_stage_dir(project_dir, "selection") / "trajectories.json")
    if tuple(samples["symbols"]) != tuple(molecule.symbols):
        raise ValueError(f"{project_dir}: the atoms of samples.h5 differ from molecule.json")
    starts = []
    for t in selected:
        if dict(t.rng) != {"stage": "dynamics", "task_index": t.index}:
            raise ValueError(f"{t.task_id} has the random stream {dict(t.rng)!r}")
        if not 1 <= t.sample <= samples["n_samples"]:
            raise ValueError(f"{t.task_id}: sample {t.sample} is not in samples.h5")
        starts.append(
            TrajectoryStart(
                t.task_id,
                t.index,
                t.sample,
                t.state,
                np.asarray(samples["positions_bohr"][t.sample - 1], dtype=float).reshape(-1),
                np.asarray(samples["velocities_au"][t.sample - 1], dtype=float).reshape(-1),
            )
        )
    return ProjectInputs(
        project_dir,
        config,
        tuple(molecule.symbols),
        int(molecule.charge),
        int(molecule.multiplicity),
        np.asarray(samples["masses_amu"], dtype=float),
        tuple(starts),
    )
```

- [ ] **Step 4: Implement `tools/diagnosis/run_variants.py`**

```python
"""Our TrajectoryRunner on a project's fixed initial conditions, for setup variants (P9a 3.3).

The orchestrator is bypassed on purpose: a changed [electronic] section would change the S0
surface and so the samples. Each trajectory uses the stage's random stream
task_rng(project.seed, "dynamics", index), so the variant "base" reproduces the project.

Example (WSL):
  python tools/diagnosis/run_variants.py ~/p8_m4/run tools/diagnosis/variants_p9a.toml \\
      --out ~/p9a/variants --workers 8 --threads 2 --memory-mb 800 --variant base
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import sys
import time
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forensics import ONSETS, jsonable, trajectory_report  # noqa: E402
from project_inputs import ProjectInputs, load_project_inputs  # noqa: E402

from gqteashmd.config.schema import DynamicsStageSection, ElectronicSection, ShmdConfig  # noqa: E402

KEPT_OUTCOMES = ("finished", "stopped_in_s0")
KEPT_JUMP_EV = 0.1
_NAME = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class Variant:
    name: str
    electronic: Mapping[str, Any] = field(default_factory=dict)
    dynamics: Mapping[str, Any] = field(default_factory=dict)


def deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def read_variants(path: Path) -> list[Variant]:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for entry in data.get("variant", []):
        name = str(entry.get("name", ""))
        if not _NAME.fullmatch(name):
            raise ValueError(f"{path}: variant name {name!r} must match [a-z0-9_]+")
        unknown = set(entry) - {"name", "electronic", "dynamics", "description"}
        if unknown:
            raise ValueError(f"{path}: variant {name!r} has unknown keys {sorted(unknown)}")
        out.append(Variant(name, dict(entry.get("electronic", {})), dict(entry.get("dynamics", {}))))
    if len({v.name for v in out}) != len(out):
        raise ValueError(f"{path}: duplicate variant names")
    return out


def variant_sections(config: ShmdConfig, variant: Variant) -> tuple[ElectronicSection, DynamicsStageSection]:
    assert config.electronic is not None and config.dynamics is not None
    electronic = ElectronicSection.model_validate(
        deep_merge(config.electronic.model_dump(mode="json"), variant.electronic)
    )
    dynamics = DynamicsStageSection.model_validate(
        deep_merge(config.dynamics.model_dump(mode="json"), variant.dynamics)
    )
    return electronic, dynamics


def is_kept(outcome: str, max_jump: float) -> bool:
    return outcome in KEPT_OUTCOMES and max_jump <= KEPT_JUMP_EV


@dataclass(frozen=True)
class Job:
    directory: str
    task_id: str
    index: int
    sample: int
    state: str
    q: np.ndarray
    v: np.ndarray
    symbols: tuple[str, ...]
    charge: int
    multiplicity: int
    masses_amu: np.ndarray
    electronic: dict[str, Any]
    dynamics: dict[str, Any]
    seed: int
    threads: int
    memory_mb: int
    backend: str


def _backend(job: Job, directory: Path):
    if job.backend == "pyscf":
        from gqteashmd.electronic.base import MoleculeSpec
        from gqteashmd.electronic.pyscf_backend import PySCFBackend

        return PySCFBackend(
            MoleculeSpec(job.symbols, job.charge, job.multiplicity),
            ElectronicSection.model_validate(job.electronic),
            log_file=directory / "electronic.log",
            threads=job.threads,
            memory_mb=job.memory_mb,
            scratch_dir=directory / "scratch",
        )
    if job.backend.startswith("lvc:"):
        from gqteashmd.dynamics.lvc import LVCModel
        from gqteashmd.testing.fake_backend import ModelBackend

        return ModelBackend(LVCModel.from_template(Path(job.backend[4:]), spin_orbit=False))
    raise ValueError(f"unknown backend {job.backend!r}")


def run_job(job: Job) -> dict[str, Any]:
    """One trajectory, as DynamicsStage.run_task would run it (top level: picklable)."""
    from gqteashmd.analysis.energy import max_jump_eV
    from gqteashmd.core.states import StateLabel
    from gqteashmd.core.units import DALTON_IN_ME
    from gqteashmd.dynamics.randomness import GeneratorSource
    from gqteashmd.stages.dynamics import run_settings
    from gqteashmd.trajectory.model import BackendModel
    from gqteashmd.trajectory.runner import OUTPUT_FILE, InitialConditions, TrajectoryRunner
    from gqteashmd.trajectory.store import read_output_h5
    from gqteashmd.util.seeds import task_rng

    directory = Path(job.directory)
    directory.mkdir(parents=True, exist_ok=True)
    section = DynamicsStageSection.model_validate(job.dynamics)
    settings = run_settings(section)
    started = time.perf_counter()
    row: dict[str, Any] = {"task": job.task_id, "sample": job.sample, "index": job.index}
    try:
        model = BackendModel(
            _backend(job, directory),
            np.repeat(np.asarray(job.masses_amu) * DALTON_IN_ME, 3),
            coupling=section.coupling,
        )
        active = model.labels.index(StateLabel.parse(job.state))
        result = TrajectoryRunner(
            model,
            settings,
            directory,
            attrs={"symbols": list(job.symbols), "sample": job.sample, "initial_state": job.state, "task": job.task_id},
        ).run(
            InitialConditions(np.asarray(job.q), np.asarray(job.v), active),
            GeneratorSource(task_rng(job.seed, "dynamics", job.index)),
        )
    except ValueError as exc:  # what the stage turns into a non-retryable TaskError
        result = {"outcome": "error", "message": f"{type(exc).__name__}: {exc}"}
    result["wall_time_s"] = time.perf_counter() - started
    (directory / "result.json").write_text(json.dumps({"status": "ok", "result": jsonable(result)}))
    row.update(outcome=str(result["outcome"]), message=result.get("message"))
    for key in ("t_final_fs", "n_steps", "n_hops", "n_rejections", "n_flagged", "max_drift_eV"):
        row[key] = result.get(key)
    row["wall_time_s"] = result["wall_time_s"]
    row["s_per_step"] = result["wall_time_s"] / max(1, int(result.get("n_steps") or 0))
    output = directory / OUTPUT_FILE
    if output.is_file():
        data = read_output_h5(output)
        mask = np.asarray(data["grid_step"]) <= settings.n_grid_steps
        row["max_jump_eV"] = max_jump_eV(np.asarray(data["e_total"])[mask])
        forensic = trajectory_report(directory)
        row["onset"] = forensic["onset"]
        row["first_jump_fs"] = forensic["jumps"][0].time_fs if forensic["jumps"] else None
        row["first_hop_fs"] = forensic["hops"][0].time_fs if forensic["hops"] else None
    else:
        row.update(max_jump_eV=None, onset="none", first_jump_fs=None, first_hop_fs=None)
    row["kept"] = row["max_jump_eV"] is not None and is_kept(row["outcome"], row["max_jump_eV"])
    return row


def run_variant(
    inputs: ProjectInputs,
    variant: Variant,
    out_dir: Path,
    *,
    workers: int,
    threads: int,
    memory_mb: int,
    backend: str = "pyscf",
    only: Sequence[str] | None = None,
) -> dict[str, Any]:
    electronic, dynamics = variant_sections(inputs.config, variant)
    base = Path(out_dir).expanduser() / variant.name
    base.mkdir(parents=True, exist_ok=True)
    jobs = [
        Job(
            str(base / s.task_id), s.task_id, s.index, s.sample, s.state, s.q, s.v,
            inputs.symbols, inputs.charge, inputs.multiplicity, inputs.masses_amu,
            electronic.model_dump(mode="json"), dynamics.model_dump(mode="json"),
            inputs.config.project.seed, threads, memory_mb, backend,
        )
        for s in inputs.starts
        if only is None or s.task_id in only
    ]
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(threads)  # inherited by the spawned workers
    os.environ["PYSCF_MAX_MEMORY"] = str(memory_mb)
    started = time.perf_counter()
    if workers <= 1:
        rows = [run_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            rows = list(pool.map(run_job, jobs))
    rows.sort(key=lambda r: r["task"])
    onsets = Counter(r["onset"] for r in rows)
    summary = {
        "variant": variant.name,
        "electronic": electronic.model_dump(mode="json"),
        "dynamics": dynamics.model_dump(mode="json"),
        "project": str(inputs.project_dir),
        "seed": inputs.config.project.seed,
        "backend": backend,
        "n_trajectories": len(rows),
        "outcomes": dict(sorted(Counter(r["outcome"] for r in rows).items())),
        "onsets": {o: int(onsets.get(o, 0)) for o in ONSETS},
        "n_kept": sum(1 for r in rows if r["kept"]),
        "wall_time_s": time.perf_counter() - started,
        "trajectories": rows,
    }
    (base / "summary.json").write_text(json.dumps(jsonable(summary), indent=1, allow_nan=False) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("project", type=Path)
    parser.add_argument("variants", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", action="append", help="run only these variants (repeatable)")
    parser.add_argument("--task", action="append", help="run only these tasks (repeatable)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-mb", type=int, default=800)
    args = parser.parse_args(argv)
    inputs = load_project_inputs(args.project)
    variants = read_variants(args.variants)
    if args.variant:
        unknown = sorted(set(args.variant) - {v.name for v in variants})
        if unknown:
            parser.error(f"unknown variants {unknown}")
        variants = [v for v in variants if v.name in args.variant]
    for variant in variants:
        summary = run_variant(
            inputs, variant, args.out, workers=args.workers, threads=args.threads,
            memory_mb=args.memory_mb, only=args.task,
        )
        print(f"{variant.name}: kept {summary['n_kept']}/{summary['n_trajectories']} "
              f"outcomes {summary['outcomes']} in {summary['wall_time_s']:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Write `tools/diagnosis/variants_p9a.toml` (spec 3.3)**

```toml
# Setup variants of P9a plan 1 (spec section 3.3). Each overrides the project's
# [electronic] and [dynamics] sections; "base" reproduces the M4 study.

[[variant]]
name = "base"
description = "M4 as run: SA2-CASSCF(2,2)/6-31G*, overlap coupling, dt 0.5 fs"

[[variant]]
name = "nacdr"
description = "NAC coupling instead of local diabatization"
[variant.dynamics]
coupling = "nacdr"

[[variant]]
name = "dt_quarter"
description = "half the time step"
[variant.dynamics]
dt_fs = 0.25

[[variant]]
name = "sa3_cas22"
description = "three states, same active space"
[variant.electronic]
singlets = 3

[[variant]]
name = "sa3_cas23"
description = "three states, a third active orbital"
[variant.electronic]
singlets = 3
[variant.electronic.casscf]
ncas = 3
nelecas = 2

[[variant]]
name = "sa2_cas43"
description = "two states, four electrons in three orbitals"
[variant.electronic.casscf]
ncas = 3
nelecas = 4
```

Add a test that this file reads (`rv.read_variants(TOOLS/"diagnosis"/"variants_p9a.toml")` gives the six names in this order and every variant's sections validate against the canned project's config except for the `singlets` Weyl-count rule; do not validate the ethylene-specific active spaces against the canned config).

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/diagnosis -q -W error`, then `ruff format . && ruff check .`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add tools/diagnosis tests/diagnosis
git commit -m "feat: run our trajectory runner on fixed initial conditions for setup variants" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Replayed single points around each failure

**Files:**
- Create: `tools/diagnosis/replay_points.py`
- Create: `tests/diagnosis/test_replay_points.py`

**Interfaces:**
- Consumes: Task 1 `forensics.read_rows`, `forensics.trajectory_report`, `forensics.failed_geometry`, `forensics.jsonable`, `forensics.find_stage_dir`; Task 2 `project_inputs.load_project_inputs`, `run_variants.Variant`, `run_variants.deep_merge`, `run_variants.read_variants`; `gqteashmd.electronic.pyscf_backend.PySCFBackend` (`compute`, `reset`, `states`, private `_live` read-only: `_live.mol`, `_live.solver` = the CASSCF object with `mo_coeff`, `ncore`, `ncas`, `nelecas`, `ci`); `gqteashmd.electronic.base.Request(coords_bohr, gradients=frozenset(), transition_dipoles=..., nacs=..., overlaps=True)` (check the field names in `src/gqteashmd/electronic/base.py`); `ElectronicResult.energies`, `.spin_squares`, `.gradients`, `.meta` (keys `casscf`, `overlap_min_singular_value`); `gqteashmd.errors.ElectronicStructureError`.
- Produces (used by Task 6):
  - `active_orbital_overlap(c_prev: np.ndarray, c_cur: np.ndarray, s_cross: np.ndarray) -> np.ndarray` (singular values of `c_prev^T S c_cur`, descending)
  - `natural_occupations(rdm1: np.ndarray) -> np.ndarray` (eigenvalues, descending)
  - `probe_window(n_records: int, end: int, size: int) -> list[int]` (record indices `max(0, end - size + 1) .. end`)
  - `failure_record(report: dict) -> int` (the first jump's record, else the last record)
  - `@dataclass PointResult: label: str; mode: str; variant: str; energies_eV: list[float] | None; spin_squares: list[float] | None; occupations: list[list[float]] | None; active_overlap: list[float] | None; state_overlap_min_sv: float | None; rung: str | None; wall_time_s: float; error: str | None`
  - `solve_path(symbols, charge, geometries: Sequence[np.ndarray], labels: Sequence[str], electronic: ElectronicSection, *, mode: str, variant: str, threads: int, memory_mb: int, scratch: Path) -> list[PointResult]` (`mode` in `MODES = ("continued", "backward", "fresh")`; results are in the order of `geometries` for every mode)
  - `determinism(symbols, charge, geometry, electronic, *, threads, memory_mb, scratch) -> dict` (two fresh solves in two spawned processes; `max_abs_diff_eV`)
  - `convergence_trials(symbols, charge, geometry, electronic, *, threads, memory_mb, scratch) -> list[dict]`
  - `REPLAY_VARIANTS: tuple[Variant, ...]` = base, `sa3_cas22`, `sa3_cas23`, `sa2_cas43` (the electronic overrides of `variants_p9a.toml`)
  - `replay_trajectory(project_dir: Path, task_id: str, *, window: int, threads: int, memory_mb: int, scratch: Path) -> dict`

- [ ] **Step 1: Write the failing tests**

```python
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
    from gqteashmd.core.units import ANGSTROM_TO_BOHR
    from gqteashmd.electronic.base import MoleculeSpec, Request
    from gqteashmd.electronic.pyscf_backend import PySCFBackend

    from pathlib import Path

    xyz = parse_xyz((Path(__file__).resolve().parents[1] / "data" / "ethylene.xyz").read_text())
    q0 = np.asarray(xyz.coords_angstrom) * ANGSTROM_TO_BOHR
    q1 = q0 + 0.01 * np.random.default_rng(3).standard_normal(q0.shape)
    settings = ElectronicSection.model_validate(
        {"method": "casscf", "basis": "6-31g*", "singlets": 2, "casscf": {"ncas": 2, "nelecas": 2}}
    )
    labels = ["S0", "S1"]
    cont = rp.solve_path(tuple(xyz.symbols), 0, [q0, q1], labels, settings, mode="continued",
                         variant="base", threads=2, memory_mb=1000, scratch=tmp_path / "a")
    fresh = rp.solve_path(tuple(xyz.symbols), 0, [q0, q1], labels, settings, mode="fresh",
                          variant="base", threads=2, memory_mb=1000, scratch=tmp_path / "b")
    backend = PySCFBackend(MoleculeSpec(tuple(xyz.symbols), 0, 1), settings, threads=2, memory_mb=1000,
                           scratch_dir=tmp_path / "c")
    ref = [backend.compute(Request(q, gradients=frozenset(), overlaps=True)) for q in (q0, q1)]
    for point, r in zip(cont, ref, strict=True):
        assert point.error is None
        np.testing.assert_allclose(point.energies_eV, [r.energies[s] * 27.211386245988 for s in backend.states], atol=1e-8)
    np.testing.assert_allclose(cont[1].energies_eV, fresh[1].energies_eV, atol=1e-5)
    assert cont[0].active_overlap is None and min(cont[1].active_overlap) > 0.99
    assert cont[1].state_overlap_min_sv > 0.99
    assert len(cont[1].occupations) == 2 and abs(sum(cont[1].occupations[0]) - 2.0) < 1e-8
```

(Use `HARTREE_IN_EV` from `gqteashmd.core.units` instead of the literal; check the `Request` constructor signature in `src/gqteashmd/electronic/base.py` and adapt the keyword names.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/diagnosis/test_replay_points.py -q -W error -m "not pyscf"`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `tools/diagnosis/replay_points.py`**

Implement exactly these behaviours (code skeleton; fill the bodies as specified):

```python
"""Replayed single points around a trajectory's failure (P9a spec 3.2).

For the records in a window before the failure (the first energy jump, else the end) and, for
overlap and electronic failures, the geometry of the failed step (the Verlet position at the
smallest step), solve SA-CASSCF in three modes and four method variants and record energies,
<S^2>, natural occupations, active-orbital and state overlaps with the previous point,
the convergence rung and the wall time. Reads PySCFBackend._live (diagnostic only, ruling P9a1-1).

Example (WSL):
  python tools/diagnosis/replay_points.py ~/p8_m4/run traj_00004 --out ~/p9a/replay/traj_00004.json
"""

from __future__ import annotations

# imports: argparse, json, multiprocessing, sys, time, dataclasses, pathlib, numpy;
# sys.path insert of this directory; from forensics import (failed_geometry, find_stage_dir,
# jsonable, read_rows, trajectory_report); from project_inputs import load_project_inputs;
# from run_variants import Variant, deep_merge; gqteashmd imports inside functions.

MODES = ("continued", "backward", "fresh")
REPLAY_VARIANTS = (
    Variant("base", {}, {}),
    Variant("sa3_cas22", {"singlets": 3}, {}),
    Variant("sa3_cas23", {"singlets": 3, "casscf": {"ncas": 3, "nelecas": 2}}, {}),
    Variant("sa2_cas43", {"casscf": {"ncas": 3, "nelecas": 4}}, {}),
)
CONVERGENCE_TRIALS = (  # (conv_tol, conv_tol_grad) of [electronic.casscf]
    (1e-10, 1e-6), (1e-8, 1e-5), (1e-8, 1e-4),
)
```

- `active_orbital_overlap`: `np.linalg.svd(c_prev.T @ s_cross @ c_cur, compute_uv=False)` sorted descending.
- `natural_occupations`: `np.sort(np.linalg.eigvalsh((rdm1 + rdm1.T) / 2))[::-1]`.
- `probe_window(n_records, end, size)`: `ValueError` unless `0 <= end < n_records` and `size >= 1`; returns `list(range(max(0, end - size + 1), end + 1))`.
- `failure_record(report)`: `report["jumps"][0]["record"]` if there is a jump (accept a `Jump` or a dict), else `report["n_records"] - 1`.
- `solve_path(...)`: builds `PySCFBackend(MoleculeSpec(symbols, charge, 1), electronic, log_file=scratch/"electronic.log", threads=, memory_mb=, scratch_dir=scratch)` (create `scratch` first). Order of evaluation: `continued` = the given order with one backend; `backward` = reversed order with one backend (results re-reversed to the given order); `fresh` = `backend.reset()` before every point. For each point: `backend.compute(Request(q, gradients=frozenset(), overlaps=True))`; on `ElectronicStructureError` record `error=str(exc)` and every other field `None` (the backend keeps its previous snapshot, so the path continues). From the result: `energies_eV` (in `backend.states` order, × `HARTREE_IN_EV`), `spin_squares` (same order; `None` entries allowed), `rung = result.meta.get("casscf")`, `state_overlap_min_sv = result.meta.get("overlap_min_singular_value")` (None at a first point). From `backend._live.solver` (`mc`): active orbitals `mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas]`; per-state occupations `natural_occupations(pyscf.fci.direct_spin1.make_rdm1(ci, mc.ncas, mc.nelecas))` for `ci in mc.ci`; `active_overlap` with the previous **successful point in evaluation order** (`continued`/`backward`) using `s_cross = pyscf.gto.intor_cross("int1e_ovlp", mol_prev, mol_cur)` (`backend._live.mol`); `None` for the first point and for every point in `fresh` mode. `label` is the caller's label of the point (e.g. `"r17"` or `"failed_step"`); `wall_time_s` per point.
- `determinism(...)`: runs a top-level function `_fresh_energies(args) -> list[float]` (one fresh `solve_path` point, returns `energies_eV`) twice through `ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1)` (two separate processes); returns `{"energies_eV": [a, b], "max_abs_diff_eV": float(max|a - b|)}` or `{"error": ...}` if either failed.
- `convergence_trials(...)`: for each `(conv_tol, conv_tol_grad)` in `CONVERGENCE_TRIALS` and each mode in (`fresh`,), solve the geometry with `electronic` updated by `deep_merge(..., {"casscf": {...}})`; returns one dict per trial: `conv_tol`, `conv_tol_grad`, `error`, `energies_eV`, `rung`, `wall_time_s`.
- `replay_trajectory(project_dir, task_id, *, window, threads, memory_mb, scratch)`:
  1. `inputs = load_project_inputs(project_dir)`; `traj = find_stage_dir(project_dir, "dynamics")/"tasks"/task_id`; `report = trajectory_report(traj)`; `rows, labels, _ = read_rows(traj)`.
  2. `end = failure_record(report)`; `records = probe_window(rows.size, end, window)`; geometries `rows["q"][i]` with labels `f"r{i}"`.
  3. If `report["outcome"]` is `overlap_failure` or `electronic_failure`: compute the active-state gradient at the last record with a continued backend over the window (`Request(..., gradients=frozenset({active label}))` at the last point), `dt_min = dt_fs/AU_TIME_IN_FS / 2**max_step_halvings` from `inputs.config.dynamics`, and append `failed_geometry(q_last, v_last, g, masses_me_per_dof, dt_min)` labelled `"failed_step"`.
  4. For every variant of `REPLAY_VARIANTS` and every mode of `MODES`: `solve_path` over the points (scratch `scratch/task/variant/mode`); a variant whose `ElectronicSection` does not validate is recorded with its validation error.
  5. `determinism` at the end record's geometry (base settings); `convergence_trials` at the `"failed_step"` geometry for `electronic_failure` trajectories, else `[]`.
  6. Return `{"task", "outcome", "onset", "records", "labels", "stored_energies_eV" (the window's stored energies), "points": {variant: {mode: [PointResult, ...]}}, "determinism", "convergence_trials"}`.
- `main(argv)`: `project`, one or more `task` ids (or `--all-lost`: every trajectory whose outcome is not `finished`/`stopped_in_s0`), `--out` (a directory: one `<task>.json` per task), `--window` (default 6), `--threads` 2, `--memory-mb` 800, `--scratch` (default `<out>/scratch`), `--workers` (default 1; with > 1 run tasks in parallel with a spawn `ProcessPoolExecutor` over a top-level `_replay_one(args)`). Write JSON with `jsonable(...)`, `allow_nan=False`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/diagnosis/test_replay_points.py -q -W error` (all markers, so the `pyscf` test runs; about a minute), then `ruff format . && ruff check .`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tools/diagnosis/replay_points.py tests/diagnosis/test_replay_points.py
git commit -m "feat: replay single points around trajectory failures with method variants" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: SHARC_PYSCF harness: the P7b minors and the ensemble parameters

**Files:**
- Modify: `tools/sharc_compare/run_sharc_pyscf_dynamics.py`
- Modify: `tests/sharc_compare/test_run_sharc_pyscf_dynamics.py`
- Modify: `tests/validation/test_sharc_pyscf_dynamics.py` (callers of `run_sharc_trajectory`; the report-only positions)

**Interfaces:**
- Consumes: the current module (read it in full first).
- Produces (used by Task 5):
  - `pyscf_template(convergence: Convergence = DEFAULT_CONVERGENCE, *, ncas: int = 2, nelecas: int = 2, roots: int = N_STATES, basis: str = BASIS) -> str`
  - `sharc_input(n_steps, *, coupling: str, dt_fs: float = DT_FS, nsubsteps: int = NSUBSTEPS, state_mch: int = INITIAL_STATE, n_states: int = N_STATES, seed: int = 1, phases_from_interface: bool = True, hopping: str = "off", decoherence: str = "none", decoherence_param_Eh: float = 0.1) -> str` (`coupling` now required)
  - `HOPPING = ("off", "sharc")`, `DECOHERENCE = ("none", "edc")`
  - `timeout_for(n_steps: int) -> float` = `max(MIN_TIMEOUT_S, TIMEOUT_PER_STEP_S * n_steps)` with `MIN_TIMEOUT_S = 600.0`, `TIMEOUT_PER_STEP_S = 60.0`
  - `run_sharc_trajectory(geometry_bohr, velocities_au, *, n_steps, workdir, symbols, masses_amu, sharc_home, provider: str, coupling: str, sharc_python=None, model=None, phases_from_interface=None, convergence=DEFAULT_CONVERGENCE, timeout_s: float | None = None, template: str | None = None, n_states: int = N_STATES, dt_fs: float = DT_FS, state_mch: int = INITIAL_STATE, hopping: str = "off", decoherence: str = "none", seed: int = 1, check: bool = True) -> dict[str, Any]` (`provider` and `coupling` required; `timeout_s=None` uses `timeout_for(n_steps)`; with `check=False` a failed sharc.x does not raise: the returned dict has `returncode`, `stderr_tail` (last 2000 characters), `stdout_tail`, `wall_time_s`, and the per-step data of a parseable partial `output.dat` (else `data: None`); with `check=True` behaviour is as before plus `returncode = 0`)
  - `step_differences` raises `ValueError` naming the side unless both `steps` arrays are strictly increasing.
  - `SwapModel.evaluate` raises `ValueError` (naming the quantity) for non-finite energies, gradients, NACs (with `nacdr`) or overlaps, like `BackendModel._snapshot`.

- [ ] **Step 1: Write the failing tests** (add to `tests/sharc_compare/test_run_sharc_pyscf_dynamics.py`; adapt existing calls of `sharc_input`/`run_sharc_trajectory` to pass `coupling`/`provider` explicitly)

```python
def test_sharc_input_requires_the_coupling():
    with pytest.raises(TypeError):
        dyn.sharc_input(10)


def test_sharc_input_hopping_and_edc():
    lines = dyn.sharc_input(10, coupling="nacdr", hopping="sharc", decoherence="edc",
                            decoherence_param_Eh=0.1, seed=7).splitlines()
    assert "hopping_procedure sharc" in lines
    assert "decoherence_scheme edc" in lines
    assert "decoherence_param 0.1" in lines
    assert "rngseed 7" in lines
    default = dyn.sharc_input(10, coupling="nacdr").splitlines()
    assert "hopping_procedure off" in default and "decoherence_scheme none" in default
    assert not any(line.startswith("decoherence_param") for line in default)
    with pytest.raises(ValueError, match="hopping"):
        dyn.sharc_input(10, coupling="nacdr", hopping="fssh")
    with pytest.raises(ValueError, match="decoherence"):
        dyn.sharc_input(10, coupling="nacdr", decoherence="afssh")


def test_pyscf_template_active_space():
    text = dyn.pyscf_template(ncas=3, nelecas=2, roots=3).splitlines()
    assert "ncas 3" in text and "nelecas 2" in text and "roots 3" in text
    assert dyn.pyscf_template() == dyn.PYSCF_TEMPLATE


def test_timeout_scales_with_the_run():
    assert dyn.timeout_for(1) == dyn.MIN_TIMEOUT_S
    assert dyn.timeout_for(200) == 200 * dyn.TIMEOUT_PER_STEP_S


def test_step_differences_needs_sorted_unique_steps():
    a = {"steps": np.array([0, 1, 1]), **{k: np.zeros((3, 2)) for k in dyn.QUANTITIES}}
    b = {"steps": np.array([0, 1, 2]), **{k: np.zeros((3, 2)) for k in dyn.QUANTITIES}}
    with pytest.raises(ValueError, match="first"):
        dyn.step_differences(a, b)
    with pytest.raises(ValueError, match="second"):
        dyn.step_differences(b, a)


def test_swap_model_refuses_non_finite_data():
    # a fake backend whose energies contain NaN (follow the existing swap-model test's fake)
    ...
```

For the last test, reuse the fake backend of the existing swap-model test in this file (it builds a `SimpleNamespace`-style backend); make one variant return `nan` in an energy and assert `pytest.raises(ValueError, match="energ")`; one with a `nan` overlap, `match="overlap"`. Also a test that `run_sharc_trajectory(..., check=False)` against the existing fake interface used in this file (the `runQM.sh` fake run) returns `returncode != 0` and `data is None` when the fake sharc command fails: if the existing tests have no fake `sharc.x`, write a tiny one: a `sharc_home` directory with `sharcvars.sh` (empty) and an executable `sharc.x` script `#!/bin/sh\necho boom >&2\nexit 3\n`; assert `out["returncode"] == 3` and `"boom" in out["stderr_tail"]`.

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/sharc_compare/test_run_sharc_pyscf_dynamics.py -q -W error`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement**

In `tools/sharc_compare/run_sharc_pyscf_dynamics.py`:
1. `pyscf_template(convergence=DEFAULT_CONVERGENCE, *, ncas=2, nelecas=2, roots=N_STATES, basis=BASIS)` writes `ncas {ncas}\nnelecas {nelecas}\nroots {roots}\nbasis {basis}\n...` (the other lines unchanged); `PYSCF_TEMPLATE = pyscf_template()` unchanged in content.
2. `sharc_input`: `coupling` keyword without default; validate `hopping in HOPPING` and `decoherence in DECOHERENCE` (`ValueError` naming the option and the allowed values); write `f"hopping_procedure {hopping}"` and `f"decoherence_scheme {decoherence}"` in place of the two fixed lines; with `"edc"` add `f"decoherence_param {float(decoherence_param_Eh)!r}"` right after it. SHARC 4.1 manual table 4.1 keywords; nothing else changes. Update the docstring ("hopping off and no decoherence" becomes the default).
3. `MIN_TIMEOUT_S`, `TIMEOUT_PER_STEP_S`, `timeout_for`.
4. `run_sharc_trajectory`: `provider` and `coupling` required keywords; the new keywords as in Interfaces; `template` default `pyscf_template(convergence, roots=n_states)`; pass `n_states`, `dt_fs`, `hopping`, `decoherence`, `seed` to `sharc_input`; `timeout_s = timeout_for(n_steps) if timeout_s is None else timeout_s`; with `check=False` never raise for a non-zero exit or a missing/unparseable `output.dat` (catch `ValueError`, `IndexError`, `OSError` from the parser and set `data=None`); return `{**trajectory_data(...), "returncode": 0}` on success and `{"returncode": rc, "stderr_tail": ..., "stdout_tail": ..., "wall_time_s": wall, "data": parsed_or_None}` on failure with `check=False`. A `run_in_session` timeout still raises (it is a harness failure).
5. `step_differences`: before aligning, `for name, side in (("first", a), ("second", b)):` raise `ValueError(f"the {name} trajectory's steps are not strictly increasing")` unless `np.all(np.diff(side["steps"]) > 0)`.
6. `SwapModel.evaluate`: after building `energies`, `gradients`, `nacs`, `overlaps`: `ValueError` with "non-finite energies" / "non-finite gradients" / "non-finite NACs" (only for `nacdr`) / "non-finite overlaps" when `np.isfinite(...).all()` is false (check `BackendModel._snapshot` in `src/gqteashmd/trajectory/model.py` and use the same wording).
7. The `runQM.sh` comment: replace the two comment lines by one quoting the measured message: `# measured (P7b Task 10): with 'step 0' SHARC_PYSCF.py stops with "File .../pyscf.chk not found, cannot move to old!"; with 'init' it runs`.
8. Update every caller in this module (`swap_report`, `main`) to pass `provider=` and `coupling=` explicitly (same values they used through the defaults).

In `tests/validation/test_sharc_pyscf_dynamics.py`: pass `provider=`/`coupling=` explicitly in every `run_sharc_trajectory` call (the previous defaults were `provider="sharc_pyscf"`, `coupling="overlap"`; keep each call's effective value); where the report-only position differences are printed, also write them with `json.dumps` to `Path(os.environ["GQTEASHMD_REPORT_DIR"]) / "sharc_pyscf_positions.json"` when that variable is set (create the directory); add a comment at the step-24 assertion: `# brittle across processes and PySCF versions (see CLAUDE.md Known limitation 3a); re-checked in P9a plan 2`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/sharc_compare -q -W error`; then the live file once with SHARC: `SHARC_HOME=/home/ajc/sharc4-4.1 SHARC_PYTHON=/home/ajc/miniforge3/envs/sharc4.1/bin/python pytest tests/validation/test_sharc_pyscf_dynamics.py -q -W error` (markers `sharc`, `pyscf`, `slow`; about 8 min; run it with `run_in_background` and wait for the notification). Then `ruff format . && ruff check .`.
Expected: all pass (the live file: 16 passed, as at P7b `188b20b`).

- [ ] **Step 5: Commit**

```bash
git add tools/sharc_compare/run_sharc_pyscf_dynamics.py tests/sharc_compare/test_run_sharc_pyscf_dynamics.py tests/validation/test_sharc_pyscf_dynamics.py
git commit -m "fix: tidy the SHARC_PYSCF harness and add hopping, decoherence and active-space options" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: SHARC on a project's initial conditions

**Files:**
- Create: `tools/sharc_compare/run_sharc_pyscf_ensemble.py`
- Create: `tests/diagnosis/test_sharc_ensemble.py`

**Interfaces:**
- Consumes: Task 4 `run_sharc_trajectory`, `pyscf_template`, `Convergence`, `interface_geometry`, `DECIMALS_ANGSTROM`; `run_sharc_lvc.format_geom`, `format_veloc`; Task 1 `forensics.energy_jumps`, `forensics.jsonable`, `JUMP_THRESHOLD_EV`; Task 2 `project_inputs.load_project_inputs`, `run_variants.read_variants`, `run_variants.variant_sections`, `run_variants.Variant`.
- Produces (used by Task 7):
  - `parse_geom(text: str) -> tuple[list[str], np.ndarray, np.ndarray]` (symbols, masses amu, coordinates bohr (natom, 3)) and `parse_veloc(text: str) -> np.ndarray` (natom, 3): our own readers of the SHARC `geom`/`veloc` formats written by `format_geom`/`format_veloc` (manual 4.2, 4.3)
  - `sharc_settings(electronic: ElectronicSection, dynamics: DynamicsStageSection) -> dict` (the keyword arguments of `run_sharc_trajectory` for one variant: `n_states`, `dt_fs`, `hopping="sharc"` unless `dynamics.hopping == "none"` (then `"off"`), `decoherence="edc"` if `dynamics.decoherence == "edc"` else `"none"`, `coupling="nacdr"`, `phases_from_interface=False`, `template=pyscf_template(Convergence(conv_tol, conv_tol_grad), ncas=, nelecas=, roots=singlets, basis=)`; `ValueError` for a non-CASSCF method)
  - `trajectory_summary(task_id: str, out: dict, *, dt_fs: float, tmax_fs: float) -> dict` (per trajectory: `returncode`, `t_final_fs`, `n_steps`, `n_hops`, `max_drift_eV`, `max_jump_eV`, `jumps` (from `energy_jumps` on the SHARC `e_total`/`energies`), `ended` = `"finished"` if the last step reaches `tmax_fs` (within half a step) else `"stopped"`, `stderr_tail`)
  - `run_ensemble(project_dir, variant: Variant, out_dir, *, sharc_home, sharc_python, workers, tmax_fs=None, only=None) -> dict` (writes `out_dir/<variant>/<task>/` (sharc.x work dir) and `out_dir/<variant>/sharc_summary.json`)

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from gqteashmd.config.schema import DynamicsStageSection, ElectronicSection
from tests.diagnosis.conftest import load_tool

ens = load_tool("sharc_compare", "run_sharc_pyscf_ensemble")
lvc = load_tool("sharc_compare", "run_sharc_lvc")


def test_geom_and_veloc_read_back():
    symbols = ["C", "C", "H", "H", "H", "H"]
    masses = np.array([12.0, 12.0, 1.00782503223, 1.00782503223, 1.00782503223, 1.00782503223])
    rng = np.random.default_rng(5)
    q = rng.standard_normal((6, 3))
    v = rng.standard_normal((6, 3)) * 1e-4
    s2, m2, q2 = ens.parse_geom(lvc.format_geom(symbols, masses, q))
    assert s2 == symbols
    np.testing.assert_allclose(m2, masses, rtol=1e-10)
    np.testing.assert_allclose(q2, q, atol=1e-10)
    np.testing.assert_allclose(ens.parse_veloc(lvc.format_veloc(v)), v, atol=1e-14)


def _sections(**dyn):
    electronic = ElectronicSection.model_validate(
        {"method": "casscf", "basis": "6-31g*", "singlets": 3, "casscf": {"ncas": 3, "nelecas": 2}}
    )
    return electronic, DynamicsStageSection.model_validate({"tmax_fs": 100.0, **dyn})


def test_sharc_settings_follow_the_variant():
    s = ens.sharc_settings(*_sections())
    assert s["n_states"] == 3 and s["dt_fs"] == 0.5
    assert s["hopping"] == "sharc" and s["decoherence"] == "edc" and s["coupling"] == "nacdr"
    assert s["phases_from_interface"] is False
    assert "ncas 3" in s["template"] and "roots 3" in s["template"] and "basis 6-31g*" in s["template"]
    off = ens.sharc_settings(*_sections(hopping="none", decoherence="none"))
    assert off["hopping"] == "off" and off["decoherence"] == "none"


def test_sharc_settings_refuse_tda():
    electronic = ElectronicSection.model_validate(
        {"method": "tda", "basis": "sto-3g", "functional": "hf", "singlets": 2}
    )
    with pytest.raises(ValueError, match="casscf"):
        ens.sharc_settings(electronic, DynamicsStageSection.model_validate({"tmax_fs": 1.0}))


def test_trajectory_summary_finished_and_jump():
    eh = 1 / 27.211386245988
    out = {
        "returncode": 0,
        "steps": np.arange(5),
        "e_total": np.array([0.0, 0.0, 0.3, 0.3, 0.3]) * eh,
        "energies": np.array([[0.0, 9.0]] * 2 + [[0.0, 9.3]] * 3) * eh,
        "active": np.array([1, 1, 1, 0, 0]),
        "dt_au": 0.5 / 0.024188843265857,
    }
    s = ens.trajectory_summary("traj_00000", out, dt_fs=0.5, tmax_fs=2.0)
    assert s["ended"] == "finished" and s["n_hops"] == 1
    assert s["max_jump_eV"] == pytest.approx(0.3, abs=1e-9)
    assert [j["record"] for j in s["jumps"]] == [2]
    failed = ens.trajectory_summary("traj_00001", {"returncode": 3, "data": None, "stderr_tail": "x"}, dt_fs=0.5, tmax_fs=2.0)
    assert failed["ended"] == "stopped" and failed["t_final_fs"] is None and failed["stderr_tail"] == "x"


@pytest.mark.sharc
@pytest.mark.pyscf
def test_one_short_sharc_trajectory(tmp_path):
    """Two steps of sharc.x + SHARC_PYSCF.py from the first initial condition of a small
    project built here from ethylene (needs SHARC_HOME/SHARC_PYTHON)."""
    ...
```

For the live test: skip unless `SHARC_HOME` and `SHARC_PYTHON` are set (as the other live tests do; reuse their skip helper); call `run_sharc_trajectory` directly with the ethylene initial conditions of `run_sharc_pyscf_dynamics.ethylene_initial_conditions()`, `**ens.sharc_settings(...)` for SA2-CAS(2,2)/6-31G\*, `n_steps=2`, `check=False`, and assert `returncode == 0` and 3 printed steps (0, 1, 2). It runs in about a minute.

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/diagnosis/test_sharc_ensemble.py -q -W error -m "not sharc"`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `tools/sharc_compare/run_sharc_pyscf_ensemble.py`**

- Module docstring: purpose (spec 3.4), clean-room note (sharc.x and SHARC_PYSCF.py executed only; `geom`/`veloc` formats from the manual 4.2/4.3 as written by `run_sharc_lvc`), an example command:
  `python tools/sharc_compare/run_sharc_pyscf_ensemble.py ~/p8_m4/run tools/diagnosis/variants_p9a.toml --variant base --out ~/p9a/sharc --sharc-home /home/ajc/sharc4-4.1 --python /home/ajc/miniforge3/envs/sharc4.1/bin/python --workers 8`.
- Imports: `sys.path.insert` of `tools/sharc_compare` and `tools/diagnosis`; `from run_sharc_pyscf_dynamics import Convergence, pyscf_template, run_sharc_trajectory, interface_geometry`; `from forensics import energy_jumps, jsonable`; `from project_inputs import load_project_inputs`; `from run_variants import read_variants, variant_sections, Variant`.
- `parse_geom`: per line `symbol Z x y z mass` (the layout `format_geom` writes; read `run_sharc_lvc.format_geom` and invert it exactly); `parse_veloc`: three floats per line.
- `sharc_settings` as in Interfaces; `n_states = electronic.singlets`; `dt_fs = dynamics.dt_fs`; `seed`: not set here (per trajectory, below).
- `trajectory_summary`: data = `out` itself on success (`returncode == 0`) or `out.get("data")`; with no data: `t_final_fs=None`, `n_steps=0`, `n_hops=0`, `max_drift_eV=None`, `max_jump_eV=None`, `jumps=[]`, `ended="stopped"`. With data: `t = steps * dt_fs`; `n_hops = count of active changes`; `max_drift_eV = max|E_tot - E_tot[0]| * HARTREE_IN_EV`; `max_jump_eV` from `gqteashmd.analysis.energy.max_jump_eV`; `jumps = energy_jumps(t, e_total, energies, active=active)` (as dicts via `jsonable`); `ended = "finished" if t[-1] >= tmax_fs - dt_fs / 2 else "stopped"`. Always `returncode`, `stderr_tail` (or None).
- `run_ensemble`: `inputs = load_project_inputs(project_dir)`; `electronic, dynamics = variant_sections(inputs.config, variant)`; `tmax = dynamics.tmax_fs if tmax_fs is None else tmax_fs`; `n_steps = round(tmax / dynamics.dt_fs)`; one job per start (filtered by `only`): `run_sharc_trajectory(interface_geometry(start.q.reshape(-1, 3)), start.v.reshape(-1, 3), n_steps=n_steps, workdir=out/variant/task, symbols=inputs.symbols, masses_amu=inputs.masses_amu, sharc_home=..., sharc_python=..., provider="sharc_pyscf", seed=start.index + 1, check=False, **sharc_settings(electronic, dynamics))`, initial state `state_mch` from the start's label (S1 -> 1, via `StateLabel.parse(start.state).index` for singlets) passed to `run_sharc_trajectory(state_mch=...)` (Task 4). A work dir that exists with an `output.dat` is not rerun (resume by skipping; report it as `"reused": true`). Run jobs with `concurrent.futures.ThreadPoolExecutor(max_workers=workers)` (each job is a subprocess); set `OMP_NUM_THREADS=1`-style variables to `1` in the sharc environment if `PYSCF.resources` uses `ncpu 1` (it does). Collect `trajectory_summary` per task, then write `sharc_summary.json`: `variant`, `project`, `n_trajectories`, `ended` counts, `n_with_jump` (any jump > 0.1 eV), `n_finished_without_jump`, `trajectories`, `wall_time_s`.
- `main(argv)`: `project`, `variants` file, `--variant` (one, required), `--out`, `--sharc-home`, `--python`, `--workers` (8), `--tmax-fs` (optional, for a short probe), `--task` (repeatable).

- [ ] **Step 4: Run the tests**

Run: `pytest tests/diagnosis/test_sharc_ensemble.py -q -W error -m "not sharc"`; then the live test once: `SHARC_HOME=/home/ajc/sharc4-4.1 SHARC_PYTHON=/home/ajc/miniforge3/envs/sharc4.1/bin/python pytest tests/diagnosis/test_sharc_ensemble.py -q -W error -m sharc`; `ruff format . && ruff check .`.
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tools/sharc_compare/run_sharc_pyscf_ensemble.py tests/diagnosis/test_sharc_ensemble.py tools/sharc_compare/run_sharc_pyscf_dynamics.py
git commit -m "feat: run sharc.x with SHARC_PYSCF.py on a project's initial conditions" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Measurement A: forensics and replayed single points of M4 (no code, no commit)

**Files:** none in the repository. Outputs under `~/p9a/` (WSL native). Report: the task report file given in the dispatch.

**Interfaces:**
- Consumes: Task 1 CLI, Task 3 CLI, the M4 projects `~/p8_m4/run` (read only).

- [ ] **Step 1: Forensics of M4**

Run: `python tools/diagnosis/trajectory_forensics.py ~/p8_m4/run --out ~/p9a/forensics_m4.json` (seconds). Check against the M4 facts (`USER_MANUAL.md` 5.9 / the P8 Task 11 report): 20 trajectories, outcomes finished 2, `overlap_failure` 8, `energy_drift` 7, `electronic_failure` 3. Report the onset counts, every jump (time, state, ΔE_state, gap, sv, rung, depth), every hop before an `overlap_failure`, and the electronic-log lines of the three `electronic_failure` trajectories.

- [ ] **Step 2: Replay every lost trajectory**

Run (background, notification on completion): `python tools/diagnosis/replay_points.py ~/p8_m4/run --all-lost --out ~/p9a/replay --window 6 --workers 4 --threads 2 --memory-mb 1500`. Time it. If a variant or mode fails systematically (e.g. a variant's CASSCF never converges), say so with its error text; do not change the code to make it pass — report it.

- [ ] **Step 3: Write the task report**

For each of the 18 lost trajectories, one line: task, outcome, onset, and per variant/mode whether the energies along the window are continuous (largest |ΔE_I| between consecutive points), whether continued and fresh agree at the failing point (largest |ΔE|), the smallest active-orbital overlap, and the state-overlap sv; then per loss class a summary of the patterns of spec 3.2 (solution switch / real surface feature / intruder) with the trajectories that show each; the determinism numbers (largest difference over all trajectories); the convergence trials of the three electronic failures; time per single point per variant. Quote numbers from the JSON files only. List anything that does not fit.

---

### Task 7: Measurement B: our runner on variants and SHARC on the same initial conditions (no code, no commit)

**Files:** none in the repository. Outputs under `~/p9a/`. Report: the task report file given in the dispatch.

**Interfaces:**
- Consumes: Task 2 CLI and `variants_p9a.toml`, Task 5 CLI, `~/p8_m4/run` (read only).

- [ ] **Step 1: Reproduce M4 with the variant runner (script check)**

Run (background): `python tools/diagnosis/run_variants.py ~/p8_m4/run tools/diagnosis/variants_p9a.toml --out ~/p9a/variants --variant base --workers 8 --threads 2 --memory-mb 800`. Compare per trajectory with `~/p8_m4/run/06_dynamics/dynamics_summary.json`: outcome and `t_final_fs`. Expected: identical outcomes and `t_final_fs` to 4 decimals (the M4 run B showed PySCF reproducibility at round-off level). Any difference is reported and investigated before Step 2 (it would mean the script does not reproduce the stage).

- [ ] **Step 2: The other variants**

Run each of `nacdr`, `dt_quarter`, `sa3_cas22`, `sa3_cas23`, `sa2_cas43` the same way (one invocation each, background). Before each full run, time one trajectory for 4 fs (`--task traj_00004` with a temporary `--out ~/p9a/probe`, then delete the probe output) and estimate the full cost; a variant estimated at more than 6 h on 8 workers is run on the first 10 trajectories only, and that is reported.

- [ ] **Step 3: SHARC on the same initial conditions**

Probe: `python tools/sharc_compare/run_sharc_pyscf_ensemble.py ~/p8_m4/run tools/diagnosis/variants_p9a.toml --variant base --out ~/p9a/sharc_probe --tmax-fs 2 --task traj_00000 --sharc-home /home/ajc/sharc4-4.1 --python /home/ajc/miniforge3/envs/sharc4.1/bin/python --workers 1` (time per step). Then the full run with `--out ~/p9a/sharc --workers 8` (background). If one variant of Step 2 keeps clearly more trajectories than `base` and its SHARC cost estimate is below 6 h, run it in SHARC too.

- [ ] **Step 4: Write the task report**

Per variant: outcomes, onsets, kept count (the Global Constraints rule), time per step, total time; a per-trajectory table across variants (outcome and `t_final_fs`); SHARC: per trajectory how it ended, `t_final_fs`, `max_jump_eV`, jumps (time, state), hops; and per loss class the comparison between our base run and SHARC (does SHARC show the same energy jump at the same time? does SHARC get past the point where we have an overlap failure? does SHARC_PYSCF fail to converge at the same place?). Quote numbers from the JSON files only.

---

### Task 8: The diagnosis report and the documents

**Files:**
- Create: `docs/superpowers/reports/2026-11-12-p9a-diagnosis.md`
- Modify: `CLAUDE.md`, `USER_MANUAL.md`

**Interfaces:**
- Consumes: the Task 6 and Task 7 reports and their JSON files; the spec (sections 2, 3.5, 4.2, 4.3).

- [ ] **Step 1: Write the diagnosis report**

Sections: (1) what was run (commands, commits, machine, times); (2) per loss class (`energy_drift`, `overlap_failure`, `electronic_failure`): the evidence (numbers with the JSON file each comes from), the **label** (ours / method / settings) with the reasoning against the spec's definition ("ours" = differs from SHARC on the same initial conditions or from a fresh solve at the same geometry), the candidate fixes F1-F4 of spec 4.2 it supports or rules out, and any new candidate; (3) the determinism check; (4) the time per step of every variant; (5) the **proposed acceptance setup** (`[electronic]` and `[dynamics]` values, with the reason; spec 4.3) and the variant-run evidence on seed 2026 for it (its kept count on the 20 M4 initial conditions); (6) findings that fit no class; (7) an honest-summary paragraph: what was decided after the numbers were seen. Every number is quoted from a task report or a JSON file; nothing is estimated without saying so.

- [ ] **Step 2: Update CLAUDE.md and USER_MANUAL.md**

CLAUDE.md: the "Read first" list gets the P9a spec, plan 1 and its ledger (`.superpowers/sdd/2026-11-12-p9a1-diagnosis/`) and the diagnosis report; the Architecture section gets a short `tools/diagnosis/` line (report scripts, never imported by `gqteashmd`); Current status: P9a plan 1 done on `feat/p9a-trajectory-losses`, plan 2 waiting for the owner's approval of the diagnosis report; the Environment tiers get the new tests' counts and times as measured in Tasks 1-5; the P7b deferred list: mark the harness minors resolved (Task 4). USER_MANUAL.md: in section 8.16 (or the section that documents the SHARC comparison tools) a short paragraph on the diagnosis tools and where the report is; section 10 (limitations): the measured root causes as **findings** (no new behaviour yet; plan 2 changes behaviour). Mark nothing as implemented that plan 2 has not done.

- [ ] **Step 3: Verify and commit**

Run: `pytest tests/test_cli.py -q -W error` (the docs-adjacent tests) and `ruff format . && ruff check .`.

```bash
git add docs/superpowers/reports/2026-11-12-p9a-diagnosis.md CLAUDE.md USER_MANUAL.md
git commit -m "docs: add the P9a diagnosis report" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Self-review notes (plan author)

- Spec coverage: 3.1 -> Task 1; 3.2 -> Task 3 (+ Task 6); 3.3 -> Task 2 (+ Task 7); 3.4 -> Tasks 4-5 (+ Task 7); 3.5 -> Task 8; section 5 plan-1 items (harness minors) -> Task 4; the step-24 pointer comment is added in Task 4 (its re-check stays in plan 2); spec 6 tests/docs -> Tasks 1-5, 8; the full suite at the end of the plan is run by the controller after Task 8.
- Plan 2 items (F1-F4, the acceptance run, the plan-2 deferred items) are not in this plan.
- Tasks 3-5 specify part of their code by behaviour (the bodies of `replay_points.py`, the live-SHARC test, the swap-model fake) rather than in full: each names the exact functions, keys, orders and error texts; the implementer reads the named source files for the remaining signatures.
- Tasks 6 and 7 are measurements with no diff; by the precedent of P8 Task 11 they get no code review, and their numbers are checked by the Task 8 document review and the final review.
