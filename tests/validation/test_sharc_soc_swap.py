"""The spin-orbit swap test (dynamics with SOC, spec 5.1; plan Task 9): sharc.x drives our PySCF
backend (``tools/sharc_compare/run_sharc_soc_swap.py``) and our runner repeats the trajectory.

Pre-registered (spec 5.1, fixed before any comparison): thioformaldehyde 6-31G, SA-CASSCF(4,3)
2S + 2T and CIS 3S + 3T (``td_conv_tol`` 1e-7), dt 0.5 fs, 25 substeps, ``coupling overlap``,
``nogradcorrect``, the diagonal surface, ``ekincorrect parallel_vel``, EDC (0.1 Eh), ``endpoint``
interpolation; start in S1 from the Wigner sample of each method chosen by the start rule of (b)
(``tests/data/h2cs_soc_swap_start.json``: project seed 2029, RHF/6-31G* frequencies, T = 0, the
first sample whose SHARC run with ``rngseed`` 2029 has a singlet-triplet hop within 200 steps).
The fallback to sample 0 for (a) when none of samples 0-19 has one was decided after the search
found no SA-CASSCF hop; it was not pre-registered (spec 5.1 assumed one start per method). Our
runner's step rejection is kept out of the way (SHARC has none; ``run_sharc_soc_swap.run_ours``).

Start search (plan Task 1 Step 3, done in Task 9): CIS sample 7, first singlet-triplet hop at
step 64, so (b) compares steps 0-84; SA-CASSCF: no hop of any kind in samples 0-19, so (b) is
BLOCKED (spec 5.1: reported to the owner, no window or setting changed) and skipped, and (a)
starts from sample 0.

- (a) hopping off, steps 0-40;
- (b) hopping on, SHARC's random numbers replayed into ours, ``hop_energy = "previous"``, steps 0
  to the first singlet-triplet hop of SHARC's run + 20 (at most 200), the same active spin-free
  state at every step;

bounds per step and component: |dq| <= 1e-5 bohr, |dv| <= 1e-6 a.u., |dE| <= 1e-6 Eh (the total
energy and the active diagonal energy), |dP| <= 1e-5 (the spin-free MCH populations). A miss is
reported with the numbers, not loosened.

Set ``GQTEASHMD_SOC_SWAP_REPORT`` to a directory to keep every per-step difference as JSON.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.sharc,
    pytest.mark.pyscf,
    pytest.mark.slow,
    pytest.mark.skipif(
        not (os.environ.get("SHARC_HOME") and os.environ.get("SHARC_PYTHON")),
        reason="needs SHARC_HOME and SHARC_PYTHON",
    ),
]

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "sharc_compare"
START = Path(__file__).resolve().parents[1] / "data" / "h2cs_soc_swap_start.json"
THREADS = 8


def _load(name):
    if name in sys.modules:  # shared with tests/sharc_compare/test_run_sharc_soc_swap.py
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _report(name: str, data: dict) -> None:
    directory = os.environ.get("GQTEASHMD_SOC_SWAP_REPORT")
    if directory:
        path = Path(directory).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        swap = _load("run_sharc_soc_swap")
        (path / f"{name}.json").write_text(json.dumps(swap.jsonable(data), indent=1) + "\n")


def _compare(method: str, part: str, n_steps: int, options, tmp_path: Path) -> dict:
    """SHARC (its own backend) then ours (a second backend), on the method's start."""
    from gqteashmd.trajectory.model import BackendModel

    swap = _load("run_sharc_soc_swap")
    data = json.loads(START.read_text())["methods"][method]
    start = swap.Start.from_json(data["start"])
    n, m = swap.SETTINGS[method]["singlets"], swap.SETTINGS[method]["triplets"]
    provider = swap.SocSwapModel(
        BackendModel(
            swap.make_backend(method, start, tmp_path / "sharc_backend", threads=THREADS),
            swap.masses_per_coordinate(start),
            coupling="overlap",
        )
    )
    traj = swap.run_sharc(
        provider,
        start,
        options,
        n_steps=n_steps,
        work_dir=tmp_path / "sharc",
        sharc_home=Path(os.environ["SHARC_HOME"]),
        rngseed=int(data["rngseed"]),
    )
    theirs = swap.sharc_records(traj, n, m)
    ours = swap.run_ours(
        swap.make_backend(method, start, tmp_path / "our_backend", threads=THREADS),
        start,
        options,
        n_steps=n_steps,
        work_dir=tmp_path / "ours",
        random_numbers=swap.replayed_numbers(traj),
    )
    mine = swap.our_records(ours["rows"])
    diffs = swap.differences(theirs, mine)
    return {
        "method": method,
        "part": part,
        "sample": start.sample,
        "n_singlets": n,
        "theirs": theirs,
        "ours": mine,
        "result": ours["result"],
        "our_wall_time_s": ours["wall_time_s"],
        "sharc_min_singular_value": min(provider.min_singular_values, default=None),
        "diffs": diffs,
        "recorded_hop": data.get("first_singlet_triplet_hop"),
    }


def _summary(run: dict, last: int) -> str:
    swap = _load("run_sharc_soc_swap")
    top = swap.maxima(run["diffs"], last)
    return f"{run['method']} ({run['part']}, sample {run['sample']}, steps 0-{last}): " + ", ".join(
        f"{k} {v['max']:.3e} at step {v['step']} (bound {v['bound']:.0e})" for k, v in top.items()
    )


@pytest.fixture(scope="module", params=["casscf", "cis"])
def hopping_off(request, tmp_path_factory):
    swap = _load("run_sharc_soc_swap")
    method = request.param
    run = _compare(
        method,
        "a",
        swap.STEPS_HOPPING_OFF,
        swap.dynamics_options(hopping="none"),
        tmp_path_factory.mktemp(f"soc_swap_a_{method}"),
    )
    last = swap.STEPS_HOPPING_OFF
    print(f"\n[soc swap a] {method}:\n{swap.format_table(run['diffs'])}")
    print(f"[soc swap a] {_summary(run, last)}; our result {run['result']}")
    _report(f"swap_a_{method}", {**run, "maxima": swap.maxima(run["diffs"], last)})
    run["last"] = last
    return run


def test_swap_hopping_off(hopping_off):
    """(a) steps 0-40: the four bounds per step and component (spec 5.1)."""
    swap = _load("run_sharc_soc_swap")
    run, last = hopping_off, hopping_off["last"]
    assert np.array_equal(run["theirs"]["steps"], np.arange(last + 1)), "SHARC's printed steps"
    assert run["result"]["outcome"] == "finished", run["result"]
    assert run["result"]["n_rejections"] == 0, f"our run rejected steps: {run['result']}"
    assert np.array_equal(run["diffs"]["steps"], np.arange(last + 1)), "our records' steps"
    found = swap.violations(run["diffs"], last)
    assert not found, "; ".join(found[:20]) + f" ... ({len(found)} in all); {_summary(run, last)}"


@pytest.fixture(scope="module", params=["casscf", "cis"])
def hopping_on(request, tmp_path_factory):
    swap = _load("run_sharc_soc_swap")
    method = request.param
    data = json.loads(START.read_text())["methods"][method]
    recorded = data.get("first_singlet_triplet_hop")
    if recorded is None:
        # spec 5.1: reported BLOCKED to the owner, no window or setting changed; nothing to compare
        pytest.skip(
            f"{method}: BLOCKED (spec 5.1): none of the Wigner samples "
            f"{data['samples_searched']} has a singlet-triplet hop in SHARC's 200-step run"
        )
    n_steps = swap.window_end(int(recorded))
    run = _compare(
        method,
        "b",
        n_steps,
        swap.dynamics_options(hopping="sharc", hop_energy="previous"),
        tmp_path_factory.mktemp(f"soc_swap_b_{method}"),
    )
    hops = swap.singlet_triplet_hops(run["theirs"], run["n_singlets"])
    run["sharc_hops"] = hops
    run["our_hops"] = swap.singlet_triplet_hops(run["ours"], run["n_singlets"])
    run["recorded_hop"] = int(recorded)
    run["last"] = swap.window_end(int(recorded))  # the pre-registered window
    print(f"\n[soc swap b] {method}:\n{swap.format_table(run['diffs'])}")
    changes = np.flatnonzero(np.diff(run["theirs"]["active"])) + 1
    print(
        f"[soc swap b] {method}: SHARC hops at steps {changes.tolist()}, singlet-triplet {hops} "
        f"(recorded {recorded}); ours singlet-triplet {run['our_hops']}; "
        f"active sf SHARC {run['theirs']['active_sf'].tolist()}, "
        f"ours {run['ours']['active_sf'].tolist()}"
    )
    last = run["last"]
    print(f"[soc swap b] {_summary(run, last)}; our result {run['result']}")
    probabilities = run["diffs"]["probabilities"][run["diffs"]["steps"] <= last]
    print(
        f"[soc swap b] {method}: max |d probability| {float(probabilities.max()):.3e} "
        f"(reported); near ties SHARC {swap.near_ties(run['theirs'], last)}, "
        f"ours {swap.near_ties(run['ours'], last)}"
    )
    _report(f"swap_b_{method}", {**run, "maxima": swap.maxima(run["diffs"], last)})
    return run


def test_swap_hopping_on(hopping_on):
    """(b) steps 0 to the first singlet-triplet hop + 20: the four bounds per step and component
    and the same active spin-free state (spec 5.1)."""
    swap = _load("run_sharc_soc_swap")
    run, last = hopping_on, hopping_on["last"]
    # the window was fixed by the start search's hop: the re-run must reproduce it (an earlier
    # hop would shorten the pre-registered window, a later one change it)
    hops, recorded = run["sharc_hops"], run["recorded_hop"]
    assert hops and hops[0] == recorded, (
        f"the re-run's first singlet-triplet hop is at step {hops[0] if hops else None}, "
        f"the start search recorded step {recorded}"
    )
    # a halved step would shift the replayed random stream: the replay needs none
    assert run["result"]["n_rejections"] == 0, f"our run rejected steps: {run['result']}"
    assert last <= int(run["theirs"]["steps"][-1]), f"SHARC's run ends before step {last}"
    assert int(run["diffs"]["steps"][-1]) >= last, f"our records end before step {last}"
    assert np.array_equal(run["diffs"]["steps"][: last + 1], np.arange(last + 1))
    found = swap.violations(run["diffs"], last)
    assert not found, "; ".join(found[:20]) + f" ... ({len(found)} in all); {_summary(run, last)}"
