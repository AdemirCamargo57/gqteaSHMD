"""The spin-orbit swap harness without SHARC or PySCF (dynamics with SOC, plan Task 9 Step 1).

The provider is checked on the P7a-2 LVC spin-orbit model served through our fake backend
(``ModelBackend``, the spin-free view of plan ruling PR4): its QM.out equals the one the LVC
writer produces from the model itself. The output.dat reduction uses the Task 1 probe
(``tests/data/sharc/soc/output.dat``)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gqteashmd.core.units import DALTON_IN_ME
from gqteashmd.dynamics.lvc import LVCModel, lvc_wigner_samples
from gqteashmd.dynamics.spin import expanded_states
from gqteashmd.testing.fake_backend import ModelBackend
from gqteashmd.trajectory.model import BackendModel

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "sharc_compare"
DATA = Path(__file__).resolve().parents[1] / "data"
LVC_SOC = DATA / "lvc" / "lvc_soc.template"
PROBE = DATA / "sharc" / "soc"


def _load(name):
    if name in sys.modules:  # shared with tests/validation/test_sharc_soc_swap.py
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


swap = _load("run_sharc_soc_swap")
lvc = _load("run_sharc_lvc")
qmout = _load("qmout")

REQUESTS = frozenset({"soc", "dm", "grad", "overlap"})  # what sharc.x asks for (Task 1)


def _lvc_model():
    return LVCModel.from_template(LVC_SOC, spin_orbit=True)


def _provider(model):
    return swap.SocSwapModel(BackendModel(ModelBackend(model), model.masses, coupling="overlap"))


def _two_geometries(model):
    q, v = lvc_wigner_samples(model, 1, np.random.default_rng(2))
    return q[0], q[0] + 5.0 * v[0]


def test_the_provider_qm_out_equals_the_lvc_writer_output():
    model = _lvc_model()
    provider = _provider(model)
    q0, q1 = _two_geometries(model)
    assert provider.states == tuple(model.states)
    assert provider.state_counts == (2, 0, 2)
    mine0 = provider.evaluate(q0, None)
    mine1 = provider.evaluate(q1, mine0)
    theirs0 = model.evaluate(q0, None)
    theirs1 = model.evaluate(q1, theirs0)
    first = frozenset(REQUESTS - {"overlap"})  # step 0 has no OVERLAP request
    assert lvc.format_qm_out(mine0, provider.states, first) == lvc.format_qm_out(
        theirs0, model.states, first
    )
    mine = qmout.parse_qm_out(lvc.format_qm_out(mine1, provider.states, REQUESTS))
    theirs = qmout.parse_qm_out(lvc.format_qm_out(theirs1, model.states, REQUESTS))
    assert mine["hamiltonian"] == theirs["hamiltonian"]
    assert mine["gradients"] == theirs["gradients"]
    assert mine["gradient_labels"] == theirs["gradient_labels"]
    # Lowdin per spin-free block in the adapter, of the whole matrix in the model (Task 4: 3e-16)
    np.testing.assert_allclose(
        np.array(mine["overlap"]), np.array(theirs["overlap"]), rtol=0, atol=1e-12
    )
    assert provider.min_singular_values and provider.min_singular_values[0] > 0.5


def test_the_provider_resets_the_backend_at_a_fresh_start():
    model = _lvc_model()
    backend = ModelBackend(model)
    provider = swap.SocSwapModel(BackendModel(backend, model.masses, coupling="overlap"))
    q0, q1 = _two_geometries(model)
    first = provider.evaluate(q0, None)
    provider.evaluate(q1, first)
    again = provider.evaluate(q0, None)  # sharc.x step 0 again: no overlaps with q1
    assert again.overlaps is None
    np.testing.assert_array_equal(again.energies, first.energies)


def test_the_provider_refuses_singlets_only_and_eselect():
    singlets = LVCModel.from_template(DATA / "lvc" / "lvc_singlets.template", spin_orbit=False)
    with pytest.raises(ValueError, match="triplets"):
        swap.SocSwapModel(BackendModel(ModelBackend(singlets), singlets.masses, coupling="overlap"))
    model = _lvc_model()
    with pytest.raises(ValueError, match="eselect"):
        swap.SocSwapModel(
            BackendModel(ModelBackend(model), model.masses, coupling="overlap", eselect_Eh=0.05)
        )


def test_the_state_order_needs_no_permutation_and_round_trips():
    """Task 1: SHARC's order is ours. The QM.out labels are the canonical expanded order and the
    Hamiltonian read back is the provider's, element for element."""
    model = _lvc_model()
    provider = _provider(model)
    snapshot = provider.evaluate(_two_geometries(model)[0], None)
    parsed = qmout.parse_qm_out(
        lvc.format_qm_out(snapshot, provider.states, REQUESTS - {"overlap"})
    )
    expected = [(s.multiplicity, s.index + 1, s.ms) for s in expanded_states((2, 0, 2))]
    assert parsed["gradient_labels"] == expected
    assert expected[:4] == [(1, 1, 0), (1, 2, 0), (3, 1, -1), (3, 2, -1)]
    np.testing.assert_array_equal(np.array(parsed["hamiltonian"]), snapshot.hamiltonian)
    np.testing.assert_array_equal(
        np.array(parsed["gradients"]).reshape(8, -1), np.asarray(snapshot.gradients)
    )


def test_the_sharc_input_carries_the_spec_settings():
    options = swap.dynamics_options(hopping="sharc", hop_energy="previous")
    lines = swap.sharc_input(options, 200, (2, 0, 2), 2029).splitlines()
    for expected in (
        "nstates 2 0 2",
        "state 2 mch",
        "coeff auto",
        "stepsize 0.5",
        "nsubsteps 25",
        "surf diagonal",
        "spinorbit",
        "coupling overlap",
        "eeom ld",
        "nogradcorrect",
        "ekincorrect parallel_vel",
        "reflect_frustrated none",
        "decoherence_scheme edc",
        "decoherence_param 0.1",
        "hopping_procedure sharc",
        "grad_all",
        "rngseed 2029",
        "write_grad",
        "write_overlap",
    ):
        assert expected in lines, expected
    off = swap.sharc_input(swap.dynamics_options(hopping="none"), 40, (3, 0, 3), 2029)
    assert "hopping_procedure off" in off.splitlines() and "nstates 3 0 3" in off.splitlines()
    assert options.substep_interpolation == "endpoint" and options.edc_convention == "sharc"
    assert options.gradient_transformation == "none" and options.hop_energy == "previous"


def _records(n_steps=5, n_dof=6, n_sf=4, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "steps": np.arange(n_steps),
        "q": rng.standard_normal((n_steps, n_dof)),
        "v": rng.standard_normal((n_steps, n_dof)),
        "e_total": rng.standard_normal(n_steps),
        "e_active": rng.standard_normal(n_steps),
        "populations": rng.random((n_steps, n_sf)),
        "active": np.array([5, 5, 4, 4, 2][:n_steps]),
        "active_sf": np.array([1, 1, 0, 0, 2][:n_steps]),
    }


def test_differences_of_identical_records_are_zero():
    a = _records()
    diffs = swap.differences(a, {k: np.copy(v) for k, v in a.items()})
    np.testing.assert_array_equal(diffs["steps"], np.arange(5))
    for key in swap.QUANTITIES:
        np.testing.assert_array_equal(diffs[key], np.zeros(5))
    assert diffs["same_active_sf"].all()
    assert swap.violations(diffs, 4) == []
    assert all(entry["max"] == 0.0 for entry in swap.maxima(diffs, 4).values())


def test_differences_of_shifted_records_are_the_shift():
    a = _records()
    b = {k: np.copy(v) for k, v in a.items()}
    b["q"][3, 2] += 2e-5
    b["v"][1] -= 3e-6
    b["e_total"] += 5e-7
    b["e_active"][4] += 4e-6
    b["populations"][2, 1] += 1e-4
    b["active_sf"][3] = 2
    diffs = swap.differences(a, b)
    np.testing.assert_allclose(diffs["q"], [0, 0, 0, 2e-5, 0], rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(diffs["v"], [0, 3e-6, 0, 0, 0], rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(diffs["e_total"], np.full(5, 5e-7), rtol=1e-6, atol=0)
    np.testing.assert_allclose(diffs["e_active"], [0, 0, 0, 0, 4e-6], rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(diffs["populations"], [0, 0, 1e-4, 0, 0], rtol=1e-9, atol=1e-15)
    assert list(diffs["same_active_sf"]) == [True, True, True, False, True]
    top = swap.maxima(diffs, 4)
    assert top["q"]["step"] == 3 and top["v"]["step"] == 1 and top["populations"]["step"] == 2
    assert top["e_active"] == {"max": pytest.approx(4e-6), "step": 4, "bound": 1e-6}
    found = swap.violations(diffs, 4)
    assert any(f.startswith("q at step 3") for f in found)
    assert any(f.startswith("v at step 1") for f in found)
    assert any(f.startswith("e_active at step 4") for f in found)
    assert any(f.startswith("populations at step 2") for f in found)
    assert "active spin-free state at step 3 differs" in found
    assert not any(f.startswith("e_total") for f in found)  # 5e-7 is inside 1e-6
    assert all("step 4" not in f for f in swap.violations(diffs, 3))  # the window is respected


def test_differences_align_by_step_and_refuse_unsorted_steps():
    a = _records()
    b = {k: np.copy(v)[1:] for k, v in a.items()}
    diffs = swap.differences(a, b)
    np.testing.assert_array_equal(diffs["steps"], [1, 2, 3, 4])
    assert not np.any(diffs["q"])
    b["steps"] = np.array([1, 3, 2, 4])
    with pytest.raises(ValueError, match="strictly increasing"):
        swap.differences(a, b)


def test_probabilities_are_compared_when_both_have_them_and_near_ties_are_found():
    a, b = _records(), _records()
    a["probabilities"] = np.zeros((5, 4))
    b["probabilities"] = np.zeros((5, 4))
    b["probabilities"][2, 3] = 0.25
    diffs = swap.differences(a, b)
    np.testing.assert_array_equal(diffs["probabilities"], [0, 0, 0.25, 0, 0])
    assert "probabilities" not in swap.differences(_records(), _records())
    records = {
        "steps": np.arange(4),
        "probabilities": np.array([[0, 0], [0.1, 0.2], [0.1, 0.2], [0.0, 0.5]]),
        "random_numbers": np.array([0.0, 0.3 + 5e-6, 0.31, 0.5]),
    }
    assert swap.near_ties(records, 3) == [1, 3]  # step 0 draws nothing
    assert swap.near_ties(records, 2) == [1]


def test_singlet_triplet_hops_and_the_window():
    records = {
        "steps": np.arange(7),
        # S1 -> S0 hop at 2 (not singlet-triplet), S0 -> T1 hop at 4, a change of the spin-free
        # state without a hop at 5, T1 -> S1 hop at 6
        "active": np.array([5, 5, 0, 0, 2, 2, 6]),
        "active_sf": np.array([1, 1, 0, 0, 2, 3, 1]),
    }
    assert swap.singlet_triplet_hops(records, 2) == [4, 6]
    assert swap.window_end(4) == 24
    assert swap.window_end(190) == 200


def test_the_sharc_records_of_the_probe_output():
    traj = lvc.parse_lvc_output_dat(PROBE / "output.dat")
    records = swap.sharc_records(traj, 2, 2)
    np.testing.assert_array_equal(records["steps"], [0, 1, 2, 3])
    # Task 1: E_tot at step 0 from output.dat (0.16654314837831505 Eh, output.lis 4.531870 eV)
    assert records["e_total"][0] == pytest.approx(0.16654314837831505, abs=1e-12)
    np.testing.assert_allclose(records["populations"].sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert records["populations"].shape == (4, 4)
    assert records["active_sf"][0] == 1 and records["active"][0] == 4  # S1, diagonal state 5
    np.testing.assert_allclose(
        records["e_total"] - records["e_active"], traj.e_kinetic, rtol=0, atol=1e-15
    )
    assert records["random_numbers"][0] == 0.0
    assert swap.replayed_numbers(traj) == [float(x) for x in traj.random_numbers[1:]]


def test_replayed_numbers_need_consecutive_steps():
    numbers = np.array([0.0, 0.25, 0.5, 0.75])
    traj = SimpleNamespace(steps=np.array([0, 1, 2, 3]), random_numbers=numbers)
    assert swap.replayed_numbers(traj) == [0.25, 0.5, 0.75]
    with pytest.raises(ValueError, match="0, 1, 2"):
        swap.replayed_numbers(SimpleNamespace(steps=np.array([0, 2, 4]), random_numbers=numbers))


def _lvc_start(model):
    q, v = lvc_wigner_samples(model, 1, np.random.default_rng(2))
    masses_amu = model.masses[::3] / DALTON_IN_ME
    return swap.Start(
        sample=0,
        symbols=tuple(model.parameters.v0.symbols),
        masses_amu=masses_amu,
        q_bohr=q[0].reshape(-1, 3),
        v_au=v[0].reshape(-1, 3),
        equilibrium_bohr=q[0].reshape(-1, 3),
    )


def test_the_replay_source_consumes_sharc_random_numbers_in_step_order(tmp_path):
    """Our run draws SHARC's number of step k at its step k (block 9; step 0 draws nothing),
    and the records carry the spin-free view the comparison reads."""
    model = _lvc_model()
    start = _lvc_start(model)
    sharc_numbers = [0.0, *np.random.default_rng(5).random(6)]  # as printed: 0.0 at step 0
    traj = SimpleNamespace(steps=np.arange(7), random_numbers=np.array(sharc_numbers))
    options = swap.dynamics_options(hopping="sharc", hop_energy="previous")
    ours = swap.run_ours(
        ModelBackend(model),
        start,
        options,
        n_steps=6,
        work_dir=tmp_path,
        random_numbers=swap.replayed_numbers(traj),
    )
    assert ours["result"]["n_rejections"] == 0 and ours["result"]["n_grid_steps"] == 6
    records = swap.our_records(ours["rows"])
    np.testing.assert_array_equal(records["steps"], np.arange(7))
    np.testing.assert_array_equal(records["random_numbers"][1:], sharc_numbers[1:])
    rows = ours["rows"]
    k = np.arange(len(rows))
    np.testing.assert_array_equal(records["e_active"], rows["energies_diag"][k, rows["active"]])
    assert records["populations"].shape == (7, 4)
    np.testing.assert_allclose(records["populations"].sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert records["active_sf"][0] == 1  # the start in S1
    np.testing.assert_allclose(records["q"][0], start.q_bohr.reshape(-1), rtol=0, atol=0)
    assert (ours["n_singlets"], ours["n_triplets"]) == (2, 2)


def test_our_run_does_not_halve_a_step_sharc_takes_whole(tmp_path):
    """SHARC has no step rejection: a low raw overlap (0.1, below the stage default 0.5) at the
    third call is accepted as SHARC accepts it, so the replayed stream stays aligned."""
    model = _lvc_model()
    start = _lvc_start(model)
    ours = swap.run_ours(
        ModelBackend(model, faults={3: "low_overlap"}),
        start,
        swap.dynamics_options(hopping="none"),
        n_steps=4,
        work_dir=tmp_path,
        random_numbers=[0.5] * 4,
    )
    assert ours["result"]["n_rejections"] == 0, ours["result"]
    assert len(ours["rows"]) == 5 and np.all(ours["rows"]["ticks"] == 0)


def _stub_search(monkeypatch, outcomes):
    """``search`` over the LVC model with ``run_sharc`` replaced: ``outcomes[sample]`` is
    ("records", n_printed, active, active_sf) or ("raise", message)."""
    model = _lvc_model()
    start = _lvc_start(model)
    monkeypatch.setattr(swap, "SETTINGS", {**swap.SETTINGS, "lvc": {"singlets": 2, "triplets": 2}})
    monkeypatch.setattr(swap, "read_start", lambda path, sample: start)
    monkeypatch.setattr(swap, "make_backend", lambda *a, **k: ModelBackend(model))
    monkeypatch.setattr(swap, "masses_per_coordinate", lambda s: model.masses)

    def fake_run_sharc(provider, start, options, *, n_steps, work_dir, sharc_home):
        sample = int(Path(work_dir).name.split("_")[1])
        outcome = outcomes[sample]
        if outcome[0] == "raise":
            raise RuntimeError(outcome[1])
        _, n_printed, active, active_sf = outcome
        return SimpleNamespace(n=n_printed, active=active, active_sf=active_sf)

    def fake_records(traj, n, m):
        return {
            "steps": np.arange(traj.n),
            "active": np.full(traj.n, 4) if traj.active is None else np.asarray(traj.active),
            "active_sf": np.full(traj.n, 1)
            if traj.active_sf is None
            else np.asarray(traj.active_sf),
        }

    monkeypatch.setattr(swap, "run_sharc", fake_run_sharc)
    monkeypatch.setattr(swap, "sharc_records", fake_records)


def test_the_search_stops_at_the_first_sample_with_a_singlet_triplet_hop(monkeypatch, tmp_path):
    full = swap.MAX_STEPS + 1
    active = np.full(full, 4)
    active[30:] = 6  # a hop at step 30 ...
    active_sf = np.full(full, 1)
    active_sf[30:] = 2  # ... from S1 to T1
    _stub_search(
        monkeypatch,
        {0: ("records", full, None, None), 1: ("records", full, active, active_sf)},
    )
    report = swap.search(
        samples_h5=tmp_path / "unused.h5",
        method="lvc",
        sharc_home=tmp_path,
        work_dir=tmp_path / "w",
        out=tmp_path / "search.json",
    )
    assert [r["sample"] for r in report["runs"]] == [0, 1]
    assert report["found"] == {"sample": 1, "first_singlet_triplet_hop": 30}
    assert "stopped" not in report


@pytest.mark.parametrize(
    "outcome", [("raise", "sharc.x failed"), ("records", 120, None, None)], ids=["fail", "short"]
)
def test_an_undetermined_sample_stops_the_search(monkeypatch, tmp_path, outcome):
    """A failed run, or one that printed fewer than 201 steps without a hop, is neither "no hop"
    nor a reason to go on: the rule takes the first sample by index."""
    _stub_search(monkeypatch, {0: ("records", swap.MAX_STEPS + 1, None, None), 1: outcome})
    report = swap.search(
        samples_h5=tmp_path / "unused.h5",
        method="lvc",
        sharc_home=tmp_path,
        work_dir=tmp_path / "w",
        out=tmp_path / "search.json",
    )
    assert [r["sample"] for r in report["runs"]] == [0, 1]
    assert report["found"] is None
    assert report["stopped"]["sample"] == 1 and "error" in report["runs"][1]
    assert "error" not in report["runs"][0]


def test_the_start_round_trips_through_json():
    model = _lvc_model()
    start = _lvc_start(model)
    again = swap.Start.from_json(start.to_json())
    assert again.sample == start.sample and again.symbols == start.symbols
    for name in ("masses_amu", "q_bohr", "v_au", "equilibrium_bohr"):
        np.testing.assert_array_equal(getattr(again, name), getattr(start, name))


def test_jsonable_turns_numpy_values_and_non_finite_floats_into_json():
    data = {"a": np.array([1.0, np.nan]), "b": np.int64(3), "c": (np.float64(2.5), np.bool_(True))}
    assert swap.jsonable(data) == {"a": [1.0, None], "b": 3, "c": [2.5, True]}
