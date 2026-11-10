import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gqteashmd.dynamics.driver import initial_state, run_trajectory
from gqteashmd.dynamics.models import TullySimpleAvoidedCrossing, TwoDimensionalCrossing
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.randomness import GeneratorSource, ReplaySource
from gqteashmd.dynamics.records import ElectronicSnapshot
from gqteashmd.dynamics.restart import load_state, save_state

MASS = 2000.0


def _sac_state(model):
    return initial_state(model, np.array([-1.5]), np.array([30.0 / MASS]), 0)


def _assert_same(a, b):
    assert np.array_equal(a.q, b.q)
    assert np.array_equal(a.v, b.v)
    assert np.array_equal(a.c, b.c)
    assert a.active == b.active
    assert a.step == b.step and a.time_au == b.time_au
    assert a.e_total_initial == b.e_total_initial


@pytest.mark.parametrize("scheme", ["sharc", "fssh"])
def test_restart_is_bit_identical(tmp_path, scheme):
    model = TullySimpleAvoidedCrossing()
    options = DynamicsOptions(dt_au=2.0, hopping=scheme, decoherence="edc")
    start = _sac_state(model)

    hops: list[int] = []
    full, _ = run_trajectory(
        model,
        start,
        options,
        GeneratorSource(np.random.default_rng(5)),
        max_steps=100,
        on_step=lambda r: hops.extend([r.step] if r.hop_to is not None else []),
    )
    # hops on both sides of the save at step 50 (measured: steps 32 and 89, both schemes)
    assert any(h <= 50 for h in hops) and any(h > 50 for h in hops)

    rng = GeneratorSource(np.random.default_rng(5))
    half, _ = run_trajectory(model, start, options, rng, max_steps=50)
    path = tmp_path / "restart.npz"
    save_state(path, half, rng)
    loaded, rng2 = load_state(path, model)
    _assert_same(loaded, half)
    resumed, _ = run_trajectory(model, loaded, options, rng2, max_steps=50)

    _assert_same(resumed, full)
    assert np.array_equal(resumed.snapshot.energies, full.snapshot.energies)
    assert np.array_equal(resumed.snapshot.vectors, full.snapshot.vectors)
    assert resumed.step == 100


def test_restart_round_trips_the_snapshot_and_replay_source(tmp_path):
    model = TwoDimensionalCrossing()
    options = DynamicsOptions(dt_au=1.0)
    state = initial_state(
        model, np.array([1.0, 0.5]), np.array([0.0, 0.002]), 1, np.array([0.6, 0.8j])
    )
    rng = ReplaySource([0.3] * 20 + [0.7] * 5)
    state, _ = run_trajectory(model, state, options, rng, max_steps=20)
    path = tmp_path / "sub" / "r.npz"
    save_state(path, state, rng)
    loaded, rng2 = load_state(path, model)
    for name in ("energies", "gradients", "nacs", "hamiltonian", "vectors"):
        a, b = getattr(state.snapshot, name), getattr(loaded.snapshot, name)
        assert a.dtype == b.dtype and np.array_equal(a, b), name
    assert loaded.c.dtype == complex
    assert rng2.state() == rng.state()
    assert [rng2.uniform() for _ in range(5)] == [0.7] * 5
    assert [p.name for p in path.parent.iterdir()] == ["r.npz"]  # no temporary file left


class _CountingModel(TullySimpleAvoidedCrossing):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def evaluate(self, q, previous):
        self.calls += 1
        return super().evaluate(q, previous)


def test_load_state_evaluates_nothing(tmp_path):
    model = TullySimpleAvoidedCrossing()
    state = _sac_state(model)
    path = tmp_path / "r.npz"
    save_state(path, state, GeneratorSource(np.random.default_rng(1)))
    counting = _CountingModel()
    load_state(path, counting)
    assert counting.calls == 0


def test_snapshot_without_vectors(tmp_path):
    from dataclasses import replace

    model = TullySimpleAvoidedCrossing()
    state = _sac_state(model)
    state.snapshot = replace(state.snapshot, vectors=None)
    path = tmp_path / "r.npz"
    save_state(path, state, ReplaySource([0.5]))
    loaded, _ = load_state(path, model)
    assert loaded.snapshot.vectors is None


def test_save_overwrites_atomically(tmp_path):
    model = TullySimpleAvoidedCrossing()
    rng = ReplaySource([0.5] * 10)
    state = _sac_state(model)
    path = tmp_path / "r.npz"
    save_state(path, state, rng)
    state2, _ = run_trajectory(model, state, DynamicsOptions(dt_au=1.0), rng, max_steps=3)
    save_state(path, state2, rng)
    loaded, _ = load_state(path, model)
    assert loaded.step == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == ["r.npz"]


def test_unreadable_or_mismatched_restart(tmp_path):
    model = TullySimpleAvoidedCrossing()
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a restart")
    with pytest.raises(ValueError, match="not a readable gqteashmd dynamics restart"):
        load_state(bad, model)
    with pytest.raises(ValueError, match="not a readable gqteashmd dynamics restart"):
        load_state(tmp_path / "missing.npz", model)
    other = tmp_path / "other.npz"
    np.savez(other, x=np.zeros(3))
    with pytest.raises(ValueError, match="not a readable gqteashmd dynamics restart"):
        load_state(other, model)

    path = tmp_path / "r.npz"
    save_state(path, _sac_state(model), ReplaySource([0.5]))
    with pytest.raises(ValueError, match="does not match the model"):
        load_state(path, TwoDimensionalCrossing())


# ---- restart format 2 (P7a-2) ----

DATA = Path(__file__).parent / "data"


def _snapshot_with_extras(model, state):
    rng = np.random.default_rng(2)
    n = model.n_states
    u = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
    overlaps = rng.normal(size=(n, n))
    snap = state.snapshot
    snap2 = ElectronicSnapshot(
        energies=snap.energies,
        gradients=snap.gradients,
        nacs=snap.nacs,
        hamiltonian=snap.hamiltonian,
        vectors=snap.vectors,
        overlaps=overlaps,
    )
    return replace(state, snapshot=snap2, u=u), u, overlaps


def test_restart_2_round_trips_u_and_overlaps(tmp_path):
    model = TullySimpleAvoidedCrossing()
    state, u, overlaps = _snapshot_with_extras(model, _sac_state(model))
    rng = GeneratorSource(np.random.default_rng(1))
    save_state(tmp_path / "r.npz", state, rng)
    loaded, _ = load_state(tmp_path / "r.npz", model)
    assert loaded.u is not None and loaded.u.dtype == complex and np.array_equal(loaded.u, u)
    assert loaded.snapshot.overlaps is not None
    assert np.array_equal(loaded.snapshot.overlaps, overlaps)


def test_restart_2_without_u_and_overlaps_loads_none(tmp_path):
    model = TullySimpleAvoidedCrossing()
    rng = GeneratorSource(np.random.default_rng(1))
    save_state(tmp_path / "r.npz", _sac_state(model), rng)
    loaded, _ = load_state(tmp_path / "r.npz", model)
    assert loaded.u is None and loaded.snapshot.overlaps is None


def test_version_1_file_written_by_p7a1_loads_with_u_none():
    model = TullySimpleAvoidedCrossing()
    loaded, rng = load_state(DATA / "restart_v1.npz", model)
    assert loaded.u is None and loaded.snapshot.overlaps is None
    assert loaded.step == 40
    # and it continues
    state, _ = run_trajectory(model, loaded, DynamicsOptions(dt_au=1.0), rng, max_steps=3)
    assert state.step == 43


def _forge(src: Path, dst: Path, *, version=None, hamiltonian=None, drop=(), **meta_changes):
    with np.load(src, allow_pickle=False) as data:
        arrays = {k: np.array(data[k]) for k in data.files}
    meta = json.loads(str(arrays["meta"]))
    if version is not None:
        meta["version"] = version
    meta.update(meta_changes)
    arrays["meta"] = np.array(json.dumps(meta))
    if hamiltonian is not None:
        arrays["hamiltonian"] = hamiltonian
    for key in drop:
        arrays.pop(key)
    np.savez(dst, **arrays)


def test_version_1_with_off_diagonal_hamiltonian_is_refused(tmp_path):
    model = TullySimpleAvoidedCrossing()
    h = np.array([[0.0, 0.01], [0.01, 0.1]], dtype=complex)
    _forge(DATA / "restart_v1.npz", tmp_path / "bad.npz", hamiltonian=h)
    with pytest.raises(ValueError, match="off-diagonal"):
        load_state(tmp_path / "bad.npz", model)


def test_unknown_version_is_refused(tmp_path):
    _forge(DATA / "restart_v1.npz", tmp_path / "bad.npz", version=3)
    with pytest.raises(ValueError, match="not a readable"):
        load_state(tmp_path / "bad.npz", TullySimpleAvoidedCrossing())


def test_u_and_overlaps_shapes_are_checked_against_the_model(tmp_path):
    model = TullySimpleAvoidedCrossing()
    state, u, overlaps = _snapshot_with_extras(model, _sac_state(model))
    rng = GeneratorSource(np.random.default_rng(1))
    save_state(tmp_path / "r.npz", state, rng)
    with np.load(tmp_path / "r.npz", allow_pickle=False) as data:
        arrays = {k: np.array(data[k]) for k in data.files}
    for key, bad in (("u", np.eye(3, dtype=complex)), ("overlaps", np.zeros((2, 3)))):
        forged = dict(arrays, **{key: bad})
        np.savez(tmp_path / f"{key}.npz", **forged)
        with pytest.raises(ValueError, match=key):
            load_state(tmp_path / f"{key}.npz", model)


def test_flags_must_match_the_arrays(tmp_path):
    model = TullySimpleAvoidedCrossing()
    state, _, _ = _snapshot_with_extras(model, _sac_state(model))
    rng = GeneratorSource(np.random.default_rng(1))
    save_state(tmp_path / "r.npz", state, rng)
    _forge(tmp_path / "r.npz", tmp_path / "a.npz", drop=("u",))
    with pytest.raises(ValueError, match="u"):
        load_state(tmp_path / "a.npz", model)
    _forge(tmp_path / "r.npz", tmp_path / "b.npz", drop=("overlaps",))
    with pytest.raises(ValueError, match="overlaps"):
        load_state(tmp_path / "b.npz", model)


LVC = Path(__file__).resolve().parents[1] / "data" / "lvc"


@pytest.mark.parametrize("coupling", ["nacdr", "overlap"])
def test_restart_with_spin_orbit_coupling_and_hops_is_bit_identical(tmp_path, coupling):
    """lvc_soc, EDC: 60 + save/load + 60 = 120 steps equal 120 uninterrupted steps, with at
    least one hop in each half; U and the snapshot overlaps are carried by the restart."""
    from gqteashmd.dynamics.lvc import LVCModel, lvc_wigner_samples

    model = LVCModel.from_template(LVC / "lvc_soc.template")
    options = DynamicsOptions(dt_au=0.5, coupling=coupling, decoherence="edc")
    q, v = lvc_wigner_samples(model, 1, np.random.default_rng(7))
    start = initial_state(model, q[0], v[0], 1)
    assert start.u is not None
    stream = _soc_stream()

    full_records: list = []
    full, _ = run_trajectory(
        model, start, options, ReplaySource(stream), max_steps=120, on_step=full_records.append
    )
    hops = [r.step for r in full_records if r.hop_to is not None]
    assert any(h <= 60 for h in hops) and any(h > 60 for h in hops), hops

    rng = ReplaySource(stream)
    half, _ = run_trajectory(model, start, options, rng, max_steps=60)
    path = tmp_path / "restart.npz"
    save_state(path, half, rng)
    loaded, rng2 = load_state(path, model)
    _assert_same(loaded, half)
    assert np.array_equal(loaded.u, half.u)
    assert np.array_equal(loaded.snapshot.overlaps, half.snapshot.overlaps)
    second: list = []
    resumed, _ = run_trajectory(model, loaded, options, rng2, max_steps=60, on_step=second.append)

    _assert_same(resumed, full)
    assert np.array_equal(resumed.u, full.u)
    assert resumed.step == 120
    for a, b in zip(second, full_records[60:], strict=True):
        assert np.array_equal(a.c, b.c) and np.array_equal(a.u, b.u)
        assert a.e_total == b.e_total and a.hop_to == b.hop_to


def _soc_stream() -> list[float]:
    """The seed-21 generator's first 120 numbers with every 10th replaced by 0 (a forced hop
    attempt wherever a probability is positive; the plain probabilities are about 1e-6), so
    both halves contain hops (measured: steps 1, 21, 24, 61, 101 for both couplings)."""
    values = np.random.default_rng(21).random(120)
    values[::10] = 0.0
    return [float(x) for x in values]


def test_restart_2_with_u_continues_bit_identically(tmp_path):
    """A state carrying U (the driver passes U through unchanged until Task 6) round-trips."""
    model = TullySimpleAvoidedCrossing()
    state, u, _ = _snapshot_with_extras(model, _sac_state(model))
    rng = GeneratorSource(np.random.default_rng(4))
    save_state(tmp_path / "a.npz", state, rng)
    a, _ = load_state(tmp_path / "a.npz", model)
    save_state(tmp_path / "b.npz", a, rng)
    b, _ = load_state(tmp_path / "b.npz", model)
    assert np.array_equal(a.u, b.u)
