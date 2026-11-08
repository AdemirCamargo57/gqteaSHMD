import json

import h5py
import numpy as np
import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.core.states import states_from_counts
from gqteashmd.electronic.base import ElectronicResult
from gqteashmd.errors import ConfigError, ElectronicStructureError, StageFailedError, TaskError
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.stages import excitations as excitations_module
from gqteashmd.stages.base import StageContext, TaskSpec
from gqteashmd.stages.excitations import ExcitationsStage
from gqteashmd.stages.pipeline import check_stage_inputs
from gqteashmd.stages.prepare import PrepareStage
from gqteashmd.stages.sampling import SamplingStage
from gqteashmd.vib.artifacts import read_samples_h5, write_frequencies_h5
from gqteashmd.vib.harmonic import harmonic_analysis

SYMBOLS = ("O", "H", "H")
MASSES = np.array([15.9949146, 1.00782503, 1.00782503])
COORDS = np.array([[0.0, 0.0, 0.2837], [0.0, 1.4309, -0.9196], [0.0, -1.4309, -0.9196]])
ELECTRONIC = '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = {n}\n'


def _hessian():
    hessian = np.zeros((9, 9))
    for (a, b), k in (((0, 1), 0.5), ((0, 2), 0.5), ((1, 2), 0.1)):
        u = COORDS[a] - COORDS[b]
        u = u / np.linalg.norm(u)
        g = np.zeros(9)
        g[3 * a : 3 * a + 3] = u
        g[3 * b : 3 * b + 3] = -u
        hessian += k * np.outer(g, g)
    return hessian


class FakeBackend:
    def __init__(self, settings, fail_on=(), retryable_on=()):
        self.states = states_from_counts(settings.singlets)
        self.log = []
        self.fail_on = fail_on
        self.retryable_on = retryable_on

    def reset(self):
        self.log.append("reset")

    def compute(self, request):
        self.log.append("compute")
        x0 = round(float(request.coords_bohr[0, 2]), 6)
        if x0 in self.fail_on:
            raise ElectronicStructureError("SCF did not converge")
        if x0 in self.retryable_on:
            raise ElectronicStructureError("disk full", retryable=True)
        energies = {s: -75.0 + 0.2 * i + 0.01 * i * x0 for i, s in enumerate(self.states)}
        mu = {s: np.array([0.1 * i, 0.0, 0.05]) for i, s in enumerate(self.states) if i > 0}
        return ElectronicResult(self.states, energies, transition_dipoles=mu)


def _context(tmp_path, write_input, extra="", singlets=3, n_samples=5):
    tmp_path.mkdir(parents=True, exist_ok=True)
    text = ELECTRONIC.format(n=singlets) + f"[sampling]\nn_samples = {n_samples}\n" + extra
    cfg = load_config(write_input(tmp_path, extra=text)).config
    layout = ProjectLayout(cfg.project.directory)
    prep_dir = layout.stage_dir(0, "prepare")
    prep_dir.mkdir(parents=True)
    prepare = PrepareStage()
    pctx = StageContext(cfg, layout, prep_dir, {})
    prepare.finalize(pctx, {"molecule": prepare.run_task(pctx, TaskSpec("molecule"), prep_dir)})
    freq_dir = layout.stage_dir(2, "frequencies")
    freq_dir.mkdir(parents=True)
    hessian = _hessian()
    write_frequencies_h5(
        freq_dir / "frequencies.h5",
        symbols=SYMBOLS,
        masses_amu=MASSES,
        coords_bohr=COORDS,
        hessian=hessian,
        analysis=harmonic_analysis(MASSES, COORDS, hessian),
    )
    samp_dir = layout.stage_dir(3, "sampling")
    samp_dir.mkdir(parents=True)
    sctx = StageContext(cfg, layout, samp_dir, {"frequencies": freq_dir})
    sampling = SamplingStage()
    task_dir = layout.task_dir(samp_dir, "samples")
    task_dir.mkdir(parents=True)
    sampling.finalize(sctx, {"samples": sampling.run_task(sctx, TaskSpec("samples"), task_dir)})
    stage_dir = layout.stage_dir(4, "excitations")
    stage_dir.mkdir(parents=True)
    return StageContext(cfg, layout, stage_dir, {"prepare": prep_dir, "sampling": samp_dir})


@pytest.fixture
def fake(monkeypatch):
    holder = {"backends": [], "fail_on": (), "retryable_on": ()}

    def factory(ctx, stage_name, molecule, task_dir, *, settings=None):
        backend = FakeBackend(settings, holder["fail_on"], holder["retryable_on"])
        holder["backends"].append(backend)
        holder["settings"] = settings
        return backend

    monkeypatch.setattr(excitations_module, "stage_backend", factory)
    return holder


def _run(ctx, stage=None, skip=()):
    stage = stage or ExcitationsStage()
    results = {}
    for task in stage.plan(ctx):
        if task.task_id in skip:
            continue
        task_dir = ctx.layout.task_dir(ctx.stage_dir, task.task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        results[task.task_id] = stage.run_task(ctx, task, task_dir)
    return stage, results


def _samples(ctx):
    return read_samples_h5(ctx.upstream_dir("sampling") / "samples.h5")


def test_stage_attributes():
    stage = ExcitationsStage()
    assert stage.name == "excitations" and stage.depends_on == ("prepare", "sampling")
    assert stage.config_sections == ("electronic", "excitations")
    assert stage.required_modules == ("pyscf>=2.6", "h5py>=3.9")
    assert stage.max_failed_fraction == 0.05 and ExcitationsStage(0.2).max_failed_fraction == 0.2


def test_plan(tmp_path, write_input):
    ctx = _context(tmp_path / "a", write_input, "[excitations]\nsamples_per_task = 2\n")
    tasks = ExcitationsStage().plan(ctx)
    assert [t.task_id for t in tasks] == ["ref", "s00001", "s00003", "s00005"]
    assert tasks[3].payload["samples"] == [5]
    extra = "[excitations]\nsamples_per_task = 2\ninclude_reference = false\n"
    ctx = _context(tmp_path / "b", write_input, extra)
    assert [t.task_id for t in ExcitationsStage().plan(ctx)] == ["s00001", "s00003", "s00005"]


def test_run_task_rows_and_resets(tmp_path, write_input, fake):
    ctx = _context(tmp_path, write_input, "[excitations]\nsamples_per_task = 2\n")
    stage = ExcitationsStage()
    task = next(t for t in stage.plan(ctx) if t.task_id == "s00001")
    task_dir = ctx.layout.task_dir(ctx.stage_dir, "s00001")
    task_dir.mkdir(parents=True)
    result = stage.run_task(ctx, task, task_dir)
    rows = result["samples"]
    assert [r["sample"] for r in rows] == [1, 2] and all(r["ok"] for r in rows)
    assert fake["backends"][0].log == ["reset", "compute", "reset", "compute"]
    r = rows[0]
    de = r["energies_hartree"][1] - r["energies_hartree"][0]
    mu = np.array(r["transition_dipoles_au"][0])
    assert r["oscillator_strengths"][0] == pytest.approx(2 / 3 * de * float(mu @ mu))
    assert np.shape(r["transition_dipoles_au"]) == (2, 3)
    json.dumps(result)


def test_backend_gets_full_settings(tmp_path, write_input, fake):
    ctx = _context(tmp_path, write_input, singlets=3)
    _run(ctx)
    assert fake["settings"].singlets == 3


def test_failed_sample_is_recorded(tmp_path, write_input, fake):
    ctx = _context(tmp_path, write_input, "[excitations]\nsamples_per_task = 5\n")
    fake["fail_on"] = (round(float(_samples(ctx)["positions_bohr"][1, 0, 2]), 6),)
    _, results = _run(ctx)
    rows = results["s00001"]["samples"]
    assert [r["ok"] for r in rows] == [True, False, True, True, True]
    assert "SCF did not converge" in rows[1]["error"]


def test_retryable_error_propagates(tmp_path, write_input, fake):
    ctx = _context(tmp_path, write_input, "[excitations]\nsamples_per_task = 5\n")
    fake["retryable_on"] = (round(float(_samples(ctx)["positions_bohr"][0, 0, 2]), 6),)
    with pytest.raises(TaskError) as info:
        _run(ctx)
    assert info.value.retryable


def test_finalize_writes_files(tmp_path, write_input, fake):
    ctx = _context(tmp_path, write_input, "[excitations]\nsamples_per_task = 2\n", n_samples=20)
    fake["fail_on"] = (round(float(_samples(ctx)["positions_bohr"][2, 0, 2]), 6),)
    stage, results = _run(ctx, ExcitationsStage(max_failed_fraction=0.1))
    stage.finalize(ctx, results)
    with h5py.File(ctx.stage_dir / "excitations.h5", "r") as f:
        assert f["energies"].shape == (20, 3)
        assert f["transition_dipoles"].shape == (20, 2, 3)
        assert f["reference/energies"].shape == (1, 3)
        ok = f["ok"][()]
        assert ok.sum() == 19 and ok[2] == 0
        assert np.isnan(f["energies"][2]).all()
    assert (ctx.stage_dir / "spectrum.csv").read_text().startswith("# gqteashmd")
    summary = json.loads((ctx.stage_dir / "excitations.json").read_text())
    assert summary["n_samples"] == 20 and summary["n_ok"] == 19
    assert summary["failed"][0]["sample"] == 3
    assert summary["state_labels"] == ["S0", "S1", "S2"]
    assert set(summary["states"]) == {"S1", "S2"}
    assert summary["states"]["S1"]["reference_excitation_eV"] > 0


def test_failure_fraction_limit(tmp_path, write_input, fake):
    ctx = _context(tmp_path, write_input, "[excitations]\nsamples_per_task = 2\n", n_samples=5)
    stage, results = _run(ctx, skip=("s00001",))  # 2 of 5 samples lost
    with pytest.raises(StageFailedError, match="max_failed_fraction"):
        stage.finalize(ctx, results)
    ExcitationsStage(max_failed_fraction=0.4).finalize(ctx, results)  # equality is allowed
    with h5py.File(ctx.stage_dir / "excitations.h5", "r") as f:
        assert f["ok"][()].tolist() == [0, 0, 1, 1, 1]


def test_failed_reference_raises(tmp_path, write_input, fake):
    ctx = _context(tmp_path, write_input, n_samples=5)
    fake["fail_on"] = (round(float(COORDS[0, 2]), 6),)
    stage, results = _run(ctx)
    assert results["ref"]["samples"][0]["ok"] is False
    with pytest.raises(StageFailedError, match="equilibrium geometry itself failed"):
        ExcitationsStage(max_failed_fraction=0.5).finalize(ctx, results)


def test_fingerprint(tmp_path, write_input):
    stage = ExcitationsStage()
    counter = iter(range(100))

    def fp(extra="", **kw):
        return stage.fingerprint_inputs(
            _context(tmp_path / f"fp{next(counter)}", write_input, extra, **kw)
        )

    base = fp()
    assert fp("[excitations]\nsamples_per_task = 2\n") != base
    assert fp("[excitations]\nbroadening_fwhm_eV = 0.3\n") == base
    assert fp("[excitations]\nmax_failed_fraction = 0.5\n") == base
    assert fp("[excitations]\nspectrum_step_eV = 0.05\n") == base
    assert fp(n_samples=6) != base
    other = stage.fingerprint_inputs(_context(tmp_path / "o", write_input))
    other["electronic"]["basis"] = "6-31g"
    assert other != base


def test_singlets_below_two_is_a_config_error(tmp_path, write_input):
    ctx = _context(tmp_path, write_input, singlets=1)
    with pytest.raises(ConfigError, match="singlets must be >= 2"):
        check_stage_inputs(ctx.config, [ExcitationsStage()])
    ok = _context(tmp_path / "ok", write_input, singlets=2)
    check_stage_inputs(ok.config, [ExcitationsStage()])
