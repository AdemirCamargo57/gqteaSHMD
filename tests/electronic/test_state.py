import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from gqteashmd.core.states import StateLabel
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.pyscf_backend import PySCFBackend

S0, S1, S2 = StateLabel(1, 0), StateLabel(1, 1), StateLabel(1, 2)


def _step(coords):
    return coords + np.random.default_rng(5).normal(scale=0.02, size=coords.shape)


@pytest.mark.parametrize(
    "fields, request_kw",
    [
        ({"basis": "6-31g"}, {"gradients": frozenset({S1}), "nacs": frozenset({(S0, S1)})}),
        (
            {
                "method": "tda",
                "basis": "sto-3g",
                "functional": "b3lyp",
                "singlets": 3,
                "casscf": None,
            },
            {"gradients": frozenset({S2}), "transition_dipoles": True},
        ),
    ],
)
def test_restart_reproduces_the_continued_run(
    tmp_path, ethylene_symbols, ethylene_bohr, make_settings, fields, request_kw
):
    settings = make_settings(**fields)
    spec = MoleculeSpec(tuple(ethylene_symbols))
    r2 = _step(ethylene_bohr)

    backend_a = PySCFBackend(spec, settings)
    backend_a.compute(Request(ethylene_bohr, overlaps=True))
    backend_a.save_state(tmp_path / "state.npz")
    a = backend_a.compute(Request(r2, overlaps=True, **request_kw))

    backend_b = PySCFBackend(spec, settings)
    backend_b.load_state(tmp_path / "state.npz")
    b = backend_b.compute(Request(r2, overlaps=True, **request_kw))

    # a.phases and b.phases are deliberately not compared: PySCF's raw CI vector signs are not
    # reproducible between objects or processes. The phased outputs below are reproducible
    # relative to the saved snapshot, which is what a restart needs.
    for label in a.states:
        assert abs(a.energies[label] - b.energies[label]) < 1e-9
    for label, g in a.gradients.items():
        assert np.abs(g - b.gradients[label]).max() < 1e-9
    for pair, d in a.nacs.items():
        assert np.abs(d - b.nacs[pair]).max() < 1e-9
    np.testing.assert_allclose(a.overlaps, b.overlaps, atol=1e-9)
    for label, mu in a.transition_dipoles.items():
        assert np.abs(mu - b.transition_dipoles[label]).max() < 1e-9


def test_restart_in_a_new_process(tmp_path, ethylene_symbols, ethylene_bohr, make_settings):
    settings = make_settings(basis="6-31g")
    spec = MoleculeSpec(tuple(ethylene_symbols))
    r2 = _step(ethylene_bohr)
    kw = {"gradients": frozenset({S1}), "nacs": frozenset({(S0, S1)})}

    backend_a = PySCFBackend(spec, settings)
    backend_a.compute(Request(ethylene_bohr, overlaps=True))
    state = tmp_path / "state.npz"
    backend_a.save_state(state)
    a = backend_a.compute(Request(r2, overlaps=True, **kw))

    job = tmp_path / "job.json"
    out = tmp_path / "out.json"
    job.write_text(
        json.dumps(
            {
                "state": str(state),
                "out": str(out),
                "symbols": list(ethylene_symbols),
                "coords": r2.tolist(),
                "settings": settings.model_dump(mode="json"),
            }
        )
    )
    script = textwrap.dedent(
        """
        import json, sys
        import numpy as np
        from gqteashmd.config.schema import ElectronicSection
        from gqteashmd.core.states import StateLabel
        from gqteashmd.electronic.base import MoleculeSpec, Request
        from gqteashmd.electronic.pyscf_backend import PySCFBackend

        job = json.load(open(sys.argv[1]))
        backend = PySCFBackend(
            MoleculeSpec(tuple(job["symbols"])), ElectronicSection(**job["settings"])
        )
        backend.load_state(job["state"])
        s0, s1 = StateLabel(1, 0), StateLabel(1, 1)
        res = backend.compute(
            Request(np.array(job["coords"]), overlaps=True, nacs=frozenset({(s0, s1)}))
        )
        json.dump(
            {"nac": res.nacs[(s0, s1)].tolist(), "overlaps": res.overlaps.tolist()},
            open(job["out"], "w"),
        )
        """
    )
    subprocess.run([sys.executable, "-c", script, str(job)], check=True, timeout=600)
    got = json.loads(out.read_text())
    assert np.abs(np.array(got["nac"]) - a.nacs[(S0, S1)]).max() < 1e-9
    np.testing.assert_allclose(np.array(got["overlaps"]), a.overlaps, atol=1e-9)


def test_state_errors(tmp_path, ethylene_symbols, ethylene_bohr, make_settings):
    spec = MoleculeSpec(tuple(ethylene_symbols))
    backend = PySCFBackend(spec, make_settings(basis="sto-3g"))
    with pytest.raises(ValueError, match="compute"):
        backend.save_state(tmp_path / "s.npz")
    backend.compute(Request(ethylene_bohr))
    backend.save_state(tmp_path / "s.npz")
    other = PySCFBackend(spec, make_settings(basis="6-31g"))
    with pytest.raises(ValueError, match="different settings"):
        other.load_state(tmp_path / "s.npz")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s.npz"]  # atomic: no temp file left
