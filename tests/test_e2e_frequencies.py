import json
from pathlib import Path

import pytest

from gqteashmd.cli import main
from gqteashmd.config.loader import load_config
from gqteashmd.project.layout import ProjectLayout

pytestmark = pytest.mark.pyscf
HF = '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 2\n'


def test_run_optimizes_and_computes_frequencies(tmp_path, write_input):
    path = write_input(tmp_path, extra=HF)
    assert main(["run", str(path)]) == 0
    layout = ProjectLayout(load_config(path).config.project.directory)
    opt = json.loads((layout.stage_dir(1, "optimize") / "optimize.json").read_text())
    freq = json.loads((layout.stage_dir(2, "frequencies") / "frequencies.json").read_text())
    assert opt["converged"] and freq["geometry_source"] == "optimize"
    assert len(freq["frequencies_cm1"]) == 3 and freq["imaginary_modes"] == []
    assert all(1500 < f < 4500 for f in freq["frequencies_cm1"])
    assert main(["run", str(path)]) == 0  # up to date: nothing recomputed


def test_disabling_optimize_after_a_run_is_refused(tmp_path, write_input):
    path = write_input(tmp_path, extra=HF)
    assert main(["run", str(path)]) == 0
    path.write_text(path.read_text() + "\n[optimize]\nenabled = false\n")
    assert main(["run", str(path)]) == 3  # frequencies was computed from the optimized geometry


@pytest.mark.slow
def test_casscf_frequencies_by_finite_differences(tmp_path, write_input):
    ethylene = (Path(__file__).parent / "data" / "ethylene.xyz").read_text()
    cas = (
        '[electronic]\nmethod = "casscf"\nbasis = "sto-3g"\nsinglets = 2\n'
        "[electronic.casscf]\nncas = 2\nnelecas = 2\n[optimize]\nenabled = false\n"
        "[frequencies]\nimaginary_cutoff_cm1 = 10000.0\n"
    )
    path = write_input(tmp_path, extra=cas)
    (tmp_path / "water.xyz").write_text(ethylene)  # write_input points [molecule] at water.xyz
    assert main(["run", str(path)]) == 0
    layout = ProjectLayout(load_config(path).config.project.directory)
    freq = json.loads((layout.stage_dir(2, "frequencies") / "frequencies.json").read_text())
    assert freq["method"] == "finite_difference" and len(freq["frequencies_cm1"]) == 12
