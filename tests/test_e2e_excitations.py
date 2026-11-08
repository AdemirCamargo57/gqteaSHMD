import json

import pytest

from gqteashmd.cli import main
from gqteashmd.config.loader import load_config
from gqteashmd.excite.artifacts import read_excitations_h5
from gqteashmd.project.layout import ProjectLayout

pytestmark = pytest.mark.pyscf
EXTRA = (
    '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 3\n'
    "[sampling]\nn_samples = 4\n[excitations]\nsamples_per_task = 2\n"
    "[selection]\nn_trajectories = 2\nallow_fewer = true\n"
)


def _layout(path):
    return ProjectLayout(load_config(path).config.project.directory)


def test_excitations_and_selection_end_to_end(tmp_path, write_input):
    path = write_input(tmp_path, extra=EXTRA)
    assert main(["run", str(path)]) == 0
    layout = _layout(path)
    edir = layout.stage_dir(4, "excitations")
    data = read_excitations_h5(edir / "excitations.h5")
    assert int(data["ok"].sum()) == 4
    lines = [x for x in (edir / "spectrum.csv").read_text().splitlines() if not x.startswith("#")]
    header = lines[0].split(",")
    col = header.index("cross_section_A2")
    assert max(float(r.split(",")[col]) for r in lines[1:]) > 0
    sel = layout.stage_dir(5, "selection")
    entries = json.loads((sel / "trajectories.json").read_text())
    entries = (
        entries
        if isinstance(entries, list)
        else next(v for v in entries.values() if isinstance(v, list))
    )
    assert 1 <= len(entries) <= 2
    assert main(["run", str(path)]) == 0  # up to date


def test_pause_after_excitations(tmp_path, write_input):
    path = write_input(tmp_path, extra=EXTRA + '[workflow]\npause_after = ["excitations"]\n')
    assert main(["run", str(path)]) == 5
    layout = _layout(path)
    assert (layout.stage_dir(4, "excitations") / "excitations.h5").exists()
    assert not (layout.stage_dir(5, "selection") / "trajectories.json").exists()
    assert main(["run", str(path)]) == 0
    assert (layout.stage_dir(5, "selection") / "trajectories.json").exists()
