import json
from pathlib import Path

from gqteashmd.cli import main

ETHYLENE = (Path(__file__).resolve().parent / "data" / "ethylene.xyz").read_text()
CAS = (
    '[electronic]\nmethod = "casscf"\nbasis = "6-31g"\nsinglets = 2\n'
    "[electronic.casscf]\nncas = 2\nnelecas = 2\n"
)


def _input(tmp_path, extra=CAS, xyz=ETHYLENE):
    (tmp_path / "ethylene.xyz").write_text(xyz)
    path = tmp_path / "input.toml"
    path.write_text(
        '[project]\nname = "sp"\ndirectory = "run"\n\n[molecule]\ngeometry = "ethylene.xyz"\n\n'
        "[resources]\ntotal_cores = 4\ntotal_memory_mb = 4000\n\n" + extra
    )
    return path


def test_single_point_prints_a_table_and_writes_json(tmp_path, capsys):
    out = tmp_path / "sp.json"
    log = tmp_path / "pyscf.log"
    assert main(["single-point", str(_input(tmp_path)), "--json", str(out), "--log", str(log)]) == 0
    text = capsys.readouterr().out
    assert "casscf / 6-31g" in text and "S0" in text and "S1" in text
    data = json.loads(out.read_text())
    assert [row["label"] for row in data["states"]] == ["S0", "S1"]
    assert data["states"][1]["oscillator_strength"] > 0.1
    assert log.stat().st_size > 0
    assert not (tmp_path / "run").exists()  # a single point does not create a project


def test_missing_electronic_section_is_exit_2(tmp_path, capsys):
    assert main(["single-point", str(_input(tmp_path, extra=""))]) == 2
    assert "[electronic]" in capsys.readouterr().err


def test_invalid_geometry_is_exit_2(tmp_path, capsys):
    assert main(["single-point", str(_input(tmp_path, xyz="2\n\nC 0 0 0\n"))]) == 2
    assert "Invalid geometry file" in capsys.readouterr().err


def test_unknown_basis_is_exit_4(tmp_path, capsys):
    path = _input(tmp_path, extra=CAS.replace("6-31g", "no-such-basis"))
    assert main(["single-point", str(path)]) == 4
    assert "basis 'no-such-basis'" in capsys.readouterr().err
