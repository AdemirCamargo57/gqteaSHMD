import json
from pathlib import Path

import pytest

from gqteashmd.cli import main

pytestmark = pytest.mark.pyscf

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


def test_single_point_uses_its_own_geometry_as_the_reference(tmp_path):
    out = tmp_path / "sp.json"
    path = _input(tmp_path, extra=CAS.replace("6-31g", "sto-3g"))
    assert main(["single-point", str(path), "--json", str(out)]) == 0
    meta = json.loads(out.read_text())["meta"]
    assert meta["casscf"] == "reference orbitals"
    assert meta["casscf_reference"]["n_guesses"] == 4


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


def test_inconsistent_charge_is_exit_2(tmp_path, capsys):
    path = _input(tmp_path)
    path.write_text(
        path.read_text().replace(
            'geometry = "ethylene.xyz"', 'geometry = "ethylene.xyz"\ncharge = 1'
        )
    )
    assert main(["single-point", str(path)]) == 2
    assert "Check [molecule]" in capsys.readouterr().err


def test_json_into_a_missing_directory_is_exit_2_before_computing(tmp_path, capsys):
    out = tmp_path / "missing" / "sp.json"
    assert main(["single-point", str(_input(tmp_path)), "--json", str(out)]) == 2
    err = capsys.readouterr()
    assert "--json" in err.err and "missing" in err.err
    assert "Single point" not in err.out  # no calculation was started


def test_unusable_log_path_is_exit_2_before_computing(tmp_path, capsys):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")
    log = blocker / "pyscf.log"
    assert main(["single-point", str(_input(tmp_path)), "--log", str(log)]) == 2
    err = capsys.readouterr()
    assert "--log" in err.err
    assert "Single point" not in err.out


WATER = "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n"
TDA_HF = (
    '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 2\n'
    "triplets = {triplets}\n"
)
HEADER = f"{'state':<6}{'energy (Eh)':>18}{'dE (eV)':>10}{'f':>9}"


def test_single_point_with_triplets_prints_an_s2_column(tmp_path, capsys):
    out = tmp_path / "sp.json"
    path = _input(tmp_path, extra=TDA_HF.format(triplets=1), xyz=WATER)
    assert main(["single-point", str(path), "--json", str(out)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == HEADER + f"{'S^2':>8}"
    row = next(line for line in lines if line.startswith("T1"))
    assert row.split()[3] == "0.0000" and row.split()[4] == "-"
    data = json.loads(out.read_text())
    assert [r["label"] for r in data["states"]] == ["S0", "S1", "T1"]
    assert data["states"][2]["oscillator_strength"] == 0.0
    assert data["meta"]["td_triplet"] == "default"


def test_single_point_without_triplets_keeps_its_table(tmp_path, capsys):
    out = tmp_path / "sp.json"
    path = _input(tmp_path, extra=TDA_HF.format(triplets=0), xyz=WATER)
    assert main(["single-point", str(path), "--json", str(out)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == HEADER
    assert [line.split()[0] for line in lines[2:]] == ["S0", "S1", "Results"]
    assert "td_triplet" not in json.loads(out.read_text())["meta"]
