import json
from pathlib import Path

import numpy as np
import pytest

from gqteashmd import cli
from gqteashmd.cli import main
from gqteashmd.core.units import HARTREE_IN_CM1

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


SOC_COLUMNS = f"{'pair':<8}{'-1':>18}{'0':>18}{'+1':>18}{'rss':>10}"


def _soc_rows(lines):
    start = next(i for i, line in enumerate(lines) if line.startswith("Spin-orbit coupling"))
    header = start + 2
    assert lines[header] == SOC_COLUMNS
    rows = []
    for line in lines[header + 1 :]:
        if not line or line.startswith("Results"):
            break
        rows.append(line)
    return lines[start], rows


@pytest.mark.parametrize("operator", ["somf", "effective_charge"])
def test_single_point_with_triplets_prints_the_spin_orbit_table(tmp_path, capsys, operator):
    out = tmp_path / "sp.json"
    extra = TDA_HF.format(triplets=1).replace("singlets = 2", "singlets = 3")
    if operator != "somf":
        extra += f'soc_operator = "{operator}"\n'
    path = _input(tmp_path, extra=extra.replace("triplets = 1", "triplets = 2"), xyz=WATER)
    assert main(["single-point", str(path), "--json", str(out)]) == 0
    lines = capsys.readouterr().out.splitlines()
    title, rows = _soc_rows(lines)
    assert title == f"Spin-orbit coupling (cm-1, {operator})"
    pairs = [row.split()[0] for row in rows]
    assert pairs == [
        "S0-T1",
        "S0-T2",
        "S1-T1",
        "S1-T2",
        "S2-T1",
        "S2-T2",
        "T1-T1",
        "T1-T2",
        "T2-T2",
    ]
    data = json.loads(out.read_text())
    assert data["soc_operator"] == operator
    labels = data["hamiltonian_labels"]
    assert labels == ["S0", "S1", "S2", "T1(-1)", "T2(-1)", "T1(0)", "T2(0)", "T1(+1)", "T2(+1)"]
    h = np.array(data["hamiltonian"]["real"]) + 1j * np.array(data["hamiltonian"]["imag"])
    assert h.shape == (9, 9)
    s0_t1 = h[0, [3, 5, 7]] * HARTREE_IN_CM1
    row = rows[0].split()
    assert row[1:4] == [cli._complex_cm1(v) for v in s0_t1]
    assert float(row[4]) == pytest.approx(np.sqrt(np.sum(np.abs(s0_t1) ** 2)), abs=0.006)
    assert float(row[4]) > 1.0  # water S0-T1 couples
    t1_t2 = h[[7, 7, 5], [6, 8, 8]] * HARTREE_IN_CM1  # (M, M') = (+1, 0), (+1, +1), (0, +1)
    assert rows[7].split()[1:4] == [cli._complex_cm1(v) for v in t1_t2]
    for same in (rows[6], rows[8]):  # T1-T1, T2-T2: the middle element is not the energy
        assert same.split()[2] == "0.00+0.00j"
        assert len(same.split()) == 5


def test_single_point_without_triplets_has_no_spin_orbit_output(tmp_path, capsys):
    out = tmp_path / "sp.json"
    path = _input(tmp_path, extra=TDA_HF.format(triplets=0), xyz=WATER)
    assert main(["single-point", str(path), "--json", str(out)]) == 0
    assert "Spin-orbit" not in capsys.readouterr().out
    data = json.loads(out.read_text())
    assert not {"hamiltonian", "hamiltonian_labels", "soc_operator"} & set(data)


def test_values_that_round_to_zero_print_without_a_sign():
    assert cli._complex_cm1(complex(-0.004, 0.001)) == "0.00+0.00j"
    assert cli._complex_cm1(complex(-1.234, -0.0049)) == "-1.23+0.00j"
    assert cli._complex_cm1(complex(0.0, -65.113)) == "0.00-65.11j"


def test_the_table_counts_each_distinct_element_once(capsys):
    """1 singlet + 2 triplets with every coupling set: an intra-triplet pair counts its three
    elements with M < M' once (not their conjugates as well), an S-T pair its 3, a T-T' pair
    its 9 elements <T, M|H|T', M'>."""
    n_s, n_t = 1, 2
    rng = np.random.default_rng(3)
    n = n_s + 3 * n_t
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h = (a + a.conj().T) / HARTREE_IN_CM1
    np.fill_diagonal(h, -1.0)

    def index(j, m):  # m = 0, 1, 2 for Ms = -1, 0, +1
        return n_s + m * n_t + j

    cli._print_soc_table(h, n_s, n_t, "somf")
    rss = {
        line.split()[0]: float(line.split()[-1])
        for line in capsys.readouterr().out.splitlines()[3:]
    }
    cm1 = h * HARTREE_IN_CM1
    expected_t1_t1 = np.sqrt(
        sum(abs(cm1[index(0, m), index(0, mb)]) ** 2 for m in range(3) for mb in range(m + 1, 3))
    )
    expected_t1_t2 = np.sqrt(
        sum(abs(cm1[index(0, m), index(1, mb)]) ** 2 for m in range(3) for mb in range(3))
    )
    expected_s0_t2 = np.sqrt(sum(abs(cm1[0, index(1, m)]) ** 2 for m in range(3)))
    assert rss["T1-T1"] == pytest.approx(expected_t1_t1, abs=0.006)
    assert rss["T1-T2"] == pytest.approx(expected_t1_t2, abs=0.006)
    assert rss["S0-T2"] == pytest.approx(expected_s0_t2, abs=0.006)
