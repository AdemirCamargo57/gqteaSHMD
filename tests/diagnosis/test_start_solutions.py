from tests.diagnosis.conftest import load_tool

tool = load_tool("diagnosis", "start_solutions")


def test_guess_pairs_for_cas22():
    # ethylene: 16 electrons, 8 doubly occupied RHF orbitals (0-based HOMO 7, LUMO 8)
    assert tool.guess_pairs(n_occupied=8, ncas=2, nelecas=2) == [(6, 8), (6, 9), (7, 8), (7, 9)]


def test_guess_pairs_for_cas43():
    # 2 of the top 3 occupied (5, 6, 7) times 1 of the lowest 2 virtuals (8, 9)
    assert tool.guess_pairs(n_occupied=8, ncas=3, nelecas=4) == [
        (5, 6, 8),
        (5, 6, 9),
        (5, 7, 8),
        (5, 7, 9),
        (6, 7, 8),
        (6, 7, 9),
    ]


def test_distinct_solutions_cluster_by_state_energies():
    entries = [
        {"guess": (7, 8), "energies_Eh": [-78.0, -77.7, -77.6]},
        {"guess": (6, 8), "energies_Eh": [-78.0 + 5e-7, -77.7, -77.6]},
        {"guess": (7, 9), "energies_Eh": [-77.9, -77.8, -77.5]},
        {"guess": (6, 9), "energies_Eh": None},
    ]
    sols = tool.distinct_solutions(entries)
    # solutions ordered by SA-average energy; guesses kept in input order
    assert [s["guesses"] for s in sols] == [[(7, 8), (6, 8)], [(7, 9)]]
    assert sols[0]["sa_energy_Eh"] < sols[1]["sa_energy_Eh"]


def test_match_solution():
    sols = [{"energies_Eh": [-78.0, -77.7, -77.6]}, {"energies_Eh": [-77.9, -77.8, -77.5]}]
    assert tool.match_solution([-77.90005, -77.8, -77.5], sols) == 1
    assert tool.match_solution([-76.0, -76.0, -76.0], sols) is None
