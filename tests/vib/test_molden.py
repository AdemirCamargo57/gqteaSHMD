import numpy as np

from gqteashmd.vib.molden import format_molden


def test_molden_sections_and_values():
    coords = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    modes = np.array([[0.0], [0.0], [-0.70710678], [0.0], [0.0], [0.70710678]])
    text = format_molden(("H", "H"), coords, np.array([-12.5]), modes)
    lines = text.splitlines()
    order = [line for line in lines if line.startswith("[")]
    assert order == ["[Molden Format]", "[Atoms] AU", "[FREQ]", "[FR-COORD]", "[FR-NORM-COORD]"]
    freq_at = lines.index("[FREQ]")
    assert float(lines[freq_at + 1]) == -12.5
    fr = lines.index("[FR-COORD]")
    assert lines[fr + 1].split()[0] == "H" and float(lines[fr + 2].split()[3]) == 0.7
    vib = lines.index("[FR-NORM-COORD]")
    assert lines[vib + 1].split() == ["vibration", "1"]
    assert float(lines[vib + 3].split()[2]) == 0.70710678
    assert lines[1 + 1].split()[:3] == ["H", "1", "1"]
