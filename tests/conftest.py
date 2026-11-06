from pathlib import Path

import pytest

WATER_XYZ = """3
water
O    0.000000    0.000000    0.117300
H    0.000000    0.757200   -0.469200
H    0.000000   -0.757200   -0.469200
"""


@pytest.fixture
def water_xyz() -> str:
    return WATER_XYZ


@pytest.fixture
def write_input():
    """Write water.xyz and input.toml into `directory`; return the input path.

    Keyword arguments insert raw TOML lines into the given section;
    `extra` is appended at the end of the file (top level).
    """

    def _write(
        directory: Path,
        *,
        molecule: str = "",
        resources: str = "",
        execution: str = "",
        extra: str = "",
    ) -> Path:
        (directory / "water.xyz").write_text(WATER_XYZ)
        text = (
            '[project]\nname = "test"\ndirectory = "run"\n\n'
            f'[molecule]\ngeometry = "water.xyz"\n{molecule}\n\n'
            f"[resources]\n{resources}\n\n"
            f"[execution]\n{execution}\n\n"
            f"{extra}\n"
        )
        path = directory / "input.toml"
        path.write_text(text)
        return path

    return _write
