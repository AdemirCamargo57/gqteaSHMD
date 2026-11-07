from pathlib import Path

import numpy as np
import pytest

from gqteashmd.chem.xyz import parse_xyz
from gqteashmd.config.schema import ElectronicSection
from gqteashmd.core.units import ANGSTROM_TO_BOHR

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def ethylene_symbols() -> tuple[str, ...]:
    return parse_xyz((DATA / "ethylene.xyz").read_text()).symbols


@pytest.fixture
def ethylene_bohr() -> np.ndarray:
    geometry = parse_xyz((DATA / "ethylene.xyz").read_text())
    return np.array(geometry.coords_angstrom) * ANGSTROM_TO_BOHR


@pytest.fixture
def make_settings():
    def _make(**fields) -> ElectronicSection:
        base = {
            "method": "casscf",
            "basis": "6-31g*",
            "singlets": 2,
            "casscf": {"ncas": 2, "nelecas": 2},
        }
        base.update(fields)
        return ElectronicSection.model_validate(base)

    return _make


@pytest.fixture
def reference_dir() -> Path:
    return DATA / "reference"
