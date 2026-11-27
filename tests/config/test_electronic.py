import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.config.schema import ElectronicSection, n_singlet_csfs, n_triplet_csfs
from gqteashmd.errors import ConfigError

CAS = (
    '[electronic]\nmethod = "casscf"\nbasis = "6-31g*"\nsinglets = 2\n'
    "[electronic.casscf]\nncas = 2\nnelecas = 2\n"
)
TDA = '[electronic]\nmethod = "tda"\nbasis = "def2-svp"\nfunctional = "b3lyp"\nsinglets = 3\n'


def _load(tmp_path, write_input, extra, molecule=""):
    return load_config(write_input(tmp_path, extra=extra, molecule=molecule)).config


def test_section_is_optional(tmp_path, write_input):
    assert _load(tmp_path, write_input, "").electronic is None


def test_casscf_section_with_defaults(tmp_path, write_input):
    el = _load(tmp_path, write_input, CAS).electronic
    assert (el.method, el.basis, el.singlets, el.triplets) == ("casscf", "6-31g*", 2, 0)
    assert (el.casscf.ncas, el.casscf.nelecas) == (2, 2)
    assert el.casscf.conv_tol == 1e-10 and el.casscf.conv_tol_grad == 1e-6
    assert el.scf_conv_tol == 1e-10 and el.scf_max_cycle == 100


def test_tda_section(tmp_path, write_input):
    el = _load(tmp_path, write_input, TDA).electronic
    assert (el.method, el.functional, el.grid_level, el.td_conv_tol) == ("tda", "b3lyp", 3, 1e-8)


@pytest.mark.parametrize(
    "extra, message",
    [
        (
            '[electronic]\nmethod = "casscf"\nbasis = "6-31g*"\nsinglets = 2\n',
            r"\[electronic\.casscf\]",
        ),
        (
            CAS.replace("singlets = 2\n", 'singlets = 2\nfunctional = "b3lyp"\n'),
            "functional is not used",
        ),
        (
            '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nsinglets = 2\n',
            "electronic.functional is required",
        ),
        (TDA + "[electronic.casscf]\nncas = 2\nnelecas = 2\n", "only used by method = 'casscf'"),
        (CAS.replace("nelecas = 2", "nelecas = 3"), "even"),
        (CAS.replace("nelecas = 2", "nelecas = 6"), "at most 2"),
        (TDA.replace('"tda"', '"cc2"'), "method"),
        (TDA.replace("singlets = 3", "singlets = 0"), "singlets"),
    ],
)
def test_invalid_sections(tmp_path, write_input, extra, message):
    with pytest.raises(ConfigError, match=message):
        _load(tmp_path, write_input, extra)


def test_open_shell_molecule_is_refused(tmp_path, write_input):
    with pytest.raises(ConfigError, match="closed-shell singlet"):
        _load(tmp_path, write_input, CAS, molecule="multiplicity = 3")


@pytest.mark.parametrize(
    "ncas, nelecas, expected", [(2, 2, 3), (4, 4, 20), (6, 6, 175), (3, 2, 6), (1, 2, 1)]
)
def test_n_singlet_csfs(ncas, nelecas, expected):
    assert n_singlet_csfs(ncas, nelecas) == expected


def test_singlets_cannot_exceed_the_cas_singlet_space(tmp_path, write_input):
    assert _load(tmp_path, write_input, CAS.replace("singlets = 2", "singlets = 3")).electronic
    with pytest.raises(
        ConfigError,
        match=r"electronic\.singlets = 4 exceeds the 3 singlet states of CAS\(2,2\) "
        r"\(nelecas = 2, ncas = 2\)",
    ):
        _load(tmp_path, write_input, CAS.replace("singlets = 2", "singlets = 4"))


def test_overlap_norm_kept(tmp_path, write_input):
    assert _load(tmp_path, write_input, CAS).electronic.overlap_norm_kept == 0.9999
    el = _load(
        tmp_path, write_input, CAS.replace("singlets = 2", "singlets = 2\noverlap_norm_kept = 1.0")
    )
    assert el.electronic.overlap_norm_kept == 1.0
    for bad in ("0.9", "1.5"):
        with pytest.raises(ConfigError, match="overlap_norm_kept"):
            _load(
                tmp_path,
                write_input,
                CAS.replace("singlets = 2", f"singlets = 2\noverlap_norm_kept = {bad}"),
            )


def test_triplet_csf_counts():
    assert n_triplet_csfs(2, 2) == 1
    assert n_triplet_csfs(3, 2) == 3
    assert n_triplet_csfs(4, 4) == 15
    assert n_singlet_csfs(4, 4) == 20  # unchanged


def test_triplets_are_accepted():
    for method, extra in (
        ("tda", {"functional": "hf"}),
        ("tddft", {"functional": "b3lyp"}),
        ("casscf", {"casscf": {"ncas": 2, "nelecas": 2}}),
    ):
        s = ElectronicSection.model_validate(
            {"method": method, "basis": "sto-3g", "singlets": 2, "triplets": 1, **extra}
        )
        assert s.triplets == 1


def test_casscf_triplets_beyond_the_active_space_are_refused():
    with pytest.raises(ValueError, match="electronic.triplets = 2 exceeds the 1 triplet states"):
        ElectronicSection.model_validate(
            {
                "method": "casscf",
                "basis": "sto-3g",
                "singlets": 2,
                "triplets": 2,
                "casscf": {"ncas": 2, "nelecas": 2},
            }
        )


def _tda(**fields) -> ElectronicSection:
    return ElectronicSection.model_validate(
        {"method": "tda", "basis": "sto-3g", "functional": "hf", "singlets": 2, **fields}
    )


def test_soc_operator_defaults_to_somf_and_accepts_effective_charges():
    assert _tda().soc_operator == "somf"
    assert _tda(soc_operator="effective_charge").soc_operator == "effective_charge"
    with pytest.raises(ValueError, match="soc_operator"):
        _tda(soc_operator="amfi")


def test_the_default_soc_operator_is_left_out_of_the_dump():
    """So that the config snapshot of a project written before the key is unchanged, and the
    dump still round-trips the value."""
    assert "soc_operator" not in _tda().model_dump(mode="json")
    assert "soc_operator" not in _tda(soc_operator="somf").model_dump_json()
    dumped = _tda(soc_operator="effective_charge").model_dump(mode="json")
    assert dumped["soc_operator"] == "effective_charge"
    assert ElectronicSection.model_validate(dumped).soc_operator == "effective_charge"
    assert ElectronicSection.model_validate_json(_tda().model_dump_json()).soc_operator == "somf"


def test_an_unknown_soc_operator_in_the_input_is_a_config_error(tmp_path, write_input):
    with pytest.raises(ConfigError, match="soc_operator"):
        _load(tmp_path, write_input, TDA + 'soc_operator = "amfi"\n')
    cfg = _load(tmp_path, write_input, TDA + 'soc_operator = "effective_charge"\n')
    assert cfg.electronic.soc_operator == "effective_charge"
