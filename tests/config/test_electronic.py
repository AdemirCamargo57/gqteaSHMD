import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.config.schema import n_singlet_csfs
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
        (TDA.replace("singlets = 3", "singlets = 3\ntriplets = 1"), "triplets"),
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
