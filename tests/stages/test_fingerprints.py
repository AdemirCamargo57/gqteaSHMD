import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.config.schema import ElectronicSection
from gqteashmd.electronic.factory import CASSCF_START_RULE, electronic_fingerprint
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.stages.base import StageContext
from gqteashmd.stages.excitations import ExcitationsStage
from gqteashmd.stages.frequencies import FrequenciesStage
from gqteashmd.stages.optimize import OptimizeStage

TDA = '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 2\n'
CAS = (
    '[electronic]\nmethod = "casscf"\nbasis = "sto-3g"\nsinglets = 2\n'
    "[electronic.casscf]\nncas = 2\nnelecas = 2\n"
)


def _fingerprints(tmp_path, write_input, name, electronic, extra=""):
    directory = tmp_path / name
    directory.mkdir()
    cfg = load_config(write_input(directory, extra=electronic + extra)).config
    ctx = StageContext(cfg, ProjectLayout(cfg.project.directory), directory, {})
    return OptimizeStage().fingerprint_inputs(ctx), FrequenciesStage().fingerprint_inputs(ctx)


@pytest.mark.parametrize(
    "key, value", [("singlets", "3"), ("td_conv_tol", "1e-6"), ("overlap_norm_kept", "0.99")]
)
def test_response_only_keys_do_not_change_tda_fingerprints(tmp_path, write_input, key, value):
    base = _fingerprints(tmp_path, write_input, "a", TDA)
    if key == "singlets":
        other = TDA.replace("singlets = 2", f"singlets = {value}")
    else:
        other = TDA + f"{key} = {value}\n"
    assert _fingerprints(tmp_path, write_input, "b", other) == base


@pytest.mark.parametrize(
    "old, new",
    [('basis = "sto-3g"', 'basis = "6-31g"'), ('functional = "hf"', 'functional = "pbe"')],
)
def test_ground_state_keys_change_the_fingerprints(tmp_path, write_input, old, new):
    base = _fingerprints(tmp_path, write_input, "a", TDA)
    changed = _fingerprints(tmp_path, write_input, "b", TDA.replace(old, new))
    assert changed[0] != base[0] and changed[1] != base[1]


def test_imaginary_cutoff_does_not_change_the_frequencies_fingerprint(tmp_path, write_input):
    base = _fingerprints(tmp_path, write_input, "a", TDA)
    other = _fingerprints(
        tmp_path, write_input, "b", TDA, "[frequencies]\nimaginary_cutoff_cm1 = 50.0\n"
    )
    assert other == base


def test_fd_step_changes_the_frequencies_fingerprint(tmp_path, write_input):
    base = _fingerprints(tmp_path, write_input, "a", TDA)
    other = _fingerprints(tmp_path, write_input, "b", TDA, "[frequencies]\nfd_step_bohr = 0.01\n")
    assert other[1] != base[1] and other[0] == base[0]


def test_optimize_options_change_only_the_optimize_fingerprint(tmp_path, write_input):
    base = _fingerprints(tmp_path, write_input, "a", TDA)
    other = _fingerprints(tmp_path, write_input, "b", TDA, "[optimize]\nmax_steps = 7\n")
    assert other[0] != base[0] and other[1] == base[1]


def test_casscf_singlets_change_the_fingerprints(tmp_path, write_input):
    base = _fingerprints(tmp_path, write_input, "a", CAS)
    other = _fingerprints(tmp_path, write_input, "b", CAS.replace("singlets = 2", "singlets = 1"))
    assert other[0] != base[0] and other[1] != base[1]


def test_casscf_fingerprints_carry_the_start_rule(tmp_path, write_input):
    optimize, frequencies = _fingerprints(tmp_path, write_input, "a", CAS)
    assert optimize["electronic"]["start_rule"] == CASSCF_START_RULE == "reference-1"
    assert frequencies["electronic"]["start_rule"] == CASSCF_START_RULE


def test_tda_fingerprints_have_no_start_rule(tmp_path, write_input):
    optimize, frequencies = _fingerprints(tmp_path, write_input, "a", TDA)
    assert "start_rule" not in optimize["electronic"]
    assert "start_rule" not in frequencies["electronic"]


def test_electronic_fingerprint_is_the_section_plus_the_marker():
    cas = ElectronicSection.model_validate(
        {"method": "casscf", "basis": "sto-3g", "singlets": 2, "casscf": {"ncas": 2, "nelecas": 2}}
    )
    tda = ElectronicSection.model_validate(
        {"method": "tda", "basis": "sto-3g", "functional": "hf", "singlets": 2}
    )
    assert electronic_fingerprint(cas) == {
        **cas.model_dump(mode="json"),
        "start_rule": CASSCF_START_RULE,
    }
    assert electronic_fingerprint(tda) == tda.model_dump(mode="json")


def test_excitations_fingerprint_carries_the_start_rule_for_casscf_only(tmp_path, write_input):
    for name, electronic, expected in (("a", CAS, True), ("b", TDA, False)):
        directory = tmp_path / name
        directory.mkdir()
        cfg = load_config(write_input(directory, extra=electronic)).config
        ctx = StageContext(cfg, ProjectLayout(cfg.project.directory), directory, {})
        fp = ExcitationsStage().fingerprint_inputs(ctx)["electronic"]
        assert ("start_rule" in fp) is expected
        assert "overlap_norm_kept" not in fp
