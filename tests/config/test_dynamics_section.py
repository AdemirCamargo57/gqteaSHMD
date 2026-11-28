"""The [dynamics] section of the workflow input (P7b)."""

import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import ConfigError

TDA = '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 3\n'
CAS = (
    '[electronic]\nmethod = "casscf"\nbasis = "sto-3g"\nsinglets = 2\n'
    "[electronic.casscf]\nncas = 2\nnelecas = 2\n"
)


def _load(tmp_path, write_input, extra):
    return load_config(write_input(tmp_path, extra=extra)).config


def test_absent_table_means_no_dynamics(tmp_path, write_input):
    assert _load(tmp_path, write_input, TDA).dynamics is None


def test_defaults(tmp_path, write_input):
    d = _load(tmp_path, write_input, TDA + "[dynamics]\ntmax_fs = 10.0\n").dynamics
    assert d is not None
    assert d.hop_energy == "current"
    assert (d.tmax_fs, d.dt_fs, d.nsubsteps, d.coupling, d.hopping) == (
        10.0,
        0.5,
        25,
        "overlap",
        "sharc",
    )
    assert (d.ekin_correction, d.frustrated, d.decoherence, d.decoherence_param_Eh) == (
        "parallel_vel",
        "keep",
        "edc",
        0.1,
    )
    assert (d.edc_convention, d.substep_interpolation, d.energy_drift_limit_eV) == (
        "sharc",
        "midpoint",
        0.5,
    )
    assert (d.stop_after_s0_fs, d.overlap_threshold, d.max_step_halvings) == (None, 0.5, 3)
    assert (d.output_every, d.max_failed_fraction) == (1, 0.1)
    assert d.energy_jump_limit_eV == 0.1


def test_tmax_is_required(tmp_path, write_input):
    with pytest.raises(ConfigError, match="tmax_fs"):
        _load(tmp_path, write_input, TDA + "[dynamics]\ndt_fs = 0.5\n")


@pytest.mark.parametrize(
    ("lines", "needle"),
    [
        ("tmax_fs = 0.5\ndt_fs = 0.5\n", "dt_fs"),
        ("tmax_fs = 10.2\ndt_fs = 0.5\n", "integer multiple"),
        ('tmax_fs = 10.0\nhopping = "fssh"\n', "fssh"),
        ('tmax_fs = 10.0\nekin_correction = "parallel_nac"\n', "parallel_nac"),
        ("tmax_fs = 10.0\nmax_step_halvings = 7\n", "max_step_halvings"),
        ("tmax_fs = 10.0\noverlap_threshold = 1.0\n", "overlap_threshold"),
        ("tmax_fs = 10.0\nstop_after_s0_fs = 0.0\n", "stop_after_s0_fs"),
        ("tmax_fs = 10.0\nenergy_jump_limit_eV = 0.0\n", "energy_jump_limit_eV"),
        ("tmax_fs = 10.0\nenergy_jump_limit_eV = inf\n", "energy_jump_limit_eV"),
        ("tmax_fs = 10.0\neselect_Eh = 0.1\n", "eselect_Eh"),
        ('tmax_fs = 10.0\nhop_energy = "sharc"\n', "hop_energy"),
        ('tmax_fs = 10.0\ncoupling = "nacdr"\n', "nacdr"),
    ],
)
def test_invalid_values_name_the_key(tmp_path, write_input, lines, needle):
    with pytest.raises(ConfigError, match=needle):
        _load(tmp_path, write_input, TDA + "[dynamics]\n" + lines)


def test_nacdr_and_parallel_nac_are_allowed_with_casscf(tmp_path, write_input):
    extra = (
        CAS + '[dynamics]\ntmax_fs = 10.0\ncoupling = "nacdr"\n'
        'ekin_correction = "parallel_nac"\nhopping = "fssh"\n'
    )
    d = _load(tmp_path, write_input, extra).dynamics
    assert d is not None and d.coupling == "nacdr"


# --- triplets and spin-orbit coupling (sub-project 3, spec 2.2-2.3) --------------------------

CAS_TRIPLETS = (
    '[electronic]\nmethod = "casscf"\nbasis = "sto-3g"\nsinglets = 2\ntriplets = 2\n'
    "[electronic.casscf]\nncas = 3\nnelecas = 4\n"
)
TDA_TRIPLETS = TDA + "triplets = 2\n"
VALID_SOC = 'coupling = "overlap"\ngradient_transformation = "none"\nhopping = "sharc"\n'
VALID_SOC += 'ekin_correction = "parallel_vel"\n'


def test_new_keys_default_to_ngt_and_no_eselect(tmp_path, write_input):
    d = _load(tmp_path, write_input, TDA + "[dynamics]\ntmax_fs = 10.0\n").dynamics
    assert d is not None
    assert d.gradient_transformation == "ngt" and d.eselect_eV is None


def test_new_keys_accept_their_values(tmp_path, write_input):
    extra = TDA + '[dynamics]\ntmax_fs = 10.0\ngradient_transformation = "none"\neselect_eV = 0.5\n'
    d = _load(tmp_path, write_input, extra).dynamics
    assert d is not None
    assert d.gradient_transformation == "none" and d.eselect_eV == 0.5


@pytest.mark.parametrize(
    ("lines", "needle"),
    [
        ("eselect_eV = 0.0\n", "eselect_eV"),
        ("eselect_eV = -0.5\n", "eselect_eV"),
        ("eselect_eV = inf\n", "eselect_eV"),
        ("eselect_eV = nan\n", "eselect_eV"),
        ('gradient_transformation = "full"\n', "gradient_transformation"),
    ],
)
def test_invalid_new_keys_name_the_key(tmp_path, write_input, lines, needle):
    with pytest.raises(ConfigError, match=needle):
        _load(tmp_path, write_input, TDA + "[dynamics]\ntmax_fs = 10.0\n" + lines)


@pytest.mark.parametrize("electronic", [CAS_TRIPLETS, TDA_TRIPLETS], ids=["casscf", "tda"])
def test_the_valid_spin_orbit_combination_is_accepted(tmp_path, write_input, electronic):
    extra = electronic + "[dynamics]\ntmax_fs = 10.0\n" + VALID_SOC
    d = _load(tmp_path, write_input, extra).dynamics
    assert d is not None and d.gradient_transformation == "none"


def test_eselect_is_accepted_with_triplets(tmp_path, write_input):
    extra = TDA_TRIPLETS + "[dynamics]\ntmax_fs = 10.0\n" + VALID_SOC + "eselect_eV = 0.5\n"
    d = _load(tmp_path, write_input, extra).dynamics
    assert d is not None and d.eselect_eV == 0.5


def test_every_triplet_violation_is_reported_at_once(tmp_path, write_input):
    extra = (
        CAS_TRIPLETS + '[dynamics]\ntmax_fs = 10.0\ncoupling = "nacdr"\n'
        'gradient_transformation = "ngt"\nhopping = "fssh"\nekin_correction = "parallel_nac"\n'
    )
    with pytest.raises(ConfigError) as info:
        _load(tmp_path, write_input, extra)
    problems = [p for p in str(info.value).split("; ") if "triplets" in p]
    for key in ("coupling", "gradient_transformation", "hopping", "ekin_correction"):
        assert any(f"dynamics.{key}" in p for p in problems), (key, str(info.value))
    assert len(problems) == 4


@pytest.mark.parametrize("electronic", [CAS_TRIPLETS, TDA_TRIPLETS], ids=["casscf", "tda"])
def test_the_default_gradient_transformation_is_refused_with_triplets(
    tmp_path, write_input, electronic
):
    with pytest.raises(ConfigError) as info:
        _load(tmp_path, write_input, electronic + "[dynamics]\ntmax_fs = 10.0\n")
    text = str(info.value)
    assert "dynamics.gradient_transformation" in text and "triplets" in text


def test_nacdr_is_refused_with_triplets(tmp_path, write_input):
    extra = (
        CAS_TRIPLETS + '[dynamics]\ntmax_fs = 10.0\ncoupling = "nacdr"\n'
        'gradient_transformation = "none"\n'
    )
    with pytest.raises(ConfigError) as info:
        _load(tmp_path, write_input, extra)
    problems = str(info.value).split("; ")
    assert [p for p in problems if "triplets" in p and "dynamics.coupling" in p]
    assert not [p for p in problems if "gradient_transformation" in p]


def test_without_triplets_the_spin_orbit_rules_do_not_apply(tmp_path, write_input):
    extra = (
        CAS + '[dynamics]\ntmax_fs = 10.0\ncoupling = "nacdr"\n'
        'ekin_correction = "parallel_nac"\nhopping = "fssh"\n'
    )
    d = _load(tmp_path, write_input, extra).dynamics
    assert d is not None and d.gradient_transformation == "ngt"
