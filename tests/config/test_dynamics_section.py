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
        ("tmax_fs = 10.0\neselect_Eh = 0.1\n", "eselect_Eh"),
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
