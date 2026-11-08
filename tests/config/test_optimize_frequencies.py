import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import ConfigError

CAS = (
    '[electronic]\nmethod = "casscf"\nbasis = "sto-3g"\nsinglets = 2\n'
    "[electronic.casscf]\nncas = 2\nnelecas = 2\n"
)


def _load(tmp_path, write_input, extra):
    return load_config(write_input(tmp_path, extra=extra)).config


def test_defaults(tmp_path, write_input):
    cfg = _load(tmp_path, write_input, "")
    assert (cfg.optimize.enabled, cfg.optimize.max_steps, cfg.optimize.convergence) == (
        True,
        100,
        "GAU",
    )
    assert (
        cfg.frequencies.method,
        cfg.frequencies.fd_step_bohr,
        cfg.frequencies.imaginary_cutoff_cm1,
    ) == ("auto", 0.005, 20.0)


def test_values(tmp_path, write_input):
    cfg = _load(
        tmp_path,
        write_input,
        '[optimize]\nenabled = false\nmax_steps = 50\nconvergence = "GAU_TIGHT"\n'
        '[frequencies]\nmethod = "finite_difference"\nfd_step_bohr = 0.01\n'
        "imaginary_cutoff_cm1 = 50.0\n",
    )
    assert not cfg.optimize.enabled and cfg.optimize.convergence == "GAU_TIGHT"
    assert cfg.frequencies.method == "finite_difference" and cfg.frequencies.fd_step_bohr == 0.01


@pytest.mark.parametrize(
    "extra, message",
    [
        ('[optimize]\nconvergence = "VERY_GOOD"\n', "convergence"),
        ("[optimize]\nmax_steps = 0\n", "max_steps"),
        ("[frequencies]\nfd_step_bohr = 0.5\n", "fd_step_bohr"),
        ('[frequencies]\nmethod = "numerical"\n', "method"),
        (CAS + '[frequencies]\nmethod = "analytic"\n', "no analytic SA-CASSCF Hessian"),
    ],
)
def test_invalid(tmp_path, write_input, extra, message):
    with pytest.raises(ConfigError, match=message):
        _load(tmp_path, write_input, extra)


def test_all_model_violations_are_reported_together(tmp_path, write_input):
    text = CAS + '[frequencies]\nmethod = "analytic"\n'
    with pytest.raises(ConfigError) as info:
        load_config(write_input(tmp_path, extra=text, molecule="multiplicity = 3"))
    assert "closed-shell" in str(info.value) and "analytic SA-CASSCF" in str(info.value)
