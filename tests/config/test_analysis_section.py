"""[analysis] input section (P8 spec section 10)."""

import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.config.schema import ENERGY_JUMP_DEFAULT_EV
from gqteashmd.errors import ConfigError

ELECTRONIC = '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 3\n'


def _load(tmp_path, write_input, extra):
    return load_config(write_input(tmp_path, extra=ELECTRONIC + extra)).config


def test_defaults_without_a_table(tmp_path, write_input):
    a = _load(tmp_path, write_input, "").analysis
    assert a.enabled is True and a.fit_states is None and a.fit_population == "classical"
    assert a.n_bootstrap == 1000 and a.energy_jump_eV == ENERGY_JUMP_DEFAULT_EV
    assert a.exclude_energy_jumps is False and a.warn_excluded_fraction == 0.2


def test_fit_states_are_upper_cased_excited_singlets(tmp_path, write_input):
    a = _load(tmp_path, write_input, '[analysis]\nfit_states = ["s1", "S2"]\n').analysis
    assert a.fit_states == ["S1", "S2"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('fit_states = ["S0"]', "S0"),
        ('fit_states = ["T1"]', "triplet"),
        ('fit_states = ["S1", "S1"]', "duplicate"),
        ("fit_states = []", "empty"),
        ('fit_states = ["S3"]', "does not exist"),
        ("n_bootstrap = 99", "n_bootstrap"),
        ("energy_jump_eV = 0.0", "energy_jump_eV"),
        ("energy_jump_eV = inf", "energy_jump_eV"),
        ("warn_excluded_fraction = 1.5", "warn_excluded_fraction"),
        ('fit_population = "both"', "fit_population"),
        ("unknown_key = 1", "unknown_key"),
    ],
)
def test_invalid_values_are_config_errors(tmp_path, write_input, text, message):
    with pytest.raises(ConfigError, match=message):
        _load(tmp_path, write_input, f"[analysis]\n{text}\n")


def test_all_violations_of_the_cross_field_rule_are_reported(tmp_path, write_input):
    with pytest.raises(ConfigError) as info:
        _load(tmp_path, write_input, '[analysis]\nfit_states = ["S3", "S4"]\n')
    assert "S3" in str(info.value) and "S4" in str(info.value)


def test_init_template_has_a_commented_analysis_block(tmp_path):
    from gqteashmd.cli import TEMPLATE

    assert "# [analysis]" in TEMPLATE and "# fit_states" in TEMPLATE
