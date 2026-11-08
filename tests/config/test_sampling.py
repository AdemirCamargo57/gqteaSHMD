import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import ConfigError


def test_defaults_and_values(tmp_path, write_input):
    cfg = load_config(write_input(tmp_path)).config
    assert (
        cfg.sampling.n_samples,
        cfg.sampling.temperature_K,
        cfg.sampling.low_frequency_cutoff_cm1,
    ) == (200, 0.0, 10.0)
    cfg = load_config(
        write_input(
            tmp_path,
            extra="[sampling]\nn_samples = 50\n"
            "temperature_K = 300.0\nlow_frequency_cutoff_cm1 = 50.0\n",
        )
    ).config
    assert (cfg.sampling.n_samples, cfg.sampling.temperature_K) == (50, 300.0)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("[sampling]\nn_samples = 0\n", "n_samples"),
        ("[sampling]\ntemperature_K = -1.0\n", "temperature_K"),
        ("[sampling]\nlow_frequency_cutoff_cm1 = -5.0\n", "low_frequency_cutoff_cm1"),
        ("[sampling]\nlow_frequency_cutoff_cm1 = 0.0\n", "low_frequency_cutoff_cm1"),
        ("[sampling]\nseed = 3\n", "seed"),
    ],
)
def test_invalid(tmp_path, write_input, extra, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write_input(tmp_path, extra=extra))
