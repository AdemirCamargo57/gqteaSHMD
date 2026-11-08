import pytest

from gqteashmd.config.loader import load_config
from gqteashmd.errors import ConfigError

TDA = '[electronic]\nmethod = "tda"\nbasis = "sto-3g"\nfunctional = "hf"\nsinglets = 3\n'


def test_defaults(tmp_path, write_input):
    cfg = load_config(write_input(tmp_path, extra=TDA)).config
    assert cfg.excitations.samples_per_task == 1 and cfg.excitations.include_reference
    assert cfg.excitations.max_failed_fraction == 0.05
    assert (cfg.selection.mode, cfg.selection.states, cfg.selection.energy_window_eV) == (
        "bright",
        None,
        None,
    )
    assert cfg.selection.n_trajectories == 100 and not cfg.selection.allow_fewer


def test_values(tmp_path, write_input):
    extra = TDA + (
        '[selection]\nmode = "bright"\nstates = ["S1", "s2"]\n'
        "energy_window_eV = [4.5, 12.0]\nn_trajectories = 7\n"
    )
    cfg = load_config(write_input(tmp_path, extra=extra)).config
    assert cfg.selection.states == ["S1", "S2"]  # normalized
    assert cfg.selection.energy_window_eV == (4.5, 12.0)


@pytest.mark.parametrize(
    "extra, message",
    [
        ('[selection]\nstates = ["S0"]\n', "S0"),
        ('[selection]\nstates = ["T1"]\n', "T1"),
        ('[selection]\nstates = ["S1", "S1"]\n', "duplicate"),
        ('[selection]\nstates = ["X9"]\n', "X9"),
        ('[selection]\nstates = ["S5"]\n', "S5 does not exist"),
        ('[selection]\nmode = "fixed"\nstates = ["S1", "S2"]\n', "exactly one state"),
        ('[selection]\nmode = "fixed"\n', "exactly one state"),
        ("[selection]\nenergy_window_eV = [6.0, 4.0]\n", "energy_window_eV"),
        ("[selection]\nenergy_window_eV = [-1.0, 4.0]\n", "energy_window_eV"),
        ("[selection]\nn_trajectories = 0\n", "n_trajectories"),
        ("[excitations]\nsamples_per_task = 0\n", "samples_per_task"),
        ("[excitations]\nmax_failed_fraction = 1.0\n", "max_failed_fraction"),
        ("[excitations]\nbroadening_fwhm_eV = 0.0\n", "broadening_fwhm_eV"),
        ("[excitations]\nspectrum_step_eV = inf\n", "spectrum_step_eV"),
    ],
)
def test_invalid(tmp_path, write_input, extra, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write_input(tmp_path, extra=TDA + extra))
