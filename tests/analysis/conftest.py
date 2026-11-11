import pytest


@pytest.fixture
def summary_example():
    return {
        "schema_version": 1,
        "n_selected": 3,
        "n_included": 2,
        "outcomes": {
            "finished": 2,
            "energy_drift": 1,
            "stopped_in_s0": 0,
            "overlap_failure": 0,
            "electronic_failure": 0,
            "task_failed": 0,
        },
        "included": ["traj_00000", "traj_00002"],
        "excluded": [
            {
                "task": "traj_00001",
                "reason": "energy_drift",
                "t_final_fs": 3.5,
                "energy_jump_eV": 0.9,
            }
        ],
        "cut": ["traj_00002"],
        "horizon_fs": 10.0,
        "dt_out_fs": 0.5,
        "fit": {
            "states": ["S1"],
            "population": "classical",
            "A": 1.0,
            "t0_fs": 5.0,
            "tau_fs": 40.0,
            "t0_interval_fs": (4.0, 6.0),
            "tau_interval_fs": (30.0, 55.0),
            "t0_std_fs": 0.5,
            "tau_std_fs": 6.0,
            "rms": 0.01,
            "flags": [],
            "bootstrap_failures": 0,
        },
        "hops": {
            "pairs": {"S1->S0": 2},
            "frustrated_from": {"S1": 1},
            "per_trajectory": {"min": 0, "median": 1.0, "max": 2, "zero": 1},
            "first_hop_fs": {"n": 2, "mean": 6.0, "median": 6.0},
            "first_s0_arrival_fs": {"n_arrived": 2, "n_censored": 0, "mean": 6.0, "median": 6.0},
            "totals_all": {"hops": 3, "frustrated": 1},
        },
        "energy": {
            "drift_eV": {"median": 0.01, "p95": 0.05, "max": 0.06},
            "jump_threshold_eV": 0.3,
            "flagged": [{"task": "traj_00001", "energy_jump_eV": 0.9}],
        },
        "consistency": {"S0": 0.02, "S1": 0.02},
        "table": [
            {
                "time_fs": 2.5 * i,
                "P": {"S0": 0.1 * i, "S1": 1 - 0.1 * i},
                "rho": {"S0": 0.1 * i, "S1": 1 - 0.1 * i},
            }
            for i in range(5)
        ],
        "settings": {
            "enabled": True,
            "fit_states": ["S1"],
            "fit_population": "classical",
            "n_bootstrap": 200,
            "energy_jump_eV": 0.3,
            "exclude_energy_jumps": True,
            "warn_excluded_fraction": 0.1,
            "project_seed": 7,
        },
        "warnings": ["1 of 3 trajectories excluded"],
        "stage_failed": False,
        "error": None,
    }
