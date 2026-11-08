"""Our harmonic Wigner sampler against SHARC 4.1 wigner.py (run as a black box) for water."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import ks_2samp

from gqteashmd.chem.masses import isotope_mass
from gqteashmd.config.schema import ElectronicSection
from gqteashmd.core.units import ANGSTROM_TO_BOHR, DALTON_IN_ME, HARTREE_IN_CM1
from gqteashmd.electronic.base import MoleculeSpec
from gqteashmd.electronic.pyscf_backend import PySCFBackend
from gqteashmd.vib.harmonic import harmonic_analysis
from gqteashmd.vib.molden import format_molden
from gqteashmd.vib.wigner import sample_wigner

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "sharc_compare"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pytestmark = [
    pytest.mark.sharc,
    pytest.mark.pyscf,
    pytest.mark.skipif(
        not (os.environ.get("SHARC_HOME") and os.environ.get("SHARC_PYTHON")),
        reason="needs SHARC_HOME and SHARC_PYTHON",
    ),
]

SYMBOLS = ("O", "H", "H")
GEOMETRY_ANGSTROM = np.array(  # RHF/STO-3G minimum
    [
        [0.0, 0.0, 0.1501683345],
        [0.0, 0.7580805047, -0.4856341673],
        [0.0, -0.7580805047, -0.4856341673],
    ]
)


def _internal(positions):
    """Both O-H distances (bohr) and the HOH angle (degrees) for each sample."""
    v1 = positions[:, 1] - positions[:, 0]
    v2 = positions[:, 2] - positions[:, 0]
    r1 = np.linalg.norm(v1, axis=1)
    r2 = np.linalg.norm(v2, axis=1)
    cos = (v1 * v2).sum(axis=1) / (r1 * r2)
    return {"r_OH1": r1, "r_OH2": r2, "angle": np.degrees(np.arccos(cos))}


def _angular_momentum(masses_me, positions, velocities):
    m = masses_me[None, :, None]
    r = positions - (m * positions).sum(axis=1, keepdims=True) / masses_me.sum()
    return np.linalg.norm((m * np.cross(r, velocities)).sum(axis=1), axis=1)


def _std_se(x):
    """Standard error of the sample standard deviation from the fourth moment."""
    var = x.var(ddof=1)
    m4 = np.mean((x - x.mean()) ** 4)
    return np.sqrt((m4 - var**2) / (4.0 * x.size * var))


def _compare(name, ours, theirs, report):
    """Return (mean z, KS p, std z): differences in standard errors and the KS p-value."""
    se = np.sqrt(ours.var(ddof=1) / ours.size + theirs.var(ddof=1) / theirs.size)
    p = ks_2samp(ours, theirs).pvalue
    s_ours, s_sharc = ours.std(ddof=1), theirs.std(ddof=1)
    se_std = np.hypot(_std_se(ours), _std_se(theirs))
    z_std = (s_ours - s_sharc) / se_std
    report.append(
        f"{name:10s} ours mean {ours.mean():.6f} std {s_ours:.6f} | "
        f"SHARC mean {theirs.mean():.6f} std {s_sharc:.6f} | "
        f"diff/SE {(ours.mean() - theirs.mean()) / se:+.2f} | KS p {p:.4f} | "
        f"std ratio {s_ours / s_sharc:.4f} std diff/SE {z_std:+.2f}"
    )
    return abs(ours.mean() - theirs.mean()) / se, p, abs(z_std)


SHARC_THERMAL_DIFFERENCE = (
    "SHARC 4.1 wigner.py -t is not the exact thermal Wigner function: at 3000 K (coth factor 2.09 "
    "for the bend, 1.32 and 1.28 for the stretches) its normal-coordinate variances are 1-8 % "
    "below c/(2 omega) and slightly heavy-tailed (kurtosis 3.04-3.12; measured by projecting "
    "10000 of its samples onto our modes, seeds 11 and 12), while at T = 0 they match exactly. "
    "Measured here: HOH angle KS p 7e-4, std ratio 1.042 (+2.5 SE). Ours follows the exact "
    "Gaussian (unit tests; the stage's Gamma check). At 1000 K (coth 1.004-1.09) the difference "
    "is below the resolution of 2000 samples."
)


# 3000 K is the discriminating thermal case for water: there the comparison fails because of the
# documented difference above (deterministic seeds, so the expected failure is strict).
@pytest.mark.parametrize(
    ("temperature_K", "sharc_args"),
    [
        (0.0, ()),
        (1000.0, ("-t", "1000")),
        pytest.param(
            3000.0,
            ("-t", "3000"),
            marks=pytest.mark.xfail(strict=True, reason=SHARC_THERMAL_DIFFERENCE),
        ),
    ],
)
def test_wigner_statistics_match_sharc(tmp_path, temperature_K, sharc_args):
    run_wigner = _load("run_wigner")
    initconds = _load("initconds")

    settings = ElectronicSection.model_validate(
        {
            "method": "tda",
            "basis": "sto-3g",
            "functional": "hf",
            "singlets": 1,
            "scf_conv_tol": 1e-11,
        }
    )
    coords = GEOMETRY_ANGSTROM * ANGSTROM_TO_BOHR
    masses = np.array([isotope_mass(s) for s in SYMBOLS])
    backend = PySCFBackend(MoleculeSpec(SYMBOLS), settings)
    analysis = harmonic_analysis(masses, coords, backend.ground_state_hessian(coords))
    molden = tmp_path / "freq.molden"
    molden.write_text(
        format_molden(SYMBOLS, coords, analysis.frequencies_cm1, analysis.modes_cartesian)
    )

    created = run_wigner.run_wigner(
        Path(os.environ["SHARC_HOME"]),
        os.environ["SHARC_PYTHON"],
        molden,
        2000,
        tmp_path / "sharc",
        extra_args=("-r", "7", *sharc_args),
    )
    path = next(p for p in created if p.name == "initconds")
    sharc = initconds.parse_initconds(path.read_text())

    ours = sample_wigner(
        masses,
        coords,
        analysis.frequencies_cm1,
        analysis.modes_mass_weighted,
        n_samples=20000,
        temperature_K=temperature_K,
        low_frequency_cutoff_cm1=10.0,
        rng=np.random.default_rng(1),
    )
    report = [f"frequencies cm-1: {analysis.frequencies_cm1}"]

    # Masses and zero-point energy.
    np.testing.assert_allclose(sharc["masses_amu"], masses, atol=1e-6)
    report.append(
        f"Eharm SHARC {sharc['eharm_hartree']:.10f} ours {ours.expected_mean_energy_hartree:.10f}"
    )
    report.append(f"SHARC header Temp {sharc['temperature_K']}")
    # Measured: SHARC's `Eharm` header is the zero-point energy sum(omega)/2 at every
    # temperature (at 1000 K it still equals our T = 0 value), so it is not the thermal mean.
    zero_point = (
        0.5 * np.sum(analysis.frequencies_cm1[analysis.frequencies_cm1 > 10.0]) / (HARTREE_IN_CM1)
    )
    assert sharc["eharm_hartree"] == pytest.approx(zero_point, abs=1e-8)
    if temperature_K == 0.0:
        assert sharc["eharm_hartree"] == pytest.approx(ours.expected_mean_energy_hartree, abs=1e-8)
    else:
        assert ours.expected_mean_energy_hartree > zero_point

    masses_me = masses * DALTON_IN_ME
    # Parser/unit check. SHARC's printed velocities have no net momentum or rotation, and its
    # `Ekin` line is the kinetic energy *before* that removal (measured: the line is never
    # smaller than the energy of the printed velocities, beyond 8-decimal rounding).
    own = 0.5 * (masses_me[None, :, None] * sharc["velocities_au"] ** 2).sum(axis=(1, 2))
    line = sharc["ekin_hartree"]
    report.append(
        f"SHARC Ekin line - Ekin(printed velocities): min {np.min(line - own):+.3e}, "
        f"mean {np.mean(line - own):.3e}, max {np.max(line - own):.3e}"
    )

    checks = []
    theirs_internal = _internal(sharc["positions_bohr"])
    ours_internal = _internal(ours.positions_bohr)
    for key in ours_internal:
        checks.append((key, *_compare(key, ours_internal[key], theirs_internal[key], report)))
    before = ours.kinetic_energy_hartree + ours.removed_kinetic_energy_hartree
    checks.append(("Ekin_after", *_compare("Ekin_after", ours.kinetic_energy_hartree, own, report)))
    checks.append(("Ekin_before", *_compare("Ekin_before", before, line, report)))

    l_ours = _angular_momentum(masses_me, ours.positions_bohr, ours.velocities_au)
    l_sharc = _angular_momentum(masses_me, sharc["positions_bohr"], sharc["velocities_au"])
    p_ours = np.linalg.norm((masses_me[None, :, None] * ours.velocities_au).sum(axis=1), axis=1)
    report.append(
        f"max|L| ours {l_ours.max():.3e} SHARC {l_sharc.max():.3e}; max|P| ours {p_ours.max():.3e}"
    )
    print("\n" + "\n".join(report))

    assert np.min(line - own) > -1e-6, "SHARC printed velocities carry more energy than its Ekin"
    for name, z, p, z_std in checks:
        assert z < 4.0, f"{name}: means differ by {z:.2f} standard errors"
        assert p > 1e-3, f"{name}: KS p-value {p:.2e}"
        assert z_std <= 4.0, f"{name}: standard deviations differ by {z_std:.2f} standard errors"
    assert l_ours.max() < 1e-3
    assert l_sharc.max() < 1e-3
    assert p_ours.max() < 1e-8
