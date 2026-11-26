# Changelog

All notable changes to gqteashmd. Versions follow SemVer 0.x; the program is
Linux only.

## Unreleased

### Changed
- `analysis` stage version 4 (USER_MANUAL 8.13, 8.19):
  - With no complete trajectory, the analysis horizon is capped at the last grid time at which
    any trajectory is observed. Before, the censored curve was held constant up to `tmax_fs`.
    The warning now says the horizon is capped, and `analysis.json` has `horizon_capped`. A cap
    at t = 0 stops the stage (`no trajectory is complete and none is observed after t = 0`).
  - A third estimate, hold-in-S0 (lost trajectories held in S0 from their censoring step), is
    reported beside the complete and censored ones (`estimators.hold_in_s0`; columns
    `hold_in_s0_*` in `populations.csv` and `populations.h5`). It is not an `estimator` value
    and is `null` without a censored trajectory.
  - The range of the three lifetimes is `analysis.json` `tau_range_fs` and a `Lifetime range`
    section of `report.md`. Which estimate is closest to the truth is not known.
  - `analysis.json` `schema_version` 3. The complete and censored numbers do not change.
- The IPCW trial (`tools/validation/censoring_report.py`, `tools/validation/ipcw.py`; report
  `docs/superpowers/reports/2026-11-26-ipcw-trial.md`): an inverse-probability-of-censoring-
  weighted estimator with the S0 kinetic energy as covariate. By the rule fixed before the run
  it is **not** built into the stage: it corrects kinetic-energy-dependent censoring in a control
  test, and the real losses do depend on the S0 kinetic energy, but on the hot-S0 test the
  hold-in-S0 estimate stays less biased (USER_MANUAL 8.19).

### Upgrade notes
Every project with `analysis` results needs `--rerun-from analysis` (analysis stage version 4;
seconds, the trajectories stay). `gqteashmd run` stops with exit 3 until then.

## 0.1.0 - 2026-11-26

The first release: the whole workflow, from the geometry to populations and a lifetime, driven
by one input file, with PySCF as the electronic-structure program.

### Implemented
- Commands: `init`, `validate`, `run [--rerun-from STAGE]`, `status`, `stop`, `single-point`,
  `model-dynamics`, `--version` (USER_MANUAL 5).
- Stages: `prepare`, `optimize` (optional), `frequencies`, `sampling`, `excitations`,
  `selection`, `dynamics` (with a `[dynamics]` table) and `analysis` (USER_MANUAL 4.2, 8).
- Electronic structure (PySCF): RHF/RKS, SA-CASSCF with the CASSCF start rule, TDA, TDDFT,
  CIS, TDHF; energies, analytic gradients, transition dipoles, oscillator strengths, SA-CASSCF
  nonadiabatic couplings, wavefunction overlaps (USER_MANUAL 8.6).
- Dynamics: velocity Verlet, local diabatization or NAC coupling, SHARC-type and Tully hopping,
  energy-based decoherence, step rejection with halving, restarts after every step, extension of
  finished trajectories by raising `tmax_fs`; singlets only with PySCF; triplets and spin-orbit
  coupling on analytic and LVC models through `model-dynamics` (USER_MANUAL 8.10-8.16).
- Analysis: classical and quantum populations with 95 % intervals, a delayed mono-exponential
  fit with bootstrap uncertainties, hop statistics, energy diagnostics; the censored
  (Aalen-Johansen) estimator by default and the complete-trajectory rule beside it
  (USER_MANUAL 8.13, 8.19).
- Crash-safe, resumable orchestration on one workstation: preflight checks, change detection,
  review pauses, provenance of every run, which records the commit of a source checkout or of
  the wheel's build stamp (USER_MANUAL 6, 7).

### Validation
Asserted results are tests that fail when the bound is missed; pre-registered results are pass
rules fixed before the run and checked by a tool run by hand, not by a test; report results are
measurements recorded without a pass rule.
- Energies, gradients and NACs against SHARC 4.1 (`SHARC_PYSCF.py`), ethylene
  SA2-CASSCF(2,2)/6-31G*: 1.4e-8 Eh, 1.4e-8 Eh/bohr, 9.1e-8 1/bohr up to sign. Finite-difference
  gradients: 1.8e-7 Eh/bohr (SA-CASSCF(2,2)/6-31G, ethylene) and 5.7e-8 Eh/bohr
  (TDA-B3LYP/STO-3G, `grid_level` 5). Asserted (USER_MANUAL 8.6).
- Harmonic frequencies against ORCA 6.1.1 (water RHF/6-31G*, within 0.052 cm^-1) and PySCF
  `thermo` (1e-6 cm^-1); SHARC's `wigner.py` reads our Molden file. Asserted (8.7).
- Wigner sampling against SHARC's `wigner.py` at 0 and 1000 K (means within 1.6, standard
  deviations within 1.4 standard errors). Asserted; SHARC's 3000 K sampling is not the exact
  thermal distribution (a strict xfail) (8.8).
- Excitations and selection against `SHARC_PYSCF.py` and `excite.py` (dE <= 8.7e-8 Eh; the
  selection weight measured as f/dE^2). Asserted (8.9).
- Dynamics core: FSSH and local diabatization against exact wavepackets on Tully's models
  (asserted); FSSH against an exact 2D LVC solver (report: largest difference 0.049 in the
  quantum and 0.072 in the classical populations); single trajectories against SHARC 4.1
  (nuclear motion 7.5e-11 bohr; LVC single points 2.5e-11; asserted). Ensembles against SHARC
  are reports (8.14, 8.15).
- The `dynamics` stage: restart and extension bit-identical with an analytic backend, to
  round-off with PySCF; a SHARC swap test (sharc.x driving our backend) dq 2.1e-7 bohr over 41
  steps. Asserted (8.16).
- The SHARC ensemble comparison (ethylene SA3-CASSCF(2,2)/6-31G*, 100 trajectories): the swap
  ensemble passes all four rules (48 paired; pre-registered, run by hand); our population
  functions agree with SHARC's `populations.py` on 100 SHARC trajectories (pre-registered, run
  by hand); the direct comparison is inconclusive (13
  paired, report) (8.17).
- The hop energy option: with `hop_energy = "previous"` a replayed trajectory agrees with SHARC
  over 20 steps after its hop (asserted, one trajectory) (8.18).
- The censored estimator: unbiased and covering on a synthetic Markov model (asserted); least
  biased of three under artificial random losses (pre-registered, run by hand) (8.19).

### Known limitations
- Singlets only from PySCF; no triplets or spin-orbit coupling for molecules (O7).
- Trajectories are lost after a hop by CASSCF solution switches (33 of 100 in the P9b ethylene
  ensemble); the dynamics are not changed to keep them (USER_MANUAL 5.10, 8.16.9, 8.17.1).
- The censored estimator's assumption is argued against by the measured back-hop rates on the
  ethylene data, so its shorter lifetime there (about 62 fs against 80 fs for the complete
  trajectories) is not shown to be unbiased (8.19).
- The direct SHARC comparison is inconclusive and the default-rule replay after a hop fails (a
  strict xfail); the stage's lifetime fit is not compared with SHARC's `make_fit.py` (8.17).
- With `hop_energy = "previous"` the drift check cannot see an energy change at an accepted hop
  (8.18).
- Restart robustness: a `meta.json` without `log_records` is accepted (it would truncate
  `steps.log`); no fsync of the restart generations (safe against a process kill, not a power
  loss); a short window between starting a worker and recording it (deferred).
- One live-SHARC test is flaky (water CAS(4,4)/STO-3G excitation energies; never in CI).

### Upgrade notes
Projects computed with code before this release need a rerun from the first changed stage:
`--rerun-from dynamics` (dynamics stage version 3; it also recomputes the analysis),
`--rerun-from analysis` (analysis stage version 3), `--rerun-from excitations` (excitations
stage version 2), and for CASSCF projects `--rerun-from optimize` (or `frequencies` without
`optimize`) for the CASSCF start rule. `gqteashmd run` stops with exit 3 and names the stage.

### Environment
Tested with Python 3.12.16, PySCF 2.14.0, NumPy 2.5.4, SciPy 1.18.1, h5py 3.16.0,
geomeTRIC 1.1.1 and pydantic 2.14.3 on Oracle Linux 9.5 (WSL2); the exact environment is
`environment.lock.txt` (linux-64).
