# SHARC 4.1 fixtures for P9b

Observed output of SHARC 4.1 (`SHARC_version 4.0` is printed by `sharc.x`; `populations.py` and
`make_fit.py` print "Version:4.0, 01.04.25"), produced on 2026-11-14 in WSL Oracle Linux 9.5.
SHARC was only executed. Nothing here is SHARC source.

## `traj_00000/`, `traj_00001/`: two swap trajectories

Ethylene (`tests/data/ethylene.xyz`, velocities from `default_rng(2026)` at 300 K), start in S1
(`state 2 mch`), 10 steps of 0.5 fs, `sharc.x` driving our `PySCFBackend`
(SA3-CASSCF(2,2)/6-31G*, CASSCF start rule with the reference at the start geometry,
`threads=2`) through the swap provider of `tools/sharc_compare/run_sharc_pyscf_dynamics.py`:

    run_sharc_trajectory(interface_geometry(q0), v0, n_steps=10, provider="ours",
        model=SwapModel(backend, coupling="overlap"), coupling="overlap", n_states=3,
        state_mch=1, hopping="sharc", decoherence="edc", seed=k + 1)

for k = 0, 1 (a fresh backend each). Files kept: `output.dat`, `output.lis`, `input` (the sharc.x
input, which `populations.py` reads to find the number of states) and
`output_data/coeff_MCH.out` (made by `data_extractor.x output.dat`). No hop happened in 10 steps
(S1 stays active; the two trajectories have the same start and differ in the seed, i.e. in the
random numbers, and by round-off of about 1e-12), so these files contain no hop line.

## `lis_samples/`: hop and frustrated-hop lines

`output.lis` and `input` of two Tully simple-avoided-crossing runs (one atom of mass 2000, 2
states, `tools/sharc_compare/run_sharc_model.py:run`, provider `fast`, dt 1 a.u., `hopping sharc`):
`tully_hop_*` x0 = -0.4, v0 = 0.02, 40 steps, rngseed 6 (an accepted hop, line 33);
`tully_frustrated_*` x0 = -0.4, v0 = 0.003, 150 steps, rngseed 8 (a frustrated hop, line 46).
They are here only to show the two line formats of `output.lis`, which do not depend on the
interface.

## `analysis/`

Run in a directory `Singlet_1/` that holds `TRAJ_00000`, `TRAJ_00001` (copies of the two
trajectories above), with `SHARC=<home>/bin`, `PYTHONPATH=<home>/lib:<home>/pysharc`,
`LD_LIBRARY_PATH=<sharc conda env>/lib`, `HOSTNAME` and `USER` set. Every prompt and answer is in
`transcripts/` (`*.transcript.txt` the tool's output, `*.answers.txt` the stdin).

- `pop_2.out`, `traj_per_step_pop_2.out`: `populations.py`, mode 2 (classical, active MCH state).
- `pop_8.out`, `traj_per_step_pop_8.out`: mode 8 (quantum, `|c|^2` in the MCH basis) after
  `data_extractor.x output.dat` was run by hand in each trajectory directory.
- `bootstrap_data/pop_1.dat`, `pop_2.dat`: `populations.py` mode 2 with "Save data for bootstrapping".
- `fit_results.txt`, `fit_results.gp`: `make_fit.py`, species S0 and S1, reaction S1 -> S0
  (`k10`), S1 initial population, `S1 = 3 4` fitted to `pop_2.out`. The hostname and directory
  in `fit_results.gp` were scrubbed.
- `fit_bootstrap.txt`: `make_fit.py` bootstrap mode (20 samples) on `bootstrap_data/`.
