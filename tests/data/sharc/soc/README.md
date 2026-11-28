# SHARC 4.1 spin-orbit QM.out probe fixtures (dynamics with SOC, Task 1)

Observed input and output of `sharc.x` (it prints `SHARC_version 4.0`), produced on 2026-11-27
in WSL Oracle Linux 9.5. SHARC was only executed. Nothing here is SHARC source.

`sharc.x` drove the P7a-2 LVC spin-orbit model `tests/data/lvc/lvc_soc.template` (2 singlets +
2 triplets, 8 expanded states) through our own `QM.out` provider
(`tools/sharc_compare/run_sharc_lvc.py`: `QmServer`, `format_qm_out`, `write_server_runqm`),
from the first Wigner sample of `q, v = lvc_initial_conditions(model, 2, 1, "wigner")`
(`q[0]`, `v[0]`; `model = LVCModel.from_template(...)`), start
`state 2 mch` (S1), 3 steps of 0.5 a.u., `coupling overlap` (`eeom ld`), `nogradcorrect`,
`surf diagonal`, `ekincorrect parallel_vel`, EDC (0.1 Eh), SHARC hopping, `rngseed 7`.

- `input`: the sharc.x input of the default run (no `grad_all`/`grad_select` line).
- `QM.in.step0`, `QM.in.step1`: what sharc.x requested at steps 0 and 1 of that run
  (`SOC`, `DM`, `GRAD all`; `OVERLAP` from step 1 on; no `NACDR`, no `PHASES`).
- `QM.in.select.step0`, `QM.in.select.step1`: the same with `grad_select`, `nac_select`,
  `eselect 0.5` (step 0 lists all 8 gradients, step 1 `GRAD 2 4 6 8`, 1-based expanded indices).
- `QM.out.step1`: our provider's answer at step 1 (full complex H with SOC, 8 expanded
  gradients, overlaps, zero dipoles), as sharc.x read it.
- `output.dat`, `output.lis`: the default run (steps 0-3). The step-0 overlap block of
  `output.dat` is a placeholder filled with -123 (no overlaps exist at step 0). The printed
  triplet diagonal of H carries sharc.x's own offsets (+1e-12 Eh on M_s = 0, +2e-12 Eh on
  M_s = +1) relative to the `QM.out` it read.

The paths in `QM.in` (`savedir`) are those of the probe machine.
