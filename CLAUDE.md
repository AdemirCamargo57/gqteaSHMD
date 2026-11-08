# CLAUDE.md — gqteashmd

`gqteashmd` is our own surface-hopping molecular dynamics (SH-MD) program. It automates the SHARC-style workflow:

```text
optimize → frequencies → Wigner sampling → initial conditions → trajectories → analysis
```

The user writes one TOML input file and runs `gqteashmd run input.toml`. The program then runs every stage to completion, survives crashes and restarts, and resumes where it stopped. Electronic structure comes from PySCF only (first release). The program runs on Linux only: an Oracle Linux 9 workstation, with no batch scheduler.

## Read first when resuming
- **Status and what's next:** the "Current status" section below.
- **The binding spec:** [docs/superpowers/plans/2026-11-06-gqteashmd-master-plan.md](docs/superpowers/plans/2026-11-06-gqteashmd-master-plan.md). It holds the architecture, the stage contracts, the error rules, decisions D1–D20 and open items O1–O20.
- **Scientific background:** [SHMD_implementation_strategy.md](SHMD_implementation_strategy.md), covering methods, SHARC 4.1 scope, the validation strategy and risks.
- **The P1 plan:** [docs/superpowers/plans/2026-11-06-p1-foundation-orchestrator.md](docs/superpowers/plans/2026-11-06-p1-foundation-orchestrator.md). It is the historical record of how the foundation was built.
- **The P1 ledger:** `.superpowers/sdd/2026-11-06-p1-foundation-orchestrator/progress.md`. It is git-ignored and local to the development machine. It contains every ruling made during P1 and the deferred findings.
- **The P3a plan and ledger:** [docs/superpowers/plans/2026-11-07-p3a-pyscf-backend.md](docs/superpowers/plans/2026-11-07-p3a-pyscf-backend.md), and `.superpowers/sdd/2026-11-07-p3a-pyscf-backend/` (git-ignored, local; `progress.md` and the task reports).
- **The P2 plan and ledger:** [docs/superpowers/plans/2026-11-06-p2-input-preflight.md](docs/superpowers/plans/2026-11-06-p2-input-preflight.md), and `.superpowers/sdd/2026-11-06-p2-input-preflight/progress.md` (git-ignored, local). The plan's "Scope rulings" say which deferred items stay deferred and to which sub-project they belong.
- **The user-facing behaviour:** [USER_MANUAL.md](USER_MANUAL.md).

## Rules that always apply
- **Clean room.** SHARC 4.1 (installed at `\\wsl.localhost\OracleLinux_9_5\home\ajc\sharc4-4.1`) is a reference for the workflow and a black box for output comparison. Implement methods from the literature. Never read, copy or adapt SHARC source. SHARC may only be *executed*, from `tools/sharc_compare/` (it runs `SHARC_PYSCF.py` and parses `QM.out`), and must never be imported by `gqteashmd`. Live comparison: `SHARC_HOME=/home/ajc/sharc4-4.1 SHARC_PYTHON=/home/ajc/miniforge3/envs/sharc4.1/bin/python pytest -m sharc` (about 12 s; skipped without those variables; never in CI). The licence is MIT ("The gqteashmd developers").
- **Run everything in WSL.** The repository lives at `C:\Users\ajc\OneDrive\gqteaSHMD` (`/mnt/c/Users/ajc/OneDrive/gqteaSHMD` in WSL), but code, tests and git run inside WSL Oracle Linux 9.5. From the Windows host, wrap each command:
  ```bash
  wsl -d OracleLinux_9_5 -- bash -lc 'cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && conda activate gqteashmd-dev && <command>'
  ```
  Edit files with the normal file tools, using LF line endings.
- **Git from WSL only.** Pre-commit hooks (ruff 0.17.4) are installed in WSL and do not run from Windows Git. Pushing from Windows is fine (`git -C C:\Users\ajc\OneDrive\gqteaSHMD push ...`), because Windows has the GitHub credentials and pushing runs no hooks. Never use `--no-verify`.
- **Keep project runs, conda environments and large outputs out of OneDrive and out of `/mnt/c`.** Use `~/...` or `/tmp` in WSL. pytest's `tmp_path` already does this.
- **Workflow.** Feature branch per sub-project (`feat/<topic>`), one commit per task, conventional commit subjects (`feat:`, `fix:`, `test:`, `docs:`, `ci:`, `build:`). The owner decides merges. P1, P2, P3a and P3b are merged into `main`. `main` up to P2 (`e73609f`) is on GitHub (`AdemirCamargo57/gqteaSHMD`); the P3a and P3b merges are local until the owner pushes. P3c is on `feat/p3c-hardening` and not merged.
- **Commit trailer.** End each commit message with the `Co-Authored-By:` trailer for the model that wrote the commit.
- **Keep the two documents current.** Update **CLAUDE.md and USER_MANUAL.md in the same branch as any change to behaviour, inputs, outputs, formulas or status.** USER_MANUAL.md must always mark what is implemented and what is planned.

## Environment
- Conda env `gqteashmd-dev` (Miniforge in WSL), created from `environment.yml`. The package is installed editable with `pip install -e . --no-deps`. Python 3.12.
- Runtime dependencies (`pyproject.toml`): `pydantic>=2.6`, `numpy>=1.26`, `pyscf>=2.6`. `environment.yml` pins `pyscf=2.14.*` (O18). SciPy, h5py and geomeTRIC are installed for P4+ and not imported yet.
- Tests (markers `slow`, `pyscf`, `sharc`). Three tiers, measured at the end of P3c on `/mnt/c`:
  - **Fast tier:** `pytest -m "not slow and not pyscf" -W error`: 293 passed in about 7 min. Use it while developing.
  - `pytest -m "not slow" -W error`: 358 passed in about 11.5 min (adds the `pyscf` tests).
  - `pytest -W error` (everything, including the slow tests; live SHARC needs `SHARC_HOME`/`SHARC_PYTHON`): 363 passed in about 14 min.
  - **Why the fast tier is not fast:** on `/mnt/c` it is dominated by the subprocess-heavy orchestrator and CLI tests, because Python start-up is slow through the WSL 9p mount, not because of PySCF. Suggestion, not measured: run the suite from a clone on the WSL native filesystem (e.g. `~/src/gqteaSHMD`), which should be much faster.
  - The slow tests are the finite-difference gradient and NAC checks and the 2 kill/resume tests from P1.
  - The tests start real subprocesses, so they need Linux.
  - Finite-difference tests build a **fresh backend for every displaced geometry**, so orbital reuse cannot bias the reference.
- Lint: `ruff format . && ruff check .`. The line length is 100, with rules E, F, W, I, B, UP and SIM. Markdown is excluded (`extend-exclude`), because ruff ≥0.17 would rewrite code blocks in the plans.
- Types: `mypy` runs clean on `src/` (configuration in `pyproject.toml`) and is part of CI.
- CI: `.github/workflows/ci.yml` runs in an `oraclelinux:9` container on push, on PRs and nightly. It first ran on 2026-11-07 (`main` at `e73609f`) and passed.
- Workstation facts in WSL: 16 cores and about 7 GB RAM. ORCA 6.1.1 and Gaussian 16 are installed, but they are not used before phase P7.

## Architecture (implemented in P1, P2, P3a and P3c)
Dependencies point down only; domain code never imports stages or the orchestrator.

```text
cli.py ─► orchestrator.Orchestrator ─► execution.local.LocalExecutor (one per stage) ─► subprocess: python -m gqteashmd.worker task.json
   │         │  preflight.py (before the project directory is touched)       worker ─► stage.run_task() ─► result.json
   │         │  project/state.py (SQLite, single writer), project/provenance.py
   │         │  project/layout.py, project/lock.py (flock, taken first)
   │         └► stages/ (Stage contract: plan → run_task → finalize) ─► chem/, core/, util/, electronic/, (later) vib/, dynamics/
   ├► single-point command ─► electronic.factory.make_backend ─► electronic.pyscf_backend.PySCFBackend (no project, no stages)
   └► logging_setup.py (console INFO, logs/gqteashmd.log DEBUG)

tools/sharc_compare/   executes SHARC 4.1 as a black box (never imported by gqteashmd); tests in tests/sharc_compare, pytest -m sharc
```

- **`core/`:** `units.py` (CODATA 2018 constants; Hartree atomic units internally, conversions only at file edges), `states.py` (`StateLabel(multiplicity, index)`, `states_from_counts`; labels `S0`, `S1`, ..., `T1`, ...). `chem/electrons.py:count_electrons` is the one electron-count/parity rule, shared by `prepare` and the backend.
- **`electronic/`** (P3a, P3b): `base.py` (`MoleculeSpec`, `Capabilities`, `Request(coords_bohr, gradients, transition_dipoles, nacs, overlaps)`, `ElectronicResult` with `nacs`, `overlaps`, `phases` and `to_json()`, `ElectronicBackend` with `compute`, `compute_more`, `reset`, `save_state`, `load_state`), `properties.py` (oscillator strength `f = 2/3 dE |mu|^2`), `factory.py` (`make_backend`), `pyscf_backend.py` (`PySCFBackend(molecule, settings, *, log_file, threads, memory_mb, scratch_dir)`, `require_pyscf`, SCF ladder, snapshot, phase application), `pyscf_casscf.py` (CASSCF ladder, gradients, NACs, determinant expansion), `pyscf_response.py` (TDA/TDDFT ladder, gradients, dipoles, expansion), `pyscf_errors.py` (`wrap_failure`).
- **`overlaps/`** (P3b): `determinants.py` (`DeterminantExpansion(alpha, beta, coeffs)`, `truncate`, `state_overlaps`) and `tracking.py` (`track`, `TrackedOverlaps`, `lowdin_orthonormalize`). Backend-independent: plain NumPy/SciPy.
  - **PySCF is imported lazily, and only in `electronic/pyscf_*.py`** (inside functions), so the rest of the package imports without it. `overlaps/` never imports PySCF.
  - **The backend's only cross-geometry state is `_Snapshot`** (coords, MOs, density, phase-aligned expansion). `save_state`/`load_state` write and read exactly that (`.npz`, settings fingerprint, atomic). The fingerprint holds the method, basis, functional, atoms, charge, multiplicity, states, `ncas`/`nelecas` and `overlap_norm_kept` (not the CASSCF tolerances). A missing, corrupt or inconsistent file is a `ValueError` ("is not a readable gqteashmd electronic state"). Restarting from an older snapshot keeps the sign frame (the snapshot is aligned); continuity is lost only if no state was saved. **One backend per trajectory:** call `reset()` (or use a new backend) before an unrelated geometry.
  - **Phases are always tracked once a snapshot exists** (ruling); `Request.overlaps` only decides whether `overlaps`/`phases` are returned. Dynamics should still request overlaps every step, because it needs them. Phases are applied to the stored expansion, NACs (`p_I p_J d_IJ`) and dipoles (`p_0 p_I mu`). PySCF's raw CI signs are **not reproducible** across processes; phased output is reproducible only relative to a snapshot.
  - **Truncation keeps the union** over states of each state's important determinants (`overlap_norm_kept`, default 0.9999); a per-state cut destroys the couplings. NAC pairs are I < J. `compute_more` reuses the live PySCF objects of the last successful `compute` (error before it, after `load_state` and after a failed `compute`, which clears them). Errors inside truncation/overlap/tracking become `ElectronicStructureError` (non-retryable; `MemoryError` retryable).
  - **Overlap memory model:** `state_overlaps` never forms the n_det x n_det matrix. Each state is scattered onto an (N_alpha_unique x N_beta_unique) matrix and contracted with the two determinant tables, so memory is O(N_alpha_uniq*N_beta_uniq) per state; it estimates the bytes first and raises `MemoryError` above `MAX_OVERLAP_BYTES` (2 GiB). It takes an optional `s_mo_beta` for future unrestricted references.
  - Ladders: CASSCF (previous orbitals, SCF orbitals, SCF + spin penalty) and TD (default, extra roots); scratch: no chkfile, `pyscf.lib.param.TMPDIR` set to `scratch_dir` only during calls.
  - NACs exist for CASSCF only (`capabilities.nacs`); TDA/TDDFT use overlaps, built from X only (Y dropped, renormalised).
  - SCF ladder: default, then level shift 0.2 Eh + damping 0.5, then second-order. Only rung 1 starts from the previous geometry's density.
  - SA-CASSCF uses a **singlet-only CI solver** (`fci.solver(mol, singlet=True)`) with equal weights `1/N`, and a **spin guard**: `|<S^2>| > 1e-3` for any root is an error. The solver excludes triplets but not higher spins (quintets can appear in larger active spaces), which is why the guard exists. SCF ladder rung 2 sets `diis_start_cycle = 8`, because PySCF damps only before DIIS starts.
  - **Orbitals are reused between calls of one backend object only via `mcscf.project_init_guess`.** Never pass old MOs directly: they are not orthonormal at a new geometry.
  - Transition dipoles: CASSCF `-sum r_pq gamma^{0I}_pq` (active orbitals, `trans_rdm1`); linear response as returned by PySCF `td.transition_dipole()` (length gauge, no sign change). Their overall sign is arbitrary, so compare up to sign.
  - **Error mapping:** user input errors are `ConfigError` (exit 2: missing PySCF or version < 2.6, multiplicity != 1, charge/multiplicity vs electron count, missing `[electronic]`). The PySCF calls that run SCF, CASSCF, TD, gradients, dipoles and orbital projection are wrapped by `pyscf_errors.wrap_failure` into `ElectronicStructureError` (a `TaskError`, exit 4: unknown basis, SCF/CASSCF/TD non-convergence, spin, active space, gradient/dipole failures). It is non-retryable, except that `MemoryError` and `OSError` raised inside PySCF give `retryable=True` (transient I/O and out-of-memory are retried, spec §6.1). Do not let a raw PySCF exception escape from those steps. Config-level `singlets` is checked against the CAS singlet space (Weyl count) when the input is loaded.
  - Not implemented: Hessian, triplets/SOC, open-shell, density fitting, TDDFT NACs, a scratch key in the input file.
- **Stage contract** (`stages/base.py`):
  - `plan(ctx)` runs in the controller and returns `TaskSpec` objects deterministically.
  - `run_task(ctx, task, task_dir)` runs in a worker subprocess. It must be idempotent, write only in `task_dir`, and return a small JSON-serialisable dict.
  - `finalize(ctx, results)` runs in the controller, writes the stage artifacts and raises `StageFailedError` if checks fail.
  - Stage attributes: `name`, `version`, `depends_on`, `config_sections`, `required_modules` (checked by preflight), `max_failed_fraction`, `enabled(config)`.
- **Worker protocol** (`worker.py`): the worker reads `task.json` and writes `result.json` atomically, either `{"status": "ok", "result": ...}` or `{"status": "error", "error_type", "message", "retryable", "traceback"?}`. Errors are also written to stderr, which goes to `worker.log`. A worker that exits without `result.json` is treated as a crash, which is retryable. An unreadable or malformed `task.json` gives a non-retryable error result (`Cannot read task file …`, `ProjectError`) written next to it; a non-serialisable result (`TypeError`/`ValueError`) is reported the same way. `write_bytes_atomic` removes its temp file if the write fails. Workers read the config snapshot `.gqteashmd/config.json` and never the user's input file.
- **Executor** (`execution/local.py`):
  - one subprocess per task, each in its own session;
  - the orchestrator builds **one executor per stage** from that stage's effective resources (`ResourcesSection.for_stage(name)`: a `[resources.stages.<name>]` key, or the global value if unset);
  - at most `max_parallel_tasks` running at once;
  - sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to `threads_per_task`, and `PYSCF_MAX_MEMORY` to `memory_mb_per_task`;
  - on stop: SIGTERM to the process group, SIGKILL after `stop_grace_period_s`;
  - **whole-group kills:** a worker's process group also gets SIGKILL after the leader exits normally, on a stop, and in `terminate_process_group(pid, grace)`, so children that stay in the worker's process group are killed while the controller runs (children that call setsid/setpgid, and strays of a worker that finished while the controller was dead, are not); the stop grace period applies to the leader only;
  - if a callback raises, it reaps every worker before re-raising.
- **Orchestrator rules** (`orchestrator.py`, spec §3.4):
  - **`Orchestrator.session()`** (context manager, re-entrant): collects the provenance record, takes the project lock, then calls `layout.ensure()` and removes stale `.<name>.tmp` files from `.gqteashmd/` (left by a crashed controller). `run()` and `cli._cmd_run` both use it, so **one session = one run = one provenance record**, and the lock comes before `ensure()` and before logging. A run refused by the lock creates only `.gqteashmd/` (for the lock file) and writes no log and no provenance.
  - **Deleted stage directory:** a DONE stage whose results directory is missing is reset (`state.reset_stage`) and recomputed, with a warning; its fingerprint is unchanged.
  - **Fingerprint:** SHA-256 over the stage name, version, `fingerprint_inputs()` and the upstream fingerprints. A DONE stage with the same fingerprint is skipped.
  - **Changed fingerprint:** a stage that is DONE or RUNNING, **or that has any DONE task**, raises `StaleStageError`, which tells the user to run `--rerun-from STAGE`. Only stages with no finished task are reset automatically.
  - **`--rerun-from`:** moves the old stage directories to `.gqteashmd/superseded/`, never deleting them. It first terminates orphaned workers.
  - **Retries:** a crash or a retryable error is retried while `attempts < max_attempts`. An interrupted task is not counted as an attempt.
  - **Stage failure:** a stage fails when `failed/total > max_failed_fraction`. Any exception inside a stage marks it FAILED.
  - **Restart:** tasks left RUNNING by a dead controller are reconciled. A task with an ok `result.json` becomes DONE; otherwise its orphan is killed (verified via `/proc/<pid>/cmdline`) and the task is requeued.
  - **`--rerun-from`** is applied once per orchestrator (`self.rerun_from` is cleared after use). The hint printed for a stale stage is shell-quoted. A fingerprint input that is not JSON-serializable raises `ProjectError`.
  - **Pause:** after a stage listed in `workflow.pause_after` completes **in this run** and is not the last stage, `RunSummary.paused_after` is set, the loop stops and the CLI prints a message and exits 5 (`EXIT_STOPPED`). A rerun continues, because that stage is then up to date. A skipped or up-to-date stage never pauses. Unknown names in `pause_after` and in `[resources.stages]` are a `ConfigError` raised by `check_stage_names` (`stages/pipeline.py`), which both `Orchestrator.__init__` and `validate` call.
- **Config** (`config/schema.py`): `[resources]` requires `total_cores` and `total_memory_mb`. A model validator checks `max_parallel_tasks × threads_per_task ≤ total_cores` and `max_parallel_tasks × memory_mb_per_task ≤ total_memory_mb` for the global values and for every stage override (merged with the global values); the message shows the factors (`3 × 2 = 6 exceeds total_cores = 4`). The keys of `[resources.stages]` must be safe stage names (`is_safe_name`). `execution.min_free_disk_mb` (default 1000) and `[workflow] pause_after` also live here. `loader.py` expands `~` (an unknown `~user` is a `ConfigError`) and resolves relative paths against the input file's directory. `[electronic]` (`ElectronicSection`, with the optional `CasscfSection` as `[electronic.casscf]`) is in the schema since P3a: `method` casscf|tda|tddft, `basis`, `functional` (required for tda/tddft, forbidden for casscf; `"hf"` gives CIS/TDHF), `singlets` (**includes S0**, O19), `triplets` (must be 0), `grid_level` (3), `scf_conv_tol`, `scf_max_cycle`, `td_conv_tol`, `overlap_norm_kept` (0.9999, 0.9 < x <= 1); casscf: `ncas`, `nelecas` (even, <= 2 ncas), `conv_tol`, `conv_tol_grad`. A present `[electronic]` requires `molecule.multiplicity = 1`. The later sections (`[optimize]`, `[frequencies]`, `[sampling]`, ...) arrive with P4-P8.
- **Preflight** (`preflight.py:run_preflight`): runs from `Orchestrator.preflight()` (called by `cli._cmd_run`; the result is cached, so `run()` does not repeat it), **before** the lock and `layout.ensure()`.
  - It raises `ConfigError` for: a project directory on a Windows drive in WSL (`/mnt/<letter>/`; `/mnt/data` is fine) or under a OneDrive folder; a project directory that exists as a file, or whose nearest existing parent is not a writable directory; free disk below `min_free_disk_mb` or a failed disk query (`min_free_disk_mb = 0` skips the query); a missing `required_modules` entry of an enabled stage.
  - It only returns a **warning** (D17) when `total_cores` or `total_memory_mb` exceeds the detected machine (`detect_machine()`: `sched_getaffinity` and `/proc/meminfo`). The CLI logs warnings after logging is configured, so they reach the console and the log file. As a backstop, `cli._cmd_run` turns an `OSError` from `session()` (lock file, `layout.ensure()`) or `configure_logging` into a `ConfigError`.
  - The location check is on the path text. A relative path is resolved (symlinks followed) against the input file's directory; an absolute path is taken as written, so a symlink in it is not followed.
- **Provenance** (`project/provenance.py`): each `run` appends one record to `.gqteashmd/provenance.json`, once the controller holds the lock (`session()`) and has opened `state.db` (so a run later refused by `StaleStageError` has a record; a run refused by preflight, the lock or a corrupt `state.db` has none) (`started_at`, `gqteashmd`, `git_commit`, `git_dirty`, `python`, `platform`, `hostname`, `cpu_count`, `cpu_model` (first `model name` of `/proc/cpuinfo`, else `null`), `packages`). `git_commit`/`git_dirty` are `null` unless gqteashmd runs from a git checkout (an editable install). A damaged file (invalid JSON, non-UTF-8 bytes, or `runs` not a list) is replaced by a fresh list.
- **State** (`project/state.py`): `StateStore(db_path, *, readonly=False)`. A failure to open or read the database at start-up (a `sqlite3.DatabaseError`, including a db path that is a directory) becomes `ProjectError` ("unreadable or corrupt", exit 3; the advice is to move the project's `.gqteashmd` directory aside, and stage results are then recomputed). Every public query method is wrapped by `_guarded`: a `sqlite3.Error` at query time becomes `ProjectError` ("The state database … failed during <method>: …", exit 3, advice: check disk space and that no other program has the database open); `close` is unguarded. `status` opens the db read-only (`mode=ro`) and never writes.
- **Lock** (`project/lock.py`): `ProjectLock` tries `flock` twice, 0.2 s apart (`locked_by` probes the same lock briefly). The error reads "pid unknown" when the holder's PID cannot be read. The file handle is closed on every failure path; an `OSError` other than "locked" (e.g. a filesystem without `flock`) becomes `ProjectError` "File locking is not available for …".
- **Logging** (`logging_setup.py`): `configure_logging` is called by `run` after the lock is held and the project directory exists; `close_logging` (in `cli.main`'s `finally`) closes all handlers and returns to a `NullHandler`. Records logged with `extra=FILE_ONLY` go to `logs/gqteashmd.log` but not to the console. `cli.main` uses it for errors and tracebacks, so each error is printed **once** on stderr (`gqteashmd: error: <message>`) with no traceback.
- **Errors** (`errors.py`):
  - Exit codes: `ShmdError` 1, `ConfigError` 2, `ProjectError` 3 (`StaleStageError` is a subclass), `StageFailedError` 4. Stopped by the user **or paused for review** is 5 (`cli.EXIT_STOPPED`).
  - `TaskError(msg, retryable=...)` is raised inside tasks; it pickles correctly (`__reduce__`).
  - **Ctrl-C** anywhere in `cli.main` (also while parsing arguments) prints `gqteashmd: interrupted` and exits 5.
  - Every error reaches the user as `gqteashmd: error: <message>`.
  - `.` and `..` are not safe stage or task names (`is_safe_name`). The XYZ parser rejects non-finite coordinates; `validate` also requires the geometry to be a regular file.
- **Testing seam:** `gqteashmd.testing.fake_stages` provides fake stages whose behaviour is set by the environment variable `GQTEASHMD_FAKE_BEHAVIOUR` (JSON). The keys are `sleep`, `fail`, `unexpected`, `crash_once`, `retryable_once`, `plan_raises`, `bad_fingerprint`, `requires` (stage name to required modules) and `skip_optimize`. The CLI's hidden `--pipeline` option selects them in tests.
- **The production pipeline** (`stages/registry.py:default_pipeline`) contains only `prepare` today. `gqteashmd single-point INPUT [--json F] [--log F]` runs `PySCFBackend` once outside the pipeline (no project directory, no preflight): exit 0, 2 (input) or 4 (calculation failed). Its table and JSON schema are in USER_MANUAL.md §5.3.

## Key decisions (full list in master plan §10)
- **Python core.** Electronic structure dominates the cost, so the core is plain Python (D1).
- **PySCF only first.** PySCF is the only backend in the first release; ORCA, Gaussian, OpenMolcas and OpenQP come later behind the same backend contract (D4).
- **Single workstation.** The resumable controller runs one subprocess per task, with no scheduler (D5).
- **Strict input file.** One strict TOML input. Units go in key names (`dt_fs`, `temperature_K`, ...), and state labels are `S0`, `S1`, `T1` (0-based internally) (D7, D13).
- **Hardware from the user.** The user enters the hardware in `[resources]`. The program warns, without stopping, when the values exceed the detected machine (D17, implemented in P2).
- **Naming.** The package and command are `gqteashmd`; internal class names keep the `Shmd` prefix (D18).
- **P3 validation target.** Ethylene (`C:\AJCamargo\TOPICOS\Eteno\Eteno.xyz` → `tests/data/ethylene.xyz`) at SA2-CASSCF(2,2)/6-31G\* first, then TDA (D19). Done in P3a: it matches SHARC 4.1 to 1.4e-8 Eh (energies) and 1.4e-8 Eh/bohr (gradients), the dipole to 4.5e-7 e a0 up to sign; finite-difference gradient checks 1.8e-7 (CASSCF) and 5.7e-8 (TDA-B3LYP, grid_level 5) Eh/bohr.
- **O18 resolved:** `pyscf=2.14.*` pinned in `environment.yml`; runtime `pyscf>=2.6`. **O19 resolved:** `singlets` includes S0 (owner decision 2026-11-07).
- **Method caveats:** PySCF DFT gradients omit the grid response, so use `grid_level >= 5` for dynamics. TDA/TDDFT have the wrong S0/S1 crossing topology; use SA-CASSCF for S1->S0 decay.

## Current status (2026-11-07)
- **P1 — foundation & orchestrator: DONE**, merged into `main`.
- **P2 — input schema & preflight: DONE**, merged into `main`. `main` is on GitHub and CI passed on 2026-11-07.
- **P3a — PySCF electronic-structure layer: DONE**, merged locally into `main` on 2026-11-07 (not pushed yet). `pytest -m "not slow" -W error`: 260 passed, 1 skipped, 4 deselected (5-6 min). With `SHARC_HOME`/`SHARC_PYTHON` set, `pytest -W error` runs everything (the 4 slow tests: 2 FD gradient checks + 2 kill/resume tests; and the live SHARC test): 265 passed in 8.5 min. ruff and mypy are clean.
- **P3b — couplings, overlaps, restarts: DONE, merged locally into `main` on 2026-11-07 (not pushed).** `pytest -m "not slow" -W error`: 335 passed, 1 skipped (live SHARC), 5 deselected (2 FD gradient, 1 NAC FD, 2 kill/resume), 6.5-11 min on /mnt/c; the slow NAC FD test takes 15 s and the live SHARC test (with NACs) 8 s. With `SHARC_HOME`/`SHARC_PYTHON` set, `pytest -W error`: 341 passed in about 13 min. NACs match SHARC 4.1 to 9.1e-8 1/bohr up to sign. Library only: no CLI command or stage uses NACs/overlaps yet (P7). The hardening sweep moved to P3c.
- **P3c — hardening sweep: DONE on `feat/p3c-hardening`, NOT merged** (owner decides). Delivered: lock first (no leaked handle, "File locking is not available"), `Orchestrator.session()`, stale `.tmp` cleanup, `cpu_model`, guarded sqlite queries, whole-group SIGKILL, unreadable `task.json`, picklable `TaskError`, recompute of a deleted DONE stage directory, Ctrl-C exit 5 (the SIGINT/SIGTERM handlers are installed before the lock is taken, so a `stop` during start-up is a graceful exit 5), the `pyscf` test marker. Test tiers: see Environment.
- **Implemented features:**
  - the CLI commands `init`, `validate`, `run [--rerun-from]`, `status`, `stop` and `single-point`;
  - the input sections `[project]`, `[molecule]`, `[resources]` (with per-stage overrides), `[execution]`, `[workflow]` and `[electronic]` / `[electronic.casscf]`;
  - preflight, provenance, review pauses (exit 5), per-stage executors, the `prepare` stage, resumable crash-safe orchestration;
  - the PySCF backend: SCF (RHF/RKS), SA-CASSCF, TDA, TDDFT, CIS, TDHF; energies, analytic gradients, S0->SI transition dipoles, oscillator strengths; validated against SHARC 4.1 and finite differences.
- **Not implemented yet:** every quantum-chemistry *stage* (optimize, frequencies, sampling, excitations, selection, dynamics, analysis) and its input section; use of NACs/overlaps in dynamics (P7); the Hessian (P4); triplets and open-shell; density fitting; TDDFT NACs.

## Known limitations
1. **Only `prepare` exists as a stage.** The electronic-structure layer is reachable only through `single-point`, so per-stage resources and `pause_after` have no visible effect (the last stage never pauses).
2. **The project-folder check is a path-text check.** An absolute path that is a symlink into a Windows drive or a OneDrive folder is not detected (relative paths are resolved, so they are).
3. **Electronic structure:** closed-shell singlets only; no Hessian (P4); TDA/TDDFT have no analytic NACs and their overlaps use X only; overlap cost grows with the number of determinants kept; overlap-based couplings have noise of order 1e-7/dt at default CASSCF convergence; `single-point` does not run the preflight and does not request NACs/overlaps.
4. **Deferred by ruling (P3c plan, Scope rulings):**
   - to **P4**: runs refused by a stale stage or preflight still writing state, and optional-stage handling; the validator reports only the first violation; the PySCF/geomeTRIC version and writable-scratch preflight checks;
   - to **P7b**: the worker orphan window between `Popen` and `on_start`;
   - to **release**: the build stamp.
   The P1-P3b ledgers under `.superpowers/sdd/` (git-ignored, local) hold the rest.

## Remaining work (master plan §9)
**P4 is next.**

| ID | Sub-project | Notes |
|---|---|---|
| P3a | PySCF electronic-structure layer | **DONE**, merged into local `main`; see Status |
| P3b | Couplings for dynamics | **DONE**, merged into local `main`; see Status |
| P3c | Hardening sweep | **DONE** on `feat/p3c-hardening` (not merged); see Status |
| P4 | `optimize` + `frequencies` stages | **Next.** Also takes the items moved by the P3c rulings (limitation 4). geomeTRIC; **PySCF Hessian (moved here from P3)**; own harmonic analysis; Molden output; stages declare `required_modules=("pyscf",)`. P4 and P6 must call `reset()` or use a fresh backend for independent geometries (a backend keeps the previous geometry) |
| P5 | `sampling` stage | Own Wigner sampler; compare with SHARC `wigner.py` |
| P6 | `excitations` + `selection` stages | Spectrum and initial-state selection |
| P7a/b | Dynamics core (models), then PySCF dynamics stage | Strategy §4; 7b needs P3b. P7b: treat `overlap_min_singular_value < 0.5` and a CASSCF rung change as step-rejection signals; consider rank-2 updates for the TDA single excitations if the overlap cost matters |
| P8 | `analysis` stage + end-to-end test | Populations and kinetics |
| P9 | SHARC validation campaign | Strategy §5 |

**Process for each sub-project:**
1. Write the detailed plan with superpowers:writing-plans in `docs/superpowers/plans/`.
2. Execute it with superpowers:subagent-driven-development. The ledger goes in `.superpowers/sdd/<plan>/progress.md`; trust the ledger and `git log` after context loss.
3. Run the final review.
4. Finish with superpowers:finishing-a-development-branch.
